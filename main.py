"""AstrBot plugin: sanitize overlong tool call IDs before provider requests.

This plugin is intentionally conservative: it only rewrites IDs that are longer
than the configured limit and keeps the mapping consistent inside the same
ProviderRequest.
"""

from __future__ import annotations

import hashlib
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

MAX_ID_LEN = 64
HASH_LEN = 16
PREFIX = "tc_"


@register(
    "astrbot_plugin_callid_sanitizer",
    "Kurisu",
    "Sanitize overlong tool_call_id / call_id before LLM requests.",
    "1.2.0",
)
class CallIdSanitizerPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.max_id_len = MAX_ID_LEN

    @filter.on_llm_request()
    async def sanitize_call_ids(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Rewrite overlong tool call IDs in-place before provider serialization."""
        id_map: dict[str, str] = {}
        changed = 0

        changed += self._sanitize_messages(req.contexts, id_map)
        changed += self._sanitize_tool_calls_result(req.tool_calls_result, id_map)
        repaired = self._repair_message_tool_call_pairs(req.contexts)

        if changed or repaired:
            logger.info(
                "[callid_sanitizer] sanitized %s overlong tool call id reference(s), removed_orphan_outputs=%s, map_size=%s",
                changed,
                repaired,
                len(id_map),
            )

    def _shorten_id(self, old_id: Any, id_map: dict[str, str]) -> str:
        old = str(old_id)
        if len(old) <= self.max_id_len:
            return old
        if old in id_map:
            return id_map[old]

        digest = hashlib.sha256(old.encode("utf-8", errors="ignore")).hexdigest()[:HASH_LEN]
        safe_head_len = max(0, self.max_id_len - len(PREFIX) - len(digest) - 1)
        safe_head = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in old[:safe_head_len]
        ).strip("_-")
        if safe_head:
            new_id = f"{PREFIX}{safe_head}_{digest}"
        else:
            new_id = f"{PREFIX}{digest}"

        # Defensive guard in case constants are changed later.
        new_id = new_id[: self.max_id_len]
        id_map[old] = new_id
        return new_id

    def _sanitize_value(self, value: Any, id_map: dict[str, str]) -> tuple[Any, int]:
        if value is None:
            return value, 0
        if isinstance(value, str):
            new_value = self._shorten_id(value, id_map)
            return new_value, int(new_value != value)
        return value, 0

    def _sanitize_messages(self, messages: Any, id_map: dict[str, str]) -> int:
        """Sanitize OpenAI-like messages or AstrBot Message-like objects."""
        if not messages:
            return 0
        changed = 0
        for msg in list(messages):
            changed += self._sanitize_message(msg, id_map)
        return changed

    def _sanitize_message(self, msg: Any, id_map: dict[str, str]) -> int:
        changed = 0

        if isinstance(msg, dict):
            for key in ("tool_call_id", "call_id"):
                if key in msg:
                    msg[key], delta = self._sanitize_value(msg.get(key), id_map)
                    changed += delta

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    changed += self._sanitize_tool_call(tool_call, id_map)

            content = msg.get("content")
            changed += self._sanitize_nested_content(content, id_map)
            return changed

        # Pydantic/dataclass Message-like objects used inside AstrBot.
        for attr in ("tool_call_id", "call_id"):
            if hasattr(msg, attr):
                old = getattr(msg, attr, None)
                new, delta = self._sanitize_value(old, id_map)
                if delta:
                    setattr(msg, attr, new)
                    changed += delta

        tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                changed += self._sanitize_tool_call(tool_call, id_map)

        content = getattr(msg, "content", None)
        changed += self._sanitize_nested_content(content, id_map)
        return changed

    def _sanitize_tool_call(self, tool_call: Any, id_map: dict[str, str]) -> int:
        changed = 0
        if isinstance(tool_call, dict):
            if "id" in tool_call:
                tool_call["id"], delta = self._sanitize_value(tool_call.get("id"), id_map)
                changed += delta
            if "tool_call_id" in tool_call:
                tool_call["tool_call_id"], delta = self._sanitize_value(
                    tool_call.get("tool_call_id"), id_map
                )
                changed += delta
            return changed

        for attr in ("id", "tool_call_id", "call_id"):
            if hasattr(tool_call, attr):
                old = getattr(tool_call, attr, None)
                new, delta = self._sanitize_value(old, id_map)
                if delta:
                    setattr(tool_call, attr, new)
                    changed += delta
        return changed

    def _sanitize_nested_content(self, content: Any, id_map: dict[str, str]) -> int:
        """Best-effort recursion for provider-specific content parts."""
        changed = 0
        if isinstance(content, list):
            for item in content:
                changed += self._sanitize_nested_content(item, id_map)
        elif isinstance(content, dict):
            content_type = content.get("type")
            if content_type in {"function_call", "function_call_output", "tool_use", "tool_result"}:
                keys = ("tool_call_id", "call_id", "id", "tool_use_id")
            else:
                keys = ("tool_call_id", "call_id", "tool_use_id")
            for key in keys:
                if key in content:
                    content[key], delta = self._sanitize_value(content.get(key), id_map)
                    changed += delta
            for value in content.values():
                if isinstance(value, (dict, list)):
                    changed += self._sanitize_nested_content(value, id_map)
        return changed

    def _repair_message_tool_call_pairs(self, messages: Any) -> int:
        """Repair tool-call history in both directions before provider requests."""
        if not isinstance(messages, list):
            return 0

        repaired = 0
        repaired += self._drop_unanswered_assistant_tool_calls(messages)

        valid_ids = self._collect_assistant_call_ids(messages)
        if not valid_ids:
            original_len = len(messages)
            messages[:] = [msg for msg in messages if not self._is_tool_output_message(msg)]
            return repaired + (original_len - len(messages))

        kept_messages = []
        for msg in messages:
            if self._is_tool_output_message(msg):
                result_ids = self._tool_result_ids(msg)
                if result_ids and not (result_ids & valid_ids):
                    repaired += 1
                    continue
            kept_messages.append(msg)
        messages[:] = kept_messages
        return repaired

    def _drop_unanswered_assistant_tool_calls(self, messages: list[Any]) -> int:
        dropped = 0
        for index, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            answered_ids = self._following_tool_result_ids(messages, index)
            dropped += self._filter_assistant_tool_calls(msg, answered_ids)
        return dropped

    def _following_tool_result_ids(self, messages: list[Any], index: int) -> set[str]:
        ids: set[str] = set()
        cursor = index + 1
        while cursor < len(messages):
            next_msg = messages[cursor]
            if not self._is_tool_output_message(next_msg):
                break
            ids.update(self._tool_result_ids(next_msg))
            cursor += 1
        return ids

    def _filter_assistant_tool_calls(self, msg: dict[str, Any], answered_ids: set[str]) -> int:
        dropped = 0
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            kept = []
            for tool_call in tool_calls:
                call_id = None
                if isinstance(tool_call, dict):
                    call_id = tool_call.get("id") or tool_call.get("call_id")
                else:
                    call_id = getattr(tool_call, "id", None) or getattr(tool_call, "call_id", None)
                if call_id and str(call_id) in answered_ids:
                    kept.append(tool_call)
                else:
                    dropped += 1
            if kept:
                msg["tool_calls"] = kept
            elif tool_calls:
                msg.pop("tool_calls", None)
                if msg.get("content") is None:
                    msg["content"] = ""

        content = msg.get("content")
        if isinstance(content, list):
            filtered = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"function_call", "tool_use"}:
                    part_ids = set()
                    for key in ("call_id", "id", "tool_call_id"):
                        value = part.get(key)
                        if value:
                            part_ids.add(str(value))
                    if part_ids and not (part_ids & answered_ids):
                        dropped += 1
                        continue
                filtered.append(part)
            msg["content"] = filtered
        return dropped

    def _collect_assistant_call_ids(self, messages: list[Any]) -> set[str]:
        ids: set[str] = set()
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        call_id = tool_call.get("id") or tool_call.get("call_id")
                        if call_id:
                            ids.add(str(call_id))
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"function_call", "tool_use"}:
                        for key in ("call_id", "id", "tool_call_id"):
                            value = part.get(key)
                            if value:
                                ids.add(str(value))
        return ids

    def _tool_result_ids(self, msg: Any) -> set[str]:
        ids: set[str] = set()
        if not isinstance(msg, dict):
            return ids
        for key in ("tool_call_id", "call_id", "tool_use_id"):
            value = msg.get(key)
            if value:
                ids.add(str(value))
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"function_call_output", "tool_result"}:
                    for key in ("call_id", "tool_call_id", "tool_use_id"):
                        value = part.get(key)
                        if value:
                            ids.add(str(value))
        return ids

    def _is_tool_output_message(self, msg: Any) -> bool:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") == "tool":
            return True
        if msg.get("type") in {"function_call_output", "tool_result"}:
            return True
        content = msg.get("content")
        return isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in {"function_call_output", "tool_result"}
            for part in content
        )

    def _sanitize_tool_calls_result(self, tool_calls_result: Any, id_map: dict[str, str]) -> int:
        if not tool_calls_result:
            return 0
        changed = 0
        items = tool_calls_result if isinstance(tool_calls_result, list) else [tool_calls_result]
        for item in items:
            changed += self._sanitize_single_tool_calls_result(item, id_map)
        return changed

    def _sanitize_single_tool_calls_result(self, item: Any, id_map: dict[str, str]) -> int:
        changed = 0
        if isinstance(item, dict):
            changed += self._sanitize_message(item.get("tool_calls_info"), id_map)
            changed += self._sanitize_messages(item.get("tool_calls_result"), id_map)
            return changed

        tool_calls_info = getattr(item, "tool_calls_info", None)
        tool_calls_result = getattr(item, "tool_calls_result", None)
        changed += self._sanitize_message(tool_calls_info, id_map)
        changed += self._sanitize_messages(tool_calls_result, id_map)
        return changed
