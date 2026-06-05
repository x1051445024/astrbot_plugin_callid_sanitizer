# astrbot_plugin_callid_sanitizer

在发送 LLM 请求前，截断过长的 `tool_call_id` / `call_id`，避免 API 400。

## 处理范围

- `req.contexts`
- `req.tool_calls_result`
- assistant tool_calls / tool 消息
- nested content dict / Pydantic-like message objects

## 原则

- 仅改写 **超长** 的 ID，不改动正常的 ID。
- 对同一个原始 ID 始终映射为相同的短 ID，避免请求内部不一致。
- 截断形式为 `tc_{前缀_}hash`，默认上限 64 字符。
- 不写线程锁、不做对话数据库清洗、不碰 LLM 响应链。
