# 工具调用 ID 清理器

在 AstrBot 发送 LLM 请求前，清理工具调用链中的 `tool_call_id` / `call_id` / `tool_use_id`，避免 OpenAI 兼容接口因为 ID 过长、空字符串或工具调用配对异常返回 400。

本插件定位很窄：它只做请求发送前的 ID 卫生和工具输出配对清理，不压缩上下文、不清理数据库、不改写模型响应。

## 功能特性

- 截断过长的工具调用 ID，默认最大长度为 64 字符。
- 对同一个原始 ID 使用稳定映射，保证一次请求内 assistant tool_call 与 tool result 仍能配对。
- 过滤空字符串或纯空白的 `call_id` / `tool_call_id` / `tool_use_id`。
- 移除孤立 tool 输出，避免 `No tool call found`、`empty string` 等 400 错误。
- 同时处理 `req.contexts` 与 `req.tool_calls_result`。
- 兼容 OpenAI Chat Completions 风格、Responses API 风格以及 AstrBot 内部 message-like 对象。

## 适用场景

适合以下错误：

```text
Invalid 'input[x].call_id': empty string
No tool call found for function call output
messages with role 'tool' must be a response to a preceding message with 'tool_calls'
tool_call_id is too long
```

如果你使用本地 octopus、OpenAI 兼容 Provider、工具调用或 MCP 插件，本插件可以作为请求发送前的轻量兜底。

## 安装方式

将插件目录放入 AstrBot 插件目录：

```text
AstrBot/data/plugins/astrbot_plugin_callid_sanitizer
```

目录结构示例：

```text
astrbot_plugin_callid_sanitizer/
├── __init__.py
├── main.py
├── metadata.yaml
└── README.md
```

然后在 AstrBot WebUI 中重载插件，或重启 AstrBot。

## 工作机制

插件监听 AstrBot 的 `on_llm_request` 事件，在 ProviderRequest 序列化前原地修复请求：

1. 扫描 `req.contexts` 中的 assistant tool_calls、tool 消息和嵌套 content block。
2. 扫描 `req.tool_calls_result` 中的工具调用信息和工具输出。
3. 对超长 ID 生成 `tc_{前缀}_{hash}` 形式的短 ID。
4. 对空字符串或纯空白 ID 不生成伪 ID，而是移除对应的无效工具输出。
5. 删除没有匹配 assistant tool_call 的孤立 tool/function 输出。
6. 保留已有合法 ID，不改变正常工具调用链。

## 清理范围

插件会检查以下字段：

- 顶层消息字段：`tool_call_id`、`call_id`、`tool_use_id`
- assistant tool_calls：`id`、`tool_call_id`、`call_id`
- content block：`function_call`、`function_call_output`、`tool_use`、`tool_result`
- AstrBot 请求对象：`req.contexts`、`req.tool_calls_result`

## 设计原则

- 只截断超长 ID，不改动正常长度的合法 ID。
- 空 ID 不补假 ID，直接清理无效输出，避免伪造不存在的调用链。
- 同一次请求内保持稳定映射，避免 assistant 调用和 tool 输出不一致。
- 不写线程锁、不清理会话数据库、不拦截模型响应链。
- 尽量保守，避免影响正常工具调用。

## 与 length_error_handler 的关系

`astrbot_plugin_length_error_handler` 负责长上下文压缩、输出预算修正和 length 错误重试。

本插件负责工具调用 ID 卫生。两个插件可以同时启用：

- 先由本插件清理空/超长/孤立工具调用 ID。
- 再由长度错误处理器处理长上下文和 length 截断恢复。

如果仍然遇到 `call_id` 相关报错，建议清理当前会话上下文，因为旧历史中可能已经保存了坏的工具调用记录。

## 排错建议

### 更新后仍报 `empty string`

1. 确认插件版本至少为 `1.2.1`。
2. 重启 AstrBot。
3. 清空出错会话上下文或新建会话。
4. 确认没有其他插件在本插件之后重新写入空 `call_id`。

### 工具调用结果丢失

本插件会移除没有有效 ID 或没有匹配 assistant tool_call 的工具输出。若某些工具结果被移除，通常说明历史上下文已经不完整。建议降低上下文压缩强度，或配合最新版 `astrbot_plugin_length_error_handler`。

## 版本信息

- 插件标识：`astrbot_plugin_callid_sanitizer`
- 当前版本：`1.2.1`
- 作者：Kurisu
- 仓库：https://github.com/x1051445024/astrbot_plugin_callid_sanitizer
