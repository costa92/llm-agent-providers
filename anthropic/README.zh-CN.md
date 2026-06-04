# anthropic

[English](./README.md) | [简体中文](./README.zh-CN.md)

面向 `github.com/costa92/llm-agent/llm` 的 Phase 1 Anthropic 适配器。

当前范围：

- `llm.ChatModel.Generate`
- 带模型绑定提供方元数据的 `Info()`
- 带原生事件映射的 `Stream()`
- 原生工具调用

有意保留的能力缺口：

- 未实现嵌入（embeddings）
- `Info().Capabilities.Embeddings` 始终为 `false`
- 调用方应基于能力检测进行分支，和/或使用 `errors.Is(err, llm.ErrCapabilityNotSupported)`

Anthropic 特有的 Phase 1 行为：

- `Request.SystemPrompt` 被提升到顶层 `system`
- `role=system` 的消息同样被提升并从 `messages` 中移除
- `529 overloaded_error` 映射为 `*llm.RateLimitError`

最小示例：

```go
model, err := anthropic.New(
	anthropic.WithModel("claude-3-5-haiku-20241022"),
	anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
)
if err != nil {
	log.Fatal(err)
}

resp, err := model.Generate(ctx, llm.Request{
	Messages: []llm.Message{{Role: "user", Content: "Say hello"}},
})
```
