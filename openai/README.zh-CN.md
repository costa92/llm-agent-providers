# openai

[English](./README.md) | [简体中文](./README.zh-CN.md)

面向 `github.com/costa92/llm-agent/llm` 的 Phase 1 OpenAI 适配器。

当前范围：

- `llm.ChatModel.Generate`
- 带模型绑定提供方元数据的 `Info()`
- `Stream()` Phase 1 桩实现（stub）

尚未纳入范围：

- 工具调用
- 嵌入（embeddings）
- 结构化输出（structured outputs）
- 流式

最小示例：

```go
model, err := openai.New(
	openai.WithModel("gpt-4o-mini"),
	openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
)
if err != nil {
	log.Fatal(err)
}

resp, err := model.Generate(ctx, llm.Request{
	Messages: []llm.Message{{Role: "user", Content: "Say hello"}},
})
```
