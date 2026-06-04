# ollama

[English](./README.md) | [简体中文](./README.zh-CN.md)

面向 `github.com/costa92/llm-agent/llm` 的 Phase 1 Ollama 适配器。

当前范围：

- `llm.ChatModel.Generate`
- 带模型绑定提供方元数据的 `Info()`
- `Stream()` Phase 1 桩实现（stub）

Ollama 特有的 Phase 1 行为：

- 无需密钥的适配器，默认使用 `OLLAMA_HOST` 或 `http://localhost:11434`
- 即使 Ollama 默认启用流式，也强制 `stream=false`
- 404 模型未拉取（model-not-pulled）映射为 `*llm.InvalidRequestError`

最小示例：

```go
model, err := ollama.New(
	ollama.WithModel("llama3.1:8b"),
)
if err != nil {
	log.Fatal(err)
}

resp, err := model.Generate(ctx, llm.Request{
	Messages: []llm.Message{{Role: "user", Content: "Say hello"}},
})
```
