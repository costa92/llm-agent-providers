# ollama

[English](./README.md) | [简体中文](./README.zh-CN.md)

Phase 1 Ollama adapter for `github.com/costa92/llm-agent/llm`.

Current scope:

- `llm.ChatModel.Generate`
- `Info()` with model-bound provider metadata
- `Stream()` Phase 1 stub

Ollama-specific Phase 1 behavior:

- keyless adapter, defaults to `OLLAMA_HOST` or `http://localhost:11434`
- forces `stream=false` even though Ollama streams by default
- 404 model-not-pulled maps to `*llm.InvalidRequestError`

Minimal example:

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
