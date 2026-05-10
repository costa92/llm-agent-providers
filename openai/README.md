# openai

Phase 1 OpenAI adapter for `github.com/costa92/llm-agent/llm`.

Current scope:

- `llm.ChatModel.Generate`
- `Info()` with model-bound provider metadata
- `Stream()` Phase 1 stub

Not in scope yet:

- tool calling
- embeddings
- structured outputs
- streaming

Minimal example:

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
