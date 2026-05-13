# deepseek

DeepSeek adapter for `github.com/costa92/llm-agent/llm`.

Current scope:

- `llm.ChatModel.Generate`
- `llm.ChatModel.Stream`
- `llm.ToolCaller.WithTools`
- `Info()` with model-bound provider metadata

Deliberate capability gap:

- embeddings are not implemented
- `Info().Capabilities.Embeddings` is always `false`

Regional routing:

- `WithRegion(RegionCN|RegionGlobal)` selects a preset endpoint
- `WithBaseURL(...)` overrides region presets

Minimal example:

```go
model, err := deepseek.New(
	deepseek.WithModel("deepseek-chat"),
	deepseek.WithAPIKey(os.Getenv("DEEPSEEK_API_KEY")),
	deepseek.WithRegion(deepseek.RegionGlobal),
)
if err != nil {
	log.Fatal(err)
}

resp, err := model.Generate(ctx, llm.Request{
	Messages: []llm.Message{{Role: "user", Content: "Say hello"}},
})
```
