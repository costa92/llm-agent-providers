# anthropic

[English](./README.md) | [简体中文](./README.zh-CN.md)

Phase 1 Anthropic adapter for `github.com/costa92/llm-agent/llm`.

Current scope:

- `llm.ChatModel.Generate`
- `Info()` with model-bound provider metadata
- `Stream()` with native event mapping
- native tool calling

Deliberate capability gap:

- embeddings are not implemented
- `Info().Capabilities.Embeddings` is always `false`
- callers should branch on capability detection and/or `errors.Is(err, llm.ErrCapabilityNotSupported)`

Anthropic-specific Phase 1 behavior:

- `Request.SystemPrompt` is lifted to top-level `system`
- `role=system` messages are also lifted and removed from `messages`
- `529 overloaded_error` maps to `*llm.RateLimitError`

Minimal example:

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
