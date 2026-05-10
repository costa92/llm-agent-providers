# anthropic

Phase 1 Anthropic adapter for `github.com/costa92/llm-agent/llm`.

Current scope:

- `llm.ChatModel.Generate`
- `Info()` with model-bound provider metadata
- `Stream()` Phase 1 stub

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
