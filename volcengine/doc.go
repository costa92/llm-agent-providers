// Package volcengine implements a 火山方舟 (Volcengine Ark / 豆包) adapter
// over the official github.com/volcengine/volcengine-go-sdk arkruntime
// client.
//
// The adapter satisfies llm.ChatModel, llm.ToolCaller, llm.ImageGenerator,
// and llm.Embedder. Capabilities reported via Info() are per-(provider ×
// model): the constructor binds a model, and Info() reflects what that
// model can do (Keystone K2) — chat/tools for doubao chat models, image
// generation for doubao-seedream*, embeddings for doubao-embedding*.
// Streaming events follow the typed K1 union with a stable per-tool-call
// Index across fragmented deltas.
package volcengine
