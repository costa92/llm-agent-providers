// Package openai implements an OpenAI adapter over
// github.com/openai/openai-go/v3.
//
// The adapter satisfies llm.ChatModel, llm.ToolCaller, llm.Embedder, and
// llm.ImageGenerator. Capabilities reported via Info() are per-(provider ×
// model): the constructor binds a model, and Info() reflects what that model
// can do (Keystone K2) — Embed and GenerateImage return
// llm.ErrCapabilityNotSupported when the bound model lacks the capability.
// Streaming events follow the typed K1 union with stable per-tool-call Index
// across deltas.
package openai
