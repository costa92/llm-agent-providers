// Package ollama implements an Ollama adapter over
// github.com/ollama/ollama/api.
//
// The adapter satisfies llm.ChatModel, llm.ToolCaller, and llm.Embedder.
// Tool calling is model-aware: for native-tools models (e.g. llama3.1
// family) the SDK's structured Message.ToolCalls are honored, and a
// fallback content-parser handles models that emit tool calls inside the
// reply text (python_tag and qwen JSON/XML strategies).
//
// Capabilities reported via Info() are per-(provider × model) (Keystone
// K2), with the per-model tool/embed support table computed from the
// model name at construction time. Streaming events follow the typed K1
// union; for content-parsed strategies, text is buffered until Done so
// tool markers can be stripped before emission.
package ollama
