// Package anthropic implements an Anthropic adapter over
// github.com/anthropics/anthropic-sdk-go.
//
// The adapter satisfies llm.ChatModel and llm.ToolCaller. Embeddings are
// surfaced as an explicit gap via llm.ErrNotSupported (Anthropic has no
// embeddings API). System prompts are lifted to the top-level `system`
// field required by Anthropic's Messages API.
//
// Capabilities reported via Info() are per-(provider × model) (Keystone
// K2). Streaming events follow the typed K1 union with stable
// per-tool-call Index across deltas (input_json_delta partials).
package anthropic
