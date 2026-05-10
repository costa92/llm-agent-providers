// Package openai implements the Phase-1 Generate-only adapter for
// github.com/openai/openai-go/v3.
//
// The adapter satisfies llm.ChatModel and intentionally reports all
// optional capabilities as false in Phase 1:
//   - tools
//   - embeddings
//   - structured outputs
//
// Streaming is deferred to Phase 2.
package openai
