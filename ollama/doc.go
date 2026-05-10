// Package ollama implements the Phase-1 Generate-only adapter for
// github.com/ollama/ollama/api.
//
// The adapter satisfies llm.ChatModel and intentionally reports all
// optional capabilities as false in Phase 1. Requests force
// `stream=false` so the callback-based SDK returns one final response.
package ollama
