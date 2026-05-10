// Package anthropic implements the Phase-1 Generate-only adapter for
// github.com/anthropics/anthropic-sdk-go.
//
// The adapter satisfies llm.ChatModel and intentionally reports all
// optional capabilities as false in Phase 1. System prompts are lifted
// to the top-level `system` field required by Anthropic's Messages API.
package anthropic
