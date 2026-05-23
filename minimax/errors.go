// Package minimax's wrapErr delegates to internal/compat. MiniMax
// piggybacks on github.com/anthropics/anthropic-sdk-go, so its error
// mapping is shared with the anthropic package via
// compat.WrapAnthropicError — including the 529 ("Overloaded") ->
// RateLimitError special case.
package minimax

import (
	"github.com/costa92/llm-agent-providers/internal/compat"
)

func wrapErr(err error) error {
	return compat.WrapAnthropicError("minimax", err)
}
