// Package anthropic's wrapErr delegates to internal/compat now that all
// Anthropic-SDK-shaped error mapping is shared with minimax (which
// piggybacks on github.com/anthropics/anthropic-sdk-go).
//
// Q2 path A: anthropic-sdk-go publicly re-exports type Error, so
// status extraction inside compat.WrapAnthropicError uses
// errors.As(err, *anthropic.Error) directly.
package anthropic

import (
	"github.com/costa92/llm-agent-providers/internal/compat"
)

func wrapErr(err error) error {
	return compat.WrapAnthropicError("anthropic", err)
}
