// Package kimi's wrapErr delegates to internal/compat now that all
// OpenAI-SDK-shaped error mapping is shared with openai. Kimi
// piggybacks on github.com/openai/openai-go/v3 so the mapping table
// is byte-for-byte identical except for the provider-name string.
package kimi

import (
	"github.com/costa92/llm-agent-providers/internal/compat"
)

func wrapErr(err error) error {
	return compat.WrapOpenAIError("kimi", err)
}
