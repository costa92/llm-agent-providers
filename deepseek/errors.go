// Package deepseek's wrapErr delegates to internal/compat now that all
// OpenAI-SDK-shaped error mapping is shared with openai. DeepSeek
// piggybacks on github.com/openai/openai-go/v3 so the mapping table
// is byte-for-byte identical except for the provider-name string.
package deepseek

import (
	"github.com/costa92/llm-agent-providers/internal/compat"
)

func wrapErr(err error) error {
	return compat.WrapOpenAIError("deepseek", err)
}
