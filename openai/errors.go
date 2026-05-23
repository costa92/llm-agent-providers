// Package openai's wrapErr delegates to internal/compat now that all
// OpenAI-SDK-shaped error mapping is shared with deepseek (which
// piggybacks on github.com/openai/openai-go/v3).
package openai

import (
	"github.com/costa92/llm-agent-providers/internal/compat"
)

func wrapErr(err error) error {
	return compat.WrapOpenAIError("openai", err)
}
