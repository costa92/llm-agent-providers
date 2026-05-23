package compat

import "time"

// DefaultTimeout returns d when d > 0, otherwise the canonical 60-second
// default applied by all OpenAI-, Anthropic-, DeepSeek-, MiniMax- and
// Ollama-family providers.
func DefaultTimeout(d time.Duration) time.Duration {
	if d == 0 {
		return 60 * time.Second
	}
	return d
}
