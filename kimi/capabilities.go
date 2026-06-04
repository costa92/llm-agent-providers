package kimi

import "github.com/costa92/llm-agent-contract/llm"

// capabilitiesForModel returns the static capability set for a Kimi model.
// Per ecosystem K2 (capabilities are per provider×model), New() must consult
// this switch instead of hard-coding caps in options.go.
//
// All currently known Kimi (Moonshot AI) models support tools; the switch
// exists so that adding a new model (e.g. a vision or embedding variant)
// forces an explicit decision rather than silently inheriting the default.
// Moonshot offers neither image generation nor embeddings, so those caps
// stay false here and across the package.
func capabilitiesForModel(model string) llm.Capabilities {
	switch model {
	case "kimi-k2", "kimi-k2-0905", "kimi-k2-thinking", "kimi-k2.5", "kimi-k2.6",
		"moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k":
		return llm.Capabilities{Tools: true}
	default:
		return llm.Capabilities{Tools: true} // fallback: assume tools (behavior-preserving with prior hardcoded default)
	}
}
