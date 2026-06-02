package deepseek

import "github.com/costa92/llm-agent-contract/llm"

// capabilitiesForModel returns the static capability set for a DeepSeek model.
// Per ecosystem K2 (capabilities are per provider×model), New() must consult
// this switch instead of hard-coding caps in options.go.
//
// All currently known DeepSeek models support tools; the switch exists so that
// adding a new model (e.g. a reasoning-only or vision variant) forces an
// explicit decision rather than silently inheriting the default.
func capabilitiesForModel(model string) llm.Capabilities {
	switch model {
	case "deepseek-chat", "deepseek-coder", "deepseek-reasoner":
		return llm.Capabilities{Tools: true}
	default:
		return llm.Capabilities{Tools: true} // fallback: assume tools (behavior-preserving with prior hardcoded default)
	}
}
