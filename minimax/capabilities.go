package minimax

import "github.com/costa92/llm-agent-contract/llm"

// capabilitiesForModel returns the static capability set for a MiniMax model.
// Per ecosystem K2 (capabilities are per provider×model), New() must consult
// this switch instead of hard-coding caps in options.go.
//
// Per D1 (P1-8 planning decision, 2026-05-22), only "MiniMax-M1" is
// explicitly enumerated — other model strings (abab*, MiniMax-Vision, future
// names) fall through to the default. Adding a new model that needs
// different caps should add an explicit case here rather than expanding
// the default.
func capabilitiesForModel(model string) llm.Capabilities {
	switch model {
	case "MiniMax-M1":
		return llm.Capabilities{Tools: true}
	case "image-01":
		return llm.Capabilities{ImageGeneration: true}
	default:
		return llm.Capabilities{Tools: true} // fallback: assume tools (behavior-preserving with prior hardcoded default)
	}
}
