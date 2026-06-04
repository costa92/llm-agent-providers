package openai

// isImageModel reports whether the bound model is an OpenAI image model.
// Mirrors the embedding-model gating in options.go (K2: capabilities are
// per provider×model). New image models must be added here explicitly.
func isImageModel(model string) bool {
	switch model {
	case "gpt-image-1", "gpt-image-1-mini", "gpt-image-2",
		"gpt-image-2-2026-04-21", "dall-e-2", "dall-e-3":
		return true
	default:
		return false
	}
}
