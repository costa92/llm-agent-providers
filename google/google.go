package google

import (
	"google.golang.org/genai"

	"github.com/costa92/llm-agent-contract/llm"
)

var (
	// Interface asserts are enabled as each interface's methods land:
	// _ llm.ChatModel      = (*Google)(nil) // Task 2 (Generate/Stream/Info)
	// _ llm.ToolCaller     = (*Google)(nil) // Task 2 (+ WithTools)
	// _ llm.Embedder       = (*Google)(nil) // Task 5 (+ Embed)
	// _ llm.ImageGenerator = (*Google)(nil) // Task 6 (+ GenerateImage)
	_ = (*Google)(nil)
)

// Google is a Gemini provider bound to one model. Safe for concurrent use.
type Google struct {
	client     *genai.Client
	info       llm.ProviderInfo
	tools      []llm.Tool
	taskType   string
	dimensions int
}

// Info returns the bound (provider × model) identity and capabilities.
func (g *Google) Info() llm.ProviderInfo { return g.info }

// EmbedDimensions returns the embedding width for the bound model, honoring
// WithDimensions; 0 when the bound model is not an embedding model.
func (g *Google) EmbedDimensions() int {
	if !g.info.Capabilities.Embeddings {
		return 0
	}
	if g.dimensions > 0 {
		return g.dimensions
	}
	switch g.info.Model {
	case "gemini-embedding-001":
		return 3072
	case "text-embedding-004":
		return 768
	default:
		return 0
	}
}
