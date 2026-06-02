package ollama

import (
	"fmt"
	"strings"

	"github.com/costa92/llm-agent-contract/llm"
)

func embeddingDimensionForModel(model string) int {
	name := strings.ToLower(model)
	switch {
	case strings.HasPrefix(name, "nomic-embed-text"):
		return 768
	case strings.HasPrefix(name, "all-minilm"):
		return 384
	default:
		return 0
	}
}

func supportsEmbeddings(model string) bool {
	return embeddingDimensionForModel(model) > 0
}

func unsupportedEmbeddingError(model string) error {
	return fmt.Errorf("ollama: model %s embeddings unavailable (ProviderInfo.Capabilities.Embeddings=false): %w", model, llm.ErrCapabilityNotSupported)
}
