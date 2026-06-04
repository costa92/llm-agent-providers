package volcengine

import (
	"context"
	"fmt"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// Embed returns one vector per input text, in input order. Only available on
// doubao-embedding* models (K2-gated). The bound dimensionality (from the
// model default or WithDimensions) is sent when non-zero.
func (v *Volcengine) Embed(ctx context.Context, texts []string) ([]llm.Vector, llm.Usage, error) {
	if !v.info.Capabilities.Embeddings {
		return nil, llm.Usage{}, fmt.Errorf("volcengine: embeddings: %w", llm.ErrCapabilityNotSupported)
	}
	if len(texts) == 0 {
		return []llm.Vector{}, llm.Usage{Source: llm.UsageReported}, nil
	}

	reqStrings := model.EmbeddingRequestStrings{
		Input: append([]string(nil), texts...),
		Model: v.info.Model,
	}
	if v.embedDimensions > 0 {
		reqStrings.Dimensions = v.embedDimensions
	}

	resp, err := v.client.CreateEmbeddings(ctx, reqStrings, v.requestOptions()...)
	if err != nil {
		return nil, llm.Usage{}, wrapErr(err)
	}

	vectors := make([]llm.Vector, 0, len(resp.Data))
	for _, item := range resp.Data {
		vec := make(llm.Vector, len(item.Embedding))
		copy(vec, item.Embedding)
		vectors = append(vectors, vec)
	}
	usage := llm.Usage{
		InputTokens: resp.Usage.PromptTokens,
		TotalTokens: resp.Usage.TotalTokens,
		Source:      llm.UsageReported,
	}
	return vectors, usage, nil
}
