package google

import (
	"context"
	"fmt"

	"github.com/costa92/llm-agent-contract/llm"
	"google.golang.org/genai"
)

// Embed returns one vector per input text, in order. On the Gemini Developer
// API no token usage is reported, so Usage is zero with Source=UsageUnknown.
func (g *Google) Embed(ctx context.Context, texts []string) ([]llm.Vector, llm.Usage, error) {
	if !g.info.Capabilities.Embeddings {
		return nil, llm.Usage{}, fmt.Errorf("google: embeddings: %w", llm.ErrCapabilityNotSupported)
	}
	if len(texts) == 0 {
		return []llm.Vector{}, llm.Usage{Source: llm.UsageUnknown}, nil
	}

	ctx, cancel := g.withTimeout(ctx)
	defer cancel()

	contents := make([]*genai.Content, 0, len(texts))
	for _, txt := range texts {
		contents = append(contents, &genai.Content{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{{Text: txt}},
		})
	}

	var cfg *genai.EmbedContentConfig
	if g.taskType != "" || g.dimensions > 0 {
		cfg = &genai.EmbedContentConfig{}
		if g.taskType != "" {
			cfg.TaskType = g.taskType
		}
		if g.dimensions > 0 {
			cfg.OutputDimensionality = genai.Ptr(int32(g.dimensions))
		}
	}

	resp, err := g.client.Models.EmbedContent(ctx, g.info.Model, contents, cfg)
	if err != nil {
		return nil, llm.Usage{}, wrapErr(err)
	}

	vectors := make([]llm.Vector, 0, len(resp.Embeddings))
	for _, emb := range resp.Embeddings {
		vec := make(llm.Vector, len(emb.Values))
		copy(vec, emb.Values)
		vectors = append(vectors, vec)
	}
	return vectors, llm.Usage{Source: llm.UsageUnknown}, nil
}
