package minimax

import (
	"context"
	"fmt"
	"net/url"

	"github.com/costa92/llm-agent-contract/llm"
)

type embedRequestBody struct {
	Model string   `json:"model"`
	Texts []string `json:"texts"`
	Type  string   `json:"type"`
}

// embedResponseBody parses MiniMax's embeddings response. vectors and
// total_tokens are TOP-LEVEL fields (not nested under data/usage).
type embedResponseBody struct {
	Vectors     [][]float32 `json:"vectors"`
	TotalTokens int         `json:"total_tokens"`
	BaseResp    baseResp    `json:"base_resp"`
}

// Embed returns vectors in input order via the raw-HTTP /v1/embeddings
// endpoint. GroupId is passed as a query parameter; the "type" field is the
// configured embedding type ("db" default / "query"). Gated by the bound
// model: non-embed models return llm.ErrCapabilityNotSupported.
func (m *MiniMax) Embed(ctx context.Context, texts []string) ([]llm.Vector, llm.Usage, error) {
	if !m.info.Capabilities.Embeddings {
		return nil, llm.Usage{}, fmt.Errorf("minimax: embeddings: %w", llm.ErrCapabilityNotSupported)
	}
	if len(texts) == 0 {
		return []llm.Vector{}, llm.Usage{Source: llm.UsageReported}, nil
	}

	body := embedRequestBody{
		Model: m.info.Model,
		Texts: append([]string(nil), texts...),
		Type:  m.embeddingType,
	}
	rawQuery := url.Values{"GroupId": {m.groupID}}.Encode()

	var out embedResponseBody
	if err := m.postJSON(ctx, "/v1/embeddings", rawQuery, body, &out); err != nil {
		return nil, llm.Usage{}, err
	}
	if err := baseRespError("minimax: embeddings", out.BaseResp); err != nil {
		return nil, llm.Usage{}, err
	}

	vectors := make([]llm.Vector, 0, len(out.Vectors))
	for _, v := range out.Vectors {
		vectors = append(vectors, append(llm.Vector(nil), v...))
	}
	usage := llm.Usage{
		InputTokens: out.TotalTokens,
		TotalTokens: out.TotalTokens,
		Source:      llm.UsageReported,
	}
	return vectors, usage, nil
}

// EmbedDimensions returns the fixed embedding dimensionality for the bound
// model, or 0 when the model has no embedding capability.
func (m *MiniMax) EmbedDimensions() int {
	switch m.info.Model {
	case "embo-01":
		return 1536
	default:
		return 0
	}
}
