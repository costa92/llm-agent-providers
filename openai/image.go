package openai

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/costa92/llm-agent-contract/llm"
	openai "github.com/openai/openai-go/v3"
)

var _ llm.ImageGenerator = (*OpenAI)(nil)

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

// GenerateImage performs text-to-image generation via the OpenAI Images API.
// Gated by the bound model: non-image models return
// llm.ErrCapabilityNotSupported (mirrors Embed gating). When the provider
// returns base64 (GPT image models, or dall-e with b64_json), Bytes is
// populated; when it returns a hosted URL (dall-e default), URL is populated.
func (o *OpenAI) GenerateImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	if !isImageModel(o.info.Model) {
		return llm.ImageResponse{}, fmt.Errorf("openai: image generation: %w", llm.ErrCapabilityNotSupported)
	}

	resp, err := o.client.Images.Generate(ctx, o.toImageParams(req))
	if err != nil {
		return llm.ImageResponse{}, wrapErr(err)
	}
	return o.fromImageResponse(resp), nil
}

func (o *OpenAI) toImageParams(req llm.ImageRequest) openai.ImageGenerateParams {
	p := openai.ImageGenerateParams{
		Prompt: req.Prompt,
		Model:  openai.ImageModel(o.info.Model),
	}
	if req.N > 0 {
		p.N = openai.Int(int64(req.N))
	}
	if req.Size != "" {
		p.Size = openai.ImageGenerateParamsSize(req.Size)
	}
	if req.Quality != "" {
		p.Quality = openai.ImageGenerateParamsQuality(req.Quality)
	}
	if req.Format != "" {
		p.OutputFormat = openai.ImageGenerateParamsOutputFormat(req.Format)
	}
	if v, ok := req.Extra["style"].(string); ok && v != "" {
		p.Style = openai.ImageGenerateParamsStyle(v)
	}
	if v, ok := req.Extra["background"].(string); ok && v != "" {
		p.Background = openai.ImageGenerateParamsBackground(v)
	}
	if v, ok := req.Extra["moderation"].(string); ok && v != "" {
		p.Moderation = openai.ImageGenerateParamsModeration(v)
	}
	return p
}

func (o *OpenAI) fromImageResponse(resp *openai.ImagesResponse) llm.ImageResponse {
	images := make([]llm.GeneratedImage, 0, len(resp.Data))
	for _, img := range resp.Data {
		gen := llm.GeneratedImage{RevisedPrompt: img.RevisedPrompt}
		if img.B64JSON != "" {
			if decoded, err := base64.StdEncoding.DecodeString(img.B64JSON); err == nil {
				gen.Bytes = decoded
			}
		}
		if len(gen.Bytes) == 0 && img.URL != "" {
			gen.URL = img.URL
		}
		images = append(images, gen)
	}
	out := llm.ImageResponse{
		Images:   images,
		Provider: "openai",
		Model:    o.info.Model,
	}
	if resp.Usage.TotalTokens != 0 || resp.Usage.InputTokens != 0 {
		out.Usage = llm.Usage{
			InputTokens:  int(resp.Usage.InputTokens),
			OutputTokens: int(resp.Usage.OutputTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
			Source:       llm.UsageReported,
		}
	}
	return out
}
