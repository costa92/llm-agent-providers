package google

import (
	"context"
	"fmt"
	"strings"

	"github.com/costa92/llm-agent-contract/llm"
	"google.golang.org/genai"
)

// GenerateImage produces images from a text prompt. Routing is by bound model
// id: imagen* uses the Imagen predict path (GenerateImages); a Gemini-native
// image model (gemini-*-image) uses GenerateContent with TEXT+IMAGE response
// modalities. Output is always inline bytes (Gemini never returns a URL).
func (g *Google) GenerateImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	if !g.info.Capabilities.ImageGeneration {
		return llm.ImageResponse{}, fmt.Errorf("google: image generation: %w", llm.ErrCapabilityNotSupported)
	}
	ctx, cancel := g.withTimeout(ctx)
	defer cancel()
	if strings.HasPrefix(g.info.Model, "imagen") {
		return g.generateImagen(ctx, req)
	}
	return g.generateGeminiImage(ctx, req)
}

// generateImagen calls the Imagen predict endpoint.
func (g *Google) generateImagen(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	cfg := &genai.GenerateImagesConfig{}
	if req.N > 0 {
		cfg.NumberOfImages = int32(req.N)
	}
	if ar, ok := req.Extra["aspect_ratio"].(string); ok && ar != "" {
		cfg.AspectRatio = ar
	}
	resp, err := g.client.Models.GenerateImages(ctx, g.info.Model, req.Prompt, cfg)
	if err != nil {
		return llm.ImageResponse{}, wrapErr(err)
	}
	images := make([]llm.GeneratedImage, 0, len(resp.GeneratedImages))
	for _, gi := range resp.GeneratedImages {
		if gi == nil || gi.Image == nil {
			continue
		}
		images = append(images, llm.GeneratedImage{
			Bytes:    gi.Image.ImageBytes,
			MimeType: gi.Image.MIMEType,
		})
	}
	return llm.ImageResponse{
		Images:   images,
		Provider: "google",
		Model:    g.info.Model,
		Usage:    llm.Usage{Source: llm.UsageUnknown},
	}, nil
}

// generateGeminiImage calls GenerateContent with TEXT+IMAGE modalities and
// extracts inline image parts (text parts are dropped). ResponseModalities
// MUST include TEXT — image-only is rejected for Gemini 2.5 Flash Image.
func (g *Google) generateGeminiImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	cfg := &genai.GenerateContentConfig{
		ResponseModalities: []string{"TEXT", "IMAGE"},
	}
	contents := []*genai.Content{{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{{Text: req.Prompt}},
	}}
	resp, err := g.client.Models.GenerateContent(ctx, g.info.Model, contents, cfg)
	if err != nil {
		return llm.ImageResponse{}, wrapErr(err)
	}
	if blocked := blockedPromptErr(resp); blocked != nil {
		return llm.ImageResponse{}, blocked
	}
	var images []llm.GeneratedImage
	if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			if part.InlineData != nil && len(part.InlineData.Data) > 0 {
				images = append(images, llm.GeneratedImage{
					Bytes:    part.InlineData.Data,
					MimeType: part.InlineData.MIMEType,
				})
			}
		}
	}
	return llm.ImageResponse{
		Images:   images,
		Provider: "google",
		Model:    g.info.Model,
		Usage:    llm.Usage{Source: llm.UsageUnknown},
	}, nil
}
