package minimax

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/costa92/llm-agent-contract/llm"
)

// imageRequestBody is the MiniMax /v1/image_generation request payload.
type imageRequestBody struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	ResponseFormat string `json:"response_format"`
	AspectRatio    string `json:"aspect_ratio,omitempty"`
	Width          int    `json:"width,omitempty"`
	Height         int    `json:"height,omitempty"`
}

type imageResponseBody struct {
	Data struct {
		ImageURLs []string `json:"image_urls"`
	} `json:"data"`
	BaseResp baseResp `json:"base_resp"`
}

// GenerateImage performs text-to-image generation via the proprietary MiniMax
// /v1/image_generation endpoint (no SDK surface — raw HTTP). Gated by the bound
// model: non-image models return llm.ErrCapabilityNotSupported. MiniMax returns
// hosted URLs, so each GeneratedImage carries URL (not Bytes).
func (m *MiniMax) GenerateImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	if !m.info.Capabilities.ImageGeneration {
		return llm.ImageResponse{}, fmt.Errorf("minimax: image generation: %w", llm.ErrCapabilityNotSupported)
	}

	body := imageRequestBody{
		Model:          m.info.Model,
		Prompt:         req.Prompt,
		N:              req.N,
		ResponseFormat: "url",
	}
	if w, h, ok := parseSize(req.Size); ok {
		body.Width = w
		body.Height = h
	} else if ar, ok := req.Extra["aspect_ratio"].(string); ok && ar != "" {
		body.AspectRatio = ar
	}

	var out imageResponseBody
	if err := m.postJSON(ctx, "/v1/image_generation", "", body, &out); err != nil {
		return llm.ImageResponse{}, err
	}
	if err := baseRespError("minimax: image generation", out.BaseResp); err != nil {
		return llm.ImageResponse{}, err
	}

	images := make([]llm.GeneratedImage, 0, len(out.Data.ImageURLs))
	for _, u := range out.Data.ImageURLs {
		images = append(images, llm.GeneratedImage{URL: u})
	}
	return llm.ImageResponse{
		Images:   images,
		Provider: "minimax",
		Model:    m.info.Model,
	}, nil
}

// parseSize splits a "WxH" string into width/height ints. Returns ok=false for
// any string that is not exactly two positive integers separated by "x".
func parseSize(size string) (w, h int, ok bool) {
	parts := strings.SplitN(size, "x", 2)
	if len(parts) != 2 {
		return 0, 0, false
	}
	w, err := strconv.Atoi(parts[0])
	if err != nil || w <= 0 {
		return 0, 0, false
	}
	h, err = strconv.Atoi(parts[1])
	if err != nil || h <= 0 {
		return 0, 0, false
	}
	return w, h, true
}
