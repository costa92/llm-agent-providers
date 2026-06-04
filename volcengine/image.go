package volcengine

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// GenerateImage runs a text-to-image generation. Only available on
// doubao-seedream* models (K2-gated); other models return
// ErrCapabilityNotSupported. Multi-image (N>1) is not supported (Ark uses
// SequentialImageGeneration, which is out of scope) — a single image request
// is sent and every returned image is mapped.
func (v *Volcengine) GenerateImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	if !v.info.Capabilities.ImageGeneration {
		return llm.ImageResponse{}, fmt.Errorf("volcengine: image generation: %w", llm.ErrCapabilityNotSupported)
	}

	sdkReq := model.GenerateImagesRequest{
		Model:  v.info.Model,
		Prompt: req.Prompt,
	}

	// Response format: default url; b64_json when caller asks for bytes.
	respFormat := model.GenerateImagesResponseFormatURL
	if req.Format == model.GenerateImagesResponseFormatBase64 || req.Format == "bytes" {
		respFormat = model.GenerateImagesResponseFormatBase64
	}
	sdkReq.ResponseFormat = strPtr(respFormat)

	if req.Size != "" {
		sdkReq.Size = strPtr(req.Size)
	}

	// Provider-specific knobs forwarded via Extra.
	if req.Extra != nil {
		if seed, ok := toInt64(req.Extra["seed"]); ok {
			sdkReq.Seed = &seed
		}
		if gs, ok := toFloat64(req.Extra["guidance_scale"]); ok {
			sdkReq.GuidanceScale = &gs
		}
		if wm, ok := req.Extra["watermark"].(bool); ok {
			sdkReq.Watermark = &wm
		}
	}

	resp, err := v.client.GenerateImages(ctx, sdkReq, v.requestOptions()...)
	if err != nil {
		return llm.ImageResponse{}, wrapErr(err)
	}

	images := make([]llm.GeneratedImage, 0, len(resp.Data))
	for _, img := range resp.Data {
		if img == nil {
			continue
		}
		var gen llm.GeneratedImage
		switch {
		case img.B64Json != nil && *img.B64Json != "":
			decoded, derr := base64.StdEncoding.DecodeString(*img.B64Json)
			if derr != nil {
				return llm.ImageResponse{}, &llm.InvalidRequestError{Provider: "volcengine", Wrapped: derr}
			}
			gen.Bytes = decoded
		case img.Url != nil:
			gen.URL = *img.Url
		}
		images = append(images, gen)
	}

	out := llm.ImageResponse{
		Images:   images,
		Provider: "volcengine",
		Model:    resp.Model,
	}
	if resp.Usage != nil {
		out.Usage = llm.Usage{
			OutputTokens: int(resp.Usage.OutputTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
			Source:       llm.UsageReported,
		}
	}
	return out, nil
}

// toInt64 coerces common numeric JSON shapes (Extra is map[string]any) to int64.
func toInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case int:
		return int64(n), true
	case int64:
		return n, true
	case float64:
		return int64(n), true
	default:
		return 0, false
	}
}

// toFloat64 coerces common numeric JSON shapes to float64.
func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}
