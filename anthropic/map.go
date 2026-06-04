package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/costa92/llm-agent-contract/llm"
)

// isVisionModel reports whether the bound Claude model accepts image input.
// All modern Claude families are multimodal: Claude 3 (claude-3-*) and Claude 4
// (claude-{sonnet,opus,haiku}-4* and other claude-*-4* ids). The legacy Claude 2
// and instant models are text-only.
func isVisionModel(model string) bool {
	switch {
	case strings.HasPrefix(model, "claude-2"),
		strings.HasPrefix(model, "claude-instant"):
		return false
	case strings.HasPrefix(model, "claude-3"),
		strings.HasPrefix(model, "claude-sonnet-4"),
		strings.HasPrefix(model, "claude-opus-4"),
		strings.HasPrefix(model, "claude-haiku-4"),
		strings.Contains(model, "-4"):
		return true
	default:
		return false
	}
}

func (a *Anthropic) toSDKRequest(req llm.Request) sdk.MessageNewParams {
	msgs := make([]sdk.MessageParam, 0, len(req.Messages))
	sysPrompt := req.SystemPrompt

	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			if len(m.Images) == 0 {
				// Text-only: keep the single text-block form (no regression).
				msgs = append(msgs, sdk.NewUserMessage(sdk.NewTextBlock(m.Content)))
			} else {
				msgs = append(msgs, sdk.NewUserMessage(userContentBlocks(m)...))
			}
		case "assistant":
			msgs = append(msgs, sdk.NewAssistantMessage(sdk.NewTextBlock(m.Content)))
		case "system":
			if sysPrompt == "" {
				sysPrompt = m.Content
			} else {
				sysPrompt += "\n\n" + m.Content
			}
		}
	}

	p := sdk.MessageNewParams{
		Model:     sdk.Model(a.info.Model),
		MaxTokens: 1024,
		Messages:  msgs,
	}
	if sysPrompt != "" {
		p.System = []sdk.TextBlockParam{{Text: sysPrompt}}
	}
	if req.MaxOutputTokens > 0 {
		p.MaxTokens = int64(req.MaxOutputTokens)
	}
	if req.Temperature != nil {
		p.Temperature = sdk.Float(float64(*req.Temperature))
	}
	if len(a.tools) > 0 {
		p.Tools = make([]sdk.ToolUnionParam, 0, len(a.tools))
		p.ToolChoice = sdk.ToolChoiceUnionParam{
			OfAuto: &sdk.ToolChoiceAutoParam{},
		}
		for _, tool := range a.tools {
			schema := toToolInputSchema(tool.Parameters)
			variant := sdk.ToolParam{
				Name:        tool.Name,
				InputSchema: schema,
			}
			if tool.Description != "" {
				variant.Description = sdk.String(tool.Description)
			}
			p.Tools = append(p.Tools, sdk.ToolUnionParam{OfTool: &variant})
		}
	}
	return p
}

// userContentBlocks builds the Anthropic content-block list for a user message
// that carries images: an optional leading text block followed by one image
// block per image.
//
// Bytes -> a base64 image source via sdk.NewImageBlockBase64(mediaType, data),
// defaulting the media type to image/png. For URL: a data: URI is split into
// its media type + base64 payload and sent as a base64 source (so callers can
// pass inline data uniformly); a plain http(s) URL uses the SDK's URL image
// source via sdk.NewImageBlock(sdk.URLImageSourceParam{URL: ...}).
func userContentBlocks(m llm.Message) []sdk.ContentBlockParamUnion {
	blocks := make([]sdk.ContentBlockParamUnion, 0, len(m.Images)+1)
	if m.Content != "" {
		blocks = append(blocks, sdk.NewTextBlock(m.Content))
	}
	for _, img := range m.Images {
		switch {
		case len(img.Bytes) > 0:
			mime := img.MimeType
			if mime == "" {
				mime = "image/png"
			}
			blocks = append(blocks, sdk.NewImageBlockBase64(mime, base64.StdEncoding.EncodeToString(img.Bytes)))
		case strings.HasPrefix(img.URL, "data:"):
			mime, data := splitDataURI(img.URL)
			blocks = append(blocks, sdk.NewImageBlockBase64(mime, data))
		case img.URL != "":
			blocks = append(blocks, sdk.NewImageBlock(sdk.URLImageSourceParam{URL: img.URL}))
		}
	}
	return blocks
}

// splitDataURI parses "data:<mime>;base64,<payload>" into (mime, payload). When
// the input is not a well-formed base64 data URI it returns ("image/png", "").
func splitDataURI(uri string) (mime, data string) {
	rest, ok := strings.CutPrefix(uri, "data:")
	if !ok {
		return "image/png", ""
	}
	meta, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return "image/png", ""
	}
	mime = strings.TrimSuffix(meta, ";base64")
	if mime == "" {
		mime = "image/png"
	}
	return mime, payload
}

func (a *Anthropic) fromSDKResponse(m *sdk.Message) llm.Response {
	var text string
	var toolCalls []llm.ToolCall
	for _, block := range m.Content {
		switch block.Type {
		case "text":
			text += block.Text
		case "tool_use":
			tool := block.AsToolUse()
			toolCalls = append(toolCalls, llm.ToolCall{
				ID:        tool.ID,
				Name:      tool.Name,
				Arguments: append([]byte(nil), tool.Input...),
			})
		}
	}

	return llm.Response{
		Text:         text,
		FinishReason: mapAnthropicStopReason(string(m.StopReason)),
		Provider:     "anthropic",
		Model:        string(m.Model),
		ToolCalls:    toolCalls,
		Usage: llm.Usage{
			InputTokens:  int(m.Usage.InputTokens),
			OutputTokens: int(m.Usage.OutputTokens),
			TotalTokens:  int(m.Usage.InputTokens + m.Usage.OutputTokens),
			Source:       llm.UsageReported,
		},
	}
}

func mapAnthropicStopReason(s string) llm.FinishReason {
	switch s {
	case "end_turn", "stop_sequence":
		return llm.FinishReasonStop
	case "max_tokens":
		return llm.FinishReasonLength
	case "tool_use":
		return llm.FinishReasonToolCalls
	default:
		return llm.FinishReasonUnknown
	}
}

func toToolInputSchema(raw json.RawMessage) sdk.ToolInputSchemaParam {
	schema := sdk.ToolInputSchemaParam{}
	if len(raw) == 0 {
		schema.Type = "object"
		return schema
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		schema.Type = "object"
		return schema
	}

	if props, ok := decoded["properties"]; ok {
		schema.Properties = props
		delete(decoded, "properties")
	}
	if required, ok := decoded["required"].([]any); ok {
		schema.Required = make([]string, 0, len(required))
		for _, item := range required {
			if s, ok := item.(string); ok {
				schema.Required = append(schema.Required, s)
			}
		}
		delete(decoded, "required")
	}
	if typ, ok := decoded["type"].(string); ok && typ == "object" {
		schema.Type = "object"
		delete(decoded, "type")
	}
	if len(decoded) > 0 {
		schema.ExtraFields = decoded
	}
	if schema.Type == "" {
		schema.Type = "object"
	}
	return schema
}
