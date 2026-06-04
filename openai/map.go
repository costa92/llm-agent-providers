package openai

import (
	"encoding/base64"
	"encoding/json"

	"github.com/costa92/llm-agent-contract/llm"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
)

func (o *OpenAI) toSDKRequest(req llm.Request) openai.ChatCompletionNewParams {
	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)
	if req.SystemPrompt != "" {
		msgs = append(msgs, openai.SystemMessage(req.SystemPrompt))
	}
	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			if len(m.Images) == 0 {
				// Text-only: keep the plain-string content form (no regression).
				msgs = append(msgs, openai.UserMessage(m.Content))
			} else {
				msgs = append(msgs, openai.UserMessage(userContentParts(m)))
			}
		case "assistant":
			msgs = append(msgs, openai.AssistantMessage(m.Content))
		case "system":
			msgs = append(msgs, openai.SystemMessage(m.Content))
		}
	}

	p := openai.ChatCompletionNewParams{
		Model:    o.info.Model,
		Messages: msgs,
	}
	if req.MaxOutputTokens > 0 {
		p.MaxCompletionTokens = openai.Int(int64(req.MaxOutputTokens))
	}
	if req.Temperature != nil {
		p.Temperature = openai.Float(float64(*req.Temperature))
	}
	if len(o.tools) > 0 {
		p.Tools = make([]openai.ChatCompletionToolUnionParam, 0, len(o.tools))
		p.ParallelToolCalls = openai.Bool(true)
		for _, tool := range o.tools {
			def := shared.FunctionDefinitionParam{Name: tool.Name}
			if tool.Description != "" {
				def.Description = param.NewOpt(tool.Description)
			}
			if len(tool.Parameters) > 0 {
				var schema shared.FunctionParameters
				if err := json.Unmarshal(tool.Parameters, &schema); err == nil {
					def.Parameters = schema
				}
			}
			p.Tools = append(p.Tools, openai.ChatCompletionFunctionTool(def))
		}
	}
	return p
}

// userContentParts builds the OpenAI multimodal content array for a user
// message that carries images: an optional leading text part followed by one
// image_url part per image. Bytes are encoded as a data:<MimeType>;base64,<...>
// URI (defaulting MimeType to image/png); a URL is passed through verbatim
// (OpenAI accepts http(s) URLs and data: URIs). Detail is set only when
// non-empty.
func userContentParts(m llm.Message) []openai.ChatCompletionContentPartUnionParam {
	parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(m.Images)+1)
	if m.Content != "" {
		parts = append(parts, openai.TextContentPart(m.Content))
	}
	for _, img := range m.Images {
		url := img.URL
		if len(img.Bytes) > 0 {
			mime := img.MimeType
			if mime == "" {
				mime = "image/png"
			}
			url = "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(img.Bytes)
		}
		imageURL := openai.ChatCompletionContentPartImageImageURLParam{URL: url}
		if img.Detail != "" {
			imageURL.Detail = img.Detail
		}
		parts = append(parts, openai.ImageContentPart(imageURL))
	}
	return parts
}

func (o *OpenAI) toSDKStreamRequest(req llm.Request) openai.ChatCompletionNewParams {
	p := o.toSDKRequest(req)
	p.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}
	return p
}

func (o *OpenAI) fromSDKResponse(c *openai.ChatCompletion) llm.Response {
	var text string
	if len(c.Choices) > 0 {
		text = c.Choices[0].Message.Content
	}

	finish := llm.FinishReasonUnknown
	if len(c.Choices) > 0 {
		finish = mapFinishReason(c.Choices[0].FinishReason)
	}

	return llm.Response{
		Text:         text,
		FinishReason: finish,
		Provider:     "openai",
		Model:        c.Model,
		ToolCalls:    sdkToolCalls(c),
		Usage: llm.Usage{
			InputTokens:  int(c.Usage.PromptTokens),
			OutputTokens: int(c.Usage.CompletionTokens),
			TotalTokens:  int(c.Usage.TotalTokens),
			Source:       llm.UsageReported,
		},
	}
}

func mapFinishReason(s string) llm.FinishReason {
	switch s {
	case "stop":
		return llm.FinishReasonStop
	case "length":
		return llm.FinishReasonLength
	case "content_filter":
		return llm.FinishReasonContentFilter
	case "tool_calls":
		return llm.FinishReasonToolCalls
	case "function_call":
		return llm.FinishReasonFunctionCall
	default:
		return llm.FinishReasonUnknown
	}
}

func sdkToolCalls(c *openai.ChatCompletion) []llm.ToolCall {
	if len(c.Choices) == 0 {
		return nil
	}
	calls := c.Choices[0].Message.ToolCalls
	if len(calls) == 0 {
		return nil
	}
	out := make([]llm.ToolCall, 0, len(calls))
	for _, call := range calls {
		if call.Type != "function" {
			continue
		}
		fn := call.AsFunction()
		out = append(out, llm.ToolCall{
			ID:        fn.ID,
			Name:      fn.Function.Name,
			Arguments: []byte(fn.Function.Arguments),
		})
	}
	return out
}
