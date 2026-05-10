package openai

import (
	"github.com/costa92/llm-agent/llm"
	openai "github.com/openai/openai-go/v3"
)

func (o *OpenAI) toSDKRequest(req llm.Request) openai.ChatCompletionNewParams {
	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages)+1)
	if req.SystemPrompt != "" {
		msgs = append(msgs, openai.SystemMessage(req.SystemPrompt))
	}
	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			msgs = append(msgs, openai.UserMessage(m.Content))
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
	return p
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
