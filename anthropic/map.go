package anthropic

import (
	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/costa92/llm-agent/llm"
)

func (a *Anthropic) toSDKRequest(req llm.Request) sdk.MessageNewParams {
	msgs := make([]sdk.MessageParam, 0, len(req.Messages))
	sysPrompt := req.SystemPrompt

	for _, m := range req.Messages {
		switch m.Role {
		case "user":
			msgs = append(msgs, sdk.NewUserMessage(sdk.NewTextBlock(m.Content)))
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
	return p
}

func (a *Anthropic) fromSDKResponse(m *sdk.Message) llm.Response {
	var text string
	for _, block := range m.Content {
		if block.Type == "text" {
			text += block.Text
		}
	}

	return llm.Response{
		Text:         text,
		FinishReason: mapAnthropicStopReason(string(m.StopReason)),
		Provider:     "anthropic",
		Model:        string(m.Model),
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
