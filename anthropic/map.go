package anthropic

import (
	"encoding/json"

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
