package ollama

import (
	"encoding/json"

	"github.com/costa92/llm-agent-contract/llm"
	api "github.com/ollama/ollama/api"
)

func (o *Ollama) toSDKRequest(req llm.Request) *api.ChatRequest {
	streamOff := false
	msgs := make([]api.Message, 0, len(req.Messages)+1)
	if req.SystemPrompt != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: req.SystemPrompt})
	}
	for _, m := range req.Messages {
		msgs = append(msgs, api.Message{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	p := &api.ChatRequest{
		Model:    o.info.Model,
		Messages: msgs,
		Stream:   &streamOff,
	}
	if len(o.tools) > 0 {
		p.Tools = make(api.Tools, 0, len(o.tools))
		for _, tool := range o.tools {
			p.Tools = append(p.Tools, toAPITool(tool))
		}
	}
	if req.Temperature != nil {
		p.Options = map[string]any{
			"temperature": *req.Temperature,
		}
	}
	return p
}

func (o *Ollama) toSDKStreamRequest(req llm.Request) *api.ChatRequest {
	streamOn := true
	p := o.toSDKRequest(req)
	p.Stream = &streamOn
	return p
}

func (o *Ollama) fromSDKResponse(resp api.ChatResponse) llm.Response {
	toolCalls, text, _ := parseResponseToolCalls(resp, o.strategy)
	return llm.Response{
		Text:         text,
		FinishReason: mapOllamaDoneReason(resp.DoneReason),
		Provider:     "ollama",
		Model:        resp.Model,
		ToolCalls:    toolCalls,
		Usage: llm.Usage{
			InputTokens:  resp.PromptEvalCount,
			OutputTokens: resp.EvalCount,
			TotalTokens:  resp.PromptEvalCount + resp.EvalCount,
			Source:       llm.UsageReported,
		},
	}
}

func mapOllamaDoneReason(s string) llm.FinishReason {
	switch s {
	case "stop", "load":
		return llm.FinishReasonStop
	case "length":
		return llm.FinishReasonLength
	default:
		return llm.FinishReasonUnknown
	}
}

func toAPITool(tool llm.Tool) api.Tool {
	out := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  api.ToolFunctionParameters{Type: "object"},
		},
	}
	if len(tool.Parameters) == 0 {
		return out
	}

	var raw map[string]any
	if err := json.Unmarshal(tool.Parameters, &raw); err != nil {
		return out
	}
	if typ, ok := raw["type"].(string); ok && typ != "" {
		out.Function.Parameters.Type = typ
	}
	if defs, ok := raw["$defs"]; ok {
		out.Function.Parameters.Defs = defs
	}
	if items, ok := raw["items"]; ok {
		out.Function.Parameters.Items = items
	}
	if required, ok := raw["required"].([]any); ok {
		out.Function.Parameters.Required = make([]string, 0, len(required))
		for _, item := range required {
			if s, ok := item.(string); ok {
				out.Function.Parameters.Required = append(out.Function.Parameters.Required, s)
			}
		}
	}
	if props, ok := raw["properties"].(map[string]any); ok {
		out.Function.Parameters.Properties = toToolPropertiesMap(props)
	}
	return out
}

func toToolPropertiesMap(props map[string]any) *api.ToolPropertiesMap {
	out := api.NewToolPropertiesMap()
	for name, raw := range props {
		propMap, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		out.Set(name, toToolProperty(propMap))
	}
	return out
}

func toToolProperty(raw map[string]any) api.ToolProperty {
	prop := api.ToolProperty{}
	if typ, ok := raw["type"].(string); ok && typ != "" {
		prop.Type = api.PropertyType{typ}
	} else if arr, ok := raw["type"].([]any); ok {
		for _, item := range arr {
			if s, ok := item.(string); ok {
				prop.Type = append(prop.Type, s)
			}
		}
	}
	if desc, ok := raw["description"].(string); ok {
		prop.Description = desc
	}
	if enum, ok := raw["enum"].([]any); ok {
		prop.Enum = append([]any(nil), enum...)
	}
	if items, ok := raw["items"]; ok {
		prop.Items = items
	}
	if required, ok := raw["required"].([]any); ok {
		prop.Required = make([]string, 0, len(required))
		for _, item := range required {
			if s, ok := item.(string); ok {
				prop.Required = append(prop.Required, s)
			}
		}
	}
	if nested, ok := raw["properties"].(map[string]any); ok {
		prop.Properties = toToolPropertiesMap(nested)
	}
	if anyOf, ok := raw["anyOf"].([]any); ok {
		prop.AnyOf = make([]api.ToolProperty, 0, len(anyOf))
		for _, item := range anyOf {
			if m, ok := item.(map[string]any); ok {
				prop.AnyOf = append(prop.AnyOf, toToolProperty(m))
			}
		}
	}
	return prop
}
