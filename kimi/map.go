package kimi

import (
	"encoding/json"

	"github.com/costa92/llm-agent-contract/llm"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
)

func (k *Kimi) toSDKRequest(req llm.Request) openai.ChatCompletionNewParams {
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
		Model:    k.info.Model,
		Messages: msgs,
	}
	if req.MaxOutputTokens > 0 {
		p.MaxCompletionTokens = openai.Int(int64(req.MaxOutputTokens))
	}
	if req.Temperature != nil {
		p.Temperature = openai.Float(float64(*req.Temperature))
	}
	if len(k.tools) > 0 {
		p.Tools = make([]openai.ChatCompletionToolUnionParam, 0, len(k.tools))
		p.ParallelToolCalls = openai.Bool(true)
		for _, tool := range k.tools {
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

func (k *Kimi) toSDKStreamRequest(req llm.Request) openai.ChatCompletionNewParams {
	p := k.toSDKRequest(req)
	p.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}
	return p
}

func (k *Kimi) fromSDKResponse(c *openai.ChatCompletion) llm.Response {
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
		Provider:     "kimi",
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
