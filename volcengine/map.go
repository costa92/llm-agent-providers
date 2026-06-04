package volcengine

import (
	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// strPtr returns a pointer to s (helper for the SDK's *string / union fields).
func strPtr(s string) *string { return &s }

// contentString wraps a plain string in the SDK's content union.
func contentString(s string) *model.ChatCompletionMessageContent {
	return &model.ChatCompletionMessageContent{StringValue: strPtr(s)}
}

// toSDKRequest maps an llm.Request to the pointer-field
// CreateChatCompletionRequest (so temperature=0 is sendable). Streaming is
// set by the caller (Stream path) via the SDK, not here.
func (v *Volcengine) toSDKRequest(req llm.Request) model.CreateChatCompletionRequest {
	msgs := make([]*model.ChatCompletionMessage, 0, len(req.Messages)+1)
	if req.SystemPrompt != "" {
		msgs = append(msgs, &model.ChatCompletionMessage{
			Role:    model.ChatMessageRoleSystem,
			Content: contentString(req.SystemPrompt),
		})
	}
	for _, m := range req.Messages {
		role := m.Role
		switch role {
		case "user", "assistant", "system", "tool":
			// pass through
		default:
			role = model.ChatMessageRoleUser
		}
		msgs = append(msgs, &model.ChatCompletionMessage{
			Role:    role,
			Content: contentString(m.Content),
		})
	}

	out := model.CreateChatCompletionRequest{
		Model:    v.info.Model,
		Messages: msgs,
	}
	if req.MaxOutputTokens > 0 {
		mt := req.MaxOutputTokens
		out.MaxTokens = &mt
	}
	if req.Temperature != nil {
		t := *req.Temperature
		out.Temperature = &t
	}
	if len(v.tools) > 0 {
		out.Tools = make([]*model.Tool, 0, len(v.tools))
		for _, tool := range v.tools {
			def := &model.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
			}
			if len(tool.Parameters) > 0 {
				// Parameters is interface{}; the SDK marshals it verbatim.
				// json.RawMessage marshals as the raw schema bytes.
				def.Parameters = tool.Parameters
			}
			out.Tools = append(out.Tools, &model.Tool{
				Type:     model.ToolTypeFunction,
				Function: def,
			})
		}
	}
	return out
}

// fromSDKResponse maps a non-stream ChatCompletionResponse to llm.Response.
func (v *Volcengine) fromSDKResponse(resp model.ChatCompletionResponse) llm.Response {
	var text string
	finish := llm.FinishReasonUnknown
	var toolCalls []llm.ToolCall
	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		if choice.Message.Content != nil && choice.Message.Content.StringValue != nil {
			text = *choice.Message.Content.StringValue
		}
		finish = mapFinishReason(choice.FinishReason)
		toolCalls = mapToolCalls(choice.Message.ToolCalls)
	}
	return llm.Response{
		Text:         text,
		FinishReason: finish,
		Provider:     "volcengine",
		Model:        resp.Model,
		ToolCalls:    toolCalls,
		Usage: llm.Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
			Source:       llm.UsageReported,
		},
	}
}

// mapToolCalls converts SDK tool calls to the contract shape.
func mapToolCalls(calls []*model.ToolCall) []llm.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]llm.ToolCall, 0, len(calls))
	for _, call := range calls {
		if call == nil || call.Type != model.ToolTypeFunction {
			continue
		}
		out = append(out, llm.ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: []byte(call.Function.Arguments),
		})
	}
	return out
}

// mapFinishReason maps the SDK finish reason to the contract enum.
func mapFinishReason(r model.FinishReason) llm.FinishReason {
	switch r {
	case model.FinishReasonStop:
		return llm.FinishReasonStop
	case model.FinishReasonLength:
		return llm.FinishReasonLength
	case model.FinishReasonContentFilter:
		return llm.FinishReasonContentFilter
	case model.FinishReasonToolCalls:
		return llm.FinishReasonToolCalls
	case model.FinishReasonFunctionCall:
		return llm.FinishReasonFunctionCall
	default:
		return llm.FinishReasonUnknown
	}
}

// toSDKStreamRequest builds the streaming variant: same as toSDKRequest but
// with StreamOptions.IncludeUsage so the final chunk carries token usage.
// (The SDK sets Stream=true internally via request.WithStream(true).)
func (v *Volcengine) toSDKStreamRequest(req llm.Request) model.CreateChatCompletionRequest {
	out := v.toSDKRequest(req)
	out.StreamOptions = &model.StreamOptions{IncludeUsage: true}
	return out
}
