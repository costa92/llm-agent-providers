package ollama

import (
	"github.com/costa92/llm-agent/llm"
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
	if req.Temperature != nil {
		p.Options = map[string]any{
			"temperature": *req.Temperature,
		}
	}
	return p
}

func (o *Ollama) fromSDKResponse(resp api.ChatResponse) llm.Response {
	return llm.Response{
		Text:         resp.Message.Content,
		FinishReason: mapOllamaDoneReason(resp.DoneReason),
		Provider:     "ollama",
		Model:        resp.Model,
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
