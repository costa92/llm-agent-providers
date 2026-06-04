package google

import (
	"encoding/json"

	"github.com/costa92/llm-agent-contract/llm"
	"google.golang.org/genai"
)

// toContents maps the dialog turns to Gemini Contents. Gemini roles are only
// user/model: assistant => model; system/tool turns are skipped (system is
// lifted to SystemInstruction in toGenConfig).
func toContents(req llm.Request) []*genai.Content {
	contents := make([]*genai.Content, 0, len(req.Messages))
	for _, m := range req.Messages {
		var role string
		switch m.Role {
		case "user":
			role = genai.RoleUser
		case "assistant":
			role = genai.RoleModel
		default:
			continue
		}
		contents = append(contents, &genai.Content{
			Role:  role,
			Parts: []*genai.Part{{Text: m.Content}},
		})
	}
	return contents
}

// toGenConfig maps request knobs + bound tools to a GenerateContentConfig.
// Returns nil when there is nothing to configure (no system prompt, no
// sampling overrides, no tools).
func (g *Google) toGenConfig(req llm.Request) *genai.GenerateContentConfig {
	cfg := &genai.GenerateContentConfig{}
	set := false
	if req.SystemPrompt != "" {
		cfg.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: req.SystemPrompt}},
		}
		set = true
	}
	if req.Temperature != nil {
		cfg.Temperature = genai.Ptr(*req.Temperature)
		set = true
	}
	if req.MaxOutputTokens > 0 {
		cfg.MaxOutputTokens = int32(req.MaxOutputTokens)
		set = true
	}
	if len(g.tools) > 0 {
		decls := make([]*genai.FunctionDeclaration, 0, len(g.tools))
		for _, tool := range g.tools {
			decl := &genai.FunctionDeclaration{
				Name:        tool.Name,
				Description: tool.Description,
			}
			if len(tool.Parameters) > 0 {
				var schema map[string]any
				if err := json.Unmarshal(tool.Parameters, &schema); err == nil {
					decl.ParametersJsonSchema = schema
				}
			}
			decls = append(decls, decl)
		}
		cfg.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
		set = true
	}
	if !set {
		return nil
	}
	return cfg
}

// fromResponse maps a GenerateContentResponse to the repo's llm.Response.
func (g *Google) fromResponse(resp *genai.GenerateContentResponse) llm.Response {
	out := llm.Response{
		Text:         resp.Text(),
		Provider:     "google",
		Model:        g.info.Model,
		FinishReason: llm.FinishReasonUnknown,
		ToolCalls:    toToolCalls(resp),
	}
	if resp.ModelVersion != "" {
		out.Model = resp.ModelVersion
	}
	if len(resp.Candidates) > 0 {
		out.FinishReason = mapFinishReason(resp.Candidates[0].FinishReason)
	}
	if um := resp.UsageMetadata; um != nil {
		out.Usage = llm.Usage{
			InputTokens:  int(um.PromptTokenCount),
			OutputTokens: int(um.CandidatesTokenCount),
			TotalTokens:  int(um.TotalTokenCount),
			Source:       llm.UsageReported,
		}
	}
	return out
}

// toToolCalls extracts function calls from the first candidate, re-marshalling
// Args (map[string]any) back to a JSON string for llm.ToolCall.Arguments.
func toToolCalls(resp *genai.GenerateContentResponse) []llm.ToolCall {
	fcs := resp.FunctionCalls()
	if len(fcs) == 0 {
		return nil
	}
	out := make([]llm.ToolCall, 0, len(fcs))
	for _, fc := range fcs {
		args, err := json.Marshal(fc.Args)
		if err != nil || fc.Args == nil {
			args = []byte("{}")
		}
		out = append(out, llm.ToolCall{
			ID:        fc.ID,
			Name:      fc.Name,
			Arguments: args,
		})
	}
	return out
}

// mapFinishReason maps Gemini finish reasons to the contract's reasons.
func mapFinishReason(fr genai.FinishReason) llm.FinishReason {
	switch fr {
	case genai.FinishReasonStop:
		return llm.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		return llm.FinishReasonLength
	case genai.FinishReasonSafety, genai.FinishReasonRecitation:
		return llm.FinishReasonContentFilter
	default:
		return llm.FinishReasonUnknown
	}
}
