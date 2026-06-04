package google

import (
	"encoding/base64"
	"encoding/json"
	"strings"

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
			Parts: userParts(m),
		})
	}
	return contents
}

// userParts builds the Gemini Part list for a dialog turn. With no images it is
// the single text part (unchanged behavior). With images, a text part is
// emitted only when Content is non-empty, followed by one image part each:
//   - Bytes -> InlineData (genai.Blob) with MIMEType (default image/png).
//   - URL that is a data: URI -> decoded into InlineData.
//   - URL that is http(s)/gs -> FileData (genai.FileData) with FileURI+MIMEType.
func userParts(m llm.Message) []*genai.Part {
	if len(m.Images) == 0 {
		return []*genai.Part{{Text: m.Content}}
	}
	parts := make([]*genai.Part, 0, len(m.Images)+1)
	if m.Content != "" {
		parts = append(parts, &genai.Part{Text: m.Content})
	}
	for _, img := range m.Images {
		switch {
		case len(img.Bytes) > 0:
			parts = append(parts, &genai.Part{InlineData: &genai.Blob{Data: img.Bytes, MIMEType: defaultMime(img.MimeType)}})
		case strings.HasPrefix(img.URL, "data:"):
			mime, data := decodeDataURI(img.URL)
			parts = append(parts, &genai.Part{InlineData: &genai.Blob{Data: data, MIMEType: mime}})
		case img.URL != "":
			parts = append(parts, &genai.Part{FileData: &genai.FileData{FileURI: img.URL, MIMEType: defaultMime(img.MimeType)}})
		}
	}
	return parts
}

func defaultMime(mime string) string {
	if mime == "" {
		return "image/png"
	}
	return mime
}

// decodeDataURI parses "data:<mime>;base64,<payload>" into (mime, rawBytes).
// On any parse failure it returns ("image/png", nil).
func decodeDataURI(uri string) (mime string, data []byte) {
	rest, ok := strings.CutPrefix(uri, "data:")
	if !ok {
		return "image/png", nil
	}
	meta, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return "image/png", nil
	}
	mime = strings.TrimSuffix(meta, ";base64")
	if mime == "" {
		mime = "image/png"
	}
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return mime, nil
	}
	return mime, decoded
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
