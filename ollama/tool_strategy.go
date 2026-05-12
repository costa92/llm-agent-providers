package ollama

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/costa92/llm-agent/llm"
	api "github.com/ollama/ollama/api"
)

type ollamaToolStrategy struct {
	family       string
	parserKind   string
	supportsTool bool
}

func strategyForModel(model string) ollamaToolStrategy {
	name := strings.ToLower(model)
	switch {
	case strings.HasPrefix(name, "llama3.1"):
		return ollamaToolStrategy{
			family:       "llama3.1",
			parserKind:   "python_tag",
			supportsTool: true,
		}
	case strings.HasPrefix(name, "qwen2.5-coder"):
		return ollamaToolStrategy{
			family:       "qwen2.5-coder",
			parserKind:   "qwen_json_or_xml",
			supportsTool: true,
		}
	case strings.HasPrefix(name, "qwen3-coder"):
		return ollamaToolStrategy{
			family:       "qwen3-coder",
			parserKind:   "qwen_json_or_xml",
			supportsTool: true,
		}
	default:
		return ollamaToolStrategy{
			family:       "unsupported",
			parserKind:   "",
			supportsTool: false,
		}
	}
}

func parseResponseToolCalls(resp api.ChatResponse, strategy ollamaToolStrategy) ([]llm.ToolCall, string, error) {
	if calls := mapNativeToolCalls(resp.Message.ToolCalls); len(calls) > 0 {
		return calls, resp.Message.Content, nil
	}

	switch strategy.parserKind {
	case "python_tag":
		return parsePythonTagToolCalls(resp.Message.Content)
	case "qwen_json_or_xml":
		return parseQwenToolCalls(resp.Message.Content)
	default:
		return nil, resp.Message.Content, nil
	}
}

func mapNativeToolCalls(calls []api.ToolCall) []llm.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]llm.ToolCall, 0, len(calls))
	for _, call := range calls {
		args, err := json.Marshal(call.Function.Arguments.ToMap())
		if err != nil {
			args = []byte("{}")
		}
		out = append(out, llm.ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: args,
		})
	}
	return out
}

type ollamaToolCallEnvelope struct {
	Name       string          `json:"name"`
	Arguments  json.RawMessage `json:"arguments"`
	Parameters json.RawMessage `json:"parameters"`
}

func parsePythonTagToolCalls(content string) ([]llm.ToolCall, string, error) {
	const tag = "<|python_tag|>"
	idx := strings.Index(content, tag)
	if idx < 0 {
		return nil, content, nil
	}

	before := strings.TrimRightFunc(content[:idx], unicode.IsSpace)
	raw := strings.TrimSpace(content[idx+len(tag):])
	if raw == "" {
		return nil, before, nil
	}

	call, err := decodeFallbackToolCall(raw)
	if err != nil {
		return nil, before, fmt.Errorf("ollama: python_tag tool parse: %w", err)
	}
	return []llm.ToolCall{call}, before, nil
}

var qwenToolCallRe = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

func parseQwenToolCalls(content string) ([]llm.ToolCall, string, error) {
	matches := qwenToolCallRe.FindAllStringSubmatchIndex(content, -1)
	if len(matches) > 0 {
		calls := make([]llm.ToolCall, 0, len(matches))
		var stripped strings.Builder
		last := 0
		for _, match := range matches {
			stripped.WriteString(content[last:match[0]])
			raw := content[match[2]:match[3]]
			call, err := decodeFallbackToolCall(strings.TrimSpace(raw))
			if err != nil {
				return nil, "", fmt.Errorf("ollama: qwen_xml tool parse: %w", err)
			}
			calls = append(calls, call)
			last = match[1]
		}
		stripped.WriteString(content[last:])
		return calls, strings.TrimSpace(stripped.String()), nil
	}

	trimmed := strings.TrimSpace(content)
	if trimmed == "" || !strings.HasPrefix(trimmed, "{") {
		return nil, content, nil
	}
	call, err := decodeFallbackToolCall(trimmed)
	if err != nil {
		return nil, content, nil
	}
	return []llm.ToolCall{call}, "", nil
}

func decodeFallbackToolCall(raw string) (llm.ToolCall, error) {
	var env ollamaToolCallEnvelope
	if err := json.Unmarshal([]byte(raw), &env); err != nil {
		return llm.ToolCall{}, err
	}
	args := env.Arguments
	if len(args) == 0 {
		args = env.Parameters
	}
	if len(args) == 0 {
		args = []byte("{}")
	}
	return llm.ToolCall{
		Name:      env.Name,
		Arguments: args,
	}, nil
}

func unsupportedToolError(model string) error {
	return fmt.Errorf("ollama: model %s native tools unavailable (ProviderInfo.Capabilities.Tools=false): %w", model, llm.ErrCapabilityNotSupported)
}
