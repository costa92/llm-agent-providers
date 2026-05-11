package contract

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/costa92/llm-agent/llm"
)

type Fixture struct {
	Scenario string `json:"scenario"`
	Tools    []struct {
		Name        string `json:"name"`
		Description string `json:"description,omitempty"`
		Parameters  string `json:"parameters"`
	} `json:"tools,omitempty"`
	Request struct {
		Method         string   `json:"method"`
		Path           string   `json:"path"`
		BodyAssertions []string `json:"body_assertions"`
	} `json:"request"`
	Response struct {
		Status  int               `json:"status"`
		Headers map[string]string `json:"headers"`
		Body    string            `json:"body"`
	} `json:"response"`
	Expect struct {
		ErrorType         string `json:"error_type,omitempty"`
		ResponseText      string `json:"response_text,omitempty"`
		FinishReason      string `json:"finish_reason,omitempty"`
		UsageInputTokens  int    `json:"usage_input_tokens,omitempty"`
		UsageOutputTokens int    `json:"usage_output_tokens,omitempty"`
		UsageSource       string `json:"usage_source,omitempty"`
		Provider          string `json:"provider,omitempty"`
		ToolCalls         []struct {
			ID        string `json:"id,omitempty"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"tool_calls,omitempty"`
	} `json:"expect"`
}

func LoadFixture(t *testing.T, provider, scenario string) Fixture {
	t.Helper()
	p := filepath.Join("testdata", provider, scenario+".json")
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("LoadFixture(%s/%s): %v", provider, scenario, err)
	}
	var f Fixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatalf("LoadFixture(%s/%s): unmarshal: %v", provider, scenario, err)
	}
	return f
}

func NewMockServer(t *testing.T, f Fixture) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.Request.Method != "" && r.Method != f.Request.Method {
			t.Errorf("method: got %q want %q", r.Method, f.Request.Method)
		}
		if f.Request.Path != "" && !strings.HasPrefix(r.URL.Path, f.Request.Path) {
			t.Errorf("path: got %q want prefix %q", r.URL.Path, f.Request.Path)
		}
		body, _ := io.ReadAll(r.Body)
		for _, want := range f.Request.BodyAssertions {
			if !assertBody(string(body), want) {
				t.Errorf("body assertion failed: %q not satisfied by %q", want, string(body))
			}
		}
		for k, v := range f.Response.Headers {
			w.Header().Set(k, v)
		}
		w.WriteHeader(f.Response.Status)
		_, _ = w.Write([]byte(f.Response.Body))
	}))
}

func assertBody(body, assertion string) bool {
	s := strings.TrimSpace(assertion)
	if i := strings.Index(s, "="); i >= 0 && !strings.Contains(s[:i], " ") {
		return strings.Contains(body, s[i+1:])
	}
	return strings.Contains(body, s)
}

type ChatModelFactory func(baseURL string) (llm.ChatModel, error)

func AssertGenerate(t *testing.T, model llm.ChatModel, f Fixture) {
	t.Helper()
	req := llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hello"}},
	}
	resp, err := model.Generate(t.Context(), req)
	assertResponse(t, resp, err, f, "Generate")
}

func AssertStream(t *testing.T, model llm.ChatModel, f Fixture) {
	t.Helper()
	req := llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hello"}},
	}
	sr, err := model.Stream(t.Context(), req)
	if err != nil {
		t.Fatalf("Stream: unexpected error creating stream: %v", err)
	}
	resp, err := llm.AccumulateStream(sr)
	info := model.Info()
	if resp.Provider == "" {
		resp.Provider = info.Provider
	}
	if resp.Model == "" {
		resp.Model = info.Model
	}
	assertResponse(t, resp, err, f, "Stream")
}

func AssertToolCalling(t *testing.T, model llm.ChatModel, f Fixture) {
	t.Helper()
	tc, ok := model.(llm.ToolCaller)
	if !ok {
		t.Fatalf("model %T does not implement llm.ToolCaller", model)
	}
	tools := make([]llm.Tool, 0, len(f.Tools))
	for _, tool := range f.Tools {
		tools = append(tools, llm.Tool{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  []byte(tool.Parameters),
		})
	}
	bound, err := tc.WithTools(tools)
	if f.Expect.ErrorType == "CapabilityNotSupported" {
		if err == nil {
			t.Fatal("WithTools(): error = nil, want capability error")
		}
		if !errors.Is(err, llm.ErrCapabilityNotSupported) {
			t.Fatalf("errors.Is(err, ErrCapabilityNotSupported) = false, err=%T %v", err, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}
	req := llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "use tools"}},
	}
	resp, err := bound.Generate(t.Context(), req)
	assertResponse(t, resp, err, f, "Generate")
}

func assertResponse(t *testing.T, resp llm.Response, err error, f Fixture, op string) {
	t.Helper()
	switch f.Expect.ErrorType {
	case "":
		if err != nil {
			t.Fatalf("%s: unexpected error: %v", op, err)
		}
		if f.Expect.ResponseText != "" && resp.Text != f.Expect.ResponseText {
			t.Errorf("Text: got %q want %q", resp.Text, f.Expect.ResponseText)
		}
		if f.Expect.FinishReason != "" && string(resp.FinishReason) != f.Expect.FinishReason {
			t.Errorf("FinishReason: got %q want %q", resp.FinishReason, f.Expect.FinishReason)
		}
		if f.Expect.UsageInputTokens != 0 && resp.Usage.InputTokens != f.Expect.UsageInputTokens {
			t.Errorf("Usage.InputTokens: got %d want %d", resp.Usage.InputTokens, f.Expect.UsageInputTokens)
		}
		if f.Expect.UsageOutputTokens != 0 && resp.Usage.OutputTokens != f.Expect.UsageOutputTokens {
			t.Errorf("Usage.OutputTokens: got %d want %d", resp.Usage.OutputTokens, f.Expect.UsageOutputTokens)
		}
		if f.Expect.UsageSource != "" && string(resp.Usage.Source) != f.Expect.UsageSource {
			t.Errorf("Usage.Source: got %q want %q", resp.Usage.Source, f.Expect.UsageSource)
		}
		if f.Expect.Provider != "" && resp.Provider != f.Expect.Provider {
			t.Errorf("Provider: got %q want %q", resp.Provider, f.Expect.Provider)
		}
		if len(f.Expect.ToolCalls) != 0 {
			if len(resp.ToolCalls) != len(f.Expect.ToolCalls) {
				t.Fatalf("ToolCalls len: got %d want %d", len(resp.ToolCalls), len(f.Expect.ToolCalls))
			}
			for i, want := range f.Expect.ToolCalls {
				got := resp.ToolCalls[i]
				if want.ID != "" && got.ID != want.ID {
					t.Errorf("ToolCalls[%d].ID: got %q want %q", i, got.ID, want.ID)
				}
				if got.Name != want.Name {
					t.Errorf("ToolCalls[%d].Name: got %q want %q", i, got.Name, want.Name)
				}
				if !jsonEqual(got.Arguments, []byte(want.Arguments)) {
					t.Errorf("ToolCalls[%d].Arguments: got %s want %s", i, string(got.Arguments), want.Arguments)
				}
			}
		}
	case "AuthError":
		var e *llm.AuthError
		if !errors.As(err, &e) {
			t.Errorf("expected *llm.AuthError, got %T: %v", err, err)
		}
	case "RateLimitError":
		var e *llm.RateLimitError
		if !errors.As(err, &e) {
			t.Errorf("expected *llm.RateLimitError, got %T: %v", err, err)
		}
	case "InvalidRequestError":
		var e *llm.InvalidRequestError
		if !errors.As(err, &e) {
			t.Errorf("expected *llm.InvalidRequestError, got %T: %v", err, err)
		}
	case "TransientError":
		var e *llm.TransientError
		if !errors.As(err, &e) {
			t.Errorf("expected *llm.TransientError, got %T: %v", err, err)
		}
	case "CapabilityNotSupported":
		if !errors.Is(err, llm.ErrCapabilityNotSupported) {
			t.Errorf("expected ErrCapabilityNotSupported, got %T: %v", err, err)
		}
	default:
		t.Errorf("unknown Fixture.Expect.ErrorType: %q", f.Expect.ErrorType)
	}
}

func jsonEqual(a, b []byte) bool {
	var av any
	var bv any
	if err := json.Unmarshal(a, &av); err != nil {
		return string(a) == string(b)
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		return string(a) == string(b)
	}
	ab, _ := json.Marshal(av)
	bb, _ := json.Marshal(bv)
	return string(ab) == string(bb)
}
