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
	Request  struct {
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

	switch f.Expect.ErrorType {
	case "":
		if err != nil {
			t.Fatalf("Generate: unexpected error: %v", err)
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
	default:
		t.Errorf("unknown Fixture.Expect.ErrorType: %q", f.Expect.ErrorType)
	}
}
