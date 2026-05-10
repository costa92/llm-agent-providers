package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/costa92/llm-agent/llm"
)

func TestNew_RequiresModel(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("New() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), "WithModel is required") {
		t.Fatalf("New() error = %q, want WithModel is required", err)
	}
}

func TestInfo_OpenAI(t *testing.T) {
	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if info.Provider != "openai" {
		t.Fatalf("Provider = %q, want openai", info.Provider)
	}
	if info.Model != "gpt-4o-mini" {
		t.Fatalf("Model = %q, want gpt-4o-mini", info.Model)
	}
	if info.Capabilities.Tools || info.Capabilities.Embeddings || info.Capabilities.StructuredOutputs || info.Capabilities.PromptCaching {
		t.Fatalf("Capabilities = %+v, want all false", info.Capabilities)
	}
}

func TestStream_OpenAI_Happy(t *testing.T) {
	var sawIncludeUsage atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"include_usage":true`) {
			sawIncludeUsage.Store(true)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "say hello"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if resp.Text != "hello" {
		t.Fatalf("Text = %q, want hello", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonStop)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Fatalf("Usage.Source = %q, want %q", resp.Usage.Source, llm.UsageReported)
	}
	if resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Fatalf("Usage = %+v, want prompt=11 completion=7 total=18", resp.Usage)
	}
	if !sawIncludeUsage.Load() {
		t.Fatal(`request body missing "stream_options.include_usage=true"`)
	}
}

func TestStream_OpenAI_RetriesBeforeFirstByte(t *testing.T) {
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		if n == 1 {
			_, _ = fmt.Fprint(w, "data: {\"error\":{\"message\":\"upstream exploded\"}}\n\n")
			return
		}

		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if resp.Text != "ok" {
		t.Fatalf("Text = %q, want ok", resp.Text)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("attempts = %d, want 2", got)
	}
}

func TestStream_OpenAI_DoesNotRetryAfterFirstByte(t *testing.T) {
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"error\":{\"message\":\"stream interrupted\"}}\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	ev, err := sr.Next()
	if err != nil {
		t.Fatalf("Next#1: %v", err)
	}
	if ev.Kind != llm.EventTextDelta || ev.Text != "par" {
		t.Fatalf("Next#1 = %+v, want text delta par", ev)
	}

	_, err = sr.Next()
	if err == nil {
		t.Fatal("Next#2 error = nil, want non-nil")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
}

func TestGenerate_OpenAI_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("Path = %s, want /chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl_123",
			"object":"chat.completion",
			"created":1710000000,
			"model":"gpt-4o-mini",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello world"},"finish_reason":"stop","logprobs":null}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("gpt-4o-mini"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		SystemPrompt: "be concise",
		Messages: []llm.Message{
			{Role: "user", Content: "say hello"},
		},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if resp.Text != "hello world" {
		t.Fatalf("Text = %q, want hello world", resp.Text)
	}
	if resp.Provider != "openai" {
		t.Fatalf("Provider = %q, want openai", resp.Provider)
	}
	if resp.Model != "gpt-4o-mini" {
		t.Fatalf("Model = %q, want gpt-4o-mini", resp.Model)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonStop)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Fatalf("Usage.Source = %q, want %q", resp.Usage.Source, llm.UsageReported)
	}
}

func TestGenerate_OpenAI_401AuthError(t *testing.T) {
	assertTypedError(t, 401, `{"error":{"message":"bad key","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}`, func(err error) bool {
		var target *llm.AuthError
		return errors.As(err, &target)
	})
}

func TestGenerate_OpenAI_403AuthError(t *testing.T) {
	assertTypedError(t, 403, `{"error":{"message":"forbidden","type":"invalid_request_error","param":null,"code":"forbidden"}}`, func(err error) bool {
		var target *llm.AuthError
		return errors.As(err, &target)
	})
}

func TestGenerate_OpenAI_429QuotaError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"error":{"message":"quota exceeded","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err == nil {
		t.Fatal("Generate() error = nil, want non-nil")
	}
	var target *llm.RateLimitError
	if !errors.As(err, &target) {
		t.Fatalf("errors.As(err, *llm.RateLimitError) = false, err=%T %v", err, err)
	}
	if target.Reason != "quota_exhausted" {
		t.Fatalf("Reason = %q, want quota_exhausted", target.Reason)
	}
	if target.RetryAfter != "30" {
		t.Fatalf("RetryAfter = %q, want 30", target.RetryAfter)
	}
}

func TestGenerate_OpenAI_429RateLimitError(t *testing.T) {
	assertTypedError(t, 429, `{"error":{"message":"too many requests","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}`, func(err error) bool {
		var target *llm.RateLimitError
		return errors.As(err, &target)
	})
}

func TestGenerate_OpenAI_500TransientError(t *testing.T) {
	assertTypedError(t, 500, `{"error":{"message":"server error","type":"server_error","param":null,"code":null}}`, func(err error) bool {
		var target *llm.TransientError
		return errors.As(err, &target)
	})
}

func TestGenerate_OpenAI_404InvalidRequestError(t *testing.T) {
	assertTypedError(t, 404, `{"error":{"message":"model not found","type":"invalid_request_error","param":null,"code":"model_not_found"}}`, func(err error) bool {
		var target *llm.InvalidRequestError
		return errors.As(err, &target)
	})
}

func assertTypedError(t *testing.T, status int, body string, match func(error) bool) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err == nil {
		t.Fatal("Generate() error = nil, want non-nil")
	}
	if !match(err) {
		t.Fatalf("typed error assertion failed for status %d: %T %v", status, err, err)
	}
}
