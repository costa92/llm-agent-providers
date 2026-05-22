package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/costa92/llm-agent/llm"
)

func TestNew_RequiresModel(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("New() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), "WithModel is required") {
		t.Fatalf("New() error = %q", err)
	}
}

func TestInfo_Ollama(t *testing.T) {
	m, err := New(WithModel("llama3.1:8b"), WithBaseURL("http://localhost:11434"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if info.Provider != "ollama" {
		t.Fatalf("Provider = %q, want ollama", info.Provider)
	}
	if info.Model != "llama3.1:8b" {
		t.Fatalf("Model = %q, want llama3.1:8b", info.Model)
	}
	if !info.Capabilities.Tools || info.Capabilities.Embeddings || info.Capabilities.StructuredOutputs || info.Capabilities.PromptCaching {
		t.Fatalf("Capabilities = %+v, want tools=true and others=false", info.Capabilities)
	}
}

func TestInfo_Ollama_Qwen25CoderSupportsTools(t *testing.T) {
	m, err := New(WithModel("qwen2.5-coder:latest"), WithBaseURL("http://localhost:11434"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if !info.Capabilities.Tools {
		t.Fatalf("Capabilities = %+v, want tools=true", info.Capabilities)
	}
}

func TestInfo_Ollama_EmbeddingModel(t *testing.T) {
	m, err := New(WithModel("nomic-embed-text"), WithBaseURL("http://localhost:11434"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if !info.Capabilities.Embeddings {
		t.Fatalf("Capabilities = %+v, want embeddings=true", info.Capabilities)
	}
	if got := m.EmbedDimensions(); got != 768 {
		t.Fatalf("EmbedDimensions() = %d, want 768", got)
	}
}

func TestEmbed_Ollama_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/api/embed" {
			t.Fatalf("Path = %s, want /api/embed", r.URL.Path)
		}
		body := readBody(t, r)
		if !strings.Contains(body, `"model":"nomic-embed-text"`) {
			t.Fatalf("request body missing embedding model: %s", body)
		}
		if !strings.Contains(body, `"input":["hello","world"]`) {
			t.Fatalf("request body missing batch input order: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model":"nomic-embed-text",
			"embeddings":[[0.1,0.2,0.3],[0.4,0.5,0.6]],
			"prompt_eval_count":9
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("nomic-embed-text"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed(): %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("len(vectors) = %d, want 2", len(vectors))
	}
	if len(vectors[0]) != 3 || len(vectors[1]) != 3 {
		t.Fatalf("vector dims = [%d %d], want [3 3]", len(vectors[0]), len(vectors[1]))
	}
	if usage.Source != llm.UsageReported || usage.InputTokens != 9 || usage.TotalTokens != 9 {
		t.Fatalf("usage = %+v, want reported prompt=9 total=9", usage)
	}
}

func TestEmbed_Ollama_UnsupportedModel(t *testing.T) {
	m, err := New(WithModel("llama3.1:8b"), WithBaseURL("http://localhost:11434"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, _, err = m.Embed(context.Background(), []string{"hello"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("errors.Is(err, ErrCapabilityNotSupported) = false, err=%v", err)
	}
	if !strings.Contains(err.Error(), "ProviderInfo.Capabilities.Embeddings=false") {
		t.Fatalf("error = %q, want capability truth hint", err)
	}
}

func TestInfo_Ollama_UnsupportedModel(t *testing.T) {
	m, err := New(WithModel("llama2"), WithBaseURL("http://localhost:11434"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if info.Capabilities.Tools {
		t.Fatalf("Capabilities = %+v, want tools=false", info.Capabilities)
	}
}

func TestStream_Ollama_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(readBody(t, r), `"stream":true`) {
			t.Fatal("request body missing stream=true")
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-10T00:00:00Z","message":{"role":"assistant","content":"hel"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-10T00:00:00Z","message":{"role":"assistant","content":"lo"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-10T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":12,"eval_count":8}` + "\n"))
	}))
	defer server.Close()

	m, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
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
	if resp.Text != "hello" {
		t.Fatalf("Text = %q, want hello", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonStop)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Fatalf("Usage.Source = %q, want %q", resp.Usage.Source, llm.UsageReported)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 8 || resp.Usage.TotalTokens != 20 {
		t.Fatalf("Usage = %+v, want prompt=12 completion=8 total=20", resp.Usage)
	}
}

func TestStream_Ollama_CancelMidStream(t *testing.T) {
	release := make(chan struct{})
	var wroteFirst atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer missing Flusher")
		}
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-10T00:00:00Z","message":{"role":"assistant","content":"par"},"done":false}` + "\n"))
		flusher.Flush()
		wroteFirst.Store(true)
		<-release
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-10T00:00:00Z","message":{"role":"assistant","content":"tial"},"done":false}` + "\n"))
		flusher.Flush()
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	m, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	sr, err := m.Stream(ctx, llm.Request{
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
	if !wroteFirst.Load() {
		t.Fatal("first chunk not observed")
	}

	cancel()
	close(release)
	start := time.Now()
	_, err = sr.Next()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Next#2 error = %v, want context.Canceled", err)
	}
	if d := time.Since(start); d > 100*time.Millisecond {
		t.Fatalf("cancel latency = %s, want <= 100ms", d)
	}
}

func TestGenerate_Ollama_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/api/chat" {
			t.Fatalf("Path = %s, want /api/chat", r.URL.Path)
		}
		if !strings.Contains(readBody(t, r), `"stream":false`) {
			t.Fatal("request body missing stream=false")
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-10T00:00:00Z","message":{"role":"assistant","content":"hello from ollama"},"done":true,"done_reason":"stop","prompt_eval_count":12,"eval_count":8}` + "\n"))
	}))
	defer server.Close()

	m, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := m.Generate(context.Background(), llm.Request{
		SystemPrompt: "be concise",
		Messages:     []llm.Message{{Role: "user", Content: "say hello"}},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if resp.Text != "hello from ollama" {
		t.Fatalf("Text = %q", resp.Text)
	}
	if resp.Provider != "ollama" {
		t.Fatalf("Provider = %q", resp.Provider)
	}
	if resp.Model != "llama3.1:8b" {
		t.Fatalf("Model = %q", resp.Model)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q", resp.FinishReason)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Fatalf("Usage.Source = %q", resp.Usage.Source)
	}
}

func TestWithTools_Ollama_UnsupportedModel(t *testing.T) {
	m, err := New(WithModel("llama2"), WithBaseURL("http://localhost:11434"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.WithTools([]llm.Tool{{Name: "calculator", Parameters: []byte(`{"type":"object"}`)}})
	if err == nil {
		t.Fatal("WithTools() error = nil, want non-nil")
	}
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("errors.Is(err, ErrCapabilityNotSupported) = false, err=%v", err)
	}
	if !strings.Contains(err.Error(), "ProviderInfo.Capabilities.Tools=false") {
		t.Fatalf("error = %q, want capability truth hint", err)
	}
}

func TestWithTools_Ollama_Llama31NativeToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if !strings.Contains(body, `"tools":[`) {
			t.Fatalf("request body missing tools: %s", body)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-11T00:00:00Z","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_calc","function":{"index":0,"name":"calculator","arguments":{"expr":"2+2"}}}]},"done":true,"done_reason":"stop","prompt_eval_count":12,"eval_count":8}` + "\n"))
	}))
	defer server.Close()

	base, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{{Name: "calculator", Description: "calc tool", Parameters: []byte(`{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}`)}})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "2+2"}},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "call_calc" || resp.ToolCalls[0].Name != "calculator" || string(resp.ToolCalls[0].Arguments) != `{"expr":"2+2"}` {
		t.Fatalf("ToolCalls[0] = %+v, want calculator tool call", resp.ToolCalls[0])
	}
}

func TestWithTools_Ollama_QwenXMLFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"qwen3-coder:latest","created_at":"2026-05-11T00:00:00Z","message":{"role":"assistant","content":"Checking now.\n<tool_call>{\"name\":\"calculator\",\"arguments\":{\"expr\":\"2+2\"}}</tool_call>"},"done":true,"done_reason":"stop","prompt_eval_count":9,"eval_count":6}` + "\n"))
	}))
	defer server.Close()

	base, err := New(WithModel("qwen3-coder:latest"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{{Name: "calculator", Description: "calc tool", Parameters: []byte(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "2+2"}},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if resp.Text != "Checking now." {
		t.Fatalf("Text = %q, want Checking now.", resp.Text)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "calculator" || string(resp.ToolCalls[0].Arguments) != `{"expr":"2+2"}` {
		t.Fatalf("ToolCalls = %+v, want qwen xml parsed tool call", resp.ToolCalls)
	}
}

func TestWithTools_Ollama_Qwen25BareJSONFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(readBody(t, r), `"tools":[`) {
			t.Fatal("request body missing tools")
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"qwen2.5-coder:latest","created_at":"2026-05-12T00:00:00Z","message":{"role":"assistant","content":"{\"name\": \"calculator\", \"arguments\": {\"expr\": \"2+2\"}}"},"done":true,"done_reason":"stop","prompt_eval_count":11,"eval_count":5}` + "\n"))
	}))
	defer server.Close()

	base, err := New(WithModel("qwen2.5-coder:latest"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{{Name: "calculator", Description: "calc tool", Parameters: []byte(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "2+2"}},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if resp.Text != "" {
		t.Fatalf("Text = %q, want empty after bare-json tool extraction", resp.Text)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "calculator" {
		t.Fatalf("ToolCalls = %+v, want qwen bare json parsed tool call", resp.ToolCalls)
	}
	var args map[string]string
	if err := json.Unmarshal(resp.ToolCalls[0].Arguments, &args); err != nil {
		t.Fatalf("unmarshal arguments: %v", err)
	}
	if args["expr"] != "2+2" {
		t.Fatalf("arguments = %+v, want expr=2+2", args)
	}
}

func TestGenerate_Ollama_404ModelNotPulled(t *testing.T) {
	assertOllamaTypedError(t, 404, `{"error":"model not found, try pulling it first"}`, func(err error) bool {
		var target *llm.InvalidRequestError
		return errors.As(err, &target)
	})
}

func TestGenerate_Ollama_401AuthError(t *testing.T) {
	assertOllamaTypedError(t, 401, `{"error":"unauthorized"}`, func(err error) bool {
		var target *llm.AuthError
		return errors.As(err, &target)
	})
}

func TestGenerate_Ollama_500TransientError(t *testing.T) {
	assertOllamaTypedError(t, 500, `{"error":"server exploded"}`, func(err error) bool {
		var target *llm.TransientError
		return errors.As(err, &target)
	})
}

func TestGenerate_Ollama_NoDaemonTransientError(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()

	m, err := New(WithModel("llama3.1:8b"), WithBaseURL("http://"+addr))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("Generate() error = nil, want non-nil")
	}
	var target *llm.TransientError
	if !errors.As(err, &target) {
		t.Fatalf("errors.As(err, *llm.TransientError) = false, err=%T %v", err, err)
	}
}

func TestGenerate_Ollama_400InvalidRequestError(t *testing.T) {
	assertOllamaTypedError(t, 400, `{"error":"bad request"}`, func(err error) bool {
		var target *llm.InvalidRequestError
		return errors.As(err, &target)
	})
}

// P1-9: 429 must reclassify as *llm.RateLimitError (not InvalidRequestError)
// and lift the raw Retry-After header value into RetryAfter (string per contract;
// consumers parse). 4 cases cover numeric seconds, RFC 7231 HTTP-date, absent
// header, and a 5xx-with-Retry-After regression guard that keeps the
// TransientError path intact.

func TestGenerate_Ollama_429SetsRetryAfterSeconds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer server.Close()

	m, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
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
	if target.RetryAfter != "30" {
		t.Fatalf("RetryAfter = %q, want 30", target.RetryAfter)
	}
	if target.Provider != "ollama" {
		t.Fatalf("Provider = %q, want ollama", target.Provider)
	}
}

func TestGenerate_Ollama_429SetsRetryAfterHTTPDate(t *testing.T) {
	const when = "Wed, 21 Oct 2026 07:28:00 GMT"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", when)
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer server.Close()

	m, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
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
	if target.RetryAfter != when {
		t.Fatalf("RetryAfter = %q, want %q (raw RFC 7231 string per contract)", target.RetryAfter, when)
	}
}

func TestGenerate_Ollama_429EmptyRetryAfterWhenHeaderMissing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer server.Close()

	m, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
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
	if target.RetryAfter != "" {
		t.Fatalf("RetryAfter = %q, want empty string when header absent", target.RetryAfter)
	}
}

// Regression guard: 5xx with Retry-After must remain a TransientError —
// the Retry-After lift is gated on 429 only and must not change the
// classification of other status codes.
func TestGenerate_Ollama_503WithRetryAfterStaysTransient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "5")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"service unavailable"}`))
	}))
	defer server.Close()

	m, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err == nil {
		t.Fatal("Generate() error = nil, want non-nil")
	}
	var transient *llm.TransientError
	if !errors.As(err, &transient) {
		t.Fatalf("errors.As(err, *llm.TransientError) = false, err=%T %v", err, err)
	}
	var rl *llm.RateLimitError
	if errors.As(err, &rl) {
		t.Fatalf("503 must not lift to *RateLimitError, got %T %v", err, err)
	}
}

func assertOllamaTypedError(t *testing.T, status int, body string, match func(error) bool) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	m, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("Generate() error = nil, want non-nil")
	}
	if !match(err) {
		t.Fatalf("typed error assertion failed for status %d: %T %v", status, err, err)
	}
}

func readBody(t *testing.T, r *http.Request) string {
	t.Helper()
	defer r.Body.Close()
	b, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return string(b)
}

func TestStream_Ollama_NativeToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-10T00:00:00Z","message":{"role":"assistant","content":"checking "},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"llama3.1:8b","created_at":"2026-05-10T00:00:00Z","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_calc","function":{"index":0,"name":"calculator","arguments":{"expr":"2+2"}}}]},"done":true,"done_reason":"stop","prompt_eval_count":12,"eval_count":8}` + "\n"))
	}))
	defer server.Close()

	base, err := New(WithModel("llama3.1:8b"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{{Name: "calculator", Description: "calc", Parameters: []byte(`{"type":"object","properties":{"expr":{"type":"string"}}}`)}})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "2+2"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	wantKinds := []llm.StreamEventKind{
		llm.EventTextDelta,
		llm.EventToolCallStart,
		llm.EventToolCallArgsDelta,
		llm.EventToolCallEnd,
		llm.EventDone,
	}
	got := drainStreamEvents(t, sr, len(wantKinds))
	if len(got) != len(wantKinds) {
		t.Fatalf("event count = %d, want %d (events = %+v)", len(got), len(wantKinds), got)
	}
	for i, want := range wantKinds {
		if got[i].Kind != want {
			t.Fatalf("event[%d].Kind = %v, want %v (full = %+v)", i, got[i].Kind, want, got)
		}
	}
	if got[0].Text != "checking " {
		t.Fatalf("text delta = %q, want %q", got[0].Text, "checking ")
	}
	if got[1].ToolCall == nil || got[1].ToolCall.Index != 0 || got[1].ToolCall.ID != "call_calc" || got[1].ToolCall.Name != "calculator" {
		t.Fatalf("ToolCallStart = %+v, want index=0 id=call_calc name=calculator", got[1].ToolCall)
	}
	if got[2].ToolCall == nil || got[2].ToolCall.Index != 0 || got[2].ToolCall.ArgsDelta != `{"expr":"2+2"}` {
		t.Fatalf("ToolCallArgsDelta = %+v, want index=0 args=%q", got[2].ToolCall, `{"expr":"2+2"}`)
	}
	if got[3].ToolCall == nil || got[3].ToolCall.Index != 0 {
		t.Fatalf("ToolCallEnd = %+v, want index=0", got[3].ToolCall)
	}
	if got[4].FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("Done.FinishReason = %q, want %q", got[4].FinishReason, llm.FinishReasonToolCalls)
	}
	if got[4].Usage == nil || got[4].Usage.InputTokens != 12 || got[4].Usage.OutputTokens != 8 {
		t.Fatalf("Done.Usage = %+v, want prompt=12 completion=8", got[4].Usage)
	}
}

func TestStream_Ollama_QwenXMLContentToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"qwen3-coder:latest","created_at":"2026-05-11T00:00:00Z","message":{"role":"assistant","content":"Checking now.\n<tool_call>"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"qwen3-coder:latest","created_at":"2026-05-11T00:00:00Z","message":{"role":"assistant","content":"{\"name\":\"calculator\",\"arguments\":{\"expr\":\"2+2\"}}"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"qwen3-coder:latest","created_at":"2026-05-11T00:00:00Z","message":{"role":"assistant","content":"</tool_call>"},"done":true,"done_reason":"stop","prompt_eval_count":9,"eval_count":6}` + "\n"))
	}))
	defer server.Close()

	base, err := New(WithModel("qwen3-coder:latest"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{{Name: "calculator", Description: "calc", Parameters: []byte(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "2+2"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	events := drainAllEvents(t, sr)
	// Content-parsed strategies MUST NOT leak raw tool markers as TextDelta.
	for _, ev := range events {
		if ev.Kind == llm.EventTextDelta && (strings.Contains(ev.Text, "<tool_call>") || strings.Contains(ev.Text, "</tool_call>")) {
			t.Fatalf("text delta leaked tool markers: %q (full events = %+v)", ev.Text, events)
		}
	}
	// Expected: at least one TextDelta with the stripped prelude, then tool events, then Done.
	var sawStart, sawArgs, sawEnd, sawDone bool
	var argsCombined string
	var doneEvent llm.StreamEvent
	for _, ev := range events {
		switch ev.Kind {
		case llm.EventToolCallStart:
			if ev.ToolCall == nil || ev.ToolCall.Name != "calculator" {
				t.Fatalf("ToolCallStart = %+v, want name=calculator", ev.ToolCall)
			}
			sawStart = true
		case llm.EventToolCallArgsDelta:
			if ev.ToolCall == nil {
				t.Fatalf("ToolCallArgsDelta missing ToolCall")
			}
			argsCombined += ev.ToolCall.ArgsDelta
			sawArgs = true
		case llm.EventToolCallEnd:
			sawEnd = true
		case llm.EventDone:
			sawDone = true
			doneEvent = ev
		}
	}
	if !sawStart || !sawArgs || !sawEnd || !sawDone {
		t.Fatalf("missing K1 sequence: start=%v args=%v end=%v done=%v (events=%+v)", sawStart, sawArgs, sawEnd, sawDone, events)
	}
	if !strings.Contains(argsCombined, `"expr":"2+2"`) {
		t.Fatalf("ArgsDelta combined = %q, want to contain expr=2+2", argsCombined)
	}
	if doneEvent.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("Done.FinishReason = %q, want FinishReasonToolCalls", doneEvent.FinishReason)
	}
}

func TestStream_Ollama_NoToolsLeavesTextStreaming(t *testing.T) {
	// Regression guard: when no tools are bound, streaming must remain
	// chunk-by-chunk text deltas (no buffering even for content-parsed
	// strategies). This preserves TestStream_Ollama_CancelMidStream's
	// "first delta returns immediately" contract for tool-less callers.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"qwen3-coder:latest","created_at":"2026-05-11T00:00:00Z","message":{"role":"assistant","content":"hel"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"qwen3-coder:latest","created_at":"2026-05-11T00:00:00Z","message":{"role":"assistant","content":"lo"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"qwen3-coder:latest","created_at":"2026-05-11T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":2}` + "\n"))
	}))
	defer server.Close()

	m, err := New(WithModel("qwen3-coder:latest"), WithBaseURL(server.URL))
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
	if ev.Kind != llm.EventTextDelta || ev.Text != "hel" {
		t.Fatalf("Next#1 = %+v, want TextDelta 'hel' (no buffering when tools unbound)", ev)
	}
}

func drainStreamEvents(t *testing.T, sr llm.StreamReader, n int) []llm.StreamEvent {
	t.Helper()
	out := make([]llm.StreamEvent, 0, n)
	for i := 0; i < n; i++ {
		ev, err := sr.Next()
		if err != nil {
			t.Fatalf("Next#%d: %v", i+1, err)
		}
		out = append(out, ev)
	}
	return out
}

func drainAllEvents(t *testing.T, sr llm.StreamReader) []llm.StreamEvent {
	t.Helper()
	out := make([]llm.StreamEvent, 0, 8)
	for {
		ev, err := sr.Next()
		if err == io.EOF {
			return out
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		out = append(out, ev)
	}
}
