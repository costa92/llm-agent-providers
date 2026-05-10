package ollama

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

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
	if info.Capabilities.Tools || info.Capabilities.Embeddings || info.Capabilities.StructuredOutputs || info.Capabilities.PromptCaching {
		t.Fatalf("Capabilities = %+v, want all false", info.Capabilities)
	}
}

func TestStream_OllamaPhase1Stub(t *testing.T) {
	m, err := New(WithModel("llama3.1:8b"), WithBaseURL("http://localhost:11434"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Stream(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Stream() error = nil, want non-nil")
	}
	if err.Error() != "ollama: streaming not implemented in Phase 1; use Generate" {
		t.Fatalf("Stream() error = %q", err)
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
