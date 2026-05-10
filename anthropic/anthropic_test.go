package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
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
		t.Fatalf("New() error = %q, want WithModel is required", err)
	}
}

func TestInfo_Anthropic(t *testing.T) {
	m, err := New(WithModel("claude-3-5-haiku-20241022"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if info.Provider != "anthropic" {
		t.Fatalf("Provider = %q, want anthropic", info.Provider)
	}
	if info.Model != "claude-3-5-haiku-20241022" {
		t.Fatalf("Model = %q", info.Model)
	}
	if info.Capabilities.Tools || info.Capabilities.Embeddings || info.Capabilities.StructuredOutputs || info.Capabilities.PromptCaching {
		t.Fatalf("Capabilities = %+v, want all false", info.Capabilities)
	}
}

func TestStream_AnthropicPhase1Stub(t *testing.T) {
	m, err := New(WithModel("claude-3-5-haiku-20241022"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Stream(context.Background(), llm.Request{})
	if err == nil {
		t.Fatal("Stream() error = nil, want non-nil")
	}
	if err.Error() != "anthropic: streaming not implemented in Phase 1; use Generate" {
		t.Fatalf("Stream() error = %q", err)
	}
}

func TestGenerate_Anthropic_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("Path = %s, want /v1/messages", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_123",
			"type":"message",
			"role":"assistant",
			"model":"claude-3-5-haiku-20241022",
			"content":[{"type":"text","text":"hello from claude"}],
			"stop_reason":"end_turn",
			"stop_sequence":null,
			"usage":{"input_tokens":11,"output_tokens":7}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("claude-3-5-haiku-20241022"),
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
	if resp.Text != "hello from claude" {
		t.Fatalf("Text = %q", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonStop)
	}
	if resp.Provider != "anthropic" {
		t.Fatalf("Provider = %q, want anthropic", resp.Provider)
	}
	if resp.Model != "claude-3-5-haiku-20241022" {
		t.Fatalf("Model = %q", resp.Model)
	}
}

func TestGenerate_Anthropic_SystemTopLevel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}

		if _, ok := payload["system"]; !ok {
			t.Fatalf("payload missing top-level system: %s", string(body))
		}

		msgs, ok := payload["messages"].([]any)
		if !ok {
			t.Fatalf("messages missing or wrong type: %T", payload["messages"])
		}
		for _, raw := range msgs {
			msg, ok := raw.(map[string]any)
			if !ok {
				t.Fatalf("message wrong type: %T", raw)
			}
			if msg["role"] == "system" {
				t.Fatalf("system role leaked into messages: %s", string(body))
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_123",
			"type":"message",
			"role":"assistant",
			"model":"claude-3-5-haiku-20241022",
			"content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"stop_sequence":null,
			"usage":{"input_tokens":11,"output_tokens":1}
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("claude-3-5-haiku-20241022"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{
		SystemPrompt: "system A",
		Messages: []llm.Message{
			{Role: "system", Content: "system B"},
			{Role: "user", Content: "hello"},
		},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
}

func TestGenerate_Anthropic_401AuthError(t *testing.T) {
	assertAnthropicTypedError(t, 401, `{"type":"error","error":{"type":"authentication_error","message":"bad key"}}`, func(err error) bool {
		var target *llm.AuthError
		return errors.As(err, &target)
	})
}

func TestGenerate_Anthropic_400InvalidRequestError(t *testing.T) {
	assertAnthropicTypedError(t, 400, `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`, func(err error) bool {
		var target *llm.InvalidRequestError
		return errors.As(err, &target)
	})
}

func TestGenerate_Anthropic_429RateLimitError(t *testing.T) {
	assertAnthropicTypedError(t, 429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`, func(err error) bool {
		var target *llm.RateLimitError
		return errors.As(err, &target)
	})
}

func TestGenerate_Anthropic_529OverloadedIsRateLimit(t *testing.T) {
	assertAnthropicTypedError(t, 529, `{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}`, func(err error) bool {
		var target *llm.RateLimitError
		return errors.As(err, &target)
	})
}

func TestGenerate_Anthropic_500TransientError(t *testing.T) {
	assertAnthropicTypedError(t, 500, `{"type":"error","error":{"type":"api_error","message":"server error"}}`, func(err error) bool {
		var target *llm.TransientError
		return errors.As(err, &target)
	})
}

func assertAnthropicTypedError(t *testing.T, status int, body string, match func(error) bool) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	m, err := New(WithModel("claude-3-5-haiku-20241022"), WithAPIKey("test-key"), WithBaseURL(server.URL))
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
