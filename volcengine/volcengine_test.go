package volcengine

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestGenerate_Volcengine_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("Path = %s, want /chat/completions", r.URL.Path)
		}
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"model":"doubao-1-5-pro-32k-250115"`) {
			t.Fatalf("request body missing model: %s", body)
		}
		if !strings.Contains(body, `"role":"system"`) || !strings.Contains(body, `be concise`) {
			t.Fatalf("request body missing system message: %s", body)
		}
		if !strings.Contains(body, `say hello`) {
			t.Fatalf("request body missing user message: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"cmpl_123",
			"object":"chat.completion",
			"created":1710000000,
			"model":"doubao-1-5-pro-32k-250115",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello world"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("doubao-1-5-pro-32k-250115"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
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
	if resp.Text != "hello world" {
		t.Fatalf("Text = %q, want hello world", resp.Text)
	}
	if resp.Provider != "volcengine" {
		t.Fatalf("Provider = %q, want volcengine", resp.Provider)
	}
	if resp.Model != "doubao-1-5-pro-32k-250115" {
		t.Fatalf("Model = %q, want doubao-1-5-pro-32k-250115", resp.Model)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.Source != llm.UsageReported || resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Fatalf("Usage = %+v, want reported 11/7/18", resp.Usage)
	}
}
