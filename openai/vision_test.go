package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestIsVisionModel_OpenAI(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gpt-4o", true},
		{"gpt-4o-2024-08-06", true},
		{"gpt-4o-mini", true},
		{"chatgpt-4o-latest", true},
		{"gpt-4-turbo", true},
		{"gpt-4-turbo-2024-04-09", true},
		{"gpt-4.1", true},
		{"gpt-4.1-mini", true},
		{"gpt-4.1-nano", true},
		{"gpt-4.5-preview", true},
		{"o1", true},
		{"o1-2024-12-17", true},
		{"o3", true},
		{"o4-mini", true},
		{"gpt-3.5-turbo", false},
		{"gpt-4", false},
		{"text-embedding-3-small", false},
		{"dall-e-3", false},
		{"gpt-image-1", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isVisionModel(tt.model); got != tt.want {
			t.Errorf("isVisionModel(%q) = %v, want %v", tt.model, got, tt.want)
		}
	}
}

func TestInfo_OpenAI_VisionGate(t *testing.T) {
	vis, err := New(WithModel("gpt-4o"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !vis.Info().Capabilities.Vision {
		t.Fatalf("gpt-4o Vision = false, want true")
	}
	non, err := New(WithModel("gpt-3.5-turbo"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if non.Info().Capabilities.Vision {
		t.Fatalf("gpt-3.5-turbo Vision = true, want false")
	}
}

// visionCaptureServer records the last request body so multimodal mapping can
// be asserted on the wire.
func visionCaptureServer(t *testing.T, capture *string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		*capture = string(body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl_img",
			"object":"chat.completion",
			"created":1710000000,
			"model":"gpt-4o",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop","logprobs":null}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
}

func TestGenerate_OpenAI_VisionInlineBytes_MultimodalContent(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("gpt-4o"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{
			Role:    "user",
			Content: "what is in this image?",
			Images: []llm.MessageImage{{
				Bytes:    []byte{0x89, 0x50, 0x4e, 0x47},
				MimeType: "image/png",
				Detail:   "high",
			}},
		}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}

	var payload struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("unmarshal body: %v\nbody=%s", err, body)
	}
	if len(payload.Messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1: %s", len(payload.Messages), body)
	}
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL struct {
			URL    string `json:"url"`
			Detail string `json:"detail"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(payload.Messages[0].Content, &parts); err != nil {
		t.Fatalf("content is not an array (multimodal): %v\ncontent=%s", err, payload.Messages[0].Content)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2: %s", len(parts), payload.Messages[0].Content)
	}
	if parts[0].Type != "text" || parts[0].Text != "what is in this image?" {
		t.Fatalf("parts[0] = %+v, want text part", parts[0])
	}
	if parts[1].Type != "image_url" {
		t.Fatalf("parts[1].Type = %q, want image_url", parts[1].Type)
	}
	if parts[1].ImageURL.Detail != "high" {
		t.Fatalf("parts[1].image_url.detail = %q, want high", parts[1].ImageURL.Detail)
	}
	const prefix = "data:image/png;base64,"
	if !strings.HasPrefix(parts[1].ImageURL.URL, prefix) {
		t.Fatalf("image_url.url = %q, want prefix %q", parts[1].ImageURL.URL, prefix)
	}
	got, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(parts[1].ImageURL.URL, prefix))
	if err != nil {
		t.Fatalf("data URI base64 not well-formed: %v", err)
	}
	if !bytes.Equal(got, []byte{0x89, 0x50, 0x4e, 0x47}) {
		t.Fatalf("decoded bytes = %x, want 89504e47", got)
	}
}

func TestGenerate_OpenAI_VisionBytesDefaultMimeType(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("gpt-4o"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Images: []llm.MessageImage{{Bytes: []byte{0x01}}}}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if !strings.Contains(body, "data:image/png;base64,") {
		t.Fatalf("body missing default image/png data URI: %s", body)
	}
}

func TestGenerate_OpenAI_VisionURLPassthrough(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("gpt-4o"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	const url = "https://example.com/cat.png"
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "describe", Images: []llm.MessageImage{{URL: url}}}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if !strings.Contains(body, url) {
		t.Fatalf("body missing passed-through URL %q: %s", url, body)
	}
}

func TestGenerate_OpenAI_TextOnly_StaysPlainStringContent(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("gpt-4o"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "plain text"}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	var payload struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if len(payload.Messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(payload.Messages))
	}
	var s string
	if err := json.Unmarshal(payload.Messages[0].Content, &s); err != nil {
		t.Fatalf("text-only content is not a plain string (regression): %v\ncontent=%s", err, payload.Messages[0].Content)
	}
	if s != "plain text" {
		t.Fatalf("content = %q, want plain text", s)
	}
}
