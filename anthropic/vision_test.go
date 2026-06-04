package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestIsVisionModel_Anthropic(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"claude-3-5-sonnet-20241022", true},
		{"claude-3-7-sonnet-20250219", true},
		{"claude-3-5-haiku-20241022", true},
		{"claude-3-opus-20240229", true},
		{"claude-sonnet-4-20250514", true},
		{"claude-opus-4-20250514", true},
		{"claude-haiku-4-5", true},
		{"claude-2.1", false},
		{"claude-2.0", false},
		{"claude-instant-1.2", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isVisionModel(tt.model); got != tt.want {
			t.Errorf("isVisionModel(%q) = %v, want %v", tt.model, got, tt.want)
		}
	}
}

func TestInfo_Anthropic_VisionGate(t *testing.T) {
	vis, err := New(WithModel("claude-3-5-sonnet-20241022"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !vis.Info().Capabilities.Vision {
		t.Fatalf("claude-3-5-sonnet Vision = false, want true")
	}
	non, err := New(WithModel("claude-2.1"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if non.Info().Capabilities.Vision {
		t.Fatalf("claude-2.1 Vision = true, want false")
	}
}

func visionCaptureServer(t *testing.T, capture *string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		*capture = string(body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_img",
			"type":"message",
			"role":"assistant",
			"model":"claude-3-5-sonnet-20241022",
			"content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	}))
}

// imageContentParts unmarshals the first user message's content blocks.
type imageContentParts []struct {
	Type   string `json:"type"`
	Text   string `json:"text"`
	Source struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
		URL       string `json:"url"`
	} `json:"source"`
}

func firstUserContent(t *testing.T, body string) imageContentParts {
	t.Helper()
	var payload struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("unmarshal body: %v\nbody=%s", err, body)
	}
	if len(payload.Messages) != 1 || payload.Messages[0].Role != "user" {
		t.Fatalf("messages = %+v, want one user message: %s", payload.Messages, body)
	}
	var parts imageContentParts
	if err := json.Unmarshal(payload.Messages[0].Content, &parts); err != nil {
		t.Fatalf("content is not a block array: %v\ncontent=%s", err, payload.Messages[0].Content)
	}
	return parts
}

func TestGenerate_Anthropic_VisionInlineBytes_Base64Block(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("claude-3-5-sonnet-20241022"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{
			Role:    "user",
			Content: "what is in this image?",
			Images:  []llm.MessageImage{{Bytes: []byte{0x89, 0x50, 0x4e, 0x47}, MimeType: "image/png"}},
		}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserContent(t, body)
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2 (text + image): %s", len(parts), body)
	}
	if parts[0].Type != "text" || parts[0].Text != "what is in this image?" {
		t.Fatalf("parts[0] = %+v, want text block", parts[0])
	}
	if parts[1].Type != "image" {
		t.Fatalf("parts[1].Type = %q, want image", parts[1].Type)
	}
	if parts[1].Source.Type != "base64" || parts[1].Source.MediaType != "image/png" {
		t.Fatalf("parts[1].source = %+v, want base64/image/png", parts[1].Source)
	}
	got, err := base64.StdEncoding.DecodeString(parts[1].Source.Data)
	if err != nil {
		t.Fatalf("source.data base64 not well-formed: %v", err)
	}
	if !bytes.Equal(got, []byte{0x89, 0x50, 0x4e, 0x47}) {
		t.Fatalf("decoded bytes = %x, want 89504e47", got)
	}
}

func TestGenerate_Anthropic_VisionBytesDefaultMimeType(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("claude-3-5-sonnet-20241022"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Images: []llm.MessageImage{{Bytes: []byte{0x01}}}}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserContent(t, body)
	if len(parts) != 1 || parts[0].Type != "image" {
		t.Fatalf("parts = %+v, want a single image block (no empty text): %s", parts, body)
	}
	if parts[0].Source.MediaType != "image/png" {
		t.Fatalf("source.media_type = %q, want default image/png", parts[0].Source.MediaType)
	}
}

func TestGenerate_Anthropic_VisionURLBlock(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("claude-3-5-sonnet-20241022"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	const url = "https://example.com/cat.png"
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "describe", Images: []llm.MessageImage{{URL: url}}}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserContent(t, body)
	if len(parts) != 2 || parts[1].Type != "image" {
		t.Fatalf("parts = %+v, want text + image: %s", parts, body)
	}
	if parts[1].Source.Type != "url" || parts[1].Source.URL != url {
		t.Fatalf("parts[1].source = %+v, want url source %q", parts[1].Source, url)
	}
}

func TestGenerate_Anthropic_VisionDataURI_DecodedToBase64Block(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("claude-3-5-sonnet-20241022"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	const dataURI = "data:image/jpeg;base64,QUJD"
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Images: []llm.MessageImage{{URL: dataURI}}}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserContent(t, body)
	if len(parts) != 1 || parts[0].Type != "image" {
		t.Fatalf("parts = %+v, want single image block: %s", parts, body)
	}
	if parts[0].Source.Type != "base64" || parts[0].Source.MediaType != "image/jpeg" || parts[0].Source.Data != "QUJD" {
		t.Fatalf("parts[0].source = %+v, want base64/image/jpeg/QUJD", parts[0].Source)
	}
}

func TestGenerate_Anthropic_TextOnly_StaysSingleTextBlock(t *testing.T) {
	var body string
	server := visionCaptureServer(t, &body)
	defer server.Close()

	m, err := New(WithModel("claude-3-5-sonnet-20241022"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "plain text"}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserContent(t, body)
	if len(parts) != 1 || parts[0].Type != "text" || parts[0].Text != "plain text" {
		t.Fatalf("parts = %+v, want single text block 'plain text' (no regression): %s", parts, body)
	}
}
