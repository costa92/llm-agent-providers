package google

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestInfo_Google_VisionGate(t *testing.T) {
	vis, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !vis.Info().Capabilities.Vision {
		t.Fatalf("gemini-2.5-flash Vision = false, want true")
	}
	// Embedding model is not a chat model -> no vision.
	non, err := New(WithModel("gemini-embedding-001"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if non.Info().Capabilities.Vision {
		t.Fatalf("gemini-embedding-001 Vision = true, want false")
	}
}

// visionPart mirrors the genai Part wire shape for the fields we assert.
type visionPart struct {
	Text       string `json:"text"`
	InlineData *struct {
		Data     string `json:"data"`
		MIMEType string `json:"mimeType"`
	} `json:"inlineData"`
	FileData *struct {
		FileURI  string `json:"fileUri"`
		MIMEType string `json:"mimeType"`
	} `json:"fileData"`
}

func firstUserParts(t *testing.T, body string) []visionPart {
	t.Helper()
	var payload struct {
		Contents []struct {
			Role  string       `json:"role"`
			Parts []visionPart `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("unmarshal body: %v\nbody=%s", err, body)
	}
	if len(payload.Contents) != 1 || payload.Contents[0].Role != "user" {
		t.Fatalf("contents = %+v, want one user content: %s", payload.Contents, body)
	}
	return payload.Contents[0].Parts
}

func TestGenerate_Google_VisionInlineBytes(t *testing.T) {
	var body string
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"modelVersion":"gemini-2.5-flash"}`))
	})
	if _, err := g.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{
			Role:    "user",
			Content: "what is in this image?",
			Images:  []llm.MessageImage{{Bytes: []byte{0x89, 0x50, 0x4e, 0x47}, MimeType: "image/png"}},
		}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserParts(t, body)
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2 (text + image): %s", len(parts), body)
	}
	if parts[0].Text != "what is in this image?" {
		t.Fatalf("parts[0].Text = %q, want the prompt", parts[0].Text)
	}
	if parts[1].InlineData == nil {
		t.Fatalf("parts[1] missing inlineData: %s", body)
	}
	if parts[1].InlineData.MIMEType != "image/png" {
		t.Fatalf("inlineData.mimeType = %q, want image/png", parts[1].InlineData.MIMEType)
	}
	got, err := base64.StdEncoding.DecodeString(parts[1].InlineData.Data)
	if err != nil {
		t.Fatalf("inlineData.data base64 not well-formed: %v", err)
	}
	if !bytes.Equal(got, []byte{0x89, 0x50, 0x4e, 0x47}) {
		t.Fatalf("decoded bytes = %x, want 89504e47", got)
	}
}

func TestGenerate_Google_VisionBytesDefaultMimeType(t *testing.T) {
	var body string
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"modelVersion":"gemini-2.5-flash"}`))
	})
	if _, err := g.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Images: []llm.MessageImage{{Bytes: []byte{0x01}}}}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserParts(t, body)
	if len(parts) != 1 || parts[0].InlineData == nil {
		t.Fatalf("parts = %+v, want a single inlineData part (no empty text): %s", parts, body)
	}
	if parts[0].InlineData.MIMEType != "image/png" {
		t.Fatalf("inlineData.mimeType = %q, want default image/png", parts[0].InlineData.MIMEType)
	}
}

func TestGenerate_Google_VisionHTTPURL_FileData(t *testing.T) {
	var body string
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"modelVersion":"gemini-2.5-flash"}`))
	})
	const url = "gs://bucket/cat.png"
	if _, err := g.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "describe", Images: []llm.MessageImage{{URL: url, MimeType: "image/png"}}}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserParts(t, body)
	if len(parts) != 2 || parts[1].FileData == nil {
		t.Fatalf("parts = %+v, want text + fileData: %s", parts, body)
	}
	if parts[1].FileData.FileURI != url {
		t.Fatalf("fileData.fileUri = %q, want %q", parts[1].FileData.FileURI, url)
	}
	if parts[1].FileData.MIMEType != "image/png" {
		t.Fatalf("fileData.mimeType = %q, want image/png", parts[1].FileData.MIMEType)
	}
}

func TestGenerate_Google_VisionDataURI_InlineData(t *testing.T) {
	var body string
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"modelVersion":"gemini-2.5-flash"}`))
	})
	const dataURI = "data:image/jpeg;base64,QUJD"
	if _, err := g.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Images: []llm.MessageImage{{URL: dataURI}}}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserParts(t, body)
	if len(parts) != 1 || parts[0].InlineData == nil {
		t.Fatalf("parts = %+v, want single inlineData part: %s", parts, body)
	}
	if parts[0].InlineData.MIMEType != "image/jpeg" {
		t.Fatalf("inlineData.mimeType = %q, want image/jpeg", parts[0].InlineData.MIMEType)
	}
	got, err := base64.StdEncoding.DecodeString(parts[0].InlineData.Data)
	if err != nil || string(got) != "ABC" {
		t.Fatalf("inlineData.data = %q decoded=%q err=%v, want ABC", parts[0].InlineData.Data, got, err)
	}
}

func TestGenerate_Google_TextOnly_StaysSingleTextPart(t *testing.T) {
	var body string
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"modelVersion":"gemini-2.5-flash"}`))
	})
	if _, err := g.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "plain text"}},
	}); err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	parts := firstUserParts(t, body)
	if len(parts) != 1 || parts[0].Text != "plain text" || parts[0].InlineData != nil || parts[0].FileData != nil {
		t.Fatalf("parts = %+v, want single text part 'plain text' (no regression): %s", parts, body)
	}
}
