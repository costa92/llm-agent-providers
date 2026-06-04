package google

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestGenerateImage_GeminiInline(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash-image", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":generateContent") {
			t.Errorf("path = %s, want :generateContent", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		s := string(body)
		if !strings.Contains(s, `"responseModalities"`) || !strings.Contains(s, `"TEXT"`) || !strings.Contains(s, `"IMAGE"`) {
			t.Errorf("body missing responseModalities [TEXT,IMAGE]: %s", s)
		}
		w.Header().Set("Content-Type", "application/json")
		// base64("PNGDATA") = UE5HREFUQQ== ; include a text part to confirm it is dropped.
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[
				{"text":"here you go"},
				{"inlineData":{"mimeType":"image/png","data":"UE5HREFUQQ=="}}
			]},"finishReason":"STOP","index":0}]
		}`))
	})

	resp, err := g.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "a fox"})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("Images len = %d, want 1 (text part dropped)", len(resp.Images))
	}
	if string(resp.Images[0].Bytes) != "PNGDATA" {
		t.Errorf("Bytes = %q, want PNGDATA", resp.Images[0].Bytes)
	}
	if resp.Images[0].MimeType != "image/png" {
		t.Errorf("MimeType = %q, want image/png", resp.Images[0].MimeType)
	}
	if resp.Provider != "google" {
		t.Errorf("Provider = %q, want google", resp.Provider)
	}
}

func TestGenerateImage_Imagen(t *testing.T) {
	g := newTestServer(t, "imagen-4.0-generate-001", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":predict") {
			t.Errorf("path = %s, want :predict", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		// base64("IMG1")=SU1HMQ==  base64("IMG2")=SU1HMg==
		_, _ = w.Write([]byte(`{
			"predictions":[
				{"bytesBase64Encoded":"SU1HMQ==","mimeType":"image/png"},
				{"bytesBase64Encoded":"SU1HMg==","mimeType":"image/png"}
			]
		}`))
	})

	resp, err := g.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "two cats", N: 2})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if len(resp.Images) != 2 {
		t.Fatalf("Images len = %d, want 2", len(resp.Images))
	}
	if string(resp.Images[0].Bytes) != "IMG1" || string(resp.Images[1].Bytes) != "IMG2" {
		t.Errorf("Bytes = %q/%q, want IMG1/IMG2", resp.Images[0].Bytes, resp.Images[1].Bytes)
	}
}

func TestGenerateImage_NonImageModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be called for a non-image model")
	}))
	t.Cleanup(server.Close)
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = g.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("GenerateImage on chat model = %v, want ErrCapabilityNotSupported", err)
	}
}
