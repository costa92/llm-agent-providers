package openai

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

func TestWithExtraHeaders_OpenAI_AppliedToRequests(t *testing.T) {
	var gotHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-My-Gateway")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"created":1710000000,
			"data":[{"b64_json":"aGVsbG8="}]
		}`))
	}))
	defer server.Close()

	o, err := New(
		WithModel("gpt-image-1"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithExtraHeaders(map[string]string{"X-My-Gateway": "route-42"}),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := o.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "a cat"}); err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if gotHeader != "route-42" {
		t.Fatalf("X-My-Gateway = %q, want route-42", gotHeader)
	}
}

func TestInfo_OpenAI_ImageModel(t *testing.T) {
	o, err := New(WithModel("gpt-image-1"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	caps := o.Info().Capabilities
	if !caps.ImageGeneration {
		t.Fatalf("Capabilities = %+v, want ImageGeneration=true", caps)
	}
	if caps.Embeddings {
		t.Fatalf("image model must not report Embeddings: %+v", caps)
	}
}

func TestInfo_OpenAI_ChatModelNoImage(t *testing.T) {
	o, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if o.Info().Capabilities.ImageGeneration {
		t.Fatalf("gpt-4o-mini must not report ImageGeneration")
	}
}

func TestIsImageModel_OpenAI(t *testing.T) {
	imageModels := []string{"gpt-image-1", "gpt-image-2", "dall-e-2", "dall-e-3"}
	for _, m := range imageModels {
		if !isImageModel(m) {
			t.Errorf("isImageModel(%q) = false, want true", m)
		}
	}
	chatModels := []string{"gpt-4o-mini", "text-embedding-3-small", ""}
	for _, m := range chatModels {
		if isImageModel(m) {
			t.Errorf("isImageModel(%q) = true, want false", m)
		}
	}
}

func TestGenerateImage_OpenAI_B64(t *testing.T) {
	var gotPath, gotBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		// "aGVsbG8=" is base64("hello").
		_, _ = w.Write([]byte(`{
			"created":1710000000,
			"data":[{"b64_json":"aGVsbG8=","revised_prompt":"a fluffy cat"}],
			"usage":{"input_tokens":5,"output_tokens":10,"total_tokens":15}
		}`))
	}))
	defer server.Close()

	o, err := New(WithModel("gpt-image-1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := o.GenerateImage(context.Background(), llm.ImageRequest{
		Prompt:  "a cat",
		N:       1,
		Size:    "1024x1024",
		Quality: "high",
		Format:  "png",
	})
	if err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if gotPath != "/images/generations" {
		t.Fatalf("path = %q, want /images/generations", gotPath)
	}
	if !strings.Contains(gotBody, `"prompt":"a cat"`) {
		t.Fatalf("body missing prompt: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"model":"gpt-image-1"`) {
		t.Fatalf("body missing model: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"size":"1024x1024"`) {
		t.Fatalf("body missing size: %s", gotBody)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("len(Images) = %d, want 1", len(resp.Images))
	}
	if string(resp.Images[0].Bytes) != "hello" {
		t.Fatalf("Bytes = %q, want decoded \"hello\"", resp.Images[0].Bytes)
	}
	if resp.Images[0].URL != "" {
		t.Fatalf("URL = %q, want empty when bytes present", resp.Images[0].URL)
	}
	if resp.Images[0].RevisedPrompt != "a fluffy cat" {
		t.Fatalf("RevisedPrompt = %q, want a fluffy cat", resp.Images[0].RevisedPrompt)
	}
	if resp.Provider != "openai" || resp.Model != "gpt-image-1" {
		t.Fatalf("provider/model = %q/%q", resp.Provider, resp.Model)
	}
}

func TestGenerateImage_OpenAI_URL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"created":1710000000,
			"data":[{"url":"https://img.example/abc.png"}]
		}`))
	}))
	defer server.Close()

	o, err := New(WithModel("dall-e-3"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := o.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "a dog"})
	if err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("len(Images) = %d, want 1", len(resp.Images))
	}
	if resp.Images[0].URL != "https://img.example/abc.png" {
		t.Fatalf("URL = %q", resp.Images[0].URL)
	}
	if len(resp.Images[0].Bytes) != 0 {
		t.Fatalf("Bytes = %v, want empty when only URL present", resp.Images[0].Bytes)
	}
}

func TestGenerateImage_OpenAI_NotSupported(t *testing.T) {
	o, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = o.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("err = %v, want ErrCapabilityNotSupported", err)
	}
}
