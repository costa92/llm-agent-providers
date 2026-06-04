package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
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
