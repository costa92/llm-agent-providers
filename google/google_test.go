package google

import (
	"strings"
	"testing"
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

func TestInfo_ChatModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := g.Info()
	if info.Provider != "google" {
		t.Fatalf("Provider = %q, want google", info.Provider)
	}
	if info.Model != "gemini-2.5-flash" {
		t.Fatalf("Model = %q, want gemini-2.5-flash", info.Model)
	}
	if !info.Capabilities.Tools {
		t.Errorf("Tools = false, want true for chat model")
	}
	if info.Capabilities.ImageGeneration || info.Capabilities.Embeddings {
		t.Errorf("Capabilities = %+v, want image_generation=false embeddings=false", info.Capabilities)
	}
}

func TestInfo_GeminiImageModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash-image"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !g.Info().Capabilities.ImageGeneration {
		t.Errorf("ImageGeneration = false, want true for gemini-*-image")
	}
}

func TestInfo_ImagenModel(t *testing.T) {
	g, err := New(WithModel("imagen-4.0-generate-001"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	caps := g.Info().Capabilities
	if !caps.ImageGeneration {
		t.Errorf("ImageGeneration = false, want true for imagen-*")
	}
	if caps.Tools {
		t.Errorf("Tools = true, want false for imagen model")
	}
}

func TestInfo_EmbedModel(t *testing.T) {
	g, err := New(WithModel("gemini-embedding-001"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !g.Info().Capabilities.Embeddings {
		t.Errorf("Embeddings = false, want true for embedding model")
	}
	if got := g.EmbedDimensions(); got != 3072 {
		t.Errorf("EmbedDimensions() = %d, want 3072", got)
	}
}

func TestEmbedDimensions_TextEmbedding004(t *testing.T) {
	g, err := New(WithModel("text-embedding-004"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 768 {
		t.Errorf("EmbedDimensions() = %d, want 768", got)
	}
}

func TestEmbedDimensions_WithDimensionsOverride(t *testing.T) {
	g, err := New(WithModel("gemini-embedding-001"), WithAPIKey("test-key"), WithDimensions(1536))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 1536 {
		t.Errorf("EmbedDimensions() = %d, want 1536 (overridden)", got)
	}
}

func TestEmbedDimensions_NonEmbedModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 0 {
		t.Errorf("EmbedDimensions() = %d, want 0 for non-embed model", got)
	}
}
