package volcengine

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

func TestCapabilitiesForModel(t *testing.T) {
	cases := []struct {
		model          string
		wantTools      bool
		wantImage      bool
		wantEmbeddings bool
	}{
		{"doubao-1-5-pro-32k-250115", true, false, false},
		{"doubao-seedream-4-5-251128", false, true, false},
		{"doubao-embedding-text-240715", false, false, true},
		{"doubao-embedding-large-text-240915", false, false, true},
	}
	for _, c := range cases {
		caps := capabilitiesForModel(c.model)
		if caps.Tools != c.wantTools || caps.ImageGeneration != c.wantImage || caps.Embeddings != c.wantEmbeddings {
			t.Fatalf("capabilitiesForModel(%q) = %+v, want tools=%v image=%v embed=%v",
				c.model, caps, c.wantTools, c.wantImage, c.wantEmbeddings)
		}
	}
}

func TestEmbedDimensions_Table(t *testing.T) {
	cases := map[string]int{
		"doubao-embedding-text-240715":       2560,
		"doubao-embedding-large-text-240915": 4096,
		"doubao-1-5-pro-32k-250115":          0,
	}
	for model, want := range cases {
		v := &Volcengine{info: providerInfo(model, capabilitiesForModel(model)), embedDimensions: defaultEmbedDimensions(model)}
		if got := v.EmbedDimensions(); got != want {
			t.Fatalf("EmbedDimensions(%q) = %d, want %d", model, got, want)
		}
	}
}
