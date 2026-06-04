package kimi

import (
	"reflect"
	"testing"
)

func TestCapabilitiesForModel_Kimi(t *testing.T) {
	tests := []struct {
		name       string
		model      string
		wantTools  bool
		wantVision bool
	}{
		{"k2", "kimi-k2", true, false},
		{"k2-0905", "kimi-k2-0905", true, false},
		{"k2-thinking", "kimi-k2-thinking", true, false},
		{"k2.5", "kimi-k2.5", true, true},
		{"k2.6", "kimi-k2.6", true, true},
		{"moonshot-8k", "moonshot-v1-8k", true, false},
		{"moonshot-32k", "moonshot-v1-32k", true, false},
		{"moonshot-128k", "moonshot-v1-128k", true, false},
		{"moonshot-8k-vision", "moonshot-v1-8k-vision-preview", true, true},
		{"moonshot-32k-vision", "moonshot-v1-32k-vision-preview", true, true},
		{"moonshot-128k-vision", "moonshot-v1-128k-vision-preview", true, true},
		{"unknown_fallback", "kimi-future-x", true, false},
		{"empty_fallback", "", true, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := capabilitiesForModel(tt.model)
			if got.Tools != tt.wantTools {
				t.Fatalf("Tools = %v, want %v", got.Tools, tt.wantTools)
			}
			if got.Vision != tt.wantVision {
				t.Fatalf("Vision = %v, want %v", got.Vision, tt.wantVision)
			}
			if got.ImageGeneration || got.Embeddings || got.StructuredOutputs || got.PromptCaching {
				t.Fatalf("non-tool caps must remain false: %+v", got)
			}
		})
	}
}

// Lock K2: New() must delegate to capabilitiesForModel(cfg.model)
func TestInfo_Kimi_ReflectsCapabilitiesForModel(t *testing.T) {
	m, err := New(WithModel("kimi-k2-thinking"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	want := capabilitiesForModel("kimi-k2-thinking")
	if !reflect.DeepEqual(m.Info().Capabilities, want) {
		t.Fatalf("Info().Capabilities = %+v, want %+v", m.Info().Capabilities, want)
	}
}
