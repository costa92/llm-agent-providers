package kimi

import (
	"reflect"
	"testing"
)

func TestCapabilitiesForModel_Kimi(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		wantTools bool
	}{
		{"k2", "kimi-k2", true},
		{"k2-0905", "kimi-k2-0905", true},
		{"k2-thinking", "kimi-k2-thinking", true},
		{"k2.5", "kimi-k2.5", true},
		{"k2.6", "kimi-k2.6", true},
		{"moonshot-8k", "moonshot-v1-8k", true},
		{"moonshot-32k", "moonshot-v1-32k", true},
		{"moonshot-128k", "moonshot-v1-128k", true},
		{"unknown_fallback", "kimi-future-x", true},
		{"empty_fallback", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := capabilitiesForModel(tt.model)
			if got.Tools != tt.wantTools {
				t.Fatalf("Tools = %v, want %v", got.Tools, tt.wantTools)
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
