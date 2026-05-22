package deepseek

import (
	"reflect"
	"testing"
)

func TestCapabilitiesForModel_DeepSeek(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		wantTools bool
	}{
		{"chat", "deepseek-chat", true},
		{"coder", "deepseek-coder", true},
		{"reasoner", "deepseek-reasoner", true},
		{"unknown_fallback", "deepseek-future-x", true},
		{"empty_fallback", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := capabilitiesForModel(tt.model)
			if got.Tools != tt.wantTools {
				t.Fatalf("Tools = %v, want %v", got.Tools, tt.wantTools)
			}
			if got.Embeddings || got.StructuredOutputs || got.PromptCaching {
				t.Fatalf("non-tool caps must remain false: %+v", got)
			}
		})
	}
}

// Lock K2: New() must delegate to capabilitiesForModel(cfg.model)
func TestInfo_DeepSeek_ReflectsCapabilitiesForModel(t *testing.T) {
	m, err := New(WithModel("deepseek-reasoner"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	want := capabilitiesForModel("deepseek-reasoner")
	if !reflect.DeepEqual(m.Info().Capabilities, want) {
		t.Fatalf("Info().Capabilities = %+v, want %+v", m.Info().Capabilities, want)
	}
}
