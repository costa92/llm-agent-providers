package minimax

import (
	"reflect"
	"testing"
)

func TestCapabilitiesForModel_MiniMax(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		wantTools bool
	}{
		{"M1", "MiniMax-M1", true},
		{"unknown_abab_fallback", "abab6.5-chat", true},
		{"unknown_vision", "MiniMax-Vision", true},
		{"empty", "", true},
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

func TestInfo_MiniMax_ReflectsCapabilitiesForModel(t *testing.T) {
	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	want := capabilitiesForModel("MiniMax-M1")
	if !reflect.DeepEqual(m.Info().Capabilities, want) {
		t.Fatalf("Info().Capabilities = %+v, want %+v", m.Info().Capabilities, want)
	}
}
