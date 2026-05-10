//go:build ollama_live

package contract

import (
	"context"
	"os"
	"testing"

	"github.com/costa92/llm-agent-providers/ollama"
	tcollama "github.com/testcontainers/testcontainers-go/modules/ollama"
)

// TestGenerate_Ollama_Live runs the Phase 1 generate conformance check
// against a real Ollama testcontainer. It is build-tagged so normal PR
// CI never pays the cold-pull cost.
func TestGenerate_Ollama_Live(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping ollama-live test in -short mode")
	}

	ctx := context.Background()
	image := getenv("OLLAMA_TC_IMAGE", "ollama/ollama:0.5.7")
	model := getenv("OLLAMA_TC_MODEL", "llama3.1:8b-instruct-q4_K_M")

	container, err := tcollama.Run(ctx, image)
	if err != nil {
		t.Fatalf("tcollama.Run(%s): %v", image, err)
	}
	t.Cleanup(func() { _ = container.Terminate(ctx) })

	// The module API does not offer a startup-time model preload hook; the
	// documented path is to exec `ollama pull` inside the live container.
	if _, _, err := container.Exec(ctx, []string{"ollama", "pull", model}); err != nil {
		t.Fatalf("ollama pull %s: %v", model, err)
	}

	baseURL, err := container.ConnectionString(ctx)
	if err != nil {
		t.Fatalf("ConnectionString: %v", err)
	}

	adapter, err := ollama.New(ollama.WithModel(model), ollama.WithBaseURL(baseURL))
	if err != nil {
		t.Fatalf("ollama.New: %v", err)
	}

	f := LoadFixture(t, "ollama", "generate_happy_llama3.1-8b")
	// Real model output and token counts are nondeterministic. Keep only the
	// structural assertions from the shared fixture.
	f.Expect.ResponseText = ""
	f.Expect.UsageInputTokens = 0
	f.Expect.UsageOutputTokens = 0
	AssertGenerate(t, adapter, f)
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
