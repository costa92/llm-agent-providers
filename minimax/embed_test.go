package minimax

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

func TestEmbed_MiniMax_Happy(t *testing.T) {
	var gotPath, gotBody, gotGroupID, gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotGroupID = r.URL.Query().Get("GroupId")
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		// Top-level vectors + total_tokens (NOT nested under data/usage).
		_, _ = w.Write([]byte(`{
			"vectors":[[0.1,0.2,0.3],[0.4,0.5,0.6]],
			"total_tokens":12,
			"base_resp":{"status_code":0,"status_msg":"success"}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("embo-01"),
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithGroupID("grp-7"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if gotPath != "/v1/embeddings" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotGroupID != "grp-7" {
		t.Fatalf("GroupId query = %q, want grp-7", gotGroupID)
	}
	if gotAuth != "Bearer k" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if !strings.Contains(gotBody, `"model":"embo-01"`) {
		t.Fatalf("body missing model: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"texts":["hello","world"]`) {
		t.Fatalf("body missing texts in order: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"type":"db"`) {
		t.Fatalf("body missing type=db: %s", gotBody)
	}
	if len(vectors) != 2 {
		t.Fatalf("len(vectors) = %d, want 2", len(vectors))
	}
	if len(vectors[0]) != 3 || vectors[1][2] != 0.6 {
		t.Fatalf("vector content wrong: %v", vectors)
	}
	if usage.TotalTokens != 12 || usage.Source != llm.UsageReported {
		t.Fatalf("usage = %+v, want total=12 reported", usage)
	}
}

func TestEmbed_MiniMax_QueryType(t *testing.T) {
	var gotBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"vectors":[[0.1]],"total_tokens":1,"base_resp":{"status_code":0}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("embo-01"), WithAPIKey("k"), WithBaseURL(server.URL),
		WithGroupID("g"), WithEmbeddingType("query"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := m.Embed(context.Background(), []string{"q"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if !strings.Contains(gotBody, `"type":"query"`) {
		t.Fatalf("body missing type=query: %s", gotBody)
	}
}

func TestEmbed_MiniMax_Empty(t *testing.T) {
	m, err := New(WithModel("embo-01"), WithAPIKey("k"), WithGroupID("g"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(empty): %v", err)
	}
	if len(vectors) != 0 {
		t.Fatalf("len(vectors) = %d, want 0", len(vectors))
	}
	if usage.Source != llm.UsageReported {
		t.Fatalf("usage.Source = %q, want reported", usage.Source)
	}
}

func TestEmbed_MiniMax_NotSupported(t *testing.T) {
	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _, err = m.Embed(context.Background(), []string{"x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("err = %v, want ErrCapabilityNotSupported", err)
	}
	if got := m.EmbedDimensions(); got != 0 {
		t.Fatalf("EmbedDimensions(non-embed) = %d, want 0", got)
	}
}

func TestEmbedDimensions_MiniMax(t *testing.T) {
	m, err := New(WithModel("embo-01"), WithAPIKey("k"), WithGroupID("g"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := m.EmbedDimensions(); got != 1536 {
		t.Fatalf("EmbedDimensions() = %d, want 1536", got)
	}
	if !m.Info().Capabilities.Embeddings {
		t.Fatalf("embo-01 must report Embeddings")
	}
}
