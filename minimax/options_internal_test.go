package minimax

import (
	"testing"
	"time"
)

// effectiveTimeoutForTest exposes the resolved cfg.timeout that New()
// passed to the SDK via option.WithRequestTimeout.
func (m *MiniMax) effectiveTimeoutForTest() time.Duration { return m.timeout }

func TestNew_HasDefaultTimeout(t *testing.T) {
	m, err := New(WithModel("MiniMax-Text-01"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := m.effectiveTimeoutForTest()
	if got <= 0 {
		t.Fatalf("default timeout = %v, want > 0", got)
	}
	if got != 60*time.Second {
		t.Errorf("default timeout = %v, want 60s", got)
	}
}

func TestNew_RespectsExplicitTimeout(t *testing.T) {
	m, err := New(WithModel("MiniMax-Text-01"), WithAPIKey("test-key"), WithTimeout(10*time.Second))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := m.effectiveTimeoutForTest(); got != 10*time.Second {
		t.Errorf("explicit timeout = %v, want 10s", got)
	}
}

// raw-HTTP accessors for testing config plumbing.
func (m *MiniMax) baseURLForTest() string                 { return m.baseURL }
func (m *MiniMax) apiKeyForTest() string                  { return m.apiKey }
func (m *MiniMax) extraHeadersForTest() map[string]string { return m.extraHeaders }
func (m *MiniMax) groupIDForTest() string                 { return m.groupID }
func (m *MiniMax) embeddingTypeForTest() string           { return m.embeddingType }

func TestNew_RetainsRawHTTPConfig(t *testing.T) {
	m, err := New(
		WithModel("image-01"),
		WithAPIKey("k"),
		WithBaseURL("https://example.test"),
		WithExtraHeaders(map[string]string{"X-Trace": "1"}),
		WithGroupID("grp-9"),
		WithEmbeddingType("query"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if m.baseURLForTest() != "https://example.test" {
		t.Fatalf("baseURL = %q", m.baseURLForTest())
	}
	if m.apiKeyForTest() != "k" {
		t.Fatalf("apiKey = %q", m.apiKeyForTest())
	}
	if m.extraHeadersForTest()["X-Trace"] != "1" {
		t.Fatalf("extraHeaders = %v", m.extraHeadersForTest())
	}
	if m.groupIDForTest() != "grp-9" {
		t.Fatalf("groupID = %q", m.groupIDForTest())
	}
	if m.embeddingTypeForTest() != "query" {
		t.Fatalf("embeddingType = %q", m.embeddingTypeForTest())
	}
}

func TestNew_EmbeddingTypeDefaultsDB(t *testing.T) {
	m, err := New(WithModel("embo-01"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := m.embeddingTypeForTest(); got != "db" {
		t.Fatalf("default embeddingType = %q, want db", got)
	}
}
