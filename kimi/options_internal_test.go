package kimi

import (
	"testing"
	"time"
)

// effectiveTimeoutForTest exposes the resolved cfg.timeout that New()
// passed to the SDK via option.WithRequestTimeout.
func (k *Kimi) effectiveTimeoutForTest() time.Duration { return k.timeout }

func TestNew_HasDefaultTimeout(t *testing.T) {
	m, err := New(WithModel("kimi-k2"), WithAPIKey("sk-test"))
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
	m, err := New(WithModel("kimi-k2"), WithAPIKey("sk-test"), WithTimeout(10*time.Second))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := m.effectiveTimeoutForTest(); got != 10*time.Second {
		t.Errorf("explicit timeout = %v, want 10s", got)
	}
}

// TestNew_DefaultBaseURL locks the Moonshot global endpoint default.
func TestNew_DefaultBaseURL(t *testing.T) {
	if defaultBaseURL != "https://api.moonshot.ai/v1" {
		t.Fatalf("defaultBaseURL = %q, want https://api.moonshot.ai/v1", defaultBaseURL)
	}
}

// TestNew_APIKeyEnvFallback verifies New() reads MOONSHOT_API_KEY when the
// caller does not pass WithAPIKey.
func TestNew_APIKeyEnvFallback(t *testing.T) {
	t.Setenv("MOONSHOT_API_KEY", "env-key")
	m, err := New(WithModel("kimi-k2"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if m.Info().Provider != "kimi" {
		t.Fatalf("Provider = %q, want kimi", m.Info().Provider)
	}
}
