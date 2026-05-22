package openai

import (
	"testing"
	"time"
)

// effectiveTimeoutForTest exposes the resolved cfg.timeout that New()
// passed to the SDK via option.WithRequestTimeout. Internal so the
// public surface stays minimal; lives in *_internal_test.go so godoc
// won't list it.
func (o *OpenAI) effectiveTimeoutForTest() time.Duration { return o.timeout }

func TestNew_HasDefaultTimeout(t *testing.T) {
	m, err := New(WithModel("gpt-4o"), WithAPIKey("sk-test"))
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
	m, err := New(WithModel("gpt-4o"), WithAPIKey("sk-test"), WithTimeout(10*time.Second))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := m.effectiveTimeoutForTest(); got != 10*time.Second {
		t.Errorf("explicit timeout = %v, want 10s", got)
	}
}
