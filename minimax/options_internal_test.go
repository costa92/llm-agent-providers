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
