package compat

import (
	"testing"
	"time"
)

func TestDefaultTimeout_ZeroReturns60s(t *testing.T) {
	if got := DefaultTimeout(0); got != 60*time.Second {
		t.Errorf("DefaultTimeout(0) = %v, want %v", got, 60*time.Second)
	}
}

func TestDefaultTimeout_NonZeroReturnsArg(t *testing.T) {
	if got := DefaultTimeout(5 * time.Second); got != 5*time.Second {
		t.Errorf("DefaultTimeout(5s) = %v, want 5s", got)
	}
	if got := DefaultTimeout(time.Millisecond); got != time.Millisecond {
		t.Errorf("DefaultTimeout(1ms) = %v, want 1ms", got)
	}
}
