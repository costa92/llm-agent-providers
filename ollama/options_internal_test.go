package ollama

import (
	"testing"
	"time"
)

// P1-6 ollama closure (5/5). Mirrors the *_internal_test.go pattern used
// by openai/anthropic/deepseek/minimax but additionally asserts the
// derived-client split: sync inherits cfg.timeout (default 60s), stream
// keeps Timeout=0 so long stream connections remain governed by ctx.

func TestNew_Ollama_HasDefaultTimeout(t *testing.T) {
	m, err := New(WithModel("llama3.1:8b"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := m.effectiveTimeoutForTest()
	if got != 60*time.Second {
		t.Fatalf("default timeout = %v, want 60s", got)
	}
}

func TestNew_Ollama_RespectsExplicitTimeout(t *testing.T) {
	m, err := New(WithModel("llama3.1:8b"), WithTimeout(5*time.Second))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := m.effectiveTimeoutForTest(); got != 5*time.Second {
		t.Fatalf("explicit timeout = %v, want 5s", got)
	}
}

// Stream client must have Timeout=0 — caller ctx governs long-running
// streams. A non-zero Timeout would cut the connection mid-stream.
func TestNew_Ollama_StreamClientHasZeroTimeout(t *testing.T) {
	m, err := New(WithModel("llama3.1:8b"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	streamClient := m.streamHTTPClientForTest()
	if streamClient == nil {
		t.Fatal("streamHTTPClientForTest() = nil")
	}
	if streamClient.Timeout != 0 {
		t.Fatalf("stream client Timeout = %v, want 0 (ctx-only)", streamClient.Timeout)
	}
}

// Sync client carries cfg.timeout — protects Generate/Embed from hung
// idle connections at upstream Ollama.
func TestNew_Ollama_SyncClientHonorsTimeout(t *testing.T) {
	m, err := New(WithModel("llama3.1:8b"), WithTimeout(3*time.Second))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	syncClient := m.syncHTTPClientForTest()
	if syncClient == nil {
		t.Fatal("syncHTTPClientForTest() = nil")
	}
	if syncClient.Timeout != 3*time.Second {
		t.Fatalf("sync client Timeout = %v, want 3s", syncClient.Timeout)
	}
}

// Trap 2 guard: sync and stream clients MUST share the same
// *statusCapturingTransport instance, otherwise lastStatus /
// lastRetryAfter observed by wrapErr would diverge between paths.
// (existing TestGenerate_Ollama_429* tests assert sync-path correctness;
// this directly asserts the transport identity invariant.)
func TestNew_Ollama_SyncAndStreamShareTransport(t *testing.T) {
	m, err := New(WithModel("llama3.1:8b"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	sync := m.syncHTTPClientForTest()
	stream := m.streamHTTPClientForTest()
	if sync == nil || stream == nil {
		t.Fatal("derived clients must both be non-nil")
	}
	if sync == stream {
		t.Fatal("sync and stream must be distinct *http.Client instances")
	}
	if sync.Transport == nil || stream.Transport == nil {
		t.Fatal("derived clients must both have a Transport")
	}
	if sync.Transport != stream.Transport {
		t.Fatalf("sync and stream must share the SAME transport instance for lastStatus/Retry-After observability; got %p vs %p", sync.Transport, stream.Transport)
	}
}
