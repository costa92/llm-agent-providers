package compat

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/url"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/costa92/llm-agent/llm"
)

// newAnthropicAPIErr constructs a *sdk.Error (alias for apierror.Error)
// mirroring what the Anthropic SDK returns on a failed HTTP call. The
// SDK only reads StatusCode and Response.Header for our mapping, so we
// populate only those fields. Request is set to a minimal http.Request
// so that any incidental .Error() formatting (in test failure output)
// does not nil-panic.
func newAnthropicAPIErr(status int, retryAfter string) *sdk.Error {
	resp := &http.Response{StatusCode: status, Header: http.Header{}}
	if retryAfter != "" {
		resp.Header.Set("Retry-After", retryAfter)
	}
	req, _ := http.NewRequest("POST", "http://example.test/v1/messages", nil)
	return &sdk.Error{
		StatusCode: status,
		Request:    req,
		Response:   resp,
	}
}

func TestWrapAnthropicError_Nil(t *testing.T) {
	if got := WrapAnthropicError("anthropic", nil); got != nil {
		t.Errorf("WrapAnthropicError(_, nil) = %v, want nil", got)
	}
}

func TestWrapAnthropicError_ContextCanceledPassthrough(t *testing.T) {
	if got := WrapAnthropicError("anthropic", context.Canceled); !errors.Is(got, context.Canceled) {
		t.Errorf("WrapAnthropicError(context.Canceled) lost the sentinel; got %v", got)
	}
}

func TestWrapAnthropicError_ContextDeadlineExceededIsTransient(t *testing.T) {
	got := WrapAnthropicError("anthropic", context.DeadlineExceeded)
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapAnthropicError(DeadlineExceeded) = %T %v, want *llm.TransientError", got, got)
	}
	if te.Provider != "anthropic" {
		t.Errorf("Provider = %q, want anthropic", te.Provider)
	}
	if !errors.Is(te.Wrapped, context.DeadlineExceeded) {
		t.Errorf("Wrapped = %v, want context.DeadlineExceeded", te.Wrapped)
	}
}

func TestWrapAnthropicError_401IsAuthError(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(401, ""))
	var ae *llm.AuthError
	if !errors.As(got, &ae) {
		t.Fatalf("WrapAnthropicError(401) = %T %v, want *llm.AuthError", got, got)
	}
	if ae.Provider != "anthropic" {
		t.Errorf("Provider = %q, want anthropic", ae.Provider)
	}
}

func TestWrapAnthropicError_403IsAuthError(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(403, ""))
	var ae *llm.AuthError
	if !errors.As(got, &ae) {
		t.Fatalf("WrapAnthropicError(403) = %T %v, want *llm.AuthError", got, got)
	}
}

func TestWrapAnthropicError_400IsInvalidRequest(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(400, ""))
	var ie *llm.InvalidRequestError
	if !errors.As(got, &ie) {
		t.Fatalf("WrapAnthropicError(400) = %T %v, want *llm.InvalidRequestError", got, got)
	}
}

func TestWrapAnthropicError_429WithRetryAfterSeconds(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(429, "30"))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapAnthropicError(429+30s) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.RetryAfter != "30" {
		t.Errorf("RetryAfter = %q, want 30", rl.RetryAfter)
	}
	if rl.Provider != "anthropic" {
		t.Errorf("Provider = %q, want anthropic", rl.Provider)
	}
}

func TestWrapAnthropicError_429WithRetryAfterHTTPDate(t *testing.T) {
	when := "Wed, 21 Oct 2026 07:28:00 GMT"
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(429, when))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapAnthropicError(429+HTTP-date) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.RetryAfter != when {
		t.Errorf("RetryAfter = %q, want %q (raw passthrough)", rl.RetryAfter, when)
	}
}

func TestWrapAnthropicError_429WithoutRetryAfter(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(429, ""))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapAnthropicError(429 no Retry-After) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.RetryAfter != "" {
		t.Errorf("RetryAfter = %q, want empty", rl.RetryAfter)
	}
}

// TestWrapAnthropicError_529IsRateLimit_NotTransient is the explicit
// pin for the Anthropic-family special case: status 529 ("Overloaded")
// is a capacity-exhaustion signal that clients should back off on, NOT
// a network blip. It must map to RateLimitError (so retry policy
// honors Retry-After), never TransientError.
func TestWrapAnthropicError_529IsRateLimit_NotTransient(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(529, ""))

	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapAnthropicError(529) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.Provider != "anthropic" {
		t.Errorf("Provider = %q, want anthropic", rl.Provider)
	}

	var te *llm.TransientError
	if errors.As(got, &te) {
		t.Fatalf("WrapAnthropicError(529) incorrectly classified as TransientError: %v", te)
	}
}

func TestWrapAnthropicError_529WithRetryAfter(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(529, "120"))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapAnthropicError(529+RetryAfter) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.RetryAfter != "120" {
		t.Errorf("RetryAfter = %q, want 120", rl.RetryAfter)
	}
}

func TestWrapAnthropicError_500IsTransient(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(500, ""))
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapAnthropicError(500) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapAnthropicError_502IsTransient(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(502, ""))
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapAnthropicError(502) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapAnthropicError_503IsTransient(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(503, ""))
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapAnthropicError(503) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapAnthropicError_504IsTransient(t *testing.T) {
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(504, ""))
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapAnthropicError(504) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapAnthropicError_418IsInvalidRequest(t *testing.T) {
	// Defensive guard for the default-4xx branch.
	got := WrapAnthropicError("anthropic", newAnthropicAPIErr(418, ""))
	var ie *llm.InvalidRequestError
	if !errors.As(got, &ie) {
		t.Fatalf("WrapAnthropicError(418) = %T %v, want *llm.InvalidRequestError", got, got)
	}
}

func TestWrapAnthropicError_MiniMaxProviderTagging_429(t *testing.T) {
	// Proves the helper is properly parameterized — same 429 input must
	// yield Provider="minimax" when invoked with that label.
	got := WrapAnthropicError("minimax", newAnthropicAPIErr(429, "5"))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapAnthropicError(minimax, 429) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.Provider != "minimax" {
		t.Errorf("Provider = %q, want minimax", rl.Provider)
	}
	if rl.RetryAfter != "5" {
		t.Errorf("RetryAfter = %q, want 5", rl.RetryAfter)
	}
}

func TestWrapAnthropicError_MiniMaxProviderTagging_529(t *testing.T) {
	// The 529 special case must hold under the minimax label too —
	// MiniMax piggybacks on the same Anthropic SDK and inherits the
	// capacity-exhaustion semantics.
	got := WrapAnthropicError("minimax", newAnthropicAPIErr(529, ""))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapAnthropicError(minimax, 529) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.Provider != "minimax" {
		t.Errorf("Provider = %q, want minimax", rl.Provider)
	}
}

func TestWrapAnthropicError_NetErrorIsTransient(t *testing.T) {
	netErr := &url.Error{Op: "Get", URL: "http://example", Err: &net.OpError{Op: "dial", Err: errors.New("connection refused")}}
	got := WrapAnthropicError("anthropic", netErr)
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapAnthropicError(netErr) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapAnthropicError_UnwrappedGenericPassthrough(t *testing.T) {
	plain := errors.New("not a typed error")
	got := WrapAnthropicError("anthropic", plain)
	if got != plain {
		t.Errorf("WrapAnthropicError(plain) = %v, want passthrough", got)
	}
}
