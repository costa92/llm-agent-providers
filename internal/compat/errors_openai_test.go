package compat

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/costa92/llm-agent/llm"
	openai "github.com/openai/openai-go/v3"
)

// newAPIErr constructs a *openai.Error mirroring what the SDK returns.
// Mirrors the openai-go/v3 SDK's apierror.Error layout: StatusCode is
// the parsed HTTP status, Response carries the raw http.Response (so
// Retry-After header lookups succeed), Type/Code carry the JSON-body
// classification.
func newAPIErr(status int, typ, code, retryAfter string) *openai.Error {
	resp := &http.Response{StatusCode: status, Header: http.Header{}}
	if retryAfter != "" {
		resp.Header.Set("Retry-After", retryAfter)
	}
	return &openai.Error{
		StatusCode: status,
		Type:       typ,
		Code:       code,
		Response:   resp,
	}
}

func TestWrapOpenAIError_Nil(t *testing.T) {
	if got := WrapOpenAIError("openai", nil); got != nil {
		t.Errorf("WrapOpenAIError(_, nil) = %v, want nil", got)
	}
}

func TestWrapOpenAIError_ContextCanceledPassthrough(t *testing.T) {
	if got := WrapOpenAIError("openai", context.Canceled); !errors.Is(got, context.Canceled) {
		t.Errorf("WrapOpenAIError(context.Canceled) lost the sentinel; got %v", got)
	}
}

func TestWrapOpenAIError_ContextDeadlineExceededIsTransient(t *testing.T) {
	got := WrapOpenAIError("openai", context.DeadlineExceeded)
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapOpenAIError(DeadlineExceeded) = %T %v, want *llm.TransientError", got, got)
	}
	if te.Provider != "openai" {
		t.Errorf("Provider = %q, want openai", te.Provider)
	}
	if !errors.Is(te.Wrapped, context.DeadlineExceeded) {
		t.Errorf("Wrapped = %v, want context.DeadlineExceeded", te.Wrapped)
	}
}

func TestWrapOpenAIError_401IsAuthError(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(401, "invalid_request_error", "invalid_api_key", ""))
	var ae *llm.AuthError
	if !errors.As(got, &ae) {
		t.Fatalf("WrapOpenAIError(401) = %T %v, want *llm.AuthError", got, got)
	}
	if ae.Provider != "openai" {
		t.Errorf("Provider = %q, want openai", ae.Provider)
	}
}

func TestWrapOpenAIError_403IsAuthError(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(403, "invalid_request_error", "forbidden", ""))
	var ae *llm.AuthError
	if !errors.As(got, &ae) {
		t.Fatalf("WrapOpenAIError(403) = %T %v, want *llm.AuthError", got, got)
	}
}

func TestWrapOpenAIError_429InsufficientQuotaWithRetryAfter(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(429, "insufficient_quota", "insufficient_quota", "30"))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapOpenAIError(429+quota) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.Reason != "quota_exhausted" {
		t.Errorf("Reason = %q, want quota_exhausted", rl.Reason)
	}
	if rl.RetryAfter != "30" {
		t.Errorf("RetryAfter = %q, want 30", rl.RetryAfter)
	}
}

func TestWrapOpenAIError_429PlainWithRetryAfter(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(429, "rate_limit_error", "rate_limit_exceeded", "12"))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapOpenAIError(429) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.Reason != "" {
		t.Errorf("Reason = %q, want empty", rl.Reason)
	}
	if rl.RetryAfter != "12" {
		t.Errorf("RetryAfter = %q, want 12", rl.RetryAfter)
	}
}

func TestWrapOpenAIError_429NoRetryAfter(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(429, "rate_limit_error", "rate_limit_exceeded", ""))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapOpenAIError(429 no Retry-After) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.RetryAfter != "" {
		t.Errorf("RetryAfter = %q, want empty", rl.RetryAfter)
	}
}

func TestWrapOpenAIError_500IsTransient(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(500, "server_error", "", ""))
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapOpenAIError(500) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapOpenAIError_502IsTransient(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(502, "server_error", "", ""))
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapOpenAIError(502) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapOpenAIError_503IsTransient(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(503, "server_error", "", ""))
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapOpenAIError(503) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapOpenAIError_504IsTransient(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(504, "server_error", "", ""))
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapOpenAIError(504) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapOpenAIError_400IsInvalidRequest(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(400, "invalid_request_error", "bad_request", ""))
	var ie *llm.InvalidRequestError
	if !errors.As(got, &ie) {
		t.Fatalf("WrapOpenAIError(400) = %T %v, want *llm.InvalidRequestError", got, got)
	}
}

func TestWrapOpenAIError_404IsInvalidRequest(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(404, "invalid_request_error", "model_not_found", ""))
	var ie *llm.InvalidRequestError
	if !errors.As(got, &ie) {
		t.Fatalf("WrapOpenAIError(404) = %T %v, want *llm.InvalidRequestError", got, got)
	}
}

func TestWrapOpenAIError_DeepSeekProviderTagging(t *testing.T) {
	// Proves the helper is properly parameterized — same 429 input must
	// yield Provider="deepseek" when invoked with that label.
	got := WrapOpenAIError("deepseek", newAPIErr(429, "rate_limit_error", "rate_limit_exceeded", "5"))
	var rl *llm.RateLimitError
	if !errors.As(got, &rl) {
		t.Fatalf("WrapOpenAIError(deepseek, 429) = %T %v, want *llm.RateLimitError", got, got)
	}
	if rl.Provider != "deepseek" {
		t.Errorf("Provider = %q, want deepseek", rl.Provider)
	}
	if rl.RetryAfter != "5" {
		t.Errorf("RetryAfter = %q, want 5", rl.RetryAfter)
	}
}

func TestWrapOpenAIError_NetErrorIsTransient(t *testing.T) {
	netErr := &url.Error{Op: "Get", URL: "http://example", Err: &net.OpError{Op: "dial", Err: errors.New("connection refused")}}
	got := WrapOpenAIError("openai", netErr)
	var te *llm.TransientError
	if !errors.As(got, &te) {
		t.Fatalf("WrapOpenAIError(netErr) = %T %v, want *llm.TransientError", got, got)
	}
}

func TestWrapOpenAIError_UnwrappedGenericPassthrough(t *testing.T) {
	plain := errors.New("not a typed error")
	got := WrapOpenAIError("openai", plain)
	if got != plain {
		t.Errorf("WrapOpenAIError(plain) = %v, want passthrough", got)
	}
}

// Defensive guard: ensures we don't accidentally classify a 418 (4xx
// other) as anything but InvalidRequestError per the OpenAI fallback
// branch. Mirrors the existing per-provider integration tests.
func TestWrapOpenAIError_418IsInvalidRequest(t *testing.T) {
	got := WrapOpenAIError("openai", newAPIErr(418, "invalid_request_error", "im_a_teapot", ""))
	var ie *llm.InvalidRequestError
	if !errors.As(got, &ie) {
		t.Fatalf("WrapOpenAIError(418) = %T %v, want *llm.InvalidRequestError", got, got)
	}
}

// Sanity: TransientError must wrap the deadline; verifying the time
// package is imported (and to keep coverage symmetric with the
// per-provider tests that exercise the same path).
var _ = time.Second
