package compat

import (
	"context"
	"errors"
	"net"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/costa92/llm-agent-contract/llm"
)

// WrapAnthropicError maps an error returned by a github.com/anthropics/anthropic-sdk-go
// SDK call into the canonical llm/* typed-error tree, tagged with the given
// provider name. Used by the anthropic and minimax packages (MiniMax piggy-
// backs on the Anthropic SDK).
//
// Q2 path A: anthropic-sdk-go publicly re-exports type Error, so status
// extraction uses errors.As(err, *sdk.Error) directly.
//
// Mapping:
//   - nil → nil
//   - context.Canceled → passthrough (caller-initiated; not a provider fault)
//   - context.DeadlineExceeded → *llm.TransientError
//   - *sdk.Error 401/403 → *llm.AuthError
//   - *sdk.Error 429 → *llm.RateLimitError (Retry-After header preserved)
//   - *sdk.Error 529 → *llm.RateLimitError (Retry-After header preserved)
//   - *sdk.Error 5xx (500/502/503/504) → *llm.TransientError
//   - *sdk.Error other 4xx → *llm.InvalidRequestError
//   - net.Error (any) → *llm.TransientError
//   - anything else → passthrough
//
// Differences from WrapOpenAIError:
//   - Status 529 ("Overloaded") maps to RateLimitError, not TransientError.
//     This is Anthropic's capacity-exhaustion signal and clients should
//     back off rather than treat it as a transient network blip.
//   - No insufficient_quota / Reason carve-out: the Anthropic SDK does not
//     surface a quota-vs-rate signal at this layer.
func WrapAnthropicError(provider string, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: provider, Wrapped: err}
	}

	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case 401, 403:
			return &llm.AuthError{Provider: provider, Wrapped: err}
		case 429:
			retryAfter := ""
			if apiErr.Response != nil {
				retryAfter = apiErr.Response.Header.Get("Retry-After")
			}
			return &llm.RateLimitError{Provider: provider, RetryAfter: retryAfter, Wrapped: err}
		case 529:
			retryAfter := ""
			if apiErr.Response != nil {
				retryAfter = apiErr.Response.Header.Get("Retry-After")
			}
			return &llm.RateLimitError{Provider: provider, RetryAfter: retryAfter, Wrapped: err}
		case 500, 502, 503, 504:
			return &llm.TransientError{Provider: provider, Wrapped: err}
		default:
			if apiErr.StatusCode >= 400 && apiErr.StatusCode < 500 {
				return &llm.InvalidRequestError{Provider: provider, Wrapped: err}
			}
		}
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return &llm.TransientError{Provider: provider, Wrapped: err}
	}
	return err
}
