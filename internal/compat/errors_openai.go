package compat

import (
	"context"
	"errors"
	"net"

	"github.com/costa92/llm-agent-contract/llm"
	openai "github.com/openai/openai-go/v3"
)

// WrapOpenAIError maps an error returned by a github.com/openai/openai-go/v3
// SDK call into the canonical llm/* typed-error tree, tagged with the given
// provider name. Used by the openai and deepseek packages (DeepSeek piggy-
// backs on the OpenAI SDK).
//
// Mapping:
//   - nil → nil
//   - context.Canceled → passthrough (caller-initiated; not a provider fault)
//   - context.DeadlineExceeded → *llm.TransientError
//   - *openai.Error 401/403 → *llm.AuthError
//   - *openai.Error 429 (Type/Code == "insufficient_quota") →
//     *llm.RateLimitError with Reason="quota_exhausted"
//   - *openai.Error 429 (other) → *llm.RateLimitError with empty Reason
//     (RetryAfter sourced from the Retry-After response header when present)
//   - *openai.Error 5xx (500/502/503/504) → *llm.TransientError
//   - *openai.Error other 4xx → *llm.InvalidRequestError
//   - net.Error (any) → *llm.TransientError
//   - anything else → passthrough
func WrapOpenAIError(provider string, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: provider, Wrapped: err}
	}

	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case 401, 403:
			return &llm.AuthError{Provider: provider, Wrapped: err}
		case 429:
			reason := ""
			if apiErr.Type == "insufficient_quota" || apiErr.Code == "insufficient_quota" {
				reason = "quota_exhausted"
			}
			retryAfter := ""
			if apiErr.Response != nil {
				retryAfter = apiErr.Response.Header.Get("Retry-After")
			}
			return &llm.RateLimitError{
				Provider:   provider,
				RetryAfter: retryAfter,
				Reason:     reason,
				Wrapped:    err,
			}
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
