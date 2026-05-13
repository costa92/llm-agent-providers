package deepseek

import (
	"context"
	"errors"
	"net"

	"github.com/costa92/llm-agent/llm"
	openai "github.com/openai/openai-go/v3"
)

func wrapErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: "deepseek", Wrapped: err}
	}

	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case 401, 403:
			return &llm.AuthError{Provider: "deepseek", Wrapped: err}
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
				Provider:   "deepseek",
				RetryAfter: retryAfter,
				Reason:     reason,
				Wrapped:    err,
			}
		case 500, 502, 503, 504:
			return &llm.TransientError{Provider: "deepseek", Wrapped: err}
		default:
			if apiErr.StatusCode >= 400 && apiErr.StatusCode < 500 {
				return &llm.InvalidRequestError{Provider: "deepseek", Wrapped: err}
			}
		}
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return &llm.TransientError{Provider: "deepseek", Wrapped: err}
	}
	return err
}
