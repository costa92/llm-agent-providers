package google

import (
	"context"
	"errors"
	"net"

	"github.com/costa92/llm-agent-contract/llm"
	"google.golang.org/genai"
)

// wrapErr maps genai SDK errors to the repo's typed llm.* errors.
func wrapErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: "google", Wrapped: err}
	}
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		switch {
		case apiErr.Code == 401 || apiErr.Code == 403:
			return &llm.AuthError{Provider: "google", Wrapped: err}
		case apiErr.Code == 429:
			return &llm.RateLimitError{Provider: "google", Wrapped: err}
		case apiErr.Code >= 500:
			return &llm.TransientError{Provider: "google", Wrapped: err}
		default:
			return &llm.InvalidRequestError{Provider: "google", Wrapped: err}
		}
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return &llm.TransientError{Provider: "google", Wrapped: err}
	}
	return &llm.InvalidRequestError{Provider: "google", Wrapped: err}
}
