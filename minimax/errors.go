package minimax

import (
	"context"
	"errors"
	"net"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/costa92/llm-agent/llm"
)

func wrapErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: "minimax", Wrapped: err}
	}

	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case 401, 403:
			return &llm.AuthError{Provider: "minimax", Wrapped: err}
		case 429:
			return &llm.RateLimitError{Provider: "minimax", Wrapped: err}
		case 529:
			return &llm.RateLimitError{Provider: "minimax", Wrapped: err}
		case 500, 502, 503, 504:
			return &llm.TransientError{Provider: "minimax", Wrapped: err}
		default:
			if apiErr.StatusCode >= 400 && apiErr.StatusCode < 500 {
				return &llm.InvalidRequestError{Provider: "minimax", Wrapped: err}
			}
		}
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return &llm.TransientError{Provider: "minimax", Wrapped: err}
	}
	return err
}
