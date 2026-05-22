package ollama

import (
	"context"
	"errors"
	"net"
	"strings"
	"sync/atomic"

	"github.com/costa92/llm-agent/llm"
	api "github.com/ollama/ollama/api"
)

func (o *Ollama) wrapErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: "ollama", Wrapped: err}
	}

	status := int(atomic.LoadInt32(o.lastStatus))
	msg := strings.ToLower(err.Error())

	var authErr api.AuthorizationError
	if errors.As(err, &authErr) {
		return &llm.AuthError{Provider: "ollama", Wrapped: err}
	}
	var statusErr api.StatusError
	if errors.As(err, &statusErr) && status == 0 {
		status = statusErr.StatusCode
	}

	switch {
	case status == 401 || status == 403:
		return &llm.AuthError{Provider: "ollama", Wrapped: err}
	case status == 404 && (strings.Contains(msg, "not found") || strings.Contains(msg, "not pulled")):
		return &llm.InvalidRequestError{Provider: "ollama", Wrapped: err}
	case status == 429:
		// P1-9: lift raw Retry-After string (consumers parse per
		// llm.RateLimitError contract). The capturing transport
		// reset this on every RoundTrip so it cannot bleed from a
		// prior request.
		retryAfter := ""
		if o.lastRetryAfter != nil {
			if p := o.lastRetryAfter.Load(); p != nil {
				retryAfter = *p
			}
		}
		return &llm.RateLimitError{Provider: "ollama", RetryAfter: retryAfter, Wrapped: err}
	case status >= 500 && status < 600:
		return &llm.TransientError{Provider: "ollama", Wrapped: err}
	case status >= 400 && status < 500:
		return &llm.InvalidRequestError{Provider: "ollama", Wrapped: err}
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return &llm.TransientError{Provider: "ollama", Wrapped: err}
	}
	if strings.Contains(msg, "connection refused") {
		return &llm.TransientError{Provider: "ollama", Wrapped: err}
	}
	return err
}
