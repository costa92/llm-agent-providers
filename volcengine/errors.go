package volcengine

import (
	"context"
	"errors"
	"net"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// wrapErr maps an arkruntime SDK error into the canonical llm/* typed-error
// tree. Both *model.APIError and *model.RequestError carry an HTTP status
// code; routing is identical to the other adapters.
//
// Mapping:
//   - nil → nil
//   - context.Canceled → passthrough (caller-initiated; not a provider fault)
//   - context.DeadlineExceeded → *llm.TransientError
//   - status 401/403 → *llm.AuthError
//   - status 429 → *llm.RateLimitError
//   - status 500/502/503/504 → *llm.TransientError
//   - other 4xx → *llm.InvalidRequestError
//   - net.Error (any) → *llm.TransientError
//   - anything else → passthrough
func wrapErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: "volcengine", Wrapped: err}
	}

	status := 0
	var apiErr *model.APIError
	var reqErr *model.RequestError
	if errors.As(err, &apiErr) {
		status = apiErr.HTTPStatusCode
	} else if errors.As(err, &reqErr) {
		status = reqErr.HTTPStatusCode
	}

	if status != 0 {
		switch status {
		case 401, 403:
			return &llm.AuthError{Provider: "volcengine", Wrapped: err}
		case 429:
			return &llm.RateLimitError{Provider: "volcengine", Wrapped: err}
		case 500, 502, 503, 504:
			return &llm.TransientError{Provider: "volcengine", Wrapped: err}
		default:
			if status >= 400 && status < 500 {
				return &llm.InvalidRequestError{Provider: "volcengine", Wrapped: err}
			}
		}
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return &llm.TransientError{Provider: "volcengine", Wrapped: err}
	}
	return err
}
