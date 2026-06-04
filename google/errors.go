package google

import (
	"context"
	"errors"
	"fmt"
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

// blockedPromptErr returns an InvalidRequestError when the response carries a
// PromptFeedback block reason and no candidates (Gemini returns HTTP 200 in
// this case). Returns nil otherwise.
func blockedPromptErr(resp *genai.GenerateContentResponse) error {
	if resp == nil {
		return nil
	}
	if len(resp.Candidates) == 0 && resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
		return &llm.InvalidRequestError{
			Provider: "google",
			Wrapped: fmt.Errorf("prompt blocked: %s (%s)",
				resp.PromptFeedback.BlockReason, resp.PromptFeedback.BlockReasonMessage),
		}
	}
	return nil
}
