package minimax

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/costa92/llm-agent-contract/llm"
)

// baseResp is MiniMax's envelope status block. MiniMax returns HTTP 200 even
// on logical failure, so StatusCode != 0 is the real error signal.
type baseResp struct {
	StatusCode int    `json:"status_code"`
	StatusMsg  string `json:"status_msg"`
}

// baseRespError maps a non-zero MiniMax status_code to a typed llm error.
// Returns nil when StatusCode == 0. The contract error structs carry a
// Provider string and a Wrapped error (NOT a Message field) — see
// llm/errors.go — so the textual detail goes into a wrapped error.
func baseRespError(prefix string, br baseResp) error {
	if br.StatusCode == 0 {
		return nil
	}
	wrapped := fmt.Errorf("%s: minimax status_code=%d: %s", prefix, br.StatusCode, br.StatusMsg)
	switch br.StatusCode {
	case 1004: // auth / invalid api key
		return &llm.AuthError{Provider: "minimax", Wrapped: wrapped}
	case 1002, 1039: // rate limit / RPM-TPM limit
		return &llm.RateLimitError{Provider: "minimax", Wrapped: wrapped}
	case 1027, 1013: // service unavailable / internal
		return &llm.TransientError{Provider: "minimax", Wrapped: wrapped}
	default:
		return &llm.InvalidRequestError{Provider: "minimax", Wrapped: wrapped}
	}
}

// postJSON issues a POST {baseURL}{path} with a JSON body, applies the Bearer
// token and extra headers, decodes the JSON response into out, and returns the
// raw HTTP status. The caller checks base_resp separately.
func (m *MiniMax) postJSON(ctx context.Context, path, rawQuery string, body any, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("minimax: marshal request: %w", err)
	}
	u := m.baseURL + path
	if rawQuery != "" {
		u += "?" + rawQuery
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("minimax: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if m.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)
	}
	m.applyExtraHeaders(httpReq)

	client := m.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		// context cancellation is surfaced as-is by net/http via the wrapped err.
		return fmt.Errorf("minimax: request: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("minimax: read response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return &llm.TransientError{
			Provider: "minimax",
			Wrapped:  fmt.Errorf("minimax: http %d: %s", resp.StatusCode, string(respBytes)),
		}
	}
	if err := json.Unmarshal(respBytes, out); err != nil {
		return fmt.Errorf("minimax: decode response: %w", err)
	}
	return nil
}

// applyExtraHeaders adds caller-supplied headers without overriding the
// reserved Authorization / Content-Type headers already set.
func (m *MiniMax) applyExtraHeaders(req *http.Request) {
	for k, v := range m.extraHeaders {
		if http.CanonicalHeaderKey(k) == "Authorization" || http.CanonicalHeaderKey(k) == "Content-Type" {
			continue
		}
		req.Header.Set(k, v)
	}
}
