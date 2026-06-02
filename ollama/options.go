package ollama

import (
	"errors"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/costa92/llm-agent-providers/internal/compat"
	"github.com/costa92/llm-agent-contract/llm"
	api "github.com/ollama/ollama/api"
)

type config struct {
	model      string
	baseURL    string
	httpClient *http.Client
	timeout    time.Duration
}

type Option func(*config)

func WithModel(m string) Option { return func(c *config) { c.model = m } }

func WithBaseURL(u string) Option { return func(c *config) { c.baseURL = u } }

func WithHost(u string) Option { return WithBaseURL(u) }

func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.httpClient = h } }

func WithTimeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

type statusCapturingTransport struct {
	inner          http.RoundTripper
	last           *int32
	lastRetryAfter *atomic.Pointer[string]
}

func (t *statusCapturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.inner.RoundTrip(req)
	if resp != nil {
		atomic.StoreInt32(t.last, int32(resp.StatusCode))
		// Reset every call (including empty) so a prior request's
		// Retry-After can't bleed into the next 429's wrapErr lookup.
		if t.lastRetryAfter != nil {
			ra := resp.Header.Get("Retry-After")
			t.lastRetryAfter.Store(&ra)
		}
	}
	return resp, err
}

func New(opts ...Option) (*Ollama, error) {
	cfg := config{}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.model == "" {
		return nil, errors.New("ollama: WithModel is required")
	}

	base := cfg.baseURL
	if base == "" {
		base = os.Getenv("OLLAMA_HOST")
	}
	if base == "" {
		base = "http://localhost:11434"
	}
	if !strings.Contains(base, "://") {
		base = "http://" + base
	}

	u, err := url.Parse(base)
	if err != nil {
		return nil, err
	}

	lastStatus := new(int32)
	lastRetryAfter := &atomic.Pointer[string]{}
	strategy := strategyForModel(cfg.model)
	embedDim := embeddingDimensionForModel(cfg.model)

	// P1-6 (ollama 5/5): default 60s request timeout when caller is silent.
	// Guards against indefinite hangs on idle connections at upstream Ollama.
	// ollama-go SDK v0.23.2 exposes no per-request timeout option and its
	// Client.http field is private, so the only clean path is to derive
	// two *http.Client wrapping the SAME *statusCapturingTransport:
	//
	//   - syncClient.Timeout  = cfg.timeout  (default 60s) → Generate, Embed
	//   - streamClient.Timeout = 0           (ctx-only)    → Stream
	//
	// Sharing the transport instance preserves lastStatus + lastRetryAfter
	// reference identity, so 429/Retry-After detection on any path remains
	// observable from wrapErr regardless of which client made the call.
	if cfg.timeout == 0 && (cfg.httpClient == nil || cfg.httpClient.Timeout == 0) {
		cfg.timeout = compat.DefaultTimeout(cfg.timeout)
	}

	// Resolve the inner RoundTripper (caller-supplied or default) ONCE so
	// both derived clients route through the same status-capturing wrapper.
	var inner http.RoundTripper
	if cfg.httpClient != nil && cfg.httpClient.Transport != nil {
		inner = cfg.httpClient.Transport
	} else {
		inner = http.DefaultTransport
	}
	sharedTransport := &statusCapturingTransport{
		inner:          inner,
		last:           lastStatus,
		lastRetryAfter: lastRetryAfter,
	}

	// Derived clients: only the *http.Client is cloned; the transport
	// instance is shared (Trap 2 — copying the transport would split
	// lastStatus/lastRetryAfter and stream-path errors would lose state
	// set by a prior sync-path RoundTrip).
	syncClient := &http.Client{
		Transport: sharedTransport,
		Timeout:   cfg.timeout,
	}
	streamClient := &http.Client{
		Transport: sharedTransport,
		Timeout:   0,
	}

	return &Ollama{
		syncClient:       api.NewClient(u, syncClient),
		streamClient:     api.NewClient(u, streamClient),
		timeout:          cfg.timeout,
		syncHTTPClient:   syncClient,
		streamHTTPClient: streamClient,
		lastStatus:       lastStatus,
		lastRetryAfter:   lastRetryAfter,
		strategy:         strategy,
		info: llm.ProviderInfo{
			Provider: "ollama",
			Model:    cfg.model,
			Capabilities: llm.Capabilities{
				Tools:             strategy.supportsTool,
				Embeddings:        embedDim > 0,
				StructuredOutputs: false,
				PromptCaching:     false,
			},
		},
	}, nil
}
