package ollama

import (
	"errors"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/costa92/llm-agent/llm"
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
	inner http.RoundTripper
	last  *int32
}

func (t *statusCapturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.inner.RoundTrip(req)
	if resp != nil {
		atomic.StoreInt32(t.last, int32(resp.StatusCode))
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
	strategy := strategyForModel(cfg.model)
	embedDim := embeddingDimensionForModel(cfg.model)
	httpClient := cfg.httpClient
	if httpClient == nil {
		httpClient = &http.Client{}
	} else {
		cp := *httpClient
		httpClient = &cp
	}
	if cfg.timeout > 0 {
		httpClient.Timeout = cfg.timeout
	}
	inner := httpClient.Transport
	if inner == nil {
		inner = http.DefaultTransport
	}
	httpClient.Transport = &statusCapturingTransport{
		inner: inner,
		last:  lastStatus,
	}

	client := api.NewClient(u, httpClient)
	return &Ollama{
		client:     client,
		lastStatus: lastStatus,
		strategy:   strategy,
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
