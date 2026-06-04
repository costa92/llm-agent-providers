package kimi

import (
	"errors"
	"net/http"
	"os"
	"time"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/costa92/llm-agent-providers/internal/compat"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const defaultBaseURL = "https://api.moonshot.ai/v1"

type config struct {
	apiKey       string
	model        string
	baseURL      string
	httpClient   *http.Client
	timeout      time.Duration
	extraHeaders map[string]string
}

type Option func(*config)

func WithModel(m string) Option { return func(c *config) { c.model = m } }

func WithAPIKey(k string) Option { return func(c *config) { c.apiKey = k } }

func WithBaseURL(u string) Option { return func(c *config) { c.baseURL = u } }

func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.httpClient = h } }

func WithTimeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

// WithExtraHeaders injects additional headers into every outbound request.
// Reserved headers (Authorization, Content-Type) are not overridable; extra
// headers are additive via option.WithHeaderAdd.
func WithExtraHeaders(h map[string]string) Option {
	return func(c *config) { c.extraHeaders = h }
}

func New(opts ...Option) (*Kimi, error) {
	var cfg config
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.model == "" {
		return nil, errors.New("kimi: WithModel is required")
	}
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("MOONSHOT_API_KEY")
	}
	// P1-6/P1-23: default 60s request timeout when caller is silent.
	// Guards against indefinite hangs on idle HTTP connections. SDK
	// applies this per-request via option.WithRequestTimeout, so
	// streaming is NOT capped by a client-level Timeout — caller ctx
	// still governs long-running streams. Default lifted into
	// internal/compat so the providers share the canonical value.
	cfg.timeout = compat.DefaultTimeout(cfg.timeout)

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	var sdkOpts []option.RequestOption
	sdkOpts = append(sdkOpts, option.WithMaxRetries(0))
	if cfg.apiKey != "" {
		sdkOpts = append(sdkOpts, option.WithAPIKey(cfg.apiKey))
	}
	if baseURL != "" {
		sdkOpts = append(sdkOpts, option.WithBaseURL(baseURL))
	}
	if cfg.httpClient != nil {
		sdkOpts = append(sdkOpts, option.WithHTTPClient(cfg.httpClient))
	}
	for k, v := range cfg.extraHeaders {
		sdkOpts = append(sdkOpts, option.WithHeaderAdd(k, v))
	}
	if cfg.timeout > 0 {
		sdkOpts = append(sdkOpts, option.WithRequestTimeout(cfg.timeout))
	}

	client := openai.NewClient(sdkOpts...)
	return &Kimi{
		client:  &client,
		timeout: cfg.timeout,
		info: llm.ProviderInfo{
			Provider:     "kimi",
			Model:        cfg.model,
			Capabilities: capabilitiesForModel(cfg.model),
		},
	}, nil
}
