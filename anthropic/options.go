package anthropic

import (
	"errors"
	"net/http"
	"os"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/costa92/llm-agent/llm"
)

type config struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
	timeout    time.Duration
	betaHeader string
}

type Option func(*config)

func WithModel(m string) Option { return func(c *config) { c.model = m } }

func WithAPIKey(k string) Option { return func(c *config) { c.apiKey = k } }

func WithBaseURL(u string) Option { return func(c *config) { c.baseURL = u } }

func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.httpClient = h } }

func WithTimeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

func WithBetaHeader(v string) Option { return func(c *config) { c.betaHeader = v } }

func New(opts ...Option) (*Anthropic, error) {
	cfg := config{}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.model == "" {
		return nil, errors.New("anthropic: WithModel is required")
	}
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}

	var sdkOpts []option.RequestOption
	sdkOpts = append(sdkOpts, option.WithMaxRetries(0))
	if cfg.apiKey != "" {
		sdkOpts = append(sdkOpts, option.WithAPIKey(cfg.apiKey))
	}
	if cfg.baseURL != "" {
		sdkOpts = append(sdkOpts, option.WithBaseURL(cfg.baseURL))
	}
	if cfg.httpClient != nil {
		sdkOpts = append(sdkOpts, option.WithHTTPClient(cfg.httpClient))
	}
	if cfg.betaHeader != "" {
		sdkOpts = append(sdkOpts, option.WithHeader("anthropic-beta", cfg.betaHeader))
	}
	if cfg.timeout > 0 {
		sdkOpts = append(sdkOpts, option.WithRequestTimeout(cfg.timeout))
	}

	client := sdk.NewClient(sdkOpts...)
	return &Anthropic{
		client: &client,
		info: llm.ProviderInfo{
			Provider: "anthropic",
			Model:    cfg.model,
			Capabilities: llm.Capabilities{
				Tools:             true,
				Embeddings:        false,
				StructuredOutputs: false,
				PromptCaching:     false,
			},
		},
	}, nil
}
