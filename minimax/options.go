package minimax

import (
	"errors"
	"net/http"
	"os"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/costa92/llm-agent-providers/internal/compat"
	"github.com/costa92/llm-agent-contract/llm"
)

type Region string

const (
	RegionCN     Region = "cn"
	RegionGlobal Region = "global"
)

const defaultBaseURL = "https://api.minimax.chat"

type config struct {
	apiKey        string
	model         string
	baseURL       string
	httpClient    *http.Client
	timeout       time.Duration
	region        Region
	extraHeaders  map[string]string
	groupID       string
	embeddingType string
}

type Option func(*config)

func WithModel(m string) Option { return func(c *config) { c.model = m } }

func WithAPIKey(k string) Option { return func(c *config) { c.apiKey = k } }

func WithBaseURL(u string) Option { return func(c *config) { c.baseURL = u } }

func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.httpClient = h } }

func WithTimeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

func WithRegion(r Region) Option { return func(c *config) { c.region = r } }

// WithExtraHeaders injects additional headers into every outbound request
// (chat/stream via the SDK, plus the raw-HTTP image/embed paths). Reserved
// headers (Authorization, Content-Type) are not overridable; extra headers
// are additive.
func WithExtraHeaders(h map[string]string) Option {
	return func(c *config) { c.extraHeaders = h }
}

// WithGroupID sets the MiniMax GroupId, passed as a query parameter on the
// embeddings request. Required for Embed; defaults to env MINIMAX_GROUP_ID.
func WithGroupID(id string) Option { return func(c *config) { c.groupID = id } }

// WithEmbeddingType sets the embedding "type" field: "db" (document, default)
// or "query".
func WithEmbeddingType(t string) Option { return func(c *config) { c.embeddingType = t } }

func baseURLForRegion(r Region) string {
	switch r {
	case RegionCN, RegionGlobal:
		return defaultBaseURL
	default:
		return defaultBaseURL
	}
}

func New(opts ...Option) (*MiniMax, error) {
	cfg := config{region: RegionGlobal}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.model == "" {
		return nil, errors.New("minimax: WithModel is required")
	}
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("MINIMAX_API_KEY")
	}
	if cfg.groupID == "" {
		cfg.groupID = os.Getenv("MINIMAX_GROUP_ID")
	}
	if cfg.embeddingType == "" {
		cfg.embeddingType = "db"
	}
	// P1-6/P1-23: default 60s request timeout when caller is silent.
	// Guards against indefinite hangs on idle HTTP connections. SDK
	// applies this per-request via option.WithRequestTimeout, so
	// streaming is NOT capped by a client-level Timeout — caller ctx
	// still governs long-running streams. Default lifted into
	// internal/compat so the 5/5 providers share the canonical value.
	cfg.timeout = compat.DefaultTimeout(cfg.timeout)

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = baseURLForRegion(cfg.region)
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

	client := sdk.NewClient(sdkOpts...)
	return &MiniMax{
		client:        &client,
		timeout:       cfg.timeout,
		baseURL:       baseURL,
		apiKey:        cfg.apiKey,
		httpClient:    cfg.httpClient,
		extraHeaders:  cfg.extraHeaders,
		groupID:       cfg.groupID,
		embeddingType: cfg.embeddingType,
		info: llm.ProviderInfo{
			Provider:     "minimax",
			Model:        cfg.model,
			Capabilities: capabilitiesForModel(cfg.model),
		},
	}, nil
}
