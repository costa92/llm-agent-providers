package volcengine

import (
	"errors"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/costa92/llm-agent-providers/internal/compat"
)

const defaultRegion = "cn-beijing"

type config struct {
	apiKey       string
	model        string
	baseURL      string
	region       string
	httpClient   *http.Client
	timeout      time.Duration
	dimensions   int
	extraHeaders map[string]string
}

// Option configures a Volcengine provider at construction time.
type Option func(*config)

// WithModel binds the (account-specific) Ark model id. Required.
func WithModel(m string) Option { return func(c *config) { c.model = m } }

// WithAPIKey sets the Ark API key. Falls back to ARK_API_KEY when empty.
func WithAPIKey(k string) Option { return func(c *config) { c.apiKey = k } }

// WithBaseURL overrides the Ark endpoint (also used to point tests at httptest).
func WithBaseURL(u string) Option { return func(c *config) { c.baseURL = u } }

// WithRegion sets the Ark region (default cn-beijing).
func WithRegion(r string) Option { return func(c *config) { c.region = r } }

// WithHTTPClient injects a custom *http.Client.
func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.httpClient = h } }

// WithTimeout sets the per-request timeout.
func WithTimeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

// WithDimensions overrides the embedding output dimensionality (embed models only).
func WithDimensions(n int) Option { return func(c *config) { c.dimensions = n } }

// WithExtraHeaders injects additional headers on every outbound request.
// Reserved headers (Authorization, Content-Type) are not overridable.
func WithExtraHeaders(h map[string]string) Option {
	return func(c *config) {
		c.extraHeaders = make(map[string]string, len(h))
		for k, v := range h {
			c.extraHeaders[k] = v
		}
	}
}

// capabilitiesForModel returns the K2 capability set for an Ark model id.
// Chat models (doubao-*-pro / doubao-*-lite / generic doubao chat) get Tools;
// doubao-seedream* get ImageGeneration; doubao-embedding* get Embeddings.
func capabilitiesForModel(model string) llm.Capabilities {
	switch {
	case strings.HasPrefix(model, "doubao-embedding"):
		return llm.Capabilities{Embeddings: true}
	case strings.HasPrefix(model, "doubao-seedream"):
		return llm.Capabilities{ImageGeneration: true}
	default:
		return llm.Capabilities{Tools: true}
	}
}

// defaultEmbedDimensions returns the native embedding dimensionality for an
// Ark embedding model, or 0 for non-embedding models.
func defaultEmbedDimensions(model string) int {
	switch {
	case strings.HasPrefix(model, "doubao-embedding-large-text"):
		return 4096
	case strings.HasPrefix(model, "doubao-embedding-text"):
		return 2560
	default:
		return 0
	}
}

// providerInfo builds the bound ProviderInfo.
func providerInfo(model string, caps llm.Capabilities) llm.ProviderInfo {
	return llm.ProviderInfo{
		Provider:     "volcengine",
		Model:        model,
		Capabilities: caps,
	}
}

// New constructs a Volcengine provider bound to one model.
func New(opts ...Option) (*Volcengine, error) {
	cfg := config{region: defaultRegion}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.model == "" {
		return nil, errors.New("volcengine: WithModel is required")
	}
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("ARK_API_KEY")
	}
	cfg.timeout = compat.DefaultTimeout(cfg.timeout)

	caps := capabilitiesForModel(cfg.model)

	embedDims := cfg.dimensions
	if embedDims == 0 {
		embedDims = defaultEmbedDimensions(cfg.model)
	}

	return &Volcengine{
		client:          newArkClient(cfg),
		info:            providerInfo(cfg.model, caps),
		timeout:         cfg.timeout,
		embedDimensions: embedDims,
		extraHeaders:    cfg.extraHeaders,
	}, nil
}
