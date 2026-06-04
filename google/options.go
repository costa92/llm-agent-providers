package google

import (
	"context"
	"errors"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/costa92/llm-agent-providers/internal/compat"
	"google.golang.org/genai"
)

type config struct {
	apiKey       string
	model        string
	baseURL      string
	httpClient   *http.Client
	timeout      time.Duration
	taskType     string
	dimensions   int
	extraHeaders map[string]string
}

// Option configures the Google provider at construction time.
type Option func(*config)

// WithModel binds the provider to one Gemini model id. Required.
func WithModel(m string) Option { return func(c *config) { c.model = m } }

// WithAPIKey sets the Gemini API key. Falls back to GEMINI_API_KEY then
// GOOGLE_API_KEY when unset.
func WithAPIKey(k string) Option { return func(c *config) { c.apiKey = k } }

// WithBaseURL overrides the API base URL (used for httptest fixtures).
func WithBaseURL(u string) Option { return func(c *config) { c.baseURL = u } }

// WithHTTPClient supplies a custom *http.Client.
func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.httpClient = h } }

// WithTimeout sets a per-request timeout; 0 uses the shared default.
func WithTimeout(d time.Duration) Option { return func(c *config) { c.timeout = d } }

// WithTaskType sets the embedding task type (e.g. RETRIEVAL_DOCUMENT);
// empty uses the model default. Embedding-only knob.
func WithTaskType(t string) Option { return func(c *config) { c.taskType = t } }

// WithDimensions sets the embedding output dimensionality (MRL truncation);
// 0 uses the model default. Embedding-only knob.
func WithDimensions(d int) Option { return func(c *config) { c.dimensions = d } }

// WithExtraHeaders injects additional HTTP headers on every outbound
// request (chat/stream/image/embed). Reserved headers (x-goog-api-key,
// Content-Type) are not overridable; extra headers are additive.
func WithExtraHeaders(h map[string]string) Option {
	return func(c *config) { c.extraHeaders = h }
}

// New constructs a Google provider bound to one model.
func New(opts ...Option) (*Google, error) {
	cfg := config{}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.model == "" {
		return nil, errors.New("google: WithModel is required")
	}
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	cfg.timeout = compat.DefaultTimeout(cfg.timeout)

	clientCfg := &genai.ClientConfig{
		APIKey:  cfg.apiKey,
		Backend: genai.BackendGeminiAPI,
	}
	if cfg.httpClient != nil {
		clientCfg.HTTPClient = cfg.httpClient
	}
	if cfg.baseURL != "" {
		clientCfg.HTTPOptions.BaseURL = cfg.baseURL
	}
	if len(cfg.extraHeaders) > 0 {
		h := http.Header{}
		for k, v := range cfg.extraHeaders {
			h.Set(k, v)
		}
		clientCfg.HTTPOptions.Headers = h
	}

	client, err := genai.NewClient(context.Background(), clientCfg)
	if err != nil {
		return nil, err
	}

	return &Google{
		client:     client,
		taskType:   cfg.taskType,
		dimensions: cfg.dimensions,
		timeout:    cfg.timeout,
		info: llm.ProviderInfo{
			Provider:     "google",
			Model:        cfg.model,
			Capabilities: capabilitiesForModel(cfg.model),
		},
	}, nil
}

// capabilitiesForModel binds capabilities to the (provider × model) tuple.
//
//   - Tools: gemini-* chat models (NOT image, NOT embedding variants).
//   - Vision: same as Tools — every Gemini generative chat model is multimodal
//     and accepts image input.
//   - ImageGeneration: gemini-*-image and imagen-* models.
//   - Embeddings: models whose id contains "embedding".
func capabilitiesForModel(model string) llm.Capabilities {
	return llm.Capabilities{
		Tools:           isChatModel(model),
		Vision:          isChatModel(model),
		ImageGeneration: isImageModel(model),
		Embeddings:      isEmbedModel(model),
	}
}

func isImageModel(model string) bool {
	return strings.HasPrefix(model, "imagen") ||
		(strings.HasPrefix(model, "gemini") && strings.HasSuffix(model, "-image"))
}

func isEmbedModel(model string) bool {
	return strings.Contains(model, "embedding")
}

// isChatModel is a gemini-* generative model that is neither an image nor
// an embedding variant.
func isChatModel(model string) bool {
	return strings.HasPrefix(model, "gemini") && !isImageModel(model) && !isEmbedModel(model)
}
