package volcengine

import (
	"context"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/utils"
)

var (
	_ llm.ChatModel  = (*Volcengine)(nil)
	_ llm.ToolCaller = (*Volcengine)(nil)
	// _ llm.ImageGenerator and _ llm.Embedder asserted once GenerateImage
	// (Task 7) and Embed (Task 8) land.
)

// Volcengine is a 火山方舟 Ark adapter bound to a single model.
type Volcengine struct {
	client          *arkruntime.Client
	info            llm.ProviderInfo
	tools           []llm.Tool
	timeout         time.Duration
	embedDimensions int
	extraHeaders    map[string]string
}

// newArkClient builds the arkruntime client from config. WithRetryTimes(0)
// keeps our single-attempt policy consistent with the other adapters.
func newArkClient(cfg config) *arkruntime.Client {
	setters := []arkruntime.ConfigOption{
		arkruntime.WithRegion(cfg.region),
		arkruntime.WithRetryTimes(0),
	}
	if cfg.baseURL != "" {
		setters = append(setters, arkruntime.WithBaseUrl(cfg.baseURL))
	}
	if cfg.httpClient != nil {
		setters = append(setters, arkruntime.WithHTTPClient(cfg.httpClient))
	}
	if cfg.timeout > 0 {
		setters = append(setters, arkruntime.WithTimeout(cfg.timeout))
	}
	return arkruntime.NewClientWithApiKey(cfg.apiKey, setters...)
}

// requestOptions returns the per-request setters (custom headers).
func (v *Volcengine) requestOptions() []arkruntime.RequestOption {
	if len(v.extraHeaders) == 0 {
		return nil
	}
	return []arkruntime.RequestOption{arkruntime.WithCustomHeaders(v.extraHeaders)}
}

// Info returns the bound provider+model identity and capabilities.
func (v *Volcengine) Info() llm.ProviderInfo { return v.info }

// WithTools returns a new tool-bound ToolCaller (immutable; receiver unchanged).
func (v *Volcengine) WithTools(tools []llm.Tool) (llm.ToolCaller, error) {
	cp := *v
	cp.tools = append([]llm.Tool(nil), tools...)
	return &cp, nil
}

// Generate is implemented in this file (Task 4).
func (v *Volcengine) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	return llm.Response{}, io.EOF // placeholder replaced in Task 4
}

// Stream is implemented in this file (Task 5).
func (v *Volcengine) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	return nil, io.EOF // placeholder replaced in Task 5
}

// EmbedDimensions returns the bound embedding dimensionality, or 0 for
// non-embedding models.
func (v *Volcengine) EmbedDimensions() int { return v.embedDimensions }

// the following symbols are referenced by later tasks; declared here so the
// skeleton compiles before they are used.
var (
	_ = sort.Ints
	_ = sync.Mutex{}
	_ = (*utils.ChatCompletionStreamReader)(nil)
	_ = model.ChatMessageRoleUser
)
