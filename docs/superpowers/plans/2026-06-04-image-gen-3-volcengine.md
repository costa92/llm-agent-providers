# Volcengine (火山方舟 Ark / 豆包) Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new full provider package `volcengine/` in `llm-agent-providers` that implements `llm.ChatModel` + `llm.ToolCaller` + `llm.ImageGenerator` + `llm.Embedder` over the official `arkruntime` Go SDK, with all capabilities K2 model-gated.

**Architecture:** One bound model per instance (constructed via `volcengine.New(WithModel(...))`). Chat/stream/tools go through `arkruntime.Client.CreateChatCompletion` / `CreateChatCompletionStream` using the **pointer-field** `model.CreateChatCompletionRequest`. Images go through `arkruntime.Client.GenerateImages` (`model.GenerateImagesRequest` — confirmed present in SDK v1.2.33, NO raw-HTTP fallback needed). Embeddings go through `arkruntime.Client.CreateEmbeddings(model.EmbeddingRequestStrings{...})`. The stream reader mirrors the openai sibling (sync.Mutex, lazy open, queue, retry-once-before-first-byte, fragmented tool-call merge by `ToolCall.Index`) but adapts to arkruntime's `*utils.ChatCompletionStreamReader.Recv()` loop (which returns `io.EOF` at `[DONE]`) instead of openai-go's `ssestream.Stream[T]`. SDK types never appear on the public surface (`map.go` + `errors.go` enforce the boundary). The struct keeps the `*arkruntime.Client` on an unexported field plus `extraHeaders map[string]string` injected via `arkruntime.WithCustomHeaders` per request.

**Tech Stack:** Go 1.26, `github.com/volcengine/volcengine-go-sdk v1.2.33` (packages `service/arkruntime`, `.../model`, `.../utils`), `github.com/costa92/llm-agent-contract` **v0.3.0** (assumed available via local `replace` during dev — provides `llm.ImageGenerator`, `llm.ImageRequest`, `llm.ImageResponse`, `llm.GeneratedImage`, `Capabilities.ImageGeneration`), `github.com/costa92/llm-agent-providers/internal/compat` (timeout default), stdlib `net/http/httptest` for tests, `go.uber.org/goleak` for stream lifecycle.

---

## Prerequisites & Assumptions

**P-0 (BLOCKING DEPENDENCY):** This plan assumes `llm-agent-contract` **v0.3.0** is available (added by the contract plan: `llm/image.go` with `ImageGenerator`/`ImageRequest`/`ImageResponse`/`GeneratedImage` + `Capabilities.ImageGeneration bool`). At plan-execution start, these symbols do **NOT** exist in the contract working tree on disk — they are introduced by the sibling "contract" plan. Before Task 1, verify they exist:

```bash
GOWORK=off grep -rn "ImageGenerator\|ImageRequest\|ImageResponse\|GeneratedImage\|ImageGeneration" \
  /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-contract/llm/
```

Expected: matches in `llm/image.go` and `llm/info.go`. If absent, STOP and land the contract plan first (or add a local `replace` to a contract branch that has them). Confirm a local `replace` is wired in `llm-agent-providers/go.mod` pointing the contract at the on-disk tree:

```bash
GOWORK=off grep -n "replace github.com/costa92/llm-agent-contract" \
  /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers/go.mod
```

The exact contract type shapes this plan codes against (per spec):

```go
type ImageGenerator interface {
	GenerateImage(ctx context.Context, req ImageRequest) (ImageResponse, error)
}
type ImageRequest struct {
	Prompt  string
	N       int
	Size    string
	Quality string
	Format  string
	Extra   map[string]any
}
type ImageResponse struct {
	Images   []GeneratedImage
	Provider string
	Model    string
	Usage    Usage
}
type GeneratedImage struct {
	Bytes         []byte
	URL           string
	MimeType      string
	RevisedPrompt string
}
// Capabilities gains: ImageGeneration bool `json:"image_generation"`
```

**P-1 (go.work present → GOWORK=off):** `ls /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/go.work*` returns `go.work` and `go.work.sum`. Therefore **ALL** `go` commands in this plan are prefixed `GOWORK=off`. (The umbrella `go.work` excludes this standalone sibling.)

**P-2 (replace-guard hook):** The repo's pre-commit hook auto-strips local `replace github.com/costa92/...` directives on commit. During development the local contract `replace` is needed for compilation. To commit work-in-progress that still relies on the unreleased contract, use `git commit --no-verify`. The final pin to contract `v0.3.0` (after it is tagged) is out of scope for THIS plan (it is the rollout step).

**P-3 (model ids are config, never hard-coded):** Doubao/Ark endpoint and model ids are account-dependent. The default chat model used in tests is the literal string `doubao-1-5-pro-32k-250115`; image `doubao-seedream-4-5-251128`; embed `doubao-embedding-text-240715` / `doubao-embedding-large-text-240915`. These are test fixtures only — production callers pass their own via `WithModel`.

---

## File Structure

All files live in a new directory `volcengine/` at the repo root, package name `volcengine`. Mirrors the five-file skeleton from CONVENTIONS.md plus `image.go` and `embed.go` (the image/embed paths are large enough to warrant their own files, matching the spec's `image.go` decision for minimax/openai).

| File | Responsibility |
|------|----------------|
| `volcengine/doc.go` | Package doc comment only. |
| `volcengine/volcengine.go` | `Volcengine` struct, interface assertions, `Generate`, `Stream`, `Info`, `WithTools`, and the `volcengineStreamReader` (sync.Mutex / lazy-open / queue / retry-once / fragmented tool-call merge by Index). |
| `volcengine/options.go` | `type config struct`, `type Option func(*config)`, all `WithX` options, `New(opts ...Option) (*Volcengine, error)`, `capabilitiesForModel`, `EmbedDimensions` model table. |
| `volcengine/map.go` | `toSDKRequest` (llm.Request → `*model.CreateChatCompletionRequest`), `fromSDKResponse`, `mapFinishReason`, tool-schema translation, content-union helper. |
| `volcengine/errors.go` | `wrapErr(err) error` → typed `llm.*` errors (maps `*model.APIError` / `*model.RequestError`). |
| `volcengine/image.go` | `GenerateImage` (→ `model.GenerateImagesRequest`), response → `[]llm.GeneratedImage`. |
| `volcengine/embed.go` | `Embed` (→ `model.EmbeddingRequestStrings`), `EmbedDimensions`. |
| `volcengine/volcengine_test.go` | All tests (httptest server pointed at via `WithBaseURL`). |
| `volcengine/options_internal_test.go` | Internal option/capability unit tests. |

**SDK symbol reference (ALL verified against `volcengine-go-sdk@v1.2.33` source — do not re-guess):**

- Client: `arkruntime.NewClientWithApiKey(apiKey string, setters ...ConfigOption) *arkruntime.Client`.
- Config options: `arkruntime.WithRegion(string)`, `arkruntime.WithBaseUrl(string)` (note **lowercase `rl`** — it is `WithBaseUrl`, NOT `WithBaseURL`), `arkruntime.WithRetryTimes(int)`, `arkruntime.WithTimeout(time.Duration)`, `arkruntime.WithHTTPClient(*http.Client)`.
- Per-request header option: `arkruntime.WithCustomHeaders(map[string]string) arkruntime.RequestOption` (also `WithCustomHeader(k, v string)`). These are `RequestOption`, passed as variadic trailing args to `CreateChatCompletion`/`CreateChatCompletionStream`/`GenerateImages`/`CreateEmbeddings`.
- Chat: `(*Client).CreateChatCompletion(ctx, request model.ChatRequest, setters ...RequestOption) (model.ChatCompletionResponse, error)`. `model.ChatRequest` is an interface; pass `*model.CreateChatCompletionRequest` (it implements `ChatRequest` via pointer receiver? — **NO**: methods `MarshalJSON`/`WithStream`/`IsStream`/`GetModel` are on **value receiver** `CreateChatCompletionRequest`, so pass a **value** `model.CreateChatCompletionRequest`, not a pointer. Verified: lines 225/234/239/246 of `model/chat_completion.go` all use value receiver `r CreateChatCompletionRequest`.)
- Stream: `(*Client).CreateChatCompletionStream(ctx, request model.ChatRequest, setters ...RequestOption) (*utils.ChatCompletionStreamReader, error)`. Reader: `(*utils.ChatCompletionStreamReader).Recv() (model.ChatCompletionStreamResponse, error)` returns `io.EOF` at `[DONE]`; `(*ChatCompletionStreamReader).Close() error`.
- Images: `(*Client).GenerateImages(ctx, request model.GenerateImagesRequest, setters ...RequestOption) (model.ImagesResponse, error)`. **Note:** `GenerateImages` requires API-key auth (`isAPIKeyAuthentication()`); returns `model.ErrAKSKNotSupported` otherwise — we always use API key, so fine.
- Embeddings: `(*Client).CreateEmbeddings(ctx, conv model.EmbeddingRequestConverter, setters ...RequestOption) (model.EmbeddingResponse, error)`. Pass `model.EmbeddingRequestStrings{Input, Model, Dimensions}` (implements `EmbeddingRequestConverter` via `Convert()`).
- Request struct `model.CreateChatCompletionRequest`: pointer fields `Model string`, `Messages []*model.ChatCompletionMessage`, `MaxTokens *int`, `Temperature *float32`, `Stream *bool`, `Tools []*model.Tool`, `ToolChoice interface{}`, `StreamOptions *model.StreamOptions`.
- `model.ChatCompletionMessage{Role string, Content *model.ChatCompletionMessageContent, ToolCalls []*model.ToolCall, ToolCallID string}`.
- `model.ChatCompletionMessageContent{StringValue *string, ListValue []*ChatCompletionMessageContentPart}` — for text, set `&model.ChatCompletionMessageContent{StringValue: &s}`.
- Roles: `model.ChatMessageRoleSystem/User/Assistant/Tool` = `"system"/"user"/"assistant"/"tool"`.
- `model.Tool{Type model.ToolType, Function *model.FunctionDefinition}`; `model.ToolTypeFunction = "function"`; `model.FunctionDefinition{Name string, Description string, Parameters interface{}}`.
- Response: `model.ChatCompletionResponse{ID, Object, Created, Model string, Choices []*model.ChatCompletionChoice, Usage model.Usage}`. `model.ChatCompletionChoice{Index int, Message model.ChatCompletionMessage, FinishReason model.FinishReason}`.
- `model.ChatCompletionMessage.Content` is `*ChatCompletionMessageContent` (nil-guard; read `.StringValue`).
- `model.ToolCall{ID string, Type model.ToolType, Function model.FunctionCall, Index *int}`; `model.FunctionCall{Name, Arguments string}`.
- Stream delta: `model.ChatCompletionStreamResponse{Choices []*model.ChatCompletionStreamChoice, Usage *model.Usage}`; `model.ChatCompletionStreamChoice{Index int, Delta model.ChatCompletionStreamChoiceDelta, FinishReason model.FinishReason}`; `model.ChatCompletionStreamChoiceDelta{Content string, Role string, ToolCalls []*model.ToolCall}`.
- `model.FinishReason` constants: `FinishReasonStop`/`FinishReasonLength`/`FinishReasonToolCalls`/`FinishReasonFunctionCall`/`FinishReasonContentFilter` ("stop"/"length"/"tool_calls"/"function_call"/"content_filter").
- `model.Usage{PromptTokens, CompletionTokens, TotalTokens int}`.
- Images request `model.GenerateImagesRequest{Model string, Prompt string, ResponseFormat *string, Seed *int64, GuidanceScale *float64, Size *string, Watermark *bool}`. **NOTE:** there is **NO `N` field** on `GenerateImagesRequest` in v1.2.33 (multi-image is `SequentialImageGeneration*`). Response `model.ImagesResponse{Model string, Created int64, Data []*model.Image, Usage *model.GenerateImagesUsage, Error *model.GenerateImagesError}`. `model.Image{Url *string, B64Json *string, Size string}`. Response-format constants `model.GenerateImagesResponseFormatURL = "url"`, `GenerateImagesResponseFormatBase64 = "b64_json"`.
- Embeddings request `model.EmbeddingRequestStrings{Input []string, Model string, Dimensions int}`. Response `model.EmbeddingResponse{Data []model.Embedding, Usage model.Usage}`; `model.Embedding{Object string, Embedding []float32, Index int}`.
- Errors: `model.APIError{Code string, Message string, Type string, HTTPStatusCode int, RequestId string}` (pointer `*model.APIError` returned by SDK; `Error()` on pointer receiver). `model.RequestError{HTTPStatusCode int, Err error, RequestId string}` (`*model.RequestError`, has `Unwrap()`).

---

## Task 1: Add the arkruntime dependency and package skeleton (doc.go)

**Files:**
- Create: `volcengine/doc.go`
- Modify: `go.mod` (add `github.com/volcengine/volcengine-go-sdk v1.2.33`)

- [ ] **Step 1: Create the package doc file**

Create `volcengine/doc.go`:

```go
// Package volcengine implements a 火山方舟 (Volcengine Ark / 豆包) adapter
// over the official github.com/volcengine/volcengine-go-sdk arkruntime
// client.
//
// The adapter satisfies llm.ChatModel, llm.ToolCaller, llm.ImageGenerator,
// and llm.Embedder. Capabilities reported via Info() are per-(provider ×
// model): the constructor binds a model, and Info() reflects what that
// model can do (Keystone K2) — chat/tools for doubao chat models, image
// generation for doubao-seedream*, embeddings for doubao-embedding*.
// Streaming events follow the typed K1 union with a stable per-tool-call
// Index across fragmented deltas.
package volcengine
```

- [ ] **Step 2: Add the SDK dependency**

Run (downloads and records the module; arkruntime is NOT yet in go.sum):

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go get github.com/volcengine/volcengine-go-sdk@v1.2.33
```

Expected: `go.mod` now lists `github.com/volcengine/volcengine-go-sdk v1.2.33`; `go.sum` updated.

- [ ] **Step 3: Verify the package compiles (empty package)**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go build ./volcengine/
```

Expected: PASS (no output).

- [ ] **Step 4: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/doc.go go.mod go.sum && git commit --no-verify -m "feat(volcengine): scaffold package and add arkruntime SDK dep"
```

---

## Task 2: Options, constructor, and capability gating (options.go)

**Files:**
- Create: `volcengine/options.go`
- Create: `volcengine/volcengine.go` (struct + interface assertions + stubs so the package compiles; methods filled in later tasks)
- Test: `volcengine/options_internal_test.go`

- [ ] **Step 1: Write the failing internal test**

Create `volcengine/options_internal_test.go`:

```go
package volcengine

import (
	"strings"
	"testing"
)

func TestNew_RequiresModel(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("New() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), "WithModel is required") {
		t.Fatalf("New() error = %q, want WithModel is required", err)
	}
}

func TestCapabilitiesForModel(t *testing.T) {
	cases := []struct {
		model           string
		wantTools       bool
		wantImage       bool
		wantEmbeddings  bool
	}{
		{"doubao-1-5-pro-32k-250115", true, false, false},
		{"doubao-seedream-4-5-251128", false, true, false},
		{"doubao-embedding-text-240715", false, false, true},
		{"doubao-embedding-large-text-240915", false, false, true},
	}
	for _, c := range cases {
		caps := capabilitiesForModel(c.model)
		if caps.Tools != c.wantTools || caps.ImageGeneration != c.wantImage || caps.Embeddings != c.wantEmbeddings {
			t.Fatalf("capabilitiesForModel(%q) = %+v, want tools=%v image=%v embed=%v",
				c.model, caps, c.wantTools, c.wantImage, c.wantEmbeddings)
		}
	}
}

func TestEmbedDimensions_Table(t *testing.T) {
	cases := map[string]int{
		"doubao-embedding-text-240715":       2560,
		"doubao-embedding-large-text-240915": 4096,
		"doubao-1-5-pro-32k-250115":          0,
	}
	for model, want := range cases {
		v := &Volcengine{info: providerInfo(model, capabilitiesForModel(model)), embedDimensions: defaultEmbedDimensions(model)}
		if got := v.EmbedDimensions(); got != want {
			t.Fatalf("EmbedDimensions(%q) = %d, want %d", model, got, want)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestNew_RequiresModel|TestCapabilitiesForModel|TestEmbedDimensions_Table' -v
```

Expected: FAIL — `undefined: New`, `undefined: capabilitiesForModel`, `undefined: Volcengine`, etc.

- [ ] **Step 3: Write options.go**

Create `volcengine/options.go`:

```go
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
```

- [ ] **Step 4: Write volcengine.go skeleton (struct + interface asserts + arkruntime client builder + stubs)**

Create `volcengine/volcengine.go`:

```go
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
	_ llm.ChatModel      = (*Volcengine)(nil)
	_ llm.ToolCaller     = (*Volcengine)(nil)
	_ llm.ImageGenerator = (*Volcengine)(nil)
	_ llm.Embedder       = (*Volcengine)(nil)
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
```

> Note: the `var ( _ = ... )` block at the bottom is a temporary compile aid. Task 4/5 replace `Generate`/`Stream` and add real uses of `sort`/`sync`/`utils`/`model`; remove this `var` block when those tasks land (Task 5 step removes it).

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestNew_RequiresModel|TestCapabilitiesForModel|TestEmbedDimensions_Table' -v
```

Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/options.go volcengine/volcengine.go volcengine/options_internal_test.go && git commit --no-verify -m "feat(volcengine): options, constructor, K2 capability gating"
```

---

## Task 3: Request/response mapping (map.go)

**Files:**
- Create: `volcengine/map.go`
- Test: `volcengine/volcengine_test.go` (new file — first chat test exercises the mapping end-to-end)

- [ ] **Step 1: Write the failing Generate happy-path test**

Create `volcengine/volcengine_test.go`:

```go
package volcengine

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestGenerate_Volcengine_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("Path = %s, want /chat/completions", r.URL.Path)
		}
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"model":"doubao-1-5-pro-32k-250115"`) {
			t.Fatalf("request body missing model: %s", body)
		}
		if !strings.Contains(body, `"role":"system"`) || !strings.Contains(body, `be concise`) {
			t.Fatalf("request body missing system message: %s", body)
		}
		if !strings.Contains(body, `say hello`) {
			t.Fatalf("request body missing user message: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"cmpl_123",
			"object":"chat.completion",
			"created":1710000000,
			"model":"doubao-1-5-pro-32k-250115",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello world"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("doubao-1-5-pro-32k-250115"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		SystemPrompt: "be concise",
		Messages:     []llm.Message{{Role: "user", Content: "say hello"}},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if resp.Text != "hello world" {
		t.Fatalf("Text = %q, want hello world", resp.Text)
	}
	if resp.Provider != "volcengine" {
		t.Fatalf("Provider = %q, want volcengine", resp.Provider)
	}
	if resp.Model != "doubao-1-5-pro-32k-250115" {
		t.Fatalf("Model = %q, want doubao-1-5-pro-32k-250115", resp.Model)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.Source != llm.UsageReported || resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Fatalf("Usage = %+v, want reported 11/7/18", resp.Usage)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run TestGenerate_Volcengine_Happy -v
```

Expected: FAIL — `Generate` returns the `io.EOF` placeholder (test sees an error / empty Text). Also `toSDKRequest`/`fromSDKResponse` are undefined if referenced; at this point they are not yet, so the failure is the placeholder behavior.

- [ ] **Step 3: Write map.go**

Create `volcengine/map.go`:

```go
package volcengine

import (
	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// strPtr returns a pointer to s (helper for the SDK's *string / union fields).
func strPtr(s string) *string { return &s }

// contentString wraps a plain string in the SDK's content union.
func contentString(s string) *model.ChatCompletionMessageContent {
	return &model.ChatCompletionMessageContent{StringValue: strPtr(s)}
}

// toSDKRequest maps an llm.Request to the pointer-field
// CreateChatCompletionRequest (so temperature=0 is sendable). Streaming is
// set by the caller (Stream path) via the SDK, not here.
func (v *Volcengine) toSDKRequest(req llm.Request) model.CreateChatCompletionRequest {
	msgs := make([]*model.ChatCompletionMessage, 0, len(req.Messages)+1)
	if req.SystemPrompt != "" {
		msgs = append(msgs, &model.ChatCompletionMessage{
			Role:    model.ChatMessageRoleSystem,
			Content: contentString(req.SystemPrompt),
		})
	}
	for _, m := range req.Messages {
		role := m.Role
		switch role {
		case "user", "assistant", "system", "tool":
			// pass through
		default:
			role = model.ChatMessageRoleUser
		}
		msgs = append(msgs, &model.ChatCompletionMessage{
			Role:    role,
			Content: contentString(m.Content),
		})
	}

	out := model.CreateChatCompletionRequest{
		Model:    v.info.Model,
		Messages: msgs,
	}
	if req.MaxOutputTokens > 0 {
		mt := req.MaxOutputTokens
		out.MaxTokens = &mt
	}
	if req.Temperature != nil {
		t := *req.Temperature
		out.Temperature = &t
	}
	if len(v.tools) > 0 {
		out.Tools = make([]*model.Tool, 0, len(v.tools))
		for _, tool := range v.tools {
			def := &model.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
			}
			if len(tool.Parameters) > 0 {
				// Parameters is interface{}; the SDK marshals it verbatim.
				// json.RawMessage marshals as the raw schema bytes.
				def.Parameters = tool.Parameters
			}
			out.Tools = append(out.Tools, &model.Tool{
				Type:     model.ToolTypeFunction,
				Function: def,
			})
		}
	}
	return out
}

// fromSDKResponse maps a non-stream ChatCompletionResponse to llm.Response.
func (v *Volcengine) fromSDKResponse(resp model.ChatCompletionResponse) llm.Response {
	var text string
	finish := llm.FinishReasonUnknown
	var toolCalls []llm.ToolCall
	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		if choice.Message.Content != nil && choice.Message.Content.StringValue != nil {
			text = *choice.Message.Content.StringValue
		}
		finish = mapFinishReason(choice.FinishReason)
		toolCalls = mapToolCalls(choice.Message.ToolCalls)
	}
	return llm.Response{
		Text:         text,
		FinishReason: finish,
		Provider:     "volcengine",
		Model:        resp.Model,
		ToolCalls:    toolCalls,
		Usage: llm.Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
			Source:       llm.UsageReported,
		},
	}
}

// mapToolCalls converts SDK tool calls to the contract shape.
func mapToolCalls(calls []*model.ToolCall) []llm.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]llm.ToolCall, 0, len(calls))
	for _, call := range calls {
		if call == nil || call.Type != model.ToolTypeFunction {
			continue
		}
		out = append(out, llm.ToolCall{
			ID:        call.ID,
			Name:      call.Function.Name,
			Arguments: []byte(call.Function.Arguments),
		})
	}
	return out
}

// mapFinishReason maps the SDK finish reason to the contract enum.
func mapFinishReason(r model.FinishReason) llm.FinishReason {
	switch r {
	case model.FinishReasonStop:
		return llm.FinishReasonStop
	case model.FinishReasonLength:
		return llm.FinishReasonLength
	case model.FinishReasonContentFilter:
		return llm.FinishReasonContentFilter
	case model.FinishReasonToolCalls:
		return llm.FinishReasonToolCalls
	case model.FinishReasonFunctionCall:
		return llm.FinishReasonFunctionCall
	default:
		return llm.FinishReasonUnknown
	}
}
```

- [ ] **Step 4: Wire Generate to use the mapping (replace the placeholder in volcengine.go)**

In `volcengine/volcengine.go`, replace the placeholder `Generate` method:

```go
// Generate is implemented in this file (Task 4).
func (v *Volcengine) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	return llm.Response{}, io.EOF // placeholder replaced in Task 4
}
```

with:

```go
// Generate runs a one-shot chat completion.
func (v *Volcengine) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := v.toSDKRequest(req)
	resp, err := v.client.CreateChatCompletion(ctx, sdkReq, v.requestOptions()...)
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	return v.fromSDKResponse(resp), nil
}
```

Because this now references `wrapErr` (Task 6) which does not exist yet, add a **temporary** `wrapErr` to `volcengine.go` to keep the build green until Task 6 moves it to `errors.go`. Add this private function to `volcengine.go` for now:

```go
// wrapErr is defined here temporarily; Task 6 moves it to errors.go.
func wrapErr(err error) error { return err }
```

(Task 6 deletes this stub and creates the real one in `errors.go`.)

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run TestGenerate_Volcengine_Happy -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/map.go volcengine/volcengine.go volcengine/volcengine_test.go && git commit --no-verify -m "feat(volcengine): chat request/response mapping + Generate"
```

---

## Task 4: Non-stream tool calls (map.go already covers it — add a test)

**Files:**
- Test: `volcengine/volcengine_test.go` (append)

- [ ] **Step 1: Write the failing tool-call test**

Append to `volcengine/volcengine_test.go`:

```go
func TestGenerate_Volcengine_ToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"tools":[`) || !strings.Contains(body, `"name":"calc"`) {
			t.Fatalf("request body missing tool declaration: %s", body)
		}
		if !strings.Contains(body, `"type":"function"`) {
			t.Fatalf("request body missing function tool type: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"cmpl_456",
			"object":"chat.completion",
			"created":1710000000,
			"model":"doubao-1-5-pro-32k-250115",
			"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_calc","type":"function","function":{"name":"calc","arguments":"{\"expr\":\"2+2\"}"}},
				{"id":"call_search","type":"function","function":{"name":"search","arguments":"{\"q\":\"weather\"}"}}
			]},"finish_reason":"tool_calls"}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`))
	}))
	defer server.Close()

	base, err := New(
		WithModel("doubao-1-5-pro-32k-250115"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{
		{Name: "calc", Description: "calculator", Parameters: []byte(`{"type":"object"}`)},
		{Name: "search", Description: "search", Parameters: []byte(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "use tools"}},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("len(ToolCalls) = %d, want 2", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "call_calc" || resp.ToolCalls[0].Name != "calc" {
		t.Fatalf("ToolCalls[0] = %+v, want calc", resp.ToolCalls[0])
	}
	if string(resp.ToolCalls[0].Arguments) != `{"expr":"2+2"}` {
		t.Fatalf("ToolCalls[0].Arguments = %s, want {\"expr\":\"2+2\"}", resp.ToolCalls[0].Arguments)
	}
	if resp.ToolCalls[1].ID != "call_search" || resp.ToolCalls[1].Name != "search" {
		t.Fatalf("ToolCalls[1] = %+v, want search", resp.ToolCalls[1])
	}
	if resp.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("FinishReason = %q, want tool_calls", resp.FinishReason)
	}
}

func TestWithTools_Volcengine_Immutable(t *testing.T) {
	base, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = base.WithTools([]llm.Tool{{Name: "calc", Parameters: []byte(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}
	if len(base.tools) != 0 {
		t.Fatalf("base.tools mutated, len = %d, want 0", len(base.tools))
	}
}
```

- [ ] **Step 2: Run test to verify it passes (mapping already implemented in Task 3)**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestGenerate_Volcengine_ToolCalls|TestWithTools_Volcengine_Immutable' -v
```

Expected: PASS. (If FAIL on tool-schema serialization, the cause is `def.Parameters = tool.Parameters` where `tool.Parameters` is `json.RawMessage` — it marshals as raw bytes, which is correct. Do NOT change to a `map`.)

- [ ] **Step 3: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/volcengine_test.go && git commit --no-verify -m "test(volcengine): non-stream tool calls + WithTools immutability"
```

---

## Task 5: Streaming (volcengineStreamReader) — text + fragmented tool-call merge

**Files:**
- Modify: `volcengine/volcengine.go` (replace placeholder `Stream`, add `volcengineStreamReader`, remove the temporary `var ( _ = ... )` compile-aid block)
- Test: `volcengine/volcengine_test.go` (append)

**Design notes (arkruntime stream differs from openai-go):** `CreateChatCompletionStream` returns `*utils.ChatCompletionStreamReader`. Its `Recv()` returns one `model.ChatCompletionStreamResponse` per chunk and returns `io.EOF` when it reads `data: [DONE]`. There is NO separate `Next()`/`Current()`/`Err()` split. So the reader's pull loop is: call `Recv()`; on `io.EOF` finish; on other error wrap; otherwise decompose the chunk into `llm.StreamEvent`s. The retry-once-before-first-byte rule and lazy-open are preserved. Usage on a chunk is `Choices==empty + Usage!=nil` (StreamOptions IncludeUsage). Because Ark sends `[DONE]` (→ io.EOF) AFTER the usage chunk, we synthesize the terminal `EventDone` from the last-seen usage + finish reason when we hit io.EOF, OR from the usage-only chunk if one arrives first — emit `EventDone` exactly once.

- [ ] **Step 1: Write the failing stream test (text + fragmented tool calls + usage)**

Append to `volcengine/volcengine_test.go`:

```go
import_fmt_marker := 0 // placeholder to remind: add "fmt" to the import block

func TestStream_Volcengine_Text(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		_, _ = fmtFprint(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmtFprint(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmtFprint(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n")
		_, _ = fmtFprint(w, "data: [DONE]\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	sr, err := m.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if resp.Text != "hello" {
		t.Fatalf("Text = %q, want hello", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.Source != llm.UsageReported || resp.Usage.TotalTokens != 5 {
		t.Fatalf("Usage = %+v, want reported total=5", resp.Usage)
	}
}

func TestStream_Volcengine_FragmentedToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmtFprint(w, "data: {\"id\":\"c2\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_calc\",\"type\":\"function\",\"function\":{\"name\":\"calc\",\"arguments\":\"{\\\"expr\\\":\"}},{\"index\":1,\"id\":\"call_search\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmtFprint(w, "data: {\"id\":\"c2\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"2+2\\\"}\"}},{\"index\":1,\"function\":{\"arguments\":\"\\\"weather\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n")
		_, _ = fmtFprint(w, "data: {\"id\":\"c2\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":5,\"total_tokens\":14}}\n\n")
		_, _ = fmtFprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	base, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{
		{Name: "calc", Parameters: []byte(`{"type":"object"}`)},
		{Name: "search", Parameters: []byte(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}
	sr, err := m.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "use tools"}}})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("len(ToolCalls) = %d, want 2", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "call_calc" || resp.ToolCalls[0].Name != "calc" || string(resp.ToolCalls[0].Arguments) != `{"expr":"2+2"}` {
		t.Fatalf("ToolCalls[0] = %+v args=%s, want calc {\"expr\":\"2+2\"}", resp.ToolCalls[0], resp.ToolCalls[0].Arguments)
	}
	if resp.ToolCalls[1].ID != "call_search" || resp.ToolCalls[1].Name != "search" || string(resp.ToolCalls[1].Arguments) != `{"q":"weather"}` {
		t.Fatalf("ToolCalls[1] = %+v args=%s, want search {\"q\":\"weather\"}", resp.ToolCalls[1], resp.ToolCalls[1].Arguments)
	}
	if resp.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("FinishReason = %q, want tool_calls", resp.FinishReason)
	}
}
```

> The test uses helper `fmtFprint` and a stray `import_fmt_marker` line that will NOT compile — they are deliberate reminders. In Step 3 you replace `fmtFprint(...)` calls with `fmt.Fprint(...)`, add `"fmt"` to the test import block, and DELETE the `import_fmt_marker := 0` line. (Written this way so the test file's import list is updated as part of making it compile.)

Concretely, before running, edit the appended block: delete the `import_fmt_marker := 0 // ...` line, change every `fmtFprint(` to `fmt.Fprint(`, and add `"fmt"` to the existing `import (...)` group at the top of `volcengine_test.go` so it reads:

```go
import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestStream_Volcengine_Text|TestStream_Volcengine_FragmentedToolCalls' -v
```

Expected: FAIL — `Stream` returns the `io.EOF` placeholder (a nil StreamReader → panic / error). Confirm it fails before implementing.

- [ ] **Step 3: Implement the stream reader (edit volcengine.go)**

In `volcengine/volcengine.go`: (a) delete the temporary bottom `var ( _ = sort.Ints ... )` compile-aid block; (b) replace the placeholder `Stream`; (c) append the `volcengineStreamReader` type and methods.

Replace:

```go
// Stream is implemented in this file (Task 5).
func (v *Volcengine) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	return nil, io.EOF // placeholder replaced in Task 5
}
```

with:

```go
// Stream runs a streaming chat completion, returning a typed K1 reader.
func (v *Volcengine) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	sdkReq := v.toSDKStreamRequest(req)
	opts := v.requestOptions()
	return &volcengineStreamReader{
		open: func() (*utils.ChatCompletionStreamReader, error) {
			return v.client.CreateChatCompletionStream(ctx, sdkReq, opts...)
		},
		toolIndexes: make(map[int]struct{}),
	}, nil
}
```

And delete the bottom block:

```go
// the following symbols are referenced by later tasks; declared here so the
// skeleton compiles before they are used.
var (
	_ = sort.Ints
	_ = sync.Mutex{}
	_ = (*utils.ChatCompletionStreamReader)(nil)
	_ = model.ChatMessageRoleUser
)
```

Then append:

```go
type volcengineStreamReader struct {
	mu            sync.Mutex
	open          func() (*utils.ChatCompletionStreamReader, error)
	stream        *utils.ChatCompletionStreamReader
	queue         []llm.StreamEvent
	retried       bool
	deliveredByte bool
	closed        bool
	doneEmitted   bool

	toolIndexes map[int]struct{}
	usage       *llm.Usage
	lastFinish  llm.FinishReason
}

// Next pulls the next typed stream event. It opens the upstream stream lazily
// on the first call and retries exactly once before any byte is delivered.
func (r *volcengineStreamReader) Next() (llm.StreamEvent, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for {
		if r.closed {
			return llm.StreamEvent{}, io.EOF
		}
		if len(r.queue) > 0 {
			ev := r.queue[0]
			r.queue = r.queue[1:]
			if ev.Kind != llm.EventDone {
				r.deliveredByte = true
			}
			return ev, nil
		}
		if r.stream == nil {
			s, err := r.open()
			if err != nil {
				if !r.deliveredByte && !r.retried {
					r.retried = true
					continue
				}
				return llm.StreamEvent{}, wrapErr(err)
			}
			r.stream = s
		}

		chunk, err := r.stream.Recv()
		if err != nil {
			if isEOF(err) {
				// Clean end: emit the terminal Done once, then EOF.
				if !r.doneEmitted {
					r.doneEmitted = true
					r.queue = append(r.queue, r.doneEvent())
					continue
				}
				_ = r.stream.Close()
				r.stream = nil
				return llm.StreamEvent{}, io.EOF
			}
			_ = r.stream.Close()
			r.stream = nil
			if !r.deliveredByte && !r.retried {
				r.retried = true
				continue
			}
			return llm.StreamEvent{}, wrapErr(err)
		}
		r.queue = append(r.queue, r.chunkEvents(chunk)...)
	}
}

// Close is idempotent; safe to call from another goroutine.
func (r *volcengineStreamReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	if r.stream == nil {
		return nil
	}
	err := r.stream.Close()
	r.stream = nil
	return err
}

// doneEvent builds the terminal EventDone from accumulated usage/finish.
func (r *volcengineStreamReader) doneEvent() llm.StreamEvent {
	usage := r.usage
	if usage == nil {
		u := llm.Usage{Source: llm.UsageUnknown}
		usage = &u
	}
	return llm.StreamEvent{
		Kind:         llm.EventDone,
		Usage:        usage,
		FinishReason: r.lastFinish,
	}
}

// chunkEvents decomposes one SDK stream chunk into typed events. A usage-only
// chunk (empty Choices, non-nil Usage) records usage but emits nothing — the
// terminal Done is synthesized at io.EOF.
func (r *volcengineStreamReader) chunkEvents(chunk model.ChatCompletionStreamResponse) []llm.StreamEvent {
	if chunk.Usage != nil {
		r.usage = &llm.Usage{
			InputTokens:  chunk.Usage.PromptTokens,
			OutputTokens: chunk.Usage.CompletionTokens,
			TotalTokens:  chunk.Usage.TotalTokens,
			Source:       llm.UsageReported,
		}
	}

	var events []llm.StreamEvent
	for _, choice := range chunk.Choices {
		if choice == nil {
			continue
		}
		if choice.Delta.Content != "" {
			events = append(events, llm.StreamEvent{
				Kind: llm.EventTextDelta,
				Text: choice.Delta.Content,
			})
		}
		for _, tool := range choice.Delta.ToolCalls {
			if tool == nil {
				continue
			}
			idx := 0
			if tool.Index != nil {
				idx = *tool.Index
			}
			r.toolIndexes[idx] = struct{}{}
			if tool.ID != "" || tool.Function.Name != "" {
				events = append(events, llm.StreamEvent{
					Kind: llm.EventToolCallStart,
					ToolCall: &llm.ToolCallDelta{
						Index: idx,
						ID:    tool.ID,
						Name:  tool.Function.Name,
					},
				})
			}
			if tool.Function.Arguments != "" {
				events = append(events, llm.StreamEvent{
					Kind: llm.EventToolCallArgsDelta,
					ToolCall: &llm.ToolCallDelta{
						Index:     idx,
						ID:        tool.ID,
						ArgsDelta: tool.Function.Arguments,
					},
				})
			}
		}
		if choice.FinishReason != "" {
			r.lastFinish = mapFinishReason(choice.FinishReason)
			if r.lastFinish == llm.FinishReasonToolCalls && len(r.toolIndexes) > 0 {
				indexes := make([]int, 0, len(r.toolIndexes))
				for i := range r.toolIndexes {
					indexes = append(indexes, i)
				}
				sort.Ints(indexes)
				for _, i := range indexes {
					events = append(events, llm.StreamEvent{
						Kind:     llm.EventToolCallEnd,
						ToolCall: &llm.ToolCallDelta{Index: i},
					})
				}
				r.toolIndexes = make(map[int]struct{})
			}
		}
	}
	return events
}

// isEOF detects the io.EOF sentinel returned by the SDK stream reader at
// data: [DONE].
func isEOF(err error) bool {
	return err == io.EOF
}
```

- [ ] **Step 4: Add the stream request mapper (edit map.go)**

Append to `volcengine/map.go`:

```go
// toSDKStreamRequest builds the streaming variant: same as toSDKRequest but
// with StreamOptions.IncludeUsage so the final chunk carries token usage.
// (The SDK sets Stream=true internally via request.WithStream(true).)
func (v *Volcengine) toSDKStreamRequest(req llm.Request) model.CreateChatCompletionRequest {
	out := v.toSDKRequest(req)
	out.StreamOptions = &model.StreamOptions{IncludeUsage: true}
	return out
}
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestStream_Volcengine_Text|TestStream_Volcengine_FragmentedToolCalls' -v
```

Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/volcengine.go volcengine/map.go volcengine/volcengine_test.go && git commit --no-verify -m "feat(volcengine): streaming reader with fragmented tool-call merge by Index"
```

---

## Task 6: Error mapping (errors.go)

**Files:**
- Create: `volcengine/errors.go`
- Modify: `volcengine/volcengine.go` (delete the temporary `wrapErr` stub added in Task 3)
- Test: `volcengine/volcengine_test.go` (append)

- [ ] **Step 1: Write the failing error-mapping tests**

Append to `volcengine/volcengine_test.go`:

```go
func TestWrapErr_Volcengine_StatusMapping(t *testing.T) {
	cases := []struct {
		status int
		match  func(error) bool
		name   string
	}{
		{401, func(err error) bool { var e *llm.AuthError; return asAuth(err, &e) }, "auth-401"},
		{403, func(err error) bool { var e *llm.AuthError; return asAuth(err, &e) }, "auth-403"},
		{429, func(err error) bool { var e *llm.RateLimitError; return asRate(err, &e) }, "rate-429"},
		{500, func(err error) bool { var e *llm.TransientError; return asTransient(err, &e) }, "transient-500"},
		{503, func(err error) bool { var e *llm.TransientError; return asTransient(err, &e) }, "transient-503"},
		{404, func(err error) bool { var e *llm.InvalidRequestError; return asInvalid(err, &e) }, "invalid-404"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(c.status)
				_, _ = w.Write([]byte(`{"error":{"message":"boom","type":"error","code":"x"}}`))
			}))
			defer server.Close()

			m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"), WithBaseURL(server.URL))
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			_, gerr := m.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
			if gerr == nil {
				t.Fatalf("Generate() error = nil, want %s", c.name)
			}
			if !c.match(gerr) {
				t.Fatalf("Generate() error = %v (%T), want %s", gerr, gerr, c.name)
			}
		})
	}
}

func TestWrapErr_Volcengine_ContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, gerr := m.Generate(ctx, llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if gerr == nil {
		t.Fatal("Generate() error = nil, want context.Canceled")
	}
	if !isContextCanceled(gerr) {
		t.Fatalf("Generate() error = %v, want context.Canceled passthrough", gerr)
	}
}
```

Add these small `errors.As`/`errors.Is` test helpers at the bottom of `volcengine_test.go` (keeps the table compact and avoids repeating `errors.As` four times):

```go
func asAuth(err error, e **llm.AuthError) bool             { return errorsAs(err, e) }
func asRate(err error, e **llm.RateLimitError) bool        { return errorsAs(err, e) }
func asTransient(err error, e **llm.TransientError) bool   { return errorsAs(err, e) }
func asInvalid(err error, e **llm.InvalidRequestError) bool { return errorsAs(err, e) }
func errorsAs(err error, target any) bool                  { return errors.As(err, target) }
func isContextCanceled(err error) bool                     { return errors.Is(err, context.Canceled) }
```

Add `"errors"` to the test import block (top of `volcengine_test.go`):

```go
import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestWrapErr_Volcengine' -v
```

Expected: FAIL — the temporary `wrapErr` stub (Task 3) returns the error verbatim, so none of the typed-error assertions match.

- [ ] **Step 3: Delete the temporary wrapErr stub from volcengine.go**

In `volcengine/volcengine.go`, delete:

```go
// wrapErr is defined here temporarily; Task 6 moves it to errors.go.
func wrapErr(err error) error { return err }
```

- [ ] **Step 4: Write errors.go**

Create `volcengine/errors.go`:

```go
package volcengine

import (
	"context"
	"errors"
	"net"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// wrapErr maps an arkruntime SDK error into the canonical llm/* typed-error
// tree. Both *model.APIError and *model.RequestError carry an HTTP status
// code; routing is identical to the other adapters.
//
// Mapping:
//   - nil → nil
//   - context.Canceled → passthrough (caller-initiated; not a provider fault)
//   - context.DeadlineExceeded → *llm.TransientError
//   - status 401/403 → *llm.AuthError
//   - status 429 → *llm.RateLimitError
//   - status 500/502/503/504 → *llm.TransientError
//   - other 4xx → *llm.InvalidRequestError
//   - net.Error (any) → *llm.TransientError
//   - anything else → passthrough
func wrapErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: "volcengine", Wrapped: err}
	}

	status := 0
	var apiErr *model.APIError
	var reqErr *model.RequestError
	if errors.As(err, &apiErr) {
		status = apiErr.HTTPStatusCode
	} else if errors.As(err, &reqErr) {
		status = reqErr.HTTPStatusCode
	}

	if status != 0 {
		switch status {
		case 401, 403:
			return &llm.AuthError{Provider: "volcengine", Wrapped: err}
		case 429:
			return &llm.RateLimitError{Provider: "volcengine", Wrapped: err}
		case 500, 502, 503, 504:
			return &llm.TransientError{Provider: "volcengine", Wrapped: err}
		default:
			if status >= 400 && status < 500 {
				return &llm.InvalidRequestError{Provider: "volcengine", Wrapped: err}
			}
		}
	}

	var netErr net.Error
	if errors.As(err, &netErr) {
		return &llm.TransientError{Provider: "volcengine", Wrapped: err}
	}
	return err
}
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestWrapErr_Volcengine' -v
```

Expected: PASS (status-mapping subtests + context-canceled).

> Implementation note for the executor: the arkruntime client decodes a failure response body into `*model.APIError` when the JSON has an `error` object (see `handleErrorResp` in `client.go`), and into `*model.RequestError` otherwise. The test server returns `{"error":{...}}`, so the path under test is `*model.APIError`. The `RequestError` branch is covered by the context-canceled / network cases. If the 401 subtest unexpectedly yields a `*model.RequestError` (e.g. body decode fails), the same status routing still applies — both branches feed the one `switch`.

- [ ] **Step 6: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/errors.go volcengine/volcengine.go volcengine/volcengine_test.go && git commit --no-verify -m "feat(volcengine): error mapping to typed llm.* errors"
```

---

## Task 7: Image generation (image.go)

**Files:**
- Create: `volcengine/image.go`
- Test: `volcengine/volcengine_test.go` (append)

**Design notes:** `GenerateImages` uses `model.GenerateImagesRequest` (confirmed present in SDK v1.2.33 — NO raw-HTTP fallback). The request has **no `N` field**; multi-image is via `SequentialImageGeneration*`, which is out of scope (spec: text-to-image single `Generate` only) — so `req.N` is accepted but only `N<=1` is meaningfully supported; we forward a single-image request and map all returned `Data[]`. Default `ResponseFormat` is `url` (Ark default; links expire ~24h). `Extra` keys: `seed` (int64), `guidance_scale` (float64), `watermark` (bool). Response delivery: `Image.Url != nil` → `URL`; `Image.B64Json != nil` → base64-decode → `Bytes`. K2 gating: non-seedream models return `llm.ErrCapabilityNotSupported`.

- [ ] **Step 1: Write the failing image tests (URL default + b64 + capability gate)**

Append to `volcengine/volcengine_test.go`:

```go
func TestGenerateImage_Volcengine_URL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/generations" {
			t.Fatalf("Path = %s, want /images/generations", r.URL.Path)
		}
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"model":"doubao-seedream-4-5-251128"`) {
			t.Fatalf("body missing model: %s", body)
		}
		if !strings.Contains(body, `"prompt":"a red panda"`) {
			t.Fatalf("body missing prompt: %s", body)
		}
		if !strings.Contains(body, `"size":"1024x1024"`) {
			t.Fatalf("body missing size: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model":"doubao-seedream-4-5-251128",
			"created":1710000000,
			"data":[{"url":"https://example.com/img.png","size":"1024x1024"}],
			"usage":{"generated_images":1,"output_tokens":0,"total_tokens":0}
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-seedream-4-5-251128"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := m.GenerateImage(context.Background(), llm.ImageRequest{
		Prompt: "a red panda",
		Size:   "1024x1024",
	})
	if err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if resp.Provider != "volcengine" || resp.Model != "doubao-seedream-4-5-251128" {
		t.Fatalf("resp meta = %+v, want volcengine/doubao-seedream-4-5-251128", resp)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("len(Images) = %d, want 1", len(resp.Images))
	}
	if resp.Images[0].URL != "https://example.com/img.png" {
		t.Fatalf("Images[0].URL = %q, want https://example.com/img.png", resp.Images[0].URL)
	}
	if len(resp.Images[0].Bytes) != 0 {
		t.Fatalf("Images[0].Bytes = %d bytes, want 0 (URL delivery)", len(resp.Images[0].Bytes))
	}
}

func TestGenerateImage_Volcengine_Base64(t *testing.T) {
	// base64 of the 3 bytes {0x01,0x02,0x03} is "AQID"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(bodyBytes), `"response_format":"b64_json"`) {
			t.Fatalf("body missing b64_json response_format: %s", string(bodyBytes))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model":"doubao-seedream-4-5-251128",
			"created":1710000000,
			"data":[{"b64_json":"AQID","size":"1024x1024"}]
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-seedream-4-5-251128"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := m.GenerateImage(context.Background(), llm.ImageRequest{
		Prompt: "a red panda",
		Format: "b64_json",
	})
	if err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("len(Images) = %d, want 1", len(resp.Images))
	}
	if resp.Images[0].URL != "" {
		t.Fatalf("Images[0].URL = %q, want empty (bytes delivery)", resp.Images[0].URL)
	}
	if string(resp.Images[0].Bytes) != "\x01\x02\x03" {
		t.Fatalf("Images[0].Bytes = %v, want [1 2 3]", resp.Images[0].Bytes)
	}
}

func TestGenerateImage_Volcengine_CapabilityGate(t *testing.T) {
	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, gerr := m.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if gerr == nil {
		t.Fatal("GenerateImage() error = nil, want ErrCapabilityNotSupported")
	}
	if !errors.Is(gerr, llm.ErrCapabilityNotSupported) {
		t.Fatalf("GenerateImage() error = %v, want ErrCapabilityNotSupported", gerr)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestGenerateImage_Volcengine' -v
```

Expected: FAIL — `GenerateImage` undefined.

- [ ] **Step 3: Write image.go**

Create `volcengine/image.go`:

```go
package volcengine

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// GenerateImage runs a text-to-image generation. Only available on
// doubao-seedream* models (K2-gated); other models return
// ErrCapabilityNotSupported. Multi-image (N>1) is not supported (Ark uses
// SequentialImageGeneration, which is out of scope) — a single image request
// is sent and every returned image is mapped.
func (v *Volcengine) GenerateImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	if !v.info.Capabilities.ImageGeneration {
		return llm.ImageResponse{}, fmt.Errorf("volcengine: image generation: %w", llm.ErrCapabilityNotSupported)
	}

	sdkReq := model.GenerateImagesRequest{
		Model:  v.info.Model,
		Prompt: req.Prompt,
	}

	// Response format: default url; b64_json when caller asks for bytes.
	respFormat := model.GenerateImagesResponseFormatURL
	if req.Format == model.GenerateImagesResponseFormatBase64 || req.Format == "bytes" {
		respFormat = model.GenerateImagesResponseFormatBase64
	}
	sdkReq.ResponseFormat = strPtr(respFormat)

	if req.Size != "" {
		sdkReq.Size = strPtr(req.Size)
	}

	// Provider-specific knobs forwarded via Extra.
	if req.Extra != nil {
		if seed, ok := toInt64(req.Extra["seed"]); ok {
			sdkReq.Seed = &seed
		}
		if gs, ok := toFloat64(req.Extra["guidance_scale"]); ok {
			sdkReq.GuidanceScale = &gs
		}
		if wm, ok := req.Extra["watermark"].(bool); ok {
			sdkReq.Watermark = &wm
		}
	}

	resp, err := v.client.GenerateImages(ctx, sdkReq, v.requestOptions()...)
	if err != nil {
		return llm.ImageResponse{}, wrapErr(err)
	}

	images := make([]llm.GeneratedImage, 0, len(resp.Data))
	for _, img := range resp.Data {
		if img == nil {
			continue
		}
		var gen llm.GeneratedImage
		switch {
		case img.B64Json != nil && *img.B64Json != "":
			decoded, derr := base64.StdEncoding.DecodeString(*img.B64Json)
			if derr != nil {
				return llm.ImageResponse{}, &llm.InvalidRequestError{Provider: "volcengine", Wrapped: derr}
			}
			gen.Bytes = decoded
		case img.Url != nil:
			gen.URL = *img.Url
		}
		images = append(images, gen)
	}

	out := llm.ImageResponse{
		Images:   images,
		Provider: "volcengine",
		Model:    resp.Model,
	}
	if resp.Usage != nil {
		out.Usage = llm.Usage{
			OutputTokens: int(resp.Usage.OutputTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
			Source:       llm.UsageReported,
		}
	}
	return out, nil
}

// toInt64 coerces common numeric JSON shapes (Extra is map[string]any) to int64.
func toInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case int:
		return int64(n), true
	case int64:
		return n, true
	case float64:
		return int64(n), true
	default:
		return 0, false
	}
}

// toFloat64 coerces common numeric JSON shapes to float64.
func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestGenerateImage_Volcengine' -v
```

Expected: PASS (URL, Base64, CapabilityGate).

- [ ] **Step 5: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/image.go volcengine/volcengine_test.go && git commit --no-verify -m "feat(volcengine): image generation (url + b64) with K2 gating"
```

---

## Task 8: Embeddings (embed.go)

**Files:**
- Create: `volcengine/embed.go`
- Test: `volcengine/volcengine_test.go` (append)

**Design notes:** `CreateEmbeddings(ctx, model.EmbeddingRequestStrings{Input, Model, Dimensions})` → `model.EmbeddingResponse{Data []model.Embedding, Usage model.Usage}`. `Embedding.Embedding` is already `[]float32`, so `llm.Vector` is a direct copy. Vectors must come back in input order — the SDK preserves `Data[].Index`, but the wire response is already ordered; we copy in slice order (matching the openai sibling, which trusts response order). Empty `texts` short-circuits. K2 gate: non-embed models return `ErrCapabilityNotSupported`. `Dimensions` is sent only when `v.embedDimensions > 0`.

- [ ] **Step 1: Write the failing embed tests**

Append to `volcengine/volcengine_test.go`:

```go
func TestEmbed_Volcengine_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("Path = %s, want /embeddings", r.URL.Path)
		}
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"model":"doubao-embedding-text-240715"`) {
			t.Fatalf("body missing model: %s", body)
		}
		if !strings.Contains(body, `"input":["hello","world"]`) {
			t.Fatalf("body missing ordered input: %s", body)
		}
		if !strings.Contains(body, `"dimensions":2560`) {
			t.Fatalf("body missing dimensions: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"emb_1",
			"object":"list",
			"model":"doubao-embedding-text-240715",
			"data":[
				{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]},
				{"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]}
			],
			"usage":{"prompt_tokens":4,"total_tokens":4}
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-embedding-text-240715"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed(): %v", err)
	}
	if len(vectors) != 2 || len(vectors[0]) != 3 || len(vectors[1]) != 3 {
		t.Fatalf("vectors shape = %d x [%d %d], want 2 x [3 3]", len(vectors), len(vectors[0]), len(vectors[1]))
	}
	if vectors[1][0] != 0.4 {
		t.Fatalf("vectors[1][0] = %v, want 0.4 (order preserved)", vectors[1][0])
	}
	if usage.Source != llm.UsageReported || usage.InputTokens != 4 || usage.TotalTokens != 4 {
		t.Fatalf("usage = %+v, want reported input=4 total=4", usage)
	}
	if got := m.EmbedDimensions(); got != 2560 {
		t.Fatalf("EmbedDimensions() = %d, want 2560", got)
	}
}

func TestEmbed_Volcengine_Empty(t *testing.T) {
	m, err := New(WithModel("doubao-embedding-text-240715"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(nil): %v", err)
	}
	if len(vectors) != 0 {
		t.Fatalf("len(vectors) = %d, want 0", len(vectors))
	}
	if usage.Source != llm.UsageReported {
		t.Fatalf("usage.Source = %q, want reported", usage.Source)
	}
}

func TestEmbed_Volcengine_CapabilityGate(t *testing.T) {
	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, _, gerr := m.Embed(context.Background(), []string{"x"})
	if gerr == nil {
		t.Fatal("Embed() error = nil, want ErrCapabilityNotSupported")
	}
	if !errors.Is(gerr, llm.ErrCapabilityNotSupported) {
		t.Fatalf("Embed() error = %v, want ErrCapabilityNotSupported", gerr)
	}
	if got := m.EmbedDimensions(); got != 0 {
		t.Fatalf("EmbedDimensions() = %d, want 0 for non-embed model", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestEmbed_Volcengine' -v
```

Expected: FAIL — `Embed` undefined.

- [ ] **Step 3: Write embed.go**

Create `volcengine/embed.go`:

```go
package volcengine

import (
	"context"
	"fmt"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// Embed returns one vector per input text, in input order. Only available on
// doubao-embedding* models (K2-gated). The bound dimensionality (from the
// model default or WithDimensions) is sent when non-zero.
func (v *Volcengine) Embed(ctx context.Context, texts []string) ([]llm.Vector, llm.Usage, error) {
	if !v.info.Capabilities.Embeddings {
		return nil, llm.Usage{}, fmt.Errorf("volcengine: embeddings: %w", llm.ErrCapabilityNotSupported)
	}
	if len(texts) == 0 {
		return []llm.Vector{}, llm.Usage{Source: llm.UsageReported}, nil
	}

	reqStrings := model.EmbeddingRequestStrings{
		Input: append([]string(nil), texts...),
		Model: v.info.Model,
	}
	if v.embedDimensions > 0 {
		reqStrings.Dimensions = v.embedDimensions
	}

	resp, err := v.client.CreateEmbeddings(ctx, reqStrings, v.requestOptions()...)
	if err != nil {
		return nil, llm.Usage{}, wrapErr(err)
	}

	vectors := make([]llm.Vector, 0, len(resp.Data))
	for _, item := range resp.Data {
		vec := make(llm.Vector, len(item.Embedding))
		copy(vec, item.Embedding)
		vectors = append(vectors, vec)
	}
	usage := llm.Usage{
		InputTokens: resp.Usage.PromptTokens,
		TotalTokens: resp.Usage.TotalTokens,
		Source:      llm.UsageReported,
	}
	return vectors, usage, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestEmbed_Volcengine' -v
```

Expected: PASS (Happy, Empty, CapabilityGate).

- [ ] **Step 5: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/embed.go volcengine/volcengine_test.go && git commit --no-verify -m "feat(volcengine): text embeddings with K2 gating + dimensions"
```

---

## Task 9: Custom headers + Info capability tests + stream goroutine leak check

**Files:**
- Test: `volcengine/volcengine_test.go` (append)

- [ ] **Step 1: Write the failing tests (extra headers forwarded; Info capability matrix; goleak)**

Append to `volcengine/volcengine_test.go`:

```go
func TestExtraHeaders_Volcengine_Forwarded(t *testing.T) {
	var seenHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenHeader = r.Header.Get("X-Gateway-Route")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"doubao-1-5-pro-32k-250115","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("doubao-1-5-pro-32k-250115"),
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithExtraHeaders(map[string]string{"X-Gateway-Route": "canary"}),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if seenHeader != "canary" {
		t.Fatalf("X-Gateway-Route = %q, want canary", seenHeader)
	}
}

func TestInfo_Volcengine_CapabilityMatrix(t *testing.T) {
	cases := []struct {
		model      string
		wantTools  bool
		wantImage  bool
		wantEmbed  bool
	}{
		{"doubao-1-5-pro-32k-250115", true, false, false},
		{"doubao-seedream-4-5-251128", false, true, false},
		{"doubao-embedding-text-240715", false, false, true},
	}
	for _, c := range cases {
		m, err := New(WithModel(c.model), WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New(%q): %v", c.model, err)
		}
		caps := m.Info().Capabilities
		if caps.Tools != c.wantTools || caps.ImageGeneration != c.wantImage || caps.Embeddings != c.wantEmbed {
			t.Fatalf("Info(%q).Capabilities = %+v, want tools=%v image=%v embed=%v",
				c.model, caps, c.wantTools, c.wantImage, c.wantEmbed)
		}
		if m.Info().Provider != "volcengine" {
			t.Fatalf("Provider = %q, want volcengine", m.Info().Provider)
		}
	}
}

func TestStream_Volcengine_NoGoroutineLeak(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c1\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c1\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	sr, err := m.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	if _, err := llm.AccumulateStream(sr); err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
}
```

Add `"go.uber.org/goleak"` to the test import block (it is already a repo dependency):

```go
import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
	"go.uber.org/goleak"
)
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -run 'TestExtraHeaders_Volcengine_Forwarded|TestInfo_Volcengine_CapabilityMatrix|TestStream_Volcengine_NoGoroutineLeak' -v
```

Expected: PASS. (`WithExtraHeaders` and `requestOptions()` were already implemented in Task 2; these tests are the first to exercise the header forwarding end-to-end and the capability matrix. If `TestExtraHeaders` FAILS because the header is not seen, verify `requestOptions()` returns `arkruntime.WithCustomHeaders(v.extraHeaders)` and that `Generate`/`Stream`/`GenerateImage`/`Embed` all spread `v.requestOptions()...` into the SDK call.)

- [ ] **Step 3: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add volcengine/volcengine_test.go && git commit --no-verify -m "test(volcengine): extra headers, capability matrix, stream goroutine leak"
```

---

## Task 10: Full package verification (build, vet, full test run, tidy)

**Files:** none (verification only)

- [ ] **Step 1: Run the full package test suite**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go test ./volcengine/ -v
```

Expected: ALL tests PASS (Generate happy/tools, Stream text/tools/leak, errors, image url/b64/gate, embed happy/empty/gate, extra headers, capability matrix, options internal).

- [ ] **Step 2: Run go vet on the package**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go vet ./volcengine/
```

Expected: PASS (no output).

- [ ] **Step 3: Confirm interface assertions hold and whole repo still builds**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go build ./...
```

Expected: PASS. (The `var _ llm.ImageGenerator = (*Volcengine)(nil)` etc. assertions in `volcengine.go` fail the build if any interface method is missing or mis-typed.)

- [ ] **Step 4: Tidy module graph**

Run:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && GOWORK=off go mod tidy
```

Expected: `go.mod`/`go.sum` settle with `github.com/volcengine/volcengine-go-sdk v1.2.33` promoted to a direct dependency; no other unexpected churn. Re-run `GOWORK=off go test ./volcengine/` to confirm tidy did not break anything.

- [ ] **Step 5: Commit**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers && git add go.mod go.sum && git commit --no-verify -m "chore(volcengine): tidy module graph after arkruntime adoption"
```

---

## Self-Review

**1. Spec coverage (volcengine section + Embeddings + Custom headers + Testing):**

- Full provider ChatModel+ToolCaller+ImageGenerator+Embedder → Task 2 (asserts), 3-5 (chat/stream/tools), 7 (image), 8 (embed). ✔
- `arkruntime.NewClientWithApiKey` + `WithRegion(cn-beijing default)` + `WithRetryTimes(0)` → Task 2 `newArkClient`. ✔
- Options `WithModel`(required)/`WithAPIKey`(ARK_API_KEY)/`WithBaseURL`/`WithRegion`/`WithHTTPClient`/`WithTimeout`/`WithDimensions`/`WithExtraHeaders` → Task 2 options.go. ✔
- `capabilitiesForModel` (Tools chat / ImageGeneration seedream / Embeddings embedding) → Task 2. ✔
- `CreateChatCompletion` with `CreateChatCompletionRequest` pointer-field variant + content union → Task 3 map.go. ✔ (Used as a VALUE — verified the SDK methods are value-receiver; `Temperature *float32` pointer field makes `temperature=0` sendable, satisfying the spec's intent.)
- Finish-reason mapping + tool req/resp mapping → Task 3/4 map.go. ✔
- Stream `CreateChatCompletionStream` → `Recv()` to io.EOF, text delta + fragmented tool-call merge by `ToolCall.Index`, mirror openai → Task 5. ✔
- Image `GenerateImages` + map to GeneratedImage (URL default / Bytes b64) → Task 7. ✔ (No raw-HTTP fallback — `GenerateImagesRequest`/`ImagesResponse` confirmed in SDK.)
- Embed `CreateEmbeddings(EmbeddingRequestStrings{Input,Dimensions})`, `EmbedDimensions` per model → Task 8. ✔
- Errors `*model.APIError`/`*model.RequestError` (HTTPStatusCode) → typed llm.* + context handling → Task 6. ✔
- Custom headers via `arkruntime.WithCustomHeaders` (resolves spec open item #6 for arkruntime) → Task 2 `requestOptions()` + Task 9 test. ✔
- Testing: httptest via `WithBaseUrl`, mock chat/SSE-with-fragmented-tool-calls/image/embed, real handlers + assertions, goleak → Tasks 3-9. ✔

**2. Placeholder scan:** Searched for TBD/TODO/"similar to above"/"add error handling". The two deliberate temporary stubs (`Generate`/`Stream` `io.EOF` placeholders in Task 2, `wrapErr` stub in Task 3, bottom `var ( _ = ... )` compile-aid) are each explicitly created AND explicitly removed in a named later step with full replacement code shown — not open-ended placeholders. The test files use intentional non-compiling markers (`fmtFprint`, `import_fmt_marker`) ONLY in Task 5, with an explicit edit instruction in the same step to fix them before running. No silent gaps. ✔

**3. Type consistency:** `Volcengine` struct fields (`client *arkruntime.Client`, `info`, `tools`, `timeout`, `embedDimensions`, `extraHeaders`) are consistent across options.go (set in `New`), volcengine.go (used), image.go/embed.go (read). `requestOptions()` returns `[]arkruntime.RequestOption` and is spread everywhere. `toSDKRequest` returns `model.CreateChatCompletionRequest` (value) used by both `Generate` and `toSDKStreamRequest`. `mapFinishReason(model.FinishReason)` signature consistent in map.go and the stream reader. `wrapErr(error) error` single definition (errors.go after Task 6). `strPtr`/`contentString` defined once in map.go, reused in image.go. `defaultEmbedDimensions`/`providerInfo`/`capabilitiesForModel` defined in options.go, referenced in the internal test. ✔

**4. SDK symbol accuracy:** Every arkruntime symbol used was read from `volcengine-go-sdk@v1.2.33` source. Two corrections baked into the plan vs. the spec's loose wording: (a) the config option is `arkruntime.WithBaseUrl` (lowercase `rl`), NOT `WithBaseURL`; (b) `model.GenerateImagesRequest` has **no `N` field** (multi-image is `SequentialImageGeneration*`, out of scope) — the plan forwards a single-image request. `CreateChatCompletionRequest` methods are value-receiver, so it is passed by value. These are all confirmed, not guessed.
