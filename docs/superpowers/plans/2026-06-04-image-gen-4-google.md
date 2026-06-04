# Google (Gemini) Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new full provider package `google/` in `llm-agent-providers` over the official `google.golang.org/genai` SDK (v1.59.0) — a first-class sibling of `openai`/`anthropic` implementing `llm.ChatModel` + `llm.ToolCaller` + `llm.ImageGenerator` (Nano Banana inline + Imagen) + `llm.Embedder`, all gated per bound (provider × model) tuple (K2).

**Architecture:** One `*Google` struct binds one model at construction; `Info().Capabilities` reflects what THAT model can do. Chat maps to `client.Models.GenerateContent`; system prompt becomes `GenerateContentConfig.SystemInstruction` (Gemini has no `system` role, only `user`/`model`). Streaming bridges the SDK's Go-1.23 `iter.Seq2[*GenerateContentResponse, error]` to the repo's pull-based `llm.StreamReader` via `iter.Pull2`. Tools use `FunctionDeclaration.ParametersJsonSchema` (raw JSON-schema → `map[string]any`, no typed `genai.Schema` translation); streamed tool calls arrive COMPLETE in one chunk (Start+ArgsDelta(full)+End emitted together — no cross-chunk accumulation). Images route by model prefix: `imagen*` → `GenerateImages` (`GeneratedImages[].Image.ImageBytes`); else Gemini-native → `GenerateContent` with `ResponseModalities ["TEXT","IMAGE"]` extracting `InlineData{Data,MIMEType}`. Embeddings use `EmbedContent` (one `*genai.Content` per text); the Gemini Developer API returns no token usage, so `Usage` is zero (`UsageUnknown`). All SDK types stay on unexported fields (`client *genai.Client`); the public surface is `llm.*` only.

**Tech Stack:** Go 1.26 (`go.mod` says `go 1.26.0`), `google.golang.org/genai v1.59.0` (NEW module dep), `github.com/costa92/llm-agent-contract` (ASSUME **v0.3.0** via a dev-time local `replace`; image-gen-1-contract plan ships the `ImageGenerator`/`ImageRequest`/`ImageResponse`/`GeneratedImage` types + `Capabilities.ImageGeneration` this plan consumes), stdlib (`context`, `encoding/json`, `errors`, `io`, `iter`, `net/http`, `os`, `strings`, `sync`, `time`). Tests use `net/http/httptest` pointed at via `genai.ClientConfig.HTTPOptions.BaseURL`.

> **`go.work` GOTCHA:** the umbrella has `go.work` at `/home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/go.work`. **Every `go` command in this plan MUST be prefixed `GOWORK=off`** so the `google` package builds against the providers module in isolation. Do not drop the prefix.

**Repo:** `/home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers` (module `github.com/costa92/llm-agent-providers`). All package paths below are relative to this repo root.

---

## SDK symbol verification (read against `google.golang.org/genai@v1.59.0` source)

Every symbol below was confirmed by reading the SDK source under `/home/hellotalk/code/go/pkg/mod/google.golang.org/genai@v1.59.0/`. The plan uses these EXACTLY.

**Client construction (`client.go`):**
- `func NewClient(ctx context.Context, cc *ClientConfig) (*Client, error)` — line 417.
- `type ClientConfig struct { APIKey string; Backend Backend; Project string; Location string; Credentials *auth.Credentials; HTTPClient *http.Client; HTTPOptions HTTPOptions; ... }` — line 91. `APIKey`/`Backend`/`HTTPClient`/`HTTPOptions` are the fields used. `Project`/`Location`/`Credentials` are mutually exclusive with `APIKey` (constructor errors if both set) — DO NOT set them.
- `const BackendGeminiAPI Backend` — line 66 (`Backend` is `int`, line 56).
- `type Client struct { ... Models *Models ... }` — line 31. Access via `client.Models.X(...)`.
- `type HTTPOptions struct { BaseURL string; APIVersion string; Headers http.Header; Timeout *time.Duration; ... }` — line 1705. `Headers` is **`http.Header`** (NOT `map[string]string`). Default `APIVersion` is `"v1beta"` (client.go:364); default Gemini `BaseURL` is `https://generativelanguage.googleapis.com/`. Setting `HTTPOptions.BaseURL` to an httptest URL overrides the host.
- API key is sent in the `x-goog-api-key` header (api_client.go:328); `HTTPOptions.Headers` is applied as `req.Header` (api_client.go:320) so `WithExtraHeaders` injects via this field.
- `func Ptr[T any](t T) *T` — common.go:36 (for `*float32`/`*int32`).
- `func Text(text string) []*Content` — models_helpers.go:18 (returns one user-role Content). We do NOT use this for chat mapping (we build Contents directly) but it documents the one-Content-per-call shape used in `embed.go`.

**Models methods (`models.go`), all on value receiver `Models`, called via pointer `client.Models.X`:**
- `func (m Models) GenerateContent(ctx context.Context, model string, contents []*Content, config *GenerateContentConfig) (*GenerateContentResponse, error)` — line 5627.
- `func (m Models) GenerateContentStream(ctx context.Context, model string, contents []*Content, config *GenerateContentConfig) iter.Seq2[*GenerateContentResponse, error]` — line 5635.
- `func (m Models) GenerateImages(ctx context.Context, model string, prompt string, config *GenerateImagesConfig) (*GenerateImagesResponse, error)` — line 5699.
- `func (m Models) EmbedContent(ctx context.Context, model string, contents []*Content, config *EmbedContentConfig) (*EmbedContentResponse, error)` — line 5778.

**Types (`types.go`):**
- `type Content struct { Parts []*Part; Role string }` — line 1595. Role consts `RoleUser="user"`, `RoleModel="model"` (lines 1607-1608).
- `type Part struct { ... FunctionCall *FunctionCall; InlineData *Blob; Text string; ... }` — line 1459. Fields used: `Text`, `InlineData`, `FunctionCall`.
- `type Blob struct { Data []byte; DisplayName string; MIMEType string }` — line 1375. (`Data` is `[]byte`, NOT base64 string — the SDK decodes JSON `"data":"<base64>"` into `[]byte` automatically.)
- `type FunctionCall struct { ID string; Args map[string]any; Name string; ... }` — line 1263.
- `type GenerateContentConfig struct { SystemInstruction *Content; Temperature *float32; TopP *float32; MaxOutputTokens int32; Tools []*Tool; ResponseModalities []string; ... }` — line 2640 (`SystemInstruction` 2646, `Temperature` 2651, `MaxOutputTokens` int32 2667, `Tools` 2730, `ResponseModalities []string` 2740).
- `type Tool struct { FunctionDeclarations []*FunctionDeclaration; ... }` — line 2390 (`FunctionDeclarations` 2419).
- `type FunctionDeclaration struct { Description string; Name string; Parameters *Schema; ParametersJsonSchema any; ... }` — line 2243. We set `Name`, `Description`, `ParametersJsonSchema any` (line 2266) — assign a `map[string]any` unmarshalled from the repo's raw JSON-schema bytes. (`Parameters *Schema` is mutually exclusive — do NOT set it.)
- `type GenerateContentResponse struct { Candidates []*Candidate; PromptFeedback *GenerateContentResponsePromptFeedback; UsageMetadata *GenerateContentResponseUsageMetadata; ModelVersion string; ... }` — line 3426.
  - `func (r *GenerateContentResponse) Text() string` — line 3486 (first candidate's concatenated text parts; safe on empty candidates).
  - `func (r *GenerateContentResponse) FunctionCalls() []*FunctionCall` — line 3537 (first candidate's function-call parts; returns nil if none).
- `type Candidate struct { Content *Content; FinishReason FinishReason; Index int32; ... }` — line 3283.
- `type GenerateContentResponsePromptFeedback struct { BlockReason BlockedReason; BlockReasonMessage string; ... }` — line 3323.
- `type GenerateContentResponseUsageMetadata struct { PromptTokenCount int32; CandidatesTokenCount int32; TotalTokenCount int32; ... }` — line 3344.
- `FinishReason` consts: `FinishReasonStop="STOP"` (345), `FinishReasonMaxTokens="MAX_TOKENS"` (347), `FinishReasonSafety="SAFETY"` (350), `FinishReasonRecitation="RECITATION"` (352) — `FinishReason` is a string type.
- `type GenerateImagesConfig struct { NumberOfImages int32; AspectRatio string; ... }` — line 3667 (`NumberOfImages` int32 3676, `AspectRatio` 3679).
- `type GenerateImagesResponse struct { GeneratedImages []*GeneratedImage; ... }` — line 3753. **Field name is `GeneratedImages` (CONFIRMED, NOT `Images`).**
- `type GeneratedImage struct { Image *Image; ... }` — line 3738.
- `type Image struct { GCSURI string; ImageBytes []byte; MIMEType string }` — line 3716. We read `ImageBytes` ([]byte) and `MIMEType`.
- `type EmbedContentConfig struct { TaskType string; Title string; OutputDimensionality *int32; ... }` — line 3599 (`TaskType` 3601, `OutputDimensionality *int32` 3609).
- `type EmbedContentResponse struct { Embeddings []*ContentEmbedding; ... }` — line 3654.
- `type ContentEmbedding struct { Values []float32; ... }` — line 3637. We read `.Values`.

**Errors (`api_client.go`):**
- `type APIError struct { Code int; Message string; Status string; Details []map[string]any }` — line 528. **Returned BY VALUE** (newAPIError returns `*respWithError.ErrorInfo` dereferenced, or `APIError{...}` literals — lines 553/558/562/564). `Error()` is on value receiver (line 568). So detect with `var e genai.APIError; errors.As(err, &e)` then switch `e.Code`.

**Streaming wire format (api_client.go:445-517):** SSE lines `data: {json}`; the SDK does `bytes.Cut(line, ":")` on `"data"`. Test stream handlers write `Content-Type: text/event-stream` and emit `data: {json}\n\n` per chunk. The request path is `models/{model}:streamGenerateContent?alt=sse`. The SDK's `iterateResponseStream` closes the response body when the range-over-func finishes.

**Request URL shape (transformer.go:97-101 + api_client.go:172-178):** model is prefixed `models/` and the path prefixed by `APIVersion`. Test handlers therefore match paths like `/v1beta/models/gemini-2.5-flash:generateContent`, `:streamGenerateContent`, `:embedContent`, and `models/imagen-4.0-generate-001:predict`.

**COULD NOT confirm from source / flagged:** (1) the exact JSON wire shape of the Imagen `:predict` response that the SDK's `generateImages` converter expects (the converter is generated code in `models.go`; the test in Task 7 mocks `{"predictions":[{"bytesBase64Encoded":"..."}]}` based on the public Imagen REST contract — **verify against a live/recorded response or the SDK's `generateImages` fromConverter if the test fails**). (2) Whether Gemini-native image (`gemini-2.5-flash-image`) returns inline image parts via `:generateContent` (assumed yes per spec) vs `:streamGenerateContent` — Task 6 mocks `:generateContent`. Both are isolated to single test handlers and do not change the provider code shape.

---

## File Structure

New package `google/` follows the repo's five-file skeleton (CONVENTIONS.md §"Per-provider package layout") plus `image.go` + `embed.go` because the image and embed paths are large:

| File | Responsibility |
|------|----------------|
| `google/doc.go` | Package doc comment only. |
| `google/google.go` | `type Google struct`, interface compile-asserts, `Generate`, `Stream` (+ `googleStreamReader`), `Info`, `WithTools`. |
| `google/options.go` | `type config struct`, `type Option`, `WithModel`/`WithAPIKey`/`WithBaseURL`/`WithHTTPClient`/`WithTimeout`/`WithTaskType`/`WithDimensions`/`WithExtraHeaders`, `New`, `capabilitiesForModel`, model classifiers. |
| `google/map.go` | `toContents` (Request → `[]*genai.Content` + roles), `toGenConfig` (Request → `*genai.GenerateContentConfig` incl. SystemInstruction + tools), `fromResponse` (`*genai.GenerateContentResponse` → `llm.Response`), `toToolCalls`, `mapFinishReason`. |
| `google/image.go` | `GenerateImage` (route imagen vs gemini-native), Imagen + inline extraction. |
| `google/embed.go` | `Embed`, `EmbedDimensions`. |
| `google/errors.go` | `wrapErr(err) error` → typed `llm.*` errors (genai `APIError` by value + ctx + net fallbacks). |
| `google/google_test.go` | All httptest-backed tests (constructor/caps, generate, stream, tools, embed, capability-not-supported gating). |
| `google/image_test.go` | Image tests (Gemini-native inline + Imagen predict). |

**Module wiring (Task 0):** add `google.golang.org/genai v1.59.0` to `go.mod` and a dev-time local `replace` for the contract.

---

## Pre-flight facts (read before starting)

1. **Contract dependency.** This plan consumes `llm.ImageGenerator`, `llm.ImageRequest{Prompt,N,Size,Quality,Format,Extra}`, `llm.ImageResponse{Images,Provider,Model,Usage}`, `llm.GeneratedImage{Bytes,URL,MimeType,RevisedPrompt}`, and `llm.Capabilities.ImageGeneration` — all added by the image-gen-1-contract plan and tagged contract **v0.3.0**. During dev, pin via a local `replace` (Task 0). The replace-guard pre-commit hook auto-strips local `replace` on commit, so the final providers PR pins `v0.3.0` for real.
2. **K2 gating is the whole point.** `google.New(WithModel("gemini-2.5-flash"))` → chat/tools work, `GenerateImage`/`Embed` return `llm.ErrCapabilityNotSupported`. `WithModel("gemini-2.5-flash-image")` or `"imagen-4.0-generate-001"` → `GenerateImage` works, chat returns the not-supported error. `WithModel("gemini-embedding-001")` → `Embed` works. `capabilitiesForModel` (Task 1) is the single source of truth; every gated method re-checks it.
3. **System prompt → SystemInstruction.** Gemini roles are ONLY `user`/`model`. `llm.Request.SystemPrompt` maps to `GenerateContentConfig.SystemInstruction` (`*genai.Content`), NOT a message turn. `assistant` role → `model`; `system`/`tool` message roles in `req.Messages` are skipped (system already lifted; tool-result turns are out of scope for these tests).
4. **`Temperature`/`TopP` are `*float32`; `MaxOutputTokens` is plain `int32`.** Use `genai.Ptr(*req.Temperature)` for temperature; assign `int32(req.MaxOutputTokens)` directly.
5. **Streamed tool calls arrive COMPLETE in one chunk** (Gemini does not fragment functionCall args on the Developer API at this version per spec; `streamFunctionCallArguments` opt-in is out of scope). So the stream reader emits `EventToolCallStart` + `EventToolCallArgsDelta`(full JSON) + `EventToolCallEnd` together per call, keyed by part index. No cross-chunk accumulation (simpler than openai/volcengine).
6. **Images are ALWAYS inline base64 → `Bytes`** (never URL). Imagen via `GenerateImages` → `GeneratedImages[i].Image.ImageBytes`. Gemini-native via `GenerateContent` with `ResponseModalities:["TEXT","IMAGE"]` → `Candidates[0].Content.Parts[].InlineData{Data,MIMEType}`; drop text parts. `ResponseModalities` MUST include `"TEXT"` (image-only is rejected for Gemini 2.5 Flash Image).
7. **Embeddings return zero `Usage`** on the Gemini Developer API (no token counts) → `llm.Usage{Source: llm.UsageUnknown}` (the zero value of `UsageSource`). `EmbedDimensions()` is per model honoring `WithDimensions`.
8. **`iter.Pull2`** (Go 1.23+, available on go 1.26) bridges the SDK's `iter.Seq2[*genai.GenerateContentResponse, error]` to the pull-based `llm.StreamReader`. `next, stop := iter.Pull2(seq)`; call `next()` per `Next()`; call `stop()` in `Close()`.
9. **`genai.APIError` is a VALUE type.** `var apiErr genai.APIError; errors.As(err, &apiErr)` — NOT `*genai.APIError`.
10. **Test convention.** Each `<provider>_test.go` builds its own `httptest.NewServer` inline (CONVENTIONS.md §"Internal shared helpers"). Point genai at it via `WithBaseURL(server.URL)` + `WithAPIKey("test-key")`. Path assertions use `strings.HasSuffix(r.URL.Path, ":generateContent")` etc. because of the `/v1beta/models/` prefix.

---

## Task 0 — Module wiring (genai dep + dev replace)

**Files:**
- Modify: `go.mod`
- Modify: `go.sum` (via tidy)

- [ ] **Step 1: Add the genai require + contract dev replace.** Run from the repo root:

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers
GOWORK=off go get google.golang.org/genai@v1.59.0
```

Then add a dev-time local `replace` so the package builds against the un-tagged contract that has `ImageGenerator` (path is the sibling checkout that the image-gen-1-contract plan edits):

```bash
GOWORK=off go mod edit -replace github.com/costa92/llm-agent-contract=../llm-agent-contract
GOWORK=off go mod edit -require github.com/costa92/llm-agent-contract@v0.3.0
```

- [ ] **Step 2: Tidy and verify the module graph resolves.**

Run: `GOWORK=off go mod tidy`
Expected: no error; `go.mod` now lists `google.golang.org/genai v1.59.0` under `require` and the `replace github.com/costa92/llm-agent-contract => ../llm-agent-contract` line.

- [ ] **Step 3: Verify the contract surface is present** (sanity that v0.3.0 symbols resolve through the replace):

```bash
GOWORK=off go doc github.com/costa92/llm-agent-contract/llm.ImageGenerator
```
Expected: prints the `ImageGenerator` interface (the GenerateImage method). If it errors `no such symbol`, the contract checkout at `../llm-agent-contract` has not landed the image-gen-1-contract plan — STOP and land that first.

- [ ] **Step 4: Commit.**

```bash
git add go.mod go.sum
git commit -m "build(google): add genai SDK dep + dev replace for contract v0.3.0

The google provider needs the official google.golang.org/genai SDK and the
ImageGenerator capability from contract v0.3.0. Pin genai v1.59.0 and add a
dev-time local replace to the sibling contract checkout; the replace-guard
hook strips it on the final PR once v0.3.0 is tagged."
```

---

## Task 1 — Options, constructor, capability gating

**Files:**
- Create: `google/options.go`
- Create: `google/doc.go`
- Create: `google/google.go` (struct + asserts only this task — methods added in Task 2)
- Test: `google/google_test.go`

- [ ] **Step 1: Write the failing tests** in `google/google_test.go`. Covers: `New` requires `WithModel`; chat-model capabilities (Tools true, others false); image-model and imagen-model capabilities; embed-model capabilities + `EmbedDimensions`.

```go
package google

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

func TestInfo_ChatModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := g.Info()
	if info.Provider != "google" {
		t.Fatalf("Provider = %q, want google", info.Provider)
	}
	if info.Model != "gemini-2.5-flash" {
		t.Fatalf("Model = %q, want gemini-2.5-flash", info.Model)
	}
	if !info.Capabilities.Tools {
		t.Errorf("Tools = false, want true for chat model")
	}
	if info.Capabilities.ImageGeneration || info.Capabilities.Embeddings {
		t.Errorf("Capabilities = %+v, want image_generation=false embeddings=false", info.Capabilities)
	}
}

func TestInfo_GeminiImageModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash-image"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !g.Info().Capabilities.ImageGeneration {
		t.Errorf("ImageGeneration = false, want true for gemini-*-image")
	}
}

func TestInfo_ImagenModel(t *testing.T) {
	g, err := New(WithModel("imagen-4.0-generate-001"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	caps := g.Info().Capabilities
	if !caps.ImageGeneration {
		t.Errorf("ImageGeneration = false, want true for imagen-*")
	}
	if caps.Tools {
		t.Errorf("Tools = true, want false for imagen model")
	}
}

func TestInfo_EmbedModel(t *testing.T) {
	g, err := New(WithModel("gemini-embedding-001"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !g.Info().Capabilities.Embeddings {
		t.Errorf("Embeddings = false, want true for embedding model")
	}
	if got := g.EmbedDimensions(); got != 3072 {
		t.Errorf("EmbedDimensions() = %d, want 3072", got)
	}
}

func TestEmbedDimensions_TextEmbedding004(t *testing.T) {
	g, err := New(WithModel("text-embedding-004"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 768 {
		t.Errorf("EmbedDimensions() = %d, want 768", got)
	}
}

func TestEmbedDimensions_WithDimensionsOverride(t *testing.T) {
	g, err := New(WithModel("gemini-embedding-001"), WithAPIKey("test-key"), WithDimensions(1536))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 1536 {
		t.Errorf("EmbedDimensions() = %d, want 1536 (overridden)", got)
	}
}

func TestEmbedDimensions_NonEmbedModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 0 {
		t.Errorf("EmbedDimensions() = %d, want 0 for non-embed model", got)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `GOWORK=off go test ./google/ -run 'TestNew_RequiresModel|TestInfo_|TestEmbedDimensions_'`
Expected: build failure — `undefined: New`, `undefined: WithModel`, etc. (RED).

- [ ] **Step 3: Write `google/doc.go`.**

```go
// Package google implements a Gemini adapter over the official
// google.golang.org/genai SDK (Gemini Developer API backend).
//
// The adapter satisfies llm.ChatModel, llm.ToolCaller, llm.ImageGenerator,
// and llm.Embedder. Capabilities reported via Info() are per-(provider ×
// model): the constructor binds one model, and Info() reflects what that
// model can do (Keystone K2). A gemini-2.5-flash instance does chat + tools
// but GenerateImage/Embed return llm.ErrCapabilityNotSupported; a
// gemini-2.5-flash-image / imagen-* instance generates images; a
// gemini-embedding-001 / text-embedding-004 instance embeds.
//
// Gemini has no system role: llm.Request.SystemPrompt maps to
// GenerateContentConfig.SystemInstruction. Streaming bridges the SDK's
// iter.Seq2 to the repo's pull-based llm.StreamReader via iter.Pull2;
// streamed tool calls arrive complete in one chunk. Images are always
// returned as inline bytes (never URL).
package google
```

- [ ] **Step 4: Write `google/options.go`.**

```go
package google

import (
	"context"
	"errors"
	"net/http"
	"os"
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
	if cfg.timeout > 0 {
		clientCfg.HTTPOptions.Timeout = &cfg.timeout
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
//   - ImageGeneration: gemini-*-image and imagen-* models.
//   - Embeddings: models whose id contains "embedding".
func capabilitiesForModel(model string) llm.Capabilities {
	return llm.Capabilities{
		Tools:           isChatModel(model),
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
```

> The `strings` import is added to `options.go` by the classifiers. Add it to the import block:

```go
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
```

- [ ] **Step 5: Write `google/google.go`** with the struct, compile-asserts, `Info`, and `EmbedDimensions` (the other methods land in later tasks; `EmbedDimensions` is here because `embed.go` only adds `Embed`, and Task 1 tests it).

```go
package google

import (
	"google.golang.org/genai"

	"github.com/costa92/llm-agent-contract/llm"
)

var (
	_ llm.ChatModel      = (*Google)(nil)
	_ llm.ToolCaller     = (*Google)(nil)
	_ llm.ImageGenerator = (*Google)(nil)
	_ llm.Embedder       = (*Google)(nil)
)

// Google is a Gemini provider bound to one model. Safe for concurrent use.
type Google struct {
	client     *genai.Client
	info       llm.ProviderInfo
	tools      []llm.Tool
	taskType   string
	dimensions int
}

// Info returns the bound (provider × model) identity and capabilities.
func (g *Google) Info() llm.ProviderInfo { return g.info }

// EmbedDimensions returns the embedding width for the bound model, honoring
// WithDimensions; 0 when the bound model is not an embedding model.
func (g *Google) EmbedDimensions() int {
	if !g.info.Capabilities.Embeddings {
		return 0
	}
	if g.dimensions > 0 {
		return g.dimensions
	}
	switch g.info.Model {
	case "gemini-embedding-001":
		return 3072
	case "text-embedding-004":
		return 768
	default:
		return 0
	}
}
```

> **Compile note:** `google.go` references `llm.ImageGenerator` and `llm.Embedder` in the assert block but `GenerateImage`/`Embed`/`Generate`/`Stream`/`WithTools` are not yet defined. The `var _ llm.X = (*Google)(nil)` asserts will FAIL to compile until Tasks 2/5/6 add those methods. To keep Task 1 independently green, **comment out the `_ llm.ChatModel`, `_ llm.ToolCaller`, `_ llm.ImageGenerator`, and `_ llm.Embedder` assert lines in this task** and uncomment them in the task that completes each interface (Task 2 uncomments ChatModel+ToolCaller; Task 5 uncomments Embedder; Task 6 uncomments ImageGenerator). Leave a `// uncommented in Task N` note on each. Alternatively, implement Tasks 1-6 as one branch and only run the asserts at the end — but per TDD, comment-then-uncomment keeps each task's `go build` green.

Concretely, in Task 1 write the assert block as:

```go
var (
	// _ llm.ChatModel      = (*Google)(nil) // uncommented in Task 2
	// _ llm.ToolCaller     = (*Google)(nil) // uncommented in Task 2
	// _ llm.ImageGenerator = (*Google)(nil) // uncommented in Task 6
	_ llm.Embedder = (*Google)(nil) // EmbedDimensions present; Embed added Task 5
)
```

> Wait — `llm.Embedder` requires BOTH `Embed` and `EmbedDimensions`. Since `Embed` is not defined until Task 5, the `_ llm.Embedder` assert also fails in Task 1. So **comment out ALL four asserts in Task 1** and uncomment per completing task:

```go
var (
	// Interface asserts are enabled as each interface's methods land:
	// _ llm.ChatModel      = (*Google)(nil) // Task 2 (Generate/Stream/Info)
	// _ llm.ToolCaller     = (*Google)(nil) // Task 2 (+ WithTools)
	// _ llm.Embedder       = (*Google)(nil) // Task 5 (+ Embed)
	// _ llm.ImageGenerator = (*Google)(nil) // Task 6 (+ GenerateImage)
	_ = (*Google)(nil)
)
```

- [ ] **Step 6: Run tests to verify they pass.**

Run: `GOWORK=off go test ./google/ -run 'TestNew_RequiresModel|TestInfo_|TestEmbedDimensions_'`
Expected: PASS.

Run: `GOWORK=off go vet ./google/` — Expected: clean.

- [ ] **Step 7: Commit.**

```bash
git add google/doc.go google/options.go google/google.go google/google_test.go
git commit -m "feat(google): options, constructor, K2 capability gating

Bind one Gemini model per instance and derive capabilities from the model id
(gemini-* chat => tools; gemini-*-image/imagen-* => image; *embedding* =>
embeddings). genai.NewClient is configured for BackendGeminiAPI with optional
httptest BaseURL, custom HTTP client, timeout, and extra headers. Interface
asserts are staged on per task as methods land."
```

---

## Task 2 — Chat: Generate, WithTools, Info; request/response mapping

**Files:**
- Create: `google/map.go`
- Modify: `google/google.go` (add `Generate`, `WithTools`; uncomment ChatModel+ToolCaller asserts)
- Test: `google/google_test.go` (add generate + tool-call tests)

- [ ] **Step 1: Write the failing tests** (append to `google/google_test.go`). Covers: happy-path generate (system prompt → SystemInstruction, text + finish + usage), and a function-call response → `llm.ToolCall`.

```go
import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

// newTestServer builds an httptest server and a Google bound to model on it.
func newTestServer(t *testing.T, model string, handler http.HandlerFunc) *Google {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	g, err := New(WithModel(model), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return g
}

func TestGenerate_Happy(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":generateContent") {
			t.Errorf("path = %s, want suffix :generateContent", r.URL.Path)
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("x-goog-api-key = %q, want test-key", got)
		}
		body, _ := io.ReadAll(r.Body)
		s := string(body)
		if !strings.Contains(s, `"systemInstruction"`) {
			t.Errorf("body missing systemInstruction: %s", s)
		}
		if !strings.Contains(s, `"role":"user"`) {
			t.Errorf("body missing user role: %s", s)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[{"text":"Hello there"}]},"finishReason":"STOP","index":0}],
			"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6},
			"modelVersion":"gemini-2.5-flash"
		}`))
	})

	resp, err := g.Generate(context.Background(), llm.Request{
		SystemPrompt: "Be brief.",
		Messages:     []llm.Message{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Text != "Hello there" {
		t.Errorf("Text = %q, want Hello there", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Errorf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Provider != "google" {
		t.Errorf("Provider = %q, want google", resp.Provider)
	}
	if resp.Usage.InputTokens != 4 || resp.Usage.OutputTokens != 2 || resp.Usage.TotalTokens != 6 {
		t.Errorf("Usage = %+v, want 4/2/6", resp.Usage)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Errorf("Usage.Source = %q, want reported", resp.Usage.Source)
	}
}

func TestGenerate_ToolCall(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s := string(body)
		// tool schema forwarded via parametersJsonSchema
		if !strings.Contains(s, `"functionDeclarations"`) {
			t.Errorf("body missing functionDeclarations: %s", s)
		}
		if !strings.Contains(s, `"parametersJsonSchema"`) {
			t.Errorf("body missing parametersJsonSchema: %s", s)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[
				{"functionCall":{"id":"call_1","name":"get_weather","args":{"city":"Paris"}}}
			]},"finishReason":"STOP","index":0}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}
		}`))
	})

	tc, err := g.WithTools([]llm.Tool{{
		Name:        "get_weather",
		Description: "Get weather for a city",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}})
	if err != nil {
		t.Fatalf("WithTools: %v", err)
	}
	resp, err := tc.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "Weather in Paris?"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(resp.ToolCalls))
	}
	call := resp.ToolCalls[0]
	if call.ID != "call_1" || call.Name != "get_weather" {
		t.Errorf("ToolCall id/name = %s/%s, want call_1/get_weather", call.ID, call.Name)
	}
	var args map[string]any
	if err := json.Unmarshal(call.Arguments, &args); err != nil {
		t.Fatalf("Arguments not valid JSON: %v (%s)", err, call.Arguments)
	}
	if args["city"] != "Paris" {
		t.Errorf("args.city = %v, want Paris", args["city"])
	}
}
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `GOWORK=off go test ./google/ -run 'TestGenerate_'`
Expected: build failure — `g.Generate undefined`, `g.WithTools undefined` (RED).

- [ ] **Step 3: Write `google/map.go`.**

```go
package google

import (
	"encoding/json"

	"github.com/costa92/llm-agent-contract/llm"
	"google.golang.org/genai"
)

// toContents maps the dialog turns to Gemini Contents. Gemini roles are only
// user/model: assistant => model; system/tool turns are skipped (system is
// lifted to SystemInstruction in toGenConfig).
func toContents(req llm.Request) []*genai.Content {
	contents := make([]*genai.Content, 0, len(req.Messages))
	for _, m := range req.Messages {
		var role string
		switch m.Role {
		case "user":
			role = genai.RoleUser
		case "assistant":
			role = genai.RoleModel
		default:
			continue
		}
		contents = append(contents, &genai.Content{
			Role:  role,
			Parts: []*genai.Part{{Text: m.Content}},
		})
	}
	return contents
}

// toGenConfig maps request knobs + bound tools to a GenerateContentConfig.
// Returns nil when there is nothing to configure (no system prompt, no
// sampling overrides, no tools).
func (g *Google) toGenConfig(req llm.Request) *genai.GenerateContentConfig {
	cfg := &genai.GenerateContentConfig{}
	set := false
	if req.SystemPrompt != "" {
		cfg.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: req.SystemPrompt}},
		}
		set = true
	}
	if req.Temperature != nil {
		cfg.Temperature = genai.Ptr(*req.Temperature)
		set = true
	}
	if req.MaxOutputTokens > 0 {
		cfg.MaxOutputTokens = int32(req.MaxOutputTokens)
		set = true
	}
	if len(g.tools) > 0 {
		decls := make([]*genai.FunctionDeclaration, 0, len(g.tools))
		for _, tool := range g.tools {
			decl := &genai.FunctionDeclaration{
				Name:        tool.Name,
				Description: tool.Description,
			}
			if len(tool.Parameters) > 0 {
				var schema map[string]any
				if err := json.Unmarshal(tool.Parameters, &schema); err == nil {
					decl.ParametersJsonSchema = schema
				}
			}
			decls = append(decls, decl)
		}
		cfg.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
		set = true
	}
	if !set {
		return nil
	}
	return cfg
}

// fromResponse maps a GenerateContentResponse to the repo's llm.Response.
func (g *Google) fromResponse(resp *genai.GenerateContentResponse) llm.Response {
	out := llm.Response{
		Text:         resp.Text(),
		Provider:     "google",
		Model:        g.info.Model,
		FinishReason: llm.FinishReasonUnknown,
		ToolCalls:    toToolCalls(resp),
	}
	if resp.ModelVersion != "" {
		out.Model = resp.ModelVersion
	}
	if len(resp.Candidates) > 0 {
		out.FinishReason = mapFinishReason(resp.Candidates[0].FinishReason)
	}
	if um := resp.UsageMetadata; um != nil {
		out.Usage = llm.Usage{
			InputTokens:  int(um.PromptTokenCount),
			OutputTokens: int(um.CandidatesTokenCount),
			TotalTokens:  int(um.TotalTokenCount),
			Source:       llm.UsageReported,
		}
	}
	return out
}

// toToolCalls extracts function calls from the first candidate, re-marshalling
// Args (map[string]any) back to a JSON string for llm.ToolCall.Arguments.
func toToolCalls(resp *genai.GenerateContentResponse) []llm.ToolCall {
	fcs := resp.FunctionCalls()
	if len(fcs) == 0 {
		return nil
	}
	out := make([]llm.ToolCall, 0, len(fcs))
	for _, fc := range fcs {
		args, err := json.Marshal(fc.Args)
		if err != nil || fc.Args == nil {
			args = []byte("{}")
		}
		out = append(out, llm.ToolCall{
			ID:        fc.ID,
			Name:      fc.Name,
			Arguments: args,
		})
	}
	return out
}

// mapFinishReason maps Gemini finish reasons to the contract's reasons.
func mapFinishReason(fr genai.FinishReason) llm.FinishReason {
	switch fr {
	case genai.FinishReasonStop:
		return llm.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		return llm.FinishReasonLength
	case genai.FinishReasonSafety, genai.FinishReasonRecitation:
		return llm.FinishReasonContentFilter
	default:
		return llm.FinishReasonUnknown
	}
}
```

- [ ] **Step 4: Add `Generate` and `WithTools` to `google/google.go`, and uncomment the ChatModel+ToolCaller asserts.**

Add the methods:

```go
import (
	"context"

	"google.golang.org/genai"

	"github.com/costa92/llm-agent-contract/llm"
)

// Generate runs a one-shot chat completion against the bound model.
func (g *Google) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	resp, err := g.client.Models.GenerateContent(ctx, g.info.Model, toContents(req), g.toGenConfig(req))
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	return g.fromResponse(resp), nil
}

// WithTools returns a new ToolCaller bound to the given tools (immutable;
// the receiver is unchanged).
func (g *Google) WithTools(tools []llm.Tool) (llm.ToolCaller, error) {
	cp := *g
	cp.tools = append([]llm.Tool(nil), tools...)
	return &cp, nil
}
```

Change the assert block to enable ChatModel + ToolCaller (note: `Stream` is added in Task 3 — `ChatModel` requires `Generate`+`Stream`+`Info`, so the `_ llm.ChatModel` assert still fails until Task 3). To keep this task's build green, enable ToolCaller only AFTER ChatModel is satisfiable. **Therefore: in Task 2, add a temporary local assert for the methods present, and enable the real interface asserts in Task 3.** Concretely, leave the four interface asserts commented in Task 2; Task 3 (which adds `Stream`) uncomments `_ llm.ChatModel` and `_ llm.ToolCaller`.

> Rationale: `Generate` + `WithTools` alone do not satisfy `ChatModel` (missing `Stream`) nor `ToolCaller` (embeds ChatModel). The compiler would reject the asserts. So Task 2 adds the methods + `errors.go` (`wrapErr`, Task 4 is pulled forward minimally) and Task 3 wires the asserts once `Stream` exists.

> **Dependency note:** `wrapErr` is referenced by `Generate`. Implement a minimal `google/errors.go` now (full version is Task 4) so Task 2 compiles. Write `google/errors.go`:

```go
package google

import (
	"context"
	"errors"
	"net"

	"github.com/costa92/llm-agent-contract/llm"
	"google.golang.org/genai"
)

// wrapErr maps genai SDK errors to the repo's typed llm.* errors.
func wrapErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return &llm.TransientError{Provider: "google", Wrapped: err}
	}
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		switch {
		case apiErr.Code == 401 || apiErr.Code == 403:
			return &llm.AuthError{Provider: "google", Wrapped: err}
		case apiErr.Code == 429:
			return &llm.RateLimitError{Provider: "google", Wrapped: err}
		case apiErr.Code >= 500:
			return &llm.TransientError{Provider: "google", Wrapped: err}
		default:
			return &llm.InvalidRequestError{Provider: "google", Wrapped: err}
		}
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return &llm.TransientError{Provider: "google", Wrapped: err}
	}
	return &llm.InvalidRequestError{Provider: "google", Wrapped: err}
}
```

- [ ] **Step 5: Run tests to verify they pass.**

Run: `GOWORK=off go test ./google/ -run 'TestGenerate_'`
Expected: PASS (both `TestGenerate_Happy` and `TestGenerate_ToolCall`).

- [ ] **Step 6: Commit.**

```bash
git add google/map.go google/google.go google/errors.go google/google_test.go
git commit -m "feat(google): chat Generate + WithTools + request/response mapping

Map llm.Request to Gemini Contents (user/model roles, system prompt lifted to
SystemInstruction) and GenerateContentConfig (temperature *float32 via
genai.Ptr, MaxOutputTokens int32, tools via FunctionDeclaration.
ParametersJsonSchema from raw JSON schema). Decode Text/FunctionCalls/finish/
usage back to llm.Response. WithTools is immutable. Minimal wrapErr added for
Generate; full error mapping in a later task."
```

---

## Task 3 — Streaming (iter.Pull2 bridge)

**Files:**
- Modify: `google/google.go` (add `Stream`, `googleStreamReader`; enable ChatModel + ToolCaller asserts)
- Test: `google/google_test.go` (add stream tests)

- [ ] **Step 1: Write the failing tests** (append). Covers: text-delta streaming with a final usage chunk, and a complete-in-one-chunk tool call (Start+ArgsDelta+End).

```go
import (
	"errors"
	"io"
)

// readAll drains a StreamReader into ordered events (excluding the io.EOF).
func readAll(t *testing.T, sr llm.StreamReader) []llm.StreamEvent {
	t.Helper()
	defer sr.Close()
	var events []llm.StreamEvent
	for {
		ev, err := sr.Next()
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		events = append(events, ev)
	}
}

func TestStream_TextDeltas(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":streamGenerateContent") {
			t.Errorf("path = %s, want suffix :streamGenerateContent", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunks := []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]},"index":0}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"index":0}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}`,
		}
		for _, c := range chunks {
			_, _ = io.WriteString(w, "data: "+c+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
	})

	sr, err := g.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := readAll(t, sr)

	var text string
	var sawDone bool
	for _, ev := range events {
		switch ev.Kind {
		case llm.EventTextDelta:
			text += ev.Text
		case llm.EventDone:
			sawDone = true
			if ev.FinishReason != llm.FinishReasonStop {
				t.Errorf("Done.FinishReason = %q, want stop", ev.FinishReason)
			}
			if ev.Usage == nil || ev.Usage.TotalTokens != 4 {
				t.Errorf("Done.Usage = %+v, want total=4", ev.Usage)
			}
		}
	}
	if text != "Hello" {
		t.Errorf("text = %q, want Hello", text)
	}
	if !sawDone {
		t.Error("no EventDone emitted")
	}
}

func TestStream_ToolCallCompleteInOneChunk(t *testing.T) {
	g0 := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunk := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call_9","name":"lookup","args":{"q":"x"}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}}`
		_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	})
	g, err := g0.WithTools([]llm.Tool{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools: %v", err)
	}

	sr, err := g.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "go"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := readAll(t, sr)

	var sawStart, sawArgs, sawEnd bool
	var argsJSON string
	for _, ev := range events {
		switch ev.Kind {
		case llm.EventToolCallStart:
			sawStart = true
			if ev.ToolCall == nil || ev.ToolCall.Name != "lookup" || ev.ToolCall.ID != "call_9" {
				t.Errorf("Start ToolCall = %+v, want name=lookup id=call_9", ev.ToolCall)
			}
		case llm.EventToolCallArgsDelta:
			sawArgs = true
			if ev.ToolCall != nil {
				argsJSON = ev.ToolCall.ArgsDelta
			}
		case llm.EventToolCallEnd:
			sawEnd = true
		}
	}
	if !sawStart || !sawArgs || !sawEnd {
		t.Fatalf("tool events start=%v args=%v end=%v, want all true", sawStart, sawArgs, sawEnd)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		t.Fatalf("ArgsDelta not valid JSON: %v (%s)", err, argsJSON)
	}
	if args["q"] != "x" {
		t.Errorf("args.q = %v, want x", args["q"])
	}
}

// Accumulate parity: the stream reduces to the same tool call via the
// contract's AccumulateStream helper.
func TestStream_AccumulateParity(t *testing.T) {
	g0 := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"c","name":"f","args":{"a":1}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"totalTokenCount":7}}`
		_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
	})
	g, _ := g0.WithTools([]llm.Tool{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}})
	sr, err := g.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "f" {
		t.Fatalf("accumulated ToolCalls = %+v, want one named f", resp.ToolCalls)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `GOWORK=off go test ./google/ -run 'TestStream_'`
Expected: build failure — `g.Stream undefined` (RED).

- [ ] **Step 3: Add `Stream` + `googleStreamReader` to `google/google.go`** and enable the ChatModel/ToolCaller asserts.

Update imports and add:

```go
import (
	"context"
	"io"
	"iter"
	"sync"

	"google.golang.org/genai"

	"github.com/costa92/llm-agent-contract/llm"
)

// Stream runs a streaming chat completion. The returned reader lazily opens
// the upstream iterator on first Next() and MUST be Closed by the caller.
func (g *Google) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	return &googleStreamReader{
		open: func() iter.Seq2[*genai.GenerateContentResponse, error] {
			return g.client.Models.GenerateContentStream(ctx, g.info.Model, toContents(req), g.toGenConfig(req))
		},
	}, nil
}

// googleStreamReader bridges genai's iter.Seq2 to the pull-based
// llm.StreamReader via iter.Pull2. One upstream chunk decomposes into many
// llm.StreamEvents; streamed tool calls arrive complete in one chunk so
// Start+ArgsDelta(full)+End are emitted together per call.
type googleStreamReader struct {
	mu     sync.Mutex
	open   func() iter.Seq2[*genai.GenerateContentResponse, error]
	next   func() (*genai.GenerateContentResponse, error, bool)
	stop   func()
	queue  []llm.StreamEvent
	closed bool

	lastFinish llm.FinishReason
	lastUsage  *llm.Usage
	sawAny     bool
}

func (r *googleStreamReader) Next() (llm.StreamEvent, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for {
		if r.closed {
			return llm.StreamEvent{}, io.EOF
		}
		if len(r.queue) > 0 {
			ev := r.queue[0]
			r.queue = r.queue[1:]
			return ev, nil
		}
		if r.next == nil {
			r.next, r.stop = iter.Pull2(r.open())
		}
		chunk, err, ok := r.next()
		if !ok {
			// Upstream exhausted: emit terminal Done once.
			r.stop()
			r.closed = true
			usage := r.lastUsage
			if usage == nil {
				usage = &llm.Usage{Source: llm.UsageUnknown}
			}
			return llm.StreamEvent{
				Kind:         llm.EventDone,
				Usage:        usage,
				FinishReason: r.lastFinish,
			}, nil
		}
		if err != nil {
			r.stop()
			r.closed = true
			return llm.StreamEvent{}, wrapErr(err)
		}
		r.sawAny = true
		r.queue = append(r.queue, r.chunkEvents(chunk)...)
	}
}

func (r *googleStreamReader) chunkEvents(chunk *genai.GenerateContentResponse) []llm.StreamEvent {
	var events []llm.StreamEvent
	if len(chunk.Candidates) == 0 {
		return events
	}
	cand := chunk.Candidates[0]
	if cand.Content != nil {
		for i, part := range cand.Content.Parts {
			if part.Text != "" {
				events = append(events, llm.StreamEvent{Kind: llm.EventTextDelta, Text: part.Text})
			}
			if part.FunctionCall != nil {
				fc := part.FunctionCall
				args, err := json.Marshal(fc.Args)
				if err != nil || fc.Args == nil {
					args = []byte("{}")
				}
				events = append(events,
					llm.StreamEvent{Kind: llm.EventToolCallStart, ToolCall: &llm.ToolCallDelta{Index: i, ID: fc.ID, Name: fc.Name}},
					llm.StreamEvent{Kind: llm.EventToolCallArgsDelta, ToolCall: &llm.ToolCallDelta{Index: i, ArgsDelta: string(args)}},
					llm.StreamEvent{Kind: llm.EventToolCallEnd, ToolCall: &llm.ToolCallDelta{Index: i}},
				)
			}
		}
	}
	if cand.FinishReason != "" {
		r.lastFinish = mapFinishReason(cand.FinishReason)
	}
	if um := chunk.UsageMetadata; um != nil {
		r.lastUsage = &llm.Usage{
			InputTokens:  int(um.PromptTokenCount),
			OutputTokens: int(um.CandidatesTokenCount),
			TotalTokens:  int(um.TotalTokenCount),
			Source:       llm.UsageReported,
		}
	}
	return events
}

func (r *googleStreamReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	if r.stop != nil {
		r.stop()
		r.stop = nil
	}
	return nil
}
```

> **Import note:** `chunkEvents` uses `encoding/json` (`json.Marshal`). Add `"encoding/json"` to `google.go`'s import block. Final `google.go` imports: `context`, `encoding/json`, `io`, `iter`, `sync`, `google.golang.org/genai`, `github.com/costa92/llm-agent-contract/llm`.

Now enable the interface asserts (Generate+Stream+Info satisfy ChatModel; +WithTools satisfies ToolCaller):

```go
var (
	_ llm.ChatModel  = (*Google)(nil)
	_ llm.ToolCaller = (*Google)(nil)
	// _ llm.Embedder       = (*Google)(nil) // Task 5
	// _ llm.ImageGenerator = (*Google)(nil) // Task 6
)
```

- [ ] **Step 4: Run tests to verify they pass.**

Run: `GOWORK=off go test ./google/ -run 'TestStream_'`
Expected: PASS (text deltas, complete-in-one-chunk tool call, accumulate parity).

- [ ] **Step 5: Goroutine-leak check.** The pull-based reader uses no goroutines of its own, but `iter.Pull2` spins one for the range-over-func; `stop()` in both the exhausted path and `Close()` releases it. Verify with goleak by adding a `TestMain` (matches the existing adapters per spec §Testing):

```go
import "go.uber.org/goleak"

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}
```

Run: `GOWORK=off go test ./google/ -run 'TestStream_'`
Expected: PASS with no leak report. If goleak flags the `iter.Pull2` goroutine, it means a test path returned early without draining to EOF and without `Close()` — `readAll` defers `Close()` so this should be clean. Ensure `go.uber.org/goleak` is in `go.mod` (it is already used repo-wide; `GOWORK=off go mod tidy` if the import is new to this module path).

- [ ] **Step 6: Commit.**

```bash
git add google/google.go google/google_test.go
git commit -m "feat(google): streaming via iter.Pull2 bridge

GenerateContentStream returns iter.Seq2; bridge it to the repo's pull-based
StreamReader with iter.Pull2 (next/stop). Emit text deltas per part; streamed
function calls arrive complete in one chunk so Start+ArgsDelta(full args)+End
are emitted together, keyed by part index (no cross-chunk accumulation).
Terminal EventDone carries the last finish reason + usage. stop() releases the
range-over-func goroutine on exhaustion and on Close. Enable ChatModel/
ToolCaller asserts."
```

---

## Task 4 — Full error mapping (PromptFeedback nil-guard + status switch)

**Files:**
- Modify: `google/errors.go` (already has the core switch from Task 2; this task adds the blocked-prompt path + a test)
- Test: `google/google_test.go` (add error-mapping tests)

- [ ] **Step 1: Write the failing tests** (append). Covers: 401 → AuthError, 429 → RateLimitError, 500 → TransientError, 400 → InvalidRequestError, and a blocked-prompt response (no candidates, PromptFeedback.BlockReason) → InvalidRequestError.

```go
func TestWrapErr_StatusMapping(t *testing.T) {
	cases := []struct {
		code   int
		expect func(error) bool
	}{
		{401, func(e error) bool { var a *llm.AuthError; return errorsAs(e, &a) }},
		{403, func(e error) bool { var a *llm.AuthError; return errorsAs(e, &a) }},
		{429, func(e error) bool { var a *llm.RateLimitError; return errorsAs(e, &a) }},
		{500, func(e error) bool { var a *llm.TransientError; return errorsAs(e, &a) }},
		{503, func(e error) bool { var a *llm.TransientError; return errorsAs(e, &a) }},
		{400, func(e error) bool { var a *llm.InvalidRequestError; return errorsAs(e, &a) }},
	}
	for _, tc := range cases {
		g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(tc.code)
			_, _ = w.Write([]byte(`{"error":{"code":` + itoa(tc.code) + `,"status":"X","message":"boom"}}`))
		})
		_, err := g.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "x"}}})
		if err == nil {
			t.Fatalf("code %d: Generate err = nil, want typed error", tc.code)
		}
		if !tc.expect(err) {
			t.Errorf("code %d: wrong error type: %v", tc.code, err)
		}
	}
}

func TestGenerate_BlockedPrompt(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// 200 OK but no candidates; prompt blocked.
		_, _ = w.Write([]byte(`{"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"blocked"}}`))
	})
	_, err := g.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "x"}}})
	if err == nil {
		t.Fatal("blocked prompt: Generate err = nil, want InvalidRequestError")
	}
	var inv *llm.InvalidRequestError
	if !errorsAs(err, &inv) {
		t.Errorf("blocked prompt: err = %v, want *llm.InvalidRequestError", err)
	}
}
```

> Add these small test helpers once near the top of `google_test.go` (avoid importing `errors`/`strconv` repeatedly inline):

```go
import (
	goerrors "errors"
	"strconv"
)

func errorsAs(err error, target any) bool { return goerrors.As(err, target) }
func itoa(n int) string                   { return strconv.Itoa(n) }
```

> If `errors` is already imported (from Task 3's `errors.Is`), use that import alias consistently — pick ONE alias for the `errors` stdlib package across the test file (the plan uses `goerrors` here; if Task 3 already imported `errors`, reuse `errors.As` directly and drop the `errorsAs` helper). Keep the test file's imports consistent: a single `errors` import, used for both `errors.Is(io.EOF)` and `errors.As`.

- [ ] **Step 2: Run tests to verify they fail / partially pass.**

Run: `GOWORK=off go test ./google/ -run 'TestWrapErr_StatusMapping|TestGenerate_BlockedPrompt'`
Expected: `TestWrapErr_StatusMapping` PASSES already (Task 2's `wrapErr` handles status codes). `TestGenerate_BlockedPrompt` FAILS — a 200 response with no candidates currently returns a `llm.Response{}` with empty text and NO error (RED for the blocked-prompt path).

- [ ] **Step 3: Add the blocked-prompt guard.** Gemini returns HTTP 200 with `promptFeedback.blockReason` and zero candidates when a prompt is blocked. Surface that as an error rather than a silent empty response. Add a guard in `Generate` (in `google.go`) BEFORE mapping the response:

```go
// Generate runs a one-shot chat completion against the bound model.
func (g *Google) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	resp, err := g.client.Models.GenerateContent(ctx, g.info.Model, toContents(req), g.toGenConfig(req))
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	if blocked := blockedPromptErr(resp); blocked != nil {
		return llm.Response{}, blocked
	}
	return g.fromResponse(resp), nil
}
```

Add `blockedPromptErr` to `google/errors.go`:

```go
// blockedPromptErr returns an InvalidRequestError when the response carries a
// PromptFeedback block reason and no candidates (Gemini returns HTTP 200 in
// this case). Returns nil otherwise.
func blockedPromptErr(resp *genai.GenerateContentResponse) error {
	if resp == nil {
		return nil
	}
	if len(resp.Candidates) == 0 && resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
		return &llm.InvalidRequestError{
			Provider: "google",
			Wrapped: fmt.Errorf("prompt blocked: %s (%s)",
				resp.PromptFeedback.BlockReason, resp.PromptFeedback.BlockReasonMessage),
		}
	}
	return nil
}
```

> Add `"fmt"` to `errors.go`'s import block. Final `errors.go` imports: `context`, `errors`, `fmt`, `net`, `google.golang.org/genai`, `github.com/costa92/llm-agent-contract/llm`.

- [ ] **Step 4: Run tests to verify they pass.**

Run: `GOWORK=off go test ./google/ -run 'TestWrapErr_StatusMapping|TestGenerate_BlockedPrompt'`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add google/errors.go google/google.go google/google_test.go
git commit -m "feat(google): map genai errors + guard blocked prompts

genai.APIError is a value type (errors.As into genai.APIError, not pointer);
switch on Code for 401/403=>Auth, 429=>RateLimit, 5xx=>Transient, else
Invalid; ctx.Canceled passes through, DeadlineExceeded/net.Error => Transient.
Gemini blocks prompts with HTTP 200 + PromptFeedback.BlockReason and no
candidates; surface that as InvalidRequestError instead of a silent empty
response."
```

---

## Task 5 — Embeddings (Embed via EmbedContent)

**Files:**
- Create: `google/embed.go`
- Modify: `google/google.go` (enable Embedder assert)
- Test: `google/google_test.go` (add embed tests)

- [ ] **Step 1: Write the failing tests** (append). Covers: happy-path embed (vector order/length, zero usage with UsageUnknown, TaskType + OutputDimensionality forwarded), and embed gating on a non-embed model.

```go
func TestEmbed_Happy(t *testing.T) {
	g := newTestServerEmbed(t, "gemini-embedding-001", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":embedContent") && !strings.HasSuffix(r.URL.Path, ":batchEmbedContents") {
			t.Errorf("path = %s, want embed suffix", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"embeddings":[
				{"values":[0.1,0.2,0.3]},
				{"values":[0.4,0.5,0.6]}
			]
		}`))
	}, WithTaskType("RETRIEVAL_DOCUMENT"), WithDimensions(3))

	vecs, usage, err := g.Embed(context.Background(), []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("vectors len = %d, want 2", len(vecs))
	}
	if len(vecs[0]) != 3 || vecs[0][0] != 0.1 || vecs[1][2] != 0.6 {
		t.Errorf("vectors = %+v, want [[0.1 0.2 0.3] [0.4 0.5 0.6]]", vecs)
	}
	// Gemini Developer API reports no token usage.
	if usage.Source != llm.UsageUnknown || usage.TotalTokens != 0 {
		t.Errorf("usage = %+v, want zero/UsageUnknown", usage)
	}
}

func TestEmbed_EmptyInput(t *testing.T) {
	g, err := New(WithModel("gemini-embedding-001"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	vecs, _, err := g.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(nil): %v", err)
	}
	if len(vecs) != 0 {
		t.Errorf("vectors len = %d, want 0", len(vecs))
	}
}

func TestEmbed_NonEmbedModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, _, err = g.Embed(context.Background(), []string{"x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("Embed on chat model = %v, want ErrCapabilityNotSupported", err)
	}
}
```

> Add the embed-specific server helper (takes extra options) near `newTestServer`:

```go
func newTestServerEmbed(t *testing.T, model string, handler http.HandlerFunc, opts ...Option) *Google {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	all := append([]Option{WithModel(model), WithAPIKey("test-key"), WithBaseURL(server.URL)}, opts...)
	g, err := New(all...)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return g
}
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `GOWORK=off go test ./google/ -run 'TestEmbed_'`
Expected: build failure — `g.Embed undefined` (RED). (`TestEmbed_NonEmbedModel` also needs `Embed` to exist.)

- [ ] **Step 3: Write `google/embed.go`.**

```go
package google

import (
	"context"
	"fmt"

	"github.com/costa92/llm-agent-contract/llm"
	"google.golang.org/genai"
)

// Embed returns one vector per input text, in order. On the Gemini Developer
// API no token usage is reported, so Usage is zero with Source=UsageUnknown.
func (g *Google) Embed(ctx context.Context, texts []string) ([]llm.Vector, llm.Usage, error) {
	if !g.info.Capabilities.Embeddings {
		return nil, llm.Usage{}, fmt.Errorf("google: embeddings: %w", llm.ErrCapabilityNotSupported)
	}
	if len(texts) == 0 {
		return []llm.Vector{}, llm.Usage{Source: llm.UsageUnknown}, nil
	}

	contents := make([]*genai.Content, 0, len(texts))
	for _, txt := range texts {
		contents = append(contents, &genai.Content{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{{Text: txt}},
		})
	}

	var cfg *genai.EmbedContentConfig
	if g.taskType != "" || g.dimensions > 0 {
		cfg = &genai.EmbedContentConfig{}
		if g.taskType != "" {
			cfg.TaskType = g.taskType
		}
		if g.dimensions > 0 {
			cfg.OutputDimensionality = genai.Ptr(int32(g.dimensions))
		}
	}

	resp, err := g.client.Models.EmbedContent(ctx, g.info.Model, contents, cfg)
	if err != nil {
		return nil, llm.Usage{}, wrapErr(err)
	}

	vectors := make([]llm.Vector, 0, len(resp.Embeddings))
	for _, emb := range resp.Embeddings {
		vec := make(llm.Vector, len(emb.Values))
		copy(vec, emb.Values)
		vectors = append(vectors, vec)
	}
	return vectors, llm.Usage{Source: llm.UsageUnknown}, nil
}
```

- [ ] **Step 4: Enable the Embedder assert** in `google/google.go`:

```go
var (
	_ llm.ChatModel  = (*Google)(nil)
	_ llm.ToolCaller = (*Google)(nil)
	_ llm.Embedder   = (*Google)(nil)
	// _ llm.ImageGenerator = (*Google)(nil) // Task 6
)
```

- [ ] **Step 5: Run tests to verify they pass.**

Run: `GOWORK=off go test ./google/ -run 'TestEmbed_'`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add google/embed.go google/google.go google/google_test.go
git commit -m "feat(google): embeddings via EmbedContent

One genai.Content per input text; map Embeddings[].Values to llm.Vector in
order. Forward WithTaskType and WithDimensions (OutputDimensionality *int32).
The Gemini Developer API reports no token counts, so Usage is zero with
Source=UsageUnknown. Embed is model-gated: non-embed models return
ErrCapabilityNotSupported. Enable Embedder assert."
```

---

## Task 6 — Image generation (Imagen + Gemini-native inline)

**Files:**
- Create: `google/image.go`
- Modify: `google/google.go` (enable ImageGenerator assert)
- Test: `google/image_test.go`

- [ ] **Step 1: Write the failing tests** in `google/image_test.go`. Covers: Gemini-native inline image via `:generateContent` (TEXT+IMAGE modalities, drop text parts), Imagen via `:predict`, and image gating on a chat model.

```go
package google

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestGenerateImage_GeminiInline(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash-image", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":generateContent") {
			t.Errorf("path = %s, want :generateContent", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		s := string(body)
		if !strings.Contains(s, `"responseModalities"`) || !strings.Contains(s, `"TEXT"`) || !strings.Contains(s, `"IMAGE"`) {
			t.Errorf("body missing responseModalities [TEXT,IMAGE]: %s", s)
		}
		w.Header().Set("Content-Type", "application/json")
		// base64("PNGDATA") = UE5HREFUQQ== ; include a text part to confirm it is dropped.
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[
				{"text":"here you go"},
				{"inlineData":{"mimeType":"image/png","data":"UE5HREFUQQ=="}}
			]},"finishReason":"STOP","index":0}]
		}`))
	})

	resp, err := g.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "a fox"})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("Images len = %d, want 1 (text part dropped)", len(resp.Images))
	}
	if string(resp.Images[0].Bytes) != "PNGDATA" {
		t.Errorf("Bytes = %q, want PNGDATA", resp.Images[0].Bytes)
	}
	if resp.Images[0].MimeType != "image/png" {
		t.Errorf("MimeType = %q, want image/png", resp.Images[0].MimeType)
	}
	if resp.Provider != "google" {
		t.Errorf("Provider = %q, want google", resp.Provider)
	}
}

func TestGenerateImage_Imagen(t *testing.T) {
	g := newTestServer(t, "imagen-4.0-generate-001", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":predict") {
			t.Errorf("path = %s, want :predict", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		// base64("IMG1")=SU1HMQ==  base64("IMG2")=SU1HMg==
		_, _ = w.Write([]byte(`{
			"predictions":[
				{"bytesBase64Encoded":"SU1HMQ==","mimeType":"image/png"},
				{"bytesBase64Encoded":"SU1HMg==","mimeType":"image/png"}
			]
		}`))
	})

	resp, err := g.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "two cats", N: 2})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if len(resp.Images) != 2 {
		t.Fatalf("Images len = %d, want 2", len(resp.Images))
	}
	if string(resp.Images[0].Bytes) != "IMG1" || string(resp.Images[1].Bytes) != "IMG2" {
		t.Errorf("Bytes = %q/%q, want IMG1/IMG2", resp.Images[0].Bytes, resp.Images[1].Bytes)
	}
}

func TestGenerateImage_NonImageModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be called for a non-image model")
	}))
	t.Cleanup(server.Close)
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = g.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("GenerateImage on chat model = %v, want ErrCapabilityNotSupported", err)
	}
}
```

> **Verification flag:** `TestGenerateImage_Imagen` mocks the Imagen `:predict` response as `{"predictions":[{"bytesBase64Encoded":"...","mimeType":"..."}]}` — the public Imagen REST shape. The SDK's `generateImages` fromConverter maps this into `GenerateImagesResponse.GeneratedImages[].Image.ImageBytes`. If this test fails with empty `Images`, inspect the converter in `models.go` (`generateImages` / `generatedImageFromMldev`) at v1.59.0 and adjust the mock JSON keys to match what it reads. This is the single uncertain wire-shape in the plan (flagged in SDK verification above).

- [ ] **Step 2: Run tests to verify they fail.**

Run: `GOWORK=off go test ./google/ -run 'TestGenerateImage_'`
Expected: build failure — `g.GenerateImage undefined` (RED).

- [ ] **Step 3: Write `google/image.go`.**

```go
package google

import (
	"context"
	"fmt"
	"strings"

	"github.com/costa92/llm-agent-contract/llm"
	"google.golang.org/genai"
)

// GenerateImage produces images from a text prompt. Routing is by bound model
// id: imagen* uses the Imagen predict path (GenerateImages); a Gemini-native
// image model (gemini-*-image) uses GenerateContent with TEXT+IMAGE response
// modalities. Output is always inline bytes (Gemini never returns a URL).
func (g *Google) GenerateImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	if !g.info.Capabilities.ImageGeneration {
		return llm.ImageResponse{}, fmt.Errorf("google: image generation: %w", llm.ErrCapabilityNotSupported)
	}
	if strings.HasPrefix(g.info.Model, "imagen") {
		return g.generateImagen(ctx, req)
	}
	return g.generateGeminiImage(ctx, req)
}

// generateImagen calls the Imagen predict endpoint.
func (g *Google) generateImagen(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	cfg := &genai.GenerateImagesConfig{}
	if req.N > 0 {
		cfg.NumberOfImages = int32(req.N)
	}
	if ar, ok := req.Extra["aspect_ratio"].(string); ok && ar != "" {
		cfg.AspectRatio = ar
	}
	resp, err := g.client.Models.GenerateImages(ctx, g.info.Model, req.Prompt, cfg)
	if err != nil {
		return llm.ImageResponse{}, wrapErr(err)
	}
	images := make([]llm.GeneratedImage, 0, len(resp.GeneratedImages))
	for _, gi := range resp.GeneratedImages {
		if gi == nil || gi.Image == nil {
			continue
		}
		images = append(images, llm.GeneratedImage{
			Bytes:    gi.Image.ImageBytes,
			MimeType: gi.Image.MIMEType,
		})
	}
	return llm.ImageResponse{
		Images:   images,
		Provider: "google",
		Model:    g.info.Model,
		Usage:    llm.Usage{Source: llm.UsageUnknown},
	}, nil
}

// generateGeminiImage calls GenerateContent with TEXT+IMAGE modalities and
// extracts inline image parts (text parts are dropped). ResponseModalities
// MUST include TEXT — image-only is rejected for Gemini 2.5 Flash Image.
func (g *Google) generateGeminiImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	cfg := &genai.GenerateContentConfig{
		ResponseModalities: []string{"TEXT", "IMAGE"},
	}
	contents := []*genai.Content{{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{{Text: req.Prompt}},
	}}
	resp, err := g.client.Models.GenerateContent(ctx, g.info.Model, contents, cfg)
	if err != nil {
		return llm.ImageResponse{}, wrapErr(err)
	}
	if blocked := blockedPromptErr(resp); blocked != nil {
		return llm.ImageResponse{}, blocked
	}
	var images []llm.GeneratedImage
	if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			if part.InlineData != nil && len(part.InlineData.Data) > 0 {
				images = append(images, llm.GeneratedImage{
					Bytes:    part.InlineData.Data,
					MimeType: part.InlineData.MIMEType,
				})
			}
		}
	}
	return llm.ImageResponse{
		Images:   images,
		Provider: "google",
		Model:    g.info.Model,
		Usage:    llm.Usage{Source: llm.UsageUnknown},
	}, nil
}
```

- [ ] **Step 4: Enable the ImageGenerator assert** in `google/google.go`:

```go
var (
	_ llm.ChatModel      = (*Google)(nil)
	_ llm.ToolCaller     = (*Google)(nil)
	_ llm.Embedder       = (*Google)(nil)
	_ llm.ImageGenerator = (*Google)(nil)
)
```

- [ ] **Step 5: Run tests to verify they pass.**

Run: `GOWORK=off go test ./google/ -run 'TestGenerateImage_'`
Expected: PASS (Gemini inline, Imagen predict, non-image gating). If `TestGenerateImage_Imagen` fails on empty `Images`, see the verification flag in Step 1 and reconcile the mock JSON with the SDK converter.

- [ ] **Step 6: Commit.**

```bash
git add google/image.go google/google.go google/image_test.go
git commit -m "feat(google): image generation (Imagen + Gemini-native inline)

Route by bound model: imagen* => GenerateImages (predict) reading
GeneratedImages[].Image.ImageBytes; gemini-*-image => GenerateContent with
ResponseModalities [TEXT,IMAGE], extracting InlineData{Data,MIMEType} and
dropping text parts. Output is always inline Bytes (Gemini never returns a
URL). Model-gated: non-image models return ErrCapabilityNotSupported. Enable
ImageGenerator assert."
```

---

## Task 7 — Custom headers + full verification gate

**Files:**
- Test: `google/google_test.go` (add an extra-headers test)
- No production changes (WithExtraHeaders already wired in Task 1).

- [ ] **Step 1: Write the failing test** (append) — asserts `WithExtraHeaders` reaches the wire on a chat call.

```go
func TestExtraHeaders_Forwarded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Route-Tag"); got != "canary" {
			t.Errorf("X-Route-Tag = %q, want canary", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}]}`))
	}))
	t.Cleanup(server.Close)
	g, err := New(
		WithModel("gemini-2.5-flash"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithExtraHeaders(map[string]string{"X-Route-Tag": "canary"}),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := g.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "x"}}}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
}
```

- [ ] **Step 2: Run the test.**

Run: `GOWORK=off go test ./google/ -run 'TestExtraHeaders_Forwarded'`
Expected: PASS already (Task 1 set `clientCfg.HTTPOptions.Headers` from `WithExtraHeaders`, and the SDK applies `req.Header = patchedHTTPOptions.Headers`). If it FAILS (header not seen), the SDK overwrote the header map — verify api_client.go:320 (`req.Header = patchedHTTPOptions.Headers`) applies BEFORE the reserved `Content-Type`/`x-goog-api-key` Sets, which it does at v1.59.0; the custom header survives. This test locks that behavior.

- [ ] **Step 3: Full verification gate.**

```bash
GOWORK=off go test -race ./google/
GOWORK=off go vet ./google/
GOWORK=off gofmt -l google/
```
Expected: all tests PASS under `-race`, vet clean, `gofmt -l` prints nothing. If `gofmt -l` lists files, run `GOWORK=off gofmt -w google/` and re-run.

- [ ] **Step 4: Confirm the public surface is clean** (SDK types must not leak — CONVENTIONS.md §"SDK boundary discipline"):

```bash
GOWORK=off go doc ./google/ | grep -E 'func|type' | grep -i genai
```
Expected: NO matches (no `genai.*` on any exported signature). The only public API is `New`, `Option`, the `WithX` options, and methods on `*Google` returning `llm.*` types.

- [ ] **Step 5: Commit.**

```bash
git add google/google_test.go
git commit -m "test(google): assert WithExtraHeaders reaches the wire

Lock that custom headers injected via ClientConfig.HTTPOptions.Headers survive
the SDK's reserved-header Sets and appear on outbound requests, so compatible
gateways can route/auth. Full gate: race + vet + gofmt clean."
```

---

## Task 8 — Cross-repo finalize (drop dev replace, pin contract v0.3.0)

**Files:**
- Modify: `go.mod` (drop local `replace`, keep `require ... v0.3.0`)

> Do this ONLY after contract `v0.3.0` is tagged and pushed (image-gen-1-contract plan §Rollout). The replace-guard pre-commit hook auto-strips the local `replace` on commit anyway, but do it explicitly so `go mod tidy` resolves against the real tag.

- [ ] **Step 1: Drop the dev replace and tidy against the tag.**

```bash
cd /home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers
GOWORK=off go mod edit -dropreplace github.com/costa92/llm-agent-contract
GOWORK=off go mod tidy
```
Expected: `go.mod` keeps `require github.com/costa92/llm-agent-contract v0.3.0` with NO `replace`. If tidy reports the version is missing, contract v0.3.0 is not pushed yet — STOP.

- [ ] **Step 2: Final clean build + test of the whole repo.**

```bash
GOWORK=off go build ./...
GOWORK=off go test -race ./google/
```
Expected: build + tests green against the pinned contract tag.

- [ ] **Step 3: Commit.**

```bash
git add go.mod go.sum
git commit -m "build(google): pin contract v0.3.0, drop dev replace

Contract v0.3.0 is tagged with the ImageGenerator capability the google
provider consumes; replace the dev-time local replace with the real pinned
version so the module resolves standalone."
```

---

## Self-review

**1. Spec coverage (design spec §google + §Embeddings + §Custom headers + §Testing):**
- `genai.NewClient(ctx, &genai.ClientConfig{APIKey, Backend: BackendGeminiAPI})` — Task 1 `New`. ✅ Project/Location/Credentials NOT set (mutual-exclusion guard respected).
- `WithModel`(required) / `WithAPIKey`(GEMINI_API_KEY → GOOGLE_API_KEY fallback) / `WithHTTPClient`(→ClientConfig.HTTPClient) / `WithBaseURL`(→HTTPOptions.BaseURL) / `WithTimeout` / `WithTaskType` + `WithDimensions`(embed) / `WithExtraHeaders`(→HTTPOptions.Headers) — Task 1. ✅
- `capabilitiesForModel` (Tools for gemini-* chat; ImageGeneration for gemini-*-image + imagen-*; Embeddings for *embedding*) — Task 1. ✅
- Generate (GenerateContent; SystemInstruction not system role; `resp.Text()`; finish + usage nil-guarded) — Tasks 2/4. ✅ Info, WithTools — Task 2. Compile asserts staged across Tasks 2/3/5/6. ✅
- map.go: Request→`[]*genai.Content` (user/model) + GenerateContentConfig; tools→FunctionDeclaration via `ParametersJsonSchema` (unmarshal raw JSON-schema bytes → `map[string]any`); `FunctionCalls()`→`llm.ToolCall` (re-marshal Args → JSON) — Task 2. ✅
- Stream: `GenerateContentStream` `iter.Seq2` bridged via `iter.Pull2`; text deltas; tool calls complete-in-one-chunk → Start+ArgsDelta(full)+End together — Task 3. ✅
- image.go: imagen* → `GenerateImages` (`GeneratedImages[].Image.ImageBytes`→Bytes); else gemini-*-image → `GenerateContent` `ResponseModalities ["TEXT","IMAGE"]`, extract `InlineData{Data,MIMEType}`→Bytes (drop text) — Task 6. ✅
- embed.go: `EmbedContent` (texts→`[]*Content` one per text); `Embeddings[].Values`→Vector; usage zero/UsageUnknown; `EmbedDimensions` per model (gemini-embedding-001→3072, text-embedding-004→768) honoring WithDimensions/OutputDimensionality — Tasks 1/5. ✅
- errors.go: `wrapErr` mapping genai `APIError` (by value, `errors.As`; switch Code 401/403/429/5xx) → llm typed errors; nil-guard Candidates (PromptFeedback.BlockReason) — Tasks 2/4. ✅
- Tests point genai at `httptest.NewServer` via `HTTPOptions.BaseURL`; mock generateContent (text + inlineData image), stream chunks, function-call response, imagen predict, embedContent; real handlers + assertions — Tasks 2/3/4/5/6/7. ✅
- `WithExtraHeaders` forwarded — Task 7. ✅ goleak stream lifecycle — Task 3. ✅

**2. Placeholder scan:** every code step shows complete Go (no TODO/"similar to above"). The two "if it fails, inspect…" notes (Imagen predict wire shape; extra-header survival) are explicit verification flags with a concrete next action, not placeholders. ✅

**3. Type consistency:**
- `*Google` struct fields (`client`, `info`, `tools`, `taskType`, `dimensions`) referenced consistently across options.go/google.go/map.go/embed.go/image.go. ✅
- `g.info.Model` used as the model arg in every SDK call; `g.info.Capabilities.{Tools,ImageGeneration,Embeddings}` are the single gating source. ✅
- `genai.Ptr` used for `*float32` (Temperature) and `*int32` (OutputDimensionality); `int32(...)` direct for MaxOutputTokens/NumberOfImages (plain int32 fields). ✅ Matches SDK field kinds verified above.
- `genai.RoleUser`/`genai.RoleModel` (string consts) used for Content.Role. ✅
- `resp.Text()` / `resp.FunctionCalls()` are methods on `*GenerateContentResponse` (verified lines 3486/3537). ✅
- `GenerateImagesResponse.GeneratedImages` (verified, NOT `Images`); `GeneratedImage.Image.ImageBytes`/`.MIMEType` (verified). ✅
- `ContentEmbedding.Values` (verified). `EmbedContentResponse.Embeddings` (verified). ✅
- `genai.APIError` value type with `errors.As(err, &apiErr)` where `apiErr genai.APIError` — verified by-value returns. ✅
- `mapFinishReason` signature `func(genai.FinishReason) llm.FinishReason` consistent between map.go (response) and google.go (stream). ✅
- Stream reader fields/methods (`open`/`next`/`stop`/`queue`/`closed`/`lastFinish`/`lastUsage`) consistent; `iter.Pull2` returns `(next func() (V,K?,bool)...)` — for `Seq2[A,B]` the pulled `next` returns `(A, B, bool)` i.e. `(*GenerateContentResponse, error, bool)`. ✅

**4. SDK symbol accuracy:** All call signatures, field names, and constants are quoted with file+line from `genai@v1.59.0` in the "SDK symbol verification" section. The ONLY unconfirmed wire detail is the Imagen `:predict` response JSON the SDK's generated converter expects (flagged in Task 6 with a concrete fallback). Everything the provider CODE touches (struct fields, method signatures, error type, role consts, finish reasons) is confirmed from source.

**5. Conventions match:** five-file skeleton + image.go/embed.go (CONVENTIONS allows extra files when a path is large); `New(opts ...Option) (*Google, error)`; required-option error wording `"google: WithModel is required"`; `WithX` one-liners; API-key env fallback; `var _ llm.X = (*Google)(nil)` asserts; receiver `g *Google` / `r *googleStreamReader`; `wrapErr` typed-error layering; `compat.DefaultTimeout`; stdlib→third-party→llm-agent import grouping; no SDK types on the public surface (Task 7 §Step 4 enforces). ✅

**6. GOWORK=off:** present on every `go` invocation (go.work confirmed at umbrella root). ✅
