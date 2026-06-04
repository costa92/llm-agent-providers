# Image Generation + Minimax Embeddings + WithExtraHeaders (openai/minimax) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add text-to-image generation to the existing `openai` and `minimax` provider packages, add raw-HTTP embeddings to `minimax`, and add a `WithExtraHeaders` option to both — all model-gated and contract-conformant.

**Architecture:** `openai` uses the existing `openai-go/v3` SDK (`client.Images.Generate`) for image generation and `option.WithHeaderAdd` for extra headers. `minimax` chats via the anthropic SDK but has no SDK surface for its proprietary image (`/v1/image_generation`) and embeddings (`/v1/embeddings`) endpoints, so those go through raw `net/http` against a retained `baseURL`/`apiKey`/`httpClient` on the struct. Capabilities (`ImageGeneration`, `Embeddings`) are gated per bound model exactly like the existing `Embedder` gating. All new behavior is covered by `httptest.NewServer` mocks.

**Tech Stack:** Go 1.26, `github.com/costa92/llm-agent-contract/llm` (v0.3.0 — adds `ImageGenerator`/`ImageRequest`/`ImageResponse`/`GeneratedImage`/`Capabilities.ImageGeneration`), `github.com/openai/openai-go/v3`, `github.com/anthropics/anthropic-sdk-go`, `net/http`, `net/http/httptest`.

---

## Preconditions (read before starting)

1. **Contract v0.3.0 must be available.** This plan ASSUMES a separate plan ("plan 1") has already added `llm.ImageGenerator`, `llm.ImageRequest`, `llm.ImageResponse`, `llm.GeneratedImage`, and the `Capabilities.ImageGeneration bool` field to `github.com/costa92/llm-agent-contract`. During development the providers `go.mod` uses a local `replace` pointing at the contract working tree:

   ```
   replace github.com/costa92/llm-agent-contract => ../llm-agent-contract
   ```

   The replace-guard pre-commit hook auto-strips this `replace` on commit and re-pins the latest tag. If contract v0.3.0 is NOT yet tagged, use `git commit --no-verify` for the dev commits in this plan, OR ensure contract is tagged v0.3.0 first. Confirm the contract types compile-resolve before Task 1:

   ```bash
   GOWORK=off go doc github.com/costa92/llm-agent-contract/llm.ImageGenerator
   ```

   Expected: prints the `ImageGenerator` interface with `GenerateImage(ctx, ImageRequest) (ImageResponse, error)`.

2. **`GOWORK=off` is MANDATORY for every go command.** A `go.work` exists at the ecosystem root that excludes the standalone providers module. All `go test` / `go build` / `go vet` / `go doc` commands below are prefixed with `GOWORK=off`. Do not drop the prefix.

3. **Working directory.** All paths are absolute under the repo root
   `/home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-providers`.
   Run go commands from that directory (the module root).

## Contract type reference (from contract v0.3.0 — do NOT redefine these)

```go
// package llm

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

---

## File Structure

### openai package (`openai/`)

- **Create `openai/image.go`** — `GenerateImage` on `*OpenAI`, the `req → openai.ImageGenerateParams` mapping, the `ImagesResponse.Data[] → []llm.GeneratedImage` mapping, the `isImageModel` helper, and the `var _ llm.ImageGenerator = (*OpenAI)(nil)` compile assert.
- **Modify `openai/options.go`** — add `WithExtraHeaders`; thread `extraHeaders` through `config` and into `sdkOpts` via `option.WithHeaderAdd`; set `Capabilities.ImageGeneration = isImageModel(cfg.model)`.
- **Create `openai/image_test.go`** — `httptest` mock for `/images/generations` (b64 + url), capability gating, not-supported path, extra-headers assertion.

### minimax package (`minimax/`)

- **Modify `minimax/minimax.go`** — add `baseURL`, `apiKey`, `httpClient` fields to the `MiniMax` struct; add the `var _` image/embedder compile asserts.
- **Modify `minimax/options.go`** — populate the three new struct fields in `New`; add `WithExtraHeaders`, `WithGroupID`, `WithEmbeddingType`; thread `extraHeaders` into the anthropic SDK opts via `option.WithHeaderAdd`; retain `groupID`/`embeddingType` in config.
- **Modify `minimax/capabilities.go`** — gate `ImageGeneration` for `image-01` and `Embeddings` for `embo-01`.
- **Create `minimax/image.go`** — raw-HTTP `POST {baseURL}/v1/image_generation`, request/response structs, `base_resp` status check, mapping to `GeneratedImage.URL`.
- **Create `minimax/embed.go`** — raw-HTTP `POST {baseURL}/v1/embeddings` with `GroupId` query param, `Embed`/`EmbedDimensions`, top-level `vectors`/`total_tokens` parsing.
- **Create `minimax/httpclient.go`** — shared raw-HTTP helpers (`postJSON`, `applyExtraHeaders`, the `baseResp` type + status mapping) used by both `image.go` and `embed.go`.
- **Create `minimax/image_test.go`** — `httptest` mock for `/v1/image_generation`.
- **Create `minimax/embed_test.go`** — `httptest` mock for `/v1/embeddings` (asserts `GroupId` query param, `type` field, top-level `vectors`).

---

## Task 1: openai — `WithExtraHeaders` option

**Files:**
- Modify: `openai/options.go`
- Test: `openai/image_test.go` (create — holds new-feature tests for this plan)

- [ ] **Step 1: Write the failing test**

Create `openai/image_test.go`:

```go
package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestWithExtraHeaders_OpenAI_AppliedToRequests(t *testing.T) {
	var gotHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-My-Gateway")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"created":1710000000,
			"data":[{"b64_json":"aGVsbG8="}]
		}`))
	}))
	defer server.Close()

	o, err := New(
		WithModel("gpt-image-1"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
		WithExtraHeaders(map[string]string{"X-My-Gateway": "route-42"}),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := o.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "a cat"}); err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if gotHeader != "route-42" {
		t.Fatalf("X-My-Gateway = %q, want route-42", gotHeader)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `GOWORK=off go test ./openai/ -run TestWithExtraHeaders_OpenAI_AppliedToRequests -v`
Expected: FAIL — `undefined: WithExtraHeaders` and `o.GenerateImage undefined` (compile error). This task implements only `WithExtraHeaders`; `GenerateImage` arrives in Task 3. The compile failure confirms the test is wired.

- [ ] **Step 3: Write minimal implementation**

In `openai/options.go`, add the `extraHeaders` field to `config` (after `organization`):

```go
type config struct {
	apiKey       string
	model        string
	baseURL      string
	httpClient   *http.Client
	timeout      time.Duration
	organization string
	extraHeaders map[string]string
}
```

Add the option (after `WithOrganization`):

```go
// WithExtraHeaders injects additional headers into every outbound request
// (chat/stream/image/embed). Reserved headers (Authorization, Content-Type)
// are not overridable; extra headers are additive via option.WithHeaderAdd.
func WithExtraHeaders(h map[string]string) Option {
	return func(c *config) { c.extraHeaders = h }
}
```

In `New`, after the `cfg.organization` block and before the `cfg.timeout` block, add:

```go
	for k, v := range cfg.extraHeaders {
		sdkOpts = append(sdkOpts, option.WithHeaderAdd(k, v))
	}
```

- [ ] **Step 4: Run test to verify it still fails (only on GenerateImage)**

Run: `GOWORK=off go test ./openai/ -run TestWithExtraHeaders_OpenAI_AppliedToRequests -v`
Expected: FAIL — now only `o.GenerateImage undefined`. `WithExtraHeaders` resolves. (This test goes green at the end of Task 3.)

- [ ] **Step 5: Verify the package still builds for the parts that exist**

Run: `GOWORK=off go build ./openai/`
Expected: PASS (the option compiles; the test file is excluded from `go build`).

- [ ] **Step 6: Commit**

```bash
git add openai/options.go openai/image_test.go
git commit --no-verify -m "feat(openai): add WithExtraHeaders option for custom request headers"
```

---

## Task 2: openai — image-model capability gating

**Files:**
- Create: `openai/image.go` (the `isImageModel` helper only, in this task)
- Modify: `openai/options.go` (set `Capabilities.ImageGeneration`)
- Test: `openai/image_test.go`

- [ ] **Step 1: Write the failing test**

Append to `openai/image_test.go`:

```go
func TestInfo_OpenAI_ImageModel(t *testing.T) {
	o, err := New(WithModel("gpt-image-1"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	caps := o.Info().Capabilities
	if !caps.ImageGeneration {
		t.Fatalf("Capabilities = %+v, want ImageGeneration=true", caps)
	}
	if caps.Embeddings {
		t.Fatalf("image model must not report Embeddings: %+v", caps)
	}
}

func TestInfo_OpenAI_ChatModelNoImage(t *testing.T) {
	o, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if o.Info().Capabilities.ImageGeneration {
		t.Fatalf("gpt-4o-mini must not report ImageGeneration")
	}
}

func TestIsImageModel_OpenAI(t *testing.T) {
	imageModels := []string{"gpt-image-1", "gpt-image-2", "dall-e-2", "dall-e-3"}
	for _, m := range imageModels {
		if !isImageModel(m) {
			t.Errorf("isImageModel(%q) = false, want true", m)
		}
	}
	chatModels := []string{"gpt-4o-mini", "text-embedding-3-small", ""}
	for _, m := range chatModels {
		if isImageModel(m) {
			t.Errorf("isImageModel(%q) = true, want false", m)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `GOWORK=off go test ./openai/ -run 'TestInfo_OpenAI_ImageModel|TestIsImageModel_OpenAI' -v`
Expected: FAIL — `undefined: isImageModel` and `caps.ImageGeneration` set false.

- [ ] **Step 3: Write minimal implementation**

Create `openai/image.go` with the helper (the `GenerateImage` method is added in Task 3):

```go
package openai

import (
	"github.com/costa92/llm-agent-contract/llm"
)

var _ llm.ImageGenerator = (*OpenAI)(nil)

// isImageModel reports whether the bound model is an OpenAI image model.
// Mirrors the embedding-model gating in options.go (K2: capabilities are
// per provider×model). New image models must be added here explicitly.
func isImageModel(model string) bool {
	switch model {
	case "gpt-image-1", "gpt-image-1-mini", "gpt-image-2",
		"gpt-image-2-2026-04-21", "dall-e-2", "dall-e-3":
		return true
	default:
		return false
	}
}
```

In `openai/options.go`, set the capability flag. Replace the returned `Capabilities` block in `New`:

```go
		info: llm.ProviderInfo{
			Provider: "openai",
			Model:    cfg.model,
			Capabilities: llm.Capabilities{
				Tools:             true,
				Embeddings:        embeddings,
				StructuredOutputs: false,
				PromptCaching:     false,
				ImageGeneration:   isImageModel(cfg.model),
			},
		},
```

- [ ] **Step 4: Run test to verify it passes**

Run: `GOWORK=off go test ./openai/ -run 'TestInfo_OpenAI_ImageModel|TestInfo_OpenAI_ChatModelNoImage|TestIsImageModel_OpenAI' -v`
Expected: PASS. (`var _ llm.ImageGenerator = (*OpenAI)(nil)` will NOT compile yet because `GenerateImage` is missing — so this step actually fails to compile. To avoid a non-compiling intermediate, the compile assert is added in Task 3 instead.)

> **Correction — do this in Step 3:** OMIT the `var _ llm.ImageGenerator = (*OpenAI)(nil)` line from `image.go` in THIS task. Add only the `isImageModel` helper and the package/import. The compile assert is introduced in Task 3 Step 3 together with the `GenerateImage` method, so the package always compiles. Re-run:

Run: `GOWORK=off go test ./openai/ -run 'TestInfo_OpenAI_ImageModel|TestInfo_OpenAI_ChatModelNoImage|TestIsImageModel_OpenAI' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add openai/image.go openai/options.go openai/image_test.go
git commit --no-verify -m "feat(openai): gate ImageGeneration capability by bound model"
```

---

## Task 3: openai — `GenerateImage` implementation

**Files:**
- Modify: `openai/image.go`
- Test: `openai/image_test.go`

- [ ] **Step 1: Write the failing test**

Append to `openai/image_test.go`:

```go
func TestGenerateImage_OpenAI_B64(t *testing.T) {
	var gotPath, gotBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		// "aGVsbG8=" is base64("hello").
		_, _ = w.Write([]byte(`{
			"created":1710000000,
			"data":[{"b64_json":"aGVsbG8=","revised_prompt":"a fluffy cat"}],
			"usage":{"input_tokens":5,"output_tokens":10,"total_tokens":15}
		}`))
	}))
	defer server.Close()

	o, err := New(WithModel("gpt-image-1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := o.GenerateImage(context.Background(), llm.ImageRequest{
		Prompt:  "a cat",
		N:       1,
		Size:    "1024x1024",
		Quality: "high",
		Format:  "png",
	})
	if err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if gotPath != "/images/generations" {
		t.Fatalf("path = %q, want /images/generations", gotPath)
	}
	if !strings.Contains(gotBody, `"prompt":"a cat"`) {
		t.Fatalf("body missing prompt: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"model":"gpt-image-1"`) {
		t.Fatalf("body missing model: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"size":"1024x1024"`) {
		t.Fatalf("body missing size: %s", gotBody)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("len(Images) = %d, want 1", len(resp.Images))
	}
	if string(resp.Images[0].Bytes) != "hello" {
		t.Fatalf("Bytes = %q, want decoded \"hello\"", resp.Images[0].Bytes)
	}
	if resp.Images[0].URL != "" {
		t.Fatalf("URL = %q, want empty when bytes present", resp.Images[0].URL)
	}
	if resp.Images[0].RevisedPrompt != "a fluffy cat" {
		t.Fatalf("RevisedPrompt = %q, want a fluffy cat", resp.Images[0].RevisedPrompt)
	}
	if resp.Provider != "openai" || resp.Model != "gpt-image-1" {
		t.Fatalf("provider/model = %q/%q", resp.Provider, resp.Model)
	}
}

func TestGenerateImage_OpenAI_URL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"created":1710000000,
			"data":[{"url":"https://img.example/abc.png"}]
		}`))
	}))
	defer server.Close()

	o, err := New(WithModel("dall-e-3"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := o.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "a dog"})
	if err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("len(Images) = %d, want 1", len(resp.Images))
	}
	if resp.Images[0].URL != "https://img.example/abc.png" {
		t.Fatalf("URL = %q", resp.Images[0].URL)
	}
	if len(resp.Images[0].Bytes) != 0 {
		t.Fatalf("Bytes = %v, want empty when only URL present", resp.Images[0].Bytes)
	}
}

func TestGenerateImage_OpenAI_NotSupported(t *testing.T) {
	o, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = o.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("err = %v, want ErrCapabilityNotSupported", err)
	}
}
```

Update the import block at the top of `openai/image_test.go` to include `encoding/base64` is NOT needed (the mock ships pre-encoded b64); ensure these imports are present: `context`, `errors`, `io`, `net/http`, `net/http/httptest`, `strings`, `testing`, and `github.com/costa92/llm-agent-contract/llm`. Final import block:

```go
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `GOWORK=off go test ./openai/ -run TestGenerateImage_OpenAI -v`
Expected: FAIL — `o.GenerateImage undefined`.

- [ ] **Step 3: Write minimal implementation**

Replace the entire contents of `openai/image.go` with:

```go
package openai

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/costa92/llm-agent-contract/llm"
	openai "github.com/openai/openai-go/v3"
)

var _ llm.ImageGenerator = (*OpenAI)(nil)

// isImageModel reports whether the bound model is an OpenAI image model.
// Mirrors the embedding-model gating in options.go (K2: capabilities are
// per provider×model). New image models must be added here explicitly.
func isImageModel(model string) bool {
	switch model {
	case "gpt-image-1", "gpt-image-1-mini", "gpt-image-2",
		"gpt-image-2-2026-04-21", "dall-e-2", "dall-e-3":
		return true
	default:
		return false
	}
}

// GenerateImage performs text-to-image generation via the OpenAI Images API.
// Gated by the bound model: non-image models return
// llm.ErrCapabilityNotSupported (mirrors Embed gating). When the provider
// returns base64 (GPT image models, or dall-e with b64_json), Bytes is
// populated; when it returns a hosted URL (dall-e default), URL is populated.
func (o *OpenAI) GenerateImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	if !isImageModel(o.info.Model) {
		return llm.ImageResponse{}, fmt.Errorf("openai: image generation: %w", llm.ErrCapabilityNotSupported)
	}

	resp, err := o.client.Images.Generate(ctx, o.toImageParams(req))
	if err != nil {
		return llm.ImageResponse{}, wrapErr(err)
	}
	return o.fromImageResponse(resp), nil
}

func (o *OpenAI) toImageParams(req llm.ImageRequest) openai.ImageGenerateParams {
	p := openai.ImageGenerateParams{
		Prompt: req.Prompt,
		Model:  openai.ImageModel(o.info.Model),
	}
	if req.N > 0 {
		p.N = openai.Int(int64(req.N))
	}
	if req.Size != "" {
		p.Size = openai.ImageGenerateParamsSize(req.Size)
	}
	if req.Quality != "" {
		p.Quality = openai.ImageGenerateParamsQuality(req.Quality)
	}
	if req.Format != "" {
		p.OutputFormat = openai.ImageGenerateParamsOutputFormat(req.Format)
	}
	if v, ok := req.Extra["style"].(string); ok && v != "" {
		p.Style = openai.ImageGenerateParamsStyle(v)
	}
	if v, ok := req.Extra["background"].(string); ok && v != "" {
		p.Background = openai.ImageGenerateParamsBackground(v)
	}
	if v, ok := req.Extra["moderation"].(string); ok && v != "" {
		p.Moderation = openai.ImageGenerateParamsModeration(v)
	}
	return p
}

func (o *OpenAI) fromImageResponse(resp *openai.ImagesResponse) llm.ImageResponse {
	images := make([]llm.GeneratedImage, 0, len(resp.Data))
	for _, img := range resp.Data {
		gen := llm.GeneratedImage{RevisedPrompt: img.RevisedPrompt}
		if img.B64JSON != "" {
			if decoded, err := base64.StdEncoding.DecodeString(img.B64JSON); err == nil {
				gen.Bytes = decoded
			}
		}
		if len(gen.Bytes) == 0 && img.URL != "" {
			gen.URL = img.URL
		}
		images = append(images, gen)
	}
	out := llm.ImageResponse{
		Images:   images,
		Provider: "openai",
		Model:    o.info.Model,
	}
	if resp.Usage.TotalTokens != 0 || resp.Usage.InputTokens != 0 {
		out.Usage = llm.Usage{
			InputTokens:  int(resp.Usage.InputTokens),
			OutputTokens: int(resp.Usage.OutputTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
			Source:       llm.UsageReported,
		}
	}
	return out
}
```

> Note: `image_test.go` already defines `isImageModel` tests against this same helper — `isImageModel` lives ONLY in `image.go` (created here). If Task 2 added a temporary copy, this Step replaces the whole file, so there is exactly one definition.

- [ ] **Step 4: Run test to verify it passes**

Run: `GOWORK=off go test ./openai/ -run 'TestGenerateImage_OpenAI|TestWithExtraHeaders_OpenAI|TestInfo_OpenAI_ImageModel|TestIsImageModel_OpenAI' -v`
Expected: PASS (all image + extra-header + gating tests now green).

- [ ] **Step 5: Run the whole openai package**

Run: `GOWORK=off go test ./openai/ -v`
Expected: PASS (existing chat/stream/embed tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add openai/image.go openai/image_test.go
git commit --no-verify -m "feat(openai): add GenerateImage via Images.Generate with b64/url mapping"
```

---

## Task 4: minimax — retain baseURL/apiKey/httpClient + WithExtraHeaders/WithGroupID/WithEmbeddingType options

**Files:**
- Modify: `minimax/minimax.go` (struct fields + compile asserts)
- Modify: `minimax/options.go` (new options + populate fields)
- Test: `minimax/options_internal_test.go` (extend) — verifies config plumbing

- [ ] **Step 1: Write the failing test**

Append to `minimax/options_internal_test.go`:

```go
// raw-HTTP accessors for testing config plumbing.
func (m *MiniMax) baseURLForTest() string         { return m.baseURL }
func (m *MiniMax) apiKeyForTest() string           { return m.apiKey }
func (m *MiniMax) extraHeadersForTest() map[string]string { return m.extraHeaders }
func (m *MiniMax) groupIDForTest() string          { return m.groupID }
func (m *MiniMax) embeddingTypeForTest() string    { return m.embeddingType }

func TestNew_RetainsRawHTTPConfig(t *testing.T) {
	m, err := New(
		WithModel("image-01"),
		WithAPIKey("k"),
		WithBaseURL("https://example.test"),
		WithExtraHeaders(map[string]string{"X-Trace": "1"}),
		WithGroupID("grp-9"),
		WithEmbeddingType("query"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if m.baseURLForTest() != "https://example.test" {
		t.Fatalf("baseURL = %q", m.baseURLForTest())
	}
	if m.apiKeyForTest() != "k" {
		t.Fatalf("apiKey = %q", m.apiKeyForTest())
	}
	if m.extraHeadersForTest()["X-Trace"] != "1" {
		t.Fatalf("extraHeaders = %v", m.extraHeadersForTest())
	}
	if m.groupIDForTest() != "grp-9" {
		t.Fatalf("groupID = %q", m.groupIDForTest())
	}
	if m.embeddingTypeForTest() != "query" {
		t.Fatalf("embeddingType = %q", m.embeddingTypeForTest())
	}
}

func TestNew_EmbeddingTypeDefaultsDB(t *testing.T) {
	m, err := New(WithModel("embo-01"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := m.embeddingTypeForTest(); got != "db" {
		t.Fatalf("default embeddingType = %q, want db", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `GOWORK=off go test ./minimax/ -run 'TestNew_RetainsRawHTTPConfig|TestNew_EmbeddingTypeDefaultsDB' -v`
Expected: FAIL — `m.baseURL undefined`, `WithExtraHeaders undefined`, `WithGroupID undefined`, `WithEmbeddingType undefined`.

- [ ] **Step 3: Write minimal implementation**

In `minimax/minimax.go`, extend the `MiniMax` struct and add compile asserts:

```go
var (
	_ llm.ChatModel      = (*MiniMax)(nil)
	_ llm.ToolCaller     = (*MiniMax)(nil)
	_ llm.ImageGenerator = (*MiniMax)(nil)
	_ llm.Embedder       = (*MiniMax)(nil)
)

type MiniMax struct {
	client  *sdk.Client
	info    llm.ProviderInfo
	tools   []llm.Tool
	timeout time.Duration

	// Raw-HTTP path config (image generation + embeddings have no SDK surface).
	baseURL       string
	apiKey        string
	httpClient    *http.Client
	extraHeaders  map[string]string
	groupID       string
	embeddingType string
}
```

Add `"net/http"` to the `minimax/minimax.go` import block:

```go
import (
	"context"
	"io"
	"net/http"
	"sync"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/costa92/llm-agent-contract/llm"
)
```

> The compile asserts reference `GenerateImage`, `Embed`, `EmbedDimensions` which do not exist yet. To keep the package compiling, ADD the asserts in Task 6 (image) / Task 7 (embed) instead. In THIS task, keep `minimax.go`'s `var (...)` block unchanged (only `ChatModel` + `ToolCaller`); add only the struct fields and the `net/http` import.

Corrected `minimax.go` var block for this task (leave as-is):

```go
var (
	_ llm.ChatModel  = (*MiniMax)(nil)
	_ llm.ToolCaller = (*MiniMax)(nil)
)
```

In `minimax/options.go`, add fields to `config` (after `region`):

```go
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
```

Add the new options (after `WithRegion`):

```go
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
```

In `New`, after the env fallback for `cfg.apiKey`, add the group-id env fallback and embedding-type default:

```go
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("MINIMAX_API_KEY")
	}
	if cfg.groupID == "" {
		cfg.groupID = os.Getenv("MINIMAX_GROUP_ID")
	}
	if cfg.embeddingType == "" {
		cfg.embeddingType = "db"
	}
```

Add the extra-headers SDK plumbing. In `New`, after the `cfg.httpClient` SDK-opt block and before the `cfg.timeout` block, add:

```go
	for k, v := range cfg.extraHeaders {
		sdkOpts = append(sdkOpts, option.WithHeaderAdd(k, v))
	}
```

Finally, populate the new struct fields in the returned `&MiniMax{...}`:

```go
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `GOWORK=off go test ./minimax/ -run 'TestNew_RetainsRawHTTPConfig|TestNew_EmbeddingTypeDefaultsDB' -v`
Expected: PASS.

- [ ] **Step 5: Run the whole minimax package**

Run: `GOWORK=off go test ./minimax/ -v`
Expected: PASS (existing chat/stream/tool tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add minimax/minimax.go minimax/options.go minimax/options_internal_test.go
git commit --no-verify -m "feat(minimax): retain raw-HTTP config; add WithExtraHeaders/WithGroupID/WithEmbeddingType"
```

---

## Task 5: minimax — shared raw-HTTP helper

**Files:**
- Create: `minimax/httpclient.go`
- Test: covered indirectly by Task 6/7; add a focused unit test for `baseResp` mapping here.

- [ ] **Step 1: Write the failing test**

Create `minimax/httpclient_internal_test.go`:

```go
package minimax

import (
	"errors"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestBaseRespError_Mapping(t *testing.T) {
	tests := []struct {
		name   string
		status int
		isAuth bool
	}{
		{"ok_zero", 0, false},
		{"auth_1004", 1004, true},
		{"generic_1008", 1008, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := baseRespError("minimax: image", baseResp{StatusCode: tt.status, StatusMsg: "boom"})
			if tt.status == 0 {
				if err != nil {
					t.Fatalf("status 0 must be nil, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("status %d must produce error", tt.status)
			}
			if tt.isAuth && !errors.As(err, new(*llm.AuthError)) {
				t.Fatalf("status %d: want AuthError, got %v", tt.status, err)
			}
		})
	}
}
```

> `llm.AuthError` is a struct (`Provider string`, `Wrapped error`), so `errors.As(err, new(*llm.AuthError))` is the correct assertion — confirmed against `llm/errors.go` and `internal/compat/errors_anthropic.go`.

- [ ] **Step 2: Run test to verify it fails**

Run: `GOWORK=off go test ./minimax/ -run TestBaseRespError_Mapping -v`
Expected: FAIL — `undefined: baseRespError`, `undefined: baseResp`.

- [ ] **Step 3: Write minimal implementation**

Create `minimax/httpclient.go`:

```go
package minimax

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/costa92/llm-agent-contract/llm"
)

// baseResp is MiniMax's envelope status block. MiniMax returns HTTP 200 even
// on logical failure, so StatusCode != 0 is the real error signal.
type baseResp struct {
	StatusCode int    `json:"status_code"`
	StatusMsg  string `json:"status_msg"`
}

// baseRespError maps a non-zero MiniMax status_code to a typed llm error.
// Returns nil when StatusCode == 0. The contract error structs carry a
// Provider string and a Wrapped error (NOT a Message field) — see
// llm/errors.go — so the textual detail goes into a wrapped error.
func baseRespError(prefix string, br baseResp) error {
	if br.StatusCode == 0 {
		return nil
	}
	wrapped := fmt.Errorf("%s: minimax status_code=%d: %s", prefix, br.StatusCode, br.StatusMsg)
	switch br.StatusCode {
	case 1004: // auth / invalid api key
		return &llm.AuthError{Provider: "minimax", Wrapped: wrapped}
	case 1002, 1039: // rate limit / RPM-TPM limit
		return &llm.RateLimitError{Provider: "minimax", Wrapped: wrapped}
	case 1027, 1013: // service unavailable / internal
		return &llm.TransientError{Provider: "minimax", Wrapped: wrapped}
	default:
		return &llm.InvalidRequestError{Provider: "minimax", Wrapped: wrapped}
	}
}

// postJSON issues a POST {baseURL}{path} with a JSON body, applies the Bearer
// token and extra headers, decodes the JSON response into out, and returns the
// raw HTTP status. The caller checks base_resp separately.
func (m *MiniMax) postJSON(ctx context.Context, path, rawQuery string, body any, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("minimax: marshal request: %w", err)
	}
	u := m.baseURL + path
	if rawQuery != "" {
		u += "?" + rawQuery
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("minimax: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if m.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)
	}
	m.applyExtraHeaders(httpReq)

	client := m.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		// context cancellation is surfaced as-is by net/http via the wrapped err.
		return fmt.Errorf("minimax: request: %w", err)
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("minimax: read response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return &llm.TransientError{
			Provider: "minimax",
			Wrapped:  fmt.Errorf("minimax: http %d: %s", resp.StatusCode, string(respBytes)),
		}
	}
	if err := json.Unmarshal(respBytes, out); err != nil {
		return fmt.Errorf("minimax: decode response: %w", err)
	}
	return nil
}

// applyExtraHeaders adds caller-supplied headers without overriding the
// reserved Authorization / Content-Type headers already set.
func (m *MiniMax) applyExtraHeaders(req *http.Request) {
	for k, v := range m.extraHeaders {
		if http.CanonicalHeaderKey(k) == "Authorization" || http.CanonicalHeaderKey(k) == "Content-Type" {
			continue
		}
		req.Header.Set(k, v)
	}
}
```

> **Error type shape (confirmed):** the contract exposes `llm.AuthError`, `llm.RateLimitError`, `llm.TransientError`, `llm.InvalidRequestError` as structs with a `Provider string` and a `Wrapped error` field (see `llm/errors.go`; `internal/compat/errors_anthropic.go` constructs them the same way, e.g. `&llm.AuthError{Provider: provider, Wrapped: err}`). There is NO `Message` field — textual detail goes into `Wrapped` via `fmt.Errorf`, exactly as written above. The MiniMax `status_code` → class mapping is best-effort; the only behavior the test pins is "0 → nil, 1004 → auth, other → non-nil". Keep that contract.

- [ ] **Step 4: Run test to verify it passes**

Run: `GOWORK=off go test ./minimax/ -run TestBaseRespError_Mapping -v`
Expected: PASS.

- [ ] **Step 5: Verify the package builds**

Run: `GOWORK=off go build ./minimax/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add minimax/httpclient.go minimax/httpclient_internal_test.go
git commit --no-verify -m "feat(minimax): add raw-HTTP helper with base_resp status mapping"
```

---

## Task 6: minimax — `GenerateImage` (raw HTTP)

**Files:**
- Create: `minimax/image.go`
- Modify: `minimax/minimax.go` (add `ImageGenerator` compile assert)
- Modify: `minimax/capabilities.go` (gate `image-01`)
- Test: `minimax/image_test.go`

- [ ] **Step 1: Write the failing test**

Create `minimax/image_test.go`:

```go
package minimax

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

func TestGenerateImage_MiniMax_Happy(t *testing.T) {
	var gotPath, gotBody, gotAuth, gotExtra string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotExtra = r.Header.Get("X-Trace")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"data":{"image_urls":["https://cdn.minimax/img1.png","https://cdn.minimax/img2.png"]},
			"base_resp":{"status_code":0,"status_msg":"success"}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("image-01"),
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithExtraHeaders(map[string]string{"X-Trace": "abc"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := m.GenerateImage(context.Background(), llm.ImageRequest{
		Prompt: "a robot",
		N:      2,
		Size:   "1024x768",
	})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if gotPath != "/v1/image_generation" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotAuth != "Bearer k" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if gotExtra != "abc" {
		t.Fatalf("X-Trace = %q", gotExtra)
	}
	if !strings.Contains(gotBody, `"model":"image-01"`) {
		t.Fatalf("body missing model: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"prompt":"a robot"`) {
		t.Fatalf("body missing prompt: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"response_format":"url"`) {
		t.Fatalf("body missing response_format: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"width":1024`) || !strings.Contains(gotBody, `"height":768`) {
		t.Fatalf("body missing width/height from Size: %s", gotBody)
	}
	if len(resp.Images) != 2 {
		t.Fatalf("len(Images) = %d, want 2", len(resp.Images))
	}
	if resp.Images[0].URL != "https://cdn.minimax/img1.png" {
		t.Fatalf("Images[0].URL = %q", resp.Images[0].URL)
	}
	if resp.Provider != "minimax" || resp.Model != "image-01" {
		t.Fatalf("provider/model = %q/%q", resp.Provider, resp.Model)
	}
}

func TestGenerateImage_MiniMax_LogicalFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// HTTP 200 but logical failure — the MiniMax gotcha.
		_, _ = w.Write([]byte(`{"base_resp":{"status_code":1004,"status_msg":"invalid api key"}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("image-01"), WithAPIKey("bad"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = m.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if err == nil {
		t.Fatal("want error on status_code=1004")
	}
	if !errors.As(err, new(*llm.AuthError)) {
		t.Fatalf("err = %v, want AuthError", err)
	}
}

func TestGenerateImage_MiniMax_NotSupported(t *testing.T) {
	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = m.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("err = %v, want ErrCapabilityNotSupported", err)
	}
}

func TestInfo_MiniMax_ImageModel(t *testing.T) {
	m, err := New(WithModel("image-01"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !m.Info().Capabilities.ImageGeneration {
		t.Fatalf("image-01 must report ImageGeneration: %+v", m.Info().Capabilities)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `GOWORK=off go test ./minimax/ -run 'TestGenerateImage_MiniMax|TestInfo_MiniMax_ImageModel' -v`
Expected: FAIL — `m.GenerateImage undefined` and `image-01` not gated.

- [ ] **Step 3: Write minimal implementation**

Create `minimax/image.go`:

```go
package minimax

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/costa92/llm-agent-contract/llm"
)

// imageRequestBody is the MiniMax /v1/image_generation request payload.
type imageRequestBody struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	ResponseFormat string `json:"response_format"`
	AspectRatio    string `json:"aspect_ratio,omitempty"`
	Width          int    `json:"width,omitempty"`
	Height         int    `json:"height,omitempty"`
}

type imageResponseBody struct {
	Data struct {
		ImageURLs []string `json:"image_urls"`
	} `json:"data"`
	BaseResp baseResp `json:"base_resp"`
}

// GenerateImage performs text-to-image generation via the proprietary MiniMax
// /v1/image_generation endpoint (no SDK surface — raw HTTP). Gated by the bound
// model: non-image models return llm.ErrCapabilityNotSupported. MiniMax returns
// hosted URLs, so each GeneratedImage carries URL (not Bytes).
func (m *MiniMax) GenerateImage(ctx context.Context, req llm.ImageRequest) (llm.ImageResponse, error) {
	if !m.info.Capabilities.ImageGeneration {
		return llm.ImageResponse{}, fmt.Errorf("minimax: image generation: %w", llm.ErrCapabilityNotSupported)
	}

	body := imageRequestBody{
		Model:          m.info.Model,
		Prompt:         req.Prompt,
		N:              req.N,
		ResponseFormat: "url",
	}
	if w, h, ok := parseSize(req.Size); ok {
		body.Width = w
		body.Height = h
	} else if ar, ok := req.Extra["aspect_ratio"].(string); ok && ar != "" {
		body.AspectRatio = ar
	}

	var out imageResponseBody
	if err := m.postJSON(ctx, "/v1/image_generation", "", body, &out); err != nil {
		return llm.ImageResponse{}, err
	}
	if err := baseRespError("minimax: image generation", out.BaseResp); err != nil {
		return llm.ImageResponse{}, err
	}

	images := make([]llm.GeneratedImage, 0, len(out.Data.ImageURLs))
	for _, u := range out.Data.ImageURLs {
		images = append(images, llm.GeneratedImage{URL: u})
	}
	return llm.ImageResponse{
		Images:   images,
		Provider: "minimax",
		Model:    m.info.Model,
	}, nil
}

// parseSize splits a "WxH" string into width/height ints. Returns ok=false for
// any string that is not exactly two positive integers separated by "x".
func parseSize(size string) (w, h int, ok bool) {
	parts := strings.SplitN(size, "x", 2)
	if len(parts) != 2 {
		return 0, 0, false
	}
	w, err := strconv.Atoi(parts[0])
	if err != nil || w <= 0 {
		return 0, 0, false
	}
	h, err = strconv.Atoi(parts[1])
	if err != nil || h <= 0 {
		return 0, 0, false
	}
	return w, h, true
}
```

In `minimax/minimax.go`, add the `ImageGenerator` compile assert (keep `Embedder` for Task 7):

```go
var (
	_ llm.ChatModel      = (*MiniMax)(nil)
	_ llm.ToolCaller     = (*MiniMax)(nil)
	_ llm.ImageGenerator = (*MiniMax)(nil)
)
```

In `minimax/capabilities.go`, add an `image-01` case:

```go
func capabilitiesForModel(model string) llm.Capabilities {
	switch model {
	case "MiniMax-M1":
		return llm.Capabilities{Tools: true}
	case "image-01":
		return llm.Capabilities{ImageGeneration: true}
	default:
		return llm.Capabilities{Tools: true} // fallback: assume tools (behavior-preserving with prior hardcoded default)
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `GOWORK=off go test ./minimax/ -run 'TestGenerateImage_MiniMax|TestInfo_MiniMax_ImageModel' -v`
Expected: PASS.

- [ ] **Step 5: Confirm existing capability test still holds**

Run: `GOWORK=off go test ./minimax/ -run TestCapabilitiesForModel_MiniMax -v`
Expected: PASS (the `image-01` case is new; existing cases — M1, abab fallback, vision, empty — still resolve to tools-only or pass through unchanged).

- [ ] **Step 6: Commit**

```bash
git add minimax/image.go minimax/minimax.go minimax/capabilities.go minimax/image_test.go
git commit --no-verify -m "feat(minimax): add GenerateImage via raw-HTTP image_generation endpoint"
```

---

## Task 7: minimax — `Embed` + `EmbedDimensions` (raw HTTP)

**Files:**
- Create: `minimax/embed.go`
- Modify: `minimax/minimax.go` (add `Embedder` compile assert)
- Modify: `minimax/capabilities.go` (gate `embo-01`)
- Test: `minimax/embed_test.go`

- [ ] **Step 1: Write the failing test**

Create `minimax/embed_test.go`:

```go
package minimax

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

func TestEmbed_MiniMax_Happy(t *testing.T) {
	var gotPath, gotBody, gotGroupID, gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotGroupID = r.URL.Query().Get("GroupId")
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		// Top-level vectors + total_tokens (NOT nested under data/usage).
		_, _ = w.Write([]byte(`{
			"vectors":[[0.1,0.2,0.3],[0.4,0.5,0.6]],
			"total_tokens":12,
			"base_resp":{"status_code":0,"status_msg":"success"}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("embo-01"),
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithGroupID("grp-7"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if gotPath != "/v1/embeddings" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotGroupID != "grp-7" {
		t.Fatalf("GroupId query = %q, want grp-7", gotGroupID)
	}
	if gotAuth != "Bearer k" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if !strings.Contains(gotBody, `"model":"embo-01"`) {
		t.Fatalf("body missing model: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"texts":["hello","world"]`) {
		t.Fatalf("body missing texts in order: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"type":"db"`) {
		t.Fatalf("body missing type=db: %s", gotBody)
	}
	if len(vectors) != 2 {
		t.Fatalf("len(vectors) = %d, want 2", len(vectors))
	}
	if len(vectors[0]) != 3 || vectors[1][2] != 0.6 {
		t.Fatalf("vector content wrong: %v", vectors)
	}
	if usage.TotalTokens != 12 || usage.Source != llm.UsageReported {
		t.Fatalf("usage = %+v, want total=12 reported", usage)
	}
}

func TestEmbed_MiniMax_QueryType(t *testing.T) {
	var gotBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"vectors":[[0.1]],"total_tokens":1,"base_resp":{"status_code":0}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("embo-01"), WithAPIKey("k"), WithBaseURL(server.URL),
		WithGroupID("g"), WithEmbeddingType("query"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, _, err := m.Embed(context.Background(), []string{"q"}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if !strings.Contains(gotBody, `"type":"query"`) {
		t.Fatalf("body missing type=query: %s", gotBody)
	}
}

func TestEmbed_MiniMax_Empty(t *testing.T) {
	m, err := New(WithModel("embo-01"), WithAPIKey("k"), WithGroupID("g"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(empty): %v", err)
	}
	if len(vectors) != 0 {
		t.Fatalf("len(vectors) = %d, want 0", len(vectors))
	}
	if usage.Source != llm.UsageReported {
		t.Fatalf("usage.Source = %q, want reported", usage.Source)
	}
}

func TestEmbed_MiniMax_NotSupported(t *testing.T) {
	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _, err = m.Embed(context.Background(), []string{"x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("err = %v, want ErrCapabilityNotSupported", err)
	}
	if got := m.EmbedDimensions(); got != 0 {
		t.Fatalf("EmbedDimensions(non-embed) = %d, want 0", got)
	}
}

func TestEmbedDimensions_MiniMax(t *testing.T) {
	m, err := New(WithModel("embo-01"), WithAPIKey("k"), WithGroupID("g"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := m.EmbedDimensions(); got != 1536 {
		t.Fatalf("EmbedDimensions() = %d, want 1536", got)
	}
	if !m.Info().Capabilities.Embeddings {
		t.Fatalf("embo-01 must report Embeddings")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `GOWORK=off go test ./minimax/ -run 'TestEmbed_MiniMax|TestEmbedDimensions_MiniMax' -v`
Expected: FAIL — `m.Embed undefined`, `m.EmbedDimensions undefined`, `embo-01` not gated.

- [ ] **Step 3: Write minimal implementation**

Create `minimax/embed.go`:

```go
package minimax

import (
	"context"
	"fmt"
	"net/url"

	"github.com/costa92/llm-agent-contract/llm"
)

type embedRequestBody struct {
	Model string   `json:"model"`
	Texts []string `json:"texts"`
	Type  string   `json:"type"`
}

// embedResponseBody parses MiniMax's embeddings response. vectors and
// total_tokens are TOP-LEVEL fields (not nested under data/usage).
type embedResponseBody struct {
	Vectors     [][]float32 `json:"vectors"`
	TotalTokens int         `json:"total_tokens"`
	BaseResp    baseResp    `json:"base_resp"`
}

// Embed returns vectors in input order via the raw-HTTP /v1/embeddings
// endpoint. GroupId is passed as a query parameter; the "type" field is the
// configured embedding type ("db" default / "query"). Gated by the bound
// model: non-embed models return llm.ErrCapabilityNotSupported.
func (m *MiniMax) Embed(ctx context.Context, texts []string) ([]llm.Vector, llm.Usage, error) {
	if !m.info.Capabilities.Embeddings {
		return nil, llm.Usage{}, fmt.Errorf("minimax: embeddings: %w", llm.ErrCapabilityNotSupported)
	}
	if len(texts) == 0 {
		return []llm.Vector{}, llm.Usage{Source: llm.UsageReported}, nil
	}

	body := embedRequestBody{
		Model: m.info.Model,
		Texts: append([]string(nil), texts...),
		Type:  m.embeddingType,
	}
	rawQuery := url.Values{"GroupId": {m.groupID}}.Encode()

	var out embedResponseBody
	if err := m.postJSON(ctx, "/v1/embeddings", rawQuery, body, &out); err != nil {
		return nil, llm.Usage{}, err
	}
	if err := baseRespError("minimax: embeddings", out.BaseResp); err != nil {
		return nil, llm.Usage{}, err
	}

	vectors := make([]llm.Vector, 0, len(out.Vectors))
	for _, v := range out.Vectors {
		vectors = append(vectors, append(llm.Vector(nil), v...))
	}
	usage := llm.Usage{
		InputTokens: out.TotalTokens,
		TotalTokens: out.TotalTokens,
		Source:      llm.UsageReported,
	}
	return vectors, usage, nil
}

// EmbedDimensions returns the fixed embedding dimensionality for the bound
// model, or 0 when the model has no embedding capability.
func (m *MiniMax) EmbedDimensions() int {
	switch m.info.Model {
	case "embo-01":
		return 1536
	default:
		return 0
	}
}
```

In `minimax/minimax.go`, add the `Embedder` compile assert (now the full set):

```go
var (
	_ llm.ChatModel      = (*MiniMax)(nil)
	_ llm.ToolCaller     = (*MiniMax)(nil)
	_ llm.ImageGenerator = (*MiniMax)(nil)
	_ llm.Embedder       = (*MiniMax)(nil)
)
```

In `minimax/capabilities.go`, add the `embo-01` case:

```go
func capabilitiesForModel(model string) llm.Capabilities {
	switch model {
	case "MiniMax-M1":
		return llm.Capabilities{Tools: true}
	case "image-01":
		return llm.Capabilities{ImageGeneration: true}
	case "embo-01":
		return llm.Capabilities{Embeddings: true}
	default:
		return llm.Capabilities{Tools: true} // fallback: assume tools (behavior-preserving with prior hardcoded default)
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `GOWORK=off go test ./minimax/ -run 'TestEmbed_MiniMax|TestEmbedDimensions_MiniMax' -v`
Expected: PASS.

- [ ] **Step 5: Run the whole minimax package**

Run: `GOWORK=off go test ./minimax/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add minimax/embed.go minimax/minimax.go minimax/capabilities.go minimax/embed_test.go
git commit --no-verify -m "feat(minimax): add Embed/EmbedDimensions via raw-HTTP embeddings endpoint"
```

---

## Task 8: full build + vet + module-wide test

**Files:** none (verification only)

- [ ] **Step 1: Vet both packages**

Run: `GOWORK=off go vet ./openai/ ./minimax/`
Expected: no output (clean).

- [ ] **Step 2: Run the full module test suite**

Run: `GOWORK=off go test ./...`
Expected: PASS for `./openai/`, `./minimax/`, and unchanged for the rest. (The `internal/contract/` conformance suite does not yet exercise `ImageGenerator`; that is out of scope for this plan — no new fixtures required here.)

- [ ] **Step 3: Update the doc comments**

In `openai/doc.go`, update the capability sentence to mention image generation:

```go
// Package openai implements an OpenAI adapter over
// github.com/openai/openai-go/v3.
//
// The adapter satisfies llm.ChatModel, llm.ToolCaller, llm.Embedder, and
// llm.ImageGenerator. Capabilities reported via Info() are per-(provider ×
// model): the constructor binds a model, and Info() reflects what that model
// can do (Keystone K2) — Embed and GenerateImage return
// llm.ErrCapabilityNotSupported when the bound model lacks the capability.
// Streaming events follow the typed K1 union with stable per-tool-call Index
// across deltas.
package openai
```

In `minimax/doc.go`, update to mention image + embeddings:

```go
// Package minimax implements a MiniMax adapter. Chat/stream/tools go over an
// Anthropic-compatible messages API (github.com/anthropics/anthropic-sdk-go);
// image generation (/v1/image_generation) and embeddings (/v1/embeddings) use
// raw net/http against the configured base URL. The adapter satisfies
// llm.ChatModel, llm.ToolCaller, llm.ImageGenerator, and llm.Embedder, with
// capabilities gated per bound model (K2): GenerateImage requires image-01 and
// Embed requires embo-01, otherwise llm.ErrCapabilityNotSupported.
package minimax
```

- [ ] **Step 4: Re-run to confirm docs compile**

Run: `GOWORK=off go build ./openai/ ./minimax/`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add openai/doc.go minimax/doc.go
git commit --no-verify -m "docs(openai,minimax): note image generation and minimax embeddings capabilities"
```

---

## Self-Review

**1. Spec coverage** (against the design spec's openai / minimax / Embeddings / Custom headers / Testing sections):

- openai `GenerateImage` on `*OpenAI`, `var _ llm.ImageGenerator`, map `req → ImageGenerateParams`, `Data[] → []GeneratedImage` (b64→Bytes else URL, RevisedPrompt), `wrapErr`, `isImageModel` gating, `Capabilities.ImageGeneration` → **Tasks 2, 3.** ✓ (Extra keys `style`/`background`/`moderation` mapped per spec §"openai".)
- minimax raw-HTTP `POST /v1/image_generation`, `base_resp.status_code` check, `Size "WxH"→width/height` else `Extra["aspect_ratio"]`, `image_urls[]→URL`, struct retains baseURL/apiKey/httpClient, capability gating for `image-01` → **Tasks 4, 5, 6.** ✓
- minimax `Embed`/`EmbedDimensions` raw-HTTP `POST /v1/embeddings`, Bearer + `GroupId` query param (`WithGroupID`, env `MINIMAX_GROUP_ID`), body `{model,texts,type}` (`WithEmbeddingType` default "db"), TOP-LEVEL `vectors`/`total_tokens`, `embo-01→1536`, gate `Capabilities.Embeddings` → **Tasks 4, 5, 7.** ✓
- `WithExtraHeaders(map[string]string)` on BOTH: openai via `option.WithHeaderAdd`; minimax via SDK option (`option.WithHeaderAdd`) AND on raw-HTTP requests (`applyExtraHeaders`) → **Tasks 1, 4, 5.** ✓
- Testing: `httptest.NewServer` mocks with real handlers/assertions — openai b64 payload (Task 3) + url (Task 3); minimax `image_urls` (Task 6) + embeddings `vectors` with GroupId/type assertions (Task 7) → ✓
- Capability-not-supported paths return `llm.ErrCapabilityNotSupported` → openai Task 3, minimax Tasks 6/7. ✓

**2. Placeholder scan:** No "TBD/TODO/implement later". Every code step contains complete Go. The two "Verify at impl" notes (contract error-type shape in Task 5; contract `ImageGenerator` doc-check in Preconditions) are legitimate cross-repo verification gates, not placeholders — each names the exact `go doc` command and the existing `internal/compat` reference to match. ✓

**3. Type consistency:**
- `isImageModel` defined once, in `openai/image.go` (Task 3 replaces the whole file; Task 2's "Correction" note explicitly omits a duplicate). ✓
- `baseResp` / `baseRespError` / `postJSON` / `applyExtraHeaders` defined once in `minimax/httpclient.go` (Task 5), consumed by `image.go` (Task 6) and `embed.go` (Task 7). ✓
- `parseSize` defined once in `minimax/image.go` (Task 6); not referenced elsewhere. ✓
- `MiniMax` struct fields (`baseURL`, `apiKey`, `httpClient`, `extraHeaders`, `groupID`, `embeddingType`) added in Task 4, used consistently in Tasks 5/6/7. ✓
- `capabilitiesForModel` switch grows monotonically: M1 (existing) → +image-01 (Task 6) → +embo-01 (Task 7); fallback unchanged. ✓
- Compile-assert ordering avoids non-compiling intermediates: openai assert lands with `GenerateImage` (Task 3); minimax `ImageGenerator` assert lands with `GenerateImage` (Task 6) and `Embedder` assert lands with `Embed` (Task 7). ✓
- `llm.ImageRequest`/`ImageResponse`/`GeneratedImage`/`Capabilities.ImageGeneration` come from contract v0.3.0 (Preconditions), never redefined here. ✓

**Known impl-time confirmations (do not block planning):**
- Contract error type shape is CONFIRMED: structs with `Provider string` + `Wrapped error` (no `Message`); the plan's `baseRespError`/`postJSON` use `Wrapped: fmt.Errorf(...)` accordingly, matching `internal/compat/errors_anthropic.go`.
- `go.mod` currently pins `openai-go/v3 v3.35.0`; the `ImageGenerateParams`/`ImagesResponse`/`Image` field names used here were read at v3.37.0 and are stable across that range. If `go build` reports a missing field, run `GOWORK=off go doc github.com/openai/openai-go/v3.ImageGenerateParams` and adjust; no logic change expected.
- MiniMax `status_code` → error-class mapping in `baseRespError` is best-effort; only "0→nil, 1004→auth, other→non-nil" is test-pinned.
