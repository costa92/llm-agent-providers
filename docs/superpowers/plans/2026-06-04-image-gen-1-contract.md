# ImageGenerator Contract Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `ImageGenerator` text-to-image capability to `llm-agent-contract` — a new `llm/image.go` (interface + `ImageRequest`/`ImageResponse`/`GeneratedImage`) plus an `ImageGeneration` flag on `Capabilities` — then tag **v0.3.0** so the `llm-agent-providers` repo can pin it. This is the first (contract-only) deliverable of the image-generation milestone; provider implementations land in a separate plan.

**Architecture:** Follow the contract's existing **orthogonal capability** convention (`Embedder`, `ToolCaller`, `StructuredOutputs`). `ImageGenerator` is a standalone interface that deliberately does **NOT** embed `ChatModel`; callers detect support via type assertion **plus** the runtime `Capabilities.ImageGeneration` bool. No behaviour is added to existing types beyond one new struct field; no existing method signatures change.

**Tech Stack:** Go 1.26 (stdlib only — `context`, `encoding/json`, `testing`), no third-party deps. The umbrella has a `go.work`, but the contract module must be built/tested in isolation, so **all `go` commands MUST be prefixed `GOWORK=off`**.

**Repo:** `/home/hellotalk/code/go/src/github.com/costa92/llm-agent-ecosystem/llm-agent-contract` (module `github.com/costa92/llm-agent-contract`). Current latest tag: `v0.2.0`.

---

## File Structure

### New files

- `llm/image.go` — `ImageGenerator` interface + `ImageRequest` / `ImageResponse` / `GeneratedImage` types.
- `llm/image_test.go` — JSON (de)serialization tests for the new types, the new capability flag, and a compile-assertion mock implementing `ImageGenerator`.

### Modified files

- `llm/capabilities.go` — add `ImageGeneration bool \`json:"image_generation"\`` to the `Capabilities` struct.
- `llm/doc.go` — enumerate the new `ImageGenerator` capability in the package doc.
- `llm/llm_test.go` — update the golden JSON string in `TestProviderInfo_JSONRoundTrip` (the `Capabilities` serialisation now carries `image_generation`).

### Existing reference files (read-only)

- `llm/capabilities.go` — capability-interface doc-comment style (`Embedder`, `StructuredOutputs`).
- `llm/types.go` — `Usage` (struct, comparable) and `Vector` (`[]float32`) definitions.
- `llm/errors.go` — `ErrCapabilityNotSupported` sentinel (referenced in doc-comments).
- `llm/llm_test.go`, `llm/errors_test.go` — test style (table-driven, stdlib `testing`, `t.Fatalf`/`t.Errorf`).

---

## Important pre-flight facts (read before starting)

1. **`go.work` exists** at the umbrella root, so a bare `go test ./...` would pull umbrella replace directives. Every `go` command below is prefixed `GOWORK=off` to build the contract module standalone — do not drop the prefix.
2. **`TestProviderInfo_JSONRoundTrip` is a golden-string test.** It asserts the exact `json.Marshal` output of a `Capabilities` value:
   ```
   {"provider":"openai","model":"gpt-4o-mini","capabilities":{"tools":true,"embeddings":true,"structured_outputs":false,"prompt_caching":false}}
   ```
   Adding `ImageGeneration` (field order **after** `PromptCaching`, so the JSON key lands **last** in the object) WILL break this assertion. Task 2 updates it. The struct-equality check (`out != in`) keeps working because `Capabilities` stays all-comparable (no slices added).
3. **Field placement:** append `ImageGeneration` as the **last** field of `Capabilities` so the JSON key order is `tools, embeddings, structured_outputs, prompt_caching, image_generation`. This keeps existing struct literals that use positional fields (there are none in-repo, but downstream callers may) lowest-risk, and matches the golden-string update in Task 2.
4. **`ChatOnlyMock` needs no change:** it builds `Capabilities{}` (zero value), so `ImageGeneration` defaults to `false` — the existing negative assertions in `llm_test.go` (`TestChatOnlyMockExcludesCapabilities`) still hold and need no edit. Do not touch `chat_only_mock.go`.
5. **`Usage` is a comparable struct** (`InputTokens int`, `OutputTokens int`, `TotalTokens int`, `Source UsageSource`) — so `ImageResponse` embedding `Usage` by value is fine and its zero value is the documented "provider did not report" case.

---

## Task 1 — `Capabilities.ImageGeneration` flag

**Files:**
- Modify: `llm/capabilities.go`
- Test: `llm/llm_test.go` (existing `TestProviderInfo_JSONRoundTrip` — this task makes it RED, Task 2 makes it GREEN; do both before committing Task 1+2 together OR commit the test update inside this task — see step 5)

### Steps

- [ ] **1. Write the failing assertion.** The repo already has `TestProviderInfo_JSONRoundTrip` in `llm/llm_test.go`. Add the new field to the test's input literal and to the golden string so the test now demands the new flag. Edit the existing test body:

  Replace the `Capabilities` literal:
  ```go
  		Capabilities: Capabilities{
  			Tools: true, Embeddings: true, StructuredOutputs: false, PromptCaching: false,
  		},
  ```
  with:
  ```go
  		Capabilities: Capabilities{
  			Tools: true, Embeddings: true, StructuredOutputs: false, PromptCaching: false, ImageGeneration: true,
  		},
  ```
  and replace the golden `want` string:
  ```go
  	want := `{"provider":"openai","model":"gpt-4o-mini","capabilities":{"tools":true,"embeddings":true,"structured_outputs":false,"prompt_caching":false}}`
  ```
  with:
  ```go
  	want := `{"provider":"openai","model":"gpt-4o-mini","capabilities":{"tools":true,"embeddings":true,"structured_outputs":false,"prompt_caching":false,"image_generation":true}}`
  ```

- [ ] **2. Run it — MUST fail to compile** (field `ImageGeneration` does not exist yet):
  ```
  GOWORK=off go test ./llm/ -run TestProviderInfo_JSONRoundTrip
  ```
  Expect a compile error: `unknown field 'ImageGeneration' in struct literal of type Capabilities`. This is the RED state.

- [ ] **3. Minimal implementation.** Add the field to `Capabilities` in `llm/capabilities.go`. Edit the struct — it currently ends:
  ```go
  type Capabilities struct {
  	Tools             bool `json:"tools"`               // Native function-calling supported by the bound model
  	Embeddings        bool `json:"embeddings"`          // Embed() returns vectors (NOT ErrCapabilityNotSupported)
  	StructuredOutputs bool `json:"structured_outputs"`  // WithSchema() applies a JSON schema constraint
  	PromptCaching     bool `json:"prompt_caching"`      // Anthropic explicit / OpenAI auto (consumed Phase 5+)
  }
  ```
  Change it to add `ImageGeneration` as the final field:
  ```go
  type Capabilities struct {
  	Tools             bool `json:"tools"`               // Native function-calling supported by the bound model
  	Embeddings        bool `json:"embeddings"`          // Embed() returns vectors (NOT ErrCapabilityNotSupported)
  	StructuredOutputs bool `json:"structured_outputs"`  // WithSchema() applies a JSON schema constraint
  	PromptCaching     bool `json:"prompt_caching"`      // Anthropic explicit / OpenAI auto (consumed Phase 5+)
  	ImageGeneration   bool `json:"image_generation"`    // GenerateImage() returns images (NOT ErrCapabilityNotSupported)
  }
  ```

- [ ] **4. Run it — MUST pass:**
  ```
  GOWORK=off go test ./llm/ -run TestProviderInfo_JSONRoundTrip
  ```
  Then run the full package to confirm nothing else regressed (the all-false `ChatOnlyMock` assertions still hold because the new field defaults to `false`):
  ```
  GOWORK=off go test ./llm/
  ```

- [ ] **5. Commit.** WHY-focused message:
  ```
  git add llm/capabilities.go llm/llm_test.go
  git commit -m "feat(llm): add ImageGeneration capability flag

  Image generation is an orthogonal per-(provider x model) capability, like
  Embeddings/Tools: a provider's Go type can implement ImageGenerator while a
  specific bound model cannot generate images. Callers need a runtime signal
  alongside the type assertion, so add Capabilities.ImageGeneration. The flag
  lands last in the JSON object to preserve existing key order; the golden
  ProviderInfo round-trip test is updated to match."
  ```

---

## Task 2 — `ImageGenerator` interface + image types

**Files:**
- Create: `llm/image.go`
- Test: `llm/image_test.go`

### Steps

- [ ] **1. Write the failing test** in `llm/image_test.go`. It covers: JSON round-trip of `ImageRequest`, JSON round-trip of `ImageResponse` (incl. nested `GeneratedImage` and `Usage`), `omitempty`/byte-encoding behaviour, and a compile-time assertion that a mock satisfies `ImageGenerator`. Write the complete file:

  ```go
  package llm

  import (
  	"context"
  	"encoding/json"
  	"errors"
  	"fmt"
  	"testing"
  )

  // imageGenMock is a minimal ImageGenerator used only to prove the
  // interface is satisfiable and to exercise ErrCapabilityNotSupported
  // on a model that does not support image generation.
  type imageGenMock struct {
  	resp      ImageResponse
  	supported bool
  }

  // Compile-time: imageGenMock implements ImageGenerator.
  var _ ImageGenerator = (*imageGenMock)(nil)

  func (m *imageGenMock) GenerateImage(_ context.Context, _ ImageRequest) (ImageResponse, error) {
  	if !m.supported {
  		return ImageResponse{}, fmt.Errorf("mock: image generation: %w", ErrCapabilityNotSupported)
  	}
  	return m.resp, nil
  }

  // ----- ImageGenerator interface is satisfiable + honours the sentinel -----
  func TestImageGenerator_Interface(t *testing.T) {
  	ctx := context.Background()

  	// Capability-gated model returns the canonical sentinel.
  	off := &imageGenMock{supported: false}
  	if _, err := off.GenerateImage(ctx, ImageRequest{Prompt: "a cat"}); !errors.Is(err, ErrCapabilityNotSupported) {
  		t.Fatalf("GenerateImage = %v, want ErrCapabilityNotSupported", err)
  	}

  	// Supported model returns its response unchanged.
  	want := ImageResponse{
  		Images:   []GeneratedImage{{Bytes: []byte{0x1, 0x2, 0x3}, MimeType: "image/png"}},
  		Provider: "openai",
  		Model:    "gpt-image-1",
  		Usage:    Usage{InputTokens: 7, TotalTokens: 7, Source: UsageReported},
  	}
  	on := &imageGenMock{supported: true, resp: want}
  	got, err := on.GenerateImage(ctx, ImageRequest{Prompt: "a cat", N: 1, Size: "1024x1024"})
  	if err != nil {
  		t.Fatalf("GenerateImage: %v", err)
  	}
  	if got.Provider != "openai" || got.Model != "gpt-image-1" {
  		t.Errorf("response identity = %s/%s, want openai/gpt-image-1", got.Provider, got.Model)
  	}
  	if len(got.Images) != 1 || string(got.Images[0].Bytes) != "\x01\x02\x03" {
  		t.Errorf("Images = %+v, want one image with bytes 0x010203", got.Images)
  	}
  }

  // ----- ImageRequest JSON round-trip -----
  func TestImageRequest_JSONRoundTrip(t *testing.T) {
  	in := ImageRequest{
  		Prompt:  "a watercolour fox",
  		N:       2,
  		Size:    "1024x1024",
  		Quality: "hd",
  		Format:  "png",
  		Extra:   map[string]any{"style": "vivid"},
  	}
  	b, err := json.Marshal(in)
  	if err != nil {
  		t.Fatalf("Marshal: %v", err)
  	}
  	want := `{"prompt":"a watercolour fox","n":2,"size":"1024x1024","quality":"hd","format":"png","extra":{"style":"vivid"}}`
  	if string(b) != want {
  		t.Errorf("Marshal:\n got  %s\n want %s", b, want)
  	}

  	var out ImageRequest
  	if err := json.Unmarshal(b, &out); err != nil {
  		t.Fatalf("Unmarshal: %v", err)
  	}
  	if out.Prompt != in.Prompt || out.N != in.N || out.Size != in.Size ||
  		out.Quality != in.Quality || out.Format != in.Format {
  		t.Errorf("round-trip scalar mismatch:\n got  %+v\n want %+v", out, in)
  	}
  	if out.Extra["style"] != "vivid" {
  		t.Errorf("round-trip Extra = %+v, want style=vivid", out.Extra)
  	}
  }

  // ----- ImageRequest omitempty: zero-value request stays minimal -----
  func TestImageRequest_JSONOmitEmpty(t *testing.T) {
  	b, err := json.Marshal(ImageRequest{Prompt: "x"})
  	if err != nil {
  		t.Fatalf("Marshal: %v", err)
  	}
  	want := `{"prompt":"x"}`
  	if string(b) != want {
  		t.Errorf("Marshal zero-value:\n got  %s\n want %s", b, want)
  	}
  }

  // ----- ImageResponse JSON round-trip (nested GeneratedImage + Usage) -----
  func TestImageResponse_JSONRoundTrip(t *testing.T) {
  	in := ImageResponse{
  		Images: []GeneratedImage{
  			{Bytes: []byte("PNG"), MimeType: "image/png", RevisedPrompt: "a fox, watercolour"},
  			{URL: "https://example.com/img.png", MimeType: "image/png"},
  		},
  		Provider: "openai",
  		Model:    "gpt-image-1",
  		Usage:    Usage{InputTokens: 5, OutputTokens: 0, TotalTokens: 5, Source: UsageReported},
  	}
  	b, err := json.Marshal(in)
  	if err != nil {
  		t.Fatalf("Marshal: %v", err)
  	}

  	var out ImageResponse
  	if err := json.Unmarshal(b, &out); err != nil {
  		t.Fatalf("Unmarshal: %v", err)
  	}
  	if out.Provider != in.Provider || out.Model != in.Model {
  		t.Errorf("identity mismatch: got %s/%s", out.Provider, out.Model)
  	}
  	if out.Usage != in.Usage {
  		t.Errorf("Usage round-trip:\n got  %+v\n want %+v", out.Usage, in.Usage)
  	}
  	if len(out.Images) != 2 {
  		t.Fatalf("Images len = %d, want 2", len(out.Images))
  	}
  	// Bytes survive base64 round-trip (encoding/json encodes []byte as base64).
  	if string(out.Images[0].Bytes) != "PNG" {
  		t.Errorf("Images[0].Bytes = %q, want PNG", out.Images[0].Bytes)
  	}
  	if out.Images[0].RevisedPrompt != "a fox, watercolour" {
  		t.Errorf("Images[0].RevisedPrompt = %q", out.Images[0].RevisedPrompt)
  	}
  	if out.Images[1].URL != "https://example.com/img.png" {
  		t.Errorf("Images[1].URL = %q", out.Images[1].URL)
  	}
  	if out.Images[1].Bytes != nil {
  		t.Errorf("Images[1].Bytes = %v, want nil (URL-delivered image)", out.Images[1].Bytes)
  	}
  }
  ```

- [ ] **2. Run it — MUST fail to compile** (none of `ImageGenerator`, `ImageRequest`, `ImageResponse`, `GeneratedImage` exist yet):
  ```
  GOWORK=off go test ./llm/ -run 'TestImage'
  ```
  Expect compile errors: `undefined: ImageGenerator`, `undefined: ImageRequest`, etc. This is the RED state.

- [ ] **3. Minimal implementation.** Create `llm/image.go` with the exact types from the design spec's "Contract additions" block, with doc-comments matching the `Embedder`/`StructuredOutputs` house style. Write the complete file:

  ```go
  package llm

  import "context"

  // ImageGenerator is the capability for text-to-image generation. Like
  // Embedder, it deliberately does NOT embed ChatModel: a provider's image
  // endpoint is orthogonal to chat. Providers without an image endpoint
  // (anthropic, deepseek, ollama) do NOT implement this interface; callers
  // detect support via type assertion AND consult Capabilities.ImageGeneration
  // on the bound ProviderInfo.
  //
  // A bound model that implements the Go interface but cannot generate images
  // returns ErrCapabilityNotSupported:
  //
  //	return ImageResponse{}, fmt.Errorf("openai: image generation: %w", ErrCapabilityNotSupported)
  type ImageGenerator interface {
  	GenerateImage(ctx context.Context, req ImageRequest) (ImageResponse, error)
  }

  // ImageRequest describes a text-to-image call. Providers ignore knobs they
  // do not support (documented per provider). Only Prompt is required.
  type ImageRequest struct {
  	Prompt  string         `json:"prompt"`            // required
  	N       int            `json:"n,omitempty"`       // number of images; 0 => provider default (1)
  	Size    string         `json:"size,omitempty"`    // e.g. "1024x1024"; "" => provider default
  	Quality string         `json:"quality,omitempty"` // e.g. "standard"/"hd"/"high"; "" => provider default
  	Format  string         `json:"format,omitempty"`  // output encoding "png"/"jpeg"/"webp"; "" => provider default
  	Extra   map[string]any `json:"extra,omitempty"`   // provider-specific knobs, forwarded verbatim
  }

  // ImageResponse is the result, in request order.
  type ImageResponse struct {
  	Images   []GeneratedImage `json:"images"`
  	Provider string           `json:"provider"`
  	Model    string           `json:"model,omitempty"`
  	Usage    Usage            `json:"usage"` // best-effort; zero when the provider does not report it
  }

  // GeneratedImage is one produced image. Exactly one of Bytes or URL is
  // populated, chosen by the provider's most direct delivery path (the caller
  // does NOT request URL-vs-bytes): OpenAI b64 => Bytes; Volcengine/Minimax
  // url => URL; Google always => Bytes.
  type GeneratedImage struct {
  	Bytes         []byte `json:"bytes,omitempty"`          // inline bytes (base64-decoded) when returned inline
  	URL           string `json:"url,omitempty"`            // hosted link when the provider returns a URL
  	MimeType      string `json:"mime_type,omitempty"`      // e.g. "image/png"; "" if unknown
  	RevisedPrompt string `json:"revised_prompt,omitempty"` // provider's rewritten prompt, if any (dall-e-3)
  }
  ```

  > **Note on JSON tags:** the spec's code block shows the Go fields without struct tags. The contract repo's house style is that every serialisable struct carries explicit `json:"..."` tags (see `types.go`, `info.go`) because `Capabilities` is JSON-emitted for OTel. The tags above follow that convention and make the golden-string tests in `image_test.go` deterministic. The field set, types, ordering, and semantics match the spec exactly.

- [ ] **4. Run it — MUST pass:**
  ```
  GOWORK=off go test ./llm/ -run 'TestImage'
  ```
  Then the full package + race detector (the repo uses `-race` in CI; the new code is data-race-free):
  ```
  GOWORK=off go test -race ./llm/
  GOWORK=off go vet ./llm/
  ```

- [ ] **5. Commit.** WHY-focused message:
  ```
  git add llm/image.go llm/image_test.go
  git commit -m "feat(llm): add ImageGenerator capability interface

  Text-to-image is a first-class capability the agents framework needs to
  call (openai/minimax/volcengine/google). Model it as an orthogonal
  interface — like Embedder, it does NOT embed ChatModel — so pure
  image-only adapters stay expressible and chat-only providers are not
  forced to stub it. ImageRequest/ImageResponse/GeneratedImage carry the
  delivery-agnostic result (Bytes XOR URL) chosen by the provider, with
  best-effort Usage."
  ```

---

## Task 3 — Package doc enumerates the new capability

**Files:**
- Modify: `llm/doc.go`

### Steps

- [ ] **1. Verification target (no test — doc-only).** The package doc in `llm/doc.go` enumerates every capability interface (`ChatModel`, `ToolCaller`, `Embedder`, `StructuredOutputs`, ...). It must now list `ImageGenerator`. The "success check" is `go doc` showing the new line and the package still building.

- [ ] **2. Edit `llm/doc.go`.** Insert the `ImageGenerator` bullet in the capability list, immediately after the `Embedder` bullet (keeping the orthogonal-capabilities grouped). The list currently reads:
  ```go
  //   - Embedder           capability: vector embeddings (does NOT embed
  //     ChatModel — orthogonal to chat)
  //   - StructuredOutputs  capability: JSON-schema-constrained output
  ```
  Change it to:
  ```go
  //   - Embedder           capability: vector embeddings (does NOT embed
  //     ChatModel — orthogonal to chat)
  //   - ImageGenerator     capability: text-to-image generation (does NOT
  //     embed ChatModel — orthogonal to chat) (NEW in v0.3)
  //   - StructuredOutputs  capability: JSON-schema-constrained output
  ```
  Then, in the `Capabilities` bullet, the parenthetical lists the bool fields:
  ```go
  //   - Capabilities       per-(provider × model) feature struct
  //     (Tools / Embeddings / StructuredOutputs /
  //     PromptCaching as bool fields; JSON-serializable
  //     for OTel attribute emission)
  ```
  Change it to include the new field:
  ```go
  //   - Capabilities       per-(provider × model) feature struct
  //     (Tools / Embeddings / StructuredOutputs /
  //     PromptCaching / ImageGeneration as bool fields;
  //     JSON-serializable for OTel attribute emission)
  ```

- [ ] **3. Verify build + doc render:**
  ```
  GOWORK=off go build ./llm/
  GOWORK=off go doc ./llm/ | grep -i ImageGenerator
  GOWORK=off go test ./llm/
  ```
  Expect `go doc` to surface the new interface and the package to build/test clean.

- [ ] **4. Commit.** WHY-focused message:
  ```
  git add llm/doc.go
  git commit -m "docs(llm): document ImageGenerator in package overview

  The package doc is the canonical capability index callers read first;
  omitting ImageGenerator would make the new orthogonal capability
  undiscoverable via go doc."
  ```

---

## Task 4 — Full verification gate

**Files:** none (verification only)

### Steps

- [ ] **1. Run the complete suite with race + vet** from the module root (`llm-agent-contract/`):
  ```
  GOWORK=off go test -race ./...
  GOWORK=off go vet ./...
  GOWORK=off gofmt -l llm/
  ```
  `gofmt -l` MUST print nothing (no unformatted files). All tests MUST pass. If `gofmt -l` lists a file, run `gofmt -w llm/image.go llm/image_test.go` and re-run; do not hand-fix tabs.

- [ ] **2. Confirm no accidental API drift.** The only public additions are `ImageGenerator`, `ImageRequest`, `ImageResponse`, `GeneratedImage`, and `Capabilities.ImageGeneration`. No existing signature changed:
  ```
  GOWORK=off go doc ./llm/ | grep -E 'Image|Capabilities'
  ```

---

## Cross-repo handoff (rollout — tag v0.3.0)

This is the **contract-first** half of the milestone's lockstep rollout (spec §Rollout). The providers repo cannot pin a real contract version until v0.3.0 is tagged and pushed (the replace-guard pre-commit hook auto-strips local `replace` directives on commit).

- [ ] **1. Open the PR** from the working branch against `llm-agent-contract` `main`:
  ```
  git push -u origin <branch>
  gh pr create --repo costa92/llm-agent-contract \
    --title "feat(llm): add ImageGenerator capability (v0.3.0)" \
    --body "Adds the orthogonal ImageGenerator capability (interface + ImageRequest/ImageResponse/GeneratedImage) and Capabilities.ImageGeneration. Unblocks image generation in llm-agent-providers (openai/minimax/volcengine/google). Contract-only; no provider impl here."
  ```
  > Note (from project memory): `gh pr edit`/`gh pr view` may fail here due to token scope (`read:org`). If so, edit PR title/body via `gh api -X PATCH repos/costa92/llm-agent-contract/pulls/<N>`. Sibling repos auto-merge owner PRs once CI (GOWORK=off tidy-drift + tests) is green.

- [ ] **2. After merge, tag and push v0.3.0** from the merged `main`:
  ```
  git fetch origin
  git checkout main
  git pull --ff-only origin main
  git tag v0.3.0
  git push origin v0.3.0
  ```

- [ ] **3. Confirm the tag is visible** (this is what the providers repo pins):
  ```
  git ls-remote --tags origin | grep v0.3.0
  ```

- [ ] **4. Handoff to the providers plan.** Once `v0.3.0` is pushed, the providers repo can replace its dev-time local `replace github.com/costa92/llm-agent-contract => ...` with a pinned `require github.com/costa92/llm-agent-contract v0.3.0` and run `GOWORK=off go mod tidy`. That work is a separate plan (`2026-06-04-image-gen-2-providers.md` or sibling) and is **out of scope here**.

---

## Self-review

- **Spec coverage:**
  - `llm/image.go` with `ImageGenerator` + `ImageRequest`/`ImageResponse`/`GeneratedImage` — Task 2. ✅ Field set, types, ordering, and per-field semantics match the spec's code block verbatim (Prompt/N/Size/Quality/Format/Extra; Images/Provider/Model/Usage; Bytes/URL/MimeType/RevisedPrompt).
  - `Capabilities.ImageGeneration bool \`json:"image_generation"\`` — Task 1. ✅ Exact tag from spec line 137.
  - Tests: JSON (de)serialization of new types ✅ (`TestImageRequest_JSONRoundTrip`, `TestImageResponse_JSONRoundTrip`, `TestImageRequest_JSONOmitEmpty`); capability flag ✅ (golden-string update in `TestProviderInfo_JSONRoundTrip`); interface compile-assertion via mock ✅ (`var _ ImageGenerator = (*imageGenMock)(nil)` + `TestImageGenerator_Interface`, also exercising `ErrCapabilityNotSupported`).
  - `doc.go` update ✅ — Task 3.
  - Rollout: PR → merge → tag v0.3.0 → push ✅ — handoff section.
- **No placeholders:** every code step shows COMPLETE Go (no `TODO`, no "similar to above"). ✅
- **Type consistency:** `ImageResponse.Usage` is the repo's comparable `Usage` struct (`types.go`) — `out.Usage != in.Usage` equality is valid. ✅ `GeneratedImage.Bytes []byte` relies on `encoding/json`'s base64 round-trip; the test asserts `string(out.Images[0].Bytes) == "PNG"` and a URL-only image has nil Bytes. ✅ `Vector` is unused here (no embedding change). ✅
- **House-style match:** tabs for indentation, `json:"..."` tags on every serialisable field (matches `types.go`/`info.go`), doc-comments mirror `Embedder`/`StructuredOutputs` (orthogonality note + canonical `ErrCapabilityNotSupported` wrap example), table/`t.Fatalf` test idiom. ✅
- **`GOWORK=off`:** present on every `go` invocation (go.work confirmed at umbrella root). ✅
- **Golden-test gotcha:** `TestProviderInfo_JSONRoundTrip` breakage is anticipated and fixed in Task 1 (field appended last → key appended last). ✅ `ChatOnlyMock` left untouched (zero-value `Capabilities` keeps `image_generation:false`). ✅
- **Field ordering note:** `ImageGeneration` is the **last** struct field, so `json.Marshal` emits `image_generation` as the **last** object key — consistent across the `Capabilities` golden string (Task 1) and the spec's stated tag. ✅
