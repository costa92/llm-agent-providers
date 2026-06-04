# Image Generation + Google/Volcengine Providers Design

Date: 2026-06-04
Repos: `llm-agent-contract` (new capability), `llm-agent-providers` (impl)
Status: proposed

## Goal

Two coupled deliverables in one milestone:

1. A first-class **text-to-image generation** capability (`ImageGenerator`)
   across four providers: `openai`, `minimax`, `volcengine` (new),
   `google` (new).
2. Two **new full providers** — `volcengine` (火山方舟 Ark / 豆包) and
   `google` (Gemini) — that are first-class siblings of `openai`/`anthropic`:
   they implement **ChatModel + ToolCaller** (Generate / Stream / WithTools /
   Info), **`ImageGenerator`**, and **`Embedder`**, with capabilities gated per
   bound model (K2).
3. **Embeddings** (`Embedder`, already in the contract — NO contract change) on
   the three providers that have an embedding product: `volcengine`, `google`,
   `minimax`.

The image capability follows the contract's existing **orthogonal capability**
convention (`Embedder`, `ToolCaller`, `StructuredOutputs`): a new
`ImageGenerator` interface lives in `llm-agent-contract`, detection is via type
assertion **plus** a `Capabilities.ImageGeneration` flag, and it does NOT embed
`ChatModel`.

### K2 model-gating (applies to all four)

A provider instance binds ONE model at construction. The Go struct implements
every interface its methods cover (ChatModel, ToolCaller, ImageGenerator), but
`Capabilities` reflects the **bound (provider × model) tuple**:

- `google.New(WithModel("gemini-2.5-flash"))` → chat works; `GenerateImage`
  returns `llm.ErrCapabilityNotSupported`.
- `google.New(WithModel("gemini-2.5-flash-image"))` or `"imagen-4.0-generate-001"`
  → `GenerateImage` works; `Generate` (chat) returns the not-supported error.
- Same split for `volcengine`: `doubao-1-5-pro-32k-*` (chat) vs
  `doubao-seedream-4-5-*` (image).

This is exactly the existing pattern (openai implements ChatModel + Embedder,
gated by model).

## Scope

### In scope

- `llm-agent-contract` → **v0.3.0**: `ImageGenerator` interface,
  `ImageRequest` / `ImageResponse` / `GeneratedImage` types,
  `Capabilities.ImageGeneration` field.
- `openai`: `GenerateImage` on `*OpenAI`, model-gated.
- `minimax`: `GenerateImage` via a new raw-HTTP path.
- `volcengine` (new package): full provider — Generate / Stream / WithTools /
  Info + GenerateImage + Embed. Uses official `arkruntime` SDK.
- `google` (new package): full provider — Generate / Stream / WithTools / Info +
  GenerateImage (Nano Banana via GenerateContent + Imagen via GenerateImages) +
  Embed. Uses official `google.golang.org/genai` SDK.
- `minimax`: add `Embed` via a raw-HTTP path (alongside the image raw-HTTP path).
- **Custom request headers**: a `WithExtraHeaders(map[string]string)` option on
  every provider (the gap go-openai has) — for compatible endpoints/gateways
  that need custom auth/routing headers. Applied across chat/stream/image/embed.
- `httptest`-style fixture coverage per provider (constructor/config, generate,
  stream, tool-calling, image generate, capability flags).
- README updates (repo-level + per-package), bilingual `.zh-CN.md` pairs.
- Cross-repo lockstep rollout (contract tagged first, then providers repin).

### Out of scope

- Image **editing** / variations / inpainting (text-to-image `Generate` only).
- **Streaming image** generation.
- **Structured outputs** (`WithSchema`) for the new packages.
- Embeddings for providers without an embedding product (deepseek, anthropic).
  Ollama/OpenAI already implement `Embedder`; unchanged here.
- Multimodal / vision embeddings (Volcengine `doubao-embedding-vision` via
  `CreateMultiModalEmbeddings`) — text embeddings only.
- go-openai-surfaced capabilities NOT adopted (YAGNI): image edit/variation,
  audio (transcription/TTS), moderation, batch, fine-tuning, files, assistants,
  legacy completions.
- Google **Vertex AI** backend (API-key Gemini Developer API only;
  `BackendVertexAI` is a later config swap).
- Volcengine **AK/SK signed** visual API (`visual.volcengineapi.com`) and Ark
  **endpoint-id (ep-xxxx)** provisioning — model-name direct calls only.
- Volcengine Ark-private extensions (Thinking / reasoning_content / encrypted
  content). Map standard chat only; leave hooks but don't surface them.
- Gemini opt-in `streamFunctionCallArguments` (Gemini 3 Pro+) — treat each
  streamed functionCall chunk as complete.
- Live integration tests (mock-only, as with existing adapters).
- A provider registry/factory (the repo intentionally has none).

## Contract additions (`llm-agent-contract`, v0.3.0)

New file `llm/image.go`:

```go
package llm

import "context"

// ImageGenerator is the capability for text-to-image generation. Like
// Embedder, it deliberately does NOT embed ChatModel. Providers without an
// image endpoint (anthropic, deepseek, ollama) do NOT implement this
// interface; callers detect via type assertion AND Capabilities.ImageGeneration.
type ImageGenerator interface {
	GenerateImage(ctx context.Context, req ImageRequest) (ImageResponse, error)
}

// ImageRequest describes a text-to-image call. Providers ignore knobs they do
// not support (documented per provider).
type ImageRequest struct {
	Prompt  string         // required
	N       int            // number of images; 0 => provider default (1)
	Size    string         // e.g. "1024x1024"; "" => provider default
	Quality string         // e.g. "standard"/"hd"/"high"; "" => provider default
	Format  string         // output encoding "png"/"jpeg"/"webp"; "" => provider default
	Extra   map[string]any // provider-specific knobs, forwarded verbatim
}

// ImageResponse is the result, in request order.
type ImageResponse struct {
	Images   []GeneratedImage
	Provider string
	Model    string
	Usage    Usage // best-effort; zero when the provider does not report it
}

// GeneratedImage is one produced image. Exactly one of Bytes or URL is
// populated, chosen by the provider's most direct delivery path.
type GeneratedImage struct {
	Bytes         []byte // inline bytes (base64-decoded) when returned inline
	URL           string // hosted link when the provider returns a URL
	MimeType      string // e.g. "image/png"; "" if unknown
	RevisedPrompt string // provider's rewritten prompt, if any (dall-e-3)
}
```

`Capabilities` gains: `ImageGeneration bool \`json:"image_generation"\``.

**Delivery rule:** caller does NOT request URL-vs-bytes. OpenAI b64 → `Bytes`;
Volcengine/Minimax url → `URL`; Google always → `Bytes`.

Since `volcengine` and `google` are now full ChatModels, `Info()` comes from the
`ChatModel` interface (no longer a "convenience" method).

## Provider × API × SDK matrix

| Provider   | Chat API                         | Image API                          | Auth               | SDK                                  | New dep |
|------------|----------------------------------|------------------------------------|--------------------|--------------------------------------|---------|
| openai     | existing                         | native `Images.Generate`           | `OPENAI_API_KEY`   | `openai-go/v3` (existing)            | no      |
| minimax    | existing                         | proprietary `/v1/image_generation` | `MINIMAX_API_KEY`  | anthropic SDK (chat) + raw HTTP (img)| no      |
| volcengine | Ark `/chat/completions` (OpenAI-shaped) | Ark `/images/generations`   | `Bearer ARK_API_KEY` | official `arkruntime`              | **yes** |
| google     | Gemini `:generateContent`        | Gemini inline + Imagen `:predict`  | `GEMINI_API_KEY`   | official `google.golang.org/genai`   | **yes** |

New module deps (verify exact versions at impl): `google.golang.org/genai`
(read at v1.59.0), `github.com/volcengine/volcengine-go-sdk` (read at v1.2.33,
`service/arkruntime` + `.../model` + `.../utils`).

## Per-provider design

Each new package follows the repo file skeleton: `doc.go`, `<provider>.go`,
`options.go`, `map.go`, `errors.go` (+ `image.go` where the image path is large).

### openai — `openai/image.go`

- `GenerateImage` on `*OpenAI`; `var _ llm.ImageGenerator = (*OpenAI)(nil)`.
- Map `req → openai.ImageGenerateParams{Prompt, Model: o.info.Model, N, Size,
  Quality, OutputFormat}`; `Extra` keys `style`/`background`/`moderation` when
  present. Call `client.Images.Generate`; `Data[]` → `[]GeneratedImage`
  (`B64JSON` decode → `Bytes`, else `URL`, `RevisedPrompt`). Errors via `wrapErr`.
- **K2 gating:** `isImageModel(cfg.model)` (`gpt-image-1/2`, `dall-e-2/3`) sets
  `Capabilities.ImageGeneration`; non-image models →
  `llm.ErrCapabilityNotSupported` (mirrors `Embed` gating).

### minimax — `minimax/image.go` (first raw-HTTP path in repo)

- `MiniMax` struct additionally retains `baseURL`/`apiKey`/`httpClient`.
- `POST {baseURL}/v1/image_generation` with `{model, prompt, n,
  response_format:"url", aspect_ratio | width+height}`. `Size` "WxH" →
  `width`/`height`; else `Extra["aspect_ratio"]`. `data.image_urls[]` → `URL`.
- **Gotcha:** Minimax returns HTTP 200 on logical failure; check
  `base_resp.status_code != 0` → typed `llm.*` error (status mapping reuses
  `internal/compat`).
- `capabilitiesForModel("image-01")` sets `ImageGeneration: true`.

### volcengine — new full provider (arkruntime SDK)

**Construction** (`arkruntime` v1.2.33):
`arkruntime.NewClientWithApiKey(apiKey, arkruntime.WithRegion("cn-beijing"),
arkruntime.WithBaseUrl(...), arkruntime.WithTimeout(...))`.
Options: `WithModel` (required, model-name direct, e.g. `doubao-1-5-pro-32k-250115`
or `doubao-seedream-4-5-251128`), `WithAPIKey` (env `ARK_API_KEY`), `WithBaseURL`,
`WithRegion`, `WithHTTPClient`, `WithTimeout`. Model id is a config value —
never hard-code (account-dependent).

**Chat (Generate):** `client.CreateChatCompletion(ctx, req)` using
`model.CreateChatCompletionRequest` (the **pointer-field** variant, so
`temperature=0` is sendable; the value-field `ChatCompletionRequest` is
deprecated). Messages → `[]*model.ChatCompletionMessage` with content union
`&model.ChatCompletionMessageContent{StringValue: ...}`; roles
`system/user/assistant/tool`. Response text from
`resp.Choices[0].Message.Content.StringValue`; `Usage{PromptTokens,
CompletionTokens, TotalTokens}`; finish reason from `Choices[0].FinishReason`.

**Stream:** `client.CreateChatCompletionStream(ctx, req)` →
`*utils.ChatCompletionStreamReader`; loop `stream.Recv()` until `io.EOF`
(`stream.Close()` to release body). Emit the repo's typed `llm.StreamEvent`s
from `Choices[0].Delta` (`Content` → `EventTextDelta`).

**Tools (WithTools):** request `model.Tool{Type: ToolTypeFunction, Function:
*model.FunctionDefinition{Name, Description, Parameters: json.RawMessage}}`,
`ToolChoice`. Non-stream tool calls at `Choices[0].Message.ToolCalls[]`
(`ID`, `Function.Name`, `Function.Arguments` JSON string). **Streamed tool
calls** arrive fragmented (OpenAI-shaped): `Delta.ToolCalls[].Index *int` is the
merge key — accumulate `Function.Arguments` substrings per `Index`, emit
`EventToolCallStart` (first fragment with Name/ID) → `EventToolCallArgsDelta` →
`EventToolCallEnd`, preserving stable per-call Index (K1). This mirrors the
existing `openai` stream reader almost verbatim.

**Image (GenerateImage):** `client.GenerateImages(ctx,
model.GenerateImagesRequest{Model, Prompt, Size, ResponseFormat, Seed,
GuidanceScale, Watermark, N})`; `Extra` carries `seed`/`guidance_scale`/
`watermark`. `model.ImagesResponse.Data[]` → `URL` (default) or `Bytes`
(b64_json).
- **Gotchas:** `response_format=url` links expire ~24h (doc note: b64_json to
  persist); `size` constraints vary by model (3.0 t2i 512–2048px; 4.x/5.0 up to
  4096 + `1K`/`2K`/`4K` tiers).

**Errors:** `*model.APIError{HTTPStatusCode, Code, RequestId}` and
`*model.RequestError{HTTPStatusCode}` — both carry HTTP status; map to
`llm.AuthError`/`RateLimitError`/`TransientError`/`InvalidRequestError`
(`errors.As`). SDK has built-in retry; set `WithRetryTimes(0)` to keep our
single-attempt policy consistent with other adapters.

**Capabilities:** `capabilitiesForModel(model)` — `Tools: true` for chat
models, `ImageGeneration: true` for `doubao-seedream*`.

### google — new full provider (genai SDK), Nano Banana + Imagen

**Construction** (`genai` v1.59.0): `genai.NewClient(ctx, &genai.ClientConfig{
APIKey, Backend: genai.BackendGeminiAPI})`. Options: `WithModel` (required),
`WithAPIKey` (env `GEMINI_API_KEY`, fallback `GOOGLE_API_KEY`), `WithHTTPClient`
(→ `ClientConfig.HTTPClient`), `WithBaseURL` (→ `ClientConfig.HTTPOptions.BaseURL`,
used for httptest), `WithTimeout`. Do NOT set Project/Location/Credentials
(mutually exclusive with APIKey).

**Chat (Generate):** `client.Models.GenerateContent(ctx, model, contents,
*genai.GenerateContentConfig)`. Roles only `user`/`model`; **system prompt →
`GenerateContentConfig.SystemInstruction *Content`** (no system role).
`Temperature`/`TopP` are `*float32` (use `genai.Ptr`), `MaxOutputTokens` is
plain `int32`. Read text via `resp.Text()`; finish reason
`Candidates[0].FinishReason` (e.g. `STOP`, `MAX_TOKENS`); usage
`UsageMetadata{PromptTokenCount, CandidatesTokenCount, TotalTokenCount}`
(nil-guard). Map `STOP`→stop, `MAX_TOKENS`→length, etc.

**Stream:** `client.Models.GenerateContentStream(...)` returns a Go-1.23
`iter.Seq2[*GenerateContentResponse, error]` (range-over-func). Bridge to the
repo's pull-based `StreamReader` with **`iter.Pull2`** (go.mod is go 1.26 — OK):
the stream reader's lazy-open returns the `next()`; each chunk's partial text is
`Candidates[0].Content.Parts[].Text` → `EventTextDelta`. Loop ends when
`iter.Pull2`'s `next` reports done.

**Tools (WithTools):** request `genai.Tool{FunctionDeclarations:
[]*genai.FunctionDeclaration{{Name, Description, ParametersJsonSchema: <raw>}}}`
— use **`ParametersJsonSchema any`** (unmarshal the repo's JSON-schema bytes to
`map[string]any` and assign; no typed `genai.Schema` translation). Non-stream
tool calls via `resp.FunctionCalls()` → `{Name, Args map[string]any, ID}`
(re-marshal `Args` to JSON string for the repo's `ToolCall.Arguments`).
**Streamed tool calls arrive COMPLETE in one chunk** (not fragmented) → emit
`EventToolCallStart` + `EventToolCallArgsDelta`(full args) + `EventToolCallEnd`
together per call; correlate parallel calls by `FunctionCall.ID`/part index. No
cross-chunk argument accumulation needed (simpler than openai/volcengine).

**Image (GenerateImage):** route by bound model id:
- `strings.HasPrefix(model, "imagen")` → `client.Models.GenerateImages(ctx,
  model, prompt, &genai.GenerateImagesConfig{NumberOfImages: N, AspectRatio})`;
  `resp.GeneratedImages[].Image.ImageBytes` → `Bytes`.
- else (Gemini-native, e.g. `gemini-2.5-flash-image`) →
  `GenerateContent(..., &GenerateContentConfig{ResponseModalities:
  []string{"TEXT","IMAGE"}})`; `Candidates[0].Content.Parts[].InlineData{Data,
  MIMEType}` → `Bytes`.
- **Gotchas:** output is ALWAYS base64 inline (no URL) → always `Bytes`.
  `ResponseModalities` must include `TEXT` (image-only rejected for Gemini 2.5
  Flash Image); drop text parts. Gemini-native has no clean `N` (≈1); `Size`
  maps loosely to `aspectRatio`/`imageSize`. Every image carries a non-removable
  SynthID watermark.

**Errors:** `genai.APIError{Code int, Status, Message}` returned **by value** —
`var e genai.APIError; errors.As(err, &e)`; switch `e.Code` (401/403/429/5xx).
Nil-guard `Candidates` (prompt may be blocked → `PromptFeedback.BlockReason`).

**Capabilities:** `capabilitiesForModel(model)` — `Tools: true` for
`gemini-*` chat models; `ImageGeneration: true` for `gemini-*-image` and
`imagen-*`.

**Default models:** chat `gemini-2.5-flash` (stable; avoid the shut-down
`gemini-2.0-flash`); image `gemini-2.5-flash-image` / `imagen-4.0-generate-001`.
Keep all model ids configurable.

### Embeddings (`Embedder` on minimax / volcengine / google)

`Embedder` already exists in the contract — **no contract change**. The
interface is fixed: `Embed(ctx, texts []string) ([]Vector, Usage, error)` +
`EmbedDimensions() int`. `Vector` is `[]float32`. It carries no per-call
query/document or dimensions knob, so those are **construction-time provider
options**, model-gated like every other capability (`Capabilities.Embeddings`).

**minimax** — raw HTTP (no SDK embeddings):
- `POST {baseURL}/v1/embeddings`, `Authorization: Bearer`, **`GroupId` as a query
  param** → new option `WithGroupID` (env `MINIMAX_GROUP_ID`; required for embed).
- Body `{model, texts, type}`; response has **top-level** `vectors [][]float32`
  and `total_tokens` (NOT nested under `data`/`usage`); check `base_resp`.
- `type` is `"db"` (document, default) / `"query"` → option `WithEmbeddingType`.
- Model `embo-01`, fixed **1536** dims → `EmbedDimensions()` returns 1536.

**volcengine** — `arkruntime` `CreateEmbeddings(ctx,
model.EmbeddingRequestStrings{Input: texts, Model, Dimensions})`; response
`Data[].Embedding []float32` (in index order), `Usage{PromptTokens, TotalTokens}`.
Models `doubao-embedding-text-240715` (2560, down to 512/1024/2048),
`doubao-embedding-large-text-240915` (4096). `WithDimensions(int)` →
`EmbedDimensions()`; default per model.

**google** — `client.Models.EmbedContent(ctx, model, contents,
*genai.EmbedContentConfig)`. Assemble `texts → []*genai.Content` (one Content per
text; `genai.Text` yields one at a time). Response `Embeddings[].Values []float32`
in order. **Gemini Developer API returns no token usage** → `Usage` is zero
(Source `UsageUnknown`). Options: `WithTaskType` (e.g. `RETRIEVAL_DOCUMENT`,
default empty = model default), `WithDimensions` → `EmbedContentConfig.
OutputDimensionality *int32`. Models `gemini-embedding-001` (3072, MRL-truncatable
to 1536/768), `text-embedding-004` (768).

**Capability gating:** `Capabilities.Embeddings = isEmbedModel(model)` per
provider; non-embed models → `llm.ErrCapabilityNotSupported`, `EmbedDimensions()`
returns 0 (matches the existing openai/ollama pattern).

### Custom request headers (`WithExtraHeaders`)

A construction-time `WithExtraHeaders(map[string]string)` option on every
provider, injecting headers into all outbound requests (chat/stream/image/embed).
Applied via each SDK's header mechanism — openai-go `option.WithHeaderAdd`,
anthropic SDK `option.WithHeader`, arkruntime request/config option, genai
`ClientConfig.HTTPOptions.Headers`; raw-HTTP paths (minimax image/embed) set them
on the `*http.Request` directly. **Verify at impl:** exact header-injection hook
for `arkruntime` and `genai`. Reserved headers (`Authorization`, `Content-Type`)
are not overridable — extra headers are additive.

## Error handling (all providers)

Per-provider `wrapErr` → typed `llm.*` errors (`AuthError`, `RateLimitError`
with `Retry-After`, `TransientError`, `InvalidRequestError`), consistent with
existing adapters. `context.Canceled` returned as-is; `DeadlineExceeded` →
`TransientError`. New mapping work: minimax `base_resp.status_code`; arkruntime
`APIError`/`RequestError`; genai `APIError` (by value).

## Testing

- **contract:** interface compile asserts + JSON (de)serialization tests for the
  new types and the `image_generation` capability flag.
- **openai / minimax:** `httptest.NewServer` mocks for image (OpenAI b64,
  Minimax `image_urls`); assert request mapping, decode, capability flag.
- **volcengine:** point `arkruntime` at httptest via `WithBaseUrl`. Cover
  generate, stream (incl. fragmented tool-call merge by `Index`), tool request
  mapping, and image (`data[].url` / `b64_json`). Mock SSE for stream.
- **google:** point `genai` at httptest via `HTTPOptions.BaseURL`. Cover
  generate, stream (`iter.Pull2` bridge), tool calls (complete-in-one-chunk),
  and both image branches (`generateContent` inlineData + Imagen
  `bytesBase64Encoded`).
- **embeddings:** per provider, mock the embed response (minimax top-level
  `vectors`; volcengine `Data[].Embedding`; google `Embeddings[].Values`),
  assert vector order/length, `EmbedDimensions()`, and the `Embeddings`
  capability flag. minimax embed also asserts the `GroupId` query param and
  `type` field.
- Capability-not-supported paths return `llm.ErrCapabilityNotSupported`.
- `go.uber.org/goleak` for stream-reader goroutine/lifecycle (matches existing).

## Rollout (cross-repo lockstep, one milestone)

1. **contract repo:** add `llm/image.go` + `Capabilities.ImageGeneration` +
   tests + doc → PR → merge → tag **v0.3.0** → push.
2. **providers repo:** dev with a local `replace` to contract; implement
   `openai` + `minimax` image methods + `minimax` embed; build full
   `volcengine` + `google` packages (chat + tools + stream + image + embed);
   add `WithExtraHeaders` across providers; add the two new module deps. When
   green: drop `replace`, pin contract `v0.3.0`, `GOWORK=off go mod tidy`,
   update READMEs (+ `.zh-CN.md`), PR → merge → tag.

The replace-guard pre-commit hook auto-strips the local `replace` on commit, so
contract MUST be tagged v0.3.0 before the providers PR can pin a real version.

## Open verification items (resolve at impl, not blocking design)

1. Pin exact SDK versions; re-confirm `arkruntime` symbol/field names
   (`CreateChatCompletionRequest`, `GenerateImagesRequest`, `ToolCall.Index`)
   and `genai` field names (`GeneratedImages` vs `Images`, `ParametersJsonSchema`,
   `iter.Seq2` signature) against the pinned versions.
2. Confirm streamed tool-call argument fragmentation order on a real Ark
   tool-call stream (SDK does not merge; provider must).
3. Verify `genai` streamed functionCall is single-chunk on the Gemini backend at
   the pinned version (research flagged an older test that skipped it).
4. Confirm `openai-go` `ImageGenerateParams` exposes the
   `Quality`/`OutputFormat` enums for the pinned `gpt-image` models.
5. Confirm embed symbols at pinned versions (`arkruntime` `CreateEmbeddings` /
   `EmbeddingRequestStrings`; `genai` `EmbedContent` / `EmbedContentResponse.
   Embeddings[].Values`); confirm Minimax embeddings endpoint, the `GroupId`
   query-param requirement, and the `type` (db/query) field against current
   Minimax docs (research leaned on Spring AI's implementation).
6. Confirm each SDK's custom-header hook (`arkruntime`, `genai`) for
   `WithExtraHeaders`.
