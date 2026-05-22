# External Integrations

**Analysis Date:** 2026-05-20

This repo is purely an integration surface — every provider package wraps one external HTTP API. Each `New(...)` constructor binds a *single* model and reports that model's capabilities via `Info()` (per K2 — see `ARCHITECTURE.md`).

## OpenAI (`openai/`)

**Endpoints hit:**
- `POST /v1/chat/completions` (sync + streaming) — `openai/openai.go:26-42` via `client.Chat.Completions.New` / `.NewStreaming`
- `POST /v1/embeddings` — `openai/openai.go:52-81` via `client.Embeddings.New`
- Base URL: SDK default (`https://api.openai.com`) unless overridden via `WithBaseURL` (`openai/options.go:54-56`)

**Auth:**
- Bearer token via `option.WithAPIKey(...)` (`openai/options.go:51-53`)
- Credential surface: `WithAPIKey(string)` explicit arg, falls back to `os.Getenv("OPENAI_API_KEY")` (`openai/options.go:45-47`)
- `WithOrganization(string)` injects `OpenAI-Organization` header (`openai/options.go:60-62`)

**Model IDs:**
- Required via `WithModel(...)` (`openai/options.go:25`). Constructor errors if empty (`openai/options.go:42-44`).
- Embedding-capable model IDs are explicit: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` (`openai/options.go:69-72`, `openai/openai.go:84-93`).
- Test surface uses `gpt-4o-mini` (`internal/contract/generate_test.go:23`).

**`Info().Capabilities` reported:**
```go
// openai/options.go:78-83
Tools:             true,
Embeddings:        embeddings, // true only for the three embedding model IDs
StructuredOutputs: false,
PromptCaching:     false,
```

**Network behavior:**
- Adapter sets `option.WithMaxRetries(0)` (`openai/options.go:50`) — disables the SDK's built-in retries. Retry policy is delegated to the caller.
- Timeout: configurable via `WithTimeout(d)` → `option.WithRequestTimeout` (`openai/options.go:63-65`). No default.
- HTTP client overridable via `WithHTTPClient` (`openai/options.go:57-59`).

**Rate-limit handling:**
- `errors.As(err, *openai.Error)` in `openai/errors.go:23`.
- 429 → `*llm.RateLimitError` with `RetryAfter` lifted from the `Retry-After` header and `Reason = "quota_exhausted"` when `apiErr.Type` or `.Code` is `"insufficient_quota"` (`openai/errors.go:28-42`).
- 5xx → `*llm.TransientError`; 401/403 → `*llm.AuthError`; other 4xx → `*llm.InvalidRequestError`.

**Tool calling:**
- Native function/tool calls via `openai.ChatCompletionFunctionTool` (`openai/map.go:38-54`); parallel tool calls enabled (`p.ParallelToolCalls = openai.Bool(true)` at `openai/map.go:40`).
- Stream emits `EventToolCallStart` / `EventToolCallArgsDelta` / `EventToolCallEnd` keyed on `tool.Index` (`openai/openai.go:183-227`).

**Vision / multimodal:** Not implemented. Messages convert only `m.Content` strings (`openai/map.go:18-26`).

**Which model families:** Chat Completions only. No Responses API, no Realtime, no Assistants. Embeddings is the second endpoint.

---

## Anthropic (`anthropic/`)

**Endpoints hit:**
- `POST /v1/messages` (sync + streaming) — `anthropic/anthropic.go:24-42` via `client.Messages.New` / `.NewStreaming`
- Base URL: SDK default unless overridden via `WithBaseURL` (`anthropic/options.go:54-56`).

**Auth:**
- `x-api-key` header via `option.WithAPIKey(...)` (`anthropic/options.go:51-53`; capture script confirms header at `scripts/capture-fixtures-anthropic.sh:18`)
- Credential surface: `WithAPIKey(string)` explicit, falls back to `os.Getenv("ANTHROPIC_API_KEY")` (`anthropic/options.go:45-47`)
- `WithBetaHeader(v)` injects `anthropic-beta` header (`anthropic/options.go:60-62`) — opt-in only; not auto-set.

**Model IDs:**
- Required via `WithModel(...)` (`anthropic/options.go:25`).
- Default fixture uses `claude-3-5-haiku-20241022` (`internal/contract/generate_test.go:30`).
- `MaxTokens` defaults to 1024 (`anthropic/map.go:31`) since the Anthropic API requires it.

**`Info().Capabilities` reported:**
```go
// anthropic/options.go:73-78
Tools:             true,
Embeddings:        false, // adapter does NOT implement llm.Embedder at all
StructuredOutputs: false,
PromptCaching:     false,
```
`Anthropic` deliberately does NOT satisfy `llm.Embedder` — only `llm.ChatModel` and `llm.ToolCaller` (`anthropic/anthropic.go:13-16`). The conformance suite confirms this gap: `embed_not_supported_claude-3-5-haiku.json` and `AssertEmbed` (`internal/contract/contract.go:162-173`).

**Network behavior:**
- `option.WithMaxRetries(0)` (`anthropic/options.go:50`) — same retry-disable as OpenAI.
- Timeout via `WithTimeout(d)` → `option.WithRequestTimeout` (`anthropic/options.go:63-65`).
- HTTP client overridable via `WithHTTPClient` (`anthropic/options.go:57-59`).

**Rate-limit handling:**
- `errors.As(err, *sdk.Error)` (`anthropic/errors.go:25`).
- 429 → `*llm.RateLimitError` with `RetryAfter` lifted from the `Retry-After` header; **529 (`overloaded_error`) → `*llm.RateLimitError`** (same `RetryAfter` lift) (`anthropic/errors.go:30-41`). The 529 bucket is an Anthropic-specific status called out in the README (`anthropic/README.md:21`).
- 5xx → `*llm.TransientError`; 401/403 → `*llm.AuthError`; other 4xx → `*llm.InvalidRequestError`.
- `Retry-After` is now parsed (P1-9, 2026-05-22) — raw RFC 7231 string passthrough matching `openai/errors.go` and `deepseek/errors.go`. Consumers parse seconds vs. HTTP-date themselves per the `llm.RateLimitError.RetryAfter` contract.

**Tool calling:**
- Native `tool_use` blocks via `sdk.ToolParam` + `sdk.ToolChoiceUnionParam{OfAuto: ...}` (`anthropic/map.go:43-58`).
- Schema mapped in `toToolInputSchema` (`anthropic/map.go:108-145`); properties, required, type, and ExtraFields are honored.
- Stream maps `ContentBlockStartEvent{ToolUseBlock}` → `EventToolCallStart`, `InputJSONDelta` → `EventToolCallArgsDelta`, `ContentBlockStopEvent` → `EventToolCallEnd` keyed by block `Index` (`anthropic/anthropic.go:137-186`).

**System prompts:**
- `Request.SystemPrompt` is lifted to top-level `system` field (Anthropic-specific) (`anthropic/map.go:12-13`, `:34-36`).
- `role=system` messages in `req.Messages` are concatenated into the same top-level field (`anthropic/map.go:20-26`).

**Extended thinking:**
- Streaming maps `ThinkingBlock` and `ThinkingDelta` to `llm.EventThinkingDelta` (`anthropic/anthropic.go:153-155`, `:171-173`). No explicit knob to *enable* thinking — controlled via `WithBetaHeader(...)` and request-side configuration not in scope here.

**Prompt caching / Files API / batch:** Not implemented. `Capabilities.PromptCaching = false`.

**Vision / multimodal:** Not implemented in the request mapper. Only text blocks via `sdk.NewTextBlock` (`anthropic/map.go:16-19`).

---

## Ollama (`ollama/`)

**Endpoints hit:**
- `POST /api/chat` (sync uses `stream=false`, stream uses `stream=true`) — `ollama/ollama.go:26-64`, request shape in `ollama/map.go:10-47`.
- `POST /api/embed` — `ollama/ollama.go:84-87`.
- Default base URL: `http://localhost:11434` (`ollama/options.go:61-63`).

**Auth:**
- **Keyless by design.** No `WithAPIKey` option (verify by inspecting `ollama/options.go:23-33` — only model/baseURL/httpClient/timeout).
- README explicitly: "keyless adapter, defaults to `OLLAMA_HOST` or `http://localhost:11434`" (`ollama/README.md:14`).
- If the server *does* enforce auth, errors map via `errors.As(err, api.AuthorizationError{...})` to `*llm.AuthError` (`ollama/errors.go:28-31`).

**Local-only assumptions:**
- Default base URL points at `http://localhost:11434`.
- `WithHost(u)` is an alias of `WithBaseURL(u)` (`ollama/options.go:29`) for ergonomic match with the `OLLAMA_HOST` env var.
- Base URL is auto-prefixed with `http://` if scheme is missing (`ollama/options.go:64-66`).
- The SDK is reached via `api.NewClient(*url.URL, *http.Client)` (`ollama/options.go:95`), not through HTTPS-only or hosted endpoints.

**Model IDs:**
- Required via `WithModel(...)` (`ollama/options.go:25`).
- Tool support is model-dependent — see `ollama/tool_strategy.go:20-48`:
  - `llama3.1*` → `"python_tag"` parser, `supportsTool=true`
  - `qwen2.5-coder*`, `qwen3-coder*` → `"qwen_json_or_xml"` parser, `supportsTool=true`
  - everything else → `supportsTool=false`, `WithTools` returns `ErrCapabilityNotSupported` (`ollama/tool_strategy.go:161-163`, `ollama/ollama.go:68-75`)
- Embedding support: `nomic-embed-text` (768), `all-minilm` (384) — `ollama/embed_strategy.go:10-20`. Other models → `Embed()` returns `ErrCapabilityNotSupported` (`ollama/embed_strategy.go:26-28`, `ollama/ollama.go:78-80`).

**`Info().Capabilities` reported:**
```go
// ollama/options.go:103-108
Tools:             strategy.supportsTool,    // per-model
Embeddings:        embedDim > 0,             // per-model
StructuredOutputs: false,
PromptCaching:     false,
```
This is the *clearest* per-(provider × model) capability expression in the repo and the cleanest K2 implementation.

**Network behavior:**
- Custom HTTP transport `statusCapturingTransport` (`ollama/options.go:35-46`) — wraps the inner `RoundTripper` to record the last response status into an `*int32` shared with the adapter, since the Ollama SDK doesn't always surface status in returned errors.
- Timeout via `WithTimeout(d)` set on `http.Client.Timeout` (`ollama/options.go:83-85`).
- No SDK-level retry to disable — `client.Chat` is a simple callback wrapper.

**Rate-limit handling:**
- Ollama has no native rate-limit semantics. `wrapErr` (`ollama/errors.go:14-56`) maps statuses generally:
  - 401/403 → `*llm.AuthError`
  - 404 + "not found"/"not pulled" → `*llm.InvalidRequestError` (model-pull miss)
  - 5xx → `*llm.TransientError`
  - other 4xx → `*llm.InvalidRequestError`
- `connection refused` substring → `*llm.TransientError` (`ollama/errors.go:52-54`).
- Status pulled from `statusCapturingTransport.last` (atomic load at `ollama/errors.go:25`), with `api.StatusError.StatusCode` as a fallback (`ollama/errors.go:32-35`).

**Tool calling:**
- Two layers (`ollama/tool_strategy.go:50-63`):
  1. Native `resp.Message.ToolCalls` — preferred when present (`mapNativeToolCalls`, `ollama/tool_strategy.go:65-82`).
  2. Per-model textual fallback parser — `parsePythonTagToolCalls` (Llama 3.1's `<|python_tag|>` marker, `ollama/tool_strategy.go:90-108`) or `parseQwenToolCalls` (regex over `<tool_call>{...}</tool_call>` blocks, `ollama/tool_strategy.go:110-141`).
- Streaming surface: only text deltas + a single `EventDone` are emitted; **tool-call delta events are NOT emitted in the stream path** (`ollama/ollama.go:120-187`). The streaming reader emits text-only, and `Done=true` produces the terminal event. Tool calls are exposed only through `Generate`.

**Vision / multimodal:** Not implemented in the request mapper (`ollama/map.go:16-21` constructs `api.Message{Role, Content}` only).

---

## DeepSeek (`deepseek/`)

**Endpoints hit:**
- `POST /v1/chat/completions` (sync + streaming) on `https://api.deepseek.com` — `deepseek/deepseek.go:25-41`, base URL constant at `deepseek/options.go:21`.
- Same `openai-go` client (DeepSeek is OpenAI-API-compatible) — `deepseek/options.go:87`.

**Auth:**
- Bearer token (OpenAI-style) — `deepseek/options.go:74-76`.
- Credential surface: `WithAPIKey(string)` explicit, fallback to `os.Getenv("DEEPSEEK_API_KEY")` (`deepseek/options.go:63-65`).

**Regional routing:**
- `WithRegion(RegionCN | RegionGlobal)` (`deepseek/options.go:44`). Both region values currently resolve to `defaultBaseURL = "https://api.deepseek.com"` (`deepseek/options.go:46-53`) — region presets are wired in but converge on the same endpoint today.
- `WithBaseURL(...)` overrides region presets (`deepseek/options.go:67-70`).

**Model IDs:**
- Required via `WithModel(...)` (`deepseek/options.go:34`); no allow-list.
- README example uses `deepseek-chat` (`deepseek/README.md:26`).

**`Info().Capabilities`:**
```go
// deepseek/options.go:93-98
Tools:             true,
Embeddings:        false,
StructuredOutputs: false,
PromptCaching:     false,
```
DeepSeek does NOT satisfy `llm.Embedder` — only `llm.ChatModel` and `llm.ToolCaller` (`deepseek/deepseek.go:14-17`).

**Network behavior:**
- `option.WithMaxRetries(0)` (`deepseek/options.go:73`).
- Timeout via `WithTimeout(d)` (`deepseek/options.go:83-85`).
- HTTP client overridable via `WithHTTPClient` (`deepseek/options.go:80-82`).

**Rate-limit handling:**
- Identical to OpenAI's logic — same `openai.Error` extraction, same 429 → `*llm.RateLimitError` with `Retry-After` header and `insufficient_quota` reason, same 5xx → transient (`deepseek/errors.go:12-57`).

**Tool calling:**
- Native OpenAI-style function tools, parallel calls enabled (`deepseek/map.go:38-54`).
- Stream → typed events with stable `Index` per tool call (`deepseek/deepseek.go:138-181`).

**Anything special:**
- Reuses the OpenAI SDK rather than wrapping its own HTTP layer.
- Region presets are present but not yet diverged — visible expansion seam.

---

## MiniMax (`minimax/`)

**Endpoints hit:**
- `POST /v1/messages` (sync + streaming) on `https://api.minimax.chat` — `minimax/minimax.go:24-42`, base URL constant at `minimax/options.go:21`.
- Uses Anthropic's SDK because MiniMax exposes an Anthropic-compatible Messages API (`minimax/options.go:87`, design doc `docs/superpowers/specs/2026-05-13-deepseek-minimax-design.md:6-7`).

**Auth:**
- Anthropic-style header via `option.WithAPIKey(...)` — `minimax/options.go:74-76`.
- Credential surface: `WithAPIKey(string)` explicit, fallback to `os.Getenv("MINIMAX_API_KEY")` (`minimax/options.go:63-65`).

**Regional routing:**
- `WithRegion(RegionCN | RegionGlobal)` (`minimax/options.go:44`). Both currently converge on `https://api.minimax.chat` (`minimax/options.go:46-53`) — same not-yet-diverged seam as DeepSeek.
- `WithBaseURL(...)` overrides region presets (`minimax/options.go:67-70`).

**Model IDs:**
- Required via `WithModel(...)` (`minimax/options.go:34`); README example uses `MiniMax-M1` (`minimax/README.md:26`).
- `MaxTokens` default 1024 (`minimax/map.go:31`) inherited from the Anthropic-style requirement.

**`Info().Capabilities`:**
```go
// minimax/options.go:93-98
Tools:             true,
Embeddings:        false,
StructuredOutputs: false,
PromptCaching:     false,
```
Does NOT satisfy `llm.Embedder` (`minimax/minimax.go:13-16`).

**Network behavior:**
- `option.WithMaxRetries(0)` (`minimax/options.go:73`).
- Timeout via `WithTimeout(d)` (`minimax/options.go:83-85`).

**Rate-limit handling:**
- Same as Anthropic: 429 → `*llm.RateLimitError`, **529 → `*llm.RateLimitError`** with `RetryAfter` lifted from the `Retry-After` header (`minimax/errors.go:28-41`). `Retry-After` is now parsed (P1-9, 2026-05-22) — same raw string passthrough as Anthropic.

**Tool calling:**
- Same Anthropic schema path (`minimax/map.go:43-58`, `:108-145`).
- Stream events: text + tool-use blocks + thinking deltas (`minimax/minimax.go:122-194`).

**Anything special:**
- Reuses Anthropic SDK; mirrors `anthropic/` 1:1 with package name swapped and provider field changed to `"minimax"`.
- Region presets converge on the same endpoint today.

---

## Cross-Provider Summary

| Capability | openai | anthropic | ollama | deepseek | minimax |
|------------|--------|-----------|--------|----------|---------|
| `ChatModel.Generate` | yes | yes | yes | yes | yes |
| `ChatModel.Stream` | yes | yes | yes (text-only) | yes | yes |
| `ToolCaller.WithTools` | yes | yes | per-model | yes | yes |
| `Embedder.Embed` | yes (3 models) | **no** | per-model | **no** | **no** |
| Stream emits tool-call deltas | yes | yes | **no** | yes | yes |
| Vision / multimodal request | no | no | no | no | no |
| Retry/backoff (internal) | none (`MaxRetries=0`) | none | none | none | none |
| Parses `Retry-After` | yes | yes | n/a | yes | yes |

## Credential Surface (consolidated)

Every provider that needs a key follows the same shape:

```go
// shared pattern, e.g. openai/options.go:45-47
if cfg.apiKey == "" {
    cfg.apiKey = os.Getenv("<PROVIDER>_API_KEY")
}
```

`WithAPIKey` is always preferred over the env var. `New(...)` does *not* fail if both are empty — the SDK call later returns 401, which maps to `*llm.AuthError`. The only constructor-required option is `WithModel`.

---

*Integration audit: 2026-05-20*
