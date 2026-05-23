<!-- refreshed: 2026-05-20 -->
# Architecture

**Analysis Date:** 2026-05-20

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                  Caller (e.g. llm-agent runtime)                      │
│                  imports llm-agent/llm interfaces                     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ binds one model per provider instance (K2)
                              ▼
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│   openai    │  anthropic  │    ollama   │  deepseek   │   minimax   │
│ `openai/`   │ `anthropic/`│  `ollama/`  │ `deepseek/` │  `minimax/` │
│ openai.go   │ anthropic.go│  ollama.go  │ deepseek.go │  minimax.go │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │             │             │
       ▼             ▼             ▼             ▼             ▼
   openai-go/v3  anthropic-sdk-  ollama/api    openai-go/v3  anthropic-sdk-
   (SSE)         go (SSE)        (ndjson cb)   (SSE)         go (SSE)
       │             │             │             │             │
       └─────────────┴─────────────┴─────────────┴─────────────┘
                              │
                              ▼
                  ┌─────────────────────────────┐
                  │   internal/contract         │
                  │   shared fixture harness    │
                  │   `internal/contract/...`   │
                  └─────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| OpenAI adapter | OpenAI Chat Completions + Embeddings; native tool calls | `openai/openai.go` |
| Anthropic adapter | Messages API; native tool use; thinking deltas; no embeddings | `anthropic/anthropic.go` |
| Ollama adapter | Local Ollama; per-model tool strategy; per-model embeddings | `ollama/ollama.go` |
| DeepSeek adapter | OpenAI-compatible Chat Completions; region presets | `deepseek/deepseek.go` |
| MiniMax adapter | Anthropic-compatible Messages; region presets | `minimax/minimax.go` |
| Contract harness | Fixture-driven mock server + per-capability assertions shared across providers | `internal/contract/contract.go` |
| Conformance suite | Drives every adapter through the harness with fixture cases | `internal/contract/generate_test.go` |

## Pattern Overview

**Overall:** Sibling-package adapter pattern. Each provider is a flat Go package (no sub-packages), constructed via `New(opts ...Option) (*Provider, error)`. The `Provider` struct satisfies one or more `llm.*` interfaces.

**Key Characteristics:**
- **Bound model (K2):** The model is baked into the provider instance at construction; `Info()` returns capabilities that reflect *that* model.
- **Typed stream events (K1):** Every streaming reader emits `llm.StreamEvent` with a `Kind` enum (`EventTextDelta`, `EventToolCallStart`, `EventToolCallArgsDelta`, `EventToolCallEnd`, `EventThinkingDelta`, `EventDone`). Tool-call indices are stable per call within a stream.
- **No internal retry/backoff.** Every SDK is configured with `MaxRetries=0`. The caller owns retry policy.
- **Functional options.** All knobs are `WithX(...)` functions; the only required option is `WithModel`.
- **Per-provider error mapper (`*/errors.go`).** Translates SDK errors into the four `llm-agent` sentinel error types.

## Layers

**Provider packages (`openai/`, `anthropic/`, `ollama/`, `deepseek/`, `minimax/`):**
- Purpose: implement `llm.ChatModel` + optional capabilities for one provider.
- Location: top-level subdirectory.
- Contains: constructor (`options.go`), request/response mapper (`map.go`), wire adapter + stream reader (`<provider>.go`), error wrapper (`errors.go`), package doc (`doc.go`), tests (`<provider>_test.go`), local README.
- Depends on: `github.com/costa92/llm-agent/llm`, official SDK for the provider.
- Used by: external consumers; `internal/contract/generate_test.go`.

**Contract harness (`internal/contract/`):**
- Purpose: cross-provider conformance — load JSON fixtures, spin up `httptest.Server`, run the adapter through `AssertGenerate` / `AssertStream` / `AssertToolCalling` / `AssertEmbed`.
- Location: `internal/contract/contract.go` (helpers), `internal/contract/generate_test.go` (cases).
- Internal-only — not importable by external consumers (Go `internal` rule).

**Scripts (`scripts/`):**
- Purpose: fixture capture against real provider APIs; workspace bootstrap.
- Not loaded by Go code.

## Data Flow

### Generate (sync) path — same shape across all providers

1. Caller invokes `provider.Generate(ctx, llm.Request)` (`openai/openai.go:26-33`, `anthropic/anthropic.go:24-31`, `ollama/ollama.go:26-37`, `deepseek/deepseek.go:25-32`, `minimax/minimax.go:24-31`).
2. `toSDKRequest(req)` translates `llm.Request` into the SDK's native param type (`*/map.go`).
3. SDK call (`client.Chat.Completions.New` / `client.Messages.New` / `client.Chat` callback).
4. On error: `wrapErr(err)` maps SDK errors → `llm.AuthError | RateLimitError | InvalidRequestError | TransientError` (`*/errors.go`).
5. On success: `fromSDKResponse(resp)` translates back into `llm.Response` (text, tool calls, usage, finish reason, provider/model labels).

### Stream path

1. `provider.Stream(ctx, llm.Request)` returns a `llm.StreamReader` immediately. The SDK call is deferred to the first `Next()` (lazy open via `r.open()`).
2. `streamReader.Next()` pulls one underlying SDK event, translates it into zero-or-more `llm.StreamEvent`s queued in `r.queue`, then returns the head of the queue.
3. On a clean run: text/tool-call/thinking deltas, then a terminal `llm.StreamEvent{Kind: EventDone, Usage, FinishReason}`.
4. On error before first byte: one retry attempt is performed inside the reader (re-opens the underlying stream). After delivering any byte (`deliveredByte = true`), errors are returned immediately. See `openai/openai.go:108-144`, `anthropic/anthropic.go:72-108`, `deepseek/deepseek.go:63-99`, `minimax/minimax.go:72-108`.
5. `Close()` is idempotent and tears down the underlying SDK stream.

### Tool-calling path

Two layers of mapping per provider:
1. Request-side: `WithTools(tools)` clones the adapter and stores tools; `toSDKRequest` converts them into the SDK's tool schema (`openai/map.go:38-55`, `anthropic/map.go:43-59`, `ollama/map.go:28-39`, `deepseek/map.go:38-55`, `minimax/map.go:43-59`).
2. Response-side: `fromSDKResponse` extracts native tool calls; Ollama additionally runs `parseResponseToolCalls` (`ollama/tool_strategy.go:50-63`) to fall back to model-specific text formats.

**State Management:**
- Provider structs are value-copied on `WithTools` (`*o`, `*a`, `*m`, `*d`), so the original is immutable.
- Stream readers carry per-call mutable state (queue, retry flag, block-kind maps) behind a `sync.Mutex`.
- Ollama additionally uses an `atomic.Int32` for last HTTP status, shared with a custom `RoundTripper`.

## Key Abstractions

**`llm.ChatModel`:** core interface, satisfied by every provider. Generate + Stream + Info.

**`llm.ToolCaller`:** optional. `WithTools(tools) (ToolCaller, error)`. All five providers satisfy it; Ollama may return `ErrCapabilityNotSupported` based on the bound model (`ollama/ollama.go:68-75`).

**`llm.Embedder`:** optional. Only OpenAI (`openai/openai.go:17`) and Ollama (`ollama/ollama.go:15`) satisfy it. Anthropic/DeepSeek/MiniMax structurally do not implement it — callers detect with a type assertion.

**`llm.StreamReader`:** interface returned by `Stream`; iterated via `Next() (StreamEvent, error)` until `io.EOF`. Each provider has its own `*<provider>StreamReader` type.

**`llm.StreamEvent`:** the typed union (K1). See K1 verification below.

## K2 Verification — model is bound at construction

For each provider's `New(...)`, the model name flows into `ProviderInfo.Model` and capabilities are computed from that model (or kept as static defaults). The model is immutable for the lifetime of the instance.

| Provider | Constructor | Model bound at | Capabilities computed from model? |
|----------|-------------|----------------|-----------------------------------|
| openai | `openai/options.go:37` `New(opts ...Option) (*OpenAI, error)` | `info.Model = cfg.model` at `openai/options.go:77`; `cfg.model` required (`:42-44`) | **Yes** — `Embeddings` is set by a model-name switch at `openai/options.go:68-72` |
| anthropic | `anthropic/options.go:37` `New(opts ...Option) (*Anthropic, error)` | `info.Model = cfg.model` at `anthropic/options.go:72`; `cfg.model` required (`:42-44`) | Partially — `Embeddings: false` is structural (no `llm.Embedder` impl); other caps are static |
| ollama | `ollama/options.go:48` `New(opts ...Option) (*Ollama, error)` | `info.Model = cfg.model` at `ollama/options.go:102`; `cfg.model` required (`:53-55`) | **Yes (strongest)** — `strategy := strategyForModel(cfg.model)` (`:74`) and `embedDim := embeddingDimensionForModel(cfg.model)` (`:75`); both feed `Capabilities.Tools` and `Capabilities.Embeddings` (`:104-105`) |
| deepseek | `deepseek/options.go:55` `New(opts ...Option) (*DeepSeek, error)` | `info.Model = cfg.model` at `deepseek/options.go:92`; `cfg.model` required (`:60-62`) | Static — all caps fixed |
| minimax | `minimax/options.go:55` `New(opts ...Option) (*MiniMax, error)` | `info.Model = cfg.model` at `minimax/options.go:92`; `cfg.model` required (`:60-62`) | Static — all caps fixed |

The model field is *only* read from `o.info.Model` thereafter (e.g. `openai/map.go:29`, `anthropic/map.go:30`, `ollama/map.go:24`, `deepseek/map.go:29`, `minimax/map.go:30`). It is never overridden on a per-request basis. The `llm.Request` type carries no model field in the request path.

**Caller-visible consequence:** to use a different model, construct a different adapter instance. `Info()` always reports the model that was bound at construction (`openai/openai.go:44`, `anthropic/anthropic.go:44`, `ollama/ollama.go:66`, `deepseek/deepseek.go:43`, `minimax/minimax.go:44`).

**Verdict:** K2 holds across all five providers. Ollama is the strongest example because it makes capability variation per (provider × model) explicit and code-visible.

## K1 Verification — typed stream events with stable per-tool-call Index

The `llm.StreamEvent.Kind` enum (defined upstream in `llm-agent/llm`) is the contract. Each provider must (a) emit `Kind`-tagged events, (b) carry a stable `Index` on `ToolCallDelta` for every event in the same tool call.

| Provider | Stream type | Where `Kind` is set | `Index` stability |
|----------|------------|---------------------|-------------------|
| openai | `openaiStreamReader` (`openai/openai.go:96`) | `chunkEvents` (`openai/openai.go:158-231`): emits `EventTextDelta` (`:179`), `EventToolCallStart` (`:190`), `EventToolCallArgsDelta` (`:200`), `EventToolCallEnd` (`:219`), `EventDone` (`:169`) | Index comes straight from `tool.Index` in the SDK chunk (`:187, :192, :202, :222`). Indices are accumulated in `r.toolIndexes` and sorted before emitting `EventToolCallEnd` (`:212-216`). Stable per call ID across all four events. ✅ |
| anthropic | `anthropicStreamReader` (`anthropic/anthropic.go:57`) | `eventToStreamEvents` (`anthropic/anthropic.go:122-197`): `EventTextDelta` (`:160`), `EventToolCallStart` (`:146`), `EventToolCallArgsDelta` (`:163`), `EventToolCallEnd` (`:178`), `EventThinkingDelta` (`:172`), `EventDone` (`:189`) | Index is `int(v.Index)` from the SDK's content-block events (`:138, :148, :157, :164, :175, :180`). The same block index identifies start → delta → stop, and `blockMeta[idx]` carries the `(id, name)` pair through. ✅ |
| ollama | `ollamaStreamReader` (`ollama/ollama.go:110`) | `ingest` (`ollama/ollama.go:211-327`): emits `EventTextDelta` (`:222, :234, :288, :304, :307`), `EventToolCallStart/ArgsDelta/End` for both native (`:255-259`) and content-parsed (`:297-301`) paths, `EventDone` (`:321-324`) | Index synthesized per-call in arrival order via `r.nextIdx` (`:245-246, :291-292`), stable across Start → ArgsDelta → End. Native tool_calls field on done frame → triple Start/ArgsDelta/End; content-parsed `<tool_call>...</tool_call>` buffered until Done, then markers stripped and tools emitted as triples (see `parseQwenToolCalls`/`parsePythonTagToolCalls`). ✅ **Full K1 compliance since commit 32f5d59 (2026-05-20).** |
| deepseek | `deepseekStreamReader` (`deepseek/deepseek.go:51`) | `chunkEvents` (`deepseek/deepseek.go:113-186`): `EventTextDelta` (`:134`), `EventToolCallStart` (`:145`), `EventToolCallArgsDelta` (`:155`), `EventToolCallEnd` (`:174`), `EventDone` (`:124`) | Same shape as openai — `int(tool.Index)` (`:147, :157, :177`). Indices tracked in `r.toolIndexes` and sorted before `EventToolCallEnd` (`:167-179`). ✅ |
| minimax | `minimaxStreamReader` (`minimax/minimax.go:57`) | `eventToStreamEvents` (`minimax/minimax.go:122-197`): same six event Kinds as anthropic | Same shape as anthropic — `int(v.Index)` for content-block start/delta/stop (`:138, :148, :157, :164, :175, :180`). ✅ |

**Verdict:** K1 GREEN across all five providers (openai/anthropic/ollama/deepseek/minimax). Ollama's stream reader emits the full typed-union including tool-call deltas via two paths: native `tool_calls` field on the done frame, and content-parsed `<tool_call>...</tool_call>` markers in the message body (qwen/llama3.1 strategies). The historical "Partial K1 compliance" classification referred to pre-`32f5d59` (2026-05-20) code state and is now lifted.

The `internal/contract/generate_test.go` `TestStream_Conformance` suite exercises happy-path streaming for all five providers; `TestStreamToolCalls_Conformance` (cross-provider K1 gate) exercises streaming-with-tool-calls across 5/5 providers (6 cases including both ollama paths) and pins per-call Index stability across Start → ArgsDelta(s) → End plus terminal Done with Usage + FinishReason. `TestStream_CancelMidStream_Conformance` and `TestStream_PartialUsageOnError_Conformance` verify the mid-stream cancellation and partial-usage semantics across all five providers.

## `internal/` — Shared Plumbing

Only one internal package: `internal/contract/`.

**Files:**
- `internal/contract/contract.go` (9260 bytes) — fixture loader (`LoadFixture`), mock-server builder (`NewMockServer`), and capability-specific assertions (`AssertGenerate`, `AssertStream`, `AssertToolCalling`, `AssertEmbed`, `assertResponse`). Defines the `Fixture` JSON schema.
- `internal/contract/main_test.go` (122 bytes) — wires `go.uber.org/goleak` into `TestMain` so any test in the package fails on goroutine leaks.
- `internal/contract/generate_test.go` (14652 bytes) — the conformance suite + all factory functions per provider. Drives openai/anthropic/ollama through the shared assertions for generate / stream / tool / embed scenarios.
- `internal/contract/ollama_live_test.go` (1765 bytes, `//go:build ollama_live`) — testcontainer-driven smoke test against a real Ollama image. Excluded from normal CI.
- `internal/contract/testdata/{openai,anthropic,ollama}/*.json` — fixture corpus. Each fixture defines request assertions, response body (sometimes SSE/ndjson), and an `expect` block for the typed result.

**What's NOT shared:** there is no shared HTTP plumbing, no shared JSON helpers, no shared test fakes. Each provider package has its own `wrapErr`, its own `*StreamReader`, its own `toSDKRequest`/`fromSDKResponse`. The only cross-cutting code is the test harness.

The conformance suite currently exercises openai/anthropic/ollama but not deepseek/minimax — the design doc (`docs/superpowers/specs/2026-05-13-deepseek-minimax-design.md:31-33`) explicitly chose fixture-style `httptest` coverage inside each package's own `_test.go` for those two providers (see `deepseek/deepseek_test.go`, `minimax/minimax_test.go`).

## How the `ChatModel` interface is satisfied

Every provider asserts interface satisfaction at file scope with a compile-time `_ = (*T)(nil)` pattern:

```go
// openai/openai.go:14-18
var (
    _ llm.ChatModel  = (*OpenAI)(nil)
    _ llm.ToolCaller = (*OpenAI)(nil)
    _ llm.Embedder   = (*OpenAI)(nil)
)

// anthropic/anthropic.go:13-16
var (
    _ llm.ChatModel  = (*Anthropic)(nil)
    _ llm.ToolCaller = (*Anthropic)(nil)
)

// ollama/ollama.go:12-16
var (
    _ llm.ChatModel  = (*Ollama)(nil)
    _ llm.ToolCaller = (*Ollama)(nil)
    _ llm.Embedder   = (*Ollama)(nil)
)

// deepseek/deepseek.go:14-17
var (
    _ llm.ChatModel  = (*DeepSeek)(nil)
    _ llm.ToolCaller = (*DeepSeek)(nil)
)

// minimax/minimax.go:13-16
var (
    _ llm.ChatModel  = (*MiniMax)(nil)
    _ llm.ToolCaller = (*MiniMax)(nil)
)
```

A consumer that needs `llm.Embedder` must type-assert: `embedder, ok := model.(llm.Embedder)`. See `internal/contract/contract.go:164-173` for the canonical example. Anthropic, DeepSeek, and MiniMax structurally fail that assertion — making the capability gap detectable at runtime without an exception.

## Tool / Function Calling — Normalization vs Divergence

| Aspect | openai | deepseek | anthropic | minimax | ollama |
|--------|--------|----------|-----------|---------|--------|
| Schema mapper | `shared.FunctionParameters` via JSON round-trip (`openai/map.go:46-51`) | identical to openai (`deepseek/map.go:46-51`) | `toToolInputSchema` (`anthropic/map.go:108-145`) | identical to anthropic (`minimax/map.go:108-145`) | `toAPITool` + recursive `toToolProperty` (`ollama/map.go:77-169`) |
| Parallel calls | `ParallelToolCalls = true` (`openai/map.go:40`) | `ParallelToolCalls = true` (`deepseek/map.go:40`) | implicit; multi-block | implicit; multi-block | N/A — single-call surface |
| Response extraction | `sdkToolCalls(c)` reads `Choices[0].Message.ToolCalls` (`openai/map.go:109-130`) | identical (`deepseek/map.go:109-130`) | `block.AsToolUse()` over content blocks (`anthropic/map.go:67-77`) | identical (`minimax/map.go:67-77`) | native first, then `parsePythonTagToolCalls` / `parseQwenToolCalls` (`ollama/tool_strategy.go:50-141`) |
| Streaming tool deltas | yes — `EventToolCall*` keyed on `Index` | yes — same | yes — keyed on block `Index` | yes — same | **no** — text deltas only |

**Pattern:** Provider pairs that share an SDK share their tool path (openai↔deepseek, anthropic↔minimax). Ollama is the unique outlier: native tool calls are surfaced for `Generate` but not the stream; per-model textual fallback parsers fill the gap when the underlying model emits in-band tool markers.

## Entry Points

**Each provider has exactly one entry point**:

| Provider | Entry | Location |
|----------|-------|----------|
| openai | `openai.New(opts ...Option)` | `openai/options.go:37` |
| anthropic | `anthropic.New(opts ...Option)` | `anthropic/options.go:37` |
| ollama | `ollama.New(opts ...Option)` | `ollama/options.go:48` |
| deepseek | `deepseek.New(opts ...Option)` | `deepseek/options.go:55` |
| minimax | `minimax.New(opts ...Option)` | `minimax/options.go:55` |

There is no top-level `providers` aggregator package and no factory dispatch by string name. Consumers import exactly the subpackages they need.

## Architectural Constraints

- **Threading:** Adapters are safe for concurrent `Generate` calls (no shared mutable state on the provider struct outside the SDK client, which the SDKs document as concurrency-safe). Stream readers are NOT safe for concurrent `Next()` calls from multiple goroutines (each reader has its own `sync.Mutex`, but the contract is single-consumer).
- **Goroutines:** Only Ollama's stream spawns a goroutine to bridge the SDK's callback interface to the `Next()`-based reader (`ollama/ollama.go:48-62`). The other readers are synchronous around the SDK's `ssestream`. `go.uber.org/goleak` in `internal/contract/main_test.go:9` guarantees no leaks slip through CI.
- **Global state:** None. There is no module-level singleton or package-level mutable state. Every adapter holds its own SDK client.
- **Circular imports:** None — providers only depend on `llm-agent/llm` and their respective SDKs.
- **No internal retry/backoff:** Every SDK is constructed with `option.WithMaxRetries(0)`. The caller (typically the `llm-agent` runtime) is responsible for retry semantics, informed by the `*llm.TransientError` / `*llm.RateLimitError` sentinels.
- **`MaxTokens` default:** Anthropic + MiniMax force `MaxTokens=1024` if the request doesn't specify (`anthropic/map.go:31`, `minimax/map.go:31`) — the upstream API requires a non-zero value.

## Anti-Patterns

### Implementing a capability you don't actually support

**What happens:** A provider could satisfy `llm.Embedder` by returning a stub `[]llm.Vector` — making capability detection lie.

**Why it's wrong in this repo:** It would break the K2 contract that `Info().Capabilities` reflects reality. Callers branch on it.

**Do this instead:** Refuse to implement the interface at all. Anthropic does this — `anthropic/anthropic.go:13-16` lists only `ChatModel` and `ToolCaller`. A `model.(llm.Embedder)` assertion returns `ok=false`, which is the correct signal. Where the interface IS satisfied but the bound model can't support it, return `llm.ErrCapabilityNotSupported` from the call (see `ollama/embed_strategy.go:26-28`, `ollama/ollama.go:78-80`).

### Per-request model override

**What happens:** Reading a model from `llm.Request` and passing it to the SDK on every call.

**Why it's wrong in this repo:** Breaks K2 — `Info()` would no longer reflect the model actually used, and capability decisions made by the caller (e.g. "this adapter supports tools") would race against the per-request choice.

**Do this instead:** Use `o.info.Model` everywhere (already the convention — `openai/map.go:29`, `anthropic/map.go:30`, `ollama/map.go:24`, `deepseek/map.go:29`, `minimax/map.go:30`). If a caller needs a different model, they construct a new adapter.

### SDK-level retries

**What happens:** Leaving the SDK default retry budget in place.

**Why it's wrong in this repo:** Hides `RateLimitError`/`TransientError` from the caller, defeats backoff coordination across the agent, and conflicts with the K1 stream-error semantics (which only retry once internally, only before the first byte).

**Do this instead:** Always `option.WithMaxRetries(0)`. Pattern is verified in `openai/options.go:50`, `anthropic/options.go:50`, `deepseek/options.go:73`, `minimax/options.go:73`. (Ollama's SDK has no retry knob.)

### Emitting untyped stream chunks

**What happens:** Leaking the SDK's native chunk type (e.g. `openai.ChatCompletionChunk`) to the caller.

**Why it's wrong in this repo:** Breaks K1 — the typed-union contract is the entire point of `llm.StreamEvent.Kind`.

**Do this instead:** Translate every SDK event into one or more `llm.StreamEvent` values inside `chunkEvents`/`eventToStreamEvents` and queue them. Every reader follows this pattern (`openai/openai.go:158-231`, `anthropic/anthropic.go:122-197`, etc.).

## Error Handling

**Strategy:** Map provider-specific HTTP/SDK errors into the four `llm-agent` sentinel error types: `AuthError`, `RateLimitError`, `InvalidRequestError`, `TransientError`. `context.Canceled` is returned as-is. `context.DeadlineExceeded` becomes `*llm.TransientError`. Capability gaps wrap `llm.ErrCapabilityNotSupported`.

**Patterns:**
- Each provider owns its own `wrapErr` in `*/errors.go`. No shared helper.
- Status extraction uses `errors.As(err, &<sdk.Error>)`. Ollama additionally uses an atomic-int32 last-status because the Ollama SDK doesn't always carry status on the error.
- `Retry-After` parsing: OpenAI, DeepSeek, Anthropic, and MiniMax (P1-9, 2026-05-22). Ollama has no `http.Response` on its `api.StatusError` value type, so it remains unparsed — tracked as a follow-up.
- `internal/contract/generate_test.go:355-380` `TestErrorString_NoSecretLeak` asserts that the wrapped error chain preserves `errors.Is` traversal to the inner error (so callers can still inspect it) but does not assert redaction is performed.

## Cross-Cutting Concerns

**Logging:** None. Adapters do not log; the caller observes via errors and stream events.

**Validation:** Minimal — only `cfg.model == ""` triggers a constructor error. Tool schemas are best-effort decoded (`toToolInputSchema`, `toAPITool`) and silently default to `type=object` on parse failure (`anthropic/map.go:108-145`, `ollama/map.go:77-115`).

**Authentication:** Each provider has its own `WithAPIKey(...)` + matching env-var fallback. No shared credential layer.

**Concurrency:** Stream readers are mutex-guarded for `Next()`/`Close()`. Sync `Generate` calls are safe to fan out per the SDK contracts.

**Observability:** None in-repo. The umbrella has a separate `llm-agent-otel` repo for OpenTelemetry integration (README links to it).

---

*Architecture analysis: 2026-05-20*
