# Coding Conventions

**Analysis Date:** 2026-05-20
**Scope:** `github.com/costa92/llm-agent-providers` — five sibling adapter packages plus `internal/contract`.

## Per-provider package layout (must-match table)

Every public provider package follows the same five-file skeleton. The table below is the contract — any deviation is a smell.

| File | openai | anthropic | ollama | deepseek | minimax | Purpose |
|------|--------|-----------|--------|----------|---------|---------|
| `doc.go` | yes | yes | yes | yes | yes | Package doc comment only |
| `<provider>.go` | `openai.go` | `anthropic.go` | `ollama.go` | `deepseek.go` | `minimax.go` | Struct, `Generate`, `Stream`, `Info`, `WithTools`, optional `Embed`, plus the `<provider>StreamReader` |
| `options.go` | yes | yes | yes | yes | yes | `type config struct`, `type Option func(*config)`, `WithX` functional options, `New(opts ...Option)` |
| `map.go` | yes | yes | yes | yes | yes | `toSDKRequest`, `fromSDKResponse`, `mapFinishReason` / `mapXStopReason`, tool-schema translation |
| `errors.go` | yes | yes | yes | yes | yes | `wrapErr(err) error` → typed `llm.*` errors |

There is **no `chatmodel.go`** — the convention is `<provider>.go` as the implementation entry point. Ollama additionally carries `tool_strategy.go` and `embed_strategy.go` for its per-model strategy table; that is an Ollama-specific extension, not the norm.

**File-naming convention:** lowercase, no underscores except the ollama `*_strategy.go` files. Test files mirror the implementation: `<provider>_test.go`. Doc-only package comment lives in `doc.go`.

## Constructor naming

**One pattern across all five providers:** `New(opts ...Option) (*Type, error)`.

Examples (look for shape, not noise):
- `openai/options.go:37` — `func New(opts ...Option) (*OpenAI, error)`
- `anthropic/options.go:37` — `func New(opts ...Option) (*Anthropic, error)`
- `ollama/options.go:48` — `func New(opts ...Option) (*Ollama, error)`
- `deepseek/options.go:55` — `func New(opts ...Option) (*DeepSeek, error)`
- `minimax/options.go:55` — `func New(opts ...Option) (*MiniMax, error)`

There is **no** `NewFromConfig`, `NewWithModel`, or `MustNew` variant. Adding one would break the parity these adapters depend on.

`New()` is the only top-level exported constructor per package. Every other public helper sits on the returned value (`Info`, `WithTools`, `Generate`, `Stream`, `Embed`, `EmbedDimensions`).

## Options pattern

**Functional options over config structs.** The internal `type config struct` is unexported; the user-facing API is `WithX` functions returning an `Option`.

Required option: `WithModel(string)`. If absent, `New` returns `errors.New("<provider>: WithModel is required")`. Identical wording across all five — keep it that way.

Standard set every provider exposes:

| Option | openai | anthropic | ollama | deepseek | minimax |
|--------|--------|-----------|--------|----------|---------|
| `WithModel` | yes | yes | yes | yes | yes |
| `WithAPIKey` | yes | yes | n/a (local daemon) | yes | yes |
| `WithBaseURL` | yes | yes | yes (alias `WithHost`) | yes | yes |
| `WithHTTPClient` | yes | yes | yes | yes | yes |
| `WithTimeout` | yes | yes | yes | yes | yes |
| Provider-unique | `WithOrganization` | `WithBetaHeader` | — | `WithRegion(RegionCN\|RegionGlobal)` | `WithRegion(RegionCN\|RegionGlobal)` |

When you add a new option, follow the established style:

```go
func WithX(x T) Option { return func(c *config) { c.x = x } }
```

One line, single-purpose, no validation inside the closure (validation happens inside `New`).

## API-key resolution

Pattern is identical and unconditional: if `WithAPIKey` is not supplied, `New` falls back to `os.Getenv("<PROVIDER>_API_KEY")`. Evidence:

- `openai/options.go:45-47` — `OPENAI_API_KEY`
- `anthropic/options.go:45-47` — `ANTHROPIC_API_KEY`
- `deepseek/options.go:63-65` — `DEEPSEEK_API_KEY`
- `minimax/options.go:63-65` — `MINIMAX_API_KEY`
- `ollama/options.go:58-63` — `OLLAMA_HOST` (URL, not a key)

No provider logs the value, prints it, or reflects it back in errors. Don't add code that does.

## Capability declarations (K2)

`llm.ProviderInfo.Capabilities` is populated inside `New`, after the model string is known. This binds capability to the per-(provider × model) tuple required by K2.

| Provider | Tools | Embeddings | StructuredOutputs | PromptCaching | Source |
|----------|-------|-----------|-------------------|---------------|--------|
| openai | true | model-derived (`text-embedding-3-*`, `text-embedding-ada-002` → true) | false | false | `openai/options.go:68-83` |
| anthropic | true | false | false | false | `anthropic/options.go:73-79` |
| ollama | model-derived (`strategyForModel.supportsTool`) | model-derived (`embeddingDimensionForModel > 0`) | false | false | `ollama/options.go:74-109` |
| deepseek | true | false | false | false | `deepseek/options.go:93-98` |
| minimax | true | false | false | false | `minimax/options.go:93-98` |

Adding a capability means: (1) update the `Capabilities` literal at `New`, (2) implement the matching method, (3) add a `var _ llm.<Cap> = (*Provider)(nil)` interface assertion at the top of `<provider>.go`.

Interface assertion block lives at the top of every `<provider>.go`:

```go
var (
    _ llm.ChatModel  = (*OpenAI)(nil)
    _ llm.ToolCaller = (*OpenAI)(nil)
    _ llm.Embedder   = (*OpenAI)(nil)
)
```

Evidence: `openai/openai.go:14-18`, `anthropic/anthropic.go:13-16`, `ollama/ollama.go:12-16`, `deepseek/deepseek.go:14-17`, `minimax/minimax.go:13-16`.

## Error handling

**Three layers, no other choices.**

1. **Sentinel-wrapping with `%w`** for capability gaps (`ollama/tool_strategy.go:162`, `ollama/embed_strategy.go:27`):

   ```go
   return fmt.Errorf("ollama: model %s native tools unavailable (ProviderInfo.Capabilities.Tools=false): %w",
       model, llm.ErrCapabilityNotSupported)
   ```

2. **Typed `llm.*Error{Provider, Wrapped}` values** for transport errors, produced by `wrapErr(err)`. Status-code routing is identical across providers:
   - 401/403 → `*llm.AuthError`
   - 429 → `*llm.RateLimitError` (openai/deepseek also extract `Retry-After` + `quota_exhausted`)
   - 500/502/503/504 (and 529 for anthropic/minimax) → `*llm.TransientError`
   - other 4xx → `*llm.InvalidRequestError`
   - `context.Canceled` → returned verbatim (do not wrap)
   - `context.DeadlineExceeded` → `*llm.TransientError`
   - `net.Error` fallback → `*llm.TransientError`

   See `openai/errors.go:12-57`, `anthropic/errors.go:14-48`, `ollama/errors.go:14-56`, `deepseek/errors.go:12-57`, `minimax/errors.go:12-46`.

3. **Plain `errors.New("<provider>: WithModel is required")`** for required-option violations from `New`.

There are **no swallowed errors** in production. `grep -rn '_ = err\|nolint:errcheck'` on non-test files returns zero. The only deliberate `_ =` is `_ = r.stream.Close()` inside stream readers (`openai/openai.go:129`, `deepseek/deepseek.go:84`, `anthropic/anthropic.go:93`, `minimax/minimax.go:93`) — acceptable because we already captured `r.stream.Err()` above it.

**There are no `panic(` sites** anywhere in the repo (verified by grep). Don't add them; return typed errors instead.

## Context handling

`ctx context.Context` is the first parameter of every public IO method and is always forwarded to the SDK call without rewrapping. Spot checks:

- `openai/openai.go:28` — `o.client.Chat.Completions.New(ctx, sdkReq)`
- `anthropic/anthropic.go:26` — `a.client.Messages.New(ctx, sdkReq)`
- `ollama/ollama.go:29` — `o.client.Chat(ctx, sdkReq, cb)`
- `deepseek/deepseek.go:27` — `d.client.Chat.Completions.New(ctx, sdkReq)`
- `minimax/minimax.go:26` — `m.client.Messages.New(ctx, sdkReq)`
- `openai/openai.go:56` — `o.client.Embeddings.New(ctx, ...)`

Ollama is the one provider that **derives** a context (`ctx, cancel := context.WithCancel(ctx)` in `ollama/ollama.go:40`) because its SDK uses a callback model — the cancel is wired to the `ollamaStreamReader.Close` path (`ollama/ollama.go:189-198`). When you touch the Ollama stream path, preserve this; otherwise `Close()` becomes a leak.

Nowhere in production code does an adapter use `context.Background()` — only test files do.

## Doc comments on exported types

Coverage is **uneven**.

- `openai/doc.go` — three-paragraph package doc, but stale ("intentionally reports all optional capabilities as false in Phase 1", "Streaming is deferred to Phase 2"). Both claims contradict current code.
- `anthropic/doc.go` — similar stale "Phase 1" wording but accurate about the system-prompt lifting trick.
- `ollama/doc.go` — same Phase-1 staleness.
- `deepseek/doc.go` — single accurate sentence.
- `minimax/doc.go` — single accurate sentence.

**Exported types** (`OpenAI`, `Anthropic`, `Ollama`, `DeepSeek`, `MiniMax`, `Option`, `Region`, `RegionCN`, `RegionGlobal`) have **no per-type doc comments**. Functions `New`, `WithModel`, etc. have no doc comments either. Only the inline comment at `anthropic/errors.go:12-13` ("Q2 path A:…") exists, and it's design-rationale, not API doc.

When you touch these files, add `// XYZ does …` doc comments per Go style. The current state would fail `golangci-lint run --enable=golint,godot,godoc`.

## Internal shared helpers

`internal/contract/` is the **only** internal package. It is test-only — `internal/contract/contract.go` exports `Fixture`, `LoadFixture`, `NewMockServer`, `AssertGenerate`, `AssertStream`, `AssertToolCalling`, `AssertEmbed`, plus the `ChatModelFactory` type. Adapters never import it; only `internal/contract/generate_test.go` does (and it imports `openai`, `anthropic`, `ollama` — not `deepseek` or `minimax`, see CONCERNS.md).

**There is no shared mocking helper for production code** — each `<provider>_test.go` builds its own `httptest.NewServer` inline.

## SDK boundary discipline

This is the strongest convention in the repo and deserves explicit policy.

**Rule:** SDK types must not appear on a provider's public surface.

How it is currently enforced:

- The SDK client lives on an **unexported** struct field (`client *openai.Client`, `client *sdk.Client`, `client *api.Client`).
- All public methods take/return `llm.*` types only: `llm.Request`, `llm.Response`, `llm.StreamEvent`, `llm.ToolCall`, `llm.ProviderInfo`, `llm.Vector`, `llm.Usage`.
- SDK request shaping lives in `<provider>/map.go` (`toSDKRequest`, `fromSDKResponse`).
- SDK error decoding lives in `<provider>/errors.go` and converts to `llm.AuthError | llm.RateLimitError | llm.TransientError | llm.InvalidRequestError`.

Pure-static check: `grep -rn 'openai\.\|sdk\.\|api\.' <provider>/<provider>.go` should only match `openai.Client`, `sdk.Client`, `api.Client`, `*ssestream.Stream[...]`. Anything else on an exported method signature is a violation.

DeepSeek and MiniMax intentionally reuse foreign SDKs (openai-go and anthropic-sdk-go respectively) because the upstream APIs are wire-compatible. This is fine **as long as** the SDK type leakage rule above holds — and it does.

## Versioning

- Repository tag (`git describe`): `v0.2.1` (latest), with `v0.2.0`, `v0.1.1`, `v0.1.0` historically. Single repo-wide tag.
- **No per-provider tags** — despite the README at line 24-28 suggesting `go get github.com/costa92/llm-agent-providers/openai@v0.1.0` (which would imply submodule tags), there is no `openai/go.mod` or per-package versioning. Subpath @-import resolves against the single top-level tag.
- **No `CHANGELOG.md`** at any level — `find . -name 'CHANGELOG*'` returns nothing.
- README references `llm-agent v0.4.0` but `go.mod` pins `v0.5.1`. See CONCERNS.md.

When tagging, follow `vMAJOR.MINOR.PATCH` repo-wide and let `release/**` branches trigger `release-precheck.yml` (which rejects `replace` directives).

## License headers in source files

**No per-file license headers.** Only the top-level `LICENSE` file (MIT, 2026 costa92). All `*.go` files start directly with `package <name>`.

This is consistent and intentional. Don't introduce per-file headers — they'd be churn and the LICENSE file plus go-module path is sufficient under MIT.

## Naming conventions inside packages

- **Receivers** — single letter matching the type's first letter: `o *OpenAI`, `a *Anthropic`, `o *Ollama`, `d *DeepSeek`, `m *MiniMax`, `r *<provider>StreamReader`. Don't use `self` or `this`.
- **Stream readers** — `<provider>StreamReader` struct, lowercase package-private.
- **SDK-translation functions** — verb form: `toSDKRequest`, `toSDKStreamRequest`, `fromSDKResponse`, `mapFinishReason`, `mapAnthropicStopReason`.
- **Helpers in `tool_strategy.go`** — verb form: `strategyForModel`, `parseResponseToolCalls`, `mapNativeToolCalls`, `parsePythonTagToolCalls`, `parseQwenToolCalls`, `decodeFallbackToolCall`, `unsupportedToolError`.
- **Constants** — exported region presets follow `Region<Suffix>` pattern: `RegionCN`, `RegionGlobal`. New regions follow the same shape.

## Imports

Imports use **stdlib first, then third-party blank-line separator, then `llm-agent` blank-line separator**. Spot-check `openai/options.go:3-12`:

```go
import (
    "errors"
    "net/http"
    "os"
    "time"

    "github.com/costa92/llm-agent/llm"
    openai "github.com/openai/openai-go/v3"
    "github.com/openai/openai-go/v3/option"
)
```

Aliased SDK imports use `sdk` (anthropic, minimax) or the project name (`openai`, `api`). When an SDK ships a name that collides with the local package, alias it; don't dot-import.

## Stream-reader implementation style (K1 contract)

Every adapter declares a `<provider>StreamReader` that implements `llm.StreamReader` (`Next() (llm.StreamEvent, error)` + `Close() error`). The shared template:

1. `sync.Mutex` guards the queue and stream pointer.
2. Lazily open the upstream stream on first `Next()` call (deferred until the consumer pulls). Lets callers `Close()` without ever opening.
3. Maintain a `queue []llm.StreamEvent`; one upstream chunk can decompose into many `llm.StreamEvent`s (e.g. multiple tool starts in one OpenAI chunk).
4. Track `deliveredByte` and `retried` so the reader will retry exactly once **before** any byte is delivered, never after (the spec'd K1 retry rule).
5. On the first byte of a tool block, emit `EventToolCallStart` with `Index`. On every argument delta, emit `EventToolCallArgsDelta` with the **same** Index. On stop, emit `EventToolCallEnd` with the same Index.
6. Final event before EOF: `EventDone` carrying `Usage` and `FinishReason`.

When adding a provider, copy this skeleton (openai or anthropic are the cleanest sources) and resist the urge to "improve" the locking — the lock is shaped to allow concurrent `Close()` from another goroutine to win the race.

---

*Convention analysis: 2026-05-20*
