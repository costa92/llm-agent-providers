# Testing Patterns

**Analysis Date:** 2026-05-20
**Scope:** `github.com/costa92/llm-agent-providers` — five provider packages + `internal/contract`.

## Test framework

- **Runner:** stdlib `testing` + `net/http/httptest`. No third-party assertion library on the import surface (stretchr/testify is a transitive dep only).
- **Goroutine-leak guard:** `go.uber.org/goleak v1.3.0` via `internal/contract/main_test.go:1-11` (`goleak.VerifyTestMain(m)`). Only the contract package wires it up — none of the five provider `*_test.go` files do. This is a coverage gap (see CONCERNS.md).
- **Live-API gate:** `//go:build ollama_live` tag on `internal/contract/ollama_live_test.go:1`, paired with the `nightly-ollama-live.yml` workflow (`cron: '0 3 * * *'`).
- **Run commands** (`.github/workflows/test.yml:39-44`):

  ```bash
  go vet ./...
  go build ./...
  go test ./...                                            # PR / push to main
  go test -tags ollama_live -run TestGenerate_Ollama_Live ./internal/contract/...   # nightly
  ```

## Per-provider test count (the 8-files-for-36-source-files audit)

Source-vs-test-file counts and `func Test...` totals per file:

| Package | Source `.go` files | Test files | Test funcs | Source LOC | Test LOC |
|---------|-------------------|------------|-----------|-----------|----------|
| openai | 5 (`openai.go`, `options.go`, `map.go`, `errors.go`, `doc.go`) | 1 (`openai_test.go`) | 17 | 515 | 590 |
| anthropic | 5 | 1 (`anthropic_test.go`) | 15 | 478 | 547 |
| ollama | 7 (incl. `tool_strategy.go`, `embed_strategy.go`) | 1 (`ollama_test.go`) | 19 | 624 | 485 |
| deepseek | 5 | 1 (`deepseek_test.go`) | 16 | 477 | 508 |
| minimax | 5 | 1 (`minimax_test.go`) | 17 | 537 | 569 |
| `internal/contract` | 1 (`contract.go`) | 3 (`main_test.go`, `generate_test.go`, `ollama_live_test.go`) | 10 + 1 main + 1 live | 282 | 545 |
| **Total** | **27 non-test (+ 9 doc/secondary)** | **8 test** | **94** | **3 113** | **3 244** |

The "36 source / 8 test" claim is correct but slightly misleading: every provider has exactly **one** test file with **15-19 test functions** in it, plus three test files inside `internal/contract`. The ratio is "one test file per package", not "8 test files cover everything". Test-LOC is roughly **1:1** with source-LOC.

**Imbalance:**

- **deepseek** and **minimax** are NOT exercised by the cross-provider conformance suite (`internal/contract/generate_test.go:20-41`). They have only their own `_test.go`.
- **openai/deepseek** share the same SSE chunk-events implementation but the deepseek tests are **near-clones** of openai's — copy/maintain risk.
- **anthropic/minimax** show the same parallel-clone pattern.

## What is actually tested

Every provider test file is built around `httptest.NewServer(http.HandlerFunc(...))` mocks that hand-craft the expected SSE / JSON / NDJSON wire bytes. Spot evidence:

- OpenAI streaming SSE chunks: `openai/openai_test.go:336-339`
- Anthropic SSE event stream: `anthropic/anthropic_test.go:154-163`
- Ollama NDJSON callback: `ollama/ollama_test.go:145-147`
- DeepSeek shares the OpenAI SSE shape: `deepseek/deepseek_test.go:268-...`
- MiniMax shares the Anthropic SSE shape: `minimax/minimax_test.go:222-232`

**No provider uses a recorded-fixture replayer** for the per-package tests. The shared contract suite uses JSON fixtures stored in `internal/contract/testdata/<provider>/*.json`, which are produced offline by the capture scripts (`scripts/capture-fixtures-{openai,anthropic,ollama}.sh`).

**No mocking framework.** No `gomock`, `moq`, `mockgen`, or `testify/mock`. Just hand-rolled HTTP test servers.

## Live-API / integration tests

**Exactly one live test:** `internal/contract/ollama_live_test.go:17` (`TestGenerate_Ollama_Live`). It spins up an Ollama testcontainer, `ollama pull`s `llama3.1:8b-instruct-q4_K_M`, and runs the shared `AssertGenerate` against it with usage/text assertions relaxed.

**Gating:**
- Build tag `ollama_live` (will not run in plain `go test ./...`).
- Nightly schedule: `.github/workflows/nightly-ollama-live.yml:6-8`.
- Skipped in `-short` mode (`ollama_live_test.go:18-20`).
- Requires Docker on the runner.

**There is no live test for openai, anthropic, deepseek, or minimax** — all four are tested only against mock SSE.

## Streaming tests — K1 contract coverage

The K1 contract is "the stream produces a typed union of `StreamEvent` kinds, with a stable per-tool-call `Index` linking Start → ArgsDelta → End → Done". Coverage map:

| Provider | Happy stream | Tool-call streaming with Index | Retry-before-byte | No-retry-after-byte | Cancel mid-stream | Partial usage on error |
|----------|--------------|--------------------------------|-------------------|---------------------|-------------------|------------------------|
| openai | `openai_test.go:201 TestStream_OpenAI_Happy` | `openai_test.go:333 TestStream_OpenAI_ToolCalls` (asserts `Index 0/1`, paired Start/End at 380-391) | `openai_test.go:251 TestStream_OpenAI_RetriesBeforeFirstByte` | `openai_test.go:292 TestStream_OpenAI_DoesNotRetryAfterFirstByte` | `contract/generate_test.go:259 TestStream_CancelMidStream_Conformance` | `contract/generate_test.go:310 TestStream_PartialUsageOnError_Conformance` |
| anthropic | `anthropic_test.go:153 TestStream_Anthropic_Happy` | `anthropic_test.go:196 TestStream_Anthropic_PartialJSONFlushesOnContentBlockStop` (asserts `Index 1`) | not covered | `anthropic_test.go:274 TestStream_Anthropic_DoesNotRetryAfterFirstByte` | `contract/generate_test.go:259` (anthropic case) | `contract/generate_test.go:310` (anthropic case) |
| ollama | `ollama_test.go:139 TestStream_Ollama_Happy` | not covered — Ollama stream emits only `TextDelta`/`Done`; tool calls flow through `Generate` only (`ollama.go:166-173`) | not covered | not covered | `ollama_test.go:181 TestStream_Ollama_CancelMidStream` + `contract` peer | `contract/generate_test.go:310` (ollama case) |
| deepseek | `deepseek_test.go:268 TestStream_DeepSeek_Happy` | `deepseek_test.go:359 TestStream_DeepSeek_ToolCalls` (asserts `Index 0/1` at 406-415) | `deepseek_test.go:318 TestStream_DeepSeek_RetriesBeforeFirstByte` | not covered explicitly | not covered (deepseek absent from contract suite) | not covered (deepseek absent from contract suite) |
| minimax | `minimax_test.go:177 TestStream_MiniMax_Happy` | `minimax_test.go:220 TestStream_MiniMax_PartialJSONFlushesOnContentBlockStop` (asserts `Index 1` at 261-285) | not covered | `minimax_test.go:298 TestStream_MiniMax_DoesNotRetryAfterFirstByte` | not covered (minimax absent from contract suite) | not covered (minimax absent from contract suite) |

**K1 score:**
- openai — **GREEN**: Index, retry, cancel, partial-usage all tested at the package level and the shared contract suite.
- anthropic — **GREEN**: Index via tool-block plumbing, no-retry-after-byte at package level, cancel+partial via contract.
- ollama — **YELLOW**: stream is text-only; tool calls are NOT exercised through the stream path because the Ollama SDK does not surface tool deltas mid-stream (`ollama/ollama.go:120-187` confirms — `EventToolCall*` is never emitted by the reader). Functionally correct for the Ollama wire format, but it means there is **no `Index` assertion** for Ollama.
- deepseek — **YELLOW**: Index is asserted in `TestStream_DeepSeek_ToolCalls`, but the provider is excluded from the cross-provider conformance, cancel, and partial-usage tests.
- minimax — **YELLOW**: same as deepseek — Index asserted in `TestStream_MiniMax_PartialJSONFlushesOnContentBlockStop` but absent from the cross-provider suite.

## Capability reporting (K2) coverage

The K2 contract is "constructor binds a model, and `Info().Capabilities` reflects that exact model — not a provider-wide flag".

| Check | Test | File |
|-------|------|------|
| `New()` rejects missing model | `TestNew_RequiresModel` in **every** provider | `openai_test.go:18`, `anthropic_test.go:18`, `ollama_test.go:19`, `deepseek_test.go:18`, `minimax_test.go:18` |
| `Info().Provider`/`Model` reflect constructor input | `TestInfo_<Provider>` in every provider | `openai_test.go:28`, `anthropic_test.go:28`, `ollama_test.go:29`, `deepseek_test.go:28`, `minimax_test.go:28` |
| Embedding-model variant flips `Capabilities.Embeddings` | `TestInfo_OpenAI_EmbeddingModel` (`openai_test.go:45`), `TestInfo_Ollama_EmbeddingModel` (`ollama_test.go:57`) | OpenAI + Ollama only |
| Tool-capable Ollama model flips `Capabilities.Tools` | `TestInfo_Ollama_Qwen25CoderSupportsTools` | `ollama_test.go:46` |
| Tool-incapable Ollama model | `TestInfo_Ollama_UnsupportedModel` (`ollama_test.go:128`), `TestEmbed_Ollama_UnsupportedModel` (`ollama_test.go:114`) | Ollama |
| `WithTools` on incapable model returns `ErrCapabilityNotSupported` | `TestWithTools_Ollama_UnsupportedModel` (`ollama_test.go:280`), `TestToolCalling_CapabilityDegrade_Ollama` + `TestToolCalling_UnsupportedErrorSentinel` (`contract/generate_test.go:159, 244`) | Ollama |
| Anthropic explicitly NOT an `llm.Embedder` | `TestAnthropic_DocumentedEmbeddingGap` (`anthropic_test.go:45-57`) — uses `_, ok := any(m).(llm.Embedder); ok { t.Fatal(...) }` | Anthropic |
| MiniMax explicitly NOT an `llm.Embedder` | `TestInfo_MiniMax` (`minimax_test.go:54-55`) — same interface negative-assert | MiniMax |

**K2 score:**
- openai — **GREEN**: model→capability mapping fully tested for chat-vs-embed variants.
- anthropic — **GREEN**: capability is constant true/false but the negative-assert on `llm.Embedder` makes K2 explicit.
- ollama — **GREEN**: this is the K2 reference implementation — Tools and Embeddings both flip on model name.
- deepseek — **YELLOW**: `TestInfo_DeepSeek` covers the happy case, but there is **no test that confirms different deepseek models produce different `Capabilities`** because the adapter pins all of them to `{Tools:true, others:false}`. If DeepSeek adds an embeddings endpoint, the contract will be broken silently.
- minimax — **YELLOW**: same gap as deepseek.

## Tool / function-calling tests

Per provider, two layers:

| Layer | openai | anthropic | ollama | deepseek | minimax |
|-------|--------|-----------|--------|----------|---------|
| Generate-with-tools (sync) | `TestGenerate_OpenAI_ToolCalls` (452) | `TestGenerate_Anthropic_MultiBlockToolUse` (371) | `TestWithTools_Ollama_Llama31NativeToolCalls` (297), `TestWithTools_Ollama_QwenXMLFallback` (331), `TestWithTools_Ollama_Qwen25BareJSONFallback` (361) | `TestGenerate_DeepSeek_ToolCalls` (124) | `TestGenerate_MiniMax_MultiBlockToolUse` (395) |
| Stream tool deltas | `TestStream_OpenAI_ToolCalls` (333) | covered inside `TestStream_Anthropic_PartialJSONFlushesOnContentBlockStop` (196) | NOT covered (Ollama stream emits no tool events) | `TestStream_DeepSeek_ToolCalls` (359) | covered inside `TestStream_MiniMax_PartialJSONFlushesOnContentBlockStop` (220) |
| `WithTools` independence (binding-doesn't-leak-across-instances) | `TestWithTools_OpenAI_ImmutableAndIndependent` (110) | `TestWithTools_Anthropic_ImmutableAndRequestShape` (59) | not as a direct test, covered via tool happy-path | `TestWithTools_DeepSeek_ImmutableAndIndependent` (177) | `TestWithTools_MiniMax_ImmutableAndRequestShape` (83) |
| Cross-provider dedupe by `messageID+call.ID` | `contract/generate_test.go:213 TestToolCalling_DedupeKey` | — (shared) | — | not exercised | not exercised |

Ollama has the deepest per-model coverage because it has **three** tool-encoding strategies (native, python_tag for llama3.1, qwen-XML / qwen-bare-JSON for qwen2.5/3-coder). See `ollama/tool_strategy.go:50-63` and the three `TestWithTools_Ollama_*` variants.

## Embedding tests

Provider-side `Embed` is implemented only by openai and ollama (`var _ llm.Embedder = (*OpenAI)(nil)`, `var _ llm.Embedder = (*Ollama)(nil)`):

| Test | File |
|------|------|
| OpenAI happy embed | `openai_test.go:59 TestEmbed_OpenAI_Happy` |
| Ollama happy embed | `ollama_test.go:71 TestEmbed_Ollama_Happy` |
| Ollama unsupported-model embed returns capability sentinel | `ollama_test.go:114 TestEmbed_Ollama_UnsupportedModel` |
| Anthropic NOT implementing `llm.Embedder` | `anthropic_test.go:45 TestAnthropic_DocumentedEmbeddingGap` |
| MiniMax NOT implementing `llm.Embedder` | `minimax_test.go:53-56` inside `TestInfo_MiniMax` |
| Cross-provider via contract | `contract/generate_test.go:179 TestEmbed_Conformance` covers openai/ollama happy + anthropic gap (no deepseek, no minimax) |

## Error-path tests (typed `llm.*Error`)

Every provider exercises 401, 403/429, 500, and one provider-specific code (529 for anthropic/minimax; 404 for ollama). All assert with `errors.As(err, &llm.AuthError{})` style.

Spot evidence:
- `openai_test.go:505-571`: 401, 403, 429-quota, 429, 500, 404
- `anthropic_test.go:489-525`: 401, 400, 429, 529, 500
- `ollama_test.go:401-470`: 404 model-not-pulled, 401, 500, no-daemon, 400
- `deepseek_test.go:423-487`: 401, 403, 429-quota, 429, 500, 404
- `minimax_test.go:513-547`: 401, 400, 429, 529, 500
- `contract/generate_test.go:355 TestErrorString_NoSecretLeak`: verifies typed errors preserve `errors.Is` chain.

Conformance equivalents live in `contract/generate_test.go:66 TestGenerate_Conformance` (openai/anthropic/ollama × all status codes).

## Fakes / shared HTTP transport

**There is no shared fake HTTP transport.** Each test file builds an `httptest.NewServer` on demand. The `internal/contract` package exposes `NewMockServer(t, Fixture)` (`internal/contract/contract.go:66`) but that is a fixture-driven assertion helper, not a transport — it builds an `httptest.NewServer` internally.

There is one custom `http.RoundTripper` in production code: `ollama/options.go:35-46` `statusCapturingTransport`. It exists to recover the HTTP status code that the Ollama SDK swallows; it is wired into the user-supplied or default `*http.Client` inside `New`.

## Provider-parity / cross-provider test

**The closest thing to a parity suite** lives in `internal/contract/generate_test.go`:

- `var AdapterFactories = map[string]ChatModelFactory{"openai": ..., "anthropic": ..., "ollama": ...}` (lines 20-41) — **deepseek and minimax are missing here**.
- `TestGenerate_Conformance` (66), `TestStream_Conformance` (103), `TestToolCalling_Conformance` (130), `TestEmbed_Conformance` (179), `TestStream_CancelMidStream_Conformance` (259), `TestStream_PartialUsageOnError_Conformance` (310) all iterate over those factories.

The fixture testdata mirror confirms this: `internal/contract/testdata/{openai,anthropic,ollama}` exist; `testdata/deepseek` and `testdata/minimax` do NOT.

So the answer to "is there a test that runs the same suite against all 5 providers?" is **no — only 3 of 5**. See CONCERNS.md.

## Gaps — riskiest untested paths

Ranked by blast-radius:

1. **DeepSeek and MiniMax have zero conformance / parity / cancel-mid-stream / partial-usage-on-error tests.** They are tested **only** against bespoke mocks inside their own `_test.go`. A regression in `wrapErr` status routing or in stream retry-once semantics will not be caught by the conformance gate.
2. **Ollama stream tool calls are not exercised.** `ollama_test.go:139-179` only exercises text deltas. `ollama/ollama.go:120-187` confirms the Ollama stream reader emits no `EventToolCall*` events at all — tool calls only come back via `Generate`. If the upstream Ollama wire format adds streamed tool deltas, our adapter has no test to catch the change.
3. **`internal/contract` is the only place `goleak.VerifyTestMain` is wired.** A provider with a leaky goroutine inside `Stream` (especially Ollama's `go func()` at `ollama/ollama.go:48-61`) would not fail any of the per-package test files. The Ollama cancel test is the only line of defense.
4. **No live test for openai, anthropic, deepseek, minimax.** Only Ollama has the nightly live verification. Wire-shape drift in upstream SSE / JSON is invisible until production.
5. **Ollama `tool_strategy.go:90-141` JSON parsing branches** are exercised by `qwen` and `llama3.1` happy paths, but the malformed-JSON / partial-tag error branches (e.g. `decodeFallbackToolCall` returning an error in `parsePythonTagToolCalls`) are not directly tested.
6. **The `statusCapturingTransport` wrap in `ollama/options.go:90-93` mutates the caller's `*http.Client`.** Currently copied (`cp := *httpClient`) but no test asserts that the caller's transport is left untouched. Side-effect regressions would pass.
7. **`ollama/embed_strategy.go:22 supportsEmbeddings`** is unused dead code (only referenced inside itself). No test, and `go vet` does not flag unused unexported functions. Worth either deleting or using inside `New`.
8. **No fuzz tests anywhere.** SSE/JSON parsers in `openai/openai.go:158` (`chunkEvents`) and `ollama/tool_strategy.go:50` (`parseResponseToolCalls`) decode untrusted byte streams from the wire — both are realistic fuzz targets.

---

*Testing analysis: 2026-05-20*
