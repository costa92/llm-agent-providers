# Changelog

All notable changes to llm-agent-providers are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet.

## [v0.3.1] - 2026-06-03

### Fixed

- **ollama**: qwen tool calls returned as a ```` ```json ```` markdown fenced
  block in message content are now unwrapped before the bare-JSON fallback in
  `parseQwenToolCalls`, so `qwen2.5-coder` / `qwen3-coder` tool calls are
  extracted and executed instead of silently dropped (#38). `llama3.1` (native
  `tool_calls`) was unaffected. Verified end-to-end against live
  `qwen2.5-coder` and `llama3.1` via the `llm-agent` `10-ollama-tools` example.

### Changed

- Pinned `llm-agent-contract` to v0.2.0 (#36).

## [v0.3.0] - 2026-06-03

### Changed

- Finalized the `llm-agent-contract` decoupling: every provider imports the
  contract types from `github.com/costa92/llm-agent-contract` directly, pinned
  at v0.1.0 with zero `replace` directives. Pure import-path migration across
  all five providers; no behavior or public-surface change.

## [v0.2.5] - 2026-05-23

K1 keystone — ollama YELLOW → GREEN via cross-provider conformance gate. Test
+ doc only; zero production code changes.

### Added (K1 keystone — closes ollama YELLOW)

- `internal/contract.AssertStreamToolCalls` — K1 conformance assertion
  for streaming-with-tool-calls. Pins per-call `Index` stability across
  `EventToolCallStart` → `EventToolCallArgsDelta`(s) → `EventToolCallEnd`
  and that `EventDone` is terminal with `Usage` + `FinishReason`
  populated. New `Fixture.Expect.StreamSequence []string` field encodes
  the expected `Kind` name sequence (text / tool_start / tool_args /
  tool_end / thinking / done).
- `TestStreamToolCalls_Conformance` — cross-provider K1 gate exercising
  5/5 providers (6 cases: openai / anthropic / ollama-native /
  ollama-qwen-xml / deepseek / minimax). Fixtures derived verbatim from
  validated per-provider streaming-tool-call test wire shapes
  (`openai/openai_test.go:333`, `anthropic/anthropic_test.go:196`,
  `ollama/ollama_test.go:609`, `ollama/ollama_test.go:670`,
  `deepseek/deepseek_test.go:359`, `minimax/minimax_test.go:220`).
- ollama K1 keystone flipped from YELLOW to GREEN. The classification
  was stale — ollama emission code at `ollama/ollama.go:211-327` has
  been K1-conformant since commit `32f5d59` (2026-05-20), emitting the
  full `EventToolCallStart/ArgsDelta/End` triple for both native
  `tool_calls`-field and content-parsed `<tool_call>...</tool_call>`
  paths with stable per-call `Index`. Three planning docs that still
  claimed YELLOW predate that commit and are now refreshed
  (`.planning/codebase/CONCERNS.md`, `.planning/codebase/ARCHITECTURE.md`,
  `.planning/codebase/TESTING.md`); the umbrella
  `docs/ecosystem-design-review.zh-CN.md` is updated in a paired
  umbrella-repo commit. Keystone scorecard now reads **12 GREEN / 0
  YELLOW / 0 RED**.
- Zero production code changes in this release. Pure test + doc work.

### Deferred (Phase 2 / v0.6.0 follow-up)

- `llm-agent/llm/stream.go::appendToolCallDelta` keys by `ID`; should
  key by `Index`. The function comment already notes "NOT the
  production accumulator". Touches the frozen core, requires a minor
  bump, and was out of scope for this YELLOW-lift PR.
- Real-capture fixtures from live providers — current `stream_tool_*`
  fixtures are hand-crafted from validated per-provider test wire
  shapes (same validity, faster to ship). If `nightly-ollama-live.yml`
  observes a wire-shape drift, refresh from real capture.

## [v0.2.4] - 2026-05-23

Phase — P1-23 (PRs 1+2+3 of 3): extract shared SDK error mapping and
default-timeout into `internal/compat/`; migrate `openai/`, `deepseek/`,
`anthropic/`, `minimax/`, and finally `ollama/` (default-timeout call
swap only — Path A). Plus P1-6 ollama closure (5/5).

### Refactored (P1-23 PR 1 of 3)

- Extracted shared OpenAI-SDK error mapping into `internal/compat/`:
  `WrapOpenAIError(provider, err)` is now used by both `openai/` and
  `deepseek/` (the two providers that piggyback on
  `github.com/openai/openai-go/v3`). The two `errors.go` files were
  character-for-character identical except for the 6 occurrences of the
  provider-name string.
- Extracted `if timeout == 0 { 60s }` block into
  `compat.DefaultTimeout(d)`. Used by openai + deepseek;
  anthropic/minimax/ollama migrate in subsequent PRs.
- `internal/compat/` is Go-`internal/`-scoped — downstream consumers
  cannot import it. No public API change. Per-provider test counts
  unchanged; conformance suite (5/5 providers) stays GREEN.

### Refactored (P1-23 PR 2 of 3)

- Extracted shared Anthropic-SDK error mapping into `internal/compat/`:
  `WrapAnthropicError(provider, err)` is now used by both `anthropic/` and
  `minimax/` (the two providers that piggyback on `github.com/anthropics/anthropic-sdk-go`).
  Preserves the 529 Overloaded → RateLimitError special case.
- Default-timeout for anthropic + minimax now via `compat.DefaultTimeout`.
- No public API change. Per-provider test counts unchanged.

### Refactored (P1-23 PR 3 of 3)

- Ollama default-timeout now flows through `compat.DefaultTimeout`,
  preserving the http-client-aware guard (the conditional that skips
  the default when the user supplied a pre-configured http.Client with
  its own Timeout). Closes the P1-23 sequence at 5/5 providers calling
  the same default-timeout helper.
- Ollama `errors.go` stays per-provider (Path A): the atomic-state
  pattern (statusCapturingTransport + atomic.Pointer[string]) is the
  outlier and refactoring it would touch the recently-painted P1-6
  derived-clients-share-transport work for ~38 LoC saved. Deferred
  to a future P1-23b if/when a 6th OpenAI-compat provider arrives.

### Changed (behavior — defensive default, P1-6 follow-up)

- **ollama `New()`** completes the 5/5 timeout closure. Sync calls
  (`Generate`, `Embed`) now honor a 60s default request timeout via a
  derived `*http.Client`; `Stream` uses a sibling client with
  `Timeout=0` so long stream connections remain governed by the
  caller `ctx` only. Both clients share the same
  `*statusCapturingTransport` instance so `lastStatus` / Retry-After
  observation stays reference-identical across paths — 429s detected
  on a sync request remain visible to subsequent error wrapping
  regardless of which path made the call.
- Rationale: ollama-go SDK v0.23.2 exposes no per-request timeout
  option and its `Client.http` field is private, so the 4-provider
  pattern (`option.WithRequestTimeout`) does not apply. The derived-
  client split is the only clean path; a single shared client with a
  non-zero `Timeout` would cut streams at 60s.
- `Ollama` struct gained unexported `timeout`, `syncHTTPClient`, and
  `streamHTTPClient` fields so internal tests (`*_internal_test.go`)
  can assert resolved timeout and transport identity. No public
  surface change.

### Notes

- Closes the deferred item from v0.2.3 ("ollama `New()` not modified
  in this PR"). All 5/5 SDK providers now ship the defensive default.
- Unblocks `P1-23` compat extraction (v1.4 window) which assumes no
  SDK provider `New()` can hang silently.
- Refs: `docs/refactor-and-optimization-roadmap.zh-CN.md` §P1-6.

## [v0.2.3] - 2026-05-23

Phase — P1-6: default 60s HTTP request timeout on 4 SDK providers.

### Changed (behavior — defensive default)

- **openai/anthropic/deepseek/minimax `New()`** now set a default 60s
  request timeout when the caller has not explicitly passed
  `WithTimeout`. Prevents indefinite hangs on idle connections when
  upstream stalls. Implementation: the default is applied to
  `cfg.timeout` and forwarded to the SDK via
  `option.WithRequestTimeout(cfg.timeout)`. **Streaming is not capped
  by a client-level `Timeout`** — the SDK-level option applies
  per-request, and caller `ctx` continues to govern long-running
  `Stream` calls.
- Each provider struct gained an unexported `timeout time.Duration`
  field so the resolved value can be observed by internal tests
  (`*_internal_test.go`, `effectiveTimeoutForTest`). No public surface
  change.

### Deferred

- **ollama `New()` not modified in this PR.** Audit found that
  `Stream` shares the same `*api.Client` (and thus the same
  `*http.Client`) as `Generate`, and `httpClient.Timeout` applies to
  the entire request lifecycle including streaming body read. A naive
  default would cut long stream connections at 60s. A follow-up will
  introduce a derived client (or equivalent) so the default only
  affects sync calls. Tracked separately; ollama callers should keep
  setting `WithTimeout` explicitly when they need a hang guard.

### Notes

- Unblocks P1-23 compat extraction (v1.4 window) which assumes the
  4 SDK provider `New()` calls cannot hang silently.
- Refs: `docs/refactor-and-optimization-roadmap.zh-CN.md` §P1-6.
