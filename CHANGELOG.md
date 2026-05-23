# Changelog

All notable changes to llm-agent-providers are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Phase — P1-23 (PRs 1+2 of 3): extract shared SDK error mapping and
default-timeout into `internal/compat/`; migrate `openai/`, `deepseek/`,
`anthropic/`, and `minimax/`. Plus P1-6 ollama closure (5/5).

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
