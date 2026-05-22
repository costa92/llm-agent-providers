# Changelog

All notable changes to llm-agent-providers are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project follows [Semantic Versioning](https://semver.org/).

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
