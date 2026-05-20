# Codebase Concerns

**Analysis Date:** 2026-05-20
**Repo:** `github.com/costa92/llm-agent-providers`
**Current tag:** `v0.2.1` (latest), `v0.2.0` (named in the prompt; one patch newer is published)

## K1/K2 compliance per provider — the most important table

**K1** = the stream emits the typed `llm.StreamEvent` union with stable per-tool-call `Index` linking Start → ArgsDelta(s) → End → Done.
**K2** = `New(WithModel(m))` binds a single model, and `Info().Capabilities` reflects **that exact model** (not provider-wide capability flags).

| Provider | K1 score | K2 score | Notes |
|----------|----------|----------|-------|
| **openai** | GREEN | GREEN | Reference implementation. K1: full `EventToolCallStart/ArgsDelta/End/Done` with `Index` (`openai/openai.go:158-231`), tested with two distinct tool indexes (`openai/openai_test.go:333-395`). Retry-before-first-byte + no-retry-after-first-byte both covered. K2: chat models flip `Tools=true`; `text-embedding-3-*` flips `Embeddings=true` and dimensions (`openai/options.go:68-83`, `openai/openai.go:83-94`). |
| **anthropic** | GREEN | GREEN | K1: Index taken from upstream `ContentBlockStartEvent.Index`, carried through ArgsDelta and End (`anthropic/anthropic.go:122-197`); tested at `anthropic/anthropic_test.go:196-272`. K2: `Capabilities.Tools=true`, embeddings explicitly false, no `Embed` method on the type and asserted in tests (`anthropic_test.go:45-57`). The Phase-1 doc-comment is stale but the runtime contract is correct. |
| **ollama** | YELLOW | GREEN | K1 is YELLOW because the Ollama stream **does not emit tool-call events at all** — `ollama/ollama.go:120-187` only emits `EventTextDelta` and `EventDone`. Ollama's wire format returns tool calls only on the final `done:true` frame, which is mapped to `llm.Response.ToolCalls` via `Generate`. This is technically faithful to the upstream protocol, but a caller wiring an `Index`-aware tool-streaming pipeline will get nothing from Ollama, breaking K1's "every adapter speaks the typed union" claim. K2 is GREEN — Ollama is the K2 reference: `strategyForModel(cfg.model)` and `embeddingDimensionForModel(cfg.model)` make `Capabilities.Tools` and `Capabilities.Embeddings` model-dependent (`ollama/options.go:74-109`). |
| **deepseek** | YELLOW | YELLOW | K1 is YELLOW only because of test surface — the code itself emits `EventToolCallStart/ArgsDelta/End/Done` with `Index` (`deepseek/deepseek.go:138-186`), asserted at `deepseek/deepseek_test.go:380-415`, **but** deepseek is absent from `internal/contract/generate_test.go AdapterFactories` (lines 20-41), so cancel-mid-stream and partial-usage-on-error are unverified. K2 is YELLOW because `Capabilities` is hard-coded to `{Tools:true, others:false}` regardless of model name (`deepseek/options.go:93-98`). If/when DeepSeek ships an embeddings endpoint or a non-tool reasoning-only model, `Info()` will lie. |
| **minimax** | YELLOW | YELLOW | Same shape as deepseek. K1: code is correct (`minimax/minimax.go:122-197`, tested at `minimax_test.go:220-296`), but missing from the cross-provider conformance suite. K2: capabilities hard-coded to `{Tools:true, others:false}` (`minimax/options.go:93-98`). |

**Bottom line:** zero RED scores. The K1/K2 invariants hold in source; the YELLOW marks are about (a) Ollama's stream contract gap, (b) deepseek/minimax being a second-class citizen in the conformance harness, and (c) deepseek/minimax encoding capability as a per-provider constant instead of per-(provider × model).

## Concern list

### Coverage gap: deepseek and minimax bypass the cross-provider conformance suite

**Description:** `internal/contract/generate_test.go:20-41` (`AdapterFactories` and `EmbeddingAdapterFactories`) lists only openai, anthropic, ollama. The corresponding fixture trees `internal/contract/testdata/deepseek/` and `internal/contract/testdata/minimax/` do not exist (`ls internal/contract/testdata/` shows only `openai/`, `anthropic/`, `ollama/`).
**Evidence:** `internal/contract/generate_test.go:20-41`; `find internal/contract/testdata -type d` returns three dirs only.
**Severity:** HIGH — the strongest gate in the repo (`TestStream_CancelMidStream_Conformance`, `TestStream_PartialUsageOnError_Conformance`, `TestToolCalling_DedupeKey`, `TestErrorString_NoSecretLeak`) silently skips the two newest adapters.
**Suggested action:** add deepseek and minimax to both factory maps, capture fixtures via new `scripts/capture-fixtures-{deepseek,minimax}.sh`, drop the fixtures into `testdata/{deepseek,minimax}/`. Mirror the openai/anthropic shape — deepseek over openai-go, minimax over anthropic-sdk-go.

### Stale README claims about `llm-agent` version

**Description:** README says "This repo is already code-compatible with the `llm-agent v0.4` core surface. Its local release-prep state now targets `github.com/costa92/llm-agent v0.4.0`" and refers to "v0.4 removal track" in three places. `go.mod` actually pins `github.com/costa92/llm-agent v0.5.1`.
**Evidence:** `README.md:9-10`, `README.md:106-108`, `README.md:139` vs `go.mod:7`.
**Severity:** MEDIUM — users following the README will pin against the wrong core version and get confused by missing/renamed symbols.
**Suggested action:** rewrite README "Versioning" section to reference `v0.5.x` and drop the "remaining Phase 7" note. Replace `@v0.1.0` in the install snippet (lines 24-28) with `@v0.2.x` (the actual published tag).

### Stale Phase-1 doc.go comments contradict shipped code

**Description:** `openai/doc.go:5-10`, `anthropic/doc.go:5`, `ollama/doc.go:5` all say "intentionally reports all optional capabilities as false in Phase 1" and the openai one says "Streaming is deferred to Phase 2". Every adapter ships `Stream`, `WithTools`, and (for openai/ollama) `Embed`; capabilities are not uniformly false.
**Evidence:** `openai/doc.go:1-11`, `anthropic/doc.go:1-7`, `ollama/doc.go:1-7` vs `openai/options.go:67-83`, `anthropic/options.go:71-80`, `ollama/options.go:99-109`.
**Severity:** MEDIUM — `go doc github.com/costa92/llm-agent-providers/openai` returns misleading documentation to end users and to anyone reading via pkg.go.dev.
**Suggested action:** rewrite each `doc.go` to describe the actual capability matrix and remove the Phase-1/Phase-2 prose. DeepSeek and MiniMax `doc.go` are already correct templates (single accurate sentence).

### Per-package versioning advertised but never tagged

**Description:** README installation block (lines 24-28) tells users to `go get github.com/costa92/llm-agent-providers/openai@v0.1.0` — implying submodule-style per-provider tagging. The repo has only top-level tags (`v0.2.1`, `v0.2.0`, `v0.1.1`, `v0.1.0`) and no per-provider `go.mod`. `go get` of those paths resolves against the root module, which works, but the explicit per-package version implies independence the repo doesn't offer.
**Evidence:** `git tag` output; absence of `openai/go.mod`, `anthropic/go.mod`, etc.; `README.md:24-28`.
**Severity:** LOW — works in practice, misleads readers in principle.
**Suggested action:** either (a) collapse the install block to a single `go get github.com/costa92/llm-agent-providers@vX.Y.Z` line, or (b) commit to per-provider submodule tagging and add `openai/v0.2.0` style tags + per-package CHANGELOG.md files.

### No `CHANGELOG.md`

**Description:** Neither at repo root nor in any provider subdirectory. With four published tags, there is no machine- or human-readable change history.
**Evidence:** `find . -name 'CHANGELOG*'` returns nothing.
**Severity:** LOW — downstream consumers (`llm-agent-customer-support`, etc.) cannot trivially see what changed between `v0.1.x` and `v0.2.x`.
**Suggested action:** add `CHANGELOG.md` at root using Keep-a-Changelog format. Backfill from `git log --tags --simplify-by-decoration --pretty='%h %s'`.

### SDK coupling: official SDK pins create churn risk

**Description:** Four direct upstream SDK dependencies, each pinned at a specific minor:
- `github.com/openai/openai-go/v3 v3.35.0` (`go.mod:9`) — used by openai **and** deepseek
- `github.com/anthropics/anthropic-sdk-go v1.41.0` (`go.mod:6`) — used by anthropic **and** minimax
- `github.com/ollama/ollama v0.23.2` (`go.mod:8`) — used by ollama
- `github.com/testcontainers/testcontainers-go/modules/ollama v0.42.0` (`go.mod:10`) — only at `ollama_live` build tag

A breaking change in any of those SDKs forces a coordinated bump across two adapters at minimum (openai+deepseek; anthropic+minimax). The current code leans on SDK types in `map.go` (e.g. `openai.ChatCompletionNewParams`, `sdk.MessageNewParams`, `api.ChatRequest`) and on SDK error types in `errors.go` (`*openai.Error`, `*sdk.Error`, `api.AuthorizationError`, `api.StatusError`).
**Evidence:** `openai/map.go:12, 58-64`; `anthropic/map.go:10`; `ollama/map.go:10`; `openai/errors.go:23`; `anthropic/errors.go:25`; `ollama/errors.go:28-34`.
**Severity:** MEDIUM — the coupling is intentional and the boundary discipline (SDK never leaks onto public symbols) keeps it manageable, but any major-version bump on openai-go (`v3 → v4`) or anthropic-sdk-go (`v1 → v2`) requires touching four files in parallel.
**Suggested action:** keep an explicit `var _ openai.X = ...` style check in CI to catch SDK signature drift early; consider extracting the SDK-translation surface into a versioned internal seam so future major bumps live behind a single import.

### Hard-coded capability matrix for deepseek and minimax (K2 risk)

**Description:** `deepseek/options.go:93-98` and `minimax/options.go:93-98` set `Capabilities` literally to `{Tools: true, Embeddings: false, StructuredOutputs: false, PromptCaching: false}` ignoring `cfg.model`. The same construction is identical regardless of model string. This violates the K2 spec ("per-(provider × model)") in spirit even though it currently agrees with reality (neither provider exposes a tools-disabled or embeddings-enabled model through these adapters yet).
**Evidence:** `deepseek/options.go:93-98`, `minimax/options.go:93-98`.
**Severity:** MEDIUM — future-proofing concern, will silently lie the moment a new model ships.
**Suggested action:** add `capabilitiesForModel(model string) llm.Capabilities` helpers to each package (matching the Ollama style in `ollama/tool_strategy.go:20-48`), even if all current entries map to the same literal. Forces a code change when the catalogue evolves.

### Credential handling: env-var fallbacks live in code, no redaction layer

**Description:** Every provider falls back to `os.Getenv("<PROVIDER>_API_KEY")` if `WithAPIKey` is not set. The key flows into the SDK via `option.WithAPIKey(cfg.apiKey)` without being stored on the `Provider` struct beyond that handoff. The unique cross-provider negative test is `internal/contract/generate_test.go:355 TestErrorString_NoSecretLeak`, which only checks that the typed `llm.*Error.Error()` envelope passes `errors.Is` and notes if the inner error happens to omit a `sk-FAKE…` token. It does not enforce non-leakage — the test comment is "upstream redaction improved" if the secret is absent.
**Evidence:** `openai/options.go:45-47`, `anthropic/options.go:45-47`, `ollama/options.go:58-63`, `deepseek/options.go:63-65`, `minimax/options.go:63-65`; `internal/contract/generate_test.go:355-380`.
**Severity:** MEDIUM — if any upstream SDK ever surfaces a request body with the `Authorization` header in its error string, the typed error wraps it transparently. The test would log but not fail.
**Suggested action:** convert `TestErrorString_NoSecretLeak` to a hard assertion (`if strings.Contains(s, "sk-FAKE") { t.Errorf(...) }`), and add a wrapper sanitizer in `wrapErr` that strips `Authorization:` and `Bearer ` prefixes from the inner error string before wrapping.

### Network safety: SDK retries disabled, but no enforced timeout

**Description:** Every provider explicitly disables SDK-level retries (`option.WithMaxRetries(0)` at `openai/options.go:50`, `anthropic/options.go:50`, `deepseek/options.go:73`, `minimax/options.go:73`). Good — retries are an `llm-agent` core concern, not an adapter concern.

`WithTimeout(d time.Duration)` is wired to `option.WithRequestTimeout` (openai/anthropic/deepseek/minimax) or `httpClient.Timeout` (ollama). But **`New` does not require a timeout** — if the user never calls `WithTimeout`, the underlying HTTP client uses Go's default (`http.DefaultClient` with no timeout) for openai/anthropic/deepseek/minimax SDKs. Ollama's `New` builds a fresh `http.Client` with `Timeout: 0` when no `WithTimeout` is given (`ollama/options.go:76-85`).
**Evidence:** `openai/options.go:37-86`, `ollama/options.go:48-111`.
**Severity:** MEDIUM — calling `Generate` with a `context.Background()` and no `WithTimeout` can hang indefinitely on a stuck connection. Streaming endpoints make this worse because a hung TCP read won't surface.
**Suggested action:** set a sensible default (e.g. 60s) inside `New` if `cfg.timeout == 0`, mirrored across all five providers. Document the default in the new `doc.go`.

### Network safety: streaming reader unbounded queue

**Description:** Every `<provider>StreamReader` buffers events in an unbounded slice `queue []llm.StreamEvent` (`openai/openai.go:100`, `anthropic/anthropic.go:61`, `deepseek/deepseek.go:55`, `minimax/minimax.go:61`). One upstream chunk can decompose into many `StreamEvent`s (start + many args + end). If the consumer pauses calling `Next()` while the upstream keeps producing, the queue grows. Ollama's reader uses a `chan api.ChatResponse` of capacity 1, which back-pressures correctly — the SSE-based readers do not.

In practice, this is bounded by the upstream chunk count, which is bounded by the model's output, so it's not unbounded in theory. But there is no explicit cap and no test that the queue ever drains.
**Evidence:** `openai/openai.go:100, 142`; `deepseek/deepseek.go:55, 97`; `anthropic/anthropic.go:61, 106`; `minimax/minimax.go:61, 106`; contrast with `ollama/ollama.go:43-44` (`respCh: make(chan api.ChatResponse, 1)`).
**Severity:** LOW-MEDIUM — slow consumers can hold memory proportional to model output. Not a leak, but a sharp edge.
**Suggested action:** either document the buffered-up-to-output-length behavior explicitly or convert to a bounded channel pattern matching the Ollama reader.

### Provider drift: openai is leader, deepseek/minimax are clones, ollama is the bespoke one

**Description:** Lineage:
- **openai** is the canonical implementation. Wire-shape: SSE via `openai-go/v3/packages/ssestream`. Most thorough test coverage (17 funcs, 590 LOC of tests).
- **anthropic** is the second-tier sibling — same shape, different upstream SDK family (`anthropic-sdk-go`). Adds `WithBetaHeader` and the system-prompt-lifting trick.
- **deepseek** is a near-verbatim copy of openai (`grep -c 'Index' deepseek/deepseek.go` matches openai line-for-line; `wc -l deepseek/deepseek.go == 186`, `wc -l openai/openai.go == 231` — the delta is openai's `Embed` block). Errors module is identical except for the literal `"deepseek"` string.
- **minimax** is a near-verbatim copy of anthropic (`minimax/minimax.go` ≈ `anthropic/anthropic.go` minus the embedding-gap comment).
- **ollama** is the outlier — bespoke `*_strategy.go` files for per-model tool encoding, a custom `statusCapturingTransport`, callback-based streaming.

**Risk:** changes to streaming behavior in openai need to be hand-replicated to deepseek; same for anthropic→minimax. There is no DRY layer.
**Evidence:** `diff -u openai/openai.go deepseek/deepseek.go` shows ~90% identical structure; same for `anthropic/anthropic.go` vs `minimax/minimax.go`.
**Severity:** MEDIUM — drift will silently accumulate if openai and deepseek diverge on streaming retry semantics.
**Suggested action:** if a third openai-compatible provider arrives, factor the SSE stream reader into `internal/openaiсompat/` and consume it from openai and deepseek. Same for anthropic-compat → anthropic and minimax. Until then, accept the duplication and add a CI lint that runs `diff openai/openai.go deepseek/deepseek.go | grep -c '^+' < threshold`.

### Dependency direction: clean — no accidental sibling imports

**Description:** Checked for accidental imports of `llm-agent-rag`, `llm-agent-otel`, `llm-agent-customer-support`. None found.
**Evidence:** `grep -rn 'llm-agent-rag\|llm-agent-otel\|llm-agent-customer-support' --include='*.go'` returns zero matches; `go.mod` lists only `github.com/costa92/llm-agent v0.5.1` plus SDK/testcontainer deps.
**Severity:** none — passes.

### TODO / FIXME / XXX / HACK count per provider

**Description:** `grep -rn 'TODO\|FIXME\|XXX\|HACK' --include='*.go'` returns **zero** matches across the entire repo, in source or test files. This is unusually clean.
**Evidence:** explicit verification, no output.
**Severity:** none — passes.

### Swallowed errors and lint escapes

**Description:** No `_ = err`, no `// nolint:errcheck`, no `// nolint` directives anywhere. The only `_ =` patterns are:
- `_ = r.stream.Close()` in the four SSE-based stream readers (`openai/openai.go:129`, `deepseek/deepseek.go:84`, `anthropic/anthropic.go:93`, `minimax/minimax.go:93`) — already captured `r.stream.Err()` above.
- `_, _ = w.Write([]byte(f.Response.Body))` in `internal/contract/contract.go:85` — test mock writer.
**Severity:** none — passes.
**Note:** the `_ = r.stream.Close()` pattern is fine; the underlying error is the same one returned by `r.stream.Err()` per `ssestream` semantics.

### `panic(` in production paths

**Description:** Zero `panic(` invocations anywhere (`grep -rn 'panic(' --include='*.go'` returns no results).
**Severity:** none — passes.

### API surface bloat: dead exported symbol candidates

**Description:**
- `ollama/embed_strategy.go:22 supportsEmbeddings(model string) bool` — unexported but **unused** within the package (`grep -rn 'supportsEmbeddings\b' --include='*.go'` only shows the declaration; the `New` flow uses `embeddingDimensionForModel(cfg.model) > 0` directly at `ollama/options.go:75, 105`). Either delete or replace the inline `embedDim > 0` checks with calls to this helper.
- `ollama.WithHost` (`ollama/options.go:29`) — public alias for `WithBaseURL`. Currently undocumented and only referenced once inside the package itself. Either document the equivalence or remove the alias.
- `Region` / `RegionCN` / `RegionGlobal` in both deepseek and minimax — exported, but their internal `baseURLForRegion` is a no-op switch (`deepseek/options.go:46-53`, `minimax/options.go:46-53`) that returns the same default for both regions. Today this is API surface promising routing it doesn't actually do.
**Evidence:** see file/line references above.
**Severity:** LOW — surface looks promising but is functionally degenerate.
**Suggested action:** (a) delete `supportsEmbeddings` or use it; (b) document or delete `WithHost`; (c) either populate `baseURLForRegion` with the actual CN endpoint or remove the `Region` API until it has meaningful behavior.

### `docs/` accuracy

**Description:** `docs/` contains `docs/superpowers/specs/2026-05-13-deepseek-minimax-design.md` (308 lines) and `docs/superpowers/plans/2026-05-13-deepseek-minimax-implementation.md` (451 lines). Both are pre-implementation design docs, not living architecture references. They are dated 2026-05-13, and the code dates from the same week. Spot checks confirm the design contracts (`deepseek.New(opts ...Option)`, `WithRegion`, `Generate/Stream/WithTools/Info`) match what is shipped.
**Evidence:** `docs/superpowers/specs/2026-05-13-deepseek-minimax-design.md:43-58` vs `deepseek/options.go:34-44`.
**Severity:** LOW — accurate for the moment they were written, but described as "proposed" in the spec while the code has shipped. Promote to `accepted` or move to an `accepted/` subdirectory.
**Suggested action:** update the front-matter `Status: proposed` to `Status: implemented (v0.2.0)`. No code change needed.

### README freshness

**Description:** Aside from the `v0.4` claim called out above:

- Line 9 "ships the full Phase 1-4 provider surface" — vague but defensible.
- Lines 24-28 install snippet uses `@v0.1.0` while the latest tag is `v0.2.1` — outdated.
- Line 110 "Versioning" section: "The only remaining Phase 7 follow-up is publishing the final coordinated tags." — false; v0.2.x tags are published.
- Lines 130-133 link to `PR-GOVERNANCE-*` docs in the upstream `llm-agent` repo; targets are presumed live (not verified here).
- Line 137 references `CLAUDE.md` in the core repo (linked but not validated here).

The full multi-repo Phase 7 narrative is the largest stale paragraph; trim to a single sentence pointing at the current release state.
**Severity:** MEDIUM (combined with the `v0.4` issue).
**Suggested action:** see the "Stale README claims" entry above for full rewrite scope.

### Logging in production paths

**Description:** Zero `log.`, `fmt.Print*`, or `println(` calls in non-test files. The repo treats observability as an `llm-agent` core concern — no leakage here.
**Evidence:** `grep -rn 'log\.\|fmt\.Print\|println(' --include='*.go' | grep -v _test.go` returns no production-path matches.
**Severity:** none — passes.

### `internal/contract` is test-only but lives outside `*_test.go`

**Description:** `internal/contract/contract.go` is a regular Go file (not `_test.go`) but exports `Fixture`, `LoadFixture`, `NewMockServer`, `AssertGenerate`, `AssertStream`, `AssertToolCalling`, `AssertEmbed`, all of which take `*testing.T`. Because of the `*testing.T` signature this can only be linked from `_test.go` callers, so functionally it's test code; but it gets `go vet`/`go build`-ed as production code.
**Evidence:** `internal/contract/contract.go:52, 66, 99, 108, 128, 162`.
**Severity:** LOW — works because the `internal/` rule prevents external consumers and the `*testing.T` signature blocks misuse. But `go doc github.com/costa92/llm-agent-providers/internal/contract` (hypothetically) would list these as production API.
**Suggested action:** either rename to `contract_test.go` (and the package becomes test-only) — would require moving fixture helpers callers consume, which is fine because callers are all `_test.go` already — or accept the current shape and add a package-level comment "//go:build ! production" style note.

### Concentration of test work in one file per provider

**Description:** Each `<provider>_test.go` runs 485 – 590 lines and 15 – 19 test functions. Single-file growth invites merge conflicts and makes `git blame` over time noisy. The contract suite already shows that fixture-driven splits scale better.
**Evidence:** see TESTING.md table.
**Severity:** LOW — readability concern, not correctness.
**Suggested action:** when a provider crosses 800 LOC of tests, split by concern: `<provider>_stream_test.go`, `<provider>_tools_test.go`, `<provider>_errors_test.go`.

### Summary — what needs attention soonest

1. **HIGH** — Wire deepseek and minimax into `internal/contract/generate_test.go` and add their fixture testdata. This single change closes the largest gap.
2. **MEDIUM** — Fix README version claims and stale `doc.go` Phase-1 prose. Cheap and improves first impressions.
3. **MEDIUM** — Add a default timeout inside each provider's `New` (no current default = unbounded hang risk).
4. **MEDIUM** — Replace hard-coded `Capabilities` in deepseek and minimax with a `capabilitiesForModel` helper to honor K2 in spirit.
5. **LOW** — Delete dead `supportsEmbeddings`; document or delete degenerate `Region` API on deepseek/minimax until CN endpoint diverges.

---

*Concerns audit: 2026-05-20*
