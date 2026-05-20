# Technology Stack

**Analysis Date:** 2026-05-20

## Languages

**Primary:**
- Go 1.26.0 — declared in `go.mod` line 3 (`go 1.26.0`). All five adapters plus `internal/contract` are pure Go.

**Secondary:**
- Bash — `scripts/*.sh` fixture-capture helpers and `scripts/workspace.sh`.
- Python 3 — embedded heredoc snippets inside the capture scripts (e.g. `scripts/capture-fixtures-openai.sh:22-51`) and inside `.github/workflows/release-precheck.yml:24-29`. Required at script-run time only.

## Runtime

**Environment:**
- Go toolchain ≥ 1.26.0 (matches `go.mod`).
- No long-running service runtime — adapters are libraries, consumed in-process by `llm-agent`.

**Package Manager:**
- `go` modules.
- Lockfile: `go.sum` present (16361 bytes, committed).

## Frameworks

**Core:**
- `github.com/costa92/llm-agent v0.5.1` (`go.mod:7`) — provides `llm.ChatModel`, `llm.ToolCaller`, `llm.Embedder`, `llm.StreamEvent`, the typed error sentinels, etc. Every adapter imports `github.com/costa92/llm-agent/llm`.

**Testing:**
- Standard library `testing` (`net/http/httptest` for mock servers).
- `go.uber.org/goleak v1.3.0` (`go.mod:11`) — used in `internal/contract/main_test.go` via `goleak.VerifyTestMain(m)` to fail tests that leak goroutines.
- `github.com/testcontainers/testcontainers-go/modules/ollama v0.42.0` (`go.mod:10`) — gated behind the `ollama_live` build tag (`internal/contract/ollama_live_test.go:1`); nightly CI only.

**Build/Dev:**
- `go vet ./...`, `go build ./...`, `go test ./...` from `.github/workflows/test.yml:39-44`. No external build tool.
- `go mod tidy` drift check runs in CI (`.github/workflows/test.yml:29-38`).

## SDK choice per provider

| Provider | SDK | Hand-rolled HTTP? | Evidence |
|----------|-----|-------------------|----------|
| openai | Official `github.com/openai/openai-go/v3 v3.35.0` | No | `openai/openai.go:10-11`, `openai/options.go:10-11` |
| anthropic | Official `github.com/anthropics/anthropic-sdk-go v1.41.0` | No | `anthropic/anthropic.go:8-9`, `anthropic/options.go:9-10` |
| ollama | Official `github.com/ollama/ollama/api v0.23.2` | No | `ollama/ollama.go:9`, `ollama/options.go:13` |
| deepseek | Reuses `openai-go/v3` (DeepSeek exposes an OpenAI-compatible API) | No | `deepseek/deepseek.go:10-11`, `deepseek/options.go:10-11`, `docs/superpowers/specs/2026-05-13-deepseek-minimax-design.md:6-7` |
| minimax | Reuses `anthropic-sdk-go` (MiniMax exposes an Anthropic-compatible API) | No | `minimax/minimax.go:8-9`, `minimax/options.go:9-10`, `docs/superpowers/specs/2026-05-13-deepseek-minimax-design.md:6-7` |

No provider rolls its own HTTP client. Adapters wrap official SDKs and normalize the result onto the `llm-agent` provider contract.

## Streaming protocol per provider

| Provider | Wire protocol | Detector |
|----------|--------------|----------|
| openai | Server-Sent Events (SSE) via `openai-go/v3/packages/ssestream` | `openai/openai.go:11`, stream type `*ssestream.Stream[openai.ChatCompletionChunk]` (`openai/openai.go:98-99`) |
| deepseek | Same SSE pipe as openai (shared SDK) | `deepseek/deepseek.go:11`, `*ssestream.Stream[openai.ChatCompletionChunk]` (`deepseek/deepseek.go:52-53`) |
| anthropic | SSE via `anthropic-sdk-go/packages/ssestream` over `MessageStreamEventUnion` events (`event: message_start`, `content_block_start`, …) | `anthropic/anthropic.go:9`, `anthropic/anthropic.go:60`; mock SSE traffic in `internal/contract/generate_test.go:412-418` |
| minimax | Same Anthropic SSE union (shared SDK) | `minimax/minimax.go:9`, `minimax/minimax.go:59-60` |
| ollama | Newline-delimited JSON (`application/x-ndjson`), driven by Ollama's callback-style `client.Chat(ctx, req, cb)` | `ollama/ollama.go:29-37` (sync) and `ollama/ollama.go:39-64` (stream). Content-Type asserted at `internal/contract/generate_test.go:427`. |

No provider uses WebSockets.

## Key Dependencies

**Critical:**
- `github.com/costa92/llm-agent v0.5.1` (`go.mod:7`) — sole upstream the adapters target. All capability interfaces, error sentinels, stream-event types live here.
- `github.com/openai/openai-go/v3 v3.35.0` — backs OpenAI + DeepSeek.
- `github.com/anthropics/anthropic-sdk-go v1.41.0` — backs Anthropic + MiniMax.
- `github.com/ollama/ollama v0.23.2` — local-model adapter.

**Infrastructure / indirect (selected):**
- `github.com/openai/openai-go/v3/option` — request-option plumbing used in `openai/options.go` and `deepseek/options.go`.
- `github.com/anthropics/anthropic-sdk-go/option` — same role in `anthropic/options.go` and `minimax/options.go`.
- `github.com/testcontainers/testcontainers-go` (indirect via `modules/ollama`) — pulled in only by the `ollama_live`-tagged test.
- `github.com/cenkalti/backoff/v4`, `github.com/tidwall/{gjson,sjson,pretty,match}`, `github.com/invopop/jsonschema`, `github.com/wk8/go-ordered-map/v2`, OpenTelemetry trace/metric — all are transitive dependencies of the SDKs, not used directly in this repo.

No JSON library beyond the stdlib `encoding/json` is used in this repo's own code (verified — every `import "encoding/json"` lives in `*/map.go` and `*/tool_strategy.go`).

## Configuration

**Environment:**
- `OPENAI_API_KEY` — fallback when `WithAPIKey` is not set (`openai/options.go:46-47`).
- `ANTHROPIC_API_KEY` — same fallback (`anthropic/options.go:45-47`).
- `DEEPSEEK_API_KEY` — fallback (`deepseek/options.go:63-65`).
- `MINIMAX_API_KEY` — fallback (`minimax/options.go:63-65`).
- `OLLAMA_HOST` — base URL fallback; defaults to `http://localhost:11434` (`ollama/options.go:58-63`).
- `GOWORK=off` — CI env (`.github/workflows/test.yml:17`, `.github/workflows/nightly-ollama-live.yml:15`) so workspace files never leak into CI builds.
- `OLLAMA_TC_IMAGE`, `OLLAMA_TC_MODEL` — testcontainer overrides (`internal/contract/ollama_live_test.go:23-24`).

`.env`, `.env.*`, and `**/.env` are git-ignored (`.gitignore:16-19`); the repo does not ship `.env` files.

**Build:**
- `go.mod` (3526 bytes) — module path `github.com/costa92/llm-agent-providers`.
- `go.sum` — committed checksum lockfile.
- No `Makefile`, no `.golangci.yml`, no separate lint config. CI is the only formatter/vetter gate.

## Platform Requirements

**Development:**
- Go 1.26 toolchain.
- For the live Ollama test: Docker, ~few GB disk for the `ollama/ollama:0.5.7` image plus the `llama3.1:8b-instruct-q4_K_M` model layers (see `.github/workflows/nightly-ollama-live.yml:14-37`).
- For fixture capture: `curl`, `python3`, and an API key for the target provider (`scripts/capture-fixtures-{openai,anthropic,ollama}.sh`).

**Production:**
- Library — no deployment target. Consumers import the relevant subpackage and embed the adapter inside their own binary.

## Build / Lint / CI Tooling

`.github/workflows/` ships four workflows:

| File | Trigger | Purpose |
|------|---------|---------|
| `test.yml` | push / PR to `main` | `go mod tidy` drift, `go vet`, `go build`, `go test ./...` with `GOWORK=off` |
| `nightly-ollama-live.yml` | cron `0 3 * * *` + manual dispatch | testcontainer-backed Ollama run, build tag `ollama_live`, only `TestGenerate_Ollama_Live` |
| `release-precheck.yml` | push / PR to `release/**` | Rejects any `replace` directive in `go.mod` (INFRA-04 guard) |
| `pr-governance.yml` | PR events + reviews | Owner-authored PRs auto-pass and auto-merge; external PRs request `costa92` review and block until approved at current HEAD |

`OWNERS` — `costa92` as both approver and reviewer, label `area/providers`. Kubernetes-style format.
`LICENSE` — MIT, 2026 costa92.

## scripts/ Contents

| Script | Purpose |
|--------|---------|
| `scripts/workspace.sh` | Writes a sibling-aware `<parent>/go.work` pointing at clones of `llm-agent`, `llm-agent-providers`, `llm-agent-otel`, `llm-agent-customer-support`. Idempotent. `go.work` is `.gitignore`d. |
| `scripts/capture-fixtures-openai.sh` | `curl` real OpenAI `/v1/chat/completions`, dump to `internal/contract/testdata/openai/generate_happy_gpt-4o-mini.json`. |
| `scripts/capture-fixtures-anthropic.sh` | Same pattern, hitting `https://api.anthropic.com/v1/messages` with `anthropic-version: 2023-06-01`. |
| `scripts/capture-fixtures-ollama.sh` | Local `OLLAMA_HOST` -> `/api/chat` with `"stream":false`. |

All three capture scripts use a Python `json` heredoc to normalize the fixture into the schema consumed by `internal/contract/contract.go:LoadFixture`.

---

*Stack analysis: 2026-05-20*
