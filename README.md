# llm-agent-providers

Provider adapters for
[`github.com/costa92/llm-agent`](https://github.com/costa92/llm-agent) —
OpenAI, Anthropic, Ollama, DeepSeek, and MiniMax. Each adapter implements the capability
interfaces from `llm-agent/llm` (`ChatModel`, `ToolCaller`, `Embedder`,
`StructuredOutputs`).

This repo now ships the full Phase 1-4 provider surface and has been verified
against the post-compat-removal `llm-agent` `v0.4` core:

- OpenAI: Generate, Stream, Tool calling, Embeddings
- Anthropic: Generate, Stream, Tool calling, explicit `ErrNotSupported` for
  embeddings
- Ollama: Generate, Stream, model-aware Tool calling, Embeddings
- DeepSeek: Generate, Stream, Tool calling
- MiniMax: Generate, Stream, Tool calling
- shared `internal/contract` conformance coverage in-repo
- nightly Ollama live CI coverage

## Install

```bash
go get github.com/costa92/llm-agent-providers/openai@v0.1.0
go get github.com/costa92/llm-agent-providers/anthropic@v0.1.0
go get github.com/costa92/llm-agent-providers/ollama@v0.1.0
go get github.com/costa92/llm-agent-providers/deepseek@v0.1.0
go get github.com/costa92/llm-agent-providers/minimax@v0.1.0
```

## Shipped provider surface

### OpenAI

- `openai.New(...)` bound-model adapter
- sync generate
- streaming with usage capture
- native function/tool calling
- embeddings

### Anthropic

- `anthropic.New(...)` bound-model adapter
- sync generate
- streaming
- native tool use
- documented embedding gap via `llm.ErrNotSupported`

### Ollama

- `ollama.New(...)` bound-model adapter
- sync generate
- streaming
- per-model tool strategy support
- embeddings

### DeepSeek

- `deepseek.New(...)` bound-model adapter
- sync generate
- streaming with usage capture
- native function/tool calling
- regional endpoint presets via `WithRegion(...)`

### MiniMax

- `minimax.New(...)` bound-model adapter
- sync generate
- streaming
- native tool use
- regional endpoint presets via `WithRegion(...)`

### Conformance

- shared fixture-driven contract coverage in `internal/contract`
- cross-provider generate/stream/tool/embed assertions
- nightly live Ollama verification path

## Cross-repo iteration pattern (INFRA-06)

This repo lives in a 4-repo umbrella alongside [`llm-agent`](https://github.com/costa92/llm-agent), [`llm-agent-otel`](https://github.com/costa92/llm-agent-otel), and [`llm-agent-customer-support`](https://github.com/costa92/llm-agent-customer-support). For local development across repos:

**Recommended:** clone all 4 repos as siblings, run `./scripts/workspace.sh` from any of them, then develop with a `go.work` file. The workspace file is `.gitignore`d in every repo:

```bash
cd <parent>
git clone https://github.com/costa92/llm-agent.git
git clone https://github.com/costa92/llm-agent-providers.git
git clone https://github.com/costa92/llm-agent-otel.git
git clone https://github.com/costa92/llm-agent-customer-support.git
cd llm-agent-providers
./scripts/workspace.sh    # writes ../go.work pointing at all 4 sibling clones
go build ./...            # now resolves llm-agent against the local sibling
```

**Escape hatch (NEVER on tagged-release branches):** for one-off iteration without `go.work`, you can use `replace`:

```bash
go mod edit -replace=github.com/costa92/llm-agent=../llm-agent
```

The `release-precheck` CI workflow rejects any non-empty `replace` block on branches matching `release/**`. Don't tag from a branch with `replace` directives — INFRA-04.

## Versioning

This repo is already code-compatible with the `llm-agent v0.4` core surface.
Its local release-prep state now targets `github.com/costa92/llm-agent v0.4.0`.
The only remaining Phase 7 follow-up is publishing the final coordinated tags.

## PR automation

This repo now expects `.github/workflows/pr-governance.yml` to enforce a simple policy:

- PRs authored by `costa92` should pass governance automatically and enable auto-merge after required checks pass.
- PRs authored by anyone else should request review from `costa92` and stay blocked until `costa92` approves the current PR head.

This policy is designed to work with branch protection that requires the `go` and `governance` status checks, instead of GitHub's built-in required-approval gate.

The intended outcome is predictable: owner-authored maintenance PRs can land
without manual review friction, while external contributions still require an
explicit current-head approval from `costa92`.

That behavior is intentionally implemented in CI so the merge policy stays
visible, reproducible, and testable across repos.

The full multi-repo governance design, including the relationship between
`llm-agent`, `llm-agent-providers`, `llm-agent-otel`, and
`llm-agent-customer-support`, lives in the core repo docs:

- [`PR-GOVERNANCE-OVERVIEW.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-OVERVIEW.md)
- [`PR-GOVERNANCE-PROJECTS.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-PROJECTS.md)
- [`PR-GOVERNANCE-RULES.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-RULES.md)
- [`PR-GOVERNANCE-OPERATIONS.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-OPERATIONS.md)

## See also

- [`llm-agent` CLAUDE.md](https://github.com/costa92/llm-agent/blob/main/CLAUDE.md) — project hard rules (stdlib-only core, no K8s, capability per-(provider x model)).
- [`llm-agent` ROADMAP](https://github.com/costa92/llm-agent/blob/main/.planning/ROADMAP.md) — 8-phase v0.3 milestone plan.
- [`DEPRECATIONS.md`](https://github.com/costa92/llm-agent/blob/main/DEPRECATIONS.md) — symbols on the v0.4 removal track.

## License

MIT — see [LICENSE](LICENSE).
