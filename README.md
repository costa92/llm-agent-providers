# llm-agent-providers

Provider adapters for
[`github.com/costa92/llm-agent`](https://github.com/costa92/llm-agent) —
OpenAI, Anthropic, Ollama, DeepSeek, and MiniMax. Each adapter implements the capability
interfaces from `llm-agent/llm` (`ChatModel`, `ToolCaller`, `Embedder`,
`StructuredOutputs`).

This repo ships the full provider surface verified against the
current `llm-agent` core contract:

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
go get github.com/costa92/llm-agent-providers/openai@latest
go get github.com/costa92/llm-agent-providers/anthropic@latest
go get github.com/costa92/llm-agent-providers/ollama@latest
go get github.com/costa92/llm-agent-providers/deepseek@latest
go get github.com/costa92/llm-agent-providers/minimax@latest
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

This repo is part of the broader `llm-agent-ecosystem`. The local helper script in this repo targets a common 4-repo development subset alongside [`llm-agent`](https://github.com/costa92/llm-agent), [`llm-agent-otel`](https://github.com/costa92/llm-agent-otel), and [`llm-agent-customer-support`](https://github.com/costa92/llm-agent-customer-support). For local development across that subset:

**Recommended for this subset:** clone all 4 repos as siblings, run `./scripts/workspace.sh` from any of them, then develop with a `go.work` file. The workspace file is `.gitignore`d in every repo:

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

This repo tracks the `llm-agent` core surface. Check `go.mod` for the
current exact sibling pins; cross-repo bump waves follow the umbrella's
"coordinated bump + re-tag" pattern (umbrella Phase 33).

## PR automation

This repo now expects `.github/workflows/pr-governance.yml` to enforce a simple policy:

- PRs authored by `costa92` should pass governance automatically and enable auto-merge after required checks pass.
- Same-repo owner branches should be deleted explicitly by that workflow after the PR is confirmed merged.
- PRs authored by anyone else should request review from `costa92` and stay blocked until `costa92` approves the current PR head.

This policy is designed to work with branch protection that requires the `go` and `governance` status checks, instead of GitHub's built-in required-approval gate.

The repo-level `deleteBranchOnMerge` setting remains enabled as a safety net, but the primary tested path is now inside `pr-governance.yml` itself: enable auto-merge, wait until the PR is visibly merged, then delete the same-repo head ref with the GitHub API. Standalone downstream cleanup workflows were tested during rollout and are no longer the documented primary mechanism.

The intended outcome is predictable: owner-authored maintenance PRs can land
without manual review friction, while external contributions still require an
explicit current-head approval from `costa92`.

That behavior is intentionally implemented in CI so the merge policy stays
visible, reproducible, and testable across repos.

The full multi-repo governance design, including the relationship between
`llm-agent`, `llm-agent-rag`, `llm-agent-flow`, `llm-agent-providers`,
`llm-agent-otel`, and `llm-agent-customer-support`, lives in the core repo docs:

- [`PR-GOVERNANCE-OVERVIEW.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-OVERVIEW.md)
- [`PR-GOVERNANCE-PROJECTS.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-PROJECTS.md)
- [`PR-GOVERNANCE-RULES.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-RULES.md)
- [`PR-GOVERNANCE-OPERATIONS.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-OPERATIONS.md)

## See also

- [`llm-agent` CLAUDE.md](https://github.com/costa92/llm-agent/blob/main/CLAUDE.md) — project hard rules (stdlib-only core, no K8s, capability per-(provider x model)).
- [`llm-agent` ROADMAP](https://github.com/costa92/llm-agent/blob/main/.planning/ROADMAP.md) — 8-phase v0.3 milestone plan.
- [`DEPRECATIONS.md`](https://github.com/costa92/llm-agent/blob/main/DEPRECATIONS.md) — current `llm-agent` removal track.

## License

MIT — see [LICENSE](LICENSE).
