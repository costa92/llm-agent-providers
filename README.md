# llm-agent-providers

Provider adapters for [`github.com/costa92/llm-agent`](https://github.com/costa92/llm-agent) — OpenAI, Anthropic, and Ollama. Each adapter implements the capability interfaces from `llm-agent/llm` (`ChatModel`, `ToolCaller`, `Embedder`, `StructuredOutputs`).

> **v0.1.0-pre / Phase 0 skeleton.** Provider implementations land in Phases 1-4 per the [llm-agent ROADMAP](https://github.com/costa92/llm-agent/blob/main/.planning/ROADMAP.md). This repo currently contains only build infrastructure - no Go source files yet.
>
> **Expected CI status:** The first push CI run may fail on `go mod tidy` because `github.com/costa92/llm-agent v0.3.0-pre.1` does not exist until the core repo tags it at the end of Phase 0. This is intentional Phase-0 signal. Once the core repo cuts the `v0.3.0-pre.1` tag, sister-repo CI goes green automatically.

## Install

```bash
go get github.com/costa92/llm-agent-providers/openai@v0.1.0   # available after Phase 1
```

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

This repo tracks `v0.1.x` for the `llm-agent v0.3.x` cycle. Sister-repo bumps coordinate with core breaking changes; coordinated tags (Phase 7) advance both repos in lockstep.

## See also

- [`llm-agent` CLAUDE.md](https://github.com/costa92/llm-agent/blob/main/CLAUDE.md) — project hard rules (stdlib-only core, no K8s, capability per-(provider x model)).
- [`llm-agent` ROADMAP](https://github.com/costa92/llm-agent/blob/main/.planning/ROADMAP.md) — 8-phase v0.3 milestone plan.
- [`DEPRECATIONS.md`](https://github.com/costa92/llm-agent/blob/main/DEPRECATIONS.md) — symbols on the v0.4 removal track.

## License

MIT — see [LICENSE](LICENSE).
