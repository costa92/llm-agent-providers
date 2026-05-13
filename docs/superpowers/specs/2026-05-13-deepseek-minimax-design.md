# DeepSeek + MiniMax Adapter Design

Date: 2026-05-13
Repo: `llm-agent-providers`
Status: proposed

## Goal

Add two new provider adapter packages:

- `deepseek`
- `minimax`

Both adapters must fit the existing `llm-agent/llm` provider contract and feel
like first-class siblings of `openai`, `anthropic`, and `ollama`.

## Scope

### In scope

- `deepseek` package
- `minimax` package
- region presets for domestic vs global endpoints
- support for:
  - `Generate`
  - `Stream`
  - `WithTools`
  - `Info`
- fixture-style `httptest` coverage for constructor/config, generate, stream,
  and tool behavior
- README updates at repo level and package level

### Out of scope

- embeddings
- structured outputs
- prompt caching
- live integration tests against DeepSeek or MiniMax
- refactoring existing `openai` / `anthropic` adapters into shared internal
  compatibility layers

## Design Choice

Use two adapter packages plus regional endpoint presets, not four separate
 packages.

Chosen shape:

- `deepseek.New(opts ...Option)`
- `minimax.New(opts ...Option)`
- `WithRegion(RegionCN|RegionGlobal)`
- `WithBaseURL(...)` overrides region presets

Why:

- keeps the public API small
- preserves current repo style: `provider.New(...)+options`
- avoids duplicating nearly identical adapters for domestic vs global routing

## Capability Contract

The adapter capabilities must be truthful.

### `deepseek`

- implements `llm.ChatModel`
- implements `llm.ToolCaller`
- does **not** implement `llm.Embedder`
- `Info().Capabilities`:
  - `Tools=true`
  - `Embeddings=false`
  - `StructuredOutputs=false`
  - `PromptCaching=false`

### `minimax`

- implements `llm.ChatModel`
- implements `llm.ToolCaller`
- does **not** implement `llm.Embedder`
- `Info().Capabilities`:
  - `Tools=true`
  - `Embeddings=false`
  - `StructuredOutputs=false`
  - `PromptCaching=false`

Rationale:

- DeepSeek official docs provide OpenAI-compatible chat + tool calling.
- MiniMax official docs provide Anthropic-compatible chat + tool calling.
- This design intentionally does **not** claim embeddings support because the
  current implementation pass only proceeds on documented, high-confidence
  capability evidence.

## Protocol Mapping

### DeepSeek

DeepSeek follows an OpenAI-compatible surface.

Implementation strategy:

- copy the current `openai` adapter structure into a dedicated `deepseek`
  package
- keep request/response mapping aligned with `openai` package behavior
- keep stream event mapping aligned with `openaiStreamReader`
- keep tool-call delta handling aligned with the existing OpenAI adapter

Expected package files:

- `deepseek.go`
- `options.go`
- `map.go`
- `errors.go`
- `doc.go`
- `README.md`
- `deepseek_test.go`

### MiniMax

MiniMax follows an Anthropic-compatible surface.

Implementation strategy:

- copy the current `anthropic` adapter structure into a dedicated `minimax`
  package
- keep request/response mapping aligned with `anthropic` package behavior
- keep stream event mapping aligned with `anthropicStreamReader`
- keep tool-call JSON delta handling aligned with the existing Anthropic
  adapter

Expected package files:

- `minimax.go`
- `options.go`
- `map.go`
- `errors.go`
- `doc.go`
- `README.md`
- `minimax_test.go`

## Region Model

Each package exposes:

```go
type Region string

const (
    RegionCN     Region = "cn"
    RegionGlobal Region = "global"
)
```

Each package also exposes:

- `WithRegion(r Region)`
- `WithBaseURL(url string)`

Precedence:

1. explicit `WithBaseURL(...)`
2. explicit `WithRegion(...)`
3. package default region preset

### DeepSeek endpoint presets

- `RegionGlobal` -> documented global API endpoint
- `RegionCN` -> documented domestic endpoint if officially distinct; otherwise
  the implementation may map both regions to the same endpoint but keep the
  region API stable for future divergence

### MiniMax endpoint presets

- `RegionGlobal` -> documented global Anthropic-compatible endpoint
- `RegionCN` -> documented domestic endpoint if officially distinct; otherwise
  the implementation may map both regions to the same endpoint but keep the
  region API stable for future divergence

If official docs do not actually expose different endpoints for one provider,
the package still keeps the region option but documents that both presets
currently resolve to the same base URL.

## Configuration

### DeepSeek options

- `WithModel(string)` required
- `WithAPIKey(string)` optional; fallback env:
  - `DEEPSEEK_API_KEY`
- `WithRegion(Region)`
- `WithBaseURL(string)`
- `WithHTTPClient(*http.Client)`
- `WithTimeout(time.Duration)`

### MiniMax options

- `WithModel(string)` required
- `WithAPIKey(string)` optional; fallback env:
  - `MINIMAX_API_KEY`
- `WithRegion(Region)`
- `WithBaseURL(string)`
- `WithHTTPClient(*http.Client)`
- `WithTimeout(time.Duration)`

No organization header or provider-specific optional knobs are added in the
first pass unless they are required to reach documented parity.

## Error Behavior

Each package gets its own `wrapErr` path, following the same design as existing
providers:

- preserve typed `llm` error behavior where possible
- avoid leaking secrets in surfaced error strings
- keep network/auth/model-not-found style errors mapped into the same user-level
  shape as sibling adapters

This phase does not try to unify provider error wrapping across the whole repo.

## Testing Strategy

Tests are local and fixture-driven via `httptest.Server`.

### DeepSeek tests

- `TestNew_RequiresModel`
- `TestInfo_DeepSeek`
- `TestGenerate_DeepSeek_Happy`
- `TestWithTools_DeepSeek_ImmutableAndIndependent`
- `TestStream_DeepSeek_Happy`
- `TestStream_DeepSeek_RetriesBeforeFirstByte`
- `TestRegionPreset_DeepSeek`
- `TestBaseURL_OverridesRegion_DeepSeek`

### MiniMax tests

- `TestNew_RequiresModel`
- `TestInfo_MiniMax`
- `TestGenerate_MiniMax_Happy`
- `TestWithTools_MiniMax_ImmutableAndIndependent`
- `TestStream_MiniMax_Happy`
- `TestStream_MiniMax_RetriesBeforeFirstByte`
- `TestRegionPreset_MiniMax`
- `TestBaseURL_OverridesRegion_MiniMax`

### Verification commands

- `GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek ./minimax`
- `GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./...`

## Documentation Changes

Update:

- repo `README.md`
  - include DeepSeek and MiniMax in shipped provider surface
  - add install examples
- `deepseek/README.md`
- `minimax/README.md`

The package READMEs should explicitly state:

- protocol family reused (`OpenAI-compatible` or `Anthropic-compatible`)
- region preset behavior
- current capability truth (`Embeddings=false`)

## Risks

### Risk 1: fake "full provider" claim

Mitigation:

- keep capability contract truthful
- do not implement `llm.Embedder` without documented support confidence

### Risk 2: domestic/global ambiguity

Mitigation:

- expose stable `Region` API now
- allow both presets to map to the same base URL if official docs do not truly
  diverge
- document that behavior explicitly

### Risk 3: accidental repo-wide abstraction refactor

Mitigation:

- no shared compatibility layer in this pass
- constrain edits to new packages plus README updates

## Recommended Implementation Order

1. DeepSeek constructor + info + region tests
2. DeepSeek generate + stream + tools
3. MiniMax constructor + info + region tests
4. MiniMax generate + stream + tools
5. README updates
6. full repo verification

## Success Criteria

- `deepseek.New(...)` and `minimax.New(...)` compile and behave like existing
  first-party adapters
- both providers support chat generate, stream, and tools
- region presets exist and are test-covered
- capabilities are truthful
- all tests pass without regressions in existing adapters
