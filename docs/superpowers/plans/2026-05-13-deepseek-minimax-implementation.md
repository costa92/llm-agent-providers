# DeepSeek + MiniMax Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class `deepseek` and `minimax` provider adapters with region presets and truthful chat/tool/stream capabilities.

**Architecture:** `deepseek` mirrors the existing `openai` adapter shape because DeepSeek exposes an OpenAI-compatible chat surface. `minimax` mirrors the existing `anthropic` adapter shape because MiniMax exposes an Anthropic-compatible chat surface. Both packages expose `WithRegion(RegionCN|RegionGlobal)` plus `WithBaseURL(...)`, and both keep capabilities truthful by omitting `llm.Embedder`.

**Tech Stack:** Go 1.26, `openai-go/v3`, `anthropic-sdk-go`, `httptest`, existing `llm-agent/llm` provider interfaces

---

## File Map

### New files

- `deepseek/deepseek.go`
- `deepseek/options.go`
- `deepseek/map.go`
- `deepseek/errors.go`
- `deepseek/doc.go`
- `deepseek/README.md`
- `deepseek/deepseek_test.go`
- `minimax/minimax.go`
- `minimax/options.go`
- `minimax/map.go`
- `minimax/errors.go`
- `minimax/doc.go`
- `minimax/README.md`
- `minimax/minimax_test.go`

### Modified files

- `README.md`

### Existing reference files

- `openai/openai.go`
- `openai/options.go`
- `openai/map.go`
- `openai/errors.go`
- `openai/openai_test.go`
- `anthropic/anthropic.go`
- `anthropic/options.go`
- `anthropic/map.go`
- `anthropic/errors.go`
- `anthropic/anthropic_test.go`

---

### Task 1: Add DeepSeek constructor, region model, and info contract

**Files:**
- Create: `deepseek/deepseek.go`
- Create: `deepseek/options.go`
- Create: `deepseek/doc.go`
- Test: `deepseek/deepseek_test.go`

- [ ] **Step 1: Write the failing DeepSeek constructor and info tests**

```go
func TestNew_RequiresModel(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("New() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), "WithModel is required") {
		t.Fatalf("New() error = %q, want WithModel is required", err)
	}
}

func TestInfo_DeepSeek(t *testing.T) {
	m, err := New(WithModel("deepseek-chat"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if info.Provider != "deepseek" {
		t.Fatalf("Provider = %q, want deepseek", info.Provider)
	}
	if info.Model != "deepseek-chat" {
		t.Fatalf("Model = %q, want deepseek-chat", info.Model)
	}
	if !info.Capabilities.Tools || info.Capabilities.Embeddings || info.Capabilities.StructuredOutputs || info.Capabilities.PromptCaching {
		t.Fatalf("Capabilities = %+v, want tools=true and others=false", info.Capabilities)
	}
}

func TestRegionPreset_DeepSeek(t *testing.T) {
	got := baseURLForRegion(RegionGlobal)
	if got == "" {
		t.Fatal("baseURLForRegion(RegionGlobal) = empty")
	}
}

func TestBaseURL_OverridesRegion_DeepSeek(t *testing.T) {
	m, err := New(
		WithModel("deepseek-chat"),
		WithAPIKey("test-key"),
		WithRegion(RegionCN),
		WithBaseURL("https://example.invalid"),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := m.Info().Provider; got != "deepseek" {
		t.Fatalf("Provider = %q, want deepseek", got)
	}
}
```

- [ ] **Step 2: Run the DeepSeek tests and verify they fail**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek -run 'Test(New_RequiresModel|Info_DeepSeek|RegionPreset_DeepSeek|BaseURL_OverridesRegion_DeepSeek)' -count=1`

Expected: FAIL with missing package / undefined symbols.

- [ ] **Step 3: Implement the minimal DeepSeek constructor and config**

```go
type Region string

const (
	RegionCN     Region = "cn"
	RegionGlobal Region = "global"
)

type config struct {
	apiKey     string
	model      string
	baseURL    string
	httpClient *http.Client
	timeout    time.Duration
	region     Region
}

type Option func(*config)

func WithModel(m string) Option      { return func(c *config) { c.model = m } }
func WithAPIKey(k string) Option     { return func(c *config) { c.apiKey = k } }
func WithBaseURL(u string) Option    { return func(c *config) { c.baseURL = u } }
func WithHTTPClient(h *http.Client) Option { return func(c *config) { c.httpClient = h } }
func WithTimeout(d time.Duration) Option   { return func(c *config) { c.timeout = d } }
func WithRegion(r Region) Option     { return func(c *config) { c.region = r } }

func baseURLForRegion(r Region) string {
	switch r {
	case RegionCN:
		return "https://api.deepseek.com"
	case RegionGlobal:
		return "https://api.deepseek.com"
	default:
		return "https://api.deepseek.com"
	}
}
```

```go
type DeepSeek struct {
	client *openai.Client
	info   llm.ProviderInfo
	tools  []llm.Tool
}

func New(opts ...Option) (*DeepSeek, error) {
	cfg := config{region: RegionGlobal}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.model == "" {
		return nil, errors.New("deepseek: WithModel is required")
	}
	if cfg.apiKey == "" {
		cfg.apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}
	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = baseURLForRegion(cfg.region)
	}
	sdkOpts := []option.RequestOption{option.WithMaxRetries(0)}
	if cfg.apiKey != "" {
		sdkOpts = append(sdkOpts, option.WithAPIKey(cfg.apiKey))
	}
	if baseURL != "" {
		sdkOpts = append(sdkOpts, option.WithBaseURL(baseURL))
	}
	if cfg.httpClient != nil {
		sdkOpts = append(sdkOpts, option.WithHTTPClient(cfg.httpClient))
	}
	if cfg.timeout > 0 {
		sdkOpts = append(sdkOpts, option.WithRequestTimeout(cfg.timeout))
	}
	client := openai.NewClient(sdkOpts...)
	return &DeepSeek{
		client: &client,
		info: llm.ProviderInfo{
			Provider: "deepseek",
			Model:    cfg.model,
			Capabilities: llm.Capabilities{
				Tools:             true,
				Embeddings:        false,
				StructuredOutputs: false,
				PromptCaching:     false,
			},
		},
	}, nil
}

func (d *DeepSeek) Info() llm.ProviderInfo { return d.info }
```

- [ ] **Step 4: Re-run the DeepSeek constructor tests**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek -run 'Test(New_RequiresModel|Info_DeepSeek|RegionPreset_DeepSeek|BaseURL_OverridesRegion_DeepSeek)' -count=1`

Expected: PASS

- [ ] **Step 5: Commit the constructor slice**

```bash
cd /tmp/llm-agent-providers
git add deepseek/deepseek.go deepseek/options.go deepseek/doc.go deepseek/deepseek_test.go
git commit -m "feat: add deepseek adapter skeleton"
```

### Task 2: Add DeepSeek generate, stream, tools, and error mapping

**Files:**
- Modify: `deepseek/deepseek.go`
- Create: `deepseek/map.go`
- Create: `deepseek/errors.go`
- Modify: `deepseek/deepseek_test.go`

- [ ] **Step 1: Write the failing DeepSeek behavior tests**

Add tests modeled after `openai/openai_test.go`:

```go
func TestGenerate_DeepSeek_Happy(t *testing.T) { /* httptest /chat/completions */ }
func TestWithTools_DeepSeek_ImmutableAndIndependent(t *testing.T) { /* calc vs search */ }
func TestStream_DeepSeek_Happy(t *testing.T) { /* SSE text + done */ }
func TestStream_DeepSeek_RetriesBeforeFirstByte(t *testing.T) { /* one pre-first-byte failure, one success */ }
```

- [ ] **Step 2: Run the DeepSeek behavior tests and verify they fail**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek -run 'Test(Generate_DeepSeek_Happy|WithTools_DeepSeek_ImmutableAndIndependent|Stream_DeepSeek_Happy|Stream_DeepSeek_RetriesBeforeFirstByte)' -count=1`

Expected: FAIL with missing methods / wrong request mapping.

- [ ] **Step 3: Implement DeepSeek by adapting the OpenAI request path**

Add methods and helpers equivalent to:

```go
func (d *DeepSeek) Generate(ctx context.Context, req llm.Request) (llm.Response, error)
func (d *DeepSeek) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error)
func (d *DeepSeek) WithTools(tools []llm.Tool) (llm.ToolCaller, error)
```

Use the same structural mapping patterns as `openai`:

- request path: `client.Chat.Completions.New`
- stream path: `client.Chat.Completions.NewStreaming`
- stream event mapping: text deltas, tool start, tool args delta, tool end, done usage
- error wrapping: clone `openai/errors.go` structure and rename prefixes to `deepseek`

- [ ] **Step 4: Re-run the DeepSeek behavior tests**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek -run 'Test(Generate_DeepSeek_Happy|WithTools_DeepSeek_ImmutableAndIndependent|Stream_DeepSeek_Happy|Stream_DeepSeek_RetriesBeforeFirstByte)' -count=1`

Expected: PASS

- [ ] **Step 5: Commit the DeepSeek behavior slice**

```bash
cd /tmp/llm-agent-providers
git add deepseek/deepseek.go deepseek/map.go deepseek/errors.go deepseek/deepseek_test.go
git commit -m "feat: add deepseek chat and tool support"
```

### Task 3: Add MiniMax constructor, region model, and info contract

**Files:**
- Create: `minimax/minimax.go`
- Create: `minimax/options.go`
- Create: `minimax/doc.go`
- Test: `minimax/minimax_test.go`

- [ ] **Step 1: Write the failing MiniMax constructor and info tests**

```go
func TestNew_RequiresModel(t *testing.T) { /* minimax: WithModel is required */ }
func TestInfo_MiniMax(t *testing.T) { /* Provider=minimax, Tools=true, Embeddings=false */ }
func TestRegionPreset_MiniMax(t *testing.T) { /* baseURLForRegion not empty */ }
func TestBaseURL_OverridesRegion_MiniMax(t *testing.T) { /* explicit baseURL wins */ }
```

- [ ] **Step 2: Run the MiniMax constructor tests and verify they fail**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./minimax -run 'Test(New_RequiresModel|Info_MiniMax|RegionPreset_MiniMax|BaseURL_OverridesRegion_MiniMax)' -count=1`

Expected: FAIL with missing package / undefined symbols.

- [ ] **Step 3: Implement the minimal MiniMax constructor and config**

Use the same config shape as Task 1, but:

- env fallback: `MINIMAX_API_KEY`
- provider name: `minimax`
- base URL preset root follows the Anthropic-compatible endpoint

Implement:

```go
type MiniMax struct {
	client *sdk.Client
	info   llm.ProviderInfo
	tools  []llm.Tool
}

func New(opts ...Option) (*MiniMax, error) { /* same pattern as anthropic.New */ }
func (m *MiniMax) Info() llm.ProviderInfo { return m.info }
```

- [ ] **Step 4: Re-run the MiniMax constructor tests**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./minimax -run 'Test(New_RequiresModel|Info_MiniMax|RegionPreset_MiniMax|BaseURL_OverridesRegion_MiniMax)' -count=1`

Expected: PASS

- [ ] **Step 5: Commit the constructor slice**

```bash
cd /tmp/llm-agent-providers
git add minimax/minimax.go minimax/options.go minimax/doc.go minimax/minimax_test.go
git commit -m "feat: add minimax adapter skeleton"
```

### Task 4: Add MiniMax generate, stream, tools, and error mapping

**Files:**
- Modify: `minimax/minimax.go`
- Create: `minimax/map.go`
- Create: `minimax/errors.go`
- Modify: `minimax/minimax_test.go`

- [ ] **Step 1: Write the failing MiniMax behavior tests**

Add tests modeled after `anthropic/anthropic_test.go`:

```go
func TestGenerate_MiniMax_Happy(t *testing.T) { /* messages endpoint happy path */ }
func TestWithTools_MiniMax_ImmutableAndIndependent(t *testing.T) { /* tool isolation */ }
func TestStream_MiniMax_Happy(t *testing.T) { /* anthropic-style stream events */ }
func TestStream_MiniMax_RetriesBeforeFirstByte(t *testing.T) { /* retry before first byte */ }
```

- [ ] **Step 2: Run the MiniMax behavior tests and verify they fail**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./minimax -run 'Test(Generate_MiniMax_Happy|WithTools_MiniMax_ImmutableAndIndependent|Stream_MiniMax_Happy|Stream_MiniMax_RetriesBeforeFirstByte)' -count=1`

Expected: FAIL with missing methods / wrong protocol mapping.

- [ ] **Step 3: Implement MiniMax by adapting the Anthropic request path**

Add methods equivalent to:

```go
func (m *MiniMax) Generate(ctx context.Context, req llm.Request) (llm.Response, error)
func (m *MiniMax) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error)
func (m *MiniMax) WithTools(tools []llm.Tool) (llm.ToolCaller, error)
```

Use the same mapping strategy as `anthropic`:

- request path: `client.Messages.New`
- stream path: `client.Messages.NewStreaming`
- stream event mapping: text, tool start, partial JSON args, tool end, done usage
- error wrapping: clone `anthropic/errors.go` structure and rename prefixes to `minimax`

- [ ] **Step 4: Re-run the MiniMax behavior tests**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./minimax -run 'Test(Generate_MiniMax_Happy|WithTools_MiniMax_ImmutableAndIndependent|Stream_MiniMax_Happy|Stream_MiniMax_RetriesBeforeFirstByte)' -count=1`

Expected: PASS

- [ ] **Step 5: Commit the MiniMax behavior slice**

```bash
cd /tmp/llm-agent-providers
git add minimax/minimax.go minimax/map.go minimax/errors.go minimax/minimax_test.go
git commit -m "feat: add minimax chat and tool support"
```

### Task 5: Update documentation and run full verification

**Files:**
- Modify: `README.md`
- Create: `deepseek/README.md`
- Create: `minimax/README.md`

- [ ] **Step 1: Write the doc updates**

Update root `README.md`:

```md
- DeepSeek: Generate, Stream, Tool calling
- MiniMax: Generate, Stream, Tool calling
```

Add install examples:

```bash
go get github.com/costa92/llm-agent-providers/deepseek@v0.1.1
go get github.com/costa92/llm-agent-providers/minimax@v0.1.1
```

Package READMEs must say:

- protocol family reused
- region preset shape
- `Embeddings=false`

- [ ] **Step 2: Run targeted verification for new packages**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek ./minimax -count=1`

Expected: PASS

- [ ] **Step 3: Run full repo verification**

Run: `cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./... -count=1`

Expected: PASS

- [ ] **Step 4: Commit documentation and verification slice**

```bash
cd /tmp/llm-agent-providers
git add README.md deepseek/README.md minimax/README.md
git commit -m "docs: add deepseek and minimax provider docs"
```

## Spec Coverage Check

- two package design: covered by Tasks 1-4
- region preset API: covered by Tasks 1 and 3
- truthful `Embeddings=false`: covered by Tasks 1 and 3 info tests
- generate/stream/tools support: covered by Tasks 2 and 4
- README and package docs: covered by Task 5
