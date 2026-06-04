# DeepSeek + MiniMax 适配器实现计划

> **致 agent 化工作进程：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现本计划。各步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 加入一等公民级的 `deepseek` 与 `minimax` 提供方适配器，带区域预设以及真实的 chat/tool/stream 能力。

**架构：** `deepseek` 复刻现有的 `openai` 适配器形态，因为 DeepSeek 暴露的是 OpenAI 兼容的 chat 表面。`minimax` 复刻现有的 `anthropic` 适配器形态，因为 MiniMax 暴露的是 Anthropic 兼容的 chat 表面。两个包都暴露 `WithRegion(RegionCN|RegionGlobal)` 与 `WithBaseURL(...)`，并且都通过省略 `llm.Embedder` 来保持能力的真实性。

**技术栈：** Go 1.26、`openai-go/v3`、`anthropic-sdk-go`、`httptest`、现有的 `llm-agent/llm` 提供方接口

---

## 文件清单

### 新增文件

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

### 修改文件

- `README.md`

### 现有参考文件

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

### 任务 1：加入 DeepSeek 构造函数、区域模型与 info 契约

**文件：**
- 创建：`deepseek/deepseek.go`
- 创建：`deepseek/options.go`
- 创建：`deepseek/doc.go`
- 测试：`deepseek/deepseek_test.go`

- [ ] **步骤 1：编写会失败的 DeepSeek 构造函数与 info 测试**

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

- [ ] **步骤 2：运行 DeepSeek 测试并确认它们失败**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek -run 'Test(New_RequiresModel|Info_DeepSeek|RegionPreset_DeepSeek|BaseURL_OverridesRegion_DeepSeek)' -count=1`

预期：FAIL，原因为缺失的 package / 未定义符号。

- [ ] **步骤 3：实现最小化的 DeepSeek 构造函数与配置**

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

- [ ] **步骤 4：重新运行 DeepSeek 构造函数测试**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek -run 'Test(New_RequiresModel|Info_DeepSeek|RegionPreset_DeepSeek|BaseURL_OverridesRegion_DeepSeek)' -count=1`

预期：PASS

- [ ] **步骤 5：提交构造函数切片**

```bash
cd /tmp/llm-agent-providers
git add deepseek/deepseek.go deepseek/options.go deepseek/doc.go deepseek/deepseek_test.go
git commit -m "feat: add deepseek adapter skeleton"
```

### 任务 2：加入 DeepSeek 的 generate、stream、tools 与错误映射

**文件：**
- 修改：`deepseek/deepseek.go`
- 创建：`deepseek/map.go`
- 创建：`deepseek/errors.go`
- 修改：`deepseek/deepseek_test.go`

- [ ] **步骤 1：编写会失败的 DeepSeek 行为测试**

加入以 `openai/openai_test.go` 为蓝本的测试：

```go
func TestGenerate_DeepSeek_Happy(t *testing.T) { /* httptest /chat/completions */ }
func TestWithTools_DeepSeek_ImmutableAndIndependent(t *testing.T) { /* calc vs search */ }
func TestStream_DeepSeek_Happy(t *testing.T) { /* SSE text + done */ }
func TestStream_DeepSeek_RetriesBeforeFirstByte(t *testing.T) { /* one pre-first-byte failure, one success */ }
```

- [ ] **步骤 2：运行 DeepSeek 行为测试并确认它们失败**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek -run 'Test(Generate_DeepSeek_Happy|WithTools_DeepSeek_ImmutableAndIndependent|Stream_DeepSeek_Happy|Stream_DeepSeek_RetriesBeforeFirstByte)' -count=1`

预期：FAIL，原因为缺失方法 / 请求映射错误。

- [ ] **步骤 3：通过改写 OpenAI 请求路径来实现 DeepSeek**

加入等价于以下内容的方法与辅助函数：

```go
func (d *DeepSeek) Generate(ctx context.Context, req llm.Request) (llm.Response, error)
func (d *DeepSeek) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error)
func (d *DeepSeek) WithTools(tools []llm.Tool) (llm.ToolCaller, error)
```

使用与 `openai` 相同的结构化映射模式：

- 请求路径：`client.Chat.Completions.New`
- 流式路径：`client.Chat.Completions.NewStreaming`
- 流事件映射：text deltas、tool start、tool args delta、tool end、done usage
- 错误包装：克隆 `openai/errors.go` 的结构，并将前缀重命名为 `deepseek`

- [ ] **步骤 4：重新运行 DeepSeek 行为测试**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek -run 'Test(Generate_DeepSeek_Happy|WithTools_DeepSeek_ImmutableAndIndependent|Stream_DeepSeek_Happy|Stream_DeepSeek_RetriesBeforeFirstByte)' -count=1`

预期：PASS

- [ ] **步骤 5：提交 DeepSeek 行为切片**

```bash
cd /tmp/llm-agent-providers
git add deepseek/deepseek.go deepseek/map.go deepseek/errors.go deepseek/deepseek_test.go
git commit -m "feat: add deepseek chat and tool support"
```

### 任务 3：加入 MiniMax 构造函数、区域模型与 info 契约

**文件：**
- 创建：`minimax/minimax.go`
- 创建：`minimax/options.go`
- 创建：`minimax/doc.go`
- 测试：`minimax/minimax_test.go`

- [ ] **步骤 1：编写会失败的 MiniMax 构造函数与 info 测试**

```go
func TestNew_RequiresModel(t *testing.T) { /* minimax: WithModel is required */ }
func TestInfo_MiniMax(t *testing.T) { /* Provider=minimax, Tools=true, Embeddings=false */ }
func TestRegionPreset_MiniMax(t *testing.T) { /* baseURLForRegion not empty */ }
func TestBaseURL_OverridesRegion_MiniMax(t *testing.T) { /* explicit baseURL wins */ }
```

- [ ] **步骤 2：运行 MiniMax 构造函数测试并确认它们失败**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./minimax -run 'Test(New_RequiresModel|Info_MiniMax|RegionPreset_MiniMax|BaseURL_OverridesRegion_MiniMax)' -count=1`

预期：FAIL，原因为缺失的 package / 未定义符号。

- [ ] **步骤 3：实现最小化的 MiniMax 构造函数与配置**

使用与任务 1 相同的 config 形态，但：

- env 回退：`MINIMAX_API_KEY`
- 提供方名称：`minimax`
- base URL 预设根遵循 Anthropic 兼容端点

实现：

```go
type MiniMax struct {
	client *sdk.Client
	info   llm.ProviderInfo
	tools  []llm.Tool
}

func New(opts ...Option) (*MiniMax, error) { /* same pattern as anthropic.New */ }
func (m *MiniMax) Info() llm.ProviderInfo { return m.info }
```

- [ ] **步骤 4：重新运行 MiniMax 构造函数测试**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./minimax -run 'Test(New_RequiresModel|Info_MiniMax|RegionPreset_MiniMax|BaseURL_OverridesRegion_MiniMax)' -count=1`

预期：PASS

- [ ] **步骤 5：提交构造函数切片**

```bash
cd /tmp/llm-agent-providers
git add minimax/minimax.go minimax/options.go minimax/doc.go minimax/minimax_test.go
git commit -m "feat: add minimax adapter skeleton"
```

### 任务 4：加入 MiniMax 的 generate、stream、tools 与错误映射

**文件：**
- 修改：`minimax/minimax.go`
- 创建：`minimax/map.go`
- 创建：`minimax/errors.go`
- 修改：`minimax/minimax_test.go`

- [ ] **步骤 1：编写会失败的 MiniMax 行为测试**

加入以 `anthropic/anthropic_test.go` 为蓝本的测试：

```go
func TestGenerate_MiniMax_Happy(t *testing.T) { /* messages endpoint happy path */ }
func TestWithTools_MiniMax_ImmutableAndIndependent(t *testing.T) { /* tool isolation */ }
func TestStream_MiniMax_Happy(t *testing.T) { /* anthropic-style stream events */ }
func TestStream_MiniMax_RetriesBeforeFirstByte(t *testing.T) { /* retry before first byte */ }
```

- [ ] **步骤 2：运行 MiniMax 行为测试并确认它们失败**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./minimax -run 'Test(Generate_MiniMax_Happy|WithTools_MiniMax_ImmutableAndIndependent|Stream_MiniMax_Happy|Stream_MiniMax_RetriesBeforeFirstByte)' -count=1`

预期：FAIL，原因为缺失方法 / 协议映射错误。

- [ ] **步骤 3：通过改写 Anthropic 请求路径来实现 MiniMax**

加入等价于以下内容的方法：

```go
func (m *MiniMax) Generate(ctx context.Context, req llm.Request) (llm.Response, error)
func (m *MiniMax) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error)
func (m *MiniMax) WithTools(tools []llm.Tool) (llm.ToolCaller, error)
```

使用与 `anthropic` 相同的映射策略：

- 请求路径：`client.Messages.New`
- 流式路径：`client.Messages.NewStreaming`
- 流事件映射：text、tool start、partial JSON args、tool end、done usage
- 错误包装：克隆 `anthropic/errors.go` 的结构，并将前缀重命名为 `minimax`

- [ ] **步骤 4：重新运行 MiniMax 行为测试**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./minimax -run 'Test(Generate_MiniMax_Happy|WithTools_MiniMax_ImmutableAndIndependent|Stream_MiniMax_Happy|Stream_MiniMax_RetriesBeforeFirstByte)' -count=1`

预期：PASS

- [ ] **步骤 5：提交 MiniMax 行为切片**

```bash
cd /tmp/llm-agent-providers
git add minimax/minimax.go minimax/map.go minimax/errors.go minimax/minimax_test.go
git commit -m "feat: add minimax chat and tool support"
```

### 任务 5：更新文档并运行完整校验

**文件：**
- 修改：`README.md`
- 创建：`deepseek/README.md`
- 创建：`minimax/README.md`

- [ ] **步骤 1：编写文档更新**

更新根 `README.md`：

```md
- DeepSeek: Generate, Stream, Tool calling
- MiniMax: Generate, Stream, Tool calling
```

加入安装示例：

```bash
go get github.com/costa92/llm-agent-providers/deepseek@v0.1.1
go get github.com/costa92/llm-agent-providers/minimax@v0.1.1
```

各包 README 必须说明：

- 复用的协议族
- 区域预设形态
- `Embeddings=false`

- [ ] **步骤 2：对新增包运行定向校验**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek ./minimax -count=1`

预期：PASS

- [ ] **步骤 3：运行完整仓库校验**

运行：`cd /tmp/llm-agent-providers && GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./... -count=1`

预期：PASS

- [ ] **步骤 4：提交文档与校验切片**

```bash
cd /tmp/llm-agent-providers
git add README.md deepseek/README.md minimax/README.md
git commit -m "docs: add deepseek and minimax provider docs"
```

## 规格覆盖检查

- 双包设计：由任务 1-4 覆盖
- 区域预设 API：由任务 1 与 3 覆盖
- 真实的 `Embeddings=false`：由任务 1 与 3 的 info 测试覆盖
- generate/stream/tools 支持：由任务 2 与 4 覆盖
- README 与包文档：由任务 5 覆盖
