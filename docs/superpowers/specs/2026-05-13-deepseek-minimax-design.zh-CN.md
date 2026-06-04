# DeepSeek + MiniMax 适配器设计

Date: 2026-05-13
Repo: `llm-agent-providers`
Status: proposed

## 目标

新增两个提供方适配器包：

- `deepseek`
- `minimax`

两个适配器都必须契合现有的 `llm-agent/llm` 提供方契约，并让人感觉它们是
`openai`、`anthropic` 与 `ollama` 的一等公民级兄弟。

## 范围

### 纳入范围

- `deepseek` 包
- `minimax` 包
- 用于国内端点与全球端点的区域预设
- 支持：
  - `Generate`
  - `Stream`
  - `WithTools`
  - `Info`
- 针对构造函数/配置、generate、stream 与工具行为的
  fixture 风格 `httptest` 覆盖
- 仓库级与包级的 README 更新

### 不纳入范围

- 嵌入（embeddings）
- 结构化输出（structured outputs）
- prompt 缓存（prompt caching）
- 针对 DeepSeek 或 MiniMax 的 live 集成测试
- 将现有的 `openai` / `anthropic` 适配器重构为共享的内部
  兼容层

## 设计选择

使用两个适配器包加上区域端点预设，而非四个独立的
包。

所选形态：

- `deepseek.New(opts ...Option)`
- `minimax.New(opts ...Option)`
- `WithRegion(RegionCN|RegionGlobal)`
- `WithBaseURL(...)` 覆盖区域预设

原因：

- 保持公共 API 精简
- 保留当前仓库风格：`provider.New(...)+options`
- 避免为国内与全球路由复制几乎完全相同的适配器

## 能力契约

适配器的能力必须真实。

### `deepseek`

- 实现 `llm.ChatModel`
- 实现 `llm.ToolCaller`
- **不** 实现 `llm.Embedder`
- `Info().Capabilities`：
  - `Tools=true`
  - `Embeddings=false`
  - `StructuredOutputs=false`
  - `PromptCaching=false`

### `minimax`

- 实现 `llm.ChatModel`
- 实现 `llm.ToolCaller`
- **不** 实现 `llm.Embedder`
- `Info().Capabilities`：
  - `Tools=true`
  - `Embeddings=false`
  - `StructuredOutputs=false`
  - `PromptCaching=false`

理由：

- DeepSeek 官方文档提供 OpenAI 兼容的 chat + 工具调用。
- MiniMax 官方文档提供 Anthropic 兼容的 chat + 工具调用。
- 本设计有意 **不** 声称支持嵌入，因为
  当前的实现轮次只在有据可查、高置信度的
  能力证据上推进。

## 协议映射

### DeepSeek

DeepSeek 遵循 OpenAI 兼容的表面。

实现策略：

- 将当前的 `openai` 适配器结构复制进专用的 `deepseek`
  包
- 保持请求/响应映射与 `openai` 包行为一致
- 保持流事件映射与 `openaiStreamReader` 一致
- 保持工具调用 delta 处理与现有 OpenAI 适配器一致

预期的包文件：

- `deepseek.go`
- `options.go`
- `map.go`
- `errors.go`
- `doc.go`
- `README.md`
- `deepseek_test.go`

### MiniMax

MiniMax 遵循 Anthropic 兼容的表面。

实现策略：

- 将当前的 `anthropic` 适配器结构复制进专用的 `minimax`
  包
- 保持请求/响应映射与 `anthropic` 包行为一致
- 保持流事件映射与 `anthropicStreamReader` 一致
- 保持工具调用 JSON delta 处理与现有 Anthropic
  适配器一致

预期的包文件：

- `minimax.go`
- `options.go`
- `map.go`
- `errors.go`
- `doc.go`
- `README.md`
- `minimax_test.go`

## 区域模型

每个包都暴露：

```go
type Region string

const (
    RegionCN     Region = "cn"
    RegionGlobal Region = "global"
)
```

每个包还暴露：

- `WithRegion(r Region)`
- `WithBaseURL(url string)`

优先级：

1. 显式的 `WithBaseURL(...)`
2. 显式的 `WithRegion(...)`
3. 包默认区域预设

### DeepSeek 端点预设

- `RegionGlobal` -> 有文档记载的全球 API 端点
- `RegionCN` -> 若官方确有区分则使用有文档记载的国内端点；否则
  实现可将两个区域映射到同一端点，但保持
  region API 稳定以备未来分化

### MiniMax 端点预设

- `RegionGlobal` -> 有文档记载的全球 Anthropic 兼容端点
- `RegionCN` -> 若官方确有区分则使用有文档记载的国内端点；否则
  实现可将两个区域映射到同一端点，但保持
  region API 稳定以备未来分化

如果某一提供方的官方文档实际上并未暴露不同的端点，
该包仍保留 region 选项，但记录两个预设
当前解析到同一 base URL。

## 配置

### DeepSeek 选项

- `WithModel(string)` 必需
- `WithAPIKey(string)` 可选；回退 env：
  - `DEEPSEEK_API_KEY`
- `WithRegion(Region)`
- `WithBaseURL(string)`
- `WithHTTPClient(*http.Client)`
- `WithTimeout(time.Duration)`

### MiniMax 选项

- `WithModel(string)` 必需
- `WithAPIKey(string)` 可选；回退 env：
  - `MINIMAX_API_KEY`
- `WithRegion(Region)`
- `WithBaseURL(string)`
- `WithHTTPClient(*http.Client)`
- `WithTimeout(time.Duration)`

首轮不加入组织 header 或提供方特定的可选旋钮，
除非它们是达到有文档记载的对等所必需的。

## 错误行为

每个包都获得各自的 `wrapErr` 路径，遵循与现有
提供方相同的设计：

- 在可能处保留类型化的 `llm` 错误行为
- 避免在暴露出的错误字符串中泄露 secret
- 将 network/auth/model-not-found 风格的错误映射为与兄弟适配器相同的
  用户级形态

本阶段不会尝试在整个仓库范围统一提供方的错误
包装。

## 测试策略

测试是本地的，并通过 `httptest.Server` 由 fixture 驱动。

### DeepSeek 测试

- `TestNew_RequiresModel`
- `TestInfo_DeepSeek`
- `TestGenerate_DeepSeek_Happy`
- `TestWithTools_DeepSeek_ImmutableAndIndependent`
- `TestStream_DeepSeek_Happy`
- `TestStream_DeepSeek_RetriesBeforeFirstByte`
- `TestRegionPreset_DeepSeek`
- `TestBaseURL_OverridesRegion_DeepSeek`

### MiniMax 测试

- `TestNew_RequiresModel`
- `TestInfo_MiniMax`
- `TestGenerate_MiniMax_Happy`
- `TestWithTools_MiniMax_ImmutableAndIndependent`
- `TestStream_MiniMax_Happy`
- `TestStream_MiniMax_RetriesBeforeFirstByte`
- `TestRegionPreset_MiniMax`
- `TestBaseURL_OverridesRegion_MiniMax`

### 校验命令

- `GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./deepseek ./minimax`
- `GOWORK=/tmp/phase7-v04-audit/go.work GOCACHE=/tmp/go-build go test ./...`

## 文档变更

更新：

- 仓库 `README.md`
  - 在已交付的提供方表面中纳入 DeepSeek 与 MiniMax
  - 加入安装示例
- `deepseek/README.md`
- `minimax/README.md`

各包 README 应明确声明：

- 复用的协议族（`OpenAI-compatible` 或 `Anthropic-compatible`）
- 区域预设行为
- 当前的能力真相（`Embeddings=false`）

## 风险

### 风险 1：虚假的「完整提供方」声称

缓解：

- 保持能力契约真实
- 在没有有文档记载的支持置信度时，不实现 `llm.Embedder`

### 风险 2：国内/全球的歧义

缓解：

- 现在就暴露稳定的 `Region` API
- 若官方文档并未真正分化，允许两个预设映射到同一 base URL
- 明确记录该行为

### 风险 3：意外的仓库级抽象重构

缓解：

- 本轮不引入共享兼容层
- 将改动约束在新增包加上 README 更新

## 推荐的实现顺序

1. DeepSeek 构造函数 + info + region 测试
2. DeepSeek generate + stream + tools
3. MiniMax 构造函数 + info + region 测试
4. MiniMax generate + stream + tools
5. README 更新
6. 完整仓库校验

## 成功标准

- `deepseek.New(...)` 与 `minimax.New(...)` 可编译，且行为与现有的
  一方适配器一致
- 两个提供方都支持 chat generate、stream 与 tools
- 区域预设存在且有测试覆盖
- 能力真实
- 全部测试通过，且不会在现有适配器中引入回归
