# llm-agent-providers

[English](./README.md) | [简体中文](./README.zh-CN.md)

面向
[`github.com/costa92/llm-agent`](https://github.com/costa92/llm-agent) 的提供方适配器 ——
OpenAI、Anthropic、Ollama、DeepSeek 与 MiniMax。每个适配器都实现了
`llm-agent/llm` 中的能力接口（`ChatModel`、`ToolCaller`、`Embedder`、
`StructuredOutputs`）。

本仓库交付完整的提供方表面，并已针对当前 `llm-agent` 核心契约完成校验：

- OpenAI：Generate、Stream、工具调用、嵌入
- Anthropic：Generate、Stream、工具调用，嵌入显式返回 `ErrNotSupported`
- Ollama：Generate、Stream、模型感知的工具调用、嵌入
- DeepSeek：Generate、Stream、工具调用
- MiniMax：Generate、Stream、工具调用
- 仓内共享的 `internal/contract` 一致性测试覆盖
- nightly Ollama live CI 覆盖

## 安装

```bash
go get github.com/costa92/llm-agent-providers/openai@latest
go get github.com/costa92/llm-agent-providers/anthropic@latest
go get github.com/costa92/llm-agent-providers/ollama@latest
go get github.com/costa92/llm-agent-providers/deepseek@latest
go get github.com/costa92/llm-agent-providers/minimax@latest
```

## 已交付的提供方表面

### OpenAI

- `openai.New(...)` 模型绑定适配器
- 同步 generate
- 带用量捕获的流式
- 原生函数/工具调用
- 嵌入

### Anthropic

- `anthropic.New(...)` 模型绑定适配器
- 同步 generate
- 流式
- 原生 tool use
- 通过 `llm.ErrNotSupported` 记录的嵌入缺口

### Ollama

- `ollama.New(...)` 模型绑定适配器
- 同步 generate
- 流式
- 按模型区分的工具策略支持
- 嵌入

### DeepSeek

- `deepseek.New(...)` 模型绑定适配器
- 同步 generate
- 带用量捕获的流式
- 原生函数/工具调用
- 通过 `WithRegion(...)` 提供的区域端点预设

### MiniMax

- `minimax.New(...)` 模型绑定适配器
- 同步 generate
- 流式
- 原生 tool use
- 通过 `WithRegion(...)` 提供的区域端点预设

### 一致性（Conformance）

- `internal/contract` 中由 fixture 驱动的共享契约覆盖
- 跨提供方的 generate/stream/tool/embed 断言
- nightly 的 Ollama live 校验路径

## 跨仓迭代模式（INFRA-06）

本仓库是更广阔的 `llm-agent-ecosystem` 的一部分。仓内的本地辅助脚本面向一个常见的 4 仓开发子集，与 [`llm-agent`](https://github.com/costa92/llm-agent)、[`llm-agent-otel`](https://github.com/costa92/llm-agent-otel) 和 [`llm-agent-customer-support`](https://github.com/costa92/llm-agent-customer-support) 协同。在该子集范围内进行本地开发：

**该子集推荐方式：** 将全部 4 个仓库作为兄弟仓克隆，在其中任意一个仓库下运行 `./scripts/workspace.sh`，然后使用 `go.work` 文件进行开发。该 workspace 文件在每个仓库中都被 `.gitignore` 忽略：

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

**逃生舱（在已打 tag 的发布分支上绝不使用）：** 若需在没有 `go.work` 的情况下做一次性迭代，可以使用 `replace`：

```bash
go mod edit -replace=github.com/costa92/llm-agent=../llm-agent
```

`release-precheck` CI workflow 会拒绝在匹配 `release/**` 的分支上出现任何非空的 `replace` 块。不要从带有 `replace` 指令的分支打 tag —— INFRA-04。

## 版本管理

本仓库跟踪 `llm-agent` 核心表面。当前确切的兄弟仓锚定请查看 `go.mod`；跨仓版本提升波次遵循伞形仓库的「协调式版本提升 + 重新打 tag」模式（umbrella Phase 33）。

## PR 自动化

本仓库现在期望由 `.github/workflows/pr-governance.yml` 强制执行一项简单策略：

- 由 `costa92` 提交的 PR 应自动通过治理，并在必需的检查通过后启用自动合并。
- 同仓的 owner 分支应在确认 PR 合并后由该 workflow 显式删除。
- 由其他任何人提交的 PR 应请求 `costa92` 评审，并在 `costa92` 批准当前 PR head 之前保持阻塞。

此策略设计为配合分支保护工作，由分支保护要求 `go` 与 `governance` 状态检查，而非使用 GitHub 内置的必需审批门禁。

仓库级的 `deleteBranchOnMerge` 设置作为安全网保持启用，但当前主要经过测试的路径已在 `pr-governance.yml` 内部：启用自动合并，等待 PR 可见地合并，然后通过 GitHub API 删除同仓的 head ref。独立的下游清理 workflow 在推广期间经过测试，已不再是文档记载的主要机制。

预期结果是可预测的：owner 提交的维护类 PR 无需人工评审摩擦即可落地，而外部贡献仍需 `costa92` 对当前 head 给出明确批准。

该行为有意在 CI 中实现，以使合并策略在各仓之间保持可见、可复现、可测试。

完整的多仓治理设计，包括 `llm-agent`、`llm-agent-rag`、`llm-agent-flow`、`llm-agent-providers`、`llm-agent-otel` 与 `llm-agent-customer-support` 之间的关系，位于核心仓库文档中：

- [`PR-GOVERNANCE-OVERVIEW.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-OVERVIEW.zh-CN.md)
- [`PR-GOVERNANCE-PROJECTS.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-PROJECTS.zh-CN.md)
- [`PR-GOVERNANCE-RULES.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-RULES.zh-CN.md)
- [`PR-GOVERNANCE-OPERATIONS.md`](https://github.com/costa92/llm-agent/blob/main/docs/PR-GOVERNANCE-OPERATIONS.zh-CN.md)

## 另见

- [`llm-agent` CLAUDE.md](https://github.com/costa92/llm-agent/blob/main/CLAUDE.zh-CN.md) —— 项目硬性规则（仅标准库核心、不使用 K8s、按 (provider x model) 区分能力）。
- [`llm-agent` ROADMAP](https://github.com/costa92/llm-agent/blob/main/.planning/ROADMAP.md) —— 8 阶段 v0.3 里程碑计划。
- [`DEPRECATIONS.md`](https://github.com/costa92/llm-agent/blob/main/DEPRECATIONS.zh-CN.md) —— 当前 `llm-agent` 的移除跟踪。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
