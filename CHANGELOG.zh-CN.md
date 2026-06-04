# Changelog

llm-agent-providers 的所有重要变更均记录于此。格式大致遵循
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，本项目遵循
[Semantic Versioning](https://semver.org/)。

## [Unreleased]

暂无。

## [v0.3.1] - 2026-06-03

### Fixed

- **ollama**：qwen 在消息内容中以 ```` ```json ```` markdown 围栏块返回的工具调用，
  现在会在 `parseQwenToolCalls` 进入裸 JSON 回退之前先被解包，因此
  `qwen2.5-coder` / `qwen3-coder` 的工具调用会被抽取并执行，而非被静默丢弃（#38）。
  `llama3.1`（原生 `tool_calls`）不受影响。已通过 `llm-agent` 的 `10-ollama-tools`
  示例针对 live 的 `qwen2.5-coder` 与 `llama3.1` 完成端到端校验。

### Changed

- 将 `llm-agent-contract` 锚定至 v0.2.0（#36）。

## [v0.3.0] - 2026-06-03

### Changed

- 完成 `llm-agent-contract` 解耦：每个提供方都直接从
  `github.com/costa92/llm-agent-contract` 导入契约类型，锚定在
  v0.1.0 且没有任何 `replace` 指令。这是横跨全部五个提供方的纯导入路径迁移；
  无行为或公共面变更。

## [v0.2.5] - 2026-05-23

K1 基石 —— ollama 经由跨提供方一致性门禁从 YELLOW 转为 GREEN。仅测试 + 文档；
零生产代码变更。

### Added (K1 keystone — closes ollama YELLOW)

- `internal/contract.AssertStreamToolCalls` —— 面向「流式 + 工具调用」的 K1
  一致性断言。锚定每次调用的 `Index` 在
  `EventToolCallStart` → `EventToolCallArgsDelta`(s) → `EventToolCallEnd`
  全程保持稳定，并确保 `EventDone` 为终止事件且已填充 `Usage` 与
  `FinishReason`。新增的 `Fixture.Expect.StreamSequence []string` 字段编码
  期望的 `Kind` 名称序列（text / tool_start / tool_args /
  tool_end / thinking / done）。
- `TestStreamToolCalls_Conformance` —— 跨提供方的 K1 门禁，覆盖
  5/5 提供方（6 个用例：openai / anthropic / ollama-native /
  ollama-qwen-xml / deepseek / minimax）。Fixtures 逐字派生自
  已校验的各提供方「流式工具调用」测试线缆形状
  （`openai/openai_test.go:333`、`anthropic/anthropic_test.go:196`、
  `ollama/ollama_test.go:609`、`ollama/ollama_test.go:670`、
  `deepseek/deepseek_test.go:359`、`minimax/minimax_test.go:220`）。
- ollama 的 K1 基石从 YELLOW 翻转为 GREEN。该分类已陈旧 ——
  `ollama/ollama.go:211-327` 的 emission 代码自提交 `32f5d59`
  （2026-05-20）起就已符合 K1，对原生 `tool_calls` 字段路径和
  内容解析的 `<tool_call>...</tool_call>` 路径都发出完整的
  `EventToolCallStart/ArgsDelta/End` 三元组，且每次调用的 `Index`
  稳定。三份仍声称 YELLOW 的规划文档早于该提交，现已刷新
  （`.planning/codebase/CONCERNS.md`、`.planning/codebase/ARCHITECTURE.md`、
  `.planning/codebase/TESTING.md`）；伞形仓库的
  `docs/ecosystem-design-review.zh-CN.md` 在配对的伞形仓提交中更新。
  基石记分卡现读作 **12 GREEN / 0 YELLOW / 0 RED**。
- 本次发布零生产代码变更。纯测试 + 文档工作。

### Deferred (Phase 2 / v0.6.0 follow-up)

- `llm-agent/llm/stream.go::appendToolCallDelta` 按 `ID` 作为键；应当
  按 `Index` 作为键。该函数注释已指出「NOT the
  production accumulator」。它触及冻结的核心，需要一次 minor
  版本提升，超出了本次 YELLOW-lift PR 的范围。
- 来自 live 提供方的真实捕获 fixtures —— 当前的 `stream_tool_*`
  fixtures 是依据已校验的各提供方测试线缆形状手工制作的
  （有效性相同，交付更快）。若 `nightly-ollama-live.yml`
  观察到线缆形状漂移，则从真实捕获刷新。

## [v0.2.4] - 2026-05-23

Phase —— P1-23（3 个 PR 中的第 1+2+3）：抽取共享的 SDK 错误映射与
默认超时进 `internal/compat/`；迁移 `openai/`、`deepseek/`、
`anthropic/`、`minimax/`，最后是 `ollama/`（仅做默认超时调用
替换 —— Path A）。外加 P1-6 ollama 收尾（5/5）。

### Refactored (P1-23 PR 1 of 3)

- 将共享的 OpenAI-SDK 错误映射抽取进 `internal/compat/`：
  `WrapOpenAIError(provider, err)` 现在被 `openai/` 和
  `deepseek/`（两个搭载
  `github.com/openai/openai-go/v3` 的提供方）共同使用。这两个
  `errors.go` 文件除了 provider 名称字符串的 6 处出现之外，逐字符完全一致。
- 将 `if timeout == 0 { 60s }` 块抽取进
  `compat.DefaultTimeout(d)`。被 openai + deepseek 使用；
  anthropic/minimax/ollama 在后续 PR 中迁移。
- `internal/compat/` 受 Go 的 `internal/` 作用域限制 —— 下游消费方
  无法导入它。无公共 API 变更。各提供方测试数量
  不变；一致性套件（5/5 提供方）保持 GREEN。

### Refactored (P1-23 PR 2 of 3)

- 将共享的 Anthropic-SDK 错误映射抽取进 `internal/compat/`：
  `WrapAnthropicError(provider, err)` 现在被 `anthropic/` 和
  `minimax/`（两个搭载 `github.com/anthropics/anthropic-sdk-go` 的提供方）共同使用。
  保留了 529 Overloaded → RateLimitError 的特例。
- anthropic + minimax 的默认超时现在经由 `compat.DefaultTimeout`。
- 无公共 API 变更。各提供方测试数量不变。

### Refactored (P1-23 PR 3 of 3)

- Ollama 的默认超时现在流经 `compat.DefaultTimeout`，
  保留了 http-client 感知的守卫（当用户提供了带有自身
  Timeout 的预配置 http.Client 时跳过默认值的那个条件分支）。
  在 5/5 提供方都调用同一个默认超时辅助函数处收尾 P1-23 序列。
- Ollama 的 `errors.go` 保持按提供方独立（Path A）：其原子状态
  模式（statusCapturingTransport + atomic.Pointer[string]）是个
  异类，重构它会触及最近刚整修过的 P1-6
  derived-clients-share-transport 工作，仅为省下约 38 行代码（LoC）。推迟
  到未来当第 6 个 OpenAI-compat 提供方出现时的 P1-23b。

### Changed (behavior — defensive default, P1-6 follow-up)

- **ollama `New()`** 完成 5/5 超时收尾。同步调用
  （`Generate`、`Embed`）现在通过派生的 `*http.Client` 遵守 60s
  默认请求超时；`Stream` 使用一个 `Timeout=0` 的同级
  客户端，使长流连接仍仅由调用方 `ctx` 管理。两个客户端
  共享同一个 `*statusCapturingTransport` 实例，因此
  `lastStatus` / Retry-After 观测在各路径之间保持引用一致 —— 在
  同步请求上检测到的 429 仍对后续的错误包装可见，
  无论调用由哪条路径发起。
- 理由：ollama-go SDK v0.23.2 未暴露任何按请求的超时
  选项，且其 `Client.http` 字段是私有的，因此 4 提供方
  模式（`option.WithRequestTimeout`）不适用。派生
  客户端拆分是唯一干净的路径；带非零 `Timeout` 的单一共享
  客户端会在 60s 处切断流。
- `Ollama` 结构体新增了未导出的 `timeout`、`syncHTTPClient` 与
  `streamHTTPClient` 字段，以便内部测试（`*_internal_test.go`）
  能断言解析后的超时与传输层身份。无公共
  面变更。

### Notes

- 收尾了 v0.2.3 中的推迟项（"ollama `New()` not modified
  in this PR"）。全部 5/5 个 SDK 提供方现在都交付了防御式默认值。
- 解锁了假定没有任何 SDK 提供方的 `New()` 会静默挂起的
  `P1-23` compat 抽取（v1.4 窗口）。
- 参考：`docs/refactor-and-optimization-roadmap.zh-CN.md` §P1-6。

## [v0.2.3] - 2026-05-23

Phase —— P1-6：在 4 个 SDK 提供方上设置默认 60s HTTP 请求超时。

### Changed (behavior — defensive default)

- **openai/anthropic/deepseek/minimax `New()`** 现在在调用方未显式
  传入 `WithTimeout` 时设置默认 60s
  请求超时。可防止在上游停滞时空闲连接上的无限期挂起。
  实现方式：默认值被应用到
  `cfg.timeout`，并经由
  `option.WithRequestTimeout(cfg.timeout)` 转发给 SDK。**流式不会受
  客户端级 `Timeout` 限制** —— SDK 级选项按
  请求生效，而调用方 `ctx` 继续管理长时间运行的
  `Stream` 调用。
- 每个提供方结构体新增了未导出的 `timeout time.Duration`
  字段，使解析后的值可被内部测试观测
  （`*_internal_test.go`、`effectiveTimeoutForTest`）。无公共面
  变更。

### Deferred

- **ollama `New()` 在本次 PR 中未修改。** 审计发现
  `Stream` 与 `Generate` 共享同一个 `*api.Client`（因而也共享同一个
  `*http.Client`），而 `httpClient.Timeout` 会应用于
  包括流式响应体读取在内的整个请求生命周期。朴素的
  默认值会在 60s 处切断长流连接。后续会
  引入派生客户端（或等价物），使默认值仅
  影响同步调用。该事项单独跟踪；ollama 调用方在需要挂起守卫时应继续
  显式设置 `WithTimeout`。

### Notes

- 解锁了假定 4 个 SDK 提供方的 `New()` 调用不会静默挂起的
  P1-23 compat 抽取（v1.4 窗口）。
- 参考：`docs/refactor-and-optimization-roadmap.zh-CN.md` §P1-6。
