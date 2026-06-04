# minimax

[English](./README.md) | [简体中文](./README.zh-CN.md)

面向 `github.com/costa92/llm-agent/llm` 的 MiniMax 适配器。

当前范围：

- `llm.ChatModel.Generate`
- `llm.ChatModel.Stream`
- `llm.ToolCaller.WithTools`
- 带模型绑定提供方元数据的 `Info()`

有意保留的能力缺口：

- 未实现嵌入（embeddings）
- `Info().Capabilities.Embeddings` 始终为 `false`

区域路由：

- `WithRegion(RegionCN|RegionGlobal)` 选择预设端点
- `WithBaseURL(...)` 覆盖区域预设

最小示例：

```go
model, err := minimax.New(
	minimax.WithModel("MiniMax-M1"),
	minimax.WithAPIKey(os.Getenv("MINIMAX_API_KEY")),
	minimax.WithRegion(minimax.RegionGlobal),
)
if err != nil {
	log.Fatal(err)
}

resp, err := model.Generate(ctx, llm.Request{
	Messages: []llm.Message{{Role: "user", Content: "Say hello"}},
})
```
