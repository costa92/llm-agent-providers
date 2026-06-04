# 图像生成 + Google/Volcengine 提供方设计

Date: 2026-06-04
Repos: `llm-agent-contract`（新能力）、`llm-agent-providers`（实现）
Status: proposed

## 目标

一个里程碑中两项相互耦合的交付物：

1. 一项一等公民级的 **文本到图像生成（text-to-image generation）** 能力（`ImageGenerator`），
   覆盖四个提供方：`openai`、`minimax`、`volcengine`（新增）、
   `google`（新增）。
2. 两个 **全新的完整提供方** —— `volcengine`（火山方舟 Ark / 豆包）与
   `google`（Gemini）—— 它们是 `openai`/`anthropic` 的一等公民级兄弟：
   它们实现 **ChatModel + ToolCaller**（Generate / Stream / WithTools /
   Info）、**`ImageGenerator`** 与 **`Embedder`**，能力按所绑定的
   模型进行门控（K2）。
3. 在三个拥有嵌入产品的提供方上提供 **嵌入（embeddings）**（`Embedder`，已在契约中
   —— 无契约改动）：`volcengine`、`google`、
   `minimax`。

图像能力遵循契约现有的 **正交能力（orthogonal capability）**
约定（`Embedder`、`ToolCaller`、`StructuredOutputs`）：一个新的
`ImageGenerator` 接口位于 `llm-agent-contract`，检测通过类型
断言 **加上** 一个 `Capabilities.ImageGeneration` 标志，且它 **不** 内嵌
`ChatModel`。

### K2 模型门控（适用于全部四个）

提供方实例在构造时绑定一个模型。Go struct 实现
其方法所覆盖的每一个接口（ChatModel、ToolCaller、ImageGenerator），但
`Capabilities` 反映的是 **所绑定的（provider × model）元组**：

- `google.New(WithModel("gemini-2.5-flash"))` → chat 可用；`GenerateImage`
  返回 `llm.ErrCapabilityNotSupported`。
- `google.New(WithModel("gemini-2.5-flash-image"))` 或 `"imagen-4.0-generate-001"`
  → `GenerateImage` 可用；`Generate`（chat）返回 not-supported 错误。
- `volcengine` 同样的拆分：`doubao-1-5-pro-32k-*`（chat）与
  `doubao-seedream-4-5-*`（image）。

这正是现有的模式（openai 实现 ChatModel + Embedder，
按模型门控）。

## 范围

### 纳入范围

- `llm-agent-contract` → **v0.3.0**：`ImageGenerator` 接口、
  `ImageRequest` / `ImageResponse` / `GeneratedImage` 类型、
  `Capabilities.ImageGeneration` 字段。
- `openai`：在 `*OpenAI` 上的 `GenerateImage`，按模型门控。
- `minimax`：通过一条新的 raw-HTTP 路径实现 `GenerateImage`。
- `volcengine`（新包）：完整提供方 —— Generate / Stream / WithTools /
  Info + GenerateImage + Embed。使用官方 `arkruntime` SDK。
- `google`（新包）：完整提供方 —— Generate / Stream / WithTools / Info +
  GenerateImage（Nano Banana 经 GenerateContent + Imagen 经 GenerateImages）+
  Embed。使用官方 `google.golang.org/genai` SDK。
- `minimax`：通过一条 raw-HTTP 路径新增 `Embed`（与图像 raw-HTTP 路径并列）。
- **自定义请求头**：每个提供方都有一个 `WithExtraHeaders(map[string]string)` 选项
  （go-openai 的缺口）—— 供需要自定义鉴权/路由头的
  兼容端点/网关使用。应用于 chat/stream/image/embed。
- 每个提供方的 `httptest` 风格 fixture 覆盖（构造函数/配置、generate、
  stream、工具调用、图像 generate、能力标志）。
- README 更新（仓库级 + 包级）、双语 `.zh-CN.md` 配对。
- 跨仓库锁步推出（契约先打 tag，然后 providers 重新 pin）。

### 不纳入范围

- 图像 **编辑（editing）** / 变体（variations）/ 局部重绘（inpainting）（仅文本到图像的 `Generate`）。
- **流式图像（streaming image）** 生成。
- 新包的 **结构化输出**（`WithSchema`）。
- 没有嵌入产品的提供方（deepseek、anthropic）的嵌入。
  Ollama/OpenAI 已实现 `Embedder`；此处不变。
- 多模态 / 视觉嵌入（Volcengine `doubao-embedding-vision` 经
  `CreateMultiModalEmbeddings`）—— 仅文本嵌入。
- 不采纳 go-openai 暴露的能力（YAGNI）：图像 edit/variation、
  音频（transcription/TTS）、moderation、batch、fine-tuning、files、assistants、
  legacy completions。
- Google **Vertex AI** 后端（仅 API-key Gemini Developer API；
  `BackendVertexAI` 是后续的配置切换）。
- Volcengine **AK/SK 签名** 的视觉 API（`visual.volcengineapi.com`）以及 Ark
  **endpoint-id（ep-xxxx）** 配置 —— 仅按模型名直接调用。
- Volcengine Ark 私有扩展（Thinking / reasoning_content / 加密
  内容）。仅映射标准 chat；预留钩子但不暴露。
- Gemini 选择性开启的 `streamFunctionCallArguments`（Gemini 3 Pro+）—— 将每个
  流式 functionCall chunk 视为完整。
- live 集成测试（仅 mock，与现有适配器一致）。
- 提供方注册表/工厂（仓库刻意不设）。

## 契约新增（`llm-agent-contract`，v0.3.0）

新文件 `llm/image.go`：

```go
package llm

import "context"

// ImageGenerator is the capability for text-to-image generation. Like
// Embedder, it deliberately does NOT embed ChatModel. Providers without an
// image endpoint (anthropic, deepseek, ollama) do NOT implement this
// interface; callers detect via type assertion AND Capabilities.ImageGeneration.
type ImageGenerator interface {
	GenerateImage(ctx context.Context, req ImageRequest) (ImageResponse, error)
}

// ImageRequest describes a text-to-image call. Providers ignore knobs they do
// not support (documented per provider).
type ImageRequest struct {
	Prompt  string         // required
	N       int            // number of images; 0 => provider default (1)
	Size    string         // e.g. "1024x1024"; "" => provider default
	Quality string         // e.g. "standard"/"hd"/"high"; "" => provider default
	Format  string         // output encoding "png"/"jpeg"/"webp"; "" => provider default
	Extra   map[string]any // provider-specific knobs, forwarded verbatim
}

// ImageResponse is the result, in request order.
type ImageResponse struct {
	Images   []GeneratedImage
	Provider string
	Model    string
	Usage    Usage // best-effort; zero when the provider does not report it
}

// GeneratedImage is one produced image. Exactly one of Bytes or URL is
// populated, chosen by the provider's most direct delivery path.
type GeneratedImage struct {
	Bytes         []byte // inline bytes (base64-decoded) when returned inline
	URL           string // hosted link when the provider returns a URL
	MimeType      string // e.g. "image/png"; "" if unknown
	RevisedPrompt string // provider's rewritten prompt, if any (dall-e-3)
}
```

`Capabilities` 新增：`ImageGeneration bool \`json:"image_generation"\``。

**交付规则：** 调用方 **不** 请求 URL-vs-bytes。OpenAI b64 → `Bytes`；
Volcengine/Minimax url → `URL`；Google 总是 → `Bytes`。

由于 `volcengine` 与 `google` 现在是完整的 ChatModel，`Info()` 来自
`ChatModel` 接口（不再是「便利」方法）。

## 提供方 × API × SDK 矩阵

| 提供方     | Chat API                         | Image API                          | 鉴权               | SDK                                  | 新依赖 |
|------------|----------------------------------|------------------------------------|--------------------|--------------------------------------|---------|
| openai     | 现有                             | 原生 `Images.Generate`             | `OPENAI_API_KEY`   | `openai-go/v3`（现有）               | 否      |
| minimax    | 现有                             | 专有 `/v1/image_generation`        | `MINIMAX_API_KEY`  | anthropic SDK（chat）+ raw HTTP（img）| 否      |
| volcengine | Ark `/chat/completions`（OpenAI 形状） | Ark `/images/generations`   | `Bearer ARK_API_KEY` | 官方 `arkruntime`                  | **是**  |
| google     | Gemini `:generateContent`        | Gemini inline + Imagen `:predict`  | `GEMINI_API_KEY`   | 官方 `google.golang.org/genai`       | **是**  |

新模块依赖（在实现时核对确切版本）：`google.golang.org/genai`
（参考 v1.59.0）、`github.com/volcengine/volcengine-go-sdk`（参考 v1.2.33，
`service/arkruntime` + `.../model` + `.../utils`）。

## 各提供方设计

每个新包遵循仓库的文件骨架：`doc.go`、`<provider>.go`、
`options.go`、`map.go`、`errors.go`（在图像路径较大时另加 `image.go`）。

### openai — `openai/image.go`

- 在 `*OpenAI` 上的 `GenerateImage`；`var _ llm.ImageGenerator = (*OpenAI)(nil)`。
- 映射 `req → openai.ImageGenerateParams{Prompt, Model: o.info.Model, N, Size,
  Quality, OutputFormat}`；存在时取 `Extra` 键 `style`/`background`/`moderation`。
  调用 `client.Images.Generate`；`Data[]` → `[]GeneratedImage`
  （`B64JSON` 解码 → `Bytes`，否则 `URL`、`RevisedPrompt`）。错误经 `wrapErr`。
- **K2 门控：** `isImageModel(cfg.model)`（`gpt-image-1/2`、`dall-e-2/3`）设置
  `Capabilities.ImageGeneration`；非图像模型 →
  `llm.ErrCapabilityNotSupported`（与 `Embed` 门控一致）。

### minimax — `minimax/image.go`（仓库中第一条 raw-HTTP 路径）

- `MiniMax` struct 额外保留 `baseURL`/`apiKey`/`httpClient`。
- `POST {baseURL}/v1/image_generation`，携带 `{model, prompt, n,
  response_format:"url", aspect_ratio | width+height}`。`Size` "WxH" →
  `width`/`height`；否则取 `Extra["aspect_ratio"]`。`data.image_urls[]` → `URL`。
- **坑：** Minimax 在逻辑失败时仍返回 HTTP 200；检查
  `base_resp.status_code != 0` → 类型化 `llm.*` 错误（状态映射复用
  `internal/compat`）。
- `capabilitiesForModel("image-01")` 设置 `ImageGeneration: true`。

### volcengine — 新的完整提供方（arkruntime SDK）

**构造**（`arkruntime` v1.2.33）：
`arkruntime.NewClientWithApiKey(apiKey, arkruntime.WithRegion("cn-beijing"),
arkruntime.WithBaseUrl(...), arkruntime.WithTimeout(...))`。
选项：`WithModel`（必需，按模型名直接调用，例如 `doubao-1-5-pro-32k-250115`
或 `doubao-seedream-4-5-251128`）、`WithAPIKey`（env `ARK_API_KEY`）、`WithBaseURL`、
`WithRegion`、`WithHTTPClient`、`WithTimeout`。模型 id 是配置值 ——
绝不硬编码（取决于账号）。

**Chat（Generate）：** `client.CreateChatCompletion(ctx, req)`，使用
`model.CreateChatCompletionRequest`（**指针字段** 变体，使
`temperature=0` 可被发送；值字段版 `ChatCompletionRequest` 已废弃）。
Messages → `[]*model.ChatCompletionMessage`，内容联合体为
`&model.ChatCompletionMessageContent{StringValue: ...}`；角色为
`system/user/assistant/tool`。响应文本取自
`resp.Choices[0].Message.Content.StringValue`；`Usage{PromptTokens,
CompletionTokens, TotalTokens}`；finish reason 取自 `Choices[0].FinishReason`。

**Stream：** `client.CreateChatCompletionStream(ctx, req)` →
`*utils.ChatCompletionStreamReader`；循环 `stream.Recv()` 直到 `io.EOF`
（`stream.Close()` 释放 body）。从 `Choices[0].Delta` 发出仓库的类型化
`llm.StreamEvent`（`Content` → `EventTextDelta`）。

**Tools（WithTools）：** 请求 `model.Tool{Type: ToolTypeFunction, Function:
*model.FunctionDefinition{Name, Description, Parameters: json.RawMessage}}`、
`ToolChoice`。非流式工具调用在 `Choices[0].Message.ToolCalls[]`
（`ID`、`Function.Name`、`Function.Arguments` JSON 字符串）。**流式工具
调用** 分片到达（OpenAI 形状）：`Delta.ToolCalls[].Index *int` 是
合并键 —— 按 `Index` 累积 `Function.Arguments` 子串，发出
`EventToolCallStart`（带 Name/ID 的第一个分片）→ `EventToolCallArgsDelta` →
`EventToolCallEnd`，保持每次调用稳定的 Index（K1）。这几乎逐字地照搬
现有的 `openai` 流读取器。

**Image（GenerateImage）：** `client.GenerateImages(ctx,
model.GenerateImagesRequest{Model, Prompt, Size, ResponseFormat, Seed,
GuidanceScale, Watermark, N})`；`Extra` 携带 `seed`/`guidance_scale`/
`watermark`。`model.ImagesResponse.Data[]` → `URL`（默认）或 `Bytes`
（b64_json）。
- **坑：** `response_format=url` 链接约 24h 过期（文档注：用 b64_json
  持久化）；`size` 约束因模型而异（3.0 t2i 512–2048px；4.x/5.0 最高
  4096 + `1K`/`2K`/`4K` 档位）。

**Errors：** `*model.APIError{HTTPStatusCode, Code, RequestId}` 与
`*model.RequestError{HTTPStatusCode}` —— 两者都携带 HTTP 状态；映射为
`llm.AuthError`/`RateLimitError`/`TransientError`/`InvalidRequestError`
（`errors.As`）。SDK 内置重试；设置 `WithRetryTimes(0)` 以保持我们与
其他适配器一致的单次尝试策略。

**Capabilities：** `capabilitiesForModel(model)` —— chat 模型 `Tools: true`，
`doubao-seedream*` 设 `ImageGeneration: true`。

### google — 新的完整提供方（genai SDK），Nano Banana + Imagen

**构造**（`genai` v1.59.0）：`genai.NewClient(ctx, &genai.ClientConfig{
APIKey, Backend: genai.BackendGeminiAPI})`。选项：`WithModel`（必需）、
`WithAPIKey`（env `GEMINI_API_KEY`，回退 `GOOGLE_API_KEY`）、`WithHTTPClient`
（→ `ClientConfig.HTTPClient`）、`WithBaseURL`（→ `ClientConfig.HTTPOptions.BaseURL`，
用于 httptest）、`WithTimeout`。不要设置 Project/Location/Credentials
（与 APIKey 互斥）。

**Chat（Generate）：** `client.Models.GenerateContent(ctx, model, contents,
*genai.GenerateContentConfig)`。角色仅 `user`/`model`；**system prompt →
`GenerateContentConfig.SystemInstruction *Content`**（无 system 角色）。
`Temperature`/`TopP` 是 `*float32`（用 `genai.Ptr`），`MaxOutputTokens` 是
普通 `int32`。文本经 `resp.Text()` 读取；finish reason
`Candidates[0].FinishReason`（例如 `STOP`、`MAX_TOKENS`）；usage
`UsageMetadata{PromptTokenCount, CandidatesTokenCount, TotalTokenCount}`
（nil 保护）。映射 `STOP`→stop、`MAX_TOKENS`→length 等。

**Stream：** `client.Models.GenerateContentStream(...)` 返回一个 Go-1.23 的
`iter.Seq2[*GenerateContentResponse, error]`（range-over-func）。用 **`iter.Pull2`**
桥接到仓库的拉取式 `StreamReader`（go.mod 为 go 1.26 —— OK）：
流读取器的惰性打开返回 `next()`；每个 chunk 的部分文本是
`Candidates[0].Content.Parts[].Text` → `EventTextDelta`。当
`iter.Pull2` 的 `next` 报告 done 时循环结束。

**Tools（WithTools）：** 请求 `genai.Tool{FunctionDeclarations:
[]*genai.FunctionDeclaration{{Name, Description, ParametersJsonSchema: <raw>}}}`
—— 使用 **`ParametersJsonSchema any`**（将仓库的 JSON-schema 字节解码为
`map[string]any` 并赋值；无需类型化的 `genai.Schema` 转换）。非流式
工具调用经 `resp.FunctionCalls()` → `{Name, Args map[string]any, ID}`
（将 `Args` 重新 marshal 为 JSON 字符串供仓库的 `ToolCall.Arguments`）。
**流式工具调用在一个 chunk 中完整到达**（不分片）→ 每次调用一起发出
`EventToolCallStart` + `EventToolCallArgsDelta`（完整 args）+ `EventToolCallEnd`；
按 `FunctionCall.ID`/part 索引关联并行调用。无需跨 chunk 的参数
累积（比 openai/volcengine 更简单）。

**Image（GenerateImage）：** 按所绑定的模型 id 路由：
- `strings.HasPrefix(model, "imagen")` → `client.Models.GenerateImages(ctx,
  model, prompt, &genai.GenerateImagesConfig{NumberOfImages: N, AspectRatio})`；
  `resp.GeneratedImages[].Image.ImageBytes` → `Bytes`。
- 否则（Gemini 原生，例如 `gemini-2.5-flash-image`）→
  `GenerateContent(..., &GenerateContentConfig{ResponseModalities:
  []string{"TEXT","IMAGE"}})`；`Candidates[0].Content.Parts[].InlineData{Data,
  MIMEType}` → `Bytes`。
- **坑：** 输出 **总是** base64 inline（无 URL）→ 总是 `Bytes`。
  `ResponseModalities` 必须包含 `TEXT`（Gemini 2.5 Flash Image 拒绝纯图像
  请求）；丢弃 text parts。Gemini 原生没有干净的 `N`（≈1）；`Size`
  松散地映射到 `aspectRatio`/`imageSize`。每张图都带一个不可移除的
  SynthID 水印。

**Errors：** `genai.APIError{Code int, Status, Message}` **按值** 返回 ——
`var e genai.APIError; errors.As(err, &e)`；switch `e.Code`（401/403/429/5xx）。
对 `Candidates` 做 nil 保护（prompt 可能被拦截 → `PromptFeedback.BlockReason`）。

**Capabilities：** `capabilitiesForModel(model)` —— `gemini-*` chat 模型
`Tools: true`；`gemini-*-image` 与 `imagen-*` 设 `ImageGeneration: true`。

**默认模型：** chat `gemini-2.5-flash`（稳定；避开已下线的
`gemini-2.0-flash`）；image `gemini-2.5-flash-image` / `imagen-4.0-generate-001`。
所有模型 id 保持可配置。

### 嵌入（minimax / volcengine / google 上的 `Embedder`）

`Embedder` 已存在于契约中 —— **无契约改动**。该
接口是固定的：`Embed(ctx, texts []string) ([]Vector, Usage, error)` +
`EmbedDimensions() int`。`Vector` 是 `[]float32`。它不携带每次调用的
query/document 或维度旋钮，因此这些是 **构造期的提供方
选项**，与其他每项能力一样按模型门控（`Capabilities.Embeddings`）。

**minimax** —— raw HTTP（SDK 无嵌入）：
- `POST {baseURL}/v1/embeddings`，`Authorization: Bearer`，**`GroupId` 作为查询
  参数** → 新选项 `WithGroupID`（env `MINIMAX_GROUP_ID`；embed 必需）。
- Body `{model, texts, type}`；响应中有 **顶层** `vectors [][]float32`
  与 `total_tokens`（**不** 嵌套在 `data`/`usage` 下）；检查 `base_resp`。
- `type` 为 `"db"`（document，默认）/ `"query"` → 选项 `WithEmbeddingType`。
- 模型 `embo-01`，固定 **1536** 维 → `EmbedDimensions()` 返回 1536。

**volcengine** —— `arkruntime` `CreateEmbeddings(ctx,
model.EmbeddingRequestStrings{Input: texts, Model, Dimensions})`；响应
`Data[].Embedding []float32`（按索引顺序）、`Usage{PromptTokens, TotalTokens}`。
模型 `doubao-embedding-text-240715`（2560，可降至 512/1024/2048）、
`doubao-embedding-large-text-240915`（4096）。`WithDimensions(int)` →
`EmbedDimensions()`；默认值按模型而定。

**google** —— `client.Models.EmbedContent(ctx, model, contents,
*genai.EmbedContentConfig)`。将 `texts → []*genai.Content` 组装（每个文本一个 Content；
`genai.Text` 每次产生一个）。响应 `Embeddings[].Values []float32`
按顺序。**Gemini Developer API 不返回 token usage** → `Usage` 为零
（Source `UsageUnknown`）。选项：`WithTaskType`（例如 `RETRIEVAL_DOCUMENT`，
默认为空 = 模型默认）、`WithDimensions` → `EmbedContentConfig.
OutputDimensionality *int32`。模型 `gemini-embedding-001`（3072，可 MRL 截断
至 1536/768）、`text-embedding-004`（768）。

**能力门控：** 每个提供方 `Capabilities.Embeddings = isEmbedModel(model)`；
非嵌入模型 → `llm.ErrCapabilityNotSupported`，`EmbedDimensions()`
返回 0（与现有的 openai/ollama 模式一致）。

### 自定义请求头（`WithExtraHeaders`）

每个提供方都有一个构造期的 `WithExtraHeaders(map[string]string)` 选项，
将 header 注入所有出站请求（chat/stream/image/embed）。
经各 SDK 的 header 机制应用 —— openai-go `option.WithHeaderAdd`、
anthropic SDK `option.WithHeader`、arkruntime 的 request/config 选项、genai
`ClientConfig.HTTPOptions.Headers`；raw-HTTP 路径（minimax image/embed）直接
在 `*http.Request` 上设置。**实现时核实：** `arkruntime` 与 `genai` 的
确切 header 注入钩子。保留 header（`Authorization`、`Content-Type`）
不可覆盖 —— 额外 header 是叠加式的。

## 错误处理（所有提供方）

每个提供方的 `wrapErr` → 类型化 `llm.*` 错误（`AuthError`、带 `Retry-After`
的 `RateLimitError`、`TransientError`、`InvalidRequestError`），与
现有适配器一致。`context.Canceled` 原样返回；`DeadlineExceeded` →
`TransientError`。新的映射工作：minimax `base_resp.status_code`；arkruntime
`APIError`/`RequestError`；genai `APIError`（按值）。

## 测试

- **contract：** 接口编译断言 + 针对新类型与 `image_generation` 能力标志的
  JSON（反）序列化测试。
- **openai / minimax：** 用 `httptest.NewServer` mock 图像（OpenAI b64、
  Minimax `image_urls`）；断言请求映射、解码、能力标志。
- **volcengine：** 经 `WithBaseUrl` 将 `arkruntime` 指向 httptest。覆盖
  generate、stream（含按 `Index` 的分片工具调用合并）、工具请求
  映射、以及图像（`data[].url` / `b64_json`）。stream 用 mock SSE。
- **google：** 经 `HTTPOptions.BaseURL` 将 `genai` 指向 httptest。覆盖
  generate、stream（`iter.Pull2` 桥接）、工具调用（一个 chunk 完整到达）、
  以及两条图像分支（`generateContent` inlineData + Imagen
  `bytesBase64Encoded`）。
- **embeddings：** 每个提供方 mock embed 响应（minimax 顶层
  `vectors`；volcengine `Data[].Embedding`；google `Embeddings[].Values`），
  断言向量顺序/长度、`EmbedDimensions()`、以及 `Embeddings`
  能力标志。minimax embed 还断言 `GroupId` 查询参数与
  `type` 字段。
- Capability-not-supported 路径返回 `llm.ErrCapabilityNotSupported`。
- `go.uber.org/goleak` 用于 stream-reader 的 goroutine/生命周期（与现有一致）。

## 推出（跨仓库锁步，一个里程碑）

1. **contract 仓库：** 新增 `llm/image.go` + `Capabilities.ImageGeneration` +
   测试 + 文档 → PR → merge → 打 tag **v0.3.0** → push。
2. **providers 仓库：** 用本地 `replace` 指向 contract 进行开发；实现
   `openai` + `minimax` 图像方法 + `minimax` embed；构建完整的
   `volcengine` + `google` 包（chat + tools + stream + image + embed）；
   在各提供方加入 `WithExtraHeaders`；加入两个新模块依赖。当
   通过时：移除 `replace`，pin contract `v0.3.0`，`GOWORK=off go mod tidy`，
   更新 README（+ `.zh-CN.md`），PR → merge → 打 tag。

replace-guard 预提交钩子会在提交时自动剥离本地 `replace`，因此
contract 必须先打 tag v0.3.0，providers PR 才能 pin 一个真实版本。

## 待核实项（实现时解决，不阻塞设计）

1. Pin 确切的 SDK 版本；针对所 pin 的版本重新确认 `arkruntime` 的
   符号/字段名（`CreateChatCompletionRequest`、`GenerateImagesRequest`、`ToolCall.Index`）
   与 `genai` 的字段名（`GeneratedImages` vs `Images`、`ParametersJsonSchema`、
   `iter.Seq2` 签名）。
2. 在真实的 Ark 工具调用流上确认流式工具调用的参数分片
   顺序（SDK 不合并；提供方必须合并）。
3. 在所 pin 的版本上核实 Gemini 后端的 `genai` 流式 functionCall 是单
   chunk（研究中标注了一个跳过它的较旧测试）。
4. 确认 `openai-go` 的 `ImageGenerateParams` 为所 pin 的 `gpt-image` 模型
   暴露了 `Quality`/`OutputFormat` 枚举。
5. 在所 pin 的版本上确认 embed 符号（`arkruntime` `CreateEmbeddings` /
   `EmbeddingRequestStrings`；`genai` `EmbedContent` / `EmbedContentResponse.
   Embeddings[].Values`）；针对当前 Minimax 文档确认 Minimax 嵌入端点、
   `GroupId` 查询参数要求、以及 `type`（db/query）字段
   （研究依据了 Spring AI 的实现）。
6. 为 `WithExtraHeaders` 确认各 SDK 的自定义 header 钩子（`arkruntime`、`genai`）。
