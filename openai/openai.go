package openai

import (
	"context"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/costa92/llm-agent-contract/llm"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

var (
	_ llm.ChatModel  = (*OpenAI)(nil)
	_ llm.ToolCaller = (*OpenAI)(nil)
	_ llm.Embedder   = (*OpenAI)(nil)
)

type OpenAI struct {
	client  *openai.Client
	info    llm.ProviderInfo
	tools   []llm.Tool
	timeout time.Duration
}

func (o *OpenAI) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := o.toSDKRequest(req)
	completion, err := o.client.Chat.Completions.New(ctx, sdkReq)
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	return o.fromSDKResponse(completion), nil
}

func (o *OpenAI) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	sdkReq := o.toSDKStreamRequest(req)
	return &openaiStreamReader{
		open: func() *ssestream.Stream[openai.ChatCompletionChunk] {
			return o.client.Chat.Completions.NewStreaming(ctx, sdkReq)
		},
	}, nil
}

func (o *OpenAI) Info() llm.ProviderInfo { return o.info }

func (o *OpenAI) WithTools(tools []llm.Tool) (llm.ToolCaller, error) {
	cp := *o
	cp.tools = append([]llm.Tool(nil), tools...)
	return &cp, nil
}

func (o *OpenAI) Embed(ctx context.Context, texts []string) ([]llm.Vector, llm.Usage, error) {
	if len(texts) == 0 {
		return []llm.Vector{}, llm.Usage{Source: llm.UsageReported}, nil
	}
	resp, err := o.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(o.info.Model),
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: append([]string(nil), texts...),
		},
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	})
	if err != nil {
		return nil, llm.Usage{}, wrapErr(err)
	}
	vectors := make([]llm.Vector, 0, len(resp.Data))
	for _, item := range resp.Data {
		vec := make(llm.Vector, 0, len(item.Embedding))
		for _, f := range item.Embedding {
			vec = append(vec, float32(f))
		}
		vectors = append(vectors, vec)
	}
	usage := llm.Usage{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: 0,
		TotalTokens:  int(resp.Usage.TotalTokens),
		Source:       llm.UsageReported,
	}
	return vectors, usage, nil
}

func (o *OpenAI) EmbedDimensions() int {
	switch o.info.Model {
	case "text-embedding-3-small":
		return 1536
	case "text-embedding-3-large":
		return 3072
	case "text-embedding-ada-002":
		return 1536
	default:
		return 0
	}
}

type openaiStreamReader struct {
	mu            sync.Mutex
	open          func() *ssestream.Stream[openai.ChatCompletionChunk]
	stream        *ssestream.Stream[openai.ChatCompletionChunk]
	queue         []llm.StreamEvent
	retried       bool
	deliveredByte bool
	lastFinish    llm.FinishReason
	closed        bool
	toolIndexes   map[int]struct{}
}

func (r *openaiStreamReader) Next() (llm.StreamEvent, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for {
		if r.closed {
			return llm.StreamEvent{}, io.EOF
		}
		if len(r.queue) > 0 {
			ev := r.queue[0]
			r.queue = r.queue[1:]
			if ev.Kind != llm.EventDone {
				r.deliveredByte = true
			}
			return ev, nil
		}
		if r.stream == nil {
			r.stream = r.open()
		}
		if !r.stream.Next() {
			err := r.stream.Err()
			_ = r.stream.Close()
			r.stream = nil
			if err != nil {
				if !r.deliveredByte && !r.retried {
					r.retried = true
					continue
				}
				return llm.StreamEvent{}, wrapErr(err)
			}
			return llm.StreamEvent{}, io.EOF
		}

		chunk := r.stream.Current()
		r.queue = append(r.queue, r.chunkEvents(chunk)...)
	}
}

func (r *openaiStreamReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	if r.stream == nil {
		return nil
	}
	err := r.stream.Close()
	r.stream = nil
	return err
}

func (r *openaiStreamReader) chunkEvents(chunk openai.ChatCompletionChunk) []llm.StreamEvent {
	var events []llm.StreamEvent

	if chunk.Usage.JSON.TotalTokens.Valid() || chunk.Usage.TotalTokens != 0 {
		usage := llm.Usage{
			InputTokens:  int(chunk.Usage.PromptTokens),
			OutputTokens: int(chunk.Usage.CompletionTokens),
			TotalTokens:  int(chunk.Usage.TotalTokens),
			Source:       llm.UsageReported,
		}
		events = append(events, llm.StreamEvent{
			Kind:         llm.EventDone,
			Usage:        &usage,
			FinishReason: r.lastFinish,
		})
		return events
	}

	for _, choice := range chunk.Choices {
		if choice.Delta.Content != "" {
			events = append(events, llm.StreamEvent{
				Kind: llm.EventTextDelta,
				Text: choice.Delta.Content,
			})
		}
		for _, tool := range choice.Delta.ToolCalls {
			if r.toolIndexes == nil {
				r.toolIndexes = make(map[int]struct{})
			}
			r.toolIndexes[int(tool.Index)] = struct{}{}
			if tool.ID != "" || tool.Function.Name != "" {
				events = append(events, llm.StreamEvent{
					Kind: llm.EventToolCallStart,
					ToolCall: &llm.ToolCallDelta{
						Index: int(tool.Index),
						ID:    tool.ID,
						Name:  tool.Function.Name,
					},
				})
			}
			if tool.Function.Arguments != "" {
				events = append(events, llm.StreamEvent{
					Kind: llm.EventToolCallArgsDelta,
					ToolCall: &llm.ToolCallDelta{
						Index:     int(tool.Index),
						ID:        tool.ID,
						ArgsDelta: tool.Function.Arguments,
					},
				})
			}
		}
		if choice.FinishReason != "" {
			r.lastFinish = mapFinishReason(choice.FinishReason)
			if r.lastFinish == llm.FinishReasonToolCalls && len(r.toolIndexes) > 0 {
				indexes := make([]int, 0, len(r.toolIndexes))
				for idx := range r.toolIndexes {
					indexes = append(indexes, idx)
				}
				sort.Ints(indexes)
				for _, idx := range indexes {
					events = append(events, llm.StreamEvent{
						Kind: llm.EventToolCallEnd,
						ToolCall: &llm.ToolCallDelta{
							Index: idx,
						},
					})
				}
				r.toolIndexes = nil
			}
		}
	}

	return events
}
