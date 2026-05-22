package deepseek

import (
	"context"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/costa92/llm-agent/llm"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

var (
	_ llm.ChatModel  = (*DeepSeek)(nil)
	_ llm.ToolCaller = (*DeepSeek)(nil)
)

type DeepSeek struct {
	client  *openai.Client
	info    llm.ProviderInfo
	tools   []llm.Tool
	timeout time.Duration
}

func (d *DeepSeek) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := d.toSDKRequest(req)
	completion, err := d.client.Chat.Completions.New(ctx, sdkReq)
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	return d.fromSDKResponse(completion), nil
}

func (d *DeepSeek) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	sdkReq := d.toSDKStreamRequest(req)
	return &deepseekStreamReader{
		open: func() *ssestream.Stream[openai.ChatCompletionChunk] {
			return d.client.Chat.Completions.NewStreaming(ctx, sdkReq)
		},
	}, nil
}

func (d *DeepSeek) Info() llm.ProviderInfo { return d.info }

func (d *DeepSeek) WithTools(tools []llm.Tool) (llm.ToolCaller, error) {
	cp := *d
	cp.tools = append([]llm.Tool(nil), tools...)
	return &cp, nil
}

type deepseekStreamReader struct {
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

func (r *deepseekStreamReader) Next() (llm.StreamEvent, error) {
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

func (r *deepseekStreamReader) Close() error {
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

func (r *deepseekStreamReader) chunkEvents(chunk openai.ChatCompletionChunk) []llm.StreamEvent {
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
