package anthropic

import (
	"context"
	"io"
	"sync"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/costa92/llm-agent/llm"
)

var (
	_ llm.ChatModel  = (*Anthropic)(nil)
	_ llm.ToolCaller = (*Anthropic)(nil)
)

type Anthropic struct {
	client *sdk.Client
	info   llm.ProviderInfo
	tools  []llm.Tool
}

func (a *Anthropic) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := a.toSDKRequest(req)
	msg, err := a.client.Messages.New(ctx, sdkReq)
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	return a.fromSDKResponse(msg), nil
}

func (a *Anthropic) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	sdkReq := a.toSDKRequest(req)
	return &anthropicStreamReader{
		open: func() *ssestream.Stream[sdk.MessageStreamEventUnion] {
			return a.client.Messages.NewStreaming(ctx, sdkReq)
		},
		blockKinds: make(map[int]string),
		blockMeta:  make(map[int]toolBlockMeta),
	}, nil
}

func (a *Anthropic) Info() llm.ProviderInfo { return a.info }

func (a *Anthropic) WithTools(tools []llm.Tool) (llm.ToolCaller, error) {
	cp := *a
	cp.tools = append([]llm.Tool(nil), tools...)
	return &cp, nil
}

type toolBlockMeta struct {
	id   string
	name string
}

type anthropicStreamReader struct {
	mu            sync.Mutex
	open          func() *ssestream.Stream[sdk.MessageStreamEventUnion]
	stream        *ssestream.Stream[sdk.MessageStreamEventUnion]
	queue         []llm.StreamEvent
	retried       bool
	deliveredByte bool
	closed        bool

	blockKinds map[int]string
	blockMeta  map[int]toolBlockMeta
	usage      llm.Usage
	finish     llm.FinishReason
}

func (r *anthropicStreamReader) Next() (llm.StreamEvent, error) {
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

		ev := r.stream.Current()
		r.queue = append(r.queue, r.eventToStreamEvents(ev)...)
	}
}

func (r *anthropicStreamReader) Close() error {
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

func (r *anthropicStreamReader) eventToStreamEvents(ev sdk.MessageStreamEventUnion) []llm.StreamEvent {
	switch v := ev.AsAny().(type) {
	case sdk.MessageStartEvent:
		r.usage = llm.Usage{
			InputTokens:  int(v.Message.Usage.InputTokens),
			OutputTokens: int(v.Message.Usage.OutputTokens),
			TotalTokens:  int(v.Message.Usage.InputTokens + v.Message.Usage.OutputTokens),
			Source:       llm.UsageReported,
		}
	case sdk.MessageDeltaEvent:
		r.finish = mapAnthropicStopReason(string(v.Delta.StopReason))
		r.usage.InputTokens = int(v.Usage.InputTokens)
		r.usage.OutputTokens = int(v.Usage.OutputTokens)
		r.usage.TotalTokens = r.usage.InputTokens + r.usage.OutputTokens
		r.usage.Source = llm.UsageReported
	case sdk.ContentBlockStartEvent:
		idx := int(v.Index)
		switch block := v.ContentBlock.AsAny().(type) {
		case sdk.TextBlock:
			r.blockKinds[idx] = "text"
		case sdk.ToolUseBlock:
			r.blockKinds[idx] = "tool_use"
			r.blockMeta[idx] = toolBlockMeta{id: block.ID, name: block.Name}
			return []llm.StreamEvent{{
				Kind: llm.EventToolCallStart,
				ToolCall: &llm.ToolCallDelta{
					Index: idx,
					ID:    block.ID,
					Name:  block.Name,
				},
			}}
		case sdk.ThinkingBlock:
			r.blockKinds[idx] = "thinking"
		}
	case sdk.ContentBlockDeltaEvent:
		idx := int(v.Index)
		switch delta := v.Delta.AsAny().(type) {
		case sdk.TextDelta:
			return []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: delta.Text}}
		case sdk.InputJSONDelta:
			meta := r.blockMeta[idx]
			return []llm.StreamEvent{{
				Kind: llm.EventToolCallArgsDelta,
				ToolCall: &llm.ToolCallDelta{
					Index:     idx,
					ID:        meta.id,
					ArgsDelta: delta.PartialJSON,
				},
			}}
		case sdk.ThinkingDelta:
			return []llm.StreamEvent{{Kind: llm.EventThinkingDelta, Text: delta.Thinking}}
		}
	case sdk.ContentBlockStopEvent:
		idx := int(v.Index)
		if r.blockKinds[idx] == "tool_use" {
			meta := r.blockMeta[idx]
			return []llm.StreamEvent{{
				Kind: llm.EventToolCallEnd,
				ToolCall: &llm.ToolCallDelta{
					Index: idx,
					ID:    meta.id,
					Name:  meta.name,
				},
			}}
		}
	case sdk.MessageStopEvent:
		usage := r.usage
		return []llm.StreamEvent{{
			Kind:         llm.EventDone,
			Usage:        &usage,
			FinishReason: r.finish,
		}}
	}

	return nil
}
