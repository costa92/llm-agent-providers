package ollama

import (
	"context"
	"io"
	"sync"

	"github.com/costa92/llm-agent/llm"
	api "github.com/ollama/ollama/api"
)

var (
	_ llm.ChatModel  = (*Ollama)(nil)
	_ llm.ToolCaller = (*Ollama)(nil)
)

type Ollama struct {
	client     *api.Client
	info       llm.ProviderInfo
	lastStatus *int32
	tools      []llm.Tool
	strategy   ollamaToolStrategy
}

func (o *Ollama) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := o.toSDKRequest(req)
	var captured api.ChatResponse
	err := o.client.Chat(ctx, sdkReq, func(resp api.ChatResponse) error {
		captured = resp
		return nil
	})
	if err != nil {
		return llm.Response{}, o.wrapErr(err)
	}
	return o.fromSDKResponse(captured), nil
}

func (o *Ollama) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	ctx, cancel := context.WithCancel(ctx)
	sr := &ollamaStreamReader{
		ctx:    ctx,
		cancel: cancel,
		respCh: make(chan api.ChatResponse, 1),
		errCh:  make(chan error, 1),
	}

	go func() {
		defer close(sr.respCh)
		sdkReq := o.toSDKStreamRequest(req)
		err := o.client.Chat(ctx, sdkReq, func(resp api.ChatResponse) error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case sr.respCh <- resp:
				return nil
			}
		})
		sr.errCh <- o.wrapErr(err)
		close(sr.errCh)
	}()

	return sr, nil
}

func (o *Ollama) Info() llm.ProviderInfo { return o.info }

func (o *Ollama) WithTools(tools []llm.Tool) (llm.ToolCaller, error) {
	if !o.strategy.supportsTool {
		return nil, unsupportedToolError(o.info.Model)
	}
	cp := *o
	cp.tools = append([]llm.Tool(nil), tools...)
	return &cp, nil
}

type ollamaStreamReader struct {
	mu     sync.Mutex
	ctx    context.Context
	cancel context.CancelFunc
	respCh chan api.ChatResponse
	errCh  chan error
	closed bool
	done   bool
}

func (r *ollamaStreamReader) Next() (llm.StreamEvent, error) {
	for {
		r.mu.Lock()
		if r.closed {
			r.mu.Unlock()
			return llm.StreamEvent{}, io.EOF
		}
		if r.done {
			r.mu.Unlock()
			return llm.StreamEvent{}, io.EOF
		}
		r.mu.Unlock()
		if err := r.ctx.Err(); err != nil {
			return llm.StreamEvent{}, err
		}

		select {
		case resp, ok := <-r.respCh:
			if !ok {
				err, ok := <-r.errCh
				if ok && err != nil {
					return llm.StreamEvent{}, err
				}
				if err := r.ctx.Err(); err != nil {
					return llm.StreamEvent{}, err
				}
				r.mu.Lock()
				r.done = true
				r.mu.Unlock()
				return llm.StreamEvent{}, io.EOF
			}
			if resp.Done {
				r.mu.Lock()
				r.done = true
				r.mu.Unlock()
				usage := llm.Usage{
					InputTokens:  resp.PromptEvalCount,
					OutputTokens: resp.EvalCount,
					TotalTokens:  resp.PromptEvalCount + resp.EvalCount,
					Source:       llm.UsageReported,
				}
				return llm.StreamEvent{
					Kind:         llm.EventDone,
					Usage:        &usage,
					FinishReason: mapOllamaDoneReason(resp.DoneReason),
				}, nil
			}
			if resp.Message.Content == "" {
				continue
			}
			return llm.StreamEvent{
				Kind: llm.EventTextDelta,
				Text: resp.Message.Content,
			}, nil
		case err, ok := <-r.errCh:
			if ok && err != nil {
				return llm.StreamEvent{}, err
			}
			if err := r.ctx.Err(); err != nil {
				return llm.StreamEvent{}, err
			}
			r.mu.Lock()
			r.done = true
			r.mu.Unlock()
			return llm.StreamEvent{}, io.EOF
		}
	}
}

func (r *ollamaStreamReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	r.cancel()
	return nil
}
