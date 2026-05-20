package ollama

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"sync"

	"github.com/costa92/llm-agent/llm"
	api "github.com/ollama/ollama/api"
)

var (
	_ llm.ChatModel  = (*Ollama)(nil)
	_ llm.ToolCaller = (*Ollama)(nil)
	_ llm.Embedder   = (*Ollama)(nil)
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
		ctx:      ctx,
		cancel:   cancel,
		respCh:   make(chan api.ChatResponse, 1),
		errCh:    make(chan error, 1),
		hasTools: len(o.tools) > 0,
		strategy: o.strategy,
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

func (o *Ollama) Embed(ctx context.Context, texts []string) ([]llm.Vector, llm.Usage, error) {
	if !o.info.Capabilities.Embeddings {
		return nil, llm.Usage{}, unsupportedEmbeddingError(o.info.Model)
	}
	if len(texts) == 0 {
		return []llm.Vector{}, llm.Usage{Source: llm.UsageReported}, nil
	}
	resp, err := o.client.Embed(ctx, &api.EmbedRequest{
		Model: o.info.Model,
		Input: append([]string(nil), texts...),
	})
	if err != nil {
		return nil, llm.Usage{}, o.wrapErr(err)
	}
	vectors := make([]llm.Vector, 0, len(resp.Embeddings))
	for _, item := range resp.Embeddings {
		vec := make(llm.Vector, len(item))
		copy(vec, item)
		vectors = append(vectors, vec)
	}
	usage := llm.Usage{
		InputTokens:  resp.PromptEvalCount,
		OutputTokens: 0,
		TotalTokens:  resp.PromptEvalCount,
		Source:       llm.UsageReported,
	}
	return vectors, usage, nil
}

func (o *Ollama) EmbedDimensions() int {
	return embeddingDimensionForModel(o.info.Model)
}

type ollamaStreamReader struct {
	mu     sync.Mutex
	ctx    context.Context
	cancel context.CancelFunc
	respCh chan api.ChatResponse
	errCh  chan error
	closed bool
	done   bool

	// K1 streaming state. When hasTools is false the reader streams plain
	// text deltas as chunks arrive (Pitfall 3: don't buffer when the caller
	// hasn't asked for tools). When hasTools is true and the strategy is
	// content-parsed (python_tag / qwen_xml), content is buffered until
	// Done so the parser can extract tool calls and emit synthetic K1
	// events; otherwise text streams chunk-by-chunk and native
	// Message.ToolCalls produce K1 events in-place.
	hasTools   bool
	strategy   ollamaToolStrategy
	pending    []llm.StreamEvent
	contentBuf strings.Builder
	nextIdx    int // synthesized stable per-tool-call Index
}

// bufferContent reports whether text content should be withheld until Done
// so a content-embedded tool-call parser (python_tag / qwen_xml) can run
// over the full message body without leaking raw tool markers as TextDelta.
func (r *ollamaStreamReader) bufferContent() bool {
	return r.hasTools && r.strategy.parserKind != ""
}

func (r *ollamaStreamReader) Next() (llm.StreamEvent, error) {
	for {
		r.mu.Lock()
		if r.closed {
			r.mu.Unlock()
			return llm.StreamEvent{}, io.EOF
		}
		if len(r.pending) > 0 {
			ev := r.pending[0]
			r.pending = r.pending[1:]
			r.mu.Unlock()
			return ev, nil
		}
		if r.done {
			r.mu.Unlock()
			return llm.StreamEvent{}, io.EOF
		}
		r.mu.Unlock()
		if err := r.ctx.Err(); err != nil {
			return llm.StreamEvent{}, err
		}

		// Only watch respCh + ctx.Done here. errCh is read only after
		// respCh closes — otherwise a buffered nil error sent before the
		// final chunk has been consumed can race-win the select and drop
		// the chunk (caught originally by TestStream_Ollama_NativeToolCalls
		// when the stream-completion signal was selected before chunk #2).
		select {
		case resp, ok := <-r.respCh:
			if !ok {
				if err := <-r.errCh; err != nil {
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
			r.ingest(resp)
		case <-r.ctx.Done():
			return llm.StreamEvent{}, r.ctx.Err()
		}
	}
}

// ingest converts one upstream chunk into 0..N typed K1 events queued on
// r.pending. Emission order within a chunk: pending buffered text →
// current chunk's text → native tool events (Start/ArgsDelta/End) →
// (on Done) content-parsed tools or flushed buffer → Done. Caller drains
// pending before reading the next chunk.
func (r *ollamaStreamReader) ingest(resp api.ChatResponse) {
	r.mu.Lock()
	defer r.mu.Unlock()

	hasNative := len(resp.Message.ToolCalls) > 0

	// If native tool events are about to be emitted, flush any prior
	// buffered text first so the K1 sequence is text → tools, not
	// tools → text.
	if hasNative && r.contentBuf.Len() > 0 {
		r.pending = append(r.pending, llm.StreamEvent{
			Kind: llm.EventTextDelta,
			Text: r.contentBuf.String(),
		})
		r.contentBuf.Reset()
	}

	// Current chunk's text: buffer only for content-parsed strategies
	// when no native tools have been seen yet AND none arrive this chunk.
	if resp.Message.Content != "" {
		if r.bufferContent() && !hasNative && r.nextIdx == 0 {
			r.contentBuf.WriteString(resp.Message.Content)
		} else {
			r.pending = append(r.pending, llm.StreamEvent{
				Kind: llm.EventTextDelta,
				Text: resp.Message.Content,
			})
		}
	}

	// Native tool calls (Ollama delivers each as a complete unit; emit
	// Start + ArgsDelta(full JSON) + End right away, with a stable
	// per-call Index synthesized in arrival order).
	for _, call := range resp.Message.ToolCalls {
		idx := r.nextIdx
		r.nextIdx++
		args, err := json.Marshal(call.Function.Arguments.ToMap())
		if err != nil {
			args = []byte("{}")
		}
		id := call.ID
		if id == "" {
			id = call.Function.Name
		}
		r.pending = append(r.pending,
			llm.StreamEvent{Kind: llm.EventToolCallStart, ToolCall: &llm.ToolCallDelta{Index: idx, ID: id, Name: call.Function.Name}},
			llm.StreamEvent{Kind: llm.EventToolCallArgsDelta, ToolCall: &llm.ToolCallDelta{Index: idx, ArgsDelta: string(args)}},
			llm.StreamEvent{Kind: llm.EventToolCallEnd, ToolCall: &llm.ToolCallDelta{Index: idx}},
		)
	}

	if !resp.Done {
		return
	}

	finish := mapOllamaDoneReason(resp.DoneReason)

	// Done: drain content buffer. If no native tool calls were emitted
	// this turn AND we're a content-parsed strategy, run the parser; tool
	// markers in the text are stripped before emission. Otherwise flush
	// the buffer as plain text.
	if r.contentBuf.Len() > 0 {
		buf := r.contentBuf.String()
		r.contentBuf.Reset()
		if r.bufferContent() && r.nextIdx == 0 {
			var (
				calls   []llm.ToolCall
				cleaned string
			)
			switch r.strategy.parserKind {
			case "python_tag":
				calls, cleaned, _ = parsePythonTagToolCalls(buf)
			case "qwen_json_or_xml":
				calls, cleaned, _ = parseQwenToolCalls(buf)
			}
			if len(calls) > 0 {
				if cleaned != "" {
					r.pending = append(r.pending, llm.StreamEvent{Kind: llm.EventTextDelta, Text: cleaned})
				}
				for _, call := range calls {
					idx := r.nextIdx
					r.nextIdx++
					args := string(call.Arguments)
					if args == "" {
						args = "{}"
					}
					r.pending = append(r.pending,
						llm.StreamEvent{Kind: llm.EventToolCallStart, ToolCall: &llm.ToolCallDelta{Index: idx, ID: call.ID, Name: call.Name}},
						llm.StreamEvent{Kind: llm.EventToolCallArgsDelta, ToolCall: &llm.ToolCallDelta{Index: idx, ArgsDelta: args}},
						llm.StreamEvent{Kind: llm.EventToolCallEnd, ToolCall: &llm.ToolCallDelta{Index: idx}},
					)
				}
			} else {
				r.pending = append(r.pending, llm.StreamEvent{Kind: llm.EventTextDelta, Text: buf})
			}
		} else {
			r.pending = append(r.pending, llm.StreamEvent{Kind: llm.EventTextDelta, Text: buf})
		}
	}

	if r.nextIdx > 0 && (finish == llm.FinishReasonStop || finish == llm.FinishReasonUnknown) {
		finish = llm.FinishReasonToolCalls
	}

	usage := llm.Usage{
		InputTokens:  resp.PromptEvalCount,
		OutputTokens: resp.EvalCount,
		TotalTokens:  resp.PromptEvalCount + resp.EvalCount,
		Source:       llm.UsageReported,
	}
	r.pending = append(r.pending, llm.StreamEvent{
		Kind:         llm.EventDone,
		Usage:        &usage,
		FinishReason: finish,
	})
	r.done = true
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
