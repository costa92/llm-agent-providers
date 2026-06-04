package volcengine

import (
	"context"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/costa92/llm-agent-contract/llm"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime/utils"
)

var (
	_ llm.ChatModel      = (*Volcengine)(nil)
	_ llm.ToolCaller     = (*Volcengine)(nil)
	_ llm.ImageGenerator = (*Volcengine)(nil)
	_ llm.Embedder       = (*Volcengine)(nil)
)

// Volcengine is a 火山方舟 Ark adapter bound to a single model.
type Volcengine struct {
	client          *arkruntime.Client
	info            llm.ProviderInfo
	tools           []llm.Tool
	timeout         time.Duration
	embedDimensions int
	extraHeaders    map[string]string
}

// newArkClient builds the arkruntime client from config. WithRetryTimes(0)
// keeps our single-attempt policy consistent with the other adapters.
func newArkClient(cfg config) *arkruntime.Client {
	setters := []arkruntime.ConfigOption{
		arkruntime.WithRegion(cfg.region),
		arkruntime.WithRetryTimes(0),
	}
	if cfg.baseURL != "" {
		setters = append(setters, arkruntime.WithBaseUrl(cfg.baseURL))
	}
	if cfg.httpClient != nil {
		setters = append(setters, arkruntime.WithHTTPClient(cfg.httpClient))
	}
	if cfg.timeout > 0 {
		setters = append(setters, arkruntime.WithTimeout(cfg.timeout))
	}
	return arkruntime.NewClientWithApiKey(cfg.apiKey, setters...)
}

// requestOptions returns the per-request setters (custom headers).
func (v *Volcengine) requestOptions() []arkruntime.RequestOption {
	if len(v.extraHeaders) == 0 {
		return nil
	}
	return []arkruntime.RequestOption{arkruntime.WithCustomHeaders(v.extraHeaders)}
}

// Info returns the bound provider+model identity and capabilities.
func (v *Volcengine) Info() llm.ProviderInfo { return v.info }

// WithTools returns a new tool-bound ToolCaller (immutable; receiver unchanged).
func (v *Volcengine) WithTools(tools []llm.Tool) (llm.ToolCaller, error) {
	cp := *v
	cp.tools = append([]llm.Tool(nil), tools...)
	return &cp, nil
}

// Generate runs a one-shot chat completion.
func (v *Volcengine) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := v.toSDKRequest(req)
	resp, err := v.client.CreateChatCompletion(ctx, sdkReq, v.requestOptions()...)
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	return v.fromSDKResponse(resp), nil
}

// Stream runs a streaming chat completion, returning a typed K1 reader.
func (v *Volcengine) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	sdkReq := v.toSDKStreamRequest(req)
	opts := v.requestOptions()
	return &volcengineStreamReader{
		open: func() (*utils.ChatCompletionStreamReader, error) {
			return v.client.CreateChatCompletionStream(ctx, sdkReq, opts...)
		},
		toolIndexes: make(map[int]struct{}),
	}, nil
}

// EmbedDimensions returns the bound embedding dimensionality, or 0 for
// non-embedding models.
func (v *Volcengine) EmbedDimensions() int { return v.embedDimensions }

type volcengineStreamReader struct {
	mu            sync.Mutex
	open          func() (*utils.ChatCompletionStreamReader, error)
	stream        *utils.ChatCompletionStreamReader
	queue         []llm.StreamEvent
	retried       bool
	deliveredByte bool
	closed        bool
	doneEmitted   bool

	toolIndexes map[int]struct{}
	usage       *llm.Usage
	lastFinish  llm.FinishReason
}

// Next pulls the next typed stream event. It opens the upstream stream lazily
// on the first call and retries exactly once before any byte is delivered.
func (r *volcengineStreamReader) Next() (llm.StreamEvent, error) {
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
			s, err := r.open()
			if err != nil {
				if !r.deliveredByte && !r.retried {
					r.retried = true
					continue
				}
				return llm.StreamEvent{}, wrapErr(err)
			}
			r.stream = s
		}

		chunk, err := r.stream.Recv()
		if err != nil {
			if isEOF(err) {
				// Clean end: emit the terminal Done once, then EOF.
				if !r.doneEmitted {
					r.doneEmitted = true
					r.queue = append(r.queue, r.doneEvent())
					continue
				}
				_ = r.stream.Close()
				r.stream = nil
				return llm.StreamEvent{}, io.EOF
			}
			_ = r.stream.Close()
			r.stream = nil
			if !r.deliveredByte && !r.retried {
				r.retried = true
				continue
			}
			return llm.StreamEvent{}, wrapErr(err)
		}
		r.queue = append(r.queue, r.chunkEvents(chunk)...)
	}
}

// Close is idempotent; safe to call from another goroutine.
func (r *volcengineStreamReader) Close() error {
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

// doneEvent builds the terminal EventDone from accumulated usage/finish.
func (r *volcengineStreamReader) doneEvent() llm.StreamEvent {
	usage := r.usage
	if usage == nil {
		u := llm.Usage{Source: llm.UsageUnknown}
		usage = &u
	}
	return llm.StreamEvent{
		Kind:         llm.EventDone,
		Usage:        usage,
		FinishReason: r.lastFinish,
	}
}

// chunkEvents decomposes one SDK stream chunk into typed events. A usage-only
// chunk (empty Choices, non-nil Usage) records usage but emits nothing — the
// terminal Done is synthesized at io.EOF.
func (r *volcengineStreamReader) chunkEvents(chunk model.ChatCompletionStreamResponse) []llm.StreamEvent {
	if chunk.Usage != nil {
		r.usage = &llm.Usage{
			InputTokens:  chunk.Usage.PromptTokens,
			OutputTokens: chunk.Usage.CompletionTokens,
			TotalTokens:  chunk.Usage.TotalTokens,
			Source:       llm.UsageReported,
		}
	}

	var events []llm.StreamEvent
	for _, choice := range chunk.Choices {
		if choice == nil {
			continue
		}
		if choice.Delta.Content != "" {
			events = append(events, llm.StreamEvent{
				Kind: llm.EventTextDelta,
				Text: choice.Delta.Content,
			})
		}
		for _, tool := range choice.Delta.ToolCalls {
			if tool == nil {
				continue
			}
			idx := 0
			if tool.Index != nil {
				idx = *tool.Index
			}
			r.toolIndexes[idx] = struct{}{}
			if tool.ID != "" || tool.Function.Name != "" {
				events = append(events, llm.StreamEvent{
					Kind: llm.EventToolCallStart,
					ToolCall: &llm.ToolCallDelta{
						Index: idx,
						ID:    tool.ID,
						Name:  tool.Function.Name,
					},
				})
			}
			if tool.Function.Arguments != "" {
				events = append(events, llm.StreamEvent{
					Kind: llm.EventToolCallArgsDelta,
					ToolCall: &llm.ToolCallDelta{
						Index:     idx,
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
				for i := range r.toolIndexes {
					indexes = append(indexes, i)
				}
				sort.Ints(indexes)
				for _, i := range indexes {
					events = append(events, llm.StreamEvent{
						Kind:     llm.EventToolCallEnd,
						ToolCall: &llm.ToolCallDelta{Index: i},
					})
				}
				r.toolIndexes = make(map[int]struct{})
			}
		}
	}
	return events
}

// isEOF detects the io.EOF sentinel returned by the SDK stream reader at
// data: [DONE].
func isEOF(err error) bool {
	return err == io.EOF
}
