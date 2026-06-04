package google

import (
	"context"
	"encoding/json"
	"io"
	"iter"
	"sync"
	"time"

	"google.golang.org/genai"

	"github.com/costa92/llm-agent-contract/llm"
)

var (
	_ llm.ChatModel  = (*Google)(nil)
	_ llm.ToolCaller = (*Google)(nil)
	// _ llm.Embedder       = (*Google)(nil) // Task 5 (+ Embed)
	// _ llm.ImageGenerator = (*Google)(nil) // Task 6 (+ GenerateImage)
)

// Google is a Gemini provider bound to one model. Safe for concurrent use.
type Google struct {
	client     *genai.Client
	info       llm.ProviderInfo
	tools      []llm.Tool
	taskType   string
	dimensions int
	timeout    time.Duration
}

// withTimeout derives a context bounded by the configured per-request timeout.
// We apply the deadline on the caller's context here rather than via
// genai's HTTPOptions.Timeout because the SDK (v1.59.0) cancels the
// stream context with a defer that fires before the iterator is drained,
// which would truncate streaming responses after the first chunk.
func (g *Google) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if g.timeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, g.timeout)
}

// Info returns the bound (provider × model) identity and capabilities.
func (g *Google) Info() llm.ProviderInfo { return g.info }

// Generate runs a one-shot chat completion against the bound model.
func (g *Google) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	ctx, cancel := g.withTimeout(ctx)
	defer cancel()
	resp, err := g.client.Models.GenerateContent(ctx, g.info.Model, toContents(req), g.toGenConfig(req))
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	if blocked := blockedPromptErr(resp); blocked != nil {
		return llm.Response{}, blocked
	}
	return g.fromResponse(resp), nil
}

// WithTools returns a new ToolCaller bound to the given tools (immutable;
// the receiver is unchanged).
func (g *Google) WithTools(tools []llm.Tool) (llm.ToolCaller, error) {
	cp := *g
	cp.tools = append([]llm.Tool(nil), tools...)
	return &cp, nil
}

// Stream runs a streaming chat completion. The returned reader lazily opens
// the upstream iterator on first Next() and MUST be Closed by the caller.
//
// The per-request timeout is applied to a context whose cancel func lives for
// the lifetime of the reader (released on exhaustion and on Close), NOT via
// genai's HTTPOptions.Timeout — the SDK cancels that context too eagerly,
// truncating the stream after the first chunk.
func (g *Google) Stream(ctx context.Context, req llm.Request) (llm.StreamReader, error) {
	streamCtx, cancel := g.withTimeout(ctx)
	return &googleStreamReader{
		cancel: cancel,
		open: func() iter.Seq2[*genai.GenerateContentResponse, error] {
			return g.client.Models.GenerateContentStream(streamCtx, g.info.Model, toContents(req), g.toGenConfig(req))
		},
	}, nil
}

// googleStreamReader bridges genai's iter.Seq2 to the pull-based
// llm.StreamReader via iter.Pull2. One upstream chunk decomposes into many
// llm.StreamEvents; streamed tool calls arrive complete in one chunk so
// Start+ArgsDelta(full)+End are emitted together per call.
type googleStreamReader struct {
	mu     sync.Mutex
	open   func() iter.Seq2[*genai.GenerateContentResponse, error]
	next   func() (*genai.GenerateContentResponse, error, bool)
	stop   func()
	cancel context.CancelFunc
	queue  []llm.StreamEvent
	closed bool

	lastFinish llm.FinishReason
	lastUsage  *llm.Usage
	sawAny     bool
}

func (r *googleStreamReader) Next() (llm.StreamEvent, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for {
		if r.closed {
			return llm.StreamEvent{}, io.EOF
		}
		if len(r.queue) > 0 {
			ev := r.queue[0]
			r.queue = r.queue[1:]
			return ev, nil
		}
		if r.next == nil {
			r.next, r.stop = iter.Pull2(r.open())
		}
		chunk, err, ok := r.next()
		if !ok {
			// Upstream exhausted: emit terminal Done once.
			r.stop()
			if r.cancel != nil {
				r.cancel()
				r.cancel = nil
			}
			r.closed = true
			usage := r.lastUsage
			if usage == nil {
				usage = &llm.Usage{Source: llm.UsageUnknown}
			}
			return llm.StreamEvent{
				Kind:         llm.EventDone,
				Usage:        usage,
				FinishReason: r.lastFinish,
			}, nil
		}
		if err != nil {
			r.stop()
			if r.cancel != nil {
				r.cancel()
				r.cancel = nil
			}
			r.closed = true
			return llm.StreamEvent{}, wrapErr(err)
		}
		r.sawAny = true
		r.queue = append(r.queue, r.chunkEvents(chunk)...)
	}
}

func (r *googleStreamReader) chunkEvents(chunk *genai.GenerateContentResponse) []llm.StreamEvent {
	var events []llm.StreamEvent
	if len(chunk.Candidates) == 0 {
		return events
	}
	cand := chunk.Candidates[0]
	if cand.Content != nil {
		for i, part := range cand.Content.Parts {
			if part.Text != "" {
				events = append(events, llm.StreamEvent{Kind: llm.EventTextDelta, Text: part.Text})
			}
			if part.FunctionCall != nil {
				fc := part.FunctionCall
				args, err := json.Marshal(fc.Args)
				if err != nil || fc.Args == nil {
					args = []byte("{}")
				}
				events = append(events,
					llm.StreamEvent{Kind: llm.EventToolCallStart, ToolCall: &llm.ToolCallDelta{Index: i, ID: fc.ID, Name: fc.Name}},
					llm.StreamEvent{Kind: llm.EventToolCallArgsDelta, ToolCall: &llm.ToolCallDelta{Index: i, ArgsDelta: string(args)}},
					llm.StreamEvent{Kind: llm.EventToolCallEnd, ToolCall: &llm.ToolCallDelta{Index: i}},
				)
			}
		}
	}
	if cand.FinishReason != "" {
		r.lastFinish = mapFinishReason(cand.FinishReason)
	}
	if um := chunk.UsageMetadata; um != nil {
		r.lastUsage = &llm.Usage{
			InputTokens:  int(um.PromptTokenCount),
			OutputTokens: int(um.CandidatesTokenCount),
			TotalTokens:  int(um.TotalTokenCount),
			Source:       llm.UsageReported,
		}
	}
	return events
}

func (r *googleStreamReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	if r.stop != nil {
		r.stop()
		r.stop = nil
	}
	if r.cancel != nil {
		r.cancel()
		r.cancel = nil
	}
	return nil
}

// EmbedDimensions returns the embedding width for the bound model, honoring
// WithDimensions; 0 when the bound model is not an embedding model.
func (g *Google) EmbedDimensions() int {
	if !g.info.Capabilities.Embeddings {
		return 0
	}
	if g.dimensions > 0 {
		return g.dimensions
	}
	switch g.info.Model {
	case "gemini-embedding-001":
		return 3072
	case "text-embedding-004":
		return 768
	default:
		return 0
	}
}
