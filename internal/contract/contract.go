package contract

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/costa92/llm-agent/llm"
)

type Fixture struct {
	Scenario string `json:"scenario"`
	Tools    []struct {
		Name        string `json:"name"`
		Description string `json:"description,omitempty"`
		Parameters  string `json:"parameters"`
	} `json:"tools,omitempty"`
	Request struct {
		Method         string   `json:"method"`
		Path           string   `json:"path"`
		BodyAssertions []string `json:"body_assertions"`
	} `json:"request"`
	Response struct {
		Status  int               `json:"status"`
		Headers map[string]string `json:"headers"`
		Body    string            `json:"body"`
	} `json:"response"`
	Expect struct {
		ErrorType         string `json:"error_type,omitempty"`
		ResponseText      string `json:"response_text,omitempty"`
		FinishReason      string `json:"finish_reason,omitempty"`
		UsageInputTokens  int    `json:"usage_input_tokens,omitempty"`
		UsageOutputTokens int    `json:"usage_output_tokens,omitempty"`
		UsageSource       string `json:"usage_source,omitempty"`
		Provider          string `json:"provider,omitempty"`
		EmbeddingCount    int    `json:"embedding_count,omitempty"`
		EmbeddingDims     []int  `json:"embedding_dims,omitempty"`
		ToolCalls         []struct {
			ID        string `json:"id,omitempty"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"tool_calls,omitempty"`
		// StreamSequence is the flat list of expected Kind names in order, used by
		// AssertStreamToolCalls to pin K1 conformance. Valid entries:
		//   "text"        EventTextDelta
		//   "tool_start"  EventToolCallStart
		//   "tool_args"   EventToolCallArgsDelta
		//   "tool_end"    EventToolCallEnd
		//   "thinking"    EventThinkingDelta
		//   "done"        EventDone
		// When empty, AssertStreamToolCalls is not run.
		StreamSequence []string `json:"stream_sequence,omitempty"`
	} `json:"expect"`
}

func LoadFixture(t *testing.T, provider, scenario string) Fixture {
	t.Helper()
	p := filepath.Join("testdata", provider, scenario+".json")
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("LoadFixture(%s/%s): %v", provider, scenario, err)
	}
	var f Fixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatalf("LoadFixture(%s/%s): unmarshal: %v", provider, scenario, err)
	}
	return f
}

func NewMockServer(t *testing.T, f Fixture) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.Request.Method != "" && r.Method != f.Request.Method {
			t.Errorf("method: got %q want %q", r.Method, f.Request.Method)
		}
		if f.Request.Path != "" && !strings.HasPrefix(r.URL.Path, f.Request.Path) {
			t.Errorf("path: got %q want prefix %q", r.URL.Path, f.Request.Path)
		}
		body, _ := io.ReadAll(r.Body)
		for _, want := range f.Request.BodyAssertions {
			if !assertBody(string(body), want) {
				t.Errorf("body assertion failed: %q not satisfied by %q", want, string(body))
			}
		}
		for k, v := range f.Response.Headers {
			w.Header().Set(k, v)
		}
		w.WriteHeader(f.Response.Status)
		_, _ = w.Write([]byte(f.Response.Body))
	}))
}

func assertBody(body, assertion string) bool {
	s := strings.TrimSpace(assertion)
	if i := strings.Index(s, "="); i >= 0 && !strings.Contains(s[:i], " ") {
		return strings.Contains(body, s[i+1:])
	}
	return strings.Contains(body, s)
}

type ChatModelFactory func(baseURL string) (llm.ChatModel, error)

func AssertGenerate(t *testing.T, model llm.ChatModel, f Fixture) {
	t.Helper()
	req := llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hello"}},
	}
	resp, err := model.Generate(t.Context(), req)
	assertResponse(t, resp, err, f, "Generate")
}

func AssertStream(t *testing.T, model llm.ChatModel, f Fixture) {
	t.Helper()
	req := llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hello"}},
	}
	sr, err := model.Stream(t.Context(), req)
	if err != nil {
		t.Fatalf("Stream: unexpected error creating stream: %v", err)
	}
	resp, err := llm.AccumulateStream(sr)
	info := model.Info()
	if resp.Provider == "" {
		resp.Provider = info.Provider
	}
	if resp.Model == "" {
		resp.Model = info.Model
	}
	assertResponse(t, resp, err, f, "Stream")
}

// AssertStreamToolCalls drains the stream WITHOUT AccumulateStream collapse,
// pinning K1 conformance: Kind sequence matches Expect.StreamSequence and
// per-Index ToolCall payload satisfies the per-Kind population rules from
// llm/stream.go:33-40.
//
// Index stability invariant: across Start → ArgsDelta(s) → End for a single
// tool call, Index is identical. After End for an Index, that Index must not
// be reused for a later Start.
//
// When f.Expect.ToolCalls is non-empty, the accumulated (Name, Arguments)
// pairs per Index are verified against the expected final tool calls.
func AssertStreamToolCalls(t *testing.T, model llm.ChatModel, f Fixture) {
	t.Helper()
	if len(f.Expect.StreamSequence) == 0 {
		t.Fatal("AssertStreamToolCalls: fixture has empty Expect.StreamSequence")
	}

	tools := make([]llm.Tool, 0, len(f.Tools))
	for _, tool := range f.Tools {
		tools = append(tools, llm.Tool{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  []byte(tool.Parameters),
		})
	}

	bound := model
	if len(tools) > 0 {
		tc, ok := model.(llm.ToolCaller)
		if !ok {
			t.Fatalf("model %T does not implement llm.ToolCaller", model)
		}
		b, err := tc.WithTools(tools)
		if err != nil {
			t.Fatalf("WithTools: %v", err)
		}
		bound = b
	}

	req := llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "use tools"}},
	}
	sr, err := bound.Stream(t.Context(), req)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer sr.Close()

	events := make([]llm.StreamEvent, 0, len(f.Expect.StreamSequence))
	for {
		ev, err := sr.Next()
		if err != nil {
			if isEOFErr(err) {
				break
			}
			t.Fatalf("Next: %v", err)
		}
		events = append(events, ev)
		if ev.Kind == llm.EventDone {
			// EventDone is terminal per K1 contract; subsequent reads should
			// return io.EOF. Don't break here so we exercise that.
		}
	}

	// 1. Kind sequence equals expected.
	got := make([]string, 0, len(events))
	for _, ev := range events {
		got = append(got, kindName(ev.Kind))
	}
	if len(got) != len(f.Expect.StreamSequence) {
		t.Fatalf("stream kind count: got %d %v, want %d %v", len(got), got, len(f.Expect.StreamSequence), f.Expect.StreamSequence)
	}
	for i, want := range f.Expect.StreamSequence {
		if got[i] != want {
			t.Fatalf("stream kind[%d]: got %q want %q (full got=%v want=%v)", i, got[i], want, got, f.Expect.StreamSequence)
		}
	}

	// 2. Per-Kind payload + Index stability invariants.
	// seenIndices state: 0 = unseen, 1 = started, 2 = ended.
	seenIndices := map[int]int{}
	// argsByIndex accumulates ArgsDelta concatenation per Index for the optional
	// ToolCalls cross-check below.
	argsByIndex := map[int]string{}
	nameByIndex := map[int]string{}
	idByIndex := map[int]string{}
	indexOrder := make([]int, 0)
	for i, ev := range events {
		switch ev.Kind {
		case llm.EventTextDelta, llm.EventThinkingDelta:
			if ev.Text == "" {
				t.Errorf("event[%d] kind=%v: empty Text (K1 rule: text/thinking must populate Text)", i, ev.Kind)
			}
		case llm.EventToolCallStart:
			if ev.ToolCall == nil {
				t.Fatalf("event[%d] EventToolCallStart: ToolCall == nil", i)
			}
			if ev.ToolCall.Index < 0 {
				t.Errorf("event[%d] EventToolCallStart: Index = %d, want >= 0", i, ev.ToolCall.Index)
			}
			if ev.ToolCall.Name == "" {
				t.Errorf("event[%d] EventToolCallStart: Name is empty (K1 rule)", i)
			}
			if state := seenIndices[ev.ToolCall.Index]; state == 2 {
				t.Errorf("event[%d] EventToolCallStart: Index %d reused after End (K1 rule: indices unique post-End)", i, ev.ToolCall.Index)
			}
			seenIndices[ev.ToolCall.Index] = 1
			nameByIndex[ev.ToolCall.Index] = ev.ToolCall.Name
			if ev.ToolCall.ID != "" {
				idByIndex[ev.ToolCall.Index] = ev.ToolCall.ID
			}
			if _, ok := argsByIndex[ev.ToolCall.Index]; !ok {
				indexOrder = append(indexOrder, ev.ToolCall.Index)
				argsByIndex[ev.ToolCall.Index] = ""
			}
		case llm.EventToolCallArgsDelta:
			if ev.ToolCall == nil {
				t.Fatalf("event[%d] EventToolCallArgsDelta: ToolCall == nil", i)
			}
			if seenIndices[ev.ToolCall.Index] != 1 {
				t.Errorf("event[%d] EventToolCallArgsDelta: Index %d has no preceding Start (state=%d)", i, ev.ToolCall.Index, seenIndices[ev.ToolCall.Index])
			}
			argsByIndex[ev.ToolCall.Index] += ev.ToolCall.ArgsDelta
		case llm.EventToolCallEnd:
			if ev.ToolCall == nil {
				t.Fatalf("event[%d] EventToolCallEnd: ToolCall == nil", i)
			}
			if seenIndices[ev.ToolCall.Index] != 1 {
				t.Errorf("event[%d] EventToolCallEnd: Index %d has no matching Start (state=%d)", i, ev.ToolCall.Index, seenIndices[ev.ToolCall.Index])
			}
			seenIndices[ev.ToolCall.Index] = 2
		case llm.EventDone:
			if i != len(events)-1 {
				t.Errorf("event[%d] EventDone: not the last event (followed by %d more)", i, len(events)-1-i)
			}
			if ev.Usage == nil {
				t.Errorf("event[%d] EventDone: Usage == nil (K1 rule: Done populates Usage)", i)
			}
			if ev.FinishReason == "" {
				t.Errorf("event[%d] EventDone: FinishReason == \"\" (K1 rule: Done populates FinishReason)", i)
			}
		}
	}

	// 3. Optional cross-check: expected accumulated tool calls match.
	if len(f.Expect.ToolCalls) > 0 {
		if len(f.Expect.ToolCalls) != len(indexOrder) {
			t.Fatalf("ToolCalls count: got %d (indices=%v), want %d", len(indexOrder), indexOrder, len(f.Expect.ToolCalls))
		}
		for i, want := range f.Expect.ToolCalls {
			idx := indexOrder[i]
			if nameByIndex[idx] != want.Name {
				t.Errorf("ToolCalls[%d] (Index %d) Name: got %q want %q", i, idx, nameByIndex[idx], want.Name)
			}
			if want.ID != "" && idByIndex[idx] != want.ID {
				t.Errorf("ToolCalls[%d] (Index %d) ID: got %q want %q", i, idx, idByIndex[idx], want.ID)
			}
			if !jsonEqual([]byte(argsByIndex[idx]), []byte(want.Arguments)) {
				t.Errorf("ToolCalls[%d] (Index %d) Arguments: got %q want %q", i, idx, argsByIndex[idx], want.Arguments)
			}
		}
	}
}

func kindName(k llm.StreamEventKind) string {
	switch k {
	case llm.EventTextDelta:
		return "text"
	case llm.EventToolCallStart:
		return "tool_start"
	case llm.EventToolCallArgsDelta:
		return "tool_args"
	case llm.EventToolCallEnd:
		return "tool_end"
	case llm.EventThinkingDelta:
		return "thinking"
	case llm.EventDone:
		return "done"
	default:
		return "unknown"
	}
}

func isEOFErr(err error) bool {
	return errors.Is(err, io.EOF)
}

func AssertToolCalling(t *testing.T, model llm.ChatModel, f Fixture) {
	t.Helper()
	tc, ok := model.(llm.ToolCaller)
	if !ok {
		t.Fatalf("model %T does not implement llm.ToolCaller", model)
	}
	tools := make([]llm.Tool, 0, len(f.Tools))
	for _, tool := range f.Tools {
		tools = append(tools, llm.Tool{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  []byte(tool.Parameters),
		})
	}
	bound, err := tc.WithTools(tools)
	if f.Expect.ErrorType == "CapabilityNotSupported" {
		if err == nil {
			t.Fatal("WithTools(): error = nil, want capability error")
		}
		if !errors.Is(err, llm.ErrCapabilityNotSupported) {
			t.Fatalf("errors.Is(err, ErrCapabilityNotSupported) = false, err=%T %v", err, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}
	req := llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "use tools"}},
	}
	resp, err := bound.Generate(t.Context(), req)
	assertResponse(t, resp, err, f, "Generate")
}

func AssertEmbed(t *testing.T, model llm.ChatModel, f Fixture) {
	t.Helper()
	embedder, ok := model.(llm.Embedder)
	if f.Expect.ErrorType == "CapabilityNotSupported" {
		if ok {
			t.Fatalf("model %T unexpectedly implements llm.Embedder", model)
		}
		return
	}
	if !ok {
		t.Fatalf("model %T does not implement llm.Embedder", model)
	}
	vectors, usage, err := embedder.Embed(t.Context(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed: unexpected error: %v", err)
	}
	if f.Expect.EmbeddingCount != 0 && len(vectors) != f.Expect.EmbeddingCount {
		t.Fatalf("embedding count: got %d want %d", len(vectors), f.Expect.EmbeddingCount)
	}
	if len(f.Expect.EmbeddingDims) != 0 {
		if len(vectors) != len(f.Expect.EmbeddingDims) {
			t.Fatalf("embedding dims len: got %d vectors want %d dim assertions", len(vectors), len(f.Expect.EmbeddingDims))
		}
		for i, want := range f.Expect.EmbeddingDims {
			if got := len(vectors[i]); got != want {
				t.Fatalf("embedding[%d] dim: got %d want %d", i, got, want)
			}
		}
	}
	if f.Expect.UsageInputTokens != 0 && usage.InputTokens != f.Expect.UsageInputTokens {
		t.Fatalf("Usage.InputTokens: got %d want %d", usage.InputTokens, f.Expect.UsageInputTokens)
	}
	if f.Expect.UsageSource != "" && string(usage.Source) != f.Expect.UsageSource {
		t.Fatalf("Usage.Source: got %q want %q", usage.Source, f.Expect.UsageSource)
	}
}

func assertResponse(t *testing.T, resp llm.Response, err error, f Fixture, op string) {
	t.Helper()
	switch f.Expect.ErrorType {
	case "":
		if err != nil {
			t.Fatalf("%s: unexpected error: %v", op, err)
		}
		if f.Expect.ResponseText != "" && resp.Text != f.Expect.ResponseText {
			t.Errorf("Text: got %q want %q", resp.Text, f.Expect.ResponseText)
		}
		if f.Expect.FinishReason != "" && string(resp.FinishReason) != f.Expect.FinishReason {
			t.Errorf("FinishReason: got %q want %q", resp.FinishReason, f.Expect.FinishReason)
		}
		if f.Expect.UsageInputTokens != 0 && resp.Usage.InputTokens != f.Expect.UsageInputTokens {
			t.Errorf("Usage.InputTokens: got %d want %d", resp.Usage.InputTokens, f.Expect.UsageInputTokens)
		}
		if f.Expect.UsageOutputTokens != 0 && resp.Usage.OutputTokens != f.Expect.UsageOutputTokens {
			t.Errorf("Usage.OutputTokens: got %d want %d", resp.Usage.OutputTokens, f.Expect.UsageOutputTokens)
		}
		if f.Expect.UsageSource != "" && string(resp.Usage.Source) != f.Expect.UsageSource {
			t.Errorf("Usage.Source: got %q want %q", resp.Usage.Source, f.Expect.UsageSource)
		}
		if f.Expect.Provider != "" && resp.Provider != f.Expect.Provider {
			t.Errorf("Provider: got %q want %q", resp.Provider, f.Expect.Provider)
		}
		if len(f.Expect.ToolCalls) != 0 {
			if len(resp.ToolCalls) != len(f.Expect.ToolCalls) {
				t.Fatalf("ToolCalls len: got %d want %d", len(resp.ToolCalls), len(f.Expect.ToolCalls))
			}
			for i, want := range f.Expect.ToolCalls {
				got := resp.ToolCalls[i]
				if want.ID != "" && got.ID != want.ID {
					t.Errorf("ToolCalls[%d].ID: got %q want %q", i, got.ID, want.ID)
				}
				if got.Name != want.Name {
					t.Errorf("ToolCalls[%d].Name: got %q want %q", i, got.Name, want.Name)
				}
				if !jsonEqual(got.Arguments, []byte(want.Arguments)) {
					t.Errorf("ToolCalls[%d].Arguments: got %s want %s", i, string(got.Arguments), want.Arguments)
				}
			}
		}
	case "AuthError":
		var e *llm.AuthError
		if !errors.As(err, &e) {
			t.Errorf("expected *llm.AuthError, got %T: %v", err, err)
		}
	case "RateLimitError":
		var e *llm.RateLimitError
		if !errors.As(err, &e) {
			t.Errorf("expected *llm.RateLimitError, got %T: %v", err, err)
		}
	case "InvalidRequestError":
		var e *llm.InvalidRequestError
		if !errors.As(err, &e) {
			t.Errorf("expected *llm.InvalidRequestError, got %T: %v", err, err)
		}
	case "TransientError":
		var e *llm.TransientError
		if !errors.As(err, &e) {
			t.Errorf("expected *llm.TransientError, got %T: %v", err, err)
		}
	case "CapabilityNotSupported":
		if !errors.Is(err, llm.ErrCapabilityNotSupported) {
			t.Errorf("expected ErrCapabilityNotSupported, got %T: %v", err, err)
		}
	default:
		t.Errorf("unknown Fixture.Expect.ErrorType: %q", f.Expect.ErrorType)
	}
}

func jsonEqual(a, b []byte) bool {
	var av any
	var bv any
	if err := json.Unmarshal(a, &av); err != nil {
		return string(a) == string(b)
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		return string(a) == string(b)
	}
	ab, _ := json.Marshal(av)
	bb, _ := json.Marshal(bv)
	return string(ab) == string(bb)
}
