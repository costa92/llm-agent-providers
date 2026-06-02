package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestNew_RequiresModel(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("New() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), "WithModel is required") {
		t.Fatalf("New() error = %q, want WithModel is required", err)
	}
}

func TestInfo_OpenAI(t *testing.T) {
	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if info.Provider != "openai" {
		t.Fatalf("Provider = %q, want openai", info.Provider)
	}
	if info.Model != "gpt-4o-mini" {
		t.Fatalf("Model = %q, want gpt-4o-mini", info.Model)
	}
	if !info.Capabilities.Tools || info.Capabilities.Embeddings || info.Capabilities.StructuredOutputs || info.Capabilities.PromptCaching {
		t.Fatalf("Capabilities = %+v, want tools=true and others=false", info.Capabilities)
	}
}

func TestInfo_OpenAI_EmbeddingModel(t *testing.T) {
	m, err := New(WithModel("text-embedding-3-small"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if !info.Capabilities.Embeddings {
		t.Fatalf("Capabilities = %+v, want embeddings=true", info.Capabilities)
	}
	if got := m.EmbedDimensions(); got != 1536 {
		t.Fatalf("EmbedDimensions() = %d, want 1536", got)
	}
}

func TestEmbed_OpenAI_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}
		body := string(bodyBytes)
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/embeddings" {
			t.Fatalf("Path = %s, want /embeddings", r.URL.Path)
		}
		if !strings.Contains(body, `"model":"text-embedding-3-small"`) {
			t.Fatalf("request body missing embedding model: %s", body)
		}
		if !strings.Contains(body, `"input":["hello","world"]`) {
			t.Fatalf("request body missing batch input order: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[
				{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]},
				{"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]}
			],
			"usage":{"prompt_tokens":7,"total_tokens":7}
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("text-embedding-3-small"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed(): %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("len(vectors) = %d, want 2", len(vectors))
	}
	if len(vectors[0]) != 3 || len(vectors[1]) != 3 {
		t.Fatalf("vector dims = [%d %d], want [3 3]", len(vectors[0]), len(vectors[1]))
	}
	if usage.Source != llm.UsageReported || usage.InputTokens != 7 || usage.TotalTokens != 7 {
		t.Fatalf("usage = %+v, want reported prompt=7 total=7", usage)
	}
}

func TestWithTools_OpenAI_ImmutableAndIndependent(t *testing.T) {
	var calcSeen atomic.Int32
	var searchSeen atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodyStr := string(body)

		switch {
		case strings.Contains(bodyStr, `"name":"calc"`):
			calcSeen.Add(1)
			if strings.Contains(bodyStr, `"name":"search"`) {
				t.Fatalf("calc-bound request leaked search tool: %s", bodyStr)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl_calc",
				"object":"chat.completion",
				"created":1710000000,
				"model":"gpt-4o-mini",
				"choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_calc","type":"function","function":{"name":"calc","arguments":"{\"expr\":\"2+2\"}"}}]},"finish_reason":"tool_calls","logprobs":null}],
				"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}
			}`))
		case strings.Contains(bodyStr, `"name":"search"`):
			searchSeen.Add(1)
			if strings.Contains(bodyStr, `"name":"calc"`) {
				t.Fatalf("search-bound request leaked calc tool: %s", bodyStr)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl_search",
				"object":"chat.completion",
				"created":1710000000,
				"model":"gpt-4o-mini",
				"choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_search","type":"function","function":{"name":"search","arguments":"{\"q\":\"sky\"}"}}]},"finish_reason":"tool_calls","logprobs":null}],
				"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}
			}`))
		default:
			t.Fatalf("request missing bound tool: %s", bodyStr)
		}
	}))
	defer server.Close()

	base, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	calcBound, err := base.WithTools([]llm.Tool{{Name: "calc", Description: "calculator", Parameters: []byte(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools(calc): %v", err)
	}
	searchBound, err := base.WithTools([]llm.Tool{{Name: "search", Description: "search", Parameters: []byte(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools(search): %v", err)
	}
	if calcBound == searchBound {
		t.Fatal("WithTools must return distinct values")
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		resp, err := calcBound.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "2+2"}}})
		if err == nil && (len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "calc") {
			err = fmt.Errorf("calc response tool calls = %+v, want calc", resp.ToolCalls)
		}
		errCh <- err
	}()
	go func() {
		defer wg.Done()
		resp, err := searchBound.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "sky"}}})
		if err == nil && (len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "search") {
			err = fmt.Errorf("search response tool calls = %+v, want search", resp.ToolCalls)
		}
		errCh <- err
	}()
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatal(err)
		}
	}
	if calcSeen.Load() != 1 || searchSeen.Load() != 1 {
		t.Fatalf("calcSeen=%d searchSeen=%d, want 1 each", calcSeen.Load(), searchSeen.Load())
	}
}

func TestStream_OpenAI_Happy(t *testing.T) {
	var sawIncludeUsage atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"include_usage":true`) {
			sawIncludeUsage.Store(true)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "say hello"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if resp.Text != "hello" {
		t.Fatalf("Text = %q, want hello", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonStop)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Fatalf("Usage.Source = %q, want %q", resp.Usage.Source, llm.UsageReported)
	}
	if resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Fatalf("Usage = %+v, want prompt=11 completion=7 total=18", resp.Usage)
	}
	if !sawIncludeUsage.Load() {
		t.Fatal(`request body missing "stream_options.include_usage=true"`)
	}
}

func TestStream_OpenAI_RetriesBeforeFirstByte(t *testing.T) {
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		if n == 1 {
			_, _ = fmt.Fprint(w, "data: {\"error\":{\"message\":\"upstream exploded\"}}\n\n")
			return
		}

		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if resp.Text != "ok" {
		t.Fatalf("Text = %q, want ok", resp.Text)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("attempts = %d, want 2", got)
	}
}

func TestStream_OpenAI_DoesNotRetryAfterFirstByte(t *testing.T) {
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"error\":{\"message\":\"stream interrupted\"}}\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	ev, err := sr.Next()
	if err != nil {
		t.Fatalf("Next#1: %v", err)
	}
	if ev.Kind != llm.EventTextDelta || ev.Text != "par" {
		t.Fatalf("Next#1 = %+v, want text delta par", ev)
	}

	_, err = sr.Next()
	if err == nil {
		t.Fatal("Next#2 error = nil, want non-nil")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("attempts = %d, want 1", got)
	}
}

func TestStream_OpenAI_ToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_tools\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_calc\",\"type\":\"function\",\"function\":{\"name\":\"calc\",\"arguments\":\"{\\\"expr\\\":\"}},{\"index\":1,\"id\":\"call_search\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_tools\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"2+2\\\"}\"}},{\"index\":1,\"function\":{\"arguments\":\"\\\"weather\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_tools\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":5,\"total_tokens\":14}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "use tools"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	wantKinds := []llm.StreamEventKind{
		llm.EventToolCallStart,
		llm.EventToolCallArgsDelta,
		llm.EventToolCallStart,
		llm.EventToolCallArgsDelta,
		llm.EventToolCallArgsDelta,
		llm.EventToolCallArgsDelta,
		llm.EventToolCallEnd,
		llm.EventToolCallEnd,
		llm.EventDone,
	}
	var got []llm.StreamEvent
	for range wantKinds {
		ev, err := sr.Next()
		if err != nil {
			t.Fatalf("Next(): %v", err)
		}
		got = append(got, ev)
	}
	for i, want := range wantKinds {
		if got[i].Kind != want {
			t.Fatalf("event[%d].Kind = %v, want %v", i, got[i].Kind, want)
		}
	}
	if got[0].ToolCall == nil || got[0].ToolCall.Index != 0 || got[0].ToolCall.ID != "call_calc" || got[0].ToolCall.Name != "calc" {
		t.Fatalf("event[0] = %+v, want calc start", got[0])
	}
	if got[2].ToolCall == nil || got[2].ToolCall.Index != 1 || got[2].ToolCall.ID != "call_search" || got[2].ToolCall.Name != "search" {
		t.Fatalf("event[2] = %+v, want search start", got[2])
	}
	if got[6].ToolCall == nil || got[6].ToolCall.Index != 0 {
		t.Fatalf("event[6] = %+v, want tool end index 0", got[6])
	}
	if got[7].ToolCall == nil || got[7].ToolCall.Index != 1 {
		t.Fatalf("event[7] = %+v, want tool end index 1", got[7])
	}
	if got[8].Usage == nil || got[8].FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("event[8] = %+v, want done with tool_calls finish", got[8])
	}
}

func TestGenerate_OpenAI_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("Path = %s, want /chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl_123",
			"object":"chat.completion",
			"created":1710000000,
			"model":"gpt-4o-mini",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello world"},"finish_reason":"stop","logprobs":null}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("gpt-4o-mini"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		SystemPrompt: "be concise",
		Messages: []llm.Message{
			{Role: "user", Content: "say hello"},
		},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if resp.Text != "hello world" {
		t.Fatalf("Text = %q, want hello world", resp.Text)
	}
	if resp.Provider != "openai" {
		t.Fatalf("Provider = %q, want openai", resp.Provider)
	}
	if resp.Model != "gpt-4o-mini" {
		t.Fatalf("Model = %q, want gpt-4o-mini", resp.Model)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonStop)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Fatalf("Usage.Source = %q, want %q", resp.Usage.Source, llm.UsageReported)
	}
}

func TestGenerate_OpenAI_ToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl_456",
			"object":"chat.completion",
			"created":1710000000,
			"model":"gpt-4o-mini",
			"choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[
				{"id":"call_calc","type":"function","function":{"name":"calc","arguments":"{\"expr\":\"2+2\"}"}},
				{"id":"call_search","type":"function","function":{"name":"search","arguments":"{\"q\":\"weather\"}"}}
			]},"finish_reason":"tool_calls","logprobs":null}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`))
	}))
	defer server.Close()

	base, err := New(
		WithModel("gpt-4o-mini"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{
		{Name: "calc", Description: "calculator", Parameters: []byte(`{"type":"object"}`)},
		{Name: "search", Description: "search", Parameters: []byte(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "use tools"}},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if resp.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonToolCalls)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("len(ToolCalls) = %d, want 2", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "call_calc" || resp.ToolCalls[0].Name != "calc" || string(resp.ToolCalls[0].Arguments) != `{"expr":"2+2"}` {
		t.Fatalf("ToolCalls[0] = %+v, want calc tool call", resp.ToolCalls[0])
	}
	if resp.ToolCalls[1].ID != "call_search" || resp.ToolCalls[1].Name != "search" || string(resp.ToolCalls[1].Arguments) != `{"q":"weather"}` {
		t.Fatalf("ToolCalls[1] = %+v, want search tool call", resp.ToolCalls[1])
	}
}

func TestGenerate_OpenAI_401AuthError(t *testing.T) {
	assertTypedError(t, 401, `{"error":{"message":"bad key","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}`, func(err error) bool {
		var target *llm.AuthError
		return errors.As(err, &target)
	})
}

func TestGenerate_OpenAI_403AuthError(t *testing.T) {
	assertTypedError(t, 403, `{"error":{"message":"forbidden","type":"invalid_request_error","param":null,"code":"forbidden"}}`, func(err error) bool {
		var target *llm.AuthError
		return errors.As(err, &target)
	})
}

func TestGenerate_OpenAI_429QuotaError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"error":{"message":"quota exceeded","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err == nil {
		t.Fatal("Generate() error = nil, want non-nil")
	}
	var target *llm.RateLimitError
	if !errors.As(err, &target) {
		t.Fatalf("errors.As(err, *llm.RateLimitError) = false, err=%T %v", err, err)
	}
	if target.Reason != "quota_exhausted" {
		t.Fatalf("Reason = %q, want quota_exhausted", target.Reason)
	}
	if target.RetryAfter != "30" {
		t.Fatalf("RetryAfter = %q, want 30", target.RetryAfter)
	}
}

func TestGenerate_OpenAI_429RateLimitError(t *testing.T) {
	assertTypedError(t, 429, `{"error":{"message":"too many requests","type":"rate_limit_error","param":null,"code":"rate_limit_exceeded"}}`, func(err error) bool {
		var target *llm.RateLimitError
		return errors.As(err, &target)
	})
}

func TestGenerate_OpenAI_500TransientError(t *testing.T) {
	assertTypedError(t, 500, `{"error":{"message":"server error","type":"server_error","param":null,"code":null}}`, func(err error) bool {
		var target *llm.TransientError
		return errors.As(err, &target)
	})
}

func TestGenerate_OpenAI_404InvalidRequestError(t *testing.T) {
	assertTypedError(t, 404, `{"error":{"message":"model not found","type":"invalid_request_error","param":null,"code":"model_not_found"}}`, func(err error) bool {
		var target *llm.InvalidRequestError
		return errors.As(err, &target)
	})
}

func assertTypedError(t *testing.T, status int, body string, match func(error) bool) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	m, err := New(WithModel("gpt-4o-mini"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err == nil {
		t.Fatal("Generate() error = nil, want non-nil")
	}
	if !match(err) {
		t.Fatalf("typed error assertion failed for status %d: %T %v", status, err, err)
	}
}
