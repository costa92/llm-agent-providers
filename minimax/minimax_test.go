package minimax

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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

func TestInfo_MiniMax(t *testing.T) {
	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if info.Provider != "minimax" {
		t.Fatalf("Provider = %q, want minimax", info.Provider)
	}
	if info.Model != "MiniMax-M1" {
		t.Fatalf("Model = %q, want MiniMax-M1", info.Model)
	}
	if !info.Capabilities.Tools || info.Capabilities.Embeddings || info.Capabilities.StructuredOutputs || info.Capabilities.PromptCaching {
		t.Fatalf("Capabilities = %+v, want tools=true and others=false", info.Capabilities)
	}
}

func TestMiniMax_DocumentedEmbeddingGap(t *testing.T) {
	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := m.Info()
	if info.Capabilities.Embeddings {
		t.Fatalf("Capabilities = %+v, want embeddings=false", info.Capabilities)
	}
	if _, ok := any(m).(llm.Embedder); ok {
		t.Fatal("MiniMax adapter must not implement llm.Embedder")
	}
}

func TestRegionPreset_MiniMax(t *testing.T) {
	if got := baseURLForRegion(RegionGlobal); got == "" {
		t.Fatal("baseURLForRegion(RegionGlobal) = empty")
	}
	if got := baseURLForRegion(RegionCN); got == "" {
		t.Fatal("baseURLForRegion(RegionCN) = empty")
	}
}

func TestBaseURL_OverridesRegion_MiniMax(t *testing.T) {
	m, err := New(
		WithModel("MiniMax-M1"),
		WithAPIKey("test-key"),
		WithRegion(RegionCN),
		WithBaseURL("https://example.invalid"),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := m.Info().Provider; got != "minimax" {
		t.Fatalf("Provider = %q, want minimax", got)
	}
}

func TestWithTools_MiniMax_ImmutableAndRequestShape(t *testing.T) {
	var weatherSeen atomic.Int32
	var calcSeen atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		if _, ok := payload["tool_choice"]; !ok {
			t.Fatalf("payload missing tool_choice: %s", string(body))
		}
		tools, ok := payload["tools"].([]any)
		if !ok || len(tools) != 1 {
			t.Fatalf("tools = %#v, want single bound tool", payload["tools"])
		}
		tool, ok := tools[0].(map[string]any)
		if !ok {
			t.Fatalf("tool wrong type: %T", tools[0])
		}
		name, _ := tool["name"].(string)
		switch name {
		case "weather":
			weatherSeen.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"msg_weather",
				"type":"message",
				"role":"assistant",
				"model":"MiniMax-M1",
				"content":[{"type":"tool_use","id":"toolu_weather","name":"weather","input":{"city":"Paris"},"caller":{"type":"direct"}}],
				"stop_reason":"tool_use",
				"stop_sequence":null,
				"usage":{"input_tokens":11,"output_tokens":7}
			}`))
		case "calc":
			calcSeen.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"msg_calc",
				"type":"message",
				"role":"assistant",
				"model":"MiniMax-M1",
				"content":[{"type":"tool_use","id":"toolu_calc","name":"calc","input":{"expr":"2+2"},"caller":{"type":"direct"}}],
				"stop_reason":"tool_use",
				"stop_sequence":null,
				"usage":{"input_tokens":11,"output_tokens":7}
			}`))
		default:
			t.Fatalf("unexpected tool binding: %q in %s", name, string(body))
		}
	}))
	defer server.Close()

	base, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	weatherBound, err := base.WithTools([]llm.Tool{{Name: "weather", Description: "weather tool", Parameters: []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}})
	if err != nil {
		t.Fatalf("WithTools(weather): %v", err)
	}
	calcBound, err := base.WithTools([]llm.Tool{{Name: "calc", Description: "calc tool", Parameters: []byte(`{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}`)}})
	if err != nil {
		t.Fatalf("WithTools(calc): %v", err)
	}
	if weatherBound == calcBound {
		t.Fatal("WithTools must return distinct values")
	}

	respA, err := weatherBound.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "weather"}}})
	if err != nil {
		t.Fatalf("weather Generate(): %v", err)
	}
	respB, err := calcBound.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "calc"}}})
	if err != nil {
		t.Fatalf("calc Generate(): %v", err)
	}
	if len(respA.ToolCalls) != 1 || respA.ToolCalls[0].Name != "weather" {
		t.Fatalf("weather ToolCalls = %+v, want weather", respA.ToolCalls)
	}
	if len(respB.ToolCalls) != 1 || respB.ToolCalls[0].Name != "calc" {
		t.Fatalf("calc ToolCalls = %+v, want calc", respB.ToolCalls)
	}
	if weatherSeen.Load() != 1 || calcSeen.Load() != 1 {
		t.Fatalf("weatherSeen=%d calcSeen=%d, want 1 each", weatherSeen.Load(), calcSeen.Load())
	}
}

func TestStream_MiniMax_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"MiniMax-M1\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":11,\"output_tokens\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello \"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"minimax\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
		_, _ = fmt.Fprint(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":\"\"},\"usage\":{\"input_tokens\":11,\"output_tokens\":7,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"server_tool_use\":{\"web_search_requests\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if resp.Text != "hello minimax" {
		t.Fatalf("Text = %q, want hello minimax", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonStop)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Fatalf("Usage.Source = %q, want %q", resp.Usage.Source, llm.UsageReported)
	}
	if resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 7 {
		t.Fatalf("Usage = %+v, want input=11 output=7", resp.Usage)
	}
}

func TestStream_MiniMax_PartialJSONFlushesOnContentBlockStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"MiniMax-M1\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":9,\"output_tokens\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"checking \"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"weather\",\"input\":{},\"caller\":{\"type\":\"direct\"}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\" \\\"Paris\\\"}\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":\"\"},\"usage\":{\"input_tokens\":9,\"output_tokens\":5,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"server_tool_use\":{\"web_search_requests\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n")
		_, _ = fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	sr, err := m.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	ev1, err := sr.Next()
	if err != nil {
		t.Fatalf("Next#1: %v", err)
	}
	if ev1.Kind != llm.EventTextDelta || ev1.Text != "checking " {
		t.Fatalf("Next#1 = %+v, want text delta", ev1)
	}

	ev2, err := sr.Next()
	if err != nil {
		t.Fatalf("Next#2: %v", err)
	}
	if ev2.Kind != llm.EventToolCallStart || ev2.ToolCall == nil || ev2.ToolCall.Index != 1 || ev2.ToolCall.ID != "toolu_1" || ev2.ToolCall.Name != "weather" {
		t.Fatalf("Next#2 = %+v, want tool start for index 1", ev2)
	}

	ev3, err := sr.Next()
	if err != nil {
		t.Fatalf("Next#3: %v", err)
	}
	if ev3.Kind != llm.EventToolCallArgsDelta || ev3.ToolCall == nil || ev3.ToolCall.ArgsDelta != "{\"city\":" {
		t.Fatalf("Next#3 = %+v, want first args delta", ev3)
	}

	ev4, err := sr.Next()
	if err != nil {
		t.Fatalf("Next#4: %v", err)
	}
	if ev4.Kind != llm.EventToolCallArgsDelta || ev4.ToolCall == nil || ev4.ToolCall.ArgsDelta != " \"Paris\"}" {
		t.Fatalf("Next#4 = %+v, want second args delta", ev4)
	}

	ev5, err := sr.Next()
	if err != nil {
		t.Fatalf("Next#5: %v", err)
	}
	if ev5.Kind != llm.EventToolCallEnd || ev5.ToolCall == nil || ev5.ToolCall.Index != 1 {
		t.Fatalf("Next#5 = %+v, want tool end after content_block_stop", ev5)
	}

	ev6, err := sr.Next()
	if err != nil {
		t.Fatalf("Next#6: %v", err)
	}
	if ev6.Kind != llm.EventDone || ev6.Usage == nil || ev6.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("Next#6 = %+v, want done/toolcalls", ev6)
	}
}

func TestStream_MiniMax_DoesNotRetryAfterFirstByte(t *testing.T) {
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"MiniMax-M1\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"par\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"stream interrupted\"}}\n\n")
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
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

func TestGenerate_MiniMax_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("Path = %s, want /v1/messages", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_123",
			"type":"message",
			"role":"assistant",
			"model":"MiniMax-M1",
			"content":[{"type":"text","text":"hello from minimax"}],
			"stop_reason":"end_turn",
			"stop_sequence":null,
			"usage":{"input_tokens":11,"output_tokens":7}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("MiniMax-M1"),
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
	if resp.Text != "hello from minimax" {
		t.Fatalf("Text = %q", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonStop)
	}
	if resp.Provider != "minimax" {
		t.Fatalf("Provider = %q, want minimax", resp.Provider)
	}
	if resp.Model != "MiniMax-M1" {
		t.Fatalf("Model = %q", resp.Model)
	}
}

func TestGenerate_MiniMax_MultiBlockToolUse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_tools",
			"type":"message",
			"role":"assistant",
			"model":"MiniMax-M1",
			"content":[
				{"type":"text","text":"checking tools"},
				{"type":"tool_use","id":"toolu_1","name":"weather","input":{"city":"Paris"},"caller":{"type":"direct"}},
				{"type":"tool_use","id":"toolu_2","name":"calculator","input":{"expr":"2+2"},"caller":{"type":"direct"}}
			],
			"stop_reason":"tool_use",
			"stop_sequence":null,
			"usage":{"input_tokens":11,"output_tokens":7}
		}`))
	}))
	defer server.Close()

	base, err := New(
		WithModel("MiniMax-M1"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{
		{Name: "weather", Description: "weather tool", Parameters: []byte(`{"type":"object"}`)},
		{Name: "calculator", Description: "calculator tool", Parameters: []byte(`{"type":"object"}`)},
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
	if resp.Text != "checking tools" {
		t.Fatalf("Text = %q, want checking tools", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, llm.FinishReasonToolCalls)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("len(ToolCalls) = %d, want 2", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "toolu_1" || resp.ToolCalls[0].Name != "weather" || string(resp.ToolCalls[0].Arguments) != `{"city":"Paris"}` {
		t.Fatalf("ToolCalls[0] = %+v, want weather tool call", resp.ToolCalls[0])
	}
	if resp.ToolCalls[1].ID != "toolu_2" || resp.ToolCalls[1].Name != "calculator" || string(resp.ToolCalls[1].Arguments) != `{"expr":"2+2"}` {
		t.Fatalf("ToolCalls[1] = %+v, want calculator tool call", resp.ToolCalls[1])
	}
}

func TestGenerate_MiniMax_SystemTopLevel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}

		if _, ok := payload["system"]; !ok {
			t.Fatalf("payload missing top-level system: %s", string(body))
		}

		msgs, ok := payload["messages"].([]any)
		if !ok {
			t.Fatalf("messages missing or wrong type: %T", payload["messages"])
		}
		for _, raw := range msgs {
			msg, ok := raw.(map[string]any)
			if !ok {
				t.Fatalf("message wrong type: %T", raw)
			}
			if msg["role"] == "system" {
				t.Fatalf("system role leaked into messages: %s", string(body))
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_123",
			"type":"message",
			"role":"assistant",
			"model":"MiniMax-M1",
			"content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"stop_sequence":null,
			"usage":{"input_tokens":11,"output_tokens":1}
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = m.Generate(context.Background(), llm.Request{
		SystemPrompt: "system A",
		Messages: []llm.Message{
			{Role: "system", Content: "system B"},
			{Role: "user", Content: "hello"},
		},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
}

func TestGenerate_MiniMax_401AuthError(t *testing.T) {
	assertMiniMaxTypedError(t, 401, `{"type":"error","error":{"type":"authentication_error","message":"bad key"}}`, func(err error) bool {
		var target *llm.AuthError
		return errors.As(err, &target)
	})
}

func TestGenerate_MiniMax_400InvalidRequestError(t *testing.T) {
	assertMiniMaxTypedError(t, 400, `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`, func(err error) bool {
		var target *llm.InvalidRequestError
		return errors.As(err, &target)
	})
}

func TestGenerate_MiniMax_429RateLimitError(t *testing.T) {
	assertMiniMaxTypedError(t, 429, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`, func(err error) bool {
		var target *llm.RateLimitError
		return errors.As(err, &target)
	})
}

func TestGenerate_MiniMax_529OverloadedIsRateLimit(t *testing.T) {
	assertMiniMaxTypedError(t, 529, `{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}`, func(err error) bool {
		var target *llm.RateLimitError
		return errors.As(err, &target)
	})
}

func TestGenerate_MiniMax_500TransientError(t *testing.T) {
	assertMiniMaxTypedError(t, 500, `{"type":"error","error":{"type":"api_error","message":"server error"}}`, func(err error) bool {
		var target *llm.TransientError
		return errors.As(err, &target)
	})
}

func TestGenerate_MiniMax_429SetsRetryAfterSeconds(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
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
	if target.RetryAfter != "30" {
		t.Fatalf("RetryAfter = %q, want 30", target.RetryAfter)
	}
}

func TestGenerate_MiniMax_429SetsRetryAfterHTTPDate(t *testing.T) {
	const when = "Wed, 21 Oct 2026 07:28:00 GMT"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", when)
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
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
	if target.RetryAfter != when {
		t.Fatalf("RetryAfter = %q, want %q (raw RFC 7231 string per contract)", target.RetryAfter, when)
	}
}

func TestGenerate_MiniMax_529SetsRetryAfter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "120")
		w.WriteHeader(529)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
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
	if target.RetryAfter != "120" {
		t.Fatalf("RetryAfter = %q, want 120", target.RetryAfter)
	}
}

func TestGenerate_MiniMax_429EmptyRetryAfterWhenHeaderMissing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
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
	if target.RetryAfter != "" {
		t.Fatalf("RetryAfter = %q, want empty string when header absent", target.RetryAfter)
	}
}

func assertMiniMaxTypedError(t *testing.T, status int, body string, match func(error) bool) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	defer server.Close()

	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("test-key"), WithBaseURL(server.URL))
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
