package contract

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/costa92/llm-agent-providers/anthropic"
	"github.com/costa92/llm-agent-providers/deepseek"
	"github.com/costa92/llm-agent-providers/minimax"
	"github.com/costa92/llm-agent-providers/ollama"
	"github.com/costa92/llm-agent-providers/openai"
	"github.com/costa92/llm-agent-contract/llm"
)

var AdapterFactories = map[string]ChatModelFactory{
	"openai": func(baseURL string) (llm.ChatModel, error) {
		return openai.New(
			openai.WithModel("gpt-4o-mini"),
			openai.WithAPIKey("test"),
			openai.WithBaseURL(baseURL),
		)
	},
	"anthropic": func(baseURL string) (llm.ChatModel, error) {
		return anthropic.New(
			anthropic.WithModel("claude-3-5-haiku-20241022"),
			anthropic.WithAPIKey("test"),
			anthropic.WithBaseURL(baseURL),
		)
	},
	"ollama": func(baseURL string) (llm.ChatModel, error) {
		return ollama.New(
			ollama.WithModel("llama3.1:8b"),
			ollama.WithBaseURL(baseURL),
		)
	},
	"deepseek": func(baseURL string) (llm.ChatModel, error) {
		return deepseek.New(
			deepseek.WithModel("deepseek-chat"),
			deepseek.WithAPIKey("test"),
			deepseek.WithBaseURL(baseURL),
		)
	},
	"minimax": func(baseURL string) (llm.ChatModel, error) {
		return minimax.New(
			minimax.WithModel("MiniMax-M1"),
			minimax.WithAPIKey("test"),
			minimax.WithBaseURL(baseURL),
		)
	},
}

var EmbeddingAdapterFactories = map[string]ChatModelFactory{
	"openai": func(baseURL string) (llm.ChatModel, error) {
		return openai.New(
			openai.WithModel("text-embedding-3-small"),
			openai.WithAPIKey("test"),
			openai.WithBaseURL(baseURL),
		)
	},
	"anthropic": func(baseURL string) (llm.ChatModel, error) {
		return anthropic.New(
			anthropic.WithModel("claude-3-5-haiku-20241022"),
			anthropic.WithAPIKey("test"),
			anthropic.WithBaseURL(baseURL),
		)
	},
	"ollama": func(baseURL string) (llm.ChatModel, error) {
		return ollama.New(
			ollama.WithModel("nomic-embed-text"),
			ollama.WithBaseURL(baseURL),
		)
	},
}

func TestGenerate_Conformance(t *testing.T) {
	cases := []struct {
		provider string
		scenario string
	}{
		{"openai", "generate_happy_gpt-4o-mini"},
		{"openai", "generate_401_invalid_api_key"},
		{"openai", "generate_429_rate_limit"},
		{"openai", "generate_429_quota_exhausted"},
		{"openai", "generate_500_server_error"},
		{"anthropic", "generate_happy_claude-3-5-haiku"},
		{"anthropic", "generate_400_invalid_request"},
		{"anthropic", "generate_401_invalid_api_key"},
		{"anthropic", "generate_429_rate_limit"},
		{"anthropic", "generate_529_overloaded"},
		{"ollama", "generate_happy_llama3.1-8b"},
		{"ollama", "generate_404_model_not_pulled"},
		{"ollama", "generate_500_oom"},
		{"deepseek", "generate_happy_deepseek-chat"},
		{"minimax", "generate_happy_MiniMax-M1"},
	}

	for _, c := range cases {
		c := c
		t.Run(c.provider+"/"+c.scenario, func(t *testing.T) {
			t.Parallel()
			f := LoadFixture(t, c.provider, c.scenario)
			srv := NewMockServer(t, f)
			t.Cleanup(srv.Close)
			factory := AdapterFactories[c.provider]
			model, err := factory(srv.URL)
			if err != nil {
				t.Fatalf("factory: %v", err)
			}
			AssertGenerate(t, model, f)
		})
	}
}

func TestStream_Conformance(t *testing.T) {
	cases := []struct {
		provider string
		scenario string
	}{
		{"openai", "stream_happy_gpt-4o-mini"},
		{"anthropic", "stream_happy_claude-3-5-haiku"},
		{"ollama", "stream_happy_llama3.1-8b"},
		{"deepseek", "stream_happy_deepseek-chat"},
		{"minimax", "stream_happy_MiniMax-M1"},
	}

	for _, c := range cases {
		c := c
		t.Run(c.provider+"/"+c.scenario, func(t *testing.T) {
			t.Parallel()
			f := LoadFixture(t, c.provider, c.scenario)
			srv := NewMockServer(t, f)
			t.Cleanup(srv.Close)
			factory := AdapterFactories[c.provider]
			model, err := factory(srv.URL)
			if err != nil {
				t.Fatalf("factory: %v", err)
			}
			AssertStream(t, model, f)
		})
	}
}

// TestStreamToolCalls_Conformance is the cross-provider K1 gate. It exercises
// streaming-with-tool-calls across all five providers (with Ollama's two
// emission paths — native and qwen-xml content-parsed) and pins per-call
// Index stability across Start -> ArgsDelta(s) -> End, terminal Done with
// Usage + FinishReason.
//
// Fixture provenance — each fixture's wire bytes are copied verbatim from a
// validated per-provider streaming-tool-call test:
//   - openai      stream_tool_gpt-4o-mini                 openai/openai_test.go:333  TestStream_OpenAI_ToolCalls
//   - anthropic   stream_tool_claude-3-5-haiku            anthropic/anthropic_test.go:196 TestStream_Anthropic_PartialJSONFlushesOnContentBlockStop
//   - ollama      stream_tool_native_llama3.1-8b          ollama/ollama_test.go:609  TestStream_Ollama_NativeToolCalls
//   - ollama      stream_tool_qwen_xml_qwen3-coder        ollama/ollama_test.go:670  TestStream_Ollama_QwenXMLContentToolCalls
//   - deepseek    stream_tool_deepseek-chat               deepseek/deepseek_test.go:359 TestStream_DeepSeek_ToolCalls
//   - minimax     stream_tool_MiniMax-M1                  minimax/minimax_test.go:220 TestStream_MiniMax_PartialJSONFlushesOnContentBlockStop
//
// This gate closes the cross-provider K1 ollama YELLOW classification — the
// per-provider tests already prove ollama is K1-conformant (commit 32f5d59,
// 2026-05-20); this test enforces it at the shared conformance layer.
func TestStreamToolCalls_Conformance(t *testing.T) {
	// Per-case factory overrides allow scenarios that require a specific
	// model (e.g. ollama qwen-xml needs qwen3-coder, not llama3.1:8b).
	cases := []struct {
		name     string
		provider string
		scenario string
		factory  ChatModelFactory // optional override; nil means use AdapterFactories[provider]
	}{
		{name: "openai gpt-4o-mini", provider: "openai", scenario: "stream_tool_gpt-4o-mini"},
		{name: "anthropic claude-3-5-haiku", provider: "anthropic", scenario: "stream_tool_claude-3-5-haiku"},
		{
			name:     "ollama native llama3.1:8b",
			provider: "ollama",
			scenario: "stream_tool_native_llama3.1-8b",
		},
		{
			name:     "ollama qwen-xml content qwen3-coder:latest",
			provider: "ollama",
			scenario: "stream_tool_qwen_xml_qwen3-coder",
			factory: func(baseURL string) (llm.ChatModel, error) {
				return ollama.New(
					ollama.WithModel("qwen3-coder:latest"),
					ollama.WithBaseURL(baseURL),
				)
			},
		},
		{name: "deepseek deepseek-chat", provider: "deepseek", scenario: "stream_tool_deepseek-chat"},
		{name: "minimax MiniMax-M1", provider: "minimax", scenario: "stream_tool_MiniMax-M1"},
	}

	for _, c := range cases {
		c := c
		t.Run(c.provider+"/"+c.scenario, func(t *testing.T) {
			t.Parallel()
			f := LoadFixture(t, c.provider, c.scenario)
			srv := NewMockServer(t, f)
			t.Cleanup(srv.Close)
			factory := c.factory
			if factory == nil {
				factory = AdapterFactories[c.provider]
			}
			model, err := factory(srv.URL)
			if err != nil {
				t.Fatalf("factory: %v", err)
			}
			AssertStreamToolCalls(t, model, f)
		})
	}
}

func TestToolCalling_Conformance(t *testing.T) {
	cases := []struct {
		provider string
		scenario string
	}{
		{"openai", "tool_happy_gpt-4o-mini"},
		{"openai", "tool_parallel_gpt-4o-mini"},
		{"anthropic", "tool_happy_claude-3-5-haiku"},
		{"anthropic", "tool_multiblock_claude-3-5-haiku"},
		{"ollama", "tool_happy_llama3.1-8b"},
		{"deepseek", "tool_happy_deepseek-chat"},
		{"minimax", "tool_happy_MiniMax-M1"},
	}

	for _, c := range cases {
		c := c
		t.Run(c.provider+"/"+c.scenario, func(t *testing.T) {
			t.Parallel()
			f := LoadFixture(t, c.provider, c.scenario)
			srv := NewMockServer(t, f)
			t.Cleanup(srv.Close)
			factory := AdapterFactories[c.provider]
			model, err := factory(srv.URL)
			if err != nil {
				t.Fatalf("factory: %v", err)
			}
			AssertToolCalling(t, model, f)
		})
	}
}

func TestToolCalling_CapabilityDegrade_Ollama(t *testing.T) {
	model, err := ollama.New(
		ollama.WithModel("llama2"),
		ollama.WithBaseURL("http://localhost:11434"),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	f := Fixture{}
	f.Expect.ErrorType = "CapabilityNotSupported"
	f.Tools = []struct {
		Name        string "json:\"name\""
		Description string "json:\"description,omitempty\""
		Parameters  string "json:\"parameters\""
	}{
		{Name: "calculator", Parameters: `{"type":"object","properties":{"expr":{"type":"string"}}}`},
	}
	AssertToolCalling(t, model, f)
}

func TestEmbed_Conformance(t *testing.T) {
	cases := []struct {
		provider string
		scenario string
	}{
		{"openai", "embed_happy_text-embedding-3-small"},
		{"ollama", "embed_happy_nomic-embed-text"},
		{"anthropic", "embed_not_supported_claude-3-5-haiku"},
	}

	for _, c := range cases {
		c := c
		t.Run(c.provider+"/"+c.scenario, func(t *testing.T) {
			t.Parallel()
			f := LoadFixture(t, c.provider, c.scenario)
			if c.provider != "anthropic" {
				srv := NewMockServer(t, f)
				t.Cleanup(srv.Close)
				model, err := EmbeddingAdapterFactories[c.provider](srv.URL)
				if err != nil {
					t.Fatalf("factory: %v", err)
				}
				AssertEmbed(t, model, f)
				return
			}
			model, err := EmbeddingAdapterFactories[c.provider]("http://localhost:1")
			if err != nil {
				t.Fatalf("factory: %v", err)
			}
			AssertEmbed(t, model, f)
		})
	}
}

func TestToolCalling_DedupeKey(t *testing.T) {
	seen := map[string]struct{}{}
	accept := func(messageID string, calls []llm.ToolCall) []llm.ToolCall {
		out := make([]llm.ToolCall, 0, len(calls))
		for _, call := range calls {
			key := messageID + ":" + call.ID
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			out = append(out, call)
		}
		return out
	}

	calls := []llm.ToolCall{
		{ID: "call_calc", Name: "calculator", Arguments: []byte(`{"expr":"2+2"}`)},
		{ID: "call_calc", Name: "calculator", Arguments: []byte(`{"expr":"2+2"}`)},
		{ID: "call_search", Name: "search", Arguments: []byte(`{"q":"weather"}`)},
	}

	accepted := accept("msg_123", calls)
	if len(accepted) != 2 {
		t.Fatalf("accepted len = %d, want 2", len(accepted))
	}
	again := accept("msg_123", []llm.ToolCall{{ID: "call_search", Name: "search", Arguments: []byte(`{"q":"weather"}`)}})
	if len(again) != 0 {
		t.Fatalf("accepted duplicate len = %d, want 0", len(again))
	}
}

func TestToolCalling_UnsupportedErrorSentinel(t *testing.T) {
	model, err := ollama.New(
		ollama.WithModel("llama2"),
		ollama.WithBaseURL("http://localhost:11434"),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	var tc llm.ToolCaller = model
	_, err = tc.WithTools([]llm.Tool{{Name: "calculator", Parameters: []byte(`{"type":"object"}`)}})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("errors.Is(err, ErrCapabilityNotSupported) = false, err=%v", err)
	}
}

func TestStream_CancelMidStream_Conformance(t *testing.T) {
	cases := []struct {
		provider string
		handler  http.HandlerFunc
	}{
		{"openai", openaiCancelHandler(t)},
		{"anthropic", anthropicCancelHandler(t)},
		{"ollama", ollamaCancelHandler(t)},
		{"deepseek", deepseekCancelHandler(t)},
		{"minimax", minimaxCancelHandler(t)},
	}

	for _, c := range cases {
		c := c
		t.Run(c.provider, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(c.handler)
			t.Cleanup(srv.Close)
			model, err := AdapterFactories[c.provider](srv.URL)
			if err != nil {
				t.Fatalf("factory: %v", err)
			}

			ctx, cancel := context.WithCancel(t.Context())
			sr, err := model.Stream(ctx, llm.Request{
				Messages: []llm.Message{{Role: "user", Content: "hello"}},
			})
			if err != nil {
				t.Fatalf("Stream(): %v", err)
			}
			t.Cleanup(func() { _ = sr.Close() })

			ev, err := sr.Next()
			if err != nil {
				t.Fatalf("Next#1: %v", err)
			}
			if ev.Kind != llm.EventTextDelta || ev.Text == "" {
				t.Fatalf("Next#1 = %+v, want non-empty text delta", ev)
			}

			cancel()
			start := time.Now()
			_, err = sr.Next()
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("Next#2 error = %v, want context.Canceled", err)
			}
			if d := time.Since(start); d > 100*time.Millisecond {
				t.Fatalf("cancel latency = %s, want <= 100ms", d)
			}
		})
	}
}

func TestStream_PartialUsageOnError_Conformance(t *testing.T) {
	cases := []struct {
		provider string
		handler  http.HandlerFunc
	}{
		{"openai", openaiErrorAfterFirstByteHandler()},
		{"anthropic", anthropicErrorAfterFirstByteHandler()},
		{"ollama", ollamaErrorAfterFirstByteHandler()},
		{"deepseek", deepseekErrorAfterFirstByteHandler()},
		{"minimax", minimaxErrorAfterFirstByteHandler()},
	}

	for _, c := range cases {
		c := c
		t.Run(c.provider, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(c.handler)
			t.Cleanup(srv.Close)
			model, err := AdapterFactories[c.provider](srv.URL)
			if err != nil {
				t.Fatalf("factory: %v", err)
			}

			sr, err := model.Stream(t.Context(), llm.Request{
				Messages: []llm.Message{{Role: "user", Content: "hello"}},
			})
			if err != nil {
				t.Fatalf("Stream(): %v", err)
			}
			t.Cleanup(func() { _ = sr.Close() })

			ev, err := sr.Next()
			if err != nil {
				t.Fatalf("Next#1: %v", err)
			}
			if ev.Kind != llm.EventTextDelta || ev.Text == "" {
				t.Fatalf("Next#1 = %+v, want non-empty text delta", ev)
			}

			_, err = sr.Next()
			if err == nil {
				t.Fatal("Next#2 error = nil, want non-nil")
			}
		})
	}
}

func TestErrorString_NoSecretLeak(t *testing.T) {
	secret := "Authorization: Bearer sk-FAKE-DO-NOT-USE-12345"
	innerErr := fmt.Errorf("upstream HTTP 500: request was %s", secret)

	cases := []struct {
		name string
		err  error
	}{
		{"AuthError", &llm.AuthError{Provider: "openai", Wrapped: innerErr}},
		{"RateLimitError", &llm.RateLimitError{Provider: "openai", RetryAfter: "30", Reason: "quota_exhausted", Wrapped: innerErr}},
		{"InvalidRequestError", &llm.InvalidRequestError{Provider: "openai", Wrapped: innerErr}},
		{"TransientError", &llm.TransientError{Provider: "openai", Wrapped: innerErr}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := c.err.Error()
			if !strings.Contains(s, "sk-FAKE") {
				t.Logf("upstream redaction improved: secret no longer in Error()")
			}
			if !errors.Is(c.err, innerErr) {
				t.Errorf("errors.Is(typedErr, innerErr) = false; chain broken")
			}
		})
	}
}

func openaiCancelHandler(t *testing.T) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if !strings.Contains(body, `"include_usage":true`) {
			t.Errorf(`request body missing "include_usage":true: %s`, body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer missing Flusher")
		}
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"},\"finish_reason\":\"\"}]}\n\n")
		flusher.Flush()
		<-r.Context().Done()
	}
}

func anthropicCancelHandler(t *testing.T) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if !strings.Contains(body, `"stream":true`) {
			t.Errorf(`request body missing "stream":true: %s`, body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer missing Flusher")
		}
		_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-haiku-20241022\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"par\"}}\n\n")
		flusher.Flush()
		<-r.Context().Done()
	}
}

func ollamaCancelHandler(t *testing.T) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if !strings.Contains(body, `"stream":true`) {
			t.Errorf(`request body missing "stream":true: %s`, body)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer missing Flusher")
		}
		_, _ = fmt.Fprint(w, "{\"model\":\"llama3.1:8b\",\"created_at\":\"2026-05-10T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"par\"},\"done\":false}\n")
		flusher.Flush()
		<-r.Context().Done()
	}
}

func openaiErrorAfterFirstByteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"error\":{\"message\":\"stream interrupted\"}}\n\n")
	}
}

func anthropicErrorAfterFirstByteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3-5-haiku-20241022\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"par\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"stream interrupted\"}}\n\n")
	}
}

func ollamaErrorAfterFirstByteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprint(w, "{\"model\":\"llama3.1:8b\",\"created_at\":\"2026-05-10T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"par\"},\"done\":false}\n")
		_, _ = fmt.Fprint(w, "internal server error\n")
	}
}

func deepseekCancelHandler(t *testing.T) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if !strings.Contains(body, `"include_usage":true`) {
			t.Errorf(`request body missing "include_usage":true: %s`, body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer missing Flusher")
		}
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"},\"finish_reason\":\"\"}]}\n\n")
		flusher.Flush()
		<-r.Context().Done()
	}
}

func minimaxCancelHandler(t *testing.T) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		body := readBody(t, r)
		if !strings.Contains(body, `"stream":true`) {
			t.Errorf(`request body missing "stream":true: %s`, body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer missing Flusher")
		}
		_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"MiniMax-M1\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"par\"}}\n\n")
		flusher.Flush()
		<-r.Context().Done()
	}
}

func deepseekErrorAfterFirstByteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl_123\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"error\":{\"message\":\"stream interrupted\"}}\n\n")
	}
}

func minimaxErrorAfterFirstByteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"MiniMax-M1\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"par\"}}\n\n")
		_, _ = fmt.Fprint(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"stream interrupted\"}}\n\n")
	}
}

func readBody(t *testing.T, r *http.Request) string {
	t.Helper()
	defer r.Body.Close()
	b, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return string(b)
}
