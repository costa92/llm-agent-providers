package volcengine

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestGenerate_Volcengine_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("Method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("Path = %s, want /chat/completions", r.URL.Path)
		}
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"model":"doubao-1-5-pro-32k-250115"`) {
			t.Fatalf("request body missing model: %s", body)
		}
		if !strings.Contains(body, `"role":"system"`) || !strings.Contains(body, `be concise`) {
			t.Fatalf("request body missing system message: %s", body)
		}
		if !strings.Contains(body, `say hello`) {
			t.Fatalf("request body missing user message: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"cmpl_123",
			"object":"chat.completion",
			"created":1710000000,
			"model":"doubao-1-5-pro-32k-250115",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello world"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("doubao-1-5-pro-32k-250115"),
		WithAPIKey("test-key"),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	resp, err := m.Generate(context.Background(), llm.Request{
		SystemPrompt: "be concise",
		Messages:     []llm.Message{{Role: "user", Content: "say hello"}},
	})
	if err != nil {
		t.Fatalf("Generate(): %v", err)
	}
	if resp.Text != "hello world" {
		t.Fatalf("Text = %q, want hello world", resp.Text)
	}
	if resp.Provider != "volcengine" {
		t.Fatalf("Provider = %q, want volcengine", resp.Provider)
	}
	if resp.Model != "doubao-1-5-pro-32k-250115" {
		t.Fatalf("Model = %q, want doubao-1-5-pro-32k-250115", resp.Model)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.Source != llm.UsageReported || resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Fatalf("Usage = %+v, want reported 11/7/18", resp.Usage)
	}
}

func TestGenerate_Volcengine_ToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"tools":[`) || !strings.Contains(body, `"name":"calc"`) {
			t.Fatalf("request body missing tool declaration: %s", body)
		}
		if !strings.Contains(body, `"type":"function"`) {
			t.Fatalf("request body missing function tool type: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"cmpl_456",
			"object":"chat.completion",
			"created":1710000000,
			"model":"doubao-1-5-pro-32k-250115",
			"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_calc","type":"function","function":{"name":"calc","arguments":"{\"expr\":\"2+2\"}"}},
				{"id":"call_search","type":"function","function":{"name":"search","arguments":"{\"q\":\"weather\"}"}}
			]},"finish_reason":"tool_calls"}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}
		}`))
	}))
	defer server.Close()

	base, err := New(
		WithModel("doubao-1-5-pro-32k-250115"),
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
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("len(ToolCalls) = %d, want 2", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "call_calc" || resp.ToolCalls[0].Name != "calc" {
		t.Fatalf("ToolCalls[0] = %+v, want calc", resp.ToolCalls[0])
	}
	if string(resp.ToolCalls[0].Arguments) != `{"expr":"2+2"}` {
		t.Fatalf("ToolCalls[0].Arguments = %s, want {\"expr\":\"2+2\"}", resp.ToolCalls[0].Arguments)
	}
	if resp.ToolCalls[1].ID != "call_search" || resp.ToolCalls[1].Name != "search" {
		t.Fatalf("ToolCalls[1] = %+v, want search", resp.ToolCalls[1])
	}
	if resp.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("FinishReason = %q, want tool_calls", resp.FinishReason)
	}
}

func TestWithTools_Volcengine_Immutable(t *testing.T) {
	base, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, err = base.WithTools([]llm.Tool{{Name: "calc", Parameters: []byte(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}
	if len(base.tools) != 0 {
		t.Fatalf("base.tools mutated, len = %d, want 0", len(base.tools))
	}
}

func TestStream_Volcengine_Text(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if fl != nil {
			fl.Flush()
		}
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	sr, err := m.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if resp.Text != "hello" {
		t.Fatalf("Text = %q, want hello", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.Source != llm.UsageReported || resp.Usage.TotalTokens != 5 {
		t.Fatalf("Usage = %+v, want reported total=5", resp.Usage)
	}
}

func TestStream_Volcengine_FragmentedToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c2\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_calc\",\"type\":\"function\",\"function\":{\"name\":\"calc\",\"arguments\":\"{\\\"expr\\\":\"}},{\"index\":1,\"id\":\"call_search\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\"}}]},\"finish_reason\":\"\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c2\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"2+2\\\"}\"}},{\"index\":1,\"function\":{\"arguments\":\"\\\"weather\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"c2\",\"model\":\"doubao-1-5-pro-32k-250115\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":5,\"total_tokens\":14}}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	base, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	m, err := base.WithTools([]llm.Tool{
		{Name: "calc", Parameters: []byte(`{"type":"object"}`)},
		{Name: "search", Parameters: []byte(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatalf("WithTools(): %v", err)
	}
	sr, err := m.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "use tools"}}})
	if err != nil {
		t.Fatalf("Stream(): %v", err)
	}
	defer sr.Close()

	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream(): %v", err)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("len(ToolCalls) = %d, want 2", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != "call_calc" || resp.ToolCalls[0].Name != "calc" || string(resp.ToolCalls[0].Arguments) != `{"expr":"2+2"}` {
		t.Fatalf("ToolCalls[0] = %+v args=%s, want calc {\"expr\":\"2+2\"}", resp.ToolCalls[0], resp.ToolCalls[0].Arguments)
	}
	if resp.ToolCalls[1].ID != "call_search" || resp.ToolCalls[1].Name != "search" || string(resp.ToolCalls[1].Arguments) != `{"q":"weather"}` {
		t.Fatalf("ToolCalls[1] = %+v args=%s, want search {\"q\":\"weather\"}", resp.ToolCalls[1], resp.ToolCalls[1].Arguments)
	}
	if resp.FinishReason != llm.FinishReasonToolCalls {
		t.Fatalf("FinishReason = %q, want tool_calls", resp.FinishReason)
	}
}

func TestWrapErr_Volcengine_StatusMapping(t *testing.T) {
	cases := []struct {
		status int
		match  func(error) bool
		name   string
	}{
		{401, func(err error) bool { var e *llm.AuthError; return asAuth(err, &e) }, "auth-401"},
		{403, func(err error) bool { var e *llm.AuthError; return asAuth(err, &e) }, "auth-403"},
		{429, func(err error) bool { var e *llm.RateLimitError; return asRate(err, &e) }, "rate-429"},
		{500, func(err error) bool { var e *llm.TransientError; return asTransient(err, &e) }, "transient-500"},
		{503, func(err error) bool { var e *llm.TransientError; return asTransient(err, &e) }, "transient-503"},
		{404, func(err error) bool { var e *llm.InvalidRequestError; return asInvalid(err, &e) }, "invalid-404"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(c.status)
				_, _ = w.Write([]byte(`{"error":{"message":"boom","type":"error","code":"x"}}`))
			}))
			defer server.Close()

			m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"), WithBaseURL(server.URL))
			if err != nil {
				t.Fatalf("New(): %v", err)
			}
			_, gerr := m.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
			if gerr == nil {
				t.Fatalf("Generate() error = nil, want %s", c.name)
			}
			if !c.match(gerr) {
				t.Fatalf("Generate() error = %v (%T), want %s", gerr, gerr, c.name)
			}
		})
	}
}

func TestWrapErr_Volcengine_ContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, gerr := m.Generate(ctx, llm.Request{Messages: []llm.Message{{Role: "user", Content: "hi"}}})
	if gerr == nil {
		t.Fatal("Generate() error = nil, want context.Canceled")
	}
	if !isContextCanceled(gerr) {
		t.Fatalf("Generate() error = %v, want context.Canceled passthrough", gerr)
	}
}

func asAuth(err error, e **llm.AuthError) bool              { return errorsAs(err, e) }
func asRate(err error, e **llm.RateLimitError) bool         { return errorsAs(err, e) }
func asTransient(err error, e **llm.TransientError) bool    { return errorsAs(err, e) }
func asInvalid(err error, e **llm.InvalidRequestError) bool { return errorsAs(err, e) }
func errorsAs(err error, target any) bool                   { return errors.As(err, target) }
func isContextCanceled(err error) bool                      { return errors.Is(err, context.Canceled) }

func TestGenerateImage_Volcengine_URL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/images/generations" {
			t.Fatalf("Path = %s, want /images/generations", r.URL.Path)
		}
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"model":"doubao-seedream-4-5-251128"`) {
			t.Fatalf("body missing model: %s", body)
		}
		if !strings.Contains(body, `"prompt":"a red panda"`) {
			t.Fatalf("body missing prompt: %s", body)
		}
		if !strings.Contains(body, `"size":"1024x1024"`) {
			t.Fatalf("body missing size: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model":"doubao-seedream-4-5-251128",
			"created":1710000000,
			"data":[{"url":"https://example.com/img.png","size":"1024x1024"}],
			"usage":{"generated_images":1,"output_tokens":0,"total_tokens":0}
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-seedream-4-5-251128"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := m.GenerateImage(context.Background(), llm.ImageRequest{
		Prompt: "a red panda",
		Size:   "1024x1024",
	})
	if err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if resp.Provider != "volcengine" || resp.Model != "doubao-seedream-4-5-251128" {
		t.Fatalf("resp meta = %+v, want volcengine/doubao-seedream-4-5-251128", resp)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("len(Images) = %d, want 1", len(resp.Images))
	}
	if resp.Images[0].URL != "https://example.com/img.png" {
		t.Fatalf("Images[0].URL = %q, want https://example.com/img.png", resp.Images[0].URL)
	}
	if len(resp.Images[0].Bytes) != 0 {
		t.Fatalf("Images[0].Bytes = %d bytes, want 0 (URL delivery)", len(resp.Images[0].Bytes))
	}
}

func TestGenerateImage_Volcengine_Base64(t *testing.T) {
	// base64 of the 3 bytes {0x01,0x02,0x03} is "AQID"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(bodyBytes), `"response_format":"b64_json"`) {
			t.Fatalf("body missing b64_json response_format: %s", string(bodyBytes))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model":"doubao-seedream-4-5-251128",
			"created":1710000000,
			"data":[{"b64_json":"AQID","size":"1024x1024"}]
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-seedream-4-5-251128"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	resp, err := m.GenerateImage(context.Background(), llm.ImageRequest{
		Prompt: "a red panda",
		Format: "b64_json",
	})
	if err != nil {
		t.Fatalf("GenerateImage(): %v", err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("len(Images) = %d, want 1", len(resp.Images))
	}
	if resp.Images[0].URL != "" {
		t.Fatalf("Images[0].URL = %q, want empty (bytes delivery)", resp.Images[0].URL)
	}
	if string(resp.Images[0].Bytes) != "\x01\x02\x03" {
		t.Fatalf("Images[0].Bytes = %v, want [1 2 3]", resp.Images[0].Bytes)
	}
}

func TestGenerateImage_Volcengine_CapabilityGate(t *testing.T) {
	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, gerr := m.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if gerr == nil {
		t.Fatal("GenerateImage() error = nil, want ErrCapabilityNotSupported")
	}
	if !errors.Is(gerr, llm.ErrCapabilityNotSupported) {
		t.Fatalf("GenerateImage() error = %v, want ErrCapabilityNotSupported", gerr)
	}
}

func TestEmbed_Volcengine_Happy(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Fatalf("Path = %s, want /embeddings", r.URL.Path)
		}
		bodyBytes, _ := io.ReadAll(r.Body)
		body := string(bodyBytes)
		if !strings.Contains(body, `"model":"doubao-embedding-text-240715"`) {
			t.Fatalf("body missing model: %s", body)
		}
		if !strings.Contains(body, `"input":["hello","world"]`) {
			t.Fatalf("body missing ordered input: %s", body)
		}
		if !strings.Contains(body, `"dimensions":2560`) {
			t.Fatalf("body missing dimensions: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"emb_1",
			"object":"list",
			"model":"doubao-embedding-text-240715",
			"data":[
				{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]},
				{"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]}
			],
			"usage":{"prompt_tokens":4,"total_tokens":4}
		}`))
	}))
	defer server.Close()

	m, err := New(WithModel("doubao-embedding-text-240715"), WithAPIKey("k"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed(): %v", err)
	}
	if len(vectors) != 2 || len(vectors[0]) != 3 || len(vectors[1]) != 3 {
		t.Fatalf("vectors shape = %d x [%d %d], want 2 x [3 3]", len(vectors), len(vectors[0]), len(vectors[1]))
	}
	if vectors[1][0] != 0.4 {
		t.Fatalf("vectors[1][0] = %v, want 0.4 (order preserved)", vectors[1][0])
	}
	if usage.Source != llm.UsageReported || usage.InputTokens != 4 || usage.TotalTokens != 4 {
		t.Fatalf("usage = %+v, want reported input=4 total=4", usage)
	}
	if got := m.EmbedDimensions(); got != 2560 {
		t.Fatalf("EmbedDimensions() = %d, want 2560", got)
	}
}

func TestEmbed_Volcengine_Empty(t *testing.T) {
	m, err := New(WithModel("doubao-embedding-text-240715"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	vectors, usage, err := m.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(nil): %v", err)
	}
	if len(vectors) != 0 {
		t.Fatalf("len(vectors) = %d, want 0", len(vectors))
	}
	if usage.Source != llm.UsageReported {
		t.Fatalf("usage.Source = %q, want reported", usage.Source)
	}
}

func TestEmbed_Volcengine_CapabilityGate(t *testing.T) {
	m, err := New(WithModel("doubao-1-5-pro-32k-250115"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, _, gerr := m.Embed(context.Background(), []string{"x"})
	if gerr == nil {
		t.Fatal("Embed() error = nil, want ErrCapabilityNotSupported")
	}
	if !errors.Is(gerr, llm.ErrCapabilityNotSupported) {
		t.Fatalf("Embed() error = %v, want ErrCapabilityNotSupported", gerr)
	}
	if got := m.EmbedDimensions(); got != 0 {
		t.Fatalf("EmbedDimensions() = %d, want 0 for non-embed model", got)
	}
}
