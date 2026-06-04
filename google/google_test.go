package google

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	// go.opencensus.io (a transitive dependency of the genai SDK) starts a
	// global stats-view worker goroutine in its package init that runs for
	// the lifetime of the process and cannot be stopped. It is not leaked by
	// the provider, so ignore it.
	goleak.VerifyTestMain(m,
		goleak.IgnoreTopFunction("go.opencensus.io/stats/view.(*worker).start"),
	)
}

// newTestServer builds an httptest server and a Google bound to model on it.
func newTestServer(t *testing.T, model string, handler http.HandlerFunc) *Google {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	g, err := New(WithModel(model), WithAPIKey("test-key"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return g
}

func TestGenerate_Happy(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":generateContent") {
			t.Errorf("path = %s, want suffix :generateContent", r.URL.Path)
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("x-goog-api-key = %q, want test-key", got)
		}
		body, _ := io.ReadAll(r.Body)
		s := string(body)
		if !strings.Contains(s, `"systemInstruction"`) {
			t.Errorf("body missing systemInstruction: %s", s)
		}
		if !strings.Contains(s, `"role":"user"`) {
			t.Errorf("body missing user role: %s", s)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[{"text":"Hello there"}]},"finishReason":"STOP","index":0}],
			"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6},
			"modelVersion":"gemini-2.5-flash"
		}`))
	})

	resp, err := g.Generate(context.Background(), llm.Request{
		SystemPrompt: "Be brief.",
		Messages:     []llm.Message{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Text != "Hello there" {
		t.Errorf("Text = %q, want Hello there", resp.Text)
	}
	if resp.FinishReason != llm.FinishReasonStop {
		t.Errorf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Provider != "google" {
		t.Errorf("Provider = %q, want google", resp.Provider)
	}
	if resp.Usage.InputTokens != 4 || resp.Usage.OutputTokens != 2 || resp.Usage.TotalTokens != 6 {
		t.Errorf("Usage = %+v, want 4/2/6", resp.Usage)
	}
	if resp.Usage.Source != llm.UsageReported {
		t.Errorf("Usage.Source = %q, want reported", resp.Usage.Source)
	}
}

func TestGenerate_ToolCall(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s := string(body)
		// tool schema forwarded via parametersJsonSchema
		if !strings.Contains(s, `"functionDeclarations"`) {
			t.Errorf("body missing functionDeclarations: %s", s)
		}
		if !strings.Contains(s, `"parametersJsonSchema"`) {
			t.Errorf("body missing parametersJsonSchema: %s", s)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[
				{"functionCall":{"id":"call_1","name":"get_weather","args":{"city":"Paris"}}}
			]},"finishReason":"STOP","index":0}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}
		}`))
	})

	tc, err := g.WithTools([]llm.Tool{{
		Name:        "get_weather",
		Description: "Get weather for a city",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}})
	if err != nil {
		t.Fatalf("WithTools: %v", err)
	}
	resp, err := tc.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "Weather in Paris?"}},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(resp.ToolCalls))
	}
	call := resp.ToolCalls[0]
	if call.ID != "call_1" || call.Name != "get_weather" {
		t.Errorf("ToolCall id/name = %s/%s, want call_1/get_weather", call.ID, call.Name)
	}
	var args map[string]any
	if err := json.Unmarshal(call.Arguments, &args); err != nil {
		t.Fatalf("Arguments not valid JSON: %v (%s)", err, call.Arguments)
	}
	if args["city"] != "Paris" {
		t.Errorf("args.city = %v, want Paris", args["city"])
	}
}

func TestNew_RequiresModel(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("New() error = nil, want non-nil")
	}
	if !strings.Contains(err.Error(), "WithModel is required") {
		t.Fatalf("New() error = %q, want WithModel is required", err)
	}
}

func TestInfo_ChatModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	info := g.Info()
	if info.Provider != "google" {
		t.Fatalf("Provider = %q, want google", info.Provider)
	}
	if info.Model != "gemini-2.5-flash" {
		t.Fatalf("Model = %q, want gemini-2.5-flash", info.Model)
	}
	if !info.Capabilities.Tools {
		t.Errorf("Tools = false, want true for chat model")
	}
	if info.Capabilities.ImageGeneration || info.Capabilities.Embeddings {
		t.Errorf("Capabilities = %+v, want image_generation=false embeddings=false", info.Capabilities)
	}
}

func TestInfo_GeminiImageModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash-image"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !g.Info().Capabilities.ImageGeneration {
		t.Errorf("ImageGeneration = false, want true for gemini-*-image")
	}
}

func TestInfo_ImagenModel(t *testing.T) {
	g, err := New(WithModel("imagen-4.0-generate-001"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	caps := g.Info().Capabilities
	if !caps.ImageGeneration {
		t.Errorf("ImageGeneration = false, want true for imagen-*")
	}
	if caps.Tools {
		t.Errorf("Tools = true, want false for imagen model")
	}
}

func TestInfo_EmbedModel(t *testing.T) {
	g, err := New(WithModel("gemini-embedding-001"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if !g.Info().Capabilities.Embeddings {
		t.Errorf("Embeddings = false, want true for embedding model")
	}
	if got := g.EmbedDimensions(); got != 3072 {
		t.Errorf("EmbedDimensions() = %d, want 3072", got)
	}
}

func TestEmbedDimensions_TextEmbedding004(t *testing.T) {
	g, err := New(WithModel("text-embedding-004"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 768 {
		t.Errorf("EmbedDimensions() = %d, want 768", got)
	}
}

func TestEmbedDimensions_WithDimensionsOverride(t *testing.T) {
	g, err := New(WithModel("gemini-embedding-001"), WithAPIKey("test-key"), WithDimensions(1536))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 1536 {
		t.Errorf("EmbedDimensions() = %d, want 1536 (overridden)", got)
	}
}

func TestEmbedDimensions_NonEmbedModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if got := g.EmbedDimensions(); got != 0 {
		t.Errorf("EmbedDimensions() = %d, want 0 for non-embed model", got)
	}
}

// readAll drains a StreamReader into ordered events (excluding the io.EOF).
func readAll(t *testing.T, sr llm.StreamReader) []llm.StreamEvent {
	t.Helper()
	defer sr.Close()
	var events []llm.StreamEvent
	for {
		ev, err := sr.Next()
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		events = append(events, ev)
	}
}

func TestStream_TextDeltas(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":streamGenerateContent") {
			t.Errorf("path = %s, want suffix :streamGenerateContent", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunks := []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]},"index":0}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"index":0}]}`,
			`{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}`,
		}
		for _, c := range chunks {
			_, _ = io.WriteString(w, "data: "+c+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
	})

	sr, err := g.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := readAll(t, sr)

	var text string
	var sawDone bool
	for _, ev := range events {
		switch ev.Kind {
		case llm.EventTextDelta:
			text += ev.Text
		case llm.EventDone:
			sawDone = true
			if ev.FinishReason != llm.FinishReasonStop {
				t.Errorf("Done.FinishReason = %q, want stop", ev.FinishReason)
			}
			if ev.Usage == nil || ev.Usage.TotalTokens != 4 {
				t.Errorf("Done.Usage = %+v, want total=4", ev.Usage)
			}
		}
	}
	if text != "Hello" {
		t.Errorf("text = %q, want Hello", text)
	}
	if !sawDone {
		t.Error("no EventDone emitted")
	}
}

func TestStream_ToolCallCompleteInOneChunk(t *testing.T) {
	g0 := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunk := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call_9","name":"lookup","args":{"q":"x"}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3,"totalTokenCount":5}}`
		_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	})
	g, err := g0.WithTools([]llm.Tool{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}})
	if err != nil {
		t.Fatalf("WithTools: %v", err)
	}

	sr, err := g.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: "user", Content: "go"}},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	events := readAll(t, sr)

	var sawStart, sawArgs, sawEnd bool
	var argsJSON string
	for _, ev := range events {
		switch ev.Kind {
		case llm.EventToolCallStart:
			sawStart = true
			if ev.ToolCall == nil || ev.ToolCall.Name != "lookup" || ev.ToolCall.ID != "call_9" {
				t.Errorf("Start ToolCall = %+v, want name=lookup id=call_9", ev.ToolCall)
			}
		case llm.EventToolCallArgsDelta:
			sawArgs = true
			if ev.ToolCall != nil {
				argsJSON = ev.ToolCall.ArgsDelta
			}
		case llm.EventToolCallEnd:
			sawEnd = true
		}
	}
	if !sawStart || !sawArgs || !sawEnd {
		t.Fatalf("tool events start=%v args=%v end=%v, want all true", sawStart, sawArgs, sawEnd)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		t.Fatalf("ArgsDelta not valid JSON: %v (%s)", err, argsJSON)
	}
	if args["q"] != "x" {
		t.Errorf("args.q = %v, want x", args["q"])
	}
}

// Accumulate parity: the stream reduces to the same tool call via the
// contract's AccumulateStream helper.
func TestStream_AccumulateParity(t *testing.T) {
	g0 := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"c","name":"f","args":{"a":1}}}]},"finishReason":"STOP","index":0}],"usageMetadata":{"totalTokenCount":7}}`
		_, _ = io.WriteString(w, "data: "+chunk+"\n\n")
	})
	g, _ := g0.WithTools([]llm.Tool{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}})
	sr, err := g.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	resp, err := llm.AccumulateStream(sr)
	if err != nil {
		t.Fatalf("AccumulateStream: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "f" {
		t.Fatalf("accumulated ToolCalls = %+v, want one named f", resp.ToolCalls)
	}
}

func TestWrapErr_StatusMapping(t *testing.T) {
	cases := []struct {
		code   int
		expect func(error) bool
	}{
		{401, func(e error) bool { var a *llm.AuthError; return errors.As(e, &a) }},
		{403, func(e error) bool { var a *llm.AuthError; return errors.As(e, &a) }},
		{429, func(e error) bool { var a *llm.RateLimitError; return errors.As(e, &a) }},
		{500, func(e error) bool { var a *llm.TransientError; return errors.As(e, &a) }},
		{503, func(e error) bool { var a *llm.TransientError; return errors.As(e, &a) }},
		{400, func(e error) bool { var a *llm.InvalidRequestError; return errors.As(e, &a) }},
	}
	for _, tc := range cases {
		g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(tc.code)
			_, _ = w.Write([]byte(`{"error":{"code":` + strconv.Itoa(tc.code) + `,"status":"X","message":"boom"}}`))
		})
		_, err := g.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "x"}}})
		if err == nil {
			t.Fatalf("code %d: Generate err = nil, want typed error", tc.code)
		}
		if !tc.expect(err) {
			t.Errorf("code %d: wrong error type: %v", tc.code, err)
		}
	}
}

func TestGenerate_BlockedPrompt(t *testing.T) {
	g := newTestServer(t, "gemini-2.5-flash", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// 200 OK but no candidates; prompt blocked.
		_, _ = w.Write([]byte(`{"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"blocked"}}`))
	})
	_, err := g.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: "user", Content: "x"}}})
	if err == nil {
		t.Fatal("blocked prompt: Generate err = nil, want InvalidRequestError")
	}
	var inv *llm.InvalidRequestError
	if !errors.As(err, &inv) {
		t.Errorf("blocked prompt: err = %v, want *llm.InvalidRequestError", err)
	}
}

func newTestServerEmbed(t *testing.T, model string, handler http.HandlerFunc, opts ...Option) *Google {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	all := append([]Option{WithModel(model), WithAPIKey("test-key"), WithBaseURL(server.URL)}, opts...)
	g, err := New(all...)
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return g
}

func TestEmbed_Happy(t *testing.T) {
	g := newTestServerEmbed(t, "gemini-embedding-001", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":embedContent") && !strings.HasSuffix(r.URL.Path, ":batchEmbedContents") {
			t.Errorf("path = %s, want embed suffix", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"embeddings":[
				{"values":[0.1,0.2,0.3]},
				{"values":[0.4,0.5,0.6]}
			]
		}`))
	}, WithTaskType("RETRIEVAL_DOCUMENT"), WithDimensions(3))

	vecs, usage, err := g.Embed(context.Background(), []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("vectors len = %d, want 2", len(vecs))
	}
	if len(vecs[0]) != 3 || vecs[0][0] != 0.1 || vecs[1][2] != 0.6 {
		t.Errorf("vectors = %+v, want [[0.1 0.2 0.3] [0.4 0.5 0.6]]", vecs)
	}
	// Gemini Developer API reports no token usage.
	if usage.Source != llm.UsageUnknown || usage.TotalTokens != 0 {
		t.Errorf("usage = %+v, want zero/UsageUnknown", usage)
	}
}

func TestEmbed_EmptyInput(t *testing.T) {
	g, err := New(WithModel("gemini-embedding-001"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	vecs, _, err := g.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed(nil): %v", err)
	}
	if len(vecs) != 0 {
		t.Errorf("vectors len = %d, want 0", len(vecs))
	}
}

func TestEmbed_NonEmbedModel(t *testing.T) {
	g, err := New(WithModel("gemini-2.5-flash"), WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	_, _, err = g.Embed(context.Background(), []string{"x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("Embed on chat model = %v, want ErrCapabilityNotSupported", err)
	}
}
