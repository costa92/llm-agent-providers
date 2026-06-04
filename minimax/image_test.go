package minimax

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestGenerateImage_MiniMax_Happy(t *testing.T) {
	var gotPath, gotBody, gotAuth, gotExtra string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotExtra = r.Header.Get("X-Trace")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"data":{"image_urls":["https://cdn.minimax/img1.png","https://cdn.minimax/img2.png"]},
			"base_resp":{"status_code":0,"status_msg":"success"}
		}`))
	}))
	defer server.Close()

	m, err := New(
		WithModel("image-01"),
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithExtraHeaders(map[string]string{"X-Trace": "abc"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := m.GenerateImage(context.Background(), llm.ImageRequest{
		Prompt: "a robot",
		N:      2,
		Size:   "1024x768",
	})
	if err != nil {
		t.Fatalf("GenerateImage: %v", err)
	}
	if gotPath != "/v1/image_generation" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotAuth != "Bearer k" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if gotExtra != "abc" {
		t.Fatalf("X-Trace = %q", gotExtra)
	}
	if !strings.Contains(gotBody, `"model":"image-01"`) {
		t.Fatalf("body missing model: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"prompt":"a robot"`) {
		t.Fatalf("body missing prompt: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"response_format":"url"`) {
		t.Fatalf("body missing response_format: %s", gotBody)
	}
	if !strings.Contains(gotBody, `"width":1024`) || !strings.Contains(gotBody, `"height":768`) {
		t.Fatalf("body missing width/height from Size: %s", gotBody)
	}
	if len(resp.Images) != 2 {
		t.Fatalf("len(Images) = %d, want 2", len(resp.Images))
	}
	if resp.Images[0].URL != "https://cdn.minimax/img1.png" {
		t.Fatalf("Images[0].URL = %q", resp.Images[0].URL)
	}
	if resp.Provider != "minimax" || resp.Model != "image-01" {
		t.Fatalf("provider/model = %q/%q", resp.Provider, resp.Model)
	}
}

func TestGenerateImage_MiniMax_LogicalFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// HTTP 200 but logical failure — the MiniMax gotcha.
		_, _ = w.Write([]byte(`{"base_resp":{"status_code":1004,"status_msg":"invalid api key"}}`))
	}))
	defer server.Close()

	m, err := New(WithModel("image-01"), WithAPIKey("bad"), WithBaseURL(server.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = m.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if err == nil {
		t.Fatal("want error on status_code=1004")
	}
	if !errors.As(err, new(*llm.AuthError)) {
		t.Fatalf("err = %v, want AuthError", err)
	}
}

func TestGenerateImage_MiniMax_NotSupported(t *testing.T) {
	m, err := New(WithModel("MiniMax-M1"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = m.GenerateImage(context.Background(), llm.ImageRequest{Prompt: "x"})
	if !errors.Is(err, llm.ErrCapabilityNotSupported) {
		t.Fatalf("err = %v, want ErrCapabilityNotSupported", err)
	}
}

func TestInfo_MiniMax_ImageModel(t *testing.T) {
	m, err := New(WithModel("image-01"), WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !m.Info().Capabilities.ImageGeneration {
		t.Fatalf("image-01 must report ImageGeneration: %+v", m.Info().Capabilities)
	}
}
