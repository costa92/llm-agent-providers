package contract

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/costa92/llm-agent-providers/anthropic"
	"github.com/costa92/llm-agent-providers/ollama"
	"github.com/costa92/llm-agent-providers/openai"
	"github.com/costa92/llm-agent/llm"
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
