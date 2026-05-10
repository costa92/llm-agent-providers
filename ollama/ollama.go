package ollama

import (
	"context"
	"errors"

	"github.com/costa92/llm-agent/llm"
	api "github.com/ollama/ollama/api"
)

var _ llm.ChatModel = (*Ollama)(nil)

type Ollama struct {
	client     *api.Client
	info       llm.ProviderInfo
	lastStatus *int32
}

func (o *Ollama) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := o.toSDKRequest(req)
	var captured api.ChatResponse
	err := o.client.Chat(ctx, sdkReq, func(resp api.ChatResponse) error {
		captured = resp
		return nil
	})
	if err != nil {
		return llm.Response{}, o.wrapErr(err)
	}
	return o.fromSDKResponse(captured), nil
}

func (o *Ollama) Stream(_ context.Context, _ llm.Request) (llm.StreamReader, error) {
	return nil, errors.New("ollama: streaming not implemented in Phase 1; use Generate")
}

func (o *Ollama) Info() llm.ProviderInfo { return o.info }
