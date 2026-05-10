package openai

import (
	"context"
	"errors"

	"github.com/costa92/llm-agent/llm"
	openai "github.com/openai/openai-go/v3"
)

var _ llm.ChatModel = (*OpenAI)(nil)

type OpenAI struct {
	client *openai.Client
	info   llm.ProviderInfo
}

func (o *OpenAI) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := o.toSDKRequest(req)
	completion, err := o.client.Chat.Completions.New(ctx, sdkReq)
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	return o.fromSDKResponse(completion), nil
}

func (o *OpenAI) Stream(_ context.Context, _ llm.Request) (llm.StreamReader, error) {
	return nil, errors.New("openai: streaming not implemented in Phase 1; use Generate")
}

func (o *OpenAI) Info() llm.ProviderInfo { return o.info }
