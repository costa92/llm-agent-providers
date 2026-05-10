package anthropic

import (
	"context"
	"errors"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/costa92/llm-agent/llm"
)

var _ llm.ChatModel = (*Anthropic)(nil)

type Anthropic struct {
	client *sdk.Client
	info   llm.ProviderInfo
}

func (a *Anthropic) Generate(ctx context.Context, req llm.Request) (llm.Response, error) {
	sdkReq := a.toSDKRequest(req)
	msg, err := a.client.Messages.New(ctx, sdkReq)
	if err != nil {
		return llm.Response{}, wrapErr(err)
	}
	return a.fromSDKResponse(msg), nil
}

func (a *Anthropic) Stream(_ context.Context, _ llm.Request) (llm.StreamReader, error) {
	return nil, errors.New("anthropic: streaming not implemented in Phase 1; use Generate")
}

func (a *Anthropic) Info() llm.ProviderInfo { return a.info }
