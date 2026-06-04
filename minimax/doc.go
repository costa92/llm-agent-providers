// Package minimax implements a MiniMax adapter. Chat/stream/tools go over an
// Anthropic-compatible messages API (github.com/anthropics/anthropic-sdk-go);
// image generation (/v1/image_generation) and embeddings (/v1/embeddings) use
// raw net/http against the configured base URL. The adapter satisfies
// llm.ChatModel, llm.ToolCaller, llm.ImageGenerator, and llm.Embedder, with
// capabilities gated per bound model (K2): GenerateImage requires image-01 and
// Embed requires embo-01, otherwise llm.ErrCapabilityNotSupported.
package minimax
