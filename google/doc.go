// Package google implements a Gemini adapter over the official
// google.golang.org/genai SDK (Gemini Developer API backend).
//
// The adapter satisfies llm.ChatModel, llm.ToolCaller, llm.ImageGenerator,
// and llm.Embedder. Capabilities reported via Info() are per-(provider ×
// model): the constructor binds one model, and Info() reflects what that
// model can do (Keystone K2). A gemini-2.5-flash instance does chat + tools
// but GenerateImage/Embed return llm.ErrCapabilityNotSupported; a
// gemini-2.5-flash-image / imagen-* instance generates images; a
// gemini-embedding-001 / text-embedding-004 instance embeds.
//
// Gemini has no system role: llm.Request.SystemPrompt maps to
// GenerateContentConfig.SystemInstruction. Streaming bridges the SDK's
// iter.Seq2 to the repo's pull-based llm.StreamReader via iter.Pull2;
// streamed tool calls arrive complete in one chunk. Images are always
// returned as inline bytes (never URL).
package google
