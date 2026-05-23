// Package compat provides shared helpers used by multiple provider
// implementations to avoid byte-for-byte duplication of error mapping
// and default-timeout logic. It is internal/ — downstream consumers
// cannot import it, and the provider packages remain the only public
// API surface.
//
// Added in v0.2.4 (P1-23 extraction). Use it from any provider that
// targets the OpenAI- or Anthropic-compatible SDK families.
package compat
