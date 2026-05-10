#!/usr/bin/env bash
# scripts/capture-fixtures-anthropic.sh -- capture a real Anthropic happy-path fixture.
#
# Usage:
#   ANTHROPIC_API_KEY=... ./scripts/capture-fixtures-anthropic.sh
#
# Writes:
#   internal/contract/testdata/anthropic/generate_happy_claude-3-5-haiku.json
set -euo pipefail

: "${ANTHROPIC_API_KEY:?must be set}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
out="$repo_dir/internal/contract/testdata/anthropic/generate_happy_claude-3-5-haiku.json"

response_body="$(curl -sS -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: ${ANTHROPIC_API_KEY}" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-3-5-haiku-20241022","max_tokens":128,"messages":[{"role":"user","content":"hello"}]}')"

python3 - <<'PY' "$out" "$response_body"
import json, sys
out, body = sys.argv[1], sys.argv[2]
payload = json.loads(body)
text = "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text")
fixture = {
    "scenario": "generate_happy_claude-3-5-haiku",
    "request": {
        "method": "POST",
        "path": "/v1/messages",
        "body_assertions": ["claude-3-5-haiku-20241022", "hello"],
    },
    "response": {
        "status": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, separators=(",", ":")),
    },
    "expect": {
        "error_type": "",
        "response_text": text,
        "finish_reason": "stop" if payload.get("stop_reason") in ("end_turn", "stop_sequence") else payload.get("stop_reason", ""),
        "usage_input_tokens": payload["usage"]["input_tokens"],
        "usage_output_tokens": payload["usage"]["output_tokens"],
        "usage_source": "reported",
        "provider": "anthropic",
    },
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(fixture, f, indent=2)
    f.write("\n")
PY

echo "Captured: $out"
