#!/usr/bin/env bash
# scripts/capture-fixtures-openai.sh -- capture a real OpenAI happy-path fixture.
#
# Usage:
#   OPENAI_API_KEY=... ./scripts/capture-fixtures-openai.sh
#
# Writes:
#   internal/contract/testdata/openai/generate_happy_gpt-4o-mini.json
set -euo pipefail

: "${OPENAI_API_KEY:?must be set}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
out="$repo_dir/internal/contract/testdata/openai/generate_happy_gpt-4o-mini.json"

response_body="$(curl -sS -X POST https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}')"

python3 - <<'PY' "$out" "$response_body"
import json, sys
out, body = sys.argv[1], sys.argv[2]
payload = json.loads(body)
fixture = {
    "scenario": "generate_happy_gpt-4o-mini",
    "request": {
        "method": "POST",
        "path": "/v1/chat/completions",
        "body_assertions": ["gpt-4o-mini", "hello"],
    },
    "response": {
        "status": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, separators=(",", ":")),
    },
    "expect": {
        "error_type": "",
        "response_text": payload["choices"][0]["message"]["content"],
        "finish_reason": payload["choices"][0]["finish_reason"],
        "usage_input_tokens": payload["usage"]["prompt_tokens"],
        "usage_output_tokens": payload["usage"]["completion_tokens"],
        "usage_source": "reported",
        "provider": "openai",
    },
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(fixture, f, indent=2)
    f.write("\n")
PY

echo "Captured: $out"
