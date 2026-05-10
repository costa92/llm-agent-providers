#!/usr/bin/env bash
# scripts/capture-fixtures-ollama.sh -- capture a real Ollama happy-path fixture.
#
# Usage:
#   OLLAMA_HOST=http://localhost:11434 ./scripts/capture-fixtures-ollama.sh
#
# Writes:
#   internal/contract/testdata/ollama/generate_happy_llama3.1-8b.json
set -euo pipefail

host="${OLLAMA_HOST:-http://localhost:11434}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
out="$repo_dir/internal/contract/testdata/ollama/generate_happy_llama3.1-8b.json"

response_body="$(curl -sS -X POST "${host}/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"hello"}],"stream":false}')"

python3 - <<'PY' "$out" "$response_body"
import json, sys
out, body = sys.argv[1], sys.argv[2]
payload = json.loads(body)
fixture = {
    "scenario": "generate_happy_llama3.1-8b",
    "request": {
        "method": "POST",
        "path": "/api/chat",
        "body_assertions": ["llama3.1:8b", "\"stream\":false", "hello"],
    },
    "response": {
        "status": 200,
        "headers": {"Content-Type": "application/x-ndjson"},
        "body": json.dumps(payload, separators=(",", ":")) + "\n",
    },
    "expect": {
        "error_type": "",
        "response_text": payload["message"]["content"],
        "finish_reason": "stop" if payload.get("done_reason") in ("stop", "load") else payload.get("done_reason", ""),
        "usage_input_tokens": payload.get("prompt_eval_count", 0),
        "usage_output_tokens": payload.get("eval_count", 0),
        "usage_source": "reported",
        "provider": "ollama",
    },
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(fixture, f, indent=2)
    f.write("\n")
PY

echo "Captured: $out"
