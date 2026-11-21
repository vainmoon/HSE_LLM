#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec bash serve_prefill4096.sh \
  --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":2}' \
  "$@"
