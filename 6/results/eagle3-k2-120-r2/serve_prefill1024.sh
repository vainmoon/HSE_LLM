#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec bash serve.sh --max-num-batched-tokens 1024 "$@"
