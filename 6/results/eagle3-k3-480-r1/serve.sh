#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Личный каталог исключает конфликт с чужими lock-файлами в /tmp.
export TMPDIR="$PWD/.tmp"
mkdir -p -m 700 "$TMPDIR"

# Включаем API очистки кеша
export VLLM_SERVER_DEV_MODE=1

exec vllm serve Qwen/Qwen3-8B "$@"
