#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

url="${URL:-http://localhost:8000}"
data="data/requests.jsonl"
output="${1:-results/$(date +%Y%m%d-%H%M%S)}"
rpm="${2:-120}"
rps="$(awk -v rpm="$rpm" 'BEGIN { print rpm / 60 }')"
mkdir -p "$(dirname "$output")"
mkdir "$output"

# Сохраняем окружение и входные параметры каждого замера.
sha256sum "$data" > "$output/data.sha256"
python -m pip freeze > "$output/packages.txt"
cp serve.sh benchmark.sh load_test.sh "$output/"
python - "$output" "$rpm" "${BENCHMARK_DURATION:-0}" <<'PY'
import json
import platform
import sys
from pathlib import Path

output, rpm, duration = sys.argv[1:]
Path(output, "run.json").write_text(json.dumps({
    "rpm": float(rpm),
    "duration_s": int(duration),
    "python": platform.python_version(),
    "platform": platform.platform(),
}, indent=2) + "\n")
PY

trap 'curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null' EXIT
curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null

common=(
  --url "$url" --model Qwen/Qwen3-8B
  --endpoint-type chat --custom-dataset-type raw_payload
  --input-file "$data" --dataset-sampling-strategy sequential
  --streaming --request-timeout-seconds 120
  --random-seed 42 --ui simple --export-outputs-json
)

# Прогрев вынесен отдельно, чтобы очистить его KV-кеш до измерений.
aiperf profile "${common[@]}" --request-count 40 --concurrency 8 \
  --output-artifact-dir "$output/warmup" 2>&1 | tee "$output/warmup.log"
curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null

limit=(--request-count "$(wc -l < "$data")")
if [[ "${BENCHMARK_DURATION:-0}" != 0 ]]; then
  limit=(--benchmark-duration "$BENCHMARK_DURATION" --benchmark-grace-period inf)
fi

aiperf profile "${common[@]}" "${limit[@]}" \
  --request-rate "$rps" --arrival-pattern constant \
  --output-artifact-dir "$output/profile" 2>&1 | tee "$output/benchmark.log"
