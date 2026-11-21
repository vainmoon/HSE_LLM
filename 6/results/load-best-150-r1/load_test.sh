#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Пример: bash load_test.sh 240 results/load-240
rpm="${1:-120}"
output="${2:-results/load-${rpm}rpm-$(date +%Y%m%d-%H%M%S)}"

BENCHMARK_DURATION=300 exec bash benchmark.sh "$output" "$rpm"
