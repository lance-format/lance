#!/usr/bin/env bash
# Run the hierarchical Lloyd vs hierarchical e-means benchmark on DINO10B.
# See README.md for the data download and the binaries.
#
#   DATA=~/data/dino10b OUT=results ./benchmarks/emeans/run.sh
set -euo pipefail

DATA=${DATA:?set DATA to the folder holding base.bvecs, queries_clean.bvecs and bin/}
OUT=${OUT:-results}
ROWS=${ROWS:-8388608}
K=${K:-32768}
SEEDS=${SEEDS:-"1 2 3 4 5"}
MAX_THREADS=${MAX_THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
SCALING_THREADS=${SCALING_THREADS:-"$MAX_THREADS 12 6 2 1"}
mkdir -p "$OUT"

COMMON=(--base "$DATA/base.bvecs" --queries "$DATA/queries_clean.bvecs" --rows "$ROWS" --k "$K")

# Speed and quality: profiling compiled out, all threads, several seeds, every
# held-out query, per-query distances kept for the paired comparison.
for seed in $SEEDS; do
  for alg in lloyd emeans; do
    label="$alg-t$MAX_THREADS-s$seed"
    "$DATA/bin/emeans_bench_timing" "${COMMON[@]}" --node-algorithm "$alg" \
      --threads "$MAX_THREADS" --seed "$seed" --label "$label" \
      --dump-query-dists "$OUT/$label.f32" --out "$OUT/timing.jsonl" \
      | grep -v '^{' | tee -a "$OUT/log.txt"
  done
done

# Time breakdown and thread scaling: profiled build, one seed. Quality comes
# from the runs above, so fewer queries keep the single-thread scoring short.
for threads in $SCALING_THREADS; do
  for alg in lloyd emeans; do
    "$DATA/bin/emeans_bench_profile" "${COMMON[@]}" --node-algorithm "$alg" \
      --threads "$threads" --seed 1 --num-queries 10000 --train-eval-rows 10000 \
      --out "$OUT/profile.jsonl" | grep -v '^{' | tee -a "$OUT/log.txt"
  done
done
