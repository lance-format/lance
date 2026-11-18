#!/usr/bin/env bash
# Sweep hierarchical Lloyd and hierarchical e-means over vector dimensions,
# e-means batch sizes and epochs, thread counts and scale. See README.md.
#
#   DATA=~/data/dino10b OUT=sweep SWEEP_GROUPS="dims single threads batch epochs large" \
#     ./benchmarks/emeans/sweep.sh
#
# Every run appends one JSON line to $OUT/sweep.jsonl. Timing runs (profiling
# compiled out, every held-out query) also dump per-query distances for paired
# MSE comparisons; profiled runs score fewer queries.
set -euo pipefail

DATA=${DATA:?set DATA to the folder holding base.bvecs, queries_clean.bvecs, pca.f32 and bin/}
OUT=${OUT:-sweep}
SWEEP_GROUPS=${SWEEP_GROUPS:-"dims single threads batch epochs large"}
MAX_THREADS=${MAX_THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
SOURCE_DIM=1024
mkdir -p "$OUT"

# run BUILD ALG DIM ROWS K THREADS SEED BATCH_SCALE EPOCHS
run() {
  local build=$1 alg=$2 dim=$3 rows=$4 k=$5 threads=$6 seed=$7 scale=$8 epochs=$9
  local label="$alg-d$dim-n$rows-k$k-b$scale-e$epochs-t$threads-s$seed-$build"
  if grep -q "\"label\":\"$label\"" "$OUT/sweep.jsonl" 2>/dev/null; then
    return
  fi
  local args=(--base "$DATA/base.bvecs" --queries "$DATA/queries_clean.bvecs"
    --rows "$rows" --k "$k" --node-algorithm "$alg" --threads "$threads" --seed "$seed"
    --batch-scale "$scale" --epochs "$epochs" --label "$label" --out "$OUT/sweep.jsonl")
  if [ "$dim" != "$SOURCE_DIM" ]; then
    args+=(--pca "$DATA/pca.f32" --dim "$dim")
  fi
  if [ "$build" = timing ]; then
    args+=(--dump-query-dists "$OUT/$label.f32")
  else
    args+=(--num-queries 10000 --train-eval-rows 10000)
  fi
  "$DATA/bin/emeans_bench_$build" "${args[@]}" | grep -v '^{' | tee -a "$OUT/log.txt"
}

N=${N:-8388608}
K=${K:-32768}
for group in $SWEEP_GROUPS; do
  case $group in
    dims)
      for dim in 32 64 128 256 512 1024; do
        for seed in 1 2 3 4 5; do
          for alg in lloyd emeans; do run timing $alg $dim $N $K "$MAX_THREADS" $seed 1 3; done
        done
        for threads in "$MAX_THREADS" 1; do
          for alg in lloyd emeans; do run profile $alg $dim $N $K "$threads" 1 1 3; done
        done
      done
      ;;
    single)
      # Unprofiled one-thread runs: the reference for speedup and efficiency,
      # free of the profiler's per-call cost.
      for dim in 32 64 128 256 512 1024; do
        for alg in lloyd emeans; do run timing $alg $dim $N $K 1 1 1 3; done
      done
      ;;
    threads)
      for dim in 64 1024; do
        for threads in 12 6 2; do
          for alg in lloyd emeans; do run profile $alg $dim $N $K "$threads" 1 1 3; done
        done
      done
      ;;
    batch)
      for dim in 64 256 1024; do
        for scale in 0.125 0.25 0.5 2 4 8 32; do
          for seed in 1 2 3; do run timing emeans $dim $N $K "$MAX_THREADS" $seed $scale 3; done
          run profile emeans $dim $N $K "$MAX_THREADS" 1 $scale 3
        done
      done
      ;;
    epochs)
      for dim in 64 256 1024; do
        for epochs in 1 2 5 8; do
          for seed in 1 2 3; do run timing emeans $dim $N $K "$MAX_THREADS" $seed 1 $epochs; done
        done
      done
      ;;
    large)
      for dim in 64 128 256; do
        for seed in 1 2 3; do
          for alg in lloyd emeans; do run timing $alg $dim 33554432 131072 "$MAX_THREADS" $seed 1 3; done
        done
        for alg in lloyd emeans; do run profile $alg $dim 33554432 131072 "$MAX_THREADS" 1 1 3; done
      done
      for alg in lloyd emeans; do run profile $alg 128 33554432 131072 1 1 1 3; done
      ;;
    *)
      echo "unknown group $group" >&2
      exit 1
      ;;
  esac
done
