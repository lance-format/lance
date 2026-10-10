# Continuous search effort with IVF_RQ 5bit

This protocol is fixed before inspecting effort recall or timing results.

## Query contract

- `search_effort` is a finite number in [0, 1], default 0.5.
- First compute the initial budget B using the existing Auto policy. An effort
  of 0.5 preserves the current behavior without floating-point interpolation.
- For other efforts, let U be the available partition count clipped by an
  explicit caller maximum, and L be the caller minimum, at least one, clipped
  to U. Clip B to [L, U]. Empty candidate sets retain a zero budget.
- For e < 0.5, select ceil(L * (B/L)^(2e)); for e > 0.5, select
  ceil(B * (U/B)^(2e-1)). Handle e=0 and e=1 exactly as L and U. Clip rounding
  to [L, U]. Do not reapply the learned floor or cap after interpolation.
- Candidate-count expansion after initial probing remains unchanged. Effort
  controls the initial partition budget, not a hard latency or recall target.
- Non-default effort with fixed nprobes is rejected. Default effort preserves
  fixed probing, explicit bounds, metric/k fallback and all existing behavior.
- Keep approximation mode, HNSW ef and refinement independent of effort.

## Frozen experiment

- Baseline: existing PR head `40fa85849ddf5dd85d2925462a0ed42b31287a7c`, built
  before applying the new implementation. Record source archive and native
  library SHA256. Build baseline and candidate with `release-with-debug`, the
  same toolchain and dependency locks on the same AWS r8i.8xlarge.
- Reuse all five frozen RQ5 indices from the previous campaign: DINO 10M (L2),
  LAION 10M and FineWeb 10M (cosine), Wiki-Cohere 35M and DPR 21,015,300 (dot).
  Preserve their centroids, row assignments, packed five-bit codes and saved
  fast-rotation models. Do not rebuild indices or alter previous artifacts.
- Efforts: 0, 0.25, 0.5, 0.75, 1. k: 1, 10, 100, 1000, 10000, 100000.
- Use the first 128 original held-out query ids for each corpus at every k and
  effort. Use the existing float64 top-100000 ground truth. These samples differ
  in size from the earlier 512-query RQ5 report; compare only matched queries.
- Use normal approximation, no refinement, no filter, query_parallelism=1,
  16 Lance/Rayon threads, one BLAS/OMP thread and CPU affinity 0-15.
- Prewarm baseline and candidate index caches. Primary latency uses the first
  64 selected queries, one query/arm at a time, with rotating arm order across
  the five candidate efforts and the actual baseline binary. Measure inside
  each process with the same scanner-to-materialized-result timing boundary;
  exclude baseline IPC, id conversion and recall calculation.
- Remaining recall queries may use eight workers, separately from all serial
  timing. Concurrent timings are diagnostic only and are not performance claims.
- Keep every returned id and per-query recall, probe count, comparison count,
  storage bytes and elapsed time. Report mean recall, recall distribution,
  mean/p50/p95 serial latency and scanned work. Quantiles describe 64 queries
  in one timed pass, not repeated-run confidence intervals.

## Acceptance and interpretation

- Independently audit query coverage, split separation, returned-id validity,
  uniqueness/count, exact recall recomputation and zero measured storage reads.
- Candidate effort 0.5 must match actual baseline default ids and probe counts
  on the same 128 queries at every corpus/k. Also test omitted effort versus
  explicit 0.5 in the candidate and on API/serialization regression fixtures.
- Effort 1 must probe every available partition in the unbounded experiment.
  Unit tests must establish monotonic initial budgets across efforts, bounds,
  endpoint handling and default identity. Final recall and latency need not be
  monotonic: quantized scoring, pruning and candidate expansion remain active.
- Use calibration queries only for a small harness smoke check. Freeze code
  and parameters before the held-out run; correct implementation or audit bugs
  transparently, without tuning effort behavior to improve measured recall.
- Rehash source, binaries, query inputs and index/model files after the run.
  Preserve old and new raw evidence, and stop the VM after backup verification.
