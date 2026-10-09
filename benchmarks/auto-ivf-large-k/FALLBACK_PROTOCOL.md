# One fallback threshold per metric, independent of k

This experiment evaluates the current fallback heuristic and a replacement with
one constant for each of L2, cosine, and dot. It does not change production
defaults. The completed and ongoing adaptive-profile studies remain unchanged.

## Inputs and acceptance criteria

- Reuse the five frozen corpora, IVF_FLAT indices, exact float64 top-100,000
  oracle, and original calibration/evaluation split from the large-k study.
- Calibrate and evaluate initial routing recall at every integer k from 1 to
  100,000, using cumulative coverage of the exact neighbor prefixes. Retain
  k=1, 2, 10, 11, 100, 101, 200, 500, 1000, 1001, 2000, 5000, 10000, 20000,
  50000, and 100000 as reporting points for scan distributions and count-based
  expansion. This finite range does not establish recall for unbounded k or
  arbitrary data.
- Select parameters using calibration queries only. Require at least 96% mean
  initial routing recall for every corpus and every integer k in this range.
  Among feasible parameters, choose the smallest representable nonnegative f32
  gap threshold,
  which also minimizes initial partition counts. Use no learned floor or cap.
- Freeze all constants before evaluating held-out queries. The held-out target
  remains 95% mean recall per corpus/k. Report misses without retuning.

## Threshold families

1. **Current fallback:** select centroid distances `d_i <= factor * d_0`, using
   the original f32 multiplication and factors 0.6, 7, and 81 for k=1, k=2..10,
   and k>10. Preserve its actual late probing and caller bounds.
2. **Constant signed multiplier:** fit one f32 factor per metric in the same
   formula. Evaluate both signs for dot. Where nearest-distance signs conflict,
   compute optimistic recall upper bounds for factor<1 and factor>=1, including
   count-based late expansion, to distinguish an impossible family from an
   insufficient search grid. These fitted multipliers are routing experiments.
3. **Constant nonnegative gap:** select `d_i - d_0 <= margin * scale`, where
   `scale=d_0` for L2/cosine and `scale=abs(1-d_0)` for dot. Match the existing
   native gap calculator's f32 margin and f64 arithmetic. There is one margin
   per metric, with caller minimum 1 and no learned cap. Preserve its existing
   behavior at zero scale and report any infeasible calibration constraint.

For the monotone gap family, exact order statistics of the normalized gaps of
true neighbors locate the calibration threshold. Start with the reporting
anchors, then check every integer k and add the worst unmet prefix constraint
until the entire range passes. Check the rounded f32 value with the actual
comparison expression; require its preceding f32 value to fail at least one
constraint unless the selected margin is zero. Record per-corpus thresholds as
diagnostic references, but use only the shared metric constant.

Report initial routing recall separately from the unfiltered count-expansion
simulation. The latter searches additional ordered partitions until enough rows
exist for k. Count-based expansion is not a recall guarantee.

## Native validation

- Validate the frozen gap constants through an isolated experimental fallback
  implementation, with one constant per metric and no dependence on k. Preserve
  the existing implementation for historical indices that lack prepared-search
  metadata. Keep fixed probes and caller limits intact.
- Force the modern fallback path with an explicit upper bound equal to the
  index's full partition count. This exercises fallback routing while using
  IVF_FLAT to isolate routing quality from quantization and HNSW traversal.
- At k=1, 10, 100, 1000, 10000, and 100000, run the first 512 original held-out
  queries per corpus. Time the first 128 serially; the remaining 384 contribute
  recall and counters only. Interleave the original fallback binary on the same
  first 32 queries for matched before/after measurements.
- Use the existing r8i.8xlarge, repository release-with-debug profile, affinity
  0-15, 16 Lance/Rayon threads, one BLAS/OMP thread, query_parallelism=1, and
  prewarmed caches. All timed queries must report zero storage bytes read.
  Preparation, compilation, and other CPU-intensive work must finish before
  serial timings begin. Do not overlap this experiment with another timed run.
- Preserve raw returned IDs, exact source/binary/input hashes, and per-query
  records. Independently verify recall, returned counts, query coverage, scan
  counters, hashes, and timing distributions. Routing evaluation uses the full
  held-out population; native results explicitly identify their 512-query subset.
- Run structural tests for fallback selection, k independence, dot sign/scaling,
  caller bounds, and preservation of fixed probes and historical-index behavior.
  Quantized/HNSW accuracy, Float16 accuracy, multivectors, refinement, selective
  filters, Hamming, and nonfinite inputs are not covered by the empirical
  IVF_FLAT recall target.

Report the tradeoff against the original fallback and the existing calibrated
profiles, particularly the cost of a single constant at small k. A threshold
that reaches the target through near-exhaustive scans is a valid measured
outcome, not evidence that the heuristic adapts efficiently.
