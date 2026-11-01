# Dot auto-probing calibration (OSS-2249)

All execution takes place on AWS. No local benchmark or test runs.

- Baseline: Lance main `00a8230ba21341966cb3edac17b475b7b9219302`.
- Data: complete `lance-format/wiki-cohere-35m` (35,000,000 rows, 5,000 queries)
  and `lance-format/dpr-wikipedia-single-nq` (21,015,300 rows, 3,610 queries).
  Pin HF revisions and verify every published SHA256 before use.
- Preserve original float32 vectors and dot semantics. Never normalize the
  corpus or queries. Ground-truth IDs are source row positions.
- Build one IVF_FLAT index per corpus from the baseline, using
  `ceil(rows / 4096)` partitions and the default training/assignment settings.
  Freeze its centroids, partition membership, source, native binary and manifest.
  Baseline and candidate queries use these identical indices.
- Split each official query set with NumPy RNG seed 2249: first half calibration,
  second half evaluation. Freeze IDs before calibration. Tune only on calibration.
- Primary objective: minimize mean scanned partitions subject to recall >= 0.95
  separately for each corpus and k in {1, 10, 100}. Report per-corpus optima and
  the best common profile. Best means best in the documented search grid;
  no universal optimum or per-query recall guarantee is claimed.
  The common-profile objective weights the two corpora equally. Select one
  score-gap family using the sum of its mean costs across the three k groups;
  tune floor, cap and margin independently within each k group.
- First collect actual ground-truth partition ranks and centroid scores. Use
  these to evaluate policies cheaply, then verify shortlisted policies using
  actual index queries. Routing coverage alone is not a latency measurement.
- Diagnose query/vector norms, score offsets, relative centroid gaps and
  query-projected residuals. Distinguish observed correlation from causation.
- Compare current main Auto, calibrated Auto, and nearby power-of-two fixed
  budgets. Select the fixed budgets on calibration queries, not evaluation.
  Also evaluate the independently frozen per-corpus profile within the selected
  score-gap family as `tuned`; report any held-out recall misses without retuning.
- Execute serial queries, balanced interleaved policy order, warmed index cache,
  identical thread/CPU settings. Record source and binary identity for every run.
  Keep compilation, index construction and other benchmarks out of timing runs.
- Freeze the first 512 evaluation queries for matched latency distributions.
  Measure candidate Auto and all nearby fixed budgets selected on calibration
  queries on every evaluation query for recall. Measure main's legacy policy
  on the first 32 evaluation queries as a diagnostic, since it scans all 35M
  rows for Wiki k=10/100. Report its smaller sample count and do not use its
  latency percentiles as the primary performance baseline.
- Report raw per-query recall, elapsed ms, scanned rows and scanned partitions;
  summarize mean/p90/p95/p99/max with NumPy linear percentiles. Capture I/O
  counters and validate actual scan metrics against the policy predictions.
- After each serial timing phase, run the remaining correctness queries with
  eight workers. Tag those records `phase=recall` and exclude their elapsed times
  from performance summaries. Join all workers before the next timing phase.
  Run per-corpus override profiles in a separate recall phase with constant
  process-wide environment settings to prevent cross-query configuration races.
- Independently evaluate the frozen common profile. If a dataset misses 95%,
  report the miss; do not tune on those evaluation queries and relabel it held out.
- Preserve fixed/min/max semantics, unsupported-index/metric behavior, late
  probing for filtered queries, and existing L2/cosine profiles. Add regression
  coverage and run the prescribed Rust/Python checks on AWS before delivery.
