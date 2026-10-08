# Auto IVF probing for k > 100 (OSS-2254)

Policy choices and acceptance thresholds were frozen before calibration.
Audit clarifications below were made before native evaluation. All execution
takes place on one AWS r8i.8xlarge.

- Baseline: Lance main `567e322b6`. Its Auto policy uses calibrated profiles
  for k <= 100 and the legacy `d_i <= 81 * d_0` threshold for k > 100.
- Corpora, preserving raw vectors, row order and metric semantics:
  DINO 10M x 1024 L2 and LAION 10M x 768 / FineWeb 10M x 768 cosine, restored
  byte-for-byte with the frozen IVF_FLAT indices that calibrated the existing
  L2/cosine profiles; HF `lance-format/wiki-cohere-35m` and
  `lance-format/dpr-wikipedia-single-nq` dot at the OSS-2249 revisions, with
  fresh IVF_FLAT indices using `ceil(rows / 4096)` partitions and default training.
- Ground truth: exact float64 top-5000 source row positions for every query,
  cross-checked against published neighbors where present. Retained equal-score
  neighbors are ordered by position; ties crossing the top-5000 cutoff follow
  NumPy argpartition selection. Strict-ID recall counts alternative tied IDs as
  misses. This clarifies the implementation's tie handling; no query set or
  acceptance threshold is changed.
- Split every query set with NumPy seed 2254: first half calibration, second half
  evaluation. Before native evaluation, exclude evaluation vectors identical to
  any calibration vector or earlier evaluation vector. The original calibration
  set and fitted parameters remain unchanged. Record exclusions in
  `split-audit.json`; this removes three LAION evaluation entries (two cross-split
  duplicates and one within-evaluation duplicate) without selecting queries based
  on recall. Only calibration queries select parameters.
- Policy family: the existing relative centroid-distance gap with a learned
  initial floor and cap (L2/cosine scale by `d_0`, dot by `abs(1 - d_0)`).
  Caller minimums, caller maximums, fixed nprobes and late probing are unchanged.
- Primary design: buckets 101-200, 201-500 and 501-1000, each calibrated at
  its upper endpoint. Alternative design: one bucket 101-1000 calibrated at 1000.
  Choose the primary design unless the alternative costs no more than 5% extra
  mean partitions at every simulated k in the bucket range.
- Objective per metric and anchor: minimize the corpus-averaged mean initial
  partition count subject to 96% mean calibration recall on every corpus of that
  metric. Coarse geometric floor/cap grid (ratio 1.08) and margin grid, then a
  local integer/fine-margin refinement. Also freeze per-corpus optima as `tuned`.
- k > 1000: calibrate anchors 2000 and 5000 on calibration queries only to
  measure how the optimum scales. Decide k > 1000 behavior from calibration
  evidence; report held-out routing recall for every option considered.
- Held-out native measurement at k = 100 (boundary control, existing profile),
  101, 200, 500 and 1000. Arms: Auto, `tuned`, fixed20, and the calibration fixed
  budgets bracketing 95% recall. The `legacy` arm runs main's actual frozen binary
  in a separate process for 32 queries per k, interleaved with the candidate.
  Both processes prewarm the same index and use the same affinity and thread
  settings; the measured interval excludes IPC. This strengthens the original
  plan to reproduce legacy with an explicit full `maximum_nprobes` on the
  candidate. An eight-query baseline audit checks that policy equivalence.
  Before/after latency ratios use only the same 32 queries in both binaries.
- Timing: first 512 held-out queries per arm, serial, rotating policy order,
  CPU affinity 0-15, Lance/Rayon threads 16, BLAS/OMP threads 1,
  query_parallelism 1, prewarmed index cache and zero storage I/O.
  Remaining held-out queries contribute recall and scan counts only.
- Filtered check: prefilter `_rowid % 1000 = 0` at k = 101 and 1000 for 256
  held-out queries, against exact filtered ground truth. Report returned counts,
  recall and partitions for Auto, legacy and fixed20.
- Acceptance: Auto mean held-out recall >= 95% for every corpus and measured k.
  Report every miss without retuning on held-out queries.
- Report recall and mean/p90/p95/p99/max latency, scanned rows and partitions.
  Use NumPy linear percentiles. Do not present different-recall comparisons as
  equal-quality speedups.
