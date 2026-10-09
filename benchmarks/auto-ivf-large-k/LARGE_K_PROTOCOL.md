# Additional calibration at k=10,000 and k=100,000

This is a separate, preregistered experiment extending the completed k<=1000
study. It measures parameters at the two requested anchors; it does not by
itself change production defaults or claim coverage of intermediate k values.

- Reuse all five frozen corpora, indices, raw vectors, and the exact existing
  calibration/evaluation split, including the LAION duplicate exclusions.
  Preserve the completed study and write new artifacts in a separate directory.
- Recompute exact float64 top-100,000 neighbors for every original query using
  the existing streaming exact-search implementation. Save IDs and scores as
  memory-mappable arrays. Compare the first 5,000 scores with the previous
  oracle; different IDs with identical cutoff scores are not oracle failures.
- Use the same relative centroid-gap family and calibration search grid.
  Independently fit (margin, initial floor, initial cap) at k=10,000 and
  k=100,000. Minimize equally weighted mean initial partitions subject to
  96% mean routing recall on every calibration corpus of each metric.
- Record shared metric profiles, per-corpus reference profiles, fixed20, and
  nearby fixed budgets bracketing 95% recall. Freeze parameters before inspecting
  held-out results. Do not retune after held-out evaluation.
- Evaluate routing coverage over every held-out query. Validate the frozen
  profiles with the actual native IVF_FLAT search, using an experimental build
  that extends only the policy gate and anchor table. Preserve the production
  binary and record the exact experimental patch and binary/source hashes.
- Native queries use the same r8i.8xlarge, release-with-debug profile, 16
  Lance/Rayon threads, one BLAS/OMP thread, CPU affinity 0-15, query_parallelism=1,
  frozen indices and prewarmed caches. Run timings serially. Time the first 512
  held-out queries for each candidate arm; the remaining held-out queries
  contribute recall and counters only. Compare the previous main binary's
  default Auto on the same first 32 queries, interleaved with the candidate.
- Preserve returned IDs in binary arrays rather than embedding 100,000 IDs in
  each CSV field. Independently recompute recall, returned counts, query
  coverage, scan counters, timing distributions and input/output hashes.
- Report mean/p90/p95/p99/max latency, searched partitions and scanned rows,
  returned counts and strict-ID recall. Scanned rows mean the native
  index_comparisons counter, which charges storage lengths before filtering.
  Record zero-storage-I/O checks and all baseline/candidate recall differences.
- The held-out target remains 95% mean recall per corpus and requested k.
  Report any miss without changing the threshold. This additional experiment
  concerns unfiltered queries; the earlier selective-filter limitations remain.

The archived k<=1000 results, parameters, binaries and raw records are immutable
inputs to this experiment, not overwritten outputs.
