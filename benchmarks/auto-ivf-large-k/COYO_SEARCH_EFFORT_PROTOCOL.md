# COYO-VE IVF_RQ 5bit search-effort protocol

This follow-up was frozen before inspecting COYO search results. It measures
the implementation published in PR #9798 at
`3d44e1965853fd058e0b745d77c359ca50128b75`, without tuning Auto on COYO.
The earlier five-corpus results retain their original identities.

## Source and index

- Source: `lance-format/coyo-ve-qwen3vl-2048`, revision
  `95efb82f4914f320e59cf286a06f23b7cfdb2112`.
- All 15,380,795 corpus rows, 2048 float32 dimensions, cosine distance.
  Preserve the original vectors, corpus order and upstream identifiers.
- Verify every downloaded file against Hub LFS SHA256 or Git blob identity.
- Build one IVF_RQ index with five bits, normal approximation and a saved
  fast-rotation model. Use 3,756 partitions (`ceil(rows / 4096)`), following
  this harness's index-building rule. Keep the native training defaults.
- Preserve the source dataset and build in a separate shallow clone. Save
  centroids, model, row-ID mapping, complete partition membership, index
  metadata and hashes. Verify every corpus row occurs exactly once.
- All effort settings and the old binary use this same frozen index.

## Queries and ground truth

- Apply the existing seed-2254 split to all 25,000 public query vectors.
  The first half of the permutation is the calibration pool. Remove from
  evaluation any vector identical to a calibration vector or an earlier
  evaluation vector, as in `prepare.save_query_split`.
- Select the first two calibration queries for smoke/default checks and
  the first 128 evaluation queries for measurement. Save their original
  source positions and `query_id` strings before index search. The compact
  query dataset retains this mapping; local query positions are 0 through
  129, with calibration at 0/1 and evaluation at 2 through 129.
- Recompute float64 exact cosine top-100,000 from every original corpus
  vector for these 130 queries. Use the existing blockwise `exact_truth`
  implementation. Both query and corpus norms enter cosine computation.
  Cutoff ties follow NumPy argpartition; selected equal-score IDs sort by
  corpus position. Alternative tied IDs count as strict-ID misses.
- Cross-check top-1,000 against the public neighbor IDs and scores. Retain
  and explain numerical/tie differences instead of changing the oracle or
  dropping queries. Verify the oracle path against a small independent
  exhaustive reference before the full computation.

## Measurement

- `search_effort`: 0, 0.25, 0.5, 0.75 and 1.
- k: 1, 10, 100, 1,000, 10,000 and 100,000.
- The actual default baseline is the preserved binary of
  `40fa85849ddf5dd85d2925462a0ed42b31287a7c`, in its own warm process.
  The candidate native binary remains the already validated
  `ce1320d527deb815a94a0c6dba070dd87289b2696991f31fa4b256f9b7c05a52`.
- Same AWS r8i.8xlarge, `release-with-debug`, CPU affinity 0-15,
  16 Lance/Rayon threads, one OMP/BLAS thread, `query_parallelism=1`.
  No refinement or filter; default caller probe bounds; warm index caches.
- Measure recall on all 128 evaluation queries. Time the first 64 in one
  serial pass with rotating arm order; compute the remaining 64 separately
  with eight workers and exclude their timings from the latency summary.
- Time scanner construction through materialized results inside each
  process. Exclude IPC, ID mapping and recall calculation from both timers.
- Report all 36 groups and 4,608 per-query records. The 768 candidate-default
  queries must exactly match baseline IDs, partition counts and comparisons.
  Effort 1 must search every partition. Require exactly k distinct valid IDs
  and zero measured storage bytes for every query.
- Independently recompute recall from saved result IDs, audit query coverage,
  disjoint splits and pre/post identities, and run a deliberate duplicate-ID
  negative control on the calibration smoke results.
- Report mean/p50/p95 latency and recall distributions. A single timed pass
  is not a repeated-run confidence interval, and no default-speedup claim
  follows from it. Full-partition RQ5 remains approximate.

Preserve all raw records, IDs, source/model/index identities and validation
evidence. Verify the local evidence archive before stopping the VM; retain
the original corpus and frozen index on its volume.
