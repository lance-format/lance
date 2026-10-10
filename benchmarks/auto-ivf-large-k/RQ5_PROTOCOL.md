# Auto IVF probing across index types: RQ5 validation

This contract is recorded before evaluating the new index-type support. The
existing per-metric/k profiles remain frozen; no RQ-specific tuning is performed.

- Candidate: the current PR with the IVF_FLAT and prepared-partition capability
  gates removed from initial probe selection. Existing query-type, metric, k,
  refinement and explicit-budget semantics remain unchanged. The partition
  policy also works with historical IVF readers without querying unsupported
  sub-index metadata.
- Native RQ5 corpus matrix: DINO 10M (L2), LAION 10M and FineWeb 10M (cosine),
  Wiki-Cohere 35M and DPR 21,015,300 (dot). Use the original, verified source
  versions, row IDs, centroids and exact partition membership. Build each RQ5
  index in a separate shallow clone with the frozen centroid matrix and a
  precomputed row-to-partition dataset. Verify all row memberships and centroid
  values after building. Do not mutate or replace the source FLAT index.
- Use `num_bits=5`, default packed storage, `approx_mode=normal`, no refinement,
  no filters and `query_parallelism=1`. Preserve the stored vectors and norms.
  Freeze the default fast-rotation RQ model in `rabitq-model.json` before
  building so the random rotation can be reproduced independently of the index.
- Ground truth: the existing float64 exact top-100,000 prefixes, with the
  published strict-ID tie semantics. Evaluate the first 512 held-out queries
  of the frozen, deduplicated split at k=1, 10, 100, 200, 500 and 1000.
- Four arms on the same query IDs: FLAT Auto, RQ5 Auto, RQ5 fixed-all-partitions,
  and RQ5 original heuristic. The original heuristic uses an explicit full
  `maximum_nprobes`; check equivalence against the preserved pre-change binary
  before using that arm. Each k is queried independently, including the full
  partition arm, because RQ pruning can depend on k.
- The primary measurement is final strict-ID recall against raw-vector truth.
  Also report RQ5 Auto overlap with RQ5 full-partition results and the paired
  difference from FLAT Auto. The full-partition RQ5 arm includes quantization
  and any normal-mode scoring/pruning approximation; it is not a pure
  quantization-only oracle. These losses can interact and are not assumed to
  add or multiply independently.
- Freeze runtime/source hashes, index identities and build settings before
  held-out measurement. Save every returned ID, result count, partition count,
  comparison count and I/O counter. Recompute recall independently from the
  saved IDs and verify that fixed-all searches all partitions and returns k.
- Reuse the original r8i.8xlarge. Prewarm indices. Recall collection may use
  multiple query workers; its timings are diagnostic and are not latency or
  throughput claims. Report the complete matrix, including any group below
  95% mean recall, without retuning on the held-out set. The FLAT routing target
  does not impose an unmeasured 95% final-recall guarantee on quantized indices.
