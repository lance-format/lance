# COYO-VE: IVF_RQ 5bit search effort

Full COYO-VE, **15,380,795 rows × 2048 float32 dimensions**, cosine,
one frozen **3,756-partition IVF_RQ5** index. This is an additional dataset
evaluation of the implementation at `3d44e1965`; Auto parameters were not
tuned on COYO. The [protocol](COYO_SEARCH_EFFORT_PROTOCOL.md) was fixed before
inspecting search results.

## Recall and latency

Each cell is **mean strict-ID recall / mean serial latency**. Higher recall
and lower latency are better. Recall uses 128 held-out queries; latency uses
the first 64 in one serial pass with rotating arm order. Every k and effort
uses the same query IDs. A single pass gives a query distribution, not a
repeated-run confidence interval or a recall guarantee.

| k | Effort 0 | Effort 0.25 | Effort 0.5 (Auto) | Effort 0.75 | Effort 1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 69.531% / 5.366 ms | 89.844% / 5.007 ms | 96.094% / 5.096 ms | 97.656% / 20.202 ms | 97.656% / 407.701 ms |
| 10 | 64.219% / 5.534 ms | 89.688% / 5.156 ms | 96.016% / 5.865 ms | 98.047% / 26.036 ms | 98.125% / 412.696 ms |
| 100 | 56.617% / 5.635 ms | 87.148% / 5.464 ms | 96.500% / 7.098 ms | 98.891% / 30.397 ms | 99.023% / 407.061 ms |
| 1000 | 46.185% / 6.329 ms | 84.347% / 6.867 ms | 97.245% / 12.532 ms | 99.097% / 51.805 ms | 99.255% / 418.567 ms |
| 10000 | 42.479% / 12.567 ms | 72.697% / 16.018 ms | 96.217% / 35.154 ms | 98.905% / 97.665 ms | 99.292% / 425.756 ms |
| 100000 | 44.249% / 80.277 ms | 46.275% / 86.380 ms | 94.750% / 249.101 ms | 98.302% / 359.102 ms | 99.236% / 591.728 ms |

The [complete CSV](coyo-search-effort-results.csv) contains all 36 groups,
including the actual old-binary default, minimum/p50/p95 recall,
mean/p50/p95/max serial latency, and partition/comparison counts.
Effort 1 still uses normal-mode quantized scoring and pruning; full partition
coverage does not imply exact search. Latency and final recall need not be
monotonic in effort.

In this sample, Auto reaches at least 95% mean recall at five of the six k
values; at k=100000 it reaches 94.750%. Effort 0.75 raises that point to
98.302% and is within 0.934 percentage points of effort 1 at every k.
These observations apply to the frozen query sample and index.

## Search work

These are final partition counts, including any extra probing needed to
obtain k candidates. Effort 0 starts at the caller minimum and can expand.

| k | Effort 0 | Effort 0.25 | Effort 0.5 (Auto) | Effort 0.75 | Effort 1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 partitions | 2.984 partitions | 8.758 partitions | 160.625 partitions | 3756.000 partitions |
| 10 | 1.000 partitions | 3.906 partitions | 16.117 partitions | 225.047 partitions | 3756.000 partitions |
| 100 | 1.000 partitions | 4.859 partitions | 25.875 partitions | 272.281 partitions | 3756.000 partitions |
| 1000 | 1.000 partitions | 7.844 partitions | 67.984 partitions | 465.852 partitions | 3756.000 partitions |
| 10000 | 2.836 partitions | 13.813 partitions | 188.586 partitions | 799.953 partitions | 3756.000 partitions |
| 100000 | 25.875 partitions | 28.336 partitions | 756.898 partitions | 1685.016 partitions | 3756.000 partitions |

## Matched default control

All **768** candidate-default queries exactly match the preserved old binary
at `40fa85849` in returned IDs, partition counts and comparisons. Both
processes use the same frozen index and warm caches. Timing is inside each
process, from scanner construction through materialized results; IPC, ID
mapping and recall computation are excluded. The ratio is baseline/current
mean latency from this pass, not an established speedup.

| Scenario | Baseline default | This PR effort 0.5 | Observed ratio |
| --- | ---: | ---: | ---: |
| k=1 | 6.569 ms | 5.096 ms | 1.289x |
| k=10 | 7.390 ms | 5.865 ms | 1.260x |
| k=100 | 8.518 ms | 7.098 ms | 1.200x |
| k=1000 | 13.704 ms | 12.532 ms | 1.094x |
| k=10000 | 36.166 ms | 35.154 ms | 1.029x |
| k=100000 | 250.379 ms | 249.101 ms | 1.005x |

## Ground truth and source

Source: [lance-format/coyo-ve-qwen3vl-2048](https://huggingface.co/datasets/lance-format/coyo-ve-qwen3vl-2048),
revision `95efb82f4914f320e59cf286a06f23b7cfdb2112`. All 25 source files
were verified against their Hub LFS SHA256 or Git blob identities. All corpus
rows are retained, with their original vectors and identifiers. A separate
shallow clone holds the index. Every corpus row occurs in exactly one partition.

The seed-2254 split uses all 25,000 public queries: 12,500 calibration and
12,500 evaluation queries, with no duplicate evaluation vectors found. Before
search, the first two calibration queries and first 128 evaluation queries
were frozen. The compact query dataset preserves source positions and public
query IDs; local positions 0/1 are calibration and 2–129 are evaluation.

Exact float64 cosine top-100000 was recomputed across the complete corpus for
all 130 selected queries, normalizing both query and corpus vectors. The
blockwise oracle passed nine checks against independent exhaustive distances,
covering three metrics and k smaller/larger than a block and equal to corpus size.
Cutoff ties use NumPy argpartition; retained equal scores sort by corpus position.
Alternative tied IDs count as strict-ID misses.

Cross-checking the published top-1000 gives identical sets for 129 queries.
Query `q022018` shares 999 of 1000 IDs: public row 3074947 is float64 rank 1001,
with similarity 0.6507627995800611; row 12963405 is rank 1000, with similarity
0.6507632116341938. The difference is about 4.12e-7. The largest published
score discrepancy over the 130,000 pairs is 2.39214179e-6.
Direct float64 evaluation of both boundary vectors agrees with the frozen
oracle. This is consistent with numerical sensitivity near the cutoff;
the evaluation uses the original float64 scores without correction or exclusions.

The dataset is CC-BY-4.0. Attribution: Qdrant (Coyo-VE and Supernova),
MVP-Lab / LLaVA-OneVision-1.5, Kakao Brain (COYO-700M), the Qwen team
(Qwen3-VL-Embedding-2B), and ImageNet authors; see the
[pinned dataset README](https://huggingface.co/datasets/lance-format/coyo-ve-qwen3vl-2048/blob/95efb82f4914f320e59cf286a06f23b7cfdb2112/README.md).

## Environment and validation

- AWS r8i.8xlarge, 32 vCPUs, 247 GiB visible memory; Linux x86-64.
- Preserved `release-with-debug` candidate and baseline runtimes from the
  [five-corpus effort study](SEARCH_EFFORT_RESULTS.md); production code is unchanged.
- CPU affinity 0–15, 16 Lance/Rayon threads, one BLAS/OMP thread,
  `query_parallelism=1`, normal approximation, no refinement or filter.
- Fresh IVF training with native defaults, five bits and a saved fast-rotation
  model; centroids, complete membership and index-file hashes are frozen.
- First 64 queries timed serially; the other 64 run afterward with eight
  workers for recall only. Their timings are excluded from latency summaries.
- All **4,608** records return exactly k distinct valid IDs with zero measured
  storage bytes. Independent set intersections reproduce every recall value.
  All effort-1 queries search every partition. Query coverage, disjoint splits
  and pre/post source, input, model, index and binary identities pass auditing.
- The calibration smoke run passes. A deliberately duplicated returned ID
  makes auditing fail; restoring it makes the audit pass again. Omitted effort
  and explicit 0.5 agree on both calibration queries at all six k values.
- Repository Python lint/type checks and the changed benchmark scripts'
  Ruff lint/format checks pass. The implementation's Rust, Python and Java
  validation is recorded in the earlier report; this follow-up changes the
  measurement harness and report only.
- Candidate native SHA256: `ce1320d527deb815a94a0c6dba070dd87289b2696991f31fa4b256f9b7c05a52`.
- Baseline native SHA256: `7cc8a98ef3dd76532b1d895e12757d2184f6bbe29c566b9d982586eaa00ba1ef`.

[Complete identities](coyo-search-effort-identities.json) include the source
manifest, index hashes, query mapping, public-truth cross-check and runtime.
Raw returned IDs, per-query CSVs, exact truth, queries, model, membership,
source snapshots and both runtimes are preserved in the evidence archive.
The complete original corpus and frozen index remain on the retained EBS volume.
