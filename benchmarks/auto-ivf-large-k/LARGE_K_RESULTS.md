# Experimental Auto IVF profiles at k=10,000 and k=100,000

Six metric/k profiles were calibrated on the original five full corpora. All
**10 held-out native corpus/k groups** exceed 95% mean strict-ID recall,
with a range of **95.8369%–96.3323%**. The independent returned-ID audit passes
all 70 native policy groups and all matched baseline comparisons.

The current default policy uses the k=10000 profile for k=1001..10000 and the
k=100000 profile for k=10001..100000, across IVF index types, vector types and
refinement factors. This report records the original experiment at two exact k
values. It does not establish recall for intermediate k values, quantized or
HNSW indices, other vector types, refinement, or filtered queries. See
[LARGE_K_PROTOCOL.md](LARGE_K_PROTOCOL.md) for the frozen contract.

## Frozen parameters

| Metric | k | Margin | Initial floor | Initial cap |
| --- | --- | --- | --- | --- |
| L2 | 10000 | 0.57 | 98 partitions | 238 partitions |
| L2 | 100000 | 0.75 | 463 partitions | 704 partitions |
| Cosine | 10000 | 0.6475 | 104 partitions | 447 partitions |
| Cosine | 100000 | 0.58 | 738 partitions | 958 partitions |
| Dot | 10000 | 0.0825 | 517 partitions | 1824 partitions |
| Dot | 100000 | 0.21 | 1009 partitions | 2528 partitions |

Parameters minimize equally weighted mean initial partitions within the preregistered search grid, subject to 96% mean routing recall on every calibration corpus of the metric. They were frozen before held-out evaluation. Floors and caps constrain initial probing; caller minimums and count-based late expansion retain their existing semantics.

## Native held-out results

| Corpus | k | Queries | Mean recall | Mean latency | Mean partitions | Mean scanned rows |
| --- | --- | --- | --- | --- | --- | --- |
| DINO | 10000 | 3072 | 95.916% | 200.518 ms | 149.0 partitions | 621647 rows |
| DINO | 100000 | 3072 | 95.837% | 791.483 ms | 542.5 partitions | 2261109 rows |
| LAION | 10000 | 3069 | 96.332% | 233.755 ms | 234.2 partitions | 973396 rows |
| LAION | 100000 | 3069 | 95.917% | 826.501 ms | 760.2 partitions | 3182094 rows |
| FineWeb | 10000 | 3200 | 96.002% | 338.508 ms | 352.9 partitions | 1439018 rows |
| FineWeb | 100000 | 3200 | 96.042% | 844.667 ms | 802.6 partitions | 3288344 rows |
| Wiki-Cohere | 10000 | 2500 | 96.069% | 1275.170 ms | 1262.2 partitions | 5567924 rows |
| Wiki-Cohere | 100000 | 2500 | 96.112% | 2650.500 ms | 2528.0 partitions | 11148462 rows |
| DPR | 10000 | 1805 | 95.879% | 481.709 ms | 517.0 partitions | 2069213 rows |
| DPR | 100000 | 1805 | 95.948% | 1139.039 ms | 1119.5 partitions | 4510527 rows |

Strict-ID recall uses every held-out query. Latency uses the first 512 serial queries. Audit target misses: none.

## Actual baseline comparison

These rows compare exactly the same 32 queries in the original main binary and the experimental binary, with interleaved serial timing. Recall differs, so the ratios describe policy tradeoffs at their observed recall. They are historical measurements of the recorded experimental patch, not new latency measurements of the current PR head. The measured profiles are now used by the default policy through k=100000.

Both binaries use the same r8i.8xlarge, release-with-debug profile, CPU affinity 0-15, 16 Lance/Rayon threads, one BLAS/OMP thread, query_parallelism=1, frozen IVF_FLAT indices, and prewarmed caches. Every query reports zero storage bytes read. DINO/LAION/FineWeb each contain 10M rows, Wiki-Cohere 35M, and DPR 21,015,300. Latency quantiles describe one pass, not repeated-run confidence intervals.

| Scenario / metric | Baseline | Experimental build | Benefit | Recall: baseline / experimental |
| --- | --- | --- | --- | --- |
| DINO, k=10000; mean latency (lower is better) | 3028.950 ms | 197.945 ms | 15.30x speedup | 100.000% / 95.736% |
| DINO, k=100000; mean latency (lower is better) | 3102.301 ms | 785.137 ms | 3.95x speedup | 99.999% / 95.821% |
| LAION, k=10000; mean latency (lower is better) | 2266.461 ms | 266.246 ms | 8.51x speedup | 100.000% / 96.254% |
| LAION, k=100000; mean latency (lower is better) | 2369.685 ms | 832.997 ms | 2.84x speedup | 100.000% / 95.474% |
| FineWeb, k=10000; mean latency (lower is better) | 2283.059 ms | 341.179 ms | 6.69x speedup | 100.000% / 96.120% |
| FineWeb, k=100000; mean latency (lower is better) | 2380.253 ms | 872.672 ms | 2.73x speedup | 100.000% / 96.129% |
| Wiki-Cohere, k=10000; mean latency (lower is better) | 7946.042 ms | 1251.870 ms | 6.35x speedup | 100.000% / 95.605% |
| Wiki-Cohere, k=100000; mean latency (lower is better) | 8041.751 ms | 2652.213 ms | 3.03x speedup | 100.000% / 95.606% |

| Scenario / metric | Baseline | Experimental build | Cost | Recall: baseline / experimental |
| --- | --- | --- | --- | --- |
| DPR, k=10000; mean latency (lower is better) | 11.310 ms | 483.898 ms | 42.79x latency | 21.268% / 96.332% |
| DPR, k=100000; mean latency (lower is better) | 79.851 ms | 1119.905 ms | 14.02x latency | 29.871% / 95.892% |

## Fixed-budget comparison

Recall uses all held-out queries and latency uses the same first 512 serial queries. Nearby fixed budgets were selected from calibration data. The complete CSV also contains the per-corpus tuned reference. Fixed policies can return fewer than k rows; the returned-count distributions are retained in the CSV.

| Corpus | k | Auto: recall / latency | fixed20: recall / latency | Nearby fixed: recall / latency |
| --- | --- | --- | --- | --- |
| DINO | 10000 | 95.916% / 200.518 ms | 74.020% / 33.975 ms | fixed128: 94.504% / 172.322 ms; fixed192: 96.556% / 254.557 ms; fixed256: 97.630% / 336.339 ms |
| DINO | 100000 | 95.837% / 791.483 ms | 37.325% / 67.112 ms | fixed384: 92.741% / 580.590 ms; fixed512: 95.160% / 746.707 ms; fixed768: 97.567% / 1075.599 ms |
| LAION | 10000 | 96.332% / 233.755 ms | 69.749% / 26.964 ms | fixed192: 94.517% / 188.847 ms; fixed256: 96.089% / 249.500 ms; fixed384: 97.760% / 371.281 ms |
| LAION | 100000 | 95.917% / 826.501 ms | 32.087% / 59.904 ms | fixed512: 91.925% / 583.331 ms; fixed768: 95.772% / 835.061 ms; fixed1024: 97.727% / 1083.266 ms |
| FineWeb | 10000 | 96.002% / 338.508 ms | 62.816% / 27.224 ms | fixed256: 93.998% / 246.912 ms; fixed384: 96.292% / 366.911 ms; fixed512: 97.519% / 487.381 ms |
| FineWeb | 100000 | 96.042% / 844.667 ms | 32.073% / 59.480 ms | fixed512: 91.681% / 571.085 ms; fixed768: 95.532% / 816.156 ms; fixed1024: 97.513% / 1057.492 ms |
| Wiki-Cohere | 10000 | 96.069% / 1275.170 ms | 46.720% / 30.854 ms | fixed1024: 94.718% / 1030.939 ms; fixed1536: 96.657% / 1544.036 ms; fixed2048: 97.744% / 2056.235 ms |
| Wiki-Cohere | 100000 | 96.112% / 2650.500 ms | 24.719% / 66.935 ms | fixed2048: 94.701% / 2166.331 ms; fixed3072: 97.223% / 3195.613 ms; fixed4096: 98.541% / 4212.727 ms |
| DPR | 10000 | 95.879% / 481.709 ms | 52.714% / 26.876 ms | fixed384: 94.125% / 359.300 ms; fixed512: 95.829% / 477.143 ms; fixed768: 97.599% / 712.884 ms |
| DPR | 100000 | 95.948% / 1139.039 ms | 26.437% / 56.404 ms | fixed768: 92.533% / 802.380 ms; fixed1024: 95.098% / 1040.283 ms; fixed1536: 97.677% / 1514.422 ms |

Auto does not dominate every fixed budget that passes the 95% held-out target. For example, fixed512 on DINO at k=100000 and fixed1024 on DPR at k=100000 have lower mean latency and lower recall than Auto, while still exceeding 95%. The calibration objective minimizes initial partitions under its 96% calibration constraint; it is not a direct native-latency optimization.

## Complete distributions

[large-k-native-results.csv](large-k-native-results.csv) contains every native arm and its mean/p50/p90/p95/p99/max latency, partitions, scanned rows, and returned counts. [large-k-baseline-comparisons.csv](large-k-baseline-comparisons.csv) contains the matched actual-binary subset. [large-k-routing-simulation.csv](large-k-routing-simulation.csv) contains the calibration and held-out routing checks; these are simulations, not native timings.

Scanned rows are the native index_comparisons counter, which charges the storage length of every searched partition before filtering. They need not equal the number of evaluated distances. Storage bytes are audited separately.

## Reproduction and validation

- Baseline source: `567e322b6644a24cef98a57b565807eea71805f3`.
- Experimental source: `0bcc82431521017fc0afe2772a214d407a44c9be` plus
  [large-k-experimental.patch](large-k-experimental.patch), generated from
  [large-k-calibration-frozen.json](large-k-calibration-frozen.json) by
  [build_large_patch.py](build_large_patch.py), then formatted with Cargo.
  The runtime source is unchanged in the documentation follow-up `12def694f`.
- Experimental native SHA256:
  `f58cdda67498a06efb798dacc9f2e32e75f7bb003af643a0344ec9056af9b30a`.
- Baseline native SHA256:
  `9c5b08417d6dfe7b319039b80b02a876b1e6d8078fe2820c46319150f359bcab`.
- The host, compiler, libraries, frozen index versions, and query splits are the
  same as the [original study](RESULTS.md#reproduction-contract): Intel Xeon
  6975P-C, 32 vCPU / 256 GiB RAM, Rust 1.98.1, Python 3.12.3, NumPy 2.5.1,
  and Lance 14.0.0-beta.6. Both binaries use `release-with-debug` with the
  repository Haswell target settings. All three lockfiles match the measured
  source. No compilation or other CPU-intensive preparation overlaps timing.
- Exact float64 ground truth was extended to the top 100,000 for every query.
  The first 5,000 distances match the previous oracle with zero maximum
  difference on every corpus. Equal-distance cutoff IDs remain subject to the
  recorded oracle's tie ordering; all strict-ID differences count as misses.
- All held-out queries contribute recall and scan counts. The first 512 are
  timed serially, and the first 32 also run against the actual baseline binary
  in interleaved order. Remaining queries use eight workers for recall only.
  Scanner construction through `to_table()` is timed, returning `_rowid` and
  `_distance`; ID mapping, recall calculations, and baseline IPC are excluded.
- Validation passed 358 Rust KNN tests, 20 Python Auto-probe tests on the
  development extension, the same 20 on the optimized extension, workspace
  Clippy with warnings denied, Cargo formatting, and `uv run make lint`.
  [large-k-run-integrity.json](large-k-run-integrity.json) records source,
  lockfile, native-library, harness, calibration, build, split, and audit hashes.

[README.md](README.md#additional-k10000-and-k100000-experiment) gives the build
and replay commands. The full oracle arrays, returned IDs, query records,
source snapshots and validation logs are retained in the campaign evidence.

## Numerical differences

The audit finds 213 native-counter differences relative to initial routing.
Of these, 64 are the original DPR fallback's expected count-based expansion
to obtain k results. After accounting for expansion, 149 differences remain:
20 change the partition count by one, and 129 keep the count but change the
sum of partition sizes.

For all 20 count differences, the disputed centroid lies within 8.683 Float32
representable spacings of the frozen gap threshold. Each of the 129 row-count
differences matches a boundary-centroid exchange within eight Float32 spacings.
These checks are consistent with distance-reduction and centroid-ordering
rounding near a decision boundary; the exact native centroid order was not
captured. Native counts remain authoritative.

The largest Auto recall difference is one LAION query at k=100000:
96.399% native versus 98.782% simulated routing. All 2,383 simulator-reachable
truth IDs absent from the native result tie with its worst returned source-vector
distance within 1.2e-16. The three largest distinct-query shortfalls on each of
DINO and LAION likewise contain only cutoff ties in the inspected cases.

The inspected FineWeb and Wiki shortfalls have different searched-partition
counts or memberships. Their missing reachable neighbors have strictly better
source-vector distances and all belong to the last partition selected by the
simulator. These observations support routing differences at the selection
boundary; the missing neighbors are not final-score ties.
DPR has no positive Auto recall shortfall against simulated initial routing.
None of these observations changes the strict-ID scores.

[large-k-numerical-audit.json](large-k-numerical-audit.json) contains all 213
counter cases, the threshold and boundary checks, the selected source-vector
distance comparisons, and missing-neighbor partition ranks.
