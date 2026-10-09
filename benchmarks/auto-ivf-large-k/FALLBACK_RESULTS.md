# One fallback threshold per metric, independent of k

The frozen gap margins are **L2 1.3116194, cosine 1.5733937, and dot
0.24105163**. They meet the 96% calibration-routing target at every integer
k=1..100000, using one constant per metric and no learned floor or cap.
Native evaluation passes **30/30 corpus/k mean-recall gates**;
the measured mean strict-ID recall range is **96.1122%–100.0000%**.

The existing multiplier heuristic has no common feasible dot constant for the
two tested dot corpora. A normalized gap resolves that sign conflict, but a
single constant can approach exhaustive scans and has a large small-k cost.
These measurements support the constants under the stated recall contract;
they do not establish a universally efficient fallback or justify enabling it
for every unsupported index/query type.

This is an isolated experiment. The PR's production runtime continues to enable
calibrated profiles only through k=1000. See
[FALLBACK_PROTOCOL.md](FALLBACK_PROTOCOL.md) for the frozen protocol.

## Frozen constants

| Metric | Gap margin | f32 bits | Original-form multiplier |
| --- | --- | --- | --- |
| l2 | 1.3116194009780884 | 0x3fa7e325 | 2.311619281768799 |
| cosine | 1.573393702507019 | 0x3fc964f7 | 2.5733935832977295 |
| dot | 0.24105162918567657 | 0x3e76d63d | No feasible constant |

The gap rule is `d_i - d_0 <= margin * scale`, using `scale=d_0` for L2/cosine and `scale=abs(1-d_0)` for dot. L2 and normalized-cosine routing use squared L2 distances; dot uses `1-inner_product`. The constants are f32 and the gap comparison uses f64 arithmetic, matching the native implementation. The original multiplier uses f32 multiplication, so its rounded constant is not necessarily exactly `1 + margin`.

Each metric has one constant for every k, no learned floor or cap, and caller minimum 1. Caller bounds and count-based late expansion still apply. Calibration chooses the smallest representable nonnegative f32 margin with at least 96% mean initial routing recall on every calibration corpus at every integer k=1..100000, using the frozen prepared centroid distances. Its preceding f32 value fails at least one calibration constraint. This minimum applies to the recorded routing model; native centroid-distance reductions can differ slightly and are evaluated separately. Constants were frozen before held-out evaluation; the held-out acceptance target is 95%. This establishes a finite benchmark range, not a guarantee for arbitrary k, corpora, indices, or filters.

| Corpus | Metric | Per-corpus minimum margin | Shared metric margin |
| --- | --- | --- | --- |
| DINO | l2 | 1.3116194009780884 | 1.3116194009780884 |
| LAION | cosine | 1.573393702507019 | 1.573393702507019 |
| FineWeb | cosine | 0.8107466101646423 | 1.573393702507019 |
| Wiki-Cohere | dot | 0.09923002123832703 | 0.24105162918567657 |
| DPR | dot | 0.24105162918567657 | 0.24105162918567657 |

The shared metric constant is the maximum of the per-corpus minima. LAION determines the cosine constant and DPR determines the dot constant. These are minima for the stated recall/partition-count contract, not measured global optima for latency.

## Full held-out routing evaluation

| Corpus | Queries | Minimum mean initial recall | k at minimum | k values below 95% |
| --- | --- | --- | --- | --- |
| DINO | 3072 | 96.634275% | 100000 | 0 |
| LAION | 3069 | 96.032965% | 100000 | 0 |
| FineWeb | 3200 | 99.938857% | 100000 | 0 |
| Wiki-Cohere | 2500 | 99.999984% | 4963 | 0 |
| DPR | 1805 | 96.078877% | 100000 | 0 |

Every integer k=1..100000 was evaluated from the exact ground-truth prefixes. These values describe simulated initial routing, before count-based late expansion. They are not native-query measurements.

## Existing fallback quality

The current fallback uses multipliers 0.6 at k=1, 7 at k=2..10, and 81 above k=10. For positive nearest distances, 0.6 usually selects no centroid beyond the caller minimum; for negative dot distances, multipliers above 1 have that problem instead. Conversely, the large factors approach exhaustive scans on the positive-distance corpora.

| Corpus | k=1 recall | k=10 recall | k=1000 recall | k=100000 recall |
| --- | --- | --- | --- | --- |
| DINO | 54.134% | 100.000% | 100.000% | 100.000% |
| LAION | 58.912% | 100.000% | 100.000% | 99.973% |
| FineWeb | 48.656% | 100.000% | 100.000% | 100.000% |
| Wiki-Cohere | 44.280% | 100.000% | 100.000% | 100.000% |
| DPR | 100.000% | 27.889% | 18.040% | 30.692% |

This table is held-out routing simulation including unfiltered count-based late expansion. Returning k rows does not establish high recall.

A single signed multiplier is infeasible for the two dot corpora under this contract. The calibration proof splits all finite factors into factor<1 and factor>=1. For the former, positive-nearest-distance queries can at best retain nearest-distance ties; for the latter, the same bound applies to negative-nearest-distance queries. The proof generously grants every other query all partitions and includes count-based late expansion. At k=2000, even this optimistic recall is about 16.4% on Wiki-Cohere for factor<1 and on DPR for factor>=1. A larger parameter search cannot remove this sign conflict.

## Cost of a single threshold

| Corpus | k | Calibrated profile scanned rows | Constant fallback scanned rows | Cost | Recall: profile / constant |
| --- | --- | --- | --- | --- | --- |
| DINO | 1 | 39693 rows | 6502771 rows | 163.83x rows | 95.475% / 100.000% |
| DINO | 1000 | 213066 rows | 6502771 rows | 30.52x rows | 96.116% / 99.956% |
| DINO | 100000 | 2261109 rows | 6503920 rows | 2.88x rows | 95.837% / 97.029% |
| LAION | 1 | 55763 rows | 7322258 rows | 131.31x rows | 96.188% / 100.000% |
| LAION | 1000 | 390129 rows | 7322258 rows | 18.77x rows | 97.073% / 99.986% |
| LAION | 100000 | 3182093 rows | 7324173 rows | 2.30x rows | 95.918% / 96.752% |
| FineWeb | 1 | 94662 rows | 9866329 rows | 104.23x rows | 94.250% / 100.000% |
| FineWeb | 1000 | 637441 rows | 9866329 rows | 15.48x rows | 95.999% / 99.999% |
| FineWeb | 100000 | 3288344 rows | 9866351 rows | 3.00x rows | 96.042% / 99.947% |
| Wiki-Cohere | 1 | 489486 rows | 34995346 rows | 71.49x rows | 94.680% / 100.000% |
| Wiki-Cohere | 1000 | 3179434 rows | 34995346 rows | 11.01x rows | 96.021% / 100.000% |
| Wiki-Cohere | 100000 | 11148459 rows | 34995346 rows | 3.14x rows | 96.112% / 100.000% |
| DPR | 1 | 375158 rows | 5255335 rows | 14.01x rows | 96.620% / 99.945% |
| DPR | 1000 | 1111012 rows | 5255335 rows | 4.73x rows | 95.926% / 99.704% |
| DPR | 100000 | 4510522 rows | 5255335 rows | 1.17x rows | 95.948% / 96.079% |

These are full held-out routing simulations with count-based late expansion, using the same queries, indices, and oracle. The reference uses the calibrated profiles implemented in this PR through k=1000 and the separate experimental profile at k=100000. Those profiles also have learned floors and caps, so this comparison does not isolate the effect of k dependence. The one-constant fallback must handle the hardest k/corpus constraint and can substantially overscan smaller queries. Wiki-Cohere and FineWeb are nearly exhaustive. These row-count ratios are not latency speedups.

## Native held-out validation

| Corpus | k | Mean recall | Mean latency | Mean partitions | Mean scanned rows |
| --- | --- | --- | --- | --- | --- |
| DINO | 1 | 100.000000% | 1826.825 ms | 1558.8 partitions | 6423778 rows |
| DINO | 10 | 100.000000% | 1819.770 ms | 1558.8 partitions | 6423778 rows |
| DINO | 100 | 99.996094% | 1814.421 ms | 1558.8 partitions | 6423778 rows |
| DINO | 1000 | 99.981055% | 1829.604 ms | 1558.8 partitions | 6423778 rows |
| DINO | 10000 | 99.587695% | 1837.947 ms | 1558.8 partitions | 6423778 rows |
| DINO | 100000 | 96.747791% | 1923.080 ms | 1559.0 partitions | 6424344 rows |
| LAION | 1 | 99.804688% | 1683.777 ms | 1814.4 partitions | 7503517 rows |
| LAION | 10 | 99.902344% | 1687.223 ms | 1814.4 partitions | 7503517 rows |
| LAION | 100 | 99.986328% | 1688.748 ms | 1814.4 partitions | 7503517 rows |
| LAION | 1000 | 99.960742% | 1688.285 ms | 1814.4 partitions | 7503517 rows |
| LAION | 10000 | 99.481895% | 1705.774 ms | 1814.4 partitions | 7503563 rows |
| LAION | 100000 | 96.883365% | 1782.442 ms | 1814.8 partitions | 7505071 rows |
| FineWeb | 1 | 100.000000% | 2224.422 ms | 2412.0 partitions | 9893402 rows |
| FineWeb | 10 | 99.902344% | 2237.668 ms | 2412.0 partitions | 9893402 rows |
| FineWeb | 100 | 99.996094% | 2250.711 ms | 2412.0 partitions | 9893402 rows |
| FineWeb | 1000 | 99.999609% | 2253.574 ms | 2412.0 partitions | 9893402 rows |
| FineWeb | 10000 | 99.997051% | 2262.075 ms | 2412.0 partitions | 9893402 rows |
| FineWeb | 100000 | 99.983232% | 2349.637 ms | 2412.0 partitions | 9893402 rows |
| Wiki-Cohere | 1 | 100.000000% | 7802.648 ms | 8496.1 partitions | 34996742 rows |
| Wiki-Cohere | 10 | 100.000000% | 7779.211 ms | 8496.1 partitions | 34996742 rows |
| Wiki-Cohere | 100 | 100.000000% | 7802.516 ms | 8496.1 partitions | 34996742 rows |
| Wiki-Cohere | 1000 | 99.999805% | 7790.934 ms | 8496.1 partitions | 34996742 rows |
| Wiki-Cohere | 10000 | 99.999688% | 7808.928 ms | 8496.1 partitions | 34996742 rows |
| Wiki-Cohere | 100000 | 99.999764% | 7923.037 ms | 8496.1 partitions | 34996742 rows |
| DPR | 1 | 100.000000% | 1207.978 ms | 1339.7 partitions | 5417334 rows |
| DPR | 10 | 99.980469% | 1209.674 ms | 1339.7 partitions | 5417334 rows |
| DPR | 100 | 99.839844% | 1207.898 ms | 1339.7 partitions | 5417334 rows |
| DPR | 1000 | 99.671680% | 1210.176 ms | 1339.7 partitions | 5417334 rows |
| DPR | 10000 | 99.064121% | 1209.860 ms | 1339.7 partitions | 5417334 rows |
| DPR | 100000 | 96.112197% | 1320.832 ms | 1339.7 partitions | 5417334 rows |

Recall and counters use the first 512 held-out queries; latency uses the first 128, run serially. Strict-ID recall is independently recomputed from saved returned IDs. Native target misses: none.

## Actual original-fallback comparison

The same first 32 held-out queries run in interleaved order against the recorded PR binary and experimental fallback binary. Both explicitly set maximum_nprobes to the full index partition count, selecting the fallback path even at small k. The candidate uses the frozen gap constants. L2/cosine original-form fitted multipliers are routing-only experiments, not separately timed native binaries.

Both binaries use the same r8i.8xlarge, repository release-with-debug profile, CPU affinity 0-15, 16 Lance/Rayon threads, one BLAS/OMP thread, query_parallelism=1, frozen IVF_FLAT indices, and prewarmed caches. Every measured query reports zero storage bytes read. Latency statistics describe one pass, not confidence intervals from repeated runs; ratios near 1x do not establish a performance improvement. Differences in recall are shown alongside timing ratios.

| Scenario / metric | Baseline | Experimental build | Benefit | Recall: baseline / experimental |
| --- | --- | --- | --- | --- |
| DINO, k=10; mean latency (lower is better) | 3039.093 ms | 1978.734 ms | 1.54x speedup | 100.000000% / 100.000000% |
| DINO, k=100; mean latency (lower is better) | 3036.439 ms | 1967.806 ms | 1.54x speedup | 100.000000% / 100.000000% |
| DINO, k=1000; mean latency (lower is better) | 3041.405 ms | 1978.964 ms | 1.54x speedup | 100.000000% / 99.965625% |
| DINO, k=10000; mean latency (lower is better) | 3034.173 ms | 1991.741 ms | 1.52x speedup | 99.999688% / 99.784063% |
| DINO, k=100000; mean latency (lower is better) | 3117.243 ms | 2076.435 ms | 1.50x speedup | 99.999437% / 96.775094% |
| LAION, k=10; mean latency (lower is better) | 2288.166 ms | 1724.996 ms | 1.33x speedup | 100.000000% / 100.000000% |
| LAION, k=100; mean latency (lower is better) | 2294.860 ms | 1718.298 ms | 1.34x speedup | 100.000000% / 100.000000% |
| LAION, k=1000; mean latency (lower is better) | 2289.760 ms | 1723.711 ms | 1.33x speedup | 100.000000% / 99.996875% |
| LAION, k=10000; mean latency (lower is better) | 2299.543 ms | 1737.032 ms | 1.32x speedup | 100.000000% / 99.857813% |
| LAION, k=100000; mean latency (lower is better) | 2397.259 ms | 1822.375 ms | 1.32x speedup | 99.999969% / 97.400375% |
| FineWeb, k=10; mean latency (lower is better) | 2268.770 ms | 2201.645 ms | 1.03x speedup | 100.000000% / 100.000000% |
| FineWeb, k=100; mean latency (lower is better) | 2264.863 ms | 2211.940 ms | 1.02x speedup | 100.000000% / 100.000000% |
| FineWeb, k=1000; mean latency (lower is better) | 2278.881 ms | 2217.750 ms | 1.03x speedup | 99.996875% / 99.996875% |
| FineWeb, k=10000; mean latency (lower is better) | 2285.802 ms | 2230.194 ms | 1.02x speedup | 99.999688% / 99.974063% |
| FineWeb, k=100000; mean latency (lower is better) | 2393.221 ms | 2312.288 ms | 1.04x speedup | 99.999844% / 99.832281% |
| Wiki-Cohere, k=100; mean latency (lower is better) | 7809.198 ms | 7766.012 ms | 1.01x speedup | 100.000000% / 100.000000% |
| Wiki-Cohere, k=1000; mean latency (lower is better) | 7812.024 ms | 7780.008 ms | 1.00x speedup | 100.000000% / 100.000000% |
| Wiki-Cohere, k=10000; mean latency (lower is better) | 7836.398 ms | 7816.952 ms | 1.00x speedup | 100.000000% / 100.000000% |
| DPR, k=1; mean latency (lower is better) | 3111.435 ms | 1098.487 ms | 2.83x speedup | 100.000000% / 100.000000% |

| Scenario / metric | Baseline | Experimental build | Cost | Recall: baseline / experimental |
| --- | --- | --- | --- | --- |
| DINO, k=1; mean latency (lower is better) | 4.792 ms | 1976.615 ms | 412.45x latency | 53.125000% / 100.000000% |
| LAION, k=1; mean latency (lower is better) | 4.401 ms | 1719.464 ms | 390.67x latency | 50.000000% / 100.000000% |
| FineWeb, k=1; mean latency (lower is better) | 4.439 ms | 2190.713 ms | 493.53x latency | 46.875000% / 100.000000% |
| Wiki-Cohere, k=1; mean latency (lower is better) | 5.998 ms | 7772.028 ms | 1295.83x latency | 46.875000% / 100.000000% |
| Wiki-Cohere, k=10; mean latency (lower is better) | 7772.630 ms | 7789.429 ms | 1.00x latency | 100.000000% / 100.000000% |
| Wiki-Cohere, k=100000; mean latency (lower is better) | 7908.796 ms | 7924.383 ms | 1.00x latency | 99.999687% / 99.999656% |
| DPR, k=10; mean latency (lower is better) | 4.754 ms | 1097.036 ms | 230.74x latency | 26.250000% / 100.000000% |
| DPR, k=100; mean latency (lower is better) | 4.911 ms | 1099.442 ms | 223.89x latency | 22.250000% / 99.812500% |
| DPR, k=1000; mean latency (lower is better) | 5.084 ms | 1099.900 ms | 216.36x latency | 17.300000% / 99.603125% |
| DPR, k=10000; mean latency (lower is better) | 10.508 ms | 1102.759 ms | 104.95x latency | 21.267500% / 98.739375% |
| DPR, k=100000; mean latency (lower is better) | 78.070 ms | 1209.725 ms | 15.50x latency | 29.870937% / 95.271406% |

## Detailed results and scope

[fallback-native-results.csv](fallback-native-results.csv) contains recall, returned-count, latency, partition and row-count distributions, including minimum recall. Recall columns are fractions, and the 95% target applies to the corpus/k mean rather than each query. [fallback-baseline-comparisons.csv](fallback-baseline-comparisons.csv) contains matched baseline distributions. [fallback-routing-simulation.csv](fallback-routing-simulation.csv) contains every reported family/anchor for both splits, including calibrated profile references. [fallback-parameters.json](fallback-parameters.json) records exact constants and the complete held-out k-range minima.

Scanned rows are the native index_comparisons counter and charge the storage lengths of searched partitions. Native tests isolate routing on unfiltered Float32 IVF_FLAT. Quantized/HNSW accuracy, Float16 accuracy, multivectors, refinement, selective filters, Hamming, and nonfinite-query accuracy were not established. Historical readers retain their frozen original behavior. No production defaults were changed by this experiment.

## Reproduction and validation

- Baseline runtime source: `0bcc82431521017fc0afe2772a214d407a44c9be`;
  its runtime source is unchanged in `db4809f41`.
- Experimental source: that PR runtime plus
  [fallback-experimental.patch](fallback-experimental.patch), generated by
  [build_fallback_patch.py](build_fallback_patch.py) from
  [fallback-calibration-frozen.json](fallback-calibration-frozen.json), then
  formatted with Cargo. Earlier experimental and original-main runtimes are
  retained separately.
- Baseline native SHA256: `17bd3bb409c92e4a5e83ea4b4573a2e49c2195fc074dd86ebb4936fc13f706d6`.
- Experimental native SHA256: `636a1dcffaccc441404ce71037dcb49c716dc3757dc3652e77a54e3a0995ef5f`.
- Frozen calibration SHA256: `313042ddeb05fc5fc09cf555150a71922485e28d66d1b51077b620f8d9e85f58`.
- The same five original corpora, index versions and query split are reused.
  DINO, LAION and FineWeb have 10M rows each; Wiki-Cohere has 35M and DPR
  21,015,300. The oracle contains exact float64 top-100,000 neighbor prefixes.
  Repeated query vectors do not cross calibration/evaluation splits.
- Environment: AWS r8i.8xlarge, Intel Xeon 6975P-C, 32 vCPU / 256 GiB RAM,
  Rust 1.98.1, Python 3.12.3, NumPy 2.5.1 and Lance 14.0.0-beta.6. Native
  binaries use the repository release-with-debug profile and Haswell settings;
  all three lockfiles match the measured source. Timing affinity and thread
  counts are recorded above. Compilation does not overlap native timing.
- Validation passed 17 independent calibration tests, 353 Rust KNN tests,
  20 Python Auto-probe tests on the development extension, the same 20 on the
  optimized extension, workspace Clippy with warnings denied, Cargo formatting,
  and `uv run make lint`. Four calibration smoke corpora across all six k
  values also passed an independent ID/counter audit before formal evaluation.
- Formal auditing verifies all 60 candidate/baseline groups, returned IDs and
  counts, exact query coverage, split separation, input and binary hashes,
  zero storage bytes read, and timing distributions. The 95% quality gate
  passes all 30 candidate groups.

[fallback-run-integrity.json](fallback-run-integrity.json) records the source,
library, harness, input, native-identity and audit hashes.
[README.md](README.md#one-fallback-threshold-per-metric) gives the replay
commands. Full oracle arrays, returned IDs, records, source snapshots and
validation logs are retained in the campaign evidence.

## Numerical checks

The independent audit records **12** native partition/row-count
differences from the simulated final routing budgets. The numerical audit
inspects the three largest absolute routing-recall differences for every
corpus/k/policy, and every discrepant query in any candidate group that misses
the primary target. It compares returned and missing neighbor source-vector
scores in float64 and records their partition ranks. Native strict-ID recall
and native counters remain authoritative; these diagnostics do not alter them.

[fallback-numerical-audit.json](fallback-numerical-audit.json) contains all
counter differences, boundary-exchange candidates and selected source-vector
checks. All 12 counter differences are the same two candidate queries repeated
at six k values: LAION query 3230 searches 1380 rather than 1381 partitions, and
DPR query 3482 searches 969 rather than 970. Their row counts match the
corresponding shorter simulated prefixes, and their native recall equals the
routing prediction at every measured k. The last included centroid in the
frozen model passes the gap comparison by only 6.81325e-8 for LAION and
1.13709e-7 for DPR, respectively 1.143 and 0.477 f32 distance spacings.
This is consistent with numerical boundary sensitivity. Exact native centroid
distances and ordering were not captured, so it does not prove the cause.

The selected recall-difference cases have matching routing counters and no
returned IDs outside the predicted partition set. The following counts sum the
selected corpus/k/policy cases, including repeated queries at different k;
they are not population-wide adjustments to recall. Distance gaps compare each
missing reachable true neighbor with the worst returned source-vector distance,
recomputed in float64 using the corpus metric.

| Corpus | Selected cases | Missing reachable IDs | Exact cutoff ties | Maximum absolute distance gap |
| --- | ---: | ---: | ---: | ---: |
| DINO | 15 | 21 | 21 | 0 |
| LAION | 17 | 2784 | 2774 | 1.935726e-8 |
| FineWeb | 18 | 18 | 3 | 1.230867e-7 |
| Wiki-Cohere | 10 | 13 | 0 | 1.603426e-6 |
| DPR | 6 | 6 | 4 | 6.887367e-7 |

For example, LAION query 4521 at k=1 has native strict-ID recall 0 despite an
exact float64 cutoff tie. Query 5977 at k=1000 has native recall 0.897 versus
routing recall 1: all 103 missing reachable IDs are within 1.12e-16 of the
returned cutoff, including 101 exact ties. These diagnostics explain why
routing coverage and strict-ID recall can differ without changing any of the
reported native scores. The acceptance target remains a corpus/k mean of 95%,
not a guarantee for every query.
