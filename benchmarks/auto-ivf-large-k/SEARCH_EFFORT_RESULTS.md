# IVF_RQ 5bit search effort: recall and latency

`search_effort` exposes a continuous initial IVF partition budget in [0, 1].
The default 0.5 preserves the existing Auto policy. Effort 0 starts at the
caller minimum; effort 1 starts at every available partition, subject to the
caller maximum. Intermediate values interpolate geometrically around Auto.
Later expansion to obtain enough candidates is unchanged.

This study measures the five representative settings 0, 0.25, 0.5, 0.75 and 1
on five frozen IVF_RQ 5bit indices. Every one of the **3,840** candidate-default
queries matches the old PR binary's returned IDs, partition counts and
comparison counts. Every effort-1 query searches all available partitions.
All **23,040** corpus/k/arm/query records pass the independent raw-ID audit.

## Recall and latency

Each cell is **mean strict-ID recall / mean serial latency**. Recall uses
128 held-out queries per corpus/k/arm; latency uses the first 64 of the same
queries in one serial pass, with rotating arm order. Each k uses the same
query IDs at every effort. These are operating points for these frozen
corpora, not recall or latency guarantees.
Higher recall and lower latency are better.

| Corpus | k | Effort 0 | Effort 0.25 | Effort 0.5 (Auto) | Effort 0.75 | Effort 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINO | 1 | 51.563% / 3.701 ms | 76.563% / 3.221 ms | 85.938% / 3.399 ms | 87.500% / 12.137 ms | 87.500% / 150.236 ms |
| DINO | 10 | 52.188% / 3.817 ms | 78.594% / 3.348 ms | 90.859% / 3.632 ms | 93.750% / 13.726 ms | 93.828% / 156.650 ms |
| DINO | 100 | 48.109% / 3.853 ms | 80.172% / 3.579 ms | 92.883% / 4.382 ms | 95.719% / 16.837 ms | 95.805% / 150.916 ms |
| DINO | 1000 | 38.428% / 4.387 ms | 76.992% / 4.733 ms | 94.487% / 8.039 ms | 96.768% / 26.573 ms | 96.904% / 154.232 ms |
| DINO | 10000 | 35.287% / 9.457 ms | 63.984% / 11.871 ms | 94.487% / 25.854 ms | 97.104% / 55.661 ms | 97.379% / 170.505 ms |
| DINO | 100000 | 41.052% / 66.362 ms | 41.738% / 68.028 ms | 94.530% / 180.345 ms | 96.962% / 228.815 ms | 97.404% / 313.288 ms |
| LAION | 1 | 55.469% / 3.452 ms | 82.813% / 3.107 ms | 88.281% / 3.429 ms | 96.094% / 11.613 ms | 96.094% / 124.177 ms |
| LAION | 10 | 50.469% / 3.429 ms | 83.125% / 3.119 ms | 95.234% / 4.176 ms | 97.109% / 15.090 ms | 97.266% / 126.974 ms |
| LAION | 100 | 45.594% / 3.747 ms | 78.891% / 3.478 ms | 95.203% / 5.590 ms | 98.078% / 19.220 ms | 98.172% / 128.299 ms |
| LAION | 1000 | 36.324% / 4.119 ms | 76.370% / 4.704 ms | 96.306% / 10.642 ms | 98.384% / 30.268 ms | 98.508% / 128.986 ms |
| LAION | 10000 | 31.685% / 8.592 ms | 62.345% / 11.516 ms | 95.662% / 31.400 ms | 98.347% / 59.105 ms | 98.631% / 143.026 ms |
| LAION | 100000 | 35.646% / 63.044 ms | 38.502% / 65.761 ms | 95.172% / 184.592 ms | 98.142% / 225.414 ms | 98.687% / 283.083 ms |
| FineWeb | 1 | 48.438% / 3.418 ms | 76.563% / 3.190 ms | 91.406% / 3.825 ms | 95.313% / 13.739 ms | 96.875% / 123.939 ms |
| FineWeb | 10 | 42.891% / 3.485 ms | 78.438% / 3.300 ms | 92.344% / 4.746 ms | 95.938% / 17.203 ms | 96.563% / 126.072 ms |
| FineWeb | 100 | 36.359% / 3.619 ms | 76.750% / 3.688 ms | 93.039% / 6.554 ms | 97.000% / 22.699 ms | 97.516% / 128.609 ms |
| FineWeb | 1000 | 27.284% / 4.071 ms | 73.642% / 4.955 ms | 94.700% / 13.478 ms | 97.460% / 35.416 ms | 97.845% / 129.516 ms |
| FineWeb | 10000 | 27.943% / 8.910 ms | 63.222% / 12.403 ms | 95.101% / 38.093 ms | 97.579% / 69.480 ms | 98.043% / 149.408 ms |
| FineWeb | 100000 | 36.869% / 63.762 ms | 39.819% / 65.203 ms | 95.354% / 202.590 ms | 97.664% / 247.456 ms | 98.210% / 309.104 ms |
| Wiki-Cohere | 1 | 46.875% / 5.279 ms | 82.813% / 5.098 ms | 93.750% / 10.448 ms | 97.656% / 57.972 ms | 97.656% / 441.921 ms |
| Wiki-Cohere | 10 | 33.359% / 5.397 ms | 69.375% / 5.625 ms | 90.547% / 19.218 ms | 94.844% / 84.881 ms | 95.078% / 452.830 ms |
| Wiki-Cohere | 100 | 27.773% / 5.550 ms | 67.266% / 6.586 ms | 91.805% / 31.169 ms | 95.109% / 110.921 ms | 95.523% / 453.285 ms |
| Wiki-Cohere | 1000 | 20.434% / 6.211 ms | 63.249% / 9.380 ms | 92.674% / 56.477 ms | 94.770% / 157.029 ms | 95.175% / 482.949 ms |
| Wiki-Cohere | 10000 | 18.665% / 10.720 ms | 54.290% / 19.293 ms | 93.185% / 129.940 ms | 95.056% / 257.228 ms | 95.464% / 550.052 ms |
| Wiki-Cohere | 100000 | 26.438% / 65.731 ms | 39.713% / 86.114 ms | 93.841% / 474.601 ms | 95.452% / 658.166 ms | 95.897% / 908.392 ms |
| DPR | 1 | 30.469% / 4.431 ms | 75.781% / 4.212 ms | 90.625% / 8.460 ms | 90.625% / 39.846 ms | 90.625% / 273.852 ms |
| DPR | 10 | 27.031% / 4.480 ms | 71.250% / 4.452 ms | 91.484% / 11.447 ms | 94.219% / 49.372 ms | 94.219% / 278.230 ms |
| DPR | 100 | 24.625% / 4.559 ms | 70.047% / 5.118 ms | 92.039% / 15.826 ms | 94.664% / 59.727 ms | 95.063% / 279.384 ms |
| DPR | 1000 | 18.842% / 4.867 ms | 65.266% / 6.740 ms | 92.727% / 25.382 ms | 95.362% / 77.371 ms | 95.645% / 289.544 ms |
| DPR | 10000 | 21.468% / 9.578 ms | 55.994% / 14.330 ms | 93.256% / 63.963 ms | 95.766% / 135.838 ms | 96.103% / 329.740 ms |
| DPR | 100000 | 30.925% / 64.013 ms | 35.386% / 68.261 ms | 93.920% / 293.181 ms | 96.235% / 407.044 ms | 96.611% / 582.983 ms |

The [complete CSV](search-effort-results.csv) also includes the old binary's
default arm, minimum/p50/p95 query recall, mean/p50/p95/max serial latency,
and partition/comparison counts for all 180 groups. A single timed pass
provides an observed query distribution, not repeated-run confidence
intervals. No statistically established default-latency improvement is claimed.

In these 30 corpus/k groups, effort 0.5 has mean recall from 85.938% to
96.306%; 7 groups reach 95%. Effort 0.75 is within 1.563 percentage points
of effort 1 in every group. These observations describe the sampled
queries only. Even searching every partition leaves quantization losses:
for example, DINO top-1 recall is 87.500% at both 0.75 and 1.

## Search work

Mean final partition counts include any candidate-count expansion after the
initial budget. Effort 0 may therefore search more than one partition,
especially at large k. Unit tests establish monotonic initial budgets;
neither final latency nor quantized final recall is assumed to be monotonic.

| Corpus | k | Effort 0 | Effort 0.25 | Effort 0.5 (Auto) | Effort 0.75 | Effort 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINO | 1 | 1.000 partitions | 3.477 partitions | 8.898 partitions | 142.078 partitions | 2441.000 partitions |
| DINO | 10 | 1.000 partitions | 3.703 partitions | 11.875 partitions | 164.031 partitions | 2441.000 partitions |
| DINO | 100 | 1.000 partitions | 5.016 partitions | 20.625 partitions | 216.516 partitions | 2441.000 partitions |
| DINO | 1000 | 1.000 partitions | 7.328 partitions | 49.805 partitions | 337.336 partitions | 2441.000 partitions |
| DINO | 10000 | 2.906 partitions | 12.172 partitions | 147.563 partitions | 587.852 partitions | 2441.000 partitions |
| DINO | 100000 | 24.469 partitions | 25.477 partitions | 542.625 partitions | 1145.930 partitions | 2441.000 partitions |
| LAION | 1 | 1.000 partitions | 3.750 partitions | 14.688 partitions | 165.516 partitions | 2441.000 partitions |
| LAION | 10 | 1.000 partitions | 4.844 partitions | 25.898 partitions | 225.539 partitions | 2441.000 partitions |
| LAION | 100 | 1.000 partitions | 6.320 partitions | 43.695 partitions | 287.875 partitions | 2441.000 partitions |
| LAION | 1000 | 1.008 partitions | 9.719 partitions | 106.141 partitions | 466.391 partitions | 2441.000 partitions |
| LAION | 10000 | 2.773 partitions | 15.977 partitions | 257.469 partitions | 751.836 partitions | 2441.000 partitions |
| LAION | 100000 | 24.313 partitions | 28.375 partitions | 766.133 partitions | 1367.008 partitions | 2441.000 partitions |
| FineWeb | 1 | 1.000 partitions | 4.875 partitions | 22.719 partitions | 216.688 partitions | 2441.000 partitions |
| FineWeb | 10 | 1.000 partitions | 6.047 partitions | 37.883 partitions | 282.531 partitions | 2441.000 partitions |
| FineWeb | 100 | 1.000 partitions | 8.164 partitions | 66.188 partitions | 377.656 partitions | 2441.000 partitions |
| FineWeb | 1000 | 1.000 partitions | 12.070 partitions | 154.492 partitions | 580.547 partitions | 2441.000 partitions |
| FineWeb | 10000 | 2.867 partitions | 19.055 partitions | 352.781 partitions | 904.961 partitions | 2441.000 partitions |
| FineWeb | 100000 | 24.750 partitions | 28.906 partitions | 802.609 partitions | 1398.070 partitions | 2441.000 partitions |
| Wiki-Cohere | 1 | 1.000 partitions | 11.000 partitions | 112.000 partitions | 979.000 partitions | 8545.000 partitions |
| Wiki-Cohere | 10 | 1.000 partitions | 15.836 partitions | 256.094 partitions | 1440.422 partitions | 8545.000 partitions |
| Wiki-Cohere | 100 | 1.000 partitions | 20.539 partitions | 430.336 partitions | 1849.141 partitions | 8545.000 partitions |
| Wiki-Cohere | 1000 | 1.000 partitions | 26.289 partitions | 704.656 partitions | 2375.219 partitions | 8545.000 partitions |
| Wiki-Cohere | 10000 | 2.750 partitions | 34.594 partitions | 1239.109 partitions | 3163.773 partitions | 8545.000 partitions |
| Wiki-Cohere | 100000 | 22.961 partitions | 51.000 partitions | 2528.000 partitions | 4648.000 partitions | 8545.000 partitions |
| DPR | 1 | 1.000 partitions | 10.109 partitions | 94.828 partitions | 692.180 partitions | 5131.000 partitions |
| DPR | 10 | 1.000 partitions | 12.000 partitions | 144.000 partitions | 860.000 partitions | 5131.000 partitions |
| DPR | 100 | 1.000 partitions | 15.000 partitions | 200.000 partitions | 1014.000 partitions | 5131.000 partitions |
| DPR | 1000 | 1.000 partitions | 17.000 partitions | 279.000 partitions | 1197.000 partitions | 5131.000 partitions |
| DPR | 10000 | 3.039 partitions | 23.000 partitions | 517.000 partitions | 1629.000 partitions | 5131.000 partitions |
| DPR | 100000 | 25.922 partitions | 33.750 partitions | 1137.383 partitions | 2400.445 partitions | 5131.000 partitions |

## Default compatibility control

The actual baseline is PR head `40fa85849ddf5dd85d2925462a0ed42b31287a7c`.
It was built and preserved before applying this change. It runs in its own
warm process on the same VM, using the same index and query IDs. Both arms
time scanner construction through materialized results inside their own
process; baseline IPC, ID mapping and recall calculation are outside timing.
The ratio below is baseline/current mean latency, with larger values meaning
less current latency. It describes this pass and is not a speedup claim.

| Scenario | Baseline default | This PR effort 0.5 | Observed ratio |
| --- | ---: | ---: | ---: |
| DINO, k=1 | 4.776 ms | 3.399 ms | 1.405x |
| DINO, k=10 | 4.889 ms | 3.632 ms | 1.346x |
| DINO, k=100 | 5.811 ms | 4.382 ms | 1.326x |
| DINO, k=1000 | 9.257 ms | 8.039 ms | 1.151x |
| DINO, k=10000 | 26.380 ms | 25.854 ms | 1.020x |
| DINO, k=100000 | 180.688 ms | 180.345 ms | 1.002x |
| LAION, k=1 | 4.812 ms | 3.429 ms | 1.403x |
| LAION, k=10 | 5.462 ms | 4.176 ms | 1.308x |
| LAION, k=100 | 6.835 ms | 5.590 ms | 1.223x |
| LAION, k=1000 | 11.620 ms | 10.642 ms | 1.092x |
| LAION, k=10000 | 31.481 ms | 31.400 ms | 1.003x |
| LAION, k=100000 | 182.344 ms | 184.592 ms | 0.988x |
| FineWeb, k=1 | 5.109 ms | 3.825 ms | 1.336x |
| FineWeb, k=10 | 5.882 ms | 4.746 ms | 1.239x |
| FineWeb, k=100 | 7.743 ms | 6.554 ms | 1.181x |
| FineWeb, k=1000 | 14.305 ms | 13.478 ms | 1.061x |
| FineWeb, k=10000 | 38.049 ms | 38.093 ms | 0.999x |
| FineWeb, k=100000 | 200.182 ms | 202.590 ms | 0.988x |
| Wiki-Cohere, k=1 | 11.906 ms | 10.448 ms | 1.140x |
| Wiki-Cohere, k=10 | 20.202 ms | 19.218 ms | 1.051x |
| Wiki-Cohere, k=100 | 31.406 ms | 31.169 ms | 1.008x |
| Wiki-Cohere, k=1000 | 55.806 ms | 56.477 ms | 0.988x |
| Wiki-Cohere, k=10000 | 127.656 ms | 129.940 ms | 0.982x |
| Wiki-Cohere, k=100000 | 470.017 ms | 474.601 ms | 0.990x |
| DPR, k=1 | 9.624 ms | 8.460 ms | 1.138x |
| DPR, k=10 | 12.455 ms | 11.447 ms | 1.088x |
| DPR, k=100 | 16.438 ms | 15.826 ms | 1.039x |
| DPR, k=1000 | 25.858 ms | 25.382 ms | 1.019x |
| DPR, k=10000 | 64.562 ms | 63.963 ms | 1.009x |
| DPR, k=100000 | 291.964 ms | 293.181 ms | 0.996x |

## Environment and frozen inputs

- AWS `r8i.8xlarge`, 32 vCPUs, 247 GiB visible memory, Linux x86-64.
- Both builds use the repository's `release-with-debug` profile, the same
  Rust toolchain and unchanged dependency lockfiles.
- CPU affinity 0–15; 16 Lance and Rayon threads; one BLAS/OMP thread;
  `query_parallelism=1`; warm index caches and zero measured storage bytes.
- Normal approximation, no refinement, no filter. The original frozen
  fast-rotation models, five-bit codes, centroids and row assignments are reused.
- The first 128 original held-out query IDs are used for recall. The first
  64 are timed serially; the other 64 run separately with eight workers and
  their timings do not enter the primary latency results.

| Corpus | Metric | Rows | Dimensions | Partitions |
| --- | --- | ---: | ---: | ---: |
| DINO | l2 | 10,000,000 | 1024 | 2,441 |
| LAION | cosine | 10,000,000 | 768 | 2,441 |
| FineWeb | cosine | 10,000,000 | 768 | 2,441 |
| Wiki-Cohere | dot | 35,000,000 | 768 | 8,545 |
| DPR | dot | 21,015,300 | 768 | 5,131 |

The [protocol](SEARCH_EFFORT_PROTOCOL.md) was frozen before the experiment.
Ground truth is the existing float64 exact top-100000 on original vectors.
Alternative tied IDs count as misses. Results use a smaller query sample
and different k grid than [the earlier RQ5 study](RQ5_RESULTS.md); comparisons
must use the matched records in this report.

Effort controls partition selection. Even effort 1 retains normal-mode
quantized scoring and pruning, so it is not exact search. HNSW ef,
approximation mode and refinement remain independent controls. An explicit
maximum bounds effort; non-default effort conflicts with fixed nprobes.

## Evidence and validation

- Final implementation: 542 Rust KNN tests, one scanner validation test,
  15 execution-plan wire tests, one Rust API doctest, 22 Rust JNI tests and
  39 Java tests pass. The selected 89 Python integration regressions pass
  against both development and optimized builds. Required formatting,
  workspace/Substrait Clippy, Python lint/type checks and Java checks pass.
- A floating-point boundary regression was reproduced before the final run:
  with lower=11, Auto=25 and effort immediately below 0.5, rounding selected
  26 partitions. The fix bounds each half of the interpolation at Auto.
  The expanded regression matrix passes. All reported numbers come from
  the full rerun after this correction, with unchanged frozen parameters;
  the earlier run and binary remain preserved separately.
- All 23,040 records return exactly k distinct, valid IDs. Independent set
  intersection recomputes recall from the saved IDs and original truth.
- Query coverage, calibration/evaluation separation, all-partition coverage,
  baseline-default identity and zero measured storage reads pass auditing.
- A calibration-only smoke run passes. Deliberately duplicating a returned
  ID makes the audit fail; restoring the ID makes it pass again.
- Omitted effort and explicit 0.5 agree on calibration queries at every k.
- Sources, native libraries, query vectors, ground truth, models and index
  files are hashed before and after measurement. Full identities and runtime
  metadata are in [search-effort-identities.json](search-effort-identities.json).
- Baseline native SHA256: `7cc8a98ef3dd76532b1d895e12757d2184f6bbe29c566b9d982586eaa00ba1ef`.
- Candidate native SHA256: `ce1320d527deb815a94a0c6dba070dd87289b2696991f31fa4b256f9b7c05a52`.

Raw per-query CSVs, returned-ID arrays, logs, source snapshots and all three
runtime packages are retained in the experiment archive. Frozen input data
and index files remain on the retained VM volume; prior input evidence keeps
its existing verified backup.
