# OSS-2249: dot Auto IVF probing

The frozen common profiles achieved at least 95% mean native recall in all six
held-out Wiki/DPR cases. These are empirical parameters for Float32 IVF_FLAT,
selected within the documented grid; they are not a universal optimum or recall
guarantee. No MS MARCO data was used. All execution ran on AWS.

## Common defaults

For sorted centroid distances `d_i = 1 - dot(q, c_i)`, take the prefix satisfying
`d_i - d_0 <= margin * abs(1 - d_0)`, then apply the learned floor and initial cap.
Caller minimums take precedence over the learned cap. Later probing remains
available when filtering or deletions exhaust the initial budget.

| k | margin | floor | initial cap | Wiki recall | DPR recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.14 | 56 partitions | 112 partitions | 95.320% | 96.565% |
| 2–10 | 0.055 | 144 partitions | 432 partitions | 95.428% | 96.454% |
| 11–100 | 0.0625 | 200 partitions | 768 partitions | 95.809% | 96.218% |

Recall was measured at k=1, 10, and 100; the buckets between those endpoints
were not separately benchmarked. The selection objective minimizes mean
partitions with a 96% calibration target on both corpora, weighting each corpus
equally. One gap family is chosen by aggregate cost across the three k groups;
parameters are independent per group. The relative-score family won that
aggregate objective. Standard-deviation normalization had a lower k=1 cost but
a higher aggregate cost. The exact search grid is in `calibrate.py` and the
archived `calibration.json`. Profiles were frozen before evaluation.

## Common Auto versus fixed budgets

The fixed budgets below are the neighboring tested powers of two that reached
95% held-out recall. All three neighboring budgets were selected on calibration
data and measured; the complete matrix, including misses, follows below.
This is a **policy comparison on the same candidate binary and frozen index**,
not an equal-recall or before/after branch speedup. Recall differs between arms.
Lower latency is better; the ratio is fixed mean / Auto mean.

| Scenario / metric | Baseline fixed policy | This PR common Auto | Benefit |
| --- | ---: | ---: | ---: |
| Wiki, k=1, mean latency | 128.26 ms (128 probes; 95.720% recall) | 113.26 ms (95.320% recall) | 1.13x mean speedup |
| Wiki, k=10, mean latency | 502.50 ms (512 probes; 97.272% recall) | 263.09 ms (95.428% recall) | 1.91x mean speedup |
| Wiki, k=100, mean latency | 518.94 ms (512 probes; 95.762% recall) | 453.49 ms (95.809% recall) | 1.14x mean speedup |
| DPR, k=1, mean latency | 120.45 ms (128 probes; 97.230% recall) | 91.38 ms (96.565% recall) | 1.32x mean speedup |
| DPR, k=10, mean latency | 120.75 ms (128 probes; 95.950% recall) | 135.99 ms (96.454% recall) | 1.13x slower |
| DPR, k=100, mean latency | 237.39 ms (256 probes; 97.232% recall) | 186.59 ms (96.218% recall) | 1.27x mean speedup |

- DPR k=10: common Auto is slower than fixed128, while both exceed 95% recall.
- Wiki k=100: Auto mean is lower, but p99 is 827.30 ms versus fixed512's
  555.05 ms (1.49x slower). There is no blanket tail-latency improvement.
- Wiki k=10: fixed256 was the calibration-selected reference but reaches only
  94.636% held-out recall. The preregistered neighboring fixed512 reaches 97.272%,
  versus Auto's 95.428%; their 1.91x mean-latency ratio is not an equal-quality gain.
- Mean recall is an empirical point estimate. In particular Wiki k=1's
  approximate 95% confidence interval includes values below 95%; this study does
  not establish a population-level 95% lower bound. Query-wise intervals are
  retained in `audited-results.json`.

## Per-corpus calibrated profiles

These minimize calibration cost within the selected relative-score family.
They were frozen independently of evaluation and were not adjusted after misses.
They can be supplied through `LANCE_AUTO_PROBE_MARGIN`,
`LANCE_AUTO_MIN_INITIAL_NPROBES`, and `LANCE_AUTO_MAX_INITIAL_NPROBES`, respectively.
Overrides are process-wide: do not change them concurrently with queries.
Choose a profile according to the recall and tail-latency requirements below.

| Corpus | k | margin / floor / cap (partitions) | native recall | mean ms | p99 ms | assessment |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Wiki | 1 | 0.033 / 72 / 336 | 94.920% | 91.90 | 313.89 | misses 95%; use common profile |
| Wiki | 10 | 0.055 / 112 / 464 | 95.416% | 260.78 | 488.12 | passes; inspect tail tradeoff |
| Wiki | 100 | 0.0605 / 248 / 768 | 95.750% | 444.52 | 820.19 | passes; inspect tail tradeoff |
| DPR | 1 | 0.11 / 32 / 256 | 96.620% | 80.55 | 242.53 | passes; inspect tail tradeoff |
| DPR | 10 | 0.125 / 64 / 240 | 96.709% | 115.81 | 232.49 | passes; inspect tail tradeoff |
| DPR | 100 | 0.135 / 96 / 496 | 96.122% | 162.31 | 465.11 | passes; inspect tail tradeoff |

## Setup and provenance

- AWS r8i.8xlarge, 32 vCPU / 256 GiB, Ubuntu 24.04.4; 2 TiB gp3 EBS,
  10,000 IOPS and 1,000 MiB/s provisioned throughput.
- Baseline main: `00a8230ba21341966cb3edac17b475b7b9219302`.
- [Wiki-Cohere 35M](https://huggingface.co/datasets/lance-format/wiki-cohere-35m/tree/cc3840e4e0c9091f26cdd0385fbb7a6ad81a7285):
  35,000,000 × 768 float32 vectors, 5,000 official queries; 8,545 IVF partitions.
- [DPR Wikipedia](https://huggingface.co/datasets/lance-format/dpr-wikipedia-single-nq/tree/d95c3715333a608fa5eb8ca6fe8e6bdb724f32bd):
  21,015,300 × 768 float32 vectors, 3,610 official queries; 5,131 IVF partitions.
  DPR retains its CC BY-NC 4.0 terms.
- Every published file's SHA256 was verified. Raw vector/query norms and row
  positions are preserved. Fresh baseline IVF_FLAT indices use default training
  and automatic HNSW assignment, with `ceil(rows / 4096)` partitions.
- Seed 2249 splits each official query set in half: 2,500 Wiki and 1,805 DPR
  evaluation queries. Calibration/evaluation IDs are disjoint, with no identical
  query-vector hashes crossing the split. Full-corpus GT supplies the top100 IDs.
- Every primary arm measures recall on every held-out query. Latency uses 512
  matched serial queries per arm, rotating policy order; affinity 0–15,
  Lance/Rayon threads 16, BLAS/OMP threads 1, query parallelism 1, prewarmed
  128 GiB index cache. All measured queries recorded zero storage bytes read.
- Remaining recall-only queries use eight workers; those elapsed times are
  excluded from latency summaries. The tuned phase uses constant global overrides.
- One timing observation per query/arm; NumPy linear percentiles. Latency
  variability across reruns or machines has not been measured. Scanned-row and
  partition distributions below use every held-out query, not just timed queries.
- Main's Auto and candidate's explicit full-bounded legacy path return identical
  IDs and scan counts on eight queries per corpus/k. Legacy uses only 32 queries
  per corpus/k for diagnostics; it is excluded from the primary comparison.

Native binary SHA256 (both report version `13.0.0-beta.9`):

```text
baseline  8f3ee435e88c2b88d782115bc114925d5c88e453f224856819a833c9aecd4f2d
candidate c04c3a7839194ed7493f014e59135096640b52493d326a2e192404b162a84a10
```

The archive contains the exact measured source patch, binaries, index identities,
centroids, membership/routing maps, query splits, every returned ID and raw
metric, calibration grid/results, audit results, and build/test/lint logs:
`s3://mmlb-us-east-1/rq-benchmark/results/oss2249-dot-auto-20260922/` (private).
`evidence.tar.gz` contains the data and logs; `binaries.tar.gz` contains the
baseline/candidate native extensions and routing diagnostic binary.
The full source datasets and frozen IVF indices remain on the stopped benchmark
VM; the archive records their file hashes rather than copying the full corpus.
The canceled 128-query baseline audit is labeled partial and excluded.

## Why the old policy fails

Fresh main indices have unit-norm dot centroids while raw query norms vary.
Median nearest-centroid scores were 0.721 for Wiki and 6.308 for DPR, producing
opposite signs in `1 - dot`. The legacy signed-distance multiplier searches one
partition for Wiki k=1, every partition for Wiki k=10/100, and one partition for
DPR k=10/100. Native baseline queries confirmed the behavior after prewarming.
The new nonnegative gap uses the magnitude of the best inner product and is
invariant to positive query scaling apart from floating-point rounding.

## Validation and independent audit

- The unmodified baseline failed both new Python regression cases (16 and 1
  partitions instead of the requested floor/cap of 2); the candidate passes.
- Rust KNN: 187 tests passed. Python dot tests: 4 passed on both development
  and optimized extensions, including multiple fragments and late filtered probing.
- `cargo fmt --all`, workspace `cargo clippy --all --tests --benches -- -D warnings`,
  and `uv run make lint` passed on AWS.
- `audit.py` independently recomputes recall from every returned ID, checks unique
  query coverage, result cardinality, zero I/O, binary identity and frozen
  calibration identity. All six common profiles pass the 95% mean-recall target.
- 215 NumPy/native prediction discrepancies were inspected using native
  `IvfModel::find_partitions`, including its partial-sort behavior. Every actual
  partition count and scanned-row count agrees with native routing. Residual
  strict-ID recall differences are near ties under float64 rescoring; they remain
  counted as misses in the reported recall. Details and numerical tolerances are
  in `routing-diagnosis.json`, not silently removed from the measurements.

The implementation applies to finite Float32 IVF_FLAT queries with k <= 100.
Fixed budgets, explicit Auto maximums, unsupported index/query types, refinement
factors greater than one, legacy readers, and k > 100 keep their prior policy.
L2/cosine behavior is unchanged. No public signature or index format changes.
Other corpora, partition counts, quantizers, cold I/O, and filtered-workload
performance have not been calibrated here.

## Complete native matrix

`auto` is the common profile; `tuned` is the per-corpus profile. Latency units are
milliseconds. Primary recall sample sizes are Wiki 2,500 and DPR 1,805; primary
timing sample size is 512. Legacy has only 32 observations for both.

### Wiki

<details>
<summary>Latency and recall</summary>

All distribution values are in ms.

| k | policy | recall | mean | p90 | p95 | p99 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | auto | 95.320% | 113.26 | 120.25 | 121.89 | 125.94 | 130.76 |
| 1 | fixed128 | 95.720% | 128.26 | 135.52 | 137.78 | 141.46 | 146.01 |
| 1 | fixed256 | 97.760% | 252.38 | 264.08 | 267.65 | 275.19 | 280.59 |
| 1 | fixed64 | 93.400% | 66.46 | 71.09 | 72.52 | 74.89 | 76.03 |
| 1 | legacy | 43.750% | 5.42 | 5.82 | 5.91 | 6.13 | 6.18 |
| 1 | tuned | 94.920% | 91.90 | 142.47 | 191.02 | 313.89 | 346.55 |
| 10 | auto | 95.428% | 263.09 | 435.91 | 442.77 | 454.95 | 463.49 |
| 10 | fixed128 | 90.908% | 128.69 | 136.24 | 138.40 | 142.62 | 146.16 |
| 10 | fixed256 | 94.636% | 252.64 | 264.75 | 267.40 | 276.11 | 282.65 |
| 10 | fixed512 | 97.272% | 502.50 | 522.35 | 528.59 | 535.72 | 542.35 |
| 10 | legacy | 100.000% | 7,876.54 | 7,895.01 | 7,898.01 | 7,907.54 | 7,910.71 |
| 10 | tuned | 95.416% | 260.78 | 467.74 | 475.86 | 488.12 | 495.69 |
| 100 | auto | 95.809% | 453.49 | 793.87 | 809.57 | 827.30 | 840.32 |
| 100 | fixed1024 | 97.905% | 1,038.29 | 1,072.57 | 1,078.50 | 1,089.57 | 1,130.21 |
| 100 | fixed256 | 92.424% | 261.95 | 274.11 | 277.33 | 287.55 | 301.96 |
| 100 | fixed512 | 95.762% | 518.94 | 539.65 | 546.11 | 555.05 | 580.98 |
| 100 | legacy | 100.000% | 8,239.30 | 8,406.48 | 8,413.79 | 8,416.41 | 8,417.19 |
| 100 | tuned | 95.750% | 444.52 | 790.46 | 806.97 | 820.19 | 861.43 |

</details>

<details>
<summary>Partitions scanned</summary>

All distribution values are in partitions.

| k | policy | recall | mean | p90 | p95 | p99 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | auto | 95.320% | 111.98 | 112.00 | 112.00 | 112.00 | 112.00 |
| 1 | fixed128 | 95.720% | 128.00 | 128.00 | 128.00 | 128.00 | 128.00 |
| 1 | fixed256 | 97.760% | 256.00 | 256.00 | 256.00 | 256.00 | 256.00 |
| 1 | fixed64 | 93.400% | 64.00 | 64.00 | 64.00 | 64.00 | 64.00 |
| 1 | legacy | 43.750% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 | tuned | 94.920% | 90.40 | 139.10 | 188.00 | 331.03 | 336.00 |
| 10 | auto | 95.428% | 268.51 | 432.00 | 432.00 | 432.00 | 432.00 |
| 10 | fixed128 | 90.908% | 128.00 | 128.00 | 128.00 | 128.00 | 128.00 |
| 10 | fixed256 | 94.636% | 256.00 | 256.00 | 256.00 | 256.00 | 256.00 |
| 10 | fixed512 | 97.272% | 512.00 | 512.00 | 512.00 | 512.00 | 512.00 |
| 10 | legacy | 100.000% | 8,545.00 | 8,545.00 | 8,545.00 | 8,545.00 | 8,545.00 |
| 10 | tuned | 95.416% | 266.92 | 464.00 | 464.00 | 464.00 | 464.00 |
| 100 | auto | 95.809% | 451.98 | 768.00 | 768.00 | 768.00 | 768.00 |
| 100 | fixed1024 | 97.905% | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 |
| 100 | fixed256 | 92.424% | 256.00 | 256.00 | 256.00 | 256.00 | 256.00 |
| 100 | fixed512 | 95.762% | 512.00 | 512.00 | 512.00 | 512.00 | 512.00 |
| 100 | legacy | 100.000% | 8,545.00 | 8,545.00 | 8,545.00 | 8,545.00 | 8,545.00 |
| 100 | tuned | 95.750% | 444.00 | 768.00 | 768.00 | 768.00 | 768.00 |

</details>

<details>
<summary>Rows scanned</summary>

All distribution values are in rows.

| k | policy | recall | mean | p90 | p95 | p99 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | auto | 95.320% | 487,846.67 | 518,601.40 | 527,357.30 | 546,888.91 | 576,565.00 |
| 1 | fixed128 | 95.720% | 557,320.73 | 590,569.30 | 599,961.75 | 619,670.59 | 651,684.00 |
| 1 | fixed256 | 97.760% | 1,112,215.75 | 1,165,693.20 | 1,179,587.50 | 1,204,510.01 | 1,239,295.00 |
| 1 | fixed64 | 93.400% | 279,741.70 | 301,192.70 | 307,661.25 | 318,967.76 | 343,753.00 |
| 1 | legacy | 43.750% | 4,766.78 | 6,136.40 | 6,413.75 | 6,855.04 | 7,015.00 |
| 1 | tuned | 94.920% | 395,659.33 | 619,157.70 | 844,534.70 | 1,436,441.19 | 1,553,201.00 |
| 10 | auto | 95.428% | 1,172,510.13 | 1,930,456.90 | 1,959,398.80 | 2,002,112.22 | 2,049,345.00 |
| 10 | fixed128 | 90.908% | 557,320.73 | 590,569.30 | 599,961.75 | 619,670.59 | 651,684.00 |
| 10 | fixed256 | 94.636% | 1,112,215.75 | 1,165,693.20 | 1,179,587.50 | 1,204,510.01 | 1,239,295.00 |
| 10 | fixed512 | 97.272% | 2,227,195.43 | 2,313,214.70 | 2,338,086.25 | 2,373,320.98 | 2,421,818.00 |
| 10 | legacy | 100.000% | 35,000,000.00 | 35,000,000.00 | 35,000,000.00 | 35,000,000.00 | 35,000,000.00 |
| 10 | tuned | 95.416% | 1,166,710.63 | 2,070,476.90 | 2,100,250.05 | 2,146,024.57 | 2,186,004.00 |
| 100 | auto | 95.809% | 1,978,009.75 | 3,425,574.00 | 3,470,017.90 | 3,530,335.76 | 3,567,863.00 |
| 100 | fixed1024 | 97.905% | 4,474,592.19 | 4,603,210.20 | 4,640,992.40 | 4,704,198.88 | 4,752,187.00 |
| 100 | fixed256 | 92.424% | 1,112,215.75 | 1,165,693.20 | 1,179,587.50 | 1,204,510.01 | 1,239,295.00 |
| 100 | fixed512 | 95.762% | 2,227,195.43 | 2,313,214.70 | 2,338,086.25 | 2,373,320.98 | 2,421,818.00 |
| 100 | legacy | 100.000% | 35,000,000.00 | 35,000,000.00 | 35,000,000.00 | 35,000,000.00 | 35,000,000.00 |
| 100 | tuned | 95.750% | 1,941,612.04 | 3,417,389.70 | 3,463,375.55 | 3,526,866.85 | 3,567,863.00 |

</details>

### DPR

<details>
<summary>Latency and recall</summary>

All distribution values are in ms.

| k | policy | recall | mean | p90 | p95 | p99 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | auto | 96.565% | 91.38 | 111.37 | 112.37 | 114.96 | 118.21 |
| 1 | fixed128 | 97.230% | 120.45 | 126.29 | 127.58 | 129.81 | 131.15 |
| 1 | fixed256 | 99.169% | 237.71 | 247.19 | 249.62 | 253.17 | 253.84 |
| 1 | fixed64 | 94.238% | 62.26 | 65.75 | 66.82 | 67.97 | 71.33 |
| 1 | legacy | 100.000% | 3,484.27 | 4,255.68 | 4,431.53 | 4,508.53 | 4,540.36 |
| 1 | tuned | 96.620% | 80.55 | 157.94 | 196.57 | 242.53 | 246.46 |
| 10 | auto | 96.454% | 135.99 | 142.41 | 143.95 | 146.15 | 147.18 |
| 10 | fixed128 | 95.950% | 120.75 | 126.84 | 128.07 | 129.97 | 131.49 |
| 10 | fixed256 | 98.366% | 238.10 | 248.07 | 250.62 | 254.04 | 257.04 |
| 10 | fixed64 | 91.861% | 62.47 | 66.12 | 66.71 | 67.66 | 70.35 |
| 10 | legacy | 32.812% | 5.72 | 6.32 | 6.44 | 6.70 | 6.79 |
| 10 | tuned | 96.709% | 115.81 | 218.46 | 227.52 | 232.49 | 238.03 |
| 100 | auto | 96.218% | 186.59 | 194.63 | 196.58 | 199.73 | 205.71 |
| 100 | fixed128 | 93.888% | 120.77 | 126.46 | 127.94 | 130.50 | 137.92 |
| 100 | fixed256 | 97.232% | 237.39 | 247.50 | 249.96 | 254.57 | 269.03 |
| 100 | fixed512 | 98.968% | 472.73 | 487.17 | 490.38 | 499.89 | 504.92 |
| 100 | legacy | 26.062% | 5.83 | 6.29 | 6.38 | 6.96 | 7.21 |
| 100 | tuned | 96.122% | 162.31 | 301.79 | 378.26 | 465.11 | 478.44 |

</details>

<details>
<summary>Partitions scanned</summary>

All distribution values are in partitions.

| k | policy | recall | mean | p90 | p95 | p99 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | auto | 96.565% | 95.15 | 112.00 | 112.00 | 112.00 | 112.00 |
| 1 | fixed128 | 97.230% | 128.00 | 128.00 | 128.00 | 128.00 | 128.00 |
| 1 | fixed256 | 99.169% | 256.00 | 256.00 | 256.00 | 256.00 | 256.00 |
| 1 | fixed64 | 94.238% | 64.00 | 64.00 | 64.00 | 64.00 | 64.00 |
| 1 | legacy | 100.000% | 3,701.38 | 4,511.30 | 4,682.50 | 4,752.86 | 4,782.00 |
| 1 | tuned | 96.620% | 80.68 | 165.00 | 200.00 | 256.00 | 256.00 |
| 10 | auto | 96.454% | 144.00 | 144.00 | 144.00 | 144.00 | 144.00 |
| 10 | fixed128 | 95.950% | 128.00 | 128.00 | 128.00 | 128.00 | 128.00 |
| 10 | fixed256 | 98.366% | 256.00 | 256.00 | 256.00 | 256.00 | 256.00 |
| 10 | fixed64 | 91.861% | 64.00 | 64.00 | 64.00 | 64.00 | 64.00 |
| 10 | legacy | 32.812% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10 | tuned | 96.709% | 119.07 | 240.00 | 240.00 | 240.00 | 240.00 |
| 100 | auto | 96.218% | 200.00 | 200.00 | 200.00 | 200.00 | 200.00 |
| 100 | fixed128 | 93.888% | 128.00 | 128.00 | 128.00 | 128.00 | 128.00 |
| 100 | fixed256 | 97.232% | 256.00 | 256.00 | 256.00 | 256.00 | 256.00 |
| 100 | fixed512 | 98.968% | 512.00 | 512.00 | 512.00 | 512.00 | 512.00 |
| 100 | legacy | 26.062% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 100 | tuned | 96.122% | 167.71 | 304.60 | 387.60 | 496.00 | 496.00 |

</details>

<details>
<summary>Rows scanned</summary>

All distribution values are in rows.

| k | policy | recall | mean | p90 | p95 | p99 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | auto | 96.565% | 373,306.53 | 458,631.00 | 464,569.80 | 474,240.76 | 483,995.00 |
| 1 | fixed128 | 97.230% | 501,553.33 | 526,285.40 | 531,654.60 | 539,997.52 | 557,562.00 |
| 1 | fixed256 | 99.169% | 1,006,389.42 | 1,047,725.20 | 1,056,201.60 | 1,072,031.64 | 1,087,576.00 |
| 1 | fixed64 | 94.238% | 250,010.18 | 264,539.80 | 267,534.20 | 273,331.48 | 280,373.00 |
| 1 | legacy | 100.000% | 15,069,122.81 | 18,440,124.50 | 19,174,839.45 | 19,482,897.12 | 19,614,786.00 |
| 1 | tuned | 96.620% | 318,219.34 | 658,136.80 | 793,762.00 | 1,023,053.80 | 1,061,549.00 |
| 10 | auto | 96.454% | 564,498.75 | 591,235.20 | 597,280.60 | 606,902.36 | 618,749.00 |
| 10 | fixed128 | 95.950% | 501,553.33 | 526,285.40 | 531,654.60 | 539,997.52 | 557,562.00 |
| 10 | fixed256 | 98.366% | 1,006,389.42 | 1,047,725.20 | 1,056,201.60 | 1,072,031.64 | 1,087,576.00 |
| 10 | fixed64 | 91.861% | 250,010.18 | 264,539.80 | 267,534.20 | 273,331.48 | 280,373.00 |
| 10 | legacy | 32.812% | 3,889.16 | 5,109.10 | 5,396.45 | 5,808.20 | 5,988.00 |
| 10 | tuned | 96.709% | 468,805.20 | 914,753.20 | 952,196.80 | 984,725.72 | 1,002,354.00 |
| 100 | auto | 96.218% | 785,402.82 | 819,545.00 | 828,141.80 | 839,150.64 | 855,206.00 |
| 100 | fixed128 | 93.888% | 501,553.33 | 526,285.40 | 531,654.60 | 539,997.52 | 557,562.00 |
| 100 | fixed256 | 97.232% | 1,006,389.42 | 1,047,725.20 | 1,056,201.60 | 1,072,031.64 | 1,087,576.00 |
| 100 | fixed512 | 98.968% | 2,023,956.78 | 2,084,049.00 | 2,096,435.40 | 2,114,013.88 | 2,133,549.00 |
| 100 | legacy | 26.062% | 3,889.16 | 5,109.10 | 5,396.45 | 5,808.20 | 5,988.00 |
| 100 | tuned | 96.122% | 660,744.19 | 1,217,256.00 | 1,516,333.60 | 1,959,414.36 | 2,016,202.00 |

</details>

