# OSS-2254: adaptive IVF probing through k=1000

This original IVF_FLAT campaign measures the extension of the calibrated initial
probe budget from k <= 100 to k <= 1000 at runtime source `0bcc82431`, using
finite Float32 queries with L2, cosine, or dot. Three new buckets
end at k=200, 500, and 1000. Caller minimums can exceed the initial cap;
late probing remains available when filters or deletions leave too few results.

All **20 newly supported unfiltered corpus/k groups** exceed 95% mean strict-ID
recall, ranging from 95.910% to 98.005%. The preregistered all-k quality gate is
**24/25**: the unchanged FineWeb k=100 control reaches **94.97875%**. Consequently,
`audit.py` exits with status 1 after saving the complete results. The target and
parameters were not changed after evaluation.

The 0.1% prefilter checks return k for every Auto query, but mean recall ranges
from 39.460% to 78.732%. The unfiltered calibration target therefore does not
establish a filtered recall guarantee.

## Native held-out results

| Corpus | Queries | k=100 control | k=101 | k=200 | k=500 | k=1000 |
| --- | --- | --- | --- | --- | --- | --- |
| DINO | 3072 | 95.309% | 96.969% | 96.218% | 96.135% | 96.116% |
| LAION | 3069 | 96.564% | 98.005% | 97.405% | 97.307% | 97.038% |
| FineWeb | 3200 | 94.979% | 96.535% | 95.910% | 95.960% | 95.999% |
| Wiki-Cohere | 2500 | 95.665% | 96.306% | 95.942% | 95.998% | 96.021% |
| DPR | 1805 | 96.222% | 96.392% | 96.009% | 95.963% | 95.926% |

Mean strict-ID recall; audit target misses: fineweb-10m/100.

## Actual baseline comparison

Both binaries use AWS r8i.8xlarge hardware, the repository `release-with-debug` profile, CPU affinity 0-15, 16 Lance/Rayon threads and one BLAS/OMP thread. They query the same frozen IVF_FLAT indices with prewarmed caches and zero storage bytes read. Corpus sizes are 10M rows for DINO/LAION/FineWeb, 35M for Wiki-Cohere and 21,015,300 for DPR; versions and index identities are listed under [Reproduction contract](#reproduction-contract).

Each row compares exactly the same 32 queries in both binaries, with interleaved serial timing. Recall differs, so these are measured policy tradeoffs, not equal-quality speedups. The benefit is baseline mean latency divided by PR mean latency. Latency quantiles describe one pass, not repeated-run confidence intervals.

| Scenario / metric | Baseline | This PR | Benefit | Recall: baseline / PR |
| --- | --- | --- | --- | --- |
| DINO, k=101; mean latency (lower is better) | 3067.730 ms | 42.426 ms | 72.31x speedup | 100.000% / 97.587% |
| DINO, k=200; mean latency (lower is better) | 3069.477 ms | 42.651 ms | 71.97x speedup | 99.984% / 97.000% |
| DINO, k=500; mean latency (lower is better) | 3073.827 ms | 53.061 ms | 57.93x speedup | 100.000% / 96.225% |
| DINO, k=1000; mean latency (lower is better) | 3071.233 ms | 69.565 ms | 44.15x speedup | 100.000% / 96.181% |
| LAION, k=101; mean latency (lower is better) | 2290.260 ms | 75.416 ms | 30.37x speedup | 99.969% / 97.649% |
| LAION, k=200; mean latency (lower is better) | 2286.090 ms | 75.261 ms | 30.38x speedup | 99.922% / 96.719% |
| LAION, k=500; mean latency (lower is better) | 2285.245 ms | 94.873 ms | 24.09x speedup | 100.000% / 96.287% |
| LAION, k=1000; mean latency (lower is better) | 2278.274 ms | 116.605 ms | 19.54x speedup | 100.000% / 96.053% |
| FineWeb, k=101; mean latency (lower is better) | 2301.909 ms | 98.312 ms | 23.41x speedup | 100.000% / 96.442% |
| FineWeb, k=200; mean latency (lower is better) | 2288.419 ms | 98.045 ms | 23.34x speedup | 100.000% / 95.766% |
| FineWeb, k=500; mean latency (lower is better) | 2292.709 ms | 125.351 ms | 18.29x speedup | 100.000% / 96.044% |
| FineWeb, k=1000; mean latency (lower is better) | 2298.063 ms | 156.305 ms | 14.70x speedup | 99.997% / 95.897% |
| Wiki-Cohere, k=101; mean latency (lower is better) | 8022.230 ms | 516.244 ms | 15.54x speedup | 100.000% / 95.328% |
| Wiki-Cohere, k=200; mean latency (lower is better) | 8036.185 ms | 516.818 ms | 15.55x speedup | 100.000% / 95.328% |
| Wiki-Cohere, k=500; mean latency (lower is better) | 8029.209 ms | 607.293 ms | 13.22x speedup | 100.000% / 95.463% |
| Wiki-Cohere, k=1000; mean latency (lower is better) | 8037.084 ms | 693.537 ms | 11.59x speedup | 100.000% / 95.600% |

| Scenario | Baseline mean latency | PR mean latency | Cost | Recall: baseline / PR |
| --- | --- | --- | --- | --- |
| DPR, k=101 | 5.463 ms | 194.468 ms | 35.60x latency | 22.277% / 95.575% |
| DPR, k=200 | 5.575 ms | 194.958 ms | 34.97x latency | 21.016% / 95.766% |
| DPR, k=500 | 5.630 ms | 224.943 ms | 39.96x latency | 18.800% / 96.112% |
| DPR, k=1000 | 5.788 ms | 259.929 ms | 44.91x latency | 17.300% / 96.122% |

## Fixed-budget comparison

Each cell reports mean recall over all held-out queries and mean latency over the first 512 serial queries. The nearby budgets were selected on calibration queries. The complete CSV also includes `tuned`, a per-corpus calibration reference; production Auto uses the shared metric-specific profiles.

| Corpus | k | Auto: recall / latency | fixed20: recall / latency | Nearby fixed: recall / latency |
| --- | --- | --- | --- | --- |
| DINO | 101 | 96.969% / 42.791 ms | 93.747% / 28.944 ms | fixed32: 96.170% / 44.174 ms; fixed48: 97.533% / 64.870 ms |
| DINO | 200 | 96.218% / 43.110 ms | 92.615% / 29.192 ms | fixed32: 95.394% / 44.386 ms; fixed48: 96.999% / 65.131 ms; fixed64: 97.836% / 85.981 ms |
| DINO | 500 | 96.135% / 55.019 ms | 90.679% / 29.289 ms | fixed32: 93.996% / 44.401 ms; fixed48: 96.004% / 65.038 ms; fixed64: 97.093% / 85.842 ms |
| DINO | 1000 | 96.116% / 69.710 ms | 88.653% / 29.733 ms | fixed48: 94.924% / 65.519 ms; fixed64: 96.246% / 86.415 ms; fixed96: 97.617% / 127.824 ms |
| LAION | 101 | 98.005% / 58.737 ms | 92.140% / 22.198 ms | fixed32: 94.644% / 33.446 ms; fixed48: 96.288% / 48.589 ms |
| LAION | 200 | 97.405% / 58.790 ms | 90.823% / 22.177 ms | fixed32: 93.667% / 33.417 ms; fixed48: 95.583% / 48.618 ms; fixed64: 96.664% / 63.800 ms |
| LAION | 500 | 97.307% / 76.867 ms | 88.516% / 22.646 ms | fixed48: 94.238% / 49.102 ms; fixed64: 95.581% / 64.365 ms; fixed96: 97.085% / 94.935 ms |
| LAION | 1000 | 97.038% / 93.865 ms | 86.171% / 22.877 ms | fixed64: 94.417% / 64.262 ms; fixed96: 96.250% / 94.852 ms; fixed128: 97.222% / 125.433 ms |
| FineWeb | 101 | 96.535% / 93.420 ms | 85.940% / 22.189 ms | fixed64: 94.146% / 62.694 ms; fixed96: 95.940% / 92.517 ms; fixed128: 96.952% / 122.490 ms |
| FineWeb | 200 | 95.910% / 94.137 ms | 84.291% / 22.501 ms | fixed64: 93.296% / 63.269 ms; fixed96: 95.314% / 93.252 ms; fixed128: 96.431% / 123.445 ms |
| FineWeb | 500 | 95.960% / 120.619 ms | 81.540% / 22.742 ms | fixed96: 94.190% / 93.341 ms; fixed128: 95.558% / 123.497 ms; fixed192: 97.092% / 183.686 ms |
| FineWeb | 1000 | 95.999% / 150.757 ms | 78.816% / 23.074 ms | fixed128: 94.639% / 123.588 ms; fixed192: 96.448% / 184.026 ms; fixed256: 97.443% / 244.415 ms |
| Wiki-Cohere | 101 | 96.306% / 537.106 ms | 68.533% / 24.998 ms | fixed384: 94.483% / 380.571 ms; fixed512: 95.706% / 506.388 ms; fixed768: 97.124% / 758.739 ms |
| Wiki-Cohere | 200 | 95.942% / 536.504 ms | 66.679% / 25.044 ms | fixed384: 93.978% / 380.118 ms; fixed512: 95.306% / 505.798 ms; fixed768: 96.836% / 757.732 ms |
| Wiki-Cohere | 500 | 95.998% / 632.442 ms | 63.833% / 25.376 ms | fixed512: 94.636% / 508.840 ms; fixed768: 96.334% / 762.946 ms; fixed1024: 97.295% / 1017.052 ms |
| Wiki-Cohere | 1000 | 96.021% / 716.126 ms | 61.034% / 25.491 ms | fixed512: 93.982% / 504.272 ms; fixed768: 95.851% / 756.043 ms; fixed1024: 96.919% / 1007.611 ms |
| DPR | 101 | 96.392% / 192.453 ms | 75.437% / 21.844 ms | fixed128: 93.808% / 117.346 ms; fixed192: 96.028% / 174.944 ms; fixed256: 97.149% / 232.864 ms |
| DPR | 200 | 96.009% / 193.347 ms | 73.787% / 22.016 ms | fixed128: 93.229% / 117.995 ms; fixed192: 95.613% / 175.863 ms; fixed256: 96.815% / 233.912 ms |
| DPR | 500 | 95.963% / 223.932 ms | 70.974% / 22.284 ms | fixed192: 94.773% / 176.137 ms; fixed256: 96.159% / 234.668 ms; fixed384: 97.662% / 351.012 ms |
| DPR | 1000 | 95.926% / 255.767 ms | 68.181% / 22.579 ms | fixed192: 93.942% / 176.246 ms; fixed256: 95.511% / 234.633 ms; fixed384: 97.229% / 350.747 ms |

## Filtered results

Each cell reports mean returned count / mean strict-ID recall / mean partitions over the same 256 held-out queries with `_rowid % 1000 = 0`. Auto and legacy return k for every query. These runs are not used for latency claims.

| Corpus | k | Auto: count / recall / partitions | Legacy: count / recall / partitions | fixed20: count / recall / partitions |
| --- | --- | --- | --- | --- |
| DINO | 101 | 101.0 / 46.972% / 35.7 | 101.0 / 100.000% / 2441.0 | 83.7 / 35.489% / 20.0 |
| DINO | 1000 | 1000.0 / 49.846% / 236.1 | 1000.0 / 100.000% / 2441.0 | 83.9 / 7.238% / 20.0 |
| LAION | 101 | 101.0 / 47.362% / 66.0 | 101.0 / 99.706% / 2431.6 | 82.2 / 30.546% / 20.0 |
| LAION | 1000 | 1000.0 / 39.460% / 246.2 | 1000.0 / 99.735% / 2432.4 | 82.5 / 6.107% / 20.0 |
| FineWeb | 101 | 101.0 / 60.725% / 97.3 | 101.0 / 100.000% / 2441.0 | 83.1 / 33.068% / 20.0 |
| FineWeb | 1000 | 1000.0 / 46.982% / 246.9 | 1000.0 / 100.000% / 2441.0 | 83.2 / 7.119% / 20.0 |
| Wiki-Cohere | 101 | 101.0 / 78.732% / 543.5 | 101.0 / 100.000% / 8545.0 | 88.0 / 23.747% / 20.0 |
| Wiki-Cohere | 1000 | 1000.0 / 60.900% / 727.5 | 1000.0 / 100.000% / 8545.0 | 89.1 / 6.646% / 20.0 |
| DPR | 101 | 101.0 / 72.803% / 211.0 | 101.0 / 30.407% / 25.8 | 80.1 / 26.284% / 20.0 |
| DPR | 1000 | 1000.0 / 45.257% / 279.0 | 1000.0 / 42.191% / 248.3 | 80.3 / 6.270% / 20.0 |

## Complete distributions

[native-results.csv](native-results.csv) contains every native arm and its mean/p50/p90/p95/p99/max latency, partitions, scanned rows and returned counts. Filtered latency fields are empty because those runs only measure correctness and counts. Scanned rows are the native `index_comparisons` counter: FLAT search charges the total storage length of each searched partition before filtering, so this counter can exceed the number of evaluated distances. Storage bytes are audited separately. [baseline-comparisons.csv](baseline-comparisons.csv) contains the matched actual-binary subset. [routing-simulation.csv](routing-simulation.csv) records calibration and held-out routing results for all candidate designs, including k=2000/5000 experiments; those rows are simulations, not native timings.

## Frozen profiles

| Metric | k | Margin | Initial floor | Initial cap |
| --- | ---: | ---: | ---: | ---: |
| L2 | 101–200 | 0.375 | 15 partitions | 56 partitions |
| L2 | 201–500 | 0.3725 | 22 partitions | 81 partitions |
| L2 | 501–1000 | 0.4175 | 29 partitions | 93 partitions |
| Cosine | 101–200 | 0.415 | 18 partitions | 156 partitions |
| Cosine | 201–500 | 0.45 | 28 partitions | 192 partitions |
| Cosine | 501–1000 | 0.47 | 34 partitions | 248 partitions |
| Dot | 101–200 | 0.0675 | 211 partitions | 833 partitions |
| Dot | 201–500 | 0.07 | 244 partitions | 971 partitions |
| Dot | 501–1000 | 0.0725 | 279 partitions | 1092 partitions |

Each profile minimizes mean initial partitions within the documented search grid,
subject to 96% mean calibration recall on every corpus for that metric. Corpus
weights are equal. Parameters depend on metric and k, not corpus identity.
Existing k <= 100 profiles, fixed nprobes, explicit caller maximums, unsupported
indices and vector types retain their previous behavior.

## Why three buckets and a finite range

A single bucket calibrated at k=1000 uses 1.32–1.65 times as many initial
partitions at k=101 as the three-bucket design on the calibration queries. It
fails the preregistered requirement that the simpler alternative cost at most
5% more on every corpus and simulated k.

Reusing the k=1000 profile beyond its range loses recall. These are calibration
routing results; they are not native held-out measurements.

| Corpus | Reused k=1000 profile at k=2000 | Reused k=1000 profile at k=5000 |
| --- | ---: | ---: |
| DINO | 94.497% recall | 91.229% recall |
| LAION | 95.890% recall | 92.846% recall |
| FineWeb | 94.828% recall | 92.403% recall |
| Wiki-Cohere | 95.362% recall | 94.093% recall |
| DPR | 95.270% recall | 93.762% recall |

Separately calibrated k=2000/5000 anchors are retained in the experiment, but
native validation covers the requested 100/101 boundary and k=200/500/1000.
Queries above k=1000 keep the legacy policy and ignore the experimental Auto
overrides. This leaves an explicit boundary at 1000/1001; it does not establish
a policy for arbitrary k.

## Reproduction contract

- Baseline source: `567e322b6644a24cef98a57b565807eea71805f3`.
- Measured candidate runtime source: `0bcc82431521017fc0afe2772a214d407a44c9be`.
  The results follow-up changes documentation and recorded artifacts only.
- Candidate native SHA256:
  `17bd3bb409c92e4a5e83ea4b4573a2e49c2195fc074dd86ebb4936fc13f706d6`.
- Baseline native SHA256:
  `9c5b08417d6dfe7b319039b80b02a876b1e6d8078fe2820c46319150f359bcab`.
- AWS r8i.8xlarge, Intel Xeon 6975P-C, 32 vCPU / 256 GiB RAM, 1600 GB gp3.
- Linux 6.17.0-1019-aws, glibc 2.39, Python 3.12.3, NumPy 2.5.1 and
  Lance 14.0.0-beta.6. Both binaries use Rust 1.98.1 and the repository
  `release-with-debug` profile (ThinLTO, 16 codegen units, debug symbols), with
  the repository Haswell/AVX2/FMA/F16C target settings. Cargo lockfiles match
  the pinned source in all three workspaces.
- Serial timing uses one query at a time, affinity 0–15, Lance/Rayon threads 16, BLAS/OMP threads 1,
  query_parallelism=1. Both processes prewarm the same frozen index. Every timed
  query must report zero storage bytes read. CPU-intensive preparation and
  compilation finish before timing starts.
- Candidate Auto, tuned and fixed policies time the first 512 held-out queries.
  Actual baseline Auto runs on the first 32 of those queries in a separate
  process, interleaved with the candidate; IPC is outside the measured interval.
  Before/after ratios use only those matched 32 queries. Remaining held-out
  queries use eight workers and contribute recall and scan counts only.
- Queries return `_rowid` and `_distance`. Wall time spans scanner construction
  through `to_table()`; ID mapping, recall evaluation and baseline IPC are outside
  the timed interval.
- Exact fixed20 is always included, alongside the calibration-selected nearby
  fixed budgets. Recall and returned counts accompany latency comparisons.
  The full mean/p90/p95/p99/max distributions are retained per corpus and k.

| Corpus | Rows | Dimensions | Metric | Dataset version | Index UUID | Calibration queries | Held-out queries |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| DINO | 10,000,000 | 1024 | L2 | 2 | `a2449e25-2c49-41f9-b3e2-3a3969491f08` | 3072 | 3072 |
| LAION | 10,000,000 | 768 | Cosine | 2 | `28448469-7d0e-49a7-931b-84f1e0467ad8` | 3072 | 3069 |
| FineWeb | 10,000,000 | 768 | Cosine | 3 | `a446f769-fa1d-4cdc-aba7-b8c978a433cb` | 3200 | 3200 |
| Wiki-Cohere | 35,000,000 | 768 | Dot | 2 | `b8ea5d4a-dc78-4528-8727-e9b3f3cfebc6` | 2500 | 2500 |
| DPR | 21,015,300 | 768 | Dot | 2 | `6f17e1d5-18cb-46f9-92db-da517f800840` | 1805 | 1805 |

DINO, LAION and FineWeb retain the checked archived indices used for the earlier
L2/cosine study. Wiki and DPR use fresh baseline indices on the pinned HF
snapshots: [Wiki-Cohere](https://huggingface.co/datasets/lance-format/wiki-cohere-35m/tree/cc3840e4e0c9091f26cdd0385fbb7a6ad81a7285)
and [DPR](https://huggingface.co/datasets/lance-format/dpr-wikipedia-single-nq/tree/d95c3715333a608fa5eb8ca6fe8e6bdb724f32bd).
Source file checksums, dataset versions, centroids, exact truth and splits are
retained with the raw query records.

The seed-2254 calibration sets remain unchanged. Three duplicate LAION evaluation
entries were removed before native evaluation: IDs 5128, 191 and 6105 (two
cross-split duplicates and one within-evaluation duplicate). All other evaluation
sets are unchanged. The eight-query baseline audit subsets are also unchanged.

Ground truth uses float64 source-vector distances. Equal-score cutoff IDs can
differ from the published truth; strict-ID recall counts them as misses. For
the three largest LAION discrepancies against published ground truth, recomputed source distances
showed only cutoff ties, with no strictly better published neighbor missing from
the generated truth. See [PROTOCOL.md](PROTOCOL.md) and [README.md](README.md) for
the complete measurement and audit procedure.

## Integrity and numerical checks

[run-integrity.json](run-integrity.json) records the final source, native-binary,
centroid and ground-truth hashes. All 18 checked files and
both binaries match the frozen run. The raw-record audit verifies disjoint query
vectors, exact group coverage, returned-ID uniqueness, filter membership,
returned counts, zero unfiltered storage I/O, baseline equivalence and matched
32-query comparisons.

The routing simulator and native search differ in scan counters for 153 of
384,092 unfiltered query runs. In 27 cases the budget differs by one partition,
with the affected centroid within four Float32 representable spacings of the
simulated threshold. Each of the remaining 126 row-count differences matches a
boundary-centroid exchange within seven spacings. These observations are
consistent with Float32 distance reduction and centroid ordering near ties;
exact native centroid order was not captured. Native counts are authoritative.

The largest Auto strict-ID recall shortfall relative to routing is 24.6 percentage
points for one LAION query at k=500. Recomputing source-vector distances shows
that all 123 missing truth IDs are tied with the worst returned distance within
6e-16. The three largest positive shortfalls on each of DINO, LAION, FineWeb and
DPR likewise contain cutoff ties; Wiki has no positive Auto shortfall. No recall
scores were adjusted. [numerical-audit.json](numerical-audit.json) records these
checks and the boundary diagnostics.

The full raw query records, splits, identities, diagnostic scripts and validation
logs are retained in the campaign archive. Large source datasets, indices,
prepared arrays and both native binaries are retained separately on the VM
volume. The checked-in CSVs contain all aggregate measurements.

## Validation

- The k=101 override regression reproduces on the baseline and passes on the
  candidate. Tests cover k=100/101/1000, L2/cosine/dot, multiple index segments,
  default recall, caller bounds, filtering and deletion.
- 314 Rust KNN tests and 20 Python Auto-probe tests pass. The Python cases pass
  on development and optimized builds, with every case below one second.
- Workspace Clippy/formatting, Python lint/type checks, benchmark-script lint
  and pre-commit checks pass.
- CI on the measured implementation commit passes. An unrelated stochastic
  IVF_HNSW_PQ recall test passed on rerun at the same commit.
- The benchmark integrity audit passes; the all-k recall gate retains the single
  unchanged FineWeb k=100 miss described above.
