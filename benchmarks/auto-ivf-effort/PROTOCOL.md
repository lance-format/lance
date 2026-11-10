# Calibrated low search effort

Expanded-study baseline: `aff6f1d1a8df56dc72f6b3ab1498086e0688125c`. Preserve the
released parameter range, explicit probe bounds, effort-zero/one endpoints,
and the default 0.5 Auto policy. No index-format change is needed.

## Selection

Select fractions of the existing metric/k-dependent Auto budget. Fit a separate
fraction for each metric, k anchor and recall tier, and interpolate fractions
geometrically in log(k). The default Auto profiles remain unchanged. Include
adjacent anchors on both sides of every Auto bucket boundary so interpolation
does not cross a change in the underlying profile.
Candidate anchors are effort 0.1 with an 80% routing-recall target and effort
0.25 with a 90% target; 0.5 retains the original nominal 95% Auto target.
Interpolate budgets geometrically between anchors. Fractions must be ordered,
and each query's budget must be monotone in effort, including explicit bounds.

Reuse the original DINO 10M L2, LAION 10M cosine, FineWeb 10M cosine,
Wiki-Cohere 35M dot and DPR 21M dot corpora, frozen centroids and membership.
Keep their original calibration/held-out splits. Remove duplicate query vectors
before selection, with calibration taking precedence. Compute strict-ID partition
coverage. Treat tied IDs as misses rather than relaxing the oracle.

The initial study's metric-wide coefficients and limited-k results are retained
as the previous implementation and its evidence. They do not establish the
expanded study's parameters or quality. This extension was requested before its
calibration or evaluation on 2026-11-04.

Extend exact source-vector ground truth to top-100000 for every admitted query
on all five corpora. Use Float64 distances, preserving original metric semantics,
row identities and query vectors. Check its prefix against the archived oracle;
investigate numerical or cutoff-tie differences without changing strict-ID
recall. Preserve the original calibration/held-out split and duplicate/reserve
exclusions. If an index must be rebuilt because its original files are no longer
available, record the new version, centroids and actual partition membership and
use that identical index for both native binaries.

The archived L2/cosine reference is an exhaustive native Float32 oracle. When
its selected neighbors differ in Float64 source-vector score by more than
1e-10, check the worst query using a separate exhaustive Float64 implementation:
direct per-vector formulas and a full global sort. Require its top-100000 score
sequence to agree with the streaming oracle within Float64 numerical tolerance
(rtol=atol=1e-12). Preserve old-reference and tie/ID differences; this check does
not replace IDs or relax the strict-ID recall metric. It validates the oracle,
independently of probe-profile fitting.

Calibrate and evaluate k=1, 2, 10, 11, 100, 101, 200, 201, 500, 501, 1000,
1001, 2000, 5000, 10000, 10001, 20000, 50000 and 100000. These cover every bucket
endpoint, every lower boundary, and interior points of the two wide buckets.
These 19 k values are mandatory anchors. Also prepare k=5, 20, 50, 150, 300,
750, 1500, 3000, 7500, 15000, 30000 and 75000 as calibration checks of
interpolation. Fit a pointwise reference at all 31 values: choose the smallest
fraction on the 0.001 grid reaching 81% or 91% mean routing recall in every
calibration corpus of that metric at that k.

Begin with the mandatory anchors. Add a check point when interpolation misses
either calibration target or uses more than 5% extra mean partitions relative
to the pointwise reference on any calibration corpus/tier. Repeat until all
calibration constraints hold. All metrics share the selected k anchors; each
has its own fractions. Select knots and fractions only from calibration data,
then freeze them and evaluate all 31 k values on held-out queries. The earlier
eight-bucket candidate is retained separately: its calibration results already
showed excessive work at the lower ends of the two wide k buckets.
Hash inputs and save the frozen profiles before evaluation. Retain any held-out
miss without retuning on evaluation data. Report the calibration coverage;
runtime support outside this matrix does not establish recall there.

## Native product paths

Use the same S3 dataset, frozen version, machine, optimized build profile and
thread configuration for both binaries. Compare the unchanged baseline default,
candidate default, and each candidate tier. Include the baseline at the same
effort values so changes to the old geometric mapping remain visible. For the
expanded study, the direct-parent baseline is the original metric-wide policy
at `aff6f1d1a8df56dc72f6b3ab1498086e0688125c`. Its product code matches the archived
measured runtime `daf655a75bd3cceedb38b5b198b7e2fd143a30b5`. Keep the initial
study against `b1722c05` clearly identified as historical evidence.

Run native IVF_FLAT queries at all eight bucket endpoints plus k=1001 and
k=10001 on every corpus, including k=100000 for all metrics. Use 128 held-out queries for recall and 64
for timing, with three balanced timed passes. Preserve returned IDs in binary
arrays at large k and independently recompute recall. Record cache capacities
and nonzero reads; a prewarm request does not prove the full index fits in cache.
The expanded matrix uses 112 GiB of index-cache capacity per worker, sufficient
for each individual index's stored vectors. Record actual resident memory after
prewarm and query I/O; serialized size alone does not establish memory use or
cache residency. Start native execution with the largest corpus so memory
capacity is exercised before collecting the remaining corpora. Cold sessions
have zero index-cache capacity. Both binaries always use the same settings.
An additional DINO RQ5 diagnostic without refinement at k=100000 uses 16 held-out
queries in one timed pass per arm. Report it separately from the full FLAT
matrix; it checks the large-k quantized path and does not replace its
128-query recall measurements.

Use the archived 10M-row DINO IVF_FLAT index for an exact-scoring routing control.
Build a separate RQ5 index on the same data and centroids for the product path;
preserve source data and measure actual RQ partition membership. Cover k=10 and
k=1000, refinement off and refinement=2, plus a selective prefilter at k=10.
The filtered oracle is obtained from the exhaustive top-4096 list only when it
contains at least k eligible neighbors. Otherwise compute an exhaustive filtered
oracle; do not call an underfilled reference exact. Include a cosine control if
the first workload shows a material metric-specific implementation concern.

Freeze query IDs before timing. Use a bounded matched subset: at least 128
held-out queries for recall and 64 for serial latency, with balanced arm order.
Run three passes for timed queries. Report per-query strict-ID Recall@k,
latency, process CPU time, searched partitions, comparisons, storage bytes and
requests. Report throughput from measured total wall time. Keep compilation and
preparation out of timing. Record the native-library and source hashes.

Measure both a warmed index cache and a bounded cold-cache S3 sample. A cold
sample uses a fresh dataset/session for every arm and query; it is not a claim
about the S3 service's internal cache. Count actual reads rather than infer I/O
from partitions. Include returned rows and filter membership checks.

Routing targets are empirical averages on unfiltered calibration data, not
per-query guarantees. Quantization, pruning, refinement, HNSW search, filter
selectivity and caller bounds can change end-to-end Recall@k. Do not attribute
older campaign timings to this PR. Report misses and cost/quality tradeoffs,
including cases where extra probing does not improve final recall.

## Delivery

Run targeted budget/API and FLAT/RQ integration tests, required Rust formatting
and workspace Clippy, and required Python lint. Inspect the integrated diff and
challenge monotonicity at fractional boundaries and under explicit min/max.
Deliver a draft PR with exact-head evidence. Release, SDK/Sophon dependency
adoption and integrated acceptance remain explicit downstream steps.
