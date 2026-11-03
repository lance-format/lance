# Calibrated low search effort

Baseline: `b1722c05f29a4cac2a2f76629b77dd6438853628`. Preserve the
released parameter range, explicit probe bounds, effort-zero/one endpoints,
and the default 0.5 Auto policy. No index-format change is needed.

## Selection

Select fractions of the existing metric/k-dependent Auto budget. Use one fraction
per metric and recall tier, rather than retuning a second margin/floor/cap grid.
Candidate anchors are effort 0.1 with an 80% routing-recall target and effort
0.25 with a 90% target; 0.5 retains the original nominal 95% Auto target.
Interpolate budgets geometrically between anchors. Fractions must be ordered,
and each query's budget must be monotone in effort, including explicit bounds.

Reuse the original DINO 10M L2, LAION 10M cosine, FineWeb 10M cosine,
Wiki-Cohere 35M dot and DPR 21M dot corpora, frozen centroids and membership.
Keep their original calibration/held-out splits. Remove duplicate query vectors
before selection, with calibration taking precedence. Compute strict-ID partition
coverage. Treat tied IDs as misses rather than relaxing the oracle.

Calibrate at k=1/10/100 for all five corpora and k=1000 for DINO/LAION/FineWeb,
using their archived exhaustive top-4096 oracle. Dot's archived oracle covers
only k<=100. Choose the smallest fraction on the 0.001 grid that reaches 81% or
91% mean routing recall in **every** calibration corpus/k for that metric.
Hash inputs and save the frozen profiles before evaluation. Retain any held-out
miss without retuning on evaluation data. Report the calibration coverage;
runtime support outside this matrix does not establish recall there.

## Native product paths

Use the same S3 dataset, frozen version, machine, optimized build profile and
thread configuration for both binaries. Compare the unchanged baseline default,
candidate default, and each candidate tier. Include the baseline at the same
effort values so changes to the old geometric mapping remain visible.

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
