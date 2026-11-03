# Search effort recall tiers

`search_effort=0.1` and `0.25` target 80% and 90% average partition-routing
recall. The default `0.5` keeps the existing Auto budget and its nominal 95%
target. Zero and one retain their minimum/all-partitions meanings. Public
parameter types and precedence are unchanged; non-default efforts below 0.5
now use the calibrated anchors instead of the old single geometric segment.

The implementation scales the existing metric/k-dependent Auto budget and
rounds up to whole partitions. Each metric has two calibrated fractions:

| Metric | Effort 0.1 fraction | Effort 0.25 fraction |
| --- | ---: | ---: |
| L2 | 0.286 | 0.601 |
| Cosine | 0.278 | 0.563 |
| Dot | 0.161 | 0.441 |

Budgets interpolate geometrically between minimum, 80%, 90%, Auto and maximum
anchors. Explicit caller bounds constrain all anchors. The initial profile cap
does not prevent later probing when too few candidates pass a filter. Hamming
and k>100000 keep their existing uncalibrated interpolation.

## Calibration evidence

The [protocol](PROTOCOL.md) freezes selection and evaluation separately.
[calibration.json](calibration.json) records the selected fractions, exact input
hashes and calibration results. [routing-evaluation.json](routing-evaluation.json)
records all held-out results. A 0.001 fraction grid minimizes the budget subject
to 81%/91% mean coverage in every calibration corpus/k for the metric. There
are no dataset-specific runtime parameters and no new margin/floor/cap grid.

| Corpus | Metric | Rows | Calibration queries | Held-out queries | Evaluated k |
| --- | --- | ---: | ---: | ---: | --- |
| DINO | L2 | 10,000,000 | 2,048 | 4,096 | 1, 10, 100, 1,000 |
| LAION | Cosine | 10,000,000 | 2,048 | 4,093 | 1, 10, 100, 1,000 |
| FineWeb | Cosine | 10,000,000 | 2,048 | 4,096 | 1, 10, 100, 1,000 |
| Wiki-Cohere | Dot | 35,000,000 | 2,500 | 2,500 | 1, 10, 100 |
| DPR Wikipedia | Dot | 21,015,300 | 1,805 | 1,805 | 1, 10, 100 |

| Effort | Nominal target | Held-out routing recall range | Groups meeting target |
| --- | ---: | ---: | ---: |
| 0.1 | 80% | 81.033%–88.881% | 18 / 18 |
| 0.25 | 90% | 90.674%–94.413% | 18 / 18 |
| 0.5 | 95% | 94.189%–97.085% | 15 / 18 |

These are strict-ID routing predictions on frozen IVF_FLAT indices, not native
latency measurements or end-to-end RQ recall. The unchanged default misses the
nominal target on DINO k=1/10 and FineWeb k=1. Keep these misses visible.
Runtime support through k=100000 does not establish lower-tier recall for every
k outside this representative matrix, other partition counts or other corpora.

Three repeated LAION vectors are excluded before selection. FineWeb's 256
`reserve` queries are excluded from held-out evaluation. The first preparation
mistakenly admitted these reserve queries; the corrected artifact records the
original frozen hash and verifies unchanged calibration arrays and fractions.
No evaluation result was used to change a parameter. The published result uses
only the original `test` split.

## Reproduce

Use the repository's `python/` environment after `make install`. Restore and
verify these existing campaign archives; `prepare.py` reuses their centroids,
membership and exhaustive oracles rather than rebuilding the calibration matrix:

- OSS-2221: `s3://mmlb-us-east-1/rq-benchmark/results/20260913-auto-recall95/artifacts/ec504126d939/campaign.tar.gz`
  (SHA256 `a3db0628a4a21c1d21c2957458115fc279583369e7654fa7f8fa18d09772bf68`).
- OSS-2249: `s3://mmlb-us-east-1/rq-benchmark/results/oss2249-dot-auto-20260922/evidence.tar.gz`.

The OSS-2221 oracle verifier expects the archived `reproduction-inputs`
directories at their original `/home/ubuntu/` paths. On a fresh benchmark VM,
symlink each archived input directory there. Do not replace existing paths.

```bash
uv run --frozen --no-sync python ../benchmarks/auto-ivf-effort/prepare.py \
  "$RECALL95" "$DOT/evidence" "$STUDY/prepared"
uv run --frozen --no-sync python ../benchmarks/auto-ivf-effort/calibrate.py calibrate \
  "$STUDY/prepared/manifest.json" "$STUDY/calibration.json"
uv run --frozen --no-sync python ../benchmarks/auto-ivf-effort/calibrate.py evaluate \
  "$STUDY/prepared/manifest.json" "$STUDY/calibration.json" "$STUDY/evaluation.json"
```

The separate native experiment uses the frozen DINO S3 IVF_FLAT index and a
task-owned RQ5 clone. `prepare_native.py` verifies the original centroids and
creates the clone without changing source data. `native.py` starts separate
baseline/candidate workers, uses balanced arm order, records CPU/probes/I/O and
strict-ID recall, and checks that default-effort IDs remain identical. See the
protocol for the bounded k/refinement/filter matrix and cold-cache definition.
