# OSS-2254: adaptive IVF probing through k=1000

This change extends the learned initial probe budget from k <= 100 to k <= 1000
for finite Float32 IVF_FLAT queries using L2, cosine, or dot. Three new buckets
end at k=200, 500, and 1000. Caller minimums can exceed the learned initial cap;
late probing remains available when filters or deletions leave too few results.

**Native held-out measurements are pending. No latency improvement or held-out
recall result is claimed in this checkpoint.** The tables below describe frozen
calibration choices, not runtime performance. The completed report will replace
this status after the native measurements and independent audit finish.

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
- Baseline native SHA256:
  `9c5b08417d6dfe7b319039b80b02a876b1e6d8078fe2820c46319150f359bcab`.
- AWS r8i.8xlarge, Intel Xeon 6975P-C, 32 vCPU / 256 GiB RAM, 1600 GB gp3.
- Rust 1.98.1 and repository `release-with-debug` profile for both measured
  binaries. Cargo lockfiles match the pinned source in all three workspaces.
- One query at a time, affinity 0–15, Lance/Rayon threads 16, BLAS/OMP threads 1,
  query_parallelism=1. Both processes prewarm the same frozen index. Every timed
  query must report zero storage bytes read. CPU-intensive preparation and
  compilation finish before timing starts.
- Candidate Auto, tuned and fixed policies time the first 512 held-out queries.
  Actual baseline Auto runs on the first 32 of those queries in a separate
  process, interleaved with the candidate; IPC is outside the measured interval.
  Before/after ratios use only those matched 32 queries. Remaining held-out
  queries contribute recall and scan counts, not latency.
- Exact fixed20 is always included, alongside the calibration-selected nearby
  fixed budgets. Recall and returned counts accompany latency comparisons.
  The full mean/p90/p95/p99/max distributions are retained per corpus and k.

| Corpus | Rows | Metric | Dataset version | Index UUID | Calibration queries | Held-out queries |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| DINO | 10,000,000 | L2 | 2 | `a2449e25-2c49-41f9-b3e2-3a3969491f08` | 3072 | 3072 |
| LAION | 10,000,000 | Cosine | 2 | `28448469-7d0e-49a7-931b-84f1e0467ad8` | 3072 | 3069 |
| FineWeb | 10,000,000 | Cosine | 3 | `a446f769-fa1d-4cdc-aba7-b8c978a433cb` | 3200 | 3200 |
| Wiki-Cohere | 35,000,000 | Dot | 2 | `b8ea5d4a-dc78-4528-8727-e9b3f3cfebc6` | 2500 | 2500 |
| DPR | 21,015,300 | Dot | 2 | `6f17e1d5-18cb-46f9-92db-da517f800840` | 1805 | 1805 |

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
the three largest LAION discrepancies, independently recomputed source distances
showed only cutoff ties, with no strictly better published neighbor missing from
the generated truth. See [PROTOCOL.md](PROTOCOL.md) and [README.md](README.md) for
the complete measurement and audit procedure.
