# Dot-product Auto IVF probing

This experiment calibrates the initial IVF_FLAT probe budget on the complete
[Wiki-Cohere 35M](https://huggingface.co/datasets/lance-format/wiki-cohere-35m)
and [DPR Wikipedia](https://huggingface.co/datasets/lance-format/dpr-wikipedia-single-nq)
datasets. See [PROTOCOL.md](PROTOCOL.md) for the frozen evaluation contract.
The completed AWS measurements and parameter tradeoffs are in [RESULTS.md](RESULTS.md).

Run all Python commands from `python/` after the repository's `make install`
setup, on a benchmark VM with enough memory to cache one complete IVF_FLAT
index (at least 256 GiB for these datasets). Build the native extension with the
repository's `release-with-debug` profile. Keep raw corpus and query norms.

Prepare a working copy of each HF snapshot under
`$STUDY/data/<dataset-name>/`, pin its revision, verify its `SHA256SUMS`, and
create a `VERIFIED` marker only after every checksum passes. Copies must preserve
row order. Immutable data files can be hard-linked; manifests and version hints
must be independent copies because index creation writes dataset metadata.

```bash
export STUDY=/path/to/study
export LANCE_CPU_THREADS=32 RAYON_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16
export LANCE_USE_HNSW_SPEEDUP_INDEXING=auto
uv run python ../benchmarks/auto-ivf-dot/prepare.py "$STUDY" wiki-cohere-35m
uv run python ../benchmarks/auto-ivf-dot/prepare.py "$STUDY" dpr-wikipedia-single-nq
uv run python ../benchmarks/auto-ivf-dot/calibrate.py "$STUDY"
```

`prepare.py` builds a frozen IVF_FLAT index, maps every indexed row to its actual
partition, and saves centroid scores, ground-truth partition ranks, partition
sizes, and diagnostic projections. `calibrate.py` reads only the calibration
half of the official queries, selects floors/caps/margins, and freezes the
chosen score normalization before independent evaluation.

Run `measure.py` first with the baseline binary and `--limit 8` to verify that
explicitly bounded Auto with a full-partition maximum produces identical results
and scan counts to main's unbounded Auto. The `legacy` timing arm uses that
existing bounded policy on the candidate binary so it can be interleaved with
candidate Auto and fixed budgets under the same cache and process conditions.
The `tuned` arm uses the per-corpus profile frozen during calibration, within
the selected gap family, through the three Auto environment overrides.

After installing a candidate implementing the **frozen** profile, run:

```bash
export LANCE_CPU_THREADS=16 RAYON_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run python ../benchmarks/auto-ivf-dot/measure.py "$STUDY" wiki-cohere-35m --native
uv run python ../benchmarks/auto-ivf-dot/measure.py "$STUDY" dpr-wikipedia-single-nq --native
uv run python ../benchmarks/auto-ivf-dot/audit.py "$STUDY"
```

Timing runs must be serial with no other compilation or benchmark workload.
Use the same CPU affinity for every run. By default Auto and all selected fixed
budgets evaluate every held-out query. Their matched latency distributions use
the first 512 held-out queries (`--timing-queries`).
The legacy diagnostic uses 32 queries (`--legacy-queries`) because its full scans
are expensive; its percentiles are descriptive only. `--limit`
is for smoke checks or a clearly labeled subset. CSVs retain
neighbor IDs, timings, recall, scan counts, I/O, and routing predictions for each
query. JSON summaries include percentiles, prediction mismatches, and binary
identity. Inspect mismatches and source-score ties before drawing conclusions.
After each serial timing phase, remaining recall-only queries use eight workers
(`--recall-workers`); their elapsed times are excluded from performance summaries.
Every worker finishes before the next serial timing phase.

The tuned 96% calibration target leaves headroom for an independent 95% recall
target. Neither the profile nor the grid search guarantees recall on arbitrary
indices, partition counts, query distributions, or other datasets.
