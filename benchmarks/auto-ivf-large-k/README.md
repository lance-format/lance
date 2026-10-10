# Auto IVF probing for k > 100

This experiment extends Auto's centroid-gap probing profiles beyond k=100.
The profiles apply to IVF partition selection across index types; IVF_FLAT
isolates routing recall from quantization and partition-local search losses.
[PROTOCOL.md](PROTOCOL.md) is the evaluation contract frozen before
calibration; [RESULTS.md](RESULTS.md) has the measurements and the decisions
they support.

Run all Python commands from `python/` after `make install`, on a benchmark VM
with enough memory to cache both baseline and candidate copies of the largest
IVF_FLAT index (256 GiB host memory for this campaign). Build the native extension
with the `release-with-debug` profile.

Keep a copy of the baseline `lance` package, including its native library, at
`$BASELINE_RUNTIME/lance` before rebuilding. Record its SHA256 in
`$STUDY/baseline-binary.sha256`; record the final candidate library SHA256 in
`$STUDY/candidate-binary.sha256`. The baseline worker and candidate each cache
the index, so allow memory for two copies (about 200 GiB total for Wiki).
After `make build`, build and run the optimized extension without an implicit
environment resync replacing it:

```bash
uv run --frozen --no-sync maturin develop --uv --profile release-with-debug
```

Lay out every corpus as `$STUDY/data/<name>/{base,queries}.lance` plus a
`VERIFIED` marker written only after checksum verification. The L2/cosine
corpora are the OSS-2221 archives with their frozen indices:

```bash
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/import_archive.py "$STUDY" "$INPUTS" dino-10m
```

`$INPUTS` is the `reproduction-inputs` directory of the OSS-2221 campaign
archive, which holds the original queries and published ground truth. The dot
corpora are pinned HF snapshots of `lance-format/wiki-cohere-35m` and
`lance-format/dpr-wikipedia-single-nq`, verified with their `SHA256SUMS`.

```bash
export LANCE_CPU_THREADS=32 RAYON_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/prepare.py "$STUDY" dino-10m
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/calibrate.py "$STUDY"
```

`prepare.py` builds (or adopts) a frozen IVF_FLAT index, computes exact float64
top-5000 ground truth for every query, maps every row to its partition and
saves sorted centroid distances, ground-truth partition ranks and cumulative
scanned rows. `calibrate.py` reads only calibration queries; it selects
profiles for k anchors 200, 500 and 1000, records the 2000 and 5000 optima, and
simulates the candidate bucket structures.

The published run uses [calibration-frozen.json](calibration-frozen.json). To
replay its policies, place that exact file at `$STUDY/calibration.json` after
restoring the matching frozen indices and prepared inputs. Freshly trained
indices define a new experiment. Source, binary and prepared-array hashes are
recorded in [run-integrity.json](run-integrity.json).

Verify the baseline first with `measure.py --limit 8` on main's binary: main's
Auto must match the candidate's bounded legacy arm for k > 100. Then install a
candidate that compiles the frozen profiles and run serially:

```bash
export LANCE_CPU_THREADS=16 RAYON_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 0-15 uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/measure.py "$STUDY" dino-10m --native --filtered --baseline-runtime "$BASELINE_RUNTIME"
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/audit.py "$STUDY"
```

Timing uses the first 512 held-out queries per arm with a rotating policy
order; all remaining held-out queries contribute recall and scan counts with
eight workers. CSVs keep every returned ID, timing, scan count and routing
prediction. These are empirical profiles for Float32 IVF_FLAT with about 4096
rows per partition, not recall guarantees for other indices or data.

`audit.py` writes `audited-results.json` before enforcing the recall gate. The
published run passes all integrity checks and all 20 newly supported unfiltered
groups, but exits with status 1 because the unchanged FineWeb k=100 control has
94.97875% mean recall against the preregistered 95% target. The target and frozen
parameters were not changed after evaluation. See [RESULTS.md](RESULTS.md) for
the full distributions, paired baseline comparisons and filtered limitations.

## Additional k=10,000 and k=100,000 experiment

[LARGE_K_RESULTS.md](LARGE_K_RESULTS.md) records the frozen parameters, complete
native comparisons, independent returned-ID audit, and numerical checks. All ten
Auto corpus/k groups pass the 95% held-out mean-recall target.

[LARGE_K_PROTOCOL.md](LARGE_K_PROTOCOL.md) defines the separate two-anchor
experiment. It reuses the original data and query split, preserves the completed
study, and writes memory-mappable top-100,000 truth and returned IDs to a new
directory. Run preparation once for every corpus before calibrating:

```bash
for corpus in dino-10m laion-10m fineweb-10m wiki-cohere-35m dpr-wikipedia-single-nq; do
  uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/prepare_large.py "$STUDY" "$LARGE_STUDY" "$corpus"
done
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/calibrate_large.py "$LARGE_STUDY"
cp "$LARGE_STUDY/calibration.json" "$LARGE_STUDY/calibration-frozen.json"
```

Build the experimental runtime in an isolated checkout of the measured source
(`0bcc82431`; the runtime at `12def694f` is identical). Preserve both the original
main runtime and the PR runtime before rebuilding. Apply only the
generated patch, which enables the two exact anchors and adds their profile and
policy-gate tests:

```bash
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/build_large_patch.py "$EXPERIMENT_CHECKOUT" "$LARGE_STUDY/calibration-frozen.json" "$LARGE_STUDY/experimental.patch"
```

Follow that checkout's environment setup, formatting, build and test instructions;
use `release-with-debug` for the native benchmark. Record the experimental
library SHA256 in `$LARGE_STUDY/candidate-binary.sha256` and copy the original
main hash to `$LARGE_STUDY/baseline-binary.sha256`. With the experimental runtime
selected and the same timing environment described above, run every corpus:

```bash
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/calibrate_large.py "$LARGE_STUDY" --evaluate
for corpus in dino-10m laion-10m fineweb-10m wiki-cohere-35m dpr-wikipedia-single-nq; do
  taskset -c 0-15 uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/measure_large.py "$LARGE_STUDY" "$corpus" --baseline-runtime "$BASELINE_RUNTIME"
done
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/audit_large.py "$LARGE_STUDY"
```

This experiment measures the requested anchors. The defaults in this PR continue to
use the previously validated range through k=1000.

## One fallback threshold per metric

[FALLBACK_RESULTS.md](FALLBACK_RESULTS.md) records the frozen constants,
complete all-k routing evaluation, and all 30 native held-out groups. Every
native candidate group meets the 95% mean-recall target, but a single constant
can still approach exhaustive scans. No production defaults are changed.

[FALLBACK_PROTOCOL.md](FALLBACK_PROTOCOL.md) defines a separate experiment with
one constant gap margin for each metric and no learned floor or cap. Calibration
and held-out routing checks cover every integer k from 1 through 100,000 using
the frozen exact-neighbor prefixes. A signed-multiplier comparison retains the
old f32 arithmetic and checks whether opposite nearest-distance signs rule out
a shared dot multiplier.

Use the prepared large-k study as immutable input and a fresh output directory:

```bash
uv run --frozen --no-sync pytest -q ../benchmarks/auto-ivf-large-k/test_fallback.py
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/calibrate_fallback.py "$LARGE_STUDY" "$FALLBACK_STUDY"
cp "$FALLBACK_STUDY/calibration.json" "$FALLBACK_STUDY/calibration-frozen.json"
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/calibrate_fallback.py "$LARGE_STUDY" "$FALLBACK_STUDY" --calibration-simulation
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/calibrate_fallback.py "$LARGE_STUDY" "$FALLBACK_STUDY" --evaluate
```

Build the constant-fallback experiment in an isolated checkout of the
PR runtime source, `0bcc82431` (also unchanged at `12def694f`). Preserve all earlier
runtimes before rebuilding, and use that checkout's environment, test, lint, and
`release-with-debug` instructions:

```bash
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/build_fallback_patch.py "$EXPERIMENT_CHECKOUT" "$FALLBACK_STUDY/calibration-frozen.json" "$FALLBACK_STUDY/experimental.patch"
```

Record the new native library hash in `$FALLBACK_STUDY/candidate-binary.sha256`.
For this experiment, the baseline is the preserved PR runtime
(`17bd3bb4...`), selected through `$FALLBACK_BASELINE_RUNTIME`; record that library
hash in `$FALLBACK_STUDY/baseline-binary.sha256`. Both processes explicitly request
the full maximum probe count to select the fallback policy even at small k.

With the experimental runtime selected and the timing environment above:

```bash
for corpus in dino-10m laion-10m fineweb-10m wiki-cohere-35m dpr-wikipedia-single-nq; do
  taskset -c 0-15 uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/measure_fallback.py "$FALLBACK_STUDY" "$corpus" --baseline-runtime "$FALLBACK_BASELINE_RUNTIME"
done
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/audit_fallback.py "$FALLBACK_STUDY"
```

Native validation uses the first 512 held-out queries at six representative k
values, with 128 serial timings and 32 matched baseline queries. The remaining
384 queries contribute recall and counters. Keep these populations distinct
from the full-query, all-k routing simulations. The experimental patch preserves
historical-index behavior and does not change the defaults in this PR.

## IVF_RQ 5bit recall

[RQ5_PROTOCOL.md](RQ5_PROTOCOL.md) freezes the comparison of FLAT Auto, RQ5
Auto, RQ5 full-partition search and the original RQ probing heuristic. The
default Auto profiles are unchanged. RQ5 indices reuse the original centroids
and exact row-to-partition assignments in separate shallow clones. Their
complete membership is verified before measurement.

After building and validating the candidate runtime, record its native-library
SHA256 in `$RQ5_STUDY/candidate-binary.sha256` and the preserved pre-change PR
runtime hash in `$RQ5_STUDY/baseline-binary.sha256`. Use the same repository
environment and `release-with-debug` profile described above:

```bash
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/rq5.py prepare "$RQ5_STUDY" all --source "$LARGE_STUDY"
taskset -c 0-15 uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/rq5.py measure "$RQ5_STUDY" all --baseline-runtime "$PR_BASELINE_RUNTIME"
uv run --frozen --no-sync python ../benchmarks/auto-ivf-large-k/rq5.py audit "$RQ5_STUDY" all
```

The 512-query held-out matrix covers k=1, 10, 100, 200, 500 and 1000. Recall
collection uses concurrent workers and does not establish latency benefits.
The full-partition RQ5 arm includes normal-mode quantized scoring and pruning;
its recall is not assumed to combine independently with FLAT routing recall.
