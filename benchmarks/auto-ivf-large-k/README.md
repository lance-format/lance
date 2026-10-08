# Auto IVF probing for k > 100

This experiment extends Auto's centroid-gap probing profiles beyond k=100.
[PROTOCOL.md](PROTOCOL.md) is the evaluation contract frozen before
calibration; [RESULTS.md](RESULTS.md) has the measurements and the decisions
they support.

Run all Python commands from `python/` after `make install`, on a benchmark VM
with enough memory to cache one complete IVF_FLAT index (256 GiB for these
corpora). Build the native extension with the `release-with-debug` profile.

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
uv run python ../benchmarks/auto-ivf-large-k/import_archive.py "$STUDY" "$INPUTS" dino-10m
```

`$INPUTS` is the `reproduction-inputs` directory of the OSS-2221 campaign
archive, which holds the original queries and published ground truth. The dot
corpora are pinned HF snapshots of `lance-format/wiki-cohere-35m` and
`lance-format/dpr-wikipedia-single-nq`, verified with their `SHA256SUMS`.

```bash
export LANCE_CPU_THREADS=32 RAYON_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32
uv run python ../benchmarks/auto-ivf-large-k/prepare.py "$STUDY" dino-10m
uv run python ../benchmarks/auto-ivf-large-k/calibrate.py "$STUDY"
```

`prepare.py` builds (or adopts) a frozen IVF_FLAT index, computes exact float64
top-5000 ground truth for every query, maps every row to its partition and
saves sorted centroid distances, ground-truth partition ranks and cumulative
scanned rows. `calibrate.py` reads only calibration queries; it selects
profiles for k anchors 200, 500 and 1000, records the 2000 and 5000 optima, and
simulates the candidate bucket structures.

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
