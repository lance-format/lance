# Duplicate-pair CPU, I/O and memory benchmark

This Linux profiler exercises the complete native partition-pair stream on
1,536-dimensional cosine IVF_RQ5/RQ8 indices with 6,408 and 17,783 rows per
partition. The fixtures are synthetic, with deterministic duplicate groups.
They match the partition dimensions/cardinalities in issue #9535, not the full
production table, centroid distribution, or distributed execution environment.

## Prepare once, then measure read-only

Follow `python/AGENTS.md` to install the project. Use the same repository
`release-with-debug` profile, toolchain and environment for baseline and candidate.
Run from `python/`:

```sh
uv run maturin develop --profile release-with-debug --uv
uv run python python/benchmarks/duplicate_pairs.py prepare \
  --prefix s3://YOUR-BUCKET/dedup/UNIQUE_RUN_ID --output cases.json
```

`prepare` creates new tables and indices and refuses to overwrite tables. Never
point it at production data. Keep the generated `cases.json`: all comparisons
must reuse these exact persisted dataset versions and segment IDs, without
rebuilding the indices between implementations.

```sh
LANCE_CPU_THREADS=8 RAYON_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 0-7 uv run --no-sync python python/benchmarks/duplicate_pairs.py run \
  --config cases.json --case 0 --threshold 0.01 --label candidate \
  --memory-limit 268435456 --max-concurrency 8 --output result.json
```

Use cases 0..3 and thresholds 0.01 (sparse output) and 1e9 (all pairs; any
threshold at or above `ALL_PAIRS_THRESHOLD` asserts N*(N-1)/2 output pairs).
Omit the two resource arguments when running a baseline that predates them.
Each invocation is a fresh process, fully consumes the stream, and asserts zero
writes on the source object store. Run only one measured process at a time, with no concurrent builds.
The optional global `--credentials` argument reads an AWS credential-process
JSON file; credential values are never written to results. Normal environment
credentials work without that argument.

## What the results mean

- `elapsed_s` includes native planning, code staging, quantizer batch scoring and
  streaming output hashing. Imports and opening the pinned dataset are excluded;
  opening is reported separately as `dataset_open_s`. Table/index creation is
  not timed.
- Three independent SHA-256 hashes cover the ordered row-ID and float32-distance
  columns; they depend on output order but not on batch boundaries. Output is
  in tile order (vector batch pairs, row-major within a tile), which differs
  from implementations that kept each `row_id_a` contiguous. Compare
  `pair_set_hash` (an order-independent hash of the unordered ID pairs) and
  `output_rows` between implementations; compare the ordered hashes only
  between runs of one implementation, for example across `--max-concurrency`
  and `--memory-limit`, where they must match exactly.
- Cosine distances are `1 - cos` of the renormalized index reconstructions, in
  [0, 2]. Compare distance hashes only between implementations with the same
  scoring semantics; versions that reported normalized L2 / 2 for quantized
  cosine differ.
- `cpu_cores` is process user+system CPU time divided by wall time. It includes
  the Python consumer and monitoring overhead, not just Rust scoring threads.
- RSS is for the whole process, including Python, shared libraries and native
  allocations. `peak_rss_bytes` is Linux's process high-water mark; it is not the
  code staging budget. Each run also records its initial RSS.
- Source bytes/IOPS come from Lance's object-store counters after dataset open.
  They are distinct from temporary-file I/O. The last observed source read and
  peak source rate use approximately 100 ms sampling, not exact span timing.
- `disk_read_bytes`/`disk_write_bytes` are Linux process storage-I/O accounting.
  `process_rchar`/`process_wchar` include logical syscall traffic, including
  page-cache reads, monitoring and IPC; do not label them physical disk traffic
  or exclusively spill traffic. Buffered write accounting is not an instantaneous
  device-throughput measurement.
- Host network counters include TLS and any background/control traffic. Source
  object-store counters are the authoritative bytes attributed to the dataset.

The adjacent `.trace.json` file contains the time series, so short I/O preparation
can be distinguished from a long compute phase. Report repetitions and statistics
explicitly; a single long scan must not be described as a median.

## Resource diagnostics

Also compare `--max-concurrency 1` to 8 and force code spill with
`--memory-limit 0`. The latter changes the storage/memory tradeoff; OS file
cache is not included in process RSS and may retain the code spill pages.
