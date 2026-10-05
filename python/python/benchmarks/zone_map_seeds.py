#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""
Benchmark: Zone Map Index Seeds vs. No Seeds

Measures the time to update a zone map index when new data is appended to an
already-indexed dataset, with and without use_seeds.

Seeds allow Lance to embed per-zone min/max statistics in data files at write
time.  When optimize_indices() runs, it harvests those pre-computed stats
instead of re-scanning the full column, which eliminates the read I/O for
wide data types.

WHY the workflow matters
------------------------
Seeds only engage when merging new unindexed fragments into an existing
non-empty index segment.  A first optimize_indices() call on a freshly
created empty-index always does a full rebuild (no old fragments to merge
against), so seeds would never be used.

Correct workflow:
  1. Write an initial batch of data and build the zone map index on it.
     This creates a non-empty indexed segment; all subsequent writes will
     embed seed buffers in their data files (when use_seeds=True).
  2. Append the bulk of the data in multiple batches.
  3. optimize_indices() — harvests seeds (phase 2 data) and merges them into
     the segment from phase 1.  Without seeds, phase 3 must re-read all of
     the phase-2 data from disk.

Phases timed:
  initial_write   write seed_fraction of total rows + create/build index
  ingest          append remaining rows (seeds collected here when enabled)
  update_index    optimize_indices() — seed harvest happens here
  total           sum of the three phases above

Three scenarios:
  int64                    10 M rows   ~80 MB on disk
  FixedSizeBinary(4096)    10 M rows   ~38 GB on disk
  LargeBinary(20 KiB)       1 M rows   ~19 GB on disk

Usage:
    uv run python python/benchmarks/zone_map_seeds.py
    uv run python python/benchmarks/zone_map_seeds.py --tmpdir /fast/nvme/bench
    uv run python python/benchmarks/zone_map_seeds.py --scale 0.01
"""

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.indices import IndexConfig

KiB = 1024

# Fraction of total rows written in the initial indexed batch.
SEED_FRACTION = 0.1


# ---------------------------------------------------------------------------
# Data generators (yield pa.Table one batch at a time)
# ---------------------------------------------------------------------------


def gen_int_batches(num_rows: int, batch_size: int):
    rng = np.random.default_rng()
    for start in range(0, num_rows, batch_size):
        n = min(batch_size, num_rows - start)
        yield pa.table({"value": pa.array(rng.integers(0, 2**31, n, dtype=np.int64))})


def gen_fsb_batches(num_rows: int, blob_bytes: int, batch_size: int):
    for start in range(0, num_rows, batch_size):
        n = min(batch_size, num_rows - start)
        raw = os.urandom(n * blob_bytes)
        arr = pa.FixedSizeBinaryArray.from_buffers(
            pa.binary(blob_bytes), n, [None, pa.py_buffer(raw)]
        )
        yield pa.table({"value": arr})


def gen_large_binary_batches(num_rows: int, blob_bytes: int, batch_size: int):
    for start in range(0, num_rows, batch_size):
        n = min(batch_size, num_rows - start)
        yield pa.table(
            {
                "value": pa.array(
                    [os.urandom(blob_bytes) for _ in range(n)],
                    type=pa.large_binary(),
                )
            }
        )


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def run_scenario(
    dataset_path: Path,
    schema: pa.Schema,
    seed_rows: int,
    seed_batch_size: int,
    gen_seed_batches,  # () -> Iterator[pa.Table] for the initial write
    gen_bulk_batches,  # () -> Iterator[pa.Table] for the appended data
    use_seeds: bool,
) -> dict:
    """
    Run one full scenario and return per-phase timings.

    Seeds are only engaged by optimize_indices() when there is at least one
    previously indexed fragment.  This scenario ensures that by writing
    seed_rows of data and building the index before any bulk ingest.
    """
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    index_config = IndexConfig(
        index_type="ZONEMAP", parameters={"use_seeds": use_seeds}
    )
    timings = {}

    # Phase 1: initial write + index build
    t0 = time.perf_counter()
    ds = None
    for batch in gen_seed_batches():
        if ds is None:
            ds = lance.write_dataset(batch, dataset_path, schema=schema)
        else:
            ds = lance.write_dataset(batch, dataset_path, mode="append")
    ds.create_scalar_index("value", index_type=index_config, replace=True)
    timings["initial_write"] = time.perf_counter() - t0

    # Phase 2: bulk ingest — seeds are collected per-batch when enabled
    t0 = time.perf_counter()
    for batch in gen_bulk_batches():
        ds = lance.write_dataset(batch, dataset_path, mode="append")
    timings["ingest"] = time.perf_counter() - t0

    # Phase 3: index update — seed harvest happens here
    t0 = time.perf_counter()
    ds.optimize.optimize_indices()
    timings["update_index"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    shutil.rmtree(dataset_path)
    return timings


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_comparison(title: str, no_seeds: dict, with_seeds: dict) -> None:
    phases = ["initial_write", "ingest", "update_index", "total"]
    col_w = 13

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(
        f"  {'Phase':<20} {'No Seeds':>{col_w}} "
        + f"{'With Seeds':>{col_w}} {'Speedup':>{col_w}}"
    )
    print(f"  {'-' * (20 + col_w * 3 + 6)}")
    for phase in phases:
        t_no = no_seeds[phase]
        t_yes = with_seeds[phase]
        speedup = t_no / t_yes if t_yes > 0 else float("inf")
        marker = "* " if phase == "total" else "  "
        print(
            f"{marker}{phase:<20} {t_no:>{col_w}.2f}s"
            f" {t_yes:>{col_w}.2f}s"
            f" {speedup:>{col_w - 1}.2f}x"
        )


# ---------------------------------------------------------------------------
# Per-type benchmark wrappers
# ---------------------------------------------------------------------------


def bench_integers(base: Path, num_rows: int) -> None:
    seed_rows = max(1, int(num_rows * SEED_FRACTION))
    bulk_rows = num_rows - seed_rows
    # ~400 KB per seed batch; ~4 MB per bulk batch → ~20 bulk fragments
    seed_bs = max(1, seed_rows // 5)
    bulk_bs = max(1, bulk_rows // 20)
    print(
        f"\nInteger (int64): {num_rows:,} rows "
        f"(initial={seed_rows:,}, bulk={bulk_rows:,}) ..."
    )
    schema = pa.schema([pa.field("value", pa.int64())])

    no_seeds = run_scenario(
        base / "int_no_seeds",
        schema,
        seed_rows,
        seed_bs,
        lambda: gen_int_batches(seed_rows, seed_bs),
        lambda: gen_int_batches(bulk_rows, bulk_bs),
        use_seeds=False,
    )
    with_seeds = run_scenario(
        base / "int_with_seeds",
        schema,
        seed_rows,
        seed_bs,
        lambda: gen_int_batches(seed_rows, seed_bs),
        lambda: gen_int_batches(bulk_rows, bulk_bs),
        use_seeds=True,
    )
    print_comparison(f"int64  —  {num_rows:,} rows", no_seeds, with_seeds)


def bench_vectors(base: Path, num_rows: int) -> None:
    blob_bytes = 4 * KiB
    seed_rows = max(1, int(num_rows * SEED_FRACTION))
    bulk_rows = num_rows - seed_rows
    total_gb = num_rows * blob_bytes / (1024**3)
    # ~200 MB per batch to stay memory-friendly
    seed_bs = max(1, (200 * KiB * KiB) // blob_bytes)
    bulk_bs = seed_bs
    print(
        f"\nVector (FixedSizeBinary {blob_bytes // KiB} KiB): {num_rows:,} rows"
        f"  (~{total_gb:.1f} GB)  (initial={seed_rows:,}, bulk={bulk_rows:,}) ..."
    )
    schema = pa.schema([pa.field("value", pa.binary(blob_bytes))])

    no_seeds = run_scenario(
        base / "vec_no_seeds",
        schema,
        seed_rows,
        seed_bs,
        lambda: gen_fsb_batches(seed_rows, blob_bytes, seed_bs),
        lambda: gen_fsb_batches(bulk_rows, blob_bytes, bulk_bs),
        use_seeds=False,
    )
    with_seeds = run_scenario(
        base / "vec_with_seeds",
        schema,
        seed_rows,
        seed_bs,
        lambda: gen_fsb_batches(seed_rows, blob_bytes, seed_bs),
        lambda: gen_fsb_batches(bulk_rows, blob_bytes, bulk_bs),
        use_seeds=True,
    )
    print_comparison(
        f"FixedSizeBinary({blob_bytes // KiB} KiB)  —  {num_rows:,} rows",
        no_seeds,
        with_seeds,
    )


def bench_large_binary(base: Path, num_rows: int) -> None:
    blob_bytes = 20 * KiB
    seed_rows = max(1, int(num_rows * SEED_FRACTION))
    bulk_rows = num_rows - seed_rows
    total_gb = num_rows * blob_bytes / (1024**3)
    # ~200 MB per batch
    seed_bs = max(1, (200 * KiB * KiB) // blob_bytes)
    bulk_bs = seed_bs
    print(
        f"\nLargeBinary ({blob_bytes // KiB} KiB blobs): {num_rows:,} rows"
        f"  (~{total_gb:.1f} GB)  (initial={seed_rows:,}, bulk={bulk_rows:,}) ..."
    )
    schema = pa.schema([pa.field("value", pa.large_binary())])

    no_seeds = run_scenario(
        base / "bin_no_seeds",
        schema,
        seed_rows,
        seed_bs,
        lambda: gen_large_binary_batches(seed_rows, blob_bytes, seed_bs),
        lambda: gen_large_binary_batches(bulk_rows, blob_bytes, bulk_bs),
        use_seeds=False,
    )
    with_seeds = run_scenario(
        base / "bin_with_seeds",
        schema,
        seed_rows,
        seed_bs,
        lambda: gen_large_binary_batches(seed_rows, blob_bytes, seed_bs),
        lambda: gen_large_binary_batches(bulk_rows, blob_bytes, bulk_bs),
        use_seeds=True,
    )
    print_comparison(
        f"LargeBinary({blob_bytes // KiB} KiB)  —  {num_rows:,} rows",
        no_seeds,
        with_seeds,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tmpdir",
        type=Path,
        default=None,
        help="Directory for benchmark datasets (default: system temp dir)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Row-count scale factor (default 1.0; use 0.01 for a quick smoke test)",
    )
    args = parser.parse_args()

    int_rows = max(2, int(10_000_000 * args.scale))
    vec_rows = max(2, int(10_000_000 * args.scale))
    bin_rows = max(2, int(1_000_000 * args.scale))

    print(
        f"Scale: {args.scale}  →  "
        f"{int_rows:,} int / {vec_rows:,} vector / {bin_rows:,} binary rows"
    )

    def run(tmp_dir: Path) -> None:
        bench_integers(tmp_dir, int_rows)
        bench_vectors(tmp_dir, vec_rows)
        bench_large_binary(tmp_dir, bin_rows)

    if args.tmpdir is not None:
        args.tmpdir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="lance_zonemap_bench_", dir=args.tmpdir
        ) as tmp:
            run(Path(tmp))
    else:
        with tempfile.TemporaryDirectory(prefix="lance_zonemap_bench_") as tmp:
            run(Path(tmp))


if __name__ == "__main__":
    main()
