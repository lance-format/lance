# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Measure fixed-batch sampling allocations and full-iteration throughput.

Run from python/ with:
    uv run python python/benchmarks/fixed_batch_sampling.py

Peak traced allocations are measured only through the first yield. Throughput
is measured separately without tracing and without retaining output batches.
"""

import argparse
import json
import random
import statistics
import time
import tracemalloc

from lance.sampler import ShardedFixedBatchSampler


def eager_batches(rows, batch_size, randomize):
    batches = [
        list(range(start, min(start + batch_size, rows)))
        for start in range(0, rows, batch_size)
    ]
    if randomize:
        random.seed(42)
        random.shuffle(batches)
    yield from batches


def measure(factory, repeats):
    tracemalloc.start()
    batches = iter(factory())
    try:
        next(batches, None)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        batches.close()
        tracemalloc.stop()

    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        rows = sum(len(batch) for batch in factory())
        elapsed.append(time.perf_counter() - start)
    seconds = statistics.median(elapsed)
    return {
        "first_yield_peak_bytes": peak,
        "full_iteration_seconds": seconds,
        "rows_per_second": rows / seconds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[10_000, 1_000_000])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.rows) < 0 or args.batch_size <= 0 or args.repeats <= 0:
        parser.error(
            "rows must be non-negative; batch-size and repeats must be positive"
        )

    for rows in args.rows:
        for randomize in (False, True):
            sampler = ShardedFixedBatchSampler(
                rank=0,
                world_size=1,
                total_num_rows=rows,
                batch_size=args.batch_size,
                randomize=randomize,
                seed=42,
            )
            for name, factory in (
                ("eager", lambda: eager_batches(rows, args.batch_size, randomize)),
                ("lazy", lambda: iter(sampler)),
            ):
                print(
                    json.dumps(
                        {
                            "implementation": name,
                            "rows": rows,
                            "batch_size": args.batch_size,
                            "randomize": randomize,
                            **measure(factory, args.repeats),
                        }
                    )
                )
