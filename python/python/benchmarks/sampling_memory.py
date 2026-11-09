# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Compare traced sampling allocations with the former NumPy choice path.

Run from python/ with:
    uv run python python/benchmarks/sampling_memory.py

The virtual dataset allocates no population data. Measurements cover Python and
NumPy allocations, including the selected IDs, rather than whole-process RSS.
"""

import argparse
import json
import tracemalloc

import numpy as np
import pyarrow as pa
from lance.sampler import maybe_sample


class VirtualDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return self.rows

    def take(self, indices, columns=None):
        return pa.table({"id": pa.array(indices, type=pa.int64())})


def measure(rows, n, legacy):
    dataset = VirtualDataset(rows)
    np.random.seed(1618)
    tracemalloc.start()
    try:
        if legacy:
            indices = np.random.choice(rows, n, replace=False)
            for start in range(0, n, 32):
                dataset.take(indices[start : start + 32])
        else:
            for _ in maybe_sample(
                dataset, n, "id", batch_size=32, max_takes=max(n, 2048)
            ):
                pass
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 10_000_000])
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[32, 512, 2048])
    args = parser.parse_args()
    # Initialize Arrow's Python integration before tracing allocations.
    pa.table({"id": pa.array([0], type=pa.int64())})
    for rows in args.rows:
        for n in args.sample_sizes:
            if not 0 <= n < rows:
                parser.error(
                    f"expected 0 <= sample size < rows, got n={n}, rows={rows}"
                )
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "sample_size": n,
                        "legacy_peak_bytes": measure(rows, n, legacy=True),
                        "peak_bytes": measure(rows, n, legacy=False),
                    }
                )
            )
