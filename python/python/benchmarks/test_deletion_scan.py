# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Scans of a fragment that carries a deletion vector.

Every row a scan reads has to be resolved against the deletion vector, so deleting
even a handful of rows changes what a full scan costs.  These compare a clean
fragment against one with ten rows removed, laid out both as one run and scattered.

A single process can load only one native pylance.  To compare official main
against this change, install each wheel into its own interpreter and run this
file twice:

    base/bin/python -m pytest python/python/benchmarks/test_deletion_scan.py
    patched/bin/python -m pytest python/python/benchmarks/test_deletion_scan.py

Read the Min column.  ``no_deletions`` should stay about the same; the two
deletion cases should drop from ~3x that baseline down to about 1x.
"""

from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

NUM_ROWS = 1_000_000
NUM_DELETED = 10


@pytest.mark.parametrize(
    "predicate",
    [
        None,
        f"id >= {NUM_ROWS // 2} AND id < {NUM_ROWS // 2 + NUM_DELETED}",
        f"id % {NUM_ROWS // NUM_DELETED} == 0",
    ],
    ids=["no_deletions", "contiguous_deletions", "scattered_deletions"],
)
@pytest.mark.benchmark(group="scan_with_deletions")
def test_scan_with_deletions(tmp_path: Path, benchmark, predicate):
    table = pa.table(
        {
            "id": pa.array(range(NUM_ROWS), type=pa.int64()),
            "value": pc.random(NUM_ROWS),
        }
    )
    # One fragment, so the whole scan runs under a single deletion vector.
    dataset = lance.write_dataset(table, tmp_path, max_rows_per_file=NUM_ROWS)
    if predicate is not None:
        dataset.delete(predicate)
        dataset = lance.dataset(tmp_path)
        assert dataset.get_fragments()[0].metadata.deletion_file is not None

    benchmark.name = benchmark.param
    result = benchmark(dataset.to_table)

    expected = NUM_ROWS if predicate is None else NUM_ROWS - NUM_DELETED
    assert result.num_rows == expected
