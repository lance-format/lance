# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate stable-row-ID fixtures with the last release before the layout change.

Run with pylance 14.0.2. The fixtures cover the adaptive RowIdSequence
encodings, inline row-ID metadata, and row version sequences.

The expected row contents are written to expected.json next to the datasets.
"""

import json
import shutil
from pathlib import Path

import lance
import pyarrow as pa

EXPECTED_LANCE_VERSION = "14.0.2"
assert lance.__version__ == EXPECTED_LANCE_VERSION, (
    f"Expected lance {EXPECTED_LANCE_VERSION}, got {lance.__version__}"
)

ROOT = Path(__file__).parent
COMPACT = dict(
    materialize_deletions=True,
    materialize_deletions_threshold=0.0,
    target_rows_per_fragment=200_000,
)


def write(name: str, ids, values=None):
    path = ROOT / f"{name}.lance"
    shutil.rmtree(path, ignore_errors=True)
    data = {"id": pa.array(ids, pa.int64())}
    if values is not None:
        data["value"] = pa.array(values, pa.int64())
    lance.write_dataset(
        pa.table(data),
        path,
        enable_stable_row_ids=True,
        max_rows_per_file=len(ids),
        max_rows_per_group=1024,
    )
    return lance.dataset(path)


def record(ds):
    table = ds.to_table(
        columns=[
            "id",
            "_rowid",
            "_row_created_at_version",
            "_row_last_updated_at_version",
        ]
    )
    return {
        column: table[column].to_pylist()
        for column in [
            "id",
            "_rowid",
            "_row_created_at_version",
            "_row_last_updated_at_version",
        ]
    }


expected = {}

# A plain append: one contiguous Range.
ds = write("stable_row_id_range", range(1000))
expected["stable_row_id_range"] = record(ds)

# Scattered deletions: compaction leaves a small hole list.
ds = write("stable_row_id_holes", range(1000))
ds.delete("id % 100 == 0")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_holes"] = record(ds)

# Dense deletions: compaction leaves a bitmap.
ds = write("stable_row_id_bitmap", range(1000))
ds.delete("id % 2 == 0")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_bitmap"] = record(ds)

# Sparse values inside a 64 KiB span: SortedArray with 16-bit offsets.
ds = write("stable_row_id_sorted_array", range(65_536))
ds.delete("id % 64 != 0")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_sorted_array"] = record(ds)

# Sparse values beyond a 64 KiB span: SortedArray with 32-bit offsets.
ds = write("stable_row_id_sorted_array_u32", range(70_000))
ds.delete("id % 64 != 0")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_sorted_array_u32"] = record(ds)

# A matched update in reverse row-id order leaves an unsorted Array.
ds = write("stable_row_id_array", range(10), range(10))
source = pa.table(
    {
        "id": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10, 11],
        "value": [900, 800, 700, 600, 500, 400, 300, 200, 100, 0, 10, 11],
    }
)
ds.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
    source
)
expected["stable_row_id_array"] = record(ds)

# Two updates make created-at and last-updated-at versions diverge, and
# compacting them leaves multiple version runs in one fragment.
ds = write("stable_row_id_lineage", range(20), range(20))
ds.update({"value": "value + 1000"}, where="id % 2 == 0")
ds.update({"value": "value + 2000"}, where="id % 2 == 1")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_lineage"] = record(ds)

with open(ROOT / "expected.json", "w", encoding="utf-8") as out:
    json.dump(expected, out, indent=2)
    out.write("\n")

print(f"Generated {len(expected)} stable-row-ID fixtures with Lance {lance.__version__}")
