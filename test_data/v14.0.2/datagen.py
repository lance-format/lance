# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate the Lance 14.0.2 fixtures.

Two independent sets share this directory: a branch name above the local
atomic-creation filename budget, and the stable-row-ID layouts, which cover the
adaptive RowIdSequence encodings, inline row-ID metadata, and row version
sequences.

Run with: uv run --no-project --with pylance==14.0.2 datagen.py
"""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import lance
import pyarrow as pa

EXPECTED_LANCE_VERSION = "14.0.2"
assert lance.__version__ == EXPECTED_LANCE_VERSION, (
    f"Expected lance {EXPECTED_LANCE_VERSION}, got {lance.__version__}"
)

ROOT = Path(__file__).parent

# A branch name above the new local atomic-creation filename budget.
path = ROOT / "long_branch.lance"
shutil.rmtree(path, ignore_errors=True)
table = pa.table({"id": pa.array([0, 1], type=pa.int32())})
branch_name = "é" * 22
with TemporaryDirectory() as root:
    source = Path(root) / "long_branch.lance"
    # Use the object-store API so creation does not use the local staging filename.
    uri = f"file-object-store://{source}"
    dataset = lance.write_dataset(
        table, uri, max_rows_per_file=1, data_storage_version="2.1"
    )
    branch = dataset.create_branch(branch_name)
    # Give the branch its own files so the fixture can be relocated without the
    # shallow clone's absolute source URI.
    branch = lance.write_dataset(
        table, branch.uri, mode="overwrite", max_rows_per_file=1
    )
    assert len(branch.get_fragments()) == 2
    assert branch.to_table() == table
    shutil.copytree(source, path)

assert lance.dataset(path).checkout_version((branch_name, None)).to_table() == table

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
    # Row IDs are handed out sequentially from zero, so a fresh write leaves the
    # high-water mark at the row count.
    return lance.dataset(path), len(ids)


def record(ds, next_row_id):
    """Snapshot the dataset, including the writer's next_row_id high-water mark.

    Deletes, updates and compaction keep the row IDs they retire, so the mark
    only moves when a row is inserted. The recorded value is derived from the
    generation steps below rather than from the rows that survive.
    """
    columns = [
        "id",
        "_rowid",
        "_row_created_at_version",
        "_row_last_updated_at_version",
    ]
    table = ds.to_table(columns=columns)
    return {
        "next_row_id": next_row_id,
        **{column: table[column].to_pylist() for column in columns},
    }


expected = {}

# A plain append: one contiguous Range.
ds, next_row_id = write("stable_row_id_range", range(1000))
expected["stable_row_id_range"] = record(ds, next_row_id)

# Scattered deletions: compaction leaves a small hole list.
ds, next_row_id = write("stable_row_id_holes", range(1000))
ds.delete("id % 100 == 0")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_holes"] = record(ds, next_row_id)

# Dense deletions: compaction leaves a bitmap.
ds, next_row_id = write("stable_row_id_bitmap", range(1000))
ds.delete("id % 2 == 0")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_bitmap"] = record(ds, next_row_id)

# Sparse values inside a 64 KiB span: SortedArray with 16-bit offsets.
ds, next_row_id = write("stable_row_id_sorted_array", range(65_536))
ds.delete("id % 64 != 0")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_sorted_array"] = record(ds, next_row_id)

# Sparse values beyond a 64 KiB span: SortedArray with 32-bit offsets. Only
# 1,094 row IDs survive, but the 70,000 IDs the append handed out still count.
ds, next_row_id = write("stable_row_id_sorted_array_u32", range(70_000))
ds.delete("id % 64 != 0")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_sorted_array_u32"] = record(ds, next_row_id)

# A matched update in reverse row-id order leaves an unsorted Array.
ds, next_row_id = write("stable_row_id_array", range(10), range(10))
rows_before = ds.count_rows()
source = pa.table(
    {
        "id": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10, 11],
        "value": [900, 800, 700, 600, 500, 400, 300, 200, 100, 0, 10, 11],
    }
)
ds.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
    source
)
# Matched rows keep their row IDs; only the new rows consume fresh ones.
next_row_id += ds.count_rows() - rows_before
expected["stable_row_id_array"] = record(ds, next_row_id)

# Two updates make created-at and last-updated-at versions diverge, and
# compacting them leaves multiple version runs in one fragment. Updates keep the
# row IDs of the rows they rewrite.
ds, next_row_id = write("stable_row_id_lineage", range(20), range(20))
ds.update({"value": "value + 1000"}, where="id % 2 == 0")
ds.update({"value": "value + 2000"}, where="id % 2 == 1")
ds.optimize.compact_files(**COMPACT)
expected["stable_row_id_lineage"] = record(ds, next_row_id)

with open(ROOT / "expected.json", "w", encoding="utf-8") as out:
    json.dump(expected, out, indent=2)
    out.write("\n")

print(
    f"Generated {len(expected)} stable-row-ID fixtures with Lance {lance.__version__}"
)
