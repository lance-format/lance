# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate the Lance 14.0.2 fixtures.

Three independent sets share this directory: a branch name above the local
atomic-creation filename budget, the stable-row-ID layouts, which cover the
adaptive RowIdSequence encodings, inline row-ID metadata, and row version
sequences, and a raw Blob v2 file whose footer keeps anonymous descriptor
children as root fields.

Run with: uv run --no-project --with pylance==14.0.2 --with pyarrow==26.0.1 datagen.py
"""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import lance
import pyarrow as pa
from lance.file import LanceFileReader, LanceFileWriter

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

# A raw v2.2 file, not a dataset, with two top-level Blob v2 columns and one
# nested Blob v2 column. Lance 14.0.2's raw file writer persists the anonymous
# descriptor children as top-level footer fields, which is the historical shape
# the raw file reader must reconstruct.
BLOB_PREPARED_FIELDS = [
    pa.field("kind", pa.uint8(), nullable=True),
    pa.field("data", pa.large_binary(), nullable=True),
    pa.field("uri", pa.utf8(), nullable=True),
    pa.field("blob_id", pa.uint32(), nullable=True),
    pa.field("blob_size", pa.uint64(), nullable=True),
    pa.field("position", pa.uint64(), nullable=True),
]
BLOB_METADATA = {
    "lance-encoding:blob": "true",
    "lance-encoding:packed": "true",
    "ARROW:extension:name": "lance.blob.v2",
}


def prepared_blob(rows):
    kind, data, blob_id, blob_size, position, valid = zip(*rows)
    return pa.StructArray.from_arrays(
        [
            pa.array(kind, pa.uint8()),
            pa.array(data, pa.large_binary()),
            pa.array([None] * len(rows), pa.utf8()),
            pa.array(blob_id, pa.uint32()),
            pa.array(blob_size, pa.uint64()),
            pa.array(position, pa.uint64()),
        ],
        names=[field.name for field in BLOB_PREPARED_FIELDS],
        mask=pa.array([not value for value in valid], pa.bool_()),
    )


raw_path = ROOT / "blob_v2_raw_reader" / "blob_v2_raw.lance"
shutil.rmtree(raw_path.parent, ignore_errors=True)
raw_path.parent.mkdir()
raw_schema = pa.schema(
    [
        pa.field("id", pa.int32(), nullable=False),
        pa.field(
            "blob_a",
            pa.struct(BLOB_PREPARED_FIELDS),
            nullable=True,
            metadata=BLOB_METADATA,
        ),
        pa.field(
            "blob_b",
            pa.struct(BLOB_PREPARED_FIELDS),
            nullable=True,
            metadata=BLOB_METADATA,
        ),
        pa.field(
            "info",
            pa.struct(
                [
                    pa.field(
                        "payload",
                        pa.struct(BLOB_PREPARED_FIELDS),
                        nullable=True,
                        metadata=BLOB_METADATA,
                    )
                ]
            ),
            nullable=True,
        ),
    ]
)
raw_batch = pa.RecordBatch.from_arrays(
    [
        pa.array([10, 20, 30], pa.int32()),
        prepared_blob(
            [
                (0, b"alpha", 0, 0, 0, True),
                (0, None, 0, 0, 0, False),
                (0, b"", 0, 0, 0, True),
            ]
        ),
        prepared_blob(
            [
                (0, b"red", 0, 0, 0, True),
                (0, b"green", 0, 0, 0, True),
                (0, None, 0, 0, 0, False),
            ]
        ),
        pa.StructArray.from_arrays(
            [
                prepared_blob(
                    [
                        (0, b"nested-0", 0, 0, 0, True),
                        (0, None, 0, 0, 0, False),
                        (0, b"nested-2", 0, 0, 0, True),
                    ]
                )
            ],
            names=["payload"],
        ),
    ],
    schema=raw_schema,
)
raw_writer = LanceFileWriter(str(raw_path), raw_schema, version="2.2")
raw_writer.write_batch(raw_batch)
assert raw_writer.close() == 3, "raw Blob v2 fixture must contain three rows"

# The released reader reproduces the historical failure on this footer: it sees
# each anonymous descriptor group as three or more extra root fields.
try:
    LanceFileReader(str(raw_path)).read_all().to_table()
except ValueError as error:
    assert "column index" in str(error), error
else:
    raise RuntimeError("fixture no longer reproduces the anonymous Blob v2 footer")
