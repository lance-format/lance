# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate 2.1 dictionary pages widened by chunk merging and compaction."""

import shutil
from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "13.0.0"

chunks = [
    pa.record_batch(
        [
            pa.DictionaryArray.from_arrays(
                pa.array([*range(100), None], type=pa.int8()),
                pa.array([f"v{i + chunk * 100}" for i in range(100)]),
            )
        ],
        names=["d"],
    )
    for chunk in range(2)
]
base = Path(__file__).parent

path = base / "merged_chunks.lance"
shutil.rmtree(path, ignore_errors=True)
table = pa.Table.from_batches(chunks)
lance.write_dataset(table, path, data_storage_version="2.1")
lance.write_dataset(table, path, mode="append", data_storage_version="2.1")

path = base / "compacted.lance"
shutil.rmtree(path, ignore_errors=True)
dataset = lance.write_dataset(
    pa.Table.from_batches(chunks[:1]), path, data_storage_version="2.1"
)
dataset = lance.write_dataset(
    pa.Table.from_batches(chunks[1:]), path, mode="append", data_storage_version="2.1"
)
dataset.optimize.compact_files()


def assert_read_fails(path: Path) -> None:
    try:
        lance.dataset(path).to_table()
    except pa.ArrowInvalid as error:
        assert "dictionary indices use 32 bits" in str(error)
    else:
        raise AssertionError(f"{path} must reproduce the widened-index read failure")


for name in ["merged_chunks.lance", "compacted.lance"]:
    assert_read_fails(base / name)
