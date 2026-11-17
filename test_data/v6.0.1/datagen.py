# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader, LanceFileWriter

EXPECTED_LANCE_VERSION = "6.0.1"
NUM_DENSE_ROWS = 513
VALUES_PER_DENSE_ROW = 32
NUM_TRAILING_ROWS = 65_536

assert lance.__version__ == EXPECTED_LANCE_VERSION

dataset_path = Path(__file__).parent / "miniblock_level_count_overflow.lance"
shutil.rmtree(dataset_path, ignore_errors=True)

captions = pa.array(
    [
        list(range(row * VALUES_PER_DENSE_ROW, (row + 1) * VALUES_PER_DENSE_ROW))
        for row in range(NUM_DENSE_ROWS)
    ]
    + [([] if row % 2 == 0 else None) for row in range(NUM_TRAILING_ROWS)],
    type=pa.list_(pa.uint32()),
)
table = pa.table(
    {
        "mime": ["image/jpeg"] * len(captions),
        "captions": captions,
    }
)
lance.write_dataset(table, dataset_path, data_storage_version="2.2")

# The v6.0.1 writer truncated a miniblock's structural level count to u16 while
# retaining the complete RLE payload. Confirm this generator still captures the defect.
try:
    lance.dataset(dataset_path).to_table()
except pa.ArrowInvalid as error:
    assert 'StructArray field "captions", expected 8192 got 513' in str(error)
else:
    raise AssertionError("expected the v6.0.1 miniblock level-count overflow")

# A legacy list page whose structural count exceeds a u16, with out-of-line
# bitpacked rep/def buffers that retain every level.
vector = np.arange(2048, dtype=np.float16).tolist()
values = [vector] * 64 + [None] * 70_000 + [vector]
table = pa.table(
    {
        "id": pa.array(range(len(values)), type=pa.int64()),
        "vector": pa.array(values, type=pa.large_list(pa.float16())),
    },
    schema=pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(
                "vector",
                pa.large_list(pa.float16()),
                metadata={
                    b"lance-encoding:structural-encoding": b"miniblock",
                },
            ),
        ]
    ),
)
file_path = str(Path(__file__).with_name("wrapped_bitpacked_levels.lance"))
with LanceFileWriter(file_path, table.schema, version="2.1") as writer:
    writer.write_batch(table)

# The v6.0.1 reader trusts the wrapped u16 level count and drops levels. Confirm
# this generator still captures the defect.
try:
    LanceFileReader(file_path).read_all(batch_size=1024).to_table()
except pa.ArrowInvalid as error:
    assert 'StructArray field "vector", expected 1024 got 432' in str(error)
else:
    raise AssertionError("expected the v6.0.1 wrapped bitpacked level count")
