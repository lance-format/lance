# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from pathlib import Path

import lance
import pyarrow as pa

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
