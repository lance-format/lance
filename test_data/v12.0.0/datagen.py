# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""A table that names Arrow layouts in `logical_type`, written before the
semantic type contract existed.

`legacy_aliases` uses data storage version 2.2, the latest stable one. It is a
legacy table: it does not set FLAG_SEMANTIC_TYPES, so readers keep returning
exactly these Arrow types and appends keep comparing exact types until the
table is migrated.
"""

import shutil
from decimal import Decimal
from pathlib import Path

import lance
import pyarrow as pa

EXPECTED_LANCE_VERSION = "12.0.0"

assert lance.__version__ == EXPECTED_LANCE_VERSION

table = pa.table(
    {
        "id": pa.array([1, 2, 3], pa.int64()),
        "s": pa.array(["a", None, "c"], pa.large_string()),
        "d": pa.array(["x", "y", "x"], pa.string()).dictionary_encode(),
        "n": pa.array([Decimal("1.25"), None, Decimal("-3.50")], pa.decimal256(10, 2)),
        "l": pa.array([[1], None, [2, 3]], pa.large_list(pa.int32())),
        "b": pa.array([b"p", b"q", None], pa.large_binary()),
        "st": pa.array(
            [{"x": "u"}, {"x": None}, None], pa.struct([("x", pa.string())])
        ),
    }
)

dataset_path = Path(__file__).parent / "legacy_aliases"
shutil.rmtree(dataset_path, ignore_errors=True)
dataset = lance.write_dataset(table, dataset_path, data_storage_version="2.2")
assert dataset.to_table() == table
