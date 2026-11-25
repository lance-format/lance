# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate dictionary columns whose stored indices are wider than the key type.

Lance 10.0.0 widened the indices of a dictionary page to 32 bits while the field
kept its narrower declared key type. The 2.0 writer did so when it appended a null
slot past the last addressable key, and every writer did so when the dictionaries of
the chunks buffered into one page did not fit the key type together. The stored
indices still point at the right values.
"""

import shutil
from pathlib import Path

import lance
import pyarrow as pa

EXPECTED_LANCE_VERSION = "10.0.0"

assert lance.__version__ == EXPECTED_LANCE_VERSION


def int8_dictionary(values, keys):
    return pa.DictionaryArray.from_arrays(pa.array(keys, pa.int8()), pa.array(values))


# Uses all 128 Int8 keys and contains nulls, so the 2.0 writer appends a null slot
# at index 128.
null_slot = pa.table(
    {
        "d": int8_dictionary(
            [f"v{i:03d}" for i in range(128)], list(range(128)) + [None, 5, None]
        )
    }
)

# Two chunks with 100 distinct values each, so one page holds 200 distinct values.
disjoint = pa.Table.from_batches(
    [
        pa.record_batch(
            [
                int8_dictionary(
                    [f"v{chunk * 100 + i:03d}" for i in range(100)],
                    [None if i % 25 == 3 else i for i in range(100)],
                )
            ],
            names=["d"],
        )
        for chunk in range(2)
    ]
)

for name, table, version in [
    ("null_slot_v2_0.lance", null_slot, "2.0"),
    ("disjoint_v2_1.lance", disjoint, "2.1"),
]:
    path = Path(__file__).parent / name
    shutil.rmtree(path, ignore_errors=True)
    lance.write_dataset(table, path, data_storage_version=version)
    # Confirm the generator still captures the defect: Lance 10.0.0 reads these
    # pages back without an error but with the wrong values.
    actual = lance.dataset(path).to_table().column("d").to_pylist()
    assert actual != table.column("d").to_pylist()
