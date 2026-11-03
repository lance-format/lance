# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate the released Packed/Dedicated fixture used by Managed adoption tests."""

import shutil
from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "11.0.0"

field = lance.blob_field("blob").with_metadata(
    {
        "lance-encoding:blob-inline-size-threshold": "8",
        "lance-encoding:blob-dedicated-size-threshold": "24",
    }
)
table = pa.Table.from_arrays(
    [lance.blob_array([b"p" * 16, b"inline", None, b"", b"d" * 32, b"inline"])],
    schema=pa.schema([field]),
)
dataset_path = Path(__file__).parent / "blob_sidecars"
shutil.rmtree(dataset_path, ignore_errors=True)
dataset = lance.write_dataset(
    table,
    dataset_path,
    data_storage_version="2.2",
    max_rows_per_file=3,
    max_rows_per_group=3,
)
assert [
    row["blob"]["kind"] if row["blob"] is not None else None
    for row in dataset.to_table().to_pylist()
] == [1, 0, None, 0, 2, 0]

# Two FixedSizeList<Struct> columns whose items are wide enough for the full-zip layout.
# The batch's list validity bitmap has nulls only in the first file's rows, so the
# second file's slice of it has none. Lance 11.0.0 then still wrote a zero control-word
# byte before each value of the second file while its page layout records no control
# word bits.
NUM_ROWS = 8
ROWS_PER_FILE = 4
LIST_SIZE = 2
VECTOR_DIM = 128
num_items = NUM_ROWS * LIST_SIZE
binary_items = pa.StructArray.from_arrays(
    [
        pa.array(
            [item.to_bytes(4, "little") * 75 for item in range(num_items)],
            pa.large_binary(),
        )
    ],
    names=["b"],
)
vector_items = pa.StructArray.from_arrays(
    [
        pa.FixedSizeListArray.from_arrays(
            pa.array(range(num_items * VECTOR_DIM), pa.float32()), VECTOR_DIM
        )
    ],
    names=["v"],
)
null_lists = pa.array([row < 2 for row in range(NUM_ROWS)])
padded = pa.table(
    {
        "binary": pa.FixedSizeListArray.from_arrays(
            binary_items, LIST_SIZE, mask=null_lists
        ),
        "vector": pa.FixedSizeListArray.from_arrays(
            vector_items, LIST_SIZE, mask=null_lists
        ),
    }
)
dataset_path = Path(__file__).parent / "fullzip_padded_control_words.lance"
shutil.rmtree(dataset_path, ignore_errors=True)
dataset = lance.write_dataset(
    padded,
    dataset_path,
    data_storage_version="2.2",
    max_rows_per_file=ROWS_PER_FILE,
)

# Confirm the generator still captures the defect: in the second file the binary column
# cannot be read and the vector column reads back shifted.
first, second = dataset.get_fragments()
assert first.to_table().equals(padded.slice(0, ROWS_PER_FILE))
try:
    second.to_table(columns=["binary"])
except pa.ArrowInvalid:
    pass
else:
    raise AssertionError("expected the padded binary page to be unreadable")
vectors = second.to_table(columns=["vector"])["vector"]
assert not vectors.equals(padded["vector"].slice(ROWS_PER_FILE))
