# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import shutil
from pathlib import Path

import lance
import pyarrow as pa
from lance.query import PhraseQuery

EXPECTED_LANCE_VERSION = "4.0.1"
EXPECTED_FTS_VERSION = 2
NUM_ROWS = 300

assert lance.__version__ == EXPECTED_LANCE_VERSION

os.environ["LANCE_FTS_FORMAT_VERSION"] = str(EXPECTED_FTS_VERSION)

dataset_path = Path(__file__).parent / "fts_v2"
shutil.rmtree(dataset_path, ignore_errors=True)

row_ids = list(range(NUM_ROWS))
texts = [
    "lance database compatibility shared"
    if row_id % 3 == 0
    else "database lance compatibility shared"
    for row_id in row_ids
]
dataset = lance.write_dataset(pa.table({"id": row_ids, "text": texts}), dataset_path)
dataset.create_scalar_index("text", "INVERTED", with_position=True)

index = dataset.describe_indices()[0]
assert index.segments[0].index_version == EXPECTED_FTS_VERSION

matches = dataset.to_table(full_text_query="compatibility")
assert set(matches["id"].to_pylist()) == set(row_ids)

phrase_matches = dataset.to_table(full_text_query=PhraseQuery("lance database", "text"))
assert set(phrase_matches["id"].to_pylist()) == set(range(0, NUM_ROWS, 3))

# A merge_insert that updates only some columns hands the writer an all-valid null
# bitmap for the updated binary column, and Lance 4.0.1 then wrote a zero control-word
# byte before each value while the page layout records no control word bits (#6338).
merge_insert_path = Path(__file__).parent / "merge_insert_padded_control_words.lance"
shutil.rmtree(merge_insert_path, ignore_errors=True)
merge_insert_schema = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("a", pa.binary()),
        pa.field("b", pa.utf8()),
    ]
)
lance.write_dataset(
    pa.table(
        {"id": [0, 1], "a": [None, None], "b": [None, None]},
        schema=merge_insert_schema,
    ),
    merge_insert_path,
    data_storage_version="2.1",
)
update_schema = pa.schema(
    [merge_insert_schema.field("id"), merge_insert_schema.field("a")]
)
for row in range(2):
    lance.dataset(merge_insert_path).merge_insert(
        "id"
    ).when_matched_update_all().execute(
        pa.table({"id": [row], "a": [bytes([row + 1]) * 256]}, schema=update_schema)
    )

# Confirm the generator still captures the defect.
try:
    lance.dataset(merge_insert_path).to_table(columns=["a"])
except pa.ArrowInvalid:
    pass
else:
    raise AssertionError("expected the padded merge_insert page to be unreadable")
