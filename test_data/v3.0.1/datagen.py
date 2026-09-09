# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import shutil
from pathlib import Path

import lance
import pyarrow as pa
from lance.query import PhraseQuery

EXPECTED_LANCE_VERSION = "3.0.1"
EXPECTED_FTS_VERSION = 1
NUM_ROWS = 300

assert lance.__version__ == EXPECTED_LANCE_VERSION

os.environ["LANCE_FTS_FORMAT_VERSION"] = str(EXPECTED_FTS_VERSION)

dataset_path = Path(__file__).parent / "fts_v1"
shutil.rmtree(dataset_path, ignore_errors=True)

row_ids = list(range(NUM_ROWS))
texts = [
    "lance database compatibility shared"
    if row_id % 3 == 0
    else "database lance compatibility shared"
    for row_id in row_ids
]
dataset = lance.write_dataset(pa.table({"id": row_ids, "text": texts}), dataset_path)
dataset.create_scalar_index(
    "text",
    "INVERTED",
    with_position=True,
    skip_merge=True,
)

index = dataset.describe_indices()[0]
assert index.segments[0].index_version == EXPECTED_FTS_VERSION

matches = dataset.to_table(full_text_query="compatibility")
assert set(matches["id"].to_pylist()) == set(row_ids)

phrase_matches = dataset.to_table(full_text_query=PhraseQuery("lance database", "text"))
assert set(phrase_matches["id"].to_pylist()) == set(range(0, NUM_ROWS, 3))
