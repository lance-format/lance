# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate row-ID B-tree segments with the last release before the migration."""

import shutil
from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "14.0.0-beta.15"

path = Path(__file__).parent / "stable_row_id_btree.lance"
shutil.rmtree(path, ignore_errors=True)
dataset = lance.write_dataset(
    pa.table(
        {"value": pa.array([None, *range(1, 8), None, *range(9, 16)], pa.int32())}
    ),
    path,
    enable_stable_row_ids=True,
    max_rows_per_file=8,
    max_rows_per_group=8,
)
segments = [
    dataset.create_index_uncommitted(
        "value", "BTREE", name="value_idx", fragment_ids=[fragment.fragment_id]
    )
    for fragment in dataset.get_fragments()
]
dataset.commit_existing_index_segments("value_idx", "value", segments)
assert dataset.count_rows("value = 9") == 1
assert dataset.count_rows("value IS NULL") == 2
