# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "3.0.1", lance.__version__

dataset_path = Path(__file__).parent / "data.lance"
shutil.rmtree(dataset_path, ignore_errors=True)
table = pa.table(
    {
        "id": pa.array(range(8), type=pa.int32()),
        "labels": pa.array(
            [[1], None, [], [2, None], [1, 2], None, [None], [3]],
            type=pa.list_(pa.int64()),
        ),
    }
)
dataset = lance.write_dataset(table, dataset_path, max_rows_per_file=4)
dataset.create_scalar_index("labels", "LABEL_LIST", name="labels_idx")
assert dataset.describe_indices()[0].segments[0].index_version == 0
assert len(dataset.get_fragments()) == 2
assert dataset.count_rows("array_has_any(labels, [1])") == 2
