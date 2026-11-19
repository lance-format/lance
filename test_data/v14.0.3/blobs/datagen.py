# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate a table whose list tags predate Blob v2 element promotion."""

import shutil
from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "14.0.3"

item = pa.field("item", pa.large_binary(), metadata={"lance-encoding:blob": "true"})
rows = [[b"before", None, b""], None, []]
table = pa.table(
    {
        "list_blobs": pa.array(rows, type=pa.list_(item)),
        "large_list_blobs": pa.array(rows, type=pa.large_list(item)),
    }
)

path = Path(__file__).parent / "promoted_lists.lance"
shutil.rmtree(path, ignore_errors=True)
lance.write_dataset(table, path, data_storage_version="2.2")
dataset = lance.dataset(path)
assert dataset.to_table(blob_handling="all_binary").to_pydict() == {
    "list_blobs": rows,
    "large_list_blobs": rows,
}
