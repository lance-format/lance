# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate a legacy Blob table with a 2.2 fragment from the released 12.0.0 wheel.

The 12.0.0 fragment writer keeps the legacy Blob layout in 2.2 files, and 12.0.0
reads the resulting table back.
"""

import shutil
from pathlib import Path

import lance
import pyarrow as pa
from lance.fragment import LanceFragment

assert lance.__version__ == "12.0.0"

schema = pa.schema(
    [
        pa.field("id", pa.uint32(), nullable=False),
        pa.field("blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"}),
    ]
)
values = [b"legacy bytes", None, b"", b"fragment bytes", None, b""]


def rows(ids):
    return pa.Table.from_pydict(
        {"id": ids, "blob": [values[i] for i in ids]}, schema=schema
    )


path = Path(__file__).parent / "legacy_fragment.lance"
shutil.rmtree(path, ignore_errors=True)
lance.write_dataset(rows([0, 1, 2]), path, data_storage_version="2.1")
fragment = LanceFragment.create(path, rows([3, 4, 5]), data_storage_version="2.2")
dataset = lance.LanceDataset.commit(
    path, lance.LanceOperation.Append([fragment]), read_version=1
)

assert dataset.schema == schema
assert [
    (data_file.file_major_version, data_file.file_minor_version)
    for fragment in dataset.get_fragments()
    for data_file in fragment.data_files()
] == [(2, 1), (2, 2)]
blobs = dataset.take_blobs("blob", indices=list(range(len(values))))
assert [blob and blob.readall() for blob in blobs] == values
