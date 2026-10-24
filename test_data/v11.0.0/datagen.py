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
