# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate legacy Blob fixtures with the released pylance 8.0.0 wheel."""

import shutil
from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "8.0.0"

schema = pa.schema(
    [
        pa.field("id", pa.uint32(), nullable=False),
        pa.field(
            "blob",
            pa.large_binary(),
            metadata={
                "lance-encoding:blob": "true",
                "lance-encoding:blob-inline-size-threshold": "4",
                "lance-encoding:blob-dedicated-size-threshold": "12",
            },
        ),
    ]
)
table = pa.Table.from_pydict(
    {"id": [0, 1, 2], "blob": [b"legacy bytes", None, b""]}, schema=schema
)
for version in ["2.0", "2.1"]:
    dataset_path = Path(__file__).parent / f"v{version}.lance"
    shutil.rmtree(dataset_path, ignore_errors=True)
    dataset = lance.write_dataset(
        table,
        dataset_path,
        data_storage_version=version,
    )
    assert dataset.count_rows() == 3
    assert dataset.schema == schema
