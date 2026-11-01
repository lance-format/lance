# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from decimal import Decimal
from pathlib import Path

import lance
import pyarrow as pa
from lance.indices import IndexConfig

EXPECTED_LANCE_VERSION = "8.0.0"

assert lance.__version__ == EXPECTED_LANCE_VERSION

dataset_path = Path(__file__).parent / "decimal_zonemap"
shutil.rmtree(dataset_path, ignore_errors=True)

values = pa.array(
    [Decimal("1.00"), Decimal("2.00"), Decimal("3.00")],
    type=pa.decimal128(10, 2),
)
dataset = lance.write_dataset(
    pa.table({"id": [1, 2, 3], "value": values}),
    dataset_path,
)
dataset.create_scalar_index("value", IndexConfig("zonemap", {}))

indices = dataset.describe_indices()
assert len(indices) == 1
assert indices[0].index_type == "ZoneMap"
