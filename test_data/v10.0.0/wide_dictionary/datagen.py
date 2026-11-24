# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate the 2.0 null-normalization regression fixture with a released writer."""

import shutil
from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "10.0.0"

path = Path(__file__).parent / "appended_null.lance"
shutil.rmtree(path, ignore_errors=True)
dictionary = pa.DictionaryArray.from_arrays(
    pa.array([*range(128), None], type=pa.int8()),
    pa.array([f"v{i}" for i in range(128)]),
)
table = pa.table({"d": dictionary})
lance.write_dataset(table, path, data_storage_version="2.0")
lance.write_dataset(table, path, mode="append", data_storage_version="2.0")
