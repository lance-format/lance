# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate a branch above the new local atomic-creation filename budget.

Run with: uv run --no-project --with pylance==14.0.2 datagen.py
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import lance
import pyarrow as pa

EXPECTED_LANCE_VERSION = "14.0.2"

assert lance.__version__ == EXPECTED_LANCE_VERSION

path = Path(__file__).parent / "long_branch.lance"
shutil.rmtree(path, ignore_errors=True)
table = pa.table({"id": pa.array([0, 1], type=pa.int32())})
branch_name = "é" * 22
with TemporaryDirectory() as root:
    source = Path(root) / "long_branch.lance"
    # Use the object-store API so creation does not use the local staging filename.
    uri = f"file-object-store://{source}"
    dataset = lance.write_dataset(
        table, uri, max_rows_per_file=1, data_storage_version="2.1"
    )
    branch = dataset.create_branch(branch_name)
    # Give the branch its own files so the fixture can be relocated without the
    # shallow clone's absolute source URI.
    branch = lance.write_dataset(
        table, branch.uri, mode="overwrite", max_rows_per_file=1
    )
    assert len(branch.get_fragments()) == 2
    assert branch.to_table() == table
    shutil.copytree(source, path)

assert lance.dataset(path).checkout_version((branch_name, None)).to_table() == table
