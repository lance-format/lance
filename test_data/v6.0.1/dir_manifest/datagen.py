# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Generate with:
# uv run --isolated --no-project --python 3.12 --with pylance==6.0.1 datagen.py
import shutil
from pathlib import Path

import lance
import pyarrow as pa
from lance.namespace import DirectoryNamespace
from lance_namespace import CreateTableRequest

assert lance.__version__ == "6.0.1"

root = Path(__file__).parent / "namespace"
shutil.rmtree(root, ignore_errors=True)
namespace = DirectoryNamespace(
    root=str(root), manifest_enabled="true", dir_listing_enabled="false"
)
batch = pa.record_batch({"id": [1]})
sink = pa.BufferOutputStream()
with pa.ipc.new_stream(sink, batch.schema) as writer:
    writer.write_batch(batch)
namespace.create_table(CreateTableRequest(id=["t1"]), sink.getvalue().to_pybytes())
manifest = lance.dataset(root / "__manifest")
assert manifest.config()["lance.auto_cleanup.interval"] == "20"
assert manifest.config()["lance.auto_cleanup.older_than"] == "14days"
