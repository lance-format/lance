# Runnable source for docs/src/quickstart/versioning.md.
# ruff: noqa: E402, B018, I001

# --8<-- [start:imports]
import shutil

import lance
import pandas as pd
import pyarrow as pa
# --8<-- [end:imports]

# --8<-- [start:create]
shutil.rmtree("/tmp/versioning.lance", ignore_errors=True)

df = pd.DataFrame({"a": [5]})
dataset = lance.write_dataset(df, "/tmp/versioning.lance")
# --8<-- [end:create]

# --8<-- [start:append]
df = pd.DataFrame({"a": [10]})
tbl = pa.Table.from_pandas(df)
dataset = lance.write_dataset(tbl, "/tmp/versioning.lance", mode="append")

dataset.to_table().to_pandas()
# --8<-- [end:append]

# --8<-- [start:overwrite]
df = pd.DataFrame({"a": [50, 100]})
tbl = pa.Table.from_pandas(df)
dataset = lance.write_dataset(tbl, "/tmp/versioning.lance", mode="overwrite")

dataset.to_table().to_pandas()
# --8<-- [end:overwrite]

# --8<-- [start:versions]
[v["version"] for v in dataset.versions()]
# --8<-- [end:versions]

# --8<-- [start:version-refs]
dataset.version_refs()
# --8<-- [end:version-refs]

# --8<-- [start:checkout-versions]
# Version 1
print(lance.dataset("/tmp/versioning.lance", version=1).to_table().to_pandas())

# Version 2
print(lance.dataset("/tmp/versioning.lance", version=2).to_table().to_pandas())
# --8<-- [end:checkout-versions]

# --8<-- [start:tags]
dataset.tags.create("stable", 2)
dataset.tags.create("nightly", 3)
dict(sorted((name, info["version"]) for name, info in dataset.tags.list().items()))
# --8<-- [end:tags]

# --8<-- [start:checkout-tag]
lance.dataset("/tmp/versioning.lance", version="stable").to_table().to_pandas()
# --8<-- [end:checkout-tag]

# --8<-- [start:branches]
# Create a branch from the current latest version
experiment_branch = dataset.create_branch("experiment")

# Write to the branch (affects only that branch's history)
tbl = pa.Table.from_pandas(pd.DataFrame({"a": [42]}))
experiment_branch = lance.write_dataset(tbl, experiment_branch, mode="append")
experiment_branch.to_table().to_pandas()
# --8<-- [end:branches]
