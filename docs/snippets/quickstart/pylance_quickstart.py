# Runnable source for docs/src/quickstart/pylance.md.
# ruff: noqa: E402, B018, I001

# --8<-- [start:imports]
import shutil

import lance
import pandas as pd
import pyarrow as pa
import pyarrow.dataset
# --8<-- [end:imports]

shutil.rmtree("/tmp/test.lance", ignore_errors=True)
shutil.rmtree("/tmp/test.parquet", ignore_errors=True)

# --8<-- [start:create-df]
df = pd.DataFrame({"a": [5]})
df
# --8<-- [end:create-df]

# --8<-- [start:write-lance]
shutil.rmtree("/tmp/test.lance", ignore_errors=True)

dataset = lance.write_dataset(df, "/tmp/test.lance")
dataset.to_table().to_pandas()
# --8<-- [end:write-lance]

# --8<-- [start:write-parquet]
shutil.rmtree("/tmp/test.parquet", ignore_errors=True)
shutil.rmtree("/tmp/test.lance", ignore_errors=True)

tbl = pa.Table.from_pandas(df)
pa.dataset.write_dataset(tbl, "/tmp/test.parquet", format="parquet")

parquet = pa.dataset.dataset("/tmp/test.parquet")
parquet.to_table().to_pandas()
# --8<-- [end:write-parquet]

# --8<-- [start:convert-parquet]
dataset = lance.write_dataset(parquet, "/tmp/test.lance")
dataset.to_table().to_pandas()
# --8<-- [end:convert-parquet]
