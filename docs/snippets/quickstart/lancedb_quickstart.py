# Runnable source for docs/src/quickstart/index.md. The sections between the
# `--8<--` markers are embedded in the page, and CI checks what each one
# displays against outputs/ (see tests/snippet_runner.py).
# Sections import what they use and end with an expression to display, like
# notebook cells.
# ruff: noqa: E402, B018
import shutil

shutil.rmtree("/tmp/lancedb-quickstart", ignore_errors=True)

# --8<-- [start:connect]
import lancedb

db = lancedb.connect("/tmp/lancedb-quickstart")
# --8<-- [end:connect]

# fmt: off
# --8<-- [start:create]
data = [
    {"id": 1, "name": "red running shoes", "category": "shoes", "price": 89.0, "vector": [1.0, 0.0, 0.0, 0.25]},
    {"id": 2, "name": "blue trail running shoes", "category": "shoes", "price": 120.0, "vector": [0.875, 0.125, 0.0, 0.375]},
    {"id": 3, "name": "leather hiking boots", "category": "shoes", "price": 150.0, "vector": [0.625, 0.0, 0.5, 0.25]},
    {"id": 4, "name": "waterproof rain jacket", "category": "outerwear", "price": 99.0, "vector": [0.0, 1.0, 0.25, 0.0]},
    {"id": 5, "name": "wool winter coat", "category": "outerwear", "price": 210.0, "vector": [0.25, 0.75, 0.625, 0.0]},
]
table = db.create_table("products", data=data)
table.to_pandas()
# --8<-- [end:create]
# fmt: on

# --8<-- [start:filter]
table.search().where("price < 100").select(["id", "name", "price"]).to_pandas()
# --8<-- [end:filter]

# --8<-- [start:vector-search]
query = [0.875, 0.125, 0.0, 0.25]
table.search(query).limit(3).select(["id", "name", "_distance"]).to_pandas()
# --8<-- [end:vector-search]

# --8<-- [start:filtered-vector-search]
(
    table.search(query)
    .where("category = 'outerwear'")
    .limit(1)
    .select(["id", "name", "_distance"])
    .to_pandas()
)
# --8<-- [end:filtered-vector-search]

# --8<-- [start:fts]
from lancedb.index import FTS

table.create_index("name", config=FTS())
table.search("running", query_type="fts").to_pandas()
# --8<-- [end:fts]

# fmt: off
# --8<-- [start:modify]
table.add(
    [{"id": 6, "name": "trail socks", "category": "accessories", "price": 15.0, "vector": [0.75, 0.25, 0.25, 0.25]}]
)
table.update(where="id = 1", values={"price": 79.0})
table.delete("category = 'outerwear'")
table.to_pandas()
# --8<-- [end:modify]
# fmt: on

# --8<-- [start:version]
table.version
# --8<-- [end:version]

# --8<-- [start:checkout]
table.checkout(1)
table.to_pandas()
# --8<-- [end:checkout]

# --8<-- [start:checkout-latest]
table.checkout_latest()
table.count_rows()
# --8<-- [end:checkout-latest]
