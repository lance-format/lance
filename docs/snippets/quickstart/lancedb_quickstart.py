# Runnable source for docs/src/quickstart/index.md. The sections between the
# `--8<--` markers are embedded in the page; everything else (setup, asserts)
# exists only so CI can check the example still works.
# ruff: noqa: E402  (each snippet section imports what it uses)
import shutil

shutil.rmtree("/tmp/lancedb-quickstart", ignore_errors=True)

# --8<-- [start:connect]
import lancedb

db = lancedb.connect("/tmp/lancedb-quickstart")
# --8<-- [end:connect]

# fmt: off
# --8<-- [start:create]
data = [
    {"id": 1, "name": "red running shoes", "category": "shoes", "price": 89.0, "vector": [0.9, 0.1, 0.1, 0.2]},
    {"id": 2, "name": "blue trail running shoes", "category": "shoes", "price": 120.0, "vector": [0.8, 0.2, 0.1, 0.3]},
    {"id": 3, "name": "leather hiking boots", "category": "shoes", "price": 150.0, "vector": [0.6, 0.1, 0.5, 0.2]},
    {"id": 4, "name": "waterproof rain jacket", "category": "outerwear", "price": 99.0, "vector": [0.1, 0.9, 0.3, 0.1]},
    {"id": 5, "name": "wool winter coat", "category": "outerwear", "price": 210.0, "vector": [0.2, 0.7, 0.6, 0.1]},
]
table = db.create_table("products", data=data)
table.to_pandas()
# --8<-- [end:create]
# fmt: on
assert table.count_rows() == 5

# --8<-- [start:filter]
table.search().where("price < 100").select(["id", "name", "price"]).to_pandas()
# --8<-- [end:filter]
cheap = table.search().where("price < 100").select(["id", "name", "price"]).to_pandas()
assert sorted(cheap["id"]) == [1, 4]

# --8<-- [start:vector-search]
query = [0.85, 0.15, 0.1, 0.25]
table.search(query).limit(3).select(["id", "name", "_distance"]).to_pandas()
# --8<-- [end:vector-search]
nearest = table.search(query).limit(3).select(["id", "name", "_distance"]).to_pandas()
assert list(nearest["id"]) == [1, 2, 3]

# --8<-- [start:filtered-vector-search]
table.search(query).where("category = 'outerwear'").limit(1).to_pandas()
# --8<-- [end:filtered-vector-search]
nearest_outerwear = (
    table.search(query).where("category = 'outerwear'").limit(1).to_pandas()
)
assert list(nearest_outerwear["id"]) == [5]

# --8<-- [start:fts]
from lancedb.index import FTS

table.create_index("name", config=FTS())
table.search("running", query_type="fts").to_pandas()
# --8<-- [end:fts]
running = table.search("running", query_type="fts").to_pandas()
assert sorted(running["id"]) == [1, 2]

# fmt: off
# --8<-- [start:modify]
table.add(
    [{"id": 6, "name": "trail socks", "category": "accessories", "price": 15.0, "vector": [0.7, 0.3, 0.2, 0.2]}]
)
table.update(where="id = 1", values={"price": 79.0})
table.delete("category = 'outerwear'")
table.to_pandas()
# --8<-- [end:modify]
# fmt: on
assert sorted(table.to_pandas()["id"]) == [1, 2, 3, 6]

# --8<-- [start:versions]
table.list_versions()

table.checkout(1)  # time travel: the table as it was when created
table.count_rows()
# --8<-- [end:versions]
assert table.count_rows() == 5

# --8<-- [start:restore]
table.checkout_latest()
table.count_rows()
# --8<-- [end:restore]
assert table.count_rows() == 4
