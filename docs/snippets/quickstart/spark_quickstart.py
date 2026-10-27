# Runnable source for the PySpark tabs in docs/src/quickstart/spark.md. The
# sections between the `--8<--` markers are embedded in the page; everything
# else (setup, asserts) exists only so CI can check the example still works.
# ruff: noqa: E402  (each snippet section imports what it uses)
import shutil

shutil.rmtree("/tmp/lance-spark-quickstart", ignore_errors=True)

# --8<-- [start:session]
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("lance-quickstart")
    .config("spark.jars.packages", "org.lance:lance-spark-bundle-4.2_2.13:0.8.0")
    .config(
        "spark.sql.extensions", "org.lance.spark.extensions.LanceSparkSessionExtensions"
    )
    .config("spark.sql.catalog.lance", "org.lance.spark.LanceNamespaceSparkCatalog")
    .config("spark.sql.catalog.lance.impl", "dir")
    .config("spark.sql.catalog.lance.root", "/tmp/lance-spark-quickstart")
    .config("spark.sql.defaultCatalog", "lance")
    .getOrCreate()
)
# --8<-- [end:session]

# --8<-- [start:create]
products = spark.createDataFrame(
    [
        (1, "red running shoes", "shoes", 89.0, [0.9, 0.1, 0.1, 0.2]),
        (2, "blue trail running shoes", "shoes", 120.0, [0.8, 0.2, 0.1, 0.3]),
        (3, "leather hiking boots", "shoes", 150.0, [0.6, 0.1, 0.5, 0.2]),
        (4, "waterproof rain jacket", "outerwear", 99.0, [0.1, 0.9, 0.3, 0.1]),
        (5, "wool winter coat", "outerwear", 210.0, [0.2, 0.7, 0.6, 0.1]),
    ],
    "id BIGINT, name STRING, category STRING, price DOUBLE, vector ARRAY<FLOAT>",
)

# Lance stores vectors as fixed-size lists, so tell it the dimension.
(
    products.writeTo("products")
    .tableProperty("vector.arrow.fixed-size-list.size", "4")
    .create()
)
# --8<-- [end:create]
assert spark.table("products").count() == 5

# --8<-- [start:query]
from pyspark.sql import functions as F

table = spark.table("products")
table.filter("price < 100").select("id", "name", "price").show()

table.groupBy("category").agg(
    F.count("*").alias("num_products"), F.avg("price").alias("avg_price")
).show()
# --8<-- [end:query]
assert sorted(r.id for r in table.filter("price < 100").collect()) == [1, 4]

# --8<-- [start:vector-search]
nearest = spark.sql("""
    SELECT id, name, _distance
    FROM VECTOR_SEARCH(
        table => 'products',
        query_vector => array(0.85, 0.15, 0.1, 0.25),
        num_results => 3
    )
    ORDER BY _distance
""")
nearest.show()
# --8<-- [end:vector-search]
assert [r.id for r in nearest.collect()] == [1, 2, 3]

# --8<-- [start:fts]
spark.sql("ALTER TABLE products CREATE INDEX name_idx USING fts (name)")

matches = spark.table("products").filter("lance_match(name, 'running')")
matches.select("id", "name").show()
# --8<-- [end:fts]
assert sorted(r.id for r in matches.collect()) == [1, 2]

# --8<-- [start:modify]
socks = spark.createDataFrame(
    [(6, "trail socks", "accessories", 15.0, [0.7, 0.3, 0.2, 0.2])],
    "id BIGINT, name STRING, category STRING, price DOUBLE, vector ARRAY<FLOAT>",
)
socks.writeTo("products").append()
spark.sql("UPDATE products SET price = 79.0 WHERE id = 1")
spark.sql("DELETE FROM products WHERE category = 'outerwear'")

spark.table("products").select("id", "name", "price").orderBy("id").show()
# --8<-- [end:modify]
final = spark.table("products").orderBy("id").collect()
assert [r.id for r in final] == [1, 2, 3, 6]
assert final[0].price == 79.0

# --8<-- [start:lancedb]
import lancedb

db = lancedb.connect("/tmp/lance-spark-quickstart")
db.open_table("products").search([0.85, 0.15, 0.1, 0.25]).limit(3).to_pandas()
# --8<-- [end:lancedb]
from_lancedb = (
    db.open_table("products").search([0.85, 0.15, 0.1, 0.25]).limit(3).to_pandas()
)
assert list(from_lancedb["id"]) == [1, 2, 6]
