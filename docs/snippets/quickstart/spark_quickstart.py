# Runnable source for the PySpark tabs in docs/src/quickstart/spark.md. The
# sections between the `--8<--` markers are embedded in the page, and CI checks
# what each one displays against outputs/ (see tests/snippet_runner.py).
# Sections import what they use and end with an expression to display, like
# notebook cells.
# ruff: noqa: E402, B018
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

# fmt: off
# --8<-- [start:create]
products = spark.createDataFrame(
    [
        (1, "red running shoes", "shoes", 89.0, [1.0, 0.0, 0.0, 0.25]),
        (2, "blue trail running shoes", "shoes", 120.0, [0.875, 0.125, 0.0, 0.375]),
        (3, "leather hiking boots", "shoes", 150.0, [0.625, 0.0, 0.5, 0.25]),
        (4, "waterproof rain jacket", "outerwear", 99.0, [0.0, 1.0, 0.25, 0.0]),
        (5, "wool winter coat", "outerwear", 210.0, [0.25, 0.75, 0.625, 0.0]),
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
# fmt: on

# --8<-- [start:query]
from pyspark.sql import functions as F

table = spark.table("products")
(
    table.filter("price < 100")
    .select("id", "name", "price")
    .orderBy("id")
    .show(truncate=False)
)

(
    table.groupBy("category")
    .agg(F.count("*").alias("num_products"), F.avg("price").alias("avg_price"))
    .orderBy("category")
    .show(truncate=False)
)
# --8<-- [end:query]

# --8<-- [start:vector-search]
spark.sql("""
    SELECT id, name, _distance
    FROM VECTOR_SEARCH(
        table => 'products',
        query_vector => array(0.875, 0.125, 0.0, 0.25),
        num_results => 3
    )
    ORDER BY _distance
""").show(truncate=False)
# --8<-- [end:vector-search]

# --8<-- [start:fts]
spark.sql("ALTER TABLE products CREATE INDEX name_idx USING fts (name)")

(
    spark.table("products")
    .filter("lance_match(name, 'running')")
    .select("id", "name")
    .orderBy("id")
    .show(truncate=False)
)
# --8<-- [end:fts]

# fmt: off
# --8<-- [start:modify]
socks = spark.createDataFrame(
    [(6, "trail socks", "accessories", 15.0, [0.75, 0.25, 0.25, 0.25])],
    "id BIGINT, name STRING, category STRING, price DOUBLE, vector ARRAY<FLOAT>",
)
socks.writeTo("products").append()
spark.sql("UPDATE products SET price = 79.0 WHERE id = 1")
spark.sql("DELETE FROM products WHERE category = 'outerwear'")

spark.table("products").select("id", "name", "price").orderBy("id").show(truncate=False)
# --8<-- [end:modify]
# fmt: on

# --8<-- [start:lancedb]
import lancedb

db = lancedb.connect("/tmp/lance-spark-quickstart")
(
    db.open_table("products")
    .search([0.875, 0.125, 0.0, 0.25])
    .limit(3)
    .select(["id", "name", "_distance"])
    .to_pandas()
)
# --8<-- [end:lancedb]
