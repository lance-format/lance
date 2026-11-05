---
title: Apache Spark Quickstart
description: Get started with Lance using Apache Spark - create a table, query it with SQL, and run vector and full-text search
---

# Getting Started with Apache Spark

The [Lance Spark connector](../integrations/spark/index.md) lets Apache Spark read and write Lance tables, so you can work with your AI data using Spark SQL and DataFrames. In this guide, you'll run Spark on your machine, create a product catalog, query it, search it by vector similarity and by keyword, and modify it. Finally, you'll open the same table from LanceDB.

Each step shows both Spark SQL and PySpark. Pick whichever you prefer.

## Install Spark

The connector needs Java 17 or later. Install PySpark, which includes the `spark-sql` shell, along with LanceDB and pandas for the last step:

```bash
--8<-- "snippets/quickstart/spark-install.sh"
```

## Start Spark

Start Spark with the Lance connector and a Lance catalog named `lance`. The `dir` catalog keeps tables in a local directory. For production, the connector also supports catalogs such as Hive, AWS Glue, and REST catalogs; see [Configuration](../integrations/spark/config.md).

=== "SQL"

    ```bash
    --8<-- "snippets/quickstart/spark-sql.sh"
    ```

=== "PySpark"

    ```python
    --8<-- "snippets/quickstart/spark_quickstart.py:session"
    ```

The `--packages` option downloads the connector from Maven Central the first time you run it. In `spark-sql`, the last option prints column headers with query results. Use the bundle that matches your Spark version: `lance-spark-bundle-4.2_2.13` is for Spark 4.2 with Scala 2.13.

## Create a Table

Lance stores vectors as fixed-size lists. Spark has no fixed-size array type, so declare the vector column as `ARRAY<FLOAT>` and set its dimension with a table property:

=== "SQL"

    ```sql
    --8<-- "snippets/quickstart/spark_quickstart.sql:create"
    ```

=== "PySpark"

    ```python
    --8<-- "snippets/quickstart/spark_quickstart.py:create"
    ```

## Query the Table

Lance tables work with standard Spark SQL. The connector pushes filters and column selections down to Lance, so Spark reads only the data it needs:

=== "SQL"

    ```sql
    --8<-- "snippets/quickstart/spark_quickstart.sql:query"
    ```

    ```output
    --8<-- "snippets/quickstart/outputs/spark_quickstart.sql.query.txt"
    ```

=== "PySpark"

    ```python
    --8<-- "snippets/quickstart/spark_quickstart.py:query"
    ```

    ```output
    --8<-- "snippets/quickstart/outputs/spark_quickstart.py.query.txt"
    ```

## Search by Vector Similarity

The `VECTOR_SEARCH` table function returns the rows nearest to a query vector, along with their `_distance`:

=== "SQL"

    ```sql
    --8<-- "snippets/quickstart/spark_quickstart.sql:vector-search"
    ```

    ```output
    --8<-- "snippets/quickstart/outputs/spark_quickstart.sql.vector-search.txt"
    ```

=== "PySpark"

    ```python
    --8<-- "snippets/quickstart/spark_quickstart.py:vector-search"
    ```

    ```output
    --8<-- "snippets/quickstart/outputs/spark_quickstart.py.vector-search.txt"
    ```

`VECTOR_SEARCH` also takes filters and index tuning options. See [Vector Search](../integrations/spark/operations/dql/vector-search.md).

## Search by Keyword

Create a full-text search index on a text column, then search it with `lance_match`:

=== "SQL"

    ```sql
    --8<-- "snippets/quickstart/spark_quickstart.sql:fts"
    ```

    ```output
    --8<-- "snippets/quickstart/outputs/spark_quickstart.sql.fts.txt"
    ```

=== "PySpark"

    ```python
    --8<-- "snippets/quickstart/spark_quickstart.py:fts"
    ```

    ```output
    --8<-- "snippets/quickstart/outputs/spark_quickstart.py.fts.txt"
    ```

## Add, Update, and Delete Rows

=== "SQL"

    ```sql
    --8<-- "snippets/quickstart/spark_quickstart.sql:modify"
    ```

    ```output
    --8<-- "snippets/quickstart/outputs/spark_quickstart.sql.modify.txt"
    ```

=== "PySpark"

    ```python
    --8<-- "snippets/quickstart/spark_quickstart.py:modify"
    ```

    ```output
    --8<-- "snippets/quickstart/outputs/spark_quickstart.py.modify.txt"
    ```

## Open the Table from LanceDB

Spark wrote an ordinary Lance table, so other engines can use it directly. For example, you can connect LanceDB to the same directory and search the table from Python:

```python
--8<-- "snippets/quickstart/spark_quickstart.py:lancedb"
```

```output
--8<-- "snippets/quickstart/outputs/spark_quickstart.py.lancedb.txt"
```

## Next Steps

- Read the [Spark connector documentation](../integrations/spark/index.md) for the full set of supported DDL, DML, and query operations.
- See the [LanceDB quickstart](index.md) to build search into an application.
- See the [PyLance quickstart](pylance.md) to work directly with the Lance format.
