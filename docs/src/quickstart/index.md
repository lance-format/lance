---
title: LanceDB Quickstart
description: Get started with Lance using LanceDB - create a table, query it, and run vector and full-text search
---

# Getting Started with LanceDB

Lance is an open table format. You can work with Lance tables from many engines, and this quickstart covers three of them:

- **LanceDB** (this page) is a retrieval database built on Lance. It runs embedded in your application or as a distributed service.
- **[Apache Spark](spark.md)** runs SQL and distributed data processing on Lance tables.
- **[PyLance](pylance.md)** is the Python SDK for the Lance format itself, for low-level control over datasets.

All three engines read and write the same format, so a table you create with one of them can be opened with the others.

In this guide, you'll create a small product catalog with LanceDB, filter it, search it by vector similarity and by keyword, modify it, and look back at earlier versions.

## Install LanceDB

This guide displays results as pandas DataFrames, so install pandas as well:

```bash
--8<-- "snippets/quickstart/lancedb-install.sh"
```

## Connect to a Database

This guide uses LanceDB in embedded mode, where a database is a directory of Lance tables, on local disk or in object storage such as `s3://`. Connect to a local path; LanceDB creates it if it doesn't exist:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:connect"
```

The same client connects to a distributed LanceDB deployment when you pass a `db://` URI instead of a path.

## Create a Table

Create a table from a list of dictionaries. LanceDB infers the schema and stores the `vector` column as a fixed-size vector:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:create"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.create.txt"
```

LanceDB also accepts pandas DataFrames, Polars DataFrames, and Arrow tables as input.

## Filter Rows

Use SQL expressions to filter rows and choose columns:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:filter"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.filter.txt"
```

## Search by Vector Similarity

Pass a query vector to `search` to find the nearest rows. The `_distance` column holds each row's distance from the query:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:vector-search"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.vector-search.txt"
```

You can combine vector search with a filter:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:filtered-vector-search"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.filtered-vector-search.txt"
```

On a small table like this one, LanceDB compares the query against every vector. For larger tables, create a [vector index](https://docs.lancedb.com/indexing) to keep searches fast.

## Search by Keyword

Create a full-text search index on a text column, then search it with keywords. The `_score` column holds each row's relevance score:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:fts"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.fts.txt"
```

## Add, Update, and Delete Rows

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:modify"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.modify.txt"
```

## Time Travel

Every change creates a new version of the table:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:version"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.version.txt"
```

Earlier versions stay readable. Check out version 1 to see the table as it was when you created it:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:checkout"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.checkout.txt"
```

Return to the latest version to continue making changes:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:checkout-latest"
```

```output
--8<-- "snippets/quickstart/outputs/lancedb_quickstart.py.checkout-latest.txt"
```

## Next Steps

- Read the [LanceDB documentation](https://docs.lancedb.com/) for embeddings, hybrid search, reranking, distributed deployments, and the TypeScript and Rust SDKs.
- See the [Spark quickstart](spark.md) to process Lance tables at scale with SQL.
- See the [PyLance quickstart](pylance.md) to work directly with the Lance format.
