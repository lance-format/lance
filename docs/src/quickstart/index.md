---
title: LanceDB Quickstart
description: Get started with Lance using LanceDB - create a table, query it, and run vector and full-text search
---

# Getting Started with LanceDB

Lance is an open table format. You can work with Lance tables from many engines, and this quickstart covers three of them:

- **LanceDB** (this page) is an embedded database built on Lance. It is the quickest way to add search to an AI application.
- **[Apache Spark](spark.md)** runs SQL and distributed data processing on Lance tables.
- **[PyLance](pylance.md)** is the Python SDK for the Lance format itself, for low-level control over datasets.

All three engines read and write the same format, so a table you create with one of them can be opened with the others.

In this guide, you'll create a small product catalog with LanceDB, filter it, search it by vector similarity and by keyword, modify it, and look back at earlier versions.

## Install LanceDB

```bash
--8<-- "snippets/quickstart/lancedb-install.sh"
```

## Connect to a Database

A LanceDB database is a directory of Lance tables. Connect to a local path; LanceDB creates it if it doesn't exist:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:connect"
```

## Create a Table

Create a table from a list of dictionaries. LanceDB infers the schema and stores the `vector` column as a fixed-size vector:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:create"
```

LanceDB also accepts pandas DataFrames, Polars DataFrames, and Arrow tables as input.

## Filter Rows

Use SQL expressions to filter rows and choose columns:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:filter"
```

## Search by Vector Similarity

Pass a query vector to `search` to find the nearest rows. The `_distance` column holds each row's distance from the query:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:vector-search"
```

You can combine vector search with a filter:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:filtered-vector-search"
```

On a small table like this one, LanceDB compares the query against every vector. For larger tables, create a [vector index](https://docs.lancedb.com/indexing) to keep searches fast.

## Search by Keyword

Create a full-text search index on a text column, then search it with keywords:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:fts"
```

## Add, Update, and Delete Rows

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:modify"
```

## Time Travel

Every change creates a new version of the table, and earlier versions stay readable:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:versions"
```

Return to the latest version to continue making changes:

```python
--8<-- "snippets/quickstart/lancedb_quickstart.py:restore"
```

## Next Steps

- Read the [LanceDB documentation](https://docs.lancedb.com/) for embeddings, hybrid search, reranking, and the TypeScript and Rust SDKs.
- See the [Spark quickstart](spark.md) to process Lance tables at scale with SQL.
- See the [PyLance quickstart](pylance.md) to work directly with the Lance format.
