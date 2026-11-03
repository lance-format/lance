---
title: PyLance Quickstart
description: Get started with PyLance, the Python SDK for the Lance format - create datasets, convert from Parquet, and learn the basics
---

# Getting Started with PyLance

PyLance is the Python SDK for the Lance format itself. It gives you direct, low-level control over Lance datasets: their files, versions, and indices. If you are building an application or a data pipeline, start with [LanceDB](index.md) or [Apache Spark](spark.md) instead. Both build on the same format and handle more of the work for you.

This quickstart guide will walk you through the core features of Lance including creating datasets, versioning, and vector search.

By the end of this tutorial, you'll be able to create Lance datasets from pandas DataFrames and convert existing Parquet files to Lance format. You'll also understand the basic workflow for working with Lance datasets and be prepared to explore advanced features like versioning and vector search.

## Install the Python SDK

Install `pylance` along with the data libraries used in these quickstarts:

```bash
--8<-- "snippets/quickstart/pylance-install.sh"
```

For the latest features and bug fixes, you can install the preview version:

=== "pip"

    ```bash
    pip install --pre --extra-index-url https://pypi.fury.io/lance-format pylance
    ```

=== "uv"

    ```bash
    uv venv
    uv pip install --prerelease allow --index https://pypi.fury.io/lance-format pylance

    # To add to pyproject.toml, just do:
    uv add --prerelease allow --index https://pypi.fury.io/lance-format pylance
    ```

!!! note
    Preview releases receive the same level of testing as regular releases.

## Set Up Your Environment

First, let's import the necessary libraries:

```python
--8<-- "snippets/quickstart/pylance_quickstart.py:imports"
```

## Create Your First Dataset

Lance is built on top of Apache Arrow, making it incredibly easy to work with pandas DataFrames and Arrow tables. You can create Lance datasets from various data sources including pandas DataFrames, Arrow tables, and existing Parquet files. Lance automatically handles the conversion and optimization for you.

### Create a Simple Dataset

You'll create a simple dataframe and then write it to Lance format. This demonstrates the basic workflow for creating Lance datasets.

Create a simple dataframe:

```python
--8<-- "snippets/quickstart/pylance_quickstart.py:create-df"
```

```output
--8<-- "snippets/quickstart/outputs/pylance_quickstart.py.create-df.txt"
```

Now you'll write this dataframe to Lance format and verify the data was saved correctly:

```python
--8<-- "snippets/quickstart/pylance_quickstart.py:write-lance"
```

```output
--8<-- "snippets/quickstart/outputs/pylance_quickstart.py.write-lance.txt"
```

### Convert Your Existing Parquet Files

You'll convert an existing Parquet file to Lance format. This shows how to migrate your existing data to Lance.

First, you'll create a Parquet file and then convert it to Lance:

```python
--8<-- "snippets/quickstart/pylance_quickstart.py:write-parquet"
```

```output
--8<-- "snippets/quickstart/outputs/pylance_quickstart.py.write-parquet.txt"
```

Now you'll convert the Parquet dataset to Lance format in a single line:

```python
--8<-- "snippets/quickstart/pylance_quickstart.py:convert-parquet"
```

```output
--8<-- "snippets/quickstart/outputs/pylance_quickstart.py.convert-parquet.txt"
```

## Next Steps

Now that you've mastered the basics of creating Lance datasets, here's what you can explore next:

- **[Versioning Your Datasets with Lance](versioning.md)** - Learn how to track changes over time with native versioning
- **[Vector Indexing and Vector Search With Lance](vector-search.md)** - Build high-performance vector search capabilities with ANN indices
