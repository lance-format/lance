---
title: Quickstart
description: Get started with Lance - create datasets, convert from Parquet, and learn the basics
---

# Getting Started with Lance Tables

This quickstart guide will walk you through the core features of Lance including creating datasets, versioning, and vector search.

By the end of this tutorial, you'll be able to create Lance datasets from pandas DataFrames and convert existing Parquet files to Lance format. You'll also understand the basic workflow for working with Lance datasets and be prepared to explore advanced features like versioning and vector search.

## Install the Python SDK

The easiest way to get started with Lance is via our Python SDK `pylance`:

```bash
pip install pylance
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
import shutil
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
```

## Create Your First Dataset

Lance is built on top of Apache Arrow, making it incredibly easy to work with pandas DataFrames and Arrow tables. You can create Lance datasets from various data sources including pandas DataFrames, Arrow tables, and existing Parquet files. Lance automatically handles the conversion and optimization for you.

### Create a Simple Dataset

You'll create a simple dataframe and then write it to Lance format. This demonstrates the basic workflow for creating Lance datasets.

Create a simple dataframe:

```python
df = pd.DataFrame({"a": [5]})
df
```

Now you'll write this dataframe to Lance format and verify the data was saved correctly:

```python
shutil.rmtree("/tmp/test.lance", ignore_errors=True)

dataset = lance.write_dataset(df, "/tmp/test.lance")
dataset.to_table().to_pandas()
```

### Convert Your Existing Parquet Files

You'll convert an existing Parquet file to Lance format. This shows how to migrate your existing data to Lance.

First, you'll create a Parquet file and then convert it to Lance:

```python
shutil.rmtree("/tmp/test.parquet", ignore_errors=True)
shutil.rmtree("/tmp/test.lance", ignore_errors=True)

tbl = pa.Table.from_pandas(df)
pa.dataset.write_dataset(tbl, "/tmp/test.parquet", format='parquet')

parquet = pa.dataset.dataset("/tmp/test.parquet")
parquet.to_table().to_pandas()
```

Now you'll convert the Parquet dataset to Lance format in a single line:

```python
dataset = lance.write_dataset(parquet, "/tmp/test.lance")

# Make sure it's the same
dataset.to_table().to_pandas()
```

## Next Steps

Now that you've mastered the basics of creating Lance datasets, here's what you can explore next:

- **[Versioning Your Datasets with Lance](versioning.md)** - Learn how to track changes over time with native versioning
- **[Vector Indexing and Vector Search With Lance](vector-search.md)** - Build high-performance vector search capabilities with ANN indices
