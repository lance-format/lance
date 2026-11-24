---
title: Versioning
description: Learn how to version your Lance datasets with append, overwrite, tags, and branches
---

# Versioning Your Datasets with Lance

Lance supports versioning natively, allowing you to track changes over time.

In this tutorial, you'll learn how to append new data to existing datasets while preserving historical versions and access specific versions using version numbers or meaningful tags. You'll also understand how to implement proper data governance practices with Lance's native versioning capabilities.

## Install the Python SDK

```bash
--8<-- "snippets/quickstart/pylance-install.sh"
```

## Set Up Your Environment

First, you should import the necessary libraries:

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:imports"
```

## Create a Dataset

Start with a fresh dataset, which becomes version 1:

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:create"
```

## Append New Data to Your Dataset

You can add new rows to your existing dataset, creating a new version while preserving the original data. Here is how to append rows:

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:append"
```

```output
--8<-- "snippets/quickstart/outputs/versioning_quickstart.py.append.txt"
```

## Overwrite Your Dataset

You can completely replace your dataset with new data, creating a new version while keeping the old version accessible.

Here is how to overwrite the data and create a new version:

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:overwrite"
```

```output
--8<-- "snippets/quickstart/outputs/versioning_quickstart.py.overwrite.txt"
```

## Access Previous Dataset Versions

You can also check what versions are available and then access specific versions of your dataset.

List all versions of a dataset with `versions()` (each entry includes `version`, `timestamp`, and `metadata`):

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:versions"
```

```output
--8<-- "snippets/quickstart/outputs/versioning_quickstart.py.versions.txt"
```

If you only need version numbers, use the lightweight reference API. It lists manifest
locations without reading and deserializing every manifest:

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:version-refs"
```

```output
--8<-- "snippets/quickstart/outputs/versioning_quickstart.py.version-refs.txt"
```

Use `dataset.latest_version` instead when only the latest version of the current branch
is needed.

You can also access any available version:

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:checkout-versions"
```

```output
--8<-- "snippets/quickstart/outputs/versioning_quickstart.py.checkout-versions.txt"
```

## Tag Your Important Versions

Create named tags for important versions, making it easier to reference them by meaningful names.

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:tags"
```

```output
--8<-- "snippets/quickstart/outputs/versioning_quickstart.py.tags.txt"
```

Tags can be checked out like versions:

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:checkout-tag"
```

```output
--8<-- "snippets/quickstart/outputs/versioning_quickstart.py.checkout-tag.txt"
```

For advanced tag operations (e.g., tagging versions on specific branches), see [Tags and Branches](../guide/tags_and_branches.md).

## Work with Branches

Branches manage parallel lines of dataset evolution. You can create branches from existing versions or tags, read and write to them independently, and checkout different branches.

`main` refers to the default branch in checkout and reference APIs, but it is reserved and cannot be used as the name of a new branch.

```python
--8<-- "snippets/quickstart/versioning_quickstart.py:branches"
```

```output
--8<-- "snippets/quickstart/outputs/versioning_quickstart.py.branches.txt"
```

For more details, see [Tags and Branches](../guide/tags_and_branches.md). To preserve logical row identity across updates and compaction, see [Stable Row IDs and Lineage](../guide/row_ids.md).

## Next Steps

Now that you've mastered dataset versioning with Lance, check out **[Vector Indexing and Vector Search With Lance](vector-search.md)**. You can learn how to build high-performance vector search capabilities on top of your Lance tables.

This will teach you how to build fast, scalable search capabilities for your versioned datasets.
