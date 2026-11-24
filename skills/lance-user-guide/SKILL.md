---
name: lance-user-guide
description: Guide Code Agents to help Lance users write/read datasets and build/choose indices. Use when a user asks how to use Lance (Python/Rust/CLI), how to write_dataset/open/scan, how to build vector indexes (IVF_PQ, IVF_HNSW_*), how to build scalar indexes (BTREE, BITMAP, LABEL_LIST, NGRAM, INVERTED, BLOOMFILTER, RTREE, etc.), how to combine filters with vector search, or how to debug indexing and scan performance.
---

# Lance User Guide

## Scope

Use this skill to answer questions about:

- Writing datasets (create/append/overwrite) and reading/scanning datasets
- Vector search (nearest-neighbor queries) and vector index creation/tuning
- Scalar index creation and choosing a scalar index type for a filter workload
- Combining filters (metadata predicates) with vector search

Do not use this skill for:

- Contributing to Lance itself (repo development, internal architecture)
- File format internals beyond what is required to use the API correctly
- Checking the health of a table or maintaining it (compaction, index optimization, cleanup of old versions): use the `lance-table-maintenance` skill

## Installation (quick)

Python:

```bash
pip install pylance
```

Verify:

```bash
python -c "import lance; print(lance.__version__)"
```

Rust:

```bash
cargo add lance
```

Or add it to `Cargo.toml` (choose an appropriate version for your project):

```toml
[dependencies]
lance = "x.y"
```

From source (this repository):

```bash
maturin develop -m python/Cargo.toml
```

## Minimal intake (ask only what you need)

Collect the minimum information required to avoid wrong guidance:

- Language/API surface: Python / Rust / CLI
- Storage: local filesystem / S3 / other object store
- Workload: scan-only / filter-heavy / vector search / hybrid (vector + filter)
- Vector details (if applicable): dimension, metric (L2/cosine/dot), latency target, recall target
- Update pattern: mostly append / frequent overwrite / frequent deletes/updates
- Data scale: approximate row count and whether there are many small files

If the user does not specify a language, default to Python examples and provide a short mapping to Rust concepts.

## Workflow decision tree

1. If the question is "How do I write or update data?": use the **Write** playbook.
2. If the question is "How do I read / scan / filter?": use the **Read** playbook.
3. If the question is "How do I do kNN / vector search?": use the **Vector search** playbook.
4. If the question is "Which index should I use?": consult `references/index-selection.md` and confirm constraints.
5. If the question is "Why is this slow / why are results missing?": use **Troubleshooting** and ask for a minimal reproduction.

## Primary playbooks (Python)

### Write

Prefer `lance.write_dataset` for most user workflows.

```python
import lance
import pyarrow as pa

vectors = pa.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    type=pa.list_(pa.float32(), 3),
)
table = pa.table({"id": [1, 2], "vector": vectors, "category": ["a", "b"]})

ds = lance.write_dataset(table, "my-data.lance", mode="create")
ds = lance.write_dataset(table, "my-data.lance", mode="append")
ds = lance.write_dataset(table, "my-data.lance", mode="overwrite")
```

Validation checklist:

- Re-open and count rows: `lance.dataset(uri).count_rows()`
- Confirm schema: `lance.dataset(uri).schema`

Notes:

- Use `storage_options={...}` when writing to an object store URI.
- If the user mentions non-atomic object stores, mention `commit_lock` and point them to the user guide.

### Read

Use `lance.dataset` + `scanner(...)` for pushdowns (projection, filter, limit, nearest).

```python
import lance

ds = lance.dataset("my-data.lance")
tbl = ds.scanner(
    columns=["id", "category"],
    filter="category = 'a' and id >= 10",
    limit=100,
).to_table()
```

Validation checklist:

- If performance is the concern, ask for a minimal `scanner(...)` call that reproduces it.
- If correctness is the concern, ask for the exact `filter` string and whether `prefilter` is enabled (when using `nearest`).

### Vector search (nearest)

Run vector search with `scanner(nearest=...)` or `to_table(nearest=...)`.

```python
import lance
import numpy as np

ds = lance.dataset("my-data.lance")
q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tbl = ds.to_table(nearest={"column": "vector", "q": q, "k": 10})
```

If combining a filter with vector search, decide whether the filter must run before the vector query:

- Use `prefilter=True` when the filter is highly selective and correctness (top-k among filtered rows) matters.
- Use `prefilter=False` when the filter is not very selective and speed matters, and accept that results can be fewer than `k`.

```python
tbl = ds.scanner(
    nearest={"column": "vector", "q": q, "k": 10},
    filter="category = 'a'",
    prefilter=True,
).to_table()
```

### Build a vector index

Create a vector index with `LanceDataset.create_index(...)`.

Start with a minimal working configuration:

```python
ds = lance.dataset("my-data.lance")
ds = ds.create_index(
    "vector",
    index_type="IVF_PQ",
    target_partition_size=8192,
    num_sub_vectors=16,
)
```

Then verify:

- `ds.describe_indices()` (preferred) or `ds.list_indices()` (can be expensive)
- A small `nearest` query that uses the index

For parameter selection and tuning, consult `references/index-selection.md`.

### Build a scalar index

Scalar indices speed up scans with filters. Use `create_scalar_index` for a stable entry point.

```python
ds = lance.dataset("my-data.lance")
ds.create_scalar_index("category", "BTREE", replace=True)
```

Then verify:

- `ds.describe_indices()`
- A representative `scanner(filter=...)` query

To choose a scalar index type (BTREE vs BITMAP vs LABEL_LIST vs NGRAM vs INVERTED, etc.), consult `references/index-selection.md`.

## Troubleshooting patterns

### "Vector search + filter returns fewer than k rows"

- Explain the difference between post-filtering and pre-filtering.
- Suggest `prefilter=True` if the user expects top-k among filtered rows.

### "Index creation is slow"

- Confirm vector dimension and `num_sub_vectors`.
- For IVF_PQ, call out the common pitfall: avoid misaligned `dimension / num_sub_vectors` (see `references/index-selection.md`).

### "Scan is slow even with a scalar index"

- Ask whether the filter is compatible with the index (equality vs range vs text search).
- Suggest checking whether scalar index usage is disabled (`use_scalar_index=False`).

## Local verification (when a repo checkout is available)

When answering API questions, confirm the exact signature and docstrings locally:

- Python I/O entry points: `python/python/lance/dataset.py` (`write_dataset`, `LanceDataset.scanner`)
- Vector indexing: `python/python/lance/dataset.py` (`create_index`)
- Scalar indexing: `python/python/lance/dataset.py` (`create_scalar_index`)

Use targeted search:

```bash
rg -n "def write_dataset\\b|def create_index\\b|def create_scalar_index\\b|def scanner\\b" python/python/lance/dataset.py
```

## Bundled resources

- Index selection and tuning: `references/index-selection.md`
- I/O and versioning cheat sheet: `references/io-cheatsheet.md`
- Runnable minimal example: `scripts/python_end_to_end.py`
