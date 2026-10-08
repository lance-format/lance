## Index selection (quick)

Use this file when the user asks "which index should I use" or "how do I tune it".

Always confirm:

- The query pattern (filter-only, vector-only, hybrid)
- Data scale (rows, vector dimension)
- Update pattern (append vs frequent updates/deletes)
- Correctness needs (must return top-k within a filtered subset vs best-effort)

## Decision table

| Workload | Recommended starting point | Notes |
| --- | --- | --- |
| Filter-only scans (`scanner(filter=...)`) | Create a scalar index on the filtered column | Choose scalar index type based on predicate shape and cardinality |
| Vector search only (`nearest=...`) on large data | Build a vector index | Start with `IVF_PQ` if you need compression; tune `nprobes` / `refine_factor` |
| Vector search + selective filter | Scalar index for filter + vector index for search | Use `prefilter=True` when you need true top-k among filtered rows |
| Vector search + non-selective filter | Vector index only (or scalar index optional) | Consider `prefilter=False` for speed; accept fewer than k results |
| Text search | Create an `INVERTED` scalar index | Use `full_text_query=...` when available; note that `FTS` is not a universal alias in all SDK versions |

## Vector index types (user-facing summary)

Vector index names typically follow a pattern like `{clustering}_{sub_index}_{quantization}`.

Common combinations:

- `IVF_PQ`: IVF clustering + PQ compression
- `IVF_HNSW_SQ`: IVF clustering + HNSW + SQ
- `IVF_SQ`: IVF clustering + SQ
- `IVF_RQ`: IVF clustering + RQ
- `IVF_HNSW_RQ`: IVF clustering + HNSW + multi-bit RQ (`num_bits > 1`)
- `IVF_FLAT`: IVF clustering + no quantization (exact vectors within clusters)

If you are unsure which types are supported in the user's environment, recommend starting with `IVF_PQ` and fall back to "try and see" (the API will error on unsupported types).

## Vector index creation defaults

Start with:

- `index_type="IVF_PQ"`
- `target_partition_size`: start with 8192 and adjust based on the dataset size and latency/recall needs
- `num_sub_vectors`: choose a value that divides the vector dimension

Practical warning (performance):

- Avoid misalignment: `(dimension / num_sub_vectors) % 8 == 0` is a common sweet spot for faster index creation.

## Vector search tuning defaults

Tune recall vs latency with:

- `nprobes`: how many IVF partitions to search
- `refine_factor`: how many candidates to re-rank to improve accuracy

When a user reports "too slow" or "bad recall", ask for:

- Current `nprobes`, `refine_factor`, and index type
- Whether the query is using `prefilter`

## Scalar index selection (starting guidance)

Choose scalar index type based on the filter expression:

- Equality filters on high-cardinality columns: start with `BTREE`
- Equality / IN-list filters on low-cardinality columns: start with `BITMAP`
- List membership filters on list-like columns: start with `LABEL_LIST`
- Substring / `contains(...)` filters on strings: start with `NGRAM`
- Full-text search (FTS): start with `INVERTED`
- Range filters: start with range-friendly options (for example `ZONEMAP` when appropriate)
- Highly selective negative membership / presence checks: consider `BLOOMFILTER` (inexact)
- Geospatial queries (if present in your build): use `RTREE`

## JSON fields

Lance scalar indices are created on physical columns. If you want to index a JSON sub-field:

1. Materialize the extracted value into a new column (for example with `add_columns`)
2. Create a scalar index on that new column

Example (Python, using SQL expressions):

```python
ds = lance.dataset(uri)
ds.add_columns({"country": "json_extract(payload, '$.country')"})
ds.create_scalar_index("country", "BTREE", replace=True)
```

If you cannot confidently map the filter to an index type, recommend `BTREE` as a safe baseline and confirm via a small benchmark on representative queries.
