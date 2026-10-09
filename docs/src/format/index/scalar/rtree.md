# R-Tree Index

The R-Tree index is a static, immutable 2D spatial index. It is built on bounding boxes to organize the data. This index is intended to accelerate rectangle-based pruning.

It is designed as a multi-level hierarchical structure: leaf pages store tuples `(bbox, id=rowaddr)` for indexed geometries; branch pages aggregate child bounding boxes and store `id=pageid` pointing to child pages; a single root page encloses the entire tree. Conceptually, it can be thought of as an extension of the B+-tree to multidimensional objects, where bounding boxes act as keys for spatial pruning.

The index uses a packed-build strategy where items are first sorted and then grouped into fixed-size leaf pages.

This packed-build flow is:
- Sort items (bboxes) according to the sorting algorithm.
- Pack consecutive items into leaf pages of `page_size` entries; then build parent pages bottom-up by aggregating child page bboxes.

## Sorting

Sorting does not change the R-Tree data structure, but it is critical to performance. Currently, Hilbert sorting is implemented, but the design is extensible to other spatial sorting algorithms.

### Hilbert Curve Sorting

Hilbert sorting imposes a linear order on 2D items using a space-filling Hilbert curve to maximize locality in both axes. This improves leaf clustering, which benefits query pruning.

Hilbert sorting is performed in three steps:

1. **Global bounding box**: compute the global bbox `[xmin_g, ymin_g, xmax_g, ymax_g]` over all items for training index.
2. **Normalize and compute Hilbert value**:
    - For each item bbox `[xmin_i, ymin_i, xmax_i, ymax_i]`, compute its center:
        - `cx = (xmin_i + xmax_i) / 2`
        - `cy = (ymin_i + ymax_i) / 2`
    - Map the center to a 16‑bit grid per axis using the global bbox. Let `W = xmax_g - xmin-g` and `H = ymax_g - ymin_g`. The normalized integer coordinates are:
        - `xi = round(((cx - xmin_g) / W) * (2^16 - 1))`
        - `yi = round(((cy - ymin_g) / H) * (2^16 - 1))`
    - If the global width or height is effectively zero, the corresponding axis is treated as degenerate and set to `0` for all items (the ordering then degenerates to 1D on the other axis).
    - For each `(xi, yi)` in `[0 .. 2^16-1] × [0 .. 2^16-1]`, compute a 32‑bit Hilbert value using a standard 2D Hilbert algorithm. In pseudocode (with `bits = 16`):
      ```
      fn hilbert_value(x, y, bits):
          # x, y: integers in [0 .. 2^bits - 1]
          h = 0
          mask = (1 << bits) - 1
 
          for s from bits-1 down to 0:
              rx = (x >> s) & 1
              ry = (y >> s) & 1
              d  = ((3 * rx) XOR ry) << (2 * s)
              h  = h | d
 
              if ry == 0:
                  if rx == 1:
                      x = (~x) & mask
                      y = (~y) & mask
                  swap(x, y)
 
          return h
      ```
      - The resulting `h` is stored as the item’s Hilbert value (type `u32` with `bits = 16`).
3. **Sort**: sort items by Hilbert value.

## Index Details

```protobuf
%%% proto.message.RTreeIndexDetails %%%
```

## Storage Layout

The R-Tree index consists of two files:

1. `page_data.lance` - Stores all pages (leaf, branch) as repeated `(bbox, id)` tuples, written bottom-up (leaves first, then branch levels)
2. `nulls.lance` - Stores a serialized RowAddrTreeMap of rows with null

### Page File Schema

| Column | Type     | Nullable | Description                                                                                                                                                                                                                                                     |
|:-------|:---------|:---------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bbox` | RectType | false    | Type is Rect defined by [geoarrow-rs](https://github.com/geoarrow/geoarrow-rs) RectType; physical storage is Struct<xmin: Float64, ymin: Float64, xmax: Float64, ymax: Float64>. Represents the node bounding box (leaf: item bbox; branch: child aggregation). |
| `id`   | UInt64   | false    | Reuse the `id` column to store the physical row address (`fragment_id << 32 \| offset`) in leaf pages and `pageid` in branch pages. A segment persisted before format version 1 (`IndexMetadata::index_version`) stores row ids in leaf pages instead. |

### Nulls File Schema

| Column  | Type   | Nullable | Description                                                  |
|:--------|:-------|:---------|:-------------------------------------------------------------|
| `nulls` | Binary | false    | Serialized RowAddrTreeMap of rows with null/invalid geometry |

### Schema Metadata

The following optional keys can be used by implementations and are stored in the schema metadata:

| Key         | Type   | Description                                       |
|:------------|:-------|:--------------------------------------------------|
| `page_size` | String | Page size per page                                |
| `num_pages` | String | Total number of pages written                     |
| `num_items` | String | Number of non-null leaf items in the index        |
| `bbox`      | String | JSON-serialized global BoundingBox of the dataset |

### Query Traversal

This index serializes the multi-level hierarchical RTree structure into a single page file following the schema above. At lookup time, the reader computes each page offset using the algorithm below and reconstructs the hierarchy for traversal.

Offsets are derived from `num_items` and `page_size` of metadata as follows:

- Leaf: `leaf_pages = ceil(num_items / page_size)`; leaf `i` has `page_offset = i * page_size`.
- Branch: let `level_offset` be the starting offset for current level, which actually represents total items from all lower levels; let `prev_pages` be pages in the level below; `level_pages = ceil(prev_pages / page_size)`. For branch `j`, `page_offset = j * page_size + level_offset`.
- Iterate levels until one page remains; the root is the last page and has `pageid = num_pages - 1`.
- Page lengths: once all page offsets are collected, compute each `page_len` by the next offset difference; for the final page (root), `page_len = page_file_total_rows - page_offset` (where `page_file_total_rows` is total rows in `page_data.lance`).

Traversal starts from the root (`pageid = num_pages - 1`):

- If `page_offset < num_items` (leaf), read items `[page_offset .. page_offset + page_len)` and emit candidate row addresses matching the query bbox.
- Otherwise (branch), descend into children whose bounding boxes match the query bbox.
- Continue until there are no more pages to visit; the union of emitted row addresses forms the candidate set for evaluation.

## Accelerated Queries

The R-Tree index accelerates the following query types by returning a candidate set of matching bounding boxes. Exact geometry verification must be performed by the execution engine.

| Query Type     | Description                | Operation                                     | Result Type |
|:---------------|:---------------------------|:----------------------------------------------|:------------|
| **Intersects** | `St_Intersects(col, geom)` | Prunes candidates by bbox intersection        | AtMost      |
| **Contains**   | `St_Contains(col, geom)`   | Prunes candidates by bbox containment         | AtMost      |
| **Within**     | `St_Within(col, geom)`     | Prunes candidates by bbox within relation     | AtMost      |
| **Touches**    | `St_Touches(col, geom)`    | Prunes candidates by bbox touch relation      | AtMost      |
| **Crosses**    | `St_Crosses(col, geom)`    | Prunes candidates by bbox crossing relation   | AtMost      |
| **Overlaps**   | `St_Overlaps(col, geom)`   | Prunes candidates by bbox overlap relation    | AtMost      |
| **Covers**     | `St_Covers(col, geom)`     | Prunes candidates by bbox cover relation      | AtMost      |
| **CoveredBy**  | `St_Coveredby(col, geom)`  | Prunes candidates by bbox covered-by relation | AtMost      |
| **IsNull**     | `col IS NULL`              | Returns rows recorded in the nulls file       | Exact       |
