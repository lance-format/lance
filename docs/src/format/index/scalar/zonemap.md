# Zone Map Index

Zone maps are a columnar database technique for predicate pushdown and scan pruning.
They break data into fixed-size chunks called "zones" and maintain summary statistics
(min, max, null count) for each zone, enabling efficient filtering by eliminating
zones that cannot contain matching values.

Zone maps are "inexact" filters - they can definitively exclude zones but may include
false positives that require rechecking.

The index may also maintain a bitmap of null rows, allowing exact results for
IS NULL queries. Without it, null counts support zone pruning and candidate rows
must be rechecked.

Table writers can collect column statistics as
[write seeds](zone_write_seeds.md), using the data file's generic metadata
and global buffers. An index builder may interpret compatible seeds instead of
scanning column values. The file reader and writer do not interpret statistics
or filters. Seeds have their own format and version, separate from the index file
described below.

## Index Details

```protobuf
%%% proto.message.ZoneMapIndexDetails %%%
```

## Storage Layout

A logical ZoneMap index may contain multiple segments. Each segment stores its
zone statistics in a `zonemap.lance` file under its own index UUID directory.
Its `IndexMetadata.fragment_bitmap` identifies the fragments it covers.

### Version 1 Statistics

`IndexMetadata.index_version = 1` selects this format, independently of the
write-seed version. Each file row describes one zone; `value_type` is the indexed
column's Arrow type.

```python
pa.schema([
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("negative_nan_count", pa.uint32(), nullable=False),
    pa.field("fragment_id", pa.uint64(), nullable=False),
    pa.field("zone_start", pa.uint64(), nullable=False),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

- `min`, `max`: bounds on non-null, non-NaN values, following
  [Arrow Rust ordering](https://arrow.apache.org/rust/arrow_ord/sort/fn.sort.html).
  Infinities are included. Bounds may be wider than the exact extrema, but must
  not exclude a summarized value. Neither bound contains a NaN. Types without
  supported bounds, or zones with no ordinary values, use null bounds.
- `null_count`: number of nulls in the summarized rows. For nested values, this
  counts parent nulls, not null elements inside a non-null value.
- `nan_count`: number of NaNs of either sign.
- `negative_nan_count`: number of those NaNs with the sign bit set. Positive
  NaNs equal `nan_count - negative_nan_count`. Both counts are zero for non-float
  types, and `negative_nan_count` must not exceed `nan_count`.
- `fragment_id`, `zone_start`, `zone_length`: the fragment and its physical
  row-offset interval `[zone_start, zone_start + zone_length)`.

Counts describe the summarized rows, not necessarily the remaining live rows
after deletions. Missing bounds mean unknown, not empty. Pruning must account for
ordinary values and both NaN signs; unknown information must not exclude matches.

### Version 0

Version 0 has the same file columns except `negative_nan_count`. Its `max` may
contain a `+NaN` marker for either NaN sign, hiding the ordinary maximum. The sign
split is unknown, not zero. Readers must handle these records conservatively.
Do not relabel V0 files as V1: recovering missing statistics can require scanning
the column values.

### Schema Metadata

| Key                 | Type   | Description                               |
|---------------------|--------|-------------------------------------------|
| `rows_per_zone`     | String | Number of rows per zone (default: "8192") |
| `null_bitmap`       | UInt32 | Optional index of the null bitmap global buffer |

### Global Buffers

| Metadata Key        | Description                                                |
|---------------------|------------------------------------------------------------|
| `null_bitmap`       | A serialized RowAddrTreeMap specifying which rows are null |

The bitmap covers all zones in this segment. Its absence means exact null
positions are unavailable, not that there are no nulls. Counts remain available.
When omitted, leave out both the metadata key and the buffer, and set
`ZoneMapIndexDetails.has_null_bitmap` to false.

## Compatibility

Readers select the decoder using each segment's `IndexMetadata.index_version`.
Unsupported versions must not be interpreted using an older contract. Existing
index-version filtering and maintenance rules apply; this upgrade adds no table
reader or writer feature flag. Write-seed compatibility requirements are unchanged.

## Accelerated Queries

ZoneMap returns candidate rows for ordinary predicates; candidates require
rechecking. Null queries are exact only when an exact null bitmap is present.

| Query Type | Description               | Operation                                   | Result Type |
|------------|---------------------------|---------------------------------------------|-------------|
| **Equals** | `column = value`          | Checks bounds and NaN counts               | AtMost      |
| **Range**  | `column BETWEEN a AND b`  | Checks bounds and both NaN signs            | AtMost      |
| **IsIn**   | `column IN (v1, v2, ...)` | Includes zones that could contain any value | AtMost      |
| **IsNull** | `column IS NULL`          | Uses the bitmap, or null counts with recheck | Exact with bitmap; otherwise AtMost |
