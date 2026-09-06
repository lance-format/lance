# Zone Map Index

Zone maps are a columnar database technique for predicate pushdown and scan pruning.
They break data into fixed-size chunks called "zones" and maintain summary statistics
(minimum, maximum, and null count) for each zone. Ordered zone maps use minimum
and maximum values to eliminate zones that cannot match. Null-only zone maps
store only top-level null information for nested values without scalar ordering.

Current writers also maintain a bitmap of null rows for exact `IS NULL` and
`IS NOT NULL` queries.

## Index Details

```protobuf
%%% proto.message.ZoneMapIndexDetails %%%
```

## Storage Layout

`zonemap.lance` is the index entry point. Each record describes one zone.
Ordered indices use the indexed logical type for extrema. Null-only indices
use Arrow `Null` extrema columns and store the indexed logical type separately.

```python
extrema_type = value_type if supports_min_max else pa.null()

pa.schema([
    pa.field("min", extrema_type, nullable=True),
    pa.field("max", extrema_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("fragment_id", pa.uint64(), nullable=False),
    pa.field("zone_start", pa.uint64(), nullable=False),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

The schema metadata contains:

- `rows_per_zone`: decimal string containing the configured maximum zone size.
- `null_bitmap`: global-buffer index of the serialized `RowAddrTreeMap`.
- `data_type`: global-buffer index of the logical type for null-only indices.

The `data_type` global buffer is an Arrow IPC schema:

```python
pa.schema([
    pa.field("value", value_type, nullable=True),
])
```

## Query Semantics

Ordered zone maps provide inexact pruning for `Equals`, `Range`, and `IsIn`.
Null-only zone maps do not advertise these predicates because they have no
value bounds. Both modes use the null bitmap for exact null predicates.

`ZoneMapIndexDetails.supports_min_max` records the mode. An absent value means
ordered mode for compatibility with existing version-0 indices.

## Index Versions

- Version 0 stores ordered zone maps.
- Version 1 stores null-only zone maps. Version-0 readers reject this version
  and fall back to scanning instead of interpreting absent value bounds.
