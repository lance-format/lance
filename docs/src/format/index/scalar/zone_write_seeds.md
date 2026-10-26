# Zone Write Seeds

Write seeds save column statistics while writing a data file. Each column is
divided into zones, normally 8,192 rows each, with one statistics record per zone.

The table layer can use these statistics for zone pruning, aggregate them for
fragment or index segment pruning, or build an index without rereading the column.

This proposal defines version 1 seeds. It does not change the ZoneMap index format
or enable version 1 writes; implementation follows separately.

## Arrow Rust ordering

Zone statistics follow [Arrow Rust ordering](https://arrow.apache.org/rust/arrow_ord/sort/fn.sort.html).

## Version 1 statistics fields

- `min`, `max`: bounds on non-null, non-NaN values. Bounds may be wider than the
  exact extrema, but must not exclude any such value.
- `null_count`: exact number of null rows.
- `nan_count`: exact number of NaNs of either sign.
- `negative_nan_count`: exact number of NaNs with the sign bit set. The positive
  count is `nan_count - negative_nan_count`.
- `zone_length`: number of physical rows summarized, before later deletions.

Types without supported bounds, such as whole `List`, `FixedSizeList`, or `Struct`
values, use null bounds. Their null counts and optional bitmap describe the outer
column, not child values. Non-float types write zero for both NaN counts.

A null bound means unknown unless the counts show that no ordinary values remain:
`zone_length = null_count + nan_count`. Unknown bounds cannot exclude values.
Counts must be exact, fit their types, and satisfy
`null_count + nan_count <= zone_length` and `negative_nan_count <= nan_count`.
Consumers must validate counts without arithmetic overflow.

## Storage layout

Data-file schema metadata locates each column's seed:
`lance.seed.<column_name> = <buffer_index>:<rows_per_zone>`.
The referenced global buffer contains a complete Arrow IPC file.

Each zone covers `zone_length` consecutive physical rows. Together, the zones
cover the file's column data in order, without gaps or overlaps.

### Version 1 seed schema

Each row describes one zone. `value_type` is the column's Arrow type.

```python
pa.schema([
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("negative_nan_count", pa.uint32(), nullable=False),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

### Seed schema metadata

- `version`: `"1"` for this seed format. Absence means version 0. It is not the
  data-file or Arrow IPC version.
- `seed_null_bitmap`: optional hex-encoded Roaring bitmap of exact null offsets,
  relative to the seed's physical row sequence. One bitmap covers all zones in
  this column's buffer and must agree with their null counts. Absence means null
  positions are unavailable, not that there are no nulls. Its encoding is unchanged.

`rows_per_zone` is stored in the outer locator, not the IPC schema metadata.

## Version 0: existing seeds

Seeds without `version` metadata use this existing schema:

```python
pa.schema([
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("nan_count", pa.uint32(), nullable=False),
    pa.field("zone_length", pa.uint64(), nullable=False),
])
```

Version 1 adds the version metadata and `negative_nan_count`, and removes NaNs
from min/max. Version 0's `max` may contain a `+NaN` marker for either NaN sign;
it proves neither a positive NaN nor an ordinary upper bound. If `nan_count > 0`,
both signs remain possible.

Existing buffers remain version 0. Do not relabel them or invent sign counts.
Producing version 1 seeds requires collecting statistics from the column values.

## Reader and writer compatibility

The first commit referencing version 1 seeds must atomically set
`FLAG_ZONE_WRITE_SEEDS` (`1 << 16`) in both `reader_feature_flags` and
`writer_feature_flags`. Unsupported clients must refuse the dataset: the seed's
version metadata alone cannot protect old readers that ignore it.

Activation is explicit, not triggered by opening a table. Both bits remain set
in derived snapshots, including clone and restore, even after dropping an index.
Historical snapshots are unchanged; version 0 and version 1 seeds may coexist.

The bit remains reserved and unsupported until runtime support is implemented.
