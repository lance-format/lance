# Fragment Column Statistics Index

The fragment column statistics index stores min/max and null/NaN counts for a
column in each summarized fragment, aggregated from available
[zone write seeds](https://github.com/lance-format/lance/blob/main/docs/src/format/index/scalar/zone_write_seeds.md).

Fragment statistics and ZoneMap serve different levels. The smaller fragment
statistics index supports query planning and narrows the fragments or index
segments to search. ZoneMap stores more detailed per-zone information for finer
pruning when reading specific fragments.

## Index metadata and storage

Each logical index covers one field, identified by `IndexMetadata.fields`.
`index_details` contains `lance.index.pb.FragmentColumnStatsIndexDetails`.
This proposal defines `IndexMetadata.index_version = 0`.

Each segment has its own UUID and a `fragment_stats.lance` file in its index
directory. A logical index can have one or many segments, each summarizing a
different set of fragments.

The manifest's `fragment_bitmap` identifies fragments with usable summaries.
It must be a subset of the fragments that hold records in the segment's file:
claiming a fragment without a record would let a reader interpret "covered
but no candidate ranges" as "no matching rows" and silently drop results.
Records outside the bitmap are ignored.

A summary is valid only for the column data it summarized: replacing that
column's data file invalidates it even when the fragment ID is unchanged, and
the commit must remove the fragment from the affected coverage bitmaps. A
relevant column overlay newer than the segment's `dataset_version` also makes
the summary unusable. Deletion-vector-only changes retain summaries, because
the remaining rows are a subset of the summarized physical rows.
## File schema

`value_type` below is the indexed column's Arrow type.

```python
pa.schema([
    pa.field("fragment_id", pa.uint64(), nullable=False),
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("row_count", pa.uint64(), nullable=False),
    pa.field("null_count", pa.uint64(), nullable=False),
    pa.field("nan_count", pa.uint64(), nullable=False),
    pa.field("negative_nan_count", pa.uint64(), nullable=True),
])
```

- `min` and `max` bound non-null, non-NaN values using Arrow Rust ordering.
  Bounds may be wider than the exact extrema, but must not exclude any such
  value.
- Null bounds mean that value bounds are unavailable, not that the fragment is
  empty. Types without supported bounds may still have exact counts.
- `row_count` is the number of physical rows summarized, including deleted rows.
- `null_count` counts nulls exactly in those physical rows.
- `nan_count` counts all NaNs exactly; zero for non-floating types. NaNs do not
  appear in bounds: an extremum that is NaN is stored as a null bound.
- `negative_nan_count` counts NaNs with the sign bit set. A null value means the
  sign split is unknown, not zero. When known, it must not exceed `nan_count`;
  the positive count is `nan_count - negative_nan_count`. Non-floating types
  and records with `nan_count = 0` write zero.

Version 1 zone write seeds provide exact sign counts, as defined in
[Zone Write Seeds](https://github.com/lance-format/lance/pull/9564). Legacy seeds with nonzero `nan_count`
but no sign information produce a null `negative_nan_count`. When combining
statistics, any unknown contributing sign split keeps the aggregate unknown.
Readers must allow either NaN sign when the split is unknown and retain any
fragment whose NaNs could match; missing sign information does not invalidate
the whole record.
