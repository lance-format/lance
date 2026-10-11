# Zone Map Index

Zone maps are a columnar database technique for predicate pushdown and scan pruning.
They break data into fixed-size chunks called "zones" and maintain summary statistics
(min, max, null count) for each zone, enabling efficient filtering by eliminating
zones that cannot contain matching values.

Zone maps are "inexact" filters - they can definitively exclude zones but may include
false positives that require rechecking.

In addition, since finding NULLs is a common query pattern, the index also maintains a
bitmap of null rows which allows it to return exact results for IS NULL queries.

## Index Details

```protobuf
%%% proto.message.ZoneMapIndexDetails %%%
```

## Storage Layout

The zone map index stores zone statistics in a single file:

1. `zonemap.lance` - Zone statistics for query pruning

### Zone Statistics File Schema

| Column        | Type       | Nullable | Description                             |
|---------------|------------|----------|-----------------------------------------|
| `min`         | {DataType} | true     | Minimum of the non-null, non-NaN values; null if there are none |
| `max`         | {DataType} | true     | Maximum of the non-null, non-NaN values; null if there are none. Written as `+NaN` when the zone holds a positive NaN (see below) |
| `null_count`  | UInt32     | false    | Number of null values in the zone       |
| `nan_count`   | UInt32     | false    | Number of NaN values of either sign (float types; 0 otherwise) |
| `negative_nan_count` | UInt32 | true  | Number of the `nan_count` NaNs with the sign bit set (float types; 0 otherwise). Null when unknown |
| `fragment_id` | UInt64     | false    | Fragment containing this zone           |
| `zone_start`  | UInt64     | false    | Starting row offset within the fragment |
| `zone_length` | UInt32     | false    | Number of rows in this zone             |

### How nulls and NaNs are accounted

A zone's rows fall into four groups, and the statistics describe each group on
its own:

- **Nulls** are counted in `null_count` and never enter `min` or `max`.
- **Ordinary values** (non-null, non-NaN) are bounded by `min` and `max`, compared
  with Arrow's total ordering (`-0.0 < 0.0`, `-inf` and `inf` are ordinary values).
  A zone without ordinary values stores null bounds.
- **NaNs** never enter `min` or `max` as extrema. `nan_count` counts all of them and
  `negative_nan_count` counts those with the sign bit set, so the number of
  positive NaNs is `nan_count - negative_nan_count`. Arrow's total ordering places
  a negative NaN below every other value and a positive NaN above, which is why
  the two signs are tracked separately: a negative NaN matches `x < 0`, a positive
  NaN matches `x > 100`.

A reader keeps a zone when any group can match the query: nulls for `IS NULL`,
ordinary values when `[min, max]` intersects the query range, negative NaNs when
the range is open below or starts at a negative NaN, positive NaNs when the range
is open above or ends at a positive NaN. A NaN literal as a bound never rules out
NaNs of its own sign, because same-sign NaNs still order by payload and the
counts do not record payloads.

#### Backward compatibility

Two rules exist only so that readers and writers from before `negative_nan_count`
interoperate with current ones:

- **`max` is written as `+NaN` when the zone holds a positive NaN.** Readers that
  predate `negative_nan_count` learn that a zone holds NaNs only from this `+NaN`
  max, and a positive NaN is the largest value in the zone under Arrow's total
  ordering, so the bound is still correct. The ordinary maximum is hidden behind
  it; a reader that needs it must scan. A zone whose NaNs are all negative keeps
  its ordinary maximum.
- **A null `negative_nan_count` means the sign split is unknown**, not zero.
  Indices and write seeds from before this column wrote `+NaN` as the max for any
  NaN and did not record signs. A reader treats such a zone with `nan_count > 0`
  as possibly holding NaNs of both signs and prunes it only when neither sign nor
  the ordinary values can match. Such zones are not rewritten; the index merges
  them as they are, and they stay unknown until the zone is rebuilt.

### Schema Metadata

| Key                 | Type   | Description                               |
|---------------------|--------|-------------------------------------------|
| `rows_per_zone`     | String | Number of rows per zone (default: "8192") |
| `null_bitmap`       | UInt32 | Index of null bitmap global buffer        |

### Global Buffers

| Metadata Key        | Description                                                |
|---------------------|------------------------------------------------------------|
| `null_bitmap`       | A serialized RowAddrTreeMap specifying which rows are null |

## Accelerated Queries

The zone map index provides inexact results for the following query types (nullability queries
return exact results):

| Query Type | Description               | Operation                                   | Result Type |
|------------|---------------------------|---------------------------------------------|-------------|
| **Equals** | `column = value`          | Includes zones where min ≤ value ≤ max, or whose NaN counts allow a NaN value | AtMost      |
| **Range**  | `column BETWEEN a AND b`  | Includes zones where ranges overlap, or whose NaN counts allow a NaN inside the range | AtMost      |
| **IsIn**   | `column IN (v1, v2, ...)` | Includes zones that could contain any value | AtMost      |
| **IsNull** | `column IS NULL`          | Includes zones where null_count > 0         | Exact       |
