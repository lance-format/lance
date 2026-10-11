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
| `min`         | {DataType} | true     | Minimum value in the zone               |
| `max`         | {DataType} | true     | Maximum value in the zone               |
| `null_count`  | UInt32     | false    | Number of null values in the zone       |
| `nan_count`   | UInt32     | false    | Number of NaN values (for float types)  |
| `fragment_id` | UInt64     | false    | Fragment containing this zone           |
| `zone_start`  | UInt64     | false    | Starting row offset within the fragment |
| `zone_length` | UInt32     | false    | Number of rows in this zone             |

`min` and `max` bound the non-null, non-NaN values of the zone, compared with Arrow's
total ordering. NaNs never enter the bounds as values: `nan_count` counts them, and for
float columns `max` is written as the canonical `+NaN` as a marker that the zone holds at
least one positive NaN (a NaN with the sign bit clear, which sorts above every ordinary
value). Negative NaNs do not change `max`; a zone whose NaNs are all negative keeps its
ordinary maximum. Indices written before this convention wrote the `+NaN` marker for NaNs
of either sign, which a reader must treat as "holds NaNs of unknown sign".

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
| **Equals** | `column = value`          | Includes zones where min ≤ value ≤ max; a NaN value matches zones with nan_count > 0 | AtMost      |
| **Range**  | `column BETWEEN a AND b`  | Includes zones where ranges overlap, and every zone with nan_count > 0 | AtMost      |
| **IsIn**   | `column IN (v1, v2, ...)` | Includes zones that could contain any value | AtMost      |
| **IsNull** | `column IS NULL`          | Includes zones where null_count > 0         | Exact       |

Under Arrow's total ordering a negative NaN sorts below every other value and a positive
NaN above, so a NaN row can satisfy a range predicate on either side. The statistics do
not record the sign of a zone's NaNs, so a range query keeps every zone whose `nan_count`
is positive and leaves the exact answer to the row filter. A NaN literal used as a bound
only excludes ordinary values on the side its sign names.
