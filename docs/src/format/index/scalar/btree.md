# BTree Index

The BTree index is a two-level structure that provides efficient range queries and sorted access. 
It strikes a balance between an expensive memory structure containing all values 
and an expensive disk structure that can't be efficiently searched.

The upper layers of the BTree are designed to be cached in memory and stored in a 
BTree structure (`page_lookup.lance`), while the leaves are searched using sub-indices 
(`page_data.lance`, currently just a flat file). 
This design enables efficient memory usage - for example, with 1 billion values, 
the index can store 256K leaves of size 4K each, requiring only a few MiB of memory 
(depending on data type) for the BTree metadata while narrowing any search to just 4K values.

## Index Details

```protobuf
%%% proto.message.BTreeIndexDetails %%%
```

## Storage Layout

The BTree index consists of two files:

1. `page_lookup.lance` - The BTree structure mapping value ranges to page numbers
2. `page_data.lance` - The actual sub-indices (flat file) containing sorted values and row addresses

### Page Lookup File Schema (BTree Structure)

| Column       | Type       | Nullable | Description                                              |
|--------------|------------|----------|----------------------------------------------------------|
| `min`        | {DataType} | true     | Minimum value in the page (forms BTree keys)             |
| `max`        | {DataType} | true     | Maximum value in the page (for range pruning)            |
| `null_count` | UInt32     | false    | Number of null values in the page                        |
| `page_idx`   | UInt32     | false    | Page number pointing to the sub-index in page_data.lance |

### Schema Metadata

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | String | Number of rows per page (default: "4096") |

### Page Data File Schema (Sub-indices)

| Column   | Type       | Nullable | Description                                       |
|----------|------------|----------|---------------------------------------------------|
| `values` | {DataType} | true     | Sorted values from the indexed column (flat file) |
| `ids`    | UInt64     | false    | Physical row address (`fragment_id << 32 \| offset`) corresponding to each value. A segment persisted before format version 1 (`IndexMetadata::index_version`) stores a row id here instead. |

## Accelerated Queries

The BTree index provides exact results for the following query types:

| Query Type | Description               | Operation                                                                   |
|------------|---------------------------|-----------------------------------------------------------------------------|
| **Equals** | `column = value`          | BTree lookup to find relevant pages, then search within sub-indices         |
| **Range**  | `column BETWEEN a AND b`  | BTree traversal for pages overlapping the range, then search each sub-index |
| **IsIn**   | `column IN (v1, v2, ...)` | Multiple BTree lookups, union results from all matching sub-indices         |
| **IsNull** | `column IS NULL`          | Returns rows from all pages where null_count > 0                            |
