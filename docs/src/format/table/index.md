# Lance Table Format

## Overview

The Lance table format organizes datasets as versioned collections of fragments, data files, deletion files, and indices. Each version is described by an immutable manifest that references the physical data for that snapshot.

The format is designed for machine learning and highly selective workloads where column additions, index maintenance, and partial rewrites must be cheap. It supports ACID transactions, schema evolution, time travel, and efficient incremental updates through Multi-Version Concurrency Control (MVCC).

## Design Goals

### Two-Dimensional Storage

Rows are partitioned into fragments, and each fragment can contain multiple data files that each provide one or more columns. This lets writers add or backfill columns by attaching new data files to existing fragments instead of rewriting the full table.

### First-Class Indices

Indices are part of the table format lifecycle. The table metadata describes index discovery and transactional coordination, while the detailed search structures remain separate index formats. This gives engines a uniform way to create, drop, update, and query indices without coupling the table format to any single indexing algorithm.

### External Manifest Store

Lance can commit directly to object storage, but deployments may also coordinate commits through an external manifest store. In that model, the external system helps serialize commits and apply governance checks, while the canonical table state is still persisted in the Lance table format.

## Manifest

![Overview](../../images/table_overview.png)

A manifest describes a single version of the dataset.
It contains the complete schema definition including nested fields, the list of data fragments comprising this version, 
a monotonically increasing version number, and an optional reference to the index section that describes a list of index metadata.

<details>
<summary>Manifest protobuf message</summary>

```protobuf
%%% proto.message.Manifest %%%
```

</details>

## Schema & Fields

The schema of the table is written as a series of fields, plus a schema metadata map.
The data types generally have a 1-1 correspondence with the Apache Arrow data types.
Each field, including nested fields, have a unique integer id. At initial table creation time, fields are assigned ids in depth-first order.
Afterwards, field IDs are assigned incrementally for newly added fields.

Column encoding configurations are specified through field metadata using the `lance-encoding:` prefix.
See [File Format Encoding Specification](../file/encoding.md) for details on available encodings, compression schemes, and configuration options.

For complete schema specification details including supported data types, field ID assignment, and metadata handling,
see the [Schema Format Specification](schema.md).

<details>
<summary>Field protobuf message</summary>

```protobuf
%%% proto.message.lance.file.Field %%%
```

</details>

### Unenforced Primary Key

Lance supports defining an unenforced primary key through field metadata.
This is useful for deduplication during merge-insert operations and other use cases that benefit from logical row identity.
The primary key is "unenforced" meaning Lance does not always validate uniqueness constraints.
Users can use specific workloads like merge-insert to enforce it if necessary.
The primary key is fixed after initial setting and must not be updated or removed.

A primary key field must satisfy:

- The field, and all its ancestors, must not be nullable.
- The field must be a leaf field (primitive data type without children).
- The field must not be within a list or map type.

When using an Arrow schema to create a Lance table, add the following metadata to the Arrow field to mark it as part of the primary key:

- `lance-schema:unenforced-primary-key`: Set to `true`, `1`, or `yes` (case-insensitive) to indicate the field is part of the primary key.
- `lance-schema:unenforced-primary-key:position` (optional): A 1-based integer specifying the position within a composite primary key.

For composite primary keys with multiple columns, the position determines the primary key field ordering:

- When positions are specified, fields are ordered by their position values (1, 2, 3, ...).
- When positions are not specified, fields are ordered by their schema field id.
- Fields with explicit positions are ordered before fields without.

## Fragments

![Fragment Structure](../../images/fragment_structure.png)

A fragment represents a horizontal partition of the dataset containing a subset of rows.
Each fragment has a unique `uint32` identifier assigned incrementally based on the dataset's maximum fragment ID.
Each fragment consists of one or more data files storing columns, plus an optional deletion file.
If present, the deletion file stores the positions (0-based) of the rows that have been deleted from the fragment.
The fragment tracks the total row count including deleted rows in its physical rows field.
Column subsets can be read without accessing all data files, and each data file is independently compressed and encoded.

<details>
<summary>DataFragment protobuf message</summary>

```protobuf
%%% proto.message.DataFragment %%%
```

</details>

### Data Evolution

This fragment design enables a new concept called data evolution, which means efficient schema evolution (add column, update column, drop column) with backfill.
For example, when adding a new column, new column data are added by appending new data files to each fragment, with values computed for all existing rows in the fragment.
There is no need to rewrite the entire table to just add data for a single column.
This enables efficient feature engineering and embedding updates for ML/AI workloads.

Each data file should contain a distinct set of field ids. 
It is not required that all field ids in the dataset schema are found in one of the data files. 
If there is no corresponding data file, that column should be read as entirely `NULL`.

Field ids might be replaced with `-2`, a tombstone value. 
In this case that column should be ignored. This used, for example, when rewriting a column: 
The old data file replaces the field id with `-2` to ignore the old data, and a new data file is appended to the fragment.

## Data Files

Data files store column data for a fragment using the Lance file format.
Each data file stores a subset of the columns in the fragment.
Field IDs are assigned either sequentially based on schema position (for Lance file format v1) 
or independently of column indices due to variable encoding widths (for Lance file format v2).

<details>
<summary>DataFile protobuf message</summary>

```protobuf
%%% proto.message.DataFile %%%
```

</details>

!!! note "Field-to-column mapping differs between data storage versions"

    In **2.0**, all fields (including non-leaf fields like struct and list containers) are assigned
    sequential column indices in `column_indices`.

    In **2.1+**, non-leaf fields (unpacked structs, list containers) are assigned `-1` in
    `column_indices` because their validity information is folded into repetition/definition
    levels. Only leaf fields and packed structs have column indices.

    See the [5.0.0 migration guide](../../guide/migration.md#500) for a detailed example.

## Deletion Files

Deletion files (a.k.a. deletion vectors) track deleted rows without rewriting data files.
Each fragment can have at most one deletion file per version.

Deletion files support two storage formats.
The Arrow IPC format (`.arrow` extension) stores a flat Int32Array of deleted row offsets and is efficient for sparse deletions.
The Roaring Bitmap format (`.bin` extension) stores a compressed roaring bitmap and is efficient for dense deletions.
Readers must filter rows whose offsets appear in the deletion file for the fragment.

Deletions can be materialized by rewriting data files with deleted rows removed.
However, this invalidates row addresses and requires rebuilding indices, which can be expensive.

<details>
<summary>DeletionFile protobuf message</summary>

```protobuf
%%% proto.message.DeletionFile %%%
```

</details>

## Data Overlay Files

!!! warning "Experimental"

    This feature is currently experimental and not yet supported in any library.

<!-- TODO: When overlay file support is implemented, update this note to state
     the released version that first supports the feature. -->

!!! note "Overlay files require feature flag 64 (data overlay files)"

Overlay files supply new values for a subset of cells within
a fragment without rewriting the base data files. They make updates cheap when only
a small percentage of rows and/or columns change: a writer appends a small file
carrying just the changed cells instead of rewriting whole columns or moving rows
to a new fragment.

For the full specification — coverage and resolution rules, dense vs. sparse layout,
versioning, index integration, compaction, and a worked example — see the
[Data Overlay Files Specification](data_overlay_file.md).



## Related Specifications

### Storage Layout

File organization, base path system, and multi-location storage.

See [Storage Layout Specification](layout.md)

### Transactions

MVCC, commit protocol, transaction types, and conflict resolution.

See [Transaction Specification](transaction.md)

### Row Lineage

Row address, Stable row ID, row version tracking, and change data feed.

See [Row ID & Lineage Specification](row_id_lineage.md)

### Indices

Vector indices, scalar indices, full-text search, and index management.

See [Index Formats](../index/index.md)

### Versioning

Feature flags and format version compatibility.

See [Format Versioning Specification](versioning.md)
