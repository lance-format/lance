# Storage Layout Specification

## Overview

This specification defines how Lance datasets are organized on object storage.
The layout design emphasizes portability, allowing datasets to be relocated or referenced across multiple storage systems with minimal metadata changes.

## Dataset Root

The dataset root is the location where the dataset was initially created.
Every Lance dataset has exactly one dataset root, which serves as the primary storage location for the dataset's files.
The dataset root contains the standard subdirectory structure (`data/`, `_versions/`, `_deletions/`, `_indices/`, `_refs/`, `tree/`) that organizes the dataset's files.

## Basic Layout

A Lance dataset in its basic form stores all files within the dataset root directory structure:

```
{dataset_root}/
    data/
        *.lance           -- Data files containing column data
    _versions/
        *.manifest                -- Manifest files (one per version)
        latest_version_hint.json  -- Optional hint of the latest version (see below)
    _transactions/
        *.txn             -- Transaction files for commit coordination
    _deletions/
        *.arrow           -- Deletion vector files (arrow format)
        *.bin             -- Deletion vector files (bitmap format)
    _indices/
        {UUID}/
            ...           -- Index content (different for each index type)
    _refs/
        tags/
            *.json        -- Tag metadata
        branches/
            *.json        -- Branch metadata
    _bt/
        ...               -- Fragment metadata tree objects (see Fragment Metadata Tree)
    tree/
        {branch_name}/
            ...           -- Branch dataset
            
```

## Base Path System

### BasePath Message

The manifest's `base_paths` field contains an array of `BasePath` entries that define alternative storage locations for dataset files.
Each base path entry has a unique numeric identifier that file metadata can reference to indicate where files are located.
The `path` field specifies an absolute path interpretable by the object store.
The `is_dataset_root` field determines how the path is interpreted: when true, the path points to a dataset root with standard subdirectories (`data/`, `_deletions/`, `_indices/`); when false, the path points directly to a file directory without subdirectories.
An optional `name` field provides a human-readable alias, which is particularly useful for referencing tags in shallow clones.

<details>
<summary>BasePath protobuf message</summary>

```protobuf
message BasePath {
  uint32 id = 1;
  optional string name = 2;
  bool is_dataset_root = 3;
  string path = 4;
}
```

</details>

### File Metadata Base References

Three types of files can specify alternative base paths: data files, deletion files, and index metadata.
Each of these file types includes an optional `base_id` field in their metadata that references a base path entry by its numeric identifier.
When a file's `base_id` is absent, the file is located relative to the dataset root.
When a file's `base_id` is present, readers must look up the corresponding base path entry in the manifest's `base_paths` array to determine where the file is stored.

Fragment metadata children and fragment files stored in the tree also carry optional `base_id`. Resolution is specified in [Fragment Metadata Tree](fragment_metadata.md#object-references).

At read time, path resolution follows a two-step process.
First, the reader determines the base path: if `base_id` is absent, the base path is the dataset root; otherwise, the reader looks up the base path entry using the `base_id` to obtain the path and its `is_dataset_root` flag.
Second, the reader constructs the full file path based on whether the base path represents a dataset root.
For dataset roots (when `is_dataset_root` is true), the full path includes standard subdirectories: data files are located under `data/`, deletion files under `_deletions/`, and indices under `_indices/`.
For non-root base paths (when `is_dataset_root` is false), the base path points directly to the file directory, and the file path is appended directly without subdirectory prefixes.

### Example Complex Layout Scenarios

#### Hot/Cold Tiering

```
Manifest base_paths:
[
  { id: 0, is_dataset_root: true, path: "s3://hot-bucket/dataset" },
  { id: 1, is_dataset_root: true, path: "s3://cold-bucket/dataset-archive" }
]

Fragment 0 (recent data):
  DataFile { path: "fragment-0.lance", base_id: 0 }
  → resolves to: s3://hot-bucket/dataset/data/fragment-0.lance

Fragment 100 (historical data):
  DataFile { path: "fragment-100.lance", base_id: 1 }
  → resolves to: s3://cold-bucket/dataset-archive/data/fragment-100.lance
```

This allows seamless querying across storage tiers without data movement.

#### Multi-Region Distribution

```
Manifest base_paths:
[
  { id: 0, is_dataset_root: true, path: "s3://us-east-bucket/dataset" },
  { id: 1, is_dataset_root: true, path: "s3://eu-west-bucket/dataset" },
  { id: 2, is_dataset_root: true, path: "s3://ap-south-bucket/dataset" }
]

Fragments distributed by data locality:
  Fragment 0 (US users): base_id: 0
  Fragment 1 (EU users): base_id: 1
  Fragment 2 (Asia users): base_id: 2
```

Compute jobs can read data from the nearest region without data transfer.

#### Shallow Clone

Shallow clones create a new dataset that references data files from a source dataset without copying:

**Example: Shallow Clone**

```
Source dataset: s3://production/main-dataset
Clone dataset:  s3://experiments/test-variant

Clone manifest base_paths:
[
  { id: 0, is_dataset_root: true, path: "s3://experiments/test-variant" },
  { id: 1, is_dataset_root: true, path: "s3://production/main-dataset",
    name: "v1.0" }
]

Original fragments (inherited):
  DataFile { path: "fragment-0.lance", base_id: 1 }
  → resolves to: s3://production/main-dataset/data/fragment-0.lance

New fragments (clone-specific):
  DataFile { path: "fragment-new.lance", base_id: 0 }
  → resolves to: s3://experiments/test-variant/data/fragment-new.lance
```

The clone can append new data, modify schemas, or delete rows without affecting the source dataset.
Only the manifest and new data files are stored in the clone location.

**Workflow:**

1. [Clone transaction](transaction.md#clone) creates new manifest in target location
2. Manifest includes base path pointing to source dataset
3. Original fragments reference source via `base_id: 1`
4. Subsequent writes reference clone location via `base_id: 0`
5. Source dataset remains immutable and can be garbage collected independently

## Dataset Portability

The base path system combined with relative file references provides strong portability guarantees for Lance datasets.
All file paths within Lance files are stored relative to their containing directory, enabling datasets to be relocated without file modifications.

To port a dataset to a new location, simply copy all contents from the dataset root directory.
The copied dataset will function immediately at the new location without any manifest updates, as all file references within the dataset root resolve through relative paths.

When a dataset uses multiple base paths (such as in shallow clones or multi-bucket configurations), users have flexibility in how to port the dataset.
The simplest approach is to copy only the dataset root, which preserves references to the original base path locations.
Alternatively, users can copy additional base paths to the new location and update the manifest's `base_paths` array to reflect the new base paths.
Since only the `base_paths` field in the manifest requires modification, this remains a lightweight metadata operation that does not require rewriting additional metadata or data files.

## File Naming Conventions

### Data Files

Pattern: `data/{uuid-based-filename}.lance`

Data files use UUID-based filenames optimized for S3 throughput.
The filename is generated from a UUID (16 bytes) by converting the first 3 bytes to a 24-character binary string and the remaining 13 bytes to a 26-character hex string, resulting in a 50-character filename.
The binary prefix (rather than hex) provides maximum entropy per character, allowing S3's internal partitioning to quickly recognize access patterns and scale appropriately, minimizing throttling.

Example: `data/101100101101010011010110a1b2c3d4e5f6g7h8i9j0.lance`

### Deletion Files

Pattern: `_deletions/{fragment_id}-{read_version}-{id}.{extension}`

Deletion files use two extensions: `.arrow` for Arrow IPC format (sparse deletions) and `.bin` for Roaring bitmap format (dense deletions).

Example: `_deletions/42-10-a1b2c3d4.arrow`

### Transaction Files

Pattern: `_transactions/{read_version}-{uuid}.txn`

Where `read_version` is the table version the transaction was built from.

Example: `_transactions/5-550e8400-e29b-41d4-a716-446655440000.txn`

### Manifest Files

Manifest files are stored in the `_versions/` directory with naming schemes that support atomic commits.

See [Manifest Naming Schemes](transaction.md#manifest-naming-schemes) for details on the V1 and V2 patterns and their implications for version discovery.

### Version Hint

The optional file `_versions/latest_version_hint.json` records the latest committed version as JSON:

```json
{"version": 42}
```

It exists to accelerate latest-version discovery on stores where listing `_versions/` is expensive: a reader can read the hint and probe higher versions with HEAD requests instead of listing the whole directory, falling back to a full listing if the hint is missing or stale.

The hint is purely an optimization. It is always safe to delete, never affects correctness, and can be ignored by readers that don't understand it. Writers may choose not to write it.

