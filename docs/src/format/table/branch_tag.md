# Branch and Tag Specification

## Overview

Lance supports branching and tagging for managing multiple independent version histories and creating named references to specific versions.
Branches enable parallel development workflows, while tags provide stable named references for important versions.

## Branching

### Branch Name

Branch names must follow these validation rules:

1. Cannot be empty
2. Cannot start or end with `/`
3. Cannot contain consecutive `//`
4. Cannot contain `..` or `\`
5. Segments must contain only alphanumeric characters, `.`, `-`, `_`
6. Cannot end with `.lock`
7. Cannot be named `main` (reserved for the default branch)

Branch names are case-sensitive, matching Git/GitHub ref semantics. The exact name `main` is a virtual name for the default branch. It may appear in API reference contexts as an alias for the default branch, but no branch metadata file named `main.json` is created.

### Branch Metadata Path

Branch metadata is stored at `_refs/branches/{branch-name}.json` in the dataset root.
Since branch names support hierarchical naming with `/` characters, the `/` is URL-encoded as `%2F` in the filename to distinguish it from directory separators (e.g., `bugfix/issue-123` becomes `bugfix%2Fissue-123.json`):

```
{dataset_root}/
    _refs/
        branches/
            feature-a.json
            bugfix%2Fissue-123.json  # Note: '/' encoded as '%2F'
```

### Branch Metadata File Format

Each branch metadata file is a JSON file with the following fields:

| JSON Key         | Type   | Optional | Description                                                                    |
|------------------|--------|----------|--------------------------------------------------------------------------------|
| `parentBranch`   | string | Yes      | Name of the branch this was created from. `null` indicates branched from the default branch. |
| `parentVersion`  | number |          | Version number of the parent branch at the time this branch was created.       |
| `createAt`       | number |          | Unix timestamp (seconds since epoch) when the branch was created.              |
| `manifestSize`   | number |          | Size of the initial manifest file in bytes.                                    |
| `metadata`       | object | Yes      | String key/value metadata map. If absent, it is treated as an empty object.    |

### Branch Dataset Layout

Each branch dataset is technically a [shallow clone](layout.md#shallow-clone) of the source dataset.
Branch datasets are organized using the `tree/` directory at the dataset root:

```
{dataset_root}/
    tree/
        {branch_name}/
            _versions/
                *.manifest
            _transactions/
                *.txn
            _deletions/
                *.arrow
                *.bin
            _indices/
                {UUID}/
                    index.idx
            _retention/
                *.json
```

Named branches store their version-specific files under `tree/{branch_name}/`, resembling the GitHub branch path convention.
It uses the branch name as is to form the path, 
which means `/` would create a logical subdirectory (e.g., `bugfix/issue-123`, `feature/user-auth`):

```
{dataset_root}/
    tree/
        feature-a/
            _versions/
                1.manifest
                2.manifest
        bugfix/
            issue-123/
                _versions/
                    1.manifest
```

## Tagging

### Tag Name

Tag names must follow these validation rules:

1. Cannot be empty
2. Must contain only alphanumeric characters, `.`, `-`, `_`
3. Cannot start or end with `.`
4. Cannot end with `.lock`
5. Cannot contain consecutive `..`

Note that tag names do not support `/` characters, unlike branch names.

### Tag Storage

Tags are stored as JSON files under `_refs/tags/` at the dataset root:

```
{dataset_root}/
    _refs/
        tags/
            v1.0.0.json
            v1.1.0.json
            production.json
```

Tags are always stored at the root dataset level, regardless of which branch they reference.

### Tag File Format

Each tag file is a JSON file with the following fields:

| JSON Key        | Type   | Optional | Description                                                              |
|-----------------|--------|----------|--------------------------------------------------------------------------|
| `branch`        | string | Yes      | Branch name being tagged. `null` or absent indicates the default branch. |
| `version`       | number |          | Version number being tagged within that branch.                          |
| `createdAt`     | string | Yes      | RFC 3339 timestamp for when the tag was first created.                  |
| `updatedAt`     | string | Yes      | RFC 3339 timestamp for the latest tag reference update.                 |
| `manifestSize`  | number |          | Size of the manifest file in bytes. Used for efficient manifest loading. |
| `metadata`      | object | Yes      | String key/value metadata map. If absent, it is treated as an empty object. |

## History Retention

Cleanup can run while another writer publishes a reference to an old version: creating or updating
a tag, creating a branch, or restoring. So that such a reference never points at a version cleanup
is deleting, cleanup first records the versions it **expires**. Readers may still check out an
expired version until it is deleted, but no new tag, branch, or restore may reference it.

When cleanup and the publishing writer both follow this section, a reference published while
cleanup runs either keeps its version or is rejected.

### Retention Files

Each branch records its rule in the `_retention/` directory of its own root (see
[Branch Dataset Layout](#branch-dataset-layout)), in files named `{n}.json` where `n` is a positive
integer. The file with the largest `n` holds the current rule:

| JSON Key        | Type             | Optional | Description                                                            |
|-----------------|------------------|----------|------------------------------------------------------------------------|
| `expiredBefore` | number           |          | A version number `F`.                                                  |
| `retained`      | array of numbers | Yes      | Ascending versions below `F` that are not expired. Absent means empty. |

A version `v` is **expired** when `v < F` and `v` is not in `retained`. A missing directory or a
current file that cannot be parsed expires nothing.

To write a rule, read the current file `{n}.json` and create `{n+1}.json`, or `1.json` if there is
none, with [put-if-not-exists](transaction.md#storage-primitives). If that file already exists,
read it and retry.

Expiration is never lifted: a new rule is always merged with the current one, so it expires every
version that either expires. `F` is the larger of the two, and `retained` lists the versions below it
that neither rule expires. Implementations may offer an override that writes any rule for repairs;
it is safe only while no cleanup is running.

### Pins

A restore writes a pin to protect its target version. Pins are stored at `_refs/pins/{uuid}.json` in
the dataset root:

| JSON Key    | Type   | Optional | Description                                                              |
|-------------|--------|----------|--------------------------------------------------------------------------|
| `branch`    | string | Yes      | Branch name being pinned. `null` or absent indicates the default branch. |
| `version`   | number |          | Version number being pinned within that branch.                          |
| `createdAt` | string |          | RFC 3339 timestamp for when the pin was created.                         |

Cleanup treats a pinned version like a tagged one, but ignores, and may delete, pins created more
than 24 hours before the cleanup began.

### Cleanup

Only one cleanup may run on a dataset at a time. Before deleting any version, cleanup must:

1. Plan which versions it will delete.
2. Write a rule with `F` above every version it will delete, retaining the versions below `F` that
   it keeps. The write may be skipped if the merged rule equals the current one.
3. List tags, branches, and pins, and keep every version they reference.
4. Delete, then delete every retention file except the current one.

### Publishing References

To create or update a tag, or create a branch, at version `v` of branch `b`, a writer must write the
reference, then read the current rule of `b`. If `v` is expired, it must remove the reference and
fail.

To restore version `v` of branch `b`, a writer must:

1. Write a pin for `v`.
2. Read the current rule of `b`. If `v` is expired, delete the pin and fail.
3. Commit within 24 hours of writing the pin, and leave the pin in place.

The pin outlives the commit because a cleanup that began earlier cannot see the restored version. A
cleanup that ignores the pin began after the commit.
