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

Cleanup can run at the same time as operations that publish a reference to an old version: tag
creation and update, branch creation, and restore. To stop a reference from being published to a
version that cleanup is deleting, cleanup records which versions it **expires** in a retention file
before it deletes anything.

Expiration affects only new references. Readers may still check out an expired version until cleanup
deletes it, but tags, branches, and restores must not newly reference it.

### Retention Files

Each branch keeps its retention files in a `_retention/` directory under its own root: the dataset
root for the default branch, and `tree/{branch_name}/` for a named branch. Files are named
`{n}.json`, where `n` is a positive decimal integer. The file with the largest `n` holds the
branch's current rule:

| JSON Key        | Type             | Description                                                                 |
|-----------------|------------------|-----------------------------------------------------------------------------|
| `expiredBefore` | number           | A version number `F`.                                                       |
| `retained`      | array of numbers | Ascending version numbers below `F` that are not expired. May be empty.     |

A version `v` of a branch is **expired** when the branch's current rule has `v < F` and `v` is not
in `retained`. Retained versions are typically tagged versions, branch points, and versions kept by
thinning retention rules.

A branch continues its parent's numbering (a branch created from version 50 starts at version 51),
and a new branch has no retention files.

To write a rule, a writer reads the current file `{n}.json` and creates `{n+1}.json` with
put-if-not-exists, the same operation that commits manifests. If `{n+1}.json` already exists, the
writer reads it and tries again. A writer that finds no retention files creates `1.json`.

Expiration is monotonic. A new rule must not lift the expiration of any version that its
predecessor expires, so a writer always merges its rule `(F_own, R_own)` with the current rule
`(F', R')`. The merged rule has `F = max(F_own, F')`, and lists as retained exactly the versions below
`F` that neither rule expires: those with `v >= F_own` or `v` in `R_own`, and also `v >= F'` or
`v` in `R'`. Implementations may offer an explicit override that writes any rule. Lifting expiration
is safe only when no cleanup that read an earlier rule is still deleting.

The protection is best-effort. Writers that do not implement this section ignore retention files
and pins. A missing `_retention/` directory or a current file that cannot be parsed must be treated
as expiring nothing. In those cases references and cleanup race as they would without this section.

### Pins

A pin is a short-lived reference that a restore holds while it commits. Pins are JSON files stored at
`_refs/pins/{uuid}.json` in the dataset root:

| JSON Key    | Type   | Optional | Description                                                              |
|-------------|--------|----------|--------------------------------------------------------------------------|
| `branch`    | string | Yes      | Branch name being pinned. `null` or absent indicates the default branch. |
| `version`   | number |          | Version number being pinned within that branch.                          |
| `createdAt` | string |          | RFC 3339 timestamp for when the pin was created.                         |

Cleanup treats a pinned version like a tagged one. It may delete pins created more than 24 hours
ago, which are left behind by restores that did not finish.

### Cleanup

Before deleting any manifest, cleanup must:

1. Determine the versions it will delete and the versions below `F` it will keep.
2. Write a rule with `F` greater than every version it will delete, merged with the current rule as
   described above. If the merged rule equals the current rule, the write is skipped.
3. After the write succeeds, list tags, branches, and pins and keep every version they reference.
4. Delete.
5. Delete retention files other than the one with the largest `n`.

A version kept in step 3 stays expired. Written rules remain valid if cleanup stops before deleting,
and a later cleanup continues from them.

### Publishing References

Tag creation, tag update, and branch creation that target version `v` of branch `b` must write the
reference first and then read the current rule of `b`. If `v` is expired there, the writer must remove
the reference it wrote and fail.

Restore of version `v` of branch `b` must:

1. Write a pin for `v`.
2. Read the current rule of `b`. If `v` is expired there, delete the pin and fail.
3. Commit the restore.
4. Delete the pin.

Implementations may offer an explicit option that skips these checks.

### Guarantees

When cleanup and the publishing writer both implement this section, a reference published while
cleanup runs either keeps its version from deletion or is rejected. There is no guarantee when
either side does not implement it, or when the current rule is missing or malformed.
