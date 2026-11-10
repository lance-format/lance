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
version that cleanup is deleting, cleanup records which versions it **expires** in two
[table config](table_config.md) keys before it deletes anything:

| Key                            | Value                                                                                  |
|--------------------------------|----------------------------------------------------------------------------------------|
| `lance.history.expired_before` | A decimal version number `F`.                                                          |
| `lance.history.retained`       | Ascending, comma-separated decimal version numbers below `F` that cleanup keeps. Missing or empty means none. |

A version `v` of a branch is **expired** when that branch's latest manifest has `v < F` and `v` is
not listed in `lance.history.retained`. Retained versions are typically tagged versions, branch
points, and versions kept by thinning retention rules.

Expiration affects only new references. Readers may still check out an expired version until cleanup
deletes it, but tags, branches, and restores must not newly reference it.

The keys are per branch and apply only to versions of that branch. A branch continues its parent's
numbering (a branch created from version 50 starts at version 51), so values copied from the parent
never match a version of the branch.

The keys are table config rather than manifest fields so that writers that do not recognize them
carry them forward unchanged. The protection is best-effort. A writer that does not implement this
section may change or drop the keys, and missing or malformed values must be treated as expiring
nothing. In those cases references and cleanup race as they would without these keys.

### Cleanup

Before deleting any manifest, cleanup must:

1. Determine the versions it will delete and the versions below `F` it will keep.
2. Commit an [UpdateConfig](transaction.md#updateconfig) that sets both keys, with `F` greater than
   every version it will delete. `F` must not be lower than the current value. If neither key would
   change, the commit is skipped.
3. After that commit succeeds, list tags and branches and keep every version they reference.
4. Delete.

A version kept in step 3 stays expired until a later cleanup lists it as retained. Committed
keys remain valid if cleanup stops before deleting, and a later cleanup continues from them.

If the commit conflicts with a concurrent commit that sets the keys, cleanup rebases onto the new
values `(F', R')`. It writes `F = max(F_own, F')`, and lists as retained exactly the versions below
`F` that neither rule expires: those with `v >= F_own` or `v` in `R_own`, and also `v >= F'` or
`v` in `R'`.

### Publishing References

Tag creation, tag update, and branch creation that target version `v` of branch `b` must write the
reference first and then read the latest manifest of `b`. If `v` is expired there, the writer
must remove the reference it wrote and fail. Implementations may offer an explicit option that skips
this check.

Restore of version `v` must fail if `v` is expired in the manifest the restore commits on top of.
This covers rules committed before the restore read the table; rules committed after it are caught
by the conflict rules in [Restore Compatibility](transaction.md#restore-compatibility).

### Writer Rules

- Only cleanup, including version expiration, sets these keys. Other UpdateConfig operations must not set or remove them.
- Restore must carry the keys of the manifest it rebases onto, not those of the restored version.
- Branch creation should not copy the keys into the new branch.

### Guarantees

When cleanup and the publishing writer both implement this section, a reference published while
cleanup runs either keeps its version from deletion or is rejected. There is no guarantee when
either side does not implement it, or when the keys were dropped or malformed.
