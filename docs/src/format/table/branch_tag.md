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

### Deletion with ongoing work pins

Deletion must resolve the [authoritative head](versioning.md#current-state-for-pin-decisions)
and reject if its active-pin list is nonempty, identifying the blocking pins.
A pin past `expires_at` still blocks deletion until removed by a committed
ReleasePin or ExpirePins. Deletion does not implicitly revoke pins. Force deletion
must perform the same check. Existing tag and dependent-branch checks also apply.

For force deletion of an orphan with no branch reference, the pin check may be
skipped only when authoritative resolution definitively establishes that no
attached manifest exists. A listing or resolution failure, or a manifest that
exists but cannot be read or validated, must fail deletion. If a selected head
disappears during inspection, restart resolution as specified in
[Format Versioning](versioning.md#current-state-for-pin-decisions); do not treat
that failed read alone as proof of absence.

A corrupt current manifest therefore requires repair or administrator recovery
outside the ordinary branch-deletion protocol. `force` does not bypass pin
protection; this proposal defines no operator override.

This follows existing tag protection: the check is separate from physical
deletion and is not atomic with concurrent acquisition. Registration after a
deleter's inspection does not force that deleter to observe the new pin.
Delayed deletion after recovery and name reuse also remains a known gap. See
[Maintenance concurrency limits](ongoing_work_pins.md#maintenance-concurrency-limits).

Resolving those existing coordination gaps is outside the pin proposal. Branch
names remain reusable; pins add no name-retirement record, deletion marker,
branch layout, or reference-history format.

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
