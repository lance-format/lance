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
| `identifier`     | object | Yes      | Branch identity. Older files may omit it; see [Branch identity](#branch-identity). |

### Branch Identity

`identifier.version_mapping` is an ordered JSON array of pairs. Each pair
contains an unsigned 64-bit parent version and a UUID string. A new branch
copies its parent's mapping and appends its own pair: the parent version it
branched from and a fresh UUID. Newly generated UUIDs use 32 lowercase
hexadecimal digits without hyphens. Main's mapping is empty.

Compare the complete ordered mapping by value. Preserve an existing identifier
when updating branch metadata. Lease requests retain the identifier read during
acquisition and reject a different identifier on later inspection. This
documents the existing branch representation; leases add no identity field.

For an older file without `identifier`, derive a stable identifier from its
branch name, `parentBranch`, `parentVersion`, and `createAt`. Apply the same
fallback to the placeholder mapping containing one pair with version zero and
the string `00000000000000000000000000000000`. The existing derivation is:

1. Encode four lines as UTF-8, separated by a single LF byte, with no trailing
   newline: `branch_name=<name>`, `parent_branch=<parent>`,
   `parent_version=<version>`, and `create_at=<time>`. Use the unescaped branch
   names, an empty parent for absent/null `parentBranch`, and unsigned decimal
   integers without leading zeros.
2. Hash those bytes twice. Start the first hash at `0xcbf29ce484222325`; start
   the second at that value XOR `0x9e3779b97f4a7c15`. For each byte, XOR it into
   the hash, then multiply by `0x100000001b3`, modulo `2^64`.
3. Concatenate the two 8-byte hash values in big-endian order and encode the
   resulting 16 bytes as 32 lowercase hexadecimal digits. The derived mapping
   contains one pair: version zero and that string.

The fallback is stable across reads, not a unique record of each creation.
Recreating an older branch with identical inputs produces the same identifier.
Applications must coordinate such recreation and must not reuse old lease
requests or handles across it. Main's empty mapping likewise does not identify
a particular creation of the whole table. No branch names are retired and no
stored metadata needs migration.

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

### Deletion with retention leases

Normal and force deletion **must resolve the
[authoritative head](versioning.md#current-state-for-lease-decisions)**, validate
its writer capabilities and lease list, and reject any listed leases, identifying
the blocking IDs. Expired records still block
deletion until a commit removes them. On an idle table, use explicit ReleaseLease;
deletion does not expire leases itself. Existing tag and dependent-branch checks
also apply.

For an orphan with no branch reference, skip the lease check only when authoritative
resolution proves there is no attached manifest. An unreadable or invalid head,
or a resolution failure, **must fail deletion**. If the selected head disappears,
resolve again rather than infer absence. Corrupt metadata requires administrator
repair; force deletion cannot bypass the check.

Inspection is separate from physical deletion. A later acquisition cannot stop
a deletion already approved, and delayed deletion after recovery and name reuse
may affect a replacement branch. See [cleanup and deletion races](retention_leases.md#cleanup-and-deletion-races).

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
