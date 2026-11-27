# Snapshot reference protocol

This proposal defines an unreleased execution wire contract for an exact Lance
snapshot. It includes generated messages and transport types. Runtime capture,
reopening, manifest I/O, caching, and their behavioral tests belong to a separate
implementation PR. This contract does not add persisted table metadata or retention.

## Identity and representation

`protos/snapshot.proto` defines `lance.snapshot.SnapshotRef`, protocol version 1.
Its transport size depends on identity strings, not the manifest size.

| Field | Contract |
| --- | --- |
| `table_identity` | Nonempty caller-defined table incarnation and catalog/storage authority. It must distinguish recreation at the same URI. |
| `table_uri` | Canonical absolute table root URI, without credentials or a branch suffix. |
| `branch` | Absent means main. A present branch is a resolved name, not a tag or unresolved head. It must pass the runtime branch-name rules, including rejection of a literal `.` path segment such as `a/./b`. |
| `version` | Exact committed numeric version, including a detached version when supported. |
| `manifest_path` | Exact resolved manifest object path in the source object store. |
| `manifest_identity` | Optional nonempty authoritative immutable commit-record identity. |
| `manifest_etag` | Optional nonempty opaque object-generation condition. |
| `manifest_digest` | Empty, or a 32-byte BLAKE3 digest of the complete original manifest file. |

At least one verification mechanism must be present. Unknown protocol versions,
empty required identities, malformed paths, and invalid digest lengths are errors.
The Rust `lance_table::snapshot::SnapshotRef` type validates and encodes these
fields. No engine runtime is required to construct or transport the type.

## Verification guarantees

The lightweight policy prefers authoritative immutable commit identity, then an
ETag. When neither exists, it falls back to full-file BLAKE3. Callers may request
complete-file verification explicitly. These mechanisms provide different guarantees.

An authoritative commit identity relies on immutable committed manifests and must
be checked against the exact resolved record. An ETag need not be a checksum;
reopening must apply `If-Match` to every manifest read, including deferred index,
inline-transaction and dictionary reads, range requests, and size/read retries.
The producer must establish that its manifest and token came from the same
generation. Resolving a new token for a separately serialized manifest is not proof.

Full-file verification hashes original bytes, including index metadata, inline
transactions, dictionaries, length prefixes, and the footer. Re-encoding the
protobuf is insufficient because map order and hydrated dictionaries can differ.
A cold verifier must consume the complete file. Lightweight workers still load
the metadata required by the current reader.

`snapshot_id()` is BLAKE3 of the validated encoded reference, including its selected
verification fields. Different verification mechanisms can therefore yield
different fingerprints for the same logical snapshot. The fingerprint is an
association check, not authentication of the caller-defined table identity.

### Exact snapshot fingerprint bytes

`snapshot_id()` hashes the current codec's canonical encoding of the validated
known `SnapshotRef` fields. It does not hash the incoming transport bytes or the
manifest file itself. The BLAKE3 hash is unkeyed, has no additional prefix or
framing, and returns 32 bytes. Its hexadecimal display below uses lowercase.

The encoding has these rules:

1. Emit known fields in ascending field-number order, from 1 through 9.
2. Use protobuf wire type 0 for fields 1 (`protocol_version`, uint32) and
   5 (`version`, uint64). Encode tags and integers as minimal unsigned varints.
3. Use wire type 2 for all other fields. Strings are UTF-8; bytes are unchanged.
   Prepend each value with its minimal-varint byte length.
4. Omit implicit proto3 defaults: zero integer values and empty non-optional
   strings/bytes. Validation already rejects zero versions and empty required
   strings. An empty `manifest_digest` therefore has no encoded field 7.
5. Preserve optional presence for fields 4 (`branch`), 8 (`manifest_identity`),
   and 9 (`manifest_etag`). Omit `None`; encode a present value. Validation rejects
   empty present values. Main is represented by absent `branch`, not `"main"`.
6. Use the validated field values. Decode rejects a noncanonical table root URI;
   it does not silently rewrite it. No extra case or Unicode normalization is
   applied to identity, branch, path, or verification strings.
7. Discard unknown protobuf fields before this encoding. Incoming field order,
   unknown fields, and non-minimal incoming varint spellings do not become part
   of the fingerprint. No message in this encoding contains a protobuf map.

The following fixed fixtures share protocol version 1, table identity
`catalog/incarnation`, root URI `s3://bucket/table`, snapshot version 7, and
manifest path `table/_versions/7.manifest`:

| Fixture | `branch` | `manifest_digest` | `manifest_identity` | `manifest_etag` |
| --- | --- | --- | --- | --- |
| `full_main` | Absent | 32 bytes of `2a` | Absent | Absent |
| `etag_branch` | `feature/test` | Empty/omitted | Absent | `generation-1` |
| `commit_main` | Absent | Empty/omitted | `record-1` | Absent |

The repeated `2a` digest is an illustrative field value, not a claim about an
existing manifest object. Each block gives the exact encoded bytes followed by
its expected `snapshot_id()`:

**`full_main` (106 bytes)**

```text
08011213636174616c6f672f696e6361726e6174696f6e1a1173333a2f2f6275636b65742f7461626c652807321a7461626c652f5f76657273696f6e732f372e6d616e69666573743a202a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a
```

```text
103d7010e5016fc37a4e73761a3f186d13dca77aa21b86f0bb5d2555760a411c
```

**`etag_branch` (100 bytes)**

```text
08011213636174616c6f672f696e6361726e6174696f6e1a1173333a2f2f6275636b65742f7461626c65220c666561747572652f746573742807321a7461626c652f5f76657273696f6e732f372e6d616e69666573744a0c67656e65726174696f6e2d31
```

```text
597c8e5b98ae26de7edcca9e5a2357bb327e0b46e4067c635f4e0c6446d6c7ac
```

**`commit_main` (82 bytes)**

```text
08011213636174616c6f672f696e6361726e6174696f6e1a1173333a2f2f6275636b65742f7461626c652807321a7461626c652f5f76657273696f6e732f372e6d616e696665737442087265636f72642d31
```

```text
38c6c30fe651071d470d210a0cafce9461e6f6543c9b87d499542d41e7fc2b11
```

The fixed-fixture tests also reorder field 1 and add unknown field 31. The
canonical encoding and identity remain unchanged. These fixtures document and
test the current encoding; the maintainer decision on standardizing this
execution representation remains open.

## Relationship to existing contracts

DataFusion's `TableIdentifier` already supports URI + version + optional ETag,
or a serialized manifest, and may include storage options. `SnapshotRef` is
engine-independent and explicitly carries branch, incarnation/authority, exact
manifest path, and verification identity. Credentials and worker bindings remain
separate. This proposal does not change `table_identifier.proto`.

Version and path alone rely on immutable commits. A transaction UUID is not
universally available and does not checksum the resulting manifest. A suitable
commit identity or conditional ETag avoids mandatory full-file verification reads.

The trusted resolver must authenticate table incarnation, storage authority,
credentials, and base routing. Capturing or holding a reference does not protect
files from cleanup. Applications retain snapshots through every deferred consumer
and retry. Persisted retention tokens use their own contract, not this transport.

The protocol is unreleased. Producers and consumers must use matching codecs.
The new schema remains under the existing format-change review and vote process.
