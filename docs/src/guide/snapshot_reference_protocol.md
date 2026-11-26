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
| `branch` | Absent means main. A present branch is a resolved name, not a tag or unresolved head. |
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
