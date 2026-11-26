# Lightweight snapshot references

A `SnapshotRef` identifies an exact Lance table snapshot without transporting its
manifest. Its encoded size depends on identity strings, not the number of
fragments or the size of the manifest. It is an execution transport object, not
table metadata or a persisted retention token.

```text
SnapshotRef
+-- protocol version
+-- table incarnation and catalog/storage authority
+-- absolute table root URI
+-- resolved branch and numeric version
+-- exact manifest object path
+-- validation
    +-- authoritative immutable commit-record identity, or
    +-- conditional-read ETag, or
    +-- BLAKE3 digest of the complete original manifest file
```

The transport type and `SnapshotVerification` live in `lance_table::snapshot`.
`lance_table_runtime::Table` implements capture and validation without depending
on the query engine. `lance::dataset::snapshot` re-exports the types and provides
the same operations on `Dataset`.

## Capture and reopen

Resolve tags and branch heads before capture. The reference records the exact
branch and version, so advancing a head or moving a tag does not change what the
worker reads.

```rust
use lance::{Dataset, Result};
use lance::dataset::{builder::DatasetBuilder, snapshot::SnapshotRef};

async fn read_on_worker(dataset: &Dataset) -> Result<Dataset> {
    let snapshot = dataset.snapshot_ref("catalog/table-incarnation").await?;
    let transferred = SnapshotRef::decode(&snapshot.encode())?;
    DatasetBuilder::from_snapshot(transferred)
        // Resolve the worker's own credentials, endpoint, session, and base bindings.
        .load()
        .await
}
```

`DatasetBuilder::from_snapshot` resolves the exact branch/version through its
commit handler, validates the manifest path and verification identity, and never
falls back to latest. A missing manifest or identity mismatch is an error.
Existing builders can use `with_snapshot_ref`; combining it with a version,
branch, tag, or serialized manifest is an error.

The standalone runtime offers `Table::snapshot_ref` and
`TableBuilder::load_snapshot_with_session` with the same guarantees.

Workers still load the complete manifest metadata required by today's reader.
A small transport object does not provide fragment-local read state or eliminate
the reader's metadata initialization cost.

## Verification policy

`snapshot_ref` uses `SnapshotVerification::Identity`. It prefers an authoritative
commit-record identity when supplied by the commit handler. That identity relies
on immutable committed manifests. Otherwise it captures the observed ETag.
Reopening an ETag reference applies `If-Match` to every manifest range read and
retry; checking with `HEAD` followed by unconditional reads is insufficient.
An ETag is an opaque object-generation condition, not a content checksum.

Identity capture from a normally opened handle does not fetch the manifest again.
A separately serialized manifest or arbitrary manifest view must first be checked
against the original file; a token resolved later is not proof of its generation.
Such handles use complete-file verification before issuing a reference.
A cold worker can read the
manifest protobuf and necessary metadata sections without fetching otherwise
unused leading sections solely for a digest. If neither a commit identity nor
an ETag is available, capture falls back to full-file BLAKE3. A decoded reference
without any validation identity is rejected.

For explicit byte-for-byte verification:

```rust
use lance::{Dataset, Result};
use lance::dataset::snapshot::SnapshotVerification;

async fn capture_verified(dataset: &Dataset) -> Result<()> {
    let snapshot = dataset.snapshot_ref_with_verification(
        "catalog/table-incarnation",
        SnapshotVerification::FullManifestDigest,
    ).await?;
    assert_eq!(snapshot.manifest_digest().len(), 32);
    Ok(())
}
```

This hashes every original file byte, including index metadata, inline
transactions, dictionaries, length prefixes, and the footer. Hashing a
re-encoded protobuf is unsuitable because map ordering and hydrated dictionaries
can change its bytes. The first full-file capture may read the manifest again;
clones share the computed digest. A cold worker also consumes the complete file
to verify it. Ordinary opens keep their existing metadata read behavior.

`manifest_digest()` returns an empty slice for lightweight references.
`manifest_identity()` and `manifest_etag()` identify the selected validation
mechanism. `snapshot_id()` fingerprints the validated encoded reference, including
its verification fields; references with different verification mechanisms can
have different fingerprints for the same logical snapshot.

## Relationship to TableIdentifier

`protos/table_identifier.proto` belongs to DataFusion reconstruction. Its
lightweight mode already carries a URI, version, and optional manifest ETag;
its other mode carries a serialized manifest, and it can include storage options.
`SnapshotRef` is reusable outside DataFusion. It explicitly carries the resolved
branch, table incarnation/authority, manifest path, and verification identity,
while keeping credentials and worker storage bindings separate. It does not
change the existing `TableIdentifier` wire fields or reconstruction behavior.

## Authority and lifetime

The application supplies `table_identity`. Its trusted resolver must distinguish
table incarnations recreated at the same URI, validate the catalog/storage
authority, and provide credentials and per-base routing. Neither an ETag nor a
digest authenticates that caller-defined identity.

Capturing, encoding, decoding, or holding a reference does not protect files
from cleanup. Identity capture can describe an already-loaded snapshot whose
files have since been removed; reopening then fails. Applications retain the
snapshot for workers, deferred consumers, and retries. External objects require
their own immutability and lifetime guarantees.
