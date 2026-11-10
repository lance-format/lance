# Lightweight snapshot references

A `SnapshotRef` identifies an exact Lance table snapshot without carrying its
manifest. Use it to send a resolved snapshot to another process, including for
ordinary reads, distributed work, or [portable blob references](blob_references.md).
Its encoded size depends on identity strings, not the number of fragments or the
size of the manifest.

```text
SnapshotRef
+-- protocol version
+-- table incarnation and catalog/storage authority
+-- absolute table root URI
+-- resolved branch and numeric version
+-- exact manifest object path
+-- BLAKE3 digest of the complete original manifest file
+-- optional authoritative commit-store record identity
```

The transport type lives in `lance_table::snapshot`, independently of `Dataset`
and blob APIs. `lance::dataset::snapshot` re-exports it and adds capture and
validation on `Dataset`.

## Capture and reopen

Resolve a tag or branch head before capturing the reference. The reference records
the resulting branch and version, so moving a tag or advancing the head does not
change what a worker reads.

```rust
use lance::{Dataset, Result};
use lance::dataset::{builder::DatasetBuilder, snapshot::SnapshotRef};

async fn read_on_worker(dataset: &Dataset) -> Result<Dataset> {
    let snapshot = dataset.snapshot_ref("catalog/table-incarnation").await?;
    let wire = snapshot.encode();

    let transferred = SnapshotRef::decode(&wire)?;
    DatasetBuilder::from_snapshot(transferred)
        // Supply the worker's own storage settings, session, and per-base bindings.
        .load()
        .await
}
```

`DatasetBuilder::from_snapshot` resolves the exact branch/version through its
commit handler and validates the manifest path, digest, and optional commit-store
identity. It never falls back to the latest snapshot. A missing manifest or an
identity mismatch is an error. Existing builders can use `with_snapshot_ref`;
combining this with a version, branch, tag, or serialized manifest is an error.

Workers still load the full manifest through today's `Dataset` read path. The
reference reduces transport size; it does not supply a partial table read state
or replace shard planning.

## Identity and storage authority

The application supplies `table_identity`. It must distinguish tables recreated
at the same URI and different catalog/storage authorities. The worker's trusted
resolver must verify that authority and supply the correct credentials, commit
handler, storage endpoint, and per-base routing. The reference contains no
credentials, native handles, or access rights.

The manifest digest hashes the complete original manifest file, including index
metadata, inline transactions, dictionaries, length prefixes, and the footer.
It is computed over chunks without joining them into another manifest-sized
buffer. Re-encoding a decoded manifest is unsuitable:
protobuf maps and hydrated dictionary values can change its bytes. Datasets opened
from storage share this digest with their cached manifest. Cache reuse is scoped to
the object store, resolved manifest path, commit identity, and generation token.
Writer-populated caches keep the digest lazy, so ordinary opens do not fetch the
manifest again. Capturing a snapshot reference or reopening an exact reference
verifies that digest before returning. Datasets returned by writes and loaded from
serialized manifests may also read the manifest once on first capture; clones and
later captures share the result.

`SnapshotRef::snapshot_id` fingerprints the validated reference. It is independent
of blob capabilities and worker access keys. A storage ETag is an opaque object
generation token and is not a substitute for the content digest or table identity.

## Lifetime

Capturing, encoding, decoding, or holding a reference does not protect files from
cleanup. An already captured identity can remain available in memory after its
source files are removed; reopening it then fails. Applications must retain the
snapshot for all participating workers, deferred consumers, and retries.

External blob contents need their own immutability and lifetime guarantee. A table
snapshot records an external location, but cannot freeze the object at that location.
