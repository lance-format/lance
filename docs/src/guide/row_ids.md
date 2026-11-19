---
title: Stable Row IDs and Lineage
description: Preserve logical row identity across updates and compaction, and read row creation and update versions.
---

# Stable Row IDs and Lineage

Lance exposes two different row locators:

- **Row address** (`_rowaddr`): the row's physical location, encoded from its fragment ID and offset. It can change after updates or compaction.
- **Stable row ID** (`_rowid`): a logical identifier assigned for the row's lifetime when stable row IDs are enabled.

Do not use `_rowaddr` as a durable external key. Enable stable row IDs when downstream systems need to retain row identity across physical rewrites.

The workflows below cover the Python (`pylance`), Rust (`lance`), and Java (`org.lance:lance-core`) SDKs.

## Enable Stable Row IDs for a New Dataset

Enable stable row IDs when creating the dataset:

=== "Python"

    ```python
    import lance
    import pyarrow as pa

    uri = "stable_rows.lance"
    table = pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    dataset = lance.write_dataset(table, uri, enable_stable_row_ids=True)
    assert dataset.has_stable_row_ids
    ```

=== "Rust"

    ```rust
    use lance::dataset::{WriteMode, WriteParams};
    use lance::Dataset;

    let write_params = WriteParams {
        mode: WriteMode::Overwrite,
        enable_stable_row_ids: true,
        ..Default::default()
    };
    let dataset = Dataset::write(reader, "stable_rows.lance", Some(write_params)).await?;
    assert!(dataset.manifest.uses_stable_row_ids());
    ```

=== "Java"

    ```java
    import org.lance.Dataset;
    import org.lance.WriteParams;

    try (Dataset dataset =
        Dataset.write()
            .allocator(allocator)
            .reader(reader)
            .uri("stable_rows.lance")
            .mode(WriteParams.WriteMode.OVERWRITE)
            .enableStableRowIds(true)
            .execute()) {
      assert dataset.hasStableRowIds();
    }
    ```

Passing this option on later writes does not migrate an existing dataset. Migration is a separate operation, currently exposed in Rust as `Dataset::migrate_to_stable_row_ids`. Before migration, quiesce data writers and index builds, drop secondary indexes, and recreate them after migration. See the [Row ID and Lineage Specification](../format/table/row_id_lineage.md#enabling-stable-row-ids) for the complete migration protocol.

## Read Stable Row IDs

Request `_rowid` explicitly on the scan:

=== "Python"

    ```python
    result = dataset.to_table(columns=["id", "value"], with_row_id=True)
    print(result)
    ```

=== "Rust"

    ```rust
    use arrow_array::RecordBatch;
    use futures::TryStreamExt;

    let batches: Vec<RecordBatch> = dataset
        .scan()
        .project(&["id", "value"])?
        .with_row_id()
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    import org.apache.arrow.vector.ipc.ArrowReader;
    import org.lance.ipc.LanceScanner;
    import org.lance.ipc.ScanOptions;
    import java.util.Arrays;

    ScanOptions options =
        new ScanOptions.Builder()
            .columns(Arrays.asList("id", "value"))
            .withRowId(true)
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // reader.getVectorSchemaRoot() includes "_rowid"
      }
    }
    ```

Stable row IDs survive row updates and compaction:

```python
before = {
    row["id"]: row["_rowid"]
    for row in dataset.to_table(with_row_id=True).to_pylist()
}

dataset.update({"value": "'updated'"}, where="id = 2")

after = {
    row["id"]: row["_rowid"]
    for row in dataset.to_table(with_row_id=True).to_pylist()
}
assert before[2] == after[2]
```

When stable row IDs are disabled, `_rowid` is not a persistent identifier and should not be stored for later reconciliation.

## Read Row Lineage

Stable row IDs also enable two version columns:

- `_row_created_at_version`: dataset version in which the logical row was created.
- `_row_last_updated_at_version`: dataset version of its latest update.

Request them as ordinary projected columns:

=== "Python"

    ```python
    lineage = dataset.to_table(
        columns=["id", "_rowid", "_row_created_at_version", "_row_last_updated_at_version"]
    )
    ```

=== "Rust"

    ```rust
    let lineage: Vec<RecordBatch> = dataset
        .scan()
        .project(&[
            "id",
            "_rowid",
            "_row_created_at_version",
            "_row_last_updated_at_version",
        ])?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    ScanOptions lineageOptions =
        new ScanOptions.Builder()
            .columns(
                Arrays.asList(
                    "id",
                    "_rowid",
                    "_row_created_at_version",
                    "_row_last_updated_at_version"))
            .build();
    try (LanceScanner scanner = dataset.newScan(lineageOptions);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process lineage columns
      }
    }
    ```

An update preserves `_rowid` and `_row_created_at_version`, while advancing `_row_last_updated_at_version`. Appending rows creates new IDs and sets both version columns to the append version. These values refer to Lance dataset versions, not wall-clock timestamps.

## Relationship to Indexes and Compaction

Secondary indexes currently reference physical row addresses. Stable row IDs preserve logical identity for applications, but they do not make index maintenance unnecessary. Compaction still needs to remap or rebuild affected indexes.

Use the normal maintenance sequence: compact first, then optimize or rebuild indexes as needed. See [Indexing and Search](indexing.md#index-lifecycle) and [Table Maintenance](read_and_write.md#table-maintenance).

## Distributed Writes

When manually assembling distributed update transactions, preserve stable row IDs by reading existing rows with `with_row_id=True` and attaching their IDs to rewritten fragments with `lance.fragment.RowIdSequence`. Otherwise, rewritten rows can silently receive new identities.

See [Distributed Write: Handling Stable Row IDs](distributed_write.md#handling-stable-row-id) for the complete worker/commit workflow.
