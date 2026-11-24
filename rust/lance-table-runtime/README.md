# Lance table runtime

Open Lance table snapshots and read fragments with projections, row IDs,
deletions, and overlays. Point takes select rows by offset, stable ID, or
physical address. The runtime can return lazy blob handles or plan payload
reads. Stream Arrow batches into `InsertBuilder` to create, append to, or
overwrite a table. Publish prewritten fragments and staged deletions or
overlays with `CommitBuilder`. Enable the Cargo feature for your cloud backend.

```rust,no_run
use lance_table_runtime::{FragReadConfig, Table};

async fn read_first_rows(uri: &str) -> Result<(), Box<dyn std::error::Error>> {
    let table = Table::open(uri).await?;
    let config = FragReadConfig::default().with_row_id(true);
    for fragment in table.get_fragments() {
        let reader = fragment.open(table.schema(), config.clone()).await?;
        let end = fragment.physical_rows().await?.min(10);
        let batch = reader.read_range_as_batch(0..end).await?;
        println!("Fragment {} returned {} rows", fragment.id(), batch.num_rows());
    }
    Ok(())
}
```

```rust,no_run
use lance_table_runtime::Table;

async fn take_rows_and_blobs(uri: &str) -> lance_core::Result<()> {
    let table = Table::open(uri).await?;
    let projection = table.schema().project_preserve_system_columns(&["id", "_rowid"])?;
    let batch = table.take(&[4, 0, 4], &projection).await?;
    assert_eq!(batch.num_rows(), 3);

    let handles = table.take_blobs_by_indices(&[0, 4], "images").await?;
    for handle in handles.into_iter().flatten() {
        let bytes = handle.read_range(0..handle.size().min(16)).await?;
        println!("Read {} bytes", bytes.len());
    }
    Ok(())
}
```

```rust,no_run
use std::sync::Arc;

use lance_table::format::Fragment;
use lance_table::transaction::{Operation, Transaction};
use lance_table_runtime::{CommitBuilder, Table};

async fn append(table: Table, fragments: Vec<Fragment>) -> lance_core::Result<Table> {
    let transaction = Transaction::new(table.version_id(), Operation::Append { fragments }, None);
    CommitBuilder::new(Arc::new(table)).execute(transaction).await
}
```

```rust,no_run
use lance_table_runtime::{InsertBuilder, Table, WriteMode, WriteParams};

async fn append_batches(
    table: &Table,
    data: impl arrow_array::RecordBatchReader + Send + 'static,
) -> lance_core::Result<Table> {
    let params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: 100_000,
        ..Default::default()
    };
    InsertBuilder::new(table)
        .with_params(&params)
        .execute_stream(data)
        .await
}
```
