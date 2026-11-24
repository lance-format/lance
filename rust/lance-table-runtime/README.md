# Lance table runtime

Open Lance table snapshots and read fragments with projections, row IDs,
deletions, and overlays. The runtime also resolves refs, version history, and
transactions. It works with local storage by default. Enable the Cargo feature
for the cloud backend you use.

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
