# Lance table runtime

Open Lance tables, resolve snapshots and refs, and inspect schemas, fragment
metadata, version history, and transactions. This crate works with local
storage by default. Enable the Cargo feature for the cloud backend you use.

```rust,no_run
use lance_table_runtime::TableBuilder;

async fn inspect(uri: &str) -> lance_core::Result<()> {
    let table = TableBuilder::from_uri(uri).load().await?;
    println!("Version {} has {} fragments", table.version_id(), table.count_fragments());
    Ok(())
}
```
