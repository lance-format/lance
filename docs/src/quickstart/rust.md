---
title: Rust Quickstart
description: Get started with the Lance Rust SDK — write Arrow batches, filter with SQL, run vector search, and time travel across versions
---

# Getting Started with Rust

The `lance` crate is the native Rust core and SDK for the Lance format. Use it when building Rust services, storage engines, or data pipelines that read and write Lance datasets directly with Apache Arrow. If you want a higher-level retrieval database API in Rust, see [LanceDB](lancedb.md) instead.

## Add Dependencies

Create a new Cargo binary project and add `lance` along with matching Apache Arrow crates to `Cargo.toml`:

```toml
--8<-- "snippets/quickstart/rust-cargo.toml"
```

!!! note "Matching Arrow Versions"
    `Dataset::write` and `Scanner` exchange `arrow_array::RecordBatch` values directly, so your `arrow-*` major version must match the Arrow version used by your `lance` release (for `lance = "14.0.2"`, use `arrow-* = "58.0"`).

## Write, Query, and Version a Dataset

Put the following example in `src/main.rs`. It creates a Lance dataset with scalar and vector columns, runs a SQL filter scan, executes a nearest-neighbor vector search, appends new data, and checks out version 1:

```rust
--8<-- "snippets/quickstart/rust_quickstart.rs"
```

Run the program with Cargo:

```bash
cargo run
```

```output
--8<-- "snippets/quickstart/outputs/rust_quickstart.rs.run.txt"
```

## Next Steps

- Browse the [Rust API Reference on docs.rs ↗](https://docs.rs/lance/latest/lance) for full crate documentation.
- See the [Rust HNSW Indexing Example](../examples/rust/hnsw.md) and [LLM Dataset Creation Example](../examples/rust/llm_dataset_creation.md).
- Read the [User Guide](../guide/read_and_write.md) for indexing, Blob v2 columns, object storage configuration, and compaction concepts.
