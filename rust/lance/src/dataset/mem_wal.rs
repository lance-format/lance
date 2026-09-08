// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL - Log-Structured Merge (LSM) tree for Lance tables
//!
//! This module implements an LSM tree architecture for high-performance
//! streaming writes with durability guarantees via Write-Ahead Log (WAL).
//!
//! ## Architecture
//!
//! Each shard has:
//! - A **MemTable** for in-memory data (immediately queryable)
//! - A **WAL Buffer** for durability (persisted to object storage)
//! - **In-memory indexes** (BTree, IVF-PQ, FTS) for indexed queries
//!
//! ## Write Path
//!
//! ```text
//! put(batch) → MemTable.insert() → WalBuffer.append() → [async flush to storage]
//!                   ↓
//!           IndexRegistry.update()
//! ```
//!
//! ## Durability
//!
//! Writers can be configured for:
//! - **Durable writes**: Wait for WAL flush before returning
//! - **Non-durable writes**: Buffer in memory, accept potential loss on crash
//!
//! ## Epoch-Based Fencing
//!
//! Each shard has exactly one active writer at any time, enforced via
//! monotonically increasing writer epochs in the shard manifest.

mod api;
mod hnsw;
pub mod index;
mod manifest;
pub mod memtable;
pub mod observer;
pub mod scanner;
pub mod sharding;
#[cfg(test)]
pub(crate) mod test_util;
pub mod util;
mod wal;
pub mod write;

use std::sync::Arc;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

/// Column name for the mem_wal tombstone (delete sentinel) marker.
///
/// `_tombstone` is a *physical* column present only in mem_wal memtables and
/// SSTables — it is deliberately kept out of the base table (hard
/// delete), so it is **not** a virtual [`is_system_column`](lance_core::is_system_column).
/// A row with `_tombstone = true` is a delete sentinel: the newest value for
/// its primary key, carrying null in every non-PK column, that wins
/// newest-per-PK resolution and is then silently dropped from query results.
///
/// The column is owned end-to-end by lance: callers pass the logical schema and
/// lance injects the column on the write path ([`write::ShardWriter::put`] /
/// [`write::ShardWriter::delete`]), so no caller ever constructs or names it.
pub const TOMBSTONE: &str = "_tombstone";

/// The mem_wal tombstone field appended to the logical schema on the way to the
/// storage schema.
///
/// Non-nullable: the write path always populates it (`false` for normal rows,
/// `true` for tombstones). Non-nullability also lets the point-lookup base arm
/// synthesize a matching `Literal(false)` column for the `CoalesceFirstExec`
/// exact-schema check.
pub fn tombstone_field() -> ArrowField {
    ArrowField::new(TOMBSTONE, DataType::Boolean, false)
}

/// Derive a shard's *storage* schema from its *logical* (base table) schema by
/// widening every top-level field to nullable except the primary key and
/// `_tombstone`.
///
/// A tombstone carries the primary key and null everywhere else, so storage
/// must permit a null wherever the base table does not. The logical schema
/// stays the caller's contract — validated at [`write::ShardWriter::put`],
/// restored at the scan's egress.
///
/// Top-level only: Arrow validates nullability only there, so a vector column's
/// item field is untouched. The primary key is excluded because
/// [`lance_core::datatypes::Schema`] requires it non-nullable, `_tombstone`
/// because the write path always populates it. Idempotent.
pub fn relax_non_pk_nullability(
    logical_schema: &ArrowSchema,
    pk_columns: &[String],
) -> Arc<ArrowSchema> {
    let fields: Vec<ArrowField> = logical_schema
        .fields()
        .iter()
        .map(|field| {
            let keep = field.is_nullable()
                || field.name() == TOMBSTONE
                || pk_columns.iter().any(|c| c == field.name());
            let field = field.as_ref().clone();
            if keep {
                field
            } else {
                field.with_nullable(true)
            }
        })
        .collect();
    Arc::new(ArrowSchema::new_with_metadata(
        fields,
        logical_schema.metadata().clone(),
    ))
}

/// Extend the logical schema with the trailing `_tombstone` column — the
/// intermediate [`relax_non_pk_nullability`] widens into the storage schema.
///
/// Idempotent: a schema that already carries `_tombstone` (a reopen/replay
/// path) is returned unchanged. Schema-level metadata and per-field metadata
/// (e.g. the `lance-schema:unenforced-primary-key` marker) are preserved.
pub fn schema_with_tombstone(base: &ArrowSchema) -> Arc<ArrowSchema> {
    if base.column_with_name(TOMBSTONE).is_some() {
        return Arc::new(base.clone());
    }
    let mut fields: Vec<ArrowField> = base.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(tombstone_field());
    Arc::new(ArrowSchema::new_with_metadata(
        fields,
        base.metadata().clone(),
    ))
}

pub use api::{DatasetMemWalExt, InitializeMemWalBuilder, validate_maintained_indexes};
pub use index::{MemIndexKind, MemTableVisibility};
pub use manifest::ShardManifestStore;
pub use memtable::scanner::MemTableScanner;
pub use scanner::{LsmDataSource, LsmGeneration, LsmScanner, ShardSnapshot};
pub use sharding::{
    evaluate_sharding_spec, evaluate_sharding_spec_with_embedded_columns,
    evaluate_sharding_spec_with_source_columns,
};
pub use wal::{BatchDurableWatcher, WalAppendResult, WalAppender, WalReadEntry, WalTailer};
pub use write::SealFence;
pub use write::ShardWriter;
pub use write::ShardWriterConfig;
pub use write::WriteResult;

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Fields;

    fn logical() -> ArrowSchema {
        ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("count", DataType::Int64, false),
            ArrowField::new("note", DataType::Utf8, true),
        ])
    }

    #[test]
    fn relax_widens_every_non_pk_field_and_leaves_the_key_alone() {
        let relaxed = relax_non_pk_nullability(&logical(), &["id".to_string()]);

        assert!(
            !relaxed.field(0).is_nullable(),
            "the primary key stays strict"
        );
        assert!(
            relaxed.field(1).is_nullable(),
            "`count` must accept a tombstone null"
        );
        assert!(
            relaxed.field(2).is_nullable(),
            "already-nullable is untouched"
        );
    }

    #[test]
    fn relax_leaves_nested_fields_exactly_as_declared() {
        // Arrow validates nullability only at the top level, and a vector
        // column's item field must not gain a validity layer.
        let item = Arc::new(ArrowField::new("item", DataType::Float32, false));
        let child = ArrowField::new("a", DataType::Int32, false);
        let schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("vector", DataType::FixedSizeList(item, 4), false),
            ArrowField::new("s", DataType::Struct(Fields::from(vec![child])), false),
        ]);

        let relaxed = relax_non_pk_nullability(&schema, &["id".to_string()]);

        assert!(relaxed.field(1).is_nullable());
        match relaxed.field(1).data_type() {
            DataType::FixedSizeList(f, _) => assert!(!f.is_nullable(), "item field untouched"),
            other => panic!("expected FixedSizeList, got {other:?}"),
        }
        match relaxed.field(2).data_type() {
            DataType::Struct(fields) => assert!(!fields[0].is_nullable(), "child field untouched"),
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    #[test]
    fn relax_keeps_tombstone_non_nullable_and_is_idempotent() {
        let pk = ["id".to_string()];
        let once = relax_non_pk_nullability(&schema_with_tombstone(&logical()), &pk);
        let twice = relax_non_pk_nullability(&once, &pk);

        let tombstone = once.field_with_name(TOMBSTONE).unwrap();
        assert!(
            !tombstone.is_nullable(),
            "the write path always populates _tombstone"
        );
        assert_eq!(once, twice);
    }

    #[test]
    fn relax_preserves_schema_and_field_metadata() {
        // The `lance-schema:unenforced-primary-key` marker rides on field
        // metadata, so losing it here would silently drop the shard's PK.
        let marked = ArrowField::new("count", DataType::Int64, false)
            .with_metadata([("k".to_string(), "v".to_string())].into());
        let schema = ArrowSchema::new_with_metadata(
            vec![ArrowField::new("id", DataType::Int32, false), marked],
            [("s".to_string(), "m".to_string())].into(),
        );

        let relaxed = relax_non_pk_nullability(&schema, &["id".to_string()]);

        assert_eq!(relaxed.metadata().get("s").map(String::as_str), Some("m"));
        assert_eq!(
            relaxed.field(1).metadata().get("k").map(String::as_str),
            Some("v")
        );
    }
}
