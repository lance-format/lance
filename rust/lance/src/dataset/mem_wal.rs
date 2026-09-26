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
pub(crate) mod reconcile;
pub mod scanner;
pub mod sharding;
#[cfg(test)]
pub(crate) mod test_util;
pub mod util;
mod wal;
pub mod write;

use std::sync::Arc;

use lance_core::datatypes::{Field, LANCE_FIELD_ID_KEY, Schema};

use arrow_array::RecordBatch;
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

/// The schema's Arrow form, with each field's id carried in its metadata.
///
/// `From<&Field> for ArrowField` drops the id, leaving everything downstream
/// matching on name — which loses a column across a rename. Arrow IPC preserves
/// field metadata, so entries written under this schema carry the id too.
///
/// Scoped to the memtable path on purpose: emitting ids from the global Arrow
/// conversion would change every schema Lance hands out, including for callers
/// that compare schemas for equality.
pub fn arrow_schema_with_field_ids(schema: &Schema) -> ArrowSchema {
    let arrow: ArrowSchema = schema.into();
    let fields: Vec<ArrowField> = arrow
        .fields()
        .iter()
        .map(|field| stamp_field_id(field, &schema.fields))
        .collect();
    ArrowSchema::new_with_metadata(fields, arrow.metadata().clone())
}

/// One field carrying its lance id, and its struct children carrying theirs.
///
/// A struct's children are fields in their own right: they have ids, a rename
/// moves one child's name and not the parent's, and a reader that cannot see a
/// child's id has only its name to go on.
fn stamp_field_id(field: &ArrowField, among: &[Field]) -> ArrowField {
    let Some(source) = among.iter().find(|f| f.name == *field.name()) else {
        return field.clone();
    };
    let field = match source.id {
        id if id >= 0 => {
            let mut metadata = field.metadata().clone();
            metadata.insert(LANCE_FIELD_ID_KEY.to_string(), id.to_string());
            field.clone().with_metadata(metadata)
        }
        _ => field.clone(),
    };
    // A container carries its children inside its own type, and each of them is
    // a field with an id of its own: a list's element, and that element's
    // children in turn.
    match field.data_type() {
        DataType::Struct(children) => {
            let children: Vec<ArrowField> = children
                .iter()
                .map(|child| stamp_field_id(child, &source.children))
                .collect();
            field.with_data_type(DataType::Struct(children.into()))
        }
        DataType::List(element) => {
            let element = stamp_field_id(element, &source.children);
            field.with_data_type(DataType::List(Arc::new(element)))
        }
        DataType::LargeList(element) => {
            let element = stamp_field_id(element, &source.children);
            field.with_data_type(DataType::LargeList(Arc::new(element)))
        }
        DataType::FixedSizeList(element, size) => {
            let size = *size;
            let element = stamp_field_id(element, &source.children);
            field.with_data_type(DataType::FixedSizeList(Arc::new(element), size))
        }
        DataType::Map(entries, sorted) => {
            let sorted = *sorted;
            let entries = stamp_field_id(entries, &source.children);
            field.with_data_type(DataType::Map(Arc::new(entries), sorted))
        }
        _ => field,
    }
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

/// `batches`, written under `source_schema`, brought to `target_schema`.
///
/// Columns match by field id where both sides carry one, by name otherwise, at
/// every level including inside structs, so a rename is followed. A column the
/// target declares and the batches lack is filled with typed nulls, `_tombstone`
/// with `false`; a column the target does not declare is dropped; a missing
/// primary key is an error.
///
/// `batches` must be in `source_schema` order. The result is in `target_schema`
/// order, without field ids.
pub fn reconcile_batches(
    source_schema: &ArrowSchema,
    target_schema: &Arc<ArrowSchema>,
    pk_columns: &[String],
    batches: Vec<RecordBatch>,
) -> lance_core::Result<Vec<RecordBatch>> {
    // A generation numbers its own columns -- `_tombstone`, and anything the
    // table has since dropped -- in its own schema, so those ids collide with
    // whatever the table gave those numbers. Stripped before resolution, or a
    // column added to the table resolves to whichever of them shares its id.
    let source = ArrowSchema::new_with_metadata(
        source_schema
            .fields()
            .iter()
            .map(|field| {
                match field.name() != TOMBSTONE && !lance_core::is_system_column(field.name()) {
                    true => field.as_ref().clone(),
                    false => reconcile::without_field_id(field),
                }
            })
            .collect::<Vec<_>>(),
        source_schema.metadata().clone(),
    );
    let plan =
        reconcile::Plan::resolve(&source, target_schema, pk_columns)?.emitting_plain_schema();
    if plan.is_identity() {
        return Ok(batches);
    }
    batches.iter().map(|batch| plan.apply(batch)).collect()
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

    fn stamped(name: &str, data_type: DataType, id: i32) -> ArrowField {
        ArrowField::new(name, data_type, true).with_metadata(
            [(LANCE_FIELD_ID_KEY.to_string(), id.to_string())]
                .into_iter()
                .collect(),
        )
    }

    /// A generation numbers `_tombstone` in its own schema, so its id is
    /// whatever that generation reached -- and the table has given that same
    /// number to a column of its own. Honouring it would resolve the two to
    /// each other and refuse the merge on their types.
    #[test]
    fn a_generations_tombstone_does_not_answer_for_a_column_sharing_its_id() {
        let source = ArrowSchema::new(vec![
            stamped("id", DataType::Int64, 0),
            stamped(TOMBSTONE, DataType::Boolean, 1),
        ]);
        // The table gave id 1 to a column added after that generation sealed.
        let target = Arc::new(ArrowSchema::new(vec![
            stamped("id", DataType::Int64, 0),
            stamped("extra", DataType::Int64, 1),
            ArrowField::new(TOMBSTONE, DataType::Boolean, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::new(source.clone()),
            vec![
                Arc::new(arrow_array::Int64Array::from(vec![1])),
                Arc::new(arrow_array::BooleanArray::from(vec![false])),
            ],
        )
        .expect("a batch under the source schema");

        let out = reconcile_batches(&source, &target, &["id".to_string()], vec![batch])
            .expect("the tombstone's id must not be honoured");
        let out = &out[0];
        assert!(
            out.column_by_name("extra").expect("extra").is_null(0),
            "the added column has no value in a generation sealed before it"
        );
        let tombstone = out
            .column_by_name(TOMBSTONE)
            .expect("_tombstone")
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .expect("boolean");
        assert!(!tombstone.value(0), "and the row is still live");
    }

    /// Two children exchanging names is the case a name match cannot survive:
    /// both sides carry the same two names, so only the ids say which values
    /// belong to which. Each child's values must follow its id to the name the
    /// target now gives it.
    #[test]
    fn a_pair_of_children_that_swapped_names_follow_their_ids() {
        let struct_of = |first: &str, second: &str, ids: (i32, i32)| {
            DataType::Struct(Fields::from(vec![
                stamped(first, DataType::Int64, ids.0),
                stamped(second, DataType::Int64, ids.1),
            ]))
        };
        let source = ArrowSchema::new(vec![
            stamped("id", DataType::Int64, 0),
            stamped("info", struct_of("a", "b", (1, 2)), 3),
        ]);
        // The table has since exchanged the two children's names; the ids stay.
        let target = Arc::new(ArrowSchema::new(vec![
            stamped("id", DataType::Int64, 0),
            stamped("info", struct_of("b", "a", (1, 2)), 3),
        ]));

        let info = arrow_array::StructArray::new(
            match source.field(1).data_type() {
                DataType::Struct(fields) => fields.clone(),
                _ => unreachable!("info is a struct"),
            },
            vec![
                Arc::new(arrow_array::Int64Array::from(vec![10])) as arrow_array::ArrayRef,
                Arc::new(arrow_array::Int64Array::from(vec![20])),
            ],
            None,
        );
        let batch = RecordBatch::try_new(
            Arc::new(source.clone()),
            vec![
                Arc::new(arrow_array::Int64Array::from(vec![1])),
                Arc::new(info),
            ],
        )
        .expect("a batch under the source schema");

        let out = reconcile_batches(&source, &target, &["id".to_string()], vec![batch])
            .expect("reconcile");
        let info = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .expect("info is a struct");

        // `b` is the name id 1 now wears, so it must hold id 1's value.
        let b = info
            .column_by_name("b")
            .expect("b")
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("int64");
        assert_eq!(b.value(0), 10, "id 1's value follows its id to `b`");

        let a = info
            .column_by_name("a")
            .expect("a")
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("int64");
        assert_eq!(a.value(0), 20, "id 2's value follows its id to `a`");
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
