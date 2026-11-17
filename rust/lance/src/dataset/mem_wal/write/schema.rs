// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Writer schema derivation, batch validation, and tombstone shaping.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, new_null_array};
use arrow_schema::Schema as ArrowSchema;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};

use super::super::reconcile::{Plan, without_field_ids};
use super::super::{
    MemTableDataTarget, TOMBSTONE, relax_non_pk_nullability, schema_with_tombstone,
};
use crate::blob::logical_to_prepared_blob_schema;
use crate::dataset::mem_wal::index::{IndexStore, MemIndexSpec, validate_index_specs};
use crate::dataset::mem_wal::memtable::{CacheConfig, MemTable};
use crate::dataset::mem_wal::wal::WriterCursors;

/// Whether two maintained sets describe the same indexes.
pub(super) fn same_index_set(current: &[MemIndexSpec], next: &[MemIndexSpec]) -> bool {
    if current.len() != next.len() {
        return false;
    }
    // Names are unique within a set, so ordering by name is canonical.
    fn by_name(specs: &[MemIndexSpec]) -> Vec<&MemIndexSpec> {
        let mut specs: Vec<&MemIndexSpec> = specs.iter().collect();
        specs.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        specs
    }
    by_name(current)
        .into_iter()
        .zip(by_name(next))
        .all(|(a, b)| a.same_index(b))
}

/// Pair each primary-key column name with its field id (both derived from the
/// schema's primary key, in the same order) for [`IndexStore::enable_pk_index`].
pub(super) fn pk_index_columns(pk_columns: &[String], pk_field_ids: &[i32]) -> Vec<(String, i32)> {
    pk_columns
        .iter()
        .cloned()
        .zip(pk_field_ids.iter().copied())
        .collect()
}

/// Reject caller input whose columns or types differ from the logical schema,
/// or that holds a null in a non-nullable column.
///
/// This is the only nullability check: nothing later rejects a bad null. It
/// must run before the WAL append, or a bad batch would fail every replay.
pub(super) fn validate_against_logical_schema(
    logical_schema: &Arc<ArrowSchema>,
    batches: &[RecordBatch],
) -> Result<()> {
    for (i, batch) in batches.iter().enumerate() {
        // Later steps match columns by position, so check names here too.
        for (col, (expected, actual)) in logical_schema
            .fields()
            .iter()
            .zip(batch.schema().fields())
            .enumerate()
        {
            if expected.name() != actual.name() {
                return Err(Error::invalid_input(format!(
                    "batch {i} column {col} is named '{}', but the base table schema \
                     declares '{}' at that position",
                    actual.name(),
                    expected.name()
                )));
            }
        }
        RecordBatch::try_new(logical_schema.clone(), batch.columns().to_vec()).map_err(|e| {
            Error::invalid_input(format!(
                "batch {i} does not match the base table schema: {e}"
            ))
        })?;
    }
    Ok(())
}

/// A batch a caller just handed in, under the storage schema.
///
/// A live batch has already been validated against the logical schema, so its
/// columns are the right ones in the right order, and any ids it carries are the
/// caller's rather than the table's. Dropping them leaves matching by name.
///
/// A replayed entry is the opposite: its ids are the table's, and they are what
/// survives a rename.
fn conform_live_batch(
    batch: RecordBatch,
    storage_schema: &Arc<ArrowSchema>,
    pk_columns: &[String],
) -> Result<RecordBatch> {
    let plain = Arc::new(without_field_ids(batch.schema().as_ref()));
    let batch = RecordBatch::try_new_with_options(
        plain,
        batch.columns().to_vec(),
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| Error::invalid_input(format!("bind a live batch to its own schema: {e}")))?;
    conform_to_storage_schema(batch, storage_schema, pk_columns)
}

/// Re-label `batch` to the storage schema, matching columns by **field id**
/// where both sides carry one, and by **name** otherwise.
///
/// A column the schema declares and the batch lacks is filled with typed nulls,
/// `_tombstone` with `false`; a column the schema does not declare is dropped.
/// This is what a replayed entry looks like once the schema has moved on.
///
/// Ids are tried first because a name match across a rename would null the new
/// name and drop the old one, losing the values. Entries without ids fall back
/// to names.
///
/// A column whose scalar type has moved is an error rather than a cast: a table
/// with a MemWAL refuses a retype, so a disagreement here is one to surface.
///
/// A primary key the batch does not carry is an error — there is no value to
/// invent.
pub(super) fn conform_to_storage_schema(
    batch: RecordBatch,
    storage_schema: &Arc<ArrowSchema>,
    pk_columns: &[String],
) -> Result<RecordBatch> {
    let plan = Plan::resolve(batch.schema().as_ref(), storage_schema, pk_columns)?;
    if plan.is_identity() {
        return Ok(batch);
    }
    plan.apply(&batch)
}

/// Build a tombstone batch from a key-only `keys` batch: primary keys carried
/// through, `_tombstone` true, every other column null.
///
/// Non-PK columns are nullable in the storage schema however the base table
/// declares them — that is what lets a strict table have tombstones at all.
/// Primary keys are not, so the validation below still rejects a null,
/// mistyped, or missing key.
fn build_tombstone_batch(
    keys: &RecordBatch,
    storage_schema: &Arc<ArrowSchema>,
    pk_columns: &[String],
) -> Result<RecordBatch> {
    let n = keys.num_rows();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(storage_schema.fields().len());
    for field in storage_schema.fields() {
        let name = field.name();
        if name == TOMBSTONE {
            columns.push(Arc::new(BooleanArray::from(vec![true; n])));
        } else if pk_columns.iter().any(|c| c == name) {
            let col = keys.column_by_name(name).ok_or_else(|| {
                Error::invalid_input(format!(
                    "delete keys batch is missing primary key column '{}'",
                    name
                ))
            })?;
            columns.push(col.clone());
        } else {
            columns.push(new_null_array(field.data_type(), n));
        }
    }
    RecordBatch::try_new(storage_schema.clone(), columns).map_err(|e| {
        Error::invalid_input(format!(
            "failed to build tombstone batch (do the delete keys match the primary key?): {}",
            e
        ))
    })
}

/// The schema a memtable-mode writer checks writes against, and what is
/// derived from it.
///
/// Each memtable keeps the one it was created under, even after
/// [`ShardWriter::evolve_schema`] or [`ShardWriter::replace_index_configs`].
#[derive(Clone)]
pub(super) struct WriterSchema {
    /// The base table's schema as the caller passed it, without field ids.
    /// Caller input is checked against it.
    pub(super) logical: Arc<ArrowSchema>,
    /// What memtables, WAL entries and generations store: `logical` with field
    /// ids, `_tombstone` added, and non-PK columns nullable.
    pub(super) storage: Arc<ArrowSchema>,
    /// `storage` with Blob v2 payload columns replaced by their prepared
    /// descriptors. Memtables are created under this one.
    pub(super) prepared: Arc<ArrowSchema>,
    /// `storage` as a Lance schema, for validating index specs and resolving
    /// FTS field ids.
    pub(super) lance_schema: Schema,
    pub(super) pk_field_ids: Vec<i32>,
    /// Primary-key column names, for each memtable's PK-position index.
    pub(super) pk_columns: Vec<String>,
    pub(super) index_specs: Arc<[MemIndexSpec]>,
}

impl WriterSchema {
    /// Derive the writer's schemas from the caller's, which should carry field
    /// ids under `lance:field_id`.
    pub(super) fn try_new(schema: &ArrowSchema, index_specs: Vec<MemIndexSpec>) -> Result<Self> {
        // lance owns `_tombstone` and appends it here — idempotent across reopens.
        let tombstoned = schema_with_tombstone(schema);
        let pk_fields = Schema::try_from(tombstoned.as_ref())?;
        let pk_fields = pk_fields.unenforced_primary_key();
        let pk_field_ids: Vec<i32> = pk_fields.iter().map(|f| f.id).collect();
        let pk_columns: Vec<String> = pk_fields.iter().map(|f| f.name.clone()).collect();
        // Widen only now that the primary key is known — a tombstone nulls
        // every non-PK column, and PK detection needs the strict schema.
        let storage = relax_non_pk_nullability(&tombstoned, &pk_columns);
        let prepared = Arc::new(logical_to_prepared_blob_schema(storage.as_ref())?);
        Self {
            // No field ids: a caller's batch has none, and Arrow compares
            // struct children in full.
            logical: Arc::new(without_field_ids(schema)),
            lance_schema: Schema::try_from(storage.as_ref())?,
            storage,
            prepared,
            pk_field_ids,
            pk_columns,
            index_specs: Arc::from([]),
        }
        .with_index_specs(index_specs)
    }

    /// The same schema with `index_specs` instead.
    ///
    /// A bad config is rejected before any row is accepted, since it would
    /// fail every insert, including WAL replay.
    pub(super) fn with_index_specs(&self, index_specs: Vec<MemIndexSpec>) -> Result<Self> {
        validate_index_specs(
            &index_specs,
            self.storage.as_ref(),
            &self.lance_schema,
            &self.pk_columns,
        )?;
        Ok(Self {
            index_specs: index_specs.into(),
            ..self.clone()
        })
    }

    /// Whether memtables need a data target, so Blob v2 payloads can be
    /// written before the WAL append.
    fn preassigns_data_target(&self) -> bool {
        self.storage != self.prepared
    }

    /// Whether switching between the two would change nothing stored or indexed.
    pub(super) fn is_equivalent(&self, other: &Self) -> bool {
        self.storage == other.storage
            && self.logical == other.logical
            && same_index_set(&self.index_specs, &other.index_specs)
    }

    /// A new memtable under this schema at the given generation and offset.
    ///
    /// Always binds an `IndexStore`, even with no indexes: readers use its
    /// `indexed_count` to avoid showing rows before they are durable.
    /// `target` is the one a replayed entry recorded; `None` makes a new one if needed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_memtable(
        &self,
        generation: u64,
        global_offset: usize,
        target: Option<MemTableDataTarget>,
        epoch: u64,
        max_memtable_rows: usize,
        max_memtable_batches: usize,
        cursors: &Arc<WriterCursors>,
    ) -> Result<MemTable> {
        let target = target.or_else(|| {
            self.preassigns_data_target()
                .then(|| MemTableDataTarget::new(generation, epoch, max_memtable_batches))
        });
        let batch_capacity = target
            .as_ref()
            .map(|target| target.batch_capacity)
            .unwrap_or(max_memtable_batches)
            .max(max_memtable_batches);
        let mut memtable = MemTable::with_capacity_at_target(
            self.prepared.clone(),
            generation,
            self.pk_field_ids.clone(),
            CacheConfig::default(),
            batch_capacity,
            global_offset,
            target,
        )?;
        let mut indexes = IndexStore::from_specs(
            &self.index_specs,
            &self.lance_schema,
            max_memtable_rows,
            batch_capacity,
        )?;
        if !self.pk_columns.is_empty() {
            indexes.enable_pk_index(&pk_index_columns(&self.pk_columns, &self.pk_field_ids));
        }
        indexes.set_durability(Arc::clone(cursors), global_offset);
        memtable.set_indexes_arc(Arc::new(indexes));
        Ok(memtable)
    }
}

/// A memtable-mode write, before it is shaped to its memtable's schema.
///
/// Shaped under the writer lock, so a write checked against one schema cannot
/// land in a memtable with another.
pub(super) enum Incoming {
    /// Rows in the logical schema.
    Rows(Vec<RecordBatch>),
    /// Key-only batches, each becoming one tombstone batch.
    Tombstones(Vec<RecordBatch>),
}

impl Incoming {
    /// Number of batches the insert will append.
    pub(super) fn batch_count(&self) -> usize {
        match self {
            Self::Rows(batches) | Self::Tombstones(batches) => batches.len(),
        }
    }

    /// Number of rows the insert will append.
    pub(super) fn row_count(&self) -> usize {
        match self {
            Self::Rows(batches) | Self::Tombstones(batches) => {
                batches.iter().map(|b| b.num_rows()).sum()
            }
        }
    }

    /// The batches to insert, under `schema`'s storage schema.
    pub(super) fn shape(self, schema: &WriterSchema) -> Result<Vec<RecordBatch>> {
        match self {
            Self::Rows(batches) => {
                validate_against_logical_schema(&schema.logical, &batches)?;
                // Callers pass logical-shaped batches and never name
                // `_tombstone`.
                batches
                    .into_iter()
                    .map(|b| conform_live_batch(b, &schema.storage, &schema.pk_columns))
                    .collect()
            }
            Self::Tombstones(keys) => {
                if schema.pk_columns.is_empty() {
                    return Err(Error::invalid_input(
                        "delete requires a primary key, but this shard has no primary key columns",
                    ));
                }
                keys.iter()
                    .map(|k| build_tombstone_batch(k, &schema.prepared, &schema.pk_columns))
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests;
