// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The memtable's primary-key index: the newest visible position of a key, for
//! deduplication.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion::common::ScalarValue;
use lance_core::{Error, Result};

use super::pk_key::{encode_pk_batch, encode_pk_tuple};
use super::{BTreeMemIndex, IndexStore, PrimaryKeyIndex, RowPosition};

/// The column a composite key index is keyed on: the order-preserving encoded
/// tuple, stored as `Binary` so a [`BTreeMemIndex`] indexes it directly.
const PK_KEY_COLUMN: &str = "__pk_key__";

/// A composite key index is held directly, never looked up by field id.
const COMPOSITE_PK_FIELD_ID: i32 = -1;

pub(super) enum PkIndex {
    /// A user index on exactly the key column, the entry named `entry`, which
    /// the insert loop maintains. Held by name, since a plugin may hand out a
    /// fresh capability wrapper each time.
    Shared {
        index: Arc<dyn PrimaryKeyIndex>,
        entry: String,
    },
    /// A B-tree the store maintains itself, held apart from the named indexes
    /// so no user index name can collide with it.
    Owned(OwnedPk),
}

pub(super) enum OwnedPk {
    /// Over the key column.
    Single(Arc<BTreeMemIndex>),
    /// Over the encoded key tuple, which no batch carries, so it is built from
    /// `columns` at insert.
    Composite {
        index: Arc<BTreeMemIndex>,
        columns: Vec<String>,
        key_schema: SchemaRef,
    },
}

impl PkIndex {
    /// The key index for `pk_columns`, sharing a user index on exactly the key
    /// column when one can serve.
    pub(super) fn new(store: &IndexStore, pk_columns: &[(String, i32)]) -> Option<Self> {
        match pk_columns {
            [] => None,
            [(column, field_id)] => Some(
                store
                    .indexes
                    .iter()
                    .filter(|(_, index)| index.columns() == std::slice::from_ref(column))
                    .find_map(|(name, index)| {
                        let index = index.clone().as_primary_key()?;
                        Some(Self::Shared {
                            index,
                            entry: name.clone(),
                        })
                    })
                    .unwrap_or_else(|| {
                        Self::Owned(OwnedPk::Single(Arc::new(BTreeMemIndex::new(
                            *field_id,
                            column.clone(),
                        ))))
                    }),
            ),
            columns => Some(Self::Owned(OwnedPk::Composite {
                index: Arc::new(BTreeMemIndex::new(
                    COMPOSITE_PK_FIELD_ID,
                    PK_KEY_COLUMN.to_string(),
                )),
                columns: columns.iter().map(|(column, _)| column.clone()).collect(),
                key_schema: Arc::new(ArrowSchema::new(vec![Field::new(
                    PK_KEY_COLUMN,
                    DataType::Binary,
                    false,
                )])),
            })),
        }
    }

    pub(super) fn index(&self) -> &dyn PrimaryKeyIndex {
        match self {
            Self::Shared { index, .. } => index.as_ref(),
            Self::Owned(owned) => owned.btree().as_ref(),
        }
    }

    /// The key index the store maintains itself, if it shares no user index.
    pub(super) fn owned(&self) -> Option<&OwnedPk> {
        match self {
            Self::Shared { .. } => None,
            Self::Owned(owned) => Some(owned),
        }
    }

    pub(super) fn describe(&self) -> String {
        match self {
            Self::Shared { entry, .. } => format!("shared({entry})"),
            Self::Owned(OwnedPk::Single(index)) => format!("owned({})", index.column_name()),
            Self::Owned(OwnedPk::Composite { columns, .. }) => {
                format!("composite({})", columns.join(", "))
            }
        }
    }
}

impl OwnedPk {
    pub(super) fn btree(&self) -> &Arc<BTreeMemIndex> {
        match self {
            Self::Single(index) | Self::Composite { index, .. } => index,
        }
    }

    /// Index `batch`, reporting whether any key was already held when
    /// `report_existing`.
    pub(super) fn insert(
        &self,
        batch: &RecordBatch,
        row_offset: RowPosition,
        report_existing: bool,
    ) -> Result<bool> {
        let insert = |index: &BTreeMemIndex, batch: &RecordBatch| {
            if report_existing {
                index.insert_and_report_existing(batch, row_offset)
            } else {
                index.insert(batch, row_offset).map(|()| false)
            }
        };
        match self {
            Self::Single(index) => insert(index, batch),
            Self::Composite {
                index,
                columns,
                key_schema,
            } => {
                let positions = columns
                    .iter()
                    .map(|column| {
                        batch.schema().index_of(column).map_err(|_| {
                            Error::invalid_input(format!(
                                "primary-key column '{column}' is not in the batch"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let keys = RecordBatch::try_new(
                    key_schema.clone(),
                    vec![Arc::new(encode_pk_batch(batch, &positions)?)],
                )?;
                insert(index, &keys)
            }
        }
    }
}

impl IndexStore {
    /// Maintain a primary-key index, so the memtable can answer the newest
    /// visible version of a key. Call once, after the indexes are added and
    /// before any row is inserted.
    pub fn enable_pk_index(&mut self, pk_columns: &[(String, i32)]) {
        assert!(
            pk_columns.is_empty() || !self.has_rows.load(Ordering::Acquire),
            "a primary-key index must be enabled before any row is inserted"
        );
        self.pk_index = PkIndex::new(self, pk_columns);
    }

    /// Whether the memtable has a primary-key index.
    pub fn has_pk_index(&self) -> bool {
        self.pk_index.is_some()
    }

    /// Sorted `(key, row id)` batches for training the generation's on-disk
    /// key index: the typed key for one column, the encoded tuple for several.
    /// Positions are the flushed file's row ids.
    pub fn pk_training_batches(&self, batch_size: usize) -> Result<Vec<RecordBatch>> {
        match &self.pk_index {
            None => Ok(Vec::new()),
            Some(pk) => pk.index().training_batches(batch_size),
        }
    }

    /// The newest position of the key `values` (in key-column order) visible
    /// at `max_visible_row`, or `None`.
    pub fn pk_newest_visible(
        &self,
        values: &[ScalarValue],
        max_visible_row: RowPosition,
    ) -> Option<RowPosition> {
        match (self.pk_index.as_ref()?, values) {
            (PkIndex::Owned(OwnedPk::Composite { index, columns, .. }), values)
                if values.len() == columns.len() =>
            {
                // Insert encoded every stored key the same way, so a key that
                // fails to encode is not stored.
                let key = encode_pk_tuple(values).ok()?;
                index.newest_visible(&ScalarValue::Binary(Some(key)), max_visible_row)
            }
            (PkIndex::Owned(OwnedPk::Composite { .. }), _) => None,
            (pk, [value]) => pk.index().newest_visible(value, max_visible_row),
            (_, _) => None,
        }
    }

    /// Whether `position` is the newest visible row of `values`. False without
    /// a primary-key index, so callers check [`Self::has_pk_index`] first.
    pub fn pk_is_newest(
        &self,
        values: &[ScalarValue],
        position: RowPosition,
        max_visible_row: RowPosition,
    ) -> bool {
        self.pk_newest_visible(values, max_visible_row) == Some(position)
    }

    /// Whether `key`, already in the index's key space, has a version visible
    /// at `max_visible_row`.
    pub fn pk_contains_key(&self, key: &ScalarValue, max_visible_row: RowPosition) -> bool {
        self.pk_index
            .as_ref()
            .is_some_and(|pk| pk.index().newest_visible(key, max_visible_row).is_some())
    }

    /// Whether the primary-key index holds no rows, or there is none.
    pub fn pk_is_empty(&self) -> bool {
        self.pk_index
            .as_ref()
            .is_none_or(|pk| pk.index().is_empty())
    }

    /// Whether any key was rewritten. Never resets: a memtable flushes as a
    /// unit.
    pub fn pk_has_overrides(&self) -> bool {
        self.pk_has_overrides.load(Ordering::Acquire)
    }

    /// Whether inserts must report rewrites. Any index may push a search's
    /// top-k down, which a rewrite makes unsafe, so every keyed table tracks
    /// them.
    pub(super) fn should_track_pk_overrides(&self) -> bool {
        self.pk_index.is_some() && !self.pk_has_overrides()
    }

    /// The shared key index, if it is the entry named `name`.
    pub(super) fn shared_pk_named(&self, name: &str) -> Option<&Arc<dyn PrimaryKeyIndex>> {
        match &self.pk_index {
            Some(PkIndex::Shared { index, entry }) if entry == name => Some(index),
            _ => None,
        }
    }

    pub(super) fn mark_pk_overrides(&self, had_existing: bool) {
        if had_existing {
            self.pk_has_overrides.store(true, Ordering::Release);
        }
    }
}
