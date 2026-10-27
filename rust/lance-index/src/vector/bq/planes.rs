// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Independent cache entries for the sign and ex columns of native RaBitQ.
//! The persisted index keeps its existing columns and quantization metadata.

use super::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN};
use super::transform::{
    ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
    SCALE_FACTORS_COLUMN,
};
use arrow_array::RecordBatch;
use lance_core::cache::{CacheCodec, CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::deepsize::{Context, DeepSizeOf};
use std::borrow::Cow;

/// One plane and its factors, or only its code columns when factors are resident.
#[derive(Debug)]
pub struct PlaneBatch(pub RecordBatch);
impl DeepSizeOf for PlaneBatch {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}
/// Which columns an IVF_RQ index's cache entries hold: native
/// sign/ex plane entries or a whole partition's entry. The
/// composition decides what an entry holds, so it is part of its key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum EntryColumns {
    /// Every column a read of the plane or partition returns.
    #[default]
    All,
    /// Only the columns the index's resident store does not keep: the codes
    /// (see [`plane_entry_columns`]). Every read
    /// attaches the store's rows of the others, views of the store for a
    /// whole plane or partition and copies for gathered rows, so the scored
    /// batch is the one an `All` entry holds. Only an index whose small
    /// columns are resident keeps such entries.
    Codes,
}

impl EntryColumns {
    /// The spelling of `LANCE_RQ_ENTRY_COLUMNS`: `all` or `codes`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Codes => "codes",
        }
    }

    /// What an index that asks for `self` keeps in its entries: `Codes`
    /// needs the resident store, so without it the entries hold `All`.
    pub fn resolve(self, resident: bool) -> Self {
        match self {
            Self::Codes if resident => Self::Codes,
            _ => Self::All,
        }
    }
}

impl std::fmt::Display for EntryColumns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A native sign (0) or ex (1) entry under an immutable index namespace.
pub struct PlaneKey {
    /// Physical partition ordinal.
    pub partition: usize,
    /// Physical plane ordinal: sign is 0 and ex is 1.
    pub plane: u8,
    /// Stored columns, included in the key to keep the compositions distinct.
    pub entry_columns: EntryColumns,
}
impl CacheKey for PlaneKey {
    type ValueType = PlaneBatch;
    fn key(&self) -> Cow<'_, str> {
        format!("{}:{}:{}", self.partition, self.plane, self.entry_columns).into()
    }
    fn type_name() -> &'static str {
        "RabitPlane"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.rabit-plane-key", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.partition as u64);
        builder.write_u8(self.plane);
        builder.write_u8(u8::from(self.entry_columns == EntryColumns::Codes));
    }
    fn codec_for_key(&self) -> Option<CacheCodec> {
        Self::codec().map(|codec| {
            codec
                .with_plane_tag(self.plane)
                .with_memory_priority(3u8.saturating_sub(self.plane))
        })
    }
    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<PlaneBatch>())
    }
}

/// Native column projection of a sign (0) or ex (1) plane; empty for other ids.
pub fn plane_columns(plane: u8) -> &'static [&'static str] {
    match plane {
        0 => &[
            lance_core::ROW_ID,
            RABIT_CODE_COLUMN,
            ADD_FACTORS_COLUMN,
            SCALE_FACTORS_COLUMN,
            ERROR_FACTORS_COLUMN,
        ],
        1 => &[
            RABIT_BLOCKED_EX_CODE_COLUMN,
            EX_ADD_FACTORS_COLUMN,
            EX_SCALE_FACTORS_COLUMN,
        ],
        _ => &[],
    }
}

/// Columns kept by an entry in the plane's original order.
pub fn plane_entry_columns(plane: u8, columns: EntryColumns) -> Vec<&'static str> {
    plane_columns(plane)
        .iter()
        .copied()
        .filter(|name| columns == EntryColumns::All || super::resident::is_file_column(name))
        .collect()
}
