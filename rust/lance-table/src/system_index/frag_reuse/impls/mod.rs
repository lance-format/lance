// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The per-format implementation behind [`FragReuseHistory`].
//!
//! Each FRI format lives in its own file and implements [`FriImpl`]; the
//! public type only frames records and delegates every format-specific
//! decision here, so a format can be read and tested without the others.
//!
//! [`FragReuseHistory`]: super::history::FragReuseHistory

pub(super) mod legacy;
mod legacy_srid;
mod lift;
pub(super) mod tagged;

use std::fmt::Debug;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use lance_core::{Error, Result};
use lance_select::Remapper;
use roaring::RoaringBitmap;

use super::history::{HistoryFormat, Record, RewriteRecord, split_records};
use super::ledger::FragReuseLedger;
use super::mapping::MappingSource;
use super::plan::{ReadContext, ReadPlan, SegmentSummary, TranslationPlan};

/// One FRI format's rules over its records.
pub trait FriImpl: Send + Sync + Debug {
    fn format(&self) -> HistoryFormat;

    fn records(&self) -> &Records;

    /// Rebuild this format's history from `content`, validating it as a
    /// freshly decoded one.
    fn rebuilt(&self, content: Bytes) -> Result<Arc<dyn FriImpl>>;

    /// See [`FragReuseHistory::fragment_bitmap`](super::history::FragReuseHistory::fragment_bitmap).
    fn fragment_bitmap(&self) -> RoaringBitmap;

    /// See [`FragReuseHistory::lift`](super::history::FragReuseHistory::lift).
    fn lift(self: Arc<Self>) -> Result<Arc<dyn FriImpl>>;

    /// Record `rewrite` in this format. The format is already the one the
    /// table's policy chose (see
    /// [`FragReuseHistory::append`](super::history::FragReuseHistory::append)).
    fn append(&self, rewrite: &RewriteRecord) -> Result<Arc<dyn FriImpl>>;

    /// See [`FragReuseHistory::remap_bases`](super::history::FragReuseHistory::remap_bases).
    fn remap_bases(
        self: Arc<Self>,
        remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Arc<dyn FriImpl>>;

    /// See [`FragReuseHistory::on_stable_row_ids`](super::history::FragReuseHistory::on_stable_row_ids).
    fn on_stable_row_ids(self: Arc<Self>) -> Result<Arc<dyn FriImpl>>;

    /// Whether `segment` holds no address the history still moves, so it can
    /// be used without translation, and no record is needed for it.
    ///
    /// `entry_version` is the `dataset_version` of the FRI entry holding the
    /// history (see [`ReadContext::entry_version`]). A segment without a
    /// recorded fragment bitmap is caught up only under the legacy rule, which
    /// released writers relied on.
    fn is_caught_up(&self, entry_version: u64, segment: &SegmentSummary) -> bool;

    /// See [`FragReuseHistory::needs_translation_support`](super::history::FragReuseHistory::needs_translation_support).
    fn needs_translation_support(&self, context: &ReadContext, segment: &SegmentSummary) -> bool;

    /// See [`FragReuseHistory::plan`](super::history::FragReuseHistory::plan).
    /// Translation plans keep the history they were planned under, so they
    /// can be bound without it.
    fn plan(self: Arc<Self>, context: &ReadContext, group: &[SegmentSummary]) -> Vec<ReadPlan>;

    /// See [`TranslationPlan::bind`]. `plan` was planned by this history.
    fn bind(&self, plan: TranslationPlan, mappings: Arc<dyn MappingSource>) -> Remapper;
}

/// The records every format stores: each record's exact wire bytes in
/// serialized order, and the lineage view decoded from them.
#[derive(Debug, Clone, Default)]
pub struct Records {
    pub stored: Vec<StoredRecord>,
    pub ledger: Arc<FragReuseLedger>,
}

#[derive(Debug, Clone)]
pub struct StoredRecord {
    /// The complete field: key, length and payload.
    pub raw: Bytes,
    pub record: Record,
}

impl Records {
    pub fn decode(format: HistoryFormat, content: Bytes) -> Result<Self> {
        let ledger = FragReuseLedger::decode_content(format, content.clone())?;
        let stored = split_records(content)?;
        Ok(Self {
            stored,
            ledger: Arc::new(ledger),
        })
    }

    pub fn is_fully_interpretable(&self) -> bool {
        !self.ledger.has_unsupported_transitions()
    }

    pub fn require_fully_interpretable(&self, operation: &str) -> Result<()> {
        if self.is_fully_interpretable() {
            return Ok(());
        }
        Err(Error::not_supported(format!(
            "the fragment reuse history contains records this writer cannot maintain, so it \
             cannot {operation} it; upgrade to a newer version of Lance"
        )))
    }
}

pub fn concat(records: &[StoredRecord]) -> Bytes {
    let mut content = BytesMut::with_capacity(records.iter().map(|r| r.raw.len()).sum());
    for stored in records {
        content.extend_from_slice(&stored.raw);
    }
    content.freeze()
}

/// A validated history became invalid by an edit: the records the caller
/// added are what is wrong, not the stored history.
pub fn invalid_edit(error: Error) -> Error {
    match error {
        Error::CorruptFile { source, .. } => Error::invalid_input(format!(
            "the records added to the fragment reuse history are invalid: {source}"
        )),
        other => other,
    }
}
