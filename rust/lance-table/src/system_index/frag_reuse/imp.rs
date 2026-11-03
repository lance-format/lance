// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The per-format implementation behind [`FragReuseHistory`].
//!
//! Each FRI format lives in its own file and implements [`FriImpl`]; the
//! public type only frames records and delegates every format-specific
//! decision here, so a format can be read and tested without the others.
//!
//! [`FragReuseHistory`]: super::history::FragReuseHistory

use std::fmt::Debug;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use super::history::{HistoryFormat, Record, split_records};
use super::ledger::FragReuseLedger;
use super::plan::{ReadContext, ReadPlan, SegmentSummary};
use crate::format::pb::fragment_reuse_index_details as pb;

/// One FRI format's rules over its records.
pub trait FriImpl: Send + Sync + Debug {
    fn format(&self) -> HistoryFormat;

    fn records(&self) -> &Records;

    /// Rebuild this format's history from `content`, validating it as a
    /// freshly decoded one.
    fn rebuilt(&self, content: Bytes) -> Result<Arc<dyn FriImpl>>;

    /// See [`FragReuseHistory::fragment_bitmap`](super::history::FragReuseHistory::fragment_bitmap).
    fn fragment_bitmap(&self) -> RoaringBitmap;

    fn into_tagged(self: Arc<Self>) -> Result<Arc<dyn FriImpl>>;

    fn append_transitions(&self, transitions: &[pb::Transition]) -> Result<Arc<dyn FriImpl>>;

    fn insert_legacy_version(&self, version: pb::Version) -> Result<Arc<dyn FriImpl>>;

    fn remap_stable_partition_bases(
        self: Arc<Self>,
        remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Arc<dyn FriImpl>>;

    /// See [`FragReuseHistory::on_stable_row_ids`](super::history::FragReuseHistory::on_stable_row_ids).
    fn on_stable_row_ids(self: Arc<Self>) -> Result<Arc<dyn FriImpl>>;

    /// See [`FragReuseHistory::is_caught_up`](super::history::FragReuseHistory::is_caught_up).
    fn is_caught_up(&self, entry_version: u64, segment: &SegmentSummary) -> bool;

    /// See [`FragReuseHistory::needs_translation_support`](super::history::FragReuseHistory::needs_translation_support).
    fn needs_translation_support(&self, context: &ReadContext, segment: &SegmentSummary) -> bool;

    /// See [`FragReuseHistory::plan`](super::history::FragReuseHistory::plan).
    fn plan(&self, context: &ReadContext, group: &[SegmentSummary]) -> Vec<ReadPlan>;
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
