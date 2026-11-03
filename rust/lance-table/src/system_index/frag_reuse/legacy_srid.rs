// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The legacy (`index_version` 0) fragment reuse history on a table with
//! stable row IDs.
//!
//! The records are those of the legacy format. What differs is reading: the
//! legacy reader applies the history to every index segment it opens, as
//! row-address moves. A segment that stores stable row IDs, or whose kind
//! cannot apply the history, would have its identifiers rewritten as if they
//! were addresses, so it is excluded whether or not it is caught up.

use std::sync::Arc;

use bytes::Bytes;
use lance_core::Result;
use roaring::RoaringBitmap;

use super::history::HistoryFormat;
use super::imp::{FriImpl, Records};
use super::plan::{ReadContext, ReadPlan, RowDomain, SegmentSummary, TranslationSupport};
use crate::format::pb::fragment_reuse_index_details as pb;

#[derive(Debug)]
pub struct LegacySridHistory(Arc<dyn FriImpl>);

impl LegacySridHistory {
    /// The stable-row-ID reading of `legacy`, a legacy history.
    pub fn new(legacy: Arc<dyn FriImpl>) -> Self {
        Self(legacy)
    }

    fn wrap(legacy: Arc<dyn FriImpl>) -> Arc<dyn FriImpl> {
        Arc::new(Self(legacy))
    }
}

impl FriImpl for LegacySridHistory {
    fn format(&self) -> HistoryFormat {
        self.0.format()
    }

    fn records(&self) -> &Records {
        self.0.records()
    }

    fn rebuilt(&self, content: Bytes) -> Result<Arc<dyn FriImpl>> {
        self.0.rebuilt(content).map(Self::wrap)
    }

    fn fragment_bitmap(&self) -> RoaringBitmap {
        self.0.fragment_bitmap()
    }

    fn into_tagged(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        self.0.clone().into_tagged()
    }

    fn append_transitions(&self, transitions: &[pb::Transition]) -> Result<Arc<dyn FriImpl>> {
        self.0.append_transitions(transitions).map(Self::wrap)
    }

    fn insert_legacy_version(&self, version: pb::Version) -> Result<Arc<dyn FriImpl>> {
        self.0.insert_legacy_version(version).map(Self::wrap)
    }

    fn remap_stable_partition_bases(
        self: Arc<Self>,
        remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Arc<dyn FriImpl>> {
        self.0
            .clone()
            .remap_stable_partition_bases(remap)
            .map(Self::wrap)
    }

    fn on_stable_row_ids(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Ok(self)
    }

    fn is_caught_up(&self, entry_version: u64, segment: &SegmentSummary) -> bool {
        self.0.is_caught_up(entry_version, segment)
    }

    fn needs_translation_support(&self, _context: &ReadContext, _segment: &SegmentSummary) -> bool {
        true
    }

    fn plan(&self, context: &ReadContext, group: &[SegmentSummary]) -> Vec<ReadPlan> {
        self.0
            .plan(context, group)
            .into_iter()
            .zip(group)
            .map(|(plan, segment)| {
                if segment.domain == RowDomain::Address
                    && segment.translation == TranslationSupport::Supported
                {
                    plan
                } else {
                    ReadPlan::Excluded
                }
            })
            .collect()
    }
}
