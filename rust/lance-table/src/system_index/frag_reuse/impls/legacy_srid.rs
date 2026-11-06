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
use lance_core::{Error, Result};
use lance_select::Remapper;
use roaring::RoaringBitmap;

use super::legacy::{
    VersionNeed, repair_versions, version_chain, version_need, warn_dropped_versions,
};
use super::{FriImpl, Records};
use crate::format::pb::fragment_reuse_index_details as pb;
use crate::system_index::frag_reuse::history::{HistoryFormat, Record, RewriteRecord};
use crate::system_index::frag_reuse::mapping::MappingSource;
use crate::system_index::frag_reuse::plan::{
    Composition, ReadContext, ReadPlan, RewriteContext, RowDomain, SegmentSummary, TranslationPlan,
    TranslationSupport,
};

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

    /// Tagged histories translate row addresses, which stable row IDs
    /// replace, so a table with stable row IDs only ever records the legacy
    /// format.
    fn lift(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Err(Error::not_supported(
            "the dataset uses stable row IDs, which exclude tagged fragment reuse histories \
             (they are address-based), so this rewrite cannot be recorded as tagged transitions \
             on a stable-row-id dataset",
        ))
    }

    fn append(&self, rewrite: &RewriteRecord) -> Result<Arc<dyn FriImpl>> {
        self.0.append(rewrite).map(Self::wrap)
    }

    fn remap_bases(
        self: Arc<Self>,
        remap: &dyn Fn(Option<u32>) -> Result<Option<u32>>,
    ) -> Result<Arc<dyn FriImpl>> {
        self.0.clone().remap_bases(remap).map(Self::wrap)
    }

    fn on_stable_row_ids(self: Arc<Self>) -> Result<Arc<dyn FriImpl>> {
        Ok(self)
    }

    fn is_caught_up(&self, entry_version: u64, segment: &SegmentSummary) -> bool {
        self.0.is_caught_up(entry_version, segment)
    }

    /// Only the oldest versions are removed, up to the first one a segment
    /// is not caught up with, and every later version is kept as it is: an
    /// address a kept version maps still translates through every later
    /// recorded move. As in the address cleanup, a trim that removes a
    /// version writes only the kept compaction versions, dropping records
    /// this build cannot interpret.
    fn trim(self: Arc<Self>, live: &[Vec<SegmentSummary>]) -> Result<Arc<dyn FriImpl>> {
        let versions: Vec<pb::Version> = self
            .0
            .records()
            .stored
            .iter()
            .filter_map(|stored| match &stored.record {
                Record::LegacyVersion(version) => Some(version.clone()),
                _ => None,
            })
            .collect();
        let chain = version_chain(&versions);
        // Unlike the address cleanup, a corrupt version is not dropped: no
        // segment's state with it can be proven, so it ends the removable
        // prefix like a needed one.
        let trimmed = versions
            .iter()
            .position(|version| version_need(version, live, &chain) != VersionNeed::CaughtUp)
            .unwrap_or(versions.len());
        if trimmed == 0 {
            return Ok(self);
        }
        let (kept, dropped) = repair_versions(versions[trimmed..].to_vec())?;
        warn_dropped_versions(&dropped);
        kept.imp().clone().on_stable_row_ids()
    }

    fn needs_translation_support(&self, _context: &ReadContext, _segment: &SegmentSummary) -> bool {
        true
    }

    fn plan(self: Arc<Self>, context: &ReadContext, group: &[SegmentSummary]) -> Vec<ReadPlan> {
        self.0
            .clone()
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

    fn compose(
        self: Arc<Self>,
        context: &RewriteContext,
        segment: &SegmentSummary,
    ) -> Result<Composition> {
        self.0.clone().compose(context, segment)
    }

    fn bind(&self, plan: TranslationPlan, mappings: Arc<dyn MappingSource>) -> Remapper {
        self.0.bind(plan, mappings)
    }
}
