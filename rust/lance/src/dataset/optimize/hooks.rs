// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lance_table::format::IndexMetadata;
use lance_table::system_index::is_system_index;
use lance_table_runtime::Table;
use lance_table_runtime::optimize::{
    CompactionHooks, CompactionIndexHook, CompactionIndexSegment, IndexRemapperFactory,
};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::dataset::index::{
    constrain_candidate_windows, is_merge_candidate, load_indices_for_remapping,
    vector_segment_merge_groups_for_compaction,
};
use crate::dataset::write::seeds::LanceWriteSeedHook;
use crate::index::frag_reuse_with_stable_row_ids::validate_compaction_index_recording;
use crate::index::segment::segment_has_vector_details;
use crate::index::vector::details::needs_vector_details_inference;
use crate::io::commit::LanceCommitHook;
use crate::{Dataset, Result};

use super::{IndexRemapper, IndexRemapperOptions};

pub(super) struct LanceCompactionHooks<'a> {
    commit: Arc<LanceCommitHook>,
    write_seeds: Arc<LanceWriteSeedHook>,
    remapper: Option<&'a dyn IndexRemapperOptions>,
}

impl<'a> LanceCompactionHooks<'a> {
    pub(super) fn new(dataset: &Dataset, remapper: Option<&'a dyn IndexRemapperOptions>) -> Self {
        Self {
            commit: Arc::new(LanceCommitHook::new(dataset.session.clone())),
            write_seeds: Arc::new(LanceWriteSeedHook::new(dataset.session.clone())),
            remapper,
        }
    }

    pub(super) fn runtime(&self) -> CompactionHooks<'_> {
        CompactionHooks {
            index: Some(self),
            remapper: self.remapper.map(|_| self as &dyn IndexRemapperFactory),
            commit_index: Some(self.commit.clone()),
            commit_lifecycle: Some(self.commit.as_ref()),
            write_seeds: Some(self.write_seeds.clone()),
        }
    }
}

#[async_trait]
impl CompactionIndexHook for LanceCompactionHooks<'_> {
    async fn merge_index_segments(
        &self,
        table: &Table,
        segments: &[CompactionIndexSegment<'_>],
        candidate_windows: &[RoaringBitmap],
    ) -> Vec<Vec<Uuid>> {
        let dataset = self.commit.dataset(table);
        let mut segments_by_name = HashMap::<&str, Vec<_>>::new();
        for segment in segments {
            segments_by_name
                .entry(segment.index.name.as_str())
                .or_default()
                .push(segment);
        }

        // Apply the manifest's known boundaries before opening any stored
        // models. Opaque and incompatible segments retain their boundaries.
        let mut potential_boundaries = Vec::new();
        for segments in segments_by_name.values() {
            let mut eligible_coverage = RoaringBitmap::new();
            for segment in segments {
                if is_merge_candidate(segment.index) {
                    eligible_coverage |= &segment.coverage;
                } else {
                    potential_boundaries.push(segment.coverage.clone());
                }
            }
            if !eligible_coverage.is_empty() {
                potential_boundaries.push(eligible_coverage);
            }
        }
        let candidate_windows =
            constrain_candidate_windows(candidate_windows, &potential_boundaries);

        let mut merge_groups = Vec::new();
        // Visit names in manifest order, and inspect models only when an
        // otherwise viable candidate window crosses a physical boundary.
        for segment in segments {
            let Some(segments) = segments_by_name.remove(segment.index.name.as_str()) else {
                continue;
            };
            let relevant = segments
                .iter()
                .map(|segment| segment.index)
                .filter(|segment| {
                    segment.fragment_bitmap.as_ref().is_some_and(|coverage| {
                        candidate_windows.iter().any(|window| {
                            !coverage.is_disjoint(window) && !window.is_subset(coverage)
                        })
                    })
                })
                .collect::<Vec<_>>();
            merge_groups.extend(
                vector_segment_merge_groups_for_compaction(&dataset, &relevant)
                    .await
                    .into_iter()
                    .filter(|group| group.len() > 1)
                    .map(|group| group.into_iter().map(|index| index.uuid).collect()),
            );
        }
        merge_groups
    }

    fn requires_whole_segment(&self, table: &Table, index: &IndexMetadata) -> bool {
        // Scalar indices handle partial coverage through fragment reuse.
        segment_has_vector_details(index) || needs_vector_details_inference(index, table.schema())
    }

    async fn needs_row_addrs(&self, table: &Table) -> Result<bool> {
        let dataset = self.commit.dataset(table);
        let Some(indices) = load_indices_for_remapping(&dataset).await? else {
            return Ok(false);
        };
        Ok(!table.manifest().uses_stable_row_ids()
            || indices
                .iter()
                .any(|index| !is_system_index(index) && index.supports_eager_remap()))
    }

    fn validate_stable_row_id_recording(&self, indices: &[IndexMetadata]) -> Result<()> {
        validate_compaction_index_recording(indices)
    }
}

#[async_trait]
impl IndexRemapperFactory for LanceCompactionHooks<'_> {
    fn supports_vector_segment_replacement(&self) -> bool {
        self.remapper
            .is_some_and(|remapper| remapper.supports_vector_segment_replacement())
    }

    async fn create_remapper(&self, table: &Table) -> Result<Option<Box<dyn IndexRemapper>>> {
        match self.remapper {
            Some(remapper) => remapper.create_remapper(&self.commit.dataset(table)).await,
            None => Ok(None),
        }
    }
}
