// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index-cache prewarming and per-segment cache diagnostics.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use lance_core::cache::{CachePrewarmPolicy, InternalCacheKey};
use lance_index::{
    FtsPrewarmDiagnostics, FtsPrewarmOptions, FtsPrewarmResult, FtsPrewarmSegmentStatus, Index,
    PrewarmOptions, metrics::NoOpMetricsCollector, scalar::inverted::InvertedIndex,
};
use lance_table::format::IndexMetadata;
use tracing::{info, warn};
use uuid::Uuid;

use crate::index::DatasetIndexExt;
use crate::index::frag_reuse::OpenPurpose;
use crate::index::loading::{DatasetIndexInternalExt, load_all_indices};
use crate::index::scalar;
use crate::index::segment::segment_has_vector_details;
use crate::{Dataset, Error, Result};

impl Dataset {
    /// Prewarm a current-format IVF index within the session index-cache budget.
    ///
    /// The branch head can reclaim older segments of the same logical index.
    /// A historical view uses free space. Success requires all partitions to
    /// be resident at the end of this call. The ordinary prewarm path remains
    /// available for other index types and cache backends.
    ///
    /// ```no_run
    /// # async fn warm(dataset: &lance::Dataset) -> lance::Result<()> {
    /// dataset.prewarm_index_with_cache_budget("embedding_idx").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn prewarm_index_with_cache_budget(&self, name: &str) -> Result<()> {
        if !self.index_cache.supports_scoped_prewarm() {
            return self.prewarm_index(name).await;
        }
        // Discovery uses the same cache as the indices. A cold listing must
        // not evict anything before we know which segments to protect.
        let (indices, protected_segments) = self
            .index_cache
            .without_admission(async {
                let manifest_indices = load_all_indices(self).await?;
                let listing = self.derive_index_listing(manifest_indices.clone()).await?;
                if listing.unverifiable.contains(name) {
                    return Err(Error::io(format!(
                        "cannot prewarm index {name:?}: a segment format could not be verified; retry prewarm"
                    )));
                }
                let visible = listing.indices.iter()
                    .filter(|segment| segment.name == name)
                    .cloned()
                    .collect::<Vec<_>>();
                // Coverage can hide a segment even after successful derivation.
                // The manifest still references it, so it must stay protected.
                let protected = manifest_indices.iter()
                    .filter(|segment| segment.name == name)
                    .map(|segment| InternalCacheKey::from_bytes(*segment.uuid.as_bytes()))
                    .collect();
                Ok((visible, protected))
            })
            .await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={name}")));
        }
        if !indices.iter().all(segment_has_vector_details) {
            return self.prewarm_index(name).await;
        }
        let version = self.version_id();
        let policy = CachePrewarmPolicy {
            collection: self.index_cache.index_collection(name),
            reclaim_before: self.prewarm_at_verified_head().await?.then_some(version),
            protected_segments,
        };
        let needs_legacy_fallback = Box::pin(self.index_cache.with_scoped_prewarm(policy, async {
            let mut opened = Vec::with_capacity(indices.len());
            for segment in &indices {
                opened.push(
                    self.open_vector_index_from_metadata_with_purpose(
                        name,
                        segment,
                        &NoOpMetricsCollector,
                        OpenPurpose::Query,
                    )
                    .await?,
                );
            }
            if opened.iter().all(|index| !index.supports_scoped_prewarm()) {
                return Ok(true);
            }
            if opened.iter().any(|index| !index.supports_scoped_prewarm()) {
                return Err(Error::not_supported(format!(
                    "index {name:?} at version {version} mixes scoped prewarm and legacy segments"
                )));
            }
            for index in &opened {
                index.prewarm().await?;
            }
            for index in &opened {
                index.verify_prewarm_residency().await?;
            }
            Ok(false)
        }))
        .await?;
        if needs_legacy_fallback {
            self.prewarm_index(name).await?;
        }
        Ok(())
    }

    async fn prewarm_at_verified_head(&self) -> Result<bool> {
        match self.table_view().ensure_branch_exists().await {
            Ok(()) => {}
            Err(Error::RefNotFound { .. } | Error::NotFound { .. }) => return Ok(false),
            Err(error) => return Err(error),
        }
        let state = &self.table_state;
        let latest = match state
            .commit_handler
            .resolve_latest_location(&state.base, &state.object_store)
            .await
        {
            Ok(latest) => latest,
            Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => return Ok(false),
            Err(error) => return Err(error),
        };
        let pinned = self.manifest_location();
        // A version number can be reused when the table is recreated at this URI.
        let same_generation = match (&pinned.identity, &latest.identity) {
            (Some(pinned), Some(latest)) => pinned == latest,
            _ => pinned
                .e_tag
                .as_ref()
                .is_some_and(|tag| latest.e_tag.as_ref() == Some(tag)),
        };
        Ok(latest.version == self.version_id() && latest.path == pinned.path && same_generation)
    }
}

async fn prewarm_opened_index(
    index: Arc<dyn Index>,
    options: Option<&PrewarmOptions>,
) -> Result<()> {
    match options {
        None => index.prewarm().await,
        Some(PrewarmOptions::Fts(fts_options)) => {
            let inverted = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "FTS prewarm options are only supported for inverted indices, got {:?}",
                        index.index_type()
                    ))
                })?;
            inverted.prewarm_with_options(fts_options).await
        }
        Some(_) => Err(Error::not_supported(
            "unsupported prewarm options for this lance version".to_owned(),
        )),
    }
}

struct SegmentPrewarmResult {
    partition_count: usize,
}

struct OpenedSegmentPrewarmResult {
    index_uuid: Uuid,
    index: Arc<dyn Index>,
    partition_count: usize,
}

async fn prewarm_opened_index_result(
    index: Arc<dyn Index>,
    options: &PrewarmOptions,
) -> Result<SegmentPrewarmResult> {
    match options {
        PrewarmOptions::Fts(fts_options) => {
            let inverted = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "FTS prewarm options are only supported for inverted indices, got {:?}",
                        index.index_type()
                    ))
                })?;
            inverted.prewarm_with_options_result(fts_options).await?;
            Ok(SegmentPrewarmResult {
                partition_count: inverted.partition_count(),
            })
        }
        _ => Err(Error::not_supported(
            "unsupported prewarm options for this lance version".to_owned(),
        )),
    }
}

async fn aggregate_fts_prewarm_results(
    dataset: &Dataset,
    segment_results: Vec<OpenedSegmentPrewarmResult>,
    options: Option<&PrewarmOptions>,
) -> Result<FtsPrewarmResult> {
    let partition_count = segment_results
        .iter()
        .map(|result| result.partition_count)
        .sum();
    let mut failing_segments = Vec::new();
    let mut failing_partitions = Vec::new();
    let mut best_effort = false;

    for segment in segment_results {
        let segment_id = segment.index_uuid.to_string();
        let prewarmed_inverted = segment.index.as_any().downcast_ref::<InvertedIndex>();
        let Some(fts_options) = (match options {
            Some(PrewarmOptions::Fts(fts_options)) => Some(fts_options.clone()),
            None if prewarmed_inverted.is_some() => Some(FtsPrewarmOptions::default()),
            _ => None,
        }) else {
            continue;
        };
        best_effort |= fts_options.mode.is_best_effort();

        let cached_index =
            scalar::cached_scalar_index_container(dataset, &segment.index_uuid).await;
        let cached_inverted = cached_index
            .as_ref()
            .and_then(|index| index.as_any().downcast_ref::<InvertedIndex>());
        let container_resident = cached_inverted.is_some();
        let container_matches_prewarmed = match (prewarmed_inverted, cached_inverted) {
            (Some(prewarmed), Some(cached)) => prewarmed.shares_prewarm_state(cached),
            _ => false,
        };

        if !container_resident || !container_matches_prewarmed {
            failing_segments.push(FtsPrewarmSegmentStatus {
                segment_id: segment_id.clone(),
                scalar_index_container_resident: container_resident,
                scalar_index_container_matches_prewarmed: container_matches_prewarmed,
            });
        }

        if let Some(cached_inverted) = cached_inverted
            && let Some(mut diagnostics) = cached_inverted
                .prewarm_residency_result(fts_options.with_position)
                .await
                .diagnostics
        {
            failing_segments.append(&mut diagnostics.failing_segments);
            failing_partitions.extend(diagnostics.failing_partitions.drain(..).map(
                |mut partition| {
                    partition.segment_id = Some(segment_id.clone());
                    partition
                },
            ));
        }
    }

    let diagnostics = FtsPrewarmDiagnostics {
        partition_count,
        failing_segments,
        failing_partitions,
    };

    if diagnostics.fully_resident() {
        Ok(FtsPrewarmResult::fully_resident())
    } else if best_effort {
        Ok(FtsPrewarmResult::partial(diagnostics))
    } else {
        Err(Error::internal(diagnostics.to_string()))
    }
}

fn total_index_segment_size_bytes(indices: &[IndexMetadata]) -> Option<u64> {
    let mut total = 0u64;
    for index_meta in indices {
        total += index_meta.total_size_bytes()?;
    }
    Some(total)
}

fn cache_size_delta(after: usize, before: usize) -> i64 {
    after.saturating_sub(before) as i64 - before.saturating_sub(after) as i64
}

fn prewarm_options_fields(options: Option<&PrewarmOptions>) -> (&'static str, bool) {
    match options {
        None => ("default", false),
        Some(PrewarmOptions::Fts(fts_options)) => ("fts", fts_options.with_position),
        Some(_) => ("unsupported", false),
    }
}

pub(super) async fn prewarm_index_segments_by_metadata(
    dataset: &Dataset,
    name: &str,
    indices: Vec<IndexMetadata>,
    options: Option<&PrewarmOptions>,
    available_segment_count: usize,
    requested_segment_count: Option<usize>,
) -> Result<FtsPrewarmResult> {
    let request_started = Instant::now();
    let selected_segment_count = indices.len();
    let selected_size_bytes = total_index_segment_size_bytes(&indices);
    let (prewarm_options, fts_with_position) = prewarm_options_fields(options);
    let cache_stats_before = dataset.session.index_cache_stats().await;
    info!(
        index_name = name,
        selected_segment_count,
        available_segment_count,
        requested_segment_count = requested_segment_count.unwrap_or(0),
        segment_filter = requested_segment_count.is_some(),
        selected_size_bytes = selected_size_bytes.unwrap_or(0),
        selected_size_bytes_known = selected_size_bytes.is_some(),
        index_cache_entries_before = cache_stats_before.num_entries,
        index_cache_size_bytes_before = cache_stats_before.size_bytes,
        prewarm_options,
        fts_with_position,
        "prewarm index segments started"
    );

    let result = futures::future::try_join_all(indices.into_iter().map(|index_meta| async move {
        let index_uuid = index_meta.uuid;
        let size_bytes = index_meta.total_size_bytes();
        let fragment_count = index_meta
            .fragment_bitmap
            .as_ref()
            .map(|bitmap| bitmap.len());
        let segment_started = Instant::now();
        info!(
            index_name = name,
            %index_uuid,
            index_version = index_meta.index_version,
            dataset_version = index_meta.dataset_version,
            fragment_count = fragment_count.unwrap_or(0),
            fragment_count_known = fragment_count.is_some(),
            size_bytes = size_bytes.unwrap_or(0),
            size_bytes_known = size_bytes.is_some(),
            "prewarm index segment started"
        );

        let index = match dataset
            .open_generic_index(name, &index_uuid, &NoOpMetricsCollector)
            .await
        {
            Ok(index) => {
                info!(
                    index_name = name,
                    %index_uuid,
                    index_type = ?index.index_type(),
                    elapsed_ms = segment_started.elapsed().as_millis() as u64,
                    "opened index segment for prewarm"
                );
                index
            }
            Err(err) => {
                warn!(
                    index_name = name,
                    %index_uuid,
                    error = %err,
                    elapsed_ms = segment_started.elapsed().as_millis() as u64,
                    "failed to open index segment for prewarm"
                );
                return Err(err);
            }
        };

        let segment_result = match options {
            Some(options) => match prewarm_opened_index_result(index.clone(), options).await {
                Ok(result) => result,
                Err(err) => {
                    warn!(
                        index_name = name,
                        %index_uuid,
                        error = %err,
                        elapsed_ms = segment_started.elapsed().as_millis() as u64,
                        "prewarm index segment failed"
                    );
                    return Err(err);
                }
            },
            None => {
                if let Err(err) = prewarm_opened_index(index.clone(), None).await {
                    warn!(
                        index_name = name,
                        %index_uuid,
                        error = %err,
                        elapsed_ms = segment_started.elapsed().as_millis() as u64,
                        "prewarm index segment failed"
                    );
                    return Err(err);
                }
                SegmentPrewarmResult { partition_count: 0 }
            }
        };

        info!(
            index_name = name,
            %index_uuid,
            elapsed_ms = segment_started.elapsed().as_millis() as u64,
            "prewarm index segment finished"
        );
        Ok(OpenedSegmentPrewarmResult {
            index_uuid,
            index,
            partition_count: segment_result.partition_count,
        })
    }))
    .await;

    match result {
        Ok(segment_results) => {
            let cache_stats_after = dataset.session.index_cache_stats().await;
            let result = aggregate_fts_prewarm_results(dataset, segment_results, options).await?;
            info!(
                index_name = name,
                selected_segment_count,
                fts_fully_resident = result.fully_resident,
                index_cache_entries_after = cache_stats_after.num_entries,
                index_cache_entries_delta = cache_size_delta(
                    cache_stats_after.num_entries,
                    cache_stats_before.num_entries,
                ),
                index_cache_size_bytes_after = cache_stats_after.size_bytes,
                index_cache_size_bytes_delta =
                    cache_size_delta(cache_stats_after.size_bytes, cache_stats_before.size_bytes,),
                elapsed_ms = request_started.elapsed().as_millis() as u64,
                "prewarm index segments finished"
            );
            Ok(result)
        }
        Err(err) => {
            let cache_stats_after = dataset.session.index_cache_stats().await;
            warn!(
                index_name = name,
                selected_segment_count,
                index_cache_entries_after = cache_stats_after.num_entries,
                index_cache_entries_delta = cache_size_delta(
                    cache_stats_after.num_entries,
                    cache_stats_before.num_entries,
                ),
                index_cache_size_bytes_after = cache_stats_after.size_bytes,
                index_cache_size_bytes_delta = cache_size_delta(
                    cache_stats_after.size_bytes,
                    cache_stats_before.size_bytes,
                ),
                error = %err,
                elapsed_ms = request_started.elapsed().as_millis() as u64,
                "prewarm index segments failed"
            );
            Err(err)
        }
    }
}

pub(super) fn filter_index_segments_by_ids(
    name: &str,
    indices: Vec<IndexMetadata>,
    segment_ids: &[Uuid],
) -> Result<Vec<IndexMetadata>> {
    if segment_ids.is_empty() {
        return Ok(Vec::new());
    }

    let requested = segment_ids.iter().copied().collect::<HashSet<_>>();
    let mut matched = HashSet::new();
    let filtered = indices
        .into_iter()
        .filter(|index_meta| {
            if requested.contains(&index_meta.uuid) {
                matched.insert(index_meta.uuid);
                true
            } else {
                false
            }
        })
        .collect::<Vec<_>>();

    if matched.len() != requested.len() {
        let mut missing = requested
            .difference(&matched)
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        missing.sort();
        return Err(Error::index_not_found(format!(
            "name={}, segment_ids=[{}]",
            name,
            missing.join(", ")
        )));
    }

    Ok(filtered)
}

#[cfg(test)]
mod tests;
