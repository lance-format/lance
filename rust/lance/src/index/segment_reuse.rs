// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Consolidation of committed index segments by reusing their stored data.
//!
//! A committed segment whose coverage the current manifest still vouches for
//! can be copied into a new segment without reading table data and without
//! consulting the manifest it was built at. Everything here is validated
//! against the current manifest: a source must still be live with exactly the
//! coverage the caller selected it with, and nothing a fragment reuse history
//! would remap may be involved. Freshly built segments can join the output;
//! only their fragments go through the historical staleness check.

use std::collections::HashSet;

use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::scalar::DEFAULT_MATERIALIZATION_BUDGET_BYTES;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::zonemap::{ZoneMapIndex, estimate_zonemap_merge_bytes};
use lance_table::format::{IndexMetadata, list_index_files_with_sizes};
use lance_table::system_index::frag_reuse::metadata::uses_tagged_fri;
use lance_table::system_index::frag_reuse::retention::{
    is_index_remap_caught_up, reuse_chain_frag_bitmap,
};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::api::IntoIndexSegment;
use crate::index::frag_reuse::load_frag_reuse_index_details;
use crate::index::frag_reuse_with_stable_row_ids::has_frag_reuse_with_stable_row_ids;
use crate::index::loading::load_all_indices;
use crate::index::provenance::prune_stale_segment_coverage;
use crate::index::segment::{
    segment_has_zonemap_details, validate_segment_index_details, validate_segment_metadata,
    validate_segment_params_compatible,
};
use crate::{Dataset, Error, Result};

/// Prefix of the message of every error raised because a reuse source, or a
/// fresh fragment, no longer matches the current manifest. Such a reuse can
/// be re-planned from the current manifest.
pub const STALE_REUSE_SOURCE_ERROR: &str = "stale index segment reuse source";

/// Prefix of the message of every error raised because the sources cannot be
/// reused by copying their stored data. Such sources need a rebuild from
/// table data instead.
pub const UNSUPPORTED_REUSE_SOURCE_ERROR: &str = "unsupported index segment reuse source";

/// Memory a reuse merge is admitted to use for decoded sources, output and
/// temporaries.
pub const REUSE_MERGE_BUDGET_BYTES: u64 = DEFAULT_MATERIALIZATION_BUDGET_BYTES;

/// A committed segment selected as a reuse source.
#[derive(Debug, Clone, PartialEq)]
pub struct CommittedSegmentSource {
    /// The segment's UUID in the manifest it was selected from.
    pub uuid: Uuid,
    /// The segment's effective coverage (its stored coverage restricted to
    /// live fragments) when it was selected. The reuse is refused unless the
    /// current manifest reports exactly this coverage.
    pub expected_fragments: RoaringBitmap,
    /// The segment's `dataset_version` when it was selected. The reuse is
    /// refused unless the current manifest reports the same version, so a
    /// source rewritten in place since its selection is not reused.
    pub expected_dataset_version: u64,
}

/// One output of a reuse, as published by
/// [`DatasetIndexExt::commit_reused_index_segments`](crate::index::DatasetIndexExt::commit_reused_index_segments).
#[derive(Debug, Clone)]
pub struct ReusedSegment {
    /// The uncommitted output segment.
    pub segment: IndexMetadata,
    /// The committed segments it replaces.
    pub sources: Vec<CommittedSegmentSource>,
    /// Fragments the output covers through freshly built segments.
    pub fresh_fragments: RoaringBitmap,
    /// Dataset version the fresh segments were built from; required when
    /// `fresh_fragments` is not empty.
    pub fresh_read_version: Option<u64>,
}

fn stale(message: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!("{STALE_REUSE_SOURCE_ERROR}: {message}"))
}

fn unsupported(message: impl std::fmt::Display) -> Error {
    Error::not_supported(format!("{UNSUPPORTED_REUSE_SOURCE_ERROR}: {message}"))
}

fn index_type_name(segment: &IndexMetadata) -> String {
    segment
        .index_details
        .as_ref()
        .map(|details| {
            details
                .type_url
                .rsplit(['.', '/'])
                .next()
                .unwrap_or(&details.type_url)
                .to_string()
        })
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn require_zonemap(segment: &IndexMetadata) -> Result<()> {
    if segment_has_zonemap_details(segment) {
        return Ok(());
    }
    Err(Error::not_supported(format!(
        "index segment reuse supports only ZoneMap segments; segment {} is {}",
        segment.uuid,
        index_type_name(segment)
    )))
}

/// Resolves the declared sources to their live entries, each holding exactly
/// its expected coverage, all of one index.
fn resolve_committed_sources(
    dataset: &Dataset,
    indices: &[IndexMetadata],
    sources: &[CommittedSegmentSource],
    index_name: Option<&str>,
) -> Result<Vec<IndexMetadata>> {
    let mut seen = HashSet::with_capacity(sources.len());
    let mut resolved: Vec<IndexMetadata> = Vec::with_capacity(sources.len());
    for source in sources {
        if !seen.insert(source.uuid) {
            return Err(Error::invalid_input(format!(
                "index segment {} is declared as a reuse source more than once",
                source.uuid
            )));
        }
        let segment = indices
            .iter()
            .find(|index| index.uuid == source.uuid)
            .ok_or_else(|| {
                stale(format!(
                    "segment {} is not in the manifest at version {}",
                    source.uuid, dataset.manifest.version
                ))
            })?;
        if let Some(index_name) = index_name
            && segment.name != index_name
        {
            return Err(stale(format!(
                "segment {} belongs to index '{}', not '{}'",
                segment.uuid, segment.name, index_name
            )));
        }
        if let Some(first) = resolved.first()
            && (segment.name != first.name || !segment.declares_same_fields(first))
        {
            return Err(stale(format!(
                "segments {} and {} are not segments of one index",
                first.uuid, segment.uuid
            )));
        }
        require_zonemap(segment)?;
        let effective = segment
            .effective_fragment_bitmap(dataset.fragment_bitmap.as_ref())
            .ok_or_else(|| stale(format!("segment {} has no fragment coverage", segment.uuid)))?;
        if effective != source.expected_fragments {
            return Err(stale(format!(
                "segment {} covers fragments {:?} at version {}, expected {:?}",
                segment.uuid,
                effective.iter().collect::<Vec<_>>(),
                dataset.manifest.version,
                source.expected_fragments.iter().collect::<Vec<_>>()
            )));
        }
        if segment.dataset_version != source.expected_dataset_version {
            return Err(stale(format!(
                "segment {} is stamped with dataset version {} at version {}, expected {}",
                segment.uuid,
                segment.dataset_version,
                dataset.manifest.version,
                source.expected_dataset_version
            )));
        }
        resolved.push(segment.clone());
    }
    Ok(resolved)
}

/// Refuses segments whose stored row addresses a fragment reuse history may
/// still have to translate. A tagged or stable-row-id history is refused
/// outright; under a legacy history every segment must be caught up with
/// every retained version, the rule the history trim uses.
async fn check_reuse_history(
    dataset: &Dataset,
    indices: &[IndexMetadata],
    segments: &[IndexMetadata],
) -> Result<()> {
    let fri = indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME);
    if uses_tagged_fri(&dataset.manifest, fri)
        || has_frag_reuse_with_stable_row_ids(&dataset.manifest, indices)
    {
        return Err(unsupported(
            "the table has a tagged or stable-row-id fragment reuse history",
        ));
    }
    let Some(fri) = fri else {
        return Ok(());
    };
    let details = load_frag_reuse_index_details(dataset, fri).await?;
    let chain = reuse_chain_frag_bitmap(&details.versions);
    for segment in segments {
        for version in &details.versions {
            if !matches!(is_index_remap_caught_up(version, segment, &chain), Ok(true)) {
                return Err(unsupported(format!(
                    "segment {} is not caught up with the retained fragment reuse version \
                     at dataset version {}",
                    segment.uuid, version.dataset_version
                )));
            }
        }
    }
    Ok(())
}

/// Admission estimate of the memory the merge needs: decoded sources, output
/// and temporaries, null positions included, bounded from each source's file
/// metadata before anything is decoded. A source whose metadata gives no
/// safe bound refuses the merge. This is an estimate made before loading, not
/// allocator accounting.
async fn estimate_merge_bytes(dataset: &Dataset, segments: &[IndexMetadata]) -> Result<u64> {
    let mut inputs = Vec::with_capacity(segments.len());
    for segment in segments {
        let store = LanceIndexStore::from_dataset_for_existing(dataset, segment).await?;
        let input = ZoneMapIndex::merge_input_from_metadata(&store)
            .await
            .map_err(|error| match error {
                Error::NotSupported { .. } => unsupported(format!(
                    "segment {}: no safe memory estimate: {error}",
                    segment.uuid
                )),
                other => other,
            })?;
        inputs.push(input);
    }
    Ok(estimate_zonemap_merge_bytes(&inputs))
}

/// [`DatasetIndexExt::merge_committed_index_segments`](crate::index::DatasetIndexExt::merge_committed_index_segments)
/// with an explicit memory budget.
pub async fn merge_committed_index_segments_with_budget(
    dataset: &Dataset,
    sources: &[CommittedSegmentSource],
    fresh: Vec<IndexMetadata>,
    budget_bytes: u64,
) -> Result<IndexMetadata> {
    if sources.is_empty() && fresh.is_empty() {
        return Err(Error::invalid_input(
            "merge_committed_index_segments requires at least one source or fresh segment",
        ));
    }
    let indices = load_all_indices(dataset).await?;
    let committed = resolve_committed_sources(dataset, &indices, sources, None)?;

    let mut coverages: Vec<RoaringBitmap> = sources
        .iter()
        .map(|source| source.expected_fragments.clone())
        .collect();
    for segment in &fresh {
        require_zonemap(segment)?;
        if indices.iter().any(|index| index.uuid == segment.uuid) {
            return Err(Error::invalid_input(format!(
                "fresh segment {} is already committed",
                segment.uuid
            )));
        }
        if segment.dataset_version > dataset.manifest.version {
            return Err(Error::invalid_input(format!(
                "fresh segment {} was built at future dataset version {} (current version {})",
                segment.uuid, segment.dataset_version, dataset.manifest.version
            )));
        }
        let reference = committed.first().or(fresh.first()).expect("non-empty");
        if segment.name != reference.name || !segment.declares_same_fields(reference) {
            return Err(Error::invalid_input(format!(
                "fresh segment {} does not belong to the index of segment {}",
                segment.uuid, reference.uuid
            )));
        }
        let coverage = segment.fragment_bitmap.clone().ok_or_else(|| {
            Error::invalid_input(format!(
                "fresh segment {} is missing fragment coverage",
                segment.uuid
            ))
        })?;
        coverages.push(coverage);
    }
    let mut claimed = RoaringBitmap::new();
    for coverage in &coverages {
        if !claimed.is_disjoint(coverage) {
            return Err(Error::invalid_input(format!(
                "reuse sources overlap on fragments {:?}",
                (&claimed & coverage).iter().collect::<Vec<_>>()
            )));
        }
        claimed |= coverage;
    }

    let all = committed
        .iter()
        .chain(fresh.iter())
        .cloned()
        .collect::<Vec<_>>();
    check_reuse_history(dataset, &indices, &all).await?;
    validate_segment_params_compatible(&[], &all).map_err(unsupported)?;

    let estimate = estimate_merge_bytes(dataset, &all).await?;
    if estimate > budget_bytes {
        return Err(unsupported(format!(
            "merging these segments is estimated to need {estimate} bytes, over the \
             {budget_bytes} byte budget"
        )));
    }

    let output_version = committed
        .iter()
        .map(|segment| segment.dataset_version)
        .min()
        .or_else(|| fresh.iter().map(|segment| segment.dataset_version).min())
        .expect("non-empty");
    let pairs = all.into_iter().zip(coverages).collect::<Vec<_>>();
    let mut merged =
        crate::index::scalar::zonemap::merge_segments_by_coverage(dataset, pairs, None, true)
            .await
            .map_err(|error| match error {
                Error::NotSupported { .. } | Error::InvalidInput { .. } => unsupported(error),
                other => other,
            })?;
    // The oldest source version keeps overlay masking conservative.
    merged.dataset_version = output_version;
    Ok(merged)
}

/// [`DatasetIndexExt::commit_reused_index_segments`](crate::index::DatasetIndexExt::commit_reused_index_segments).
pub async fn commit_reused_index_segments(
    dataset: &mut Dataset,
    index_name: &str,
    column: &str,
    outputs: Vec<ReusedSegment>,
) -> Result<()> {
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::index(format!(
            "CreateIndex: column '{column}' does not exist"
        )));
    };
    let field_id = field.id;
    if outputs.is_empty() {
        return Err(Error::invalid_input(
            "commit_reused_index_segments requires at least one output",
        ));
    }
    let indices = load_all_indices(dataset).await?;

    // Every declared source, resolved against this manifest.
    let declared = outputs
        .iter()
        .flat_map(|output| output.sources.iter().cloned())
        .collect::<Vec<_>>();
    let removed_indices =
        resolve_committed_sources(dataset, &indices, &declared, Some(index_name))?;

    let mut published = RoaringBitmap::new();
    let mut fresh_checks = Vec::new();
    for output in &outputs {
        let segment = &output.segment;
        require_zonemap(segment)?;
        if segment.keyed_field() != Some(field_id) {
            return Err(Error::invalid_input(format!(
                "reused segment {} was built for fields {:?}, expected keyed field [{}]",
                segment.uuid, segment.fields, field_id
            )));
        }
        if let Some(source) = removed_indices.first()
            && !segment.declares_same_fields(source)
        {
            return Err(Error::invalid_input(format!(
                "reused segment {} declares different fields than its sources",
                segment.uuid
            )));
        }
        if indices.iter().any(|index| index.uuid == segment.uuid) {
            return Err(Error::invalid_input(format!(
                "reused segment {} is already committed",
                segment.uuid
            )));
        }
        if segment.dataset_version > dataset.manifest.version {
            return Err(Error::invalid_input(format!(
                "reused segment {} was built at future dataset version {} (current version {})",
                segment.uuid, segment.dataset_version, dataset.manifest.version
            )));
        }

        let mut expected = RoaringBitmap::new();
        for source in &output.sources {
            if !expected.is_disjoint(&source.expected_fragments) {
                return Err(Error::invalid_input(format!(
                    "reused segment {} has sources with overlapping coverage",
                    segment.uuid
                )));
            }
            expected |= &source.expected_fragments;
        }
        if !expected.is_disjoint(&output.fresh_fragments) {
            return Err(Error::invalid_input(format!(
                "reused segment {} declares fresh fragments its sources already cover",
                segment.uuid
            )));
        }
        expected |= &output.fresh_fragments;
        // The output keeps the oldest stamp of what it reuses, so overlays
        // newer than any source stay masked on all its fragments.
        let oldest = output
            .sources
            .iter()
            .map(|source| source.expected_dataset_version)
            .chain(
                output
                    .fresh_read_version
                    .filter(|_| !output.fresh_fragments.is_empty()),
            )
            .min();
        if let Some(oldest) = oldest
            && segment.dataset_version > oldest
        {
            return Err(Error::invalid_input(format!(
                "reused segment {} is stamped with dataset version {}, newer than the oldest \
                 version {} of its sources and fresh fragments",
                segment.uuid, segment.dataset_version, oldest
            )));
        }
        if expected.is_empty() {
            return Err(Error::invalid_input(format!(
                "reused segment {} declares no sources and no fresh fragments",
                segment.uuid
            )));
        }
        let coverage = segment.fragment_bitmap.as_ref().ok_or_else(|| {
            Error::invalid_input(format!(
                "reused segment {} is missing fragment coverage",
                segment.uuid
            ))
        })?;
        if *coverage != expected {
            return Err(Error::invalid_input(format!(
                "reused segment {} covers fragments {:?}, but its sources and fresh fragments \
                 cover {:?}",
                segment.uuid,
                coverage.iter().collect::<Vec<_>>(),
                expected.iter().collect::<Vec<_>>()
            )));
        }
        if !published.is_disjoint(coverage) {
            return Err(Error::invalid_input(format!(
                "reused segments overlap on fragments {:?}",
                (&published & coverage).iter().collect::<Vec<_>>()
            )));
        }
        published |= coverage;

        if !output.fresh_fragments.is_empty() {
            let read_version = output.fresh_read_version.ok_or_else(|| {
                Error::invalid_input(format!(
                    "reused segment {} has fresh fragments but no fresh_read_version",
                    segment.uuid
                ))
            })?;
            if read_version > dataset.manifest.version {
                return Err(Error::invalid_input(format!(
                    "reused segment {} has fresh_read_version {} past the current version {}",
                    segment.uuid, read_version, dataset.manifest.version
                )));
            }
            // Only the fresh part is held to its build version; the
            // committed part was validated against this manifest above.
            fresh_checks.push(IndexMetadata {
                name: index_name.to_string(),
                fragment_bitmap: Some(output.fresh_fragments.clone()),
                dataset_version: read_version,
                ..segment.clone()
            });
        }
    }

    check_reuse_history(dataset, &indices, &removed_indices).await?;
    check_reuse_history(dataset, &indices, &fresh_checks).await?;

    if !fresh_checks.is_empty() {
        let mut checked = fresh_checks
            .iter()
            .cloned()
            .map(IntoIndexSegment::into_index_segment)
            .collect::<Result<Vec<_>>>()?;
        // A cleaned-up build version clears the coverage instead of failing,
        // so any shrinkage, including that one, refuses the publication.
        prune_stale_segment_coverage(dataset, &mut checked, true, &RoaringBitmap::new(), false)
            .await?;
        for (requested, checked) in fresh_checks.iter().zip(&checked) {
            let dropped =
                requested.fragment_bitmap.as_ref().expect("set above") - checked.fragment_bitmap();
            if !dropped.is_empty() {
                return Err(stale(format!(
                    "fresh fragments {:?} of reused segment {} cannot be shown unchanged since \
                     dataset version {} (changed, or that version was cleaned up)",
                    dropped.iter().collect::<Vec<_>>(),
                    requested.uuid,
                    requested.dataset_version,
                )));
            }
        }
    }

    // Exactly the declared sources are removed; any other segment of the
    // index that would overlap an output means the plan is out of date.
    let removed_uuids = removed_indices
        .iter()
        .map(|index| index.uuid)
        .collect::<HashSet<_>>();
    let retained = indices
        .iter()
        .filter(|index| index.name == index_name && !removed_uuids.contains(&index.uuid))
        .cloned()
        .collect::<Vec<_>>();
    for index in &retained {
        let overlaps = index
            .effective_fragment_bitmap(dataset.fragment_bitmap.as_ref())
            .is_none_or(|coverage| !coverage.is_disjoint(&published));
        if overlaps {
            return Err(stale(format!(
                "segment {} of index '{}' is not a declared source but overlaps the reused \
                 coverage {:?}",
                index.uuid,
                index_name,
                published.iter().collect::<Vec<_>>()
            )));
        }
        if !segment_has_zonemap_details(index) || !index.declares_same_fields(&outputs[0].segment) {
            return Err(Error::invalid_input(format!(
                "retained segment {} of index '{}' is not compatible with the reused segments",
                index.uuid, index_name
            )));
        }
    }

    let mut new_indices = Vec::with_capacity(outputs.len());
    for output in outputs {
        let mut segment = output.segment;
        segment.name = index_name.to_string();
        if segment.files.is_none() {
            let dir = dataset.indice_files_dir(&segment)?;
            segment.files = Some(list_index_files_with_sizes(&dataset.object_store, &dir).await?);
        }
        segment.created_at = Some(chrono::Utc::now());
        new_indices.push(segment);
    }
    validate_segment_metadata(index_name, &new_indices)?;
    validate_segment_index_details(index_name, &new_indices)?;
    validate_segment_params_compatible(&retained, &new_indices)?;

    // Everything above was validated against this manifest, so the commit
    // may only land directly on it. Any commit in between (a compaction that
    // rewrote or deferred-remapped a source's fragments, a replaced source,
    // or an unrelated append) fails it with a retryable conflict instead of
    // rebasing; reloading and calling again revalidates every source.
    let read_version = dataset.manifest.version;
    let transaction = Transaction::new(
        read_version,
        Operation::CreateIndex {
            new_indices,
            removed_indices,
        },
        None,
    );
    dataset
        .apply_commit_with_expected_latest_version(
            transaction,
            &Default::default(),
            &Default::default(),
            Some(read_version),
        )
        .await
}

#[cfg(test)]
mod tests;
