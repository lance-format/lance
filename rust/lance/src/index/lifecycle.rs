// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset-level index lifecycle and publication.
//!
//! This module owns index creation/drop entry points, segment merge and commit,
//! loading the usable listing, description dispatch, statistics dispatch, and
//! optimize entry. Low-level source freshness lives in `provenance`; opening
//! and cache behavior lives in `loading`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::{FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_index::{
    FtsPrewarmResult, IndexCriteria, IndexDescription, IndexParams, IndexType, PrewarmOptions,
    frag_reuse::FRAG_REUSE_INDEX_NAME, is_system_index, mem_wal::MEM_WAL_INDEX_NAME,
    metrics::NoOpMetricsCollector, optimize::OptimizeOptions,
};
use lance_table::format::{IndexMetadata, list_index_files_with_sizes};
use lance_table::io::commit::CommitConfig;
use roaring::RoaringBitmap;
use tracing::instrument;
use uuid::Uuid;

use crate::dataset::ManifestWriteConfig;
use crate::dataset::transaction::{Operation, ReadVersionState, Transaction};
use crate::index::api::{DatasetIndexExt, IndexSegment, IndexSegmentStatistics, IntoIndexSegment};
use crate::index::append;
use crate::index::create::CreateIndexBuilder;
use crate::index::frag_reuse;
use crate::index::frag_reuse::{CoverageExclusion, MissingCoverageReason};
use crate::index::frag_reuse_reader;
use crate::index::frag_reuse_with_stable_row_ids::{
    has_frag_reuse_with_stable_row_ids, is_hidden_by_frag_reuse,
};
use crate::index::loading::{DatasetIndexInternalExt, load_all_indices, load_all_indices_by_name};
use crate::index::optimize;
use crate::index::prewarm::{filter_index_segments_by_ids, prewarm_index_segments_by_metadata};
use crate::index::provenance::{
    prune_stale_segment_coverage, remap_merged_segment_coverage, replay_staged_segments,
};
use crate::index::scalar::{self, IndexDetails, index_matches_criteria};
use crate::index::segment::{
    index_is_usable, retain_committed_inverted_files, segment_has_bitmap_details,
    segment_has_bloomfilter_details, segment_has_btree_details, segment_has_fmindex_details,
    segment_has_inverted_details, segment_has_label_list_details, segment_has_minhashlsh_details,
    segment_has_ngram_details, segment_has_rtree_details, segment_has_vector_details,
    segment_has_zonemap_details, validate_segment_index_details, validate_segment_metadata,
    validate_segment_params_compatible,
};
use crate::index::statistics::{
    IndexDescriptionImpl, collect_index_segment_statistics, index_statistics_frag_reuse,
    index_statistics_mem_wal, index_statistics_scalar, serialize_regular_index_statistics,
};
use crate::index::vector;
use crate::index::vector::details::needs_vector_details_inference;
use crate::io::commit::{ConcurrentVersion, check_versions_since_read, commit_attempt_budget};
use crate::session::index_caches::{
    DerivedIndexListingKey, QueryIndexListing, QueryIndexListingKey,
};
use crate::{Dataset, Error, Result};

pub(super) async fn build_index_metadata_from_segments(
    dataset: &Dataset,
    index_name: &str,
    field_id: i32,
    mut segments: Vec<IndexSegment>,
) -> Result<Vec<IndexMetadata>> {
    if segments.is_empty() {
        return Err(Error::invalid_input(
            "CreateIndex: at least one index segment is required".to_string(),
        ));
    }

    let mut seen_segment_ids = HashSet::with_capacity(segments.len());
    let mut covered_fragments = RoaringBitmap::new();
    // Per-segment validation does not establish one ordered declaration for
    // the logical index: both carried columns and plugin dependencies can
    // differ. Keep the same invariant as `IndexDescriptionImpl::try_new` and
    // `merge_existing_index_segments`.
    let expected = &segments[0];
    for segment in &segments {
        if segment.dataset_version() > dataset.manifest.version {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} was built at future dataset version {} (current version {})",
                segment.uuid(),
                segment.dataset_version(),
                dataset.manifest.version
            )));
        }
        // Multi-field statistics are built and maintained by the plugin. Empty
        // coverage keeps them out of the single-column query and remap paths.
        let is_multi_field_plugin = segment.fields().len() > 1
            && segment.covering_fields().is_empty()
            && segment.fragment_bitmap().is_empty()
            && scalar::SCALAR_INDEX_PLUGIN_REGISTRY
                .get_plugin_by_details(segment.index_details())
                .is_ok_and(|plugin| !plugin.supports_optimization());
        if segment.keyed_field() != Some(field_id)
            && !(is_multi_field_plugin && segment.fields().first() == Some(&field_id))
        {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} was built for fields {:?} (carried {:?}), \
                 expected keyed field [{}]",
                segment.uuid(),
                segment.fields(),
                segment.covering_fields(),
                field_id
            )));
        }
        if is_multi_field_plugin {
            let mut seen_fields = HashSet::with_capacity(segment.fields().len());
            for id in segment.fields() {
                if dataset.schema().field_by_id(*id).is_none() {
                    return Err(Error::invalid_input(format!(
                        "CreateIndex: segment {} references field id {} absent from dataset version {}",
                        segment.uuid(),
                        id,
                        dataset.manifest.version
                    )));
                }
                if !seen_fields.insert(*id) {
                    return Err(Error::invalid_input(format!(
                        "CreateIndex: segment {} repeats field id {} in {:?}",
                        segment.uuid(),
                        id,
                        segment.fields()
                    )));
                }
            }
        }
        // Hold the covering declaration to the same rule a committed
        // `IndexMetadata` is held to, rather than re-deriving it.
        IndexMetadata {
            uuid: segment.uuid(),
            fields: segment.fields().to_vec(),
            covering_fields: segment.covering_fields().to_vec(),
            name: index_name.to_string(),
            dataset_version: segment.dataset_version(),
            fragment_bitmap: None,
            index_details: None,
            index_version: segment.index_version(),
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
        .validate_covering_fields()?;
        if !segment.declares_same_fields(expected) {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} declares fields {:?} (carried {:?}) but index \
                 '{}' declares fields {:?} (carried {:?}); every segment of one index \
                 must declare the same columns",
                segment.uuid(),
                segment.fields(),
                segment.covering_fields(),
                index_name,
                expected.fields(),
                expected.covering_fields(),
            )));
        }
        if !seen_segment_ids.insert(segment.uuid()) {
            return Err(Error::invalid_input(format!(
                "CreateIndex: duplicate segment uuid {} for index '{}'",
                segment.uuid(),
                index_name
            )));
        }
        if !covered_fragments.is_disjoint(segment.fragment_bitmap()) {
            return Err(Error::invalid_input(format!(
                "CreateIndex: overlapping fragment coverage in segment set for index '{}'",
                index_name
            )));
        }
        covered_fragments |= segment.fragment_bitmap().clone();
    }

    prune_stale_segment_coverage(dataset, &mut segments, false, &RoaringBitmap::new(), false)
        .await?;

    let new_indices = futures::stream::iter(segments.into_iter().map(|segment| async move {
        let clustering_id = segment.clustering_id();
        let metadata = segment.metadata().clone();
        let (
            uuid,
            fragment_bitmap,
            fields,
            covering_fields,
            index_details,
            index_version,
            dataset_version,
        ) = segment.into_parts();
        let is_inverted_index = index_details.type_url.ends_with("InvertedIndexDetails");
        if is_inverted_index {
            let metadata = IndexMetadata {
                uuid,
                name: index_name.to_string(),
                fields: fields.clone(),
                covering_fields: covering_fields.clone(),
                dataset_version,
                fragment_bitmap: Some(fragment_bitmap.clone()),
                index_details: Some(index_details.clone()),
                index_version,
                created_at: Some(chrono::Utc::now()),
                base_id: None,
                files: None,
                clustering_id,
                metadata: metadata.clone(),
            };
            crate::index::scalar::inverted::finalize_segment_files_if_needed(dataset, &metadata)
                .await?;
        }
        let index_dir = dataset.indices_dir().join(uuid.to_string());
        let mut files = list_index_files_with_sizes(&dataset.object_store, &index_dir).await?;
        if is_inverted_index {
            retain_committed_inverted_files(&mut files);
        }
        Ok::<_, Error>(IndexMetadata {
            uuid,
            name: index_name.to_string(),
            fields,
            covering_fields,
            dataset_version,
            fragment_bitmap: Some(fragment_bitmap),
            index_details: Some(index_details),
            index_version,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: Some(files),
            clustering_id,
            metadata,
        })
    }))
    .buffered(dataset.object_store.io_parallelism())
    .try_collect::<Vec<_>>()
    .await?;

    Ok(new_indices)
}

impl Dataset {
    /// Whether an otherwise empty commit would record a new MemWAL catch-up
    /// position.
    ///
    /// Dry-runs the derivation rather than restating its conditions: a second
    /// copy of "is this index behind" would be one more place to keep in step
    /// with the real rule. A no-work optimize publishes no new segment, so the
    /// index list it would commit is the one already loaded, and the version it
    /// would read is the current one -- which makes the speculative answer the
    /// same one the commit reaches.
    pub(super) fn mem_wal_catch_up_would_advance(&self, indices: &[IndexMetadata]) -> Result<bool> {
        if !indices.iter().any(|index| index.name == MEM_WAL_INDEX_NAME) {
            return Ok(false);
        }
        let catchup_of = |indices: &[IndexMetadata]| -> Result<Option<Vec<_>>> {
            indices
                .iter()
                .find(|index| index.name == MEM_WAL_INDEX_NAME)
                .cloned()
                .map(|index| {
                    crate::index::mem_wal::load_mem_wal_index_details(index)
                        .map(|details| details.index_catchup)
                })
                .transpose()
        };

        let mut speculative = indices.to_vec();
        Transaction::apply_mem_wal_index_coverage(
            &mut speculative,
            &Transaction::logical_index_segments(indices),
            Some(ReadVersionState {
                manifest: &self.manifest,
                indices,
            }),
            self.manifest.version + 1,
        )?;
        Ok(catchup_of(&speculative)? != catchup_of(indices)?)
    }

    /// The tagged reader's derived index listing for this snapshot, with the
    /// flag telling whether a segment is absent from it only because its
    /// format marker could not be read.
    ///
    /// [`DatasetIndexExt::load_indices`] drops the flag once it has decided
    /// whether to cache; a caller that caches a SECOND derivation of its own
    /// (the FRI query plan) needs the same flag for the same reason, so the
    /// decision is made from one source here instead of re-derived from the
    /// plan's contents.
    pub(crate) async fn derived_index_listing(&self) -> Result<DerivedIndexListing> {
        let indices = load_all_indices(self).await?;
        self.derive_index_listing(indices).await
    }

    /// [`Self::derived_index_listing`] for a raw listing the caller already
    /// loaded, so a caller that must inspect the raw listing first does not
    /// load it twice.
    async fn derive_index_listing(
        &self,
        indices: Arc<Vec<IndexMetadata>>,
    ) -> Result<DerivedIndexListing> {
        // Readers apply the fragment reuse index to every index they open.
        let indices = if has_frag_reuse_with_stable_row_ids(&self.manifest, &indices)
            && indices.iter().any(is_hidden_by_frag_reuse)
        {
            Arc::new(
                indices
                    .iter()
                    .filter(|idx| !is_hidden_by_frag_reuse(idx))
                    .cloned()
                    .collect(),
            )
        } else {
            indices
        };
        let tagged = indices
            .iter()
            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            .map(|fri| fri.index_version);
        match tagged {
            // Legacy FRI index version 0 already had its fragment coverage
            // remapped in load_all_indices().
            None | Some(0) => Ok(DerivedIndexListing {
                indices: filter_unusable(indices),
                unverifiable: BTreeSet::new(),
            }),
            Some(1) => {
                let fri = indices
                    .iter()
                    .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
                    .expect("the v1 FRI entry was just located");
                let (derived, unverifiable) =
                    frag_reuse_reader::load_indices(self, fri, &indices).await?;
                Ok(DerivedIndexListing {
                    indices: derived,
                    unverifiable,
                })
            }
            Some(version) => Err(Error::not_supported(format!(
                "FRI index_version {version} is unsupported. Please upgrade to a newer version",
            ))),
        }
    }
}

/// The index listing a snapshot's readers see, with what the caller needs to
/// decide whether to cache it.
pub struct DerivedIndexListing {
    pub indices: Arc<Vec<IndexMetadata>>,
    /// Logical indices with at least one segment whose format marker could not
    /// be read; empty when every segment was verified. Those indices' listings
    /// hold for this call alone.
    pub unverifiable: BTreeSet<String>,
}

/// Drop the segments no reader can open, keeping the input allocation when
/// every segment is usable.
fn filter_unusable(indices: Arc<Vec<IndexMetadata>>) -> Arc<Vec<IndexMetadata>> {
    if indices.iter().all(index_is_usable) {
        return indices;
    }
    Arc::new(
        indices
            .iter()
            .filter(|idx| index_is_usable(idx))
            .cloned()
            .collect(),
    )
}

/// A same-name segment a partial commit on a tagged history may keep whole
/// although none of its stored fragments is live: its stored bitmap is
/// provenance the reader still translates to the live fragments rewrites
/// produced. One whose provenance meets the incoming segments' is not a
/// candidate: kept whole, it would contribute those source rows a second time.
/// Nor is one none of whose stored fragments the history names (see
/// `CompositionPlanner::traceable_fragments`): they were deleted outright, so
/// it is removed like a segment that stores no fragment. That last test needs
/// the mapping, so the caller makes it on the segments passing this one.
fn is_historical_candidate(
    idx: &IndexMetadata,
    live: &RoaringBitmap,
    incoming_fragments: &RoaringBitmap,
) -> bool {
    idx.fragment_bitmap.as_ref().is_some_and(|stored| {
        !stored.is_empty() && stored.is_disjoint(live) && stored.is_disjoint(incoming_fragments)
    })
}

/// A retained segment outlives the commit, so its declaration has to match
/// what is being written: `IndexDescriptionImpl::try_new` requires the same
/// keyed and carried fields across the segments of one logical index. Nothing
/// before this catches a disagreement, because `keyed_fields` is the prefix
/// left after the carried ones -- a segment carrying columns keys on exactly
/// what a plain one keys on, and both pass the keyed-field guard.
fn check_retained_declaration(
    index_name: &str,
    expected: &IndexMetadata,
    idx: &IndexMetadata,
) -> Result<()> {
    if idx.declares_same_fields(expected) {
        return Ok(());
    }
    Err(Error::invalid_input(format!(
        "CreateIndex: incoming segments for '{}' declare fields {:?} and covering_fields {:?}, \
         but retained segment {} declares fields {:?} and covering_fields {:?}; \
         a logical index cannot mix declarations - rebuild every segment in one commit",
        index_name,
        expected.fields,
        expected.covering_fields,
        idx.uuid,
        idx.fields,
        idx.covering_fields
    )))
}

fn covered_by(coverage: &[RoaringBitmap]) -> RoaringBitmap {
    coverage
        .iter()
        .fold(RoaringBitmap::new(), |covered, bitmap| covered | bitmap)
}

/// Which removed segments the incoming segments supply again.
struct Resupply<'a> {
    /// Removed segments whose stored fragments are not supplied again.
    unresupplied: Vec<&'a IndexMetadata>,
    /// At least one removal was supplied again only by the exception for a
    /// type without a batch remapper, which rests on this snapshot's
    /// classification.
    by_exception: bool,
}

/// Decide, per removed segment, whether the incoming segments supply its
/// stored fragments again.
///
/// A removed segment is supplied again when incoming segments taking part in
/// coverage store all its fragments: each destination it contributed to
/// resolves through them instead. As an exception, a removed segment the
/// reader leaves out because its type has no batch remapper derives nothing
/// in this build, so it is also supplied again when incoming segments that
/// take part or are left out for that same reason store all its fragments.
/// The exception never applies to a removed segment that takes part in
/// coverage, nor to any other reason (segment-specific, unreadable marker,
/// uninterpretable history).
///
/// Only the stored fragments `traceable` keeps count: one that is neither live
/// nor named by the history was deleted outright, no reader translates it, so
/// it carries no coverage to supply again.
fn unresupplied_removals<'a>(
    removed: &'a [IndexMetadata],
    incoming: &[IndexMetadata],
    exclusion: impl Fn(&IndexMetadata) -> Option<CoverageExclusion>,
    traceable: impl Fn(&RoaringBitmap) -> RoaringBitmap,
) -> Resupply<'a> {
    let stored_by = |admit: &dyn Fn(Option<CoverageExclusion>) -> bool| {
        incoming
            .iter()
            .filter(|idx| admit(exclusion(idx)))
            .filter_map(|idx| idx.fragment_bitmap.as_ref())
            .fold(RoaringBitmap::new(), |supplied, stored| supplied | stored)
    };
    let by_members = stored_by(&|reason| reason.is_none());
    let by_members_or_no_remapper =
        stored_by(&|reason| matches!(reason, None | Some(CoverageExclusion::NoBatchRemapper)));
    let mut resupply = Resupply {
        unresupplied: Vec::new(),
        by_exception: false,
    };
    for idx in removed {
        let Some(stored) = idx.fragment_bitmap.as_ref() else {
            resupply.unresupplied.push(idx);
            continue;
        };
        let stored = traceable(stored);
        if stored.is_subset(&by_members) {
            continue;
        }
        if exclusion(idx) == Some(CoverageExclusion::NoBatchRemapper)
            && stored.is_subset(&by_members_or_no_remapper)
        {
            resupply.by_exception = true;
            continue;
        }
        resupply.unresupplied.push(idx);
    }
    resupply
}

/// Every decision of a commit that has to land on the version it was decided
/// on lost that version to another writer; `source` is the last loss.
#[derive(Debug)]
struct NotAppliedAtReadVersion {
    index_name: String,
    read_version: u64,
    attempts: u32,
    source: Error,
}

impl std::fmt::Display for NotAppliedAtReadVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CreateIndex: the segments of '{}' this commit keeps and removes were chosen from \
             the coverage derived at the version they were to land on, and another commit \
             landed first each of {} times; the last decision, made at version {}, could not \
             be applied there because a newer version exists; commit the segments again so the \
             choice is made against the latest version: {}",
            self.index_name, self.attempts, self.read_version, self.source
        )
    }
}

impl std::error::Error for NotAppliedAtReadVersion {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Whether a commit that has to land on the version it was decided on decides
/// again on the latest version after its conditional attempt `attempt`
/// (counted from 1) of `budget` ended in `outcome`: only when another writer
/// committed first and an attempt is left. A failed write, an unknown outcome
/// or any other result is returned as it is.
fn decides_again(outcome: &Result<()>, attempt: u32, budget: u32) -> bool {
    attempt < budget && matches!(outcome, Err(error) if ConcurrentVersion::caused(error))
}

/// This build refuses to open a segment a commit would retain, so its
/// compatibility with the incoming segments cannot be checked.
#[derive(Debug)]
struct RetainedSegmentUnopenable {
    index_name: String,
    segment: Uuid,
    source: Error,
}

impl std::fmt::Display for RetainedSegmentUnopenable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CreateIndex: this build cannot open segment {} of index '{}', which this commit \
             would retain, to check that it is compatible with the incoming segments; nothing \
             was committed. Replacing every segment of the index over the current fragments in \
             one commit does not retain it: {}",
            self.segment, self.index_name, self.source
        )
    }
}

impl std::error::Error for RetainedSegmentUnopenable {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// The error for retained segment `segment` of `index_name` failing to open
/// with `error`. When classifying it already failed to read (`read_error`)
/// and the open was refused as not supported, that read failure is what keeps
/// it closed and is returned unchanged; any other refusal by this build stays
/// `NotSupported` and names the segment; anything else is returned unchanged.
fn retained_segment_unopenable(
    index_name: &str,
    segment: Uuid,
    error: Error,
    read_error: Option<Error>,
) -> Error {
    let refused = matches!(error.find_root(), Error::NotSupported { .. });
    if let Some(read_error) = read_error.filter(|_| refused) {
        log::warn!(
            "CreateIndex: nothing was committed for '{index_name}': retained segment {segment} \
             could not be opened to check its compatibility ({error}) because a read failed"
        );
        return read_error;
    }
    if refused {
        return Error::not_supported_source(Box::new(RetainedSegmentUnopenable {
            index_name: index_name.to_owned(),
            segment,
            source: error,
        }));
    }
    log::warn!(
        "CreateIndex: nothing was committed for '{index_name}': retained segment {segment} \
         could not be opened to check its compatibility"
    );
    error
}

/// What a partial commit on a tagged history does with the segments whose
/// fate depends on derived coverage.
struct PartialCommitDecision {
    /// Historical candidates that stay.
    retained: HashSet<Uuid>,
    /// Historical candidates the published composition derives nothing for.
    dropped: Vec<IndexMetadata>,
    /// The decision rests on coverage or classification derived from this
    /// snapshot: a removed segment's traceable provenance is not wholly
    /// supplied again by incoming coverage members, or a historical candidate
    /// was dropped. Such a commit must land on the snapshot it was decided on.
    requires_unchanged_snapshot: bool,
}

/// Whether a refusal over unproven coverage is reported as the failed read:
/// only when every unproven segment (`read_failed`, one per segment, at least
/// one) is unproven because its read failed: after the read recovers the
/// decision is made again.
fn refusal_is_the_read_error(read_failed: impl IntoIterator<Item = bool>) -> bool {
    let mut any = false;
    for failed in read_failed {
        if !failed {
            return false;
        }
        any = true;
    }
    any
}

/// Decide a partial commit on a tagged history against the composition it
/// would publish, and refuse one that would lose coverage.
///
/// Segments are kept or removed whole; a stored bitmap is never narrowed. A
/// historical candidate stays when the published composition (the kept
/// siblings, every candidate and the incoming segments, under the reader's
/// membership rule) still derives coverage for it, so an incoming segment
/// completing a destination together with it keeps it. The commit is then
/// refused when a live fragment the logical index covers now would be left
/// uncovered by the published composition.
///
/// A segment the reader leaves out of coverage (no batch remapper, details
/// it cannot interpret, an unreadable format marker, a segment this build
/// cannot open) makes every sibling's derived coverage unproven. Then every
/// candidate stays, and a removal whose provenance the incoming segments do
/// not wholly supply again is refused: neither is decided from coverage
/// nobody could derive. A removed segment left out because its type has no
/// batch remapper may be supplied again by incoming segments left out for the
/// same reason (see [`unresupplied_removals`]); a commit relying on that lands
/// only on the snapshot it was decided on. The refusal is the failed read
/// itself only when every unproven segment is unproven because a read failed,
/// since after the read recovers the decision is made again; otherwise it is
/// invalid input naming each segment's reason, read failures included.
///
/// A removed segment that stores a bitmap none of whose fragments the history
/// traces serves nothing: it is removed, but neither classified nor counted in
/// any coverage, so it never leaves the decision unproven. Without candidates
/// and other removals nothing is classified, and when no removal stores a
/// fragment the mapping is not opened at all.
#[allow(clippy::too_many_arguments)]
async fn decide_partial_composition(
    dataset: &Dataset,
    history: &mut frag_reuse::TaggedHistory,
    index_name: &str,
    existing: &[IndexMetadata],
    incoming: &[IndexMetadata],
    removed: &[IndexMetadata],
    candidates: &HashSet<Uuid>,
    incoming_fragments: &RoaringBitmap,
) -> Result<PartialCommitDecision> {
    let decided_without_coverage = |requires_unchanged_snapshot| PartialCommitDecision {
        retained: HashSet::new(),
        dropped: Vec::new(),
        requires_unchanged_snapshot,
    };
    // Removing only segments that store no fragment depends on no coverage
    // and needs no mapping; a segment without a stored bitmap is not one.
    let removes_no_provenance = removed.iter().all(|idx| {
        idx.fragment_bitmap
            .as_ref()
            .is_some_and(RoaringBitmap::is_empty)
    });
    if candidates.is_empty() && removes_no_provenance {
        return Ok(decided_without_coverage(false));
    }
    let planner = history.planner(dataset).await?;
    // A removed segment none of whose stored fragments the history traces
    // serves nothing: it takes no part in the coverage decision, so its own
    // exclusion cannot leave the rest unproven. It is still removed.
    let serves_nothing: HashSet<Uuid> = removed
        .iter()
        .filter(|idx| {
            idx.fragment_bitmap
                .as_ref()
                .is_some_and(|stored| planner.traceable_fragments(stored).is_empty())
        })
        .map(|idx| idx.uuid)
        .collect();
    let relevant = |segments: &[IndexMetadata]| -> Vec<IndexMetadata> {
        segments
            .iter()
            .filter(|idx| !serves_nothing.contains(&idx.uuid))
            .cloned()
            .collect()
    };
    let removed = relevant(removed);
    let existing = relevant(existing);
    if candidates.is_empty() && removed.is_empty() {
        return Ok(decided_without_coverage(false));
    }

    let group: Vec<&IndexMetadata> = existing.iter().chain(incoming.iter()).collect();
    planner.classify(dataset, &group).await;
    let Resupply {
        unresupplied,
        by_exception,
    } = unresupplied_removals(
        &removed,
        incoming,
        |idx| planner.coverage_exclusion(idx),
        |stored| planner.traceable_fragments(stored),
    );
    if candidates.is_empty() && unresupplied.is_empty() {
        return Ok(decided_without_coverage(by_exception));
    }
    let unproven: Vec<(Uuid, MissingCoverageReason)> = group
        .iter()
        .filter_map(|idx| planner.exclusion(idx).map(|reason| (idx.uuid, reason)))
        .collect();
    if !unproven.is_empty() {
        if !unresupplied.is_empty() {
            let mut read_errors: HashMap<Uuid, Error> = unproven
                .iter()
                .filter_map(|(uuid, _)| planner.take_read_error(uuid).map(|error| (*uuid, error)))
                .collect();
            // A failed read says nothing about the request: when it is the only
            // obstacle it is reported as is, keeping its own classification,
            // and after the read recovers the decision is made again.
            if refusal_is_the_read_error(
                unproven
                    .iter()
                    .map(|(uuid, _)| read_errors.contains_key(uuid)),
            ) && let Some(error) = unproven
                .iter()
                .find_map(|(uuid, _)| read_errors.remove(uuid))
            {
                log::warn!(
                    "CreateIndex: nothing was committed for '{}': removing segments {:?} \
                     depends on coverage a failed read left unknown",
                    index_name,
                    unresupplied.iter().map(|idx| idx.uuid).collect::<Vec<_>>(),
                );
                return Err(error);
            }
            return Err(Error::invalid_input(format!(
                "CreateIndex: cannot establish that committing these segments for '{}' keeps \
                 the index's coverage: coverage cannot be established for {}, so removing \
                 segments {:?}, whose fragments the incoming segments that can stand in for \
                 them do not all store, could lose coverage nobody can see. Nothing was \
                 committed; build the incoming segments over fragments that do not overlap \
                 the segments being removed, or replace every segment of the index in one \
                 commit",
                index_name,
                unproven
                    .iter()
                    .map(|(uuid, reason)| match read_errors.get(uuid) {
                        Some(error) => format!("segment {uuid} (read failed: {error})"),
                        None => format!("segment {uuid} ({reason})"),
                    })
                    .join(", "),
                unresupplied.iter().map(|idx| idx.uuid).collect::<Vec<_>>(),
            )));
        }
        return Ok(PartialCommitDecision {
            retained: candidates.clone(),
            dropped: Vec::new(),
            requires_unchanged_snapshot: by_exception,
        });
    }

    let removed_uuids: HashSet<Uuid> = removed.iter().map(|idx| idx.uuid).collect();
    let kept: Vec<&IndexMetadata> = existing
        .iter()
        .filter(|idx| !removed_uuids.contains(&idx.uuid))
        .chain(incoming.iter())
        .collect();
    let mut retained = HashSet::new();
    let mut dropped = Vec::new();
    for (idx, coverage) in kept.iter().zip(planner.coverage(&kept)) {
        if !candidates.contains(&idx.uuid) {
            continue;
        }
        if coverage.is_empty() {
            dropped.push((*idx).clone());
        } else {
            retained.insert(idx.uuid);
        }
    }

    let published: Vec<&IndexMetadata> = kept
        .iter()
        .copied()
        .filter(|idx| !dropped.iter().any(|gone| gone.uuid == idx.uuid))
        .collect();
    let existing_refs: Vec<&IndexMetadata> = existing.iter().collect();
    let before_by_segment = planner.coverage(&existing_refs);
    let lost = covered_by(&before_by_segment) - covered_by(&planner.coverage(&published));
    if !lost.is_empty() {
        let live = dataset.fragment_bitmap.as_ref();
        let gone = |uuid: &Uuid| {
            removed_uuids.contains(uuid) || dropped.iter().any(|gone| gone.uuid == *uuid)
        };
        let culprits = existing
            .iter()
            .zip(&before_by_segment)
            .filter(|(idx, coverage)| gone(&idx.uuid) && !coverage.is_disjoint(&lost))
            .map(|(idx, _)| {
                let overlap = idx
                    .fragment_bitmap
                    .as_ref()
                    .map(|stored| stored & live & incoming_fragments)
                    .unwrap_or_default();
                format!(
                    "segment {} (live fragments the incoming segments overlap: {:?})",
                    idx.uuid,
                    overlap.iter().collect::<Vec<_>>()
                )
            })
            .join(", ");
        return Err(Error::invalid_input(format!(
            "CreateIndex: committing these segments for '{}' would remove {} and leave live \
             fragments {:?} without the index coverage they have now. Nothing was committed; \
             build the incoming segments over fragments that do not overlap those segments' \
             live fragments, or also over every fragment listed",
            index_name,
            culprits,
            lost.iter().collect::<Vec<_>>()
        )));
    }

    Ok(PartialCommitDecision {
        retained,
        requires_unchanged_snapshot: by_exception
            || !unresupplied.is_empty()
            || !dropped.is_empty(),
        dropped,
    })
}

/// Decides, on `dataset`'s snapshot, the CreateIndex transaction committing
/// `segments` as `index_name` on `column`, and whether it has to land on that
/// snapshot unchanged: true when which segments stay rests on coverage
/// derived from it, which the commit loop's rebase does not derive again.
/// Every check runs on this snapshot, starting from the segments as their
/// builds produced them.
pub(in crate::index) async fn decide_segment_commit(
    dataset: &Dataset,
    index_name: &str,
    column: &str,
    segments: Vec<IndexSegment>,
) -> Result<(Transaction, bool)> {
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::index(format!(
            "CreateIndex: column '{column}' does not exist"
        )));
    };

    let dataset_fragments = dataset.fragment_bitmap.as_ref().clone();
    if segments.first().is_some_and(|segment| {
        segment
            .index_details()
            .type_url
            .ends_with("NGramIndexDetails")
    }) {
        let has_retired_coverage = segments
            .iter()
            .any(|segment| !(segment.fragment_bitmap() - &dataset_fragments).is_empty());
        let frag_reuse_index = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
        let requires_rebuild = frag_reuse_index.as_ref().is_some_and(|frag_reuse_index| {
            segments.iter().any(|segment| {
                append::fragment_reuse_affects_segment(
                    frag_reuse_index,
                    segment.fragment_bitmap(),
                    segment.dataset_version(),
                )
            })
        });
        if has_retired_coverage || requires_rebuild {
            return Err(Error::invalid_input(
                "CreateIndex: NGram segments built before compaction must be rebuilt or merged \
                 with merge_existing_index_segments before commit"
                    .to_string(),
            ));
        }
    }
    // On a tagged table the segments are first validated against this
    // snapshot (`replay_staged_segments`): a rewrite since their build
    // keeps their provenance translating, an in-place column rewrite
    // withdraws the affected coverage, a rewrite they cannot be carried
    // across says to rebuild. The commit then reads from this snapshot,
    // and the commit loop carries the segments across anything newer
    // through the same rules.
    let segments = replay_staged_segments(
        dataset,
        segments
            .into_iter()
            .map(|segment| segment.into_metadata(index_name))
            .collect(),
    )
    .await?
    .into_iter()
    .map(IntoIndexSegment::into_index_segment)
    .collect::<Result<Vec<_>>>()?;
    let new_indices =
        build_index_metadata_from_segments(dataset, index_name, field.id, segments).await?;
    validate_segment_metadata(index_name, &new_indices)?;
    validate_segment_index_details(index_name, &new_indices)?;

    let incoming_type_url = new_indices[0]
        .index_details
        .as_ref()
        .map(|details| details.type_url.clone());
    let mut incoming_fragments = RoaringBitmap::new();
    for segment in &new_indices {
        if let Some(fragment_bitmap) = &segment.fragment_bitmap {
            incoming_fragments |= fragment_bitmap.clone();
        }
    }

    let existing_named_indices = load_all_indices_by_name(dataset, index_name).await?;
    if existing_named_indices.iter().any(|idx| {
        // Field order is part of a plugin scope's identity. Covered
        // single-column indices still compare only their keyed prefix.
        idx.keyed_fields() != new_indices[0].keyed_fields()
    }) {
        return Err(Error::index(format!(
            "Index name '{index_name}' already exists with different fields, \
            please specify a different name"
        )));
    }
    let existing_different_type_url = existing_named_indices.iter().find_map(|idx| {
        // Legacy metadata may omit index details. Its type cannot be proven
        // compatible, so only a full replacement is safe.
        let Some(existing_details) = idx.index_details.as_ref() else {
            return Some("<unknown>".to_owned());
        };
        let existing_type_url = existing_details.type_url.as_str();
        (Some(existing_type_url) != incoming_type_url.as_deref())
            .then(|| existing_type_url.to_owned())
    });
    let missing_fragments = &dataset_fragments - &incoming_fragments;
    if let Some(existing_type_url) = &existing_different_type_url
        && !missing_fragments.is_empty()
    {
        return Err(Error::invalid_input(format!(
            "CreateIndex: cannot change index '{}' from type '{}' to type '{}' with partial fragment coverage; incoming segments are missing current fragments {:?}",
            index_name,
            existing_type_url,
            incoming_type_url.as_deref().unwrap_or("<missing>"),
            missing_fragments.iter().collect::<Vec<_>>()
        )));
    }

    let is_index_type_change = existing_different_type_url.is_some();
    // A partial commit on a tagged history is decided on the coverage the
    // reader derives for the composition it publishes. A replacement of
    // every live fragment, or a type change, replaces every segment.
    let mut history = if is_index_type_change || missing_fragments.is_empty() {
        None
    } else {
        frag_reuse::TaggedHistory::find(dataset).await?
    };
    let is_tagged_partial = history.is_some();
    let mut historical_candidates: HashSet<Uuid> = HashSet::new();
    if let Some(history) = history.as_mut() {
        let zero_live: Vec<&IndexMetadata> = existing_named_indices
            .iter()
            .filter(|idx| is_historical_candidate(idx, &dataset_fragments, &incoming_fragments))
            .collect();
        if !zero_live.is_empty() {
            let planner = history.planner(dataset).await?;
            historical_candidates = zero_live
                .into_iter()
                .filter(|idx| {
                    idx.fragment_bitmap
                        .as_ref()
                        .is_some_and(|stored| !planner.traceable_fragments(stored).is_empty())
                })
                .map(|idx| idx.uuid)
                .collect();
        }
    }
    let existing_snapshot = existing_named_indices.clone();
    // What a retained sibling has to agree with. Every incoming segment
    // already declares the same fields: `build_index_metadata_from_segments`
    // compares them against each other before this point.
    let expected = &new_indices[0];
    let mut removed_indices = existing_named_indices
        .into_iter()
        .map(|idx| -> Result<Option<IndexMetadata>> {
            // A logical index cannot combine segment types. Full current-fragment
            // coverage was verified above, so a type change replaces every segment.
            if is_index_type_change {
                return Ok(Some(idx));
            }

            let Some(existing_fragments) = idx.effective_fragment_bitmap(&dataset_fragments)
            else {
                if incoming_fragments != dataset_fragments {
                    return Err(Error::invalid_input(format!(
                        "CreateIndex: cannot replace legacy index segment {} for '{}' with partial fragment coverage; rebuild all fragments in one commit",
                        idx.uuid, index_name
                    )));
                }
                return Ok(Some(idx));
            };

            // A zero-fragment segment can be used to create an index while
            // deferring the actual build. Such a segment is disjoint from every
            // other segment but should still be removed. So is a segment whose
            // fragments were all retired, unless it is a historical candidate:
            // whether that one stays is decided on the published composition
            // below.
            if existing_fragments.is_empty() {
                if historical_candidates.contains(&idx.uuid) {
                    return Ok(None);
                }
                return Ok(Some(idx));
            }

            // On a tagged history the stored bitmap is provenance: a segment
            // sharing a retired source with the incoming segments would
            // contribute those rows a second time, so it is replaced.
            let shares_provenance = is_tagged_partial
                && idx
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|stored| !stored.is_disjoint(&incoming_fragments));
            if existing_fragments.is_disjoint(&incoming_fragments) && !shares_provenance {
                check_retained_declaration(index_name, expected, &idx)?;
                return Ok(None);
            }

            // Only the shared provenance sent a segment with disjoint live
            // fragments here.
            if existing_fragments.is_disjoint(&incoming_fragments) {
                let shared = idx
                    .fragment_bitmap
                    .as_ref()
                    .map(|stored| stored & &incoming_fragments)
                    .unwrap_or_default();
                return Err(Error::invalid_input(format!(
                    "CreateIndex: existing segment {} for '{}' shares already-rewritten \
                     source fragments {:?} with the incoming segments; keeping both would \
                     serve those rows twice. Nothing was committed; the incoming segments \
                     must also cover its live fragments {:?}, or not include the shared \
                     sources",
                    idx.uuid,
                    index_name,
                    shared.iter().collect::<Vec<_>>(),
                    existing_fragments.iter().collect::<Vec<_>>()
                )));
            }
            let uncovered = existing_fragments - &incoming_fragments;
            if !uncovered.is_empty() {
                return Err(Error::invalid_input(format!(
                    "CreateIndex: incoming segments for '{}' would orphan fragments {:?} from existing segment {}",
                    index_name,
                    uncovered.iter().collect::<Vec<_>>(),
                    idx.uuid
                )));
            }

            Ok(Some(idx))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut requires_unchanged_snapshot = false;
    if let Some(history) = history.as_mut() {
        let decision = decide_partial_composition(
            dataset,
            history,
            index_name,
            &existing_snapshot,
            &new_indices,
            &removed_indices,
            &historical_candidates,
            &incoming_fragments,
        )
        .await?;
        for idx in existing_snapshot
            .iter()
            .filter(|idx| decision.retained.contains(&idx.uuid))
        {
            check_retained_declaration(index_name, expected, idx)?;
        }
        removed_indices.extend(decision.dropped);
        requires_unchanged_snapshot = decision.requires_unchanged_snapshot;
    }
    // Segments that stay together must be compatible under the plugin's
    // rules (for example identical hash parameters). Segments this commit
    // removes are not consulted, so a full rebuild may change parameters.
    let retained_indices: Vec<IndexMetadata> = existing_snapshot
        .into_iter()
        .filter(|idx| {
            !removed_indices
                .iter()
                .any(|removed| removed.uuid == idx.uuid)
        })
        .collect();
    validate_segment_params_compatible(&retained_indices, &new_indices)?;

    let coexisting_indices = new_indices.iter().chain(retained_indices.iter());
    if segment_has_inverted_details(&new_indices[0]) {
        // Apply the merge/query semantic contract only to segments that will
        // coexist, so a full rebuild can change the analyzer settings.
        let segments = coexisting_indices.clone().cloned().collect::<Vec<_>>();
        scalar::inverted::load_segment_details(dataset, column, &segments).await?;
    }

    // The query planner ranks coexisting vector segments under one contract.
    // Validate after replacement selection so a full rebuild may change it.
    let vector_segment_count = coexisting_indices
        .clone()
        .filter(|segment| segment_has_vector_details(segment))
        .count();
    if vector_segment_count > 1 {
        // Only the query contract is compared, never a row: a staged
        // segment is not in the manifest, so it opens on an identity plan,
        // and a retained one opens for maintenance so that a segment the
        // tagged reader excludes from the listing still takes part.
        let mut vector_indices = Vec::with_capacity(vector_segment_count);
        for segment in coexisting_indices {
            let staged_plan = new_indices
                .iter()
                .any(|staged| staged.uuid == segment.uuid)
                .then_some(&frag_reuse::SegmentRemappingPlan::Identity);
            let opened = dataset
                .open_vector_index_from_metadata_with_plan(
                    column,
                    segment,
                    staged_plan,
                    frag_reuse::OpenPurpose::Maintenance,
                    &NoOpMetricsCollector,
                )
                .await;
            // A retained segment that cannot be checked is neither skipped
            // nor removed: the commit is refused.
            let index = match opened {
                Ok(index) => index,
                Err(error) if staged_plan.is_none() => {
                    let read_error = history
                        .as_mut()
                        .and_then(|history| history.take_read_error(&segment.uuid));
                    return Err(retained_segment_unopenable(
                        index_name,
                        segment.uuid,
                        error,
                        read_error,
                    ));
                }
                Err(error) => return Err(error),
            };
            vector_indices.push(index);
        }
        vector::ivf::validate_vector_query_compatibility(
            &vector_indices,
            &format!("CreateIndex: index '{index_name}'"),
        )
        .map_err(|error| Error::invalid_input(error.to_string()))?;
    }

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices,
            removed_indices,
        },
        None,
    );

    Ok((transaction, requires_unchanged_snapshot))
}

#[async_trait]
impl DatasetIndexExt for Dataset {
    type IndexBuilder<'a> = CreateIndexBuilder<'a>;
    /// Create a builder for creating an index on columns.
    ///
    /// This returns a builder that can be configured with additional options
    /// before awaiting to execute.
    ///
    /// # Examples
    ///
    /// Create a scalar BTREE index:
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::index::DatasetIndexExt;
    /// # use lance_index::{IndexType, scalar::ScalarIndexParams};
    /// # async fn example(dataset: &mut Dataset) -> Result<()> {
    /// let params = ScalarIndexParams::default();
    /// dataset
    ///     .create_index_builder(&["id"], IndexType::BTree, &params)
    ///     .name("id_index".to_string())
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Create an empty index that will be populated later:
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::index::DatasetIndexExt;
    /// # use lance_index::{IndexType, scalar::ScalarIndexParams};
    /// # async fn example(dataset: &mut Dataset) -> Result<()> {
    /// let params = ScalarIndexParams::default();
    /// dataset
    ///     .create_index_builder(&["category"], IndexType::Bitmap, &params)
    ///     .train(false)  // Create empty index
    ///     .replace(true)  // Replace if exists
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn create_index_builder<'a>(
        &'a mut self,
        columns: &[&str],
        index_type: IndexType,
        params: &'a dyn IndexParams,
    ) -> CreateIndexBuilder<'a> {
        CreateIndexBuilder::new(self, columns, index_type, params)
    }

    #[instrument(skip_all)]
    async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
        replace: bool,
    ) -> Result<IndexMetadata> {
        // Use the builder pattern with default train=true for backward compatibility
        let mut builder = self.create_index_builder(columns, index_type, params);

        if let Some(name) = name {
            builder = builder.name(name);
        }

        builder.replace(replace).await
    }

    async fn drop_index(&mut self, name: &str) -> Result<()> {
        // Removal never opens the index, so an index this build cannot read is
        // still droppable - and has to be, since it is otherwise unremovable.
        let indices = load_all_indices_by_name(self, name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: indices.clone(),
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    async fn prewarm_index(&self, name: &str) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }

        let available_segment_count = indices.len();
        prewarm_index_segments_by_metadata(self, name, indices, None, available_segment_count, None)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_with_options(&self, name: &str, options: &PrewarmOptions) -> Result<()> {
        self.prewarm_index_with_options_result(name, options)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_with_options_result(
        &self,
        name: &str,
        options: &PrewarmOptions,
    ) -> Result<FtsPrewarmResult> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }

        let available_segment_count = indices.len();
        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            Some(options),
            available_segment_count,
            None,
        )
        .await
    }

    async fn prewarm_index_segments(&self, name: &str, segment_ids: &[Uuid]) -> Result<()> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }
        let available_segment_count = indices.len();
        let indices = filter_index_segments_by_ids(name, indices, segment_ids)?;

        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            None,
            available_segment_count,
            Some(segment_ids.len()),
        )
        .await
        .map(|_| ())
    }

    async fn prewarm_index_segments_with_options(
        &self,
        name: &str,
        segment_ids: &[Uuid],
        options: &PrewarmOptions,
    ) -> Result<()> {
        self.prewarm_index_segments_with_options_result(name, segment_ids, options)
            .await
            .map(|_| ())
    }

    async fn prewarm_index_segments_with_options_result(
        &self,
        name: &str,
        segment_ids: &[Uuid],
        options: &PrewarmOptions,
    ) -> Result<FtsPrewarmResult> {
        let indices = self.load_indices_by_name(name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", name)));
        }
        let available_segment_count = indices.len();
        let indices = filter_index_segments_by_ids(name, indices, segment_ids)?;

        prewarm_index_segments_by_metadata(
            self,
            name,
            indices,
            Some(options),
            available_segment_count,
            Some(segment_ids.len()),
        )
        .await
    }

    async fn describe_indices<'a, 'b>(
        &'a self,
        criteria: Option<IndexCriteria<'b>>,
    ) -> Result<Vec<Arc<dyn IndexDescription>>> {
        let indices = self.load_indices().await?;
        let mut indices = if let Some(criteria) = criteria {
            indices.iter().filter(|idx| {
                if idx.index_details.is_none() {
                    log::warn!("The method describe_indices does not support indexes without index details.  Please retrain the index {}", idx.name);
                    return false;
                }
                // Only the keyed prefix. `index_matches_criteria` compares this
                // slice against the single column named by `for_column`, so
                // passing the carried columns too would push its length past one
                // and silently drop every covered index from a filtered
                // describe.
                let fields = idx
                    .keyed_fields()
                    .iter()
                    .filter_map(|id| self.schema().field_by_id(*id))
                    .collect::<Vec<_>>();
                match index_matches_criteria(idx, &criteria, &fields, false, self.schema()) {
                    Ok(matched) => matched,
                    Err(err) => {
                        log::warn!("Could not describe index {}: {}", idx.name, err);
                        false
                    }
                }
            }).collect::<Vec<_>>()
        } else {
            indices.iter().collect::<Vec<_>>()
        };
        indices.sort_by_key(|idx| &idx.name);

        let grouped: Vec<Vec<IndexMetadata>> = indices
            .into_iter()
            .chunk_by(|idx| idx.name.clone())
            .into_iter()
            .map(|(_, segments)| segments.cloned().collect::<Vec<_>>())
            .collect();

        let mut results = Vec::with_capacity(grouped.len());
        for segments in grouped {
            let desc = IndexDescriptionImpl::try_new(segments, self).await?;
            results.push(Arc::new(desc) as Arc<dyn IndexDescription>);
        }
        Ok(results)
    }

    /// From the derived listing only: a segment the tagged reader excludes
    /// (it derives no coverage for it) is absent here, so a query opening it
    /// by uuid fails instead of answering with nothing. Maintenance reaches
    /// such a segment through `Dataset::load_index_with_purpose`.
    async fn load_index(&self, uuid: &Uuid) -> Result<Option<IndexMetadata>> {
        self.load_index_with_purpose(uuid, frag_reuse::OpenPurpose::Query)
            .await
    }

    async fn load_indices(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let derive = || async {
            let indices = load_all_indices(self).await?;
            // Best-effort inference and unreadable format markers must be
            // retried on later calls instead of freezing an incomplete listing.
            let retry_inference = indices
                .iter()
                .any(|idx| needs_vector_details_inference(idx, self.schema()));
            let listing = self.derive_index_listing(indices).await?;
            Ok(QueryIndexListing {
                indices: listing.indices,
                retry_derivation: retry_inference || !listing.unverifiable.is_empty(),
            })
        };
        // The key's e-tag is its generation token. Without one, a dataset
        // dropped and recreated at this URI restarts its version history, so a
        // version-only key could serve the previous incarnation's listing from
        // this cache, which is shared by every dataset opened at the URI.
        // `None` is not proof that two observations share an incarnation (see
        // `ManifestLocation::e_tag`), so derive privately and never publish.
        let Some(e_tag) = self.manifest_location.e_tag.as_deref() else {
            return Ok(derive().await?.indices);
        };
        let derived_key = QueryIndexListingKey(DerivedIndexListingKey {
            version: self.manifest.version,
            store_identity: &self.object_store.store_prefix,
            e_tag: Some(e_tag),
        });
        let (listing, was_cached) = self
            .index_cache
            .get_or_insert_with_key_hit(derived_key, derive)
            .await?;
        if was_cached && listing.retry_derivation {
            let refreshed = Arc::new(derive().await?);
            self.index_cache
                .insert_with_key(&derived_key, refreshed.clone())
                .await;
            return Ok(refreshed.indices.clone());
        }
        Ok(listing.indices.clone())
    }

    async fn merge_existing_index_segments(
        &self,
        source_segments: Vec<IndexMetadata>,
    ) -> Result<IndexMetadata> {
        validate_segment_metadata("uncommitted", &source_segments)?;
        if let Some(segment) = source_segments
            .iter()
            .find(|segment| segment.dataset_version > self.manifest.version)
        {
            return Err(Error::invalid_input(format!(
                "merge_existing_index_segments: segment {} was built at future dataset version {} (current version {})",
                segment.uuid, segment.dataset_version, self.manifest.version
            )));
        }
        // On a tagged table the staged segments are validated against this
        // snapshot before anything reads their addresses or projects their
        // coverage: a rewrite since their build may have moved their rows
        // (fine, they translate) or rewritten an indexed column in place (the
        // affected coverage is withdrawn). See `replay_staged_segments`.
        let source_segments = replay_staged_segments(self, source_segments).await?;
        let mut source_segments = source_segments;
        let source_dataset_version = source_segments
            .iter()
            .map(|segment| segment.dataset_version)
            .min()
            .unwrap_or(self.manifest.version);
        let field_id = *source_segments[0].fields.first().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} is missing field ids",
                source_segments[0].uuid
            ))
        })?;
        let expected_covering_fields = source_segments[0].covering_fields.clone();
        for segment in &source_segments {
            // Same rule as `build_index_metadata_from_segments`: only the
            // keyed-field count matters here, not the exact `fields` vector,
            // so a covered segment is not rejected outright.
            if segment.keyed_field() != Some(field_id) {
                return Err(Error::invalid_input(format!(
                    "merge_existing_index_segments: segment {} was built for fields {:?} \
                     (carried {:?}), expected keyed field [{}]",
                    segment.uuid, segment.fields, segment.covering_fields, field_id
                )));
            }
            segment.validate_covering_fields()?;
            // Merging requires one coherent output declaration, so every
            // input must carry the same columns (not merely the same count).
            if segment.covering_fields != expected_covering_fields {
                return Err(Error::invalid_input(
                    "merge_existing_index_segments requires segments with identical fields"
                        .to_string(),
                ));
            }
        }
        let all_vector = source_segments.iter().all(segment_has_vector_details);
        let all_inverted = source_segments.iter().all(segment_has_inverted_details);
        let all_bitmap = source_segments.iter().all(segment_has_bitmap_details);
        let all_bloomfilter = source_segments.iter().all(segment_has_bloomfilter_details);
        let all_btree = source_segments.iter().all(segment_has_btree_details);
        let all_fmindex = source_segments.iter().all(segment_has_fmindex_details);
        let all_zonemap = source_segments.iter().all(segment_has_zonemap_details);
        let all_label_list = source_segments.iter().all(segment_has_label_list_details);
        let all_rtree = source_segments.iter().all(segment_has_rtree_details);
        let all_ngram = source_segments.iter().all(segment_has_ngram_details);
        let all_minhashlsh = source_segments.iter().all(segment_has_minhashlsh_details);
        if !all_vector
            && !all_inverted
            && !all_bitmap
            && !all_bloomfilter
            && !all_btree
            && !all_fmindex
            && !all_zonemap
            && !all_label_list
            && !all_rtree
            && !all_ngram
            && !all_minhashlsh
        {
            return Err(Error::invalid_input(
                "merge_existing_index_segments requires all segments to have the same supported index type"
                    .to_string(),
            ));
        }

        validate_segment_params_compatible(&[], &source_segments)?;

        // Before the coverage work below, which would be wasted without `geo`.
        #[cfg(not(feature = "geo"))]
        if all_rtree {
            return Err(Error::not_supported(
                "RTree segment merge requires the `geo` feature".to_string(),
            ));
        }

        // Coverage may only move with the row addresses. A scalar merge loads its
        // sources through the reuse index as a row-address remapper, so those
        // addresses land in the current fragment space and the coverage has to
        // follow them.
        //
        // Vector is exempt because its merge cannot remap: it hands the segments
        // to the distributed file merger with an object store and a directory,
        // reaching no dataset and so no reuse index.
        //
        // Staged segments are unknown to the snapshot plan: on a tagged table
        // they are planned here as one group and every open below uses that
        // plan (see `frag_reuse::plan_staged_segments`).
        let staged = if all_vector {
            None
        } else {
            frag_reuse::plan_staged_segments(self, &source_segments).await?
        };
        // Before the staleness pass, which drops the retired fragments this remaps.
        let staged_coverage: Vec<Option<RoaringBitmap>> = source_segments
            .iter()
            .map(|segment| segment.fragment_bitmap.clone())
            .collect();
        let remapped_fragments = if !all_vector {
            let index_name = source_segments[0].name.clone();
            remap_merged_segment_coverage(self, &index_name, &mut source_segments, staged.as_ref())
                .await?
        } else {
            None
        };
        let has_remapped_source_coverage = remapped_fragments.is_some();
        // A tagged history reports its remap even when every plan is the
        // identity, so whether row addresses moved is read off the coverage.
        let source_coverage_moved = source_segments
            .iter()
            .zip(&staged_coverage)
            .any(|(segment, staged)| &segment.fragment_bitmap != staged);

        if all_rtree {
            let mut source_coverage = source_segments
                .iter()
                .cloned()
                .map(IntoIndexSegment::into_index_segment)
                .collect::<Result<Vec<_>>>()?;
            // Only fragments the remap added are exempt.
            let exempt = remapped_fragments.unwrap_or_default();
            prune_stale_segment_coverage(self, &mut source_coverage, true, &exempt, true).await?;
            for (source, coverage) in source_segments.iter_mut().zip(source_coverage) {
                source.fragment_bitmap = Some(coverage.fragment_bitmap().clone());
            }
        }

        let merged_dataset_version = if all_rtree || has_remapped_source_coverage {
            self.manifest.version
        } else {
            source_dataset_version
        };

        let staged = staged.as_ref();
        let mut merged_segment = if all_vector {
            crate::index::vector::ivf::merge_segments(self, source_segments).await?
        } else if all_inverted {
            crate::index::scalar::inverted::merge_segments(
                self,
                source_segments,
                staged,
                source_coverage_moved,
            )
            .await?
        } else if all_fmindex {
            crate::index::scalar::fmindex::merge_segments(
                self,
                source_segments,
                has_remapped_source_coverage,
                staged,
            )
            .await?
        } else if all_bitmap {
            crate::index::scalar::bitmap::merge_segments(self, source_segments, staged).await?
        } else if all_bloomfilter {
            crate::index::scalar::bloomfilter::merge_segments(self, source_segments, staged).await?
        } else if all_label_list {
            crate::index::scalar::label_list::merge_segments(self, source_segments, staged).await?
        } else if all_zonemap {
            crate::index::scalar::zonemap::merge_segments(self, source_segments, staged).await?
        } else if all_ngram {
            crate::index::scalar::ngram::merge_segments(self, source_segments, staged).await?
        } else if all_minhashlsh {
            crate::index::scalar::minhash_lsh::merge_segments(self, source_segments, staged).await?
        } else if all_rtree {
            #[cfg(feature = "geo")]
            {
                crate::index::scalar::rtree::merge_segments(self, source_segments, staged).await?
            }
            // Refused above, before the coverage work.
            #[cfg(not(feature = "geo"))]
            unreachable!("an RTree merge without `geo` returns before this point")
        } else {
            crate::index::scalar::btree::merge_segments(self, source_segments, staged).await?
        };
        if !all_ngram && !all_fmindex {
            merged_segment.dataset_version = merged_dataset_version;
        }
        Ok(merged_segment)
    }

    /// A decision that depends on derived coverage is made again on the latest
    /// version when another commit lands first, a bounded number of times and
    /// without waiting between attempts, so it improves progress but does not
    /// guarantee it under continuous writes. A decision is made again only
    /// when the ordinary conflict rules accept the intervening commits. On
    /// error the dataset is left at
    /// the version it was called with.
    async fn commit_existing_index_segments(
        &mut self,
        index_name: &str,
        column: &str,
        segments: Vec<impl IntoIndexSegment + Send>,
    ) -> Result<()> {
        let segments = segments
            .into_iter()
            .map(IntoIndexSegment::into_index_segment)
            .collect::<Result<Vec<_>>>()?;
        let entry = self.clone();
        // A commit whose choice of segments rests on coverage derived from
        // its snapshot lands only on that snapshot. When another writer
        // commits first, the whole decision is made again on the latest
        // version from the original segments, within the attempts the commit
        // loop allows a rebasing commit.
        let budget = commit_attempt_budget(&CommitConfig::default());
        let result = async {
            let mut attempt = 1;
            loop {
                let (transaction, requires_unchanged_snapshot) =
                    decide_segment_commit(self, index_name, column, segments.clone()).await?;
                if !requires_unchanged_snapshot {
                    // The attempts left after earlier decisions, so all of
                    // them share one budget.
                    let commit_config = if attempt == 1 {
                        CommitConfig::default()
                    } else {
                        CommitConfig {
                            num_retries: budget - attempt + 1,
                            ..CommitConfig::default()
                        }
                    };
                    return self
                        .apply_commit(transaction, &Default::default(), &commit_config)
                        .await;
                }
                let read_version = self.manifest.version;
                let write_config = ManifestWriteConfig::default();
                let outcome = self
                    .apply_commit_with_expected_latest_version(
                        transaction.clone(),
                        &write_config,
                        &Default::default(),
                        Some(read_version),
                    )
                    .await;
                if decides_again(&outcome, attempt, budget) {
                    attempt += 1;
                    // The next decision runs on exactly the versions checked
                    // here; a conflict ends the call.
                    *self = check_versions_since_read(self, &transaction, &write_config).await?;
                    continue;
                }
                return match outcome {
                    Err(error) if ConcurrentVersion::caused(&error) => {
                        let version = match error.find_root() {
                            Error::RetryableCommitConflict { version, .. } => *version,
                            _ => read_version,
                        };
                        Err(Error::retryable_commit_conflict_source(
                            version,
                            Box::new(NotAppliedAtReadVersion {
                                index_name: index_name.to_owned(),
                                read_version,
                                attempts: attempt,
                                source: error,
                            }),
                        ))
                    }
                    other => other,
                };
            }
        }
        .await;
        if result.is_err() {
            *self = entry;
        }
        result
    }

    async fn load_scalar_index<'a, 'b>(
        &'a self,
        criteria: IndexCriteria<'b>,
    ) -> Result<Option<IndexMetadata>> {
        let indices = self.load_indices().await?;

        let mut indices = indices
            .iter()
            .filter(|idx| {
                // System indices are fieldless by design. Report malformed user indices,
                // but don't fail the operation (we might not be using that index).
                if idx.fields.is_empty() {
                    if !is_system_index(idx) {
                        log::error!("Index {} has no fields", idx.name);
                    }
                    false
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();
        // This sorting & chunking is only needed to calculate if there are multiple indexes on the same
        // field.  This fact is only needed for backwards compatibility behavior for indexes that don't have
        // index details.  At some point we should deprecate indexes without index details.
        //
        // TODO: At some point we should just fail if the index details are missing and ask the user to
        // retrain the index.
        indices.sort_by_key(|idx| idx.fields[0]);
        let indice_by_field = indices.into_iter().chunk_by(|idx| idx.fields[0]);
        for (field_id, indices) in &indice_by_field {
            let indices = indices.collect::<Vec<_>>();
            let has_multiple = indices.len() > 1;
            for idx in indices {
                let field = self.schema().field_by_id(field_id);
                if let Some(field) = field
                    && index_matches_criteria(
                        idx,
                        &criteria,
                        &[field],
                        has_multiple,
                        self.schema(),
                    )?
                {
                    let non_empty = idx.fragment_bitmap.as_ref().is_some_and(|bitmap| {
                        bitmap.intersection_len(self.fragment_bitmap.as_ref()) > 0
                    });
                    let is_fts_index = if let Some(details) = &idx.index_details {
                        IndexDetails(details.clone()).supports_fts()
                    } else {
                        false
                    };
                    // FTS indices must always be returned even if empty, because FTS queries
                    // require an index to exist. The query execution will handle the empty
                    // bitmap appropriately and fall back to scanning unindexed data.
                    // Other index types can be skipped if empty since they're optional optimizations.
                    if non_empty || is_fts_index {
                        return Ok(Some(idx.clone()));
                    }
                }
            }
        }
        return Ok(None);
    }

    async fn index_statistics(&self, index_name: &str) -> Result<String> {
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.is_empty() {
            return Err(Error::index_not_found(format!("name={}", index_name)));
        }

        if index_name == FRAG_REUSE_INDEX_NAME {
            return index_statistics_frag_reuse(self).boxed().await;
        }

        if index_name == MEM_WAL_INDEX_NAME {
            return index_statistics_mem_wal(self).boxed().await;
        }

        index_statistics_scalar(self, index_name, metadatas)
            .boxed()
            .await
    }

    async fn index_segment_statistics(
        &self,
        index_name: &str,
        index_uuids: &[Uuid],
    ) -> Result<Vec<IndexSegmentStatistics>> {
        if index_uuids.is_empty() {
            return Err(Error::invalid_input(
                "At least one index segment UUID is required".to_string(),
            ));
        }
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.iter().any(is_system_index) {
            return Err(Error::invalid_input(
                "Segment statistics are not supported for system indexes".to_string(),
            ));
        }
        let metadata_by_uuid: HashMap<_, _> = metadatas
            .iter()
            .map(|metadata| (metadata.uuid, metadata))
            .collect();
        let mut seen_uuids = HashSet::with_capacity(index_uuids.len());
        let mut statistics = Vec::with_capacity(index_uuids.len());
        for index_uuid in index_uuids {
            if !seen_uuids.insert(*index_uuid) {
                return Err(Error::invalid_input(format!(
                    "Duplicate index segment UUID: {index_uuid}"
                )));
            }
            let metadata = metadata_by_uuid.get(index_uuid).ok_or_else(|| {
                Error::index_not_found(format!("name={index_name}, uuid={index_uuid}"))
            })?;
            let field_id = metadata.fields.first().ok_or_else(|| {
                Error::index(format!("Index segment {index_uuid} has no indexed field"))
            })?;
            let field_path = self.schema().field_path(*field_id)?;
            statistics.push(collect_index_segment_statistics(self, metadata, &field_path).await?);
        }
        Ok(statistics)
    }

    async fn index_statistics_from_segments(
        &self,
        index_name: &str,
        segment_statistics: Vec<IndexSegmentStatistics>,
    ) -> Result<String> {
        let metadatas = self.load_indices_by_name(index_name).await?;
        if metadatas.iter().any(is_system_index) {
            return Err(Error::invalid_input(
                "Segment statistics are not supported for system indexes".to_string(),
            ));
        }
        if metadatas.is_empty() {
            return Err(Error::index_not_found(format!("name={index_name}")));
        }
        serialize_regular_index_statistics(self, index_name, &metadatas, segment_statistics)
            .await?
            .ok_or_else(|| {
                Error::index(format!(
                    "Outdated fragment metadata for index {index_name}: collect index statistics \
                     in non-distributed mode to migrate before retrying"
                ))
            })
    }

    async fn read_index_partition(
        &self,
        index_name: &str,
        partition_id: usize,
        with_vector: bool,
    ) -> Result<SendableRecordBatchStream> {
        let indices = self.load_indices_by_name(index_name).await?;
        if indices.is_empty() {
            return Err(Error::index_not_found(format!("name={}", index_name)));
        }
        let field_id = indices[0].fields[0];
        let field_path = self.schema().field_path(field_id)?;
        let logical_index = self
            .open_logical_vector_index(&field_path, index_name)
            .await?;
        logical_index
            .as_ivf()?
            .read_partition(partition_id, with_vector)
            .await
    }

    #[instrument(skip_all)]

    async fn optimize_indices(&mut self, options: &OptimizeOptions) -> Result<()> {
        // Grouped from the complete list so a name's segments are all accounted
        // for. A segment this build cannot read is still coverage, and merging
        // against a group whose coverage is only partly visible would commit a
        // new segment claiming fragments an existing one already holds.
        let indices = load_all_indices(self).await?;

        // The stages a distributed optimize runs across workers, run here in
        // turn: without row bounds every index is one task that does the
        // whole single-process optimize, and every merge passes it through.
        let Some(plan) = optimize::plan_optimize_indices_or_skip(
            self,
            &indices,
            options,
            optimize::OptimizeExecution::Local,
        )
        .await?
        else {
            return Ok(());
        };
        let mut built = Vec::with_capacity(plan.tasks.len());
        for task in &plan.tasks {
            built.extend(
                task.execute_with_progress(self, options.progress.clone())
                    .await?,
            );
        }
        let mut merged = Vec::with_capacity(plan.merges.len());
        for merge in &plan.merges {
            merged.extend(
                merge
                    .execute_with_progress(self, &built, options.progress.clone())
                    .await?,
            );
        }

        // Without results this still commits when MemWAL catch-up needs it.
        optimize::commit_optimize_indices(self, merged, options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_is_the_read_error_only_when_every_unproven_segment_failed_a_read() {
        assert!(refusal_is_the_read_error([true]));
        assert!(refusal_is_the_read_error([true, true]));
        assert!(!refusal_is_the_read_error([true, false]));
        assert!(!refusal_is_the_read_error([false, true]));
        assert!(!refusal_is_the_read_error([false]));
        assert!(!refusal_is_the_read_error([]));
    }

    fn segment(stored: &[u32]) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![0],
            covering_fields: vec![],
            name: "i".into(),
            dataset_version: 1,
            fragment_bitmap: Some(stored.iter().copied().collect()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    /// The removed segments not supplied again, and whether the exception
    /// was relied on.
    fn resupply(
        removed: &[IndexMetadata],
        incoming: &[IndexMetadata],
        exclusion: impl Fn(&IndexMetadata) -> Option<CoverageExclusion>,
    ) -> (Vec<Uuid>, bool) {
        resupply_tracing(removed, incoming, exclusion, RoaringBitmap::clone)
    }

    /// [`resupply`] counting only the stored fragments `traceable` keeps.
    fn resupply_tracing(
        removed: &[IndexMetadata],
        incoming: &[IndexMetadata],
        exclusion: impl Fn(&IndexMetadata) -> Option<CoverageExclusion>,
        traceable: impl Fn(&RoaringBitmap) -> RoaringBitmap,
    ) -> (Vec<Uuid>, bool) {
        let resupply = unresupplied_removals(removed, incoming, exclusion, traceable);
        (
            resupply.unresupplied.iter().map(|idx| idx.uuid).collect(),
            resupply.by_exception,
        )
    }

    /// Looks a segment's classification up by UUID; unlisted segments take
    /// part in coverage.
    fn classified(
        classes: &[(Uuid, CoverageExclusion)],
    ) -> impl Fn(&IndexMetadata) -> Option<CoverageExclusion> + '_ {
        move |idx| {
            classes
                .iter()
                .find(|(uuid, _)| *uuid == idx.uuid)
                .map(|(_, exclusion)| *exclusion)
        }
    }

    /// R over {5} removed, I over {5, 7} incoming, both taking part in
    /// coverage: the ordinary path, which rests on no classification.
    #[test]
    fn a_member_supplies_a_removed_member_again() {
        let removed = vec![segment(&[5])];
        let incoming = vec![segment(&[5, 7])];

        assert_eq!(resupply(&removed, &incoming, |_| None), (vec![], false));
    }

    /// R and I both left out because the type has no batch remapper: R
    /// derives nothing in this build, and I stores all of it again.
    #[test]
    fn a_removed_segment_without_a_batch_remapper_is_supplied_by_the_same_class() {
        let removed = vec![segment(&[5])];
        let incoming = vec![segment(&[5, 7])];

        assert_eq!(
            resupply(&removed, &incoming, |_| Some(
                CoverageExclusion::NoBatchRemapper
            )),
            (vec![], true)
        );
    }

    /// R takes part in coverage and directly covers 5; I is left out because
    /// the type has no batch remapper. Removing R would lose that coverage.
    #[test]
    fn a_removed_member_is_not_supplied_by_a_segment_without_a_batch_remapper() {
        let removed = vec![segment(&[5])];
        let incoming = vec![segment(&[5, 7])];
        let i = incoming[0].uuid;

        assert_eq!(
            resupply(
                &removed,
                &incoming,
                classified(&[(i, CoverageExclusion::NoBatchRemapper)])
            ),
            (vec![removed[0].uuid], false)
        );
    }

    /// R is left out because the type has no batch remapper, but I is left
    /// out for a reason of its own or an unreadable marker: I supplies
    /// nothing.
    #[test]
    fn a_segment_specific_incoming_exclusion_supplies_nothing() {
        let removed = vec![segment(&[5])];
        let incoming = vec![segment(&[5, 7])];
        let r = removed[0].uuid;
        let i = incoming[0].uuid;

        for reason in [
            MissingCoverageReason::Corrupt,
            MissingCoverageReason::Unverifiable,
            MissingCoverageReason::Unsupported,
        ] {
            assert_eq!(
                resupply(
                    &removed,
                    &incoming,
                    classified(&[
                        (r, CoverageExclusion::NoBatchRemapper),
                        (i, CoverageExclusion::Segment(reason)),
                    ])
                ),
                (vec![r], false),
                "{reason:?}"
            );
        }
    }

    /// Under a history this build cannot interpret no exception applies,
    /// whatever the incoming segment's class.
    #[test]
    fn an_unsupported_history_grants_no_exception() {
        let removed = vec![segment(&[5])];
        let incoming = vec![segment(&[5, 7])];
        let r = removed[0].uuid;
        let i = incoming[0].uuid;

        for incoming_class in [
            CoverageExclusion::UnsupportedHistory,
            CoverageExclusion::NoBatchRemapper,
        ] {
            assert_eq!(
                resupply(
                    &removed,
                    &incoming,
                    classified(&[
                        (r, CoverageExclusion::UnsupportedHistory),
                        (i, incoming_class),
                    ])
                ),
                (vec![r], false),
                "{incoming_class:?}"
            );
        }
        assert_eq!(
            resupply(
                &removed,
                &incoming,
                classified(&[
                    (r, CoverageExclusion::NoBatchRemapper),
                    (i, CoverageExclusion::UnsupportedHistory),
                ])
            ),
            (vec![r], false)
        );
    }

    /// R stores {5, 6}; an incoming segment over {5} alone does not store all
    /// of it again, on either path.
    #[test]
    fn a_partial_superset_does_not_resupply() {
        let removed = vec![segment(&[5, 6])];
        let incoming = vec![segment(&[5])];

        for exclusion in [None, Some(CoverageExclusion::NoBatchRemapper)] {
            assert_eq!(
                resupply(&removed, &incoming, |_| exclusion),
                (vec![removed[0].uuid], false),
                "{exclusion:?}"
            );
        }
    }

    /// R stores {5, 6}; 6 was deleted outright and the history does not name
    /// it. An incoming segment over {5} supplies R again, as a member or by
    /// the exception.
    #[test]
    fn a_stored_fragment_off_the_history_and_not_live_is_ignored() {
        let removed = vec![segment(&[5, 6])];
        let incoming = vec![segment(&[5])];
        let traceable = |stored: &RoaringBitmap| stored - &RoaringBitmap::from_iter([6u32]);

        assert_eq!(
            resupply_tracing(&removed, &incoming, |_| None, traceable),
            (vec![], false)
        );
        assert_eq!(
            resupply_tracing(
                &removed,
                &incoming,
                |_| Some(CoverageExclusion::NoBatchRemapper),
                traceable
            ),
            (vec![], true)
        );
    }

    /// R stores only {6}, deleted outright off the history: nothing it stores
    /// needs supplying again.
    #[test]
    fn a_removed_segment_with_nothing_traceable_is_supplied_again() {
        let removed = vec![segment(&[6])];
        let incoming = vec![segment(&[5])];

        assert_eq!(
            resupply_tracing(&removed, &incoming, |_| None, |_| RoaringBitmap::new()),
            (vec![], false)
        );
    }

    /// Only a loss to another writer's commit is decided again, and only
    /// while an attempt is left: the last loss, a write that failed on its
    /// own, an unknown outcome and a success are returned.
    #[test]
    fn only_a_lost_version_with_an_attempt_left_is_decided_again() {
        let lost = || -> Result<()> {
            Err(ConcurrentVersion::conflict(
                3,
                "latest version advanced".to_owned(),
            ))
        };
        let budget = 3;
        assert!(decides_again(&lost(), 1, budget));
        assert!(decides_again(&lost(), 2, budget));
        assert!(!decides_again(&lost(), 3, budget));

        let failed_write = Err(Error::retryable_commit_conflict_source(
            3,
            Box::new(Error::io("manifest write failed")),
        ));
        assert!(!decides_again(&failed_write, 1, budget));
        let unknown = Err(Error::commit_status_unknown_source(
            3,
            "the outcome could not be read back".into(),
        ));
        assert!(!decides_again(&unknown, 1, budget));
        assert!(!decides_again(&Ok(()), 1, budget));
    }

    /// A retained segment that fails to open is refused with the failure
    /// classified as it was: this build's refusal stays `NotSupported` and
    /// names the segment, a read failure that caused the refusal is returned
    /// unchanged, and an unrelated open failure is not masked by it.
    #[test]
    fn a_retained_segment_that_cannot_be_opened_keeps_its_classification() {
        let segment = Uuid::new_v4();
        let refused = retained_segment_unopenable(
            "idx",
            segment,
            Error::not_supported("this file version is not readable"),
            None,
        );
        assert!(matches!(refused, Error::NotSupported { .. }), "{refused:?}");
        let message = refused.to_string();
        assert!(message.contains(&segment.to_string()), "{message}");
        assert!(message.contains("'idx'"), "{message}");
        assert!(
            message.contains("this file version is not readable"),
            "{message}"
        );
        assert!(
            std::error::Error::source(&refused)
                .and_then(|source| source.downcast_ref::<RetainedSegmentUnopenable>())
                .is_some_and(|unopenable| matches!(unopenable.source, Error::NotSupported { .. })),
            "{refused:?}"
        );

        let read_failed = retained_segment_unopenable(
            "idx",
            segment,
            Error::not_supported("format marker could not be read"),
            Some(Error::io("injected read failure")),
        );
        assert!(matches!(read_failed, Error::IO { .. }), "{read_failed:?}");
        assert!(
            read_failed.to_string().contains("injected read failure"),
            "{read_failed}"
        );

        let unrelated = retained_segment_unopenable(
            "idx",
            segment,
            Error::io("open timed out"),
            Some(Error::io("injected read failure")),
        );
        assert!(
            unrelated.to_string().contains("open timed out"),
            "{unrelated}"
        );

        let io = retained_segment_unopenable("idx", segment, Error::io("timed out"), None);
        assert!(matches!(io, Error::IO { .. }), "{io:?}");
        assert!(io.to_string().contains("timed out"), "{io}");
    }

    /// Incoming segments together supply a removed segment's sources; one
    /// excluded for a reason of its own does not count towards that.
    #[test]
    fn only_resupplying_incoming_segments_count_towards_a_removed_segment() {
        let removed = vec![segment(&[0, 2])];
        let incoming = vec![segment(&[0]), segment(&[2])];
        let excluded = incoming[1].uuid;

        assert_eq!(resupply(&removed, &incoming, |_| None), (vec![], false));
        assert_eq!(
            resupply(
                &removed,
                &incoming,
                classified(&[(
                    excluded,
                    CoverageExclusion::Segment(MissingCoverageReason::Corrupt)
                )])
            ),
            (vec![removed[0].uuid], false)
        );
    }

    /// Removed M takes part and is supplied by incoming member J; removed S,
    /// left out for the missing batch remapper, is supplied by J and K of the
    /// same class. Each removal is decided on its own.
    #[test]
    fn mixed_removals_are_decided_per_removed_segment() {
        let m = segment(&[12]);
        let s = segment(&[10]);
        let s_uuid = s.uuid;
        let removed = vec![m, s];
        let j = segment(&[12]);
        let k = segment(&[10]);
        let k_uuid = k.uuid;
        let classes = [
            (s_uuid, CoverageExclusion::NoBatchRemapper),
            (k_uuid, CoverageExclusion::NoBatchRemapper),
        ];

        assert_eq!(
            resupply(&removed, &[j, k.clone()], classified(&classes)),
            (vec![], true)
        );
        // Without J, M is offered only K's {10, 12}, which cannot stand in for
        // a member.
        let k_over_both = IndexMetadata {
            fragment_bitmap: Some(RoaringBitmap::from_iter([10u32, 12])),
            ..k
        };
        assert_eq!(
            resupply(&removed, &[k_over_both], classified(&classes)),
            (vec![removed[0].uuid], true)
        );
    }
}
