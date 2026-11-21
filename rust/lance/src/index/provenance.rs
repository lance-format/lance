// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build provenance and freshness checks for committed index coverage.
//!
//! This module owns how a segment's stored fragment coverage is traced to the
//! current snapshot, including staged replays, overlays, row revival, and
//! fragment-reuse remapping. Query-time masking lives in `segment_coverage`.

// These helpers stay crate-visible so callers do not depend on private parameter types.
#![allow(clippy::redundant_pub_crate)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use lance_core::datatypes::{Field, Schema as LanceSchema};
use lance_core::utils::deletion::DeletionVector;
use lance_index::frag_reuse::{CompactFragReuseIndex, FragReuseVersion};
use lance_index::metrics::NoOpMetricsCollector;
use lance_table::format::overlay::freshness::{
    OverlayFreshness, has_overlay_freshness, has_ranked_details,
};
use lance_table::format::overlay::staleness::{
    field_affects_index, fragment_has_stale_overlay, overlay_affects_index,
};
use lance_table::format::{DataFile, DeletionFile, Fragment, IndexMetadata, Manifest};
use roaring::RoaringBitmap;

use crate::dataset::transaction::{Operation, Transaction};
use crate::index::api::{IndexSegment, IntoIndexSegment};
use crate::index::append;
use crate::index::frag_reuse::{self, SegmentRemappingPlan};
use crate::index::loading::{DatasetIndexInternalExt, load_all_indices};
use crate::io::deletion::read_dataset_deletion_file;
use crate::{Error, Result, dataset::Dataset};

/// Move a caller-defined segment group's coverage into the current fragment space.
///
/// A deferred compaction can combine several independently built segments into one
/// new fragment. Remapping each segment bitmap separately would treat every segment
/// as only partially covering the rewrite group and drop the new fragment. The
/// merge owns the whole caller-defined group, so remap its union and use that
/// representable group coverage while materializing every source.
///
/// Returns the fragments the remap added beyond the segments' own coverage (on
/// a tagged history, beyond their provenance), or `None` when the coverage was
/// left as staged. Coverage only moves with the row addresses the reuse index
/// maps, and is kept only on fragments whose indexed data is shown unchanged
/// since the segments were built, compacted or not; other rows fall back to a
/// flat scan.
pub(super) async fn remap_merged_segment_coverage(
    dataset: &Dataset,
    index_name: &str,
    segments: &mut [IndexMetadata],
    staged: Option<&frag_reuse::StagedRemappingPlans>,
) -> Result<Option<RoaringBitmap>> {
    // Under a tagged history the merge opens each source through the
    // translating loader with its staged plan, so the merged files hold live
    // addresses; the coverage moves with them to the live fragments the group
    // serves, as derived by the reader's own planning of the group. Nothing
    // is claimed that the group cannot serve: a destination the group only
    // partly covers is absent from that coverage and stays scanned, and a
    // provenance bitmap left in place would be pruned to nothing at commit.
    // The v0 handle is never consulted on a tagged table.
    if let Some(plans) = staged {
        let mut claimed = RoaringBitmap::new();
        let mut provenance = RoaringBitmap::new();
        for segment in segments.iter_mut() {
            if let Some(bitmap) = &segment.fragment_bitmap {
                provenance |= bitmap;
            }
            let coverage = match plans.get(&segment.uuid) {
                Some(SegmentRemappingPlan::Translate { coverage, .. }) => coverage.clone(),
                Some(SegmentRemappingPlan::Identity) => segment
                    .fragment_bitmap
                    .as_ref()
                    .map(|bitmap| bitmap & dataset.fragment_bitmap.as_ref())
                    .unwrap_or_default(),
                Some(SegmentRemappingPlan::MissingCoverage(_)) | None => {
                    return Err(Error::not_supported(format!(
                        "FRI query coverage is unavailable for staged segment {}",
                        segment.uuid
                    )));
                }
            };
            claimed |= &coverage;
            segment.fragment_bitmap = Some(coverage);
        }
        if claimed.is_empty() {
            tracing::warn!(
                index_name,
                "Merged index covers no rows under the tagged fragment reuse history: its \
                 segments together cover no rewrite destination completely, so nothing is \
                 claimed and those rows stay on the scan path. Rebuild the index to cover them."
            );
        }
        return Ok(Some(&claimed - &provenance));
    }
    let staged_coverage = segments
        .iter()
        .map(|segment| {
            segment.fragment_bitmap.as_ref().cloned().ok_or_else(|| {
                Error::invalid_input(format!(
                    "CreateIndex: segment {} is missing fragment coverage",
                    segment.uuid
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .fold(RoaringBitmap::new(), |coverage, segment| coverage | segment);

    let frag_reuse_index = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await?;
    let has_fragment_reuse_index = frag_reuse_index.is_some();
    // Partly retired counts as stale: the retired half is about to be
    // intersected away, which drops its rows from the merged index.
    let coverage_is_stale = !staged_coverage.is_subset(&dataset.fragment_bitmap);
    let Some(frag_reuse_index) = frag_reuse_index
        .filter(|index| append::fragment_reuse_affects_segments(index, segments.iter()))
    else {
        // Nothing to remap through, so the retired fragments are intersected away
        // and the merged index will not cover their rows. It cannot claim them
        // either: moving coverage without the row addresses, which
        // `open_scalar_index` derives from the dataset's own mapping, would leave
        // an index asserting coverage it cannot serve and suppress the scan
        // fallback those rows need. So report and let the coverage shrink.
        if coverage_is_stale {
            tracing::warn!(
                index_name,
                staged_fragments = staged_coverage.len(),
                has_fragment_reuse_index,
                "Merging index segments over retired fragments with no applicable reuse \
                 mapping: the merged index will not cover their rows, which fall back to \
                 a flat scan. Rebuild the index to cover them."
            );
        }
        return Ok(None);
    };

    let merged_coverage = carry_coverage(
        dataset,
        index_name,
        &frag_reuse_index,
        segments,
        &staged_coverage,
    )
    .await?;

    // Fragments the remap added: compaction created them after the segments were built.
    let introduced = &merged_coverage - &staged_coverage;

    for segment in segments {
        segment.fragment_bitmap = Some(merged_coverage.clone());
    }
    Ok(Some(introduced))
}

/// `staged_coverage` moved through `frag_reuse_index` onto `dataset`'s
/// fragments. Coverage is kept, through compactions or not, only on fragments
/// whose indexed data is shown unchanged since the segments were built.
async fn carry_coverage<T: TableVersion>(
    dataset: &T,
    index_name: &str,
    frag_reuse_index: &CompactFragReuseIndex,
    segments: &[IndexMetadata],
    staged_coverage: &RoaringBitmap,
) -> Result<RoaringBitmap> {
    let fresh = fresh_fragments(dataset, frag_reuse_index, segments).await?;
    let live = dataset
        .manifest()
        .fragments
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect::<RoaringBitmap>();

    let mut merged_coverage = staged_coverage.clone();
    frag_reuse_index.remap_fragment_bitmap(&mut merged_coverage)?;

    // An applicable mapping can still be short of what these segments need.
    // Fragments compacted more than once need every link in the chain, and a trim
    // that dropped an earlier link leaves the remap on an intermediate fragment
    // the dataset no longer has. The intersect below would remove it silently.
    let unmapped = &merged_coverage - &live;
    if !unmapped.is_empty() {
        tracing::warn!(
            index_name,
            unmapped_fragments = unmapped.len(),
            "Merged index will not cover rows whose fragments the reuse history only \
             partly maps: a link in their rewrite chain is missing. Rebuild the index \
             to cover those rows."
        );
    }

    merged_coverage &= &live;

    let unverified = &merged_coverage - &fresh;
    if !unverified.is_empty() {
        merged_coverage -= &unverified;
        tracing::warn!(
            index_name,
            unverified_fragments = unverified.len(),
            "Merged index will not cover fragments whose indexed data changed since the \
             segments were built, or whose history to show otherwise has been cleaned up: \
             those rows fall back to a flat scan. Rebuild the index to cover them."
        );
    }
    if merged_coverage.is_empty() {
        tracing::warn!(
            index_name,
            staged_fragments = staged_coverage.len(),
            "Merged index covers no rows: none of the fragments its segments cover could \
             be carried to the current version (see the warnings above, or the remapper's \
             report of a rewrite group the segments only partly cover)."
        );
    }

    Ok(merged_coverage)
}

fn collect_subtree_field_ids(field: &Field, field_ids: &mut HashSet<i32>) {
    field_ids.insert(field.id);
    for child in &field.children {
        collect_subtree_field_ids(child, field_ids);
    }
}

/// Stable identity fields for a physical data file.
///
/// This mirrors transaction rewrite validation, additionally resolves a
/// registered base to its physical binding, and deliberately excludes
/// `file_size_bytes`, which is a mutable cache rather than file identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicalBaseBinding<'a> {
    Primary,
    Registered {
        path: &'a str,
        is_dataset_root: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PhysicalDataFileIdentity<'a> {
    base_id: Option<u32>,
    base_binding: PhysicalBaseBinding<'a>,
    path: &'a str,
    /// The column holding the field this identity is for.
    column_index: Option<i32>,
    file_major_version: u32,
    file_minor_version: u32,
}

impl<'a> PhysicalDataFileIdentity<'a> {
    fn try_new(manifest: &'a Manifest, file: &'a DataFile, field_position: usize) -> Option<Self> {
        let base_binding = match file.base_id {
            Some(base_id) => {
                let base = manifest.base_paths.get(&base_id)?;
                PhysicalBaseBinding::Registered {
                    path: &base.path,
                    is_dataset_root: base.is_dataset_root,
                }
            }
            None => PhysicalBaseBinding::Primary,
        };
        Some(Self {
            base_id: file.base_id,
            base_binding,
            path: &file.path,
            column_index: file.column_indices.get(field_position).copied(),
            file_major_version: file.file_major_version,
            file_minor_version: file.file_minor_version,
        })
    }
}

/// Where each field `indexed_field_ids` depend on is stored in `fragment`:
/// the indexed fields themselves, and a packed struct holding one. Other
/// fields sharing a file do not matter.
fn fragment_field_files<'a>(
    manifest: &'a Manifest,
    fragment: &'a Fragment,
    indexed_field_ids: &HashSet<i32>,
) -> Option<HashMap<i32, PhysicalDataFileIdentity<'a>>> {
    let indexed = indexed_field_ids.iter().copied().collect::<Vec<_>>();
    fragment
        .files
        .iter()
        .flat_map(|file| {
            file.fields
                .iter()
                .enumerate()
                .filter(|(_, field_id)| field_affects_index(**field_id, &indexed, &manifest.schema))
                .map(|(position, field_id)| {
                    PhysicalDataFileIdentity::try_new(manifest, file, position)
                        .map(|identity| (*field_id, identity))
                })
        })
        .collect()
}

/// Whether `fragment` has an overlay on an `indexed` field committed after `version`.
fn has_overlay_newer_than(
    fragment: &Fragment,
    version: u64,
    indexed: &HashSet<i32>,
    schema: &LanceSchema,
) -> bool {
    let indexed = indexed.iter().copied().collect::<Vec<_>>();
    fragment
        .overlays
        .iter()
        .any(|overlay| overlay_affects_index(overlay, &indexed, version, schema))
}

/// One version of the table, as the coverage check reads it. A [`Dataset`]
/// reads its history from storage; unit tests describe one in memory.
pub(crate) trait TableVersion: Clone {
    fn manifest(&self) -> &Manifest;

    /// Every version the table still lists, ascending.
    async fn retained_versions(&self) -> Result<Vec<u64>>;

    /// The table at `version`, which [`Self::retained_versions`] lists.
    async fn checkout(&self, version: u64) -> Result<Self>;

    /// The fragments written by the rewrite committed at `version`, or `None`
    /// when that commit is not a rewrite.
    async fn rewrite_outputs(&self, version: u64) -> Result<Option<HashSet<u32>>>;

    /// The rows of `fragment` that `deletion_file`, as this version names it, deletes.
    async fn deleted_rows(
        &self,
        fragment: &Fragment,
        deletion_file: &DeletionFile,
    ) -> Result<Arc<DeletionVector>>;
}

impl TableVersion for Dataset {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    async fn retained_versions(&self) -> Result<Vec<u64>> {
        Ok(self
            .version_refs()
            .await?
            .iter()
            .map(|version| version.version)
            .collect())
    }

    async fn checkout(&self, version: u64) -> Result<Self> {
        self.checkout_version(version).await
    }

    async fn rewrite_outputs(&self, version: u64) -> Result<Option<HashSet<u32>>> {
        let Some(transaction) = self.read_transaction_by_version(version).await? else {
            return Ok(None);
        };
        let Operation::Rewrite { groups, .. } = &transaction.operation else {
            return Ok(None);
        };
        Ok(Some(
            groups
                .iter()
                .flat_map(|group| group.new_fragments.iter())
                .map(|fragment| fragment.id as u32)
                .collect(),
        ))
    }

    async fn deleted_rows(
        &self,
        fragment: &Fragment,
        deletion_file: &DeletionFile,
    ) -> Result<Arc<DeletionVector>> {
        read_dataset_deletion_file(self, fragment.id, deletion_file).await
    }
}

/// A table version with its fragments indexed by id.
struct Snapshot<T> {
    table: T,
    fragments: HashMap<u32, usize>,
}

impl<T: TableVersion> Snapshot<T> {
    fn new(table: T) -> Self {
        let fragments = table
            .manifest()
            .fragments
            .iter()
            .enumerate()
            .map(|(position, fragment)| (fragment.id as u32, position))
            .collect();
        Self { table, fragments }
    }

    fn manifest(&self) -> &Manifest {
        self.table.manifest()
    }

    fn fragment(&self, id: u32) -> Option<&Fragment> {
        self.fragments
            .get(&id)
            .map(|position| &self.manifest().fragments[*position])
    }
}

/// Table versions read by the coverage check, each opened once.
///
/// A version is gone only if the table no longer lists it; any other read
/// failure is an error, so a transient fault cannot drop coverage for good.
struct History<'a, T> {
    table: &'a T,
    /// Every version the table still lists, ascending.
    retained: Vec<u64>,
    snapshots: HashMap<u64, Arc<Snapshot<T>>>,
}

impl<'a, T: TableVersion> History<'a, T> {
    async fn new(table: &'a T) -> Result<Self> {
        // Ids only; manifests are opened on demand.
        let retained = table.retained_versions().await?;
        let mut snapshots = HashMap::new();
        snapshots.insert(
            table.manifest().version,
            Arc::new(Snapshot::new(table.clone())),
        );
        Ok(Self {
            table,
            retained,
            snapshots,
        })
    }

    fn current(&self) -> Arc<Snapshot<T>> {
        self.snapshots[&self.table.manifest().version].clone()
    }

    /// The table at `version`, or `None` once cleanup has removed it.
    async fn at(&mut self, version: u64) -> Result<Option<Arc<Snapshot<T>>>> {
        if let Some(snapshot) = self.snapshots.get(&version) {
            return Ok(Some(snapshot.clone()));
        }
        if self.retained.binary_search(&version).is_err() {
            return Ok(None);
        }
        let table = self.table.checkout(version).await.map_err(|error| {
            Error::io(format!(
                "merge_existing_index_segments: cannot read dataset version {version} to \
                 check merged index coverage: {error}"
            ))
        })?;
        let snapshot = Arc::new(Snapshot::new(table));
        self.snapshots.insert(version, snapshot.clone());
        Ok(Some(snapshot))
    }

    /// The version that committed the compaction the reuse index records at
    /// `recorded`, or `None` unless that commit is a rewrite that produced
    /// `produced`. The commit is `recorded + 1` because conflict resolution
    /// restamps the record on every attempt; a record that was not restamped
    /// fails the check and costs only coverage.
    async fn rewrite_commit(
        &self,
        recorded: u64,
        mut produced: impl Iterator<Item = u32>,
    ) -> Result<Option<u64>> {
        let committed = recorded + 1;
        if self.retained.binary_search(&committed).is_err() {
            return Ok(None);
        }
        let Some(written) = self.table.rewrite_outputs(committed).await? else {
            return Ok(None);
        };
        Ok(produced
            .all(|fragment| written.contains(&fragment))
            .then_some(committed))
    }
}

/// The overlays on `fragment` over `indexed` fields, as (version, base, path).
fn indexed_overlays<'a>(
    fragment: &'a Fragment,
    indexed: &HashSet<i32>,
    schema: &LanceSchema,
) -> HashSet<(u64, Option<u32>, &'a str)> {
    let indexed = indexed.iter().copied().collect::<Vec<_>>();
    fragment
        .overlays
        .iter()
        .filter(|overlay| {
            overlay
                .data_file
                .fields
                .iter()
                .any(|field_id| field_affects_index(*field_id, &indexed, schema))
        })
        .map(|overlay| {
            (
                overlay.committed_version,
                overlay.data_file.base_id,
                overlay.data_file.path.as_str(),
            )
        })
        .collect()
}

/// Whether a row deleted in `before` is live again in `after` (only a restore
/// does this; an index built in between never saw the row).
pub(crate) async fn revives_rows<T: TableVersion>(
    before: &T,
    after: &T,
    then: &Fragment,
    now: &Fragment,
) -> Result<bool> {
    let Some(deleted_then) = &then.deletion_file else {
        return Ok(false);
    };
    let Some(deleted_now) = &now.deletion_file else {
        return Ok(true);
    };
    if deleted_now.id == deleted_then.id && deleted_now.read_version == deleted_then.read_version {
        return Ok(false);
    }
    let rows_then = before.deleted_rows(then, deleted_then).await?;
    let rows_now = after.deleted_rows(now, deleted_now).await?;
    Ok(rows_then.iter().any(|row| !rows_now.contains(row)))
}

/// Whether `fragment`'s indexed data differs between `before` and `after`: a
/// different file or overlay set for the indexed fields, or a revived row.
async fn indexed_data_differs<T: TableVersion>(
    before: &Snapshot<T>,
    after: &Snapshot<T>,
    fragment: u32,
    indexed: &HashSet<i32>,
) -> Result<bool> {
    let (Some(then), Some(now)) = (before.fragment(fragment), after.fragment(fragment)) else {
        return Ok(true);
    };
    let files_then = fragment_field_files(before.manifest(), then, indexed);
    let schema = &after.manifest().schema;
    if files_then.is_none()
        || files_then != fragment_field_files(after.manifest(), now, indexed)
        || indexed_overlays(then, indexed, schema) != indexed_overlays(now, indexed, schema)
    {
        return Ok(true);
    }
    revives_rows(&before.table, &after.table, then, now).await
}

/// The fragments of a segment's build snapshot that the recorded rewrites
/// chained into `fragment`, which compaction produced after the build at
/// `version`: each source the snapshot does not hold is followed back to the
/// group, recorded at or after `version`, that produced it. `None` when a link
/// is missing, a source neither the snapshot holds nor a recorded group
/// produced, so the segment's rows there cannot be traced to the build.
///
/// A group that produced several fragments gives them the same sources, so
/// repeated compactions reach a fragment along many paths: each fragment is
/// followed once, which keeps the work within the fragments the history names.
fn sources_at_build(
    versions: &[FragReuseVersion],
    version: u64,
    fragment: u32,
    is_built: impl Fn(u32) -> bool,
) -> Option<Vec<u32>> {
    let mut sources = Vec::new();
    let mut visited = RoaringBitmap::new();
    let mut pending = vec![fragment];
    while let Some(fragment) = pending.pop() {
        if !visited.insert(fragment) {
            continue;
        }
        if is_built(fragment) {
            sources.push(fragment);
            continue;
        }
        let group = versions
            .iter()
            .filter(|reuse| reuse.dataset_version >= version)
            .flat_map(|reuse| reuse.groups.iter())
            .find(|group| {
                group
                    .new_frags
                    .iter()
                    .any(|produced| produced.id as u32 == fragment)
            })?;
        pending.extend(group.old_frags.iter().map(|source| source.id as u32));
    }
    Some(sources)
}

/// The versions each covered fragment's index entries describe: its segments'
/// versions, or for a fragment compactions produced after the build, the
/// fragments of the segment's version that they rewrote into it, through every
/// recorded rewrite in between ([`sources_at_build`]). Every contributing build
/// must be checked, since they may have indexed different data. `None` records
/// a contribution whose build version has been cleaned up, or whose chain of
/// rewrites is not fully recorded.
async fn build_provenance<T: TableVersion>(
    history: &mut History<'_, T>,
    frag_reuse_index: &CompactFragReuseIndex,
    segments: &[IndexMetadata],
) -> Result<HashMap<u32, Vec<Option<u64>>>> {
    let mut provenance = HashMap::<u32, Vec<Option<u64>>>::new();
    for segment in segments {
        let freshness = OverlayFreshness::read(segment)?;
        let validation = freshness
            .validation_version()
            .filter(|&v| v <= history.table.manifest().version);
        let original = history
            .at(validation.unwrap_or(segment.dataset_version))
            .await?;
        let fields = segment.dependency_fields();
        // The query listing may already report compaction destinations. Overlay
        // facts retain the physical source fragment, even when it was appended
        // after the segment-wide build version.
        let is_source = |id| {
            freshness.for_fragment(id).is_some()
                || original
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.fragment(id).is_some())
        };
        for fragment in segment.fragment_bitmap.iter().flatten() {
            let Some(sources) = sources_at_build(
                &frag_reuse_index.details.versions,
                segment.dataset_version,
                fragment,
                is_source,
            ) else {
                provenance.entry(fragment).or_default().push(None);
                continue;
            };
            for source in sources {
                let recorded = provenance.entry(source).or_default();
                if let Some(validation) = validation {
                    // A clone can endorse its files locally. An overlay still excluded
                    // at that snapshot cannot vouch for a later fold into base data.
                    let verified = original.as_ref().is_some_and(|snapshot| {
                        snapshot.fragment(source).is_some_and(|fragment| {
                            !fragment_has_stale_overlay(
                                fragment,
                                &fields,
                                segment.dataset_version,
                                &freshness,
                                &snapshot.manifest().schema,
                            )
                        })
                    });
                    recorded.push(verified.then_some(validation));
                    continue;
                }
                let versions = freshness.source_versions(source, &segment.dataset_version);
                if versions.is_empty() {
                    recorded.push(None);
                } else {
                    // fresh_fragments verifies each snapshot still contains this
                    // physical source before carrying its rows through a rewrite.
                    recorded.extend(versions.iter().copied().map(Some));
                }
            }
        }
    }
    for versions in provenance.values_mut() {
        versions.sort_unstable();
        versions.dedup();
    }
    Ok(provenance)
}

/// The current fragments holding rows the segments indexed whose indexed data
/// is shown unchanged since, following each covered fragment through the
/// compactions that rewrote it. A compaction's inputs and the final fragments
/// must each match the version their entries describe
/// ([`indexed_data_differs`]). A changed input, or a manifest the comparison
/// needs that was cleaned up, leaves out every fragment its group produced.
pub(crate) async fn fresh_fragments<T: TableVersion>(
    dataset: &T,
    frag_reuse_index: &CompactFragReuseIndex,
    segments: &[IndexMetadata],
) -> Result<RoaringBitmap> {
    let mut indexed = HashSet::new();
    for segment in segments {
        indexed.extend(indexed_field_ids(
            &dataset.manifest().schema,
            &segment.fields,
            &segment.covering_fields,
        )?);
    }
    let mut history = History::new(dataset).await?;
    // Every followed fragment's contributing versions; a `None` contribution
    // prevents it from being shown fresh.
    let mut following = build_provenance(&mut history, frag_reuse_index, segments).await?;

    let mut versions = frag_reuse_index.details.versions.iter().collect::<Vec<_>>();
    versions.sort_by_key(|version| version.dataset_version);

    for version in versions {
        let ours = version
            .groups
            .iter()
            .filter(|group| {
                group
                    .old_frags
                    .iter()
                    .any(|fragment| following.contains_key(&(fragment.id as u32)))
            })
            .collect::<Vec<_>>();
        if ours.is_empty() {
            continue;
        }
        let before_commit = history.at(version.dataset_version).await?;
        let produced = ours
            .iter()
            .flat_map(|group| group.new_frags.iter())
            .map(|fragment| fragment.id as u32);
        let committed = match before_commit {
            Some(_) => {
                history
                    .rewrite_commit(version.dataset_version, produced)
                    .await?
            }
            None => None,
        };
        for group in ours {
            let mut stale = committed.is_none();
            for old in &group.old_frags {
                let old = old.id as u32;
                let (Some(recorded_versions), Some(before_commit)) =
                    (following.remove(&old), &before_commit)
                else {
                    continue;
                };
                for recorded in recorded_versions {
                    stale |= match recorded {
                        None => true,
                        Some(recorded) => match history.at(recorded).await? {
                            None => true,
                            Some(against) => {
                                indexed_data_differs(&against, before_commit, old, &indexed).await?
                            }
                        },
                    };
                }
            }
            for new in &group.new_frags {
                let new = new.id as u32;
                // The merge can retain both directly indexed entries and entries
                // remapped from older inputs. Neither contribution can vouch for
                // the other's freshness, including when either is undated.
                let recorded = committed.filter(|_| !stale);
                let recorded_versions = following
                    .entry(new)
                    .or_insert_with(|| Vec::with_capacity(1));
                if !recorded_versions.contains(&recorded) {
                    recorded_versions.push(recorded);
                }
            }
        }
    }

    let current = history.current();
    let mut fresh = RoaringBitmap::new();
    for (fragment, recorded_versions) in following {
        // A fragment no longer in the manifest is dropped from coverage anyway.
        if current.fragment(fragment).is_none() {
            continue;
        }
        let mut differs = false;
        for recorded in recorded_versions {
            differs |= match recorded {
                None => true,
                Some(recorded) => match history.at(recorded).await? {
                    None => true,
                    Some(against) => {
                        indexed_data_differs(&against, &current, fragment, &indexed).await?
                    }
                },
            };
        }
        if !differs {
            fresh.insert(fragment);
        }
    }
    Ok(fresh)
}

/// Resolve the field ids a segment's staleness check must consider: the subtree of
/// every field the segment declares, keyed and carried alike (see
/// [`IndexSegment::fields`] and [`IndexSegment::covering_fields`]). A covered
/// segment's carried columns can go stale independently of its keyed column, so
/// checking only the keyed subtree would leave a fragment covered after a carried
/// column was rewritten, and the segment would answer with the obsolete value.
fn indexed_field_ids(
    schema: &LanceSchema,
    fields: &[i32],
    carried: &[i32],
) -> Result<HashSet<i32>> {
    let mut indexed_field_ids = HashSet::new();
    for field_id in fields.iter().chain(carried) {
        let field = schema.field_by_id(*field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: field id {field_id} does not exist in the current schema"
            ))
        })?;
        collect_subtree_field_ids(field, &mut indexed_field_ids);
    }
    Ok(indexed_field_ids)
}

/// The current snapshot already endorses exactly these physical files, their
/// build provenance and their scope. In particular a deep clone can endorse
/// a local segment whose source manifests belong to the original table.
fn same_ranked_source(current: &IndexMetadata, source: &IndexMetadata) -> bool {
    current.uuid == source.uuid
        && current.base_id == source.base_id
        && current.dataset_version == source.dataset_version
        && current.index_version == source.index_version
        && current.fields == source.fields
        && current.covering_fields == source.covering_fields
        && current.fragment_bitmap == source.fragment_bitmap
        && match (&current.index_details, &source.index_details) {
            (Some(left), Some(right)) => Arc::ptr_eq(left, right) || left == right,
            _ => false,
        }
}

/// Group physical coverage by the snapshots that can verify its values.
/// Fragments contributed by more than one build must pass every check.
fn coverage_by_build_version(
    coverage: &mut RoaringBitmap,
    segment_version: u64,
    facts: &OverlayFreshness,
    current_version: u64,
) -> BTreeMap<u64, RoaringBitmap> {
    if let Some(version) = facts.validation_version().filter(|&v| v <= current_version) {
        return BTreeMap::from([(version, coverage.clone())]);
    }
    let mut by_version = BTreeMap::<u64, RoaringBitmap>::new();
    let mut ordinary = coverage.clone();
    for entry in facts.entries() {
        if !ordinary.remove(entry.fragment_id) {
            continue;
        }
        if entry.source_versions.is_empty()
            || entry.source_versions.iter().any(|&v| v > current_version)
        {
            coverage.remove(entry.fragment_id);
            continue;
        }
        for &version in &entry.source_versions {
            by_version
                .entry(version)
                .or_default()
                .insert(entry.fragment_id);
        }
    }
    if !ordinary.is_empty() {
        *by_version.entry(segment_version).or_default() |= ordinary;
    }
    by_version
}

/// Check live staged inputs before folding their individual build snapshots into
/// one version. Retired fragments still belong to the fragment reuse planner.
pub(super) async fn prune_ranked_merge_sources(
    dataset: &Dataset,
    segments: &mut [IndexMetadata],
) -> Result<HashMap<uuid::Uuid, RoaringBitmap>> {
    let mut candidates = Vec::new();
    let mut positions = Vec::new();
    let mut withdrawn = HashMap::new();
    let indices = load_all_indices(dataset).await?;
    let tagged = indices
        .iter()
        .any(lance_table::system_index::frag_reuse::metadata::is_tagged);
    let mut published = None;
    for (position, source) in segments.iter().enumerate() {
        let (Some(details), Some(coverage)) = (&source.index_details, &source.fragment_bitmap)
        else {
            continue;
        };
        if !has_ranked_details(details)
            || (source.dataset_version == dataset.manifest.version
                && !has_overlay_freshness(details)?)
        {
            continue;
        }
        // The current manifest already owns these physical entries, including
        // those copied from a clone with a different version history.
        if published
            .get_or_insert_with(|| {
                indices
                    .iter()
                    .map(|i| (i.uuid, i))
                    .collect::<HashMap<_, _>>()
            })
            .get(&source.uuid)
            .is_some_and(|current| same_ranked_source(current, source))
        {
            continue;
        }
        if tagged && !has_overlay_freshness(details)? {
            // Ordinary source data already went through the tagged replay at
            // exactly its build version.
            continue;
        }
        let live = coverage & dataset.fragment_bitmap.as_ref();
        if live.is_empty() {
            continue;
        }
        // An unpublished external source has no matching local build history.
        if source.base_id.is_some() {
            withdrawn.insert(source.uuid, live);
            continue;
        }
        let mut candidate = source.clone();
        candidate.fragment_bitmap = Some(live.clone());
        candidates.push(candidate.into_index_segment()?);
        positions.push((position, live));
    }
    if !candidates.is_empty() {
        let check = if tagged {
            CoverageCheck::Replayed
        } else {
            CoverageCheck::Staged
        };
        prune_stale_segment_coverage(dataset, &mut candidates, check).await?;
    }
    for ((position, requested), checked) in positions.into_iter().zip(candidates) {
        let removed = requested - checked.fragment_bitmap();
        if !removed.is_empty() {
            withdrawn.insert(segments[position].uuid, removed);
        }
    }
    for source in segments {
        if let Some(removed) = withdrawn.get(&source.uuid)
            && let Some(coverage) = &mut source.fragment_bitmap
        {
            *coverage -= removed;
        }
    }
    Ok(withdrawn)
}

#[derive(Clone, Copy)]
pub(crate) enum CoverageCheck<'a> {
    /// Compare a staged source with its historical data files.
    Staged,
    /// Tagged replay checked file changes from each contributing build snapshot.
    Replayed,
    /// Materialize current data, allowing these fragments to be absent before compaction.
    Remapped(&'a RoaringBitmap),
}

/// Drops from each segment's coverage the fragments whose indexed data changed
/// since it was built. A tagged replay can already have validated physical
/// file changes, and a remap can introduce fragments absent from the build.
pub(crate) async fn prune_stale_segment_coverage(
    dataset: &Dataset,
    segments: &mut [IndexSegment],
    check: CoverageCheck<'_>,
) -> Result<()> {
    let remapped = match check {
        CoverageCheck::Remapped(fragments) => Some(fragments),
        _ => None,
    };
    let current_fragments = dataset
        .fragments()
        .iter()
        .map(|fragment| (fragment.id as u32, fragment))
        .collect::<HashMap<_, _>>();
    // Under a tagged fragment reuse history a segment's bitmap is provenance:
    // a fragment it names that a recorded transition retired is not stale,
    // its rows translate to the transition's destinations when the segment
    // is opened. Such a fragment is on the entry's lineage (the sources and
    // destinations of every transition); a retired fragment off the lineage
    // (a bare rewrite of uncovered data, a whole-fragment delete) holds
    // nothing the segment can reach and is pruned as before.
    let indices = load_all_indices(dataset).await?;
    let mut published = None;
    let lineage = indices
        .iter()
        .find(|index| lance_table::system_index::frag_reuse::metadata::is_tagged(index))
        .and_then(|entry| entry.fragment_bitmap.clone());
    // Resolve an exceptional fragment against the builds that actually supplied
    // its entries. The oldest segment-wide version still owns its row addresses.
    let mut checks: BTreeMap<u64, Vec<(usize, Option<RoaringBitmap>)>> = BTreeMap::new();
    let mut freshness = Vec::with_capacity(segments.len());
    for (position, segment) in segments.iter_mut().enumerate() {
        if remapped.is_none()
            && has_ranked_details(segment.index_details())
            && (segment.dataset_version() < dataset.manifest.version
                || has_overlay_freshness(segment.index_details())?)
            && published
                .get_or_insert_with(|| {
                    indices
                        .iter()
                        .map(|i| (i.uuid, i))
                        .collect::<HashMap<_, _>>()
                })
                .get(&segment.uuid())
                .is_some_and(|stored| {
                    same_ranked_source(stored, &segment.clone().into_metadata(""))
                })
        {
            freshness.push(OverlayFreshness::default());
            continue;
        }
        let facts = OverlayFreshness::from_details(segment.uuid(), segment.index_details(), None)?;
        let version = segment.dataset_version();
        if facts.is_empty() {
            if version < dataset.manifest.version {
                checks.entry(version).or_default().push((position, None));
            }
        } else {
            for (version, fragments) in coverage_by_build_version(
                segment.fragment_bitmap_mut(),
                version,
                &facts,
                dataset.manifest.version,
            ) {
                checks
                    .entry(version)
                    .or_default()
                    .push((position, Some(fragments)));
            }
        }
        freshness.push(facts);
    }
    for (version, checks) in checks {
        let historical = match if version == dataset.manifest.version {
            Ok(dataset.clone())
        } else {
            dataset.checkout_version(version).await
        } {
            Ok(historical) => historical,
            Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                for (position, fragments) in checks {
                    let covered = segments[position].fragment_bitmap_mut();
                    if let Some(fragments) = fragments {
                        *covered -= fragments;
                    } else {
                        covered.clear();
                    }
                }
                continue;
            }
            Err(error) => {
                return Err(Error::invalid_input(format!(
                    "CreateIndex: cannot validate segment coverage built at dataset version \
                     {version}: {error}"
                )));
            }
        };
        let historical_fragments = historical
            .fragments()
            .iter()
            .map(|fragment| (fragment.id as u32, fragment))
            .collect::<HashMap<_, _>>();

        for (position, subset) in checks {
            let segment = &mut segments[position];
            let facts = &freshness[position];
            let files_verified = matches!(check, CoverageCheck::Replayed)
                && has_ranked_details(segment.index_details());
            let indexed_field_ids = indexed_field_ids(
                dataset.schema(),
                segment.fields(),
                segment.covering_fields(),
            )?;
            let mut stale_fragments = Vec::new();
            let fragments = subset.as_ref().unwrap_or_else(|| segment.fragment_bitmap());
            for fragment_id in fragments.iter() {
                let accepted = facts
                    .for_fragment(fragment_id)
                    .map_or(&[][..], |entry| entry.overlays.as_slice());
                let historical_fragment = historical_fragments.get(&fragment_id);
                if historical_fragment.is_none()
                    && remapped.is_none_or(|introduced| !introduced.contains(fragment_id))
                {
                    if remapped.is_some() || !accepted.is_empty() {
                        stale_fragments.push(fragment_id);
                    }
                    continue;
                }
                let Some(current_fragment) = current_fragments.get(&fragment_id) else {
                    if !lineage
                        .as_ref()
                        .is_some_and(|lineage| lineage.contains(fragment_id))
                    {
                        stale_fragments.push(fragment_id);
                    }
                    continue;
                };
                // A fragment the remap added may have no counterpart at
                // this version to compare; the overlay check still applies.
                if !files_verified && let Some(then) = historical_fragment {
                    let before =
                        fragment_field_files(&historical.manifest, then, &indexed_field_ids);
                    let after = fragment_field_files(
                        &dataset.manifest,
                        current_fragment,
                        &indexed_field_ids,
                    );
                    if before.is_none() || before != after {
                        stale_fragments.push(fragment_id);
                        continue;
                    }
                }
                // A restore can remove an incorporated overlay or bring back one
                // absent at build time. Newer unincorporated files are masked at query time.
                let historical_overlays = historical_fragment.filter(|f| !f.overlays.is_empty());
                if has_ranked_details(segment.index_details())
                    && (!accepted.is_empty()
                        || historical_overlays.is_some()
                        || !current_fragment.overlays.is_empty())
                {
                    let live =
                        indexed_overlays(current_fragment, &indexed_field_ids, dataset.schema());
                    let previous = historical_overlays
                        .map(|f| indexed_overlays(f, &indexed_field_ids, dataset.schema()))
                        .unwrap_or_default();
                    let missing_exception = accepted.iter().any(|file| {
                        let key = (file.committed_version, file.base_id, file.path.as_str());
                        !live.contains(&key) || !previous.contains(&key)
                    });
                    let changed_before_watermark = previous
                        .symmetric_difference(&live)
                        .any(|key| key.0 <= segment.dataset_version());
                    if missing_exception || changed_before_watermark {
                        stale_fragments.push(fragment_id);
                        continue;
                    }
                }
                let changed_overlays = remapped.is_some()
                    && has_overlay_newer_than(
                        current_fragment,
                        version,
                        &indexed_field_ids,
                        dataset.schema(),
                    );
                if changed_overlays {
                    stale_fragments.push(fragment_id);
                    continue;
                }
                if let Some(historical_fragment) = historical_fragment
                    && revives_rows(&historical, dataset, historical_fragment, current_fragment)
                        .await?
                {
                    stale_fragments.push(fragment_id);
                }
            }
            for fragment_id in stale_fragments {
                segment.fragment_bitmap_mut().remove(fragment_id);
            }
        }
    }
    Ok(())
}

pub(super) struct StagedReplay {
    pub(super) segments: Vec<IndexMetadata>,
    /// Present on a tagged history, where replay checked every intervening version.
    pub(super) fts_withdrawals: Option<HashMap<uuid::Uuid, RoaringBitmap>>,
}

/// Validate staged (uncommitted) segments against the current snapshot on a
/// table with a tagged fragment reuse history.
///
/// A staged segment was built at its own dataset version; every commit
/// since then may have moved its rows (a rewrite recording a transition:
/// the provenance stays and translates) or rewritten an indexed column in
/// place (the values it holds are stale: the transition's coverage is
/// withdrawn). Those are the conflict resolver's own rules for an index
/// committed from an old read version, so the segments are replayed through
/// it: a CreateIndex of the segments at the oldest build version, checked
/// against every transaction up to this snapshot. The returned segments
/// carry the corrected bitmaps; their addresses are still the ones they
/// were built with, which the merge translates. A rewrite the rules cannot
/// carry the segment across (a bare compaction of a fragment it covers, or a
/// group it covers only partly) means the segment must be rebuilt, as does a
/// version in the window that can no longer be read. The cost is
/// proportional to the commits crossed. On a table without a tagged history
/// nothing is replayed: coverage is reconciled when the segments commit.
pub(super) async fn replay_staged_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
) -> Result<StagedReplay> {
    let indices = load_all_indices(dataset).await?;
    let Some(tagged) = indices
        .iter()
        .find(|index| lance_table::system_index::frag_reuse::metadata::is_tagged(index))
    else {
        return Ok(StagedReplay {
            segments,
            fts_withdrawals: None,
        });
    };
    let mut original_fts: HashMap<_, _> = segments
        .iter()
        .filter(|segment| {
            segment.dataset_version < dataset.manifest.version
                && super::segment::segment_has_inverted_details(segment)
        })
        .filter_map(|segment| {
            Some((
                segment.uuid,
                segment.fragment_bitmap.as_ref()? & dataset.fragment_bitmap.as_ref(),
            ))
        })
        .collect();
    // One replay per build version: a segment is carried across the commits
    // it did not see, so a segment built after a rewrite is not withdrawn for
    // it again.
    let mut by_build_version: BTreeMap<u64, Vec<IndexMetadata>> = BTreeMap::new();
    let mut mixed = Vec::new();
    let mut input_order = HashMap::with_capacity(segments.len());
    for (position, segment) in segments.into_iter().enumerate() {
        if input_order.insert(segment.uuid, position).is_some() {
            return Err(Error::invalid_input(format!(
                "CreateIndex: duplicate segment uuid {} for index '{}'",
                segment.uuid, segment.name
            )));
        }
        if segment.dataset_version < dataset.manifest.version
            && let Some(details) = &segment.index_details
            && has_overlay_freshness(details)?
            && segment.fragment_bitmap.is_some()
        {
            mixed.push(segment);
        } else {
            by_build_version
                .entry(segment.dataset_version)
                .or_default()
                .push(segment);
        }
    }
    let has_mixed_sources = !mixed.is_empty();
    if has_mixed_sources {
        replay_mixed_segment_coverage(dataset, tagged, &indices, &mut mixed).await?;
    }
    let mut validated = mixed;
    for (read_version, group) in by_build_version {
        if read_version >= dataset.manifest.version {
            validated.extend(group);
        } else {
            validated.extend(replay_staged_segments_from(dataset, read_version, group).await?);
        }
    }
    if has_mixed_sources {
        // Match the ordinary replay order, including the caller's first model at each version.
        validated.sort_by_key(|segment| {
            (
                segment.dataset_version,
                input_order.get(&segment.uuid).copied(),
            )
        });
    }
    for segment in &validated {
        if let Some(original) = original_fts.get_mut(&segment.uuid)
            && let Some(coverage) = &segment.fragment_bitmap
        {
            *original -= coverage;
        }
    }
    original_fts.retain(|_, removed| !removed.is_empty());
    Ok(StagedReplay {
        segments: validated,
        fts_withdrawals: Some(original_fts),
    })
}

/// Replay each exceptional fragment from the snapshot that supplied its indexed values.
/// The segment-wide version continues to date its physical row addresses.
async fn replay_mixed_segment_coverage(
    dataset: &Dataset,
    tagged: &IndexMetadata,
    indices: &[IndexMetadata],
    segments: &mut [IndexMetadata],
) -> Result<()> {
    let live = dataset.fragment_bitmap.as_ref();
    let published = indices
        .iter()
        .map(|i| (i.uuid, i))
        .collect::<HashMap<_, _>>();
    let mut retired = Vec::with_capacity(segments.len());
    let mut builds = BTreeMap::<(u64, bool), Vec<(usize, RoaringBitmap)>>::new();
    for (position, segment) in segments.iter_mut().enumerate() {
        if published
            .get(&segment.uuid)
            .is_some_and(|stored| same_ranked_source(stored, segment))
        {
            // A clone can endorse these exact files without retaining the parent's snapshots.
            retired.push(RoaringBitmap::new());
            continue;
        }
        let facts = OverlayFreshness::read(segment)?;
        let coverage = segment.fragment_bitmap.as_mut().ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: segment {} lacks fragment coverage",
                segment.uuid
            ))
        })?;
        retired.push(&*coverage - live);
        for (version, subset) in coverage_by_build_version(
            coverage,
            segment.dataset_version,
            &facts,
            dataset.manifest.version,
        ) {
            if version >= dataset.manifest.version {
                continue;
            }
            // During a withdrawal the resolver can drop an entire physical file when
            // only live, untranslated rows remain. Keep live and retired slices apart
            // so that this decision is made once below, on the reassembled coverage.
            for (is_retired, fragments) in [(false, &subset & live), (true, &subset - live)] {
                if !fragments.is_empty() {
                    builds
                        .entry((version, is_retired))
                        .or_default()
                        .push((position, fragments));
                }
            }
        }
    }
    for ((version, _), group) in builds {
        let sources = group
            .iter()
            .map(|(position, fragments)| IndexMetadata {
                fragment_bitmap: Some(fragments.clone()),
                ..segments[*position].clone()
            })
            .collect();
        let surviving = replay_staged_segments_from(dataset, version, sources)
            .await?
            .into_iter()
            .filter_map(|source| Some((source.uuid, source.fragment_bitmap?)))
            .collect::<HashMap<_, _>>();
        // Several builds can contribute to the same fragment. A withdrawal from
        // any one of them must survive the replay of every other contribution.
        for (position, mut requested) in group {
            let segment = &mut segments[position];
            if let Some(verified) = surviving.get(&segment.uuid) {
                requested -= verified;
            }
            if let Some(coverage) = &mut segment.fragment_bitmap {
                *coverage -= requested;
            }
        }
    }
    for (segment, retired) in segments.iter_mut().zip(retired) {
        if let Some(coverage) = &mut segment.fragment_bitmap
            && !retired.is_subset(coverage)
            && !coverage.is_empty()
            && coverage.is_subset(live)
            && !super::frag_reuse_reader::FragmentReuseIndex::open(dataset, tagged)
                .await?
                .may_need_translation(Some(coverage))
        {
            // An identity reader would expose retired rows still stored in this file.
            coverage.clear();
        }
    }
    Ok(())
}

async fn replay_staged_segments_from(
    dataset: &Dataset,
    read_version: u64,
    segments: Vec<IndexMetadata>,
) -> Result<Vec<IndexMetadata>> {
    let rebuild = |version: u64, error: &Error| {
        Error::invalid_input(format!(
            "staged index segments built at dataset version {read_version} cannot be carried \
             across version {version}: {error}. Rebuild the segments from the current dataset"
        ))
    };
    let read_dataset = dataset
        .checkout_version(read_version)
        .await
        .map_err(|error| rebuild(read_version, &error))?;
    let transaction = Transaction::new(
        read_version,
        Operation::CreateIndex {
            new_indices: segments,
            removed_indices: vec![],
        },
        None,
    );
    let mut rebase = crate::io::commit::conflict_resolver::TransactionRebase::try_new(
        &read_dataset,
        transaction,
        None,
    )
    .await?
    .for_staged_replay();
    rebase.load_current_lineage(dataset).await?;
    let (_, transactions) = crate::io::commit::load_and_sort_new_transactions(&read_dataset)
        .await
        .map_err(|error| rebuild(read_version, &error))?;
    let snapshot = dataset.manifest.version;
    let window: Vec<_> = transactions
        .into_iter()
        .filter(|(version, _)| *version <= snapshot)
        .collect();
    // The listing only knows the manifests that still exist. A version
    // cleaned up between the build and the snapshot leaves no trace in it,
    // and whatever that commit did to the indexed columns cannot be
    // replayed: the history has to be complete, one manifest per version.
    let gone = || {
        Error::invalid_input(
            "its manifest is gone (cleaned up), so the commits since the build cannot be replayed",
        )
    };
    let mut expected = read_version + 1;
    for (version, _) in &window {
        if *version != expected {
            return Err(rebuild(expected, &gone()));
        }
        expected += 1;
    }
    if expected != snapshot + 1 {
        return Err(rebuild(expected, &gone()));
    }
    // The complete window distinguishes files rewritten with new values from
    // files a column repack merely moved.
    rebase.track_moved_files();
    for (version, transaction) in &window {
        rebase
            .check_txn(transaction, *version)
            .map_err(|error| match error {
                Error::RetryableCommitConflict { .. } | Error::CommitConflict { .. } => {
                    rebuild(*version, &error)
                }
                other => other,
            })?;
    }
    match rebase.finish(dataset).await?.operation {
        Operation::CreateIndex { new_indices, .. } => Ok(new_indices),
        _ => unreachable!("a CreateIndex rebase yields a CreateIndex"),
    }
}

#[cfg(test)]
mod tests;
