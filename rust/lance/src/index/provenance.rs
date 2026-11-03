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
use lance_index::frag_reuse::{CompactFragReuseIndex, FragReuseVersion};
use lance_index::metrics::NoOpMetricsCollector;
use lance_table::format::overlay::staleness::{field_affects_index, overlay_affects_index};
use lance_table::format::{DataFile, Fragment, IndexMetadata};
use roaring::RoaringBitmap;

use crate::dataset::transaction::{Operation, Transaction};
use crate::index::api::IndexSegment;
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

    // Coverage is kept, through compactions or not, only on fragments whose
    // indexed data is shown unchanged since the segments were built.
    let fresh = fresh_fragments(dataset, &frag_reuse_index, segments).await?;

    let mut merged_coverage = staged_coverage.clone();
    frag_reuse_index.remap_fragment_bitmap(&mut merged_coverage)?;

    // An applicable mapping can still be short of what these segments need.
    // Fragments compacted more than once need every link in the chain, and a trim
    // that dropped an earlier link leaves the remap on an intermediate fragment
    // the dataset no longer has. The intersect below would remove it silently.
    let unmapped = &merged_coverage - dataset.fragment_bitmap.as_ref();
    if !unmapped.is_empty() {
        tracing::warn!(
            index_name,
            unmapped_fragments = unmapped.len(),
            "Merged index will not cover rows whose fragments the reuse history only \
             partly maps: a link in their rewrite chain is missing. Rebuild the index \
             to cover those rows."
        );
    }

    merged_coverage &= dataset.fragment_bitmap.as_ref();

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

    // Fragments the remap added: compaction created them after the segments were built.
    let introduced = &merged_coverage - &staged_coverage;

    for segment in segments {
        segment.fragment_bitmap = Some(merged_coverage.clone());
    }
    Ok(Some(introduced))
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
    fn try_new(dataset: &'a Dataset, file: &'a DataFile, field_position: usize) -> Option<Self> {
        let base_binding = match file.base_id {
            Some(base_id) => {
                let base = dataset.manifest.base_paths.get(&base_id)?;
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
    dataset: &'a Dataset,
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
                .filter(|(_, field_id)| field_affects_index(**field_id, &indexed, dataset.schema()))
                .map(|(position, field_id)| {
                    PhysicalDataFileIdentity::try_new(dataset, file, position)
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

/// A dataset version with its fragments indexed by id.
struct Snapshot {
    dataset: Dataset,
    fragments: HashMap<u32, usize>,
}

impl Snapshot {
    fn new(dataset: Dataset) -> Self {
        let fragments = dataset
            .fragments()
            .iter()
            .enumerate()
            .map(|(position, fragment)| (fragment.id as u32, position))
            .collect();
        Self { dataset, fragments }
    }

    fn fragment(&self, id: u32) -> Option<&Fragment> {
        self.fragments
            .get(&id)
            .map(|position| &self.dataset.fragments()[*position])
    }
}

/// Dataset versions read by the coverage check, each opened once.
///
/// A version is gone only if the dataset no longer lists it; any other read
/// failure is an error, so a transient fault cannot drop coverage for good.
struct History<'a> {
    dataset: &'a Dataset,
    /// Every version the dataset still lists, ascending.
    retained: Vec<u64>,
    snapshots: HashMap<u64, Arc<Snapshot>>,
}

impl<'a> History<'a> {
    async fn new(dataset: &'a Dataset) -> Result<Self> {
        // Ids only; manifests are opened on demand.
        let retained = dataset
            .version_refs()
            .await?
            .iter()
            .map(|version| version.version)
            .collect();
        let mut snapshots = HashMap::new();
        snapshots.insert(
            dataset.manifest.version,
            Arc::new(Snapshot::new(dataset.clone())),
        );
        Ok(Self {
            dataset,
            retained,
            snapshots,
        })
    }

    fn current(&self) -> Arc<Snapshot> {
        self.snapshots[&self.dataset.manifest.version].clone()
    }

    /// The dataset at `version`, or `None` once cleanup has removed it.
    async fn at(&mut self, version: u64) -> Result<Option<Arc<Snapshot>>> {
        if let Some(snapshot) = self.snapshots.get(&version) {
            return Ok(Some(snapshot.clone()));
        }
        if self.retained.binary_search(&version).is_err() {
            return Ok(None);
        }
        let dataset = self
            .dataset
            .checkout_version(version)
            .await
            .map_err(|error| {
                Error::io(format!(
                    "merge_existing_index_segments: cannot read dataset version {version} to \
                     check merged index coverage: {error}"
                ))
            })?;
        let snapshot = Arc::new(Snapshot::new(dataset));
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
        produced: impl Iterator<Item = u32>,
    ) -> Result<Option<u64>> {
        let committed = recorded + 1;
        if self.retained.binary_search(&committed).is_err() {
            return Ok(None);
        }
        let Some(transaction) = self.dataset.read_transaction_by_version(committed).await? else {
            return Ok(None);
        };
        let Operation::Rewrite { groups, .. } = &transaction.operation else {
            return Ok(None);
        };
        let written = groups
            .iter()
            .flat_map(|group| group.new_fragments.iter())
            .map(|fragment| fragment.id as u32)
            .collect::<HashSet<_>>();
        Ok(produced
            .into_iter()
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
async fn revives_rows(
    before: &Snapshot,
    after: &Snapshot,
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
    let rows_then = read_dataset_deletion_file(&before.dataset, then.id, deleted_then).await?;
    let rows_now = read_dataset_deletion_file(&after.dataset, now.id, deleted_now).await?;
    Ok(rows_then.iter().any(|row| !rows_now.contains(row)))
}

/// Whether `fragment`'s indexed data differs between `before` and `after`: a
/// different file or overlay set for the indexed fields, or a revived row.
async fn indexed_data_differs(
    before: &Snapshot,
    after: &Snapshot,
    fragment: u32,
    indexed: &HashSet<i32>,
) -> Result<bool> {
    let (Some(then), Some(now)) = (before.fragment(fragment), after.fragment(fragment)) else {
        return Ok(true);
    };
    let files_then = fragment_field_files(&before.dataset, then, indexed);
    if files_then.is_none()
        || files_then != fragment_field_files(&after.dataset, now, indexed)
        || indexed_overlays(then, indexed, after.dataset.schema())
            != indexed_overlays(now, indexed, after.dataset.schema())
    {
        return Ok(true);
    }
    revives_rows(before, after, then, now).await
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

/// The version each covered fragment's index entries describe: its segment's
/// version, or for a fragment compactions produced after the build, the
/// fragments of the segment's version that they rewrote into it, through every
/// recorded rewrite in between ([`sources_at_build`]). `None` for a fragment
/// whose segment's version has been cleaned up, or whose chain of rewrites is
/// not fully recorded.
async fn build_provenance(
    history: &mut History<'_>,
    frag_reuse_index: &CompactFragReuseIndex,
    segments: &[IndexMetadata],
) -> Result<HashMap<u32, Option<u64>>> {
    let mut by_version = segments.iter().collect::<Vec<_>>();
    // Oldest first: a fragment reached through two segments is checked
    // against the older one.
    by_version.sort_by_key(|segment| segment.dataset_version);

    let mut provenance = HashMap::new();
    for segment in by_version {
        let version = segment.dataset_version;
        let built_at = history.at(version).await?;
        for fragment in segment.fragment_bitmap.iter().flatten() {
            let Some(built_at) = &built_at else {
                provenance.entry(fragment).or_insert(None);
                continue;
            };
            let sources = sources_at_build(
                &frag_reuse_index.details.versions,
                version,
                fragment,
                |id| built_at.fragment(id).is_some(),
            );
            match sources {
                Some(sources) => {
                    for source in sources {
                        provenance.entry(source).or_insert(Some(version));
                    }
                }
                None => {
                    provenance.entry(fragment).or_insert(None);
                }
            }
        }
    }
    Ok(provenance)
}

/// The current fragments holding rows the segments indexed whose indexed data
/// is shown unchanged since, following each covered fragment through the
/// compactions that rewrote it. A compaction's inputs and the final fragments
/// must each match the version their entries describe
/// ([`indexed_data_differs`]). A changed input, or a manifest the comparison
/// needs that was cleaned up, leaves out every fragment its group produced.
pub(crate) async fn fresh_fragments(
    dataset: &Dataset,
    frag_reuse_index: &CompactFragReuseIndex,
    segments: &[IndexMetadata],
) -> Result<RoaringBitmap> {
    let mut indexed = HashSet::new();
    for segment in segments {
        indexed.extend(indexed_field_ids(
            dataset,
            &segment.fields,
            &segment.covering_fields,
        )?);
    }
    let mut history = History::new(dataset).await?;
    // Each followed fragment's version, or `None` once it cannot be shown fresh.
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
                let (Some(recorded), Some(before_commit)) =
                    (following.remove(&old), &before_commit)
                else {
                    continue;
                };
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
            for new in &group.new_frags {
                let new = new.id as u32;
                // A covered fragment whose entries could not be dated stays
                // out, though another covered fragment's sources lead to the
                // group that produced it: those vouch for none of the group's
                // other inputs.
                if following.get(&new) != Some(&None) {
                    following.insert(new, committed.filter(|_| !stale));
                }
            }
        }
    }

    let current = history.current();
    let mut fresh = RoaringBitmap::new();
    for (fragment, recorded) in following {
        // A fragment no longer in the manifest is dropped from coverage anyway.
        if current.fragment(fragment).is_none() {
            continue;
        }
        let differs = match recorded {
            None => true,
            Some(recorded) => match history.at(recorded).await? {
                None => true,
                Some(against) => {
                    indexed_data_differs(&against, &current, fragment, &indexed).await?
                }
            },
        };
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
fn indexed_field_ids(dataset: &Dataset, fields: &[i32], carried: &[i32]) -> Result<HashSet<i32>> {
    let mut indexed_field_ids = HashSet::new();
    for field_id in fields.iter().chain(carried) {
        let field = dataset.schema().field_by_id(*field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "CreateIndex: field id {field_id} does not exist in the current schema"
            ))
        })?;
        collect_subtree_field_ids(field, &mut indexed_field_ids);
    }
    Ok(indexed_field_ids)
}

/// Drops from each segment's coverage the fragments whose indexed data changed
/// since it was built. `historically_missing_exempt` names fragments compaction
/// created after the segments were built, whose absence from a segment's
/// history is expected rather than stale.
pub(crate) async fn prune_stale_segment_coverage(
    dataset: &Dataset,
    segments: &mut [IndexSegment],
    prune_historically_missing: bool,
    historically_missing_exempt: &RoaringBitmap,
    prune_newer_overlays: bool,
) -> Result<()> {
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
    let lineage = load_all_indices(dataset)
        .await?
        .iter()
        .find(|index| lance_table::system_index::frag_reuse::metadata::is_tagged(index))
        .and_then(|entry| entry.fragment_bitmap.clone());
    let historical_versions = segments
        .iter()
        .map(IndexSegment::dataset_version)
        .filter(|version| *version < dataset.manifest.version)
        .collect::<HashSet<_>>();

    for version in historical_versions {
        let historical = match dataset.checkout_version(version).await {
            Ok(historical) => historical,
            // Cleaned up: nothing the segments built there cover can be shown
            // unchanged, so they cover nothing and those rows are scanned.
            Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                for segment in segments
                    .iter_mut()
                    .filter(|segment| segment.dataset_version() == version)
                {
                    segment.fragment_bitmap_mut().clear();
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

        for segment in segments
            .iter_mut()
            .filter(|segment| segment.dataset_version() == version)
        {
            let indexed_field_ids =
                indexed_field_ids(dataset, segment.fields(), segment.covering_fields())?;
            let stale_fragments = segment
                .fragment_bitmap()
                .iter()
                .filter(|fragment_id| {
                    let historical_fragment = historical_fragments.get(fragment_id);
                    if historical_fragment.is_none()
                        && !historically_missing_exempt.contains(*fragment_id)
                    {
                        return prune_historically_missing;
                    }
                    let Some(current_fragment) = current_fragments.get(fragment_id) else {
                        return !lineage
                            .as_ref()
                            .is_some_and(|lineage| lineage.contains(*fragment_id));
                    };
                    // A fragment the remap added may have no counterpart at
                    // this version to compare; the overlay check still applies.
                    let changed_files = historical_fragment.is_some_and(|historical_fragment| {
                        let historical_files = fragment_field_files(
                            &historical,
                            historical_fragment,
                            &indexed_field_ids,
                        );
                        let current_files =
                            fragment_field_files(dataset, current_fragment, &indexed_field_ids);
                        historical_files.is_none() || historical_files != current_files
                    });
                    let changed_overlays = prune_newer_overlays
                        && has_overlay_newer_than(
                            current_fragment,
                            version,
                            &indexed_field_ids,
                            dataset.schema(),
                        );
                    changed_files || changed_overlays
                })
                .collect::<Vec<_>>();
            for fragment_id in stale_fragments {
                segment.fragment_bitmap_mut().remove(fragment_id);
            }
        }
    }
    Ok(())
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
pub(crate) async fn replay_staged_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
) -> Result<Vec<IndexMetadata>> {
    let tagged = load_all_indices(dataset)
        .await?
        .iter()
        .any(lance_table::system_index::frag_reuse::metadata::is_tagged);
    if !tagged {
        return Ok(segments);
    }
    // One replay per build version: a segment is carried across the commits
    // it did not see, so a segment built after a rewrite is not withdrawn for
    // it again.
    let mut by_build_version: BTreeMap<u64, Vec<IndexMetadata>> = BTreeMap::new();
    for segment in segments {
        by_build_version
            .entry(segment.dataset_version)
            .or_default()
            .push(segment);
    }
    let mut validated = Vec::new();
    for (read_version, group) in by_build_version {
        if read_version >= dataset.manifest.version {
            validated.extend(group);
        } else {
            validated.extend(replay_staged_segments_from(dataset, read_version, group).await?);
        }
    }
    Ok(validated)
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
