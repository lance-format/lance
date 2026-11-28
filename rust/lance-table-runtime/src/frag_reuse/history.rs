// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! On a stable-row-id table, indices that store row addresses translate them
//! through the fragment reuse history after a deferred compaction. A
//! publication must keep every recorded version exactly. A rewrite that carries
//! the history appends the one version recording its own moves; a rewrite that
//! does not leaves it unchanged, and either remaps an address index eagerly or
//! stops it covering the fragments it rewrites. Only a cleanup, a commit that
//! replaces the fragment reuse index, may remove versions: the oldest ones, and
//! only those no index of the published manifest still needs.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use lance_core::{Error, Result};
use lance_table::format::{IndexMetadata, Manifest, is_detached_version};
use lance_table::system_index::frag_reuse::{
    FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion, is_frag_reuse_index_entry,
};
use lance_table::system_index::is_system_index;
use roaring::{RoaringBitmap, RoaringTreemap};

use super::{live_source_row_addrs, load_frag_reuse_index_details, normalize_source_fragments};
use crate::Table;
use crate::commit::load_all_indices;
use lance_table::system_index::frag_reuse::retention::{
    ReuseVersionReachIndex, compute_v0_retention, index_postdates_rewrite,
};
use lance_table::transaction::{Operation, RewriteGroup};

/// The snapshot a writer read, fixed before any rebase, so a refusal can tell
/// history the writer saw from history published after it read.
pub enum ReadSnapshot<'a> {
    Loaded {
        dataset: &'a Table,
        indices: &'a [IndexMetadata],
    },
    /// Loaded only to classify a refusal.
    Version { base: &'a Table, version: u64 },
}

/// The single fragment reuse entry with `index_version = 0` (the format that
/// stores compaction records). `Err` for multiple entries or a tagged entry,
/// which the publication check refuses separately.
fn sole_v0_entry(indices: &[IndexMetadata]) -> std::result::Result<Option<&IndexMetadata>, ()> {
    let mut entries = indices
        .iter()
        .filter(|index| is_frag_reuse_index_entry(index));
    let entry = entries.next();
    if entries.next().is_some() || entry.is_some_and(|entry| entry.index_version != 0) {
        return Err(());
    }
    Ok(entry)
}

/// Validate fragment reuse history stored as compaction records
/// (`index_version = 0`) when committing a stable-row-id table:
///
/// 1. Keep the existing entry and its recorded versions unchanged and in order,
///    except for cleanup, restore and overwrite below. A rewrite without FRI
///    may leave the history unchanged.
/// 2. Only a rewrite may append a version. A rewrite carrying FRI must append
///    exactly one, matching its source fragments, live source rows and published
///    destinations. Its stamp must be at least the preceding stamp and the
///    version of every index that may cover its source fragments.
/// 3. Cleanup is a `CreateIndex` that replaces the FRI entry, and it may add or
///    remove other indices too. It may remove an oldest prefix the writer read,
///    retaining the entry and every later version unchanged. No index in the
///    proposed manifest, including one the commit adds, may still need a removed
///    version.
/// 4. Overwrite may replace history. Restore republishes the selected history,
///    but refuses history or index versions ahead of the new main-chain version.
///
/// `base` is the manifest the commit replaces (for a detached commit, the one it
/// builds on).
///
/// A refusal is retryable when the history or index it misses was published
/// after the writer's `read` snapshot, and `InvalidInput` when the writer saw
/// it or the commit supplies it.
pub async fn validate_frag_reuse_history(
    operation: &Operation,
    base: &Table,
    read: ReadSnapshot<'_>,
    proposed_manifest: &Manifest,
    proposed_indices: &[IndexMetadata],
) -> Result<()> {
    match operation {
        Operation::Restore { version } => {
            return validate_restore(base, *version, proposed_manifest, proposed_indices).await;
        }
        Operation::Overwrite { .. } => return Ok(()),
        // Non-stable-row-id tables still protect v0 FRI cleanups against
        // dropping versions needed by an index in the published manifest
        // (including concurrent CreateIndex commits), while allowing other
        // operations such as migration to stable row ids.
        _ if !base.manifest.uses_stable_row_ids() => {
            return validate_non_stable_v0_trim(
                operation,
                base,
                read,
                proposed_manifest,
                proposed_indices,
            )
            .await;
        }
        _ => {}
    }
    let base_indices = load_all_indices(base, None).await?;
    let (Ok(replaced), Ok(proposed)) = (
        sole_v0_entry(&base_indices),
        sole_v0_entry(proposed_indices),
    ) else {
        return Ok(());
    };
    // The groups of a rewrite, and whether it carries the history to record them in.
    let rewrite = match operation {
        Operation::Rewrite {
            groups,
            frag_reuse_index,
            ..
        } if !groups.is_empty() => Some((groups.as_slice(), frag_reuse_index.is_some())),
        _ => None,
    };

    let (last_stamp, appended) = match (replaced, proposed) {
        (None, None) => return Ok(()),
        (Some(_), None) => return Err(read.refuse(Loss::Removed, base).await),
        (None, Some(proposed)) => {
            let published = load_proposed_history(base, proposed_manifest, proposed).await?;
            (None, published.versions.clone())
        }
        // The same immutable details: the entry is carried over untouched.
        (Some(replaced), Some(proposed))
            if replaced == proposed
                && replaced.base_id.is_none_or(|id| {
                    base.manifest.base_paths.get(&id) == proposed_manifest.base_paths.get(&id)
                }) =>
        {
            (None, Vec::new())
        }
        (Some(replaced), Some(proposed)) => {
            let recorded = load_frag_reuse_index_details(base, replaced).await?;
            let published = load_proposed_history(base, proposed_manifest, proposed).await?;
            if let Operation::CreateIndex {
                new_indices,
                removed_indices,
            } = operation
                && replaces_frag_reuse_entry(new_indices, removed_indices)
                && let Some(trimmed) = trimmed_versions(&recorded.versions, &published.versions)?
            {
                return validate_trim(
                    &recorded.versions,
                    trimmed,
                    new_indices,
                    proposed_indices,
                    &read,
                    base,
                )
                .await;
            }
            for (position, version) in recorded.versions.iter().enumerate() {
                let kept = match published.versions.get(position) {
                    Some(candidate) => same_version(version, candidate)?,
                    None => false,
                };
                if !kept {
                    return Err(read.refuse(Loss::Version(version), base).await);
                }
            }
            (
                recorded
                    .versions
                    .last()
                    .map(|version| version.dataset_version),
                published.versions[recorded.versions.len()..].to_vec(),
            )
        }
    };

    match (rewrite, appended.as_slice()) {
        (None, []) => Ok(()),
        (None, versions) => Err(Error::invalid_input(format!(
            "Cannot commit a {} that adds {} fragment reuse version(s): a version may only be \
             appended by the rewrite whose row-address moves it records",
            operation.name(),
            versions.len()
        ))),
        // Unrecorded: the rewrite remaps the address indices it can, and the
        // manifest build drops the rewritten fragments from the others.
        (Some((_, false)), []) => Ok(()),
        (Some((_, true)), []) => Err(Error::invalid_input(
            "Cannot commit a rewrite that carries a fragment reuse index without appending the \
             version that records its own row-address moves",
        )),
        (Some((groups, _)), [version]) => {
            validate_appended_version(base, proposed_manifest, groups, version).await?;
            validate_appended_stamp(&base_indices, last_stamp, groups, version)
        }
        (Some((groups, _)), versions) => Err(Error::invalid_input(format!(
            "Cannot commit a rewrite of {} group(s) that appends {} fragment reuse versions: a \
             rewrite records its moves in exactly one version",
            groups.len(),
            versions.len()
        ))),
    }
}

/// A restore republishes an earlier manifest, history included, as the next
/// main-chain version, and every later rewrite is stamped at or above it. A
/// history stamp or index version above it, which only a detached version can
/// hold, would refuse every later compaction of those fragments, so such a
/// restore is refused before it publishes.
async fn validate_restore(
    base: &Table,
    restored: u64,
    proposed_manifest: &Manifest,
    proposed_indices: &[IndexMetadata],
) -> Result<()> {
    let version = proposed_manifest.version;
    if is_detached_version(version) || !proposed_manifest.uses_stable_row_ids() {
        return Ok(());
    }
    let Ok(Some(entry)) = sole_v0_entry(proposed_indices) else {
        return Ok(());
    };
    let history = load_proposed_history(base, proposed_manifest, entry).await?;
    let last_stamp = history
        .versions
        .iter()
        .map(|version| version.dataset_version)
        .max();
    if let Some(stamp) = last_stamp.filter(|stamp| *stamp > version) {
        return Err(Error::invalid_input(format!(
            "Cannot restore version {restored} as version {version}: its fragment reuse history \
             holds a version stamped {stamp}, and every later rewrite is stamped below that, so \
             none could record its row-address moves. Restore a version whose history was \
             recorded on this chain"
        )));
    }
    let newer = proposed_indices
        .iter()
        .find(|index| !is_system_index(index) && index.dataset_version > version);
    if let Some(index) = newer {
        return Err(Error::invalid_input(format!(
            "Cannot restore version {restored} as version {version}: index '{}' was built at \
             version {}, and no later rewrite of the fragments it covers could be stamped at or \
             above that, so none could record its row-address moves. Drop the index from that \
             version before restoring it, and rebuild it on this chain",
            index.name, index.dataset_version
        )));
    }
    Ok(())
}

/// The history the proposed manifest publishes, read where a reader of that
/// manifest finds it rather than where the replaced manifest put it.
async fn load_proposed_history(
    base: &Table,
    proposed_manifest: &Manifest,
    entry: &IndexMetadata,
) -> Result<Arc<FragReuseIndexDetails>> {
    let proposed = base.with_manifest_view(Arc::new(proposed_manifest.clone()));
    load_frag_reuse_index_details(&proposed, entry)
        .await
        .map_err(|error| match error {
            Error::NotFound { .. } => Error::invalid_input(format!(
                "Cannot commit a manifest whose fragment reuse history is missing where the \
                 manifest locates it: {error}"
            )),
            error => error,
        })
}

/// Whether a `CreateIndex` replaces the fragment reuse entry, removing one entry
/// and publishing one in its place, whatever other indices it adds or removes.
fn replaces_frag_reuse_entry(
    new_indices: &[IndexMetadata],
    removed_indices: &[IndexMetadata],
) -> bool {
    let entries = |indices: &[IndexMetadata]| {
        indices
            .iter()
            .filter(|index| is_frag_reuse_index_entry(index))
            .count()
    };
    entries(new_indices) == 1 && entries(removed_indices) == 1
}

/// How many versions a cleanup removes from the front of the history: `Some`
/// only when it publishes every later version exactly. Anything else is held
/// to keeping every recorded version.
fn trimmed_versions(
    recorded: &[FragReuseVersion],
    published: &[FragReuseVersion],
) -> Result<Option<usize>> {
    let trimmed = recorded.len().saturating_sub(published.len());
    if trimmed == 0 {
        return Ok(None);
    }
    for (kept, candidate) in recorded[trimmed..].iter().zip(published) {
        if !same_version(kept, candidate)? {
            return Ok(None);
        }
    }
    Ok(Some(trimmed))
}

/// Whether `index` may still hold row addresses `version` moved: either
/// behind it under [`index_postdates_rewrite`] or undecided (`InvalidInput`
/// for a rewrite group split across its coverage). `overlaps_reach` must be
/// `true` when `index.fragment_bitmap` is absent or intersects `version`'s
/// per-version reach.
fn index_needs_version(
    version: &FragReuseVersion,
    index: &IndexMetadata,
    overlaps_reach: bool,
) -> bool {
    !is_system_index(index)
        && overlaps_reach
        && !matches!(index_postdates_rewrite(version, index), Ok(true))
}

/// For each position in `0..prefix_len`, the first index in `indices` that
/// may still need `versions[pos]`, evaluated via reverse-topological boolean
/// reach propagation without materializing per-version reach bitmaps.
fn first_needing_indices<'a>(
    versions: &[FragReuseVersion],
    prefix_len: usize,
    indices: &'a [IndexMetadata],
) -> Vec<Option<&'a IndexMetadata>> {
    let mut needing = vec![None; prefix_len];
    if prefix_len == 0 || versions.is_empty() {
        return needing;
    }
    let reach_index = ReuseVersionReachIndex::new(versions.iter());
    let mut remaining = prefix_len;
    for index in indices {
        if is_system_index(index) {
            continue;
        }
        let overlaps = index
            .fragment_bitmap
            .as_ref()
            .map(|bitmap| reach_index.versions_overlapping_bitmap(bitmap));
        for pos in 0..prefix_len {
            if needing[pos].is_none()
                && index_needs_version(
                    &versions[pos],
                    index,
                    overlaps.as_ref().is_none_or(|hits| hits[pos]),
                )
            {
                needing[pos] = Some(index);
                remaining -= 1;
            }
        }
        if remaining == 0 {
            break;
        }
    }
    needing
}

/// How many of the oldest versions a cleanup of a stable-row-id table removes:
/// those before the first version an index of `indices` still needs. The kept
/// versions stay a complete suffix of the history, so an address they map
/// translates through every later move.
pub fn trimmable_versions(versions: &[FragReuseVersion], indices: &[IndexMetadata]) -> usize {
    if versions.is_empty() {
        return 0;
    }
    let reach_index = ReuseVersionReachIndex::new(versions.iter());
    let mut first_needed = versions.len();
    for index in indices {
        if is_system_index(index) {
            continue;
        }
        let overlaps = index
            .fragment_bitmap
            .as_ref()
            .map(|bitmap| reach_index.versions_overlapping_bitmap(bitmap));
        if let Some(pos) = (0..first_needed).find(|&pos| {
            index_needs_version(
                &versions[pos],
                index,
                overlaps.as_ref().is_none_or(|hits| hits[pos]),
            )
        }) {
            first_needed = pos;
            if first_needed == 0 {
                break;
            }
        }
    }
    first_needed
}

fn find_version(
    versions: &[FragReuseVersion],
    version: &FragReuseVersion,
) -> Result<Option<usize>> {
    for (pos, candidate) in versions.iter().enumerate() {
        if same_version(candidate, version)? {
            return Ok(Some(pos));
        }
    }
    Ok(None)
}

/// A cleanup removes the oldest `trimmed` recorded versions. Each must be one
/// the writer read, and no index the commit publishes may still need it. An
/// index the writer read or supplies makes a refusal `InvalidInput`. A version
/// recorded or an index committed since the writer read makes it retryable: run
/// again on the latest version, the cleanup keeps them.
async fn validate_trim(
    recorded: &[FragReuseVersion],
    trimmed: usize,
    supplied_indices: &[IndexMetadata],
    proposed_indices: &[IndexMetadata],
    read: &ReadSnapshot<'_>,
    base: &Table,
) -> Result<()> {
    let seen = read.load().await.map_err(|error| {
        Error::retryable_commit_conflict_source(
            base.manifest.version,
            format!(
                "This cleanup removes fragment reuse versions on a table with stable row ids, but \
                 the version it read cannot be loaded to check them: {error}. Run the cleanup \
                 again on the latest version."
            )
            .into(),
        )
    })?;
    validate_trim_against_read_state(
        recorded,
        trimmed,
        supplied_indices,
        proposed_indices,
        &seen,
        base.manifest.version,
    )
}

fn validate_trim_against_read_state(
    recorded: &[FragReuseVersion],
    trimmed: usize,
    supplied_indices: &[IndexMetadata],
    proposed_indices: &[IndexMetadata],
    seen: &ReadState,
    base_version: u64,
) -> Result<()> {
    let retry = |message: String| {
        Error::retryable_commit_conflict_source(
            base_version,
            format!("{message}. Run the cleanup again on the latest version.").into(),
        )
    };
    if trimmed == 0 {
        return Ok(());
    }
    let read_versions = seen
        .history
        .as_ref()
        .map_or(&[][..], |history| history.versions.as_slice());
    let mut read_by_stamp: HashMap<u64, Vec<(usize, &FragReuseVersion)>> =
        HashMap::with_capacity(read_versions.len());
    for (read_pos, candidate) in read_versions.iter().enumerate() {
        read_by_stamp
            .entry(candidate.dataset_version)
            .or_default()
            .push((read_pos, candidate));
    }
    let needing = first_needing_indices(recorded, trimmed, proposed_indices);
    for (pos, version) in recorded[..trimmed].iter().enumerate() {
        let stamp = version.dataset_version;
        let mut found_read_pos = None;
        if let Some(candidates) = read_by_stamp.get(&stamp) {
            for &(candidate_pos, candidate) in candidates {
                if same_version(candidate, version)? {
                    found_read_pos = Some(candidate_pos);
                    break;
                }
            }
        }
        let Some(read_pos) = found_read_pos else {
            return Err(retry(format!(
                "This cleanup would remove the fragment reuse version stamped {stamp}, which was \
                 recorded after version {}, the version it read",
                seen.version
            )));
        };
        let Some(index) = needing[pos] else {
            continue;
        };
        // Judged as the writer saw it: against the history it read, through the
        // metadata it read or supplies.
        let read_reaches = ReuseVersionReachIndex::new(read_versions.iter());
        let needed_when_read = seen
            .indices
            .iter()
            .chain(supplied_indices)
            .filter(|known| known.uuid == index.uuid)
            .any(|known| {
                let overlaps_reach = known.fragment_bitmap.as_ref().is_none_or(|bitmap| {
                    read_reaches.versions_overlapping_bitmap(bitmap)[read_pos]
                });
                index_needs_version(version, known, overlaps_reach)
            });
        if needed_when_read {
            return Err(Error::invalid_input(format!(
                "Cannot remove the fragment reuse version stamped {stamp} on a table with stable \
                 row ids: index '{}' ({}), built at version {}, may still hold row addresses it \
                 moved. A cleanup removes only the oldest versions no index needs",
                index.name, index.uuid, index.dataset_version
            )));
        }
        return Err(retry(format!(
            "This cleanup would remove the fragment reuse version stamped {stamp}, but index \
             '{}' ({}), committed after version {}, the version the cleanup read, may still hold \
             row addresses it moved",
            index.name, index.uuid, seen.version
        )));
    }
    Ok(())
}

/// Protect a non-stable-row-id v0 FRI cleanup against dropping a version that
/// an index in the published manifest still requires (directly or through a
/// downstream mapping hop), such as when a concurrent `CreateIndex` commits
/// after the cleanup read its snapshot.
async fn validate_non_stable_v0_trim(
    operation: &Operation,
    base: &Table,
    read: ReadSnapshot<'_>,
    proposed_manifest: &Manifest,
    proposed_indices: &[IndexMetadata],
) -> Result<()> {
    let Operation::CreateIndex {
        new_indices,
        removed_indices,
    } = operation
    else {
        return Ok(());
    };
    if !replaces_frag_reuse_entry(new_indices, removed_indices) {
        return Ok(());
    }
    let base_indices = load_all_indices(base, None).await?;
    let (Ok(Some(replaced)), Ok(Some(proposed))) = (
        sole_v0_entry(&base_indices),
        sole_v0_entry(proposed_indices),
    ) else {
        return Ok(());
    };
    if replaced == proposed
        && replaced.base_id.is_none_or(|id| {
            base.manifest.base_paths.get(&id) == proposed_manifest.base_paths.get(&id)
        })
    {
        return Ok(());
    }
    let recorded = load_frag_reuse_index_details(base, replaced).await?;
    let published = load_proposed_history(base, proposed_manifest, proposed).await?;
    let Some(dropped) =
        first_dropped_required_version(&recorded.versions, &published.versions, proposed_indices)?
    else {
        return Ok(());
    };
    let seen = read.load().await.map_err(|error| {
        Error::retryable_commit_conflict_source(
            base.manifest.version,
            format!(
                "This cleanup removes fragment reuse versions, but the version it read cannot be \
                 loaded to check them: {error}. Run the cleanup again on the latest version."
            )
            .into(),
        )
    })?;
    classify_non_stable_v0_trim_drop(
        dropped,
        new_indices,
        removed_indices,
        &seen,
        base.manifest.version,
    )
}

fn first_dropped_required_version<'a>(
    recorded: &'a [FragReuseVersion],
    published: &[FragReuseVersion],
    proposed_indices: &[IndexMetadata],
) -> Result<Option<&'a FragReuseVersion>> {
    Ok(first_dropped_required_version_with_comparisons(recorded, published, proposed_indices)?.0)
}

fn first_dropped_required_version_with_comparisons<'a>(
    recorded: &'a [FragReuseVersion],
    published: &[FragReuseVersion],
    proposed_indices: &[IndexMetadata],
) -> Result<(Option<&'a FragReuseVersion>, usize)> {
    let required = compute_v0_retention(recorded, proposed_indices);
    let mut published_by_stamp: HashMap<u64, Vec<&FragReuseVersion>> =
        HashMap::with_capacity(published.len());
    for candidate in published {
        published_by_stamp
            .entry(candidate.dataset_version)
            .or_default()
            .push(candidate);
    }

    let mut comparisons = 0usize;
    for (version, needed) in recorded.iter().zip(required) {
        if !needed {
            continue;
        }
        let mut present = false;
        if let Some(candidates) = published_by_stamp.get(&version.dataset_version) {
            for candidate in candidates {
                comparisons += 1;
                if same_version(candidate, version)? {
                    present = true;
                    break;
                }
            }
        }
        if !present {
            return Ok((Some(version), comparisons));
        }
    }
    Ok((None, comparisons))
}

fn classify_non_stable_v0_trim_drop(
    dropped: &FragReuseVersion,
    supplied_indices: &[IndexMetadata],
    removed_indices: &[IndexMetadata],
    seen: &ReadState,
    base_version: u64,
) -> Result<()> {
    let stamp = dropped.dataset_version;
    let read_versions = seen
        .history
        .as_ref()
        .map_or(&[][..], |history| history.versions.as_slice());
    if let Some(read_pos) = find_version(read_versions, dropped)? {
        let known_indices: Vec<IndexMetadata> = seen
            .indices
            .iter()
            .filter(|known| !removed_indices.iter().any(|r| r.uuid == known.uuid))
            .chain(supplied_indices.iter())
            .cloned()
            .collect();
        let read_required = compute_v0_retention(read_versions, &known_indices);
        if read_required[read_pos] {
            return Err(Error::invalid_input(format!(
                "Cannot remove the fragment reuse version stamped {stamp}: an index in the \
                 manifest may still need its row-address mappings"
            )));
        }
    }
    Err(Error::retryable_commit_conflict_source(
        base_version,
        format!(
            "This cleanup would remove the fragment reuse version stamped {stamp}, which is \
             required by an index committed after version {}, the version the cleanup read. Run \
             the cleanup again on the latest version.",
            seen.version
        )
        .into(),
    ))
}

enum Loss<'a> {
    Removed,
    /// A recorded version is missing or no longer the same.
    Version(&'a FragReuseVersion),
}

impl ReadSnapshot<'_> {
    /// `InvalidInput` when the writer's snapshot already held what this loses,
    /// otherwise a retryable conflict: that history arrived after the writer read.
    async fn refuse(&self, loss: Loss<'_>, base: &Table) -> Error {
        let (read_version, read_history) = match self.load().await {
            Ok(seen) => (seen.version, seen.history),
            Err(error) => {
                return Error::invalid_input(format!(
                    "Cannot commit: it would lose fragment reuse history on a table with stable \
                     row ids, and the version it read cannot be loaded to tell whether it saw \
                     that history: {error}"
                ));
            }
        };
        let was_seen = match (&loss, &read_history) {
            (_, None) => false,
            (Loss::Removed, Some(_)) => true,
            (Loss::Version(lost), Some(details)) => details
                .versions
                .iter()
                .any(|seen| same_version(seen, lost).unwrap_or(false)),
        };
        if was_seen {
            return Error::invalid_input(match loss {
                Loss::Removed => format!(
                    "Cannot remove the fragment reuse index of a table with stable row ids: \
                     indices translate their row addresses through its history, which this table \
                     held at version {read_version}, the version this commit read. A cleanup \
                     removes the versions no index needs and keeps the entry"
                ),
                Loss::Version(lost) => format!(
                    "Cannot drop or alter the fragment reuse version stamped {} on a table with \
                     stable row ids: it was recorded at version {read_version}, the version this \
                     commit read. Only a cleanup, a commit replacing the fragment reuse index, may \
                     remove versions, the oldest ones no index needs, and every other version must \
                     be kept exactly",
                    lost.dataset_version
                ),
            });
        }
        let message = match loss {
            Loss::Removed => format!(
                "This commit would remove the fragment reuse index, which the table gained \
                 after version {read_version}, the version this commit read"
            ),
            Loss::Version(lost) => format!(
                "This commit would drop or alter the fragment reuse version stamped {}, which \
                 was recorded after version {read_version}, the version this commit read",
                lost.dataset_version
            ),
        };
        Error::retryable_commit_conflict_source(
            base.manifest.version,
            format!("{message}. Plan and run the operation again from the latest version.").into(),
        )
    }

    async fn load(&self) -> Result<ReadState> {
        let checked_out;
        let (dataset, indices) = match self {
            Self::Loaded { dataset, indices } => (*dataset, indices.to_vec()),
            Self::Version { base, version } => {
                let dataset = if base.manifest.version == *version {
                    *base
                } else {
                    checked_out = base.checkout_version(*version).await?;
                    &checked_out
                };
                (
                    dataset,
                    load_all_indices(dataset, None).await?.as_ref().clone(),
                )
            }
        };
        let history = match sole_v0_entry(&indices).ok().flatten() {
            Some(entry) => Some(load_frag_reuse_index_details(dataset, entry).await?),
            None => None,
        };
        Ok(ReadState {
            version: dataset.manifest.version,
            indices,
            history,
        })
    }
}

struct ReadState {
    version: u64,
    indices: Vec<IndexMetadata>,
    history: Option<Arc<FragReuseIndexDetails>>,
}

fn decode_row_addrs(group: &FragReuseGroup) -> Result<RoaringTreemap> {
    RoaringTreemap::deserialize_from(Cursor::new(&group.changed_row_addrs)).map_err(|error| {
        Error::invalid_input(format!(
            "the fragment reuse group over fragments {:?} has undecodable row addresses: {error}",
            group
                .old_frags
                .iter()
                .map(|frag| frag.id)
                .collect::<Vec<_>>()
        ))
    })
}

/// Whether two versions record the same moves under the same stamp. Groups
/// cover disjoint fragments, so their order carries no meaning; fragment order
/// within a group decides where rows land and must match.
pub fn same_version(left: &FragReuseVersion, right: &FragReuseVersion) -> Result<bool> {
    if left == right {
        return Ok(true);
    }
    if left.dataset_version != right.dataset_version || left.groups.len() != right.groups.len() {
        return Ok(false);
    }
    fn by_first_source(groups: &[FragReuseGroup]) -> Vec<&FragReuseGroup> {
        let mut groups = groups.iter().collect::<Vec<_>>();
        groups.sort_by_key(|group| group.old_frags.first().map(|frag| frag.id));
        groups
    }
    for (left, right) in by_first_source(&left.groups)
        .into_iter()
        .zip(by_first_source(&right.groups))
    {
        if left.old_frags != right.old_frags
            || left.new_frags != right.new_frags
            || decode_row_addrs(left)? != decode_row_addrs(right)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The version a rewrite appends must record each of its groups: the source
/// fragments as the replaced manifest holds them, in ascending id order, every
/// live source row, and the destination fragments as published.
async fn validate_appended_version(
    base: &Table,
    proposed_manifest: &Manifest,
    groups: &[RewriteGroup],
    version: &FragReuseVersion,
) -> Result<()> {
    let mismatch = |position: usize, detail: String| {
        Error::invalid_input(format!(
            "Cannot commit a rewrite whose fragment reuse version does not record rewrite group \
             {position}: {detail}"
        ))
    };
    if version.groups.len() != groups.len() {
        return Err(Error::invalid_input(format!(
            "Cannot commit a rewrite of {} group(s) whose fragment reuse version records {} \
             group(s)",
            groups.len(),
            version.groups.len()
        )));
    }
    for (position, (group, recorded)) in groups.iter().zip(&version.groups).enumerate() {
        if let Some(pair) = group
            .old_fragments
            .windows(2)
            .find(|pair| pair[0].id >= pair[1].id)
        {
            return Err(mismatch(
                position,
                format!(
                    "fragment {} precedes fragment {}, but recorded moves need ascending source \
                     ids",
                    pair[0].id, pair[1].id
                ),
            ));
        }
        let sources = group
            .old_fragments
            .iter()
            .map(|old| {
                base.manifest
                    .fragments
                    .iter()
                    .find(|fragment| fragment.id == old.id)
                    .cloned()
                    .ok_or_else(|| {
                        mismatch(
                            position,
                            format!("source fragment {} is not in the replaced manifest", old.id),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = normalize_source_fragments(base, &sources).await?;
        let source_digests = sources.iter().map(FragDigest::from).collect::<Vec<_>>();
        if recorded.old_frags != source_digests {
            return Err(mismatch(
                position,
                format!(
                    "it records sources {:?}, but the replaced manifest holds {:?}",
                    recorded.old_frags, source_digests
                ),
            ));
        }
        let destinations = group
            .new_fragments
            .iter()
            .map(|new| {
                proposed_manifest
                    .fragments
                    .iter()
                    .find(|fragment| fragment.id == new.id && fragment.physical_rows.is_some())
                    .cloned()
                    .ok_or_else(|| {
                        mismatch(
                            position,
                            format!(
                                "destination fragment {} is not in the published manifest; \
                                 reserve fragment ids before recording",
                                new.id
                            ),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let destination_digests = destinations
            .iter()
            .map(FragDigest::from)
            .collect::<Vec<_>>();
        if recorded.new_frags != destination_digests {
            return Err(mismatch(
                position,
                format!(
                    "it records destinations {:?}, but the rewrite publishes {:?}",
                    recorded.new_frags, destination_digests
                ),
            ));
        }
        let moved_rows = live_source_row_addrs(base, &sources).await?;
        if decode_row_addrs(recorded)? != moved_rows {
            return Err(mismatch(
                position,
                format!(
                    "its moved rows are not the {} row addresses moved from its sources",
                    moved_rows.len()
                ),
            ));
        }
        let written = destination_digests
            .iter()
            .map(|digest| digest.physical_rows as u64)
            .sum::<u64>();
        if written != moved_rows.len() {
            return Err(mismatch(
                position,
                format!(
                    "its destinations hold {written} rows, but its sources moved {} rows",
                    moved_rows.len()
                ),
            ));
        }
    }
    Ok(())
}

/// Index maintenance treats an index built at or before a version's stamp as
/// still holding the addresses that version moved, so the stamp must be at
/// least the version of every index the base holds over the rewritten
/// fragments. It must also keep the history in stamp order, the order its
/// encoding sorts versions into.
fn validate_appended_stamp(
    base_indices: &[IndexMetadata],
    last_stamp: Option<u64>,
    groups: &[RewriteGroup],
    version: &FragReuseVersion,
) -> Result<()> {
    let stamp = version.dataset_version;
    if let Some(last_stamp) = last_stamp.filter(|last_stamp| stamp < *last_stamp) {
        return Err(Error::invalid_input(format!(
            "Cannot commit a rewrite whose fragment reuse version is stamped {stamp}, below \
             {last_stamp}, the stamp of the recorded version it follows: the history is kept in \
             stamp order"
        )));
    }
    let sources = groups
        .iter()
        .flat_map(|group| {
            group
                .old_fragments
                .iter()
                .map(|fragment| fragment.id as u32)
        })
        .collect::<RoaringBitmap>();
    let newer = base_indices.iter().find(|index| {
        !is_system_index(index)
            && index.dataset_version > stamp
            && index
                .fragment_bitmap
                .as_ref()
                .is_none_or(|coverage| !coverage.is_disjoint(&sources))
    });
    if let Some(index) = newer {
        return Err(Error::invalid_input(format!(
            "Cannot commit a rewrite whose fragment reuse version is stamped {stamp}: index '{}' \
             covers its source fragments and was built at version {}, so index maintenance would \
             take it for an index built after the rewrite. The stamp must be at least the version \
             of every index over the rewritten fragments",
            index.name, index.dataset_version
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stamp is held to the comparison index maintenance makes, against
    /// the indices over the rewritten fragments only, and version numbers are
    /// never compared with one another otherwise.
    #[test]
    fn test_stamp_reaches_every_index_over_the_sources() {
        use lance_table::format::{DETACHED_VERSION_MASK, Fragment};
        use lance_table::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;

        let index = |name: &str, dataset_version: u64, coverage: Option<&[u32]>| IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: name.to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version,
            fragment_bitmap: coverage.map(|ids| ids.iter().copied().collect()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        let groups = [RewriteGroup {
            old_fragments: vec![Fragment::new(6), Fragment::new(7)],
            new_fragments: vec![Fragment::new(9)],
        }];
        let detached = |version: u64| DETACHED_VERSION_MASK | version;
        let cases = [
            // Inclusive, as maintenance compares.
            (10, None, index("i_idx", 10, Some(&[6])), None),
            (
                10,
                None,
                index("i_idx", 11, Some(&[7, 8])),
                Some("covers its source"),
            ),
            (10, None, index("i_idx", 11, Some(&[1, 2])), None),
            (
                10,
                None,
                index("i_idx", 11, None),
                Some("covers its source"),
            ),
            (10, None, index(FRAG_REUSE_INDEX_NAME, 11, Some(&[6])), None),
            (10, Some(10), index("i_idx", 10, Some(&[6])), None),
            (
                10,
                Some(11),
                index("i_idx", 10, Some(&[6])),
                Some("stamp order"),
            ),
            // An index the base holds is taken by its number, random or not.
            (
                detached(5),
                None,
                index("i_idx", detached(900), Some(&[6])),
                Some("covers its source"),
            ),
            (detached(5), None, index("i_idx", 7, Some(&[6])), None),
        ];
        for (stamp, last_stamp, index, refusal) in cases {
            let version = FragReuseVersion {
                dataset_version: stamp,
                groups: vec![],
            };
            let result = validate_appended_stamp(
                std::slice::from_ref(&index),
                last_stamp,
                &groups,
                &version,
            );
            let case = format!("stamp {stamp} after {last_stamp:?} with {index:?}");
            match refusal {
                None => result.unwrap_or_else(|error| panic!("{case}: {error}")),
                Some(message) => {
                    let error = result.expect_err(&case);
                    assert!(
                        matches!(error, Error::InvalidInput { .. }),
                        "{case}: {error}"
                    );
                    assert!(error.to_string().contains(message), "{case}: {error}");
                }
            }
        }
    }

    fn test_digest(id: u64) -> FragDigest {
        FragDigest {
            id,
            physical_rows: 100,
            num_deleted_rows: 0,
        }
    }

    fn test_version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
        FragReuseVersion {
            dataset_version,
            groups: vec![FragReuseGroup {
                changed_row_addrs: Vec::new(),
                old_frags: old.iter().copied().map(test_digest).collect(),
                new_frags: new.iter().copied().map(test_digest).collect(),
            }],
        }
    }

    fn test_index_meta(
        uuid: uuid::Uuid,
        name: &str,
        dataset_version: u64,
        frags: &[u32],
    ) -> IndexMetadata {
        IndexMetadata {
            uuid,
            name: name.to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version,
            fragment_bitmap: Some(frags.iter().copied().collect()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_trimmable_versions_uses_per_version_reach() {
        let v1 = test_version(10, &[4, 5], &[6]);
        let v2 = test_version(20, &[8, 9], &[10]);
        let rebuilt_after_v1 = test_index_meta(uuid::Uuid::new_v4(), "idx_a", 15, &[6]);

        // Rebuilt after v1 over [6], disjoint from v2: neither version is needed.
        assert_eq!(trimmable_versions(&[v1, v2], &[rebuilt_after_v1]), 2);
    }

    #[test]
    fn test_validate_trim_classifies_reach_change_between_read_and_commit() {
        let v1 = test_version(10, &[1], &[2]);
        let v2 = test_version(20, &[2, 3], &[4]);
        let idx_uuid = uuid::Uuid::new_v4();
        let idx_before_v2 = test_index_meta(idx_uuid, "idx", 5, &[3]);
        let idx_after_v2 = test_index_meta(idx_uuid, "idx", 5, &[4]);

        // Read snapshot only saw v1 (reach {1, 2}), disjoint from idx covering [3].
        // By commit time, v2 merged [2, 3] -> [4], extending v1's reach to {1, 2, 4}
        // and advancing idx onto [4]: refusal must be retryable.
        let seen_before_v2 = ReadState {
            version: 10,
            indices: vec![idx_before_v2],
            history: Some(Arc::new(FragReuseIndexDetails {
                versions: vec![v1.clone()],
            })),
        };
        let err = validate_trim_against_read_state(
            &[v1.clone(), v2.clone()],
            1,
            &[],
            std::slice::from_ref(&idx_after_v2),
            &seen_before_v2,
            20,
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );

        // When the writer's read snapshot already included v2 and idx over [4],
        // the same trim is invalid input rather than a retryable conflict.
        let seen_after_v2 = ReadState {
            version: 20,
            indices: vec![idx_after_v2.clone()],
            history: Some(Arc::new(FragReuseIndexDetails {
                versions: vec![v1.clone(), v2.clone()],
            })),
        };
        let err = validate_trim_against_read_state(
            &[v1, v2],
            1,
            &[],
            std::slice::from_ref(&idx_after_v2),
            &seen_after_v2,
            20,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");

        // Disjoint versions where the only index rebuilt after v1 over [6] are
        // both trimmable without conflict.
        let d1 = test_version(10, &[4, 5], &[6]);
        let d2 = test_version(20, &[8, 9], &[10]);
        let rebuilt = test_index_meta(uuid::Uuid::new_v4(), "idx_a", 15, &[6]);
        let seen_disjoint = ReadState {
            version: 20,
            indices: vec![rebuilt.clone()],
            history: Some(Arc::new(FragReuseIndexDetails {
                versions: vec![d1.clone(), d2.clone()],
            })),
        };
        validate_trim_against_read_state(&[d1, d2], 2, &[], &[rebuilt], &seen_disjoint, 20)
            .unwrap();
    }

    #[test]
    fn test_non_stable_v0_trim_rejects_dropping_required_version() {
        let v1 = test_version(10, &[1], &[2]);
        let v2 = test_version(20, &[2], &[3]);
        let recorded = vec![v1.clone(), v2];
        let published = vec![v1];

        // Concurrent stale index committed after the cleanup read its snapshot:
        // raw bitmap {1} needs v1 directly and v2 via forward closure -> retryable conflict.
        let stale_concurrent = test_index_meta(uuid::Uuid::new_v4(), "idx_b", 5, &[1]);
        let dropped = first_dropped_required_version(
            &recorded,
            &published,
            std::slice::from_ref(&stale_concurrent),
        )
        .unwrap()
        .expect("v2 must be required via forward closure");
        assert_eq!(dropped.dataset_version, 20);

        let seen_without_idx = ReadState {
            version: 20,
            indices: vec![],
            history: Some(Arc::new(FragReuseIndexDetails {
                versions: recorded.clone(),
            })),
        };
        let err =
            classify_non_stable_v0_trim_drop(dropped, &[], &[], &seen_without_idx, 21).unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );

        // When the read snapshot already held the stale index, dropping v2 is InvalidInput.
        let seen_with_idx = ReadState {
            version: 21,
            indices: vec![stale_concurrent],
            history: Some(Arc::new(FragReuseIndexDetails {
                versions: recorded.clone(),
            })),
        };
        let err =
            classify_non_stable_v0_trim_drop(dropped, &[], &[], &seen_with_idx, 21).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    #[test]
    fn test_non_stable_v0_trim_check_scales_linearly_on_disjoint_history() {
        // 2,000 disjoint required versions plus 1 droppable version: stamp-bucketed
        // lookup checks each required version once rather than scanning published quadratically.
        const REQUIRED: usize = 2_000;
        let mut recorded: Vec<FragReuseVersion> = (0..REQUIRED)
            .map(|i| {
                let base = (i as u64) * 2;
                test_version(i as u64 + 10, &[base], &[base + 1])
            })
            .collect();
        let droppable = test_version(REQUIRED as u64 + 10, &[900_000], &[900_001]);
        let published = recorded.clone();
        recorded.push(droppable);

        let stale_all_required = test_index_meta(
            uuid::Uuid::new_v4(),
            "stale",
            1,
            &(0..REQUIRED)
                .map(|i| (i as u32) * 2 + 1)
                .collect::<Vec<_>>(),
        );
        let (dropped, comparisons) = first_dropped_required_version_with_comparisons(
            &recorded,
            &published,
            &[stale_all_required],
        )
        .unwrap();
        assert!(dropped.is_none());
        assert_eq!(comparisons, REQUIRED);
    }

    #[test]
    fn test_stable_v0_trim_handles_long_chain_suffix_and_straddled_group() {
        // Multi-container chain where a single index rebuilt after the last hop
        // covers only the final live fragment: every hop is trimmable via boolean
        // reach propagation without materializing per-version suffix bitmaps.
        const HOPS: usize = 500;
        let mut chain: Vec<FragReuseVersion> = (0..HOPS)
            .map(|i| {
                let src = ((i as u64) + 1) * 100_000;
                let dst = ((i as u64) + 2) * 100_000;
                test_version(i as u64 + 1, &[src], &[dst])
            })
            .collect();
        let final_live = ((HOPS as u32) + 1) * 100_000;
        let rebuilt_after_all =
            test_index_meta(uuid::Uuid::new_v4(), "idx", HOPS as u64 + 1, &[final_live]);
        assert_eq!(
            trimmable_versions(&chain, std::slice::from_ref(&rebuilt_after_all)),
            HOPS
        );
        let seen_all = ReadState {
            version: HOPS as u64 + 1,
            indices: vec![rebuilt_after_all.clone()],
            history: Some(Arc::new(FragReuseIndexDetails {
                versions: chain.clone(),
            })),
        };
        validate_trim_against_read_state(
            &chain,
            HOPS,
            &[],
            &[rebuilt_after_all],
            &seen_all,
            HOPS as u64 + 1,
        )
        .unwrap();

        // Adding a disjoint version at the tail while an index still needs hop 250
        // (dataset_version 251) keeps the full suffix starting at hop 250.
        chain.push(test_version(
            HOPS as u64 + 10,
            &[900_000_000],
            &[900_000_001],
        ));
        let stale_mid = test_index_meta(uuid::Uuid::new_v4(), "stale_mid", 251, &[251 * 100_000]);
        assert_eq!(trimmable_versions(&chain, &[stale_mid]), 250);

        // A post-rewrite index straddling v2's old_frags [2, 3] causes
        // `index_postdates_rewrite` to return InvalidInput, which stable-v0 trim
        // treats as still needed: v1 is trimmable, while v2 and its trailing
        // disjoint suffix v3 are retained.
        let v1 = test_version(10, &[1], &[2]);
        let v2 = test_version(20, &[2, 3], &[4]);
        let v3 = test_version(30, &[8], &[9]);
        let straddling = test_index_meta(uuid::Uuid::new_v4(), "straddling", 25, &[2]);
        let straddle_history = vec![v1, v2, v3];
        assert_eq!(
            trimmable_versions(&straddle_history, std::slice::from_ref(&straddling)),
            1
        );

        let seen_with_straddle = ReadState {
            version: 30,
            indices: vec![straddling.clone()],
            history: Some(Arc::new(FragReuseIndexDetails {
                versions: straddle_history.clone(),
            })),
        };
        let err = validate_trim_against_read_state(
            &straddle_history,
            2,
            &[],
            std::slice::from_ref(&straddling),
            &seen_with_straddle,
            30,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");

        let seen_without_straddle = ReadState {
            version: 30,
            indices: vec![],
            history: Some(Arc::new(FragReuseIndexDetails {
                versions: straddle_history.clone(),
            })),
        };
        let err = validate_trim_against_read_state(
            &straddle_history,
            2,
            &[],
            &[straddling],
            &seen_without_straddle,
            30,
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );
    }
}
