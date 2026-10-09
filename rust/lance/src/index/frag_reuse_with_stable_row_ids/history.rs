// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! On a stable-row-id table the fragment reuse index is the row-address
//! history readers translate old addresses through. A publication must keep
//! every recorded version exactly, and a rewrite must append the one version
//! that records its own moves.

use std::io::Cursor;
use std::sync::Arc;

use lance_core::{Error, Result};
use lance_table::format::{IndexMetadata, Manifest, is_detached_version};
use lance_table::system_index::frag_reuse::{
    FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion, is_frag_reuse_index_entry,
};
use lance_table::system_index::is_system_index;
use roaring::{RoaringBitmap, RoaringTreemap};

use crate::Dataset;
use crate::dataset::optimize::{live_source_row_addrs, normalize_source_fragments};
use crate::dataset::transaction::{Operation, RewriteGroup};
use crate::index::frag_reuse::load_frag_reuse_index_details;
use crate::index::load_all_indices;

/// The snapshot a writer read, fixed before any rebase, so a refusal can tell
/// history the writer saw from history published after it read.
pub enum ReadSnapshot<'a> {
    Loaded {
        dataset: &'a Dataset,
        indices: &'a [IndexMetadata],
    },
    /// Loaded only to classify a refusal.
    Version { base: &'a Dataset, version: u64 },
}

/// The single v0 entry. `Err` when the list holds several entries or a tagged
/// one, which the publication check refuses on its own.
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

/// Refuse a commit that would lose or alter recorded history on a stable-row-id
/// table, or append a version that does not record its rewrite or whose stamp
/// misplaces it. `base` is the manifest the commit replaces (for a detached
/// commit, the one it builds on).
///
/// A refusal is retryable when the history it misses was published after the
/// writer's `read` snapshot, and `InvalidInput` when the writer saw it. Restore
/// and overwrite replace the history by design ([`validate_restore`]).
pub async fn validate_frag_reuse_history(
    operation: &Operation,
    base: &Dataset,
    read: ReadSnapshot<'_>,
    proposed_manifest: &Manifest,
    proposed_indices: &[IndexMetadata],
) -> Result<()> {
    match operation {
        Operation::Restore { version } => {
            return validate_restore(base, *version, proposed_manifest, proposed_indices).await;
        }
        Operation::Overwrite { .. } => return Ok(()),
        _ if !base.manifest.uses_stable_row_ids() => return Ok(()),
        _ => {}
    }
    let base_indices = load_all_indices(base).await?;
    let (Ok(replaced), Ok(proposed)) = (
        sole_v0_entry(&base_indices),
        sole_v0_entry(proposed_indices),
    ) else {
        return Ok(());
    };
    let groups = match operation {
        Operation::Rewrite { groups, .. } if !groups.is_empty() => Some(groups.as_slice()),
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

    match (groups, appended.as_slice()) {
        (None, []) => Ok(()),
        (None, versions) => Err(Error::invalid_input(format!(
            "Cannot commit a {} that adds {} fragment reuse version(s): a version may only be \
             appended by the rewrite whose row-address moves it records",
            operation.name(),
            versions.len()
        ))),
        (Some(_), []) => match replaced {
            Some(_) => Err(read.refuse(Loss::Unrecorded, base).await),
            None => Err(Error::invalid_input(
                "Cannot commit a rewrite that creates a fragment reuse index without recording \
                 its own row-address moves in it",
            )),
        },
        (Some(groups), [version]) => {
            validate_appended_version(base, proposed_manifest, groups, version).await?;
            validate_appended_stamp(&base_indices, last_stamp, groups, version)
        }
        (Some(groups), versions) => Err(Error::invalid_input(format!(
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
    base: &Dataset,
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
    base: &Dataset,
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

enum Loss<'a> {
    /// The entry itself is gone.
    Removed,
    /// A recorded version is missing or no longer the same.
    Version(&'a FragReuseVersion),
    /// A rewrite moves rows without recording them.
    Unrecorded,
}

impl ReadSnapshot<'_> {
    /// `InvalidInput` when the writer's snapshot already held what this loses,
    /// otherwise a retryable conflict: that history arrived after the writer read.
    async fn refuse(&self, loss: Loss<'_>, base: &Dataset) -> Error {
        let (read_version, read_history) = match self.frag_reuse_history().await {
            Ok(history) => history,
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
            (Loss::Removed | Loss::Unrecorded, Some(_)) => true,
            (Loss::Version(lost), Some(details)) => details
                .versions
                .iter()
                .any(|seen| same_version(seen, lost).unwrap_or(false)),
        };
        if was_seen {
            return Error::invalid_input(match loss {
                Loss::Removed => format!(
                    "Cannot remove the fragment reuse index of a table with stable row ids: it \
                     holds the row-address history readers translate old addresses through, \
                     which this table recorded at version {read_version}, the version this \
                     commit read, and there is no supported way to stop recording"
                ),
                Loss::Version(lost) => format!(
                    "Cannot drop or alter the fragment reuse version stamped {} on a table with \
                     stable row ids: it was recorded at version {read_version}, the version this \
                     commit read, and every recorded version must be kept exactly",
                    lost.dataset_version
                ),
                Loss::Unrecorded => format!(
                    "Cannot commit a rewrite that moves rows without recording their \
                     row-address moves: this table with stable row ids already recorded them in \
                     its fragment reuse index at version {read_version}, the version this commit \
                     read, and every rewrite records from then on"
                ),
            });
        }
        let message = match loss {
            Loss::Removed => format!(
                "This commit would remove the fragment reuse index, which the table started \
                 recording after version {read_version}, the version this commit read"
            ),
            Loss::Version(lost) => format!(
                "This commit would drop or alter the fragment reuse version stamped {}, which \
                 was recorded after version {read_version}, the version this commit read",
                lost.dataset_version
            ),
            Loss::Unrecorded => format!(
                "This rewrite moves rows without recording their row-address moves, but the \
                 table started recording them after version {read_version}, the version this \
                 commit read"
            ),
        };
        Error::retryable_commit_conflict_source(
            base.manifest.version,
            format!("{message}. Plan and run the operation again from the latest version.").into(),
        )
    }

    /// The read version and its v0 history, if it had one.
    async fn frag_reuse_history(&self) -> Result<(u64, Option<Arc<FragReuseIndexDetails>>)> {
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
                (dataset, load_all_indices(dataset).await?.as_ref().clone())
            }
        };
        let details = match sole_v0_entry(&indices).ok().flatten() {
            Some(entry) => Some(load_frag_reuse_index_details(dataset, entry).await?),
            None => None,
        };
        Ok((dataset.manifest.version, details))
    }
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
fn same_version(left: &FragReuseVersion, right: &FragReuseVersion) -> Result<bool> {
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
    base: &Dataset,
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
        let destination_digests = group
            .new_fragments
            .iter()
            .map(|new| {
                proposed_manifest
                    .fragments
                    .iter()
                    .find(|fragment| fragment.id == new.id && fragment.physical_rows.is_some())
                    .map(FragDigest::from)
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
        if recorded.new_frags != destination_digests {
            return Err(mismatch(
                position,
                format!(
                    "it records destinations {:?}, but the rewrite publishes {:?}",
                    recorded.new_frags, destination_digests
                ),
            ));
        }
        let live_rows = live_source_row_addrs(base, &sources).await?;
        if decode_row_addrs(recorded)? != live_rows {
            return Err(mismatch(
                position,
                format!(
                    "its moved rows are not the {} live rows of its sources",
                    live_rows.len()
                ),
            ));
        }
        let written = destination_digests
            .iter()
            .map(|digest| digest.physical_rows as u64)
            .sum::<u64>();
        if written != live_rows.len() {
            return Err(mismatch(
                position,
                format!(
                    "its destinations hold {written} rows, but its sources hold {} live rows",
                    live_rows.len()
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
    use std::collections::HashMap;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
    use lance_core::ROW_ID;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{array, gen_batch};
    use lance_io::object_store::ObjectStore;
    use lance_table::format::BasePath;
    use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
    use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
    use lance_table::io::commit::write_manifest_file_to_path;
    use lance_table::system_index::frag_reuse::FRAG_REUSE_DETAILS_FILE_NAME;
    use prost::Message;
    use roaring::RoaringBitmap;
    use rstest::rstest;

    use lance_index::IndexType;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};

    use super::*;
    use crate::dataset::INDICES_DIR;
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
    use crate::dataset::optimize::{CompactionOptions, compact_files, plan_compaction};
    use crate::dataset::transaction::Transaction;
    use crate::dataset::write::{CommitBuilder, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::append::fragment_reuse_affects_segments;
    use crate::index::frag_reuse::{
        build_frag_reuse_index_metadata, build_new_frag_reuse_index, open_frag_reuse_index,
    };
    use crate::index::frag_reuse_with_stable_row_ids::apply_frag_reuse_with_stable_row_ids_flag;
    use crate::session::Session;
    use crate::session::caches::TransactionKey;
    use crate::utils::test::{
        DatagenExt, FragmentCount, FragmentRowCount, inline_padding_capacity,
        padding_reuse_versions,
    };

    #[derive(Clone, Copy, Debug)]
    enum Storage {
        Inline,
        External,
    }

    fn external_details_file(entry: &IndexMetadata) -> Option<ExternalFile> {
        let details = entry
            .index_details
            .as_ref()
            .unwrap()
            .to_msg::<FragmentReuseIndexDetails>()
            .unwrap();
        match details.content {
            Some(Content::External(file)) => Some(file),
            _ => None,
        }
    }

    async fn rows_by_addr(dataset: &Dataset) -> HashMap<u64, (u64, i32)> {
        let batch = dataset
            .scan()
            .project(&["i"])
            .unwrap()
            .with_row_id()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let addrs = batch[lance_core::ROW_ADDR].as_primitive::<UInt64Type>();
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
        let values = batch["i"].as_primitive::<Int32Type>();
        (0..batch.num_rows())
            .map(|row| (addrs.value(row), (row_ids.value(row), values.value(row))))
            .collect()
    }

    /// Every row live in `before` translates through the latest history to a
    /// row with the same stable row id and value, read from a cold handle.
    async fn assert_translations_hold(before: &HashMap<u64, (u64, i32)>, uri: &str) {
        assert_translations_hold_at(before, uri, None).await;
    }

    /// [`assert_translations_hold`] at `version`, detached or not, when given.
    async fn assert_translations_hold_at(
        before: &HashMap<u64, (u64, i32)>,
        uri: &str,
        version: Option<u64>,
    ) {
        let mut dataset = Dataset::open(uri).await.unwrap();
        if let Some(version) = version {
            dataset = dataset.checkout_version(version).await.unwrap();
        }
        assert_translations_hold_through(before, &dataset).await;
    }

    /// [`assert_translations_hold`] through `dataset`, warm or cold.
    async fn assert_translations_hold_through(
        before: &HashMap<u64, (u64, i32)>,
        dataset: &Dataset,
    ) {
        let frag_reuse = dataset.frag_reuse_index().await.unwrap().unwrap();
        let after = rows_by_addr(dataset).await;
        for (old_addr, row) in before {
            let new_addr = frag_reuse
                .remap_row_id(*old_addr)
                .unwrap_or_else(|| panic!("live row {row:?} at {old_addr} maps to a deletion"));
            assert_eq!(after.get(&new_addr), Some(row), "moved from {old_addr}");
        }
    }

    async fn frag_reuse_entry(dataset: &Dataset) -> IndexMetadata {
        sole_v0_entry(&load_all_indices(dataset).await.unwrap())
            .unwrap()
            .unwrap()
            .clone()
    }

    async fn history(dataset: &Dataset) -> Vec<FragReuseVersion> {
        load_frag_reuse_index_details(dataset, &frag_reuse_entry(dataset).await)
            .await
            .unwrap()
            .versions
            .clone()
    }

    fn assert_same_history(left: &[FragReuseVersion], right: &[FragReuseVersion]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert!(same_version(left, right).unwrap(), "{left:?} != {right:?}");
        }
    }

    /// Padding versions large enough to store the history externally, published
    /// below the publication checks as an earlier writer could have.
    async fn plant_padding_history(dataset: &mut Dataset) {
        let count = inline_padding_capacity() + 1;
        let details = FragReuseIndexDetails {
            versions: padding_reuse_versions(count as u64),
        };
        let new_fragments = (0..count as u32).map(|i| 100_000 + i).collect();
        let entry = build_frag_reuse_index_metadata(dataset, None, details, new_fragments)
            .await
            .unwrap();
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.version += 1;
        manifest.transaction_file = None;
        manifest.transaction_section = None;
        let indices = vec![entry];
        apply_frag_reuse_with_stable_row_ids_flag(&mut manifest, &indices);
        let location = dataset
            .commit_handler
            .commit(
                &mut manifest,
                Some(indices),
                &dataset.base,
                &dataset.object_store,
                write_manifest_file_to_path,
                dataset.manifest_location.naming_scheme,
                None,
            )
            .await
            .unwrap();
        *dataset = dataset.checkout_version(location.version).await.unwrap();
    }

    fn compaction_of(dataset: &Dataset, fragment_ids: &[u64]) -> CompactionOptions {
        CompactionOptions {
            target_rows_per_fragment: 20,
            defer_index_remap: true,
            excluded_fragment_ids: dataset
                .manifest
                .fragments
                .iter()
                .filter(|fragment| !fragment_ids.contains(&fragment.id))
                .map(|fragment| fragment.id as u32)
                .collect(),
            ..Default::default()
        }
    }

    /// A recording table with stable row ids. Deletions in fragment 0 make row
    /// ids and addresses diverge. One recorded compaction rewrote fragments 0-1
    /// and 3-4 as two groups, behind padding versions when the history is
    /// external; fragments 6 and 7, one with a deletion, are left to rewrite.
    /// Returns the rows as they were before any recorded move.
    async fn recording_table(uri: &str, storage: Storage) -> (Dataset, HashMap<u64, (u64, i32)>) {
        let mut dataset = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_dataset_with_params(
                uri,
                FragmentCount::from(8),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 10,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset.delete("i IN (1, 2, 5, 61)").await.unwrap();
        let before = rows_by_addr(&dataset).await;
        if matches!(storage, Storage::External) {
            plant_padding_history(&mut dataset).await;
        }
        // Fragment 2 is left out, so it splits the compaction into two tasks.
        let options = compaction_of(&dataset, &[0, 1, 3, 4]);
        compact_files(&mut dataset, options, None).await.unwrap();
        let recorded = history(&dataset).await;
        assert_eq!(recorded.last().unwrap().groups.len(), 2);
        let details = frag_reuse_entry(&dataset)
            .await
            .index_details
            .unwrap()
            .to_msg::<lance_table::format::pb::FragmentReuseIndexDetails>()
            .unwrap();
        assert_eq!(
            matches!(
                details.content,
                Some(lance_table::format::pb::fragment_reuse_index_details::Content::External(_))
            ),
            matches!(storage, Storage::External)
        );
        (dataset, before)
    }

    /// A real compaction of `fragment_ids` with its fragment ids reserved: the
    /// group a rewrite publishes and the record of its moves.
    async fn rewrite_of(
        dataset: &mut Dataset,
        fragment_ids: &[u64],
    ) -> (RewriteGroup, FragReuseGroup) {
        let plan = plan_compaction(dataset, &compaction_of(dataset, fragment_ids))
            .await
            .unwrap();
        let tasks = plan.compaction_tasks().collect::<Vec<_>>();
        assert_eq!(tasks.len(), 1);
        let task = tasks[0].execute(dataset).await.unwrap();
        let mut new_fragments = task.new_fragments.clone();
        let reservation = Transaction::new(
            dataset.manifest.version,
            Operation::ReserveFragments {
                num_fragments: new_fragments.len() as u32,
            },
            None,
        );
        dataset
            .apply_commit(reservation, &Default::default(), &Default::default())
            .await
            .unwrap();
        let first_id =
            dataset.manifest.max_fragment_id.unwrap() as u64 + 1 - new_fragments.len() as u64;
        for (fragment, id) in new_fragments.iter_mut().zip(first_id..) {
            fragment.id = id;
        }
        let recorded = FragReuseGroup {
            changed_row_addrs: task.row_addrs.unwrap(),
            old_frags: task
                .original_fragments
                .iter()
                .map(FragDigest::from)
                .collect(),
            new_frags: new_fragments.iter().map(FragDigest::from).collect(),
        };
        let group = RewriteGroup {
            old_fragments: task.original_fragments,
            new_fragments,
        };
        (group, recorded)
    }

    async fn commit(
        dataset: &Dataset,
        read_version: u64,
        operation: Operation,
        detached: bool,
    ) -> Result<Dataset> {
        CommitBuilder::new(Arc::new(dataset.clone()))
            .with_detached(detached)
            .execute(Transaction::new(read_version, operation, None))
            .await
    }

    /// Replaces the entry with one holding `versions`, as a raw `CreateIndex` can.
    async fn replacement(dataset: &Dataset, versions: Vec<FragReuseVersion>) -> Operation {
        let current = frag_reuse_entry(dataset).await;
        let details = FragReuseIndexDetails { versions };
        let bitmap = details.new_frag_bitmap();
        let entry = build_frag_reuse_index_metadata(dataset, Some(&current), details, bitmap)
            .await
            .unwrap();
        Operation::CreateIndex {
            new_indices: vec![entry],
            removed_indices: vec![current],
        }
    }

    fn assert_refused(result: Result<Dataset>, message: &str) {
        let error = result.expect_err("the publication must be refused");
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }

    #[derive(Clone, Copy, Debug)]
    enum Change {
        Dropped,
        Shortened,
        Altered,
        Restamped,
        Fabricated,
        Republished,
    }

    /// No publication may drop the history or a version of it, or change a
    /// recorded version's stamp or moves, or record moves no rewrite made. A
    /// republished history that records the same moves is the same history.
    #[rstest]
    #[tokio::test]
    async fn test_recorded_history_is_kept(
        #[values(
            Change::Dropped,
            Change::Shortened,
            Change::Altered,
            Change::Restamped,
            Change::Fabricated,
            Change::Republished
        )]
        change: Change,
        #[values(Storage::Inline, Storage::External)] storage: Storage,
        #[values(false, true)] detached: bool,
    ) {
        let dir = TempStrDir::default();
        let (dataset, before) = recording_table(dir.as_str(), storage).await;
        let recorded = history(&dataset).await;
        let mut versions = recorded.clone();
        let operation = match change {
            Change::Dropped => Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: vec![frag_reuse_entry(&dataset).await],
            },
            Change::Shortened => {
                versions.remove(0);
                replacement(&dataset, versions).await
            }
            Change::Altered => {
                // The same rows in another source order land elsewhere.
                versions.last_mut().unwrap().groups[0].old_frags.reverse();
                replacement(&dataset, versions).await
            }
            Change::Restamped => {
                versions[0].dataset_version += 1;
                replacement(&dataset, versions).await
            }
            Change::Fabricated => {
                let mut fabricated = padding_reuse_versions(1).remove(0);
                fabricated.dataset_version = dataset.manifest.version;
                versions.push(fabricated);
                replacement(&dataset, versions).await
            }
            Change::Republished => {
                versions.last_mut().unwrap().groups.reverse();
                replacement(&dataset, versions).await
            }
        };
        let result = commit(&dataset, dataset.manifest.version, operation, detached).await;

        match change {
            Change::Dropped => assert_refused(result, "Cannot remove the fragment reuse index"),
            Change::Shortened | Change::Altered | Change::Restamped => {
                assert_refused(result, "Cannot drop or alter the fragment reuse version")
            }
            Change::Fabricated => assert_refused(result, "may only be appended by the rewrite"),
            Change::Republished => assert_same_history(&history(&result.unwrap()).await, &recorded),
        }
        let latest = Dataset::open(dir.as_str()).await.unwrap();
        let published = matches!(change, Change::Republished) && !detached;
        assert_eq!(
            latest.manifest.version,
            dataset.manifest.version + u64::from(published)
        );
        assert_same_history(&history(&latest).await, &recorded);
        assert_translations_hold(&before, dir.as_str()).await;
    }

    #[derive(Clone, Copy, Debug)]
    enum Recording {
        Intact,
        MovedRowMissing,
        DeletedRowMoved,
        SourceDigest,
        DestinationDigest,
        SourcesOutOfOrder,
        ExtraGroup,
        TwoVersions,
    }

    /// The version a rewrite appends must record that rewrite exactly: its
    /// groups, its sources as the table holds them, every live source row and
    /// no deleted one, its destinations as published, and ascending sources.
    #[rstest]
    #[tokio::test]
    async fn test_appended_version_records_its_rewrite(
        #[values(
            Recording::Intact,
            Recording::MovedRowMissing,
            Recording::DeletedRowMoved,
            Recording::SourceDigest,
            Recording::DestinationDigest,
            Recording::SourcesOutOfOrder,
            Recording::ExtraGroup,
            Recording::TwoVersions
        )]
        recording: Recording,
        #[values(false, true)] detached: bool,
    ) {
        let dir = TempStrDir::default();
        let (mut dataset, _) = recording_table(dir.as_str(), Storage::Inline).await;
        let before = rows_by_addr(&dataset).await;
        let (mut group, mut recorded) = rewrite_of(&mut dataset, &[6, 7]).await;
        let mut rows =
            RoaringTreemap::deserialize_from(Cursor::new(&recorded.changed_row_addrs)).unwrap();
        let mut groups = vec![recorded.clone()];
        let mut extra_version = None;
        match recording {
            Recording::Intact => {}
            Recording::MovedRowMissing => {
                rows.remove(rows.max().unwrap());
            }
            Recording::DeletedRowMoved => {
                // Fragment 6 lost `i = 61`, its offset 1, before the rewrite.
                rows.remove(rows.max().unwrap());
                rows.insert((6 << 32) | 1);
            }
            Recording::SourceDigest => recorded.old_frags[0].num_deleted_rows += 1,
            Recording::DestinationDigest => recorded.new_frags[0].physical_rows -= 1,
            Recording::SourcesOutOfOrder => {
                group.old_fragments.reverse();
                recorded.old_frags.reverse();
            }
            Recording::ExtraGroup => groups.push(recorded.clone()),
            Recording::TwoVersions => {
                extra_version = Some(FragReuseVersion {
                    dataset_version: dataset.manifest.version,
                    groups: vec![recorded.clone()],
                });
            }
        }
        let mut changed_row_addrs = Vec::new();
        rows.serialize_into(&mut changed_row_addrs).unwrap();
        recorded.changed_row_addrs = changed_row_addrs;
        groups[0] = recorded;
        let bitmap = RoaringBitmap::from_iter(group.new_fragments.iter().map(|f| f.id as u32));
        let version = dataset.manifest.version;
        let mut entry = build_new_frag_reuse_index(&mut dataset, groups, bitmap.clone(), version)
            .await
            .unwrap();
        if let Some(extra) = extra_version {
            let mut versions = history(&dataset).await;
            versions.extend([
                load_frag_reuse_index_details(&dataset, &entry)
                    .await
                    .unwrap()
                    .versions
                    .last()
                    .unwrap()
                    .clone(),
                extra,
            ]);
            entry = build_frag_reuse_index_metadata(
                &dataset,
                Some(&frag_reuse_entry(&dataset).await),
                FragReuseIndexDetails { versions },
                bitmap,
            )
            .await
            .unwrap();
        }
        let operation = Operation::Rewrite {
            groups: vec![group],
            rewritten_indices: vec![],
            frag_reuse_index: Some(entry),
        };
        let recorded_before = history(&dataset).await;
        let result = commit(&dataset, dataset.manifest.version, operation, detached).await;

        let latest = Dataset::open(dir.as_str()).await.unwrap();
        match recording {
            Recording::Intact => {
                let published = result.unwrap();
                assert_eq!(history(&published).await.len(), recorded_before.len() + 1);
                let version = detached.then_some(published.manifest.version);
                assert_translations_hold_at(&before, dir.as_str(), version).await;
            }
            Recording::SourcesOutOfOrder => {
                assert_refused(result, "recorded moves need ascending source ids")
            }
            Recording::ExtraGroup => assert_refused(result, "records 2 group(s)"),
            // On the main chain the rebase's shape check refuses it first.
            Recording::TwoVersions if detached => {
                assert_refused(result, "appends 2 fragment reuse versions")
            }
            Recording::TwoVersions => assert_refused(result, "plus one version of its own"),
            _ => assert_refused(result, "does not record rewrite group 0"),
        }
        if !matches!(recording, Recording::Intact) || detached {
            assert_eq!(latest.manifest.version, dataset.manifest.version);
            assert_same_history(&history(&latest).await, &recorded_before);
        }
    }

    /// A rewrite that skips recording on a table that already records was
    /// built wrong, whichever way it reaches publication.
    #[rstest]
    #[tokio::test]
    async fn test_rewrite_without_its_moves_is_refused(#[values(false, true)] detached: bool) {
        let dir = TempStrDir::default();
        let (mut dataset, before) = recording_table(dir.as_str(), Storage::Inline).await;
        let (group, _) = rewrite_of(&mut dataset, &[6, 7]).await;
        let recorded = history(&dataset).await;
        let operation = Operation::Rewrite {
            groups: vec![group],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        };
        assert_refused(
            commit(&dataset, dataset.manifest.version, operation, detached).await,
            "already recorded them in its fragment reuse index",
        );
        let latest = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(latest.manifest.version, dataset.manifest.version);
        assert_same_history(&history(&latest).await, &recorded);
        assert_translations_hold(&before, dir.as_str()).await;
    }

    /// A detached commit is checked against the manifest it builds on. History
    /// that base holds beyond the commit's read version is a retryable conflict;
    /// history the commit read is not.
    #[rstest]
    #[tokio::test]
    async fn test_detached_commit_is_checked_against_its_base(
        #[values(false, true)] read_the_base: bool,
    ) {
        let dir = TempStrDir::default();
        let (mut dataset, _) = recording_table(dir.as_str(), Storage::Inline).await;
        let read = dataset.clone();
        let (group, recorded) = rewrite_of(&mut dataset, &[6, 7]).await;
        let bitmap = RoaringBitmap::from_iter(group.new_fragments.iter().map(|f| f.id as u32));
        let version = dataset.manifest.version;
        let entry = build_new_frag_reuse_index(&mut dataset, vec![recorded], bitmap, version)
            .await
            .unwrap();
        let base = commit(
            &dataset,
            dataset.manifest.version,
            Operation::Rewrite {
                groups: vec![group],
                rewritten_indices: vec![],
                frag_reuse_index: Some(entry),
            },
            false,
        )
        .await
        .unwrap();
        assert_eq!(history(&base).await.len(), history(&read).await.len() + 1);

        let read_version = if read_the_base {
            base.manifest.version
        } else {
            read.manifest.version
        };
        let mut operation = replacement(&read, history(&read).await).await;
        if let Operation::CreateIndex {
            removed_indices, ..
        } = &mut operation
        {
            *removed_indices = vec![frag_reuse_entry(&base).await];
        }
        let error = commit(&base, read_version, operation, true)
            .await
            .expect_err("dropping a version must be refused");
        if read_the_base {
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        } else {
            assert!(
                matches!(error, Error::RetryableCommitConflict { .. }),
                "{error}"
            );
            assert!(
                error.to_string().contains("recorded after version"),
                "{error}"
            );
        }
    }

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

    async fn zone_map_entry(dataset: &Dataset) -> IndexMetadata {
        load_all_indices(dataset)
            .await
            .unwrap()
            .iter()
            .find(|index| index.name == "i_idx")
            .cloned()
            .unwrap()
    }

    /// `entry` with its versions stored inline in the given order, which the
    /// encoder would sort by stamp.
    fn stored_in_order(entry: &IndexMetadata, versions: &[FragReuseVersion]) -> IndexMetadata {
        let details = FragmentReuseIndexDetails {
            content: Some(Content::Inline(InlineContent {
                legacy_versions: versions.iter().map(Into::into).collect(),
                transitions: vec![],
            })),
        };
        IndexMetadata {
            index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
            ..entry.clone()
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Stamp {
        /// The version the commit builds on, as a compaction planned there
        /// stamps it.
        Base,
        /// Above it, which only makes index maintenance do more than it needs.
        AboveBase,
        /// The version the rewrite was planned at, below an index built over
        /// its sources since.
        Planned,
        /// Below the stamp of the recorded version it is stored after.
        BelowRecorded,
    }

    /// The appended version's stamp must make index maintenance take every
    /// index the base holds over the rewritten fragments for one built
    /// before the rewrite. A detached commit keeps the stamp it was given, so
    /// it is checked against the manifest it builds on; the main chain
    /// restamps it with the version it publishes on top of.
    #[rstest]
    #[tokio::test]
    async fn test_appended_stamp_follows_every_index_over_the_sources(
        #[values(Stamp::Base, Stamp::AboveBase, Stamp::Planned, Stamp::BelowRecorded)] stamp: Stamp,
        #[values(false, true)] detached: bool,
    ) {
        let dir = TempStrDir::default();
        let (mut dataset, _) = recording_table(dir.as_str(), Storage::Inline).await;
        let recorded = history(&dataset).await;
        let (group, moves) = rewrite_of(&mut dataset, &[6, 7]).await;
        let planned_at = dataset.manifest.version;
        // Meanwhile another writer appends, and an index is built over every
        // fragment, the ones being rewritten included.
        let batch = arrow_array::record_batch!(("i", Int32, [1000, 1001, 1002])).unwrap();
        dataset
            .append(
                arrow_array::RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();
        create_zone_map(&mut dataset).await;
        let zone_map = zone_map_entry(&dataset).await;
        assert!(zone_map.dataset_version > planned_at);
        let before = rows_by_addr(&dataset).await;
        let base_version = dataset.manifest.version;

        let own_stamp = match stamp {
            Stamp::Base => base_version,
            Stamp::AboveBase => base_version + 5,
            Stamp::Planned => planned_at,
            Stamp::BelowRecorded => recorded.last().unwrap().dataset_version - 1,
        };
        let own = FragReuseVersion {
            dataset_version: own_stamp,
            groups: vec![moves],
        };
        let bitmap = RoaringBitmap::from_iter(group.new_fragments.iter().map(|f| f.id as u32));
        let entry = build_new_frag_reuse_index(&mut dataset, own.groups.clone(), bitmap, own_stamp)
            .await
            .unwrap();
        let versions = recorded.iter().cloned().chain([own]).collect::<Vec<_>>();
        let entry = stored_in_order(&entry, &versions);
        // Whether segment merge would translate the zone map through this
        // history; remap and cleanup make the same comparison.
        let proposed = open_frag_reuse_index(entry.uuid, &FragReuseIndexDetails { versions })
            .await
            .unwrap();
        assert_eq!(
            fragment_reuse_affects_segments(&proposed, [&zone_map]),
            own_stamp >= zone_map.dataset_version
        );
        let operation = Operation::Rewrite {
            groups: vec![group],
            rewritten_indices: vec![],
            frag_reuse_index: Some(entry),
        };
        let result = commit(&dataset, base_version, operation, detached).await;

        let latest = Dataset::open(dir.as_str()).await.unwrap();
        let refusal = match stamp {
            _ if !detached => None,
            Stamp::Base | Stamp::AboveBase => None,
            Stamp::Planned => Some("covers its source fragments and was built at version"),
            Stamp::BelowRecorded => Some("the history is kept in stamp order"),
        };
        if let Some(message) = refusal {
            assert_refused(result, message);
            assert_eq!(latest.manifest.version, base_version);
            assert_same_history(&history(&latest).await, &recorded);
            return;
        }
        let published = result.unwrap();
        let version = published.manifest.version;
        assert_eq!(latest.manifest.version, base_version + u64::from(!detached));
        // Read back cold, from the version itself.
        let reopened = Dataset::open(dir.as_str())
            .await
            .unwrap()
            .checkout_version(version)
            .await
            .unwrap();
        let published_history = history(&reopened).await;
        assert_same_history(&published_history[..recorded.len()], &recorded);
        assert_eq!(published_history.len(), recorded.len() + 1);
        let expected_stamp = if detached { own_stamp } else { base_version };
        assert_eq!(
            published_history.last().unwrap().dataset_version,
            expected_stamp
        );
        let frag_reuse = reopened.frag_reuse_index().await.unwrap().unwrap();
        assert!(fragment_reuse_affects_segments(
            &frag_reuse,
            [&zone_map_entry(&reopened).await]
        ));
        assert_translations_hold_at(&before, dir.as_str(), Some(version)).await;
    }

    /// An ordinary compaction of the two newest fragments, adjacent in id
    /// order, records its moves, and every row still translates.
    async fn assert_ordinary_compaction_records(uri: &str, before: &HashMap<u64, (u64, i32)>) {
        let mut compactor = Dataset::open(uri).await.unwrap();
        let recorded = history(&compactor).await.len();
        let newest = compactor
            .manifest
            .fragments
            .iter()
            .rev()
            .take(2)
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>();
        let options = CompactionOptions {
            target_rows_per_fragment: 1_000,
            defer_index_remap: false,
            ..compaction_of(&compactor, &newest)
        };
        let metrics = compact_files(&mut compactor, options, None).await.unwrap();
        assert_eq!(metrics.fragments_removed, 2);
        let latest = Dataset::open(uri).await.unwrap();
        assert_eq!(history(&latest).await.len(), recorded + 1);
        assert_translations_hold(before, uri).await;
    }

    /// The promotion `restored` was refused: the main chain kept its version
    /// and history, and ordinary compaction keeps recording on it.
    async fn assert_promotion_refused(
        restored: Result<()>,
        message: &str,
        uri: &str,
        version: u64,
        recorded: &[FragReuseVersion],
        before: &HashMap<u64, (u64, i32)>,
    ) {
        let error = restored.expect_err("the promotion must be refused");
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
        let latest = Dataset::open(uri).await.unwrap();
        assert_eq!(latest.manifest.version, version);
        assert_same_history(&history(&latest).await, recorded);
        assert_ordinary_compaction_records(uri, before).await;
    }

    /// Restoring a detached version promotes its history onto the main chain
    /// at a new version. A stamp from the detached range sorts after every
    /// version that chain will publish, so no later rewrite could follow it:
    /// the restore is refused before it publishes.
    #[tokio::test]
    async fn test_promoting_a_detached_stamp_is_refused() {
        use lance_table::format::DETACHED_VERSION_MASK;

        let dir = TempStrDir::default();
        let (mut dataset, before) = recording_table(dir.as_str(), Storage::Inline).await;
        let recorded = history(&dataset).await;
        let (group, moves) = rewrite_of(&mut dataset, &[6, 7]).await;
        let stamp = DETACHED_VERSION_MASK | 5;
        let bitmap = RoaringBitmap::from_iter(group.new_fragments.iter().map(|f| f.id as u32));
        let entry = build_new_frag_reuse_index(&mut dataset, vec![moves], bitmap, stamp)
            .await
            .unwrap();
        let rewrite = Operation::Rewrite {
            groups: vec![group],
            rewritten_indices: vec![],
            frag_reuse_index: Some(entry),
        };
        let detached = commit(&dataset, dataset.manifest.version, rewrite, true)
            .await
            .unwrap();
        assert_eq!(
            history(&detached).await.last().unwrap().dataset_version,
            stamp
        );

        let mut promoted = dataset
            .checkout_version(detached.manifest.version)
            .await
            .unwrap();
        let restored = promoted.restore().await;
        let message = &format!("stamped {stamp}");
        let version = dataset.manifest.version;
        let refused =
            assert_promotion_refused(restored, message, dir.as_str(), version, &recorded, &before);
        Box::pin(refused).await;
    }

    /// The same for an index built on a detached version: a later rewrite of
    /// the fragments it covers could not be stamped at or above its version.
    #[tokio::test]
    async fn test_promoting_an_index_built_on_a_detached_version_is_refused() {
        let dir = TempStrDir::default();
        let (dataset, before) = recording_table(dir.as_str(), Storage::Inline).await;
        let recorded = history(&dataset).await;
        // Staged rows, and an index built over them where they were staged.
        let batch = arrow_array::record_batch!(("i", Int32, [1000, 1001, 1002])).unwrap();
        let append = crate::dataset::write::InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: crate::dataset::WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(vec![batch])
            .await
            .unwrap();
        let mut staged = CommitBuilder::new(Arc::new(dataset.clone()))
            .with_detached(true)
            .execute(append)
            .await
            .unwrap();
        let index = staged
            .create_index_builder(
                &["i"],
                IndexType::ZoneMap,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
            )
            .name("late".into())
            .execute_uncommitted()
            .await
            .unwrap();
        assert!(lance_table::format::is_detached_version(
            index.dataset_version
        ));
        let indexed = commit(
            &staged,
            staged.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![index],
                removed_indices: vec![],
            },
            true,
        )
        .await
        .unwrap();

        let mut promoted = dataset
            .checkout_version(indexed.manifest.version)
            .await
            .unwrap();
        let restored = promoted.restore().await;
        let version = dataset.manifest.version;
        let refused = assert_promotion_refused(
            restored,
            "index 'late'",
            dir.as_str(),
            version,
            &recorded,
            &before,
        );
        Box::pin(refused).await;
    }

    /// A detached version whose stamps and index versions are main-chain
    /// versions, as a compaction planned on the main chain records them,
    /// promotes, and recording continues after it.
    #[tokio::test]
    async fn test_promoting_a_detached_version_within_the_main_chain_keeps_recording() {
        let dir = TempStrDir::default();
        let (mut dataset, before) = recording_table(dir.as_str(), Storage::Inline).await;
        create_zone_map(&mut dataset).await;
        let recorded = history(&dataset).await;
        let (group, moves) = rewrite_of(&mut dataset, &[6, 7]).await;
        let stamp = dataset.manifest.version;
        let bitmap = RoaringBitmap::from_iter(group.new_fragments.iter().map(|f| f.id as u32));
        let entry = build_new_frag_reuse_index(&mut dataset, vec![moves], bitmap, stamp)
            .await
            .unwrap();
        let rewrite = Operation::Rewrite {
            groups: vec![group],
            rewritten_indices: vec![],
            frag_reuse_index: Some(entry),
        };
        let detached = commit(&dataset, stamp, rewrite, true).await.unwrap();

        let mut promoted = dataset
            .checkout_version(detached.manifest.version)
            .await
            .unwrap();
        promoted.restore().await.unwrap();
        let latest = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(latest.manifest.version, dataset.manifest.version + 1);
        assert_eq!(history(&latest).await.len(), recorded.len() + 1);
        assert_translations_hold(&before, dir.as_str()).await;
        assert_ordinary_compaction_records(dir.as_str(), &before).await;
        let latest = Dataset::open(dir.as_str()).await.unwrap();
        assert_answers_through(&latest, "i_idx", "i >= 50").await;
    }

    /// `entry` holding `versions` in the order given, stored as `storage` says
    /// whatever their size.
    async fn stored_as(
        dataset: &Dataset,
        entry: &IndexMetadata,
        versions: &[FragReuseVersion],
        storage: Storage,
    ) -> IndexMetadata {
        let uuid = uuid::Uuid::new_v4();
        stored_under(
            dataset,
            entry,
            uuid,
            FRAG_REUSE_DETAILS_FILE_NAME,
            versions,
            storage,
        )
        .await
    }

    /// [`stored_as`] under `uuid`, an external history in `file_name`.
    async fn stored_under(
        dataset: &Dataset,
        entry: &IndexMetadata,
        uuid: uuid::Uuid,
        file_name: &str,
        versions: &[FragReuseVersion],
        storage: Storage,
    ) -> IndexMetadata {
        let content = InlineContent {
            legacy_versions: versions.iter().map(Into::into).collect(),
            transitions: vec![],
        };
        let content = match storage {
            Storage::Inline => Content::Inline(content),
            Storage::External => {
                let bytes = content.encode_to_vec();
                let path = dataset.indices_dir().join(uuid.to_string()).join(file_name);
                dataset.object_store.put(&path, &bytes).await.unwrap();
                Content::External(ExternalFile {
                    path: file_name.to_string(),
                    offset: 0,
                    size: bytes.len() as u64,
                })
            }
        };
        let details = FragmentReuseIndexDetails {
            content: Some(content),
        };
        IndexMetadata {
            uuid,
            index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
            ..entry.clone()
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Encoding {
        /// The moved rows in the other container kind: arrays where the
        /// recording wrote runs, runs where it wrote arrays.
        OtherContainers,
        /// The same versions stored externally if they were inline, and
        /// inline if they were external.
        OtherStorage,
    }

    /// `group`'s moved rows serialized as arrays, or as runs if the recording
    /// already used arrays.
    fn in_other_containers(group: &FragReuseGroup) -> Vec<u8> {
        let mut rows = decode_row_addrs(group)
            .unwrap()
            .iter()
            .collect::<RoaringTreemap>();
        let mut bytes = Vec::new();
        rows.serialize_into(&mut bytes).unwrap();
        if bytes == group.changed_row_addrs {
            rows.optimize();
            bytes.clear();
            rows.serialize_into(&mut bytes).unwrap();
        }
        bytes
    }

    /// A history republished in another encoding records the same moves. It
    /// publishes as the same history, and a cold reader translates through
    /// the new encoding alike.
    #[rstest]
    #[tokio::test]
    async fn test_equivalent_encodings_are_the_same_history(
        #[values(Encoding::OtherContainers, Encoding::OtherStorage)] encoding: Encoding,
        #[values(Storage::Inline, Storage::External)] storage: Storage,
    ) {
        let dir = TempStrDir::default();
        let (dataset, before) = recording_table(dir.as_str(), storage).await;
        let recorded = history(&dataset).await;
        let (versions, stored) = match encoding {
            Encoding::OtherContainers => {
                let mut versions = recorded.clone();
                for group in versions.iter_mut().flat_map(|version| &mut version.groups) {
                    group.changed_row_addrs = in_other_containers(group);
                }
                let last = recorded.len() - 1;
                for (left, right) in versions[last].groups.iter().zip(&recorded[last].groups) {
                    assert_ne!(left.changed_row_addrs, right.changed_row_addrs);
                }
                (versions, storage)
            }
            Encoding::OtherStorage => {
                let other = match storage {
                    Storage::Inline => Storage::External,
                    Storage::External => Storage::Inline,
                };
                (recorded.clone(), other)
            }
        };
        let current = frag_reuse_entry(&dataset).await;
        let entry = stored_as(&dataset, &current, &versions, stored).await;
        let operation = Operation::CreateIndex {
            new_indices: vec![entry],
            removed_indices: vec![current],
        };
        commit(&dataset, dataset.manifest.version, operation, false)
            .await
            .unwrap();

        let latest = Dataset::open(dir.as_str()).await.unwrap();
        let published = frag_reuse_entry(&latest).await;
        assert_eq!(
            external_details_file(&published).is_some(),
            matches!(stored, Storage::External)
        );
        assert_eq!(history(&latest).await, versions);
        assert_same_history(&history(&latest).await, &recorded);
        assert_translations_hold(&before, dir.as_str()).await;
    }

    /// `uri` at exactly `version`, in `session` or a fresh one.
    async fn open_at(uri: &str, version: u64, session: Option<Arc<Session>>) -> Dataset {
        let mut builder = DatasetBuilder::from_uri(uri).with_version(version);
        if let Some(session) = session {
            builder = builder.with_session(session);
        }
        builder.load().await.unwrap()
    }

    #[derive(Clone, Copy, Debug)]
    enum Identity {
        /// The UUID of the history a reader warmed.
        Reused,
        Fresh,
    }

    /// A raw writer can publish a different history under a UUID the table
    /// already used: the one of the entry it replaces or, after a restore
    /// dropped that entry, an earlier one. It is a different history all the
    /// same. The handle the commit returns, a later handle in the same session
    /// and a cold one at the exact version must translate every row to the same
    /// row, and answer an indexed query as a scan does.
    #[rstest]
    #[tokio::test]
    async fn test_a_history_is_identified_by_its_content_not_its_uuid(
        #[values(Identity::Reused, Identity::Fresh)] identity: Identity,
        #[values(false, true)] after_restore: bool,
        #[values(Storage::Inline, Storage::External)] storage: Storage,
        #[values(false, true)] detached: bool,
    ) {
        const FILTER: &str = "i >= 15 AND i < 45";
        let dir = TempStrDir::default();
        let uri = dir.as_str();
        let mut dataset = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_dataset_with_params(
                uri,
                FragmentCount::from(8),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 10,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset.delete("i IN (1, 2, 21, 33)").await.unwrap();
        create_zone_map(&mut dataset).await;
        let unrecorded = dataset.manifest.version;
        let options = compaction_of(&dataset, &[0, 1]);
        compact_files(&mut dataset, options, None).await.unwrap();
        if matches!(storage, Storage::External) {
            let current = frag_reuse_entry(&dataset).await;
            let entry = stored_as(&dataset, &current, &history(&dataset).await, storage).await;
            let operation = Operation::CreateIndex {
                new_indices: vec![entry],
                removed_indices: vec![current],
            };
            dataset = commit(&dataset, dataset.manifest.version, operation, false)
                .await
                .unwrap();
        }
        // A reader warms the history, the coverage it derives and the zone map.
        let warmed = frag_reuse_entry(&dataset).await;
        let warmed_history = history(&dataset).await;
        assert_translations_hold_through(&rows_by_addr(&dataset).await, &dataset).await;
        assert_answers_through(&dataset, "i_idx", FILTER).await;
        if after_restore {
            dataset = dataset.checkout_version(unrecorded).await.unwrap();
            dataset.restore().await.unwrap();
            assert!(dataset.frag_reuse_index().await.unwrap().is_none());
        }

        let before = rows_by_addr(&dataset).await;
        let (group, moves) = rewrite_of(&mut dataset, &[2, 3]).await;
        let mut versions = if after_restore {
            vec![]
        } else {
            warmed_history
        };
        versions.push(FragReuseVersion {
            dataset_version: dataset.manifest.version,
            groups: vec![moves],
        });
        let uuid = match identity {
            Identity::Reused => warmed.uuid,
            Identity::Fresh => uuid::Uuid::new_v4(),
        };
        // A file of its own, so a reused UUID leaves the warmed one in place.
        let file_name = format!("details-{}.binpb", versions.len());
        let mut entry = stored_under(&dataset, &warmed, uuid, &file_name, &versions, storage).await;
        entry.fragment_bitmap = Some(
            FragReuseIndexDetails {
                versions: versions.clone(),
            }
            .new_frag_bitmap(),
        );
        entry.dataset_version = dataset.manifest.version;
        let rewrite = Operation::Rewrite {
            groups: vec![group],
            rewritten_indices: vec![],
            frag_reuse_index: Some(entry),
        };
        let published = commit(&dataset, dataset.manifest.version, rewrite, detached)
            .await
            .unwrap();
        let version = published.manifest.version;

        let later = open_at(uri, version, Some(dataset.session())).await;
        let cold = open_at(uri, version, None).await;
        let mut failures = Vec::new();
        for (handle, name) in [
            (&published, "returned"),
            (&later, "same session"),
            (&cold, "cold"),
        ] {
            for failure in reader_failures(handle, &before, &versions, FILTER).await {
                failures.push(format!("{name}: {failure}"));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// Where `dataset` disagrees with the published `versions`: the history it
    /// decodes, where it translates each row of `before`, and what a query
    /// through the zone map answers compared with a scan.
    async fn reader_failures(
        dataset: &Dataset,
        before: &HashMap<u64, (u64, i32)>,
        versions: &[FragReuseVersion],
        filter: &str,
    ) -> Vec<String> {
        let mut failures = Vec::new();
        let decoded = dataset.frag_reuse_index().await.unwrap().unwrap();
        let same_history = decoded.details.versions.len() == versions.len()
            && decoded
                .details
                .versions
                .iter()
                .zip(versions)
                .all(|(left, right)| same_version(left, right).unwrap());
        if !same_history {
            failures.push(format!(
                "decodes {} version(s) stamped {:?}, not the {} published",
                decoded.details.versions.len(),
                decoded
                    .details
                    .versions
                    .iter()
                    .map(|version| version.dataset_version)
                    .collect::<Vec<_>>(),
                versions.len()
            ));
        }
        let after = rows_by_addr(dataset).await;
        let misplaced = before
            .iter()
            .filter(|(old_addr, row)| {
                decoded
                    .remap_row_id(**old_addr)
                    .is_none_or(|new_addr| after.get(&new_addr) != Some(*row))
            })
            .count();
        if misplaced > 0 {
            failures.push(format!(
                "translates {misplaced} of {} rows to another row",
                before.len()
            ));
        }
        let mut answers = Vec::new();
        for use_scalar_index in [false, true] {
            let mut scanner = dataset.scan();
            scanner
                .filter(filter)
                .unwrap()
                .project(&["i"])
                .unwrap()
                .use_scalar_index(use_scalar_index);
            let batch = scanner.try_into_batch().await.unwrap();
            let mut values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
            values.sort_unstable();
            answers.push(values);
        }
        if answers[1] != answers[0] {
            failures.push(format!(
                "answers {filter} through the zone map with {} rows, a scan with {}",
                answers[1].len(),
                answers[0].len()
            ));
        }
        let mut scanner = dataset.scan();
        scanner.filter(filter).unwrap();
        let plan = scanner.explain_plan(false).await.unwrap();
        if !plan.contains("@i_idx(") {
            failures.push(format!("does not query through the zone map: {plan}"));
        }
        failures
    }

    /// Queries through `index_name` answer `filter` as a full scan does.
    async fn assert_answers_through(dataset: &Dataset, index_name: &str, filter: &str) {
        let mut answers = Vec::new();
        for use_scalar_index in [false, true] {
            let mut scanner = dataset.scan();
            scanner
                .filter(filter)
                .unwrap()
                .project(&["i"])
                .unwrap()
                .use_scalar_index(use_scalar_index);
            if use_scalar_index {
                let plan = scanner.explain_plan(false).await.unwrap();
                assert!(
                    plan.contains(&format!("@{index_name}(")),
                    "{filter}: {plan}"
                );
            }
            let batch = scanner.try_into_batch().await.unwrap();
            let mut values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
            values.sort_unstable();
            answers.push(values);
        }
        assert!(!answers[0].is_empty(), "{filter}");
        assert_eq!(answers[1], answers[0], "{filter} through {index_name}");
    }

    /// Whether `dataset`'s session caches the transaction of `version` with
    /// the history a rewrite carries in memory; `None` if it caches none.
    async fn cached_rewrite_history(dataset: &Dataset, version: u64) -> Option<bool> {
        let transaction = dataset
            .metadata_cache
            .get_with_key(&TransactionKey { version })
            .await?;
        let Operation::Rewrite {
            frag_reuse_index, ..
        } = &transaction.operation
        else {
            panic!("version {version} is not a rewrite");
        };
        Some(frag_reuse_index.is_some())
    }

    /// The stable-row-id case of the F6 race: an index built before a
    /// recorded compaction commits after it, and a cleanup that read before
    /// that index commits on either side of it. Cleanup never trims this
    /// table's history, so the index keeps the mapping it needs. The race needs
    /// the session that compacted, whose cached rewrite carries its history:
    /// another one reads the rewrite back without it, and the index retries
    /// instead. The table is in memory: a store that lists a manifest under
    /// another e_tag than its write returned (CIFS, where mtime settles after
    /// the write) makes the commit reload it, and the reloaded transaction
    /// replaces the cached one.
    #[rstest]
    #[tokio::test]
    async fn test_cleanup_racing_index_creation_keeps_the_mapping_it_needs(
        #[values(false, true)] index_first: bool,
        #[values(false, true)] separate_sessions: bool,
    ) {
        let uri = format!("shared-memory://{}/table", uuid::Uuid::new_v4());
        let mut dataset = gen_batch()
            .col("i", array::step::<Int32Type>())
            .col("j", array::step::<Int32Type>())
            .into_dataset_with_params(
                &uri,
                FragmentCount::from(4),
                FragmentRowCount::from(50),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 50,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset.delete("i IN (1, 2, 60)").await.unwrap();
        create_zone_map(&mut dataset).await;
        let before = rows_by_addr(&dataset).await;
        let handle = async |dataset: &Dataset| {
            if separate_sessions {
                Dataset::open(&uri).await.unwrap()
            } else {
                dataset.clone()
            }
        };
        let mut pre = handle(&dataset).await;
        let options = CompactionOptions {
            target_rows_per_fragment: 10_000,
            defer_index_remap: true,
            ..Default::default()
        };
        compact_files(&mut dataset, options, None).await.unwrap();
        let rewrite = dataset.manifest.version;
        crate::dataset::optimize::remapping::remap_column_index(
            &mut dataset,
            &["i"],
            Some("i_idx".into()),
        )
        .await
        .unwrap();
        let recorded = history(&dataset).await;
        let mut cleaner = handle(&dataset).await;
        assert_eq!(
            cached_rewrite_history(&pre, rewrite).await,
            (!separate_sessions).then_some(true),
            "the rewrite cached in the late index's session"
        );

        let create_late = async |pre: &mut Dataset| {
            let created = pre
                .create_index_builder(
                    &["j"],
                    IndexType::ZoneMap,
                    &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
                )
                .name("late".into())
                .await;
            (created, cached_rewrite_history(pre, rewrite).await)
        };
        let (created, read) = if index_first {
            let late = create_late(&mut pre).await;
            cleanup_frag_reuse_index(&mut cleaner).await.unwrap();
            late
        } else {
            cleanup_frag_reuse_index(&mut cleaner).await.unwrap();
            create_late(&mut pre).await
        };
        assert_eq!(
            read,
            Some(!separate_sessions),
            "the rewrite as the late index's commit read it"
        );

        let latest = Dataset::open(&uri).await.unwrap();
        assert_same_history(&history(&latest).await, &recorded);
        assert_translations_hold(&before, &uri).await;
        if separate_sessions {
            let error = created.expect_err("the index must be rebuilt after the rewrite");
            assert!(
                matches!(error, Error::RetryableCommitConflict { .. }),
                "{error}"
            );
            return;
        }
        created.unwrap();
        let late = load_all_indices(&latest)
            .await
            .unwrap()
            .iter()
            .find(|index| index.name == "late")
            .cloned()
            .unwrap();
        assert!(late.dataset_version <= recorded.last().unwrap().dataset_version);
        assert_answers_through(&latest, "late", "j < 150").await;
        assert_answers_through(&latest, "late", "j >= 40 AND j < 120").await;
    }

    #[derive(Clone, Copy, Debug)]
    enum Pruning {
        /// `cleanup_frag_reuse_index`.
        Cleanup,
        /// A raw replacement of the entry with no versions.
        RawTrim,
        /// A raw republish of the history the cleaner read, nothing trimmed.
        RawRepublish,
    }

    /// A cleanup and a recorded compaction race from separate handles, in
    /// either order. Cleanup keeps a stable-row-id table's history; a raw
    /// trim is refused; a republish of what the cleaner read is accepted
    /// unless a version was recorded since, which it would drop.
    #[rstest]
    #[tokio::test]
    async fn test_cleanup_and_a_recorded_rewrite_keep_the_history_in_either_order(
        #[values(Pruning::Cleanup, Pruning::RawTrim, Pruning::RawRepublish)] pruning: Pruning,
        #[values(false, true)] rewrite_first: bool,
    ) {
        let dir = TempStrDir::default();
        let (dataset, before) = recording_table(dir.as_str(), Storage::Inline).await;
        let recorded = history(&dataset).await;
        let mut cleaner = Dataset::open(dir.as_str()).await.unwrap();
        let mut compactor = Dataset::open(dir.as_str()).await.unwrap();
        let operation = match pruning {
            Pruning::Cleanup => None,
            Pruning::RawTrim => Some(replacement(&cleaner, vec![]).await),
            Pruning::RawRepublish => Some(replacement(&cleaner, recorded.clone()).await),
        };
        let compaction = compaction_of(&compactor, &[6, 7]);

        if rewrite_first {
            compact_files(&mut compactor, compaction.clone(), None)
                .await
                .unwrap();
        }
        let trimmed = match operation {
            None => cleanup_frag_reuse_index(&mut cleaner).await,
            Some(operation) => commit(&cleaner, cleaner.manifest.version, operation, false)
                .await
                .map(|_| ()),
        };
        if !rewrite_first {
            compact_files(&mut compactor, compaction, None)
                .await
                .unwrap();
        }

        match (pruning, rewrite_first) {
            (Pruning::Cleanup, _) | (Pruning::RawRepublish, false) => trimmed.unwrap(),
            (Pruning::RawTrim, false) => {
                let error = trimmed.expect_err("a trim must be refused");
                assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            }
            // The rebase sees the entry it replaces changed first.
            (Pruning::RawTrim | Pruning::RawRepublish, true) => {
                let error = trimmed.expect_err("a stale replacement must be refused");
                assert!(
                    matches!(error, Error::RetryableCommitConflict { .. }),
                    "{error}"
                );
            }
        }
        let latest = Dataset::open(dir.as_str()).await.unwrap();
        let kept = history(&latest).await;
        assert_eq!(kept.len(), recorded.len() + 1);
        assert_same_history(&kept[..recorded.len()], &recorded);
        assert_translations_hold(&before, dir.as_str()).await;
    }

    /// Restore and overwrite replace the history by design.
    #[tokio::test]
    async fn test_restore_and_overwrite_replace_the_history() {
        let dir = TempStrDir::default();
        let (mut dataset, _) = recording_table(dir.as_str(), Storage::Inline).await;
        let one_version = dataset.manifest.version;
        let options = compaction_of(&dataset, &[6, 7]);
        compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(history(&dataset).await.len(), 2);

        let mut restored = dataset.checkout_version(one_version).await.unwrap();
        restored.restore().await.unwrap();
        assert_eq!(
            history(&Dataset::open(dir.as_str()).await.unwrap())
                .await
                .len(),
            1
        );

        let batch = arrow_array::record_batch!(("i", Int32, [1, 2, 3])).unwrap();
        let overwritten = Dataset::write(
            arrow_array::RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            dir.as_str(),
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert!(overwritten.frag_reuse_index().await.unwrap().is_none());
    }

    fn has_frag_reuse_flag(dataset: &Dataset) -> bool {
        let flag = lance_table::feature_flags::FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
        let in_reader = dataset.manifest.reader_feature_flags & flag != 0;
        assert_eq!(in_reader, dataset.manifest.writer_feature_flags & flag != 0);
        in_reader
    }

    async fn clone_table(source: &mut Dataset, uri: &str, is_shallow: bool) -> Dataset {
        let version = source.manifest.version;
        if is_shallow {
            source.shallow_clone(uri, version, None).await.unwrap();
        } else {
            source.deep_clone(uri, version, None).await.unwrap();
        }
        Dataset::open(uri).await.unwrap()
    }

    /// A clone starts its history from the source's: stable row ids and every
    /// recorded version, whether the details stay with the source (shallow) or
    /// are copied (deep).
    #[rstest]
    #[tokio::test]
    async fn test_clone_keeps_the_recorded_history(
        #[values(false, true)] is_shallow: bool,
        #[values(Storage::Inline, Storage::External)] storage: Storage,
    ) {
        let dir = TempStrDir::default();
        let source_uri = format!("{}/source", dir.as_str());
        let clone_uri = format!("{}/clone", dir.as_str());
        let (mut source, before) = recording_table(&source_uri, storage).await;

        let clone = clone_table(&mut source, &clone_uri, is_shallow).await;
        assert!(clone.manifest.uses_stable_row_ids());
        assert!(has_frag_reuse_flag(&clone));
        assert_same_history(&history(&clone).await, &history(&source).await);
        assert_translations_hold(&before, &clone_uri).await;
    }

    /// A clone of a recording table whose rows were all deleted keeps stable
    /// row ids, the history and the row id high-water mark, so its next
    /// compaction records, and cleanup keeps every version.
    #[rstest]
    #[tokio::test]
    async fn test_clone_of_an_emptied_recording_table_keeps_recording(
        #[values(false, true)] is_shallow: bool,
    ) {
        let dir = TempStrDir::default();
        let source_uri = format!("{}/source", dir.as_str());
        let clone_uri = format!("{}/clone", dir.as_str());
        let (mut source, _) = recording_table(&source_uri, Storage::Inline).await;
        source.delete("i >= 0").await.unwrap();
        assert!(source.manifest.fragments.is_empty());
        let recorded = history(&source).await;
        let next_row_id = source.manifest.next_row_id;

        let mut clone = clone_table(&mut source, &clone_uri, is_shallow).await;
        assert!(clone.manifest.uses_stable_row_ids());
        assert!(has_frag_reuse_flag(&clone));
        assert_eq!(clone.manifest.next_row_id, next_row_id);
        assert_same_history(&history(&clone).await, &recorded);

        let batch =
            arrow_array::record_batch!(("i", Int32, (1000..1020).collect::<Vec<i32>>())).unwrap();
        clone
            .append(
                arrow_array::RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        clone.delete("i = 1001").await.unwrap();
        let before = rows_by_addr(&clone).await;
        assert!(before.values().all(|(row_id, _)| *row_id >= next_row_id));
        compact_files(
            &mut clone,
            CompactionOptions {
                target_rows_per_fragment: 20,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(history(&clone).await.len(), recorded.len() + 1);
        assert_translations_hold(&before, &clone_uri).await;

        cleanup_frag_reuse_index(&mut clone).await.unwrap();
        let latest = Dataset::open(&clone_uri).await.unwrap();
        assert_eq!(history(&latest).await.len(), recorded.len() + 1);
        assert_translations_hold(&before, &clone_uri).await;
    }

    #[derive(Clone, Copy, Debug)]
    enum Maintenance {
        OptimizeIndices,
        CreateAndDropIndex,
        AppendAndDelete,
        CleanupOldVersions,
        CleanupWithoutIndices,
        CleanupWithACaughtUpIndex,
    }

    async fn create_zone_map(dataset: &mut Dataset) {
        dataset
            .create_index_builder(
                &["i"],
                IndexType::ZoneMap,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
            )
            .name("i_idx".into())
            .await
            .unwrap();
    }

    /// Ordinary maintenance never trims or alters the history, with or without
    /// indices, even once every index has caught up.
    #[rstest]
    #[tokio::test]
    async fn test_maintenance_keeps_the_recorded_history(
        #[values(
            Maintenance::OptimizeIndices,
            Maintenance::CreateAndDropIndex,
            Maintenance::AppendAndDelete,
            Maintenance::CleanupOldVersions,
            Maintenance::CleanupWithoutIndices,
            Maintenance::CleanupWithACaughtUpIndex
        )]
        maintenance: Maintenance,
        #[values(Storage::Inline, Storage::External)] storage: Storage,
    ) {
        let dir = TempStrDir::default();
        let (mut dataset, before) = recording_table(dir.as_str(), storage).await;
        let recorded = history(&dataset).await;
        let batch = arrow_array::record_batch!(("i", Int32, [1000, 1001, 1002])).unwrap();
        let appended = arrow_array::RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        match maintenance {
            Maintenance::OptimizeIndices => {
                create_zone_map(&mut dataset).await;
                dataset.append(appended, None).await.unwrap();
                dataset
                    .optimize_indices(&OptimizeOptions::default())
                    .await
                    .unwrap();
            }
            Maintenance::CreateAndDropIndex => {
                create_zone_map(&mut dataset).await;
                dataset.drop_index("i_idx").await.unwrap();
            }
            Maintenance::AppendAndDelete => {
                dataset.append(appended, None).await.unwrap();
                dataset.delete("i >= 1000").await.unwrap();
            }
            Maintenance::CleanupOldVersions => {
                dataset
                    .cleanup_old_versions(chrono::TimeDelta::zero(), Some(true), None)
                    .await
                    .unwrap();
            }
            Maintenance::CleanupWithoutIndices => {
                cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            }
            Maintenance::CleanupWithACaughtUpIndex => {
                create_zone_map(&mut dataset).await;
                cleanup_frag_reuse_index(&mut dataset).await.unwrap();
            }
        }
        let latest = Dataset::open(dir.as_str()).await.unwrap();
        assert_same_history(&history(&latest).await, &recorded);
        assert_translations_hold(&before, dir.as_str()).await;
    }

    #[derive(Clone, Copy, Debug)]
    enum Relocation {
        /// Nothing is stored at the new location.
        Missing,
        /// A different history of the same size is stored there.
        Different,
        /// A byte-identical copy of the history is stored there.
        Identical,
        /// A byte-identical copy is stored there, in another object store.
        IdenticalInAnotherStore,
    }

    /// A shallow clone of an externally stored history whose entry resolves
    /// through an explicit base id, and a location to point that base at,
    /// holding what `relocation` says.
    async fn relocatable_history(
        dir: &TempStrDir,
        relocation: Relocation,
    ) -> (Dataset, HashMap<u64, (u64, i32)>, u32, String) {
        let origin_uri = format!("{}/origin", dir.as_str());
        let source_uri = format!("{}/source", dir.as_str());
        let clone_uri = format!("{}/clone", dir.as_str());
        let elsewhere = match relocation {
            Relocation::IdenticalInAnotherStore => {
                format!("shared-memory://{}/elsewhere", uuid::Uuid::new_v4())
            }
            _ => format!("{}/elsewhere", dir.as_str()),
        };
        let (mut origin, before) = recording_table(&origin_uri, Storage::External).await;
        // A source that is itself a shallow clone holds base id 0, so the clone's
        // history base gets an explicit id rather than 0, which `UpdateBases`
        // reads as "assign one". Its own recorded compaction stores the history
        // under the source.
        let mut source = clone_table(&mut origin, &source_uri, true).await;
        let options = compaction_of(&source, &[6, 7]);
        compact_files(&mut source, options, None).await.unwrap();
        let clone = clone_table(&mut source, &clone_uri, true).await;
        let entry = frag_reuse_entry(&clone).await;
        let base_id = entry.base_id.unwrap();
        assert_ne!(base_id, 0);
        let file = external_details_file(&entry).unwrap();
        let source_file = std::path::Path::new(&source_uri)
            .join(INDICES_DIR)
            .join(entry.uuid.to_string())
            .join(&file.path);
        let bytes = match relocation {
            Relocation::Missing => None,
            Relocation::Identical | Relocation::IdenticalInAnotherStore => {
                Some(std::fs::read(&source_file).unwrap())
            }
            Relocation::Different => {
                let mut altered = history(&clone).await;
                altered[0].dataset_version += 1;
                Some(
                    InlineContent::from(&FragReuseIndexDetails { versions: altered })
                        .encode_to_vec(),
                )
            }
        };
        if let Some(bytes) = bytes {
            assert_eq!(bytes.len() as u64, file.size);
            let (store, root) = ObjectStore::from_uri_and_params(
                clone.session().store_registry(),
                &elsewhere,
                &Default::default(),
            )
            .await
            .unwrap();
            let moved_file = root
                .join(INDICES_DIR)
                .join(entry.uuid.to_string())
                .join(file.path.as_str());
            store.put(&moved_file, &bytes).await.unwrap();
        }
        (clone, before, base_id, elsewhere)
    }

    /// The history a proposed manifest publishes is read through that
    /// manifest's base paths and the stores they name, not the replaced one's:
    /// a history that moved is missing or different there, and an identical
    /// copy is the same history, in whichever store.
    #[rstest]
    #[tokio::test]
    async fn test_proposed_history_resolves_through_the_proposed_manifest(
        #[values(
            Relocation::Missing,
            Relocation::Different,
            Relocation::Identical,
            Relocation::IdenticalInAnotherStore
        )]
        relocation: Relocation,
    ) {
        let dir = TempStrDir::default();
        let (clone, _, base_id, elsewhere) = Box::pin(relocatable_history(&dir, relocation)).await;
        // Read through the replaced manifest first, so a stale resolution
        // would find the history cached at its old location.
        let recorded = history(&clone).await;
        let mut proposed = clone.manifest.as_ref().clone();
        proposed.base_paths.insert(
            base_id,
            BasePath::new(base_id, elsewhere, Some("elsewhere".to_string()), true),
        );
        let indices = load_all_indices(&clone).await.unwrap();
        let result = validate_frag_reuse_history(
            &Operation::UpdateBases { new_bases: vec![] },
            &clone,
            ReadSnapshot::Loaded {
                dataset: &clone,
                indices: &indices,
            },
            &proposed,
            &indices,
        )
        .await;

        match relocation {
            Relocation::Identical | Relocation::IdenticalInAnotherStore => result.unwrap(),
            Relocation::Missing => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
                assert!(error.to_string().contains("missing where"), "{error}");
            }
            Relocation::Different => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
                assert!(
                    error
                        .to_string()
                        .contains("Cannot drop or alter the fragment reuse version"),
                    "{error}"
                );
            }
        }
        assert_eq!(recorded.len(), history(&clone).await.len());
    }

    /// Pointing the base a cloned history resolves through at another
    /// location must not publish when a reader of the new manifest would find
    /// no history there, or a different one. The commit path reaches the
    /// history check: nothing earlier refuses re-pointing an existing base id.
    #[rstest]
    #[tokio::test]
    async fn test_moving_the_history_base_is_refused(
        #[values(Relocation::Missing, Relocation::Different)] relocation: Relocation,
    ) {
        let dir = TempStrDir::default();
        let (clone, before, base_id, elsewhere) =
            Box::pin(relocatable_history(&dir, relocation)).await;
        let clone_uri = clone.uri().to_string();
        let recorded = history(&clone).await;
        let operation = Operation::UpdateBases {
            new_bases: vec![BasePath::new(
                base_id,
                elsewhere.clone(),
                Some("elsewhere".to_string()),
                true,
            )],
        };

        let result = commit(&clone, clone.manifest.version, operation, false).await;
        if let Ok(published) = &result {
            let reopened = Dataset::open(&clone_uri).await.unwrap();
            let resolved = reopened
                .frag_reuse_index()
                .await
                .map(|index| index.map(|index| index.details.versions.clone()));
            panic!(
                "published version {} although a cold reopen resolves its history to {}",
                published.manifest.version,
                match resolved {
                    Err(error) => format!("an error: {error}"),
                    Ok(Some(versions)) => format!(
                        "{} versions, the same as recorded: {}",
                        versions.len(),
                        versions.len() == recorded.len()
                            && versions
                                .iter()
                                .zip(&recorded)
                                .all(|(left, right)| same_version(left, right).unwrap())
                    ),
                    Ok(None) => "no index".to_string(),
                }
            );
        }
        assert_refused(
            result,
            match relocation {
                Relocation::Missing => "history is missing where the manifest locates it",
                _ => "Cannot drop or alter the fragment reuse version",
            },
        );
        let latest = Dataset::open(&clone_uri).await.unwrap();
        assert_eq!(latest.manifest.version, clone.manifest.version);
        assert_same_history(&history(&latest).await, &recorded);
        assert_translations_hold(&before, &clone_uri).await;
    }
}
