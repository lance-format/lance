// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::index::frag_reuse::{
    TaggedTrimOutcome, derive_superseded_segments, derive_tagged_trim,
    is_superseded_prune_transaction, is_tagged_trim_operation,
};
use crate::index::DatasetIndexExt;
use crate::index::frag_reuse::{
    build_frag_reuse_index_metadata, build_frag_reuse_rewrite_entry, legacy_version_transitions,
    load_frag_reuse_index_details, load_frag_reuse_records, records_added_since,
};
use crate::index::mem_wal::{load_mem_wal_index_details, new_mem_wal_index_meta};
use crate::io::deletion::read_dataset_deletion_file;
use crate::{
    Dataset,
    dataset::transaction::{DataOverlayGroup, Operation, RewriteGroup, Transaction, UpdateMode},
};
use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, Result, utils::deletion::DeletionVector};
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::mem_wal::{CompactedSsTable, MEM_WAL_INDEX_NAME};
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::format::IndexMetadata;
use lance_table::format::overlay::OverlayCoverage;
use lance_table::format::overlay::staleness::{field_affects_index, overlay_affects_index};
use lance_table::format::pb::fragment_reuse_index_details::{InlineContent, Transition};
use lance_table::system_index::frag_reuse::FragReuseVersion;
use lance_table::system_index::frag_reuse::lineage::TaggedLineage;
use lance_table::system_index::frag_reuse::metadata::is_tagged;
use lance_table::system_index::is_system_index;
use lance_table::transaction::TaggedRewriteAssembly;
use lance_table::{format::Fragment, io::deletion::write_deletion_file};
use roaring::RoaringBitmap;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(Debug)]
pub struct TransactionRebase<'a> {
    transaction: Transaction,
    /// Relevant fragments as they were at the read version of the transaction.
    /// Has original fragment, plus a bool indicating whether a rewrite is needed.
    initial_fragments: HashMap<u64, (Fragment, bool)>,
    /// Fragments that have been deleted or modified
    modified_fragment_ids: HashSet<u64>,
    affected_rows: Option<&'a RowAddrTreeMap>,
    /// Boxed: the commit loop holds the rebase across awaits.
    frag_reuse_base: Option<Box<FragReuseBase>>,
    /// Compacted SSTables from conflicting UpdateMemWalState transactions.
    /// Used when rebasing CreateIndex of MemWalIndex.
    conflicting_mem_wal_compacted_sstables: Vec<CompactedSsTable>,
    /// The lineage the latest manifest's tagged fragment reuse entry records,
    /// with its transitions, loaded by [`Self::load_current_lineage`] for a
    /// CreateIndex; `None` otherwise.
    current_lineage: Option<TaggedLineage>,
    /// The latest manifest's live fragments, loaded with `current_lineage`.
    current_live: Option<RoaringBitmap>,
    /// The latest manifest's untagged fragment reuse index, as a CreateIndex
    /// on an untagged table sees it (empty when the table has none). Loaded by
    /// [`Self::load_current_lineage`]; `None` otherwise.
    untagged_reuse: Option<Box<UntaggedReuse>>,
    /// The latest manifest's schema, loaded with `current_lineage`: a
    /// rewritten field is expanded to its descendants through it (a packed
    /// struct is rewritten whole while an index on a child records the
    /// child's id).
    current_schema: Option<lance_core::datatypes::Schema>,
    /// The fragments and schema at the read version, kept for a CreateIndex
    /// to detect in-place column rewrites even after transaction cleanup,
    /// and fields a concurrent `Merge` removed or retyped.
    read_fragments: Option<Arc<Vec<Fragment>>>,
    read_schema: Option<lance_core::datatypes::Schema>,
    /// For a `Rewrite` carrying a fragment reuse entry: what it adds relative
    /// to the entry at its read version (`RewriteReuseState`).
    reuse: RewriteReuseState,
    /// Set by the staged-segment replay (`replay_staged_segments`): the
    /// CreateIndex being carried is not about to be committed, so a
    /// same-name CreateIndex in the window (routine `optimize_indices` on the
    /// logical index the segments extend) is no conflict; the overlap rules
    /// settle it at commit time. A same-name CreateIndex that changed what
    /// the index is (its fields or type) still conflicts.
    staged_replay: bool,
}

/// Whether a fragment-reuse index would corrupt `index` rather than repair it.
///
/// This happens when we have SRID, FRI, and row-id based indexes
fn corrupted_by_frag_reuse(index: &IndexMetadata) -> bool {
    !is_system_index(index) && !index.results_are_row_addrs()
}

/// A rewrite's fragment reuse intent as the rebase reads it: the entry at
/// the read version (the base the caller's complete entry is diffed against)
/// and, for a tagged entry, the transitions the caller's entry adds. Worked
/// out once per attempt from the same read version and the same transaction,
/// so every retry merges the same intent onto whatever the current entry has
/// become; the assembled result never feeds back into the transaction.
///
/// Contract: the caller's tagged entry was built from the snapshot at the
/// transaction's read version (its `dataset_version` says which), so the
/// base and the entry describe the same history plus this rewrite's own
/// records. An entry built from a later snapshot is refused.
#[derive(Debug, Default)]
struct RewriteReuseState {
    /// The fragment reuse entry at the read version, `None` when none existed.
    read_entry: Option<IndexMetadata>,
    /// The transitions the rewrite's entry adds relative to `read_entry`, for
    /// a tagged entry. `None` for a v0 entry, which `finish_rewrite` converts
    /// only when the table turned out tagged at commit.
    added_transitions: Option<Vec<Transition>>,
}

/// The fragment reuse entry of the handle a replacement is committed through, fixed across
/// attempts. Checked against the latest manifest since a Rewrite's transaction file omits it.
#[derive(Debug)]
struct FragReuseBase {
    version: u64,
    entry: Option<IndexMetadata>,
}

impl FragReuseBase {
    async fn load(dataset: &Dataset) -> Result<Self> {
        Ok(Self {
            version: dataset.manifest.version,
            entry: stored_frag_reuse_entry(dataset).await?,
        })
    }

    fn is_current(&self, current: Option<&IndexMetadata>) -> bool {
        self.entry.as_ref().map(|entry| entry.uuid) == current.map(|entry| entry.uuid)
    }

    fn changed_err(
        &self,
        operation: &Operation,
        current: Option<&IndexMetadata>,
        dataset: &Dataset,
        retry: &str,
    ) -> Error {
        let describe = |entry: Option<&IndexMetadata>| {
            entry.map_or_else(
                || "no entry".to_string(),
                |entry| format!("entry {}", entry.uuid),
            )
        };
        Error::retryable_commit_conflict_source(
            dataset.manifest.version,
            format!(
                "This {operation} transaction replaces the fragment reuse index, built from \
                 {} at version {}, but version {} has {}: a concurrent commit changed the \
                 fragment reuse history. {retry}",
                describe(self.entry.as_ref()),
                self.version,
                dataset.manifest.version,
                describe(current)
            )
            .into(),
        )
    }
}

async fn stored_frag_reuse_entry(dataset: &Dataset) -> Result<Option<IndexMetadata>> {
    Ok(crate::index::loading::load_all_indices(dataset)
        .await?
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .cloned())
}

fn is_ordered_subsequence(sub: &[FragReuseVersion], sup: &[FragReuseVersion]) -> bool {
    let mut sup = sup.iter();
    sub.iter()
        .all(|version| sup.any(|candidate| candidate == version))
}

/// Whether two segments describe the same logical index: the same keyed and
/// covering fields and the same index type.
fn same_logical_index(a: &IndexMetadata, b: &IndexMetadata) -> bool {
    let type_url = |index: &IndexMetadata| {
        index
            .index_details
            .as_ref()
            .map(|details| details.type_url.clone())
    };
    a.fields == b.fields && a.covering_fields == b.covering_fields && type_url(a) == type_url(b)
}

/// Whether `operation` may make a nullability-affecting schema change: a
/// projection or merge that does not assert `preserves_nullability`. A
/// tightening projection scanned for nulls at its read version, and a merge
/// may introduce a field that data staged against an earlier schema cannot
/// safely omit; either is falsified by a concurrent value-write.
fn may_alter_nullability(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Project {
            preserves_nullability: false,
            ..
        } | Operation::Merge {
            preserves_nullability: false,
            ..
        }
    )
}

/// Whether `operation` can commit rows that falsify such a change: by writing
/// a null into a scanned field, or by omitting a required column entirely (a
/// stale append's fragments read as null for columns they predate). `Delete`
/// only removes rows and `Rewrite` preserves the values it moves; `Project`
/// already conflicts with projections and merges elsewhere.
fn supplies_values(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Append { .. }
            | Operation::Update { .. }
            | Operation::DataReplacement { .. }
            | Operation::DataOverlay { .. }
    )
}

/// Whether an operation changes schema-level or per-field metadata. A merge or
/// projection carries the complete schema from its read version, so rebasing
/// either operation over the other can discard one side's metadata changes.
fn updates_schema_or_field_metadata(operation: &Operation) -> bool {
    let Operation::UpdateConfig {
        schema_metadata_updates,
        field_metadata_updates,
        ..
    } = operation
    else {
        return false;
    };

    schema_metadata_updates
        .as_ref()
        .is_some_and(|update| update.replace || !update.update_entries.is_empty())
        || field_metadata_updates
            .values()
            .any(|update| update.replace || !update.update_entries.is_empty())
}

mod check;
mod finish;
mod init;

impl<'a> TransactionRebase<'a> {
    #[track_caller]
    fn retryable_conflict_err(&self, other_transaction: &Transaction, other_version: u64) -> Error {
        Error::retryable_commit_conflict_source(
            other_version,
            format!(
                "This {} transaction was preempted by concurrent transaction {} at version {}. Please retry.",
                self.transaction.operation, other_transaction.operation, other_version
            )
            .into(),
        )
    }

    #[track_caller]
    fn incompatible_conflict_err(
        &self,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Error {
        Error::incompatible_transaction_source(
            format!(
                "This {} transaction is incompatible with concurrent transaction {} at version {}.",
                self.transaction.operation, other_transaction.operation, other_version
            )
            .into(),
        )
    }

    #[track_caller]
    fn data_replacement_target_removed_err(
        &self,
        fragment_id: u64,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Error {
        Error::incompatible_transaction_source(
            format!(
                "DataReplacement target fragment {} was removed by concurrent {} at version {}.",
                fragment_id, other_transaction.operation, other_version
            )
            .into(),
        )
    }

    #[track_caller]
    fn data_replacement_field_removed_err(
        &self,
        field_id: i32,
        fragment_id: u64,
        other_transaction: &Transaction,
        other_version: u64,
    ) -> Error {
        Error::incompatible_transaction_source(
            format!(
                "DataReplacement target field {} in fragment {} was dropped by concurrent {} at version {}.",
                field_id, fragment_id, other_transaction.operation, other_version
            )
            .into(),
        )
    }
}

/// The physical offsets a group's overlays cover, unioned across every overlay
/// and every field. This is the set of cells whose values the overlay supplies,
/// used to test whether a concurrent row-moving Update actually invalidates the
/// overlay.
fn overlay_group_coverage(group: &DataOverlayGroup) -> RoaringBitmap {
    let mut union = RoaringBitmap::new();
    for overlay in &group.overlays {
        match &overlay.coverage {
            OverlayCoverage::Shared(bitmap) => union |= bitmap.as_ref(),
            OverlayCoverage::PerField(bitmaps) => {
                for bitmap in bitmaps {
                    union |= bitmap.as_ref();
                }
            }
        }
    }
    union
}

/// Refuse a concurrent transaction that could hand one of an allocating
/// clustering submission's key field ids to another column.
///
/// A definition keys on field ids the writer resolved in `read_schema`, a
/// snapshot that still used reusable ids. Until the allocation itself
/// activates the non-reusable contract, a schema replacement in the window
/// (an overwrite, a drop, a restore, or a merge that does not keep every read
/// field under its id with the same name, parent and type) may reassign an id,
/// and a drop followed by a re-add with the same name and type hides that in
/// the final shape. So the transaction, not the landing schema, is judged.
/// Appends, deletes, updates, index builds and additive merges rebase as usual.
pub(crate) fn refuse_field_identity_change(
    read_schema: &lance_core::datatypes::Schema,
    other_transaction: &Transaction,
    other_version: u64,
) -> Result<()> {
    let may_change_identity = match &other_transaction.operation {
        Operation::Overwrite { .. } | Operation::Project { .. } | Operation::Restore { .. } => true,
        Operation::Merge { schema, .. } => !keeps_every_field_identity(read_schema, schema),
        _ => false,
    };
    if may_change_identity {
        return Err(Error::incompatible_transaction_source(
            format!(
                "This UpdateConfig allocates clustering definitions keyed on field ids read \
                 before concurrent transaction {} at version {other_version}, which may have \
                 given one of those ids to another column; reload the table and rebuild the list.",
                other_transaction.operation
            )
            .into(),
        ));
    }
    Ok(())
}

/// Every field of `read` is still the same field under the same id in
/// `landing`: same name, logical type and parent. Since every read field is
/// checked, a parent's own identity covers the path above it, and a child's
/// covers the shape below. Added fields are fine; a missing, renamed, moved
/// or retyped one is not, while metadata and nullability edits are. One map
/// of the landing fields keeps this linear on wide schemas.
fn keeps_every_field_identity(
    read: &lance_core::datatypes::Schema,
    landing: &lance_core::datatypes::Schema,
) -> bool {
    let landed: HashMap<i32, &lance_core::datatypes::Field> = landing
        .fields_pre_order()
        .map(|field| (field.id, field))
        .collect();
    read.fields_pre_order().all(|field| {
        landed.get(&field.id).is_some_and(|other| {
            other.name == field.name
                && other.logical_type == field.logical_type
                && other.parent_id == field.parent_id
        })
    })
}

fn wrong_operation_err(op: &Operation) -> Error {
    Error::internal(format!("function called against a wrong operation: {}", op))
}

/// The latest manifest's untagged fragment reuse index, as a CreateIndex sees it.
#[derive(Debug)]
struct UntaggedReuse {
    /// Each recorded group's source fragment ids, sorted.
    groups: HashSet<Arc<[u64]>>,
    /// Each recorded group's output fragments, mapped to its source ids.
    sources: HashMap<u32, Arc<[u64]>>,
    /// The latest manifest, whose schema overlays are matched against.
    manifest: Arc<lance_table::format::Manifest>,
}

impl UntaggedReuse {
    /// Whether every group of a rewrite is recorded, so it deferred its index
    /// remap and the record is still there to translate an index built before it.
    fn records(&self, groups: &[RewriteGroup]) -> bool {
        !groups.is_empty()
            && groups.iter().all(|group| {
                self.groups
                    .contains(sorted_ids(group.old_fragments.iter().map(|f| f.id)).as_slice())
            })
    }

    fn compacted_from(&self, fragment: u32) -> RoaringBitmap {
        let mut ids = RoaringBitmap::new();
        let mut pending = vec![fragment];
        while let Some(id) = pending.pop() {
            if ids.insert(id)
                && let Some(sources) = self.sources.get(&id)
            {
                pending.extend(sources.iter().map(|source| *source as u32));
            }
        }
        ids
    }

    /// Withdraws from each index the groups that folded in an overlay on one of
    /// its fields newer than it: nothing is left to mask the stale values.
    /// Returns false if an index with no fragment bitmap, which covers every
    /// fragment, needs a group withdrawn.
    fn withdraw_folded_overlays(
        &self,
        groups: &[RewriteGroup],
        indices: &mut [IndexMetadata],
    ) -> bool {
        for group in groups {
            for index in indices.iter_mut() {
                let dependency_fields = index.dependency_fields();
                let folded = group.old_fragments.iter().any(|fragment| {
                    fragment.overlays.iter().any(|overlay| {
                        overlay_affects_index(
                            overlay,
                            &dependency_fields,
                            index.dataset_version,
                            &self.manifest.schema,
                        )
                    })
                });
                if !folded {
                    continue;
                }
                let Some(bitmap) = &mut index.fragment_bitmap else {
                    return false;
                };
                for fragment in &group.old_fragments {
                    *bitmap -= self.compacted_from(fragment.id as u32);
                }
            }
        }
        true
    }
}

/// `fragment` and every fragment a recorded deferred compaction made it from:
/// the ids an index built before those compactions covers its rows under.
fn compacted_from(reuse: Option<&UntaggedReuse>, fragment: u32) -> RoaringBitmap {
    match reuse {
        Some(reuse) => reuse.compacted_from(fragment),
        None => RoaringBitmap::from_iter([fragment]),
    }
}

fn sorted_ids(ids: impl Iterator<Item = u64>) -> Vec<u64> {
    let mut ids: Vec<u64> = ids.collect();
    ids.sort_unstable();
    ids
}

#[cfg(test)]
mod tests;
