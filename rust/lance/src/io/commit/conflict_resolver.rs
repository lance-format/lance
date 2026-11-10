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
    /// The fragments live in the snapshot the transaction's `read_version`
    /// names, set by [`Self::with_read_version_fragments`]. `read_fragments`
    /// comes from the handle the commit was submitted through, which may
    /// already be past the rewrite being rebased over; the undeferrable
    /// withdrawal must be measured against the read version instead.
    read_version_fragments: Option<Arc<Vec<Fragment>>>,
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
    /// For an `Update` that merges one MemWAL shard ([`mem_wal_merge_shard`]):
    /// the `dir_id` of that MemWAL when, at the read version, it routes every
    /// row by primary-key columns alone ([`mem_wal_routes_by_key`]). Two such
    /// updates of different shards of this MemWAL then touch disjoint keys,
    /// whatever their inserted-rows filters say.
    mem_wal_key_routed_dir: Option<String>,
}

/// Transaction property naming the MemWAL shard every row of a merge belongs
/// to, set by [`MergeInsertBuilder::mem_wal_shard`].
///
/// [`MergeInsertBuilder::mem_wal_shard`]: crate::dataset::MergeInsertBuilder::mem_wal_shard
pub(crate) const MEM_WAL_SHARD_PROPERTY: &str = "lance.mem_wal.shard_id";

/// The MemWAL (`dir_id`) and shard an `Update` merges, when it merges exactly
/// one shard.
///
/// The shard comes from the [`MEM_WAL_SHARD_PROPERTY`] the merge declared, or
/// else from its `compacted_sstables` when they name a single shard. Every pass
/// of a multi-pass merge carries the declaration, while only the last one
/// carries `compacted_sstables`. An update naming several shards, two
/// different shards, or an unreadable shard id has none.
fn mem_wal_merge_shard(transaction: &Transaction) -> Option<(&str, uuid::Uuid)> {
    let Operation::Update {
        compacted_sstables,
        mem_wal_dir_id: Some(dir_id),
        ..
    } = &transaction.operation
    else {
        return None;
    };
    let declared = match transaction
        .transaction_properties
        .as_ref()
        .and_then(|properties| properties.get(MEM_WAL_SHARD_PROPERTY))
    {
        Some(shard) => Some(uuid::Uuid::parse_str(shard).ok()?),
        None => None,
    };
    let marked = match compacted_sstables.as_slice() {
        [] => None,
        [sstable] => Some(sstable.shard_id),
        _ => return None,
    };
    let shard = match (declared, marked) {
        (Some(declared), Some(marked)) if declared != marked => return None,
        (declared, marked) => declared.or(marked)?,
    };
    Some((dir_id.as_str(), shard))
}

/// Whether the MemWAL `dir_id` of `dataset` sends every row to the shard its
/// primary key alone decides, so one key lives in exactly one shard.
///
/// That holds when the MemWAL index is that incarnation (`dir_id`), has a
/// single sharding spec, and every field of the spec is computed from
/// primary-key columns only; a deterministic function of the key then routes
/// equal keys to the same shard. `filter_field_ids`, the columns an
/// inserted-rows filter was built on, must be that primary key. Without a
/// spec (shards managed by hand) nothing ties a key to a shard; an unsharded
/// spec reads no column at all; two specs could route one key two ways.
async fn mem_wal_routes_by_key(
    dataset: &Dataset,
    dir_id: &str,
    filter_field_ids: Option<&[i32]>,
) -> Result<bool> {
    let key_field_ids: HashSet<i32> = dataset
        .schema()
        .unenforced_primary_key()
        .iter()
        .map(|field| field.id)
        .collect();
    if key_field_ids.is_empty() {
        return Ok(false);
    }
    if let Some(field_ids) = filter_field_ids
        && (field_ids.len() != key_field_ids.len()
            || !field_ids.iter().all(|id| key_field_ids.contains(id)))
    {
        return Ok(false);
    }
    let indices = dataset.load_indices().await?;
    let Some(index) = indices.iter().find(|idx| idx.name == MEM_WAL_INDEX_NAME) else {
        return Ok(false);
    };
    let details = load_mem_wal_index_details(index.clone())?;
    let [spec] = details.sharding_specs.as_slice() else {
        return Ok(false);
    };
    Ok(details.dir_id == dir_id
        && !spec.fields.is_empty()
        && spec.fields.iter().all(|field| {
            !field.source_ids.is_empty()
                && field.source_ids.iter().all(|id| key_field_ids.contains(id))
        }))
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

/// Concurrent MemWAL compaction merges of different shards, in the shape
/// LanceDB Enterprise commits them: a composite primary key
/// `(row_index, shard_key)`, a bucket spec on `shard_key`, and each shard's
/// merge split into a delete pass, upsert passes that record no progress, and
/// a last upsert pass that marks the shard's SSTables compacted.
#[cfg(test)]
mod mem_wal_shard_merge_tests {
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::Arc;

    use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_plan::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use lance_index::mem_wal::CompactedSsTable;
    use uuid::Uuid;

    use crate::Dataset;
    use crate::dataset::mem_wal::{DatasetMemWalExt, evaluate_sharding_spec};
    use crate::dataset::transaction::{Operation, Transaction};
    use crate::dataset::write::merge_insert::{
        MergeInsertBuilder, UncommittedMergeInsert, WhenMatched, WhenNotMatched,
    };
    use crate::dataset::write::{CommitBuilder, InsertBuilder};
    use lance_select::RowAddrTreeMap;

    /// Above the ~2-3k keys where two disjoint 8,192-key filters stop telling
    /// themselves apart, so a conflict between upserts comes from the filter.
    const KEYS_PER_PASS: i64 = 6_000;
    /// Rows already in the base table for each shard, which delete passes remove.
    const SEEDED_PER_SHARD: i64 = 100;

    #[derive(Clone, Copy, Debug)]
    enum Sharding {
        /// What Enterprise installs: `bucket(shard_key)`, a primary-key column.
        KeyBucket,
        /// A bucket spec on `grp`, which is not part of the primary key.
        NonKeyBucket,
        Unsharded,
        /// No spec: shards are managed by hand.
        Manual,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Pass {
        /// Key-only `when_matched_delete` pass; inserts nothing.
        Delete,
        /// Upsert pass that records no progress (not the shard's last commit).
        Upsert,
        /// The shard's last upsert pass, which marks its SSTables compacted.
        UpsertMarking,
    }

    fn primary_key(position: &str) -> HashMap<String, String> {
        HashMap::from([
            (
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            ),
            (
                "lance-schema:unenforced-primary-key:position".to_string(),
                position.to_string(),
            ),
        ])
    }

    /// The declared primary key in key order, which a merge must be keyed on
    /// to carry an inserted-rows filter. Enterprise's tables write position 0
    /// on `row_index` (read as an unpositioned, legacy key column) and 1 on
    /// `shard_key`, so the key reads `(shard_key, row_index)`.
    fn merge_on() -> Vec<String> {
        vec!["shard_key".to_string(), "row_index".to_string()]
    }

    fn key_fields() -> Vec<Field> {
        vec![
            Field::new("row_index", DataType::Int64, false).with_metadata(primary_key("0")),
            Field::new("shard_key", DataType::Int64, false).with_metadata(primary_key("1")),
        ]
    }

    fn schema() -> Arc<Schema> {
        let mut fields = key_fields();
        fields.push(Field::new("grp", DataType::Int64, true));
        fields.push(Field::new("text", DataType::Utf8, true));
        Arc::new(Schema::new(fields))
    }

    fn rows(rows: Range<i64>, shard_key: i64) -> RecordBatch {
        let n = (rows.end - rows.start) as usize;
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(rows.clone())),
                Arc::new(Int64Array::from(vec![shard_key; n])),
                Arc::new(Int64Array::from(vec![shard_key; n])),
                Arc::new(StringArray::from_iter_values(
                    rows.map(|i| format!("document {i}")),
                )),
            ],
        )
        .unwrap()
    }

    fn keys(rows: Range<i64>, shard_key: i64) -> RecordBatch {
        let n = (rows.end - rows.start) as usize;
        RecordBatch::try_new(
            Arc::new(Schema::new(key_fields())),
            vec![
                Arc::new(Int64Array::from_iter_values(rows)),
                Arc::new(Int64Array::from(vec![shard_key; n])),
            ],
        )
        .unwrap()
    }

    /// The two shards of a test table: shard id, the `shard_key` its rows
    /// carry, and the rows seeded for it.
    #[derive(Clone, Copy)]
    struct Shard {
        id: Uuid,
        shard_key: i64,
        seeded: (i64, i64),
    }

    struct Table {
        dataset: Arc<Dataset>,
        dir_id: String,
        a: Shard,
        b: Shard,
    }

    /// Two `shard_key` values that the spec routes to different buckets, as
    /// Enterprise's writers pick them.
    async fn shard_keys(dataset: &Dataset) -> (i64, i64) {
        let details = dataset.mem_wal_index_details().await.unwrap().unwrap();
        let spec = &details.sharding_specs[0];
        let candidates = rows(0..64, 0);
        let candidates = RecordBatch::try_new(
            candidates.schema(),
            vec![
                candidates.column(0).clone(),
                Arc::new(Int64Array::from_iter_values(0..64)),
                candidates.column(2).clone(),
                candidates.column(3).clone(),
            ],
        )
        .unwrap();
        let buckets = evaluate_sharding_spec(&candidates, spec, dataset.schema()).unwrap();
        let buckets = buckets
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let other = (1..buckets.len())
            .find(|&i| buckets.value(i) != buckets.value(0))
            .unwrap();
        (0, other as i64)
    }

    async fn table(sharding: Sharding) -> Table {
        let mut dataset = InsertBuilder::new(&format!("memory://shard_merge_{}", Uuid::new_v4()))
            .execute(vec![
                rows(0..SEEDED_PER_SHARD, 0),
                rows(SEEDED_PER_SHARD..2 * SEEDED_PER_SHARD, 1),
            ])
            .await
            .unwrap();
        let builder = dataset.initialize_mem_wal();
        match sharding {
            Sharding::KeyBucket => builder.bucket_sharding("shard_key", 4),
            Sharding::NonKeyBucket => builder.bucket_sharding("grp", 4),
            Sharding::Unsharded => builder.unsharded(),
            Sharding::Manual => builder,
        }
        .execute()
        .await
        .unwrap();
        let (key_a, key_b) = if matches!(sharding, Sharding::KeyBucket) {
            shard_keys(&dataset).await
        } else {
            (0, 1)
        };
        // Re-seed so each shard's rows carry its routed `shard_key`.
        let pk = merge_on();
        let mut reseed = MergeInsertBuilder::try_new(Arc::new(dataset), pk.clone()).unwrap();
        reseed
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll);
        let (dataset, _) = reseed
            .try_build()
            .unwrap()
            .execute_reader(arrow_array::RecordBatchIterator::new(
                vec![
                    Ok(rows(2 * SEEDED_PER_SHARD..3 * SEEDED_PER_SHARD, key_a)),
                    Ok(rows(3 * SEEDED_PER_SHARD..4 * SEEDED_PER_SHARD, key_b)),
                ],
                schema(),
            ))
            .await
            .unwrap();
        let dir_id = dataset
            .mem_wal_index_details()
            .await
            .unwrap()
            .unwrap()
            .dir_id;
        Table {
            dataset,
            dir_id,
            a: Shard {
                id: Uuid::new_v4(),
                shard_key: key_a,
                seeded: (2 * SEEDED_PER_SHARD, 3 * SEEDED_PER_SHARD),
            },
            b: Shard {
                id: Uuid::new_v4(),
                shard_key: key_b,
                seeded: (3 * SEEDED_PER_SHARD, 4 * SEEDED_PER_SHARD),
            },
        }
    }

    /// How a pass names its shard.
    #[derive(Clone, Copy)]
    struct Declares {
        /// Calls `mem_wal_shard`, as every Enterprise pass does.
        shard: Option<Uuid>,
        /// The shard a marking pass records progress for.
        marks: Uuid,
    }

    impl Declares {
        fn shard(shard: &Shard) -> Self {
            Self {
                shard: Some(shard.id),
                marks: shard.id,
            }
        }
    }

    /// One uncommitted pass of `shard`'s merge, read from `dataset`.
    async fn pass(
        dataset: &Arc<Dataset>,
        dir_id: &str,
        shard: &Shard,
        declares: Declares,
        kind: Pass,
        rows_from: i64,
    ) -> (Transaction, Option<RowAddrTreeMap>) {
        let data = match kind {
            Pass::Delete => keys(shard.seeded.0..shard.seeded.1, shard.shard_key),
            Pass::Upsert | Pass::UpsertMarking => {
                rows(rows_from..rows_from + KEYS_PER_PASS, shard.shard_key)
            }
        };
        let stream =
            RecordBatchStreamAdapter::new(data.schema(), futures::stream::iter(vec![Ok(data)]));
        let mut builder = MergeInsertBuilder::try_new(dataset.clone(), merge_on()).unwrap();
        if kind == Pass::Delete {
            builder
                .when_matched(WhenMatched::Delete)
                .when_not_matched(WhenNotMatched::DoNothing);
        } else {
            builder
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll);
        }
        let marks = if kind == Pass::UpsertMarking {
            vec![CompactedSsTable::new(declares.marks, 1)]
        } else {
            Vec::new()
        };
        builder.mark_sstables_as_compacted(dir_id, marks);
        if let Some(shard) = declares.shard {
            builder.mem_wal_shard(shard);
        }
        let UncommittedMergeInsert {
            transaction,
            affected_rows,
            ..
        } = builder
            .try_build()
            .unwrap()
            .execute_uncommitted(Box::pin(stream) as SendableRecordBatchStream)
            .await
            .unwrap();
        if kind != Pass::Delete {
            assert!(
                matches!(
                    &transaction.operation,
                    Operation::Update {
                        inserted_rows_filter: Some(_),
                        ..
                    }
                ),
                "an upsert on the primary key carries an inserted-rows filter"
            );
        }
        (transaction, affected_rows)
    }

    async fn commit(
        dataset: &Arc<Dataset>,
        (transaction, affected_rows): (Transaction, Option<RowAddrTreeMap>),
    ) -> crate::Result<Dataset> {
        let mut builder = CommitBuilder::new(dataset.clone());
        if let Some(affected_rows) = affected_rows {
            builder = builder.with_affected_rows(affected_rows);
        }
        builder.execute(transaction).await
    }

    /// Passes of the two shards read from the same version; `b` commits over
    /// `a`. Returns `b`'s commit.
    async fn race(
        table: &Table,
        (a, a_declares, a_kind): (Shard, Declares, Pass),
        (b, b_declares, b_kind): (Shard, Declares, Pass),
    ) -> crate::Result<Dataset> {
        let first = pass(
            &table.dataset,
            &table.dir_id,
            &a,
            a_declares,
            a_kind,
            1_000_000,
        )
        .await;
        let second = pass(
            &table.dataset,
            &table.dir_id,
            &b,
            b_declares,
            b_kind,
            2_000_000,
        )
        .await;
        commit(&table.dataset, first).await.unwrap();
        commit(&table.dataset, second).await
    }

    #[tokio::test]
    async fn test_passes_of_different_shards_commit_over_each_other() {
        use Pass::*;
        for (a_kind, b_kind) in [
            (Upsert, Upsert),
            (Upsert, UpsertMarking),
            (UpsertMarking, Upsert),
            (UpsertMarking, UpsertMarking),
            (Delete, Upsert),
            (Upsert, Delete),
            (Delete, UpsertMarking),
            (Delete, Delete),
        ] {
            let table = table(Sharding::KeyBucket).await;
            let (a, b) = (table.a, table.b);
            let committed = race(
                &table,
                (a, Declares::shard(&a), a_kind),
                (b, Declares::shard(&b), b_kind),
            )
            .await;
            assert!(
                committed.is_ok(),
                "{a_kind:?} of one shard then {b_kind:?} of another: {committed:?}"
            );
        }
    }

    /// Both shards run their whole merge -- delete, an upsert that records no
    /// progress, the marking upsert -- each pass racing the other shard's.
    #[tokio::test]
    async fn test_multi_pass_merges_of_different_shards_interleave() {
        let table = table(Sharding::KeyBucket).await;
        let (a, b) = (table.a, table.b);
        let mut dataset = table.dataset.clone();
        let before = dataset.count_rows(None).await.unwrap() as i64;
        for (step, kind) in [Pass::Delete, Pass::Upsert, Pass::UpsertMarking]
            .into_iter()
            .enumerate()
        {
            let rows_from = 1_000_000 * (step as i64 + 1);
            let first = pass(
                &dataset,
                &table.dir_id,
                &a,
                Declares::shard(&a),
                kind,
                rows_from,
            )
            .await;
            let second = pass(
                &dataset,
                &table.dir_id,
                &b,
                Declares::shard(&b),
                kind,
                rows_from + KEYS_PER_PASS,
            )
            .await;
            commit(&dataset, first).await.unwrap();
            dataset = Arc::new(
                commit(&dataset, second)
                    .await
                    .unwrap_or_else(|e| panic!("{kind:?} pass of the second shard: {e:?}")),
            );
        }
        assert_eq!(
            dataset.count_rows(None).await.unwrap() as i64,
            before - 2 * SEEDED_PER_SHARD + 4 * KEYS_PER_PASS
        );
        let mut compacted = dataset
            .mem_wal_index_details()
            .await
            .unwrap()
            .unwrap()
            .compacted_sstables
            .into_iter()
            .map(|sstable| sstable.shard_id)
            .collect::<Vec<_>>();
        compacted.sort();
        let mut expected = vec![a.id, b.id];
        expected.sort();
        assert_eq!(compacted, expected);
    }

    fn assert_conflict(result: crate::Result<Dataset>, case: &str) {
        let err = result.expect_err(case);
        assert!(
            matches!(err, crate::Error::RetryableCommitConflict { .. }),
            "{case}: {err:?}"
        );
    }

    #[tokio::test]
    async fn test_passes_of_the_same_shard_still_conflict() {
        for kind in [Pass::Upsert, Pass::UpsertMarking] {
            let table = table(Sharding::KeyBucket).await;
            let a = table.a;
            assert_conflict(
                race(
                    &table,
                    (a, Declares::shard(&a), Pass::Upsert),
                    (a, Declares::shard(&a), kind),
                )
                .await,
                "same shard: the filter test applies",
            );
        }
    }

    #[tokio::test]
    async fn test_shards_that_do_not_route_by_key_still_conflict() {
        for sharding in [
            Sharding::NonKeyBucket,
            Sharding::Unsharded,
            Sharding::Manual,
        ] {
            let table = table(sharding).await;
            let (a, b) = (table.a, table.b);
            assert_conflict(
                race(
                    &table,
                    (a, Declares::shard(&a), Pass::Upsert),
                    (b, Declares::shard(&b), Pass::Upsert),
                )
                .await,
                &format!("{sharding:?}: nothing ties a key to one shard"),
            );
        }
    }

    #[tokio::test]
    async fn test_passes_without_a_single_shard_still_conflict() {
        // An undeclared pass that records no progress names no shard.
        fn undeclared(shard: &Shard) -> Declares {
            Declares {
                shard: None,
                marks: shard.id,
            }
        }
        // A pass that declares one shard and marks another names none.
        fn contradictory(shard: &Shard) -> Declares {
            Declares {
                shard: Some(shard.id),
                marks: Uuid::new_v4(),
            }
        }
        type Case = (&'static str, fn(&Table) -> [(Shard, Declares, Pass); 2]);
        let cases: [Case; 4] = [
            ("the committed pass names no shard", |t| {
                [
                    (t.a, undeclared(&t.a), Pass::Upsert),
                    (t.b, Declares::shard(&t.b), Pass::Upsert),
                ]
            }),
            ("the committing pass names no shard", |t| {
                [
                    (t.b, Declares::shard(&t.b), Pass::Upsert),
                    (t.a, undeclared(&t.a), Pass::Upsert),
                ]
            }),
            ("a pass declaring one shard and marking another", |t| {
                [
                    (t.b, Declares::shard(&t.b), Pass::Upsert),
                    (t.a, contradictory(&t.a), Pass::UpsertMarking),
                ]
            }),
            // An undeclared delete pass carries no filter, so nothing says
            // what keys it touched.
            ("an undeclared delete pass", |t| {
                [
                    (t.a, undeclared(&t.a), Pass::Delete),
                    (t.b, Declares::shard(&t.b), Pass::Upsert),
                ]
            }),
        ];
        for (case, passes) in cases {
            let table = table(Sharding::KeyBucket).await;
            let [first, second] = passes(&table);
            assert_conflict(race(&table, first, second).await, case);
        }
    }

    #[tokio::test]
    async fn test_an_undeclared_marking_pass_still_names_its_shard() {
        // Callers that only mark progress (a single-pass merge) keep the skip.
        let table = table(Sharding::KeyBucket).await;
        let (a, b) = (table.a, table.b);
        let marking_only = |shard: &Shard| Declares {
            shard: None,
            marks: shard.id,
        };
        race(
            &table,
            (a, marking_only(&a), Pass::UpsertMarking),
            (b, marking_only(&b), Pass::UpsertMarking),
        )
        .await
        .expect("marking passes of different shards");
    }

    #[tokio::test]
    async fn test_mem_wal_shard_requires_a_mem_wal() {
        let table = table(Sharding::KeyBucket).await;
        let mut builder = MergeInsertBuilder::try_new(table.dataset.clone(), merge_on()).unwrap();
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .mem_wal_shard(table.a.id);
        assert!(matches!(
            builder.try_build(),
            Err(crate::Error::InvalidInput { .. })
        ));
    }
}
