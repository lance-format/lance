// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index-domain action payloads.
//!
//! Extracted from `action.rs` to keep that module focused on the `Action`
//! trait and dispatch. Items here are re-exported from `action.rs`, so the
//! public surface of `crate::transaction::action::*` is unchanged.

use std::collections::HashSet;

use crate::format::{IndexFile, IndexMetadata, Manifest};
use crate::system_index::frag_reuse::FragReuseVersion;
use crate::system_index::mem_wal::MergedGeneration;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::{Action, invalidate_index_coverage};
use crate::impl_dyn_action;
use crate::transaction::mem_wal::update_mem_wal_index_merged_generations;

/// Which fragments an index covers, the criterion half of an [`AddIndex`]
/// payload (spike #6448 §3.1).
///
/// The criterion is resolved into a concrete fragment bitmap by
/// [`FragmentCoverage::resolve`] at apply time, against the post-rebase
/// manifest — so `All` picks up any fragment a concurrent `AddFragments`
/// committed in the meantime.
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentCoverage {
    /// Every fragment present in the manifest at apply time.
    All,
    /// Exactly the fragment ids in this bitmap.
    Bitmap(RoaringBitmap),
    /// The index is not bound to any fragments — system indices like the
    /// inline MemWAL state ([`MEM_WAL_INDEX_NAME`]) carry
    /// `IndexMetadata::fragment_bitmap = None` because they cover no
    /// data-fragment rows. `apply` preserves the `None` rather than
    /// overwriting it with a resolved bitmap.
    Unbound,
}

impl FragmentCoverage {
    /// Resolve the criterion into a concrete fragment bitmap. `Unbound` has no
    /// resolved bitmap; callers must special-case it before calling this.
    pub(super) fn resolve(&self, manifest: &Manifest) -> Option<RoaringBitmap> {
        match self {
            Self::All => Some(manifest.fragments.iter().map(|f| f.id as u32).collect()),
            Self::Bitmap(bitmap) => Some(bitmap.clone()),
            Self::Unbound => None,
        }
    }
}

impl DeepSizeOf for FragmentCoverage {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        match self {
            // `RoaringBitmap` has no `DeepSizeOf` impl; approximate it by its
            // cardinality, mirroring `UpdateDeletionVector`.
            Self::Bitmap(bitmap) => {
                (bitmap.len() as usize).saturating_mul(std::mem::size_of::<u32>())
            }
            Self::All | Self::Unbound => 0,
        }
    }
}

/// Payload for [`AddIndex`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddIndex {
    /// The index entry to add. Its `fragment_bitmap` is ignored — apply
    /// overwrites it with the bitmap resolved from `coverage` — so callers
    /// need not keep the two in sync.
    pub index: IndexMetadata,
    /// Which fragments the index covers, resolved against the manifest at
    /// apply time.
    pub coverage: FragmentCoverage,
    /// Eagerly-loaded `FragReuseVersion`s when `index.name` is the reserved
    /// `FRAG_REUSE_INDEX_NAME`, `None` otherwise. Embedding the parsed
    /// version list lets the rebase rule smart-merge against a concurrent
    /// `RegisterFragReuse`/`AddIndex` of the same system index without doing
    /// async object-store I/O inside [`Action::rebase`]. A post-driver hook
    /// in `commit_transaction` materialises the merged list into a fresh
    /// details file before the manifest commit. Always `None` for every
    /// non-`FRAG_REUSE_INDEX_NAME` index.
    pub frag_reuse_versions: Option<Vec<FragReuseVersion>>,
}

impl Action for AddIndex {
    impl_dyn_action!(AddIndex);

    fn apply(&self, manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        let mut index = self.index.clone();
        // The criterion is resolved here, against the post-rebase manifest, so
        // a concurrently-appended fragment is covered correctly when
        // `coverage` is `All` (spike #6448 §3.1). `Unbound` preserves a `None`
        // bitmap — the shape system indices like the inline MemWAL state carry.
        index.fragment_bitmap = self.coverage.resolve(manifest);
        // Replacing an entry with the same UUID mirrors the
        // `Operation::CreateIndex` dedup in `build_manifest`.
        indices.retain(|existing| existing.uuid != index.uuid);
        indices.push(index);
        Ok(())
    }
}

/// Payload for [`RemoveIndex`].
#[derive(Debug, Clone, PartialEq)]
pub struct RemoveIndex {
    /// UUID of the index entry to remove.
    pub uuid: Uuid,
}

impl DeepSizeOf for RemoveIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.uuid.as_bytes().deep_size_of_children(context)
    }
}

impl Action for RemoveIndex {
    impl_dyn_action!(RemoveIndex);

    fn apply(&self, _manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        indices.retain(|index| index.uuid != self.uuid);
        Ok(())
    }
}

/// Payload for [`RewriteIndex`].
#[derive(Debug, Clone, PartialEq)]
pub struct RewriteIndex {
    /// UUID of the existing index entry being replaced.
    pub old_uuid: Uuid,
    /// UUID for the replacement entry.
    pub new_uuid: Uuid,
    /// Files (with sizes) backing the replacement entry. `None` from writers
    /// that do not persist file sizes.
    pub new_index_files: Option<Vec<IndexFile>>,
    /// Fragment bitmap for the replacement entry, already re-pointed at the
    /// rewritten fragments by the caller.
    pub new_fragment_bitmap: RoaringBitmap,
}

impl DeepSizeOf for RewriteIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.old_uuid.as_bytes().deep_size_of_children(context)
            + self.new_uuid.as_bytes().deep_size_of_children(context)
            + self.new_index_files.deep_size_of_children(context)
            // `RoaringBitmap` has no `DeepSizeOf` impl; approximate by its
            // cardinality, mirroring `UpdateDeletionVector`.
            + (self.new_fragment_bitmap.len() as usize)
                .saturating_mul(std::mem::size_of::<u32>())
    }
}

impl Action for RewriteIndex {
    impl_dyn_action!(RewriteIndex);

    fn apply(&self, _manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        // An index that no longer exists (e.g. removed by a concurrent
        // transaction) is silently skipped, exactly as
        // `Transaction::handle_rewrite_indices` does.
        if let Some(index) = indices.iter_mut().find(|index| index.uuid == self.old_uuid) {
            index.uuid = self.new_uuid;
            index.files = self.new_index_files.clone();
            index.fragment_bitmap = Some(self.new_fragment_bitmap.clone());
        }
        Ok(())
    }
}

/// Payload for [`InvalidateIndexCoverage`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct InvalidateIndexCoverage {
    /// Fragment ids to drop from the bitmap of every affected index.
    pub fragment_ids: Vec<u64>,
    /// An index is affected iff it covers at least one of these field ids.
    pub field_ids: Vec<i32>,
}

impl Action for InvalidateIndexCoverage {
    impl_dyn_action!(InvalidateIndexCoverage);

    fn apply(&self, _manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        invalidate_index_coverage(indices, &self.fragment_ids, &self.field_ids);
        Ok(())
    }
}

/// Payload for [`RebindIndexCoverage`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RebindIndexCoverage {
    /// How the fragment ids in each affected index's coverage bitmap are
    /// re-pointed.
    pub remap: IndexCoverageRemap,
    /// An index is rebound iff it covers *none* of these field ids — its
    /// data is still valid and only the fragment ids moved (spike #6843).
    pub modified_field_ids: Vec<i32>,
}

/// A single rewrite group's fragment-id re-pointing, used by
/// [`IndexCoverageRemap::RewriteGroups`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RewriteGroupRemap {
    /// The fragments the group consumed.
    pub old_fragment_ids: Vec<u64>,
    /// The fragments the group produced. May hold more, fewer, or the same
    /// number of ids as `old_fragment_ids` — a group can split one fragment
    /// into many or compact a fully-deleted fragment into none.
    pub new_fragment_ids: Vec<u64>,
}

/// The fragment-id re-pointing an [`RebindIndexCoverage`] performs on
/// every index whose coverage is still valid (covers none of
/// [`RebindIndexCoverage::modified_field_ids`]).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum IndexCoverageRemap {
    /// Per-rewrite-group fragment-id re-pointing for a stable-row-id `Rewrite`
    /// (compaction). An index that covered *all* of a group's
    /// `old_fragment_ids` has them all removed and all of `new_fragment_ids`
    /// inserted; an index covering only some is a split of indexed and
    /// non-indexed data and `apply` rejects it — the all-or-nothing
    /// recalculation [`Transaction::recalculate_fragment_bitmap`] performs.
    /// Unlike a 1:1 pair list this expresses arbitrary group cardinality.
    RewriteGroups(Vec<RewriteGroupRemap>),

    /// Insert-only gated registration for a stable-row-id `RewriteRows`
    /// `Update` — reproduces the legacy
    /// `register_pure_rewrite_rows_update_frags_in_indices` (#6914).
    ///
    /// For each affected index, `new_fragment_ids` are inserted into its
    /// coverage bitmap *only if* the bitmap already covers every id in
    /// `original_fragment_ids` — i.e. every original fragment whose rows were
    /// rewritten into the new fragments was already indexed. When the gate
    /// fails the index is left untouched: the new fragments' rows are not all
    /// indexed, so a query must keep flat-scanning them. Unlike `Pairs` this
    /// removes nothing — the still-present `updated_fragments` keep their
    /// coverage.
    InsertPureRewrite {
        /// The pure-rewrite fragments the update produced.
        new_fragment_ids: Vec<u64>,
        /// Every original fragment (the emptied `removed_fragment_ids` plus the
        /// still-present `updated_fragments`) that contributed the rewritten
        /// rows — the all-or-nothing coverage gate.
        original_fragment_ids: Vec<u64>,
    },
}

impl Action for RebindIndexCoverage {
    impl_dyn_action!(RebindIndexCoverage);

    fn apply(&self, _manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        let modified: HashSet<i32> = self.modified_field_ids.iter().copied().collect();
        for index in indices.iter_mut() {
            let covers_modified_field = index.fields.iter().any(|id| modified.contains(id));
            if covers_modified_field {
                continue;
            }
            let Some(bitmap) = &mut index.fragment_bitmap else {
                continue;
            };
            match &self.remap {
                IndexCoverageRemap::RewriteGroups(groups) => {
                    for group in groups {
                        // Re-point a group only when the index covered *all*
                        // of it; partial coverage is a split of indexed and
                        // non-indexed data, which `recalculate_fragment_bitmap`
                        // rejects.
                        let any = group
                            .old_fragment_ids
                            .iter()
                            .any(|id| bitmap.contains(*id as u32));
                        if !any {
                            continue;
                        }
                        let all = group
                            .old_fragment_ids
                            .iter()
                            .all(|id| bitmap.contains(*id as u32));
                        if !all {
                            return Err(Error::invalid_input(
                                "The compaction plan included a rewrite group that was a split of indexed and non-indexed data",
                            ));
                        }
                        for old in &group.old_fragment_ids {
                            bitmap.remove(*old as u32);
                        }
                        for new in &group.new_fragment_ids {
                            bitmap.insert(*new as u32);
                        }
                    }
                }
                IndexCoverageRemap::InsertPureRewrite {
                    new_fragment_ids,
                    original_fragment_ids,
                } => {
                    // Register the rewritten fragments only when every
                    // original fragment that fed them was already indexed —
                    // otherwise some rewritten rows were never indexed and
                    // claiming coverage would skip a needed flat scan (legacy
                    // `register_pure_rewrite_rows_update_frags_in_indices`).
                    let covers_all_originals = original_fragment_ids
                        .iter()
                        .all(|&id| bitmap.contains(id as u32));
                    if covers_all_originals {
                        for &id in new_fragment_ids {
                            bitmap.insert(id as u32);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Payload for [`RegisterFragReuse`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RegisterFragReuse {
    /// The `frag_reuse` system index entry to register. Any existing entry
    /// with the same name is replaced.
    pub frag_reuse_index: IndexMetadata,
    /// Eagerly-loaded `FragReuseVersion`s parsed from `frag_reuse_index`'s
    /// details. Mirrors [`AddIndex::frag_reuse_versions`]: the rebase rule
    /// smart-merges these against a concurrent `AddIndex`/`RegisterFragReuse`
    /// of the frag-reuse index, and a post-driver hook materialises the
    /// merged list into a fresh details file. `None` until the translation
    /// step populates it (see #6454 R3).
    pub frag_reuse_versions: Option<Vec<FragReuseVersion>>,
}

impl Action for RegisterFragReuse {
    impl_dyn_action!(RegisterFragReuse);

    fn apply(&self, _manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        indices.retain(|index| index.name != self.frag_reuse_index.name);
        indices.push(self.frag_reuse_index.clone());
        Ok(())
    }
}

/// Payload for [`UpdateMergedGenerations`].
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UpdateMergedGenerations {
    /// MemWAL region generations to mark merged. Merged into the MemWAL
    /// index by keeping the higher generation per shard.
    pub merged_generations: Vec<MergedGeneration>,
}

impl Action for UpdateMergedGenerations {
    impl_dyn_action!(UpdateMergedGenerations);

    fn apply(&self, manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()> {
        // `version` is the current manifest's; the MemWAL entry records the
        // version it is updated *to*, matching the legacy
        // `current_manifest.map_or(1, |m| m.version + 1)`.
        update_mem_wal_index_merged_generations(
            indices,
            manifest.version + 1,
            self.merged_generations.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::format::IndexFile;
    use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use crate::system_index::mem_wal::{MEM_WAL_INDEX_NAME, MergedGeneration};
    use uuid::Uuid;

    use super::super::test_support::*;
    use super::super::*;
    use super::*;

    #[test]
    fn add_index_apply_resolves_all_coverage_against_manifest() {
        let mut manifest = manifest_with_fragments(&[1, 2, 3]);
        let mut indices = vec![];

        let action = AddIndex {
            index: index_meta("idx", &[0], None),
            coverage: FragmentCoverage::All,
            frag_reuse_versions: None,
        };
        action.apply(&mut manifest, &mut indices).unwrap();

        assert_eq!(indices.len(), 1);
        // `All` resolves to every fragment present at apply time.
        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[1, 2, 3])));
    }

    #[test]
    fn add_index_apply_uses_bitmap_coverage_verbatim() {
        let mut manifest = manifest_with_fragments(&[1, 2, 3]);
        let mut indices = vec![];

        let action = AddIndex {
            index: index_meta("idx", &[0], None),
            coverage: FragmentCoverage::Bitmap(bitmap(&[1, 3])),
            frag_reuse_versions: None,
        };
        action.apply(&mut manifest, &mut indices).unwrap();

        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[1, 3])));
    }

    #[test]
    fn add_index_apply_replaces_entry_with_same_uuid() {
        let mut manifest = manifest_with_fragments(&[1]);
        let existing = index_meta("idx", &[0], Some(&[1]));
        let mut indices = vec![existing.clone()];

        // A second AddIndex carrying the same UUID replaces the entry.
        let mut replacement = index_meta("idx-renamed", &[0], None);
        replacement.uuid = existing.uuid;
        let action = AddIndex {
            index: replacement,
            coverage: FragmentCoverage::All,
            frag_reuse_versions: None,
        };
        action.apply(&mut manifest, &mut indices).unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "idx-renamed");
    }

    #[test]
    fn remove_index_apply_drops_by_uuid_and_ignores_unknown() {
        let mut manifest = empty_manifest();
        let keep = index_meta("keep", &[0], None);
        let drop = index_meta("drop", &[1], None);
        let mut indices = vec![keep.clone(), drop.clone()];

        RemoveIndex { uuid: drop.uuid }
            .apply(&mut manifest, &mut indices)
            .unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].uuid, keep.uuid);

        // An unknown UUID is silently ignored.
        RemoveIndex {
            uuid: Uuid::new_v4(),
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();
        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn rewrite_index_apply_swaps_uuid_files_and_bitmap() {
        let mut manifest = empty_manifest();
        let old = index_meta("idx", &[0], Some(&[1, 2]));
        let mut indices = vec![old.clone()];

        let new_uuid = Uuid::new_v4();
        let new_files = vec![IndexFile {
            path: "new.idx".into(),
            size_bytes: 42,
        }];
        RewriteIndex {
            old_uuid: old.uuid,
            new_uuid,
            new_index_files: Some(new_files.clone()),
            new_fragment_bitmap: bitmap(&[3, 4]),
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].uuid, new_uuid);
        assert_eq!(indices[0].files, Some(new_files));
        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[3, 4])));
    }

    #[test]
    fn rewrite_index_apply_ignores_missing_old_uuid() {
        let mut manifest = empty_manifest();
        let mut indices = vec![index_meta("idx", &[0], Some(&[1]))];

        // An old UUID no longer present is silently skipped.
        RewriteIndex {
            old_uuid: Uuid::new_v4(),
            new_uuid: Uuid::new_v4(),
            new_index_files: None,
            new_fragment_bitmap: bitmap(&[2]),
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[1])));
    }

    #[test]
    fn invalidate_index_coverage_apply_shrinks_only_covering_indices() {
        let mut manifest = empty_manifest();
        let covering = index_meta("covers-0", &[0], Some(&[1, 2, 3]));
        let other = index_meta("covers-9", &[9], Some(&[1, 2, 3]));
        let mut indices = vec![covering, other];

        InvalidateIndexCoverage {
            fragment_ids: vec![2],
            field_ids: vec![0],
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        // Only the index covering field 0 loses fragment 2.
        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[1, 3])));
        assert_eq!(indices[1].fragment_bitmap, Some(bitmap(&[1, 2, 3])));
    }
    #[test]
    fn rebind_index_coverage_apply_remaps_non_covering_indices() {
        let mut manifest = empty_manifest();
        // `untouched` covers none of the modified fields, so it is rebound.
        let untouched = index_meta("untouched", &[0], Some(&[1, 2]));
        // `modified` covers field 7, a modified field, so it is left alone.
        let modified = index_meta("modified", &[7], Some(&[1, 2]));
        let mut indices = vec![untouched, modified];

        RebindIndexCoverage {
            // Group {1} -> {10} is fully covered and rebound; group {99} -> {100}
            // is not covered at all, so it is a no-op.
            remap: IndexCoverageRemap::RewriteGroups(vec![
                group_remap(&[1], &[10]),
                group_remap(&[99], &[100]),
            ]),
            modified_field_ids: vec![7],
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[2, 10])));
        assert_eq!(indices[1].fragment_bitmap, Some(bitmap(&[1, 2])));
    }

    #[test]
    fn rebind_index_coverage_apply_handles_cardinality_changing_groups() {
        let mut manifest = empty_manifest();
        // Coverage {1, 2, 3}: group {1} splits into {10, 11}; group {2, 3}
        // compacts into nothing. Both old id sets are fully covered.
        let index = index_meta("idx", &[0], Some(&[1, 2, 3]));
        let mut indices = vec![index];

        RebindIndexCoverage {
            remap: IndexCoverageRemap::RewriteGroups(vec![
                group_remap(&[1], &[10, 11]),
                group_remap(&[2, 3], &[]),
            ]),
            modified_field_ids: vec![],
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[10, 11])));
    }

    #[test]
    fn rebind_index_coverage_apply_rejects_partially_covered_group() {
        let mut manifest = empty_manifest();
        // The index covers fragment 1 but not 2; a group spanning both is a
        // split of indexed and non-indexed data, which `apply` rejects —
        // exactly as `recalculate_fragment_bitmap` does.
        let index = index_meta("idx", &[0], Some(&[1]));
        let mut indices = vec![index];

        let err = RebindIndexCoverage {
            remap: IndexCoverageRemap::RewriteGroups(vec![group_remap(&[1, 2], &[10])]),
            modified_field_ids: vec![],
        }
        .apply(&mut manifest, &mut indices)
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "got: {err}");
    }

    #[test]
    fn rebind_index_coverage_insert_pure_rewrite_gates_on_full_coverage() {
        let mut manifest = empty_manifest();
        // `full` covers every original fragment (1, 2), so the rewritten
        // fragment 10 is registered into its bitmap.
        let full = index_meta("full", &[0], Some(&[1, 2]));
        // `partial` covers only original fragment 1; the gate fails and the
        // bitmap is left untouched — fragment 10's rows are not all indexed.
        let partial = index_meta("partial", &[0], Some(&[1]));
        // `modified` covers field 7, a modified field, so it is skipped.
        let modified = index_meta("modified", &[7], Some(&[1, 2]));
        // An index with no coverage bitmap is skipped without panicking.
        let unindexed = index_meta("unindexed", &[0], None);
        let mut indices = vec![full, partial, modified, unindexed];

        RebindIndexCoverage {
            remap: IndexCoverageRemap::InsertPureRewrite {
                new_fragment_ids: vec![10],
                original_fragment_ids: vec![1, 2],
            },
            modified_field_ids: vec![7],
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        assert_eq!(indices[0].fragment_bitmap, Some(bitmap(&[1, 2, 10])));
        assert_eq!(indices[1].fragment_bitmap, Some(bitmap(&[1])));
        assert_eq!(indices[2].fragment_bitmap, Some(bitmap(&[1, 2])));
        assert_eq!(indices[3].fragment_bitmap, None);
    }

    #[test]
    fn register_frag_reuse_apply_replaces_entry_with_same_name() {
        let mut manifest = empty_manifest();
        let mut indices = vec![index_meta(FRAG_REUSE_INDEX_NAME, &[], None)];

        let replacement = index_meta(FRAG_REUSE_INDEX_NAME, &[], None);
        let replacement_uuid = replacement.uuid;
        RegisterFragReuse {
            frag_reuse_index: replacement,
            frag_reuse_versions: None,
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].uuid, replacement_uuid);
    }

    #[test]
    fn update_merged_generations_apply_creates_and_merges_memwal_entry() {
        use crate::transaction::mem_wal::load_mem_wal_index_details;

        let mut manifest = empty_manifest();
        let mut indices = vec![];
        let shard = Uuid::new_v4();

        // First update creates the MemWAL index entry.
        UpdateMergedGenerations {
            merged_generations: vec![MergedGeneration::new(shard, 5)],
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, MEM_WAL_INDEX_NAME);

        // A lower generation for the same shard is ignored (keep the higher).
        UpdateMergedGenerations {
            merged_generations: vec![MergedGeneration::new(shard, 3)],
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();
        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(details.merged_generations.len(), 1);
        assert_eq!(details.merged_generations[0].generation, 5);

        // A higher generation wins.
        UpdateMergedGenerations {
            merged_generations: vec![MergedGeneration::new(shard, 8)],
        }
        .apply(&mut manifest, &mut indices)
        .unwrap();
        let details = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(details.merged_generations[0].generation, 8);
    }
}
