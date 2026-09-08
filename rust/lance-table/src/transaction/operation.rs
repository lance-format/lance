// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The vocabulary of changes a transaction can describe.
//!
//! Each [`Operation`] variant names one kind of change and carries exactly the
//! inputs needed to apply it: the fragments to add, the fields that were
//! rewritten, the indices that were rebuilt. Applying them is
//! [`super::manifest_build`]; deciding whether two of them collide is
//! [`super::conflicts`].

use crate::format::key_existence::KeyExistenceFilter;
use crate::format::overlay::DataOverlayFile;
use crate::format::{BasePath, DataFile, Fragment, IndexFile, IndexMetadata};
use crate::system_index::mem_wal::CompactedSsTable;
use crate::transaction::UpdateMap;
use lance_core::datatypes::Schema;
use lance_core::deepsize::DeepSizeOf;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct DataReplacementGroup(pub u64, pub DataFile);

/// Overlay files to append to a single fragment, in order (the last entry is
/// newest). The overlays are appended to the fragment's existing `overlays`
/// list rather than replacing it, so overlays written by concurrent commits are
/// preserved. Each overlay's `committed_version` is stamped to the new dataset
/// version at commit time (re-stamped on retry).
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct DataOverlayGroup {
    pub fragment_id: u64,
    pub overlays: Vec<DataOverlayFile>,
}

/// An operation on a dataset.
#[derive(Debug, Clone, DeepSizeOf)]
pub enum Operation {
    /// Adding new fragments to the dataset. The fragments contained within
    /// haven't yet been assigned a final ID.
    Append { fragments: Vec<Fragment> },
    /// Updated fragments contain those that have been modified with new deletion
    /// files. The deleted fragment IDs are those that should be removed from
    /// the manifest.
    Delete {
        updated_fragments: Vec<Fragment>,
        deleted_fragment_ids: Vec<u64>,
        predicate: String,
    },
    /// Overwrite the entire dataset with the given fragments. This is also
    /// used when initially creating a table.
    ///
    /// The fragments are newly written ones and are assigned fresh ids at commit
    /// time, continuing from the dataset's highest id ever used; the ids they
    /// arrive with are ignored.
    ///
    /// A fragment carrying a deletion file is rejected. A deletion file's path
    /// embeds the fragment id, so it cannot follow its fragment to the new id:
    /// minting a fragment and giving it a deletion file are mutually exclusive in
    /// one transaction. Use [`Self::Delete`] to commit deletions against existing
    /// fragments, or [`Self::Merge`] to change their schema.
    Overwrite {
        fragments: Vec<Fragment>,
        schema: Schema,
        config_upsert_values: Option<HashMap<String, String>>,
        initial_bases: Option<Vec<BasePath>>,
    },
    /// A new index has been created.
    CreateIndex {
        /// The new secondary indices,
        /// any existing indices with the same name will be replaced.
        new_indices: Vec<IndexMetadata>,
        /// The indices that have been modified.
        removed_indices: Vec<IndexMetadata>,
    },
    /// Data is rewritten but *not* modified. This is used for things like
    /// compaction or re-ordering. Contains the old fragments and the new
    /// ones that have been replaced.
    ///
    /// This operation will modify the row addresses of existing rows and
    /// so any existing index covering a rewritten fragment will need to be
    /// remapped.
    Rewrite {
        /// Groups of fragments that have been modified
        groups: Vec<RewriteGroup>,
        /// Indices that have been updated with the new row addresses
        rewritten_indices: Vec<RewrittenIndex>,
        /// The fragment reuse index to be created or updated to
        frag_reuse_index: Option<IndexMetadata>,
    },
    /// Replace data in a column in the dataset with new data. This is used for
    /// null column population where we replace an entirely null column with a
    /// new column that has data.
    ///
    /// This operation will only allow replacing files that contain the same schema
    /// e.g. if the original files contain columns A, B, C and the new files contain
    /// only columns A, B then the operation is not allowed. As we would need to split
    /// the original files into two files, one with column A, B and the other with column C.
    ///
    /// Corollary to the above: the operation will also not allow replacing files unless the
    /// affected columns all have the same datafile layout across the fragments being replaced.
    ///
    /// e.g. if fragments being replaced contain files with different schema layouts on
    /// the column being replaced, the operation is not allowed.
    /// say `frag_1: [A] [B, C]` and `frag_2: [A, B] [C]` and we are trying to replace column A
    /// with a new column A, the operation is not allowed.
    DataReplacement {
        replacements: Vec<DataReplacementGroup>,
    },
    /// Attach overlay files to fragments, supplying new values for a subset of
    /// `(physical offset, field)` cells without rewriting the fragments' base
    /// data files. See [`DataOverlayFile`] and the Data Overlay Files
    /// specification for resolution, coverage, and versioning rules.
    DataOverlay { groups: Vec<DataOverlayGroup> },
    /// Merge a new column in
    /// 'fragments' is the final fragments include all data files, the new fragments must align with old ones at rows.
    /// 'schema' is not forced to include existed columns, which means we could use Merge to drop column data
    Merge {
        fragments: Vec<Fragment>,
        schema: Schema,
        /// Set when this merge makes no nullability-affecting schema change:
        /// it introduces no field that data staged against an earlier schema
        /// could not safely omit. Without the assertion the merge conflicts
        /// with concurrent appends in either commit order, since a stale
        /// append omits new columns entirely and its rows read as null.
        preserves_nullability: bool,
    },
    /// Restore an old version of the database
    Restore { version: u64 },
    /// Reserves fragment ids for future use
    /// This can be used when row ids need to be known before a transaction
    /// has been committed.  It is used during a rewrite operation to allow
    /// indices to be remapped to the new row ids as part of the operation.
    ReserveFragments { num_fragments: u32 },

    /// Update values in the dataset.
    ///
    /// Updates are generally vertical or horizontal.
    ///
    /// A vertical update adds new rows.  In this case, the updated_fragments
    /// will only have existing rows deleted and will not have any new fields added.
    /// All new data will be contained in new_fragments.
    /// This is what is used by a merge_insert that matches the whole schema and what
    /// is used by the dataset updater.
    ///
    /// A horizontal update adds new columns.  In this case, the updated fragments
    /// may have fields removed or added.  It is even possible for a field to be tombstoned
    /// and then added back in the same update. (which is a field modification).  If any
    /// fields are modified in this way then they need to be added to the fields_modified list.
    /// This way we can correctly update the indices.
    /// This is what is used by a merge insert that does not match the whole schema.
    Update {
        /// Ids of fragments that have been moved
        removed_fragment_ids: Vec<u64>,
        /// Fragments that have been updated
        updated_fragments: Vec<Fragment>,
        /// Fragments that have been added
        new_fragments: Vec<Fragment>,
        /// The fields that have been modified
        fields_modified: Vec<u32>,
        /// MemWAL SSTables to mark as compacted after this transaction.
        compacted_sstables: Vec<CompactedSsTable>,
        /// The fields that used to judge whether to preserve the new frag's id into
        /// the frag bitmap of the specified indices.
        fields_for_preserving_frag_bitmap: Vec<u32>,
        /// The mode of update
        update_mode: Option<UpdateMode>,
        /// Optional filter for detecting conflicts on inserted row keys.
        /// Only tracks keys from INSERT operations during merge insert, not updates.
        inserted_rows_filter: Option<KeyExistenceFilter>,
        /// Physical row offsets (per fragment) that matched `update_columns` for RewriteColumns.
        /// `None` means callers did not supply offsets; `build_manifest` skips partial refresh then.
        updated_fragment_offsets: Option<UpdatedFragmentOffsets>,
    },

    /// Project to a new schema.
    Project {
        schema: Schema,
        /// Set when this projection makes no nullability-affecting schema
        /// change, as a rename or a drop does not. A nullability tightening
        /// must not set this: its producer proved the claim by scanning at its
        /// read version, so a concurrent write can falsify it and the
        /// projection conflicts with value-writes in either commit order.
        preserves_nullability: bool,
    },

    /// Update the dataset configuration and metadata.
    ///
    /// Schema or field metadata updates conflict with a concurrent
    /// [`Self::Merge`] in either commit order. A merge carries complete schema
    /// state from its read version, so rebasing the operations could discard
    /// metadata installed by the other transaction.
    UpdateConfig {
        config_updates: Option<UpdateMap>,
        table_metadata_updates: Option<UpdateMap>,
        schema_metadata_updates: Option<UpdateMap>,
        field_metadata_updates: HashMap<i32, UpdateMap>,
    },
    /// Update SSTable compaction progress in the MemWAL index.
    ///
    /// This is used during merge-insert to atomically record which
    /// SSTables have been compacted into the base table.
    UpdateMemWalState {
        compacted_sstables: Vec<CompactedSsTable>,
    },

    /// Clone a dataset.
    Clone {
        is_shallow: bool,
        ref_name: Option<String>,
        ref_version: u64,
        ref_path: String,
        branch_name: Option<String>,
    },

    // Update base paths in the dataset (currently only supports adding new bases).
    UpdateBases {
        /// The new base paths to add to the manifest.
        new_bases: Vec<BasePath>,
    },
}

#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum UpdateMode {
    /// rows are deleted in current fragments and rewritten in new fragments.
    /// This is most optimal when the majority of columns are being rewritten
    /// or only a few rows are being updated.
    RewriteRows,

    /// within each fragment, columns are fully rewritten and inserted as new data files.
    /// Old versions of columns are tombstoned. This is most optimal when most rows are affected
    /// but a small subset of columns are affected.
    RewriteColumns,
}

/// Matched physical row offsets per fragment for a partial [`UpdateMode::RewriteColumns`] update.
///
/// Used with stable row IDs so `build_manifest` can refresh row-level version
/// metadata only for rows that were rewritten.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UpdatedFragmentOffsets(pub HashMap<u64, RoaringBitmap>);

impl DeepSizeOf for UpdatedFragmentOffsets {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.0.iter().fold(0_usize, |acc, (frag_id, bitmap)| {
            acc + frag_id.deep_size_of_children(context)
                + (bitmap.len() as usize).saturating_mul(std::mem::size_of::<u32>())
        })
    }
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Append { .. } => write!(f, "Append"),
            Self::Delete { .. } => write!(f, "Delete"),
            Self::Overwrite { .. } => write!(f, "Overwrite"),
            Self::CreateIndex { .. } => write!(f, "CreateIndex"),
            Self::Rewrite { .. } => write!(f, "Rewrite"),
            Self::Merge { .. } => write!(f, "Merge"),
            Self::Restore { .. } => write!(f, "Restore"),
            Self::ReserveFragments { .. } => write!(f, "ReserveFragments"),
            Self::Update { .. } => write!(f, "Update"),
            Self::Project { .. } => write!(f, "Project"),
            Self::UpdateConfig { .. } => write!(f, "UpdateConfig"),
            Self::DataReplacement { .. } => write!(f, "DataReplacement"),
            Self::DataOverlay { .. } => write!(f, "DataOverlay"),
            Self::Clone { .. } => write!(f, "Clone"),
            Self::UpdateMemWalState { .. } => write!(f, "UpdateMemWalState"),
            Self::UpdateBases { .. } => write!(f, "UpdateBases"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RewrittenIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
    pub new_index_details: prost_types::Any,
    pub new_index_version: u32,
    /// Files in the new index with their sizes.
    /// Empty list from older writers that didn't persist this field.
    pub new_index_files: Option<Vec<IndexFile>>,
}

impl DeepSizeOf for RewrittenIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.new_index_details
            .type_url
            .deep_size_of_children(context)
            + self.new_index_details.value.deep_size_of_children(context)
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct RewriteGroup {
    pub old_fragments: Vec<Fragment>,
    pub new_fragments: Vec<Fragment>,
}

impl PartialEq for RewriteGroup {
    fn eq(&self, other: &Self) -> bool {
        fn compare_vec<T: PartialEq>(a: &[T], b: &[T]) -> bool {
            a.len() == b.len() && a.iter().all(|f| b.contains(f))
        }
        compare_vec(&self.old_fragments, &other.old_fragments)
            && compare_vec(&self.new_fragments, &other.new_fragments)
    }
}

impl Operation {
    pub fn name(&self) -> &str {
        match self {
            Self::Append { .. } => "Append",
            Self::Delete { .. } => "Delete",
            Self::Overwrite { .. } => "Overwrite",
            Self::CreateIndex { .. } => "CreateIndex",
            Self::Rewrite { .. } => "Rewrite",
            Self::Merge { .. } => "Merge",
            Self::ReserveFragments { .. } => "ReserveFragments",
            Self::Restore { .. } => "Restore",
            Self::Update { .. } => "Update",
            Self::Project { .. } => "Project",
            Self::UpdateConfig { .. } => "UpdateConfig",
            Self::DataReplacement { .. } => "DataReplacement",
            Self::DataOverlay { .. } => "DataOverlay",
            Self::UpdateMemWalState { .. } => "UpdateMemWalState",
            Self::Clone { .. } => "Clone",
            Self::UpdateBases { .. } => "UpdateBases",
        }
    }
}
