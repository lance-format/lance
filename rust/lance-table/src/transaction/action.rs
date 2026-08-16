// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The action vocabulary of Transaction V2.
//!
//! Where an [`Operation`](super::Operation) names one whole change and carries a
//! post-image of the parts of the manifest it touches, a [`UserOperation`] is an
//! ordered list of [`Action`]s, each recording a single *delta*. Composing
//! several changes into one atomic commit and replaying a change onto a
//! different version both fall out of that, with no per-operation logic.
//!
//! The wire format and the reasoning behind it live in
//! `protos/transaction/actions.proto`; the two definitions must stay in step.
//! Only the subset of the drafted vocabulary that is implemented appears here --
//! an action this build does not know is rejected on load rather than skipped.
//!
//! ```text
//! action             the vocabulary (this module)
//! action::proto      its persisted protobuf encoding
//! ```
//!
//! # Stability
//!
//! Transaction V2 is a pre-vote draft. Nothing in this module is a compatibility
//! contract, and a transaction carrying a [`UserOperation`] is rejected outright
//! by libraries that predate it.

mod proto;

use crate::format::{BasePath, DataFile, DeletionFile, RowIdMeta};
use crate::rowids::version::RowDatasetVersionMeta;
use lance_core::datatypes::Field;
use lance_core::deepsize::DeepSizeOf;

/// A reference to a counter-allocated identifier -- a field id, fragment id, or
/// base id -- that may not have been assigned yet.
///
/// [`Ref::Committed`] is a concrete id that already exists in the manifest.
/// [`Ref::Local`] is a placeholder token minted by an `Add*` action earlier in
/// the same [`UserOperation`]; it resolves to a freshly-allocated id at apply,
/// and re-resolves against the target's counters when the operation is replayed
/// onto a newer version. That re-resolution is what lets two independent
/// `AddField`s on divergent branches become two distinct fields rather than a
/// collision.
///
/// Local tokens are scoped to one [`UserOperation`] and must be distinct within
/// it. The three id spaces do not share a token namespace: a fragment token 0
/// and a field token 0 are unrelated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, DeepSizeOf)]
pub enum Ref {
    Committed(u64),
    Local(u32),
}

impl Ref {
    /// The committed id, or `None` if this is an unresolved local token.
    pub fn committed(&self) -> Option<u64> {
        match self {
            Self::Committed(id) => Some(*id),
            Self::Local(_) => None,
        }
    }

    /// The local token, or `None` if this reference is already committed.
    pub fn local(&self) -> Option<u32> {
        match self {
            Self::Local(token) => Some(*token),
            Self::Committed(_) => None,
        }
    }
}

/// A composable transaction: an ordered list of user actions that commit
/// atomically as a single manifest change.
///
/// The `uuid` and `read_version` carried on the wire mirror the enclosing
/// [`Transaction`](super::Transaction) and are filled in from it, so they are
/// not repeated here.
#[derive(Debug, Clone, PartialEq, DeepSizeOf, Default)]
pub struct UserOperation {
    /// Human-readable description of the whole operation, e.g. `"INSERT INTO t"`.
    pub description: String,
    /// The ordered steps this operation applies.
    pub actions: Vec<UserAction>,
}

impl UserOperation {
    pub fn new(description: impl Into<String>, actions: Vec<UserAction>) -> Self {
        Self {
            description: description.into(),
            actions,
        }
    }

    /// Every action in every step, in application order.
    pub fn iter_actions(&self) -> impl Iterator<Item = &Action> {
        self.actions.iter().flat_map(|step| step.actions.iter())
    }
}

/// A single user-recognizable step within a [`UserOperation`], e.g. "append
/// batch" or "rebuild index".
///
/// The description keeps transaction history readable: when a range of versions
/// is squashed, each original operation collapses into one step, so the sequence
/// a user performed survives even though the deltas are flattened when applied.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserAction {
    pub description: String,
    pub actions: Vec<Action>,
}

impl UserAction {
    pub fn new(description: impl Into<String>, actions: Vec<Action>) -> Self {
        Self {
            description: description.into(),
            actions,
        }
    }
}

/// A single granular change to the manifest.
///
/// The drafted vocabulary is larger than this; the variants here are the ones
/// this build implements end to end.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum Action {
    AddFragment(AddFragment),
    AddDataFile(AddDataFile),
    AddField(AddField),
    AddBase(AddBase),
    TombstoneFieldData(TombstoneFieldData),
    RemoveFragment(RemoveFragment),
    SetDeletionFile(SetDeletionFile),
    AlterField(AlterField),
}

impl Action {
    pub fn name(&self) -> &'static str {
        match self {
            Self::AddFragment(_) => "AddFragment",
            Self::AddDataFile(_) => "AddDataFile",
            Self::AddField(_) => "AddField",
            Self::AddBase(_) => "AddBase",
            Self::TombstoneFieldData(_) => "TombstoneFieldData",
            Self::RemoveFragment(_) => "RemoveFragment",
            Self::SetDeletionFile(_) => "SetDeletionFile",
            Self::AlterField(_) => "AlterField",
        }
    }

    /// Whether this action changes the data a reader would see, as opposed to
    /// rearranging how it is stored (compaction, a segment rebuild).
    ///
    /// CDC and streaming consumers use this to skip commits that cannot have
    /// changed any row's value.
    pub fn is_data_change(&self) -> bool {
        match self {
            Self::AddFragment(action) => action.data_change,
            Self::AddDataFile(action) => action.data_change,
            Self::TombstoneFieldData(action) => action.data_change,
            Self::RemoveFragment(action) => action.data_change,
            Self::SetDeletionFile(action) => action.data_change,
            // Schema and base-path changes touch no row values.
            Self::AddField(_) | Self::AddBase(_) | Self::AlterField(_) => false,
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Mint a new, empty fragment.
///
/// Its data files arrive via [`AddDataFile`] actions naming this fragment's
/// local token. A freshly-minted fragment has no deletion vector: it has no
/// committed rows to delete yet.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddFragment {
    /// Token standing in for the fragment id until it is allocated at apply.
    pub local: u32,
    /// Physical rows in the fragment, including rows later tombstoned.
    pub physical_rows: u64,
    /// Stable row id sequence. `None` on datasets without stable row ids, and
    /// on datasets that have them but where the ids are assigned at apply.
    pub row_id_meta: Option<RowIdMeta>,
    /// Per-row version metadata, carried exactly as on
    /// [`Fragment`](crate::format::Fragment). `None` means "stamp at apply".
    pub last_updated_at_version_meta: Option<RowDatasetVersionMeta>,
    pub created_at_version_meta: Option<RowDatasetVersionMeta>,
    /// `false` marks a pure rearrangement, e.g. a compaction rewrite.
    pub data_change: bool,
}

/// Add a data file to a fragment.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddDataFile {
    /// The fragment to add the file to: committed, or a fragment minted earlier
    /// in the same operation.
    pub fragment: Ref,
    /// The file. Its `fields` are placeholders and are stamped in at apply from
    /// `field_ids`, which is the authority for the column -> field mapping.
    pub file: DataFile,
    /// One entry per column in `file`, in column order.
    pub field_ids: Vec<Ref>,
    /// See [`AddFragment::data_change`].
    pub data_change: bool,
}

/// Mint a new schema field.
///
/// A nested column that introduces several fields is several ordered
/// `AddField`s -- parent first, each child naming its parent's local token.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddField {
    /// Token standing in for the field id until it is allocated at apply.
    pub local: u32,
    /// The parent field, or `None` for a top-level column.
    pub parent: Option<Ref>,
    /// The field definition. Its `id`, `parent_id`, and `children` are ignored:
    /// `local` and `parent` carry that structure, and each child is its own
    /// action.
    pub def: Field,
}

/// Mint a new base path.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddBase {
    /// Token standing in for the base id until it is allocated at apply.
    pub local: u32,
    /// The base path. Its `id` is ignored and stamped in at apply.
    pub base: BasePath,
}

/// Tombstone the data-file binding of committed fields within one fragment.
///
/// Each field's slot in whatever file currently backs it is marked tombstoned,
/// and a file left with no live field is pruned at apply. Data files have no id
/// of their own and a live field is backed by exactly one file, so this is how a
/// column's data is dropped or superseded: re-encoding a column is a tombstone
/// followed by an [`AddDataFile`] for the same field.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct TombstoneFieldData {
    pub fragment: Ref,
    /// Committed field ids whose current backing is tombstoned.
    pub field_ids: Vec<i32>,
    /// See [`AddFragment::data_change`].
    pub data_change: bool,
}

/// Remove a fragment entirely -- every row deleted, or the fragment replaced by
/// compaction.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RemoveFragment {
    pub fragment: Ref,
    /// See [`AddFragment::data_change`].
    pub data_change: bool,
}

/// Set (replace) a fragment's deletion file.
///
/// This is reference-stable rather than a delta: the fragment id is committed
/// and physical row offsets never move, so the post-image is unambiguous. The
/// newly-deleted rows -- the delta rebase and conflict detection need -- are
/// derived by diffing against the read version rather than serialized.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct SetDeletionFile {
    /// The fragment, by committed id. Unlike its sibling fragment actions this
    /// takes no [`Ref`]: a fragment minted in the same operation has no
    /// committed rows to delete.
    pub fragment: u64,
    /// The new deletion file, or `None` to clear the fragment's deletions.
    pub deletion_file: Option<DeletionFile>,
    /// See [`AddFragment::data_change`].
    pub data_change: bool,
}

/// Alter facets of an existing field in place, preserving its id.
///
/// Each facet is independently optional -- present means "change this", absent
/// means "leave it alone" -- so a widening cast and a nullability relaxation on
/// the same field commute. A cast additionally needs a [`TombstoneFieldData`]
/// plus a fresh [`AddDataFile`] to rewrite the data.
#[derive(Debug, Clone, PartialEq, DeepSizeOf, Default)]
pub struct AlterField {
    pub field: i32,
    pub name: Option<String>,
    /// The new Arrow logical type. The cast.
    pub logical_type: Option<String>,
    pub nullable: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_accessors() {
        assert_eq!(Ref::Committed(7).committed(), Some(7));
        assert_eq!(Ref::Committed(7).local(), None);
        assert_eq!(Ref::Local(2).local(), Some(2));
        assert_eq!(Ref::Local(2).committed(), None);
    }

    #[test]
    fn test_data_change_defaults_by_action_kind() {
        let alter = Action::AlterField(AlterField {
            field: 1,
            name: Some("renamed".into()),
            ..Default::default()
        });
        assert!(!alter.is_data_change(), "a rename changes no row values");

        let remove = Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(3),
            data_change: true,
        });
        assert!(remove.is_data_change());
    }

    #[test]
    fn test_iter_actions_flattens_steps_in_order() {
        let operation = UserOperation::new(
            "two steps",
            vec![
                UserAction::new(
                    "first",
                    vec![Action::RemoveFragment(RemoveFragment {
                        fragment: Ref::Committed(1),
                        data_change: true,
                    })],
                ),
                UserAction::new(
                    "second",
                    vec![Action::RemoveFragment(RemoveFragment {
                        fragment: Ref::Committed(2),
                        data_change: true,
                    })],
                ),
            ],
        );
        let fragments = operation
            .iter_actions()
            .map(|action| match action {
                Action::RemoveFragment(remove) => remove.fragment,
                other => panic!("unexpected action {other}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(fragments, vec![Ref::Committed(1), Ref::Committed(2)]);
    }
}
