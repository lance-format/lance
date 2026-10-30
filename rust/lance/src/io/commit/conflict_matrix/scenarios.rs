// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Named ways to stage a real [`Transaction`] against a dataset version.
//!
//! Every scenario goes through a production write path wherever one exists
//! (`InsertBuilder`, `DeleteBuilder`, `MergeInsertBuilder`, `CreateIndexBuilder`,
//! `FileFragment::write_overlay`), so the transaction under test carries the
//! fragment contents, deletion files and affected-row sets a real writer would
//! produce. Operations with no uncommitted builder are assembled by hand from
//! state read out of the fixture, never from placeholder fragments.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::ROW_ADDR;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::utils::address::RowAddress;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_select::RowAddrTreeMap;
use lance_table::format::{DataFile, Fragment};
use lance_table::transaction::action::{
    Action, AddDataFile, AddField, AddFragment, AddIndexSegment, AlterField, AssertUniqueKeys,
    CompositeOperation, ConfigUpdate, FieldRef, Ref, RefreshRowVersionMetadata, RemoveFragment,
    SetDeletionFile, TombstoneFieldData, UserStep,
};
use lance_table::transaction::{UpdateMap, UpdateMapEntry};

use crate::Result;
use crate::dataset::transaction::{
    DataReplacementGroup, Operation, RewriteGroup, Transaction, UpdateMode,
};
use crate::dataset::{
    CommitBuilder, Dataset, DeleteBuilder, InsertBuilder, MergeInsertBuilder, NewColumnTransform,
    WriteMode, WriteParams,
};
use crate::index::CreateIndexBuilder;

/// A transaction staged against a dataset version but not yet committed.
pub(super) struct Staged {
    pub transaction: Transaction,
    /// Carried separately because `CommitBuilder` takes it separately; the
    /// second phase of delete/update conflict detection needs it.
    pub affected_rows: Option<RowAddrTreeMap>,
}

impl Staged {
    fn new(transaction: Transaction) -> Self {
        Self {
            transaction,
            affected_rows: None,
        }
    }

    fn with_affected_rows(transaction: Transaction, affected_rows: Option<RowAddrTreeMap>) -> Self {
        Self {
            transaction,
            affected_rows,
        }
    }

    /// Commit against `base`, which pins the read version. `base` being behind
    /// the latest version is what forces the rebase under test.
    pub async fn commit(self, base: &Arc<Dataset>) -> Result<Dataset> {
        // The opt-in only gates committing an action set, so turning it on for
        // every scenario changes nothing for the legacy ones.
        let mut builder = CommitBuilder::new(base.clone())
            .with_max_retries(1)
            .with_experimental_composite_operations(true);
        if let Some(rows) = self.affected_rows {
            builder = builder.with_affected_rows(rows);
        }
        builder.execute(self.transaction).await
    }
}

/// The operations this suite stages, one variant per column and row of the
/// matrix. `Update` appears twice because its two modes take different arms of
/// `check_update_txn` and build different manifests.
///
/// The `Action*` variants are Transaction V2 action sets. Each one names the
/// legacy scenarios it stands in for in [`Scenario::analogues`], which is what
/// the parity check in `cases` holds it to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Scenario {
    Append,
    Delete,
    /// A delete whose predicate covers every row of a fragment, so the fragment
    /// is removed outright. That populates `deleted_fragment_ids` instead of
    /// `updated_fragments`, which several conflict arms key on.
    DeleteWholeFragment,
    UpdateRewriteRows,
    UpdateRewriteColumns,
    DataOverlay,
    DataReplacement,
    Project,
    /// A projection that does not assert `preserves_nullability`, which is the
    /// only thing `may_alter_nullability` keys on.
    ProjectAlterNullability,
    Merge,
    /// See [`Scenario::ProjectAlterNullability`].
    MergeAlterNullability,
    CreateIndex,
    Rewrite,
    Overwrite,
    Restore,
    /// A schema-metadata update. `updates_schema_or_field_metadata` keys on
    /// schema or field metadata specifically, not on config entries.
    UpdateConfig,
    /// [`Scenario::Append`]'s data files as `AddFragment` + `AddDataFile`.
    ActionAppend,
    /// [`Scenario::UpdateConfig`]'s schema-metadata key as a `ConfigUpdate`.
    ActionUpdateConfig,
    /// Both of the above in one commit, which no single legacy operation can
    /// express.
    ActionAppendAndConfig,
    /// A new nullable column `e`, backed in every committed fragment: what a
    /// legacy add-column commits as a `Merge`.
    ActionAddColumn,
    /// [`Scenario::DataReplacement`]'s file as `TombstoneFieldData` +
    /// `AddDataFile`.
    ActionReplaceColumn,
    /// A rename of `c`, which legacy commits as a `Project`.
    ActionRenameColumn,
    /// [`Scenario::CreateIndex`]'s segment as an `AddIndexSegment`.
    ActionCreateIndex,
    /// [`Scenario::ActionAppend`] and [`Scenario::ActionCreateIndex`] in one
    /// commit.
    ///
    /// The segment covers a committed fragment rather than the appended one:
    /// an index built before the commit cannot know the ids the appended rows
    /// will get. A segment over a minted fragment claims no committed coverage,
    /// so the dataset-free tier covers that shape instead.
    ActionAppendAndIndex,
    /// [`Scenario::Delete`]'s deletion file as a `SetDeletionFile`, carrying
    /// the same affected rows.
    ActionDelete,
    /// [`Scenario::DeleteWholeFragment`]'s removal as a `RemoveFragment`.
    ActionDeleteWholeFragment,
    /// [`Scenario::UpdateRewriteRows`] as a vertical update: the moved rows'
    /// deletions as `SetDeletionFile`, their new fragment as `AddFragment` +
    /// `AddDataFile`, and the inserted keys as `AssertUniqueKeys`.
    ActionUpdate,
}

/// What a staged `theirs` touches, relative to what every `ours` in the matrix
/// touches.
///
/// `ours` always works on fragment 0, field `c`. Staging `theirs` somewhere else
/// is the only way to ask "do these two conflict *because* of what they touch,
/// or merely because of what they are?". Without the axis the matrix scores a
/// conflict detector that rejects everything just as well as one that reasons
/// about what each side touches.
///
/// Two operations can be disjoint in two different ways, and legacy need not
/// treat them alike: a different fragment is visible in the fragment ids a
/// transaction carries, while a different column of the *same* fragment is only
/// visible to something that reasons about fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Placement {
    /// Fragment 0, field `c` — exactly what `ours` touches.
    Same,
    /// Fragment 0, field `d`. Same fragment, no column in common.
    ///
    /// Only meaningful for a field-scoped `theirs`; see
    /// [`Scenario::is_field_scoped`].
    OtherField,
    /// Fragment 1, which no `ours` touches.
    OtherFragment,
}

impl Placement {
    pub const ALL: [Self; 3] = [Self::Same, Self::OtherField, Self::OtherFragment];

    /// The fixture fragment this placement names.
    fn fragment(self) -> usize {
        match self {
            Self::Same | Self::OtherField => 0,
            Self::OtherFragment => 1,
        }
    }

    /// The field this placement names. Both `c` and `d` hold a data file to
    /// themselves, which is what lets a scenario swap one without orphaning
    /// another field.
    fn field(self) -> &'static str {
        match self {
            Self::Same | Self::OtherFragment => "c",
            Self::OtherField => "d",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Same => "same",
            Self::OtherField => "other_field",
            Self::OtherFragment => "other_fragment",
        }
    }
}

impl Scenario {
    pub const ALL: [Self; 27] = [
        Self::Append,
        Self::Delete,
        Self::DeleteWholeFragment,
        Self::UpdateRewriteRows,
        Self::UpdateRewriteColumns,
        Self::DataOverlay,
        Self::DataReplacement,
        Self::Project,
        Self::ProjectAlterNullability,
        Self::Merge,
        Self::MergeAlterNullability,
        Self::CreateIndex,
        Self::Rewrite,
        Self::Overwrite,
        Self::Restore,
        Self::UpdateConfig,
        Self::ActionAppend,
        Self::ActionUpdateConfig,
        Self::ActionAppendAndConfig,
        Self::ActionAddColumn,
        Self::ActionReplaceColumn,
        Self::ActionRenameColumn,
        Self::ActionCreateIndex,
        Self::ActionAppendAndIndex,
        Self::ActionDelete,
        Self::ActionDeleteWholeFragment,
        Self::ActionUpdate,
    ];

    /// Two-letter column heading for the matrix grid in `cases`.
    pub fn code(self) -> &'static str {
        match self {
            Self::Append => "ap",
            Self::Delete => "dl",
            Self::DeleteWholeFragment => "df",
            Self::UpdateRewriteRows => "ur",
            Self::UpdateRewriteColumns => "uc",
            Self::DataOverlay => "ov",
            Self::DataReplacement => "dr",
            Self::Project => "pj",
            Self::ProjectAlterNullability => "pa",
            Self::Merge => "mg",
            Self::MergeAlterNullability => "ma",
            Self::CreateIndex => "ci",
            Self::Rewrite => "rw",
            Self::Overwrite => "ow",
            Self::Restore => "rs",
            Self::UpdateConfig => "cf",
            Self::ActionAppend => "va",
            Self::ActionUpdateConfig => "vc",
            Self::ActionAppendAndConfig => "v+",
            Self::ActionAddColumn => "vn",
            Self::ActionReplaceColumn => "vr",
            Self::ActionRenameColumn => "vm",
            Self::ActionCreateIndex => "vi",
            Self::ActionAppendAndIndex => "vj",
            Self::ActionDelete => "vd",
            Self::ActionDeleteWholeFragment => "vf",
            Self::ActionUpdate => "vu",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Append => "append",
            Self::Delete => "delete",
            Self::DeleteWholeFragment => "delete_whole_fragment",
            Self::UpdateRewriteRows => "update_rewrite_rows",
            Self::UpdateRewriteColumns => "update_rewrite_columns",
            Self::DataOverlay => "data_overlay",
            Self::DataReplacement => "data_replacement",
            Self::Project => "project",
            Self::ProjectAlterNullability => "project_alter_nullability",
            Self::Merge => "merge",
            Self::MergeAlterNullability => "merge_alter_nullability",
            Self::CreateIndex => "create_index",
            Self::Rewrite => "rewrite",
            Self::Overwrite => "overwrite",
            Self::Restore => "restore",
            Self::UpdateConfig => "update_config",
            Self::ActionAppend => "action_append",
            Self::ActionUpdateConfig => "action_update_config",
            Self::ActionAppendAndConfig => "action_append_and_config",
            Self::ActionAddColumn => "action_add_column",
            Self::ActionReplaceColumn => "action_replace_column",
            Self::ActionRenameColumn => "action_rename_column",
            Self::ActionCreateIndex => "action_create_index",
            Self::ActionAppendAndIndex => "action_append_and_index",
            Self::ActionDelete => "action_delete",
            Self::ActionDeleteWholeFragment => "action_delete_whole_fragment",
            Self::ActionUpdate => "action_update",
        }
    }

    /// The legacy scenarios this one stands in for: itself, for a legacy
    /// scenario.
    ///
    /// An action set with several analogues is held to the strictest of them,
    /// since a legacy writer would have needed one commit per analogue and
    /// could land only where every one of those commits does.
    pub fn analogues(self) -> Vec<Self> {
        match self {
            Self::ActionAppend => vec![Self::Append],
            Self::ActionUpdateConfig => vec![Self::UpdateConfig],
            Self::ActionAppendAndConfig => vec![Self::Append, Self::UpdateConfig],
            Self::ActionAddColumn => vec![Self::Merge],
            Self::ActionReplaceColumn => vec![Self::DataReplacement],
            Self::ActionRenameColumn => vec![Self::Project],
            Self::ActionCreateIndex => vec![Self::CreateIndex],
            Self::ActionAppendAndIndex => vec![Self::Append, Self::CreateIndex],
            Self::ActionDelete => vec![Self::Delete],
            Self::ActionDeleteWholeFragment => vec![Self::DeleteWholeFragment],
            Self::ActionUpdate => vec![Self::UpdateRewriteRows],
            legacy => vec![legacy],
        }
    }

    /// Whether this operation replaces whole-dataset state rather than editing
    /// it incrementally. The fragment-level invariants do not apply across such
    /// a commit: `Overwrite` mints a fresh fragment set and `Restore` reinstates
    /// an older manifest, so row counts and overlay counts legitimately drop.
    pub fn replaces_state(self) -> bool {
        matches!(self, Self::Overwrite | Self::Restore)
    }

    /// Whether this operation's conflict behaviour can depend on *which*
    /// fragment it touches.
    ///
    /// The rest are dataset-scoped: an append mints fragments and inspects
    /// none, a projection and a merge act on the schema, and overwrite, restore
    /// and a config update replace or annotate whole-dataset state. Staging
    /// them at a different fragment would produce the same transaction, so the
    /// placement axis would cost runtime and prove nothing.
    pub fn is_fragment_scoped(self) -> bool {
        matches!(
            self,
            Self::Delete
                | Self::DeleteWholeFragment
                | Self::UpdateRewriteRows
                | Self::UpdateRewriteColumns
                | Self::DataOverlay
                | Self::DataReplacement
                | Self::CreateIndex
                | Self::Rewrite
                | Self::ActionDelete
                | Self::ActionDeleteWholeFragment
                | Self::ActionUpdate
        )
    }

    /// Whether this operation's conflict behaviour can depend on *which field*
    /// it touches, as opposed to only which fragment.
    ///
    /// A delete, a rewrite and a row-level update act on whole rows, and an
    /// index here covers `a` whatever else is going on, so pointing any of them
    /// at a different column produces the same transaction.
    pub fn is_field_scoped(self) -> bool {
        matches!(
            self,
            Self::UpdateRewriteColumns | Self::DataOverlay | Self::DataReplacement
        )
    }

    /// Stage against the fragment every `ours` in the matrix uses.
    pub async fn stage(self, dataset: &Arc<Dataset>) -> Result<Staged> {
        self.stage_with(dataset, Placement::Same).await
    }

    pub async fn stage_with(self, dataset: &Arc<Dataset>, placement: Placement) -> Result<Staged> {
        debug_assert!(
            placement == Placement::Same || self.is_fragment_scoped(),
            "{} is dataset-scoped; staging it at {} would produce the same transaction",
            self.name(),
            placement.name(),
        );
        debug_assert!(
            placement != Placement::OtherField || self.is_field_scoped(),
            "{} is not field-scoped; staging it at {} would produce the same transaction",
            self.name(),
            placement.name(),
        );
        let fragment = placement.fragment();
        let field = placement.field();
        match self {
            Self::Append => stage_append(dataset).await,
            Self::Delete => stage_delete(dataset, fragment).await,
            Self::DeleteWholeFragment => stage_delete_whole_fragment(dataset, fragment).await,
            Self::UpdateRewriteRows => stage_update_rewrite_rows(dataset, fragment).await,
            Self::UpdateRewriteColumns => {
                stage_update_rewrite_columns(dataset, fragment, field).await
            }
            Self::DataOverlay => stage_data_overlay(dataset, fragment, field).await,
            Self::DataReplacement => stage_data_replacement(dataset, fragment, field).await,
            Self::Project => stage_project(dataset, true).await,
            Self::ProjectAlterNullability => stage_project(dataset, false).await,
            Self::Merge => stage_merge(dataset, true).await,
            Self::MergeAlterNullability => stage_merge(dataset, false).await,
            Self::CreateIndex => stage_create_index(dataset, fragment).await,
            Self::Rewrite => stage_rewrite(dataset, fragment).await,
            Self::Overwrite => stage_overwrite(dataset).await,
            Self::Restore => stage_restore(dataset).await,
            Self::UpdateConfig => stage_update_config(dataset).await,
            Self::ActionAppend => stage_action_append(dataset, false).await,
            Self::ActionUpdateConfig => stage_action_update_config(dataset).await,
            Self::ActionAppendAndConfig => stage_action_append(dataset, true).await,
            Self::ActionAddColumn => stage_action_add_column(dataset).await,
            Self::ActionReplaceColumn => stage_action_replace_column(dataset).await,
            Self::ActionRenameColumn => stage_action_rename_column(dataset),
            Self::ActionCreateIndex => stage_action_create_index(dataset, false).await,
            Self::ActionAppendAndIndex => stage_action_create_index(dataset, true).await,
            Self::ActionDelete => as_action_set(dataset, stage_delete(dataset, fragment).await?),
            Self::ActionDeleteWholeFragment => as_action_set(
                dataset,
                stage_delete_whole_fragment(dataset, fragment).await?,
            ),
            Self::ActionUpdate => {
                as_action_set(dataset, stage_update_rewrite_rows(dataset, fragment).await?)
            }
        }
    }
}

/// Two rows of new data, used by the scenarios that add rows. Carries the full
/// fixture schema, which an append requires.
fn new_rows(start: i32) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
            ArrowField::new("d", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![start, start + 1])),
            Arc::new(Int32Array::from(vec![start * 10, start * 10 + 1])),
            Arc::new(Int32Array::from(vec![start + 100, start + 101])),
            Arc::new(Int32Array::from(vec![start + 200, start + 201])),
        ],
    )
    .unwrap()
}

async fn stage_append(dataset: &Arc<Dataset>) -> Result<Staged> {
    let params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let transaction = InsertBuilder::new(dataset.clone())
        .with_params(&params)
        .execute_uncommitted(vec![new_rows(100)])
        .await?;
    Ok(Staged::new(transaction))
}

async fn stage_overwrite(dataset: &Arc<Dataset>) -> Result<Staged> {
    let params = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    let transaction = InsertBuilder::new(dataset.clone())
        .with_params(&params)
        .execute_uncommitted(vec![new_rows(200)])
        .await?;
    Ok(Staged::new(transaction))
}

/// A single-row delete on `fragment`. The fixture's fragment `n` holds
/// `a = 3n..3n+2`, so the predicate picks that fragment's first row.
async fn stage_delete(dataset: &Arc<Dataset>, fragment: usize) -> Result<Staged> {
    stage_delete_row(dataset, fragment as i32 * 3).await
}

/// A delete of the one row whose `a` is `key`.
pub(super) async fn stage_delete_row(dataset: &Arc<Dataset>, key: i32) -> Result<Staged> {
    let staged = DeleteBuilder::new(dataset.clone(), &format!("a = {key}"))
        .execute_uncommitted()
        .await?;
    Ok(Staged::with_affected_rows(
        staged.transaction,
        staged.affected_rows,
    ))
}

/// A delete whose predicate covers every row of fragment 0, so the fragment is
/// removed rather than gaining a deletion file.
///
/// `"a < 3"` rather than a literal `true`: the whole-dataset form takes a
/// separate short-circuit in `DeleteBuilder` that reports no affected rows at
/// all, whereas this keeps `affected_rows` populated and still empties the
/// fragment. The assertion pins that shape, because a fixture change that left
/// even one row alive would silently turn this back into [`stage_delete`].
async fn stage_delete_whole_fragment(dataset: &Arc<Dataset>, fragment: usize) -> Result<Staged> {
    let first = fragment * 3;
    let predicate = format!("a >= {first} AND a < {}", first + 3);
    let staged = DeleteBuilder::new(dataset.clone(), &predicate)
        .execute_uncommitted()
        .await?;
    let Operation::Delete {
        deleted_fragment_ids,
        updated_fragments,
        ..
    } = &staged.transaction.operation
    else {
        unreachable!("a delete in uncommitted form always carries Delete");
    };
    assert!(
        !deleted_fragment_ids.is_empty() && updated_fragments.is_empty(),
        "this scenario exists to remove a fragment outright, but it removed {deleted_fragment_ids:?} \
         and updated {} fragments",
        updated_fragments.len(),
    );
    Ok(Staged::with_affected_rows(
        staged.transaction,
        staged.affected_rows,
    ))
}

/// A schema-metadata update.
///
/// Schema metadata rather than a config entry because
/// `updates_schema_or_field_metadata` keys on the schema and field maps
/// specifically — a plain config upsert leaves the pre-check against `Merge`
/// unreached. Assembled directly: the production path commits as it goes, and
/// the operation carries no data for a hand-built form to get wrong.
async fn stage_update_config(dataset: &Arc<Dataset>) -> Result<Staged> {
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: None,
            schema_metadata_updates: Some(schema_metadata_update(Scenario::UpdateConfig)),
            field_metadata_updates: HashMap::new(),
        },
    )))
}

/// The one schema-metadata key every config scenario writes, so that the legacy
/// and action forms contend for it.
fn schema_metadata_update(writer: Scenario) -> UpdateMap {
    UpdateMap {
        update_entries: vec![UpdateMapEntry {
            key: "conflict_matrix".into(),
            value: Some(writer.name().into()),
        }],
        replace: false,
    }
}

fn stage_action_set(dataset: &Dataset, actions: Vec<Action>) -> Staged {
    Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "conflict_matrix",
            actions,
        )])),
    ))
}

/// [`stage_append`]'s data files, re-expressed as actions. The files are the
/// ones the legacy append wrote, so the two forms add identical data.
///
/// Row ids and per-row versions are left for apply to assign, as they are for
/// a legacy append.
async fn stage_action_append(dataset: &Arc<Dataset>, with_config: bool) -> Result<Staged> {
    let Operation::Append { fragments } = stage_append(dataset).await?.transaction.operation else {
        unreachable!("an uncommitted append stages Operation::Append");
    };
    let mut actions = Vec::new();
    for (token, fragment) in (0u32..).zip(fragments) {
        actions.push(Action::AddFragment(AddFragment {
            id: Ref::Local(token),
            physical_rows: fragment
                .physical_rows
                .expect("a freshly written fragment knows its row count")
                as u64,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        }));
        for mut file in fragment.files {
            let field_ids = file
                .fields
                .iter()
                .copied()
                .map(FieldRef::Committed)
                .collect();
            file.fields = Default::default();
            actions.push(Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(token),
                file,
                field_ids,
                data_change: true,
                base: None,
            }));
        }
    }
    if with_config {
        actions.push(Action::ConfigUpdate(ConfigUpdate {
            schema_metadata: Some(schema_metadata_update(Scenario::ActionAppendAndConfig)),
            ..Default::default()
        }));
    }
    Ok(stage_action_set(dataset, actions))
}

async fn stage_action_update_config(dataset: &Arc<Dataset>) -> Result<Staged> {
    Ok(stage_action_set(
        dataset,
        vec![Action::ConfigUpdate(ConfigUpdate {
            schema_metadata: Some(schema_metadata_update(Scenario::ActionUpdateConfig)),
            ..Default::default()
        })],
    ))
}

/// A nullable column `e` added and backed in every committed fragment, the
/// shape a legacy add-column commits as a `Merge`.
async fn stage_action_add_column(dataset: &Arc<Dataset>) -> Result<Staged> {
    let mut schema = LanceSchema::try_from(&ArrowSchema::new(vec![ArrowField::new(
        "e",
        DataType::Int32,
        true,
    )]))?;
    // The id the field is minted with, so the file's own metadata agrees with
    // the manifest it lands in.
    schema.set_field_id(Some(dataset.manifest.max_field_id()));
    let mut actions = vec![Action::AddField(AddField {
        local: 0,
        parent: None,
        def: schema.fields[0].clone(),
    })];
    for fragment in dataset.fragments().iter() {
        let rows = fragment
            .physical_rows
            .expect("the fixture's fragments know their row counts");
        let values = (0..rows as i32).collect::<Vec<_>>();
        let mut file = write_file(dataset, "add_column", schema.clone(), &values).await?;
        file.fields = Default::default();
        actions.push(Action::AddDataFile(AddDataFile {
            fragment: Ref::Committed(fragment.id),
            file,
            field_ids: vec![FieldRef::Local(0)],
            data_change: true,
            base: None,
        }));
    }
    refresh_row_versions(dataset, &mut actions);
    Ok(stage_action_set(dataset, actions))
}

/// Restamp every fragment `actions` writes changed data into, as legacy
/// `Merge` does, which a dataset with stable row ids requires.
fn refresh_row_versions(dataset: &Dataset, actions: &mut Vec<Action>) {
    if !dataset.manifest.uses_stable_row_ids() {
        return;
    }
    let mut fragment_ids: Vec<u64> = actions
        .iter()
        .filter_map(|action| match action {
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(id),
                data_change: true,
                ..
            }) => Some(*id),
            _ => None,
        })
        .collect();
    fragment_ids.dedup();
    actions.push(Action::RefreshRowVersionMetadata(
        RefreshRowVersionMetadata { fragment_ids },
    ));
}

/// [`stage_data_replacement`]'s new file for `c` in fragment 0, installed by
/// tombstoning the file it replaces.
async fn stage_action_replace_column(dataset: &Arc<Dataset>) -> Result<Staged> {
    let Operation::DataReplacement { replacements } = stage_data_replacement(dataset, 0, "c")
        .await?
        .transaction
        .operation
    else {
        unreachable!("a data replacement stages Operation::DataReplacement");
    };
    let field = FieldRef::Committed(field_id(dataset, "c")?);
    let mut actions = Vec::new();
    for DataReplacementGroup(fragment, mut file) in replacements {
        file.fields = Default::default();
        actions.push(Action::TombstoneFieldData(TombstoneFieldData {
            fragment: Ref::Committed(fragment),
            field_ids: vec![field],
            data_change: true,
        }));
        actions.push(Action::AddDataFile(AddDataFile {
            fragment: Ref::Committed(fragment),
            file,
            field_ids: vec![field],
            data_change: true,
            base: None,
        }));
    }
    refresh_row_versions(dataset, &mut actions);
    Ok(stage_action_set(dataset, actions))
}

fn stage_action_rename_column(dataset: &Dataset) -> Result<Staged> {
    Ok(stage_action_set(
        dataset,
        vec![Action::AlterField(AlterField {
            field: FieldRef::Committed(field_id(dataset, "c")?),
            name: Some("c_renamed".into()),
            nullable: None,
        })],
    ))
}

/// [`stage_create_index`]'s segment, committed as an `AddIndexSegment`, after
/// [`stage_action_append`]'s actions when `with_append` is set.
async fn stage_action_create_index(dataset: &Arc<Dataset>, with_append: bool) -> Result<Staged> {
    let Operation::CreateIndex { new_indices, .. } =
        stage_create_index(dataset, 0).await?.transaction.operation
    else {
        unreachable!("an uncommitted index build stages Operation::CreateIndex");
    };
    let mut actions = if with_append {
        let Operation::CompositeOperation(appended) = stage_action_append(dataset, false)
            .await?
            .transaction
            .operation
        else {
            unreachable!("an action append stages a composite operation");
        };
        appended.iter_actions().cloned().collect()
    } else {
        Vec::new()
    };
    for index in new_indices {
        actions.push(Action::AddIndexSegment(AddIndexSegment {
            uuid: index.uuid,
            name: index.name,
            fields: index
                .fields
                .iter()
                .map(|id| FieldRef::Committed(*id))
                .collect(),
            covering_fields: Vec::new(),
            index_details: index.index_details,
            index_version: index.index_version,
            covered_fragments: index
                .fragment_bitmap
                .map(|ids| ids.iter().map(|id| Ref::Committed(u64::from(id))).collect()),
            files: index.files.unwrap_or_default(),
            base: None,
            created_at: index.created_at,
            dataset_version: Some(index.dataset_version),
            data_change: false,
        }));
    }
    Ok(stage_action_set(dataset, actions))
}

/// A staged legacy `Delete` or row-moving `Update`, re-expressed as actions
/// that write the same files and carry the same affected rows.
///
/// Rows removed from a fragment in place become a `SetDeletionFile`, a
/// fragment emptied a `RemoveFragment`, and an update's new fragments an
/// `AddFragment` + `AddDataFile`, keeping the moved rows' ids and versions.
/// The inserted-key filter becomes an `AssertUniqueKeys`.
pub(super) fn as_action_set(dataset: &Dataset, staged: Staged) -> Result<Staged> {
    let (updated, removed, added, filter) = match staged.transaction.operation {
        Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            ..
        } => (updated_fragments, deleted_fragment_ids, Vec::new(), None),
        Operation::Update {
            updated_fragments,
            removed_fragment_ids,
            new_fragments,
            inserted_rows_filter,
            ..
        } => (
            updated_fragments,
            removed_fragment_ids,
            new_fragments,
            inserted_rows_filter,
        ),
        other => {
            return Err(crate::Error::invalid_input(format!(
                "only a Delete or an Update re-expresses as a delete-and-append, not {other}"
            )));
        }
    };
    let mut actions = Vec::new();
    for fragment in updated {
        actions.push(Action::SetDeletionFile(SetDeletionFile {
            fragment: fragment.id,
            deletion_file: fragment.deletion_file,
            data_change: true,
            base: None,
        }));
    }
    for fragment in removed {
        actions.push(Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(fragment),
            data_change: true,
        }));
    }
    for (token, fragment) in (0u32..).zip(added) {
        actions.push(Action::AddFragment(AddFragment {
            id: Ref::Local(token),
            physical_rows: fragment
                .physical_rows
                .expect("a freshly written fragment knows its row count")
                as u64,
            row_id_meta: fragment.row_id_meta,
            last_updated_at_version_meta: fragment.last_updated_at_version_meta,
            created_at_version_meta: fragment.created_at_version_meta,
            data_change: true,
        }));
        for mut file in fragment.files {
            let field_ids = file
                .fields
                .iter()
                .copied()
                .map(FieldRef::Committed)
                .collect();
            file.fields = Default::default();
            actions.push(Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(token),
                file,
                field_ids,
                data_change: true,
                base: None,
            }));
        }
    }
    if let Some(filter) = filter {
        actions.push(Action::AssertUniqueKeys(AssertUniqueKeys {
            key_fields: filter
                .field_ids
                .iter()
                .copied()
                .map(FieldRef::Committed)
                .collect(),
            filter,
        }));
    }
    let mut action_set = stage_action_set(dataset, actions);
    action_set.affected_rows = staged.affected_rows;
    Ok(action_set)
}

/// The id of the field called `name`. An error rather than a panic, so a
/// scenario staged after a concurrent rename or drop reports that it cannot be
/// staged, which the oracle's serial run takes as nothing to compare.
fn field_id(dataset: &Dataset, name: &str) -> Result<i32> {
    dataset
        .schema()
        .field(name)
        .map(|field| field.id)
        .ok_or_else(|| crate::Error::invalid_input(format!("the schema has no field {name:?}")))
}

/// A merge insert that matches the whole schema, which is the production
/// producer of `UpdateMode::RewriteRows`: matched rows are deleted in place and
/// rewritten into a new fragment.
async fn stage_update_rewrite_rows(dataset: &Arc<Dataset>, fragment: usize) -> Result<Staged> {
    // `new_rows(n)` carries keys n and n+1, so starting at the fragment's second
    // row keeps both matches inside it and inserts nothing.
    stage_merge_insert(dataset, new_rows(fragment as i32 * 3 + 1)).await
}

/// A merge insert of the one row whose `a` is `key`, which moves it out of its
/// fragment with `b = key * 10`.
pub(super) async fn stage_update_row(dataset: &Arc<Dataset>, key: i32) -> Result<Staged> {
    stage_merge_insert(dataset, new_rows(key).slice(0, 1)).await
}

async fn stage_merge_insert(dataset: &Arc<Dataset>, source: RecordBatch) -> Result<Staged> {
    let reader = arrow_array::RecordBatchIterator::new(vec![Ok(source.clone())], source.schema());
    let staged = MergeInsertBuilder::try_new(dataset.clone(), vec!["a".into()])?
        .when_matched(crate::dataset::WhenMatched::UpdateAll)
        .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
        .try_build()?
        .execute_uncommitted(reader)
        .await?;
    Ok(Staged::with_affected_rows(
        staged.transaction,
        staged.affected_rows,
    ))
}

/// `RewriteColumns` has no uncommitted builder, so it is assembled from the
/// fixture's own fragments: field `c`'s data is restated for fragment 0 while
/// every other field keeps its existing file.
///
/// `field` is `c` or `d`, the only two with a data file to themselves. Swapping
/// the file that `a` and `b` share for one holding just `b` would leave `a` with
/// no file at all, and the fragment would read `a` back as null.
async fn stage_update_rewrite_columns(
    dataset: &Arc<Dataset>,
    fragment_index: usize,
    field: &str,
) -> Result<Staged> {
    let mut fragment = dataset.fragments()[fragment_index].clone();
    let field_id = field_id(dataset, field)?;
    let replacement = write_value_file(dataset, "update_columns", &[field], &[7, 7, 7]).await?;
    // Replace the file carrying the field rather than appending one, so the
    // fragment keeps exactly one file per field.
    let target = fragment
        .files
        .iter_mut()
        .find(|file| file.fields.contains(&field_id))
        .expect("the fixture gives this field a data file of its own");
    *target = replacement;
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![fragment],
            new_fragments: vec![],
            fields_modified: vec![field_id as u32],
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
            mem_wal_dir_id: None,
        },
    )))
}

/// The overlay this suite installs everywhere: fragment 0, field `c`, the last
/// row of the fragment. That row survives the `delete` scenario, so an overlay
/// that disappears did so because it was dropped, not because its row went away.
pub(super) const OVERLAY_VALUE: i32 = 42;

async fn stage_data_overlay(
    dataset: &Arc<Dataset>,
    fragment_id: usize,
    field: &str,
) -> Result<Staged> {
    let fragment = dataset.get_fragment(fragment_id).unwrap();
    let last_offset = fragment.physical_rows().await? as u32 - 1;
    let overlay_schema = dataset.schema().project(&[field])?;
    let mut writer = fragment.write_overlay(&overlay_schema).await?;

    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ADDR, DataType::UInt64, false),
            ArrowField::new(field, DataType::Int32, true),
        ])),
        vec![
            Arc::new(UInt64Array::from(vec![u64::from(
                RowAddress::new_from_parts(fragment_id as u32, last_offset),
            )])),
            Arc::new(Int32Array::from(vec![OVERLAY_VALUE])),
        ],
    )
    .unwrap();
    writer.write_batch(&batch).await?;
    let group = writer
        .finish()
        .await?
        .expect("overlay covers one cell, so a group is produced");

    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::DataOverlay {
            groups: vec![group],
        },
    )))
}

/// Replace the data file holding `field` with one carrying different values.
///
/// The field's own file rather than the one `a` and `b` share: `DataReplacement`
/// requires the new file to carry exactly the fields of the file it replaces, so
/// targeting the shared file would overwrite `a` as well and leave the dataset
/// with no usable identity column for the content oracle to compare on.
async fn stage_data_replacement(
    dataset: &Arc<Dataset>,
    fragment: usize,
    field: &str,
) -> Result<Staged> {
    let field_id = field_id(dataset, field)?;
    let existing = dataset.fragments()[fragment]
        .files
        .iter()
        .find(|file| file.fields.contains(&field_id))
        .expect("the fixture gives this field a data file of its own");
    assert_eq!(
        existing.fields.len(),
        1,
        "{field} must have a data file to itself for the replacement to be field-scoped",
    );
    let replacement = write_value_file(dataset, "replacement", &[field], &[5, 5, 5]).await?;
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(fragment as u64, replacement)],
        },
    )))
}

/// `preserves_nullability` is the flag `may_alter_nullability` keys on: a
/// projection that does not assert it is taken to have scanned for nulls at its
/// read version, so a concurrent write of values falsifies it. Staged both ways
/// because `false` is the only setting that reaches the global pre-check.
async fn stage_project(dataset: &Arc<Dataset>, preserves_nullability: bool) -> Result<Staged> {
    // Drop `c`, which holds a data file of its own and so can be pruned whole.
    // `d` stays: dropping both would leave a merge with no field-scoped column
    // to keep, and the point here is a projection that prunes one file rather
    // than most of the schema.
    let schema = dataset.schema().project(&["a", "b", "d"])?;
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::Project {
            schema,
            preserves_nullability,
        },
    )))
}

/// See [`stage_project`] for `preserves_nullability`.
async fn stage_merge(dataset: &Arc<Dataset>, preserves_nullability: bool) -> Result<Staged> {
    // A merge restates every fragment against a schema; here it drops `c`'s
    // data the way a merge that omits a column does. See `stage_project` for
    // why `d` stays.
    let schema = dataset.schema().project(&["a", "b", "d"])?;
    let field_c = field_id(dataset, "c")?;
    let fragments = dataset
        .fragments()
        .iter()
        .map(|fragment| {
            let mut fragment = fragment.clone();
            fragment
                .files
                .retain(|file| !file.fields.contains(&field_c));
            fragment
        })
        .collect::<Vec<_>>();
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::Merge {
            fragments,
            schema,
            preserves_nullability,
        },
    )))
}

/// A BTree over `a`, covering only `fragment`.
///
/// Scoped rather than whole-dataset so the index has a placement at all: the
/// `fragment_bitmap` it carries is what the rewrite arms of
/// `check_create_index_txn` compare against.
async fn stage_create_index(dataset: &Arc<Dataset>, fragment: usize) -> Result<Staged> {
    let mut owned = dataset.as_ref().clone();
    let params = ScalarIndexParams::new("btree".to_string());
    let index = CreateIndexBuilder::new(&mut owned, &["a"], IndexType::Scalar, &params)
        .name("a_idx".to_string())
        .fragments(vec![fragment as u32])
        .execute_uncommitted()
        .await?;
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![index],
            removed_indices: vec![],
        },
    )))
}

/// Compaction of fragment 0 into a freshly written fragment holding the same
/// rows. Written through `InsertBuilder` so the new fragment has real files.
async fn stage_rewrite(dataset: &Arc<Dataset>, fragment: usize) -> Result<Staged> {
    let old = dataset.fragments()[fragment].clone();
    let rows = dataset
        .scan()
        .with_fragments(vec![old.clone()])
        .try_into_batch()
        .await?;
    let params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let written = InsertBuilder::new(dataset.clone())
        .with_params(&params)
        .execute_uncommitted(vec![rows])
        .await?;
    let Operation::Append { mut fragments } = written.operation else {
        unreachable!("an append in uncommitted form always carries Append");
    };
    if dataset.manifest.uses_stable_row_ids() {
        // A rewrite carries the old rows' identities onto the new fragment.
        // Reuse compaction's own transfer rather than reimplementing it, or the
        // new fragment arrives with no row ids and the commit is rejected.
        crate::dataset::optimize::rechunk_row_lineage(
            dataset.as_ref(),
            &mut fragments,
            std::slice::from_ref(&old),
        )
        .await?;
    }
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        Operation::Rewrite {
            groups: vec![RewriteGroup {
                old_fragments: vec![old],
                new_fragments: fragments,
            }],
            rewritten_indices: vec![],
            frag_reuse_index: None,
        },
    )))
}

async fn stage_restore(dataset: &Arc<Dataset>) -> Result<Staged> {
    Ok(Staged::new(Transaction::new_from_version(
        dataset.manifest.version,
        // The fixture's first version, before the column `c` was added.
        Operation::Restore { version: 1 },
    )))
}

/// Write a standalone data file holding `values` for `fields`, laid out so it
/// can stand in for an existing file of the same fields.
async fn write_value_file(
    dataset: &Arc<Dataset>,
    name: &str,
    fields: &[&str],
    values: &[i32],
) -> Result<DataFile> {
    write_file(dataset, name, dataset.schema().project(fields)?, values).await
}

/// Write a standalone data file holding `values` in every column of `schema`.
async fn write_file(
    dataset: &Arc<Dataset>,
    name: &str,
    schema: LanceSchema,
    values: &[i32],
) -> Result<DataFile> {
    let columns = schema.fields.len();
    // A fresh uuid per call, the way real writers name data files. `name` is
    // only a human-readable hint: two stagings of the same scenario must not
    // collide, or the second silently overwrites the first's contents.
    let filename = format!("{name}-{}.lance", uuid::Uuid::new_v4());
    let path = dataset.base.clone().join("data").join(filename.as_str());
    let writer = dataset.object_store.create(&path).await?;
    let file_version = dataset.manifest.data_storage_format.lance_file_format();
    let mut writer = lance_file::versions::create_writer(
        file_version,
        writer,
        schema,
        lance_file::writer::FileWriterOptions::default(),
    )?;
    for column in 0..columns {
        writer
            .write_column(column, Arc::new(Int32Array::from(values.to_vec())))
            .await?;
    }
    let summary = writer.finish().await?;

    let mut data_file = DataFile::new_unstarted(filename, file_version);
    data_file.fields = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(field_id, _)| *field_id as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.column_indices = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(_, column_index)| *column_index as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.file_size_bytes = lance_io::utils::CachedFileSize::new(summary.size_bytes);
    Ok(data_file)
}

/// The dataset every case starts from.
///
/// Deliberately tiny — three rows per fragment, two fragments, four fields — so
/// a case costs a few milliseconds. The shape is still the smallest one that
/// lets every axis of the suite ask its question:
///
/// - **Two fragments**, so [`Placement::OtherFragment`] can point an operation
///   somewhere `ours` never touches.
/// - **`c` and `d`, each alone in its own data file**, so [`Placement::Same`]
///   and [`Placement::OtherField`] differ at file granularity, and so `project`
///   and `merge` have a file they can prune whole.
/// - **`a` and `b` sharing the file written first**, which is why no scenario
///   targets them: replacing that file to reach one orphans the other.
///
/// Version 1 holds `a`/`b`; versions 2 and 3 add `c` and `d`. Cases stage
/// against the last, which gives `restore` a version to fall back to.
pub(super) async fn fixture() -> Arc<Dataset> {
    let data = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5])),
            Arc::new(Int32Array::from(vec![0, 0, 0, 0, 0, 0])),
        ],
    )
    .unwrap();
    let params = WriteParams {
        max_rows_per_file: 3,
        // On, so `Dataset::validate` exercises the stable row id invariants —
        // notably that no row id is live in two fragments at once. They are a
        // no-op on a dataset that does not use them.
        enable_stable_row_ids: true,
        ..Default::default()
    };
    let mut dataset = InsertBuilder::new("memory://")
        .with_params(&params)
        .execute(vec![data])
        .await
        .unwrap();
    // Two separate `add_columns` calls, because each one gives its column a data
    // file to itself. `c` and `d` are the only fields a scenario can target
    // without disturbing another: `a` and `b` share the file written above, so
    // replacing it to reach one of them orphans the other.
    for (name, expression) in [("c", "a + 100"), ("d", "a + 200")] {
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(name.into(), expression.into())]),
                None,
                None,
            )
            .await
            .unwrap();
    }
    Arc::new(dataset)
}

/// Fragment ids and their logical row counts, for the row-count invariant.
pub(super) fn row_counts(dataset: &Dataset) -> HashMap<u64, u64> {
    dataset
        .fragments()
        .iter()
        .map(|fragment: &Fragment| {
            let deleted = fragment
                .deletion_file
                .as_ref()
                .and_then(|f| f.num_deleted_rows)
                .unwrap_or(0) as u64;
            (
                fragment.id,
                fragment.physical_rows.unwrap_or(0) as u64 - deleted,
            )
        })
        .collect()
}
