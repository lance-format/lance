// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Behaviour of the check: a pending transaction read at a small table, a
//! concurrent one applied to it for real, and the verdict on the pending one.
//! No IO: manifests are built in memory.

use super::*;
use crate::format::key_existence::FilterType;
use crate::format::{
    DataFile, DataStorageFormat, DeletionFile, DeletionFileType, Fragment, Manifest,
};
use crate::transaction::action::{
    AddBase, AddDataFile, AddField, AddFragment, AddFragmentParts, AddIndexSegment,
    AddIndexSegmentParts, AddOverlays, AdjustIndexCoverage, AdjustIndexCoverageParts, AlterField,
    AssertCellsUnchanged, AssertUniqueKeys, ConfigUpdate, DropField, DropIndex, RemoveFragment,
    RemoveIndexSegment, ReserveFragmentIds, ResetTable, RowSelection, SelectedRows,
    SetDeletionFile, TombstoneFieldData, UpdateCompactedSsTables, UserStep,
};
use crate::transaction::test_support::{default_build_config, overlay_with_field};
use crate::transaction::{Operation, RewriteGroup, Transaction, UpdateMapEntry, UpdateMode};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
use lance_core::datatypes::{Field, Schema};
use lance_file::version::ConcreteFileVersion;
use rstest::rstest;
use std::collections::HashMap;

const A: i32 = 0;
const C: i32 = 1;
const S: i32 = 2;

/// Two fragments of three rows over `a` and `c`, each in a data file of its
/// own, plus a struct `s` with one child, unbacked.
fn table() -> Manifest {
    let schema = ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, true),
        ArrowField::new("c", DataType::Int32, true),
        ArrowField::new(
            "s",
            DataType::Struct(Fields::from(vec![ArrowField::new(
                "x",
                DataType::Int32,
                true,
            )])),
            true,
        ),
    ]);
    let fragments = (0..2)
        .map(|id| {
            let mut fragment = Fragment::new(id);
            fragment.physical_rows = Some(3);
            fragment.files = vec![
                field_file(&format!("a-{id}"), A),
                field_file(&format!("c-{id}"), C),
            ];
            fragment
        })
        .collect();
    let mut manifest = Manifest::new(
        Schema::try_from(&schema).unwrap(),
        Arc::new(fragments),
        DataStorageFormat::new(ConcreteFileVersion::V2_0),
        HashMap::new(),
    );
    manifest.version = 2;
    manifest
}

fn field_file(name: &str, field: i32) -> DataFile {
    DataFile::new(
        format!("data/{name}.lance"),
        vec![field],
        vec![0],
        ConcreteFileVersion::V2_0,
        None,
        None,
    )
}

fn unbound(file: DataFile) -> DataFile {
    DataFile {
        fields: Default::default(),
        ..file
    }
}

fn set(actions: Vec<Action>) -> Operation {
    Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
        "step", actions,
    )]))
}

fn new_fragment(data_change: bool) -> Action {
    Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
        id: Ref::Local(0),
        physical_rows: 3,
        row_id_meta: None,
        last_updated_at_version_meta: None,
        created_at_version_meta: None,
        data_change,
    }))
}

fn append_actions() -> Vec<Action> {
    vec![
        new_fragment(true),
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Local(0),
                DataFile {
                    column_indices: vec![0, 1].into(),
                    ..unbound(field_file("new", A))
                },
                vec![FieldRef::Committed(A), FieldRef::Committed(C)],
                None,
                true,
            )
            .unwrap(),
        ),
    ]
}

fn append() -> Operation {
    set(append_actions())
}

/// `c`'s data in `fragment` re-encoded, its values unchanged.
fn reencode_c(fragment: u64) -> Operation {
    set(vec![
        Action::TombstoneFieldData(TombstoneFieldData::new_unchecked(
            Ref::Committed(fragment),
            vec![FieldRef::Committed(C)],
            false,
        )),
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Committed(fragment),
                unbound(field_file(&format!("c-reencoded-{fragment}"), C)),
                vec![FieldRef::Committed(C)],
                None,
                false,
            )
            .unwrap(),
        ),
    ])
}

/// `c`'s data in `fragment`, rewritten.
fn replace_c(fragment: u64) -> Operation {
    replace(C, fragment)
}

fn replace(field: i32, fragment: u64) -> Operation {
    set(vec![
        Action::TombstoneFieldData(TombstoneFieldData::new(
            Ref::Committed(fragment),
            vec![FieldRef::Committed(field)],
        )),
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Committed(fragment),
                unbound(field_file(&format!("{field}-new-{fragment}"), field)),
                vec![FieldRef::Committed(field)],
                None,
                true,
            )
            .unwrap(),
        ),
    ])
}

fn deletion_file(deleted: u64) -> DeletionFile {
    DeletionFile {
        read_version: 2,
        id: deleted,
        file_type: DeletionFileType::Array,
        num_deleted_rows: Some(deleted as usize),
        base_id: None,
        inline: None,
    }
}

fn delete_rows_actions(fragment: u64) -> Vec<Action> {
    vec![Action::SetDeletionFile(
        SetDeletionFile::try_new_unchecked(fragment, deletion_file(1), None, true).unwrap(),
    )]
}

fn delete_rows(fragment: u64) -> Operation {
    set(delete_rows_actions(fragment))
}

fn remove(fragment: u64) -> Operation {
    set(vec![Action::RemoveFragment(RemoveFragment::new(
        Ref::Committed(fragment),
    ))])
}

/// Rows of `fragment` deleted there and written anew in a fragment of its own.
fn moving_update(fragment: u64) -> Operation {
    let mut actions = append_actions();
    actions.extend(delete_rows_actions(fragment));
    set(actions)
}

/// [`moving_update`] asserting the keys in `hashes` over `a`.
fn merge_insert(hashes: &[u64]) -> Operation {
    let mut actions = append_actions();
    actions.push(key_assertion(hashes));
    set(actions)
}

fn key_filter(hashes: &[u64]) -> KeyExistenceFilter {
    KeyExistenceFilter {
        field_ids: vec![A],
        filter: FilterType::ExactSet(hashes.iter().copied().collect()),
    }
}

fn key_assertion(hashes: &[u64]) -> Action {
    Action::AssertUniqueKeys(
        AssertUniqueKeys::try_new_unchecked(vec![FieldRef::Committed(A)], key_filter(hashes))
            .unwrap(),
    )
}

fn overlay(fragment: u64) -> Operation {
    set(vec![Action::AddOverlays(AddOverlays::new_unchecked(
        Ref::Committed(fragment),
        vec![overlay_with_field(C, 0)],
    ))])
}

fn alter_c(alter: impl FnOnce(AlterField) -> AlterField) -> Operation {
    set(vec![Action::AlterField(alter(AlterField::new(
        FieldRef::Committed(C),
    )))])
}

fn drop_c() -> Operation {
    set(vec![Action::DropField(DropField::new(
        FieldRef::Committed(C),
    ))])
}

/// Drops `s`, children first.
fn drop_s() -> Operation {
    set(vec![
        Action::DropField(DropField::new(FieldRef::Committed(S + 1))),
        Action::DropField(DropField::new(FieldRef::Committed(S))),
    ])
}

fn added_field(name: &str, nullable: bool) -> Field {
    Field::try_from(ArrowField::new(name, DataType::Int32, nullable)).unwrap()
}

/// A column `e` added and backed in both fragments.
fn add_column(nullable: bool) -> Operation {
    let mut actions = vec![Action::AddField(AddField::new(
        0,
        added_field("e", nullable),
    ))];
    for fragment in 0..2 {
        actions.push(Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Committed(fragment),
                unbound(field_file(&format!("e-{fragment}"), 9)),
                vec![FieldRef::Local(0)],
                None,
                true,
            )
            .unwrap(),
        ));
    }
    set(actions)
}

fn add_child_under_s() -> Operation {
    set(vec![Action::AddField(
        AddField::new(0, added_field("y", true)).with_parent(FieldRef::Committed(S)),
    )])
}

fn index_over(fragments: &[u64]) -> Operation {
    set(vec![Action::AddIndexSegment(
        AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
            uuid: Uuid::new_v4(),
            name: "idx".into(),
            fields: vec![FieldRef::Committed(A)],
            covering_fields: vec![],
            index_details: None,
            index_version: 1,
            covered_fragments: Some(fragments.iter().copied().map(Ref::Committed).collect()),
            files: vec![],
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: false,
        })
        .unwrap(),
    )])
}

fn config(key: &str) -> Operation {
    set(vec![Action::ConfigUpdate(
        ConfigUpdate::default().with_config(UpdateMap {
            update_entries: vec![UpdateMapEntry::from((key, Some("v")))],
            replace: false,
        }),
    )])
}

fn add_base(name: &str, path: &str) -> Operation {
    set(vec![Action::AddBase(AddBase::new(
        0,
        BasePath::new(0, path.into(), Some(name.into()), false),
    ))])
}

fn assert_c_unchanged_in(fragment: u32) -> Operation {
    set(vec![Action::AssertCellsUnchanged(
        AssertCellsUnchanged::try_new(
            vec![FieldRef::Committed(C)],
            vec![RowSelection {
                fragment_id: fragment,
                rows: SelectedRows::Full,
            }],
        )
        .unwrap(),
    )])
}

fn reset() -> Operation {
    set(vec![Action::ResetTable(ResetTable)])
}

fn legacy_update(updated: Vec<Fragment>, new: Vec<Fragment>) -> Operation {
    Operation::Update {
        removed_fragment_ids: vec![],
        updated_fragments: updated,
        new_fragments: new,
        fields_modified: vec![],
        compacted_sstables: vec![],
        mem_wal_dir_id: None,
        fields_for_preserving_frag_bitmap: vec![],
        update_mode: Some(UpdateMode::RewriteRows),
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    }
}

/// Fragment 0 of [`table`] with one row deleted, as a legacy delete
/// restates it.
fn restated_with_deletions() -> Fragment {
    let mut fragment = table().fragments[0].clone();
    fragment.deletion_file = Some(deletion_file(1));
    fragment
}

fn legacy_delete() -> Operation {
    Operation::Delete {
        updated_fragments: vec![restated_with_deletions()],
        deleted_fragment_ids: vec![],
        predicate: "a = 0".into(),
    }
}

fn legacy_rewrite(old: u64) -> Operation {
    Operation::Rewrite {
        groups: vec![RewriteGroup {
            old_fragments: vec![table().fragments[old as usize].clone()],
            new_fragments: vec![table().fragments[old as usize].clone()],
        }],
        rewritten_indices: vec![],
        frag_reuse_index: None,
    }
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Lands,
    Conflicts,
}
use Outcome::{Conflicts, Lands};

/// What it takes for `ours`, read at `read`, to land once `theirs`, which
/// read `read` too, has committed.
fn outcome_at(read: &Manifest, ours: &Operation, theirs: &Operation) -> Outcome {
    let committed = Transaction::new(read.version, theirs.clone(), None);
    let latest = if matches!(theirs, Operation::Restore { .. }) {
        // Built from the manifest it restores, which no test has.
        let mut latest = read.clone();
        latest.version += 1;
        latest
    } else {
        committed
            .build_manifest(Some(read), vec![], "tx.txn", &default_build_config())
            .unwrap_or_else(|error| panic!("{theirs:?} does not apply: {error}"))
            .0
    };
    let pending = Footprint::of(&Transaction::new(read.version, ours.clone(), None), read);
    let window = Window::new(read, &latest, vec![Footprint::of(&committed, read)]);
    match pending.check(&window) {
        Verdict::Compatible => Outcome::Lands,
        Verdict::Conflict(_) => Outcome::Conflicts,
    }
}

fn outcome(ours: &Operation, theirs: &Operation) -> Outcome {
    outcome_at(&table(), ours, theirs)
}

/// Each pair in both orders: `ours` landing after `theirs`, then `theirs`
/// landing after `ours`.
#[rstest]
// Data: first committer wins on one field's data; other fields, other
// fragments, and deletions are untouched by it.
#[case::one_field_rewritten_twice(replace_c(0), replace_c(0), Conflicts, Conflicts)]
#[case::one_field_in_different_fragments(replace_c(0), replace_c(1), Lands, Lands)]
#[case::a_rewrite_and_a_delete(replace_c(0), delete_rows(0), Lands, Lands)]
#[case::a_rewrite_and_an_append(replace_c(0), append(), Lands, Lands)]
#[case::two_appends(append(), append(), Lands, Lands)]
// Deletions added on both sides conflict: the fragment's deletions are read
// whole.
#[case::two_deletes(delete_rows(0), delete_rows(0), Conflicts, Conflicts)]
#[case::a_delete_and_a_moving_update(delete_rows(0), moving_update(0), Conflicts, Conflicts)]
// A row-moving update copies the values it read; a write of the fragment's
// files is lost on the moved rows, in either order.
#[case::a_moving_update_and_a_rewrite(moving_update(0), replace_c(0), Conflicts, Conflicts)]
#[case::a_moving_update_and_a_rewrite_elsewhere(moving_update(0), replace_c(1), Lands, Lands)]
#[case::a_moving_update_and_an_overlay(moving_update(0), overlay(0), Conflicts, Conflicts)]
#[case::a_moving_update_and_an_added_column(
    moving_update(0),
    add_column(true),
    Conflicts,
    Conflicts
)]
// A removal discards what an overlay wrote over its rows, but an overlay
// cannot land on rows that are gone.
#[case::a_removal_and_an_overlay(remove(0), overlay(0), Lands, Conflicts)]
#[case::a_removal_and_a_rewrite(remove(0), replace_c(0), Conflicts, Conflicts)]
#[case::a_removal_and_an_added_column(remove(0), add_column(true), Conflicts, Conflicts)]
#[case::a_removal_and_a_delete(remove(0), delete_rows(0), Conflicts, Conflicts)]
// Overlays commute with each other and with a re-encode of the data beneath.
// New base values tombstone every overlay on the column as they land, a newer
// one included, so they conflict with an overlay in either order.
#[case::two_overlays(overlay(0), overlay(0), Lands, Lands)]
#[case::an_overlay_and_a_rewrite(overlay(0), replace_c(0), Conflicts, Conflicts)]
#[case::an_overlay_and_a_rewrite_elsewhere(overlay(0), replace_c(1), Lands, Lands)]
#[case::an_overlay_and_a_reencode(overlay(0), reencode_c(0), Lands, Lands)]
// A drop tombstones the field out of every fragment it finds; data written
// for it into new fragments needs it to still be there.
#[case::a_drop_and_an_append(drop_c(), append(), Lands, Conflicts)]
#[case::a_drop_and_a_rewrite(drop_c(), replace_c(0), Lands, Conflicts)]
#[case::a_drop_and_an_overlay(drop_c(), overlay(0), Lands, Conflicts)]
#[case::a_drop_and_a_rename(drop_c(), alter_c(|alter| alter.with_name("d")), Conflicts, Conflicts)]
#[case::a_struct_drop_and_a_child_added(drop_s(), add_child_under_s(), Conflicts, Conflicts)]
// Schema changes against rows.
#[case::a_rename_and_an_append(alter_c(|alter| alter.with_name("d")), append(), Lands, Lands)]
#[case::two_renames(
    alter_c(|alter| alter.with_name("d")),
    alter_c(|alter| alter.with_name("e")),
    Conflicts,
    Conflicts
)]
#[case::a_tightening_and_an_append(
    alter_c(|alter| alter.with_nullable(false)),
    append(),
    Conflicts,
    Conflicts
)]
#[case::a_tightening_and_a_rewrite(
    alter_c(|alter| alter.with_nullable(false)),
    replace_c(1),
    Conflicts,
    Conflicts
)]
// A tightening holds while no values were written for that field.
#[case::a_tightening_and_a_rewrite_of_another_field(
    alter_c(|alter| alter.with_nullable(false)),
    replace(A, 0),
    Lands,
    Conflicts
)]
#[case::a_tightening_and_a_delete(
    alter_c(|alter| alter.with_nullable(false)),
    delete_rows(0),
    Lands,
    Lands
)]
#[case::a_required_column_and_an_append(add_column(false), append(), Conflicts, Conflicts)]
#[case::a_nullable_column_and_an_append(add_column(true), append(), Lands, Lands)]
#[case::a_child_added_and_an_append(add_child_under_s(), append(), Lands, Conflicts)]
// Keys: an assertion is a precondition on its own commit.
#[case::an_assertion_and_an_append(merge_insert(&[1]), append(), Conflicts, Lands)]
#[case::an_assertion_and_a_rewrite_of_the_key(merge_insert(&[1]), set(vec![
    Action::TombstoneFieldData(TombstoneFieldData::new(Ref::Committed(0), vec![FieldRef::Committed(A)])),
]), Conflicts, Lands)]
#[case::disjoint_assertions(merge_insert(&[1]), merge_insert(&[2]), Lands, Lands)]
#[case::colliding_assertions(merge_insert(&[1]), merge_insert(&[1]), Conflicts, Conflicts)]
// Cells read stay as read.
#[case::cells_and_a_rewrite(assert_c_unchanged_in(0), replace_c(0), Conflicts, Lands)]
#[case::cells_and_an_overlay(assert_c_unchanged_in(0), overlay(0), Conflicts, Lands)]
#[case::cells_and_a_delete(assert_c_unchanged_in(0), delete_rows(0), Lands, Lands)]
#[case::cells_elsewhere(assert_c_unchanged_in(1), replace_c(0), Lands, Lands)]
// Indices: segments may not describe one fragment twice.
#[case::segments_over_one_fragment(index_over(&[0]), index_over(&[0]), Conflicts, Conflicts)]
#[case::segments_over_different_fragments(index_over(&[0]), index_over(&[1]), Lands, Lands)]
#[case::a_segment_and_a_rewrite(index_over(&[0]), replace_c(0), Lands, Lands)]
// String maps and bases, key by key.
#[case::one_config_key(config("k"), config("k"), Conflicts, Conflicts)]
#[case::different_config_keys(config("k"), config("j"), Lands, Lands)]
#[case::one_base_name(add_base("b", "s3://x"), add_base("b", "s3://y"), Conflicts, Conflicts)]
#[case::different_bases(add_base("b", "s3://x"), add_base("d", "s3://y"), Lands, Lands)]
// A reset that landed first leaves nothing of what was read; one landing
// second wins.
#[case::a_reset_and_an_append(reset(), append(), Lands, Conflicts)]
#[case::a_reset_and_a_rewrite(reset(), replace_c(0), Lands, Conflicts)]
// Named operations: post-images are read whole.
#[case::a_legacy_delete_and_a_delete(legacy_delete(), delete_rows(0), Conflicts, Conflicts)]
#[case::a_legacy_delete_and_a_rewrite(legacy_delete(), replace_c(0), Conflicts, Lands)]
#[case::a_legacy_update_and_a_rename(
    legacy_update(vec![restated_with_deletions()], vec![]),
    alter_c(|alter| alter.with_name("d")),
    Lands,
    Lands
)]
#[case::a_legacy_moving_update_and_a_rewrite(
    legacy_update(vec![restated_with_deletions()], vec![table().fragments[0].clone()]),
    replace_c(0),
    Conflicts,
    Conflicts
)]
// A compaction carries the indices over the rows it moves through a remap
// computed when it read; a segment over them since is left stale.
#[case::a_compaction_and_a_segment_over_it(legacy_rewrite(0), index_over(&[0]), Conflicts, Lands)]
#[case::a_compaction_and_a_segment_elsewhere(legacy_rewrite(0), index_over(&[1]), Lands, Lands)]
#[case::a_compaction_and_a_rewrite_of_it(legacy_rewrite(0), replace_c(0), Conflicts, Conflicts)]
#[case::a_restore_and_a_schema_metadata_edit(
    Operation::Restore { version: 1 },
    schema_metadata("k"),
    Conflicts,
    Conflicts
)]
#[case::a_restore_and_a_table_metadata_edit(
    Operation::Restore { version: 1 },
    table_metadata("k"),
    Conflicts,
    Conflicts
)]
// A row-moving update re-creates the rows it moves elsewhere and only adds
// deletions here, so the cells read show no change.
#[case::cells_and_a_moving_update(assert_c_unchanged_in(0), moving_update(0), Conflicts, Lands)]
#[case::an_opaque_operation(Operation::UpdateMemWalState { compacted_sstables: vec![], dir_id: String::new() }, config("k"), Conflicts, Conflicts)]
// Data, field by field and fragment by fragment.
#[case::different_fields_in_one_fragment(replace(A, 0), replace_c(0), Lands, Lands)]
#[case::a_removal_and_a_rewrite_elsewhere(remove(0), replace_c(1), Lands, Lands)]
#[case::two_removals(remove(0), remove(0), Conflicts, Conflicts)]
#[case::an_added_column_and_a_rewrite(add_column(true), replace_c(0), Lands, Lands)]
#[case::an_added_column_and_a_delete(add_column(true), delete_rows(0), Lands, Lands)]
#[case::a_drop_and_a_rewrite_of_another_field(drop_c(), replace(A, 0), Lands, Lands)]
#[case::a_drop_and_a_delete(drop_c(), delete_rows(0), Lands, Lands)]
#[case::a_drop_and_an_append_without_it(drop_c(), append_of(&[A]), Lands, Lands)]
#[case::a_drop_and_an_append_of_a_child(drop_s(), append_of(&[S + 1]), Lands, Conflicts)]
#[case::a_struct_drop_and_a_field_added_elsewhere(
    drop_s(),
    set(vec![Action::AddField(AddField::new(0, added_field("y", true)))]),
    Lands,
    Lands
)]
// An overlay commutes with deletions; its field must still be there, in a
// fragment the transaction mints too.
#[case::an_overlay_and_a_delete(overlay(0), delete_rows(0), Lands, Lands)]
#[case::an_overlay_and_a_rewrite_of_another_column(overlay(0), replace(A, 0), Lands, Lands)]
#[case::an_overlay_materialized_and_an_overlay(materialize_c(0), overlay(0), Conflicts, Conflicts)]
#[case::an_overlay_materialized_and_a_rewrite_of_another_column(
    materialize_c(0),
    replace(A, 0),
    Lands,
    Lands
)]
#[case::a_partly_tombstoned_overlay_and_a_rewrite(
    partly_tombstoned_overlay(0),
    replace_c(0),
    Conflicts,
    Conflicts
)]
#[case::a_partly_tombstoned_overlay_and_a_rewrite_of_another_column(
    partly_tombstoned_overlay(0),
    replace(A, 0),
    Lands,
    Lands
)]
#[case::an_overlay_in_a_new_fragment_and_a_drop(
    overlay_in_a_new_fragment(),
    drop_c(),
    Conflicts,
    Lands
)]
#[case::an_overlay_in_a_new_fragment_and_a_cast(
    overlay_in_a_new_fragment(),
    action_cast_c(),
    Conflicts,
    Conflicts
)]
#[case::an_overlay_in_a_new_fragment_and_a_delete(
    overlay_in_a_new_fragment(),
    delete_rows(0),
    Lands,
    Lands
)]
#[case::a_tightening_and_an_overlay(
    alter_c(|alter| alter.with_nullable(false)),
    overlay(0),
    Conflicts,
    Conflicts
)]
#[case::a_tightening_and_an_overlay_in_a_new_fragment(
    alter_c(|alter| alter.with_nullable(false)),
    overlay_in_a_new_fragment(),
    Conflicts,
    Conflicts
)]
// Schema changes against values: a cast holds while no values were written
// for the field it replaces, and a tightening while no nulls can have been.
#[case::a_cast_and_an_append(action_cast_c(), append(), Conflicts, Conflicts)]
#[case::a_cast_and_a_rewrite(action_cast_c(), replace_c(0), Conflicts, Conflicts)]
#[case::a_cast_and_an_append_without_it(action_cast_c(), append_of(&[A]), Conflicts, Lands)]
#[case::a_tightening_and_a_reencode(
    alter_c(|alter| alter.with_nullable(false)),
    reencode_c(0),
    Lands,
    Lands
)]
#[case::a_tightening_and_a_config_update(
    alter_c(|alter| alter.with_nullable(false)),
    config("k"),
    Lands,
    Lands
)]
#[case::a_tightening_and_an_append_without_it(
    alter_c(|alter| alter.with_nullable(false)),
    append_of(&[A]),
    Conflicts,
    Conflicts
)]
#[case::a_relaxation_and_an_append(alter_c(|alter| alter.with_nullable(true)), append(), Lands, Lands)]
#[case::a_rename_and_a_tightening(
    alter_c(|alter| alter.with_name("d")),
    alter_c(|alter| alter.with_nullable(false)),
    Conflicts,
    Conflicts
)]
#[case::alterations_of_different_fields(
    alter_c(|alter| alter.with_name("d")),
    set(vec![Action::AlterField(AlterField::new(FieldRef::Committed(A)).with_name("b"))]),
    Lands,
    Lands
)]
// Keys and cells.
#[case::an_assertion_and_a_rewrite_of_another_column(merge_insert(&[1]), replace_c(0), Lands, Lands)]
#[case::an_assertion_and_a_removal(merge_insert(&[1]), remove(1), Lands, Lands)]
#[case::two_columns_swapped(set_from(A, C), set_from(C, A), Conflicts, Conflicts)]
#[case::cells_and_a_removal(assert_c_unchanged_in(0), remove(0), Conflicts, Lands)]
#[case::cells_and_a_drop(assert_c_unchanged_in(0), drop_c(), Conflicts, Lands)]
#[case::cells_and_a_rewrite_of_another_field(assert_c_unchanged_in(0), replace(A, 0), Lands, Lands)]
// Indices: a segment landing after a rewrite of what it indexes would describe
// values it never saw; landing first, the rewrite prunes it as it applies.
// Removals and overlays leave a trace a reader can act on.
#[case::a_segment_and_a_rewrite_of_its_column(index_over(&[0]), replace(A, 0), Conflicts, Lands)]
#[case::a_segment_and_a_reencode_of_its_column(index_over(&[0]), reencode(A, 0), Conflicts, Lands)]
#[case::a_segment_and_a_rewrite_of_a_column_it_carries(
    index_of(segment_parts("idx", &[A], &[C], &[0])),
    replace_c(0),
    Conflicts,
    Lands
)]
#[case::a_segment_and_a_rewrite_beside_a_removal(
    index_over(&[0]),
    set(vec![
        Action::RemoveFragment(RemoveFragment::new(Ref::Committed(1))),
        Action::TombstoneFieldData(TombstoneFieldData::new(Ref::Committed(0), vec![FieldRef::Committed(A)])),
    ]),
    Conflicts,
    Lands
)]
#[case::a_segment_over_a_struct_and_a_write_of_its_child(
    index_of(segment_parts("idx", &[S], &[], &[1])),
    write_into(1, S + 1),
    Conflicts,
    Lands
)]
#[case::a_segment_and_a_drop_of_its_column(
    index_over(&[0]),
    set(vec![Action::DropField(DropField::new(FieldRef::Committed(A)))]),
    Conflicts,
    Lands
)]
#[case::a_segment_over_a_child_and_a_struct_drop(
    index_of(segment_parts("idx", &[S + 1], &[], &[0])),
    drop_s(),
    Conflicts,
    Lands
)]
#[case::a_segment_and_a_removal_of_its_fragment(index_over(&[0]), remove(0), Lands, Lands)]
#[case::a_segment_and_an_overlay_of_its_column(
    index_of(segment_parts("idx", &[C], &[], &[0])),
    overlay(0),
    Lands,
    Lands
)]
#[case::segments_of_different_indices(
    index_over(&[0]),
    index_of(segment_parts("other", &[A], &[], &[0])),
    Lands,
    Lands
)]
// String maps, field metadata, and bases.
#[case::a_map_replaced_and_an_edit(replace_config("a"), config("k"), Conflicts, Lands)]
#[case::a_map_replaced_and_another_map_edited(
    replace_config("a"),
    table_metadata("k"),
    Lands,
    Lands
)]
#[case::a_field_metadata_edit_and_a_drop(field_metadata(C), drop_c(), Conflicts, Lands)]
#[case::a_field_metadata_edit_and_a_drop_of_another_field(
    field_metadata(C),
    set(vec![Action::DropField(DropField::new(FieldRef::Committed(A)))]),
    Lands,
    Lands
)]
#[case::one_base_location(add_base("b", "s3://x"), add_base("d", "s3://x"), Conflicts, Conflicts)]
#[case::unnamed_bases_at_different_locations(
    add_unnamed_base("s3://x"),
    add_unnamed_base("s3://y"),
    Lands,
    Lands
)]
#[case::unnamed_bases_at_one_location(
    add_unnamed_base("s3://x"),
    add_unnamed_base("s3://x"),
    Conflicts,
    Conflicts
)]
// A reset preempts even what claims nothing.
#[case::a_reset_and_rows_moved_in(reset(), rows_moved_in(), Lands, Conflicts)]
#[case::a_reset_and_a_reservation(
    reset(),
    set(vec![Action::ReserveFragmentIds(ReserveFragmentIds::new(3))]),
    Lands,
    Conflicts
)]
#[case::two_resets(reset(), reset(), Conflicts, Conflicts)]
fn test_pairs(
    #[case] ours: Operation,
    #[case] theirs: Operation,
    #[case] ours_over_theirs: Outcome,
    #[case] theirs_over_ours: Outcome,
) {
    // A MemWAL state update cannot be built without a MemWAL.
    if !matches!(ours, Operation::UpdateMemWalState { .. }) {
        assert_eq!(
            outcome(&theirs, &ours),
            theirs_over_ours,
            "theirs over ours"
        );
    }
    assert_eq!(
        outcome(&ours, &theirs),
        ours_over_theirs,
        "ours over theirs"
    );
}

/// A legacy V1 file has no column indices: its reader places a field's
/// columns by the field's position in the list, so leaving a dropped field
/// out would shift every field after it onto the wrong columns. Such a file
/// requires the fields it backs instead of being adapted to a drop.
#[test]
fn test_a_new_v1_file_conflicts_with_a_drop_of_a_field_it_backs() {
    let v1_append = set(vec![
        new_fragment(true),
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Local(0),
                DataFile::new_unstarted("data/x.lance", ConcreteFileVersion::V1),
                vec![FieldRef::Committed(A), FieldRef::Committed(C)],
                None,
                true,
            )
            .unwrap(),
        ),
    ]);

    assert_eq!(outcome(&v1_append, &drop_c()), Conflicts);
}

/// A file's path is relative to its base, so moving a column to another base
/// with the same path and column replaces its values.
#[test]
fn test_a_column_replacement_over_a_replacement_moving_it_to_another_base() {
    let mut read = table();
    for id in [1, 2] {
        read.base_paths.insert(
            id,
            crate::format::BasePath::new(id, format!("s3://example/base{id}"), None, false),
        );
    }
    let mut fragments = read.fragments.as_ref().clone();
    fragments[0].files[1].base_id = Some(1);
    read.fragments = Arc::new(fragments);
    let mut moved = read.fragments[0].files[1].clone();
    moved.base_id = Some(2);
    let replacement = Operation::DataReplacement {
        replacements: vec![crate::transaction::DataReplacementGroup(0, moved)],
        data_change: true,
    };

    assert_eq!(outcome_at(&read, &replace_c(0), &replacement), Conflicts);
}

/// Row ids a named operation's new fragments arrive carrying are claimed as
/// an `AddFragment`'s are: two writers that spend the same reserved ids would
/// leave one row id live in two fragments. Ids that do not decode claim
/// everything.
#[rstest]
#[case::the_same_ids(10..15, Conflicts)]
#[case::other_ids(15..20, Lands)]
fn test_named_new_fragments_claim_their_row_ids(
    #[values(false, true)] named_update: bool,
    #[case] action_ids: std::ops::Range<u64>,
    #[case] expected: Outcome,
) {
    use crate::format::RowIdMeta;
    use crate::rowids::write_row_ids;

    let inline = |ids: std::ops::Range<u64>| RowIdMeta::Inline(write_row_ids(&ids.into()).into());
    let mut actions = append_actions();
    actions[0] = Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
        id: Ref::Local(0),
        physical_rows: 5,
        row_id_meta: Some(inline(action_ids)),
        last_updated_at_version_meta: None,
        created_at_version_meta: None,
        data_change: true,
    }));
    let action_set = set(actions);
    let named = |row_id_meta: RowIdMeta| {
        let mut fragment = table().fragments[0].clone();
        fragment.id = 7;
        fragment.row_id_meta = Some(row_id_meta);
        if named_update {
            legacy_update(vec![], vec![fragment])
        } else {
            Operation::Append {
                fragments: vec![fragment],
            }
        }
    };
    let window_outcome = |ours: &Operation, theirs: &Operation| {
        // Applying stable row ids needs a table that keeps them, so the
        // window is the read manifest itself: only the footprints decide.
        let read = table();
        let pending = Footprint::of(&Transaction::new(read.version, ours.clone(), None), &read);
        let committed = Footprint::of(&Transaction::new(read.version, theirs.clone(), None), &read);
        match pending.check(&Window::new(&read, &read, vec![committed])) {
            Verdict::Compatible => Lands,
            Verdict::Conflict(_) => Conflicts,
        }
    };

    let reserved = named(inline(10..15));
    assert_eq!(window_outcome(&action_set, &reserved), expected);
    assert_eq!(window_outcome(&reserved, &action_set), expected);

    let undecodable = named(RowIdMeta::Inline(vec![0xff; 3].into()));
    assert_eq!(window_outcome(&action_set, &undecodable), Conflicts);
    assert_eq!(window_outcome(&undecodable, &action_set), Conflicts);
}

/// A fragment id reserved once and claimed by two writers collides; row ids
/// supplied twice collide whatever fragments carry them.
#[test]
fn test_reserved_ids_are_claimed_once() {
    let claim = |id: u64| {
        set(vec![Action::AddFragment(AddFragment::new_unchecked(
            AddFragmentParts {
                id: Ref::Committed(id),
                physical_rows: 3,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            },
        ))])
    };
    let mut read = table();
    read.max_fragment_id = Some(5);
    assert_eq!(outcome_at(&read, &claim(4), &claim(4)), Conflicts);
    assert_eq!(outcome_at(&read, &claim(4), &claim(5)), Lands);

    let supplying = |ids: &RoaringTreemap| {
        let mut footprint = Footprint::default();
        footprint.supply_row_ids(ids);
        footprint
    };
    let window = Window::new(
        &read,
        &read,
        vec![supplying(&RoaringTreemap::from_iter([1u64, 2]))],
    );
    assert!(matches!(
        supplying(&RoaringTreemap::from_iter([2u64, 3])).check(&window),
        Verdict::Conflict(_)
    ));
    assert!(matches!(
        supplying(&RoaringTreemap::from_iter([3u64])).check(&window),
        Verdict::Compatible
    ));
}

/// Deletions cleared since, which only a named operation restating the
/// fragment can do, conflict with deletions added.
#[test]
fn test_cleared_deletions_do_not_merge() {
    let mut read = table();
    let mut fragments = read.fragments.as_ref().clone();
    fragments[0].deletion_file = Some(deletion_file(1));
    read.fragments = Arc::new(fragments);
    let mut cleared = read.fragments[0].clone();
    cleared.deletion_file = None;
    let clear = Operation::Delete {
        updated_fragments: vec![cleared],
        deleted_fragment_ids: vec![],
        predicate: String::new(),
    };
    let add = set(vec![Action::SetDeletionFile(
        SetDeletionFile::try_new_unchecked(0, deletion_file(2), None, true).unwrap(),
    )]);

    assert_eq!(outcome_at(&read, &add, &clear), Conflicts);
}

/// The window is checked as one change of state: a pending transaction is
/// held to what the last manifest differs in from the one it read, plus every
/// committed transaction's facts, whatever order they landed in.
#[test]
fn test_a_window_of_several_commits() {
    let read = table();
    let commit = |before: &Manifest, operation: &Operation| {
        let transaction = Transaction::new(before.version, operation.clone(), None);
        let (after, _) = transaction
            .build_manifest(Some(before), vec![], "tx.txn", &default_build_config())
            .unwrap();
        (Footprint::of(&transaction, before), after)
    };
    let (rewrite, after_rewrite) = commit(&read, &replace_c(1));
    let (appended, latest) = commit(&after_rewrite, &append());
    let window = Window::new(&read, &latest, vec![rewrite, appended]);
    let pending = |operation: &Operation| {
        Footprint::of(
            &Transaction::new(read.version, operation.clone(), None),
            &read,
        )
        .check(&window)
    };

    assert!(matches!(pending(&replace_c(0)), Verdict::Compatible));
    assert!(matches!(pending(&replace_c(1)), Verdict::Conflict(_)));
    // The append is a fact of the window, not of the latest manifest alone.
    assert!(matches!(
        pending(&alter_c(|alter| alter.with_nullable(false))),
        Verdict::Conflict(_)
    ));
}

/// What only the committed transactions can say, with no change of state to
/// look at.
#[rstest]
#[case::one_segment_removed_twice(
    vec![remove_segment(SEGMENT)],
    vec![remove_segment(SEGMENT)],
    true
)]
#[case::different_segments_removed(
    vec![remove_segment(SEGMENT)],
    vec![remove_segment(Uuid::nil())],
    false
)]
#[case::an_index_dropped_and_built(vec![drop_index("idx")], index_actions(&[1]), true)]
#[case::an_index_dropped_and_another_built(vec![drop_index("other")], index_actions(&[1]), false)]
#[case::memwal_progress_recorded_twice(
    vec![compacted_sstables()],
    vec![compacted_sstables()],
    true
)]
// A compaction's rows were already in the table.
#[case::an_assertion_and_a_compaction(
    vec![key_assertion(&[1])],
    vec![new_fragment(false)],
    false
)]
#[case::assertions_over_different_key_columns(
    vec![key_assertion(&[1])],
    {
        let mut actions = append_actions();
        actions.push(Action::AssertUniqueKeys(
            AssertUniqueKeys::try_new_unchecked(
                vec![FieldRef::Committed(C)],
                KeyExistenceFilter {
                    field_ids: vec![C],
                    filter: FilterType::ExactSet([2].into_iter().collect()),
                },
            )
            .unwrap(),
        ));
        actions
    },
    true
)]
#[case::one_segment_adjusted_twice(
    vec![adjust(SEGMENT, "idx", &[1], &[])],
    vec![adjust(SEGMENT, "idx", &[], &[0])],
    true
)]
#[case::different_segments_adjusted(
    vec![adjust(SEGMENT, "idx", &[1], &[])],
    vec![adjust(Uuid::nil(), "idx", &[], &[0])],
    false
)]
#[case::a_widening_onto_a_fragment_a_segment_covers(
    vec![adjust(SEGMENT, "idx", &[0], &[])],
    index_actions(&[0]),
    true
)]
#[case::a_widening_onto_another_fragment(
    vec![adjust(SEGMENT, "idx", &[0], &[])],
    index_actions(&[1]),
    false
)]
#[case::a_widening_of_another_index(
    vec![adjust(SEGMENT, "other", &[0], &[])],
    index_actions(&[0]),
    false
)]
// Coverage given up cannot collide with coverage taken on.
#[case::coverage_given_up(vec![adjust(SEGMENT, "idx", &[], &[0])], index_actions(&[0]), false)]
#[case::a_segment_removed_and_its_index_built(vec![remove_segment(SEGMENT)], index_actions(&[0]), false)]
#[case::an_index_built_and_a_segment_of_it_removed(index_actions(&[0]), vec![remove_segment(SEGMENT)], false)]
#[case::an_index_built_and_dropped(index_actions(&[1]), vec![drop_index("idx")], true)]
#[case::an_index_dropped_twice(vec![drop_index("idx")], vec![drop_index("idx")], true)]
#[case::an_index_dropped_and_an_append(vec![drop_index("idx")], append_actions(), false)]
// Segment-level maintenance beside a writer indexing what it appends.
#[case::an_optimize_and_an_append_indexed(optimize(), append_indexed(), false)]
#[case::an_append_indexed_and_an_optimize(append_indexed(), optimize(), false)]
// The MemWAL index is one entry: recording progress rewrites all of it.
#[case::memwal_progress_and_a_memwal_rebuild(
    vec![compacted_sstables()],
    vec![memwal_segment()],
    true
)]
#[case::a_memwal_rebuild_and_memwal_progress(
    vec![memwal_segment()],
    vec![compacted_sstables()],
    true
)]
#[case::memwal_progress_and_an_ordinary_index(vec![compacted_sstables()], index_actions(&[0]), false)]
#[case::an_ordinary_index_and_memwal_progress(index_actions(&[0]), vec![compacted_sstables()], false)]
// A key column the transaction mints is no concurrent writer's to insert into.
#[case::assertions_over_minted_key_columns(
    keyed_insert(&[(FieldRef::Local(3), &[1, 2])]),
    keyed_insert(&[(FieldRef::Local(3), &[2, 3])]),
    false
)]
#[case::an_assertion_over_a_minted_key_column_and_an_append(
    keyed_insert(&[(FieldRef::Local(3), &[1])]),
    append_actions(),
    false
)]
// Each assertion is matched by its key columns.
#[case::assertions_disjoint_on_each_key(
    keyed_insert(&[(FieldRef::Committed(A), &[1]), (FieldRef::Committed(C), &[5])]),
    keyed_insert(&[(FieldRef::Committed(A), &[2]), (FieldRef::Committed(C), &[6])]),
    false
)]
#[case::assertions_overlapping_on_the_second_key(
    keyed_insert(&[(FieldRef::Committed(A), &[1]), (FieldRef::Committed(C), &[5])]),
    keyed_insert(&[(FieldRef::Committed(A), &[2]), (FieldRef::Committed(C), &[5])]),
    true
)]
#[case::an_insert_silent_on_the_second_key(
    keyed_insert(&[(FieldRef::Committed(A), &[1]), (FieldRef::Committed(C), &[5])]),
    keyed_insert(&[(FieldRef::Committed(A), &[2])]),
    true
)]
#[case::an_assertion_and_an_overlay_of_its_key_column(
    keyed_insert(&[(FieldRef::Committed(A), &[1])]),
    vec![Action::AddOverlays(AddOverlays::new_unchecked(Ref::Committed(0), vec![overlay_with_field(A, 0)]))],
    true
)]
#[case::an_assertion_and_an_overlay_of_another_column(
    keyed_insert(&[(FieldRef::Committed(A), &[1])]),
    vec![Action::AddOverlays(AddOverlays::new_unchecked(Ref::Committed(0), vec![overlay_with_field(C, 0)]))],
    false
)]
#[case::assertions_with_incomparable_filters(bloom_insert(100), bloom_insert(200), true)]
fn test_facts(#[case] ours: Vec<Action>, #[case] theirs: Vec<Action>, #[case] conflicts: bool) {
    let read = table();
    let committed = |actions: Vec<Action>| match set(actions) {
        Operation::CompositeOperation(set) => Footprint::from(&set),
        _ => unreachable!(),
    };
    let window = Window::new(&read, &read, vec![committed(theirs)]);
    assert_eq!(
        matches!(committed(ours).check(&window), Verdict::Conflict(_)),
        conflicts
    );
}

/// A compaction carries along the indices it finds at apply through a remap
/// computed when it read. Only coverage added since is left stale; an index
/// dropped or rewritten without new coverage leaves nothing stale to carry.
#[rstest]
#[case::a_segment_over_it(index_actions(&[0]), true)]
#[case::a_segment_elsewhere(index_actions(&[1]), false)]
#[case::an_index_dropped(vec![drop_index("idx")], false)]
#[case::memwal_progress(vec![compacted_sstables()], false)]
fn test_a_compaction_over_index_writes(#[case] theirs: Vec<Action>, #[case] conflicts: bool) {
    let read = table();
    let Operation::CompositeOperation(theirs) = set(theirs) else {
        unreachable!()
    };
    let window = Window::new(&read, &read, vec![Footprint::from(&theirs)]);
    let ours = Footprint::of(
        &Transaction::new(read.version, legacy_rewrite(0), None),
        &read,
    );
    assert_eq!(
        matches!(ours.check(&window), Verdict::Conflict(_)),
        conflicts
    );
}

/// A struct made non-nullable holds only if no values were written for it or,
/// through it, for a child, which may null the struct's rows. Values for an
/// unrelated field leave it valid.
#[rstest]
#[case::a_child_written(S + 1, Conflicts)]
#[case::an_unrelated_field_written(A, Lands)]
fn test_a_struct_tightening_over_a_write(#[case] written: i32, #[case] expected: Outcome) {
    let tighten_s = set(vec![Action::AlterField(
        AlterField::new(FieldRef::Committed(S)).with_nullable(false),
    )]);
    let write = set(vec![Action::AddDataFile(
        AddDataFile::try_new_unchecked(
            Ref::Committed(1),
            unbound(field_file("written-1", written)),
            vec![FieldRef::Committed(written)],
            None,
            true,
        )
        .unwrap(),
    )]);
    assert_eq!(by_footprints(&tighten_s, &write), expected);
}

/// What it takes for `ours` to land over `theirs`, both read at [`table`],
/// from their footprints alone: for a committed operation that cannot be
/// applied to [`table`] as it stands.
fn by_footprints(ours: &Operation, theirs: &Operation) -> Outcome {
    let read = table();
    let pending = Footprint::of(&Transaction::new(read.version, ours.clone(), None), &read);
    let committed = Footprint::of(&Transaction::new(read.version, theirs.clone(), None), &read);
    match pending.check(&Window::new(&read, &read, vec![committed])) {
        Verdict::Compatible => Lands,
        Verdict::Conflict(_) => Conflicts,
    }
}

fn action_cast_c() -> Operation {
    set(vec![
        Action::DropField(DropField::new(FieldRef::Committed(C))),
        Action::AddField(AddField::new(0, added_field("c", true))),
    ])
}

fn compacted_sstables() -> Action {
    Action::UpdateCompactedSsTables(
        UpdateCompactedSsTables::try_new_unchecked(vec![
            crate::system_index::mem_wal::CompactedSsTable::new(Uuid::nil(), 1),
        ])
        .unwrap(),
    )
}

const SEGMENT: Uuid = Uuid::from_u128(7);

fn remove_segment(uuid: Uuid) -> Action {
    Action::RemoveIndexSegment(RemoveIndexSegment::try_new(uuid, "idx").unwrap())
}

fn drop_index(name: &str) -> Action {
    Action::DropIndex(DropIndex::try_new(name).unwrap())
}

fn index_actions(fragments: &[u64]) -> Vec<Action> {
    match index_over(fragments) {
        Operation::CompositeOperation(set) => set.iter_actions().cloned().collect(),
        _ => unreachable!(),
    }
}

/// Row ids are claimed as exact sets, not ranges: two supplies whose ids
/// interleave do not collide.
#[test]
fn test_interleaved_row_ids_do_not_collide() {
    let read = table();
    let supplying = |ids: &[u64]| {
        let mut footprint = Footprint::default();
        footprint.supply_row_ids(&ids.iter().copied().collect());
        footprint
    };
    let window = Window::new(&read, &read, vec![supplying(&[10, 30])]);
    assert!(matches!(
        supplying(&[20, 40]).check(&window),
        Verdict::Compatible
    ));
}

/// A widening holds only while the columns it covers are the ones its remap
/// was planned against, so it cannot land after a rewrite of a keyed or
/// carried column, nor over a fragment removed since; an overlay is masked by
/// the read path instead. Landing first, a widening leaves the manifest's
/// fragments as they were.
#[rstest]
#[case::a_keyed_column_rewritten(replace(A, 0), Conflicts)]
#[case::a_carried_column_rewritten(replace_c(0), Conflicts)]
#[case::an_unrelated_column_rewritten(replace(A, 1), Lands)]
#[case::the_fragment_removed(remove(0), Conflicts)]
#[case::an_overlay(overlay(0), Lands)]
fn test_a_widening_over_a_change_to_what_it_covers(
    #[case] theirs: Operation,
    #[case] expected: Outcome,
) {
    let widen = set(vec![widening(&[A], &[C], 0)]);
    assert_eq!(outcome(&widen, &theirs), expected);
    assert_eq!(by_footprints(&theirs, &widen), Lands);
}

/// A child made non-nullable holds only if no values were written for it or
/// the struct it is in, which may null it.
#[test]
fn test_a_child_tightening_over_a_write_of_its_struct() {
    let tighten_x = set(vec![Action::AlterField(
        AlterField::new(FieldRef::Committed(S + 1)).with_nullable(false),
    )]);
    assert_eq!(by_footprints(&tighten_x, &write_into(1, S)), Conflicts);
}

/// A fragment compacted by two writers is moved twice; compactions of
/// different fragments land.
#[test]
fn test_two_compactions() {
    assert_eq!(outcome(&compaction(0), &compaction(0)), Conflicts);
    assert_eq!(outcome(&compaction(0), &compaction(1)), Lands);
}

/// A refresh restamps rows that must still be there; a removal discards
/// restamped row versions along with the rows.
#[test]
fn test_a_refresh_and_a_removal() {
    let refresh = set(vec![crate::transaction::action::test_support::refresh(&[
        0,
    ])]);
    assert_eq!(outcome(&refresh, &remove(0)), Conflicts);
    assert_eq!(by_footprints(&remove(0), &refresh), Lands);
}

/// Fragment 0's overlay over `a` and `c` outlives a drop of `c`, tombstoned;
/// a compaction moving the fragment out carries it as it was, so the two
/// land.
#[test]
fn test_a_compaction_over_a_drop_of_an_overlaid_field() {
    let mut read = table();
    let mut fragments = read.fragments.as_ref().clone();
    let mut overlay = overlay_with_field(A, 1);
    overlay.data_file.fields = vec![A, C].into();
    overlay.data_file.column_indices = vec![0, 1].into();
    fragments[0].overlays = vec![overlay];
    read.fragments = Arc::new(fragments);

    let move_out = set(vec![Action::RemoveFragment(RemoveFragment::new_unchecked(
        Ref::Committed(0),
        false,
    ))]);
    assert_eq!(outcome_at(&read, &move_out, &drop_c()), Lands);
}

fn schema_metadata(key: &str) -> Operation {
    set(vec![Action::ConfigUpdate(
        ConfigUpdate::default().with_schema_metadata(UpdateMap {
            update_entries: vec![UpdateMapEntry::from((key, Some("v")))],
            replace: false,
        }),
    )])
}

fn table_metadata(key: &str) -> Operation {
    set(vec![Action::ConfigUpdate(
        ConfigUpdate::default().with_table_metadata(UpdateMap {
            update_entries: vec![UpdateMapEntry::from((key, Some("v")))],
            replace: false,
        }),
    )])
}

/// The dataset config replaced by one holding only `key`.
fn replace_config(key: &str) -> Operation {
    set(vec![Action::ConfigUpdate(
        ConfigUpdate::default().with_config(UpdateMap {
            update_entries: vec![UpdateMapEntry::from((key, Some("v")))],
            replace: true,
        }),
    )])
}

fn field_metadata(field: i32) -> Operation {
    set(vec![Action::ConfigUpdate(
        ConfigUpdate::default().with_field_metadata(
            FieldRef::Committed(field),
            UpdateMap {
                update_entries: vec![UpdateMapEntry::from(("comment", Some("v")))],
                replace: false,
            },
        ),
    )])
}

fn add_unnamed_base(path: &str) -> Operation {
    set(vec![Action::AddBase(AddBase::new(
        0,
        BasePath::new(0, path.into(), None, false),
    ))])
}

/// A new fragment with one file binding `fields`.
fn append_of(fields: &[i32]) -> Operation {
    set(vec![
        new_fragment(true),
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Local(0),
                DataFile {
                    column_indices: (0..).take(fields.len()).collect(),
                    ..unbound(field_file("new", A))
                },
                fields.iter().copied().map(FieldRef::Committed).collect(),
                None,
                true,
            )
            .unwrap(),
        ),
    ])
}

/// New values for `field`, unbacked in `fragment`, written there.
fn write_into(fragment: u64, field: i32) -> Operation {
    set(vec![Action::AddDataFile(
        AddDataFile::try_new_unchecked(
            Ref::Committed(fragment),
            unbound(field_file(&format!("written-{fragment}"), field)),
            vec![FieldRef::Committed(field)],
            None,
            true,
        )
        .unwrap(),
    )])
}

/// `field`'s data in `fragment` re-encoded, its values unchanged.
fn reencode(field: i32, fragment: u64) -> Operation {
    set(vec![
        Action::TombstoneFieldData(TombstoneFieldData::new_unchecked(
            Ref::Committed(fragment),
            vec![FieldRef::Committed(field)],
            false,
        )),
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Committed(fragment),
                unbound(field_file(&format!("{field}-reencoded-{fragment}"), field)),
                vec![FieldRef::Committed(field)],
                None,
                false,
            )
            .unwrap(),
        ),
    ])
}

/// `UPDATE t SET written = read` in fragment 0: the read column's cells are
/// asserted unchanged.
fn set_from(written: i32, read: i32) -> Operation {
    let Operation::CompositeOperation(rewrite) = replace(written, 0) else {
        unreachable!()
    };
    let mut actions: Vec<Action> = rewrite.iter_actions().cloned().collect();
    actions.push(Action::AssertCellsUnchanged(
        AssertCellsUnchanged::try_new(
            vec![FieldRef::Committed(read)],
            vec![RowSelection {
                fragment_id: 0,
                rows: SelectedRows::Full,
            }],
        )
        .unwrap(),
    ));
    set(actions)
}

/// An overlay of `c` in `fragment`, then `c` rewritten there in the same set.
fn materialize_c(fragment: u64) -> Operation {
    let Operation::CompositeOperation(rewrite) = replace_c(fragment) else {
        unreachable!()
    };
    let mut actions = vec![Action::AddOverlays(AddOverlays::new_unchecked(
        Ref::Committed(fragment),
        vec![overlay_with_field(C, 0)],
    ))];
    actions.extend(rewrite.iter_actions().cloned());
    set(actions)
}

/// An overlay over `c` and a tombstoned column, which holds nothing a reader
/// sees.
fn partly_tombstoned_overlay(fragment: u64) -> Operation {
    let mut overlay = overlay_with_field(C, 0);
    overlay.data_file.fields = vec![-1, C].into();
    overlay.data_file.column_indices = vec![0, 1].into();
    set(vec![Action::AddOverlays(AddOverlays::new_unchecked(
        Ref::Committed(fragment),
        vec![overlay],
    ))])
}

fn overlay_in_a_new_fragment() -> Operation {
    set(vec![
        new_fragment(true),
        Action::AddOverlays(AddOverlays::new_unchecked(
            Ref::Local(0),
            vec![overlay_with_field(C, 0)],
        )),
    ])
}

/// Rows already in the table, moved into a new fragment: what a compaction
/// adds.
fn rows_moved_in() -> Operation {
    set(vec![
        new_fragment(false),
        Action::AddDataFile(
            AddDataFile::try_new_unchecked(
                Ref::Local(0),
                DataFile {
                    column_indices: vec![0, 1].into(),
                    ..unbound(field_file("moved", A))
                },
                vec![FieldRef::Committed(A), FieldRef::Committed(C)],
                None,
                false,
            )
            .unwrap(),
        ),
    ])
}

/// `fragment`'s rows moved into a new fragment.
fn compaction(fragment: u64) -> Operation {
    let Operation::CompositeOperation(moved_in) = rows_moved_in() else {
        unreachable!()
    };
    let mut actions = vec![Action::RemoveFragment(RemoveFragment::new_unchecked(
        Ref::Committed(fragment),
        false,
    ))];
    actions.extend(moved_in.iter_actions().cloned());
    set(actions)
}

fn segment_parts(
    name: &str,
    fields: &[i32],
    covering_fields: &[i32],
    fragments: &[u64],
) -> AddIndexSegmentParts {
    AddIndexSegmentParts {
        uuid: Uuid::new_v4(),
        name: name.into(),
        fields: fields.iter().copied().map(FieldRef::Committed).collect(),
        covering_fields: covering_fields
            .iter()
            .copied()
            .map(FieldRef::Committed)
            .collect(),
        index_details: None,
        index_version: 1,
        covered_fragments: Some(fragments.iter().copied().map(Ref::Committed).collect()),
        files: vec![],
        base: None,
        created_at: None,
        dataset_version: None,
        data_change: false,
    }
}

fn index_of(parts: AddIndexSegmentParts) -> Operation {
    set(vec![Action::AddIndexSegment(
        AddIndexSegment::try_new_unchecked(parts).unwrap(),
    )])
}

fn widening(fields: &[i32], covering_fields: &[i32], fragment: u64) -> Action {
    Action::AdjustIndexCoverage(
        AdjustIndexCoverage::try_new_unchecked(AdjustIndexCoverageParts {
            uuid: SEGMENT,
            name: "idx".into(),
            add_fragments: vec![Ref::Committed(fragment)],
            remove_fragments: vec![],
            fields: fields.iter().copied().map(FieldRef::Committed).collect(),
            covering_fields: covering_fields
                .iter()
                .copied()
                .map(FieldRef::Committed)
                .collect(),
        })
        .unwrap(),
    )
}

fn adjust(uuid: Uuid, name: &str, add: &[u64], remove: &[u64]) -> Action {
    Action::AdjustIndexCoverage(
        AdjustIndexCoverage::try_new_unchecked(AdjustIndexCoverageParts {
            uuid,
            name: name.into(),
            add_fragments: add.iter().copied().map(Ref::Committed).collect(),
            remove_fragments: remove.to_vec(),
            fields: vec![FieldRef::Committed(A)],
            covering_fields: vec![],
        })
        .unwrap(),
    )
}

/// A segment of `idx` replaced by one over fragment 0.
fn optimize() -> Vec<Action> {
    let mut actions = vec![remove_segment(SEGMENT)];
    actions.extend(index_actions(&[0]));
    actions
}

/// An append, and a segment of `idx` over what it appended.
fn append_indexed() -> Vec<Action> {
    let mut actions = append_actions();
    actions.push(Action::AddIndexSegment(
        AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
            covered_fragments: Some(vec![Ref::Local(0)]),
            ..segment_parts("idx", &[A], &[], &[])
        })
        .unwrap(),
    ));
    actions
}

fn memwal_segment() -> Action {
    Action::AddIndexSegment(
        AddIndexSegment::try_new_unchecked(AddIndexSegmentParts {
            covered_fragments: None,
            data_change: true,
            ..segment_parts(MEM_WAL_INDEX_NAME, &[A], &[], &[])
        })
        .unwrap(),
    )
}

/// An append asserting, for each key column, the keys in its hashes.
fn keyed_insert(keys: &[(FieldRef, &[u64])]) -> Vec<Action> {
    let mut actions = append_actions();
    for (field, hashes) in keys {
        actions.push(Action::AssertUniqueKeys(
            AssertUniqueKeys::try_new_unchecked(
                vec![*field],
                KeyExistenceFilter {
                    field_ids: vec![],
                    filter: FilterType::ExactSet(hashes.iter().copied().collect()),
                },
            )
            .unwrap(),
        ));
    }
    actions
}

/// An append asserting its keys over `a` with a bloom filter sized for
/// `number_of_items`.
fn bloom_insert(number_of_items: u64) -> Vec<Action> {
    let mut actions = append_actions();
    actions.push(Action::AssertUniqueKeys(
        AssertUniqueKeys::try_new_unchecked(
            vec![FieldRef::Committed(A)],
            KeyExistenceFilter {
                field_ids: Vec::new(),
                filter: FilterType::Bloom {
                    bitmap: vec![0; 32],
                    num_bits: 256,
                    number_of_items,
                    probability: 0.01,
                },
            },
        )
        .unwrap(),
    ));
    actions
}
