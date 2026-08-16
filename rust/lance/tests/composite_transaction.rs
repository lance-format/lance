// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end coverage for committing an action set against a real dataset.
//!
//! Each of these is a single commit that does work a named operation would
//! have needed several commits for: a fragment is added and then modified, a
//! field is added and then filled, all inside one version. That is what the
//! action vocabulary buys -- steps that reference each other's minted ids can
//! be squashed into one atomic manifest change.
//!
//! The commit path checks that a referenced data file exists, so these tests
//! point their actions at files the fixture dataset already wrote. The
//! resulting datasets are inspected through their manifests rather than read --
//! the files hold the wrong columns for where they end up attached.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::{CommitBuilder, InsertBuilder, WriteParams};
use lance_table::format::DataFile;
use lance_table::transaction::action::{
    Action, AddDataFile, AddField, AddFragment, Ref, TombstoneFieldData, UserAction, UserOperation,
};
use lance_table::transaction::{Operation, Transaction};

/// A two-fragment dataset, so its two data files can stand in for the files an
/// action set would otherwise have had to write.
async fn test_dataset(enable_stable_row_ids: bool) -> Dataset {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let data =
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from_iter_values(0..10))]).unwrap();

    InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            enable_stable_row_ids,
            max_rows_per_file: 5,
            ..Default::default()
        })
        .execute(vec![data])
        .await
        .unwrap()
}

fn existing_data_file(dataset: &Dataset, fragment: usize) -> DataFile {
    dataset.fragments()[fragment].files[0].clone()
}

async fn commit(dataset: Dataset, actions: Vec<Action>) -> Dataset {
    let read_version = dataset.version().version;
    CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::UserOperation(UserOperation::new(
                "composite",
                vec![UserAction::new("step", actions)],
            )),
            None,
        ))
        .await
        .unwrap()
}

#[tokio::test]
async fn test_one_commit_adds_a_fragment_and_then_modifies_it() {
    let dataset = test_dataset(false).await;
    let before = dataset.version().version;
    let first_file = existing_data_file(&dataset, 0);
    let second_file = existing_data_file(&dataset, 1);
    let second_path = second_file.path.clone();

    let dataset = commit(
        dataset,
        vec![
            Action::AddFragment(AddFragment {
                local: 0,
                physical_rows: 4,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: first_file,
                field_ids: vec![Ref::Committed(0)],
                data_change: true,
            }),
            // The same commit then replaces the data it just added, naming the
            // fragment by the token it was minted under.
            Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Local(0),
                field_ids: vec![0],
                data_change: true,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: second_file,
                field_ids: vec![Ref::Committed(0)],
                data_change: true,
            }),
        ],
    )
    .await;

    assert_eq!(dataset.version().version, before + 1);
    let fragments = dataset.fragments();
    assert_eq!(fragments.len(), 3);
    let added = fragments.last().unwrap();
    assert_eq!(added.physical_rows, Some(4));
    // The tombstoned file is gone; only the replacement survives the commit.
    let paths = added
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect::<Vec<_>>();
    assert_eq!(paths, vec![second_path]);
}

#[tokio::test]
async fn test_one_commit_adds_a_field_and_then_fills_it() {
    let dataset = test_dataset(false).await;
    let fragment_id = dataset.fragments()[0].id;
    let file = existing_data_file(&dataset, 1);
    let path = file.path.clone();

    let dataset = commit(
        dataset,
        vec![
            Action::AddField(AddField {
                local: 0,
                parent: None,
                def: lance_core::datatypes::Field::try_from(Field::new("b", DataType::Int32, true))
                    .unwrap(),
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Committed(fragment_id),
                file,
                field_ids: vec![Ref::Local(0)],
                data_change: true,
            }),
        ],
    )
    .await;

    let field = dataset.schema().field("b").expect("field b was added");
    let fragment = &dataset.fragments()[0];
    let added = fragment
        .files
        .iter()
        .find(|file| file.path == path)
        .expect("the new field's data file was attached");
    // The file points at the id the commit minted, which the caller never knew.
    assert_eq!(added.fields.as_ref(), &[field.id]);
}

#[tokio::test]
async fn test_one_commit_assigns_row_ids_to_the_fragments_it_mints() {
    let dataset = test_dataset(true).await;
    let next_row_id = dataset.manifest().next_row_id;
    assert_eq!(next_row_id, 10);

    let dataset = commit(
        dataset,
        vec![
            Action::AddFragment(AddFragment {
                local: 0,
                physical_rows: 4,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
            Action::AddFragment(AddFragment {
                local: 1,
                physical_rows: 6,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            }),
        ],
    )
    .await;

    assert_eq!(dataset.manifest().next_row_id, 20);
    for fragment in dataset.fragments().iter().skip(2) {
        assert!(
            fragment.row_id_meta.is_some(),
            "fragment {} was minted without row ids",
            fragment.id
        );
        assert!(fragment.created_at_version_meta.is_some());
    }
}

#[tokio::test]
async fn test_two_action_sets_on_disjoint_coordinates_both_commit() {
    let dataset = Arc::new(test_dataset(false).await);
    let read_version = dataset.version().version;

    let append = |local| {
        vec![Action::AddFragment(AddFragment {
            local,
            physical_rows: 4,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })]
    };

    let first = CommitBuilder::new(dataset.clone())
        .execute(Transaction::new(
            read_version,
            Operation::UserOperation(UserOperation::new(
                "first",
                vec![UserAction::new("step", append(0))],
            )),
            None,
        ))
        .await
        .unwrap();

    // The second commit still reads the original version, so it has to be
    // checked against the first. Both only mint, so neither writes anything
    // the other does.
    let second = CommitBuilder::new(dataset)
        .execute(Transaction::new(
            read_version,
            Operation::UserOperation(UserOperation::new(
                "second",
                vec![UserAction::new("step", append(0))],
            )),
            None,
        ))
        .await
        .unwrap();

    assert_eq!(second.version().version, first.version().version + 1);
    assert_eq!(second.fragments().len(), 4);
}

#[tokio::test]
async fn test_two_action_sets_writing_the_same_field_data_conflict() {
    let dataset = Arc::new(test_dataset(false).await);
    let read_version = dataset.version().version;
    let fragment_id = dataset.fragments()[0].id;

    let tombstone = || {
        Transaction::new(
            read_version,
            Operation::UserOperation(UserOperation::new(
                "tombstone",
                vec![UserAction::new(
                    "step",
                    vec![Action::TombstoneFieldData(TombstoneFieldData {
                        fragment: Ref::Committed(fragment_id),
                        field_ids: vec![0],
                        data_change: true,
                    })],
                )],
            )),
            None,
        )
    };

    CommitBuilder::new(dataset.clone())
        .execute(tombstone())
        .await
        .unwrap();

    let error = CommitBuilder::new(dataset)
        .with_max_retries(0)
        .execute(tombstone())
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("preempted"),
        "unexpected error: {error}"
    );
}
