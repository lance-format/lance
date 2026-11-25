// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::num::NonZeroU32;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::record_batch;
use arrow_array::types::{Int32Type, UInt64Type};
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{
    Error, ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION, ROW_OFFSET,
    datatypes::Schema,
};
use lance_file::version::ConcreteFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::format::{DataStorageFormat, Fragment, Manifest, RowIdMeta};
use lance_table::io::deletion::{deletion_file_path, write_deletion_file};
use lance_table::rowids::{RowIdSequence, write_row_ids};
use object_store::path::Path;
use rstest::rstest;
use tempfile::TempDir;

use super::{
    MissingRowPolicy, TakeOptions, TakeProjection, TakeSelection, row_offsets_to_row_addresses,
};
use crate::Table;
use crate::test_util::{publish_manifest, write_file};

async fn number_table(
    row_id_starts: Option<&[u64]>,
    fragment_deletions: &[&[u32]],
) -> (TempDir, Table) {
    const ROWS_PER_FRAGMENT: usize = 4;
    let dir = tempfile::tempdir().unwrap();
    let base = Path::from_absolute_path(dir.path()).unwrap();
    let store = ObjectStore::local();
    let version = ConcreteFileVersion::V2_1;
    let mut fragments = Vec::with_capacity(fragment_deletions.len());
    for slot in 0..fragment_deletions.len() {
        let rows = (slot * ROWS_PER_FRAGMENT..(slot + 1) * ROWS_PER_FRAGMENT)
            .map(|value| value as i32)
            .collect::<Vec<_>>();
        let batch = record_batch!(("i", Int32, rows)).unwrap();
        let data_file = write_file(&store, &base, &format!("{slot}.lance"), &batch, version).await;
        let fragment = Fragment {
            files: vec![data_file],
            physical_rows: Some(ROWS_PER_FRAGMENT),
            row_id_meta: row_id_starts.map(|starts| {
                let start = starts[slot];
                RowIdMeta::Inline(
                    write_row_ids(&RowIdSequence::from(
                        start..start + ROWS_PER_FRAGMENT as u64,
                    ))
                    .into(),
                )
            }),
            ..Fragment::new(slot as u64)
        };
        fragments.push(fragment);
    }
    let schema_batch = record_batch!(("i", Int32, [0])).unwrap();
    let mut manifest = Manifest::new(
        Schema::try_from(schema_batch.schema_ref().as_ref()).unwrap(),
        Arc::new(fragments),
        DataStorageFormat::new(version),
        Default::default(),
    );
    manifest.next_row_id = row_id_starts
        .and_then(|starts| starts.iter().max())
        .map_or(0, |start| start + ROWS_PER_FRAGMENT as u64);
    publish_manifest(&store, &base, &mut manifest).await;
    if fragment_deletions.iter().any(|deleted| !deleted.is_empty()) {
        manifest.version += 1;
        for (fragment, deleted) in Arc::make_mut(&mut manifest.fragments)
            .iter_mut()
            .zip(fragment_deletions)
        {
            fragment.deletion_file = write_deletion_file(
                &base,
                fragment.id,
                manifest.version,
                &DeletionVector::from_iter(deleted.iter().copied()),
                &store,
                0,
            )
            .await
            .unwrap();
        }
        publish_manifest(&store, &base, &mut manifest).await;
    }
    let table = Table::open(dir.path().to_str().unwrap()).await.unwrap();
    (dir, table)
}

fn deletion_reads(table: &Table) -> Vec<Path> {
    table
        .state
        .object_store
        .io_stats_incremental()
        .requests
        .into_iter()
        .map(|request| request.path)
        .filter(|path| path.parts().any(|part| part.as_ref() == "_deletions"))
        .collect()
}

fn deletion_path(table: &Table, slot: usize) -> Path {
    let fragment = &table.manifest().fragments[slot];
    deletion_file_path(
        &table.state.base,
        fragment.id,
        fragment.deletion_file.as_ref().unwrap(),
    )
}

fn values(batch: &RecordBatch) -> &[i32] {
    batch["i"].as_primitive::<Int32Type>().values().as_ref()
}

#[tokio::test]
async fn test_offset_mapping_skips_wholly_deleted_fragment() {
    // Dataset deletion normally removes wholly deleted fragments from the manifest.
    let (_dir, table) = number_table(None, &[&[], &[0, 1, 2, 3], &[]]).await;
    table.state.object_store.io_stats_incremental();
    let addresses = row_offsets_to_row_addresses(&table.get_fragments(), &[4, 3, 4, 8])
        .await
        .unwrap();
    let third_fragment = u64::from(RowAddress::new_from_parts(2, 0));
    assert_eq!(
        addresses,
        [third_fragment, 3, third_fragment, RowAddress::TOMBSTONE_ROW]
    );
    assert!(deletion_reads(&table).is_empty());
}

#[rstest]
#[case::last_fragment(&[23], &[31], &[7], true, true)]
#[case::duplicates_unordered(&[23, 21, 23, 22], &[31, 29, 31, 30], &[7], true, true)]
#[case::multiple_fragments(&[23, 0, 22, 3], &[31, 1, 30, 5], &[0, 1, 7], true, true)]
#[case::empty(&[], &[], &[], true, true)]
#[case::empty_missing_deletion_counts(&[], &[], &[], true, false)]
#[case::no_deletions(&[31, 28, 31, 0], &[31, 28, 31, 0], &[], false, true)]
#[case::missing_deletion_counts(&[23], &[31], &[0, 1, 2, 3, 4, 5, 6, 7], true, false)]
#[tokio::test]
async fn test_take_selected_deletion_reads(
    #[case] offsets: &[u64],
    #[case] expected_values: &[i32],
    #[case] expected_deletion_fragments: &[usize],
    #[case] has_deletions: bool,
    #[case] has_deletion_counts: bool,
) {
    let deleted: &[u32] = if has_deletions { &[0] } else { &[] };
    let (_dir, mut table) = number_table(None, &[deleted; 8]).await;
    if !has_deletion_counts {
        let manifest = Arc::make_mut(&mut table.manifest);
        for fragment in Arc::make_mut(&mut manifest.fragments) {
            fragment.deletion_file.as_mut().unwrap().num_deleted_rows = None;
        }
    }
    let expected_paths: Vec<_> = expected_deletion_fragments
        .iter()
        .map(|&slot| deletion_path(&table, slot))
        .collect();
    table.state.object_store.io_stats_incremental();

    let batch = table.take(offsets, table.schema()).await.unwrap();
    assert_eq!(values(&batch), expected_values);
    if offsets.is_empty() {
        assert_eq!(table.state.object_store.io_stats_snapshot().read_iops, 0);
    }
    assert_eq!(deletion_reads(&table), expected_paths);
}

#[tokio::test]
async fn test_stable_point_take_reads_candidate_deletions() {
    let starts = [0, 4, 8, 12, 16, 20, 24, 28];
    let (_dir, table) = number_table(Some(&starts), &[&[0][..]; 8]).await;
    let expected_path = deletion_path(&table, 7);
    let schema = table.schema();
    table.state.object_store.io_stats_incremental();

    assert_eq!(values(&table.take_rows(&[31], schema).await.unwrap()), [31]);
    assert_eq!(deletion_reads(&table), [expected_path]);
    assert_eq!(values(&table.take_rows(&[31], schema).await.unwrap()), [31]);
    assert!(deletion_reads(&table).is_empty());
    let projection = TakeProjection::try_new(Arc::new(table.clone()), schema).unwrap();
    for missing in [28, u64::MAX] {
        assert!(values(&table.take_rows(&[missing], schema).await.unwrap()).is_empty());
        let error = table
            .take_with_options(
                TakeSelection::RowIds(&[missing]),
                &projection,
                TakeOptions {
                    missing_row_policy: MissingRowPolicy::Error,
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains(&format!("first missing row ID {missing}")),
            "{error}"
        );
    }
    assert!(deletion_reads(&table).is_empty());
    let ignored = table
        .take_with_options(
            TakeSelection::RowIds(&[28]),
            &projection,
            TakeOptions {
                ignore_deletions: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(values(&ignored), [28]);
    assert_eq!(
        values(&table.take_rows(&[31, 28, 2, 31], schema).await.unwrap()),
        [31, 2, 31]
    );

    let original = table.checkout_version(1).await.unwrap();
    assert_eq!(
        values(&original.take_rows(&[28], original.schema()).await.unwrap()),
        [28]
    );
}

#[tokio::test]
async fn test_stable_point_take_checks_each_candidate_copy() {
    // Both fragments cover row ID 2, but only fragment 1 has it tombstoned.
    let (_dir, mut table) = number_table(Some(&[0, 2, 8, 12]), &[&[0][..]; 4]).await;
    table.state.manifest_location.e_tag = None;
    let mut expected = [deletion_path(&table, 0), deletion_path(&table, 1)];
    expected.sort();
    table.state.object_store.io_stats_incremental();
    let projection = TakeProjection::try_new(Arc::new(table.clone()), table.schema()).unwrap();
    let ignore = TakeOptions {
        ignore_deletions: true,
        ..Default::default()
    };
    let table_ref = &table;
    let projection = &projection;
    let take = |id| async move {
        table_ref
            .take_with_options(TakeSelection::RowIds(&[id]), projection, ignore)
            .await
    };
    assert_eq!(values(&take(2).await.unwrap()), [2]);
    let mut paths = deletion_reads(&table);
    paths.sort();
    assert_eq!(paths, expected);
    assert_eq!(take(0).await.unwrap().num_rows(), 0);
    for error in [
        take(3).await.unwrap_err(),
        table.take_rows(&[3], table.schema()).await.unwrap_err(),
    ] {
        assert!(matches!(error, Error::Internal { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("stable row id 3 is live in multiple fragments"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn test_take_duplicates_with_system_and_empty_projections() {
    let (_dir, table) = number_table(Some(&[10, 20]), &[&[1], &[]]).await;
    let columns = [
        ROW_OFFSET,
        ROW_CREATED_AT_VERSION,
        "i",
        ROW_ADDR,
        ROW_LAST_UPDATED_AT_VERSION,
        ROW_ID,
    ];
    let projection = table
        .schema()
        .project_preserve_system_columns(&columns)
        .unwrap();
    let batch = table
        .take_with_io_parallelism(&[4, 0, 4, 2], &projection, NonZeroU32::new(2))
        .await
        .unwrap();
    assert_eq!(
        batch
            .schema_ref()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        columns
    );
    let system = |name: &str| batch[name].as_primitive::<UInt64Type>().values().as_ref();
    let second_fragment = u64::from(RowAddress::new_from_parts(1, 1));
    assert_eq!(values(&batch), [5, 0, 5, 3]);
    assert_eq!(system(ROW_OFFSET), [4, 0, 4, 2]);
    assert_eq!(system(ROW_ADDR), [second_fragment, 0, second_fragment, 3]);
    assert_eq!(system(ROW_ID), [21, 10, 21, 13]);
    assert_eq!(system(ROW_CREATED_AT_VERSION), [1, 1, 1, 1]);
    assert_eq!(system(ROW_LAST_UPDATED_AT_VERSION), [1, 1, 1, 1]);

    let empty = Schema::default();
    table.state.object_store.io_stats_incremental();
    let batch = table.take_rows(&[21, 11, 10, 21], &empty).await.unwrap();
    assert_eq!((batch.num_rows(), batch.num_columns()), (3, 0));
    let batch = table.take(&[4, 0, 4, 2], &empty).await.unwrap();
    assert_eq!((batch.num_rows(), batch.num_columns()), (4, 0));
    assert!(
        table
            .state
            .object_store
            .io_stats_incremental()
            .requests
            .iter()
            .all(|request| !request.path.parts().any(|part| part.as_ref() == "data"))
    );
}
