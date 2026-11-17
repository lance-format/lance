// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::{UInt64Array, cast::AsArray};
use lance_core::{
    ROW_ADDR, ROW_ID,
    utils::{address::RowAddress, deletion::DeletionVector},
};

#[rstest]
#[tokio::test]
async fn deletion_markers_preserve_overlay_payload_and_physical_identity(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] stable: bool,
) {
    let mut dataset = write_payload_dataset(version, true, stable).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("marker-old-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([0, 2, 4])),
            vec![payload("old", 3, true, usize::MAX)],
            version,
        )
        .await;
        dataset = commit_overlay(
            dataset,
            &format!("marker-new-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([2, 3, 4])),
            vec![payload("new", 3, true, 0)],
            version,
        )
        .await;
    }
    dataset
        .delete("untouched IN ('other1', 'other4', 'other7', 'other10')")
        .await
        .unwrap();
    for fragment in 0..2 {
        let original = open(&dataset, fragment, dataset.schema(), lite).await;
        let unfiltered = dataset
            .get_fragment(fragment)
            .unwrap()
            .open(
                dataset.schema(),
                FragReadConfig::default().with_ignore_deletions(true),
            )
            .await
            .unwrap();
        for (id, address) in [(true, false), (false, true), (true, true)] {
            let mut reader = original.clone();
            if id {
                reader.with_row_id();
            }
            if address {
                reader.with_row_address();
            }
            reader.with_make_deletions_null();
            for selected in [vec![0, 1, 2, 3, 4, 5], vec![1, 4], vec![0, 5], vec![]] {
                let indices = UInt32Array::from(selected.clone());
                let budget = AllocationBudget::new(65536);
                let output = reader
                    .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(output, reader.take_as_batch(&selected, None).await.unwrap());
                assert_eq!(output.num_rows(), selected.len());
                let payload = unfiltered.take_as_batch(&selected, None).await.unwrap();
                for field in ["payload", "untouched"] {
                    assert_eq!(output[field].as_ref(), payload[field].as_ref());
                }
                for (name, enabled) in [(ROW_ID, id), (ROW_ADDR, address)] {
                    if !enabled {
                        continue;
                    }
                    let expected = UInt64Array::from(
                        selected
                            .iter()
                            .map(|&row| {
                                if [1, 4].contains(&row) {
                                    None
                                } else if name == ROW_ID && stable {
                                    Some((fragment * 6) as u64 + row as u64)
                                } else {
                                    Some(u64::from(RowAddress::new_from_parts(
                                        fragment as u32,
                                        row,
                                    )))
                                }
                            })
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        output[name].as_primitive::<arrow_array::types::UInt64Type>(),
                        &expected
                    );
                }
                if let Some(position) = selected.iter().position(|&row| row == 4) {
                    // A null identifier must not mask a nonnull overlay value.
                    assert!(!output["payload"].is_null(position));
                }
                let column = if id { ROW_ID } else { ROW_ADDR };
                if let Some(nulls) = output[column].nulls() {
                    if id && address {
                        assert_eq!(
                            nulls.buffer().as_ptr(),
                            output[ROW_ADDR].nulls().unwrap().buffer().as_ptr()
                        );
                    }
                    let alias = nulls.buffer().slice_with_length(0, 1);
                    drop(output);
                    assert_eq!(budget.reserved(), selected.len().div_ceil(8));
                    assert!(budget.try_reserve(65536).is_none());
                    drop(alias);
                } else {
                    drop(output);
                }
                assert_eq!(budget.reserved(), 0);
            }
        }
        let mut reader = original;
        reader
            .with_row_id()
            .with_row_address()
            .with_make_deletions_null();
        let indices = UInt32Array::from(vec![0, 1, 2, 4, 5]);
        let budget = AllocationBudget::new(2 * 65536);
        let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(
                reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
                reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
            )
        })
        .await
        .expect("marker composition must progress with one CPU worker");
        let left = left.unwrap().unwrap();
        let right = right.unwrap().unwrap();
        assert_eq!(left, right);
        drop((left, right));
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[tokio::test]
async fn marker_bitmap_has_exact_shared_credit_across_byte_boundary(
    #[values(false, true)] stable: bool,
    #[values(false, true)] bitmap: bool,
) {
    let data = lance_datagen::gen_batch()
        .col(
            "i",
            lance_datagen::array::step::<arrow_array::types::Int32Type>(),
        )
        .into_reader_rows(
            lance_datagen::RowCount::from(24),
            lance_datagen::BatchCount::from(1),
        );
    let mut dataset = Dataset::write(
        data,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 12,
            enable_stable_row_ids: stable,
            data_storage_version: Some(LanceFileVersion::V2_3),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .delete("i IN (1, 7, 8, 11, 13, 19, 20, 23)")
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    for fragment in 0..2 {
        let mut reader = dataset
            .get_fragment(fragment)
            .unwrap()
            .open(
                &Schema::default(),
                FragReadConfig::default()
                    .with_row_id(true)
                    .with_row_address(true)
                    .with_make_deletions_null(true),
            )
            .await
            .unwrap();
        assert_eq!(
            reader
                .deletion_vec
                .as_ref()
                .unwrap()
                .to_sorted_iter()
                .collect::<Vec<_>>(),
            vec![1, 7, 8, 11]
        );
        reader.deletion_vec = Some(Arc::new(if bitmap {
            DeletionVector::Bitmap([1, 7, 8, 11].into_iter().collect())
        } else {
            DeletionVector::Set([1, 7, 8, 11].into_iter().collect())
        }));
        for selected in [
            (0..12).collect::<Vec<u32>>(),
            vec![1, 7, 8, 11],
            vec![0, 7, 8, 11],
            vec![0, 2, 3],
            vec![],
        ] {
            // Borrow a sliced physical selection; bitmap positions are relative
            // to the output, not to either its backing or physical offsets.
            let mut storage = vec![u32::MAX];
            storage.extend(&selected);
            let indices = UInt32Array::from(storage).slice(1, selected.len());
            let values = selected.len() * 8 * if stable { 2 } else { 1 };
            let mask = if selected.iter().any(|row| [1, 7, 8, 11].contains(row)) {
                selected.len().div_ceil(8)
            } else {
                0
            };
            let exact = values + mask;
            let budget = AllocationBudget::new(1024);
            if exact > 0 {
                assert!(
                    reader
                        .try_take_admitted_fullzip(&indices, budget.try_reserve(exact - 1).unwrap())
                        .await
                        .unwrap()
                        .is_none()
                );
                assert_eq!(budget.reserved(), 0);
            }
            let output = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(exact).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(output, reader.take_as_batch(&selected, None).await.unwrap());
            assert_eq!(budget.reserved(), exact);
            if mask > 0 {
                let nulls = output[ROW_ID].nulls().unwrap();
                assert_eq!(
                    nulls.buffer().as_ptr(),
                    output[ROW_ADDR].nulls().unwrap().buffer().as_ptr()
                );
                assert_eq!(nulls.buffer().len(), mask);
                let alias = nulls.buffer().slice_with_length(mask - 1, 1);
                drop(output);
                assert_eq!(budget.reserved(), mask);
                drop(alias);
            } else {
                drop(output);
            }
            assert_eq!(budget.reserved(), 0);
            let retry = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(exact).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(retry, reader.take_as_batch(&selected, None).await.unwrap());
            drop(retry);
            assert_eq!(budget.reserved(), 0);
        }
    }
}

#[tokio::test]
async fn markers_use_reader_snapshot_and_ignore_deletions_setting() {
    let mut dataset = write_payload_dataset(LanceFileVersion::V2_3, false, true).await;
    dataset.delete("untouched = 'other1'").await.unwrap();
    let config = FragReadConfig::default()
        .with_row_id(true)
        .with_make_deletions_null(true);
    let first = dataset
        .get_fragment(0)
        .unwrap()
        .open(dataset.schema(), config.clone())
        .await
        .unwrap();
    dataset.delete("untouched = 'other2'").await.unwrap();
    let second = dataset
        .get_fragment(0)
        .unwrap()
        .open(dataset.schema(), config.clone())
        .await
        .unwrap();
    let ignored = dataset
        .get_fragment(0)
        .unwrap()
        .open(dataset.schema(), config.with_ignore_deletions(true))
        .await
        .unwrap();
    let indices = UInt32Array::from(vec![0, 1, 2, 3]);
    let budget = AllocationBudget::new(65536);
    for (reader, nulls) in [(&first, 1), (&second, 2), (&ignored, 0)] {
        let output = reader
            .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            output,
            reader.take_as_batch(indices.values(), None).await.unwrap()
        );
        assert_eq!(output.num_rows(), 4);
        assert_eq!(output[ROW_ID].null_count(), nulls);
        drop(output);
        assert_eq!(budget.reserved(), 0);
    }
}
