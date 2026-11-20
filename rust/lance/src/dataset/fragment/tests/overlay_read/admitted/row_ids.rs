// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::{UInt64Array, cast::AsArray};
use lance_core::{ROW_ADDR, ROW_ID, utils::address::RowAddress};
use lance_table::{format::RowIdMeta, rowids::RowIdSequence};
use tokio::sync::OnceCell;

#[rstest]
#[tokio::test]
async fn generated_ids_compose_with_real_overlays_and_deletions(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] stable: bool,
) {
    let mut dataset = write_payload_dataset(version, true, stable).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("ids-old-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([0, 2, 4])),
            vec![payload("old", 3, true, usize::MAX)],
            version,
        )
        .await;
        dataset = commit_overlay(
            dataset,
            &format!("ids-new-{fragment}"),
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
        if stable {
            assert!(matches!(
                original.fragment.metadata.row_id_meta,
                Some(RowIdMeta::Inline(_))
            ));
            assert!(original.row_id_sequence.get().is_none());
        }
        // Enable IDs after open, on a clone. Load the stable snapshot in async
        // preparation, never from the single CPU worker doing assembly.
        for (has_id, has_address) in [(false, true), (true, false), (true, true)] {
            let mut reader = original.clone();
            if has_address {
                reader.with_row_address();
            }
            if has_id {
                reader.with_row_id();
            }
            // Placement must follow the actual output schema, not assume IDs
            // are appended in a fixed builder order.
            let fields = reader
                .output_schema
                .fields()
                .iter()
                .rev()
                .cloned()
                .collect::<Vec<_>>();
            reader.output_schema =
                ArrowSchema::new_with_metadata(fields, reader.output_schema.metadata().clone());
            for selected in [vec![0, 1, 2, 4, 5], vec![1, 4], vec![]] {
                let indices = UInt32Array::from(selected.clone());
                let budget = AllocationBudget::new(65536);
                let output = reader
                    .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
                    .await
                    .unwrap()
                    .unwrap();
                if stable && !has_id {
                    assert!(reader.row_id_sequence.get().is_none());
                }
                let expected = reader.take_as_batch(&selected, None).await.unwrap();
                assert_eq!(output, expected);
                let survivors = selected
                    .iter()
                    .copied()
                    .filter(|row| ![1, 4].contains(row))
                    .collect::<Vec<_>>();
                let addresses = survivors
                    .iter()
                    .map(|&row| u64::from(RowAddress::new_from_parts(fragment as u32, row)))
                    .collect::<Vec<_>>();
                if has_address {
                    assert_eq!(
                        output[ROW_ADDR]
                            .as_primitive::<arrow_array::types::UInt64Type>()
                            .values()
                            .as_ref(),
                        addresses
                    );
                }
                if has_id {
                    let ids = if stable {
                        survivors
                            .iter()
                            .map(|&row| (fragment * 6) as u64 + row as u64)
                            .collect::<Vec<_>>()
                    } else {
                        addresses
                    };
                    assert_eq!(
                        output[ROW_ID]
                            .as_primitive::<arrow_array::types::UInt64Type>()
                            .values()
                            .as_ref(),
                        ids
                    );
                }
                if !survivors.is_empty() {
                    let generated = if has_id {
                        &output[ROW_ID]
                    } else {
                        &output[ROW_ADDR]
                    };
                    let alias = generated.to_data().buffers()[0].slice_with_length(0, 1);
                    if has_id && has_address && !stable {
                        assert_eq!(
                            output[ROW_ID].to_data().buffers()[0].as_ptr(),
                            output[ROW_ADDR].to_data().buffers()[0].as_ptr()
                        );
                    }
                    drop(output);
                    assert_eq!(budget.reserved(), survivors.len() * 8);
                    assert!(budget.try_reserve(65536).is_none());
                    drop(alias);
                } else {
                    drop(output);
                }
                assert_eq!(budget.reserved(), 0);
            }
        }
        let mut reader = original;
        reader.with_row_id().with_row_address();
        let indices = UInt32Array::from(vec![0, 1, 2, 4, 5]);
        let budget = AllocationBudget::new(2 * 65536);
        let (a, b) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(
                reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
                reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
            )
        })
        .await
        .expect("identifier generation must progress with one CPU worker");
        let a = a.unwrap().unwrap();
        let b = b.unwrap().unwrap();
        assert_eq!(a, b);
        drop((a, b));
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[tokio::test]
async fn identifier_only_projection_has_exact_budget_and_alias_lifetime(
    #[values(false, true)] stable: bool,
) {
    let mut dataset = write_payload_dataset(LanceFileVersion::V2_3, false, stable).await;
    dataset.delete("untouched = 'other7'").await.unwrap();
    let projection = Schema::default();
    let reader = dataset
        .get_fragment(1)
        .unwrap()
        .open(
            &projection,
            FragReadConfig::default()
                .with_row_id(true)
                .with_row_address(true),
        )
        .await
        .unwrap();
    let budget = AllocationBudget::new(1024);
    for selected in [vec![0, 2, 5], vec![0, 1, 2, 5], vec![1], vec![]] {
        let count = selected.iter().filter(|&&row| row != 1).count();
        let filtered_bytes = if count < selected.len() { count * 4 } else { 0 };
        let output_bytes = count * 8 * if stable { 2 } else { 1 };
        let exact = filtered_bytes + output_bytes;
        let indices = UInt32Array::from(selected.clone());
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
        let result = reader
            .try_take_admitted_fullzip(&indices, budget.try_reserve(exact).unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result, reader.take_as_batch(&selected, None).await.unwrap());
        assert_eq!(budget.reserved(), output_bytes);
        if count > 0 {
            let id = result[ROW_ID].clone();
            let addr = result[ROW_ADDR].to_data().buffers()[0].slice_with_length(0, 1);
            drop(result);
            assert_eq!(budget.reserved(), output_bytes);
            drop(id);
            assert_eq!(budget.reserved(), count * 8);
            drop(addr);
        } else {
            drop(result);
        }
        assert_eq!(budget.reserved(), 0);
    }
}

#[tokio::test]
async fn sparse_stable_ids_validate_snapshot_length_and_keep_nonmonotonic_values() {
    let dataset = write_payload_dataset(LanceFileVersion::V2_3, false, true).await;
    let mut reader = open(&dataset, 1, dataset.schema(), false).await;
    reader.with_row_id().with_row_address();
    // An immutable fragmented/nonmonotonic mapping need not resemble addresses.
    // Use the same snapshot for ordinary and admitted paths, plus explicit IDs.
    let ids = [90, 7, 10000, 4, 500, 2];
    reader.row_id_sequence = Arc::new(OnceCell::new_with(Some(Arc::new(RowIdSequence::from(
        ids.as_slice(),
    )))));
    let indices = UInt32Array::from(vec![0, 2, 5]);
    let budget = AllocationBudget::new(65536);
    let result = reader
        .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        result,
        reader.take_as_batch(indices.values(), None).await.unwrap()
    );
    assert_eq!(
        result[ROW_ID].as_primitive::<arrow_array::types::UInt64Type>(),
        &UInt64Array::from(vec![90, 10000, 2])
    );
    drop(result);
    assert_eq!(budget.reserved(), 0);
    reader.row_id_sequence = Arc::new(OnceCell::new_with(Some(Arc::new(RowIdSequence::from(
        0..5,
    )))));
    let error = reader
        .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("sequence length 5 differs from fragment physical length 6")
    );
    assert_eq!(budget.reserved(), 0);
    reader.fragment_id = u32::MAX as usize + 1;
    let error = reader
        .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("row-address domain"));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn spilled_ids_require_admitted_input_but_addresses_do_not_load_them() {
    let dataset = write_payload_dataset(LanceFileVersion::V2_3, false, true).await;
    let mut reader = open(&dataset, 1, dataset.schema(), false).await;
    let mut fragment = reader.fragment.as_ref().clone();
    // Deliberately omit a lineage file: an accidental ordinary loader would
    // fail differently. The admitted API must reject the unsupported descriptor
    // before I/O, including when an earlier reader has cached a sequence.
    fragment.metadata.row_id_meta = Some(RowIdMeta::Column);
    reader.fragment = Arc::new(fragment);
    reader.with_row_address();
    let indices = UInt32Array::from(vec![0, 2, 5]);
    let budget = AllocationBudget::new(65536);
    let output = reader
        .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert!(reader.row_id_sequence.get().is_none());
    assert_eq!(
        output[ROW_ADDR].as_primitive::<arrow_array::types::UInt64Type>(),
        &UInt64Array::from(
            [0, 2, 5]
                .into_iter()
                .map(|row| u64::from(RowAddress::new_from_parts(1, row)))
                .collect::<Vec<_>>()
        )
    );
    drop(output);
    assert_eq!(budget.reserved(), 0);
    reader.with_row_id();
    for cached in [false, true] {
        reader.row_id_sequence = Arc::new(OnceCell::new_with(
            cached.then(|| Arc::new(RowIdSequence::from(6..12))),
        ));
        let error = reader
            .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("spilled row-ID columns"));
        assert_eq!(budget.reserved(), 0);
    }
}
