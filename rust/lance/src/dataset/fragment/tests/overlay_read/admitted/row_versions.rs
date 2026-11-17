// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::{UInt64Array, cast::AsArray};
use lance_core::{ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION};
use lance_table::rowids::version::{RowDatasetVersionMeta, RowDatasetVersionSequence};

#[rstest]
#[tokio::test]
async fn versions_compose_with_real_overlay_and_deletion_snapshots(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    let mut dataset = write_payload_dataset(version, true, true).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("version-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([1, 3, 4])),
            vec![payload("overlay", 3, true, 0)],
            version,
        )
        .await;
    }
    dataset
        .delete("untouched IN ('other1', 'other4', 'other7', 'other10')")
        .await
        .unwrap();
    for fragment in 0..2 {
        let scheduler = ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
                use_lite_scheduler: Some(lite),
            },
        );
        let original = dataset
            .get_fragment(fragment)
            .unwrap()
            .open(
                dataset.schema(),
                FragReadConfig::default()
                    .with_scan_scheduler(scheduler)
                    .with_row_id(true)
                    .with_row_address(true)
                    .with_row_created_at_version(true)
                    .with_row_last_updated_at_version(true),
            )
            .await
            .unwrap();
        for markers in [false, true] {
            let mut reader = original.clone();
            if markers {
                reader.with_make_deletions_null();
            }
            let budget = AllocationBudget::new(2 * 65536);
            for selected in [vec![0, 1, 2, 4, 5], vec![1, 4], vec![]] {
                let indices = UInt32Array::from(selected.clone());
                let output = reader
                    .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(output, reader.take_as_batch(&selected, None).await.unwrap());
                assert_eq!(output[ROW_CREATED_AT_VERSION].null_count(), 0);
                assert_eq!(output[ROW_LAST_UPDATED_AT_VERSION].null_count(), 0);
                assert_eq!(
                    output.num_rows(),
                    if markers {
                        selected.len()
                    } else {
                        selected.iter().filter(|row| ![1, 4].contains(row)).count()
                    }
                );
                drop(output);
                assert_eq!(budget.reserved(), 0);
            }
            let indices = UInt32Array::from(vec![0, 1, 4, 5]);
            let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                tokio::join!(
                    reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
                    reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
                )
            })
            .await
            .expect("version composition must progress with one CPU worker");
            let (left, right) = (left.unwrap().unwrap(), right.unwrap().unwrap());
            assert_eq!(left, right);
            drop((left, right));
            assert_eq!(budget.reserved(), 0);
        }
    }
}

#[rstest]
#[case::default(0)]
#[case::shared(1)]
#[case::distinct(2)]
#[tokio::test]
async fn version_outputs_reserve_exact_credit_and_keep_aliases(#[case] mode: usize) {
    let dataset = write_payload_dataset(LanceFileVersion::V2_3, false, true).await;
    let created = [3, 3, 8, 2, 2, 9];
    let updated = [5, 4, 4, 7, 10, 10];
    let source = Arc::new(RowDatasetVersionSequence::from_versions(&created));
    let other = Arc::new(RowDatasetVersionSequence::from_versions(&updated));
    for fragment in 0..2 {
        for (has_created, has_updated) in [(true, false), (false, true), (true, true)] {
            // Empty payload projections require system columns at open, before
            // substituting the immutable snapshots used by this budget fixture.
            let mut reader = dataset
                .get_fragment(fragment)
                .unwrap()
                .open(
                    &Schema::default(),
                    FragReadConfig::default()
                        .with_row_created_at_version(has_created)
                        .with_row_last_updated_at_version(has_updated),
                )
                .await
                .unwrap();
            if has_created {
                reader.created_at_sequence = (mode != 0).then(|| source.clone());
            }
            if has_updated {
                reader.last_updated_at_sequence = match mode {
                    0 => None,
                    1 => Some(source.clone()),
                    _ => Some(other.clone()),
                };
            }
            // Generated columns must follow output-schema placement.
            reader.output_schema = ArrowSchema::new(
                reader
                    .output_schema
                    .fields()
                    .iter()
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            for selected in [vec![0, 2, 3, 5], vec![5], vec![]] {
                let mut storage = vec![u32::MAX];
                storage.extend(&selected);
                let indices = UInt32Array::from(storage).slice(1, selected.len());
                let buffers = if mode == 2 && has_created && has_updated {
                    2
                } else {
                    1
                };
                let exact = selected.len() * 8 * buffers;
                let budget = AllocationBudget::new(1024);
                if exact > 0 {
                    assert!(
                        reader
                            .try_take_admitted_fullzip(
                                &indices,
                                budget.try_reserve(exact - 1).unwrap()
                            )
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
                for (name, enabled) in [
                    (ROW_CREATED_AT_VERSION, has_created),
                    (ROW_LAST_UPDATED_AT_VERSION, has_updated),
                ] {
                    if !enabled {
                        continue;
                    }
                    let expected = UInt64Array::from(
                        selected
                            .iter()
                            .map(|&row| {
                                if mode == 0 {
                                    1
                                } else if mode == 2 && name == ROW_LAST_UPDATED_AT_VERSION {
                                    updated[row as usize]
                                } else {
                                    created[row as usize]
                                }
                            })
                            .collect::<Vec<_>>(),
                    );
                    assert_eq!(
                        output[name].as_primitive::<arrow_array::types::UInt64Type>(),
                        &expected
                    );
                }
                if has_created && has_updated && mode != 2 {
                    assert_eq!(
                        output[ROW_CREATED_AT_VERSION].to_data().buffers()[0].as_ptr(),
                        output[ROW_LAST_UPDATED_AT_VERSION].to_data().buffers()[0].as_ptr()
                    );
                }
                if !selected.is_empty() {
                    let alias = output.column(0).to_data().buffers()[0].slice_with_length(0, 1);
                    drop(output);
                    assert_eq!(budget.reserved(), selected.len() * 8);
                    assert!(budget.try_reserve(1024).is_none());
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
                drop(retry);
                assert_eq!(budget.reserved(), 0);
            }
        }
    }
}

#[rstest]
#[tokio::test]
async fn version_sources_reject_wrong_lengths_and_spilled_cache_bypasses(
    #[values(false, true)] updated: bool,
) {
    let dataset = write_payload_dataset(LanceFileVersion::V2_3, false, true).await;
    let mut reader = open(&dataset, 0, dataset.schema(), false).await;
    let short = Some(Arc::new(RowDatasetVersionSequence::from_uniform_row_count(
        5, 8,
    )));
    if updated {
        reader.with_row_last_updated_at_version(short);
    } else {
        reader.with_row_created_at_version(short);
    }
    let indices = UInt32Array::from(vec![0]);
    let budget = AllocationBudget::new(65536);
    let error = reader
        .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("sequence length differs"));
    assert_eq!(budget.reserved(), 0);
    let mut fragment = reader.fragment.as_ref().clone();
    if updated {
        fragment.metadata.last_updated_at_version_meta = Some(RowDatasetVersionMeta::Column);
    } else {
        fragment.metadata.created_at_version_meta = Some(RowDatasetVersionMeta::Column);
    }
    reader.fragment = Arc::new(fragment);
    for cached in [false, true] {
        let source =
            cached.then(|| Arc::new(RowDatasetVersionSequence::from_uniform_row_count(6, 8)));
        if updated {
            reader.last_updated_at_sequence = source;
        } else {
            reader.created_at_sequence = source;
        }
        let error = reader
            .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("spilled version columns"));
        assert_eq!(budget.reserved(), 0);
    }
}
