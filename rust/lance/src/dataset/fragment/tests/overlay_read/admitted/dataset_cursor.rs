// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::{
    AdmittedDatasetScanOutcome as Outcome, AdmittedScanPosition,
    schema_evolution::NewColumnTransform,
};
use lance_encoding::{
    constants::{COMPRESSION_META_KEY, DICT_DIVISOR_META_KEY},
    format::metadata::compressive_encoding::Compression,
};

mod cancellation;
mod dictionary;
mod prefetch;
mod waiting;

#[rstest]
#[tokio::test]
async fn dataset_cursor_composes_snapshot_overlay_deletions_and_missing_columns(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    let mut dataset = write_payload_dataset(version, true, true).await;
    let field = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("dataset-cursor-{fragment}"),
            fragment,
            &[field],
            OverlayCoverage::dense(bitmap([1, 3, 4])),
            vec![payload("new", 3, true, 0)],
            version,
        )
        .await;
    }
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "late",
                DataType::LargeBinary,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    dataset
        .delete("untouched IN ('other0', 'other1', 'other6', 'other7')")
        .await
        .unwrap();
    // Real IDs differ from manifest positions. The cursor must preserve both.
    Arc::make_mut(&mut Arc::make_mut(&mut dataset.manifest).fragments).reverse();
    let oracle = dataset.clone();
    let config = FragReadConfig::default()
        .with_row_id(true)
        .with_row_address(true)
        .with_scan_scheduler(ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
                use_lite_scheduler: Some(lite),
            },
        ));
    let budget = AllocationBudget::new(2 * 65536);
    let mut left = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().clone(),
            config.clone(),
            2,
            65536,
            budget.clone(),
        )
        .unwrap();
    let mut right = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().clone(),
            config.clone(),
            2,
            65536,
            budget.clone(),
        )
        .unwrap();
    // Changing the caller handle does not replace the cursor's manifest snapshot.
    Arc::make_mut(&mut dataset.manifest).fragments = Arc::new(Vec::new());
    for id in [1, 0] {
        let reader = oracle
            .get_fragment(id)
            .unwrap()
            .open(oracle.schema(), config.clone())
            .await
            .unwrap();
        for start in [0, 2, 4] {
            let expected_position = Some(AdmittedScanPosition {
                fragment_id: id as u64,
                row_offset: start,
            });
            assert_eq!(left.position(), expected_position);
            assert_eq!(right.position(), expected_position);
            let (a, b) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                tokio::join!(left.try_next(), right.try_next())
            })
            .await
            .expect("two dataset cursors must progress with one CPU worker");
            let expected = reader
                .take_as_batch(&[start as u32, start as u32 + 1], None)
                .await
                .unwrap();
            for result in [a, b] {
                let Outcome::Batch {
                    fragment_id,
                    rows,
                    batch,
                } = result.unwrap()
                else {
                    panic!("a complete snapshot prefix must be returned");
                };
                assert_eq!(fragment_id, id as u64);
                assert_eq!(rows, start..start + 2);
                assert_eq!(batch, expected);
                if start == 0 {
                    assert_eq!(batch.num_rows(), 0);
                }
            }
            assert_eq!(budget.reserved(), 0);
        }
    }
    assert_eq!(left.position(), None);
    assert_eq!(right.position(), None);
    assert!(matches!(left.try_next().await.unwrap(), Outcome::Finished));
    assert!(matches!(right.try_next().await.unwrap(), Outcome::Finished));
}

async fn null_dataset(rows: u32) -> Dataset {
    let batch = arrow_array::record_batch!(("id", UInt32, (0..rows).collect::<Vec<_>>())).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 7,
            data_storage_version: Some(LanceFileVersion::V2_3),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "late",
                DataType::LargeBinary,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    dataset
}

#[rstest]
#[case::smaller(128)]
#[case::larger(512)]
#[tokio::test]
async fn dataset_cursor_shares_capacity_across_unequal_fragments(#[case] rows: u32) {
    let dataset = null_dataset(rows).await;
    let budget = AllocationBudget::new(64);
    let mut scan = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().project(&["late"]).unwrap(),
            Default::default(),
            u32::MAX,
            64,
            budget.clone(),
        )
        .unwrap();
    let mut delivered = 0;
    let mut offsets = 0;
    let mut largest_batch = 0;
    let mut crossed = 0;
    for fragment in dataset.iter_fragments() {
        let mut position = 0;
        let end = fragment.physical_rows.unwrap() as u64;
        while position < end {
            assert_eq!(
                scan.position(),
                Some(AdmittedScanPosition {
                    fragment_id: fragment.id,
                    row_offset: position
                })
            );
            let Outcome::Batch {
                fragment_id,
                rows,
                batch,
            } = scan.try_next().await.unwrap()
            else {
                panic!("a minimal null row fits this workspace");
            };
            assert_eq!(fragment_id, fragment.id);
            assert_eq!(rows.start, position);
            assert!(rows.end <= end);
            assert_eq!(batch.num_rows() as u64, rows.end - rows.start);
            assert_eq!(
                batch["late"].as_ref(),
                arrow_array::new_null_array(&DataType::LargeBinary, batch.num_rows()).as_ref()
            );
            delivered += batch.num_rows();
            largest_batch = largest_batch.max(batch.num_rows());
            offsets += (batch.num_rows() + 1) * 8;
            position = rows.end;
            let alias = batch["late"].to_data().buffers()[0].slice_with_length(0, 1);
            drop(batch);
            assert!(budget.reserved() > 0 && budget.reserved() <= 64);
            let next = scan.position();
            if next.is_some() {
                if position == end {
                    crossed += 1;
                }
                assert!(matches!(
                    scan.try_next().await.unwrap(),
                    Outcome::NeedsCapacity
                ));
                assert_eq!(scan.position(), next);
            }
            drop(alias);
            assert_eq!(budget.reserved(), 0);
        }
    }
    assert_eq!(delivered, rows as usize);
    assert_eq!(crossed, dataset.fragments().len() - 1);
    assert!(offsets > 16 * 64);
    assert!(largest_batch > 1);
    assert!(matches!(scan.try_next().await.unwrap(), Outcome::Finished));
}

#[tokio::test]
async fn dataset_cursor_opens_only_the_current_fragment_and_retries_open_errors() {
    let mut dataset = write_payload_dataset(LanceFileVersion::V2_3, false, false).await;
    let original = dataset.fragments()[1].files[0].path.clone();
    let missing = "deferred-fragment.lance";
    Arc::make_mut(&mut Arc::make_mut(&mut dataset.manifest).fragments)[1].files[0].path =
        missing.into();
    let budget = AllocationBudget::new(65536);
    let mut scan = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().clone(),
            Default::default(),
            6,
            65536,
            budget.clone(),
        )
        .unwrap();
    let Outcome::Batch {
        fragment_id,
        rows,
        batch,
    } = scan.try_next().await.unwrap()
    else {
        panic!("first file is valid");
    };
    assert_eq!(fragment_id, 0);
    assert_eq!(rows, 0..6);
    drop(batch);
    let next = scan.position();
    let error = scan.try_next().await.unwrap_err();
    assert!(matches!(error, Error::NotFound { .. } | Error::IO { .. }));
    assert!(error.to_string().contains(missing));
    assert_eq!(scan.position(), next);
    assert_eq!(budget.reserved(), 0);
    dataset
        .object_store
        .copy(
            &dataset.data_dir().join(original),
            &dataset.data_dir().join(missing),
        )
        .await
        .unwrap();
    let Outcome::Batch {
        fragment_id,
        rows,
        batch,
    } = scan.try_next().await.unwrap()
    else {
        panic!("the same failed position must retry");
    };
    assert_eq!(fragment_id, 1);
    assert_eq!(rows, 0..6);
    assert_eq!(batch.num_rows(), 6);
    drop(batch);
    assert!(matches!(scan.try_next().await.unwrap(), Outcome::Finished));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn dataset_cursor_replans_without_restarting_or_skipping_a_row() {
    let dataset = null_dataset(3).await;
    let budget = AllocationBudget::new(64);
    let mut scan = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().project(&["late"]).unwrap(),
            Default::default(),
            2,
            20,
            budget.clone(),
        )
        .unwrap();
    let position = scan.position();
    assert!(matches!(
        scan.try_next().await.unwrap(),
        Outcome::NeedsCapacity
    ));
    assert_eq!(scan.position(), position);
    assert_eq!(budget.reserved(), 0);
    let error = scan.set_workspace_bytes(0).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("positive workspace_bytes"));
    assert_eq!(scan.position(), position);
    scan.set_workspace_bytes(64).unwrap();
    let Outcome::Batch { rows, batch, .. } = scan.try_next().await.unwrap() else {
        panic!("replanned prefix fits");
    };
    assert_eq!(rows, 0..2);
    drop(batch);
    // Replan again after a committed prefix; do not restart at row zero.
    scan.set_workspace_bytes(20).unwrap();
    assert!(matches!(
        scan.try_next().await.unwrap(),
        Outcome::NeedsCapacity
    ));
    assert_eq!(scan.position().unwrap().row_offset, 2);
    scan.set_workspace_bytes(64).unwrap();
    let Outcome::Batch { rows, batch, .. } = scan.try_next().await.unwrap() else {
        panic!("last row fits");
    };
    assert_eq!(rows, 2..3);
    drop(batch);
    assert!(matches!(scan.try_next().await.unwrap(), Outcome::Finished));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn dataset_cursor_carries_prefix_hint_and_resets_for_replanning() {
    let dataset = null_dataset(21).await;
    let budget = AllocationBudget::new(512);
    let mut scan = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().project(&["late"]).unwrap(),
            Default::default(),
            7,
            64,
            budget.clone(),
        )
        .unwrap();
    let Outcome::Batch { rows, batch, .. } = scan.try_next().await.unwrap() else {
        panic!("a reduced prefix fits");
    };
    assert_eq!(rows.start, 0);
    let accepted = rows.end;
    assert!(accepted > 1 && accepted * 2 < 7);
    drop(batch);
    // Repeating the same workspace setting must not forget the successful size.
    scan.set_workspace_bytes(64).unwrap();
    let Outcome::Batch { rows, batch, .. } = scan.try_next().await.unwrap() else {
        panic!("the next prefix fits");
    };
    assert_eq!(rows, accepted..accepted * 2);
    drop(batch);
    scan.set_workspace_bytes(512).unwrap();
    let mut delivered = accepted * 2;
    while let Outcome::Batch {
        fragment_id,
        rows,
        batch,
    } = scan.try_next().await.unwrap()
    {
        assert_eq!(
            rows.end, 7,
            "replanned or new fragment uses its full ceiling"
        );
        if fragment_id != 0 {
            assert_eq!(rows.start, 0);
        }
        assert_eq!(batch.num_rows() as u64, rows.end - rows.start);
        assert_eq!(
            batch["late"].as_ref(),
            arrow_array::new_null_array(&DataType::LargeBinary, batch.num_rows()).as_ref()
        );
        delivered += rows.end - rows.start;
        drop(batch);
        assert_eq!(budget.reserved(), 0);
    }
    assert_eq!(delivered, 21);
    assert!(scan.position().is_none());
}

async fn growth_fixture(version: LanceFileVersion, is_raw: bool) -> (Dataset, RecordBatch) {
    let values = (0..2048)
        .map(|row| {
            let len = if row < 512 { 128 } else { 1 };
            vec![(row % 251) as u8; len]
        })
        .collect::<Vec<_>>();
    let payload = Arc::new(LargeBinaryArray::from_iter_values(
        values.iter().map(Vec::as_slice),
    ));
    let mut field = fullzip_field("payload", DataType::LargeBinary);
    if is_raw {
        // A structural preference does not disable dictionary encoding. Keep
        // this fixture's minimum admission dependent on row width, not a shared
        // page dictionary that is larger than the entire test workspace.
        let mut metadata = field.metadata().clone();
        metadata.insert(DICT_DIVISOR_META_KEY.into(), u64::MAX.to_string());
        metadata.insert(COMPRESSION_META_KEY.into(), "none".into());
        field = field.with_metadata(metadata);
    }
    let schema = Arc::new(ArrowSchema::new(vec![field]));
    let expected = RecordBatch::try_new(schema.clone(), vec![payload]).unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(expected.clone())], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 1024,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 2);
    let scheduler = ScanScheduler::new(dataset.object_store.clone(), SchedulerConfig::new(1024));
    for fragment in dataset.fragments() {
        let file = &fragment.files[0];
        let source = scheduler
            .open_file(
                &dataset.data_dir().join(file.path.as_str()),
                &file.file_size_bytes,
            )
            .await
            .unwrap();
        let reader = FileReader::try_open(
            source,
            None,
            Default::default(),
            &Arc::new(LanceCache::no_cache()),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        let pages = &reader.metadata().column_infos[0].page_infos;
        assert!(!pages.is_empty());
        for page in pages.iter() {
            let layout = match &page.encoding {
                PageEncoding::Structural(layout) => layout.clone(),
                PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
                PageEncoding::Legacy(_) => panic!("current-format fixture"),
            };
            match &layout.layout {
                Some(Layout::FullZipLayout(layout)) if is_raw => assert!(matches!(
                    layout.value_compression.as_ref().unwrap().compression,
                    Some(Compression::Variable(_))
                )),
                Some(Layout::MiniBlockLayout(layout)) if !is_raw => {
                    assert!(layout.dictionary.is_some());
                    assert!(layout.num_dictionary_items > 1);
                }
                _ => panic!("unexpected growth fixture layout: {layout:?}"),
            }
        }
    }
    (dataset, expected)
}

#[rstest]
#[tokio::test]
async fn dataset_cursor_dictionary_refusal_does_not_consume_rows(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
) {
    // Preserve the original counterexample: even one row needs the shared
    // dictionary, so halving the prefix cannot make this workspace sufficient.
    let (dataset, _) = growth_fixture(version, false).await;
    let budget = AllocationBudget::new(2048);
    let mut scan = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().clone(),
            Default::default(),
            16,
            2048,
            budget.clone(),
        )
        .unwrap();
    let position = scan.position();
    assert!(matches!(
        scan.try_next().await.unwrap(),
        Outcome::NeedsCapacity
    ));
    assert_eq!(scan.position(), position);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn dataset_cursor_recovers_from_repeated_growth_refusals(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
) {
    // A long wide region triggers repeated failed growth. The narrow tail must
    // still recover within the same fragment; permanently pinning the small
    // prefix would also reduce failed probes but is not an acceptable fix.
    let (dataset, expected) = growth_fixture(version, true).await;
    let budget = AllocationBudget::new(2048);
    let mut scan = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().clone(),
            Default::default(),
            16,
            2048,
            budget.clone(),
        )
        .unwrap();
    let mut delivered = 0;
    let mut recovered_before_boundary = false;
    loop {
        let (rows, batch) = match scan.try_next().await.unwrap() {
            Outcome::Batch { rows, batch, .. } => (rows, batch),
            Outcome::NeedsCapacity => panic!("a minimal raw row must fit the empty workspace"),
            Outcome::Finished => break,
        };
        assert_eq!(rows.start as usize, delivered % 1024);
        assert_eq!(batch, expected.slice(delivered, batch.num_rows()));
        if delivered < 496 {
            assert!(batch.num_rows() < 16, "wide rows cannot fit the ceiling");
        }
        if (512..1008).contains(&delivered) && batch.num_rows() == 16 {
            recovered_before_boundary = true;
        }
        if delivered == 1024 {
            assert_eq!(batch.num_rows(), 16, "new fragment resets the hint");
        }
        delivered += batch.num_rows();
        drop(batch);
        assert_eq!(budget.reserved(), 0);
    }
    assert_eq!(delivered, expected.num_rows());
    assert!(recovered_before_boundary, "growth backoff must stay finite");
    assert!(scan.position().is_none());
}

#[tokio::test]
async fn dataset_cursor_empty_snapshot_and_invalid_options() {
    let dataset = null_dataset(0).await;
    assert!(dataset.fragments().is_empty());
    let budget = AllocationBudget::new(64);
    let held = budget.try_reserve(64).unwrap();
    let mut scan = dataset
        .try_scan_admitted_fullzip(
            dataset.schema().clone(),
            Default::default(),
            1,
            64,
            budget.clone(),
        )
        .unwrap();
    assert_eq!(scan.position(), None);
    assert!(matches!(scan.try_next().await.unwrap(), Outcome::Finished));
    for (max_rows, bytes) in [(0, 64), (1, 0)] {
        let Err(error) = dataset.try_scan_admitted_fullzip(
            dataset.schema().clone(),
            Default::default(),
            max_rows,
            bytes,
            budget.clone(),
        ) else {
            panic!("invalid sizing must fail at construction");
        };
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("positive max_rows and workspace_bytes")
        );
    }
    let Err(error) = dataset.try_scan_admitted_fullzip(
        dataset.schema().clone(),
        FragReadConfig::default().with_make_deletions_null(true),
        1,
        64,
        budget.clone(),
    ) else {
        panic!("invalid marker configuration must fail even for an empty dataset");
    };
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("with_row_id or with_row_address")
    );
    drop(held);
    assert_eq!(budget.reserved(), 0);
}
