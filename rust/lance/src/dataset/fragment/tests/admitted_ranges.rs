// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{cache::LanceCache, utils::allocation::AllocationBudget};
use lance_file::reader::{
    FileReader, FullZipScanOutcome, PhysicalRowIdSource, PhysicalRowVersionSource,
};
use pretty_assertions::assert_eq;

use super::*;

#[rstest]
#[tokio::test]
async fn admitted_logical_range_matches_fragment_markers(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] stable_ids: bool,
) {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("payload", DataType::Utf8, false).with_metadata(HashMap::from([(
            "lance-encoding:structural-encoding".to_string(),
            "fullzip".to_string(),
        )])),
    ]));
    let values = StringArray::from_iter_values((0..40).map(|row| format!("r{row:02}")));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 20,
            data_storage_version: Some(version),
            enable_stable_row_ids: stable_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let offsets = [0_u64, 1, 6, 7, 11, 19];
    let deleted_values = (0..2)
        .flat_map(|fragment| offsets.map(move |row| format!("'r{:02}'", fragment * 20 + row)))
        .collect::<Vec<_>>()
        .join(", ");
    dataset
        .delete(&format!("payload IN ({deleted_values})"))
        .await
        .unwrap();
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    let live = (0..20_u64)
        .filter(|row| !offsets.contains(row))
        .collect::<Vec<_>>();
    for fragment in fragments {
        let ordinary = fragment
            .open(
                dataset.schema(),
                FragReadConfig::default()
                    .with_row_id(true)
                    .with_row_address(true)
                    .with_row_created_at_version(true)
                    .with_row_last_updated_at_version(true)
                    .with_make_deletions_null(true),
            )
            .await
            .unwrap();
        assert_eq!(fragment.metadata.files.len(), 1);
        let path = dataset
            .data_dir()
            .join(fragment.metadata.files[0].path.as_str());
        let scheduler = ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
                use_lite_scheduler: Some(lite),
            },
        );
        let file = scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file,
            None,
            Default::default(),
            &Arc::new(LanceCache::no_cache()),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(reader.num_rows(), 20);
        let budget = AllocationBudget::new(65536);
        let mut admitted = reader
            .open_admitted_fullzip_scan(0..0, budget.clone(), 65536, 3)
            .await
            .unwrap()
            .with_row_id(
                fragment.id() as u32,
                ordinary
                    .row_id_sequence()
                    .await
                    .unwrap()
                    .map(|ids| ids as Arc<dyn PhysicalRowIdSource>),
            )
            .unwrap()
            .with_row_address(fragment.id() as u32)
            .unwrap()
            .with_row_created_at_version(
                ordinary
                    .created_at_sequence
                    .clone()
                    .map(|versions| versions as Arc<dyn PhysicalRowVersionSource>),
            )
            .unwrap()
            .with_row_last_updated_at_version(
                ordinary
                    .last_updated_at_sequence
                    .clone()
                    .map(|versions| versions as Arc<dyn PhysicalRowVersionSource>),
            )
            .unwrap()
            .with_deleted_rows_as_nulls(ordinary.deletion_vec.clone().unwrap())
            .unwrap();
        for selection in [0..7_u64, 7..10, live.len() as u64..live.len() as u64] {
            let reference = ordinary
                .read_range(selection.start as u32..selection.end as u32, 2)
                .await
                .unwrap()
                .buffered(2)
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let reference =
                concat_batches(&Arc::new(ordinary.output_schema.clone()), &reference).unwrap();
            let expected_rows = if selection.is_empty() {
                20..20
            } else {
                live[selection.start as usize]..live[selection.end as usize - 1] + 1
            };
            assert_eq!(
                reference.num_rows() as u64,
                expected_rows.end - expected_rows.start
            );
            if selection.start == 0 {
                assert!(
                    reference.num_rows() as u64 > selection.end - selection.start,
                    "interior deleted rows must distinguish markers from live filtering"
                );
            }
            admitted.advance_to_live_range(selection.clone()).unwrap();
            let mut position = expected_rows.start;
            let mut output_rows = 0;
            let mut valid_ids = 0;
            loop {
                match admitted.try_next().await.unwrap() {
                    FullZipScanOutcome::Batch { rows, batch } => {
                        assert_eq!(rows.start, position);
                        assert!(rows.end <= expected_rows.end);
                        assert_eq!(batch, reference.slice(output_rows, batch.num_rows()));
                        position = rows.end;
                        output_rows += batch.num_rows();
                        valid_ids += batch.num_rows() - batch[ROW_ID].null_count();
                        let alias = batch[ROW_ID].slice(0, 1);
                        drop(batch);
                        assert!(budget.reserved() > 0 && budget.reserved() < 65536);
                        if position < expected_rows.end {
                            assert!(matches!(
                                admitted.try_next().await.unwrap(),
                                FullZipScanOutcome::NeedsCapacity
                            ));
                            assert_eq!(admitted.position(), position);
                        }
                        drop(alias);
                        assert_eq!(budget.reserved(), 0);
                    }
                    FullZipScanOutcome::Finished => break,
                    FullZipScanOutcome::NeedsCapacity => {
                        panic!("small fragment range must fit after drain")
                    }
                }
            }
            assert_eq!(position, expected_rows.end);
            assert_eq!(output_rows, reference.num_rows());
            assert_eq!(valid_ids as u64, selection.end - selection.start);
        }
    }
}
