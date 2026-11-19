// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::session::Session;

async fn width_changes(version: LanceFileVersion) -> (Dataset, RecordBatch) {
    let values = (0..2048)
        .map(|row| vec![(row % 251) as u8; [128, 128, 1, 512][row / 512]])
        .collect::<Vec<_>>();
    let field = fullzip_field("payload", DataType::LargeBinary).with_metadata(HashMap::from([
        (
            "lance-encoding:structural-encoding".into(),
            "fullzip".into(),
        ),
        (DICT_DIVISOR_META_KEY.into(), u64::MAX.to_string()),
        (COMPRESSION_META_KEY.into(), "none".into()),
    ]));
    let schema = Arc::new(ArrowSchema::new(vec![field]));
    let expected = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(LargeBinaryArray::from_iter_values(
            values.iter().map(Vec::as_slice),
        ))],
    )
    .unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(expected.clone())], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 512,
            data_storage_version: Some(version),
            // The two scans must pay identical metadata I/O. A shared cache
            // would let whichever scan runs second reuse the first scan's
            // metadata, confounding the payload reread comparison.
            session: Some(Arc::new(Session::new(0, 0, Default::default()))),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 4);
    // Inspect actual stored layouts: hints alone do not establish the codec.
    let scheduler = ScanScheduler::new(dataset.object_store.clone(), SchedulerConfig::new(1024));
    for fragment in dataset.fragments().iter() {
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
        for page in reader.metadata().column_infos[0].page_infos.iter() {
            let layout = match &page.encoding {
                PageEncoding::Structural(layout) => layout.clone(),
                PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
                PageEncoding::Legacy(_) => panic!("current-format fixture"),
            };
            let Some(Layout::FullZipLayout(layout)) = &layout.layout else {
                panic!("expected stored FullZip: {layout:?}");
            };
            assert!(matches!(
                layout.value_compression.as_ref().unwrap().compression,
                Some(Compression::Variable(_))
            ));
        }
    }
    (dataset, expected)
}

#[rstest]
#[tokio::test]
async fn fragment_hint_reduces_rereads_and_recovers_across_width_changes(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] reset_first: bool,
) {
    let (dataset, expected) = width_changes(version).await;
    let mut second_fragment_bytes = [0; 2];
    // The control explicitly replans at boundaries, with the same final budget.
    // It therefore forgets its history through the public API. No timing gate
    // or assertion about private growth counters is needed.
    for reset_at_boundary in [reset_first, !reset_first] {
        let scheduler = ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
                use_lite_scheduler: Some(lite),
            },
        );
        let budget = AllocationBudget::new(2048);
        let mut scan = dataset
            .try_scan_admitted_fullzip(
                dataset.schema().clone(),
                FragReadConfig::default().with_scan_scheduler(scheduler.clone()),
                16,
                2048,
                budget.clone(),
            )
            .unwrap();
        for fragment in 0..4 {
            if reset_at_boundary && fragment != 0 {
                scan.set_workspace_bytes(1024).unwrap();
                scan.set_workspace_bytes(2048).unwrap();
            }
            let before = scheduler.stats().bytes_read;
            let mut delivered = 0;
            let mut largest = 0;
            while delivered < 512 {
                let Outcome::Batch {
                    fragment_id,
                    rows,
                    batch,
                } = scan.next_when_available().await.unwrap()
                else {
                    panic!("a minimal raw row fits the original workspace");
                };
                assert_eq!(fragment_id as usize, fragment);
                assert_eq!(rows.start as usize, delivered);
                assert_eq!(
                    batch,
                    expected.slice(fragment * 512 + delivered, batch.num_rows())
                );
                largest = largest.max(batch.num_rows());
                delivered += batch.num_rows();
                drop(batch);
                assert_eq!(budget.reserved(), 0);
            }
            assert_eq!(delivered, 512);
            if fragment == 1 {
                second_fragment_bytes[usize::from(reset_at_boundary)] =
                    scheduler.stats().bytes_read - before;
            }
            if fragment == 2 {
                assert_eq!(largest, 16, "a narrower fragment must regain the ceiling");
            }
            if fragment == 3 {
                assert!(largest < 16, "a wider fragment must shrink again");
            }
        }
        assert!(matches!(scan.try_next().await.unwrap(), Outcome::Finished));
        assert_eq!(budget.reserved(), 0);
    }
    assert!(
        second_fragment_bytes[0] < second_fragment_bytes[1],
        "carried history must avoid repeated input from restarting probes: {second_fragment_bytes:?}"
    );
}
