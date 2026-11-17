// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::AdmittedDatasetScan;
use lance_encoding::{
    constants::{COMPRESSION_META_KEY, DICT_DIVISOR_META_KEY, STRUCTURAL_ENCODING_META_KEY},
    format::metadata::{CompressionScheme, compressive_encoding::Compression},
};
use std::{task::Poll, time::Duration};

const ROWS: usize = 128;
const WORKSPACE: usize = 128 * 1024;

fn metadata() -> HashMap<String, String> {
    HashMap::from([
        (STRUCTURAL_ENCODING_META_KEY.into(), "miniblock".into()),
        // Index compression is independent of dictionary-value compression.
        // Keep indices raw; leave the dictionary's default LZ4 policy intact.
        (COMPRESSION_META_KEY.into(), "none".into()),
        (DICT_DIVISOR_META_KEY.into(), "1".into()),
    ])
}

fn values(prefix: &str, rows: usize, nested: bool) -> ArrayRef {
    let mut offsets = vec![0_i32];
    let mut items = Vec::new();
    for row in 0..rows {
        let count = if nested {
            if row % 23 == 3 || row % 19 == 7 { 0 } else { 2 }
        } else {
            1
        };
        for child in 0..count {
            let index = row * 2 + child;
            items.push(if index % 17 == 3 || (!nested && row % 23 == 3) {
                None
            } else if index % 19 == 7 {
                Some(Vec::new())
            } else {
                Some(
                    format!("{prefix}-{:02}-", index % 31)
                        .repeat(128 + index % 31)
                        .into_bytes(),
                )
            });
        }
        offsets.push(items.len() as i32);
    }
    let leaf = Arc::new(LargeBinaryArray::from_iter(
        items.iter().map(|v| v.as_deref()),
    )) as ArrayRef;
    if nested {
        Arc::new(ListArray::new(
            Arc::new(
                ArrowField::new("item", DataType::LargeBinary, true).with_metadata(metadata()),
            ),
            OffsetBuffer::new(offsets.into()),
            leaf,
            Some(NullBuffer::from(
                (0..rows).map(|row| row % 23 != 3).collect::<Vec<_>>(),
            )),
        ))
    } else {
        leaf
    }
}

async fn assert_dictionary(dataset: &Dataset, file: &DataFile, field: i32, nested: bool) {
    let scheduler = ScanScheduler::new(dataset.object_store.clone(), SchedulerConfig::new(1024));
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
    let index = file.fields.iter().position(|id| *id == field).unwrap();
    let column = file.column_indices[index] as usize;
    let pages = &reader.metadata().column_infos[column].page_infos;
    assert!(!pages.is_empty());
    for page in pages.iter() {
        let layout = match &page.encoding {
            PageEncoding::Structural(layout) => layout.clone(),
            PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
            PageEncoding::Legacy(_) => panic!("current writer fixture"),
        };
        let Some(Layout::MiniBlockLayout(layout)) = layout.layout else {
            panic!("stored MiniBlock required");
        };
        let Some(Compression::General(general)) = layout
            .dictionary
            .as_ref()
            .and_then(|d| d.compression.as_ref())
        else {
            panic!("stored compressed dictionary required");
        };
        assert_eq!(
            general.compression.as_ref().unwrap().scheme(),
            CompressionScheme::CompressionAlgorithmLz4
        );
        assert!(layout.num_dictionary_items > 1);
        assert_eq!(layout.repetition_index_depth, u32::from(nested));
    }
}

fn deleted(id: usize) -> bool {
    id % 29 == 11 || id == ROWS - 1 || id == ROWS + 1
}

async fn fixture(version: LanceFileVersion, nested: bool) -> (Dataset, Vec<ArrayRef>) {
    let base = values("base", 2 * ROWS, nested);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("key", DataType::Int32, false),
        ArrowField::new("payload", base.data_type().clone(), true).with_metadata(metadata()),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..(2 * ROWS) as i32)),
            base.clone(),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: ROWS,
            enable_stable_row_ids: true,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    let field = dataset.schema().field("payload").unwrap().id;
    let old_offsets = (0..ROWS as u32).step_by(2).collect::<Vec<_>>();
    let new_offsets = (16..112).step_by(2).collect::<Vec<u32>>();
    let old = values("old", old_offsets.len(), nested);
    let new = values("new", new_offsets.len(), nested);
    let mut expected = Vec::new();
    for fragment in 0..2 {
        for (label, offsets, payload) in [("old", &old_offsets, &old), ("new", &new_offsets, &new)]
        {
            dataset = commit_overlay(
                dataset,
                &format!("dictionary-{label}-{fragment}"),
                fragment as u64,
                &[field],
                OverlayCoverage::dense(bitmap(offsets.iter().copied())),
                vec![payload.clone()],
                version,
            )
            .await;
        }
        let original = base.slice(fragment * ROWS, ROWS);
        let routing = (0..ROWS as u32)
            .map(|row| {
                if let Ok(index) = new_offsets.binary_search(&row) {
                    (2, index)
                } else if let Ok(index) = old_offsets.binary_search(&row) {
                    (1, index)
                } else {
                    (0, row as usize)
                }
            })
            .collect::<Vec<_>>();
        expected.push(
            arrow_select::interleave::interleave(
                &[original.as_ref(), old.as_ref(), new.as_ref()],
                &routing,
            )
            .unwrap(),
        );
    }
    for fragment in dataset.iter_fragments() {
        for file in &fragment.files {
            assert_dictionary(&dataset, file, field, nested).await;
        }
        for overlay in &fragment.overlays {
            assert_dictionary(&dataset, &overlay.data_file, field, nested).await;
        }
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
        .delete("key % 29 = 11 OR key IN (127, 129)")
        .await
        .unwrap();
    Arc::make_mut(&mut Arc::make_mut(&mut dataset.manifest).fragments).reverse();
    (dataset, expected)
}

fn payload_bytes(array: &dyn Array) -> usize {
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        (0..list.len())
            .filter(|row| !list.is_null(*row))
            .map(|row| payload_bytes(list.value(row).as_ref()))
            .sum()
    } else {
        let bytes = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        (0..bytes.len())
            .filter(|row| !bytes.is_null(*row))
            .map(|row| bytes.value(row).len())
            .sum()
    }
}

#[rstest]
#[tokio::test]
async fn dataset_dictionary_overlay_deletion_and_shared_credit(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] nested: bool,
) {
    let (dataset, expected) = fixture(version, nested).await;
    let projection = dataset.schema().project(&["payload", "late"]).unwrap();
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
    let budget = AllocationBudget::new(WORKSPACE);
    let new_scan = |bytes| {
        dataset
            .try_scan_admitted_fullzip(
                projection.clone(),
                config.clone(),
                17,
                bytes,
                budget.clone(),
            )
            .unwrap()
    };
    let mut refused = new_scan(64);
    let position = refused.position();
    assert!(matches!(
        refused.try_next().await.unwrap(),
        Outcome::NeedsCapacity
    ));
    assert_eq!(refused.position(), position);
    assert_eq!(budget.reserved(), 0);
    refused.set_workspace_bytes(WORKSPACE).unwrap();
    let Outcome::Batch { batch, .. } = refused.try_next().await.unwrap() else {
        panic!("minimal prefix must fit after replanning");
    };
    let alias = batch["payload"].to_data().buffers()[0].slice_with_length(0, 1);
    drop((batch, refused));
    assert!(budget.reserved() > 1 && budget.reserved() <= WORKSPACE);
    let mut blocked = new_scan(WORKSPACE);
    let position = blocked.position();
    assert!(matches!(
        blocked.try_next().await.unwrap(),
        Outcome::NeedsCapacity
    ));
    assert_eq!(blocked.position(), position);
    let mut cancelled = Box::pin(blocked.next_when_available());
    assert!(matches!(futures::poll!(cancelled.as_mut()), Poll::Pending));
    drop(cancelled);
    assert_eq!(blocked.position(), position);
    drop(alias);
    let Outcome::Batch { batch, .. } = blocked.next_when_available().await.unwrap() else {
        panic!("final alias release must restore progress");
    };
    drop((batch, blocked));
    assert_eq!(budget.reserved(), 0);

    let mut readers = Vec::with_capacity(2);
    for id in 0..2 {
        readers.push(
            dataset
                .get_fragment(id)
                .unwrap()
                .open(&projection, config.clone())
                .await
                .unwrap(),
        );
    }
    let drain = |mut scan: AdmittedDatasetScan| {
        let (readers, expected, budget) = (&readers, &expected, &budget);
        async move {
            let (mut delivered, mut bytes) = (0, 0);
            for id in [1, 0] {
                let mut start = 0;
                while start < ROWS as u64 {
                    assert_eq!(
                        scan.position(),
                        Some(AdmittedScanPosition {
                            fragment_id: id as u64,
                            row_offset: start,
                        })
                    );
                    let Outcome::Batch {
                        fragment_id,
                        rows,
                        batch,
                    } = scan.next_when_available().await.unwrap()
                    else {
                        panic!("finite dictionary prefix must fit the existing workspace");
                    };
                    assert_eq!(fragment_id, id as u64);
                    assert_eq!(rows.start, start);
                    assert!(rows.end > start && rows.end <= ROWS as u64);
                    let offsets = rows.clone().map(|row| row as u32).collect::<Vec<_>>();
                    let oracle = readers[id].take_as_batch(&offsets, None).await.unwrap();
                    assert_eq!(batch, oracle);
                    let live = UInt32Array::from(
                        offsets
                            .into_iter()
                            .filter(|row| !deleted(id * ROWS + *row as usize))
                            .collect::<Vec<_>>(),
                    );
                    let payload =
                        arrow_select::take::take(expected[id].as_ref(), &live, None).unwrap();
                    assert_eq!(batch["payload"].to_data(), payload.to_data());
                    assert_eq!(batch.num_rows(), live.len());
                    assert_eq!(batch["late"].null_count(), batch.num_rows());
                    delivered += batch.num_rows();
                    bytes += payload_bytes(batch["payload"].as_ref());
                    start = rows.end;
                    assert!(budget.reserved() <= WORKSPACE);
                    // Another scan must wait outside the CPU worker while the
                    // consumer owns this output, even across fragment boundaries.
                    tokio::task::yield_now().await;
                    drop(batch);
                }
            }
            assert!(matches!(
                scan.next_when_available().await.unwrap(),
                Outcome::Finished
            ));
            assert_eq!(scan.position(), None);
            assert_eq!(delivered, (0..2 * ROWS).filter(|id| !deleted(*id)).count());
            assert!(
                bytes > WORKSPACE,
                "cumulative output must exceed the fixed shared pool"
            );
        }
    };
    tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(drain(new_scan(WORKSPACE)), drain(new_scan(WORKSPACE)));
    })
    .await
    .expect("both consumers must finish with one workspace and one CPU worker");
    assert_eq!(budget.reserved(), 0);
}
