// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::fragment::FragmentReader;
use arrow_array::{LargeBinaryArray, ListArray, StringArray, UInt32Array};
use arrow_buffer::OffsetBuffer;
use lance_core::{Error, cache::LanceCache, utils::allocation::AllocationBudget};
use lance_encoding::{decoder::PageEncoding, format::metadata::page_layout::Layout};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use std::collections::HashMap;

use super::*;

fn payload(prefix: &str, rows: usize, nested: bool, null_at: usize) -> ArrayRef {
    let values = (0..rows)
        .map(|row| format!("{prefix}{row:03}").repeat(24))
        .collect::<Vec<_>>();
    let leaf: ArrayRef =
        Arc::new(LargeBinaryArray::from_iter(values.iter().enumerate().map(
            |(row, value)| (row != null_at || nested).then_some(value.as_bytes()),
        )));
    if nested {
        Arc::new(ListArray::new(
            Arc::new(ArrowField::new("item", DataType::LargeBinary, true)),
            OffsetBuffer::new((0..=rows as i32).collect::<Vec<_>>().into()),
            leaf,
            Some(NullBuffer::from(
                (0..rows).map(|row| row != null_at).collect::<Vec<_>>(),
            )),
        ))
    } else {
        leaf
    }
}

fn fullzip_field(name: &str, data_type: DataType) -> ArrowField {
    ArrowField::new(name, data_type, true).with_metadata(HashMap::from([(
        "lance-encoding:structural-encoding".into(),
        "fullzip".into(),
    )]))
}

async fn open(
    dataset: &Dataset,
    fragment_id: usize,
    projection: &Schema,
    lite: bool,
) -> FragmentReader {
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig {
            io_buffer_size_bytes: 1024,
            use_lite_scheduler: Some(lite),
        },
    );
    dataset
        .get_fragment(fragment_id)
        .unwrap()
        .open(
            projection,
            FragReadConfig::default().with_scan_scheduler(scheduler),
        )
        .await
        .unwrap()
}

async fn write_payload_dataset(version: LanceFileVersion, nested: bool) -> Dataset {
    let values = payload("base", 12, nested, 1);
    let schema = Arc::new(ArrowSchema::new(vec![
        fullzip_field("payload", values.data_type().clone()),
        fullzip_field("untouched", DataType::Utf8),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            values,
            Arc::new(StringArray::from_iter_values(
                (0..12).map(|row| format!("other{row}")),
            )),
        ],
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 6,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

// Field metadata is a dense-encoding preference, not a ban on Constant layout.
// Check the stored fixture layout so admitted-path coverage cannot silently drift.
async fn assert_overlay_layout(dataset: &Dataset, name: &str, is_constant: bool) {
    let scheduler = ScanScheduler::new(dataset.object_store.clone(), SchedulerConfig::new(1024));
    let file = scheduler
        .open_file(
            &dataset.data_dir().join(format!("{name}.lance")),
            &CachedFileSize::unknown(),
        )
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
    let pages = &reader.metadata().column_infos[0].page_infos;
    assert_eq!(pages.len(), 1);
    let layout = match &pages[0].encoding {
        PageEncoding::Structural(layout) => layout.clone(),
        PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
        PageEncoding::Legacy(_) => panic!("current-format fixture must not use legacy pages"),
    };
    if is_constant {
        assert!(matches!(layout.layout, Some(Layout::ConstantLayout(_))));
    } else {
        assert!(matches!(layout.layout, Some(Layout::FullZipLayout(_))));
    }
}

#[rstest]
#[tokio::test]
async fn admitted_fragment_take_matches_actual_overlay_resolution(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] nested: bool,
) {
    const WORKSPACE: usize = 65536;
    let mut dataset = write_payload_dataset(version, nested).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("admitted-old-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([0, 2, 4])),
            vec![payload("old", 3, nested, usize::MAX)],
            version,
        )
        .await;
        dataset = commit_overlay(
            dataset,
            &format!("admitted-new-{fragment}"),
            fragment,
            &[field_id],
            // Two distinct non-null values keep v2.2's earlier Constant encoder
            // from intercepting this dense-FullZip fixture. Null still wins at row 2.
            OverlayCoverage::dense(bitmap([2, 3, 4])),
            vec![payload("new", 3, nested, 0)],
            version,
        )
        .await;
        assert_overlay_layout(&dataset, &format!("admitted-old-{fragment}"), false).await;
        assert_overlay_layout(&dataset, &format!("admitted-new-{fragment}"), false).await;
    }
    assert_eq!(dataset.get_fragments().len(), 2);
    for fragment in 0..2 {
        let reader = open(&dataset, fragment, dataset.schema(), lite).await;
        for offsets in [vec![], vec![0, 1, 2, 4, 5], vec![1, 5]] {
            let expected = reader.take_as_batch(&offsets, None).await.unwrap();
            let indices = UInt32Array::from(offsets);
            let budget = AllocationBudget::new(WORKSPACE);
            if !indices.is_empty() {
                // Include nonzero shortages; no partial result or leaked owner is accepted.
                for available in [0, 32, 128] {
                    let refused = reader
                        .try_take_admitted_fullzip(&indices, budget.try_reserve(available).unwrap())
                        .await
                        .unwrap();
                    assert!(
                        refused.is_none(),
                        "no ordinary fallback on admission refusal"
                    );
                    assert_eq!(budget.reserved(), 0);
                }
            }
            let output = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(WORKSPACE).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(output, expected);
            if !indices.is_empty() {
                let alias = output.column(0).to_data().buffers()[0].slice_with_length(0, 1);
                drop(output);
                assert!(
                    budget.reserved() > 1,
                    "one byte retains its full physical allocation"
                );
                assert!(budget.try_reserve(WORKSPACE).is_none());
                drop(alias);
            } else {
                drop(output);
            }
            assert_eq!(budget.reserved(), 0);
            let output = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(WORKSPACE).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                output, expected,
                "same fragment makes progress after its consumer drains"
            );
            drop(output);
            assert_eq!(budget.reserved(), 0);
        }
        // Both operations own their complete workspace before dependent reads.
        // CI uses a single CPU worker: assembly must never wait for more credit.
        let budget = AllocationBudget::new(2 * WORKSPACE);
        let indices = UInt32Array::from(vec![0, 2, 5]);
        let expected = reader.take_as_batch(&[0, 2, 5], None).await.unwrap();
        let left =
            reader.try_take_admitted_fullzip(&indices, budget.try_reserve(WORKSPACE).unwrap());
        let right =
            reader.try_take_admitted_fullzip(&indices, budget.try_reserve(WORKSPACE).unwrap());
        let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(left, right)
        })
        .await
        .expect("finite concurrent reads must make progress");
        let left = left.unwrap().unwrap();
        let right = right.unwrap().unwrap();
        assert_eq!(left, expected);
        assert_eq!(right, expected);
        drop((left, right));
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[tokio::test]
async fn admitted_fragment_uses_projected_nonzero_column_metadata(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
) {
    let mut fields = (0..7)
        .map(|column| ArrowField::new(format!("unselected{column}"), DataType::Int32, false))
        .collect::<Vec<_>>();
    fields.push(fullzip_field("payload", DataType::LargeBinary));
    let schema = Arc::new(ArrowSchema::new(fields));
    let mut columns: Vec<ArrayRef> = (0..7)
        .map(|_| Arc::new(Int32Array::from_iter_values(0..12)) as ArrayRef)
        .collect();
    columns.push(payload("projected", 12, false, 2));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 6,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let projection = dataset.schema().project(&["payload"]).unwrap();
    // The actual fragment opener chooses indexed metadata for 1/8 columns. Column
    // 7 must be remapped by the version-specific projection path, not loaded as0.
    for fragment in 0..2 {
        let reader = open(&dataset, fragment, &projection, false).await;
        let expected = reader.take_as_batch(&[0, 2, 5], None).await.unwrap();
        let budget = AllocationBudget::new(16384);
        let output = reader
            .try_take_admitted_fullzip(
                &UInt32Array::from(vec![0, 2, 5]),
                budget.try_reserve(16384).unwrap(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(output, expected);
        drop(output);
        assert_eq!(budget.reserved(), 0);
    }
}

#[tokio::test]
async fn admitted_fragment_rejects_invalid_or_unsupported_requests_without_fallback() {
    let dataset = create_base_dataset(LanceFileVersion::V2_2).await;
    let reader = open(&dataset, 0, dataset.schema(), false).await;
    let budget = AllocationBudget::new(8192);
    for indices in [
        UInt32Array::from(vec![Some(0), None]),
        UInt32Array::from(vec![1, 0]),
        UInt32Array::from(vec![1, 1]),
        UInt32Array::from(vec![6]),
    ] {
        let error = reader
            .try_take_admitted_fullzip(&indices, budget.try_reserve(8192).unwrap())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("sorted unique"));
        assert_eq!(budget.reserved(), 0);
    }
    let error = reader
        .try_take_admitted_fullzip(
            &UInt32Array::from(vec![0]),
            budget.try_reserve(8192).unwrap(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("binary/string leaves"));
    assert_eq!(budget.reserved(), 0);
    let marker_reader = dataset
        .get_fragment(0)
        .unwrap()
        .open(
            dataset.schema(),
            FragReadConfig::default().with_row_id(true),
        )
        .await
        .unwrap();
    let error = marker_reader
        .try_take_admitted_fullzip(
            &UInt32Array::from(vec![0]),
            budget.try_reserve(8192).unwrap(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("generated columns"));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn admitted_fragment_reads_constant_overlay_in_shared_workspace(
    #[values(false, true)] lite: bool,
    #[values(false, true)] nested: bool,
) {
    let version = LanceFileVersion::V2_2;
    let mut dataset = write_payload_dataset(version, nested).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        let name = format!("constant-{fragment}");
        dataset = commit_overlay(
            dataset,
            &name,
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([2, 3])),
            vec![payload("constant", 2, nested, 0)],
            version,
        )
        .await;
        assert_overlay_layout(&dataset, &name, true).await;
    }
    for fragment in 0..2 {
        let reader = open(&dataset, fragment, dataset.schema(), lite).await;
        let ordinary = reader.take_as_batch(&[2, 3], None).await.unwrap();
        assert_eq!(ordinary.num_rows(), 2);
        assert!(ordinary["payload"].is_null(0));
        assert!(ordinary["payload"].is_valid(1));
        let budget = AllocationBudget::new(2 * 65536);
        // This formerly rejected real Constant layout now uses the same admitted
        // operation as base reads and overlay assembly. Small credit is refusal,
        // not a fallback to an ordinary decoder or partial result.
        assert!(
            reader
                .try_take_admitted_fullzip(
                    &UInt32Array::from(vec![2, 3]),
                    budget.try_reserve(16).unwrap()
                )
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(budget.reserved(), 0);
        let output = reader
            .try_take_admitted_fullzip(
                &UInt32Array::from(vec![2, 3]),
                budget.try_reserve(65536).unwrap(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(output, ordinary);
        drop(output);
        assert_eq!(budget.reserved(), 0);
        let indices = UInt32Array::from(vec![2, 3]);
        let left = reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap());
        let right = reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap());
        let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(left, right)
        })
        .await
        .expect("constant reads must progress with one CPU worker");
        let left = left.unwrap().unwrap();
        let right = right.unwrap().unwrap();
        assert_eq!(left, ordinary);
        assert_eq!(right, ordinary);
        let alias = left.column(0).to_data().buffers()[0].slice_with_length(0, 1);
        drop((left, right));
        assert!(budget.reserved() > 0);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
        // Non-overlapping overlays remain pruned by the actual planner.
        let expected = reader.take_as_batch(&[0, 5], None).await.unwrap();
        let output = reader
            .try_take_admitted_fullzip(
                &UInt32Array::from(vec![0, 5]),
                budget.try_reserve(65536).unwrap(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(output, expected);
        drop(output);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[tokio::test]
async fn admitted_fragment_null_overlay_wins_over_older_values(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    let mut dataset = write_payload_dataset(version, false).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("older-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([1, 2])),
            vec![payload("old", 2, false, usize::MAX)],
            version,
        )
        .await;
        dataset = commit_overlay(
            dataset,
            &format!("null-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([2, 3, 4])),
            vec![Arc::new(LargeBinaryArray::new_null(3))],
            version,
        )
        .await;
    }
    for fragment in 0..2 {
        let reader = open(&dataset, fragment, dataset.schema(), lite).await;
        let expected = reader.take_as_batch(&[0, 1, 2, 4, 5], None).await.unwrap();
        assert!(expected["payload"].is_valid(1));
        assert!(expected["payload"].is_null(2));
        assert!(expected["payload"].is_null(3));
        let indices = UInt32Array::from(vec![0, 1, 2, 4, 5]);
        let budget = AllocationBudget::new(2 * 65536);
        assert!(
            reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(16).unwrap())
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(budget.reserved(), 0);
        let left = reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap());
        let right = reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap());
        let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(left, right)
        })
        .await
        .expect("null overlay reads must progress with one CPU worker");
        let left = left.unwrap().unwrap();
        let right = right.unwrap().unwrap();
        assert_eq!(left, expected);
        assert_eq!(right, expected);
        let alias = left["payload"].nulls().unwrap().clone();
        drop((left, right));
        assert!(budget.reserved() > 0);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[tokio::test]
async fn admitted_fragment_empty_list_overlay_preserves_null_precedence(
    #[values(false, true)] lite: bool,
) {
    let version = LanceFileVersion::V2_3;
    let mut dataset = write_payload_dataset(version, true).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    let original = payload("base", 1, true, usize::MAX);
    let DataType::List(field) = original.data_type() else {
        unreachable!()
    };
    let empty: ArrayRef = Arc::new(ListArray::new(
        field.clone(),
        OffsetBuffer::new(vec![0_i32; 4].into()),
        arrow_array::new_empty_array(field.data_type()),
        Some(NullBuffer::from(vec![true, false, true])),
    ));
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("old-list-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([1, 2, 3, 4])),
            vec![payload("old", 4, true, usize::MAX)],
            version,
        )
        .await;
        dataset = commit_overlay(
            dataset,
            &format!("empty-list-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([2, 3, 4])),
            vec![empty.clone()],
            version,
        )
        .await;
    }
    for fragment in 0..2 {
        let reader = open(&dataset, fragment, dataset.schema(), lite).await;
        let expected = reader
            .take_as_batch(&[0, 1, 2, 3, 4, 5], None)
            .await
            .unwrap();
        let values = expected["payload"]
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(values.slice(2, 3).to_data(), empty.to_data());
        assert!(values.value_length(1) > 0);
        let indices = UInt32Array::from(vec![0, 1, 2, 3, 4, 5]);
        let budget = AllocationBudget::new(2 * 65536);
        assert!(
            reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(16).unwrap())
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(budget.reserved(), 0);
        let left = reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap());
        let right = reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap());
        let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(left, right)
        })
        .await
        .expect("empty List overlay must progress with one CPU worker");
        let left = left.unwrap().unwrap();
        let right = right.unwrap().unwrap();
        assert_eq!(left, expected);
        assert_eq!(right, expected);
        let alias = left["payload"].nulls().unwrap().clone();
        drop((left, right));
        assert!(budget.reserved() > 0);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[tokio::test]
async fn admitted_fragment_nonempty_null_children_overlay_preserves_parent_validity(
    #[values(false, true)] lite: bool,
    #[values(false, true)] payload_counts: bool,
) {
    let version = LanceFileVersion::V2_3;
    let mut dataset = write_payload_dataset(version, true).await;
    let field_id = dataset.schema().field("payload").unwrap().id;
    let original = payload("base", 1, true, usize::MAX);
    let DataType::List(field) = original.data_type() else {
        unreachable!()
    };
    let (offsets, valid, covered) = if payload_counts {
        (vec![0_i32, 1, 4, 6, 10], vec![true; 4], vec![2, 3, 4, 5])
    } else {
        (vec![0_i32, 1, 1, 3], vec![true, false, true], vec![2, 3, 4])
    };
    let null_children: ArrayRef = Arc::new(ListArray::new(
        field.clone(),
        OffsetBuffer::new(offsets.clone().into()),
        arrow_array::new_null_array(field.data_type(), *offsets.last().unwrap() as usize),
        Some(NullBuffer::from(valid)),
    ));
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("old-null-child-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap([1, 2, 3, 4])),
            vec![payload("old", 4, true, usize::MAX)],
            version,
        )
        .await;
        dataset = commit_overlay(
            dataset,
            &format!("null-child-{fragment}"),
            fragment,
            &[field_id],
            OverlayCoverage::dense(bitmap(covered.clone())),
            vec![null_children.clone()],
            version,
        )
        .await;
    }
    for fragment in 0..2 {
        let reader = open(&dataset, fragment, dataset.schema(), lite).await;
        let expected = reader
            .take_as_batch(&[0, 1, 2, 3, 4, 5], None)
            .await
            .unwrap();
        let values = expected["payload"]
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(
            values.slice(2, covered.len()).to_data(),
            null_children.to_data()
        );
        assert!(values.is_valid(2));
        assert_eq!(values.is_null(3), !payload_counts);
        assert_eq!(values.value_length(4), 2);
        let indices = UInt32Array::from(vec![0, 1, 2, 3, 4, 5]);
        let budget = AllocationBudget::new(2 * 65536);
        assert!(
            reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(16).unwrap())
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(budget.reserved(), 0);
        let left = reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap());
        let right = reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap());
        let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(left, right)
        })
        .await
        .expect("null-child overlays must progress with one CPU worker");
        let left = left.unwrap().unwrap();
        let right = right.unwrap().unwrap();
        assert_eq!(left, expected);
        assert_eq!(right, expected);
        let alias = left["payload"].to_data().child_data()[0]
            .nulls()
            .unwrap()
            .clone();
        drop((left, right));
        assert!(budget.reserved() > 0);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}
