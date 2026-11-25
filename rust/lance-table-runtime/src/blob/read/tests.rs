// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

use crate::blob::file::shared_blob_source;
use crate::blob::locate::collect_blob_files_v1;
use crate::blob::test_util::{
    gated_range_store, recording_range_store, recording_range_store_with_url,
};
use crate::session::UnregisteredExternalBlobPolicy;
use crate::test_util::{publish_manifest, write_file};
use arrow_array::{
    ArrayRef, LargeBinaryArray, RecordBatch, StringArray, StructArray, UInt8Array, UInt32Array,
    UInt64Array,
};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_META_KEY, BLOB_V2_EXT_NAME};
use lance_core::datatypes::BlobKind;
use lance_core::datatypes::{BLOB_V2_LOGICAL_TYPE, BLOB_V2_PREPARED_FIELDS, Schema};
use lance_core::utils::deletion::DeletionVector;
use lance_file::version::ConcreteFileVersion;
use lance_table::format::{DataStorageFormat, Fragment, Manifest, RowIdMeta};
use lance_table::io::deletion::write_deletion_file;
use lance_table::rowids::{RowIdSequence, write_row_ids};
use object_store::path::Path;
use rstest::rstest;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[tokio::test]
async fn test_read_blob_execution_shares_scheduler_across_paths() {
    let store = Arc::new(ObjectStore::memory());
    let other_store = Arc::new(ObjectStore::memory());
    assert_eq!(store.store_prefix, other_store.store_prefix);
    let execution = ReadBlobsExecution::new(Some(1024));
    let first = execution.scheduler_for(&BlobSource::new(
        store.clone(),
        Path::from("blobs/first.bin"),
    ));
    for index in 0..4 {
        let source = BlobSource::new(store.clone(), Path::from(format!("blobs/{index}.bin")));
        assert!(Arc::ptr_eq(&first, &execution.scheduler_for(&source)));
    }
    assert_eq!(execution.num_schedulers(), 1);
    let other =
        execution.scheduler_for(&BlobSource::new(other_store, Path::from("blobs/first.bin")));
    assert!(!Arc::ptr_eq(&first, &other));
    assert_eq!(execution.num_schedulers(), 2);
}

#[rstest]
#[case::whole(false)]
#[case::ranges(true)]
#[tokio::test]
async fn test_read_blob_files_isolate_store_instances(#[case] ranges: bool) {
    let (first_store, first_inner) = recording_range_store(Bytes::from_static(b"abcdef"));
    let (second_store, second_inner) = recording_range_store(Bytes::from_static(b"uvwxyz"));
    assert_eq!(first_store.store_prefix, second_store.store_prefix);
    let path = Path::from("blobs/shared.bin");
    let mut cache = HashMap::new();
    let first_source = shared_blob_source(&mut cache, first_store.clone(), &path);
    let second_source = shared_blob_source(&mut cache, second_store, &path);
    assert!(!Arc::ptr_eq(&first_source, &second_source));
    assert!(Arc::ptr_eq(
        &first_source,
        &shared_blob_source(&mut cache, first_store.clone(), &path),
    ));
    let execution = ReadBlobsExecution::new(None);
    assert!(Arc::ptr_eq(
        &execution.scheduler_for(&first_source),
        &execution.scheduler_for(&first_source),
    ));
    assert!(!Arc::ptr_eq(
        &execution.scheduler_for(&first_source),
        &execution.scheduler_for(&second_source),
    ));
    let first = BlobFile::with_source(first_source, 1, 3, BlobKind::Packed, None);
    let second = BlobFile::with_source(second_source, 1, 3, BlobKind::Packed, None);
    // Independent source handles on the same store and path still coalesce.
    let duplicate = BlobFile::new_inline(first_store, path, 1, 3);
    let handles = [None, Some(&second), Some(&first), Some(&duplicate)];
    let builder = if ranges {
        read_blob_file_ranges(handles.map(|file| (file, BlobReadRange::new(0, 2))))
    } else {
        read_blob_files(handles)
    };
    let results = builder.execute().await.unwrap();
    let expected = if ranges {
        vec![
            None,
            Some(b"vw".as_slice()),
            Some(b"bc".as_slice()),
            Some(b"bc".as_slice()),
        ]
    } else {
        vec![
            None,
            Some(b"vwx".as_slice()),
            Some(b"bcd".as_slice()),
            Some(b"bcd".as_slice()),
        ]
    };
    assert_eq!(
        results
            .iter()
            .map(|blob| blob.data.as_deref())
            .collect::<Vec<_>>(),
        expected
    );
    let end = if ranges { 3 } else { 4 };
    assert_eq!(first_inner.requested_blob_ranges(), vec![1..end]);
    assert_eq!(second_inner.requested_blob_ranges(), vec![1..end]);
}

#[tokio::test]
async fn test_read_blob_files_empty_inputs_issue_no_io() {
    assert!(
        read_blob_files(Vec::<Option<&BlobFile>>::new())
            .execute()
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        read_blob_file_ranges(Vec::<(Option<&BlobFile>, BlobReadRange)>::new())
            .execute()
            .await
            .unwrap()
            .is_empty()
    );
    let results = read_blob_files([None::<&BlobFile>, None])
        .execute()
        .await
        .unwrap();
    assert_eq!(
        results,
        vec![
            ReadBlobFile {
                request_index: 0,
                data: None
            },
            ReadBlobFile {
                request_index: 1,
                data: None
            }
        ]
    );
    let results = read_blob_file_ranges([(None::<&BlobFile>, BlobReadRange::new(8, 5))])
        .execute()
        .await
        .unwrap();
    assert_eq!(
        results,
        vec![ReadBlobFile {
            request_index: 0,
            data: None
        }]
    );
    let (store, inner) = recording_range_store(Bytes::new());
    let empty = BlobFile::new_inline(store, Path::from("missing"), 0, 0);
    assert_eq!(
        read_blob_files([Some(&empty)]).execute().await.unwrap()[0].data,
        Some(Bytes::new())
    );
    assert!(inner.requested_ranges().is_empty());
}

#[rstest]
#[case::from_first_store(None, 1)]
#[case::override_first_store(Some(2), 2)]
#[tokio::test]
async fn test_read_blob_files_parallelism(
    #[case] parallelism: Option<usize>,
    #[case] expected: usize,
) {
    let (store, first_inner, first_gate) =
        gated_range_store(Bytes::from_static(b"abc"), "mock://first/blob");
    let first_store = Arc::new(ObjectStore::new(
        store.inner.clone(),
        url::Url::parse("mock://first/blob").unwrap(),
        None,
        None,
        false,
        true,
        1,
        0,
        None,
    ));
    let (second_store, second_inner, second_gate) =
        gated_range_store(Bytes::from_static(b"xyz"), "mock://second/blob");
    let first = BlobFile::new_inline(first_store, Path::from("first"), 0, 3);
    let second = BlobFile::new_inline(second_store, Path::from("second"), 0, 3);
    let mut builder = read_blob_files([None, Some(&first), Some(&second)]);
    if let Some(parallelism) = parallelism {
        builder = builder.with_io_parallelism(parallelism);
    }
    let mut stream = builder.try_into_stream().await.unwrap();
    assert_eq!(stream.try_next().await.unwrap().unwrap().data, None);
    assert!(
        tokio::time::timeout(Duration::from_millis(100), stream.try_next())
            .await
            .is_err()
    );
    assert_eq!(
        first_inner.requested_blob_ranges().len() + second_inner.requested_blob_ranges().len(),
        expected
    );
    first_gate.add_permits(1);
    second_gate.add_permits(1);
    let remaining = stream.try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(remaining[0].data.as_deref(), Some(b"abc".as_slice()));
    assert_eq!(remaining[1].data.as_deref(), Some(b"xyz".as_slice()));
}

#[tokio::test]
async fn test_blob_execution_isolates_store_instances_with_the_same_prefix() {
    let (cached, _) = recording_range_store(Bytes::from_static(b"cached"));
    let (fresh, _) = recording_range_store(Bytes::from_static(b"origin"));
    assert_eq!(cached.store_prefix, fresh.store_prefix);
    let path = Path::from("blobs/shared.bin");
    let mut sources = HashMap::new();
    let cached_source = shared_blob_source(&mut sources, cached.clone(), &path);
    let fresh_source = shared_blob_source(&mut sources, fresh, &path);
    assert!(!Arc::ptr_eq(&cached_source, &fresh_source));
    assert!(Arc::ptr_eq(
        &cached_source,
        &shared_blob_source(&mut sources, cached, &path)
    ));
    let entries = [cached_source, fresh_source]
        .into_iter()
        .enumerate()
        .map(|(selection_index, source)| BlobEntry {
            selection_index,
            file: BlobFile::with_source(source, 0, 6, BlobKind::External, None),
            requested_range: None,
        })
        .collect();
    let mut blobs = execute_blob_entries(entries, 2, None).await.unwrap();
    blobs.sort_by_key(|blob| blob.selection_index);
    assert_eq!(blobs[0].data.as_ref(), b"cached");
    assert_eq!(blobs[1].data.as_ref(), b"origin");
}

#[tokio::test]
async fn test_execute_blob_entries_preserves_order_and_coalesces() {
    let (store, inner) = recording_range_store(Bytes::from_static(b"abcdefghij"));
    let source = Arc::new(BlobSource::new(store, Path::from("blobs/test.bin")));
    let entries = vec![
        BlobEntry {
            selection_index: 0,
            file: BlobFile::with_source(source.clone(), 4, 3, BlobKind::Packed, None),
            requested_range: None,
        },
        BlobEntry {
            selection_index: 1,
            file: BlobFile::with_source(source, 1, 3, BlobKind::Packed, None),
            requested_range: None,
        },
    ];
    let mut blobs = execute_blob_entries(entries, 2, None).await.unwrap();
    blobs.sort_by_key(|blob| blob.selection_index);

    assert_eq!(blobs.len(), 2);
    assert_eq!(blobs[0].selection_index, 0);
    assert_eq!(blobs[0].data.as_ref(), b"efg");
    assert_eq!(blobs[1].selection_index, 1);
    assert_eq!(blobs[1].data.as_ref(), b"bcd");
    assert_eq!(inner.requested_blob_ranges(), vec![1..7]);
}

#[tokio::test]
async fn test_read_blob_ranges_handles_overlaps_across_scheduler_splits() {
    let max_iop_size = *lance_io::object_store::DEFAULT_MAX_IOP_SIZE;
    let value_size = max_iop_size + 12;
    let nested_start = value_size / 2 - 1;
    let nested_end = value_size;
    let (store, inner) = recording_range_store(Bytes::from(vec![7; value_size as usize]));
    let source = Arc::new(BlobSource::new(store, Path::from("blobs/overlapping.bin")));
    let entries = vec![
        BlobEntry {
            selection_index: 0,
            file: BlobFile::with_source(source.clone(), 0, value_size, BlobKind::Dedicated, None),
            requested_range: Some(BlobReadRange::new(0, value_size)),
        },
        BlobEntry {
            selection_index: 1,
            file: BlobFile::with_source(source, 0, value_size, BlobKind::Dedicated, None),
            requested_range: Some(BlobReadRange::new(nested_start, nested_end - nested_start)),
        },
    ];

    let mut plans = plan_blob_read_plans(entries).unwrap();
    let results = execute_blob_read_plan(
        plans.pop().unwrap(),
        Arc::new(ReadBlobsExecution::new(None)),
    )
    .await
    .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].selection_index, 0);
    assert_eq!(results[0].data.len(), value_size as usize);
    assert!(results[0].data.iter().all(|byte| *byte == 7));
    assert_eq!(results[1].selection_index, 1);
    assert_eq!(results[1].data.len(), (nested_end - nested_start) as usize);
    assert!(results[1].data.iter().all(|byte| *byte == 7));

    let mut physical_ranges = inner.requested_blob_ranges();
    physical_ranges.sort_by_key(|range| range.start);
    assert_eq!(
        physical_ranges.len(),
        value_size.div_ceil(max_iop_size) as usize
    );
    assert_eq!(physical_ranges[0].start, 0);
    assert_eq!(physical_ranges.last().unwrap().end, value_size);
    assert!(
        physical_ranges
            .windows(2)
            .all(|ranges| ranges[0].end == ranges[1].start)
    );
}

#[derive(Clone, Copy, Debug)]
enum BlobSelectionStreamCase {
    Normal,
    Empty,
    Missing,
    Duplicate,
    DuplicateReplacesMissing,
    OutOfRange,
    UpstreamError,
}

#[rstest]
#[case::normal(BlobSelectionStreamCase::Normal)]
#[case::empty(BlobSelectionStreamCase::Empty)]
#[case::missing(BlobSelectionStreamCase::Missing)]
#[case::duplicate(BlobSelectionStreamCase::Duplicate)]
#[case::duplicate_replaces_missing(BlobSelectionStreamCase::DuplicateReplacesMissing)]
#[case::out_of_range(BlobSelectionStreamCase::OutOfRange)]
#[case::upstream_error(BlobSelectionStreamCase::UpstreamError)]
#[tokio::test]
async fn test_blob_selection_stream(
    #[case] case: BlobSelectionStreamCase,
    #[values(true, false)] preserve_order: bool,
) {
    let (non_null, null_values, selection_count, mut expected_values, expected_error) = match case {
        BlobSelectionStreamCase::Normal => (
            vec![Ok((2, "row2")), Ok((0, "row0"))],
            vec![(1, "null1")],
            3,
            vec!["row0", "null1", "row2"],
            None,
        ),
        BlobSelectionStreamCase::Empty => (Vec::new(), Vec::new(), 0, Vec::new(), None),
        BlobSelectionStreamCase::Missing => (
            vec![Ok((0, "row0"))],
            vec![(1, "null1")],
            3,
            vec!["row0", "null1"],
            Some(if preserve_order {
                "completed before selection index 2 was produced"
            } else {
                "completed after producing 2 of 3 selections"
            }),
        ),
        BlobSelectionStreamCase::Duplicate => (
            vec![Ok((0, "row0")), Ok((0, "duplicate"))],
            Vec::new(),
            1,
            vec!["row0"],
            Some("selection index 0 was produced more than once"),
        ),
        BlobSelectionStreamCase::DuplicateReplacesMissing => (
            vec![Ok((0, "row0")), Ok((0, "duplicate"))],
            Vec::new(),
            2,
            vec!["row0"],
            Some("selection index 0 was produced more than once"),
        ),
        BlobSelectionStreamCase::OutOfRange => (
            vec![Ok((1, "out-of-range"))],
            Vec::new(),
            1,
            Vec::new(),
            Some("selection index 1 exceeded selected row count 1"),
        ),
        BlobSelectionStreamCase::UpstreamError => (
            vec![
                Ok((0, "row0")),
                Err(Error::internal("planned blob read failed".to_string())),
            ],
            Vec::new(),
            1,
            vec!["row0"],
            Some("planned blob read failed"),
        ),
    };
    let non_null = futures::stream::iter(non_null).boxed();
    let results =
        totalize_blob_selection_stream(non_null, null_values, selection_count, preserve_order)
            .take(8)
            .collect::<Vec<_>>()
            .await;

    let mut actual_values = results
        .iter()
        .filter_map(|result| result.as_ref().ok().copied())
        .collect::<Vec<_>>();
    if !preserve_order {
        actual_values.sort_unstable();
        expected_values.sort_unstable();
    }
    assert_eq!(actual_values, expected_values);

    let errors = results
        .iter()
        .filter_map(|result| result.as_ref().err())
        .collect::<Vec<_>>();
    assert_eq!(
        errors.len(),
        if expected_error.is_some() { 1 } else { 0 },
        "stream must terminate after its first error: {errors:?}"
    );
    if let Some(expected_error) = expected_error {
        assert!(errors[0].to_string().contains(expected_error));
    }
}

#[tokio::test]
async fn test_ordered_blob_selection_stream_drains_ready_prefix_before_reads() {
    let read_poll_count = Arc::new(AtomicUsize::new(0));
    let read_poll_count_for_stream = read_poll_count.clone();
    let read_values = futures::stream::once(async move {
        read_poll_count_for_stream.fetch_add(1, Ordering::SeqCst);
        Ok::<_, Error>((2, "row2"))
    })
    .boxed();
    let mut values =
        totalize_blob_selection_stream(read_values, vec![(0, "null0"), (1, "null1")], 3, true);

    assert_eq!(values.try_next().await.unwrap(), Some("null0"));
    assert_eq!(read_poll_count.load(Ordering::SeqCst), 0);
    assert_eq!(values.try_next().await.unwrap(), Some("null1"));
    assert_eq!(read_poll_count.load(Ordering::SeqCst), 0);
    assert_eq!(values.try_next().await.unwrap(), Some("row2"));
    assert_eq!(read_poll_count.load(Ordering::SeqCst), 1);
    assert_eq!(values.try_next().await.unwrap(), None);
}

#[tokio::test]
async fn test_planned_blob_range_avoids_whole_value_read_amplification() {
    const VALUE_SIZE: u64 = 500 * 1024 * 1024;
    const WINDOW_SIZE: u64 = 100 * 1024;

    let (store, inner) = recording_range_store(Bytes::from(vec![7; WINDOW_SIZE as usize]));
    let source = Arc::new(BlobSource::new(store, Path::from("blobs/large.bin")));
    let entries = vec![BlobEntry {
        selection_index: 0,
        file: BlobFile::with_source(source, 0, VALUE_SIZE, BlobKind::Dedicated, None),
        requested_range: Some(BlobReadRange::new(0, WINDOW_SIZE)),
    }];
    let mut plans = plan_blob_read_plans(entries).unwrap();
    let results = execute_blob_read_plan(
        plans.pop().unwrap(),
        Arc::new(ReadBlobsExecution::new(None)),
    )
    .await
    .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data.len(), WINDOW_SIZE as usize);
    let physical_ranges = inner.requested_ranges();
    assert_eq!(inner.head_requests.load(Ordering::Relaxed), 0);
    assert_eq!(physical_ranges, vec![0..WINDOW_SIZE]);
    assert_eq!(
        physical_ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum::<u64>(),
        WINDOW_SIZE
    );
    assert!(!physical_ranges.contains(&(0..VALUE_SIZE)));
}

#[tokio::test]
async fn test_read_blobs_stream_emits_ready_plan_without_waiting_for_slower_ones() {
    let (slow_store, _, slow_gate) = gated_range_store(
        Bytes::from_static(b"abcdef"),
        "mock://slow/blob-range-tests",
    );
    let (fast_store, _) = recording_range_store_with_url(
        Bytes::from_static(b"uvwxyz"),
        "mock://fast/blob-range-tests",
    );
    let entries = vec![
        BlobEntry {
            selection_index: 0,
            file: BlobFile::with_source(
                Arc::new(BlobSource::new(slow_store, Path::from("blobs/slow.bin"))),
                0,
                3,
                BlobKind::Packed,
                None,
            ),
            requested_range: None,
        },
        BlobEntry {
            selection_index: 1,
            file: BlobFile::with_source(
                Arc::new(BlobSource::new(fast_store, Path::from("blobs/fast.bin"))),
                0,
                3,
                BlobKind::Packed,
                None,
            ),
            requested_range: None,
        },
    ];
    let execution = Arc::new(ReadBlobsExecution::new(None));
    let mut stream: ReadBlobsStream = futures::stream::iter(
        plan_blob_read_plans(entries)
            .unwrap()
            .into_iter()
            .map(move |plan| execute_blob_read_plan(plan, execution.clone())),
    )
    .buffer_unordered(2)
    .map_ok(|blobs: Vec<IndexedReadBlob>| {
        futures::stream::iter(blobs.into_iter().map(|blob| {
            let row_address = [10, 11][blob.selection_index];
            Ok::<ReadBlob, Error>(into_read_blob(blob, row_address))
        }))
    })
    .try_flatten()
    .boxed();

    let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.row_address, 11);
    assert_eq!(first.data.as_deref(), Some(b"uvw".as_slice()));

    slow_gate.add_permits(1);
    let second = tokio::time::timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(second.row_address, 10);
    assert_eq!(second.data.as_deref(), Some(b"abc".as_slice()));
}

#[tokio::test]
async fn test_blob_selection_apis_preserve_v1_nulls_and_empty_values() {
    let dir = tempfile::tempdir().unwrap();
    let base = Path::from_absolute_path(dir.path()).unwrap();
    let store = ObjectStore::local();
    let version = ConcreteFileVersion::V2_0;
    let blob_metadata = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("blob", DataType::LargeBinary, true).with_metadata(blob_metadata),
    ]));
    let values = LargeBinaryArray::from(vec![None, Some(b"".as_slice()), Some(b"abc".as_slice())]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let data_file = write_file(&store, &base, "blob.lance", &batch, version).await;
    let mut manifest = Manifest::new(
        Schema::try_from(schema.as_ref()).unwrap(),
        Arc::new(vec![Fragment {
            files: vec![data_file],
            physical_rows: Some(3),
            ..Fragment::new(0)
        }]),
        DataStorageFormat::new(version),
        Default::default(),
    );
    publish_manifest(&store, &base, &mut manifest).await;
    let table = Table::open(dir.path().to_str().unwrap()).await.unwrap();

    let blob_files = table
        .take_blobs_by_indices(&[2, 1, 0], "blob")
        .await
        .unwrap();
    assert_eq!(blob_files.len(), 3);
    assert_eq!(
        blob_files[0]
            .as_ref()
            .unwrap()
            .read()
            .await
            .unwrap()
            .as_ref(),
        b"abc"
    );
    assert!(
        blob_files[1]
            .as_ref()
            .unwrap()
            .read()
            .await
            .unwrap()
            .is_empty()
    );
    assert!(blob_files[2].is_none());

    let blobs = table
        .read_blobs("blob")
        .unwrap()
        .with_row_indices([2, 1, 0])
        .execute()
        .await
        .unwrap();
    assert_eq!(
        blobs
            .iter()
            .map(|blob| (blob.row_address, blob.data.as_deref()))
            .collect::<Vec<_>>(),
        vec![
            (2, Some(b"abc".as_slice())),
            (1, Some(b"".as_slice())),
            (0, None),
        ]
    );

    let results = table
        .read_blob_ranges("blob")
        .unwrap()
        .with_row_indices([
            BlobRangeRequest::new(0, 0, 0),
            BlobRangeRequest::new(0, 0, 1),
            BlobRangeRequest::new(1, 0, 0),
            BlobRangeRequest::new(2, 1, 1),
        ])
        .execute()
        .await
        .unwrap();

    assert_eq!(
        results
            .iter()
            .map(|result| (result.request_index, result.data.as_deref()))
            .collect::<Vec<_>>(),
        vec![
            (0, None),
            (1, None),
            (2, Some(b"".as_slice())),
            (3, Some(b"b".as_slice())),
        ]
    );

    let blobs = read_blob_files(blob_files.iter().map(Option::as_ref))
        .execute()
        .await
        .unwrap();
    assert_eq!(
        blobs
            .iter()
            .map(|blob| (blob.request_index, blob.data.as_deref()))
            .collect::<Vec<_>>(),
        vec![
            (0, Some(b"abc".as_slice())),
            (1, Some(b"".as_slice())),
            (2, None),
        ]
    );

    let results = read_blob_file_ranges([
        (blob_files[2].as_ref(), BlobReadRange::new(0, 0)),
        (blob_files[2].as_ref(), BlobReadRange::new(0, 1)),
        (blob_files[1].as_ref(), BlobReadRange::new(0, 0)),
        (blob_files[0].as_ref(), BlobReadRange::new(1, 1)),
    ])
    .execute()
    .await
    .unwrap();
    assert_eq!(
        results
            .iter()
            .map(|result| (result.request_index, result.data.as_deref()))
            .collect::<Vec<_>>(),
        vec![
            (0, None),
            (1, None),
            (2, Some(b"".as_slice())),
            (3, Some(b"b".as_slice())),
        ]
    );

    let descriptions = StructArray::try_new(
        vec![
            Arc::new(Field::new("position", DataType::UInt64, false)),
            Arc::new(Field::new("size", DataType::UInt64, false)),
        ]
        .into(),
        vec![
            Arc::new(UInt64Array::from(vec![0])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![0])) as ArrayRef,
        ],
        Some(NullBuffer::from(vec![false])),
    )
    .unwrap();
    let row_addrs = UInt64Array::from(vec![u64::MAX]);
    let files = collect_blob_files_v1(&table, u32::MAX, &descriptions, &row_addrs)
        .await
        .unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].is_none());
    let non_null = StructArray::new(
        descriptions.fields().clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1])),
            Arc::new(UInt64Array::from(vec![3])),
        ],
        None,
    );
    let error = collect_blob_files_v1(&table, 0, &non_null, &row_addrs)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error.to_string().contains("references missing fragment"),
        "{error}"
    );
}

#[tokio::test]
async fn table_reads_blob_v2_handles_and_rejects_missing_rows() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let base = Path::from_absolute_path(dir.path()).unwrap();
    let object = dir.path().join("external.bin");
    std::fs::write(&object, b"external").unwrap();
    let external_uri = url::Url::from_file_path(&object).unwrap();
    let store = ObjectStore::local();
    let version = ConcreteFileVersion::V2_2;
    let metadata = HashMap::from([(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]);
    let prepared_field = Field::new(
        "blob",
        DataType::Struct(BLOB_V2_PREPARED_FIELDS.clone()),
        true,
    )
    .with_metadata(metadata.clone());
    let prepared = StructArray::try_new(
        BLOB_V2_PREPARED_FIELDS.clone(),
        vec![
            Arc::new(UInt8Array::from(vec![
                Some(BlobKind::Inline as u8),
                None,
                Some(BlobKind::Inline as u8),
                Some(BlobKind::Inline as u8),
                Some(BlobKind::External as u8),
            ])),
            Arc::new(LargeBinaryArray::from(vec![
                Some(b"payload".as_slice()),
                None,
                Some(b"".as_slice()),
                Some(b"deleted".as_slice()),
                None,
            ])),
            Arc::new(StringArray::from(vec![
                None,
                None,
                None,
                None,
                Some(external_uri.as_str()),
            ])),
            Arc::new(UInt32Array::from(vec![None, None, None, None, Some(0)])),
            Arc::new(UInt64Array::from(vec![None, None, None, None, Some(0)])),
            Arc::new(UInt64Array::from(vec![None, None, None, None, Some(0)])),
        ],
        Some(NullBuffer::from(vec![true, false, true, true, true])),
    )
    .unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![prepared_field])),
        vec![Arc::new(prepared)],
    )
    .unwrap();
    let mut data_file = write_file(&store, &base, "blobs.lance", &batch, version).await;
    let schema = Schema::try_from(&ArrowSchema::new(vec![
        Field::new("blob", BLOB_V2_LOGICAL_TYPE.clone(), true).with_metadata(metadata),
    ]))
    .unwrap();
    // The blob field maps to one physical column, irrespective of its logical children.
    data_file.fields = Arc::from([schema.fields[0].id]);
    let deleted = write_deletion_file(&base, 0, 1, &DeletionVector::from_iter([3]), &store, 0)
        .await
        .unwrap();
    let mut manifest = Manifest::new(
        schema,
        Arc::new(vec![Fragment {
            files: vec![data_file],
            physical_rows: Some(5),
            row_id_meta: Some(RowIdMeta::Inline(
                write_row_ids(&RowIdSequence::from(100..105)).into(),
            )),
            deletion_file: deleted,
            ..Fragment::new(0)
        }]),
        DataStorageFormat::new(version),
        Default::default(),
    );
    manifest.next_row_id = 105;
    publish_manifest(&store, &base, &mut manifest).await;

    let table = Table::open(uri).await.unwrap();
    let handles = table
        .take_blobs_by_indices(&[3, 1, 0, 2, 0], "blob")
        .await
        .unwrap();
    let read = read_blob_files(handles.iter().map(Option::as_ref))
        .execute()
        .await
        .unwrap();
    let payloads: Vec<_> = read.iter().map(|blob| blob.data.as_deref()).collect();
    assert_eq!(
        payloads,
        [
            Some(b"external".as_slice()),
            None,
            Some(b"payload"),
            Some(b""),
            Some(b"payload")
        ]
    );

    let ranges = table
        .read_blob_ranges("blob")
        .unwrap()
        .with_row_ids([
            BlobRangeRequest::new(104, 1, 3),
            BlobRangeRequest::new(101, 1, 3),
            BlobRangeRequest::new(100, 0, 3),
        ])
        .execute()
        .await
        .unwrap();
    assert_eq!(
        ranges
            .iter()
            .map(|blob| (blob.row_address, blob.data.as_deref()))
            .collect::<Vec<_>>(),
        [
            (4, Some(b"xte".as_slice())),
            (1, None),
            (0, Some(b"pay".as_slice()))
        ],
    );

    let error = table.take_blobs(&[100, 103], "blob").await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error.to_string().contains("first missing row ID 103"),
        "{error}"
    );
    let error = table
        .take_blobs_by_addresses(&[3, 0], "blob")
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(
        error.to_string().contains("must not target deleted rows"),
        "{error}"
    );

    let session = crate::TableSession::default()
        .with_unregistered_external_blob_policy(UnregisteredExternalBlobPolicy::Deny);
    let restricted = crate::TableBuilder::from_uri(uri)
        .with_session(Arc::new(session))
        .load()
        .await
        .unwrap();
    let error = restricted.take_blobs(&[104], "blob").await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("external blobs must reference a registered storage base"),
        "{error}"
    );
}
