// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::record_batch;

#[rstest::rstest]
#[case::first_key("key", false, true)]
#[case::second_key("other_key", false, true)]
#[case::unindexed_field("value", false, false)]
#[case::rebuilt_index("key", true, false)]
#[tokio::test]
async fn test_indexed_merge_insert_overlay_staleness(
    #[case] column: &str,
    #[case] rebuild_index: bool,
    #[case] is_stale: bool,
) {
    let initial = record_batch!(
        ("key", Int32, [0, 1]),
        ("other_key", Int32, [10, 20]),
        ("value", Int32, [0, 1])
    )
    .unwrap();
    let schema = initial.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(initial)], schema.clone()),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    for key in ["key", "other_key"] {
        dataset
            .create_index(
                &[key],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }
    let dataset = Arc::new(dataset);
    let overlay_schema = dataset
        .schema()
        .project_by_ids(&[dataset.schema().field(column).unwrap().id], true);
    let batch = record_batch!((ROW_ADDR, UInt64, [1]), (column, Int32, [100])).unwrap();
    let mut writer = dataset
        .get_fragment(0)
        .unwrap()
        .write_overlay(&overlay_schema)
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    let group = writer.finish().await.unwrap().unwrap();
    let read_version = dataset.version().version;
    let mut dataset = Dataset::commit(
        WriteDestination::Dataset(dataset),
        Operation::DataOverlay {
            groups: vec![group],
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();
    if rebuild_index {
        dataset
            .create_index(
                &[column],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
    }
    let job = MergeInsertBuilder::try_new(
        Arc::new(dataset),
        vec!["key".to_string(), "other_key".to_string()],
    )
    .unwrap()
    .when_matched(WhenMatched::UpdateAll)
    .when_not_matched(WhenNotMatched::InsertAll)
    .try_build()
    .unwrap();
    let indexed_keys = job.indexed_join_keys().await.unwrap();
    let stale_rows = job
        .overlay_stale_rows_for_keys(&indexed_keys)
        .await
        .unwrap();
    assert_eq!(
        stale_rows,
        if is_stale {
            RowAddrTreeMap::from_iter([1u64])
        } else {
            RowAddrTreeMap::new()
        }
    );

    let source = record_batch!(
        ("key", Int32, [if column == "key" { 100 } else { 1 }]),
        (
            "other_key",
            Int32,
            [if column == "other_key" { 100 } else { 20 }]
        ),
        ("value", Int32, [999])
    )
    .unwrap();
    let (dataset, stats) = job
        .execute_reader(Box::new(RecordBatchIterator::new([Ok(source)], schema)))
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 0);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
    assert_eq!(
        dataset
            .count_rows(Some("value = 999".to_string()))
            .await
            .unwrap(),
        1
    );
}
