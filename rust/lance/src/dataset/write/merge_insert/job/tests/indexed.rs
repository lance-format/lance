// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::record_batch;
use arrow_array::types::Int32Type;
use lance_core::utils::address::RowAddress;

#[tokio::test]
async fn test_stable_row_order_releases_staged_inserts() {
    let scope = Arc::new(analysis::AnalysisScope::new("memory://"));
    let session = new_session_context(&merge_insert_execution_options());
    let session = SessionContext::new_with_config_rt(
        session.copied_config().with_extension(scope.clone()),
        session.runtime_env(),
    );

    let make_batch = |id: Option<u64>, value: i32| {
        record_batch!((ROW_ID, UInt64, [id]), ("value", Int32, [value])).unwrap()
    };
    let schema = make_batch(None, 0).schema();
    let mut previous: Option<std::sync::Weak<dyn Array>> = None;
    let batches = stream::iter([(None, 8), (None, 7), (Some(91), 6), (Some(80), 5)]).map(
        move |(id, value)| {
            // The first pass must release each input before asking for another,
            // including inserts whose output is deferred until the second pass.
            if previous
                .as_ref()
                .is_some_and(|array| array.upgrade().is_some())
            {
                return Err(DataFusionError::Execution(
                    "row-ID staging retained its previous input batch".into(),
                ));
            }
            let batch = make_batch(id, value);
            previous = Some(Arc::downgrade(&batch["value"]));
            Ok(batch)
        },
    );
    let source = RecordBatchStreamAdapter::new(schema, batches);
    let mut output = row_ids::updates_before_inserts(Box::pin(source), &session)
        .await
        .unwrap();
    for expected in [6, 5, 8, 7] {
        let batch = output.try_next().await.unwrap().unwrap();
        assert_eq!(batch.schema().fields().len(), 1);
        assert_eq!(
            batch["value"].as_primitive::<Int32Type>().values(),
            &[expected]
        );
    }
    assert!(output.try_next().await.unwrap().is_none());
    let replay = scope.replay.snapshot();
    assert!(replay.spill_count > 0);
    assert!(replay.bytes_written > 0 && replay.bytes_read > 0);
}

#[tokio::test]
async fn test_indexed_merge_releases_reservations_between_stages() {
    let indexed = record_batch!(
        ("id", Int32, [1, 3]),
        ("value", Int32, [0, 0]),
        ("other", Int32, [7, 7])
    )
    .unwrap();
    let tail = record_batch!(
        ("id", Int32, [2]),
        ("value", Int32, [0]),
        ("other", Int32, [7])
    )
    .unwrap();
    let reader = |batch: RecordBatch| RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(reader(indexed), "memory://", None)
        .await
        .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset.append(reader(tail), None).await.unwrap();

    let mut job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .conflict_retries(0)
        .try_build()
        .unwrap();
    job.spill_execution_options.mem_pool_size = Some(4 * 1024 * 1024);
    let session = new_session_context(&job.spill_execution_options);
    let scope = analysis::AnalysisScope::observing_subplan_starts(job.dataset.uri());
    job.spill_session_context = SessionContext::new_with_config_rt(
        session.copied_config().with_extension(scope.clone()),
        session.runtime_env(),
    );

    let source = record_batch!(
        ("id", Int32, [3, 0, 2, 1]),
        ("value", Int32, [30, 0, 20, 10])
    )
    .unwrap();
    let (dataset, stats) = job.execute_reader(reader(source)).await.unwrap();
    assert_eq!((stats.num_updated_rows, stats.num_inserted_rows), (3, 1));
    let merged = dataset
        .scan()
        .project(&["id", "value"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let mut actual: Vec<_> = merged["id"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .zip(
            merged["value"]
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied(),
        )
        .collect();
    actual.sort_unstable();
    assert_eq!(actual, [(0, 0), (1, 10), (2, 20), (3, 30)]);
    let starts = scope.subplan_starts.as_ref().unwrap().lock().unwrap();
    for (previous, next) in [
        ("index_probe", "indexed_source_sort"),
        ("indexed_source_sort", "indexed_join"),
        ("indexed_join", "fragment_update_sort"),
    ] {
        let start = starts.iter().find(|start| start.purpose == next).unwrap();
        assert!(
            start.completed.iter().any(|name| name == previous),
            "{previous} must finish before {next}"
        );
        let consumers = start.consumers.as_ref().unwrap();
        assert!(
            consumers.iter().all(|c| c.reserved == 0),
            "{next} started with live reservations: {consumers:?}"
        );
    }
    let plans = scope.plans();
    // Gathering the fallback scan through a repartition would take a third share
    // of the pool. The union must feed a coalescer without its own reservation.
    let join = plans
        .iter()
        .find(|plan| plan.purpose == "indexed_join")
        .unwrap();
    let nodes = &join.execution.operators;
    let union = nodes.iter().find(|node| node.name == "UnionExec").unwrap();
    let (_, parent_path) = union.path.split_last().unwrap();
    assert!(
        nodes
            .iter()
            .any(|node| node.path == parent_path && node.name == "CoalescePartitionsExec")
    );
}

#[tokio::test]
async fn test_sequential_source_pulls_only_on_demand() {
    let reads = Arc::new(AtomicU32::new(0));
    let partitions = [0, 2].map(|first| {
        let reads = reads.clone();
        let batch = record_batch!(("id", Int32, [first])).unwrap();
        let schema = batch.schema();
        let stream = stream::iter([batch.clone(), batch]).map(move |batch| {
            reads.fetch_add(1, Ordering::Relaxed);
            Ok(batch)
        });
        Arc::new(OneShotExec::new(Box::pin(RecordBatchStreamAdapter::new(
            schema, stream,
        )))) as Arc<dyn ExecutionPlan>
    });
    let plan = UnionExec::try_new(partitions.to_vec()).unwrap();
    let source = SequentialSourcePartitionStream::try_new(
        plan,
        Some(Field::new("ordinal", DataType::UInt64, false)),
        None,
    )
    .unwrap();
    let mut output = source.execute(SessionContext::new().task_ctx());
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    for ordinal in 0..4 {
        let batch = output.try_next().await.unwrap().unwrap();
        assert_eq!(
            batch["id"].as_primitive::<Int32Type>().value(0),
            if ordinal < 2 { 0 } else { 2 }
        );
        assert_eq!(
            batch["ordinal"].as_primitive::<UInt64Type>().value(0),
            u64::from(ordinal)
        );
        assert_eq!(reads.load(Ordering::Relaxed), ordinal + 1);
    }
    assert!(output.try_next().await.unwrap().is_none());
}

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
        stale_rows.physical_selection(),
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

/// Stale writers must detect an overlapping insert regardless of which
/// writer uses the scalar index, then re-run the merge if retries are enabled.
#[rstest::rstest]
#[case::single_key(false, false)]
#[case::composite_key(true, false)]
#[case::partial_source(false, true)]
#[case::partial_composite_source(true, true)]
#[tokio::test]
async fn test_concurrent_insert_same_new_key(
    #[case] is_composite: bool,
    #[case] is_partial: bool,
    #[values(false, true)] first_uses_index: bool,
    #[values(false, true)] second_uses_index: bool,
    #[values(0, 1)] conflict_retries: u32,
) {
    let key_metadata = HashMap::from([(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    )]);
    let mut fields = vec![
        Field::new("id", DataType::UInt32, false).with_metadata(key_metadata.clone()),
        Field::new("value", DataType::UInt32, false),
        Field::new("extra", DataType::UInt32, true),
    ];
    let mut on = vec!["id".to_string()];
    if is_composite {
        fields.push(Field::new("key2", DataType::UInt32, false).with_metadata(key_metadata));
        on.push("key2".to_string());
    }
    let schema = Arc::new(Schema::new(fields));
    let make_batch = |ids: Vec<u32>, value: u32| {
        let len = ids.len();
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(UInt32Array::from(ids)),
            Arc::new(UInt32Array::from(vec![value; len])),
            Arc::new(UInt32Array::from(vec![None; len])),
        ];
        if is_composite {
            columns.push(Arc::new(UInt32Array::from(vec![7; len])));
        }
        RecordBatch::try_new(schema.clone(), columns).unwrap()
    };
    let mut dataset = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        })
        .execute(vec![make_batch(vec![0, 1, 2, 3], 0)])
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    for key in &on {
        dataset
            .create_index(
                &[key.as_str()],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
    }
    let dataset = Arc::new(dataset);
    let source_batch = |value| {
        let batch = make_batch(vec![100], value);
        if is_partial {
            let columns = if is_composite {
                vec![1, 3, 0]
            } else {
                vec![1, 0]
            };
            batch.project(&columns).unwrap()
        } else {
            batch
        }
    };
    let build_job = |use_index, retries| {
        MergeInsertBuilder::try_new(dataset.clone(), on.clone())
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .use_index(use_index)
            .conflict_retries(retries)
            .try_build()
            .unwrap()
    };

    // Both jobs read the same version, making the conflict deterministic.
    let b1 = build_job(first_uses_index, 0);
    let b2 = build_job(second_uses_index, conflict_retries);
    assert_eq!(
        b1.can_use_create_plan(source_batch(1).schema().as_ref())
            .await
            .unwrap(),
        !first_uses_index,
    );
    assert_eq!(
        b2.can_use_create_plan(source_batch(2).schema().as_ref())
            .await
            .unwrap(),
        !second_uses_index,
    );
    let (_, stats) = b1.execute_batches(vec![source_batch(1)]).await.unwrap();
    assert_eq!(stats.num_inserted_rows, 1);
    let result = b2.execute_batches(vec![source_batch(2)]).await;
    if conflict_retries == 0 {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::TooMuchWriteContention { .. }));
        assert!(error.to_string().contains("Attempted 0 retries"));
    } else {
        let (dataset, stats) = result.unwrap();
        assert_eq!(stats.num_attempts, 2);
        assert_eq!(stats.num_inserted_rows, 0);
        assert_eq!(stats.num_updated_rows, 1);
        let found = dataset
            .scan()
            .filter("id = 100")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(found.num_rows(), 1);
        assert_eq!(found["value"].as_primitive::<UInt32Type>().value(0), 2);
    }
}

#[rstest::rstest]
#[tokio::test]
async fn test_indexed_merge_insert_overlay_partial_coverage(
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] partial_coverage: bool,
) {
    let mut results = Vec::new();
    for use_index in [false, true] {
        let initial = record_batch!(
            ("key", Int32, [0, 1, 2, 3, 4, 5]),
            ("other_key", Int32, [10, 20, 30, 40, 50, 60]),
            ("value", Int32, [0, 1, 2, 3, 4, 5])
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial.clone())], initial.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        for (column, fragments) in [
            ("key", vec![0, 1]),
            (
                "other_key",
                if partial_coverage {
                    vec![0, 2]
                } else {
                    vec![0, 1, 2]
                },
            ),
        ] {
            let name = format!("{column}_idx");
            let segment = crate::index::CreateIndexBuilder::new(
                &mut dataset,
                &[column],
                IndexType::BTree,
                &ScalarIndexParams::default(),
            )
            .name(name.clone())
            .fragments(fragments)
            .execute_uncommitted()
            .await
            .unwrap();
            dataset
                .commit_existing_index_segments(&name, column, vec![segment])
                .await
                .unwrap();
        }
        dataset.delete("key = 2").await.unwrap();
        let field = dataset.schema().field("key").unwrap().id;
        let schema = dataset.schema().project_by_ids(&[field], true);
        let patch = record_batch!(
            (
                ROW_ADDR,
                UInt64,
                [u64::from(RowAddress::new_from_parts(1, 1))]
            ),
            ("key", Int32, [100])
        )
        .unwrap();
        let mut writer = dataset
            .get_fragment(1)
            .unwrap()
            .write_overlay(&schema)
            .await
            .unwrap();
        writer.write_batch(&patch).await.unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        let version = dataset.version().version;
        let dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataOverlay {
                groups: vec![group],
            },
            Some(version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap();
        let source = record_batch!(
            ("key", Int32, [100, 3, 0, 5]),
            ("other_key", Int32, [40, 40, 10, 60]),
            ("value", Int32, [900, 901, 902, 903])
        )
        .unwrap();
        let job = MergeInsertBuilder::try_new(
            Arc::new(dataset.clone()),
            vec!["key".into(), "other_key".into()],
        )
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .use_index(use_index)
        .try_build()
        .unwrap();
        if use_index {
            let keys = job.indexed_join_keys().await.unwrap();
            assert_eq!(keys.len(), 2);
            let stale = job.overlay_stale_rows_for_keys(&keys).await.unwrap();
            assert_eq!(
                stale.physical_selection(),
                if partial_coverage {
                    RowAddrTreeMap::new()
                } else {
                    RowAddrTreeMap::from_iter([u64::from(RowAddress::new_from_parts(1, 1))])
                }
            );
        }
        let (updated, stats) = job
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(source.slice(0, 2)), Ok(source.slice(2, 2))],
                source.schema(),
            )))
            .await
            .unwrap();
        assert_eq!((stats.num_updated_rows, stats.num_inserted_rows), (3, 1));
        let batch = updated.scan().try_into_batch().await.unwrap();
        let mut rows: Vec<_> = (0..batch.num_rows())
            .map(|row| {
                (
                    batch["key"].as_primitive::<Int32Type>().value(row),
                    batch["other_key"].as_primitive::<Int32Type>().value(row),
                    batch["value"].as_primitive::<Int32Type>().value(row),
                )
            })
            .collect();
        rows.sort_unstable();
        results.push(rows);
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[1].len(), 6);
}
