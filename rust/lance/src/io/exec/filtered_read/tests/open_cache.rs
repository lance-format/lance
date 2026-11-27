// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fragment reads through the scan's shared reader-open cache.

use super::*;

async fn plan_scoped_fragments(
    fixture: &TestFixture,
    options: &FilteredReadOptions,
    index_input: Option<Arc<dyn ExecutionPlan>>,
) -> Vec<ScopedFragmentRead> {
    let read =
        FilteredReadExec::try_new(fixture.dataset.clone(), options.clone(), index_input).unwrap();
    read.get_or_create_plan(Arc::new(TaskContext::default()))
        .await
        .unwrap();
    FilteredReadStream::plan_to_scoped_fragments(
        read.plan.get().unwrap(),
        fixture.dataset.fragments(),
        &fixture.dataset,
        options,
        FilteredReadStream::make_scan_scheduler(&fixture.dataset, options),
        FilteredReadStream::make_base_schedulers(options),
        true,
    )
    .unwrap()
}

async fn count_fragment_rows(dataset: Arc<Dataset>, scoped: ScopedFragmentRead) -> usize {
    let metrics = Arc::new(FilteredReadGlobalMetrics::new(
        &ExecutionPlanMetricsSet::new(),
    ));
    FilteredReadStream::read_fragment(
        dataset,
        scoped,
        metrics,
        None,
        BlobMaterializationContext::new(None, None),
    )
    .await
    .unwrap()
    .try_buffered(1)
    .try_collect::<Vec<_>>()
    .await
    .unwrap()
    .iter()
    .flatten()
    .map(|batch| batch.batch().num_rows())
    .sum()
}

/// Fragments sharing a read projection share its schemas and one reader-open
/// cache, which keeps one entry per data-file layout
#[tokio::test]
async fn test_fragments_share_open_cache() {
    let fixture = TestFixture::new().await;
    let options = FilteredReadOptions::basic_full_read(&fixture.dataset)
        .with_filter_plan(fixture.filter_plan("not_indexed >= 75", false).await);
    let scoped = plan_scoped_fragments(&fixture, &options, None).await;
    assert_eq!(scoped.len(), 3);
    let schemas = scoped[0].schemas.clone();
    assert!(scoped.iter().all(|s| Arc::ptr_eq(&s.schemas, &schemas)));

    let mut num_rows = 0;
    for scoped_fragment in scoped {
        num_rows += count_fragment_rows(fixture.dataset.clone(), scoped_fragment).await;
    }
    // values 75..100, 250..300 and 300..400
    assert_eq!(num_rows, 175);
    // Fragment 0 holds all five columns; the appended fragments hold three
    assert_eq!(schemas.open_cache.num_entries(), 2);

    // `partly_indexed` is indexed on fragment 0 only, so fragment 0 reads just
    // the refine filter's column while fragments 2 and 3 read both filter
    // columns: two read projections, each with its own schemas and cache.
    let options = FilteredReadOptions::basic_full_read(&fixture.dataset).with_filter_plan(
        fixture
            .filter_plan("partly_indexed < 50 AND not_indexed % 3 = 0", true)
            .await,
    );
    let index_input = fixture.index_input(&options).await;
    let scoped = plan_scoped_fragments(&fixture, &options, index_input).await;
    assert_eq!(scoped.len(), 3);
    let distinct_schemas: HashSet<*const FragmentReadSchemas> =
        scoped.iter().map(|s| Arc::as_ptr(&s.schemas)).collect();
    assert_eq!(distinct_schemas.len(), 2);
    fixture
        .test_plan(options, &UInt32Array::from_iter_values((0..50).step_by(3)))
        .await;
}

/// Fragments opened in parallel still leave one cache entry per layout
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_fragment_opens_share_open_cache_entries() {
    let fixture = TestFixture::new().await;
    let options = FilteredReadOptions::basic_full_read(&fixture.dataset)
        .with_filter_plan(fixture.filter_plan("not_indexed >= 75", false).await);
    for _ in 0..20 {
        let scoped = plan_scoped_fragments(&fixture, &options, None).await;
        let schemas = scoped[0].schemas.clone();
        let tasks = scoped
            .into_iter()
            .map(|scoped_fragment| {
                tokio::spawn(count_fragment_rows(
                    fixture.dataset.clone(),
                    scoped_fragment,
                ))
            })
            .collect::<Vec<_>>();
        let num_rows: usize = futures::future::join_all(tasks)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .sum();
        assert_eq!(num_rows, 175);
        assert_eq!(schemas.open_cache.num_entries(), 2);
    }
}

fn sorted_i32(column: &dyn Array) -> Vec<i32> {
    let mut values = column.as_primitive::<Int32Type>().values().to_vec();
    values.sort_unstable();
    values
}

/// After schema evolution a scan reads fragments of several layouts: add-on
/// files for added and cast columns, ids of dropped columns still in old
/// files, a column re-added under a dropped name, an all-null column stored in
/// no file, appended single-file fragments and deleted rows. Reads through the
/// open cache must return exactly what was written, for every projection.
#[tokio::test]
async fn test_filtered_read_across_evolved_layouts() {
    fn evolved_batch(schema: &Arc<ArrowSchema>, range: Range<i32>) -> RecordBatch {
        let columns = schema
            .fields()
            .iter()
            .map(|field| -> ArrayRef {
                match (field.name().as_str(), field.data_type()) {
                    ("i", _) | ("n", _) => Arc::new(Int32Array::from_iter_values(range.clone())),
                    ("s", DataType::Struct(fields)) => Arc::new(StructArray::new(
                        fields.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(range.clone())),
                            Arc::new(Int32Array::from_iter_values(range.clone().map(|i| -i))),
                        ],
                        None,
                    )),
                    ("double", DataType::Int32) => {
                        Arc::new(Int32Array::from_iter_values(range.clone().map(|i| i * 2)))
                    }
                    ("double", DataType::Int64) => Arc::new(Int64Array::from_iter_values(
                        range.clone().map(|i| i as i64 * 2),
                    )),
                    ("dropped", _) => {
                        Arc::new(Int32Array::from_iter_values(range.clone().map(|i| i + 3)))
                    }
                    (name, data_type) => panic!("unexpected field {name}: {data_type}"),
                }
            })
            .collect();
        RecordBatch::try_new(schema.clone(), columns).unwrap()
    }

    let tmp = TempStrDir::default();
    let write_params = |mode| WriteParams {
        max_rows_per_file: 100,
        mode,
        ..Default::default()
    };
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, false),
        ArrowField::new(
            "s",
            DataType::Struct(ArrowFields::from(vec![
                ArrowField::new("a", DataType::Int32, false),
                ArrowField::new("b", DataType::Int32, false),
            ])),
            false,
        ),
    ]));
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(evolved_batch(&schema, 0..400))], schema),
        &tmp,
        Some(write_params(WriteMode::Create)),
    )
    .await
    .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![
                ("double".into(), "i * 2".into()),
                ("dropped".into(), "i + 1".into()),
            ]),
            None,
            None,
        )
        .await
        .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "n",
                DataType::Int32,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    dataset.drop_columns(&["dropped"]).await.unwrap();

    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(evolved_batch(&schema, 400..600))], schema),
        &tmp,
        Some(write_params(WriteMode::Append)),
    )
    .await
    .unwrap();
    dataset
        .alter_columns(&[ColumnAlteration::new("double".into()).cast_to(DataType::Int64)])
        .await
        .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("dropped".into(), "i + 3".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(evolved_batch(&schema, 600..700))], schema),
        &tmp,
        Some(write_params(WriteMode::Append)),
    )
    .await
    .unwrap();
    dataset.delete("i % 50 = 0").await.unwrap();

    let layouts: HashSet<&[i32]> = dataset
        .fragments()
        .iter()
        .flat_map(|f| f.files.iter().map(|file| file.fields.as_ref()))
        .collect();
    assert!(layouts.len() >= 4, "{layouts:?}");

    let expected: Vec<i32> = (0..700).filter(|i| i % 7 == 0 && i % 50 != 0).collect();
    for projection in [
        vec!["i", "s", "double", "n", "dropped"],
        vec!["double"],
        vec!["dropped"],
        vec!["s.a"],
        vec!["n"],
    ] {
        let batch = dataset
            .scan()
            .project(&projection)
            .unwrap()
            .filter("i % 7 = 0")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), expected.len(), "{projection:?}");
        for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
            match field.name().as_str() {
                "i" | "s.a" => assert_eq!(sorted_i32(column), expected),
                "double" => {
                    let mut doubles = column.as_primitive::<Int64Type>().values().to_vec();
                    doubles.sort_unstable();
                    let expected: Vec<i64> = expected.iter().map(|i| *i as i64 * 2).collect();
                    assert_eq!(doubles, expected, "{projection:?}");
                }
                "dropped" => assert_eq!(
                    sorted_i32(column),
                    expected.iter().map(|i| i + 3).collect::<Vec<_>>(),
                    "{projection:?}"
                ),
                "n" => {
                    let n = column.as_primitive::<Int32Type>();
                    let mut values: Vec<i32> = n.iter().flatten().collect();
                    values.sort_unstable();
                    let (nulls, filled): (Vec<i32>, Vec<i32>) =
                        expected.iter().partition(|i| **i < 400);
                    assert_eq!(n.null_count(), nulls.len(), "{projection:?}");
                    assert_eq!(values, filled, "{projection:?}");
                }
                "s" => {
                    let s = column.as_struct();
                    assert_eq!(sorted_i32(s.column_by_name("a").unwrap()), expected);
                    assert_eq!(
                        sorted_i32(s.column_by_name("b").unwrap()),
                        expected.iter().rev().map(|i| -i).collect::<Vec<_>>()
                    );
                }
                name => panic!("unexpected column {name}"),
            }
        }
        // Columns from different data files of one fragment must line up.
        if let (Some(i), Some(double), Some(dropped), Some(n)) = (
            batch.column_by_name("i"),
            batch.column_by_name("double"),
            batch.column_by_name("dropped"),
            batch.column_by_name("n"),
        ) {
            let i = i.as_primitive::<Int32Type>();
            let double = double.as_primitive::<Int64Type>();
            let dropped = dropped.as_primitive::<Int32Type>();
            let n = n.as_primitive::<Int32Type>();
            for row in 0..batch.num_rows() {
                assert_eq!(double.value(row), i.value(row) as i64 * 2);
                assert_eq!(dropped.value(row), i.value(row) + 3);
                assert_eq!(n.is_null(row), i.value(row) < 400);
                if !n.is_null(row) {
                    assert_eq!(n.value(row), i.value(row));
                }
            }
        }
    }

    // A projection of system columns only opens no data file for the output
    let batch = dataset
        .scan()
        .project::<&str>(&[])
        .unwrap()
        .with_row_id()
        .filter("i % 7 = 0")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), expected.len());
}

/// Fields spanning several physical columns or none (blob, packed struct,
/// list, fixed-size list) give `column_indices` with `-1` and extra entries;
/// the reader projection derived from them must open every layout.
#[tokio::test]
async fn test_filtered_read_multi_column_fields_through_open_cache() {
    let packed = ArrowFields::from(vec![
        ArrowField::new("x", DataType::Int32, false),
        ArrowField::new("y", DataType::Int32, false),
    ]);
    let item = Arc::new(ArrowField::new("item", DataType::Float32, true));
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("i", DataType::Int32, false),
        ArrowField::new("blob", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            BLOB_META_KEY.to_string(),
            "true".to_string(),
        )])),
        ArrowField::new("ps", DataType::Struct(packed.clone()), false).with_metadata(
            HashMap::from([("lance-encoding:packed".to_string(), "true".to_string())]),
        ),
        ArrowField::new("fsl", DataType::FixedSizeList(item.clone(), 2), false),
        ArrowField::new(
            "l",
            DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
            false,
        ),
    ]));
    let make_batch = |range: Range<i32>| {
        let fsl = FixedSizeListArray::try_new(
            item.clone(),
            2,
            Arc::new(Float32Array::from_iter_values(
                range.clone().flat_map(|i| [i as f32, -i as f32]),
            )),
            None,
        )
        .unwrap();
        let l = ListArray::from_iter_primitive::<Int32Type, _, _>(
            range.clone().map(|i| Some(vec![Some(i), Some(i * 10)])),
        );
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(range.clone())),
                Arc::new(LargeBinaryArray::from_iter_values(
                    range.clone().map(|i| i.to_le_bytes()),
                )),
                Arc::new(StructArray::new(
                    packed.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(range.clone())),
                        Arc::new(Int32Array::from_iter_values(range.map(|i| i + 1))),
                    ],
                    None,
                )),
                Arc::new(fsl),
                Arc::new(l),
            ],
        )
        .unwrap()
    };
    let tmp = TempStrDir::default();
    let write_params = |mode| WriteParams {
        max_rows_per_file: 100,
        mode,
        ..Default::default()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(make_batch(0..300))], schema.clone()),
        &tmp,
        Some(write_params(WriteMode::Create)),
    )
    .await
    .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("double".into(), "i * 2".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    let batch = make_batch(300..400)
        .try_with_column(
            ArrowField::new("double", DataType::Int32, true),
            Arc::new(Int32Array::from_iter_values((300..400).map(|i| i * 2))),
        )
        .unwrap();
    let batch_schema = batch.schema();
    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], batch_schema),
        &tmp,
        Some(write_params(WriteMode::Append)),
    )
    .await
    .unwrap();

    let expected: Vec<i32> = (0..400).filter(|i| i % 7 == 0).collect();
    for projection in [
        vec!["i", "blob", "ps", "fsl", "l", "double"],
        vec!["blob"],
        vec!["ps"],
        vec!["l", "fsl"],
        vec!["double"],
    ] {
        let batch = dataset
            .scan()
            .project(&projection)
            .unwrap()
            .filter("i % 7 = 0")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), expected.len(), "{projection:?}");
        if let Some(ps) = batch.column_by_name("ps") {
            let x = ps.as_struct().column_by_name("x").unwrap();
            assert_eq!(sorted_i32(x), expected, "{projection:?}");
        }
        if let Some(l) = batch.column_by_name("l") {
            let l = l.as_list::<i32>();
            let mut firsts: Vec<i32> = (0..l.len())
                .map(|row| l.value(row).as_primitive::<Int32Type>().value(0))
                .collect();
            firsts.sort_unstable();
            assert_eq!(firsts, expected, "{projection:?}");
        }
        if let Some(double) = batch.column_by_name("double") {
            assert_eq!(
                sorted_i32(double),
                expected.iter().map(|i| i * 2).collect::<Vec<_>>(),
                "{projection:?}"
            );
        }
    }
}
