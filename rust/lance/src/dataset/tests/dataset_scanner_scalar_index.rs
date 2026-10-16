// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::datatypes::Int32Type;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    Array, Int32Array, PrimitiveArray, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::TryStreamExt;
use futures::stream::StreamExt;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_linalg::distance::MetricType;
use rstest::rstest;

use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::scanner::test_dataset::assert_plan_equals;
use crate::dataset::{Dataset, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

struct ScalarIndexTestFixture {
    _test_dir: TempStrDir,
    dataset: Dataset,
    sample_query: Arc<dyn Array>,
    delete_query: Arc<dyn Array>,
    // The original version of the data, two fragments, rows 0-1000
    original_version: u64,
    // The original version of the data, 1 row deleted, compacted to a single fragment
    compact_version: u64,
    // The original version of the data + an extra 1000 unindexed
    append_version: u64,
    // The original version of the data + an extra 1000 rows, with indices updated so all rows indexed
    updated_version: u64,
    // The original version of the data with 1 deleted row
    delete_version: u64,
    // The original version of the data + an extra 1000 uindexed + 1 deleted row
    append_then_delete_version: u64,
}

#[derive(Debug, PartialEq)]
struct ScalarTestParams {
    use_index: bool,
    use_projection: bool,
    use_deleted_data: bool,
    use_new_data: bool,
    with_row_id: bool,
    use_compaction: bool,
    use_updated: bool,
}

impl ScalarIndexTestFixture {
    async fn new(data_storage_version: LanceFileVersion, use_stable_row_ids: bool) -> Self {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Write 1000 rows.  Train indices.  Then write 1000 new rows with the same vector data.
        // Then delete a row from the trained data.
        //
        // The first row where indexed == 50 is our sample query.
        // The first row where indexed == 75 is our deleted row (and delete query)
        let data = gen_batch()
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(32)),
            )
            .col("indexed", array::step::<Int32Type>())
            .col("not_indexed", array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(1000))
            .unwrap();

        // Write as two batches so we can later compact
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.clone())], data.schema().clone()),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 500,
                data_storage_version: Some(data_storage_version),
                enable_stable_row_ids: use_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2),
                false,
            )
            .await
            .unwrap();

        dataset
            .create_index(
                &["indexed"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let original_version = dataset.version().version;
        let sample_query = data["vector"].as_fixed_size_list().value(50);
        let delete_query = data["vector"].as_fixed_size_list().value(75);

        // APPEND DATA

        // Re-use the vector column in the new batch but add 1000 to the indexed/not_indexed columns so
        // they are distinct.  This makes our checks easier.
        let new_indexed =
            arrow_arith::numeric::add(&data["indexed"], &Int32Array::new_scalar(1000)).unwrap();
        let new_not_indexed =
            arrow_arith::numeric::add(&data["indexed"], &Int32Array::new_scalar(1000)).unwrap();
        let append_data = RecordBatch::try_new(
            data.schema(),
            vec![data["vector"].clone(), new_indexed, new_not_indexed],
        )
        .unwrap();

        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(append_data)], data.schema()),
                Some(WriteParams {
                    data_storage_version: Some(data_storage_version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        let append_version = dataset.version().version;

        // UPDATE

        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();
        let updated_version = dataset.version().version;

        // APPEND -> DELETE

        dataset.checkout_version(append_version).await.unwrap();
        dataset.restore().await.unwrap();

        dataset.delete("not_indexed = 75").await.unwrap();

        let append_then_delete_version = dataset.version().version;

        // DELETE

        let mut dataset = dataset.checkout_version(original_version).await.unwrap();
        dataset.restore().await.unwrap();

        dataset.delete("not_indexed = 75").await.unwrap();

        let delete_version = dataset.version().version;

        // COMPACT (this should materialize the deletion)
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        let compact_version = dataset.version().version;
        dataset.checkout_version(original_version).await.unwrap();
        dataset.restore().await.unwrap();

        Self {
            _test_dir: test_dir,
            dataset,
            sample_query,
            delete_query,
            original_version,
            compact_version,
            append_version,
            updated_version,
            delete_version,
            append_then_delete_version,
        }
    }

    fn sample_query(&self) -> &PrimitiveArray<Float32Type> {
        self.sample_query.as_primitive::<Float32Type>()
    }

    fn delete_query(&self) -> &PrimitiveArray<Float32Type> {
        self.delete_query.as_primitive::<Float32Type>()
    }

    async fn get_dataset(&self, params: &ScalarTestParams) -> Dataset {
        let version = if params.use_compaction {
            // These combinations should not be possible
            if params.use_deleted_data || params.use_new_data || params.use_updated {
                panic!("There is no test data combining new/deleted/updated data with compaction");
            } else {
                self.compact_version
            }
        } else if params.use_updated {
            // These combinations should not be possible
            if params.use_deleted_data || params.use_new_data || params.use_compaction {
                panic!("There is no test data combining updated data with new/deleted/compaction");
            } else {
                self.updated_version
            }
        } else {
            match (params.use_new_data, params.use_deleted_data) {
                (false, false) => self.original_version,
                (false, true) => self.delete_version,
                (true, false) => self.append_version,
                (true, true) => self.append_then_delete_version,
            }
        };
        self.dataset.checkout_version(version).await.unwrap()
    }

    async fn run_query(
        &self,
        query: &str,
        vector: Option<&PrimitiveArray<Float32Type>>,
        params: &ScalarTestParams,
    ) -> (String, RecordBatch) {
        let dataset = self.get_dataset(params).await;
        let mut scan = dataset.scan();
        if let Some(vector) = vector {
            scan.nearest("vector", vector, 10).unwrap();
        }
        if params.use_projection {
            scan.project(&["indexed"]).unwrap();
        }
        if params.with_row_id {
            scan.with_row_id();
        }
        scan.scan_in_order(true);
        scan.use_index(params.use_index);
        scan.filter(query).unwrap();
        scan.prefilter(true);

        let plan = scan.explain_plan(true).await.unwrap();
        let batch = scan.try_into_batch().await.unwrap();

        if params.use_projection {
            // 1 projected column
            let mut expected_columns = 1;
            if vector.is_some() {
                // distance column if included always (TODO: it shouldn't)
                expected_columns += 1;
            }
            if params.with_row_id {
                expected_columns += 1;
            }
            assert_eq!(batch.num_columns(), expected_columns);
        } else {
            let mut expected_columns = 3;
            if vector.is_some() {
                // distance column
                expected_columns += 1;
            }
            if params.with_row_id {
                expected_columns += 1;
            }
            // vector, indexed, not_indexed, _distance
            assert_eq!(batch.num_columns(), expected_columns);
        }

        (plan, batch)
    }

    fn assert_none<F: Fn(i32) -> bool>(&self, batch: &RecordBatch, predicate: F, message: &str) {
        let indexed = batch["indexed"].as_primitive::<Int32Type>();
        if indexed.iter().map(|val| val.unwrap()).any(predicate) {
            panic!("{}", message);
        }
    }

    fn assert_one<F: Fn(i32) -> bool>(&self, batch: &RecordBatch, predicate: F, message: &str) {
        let indexed = batch["indexed"].as_primitive::<Int32Type>();
        if !indexed.iter().map(|val| val.unwrap()).any(predicate) {
            panic!("{}", message);
        }
    }

    fn uses_legacy_scan(&self) -> bool {
        self.dataset
            .manifest()
            .data_storage_format
            .lance_file_format()
            == lance_file::version::ConcreteFileVersion::V1
    }

    async fn check_vector_scalar_indexed_and_refine(&self, params: &ScalarTestParams) {
        let (query_plan, batch) = self
            .run_query(
                "indexed != 50 AND ((not_indexed < 100) OR (not_indexed >= 1000 AND not_indexed < 1100))",
                Some(self.sample_query()),
                params,
            )
            .await;
        // Materialization is always required if there is a refine
        if self.uses_legacy_scan() {
            assert!(query_plan.contains("MaterializeIndex"));
        }
        // The result should not include the sample query
        self.assert_none(
            &batch,
            |val| val == 50,
            "The query contained 50 even though it was filtered",
        );
        if !params.use_new_data {
            // Refine should have been applied
            self.assert_none(
                &batch,
                |val| (100..1000).contains(&val) || (val >= 1100),
                "The non-indexed refine filter was not applied",
            );
        }

        // If there is new data then the dupe of row 50 should be in the results
        if params.use_new_data || params.use_updated {
            self.assert_one(
                &batch,
                |val| val == 1050,
                "The query did not contain 1050 from the new data",
            );
        }
    }

    async fn check_vector_scalar_indexed_only(&self, params: &ScalarTestParams) {
        let (query_plan, batch) = self
            .run_query("indexed != 50", Some(self.sample_query()), params)
            .await;
        if self.uses_legacy_scan() {
            if params.use_index {
                // An ANN search whose prefilter is fully satisfied by the index should be
                // able to use a ScalarIndexQuery
                assert!(query_plan.contains("ScalarIndexQuery"));
            } else {
                // A KNN search requires materialization of the index
                assert!(query_plan.contains("MaterializeIndex"));
            }
        }
        // The result should not include the sample query
        self.assert_none(
            &batch,
            |val| val == 50,
            "The query contained 50 even though it was filtered",
        );
        // If there is new data then the dupe of row 50 should be in the results
        if params.use_new_data {
            self.assert_one(
                &batch,
                |val| val == 1050,
                "The query did not contain 1050 from the new data",
            );
            if !params.use_new_data {
                // Let's also make sure our filter can target something in the new data only
                let (_, batch) = self
                    .run_query("indexed == 1050", Some(self.sample_query()), params)
                    .await;
                assert_eq!(batch.num_rows(), 1);
            }
        }
        if params.use_deleted_data {
            let (_, batch) = self
                .run_query("indexed == 75", Some(self.delete_query()), params)
                .await;
            if !params.use_new_data {
                assert_eq!(batch.num_rows(), 0);
            }
        }
    }

    async fn check_vector_queries(&self, params: &ScalarTestParams) {
        self.check_vector_scalar_indexed_only(params).await;
        self.check_vector_scalar_indexed_and_refine(params).await;
    }

    async fn check_simple_indexed_only(&self, params: &ScalarTestParams) {
        let (query_plan, batch) = self.run_query("indexed != 50", None, params).await;
        // Materialization is always required for non-vector search
        if self.uses_legacy_scan() {
            assert!(query_plan.contains("MaterializeIndex"));
        } else {
            assert!(query_plan.contains("LanceRead"));
        }
        // The result should not include the targeted row
        self.assert_none(
            &batch,
            |val| val == 50,
            "The query contained 50 even though it was filtered",
        );
        let mut expected_num_rows = if params.use_new_data || params.use_updated {
            1999
        } else {
            999
        };
        if params.use_deleted_data || params.use_compaction {
            expected_num_rows -= 1;
        }
        assert_eq!(batch.num_rows(), expected_num_rows);

        // Let's also make sure our filter can target something in the new data only
        if params.use_new_data || params.use_updated {
            let (_, batch) = self.run_query("indexed == 1050", None, params).await;
            assert_eq!(batch.num_rows(), 1);
        }

        // Also make sure we don't return deleted data
        if params.use_deleted_data || params.use_compaction {
            let (_, batch) = self.run_query("indexed == 75", None, params).await;
            assert_eq!(batch.num_rows(), 0);
        }
    }

    async fn check_simple_indexed_and_refine(&self, params: &ScalarTestParams) {
        let (query_plan, batch) = self.run_query(
            "indexed != 50 AND ((not_indexed < 100) OR (not_indexed >= 1000 AND not_indexed < 1100))",
            None,
            params
        ).await;
        // Materialization is always required for non-vector search
        if self.uses_legacy_scan() {
            assert!(query_plan.contains("MaterializeIndex"));
        } else {
            assert!(query_plan.contains("LanceRead"));
        }
        // The result should not include the targeted row
        self.assert_none(
            &batch,
            |val| val == 50,
            "The query contained 50 even though it was filtered",
        );
        // The refine should be applied
        self.assert_none(
            &batch,
            |val| (100..1000).contains(&val) || (val >= 1100),
            "The non-indexed refine filter was not applied",
        );

        let mut expected_num_rows = if params.use_new_data || params.use_updated {
            199
        } else {
            99
        };
        if params.use_deleted_data || params.use_compaction {
            expected_num_rows -= 1;
        }
        assert_eq!(batch.num_rows(), expected_num_rows);
    }

    async fn check_simple_queries(&self, params: &ScalarTestParams) {
        self.check_simple_indexed_only(params).await;
        self.check_simple_indexed_and_refine(params).await;
    }
}

// There are many different ways that a query can be run and they all have slightly different
// effects on the plan that gets built.  This test attempts to run the same queries in various
// different configurations to ensure that we get consistent results
#[rstest]
#[tokio::test]
async fn test_secondary_index_scans(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] use_stable_row_ids: bool,
    #[values(false, true)] use_index: bool,
    #[values(false, true)] use_projection: bool,
) {
    let fixture = Box::pin(ScalarIndexTestFixture::new(
        data_storage_version,
        use_stable_row_ids,
    ))
    .await;

    for use_deleted_data in [false, true] {
        for use_new_data in [false, true] {
            // Don't test compaction in conjunction with deletion and new data, it's too
            // many combinations with no clear benefit.  Feel free to update if there is
            // a need
            // TODO: enable compaction for stable row id once supported.
            let compaction_choices = if use_deleted_data || use_new_data || use_stable_row_ids {
                vec![false]
            } else {
                vec![false, true]
            };
            for use_compaction in compaction_choices {
                let updated_choices = if use_deleted_data || use_new_data || use_compaction {
                    vec![false]
                } else {
                    vec![false, true]
                };
                for use_updated in updated_choices {
                    for with_row_id in [false, true] {
                        let params = ScalarTestParams {
                            use_index,
                            use_projection,
                            use_deleted_data,
                            use_new_data,
                            with_row_id,
                            use_compaction,
                            use_updated,
                        };
                        fixture.check_vector_queries(&params).await;
                        fixture.check_simple_queries(&params).await;
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn can_filter_row_id() {
    let dataset = lance_datagen::gen_batch()
        .col("x", array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(1000))
        .await
        .unwrap();

    let mut scan = dataset.scan();
    scan.with_row_id();
    scan.project::<&str>(&[]).unwrap();
    scan.filter("_rowid == 50").unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.column(0).as_primitive::<UInt64Type>().values()[0], 50);
}

#[rstest]
#[tokio::test]
async fn test_index_take_batch_size() {
    let fixture = Box::pin(ScalarIndexTestFixture::new(LanceFileVersion::Stable, false)).await;
    let stream = fixture
        .dataset
        .scan()
        .filter("indexed > 0")
        .unwrap()
        .batch_size(16)
        .try_into_stream()
        .await
        .unwrap();
    let batches = stream.collect::<Vec<_>>().await;
    assert_eq!(batches.len(), 1000_usize.div_ceil(16));
}

#[tokio::test]
async fn test_inexact_scalar_index_plans() {
    let data = gen_batch()
        .col("ngram", array::rand_utf8(ByteCount::from(5), false))
        .col("exact", array::rand_type(&DataType::UInt32))
        .col("no_index", array::rand_type(&DataType::UInt32))
        .into_reader_rows(RowCount::from(32), BatchCount::from(2));

    let mut dataset = Dataset::write(data, "memory://test", None).await.unwrap();
    dataset
        .create_index(
            &["ngram"],
            IndexType::NGram,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["exact"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Simple in-exact filter
    assert_plan_equals(
        &dataset,
        |scanner| scanner.filter("contains(ngram, 'test string')"),
        "LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, \
             range_before=None, range_after=None, row_id=false, row_addr=false, \
             full_filter=contains(ngram, Utf8(\"test string\")), refine_filter=--
               ScalarIndexQuery: query=[contains(ngram, Utf8(\"test string\"))]@ngram_idx(NGram)",
    )
    .await
    .unwrap();

    // Combined with exact filter
    assert_plan_equals(
        &dataset,
        |scanner| scanner.filter("contains(ngram, 'test string') and exact < 50"),
        "LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, \
            range_before=None, range_after=None, row_id=false, row_addr=false, \
            full_filter=contains(ngram, Utf8(\"test string\")) AND exact < UInt32(50), \
            refine_filter=--
              ScalarIndexQuery: query=AND([contains(ngram, Utf8(\"test string\"))]@ngram_idx(NGram),[exact < 50]@exact_idx(BTree))",
    )
    .await
    .unwrap();

    // All three filters
    assert_plan_equals(
        &dataset,
        |scanner| {
            scanner.filter("contains(ngram, 'test string') and exact < 50 AND no_index > 100")
        },
        "ProjectionExec: expr=[ngram@0 as ngram, exact@1 as exact, no_index@2 as no_index]
  LanceRead: uri=..., projection=[ngram, exact, no_index], num_fragments=1, range_before=None, \
  range_after=None, row_id=true, row_addr=false, full_filter=contains(ngram, Utf8(\"test string\")) AND exact < UInt32(50) AND no_index > UInt32(100), \
  refine_filter=no_index > UInt32(100)
    ScalarIndexQuery: query=AND([contains(ngram, Utf8(\"test string\"))]@ngram_idx(NGram),[exact < 50]@exact_idx(BTree))",
    )
    .await
    .unwrap();
}

#[rstest]
#[case::range(
    "NOT (colour = 'c41') OR (seq >= 0 AND seq < 2)",
    vec![0, 1, 2, 4, 5, 7]
)]
#[case::equality("NOT (colour = 'c41') OR seq = 0", vec![0, 2, 4, 5, 7])]
#[case::inequality("colour <> 'c41' OR seq = 0", vec![0, 2, 4, 5, 7])]
#[case::not("NOT (colour = 'c41')", vec![2, 5, 7])]
#[tokio::test]
async fn test_not_or_zone_map(
    #[case] filter: &str,
    #[case] expected_ids: Vec<i32>,
    #[values(None, Some(8))] limit: Option<i64>,
) {
    let batch = arrow_array::record_batch!(
        ("id", Int32, [0, 1, 2, 3, 4, 5, 6, 7]),
        (
            "colour",
            Utf8,
            [
                None,
                Some("c41"),
                Some("c42"),
                None,
                Some("c41"),
                Some("c42"),
                None,
                Some("c42")
            ]
        ),
        (
            "seq",
            Int32,
            [
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(0),
                Some(1),
                None,
                None
            ]
        )
    )
    .unwrap();
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    for (column, index_type, builtin) in [
        ("colour", IndexType::Bitmap, BuiltinIndexType::Bitmap),
        ("seq", IndexType::ZoneMap, BuiltinIndexType::ZoneMap),
    ] {
        dataset
            .create_index(
                &[column],
                index_type,
                None,
                &ScalarIndexParams::for_builtin(builtin),
                false,
            )
            .await
            .unwrap();
    }

    let mut full_scan = dataset.scan();
    full_scan.use_scalar_index(false).filter(filter).unwrap();
    full_scan.limit(limit, None).unwrap();
    let full_batch = full_scan.try_into_batch().await.unwrap();
    let full_ids = full_batch["id"].as_primitive::<Int32Type>().values();
    assert_eq!(full_ids.as_ref(), expected_ids.as_slice());

    let mut indexed_scan = dataset.scan();
    indexed_scan.filter(filter).unwrap();
    // Eight rows exceed the guaranteed matches, so the limit must fall back to
    // checking candidates instead of reading only the lower bound.
    indexed_scan.limit(limit, None).unwrap();
    let plan = indexed_scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    if filter.contains("seq") {
        assert!(plan.contains("ZoneMap"), "{plan}");
    }
    let indexed_batch = indexed_scan.try_into_batch().await.unwrap();
    assert_eq!(
        indexed_batch["id"].as_primitive::<Int32Type>().values(),
        full_ids
    );
}

#[tokio::test]
async fn test_fm_index_with_stable_row_ids() {
    let batch = arrow_array::record_batch!(
        (
            "text",
            Utf8,
            [
                "alpha",
                "needle in first",
                "beta",
                "first needle suffix",
                "delta",
                "needle in second",
                "epsilon",
                "second needle suffix"
            ]
        ),
        ("id", Int32, [0, 1, 2, 3, 4, 5, 6, 7])
    )
    .unwrap();
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let write_params = WriteParams {
        max_rows_per_file: 4,
        enable_stable_row_ids: true,
        ..Default::default()
    };
    let mut dataset = Dataset::write(
        reader,
        "memory://test_fm_index_with_stable_row_ids",
        Some(write_params),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::Fm);
    dataset
        .create_index(&["text"], IndexType::Fm, None, &params, true)
        .await
        .unwrap();

    let mut indexed_scan = dataset.scan();
    indexed_scan.filter("contains(text, 'needle')").unwrap();
    let indexed_plan = indexed_scan.explain_plan(false).await.unwrap();
    assert!(
        indexed_plan.contains("ScalarIndexQuery") && indexed_plan.contains("Fm"),
        "expected the FM index in the plan, got:\n{indexed_plan}"
    );
    let indexed = indexed_scan.try_into_batch().await.unwrap();

    let unindexed = dataset
        .scan()
        .use_scalar_index(false)
        .filter("contains(text, 'needle')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let indexed_ids = indexed["id"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let unindexed_ids = unindexed["id"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

    assert_eq!(unindexed_ids, BTreeSet::from([1, 3, 5, 7]));
    assert_eq!(indexed_ids, unindexed_ids);
}

#[tokio::test]
async fn test_ngram_regex_index_scan() {
    use arrow::array::AsArray;

    // A small, fixed corpus written across multiple fragments so the ngram
    // index spans fragment boundaries.
    let values = [
        "rhino",       // 0
        "rhinos nose", // 1
        "cat",         // 2
        "dog",         // 3
        "cat dog",     // 4
        "elephant",    // 5
        "catalog",     // 6
        "scatter",     // 7
        "rhino horn",  // 8
        "mouse",       // 9
        "category",    // 10
        "dogma",       // 11
    ];
    let array = StringArray::from_iter_values(values);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let write_params = WriteParams {
        max_rows_per_file: 4, // 12 rows -> 3 fragments
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, "memory://test_ngram_regex", Some(write_params))
        .await
        .unwrap();
    dataset
        .create_index(
            &["s"],
            IndexType::NGram,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    assert!(
        dataset.get_fragments().len() > 1,
        "expected a multi-fragment dataset"
    );

    // Scan with `filter` and return the matched `s` values, sorted.
    async fn matched(dataset: &Dataset, filter: &str) -> Vec<String> {
        let mut scan = dataset.scan();
        scan.filter(filter).unwrap();
        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut out = Vec::new();
        for batch in batches {
            let col = batch.column_by_name("s").unwrap().as_string::<i32>();
            out.extend(col.iter().flatten().map(|s| s.to_string()));
        }
        out.sort();
        out
    }

    // `regexp_like`: a plain literal substring.
    assert_eq!(
        matched(&dataset, "regexp_like(s, 'rhino')").await,
        ["rhino", "rhino horn", "rhinos nose"]
    );
    // `regexp_match` (coerced to `IsNotNull(regexp_match(...))`) accelerates too.
    assert_eq!(
        matched(&dataset, "regexp_match(s, 'rhino')").await,
        ["rhino", "rhino horn", "rhinos nose"]
    );
    // Anchored: recheck must drop trigram false positives -- the `cat`
    // trigram also occurs in cat dog / catalog / scatter / category.
    assert_eq!(matched(&dataset, "regexp_like(s, 'cat$')").await, ["cat"]);
    // AND across `.*`: row 8 ("rhino horn") shares the rhino trigrams but
    // lacks the nose trigrams, so only "rhinos nose" survives.
    assert_eq!(
        matched(&dataset, "regexp_like(s, 'rhino.*nose')").await,
        ["rhinos nose"]
    );
    // Alternation.
    assert_eq!(
        matched(&dataset, "regexp_like(s, '(catalog|elephant)')").await,
        ["catalog", "elephant"]
    );
    // A non-accelerable pattern (no trigram derivable) still returns correct
    // results via a full recheck.
    assert_eq!(matched(&dataset, "regexp_like(s, 'o.m')").await, ["dogma"]);
    // A case-insensitive flag is not accelerated (the index normalization
    // disagrees with Unicode case folding) but must still return correct
    // results via a full recheck -- here matching despite the upper-case
    // pattern. This exercises the three-argument `regexp_like` flags path.
    assert_eq!(
        matched(&dataset, "regexp_like(s, 'RHINO', 'i')").await,
        ["rhino", "rhino horn", "rhinos nose"]
    );

    // Infix LIKE is accelerated through the same machinery (a plain-literal
    // `regexp_like` is rewritten to LIKE before it reaches the index).
    assert_eq!(
        matched(&dataset, "s LIKE '%rhino%'").await,
        ["rhino", "rhino horn", "rhinos nose"]
    );
    // Prefix LIKE: recheck drops "scatter", which contains the `cat` trigram
    // but does not start with "cat".
    assert_eq!(
        matched(&dataset, "s LIKE 'cat%'").await,
        ["cat", "cat dog", "catalog", "category"]
    );

    // The ngram index is actually engaged for every accelerated form.
    for filter in [
        "regexp_like(s, 'rhino')",
        "regexp_match(s, 'rhino')",
        "s LIKE '%rhino%'",
    ] {
        let mut scan = dataset.scan();
        scan.filter(filter).unwrap();
        let plan = scan.create_plan().await.unwrap();
        let plan_str = format!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
        assert!(
            plan_str.contains("ScalarIndexQuery") && plan_str.contains("NGram"),
            "expected ngram index usage for `{filter}`, got plan:\n{plan_str}"
        );
    }

    // contains with >= 3 characters (uses index)
    assert_eq!(
        matched(&dataset, "contains(s, 'rhino')").await,
        ["rhino", "rhino horn", "rhinos nose"]
    );
    assert_eq!(
        matched(&dataset, "contains(s, 'cat')").await,
        ["cat", "cat dog", "catalog", "category", "scatter"]
    );

    // contains with < 3 characters (must NOT use index, i.e., falls back to full scan, and returns correct results)
    assert_eq!(
        matched(&dataset, "contains(s, 'ca')").await,
        ["cat", "cat dog", "catalog", "category", "scatter"]
    );
    assert_eq!(
        matched(&dataset, "contains(s, 'a')").await,
        [
            "cat", "cat dog", "catalog", "category", "dogma", "elephant", "scatter"
        ]
    );

    // Verify index is used for contains >= 3 characters, but NOT used for < 3 characters
    for filter in ["contains(s, 'rhino')", "contains(s, 'cat')"] {
        let mut scan = dataset.scan();
        scan.filter(filter).unwrap();
        let plan = scan.create_plan().await.unwrap();
        let plan_str = format!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
        assert!(
            plan_str.contains("ScalarIndexQuery") && plan_str.contains("NGram"),
            "expected ngram index usage for `{filter}`, got plan:\n{plan_str}"
        );
    }
    for filter in ["contains(s, 'ca')", "contains(s, 'a')"] {
        let mut scan = dataset.scan();
        scan.filter(filter).unwrap();
        let plan = scan.create_plan().await.unwrap();
        let plan_str = format!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
        assert!(
            !plan_str.contains("ScalarIndexQuery"),
            "expected NO ngram index usage for `{filter}`, got plan:\n{plan_str}"
        );
    }
}

#[tokio::test]
async fn test_ngram_regex_non_accelerable_recheck() {
    // `a.b` yields no trigram, so the index returns "recheck everything".
    // This must still produce ALL correct matches across fragments, not an
    // empty set (a regression test for the AtLeast recheck path, which a
    // single-match case would not catch).
    let unit = ["acb", "dog", "axb", "cat", "qqq", "rhino"];
    let values: Vec<&str> = unit.iter().copied().cycle().take(60).collect();
    let array = StringArray::from_iter_values(values);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "text",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let write_params = WriteParams {
        max_rows_per_file: 20, // 60 rows -> 3 fragments
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, "memory://test_ngram_regex_ne", Some(write_params))
        .await
        .unwrap();
    dataset
        .create_index(
            &["text"],
            IndexType::NGram,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    async fn count(dataset: &Dataset, filter: &str) -> usize {
        let mut scan = dataset.scan();
        scan.filter(filter).unwrap();
        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        batches.iter().map(|b| b.num_rows()).sum()
    }

    // "acb" and "axb" each appear 10 times in the 60 rows -> 20 matches.
    assert_eq!(count(&dataset, "regexp_match(text, 'a.b')").await, 20);
    assert_eq!(count(&dataset, "regexp_like(text, 'a.b')").await, 20);
}

#[tokio::test]
async fn test_ngram_contains_untokenizable_needle() {
    // A needle the tokenizer cannot turn into a trigram must fall back to a
    // full recheck rather than silently matching nothing.  Byte length is
    // not a usable proxy for this: "éé" is two characters but four bytes.
    let unit = ["ééb", "aéé", "abc", "dog", "éab"];
    let values: Vec<&str> = unit.iter().copied().cycle().take(60).collect();
    let array = StringArray::from_iter_values(values);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "text",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let write_params = WriteParams {
        max_rows_per_file: 20, // 60 rows -> 3 fragments
        ..Default::default()
    };
    let mut dataset = Dataset::write(
        reader,
        "memory://test_ngram_contains_utf8",
        Some(write_params),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["text"],
            IndexType::NGram,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    async fn count(dataset: &Dataset, filter: &str) -> usize {
        let mut scan = dataset.scan();
        scan.filter(filter).unwrap();
        let batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        batches.iter().map(|b| b.num_rows()).sum()
    }

    async fn plan_of(dataset: &Dataset, filter: &str) -> String {
        let mut scan = dataset.scan();
        scan.filter(filter).unwrap();
        let plan = scan.create_plan().await.unwrap();
        format!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        )
    }

    // Each unit value appears 12 times in the 60 rows.
    // Two characters (four bytes): shorter than a trigram, so the index is
    // bypassed and "ééb" / "aéé" are still found.
    assert_eq!(count(&dataset, "contains(text, 'éé')").await, 24);
    let plan_str = plan_of(&dataset, "contains(text, 'éé')").await;
    assert!(
        !plan_str.contains("ScalarIndexQuery"),
        "expected NO ngram index usage for a two character needle, got plan:\n{plan_str}"
    );

    // Three characters: long enough to tokenize, so the index is used.
    assert_eq!(count(&dataset, "contains(text, 'ééb')").await, 12);
    let plan_str = plan_of(&dataset, "contains(text, 'ééb')").await;
    assert!(
        plan_str.contains("ScalarIndexQuery") && plan_str.contains("NGram"),
        "expected ngram index usage for a three character needle, got plan:\n{plan_str}"
    );
}

#[tokio::test]
async fn test_like_prefix_with_btree_index() {
    // Create dataset with string data that has various prefixes
    // Avoid LIKE special characters (%, _) in data to keep tests simple
    let data = gen_batch()
        .col(
            "name",
            array::cycle_utf8_literals(&[
                "apple",
                "application",
                "app",
                "banana",
                "band",
                "testns1",
                "testns2",
                "test",
                "testing",
                "zoo",
            ]),
        )
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));

    let mut dataset = Dataset::write(data, "memory://test_like", None)
        .await
        .unwrap();

    // Create BTree index on string column
    dataset
        .create_index(
            &["name"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Test 1: Verify LIKE 'app%' uses scalar index and returns correct results
    assert_plan_equals(
        &dataset,
        |scanner| scanner.filter("name LIKE 'app%'"),
        "LanceRead: uri=..., projection=[name, id], num_fragments=1, \
             range_before=None, range_after=None, row_id=false, row_addr=false, \
             full_filter=name LIKE Utf8(\"app%\"), refine_filter=--
               ScalarIndexQuery: query=[name LIKE 'app%']@name_idx(BTree)",
    )
    .await
    .unwrap();

    // Verify correct results for LIKE 'app%'
    let results = dataset
        .scan()
        .filter("name LIKE 'app%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let names: Vec<&str> = results
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap())
        .collect();
    // Should match: apple, application, app (repeated in cycle)
    assert!(names.iter().all(|n| n.starts_with("app")));
    assert!(!names.is_empty());

    // Test 2: Verify starts_with() uses scalar index (simple prefix without special chars)
    // Note: DataFusion optimizes starts_with() to LIKE before our index planning
    assert_plan_equals(
        &dataset,
        |scanner| scanner.filter("starts_with(name, 'ban')"),
        "LanceRead: uri=..., projection=[name, id], num_fragments=1, \
             range_before=None, range_after=None, row_id=false, row_addr=false, \
             full_filter=name LIKE Utf8(\"ban%\"), refine_filter=--
               ScalarIndexQuery: query=[name LIKE 'ban%']@name_idx(BTree)",
    )
    .await
    .unwrap();

    // Verify correct results for starts_with
    let results = dataset
        .scan()
        .filter("starts_with(name, 'ban')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let names: Vec<&str> = results
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap())
        .collect();
    // Should match: banana, band
    assert!(names.iter().all(|n| n.starts_with("ban")));
    assert!(!names.is_empty());

    // Test 3: LIKE with pattern requiring refine (e.g., 'test%2')
    assert_plan_equals(
        &dataset,
        |scanner| scanner.filter("name LIKE 'test%2'"),
        "ProjectionExec: expr=[name@0 as name, id@1 as id]
  LanceRead: uri=..., projection=[name, id], num_fragments=1, \
range_before=None, range_after=None, row_id=true, row_addr=false, \
full_filter=name LIKE Utf8(\"test%2\"), refine_filter=name LIKE Utf8(\"test%2\")
    ScalarIndexQuery: query=[name LIKE 'test%']@name_idx(BTree)",
    )
    .await
    .unwrap();

    // Verify correct results for LIKE 'test%2' (needs refine)
    let results = dataset
        .scan()
        .filter("name LIKE 'test%2'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let names: Vec<&str> = results
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap())
        .collect();
    // Should match: testns2 (ends with '2')
    assert!(
        names
            .iter()
            .all(|n| n.starts_with("test") && n.ends_with("2"))
    );

    // Test 4: LIKE starting with wildcard should NOT use scalar index for pruning
    // Verify by checking the plan does NOT have ScalarIndexQuery
    let mut scanner = dataset.scan();
    scanner.filter("name LIKE '%app%'").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    assert!(
        !plan_str.contains("ScalarIndexQuery"),
        "LIKE '%app%' should not use scalar index, but got: {}",
        plan_str
    );

    // Verify correct results for LIKE '%app%'
    let results = dataset
        .scan()
        .filter("name LIKE '%app%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let names: Vec<&str> = results
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap())
        .collect();
    // Should match: apple, application, app (contain 'app')
    assert!(names.iter().all(|n| n.contains("app")));

    // Test 5: NOT LIKE should NOT use scalar index
    let mut scanner = dataset.scan();
    scanner.filter("name NOT LIKE 'app%'").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    assert!(
        !plan_str.contains("ScalarIndexQuery"),
        "NOT LIKE should not use scalar index, but got: {}",
        plan_str
    );
}

#[tokio::test]
async fn test_like_prefix_correctness_with_btree_index() {
    // Create dataset with deterministic string data for exact result verification
    let names: Vec<&str> = vec![
        "alpha", "alphabet", "beta", "gamma", "delta", "epsilon", "eta", "theta", "iota", "kappa",
    ];
    let data = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("name", DataType::Utf8, false),
            ArrowField::new("id", DataType::Int32, false),
        ])),
        vec![
            Arc::new(StringArray::from(names.clone())),
            Arc::new(Int32Array::from_iter_values(0..10)),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(
        vec![Ok(data)],
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("name", DataType::Utf8, false),
            ArrowField::new("id", DataType::Int32, false),
        ])),
    );

    let mut dataset = Dataset::write(reader, "memory://test_like_correctness", None)
        .await
        .unwrap();

    // Create BTree index
    dataset
        .create_index(
            &["name"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Test with index
    let with_index = dataset
        .scan()
        .filter("name LIKE 'alpha%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    // Test without index (for comparison)
    let without_index = dataset
        .scan()
        .use_scalar_index(false)
        .filter("name LIKE 'alpha%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    // Both should return same results: alpha, alphabet
    assert_eq!(with_index.num_rows(), without_index.num_rows());
    assert_eq!(with_index.num_rows(), 2);

    let with_index_names: BTreeSet<String> = with_index
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect();

    let without_index_names: BTreeSet<String> = without_index
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect();

    assert_eq!(with_index_names, without_index_names);
    assert_eq!(
        with_index_names,
        BTreeSet::from(["alpha".to_string(), "alphabet".to_string()])
    );

    // Test starts_with correctness
    let starts_with_result = dataset
        .scan()
        .filter("starts_with(name, 'e')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let starts_with_names: BTreeSet<String> = starts_with_result
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect();

    // Should match: epsilon, eta
    assert_eq!(
        starts_with_names,
        BTreeSet::from(["epsilon".to_string(), "eta".to_string()])
    );
}

#[tokio::test]
async fn test_like_prefix_with_zone_map() {
    use lance_index::scalar::BuiltinIndexType;

    // Create dataset with string data that has various prefixes
    let data = gen_batch()
        .col(
            "name",
            array::cycle_utf8_literals(&[
                "apple",
                "application",
                "app",
                "banana",
                "band",
                "testns1",
                "testns2",
                "test",
                "testing",
                "zoo",
            ]),
        )
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));

    let mut dataset = Dataset::write(data, "memory://test_like_zonemap", None)
        .await
        .unwrap();

    // Create ZoneMap index on string column
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
    dataset
        .create_index(
            &["name"],
            IndexType::Scalar,
            Some("name_zonemap".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Test 1: Verify LIKE 'app%' uses zone map index
    let mut scanner = dataset.scan();
    scanner.filter("name LIKE 'app%'").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    // Zone map uses ScalarIndexExec with LikePrefix query
    assert!(
        plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix"),
        "LIKE 'app%' should use zone map index with LikePrefix, but got: {}",
        plan_str
    );

    // Verify correct results for LIKE 'app%'
    let results = dataset
        .scan()
        .filter("name LIKE 'app%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let names: Vec<&str> = results
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap())
        .collect();
    assert!(names.iter().all(|n| n.starts_with("app")));
    assert!(!names.is_empty());

    // Test 2: Verify starts_with() uses zone map index
    let mut scanner = dataset.scan();
    scanner.filter("starts_with(name, 'ban')").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    assert!(
        plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix"),
        "starts_with should use zone map index with LikePrefix, but got: {}",
        plan_str
    );

    // Verify correct results
    let results = dataset
        .scan()
        .filter("starts_with(name, 'ban')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let names: Vec<&str> = results
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap())
        .collect();
    assert!(names.iter().all(|n| n.starts_with("ban")));

    // Test 3: LIKE with refine pattern still uses zone map for prefix pruning
    let mut scanner = dataset.scan();
    scanner.filter("name LIKE 'test%2'").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    assert!(
        plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix"),
        "LIKE 'test%2' should use zone map index for prefix, but got: {}",
        plan_str
    );

    // Test 4: LIKE starting with wildcard should NOT use zone map
    let mut scanner = dataset.scan();
    scanner.filter("name LIKE '%app%'").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    assert!(
        !plan_str.contains("LikePrefix"),
        "LIKE '%app%' should not use LikePrefix index, but got: {}",
        plan_str
    );
}

/// Regression for over-matching on the zone-map recheck path: a literal prefix
/// containing LIKE metacharacters (`_`, `%`) must be matched literally, not as
/// wildcards. The indexed (recheck) result must equal the unindexed ground truth.
#[tokio::test]
async fn test_like_prefix_zone_map_escapes_metacharacters() {
    use lance_index::scalar::BuiltinIndexType;

    let names: Vec<&str> = vec!["a_b", "a_c", "axb", "a1c", "b%c", "bxc", "bcc", "zoo"];
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("name", DataType::Utf8, false),
        ArrowField::new("id", DataType::Int32, false),
    ]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(names.clone())),
            Arc::new(Int32Array::from_iter_values(0..names.len() as i32)),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
    let mut dataset = Dataset::write(reader, "memory://test_like_zonemap_escape", None)
        .await
        .unwrap();

    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
    dataset
        .create_index(
            &["name"],
            IndexType::Scalar,
            Some("name_zonemap".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    let collect_names = |batch: &RecordBatch| -> BTreeSet<String> {
        batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect()
    };

    // Ensure the predicate actually exercises the zone-map LikePrefix recheck path.
    let mut scanner = dataset.scan();
    scanner.filter("starts_with(name, 'a_')").unwrap();
    let plan_str = format!("{:?}", scanner.create_plan().await.unwrap());
    assert!(
        plan_str.contains("LikePrefix"),
        "expected a zone-map LikePrefix plan, got: {plan_str}"
    );

    // `_` and `%` in the prefix are literal characters; the indexed result must match
    // the unindexed evaluation for each predicate (no wildcard over-match).
    for predicate in ["starts_with(name, 'a_')", "starts_with(name, 'b%')"] {
        let with_index = dataset
            .scan()
            .filter(predicate)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let without_index = dataset
            .scan()
            .use_scalar_index(false)
            .filter(predicate)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            collect_names(&with_index),
            collect_names(&without_index),
            "indexed result over-matched for predicate `{predicate}`"
        );
    }

    // Explicit expectation so the intended (non-over-matching) result is obvious.
    let result = dataset
        .scan()
        .filter("starts_with(name, 'a_')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        collect_names(&result),
        BTreeSet::from(["a_b".to_string(), "a_c".to_string()])
    );
}

/// A bitmap index cannot answer prefix queries, so `LIKE 'prefix%'` / `starts_with`
/// must fall back to ordinary filtering (returning correct results) instead of being
/// planned as a `LikePrefix` index scan that bitmap search would reject.
#[tokio::test]
async fn test_like_prefix_bitmap_falls_back_to_filter() {
    use lance_index::scalar::BuiltinIndexType;

    let names: Vec<&str> = vec!["apple", "app", "application", "banana", "band", "zoo"];
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("name", DataType::Utf8, false),
        ArrowField::new("id", DataType::Int32, false),
    ]));
    let data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(names.clone())),
            Arc::new(Int32Array::from_iter_values(0..names.len() as i32)),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(data)], schema.clone());
    let mut dataset = Dataset::write(reader, "memory://test_like_bitmap_fallback", None)
        .await
        .unwrap();

    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    dataset
        .create_index(
            &["name"],
            IndexType::Scalar,
            Some("name_bitmap".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // The predicate must not be planned as a LikePrefix index scan (bitmap rejects it).
    let mut scanner = dataset.scan();
    scanner.filter("name LIKE 'app%'").unwrap();
    let plan_str = format!("{:?}", scanner.create_plan().await.unwrap());
    assert!(
        !plan_str.contains("LikePrefix"),
        "bitmap LIKE must not use a LikePrefix index scan, got: {plan_str}"
    );

    let collect_names = |batch: &RecordBatch| -> BTreeSet<String> {
        batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap().to_string())
            .collect()
    };

    // And it must execute successfully with correct results (previously errored).
    let result = dataset
        .scan()
        .filter("name LIKE 'app%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        collect_names(&result),
        BTreeSet::from([
            "apple".to_string(),
            "app".to_string(),
            "application".to_string(),
        ])
    );

    let result = dataset
        .scan()
        .filter("starts_with(name, 'ban')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        collect_names(&result),
        BTreeSet::from(["banana".to_string(), "band".to_string()])
    );
}

/// Build an in-memory dataset with a single `Dictionary(Int16, Utf8)` column.
/// The dictionary cycles through "a", "b", "c" so each value appears in a
/// predictable, repeated pattern.
async fn dictionary_string_dataset() -> Dataset {
    use arrow_array::{Int16Array, Int16DictionaryArray};

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "etld",
        DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
        false,
    )]));

    let dictionary = Arc::new(StringArray::from(vec!["a", "b", "c"]));
    let indices = Int16Array::from((0..30).map(|i| i % 3).collect::<Vec<_>>());
    let dict_array = Int16DictionaryArray::try_new(indices, dictionary).unwrap();

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(dict_array)]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(reader, "memory://test_dict_filter", None)
        .await
        .unwrap()
}

/// Regression test for filtering a dictionary-encoded string column via the
/// SQL string path (`Scanner::filter`). This used to fail to plan with
/// "could not convert to literal of type 'Dictionary(Int16, Utf8)'".
#[tokio::test]
async fn test_filter_on_dictionary_string_column() {
    let dataset = dictionary_string_dataset().await;

    // Equality predicate.
    let count = dataset
        .scan()
        .filter("etld = 'a'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap()
        .num_rows();
    assert_eq!(count, 10);

    // IN-list predicate.
    let count = dataset
        .scan()
        .filter("etld IN ('a', 'b')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap()
        .num_rows();
    assert_eq!(count, 20);
}

/// An `IN`/`=` predicate on a dictionary column with a scalar index should be
/// pushed down to the index rather than falling back to a full scan.
#[tokio::test]
async fn test_dictionary_string_column_uses_scalar_index() {
    use lance_index::scalar::BuiltinIndexType;

    let mut dataset = dictionary_string_dataset().await;
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    dataset
        .create_index(&["etld"], IndexType::Scalar, None, &params, true)
        .await
        .unwrap();

    let mut scanner = dataset.scan();
    scanner.filter("etld IN ('a', 'b')").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    assert!(
        plan_str.contains("ScalarIndexExec") || plan_str.contains("MaterializeIndex"),
        "IN on a dictionary column should use the scalar index, but got: {}",
        plan_str
    );

    let count = dataset
        .scan()
        .filter("etld IN ('a', 'b')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap()
        .num_rows();
    assert_eq!(count, 20);
}

#[tokio::test]
async fn test_like_prefix_with_segmented_zone_map() {
    use lance_index::scalar::BuiltinIndexType;

    let data = gen_batch()
        .col(
            "name",
            array::cycle_utf8_literals(&[
                "apple",
                "application",
                "app",
                "banana",
                "band",
                "testns1",
                "testns2",
                "test",
                "testing",
                "zoo",
            ]),
        )
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(150), BatchCount::from(6));

    let write_params = WriteParams {
        max_rows_per_file: 25,
        max_rows_per_group: 10,
        ..Default::default()
    };

    let mut dataset = Dataset::write(
        data,
        "memory://test_like_segmented_zonemap",
        Some(write_params),
    )
    .await
    .unwrap();

    let fragments = dataset.get_fragments();
    assert!(fragments.len() > 1, "expected multiple fragments");

    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
    let mut segments = Vec::with_capacity(fragments.len());
    for fragment in &fragments {
        let mut builder = dataset.create_index_builder(&["name"], IndexType::Scalar, &params);
        builder = builder
            .name("name_zonemap".to_string())
            .fragments(vec![fragment.id() as u32]);
        segments.push(builder.execute_uncommitted().await.unwrap());
    }

    dataset
        .commit_existing_index_segments("name_zonemap", "name", segments)
        .await
        .unwrap();

    let committed = dataset.load_indices_by_name("name_zonemap").await.unwrap();
    assert_eq!(committed.len(), fragments.len());

    let mut scanner = dataset.scan();
    scanner.filter("name LIKE 'app%'").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    assert!(
        plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix"),
        "segmented zonemap should use LikePrefix pruning, but got: {}",
        plan_str
    );

    let with_index = dataset
        .scan()
        .filter("name LIKE 'app%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let without_index = dataset
        .scan()
        .use_scalar_index(false)
        .filter("name LIKE 'app%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let with_index_ids = with_index
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let without_index_ids = without_index
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(with_index_ids, without_index_ids);
    assert!(!with_index_ids.is_empty());

    let names = with_index
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|value| value.unwrap())
        .collect::<Vec<_>>();
    assert!(names.iter().all(|name| name.starts_with("app")));
}

#[tokio::test]
async fn test_like_prefix_with_segmented_btree() {
    let data = gen_batch()
        .col(
            "name",
            array::cycle_utf8_literals(&[
                "apple",
                "application",
                "app",
                "banana",
                "band",
                "testns1",
                "testns2",
                "test",
                "testing",
                "zoo",
            ]),
        )
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(150), BatchCount::from(6));

    let write_params = WriteParams {
        max_rows_per_file: 25,
        max_rows_per_group: 10,
        ..Default::default()
    };

    let mut dataset = Dataset::write(
        data,
        "memory://test_like_segmented_btree",
        Some(write_params),
    )
    .await
    .unwrap();

    let fragments = dataset.get_fragments();
    assert!(fragments.len() > 1, "expected multiple fragments");

    let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
    let mut segments = Vec::with_capacity(fragments.len());
    for fragment in &fragments {
        let mut builder = dataset.create_index_builder(&["name"], IndexType::BTree, &params);
        builder = builder
            .name("name_btree".to_string())
            .fragments(vec![fragment.id() as u32]);
        segments.push(builder.execute_uncommitted().await.unwrap());
    }

    dataset
        .commit_existing_index_segments("name_btree", "name", segments)
        .await
        .unwrap();

    let committed = dataset.load_indices_by_name("name_btree").await.unwrap();
    assert_eq!(committed.len(), fragments.len());

    let mut scanner = dataset.scan();
    scanner.filter("name LIKE 'app%'").unwrap();
    let plan = scanner.create_plan().await.unwrap();
    let plan_str = format!("{:?}", plan);
    assert!(
        plan_str.contains("ScalarIndexExec") && plan_str.contains("LikePrefix(Utf8(\"app\"))"),
        "segmented btree should use scalar index pruning, but got: {}",
        plan_str
    );

    let with_index = dataset
        .scan()
        .filter("name LIKE 'app%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let without_index = dataset
        .scan()
        .use_scalar_index(false)
        .filter("name LIKE 'app%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let with_index_ids = with_index
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let without_index_ids = without_index
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(with_index_ids, without_index_ids);
    assert!(!with_index_ids.is_empty());

    let names = with_index
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|value| value.unwrap())
        .collect::<Vec<_>>();
    assert!(names.iter().all(|name| name.starts_with("app")));
}

#[tokio::test]
async fn test_like_prefix_correctness_with_zone_map() {
    use lance_index::scalar::BuiltinIndexType;

    // Create dataset with deterministic string data for exact result verification
    let names: Vec<&str> = vec![
        "alpha", "alphabet", "beta", "gamma", "delta", "epsilon", "eta", "theta", "iota", "kappa",
    ];
    let data = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("name", DataType::Utf8, false),
            ArrowField::new("id", DataType::Int32, false),
        ])),
        vec![
            Arc::new(StringArray::from(names.clone())),
            Arc::new(Int32Array::from_iter_values(0..10)),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(
        vec![Ok(data)],
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("name", DataType::Utf8, false),
            ArrowField::new("id", DataType::Int32, false),
        ])),
    );

    let mut dataset = Dataset::write(reader, "memory://test_like_correctness_zonemap", None)
        .await
        .unwrap();

    // Create ZoneMap index
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
    dataset
        .create_index(
            &["name"],
            IndexType::Scalar,
            Some("name_zonemap".to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    // Test with zone map index
    let with_index = dataset
        .scan()
        .filter("name LIKE 'alpha%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    // Test without index (for comparison)
    let without_index = dataset
        .scan()
        .use_scalar_index(false)
        .filter("name LIKE 'alpha%'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    // Both should return same results: alpha, alphabet
    assert_eq!(with_index.num_rows(), without_index.num_rows());
    assert_eq!(with_index.num_rows(), 2);

    let with_index_names: BTreeSet<String> = with_index
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect();

    let without_index_names: BTreeSet<String> = without_index
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect();

    assert_eq!(with_index_names, without_index_names);
    assert_eq!(
        with_index_names,
        BTreeSet::from(["alpha".to_string(), "alphabet".to_string()])
    );

    // Test starts_with correctness with zone map
    let starts_with_result = dataset
        .scan()
        .filter("starts_with(name, 'e')")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let starts_with_names: BTreeSet<String> = starts_with_result
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect();

    // Should match: epsilon, eta
    assert_eq!(
        starts_with_names,
        BTreeSet::from(["epsilon".to_string(), "eta".to_string()])
    );
}
