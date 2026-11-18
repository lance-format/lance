// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod basic;
mod concurrency;
/// `UpdateIf` / `DeleteIf` conditions over target columns are evaluated by
/// `MergeConditionExec` on the rows they gate, never through the target scan.
mod deferred_conditions;
mod indexed;
mod semantics;
mod subcols;
mod subcols_indexed;
mod types;
mod units;

use super::*;
use crate::dataset::scanner::ColumnOrdering;
use crate::dataset::transaction::Operation;
use crate::dataset::transaction::UpdateMode::RewriteColumns;
use crate::dataset::write::CommitBuilder;
use crate::dataset::write::merge_insert::inserted_rows::{
    KeyExistenceFilter, KeyExistenceFilterBuilder, extract_key_value_from_batch,
};
use crate::dataset::write::merge_insert::job::{
    fragment_update_session_context, merge_insert_execution_options,
};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;
use crate::io::commit::read_transaction_file;
use crate::{
    dataset::{
        ExternalBlobMode, InsertBuilder, ReadParams, WriteDestination, WriteMode, WriteParams,
        builder::DatasetBuilder,
    },
    session::Session,
    utils::test::{
        DatagenExt, FragmentCount, FragmentRowCount, ThrottledStoreWrapper, assert_string_matches,
    },
};
use arrow_array::ArrayRef;
use arrow_array::UInt64Array;
use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array, ListArray,
    NullArray, RecordBatchIterator, RecordBatchReader, StringArray, StructArray, UInt32Array,
    types::{Int32Type, UInt32Type},
};
use arrow_array::{RecordBatch, record_batch};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::concat::concat_batches;
use datafusion::common::Column;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{JoinType, LogicalPlan};
use datafusion::physical_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::{FutureExt, StreamExt, TryStreamExt, future::try_join_all};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_datafusion::{
    datagen::DatafusionDatagenExt, exec::LanceExecutionOptions, utils::reader_to_stream,
};
use lance_datagen::{BatchCount, Dimension, RowCount, Seed, array};
use lance_file::version::LanceFileVersion;
use lance_index::IndexCriteria;
use lance_index::IndexType;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams};
use lance_io::object_store::ObjectStoreParams;
use lance_linalg::distance::MetricType;
use lance_table::feature_flags::FLAG_MIXED_DATA_FILE_VERSIONS;
use mock_instant::thread_local::MockClock;
use object_store::throttle::ThrottleConfig;
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;
use tokio::sync::{Barrier, Notify};
use url::Url;

// Used to validate that futures returned are Send.
fn assert_send<T: Send>(t: T) -> T {
    t
}

async fn check_then_refresh_dataset(
    new_data: RecordBatch,
    mut job: MergeInsertJob,
    keys_from_left: &[u32],
    keys_from_right: &[u32],
    stats: &[u64],
) -> Arc<Dataset> {
    let mut dataset = (*job.dataset).clone();
    dataset.restore().await.unwrap();
    job.dataset = Arc::new(dataset);

    let schema = new_data.schema();
    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_data)], schema.clone()));
    let new_stream = reader_to_stream(new_reader);

    let (merged_dataset, merge_stats) = job.execute(new_stream).boxed().await.unwrap();

    let batches = merged_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let merged = concat_batches(&schema, &batches).unwrap();

    let keyvals = merged
        .column(0)
        .as_primitive::<UInt32Type>()
        .values()
        .iter()
        .zip(
            merged
                .column(1)
                .as_primitive::<UInt32Type>()
                .values()
                .iter(),
        );
    let mut left_keys = keyvals
        .clone()
        .filter(|&(_, &val)| val == 1)
        .map(|(key, _)| key)
        .copied()
        .collect::<Vec<_>>();
    let mut right_keys = keyvals
        .clone()
        .filter(|&(_, &val)| val == 2)
        .map(|(key, _)| key)
        .copied()
        .collect::<Vec<_>>();
    left_keys.sort();
    right_keys.sort();
    assert_eq!(left_keys, keys_from_left);
    assert_eq!(right_keys, keys_from_right);
    assert_eq!(merge_stats.num_inserted_rows, stats[0]);
    assert_eq!(merge_stats.num_updated_rows, stats[1]);
    assert_eq!(merge_stats.num_deleted_rows, stats[2]);

    merged_dataset
}

pub(super) fn create_test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, true),
        Field::new("value", DataType::UInt32, true),
        Field::new("filterme", DataType::Utf8, true),
    ]))
}

pub(super) fn create_new_batch(schema: Arc<Schema>) -> RecordBatch {
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt32Array::from(vec![4, 5, 6, 7, 8, 9])),
            Arc::new(UInt32Array::from(vec![2, 2, 2, 2, 2, 2])),
            Arc::new(StringArray::from(vec!["A", "B", "C", "A", "B", "C"])),
        ],
    )
    .unwrap()
}

pub(super) async fn create_test_dataset(
    test_uri: &str,
    version: LanceFileVersion,
    enable_stable_row_ids: bool,
) -> Arc<Dataset> {
    let dataset = lance_datagen::gen_batch()
        .col("key", array::step_custom::<UInt32Type>(1, 1))
        .col("value", array::fill::<UInt32Type>(1u32))
        .col(
            "filterme",
            array::cycle_utf8_literals(&["A", "B", "A", "A", "B", "A"]),
        )
        .into_dataset_with_params(
            test_uri,
            FragmentCount(2),
            FragmentRowCount(3),
            Some(WriteParams {
                max_rows_per_file: 3,
                data_storage_version: Some(version),
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    assert_eq!(2, dataset.get_fragments().len());

    Arc::new(dataset)
}

async fn get_row_ids_for_keys(dataset: &Dataset, keys: &[u32]) -> UInt64Array {
    let filter = format!(
        "key IN ({})",
        keys.iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    let batch = dataset
        .scan()
        .filter(&filter)
        .unwrap()
        .with_row_id()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "key".to_string(),
        )]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    batch
        .column_by_name(ROW_ID)
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .clone()
}

fn create_delete_condition() -> Expr {
    Expr::gt(
        Expr::Column(Column::new_unqualified("key")),
        Expr::Literal(ScalarValue::UInt32(Some(1)), None),
    )
}

struct MergeInsertTestBuilder {
    version: LanceFileVersion,
    enable_stable_row_ids: bool,
    test_keys: Vec<u32>,
    expected_left_keys: Vec<u32>,
    expected_right_keys: Vec<u32>,
    expected_stats: Vec<u64>,
    job_builder: Option<Box<dyn FnOnce(Arc<Dataset>) -> MergeInsertJob>>,
}

impl MergeInsertTestBuilder {
    fn new() -> Self {
        Self {
            version: LanceFileVersion::default(),
            enable_stable_row_ids: false,
            test_keys: vec![],
            expected_left_keys: vec![],
            expected_right_keys: vec![],
            expected_stats: vec![],
            job_builder: None,
        }
    }

    fn with_version(mut self, version: LanceFileVersion) -> Self {
        self.version = version;
        self
    }

    fn with_stable_row_ids(mut self, enable: bool) -> Self {
        self.enable_stable_row_ids = enable;
        self
    }

    fn with_test_keys(mut self, keys: &[u32]) -> Self {
        self.test_keys = keys.to_vec();
        self
    }

    fn with_expected_left_keys(mut self, keys: &[u32]) -> Self {
        self.expected_left_keys = keys.to_vec();
        self
    }

    fn with_expected_right_keys(mut self, keys: &[u32]) -> Self {
        self.expected_right_keys = keys.to_vec();
        self
    }

    fn with_expected_stats(mut self, stats: &[u64]) -> Self {
        self.expected_stats = stats.to_vec();
        self
    }

    fn with_job_builder<F>(mut self, builder: F) -> Self
    where
        F: FnOnce(Arc<Dataset>) -> MergeInsertJob + 'static,
    {
        self.job_builder = Some(Box::new(builder));
        self
    }

    async fn run_test(self) {
        let schema = create_test_schema();
        let new_batch = create_new_batch(schema.clone());
        let test_uri = "memory://test.lance";

        let ds = create_test_dataset(test_uri, self.version, self.enable_stable_row_ids).await;
        let row_ids_before = get_row_ids_for_keys(&ds, &self.test_keys).await;

        let job_builder = self.job_builder.expect("job_builder must be set");
        let job = job_builder(ds);
        let ds = check_then_refresh_dataset(
            new_batch,
            job,
            &self.expected_left_keys,
            &self.expected_right_keys,
            &self.expected_stats,
        )
        .boxed()
        .await;

        let row_ids_after = get_row_ids_for_keys(&ds, &self.test_keys).await;

        if self.enable_stable_row_ids {
            assert_eq!(row_ids_before, row_ids_after);
        } else {
            assert_ne!(row_ids_before, row_ids_after);
        }
    }
}

/// Perform a partial-schema merge_insert (only id + value_a) targeting specific id ranges.
/// This causes touched fragments to drop from the index bitmap while btree data retains
/// stale entries.
async fn partial_merge_insert(
    dataset: Arc<Dataset>,
    id_range: std::ops::Range<usize>,
    value_a_val: f64,
) -> Arc<Dataset> {
    let ids: Vec<String> = id_range.map(|j| format!("id-{j:04}")).collect();
    let n = ids.len();
    let sub_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("value_a", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        sub_schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(Float64Array::from(vec![value_a_val; n])),
        ],
    )
    .unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(batch)], sub_schema));

    let (ds, _) = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(reader)
        .await
        .unwrap();
    ds
}
