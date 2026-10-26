// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A downstream, statistics-only plugin using the public Rust extension interface.

use std::{
    any::Any,
    process::Command,
    sync::{Arc, LazyLock},
};

use arrow_array::{RecordBatchIterator, record_batch};
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance::{
    Dataset,
    dataset::{
        ColumnAlteration, WriteParams,
        optimize::{CompactionOptions, compact_files},
    },
    index::{DatasetIndexExt, DatasetIndexInternalExt, register_scalar_index_plugin},
};
use lance_core::{
    Error, Result, cache::LanceCache, deepsize::DeepSizeOf, utils::row_addr_remap::RowAddrRemap,
};
use lance_index::{
    Index, IndexType,
    metrics::{MetricsCollector, NoOpMetricsCollector},
    optimize::OptimizeOptions,
    progress::IndexBuildProgress,
    scalar::{
        AnyQuery, CreatedIndex, IndexStore, OldIndexDataFilter, RowIdRemapper, ScalarIndex,
        ScalarIndexParams, SearchResult, TrainingCriteria, TrainingOrdering, UpdateCriteria,
        expression::{IndexInformationProvider, ScalarQueryParser},
        registry::{BasicTrainer, ScalarIndexPlugin, TrainingRequest},
    },
};
use prost::Name;
use roaring::RoaringBitmap;
use rstest::rstest;
use tempfile::{TempDir, tempdir};

#[derive(Clone, PartialEq, prost::Message)]
struct ExampleIndexDetails {}

impl Name for ExampleIndexDetails {
    const NAME: &'static str = "ExampleIndexDetails";
    const PACKAGE: &'static str = "example.index";
}

#[derive(Debug)]
struct ExampleIndexPlugin;

impl Default for ExampleIndexPlugin {
    fn default() -> Self {
        // Constructors run after freezing, outside the registration lock.
        // A constructor calling registration must fail rather than deadlock.
        let error = register_scalar_index_plugin::<ExampleIndexDetails, Self>().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("before the first plugin registry")
        );
        Self
    }
}

// All tests in this binary register before any dataset or registry operation.
static REGISTRATION: LazyLock<()> = LazyLock::new(|| {
    let error = register_scalar_index_plugin::<
        lance_index::pbold::ZoneMapIndexDetails,
        ExampleIndexPlugin,
    >()
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("already registered"));

    register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap();
    let error =
        register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("already registered"));
});

struct ExampleTrainingRequest(TrainingCriteria);

impl TrainingRequest for ExampleTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn criteria(&self) -> &TrainingCriteria {
        &self.0
    }
}

#[async_trait]
impl BasicTrainer for ExampleIndexPlugin {
    fn new_training_request(
        &self,
        _params: &str,
        _field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        Ok(Box::new(ExampleTrainingRequest(TrainingCriteria::new(
            TrainingOrdering::None,
        ))))
    }

    async fn train_index(
        &self,
        mut data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        _request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let mut rows = 0u64;
        while let Some(batch) = data.try_next().await? {
            rows += batch.num_rows() as u64;
        }
        let payload = record_batch!(("rows", UInt64, [rows]))?;
        let mut writer = index_store
            .new_index_file("statistics.lance", payload.schema())
            .await?;
        writer.write_record_batch(payload).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&ExampleIndexDetails {})?,
            index_version: self.version(),
            files: vec![writer.finish().await?],
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for ExampleIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }
    fn name(&self) -> &str {
        "Example"
    }
    fn provides_exact_answer(&self) -> bool {
        false
    }
    fn supports_optimization(&self) -> bool {
        false
    }
    fn version(&self) -> u32 {
        1
    }

    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        index_version: u32,
        _frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        _cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        index_details.to_msg::<ExampleIndexDetails>()?;
        assert_eq!(index_version, self.version());
        let reader = index_store.open_index_file("statistics.lance").await?;
        let batch = reader.read_record_batch(0, 1).await?;
        let rows = batch["rows"]
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .unwrap()
            .value(0);
        Ok(Arc::new(ExampleIndex { rows }))
    }
}

#[derive(Debug, DeepSizeOf)]
struct ExampleIndex {
    rows: u64,
}

#[async_trait]
impl Index for ExampleIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }
    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({ "rows": self.rows }))
    }
    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }
    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }
    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }
}

#[async_trait]
impl ScalarIndex for ExampleIndex {
    async fn search(
        &self,
        _query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        Err(Error::not_supported("statistics do not answer predicates"))
    }
    fn can_remap(&self) -> bool {
        false
    }
    async fn remap(
        &self,
        _mapping: &RowAddrRemap,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::not_supported("statistics are maintained externally"))
    }
    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
        _old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        Err(Error::not_supported("statistics are maintained externally"))
    }
    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria {
            requires_old_data: false,
            data_criteria: TrainingCriteria::new(TrainingOrdering::None),
        }
    }
    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::new("example".to_string()))
    }
}

async fn assert_external_index(dataset: &Dataset, rows: u64, coverage: RoaringBitmap) {
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let metadata = &indices[0];
    assert_eq!(
        metadata.index_details.as_ref().unwrap().type_url,
        ExampleIndexDetails::type_url()
    );
    assert_eq!(metadata.fragment_bitmap.as_ref().unwrap(), &coverage);
    assert_eq!(metadata.index_version, 1);
    assert_eq!(metadata.files.as_ref().unwrap()[0].path, "statistics.lance");
    let first = dataset
        .open_scalar_index("value", &metadata.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(first.as_any().is::<ExampleIndex>());
    assert_eq!(
        first.statistics().unwrap(),
        serde_json::json!({ "rows": rows })
    );
    let second = dataset
        .open_scalar_index("value", &metadata.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn subsequent_registries_include_external_plugins() {
    LazyLock::force(&REGISTRATION);
    for _ in 0..2 {
        let registry = lance_index::registry::IndexPluginRegistry::with_default_plugins();
        let details = prost_types::Any::from_msg(&ExampleIndexDetails {}).unwrap();
        assert!(registry.supports_details(&details));
        assert_eq!(
            registry.get_plugin_by_details(&details).unwrap().name(),
            "Example"
        );
        assert!(registry.get_plugin_by_name("EXAMPLE").is_ok());
        assert!(registry.get_plugin_by_name("btree").is_ok());
    }
}

#[rstest]
#[case::trained(4, true)]
#[case::definition_only(4, false)]
#[case::empty_dataset(0, true)]
#[tokio::test]
async fn create_and_reopen_external_index(#[case] rows: usize, #[case] train: bool) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(("value", Int64, [Some(1), None, Some(3), Some(4)]))
        .unwrap()
        .slice(0, rows);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        directory.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), rows / 2);
    dataset
        .create_index_builder(
            &["value"],
            IndexType::Scalar,
            &ScalarIndexParams::new("EXAMPLE".to_string()),
        )
        .name("example_index".to_string())
        .train(train)
        .await
        .unwrap();
    let coverage = if train {
        (0..rows as u32 / 2).collect()
    } else {
        RoaringBitmap::new()
    };
    let expected_rows = if train { rows as u64 } else { 0 };
    assert_external_index(&dataset, expected_rows, coverage.clone()).await;
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_external_index(&reopened, expected_rows, coverage).await;

    let error =
        register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("before the first plugin registry")
    );
}

#[rstest]
#[case::fragment_coverage(false)]
#[case::empty_coverage(true)]
#[tokio::test]
async fn commit_and_reopen_external_segment(#[case] empty_coverage: bool) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(("value", Int64, [1, 2, 3, 4])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        directory.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::new("example".to_string());
    let mut segment = dataset
        .create_index_builder(&["value"], IndexType::Scalar, &params)
        .name("example_index".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    assert!(dataset.load_indices().await.unwrap().is_empty());
    // An externally built statistics payload can cover rows without claiming
    // fragment coverage for filtering or row-ID remapping.
    if empty_coverage {
        segment.fragment_bitmap = Some(RoaringBitmap::new());
    }
    let coverage = segment.fragment_bitmap.clone().unwrap();
    let uuid = segment.uuid;
    dataset
        .commit_existing_index_segments("example_index", "value", vec![segment])
        .await
        .unwrap();
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(reopened.load_indices().await.unwrap()[0].uuid, uuid);
    assert_external_index(&reopened, 4, coverage).await;
}

#[tokio::test]
async fn reopen_in_another_process() {
    const DATASET_ENV: &str = "LANCE_EXTERNAL_PLUGIN_TEST_DATASET";
    LazyLock::force(&REGISTRATION);
    if let Some(uri) = std::env::var_os(DATASET_ENV) {
        let dataset = Dataset::open(uri.to_str().unwrap()).await.unwrap();
        assert_external_index(&dataset, 2, RoaringBitmap::new()).await;
        return;
    }

    let directory = tempdir().unwrap();
    let batch = record_batch!(("value", Int64, [1, 2])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        directory.path().to_str().unwrap(),
        None,
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::new("example".to_string());
    let mut segment = dataset
        .create_index_builder(&["value"], IndexType::Scalar, &params)
        .name("example_index".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    segment.fragment_bitmap = Some(RoaringBitmap::new());
    dataset
        .commit_existing_index_segments("example_index", "value", vec![segment])
        .await
        .unwrap();
    // A fresh executable has its own registration state and index cache.
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "reopen_in_another_process", "--nocapture"])
        .env(DATASET_ENV, dataset.uri())
        .output()
        .unwrap();
    assert!(output.status.success(), "child process failed: {output:?}");
}

async fn dataset_with_example_index(stable_row_ids: bool, columns: &[&str]) -> (TempDir, Dataset) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(
        ("value", Int64, [Some(1), None, Some(3), Some(4)]),
        ("other", Int64, [1, 2, 3, 4]),
        ("keep", Int64, [1, 2, 3, 4])
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        directory.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 2,
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 2);
    let mut segment = dataset
        .create_index_builder(
            &columns[..1],
            IndexType::Scalar,
            &ScalarIndexParams::new("example".into()),
        )
        .name("profile".into())
        .execute_uncommitted()
        .await
        .unwrap();
    segment.fragment_bitmap = Some(RoaringBitmap::new());
    segment.fields = columns
        .iter()
        .map(|column| dataset.schema().field(column).unwrap().id)
        .collect();
    dataset
        .commit_existing_index_segments("profile", columns[0], vec![segment])
        .await
        .unwrap();
    // Compare persisted metadata across maintenance operations: serialization
    // stores creation timestamps at millisecond precision.
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    (directory, reopened)
}

#[rstest]
#[case::default(OptimizeOptions::default())]
#[case::append(OptimizeOptions::append())]
#[case::merge(OptimizeOptions::merge(1))]
#[case::retrain(OptimizeOptions::retrain())]
#[case::bounded(OptimizeOptions::default().max_rows_per_task(Some(2)))]
#[tokio::test]
async fn externally_maintained_index_is_skipped(
    #[case] options: OptimizeOptions,
    #[values(&["value"][..], &["other", "value"][..])] columns: &[&str],
) {
    let (_directory, mut dataset) = dataset_with_example_index(false, columns).await;
    let before = dataset.load_indices().await.unwrap();
    let version = dataset.version().version;
    let plan = lance::index::optimize::plan_optimize_indices(&dataset, &options)
        .await
        .unwrap();
    assert!(plan.tasks.is_empty());
    assert!(plan.merges.is_empty());
    dataset.optimize_indices(&options).await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.load_indices().await.unwrap(), before);

    let error = dataset
        .optimize_indices(&options.index_names(vec!["profile".into()]))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("profile"));
    assert!(error.to_string().contains("maintained externally"));
}

#[tokio::test]
async fn externally_maintained_index_does_not_block_builtin_optimization() {
    let (_directory, mut dataset) = dataset_with_example_index(false, &["value"]).await;
    let profile = dataset.load_indices().await.unwrap()[0].clone();
    dataset
        .create_index(
            &["other"],
            IndexType::BTree,
            Some("lookup".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let batch = record_batch!(
        ("value", Int64, [5, 6]),
        ("other", Int64, [5, 6]),
        ("keep", Int64, [5, 6])
    )
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            None,
        )
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 2);
    assert!(indices.iter().any(|index| index == &profile));
    let lookup = indices.iter().find(|index| index.name == "lookup").unwrap();
    assert_eq!(lookup.fragment_bitmap.as_ref().unwrap().len(), 3);
    assert_eq!(
        dataset
            .scan()
            .filter("other >= 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        2
    );
}

#[rstest]
#[case::row_addresses(false)]
#[case::stable_row_ids(true)]
#[tokio::test]
async fn empty_coverage_survives_compaction(
    #[case] stable_row_ids: bool,
    #[values(&["value"][..], &["other", "value"][..])] columns: &[&str],
) {
    let (_directory, mut dataset) = dataset_with_example_index(stable_row_ids, columns).await;
    let before = dataset.load_indices().await.unwrap();
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_eq!(dataset.fragments().len(), 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
    assert_eq!(dataset.load_indices().await.unwrap(), before);
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    let index = reopened
        .open_scalar_index(columns[0], &before[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(!index.can_remap());
    assert_eq!(index.statistics().unwrap()["rows"], 4);
}

#[tokio::test]
async fn registered_plugin_type_is_visible() {
    let (_directory, dataset) = dataset_with_example_index(false, &["value"]).await;
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    assert_eq!(descriptions[0].name(), "profile");
    assert_eq!(descriptions[0].index_type(), "Example");
    let statistics: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("profile").await.unwrap()).unwrap();
    assert_eq!(statistics["index_type"], "Example");
    assert_eq!(statistics["indices"][0]["rows"], 4);
}

#[tokio::test]
async fn unregistered_plugin_does_not_block_maintenance() {
    const DATASET_ENV: &str = "LANCE_UNREGISTERED_PLUGIN_TEST_DATASET";
    if let Some(uri) = std::env::var_os(DATASET_ENV) {
        let mut dataset = Dataset::open(uri.to_str().unwrap()).await.unwrap();
        assert!(dataset.load_indices().await.unwrap().is_empty());
        assert!(dataset.describe_indices(None).await.unwrap().is_empty());
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        assert_eq!(dataset.fragments().len(), 1);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
        return;
    }

    for stable_row_ids in [false, true] {
        let (_directory, dataset) = dataset_with_example_index(stable_row_ids, &["value"]).await;
        let before = Dataset::open(dataset.uri())
            .await
            .unwrap()
            .load_indices()
            .await
            .unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "unregistered_plugin_does_not_block_maintenance",
                "--nocapture",
            ])
            .env(DATASET_ENV, dataset.uri())
            .output()
            .unwrap();
        assert!(output.status.success(), "child process failed: {output:?}");
        let reopened = Dataset::open(dataset.uri()).await.unwrap();
        assert_eq!(reopened.load_indices().await.unwrap(), before);
        assert_external_index(&reopened, 4, RoaringBitmap::new()).await;
    }
}

#[tokio::test]
async fn schema_changes_and_drop_index_respect_plugin_fields() {
    let (_directory, mut dataset) = dataset_with_example_index(false, &["value"]).await;
    let before = dataset.load_indices().await.unwrap();
    let version = dataset.version().version;
    let error = dataset
        .alter_columns(&[ColumnAlteration::new("value".into()).cast_to(DataType::Int32)])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("profile"));
    assert!(error.to_string().contains("drop_index()"));
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.load_indices().await.unwrap(), before);

    dataset.drop_index("profile").await.unwrap();
    assert!(dataset.load_indices().await.unwrap().is_empty());
    dataset
        .alter_columns(&[ColumnAlteration::new("value".into()).cast_to(DataType::Int32)])
        .await
        .unwrap();
    assert_eq!(
        dataset.schema().field("value").unwrap().data_type(),
        DataType::Int32
    );
}

#[tokio::test]
async fn dropping_plugin_field_removes_index() {
    let (_directory, mut dataset) = dataset_with_example_index(false, &["value"]).await;
    dataset.drop_columns(&["other"]).await.unwrap();
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
    dataset.drop_columns(&["value"]).await.unwrap();
    assert!(dataset.load_indices().await.unwrap().is_empty());
    assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
}

#[tokio::test]
async fn multi_field_scope_preserves_order_on_reopen_rename_and_replace() {
    // Deliberately reverse schema order: field order belongs to the plugin.
    let (_directory, mut dataset) = dataset_with_example_index(false, &["other", "value"]).await;
    let before = dataset.load_indices().await.unwrap();
    let fields = vec![
        dataset.schema().field("other").unwrap().id,
        dataset.schema().field("value").unwrap().id,
    ];
    assert_eq!(before[0].fields, fields);
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(
        descriptions[0].field_ids(),
        fields.iter().map(|id| *id as u32).collect::<Vec<_>>()
    );
    assert_eq!(descriptions[0].index_type(), "Example");
    let statistics: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("profile").await.unwrap()).unwrap();
    assert_eq!(statistics["indices"][0]["rows"], 4);
    assert_eq!(statistics["num_indexed_rows"], 0);
    let info = dataset.scalar_index_info().await.unwrap();
    assert!(info.get_index("other").is_none());
    assert!(info.get_index("value").is_none());

    dataset
        .alter_columns(&[
            ColumnAlteration::new("other".into()).rename("group".into()),
            ColumnAlteration::new("value".into()).rename("input".into()),
        ])
        .await
        .unwrap();
    let mut reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(reopened.load_indices().await.unwrap(), before);
    let index = reopened
        .open_scalar_index("group", &before[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(index.statistics().unwrap()["rows"], 4);
    reopened
        .commit_existing_index_segments("profile", "group", vec![before[0].clone()])
        .await
        .unwrap();
    assert_eq!(reopened.load_indices().await.unwrap()[0].fields, fields);

    let mut reversed = before[0].clone();
    reversed.fields.reverse();
    let version = reopened.version().version;
    let error = reopened
        .commit_existing_index_segments("profile", "input", vec![reversed])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Index { .. }));
    assert!(error.to_string().contains("different fields"));
    assert_eq!(reopened.version().version, version);
}

#[rstest]
#[case::first("other")]
#[case::second("value")]
#[tokio::test]
async fn multi_field_scope_protects_every_dependency(#[case] column: &str) {
    let (_directory, mut dataset) = dataset_with_example_index(false, &["other", "value"]).await;
    let before = dataset.load_indices().await.unwrap();
    let version = dataset.version().version;
    let error = dataset
        .alter_columns(&[ColumnAlteration::new(column.into()).cast_to(DataType::Int32)])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("profile"));
    assert!(error.to_string().contains("drop_index()"));
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.load_indices().await.unwrap(), before);

    dataset.drop_columns(&["keep"]).await.unwrap();
    assert_eq!(dataset.load_indices().await.unwrap(), before);
    dataset.drop_columns(&[column]).await.unwrap();
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert!(reopened.load_indices().await.unwrap().is_empty());
    assert_eq!(reopened.count_rows(None).await.unwrap(), 4);
}

#[rstest]
#[case::missing_field(vec![0, 99], false, "absent from dataset")]
#[case::duplicate_field(vec![0, 0], false, "repeats field id")]
#[case::wrong_first_field(vec![1, 0], false, "expected keyed field")]
#[case::filtering_coverage(vec![0, 1], true, "expected keyed field")]
#[tokio::test]
async fn multi_field_scope_rejects_invalid_declarations(
    #[case] fields: Vec<i32>,
    #[case] has_coverage: bool,
    #[case] message: &str,
) {
    let (_directory, mut dataset) = dataset_with_example_index(false, &["value"]).await;
    let before = dataset.load_indices().await.unwrap();
    let mut segment = before[0].clone();
    segment.fields = fields;
    if has_coverage {
        segment.fragment_bitmap = Some([0].into_iter().collect());
    }
    let version = dataset.version().version;
    let error = dataset
        .commit_existing_index_segments("multi", "value", vec![segment])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.load_indices().await.unwrap(), before);
}

#[tokio::test]
async fn multi_field_scope_rejects_mixed_segment_order() {
    let (_directory, mut dataset) =
        dataset_with_example_index(false, &["value", "other", "keep"]).await;
    let before = dataset.load_indices().await.unwrap();
    let mut second = before[0].clone();
    second.uuid = uuid::Uuid::new_v4();
    second.fields.swap(1, 2);
    let error = dataset
        .commit_existing_index_segments("multi", "value", vec![before[0].clone(), second.clone()])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("must declare the same columns"));
    assert_eq!(dataset.load_indices().await.unwrap(), before);

    // Keeping the first field does not permit changing the rest of the scope.
    second.uuid = before[0].uuid;
    let error = dataset
        .commit_existing_index_segments("profile", "value", vec![second])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Index { .. }));
    assert!(error.to_string().contains("different fields"));
    assert_eq!(dataset.load_indices().await.unwrap(), before);
}

#[rstest]
#[case::drop(false)]
#[case::cast(true)]
#[tokio::test]
async fn staged_multi_field_scope_rejects_removed_dependency(#[case] cast: bool) {
    let (_directory, mut dataset) = dataset_with_example_index(false, &["value", "other"]).await;
    let segments = dataset.load_indices().await.unwrap().as_ref().clone();
    dataset.drop_index("profile").await.unwrap();
    if cast {
        dataset
            .alter_columns(&[ColumnAlteration::new("other".into()).cast_to(DataType::Int32)])
            .await
            .unwrap();
    } else {
        dataset.drop_columns(&["other"]).await.unwrap();
    }
    let version = dataset.version().version;
    let error = dataset
        .commit_existing_index_segments("profile", "value", segments)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("absent from dataset"), "{error}");
    assert_eq!(dataset.version().version, version);
    assert!(dataset.load_indices().await.unwrap().is_empty());
}

#[tokio::test]
async fn builtin_segments_and_builders_still_require_one_keyed_field() {
    let (_directory, mut dataset) = dataset_with_example_index(false, &["value"]).await;
    for params in [
        ScalarIndexParams::default(),
        ScalarIndexParams::new("example".into()),
    ] {
        let error = dataset
            .create_index_builder(&["value", "other"], IndexType::Scalar, &params)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Index { .. }));
        assert!(error.to_string().contains("1 column"), "{error}");
    }
    let mut segment = dataset
        .create_index_builder(&["value"], IndexType::BTree, &ScalarIndexParams::default())
        .execute_uncommitted()
        .await
        .unwrap();
    segment
        .fields
        .push(dataset.schema().field("other").unwrap().id);
    segment.fragment_bitmap = Some(RoaringBitmap::new());
    let error = dataset
        .commit_existing_index_segments("builtin", "value", vec![segment])
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("expected keyed field"));
}
