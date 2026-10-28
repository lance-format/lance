// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exercise registration from a downstream crate, without changing the builtin registry.

use std::sync::{Arc, LazyLock};

use arrow_array::{RecordBatchIterator, record_batch};
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use lance::{
    Dataset,
    dataset::{
        UpdateBuilder, WriteParams,
        optimize::{CompactionOptions, compact_files},
    },
    index::{DatasetIndexExt, DatasetIndexInternalExt, register_scalar_index_plugin},
};
use lance_core::{Error, Result, cache::LanceCache};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_index::{
    IndexType,
    metrics::NoOpMetricsCollector,
    progress::IndexBuildProgress,
    scalar::{
        CreatedIndex, IndexStore, RowIdRemapper, ScalarIndex, ScalarIndexParams,
        expression::{SargableQueryParser, ScalarQueryParser},
        registry::{BasicTrainer, ScalarIndexPlugin, TrainingRequest},
        seed::{IndexSeedWriter, SEED_META_KEY_PREFIX},
        zonemap::{ZoneMapIndexPlugin, ZoneMapSeedWriter},
    },
};
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use prost::Name;
use rstest::rstest;
use tempfile::tempdir;

#[derive(Clone, PartialEq, prost::Message)]
struct ExampleIndexDetails {}

impl Name for ExampleIndexDetails {
    const NAME: &'static str = "ExampleIndexDetails";
    const PACKAGE: &'static str = "example.index";
}

#[derive(Debug, Default)]
struct ExampleIndexPlugin;

const ROWS_PER_ZONE: u64 = 2;

// Every test registers before its first dataset operation, even when tests run concurrently.
static REGISTRATION: LazyLock<()> = LazyLock::new(|| {
    register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap();
});

#[async_trait]
impl BasicTrainer for ExampleIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        ZoneMapIndexPlugin.new_training_request(params, field)
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let mut created = ZoneMapIndexPlugin
            .train_index(data, index_store, request, fragment_ids, progress)
            .await?;
        created.index_details = prost_types::Any::from_msg(&ExampleIndexDetails {})?;
        Ok(created)
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

    /// The example delegates its storage to ZoneMap, so it supports exactly
    /// the file versions that plugin writes and reads.
    fn version(&self) -> u32 {
        ZoneMapIndexPlugin.version()
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(SargableQueryParser::new(
            index_name,
            self.name().to_string(),
            true,
        )))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        index_version: u32,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        ZoneMapIndexPlugin
            .load_index(
                index_store,
                index_details,
                index_version,
                frag_reuse_index,
                cache,
            )
            .await
    }

    async fn create_seed_writer(
        &self,
        field_path: &str,
        data_type: &DataType,
        index_details: &prost_types::Any,
    ) -> Result<Option<Box<dyn IndexSeedWriter>>> {
        index_details.to_msg::<ExampleIndexDetails>()?;
        Ok(Some(Box::new(ZoneMapSeedWriter::new(
            field_path,
            ROWS_PER_ZONE,
            data_type.clone(),
        )?)))
    }
}

#[rstest]
#[case::compaction(true)]
#[case::row_rewrite(false)]
#[tokio::test]
async fn external_address_index_keeps_rewritten_rows_visible(#[case] compact: bool) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(
        ("value", Int64, [1, 2, 3, 4]),
        ("other", Int64, [10, 20, 30, 40])
    )
    .unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        directory.path().to_str().unwrap(),
        Some(WriteParams {
            max_rows_per_file: 2,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["value"],
            IndexType::Scalar,
            Some("example_index".to_string()),
            &ScalarIndexParams::new("example".to_string()),
            false,
        )
        .await
        .unwrap();
    let mut before = dataset.scan();
    before.filter("value = 3").unwrap();
    assert!(
        before
            .explain_plan(false)
            .await
            .unwrap()
            .contains("ScalarIndex")
    );
    assert_eq!(before.try_into_batch().await.unwrap().num_rows(), 1);

    if compact {
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 4,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 1);
    } else {
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("value = 3")
            .unwrap()
            .set("other", "99")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        assert_eq!(result.rows_updated, 1);
        dataset = result.new_dataset.as_ref().clone();
    }

    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    let indices = reopened.load_indices().await.unwrap();
    let coverage = indices[0].fragment_bitmap.as_ref().unwrap();
    assert!(coverage.iter().all(|fragment| fragment < 2));
    if compact {
        assert!(coverage.is_empty());
    } else {
        assert_eq!(coverage.iter().collect::<Vec<_>>(), [0, 1]);
    }
    let mut full_scan = reopened.scan();
    full_scan
        .filter("value = 3")
        .unwrap()
        .use_scalar_index(false);
    let expected = full_scan.try_into_batch().await.unwrap();
    assert_eq!(expected.num_rows(), 1);
    let mut indexed_scan = reopened.scan();
    indexed_scan.filter("value = 3").unwrap();
    let actual = indexed_scan.try_into_batch().await.unwrap();
    assert_eq!(actual, expected, "rewritten rows must remain visible");
}

#[tokio::test]
async fn register_create_open_and_cache_external_index() {
    LazyLock::force(&REGISTRATION);
    let batch = record_batch!(("value", Int64, [1, 2, 3, 4])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 2);
    dataset
        .create_index(
            &["value"],
            IndexType::Scalar,
            Some("example_index".to_string()),
            &ScalarIndexParams::new("example".to_string()),
            true,
        )
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    let metadata = &indices[0];
    assert_eq!(
        metadata.index_details.as_ref().unwrap().type_url,
        ExampleIndexDetails::type_url()
    );
    assert_eq!(metadata.fragment_bitmap.as_ref().unwrap().len(), 2);
    let first = dataset
        .open_scalar_index("value", &metadata.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let second = dataset
        .open_scalar_index("value", &metadata.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));

    let err =
        register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(err.to_string().contains("before the first plugin registry"));
}

#[rstest]
#[case::empty(0)]
#[case::populated(4)]
#[tokio::test]
async fn register_only_external_index_can_be_reopened(#[case] rows: usize) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(("value", Int64, [1, 2, 3, 4]))
        .unwrap()
        .slice(0, rows);
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        directory.path().to_str().unwrap(),
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index_builder(
            &["value"],
            IndexType::Scalar,
            &ScalarIndexParams::new("example".to_string()),
        )
        .name("example_index".to_string())
        .train(false)
        .await
        .unwrap();

    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    let indices = reopened.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    let metadata = &indices[0];
    assert_eq!(
        metadata.index_details.as_ref().unwrap().type_url,
        ExampleIndexDetails::type_url()
    );
    assert!(metadata.fragment_bitmap.as_ref().unwrap().is_empty());
    reopened
        .open_scalar_index("value", &metadata.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
}

#[tokio::test]
async fn append_collects_seeds_from_external_plugin() {
    LazyLock::force(&REGISTRATION);
    let batch = record_batch!(("value", Int64, [Some(3), None, Some(9)])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.slice(0, 0))], schema.clone()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index_builder(
            &["value"],
            IndexType::Scalar,
            &ScalarIndexParams::new("example".to_string()),
        )
        .name("example_index".to_string())
        .train(false)
        .await
        .unwrap();
    dataset
        .append(RecordBatchIterator::new([Ok(batch)], schema), None)
        .await
        .unwrap();

    assert_eq!(dataset.fragments().len(), 1);
    let fragment = &dataset.fragments()[0];
    let file = &fragment.files[0];
    let path = dataset.data_dir().join(file.path.as_str());
    let object_store = dataset.object_store(file.base_id).await.unwrap();
    let scheduler = ScanScheduler::new(
        object_store.clone(),
        SchedulerConfig::max_bandwidth(&object_store),
    );
    let file_scheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await
        .unwrap();
    let reader = FileReader::try_open(
        file_scheduler,
        None,
        Default::default(),
        dataset.session().file_metadata_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();
    let location = &reader.metadata().file_schema.metadata[&format!("{SEED_META_KEY_PREFIX}value")];
    let (buffer_index, rows_per_zone) = location.split_once(':').unwrap();
    assert_eq!(rows_per_zone.parse::<u64>().unwrap(), ROWS_PER_ZONE);
    let bytes = reader
        .read_global_buffer(buffer_index.parse().unwrap())
        .await
        .unwrap();
    assert!(!bytes.is_empty());
    let indices = dataset.load_indices().await.unwrap();
    assert!(indices[0].fragment_bitmap.as_ref().unwrap().is_empty());
}

#[tokio::test]
async fn commit_external_segments_replaces_only_overlapping_coverage() {
    LazyLock::force(&REGISTRATION);
    let batch = record_batch!(("value", Int64, [1, 2, 3, 4])).unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::new("example".to_string());
    let placeholder = dataset
        .create_index_builder(&["value"], IndexType::Scalar, &params)
        .name("example_index".to_string())
        .train(false)
        .await
        .unwrap();
    let first_id = dataset.fragments()[0].id as u32;
    let second_id = dataset.fragments()[1].id as u32;

    let first = dataset
        .create_index_builder(&["value"], IndexType::Scalar, &params)
        .name("example_index".to_string())
        .fragments(vec![first_id])
        .replace(true)
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("example_index", "value", vec![first.clone()])
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].uuid, first.uuid);
    assert_ne!(indices[0].uuid, placeholder.uuid);

    let second = dataset
        .create_index_builder(&["value"], IndexType::Scalar, &params)
        .name("example_index".to_string())
        .fragments(vec![second_id])
        .replace(true)
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("example_index", "value", vec![second.clone()])
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 2);
    assert!(indices.iter().any(|index| index.uuid == first.uuid));
    assert!(indices.iter().any(|index| index.uuid == second.uuid));

    let replacement = dataset
        .create_index_builder(&["value"], IndexType::Scalar, &params)
        .name("example_index".to_string())
        .fragments(vec![first_id])
        .replace(true)
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("example_index", "value", vec![replacement.clone()])
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 2);
    assert!(!indices.iter().any(|index| index.uuid == first.uuid));
    assert!(indices.iter().any(|index| index.uuid == second.uuid));
    let replaced = indices
        .iter()
        .find(|index| index.uuid == replacement.uuid)
        .unwrap();
    assert_eq!(replaced.fragment_bitmap, replacement.fragment_bitmap);
    assert_eq!(replaced.dataset_version, replacement.dataset_version);
    assert_eq!(
        replaced.index_details.as_ref().unwrap().type_url,
        ExampleIndexDetails::type_url()
    );
    dataset
        .open_scalar_index("value", &replaced.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
}
