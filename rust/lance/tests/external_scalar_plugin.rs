// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exercise registration from a downstream crate, without changing the builtin registry.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, record_batch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use lance::{
    Dataset,
    dataset::{
        UpdateBuilder, WriteParams,
        index::frag_reuse::cleanup_frag_reuse_index,
        optimize::{CompactionOptions, compact_files, remapping::remap_column_index},
    },
    index::{DatasetIndexExt, DatasetIndexInternalExt, register_scalar_index_plugin},
};
use lance_arrow::ARROW_EXT_NAME_KEY;
use lance_arrow::json::ARROW_JSON_EXT_NAME;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result, cache::LanceCache};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_index::{
    Index, IndexType,
    metrics::{MetricsCollector, NoOpMetricsCollector},
    progress::IndexBuildProgress,
    scalar::{
        AnyQuery, CreatedIndex, IndexStore, OldIndexDataFilter, Remapper,
        ScalarIndex, ScalarIndexParams, SearchResult, UpdateCriteria,
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
use lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
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
    register_scalar_index_plugin::<StaleIndexDetails, StaleIndexPlugin>().unwrap();
    register_scalar_index_plugin::<LegacyRemapIndexDetails, LegacyRemapPlugin>().unwrap();
});

#[derive(Clone, PartialEq, prost::Message)]
struct StaleIndexDetails {}

impl Name for StaleIndexDetails {
    const NAME: &'static str = "StaleIndexDetails";
    const PACKAGE: &'static str = "example.index";
}

/// Like [`ExampleIndexPlugin`], but does not translate row addresses: it
/// keeps answering with the row addresses it was built with.
#[derive(Debug, Default)]
struct StaleIndexPlugin;

#[async_trait]
impl BasicTrainer for StaleIndexPlugin {
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
        created.index_details = prost_types::Any::from_msg(&StaleIndexDetails {})?;
        Ok(created)
    }
}

#[async_trait]
impl ScalarIndexPlugin for StaleIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "Stale"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

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
        _remapper: Option<Remapper>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        ZoneMapIndexPlugin
            .load_index(index_store, index_details, index_version, None, cache)
            .await
    }
}

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
    // Deliberately does not implement `can_remap`: this plugin predates the
    // metadata capability and must keep compiling through its default.
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
        remapper: Option<Remapper>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        ZoneMapIndexPlugin
            .load_index(index_store, index_details, index_version, remapper, cache)
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

/// The example plugin predates `ScalarIndexPlugin::can_remap`, so the metadata
/// capability defaults to `true` and compaction keeps the existing
/// load-and-check path. The loaded index still reports
/// `ScalarIndex::can_remap == false`; that must keep it on the safe fallback
/// path instead of attempting a remap it does not support.
#[tokio::test]
async fn legacy_external_plugin_can_remap_false_stays_on_fallback() {
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

    let before = dataset.load_indices().await.unwrap();
    let opened = dataset
        .open_scalar_index("value", &before[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    #[allow(deprecated)]
    let can_remap = opened.can_remap();
    assert!(!can_remap);

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
    let after = dataset.load_indices().await.unwrap();
    assert_eq!(after.len(), before.len());
    // `created_at` is re-encoded at a different subsecond precision on
    // Windows, so compare the stable fields that identify an untouched index.
    assert_eq!(after[0].uuid, before[0].uuid);
    assert_eq!(after[0].fragment_bitmap, before[0].fragment_bitmap);
    assert_eq!(after[0].index_details, before[0].index_details);
    assert_eq!(after[0].index_version, before[0].index_version);

    let mut full_scan = dataset.scan();
    full_scan
        .filter("value = 3")
        .unwrap()
        .use_scalar_index(false);
    let expected = full_scan.try_into_batch().await.unwrap();
    assert_eq!(expected.num_rows(), 1);
    let mut indexed_scan = dataset.scan();
    indexed_scan.filter("value = 3").unwrap();
    let actual = indexed_scan.try_into_batch().await.unwrap();
    assert_eq!(actual, expected);
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

/// Under either fragment reuse history format, a segment of a plugin that
/// does not translate row addresses leaves the query listing once its rows move, so they
/// are scanned instead of answered from stale addresses. So does a JSON
/// index over such a plugin, which answers through it. A built-in index
/// keeps serving its segment through the history.
#[rstest]
#[case::v0(false)]
#[case::tagged(true)]
#[tokio::test]
async fn fragment_reuse_scans_rows_of_plugins_that_cannot_translate(#[case] tagged: bool) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(
        ("value", Int64, [1, 2, 3, 4]),
        ("other", Int64, [10, 20, 30, 40])
    )
    .unwrap();
    let json_field = Field::new("doc", DataType::Utf8, false).with_metadata(HashMap::from([(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    )]));
    let docs = StringArray::from_iter_values((1..=4).map(|val| format!(r#"{{"val": {val}}}"#)));
    let mut fields = batch.schema().fields().to_vec();
    fields.push(Arc::new(json_field));
    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(docs));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
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
    let json_over_stale = ScalarIndexParams::new("json".to_string())
        .with_params(&serde_json::json!({"target_index_type": "stale", "path": "val"}));
    for (column, name, params) in [
        (
            "value",
            "btree_index",
            ScalarIndexParams::new("btree".to_string()),
        ),
        (
            "other",
            "stale_index",
            ScalarIndexParams::new("stale".to_string()),
        ),
        ("doc", "json_stale_index", json_over_stale),
    ] {
        dataset
            .create_index(
                &[column],
                IndexType::Scalar,
                Some(name.to_string()),
                &params,
                false,
            )
            .await
            .unwrap();
    }
    dataset.delete("value = 1").await.unwrap();
    if tagged {
        // The sticky flag makes the compaction record a tagged history.
        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        manifest.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
    }
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 4,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 1);
    let fri = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .find(|index| index.name == "__lance_frag_reuse")
        .map(|index| index.index_version);
    assert_eq!(fri, Some(i32::from(tagged)));

    let listed: Vec<String> = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .map(|index| index.name.clone())
        .collect();
    assert!(listed.contains(&"btree_index".to_string()), "{listed:?}");
    assert!(!listed.contains(&"stale_index".to_string()), "{listed:?}");
    assert!(
        !listed.contains(&"json_stale_index".to_string()),
        "{listed:?}"
    );

    for filter in ["value = 3", "other = 30", "json_get_int(doc, 'val') = 3"] {
        let mut full_scan = dataset.scan();
        full_scan.filter(filter).unwrap().use_scalar_index(false);
        let expected = full_scan.try_into_batch().await.unwrap();
        assert_eq!(expected.num_rows(), 1, "{filter}");
        let mut indexed_scan = dataset.scan();
        indexed_scan.filter(filter).unwrap();
        assert_eq!(
            indexed_scan.try_into_batch().await.unwrap(),
            expected,
            "{filter}"
        );
    }
}

/// Under either fragment reuse history format, a segment of a plugin that
/// does not translate row addresses stays listed, and answers queries, when the history
/// moves none of its rows: it covers only fragments no rewrite touched, or it
/// was built after the rewrite.
#[rstest]
#[case::v0(false)]
#[case::tagged(true)]
#[tokio::test]
async fn fragment_reuse_keeps_listing_plugins_that_need_no_translation(#[case] tagged: bool) {
    LazyLock::force(&REGISTRATION);
    let directory = tempdir().unwrap();
    let batch = record_batch!(
        ("value", Int64, [1, 2, 3, 4]),
        ("other", Int64, [10, 20, 30, 40]),
        ("third", Int64, [100, 200, 300, 400])
    )
    .unwrap();
    let schema = batch.schema();
    let write_params = WriteParams {
        max_rows_per_file: 2,
        ..Default::default()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema.clone()),
        directory.path().to_str().unwrap(),
        Some(write_params.clone()),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["other"],
            IndexType::Scalar,
            Some("untouched_index".to_string()),
            &ScalarIndexParams::new("stale".to_string()),
            false,
        )
        .await
        .unwrap();
    let appended = record_batch!(
        ("value", Int64, [5, 6]),
        ("other", Int64, [50, 60]),
        ("third", Int64, [500, 600])
    )
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(appended)], schema),
            Some(write_params),
        )
        .await
        .unwrap();
    // A built-in index over the appended fragment makes the compaction
    // record a history.
    dataset
        .create_index(
            &["third"],
            IndexType::Scalar,
            Some("btree_index".to_string()),
            &ScalarIndexParams::new("btree".to_string()),
            false,
        )
        .await
        .unwrap();
    dataset.delete("value = 5").await.unwrap();
    if tagged {
        // The sticky flag makes the compaction record a tagged history.
        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
        manifest.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
    }
    let untouched: Vec<u64> = dataset.fragments()[..2].iter().map(|f| f.id).collect();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 2,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    // Only the appended fragment, with its deletion, was rewritten.
    let fragments: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
    assert_eq!(fragments.len(), 3, "{fragments:?}");
    assert_eq!(fragments[..2], untouched);
    dataset
        .create_index(
            &["value"],
            IndexType::Scalar,
            Some("built_after_index".to_string()),
            &ScalarIndexParams::new("stale".to_string()),
            false,
        )
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    let fri = indices
        .iter()
        .find(|index| index.name == "__lance_frag_reuse")
        .map(|index| index.index_version);
    assert_eq!(fri, Some(i32::from(tagged)));
    for (column, name) in [("other", "untouched_index"), ("value", "built_after_index")] {
        let segment = indices
            .iter()
            .find(|index| index.name == name)
            .unwrap_or_else(|| panic!("{name} is listed: {indices:?}"));
        dataset
            .open_scalar_index(column, &segment.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
    }

    for filter in ["other = 30", "value = 6"] {
        let mut full_scan = dataset.scan();
        full_scan.filter(filter).unwrap().use_scalar_index(false);
        let expected = full_scan.try_into_batch().await.unwrap();
        assert_eq!(expected.num_rows(), 1, "{filter}");
        let mut indexed_scan = dataset.scan();
        indexed_scan.filter(filter).unwrap();
        let plan = indexed_scan.explain_plan(false).await.unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{filter}: {plan}");
        assert_eq!(
            indexed_scan.try_into_batch().await.unwrap(),
            expected,
            "{filter}"
        );
    }
}

#[derive(Clone, PartialEq, prost::Message)]
struct LegacyRemapIndexDetails {}

impl Name for LegacyRemapIndexDetails {
    const NAME: &'static str = "LegacyRemapIndexDetails";
    const PACKAGE: &'static str = "example.index";
}

/// A plugin that translates through the remapper it is loaded with, but
/// whose index is rewritten only through the legacy in-memory `remap` and
/// does not report the fragments its files hold, so a rewrite through a
/// tagged history cannot be proven complete.
#[derive(Debug, Default)]
struct LegacyRemapPlugin;

/// How often the default `remap_streaming` asked a [`LegacyRemapIndex`]
/// for its stored fragments, i.e. attempted the in-memory fallback.
static LEGACY_REMAP_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// A zone map that keeps the default `remap_streaming` and reports no
/// stored fragments.
#[derive(Debug)]
struct LegacyRemapIndex(Arc<dyn ScalarIndex>);

impl DeepSizeOf for LegacyRemapIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

#[async_trait]
impl Index for LegacyRemapIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }
    fn statistics(&self) -> Result<serde_json::Value> {
        self.0.statistics()
    }
    async fn prewarm(&self) -> Result<()> {
        self.0.prewarm().await
    }
    fn index_type(&self) -> IndexType {
        self.0.index_type()
    }
    async fn calculate_included_frags(&self) -> Result<roaring::RoaringBitmap> {
        self.0.calculate_included_frags().await
    }
}

#[async_trait]
impl ScalarIndex for LegacyRemapIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        self.0.search(query, metrics).await
    }
    fn results_are_row_addresses(&self) -> bool {
        self.0.results_are_row_addresses()
    }
    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        self.0.remap(mapping, dest_store).await
    }
    fn stored_fragments(&self) -> Option<roaring::RoaringBitmap> {
        LEGACY_REMAP_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        self.0.update(new_data, dest_store, old_data_filter).await
    }
    fn update_criteria(&self) -> UpdateCriteria {
        self.0.update_criteria()
    }
    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::new("legacyremap".to_string()))
    }
}

#[async_trait]
impl BasicTrainer for LegacyRemapPlugin {
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
        created.index_details = prost_types::Any::from_msg(&LegacyRemapIndexDetails {})?;
        Ok(created)
    }
}

#[async_trait]
impl ScalarIndexPlugin for LegacyRemapPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn translates_row_addrs(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "LegacyRemap"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

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
        remapper: Option<Remapper>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let zone_map = ZoneMapIndexPlugin
            .load_index(index_store, index_details, index_version, remapper, cache)
            .await?;
        Ok(Arc::new(LegacyRemapIndex(zone_map)))
    }
}

/// A remap the index declines (it cannot be rewritten through a complete
/// map this build can prove) publishes no manifest version: the segment
/// keeps translating through the tagged history, which a later trim keeps.
#[tokio::test]
async fn declined_remap_commits_nothing_and_keeps_the_transition() {
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
    dataset
        .create_index(
            &["value"],
            IndexType::Scalar,
            Some("legacy_remap_index".to_string()),
            &ScalarIndexParams::new("legacyremap".to_string()),
            false,
        )
        .await
        .unwrap();
    // The sticky flag makes the compaction record a tagged history.
    let manifest = Arc::make_mut(&mut dataset.manifest);
    manifest.reader_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
    manifest.writer_feature_flags |= FLAG_FRAGMENT_REUSE_INDEX;
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 4,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let segment = |dataset: &Dataset| {
        let dataset = dataset.clone();
        async move {
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .find(|index| index.name == "legacy_remap_index")
                .cloned()
                .unwrap()
        }
    };
    let before = segment(&dataset).await;
    let version = dataset.version().version;

    remap_column_index(&mut dataset, &["value"], Some("legacy_remap_index".into()))
        .await
        .unwrap();
    assert_eq!(
        LEGACY_REMAP_ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "the remap reached the in-memory fallback"
    );
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_eq!(segment(&dataset).await.uuid, before.uuid);

    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    let history = dataset.frag_reuse_history().await.unwrap().unwrap();
    assert!(!history.is_empty());

    let mut full_scan = dataset.scan();
    full_scan
        .filter("value = 3")
        .unwrap()
        .use_scalar_index(false);
    let expected = full_scan.try_into_batch().await.unwrap();
    assert_eq!(expected.num_rows(), 1);
    let mut indexed_scan = dataset.scan();
    indexed_scan.filter("value = 3").unwrap();
    assert_eq!(indexed_scan.try_into_batch().await.unwrap(), expected);
}
