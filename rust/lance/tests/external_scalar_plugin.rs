// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exercise registration from a downstream crate, without changing the builtin registry.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow_array::{RecordBatchIterator, record_batch};
use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::{common::ScalarValue, execution::SendableRecordBatchStream};
use lance::{
    Dataset,
    dataset::WriteParams,
    index::{DatasetIndexExt, DatasetIndexInternalExt, register_scalar_index_plugin},
    io::exec::scalar_index::ScalarIndexExec,
};
use lance_core::{Error, Result, cache::LanceCache};
use lance_index::{
    IndexType,
    metrics::NoOpMetricsCollector,
    progress::IndexBuildProgress,
    scalar::{
        CreatedIndex, IndexStore, RowIdRemapper, SargableQuery, ScalarIndex, ScalarIndexParams,
        expression::{ScalarIndexExpr, ScalarIndexSearch, ScalarQueryParser},
        registry::{BasicTrainer, ScalarIndexPlugin, TrainingRequest},
        zonemap::ZoneMapIndexPlugin,
    },
};
use prost::Name;

#[derive(Clone, PartialEq, prost::Message)]
struct ExampleIndexDetails {}

impl Name for ExampleIndexDetails {
    const NAME: &'static str = "ExampleIndexDetails";
    const PACKAGE: &'static str = "example.index";
}

#[derive(Debug, Default)]
struct ExampleIndexPlugin;

static LOADS: AtomicUsize = AtomicUsize::new(0);

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

    fn version(&self) -> u32 {
        0
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
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        LOADS.fetch_add(1, Ordering::Relaxed);
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
}

// One test owns this process's registration lifecycle; it must register before
// any dataset operation initializes the process-global registry.
#[tokio::test]
async fn register_create_open_and_cache_external_index() {
    register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap();
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
    assert_eq!(LOADS.load(Ordering::Relaxed), 1);

    let query = ScalarIndexExpr::Query(ScalarIndexSearch {
        column: "value".to_string(),
        index_name: "example_index".to_string(),
        index_type: "Example".to_string(),
        query: Arc::new(SargableQuery::Equals(ScalarValue::Int64(Some(3)))),
        needs_recheck: true,
        fragment_bitmap: None,
    });
    let dataset = Arc::new(dataset);
    let unrestricted = ScalarIndexExec::new(dataset, query, Default::default());
    assert!(unrestricted.fragment_scope().is_none());
    assert!(unrestricted.segment_uuids().is_none());
    let scoped = unrestricted
        .with_fragment_scope([1].into_iter().collect())
        .with_segment_uuids(Arc::new([metadata.uuid].into_iter().collect()));
    let reconstructed = ScalarIndexExec::new(
        scoped.dataset().clone(),
        scoped.expr().clone(),
        scoped.result_format(),
    )
    .with_fragment_scope(scoped.fragment_scope().unwrap().clone())
    .with_segment_uuids(Arc::new(scoped.segment_uuids().unwrap().clone()));
    assert_eq!(reconstructed.fragment_scope(), scoped.fragment_scope());
    assert_eq!(reconstructed.segment_uuids(), scoped.segment_uuids());
    let empty = reconstructed.with_fragment_scope(Default::default());
    assert!(empty.fragment_scope().unwrap().is_empty());

    let err =
        register_scalar_index_plugin::<ExampleIndexDetails, ExampleIndexPlugin>().unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }));
    assert!(err.to_string().contains("before the first plugin registry"));
}
