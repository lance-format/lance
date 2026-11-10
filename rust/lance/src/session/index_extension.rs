// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::sync::Arc;

use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;
use lance_file::versions::v1::reader::FileReader as V1FileReader;
use lance_index::{IndexParams, IndexType, vector::VectorIndex};
use uuid::Uuid;

use crate::Dataset;

pub trait IndexExtension: Send + Sync + DeepSizeOf {
    fn index_type(&self) -> IndexType;

    // TODO: this shouldn't exist, as upcasting should be well defined
    // fix after https://github.com/rust-lang/rust/issues/65991
    fn to_generic(self: Arc<Self>) -> Arc<dyn IndexExtension>;

    fn to_scalar(self: Arc<Self>) -> Option<Arc<dyn ScalarIndexExtension>>;

    fn to_vector(self: Arc<Self>) -> Option<Arc<dyn VectorIndexExtension>>;
}

pub trait ScalarIndexExtension: IndexExtension {
    // TODO: implement this trait and wire it in
}

#[async_trait::async_trait]
pub trait VectorIndexExtension: IndexExtension {
    async fn create_index(
        &self,
        // Can't use Arc<Dataset> here
        // because we need &mut Dataset to call `create_index`
        // if we wrap into an Arc, the mutable reference is lost
        dataset: &Dataset,
        column: &str,
        uuid: &Uuid,
        params: &dyn IndexParams,
    ) -> Result<()>;

    /// Load a vector index from a file.
    async fn load_index(
        &self,
        dataset: Arc<Dataset>,
        column: &str,
        uuid: &Uuid,
        reader: V1FileReader,
    ) -> Result<Arc<dyn VectorIndex>>;
}

#[cfg(test)]
mod test {
    use crate::{
        dataset::{builder::DatasetBuilder, scanner::test_dataset::TestVectorDataset},
        index::{PreFilter, loading::DatasetIndexInternalExt},
        session::Session,
    };

    use super::*;

    use std::{
        any::Any,
        sync::{Arc, atomic::AtomicBool},
    };

    use crate::index::DatasetIndexExt;
    use arrow_array::{Float32Array, RecordBatch, UInt32Array};
    use arrow_schema::Schema;
    use datafusion::execution::SendableRecordBatchStream;
    use lance_core::deepsize::DeepSizeOf;
    use lance_file::version::LanceFileVersion;
    use lance_file::versions::v1::writer::{
        FileWriter as V1FileWriter, FileWriterOptions as V1FileWriterOptions,
    };
    use lance_index::vector::v3::subindex::SubIndexType;
    use lance_index::{
        INDEX_FILE_NAME, INDEX_METADATA_SCHEMA_KEY, Index, IndexMetadata, IndexType,
        vector::{Query, hnsw::VECTOR_ID_FIELD},
    };
    use lance_index::{
        metrics::MetricsCollector,
        vector::quantizer::{QuantizationType, Quantizer},
    };
    use lance_index::{metrics::NoOpMetricsCollector, vector::ivf::storage::IvfModel};
    use lance_io::traits::Reader;
    use lance_linalg::distance::MetricType;
    use lance_table::io::manifest::ManifestDescribing;
    use roaring::RoaringBitmap;
    use rstest::rstest;
    use serde_json::json;

    #[derive(Debug)]
    struct MockIndex;

    impl DeepSizeOf for MockIndex {
        fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
            0
        }
    }

    #[async_trait::async_trait]
    impl Index for MockIndex {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
            self
        }

        async fn prewarm(&self) -> Result<()> {
            Ok(())
        }

        fn statistics(&self) -> Result<serde_json::Value> {
            Ok(json!(()))
        }

        fn index_type(&self) -> IndexType {
            IndexType::Vector
        }

        async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
            Ok(RoaringBitmap::new())
        }
    }

    #[async_trait::async_trait]
    impl VectorIndex for MockIndex {
        async fn search(
            &self,
            _: &Query,
            _: Arc<dyn PreFilter>,
            _: &dyn MetricsCollector,
        ) -> Result<RecordBatch> {
            unimplemented!()
        }

        fn find_partitions(&self, _: &Query) -> Result<(UInt32Array, Float32Array)> {
            unimplemented!()
        }

        fn total_partitions(&self) -> usize {
            unimplemented!()
        }

        async fn search_in_partition(
            &self,
            _: usize,
            _: &Query,
            _: Arc<dyn PreFilter>,
            _: &dyn MetricsCollector,
        ) -> Result<RecordBatch> {
            unimplemented!()
        }

        fn is_loadable(&self) -> bool {
            true
        }

        fn use_residual(&self) -> bool {
            true
        }

        async fn load(
            &self,
            _: Arc<dyn Reader>,
            _: usize,
            _: usize,
        ) -> Result<Box<dyn VectorIndex>> {
            unimplemented!()
        }

        fn num_rows(&self) -> u64 {
            unimplemented!()
        }

        fn row_ids(&self) -> Box<dyn Iterator<Item = &u64>> {
            unimplemented!()
        }

        async fn remap(&mut self, _: &RowAddrRemap) -> Result<()> {
            Ok(())
        }

        async fn to_batch_stream(&self, _with_vector: bool) -> Result<SendableRecordBatchStream> {
            unimplemented!()
        }

        fn ivf_model(&self) -> &IvfModel {
            unimplemented!()
        }

        fn quantizer(&self) -> Quantizer {
            unimplemented!()
        }

        fn partition_size(&self, _: usize) -> usize {
            unimplemented!()
        }

        /// the index type of this vector index.
        fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
            unimplemented!()
        }

        fn metric_type(&self) -> MetricType {
            MetricType::L2
        }
    }

    struct MockIndexExtension {
        create_index_called: AtomicBool,
        load_index_called: AtomicBool,
    }

    impl MockIndexExtension {
        fn new() -> Self {
            Self {
                create_index_called: AtomicBool::new(false),
                load_index_called: AtomicBool::new(false),
            }
        }
    }

    impl DeepSizeOf for MockIndexExtension {
        fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
            todo!()
        }
    }

    impl IndexExtension for MockIndexExtension {
        fn index_type(&self) -> IndexType {
            IndexType::Vector
        }

        fn to_generic(self: Arc<Self>) -> Arc<dyn IndexExtension> {
            self
        }

        fn to_scalar(self: Arc<Self>) -> Option<Arc<dyn ScalarIndexExtension>> {
            None
        }

        fn to_vector(self: Arc<Self>) -> Option<Arc<dyn VectorIndexExtension>> {
            Some(self)
        }
    }

    #[async_trait::async_trait]
    impl VectorIndexExtension for MockIndexExtension {
        async fn create_index(
            &self,
            dataset: &Dataset,
            _column: &str,
            uuid: &Uuid,
            _params: &dyn IndexParams,
        ) -> Result<()> {
            let store = dataset.object_store.clone();
            let path = dataset
                .indices_dir()
                .join(uuid.to_string())
                .join(INDEX_FILE_NAME);

            let writer = store.create(&path).await.unwrap();

            let arrow_schema = Arc::new(Schema::new(vec![VECTOR_ID_FIELD.clone()]));
            let schema = lance_core::datatypes::Schema::try_from(arrow_schema.as_ref()).unwrap();
            let mut writer: V1FileWriter<ManifestDescribing> =
                V1FileWriter::with_object_writer(writer, schema, &V1FileWriterOptions::default())
                    .unwrap();
            writer.add_metadata(
                INDEX_METADATA_SCHEMA_KEY,
                json!(IndexMetadata {
                    index_type: "TEST".to_string(),
                    distance_type: "cosine".to_string(),
                })
                .to_string()
                .as_str(),
            );

            writer
                .write(&[RecordBatch::new_empty(arrow_schema)])
                .await
                .unwrap();
            writer.finish().await.unwrap();

            self.create_index_called
                .store(true, std::sync::atomic::Ordering::Release);

            Ok(())
        }

        async fn load_index(
            &self,
            _dataset: Arc<Dataset>,
            _column: &str,
            _uuid: &Uuid,
            _reader: V1FileReader,
        ) -> Result<Arc<dyn VectorIndex>> {
            self.load_index_called
                .store(true, std::sync::atomic::Ordering::Release);

            Ok(Arc::new(MockIndex))
        }
    }

    struct MockIndexParams;

    impl IndexParams for MockIndexParams {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn index_name(&self) -> &str {
            "TEST"
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_vector_index_extension_roundtrip(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // make dataset and index that is not supported natively
        let test_ds = TestVectorDataset::new(data_storage_version, false)
            .await
            .unwrap();
        let idx = test_ds.dataset.load_indices().await.unwrap();
        assert_eq!(idx.len(), 0);

        let idx_ext = Arc::new(MockIndexExtension::new());
        // make a new index with the extension
        let mut session = Session::default();
        session
            .register_index_extension("TEST".into(), idx_ext.clone())
            .unwrap();

        // neither has been called
        assert!(
            !idx_ext
                .create_index_called
                .load(std::sync::atomic::Ordering::Acquire)
        );
        assert!(
            !idx_ext
                .load_index_called
                .load(std::sync::atomic::Ordering::Acquire)
        );

        let mut ds_with_extension = DatasetBuilder::from_uri(&test_ds.tmp_dir)
            .with_session(Arc::new(session))
            .load()
            .await
            .unwrap();

        // create index
        ds_with_extension
            .create_index(&["vec"], IndexType::Vector, None, &MockIndexParams, false)
            .await
            .unwrap();

        // create index should have been called
        assert!(
            idx_ext
                .create_index_called
                .load(std::sync::atomic::Ordering::Acquire)
        );
        assert!(
            !idx_ext
                .load_index_called
                .load(std::sync::atomic::Ordering::Acquire)
        );

        // check that the index was created
        let ds_without_extension = DatasetBuilder::from_uri(&test_ds.tmp_dir)
            .load()
            .await
            .unwrap();
        let idx = ds_without_extension.load_indices().await.unwrap();
        assert_eq!(idx.len(), 1);
        // get the index uuid
        let index_uuid = idx.first().unwrap().uuid;

        // trying to open the index should fail as there is no extension loader
        assert!(
            ds_without_extension
                .open_vector_index("vec", &index_uuid, &NoOpMetricsCollector)
                .await
                .unwrap_err()
                .to_string()
                .contains("Unsupported index type: TEST")
        );

        // trying to open the index should succeed with the extension loader
        let vector_index = ds_with_extension
            .open_vector_index("vec", &index_uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        // load index should have been called
        assert!(
            idx_ext
                .create_index_called
                .load(std::sync::atomic::Ordering::Acquire)
        );
        assert!(
            idx_ext
                .load_index_called
                .load(std::sync::atomic::Ordering::Acquire)
        );

        // should be able to downcast to the mock index
        let _downcasted = vector_index.as_any().downcast_ref::<MockIndex>().unwrap();
    }

    /// A commit opens extension segments through their extension. They name no
    /// index type this build knows, which only another such segment matches.
    #[tokio::test]
    async fn test_vector_index_extension_segments_commit_by_their_contract() {
        use crate::dataset::CommitBuilder;
        use crate::dataset::transaction::{Operation, Transaction};

        let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
            .await
            .unwrap();
        let idx_ext = Arc::new(MockIndexExtension::new());
        let mut session = Session::default();
        session
            .register_index_extension("TEST".into(), idx_ext.clone())
            .unwrap();
        let mut dataset = DatasetBuilder::from_uri(&test_ds.tmp_dir)
            .with_session(Arc::new(session))
            .load()
            .await
            .unwrap();
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("ext".to_string()),
                &MockIndexParams,
                false,
            )
            .await
            .unwrap();
        let first = dataset
            .load_indices_by_name("ext")
            .await
            .unwrap()
            .pop()
            .unwrap();
        // A fragment of its own for each of the next two segments.
        test_ds.append_data_with_range(400, 410).await.unwrap();
        test_ds.append_data_with_range(410, 420).await.unwrap();
        dataset.checkout_latest().await.unwrap();
        let mut fragments = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        fragments.sort_unstable();
        let flat_fragment = fragments.pop().unwrap();
        let extension_fragment = fragments.pop().unwrap();

        let commit = |dataset: &Dataset, segment: lance_table::format::IndexMetadata| {
            let transaction = Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![lance_table::format::IndexMetadata {
                        name: "ext".to_string(),
                        ..segment
                    }],
                    removed_indices: vec![],
                },
                None,
            );
            CommitBuilder::new(Arc::new(dataset.clone())).execute(transaction)
        };

        // A second extension segment, as a coordinator could commit it.
        let uuid = Uuid::new_v4();
        idx_ext
            .create_index(&dataset, "vec", &uuid, &MockIndexParams)
            .await
            .unwrap();
        let second = lance_table::format::IndexMetadata {
            uuid,
            files: None,
            fragment_bitmap: Some([extension_fragment].into_iter().collect()),
            ..first.clone()
        };
        let mut dataset = commit(&dataset, second).await.unwrap();
        assert_eq!(dataset.load_indices_by_name("ext").await.unwrap().len(), 2);

        // A built-in segment with the same metric still differs in index type.
        let flat = dataset
            .create_index_builder(
                &["vec"],
                IndexType::Vector,
                &crate::index::vector::VectorIndexParams::ivf_flat(1, MetricType::L2),
            )
            .name("worker".to_string())
            .fragments(vec![flat_fragment])
            .execute_uncommitted()
            .await
            .unwrap();
        let err = commit(&dataset, flat.clone()).await.unwrap_err();
        assert!(
            matches!(err, lance_core::Error::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );
        // The mock's file says cosine, but the opened segment reports L2.
        assert!(
            err.to_string().contains("IVF_FLAT")
                && err
                    .to_string()
                    .contains("metric L2, an index extension's type"),
            "{err}"
        );

        // Only the extension opens its segments, so a session without it
        // cannot commit to the index.
        let without_extension = DatasetBuilder::from_uri(&test_ds.tmp_dir)
            .load()
            .await
            .unwrap();
        let err = commit(&without_extension, flat).await.unwrap_err();
        assert!(
            matches!(err, lance_core::Error::IO { .. })
                && err.to_string().contains(&first.uuid.to_string()),
            "{err:?}"
        );
    }
}
