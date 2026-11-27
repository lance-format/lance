// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream as DataFusionRecordBatchStream;
use futures::Stream;
use lance_core::{Result, datatypes::Schema};
use lance_datafusion::utils::{StreamingWriteSource, from_datafusion_stream};
use lance_file::{version::ConcreteFileVersion, writer::FileWriterOptions};
use lance_io::object_store::ObjectStore;
use lance_io::stream::{SendableRecordBatchStream, StreamingWriteSource as RuntimeWriteSource};
use lance_table::format::{BasePath, DataFile, Fragment};
use lance_table_runtime::WriteSeedHook;
use lance_table_runtime::blob::ExternalBaseResolver;
use lance_table_runtime::commit::CommitDestination;
use lance_table_runtime::write::FragmentWriteOptions;
use object_store::path::Path;
use tracing::instrument;

use crate::Dataset;
use crate::blob::BlobWriteSources;
use crate::session::Session;

use super::blob::{BlobPayloadInventory, SidecarRewrite};
use super::transaction::Transaction;
use super::{DataFilePart, DataFileTarget};

mod commit;
pub mod delete;
mod insert;
pub mod merge_insert;
mod retry;
pub(crate) mod seeds;
pub mod update;

pub use super::progress::{WriteProgressFn, WriteStats};
pub use commit::{CommitBuilder, DEFAULT_COMMIT_TIMEOUT};
pub use delete::{DeleteBuilder, DeleteResult, UncommittedDelete};
pub use insert::InsertBuilder;
pub use lance_table_runtime::write::{
    AutoCleanupParams, ExternalBlobMode, GenericWriter, PRIMARY_BASE_ID, TargetBaseInfo, WriteMode,
};
pub(crate) use lance_table_runtime::write::{
    cleanup_data_fragments, dictionary_budget, promote_legacy_blob_schema,
    validate_and_resolve_target_bases_with_primary,
};

/// Dataset write parameters with a Lance session.
pub type WriteParams = lance_table_runtime::write::WriteParamsWithSession<Session>;

pub(crate) fn into_runtime_write_params(params: WriteParams) -> lance_table_runtime::WriteParams {
    let session = params
        .session
        .as_ref()
        .map(|session| session.table_session.clone());
    params.with_session_type(session)
}

pub(super) async fn blob_v2_external_base_resolver(
    dataset: Option<&Dataset>,
    params: &WriteParams,
    schema: &Schema,
) -> Result<Option<Arc<ExternalBaseResolver>>> {
    let table = dataset.map(|dataset| dataset.table_view().to_table());
    let params = into_runtime_write_params(params.clone());
    Box::pin(lance_table_runtime::write::blob_v2_external_base_resolver(
        table.as_ref(),
        &params,
        schema,
    ))
    .await
}

/// Adapt the public Lance source without changing when it infers the schema.
pub(crate) struct DataFusionWriteSource<S>(pub(crate) S);

#[async_trait]
impl<S: StreamingWriteSource> RuntimeWriteSource for DataFusionWriteSource<S> {
    fn arrow_schema(&self) -> SchemaRef {
        self.0.arrow_schema()
    }

    fn into_stream(self) -> SendableRecordBatchStream {
        from_datafusion_stream(self.0.into_stream())
    }

    async fn into_stream_and_schema(self) -> Result<(SendableRecordBatchStream, Schema)> {
        let (stream, schema) = self.0.into_stream_and_schema().await?;
        Ok((from_datafusion_stream(stream), schema))
    }
}

impl Dataset {
    /// Encode one managed part and return its serializable description.
    ///
    /// Lance generates a unique staging name in the target's base. With schema
    /// metadata `lance.blob.managed=true`, Managed Blob payloads use independent
    /// `_blobs/<uuid>.blob` objects in that base. Otherwise payloads use sidecars
    /// addressed relative to the final data file. Every
    /// non-empty logical Inline value is spilled so final concatenation never
    /// copies Blob payload bytes. Managed parts accept `blob_ids=None`; only
    /// file-relative sidecars require disjoint reservations. Supplied ranges
    /// are ignored for Managed parts.
    /// Every use of `target` must refer to the same dataset and resolved base;
    /// associating a target with that storage context is the caller's
    /// responsibility.
    /// Persist the target before writing. A failed write may leave files; after
    /// stopping all users of the target, [`DataFileTarget::cleanup`] removes
    /// staging files and file-relative sidecars without a completed part description.
    /// File-relative retries must use fresh, disjoint Blob ID ranges, including
    /// ranges from failed writes. Independent Managed objects follow ordinary
    /// dataset GC rules, which must be coordinated with uncommitted writes
    /// and checkpoints.
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_array::RecordBatch;
    /// use futures::stream;
    /// use lance::{Dataset, dataset::DataFileTarget};
    ///
    /// # async fn write_part(
    /// #     dataset: &Dataset,
    /// #     target: &DataFileTarget,
    /// #     batch: RecordBatch,
    /// # ) -> lance_core::Result<()> {
    /// let part = dataset
    ///     .write_data_file_part(target, None, stream::iter([Ok(batch)]))
    ///     .await?;
    /// // Serialize part into the caller's checkpoint.
    /// # let _ = part;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_data_file_part(
        &self,
        target: &DataFileTarget,
        blob_ids: Option<Range<u32>>,
        data: impl Stream<Item = Result<RecordBatch>> + Send,
    ) -> Result<DataFilePart> {
        let table = self.table_view().to_table();
        Box::pin(table.write_data_file_part(&target.0, blob_ids, data)).await
    }

    /// Encode one managed part with runtime file-writer options.
    ///
    /// This has the same part description, Blob handling and concat semantics
    /// as [`Self::write_data_file_part`]. The options control buffering and
    /// execution only: the target still selects the file grammar and physical
    /// layout, so an assembled target is indistinguishable from one whose parts
    /// were written without them.
    ///
    /// Use this when the caller owns a memory budget for the writer. A part
    /// retains its encoded pages until [`Self::concat_data_file_parts`], so
    /// without an explicit budget each cache-owning field encoder buffers its
    /// own default allowance regardless of how many fields the schema has.
    pub async fn write_data_file_part_with_file_writer_options(
        &self,
        target: &DataFileTarget,
        blob_ids: Option<Range<u32>>,
        data: impl Stream<Item = Result<RecordBatch>> + Send,
        file_writer_options: FileWriterOptions,
    ) -> Result<DataFilePart> {
        let table = self.table_view().to_table();
        Box::pin(table.write_data_file_part_with_file_writer_options(
            &target.0,
            blob_ids,
            data,
            file_writer_options,
        ))
        .await
    }

    /// Encode one part with trusted payload sources accompanying its batches.
    ///
    /// Each source map belongs to one batch. Use prepared Blob v2 leaves and
    /// insert the descriptors returned by [`BlobWriteSources`] at the relevant
    /// slots. Uploaded keys are relative to the base of `target`. On an ordinary
    /// table every non-empty payload is spilled to sidecars using the target's
    /// allocated Blob IDs. A Managed-enabled table can reference the completed
    /// upload directly. The part and its sidecars use the same lifetime and
    /// concatenation rules as [`Self::write_data_file_part`].
    ///
    /// ```
    /// # use std::ops::Range;
    /// # use arrow_array::RecordBatch;
    /// # use lance::{Dataset, Result, blob::BlobWriteSources, dataset::DataFileTarget};
    /// # use lance_file::writer::FileWriterOptions;
    /// # async fn example(dataset: &Dataset, target: &DataFileTarget, ids: Option<Range<u32>>,
    /// #   batch: RecordBatch, sources: BlobWriteSources) -> Result<()> {
    /// let part = dataset.write_data_file_part_with_blob_sources(
    ///     target, ids, futures::stream::iter([Ok((batch, sources))]),
    ///     FileWriterOptions::default(),
    /// ).await?;
    /// # let _ = part;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_data_file_part_with_blob_sources(
        &self,
        target: &DataFileTarget,
        blob_ids: Option<Range<u32>>,
        data: impl Stream<Item = Result<(RecordBatch, BlobWriteSources)>> + Send,
        file_writer_options: FileWriterOptions,
    ) -> Result<DataFilePart> {
        let table = self.table_view().to_table();
        Box::pin(table.write_data_file_part_with_blob_sources(
            &target.0,
            blob_ids,
            data,
            file_writer_options,
        ))
        .await
    }

    /// Reopen, validate, and concatenate parts into the final data file.
    ///
    /// Part order is the final physical row order. The operation copies
    /// encoded page buffers and regenerates metadata and the footer; incompatible
    /// inputs fail without a decode/re-encode fallback or dataset commit. The
    /// caller owns cleanup of all durable part, Blob, and final-file objects.
    /// The caller must also assemble the target through the same dataset and
    /// resolved base used to write managed Blob payloads.
    /// The target must still be uncommitted and have no concurrent assembler.
    /// Assembly can overwrite a previous uncommitted output for the same target;
    /// it does not check current or historical manifests for references.
    ///
    /// To replace a fragment's columns, wrap the returned file in a
    /// [`DataReplacementGroup`](super::transaction::DataReplacementGroup).
    /// The caller must check that the ordered parts cover the fragment's physical
    /// rows exactly, including deleted rows; this operation does not check
    /// fragment coverage or commit the replacement.
    ///
    /// # Example
    ///
    /// ```
    /// use lance::{Dataset, dataset::{DataFilePart, DataFileTarget}};
    ///
    /// # async fn concat(
    /// #     dataset: &Dataset,
    /// #     target: &DataFileTarget,
    /// #     ordered_parts: &[DataFilePart],
    /// # ) -> lance_core::Result<()> {
    /// let data_file = dataset.concat_data_file_parts(target, ordered_parts).await?;
    /// // The caller decides when and how to commit `data_file`.
    /// # let _ = data_file;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn concat_data_file_parts(
        &self,
        target: &DataFileTarget,
        ordered_parts: &[DataFilePart],
    ) -> Result<DataFile> {
        let table = self.table_view().to_table();
        Box::pin(table.concat_data_file_parts(&target.0, ordered_parts)).await
    }
}

/// The destination to write data to.
#[derive(Debug, Clone)]
pub enum WriteDestination<'a> {
    /// An existing dataset to write to.
    Dataset(Arc<Dataset>),
    /// A URI to write to.
    Uri(&'a str),
}

impl<'a> WriteDestination<'a> {
    pub fn dataset(&self) -> Option<&Dataset> {
        match self {
            WriteDestination::Dataset(dataset) => Some(dataset.as_ref()),
            WriteDestination::Uri(_) => None,
        }
    }

    pub(crate) fn to_runtime(&self) -> CommitDestination<'a> {
        match self {
            Self::Dataset(dataset) => dataset.table_view().to_table().into(),
            Self::Uri(uri) => CommitDestination::Uri(uri),
        }
    }

    pub fn uri(&self) -> String {
        match self {
            WriteDestination::Dataset(dataset) => dataset.table_state.uri.clone(),
            WriteDestination::Uri(uri) => uri.to_string(),
        }
    }
}

impl From<Arc<Dataset>> for WriteDestination<'_> {
    fn from(dataset: Arc<Dataset>) -> Self {
        WriteDestination::Dataset(dataset)
    }
}

impl<'a> From<&'a str> for WriteDestination<'a> {
    fn from(uri: &'a str) -> Self {
        WriteDestination::Uri(uri)
    }
}

impl<'a> From<&'a String> for WriteDestination<'a> {
    fn from(uri: &'a String) -> Self {
        WriteDestination::Uri(uri.as_str())
    }
}

impl<'a> From<&'a Path> for WriteDestination<'a> {
    fn from(path: &'a Path) -> Self {
        WriteDestination::Uri(path.as_ref())
    }
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
#[deprecated(
    since = "0.20.0",
    note = "Use [`InsertBuilder::execute_uncommitted_stream`] instead"
)]
pub async fn write_fragments(
    dest: impl Into<WriteDestination<'_>>,
    data: impl StreamingWriteSource,
    params: WriteParams,
) -> Result<Transaction> {
    InsertBuilder::new(dest.into())
        .with_params(&params)
        .execute_uncommitted_stream(data)
        .await
}

/// Resolve data-file storage bases against write options and existing metadata.
pub async fn validate_and_resolve_target_bases(
    params: &mut WriteParams,
    existing_base_paths: Option<&HashMap<u32, BasePath>>,
) -> Result<Option<Vec<TargetBaseInfo>>> {
    Box::pin(
        lance_table_runtime::write::validate_and_resolve_target_bases(params, existing_base_paths),
    )
    .await
}

/// Writes the given data to the dataset and returns fragments.
///
/// NOTE: the fragments have not yet been assigned an ID. That must be done
/// by the caller. This is so this function can be called in parallel, and the
/// IDs can be assigned after writing is complete.
///
/// Accepts a DataFusion stream and an already resolved write schema.
///
/// The caller must resolve `storage_version` once for the operation. Operations
/// that also select a commit format must reuse the same value when committing.
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip_all)]
pub async fn write_fragments_internal(
    storage_version: ConcreteFileVersion,
    dataset: Option<&Dataset>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: Schema,
    data: DataFusionRecordBatchStream,
    params: WriteParams,
    target_bases_info: Option<Vec<TargetBaseInfo>>,
) -> Result<(Vec<Fragment>, Schema)> {
    write_fragments_internal_with_file_row_counts(
        storage_version,
        dataset,
        object_store,
        base_dir,
        schema,
        from_datafusion_stream(data),
        params,
        target_bases_info,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip_all)]
pub(crate) async fn write_fragments_internal_with_file_row_counts(
    storage_version: ConcreteFileVersion,
    dataset: Option<&Dataset>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: Schema,
    data: SendableRecordBatchStream,
    params: WriteParams,
    target_bases_info: Option<Vec<TargetBaseInfo>>,
    file_row_counts: Option<Vec<usize>>,
) -> Result<(Vec<Fragment>, Schema)> {
    write_fragments_internal_impl(
        storage_version,
        dataset,
        object_store,
        base_dir,
        schema,
        data,
        params,
        target_bases_info,
        file_row_counts,
        None,
        false,
        false,
        None,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(in crate::dataset) async fn write_fragments_internal_impl(
    storage_version: ConcreteFileVersion,
    dataset: Option<&Dataset>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: Schema,
    data: SendableRecordBatchStream,
    params: WriteParams,
    target_bases_info: Option<Vec<TargetBaseInfo>>,
    file_row_counts: Option<Vec<usize>>,
    preassigned_data_file_name: Option<Arc<String>>,
    inherited_field_ids: bool,
    disable_dictionary_split: bool,
    created_blob_payloads: Option<BlobPayloadInventory>,
    sidecar_rewrite: Option<Arc<SidecarRewrite>>,
) -> Result<(Vec<Fragment>, Schema)> {
    let hook = dataset.map(|dataset| seeds::LanceWriteSeedHook::new(dataset.session.clone()));
    let table = dataset.map(|dataset| dataset.table_view().to_table());
    Box::pin(lance_table_runtime::write::write_fragments(
        storage_version,
        table.as_ref(),
        object_store,
        base_dir,
        schema,
        data,
        into_runtime_write_params(params),
        FragmentWriteOptions {
            target_bases_info,
            file_row_counts,
            preassigned_data_file_name,
            inherited_field_ids,
            disable_dictionary_split,
            created_blob_payloads,
            sidecar_rewrite,
            seed_hook: hook.as_ref().map(|hook| hook as &dyn WriteSeedHook),
        },
    ))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DATA_DIR;
    use crate::dataset::progress::WriteFragmentProgress;
    use crate::index::DatasetIndexExt;
    use arrow_array::ArrayRef;
    use lance_core::Error;
    use lance_file::version::LanceFileVersion;
    use std::collections::HashMap;
    #[cfg(windows)]
    use std::path::{Component, Prefix};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    use arrow_array::cast::AsArray;
    use arrow_array::{
        Int32Array, LargeBinaryArray, RecordBatchIterator, RecordBatchReader, StringArray,
        StringViewArray, StructArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
    use async_trait::async_trait;
    use datafusion::{error::DataFusionError, physical_plan::stream::RecordBatchStreamAdapter};
    use datafusion_physical_plan::RecordBatchStream;
    use futures::TryStreamExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datafusion::chunker::chunk_stream;
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use lance_file::version::ConcreteFileVersion;
    use lance_file::versions::v1::reader::FileReader as V1FileReader;
    use lance_index::IndexType;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_io::object_store::{ObjectStoreParams, StorageOptionsAccessor};
    use lance_io::stream::RecordBatchStreamAdapter as LanceRecordBatchStreamAdapter;
    use lance_io::traits::Reader;
    use lance_table::format::BasePath;
    use parquet_variant_compute::{
        VariantArray, json_to_variant, shred_variant, unshred_variant, variant_to_json,
    };
    use rstest::rstest;

    async fn scan_sorted_ids(dataset: &Dataset) -> Vec<i32> {
        let batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut ids: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn test_auto_cleanup_disabled_by_default() {
        // Auto-cleanup must be off by default: the cleanup hook is expensive on
        // object stores and the 14-day default rarely deletes anything anyway.
        // See https://github.com/lance-format/lance/issues/6728
        let params = WriteParams::default();
        assert!(params.auto_cleanup.is_none());
        assert!(!params.skip_auto_cleanup);
    }

    #[tokio::test]
    async fn test_disable_transaction_file_applies_to_its_commit_only() {
        use crate::dataset::transaction::Operation;
        use lance_table::feature_flags::FLAG_DISABLE_TRANSACTION_FILE;

        let tmp = TempStrDir::default();
        let uri = tmp.as_str();
        let transaction_file_count = || {
            std::fs::read_dir(std::path::Path::new(uri).join("_transactions"))
                .map(|entries| entries.count())
                .unwrap_or(0)
        };
        let data = || {
            gen_batch()
                .col("id", array::step::<arrow_array::types::Int32Type>())
                .into_reader_rows(RowCount::from(3), BatchCount::from(1))
        };

        let params = WriteParams {
            disable_transaction_file: true,
            ..Default::default()
        };
        let dataset = Dataset::write(data(), uri, Some(params)).await.unwrap();
        assert!(dataset.manifest.transaction_file.is_none());
        assert!(dataset.manifest.transaction_section.is_some());
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_DISABLE_TRANSACTION_FILE,
            0
        );
        assert_eq!(transaction_file_count(), 0);

        // A fresh handle has no cached transaction, so this reads the inline section.
        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), 3);
        let transaction = reopened.read_transaction().await.unwrap().unwrap();
        assert!(matches!(transaction.operation, Operation::Overwrite { .. }));

        let params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = Dataset::write(data(), uri, Some(params)).await.unwrap();
        assert!(dataset.manifest.transaction_file.is_some());
        assert_eq!(
            dataset.manifest.writer_feature_flags & FLAG_DISABLE_TRANSACTION_FILE,
            0
        );
        assert_eq!(transaction_file_count(), 1);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn test_create_and_reopen_from_unc_uri() {
        let tempdir = tempfile::tempdir().unwrap();
        let dataset_path = tempdir.path().join("dataset with spaces");
        let mut components = dataset_path.components();
        let drive_letter = match components.next() {
            Some(Component::Prefix(prefix)) => match prefix.kind() {
                Prefix::Disk(letter) | Prefix::VerbatimDisk(letter) => letter,
                other => panic!("expected a disk path, found {other:?}"),
            },
            other => panic!("expected a disk path, found {other:?}"),
        };
        assert!(matches!(components.next(), Some(Component::RootDir)));

        // The administrative disk share provides a real loopback UNC path without
        // requiring an external SMB service.
        let computer_name = std::env::var("COMPUTERNAME").unwrap();
        let mut dataset_uri = url::Url::parse(&format!("file://{computer_name}/")).unwrap();
        let share = format!("{}$", char::from(drive_letter));
        {
            let mut segments = dataset_uri.path_segments_mut().unwrap();
            segments.pop_if_empty().push(&share);
            for component in components {
                let Component::Normal(segment) = component else {
                    panic!("unexpected dataset path component: {component:?}");
                };
                segments.push(segment.to_str().unwrap());
            }
        }
        assert!(dataset_uri.as_str().contains("%20"));

        let reader = gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(3), BatchCount::from(1));
        let dataset = Dataset::write(reader, dataset_uri.as_str(), None)
            .await
            .unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
        drop(dataset);

        let reopened = Dataset::open(dataset_uri.as_str()).await.unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_chunking_large_batches() {
        // Create a stream of 3 batches of 10 rows
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from_iter(0..28))])
                .unwrap();
        let batches: Vec<RecordBatch> =
            vec![batch.slice(0, 10), batch.slice(10, 10), batch.slice(20, 8)];
        let stream = RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(batches.into_iter().map(Ok::<_, DataFusionError>)),
        );

        // Chunk into a stream of 3 row batches
        let chunks: Vec<Vec<RecordBatch>> = chunk_stream(Box::pin(stream), 3)
            .try_collect()
            .await
            .unwrap();

        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks[0].len(), 1);

        for (i, chunk) in chunks.iter().enumerate() {
            let num_rows = chunk.iter().map(|batch| batch.num_rows()).sum::<usize>();
            if i < chunks.len() - 1 {
                assert_eq!(num_rows, 3);
            } else {
                // Last chunk is shorter
                assert_eq!(num_rows, 1);
            }
        }

        // The fourth chunk is split along the boundary between the original first
        // two batches.
        assert_eq!(chunks[3].len(), 2);
        assert_eq!(chunks[3][0].num_rows(), 1);
        assert_eq!(chunks[3][1].num_rows(), 2);
    }

    #[tokio::test]
    async fn test_chunking_small_batches() {
        // Create a stream of 10 batches of 3 rows
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from_iter(0..30))])
                .unwrap();

        let batches: Vec<RecordBatch> = (0..10).map(|i| batch.slice(i * 3, 3)).collect();
        let stream = RecordBatchStreamAdapter::new(
            schema.clone(),
            futures::stream::iter(batches.into_iter().map(Ok::<_, DataFusionError>)),
        );

        // Chunk into a stream of 10 row batches
        let chunks: Vec<Vec<RecordBatch>> = chunk_stream(Box::pin(stream), 10)
            .try_collect()
            .await
            .unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 4);
        assert_eq!(chunks[0][0], batch.slice(0, 3));
        assert_eq!(chunks[0][1], batch.slice(3, 3));
        assert_eq!(chunks[0][2], batch.slice(6, 3));
        assert_eq!(chunks[0][3], batch.slice(9, 1));

        for chunk in &chunks {
            let num_rows = chunk.iter().map(|batch| batch.num_rows()).sum::<usize>();
            assert_eq!(num_rows, 10);
        }
    }

    #[tokio::test]
    async fn test_file_size() {
        let reader_to_frags = |data_reader: Box<dyn RecordBatchReader + Send>| {
            let schema = data_reader.schema();
            let data_reader =
                data_reader.map(|rb| rb.map_err(datafusion::error::DataFusionError::from));

            let data_stream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter(data_reader),
            ));

            let write_params = WriteParams {
                max_rows_per_file: 1024 * 1024, // Won't be limited by this
                max_bytes_per_file: 2 * 1024,
                mode: WriteMode::Create,
                ..Default::default()
            };

            async move {
                let schema = Schema::try_from(schema.as_ref()).unwrap();

                let object_store = Arc::new(ObjectStore::memory());
                write_fragments_internal(
                    write_params.storage_version_or_default(),
                    None,
                    object_store,
                    &Path::from("test"),
                    schema,
                    data_stream,
                    write_params,
                    None,
                )
                .await
            }
        };

        // The writer will not generate a new file until at enough data is *written* (not
        // just accumulated) to justify a new file.  Since the default page size is 8MiB
        // we actually need to generate quite a bit of data to trigger this.
        //
        // To avoid generating and writing millions of rows (which is a bit slow for a unit
        // test) we can use a large data type (1KiB binary)
        let data_reader = Box::new(
            gen_batch()
                .anon_col(array::rand_fsb(1024))
                .into_reader_rows(RowCount::from(10 * 1024), BatchCount::from(2)),
        );

        let (fragments, _) = reader_to_frags(data_reader).await.unwrap();

        assert_eq!(fragments.len(), 2);
    }

    #[rstest]
    #[case::rebalance_pending_remainder(
        &[9_999, 10_001],
        &[10_000, 10_000],
        2 * 1024,
        &[9_999, 5_001, 5_000]
    )]
    #[case::replan_pending_boundary(
        &[9_999, 1, 10_000, 10_000],
        &[20_000, 10_000],
        100 * 1024,
        &[9_999, 10_001, 10_000]
    )]
    #[tokio::test]
    async fn test_planned_file_boundary_with_byte_limit(
        #[case] input_batch_sizes: &[usize],
        #[case] file_row_counts: &[usize],
        #[case] max_bytes_per_file: usize,
        #[case] expected_file_rows: &[usize],
    ) {
        let value = vec![0_u8; 1024];
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::LargeBinary,
            false,
        )]));
        let total_rows = input_batch_sizes.iter().sum::<usize>();
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(LargeBinaryArray::from_iter_values(
                (0..total_rows).map(|_| value.as_slice()),
            ))],
        )
        .unwrap();
        let mut offset = 0;
        let batches = input_batch_sizes
            .iter()
            .map(|&batch_rows| {
                let batch = data.slice(offset, batch_rows);
                offset += batch_rows;
                Ok(batch)
            })
            .collect::<Vec<_>>();
        let stream = LanceRecordBatchStreamAdapter::new(
            arrow_schema.clone(),
            futures::stream::iter(batches),
        );
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let object_store = Arc::new(ObjectStore::memory());

        let (fragments, _) = write_fragments_internal_with_file_row_counts(
            ConcreteFileVersion::V2_0,
            None,
            object_store,
            &Path::from("planned_byte_boundary"),
            schema,
            Box::pin(stream),
            WriteParams {
                max_rows_per_file: file_row_counts[0],
                max_bytes_per_file,
                mode: WriteMode::Create,
                ..Default::default()
            },
            None,
            Some(file_row_counts.to_vec()),
        )
        .await
        .unwrap();

        assert_eq!(
            fragments
                .iter()
                .map(|fragment| fragment.physical_rows.unwrap())
                .collect::<Vec<_>>(),
            expected_file_rows
        );
    }

    #[tokio::test]
    async fn test_repeated_byte_closes_rebalance_planned_rows() {
        let large_value = vec![0_u8; 16 * 1024 * 1024];
        let mut values = Vec::with_capacity(15);
        values.extend(std::iter::repeat_n(large_value.as_slice(), 2));
        values.extend(std::iter::repeat_n(&[][..], 13));
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::LargeBinary,
            false,
        )]));
        let data = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(LargeBinaryArray::from_iter_values(values))],
        )
        .unwrap();
        let input_batch_sizes = [1, 1, 3, 5, 5];
        let mut offset = 0;
        let batches = input_batch_sizes.map(|batch_rows| {
            let batch = data.slice(offset, batch_rows);
            offset += batch_rows;
            Ok(batch)
        });
        let stream = LanceRecordBatchStreamAdapter::new(
            arrow_schema.clone(),
            futures::stream::iter(batches),
        );
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let object_store = Arc::new(ObjectStore::memory());

        let (fragments, _) = write_fragments_internal_with_file_row_counts(
            ConcreteFileVersion::V2_0,
            None,
            object_store,
            &Path::from("repeated_planned_byte_boundaries"),
            schema,
            Box::pin(stream),
            WriteParams {
                max_rows_per_file: 5,
                max_bytes_per_file: 100 * 1024,
                mode: WriteMode::Create,
                ..Default::default()
            },
            None,
            Some(vec![5, 5, 5]),
        )
        .await
        .unwrap();

        assert_eq!(
            fragments
                .iter()
                .map(|fragment| fragment.physical_rows.unwrap())
                .collect::<Vec<_>>(),
            [1, 1, 5, 4, 4]
        );
    }

    #[tokio::test]
    async fn test_max_rows_per_file_rejects_fragment_row_overflow() {
        let tmpdir = lance_core::utils::tempfile::TempStrDir::default();
        let batch = RecordBatch::try_from_iter([(
            "value",
            Arc::new(UInt64Array::from_iter_values(0..4)) as ArrayRef,
        )])
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());

        let error = Dataset::write(
            reader,
            tmpdir.as_str(),
            Some(WriteParams {
                max_rows_per_file: u32::MAX as usize + 1,
                data_storage_version: Some(lance_file::version::LanceFileVersion::V2_0),
                ..Default::default()
            }),
        )
        .await
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("cannot exceed the fragment row capacity"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_max_rows_per_file() {
        let reader_to_frags = |data_reader: Box<dyn RecordBatchReader + Send>| {
            let schema = data_reader.schema();
            let data_reader =
                data_reader.map(|rb| rb.map_err(datafusion::error::DataFusionError::from));

            let data_stream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter(data_reader),
            ));

            let write_params = WriteParams {
                max_rows_per_file: 5000,                // Limit by rows
                max_bytes_per_file: 1024 * 1024 * 1024, // Won't be limited by this
                mode: WriteMode::Create,
                ..Default::default()
            };

            async move {
                let schema = Schema::try_from(schema.as_ref()).unwrap();

                let object_store = Arc::new(ObjectStore::memory());
                write_fragments_internal(
                    write_params.storage_version_or_default(),
                    None,
                    object_store,
                    &Path::from("test"),
                    schema,
                    data_stream,
                    write_params,
                    None,
                )
                .await
            }
        };

        // Generate 12000 rows total, which should create 3 files:
        // - File 1: 5000 rows
        // - File 2: 5000 rows
        // - File 3: 2000 rows
        let data_reader = Box::new(
            gen_batch()
                .anon_col(array::rand_type(&DataType::Int32))
                .into_reader_rows(RowCount::from(12000), BatchCount::from(1)),
        );

        let (fragments, _) = reader_to_frags(data_reader).await.unwrap();

        // Should have 3 fragments
        assert_eq!(fragments.len(), 3);

        // Verify the row count distribution
        let row_counts: Vec<usize> = fragments
            .iter()
            .map(|f| f.physical_rows.unwrap_or(0))
            .collect();
        assert_eq!(row_counts, vec![5000, 5000, 2000]);
    }

    /// A zero row limit used to panic on V2 and, on V1, commit an empty
    /// dataset. V2 ignores `max_rows_per_group`, so a zero there still writes.
    #[rstest]
    #[case::v2_rows_per_file(LanceFileVersion::Stable, 0, 1024, Some("max_rows_per_file"))]
    #[case::v1_rows_per_file(LanceFileVersion::Legacy, 0, 1024, Some("max_rows_per_file"))]
    #[case::v1_rows_per_group(LanceFileVersion::Legacy, 1024, 0, Some("max_rows_per_group"))]
    #[case::v2_ignores_rows_per_group(LanceFileVersion::Stable, 1024, 0, None)]
    #[tokio::test]
    async fn test_write_rejects_zero_row_limits(
        #[case] version: LanceFileVersion,
        #[case] max_rows_per_file: usize,
        #[case] max_rows_per_group: usize,
        #[case] rejected: Option<&str>,
    ) {
        let data = gen_batch()
            .col("a", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(2));
        let params = WriteParams {
            data_storage_version: Some(version),
            max_rows_per_file,
            max_rows_per_group,
            ..Default::default()
        };
        let result = Dataset::write(data, "memory://zero-row-limits", Some(params)).await;
        match rejected {
            Some(param) => {
                let err = result.unwrap_err();
                assert!(
                    matches!(&err, Error::InvalidInput { source, .. }
                    if source.to_string().contains(&format!("{param} must be greater than zero"))),
                    "{err:?}"
                );
            }
            None => assert_eq!(result.unwrap().count_rows(None).await.unwrap(), 20),
        }
    }

    #[tokio::test]
    async fn test_max_rows_per_group() {
        let reader_to_frags = |data_reader: Box<dyn RecordBatchReader + Send>,
                               version: LanceFileVersion| {
            let schema = data_reader.schema();
            let data_reader =
                data_reader.map(|rb| rb.map_err(datafusion::error::DataFusionError::from));

            let data_stream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter(data_reader),
            ));

            let write_params = WriteParams {
                max_rows_per_file: 5000,  // Smaller than total data to force multiple files
                max_rows_per_group: 3000, // Row group size affects V1 only
                mode: WriteMode::Create,
                data_storage_version: Some(version),
                ..Default::default()
            };

            async move {
                let schema = Schema::try_from(schema.as_ref()).unwrap();

                let object_store = Arc::new(ObjectStore::memory());
                write_fragments_internal(
                    write_params.storage_version_or_default(),
                    None,
                    object_store,
                    &Path::from("test"),
                    schema,
                    data_stream,
                    write_params,
                    None,
                )
                .await
            }
        };

        // Test V1 (Legacy) version: max_rows_per_group affects chunking
        // With max_rows_per_group=3000 and max_rows_per_file=5000:
        // - Stream is chunked into batches of max 3000 rows
        // - Batches are written to files, splitting when file exceeds 5000 rows
        // For 9000 rows:
        //   - Chunk 1 (3000 rows) -> File 1 (6000 rows) - exceeds limit, triggers new file
        //   - Chunk 2 (3000 rows) -> File 2 (3000 rows) - start of new file
        // Result: 2 fragments with [6000, 3000] rows
        // Note: The exact behavior depends on when file splitting occurs
        let data_reader_v1 = Box::new(
            gen_batch()
                .anon_col(array::rand_type(&DataType::Int32))
                .into_reader_rows(RowCount::from(9000), BatchCount::from(1)),
        );

        let (fragments_v1, _) = reader_to_frags(data_reader_v1, LanceFileVersion::Legacy)
            .await
            .unwrap();
        let row_counts_v1: Vec<usize> = fragments_v1
            .iter()
            .map(|f| f.physical_rows.unwrap_or(0))
            .collect();

        // V1 creates 2 fragments based on row group chunking and file size limit
        assert_eq!(fragments_v1.len(), 2);
        assert_eq!(row_counts_v1, vec![6000, 3000]);

        // Test V2+ version: max_rows_per_group is ignored, only max_rows_per_file matters
        // With max_rows_per_file=5000 and 9000 rows:
        // - Stream is not chunked by row group size
        // - Data is split only at file boundaries (5000 rows per file)
        // Result: 2 fragments with [5000, 4000] rows
        // V2 splits data more evenly at file boundaries regardless of row group size
        let data_reader_v2 = Box::new(
            gen_batch()
                .anon_col(array::rand_type(&DataType::Int32))
                .into_reader_rows(RowCount::from(9000), BatchCount::from(1)),
        );

        let (fragments_v2, _) = reader_to_frags(data_reader_v2, LanceFileVersion::Stable)
            .await
            .unwrap();
        let row_counts_v2: Vec<usize> = fragments_v2
            .iter()
            .map(|f| f.physical_rows.unwrap_or(0))
            .collect();

        // V2 should create 2 fragments based on file size only
        assert_eq!(fragments_v2.len(), 2);
        assert_eq!(row_counts_v2, vec![5000, 4000]);

        // Key difference: Both V1 and V2 create 2 fragments, but with different distributions
        // - V1: [6000, 3000] - chunking by row groups affects distribution
        // - V2: [5000, 4000] - split only at file boundaries, more even
        // V2 distribution should be more even (closer to 5000/5000 split)
        // V1 distribution is affected by row group chunking (3000)
        assert_eq!(fragments_v1.len(), fragments_v2.len());
        assert_ne!(row_counts_v1, row_counts_v2);
    }

    #[tokio::test]
    async fn test_file_write_version() {
        let schema = Arc::new(ArrowSchema::new(vec![arrow::datatypes::Field::new(
            "a",
            DataType::Int32,
            false,
        )]));

        // Write 1024 rows
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter(0..1024))],
        )
        .unwrap();

        let versions = vec![
            LanceFileVersion::Legacy,
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::V2_2,
            LanceFileVersion::Stable,
            LanceFileVersion::Next,
        ];
        for version in versions {
            let (major, minor) = version.resolve().to_data_file_numbers();

            let write_params = WriteParams {
                data_storage_version: Some(version),
                // This parameter should be ignored
                max_rows_per_group: 1,
                ..Default::default()
            };

            let data_stream = Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter(std::iter::once(Ok(batch.clone()))),
            ));

            let schema = Schema::try_from(schema.as_ref()).unwrap();

            let object_store = Arc::new(ObjectStore::memory());
            let (fragments, _) = write_fragments_internal(
                version.resolve(),
                None,
                object_store,
                &Path::from("test"),
                schema,
                data_stream,
                write_params,
                None,
            )
            .await
            .unwrap();

            assert_eq!(fragments.len(), 1);
            let fragment = &fragments[0];
            assert_eq!(fragment.files.len(), 1);
            assert_eq!(fragment.physical_rows, Some(1024));
            assert_eq!(
                fragment.files[0].file_major_version, major,
                "version: {}",
                version
            );
            assert_eq!(
                fragment.files[0].file_minor_version, minor,
                "version: {}",
                version
            );
        }
    }

    #[tokio::test]
    async fn test_file_v1_schema_order() {
        // Create a schema where fields ids are not in order and contain holes.
        // Also first field id is a struct.
        let struct_fields = Fields::from(vec![ArrowField::new("b", DataType::Int32, false)]);
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("d", DataType::Int32, false),
            ArrowField::new("a", DataType::Struct(struct_fields.clone()), false),
        ]);
        let mut schema = Schema::try_from(&arrow_schema).unwrap();
        // Make schema:
        // 0: a
        // 1: a.b
        // (hole at 2)
        // 3: d
        schema.mut_field_by_id(0).unwrap().id = 3;
        schema.mut_field_by_id(1).unwrap().id = 0;
        schema.mut_field_by_id(2).unwrap().id = 1;

        let field_ids = schema.fields_pre_order().map(|f| f.id).collect::<Vec<_>>();
        assert_eq!(field_ids, vec![3, 0, 1]);

        let data = RecordBatch::try_new(
            Arc::new(arrow_schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StructArray::new(
                    struct_fields,
                    vec![Arc::new(Int32Array::from(vec![3, 4]))],
                    None,
                )),
            ],
        )
        .unwrap();

        let write_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        };
        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            Arc::new(arrow_schema),
            futures::stream::iter(std::iter::once(Ok(data.clone()))),
        ));

        let object_store = Arc::new(ObjectStore::memory());
        let base_path = Path::from("test");
        let (fragments, written_schema) = write_fragments_internal(
            ConcreteFileVersion::V1,
            None,
            object_store.clone(),
            &base_path,
            schema.clone(),
            data_stream,
            write_params,
            None,
        )
        .await
        .unwrap();

        assert_eq!(fragments.len(), 1);
        let fragment = &fragments[0];
        assert_eq!(fragment.files.len(), 1);
        // A new dataset keeps the supplied field IDs, in the schema's field order.
        assert_eq!(
            written_schema
                .fields_pre_order()
                .map(|field| field.id)
                .collect::<Vec<_>>(),
            vec![3, 0, 1]
        );
        assert_eq!(fragment.files[0].fields.as_ref(), &[0, 1, 3]);

        let path = base_path
            .clone()
            .join(DATA_DIR)
            .join(fragment.files[0].path.as_str());
        let file_reader: Arc<dyn Reader> = object_store.open(&path).await.unwrap().into();
        let reader = V1FileReader::try_new_from_reader(
            &path,
            file_reader,
            None,
            written_schema.clone(),
            0,
            0,
            3,
            None,
        )
        .await
        .unwrap();
        assert_eq!(reader.num_batches(), 1);
        let batch = reader.read_batch(0, .., &written_schema).await.unwrap();
        assert_eq!(batch, data);
    }

    #[cfg(feature = "azure")]
    fn azure_store_params(account_name: &str) -> ObjectStoreParams {
        ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("account_name".to_string(), account_name.to_string()),
                    ("account_key".to_string(), "dGVzdA==".to_string()),
                ]),
            ))),
            ..Default::default()
        }
    }

    #[cfg(feature = "azure")]
    #[tokio::test]
    async fn test_validate_and_resolve_target_bases_uses_base_store_params() {
        let mut params = WriteParams::default()
            .with_target_bases(vec![1, 2])
            .with_base_store_params("az://container/path-a", azure_store_params("account-a"))
            .with_base_store_params("az://container/path-b", azure_store_params("account-b"));

        let existing_base_paths = azure_base_paths_a_b();

        let target_bases =
            validate_and_resolve_target_bases(&mut params, Some(&existing_base_paths))
                .await
                .unwrap()
                .unwrap();

        assert_eq!(target_bases.len(), 2);
        assert_eq!(
            target_bases[0].object_store.store_prefix,
            "az$container@account-a"
        );
        assert_eq!(
            target_bases[1].object_store.store_prefix,
            "az$container@account-b"
        );
    }

    #[cfg(feature = "azure")]
    fn azure_base_paths_a_b() -> HashMap<u32, BasePath> {
        HashMap::from([
            (
                1,
                BasePath::new(
                    1,
                    "az://container/path-a".to_string(),
                    Some("base-a".to_string()),
                    false,
                ),
            ),
            (
                2,
                BasePath::new(
                    2,
                    "az://container/path-b".to_string(),
                    Some("base-b".to_string()),
                    false,
                ),
            ),
        ])
    }

    #[cfg(feature = "azure")]
    #[tokio::test]
    async fn test_validate_and_resolve_target_bases_uses_base_scoped_storage_options() {
        // A single flat storage options map carries per-base credentials via
        // the `base_<id>.<key>` convention; unscoped keys are shared defaults.
        let store_params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("account_name".to_string(), "account-shared".to_string()),
                    ("account_key".to_string(), "dGVzdA==".to_string()),
                    ("base_1.account_name".to_string(), "account-a".to_string()),
                    ("base_2.account_name".to_string(), "account-b".to_string()),
                ]),
            ))),
            ..Default::default()
        };
        let mut params = WriteParams {
            store_params: Some(store_params),
            ..Default::default()
        }
        .with_target_bases(vec![1, 2]);

        let existing_base_paths = azure_base_paths_a_b();

        let target_bases =
            validate_and_resolve_target_bases(&mut params, Some(&existing_base_paths))
                .await
                .unwrap()
                .unwrap();

        assert_eq!(target_bases.len(), 2);
        assert_eq!(
            target_bases[0].object_store.store_prefix,
            "az$container@account-a"
        );
        assert_eq!(
            target_bases[1].object_store.store_prefix,
            "az$container@account-b"
        );
    }

    #[cfg(feature = "azure")]
    #[tokio::test]
    async fn test_base_store_params_take_precedence_over_base_scoped_options() {
        let store_params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("account_key".to_string(), "dGVzdA==".to_string()),
                    (
                        "base_1.account_name".to_string(),
                        "account-scoped".to_string(),
                    ),
                ]),
            ))),
            ..Default::default()
        };
        let mut params = WriteParams {
            store_params: Some(store_params),
            ..Default::default()
        }
        .with_target_bases(vec![1])
        .with_base_store_params("az://container/path-a", azure_store_params("account-exact"));

        let existing_base_paths = azure_base_paths_a_b();

        let target_bases =
            validate_and_resolve_target_bases(&mut params, Some(&existing_base_paths))
                .await
                .unwrap()
                .unwrap();

        assert_eq!(target_bases.len(), 1);
        assert_eq!(
            target_bases[0].object_store.store_prefix,
            "az$container@account-exact"
        );
    }

    #[tokio::test]
    async fn test_explicit_data_file_bases_path_parsing() {
        // Test URI parsing logic
        let test_cases = vec![
            ("s3://multi-path-test/test1/subBucket2", "test1/subBucket2"),
            ("gs://my-bucket/path/to/data", "path/to/data"),
            ("az://container/path/to/data", "path/to/data"),
            (
                "abfss://filesystem@account.dfs.core.windows.net/path/to/data",
                "path/to/data",
            ),
            ("file:///tmp/test/bucket", "tmp/test/bucket"),
        ];

        for (uri, expected_path) in test_cases {
            let url = url::Url::parse(uri).unwrap();
            let parsed_path = url.path().trim_start_matches('/');
            assert_eq!(parsed_path, expected_path, "Failed for URI: {}", uri);
        }
    }

    #[tokio::test]
    async fn test_write_params_validation() {
        // Test CREATE mode validation
        let mut params = WriteParams {
            mode: WriteMode::Create,
            initial_bases: Some(vec![
                BasePath {
                    id: 1,
                    name: Some("bucket1".to_string()),
                    path: "s3://bucket1/path1".to_string(),
                    is_dataset_root: true,
                },
                BasePath {
                    id: 2,
                    name: Some("bucket2".to_string()),
                    path: "s3://bucket2/path2".to_string(),
                    is_dataset_root: true,
                },
                BasePath {
                    id: 3,
                    name: Some("azure-az-base".to_string()),
                    path: "az://container/path1".to_string(),
                    is_dataset_root: true,
                },
                BasePath {
                    id: 4,
                    name: Some("azure-abfss-base".to_string()),
                    path: "abfss://filesystem@account.dfs.core.windows.net/path1".to_string(),
                    is_dataset_root: true,
                },
            ]),
            target_bases: Some(vec![1]), // Use ID 1 which corresponds to bucket1
            ..Default::default()
        };

        // This should be valid
        let result = validate_write_params(&params);
        assert!(result.is_ok());

        // Test target_bases with ID not in initial_bases (should fail)
        params.target_bases = Some(vec![99]); // ID 99 doesn't exist
        let result = validate_write_params(&params);
        assert!(result.is_err());

        // Test CREATE mode with target_bases but no initial_bases (should fail)
        params.initial_bases = None;
        params.target_bases = Some(vec![1]);
        let result = validate_write_params(&params);
        assert!(result.is_err());
    }

    fn validate_write_params(params: &WriteParams) -> Result<()> {
        // Replicate the validation logic from the main write function
        if matches!(params.mode, WriteMode::Create)
            && let Some(target_bases) = &params.target_bases
        {
            if target_bases.len() != 1 {
                return Err(Error::invalid_input(format!(
                    "target_bases with {} elements is not supported",
                    target_bases.len()
                )));
            }
            let target_base_id = target_bases[0];
            if let Some(initial_bases) = &params.initial_bases {
                if !initial_bases.iter().any(|bp| bp.id == target_base_id) {
                    return Err(Error::invalid_input(format!(
                        "target_base_id {} must be one of the initial_bases in CREATE mode",
                        target_base_id
                    )));
                }
            } else {
                return Err(Error::invalid_input(
                    "initial_bases must be provided when target_bases is specified in CREATE mode",
                ));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_external_blob_mode_validation() {
        let params = WriteParams {
            external_blob_mode: ExternalBlobMode::Ingest,
            allow_external_blob_outside_bases: true,
            ..Default::default()
        };

        let batch = arrow_array::record_batch!(("id", Int32, [1])).unwrap();
        let err = InsertBuilder::new("memory://")
            .with_params(&params)
            .execute(vec![batch])
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("allow_external_blob_outside_bases only applies")
        );
    }

    #[tokio::test]
    async fn test_multi_base_create() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create dataset with multi-base configuration
        let test_uri = "shared-memory://multi_base_test";
        let primary_uri = format!("{}/primary", test_uri);
        let base1_uri = format!("{}/base1", test_uri);
        let base2_uri = format!("{}/base2", test_uri);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = crate::dataset::Dataset::write(
            data_gen.batch(5),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: true,
                    },
                ]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Verify dataset was created
        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);

        // Verify base_paths are registered in manifest
        assert_eq!(dataset.manifest.base_paths.len(), 2);
        assert!(
            dataset
                .manifest
                .base_paths
                .values()
                .any(|bp| bp.name == Some("base1".to_string()))
        );
        assert!(
            dataset
                .manifest
                .base_paths
                .values()
                .any(|bp| bp.name == Some("base2".to_string()))
        );

        // Verify data was written to base1
        let fragments = dataset.get_fragments();
        assert!(!fragments.is_empty());
        for fragment in fragments {
            assert!(
                fragment
                    .metadata()
                    .files
                    .iter()
                    .any(|file| file.base_id == Some(1))
            );
        }

        assert_eq!(scan_sorted_ids(&dataset).await, (0..5).collect::<Vec<_>>());
        let reopened = Dataset::open(&primary_uri).await.unwrap();
        assert_eq!(scan_sorted_ids(&reopened).await, (0..5).collect::<Vec<_>>());

        // Test validation: cannot specify both target_bases and target_base_names_or_paths
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let result = Dataset::write(
            data_gen2.batch(5),
            &format!("{}/test_validation", test_uri),
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("base1".to_string()),
                    path: base1_uri.clone(),
                    is_dataset_root: true,
                }]),
                target_bases: Some(vec![1]),
                target_base_names_or_paths: Some(vec!["base1".to_string()]),
                ..Default::default()
            }),
        )
        .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot specify both target_base_names_or_paths and target_bases")
        );
    }

    #[rstest]
    #[case::auto_before_explicit("auto_before_explicit", &[0, 1])]
    #[case::explicit_before_auto("explicit_before_auto", &[1, 0])]
    #[tokio::test]
    async fn test_initial_base_ids_reserve_explicit_ids(
        #[case] case_name: &str,
        #[case] initial_ids: &[u32],
    ) {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        let test_uri = format!("shared-memory://initial_base_ids_{case_name}");
        let primary_uri = format!("{test_uri}/primary");
        let auto_uri = format!("{test_uri}/auto");
        let explicit_uri = format!("{test_uri}/explicit");

        let initial_bases = initial_ids
            .iter()
            .map(|id| {
                if *id == 0 {
                    BasePath::new(0, auto_uri.clone(), Some("auto".to_string()), true)
                } else {
                    BasePath::new(
                        *id,
                        explicit_uri.clone(),
                        Some("explicit".to_string()),
                        true,
                    )
                }
            })
            .collect::<Vec<_>>();

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let dataset = Dataset::write(
            data_gen.batch(2),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: 1,
                initial_bases: Some(initial_bases),
                target_base_names_or_paths: Some(vec!["auto".to_string(), "explicit".to_string()]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // The automatic base must not take the explicit ID or primary ID 0,
        // whichever order the caller listed them in.
        let base_paths = &dataset.manifest.base_paths;
        assert_eq!(base_paths.len(), 2, "both initial bases must survive");
        assert!(!base_paths.contains_key(&PRIMARY_BASE_ID));
        assert_eq!(base_paths[&1].path, explicit_uri);
        assert_eq!(base_paths[&1].name.as_deref(), Some("explicit"));
        assert_eq!(base_paths[&2].path, auto_uri);
        assert_eq!(base_paths[&2].name.as_deref(), Some("auto"));

        // Target resolution follows the assigned ids, and the round-robin
        // order follows the requested target order.
        let file_bases = dataset
            .get_fragments()
            .iter()
            .flat_map(|fragment| fragment.metadata().files.iter().map(|file| file.base_id))
            .collect::<Vec<_>>();
        assert_eq!(file_bases, vec![Some(2), Some(1)]);

        let reopened = Dataset::open(&primary_uri).await.unwrap();
        assert_eq!(scan_sorted_ids(&reopened).await, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_initial_base_ids_reject_duplicate_explicit_ids() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        let test_uri = "shared-memory://duplicate_initial_base_ids";
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let error = Dataset::write(
            data_gen.batch(1),
            &format!("{test_uri}/primary"),
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![
                    BasePath::new(
                        1,
                        format!("{test_uri}/base1"),
                        Some("base1".to_string()),
                        true,
                    ),
                    BasePath::new(
                        1,
                        format!("{test_uri}/base2"),
                        Some("base2".to_string()),
                        true,
                    ),
                ]),
                ..Default::default()
            }),
        )
        .await
        .expect_err("duplicate explicit base IDs must be rejected");
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("Duplicate explicit base ID 1"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_initial_base_ids_reject_existing_conflicts_and_reserve_existing_ids() {
        let existing = HashMap::from([(
            1,
            BasePath::new(
                1,
                "memory://existing".to_string(),
                Some("existing".to_string()),
                true,
            ),
        )]);
        let mut params = WriteParams {
            mode: WriteMode::Create,
            initial_bases: Some(vec![BasePath::new(
                1,
                "memory://conflict".to_string(),
                Some("conflict".to_string()),
                true,
            )]),
            ..Default::default()
        };

        let result = validate_and_resolve_target_bases(&mut params, Some(&existing)).await;
        let Err(error) = result else {
            panic!("an explicit ID that already exists must be rejected");
        };
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("already bound to 'memory://existing'"),
            "{error}"
        );

        // Automatic assignment must skip primary ID 0 and existing IDs.
        params.initial_bases = Some(vec![BasePath::new(
            0,
            "memory://auto".to_string(),
            Some("auto".to_string()),
            true,
        )]);
        let result = validate_and_resolve_target_bases(&mut params, Some(&existing))
            .await
            .unwrap();
        assert!(result.is_none());
        assert_eq!(params.initial_bases.as_ref().unwrap()[0].id, 2);

        // Using the highest u32 id must not overflow automatic allocation, and
        // it must not consume a lower unused id.
        let high_existing = HashMap::from([(
            u32::MAX,
            BasePath::new(
                u32::MAX,
                "memory://high".to_string(),
                Some("high".to_string()),
                true,
            ),
        )]);
        params.initial_bases = Some(vec![BasePath::new(
            0,
            "memory://auto".to_string(),
            Some("auto".to_string()),
            true,
        )]);
        let result = validate_and_resolve_target_bases(&mut params, Some(&high_existing))
            .await
            .unwrap();
        assert!(result.is_none());
        assert_eq!(params.initial_bases.as_ref().unwrap()[0].id, 1);
    }

    #[tokio::test]
    async fn test_multi_base_write_read_with_base_scoped_storage_options() {
        use crate::dataset::builder::DatasetBuilder;
        use lance_core::utils::tempfile::TempStrDir;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        let primary_dir = TempStrDir::default();
        let base1_dir = TempStrDir::default();

        // Local stores ignore these options; this verifies base-scoped entries
        // flow through the full write/read path without breaking anything.
        let scoped_options = HashMap::from([
            ("shared_option".to_string(), "shared".to_string()),
            (
                "base_1.scoped_option".to_string(),
                "base1-value".to_string(),
            ),
        ]);
        let store_params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                scoped_options.clone(),
            ))),
            ..Default::default()
        };

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let dataset = Dataset::write(
            data_gen.batch(5),
            primary_dir.as_str(),
            Some(WriteParams {
                mode: WriteMode::Create,
                store_params: Some(store_params),
                initial_bases: Some(vec![BasePath {
                    id: 1,
                    name: Some("base1".to_string()),
                    path: base1_dir.as_str().to_string(),
                    is_dataset_root: true,
                }]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);
        for fragment in dataset.get_fragments() {
            assert!(
                fragment
                    .metadata()
                    .files
                    .iter()
                    .all(|file| file.base_id == Some(1))
            );
        }

        // Reopen with the same flat options and scan through the base store.
        let dataset = DatasetBuilder::from_uri(primary_dir.as_str())
            .with_storage_options(scoped_options)
            .load()
            .await
            .unwrap();
        let batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let num_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(num_rows, 5);
    }

    #[tokio::test]
    async fn test_multi_base_overwrite() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create initial dataset
        let test_uri = "shared-memory://overwrite_test";
        let primary_uri = format!("{}/primary", test_uri);
        let base1_uri = format!("{}/base1", test_uri);
        let base2_uri = format!("{}/base2", test_uri);
        let _base3_uri = format!("{}/base3", test_uri);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(3),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: true,
                    },
                ]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);

        // Overwrite - should inherit existing base configuration (base1, base2)
        // Write to base2
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen2.batch(2),
            std::sync::Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                // No initial_bases - inherits existing base_paths
                target_bases: Some(vec![2]), // Write to base2
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Verify data was overwritten
        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);

        // Verify base_paths were inherited (still base1 and base2)
        assert_eq!(dataset.manifest.base_paths.len(), 2);
        assert!(
            dataset
                .manifest
                .base_paths
                .values()
                .any(|bp| bp.name == Some("base1".to_string()))
        );
        assert!(
            dataset
                .manifest
                .base_paths
                .values()
                .any(|bp| bp.name == Some("base2".to_string()))
        );

        // Verify data was written to base2 (ID 2)
        let fragments = dataset.get_fragments();
        assert!(fragments.iter().all(|f| {
            f.metadata()
                .files
                .iter()
                .all(|file| file.base_id == Some(2))
        }));

        let reopened = Dataset::open(&primary_uri).await.unwrap();
        assert_eq!(scan_sorted_ids(&reopened).await, (0..2).collect::<Vec<_>>());

        // Test validation: cannot specify initial_bases in OVERWRITE mode
        let mut data_gen3 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let result = Dataset::write(
            data_gen3.batch(2),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                initial_bases: Some(vec![BasePath {
                    id: 3,
                    name: Some("base3".to_string()),
                    path: _base3_uri.clone(),
                    is_dataset_root: true,
                }]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot register new bases in Overwrite mode")
        );
    }

    #[tokio::test]
    async fn test_multi_base_append() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create initial dataset with multi-base configuration
        let test_uri = "shared-memory://append_test";
        let primary_uri = format!("{}/primary", test_uri);
        let base1_uri = format!("{}/base1", test_uri);
        let base2_uri = format!("{}/base2", test_uri);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(3),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: true,
                    },
                ]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);

        // Append to base1 (same base as initial write)
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen2.batch(2),
            std::sync::Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);

        // Verify base_paths are still registered
        assert_eq!(dataset.manifest.base_paths.len(), 2);

        // Append to base2 (different base)
        let mut data_gen3 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen3.batch(4),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                target_bases: Some(vec![2]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 9);

        // Verify data is distributed across both bases
        let fragments = dataset.get_fragments();
        let has_base1_data = fragments.iter().any(|f| {
            f.metadata()
                .files
                .iter()
                .any(|file| file.base_id == Some(1))
        });
        let has_base2_data = fragments.iter().any(|f| {
            f.metadata()
                .files
                .iter()
                .any(|file| file.base_id == Some(2))
        });

        assert!(has_base1_data, "Should have data in base1");
        assert!(has_base2_data, "Should have data in base2");

        let mut expected: Vec<i32> = (0..3).chain(0..2).chain(0..4).collect();
        expected.sort_unstable();
        let reopened = Dataset::open(&primary_uri).await.unwrap();
        assert_eq!(scan_sorted_ids(&reopened).await, expected);

        // Test validation: cannot specify initial_bases in APPEND mode
        let mut data_gen4 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let base3_uri = format!("{}/base3", test_uri);

        let result = Dataset::write(
            data_gen4.batch(2),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                initial_bases: Some(vec![BasePath {
                    id: 3,
                    name: Some("base3".to_string()),
                    path: base3_uri,
                    is_dataset_root: true,
                }]),
                target_bases: Some(vec![1]),
                ..Default::default()
            }),
        )
        .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot register new bases in Append mode")
        );
    }

    #[tokio::test]
    async fn test_multi_base_is_dataset_root_flag() {
        use lance_core::utils::tempfile::TempDir;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create dataset with different is_dataset_root settings using tempdir
        let test_dir = TempDir::default();
        let primary_uri = test_dir.path_str();
        let base1_dir = test_dir.std_path().join("base1");
        let base2_dir = test_dir.std_path().join("base2");

        std::fs::create_dir_all(&base1_dir).unwrap();
        std::fs::create_dir_all(&base2_dir).unwrap();

        let base1_uri = format!("file://{}", base1_dir.display());
        let base2_uri = format!("file://{}", base2_dir.display());

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen.batch(10),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: 5, // Create multiple fragments
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true, // Files will go to base1/data/
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: false, // Files will go directly to base2/
                    },
                ]),
                target_bases: Some(vec![1, 2]), // Write to both bases
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 10);

        // Verify base_paths configuration
        assert_eq!(dataset.manifest.base_paths.len(), 2);

        let base1 = dataset
            .manifest
            .base_paths
            .values()
            .find(|bp| bp.name == Some("base1".to_string()))
            .expect("base1 not found");
        let base2 = dataset
            .manifest
            .base_paths
            .values()
            .find(|bp| bp.name == Some("base2".to_string()))
            .expect("base2 not found");

        // Verify is_dataset_root flags are persisted correctly in manifest
        assert!(
            base1.is_dataset_root,
            "base1 should have is_dataset_root=true"
        );
        assert!(
            !base2.is_dataset_root,
            "base2 should have is_dataset_root=false"
        );

        // Verify data was written to both bases
        let fragments = dataset.get_fragments();
        assert!(!fragments.is_empty());

        let has_base1_data = fragments.iter().any(|f| {
            f.metadata()
                .files
                .iter()
                .any(|file| file.base_id == Some(1))
        });
        let has_base2_data = fragments.iter().any(|f| {
            f.metadata()
                .files
                .iter()
                .any(|file| file.base_id == Some(2))
        });

        assert!(has_base1_data, "Should have data in base1");
        assert!(has_base2_data, "Should have data in base2");

        // Verify actual file paths on disk
        // For base1 (is_dataset_root=true), files should be in base1/data/
        let base1_data_dir = base1_dir.join("data");
        assert!(base1_data_dir.exists(), "base1/data directory should exist");
        let base1_files: Vec<_> = std::fs::read_dir(&base1_data_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "lance")
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            !base1_files.is_empty(),
            "base1/data should contain .lance files"
        );

        // For base2 (is_dataset_root=false), files should be directly in base2/
        let base2_files: Vec<_> = std::fs::read_dir(&base2_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "lance")
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            !base2_files.is_empty(),
            "base2 should contain .lance files directly"
        );

        // Verify base2 does NOT have a data subdirectory with lance files
        let base2_data_dir = base2_dir.join("data");
        if base2_data_dir.exists() {
            let base2_data_files: Vec<_> = std::fs::read_dir(&base2_data_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "lance")
                        .unwrap_or(false)
                })
                .collect();
            assert!(
                base2_data_files.is_empty(),
                "base2/data should NOT contain .lance files"
            );
        }
    }

    #[tokio::test]
    async fn test_multi_base_target_by_path_uri() {
        use lance_core::utils::tempfile::TempDir;
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        // Create dataset with named bases
        let test_dir = TempDir::default();
        let primary_uri = test_dir.path_str();
        let base1_dir = test_dir.std_path().join("base1");
        let base2_dir = test_dir.std_path().join("base2");

        std::fs::create_dir_all(&base1_dir).unwrap();
        std::fs::create_dir_all(&base2_dir).unwrap();

        let base1_uri = format!("file://{}", base1_dir.display());
        let base2_uri = format!("file://{}", base2_dir.display());

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        // Create initial dataset writing to base1 using name
        let dataset = Dataset::write(
            data_gen.batch(10),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: 5,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        path: base1_uri.clone(),
                        is_dataset_root: true,
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        path: base2_uri.clone(),
                        is_dataset_root: true,
                    },
                ]),
                target_base_names_or_paths: Some(vec!["base1".to_string()]), // Use name
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 10);

        // Verify data was written to base1
        let fragments = dataset.get_fragments();
        assert!(fragments.iter().all(|f| {
            f.metadata()
                .files
                .iter()
                .all(|file| file.base_id == Some(1))
        }));

        // Now append using the path URI instead of name
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        let dataset = Dataset::write(
            data_gen2.batch(5),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                // Use the actual path URI instead of the name
                target_base_names_or_paths: Some(vec![base2_uri.clone()]),
                max_rows_per_file: 5,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 15);

        // Verify data is now in both bases
        let fragments = dataset.get_fragments();
        let has_base1_data = fragments.iter().any(|f| {
            f.metadata()
                .files
                .iter()
                .any(|file| file.base_id == Some(1))
        });
        let has_base2_data = fragments.iter().any(|f| {
            f.metadata()
                .files
                .iter()
                .any(|file| file.base_id == Some(2))
        });

        assert!(has_base1_data, "Should have data in base1");
        assert!(has_base2_data, "Should have data in base2");

        // Verify base2 has exactly 1 fragment (from the append)
        let base2_fragments: Vec<_> = fragments
            .iter()
            .filter(|f| {
                f.metadata()
                    .files
                    .iter()
                    .all(|file| file.base_id == Some(2))
            })
            .collect();
        assert_eq!(base2_fragments.len(), 1, "Should have 1 fragment in base2");
    }

    #[tokio::test]
    async fn test_empty_stream_write() {
        use lance_io::object_store::ObjectStore;

        // Test writing an empty stream
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        // Create an empty stream
        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema.clone(),
            futures::stream::iter(std::iter::empty::<
                std::result::Result<RecordBatch, DataFusionError>,
            >()),
        ));

        let object_store = Arc::new(ObjectStore::memory());
        let write_params = WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        };

        let result = write_fragments_internal(
            write_params.storage_version_or_default(),
            None,
            object_store,
            &Path::from("test_empty"),
            schema,
            data_stream,
            write_params,
            None,
        )
        .await;

        // Empty stream should be handled gracefully
        // It should create an empty dataset or return an appropriate result
        match result {
            Ok((fragments, _)) => {
                // If successful, verify it creates an empty result
                assert!(
                    fragments.is_empty(),
                    "Empty stream should create no fragments"
                );
            }
            Err(e) => {
                panic!("Expected write empty stream success, got error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_schema_mismatch_on_append() {
        use arrow_array::record_batch;

        // Create initial dataset with two Int32 columns
        let batch1 = record_batch!(
            ("id", Int32, [1, 2, 3, 4, 5]),
            ("value", Int32, [10, 20, 30, 40, 50])
        )
        .unwrap();

        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            })
            .execute(vec![batch1])
            .await
            .unwrap();

        // Verify initial dataset
        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);
        assert_eq!(dataset.schema().fields.len(), 2);

        // Try to append with different schema (Float64 instead of Int32 for 'value' column)
        let batch2 = record_batch!(
            ("id", Int32, [6, 7, 8]),
            ("value", Float64, [60.0, 70.0, 80.0])
        )
        .unwrap();

        let result = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![batch2])
            .await;

        // Should fail due to schema mismatch
        assert!(result.is_err(), "Append with mismatched schema should fail");
        let error = result.unwrap_err();
        let error_msg = error.to_string().to_lowercase();
        assert!(
            error_msg.contains("schema")
                || error_msg.contains("type")
                || error_msg.contains("mismatch")
                || error_msg.contains("field")
                || error_msg.contains("not found"),
            "Error should mention schema or type mismatch: {}",
            error_msg
        );

        // Verify original dataset is still intact
        assert_eq!(dataset.count_rows(None).await.unwrap(), 5);
        assert_eq!(dataset.schema().fields.len(), 2);
    }

    #[tokio::test]
    async fn test_disk_full_error() {
        use std::io::{self, ErrorKind};
        use std::sync::Arc;

        use async_trait::async_trait;
        use futures::stream::BoxStream;
        use object_store::{
            CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
            PutMultipartOptions, PutOptions, PutPayload, PutResult,
        };

        // Create a custom ObjectStore that simulates disk full error
        #[derive(Debug)]
        struct DiskFullObjectStore;

        impl std::fmt::Display for DiskFullObjectStore {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "DiskFullObjectStore")
            }
        }

        #[async_trait]
        impl object_store::ObjectStore for DiskFullObjectStore {
            async fn put_opts(
                &self,
                _location: &object_store::path::Path,
                _bytes: PutPayload,
                _opts: PutOptions,
            ) -> object_store::Result<PutResult> {
                Err(object_store::Error::Generic {
                    store: "DiskFullStore",
                    source: Box::new(io::Error::new(
                        ErrorKind::StorageFull,
                        "No space left on device",
                    )),
                })
            }

            async fn put_multipart_opts(
                &self,
                _location: &object_store::path::Path,
                _opts: PutMultipartOptions,
            ) -> object_store::Result<Box<dyn MultipartUpload>> {
                Err(object_store::Error::NotSupported {
                    source: "Multipart upload not supported".into(),
                })
            }

            async fn get_opts(
                &self,
                _location: &object_store::path::Path,
                _options: GetOptions,
            ) -> object_store::Result<GetResult> {
                Err(object_store::Error::NotFound {
                    path: "".into(),
                    source: "".into(),
                })
            }

            fn delete_stream(
                &self,
                locations: BoxStream<'static, object_store::Result<object_store::path::Path>>,
            ) -> BoxStream<'static, object_store::Result<object_store::path::Path>> {
                locations
            }

            fn list(
                &self,
                _prefix: Option<&object_store::path::Path>,
            ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectMeta>> {
                Box::pin(futures::stream::empty())
            }

            async fn list_with_delimiter(
                &self,
                _prefix: Option<&object_store::path::Path>,
            ) -> object_store::Result<ListResult> {
                Ok(ListResult {
                    common_prefixes: vec![],
                    objects: vec![],
                    extensions: Default::default(),
                })
            }

            async fn copy_opts(
                &self,
                _from: &object_store::path::Path,
                _to: &object_store::path::Path,
                _options: CopyOptions,
            ) -> object_store::Result<()> {
                Ok(())
            }
        }

        let object_store = Arc::new(lance_io::object_store::ObjectStore::new(
            Arc::new(DiskFullObjectStore) as Arc<dyn object_store::ObjectStore>,
            // Use a non-"file" scheme so writes go through ObjectWriter (which
            // uses the DiskFullObjectStore) instead of the optimized LocalWriter.
            url::Url::parse("mock:///test").unwrap(),
            None,
            None,
            false,
            true,
            lance_io::object_store::DEFAULT_LOCAL_IO_PARALLELISM,
            lance_io::object_store::DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));

        // Create test data
        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
        )
        .unwrap();

        let data_reader = Box::new(RecordBatchIterator::new(
            vec![Ok(batch)].into_iter(),
            arrow_schema.clone(),
        ));

        let data_stream = Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            futures::stream::iter(data_reader.map(|rb| rb.map_err(DataFusionError::from))),
        ));

        let schema = Schema::try_from(data_stream.schema().as_ref()).unwrap();

        let write_params = WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        };

        // Attempt to write data - should fail with IO error due to disk full
        let result = write_fragments_internal(
            write_params.storage_version_or_default(),
            None,
            object_store,
            &Path::from("test_disk_full"),
            schema,
            data_stream,
            write_params,
            None,
        )
        .await;

        // Verify that the error is an IO error (which wraps the disk full error)
        assert!(result.is_err(), "Write should fail when disk is full");
        let error = result.unwrap_err();
        let error_msg = error.to_string().to_lowercase();

        // The error should mention IO, space, or storage
        assert!(
            error_msg.contains("io")
                || error_msg.contains("space")
                || error_msg.contains("storage")
                || error_msg.contains("full"),
            "Error should mention IO, space, or storage: {}",
            error_msg
        );

        // Verify it's an IO error type
        assert!(
            matches!(error, lance_core::Error::IO { .. }),
            "Expected IO error, got: {}",
            error
        );
    }

    /// Test that dataset remains consistent after write interruption and can recover.
    /// This verifies that:
    /// 1. The dataset is not corrupted when a write is interrupted (not committed)
    /// 2. Incomplete data files are not visible until committed
    /// 3. The transaction can be retried successfully
    #[tokio::test]
    async fn test_write_interruption_recovery() {
        use super::commit::CommitBuilder;
        use arrow_array::record_batch;
        use lance_core::utils::tempfile::TempDir;

        // Create a temporary directory for testing
        let temp_dir = TempDir::default();
        let dataset_uri = format!("file://{}", temp_dir.std_path().display());

        // First, create a normal dataset with some initial data
        let batch =
            record_batch!(("id", Int32, [1, 2, 3]), ("value", Utf8, ["a", "b", "c"])).unwrap();

        // Write initial dataset normally
        let dataset = InsertBuilder::new(&dataset_uri)
            .execute(vec![batch.clone()])
            .await
            .unwrap();

        // Verify initial dataset is valid
        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);

        // Prepare additional data to write
        let new_batch =
            record_batch!(("id", Int32, [4, 5, 6]), ("value", Utf8, ["d", "e", "f"])).unwrap();

        // Step 1: Write uncommitted data (simulates interrupted write before commit)
        let uncommitted_result = InsertBuilder::new(WriteDestination::Dataset(Arc::new(
            Dataset::open(&dataset_uri).await.unwrap(),
        )))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![new_batch])
        .await;

        // The uncommitted write should succeed (data is written to files)
        assert!(
            uncommitted_result.is_ok(),
            "Uncommitted write should succeed"
        );
        let transaction = uncommitted_result.unwrap();

        // Step 2: Verify dataset is still consistent (uncommitted changes not visible)
        let dataset_before_commit = Dataset::open(&dataset_uri).await.unwrap();
        let row_count_before = dataset_before_commit.count_rows(None).await.unwrap();
        assert_eq!(
            row_count_before, 3,
            "Dataset should still have only original 3 rows (uncommitted data not visible)"
        );

        // Step 3: Commit to transaction (simulates retry after interruption)
        let commit_result = CommitBuilder::new(&dataset_uri).execute(transaction).await;
        commit_result.unwrap();

        // Step 4: Verify dataset now has all 6 rows after successful commit
        let dataset_after_commit = Dataset::open(&dataset_uri).await.unwrap();
        let row_count_after = dataset_after_commit.count_rows(None).await.unwrap();
        assert_eq!(
            row_count_after, 6,
            "Dataset should have all 6 rows after commit"
        );

        // Verify data integrity
        let mut scanner = dataset_after_commit.scan();
        scanner.project(&["id", "value"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let all_ids: Vec<i32> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .iter()
                    .flatten()
            })
            .collect();

        assert_eq!(
            all_ids,
            vec![1, 2, 3, 4, 5, 6],
            "All data should be correctly written"
        );
    }

    /// Returns the number of files in `<base_dir>/data/`.
    fn count_data_files(base_dir: &str) -> usize {
        let data_dir = std::path::Path::new(base_dir).join("data");
        if !data_dir.exists() {
            return 0;
        }
        std::fs::read_dir(data_dir)
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().is_file())
            .count()
    }

    #[tokio::test]
    async fn test_cleanup_data_files_on_failed_write() {
        use lance_core::utils::tempfile::TempStrDir;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let (object_store, base_dir) =
            ObjectStore::from_uri_and_params(Default::default(), test_uri, &Default::default())
                .await
                .unwrap();

        let good_batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        // Build a stream: one good batch, then an error.
        let items: Vec<Result<RecordBatch>> = vec![
            Ok(good_batch.clone()),
            Err(Error::External {
                source: "injected failure".into(),
            }),
        ];
        let stream = Box::pin(LanceRecordBatchStreamAdapter::new(
            arrow_schema.clone(),
            futures::stream::iter(items),
        ));

        let result = lance_table_runtime::write::write_fragments(
            ConcreteFileVersion::V2_1,
            None,
            object_store.clone(),
            &base_dir,
            schema,
            stream,
            lance_table_runtime::WriteParams::default(),
            FragmentWriteOptions::default(),
        )
        .await;

        assert!(result.is_err(), "Expected write to fail");
        assert_eq!(
            count_data_files(test_uri),
            0,
            "All partial data files should be cleaned up on failure"
        );
    }

    #[tokio::test]
    async fn test_cleanup_data_files_on_failed_write_multi_file() {
        // Verify cleanup when a failure occurs after one file has already been completed
        // (i.e., max_rows_per_file causes a file boundary before the error).
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let progress = Arc::new(RecordingProgress::default());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let (object_store, base_dir) =
            ObjectStore::from_uri_and_params(Default::default(), test_uri, &Default::default())
                .await
                .unwrap();

        // 3 rows per file; the first batch fills one file, the second leaves the
        // next file open when the stream fails.
        let good_batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let items: Vec<Result<RecordBatch>> = vec![
            Ok(good_batch.clone()),
            Ok(good_batch.slice(0, 1)),
            Err(Error::External {
                source: "injected failure".into(),
            }),
        ];
        let stream = Box::pin(LanceRecordBatchStreamAdapter::new(
            arrow_schema.clone(),
            futures::stream::iter(items),
        ));

        let result = lance_table_runtime::write::write_fragments(
            ConcreteFileVersion::V2_1,
            None,
            object_store.clone(),
            &base_dir,
            schema,
            stream,
            lance_table_runtime::WriteParams {
                max_rows_per_file: 3,
                progress: progress.clone(),
                ..Default::default()
            },
            FragmentWriteOptions::default(),
        )
        .await;

        let error = result.unwrap_err();
        assert!(
            matches!(error, Error::External { .. }),
            "unexpected error: {error}"
        );
        assert!(
            error.to_string().contains("injected failure"),
            "unexpected error: {error}"
        );
        // The closed file is reported, the one open at the failure is not.
        let calls = progress.calls.lock().unwrap();
        assert!(
            matches!(
                calls.as_slice(),
                [HookCall::Begin(_), HookCall::Complete(closed), HookCall::Begin(_)]
                    if closed.physical_rows == Some(3)
            ),
            "unexpected calls: {calls:?}"
        );
        assert_eq!(
            count_data_files(test_uri),
            0,
            "All data files (including completed ones) should be cleaned up on failure"
        );
    }

    #[derive(Debug, PartialEq)]
    enum HookCall {
        Begin(Fragment),
        Complete(Fragment),
    }

    /// Records every hook call with the fragment it saw, and fails the `complete`
    /// call numbered `fail_complete_at`, counted from 1.
    #[derive(Debug, Default)]
    struct RecordingProgress {
        calls: Mutex<Vec<HookCall>>,
        fail_complete_at: Option<usize>,
    }

    #[async_trait]
    impl WriteFragmentProgress for RecordingProgress {
        async fn begin(&self, fragment: &Fragment) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(HookCall::Begin(fragment.clone()));
            Ok(())
        }

        async fn complete(&self, fragment: &Fragment) -> Result<()> {
            let mut calls = self.calls.lock().unwrap();
            calls.push(HookCall::Complete(fragment.clone()));
            let completed = calls
                .iter()
                .filter(|call| matches!(call, HookCall::Complete(_)))
                .count();
            if self.fail_complete_at == Some(completed) {
                return Err(Error::invalid_input(format!(
                    "injected failure of complete call {completed}"
                )));
            }
            Ok(())
        }
    }

    fn id_batch(num_rows: u64) -> RecordBatch {
        gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_batch_rows(RowCount::from(num_rows))
            .unwrap()
    }

    async fn write_id_batches(
        test_uri: &str,
        items: Vec<Result<RecordBatch>>,
        params: WriteParams,
    ) -> Result<Vec<Fragment>> {
        let arrow_schema = id_batch(0).schema();
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
        let (object_store, base_dir) =
            ObjectStore::from_uri_and_params(Default::default(), test_uri, &Default::default())
                .await
                .unwrap();
        let stream = Box::pin(LanceRecordBatchStreamAdapter::new(
            arrow_schema,
            futures::stream::iter(items),
        ));

        lance_table_runtime::write::write_fragments(
            ConcreteFileVersion::V2_1,
            None,
            object_store,
            &base_dir,
            schema,
            stream,
            into_runtime_write_params(params),
            FragmentWriteOptions::default(),
        )
        .await
        .map(|(fragments, _)| fragments)
    }

    #[rstest]
    #[case::exact_multiple(10, vec![5, 5])]
    #[case::last_file_under_limit(7, vec![5, 2])]
    #[case::single_file_under_limit(3, vec![3])]
    #[case::empty_batch(0, vec![])]
    #[tokio::test]
    async fn test_progress_completes_every_file(
        #[case] num_rows: u64,
        #[case] expected_file_rows: Vec<usize>,
    ) {
        let test_dir = TempStrDir::default();
        let progress = Arc::new(RecordingProgress::default());
        let params = WriteParams {
            max_rows_per_file: 5,
            progress: progress.clone(),
            ..Default::default()
        };

        let fragments = write_id_batches(test_dir.as_str(), vec![Ok(id_batch(num_rows))], params)
            .await
            .unwrap();

        let written_rows = fragments
            .iter()
            .map(|fragment| fragment.physical_rows.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(written_rows, expected_file_rows);
        assert!(fragments.iter().all(|fragment| fragment.files.len() == 1));
        assert_eq!(
            count_data_files(test_dir.as_str()),
            expected_file_rows.len()
        );
        // A file completes before the next one begins, and only `complete` sees it.
        let expected_calls = fragments
            .iter()
            .flat_map(|fragment| {
                [
                    HookCall::Begin(Fragment::new(0)),
                    HookCall::Complete(fragment.clone()),
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(*progress.calls.lock().unwrap(), expected_calls);
    }

    #[tokio::test]
    async fn test_cleanup_data_files_on_failed_complete_of_last_file() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let progress = Arc::new(RecordingProgress {
            fail_complete_at: Some(2),
            ..Default::default()
        });
        let files_reported = Arc::new(AtomicU32::new(0));
        let params = WriteParams {
            max_rows_per_file: 5,
            progress: progress.clone(),
            write_progress: Some(WriteProgressFn::new({
                let files_reported = files_reported.clone();
                move |stats| files_reported.store(stats.files_written, Ordering::SeqCst)
            })),
            ..Default::default()
        };

        let error = write_id_batches(test_uri, vec![Ok(id_batch(7))], params)
            .await
            .unwrap_err();

        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("injected failure of complete call 2"),
            "unexpected error: {error}"
        );
        // Two rows are under the row limit, so the failing call is for the file
        // closed after the write loop.
        let calls = progress.calls.lock().unwrap();
        assert!(
            matches!(
                calls.as_slice(),
                [
                    HookCall::Begin(_),
                    HookCall::Complete(closed),
                    HookCall::Begin(_),
                    HookCall::Complete(last),
                ] if closed.physical_rows == Some(5) && last.physical_rows == Some(2)
            ),
            "unexpected calls: {calls:?}"
        );
        assert_eq!(files_reported.load(Ordering::SeqCst), 1);
        assert_eq!(
            count_data_files(test_uri),
            0,
            "All data files should be cleaned up when the hook fails"
        );
    }

    /// Verifies the external-base branch in `cleanup_data_fragments`: files with
    /// `base_id == Some(_)` are skipped (logged but not deleted via the dataset's
    /// object store), while same-fragment files with `base_id == None` are deleted.
    #[tokio::test]
    async fn test_cleanup_data_fragments_skips_external_base() {
        use lance_core::utils::tempfile::TempStrDir;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (object_store, base_dir) =
            ObjectStore::from_uri_and_params(Default::default(), test_uri, &Default::default())
                .await
                .unwrap();

        // Create a real local data file we expect to be cleaned up.
        let data_dir = base_dir.clone().join(DATA_DIR);
        let local_filename = "local.lance";
        let local_path = data_dir.clone().join(local_filename);
        object_store.put(&local_path, b"x").await.unwrap();
        // Sanity check: file is on disk.
        assert_eq!(count_data_files(test_uri), 1);

        let mut external_file =
            DataFile::new_unstarted("external.lance", ConcreteFileVersion::V2_1);
        external_file.base_id = Some(42);
        let local_file = DataFile::new_unstarted(local_filename, ConcreteFileVersion::V2_1);
        let fragments = vec![Fragment {
            id: 0,
            files: vec![external_file, local_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(0),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        }];

        cleanup_data_fragments(&object_store, &base_dir, None, &fragments).await;

        // The local file should be removed; the external file is skipped without
        // erroring (its base store isn't known here).
        assert_eq!(
            count_data_files(test_uri),
            0,
            "Local data file should be deleted by cleanup"
        );
    }

    /// Verifies the target-base branch in `cleanup_data_fragments`: files whose
    /// `base_id` matches a provided [`TargetBaseInfo`] are deleted via that base's
    /// object store (respecting `is_dataset_root` layout), while files in bases
    /// without a provided store are still skipped.
    #[tokio::test]
    async fn test_cleanup_data_fragments_deletes_target_base_files() {
        use lance_core::utils::tempfile::TempStrDir;

        let primary_dir = TempStrDir::default();
        let base1_dir = TempStrDir::default();
        let base2_dir = TempStrDir::default();

        let (object_store, base_dir) = ObjectStore::from_uri_and_params(
            Default::default(),
            primary_dir.as_str(),
            &Default::default(),
        )
        .await
        .unwrap();
        let (base1_store, base1_path) = ObjectStore::from_uri_and_params(
            Default::default(),
            base1_dir.as_str(),
            &Default::default(),
        )
        .await
        .unwrap();
        let (base2_store, base2_path) = ObjectStore::from_uri_and_params(
            Default::default(),
            base2_dir.as_str(),
            &Default::default(),
        )
        .await
        .unwrap();

        // base2 is a plain data directory: files sit at its root, not under data/.
        let count_plain_files = |dir: &str| {
            std::fs::read_dir(dir)
                .map(|entries| {
                    entries
                        .filter(|e| e.as_ref().unwrap().path().is_file())
                        .count()
                })
                .unwrap_or(0)
        };

        // base1 is a dataset root (files under data/), base2 is a plain data dir.
        let base1_file_path = base1_path.clone().join(DATA_DIR).join("one.lance");
        base1_store.put(&base1_file_path, b"x").await.unwrap();
        let base2_file_path = base2_path.clone().join("two.lance");
        base2_store.put(&base2_file_path, b"x").await.unwrap();
        assert_eq!(count_data_files(base1_dir.as_str()), 1);
        assert_eq!(count_plain_files(base2_dir.as_str()), 1);

        let mut base1_file = DataFile::new_unstarted("one.lance", ConcreteFileVersion::V2_1);
        base1_file.base_id = Some(1);
        let mut base2_file = DataFile::new_unstarted("two.lance", ConcreteFileVersion::V2_1);
        base2_file.base_id = Some(2);
        let mut unknown_file = DataFile::new_unstarted("unknown.lance", ConcreteFileVersion::V2_1);
        unknown_file.base_id = Some(42);
        let fragments = vec![Fragment {
            id: 0,
            files: vec![base1_file, base2_file, unknown_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(0),
            created_at_version_meta: None,
            last_updated_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        }];

        let target_bases = vec![
            TargetBaseInfo {
                base_id: 1,
                object_store: base1_store,
                base_dir: base1_path,
                is_dataset_root: true,
            },
            TargetBaseInfo {
                base_id: 2,
                object_store: base2_store,
                base_dir: base2_path,
                is_dataset_root: false,
            },
        ];

        cleanup_data_fragments(&object_store, &base_dir, Some(&target_bases), &fragments).await;

        assert_eq!(
            count_data_files(base1_dir.as_str()),
            0,
            "File in dataset-root target base should be deleted"
        );
        assert_eq!(
            count_plain_files(base2_dir.as_str()),
            0,
            "File in plain-directory target base should be deleted"
        );
    }

    #[rstest]
    #[case::stream_error(3, None)]
    #[case::complete_of_last_file(2, Some(2))]
    #[tokio::test]
    async fn test_cleanup_routed_data_files_on_failed_write(
        #[case] second_batch_rows: usize,
        #[case] fail_complete_at: Option<usize>,
    ) {
        // Files already completed in target bases must be removed when the
        // write later fails.
        let primary_dir = TempStrDir::default();
        let base1_dir = TempStrDir::default();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();

        let (object_store, base_dir) = ObjectStore::from_uri_and_params(
            Default::default(),
            primary_dir.as_str(),
            &Default::default(),
        )
        .await
        .unwrap();
        let (base1_store, base1_path) = ObjectStore::from_uri_and_params(
            Default::default(),
            base1_dir.as_str(),
            &Default::default(),
        )
        .await
        .unwrap();

        let good_batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        // 3 rows per file: the first batch fills and completes a file in the
        // target base, then the stream fails or the hook fails on the last file.
        let mut items: Vec<Result<RecordBatch>> = vec![
            Ok(good_batch.clone()),
            Ok(good_batch.slice(0, second_batch_rows)),
        ];
        if fail_complete_at.is_none() {
            items.push(Err(Error::External {
                source: "injected failure".into(),
            }));
        }
        let stream = Box::pin(LanceRecordBatchStreamAdapter::new(
            arrow_schema.clone(),
            futures::stream::iter(items),
        ));

        let target_bases = vec![TargetBaseInfo {
            base_id: 1,
            object_store: base1_store,
            base_dir: base1_path,
            is_dataset_root: true,
        }];
        let progress = Arc::new(RecordingProgress {
            fail_complete_at,
            ..Default::default()
        });

        let result = lance_table_runtime::write::write_fragments(
            ConcreteFileVersion::V2_1,
            None,
            object_store.clone(),
            &base_dir,
            schema,
            stream,
            lance_table_runtime::WriteParams {
                max_rows_per_file: 3,
                progress: progress.clone(),
                ..Default::default()
            },
            FragmentWriteOptions {
                target_bases_info: Some(target_bases),
                ..Default::default()
            },
        )
        .await;

        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("injected failure"),
            "unexpected error: {error}"
        );
        // Both files went through the target base before the failure.
        let completed = progress
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter_map(|call| match call {
                HookCall::Complete(fragment) => {
                    Some((fragment.physical_rows, fragment.files[0].base_id))
                }
                HookCall::Begin(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            completed,
            vec![(Some(3), Some(1)), (Some(second_batch_rows), Some(1))]
        );
        assert_eq!(
            count_data_files(base1_dir.as_str()),
            0,
            "Data files routed to the target base should be cleaned up on failure"
        );
        assert_eq!(count_data_files(primary_dir.as_str()), 0);
    }

    /// PRIMARY_BASE_ID (0) and the dataset URI include primary storage in the
    /// target rotation, alongside registered bases.
    #[tokio::test]
    async fn test_multi_base_target_primary_and_bases() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        let test_uri = "shared-memory://primary_slot_test";
        let primary_uri = format!("{}/primary", test_uri);
        let base1_uri = format!("{}/base1", test_uri);
        let base2_uri = format!("{}/base2", test_uri);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));

        // CREATE mode targeting primary + a new base: also verifies the id
        // assignment on initial_bases reaches the committed manifest.
        let dataset = Dataset::write(
            data_gen.batch(6),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: 3,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        is_dataset_root: true,
                        path: base1_uri.clone(),
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        is_dataset_root: false,
                        path: base2_uri.clone(),
                    },
                ]),
                target_bases: Some(vec![PRIMARY_BASE_ID, 1]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.manifest.base_paths.len(), 2);
        let file_bases: Vec<_> = dataset
            .get_fragments()
            .iter()
            .flat_map(|f| f.metadata().files.iter().map(|file| file.base_id))
            .collect();
        assert_eq!(file_bases, vec![None, Some(1)]);

        // APPEND across primary + both bases, one file per slot in order.
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let dataset = Dataset::write(
            data_gen2.batch(9),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 3,
                target_bases: Some(vec![PRIMARY_BASE_ID, 1, 2]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let file_bases: Vec<_> = dataset
            .get_fragments()
            .iter()
            .skip(2)
            .flat_map(|f| f.metadata().files.iter().map(|file| file.base_id))
            .collect();
        assert_eq!(file_bases, vec![None, Some(1), Some(2)]);

        // Names variant: the dataset's own URI selects primary storage.
        let mut data_gen3 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let dataset = Dataset::write(
            data_gen3.batch(6),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: 3,
                target_base_names_or_paths: Some(vec![primary_uri.clone(), "base2".to_string()]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let file_bases: Vec<_> = dataset
            .get_fragments()
            .iter()
            .skip(5)
            .flat_map(|f| f.metadata().files.iter().map(|file| file.base_id))
            .collect();
        assert_eq!(file_bases, vec![None, Some(2)]);

        assert_eq!(dataset.count_rows(None).await.unwrap(), 21);

        let mut expected: Vec<i32> = (0..6).chain(0..9).chain(0..6).collect();
        expected.sort_unstable();
        let reopened = Dataset::open(&primary_uri).await.unwrap();
        assert_eq!(scan_sorted_ids(&reopened).await, expected);
    }

    /// `target_all_bases` resolves to every registered base at execution
    /// time, with primary storage as the first slot when included.
    #[tokio::test]
    async fn test_multi_base_target_all_bases() {
        use lance_testing::datagen::{BatchGenerator, IncrementingInt32};

        let test_uri = "shared-memory://all_bases_test";
        let primary_uri = format!("{}/primary", test_uri);
        let base1_uri = format!("{}/base1", test_uri);
        let base2_uri = format!("{}/base2", test_uri);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let dataset = Dataset::write(
            data_gen.batch(3),
            &primary_uri,
            Some(WriteParams {
                mode: WriteMode::Create,
                initial_bases: Some(vec![
                    BasePath {
                        id: 1,
                        name: Some("base1".to_string()),
                        is_dataset_root: true,
                        path: base1_uri.clone(),
                    },
                    BasePath {
                        id: 2,
                        name: Some("base2".to_string()),
                        is_dataset_root: false,
                        path: base2_uri.clone(),
                    },
                ]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // All bases including primary: slots are [primary, base1, base2].
        let mut data_gen2 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let dataset = Dataset::write(
            data_gen2.batch(9),
            Arc::new(dataset),
            Some(
                WriteParams {
                    mode: WriteMode::Append,
                    max_rows_per_file: 3,
                    ..Default::default()
                }
                .with_target_all_bases(true),
            ),
        )
        .await
        .unwrap();
        let file_bases: Vec<_> = dataset
            .get_fragments()
            .iter()
            .skip(1)
            .flat_map(|f| f.metadata().files.iter().map(|file| file.base_id))
            .collect();
        assert_eq!(file_bases, vec![None, Some(1), Some(2)]);

        // Without primary: slots are [base1, base2].
        let mut data_gen3 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let dataset = Dataset::write(
            data_gen3.batch(6),
            Arc::new(dataset),
            Some(
                WriteParams {
                    mode: WriteMode::Append,
                    max_rows_per_file: 3,
                    ..Default::default()
                }
                .with_target_all_bases(false),
            ),
        )
        .await
        .unwrap();
        let file_bases: Vec<_> = dataset
            .get_fragments()
            .iter()
            .skip(4)
            .flat_map(|f| f.metadata().files.iter().map(|file| file.base_id))
            .collect();
        assert_eq!(file_bases, vec![Some(1), Some(2)]);

        let mut expected: Vec<i32> = (0..3).chain(0..9).chain(0..6).collect();
        expected.sort_unstable();
        let reopened = Dataset::open(&primary_uri).await.unwrap();
        assert_eq!(scan_sorted_ids(&reopened).await, expected);

        // Cannot be combined with explicit target bases.
        let mut data_gen4 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let result = Dataset::write(
            data_gen4.batch(3),
            Arc::new(dataset),
            Some(
                WriteParams {
                    mode: WriteMode::Append,
                    target_bases: Some(vec![1]),
                    ..Default::default()
                }
                .with_target_all_bases(true),
            ),
        )
        .await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot specify target_all_bases together with")
        );

        // On a dataset with no registered bases: include_primary=true is a
        // no-op rotation over primary, false is rejected.
        let plain_uri = "shared-memory://all_bases_plain/primary";
        let mut data_gen5 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let plain = Dataset::write(data_gen5.batch(3), plain_uri, None)
            .await
            .unwrap();
        let mut data_gen6 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let plain = Dataset::write(
            data_gen6.batch(3),
            Arc::new(plain),
            Some(
                WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }
                .with_target_all_bases(true),
            ),
        )
        .await
        .unwrap();
        assert!(
            plain.get_fragments().iter().all(|f| f
                .metadata()
                .files
                .iter()
                .all(|file| file.base_id.is_none()))
        );
        let mut data_gen7 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let result = Dataset::write(
            data_gen7.batch(3),
            Arc::new(plain),
            Some(
                WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }
                .with_target_all_bases(false),
            ),
        )
        .await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("target_all_bases found no registered bases")
        );

        // CREATE mode: initial_bases join the rotation before their ids are
        // committed to a manifest.
        let create_root = "shared-memory://all_bases_create";
        let create_uri = format!("{}/primary", create_root);
        let mut data_gen8 =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("id".to_owned())));
        let dataset = Dataset::write(
            data_gen8.batch(9),
            &create_uri,
            Some(
                WriteParams {
                    mode: WriteMode::Create,
                    max_rows_per_file: 3,
                    initial_bases: Some(vec![
                        // Listed before the explicit ID 1: the automatic base
                        // must not take that reserved ID.
                        BasePath {
                            id: 0,
                            name: Some("base2".to_string()),
                            is_dataset_root: false,
                            path: format!("{}/base2", create_root),
                        },
                        BasePath {
                            id: 1,
                            name: Some("base1".to_string()),
                            is_dataset_root: true,
                            path: format!("{}/base1", create_root),
                        },
                    ]),
                    ..Default::default()
                }
                .with_target_all_bases(true),
            ),
        )
        .await
        .unwrap();
        assert_eq!(dataset.manifest.base_paths.len(), 2);
        assert_eq!(
            dataset.manifest.base_paths[&1].name.as_deref(),
            Some("base1")
        );
        assert_eq!(
            dataset.manifest.base_paths[&2].name.as_deref(),
            Some("base2")
        );
        let file_bases: Vec<_> = dataset
            .get_fragments()
            .iter()
            .flat_map(|f| f.metadata().files.iter().map(|file| file.base_id))
            .collect();
        assert_eq!(file_bases, vec![None, Some(1), Some(2)]);
        let reopened = Dataset::open(&create_uri).await.unwrap();
        assert_eq!(scan_sorted_ids(&reopened).await, (0..9).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn test_file_boundary_keeps_one_page_per_file() {
        use lance_file::reader::FileReaderOptions;
        use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
        use lance_io::utils::CachedFileSize;

        let tmpdir = lance_core::utils::tempfile::TempStrDir::default();
        let values = Arc::new(UInt64Array::from_iter_values(0..4096));
        let batches = (0..12)
            .map(|batch_index| {
                let slice = values.slice(batch_index * 32, 32);
                RecordBatch::try_from_iter([("value", Arc::new(slice) as ArrayRef)]).unwrap()
            })
            .collect::<Vec<_>>();
        let schema = batches[0].schema();
        let reader =
            RecordBatchIterator::new(batches.into_iter().map(Ok).collect::<Vec<_>>(), schema);
        let params = WriteParams {
            max_rows_per_file: 128,
            max_rows_per_group: 32,
            data_storage_version: Some(lance_file::version::LanceFileVersion::V2_0),
            file_writer_options: Some(FileWriterOptions {
                data_cache_bytes: Some(4096),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = Dataset::write(reader, tmpdir.as_str(), Some(params))
            .await
            .unwrap();

        let scheduler = ScanScheduler::new(
            dataset.table_state.object_store.clone(),
            SchedulerConfig::max_bandwidth(&dataset.table_state.object_store),
        );
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 3);
        for fragment in fragments {
            let data_file = fragment.metadata().files.first().unwrap();
            let path = dataset
                .table_state
                .base
                .clone()
                .join(DATA_DIR)
                .join(data_file.path.as_str());
            let file_scheduler = scheduler
                .open_file(&path, &CachedFileSize::unknown())
                .await
                .unwrap();
            let reader = lance_file::reader::FileReader::try_open(
                file_scheduler,
                None,
                Default::default(),
                &dataset
                    .table_state
                    .metadata_cache
                    .file_metadata_cache(&path),
                FileReaderOptions::default(),
            )
            .await
            .unwrap();
            assert_eq!(reader.metadata().column_infos[0].page_infos.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_zone_map_seeds_used_during_update() {
        use crate::Dataset;
        use crate::index::DatasetIndexExt;
        use crate::index::scalar::open_scalar_index;
        use arrow::datatypes::Int32Type;
        use lance_datagen::{BatchCount, RowCount};
        use lance_datagen::{array, gen_batch};
        use lance_file::reader::FileReaderOptions;
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::seed::SEED_META_KEY_PREFIX;
        use lance_index::{IndexType, scalar::ScalarIndexParams};
        use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
        use lance_io::utils::CachedFileSize;

        let tmpdir = lance_core::utils::tempfile::TempStrDir::default();
        let uri = tmpdir.as_str();

        // Step 1: Create initial dataset
        let reader = gen_batch()
            .col("val", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, uri, None).await.unwrap();

        // Step 2: Create a zone map index with seeds explicitly enabled (Int32 defaults to off).
        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::json!({"use_seeds": true}));
        dataset
            .create_index(&["val"], IndexType::ZoneMap, None, &params, false)
            .await
            .unwrap();
        // Step 3: Append new data - seeds should be written automatically
        let reader = gen_batch()
            .col("val", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(50), BatchCount::from(1));
        let dataset = Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(lance_file::version::LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Step 4: Verify that the newly appended fragment has a seed embedded
        let fragments = dataset.fragments();
        let new_fragment = fragments.last().unwrap();
        let data_file = new_fragment.files.first().unwrap();

        let scheduler = ScanScheduler::new(
            dataset.table_state.object_store.clone(),
            SchedulerConfig::max_bandwidth(&dataset.table_state.object_store),
        );
        let path = dataset
            .table_state
            .base
            .clone()
            .join(DATA_DIR)
            .join(data_file.path.as_str());
        let file_scheduler = scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = lance_file::reader::FileReader::try_open(
            file_scheduler,
            None,
            Default::default(),
            &dataset
                .table_state
                .metadata_cache
                .file_metadata_cache(&path),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();

        let meta_key = format!("{}val", SEED_META_KEY_PREFIX);
        let has_seed = reader
            .metadata()
            .file_schema
            .metadata
            .contains_key(&meta_key);
        assert!(
            has_seed,
            "Newly appended fragment should have a zone map seed in metadata"
        );

        // Step 5: Optimize the index (should use seeds)
        let mut dataset = Dataset::open(uri).await.unwrap();
        dataset.optimize_indices(&Default::default()).await.unwrap();

        // Step 6: Query the updated index to verify it's correct
        let dataset = Dataset::open(uri).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert!(
            !indices.is_empty(),
            "Dataset should still have an index after optimization"
        );

        // Verify the index is a ZoneMap and covers all fragments
        let index = indices.iter().find(|i| i.name.contains("val")).unwrap();
        let scalar_index = open_scalar_index(&dataset, "val", index, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            scalar_index.index_type(),
            IndexType::ZoneMap,
            "Index should still be a ZoneMap after optimization"
        );
        let frags = scalar_index.calculate_included_frags().await.unwrap();
        assert_eq!(frags.len(), 2, "Index should cover both fragments");
    }

    /// Counts the `plan_run` execution events emitted while it is the default
    /// subscriber. A seed harvest never executes a plan; the fallback column
    /// scan does, so the count tells which path an index update took.
    #[derive(Clone, Default)]
    struct PlanRunCounter(Arc<std::sync::atomic::AtomicUsize>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for PlanRunCounter {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            use lance_core::utils::tracing::{EXECUTION_PLAN_RUN, TRACE_EXECUTION};

            #[derive(Default)]
            struct TypeField(Option<String>);
            impl tracing::field::Visit for TypeField {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "type" {
                        self.0 = Some(format!("{value:?}").trim_matches('"').to_string());
                    }
                }
            }

            if event.metadata().target() != TRACE_EXECUTION {
                return;
            }
            let mut ty = TypeField::default();
            event.record(&mut ty);
            if ty.0.as_deref() == Some(EXECUTION_PLAN_RUN) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }

    /// Seeds for a nested column record parent-level nulls only, so after an
    /// incremental update the index answers `IS NULL` exactly and agrees with
    /// a plain scan, even when lists carry null or NaN children. With seeds
    /// the update harvests them (no column scan is executed); the same
    /// scenario without seeds is the control that proves the scan is visible.
    #[rstest]
    #[case::seeds(true)]
    #[case::scan(false)]
    #[tokio::test]
    async fn test_zone_map_seeds_for_nested_column_track_parent_nulls(#[case] use_seeds: bool) {
        use crate::Dataset;
        use crate::index::DatasetIndexExt;
        use crate::index::scalar::open_scalar_index;
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
        use arrow_array::{Array, UInt64Array};
        use lance_core::ROW_ADDR;
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::scalar::{SargableQuery, SearchResult};
        use lance_select::RowAddrTreeMap;
        use tracing_subscriber::layer::SubscriberExt;

        const DIM: i32 = 4;
        // Row `i` is a null vector when `null_at(i)`; other rows carry NaN and
        // null children on purpose, which must not count as null rows.
        fn null_at(i: usize) -> bool {
            i % 7 == 3 || (20..26).contains(&i)
        }
        fn vectors(rows: usize) -> RecordBatch {
            let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), DIM);
            for i in 0..rows {
                if null_at(i) {
                    for _ in 0..DIM {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                builder.values().append_value(i as f32);
                builder
                    .values()
                    .append_value(if i % 5 == 0 { f32::NAN } else { 1.0 });
                if i % 4 == 0 {
                    builder.values().append_null();
                } else {
                    builder.values().append_value(2.0);
                }
                builder.values().append_value(3.0);
                builder.append(true);
            }
            let array = builder.finish();
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "vec",
                array.data_type().clone(),
                true,
            )]));
            RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap()
        }
        fn reader(batch: RecordBatch) -> impl RecordBatchReader + Send + 'static {
            let schema = batch.schema();
            RecordBatchIterator::new(vec![Ok(batch)], schema)
        }

        let tmpdir = lance_core::utils::tempfile::TempStrDir::default();
        let uri = tmpdir.as_str();
        let mut dataset = Dataset::write(reader(vectors(40)), uri, None)
            .await
            .unwrap();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::json!({"use_seeds": use_seeds, "rows_per_zone": 8}));
        dataset
            .create_index(&["vec"], IndexType::ZoneMap, None, &params, false)
            .await
            .unwrap();

        // Appending writes a seed for the new fragment when seeds are on.
        Dataset::write(
            reader(vectors(60)),
            uri,
            Some(WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(lance_file::version::LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Count plan executions during the update: a harvest runs none, the
        // fallback scan runs at least one. tracing caches each callsite's
        // interest from the firing thread's default dispatcher while exactly
        // one dispatcher is registered; a second live dispatcher makes it
        // consult every registered dispatcher, including this thread-local one.
        let plan_runs = PlanRunCounter::default();
        let subscriber = tracing_subscriber::registry().with(plan_runs.clone());
        let _second_dispatcher = tracing::Dispatch::new(tracing_subscriber::registry());
        let mut dataset = Dataset::open(uri).await.unwrap();
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            dataset.optimize_indices(&Default::default()).await.unwrap();
        }
        let plan_runs = plan_runs.0.load(std::sync::atomic::Ordering::SeqCst);
        if use_seeds {
            assert_eq!(plan_runs, 0, "seed harvest must not scan the column");
        } else {
            assert!(plan_runs >= 1, "fallback update must scan the column");
        }
        let dataset = Dataset::open(uri).await.unwrap();

        // Ground truth from a scan without the index.
        async fn row_addrs(dataset: &Dataset, filter: &str, use_index: bool) -> Vec<u64> {
            let batches = dataset
                .scan()
                .filter(filter)
                .unwrap()
                .with_row_address()
                .project::<&str>(&[])
                .unwrap()
                .use_scalar_index(use_index)
                .try_into_stream()
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            let mut addrs: Vec<u64> = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column_by_name(ROW_ADDR)
                        .unwrap()
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect();
            addrs.sort_unstable();
            addrs
        }
        let null_rows = row_addrs(&dataset, "vec IS NULL", false).await;
        let expected_nulls =
            (0..40).filter(|i| null_at(*i)).count() + (0..60).filter(|i| null_at(*i)).count();
        assert_eq!(null_rows.len(), expected_nulls);
        assert_eq!(row_addrs(&dataset, "vec IS NULL", true).await, null_rows);
        assert_eq!(
            row_addrs(&dataset, "vec IS NOT NULL", true).await,
            row_addrs(&dataset, "vec IS NOT NULL", false).await
        );

        // The merged index knows the exact null rows of both fragments.
        let indices = dataset.load_indices().await.unwrap();
        let index = indices.iter().find(|i| i.name.contains("vec")).unwrap();
        let scalar_index = open_scalar_index(&dataset, "vec", index, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            scalar_index.calculate_included_frags().await.unwrap().len(),
            2
        );
        let result = scalar_index
            .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
            .await
            .unwrap();
        let expected = RowAddrTreeMap::from_iter(null_rows.iter().copied());
        assert_eq!(result, SearchResult::exact(expected));
    }

    /// Seed observers see the values the data file stores, so appending a
    /// view array to a seeded string column succeeds.
    #[tokio::test]
    async fn test_append_view_array_to_seeded_string_column() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "val",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec!["a", "b"]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::json!({"use_seeds": true}));
        dataset
            .create_index(&["val"], IndexType::ZoneMap, None, &params, false)
            .await
            .unwrap();
        assert!(
            !seeds::collect_write_seeds(
                dataset.manifest.data_storage_format.lance_file_format(),
                &dataset,
                dataset.schema(),
            )
            .await
            .unwrap()
            .is_empty(),
            "the append must observe seeds for this test to cover them"
        );

        let view_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "val",
            DataType::Utf8View,
            false,
        )]));
        let view_batch = RecordBatch::try_new(
            view_schema.clone(),
            vec![Arc::new(StringViewArray::from(vec!["c", "d"]))],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(view_batch)], view_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            dataset
                .count_rows(Some("val = 'c'".to_string()))
                .await
                .unwrap(),
            1
        );
        assert_eq!(dataset.count_rows(None).await.unwrap(), 4);
    }

    #[rstest]
    #[case::unshredded(false)]
    #[case::shredded(true)]
    #[tokio::test]
    async fn test_variant_storage(#[case] is_shredded: bool) {
        let documents = vec![
            None,
            Some("null"),
            Some("{}"),
            Some(
                r#"{"id":42,"name":"a long string stored outside a view","items":[1,null,3],"other":{"keep":true}}"#,
            ),
            Some(r#"{"id":"not an integer","name":false,"items":["mixed",2]}"#),
            Some(r#"{"id":null,"items":[]}"#),
            Some(r#"[1,{"nested":true},null]"#),
            Some("[]"),
            Some("17"),
        ];
        let input: ArrayRef = Arc::new(StringArray::from(documents.clone()));
        let variant = json_to_variant(&input).unwrap();
        let variant = if is_shredded {
            shred_variant(
                &variant,
                &DataType::Struct(
                    vec![
                        ArrowField::new("id", DataType::Int64, true),
                        ArrowField::new("name", DataType::Utf8View, true),
                        ArrowField::new(
                            "items",
                            DataType::List(Arc::new(ArrowField::new(
                                "item",
                                DataType::Int64,
                                true,
                            ))),
                            true,
                        ),
                    ]
                    .into(),
                ),
            )
            .unwrap()
        } else {
            variant
        };
        let field = variant.field("event");
        let schema = Arc::new(ArrowSchema::new(vec![field.clone()]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(variant.into_inner())]).unwrap();
        // The second batch has a nonzero offset, and the first fragment contains
        // only SQL null, Variant null and an empty object.
        let reader =
            RecordBatchIterator::new([Ok(batch.slice(0, 3)), Ok(batch.slice(3, 6))], schema);
        let dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 3,
                max_rows_per_group: 3,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 3);

        let result = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(result.schema().field(0).metadata(), field.metadata());
        let stored = VariantArray::try_new(result["event"].as_ref()).unwrap();
        assert_eq!(stored.metadata_column().data_type(), &DataType::Binary);
        assert_eq!(stored.value_column().data_type(), &DataType::Binary);
        assert!(stored.is_null(0));
        assert!(stored.is_valid(1), "Variant null is not SQL null");

        // Reassemble with the upstream kernel, then compare JSON values so
        // object key order and integer widths do not affect the assertion.
        let unshredded = unshred_variant(&stored).unwrap();
        let binary_type = DataType::Struct(
            vec![
                ArrowField::new("metadata", DataType::Binary, false),
                ArrowField::new("value", DataType::Binary, true),
            ]
            .into(),
        );
        let unshredded = arrow_cast::cast(unshredded.inner(), &binary_type).unwrap();
        let json = variant_to_json(&unshredded).unwrap();
        for (actual, expected) in json.iter().zip(documents) {
            assert_eq!(
                actual.map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap()),
                expected.map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap()),
            );
        }

        if is_shredded {
            let typed_value = stored.typed_value_column().unwrap().as_struct();
            let name = typed_value.column_by_name("name").unwrap().as_struct();
            assert_eq!(
                name.column_by_name("typed_value").unwrap().data_type(),
                &DataType::Utf8
            );
            let projected = dataset
                .scan()
                .project(&["event.typed_value.id.typed_value"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(projected.num_columns(), 1);
            let values = projected["event.typed_value.id.typed_value"]
                .as_primitive::<arrow_array::types::Int64Type>();
            assert_eq!(
                values,
                &arrow_array::Int64Array::from(vec![
                    None,
                    None,
                    None,
                    Some(42),
                    None,
                    None,
                    None,
                    None,
                    None,
                ]),
            );
        }
    }

    /// A covered scalar index must still get a seed writer. The loop skipped any
    /// index with more than one field, silently dropping seed writing for it.
    #[tokio::test]
    async fn test_seed_writers_for_a_covered_index() {
        use crate::dataset::transaction::{Operation, Transaction};
        use lance_core::utils::tempfile::TempStrDir;
        use lance_index::{IndexType, scalar::ScalarIndexParams};

        let test_uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("payload", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..256)),
                Arc::new(Int32Array::from_iter_values(0..256)),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(reader, &test_uri, None).await.unwrap();

        // BTree (the default `ScalarIndexParams`) never writes seeds -- only ZoneMap
        // does, and only when `use_seeds` is explicitly requested for a fixed-width
        // type like Int32 (see `test_zone_map_seeds_used_during_update` above).
        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::json!({"use_seeds": true}));
        dataset
            .create_index(&["id"], IndexType::ZoneMap, None, &params, true)
            .await
            .unwrap();

        let baseline = seeds::collect_write_seeds(
            dataset.manifest.data_storage_format.lance_file_format(),
            &dataset,
            dataset.schema(),
        )
        .await
        .unwrap();
        assert!(
            !baseline.is_empty(),
            "a plain scalar index should produce a seed writer; if this is empty \
             the rest of the test proves nothing"
        );

        // Declare `payload` as carried, then re-check.
        let id_field = dataset.schema().field_id("id").unwrap();
        let payload_field = dataset.schema().field_id("payload").unwrap();
        let current = dataset.load_indices().await.unwrap();
        let mut covered = current[0].clone();
        covered.fields = vec![id_field, payload_field];
        covered.covering_fields = vec![payload_field];

        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![covered],
                removed_indices: current.to_vec(),
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let covered_writers = seeds::collect_write_seeds(
            dataset.manifest.data_storage_format.lance_file_format(),
            &dataset,
            dataset.schema(),
        )
        .await
        .unwrap();
        assert_eq!(
            covered_writers.len(),
            baseline.len(),
            "a covered index must still produce a seed writer"
        );
    }
}
