// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;
use lance_datafusion::utils::StreamingWriteSource;
use lance_table_runtime::InsertBuilder as RuntimeInsertBuilder;

use crate::dataset::transaction::Transaction;
use crate::io::commit::LanceCommitHook;
use crate::session::Session;
use crate::{Dataset, Result};

#[allow(unused_imports)]
use super::CommitBuilder;
use super::seeds::LanceWriteSeedHook;
use super::{DataFusionWriteSource, WriteDestination, WriteParams, WriteProgressFn, WriteStats};

/// Insert or create a new dataset.
///
/// There are different variants of `execute()` methods. Those with the `_stream`
/// suffix take an iterator of data so that larger than memory data can be written
/// out. However, this eliminates optimizations that can be made when the full
/// data is known up-front.
///
/// Those with the `_uncommitted` suffix write the data files but do not commit
/// the transactions. These changes to the dataset will not be visible until
/// they are passed to the [`CommitBuilder`].
#[derive(Debug, Clone)]
pub struct InsertBuilder<'a> {
    dest: WriteDestination<'a>,
    // TODO: make these parameters a part of the builder, and add specific methods.
    params: Option<&'a WriteParams>,
    write_progress: Option<WriteProgressFn>,
    preassigned_data_file_name: Option<Arc<String>>,
    inherited_field_ids: bool,
    disable_dictionary_split: bool,
    observed: bool,
}

impl<'a> InsertBuilder<'a> {
    pub fn new(dest: impl Into<WriteDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            params: None,
            write_progress: None,
            preassigned_data_file_name: None,
            inherited_field_ids: false,
            disable_dictionary_split: false,
            observed: true,
        }
    }

    pub fn with_params(mut self, params: &'a WriteParams) -> Self {
        self.params = Some(params);
        self
    }

    pub(crate) fn with_preassigned_data_file_name(
        mut self,
        preassigned_data_file_name: impl Into<String>,
    ) -> Self {
        self.preassigned_data_file_name = Some(Arc::new(preassigned_data_file_name.into()));
        self
    }

    /// The input's field IDs are an existing table's, so a new table keeps them
    /// even past the limit for caller-supplied IDs. For a MemWAL generation.
    pub(crate) fn with_inherited_field_ids(mut self) -> Self {
        self.inherited_field_ids = true;
        self
    }

    /// Never starts another data file because a dictionary column outgrows its key
    /// type; the encoder's own key-type check applies instead. For a MemWAL generation,
    /// whose deletion vector and indexes address a single fragment.
    pub(crate) fn with_dictionary_split_disabled(mut self) -> Self {
        self.disable_dictionary_split = true;
        self
    }

    /// Observes no write of its own, for a write that is only one step of
    /// another operation, which reports it in a stage of its own.
    pub(crate) fn without_memory_observation(mut self) -> Self {
        self.observed = false;
        self
    }

    /// Register a callback that is invoked after each batch of rows is written.
    ///
    /// The callback receives cumulative [`WriteStats`] and can be used to drive
    /// a progress bar or compute throughput. It must be cheap and non-blocking;
    /// spawn a task if you need async work.
    ///
    /// This overrides any `write_progress` set in [`WriteParams`].
    pub fn progress(mut self, callback: impl Fn(WriteStats) + Send + Sync + 'static) -> Self {
        self.write_progress = Some(WriteProgressFn::new(callback));
        self
    }

    /// Write the data fragments and commit them into the dataset.
    pub async fn execute(&self, data: Vec<RecordBatch>) -> Result<Dataset> {
        let (params, session) = self.runtime_params();
        let table = Box::pin(self.runtime_builder(&params, &session).execute(data)).await?;
        Ok(Dataset::from_committed_table(table, session))
    }

    /// Write the data fragments from a stream and commit them into the dataset.
    pub async fn execute_stream(&self, source: impl StreamingWriteSource) -> Result<Dataset> {
        let (params, session) = self.runtime_params();
        let table = Box::pin(
            self.runtime_builder(&params, &session)
                .execute_stream(DataFusionWriteSource(source)),
        )
        .await?;
        Ok(Dataset::from_committed_table(table, session))
    }

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    ///
    /// # Example: Append data to a dataset
    ///
    /// ```rust
    /// use lance::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
    ///
    /// # use std::sync::Arc;
    /// # use arrow_array::RecordBatch;
    /// # use lance::Result;
    /// # use lance::dataset::Dataset;
    /// # async fn example(dataset: Arc<Dataset>, data: Vec<RecordBatch>) -> Result<()> {
    /// let transaction = InsertBuilder::new(dataset.clone())
    ///     .with_params(&WriteParams { mode: WriteMode::Append, ..Default::default() })
    ///     .execute_uncommitted(data)
    ///     .await?;
    /// CommitBuilder::new(dataset)
    ///     .execute(transaction)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_uncommitted(&self, data: Vec<RecordBatch>) -> Result<Transaction> {
        let (params, session) = self.runtime_params();
        Box::pin(
            self.runtime_builder(&params, &session)
                .execute_uncommitted(data),
        )
        .await
    }

    /// Write data files from a stream without committing the transaction.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    pub async fn execute_uncommitted_stream(
        &self,
        source: impl StreamingWriteSource,
    ) -> Result<Transaction> {
        let (params, session) = self.runtime_params();
        Box::pin(
            self.runtime_builder(&params, &session)
                .execute_uncommitted_stream(DataFusionWriteSource(source)),
        )
        .await
    }

    fn runtime_params(&self) -> (lance_table_runtime::WriteParams, Arc<Session>) {
        let mut params = self.params.cloned().unwrap_or_default();
        if let Some(progress) = &self.write_progress {
            params.write_progress = Some(progress.clone());
        }
        let session = params
            .session
            .clone()
            .or_else(|| self.dest.dataset().map(|dataset| dataset.session.clone()))
            .unwrap_or_default();
        (super::into_runtime_write_params(params), session)
    }

    fn runtime_builder<'b>(
        &'b self,
        params: &'b lance_table_runtime::WriteParams,
        session: &Arc<Session>,
    ) -> RuntimeInsertBuilder<'b> {
        let hook_session = self
            .dest
            .dataset()
            .map(|dataset| dataset.session.clone())
            .unwrap_or_else(|| session.clone());
        let seed_hook = Arc::new(LanceWriteSeedHook::new(hook_session.clone()));
        let commit_hook = Arc::new(LanceCommitHook::new(hook_session));
        let mut builder = RuntimeInsertBuilder::new(self.dest.to_runtime())
            .with_params(params)
            .with_default_session(session.table_session.clone())
            .with_seed_hook(seed_hook)
            .with_index_hook(commit_hook.clone())
            .with_lifecycle_hook(commit_hook);
        if let Some(name) = &self.preassigned_data_file_name {
            builder = builder.with_preassigned_data_file_name(name.as_str());
        }
        if !self.observed {
            builder = builder.without_memory_observation();
        }
        if self.inherited_field_ids {
            builder = builder.with_inherited_field_ids();
        }
        if self.disable_dictionary_split {
            builder = builder.with_dictionary_split_disabled();
        }
        builder
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use arrow_array::{
        ArrayRef, BinaryArray, Int32Array, RecordBatchIterator, RecordBatchReader, StructArray,
        record_batch,
    };
    use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
    use datafusion::physical_plan::{
        SendableRecordBatchStream as DataFusionRecordBatchStream, stream::RecordBatchStreamAdapter,
    };
    use lance_arrow::BLOB_META_KEY;
    use lance_core::{Error, utils::tempfile::TempStrDir};
    use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
    use lance_table::feature_flags::FLAG_MIXED_DATA_FILE_VERSIONS;
    use lance_table::io::commit::{RenameCommitHandler, commit_handler_from_url};
    use lance_table::transaction::Operation;
    use rstest::rstest;

    use crate::dataset::WriteMode;
    use crate::session::Session;

    use super::*;

    #[tokio::test]
    async fn insert_uses_custom_streaming_write_source_schema() {
        struct CustomWriteSource(DataFusionRecordBatchStream);

        #[async_trait::async_trait]
        impl StreamingWriteSource for CustomWriteSource {
            fn arrow_schema(&self) -> SchemaRef {
                self.0.schema()
            }

            fn into_stream(self) -> DataFusionRecordBatchStream {
                self.0
            }

            async fn into_stream_and_schema(
                self,
            ) -> Result<(DataFusionRecordBatchStream, lance_core::datatypes::Schema)> {
                let mut schema =
                    lance_core::datatypes::Schema::try_from(self.arrow_schema().as_ref())?;
                schema
                    .metadata
                    .insert("write_source".into(), "custom".into());
                Ok((self.0, schema))
            }
        }

        let batch = record_batch!(("id", Int32, [1, 2, 3])).unwrap();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            futures::stream::iter([Ok(batch)]),
        ));
        let dataset = InsertBuilder::new("memory://")
            .execute_stream(CustomWriteSource(stream))
            .await
            .unwrap();

        assert_eq!(dataset.schema().metadata["write_source"], "custom");
        assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
    }

    #[rstest]
    #[tokio::test]
    async fn append_resolves_explicit_or_default_exact_version(
        #[values(
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::V2_2,
            LanceFileVersion::V2_3
        )]
        default_version: LanceFileVersion,
        #[values(
            LanceFileVersion::V2_0,
            LanceFileVersion::V2_1,
            LanceFileVersion::V2_2,
            LanceFileVersion::V2_3
        )]
        target_version: LanceFileVersion,
    ) {
        let batch = record_batch!(("id", Int32, [0])).unwrap();
        let create_params = WriteParams {
            data_storage_version: Some(default_version),
            ..Default::default()
        };
        let dataset = InsertBuilder::new("memory://")
            .with_params(&create_params)
            .execute(vec![batch.clone()])
            .await
            .unwrap();

        let default_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&default_params)
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        assert_eq!(
            dataset.manifest.fragments[1].files[0]
                .file_version()
                .unwrap(),
            default_version.resolve()
        );
        assert_eq!(
            dataset.manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );

        let explicit_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(target_version),
            ..Default::default()
        };
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&explicit_params)
            .execute(vec![batch.clone()])
            .await
            .unwrap();

        assert_eq!(
            dataset.manifest.data_storage_format.lance_file_format(),
            default_version.resolve()
        );
        assert_eq!(
            dataset.manifest.fragments[2].files[0]
                .file_version()
                .unwrap(),
            target_version.resolve()
        );
        let expected_flags = if default_version == target_version {
            0
        } else {
            FLAG_MIXED_DATA_FILE_VERSIONS
        };
        assert_eq!(
            dataset.manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            expected_flags
        );
        assert_eq!(
            dataset.manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            expected_flags
        );
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&default_params)
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        assert_eq!(
            dataset.manifest.fragments[3].files[0]
                .file_version()
                .unwrap(),
            default_version.resolve()
        );
        let expected =
            arrow_select::concat::concat_batches(&batch.schema(), [&batch, &batch, &batch, &batch])
                .unwrap();
        let reopened = dataset
            .checkout_version(dataset.version().version)
            .await
            .unwrap();
        assert_eq!(reopened.scan().try_into_batch().await.unwrap(), expected);
    }

    #[rstest]
    #[case::default(None, ConcreteFileVersion::V2_0)]
    #[case::explicit(Some(LanceFileVersion::V2_1), ConcreteFileVersion::V2_1)]
    #[tokio::test]
    async fn overwrite_resolves_exact_version(
        #[case] target: Option<LanceFileVersion>,
        #[case] expected: ConcreteFileVersion,
    ) {
        let batch = record_batch!(("id", Int32, [0])).unwrap();
        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_0),
                ..Default::default()
            })
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Overwrite,
                data_storage_version: target,
                ..Default::default()
            })
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        assert_eq!(
            dataset.manifest.data_storage_format.lance_file_format(),
            expected
        );
        assert_eq!(dataset.manifest.fragments.len(), 1);
        assert_eq!(
            dataset.manifest.fragments[0].files[0]
                .file_version()
                .unwrap(),
            expected
        );
        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), batch);
    }

    #[tokio::test]
    async fn create_from_mixed_prewritten_files_requires_explicit_default() {
        let test_dir = TempStrDir::default();
        let mut transaction = InsertBuilder::new(test_dir.as_str())
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_0),
                ..Default::default()
            })
            .execute_uncommitted(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let other = InsertBuilder::new(test_dir.as_str())
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_uncommitted(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        let Operation::Overwrite { fragments, .. } = &mut transaction.operation else {
            panic!("expected overwrite");
        };
        let Operation::Overwrite {
            fragments: other_fragments,
            ..
        } = other.operation
        else {
            panic!("expected overwrite");
        };
        fragments.extend(other_fragments);
        let error = CommitBuilder::new(test_dir.as_str())
            .execute(transaction.clone())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("All data files must have the same version"),
            "{error}"
        );
        let dataset = CommitBuilder::new(test_dir.as_str())
            .with_storage_format(LanceFileVersion::V2_0)
            .execute(transaction)
            .await
            .unwrap();
        assert_eq!(
            dataset.manifest.data_storage_format.lance_file_format(),
            ConcreteFileVersion::V2_0
        );
        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        assert_eq!(
            dataset.scan().try_into_batch().await.unwrap(),
            record_batch!(("id", Int32, [0, 1])).unwrap()
        );
    }

    #[tokio::test]
    async fn concurrent_exact_version_appends_derive_capability_from_final_manifest() {
        let create_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        };
        let original = Arc::new(
            InsertBuilder::new("memory://")
                .with_params(&create_params)
                .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
                .await
                .unwrap(),
        );
        let v2_1_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };
        let v2_2_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let tx_v2_1 = InsertBuilder::new(original.clone())
            .with_params(&v2_1_params)
            .execute_uncommitted(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        let tx_v2_2 = InsertBuilder::new(original.clone())
            .with_params(&v2_2_params)
            .execute_uncommitted(vec![record_batch!(("id", Int32, [2])).unwrap()])
            .await
            .unwrap();
        let Operation::Append { fragments } = &tx_v2_2.operation else {
            panic!("expected append");
        };
        let prewritten_files = fragments[0].files.clone();
        CommitBuilder::new(original.clone())
            .execute(tx_v2_1)
            .await
            .unwrap();
        // The second writer still holds the original snapshot, forcing a retry.
        let dataset = CommitBuilder::new(original).execute(tx_v2_2).await.unwrap();

        let versions = dataset
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.files[0].file_version().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            versions,
            vec![
                ConcreteFileVersion::V2_0,
                ConcreteFileVersion::V2_1,
                ConcreteFileVersion::V2_2
            ]
        );
        assert_eq!(dataset.manifest.fragments[2].files, prewritten_files);
        assert_eq!(
            dataset.manifest.data_storage_format.lance_file_format(),
            ConcreteFileVersion::V2_0
        );
        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        assert_eq!(
            dataset.scan().try_into_batch().await.unwrap(),
            record_batch!(("id", Int32, [0, 1, 2])).unwrap()
        );
    }

    #[tokio::test]
    async fn test_pass_session() {
        let session = Arc::new(Session::new(0, 0, Default::default()));
        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                session: Some(session.clone()),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![],
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int32, false)])),
            ))
            .await
            .unwrap();

        assert_eq!(Arc::as_ptr(&dataset.session()), Arc::as_ptr(&session));
    }

    #[tokio::test]
    async fn test_write_empty_struct() {
        // Regresses a 2.1 issue where empty structs did not get assigned any columns
        // in the file because we only look at leaf columns.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "empties",
            DataType::Struct(Vec::<Field>::new().into()),
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StructArray::new_empty_fields(1, None))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://")
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await
            .unwrap();

        assert_eq!(
            dataset
                .count_rows(Some("empties IS NOT NULL".to_string()))
                .await
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn dataset_destination_honors_explicit_commit_handler() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let initial_batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let mut dataset = InsertBuilder::new("memory://")
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(initial_batch)],
                schema.clone(),
            ))
            .await
            .unwrap();
        dataset.table_state.commit_handler = commit_handler_from_url("cos://bucket/dataset", &None)
            .await
            .unwrap();

        let append_batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![2]))])
                .unwrap();
        let params = WriteParams {
            mode: WriteMode::Append,
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            ..Default::default()
        };
        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(append_batch)], schema),
            Arc::new(dataset),
            Some(params),
        )
        .await
        .expect("the explicit per-write commit handler should override the dataset handler");

        assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
    }

    #[rstest::rstest]
    #[case::row_id("_rowid")]
    #[case::row_addr("_rowaddr")]
    #[case::row_offset("_rowoffset")]
    #[case::row_created_at_version("_row_created_at_version")]
    #[case::row_last_updated_at_version("_row_last_updated_at_version")]
    #[case::empty_name("")]
    #[tokio::test]
    async fn rejects_unwritable_top_level_column_names(#[case] name: &str) {
        // Every system column name must be rejected on write. The row-version
        // columns (`_row_created_at_version`, `_row_last_updated_at_version`) are
        // computed at read time and appended by `Projection::to_schema`; a user
        // data column sharing one of those names would otherwise pass ingest and
        // later collide with the appended field. An empty top-level name cannot
        // be addressed as a field path after the dataset is written.
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();

        let result = InsertBuilder::new("memory://")
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await;

        let err = result.expect_err("writing an unwritable top-level column name should fail");
        assert!(matches!(&err, crate::Error::InvalidInput { .. }), "{err}");
        let expected_message = if name.is_empty() {
            "empty name"
        } else {
            "reserved name"
        };
        assert!(
            err.to_string().contains(expected_message),
            "unexpected error for {name:?}: {err}"
        );
    }

    #[tokio::test]
    async fn allow_overwrite_to_v2_2_without_blob_upgrade() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();

        let dataset = InsertBuilder::new("memory://blob-version-guard")
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(batch.clone())],
                schema.clone(),
            ))
            .await
            .unwrap();

        let dataset = Arc::new(dataset);
        let params = WriteParams {
            mode: WriteMode::Overwrite,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };

        let result = InsertBuilder::new(dataset.clone())
            .with_params(&params)
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn create_v2_2_dataset_accepts_legacy_blob_input() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("blob", DataType::Binary, false).with_metadata(HashMap::from([(
                BLOB_META_KEY.to_string(),
                "true".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(BinaryArray::from(vec![Some(b"abc".as_slice())]))],
        )
        .unwrap();

        let dataset = InsertBuilder::new("memory://forced-blob-v2")
            .with_params(&WriteParams {
                mode: WriteMode::Create,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await;

        let dataset = Arc::new(dataset.unwrap());
        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(
            blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"abc"
        );
        assert!(dataset.schema().field("blob").unwrap().is_blob_v2());
    }

    #[rstest]
    #[case::create(None)]
    #[case::append_v20(Some(LanceFileVersion::V2_0))]
    #[case::append_v21(Some(LanceFileVersion::V2_1))]
    #[tokio::test]
    async fn create_v2_2_dataset_accepts_nested_legacy_blob_input(
        #[case] initial_version: Option<LanceFileVersion>,
    ) {
        let image_field =
            Field::new("image_bytes", DataType::LargeBinary, true).with_metadata(HashMap::from([
                (BLOB_META_KEY.to_string(), "true".to_string()),
            ]));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "summary_image_nested",
            DataType::Struct(vec![image_field.clone()].into()),
            true,
        )]));
        let image_values: ArrayRef = Arc::new(arrow_array::LargeBinaryArray::from(vec![Some(
            b"abc".as_slice(),
        )]));
        let nested_values = StructArray::from(vec![(Arc::new(image_field), image_values)]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(nested_values)]).unwrap();

        let dir = TempStrDir::default();
        let uri = dir.as_str();
        if let Some(version) = initial_version {
            InsertBuilder::new(uri)
                .with_params(&WriteParams {
                    data_storage_version: Some(version),
                    ..Default::default()
                })
                .execute_stream(RecordBatchIterator::new(
                    [Ok(batch.clone())],
                    schema.clone(),
                ))
                .await
                .unwrap();
        }
        let dataset = InsertBuilder::new(uri)
            .with_params(&WriteParams {
                mode: if initial_version.is_some() {
                    WriteMode::Append
                } else {
                    WriteMode::Create
                },
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await;

        let dataset = Arc::new(dataset.unwrap());
        let blobs = dataset
            .take_blobs_by_indices(&[0], "summary_image_nested.image_bytes")
            .await
            .unwrap();
        assert_eq!(
            blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"abc"
        );
        let descriptors = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(
            descriptors.num_rows(),
            if initial_version.is_some() { 2 } else { 1 }
        );
        for fragment in dataset.get_fragments() {
            assert_eq!(
                fragment.scan().try_into_batch().await.unwrap().schema(),
                descriptors.schema()
            );
        }
        assert!(
            dataset
                .schema()
                .field("summary_image_nested.image_bytes")
                .unwrap()
                .is_blob_v2()
        );
    }

    mod external_error {
        use super::*;
        use std::fmt;

        #[derive(Debug)]
        struct MyTestError {
            code: i32,
            details: String,
        }

        impl fmt::Display for MyTestError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "MyTestError({}): {}", self.code, self.details)
            }
        }

        impl std::error::Error for MyTestError {}

        fn create_failing_iterator(
            schema: Arc<Schema>,
            fail_at_batch: usize,
            error_code: i32,
        ) -> impl Iterator<Item = std::result::Result<RecordBatch, ArrowError>> {
            let mut batch_count = 0;
            std::iter::from_fn(move || {
                if batch_count >= 5 {
                    return None;
                }
                batch_count += 1;
                if batch_count == fail_at_batch {
                    Some(Err(ArrowError::ExternalError(Box::new(MyTestError {
                        code: error_code,
                        details: format!("Failed at batch {}", batch_count),
                    }))))
                } else {
                    let batch = RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(vec![batch_count as i32; 10]))],
                    )
                    .unwrap();
                    Some(Ok(batch))
                }
            })
        }

        #[tokio::test]
        async fn test_insert_builder_preserves_external_error() {
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

            let error_code = 42;
            let iter = create_failing_iterator(schema.clone(), 3, error_code);
            let reader = RecordBatchIterator::new(iter, schema);

            let result = InsertBuilder::new("memory://test_external_error")
                .execute_stream(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
                .await;

            match result {
                Err(Error::External { source }) => {
                    let original = source
                        .downcast_ref::<MyTestError>()
                        .expect("Should be able to downcast to MyTestError");
                    assert_eq!(original.code, error_code);
                    assert!(original.details.contains("batch 3"));
                }
                Err(other) => panic!("Expected Error::External variant, got: {:?}", other),
                Ok(_) => panic!("Expected error, got success"),
            }
        }

        #[tokio::test]
        async fn test_insert_builder_first_batch_error() {
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

            let error_code = 999;
            let iter = std::iter::once(Err(ArrowError::ExternalError(Box::new(MyTestError {
                code: error_code,
                details: "immediate failure".to_string(),
            }))));
            let reader = RecordBatchIterator::new(iter, schema);

            let result = InsertBuilder::new("memory://test_first_batch_error")
                .execute_stream(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
                .await;

            match result {
                Err(Error::External { source }) => {
                    let original = source.downcast_ref::<MyTestError>().unwrap();
                    assert_eq!(original.code, error_code);
                }
                Err(other) => panic!("Expected External, got: {:?}", other),
                Ok(_) => panic!("Expected error"),
            }
        }
    }

    #[tokio::test]
    async fn test_write_progress_callback() {
        use std::sync::Mutex;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        // Three batches of 100 rows each.
        let batches: Vec<_> = (0..3)
            .map(|_| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from(vec![0i32; 100]))],
                )
                .unwrap()
            })
            .collect();

        let stats_log: Arc<Mutex<Vec<crate::dataset::WriteStats>>> =
            Arc::new(Mutex::new(Vec::new()));
        let log_clone = stats_log.clone();

        InsertBuilder::new("memory://test_write_progress")
            .progress(move |stats| {
                log_clone.lock().unwrap().push(stats);
            })
            .execute_stream(RecordBatchIterator::new(
                batches.into_iter().map(Ok),
                schema,
            ))
            .await
            .unwrap();

        let log = stats_log.lock().unwrap();
        assert!(
            !log.is_empty(),
            "progress callback must be called at least once"
        );
        // bytes_written and rows_written must be monotonically non-decreasing.
        for window in log.windows(2) {
            assert!(
                window[1].bytes_written >= window[0].bytes_written,
                "bytes_written must not decrease: {:?} -> {:?}",
                window[0].bytes_written,
                window[1].bytes_written,
            );
            assert!(
                window[1].rows_written >= window[0].rows_written,
                "rows_written must not decrease",
            );
        }
        let last = log.last().unwrap();
        assert!(last.bytes_written > 0, "final bytes_written must be > 0");
        assert_eq!(last.rows_written, 300, "all 300 rows must be reported");
        assert_eq!(last.files_written, 1, "a single file should be written");
    }
}
