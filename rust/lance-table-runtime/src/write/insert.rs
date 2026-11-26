// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator};
use humantime::format_duration;
use lance_core::datatypes::{NullabilityComparison, Schema};
use lance_core::utils::tracing::{DATASET_WRITING_EVENT, TRACE_DATASET_EVENTS};
use lance_core::{Error, Result, is_system_column};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_io::object_store::ObjectStore;
use lance_io::stream::{SendableRecordBatchStream, StreamingWriteSource};
use lance_table::feature_flags::can_write_dataset;
use lance_table::format::Fragment;
use lance_table::io::commit::CommitHandler;
use lance_table::transaction::{Operation, Transaction, TransactionBuilder};
use object_store::path::Path;
use tracing::info;

use crate::blob::prepared_to_logical_blob_schema;
use crate::commit::{
    CommitDestination, CommitIndexHook, CommitLifecycleHook, resolve_commit_handler,
};
use crate::{CommitBuilder, Table, TableBuilder, TableSession};

use super::representation::TableRepresentation;
use super::schema::{promote_legacy_blob_schema, schema_compare_options};
use super::{
    FragmentWriteOptions, WriteMode, WriteParams, WriteProgressFn, WriteSeedHook, WriteStats,
};
use super::{validate_and_resolve_target_bases_with_primary, write_fragments};

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
#[derive(Clone)]
pub struct InsertBuilder<'a> {
    dest: CommitDestination<'a>,
    // TODO: make these parameters a part of the builder, and add specific methods.
    params: Option<&'a WriteParams>,
    write_progress: Option<WriteProgressFn>,
    preassigned_data_file_name: Option<Arc<String>>,
    inherited_field_ids: bool,
    disable_dictionary_split: bool,
    default_session: Option<Arc<TableSession>>,
    seed_hook: Option<Arc<dyn WriteSeedHook>>,
    index_hook: Option<Arc<dyn CommitIndexHook>>,
    lifecycle_hook: Option<Arc<dyn CommitLifecycleHook>>,
}

impl<'a> InsertBuilder<'a> {
    /// Insert into an existing snapshot or open a destination by URI.
    pub fn new(dest: impl Into<CommitDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            params: None,
            write_progress: None,
            preassigned_data_file_name: None,
            inherited_field_ids: false,
            disable_dictionary_split: false,
            default_session: None,
            seed_hook: None,
            index_hook: None,
            lifecycle_hook: None,
        }
    }

    /// Set the options for fragment writing and publication.
    pub fn with_params(mut self, params: &'a WriteParams) -> Self {
        self.params = Some(params);
        self
    }

    #[doc(hidden)]
    pub fn with_preassigned_data_file_name(
        mut self,
        preassigned_data_file_name: impl Into<String>,
    ) -> Self {
        self.preassigned_data_file_name = Some(Arc::new(preassigned_data_file_name.into()));
        self
    }

    /// The input's field IDs are an existing table's, so a new table keeps them
    /// even past the limit for caller-supplied IDs. For a MemWAL generation.
    #[doc(hidden)]
    pub fn with_inherited_field_ids(mut self) -> Self {
        self.inherited_field_ids = true;
        self
    }

    /// Never starts another data file because a dictionary column outgrows its key
    /// type; the encoder's own key-type check applies instead. For a MemWAL generation,
    /// whose deletion vector and indexes address a single fragment.
    #[doc(hidden)]
    pub fn with_dictionary_split_disabled(mut self) -> Self {
        self.disable_dictionary_split = true;
        self
    }

    /// Use this session when the write options and destination supply none.
    #[doc(hidden)]
    pub fn with_default_session(mut self, session: Arc<TableSession>) -> Self {
        self.default_session = Some(session);
        self
    }

    /// Use the host's registry to collect write seeds when appending.
    pub fn with_seed_hook(mut self, hook: Arc<dyn WriteSeedHook>) -> Self {
        self.seed_hook = Some(hook);
        self
    }

    /// Use index support during manifest publication.
    pub fn with_index_hook(mut self, hook: Arc<dyn CommitIndexHook>) -> Self {
        self.index_hook = Some(hook);
        self
    }

    /// Register publication callbacks supplied by the host.
    pub fn with_lifecycle_hook(mut self, hook: Arc<dyn CommitLifecycleHook>) -> Self {
        self.lifecycle_hook = Some(hook);
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

    /// Execute the insert operation with the given data.
    ///
    /// This writes the data fragments and commits them into the dataset.
    pub async fn execute(&self, data: Vec<RecordBatch>) -> Result<Table> {
        let (transaction, context) = self.write_uncommitted_impl(data).await?;
        self.do_commit(&context, transaction).await
    }

    /// Execute the insert operation with the given stream.
    ///
    /// This writes the data fragments and commits them into the dataset.
    pub async fn execute_stream(&self, source: impl StreamingWriteSource) -> Result<Table> {
        let (stream, schema) = source.into_stream_and_schema().await?;
        self.execute_stream_impl(stream, schema).await
    }

    async fn execute_stream_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<Table> {
        let (transaction, context) = self.write_uncommitted_stream_impl(stream, schema).await?;
        self.do_commit(&context, transaction).await
    }

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    ///
    /// # Example: Append data to a dataset
    ///
    /// ```rust
    /// use lance_table_runtime::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
    ///
    /// # use std::sync::Arc;
    /// # use arrow_array::RecordBatch;
    /// # use lance_core::Result;
    /// # use lance_table_runtime::Table;
    /// # async fn example(dataset: Arc<Table>, data: Vec<RecordBatch>) -> Result<()> {
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
        self.write_uncommitted_impl(data).await.map(|(t, _)| t)
    }

    async fn do_commit(
        &self,
        context: &WriteContext<'_>,
        transaction: Transaction,
    ) -> Result<Table> {
        let mut commit_builder = CommitBuilder::new(context.dest.clone())
            .use_stable_row_ids(context.params.enable_stable_row_ids)
            .with_exact_storage_format(context.storage_version)
            .enable_v2_manifest_paths(context.params.enable_v2_manifest_paths)
            .with_commit_handler(context.commit_handler.clone())
            .with_object_store(context.object_store.clone())
            .with_skip_auto_cleanup(context.params.skip_auto_cleanup)
            .with_disable_transaction_file(context.params.disable_transaction_file);

        if let Some(params) = context.params.store_params.as_ref() {
            commit_builder = commit_builder.with_store_params(params.clone());
        }

        if let Some(session) = context.params.session.as_ref() {
            commit_builder = commit_builder.with_session(session.clone());
        } else if context.dest.table().is_none()
            && let Some(session) = &self.default_session
        {
            commit_builder = commit_builder.with_session(session.clone());
        }
        if self.inherited_field_ids {
            commit_builder = commit_builder.with_inherited_field_ids();
        }

        if let Some(hook) = &self.index_hook {
            commit_builder = commit_builder.with_index_hook(hook.clone());
        }

        Box::pin(commit_builder.execute_with_lifecycle(transaction, self.lifecycle_hook.as_deref()))
            .await
    }

    async fn write_uncommitted_impl(
        &self,
        data: Vec<RecordBatch>,
    ) -> Result<(Transaction, WriteContext<'_>)> {
        // TODO: This should be able to split the data up based on max_rows_per_file
        // and write in parallel. https://github.com/lance-format/lance/issues/1980
        if data.is_empty() {
            return Err(Error::invalid_input_source("No data to write".into()));
        }
        let schema = data[0].schema();
        for batch in data.iter().skip(1) {
            if batch.schema() != schema {
                return Err(Error::invalid_input_source(
                    "All record batches must have the same schema".into(),
                ));
            }
        }
        let reader = RecordBatchIterator::new(data.into_iter().map(Ok), schema);
        let (stream, schema) = StreamingWriteSource::into_stream_and_schema(reader).await?;
        self.write_uncommitted_stream_impl(stream, schema).await
    }

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    pub async fn execute_uncommitted_stream(
        &self,
        source: impl StreamingWriteSource,
    ) -> Result<Transaction> {
        let (stream, schema) = source.into_stream_and_schema().await?;
        let (transaction, _) = self.write_uncommitted_stream_impl(stream, schema).await?;
        Ok(transaction)
    }

    async fn write_uncommitted_stream_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<(Transaction, WriteContext<'_>)> {
        let mut context = Box::pin(self.resolve_context()).await?;

        info!(
            target: TRACE_DATASET_EVENTS,
            event=DATASET_WRITING_EVENT,
            uri=context.dest.uri(),
            mode=?context.params.mode
        );

        let (stream, schema) = Self::convert_to_table_representation(&context, stream, schema)?;
        self.validate_write(&mut context, &schema)?;

        let existing_base_paths = context.dest.table().map(|ds| &ds.manifest.base_paths);
        let target_base_info = validate_and_resolve_target_bases_with_primary(
            &mut context.params,
            existing_base_paths,
            &context.object_store,
            &context.base_path,
            &context.dest.uri(),
        )
        .await?;

        if self.preassigned_data_file_name.is_some() && target_base_info.is_some() {
            return Err(Error::invalid_input(
                "a fixed data file name cannot be combined with target bases",
            ));
        }
        let (written_fragments, written_schema) = Box::pin(write_fragments(
            context.storage_version,
            context.dest.table(),
            context.object_store.clone(),
            &context.base_path,
            schema.clone(),
            stream,
            context.params.clone(),
            FragmentWriteOptions {
                target_bases_info: target_base_info,
                preassigned_data_file_name: self.preassigned_data_file_name.clone(),
                inherited_field_ids: self.inherited_field_ids,
                disable_dictionary_split: self.disable_dictionary_split,
                seed_hook: self.seed_hook.as_deref(),
                ..Default::default()
            },
        ))
        .await?;

        let transaction = Self::build_transaction(written_schema, written_fragments, &context)?;

        Ok((transaction, context))
    }

    fn build_transaction(
        schema: Schema,
        fragments: Vec<Fragment>,
        context: &WriteContext<'_>,
    ) -> Result<Transaction> {
        let operation = match context.params.mode {
            WriteMode::Create => {
                let mut upsert_values = HashMap::new();
                if let Some(auto_cleanup_params) = context.params.auto_cleanup.as_ref() {
                    upsert_values.insert(
                        String::from("lance.auto_cleanup.interval"),
                        auto_cleanup_params.interval.to_string(),
                    );

                    let duration = auto_cleanup_params
                        .older_than
                        .to_std()
                        .map_err(|e| Error::invalid_input_source(e.into()))?;
                    upsert_values.insert(
                        String::from("lance.auto_cleanup.older_than"),
                        format_duration(duration).to_string(),
                    );
                }
                let config_upsert_values = if upsert_values.is_empty() {
                    None
                } else {
                    Some(upsert_values)
                };
                Operation::Overwrite {
                    // Use the full schema, not the written schema
                    schema,
                    fragments,
                    config_upsert_values,
                    initial_bases: context.params.initial_bases.clone(),
                }
            }
            WriteMode::Overwrite => Operation::Overwrite {
                schema,
                fragments,
                config_upsert_values: None,
                initial_bases: context.params.initial_bases.clone(),
            },
            WriteMode::Append => Operation::Append { fragments },
        };

        let transaction = TransactionBuilder::new(
            context
                .dest
                .table()
                .map(|ds| ds.manifest.version)
                .unwrap_or(0),
            operation,
        )
        .transaction_properties(context.params.transaction_properties.clone())
        .build();

        Ok(transaction)
    }

    /// Convert string and binary columns appended to an existing table to the
    /// representations the table stores, so the rest of the write sees the
    /// table's types.
    fn convert_to_table_representation(
        context: &WriteContext,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<(SendableRecordBatchStream, Schema)> {
        let (WriteMode::Append, CommitDestination::Table(table)) =
            (&context.params.mode, &context.dest)
        else {
            return Ok((stream, schema));
        };
        let Some(representation) =
            TableRepresentation::try_new(stream.schema().as_ref(), table.schema())
        else {
            return Ok((stream, schema));
        };
        let schema = Schema::try_from(representation.schema().as_ref())?;
        Ok((representation.convert_stream(stream), schema))
    }

    fn validate_write(&self, context: &mut WriteContext, data_schema: &Schema) -> Result<()> {
        // Write mode
        match (&context.params.mode, &context.dest) {
            (WriteMode::Create, CommitDestination::Table(ds)) => {
                return Err(Error::dataset_already_exists(ds.uri().to_string()));
            }
            (WriteMode::Append | WriteMode::Overwrite, CommitDestination::Uri(uri)) => {
                log::warn!("No existing dataset at {uri}, it will be created");
                context.params.mode = WriteMode::Create;
            }
            _ => {}
        }

        // Validate schema
        if matches!(context.params.mode, WriteMode::Append)
            && let CommitDestination::Table(dataset) = &context.dest
        {
            // If the dataset is already using (or not using) stable row ids, we need to match
            // and ignore whatever the user provided as input
            if context.params.enable_stable_row_ids != dataset.manifest.uses_stable_row_ids() {
                log::info!(
                    "Ignoring user provided stable row ids setting of {}, dataset already has it set to {}",
                    context.params.enable_stable_row_ids,
                    dataset.manifest.uses_stable_row_ids()
                );
                context.params.enable_stable_row_ids = dataset.manifest.uses_stable_row_ids();
            }

            let version = dataset.manifest.data_storage_format.lance_file_format();
            if (version == ConcreteFileVersion::V1)
                != (context.storage_version == ConcreteFileVersion::V1)
            {
                return Err(Error::invalid_input(format!(
                    "Cannot append data files in version {} to a dataset with default version {version}: V1 and V2 storage versions cannot be mixed",
                    context.storage_version
                )));
            }
            let mut schema_cmp_opts = schema_compare_options(version);
            schema_cmp_opts.compare_nullability = NullabilityComparison::Ignore;
            schema_cmp_opts.allow_missing_if_nullable = true;
            schema_cmp_opts.ignore_field_order = true;

            let normalized_data_schema = prepared_to_logical_blob_schema(data_schema)?;
            if normalized_data_schema
                .check_compatible(dataset.schema(), &schema_cmp_opts)
                .is_err()
            {
                let normalized_data_schema = promote_legacy_blob_schema(&normalized_data_schema)?;
                let dataset_schema = promote_legacy_blob_schema(dataset.schema())?;
                normalized_data_schema.check_compatible(&dataset_schema, &schema_cmp_opts)?;
            }
        }

        for field in data_schema.fields.iter() {
            if is_system_column(&field.name) {
                return Err(Error::invalid_input_source(
                    format!(
                        "The column {} is a reserved name and cannot be used in a Lance dataset",
                        field.name
                    )
                    .into(),
                ));
            }
        }

        // Feature flags
        if let CommitDestination::Table(dataset) = &context.dest
            && !can_write_dataset(dataset.manifest.writer_feature_flags)
        {
            let message = format!(
                "This dataset cannot be written by this version of Lance. \
                Please upgrade Lance to write to this dataset.\n Flags: {}",
                dataset.manifest.writer_feature_flags
            );
            return Err(Error::not_supported_source(message.into()));
        }

        Ok(())
    }

    async fn resolve_context(&self) -> Result<WriteContext<'a>> {
        let mut params = self.params.cloned().unwrap_or_default();
        if let Some(cb) = self.write_progress.clone() {
            params.write_progress = Some(cb);
        }
        let (object_store, base_path, commit_handler) = match &self.dest {
            CommitDestination::Table(dataset) => (
                dataset.state.object_store.clone(),
                dataset.state.base.clone(),
                params
                    .commit_handler
                    .clone()
                    .unwrap_or_else(|| dataset.state.commit_handler.clone()),
            ),
            CommitDestination::Uri(uri) => {
                let registry = params
                    .session
                    .as_ref()
                    .map(|s| s.store_registry())
                    .unwrap_or_else(|| Arc::new(Default::default()));
                let (object_store, base_path) = ObjectStore::from_uri_and_params(
                    registry,
                    uri,
                    &params.store_params.clone().unwrap_or_default(),
                )
                .await?;
                let commit_handler = resolve_commit_handler(
                    uri,
                    params.commit_handler.clone(),
                    &params.store_params,
                )
                .await?;
                (object_store, base_path, commit_handler)
            }
        };
        let dest = match &self.dest {
            CommitDestination::Table(dataset) => CommitDestination::Table(dataset.clone()),
            CommitDestination::Uri(uri) => {
                // Check if it already exists.
                let mut builder = TableBuilder::from_uri(uri);
                if let Some(store_params) = &params.store_params {
                    builder = builder.with_store_params(store_params.clone());
                }
                if let Some(commit_handler) = &params.commit_handler {
                    builder = builder.with_commit_handler(commit_handler.clone());
                }
                if let Some(session) = &params.session {
                    builder = builder.with_session(session.clone());
                }
                let table_session = params
                    .session
                    .clone()
                    .or_else(|| self.default_session.clone())
                    .unwrap_or_default();
                match builder.load_with_session(table_session).await {
                    Ok(dataset) => CommitDestination::Table(Arc::new(dataset)),
                    Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                        CommitDestination::Uri(uri)
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        let storage_version = match &dest {
            CommitDestination::Table(dataset) => params
                .data_storage_version
                .map(LanceFileVersion::resolve)
                .unwrap_or_else(|| dataset.manifest.data_storage_format.lance_file_format()),
            // Otherwise (no existing dataset) fallback to the default if the user didn't specify
            CommitDestination::Uri(_) => params.storage_version_or_default(),
        };

        Ok(WriteContext {
            params,
            dest,
            object_store,
            base_path,
            commit_handler,
            storage_version,
        })
    }
}

struct WriteContext<'a> {
    params: WriteParams,
    dest: CommitDestination<'a>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    commit_handler: Arc<dyn CommitHandler>,
    storage_version: ConcreteFileVersion,
}
