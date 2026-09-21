// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::index_progress::JavaIndexBuildProgress;
use crate::namespace::{
    BlockingDirectoryNamespace, BlockingRestNamespace, create_java_lance_namespace,
};
use crate::session::{handle_from_session, session_from_handle};
use crate::traits::{FromJObjectWithEnv, FromJString, export_vec, import_vec, import_vec_to_rust};
use crate::utils::{
    build_compaction_options, extract_base_store_params, extract_storage_options,
    extract_write_params, get_scalar_index_params, get_vector_index_params, to_java_map,
    to_rust_map,
};
use crate::{block_on, traits::IntoJava};
use arrow::array::RecordBatchReader;
use arrow::datatypes::Schema;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatchIterator;
use arrow_schema::Schema as ArrowSchema;
use chrono::{DateTime, Utc};
use jni::objects::{JMap, JString, JValue};
use jni::sys::{jboolean, jint};
use jni::sys::{jbyteArray, jlong};
use jni::{JNIEnv, objects::JObject};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::cleanup::{
    CleanupCandidateFile, CleanupExplanation, CleanupFileKind, CleanupPolicy,
    CleanupReferencedBranch, RemovalStats,
};
use lance::dataset::optimize::{CompactionOptions as RustCompactionOptions, compact_files};
use lance::dataset::refs::{Ref, TagContents};
use lance::dataset::statistics::{DataStatistics, DatasetStatisticsExt};
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{
    ColumnAlteration, CommitBuilder, Dataset, NewColumnTransform, ProjectionRequest, ReadParams,
    Version, WriteParams,
};
use lance::index::{DatasetIndexExt, IndexSegment, IntoIndexSegment};
use lance::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
use lance::io::{ObjectStore, ObjectStoreParams};
use lance::session::Session as LanceSession;
use lance::table::format::IndexMetadata;
use lance::table::format::{BasePath, Fragment, WriterVersion};
use lance_core::datatypes::Schema as LanceSchema;
use lance_file::version::LanceFileVersion;
use lance_index::IndexCriteria as RustIndexCriteria;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::noop_progress;
use lance_index::{IndexParams, IndexType};
use lance_io::object_store::ObjectStoreRegistry;
use lance_io::object_store::{LanceNamespaceStorageOptionsProvider, StorageOptionsProvider};
use lance_namespace::LanceNamespace;
use lance_table::io::commit::CommitHandler;
use lance_table::io::commit::external_manifest::ExternalManifestCommitHandler;
use lance_table::io::commit::{ManifestLocation, ManifestNamingScheme};
use std::collections::HashMap;
use std::future::IntoFuture;
use std::iter::empty;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};
use uuid::Uuid;

pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

impl FromJObjectWithEnv<BasePath> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<BasePath> {
        let id = env.get_u32_from_method(self, "getId")?;
        let name = env.get_optional_string_from_method(self, "getName")?;
        let path = env.get_string_from_method(self, "getPath")?;
        let is_dataset_root = env.get_boolean_from_method(self, "isDatasetRoot")?;
        Ok(BasePath {
            id,
            name,
            path,
            is_dataset_root,
        })
    }
}

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

impl BlockingDataset {
    /// Get the initial storage options used to open this dataset.
    ///
    /// Returns the options that were provided when the dataset was opened,
    /// without any refresh from the provider. Returns None if no storage options
    /// were provided.
    pub fn initial_storage_options(&self) -> Option<HashMap<String, String>> {
        self.inner.initial_storage_options().cloned()
    }

    /// Get the latest storage options, potentially refreshed from the provider.
    ///
    /// If a storage options provider was configured and credentials are expiring,
    /// this will refresh them.
    pub fn latest_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
        block_on(async { self.inner.latest_storage_options().await })
            .map(|opt| opt.map(|opts| opts.0))
            .map_err(|e| Error::io_error(e.to_string()))
    }

    pub fn drop(uri: &str, storage_options: HashMap<String, String>) -> Result<()> {
        block_on(async move {
            let registry = Arc::new(ObjectStoreRegistry::default());
            let object_store_params = ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance::io::StorageOptionsAccessor::with_static_options(storage_options),
                )),
                ..Default::default()
            };
            let (object_store, path) =
                ObjectStore::from_uri_and_params(registry, uri, &object_store_params)
                    .await
                    .map_err(|e| Error::io_error(e.to_string()))?;
            lance::dataset::validate_dataset_root_for_drop(&object_store, &path)
                .await
                .map_err(|e| Error::input_error(e.to_string()))?;
            object_store
                .remove_dir_all(path)
                .await
                .map_err(|e| Error::io_error(e.to_string()))
        })
    }

    pub fn list_manifest_locations(
        uri: &str,
        storage_options: HashMap<String, String>,
    ) -> Result<Vec<ManifestLocation>> {
        let accessor = (!storage_options.is_empty()).then(|| {
            Arc::new(lance::io::StorageOptionsAccessor::with_static_options(
                storage_options,
            ))
        });
        let params = ReadParams {
            store_options: Some(ObjectStoreParams {
                storage_options_accessor: accessor,
                ..Default::default()
            }),
            ..Default::default()
        };
        Ok(block_on(
            DatasetBuilder::from_uri(uri)
                .with_read_params(params)
                .list_manifest_locations(),
        )?)
    }

    pub fn write(
        reader: impl RecordBatchReader + Send + 'static,
        uri: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let inner = block_on(Dataset::write(reader, uri, params))?;
        Ok(Self { inner })
    }

    pub fn new(dataset: Dataset) -> Self {
        Self { inner: dataset }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open(
        uri: &str,
        version: Option<u64>,
        block_size: Option<i32>,
        index_cache_size_bytes: i64,
        metadata_cache_size_bytes: i64,
        storage_options: HashMap<String, String>,
        base_store_params: HashMap<String, ObjectStoreParams>,
        serialized_manifest: Option<&[u8]>,
        storage_options_provider: Option<Arc<dyn StorageOptionsProvider>>,
        session: Option<Arc<LanceSession>>,
        namespace: Option<Arc<dyn LanceNamespace>>,
        table_id: Option<Vec<String>>,
        namespace_client_managed_versioning: bool,
    ) -> Result<Self> {
        // Create storage options accessor from storage_options and provider
        let accessor = match (storage_options.is_empty(), storage_options_provider) {
            (false, Some(provider)) => Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_initial_and_provider(
                    storage_options,
                    provider,
                ),
            )),
            (false, None) => Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_static_options(storage_options),
            )),
            (true, Some(provider)) => Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_provider(provider),
            )),
            (true, None) => None,
        };

        let store_params = ObjectStoreParams {
            block_size: block_size.map(|size| size as usize),
            storage_options_accessor: accessor,
            ..Default::default()
        };
        let params = ReadParams {
            index_cache_size_bytes: index_cache_size_bytes as usize,
            metadata_cache_size_bytes: metadata_cache_size_bytes as usize,
            store_options: Some(store_params),
            session,
            ..Default::default()
        };

        let mut builder = DatasetBuilder::from_uri(uri).with_read_params(params);

        for (base_path, store_params) in base_store_params {
            builder = builder.with_base_store_params(base_path, store_params);
        }

        if let Some(ver) = version {
            builder = builder.with_version(ver);
        }

        if let Some(serialized_manifest) = serialized_manifest {
            builder = builder.with_serialized_manifest(serialized_manifest)?;
        }

        // Set up namespace commit handler only if namespace manages versioning
        if namespace_client_managed_versioning
            && let (Some(namespace_client), Some(tid)) = (namespace, table_id)
        {
            let external_store =
                LanceNamespaceExternalManifestStore::for_table_uri(namespace_client, tid, uri)?;
            let commit_handler: Arc<dyn CommitHandler> = Arc::new(ExternalManifestCommitHandler {
                external_manifest_store: Arc::new(external_store),
            });
            builder = builder.with_commit_handler(commit_handler);
        }

        let inner = block_on(builder.load())?;
        Ok(Self { inner })
    }

    pub fn commit(
        uri: &str,
        operation: Operation,
        read_version: Option<u64>,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        let accessor = if storage_options.is_empty() {
            None
        } else {
            Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_static_options(storage_options),
            ))
        };
        let inner = block_on(Dataset::commit(
            uri,
            operation,
            read_version,
            Some(ObjectStoreParams {
                storage_options_accessor: accessor,
                ..Default::default()
            }),
            None,
            Default::default(),
            false,
        ))?;
        Ok(Self { inner })
    }

    pub fn latest_version(&self) -> Result<u64> {
        let version = block_on(self.inner.latest_version_id())?;
        Ok(version)
    }

    pub fn list_versions(&self) -> Result<Vec<Version>> {
        let versions = block_on(self.inner.versions())?;
        Ok(versions)
    }

    pub fn count_versions(&self) -> Result<u64> {
        Ok(block_on(self.inner.count_versions())?)
    }

    pub fn version(&self) -> Result<Version> {
        Ok(self.inner.version())
    }

    pub fn checkout_version(&mut self, version: u64) -> Result<Self> {
        let inner = block_on(self.inner.checkout_version(version))?;
        Ok(Self { inner })
    }

    pub fn checkout_tag(&mut self, tag: &str) -> Result<Self> {
        let inner = block_on(self.inner.checkout_version(tag))?;
        Ok(Self { inner })
    }

    pub fn checkout_latest(&mut self) -> Result<()> {
        block_on(self.inner.checkout_latest())?;
        Ok(())
    }

    pub fn restore(&mut self) -> Result<()> {
        block_on(self.inner.restore())?;
        Ok(())
    }

    pub fn list_tags(&self) -> Result<HashMap<String, TagContents>> {
        let tags = block_on(self.inner.tags().list())?;
        Ok(tags)
    }

    pub fn list_branches(&self) -> Result<HashMap<String, lance::dataset::refs::BranchContents>> {
        let branches = block_on(self.inner.branches().list())?;
        Ok(branches)
    }

    pub fn delete_branch(&mut self, branch: &str) -> Result<()> {
        block_on(self.inner.branches().delete(branch, true))?;
        Ok(())
    }

    pub fn checkout_reference(
        &mut self,
        branch: Option<String>,
        version: Option<u64>,
        tag: Option<String>,
    ) -> Result<Self> {
        let reference = if let Some(tag_name) = tag {
            Ref::from(tag_name.as_str())
        } else {
            Ref::Version(branch, version)
        };
        let inner = block_on(self.inner.checkout_version(reference))?;
        Ok(Self { inner })
    }

    pub fn create_tag(&mut self, tag: &str, reference: Ref) -> Result<()> {
        block_on(self.inner.tags().create(tag, reference))?;
        Ok(())
    }

    pub fn delete_tag(&mut self, tag: &str) -> Result<()> {
        block_on(self.inner.tags().delete(tag))?;
        Ok(())
    }

    pub fn update_tag(&mut self, tag: &str, reference: Ref) -> Result<()> {
        block_on(self.inner.tags().update(tag, reference))?;
        Ok(())
    }

    pub fn replace_tag_metadata(
        &mut self,
        tag: &str,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        block_on(self.inner.tags().replace_metadata(tag, metadata))?;
        Ok(())
    }

    pub fn replace_branch_metadata(
        &mut self,
        branch: &str,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        block_on(self.inner.branches().replace_metadata(branch, metadata))?;
        Ok(())
    }

    pub fn get_version(&self, tag: &str) -> Result<u64> {
        let version = block_on(self.inner.tags().get_version(tag))?;
        Ok(version)
    }

    pub fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        let rows = block_on(self.inner.count_rows(filter))?;
        Ok(rows)
    }

    pub fn calculate_data_stats(&self) -> Result<DataStatistics> {
        let stats = block_on(Arc::new(self.clone().inner).calculate_data_stats())?;
        Ok(stats)
    }

    pub fn list_indexes(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let indexes = block_on(self.inner.load_indices())?;
        Ok(indexes)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn commit_transaction(
        &mut self,
        transaction: Transaction,
        store_params: ObjectStoreParams,
        detached: bool,
        enable_v2_manifest_paths: bool,
        use_stable_row_ids: Option<bool>,
        storage_format: Option<LanceFileVersion>,
        max_retries: u32,
        skip_auto_cleanup: bool,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        commit_timeout: Option<std::time::Duration>,
    ) -> Result<Self> {
        let mut builder = CommitBuilder::new(Arc::new(self.clone().inner))
            .with_store_params(store_params)
            .with_detached(detached)
            .enable_v2_manifest_paths(enable_v2_manifest_paths)
            .with_timeout(commit_timeout);
        if let Some(use_stable) = use_stable_row_ids {
            builder = builder.use_stable_row_ids(use_stable);
        }
        if let Some(format) = storage_format {
            builder = builder.with_storage_format(format);
        }
        if max_retries > 0 {
            builder = builder.with_max_retries(max_retries);
        }
        if skip_auto_cleanup {
            builder = builder.with_skip_auto_cleanup(true);
        }
        if let Some(handler) = commit_handler {
            builder = builder.with_commit_handler(handler);
        }
        let new_dataset = block_on(builder.execute(transaction))?;
        Ok(BlockingDataset { inner: new_dataset })
    }

    pub fn read_transaction(&self) -> Result<Option<Transaction>> {
        let transaction = block_on(self.inner.read_transaction())?;
        Ok(transaction)
    }

    pub fn get_table_metadata(&self) -> Result<HashMap<String, String>> {
        Ok(self.inner.metadata().clone())
    }

    pub fn compact(&mut self, options: RustCompactionOptions) -> Result<()> {
        block_on(compact_files(&mut self.inner, options, None))?;
        Ok(())
    }

    pub fn cleanup_with_policy(&mut self, policy: CleanupPolicy) -> Result<RemovalStats> {
        Ok(block_on(self.inner.cleanup_with_policy(policy))?)
    }

    pub fn explain_cleanup_with_policy(
        &self,
        policy: CleanupPolicy,
        max_candidate_files: Option<usize>,
    ) -> Result<CleanupExplanation> {
        let mut op = self.inner.cleanup(policy);
        if let Some(limit) = max_candidate_files {
            op = op.with_max_candidate_files(limit);
        }
        Ok(block_on(op.explain())?)
    }

    pub fn close(&self) {}
}

///////////////////
// Write Methods //
///////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_createWithFfiSchema<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,        // Optional<Integer>
    max_rows_per_group: JObject,       // Optional<Integer>
    max_bytes_per_file: JObject,       // Optional<Long>
    mode: JObject,                     // Optional<String>
    enable_stable_row_ids: JObject,    // Optional<Boolean>
    data_storage_version: JObject,     // Optional<String>
    enable_v2_manifest_paths: JObject, // Optional<Boolean>
    storage_options_obj: JObject,      // Map<String, String>
    base_store_params_obj: JObject,    // Map<String, Map<String, String>>
    initial_bases: JObject,
    target_bases: JObject,
    allow_external_blob_outside_bases: JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,     // Optional<Long>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_with_ffi_schema(
            &mut env,
            arrow_schema_addr,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            enable_stable_row_ids,
            data_storage_version,
            enable_v2_manifest_paths,
            storage_options_obj,
            base_store_params_obj,
            initial_bases,
            target_bases,
            allow_external_blob_outside_bases,
            blob_pack_file_size_threshold,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_schema<'local>(
    env: &mut JNIEnv<'local>,
    arrow_schema_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,        // Optional<Integer>
    max_rows_per_group: JObject,       // Optional<Integer>
    max_bytes_per_file: JObject,       // Optional<Long>
    mode: JObject,                     // Optional<String>
    enable_stable_row_ids: JObject,    // Optional<Boolean>
    data_storage_version: JObject,     // Optional<String>
    enable_v2_manifest_paths: JObject, // Optional<Boolean>
    storage_options_obj: JObject,      // Map<String, String>
    base_store_params_obj: JObject,    // Map<String, Map<String, String>>
    initial_bases: JObject,
    target_bases: JObject,
    allow_external_blob_outside_bases: JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,     // Optional<Long>
) -> Result<JObject<'local>> {
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let schema = Schema::try_from(&c_schema)?;

    let reader = RecordBatchIterator::new(empty(), Arc::new(schema));
    create_dataset(
        env,
        path,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        enable_stable_row_ids,
        data_storage_version,
        enable_v2_manifest_paths,
        storage_options_obj,
        base_store_params_obj,
        initial_bases,
        target_bases,
        allow_external_blob_outside_bases,
        blob_pack_file_size_threshold,
        reader,
        None,  // No namespace for schema-only creation
        false, // No managed versioning for schema-only creation
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_drop<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString<'local>,
    storage_options_obj: JObject<'local>,
) -> JObject<'local> {
    let path_str = ok_or_throw!(env, path.extract(&mut env));
    let storage_options =
        ok_or_throw!(env, extract_storage_options(&mut env, &storage_options_obj));
    ok_or_throw!(env, BlockingDataset::drop(&path_str, storage_options));
    JObject::null()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeMigrateManifestPathsV2(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_native_migrate_manifest_paths_v2(&mut env, java_dataset)
    )
}

fn inner_native_migrate_manifest_paths_v2(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.migrate_manifest_paths_v2())?;
    Ok(())
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lance_Dataset_createWithFfiStream<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,                    // Optional<Integer>
    max_rows_per_group: JObject,                   // Optional<Integer>
    max_bytes_per_file: JObject,                   // Optional<Long>
    mode: JObject,                                 // Optional<String>
    enable_stable_row_ids: JObject,                // Optional<Boolean>
    data_storage_version: JObject,                 // Optional<String>
    enable_v2_manifest_paths: JObject,             // Optional<Boolean>
    storage_options_obj: JObject,                  // Map<String, String>
    base_store_params_obj: JObject,                // Map<String, Map<String, String>>
    initial_bases: JObject,                        // Optional<List<BasePath>>
    target_bases: JObject,                         // Optional<List<String>>
    allow_external_blob_outside_bases: JObject,    // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,        // Optional<Long>
    namespace_obj: JObject,                        // LanceNamespace (can be null)
    table_id_obj: JObject,                         // List<String> (can be null)
    namespace_client_managed_versioning: jboolean, // Whether namespace manages versioning
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_with_ffi_stream(
            &mut env,
            arrow_array_stream_addr,
            path,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            enable_stable_row_ids,
            data_storage_version,
            enable_v2_manifest_paths,
            storage_options_obj,
            base_store_params_obj,
            initial_bases,
            target_bases,
            allow_external_blob_outside_bases,
            blob_pack_file_size_threshold,
            namespace_obj,
            table_id_obj,
            namespace_client_managed_versioning != 0,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_stream<'local>(
    env: &mut JNIEnv<'local>,
    arrow_array_stream_addr: jlong,
    path: JString,
    max_rows_per_file: JObject,                 // Optional<Integer>
    max_rows_per_group: JObject,                // Optional<Integer>
    max_bytes_per_file: JObject,                // Optional<Long>
    mode: JObject,                              // Optional<String>
    enable_stable_row_ids: JObject,             // Optional<Boolean>
    data_storage_version: JObject,              // Optional<String>
    enable_v2_manifest_paths: JObject,          // Optional<Boolean>
    storage_options_obj: JObject,               // Map<String, String>
    base_store_params_obj: JObject,             // Map<String, Map<String, String>>
    initial_bases: JObject,                     // Optional<List<BasePath>>
    target_bases: JObject,                      // Optional<List<String>>
    allow_external_blob_outside_bases: JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,     // Optional<Long>
    namespace_obj: JObject,                     // LanceNamespace (can be null)
    table_id_obj: JObject,                      // List<String> (can be null)
    namespace_client_managed_versioning: bool,  // Whether namespace manages versioning
) -> Result<JObject<'local>> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;

    // Create the namespace wrapper for commit handling (if provided)
    let namespace_info = extract_namespace_info(env, &namespace_obj, &table_id_obj)?;

    create_dataset(
        env,
        path,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        enable_stable_row_ids,
        data_storage_version,
        enable_v2_manifest_paths,
        storage_options_obj,
        base_store_params_obj,
        initial_bases,
        target_bases,
        allow_external_blob_outside_bases,
        blob_pack_file_size_threshold,
        reader,
        namespace_info,
        namespace_client_managed_versioning,
    )
}

/// Creates a dataset from a record batch reader.
///
/// When `namespace_info` is provided, sets up the storage options provider for
/// credential refresh. When `namespace_client_managed_versioning` is true, also sets up the
/// commit handler for namespace-managed versioning.
#[allow(clippy::too_many_arguments)]
fn create_dataset<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    max_rows_per_file: JObject,
    max_rows_per_group: JObject,
    max_bytes_per_file: JObject,
    mode: JObject,
    enable_stable_row_ids: JObject,
    data_storage_version: JObject,
    enable_v2_manifest_paths: JObject,
    storage_options_obj: JObject,
    base_store_params_obj: JObject,
    initial_bases: JObject,
    target_bases: JObject,
    allow_external_blob_outside_bases: JObject,
    blob_pack_file_size_threshold: JObject,
    reader: impl RecordBatchReader + Send + 'static,
    namespace_info: Option<(Arc<dyn LanceNamespace>, Vec<String>)>,
    namespace_client_managed_versioning: bool,
) -> Result<JObject<'local>> {
    let path_str = path.extract(env)?;

    let mut write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
        &enable_stable_row_ids,
        &data_storage_version,
        Some(&enable_v2_manifest_paths),
        &storage_options_obj,
        &base_store_params_obj,
        &initial_bases,
        &target_bases,
        &allow_external_blob_outside_bases,
        &blob_pack_file_size_threshold,
    )?;

    // Set up namespace commit handler and storage options provider if namespace is provided
    if let Some((namespace, table_id)) = namespace_info {
        // Set up commit handler only if namespace manages versioning
        if namespace_client_managed_versioning {
            let external_store = LanceNamespaceExternalManifestStore::for_table_uri(
                namespace.clone(),
                table_id.clone(),
                &path_str,
            )?;
            let commit_handler: Arc<dyn CommitHandler> = Arc::new(ExternalManifestCommitHandler {
                external_manifest_store: Arc::new(external_store),
            });
            write_params.commit_handler = Some(commit_handler);
        }

        // Set up storage options provider for credential refresh
        let provider: Arc<dyn StorageOptionsProvider> = Arc::new(
            LanceNamespaceStorageOptionsProvider::new(namespace, table_id),
        );

        // Get existing storage options to combine with provider
        let storage_options: HashMap<String, String> =
            extract_storage_options(env, &storage_options_obj)?;

        let accessor = if storage_options.is_empty() {
            Arc::new(lance::io::StorageOptionsAccessor::with_provider(provider))
        } else {
            Arc::new(
                lance::io::StorageOptionsAccessor::with_initial_and_provider(
                    storage_options,
                    provider,
                ),
            )
        };
        write_params.store_params = Some(ObjectStoreParams {
            storage_options_accessor: Some(accessor),
            ..Default::default()
        });
    }

    let dataset = BlockingDataset::write(reader, &path_str, Some(write_params))?;
    dataset.into_java(env)
}

impl IntoJava for BlockingDataset {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        attach_native_dataset(env, self)
    }
}

impl IntoJava for Version {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let timestamp_str = self.timestamp.to_rfc3339();
        let jtimestamp = env.new_string(timestamp_str)?;
        let zdt = env
            .call_static_method(
                "java/time/ZonedDateTime",
                "parse",
                "(Ljava/lang/CharSequence;)Ljava/time/ZonedDateTime;",
                &[JValue::Object(&jtimestamp)],
            )?
            .l()?;

        let jmap = env.new_object("java/util/TreeMap", "()V", &[])?;
        let map = JMap::from_env(env, &jmap)?;

        for (k, v) in self.metadata {
            let jkey = env.new_string(k)?;
            let jval = env.new_string(v)?;
            map.put(env, &jkey, &jval).expect("ERROR: calling jmap.put");
        }

        let java_version = env.new_object(
            "org/lance/Version",
            "(JLjava/time/ZonedDateTime;Ljava/util/TreeMap;)V",
            &[
                JValue::Long(self.version as i64),
                JValue::Object(&zdt),
                JValue::Object(&jmap),
            ],
        )?;
        Ok(java_version)
    }
}

impl IntoJava for WriterVersion {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let library = env.new_string(self.library)?;
        let version = env.new_string(self.version)?;
        let prerelease = match self.prerelease {
            Some(value) => JObject::from(env.new_string(value)?),
            None => JObject::null(),
        };
        let build_metadata = match self.build_metadata {
            Some(value) => JObject::from(env.new_string(value)?),
            None => JObject::null(),
        };

        Ok(env.new_object(
            "org/lance/WriterVersion",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V",
            &[
                JValue::Object(&library),
                JValue::Object(&version),
                JValue::Object(&prerelease),
                JValue::Object(&build_metadata),
            ],
        )?)
    }
}

impl IntoJava for ManifestLocation {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let size = self.size.ok_or_else(|| {
            Error::runtime_error(format!("Manifest size is unavailable for {}", self.path))
        })?;
        let path = env.new_string(self.path.to_string())?;
        let naming_scheme = env.new_string(match self.naming_scheme {
            ManifestNamingScheme::V1 => "V1",
            ManifestNamingScheme::V2 => "V2",
        })?;
        let e_tag = match self.e_tag {
            Some(value) => JObject::from(env.new_string(value)?),
            None => JObject::null(),
        };
        Ok(env.new_object(
            "org/lance/ManifestLocation",
            "(JLjava/lang/String;JLjava/lang/String;Ljava/lang/String;)V",
            &[
                JValue::Long(self.version as i64),
                JValue::Object(&path),
                JValue::Long(size as i64),
                JValue::Object(&naming_scheme),
                JValue::Object(&e_tag),
            ],
        )?)
    }
}

fn attach_native_dataset<'local>(
    env: &mut JNIEnv<'local>,
    dataset: BlockingDataset,
) -> Result<JObject<'local>> {
    let j_dataset = create_java_dataset_object(env)?;
    // This block sets a native Rust object (dataset) as a field in the Java object (j_dataset).
    // Caution: This creates a potential for memory leaks. The Rust object (dataset) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_dataset`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
    unsafe { env.set_rust_field(&j_dataset, NATIVE_DATASET, dataset) }?;
    Ok(j_dataset)
}

fn create_java_dataset_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let object = env.new_object("org/lance/Dataset", "()V", &[])?;
    Ok(object)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_commitAppend<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_append(
            &mut env,
            path,
            read_version_obj,
            fragments_obj,
            storage_options_obj
        )
    )
}

pub fn inner_commit_append<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    read_version_obj: JObject,    // Optional<Long>
    fragment_objs: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let fragment_objs = import_vec(env, &fragment_objs)?;
    let mut fragments = Vec::with_capacity(fragment_objs.len());
    for f in fragment_objs {
        fragments.push(f.extract_object(env)?);
    }
    let op = Operation::Append { fragments };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let storage_options = extract_storage_options(env, &storage_options_obj)?;
    let dataset = BlockingDataset::commit(&path_str, op, read_version, storage_options)?;
    dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_commitOverwrite<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    arrow_schema_addr: jlong,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_overwrite(
            &mut env,
            path,
            arrow_schema_addr,
            read_version_obj,
            fragments_obj,
            storage_options_obj
        )
    )
}

pub fn inner_commit_overwrite<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    arrow_schema_addr: jlong,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let fragment_objs = import_vec(env, &fragments_obj)?;
    let mut fragments = Vec::with_capacity(fragment_objs.len());
    for f in fragment_objs {
        fragments.push(f.extract_object(env)?);
    }
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let arrow_schema = Schema::try_from(&c_schema)?;
    let schema = LanceSchema::try_from(&arrow_schema)?;

    let op = Operation::Overwrite {
        fragments,
        schema,
        config_upsert_values: None,
        initial_bases: None,
    };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options = to_rust_map(env, &jmap)?;
    let dataset = BlockingDataset::commit(&path_str, op, read_version, storage_options)?;
    dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_releaseNativeDataset(mut env: JNIEnv, obj: JObject) {
    ok_or_throw_without_return!(env, inner_release_native_dataset(&mut env, obj))
}

fn inner_release_native_dataset(env: &mut JNIEnv, obj: JObject) -> Result<()> {
    let dataset: BlockingDataset = unsafe { env.take_rust_field(obj, NATIVE_DATASET)? };
    dataset.close();
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCreateIndex<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject<'local>,
    columns_jobj: JObject<'local>, // List<String>
    index_type_code_jobj: jint,
    name_jobj: JObject<'local>,              // Optional<String>
    params_jobj: JObject<'local>,            // IndexParams
    replace_jobj: jboolean,                  // replace
    train_jobj: jboolean,                    // train
    fragments_jobj: JObject<'local>,         // List<Integer>
    index_uuid_jobj: JObject<'local>,        // String
    arrow_stream_addr_jobj: JObject<'local>, // Optional<Long>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_index(
            &mut env,
            java_dataset,
            columns_jobj,
            index_type_code_jobj,
            name_jobj,
            params_jobj,
            replace_jobj,
            train_jobj,
            fragments_jobj,
            index_uuid_jobj,
            arrow_stream_addr_jobj,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_index<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject<'local>,
    columns_jobj: JObject<'local>, // List<String>
    index_type_code_jobj: jint,
    name_jobj: JObject<'local>,              // Optional<String>
    params_jobj: JObject<'local>,            // IndexParams
    replace_jobj: jboolean,                  // replace
    train_jobj: jboolean,                    // train
    fragments_jobj: JObject<'local>,         // Optional<List<String>>
    index_uuid_jobj: JObject<'local>,        // Optional<String>
    arrow_stream_addr_jobj: JObject<'local>, // Optional<Long>
) -> Result<JObject<'local>> {
    let columns = env.get_strings(&columns_jobj)?;
    let index_type = IndexType::try_from(index_type_code_jobj)?;
    let name = env.get_string_opt(&name_jobj)?;
    let columns_slice: Vec<&str> = columns.iter().map(AsRef::as_ref).collect();
    let replace = replace_jobj != 0;
    let train = train_jobj != 0;
    let fragment_ids = env
        .get_ints_opt(&fragments_jobj)?
        .map(|vec| vec.into_iter().map(|i| i as u32).collect());
    let index_uuid = env
        .get_string_opt(&index_uuid_jobj)?
        .map(|s| {
            Uuid::parse_str(&s)
                .map_err(|e| Error::input_error(format!("Invalid UUID string for index_uuid: {e}")))
        })
        .transpose()?;
    let arrow_stream_addr_opt = env.get_long_opt(&arrow_stream_addr_jobj)?;
    let batch_reader = if let Some(arrow_stream_addr) = arrow_stream_addr_opt {
        let stream_ptr = arrow_stream_addr as *mut FFI_ArrowArrayStream;
        let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
        Some(reader)
    } else {
        None
    };

    // we should skip committing index when building distributed indices.
    let mut skip_commit = fragment_ids.is_some();

    // Handle scalar vs vector indices differently and get params before borrowing dataset
    let params_result: Result<Box<dyn IndexParams>> = match index_type {
        IndexType::Scalar
        | IndexType::BTree
        | IndexType::Bitmap
        | IndexType::LabelList
        | IndexType::Inverted
        | IndexType::NGram
        | IndexType::ZoneMap
        | IndexType::BloomFilter
        | IndexType::Fm
        | IndexType::RTree => {
            // For scalar indices, create a scalar IndexParams
            let (index_type_str, params_opt) = get_scalar_index_params(env, params_jobj)?;
            let scalar_params = lance_index::scalar::ScalarIndexParams {
                index_type: index_type_str,
                params: params_opt.clone(),
            };
            skip_commit = skip_commit || (index_type == IndexType::BTree && batch_reader.is_some());
            Ok(Box::new(scalar_params))
        }
        IndexType::FragmentReuse | IndexType::MemWal => {
            // System indices - not user-creatable
            Err(Error::input_error(format!(
                "Cannot create system index type: {:?}. System indices are managed internally.",
                index_type
            )))
        }
        IndexType::Vector
        | IndexType::IvfFlat
        | IndexType::IvfSq
        | IndexType::IvfPq
        | IndexType::IvfRq
        | IndexType::IvfHnswSq
        | IndexType::IvfHnswPq
        | IndexType::IvfHnswFlat => {
            // For vector indices, use the existing parameter handling
            get_vector_index_params(env, params_jobj)
        }
    };

    let params = params_result?;

    // Execute index creation in a block to ensure dataset_guard is dropped
    // before we call into_java (which needs to borrow env again)
    let index_metadata = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

        let mut index_builder = dataset_guard
            .inner
            .create_index_builder(&columns_slice, index_type, params.as_ref())
            .replace(replace)
            .train(train);

        if let Some(name) = name {
            index_builder = index_builder.name(name);
        }

        if let Some(fragment_ids) = fragment_ids {
            index_builder = index_builder.fragments(fragment_ids);
        }

        if let Some(index_uuid) = index_uuid {
            index_builder = index_builder.index_uuid(index_uuid);
        }

        if let Some(reader) = batch_reader {
            index_builder = index_builder.preprocessed_data(Box::new(reader));
        }

        if skip_commit {
            block_on(index_builder.execute_uncommitted())?
        } else {
            block_on(index_builder.into_future())?
        }
    };

    (&index_metadata).into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDropIndex(
    mut env: JNIEnv,
    java_dataset: JObject,
    name: JString,
) {
    ok_or_throw_without_return!(env, inner_drop_index(&mut env, java_dataset, name))
}

fn inner_drop_index(env: &mut JNIEnv, java_dataset: JObject, name: JString) -> Result<()> {
    let name = name.extract(env)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.drop_index(&name))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_innerMergeIndexMetadata<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    index_uuid: JString,
    index_type_code_jobj: jint,
    batch_readhead_jobj: JObject, // Optional<Integer>
) {
    ok_or_throw_without_return!(
        env,
        inner_merge_index_metadata(
            &mut env,
            java_dataset,
            index_uuid,
            index_type_code_jobj,
            batch_readhead_jobj
        )
    );
}

fn inner_merge_index_metadata(
    env: &mut JNIEnv,
    java_dataset: JObject,
    index_uuid: JString,
    index_type_code_jobj: jint,
    batch_readhead_jobj: JObject, // Optional<Integer>
) -> Result<()> {
    let (index_uuid, index_type, batch_readhead) = parse_merge_index_metadata_args(
        env,
        index_uuid,
        index_type_code_jobj,
        batch_readhead_jobj,
    )?;

    // Clone the inner Dataset out of the `get_rust_field` guard and drop the
    // guard before the long-lived merge. Otherwise nested JNI callbacks that
    // touch the same Dataset would deadlock on the native field mutex.
    let inner_dataset = unsafe {
        let dataset_guard =
            env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET)?;
        dataset_guard.inner.clone()
    };

    block_on(async {
        inner_dataset
            .merge_index_metadata(&index_uuid, index_type, batch_readhead, noop_progress())
            .await
    })?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_innerMergeIndexMetadataWithProgress<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    index_uuid: JString,
    index_type_code_jobj: jint,
    batch_readhead_jobj: JObject,
    progress_jobj: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_merge_index_metadata_with_progress(
            &mut env,
            java_dataset,
            index_uuid,
            index_type_code_jobj,
            batch_readhead_jobj,
            progress_jobj,
        )
    );
}

fn inner_merge_index_metadata_with_progress(
    env: &mut JNIEnv,
    java_dataset: JObject,
    index_uuid: JString,
    index_type_code_jobj: jint,
    batch_readhead_jobj: JObject,
    progress_jobj: JObject,
) -> Result<()> {
    let (index_uuid, index_type, batch_readhead) = parse_merge_index_metadata_args(
        env,
        index_uuid,
        index_type_code_jobj,
        batch_readhead_jobj,
    )?;
    let progress = Arc::new(JavaIndexBuildProgress::new(env, &progress_jobj)?);

    // Clone the inner Dataset out of the `get_rust_field` guard and drop the
    // guard before the long-lived merge. Progress callbacks are allowed to
    // re-enter Dataset JNI methods; holding the guard across those callbacks
    // would deadlock on the native field mutex (see update.rs).
    let inner_dataset = unsafe {
        let dataset_guard =
            env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET)?;
        dataset_guard.inner.clone()
    };

    block_on(async {
        inner_dataset
            .merge_index_metadata(&index_uuid, index_type, batch_readhead, progress)
            .await
    })?;
    Ok(())
}

fn parse_merge_index_metadata_args(
    env: &mut JNIEnv,
    index_uuid: JString,
    index_type_code_jobj: jint,
    batch_readhead_jobj: JObject,
) -> Result<(Uuid, IndexType, Option<usize>)> {
    let index_uuid_str = index_uuid.extract(env)?;
    let index_uuid = Uuid::parse_str(&index_uuid_str)
        .map_err(|e| Error::input_error(format!("Invalid UUID string for index_uuid: {e}")))?;
    let index_type = IndexType::try_from(index_type_code_jobj)?;
    let batch_readhead = env
        .get_int_opt(&batch_readhead_jobj)?
        .map(|val| val as usize);
    Ok((index_uuid, index_type, batch_readhead))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeMergeExistingIndexSegments<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    java_segments: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_merge_existing_index_segments(&mut env, java_dataset, java_segments)
    )
}

fn inner_merge_existing_index_segments<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    java_segments: JObject,
) -> Result<JObject<'local>> {
    let segments = import_vec_to_rust(env, &java_segments, |env, obj| obj.extract_object(env))?;
    let merged_segment = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.merge_existing_index_segments(segments))?
    };
    (&merged_segment).into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCommitExistingIndexSegments<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    index_name: JString,
    column: JString,
    java_segments: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_existing_index_segments(
            &mut env,
            java_dataset,
            index_name,
            column,
            java_segments
        )
    )
}

fn inner_commit_existing_index_segments<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    index_name: JString,
    column: JString,
    java_segments: JObject,
) -> Result<JObject<'local>> {
    let index_name = index_name.extract(env)?;
    let column = column.extract(env)?;
    let segment_metadata =
        import_vec_to_rust(env, &java_segments, |env, obj| obj.extract_object(env))?;
    let segments = segment_metadata
        .iter()
        .map(index_metadata_to_segment)
        .collect::<Result<Vec<_>>>()?;

    let committed = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.commit_existing_index_segments(
            &index_name,
            &column,
            segments,
        ))?;
        block_on(dataset_guard.inner.load_indices_by_name(&index_name))?
    };

    export_vec(env, &committed)
}

fn index_metadata_to_segment(metadata: &IndexMetadata) -> Result<IndexSegment> {
    Ok(metadata.clone().into_index_segment()?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeOptimizeIndices(
    mut env: JNIEnv,
    java_dataset: JObject,
    options_obj: JObject, // OptimizeOptions
) {
    ok_or_throw_without_return!(
        env,
        inner_optimize_indices(&mut env, java_dataset, options_obj)
    );
}

fn inner_optimize_indices(
    env: &mut JNIEnv,
    java_dataset: JObject,
    java_options: JObject, // OptimizeOptions
) -> Result<()> {
    let mut options = OptimizeOptions::default();

    if !java_options.is_null() {
        options.num_indices_to_merge =
            env.get_optional_usize_from_method(&java_options, "getNumIndicesToMerge")?;

        // getIndexNames(): Optional<List<String>>
        let index_names_obj = env
            .call_method(
                &java_options,
                "getIndexNames",
                "()Ljava/util/Optional;",
                &[],
            )?
            .l()?;
        let index_names = env.get_strings_opt(&index_names_obj)?;
        options.index_names = index_names;

        // isRetrain(): boolean
        let retrain = env
            .call_method(&java_options, "isRetrain", "()Z", &[])?
            .z()?;
        options.retrain = retrain;
    }

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.optimize_indices(&options))?;
    Ok(())
}

//////////////////
// Read Methods //
//////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_openNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    version_obj: JObject,    // Optional<Long>
    block_size_obj: JObject, // Optional<Integer>
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
    storage_options_obj: JObject,                  // Map<String, String>
    base_store_params_obj: JObject,                // Map<String, Map<String, String>>
    serialized_manifest: JObject,                  // Optional<ByteBuffer>
    session_handle: jlong,                         // Session handle, 0 means no session
    namespace_obj: JObject,                        // LanceNamespace object, null if no namespace
    table_id_obj: JObject,                         // List<String>, null if no namespace
    namespace_client_managed_versioning: jboolean, // Whether namespace manages versioning
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_open_native(
            &mut env,
            path,
            version_obj,
            block_size_obj,
            index_cache_size_bytes,
            metadata_cache_size_bytes,
            storage_options_obj,
            base_store_params_obj,
            serialized_manifest,
            session_handle,
            namespace_obj,
            table_id_obj,
            namespace_client_managed_versioning != 0,
        )
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_listManifestLocationsNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    storage_options_obj: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_list_manifest_locations(&mut env, path, storage_options_obj)
    )
}

fn inner_list_manifest_locations<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    storage_options_obj: JObject,
) -> Result<JObject<'local>> {
    let path: String = path.extract(env)?;
    let storage_options = JMap::from_env(env, &storage_options_obj)?;
    let storage_options = to_rust_map(env, &storage_options)?;
    let locations = BlockingDataset::list_manifest_locations(&path, storage_options)?;
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for location in locations {
        env.with_local_frame(8, |env| {
            let java_location = location.into_java(env)?;
            env.call_method(
                &list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&java_location)],
            )?;
            Ok::<(), Error>(())
        })?;
    }
    Ok(list)
}

#[allow(clippy::too_many_arguments)]
fn inner_open_native<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    version_obj: JObject,    // Optional<Long>
    block_size_obj: JObject, // Optional<Integer>
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
    storage_options_obj: JObject,              // Map<String, String>
    base_store_params_obj: JObject,            // Map<String, Map<String, String>>
    serialized_manifest: JObject,              // Optional<ByteBuffer>
    session_handle: jlong,                     // Session handle, 0 means no session
    namespace_obj: JObject,                    // LanceNamespace object, null if no namespace
    table_id_obj: JObject,                     // List<String>, null if no namespace
    namespace_client_managed_versioning: bool, // Whether namespace manages versioning
) -> Result<JObject<'local>> {
    let path_str: String = path.extract(env)?;
    let version = env.get_u64_opt(&version_obj)?;
    let block_size = env.get_int_opt(&block_size_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options = to_rust_map(env, &jmap)?;
    let base_store_params = extract_base_store_params(env, &base_store_params_obj)?;

    // Extract namespace and table_id if provided (before get_bytes_opt which holds borrow)
    let namespace_info = extract_namespace_info(env, &namespace_obj, &table_id_obj)?;
    let (namespace, table_id) = match namespace_info {
        Some((ns, tid)) => (Some(ns), Some(tid)),
        None => (None, None),
    };

    // When namespace is provided, automatically create a storage options provider
    // for credential refresh
    let storage_options_provider_arc: Option<Arc<dyn StorageOptionsProvider>> =
        if let (Some(ns), Some(tid)) = (namespace.clone(), table_id.clone()) {
            Some(Arc::new(LanceNamespaceStorageOptionsProvider::new(ns, tid)))
        } else {
            None
        };

    let serialized_manifest = env.get_bytes_opt(&serialized_manifest)?;

    // Convert session handle to Arc<LanceSession> if provided
    let session = session_from_handle(session_handle);

    let dataset = BlockingDataset::open(
        &path_str,
        version,
        block_size,
        index_cache_size_bytes,
        metadata_cache_size_bytes,
        storage_options,
        base_store_params,
        serialized_manifest,
        storage_options_provider_arc,
        session,
        namespace,
        table_id,
        namespace_client_managed_versioning,
    )?;
    dataset.into_java(env)
}

/// Check if the Java object is an instance of DirectoryNamespace.
fn is_directory_namespace(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<bool> {
    let class = env
        .find_class("org/lance/namespace/DirectoryNamespace")
        .map_err(|e| {
            Error::runtime_error(format!("Failed to find DirectoryNamespace class: {}", e))
        })?;
    env.is_instance_of(namespace_obj, class)
        .map_err(|e| Error::runtime_error(format!("Failed to check instanceof: {}", e)))
}

/// Check if the Java object is an instance of RestNamespace.
fn is_rest_namespace(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<bool> {
    let class = env
        .find_class("org/lance/namespace/RestNamespace")
        .map_err(|e| Error::runtime_error(format!("Failed to find RestNamespace class: {}", e)))?;
    env.is_instance_of(namespace_obj, class)
        .map_err(|e| Error::runtime_error(format!("Failed to check instanceof: {}", e)))
}

/// Get the native handle from a Java LanceNamespace object.
fn get_native_namespace_handle(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<jlong> {
    env.call_method(namespace_obj, "getNativeHandle", "()J", &[])
        .map_err(|e| Error::runtime_error(format!("Failed to call getNativeHandle: {}", e)))?
        .j()
        .map_err(|e| Error::runtime_error(format!("getNativeHandle did not return a long: {}", e)))
}

/// Extract namespace and table_id from Java objects into Rust types.
///
/// Returns `None` if `namespace_obj` is null, otherwise returns the namespace
/// and table_id pair.
#[allow(clippy::type_complexity)]
pub(crate) fn extract_namespace_info(
    env: &mut JNIEnv,
    namespace_obj: &JObject,
    table_id_obj: &JObject,
) -> Result<Option<(Arc<dyn LanceNamespace>, Vec<String>)>> {
    if namespace_obj.is_null() {
        return Ok(None);
    }

    let namespace_client: Arc<dyn LanceNamespace> = if is_directory_namespace(env, namespace_obj)? {
        let native_handle = get_native_namespace_handle(env, namespace_obj)?;
        let dir_namespace_client =
            unsafe { &*(native_handle as *const BlockingDirectoryNamespace) };
        dir_namespace_client.inner.clone()
    } else if is_rest_namespace(env, namespace_obj)? {
        let native_handle = get_native_namespace_handle(env, namespace_obj)?;
        let rest_namespace_client = unsafe { &*(native_handle as *const BlockingRestNamespace) };
        rest_namespace_client.inner.clone()
    } else {
        create_java_lance_namespace(env, namespace_obj)?
    };

    let table_id = env.get_strings(table_id_obj)?;
    Ok(Some((namespace_client, table_id)))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_getFragmentsNative<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
) -> JObject<'a> {
    ok_or_throw!(env, inner_get_fragments(&mut env, jdataset))
}

fn inner_get_fragments<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
) -> Result<JObject<'local>> {
    let fragments = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.inner.get_fragments()
    };
    let fragments = fragments
        .iter()
        .map(|f| f.metadata().clone())
        .collect::<Vec<Fragment>>();
    export_vec(env, &fragments)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetFragmentStatistics<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
) -> JObject<'a> {
    ok_or_throw!(env, inner_get_fragment_statistics(&mut env, jdataset))
}

/// Returns per-fragment statistics in their final Java primitive arrays.
///
/// Row count semantics match Java `FragmentMetadata.getNumRows()`:
/// physical rows minus deleted rows, with absent values treated as 0.
/// Data file count is the number of data files in the fragment.
fn inner_get_fragment_statistics<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
) -> Result<JObject<'local>> {
    let fragments = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.inner.fragments().clone()
    };
    let fragment_count = i32::try_from(fragments.len()).map_err(|_| {
        Error::runtime_error(format!(
            "Fragment statistics contain {} fragments, exceeding the Java array limit of {}",
            fragments.len(),
            i32::MAX
        ))
    })?;
    let ids = env.new_int_array(fragment_count)?;
    let row_counts = env.new_long_array(fragment_count)?;
    let data_file_nums = env.new_int_array(fragment_count)?;

    let mut id_values = Vec::with_capacity(fragments.len());
    let mut row_count_values = Vec::with_capacity(fragments.len());
    let mut data_file_num_values = Vec::with_capacity(fragments.len());

    for fragment in fragments.iter() {
        let physical_rows = fragment.physical_rows.unwrap_or(0) as i64;
        let deleted_rows = fragment
            .deletion_file
            .as_ref()
            .and_then(|deletion_file| deletion_file.num_deleted_rows)
            .unwrap_or(0) as i64;
        id_values.push(fragment.id as i32);
        row_count_values.push(physical_rows - deleted_rows);
        data_file_num_values.push(fragment.files.len() as i32);
    }

    env.set_int_array_region(&ids, 0, &id_values)?;
    env.set_long_array_region(&row_counts, 0, &row_count_values)?;
    env.set_int_array_region(&data_file_nums, 0, &data_file_num_values)?;

    Ok(env.new_object(
        "org/lance/FragmentStatistics",
        "([I[J[I)V",
        &[
            JValue::Object(&ids),
            JValue::Object(&row_counts),
            JValue::Object(&data_file_nums),
        ],
    )?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_getFragmentNative<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
    fragment_id: jint,
) -> JObject<'a> {
    ok_or_throw!(env, inner_get_fragment(&mut env, jdataset, fragment_id))
}

fn inner_get_fragment<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    fragment_id: jint,
) -> Result<JObject<'local>> {
    let fragment = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.inner.get_fragment(fragment_id as usize)
    };
    let obj = match fragment {
        Some(f) => f.metadata().into_java(env)?,
        None => JObject::default(),
    };
    Ok(obj)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetLanceSchema<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_lance_schema(&mut env, java_dataset))
}

fn inner_get_lance_schema<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let schema = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset.inner.schema().clone()
    };
    schema.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_importFfiSchema(
    mut env: JNIEnv,
    jdataset: JObject,
    arrow_schema_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_import_ffi_schema(&mut env, jdataset, arrow_schema_addr)
    )
}

fn inner_import_ffi_schema(
    env: &mut JNIEnv,
    jdataset: JObject,
    arrow_schema_addr: jlong,
) -> Result<()> {
    let schema = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        Schema::from(dataset.inner.schema())
    };

    let ffi_schema = FFI_ArrowSchema::try_from(&schema)?;
    unsafe { std::ptr::write_unaligned(arrow_schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeUri<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JString<'local> {
    ok_or_throw_with_return!(
        env,
        inner_uri(&mut env, java_dataset).map_err(|err| Error::input_error(err.to_string())),
        JString::from(JObject::null())
    )
}

fn inner_uri<'local>(env: &mut JNIEnv<'local>, java_dataset: JObject) -> Result<JString<'local>> {
    let uri = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.inner.uri().to_string()
    };

    let jstring_uri = env.new_string(uri)?;
    Ok(jstring_uri)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeListVersions<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_versions(&mut env, java_dataset))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetVersionCount(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_get_version_count(&mut env, java_dataset), -1) as jlong
}

fn inner_get_version_count(env: &mut JNIEnv, java_dataset: JObject) -> Result<u64> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.count_versions()
}

fn inner_list_versions<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let versions = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.list_versions()?
    };
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    versions
        .into_iter()
        .map(|inner_ver| inner_ver.into_java(env))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .try_for_each(|java_ver| -> Result<()> {
            env.call_method(
                &array_list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&java_ver)],
            )?;
            Ok(())
        })?;
    Ok(array_list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetVersion<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_version(&mut env, java_dataset))
}

fn inner_get_version<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let version = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.version()?
    };
    version.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetLatestVersionId(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_latest_version_id(&mut env, java_dataset), -1) as jlong
}

fn inner_latest_version_id(env: &mut JNIEnv, java_dataset: JObject) -> Result<u64> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.latest_version()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetInitialStorageOptions<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_get_initial_storage_options(&mut env, java_dataset)
    )
}

fn inner_get_initial_storage_options<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let storage_options = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.initial_storage_options()
    };
    match storage_options {
        Some(opts) => opts.into_java(env),
        None => Ok(JObject::null()),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetLatestStorageOptions<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_get_latest_storage_options(&mut env, java_dataset)
    )
}

fn inner_get_latest_storage_options<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let storage_options = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.latest_storage_options()?
    };
    match storage_options {
        Some(opts) => opts.into_java(env),
        None => Ok(JObject::null()),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutLatest(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_checkout_latest(&mut env, java_dataset));
}

fn inner_checkout_latest(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.checkout_latest()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutVersion<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    version: jlong,
) -> JObject<'local> {
    ok_or_throw!(env, inner_checkout_version(&mut env, java_dataset, version))
}

fn inner_checkout_version<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    version: jlong,
) -> Result<JObject<'local>> {
    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_version(version as u64)?
    };

    new_dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutTag<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jtag: JString,
) -> JObject<'local> {
    ok_or_throw!(env, inner_checkout_tag(&mut env, java_dataset, jtag))
}

fn inner_checkout_tag<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jtag_name: JString,
) -> Result<JObject<'local>> {
    let tag_name = jtag_name.extract(env)?;
    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_tag(tag_name.as_str())?
    };

    new_dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeRestore(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_restore(&mut env, java_dataset))
}

fn inner_restore(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.restore()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeShallowClone<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    target_path: JString,
    reference: JObject,
    storage_options: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_shallow_clone(
            &mut env,
            java_dataset,
            target_path,
            reference,
            storage_options
        )
    )
}

fn inner_shallow_clone<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    target_path: JString,
    jref: JObject,
    storage_options: JObject,
) -> Result<JObject<'local>> {
    let target_path_str = target_path.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let storage_opts = transform_jstorage_options(storage_options, env)?;
    let new_ds = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.shallow_clone(
            target_path_str.as_str(),
            reference,
            storage_opts,
        ))?
    };

    BlockingDataset { inner: new_ds }.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCountRows(
    mut env: JNIEnv,
    java_dataset: JObject,
    filter_jobj: JObject, // Optional<String>
) -> jlong {
    ok_or_throw_with_return!(
        env,
        inner_count_rows(&mut env, java_dataset, filter_jobj),
        -1
    ) as jlong
}

fn inner_count_rows(
    env: &mut JNIEnv,
    java_dataset: JObject,
    filter_jobj: JObject,
) -> Result<usize> {
    let filter = env.get_string_opt(&filter_jobj)?;
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.count_rows(filter)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetDataStatistics<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_data_statistics(&mut env, java_dataset))
}

fn inner_get_data_statistics<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let stats = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.calculate_data_stats()?
    };
    let data_stats = env.new_object("org/lance/ipc/DataStatistics", "()V", &[])?;

    for field in stats.fields {
        let id = field.id as jint;
        let byte_size = field.bytes_on_disk as jlong;
        let filed_jobj = env.new_object(
            "org/lance/ipc/FieldStatistics",
            "(IJ)V",
            &[JValue::Int(id), JValue::Long(byte_size)],
        )?;
        env.call_method(
            &data_stats,
            "addFieldStatistics",
            "(Lorg/lance/ipc/FieldStatistics;)V",
            &[JValue::Object(&filed_jobj)],
        )?;
    }
    Ok(data_stats)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeListIndexes<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_indexes(&mut env, java_dataset))
}

fn inner_list_indexes<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let index_names = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let indexes = dataset_guard.list_indexes()?;
        indexes
            .iter()
            .map(|index| index.name.clone())
            .collect::<Vec<String>>()
    };

    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for name in index_names {
        let java_string = env.new_string(&name)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&java_string)],
        )?;
    }

    Ok(array_list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetConfig<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_config(&mut env, java_dataset))
}

fn inner_get_config<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let config = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.inner.config().clone()
    };

    let java_hashmap = env
        .new_object("java/util/HashMap", "()V", &[])
        .expect("Failed to create Java HashMap");

    for (k, v) in config {
        let java_key = env
            .new_string(&k)
            .expect("Failed to create Java String (key)");
        let java_value = env
            .new_string(&v)
            .expect("Failed to create Java String (value)");

        env.call_method(
            &java_hashmap,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&java_key), JValue::Object(&java_value)],
        )
        .expect("Failed to call HashMap.put()");
    }

    Ok(java_hashmap)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeHasStableRowIds(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jboolean {
    ok_or_throw_with_return!(env, inner_has_stable_row_ids(&mut env, java_dataset), 0u8)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetWriterVersion<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_writer_version(&mut env, java_dataset))
}

fn inner_get_writer_version<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let writer_version = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.inner.manifest().writer_version.clone()
    };
    match writer_version {
        Some(writer_version) => writer_version.into_java(env),
        None => Ok(JObject::null()),
    }
}

fn inner_has_stable_row_ids(env: &mut JNIEnv, java_dataset: JObject) -> Result<u8> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    Ok(dataset_guard.inner.manifest().uses_stable_row_ids() as u8)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetLanceFileFormatVersion<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JString<'local> {
    ok_or_throw_with_return!(
        env,
        inner_get_lance_file_format_version(&mut env, java_dataset),
        JObject::null().into()
    )
}

fn inner_get_lance_file_format_version<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JString<'local>> {
    let version_string = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard
            .inner
            .manifest()
            .data_storage_format
            .lance_file_format()
            .to_manifest_string()
            .to_string()
    };

    Ok(env
        .new_string(&version_string)
        .expect("Failed to create Java String"))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeTake(
    mut env: JNIEnv,
    java_dataset: JObject,
    indices_obj: JObject, // List<Long>
    columns_obj: JObject, // List<String>
) -> jbyteArray {
    match inner_take(&mut env, java_dataset, indices_obj, columns_obj) {
        Ok(byte_array) => byte_array,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", format!("{:?}", e));
            std::ptr::null_mut()
        }
    }
}

fn inner_take(
    env: &mut JNIEnv,
    java_dataset: JObject,
    indices_obj: JObject, // List<Long>
    columns_obj: JObject, // List<String>
) -> Result<jbyteArray> {
    let indices: Vec<i64> = env.get_longs(&indices_obj)?;
    let indices_u64: Vec<u64> = indices.iter().map(|&x| x as u64).collect();
    let indices_slice: &[u64] = &indices_u64;
    let columns: Vec<String> = env.get_strings(&columns_obj)?;

    let result = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let dataset = &dataset_guard.inner;

        let projection = ProjectionRequest::from_columns(columns, dataset.schema());

        match block_on(dataset.take(indices_slice, projection)) {
            Ok(res) => res,
            Err(e) => {
                return Err(e.into());
            }
        }
    };

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &result.schema())?;
        writer.write(&result)?;
        writer.finish()?;
    }

    let byte_array = env.byte_array_from_slice(&buffer)?;
    Ok(**byte_array)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeTakeRows(
    mut env: JNIEnv,
    java_dataset: JObject,
    row_ids_obj: JObject, // List<Long>
    columns_obj: JObject, // List<String>
) -> jbyteArray {
    match inner_take_rows(&mut env, java_dataset, row_ids_obj, columns_obj) {
        Ok(byte_array) => byte_array,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", format!("{:?}", e));
            std::ptr::null_mut()
        }
    }
}

fn inner_take_rows(
    env: &mut JNIEnv,
    java_dataset: JObject,
    row_ids_obj: JObject, // List<Long>
    columns_obj: JObject, // List<String>
) -> Result<jbyteArray> {
    let row_ids: Vec<i64> = env.get_longs(&row_ids_obj)?;
    let row_ids_u64: Vec<u64> = row_ids.iter().map(|&x| x as u64).collect();
    let columns: Vec<String> = env.get_strings(&columns_obj)?;

    let result = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let dataset = &dataset_guard.inner;

        let projection = ProjectionRequest::from_columns(columns, dataset.schema());

        match block_on(dataset.take_rows(&row_ids_u64, projection)) {
            Ok(res) => res,
            Err(e) => {
                return Err(e.into());
            }
        }
    };

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &result.schema())?;
        writer.write(&result)?;
        writer.finish()?;
    }

    let byte_array = env.byte_array_from_slice(&buffer)?;
    Ok(**byte_array)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeSample(
    mut env: JNIEnv,
    java_dataset: JObject,
    n: jlong,
    columns_obj: JObject,      // List<String>
    fragment_ids_obj: JObject, // Optional<List<Integer>>
) -> jbyteArray {
    match inner_sample(&mut env, java_dataset, n, columns_obj, fragment_ids_obj) {
        Ok(byte_array) => byte_array,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", format!("{:?}", e));
            std::ptr::null_mut()
        }
    }
}

fn inner_sample(
    env: &mut JNIEnv,
    java_dataset: JObject,
    n: jlong,
    columns_obj: JObject,      // List<String>
    fragment_ids_obj: JObject, // Optional<List<Integer>>
) -> Result<jbyteArray> {
    let columns: Vec<String> = env.get_strings(&columns_obj)?;
    let fragment_ids: Option<Vec<i32>> = env.get_ints_opt(&fragment_ids_obj)?;
    let fragment_ids_u32: Option<Vec<u32>> =
        fragment_ids.map(|ids| ids.iter().map(|&id| id as u32).collect());

    let result = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let dataset = &dataset_guard.inner;

        let projection = dataset
            .schema()
            .project_preserve_system_columns(&columns)
            .map_err(|e| Error::runtime_error(e.to_string()))?;

        match block_on(dataset.sample(n as usize, &projection, fragment_ids_u32.as_deref())) {
            Ok(res) => res,
            Err(e) => {
                return Err(e.into());
            }
        }
    };

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &result.schema())?;
        writer.write(&result)?;
        writer.finish()?;
    }

    let byte_array = env.byte_array_from_slice(&buffer)?;
    Ok(**byte_array)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDelete(
    mut env: JNIEnv,
    java_dataset: JObject,
    predicate: JString,
) {
    ok_or_throw_without_return!(env, inner_delete(&mut env, java_dataset, predicate))
}

fn inner_delete(env: &mut JNIEnv, java_dataset: JObject, predicate: JString) -> Result<()> {
    let predicate_str = predicate.extract(env)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.delete(&predicate_str))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeTruncateTable(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_truncate_table(&mut env, java_dataset))
}

fn inner_truncate_table(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.truncate_table())?;
    Ok(())
}

//////////////////////////////
// Schema evolution Methods //
//////////////////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDropColumns(
    mut env: JNIEnv,
    java_dataset: JObject,
    columns_obj: JObject, // List<String>
) {
    ok_or_throw_without_return!(env, inner_drop_columns(&mut env, java_dataset, columns_obj))
}

fn inner_drop_columns(
    env: &mut JNIEnv,
    java_dataset: JObject,
    columns_obj: JObject, // List<String>
) -> Result<()> {
    let columns: Vec<String> = env.get_strings(&columns_obj)?;
    let columns_slice: Vec<&str> = columns.iter().map(AsRef::as_ref).collect();
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.drop_columns(&columns_slice))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeAlterColumns(
    mut env: JNIEnv,
    java_dataset: JObject,
    column_alterations_obj: JObject, // List<ColumnAlteration>
    cast_schema_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_alter_columns(
            &mut env,
            java_dataset,
            column_alterations_obj,
            cast_schema_addr
        )
    )
}

fn create_column_alteration(
    env: &mut JNIEnv,
    column_alteration_jobj: JObject, // ColumnAlteration
) -> Result<(ColumnAlteration, bool)> {
    let path_obj = env
        .get_field(&column_alteration_jobj, "path", "Ljava/lang/String;")?
        .l()?;
    let path_jstring: JString = path_obj.into();
    let path: String = env.get_string(&path_jstring)?.into();

    let rename_obj = env
        .get_field(&column_alteration_jobj, "rename", "Ljava/util/Optional;")?
        .l()?;
    let rename = if env.call_method(&rename_obj, "isPresent", "()Z", &[])?.z()? {
        let jstring: JObject = env
            .call_method(rename_obj, "get", "()Ljava/lang/Object;", &[])?
            .l()?;
        let jstring: JString = jstring.into();
        let rename_str: String = env.get_string(&jstring)?.into(); // Intermediate variable
        Some(rename_str)
    } else {
        None
    };

    let nullable_obj = env
        .get_field(&column_alteration_jobj, "nullable", "Ljava/util/Optional;")?
        .l()?;
    let nullable = if env
        .call_method(&nullable_obj, "isPresent", "()Z", &[])?
        .z()?
    {
        let nullable_value = env
            .call_method(nullable_obj, "get", "()Ljava/lang/Object;", &[])?
            .l()?;
        Some(
            env.call_method(nullable_value, "booleanValue", "()Z", &[])?
                .z()?,
        )
    } else {
        None
    };

    // The cast target type (if any) is not read here: it is transferred separately through the
    // Arrow C Data Interface (see inner_alter_columns), because ArrowType#toString() does not
    // round-trip through DataType::from_str for parameterized types. This flag records whether a
    // cast was requested so the caller can attach the imported type in order.
    let data_type_obj = env
        .get_field(&column_alteration_jobj, "dataType", "Ljava/util/Optional;")?
        .l()?;
    let wants_cast = env
        .call_method(&data_type_obj, "isPresent", "()Z", &[])?
        .z()?;

    let alteration = ColumnAlteration {
        path,
        rename,
        nullable,
        data_type: None,
    };
    Ok((alteration, wants_cast))
}

fn inner_alter_columns(
    env: &mut JNIEnv,
    java_dataset: JObject,
    column_alterations_obj: JObject, // List<ColumnAlteration>
    cast_schema_addr: jlong,
) -> Result<()> {
    let list = env.get_list(&column_alterations_obj)?;
    let mut iter = list.iter(env)?;
    let mut column_alterations = Vec::new();
    let mut cast_flags = Vec::new();

    while let Some(elem) = iter.next(env)? {
        let (alteration, wants_cast) = create_column_alteration(env, elem)?;
        column_alterations.push(alteration);
        cast_flags.push(wants_cast);
    }

    // Cast target types arrive as one Arrow schema field per requested cast, in the same order
    // as the alterations that requested one.
    let cast_schema = unsafe { FFI_ArrowSchema::from_raw(cast_schema_addr as *mut _) };
    let cast_schema = ArrowSchema::try_from(&cast_schema)
        .map_err(|_| Error::input_error("ArrowSchema conversion error".to_string()))?;
    let mut cast_types = cast_schema.fields.iter().map(|f| f.data_type().clone());
    for (alteration, wants_cast) in column_alterations.iter_mut().zip(cast_flags) {
        if wants_cast {
            alteration.data_type = Some(cast_types.next().ok_or_else(|| {
                Error::input_error("Missing cast type for column alteration".to_string())
            })?);
        }
    }

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    block_on(dataset_guard.inner.alter_columns(&column_alterations))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeAddColumnsBySqlExpressions(
    mut env: JNIEnv,
    java_dataset: JObject,
    sql_expressions: JObject, // SqlExpressions
    batch_size: JObject,      // Optional<Long>
) {
    ok_or_throw_without_return!(
        env,
        inner_add_columns_by_sql_expressions(&mut env, java_dataset, sql_expressions, batch_size)
    )
}

fn inner_add_columns_by_sql_expressions(
    env: &mut JNIEnv,
    java_dataset: JObject,
    sql_expressions: JObject, // SqlExpressions
    batch_size: JObject,      // Optional<Long>
) -> Result<()> {
    let sql_expressions_obj = env
        .get_field(sql_expressions, "sqlExpressions", "Ljava/util/List;")?
        .l()?;

    let sql_expressions_obj_list = env.get_list(&sql_expressions_obj)?;
    let mut expressions: Vec<(String, String)> = Vec::new();

    let mut iterator = sql_expressions_obj_list.iter(env)?;

    while let Some(item) = iterator.next(env)? {
        let name = env
            .call_method(&item, "getName", "()Ljava/lang/String;", &[])?
            .l()?;
        let value = env
            .call_method(&item, "getExpression", "()Ljava/lang/String;", &[])?
            .l()?;
        let key_str: String = env.get_string(&JString::from(name))?.into();
        let value_str: String = env.get_string(&JString::from(value))?.into();
        expressions.push((key_str, value_str));
    }

    let rust_transform = NewColumnTransform::SqlExpressions(expressions);

    let batch_size = match env.get_long_opt(&batch_size)? {
        Some(value) => Some(
            value
                .try_into()
                .map_err(|_| Error::input_error("Batch size conversion error".to_string()))?,
        ),
        None => None,
    };

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    block_on(
        dataset_guard
            .inner
            .add_columns(rust_transform, None, batch_size),
    )?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeAddColumnsByReader(
    mut env: JNIEnv,
    java_dataset: JObject,
    arrow_array_stream_addr: jlong,
    batch_size: JObject, // Optional<Long>
) {
    ok_or_throw_without_return!(
        env,
        inner_add_columns_by_reader(&mut env, java_dataset, arrow_array_stream_addr, batch_size)
    )
}

fn inner_add_columns_by_reader(
    env: &mut JNIEnv,
    java_dataset: JObject,
    arrow_array_stream_addr: jlong,
    batch_size: JObject, // Optional<Long>
) -> Result<()> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;

    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;

    let transform = NewColumnTransform::Reader(Box::new(reader));

    let batch_size = match env.get_long_opt(&batch_size)? {
        Some(value) => Some(
            value
                .try_into()
                .map_err(|_| Error::input_error("Batch size conversion error".to_string()))?,
        ),
        None => None,
    };

    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    block_on(dataset_guard.inner.add_columns(transform, None, batch_size))?;

    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeAddColumnsBySchema(
    mut env: JNIEnv,
    java_dataset: JObject,
    schema_ptr: jlong, // Schema pointer
) {
    ok_or_throw_without_return!(
        env,
        inner_add_columns_by_schema(&mut env, java_dataset, schema_ptr)
    )
}

fn inner_add_columns_by_schema(
    env: &mut JNIEnv,
    java_dataset: JObject,
    schema_ptr: jlong,
) -> Result<()> {
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(schema_ptr as *mut _) };

    let schema = ArrowSchema::try_from(&c_schema)
        .map_err(|_| Error::input_error("ArrowSchema conversion error".to_string()))?;

    let transform = NewColumnTransform::AllNulls(Arc::new(schema));
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    block_on(dataset_guard.inner.add_columns(transform, None, None))?;

    Ok(())
}

//////////////////////////////
// Tag operation Methods    //
//////////////////////////////
fn optional_datetime_to_java_instant<'local>(
    env: &mut JNIEnv<'local>,
    timestamp: Option<&DateTime<Utc>>,
) -> Result<JObject<'local>> {
    if let Some(timestamp) = timestamp {
        let seconds = timestamp.timestamp();
        let nanos = timestamp.timestamp_subsec_nanos() as i64;
        Ok(env
            .call_static_method(
                "java/time/Instant",
                "ofEpochSecond",
                "(JJ)Ljava/time/Instant;",
                &[JValue::Long(seconds), JValue::Long(nanos)],
            )?
            .l()?)
    } else {
        Ok(JObject::null())
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeListTags<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_tags(&mut env, java_dataset))
}

fn inner_list_tags<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let mut tags: Vec<_> = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.list_tags()?.into_iter().collect()
    };
    tags.sort_unstable_by(|(left_name, _), (right_name, _)| left_name.cmp(right_name));
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for (tag_name, tag_contents) in tags {
        let branch_name: JObject = if let Some(branch_name) = tag_contents.branch.as_ref() {
            env.new_string(branch_name)?.into()
        } else {
            JObject::null()
        };
        let created_at = optional_datetime_to_java_instant(env, tag_contents.created_at.as_ref())?;
        let updated_at = optional_datetime_to_java_instant(env, tag_contents.updated_at.as_ref())?;
        let java_metadata = to_java_map(env, &tag_contents.metadata)?;
        let java_tag = env.new_object(
            "org/lance/Tag",
            "(Ljava/lang/String;Ljava/lang/String;JILjava/time/Instant;Ljava/time/Instant;Ljava/util/Map;)V",
            &[
                JValue::Object(&env.new_string(tag_name)?.into()),
                JValue::Object(&branch_name),
                JValue::Long(tag_contents.version as i64),
                JValue::Int(tag_contents.manifest_size as i32),
                JValue::Object(&created_at),
                JValue::Object(&updated_at),
                JValue::Object(&java_metadata),
            ],
        )?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&java_tag)],
        )?;
    }
    Ok(array_list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCreateTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_create_tag(&mut env, java_dataset, jtag_name, jref)
    )
}

fn inner_create_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.create_tag(tag.as_str(), reference)?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDeleteTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) {
    ok_or_throw_without_return!(env, inner_delete_tag(&mut env, java_dataset, jtag_name))
}

fn inner_delete_tag(env: &mut JNIEnv, java_dataset: JObject, jtag_name: JString) -> Result<()> {
    let tag = { jtag_name.extract(env)? };
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.delete_tag(tag.as_str())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeUpdateTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_update_tag(&mut env, java_dataset, jtag_name, jref)
    )
}

fn inner_update_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.update_tag(tag.as_str(), reference)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeReplaceTagMetadata(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jmetadata: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_replace_tag_metadata(&mut env, java_dataset, jtag_name, jmetadata)
    )
}

fn inner_replace_tag_metadata(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jmetadata: JObject,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let metadata = extract_metadata_map(env, &jmetadata)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.replace_tag_metadata(tag.as_str(), metadata)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetVersionByTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        inner_get_version_by_tag(&mut env, java_dataset, jtag_name),
        -1
    ) as jlong
}

fn inner_get_version_by_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) -> Result<u64> {
    let tag = { jtag_name.extract(env)? };
    let dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.get_version(tag.as_str())
}

//////////////////////////////
// Branch operation Methods  //
//////////////////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeListBranches<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_branches(&mut env, java_dataset))
}

fn inner_list_branches<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let mut branches: Vec<_> = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.list_branches()?.into_iter().collect()
    };
    branches.sort_unstable_by(|(left_name, _), (right_name, _)| left_name.cmp(right_name));
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for (name, contents) in branches {
        let jname = env.new_string(name)?;
        let jparent = if let Some(p) = contents.parent_branch {
            env.new_string(p)?.into()
        } else {
            JObject::null()
        };
        let jbranch_identifier = env.new_object(
            "java/util/ArrayList",
            "(I)V",
            &[JValue::Int(contents.identifier.version_mapping.len() as i32)],
        )?;
        for (version, uuid) in contents.identifier.version_mapping.iter() {
            let juuid = env.new_string(uuid)?;
            let jmapping = env.new_object(
                "org/lance/Branch$BranchVersionMapping",
                "(JLjava/lang/String;)V",
                &[JValue::Long(*version as i64), JValue::Object(&juuid)],
            )?;
            env.call_method(
                &jbranch_identifier,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&jmapping)],
            )?;
        }
        let java_metadata = to_java_map(env, &contents.metadata)?;
        let jbranch = env.new_object(
            "org/lance/Branch",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/util/List;JJILjava/util/Map;)V",
            &[
                JValue::Object(&jname),
                JValue::Object(&jparent),
                JValue::Object(&jbranch_identifier),
                JValue::Long(contents.parent_version as i64),
                JValue::Long(contents.create_at as i64),
                JValue::Int(contents.manifest_size as i32),
                JValue::Object(&java_metadata),
            ],
        )?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&jbranch)],
        )?;
    }
    Ok(array_list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCreateBranch<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jref: JObject,
    jstorage_options: JObject, // Optional<String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_branch(&mut env, java_dataset, jbranch, jref, jstorage_options)
    )
}

fn inner_create_branch<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jref: JObject,
    jstorage_options: JObject, // Optional<String>
) -> Result<JObject<'local>> {
    let branch_name: String = jbranch.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let storage_opts = transform_jstorage_options(jstorage_options, env)?;

    let new_blocking_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let inner = block_on(dataset_guard.inner.create_branch(
            branch_name.as_str(),
            reference,
            storage_opts,
        ))?;
        BlockingDataset { inner }
    };
    new_blocking_dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeReplaceBranchMetadata(
    mut env: JNIEnv,
    java_dataset: JObject,
    jbranch: JString,
    jmetadata: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_replace_branch_metadata(&mut env, java_dataset, jbranch, jmetadata)
    )
}

fn inner_replace_branch_metadata(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jbranch: JString,
    jmetadata: JObject,
) -> Result<()> {
    let branch: String = jbranch.extract(env)?;
    let metadata = extract_metadata_map(env, &jmetadata)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.replace_branch_metadata(&branch, metadata)
}

fn extract_metadata_map(env: &mut JNIEnv, jmetadata: &JObject) -> Result<HashMap<String, String>> {
    let jmap = JMap::from_env(env, jmetadata)?;
    to_rust_map(env, &jmap)
}

fn transform_jref_to_ref(jref: JObject, env: &mut JNIEnv) -> Result<Ref> {
    let source_tag_name = env.get_optional_string_from_method(&jref, "getTagName")?;
    let source_version_number = env.get_optional_u64_from_method(&jref, "getVersionNumber")?;
    let source_branch = env.get_optional_string_from_method(&jref, "getBranchName")?;
    if let Some(tag_name) = source_tag_name {
        Ok(Ref::Tag(tag_name))
    } else {
        Ok(Ref::Version(source_branch, source_version_number))
    }
}

fn transform_jstorage_options(
    jstorage_options: JObject,
    env: &mut JNIEnv,
) -> Result<Option<ObjectStoreParams>> {
    let storage_options = env.get_optional(&jstorage_options, |env, map_obj| {
        let jmap = JMap::from_env(env, &map_obj)?;
        to_rust_map(env, &jmap)
    })?;
    Ok(storage_options
        .map(|options| {
            Some(ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance::io::StorageOptionsAccessor::with_static_options(options),
                )),
                ..Default::default()
            })
        })
        .unwrap_or(None))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDeleteBranch(
    mut env: JNIEnv,
    java_dataset: JObject,
    jbranch: JString,
) {
    ok_or_throw_without_return!(env, inner_delete_branch(&mut env, java_dataset, jbranch))
}

fn inner_delete_branch(env: &mut JNIEnv, java_dataset: JObject, jbranch: JString) -> Result<()> {
    let branch_name: String = jbranch.extract(env)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.delete_branch(branch_name.as_str())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckout<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    reference_obj: JObject, // Reference
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_checkout_ref(&mut env, java_dataset, reference_obj)
    )
}

fn inner_checkout_ref<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    reference_obj: JObject, // Reference
) -> Result<JObject<'local>> {
    // Extract Optional fields from Reference
    let branch_opt_obj = env
        .call_method(
            &reference_obj,
            "getBranchName",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let version_opt_obj = env
        .call_method(
            &reference_obj,
            "getVersionNumber",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let tag_opt_obj = env
        .call_method(&reference_obj, "getTagName", "()Ljava/util/Optional;", &[])?
        .l()?;

    let branch_opt = env.get_string_opt(&branch_opt_obj)?;
    let version_opt = env.get_u64_opt(&version_opt_obj)?;
    let tag_opt = env.get_string_opt(&tag_opt_obj)?;

    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_reference(branch_opt, version_opt, tag_opt)?
    };
    new_dataset.into_java(env)
}

// Unified metadata API JNI methods

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetTableMetadata<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_table_metadata(&mut env, java_dataset))
}

fn inner_get_table_metadata<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let table_metadata = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.get_table_metadata()?
    };

    let java_hashmap = env
        .new_object("java/util/HashMap", "()V", &[])
        .expect("Failed to create Java HashMap");

    for (k, v) in table_metadata {
        let java_key = env
            .new_string(&k)
            .expect("Failed to create Java String (key)");
        let java_value = env
            .new_string(&v)
            .expect("Failed to create Java String (value)");

        env.call_method(
            &java_hashmap,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&java_key), JValue::Object(&java_value)],
        )
        .expect("Failed to call HashMap.put()");
    }

    Ok(java_hashmap)
}

//////////////////////////////
// Compaction Methods       //
//////////////////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCompact(
    mut env: JNIEnv,
    java_dataset: JObject,
    compaction_options: JObject, // CompactionOptions
) {
    ok_or_throw_without_return!(
        env,
        inner_compact(&mut env, java_dataset, compaction_options)
    )
}

fn inner_compact(
    env: &mut JNIEnv,
    java_dataset: JObject,
    compaction_options: JObject, // CompactionOptions
) -> Result<()> {
    let config = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset.inner.manifest.config.clone()
    };
    let rust_options = convert_java_compaction_options_to_rust(env, compaction_options, &config)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.compact(rust_options)?;
    Ok(())
}

fn convert_java_compaction_options_to_rust(
    env: &mut JNIEnv,
    java_options: JObject,
    config: &std::collections::HashMap<String, String>,
) -> Result<RustCompactionOptions> {
    let target_rows_per_fragment = env
        .call_method(
            &java_options,
            "getTargetRowsPerFragment",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_rows_per_group = env
        .call_method(
            &java_options,
            "getMaxRowsPerGroup",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_bytes_per_file = env
        .call_method(
            &java_options,
            "getMaxBytesPerFile",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let materialize_deletions = env
        .call_method(
            &java_options,
            "getMaterializeDeletions",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let materialize_deletions_threshold = env
        .call_method(
            &java_options,
            "getMaterializeDeletionsThreshold",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let num_threads = env
        .call_method(
            &java_options,
            "getNumThreads",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let batch_size = env
        .call_method(&java_options, "getBatchSize", "()Ljava/util/Optional;", &[])?
        .l()?;
    let defer_index_remap = env
        .call_method(
            &java_options,
            "getDeferIndexRemap",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let compaction_mode = env
        .call_method(
            &java_options,
            "getCompactionMode",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let binary_copy_read_batch_bytes = env
        .call_method(
            &java_options,
            "getBinaryCopyReadBatchBytes",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_source_fragments = env
        .call_method(
            &java_options,
            "getMaxSourceFragments",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_source_rows = env
        .call_method(
            &java_options,
            "getMaxSourceRows",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_source_bytes = env
        .call_method(
            &java_options,
            "getMaxSourceBytes",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let excluded_fragment_ids = env
        .call_method(
            &java_options,
            "getExcludedFragmentIds",
            "()Ljava/util/List;",
            &[],
        )?
        .l()?;

    build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
        &compaction_mode,
        &binary_copy_read_batch_bytes,
        &max_source_fragments,
        &max_source_rows,
        &max_source_bytes,
        &excluded_fragment_ids,
        config,
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCleanupWithPolicy<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_cleanup_with_policy(&mut env, jdataset, jpolicy))
}

fn inner_cleanup_with_policy<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
) -> Result<JObject<'local>> {
    let policy = extract_cleanup_policy(env, &jpolicy)?;

    let stats = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.cleanup_with_policy(policy)
    }?;

    cleanup_stats_to_java(env, stats)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeExplainCleanupWithPolicy<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
    jmax_candidate_files: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_explain_cleanup_with_policy(&mut env, jdataset, jpolicy, jmax_candidate_files)
    )
}

fn inner_explain_cleanup_with_policy<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
    jmax_candidate_files: JObject,
) -> Result<JObject<'local>> {
    let policy = extract_cleanup_policy(env, &jpolicy)?;
    let max_candidate_files = env
        .get_optional(&jmax_candidate_files, |env, inner| {
            Ok(env.call_method(inner, "longValue", "()J", &[])?.j()?)
        })?
        .map(|v| {
            usize::try_from(v).map_err(|e| {
                Error::input_error(format!(
                    "maxCandidateFiles must be a non-negative usize value, got {}: {:?}",
                    v, e
                ))
            })
        })
        .transpose()?;

    let explanation = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.explain_cleanup_with_policy(policy, max_candidate_files)
    }?;

    cleanup_explanation_to_java(env, explanation)
}

fn extract_cleanup_policy(env: &mut JNIEnv<'_>, jpolicy: &JObject) -> Result<CleanupPolicy> {
    let before_ts_millis = env.get_optional_u64_from_method(jpolicy, "getBeforeTimestampMillis")?;
    let before_timestamp = before_ts_millis.map(|millis| {
        let st = UNIX_EPOCH + Duration::from_millis(millis);
        DateTime::<Utc>::from(st)
    });

    let before_version = env.get_optional_u64_from_method(jpolicy, "getBeforeVersion")?;

    let delete_unverified = env
        .get_optional_from_method(jpolicy, "getDeleteUnverified", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(false);

    let error_if_tagged_old_versions = env
        .get_optional_from_method(jpolicy, "getErrorIfTaggedOldVersions", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(true);

    let clean_referenced_branches = env
        .get_optional_from_method(jpolicy, "getCleanReferencedBranches", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(false);

    let delete_rate_limit = env.get_optional_u64_from_method(jpolicy, "getDeleteRateLimit")?;

    Ok(CleanupPolicy {
        before_timestamp,
        before_version,
        delete_unverified,
        error_if_tagged_old_versions,
        clean_referenced_branches,
        delete_rate_limit,
    })
}

fn cleanup_stats_to_java<'local>(
    env: &mut JNIEnv<'local>,
    stats: RemovalStats,
) -> Result<JObject<'local>> {
    Ok(env.new_object(
        "org/lance/cleanup/RemovalStats",
        "(JJJJJJ)V",
        &[
            JValue::Long(stats.bytes_removed as i64),
            JValue::Long(stats.old_versions as i64),
            JValue::Long(stats.data_files_removed as i64),
            JValue::Long(stats.transaction_files_removed as i64),
            JValue::Long(stats.index_files_removed as i64),
            JValue::Long(stats.deletion_files_removed as i64),
        ],
    )?)
}

fn cleanup_file_kind_to_java(kind: CleanupFileKind) -> &'static str {
    match kind {
        CleanupFileKind::Manifest => "manifest",
        CleanupFileKind::Data => "data",
        CleanupFileKind::Transaction => "transaction",
        CleanupFileKind::Index => "index",
        CleanupFileKind::Deletion => "deletion",
        CleanupFileKind::TemporaryManifest => "temporary_manifest",
    }
}

fn cleanup_candidate_files_to_java<'local>(
    env: &mut JNIEnv<'local>,
    files: Vec<CleanupCandidateFile>,
) -> Result<JObject<'local>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    // Wrap each iteration in a local frame so the temporary path/kind/candidate
    // references do not accumulate on the JNI local reference table for large
    // explanations (default limit is 1000 candidate files, but users can raise it).
    for file in files {
        env.with_local_frame(8, |env| {
            let path = env.new_string(file.path)?;
            let kind = env.new_string(cleanup_file_kind_to_java(file.kind))?;
            let candidate = env.new_object(
                "org/lance/cleanup/CleanupCandidateFile",
                "(Ljava/lang/String;Ljava/lang/String;ZJ)V",
                &[
                    JValue::Object(&path),
                    JValue::Object(&kind),
                    JValue::Bool(file.unverified as jboolean),
                    JValue::Long(file.size_bytes as i64),
                ],
            )?;
            env.call_method(
                &list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&candidate)],
            )?;
            Ok::<(), Error>(())
        })?;
    }
    Ok(list)
}

fn cleanup_referenced_branches_to_java<'local>(
    env: &mut JNIEnv<'local>,
    branches: Vec<CleanupReferencedBranch>,
) -> Result<JObject<'local>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for branch in branches {
        env.with_local_frame(8, |env| {
            let name = env.new_string(branch.name)?;
            let referenced_branch = env.new_object(
                "org/lance/cleanup/CleanupReferencedBranch",
                "(Ljava/lang/String;JZ)V",
                &[
                    JValue::Object(&name),
                    JValue::Long(branch.referenced_version as i64),
                    JValue::Bool(branch.cleanup_candidate as jboolean),
                ],
            )?;
            env.call_method(
                &list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&referenced_branch)],
            )?;
            Ok::<(), Error>(())
        })?;
    }
    Ok(list)
}

fn cleanup_warnings_to_java<'local>(
    env: &mut JNIEnv<'local>,
    warnings: Vec<String>,
) -> Result<JObject<'local>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for warning in warnings {
        env.with_local_frame(4, |env| {
            let warning = env.new_string(warning)?;
            env.call_method(
                &list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&warning)],
            )?;
            Ok::<(), Error>(())
        })?;
    }
    Ok(list)
}

fn cleanup_explanation_to_java<'local>(
    env: &mut JNIEnv<'local>,
    explanation: CleanupExplanation,
) -> Result<JObject<'local>> {
    let stats = cleanup_stats_to_java(env, explanation.stats)?;
    let candidate_files = cleanup_candidate_files_to_java(env, explanation.candidate_files)?;
    let referenced_branches =
        cleanup_referenced_branches_to_java(env, explanation.referenced_branches)?;
    let warnings = cleanup_warnings_to_java(env, explanation.warnings)?;

    Ok(env.new_object(
        "org/lance/cleanup/CleanupExplanation",
        "(JLorg/lance/cleanup/RemovalStats;Ljava/util/List;ZJLjava/util/List;Ljava/util/List;)V",
        &[
            JValue::Long(explanation.read_version as i64),
            JValue::Object(&stats),
            JValue::Object(&candidate_files),
            JValue::Bool(explanation.candidate_files_truncated as jboolean),
            JValue::Long(explanation.candidate_file_limit as i64),
            JValue::Object(&referenced_branches),
            JValue::Object(&warnings),
        ],
    )?)
}

//////////////////////////////
// Index operation Methods   //
//////////////////////////////

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetIndexes<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_indexes(&mut env, java_dataset))
}

fn inner_get_indexes<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let indexes = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.list_indexes()?
    };

    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for index_meta in indexes.iter() {
        let java_index = index_meta.into_java(env)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&java_index)],
        )?;
    }

    Ok(array_list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetIndexStatistics<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jindex_name: JString,
) -> JString<'local> {
    ok_or_throw_with_return!(
        env,
        inner_get_index_statistics(&mut env, java_dataset, jindex_name),
        JString::from(JObject::null())
    )
}

fn inner_get_index_statistics<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jindex_name: JString,
) -> Result<JString<'local>> {
    let index_name: String = jindex_name.extract(env)?;
    let stats_json = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.index_statistics(&index_name))?
    };
    let jstats = env.new_string(stats_json)?;
    Ok(jstats)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDescribeIndices<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    criteria_obj: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_describe_indices(&mut env, java_dataset, criteria_obj)
    )
}

fn inner_describe_indices<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    java_index_criteria: JObject,
) -> Result<JObject<'local>> {
    let mut for_column = None;
    let mut has_name = None;
    let index_criteria = env.get_optional(&java_index_criteria, |env, obj| {
        for_column = env.get_optional_string_from_method(&obj, "getForColumn")?;
        has_name = env.get_optional_string_from_method(&obj, "getHasName")?;
        let must_support_fts = env.get_boolean_from_method(&obj, "mustSupportFts")?;
        let must_support_exact_equality =
            env.get_boolean_from_method(&obj, "mustSupportExactEquality")?;
        Ok(RustIndexCriteria {
            for_column: for_column.as_deref(),
            has_name: has_name.as_deref(),
            must_support_fts,
            fts_document_granularity: None,
            must_support_exact_equality,
        })
    })?;

    let descriptions = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.describe_indices(index_criteria))?
    };

    export_vec(env, &descriptions)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCountIndexedRows(
    mut env: JNIEnv,
    java_dataset: JObject,
    jindex_name: JString,
    jfilter: JString,
    jfragment_ids: JObject, // Optional<List<Integer>>
) -> jlong {
    ok_or_throw_with_return!(
        env,
        inner_count_indexed_rows(&mut env, java_dataset, jindex_name, jfilter, jfragment_ids),
        -1
    )
}

fn inner_count_indexed_rows(
    env: &mut JNIEnv,
    java_dataset: JObject,
    _jindex_name: JString,
    jfilter: JString,
    jfragment_ids: JObject, // Optional<List<Integer>>
) -> Result<i64> {
    let filter: String = jfilter.extract(env)?;

    // Extract optional fragment IDs
    let fragment_ids: Option<Vec<u32>> = if env
        .call_method(&jfragment_ids, "isPresent", "()Z", &[])?
        .z()?
    {
        let list_obj = env
            .call_method(&jfragment_ids, "get", "()Ljava/lang/Object;", &[])?
            .l()?;
        let list = env.get_list(&list_obj)?;
        let mut ids = Vec::new();
        let mut iter = list.iter(env)?;
        while let Some(elem) = iter.next(env)? {
            let int_val = env.call_method(&elem, "intValue", "()I", &[])?.i()?;
            ids.push(int_val as u32);
        }
        Some(ids)
    } else {
        None
    };

    let count = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

        // Use a scanner with fragment filtering to count rows
        // This ensures we only count rows in the specified fragments
        let inner = dataset_guard.inner.clone();

        block_on(async {
            let mut scanner = inner.scan();

            // Apply filter
            if !filter.is_empty() {
                scanner.filter(&filter)?;
            }

            // Empty projection and enable row_id for count_rows to work
            // count_rows() requires metadata-only projection
            scanner.project::<String>(&[])?;
            scanner.with_row_id();

            // Apply fragment filter if specified
            if let Some(frag_ids) = fragment_ids {
                // Convert FileFragment to Fragment by extracting metadata
                let filtered_fragments: Vec<_> = inner
                    .get_fragments()
                    .into_iter()
                    .filter(|f| frag_ids.contains(&(f.id() as u32)))
                    .map(|f| f.metadata().clone())
                    .collect();
                scanner.with_fragments(filtered_fragments);
            }

            // Use the scanner's count_rows method
            let count = scanner.count_rows().await?;

            Ok::<i64, lance::Error>(count as i64)
        })?
    };

    Ok(count)
}

//////////////////////////////
// Session Methods          //
//////////////////////////////

/// Returns the session handle from a dataset.
/// The returned handle can be used to create a Java Session object.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetSessionHandle(
    mut env: JNIEnv,
    java_dataset: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_get_session_handle(&mut env, java_dataset), 0)
}

fn inner_get_session_handle(env: &mut JNIEnv, java_dataset: JObject) -> Result<jlong> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    let session = dataset_guard.inner.session();
    Ok(handle_from_session(session))
}

/////////////////////////////////
// Zonemap Stats Methods       //
/////////////////////////////////

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetZonemapStats<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jcolumn_name: JString,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_get_zonemap_stats(&mut env, java_dataset, jcolumn_name)
    )
}

fn inner_get_zonemap_stats<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jcolumn_name: JString,
) -> Result<JObject<'local>> {
    use arrow_array::Array;
    use datafusion_common::ScalarValue;
    use lance::dataset::index::LanceIndexStoreExt;
    use lance::index::DatasetIndexExt;
    use lance_index::scalar::IndexStore;
    use lance_index::scalar::lance_format::LanceIndexStore;

    let column_name: String = jcolumn_name.extract(env)?;

    // 1. Get the dataset and find every committed zonemap segment for this column
    let zonemap_batches = {
        let dataset = {
            let dataset_guard = unsafe {
                env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET)
            }?;
            dataset_guard.inner.clone()
        };
        // Guard is dropped here — dataset is owned

        // Find the field for the requested column (validates it exists)
        dataset.schema().field(&column_name).ok_or_else(|| {
            Error::input_error(format!(
                "Column '{}' not found in dataset schema",
                column_name
            ))
        })?;

        // Do all async work in a single block_on call to avoid nested runtime issues
        block_on(async {
            // Find the zonemap index for this column using describe_indices
            let descriptions = dataset
                .describe_indices(Some(lance_index::IndexCriteria {
                    for_column: Some(&column_name),
                    has_name: None,
                    must_support_fts: false,
                    fts_document_granularity: None,
                    must_support_exact_equality: false,
                }))
                .await
                .map_err(Error::from)?;

            let zonemap_desc = descriptions
                .iter()
                .find(|desc| desc.index_type().to_lowercase().contains("zonemap"));

            match zonemap_desc {
                Some(desc) => {
                    let mut indices = dataset
                        .load_indices_by_name(desc.name())
                        .await
                        .map_err(Error::from)?;
                    indices.sort_by_key(|idx| {
                        idx.fragment_bitmap
                            .as_ref()
                            .and_then(|bitmap| bitmap.iter().next())
                            .unwrap_or_default()
                    });

                    let mut batches = Vec::with_capacity(indices.len());
                    for index in &indices {
                        let index_store = Arc::new(
                            LanceIndexStore::from_dataset_for_existing(&dataset, index)
                                .await
                                .map_err(Error::from)?,
                        );
                        let index_file = index_store
                            .open_index_file("zonemap.lance")
                            .await
                            .map_err(Error::from)?;
                        if index_file.num_rows() == 0 {
                            continue;
                        }
                        batches.push(
                            index_file
                                .read_range(0..index_file.num_rows(), None)
                                .await
                                .map_err(Error::from)?,
                        );
                    }
                    Ok::<_, Error>(batches)
                }
                None => Ok(Vec::new()),
            }
        })?
    };

    // 3. Convert the RecordBatch rows to a Java ArrayList<ZoneStats>
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
    if zonemap_batches.is_empty() {
        return Ok(array_list);
    }

    for record_batch in zonemap_batches {
        let min_col = record_batch.column_by_name("min").ok_or_else(|| {
            Error::input_error("ZoneMap index file missing 'min' column".to_string())
        })?;
        let max_col = record_batch.column_by_name("max").ok_or_else(|| {
            Error::input_error("ZoneMap index file missing 'max' column".to_string())
        })?;
        let null_count_col = record_batch
            .column_by_name("null_count")
            .ok_or_else(|| {
                Error::input_error("ZoneMap index file missing 'null_count' column".to_string())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| {
                Error::input_error("ZoneMap 'null_count' column is not UInt32".to_string())
            })?;
        let fragment_id_col = record_batch
            .column_by_name("fragment_id")
            .ok_or_else(|| {
                Error::input_error("ZoneMap index file missing 'fragment_id' column".to_string())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::input_error("ZoneMap 'fragment_id' column is not UInt64".to_string())
            })?;
        let zone_start_col = record_batch
            .column_by_name("zone_start")
            .ok_or_else(|| {
                Error::input_error("ZoneMap index file missing 'zone_start' column".to_string())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::input_error("ZoneMap 'zone_start' column is not UInt64".to_string())
            })?;
        let zone_length_col = record_batch
            .column_by_name("zone_length")
            .ok_or_else(|| {
                Error::input_error("ZoneMap index file missing 'zone_length' column".to_string())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::input_error("ZoneMap 'zone_length' column is not UInt64".to_string())
            })?;

        for i in 0..record_batch.num_rows() {
            let fragment_id = fragment_id_col.value(i) as i32;
            let zone_start = zone_start_col.value(i) as i64;
            let zone_length = zone_length_col.value(i) as i64;
            let null_count = null_count_col.value(i) as i64;

            let min_scalar = ScalarValue::try_from_array(min_col, i).map_err(|e| {
                Error::input_error(format!("Failed to read min value at row {}: {}", i, e))
            })?;
            let max_scalar = ScalarValue::try_from_array(max_col, i).map_err(|e| {
                Error::input_error(format!("Failed to read max value at row {}: {}", i, e))
            })?;

            let j_min = crate::utils::scalar_value_to_java(env, &min_scalar)?;
            let j_max = crate::utils::scalar_value_to_java(env, &max_scalar)?;

            let zone_stats = env.new_object(
                "org/lance/index/scalar/ZoneStats",
                "(IJJLjava/lang/Comparable;Ljava/lang/Comparable;J)V",
                &[
                    JValue::Int(fragment_id),
                    JValue::Long(zone_start),
                    JValue::Long(zone_length),
                    JValue::Object(&j_min),
                    JValue::Object(&j_max),
                    JValue::Long(null_count),
                ],
            )?;

            env.call_method(
                &array_list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&zone_stats)],
            )?;
        }
    }

    Ok(array_list)
}
