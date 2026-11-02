// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::conversions::extract_namespace_info;
use super::refs::transform_jref_to_ref;
use super::{BlockingDataset, NATIVE_DATASET};
use crate::block_on;
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::session::session_from_handle;
use crate::traits::{FromJString, IntoJava};
use crate::utils::{
    extract_base_store_params, extract_storage_options, extract_write_params, to_rust_map,
};
use arrow::array::RecordBatchReader;
use arrow::datatypes::Schema;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::RecordBatchIterator;
use jni::JNIEnv;
use jni::objects::{JMap, JObject, JString, JValue};
use jni::sys::{jboolean, jlong};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::refs::Ref;
use lance::dataset::transaction::Operation;
use lance::dataset::{Dataset, ReadParams, WriteParams};
use lance::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
use lance::io::{ObjectStore, ObjectStoreParams};
use lance::session::Session as LanceSession;
use lance_io::object_store::ObjectStoreRegistry;
use lance_io::object_store::{LanceNamespaceStorageOptionsProvider, StorageOptionsProvider};
use lance_namespace::LanceNamespace;
use lance_table::io::commit::CommitHandler;
use lance_table::io::commit::ManifestLocation;
use lance_table::io::commit::external_manifest::ExternalManifestCommitHandler;
use std::collections::HashMap;
use std::iter::empty;
use std::sync::Arc;

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

    #[allow(clippy::too_many_arguments)]
    pub fn open(
        uri: &str,
        reference: Option<Ref>,
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

        builder = match reference {
            Some(Ref::VersionNumber(version)) | Some(Ref::Version(None, Some(version))) => {
                builder.with_version(version)
            }
            Some(Ref::Version(Some(branch), version)) => builder.with_branch(&branch, version),
            Some(Ref::Tag(tag)) => builder.with_tag(&tag),
            Some(Ref::Version(None, None)) | None => builder,
        };

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
}

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
    file_write_options: JObject,                // FileWriteOptions
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
            file_write_options,
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
    file_write_options: JObject,                // FileWriteOptions
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
        file_write_options,
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
    file_write_options: JObject,                   // FileWriteOptions
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
            file_write_options,
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
    file_write_options: JObject,                // FileWriteOptions
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
        file_write_options,
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
    file_write_options: JObject,
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
        &file_write_options,
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

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_releaseNativeDataset(mut env: JNIEnv, obj: JObject) {
    ok_or_throw_without_return!(env, inner_release_native_dataset(&mut env, obj))
}

fn inner_release_native_dataset(env: &mut JNIEnv, obj: JObject) -> Result<()> {
    let dataset: BlockingDataset = unsafe { env.take_rust_field(obj, NATIVE_DATASET)? };
    dataset.close();
    Ok(())
}

//////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_openNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    version_obj: JObject,    // Optional<Long>
    ref_obj: JObject,        // Optional<Ref>
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
            ref_obj,
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
    ref_obj: JObject,        // Optional<Ref>
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
    // ReadOptions.Builder rejects setting both, so at most one of them is present.
    let reference = env
        .get_optional(&ref_obj, |env, jref| transform_jref_to_ref(jref, env))?
        .or(version.map(Ref::from));
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
        reference,
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
