// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::array::{RecordBatch, RecordBatchIterator, StructArray};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi_and_data_type};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_schema::{DataType, Schema as ArrowSchema};
use jni::objects::{JByteArray, JIntArray, JValue, JValueGen};
use jni::{
    JNIEnv,
    objects::{JClass, JLongArray, JObject, JString},
    sys::{jint, jlong, jstring},
};
use lance::datatypes::Schema;
use lance::table::format::{
    DataFile, DeletionFile, DeletionFileType, Fragment, RowDatasetVersionMeta, RowIdMeta,
};
use lance_io::utils::CachedFileSize;
use lance_table::rowids::{RowIdSequence, write_row_ids};
use std::iter::once;

use roaring::RoaringBitmap;

use lance::dataset::fragment::write::FragmentCreateBuilder;
use lance::io::ObjectStoreParams;
use lance_datafusion::utils::StreamingWriteSource;
use lance_io::object_store::{LanceNamespaceStorageOptionsProvider, StorageOptionsProvider};
use std::collections::HashMap;
use std::sync::Arc;

use crate::blocking_dataset::extract_namespace_info;
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::session::session_from_handle;
use crate::traits::{FromJObjectWithEnv, IntoJava, JLance, export_vec, import_vec};
use crate::utils::extract_storage_options;
use crate::{
    block_on,
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    traits::FromJString,
    utils::extract_write_params,
};

#[derive(Debug, Clone)]
pub(crate) struct FragmentMergeResult {
    fragment: Fragment,
    schema: Schema,
}

#[derive(Debug, Clone)]
pub(crate) struct FragmentUpdateResult {
    updated_fragment: Fragment,
    fields_modified: Vec<u32>,
    /// Matched row offsets serialized as portable RoaringBitmap bytes.
    updated_row_offset_bytes: Vec<u8>,
}

//////////////////
// Read Methods //
//////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Fragment_countRowsNative(
    mut env: JNIEnv,
    _jfragment: JObject,
    jdataset: JObject,
    fragment_id: jlong,
) -> jint {
    ok_or_throw_with_return!(
        env,
        inner_count_rows_native(&mut env, jdataset, fragment_id),
        -1
    ) as jint
}

fn inner_count_rows_native(
    env: &mut JNIEnv,
    jdataset: JObject,
    fragment_id: jlong,
) -> Result<usize> {
    let dataset = unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
    let Some(fragment) = dataset.inner.get_fragment(fragment_id as usize) else {
        return Err(Error::input_error(format!(
            "Fragment not found: {fragment_id}"
        )));
    };
    let res = block_on(fragment.count_rows(None))?;
    Ok(res)
}

///////////////////
// Write Methods //
///////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Fragment_createWithFfiArray<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    dataset_uri: JString,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
    max_rows_per_file: JObject,                 // Optional<Integer>
    max_rows_per_group: JObject,                // Optional<Integer>
    max_bytes_per_file: JObject,                // Optional<Long>
    mode: JObject,                              // Optional<String>
    enable_stable_row_ids: JObject,             // Optional<Boolean>
    data_storage_version: JObject,              // Optional<String>
    storage_options_obj: JObject,               // Map<String, String>
    base_store_params_obj: JObject,             // Map<String, Map<String, String>>
    initial_bases: JObject,                     // Optional<List<BasePath>>
    target_bases: JObject,                      // Optional<List<String>>
    namespace_obj: JObject,                     // LanceNamespace (can be null)
    table_id_obj: JObject,                      // List<String> (can be null)
    allow_external_blob_outside_bases: JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,     // Optional<Long>
    schema_addr: jlong,
    session_handle: jlong, // Session handle, 0 means no session
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_create_with_ffi_array(
            &mut env,
            dataset_uri,
            arrow_array_addr,
            arrow_schema_addr,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            enable_stable_row_ids,
            data_storage_version,
            storage_options_obj,
            base_store_params_obj,
            initial_bases,
            target_bases,
            namespace_obj,
            table_id_obj,
            allow_external_blob_outside_bases,
            blob_pack_file_size_threshold,
            schema_addr,
            session_handle,
        ),
        JObject::default()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_array<'local>(
    env: &mut JNIEnv<'local>,
    dataset_uri: JString,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
    max_rows_per_file: JObject,                 // Optional<Integer>
    max_rows_per_group: JObject,                // Optional<Integer>
    max_bytes_per_file: JObject,                // Optional<Long>
    mode: JObject,                              // Optional<String>
    enable_stable_row_ids: JObject,             // Optional<Boolean>
    data_storage_version: JObject,              // Optional<String>
    storage_options_obj: JObject,               // Map<String, String>
    base_store_params_obj: JObject,             // Map<String, Map<String, String>>
    initial_bases: JObject,                     // Optional<List<BasePath>>
    target_bases: JObject,                      // Optional<List<String>>
    namespace_obj: JObject,                     // LanceNamespace (can be null)
    table_id_obj: JObject,                      // List<String> (can be null)
    allow_external_blob_outside_bases: JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,     // Optional<Long>
    schema_addr: jlong,
    session_handle: jlong, // Session handle, 0 means no session
) -> Result<JObject<'local>> {
    let c_array_ptr = arrow_array_addr as *mut FFI_ArrowArray;
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;

    let c_array = unsafe { FFI_ArrowArray::from_raw(c_array_ptr) };
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let data_type = DataType::try_from(&c_schema)?;

    let array_data = unsafe { from_ffi_and_data_type(c_array, data_type) }?;

    let record_batch = RecordBatch::from(StructArray::from(array_data));
    let batch_schema = record_batch.schema().clone();
    let reader = RecordBatchIterator::new(once(Ok(record_batch)), batch_schema);

    create_fragment(
        env,
        dataset_uri,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        enable_stable_row_ids,
        data_storage_version,
        storage_options_obj,
        base_store_params_obj,
        initial_bases,
        target_bases,
        namespace_obj,
        table_id_obj,
        allow_external_blob_outside_bases,
        blob_pack_file_size_threshold,
        schema_addr,
        session_handle,
        reader,
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Fragment_createWithFfiStream<'a>(
    mut env: JNIEnv<'a>,
    _obj: JObject,
    dataset_uri: JString,
    arrow_array_stream_addr: jlong,
    max_rows_per_file: JObject,                 // Optional<Integer>
    max_rows_per_group: JObject,                // Optional<Integer>
    max_bytes_per_file: JObject,                // Optional<Long>
    mode: JObject,                              // Optional<String>
    enable_stable_row_ids: JObject,             // Optional<Boolean>
    data_storage_version: JObject,              // Optional<String>
    storage_options_obj: JObject,               // Map<String, String>
    base_store_params_obj: JObject,             // Map<String, Map<String, String>>
    initial_bases: JObject,                     // Optional<List<BasePath>>
    target_bases: JObject,                      // Optional<List<String>>
    namespace_obj: JObject,                     // LanceNamespace (can be null)
    table_id_obj: JObject,                      // List<String> (can be null)
    allow_external_blob_outside_bases: JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,     // Optional<Long>
    schema_addr: jlong,
    session_handle: jlong, // Session handle, 0 means no session
) -> JObject<'a> {
    ok_or_throw_with_return!(
        env,
        inner_create_with_ffi_stream(
            &mut env,
            dataset_uri,
            arrow_array_stream_addr,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            mode,
            enable_stable_row_ids,
            data_storage_version,
            storage_options_obj,
            base_store_params_obj,
            initial_bases,
            target_bases,
            namespace_obj,
            table_id_obj,
            allow_external_blob_outside_bases,
            blob_pack_file_size_threshold,
            schema_addr,
            session_handle,
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_with_ffi_stream<'local>(
    env: &mut JNIEnv<'local>,
    dataset_uri: JString,
    arrow_array_stream_addr: jlong,
    max_rows_per_file: JObject,                 // Optional<Integer>
    max_rows_per_group: JObject,                // Optional<Integer>
    max_bytes_per_file: JObject,                // Optional<Long>
    mode: JObject,                              // Optional<String>
    enable_stable_row_ids: JObject,             // Optional<Boolean>
    data_storage_version: JObject,              // Optional<String>
    storage_options_obj: JObject,               // Map<String, String>
    base_store_params_obj: JObject,             // Map<String, Map<String, String>>
    initial_bases: JObject,                     // Optional<List<BasePath>>
    target_bases: JObject,                      // Optional<List<String>>
    namespace_obj: JObject,                     // LanceNamespace (can be null)
    table_id_obj: JObject,                      // List<String> (can be null)
    allow_external_blob_outside_bases: JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,     // Optional<Long>
    schema_addr: jlong,
    session_handle: jlong, // Session handle, 0 means no session
) -> Result<JObject<'local>> {
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;

    create_fragment(
        env,
        dataset_uri,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        mode,
        enable_stable_row_ids,
        data_storage_version,
        storage_options_obj,
        base_store_params_obj,
        initial_bases,
        target_bases,
        namespace_obj,
        table_id_obj,
        allow_external_blob_outside_bases,
        blob_pack_file_size_threshold,
        schema_addr,
        session_handle,
        reader,
    )
}

#[allow(clippy::too_many_arguments)]
fn create_fragment<'a>(
    env: &mut JNIEnv<'a>,
    dataset_uri: JString,
    max_rows_per_file: JObject,                 // Optional<Integer>
    max_rows_per_group: JObject,                // Optional<Integer>
    max_bytes_per_file: JObject,                // Optional<Long>
    mode: JObject,                              // Optional<String>
    enable_stable_row_ids: JObject,             // Optional<Boolean>
    data_storage_version: JObject,              // Optional<String>
    storage_options_obj: JObject,               // Map<String, String>
    base_store_params_obj: JObject,             // Map<String, Map<String, String>>
    initial_bases: JObject,                     // Optional<List<BasePath>>
    target_bases: JObject,                      // Optional<List<String>>
    namespace_obj: JObject,                     // LanceNamespace (can be null)
    table_id_obj: JObject,                      // List<String> (can be null)
    allow_external_blob_outside_bases: JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: JObject,     // Optional<Long>
    schema_addr: jlong,
    session_handle: jlong, // Session handle, 0 means no session
    source: impl StreamingWriteSource,
) -> Result<JObject<'a>> {
    let path_str = dataset_uri.extract(env)?;

    let mut write_params = extract_write_params(
        env,
        &max_rows_per_file,
        &max_rows_per_group,
        &max_bytes_per_file,
        &mode,
        &enable_stable_row_ids,
        &data_storage_version,
        None,
        &storage_options_obj,
        &base_store_params_obj,
        &initial_bases,
        &target_bases,
        &allow_external_blob_outside_bases,
        &blob_pack_file_size_threshold,
    )?;

    write_params.session = session_from_handle(session_handle);

    // Set up storage options provider if namespace is provided
    let namespace_info = extract_namespace_info(env, &namespace_obj, &table_id_obj)?;
    if let Some((namespace, table_id)) = namespace_info {
        let provider: Arc<dyn StorageOptionsProvider> = Arc::new(
            LanceNamespaceStorageOptionsProvider::new(namespace, table_id),
        );

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

    let mut builder = FragmentCreateBuilder::new(&path_str).write_params(&write_params);
    let schema;
    if schema_addr != 0 {
        let c_schema_ptr = schema_addr as *mut FFI_ArrowSchema;
        let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
        let arrow_schema = ArrowSchema::try_from(&c_schema)?;
        // Schema::try_from restores Lance field IDs from the LANCE_FIELD_ID_KEY
        // metadata inserted by LanceSchema.asArrowSchemaWithFieldIds().
        schema = Schema::try_from(&arrow_schema)?;
        builder = builder.schema(&schema);
    }

    let fragments = block_on(builder.write_fragments(source))?;
    export_vec(env, &fragments)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Fragment_nativeDeleteRows<'a>(
    mut env: JNIEnv<'a>,
    _obj: JObject,
    jdataset: JObject,
    fragment_id: jint,
    row_indexes: JObject, // List<Integer>
) -> JObject<'a> {
    ok_or_throw!(
        env,
        inner_delete_rows(&mut env, jdataset, fragment_id, row_indexes)
    )
}

fn inner_delete_rows<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    fragment_id: jint,
    row_indexes: JObject, // List<Integer>
) -> Result<JObject<'local>> {
    let fragment_id = fragment_id as usize;
    let fragment = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        let Some(fragment) = dataset.inner.get_fragment(fragment_id) else {
            return Err(Error::input_error(format!(
                "Fragment not found: {fragment_id}"
            )));
        };
        fragment
    };

    let indexes: Vec<u32> = env
        .get_integers(&row_indexes)?
        .into_iter()
        .map(|x| x as u32)
        .collect();

    let res = block_on(async move { fragment.extend_deletions(indexes).await });

    let obj = match res {
        Ok(Some(f)) => f.metadata().into_java(env)?,
        Ok(None) => JObject::default(),
        Err(e) => {
            return Err(Error::runtime_error(format!(
                "Cannot delete rows in fragment {}: {:?}",
                fragment_id, e
            )));
        }
    };

    Ok(obj)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Fragment_nativeMergeColumns<'a>(
    mut env: JNIEnv<'a>,
    _obj: JObject,
    jdataset: JObject,              // Java DataSet
    fragment_id: jlong,             // FragmentID
    arrow_array_stream_addr: jlong, // memoryAddress of ArrowStream
    left_on: JString,               // left column name to join on
    right_on: JString,              // right column name to join on
) -> JObject<'a> {
    ok_or_throw_with_return!(
        env,
        inner_merge_column(
            &mut env,
            jdataset,
            fragment_id,
            arrow_array_stream_addr,
            left_on,
            right_on
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_merge_column<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    fragment_id: jlong,
    arrow_array_stream_addr: jlong,
    left_on: JString,
    right_on: JString,
) -> Result<JObject<'local>> {
    let (fragment_opt, max_field_id) = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        (
            dataset.inner.get_fragment(fragment_id as usize),
            dataset.inner.manifest().max_field_id(),
        )
    };
    let mut fragment = match fragment_opt {
        Some(fragment) => fragment,
        None => {
            return Err(Error::input_error(format!(
                "Fragment not found: {fragment_id}"
            )));
        }
    };

    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
    let left_on_str: String = left_on.extract(env)?;
    let right_on_str: String = right_on.extract(env)?;

    let (new_frag, new_schema) =
        block_on(fragment.merge_columns(reader, &left_on_str, &right_on_str, max_field_id))?;
    let result = FragmentMergeResult {
        fragment: new_frag,
        schema: new_schema,
    };
    result.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Fragment_nativeUpdateColumns<'a>(
    mut env: JNIEnv<'a>,
    _obj: JObject,
    jdataset: JObject,              // Java DataSet
    fragment_id: jlong,             // FragmentID
    arrow_array_stream_addr: jlong, // memoryAddress of ArrowStream
    left_on: JString,               // left column name to join on
    right_on: JString,              // right column name to join on
) -> JObject<'a> {
    ok_or_throw_with_return!(
        env,
        inner_update_column(
            &mut env,
            jdataset,
            fragment_id,
            arrow_array_stream_addr,
            left_on,
            right_on
        ),
        JObject::null()
    )
}
#[allow(clippy::too_many_arguments)]
fn inner_update_column<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    fragment_id: jlong,
    arrow_array_stream_addr: jlong,
    left_on: JString,
    right_on: JString,
) -> Result<JObject<'local>> {
    let fragment_opt = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.inner.get_fragment(fragment_id as usize)
    };
    let mut fragment = match fragment_opt {
        Some(fragment) => fragment,
        None => {
            return Err(Error::input_error(format!(
                "Fragment not found: {fragment_id}"
            )));
        }
    };
    let stream_ptr = arrow_array_stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
    let left_on_str: String = left_on.extract(env)?;
    let right_on_str: String = right_on.extract(env)?;
    let r = block_on(fragment.update_columns_with_offsets(reader, &left_on_str, &right_on_str))?;
    let updated_row_offset_bytes = serialize_matched_offsets(&r.matched_offsets)?;
    let result = FragmentUpdateResult {
        updated_fragment: r.fragment,
        fields_modified: r.fields_modified,
        updated_row_offset_bytes,
    };
    result.into_java(env)
}

fn serialize_matched_offsets(bitmap: &RoaringBitmap) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    bitmap.serialize_into(&mut buf).map_err(|e| {
        Error::runtime_error(format!(
            "failed to serialize matched row offsets RoaringBitmap: {e}"
        ))
    })?;
    Ok(buf)
}

fn deserialize_row_offset_bytes(bytes: &[u8]) -> Result<RoaringBitmap> {
    if bytes.is_empty() {
        return Ok(RoaringBitmap::new());
    }
    RoaringBitmap::deserialize_from(bytes).map_err(|e| {
        Error::input_error(format!(
            "invalid updatedRowOffsetBytes RoaringBitmap bytes: {e}"
        ))
    })
}

fn expand_row_offset_bytes_to_i64(bitmap: &RoaringBitmap) -> Vec<i64> {
    bitmap.iter().map(|o| o as i64).collect()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_fragment_FragmentUpdateResult_expandRowOffsetsFromBytes<
    'local,
>(
    mut env: JNIEnv<'local>,
    _cls: JClass,
    jbytes: JByteArray,
) -> JLongArray<'local> {
    ok_or_throw_with_return!(
        env,
        inner_expand_updated_row_offset_bytes(&mut env, jbytes),
        unsafe { JLongArray::from_raw(std::ptr::null_mut()) }
    )
}

fn inner_expand_updated_row_offset_bytes<'local>(
    env: &mut JNIEnv<'local>,
    jbytes: JByteArray,
) -> Result<JLongArray<'local>> {
    let buf = env.convert_byte_array(&jbytes)?;
    let bitmap = deserialize_row_offset_bytes(&buf)?;
    let offsets = expand_row_offset_bytes_to_i64(&bitmap);
    let arr = env.new_long_array(offsets.len() as i32)?;
    if !offsets.is_empty() {
        env.set_long_array_region(&arr, 0, &offsets)?;
    }
    Ok(arr)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_fragment_FragmentUpdateResult_encodeRowOffsetsToBytes<
    'local,
>(
    mut env: JNIEnv<'local>,
    _cls: JClass,
    joffsets: JLongArray,
) -> JByteArray<'local> {
    ok_or_throw_with_return!(
        env,
        inner_encode_updated_row_offset_bytes(&mut env, joffsets),
        unsafe { JByteArray::from_raw(std::ptr::null_mut()) }
    )
}

fn inner_encode_updated_row_offset_bytes<'local>(
    env: &mut JNIEnv<'local>,
    joffsets: JLongArray,
) -> Result<JByteArray<'local>> {
    let len = env.get_array_length(&joffsets)?;
    let mut buf: Vec<i64> = vec![0; len as usize];
    if len > 0 {
        env.get_long_array_region(&joffsets, 0, buf.as_mut_slice())?;
    }
    let mut bitmap = RoaringBitmap::new();
    for offset in buf {
        if offset < 0 {
            return Err(Error::input_error(format!(
                "updatedRowOffsets must be non-negative, got {offset}"
            )));
        }
        if offset > u32::MAX as i64 {
            return Err(Error::input_error(format!(
                "updatedRowOffsets value {offset} exceeds u32::MAX"
            )));
        }
        bitmap.insert(offset as u32);
    }
    let bytes = serialize_matched_offsets(&bitmap)?;
    Ok(env.byte_array_from_slice(&bytes)?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_fragment_RowIdMeta_nativeEncodeRowIds(
    mut env: JNIEnv,
    _cls: JClass,
    row_ids: JLongArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        inner_encode_row_ids(&mut env, &row_ids)
            .and_then(|json| env.new_string(json).map_err(Error::from)),
        std::ptr::null_mut()
    )
    .into_raw()
}

fn inner_encode_row_ids(env: &mut JNIEnv, row_ids: &JLongArray) -> Result<String> {
    let len = env.get_array_length(row_ids)?;
    let mut buf: Vec<i64> = vec![0; len as usize];
    env.get_long_array_region(row_ids, 0, buf.as_mut_slice())?;
    let ids: Vec<u64> = buf.into_iter().map(|x| x as u64).collect();
    let seq = RowIdSequence::from(ids.as_slice());
    let meta = RowIdMeta::Inline(write_row_ids(&seq).into());
    let json = serde_json::to_string(&meta)?;
    Ok(json)
}

const DATA_FILE_CLASS: &str = "org/lance/fragment/DataFile";
const DATA_FILE_CONSTRUCTOR_SIG: &str =
    "(Ljava/lang/String;[I[IIILjava/lang/Long;Ljava/lang/Integer;)V";
const DELETE_FILE_CLASS: &str = "org/lance/fragment/DeletionFile";
const DELETE_FILE_CONSTRUCTOR_SIG: &str =
    "(JJLjava/lang/Long;Lorg/lance/fragment/DeletionFileType;Ljava/lang/Integer;)V";
const DELETE_FILE_TYPE_CLASS: &str = "org/lance/fragment/DeletionFileType";
const FRAGMENT_METADATA_CLASS: &str = "org/lance/FragmentMetadata";
const FRAGMENT_METADATA_CONSTRUCTOR_SIG: &str = "(ILjava/util/List;Ljava/lang/Long;Lorg/lance/fragment/DeletionFile;Lorg/lance/fragment/RowIdMeta;Lorg/lance/fragment/VersionMeta;Lorg/lance/fragment/VersionMeta;Ljava/util/List;)V";
const ROW_ID_META_CLASS: &str = "org/lance/fragment/RowIdMeta";
const ROW_ID_META_CONSTRUCTOR_SIG: &str = "(Ljava/lang/String;)V";
const VERSION_META_CLASS: &str = "org/lance/fragment/VersionMeta";
const VERSION_META_CONSTRUCTOR_SIG: &str = "(Ljava/lang/String;)V";
const FRAGMENT_MERGE_RESULT_CLASS: &str = "org/lance/fragment/FragmentMergeResult";
const FRAGMENT_MERGE_RESULT_CONSTRUCTOR_SIG: &str =
    "(Lorg/lance/FragmentMetadata;Lorg/lance/schema/LanceSchema;)V";
const FRAGMENT_UPDATE_RESULT_CLASS: &str = "org/lance/fragment/FragmentUpdateResult";
const FRAGMENT_UPDATE_RESULT_CONSTRUCTOR_SIG: &str = "(Lorg/lance/FragmentMetadata;[J[B)V";

impl IntoJava for &FragmentMergeResult {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let java_fragment_meta_data = self.fragment.into_java(env)?;
        let java_lance_schema = self.schema.clone().into_java(env)?;
        Ok(env.new_object(
            FRAGMENT_MERGE_RESULT_CLASS,
            FRAGMENT_MERGE_RESULT_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&java_fragment_meta_data),
                JValueGen::Object(&java_lance_schema),
            ],
        )?)
    }
}

impl IntoJava for &FragmentUpdateResult {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let java_updated_fragment = self.updated_fragment.into_java(env)?;
        let java_fields_modified = JLance(self.fields_modified.clone()).into_java(env)?;
        let java_updated_row_offset_bytes =
            env.byte_array_from_slice(&self.updated_row_offset_bytes)?;
        Ok(env.new_object(
            FRAGMENT_UPDATE_RESULT_CLASS,
            FRAGMENT_UPDATE_RESULT_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&java_updated_fragment),
                JValueGen::Object(&java_fields_modified),
                JValueGen::Object(&java_updated_row_offset_bytes),
            ],
        )?)
    }
}

impl IntoJava for &DataFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let path = env.new_string(self.path.clone())?.into();
        let fields = JLance(self.fields.to_vec()).into_java(env)?;
        let column_indices = JLance(self.column_indices.to_vec()).into_java(env)?;
        let file_size_bytes = match self.file_size_bytes.get() {
            Some(f) => JLance(u64::from(f) as i64).into_java(env)?,
            None => JObject::null(),
        };
        let base_id = convert_to_java_integer(env, self.base_id)?;
        Ok(env.new_object(
            DATA_FILE_CLASS,
            DATA_FILE_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&path),
                JValueGen::Object(&fields),
                JValueGen::Object(&column_indices),
                JValueGen::Int(self.file_major_version as i32),
                JValueGen::Int(self.file_minor_version as i32),
                JValueGen::Object(&file_size_bytes),
                JValueGen::Object(&base_id),
            ],
        )?)
    }
}

impl IntoJava for &DeletionFileType {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let name = match self {
            lance::table::format::DeletionFileType::Array => "ARRAY",
            lance::table::format::DeletionFileType::Bitmap => "BITMAP",
        };
        env.get_static_field(
            DELETE_FILE_TYPE_CLASS,
            name,
            format!("L{};", DELETE_FILE_TYPE_CLASS),
        )?
        .l()
        .map_err(|e| {
            Error::runtime_error(format!("failed to get {}: {}", DELETE_FILE_TYPE_CLASS, e))
        })
    }
}

impl IntoJava for &DeletionFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let num_deleted_rows = match self.num_deleted_rows {
            Some(f) => JLance(f).into_java(env)?,
            None => JObject::null(),
        };
        let file_type = self.file_type.into_java(env)?;
        let base_id = convert_to_java_integer(env, self.base_id)?;
        Ok(env.new_object(
            DELETE_FILE_CLASS,
            DELETE_FILE_CONSTRUCTOR_SIG,
            &[
                JValueGen::Long(self.id as i64),
                JValueGen::Long(self.read_version as i64),
                JValueGen::Object(&num_deleted_rows),
                JValueGen::Object(&file_type),
                JValueGen::Object(&base_id),
            ],
        )?)
    }
}

impl IntoJava for &RowIdMeta {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let json_str = serde_json::to_string(self)?;
        let json = env.new_string(json_str)?.into();
        Ok(env.new_object(
            ROW_ID_META_CLASS,
            ROW_ID_META_CONSTRUCTOR_SIG,
            &[JValueGen::Object(&json)],
        )?)
    }
}

impl IntoJava for &RowDatasetVersionMeta {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let json_str = serde_json::to_string(self)?;
        let json = env.new_string(json_str)?.into();
        Ok(env.new_object(
            VERSION_META_CLASS,
            VERSION_META_CONSTRUCTOR_SIG,
            &[JValueGen::Object(&json)],
        )?)
    }
}

impl IntoJava for &Fragment {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        let files = self.files.clone();
        let files = export_vec::<DataFile>(env, &files)?;
        let referenced_data_files = self
            .referenced_lance_files()
            .cloned()
            .collect::<Vec<DataFile>>();
        let referenced_data_files = export_vec::<DataFile>(env, &referenced_data_files)?;
        let deletion_file = match &self.deletion_file {
            Some(f) => f.into_java(env)?,
            None => JObject::null(),
        };
        let physical_rows = &JLance(self.physical_rows).into_java(env)?;
        let row_id_meta = match &self.row_id_meta {
            Some(m) => m.into_java(env)?,
            None => JObject::null(),
        };
        let created_at = match &self.created_at_version_meta {
            Some(m) => m.into_java(env)?,
            None => JObject::null(),
        };
        let last_updated_at = match &self.last_updated_at_version_meta {
            Some(m) => m.into_java(env)?,
            None => JObject::null(),
        };

        env.new_object(
            FRAGMENT_METADATA_CLASS,
            FRAGMENT_METADATA_CONSTRUCTOR_SIG,
            &[
                JValueGen::Int(self.id as i32),
                JValueGen::Object(&files),
                JValueGen::Object(physical_rows),
                JValueGen::Object(&deletion_file),
                JValueGen::Object(&row_id_meta),
                JValueGen::Object(&created_at),
                JValueGen::Object(&last_updated_at),
                JValueGen::Object(&referenced_data_files),
            ],
        )
        .map_err(|e| {
            Error::runtime_error(format!("failed to get {}: {}", FRAGMENT_METADATA_CLASS, e))
        })
    }
}

impl FromJObjectWithEnv<RowIdMeta> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RowIdMeta> {
        let metadata = env
            .call_method(self, "getMetadata", "()Ljava/lang/String;", &[])?
            .l()?;
        let s: String = env.get_string(&JString::from(metadata))?.into();
        let meta: RowIdMeta = serde_json::from_str(&s)?;
        Ok(meta)
    }
}

impl FromJObjectWithEnv<RowDatasetVersionMeta> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RowDatasetVersionMeta> {
        let metadata = env
            .call_method(self, "getMetadata", "()Ljava/lang/String;", &[])?
            .l()?;
        let s: String = env.get_string(&JString::from(metadata))?.into();
        let meta: RowDatasetVersionMeta = serde_json::from_str(&s)?;
        Ok(meta)
    }
}

/// Extract an optional field from a Java object by calling a getter method.
/// Returns `None` if the getter returns null, otherwise deserializes the JObject.
fn extract_nullable_field<T>(
    env: &mut JNIEnv<'_>,
    obj: &JObject<'_>,
    method: &str,
    class: &str,
) -> Result<Option<T>>
where
    for<'a> JObject<'a>: FromJObjectWithEnv<T>,
{
    let result = env
        .call_method(obj, method, format!("()L{};", class), &[])?
        .l()?;
    if result.is_null() {
        Ok(None)
    } else {
        Ok(Some(result.extract_object(env)?))
    }
}

impl FromJObjectWithEnv<Fragment> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<Fragment> {
        let id = env.call_method(self, "getId", "()I", &[])?.i()? as u64;
        let file_objs = env
            .call_method(self, "getFiles", "()Ljava/util/List;", &[])?
            .l()?;
        let physical_rows = env.call_method(self, "getPhysicalRows", "()J", &[])?.j()? as usize;
        let file_objs = import_vec(env, &file_objs)?;
        let mut files = Vec::with_capacity(file_objs.len());
        for f in file_objs {
            files.push(f.extract_object(env)?);
        }

        let deletion_file =
            extract_nullable_field(env, self, "getDeletionFile", DELETE_FILE_CLASS)?;
        let row_id_meta = extract_nullable_field(env, self, "getRowIdMeta", ROW_ID_META_CLASS)?;
        let created_at_version_meta =
            extract_nullable_field(env, self, "getCreatedAtVersionMeta", VERSION_META_CLASS)?;
        let last_updated_at_version_meta =
            extract_nullable_field(env, self, "getLastUpdatedAtVersionMeta", VERSION_META_CLASS)?;

        Ok(Fragment {
            id,
            files,
            deletion_file,
            physical_rows: Some(physical_rows),
            row_id_meta,
            created_at_version_meta,
            last_updated_at_version_meta,
            // Overlays are not exposed to Java yet, and the reverse conversion
            // does not export them, so this round-trip is overlay-free.
            overlays: vec![],
        })
    }
}

impl FromJObjectWithEnv<DeletionFile> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DeletionFile> {
        let id = env.call_method(self, "getId", "()J", &[])?.j()? as u64;
        let read_version = env.call_method(self, "getReadVersion", "()J", &[])?.j()? as u64;
        let num_deleted_rows: Option<i64> = env
            .call_method(self, "getNumDeletedRows", "()Ljava/lang/Long;", &[])?
            .l()?
            .extract_object(env)?;
        let num_deleted_rows = num_deleted_rows.map(|r| r as usize);
        let file_type: DeletionFileType = env
            .call_method(
                self,
                "getFileType",
                format!("()L{};", DELETE_FILE_TYPE_CLASS),
                &[],
            )?
            .l()?
            .extract_object(env)?;
        let base_id = get_base_id(env, self)?;
        Ok(DeletionFile {
            read_version,
            id,
            num_deleted_rows,
            file_type,
            base_id,
        })
    }
}

impl FromJObjectWithEnv<DeletionFileType> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DeletionFileType> {
        let s = env
            .call_method(self, "toString", "()Ljava/lang/String;", &[])?
            .l()?;
        let s: String = env.get_string(&JString::from(s))?.into();
        let t = if s == "ARRAY" {
            DeletionFileType::Array
        } else {
            DeletionFileType::Bitmap
        };
        Ok(t)
    }
}

impl FromJObjectWithEnv<DataFile> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DataFile> {
        let path = env
            .call_method(self, "getPath", "()Ljava/lang/String;", &[])?
            .l()?;
        let path: String = env.get_string(&JString::from(path))?.into();
        let fields = env.call_method(self, "getFields", "()[I", &[])?.l()?;
        let fields = JIntArray::from(fields).extract_object(env)?;
        let column_indices = env
            .call_method(self, "getColumnIndices", "()[I", &[])?
            .l()?;
        let column_indices = JIntArray::from(column_indices).extract_object(env)?;
        let file_major_version = env
            .call_method(self, "getFileMajorVersion", "()I", &[])?
            .i()? as u32;
        let file_minor_version = env
            .call_method(self, "getFileMinorVersion", "()I", &[])?
            .i()? as u32;
        let file_size_bytes: Option<i64> = env
            .call_method(self, "getFileSizeBytes", "()Ljava/lang/Long;", &[])?
            .l()?
            .extract_object(env)?;
        let file_size_bytes =
            file_size_bytes.map_or(Default::default(), |r| CachedFileSize::new(r as u64));
        let base_id = get_base_id(env, self)?;
        Ok(DataFile {
            path,
            fields: fields.into(),
            column_indices: column_indices.into(),
            file_major_version,
            file_minor_version,
            file_size_bytes,
            base_id,
        })
    }
}

fn get_base_id(env: &mut JNIEnv, obj: &JObject) -> Result<Option<u32>> {
    env.get_optional_u32_from_method(obj, "getBaseId")
}

fn convert_to_java_integer<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<u32>,
) -> Result<JObject<'local>> {
    match value {
        Some(base_index) => Ok(env.new_object(
            "java/lang/Integer",
            "(I)V",
            &[JValue::Int(base_index as jint)],
        )?),
        None => Ok(JObject::null()),
    }
}
