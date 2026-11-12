// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{BlockingDataset, NATIVE_DATASET, clone_dataset};
use crate::block_on;
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::session::handle_from_session;
use crate::traits::{IntoJava, export_vec};
use arrow::datatypes::Schema;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ipc::writer::StreamWriter;
use jni::JNIEnv;
use jni::objects::{JLongArray, JObject, JString, JValue};
use jni::sys::{jboolean, jbyteArray, jint, jlong};
use lance::dataset::rowids::row_ids_to_row_addrs;
use lance::dataset::statistics::{DataStatistics, DatasetStatisticsExt};
use lance::dataset::{ProjectionRequest, Version};
use lance::table::format::Fragment;
use std::collections::HashMap;
use std::sync::Arc;

impl BlockingDataset {
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

    pub fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        let rows = block_on(self.inner.count_rows(filter))?;
        Ok(rows)
    }

    pub fn calculate_data_stats(&self) -> Result<DataStatistics> {
        let stats = block_on(Arc::new(self.clone().inner).calculate_data_stats())?;
        Ok(stats)
    }

    pub fn get_table_metadata(&self) -> Result<HashMap<String, String>> {
        Ok(self.inner.metadata().clone())
    }
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetBasePaths<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
) -> JObject<'a> {
    ok_or_throw!(env, inner_get_base_paths(&mut env, jdataset))
}

fn inner_get_base_paths<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
) -> Result<JObject<'local>> {
    let mut base_paths = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset
            .inner
            .manifest()
            .base_paths
            .values()
            .cloned()
            .collect::<Vec<_>>()
    };
    base_paths.sort_by_key(|base_path| base_path.id);
    export_vec(env, &base_paths)
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetColumnLayoutStatistics<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,
) -> JObject<'a> {
    ok_or_throw!(env, inner_get_column_layout_statistics(&mut env, jdataset))
}

/// `Dataset::column_layout_stats` as parallel Java arrays.
fn inner_get_column_layout_statistics<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
) -> Result<JObject<'local>> {
    let stats = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.inner.column_layout_stats()
    };
    let to_int = |value: usize| {
        i32::try_from(value)
            .map_err(|_| Error::runtime_error(format!("{value} does not fit in a Java int")))
    };
    let len = to_int(stats.len())?;
    let fragment_ids: Vec<i64> = stats.iter().map(|s| s.fragment_id as i64).collect();
    let live_file_counts = stats
        .iter()
        .map(|s| to_int(s.live_file_count))
        .collect::<Result<Vec<_>>>()?;
    let ratios: Vec<f64> = stats.iter().map(|s| s.dead_slot_ratio).collect();
    let overlay_counts = stats
        .iter()
        .map(|s| to_int(s.overlay_count))
        .collect::<Result<Vec<_>>>()?;

    let jfragment_ids = env.new_long_array(len)?;
    let jlive_file_counts = env.new_int_array(len)?;
    let jratios = env.new_double_array(len)?;
    let joverlay_counts = env.new_int_array(len)?;
    env.set_long_array_region(&jfragment_ids, 0, &fragment_ids)?;
    env.set_int_array_region(&jlive_file_counts, 0, &live_file_counts)?;
    env.set_double_array_region(&jratios, 0, &ratios)?;
    env.set_int_array_region(&joverlay_counts, 0, &overlay_counts)?;

    let jfile_sizes = env.new_object_array(len, "[J", JObject::null())?;
    let jfields_per_file = env.new_object_array(len, "[I", JObject::null())?;
    for (index, s) in stats.iter().enumerate() {
        let sizes: Vec<i64> = s
            .file_sizes
            .iter()
            .map(|size| size.map_or(-1, |size| size as i64))
            .collect();
        let fields = s
            .fields_per_file
            .iter()
            .map(|count| to_int(*count))
            .collect::<Result<Vec<_>>>()?;
        let jsizes = env.new_long_array(to_int(sizes.len())?)?;
        env.set_long_array_region(&jsizes, 0, &sizes)?;
        env.set_object_array_element(&jfile_sizes, index as i32, &jsizes)?;
        env.delete_local_ref(jsizes)?;
        let jfields = env.new_int_array(to_int(fields.len())?)?;
        env.set_int_array_region(&jfields, 0, &fields)?;
        env.set_object_array_element(&jfields_per_file, index as i32, &jfields)?;
        env.delete_local_ref(jfields)?;
    }

    Ok(env.new_object(
        "org/lance/ColumnLayoutStatistics",
        "([J[I[[J[[I[D[I)V",
        &[
            JValue::Object(&jfragment_ids),
            JValue::Object(&jlive_file_counts),
            JValue::Object(&jfile_sizes),
            JValue::Object(&jfields_per_file),
            JValue::Object(&jratios),
            JValue::Object(&joverlay_counts),
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
    let dataset = clone_dataset(env, &java_dataset)?;
    dataset.count_versions()
}

fn inner_list_versions<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let versions = {
        let dataset = clone_dataset(env, &java_dataset)?;
        dataset.list_versions()?
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
    let dataset = clone_dataset(env, &java_dataset)?;
    dataset.latest_version()
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
        let dataset = clone_dataset(env, &java_dataset)?;
        dataset.latest_storage_options()?
    };
    match storage_options {
        Some(opts) => opts.into_java(env),
        None => Ok(JObject::null()),
    }
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
    let dataset = clone_dataset(env, &java_dataset)?;
    dataset.count_rows(filter)
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
        let dataset = clone_dataset(env, &java_dataset)?;
        dataset.calculate_data_stats()?
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
pub extern "system" fn Java_org_lance_Dataset_nativeGetRowAddresses<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    row_ids_obj: JObject, // List<Long>
) -> JLongArray<'local> {
    ok_or_throw_with_return!(
        env,
        inner_get_row_addresses(&mut env, java_dataset, row_ids_obj),
        unsafe { JLongArray::from_raw(std::ptr::null_mut()) }
    )
}

/// Unresolved ids are returned as -1, which the Java side maps to null.
fn inner_get_row_addresses<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    row_ids_obj: JObject,
) -> Result<JLongArray<'local>> {
    let row_ids: Vec<u64> = env
        .get_longs(&row_ids_obj)?
        .into_iter()
        .map(|id| id as u64)
        .collect();
    let addrs = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(row_ids_to_row_addrs(&dataset_guard.inner, &row_ids))?
    };
    let addrs: Vec<i64> = addrs
        .into_iter()
        .map(|addr| addr.map_or(-1, |addr| addr as i64))
        .collect();
    let arr = env.new_long_array(addrs.len() as i32)?;
    env.set_long_array_region(&arr, 0, &addrs)?;
    Ok(arr)
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
        let dataset = clone_dataset(env, &java_dataset)?.inner;

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
        let dataset = clone_dataset(env, &java_dataset)?.inner;

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
        let dataset = clone_dataset(env, &java_dataset)?.inner;

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
        let dataset = clone_dataset(env, &java_dataset)?;
        dataset.get_table_metadata()?
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
