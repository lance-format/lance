// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{BlockingDataset, NATIVE_DATASET, clone_dataset};
use crate::blocking_scanner::build_full_text_search_query;
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::index::IndexDescriptionSummary;
use crate::index_progress::JavaIndexBuildProgress;
use crate::traits::{FromJObjectWithEnv, FromJString, export_vec, import_vec_to_rust};
use crate::utils::{get_scalar_index_params, get_vector_index_params};
use crate::{block_on, traits::IntoJava};
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use jni::JNIEnv;
use jni::objects::{JObject, JString, JValue};
use jni::sys::{jboolean, jbyteArray, jint, jlong};
use lance::dataset::Dataset;
use lance::index::{DatasetIndexExt, IndexSegment, IndexSegmentStatistics, IntoIndexSegment};
use lance::table::format::IndexMetadata;
use lance_index::IndexCriteria as RustIndexCriteria;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::{IndexBuildProgress, noop_progress};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::{IndexParams, IndexType};
use std::future::IntoFuture;
use std::sync::Arc;
use uuid::Uuid;

impl BlockingDataset {
    pub fn list_indexes(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        let indexes = block_on(self.inner.load_indices())?;
        Ok(indexes)
    }
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
        let dataset = clone_dataset(env, &java_dataset)?;
        let indexes = dataset.list_indexes()?;
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
            None,
        )
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCreateIndexWithProgress<'local>(
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
    progress_jobj: JObject<'local>,          // IndexBuildProgress
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_index_with_progress(
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
            progress_jobj,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_index_with_progress<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject<'local>,
    columns_jobj: JObject<'local>,
    index_type_code_jobj: jint,
    name_jobj: JObject<'local>,
    params_jobj: JObject<'local>,
    replace_jobj: jboolean,
    train_jobj: jboolean,
    fragments_jobj: JObject<'local>,
    index_uuid_jobj: JObject<'local>,
    arrow_stream_addr_jobj: JObject<'local>,
    progress_jobj: JObject<'local>,
) -> Result<JObject<'local>> {
    let progress = Arc::new(JavaIndexBuildProgress::new(env, &progress_jobj)?);
    inner_create_index(
        env,
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
        Some(progress),
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
    progress: Option<Arc<dyn IndexBuildProgress>>,
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
        | IndexType::RTree
        | IndexType::MinHashLsh => {
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
        | IndexType::IvfHnswFlat
        | IndexType::IvfHnswRq => {
            // For vector indices, use the existing parameter handling
            get_vector_index_params(env, params_jobj)
        }
    };

    let params = params_result?;

    let index_metadata = if progress.is_some() || skip_commit {
        // Progress callbacks may re-enter read-only methods on this Dataset. Clone the Dataset
        // and release the native field guard before the long-running build so those callbacks
        // do not deadlock on the field mutex. Uncommitted builds never publish a new version, so
        // they also build on a clone instead of serializing every build on this handle.
        let mut working_dataset = {
            let dataset_guard = unsafe {
                env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET)
            }?;
            dataset_guard.inner.clone()
        };
        let initial_version = working_dataset.version().version;
        let index_metadata = execute_create_index(
            &mut working_dataset,
            &columns_slice,
            index_type,
            params.as_ref(),
            replace,
            train,
            name,
            fragment_ids,
            index_uuid,
            batch_reader,
            progress,
            skip_commit,
        )
        .inspect_err(|_| {
            // A committed build may fail while materializing its return metadata. The clone
            // has already observed the durable commit, so publish it before propagating.
            if should_publish_working_dataset_on_error(
                skip_commit,
                initial_version,
                working_dataset.version().version,
            ) {
                match unsafe {
                    env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET)
                } {
                    Ok(mut dataset_guard) => {
                        dataset_guard.inner = working_dataset.clone();
                    }
                    Err(publish_error) => {
                        log::warn!(
                            "Failed to publish committed dataset version {} after create-index failure: {}",
                            working_dataset.version().version,
                            publish_error
                        );
                    }
                }
            }
        })?;

        if !skip_commit {
            let mut dataset_guard = unsafe {
                env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET)
            }?;
            dataset_guard.inner = working_dataset;
        }
        index_metadata
    } else {
        // Preserve the existing committed no-progress path, including its native-field locking
        // behavior.
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        execute_create_index(
            &mut dataset_guard.inner,
            &columns_slice,
            index_type,
            params.as_ref(),
            replace,
            train,
            name,
            fragment_ids,
            index_uuid,
            batch_reader,
            None,
            skip_commit,
        )?
    };

    (&index_metadata).into_java(env)
}

fn should_publish_working_dataset_on_error(
    skip_commit: bool,
    initial_version: u64,
    current_version: u64,
) -> bool {
    !skip_commit && current_version != initial_version
}

#[allow(clippy::too_many_arguments)]
fn execute_create_index(
    dataset: &mut Dataset,
    columns: &[&str],
    index_type: IndexType,
    params: &dyn IndexParams,
    replace: bool,
    train: bool,
    name: Option<String>,
    fragment_ids: Option<Vec<u32>>,
    index_uuid: Option<Uuid>,
    batch_reader: Option<ArrowArrayStreamReader>,
    progress: Option<Arc<dyn IndexBuildProgress>>,
    skip_commit: bool,
) -> Result<IndexMetadata> {
    let mut index_builder = dataset
        .create_index_builder(columns, index_type, params)
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

    if let Some(progress) = progress {
        index_builder = index_builder.progress(progress);
    }

    if skip_commit {
        Ok(block_on(index_builder.execute_uncommitted())?)
    } else {
        Ok(block_on(index_builder.into_future())?)
    }
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

/// `OptimizeOptions` from its Java counterpart; the defaults for `null`.
pub(crate) fn optimize_options_from_java(
    env: &mut JNIEnv,
    java_options: &JObject, // OptimizeOptions
) -> Result<OptimizeOptions> {
    let mut options = OptimizeOptions::default();

    if !java_options.is_null() {
        options.num_indices_to_merge =
            env.get_optional_usize_from_method(java_options, "getNumIndicesToMerge")?;

        // getIndexNames(): Optional<List<String>>
        let index_names_obj = env
            .call_method(java_options, "getIndexNames", "()Ljava/util/Optional;", &[])?
            .l()?;
        let index_names = env.get_strings_opt(&index_names_obj)?;
        options.index_names = index_names;

        // isRetrain(): boolean
        let retrain = env
            .call_method(java_options, "isRetrain", "()Z", &[])?
            .z()?;
        options.retrain = retrain;

        options.max_rows_per_task =
            env.get_optional_u64_from_method(java_options, "getMaxRowsPerTask")?;
        options.max_rows_per_segment =
            env.get_optional_u64_from_method(java_options, "getMaxRowsPerSegment")?;
        options.max_rows_to_consolidate =
            env.get_optional_u64_from_method(java_options, "getMaxRowsToConsolidate")?;
    }

    Ok(options)
}

fn inner_optimize_indices(
    env: &mut JNIEnv,
    java_dataset: JObject,
    java_options: JObject, // OptimizeOptions
) -> Result<()> {
    let options = optimize_options_from_java(env, &java_options)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.optimize_indices(&options))?;
    Ok(())
}

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
        let dataset = clone_dataset(env, &java_dataset)?;
        dataset.list_indexes()?
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
        let dataset = clone_dataset(env, &java_dataset)?;
        block_on(dataset.inner.index_statistics(&index_name))?
    };
    let jstats = env.new_string(stats_json)?;
    Ok(jstats)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetIndexSegmentStatistics<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jindex_name: JString,
    jindex_uuids: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_get_index_segment_statistics(&mut env, java_dataset, jindex_name, jindex_uuids)
    )
}

fn inner_get_index_segment_statistics<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jindex_name: JString,
    jindex_uuids: JObject,
) -> Result<JObject<'local>> {
    let index_name: String = jindex_name.extract(env)?;
    let index_uuids = import_vec_to_rust(env, &jindex_uuids, |env, index_uuid| {
        env.with_local_frame(8, |env| {
            let index_uuid = env.get_string_from_method(&index_uuid, "toString")?;
            Uuid::parse_str(&index_uuid).map_err(|error| Error::input_error(error.to_string()))
        })
    })?;
    let statistics = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(
            dataset_guard
                .inner
                .index_segment_statistics(&index_name, &index_uuids),
        )?
    };
    let results = env.new_object("java/util/ArrayList", "()V", &[])?;
    for statistics in statistics {
        env.with_local_frame(8, |env| {
            let index_uuid = env.new_string(statistics.index_uuid.to_string())?;
            let index_type_uri = env.new_string(statistics.index_type_uri)?;
            let statistics_json = env.new_string(serde_json::to_string(&statistics.statistics)?)?;
            let result = env.new_object(
                "org/lance/index/IndexSegmentStatistics",
                "(JLjava/lang/String;Ljava/lang/String;Ljava/lang/String;)V",
                &[
                    JValue::Long(statistics.read_version as i64),
                    JValue::Object(&index_uuid),
                    JValue::Object(&index_type_uri),
                    JValue::Object(&statistics_json),
                ],
            )?;
            env.call_method(
                &results,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&result)],
            )?;
            Ok::<(), Error>(())
        })?;
    }
    Ok(results)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetIndexStatisticsFromSegments<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jindex_name: JString,
    jsegment_statistics: JObject,
) -> JString<'local> {
    ok_or_throw_with_return!(
        env,
        inner_get_index_statistics_from_segments(
            &mut env,
            java_dataset,
            jindex_name,
            jsegment_statistics,
        ),
        JString::from(JObject::null())
    )
}

fn inner_get_index_statistics_from_segments<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jindex_name: JString,
    jsegment_statistics: JObject,
) -> Result<JString<'local>> {
    let index_name: String = jindex_name.extract(env)?;
    let segment_statistics = import_vec_to_rust(env, &jsegment_statistics, |env, statistics| {
        env.with_local_frame(8, |env| {
            let read_version = env
                .call_method(&statistics, "getReadVersion", "()J", &[])?
                .j()? as u64;
            let index_uuid = env
                .call_method(&statistics, "getIndexUuid", "()Ljava/util/UUID;", &[])?
                .l()?;
            let index_uuid = env.get_string_from_method(&index_uuid, "toString")?;
            let index_type_uri = env.get_string_from_method(&statistics, "getIndexTypeUri")?;
            let statistics_json = env.get_string_from_method(&statistics, "getStatisticsJson")?;
            Ok(IndexSegmentStatistics {
                read_version,
                index_uuid: Uuid::parse_str(&index_uuid)
                    .map_err(|error| Error::input_error(error.to_string()))?,
                index_type_uri,
                statistics: serde_json::from_str(&statistics_json)
                    .map_err(|error| Error::input_error(error.to_string()))?,
            })
        })
    })?;
    let statistics_json = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(
            dataset_guard
                .inner
                .index_statistics_from_segments(&index_name, segment_statistics),
        )?
    };
    Ok(env.new_string(statistics_json)?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetFtsGlobalStatistics(
    mut env: JNIEnv,
    java_dataset: JObject,
    query_obj: JObject,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_get_fts_global_statistics(&mut env, java_dataset, query_obj),
        std::ptr::null_mut()
    )
}

fn inner_get_fts_global_statistics(
    env: &mut JNIEnv,
    java_dataset: JObject,
    query_obj: JObject,
) -> Result<jbyteArray> {
    let query = build_full_text_search_query(env, query_obj)?;
    let mut scanner = {
        let dataset = clone_dataset(env, &java_dataset)?;
        dataset.inner.scan()
    };
    scanner.full_text_search(FullTextSearchQuery::new_query(query))?;
    let statistics = block_on(scanner.fts_global_statistics())?;
    Ok(**env.byte_array_from_slice(&statistics)?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDescribeIndices<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    criteria_obj: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_describe_indices(&mut env, java_dataset, criteria_obj, true)
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDescribeIndexSummaries<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    criteria_obj: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_describe_indices(&mut env, java_dataset, criteria_obj, false)
    )
}

fn inner_describe_indices<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    java_index_criteria: JObject,
    include_metadata: bool,
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
            must_support_minhash: false,
        })
    })?;

    let descriptions = {
        let dataset = clone_dataset(env, &java_dataset)?;
        block_on(dataset.inner.describe_indices(index_criteria))?
    };

    if include_metadata {
        export_vec(env, &descriptions)
    } else {
        let summaries = descriptions
            .iter()
            .map(IndexDescriptionSummary)
            .collect::<Vec<_>>();
        export_vec(env, &summaries)
    }
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
    jindex_name: JString,
    jfilter: JString,
    jfragment_ids: JObject, // Optional<List<Integer>>
) -> Result<i64> {
    let index_name: String = jindex_name.extract(env)?;
    let filter: String = jfilter.extract(env)?;
    let fragment_ids = extract_count_fragment_ids(env, &jfragment_ids)?;

    count_indexed_rows(env, java_dataset, index_name, filter, None, fragment_ids)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCountIndexedRowsWithSegments(
    mut env: JNIEnv,
    java_dataset: JObject,
    jindex_name: JString,
    jfilter: JString,
    jsegment_uuids: JObject, // List<UUID>
    jfragment_ids: JObject,  // Optional<List<Integer>>
) -> jlong {
    ok_or_throw_with_return!(
        env,
        inner_count_indexed_rows_with_segments(
            &mut env,
            java_dataset,
            jindex_name,
            jfilter,
            jsegment_uuids,
            jfragment_ids,
        ),
        -1
    )
}

fn inner_count_indexed_rows_with_segments(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jindex_name: JString,
    jfilter: JString,
    jsegment_uuids: JObject,
    jfragment_ids: JObject,
) -> Result<i64> {
    let index_name: String = jindex_name.extract(env)?;
    let filter: String = jfilter.extract(env)?;
    let segment_uuids =
        import_vec_to_rust(env, &jsegment_uuids, |env, obj| obj.extract_object(env))?;
    let fragment_ids = extract_count_fragment_ids(env, &jfragment_ids)?;

    count_indexed_rows(
        env,
        java_dataset,
        index_name,
        filter,
        Some(segment_uuids),
        fragment_ids,
    )
}

fn extract_count_fragment_ids(
    env: &mut JNIEnv,
    jfragment_ids: &JObject,
) -> Result<Option<Vec<u32>>> {
    env.get_ints_opt(jfragment_ids)?
        .map(|fragment_ids| {
            fragment_ids
                .into_iter()
                .map(|fragment_id| {
                    u32::try_from(fragment_id).map_err(|_| {
                        Error::input_error(format!(
                            "fragment ID must be non-negative, got {fragment_id}"
                        ))
                    })
                })
                .collect()
        })
        .transpose()
}

fn count_indexed_rows(
    env: &mut JNIEnv,
    java_dataset: JObject,
    index_name: String,
    filter: String,
    segment_uuids: Option<Vec<Uuid>>,
    fragment_ids: Option<Vec<u32>>,
) -> Result<i64> {
    let count = {
        let dataset = clone_dataset(env, &java_dataset)?;
        block_on(dataset.inner.count_indexed_rows(
            &index_name,
            &filter,
            segment_uuids.as_deref(),
            fragment_ids.as_deref(),
        ))?
    };

    i64::try_from(count)
        .map_err(|_| Error::input_error(format!("indexed row count {count} exceeds Java long")))
}

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
                    must_support_minhash: false,
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

#[cfg(test)]
mod tests {
    use super::should_publish_working_dataset_on_error;

    #[test]
    fn post_commit_failure_publishes_advanced_dataset() {
        assert!(should_publish_working_dataset_on_error(false, 1, 2));
    }

    #[test]
    fn pre_commit_failure_keeps_original_dataset() {
        assert!(!should_publish_working_dataset_on_error(false, 1, 1));
    }

    #[test]
    fn uncommitted_failure_keeps_original_dataset() {
        assert!(!should_publish_working_dataset_on_error(true, 1, 2));
    }
}
