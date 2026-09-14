// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array};
use arrow_schema::{DataType, Field};
use jni::JNIEnv;
use jni::objects::{JFloatArray, JMap, JObject, JString, JValue, JValueGen};
use jni::sys::{jboolean, jfloat, jlong};
use lance::dataset::optimize::{CompactionMode, CompactionOptions};
use lance::dataset::{WriteMode, WriteParams};
use lance::index::vector::{IndexFileVersion, StageParams, VectorIndexParams};
use lance::io::ObjectStoreParams;
use lance_file::version::LanceFileVersion;
use lance_index::IndexParams;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::DistanceType;

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;

use crate::traits::FromJObjectWithEnv;
use lance_index::vector::{ApproxMode, Query};
use std::collections::HashMap;
use std::str::FromStr;

pub fn extract_storage_options(
    env: &mut JNIEnv,
    storage_options_obj: &JObject,
) -> Result<HashMap<String, String>> {
    let jmap = JMap::from_env(env, storage_options_obj)?;
    let storage_options: HashMap<String, String> = to_rust_map(env, &jmap)?;
    Ok(storage_options)
}

pub fn extract_base_store_params(
    env: &mut JNIEnv,
    base_store_params_obj: &JObject,
) -> Result<HashMap<String, ObjectStoreParams>> {
    if base_store_params_obj.is_null() {
        return Ok(HashMap::new());
    }

    env.with_local_frame(16, |env| {
        let jmap = JMap::from_env(env, base_store_params_obj)?;
        let mut base_store_params = HashMap::new();
        let mut iter = jmap.iter(env)?;

        while let Some((key, value)) = iter.next(env)? {
            if value.is_null() {
                return Err(Error::input_error(
                    "baseStoreParams values must be non-null maps".to_string(),
                ));
            }

            let key_jstring = JString::from(key);
            let base_path: String = env.get_string(&key_jstring)?.into();
            let storage_options = extract_storage_options(env, &value)?;
            base_store_params.insert(
                base_path,
                ObjectStoreParams {
                    storage_options_accessor: Some(Arc::new(
                        lance::io::StorageOptionsAccessor::with_static_options(storage_options),
                    )),
                    ..Default::default()
                },
            );
        }

        Ok::<_, Error>(base_store_params)
    })
}

pub(crate) fn parse_approx_mode(value: &str) -> Result<ApproxMode> {
    match value {
        "fast" => Ok(ApproxMode::Fast),
        "normal" => Ok(ApproxMode::Normal),
        "accurate" => Ok(ApproxMode::Accurate),
        _ => Err(Error::input_error(format!(
            "Invalid approx mode '{}'. Expected one of: fast, normal, accurate",
            value
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn extract_write_params(
    env: &mut JNIEnv,
    max_rows_per_file: &JObject,
    max_rows_per_group: &JObject,
    max_bytes_per_file: &JObject,
    mode: &JObject,
    enable_stable_row_ids: &JObject,
    data_storage_version: &JObject,
    enable_v2_manifest_paths: Option<&JObject>,
    storage_options_obj: &JObject,
    base_store_params_obj: &JObject,
    initial_bases: &JObject,                     // Optional<BasePath>
    target_bases: &JObject,                      // Optional<String>
    allow_external_blob_outside_bases: &JObject, // Optional<Boolean>
    blob_pack_file_size_threshold: &JObject,     // Optional<Long>
) -> Result<WriteParams> {
    let mut write_params = WriteParams::default();

    if let Some(max_rows_per_file_val) = env.get_int_opt(max_rows_per_file)? {
        write_params.max_rows_per_file = max_rows_per_file_val as usize;
    }
    if let Some(max_rows_per_group_val) = env.get_int_opt(max_rows_per_group)? {
        write_params.max_rows_per_group = max_rows_per_group_val as usize;
    }
    if let Some(max_bytes_per_file_val) = env.get_long_opt(max_bytes_per_file)? {
        write_params.max_bytes_per_file = max_bytes_per_file_val as usize;
    }
    if let Some(mode_val) = env.get_string_opt(mode)? {
        write_params.mode = WriteMode::try_from(mode_val.as_str())?;
    }
    if let Some(enable_stable_row_ids_val) = env.get_boolean_opt(enable_stable_row_ids)? {
        write_params.enable_stable_row_ids = enable_stable_row_ids_val;
    }
    if let Some(data_storage_version_val) = env.get_string_opt(data_storage_version)? {
        write_params.data_storage_version = Some(LanceFileVersion::from_str(
            data_storage_version_val.as_str(),
        )?);
    }

    // Enable v2 manifest paths by default.
    write_params.enable_v2_manifest_paths =
        if let Some(enable_v2_manifest_paths) = enable_v2_manifest_paths {
            env.get_boolean_opt(enable_v2_manifest_paths)?
                .unwrap_or(true)
        } else {
            true
        };

    let storage_options: HashMap<String, String> =
        extract_storage_options(env, storage_options_obj)?;
    let base_store_params = extract_base_store_params(env, base_store_params_obj)?;

    if let Some(initial_bases) =
        env.get_list_opt(initial_bases, |env, elem| elem.extract_object(env))?
    {
        write_params.initial_bases = Some(initial_bases);
    }

    if let Some(names) = env.get_strings_opt(target_bases)? {
        write_params.target_base_names_or_paths = Some(names);
    }

    if let Some(allow) = env.get_boolean_opt(allow_external_blob_outside_bases)? {
        write_params.allow_external_blob_outside_bases = allow;
    }
    if let Some(max_bytes) = env.get_long_opt(blob_pack_file_size_threshold)? {
        write_params.blob_pack_file_size_threshold = Some(max_bytes as usize);
    }

    // Create storage options accessor from static storage_options
    let accessor = if storage_options.is_empty() {
        None
    } else {
        Some(Arc::new(
            lance::io::StorageOptionsAccessor::with_static_options(storage_options),
        ))
    };

    write_params.store_params = Some(ObjectStoreParams {
        storage_options_accessor: accessor,
        ..Default::default()
    });
    for (base_path, store_params) in base_store_params {
        write_params = write_params.with_base_store_params(base_path, store_params);
    }
    Ok(write_params)
}

#[allow(clippy::too_many_arguments)]
pub fn build_compaction_options(
    env: &mut JNIEnv,
    target_rows_per_fragment: &JObject,        // Optional<Long>
    max_rows_per_group: &JObject,              // Optional<Long>
    max_bytes_per_file: &JObject,              // Optional<Long>
    materialize_deletions: &JObject,           // Optional<Boolean>
    materialize_deletions_threshold: &JObject, // Optional<Float>
    num_threads: &JObject,                     // Optional<Long>
    batch_size: &JObject,                      // Optional<Long>
    defer_index_remap: &JObject,               // Optional<Boolean>
    compaction_mode: &JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: &JObject,    // Optional<Long>
    max_source_fragments: &JObject,            // Optional<Long>
    max_source_rows: &JObject,                 // Optional<Long>
    max_source_bytes: &JObject,                // Optional<Long>
    excluded_fragment_ids: &JObject,           // List<Long>
    data_storage_version: &JObject,            // Optional<String>
    config: &std::collections::HashMap<String, String>,
) -> Result<CompactionOptions> {
    let mut compaction_options = CompactionOptions::from_dataset_config(config)?;

    if let Some(target_rows_per_fragment_val) = env.get_long_opt(target_rows_per_fragment)? {
        compaction_options.target_rows_per_fragment = target_rows_per_fragment_val as usize;
    }
    if let Some(max_rows_per_group_val) = env.get_long_opt(max_rows_per_group)? {
        compaction_options.max_rows_per_group = max_rows_per_group_val as usize;
    }
    if let Some(max_bytes_per_file_val) = env.get_long_opt(max_bytes_per_file)? {
        compaction_options.max_bytes_per_file = Some(max_bytes_per_file_val as usize);
    }
    if let Some(materialize_deletions_val) = env.get_boolean_opt(materialize_deletions)? {
        compaction_options.materialize_deletions = materialize_deletions_val;
    }
    if let Some(materialize_deletions_threshold_val) =
        env.get_f32_opt(materialize_deletions_threshold)?
    {
        compaction_options.materialize_deletions_threshold = materialize_deletions_threshold_val;
    }
    if let Some(num_threads_val) = env.get_long_opt(num_threads)? {
        compaction_options.num_threads = Some(num_threads_val as usize);
    }
    if let Some(batch_size_val) = env.get_long_opt(batch_size)? {
        compaction_options.batch_size = Some(batch_size_val as usize);
    }
    if let Some(defer_index_remap_val) = env.get_boolean_opt(defer_index_remap)? {
        compaction_options.defer_index_remap = defer_index_remap_val;
    }
    if let Some(compaction_mode_val) = env.get_string_opt(compaction_mode)? {
        compaction_options.compaction_mode =
            Some(CompactionMode::try_from(compaction_mode_val.as_str())?);
    }
    if let Some(binary_copy_read_batch_bytes_val) =
        env.get_long_opt(binary_copy_read_batch_bytes)?
    {
        compaction_options.binary_copy_read_batch_bytes =
            Some(binary_copy_read_batch_bytes_val as usize);
    }
    if let Some(max_source_fragments_val) = env.get_long_opt(max_source_fragments)? {
        compaction_options.max_source_fragments = Some(max_source_fragments_val as usize);
    }
    if let Some(max_source_rows_val) = env.get_long_opt(max_source_rows)? {
        compaction_options.max_source_rows = Some(max_source_rows_val as usize);
    }
    if let Some(max_source_bytes_val) = env.get_long_opt(max_source_bytes)? {
        compaction_options.max_source_bytes = Some(max_source_bytes_val as u64);
    }
    compaction_options.excluded_fragment_ids = env
        .get_longs(excluded_fragment_ids)?
        .into_iter()
        .map(|fragment_id| {
            u32::try_from(fragment_id).map_err(|_| {
                Error::input_error(format!(
                    "excluded_fragment_ids must contain values between 0 and {}, got {}",
                    u32::MAX,
                    fragment_id
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(version) = env.get_string_opt(data_storage_version)? {
        compaction_options.data_storage_version = Some(LanceFileVersion::from_str(&version)?);
    }

    Ok(compaction_options)
}

// Convert from Java Optional<Query> to Rust Option<Query>
pub fn get_query(env: &mut JNIEnv, query_obj: JObject) -> Result<Option<Query>> {
    let query = env.get_optional(&query_obj, |env, java_obj| {
        let column = env.get_string_from_method(&java_obj, "getColumn")?;
        let key_array = env.get_vec_f32_from_method(&java_obj, "getKey")?;
        let key = Arc::new(Float32Array::from(key_array));

        let k = env.get_int_as_usize_from_method(&java_obj, "getK")?;
        let minimum_nprobes = env.get_int_as_usize_from_method(&java_obj, "getMinimumNprobes")?;
        let maximum_nprobes = env.get_optional_usize_from_method(&java_obj, "getMaximumNprobes")?;

        let ef = env.get_optional_usize_from_method(&java_obj, "getEf")?;

        let refine_factor = env.get_optional_u32_from_method(&java_obj, "getRefineFactor")?;

        let distance_type = if let Some(distance_type_str) =
            env.get_optional_string_from_method(&java_obj, "getDistanceTypeString")?
        {
            Some(DistanceType::try_from(distance_type_str.as_str())?)
        } else {
            None
        };

        let use_index = env.get_boolean_from_method(&java_obj, "isUseIndex")?;
        let query_parallelism = env
            .call_method(&java_obj, "getQueryParallelism", "()I", &[])?
            .i()?;
        let approx_mode_str = env.get_string_from_method(&java_obj, "getApproxModeString")?;
        let approx_mode = parse_approx_mode(&approx_mode_str)?;

        Ok(Query {
            column,
            key,
            k,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes,
            maximum_nprobes,
            ef,
            refine_factor,
            metric_type: distance_type,
            use_index,
            dist_q_c: 0.0,
            query_parallelism,
            approx_mode,
        })
    })?;

    Ok(query)
}

pub fn get_vector_index_params(
    env: &mut JNIEnv,
    index_params_obj: JObject,
) -> Result<Box<dyn IndexParams>> {
    let vector_index_params_option = env.get_optional_from_method(
        &index_params_obj,
        "getVectorIndexParams",
        |env, vector_index_params_obj| {
            // Get distance type from VectorIndexParams
            let distance_type_obj: JString = env
                .call_method(
                    &vector_index_params_obj,
                    "getDistanceTypeString",
                    "()Ljava/lang/String;",
                    &[],
                )?
                .l()?
                .into();
            let distance_type_str: String = env.get_string(&distance_type_obj)?.into();
            let distance_type = DistanceType::try_from(distance_type_str.as_str())?;

            let ivf_params_obj = env
                .call_method(
                    &vector_index_params_obj,
                    "getIvfParams",
                    "()Lorg/lance/index/vector/IvfBuildParams;",
                    &[],
                )?
                .l()?;

            let mut stages = Vec::new();

            // Parse IvfBuildParams
            let num_partitions =
                env.get_int_as_usize_from_method(&ivf_params_obj, "getNumPartitions")?;
            let max_iters = env.get_int_as_usize_from_method(&ivf_params_obj, "getMaxIters")?;
            let sample_rate = env.get_int_as_usize_from_method(&ivf_params_obj, "getSampleRate")?;
            let shuffle_partition_batches =
                env.get_int_as_usize_from_method(&ivf_params_obj, "getShufflePartitionBatches")?;
            let shuffle_partition_concurrency = env
                .get_int_as_usize_from_method(&ivf_params_obj, "getShufflePartitionConcurrency")?;

            let mut ivf_params = IvfBuildParams {
                num_partitions: Some(num_partitions),
                max_iters,
                sample_rate,
                shuffle_partition_batches,
                shuffle_partition_concurrency,
                ..Default::default()
            };

            // Optional pre-trained IVF centroids from Java IvfBuildParams
            // Method signature: float[] getCentroids()
            let centroids_obj = env
                .call_method(&ivf_params_obj, "getCentroids", "()[F", &[])?
                .l()?;

            if !centroids_obj.is_null() {
                let jarray: JFloatArray = centroids_obj.into();
                let length = env.get_array_length(&jarray)?;
                if length > 0 {
                    if !(length as usize).is_multiple_of(num_partitions) {
                        return Err(Error::input_error(format!(
                            "Invalid IVF centroids: length {} is not divisible by num_partitions {}",
                            length, num_partitions
                        )));
                    }
                    let mut buffer = vec![0.0f32; length as usize];
                    env.get_float_array_region(&jarray, 0, &mut buffer)?;
                    let dimension = buffer.len() / num_partitions;

                    let values = Float32Array::from(buffer);
                    let fsl = FixedSizeListArray::try_new(
                        Arc::new(Field::new("item", DataType::Float32, false)),
                        dimension as i32,
                        Arc::new(values) as ArrayRef,
                        None,
                    )
                    .map_err(|e| {
                        Error::input_error(format!(
                            "Failed to construct FixedSizeListArray for IVF centroids: {e}"
                        ))
                    })?;

                    ivf_params.centroids = Some(Arc::new(fsl));
                }
            }

            stages.push(StageParams::Ivf(ivf_params));

            // Parse HnswBuildParams
            let hnsw_params = env.get_optional_from_method(
                &vector_index_params_obj,
                "getHnswParams",
                |env, hnsw_obj| {
                    let max_level =
                        env.call_method(&hnsw_obj, "getMaxLevel", "()S", &[])?.s()? as u16;
                    let m = env.get_int_as_usize_from_method(&hnsw_obj, "getM")?;
                    let ef_construction =
                        env.get_int_as_usize_from_method(&hnsw_obj, "getEfConstruction")?;
                    let prefetch_distance =
                        env.get_optional_usize_from_method(&hnsw_obj, "getPrefetchDistance")?;

                    Ok(HnswBuildParams {
                        max_level,
                        m,
                        ef_construction,
                        prefetch_distance,
                    })
                },
            )?;

            if let Some(hnsw_params) = hnsw_params {
                stages.push(StageParams::Hnsw(hnsw_params));
            }

            // Parse PQBuildParams
            let pq_params = env.get_optional_from_method(
                &vector_index_params_obj,
                "getPqParams",
                |env, pq_obj| {
                    let num_sub_vectors =
                        env.get_int_as_usize_from_method(&pq_obj, "getNumSubVectors")?;
                    let num_bits = env.get_int_as_usize_from_method(&pq_obj, "getNumBits")?;
                    let max_iters = env.get_int_as_usize_from_method(&pq_obj, "getMaxIters")?;
                    let kmeans_redos =
                        env.get_int_as_usize_from_method(&pq_obj, "getKmeansRedos")?;
                    let sample_rate = env.get_int_as_usize_from_method(&pq_obj, "getSampleRate")?;

                    // Optional pre-trained PQ codebook from Java PQBuildParams
                    // Method signature: float[] getCodebook()
                    let codebook_obj = env
                        .call_method(&pq_obj, "getCodebook", "()[F", &[])?
                        .l()?;

                    let codebook = if !codebook_obj.is_null() {
                        let jarray: JFloatArray = codebook_obj.into();
                        let length = env.get_array_length(&jarray)?;
                        if length > 0 {
                            let mut buffer = vec![0.0f32; length as usize];
                            env.get_float_array_region(&jarray, 0, &mut buffer)?;
                            let values = Float32Array::from(buffer);
                            Some(Arc::new(values) as _)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    Ok(PQBuildParams {
                        num_sub_vectors,
                        num_bits,
                        max_iters,
                        kmeans_redos,
                        codebook,
                        sample_rate,
                    })
                },
            )?;

            if let Some(pq_params) = pq_params {
                stages.push(StageParams::PQ(pq_params));
            }

            // Parse SQBuildParams
            let sq_params = env.get_optional_from_method(
                &vector_index_params_obj,
                "getSqParams",
                |env, sq_obj| {
                    let num_bits = env.call_method(&sq_obj, "getNumBits", "()S", &[])?.s()? as u16;
                    let sample_rate = env.get_int_as_usize_from_method(&sq_obj, "getSampleRate")?;

                    Ok(SQBuildParams {
                        num_bits,
                        sample_rate,
                    })
                },
            )?;

            if let Some(sq_params) = sq_params {
                stages.push(StageParams::SQ(sq_params));
            }

            // Parse RQBuildParams
            let rq_params = env.get_optional_from_method(
                &vector_index_params_obj,
                "getRqParams",
                |env, rq_obj| {
                    let num_bits = env.call_method(&rq_obj, "getNumBits", "()B", &[])?.b()? as u8;
                    Ok(RQBuildParams::new(num_bits))
                },
            )?;

            if let Some(rq_params) = rq_params {
                stages.push(StageParams::RQ(rq_params));
            }

            Ok(VectorIndexParams {
                metric_type: distance_type,
                stages,
                version: IndexFileVersion::V3,
                skip_transpose: false,
                runtime_hints: Default::default(),
            })
        },
    )?;

    match vector_index_params_option {
        Some(params) => Ok(Box::new(params) as Box<dyn IndexParams>),
        None => Err(Error::input_error(
            "VectorIndexParams not present".to_string(),
        )),
    }
}

pub fn get_scalar_index_params(
    env: &mut JNIEnv,
    index_params_obj: JObject,
) -> Result<(String, Option<String>)> {
    env.get_optional_from_method(
        &index_params_obj,
        "getScalarIndexParams",
        |env, scalar_params_obj| {
            let index_type = env.get_string_from_method(&scalar_params_obj, "getIndexType")?;

            let params = env.get_optional_from_method(
                &scalar_params_obj,
                "getJsonParams",
                |env, params_obj| {
                    let params_str: JString = params_obj.into();
                    let params_string: String = env.get_string(&params_str)?.into();
                    Ok(params_string)
                },
            )?;

            Ok((index_type, params))
        },
    )?
    .ok_or_else(|| Error::input_error("ScalarIndexParams not present".to_string()))
}

pub fn to_rust_map(env: &mut JNIEnv, jmap: &JMap) -> Result<HashMap<String, String>> {
    env.with_local_frame(16, |env| {
        let mut map = HashMap::new();
        let mut iter = jmap.iter(env)?;

        while let Some((key, value)) = iter.next(env)? {
            let key_jstring = JString::from(key);
            let value_jstring = JString::from(value);
            let key_string: String = env.get_string(&key_jstring)?.into();
            let value_string: String = env.get_string(&value_jstring)?.into();
            map.insert(key_string, value_string);
        }

        Ok::<_, Error>(map)
    })
}

pub fn to_java_map<'local>(
    env: &mut JNIEnv<'local>,
    map: &HashMap<String, String>,
) -> Result<JObject<'local>> {
    let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
    for (k, v) in map {
        let jkey = env.new_string(k)?;
        let jval = env.new_string(v)?;
        env.call_method(
            &java_map,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&jkey), JValue::Object(&jval)],
        )?;
    }
    Ok(java_map)
}

pub fn to_java_list<'local>(
    env: &mut JNIEnv<'local>,
    list: &Vec<JObject>,
) -> Result<JObject<'local>> {
    let java_list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for item in list {
        env.call_method(
            &java_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(item)],
        )?;
    }
    Ok(java_list)
}

pub fn to_java_optional<'local>(
    env: &mut JNIEnv<'local>,
    value: JObject,
) -> Result<JObject<'local>> {
    Ok(env
        .call_static_method(
            "java/util/Optional",
            "ofNullable",
            "(Ljava/lang/Object;)Ljava/util/Optional;",
            &[JValueGen::Object(&value)],
        )?
        .l()?)
}

pub fn to_java_long_obj<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<i64>,
) -> Result<JObject<'local>> {
    match value {
        Some(base_index) => Ok(env.new_object(
            "java/lang/Long",
            "(J)V",
            &[JValue::Long(base_index as jlong)],
        )?),
        None => Ok(JObject::null()),
    }
}

pub fn to_java_boolean_obj<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<bool>,
) -> Result<JObject<'local>> {
    match value {
        Some(base_index) => Ok(env.new_object(
            "java/lang/Boolean",
            "(Z)V",
            &[JValue::Bool(base_index as jboolean)],
        )?),
        None => Ok(JObject::null()),
    }
}

pub fn to_java_float_obj<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<f32>,
) -> Result<JObject<'local>> {
    match value {
        Some(base_index) => Ok(env.new_object(
            "java/lang/Float",
            "(F)V",
            &[JValue::Float(base_index as jfloat)],
        )?),
        None => Ok(JObject::null()),
    }
}

pub fn to_java_double_obj<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<f64>,
) -> Result<JObject<'local>> {
    match value {
        Some(v) => Ok(env.new_object("java/lang/Double", "(D)V", &[JValue::Double(v)])?),
        None => Ok(JObject::null()),
    }
}

pub fn to_java_string_obj<'local>(
    env: &mut JNIEnv<'local>,
    value: Option<&str>,
) -> Result<JObject<'local>> {
    match value {
        Some(v) => {
            let jstr = env.new_string(v)?;
            Ok(jstr.into())
        }
        None => Ok(JObject::null()),
    }
}

/// Convert a DataFusion ScalarValue to a Java Comparable object.
///
/// Maps numeric types to their boxed Java equivalents (Long, Double)
/// and string types to Java String. Null ScalarValues produce JObject::null().
///
/// This is useful for exposing index statistics (e.g., zonemap min/max)
/// to Java clients in a type-safe, Comparable-compatible way.
pub fn scalar_value_to_java<'a>(
    env: &mut JNIEnv<'a>,
    value: &datafusion_common::ScalarValue,
) -> Result<JObject<'a>> {
    use datafusion_common::ScalarValue;

    match value {
        ScalarValue::Null => Ok(JObject::null()),

        ScalarValue::Boolean(v) => to_java_boolean_obj(env, *v),

        // Integer types → Java Long
        ScalarValue::Int8(v) => to_java_long_obj(env, v.map(|x| x as i64)),
        ScalarValue::Int16(v) => to_java_long_obj(env, v.map(|x| x as i64)),
        ScalarValue::Int32(v) => to_java_long_obj(env, v.map(|x| x as i64)),
        ScalarValue::Int64(v) => to_java_long_obj(env, *v),
        ScalarValue::UInt8(v) => to_java_long_obj(env, v.map(|x| x as i64)),
        ScalarValue::UInt16(v) => to_java_long_obj(env, v.map(|x| x as i64)),
        ScalarValue::UInt32(v) => to_java_long_obj(env, v.map(|x| x as i64)),
        // UInt64 may overflow i64, but for min/max stats this is acceptable
        ScalarValue::UInt64(v) => to_java_long_obj(env, v.map(|x| x as i64)),

        // Float types → Java Double
        ScalarValue::Float16(v) => to_java_double_obj(env, v.map(|x| f64::from(x.to_f32()))),
        ScalarValue::Float32(v) => to_java_double_obj(env, v.map(|x| x as f64)),
        ScalarValue::Float64(v) => to_java_double_obj(env, *v),

        // String types → Java String
        ScalarValue::Utf8(v) => to_java_string_obj(env, v.as_deref()),
        ScalarValue::LargeUtf8(v) => to_java_string_obj(env, v.as_deref()),

        // Date types → Java Long
        ScalarValue::Date32(v) => to_java_long_obj(env, v.map(|x| x as i64)),
        ScalarValue::Date64(v) => to_java_long_obj(env, *v),

        // Timestamp types → Java Long
        ScalarValue::TimestampSecond(v, _) => to_java_long_obj(env, *v),
        ScalarValue::TimestampMillisecond(v, _) => to_java_long_obj(env, *v),
        ScalarValue::TimestampMicrosecond(v, _) => to_java_long_obj(env, *v),
        ScalarValue::TimestampNanosecond(v, _) => to_java_long_obj(env, *v),

        // For any unsupported type, return null (conservative: caller will skip)
        _ => {
            log::warn!(
                "Unsupported ScalarValue type for Java conversion: {:?}",
                value.data_type()
            );
            Ok(JObject::null())
        }
    }
}
