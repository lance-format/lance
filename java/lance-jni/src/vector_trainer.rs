// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::block_on;
use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;

use arrow::array::{FixedSizeListArray, Float32Array};
use jni::JNIEnv;
use jni::objects::{JClass, JFloatArray, JObject, JString};
use jni::sys::jfloatArray;
use lance::index::NoopIndexBuildProgress;
use lance::index::vector::utils::get_vector_dim;
use lance_index::vector::ivf::builder::IvfBuildParams as RustIvfBuildParams;
use lance_index::vector::pq::builder::PQBuildParams as RustPQBuildParams;
use lance_linalg::distance::MetricType;

/// Flatten a FixedSizeList<Float32> into a contiguous Vec<f32>.
fn flatten_fixed_size_list_to_f32(arr: &FixedSizeListArray) -> Result<Vec<f32>> {
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            Error::input_error(format!(
                "Expected FixedSizeList<Float32>, got value type {}",
                arr.value_type()
            ))
        })?;

    Ok(values.values().to_vec())
}

fn parse_distance_type(env: &mut JNIEnv, distance_type_jstr: &JString) -> Result<MetricType> {
    let distance_type_str: String = env.get_string(distance_type_jstr)?.into();
    MetricType::try_from(distance_type_str.as_str()).map_err(|e| Error::input_error(e.to_string()))
}

fn build_ivf_params_from_java(
    env: &mut JNIEnv,
    ivf_params_obj: &JObject,
) -> Result<RustIvfBuildParams> {
    let num_partitions = env.get_int_as_usize_from_method(ivf_params_obj, "getNumPartitions")?;
    let max_iters = env.get_int_as_usize_from_method(ivf_params_obj, "getMaxIters")?;
    let sample_rate = env.get_int_as_usize_from_method(ivf_params_obj, "getSampleRate")?;
    let shuffle_partition_batches =
        env.get_int_as_usize_from_method(ivf_params_obj, "getShufflePartitionBatches")?;
    let shuffle_partition_concurrency =
        env.get_int_as_usize_from_method(ivf_params_obj, "getShufflePartitionConcurrency")?;

    Ok(RustIvfBuildParams {
        num_partitions: Some(num_partitions),
        max_iters,
        sample_rate,
        shuffle_partition_batches,
        shuffle_partition_concurrency,
        ..Default::default()
    })
}

fn build_pq_params_from_java(
    env: &mut JNIEnv,
    pq_params_obj: &JObject,
) -> Result<RustPQBuildParams> {
    let num_sub_vectors = env.get_int_as_usize_from_method(pq_params_obj, "getNumSubVectors")?;
    let num_bits = env.get_int_as_usize_from_method(pq_params_obj, "getNumBits")?;
    let max_iters = env.get_int_as_usize_from_method(pq_params_obj, "getMaxIters")?;
    let kmeans_redos = env.get_int_as_usize_from_method(pq_params_obj, "getKmeansRedos")?;
    let sample_rate = env.get_int_as_usize_from_method(pq_params_obj, "getSampleRate")?;

    Ok(RustPQBuildParams {
        num_sub_vectors,
        num_bits,
        max_iters,
        kmeans_redos,
        codebook: None,
        sample_rate,
        kmeans_seed: None,
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_VectorTrainer_nativeTrainIvfCentroids<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dataset_obj: JObject<'local>,        // org.lance.Dataset
    column_jstr: JString<'local>,        // java.lang.String
    ivf_params_obj: JObject<'local>,     // org.lance.index.vector.IvfBuildParams
    distance_type_jstr: JString<'local>, // org.lance.index.DistanceType#toString()
) -> jfloatArray {
    ok_or_throw_with_return!(
        env,
        inner_train_ivf_centroids(
            &mut env,
            dataset_obj,
            column_jstr,
            ivf_params_obj,
            distance_type_jstr,
        )
        .map(|arr| arr.into_raw()),
        JFloatArray::default().into_raw()
    )
}

fn inner_train_ivf_centroids<'local>(
    env: &mut JNIEnv<'local>,
    dataset_obj: JObject<'local>,
    column_jstr: JString<'local>,
    ivf_params_obj: JObject<'local>,
    distance_type_jstr: JString<'local>,
) -> Result<JFloatArray<'local>> {
    let column: String = env.get_string(&column_jstr)?.into();
    let ivf_params = build_ivf_params_from_java(env, &ivf_params_obj)?;
    let metric_type = parse_distance_type(env, &distance_type_jstr)?;

    let flattened: Vec<f32> = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(dataset_obj, NATIVE_DATASET) }?;
        let dataset = &dataset_guard.inner;

        let dim = get_vector_dim(dataset.schema(), &column)?;

        let ivf_model = block_on(lance::index::vector::ivf::build_ivf_model(
            dataset,
            &column,
            dim,
            metric_type,
            &ivf_params,
            None,
            Arc::new(NoopIndexBuildProgress),
        ))?;

        let centroids = ivf_model
            .centroids
            .ok_or_else(|| Error::runtime_error("IVF model missing centroids".to_string()))?;

        flatten_fixed_size_list_to_f32(&centroids)?
    };

    let jarray = env.new_float_array(flattened.len() as i32)?;
    env.set_float_array_region(&jarray, 0, &flattened)?;
    Ok(jarray)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_VectorTrainer_nativeTrainPqCodebook<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dataset_obj: JObject<'local>,        // org.lance.Dataset
    column_jstr: JString<'local>,        // java.lang.String
    pq_params_obj: JObject<'local>,      // org.lance.index.vector.PQBuildParams
    distance_type_jstr: JString<'local>, // org.lance.index.DistanceType#toString()
) -> jfloatArray {
    ok_or_throw_with_return!(
        env,
        inner_train_pq_codebook(
            &mut env,
            dataset_obj,
            column_jstr,
            pq_params_obj,
            distance_type_jstr,
        )
        .map(|arr| arr.into_raw()),
        JFloatArray::default().into_raw()
    )
}

fn inner_train_pq_codebook<'local>(
    env: &mut JNIEnv<'local>,
    dataset_obj: JObject<'local>,
    column_jstr: JString<'local>,
    pq_params_obj: JObject<'local>,
    distance_type_jstr: JString<'local>,
) -> Result<JFloatArray<'local>> {
    let column: String = env.get_string(&column_jstr)?.into();
    let pq_params = build_pq_params_from_java(env, &pq_params_obj)?;
    let metric_type = parse_distance_type(env, &distance_type_jstr)?;

    let flattened: Vec<f32> = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(dataset_obj, NATIVE_DATASET) }?;
        let dataset = &dataset_guard.inner;

        let dim = get_vector_dim(dataset.schema(), &column)?;

        let pq = block_on(lance::index::vector::pq::build_pq_model(
            dataset,
            &column,
            dim,
            metric_type,
            &pq_params,
            None,
        ))?;

        flatten_fixed_size_list_to_f32(&pq.codebook)?
    };

    let jarray = env.new_float_array(flattened.len() as i32)?;
    env.set_float_array_region(&jarray, 0, &flattened)?;
    Ok(jarray)
}
