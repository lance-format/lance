// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::ffi::JNIEnvExt;
use crate::transaction::convert_to_java_transaction;
use crate::{RT, block_on};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use jni::JNIEnv;
use jni::objects::{JObject, JValue};
use jni::sys::jlong;
use lance::dataset::delta::DatasetDelta as RustDatasetDelta;
use lance::dataset::scanner::DatasetRecordBatchStream;
use lance::dataset::transaction::Transaction;
use lance_io::ffi::to_ffi_arrow_array_stream;

pub const NATIVE_DELTA: &str = "nativeDeltaHandle";

pub struct BlockingDatasetDelta {
    pub(crate) inner: RustDatasetDelta,
}

fn attach_native_delta<'local>(
    env: &mut JNIEnv<'local>,
    delta: BlockingDatasetDelta,
    java_dataset: &JObject<'local>,
) -> Result<JObject<'local>> {
    let j_delta = env.new_object("org/lance/delta/DatasetDelta", "()V", &[])?;

    unsafe { env.set_rust_field(&j_delta, NATIVE_DELTA, delta) }?;

    env.set_field(
        &j_delta,
        "dataset",
        "Lorg/lance/Dataset;",
        JValue::Object(java_dataset),
    )?;
    Ok(j_delta)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_delta_DatasetDeltaBuilder_nativeBuild<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject<'local>,
    java_dataset: JObject<'local>,
    compared_against_obj: JObject<'local>,
    begin_version_obj: JObject<'local>,
    end_version_obj: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_native_build(
            &mut env,
            java_dataset,
            compared_against_obj,
            begin_version_obj,
            end_version_obj
        )
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeBuildDelta<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject<'local>,
    compared_against_obj: JObject<'local>,
    begin_version_obj: JObject<'local>,
    end_version_obj: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_native_build(
            &mut env,
            java_dataset,
            compared_against_obj,
            begin_version_obj,
            end_version_obj
        )
    )
}

fn inner_native_build<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject<'local>,
    compared_against_obj: JObject<'local>,
    begin_version_obj: JObject<'local>,
    end_version_obj: JObject<'local>,
) -> Result<JObject<'local>> {
    let compared_against = env.get_u64_opt(&compared_against_obj)?;
    let begin_version = env.get_u64_opt(&begin_version_obj)?;
    let end_version = env.get_u64_opt(&end_version_obj)?;

    let delta = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET)? };

        let mut builder = dataset_guard.inner.delta();
        if let Some(compared) = compared_against {
            builder = builder.compared_against_version(compared);
        } else if let (Some(begin), Some(end)) = (begin_version, end_version) {
            builder = builder.with_begin_version(begin).with_end_version(end);
        }
        builder.build()?
    };

    let blocking_delta = BlockingDatasetDelta { inner: delta };
    attach_native_delta(env, blocking_delta, &java_dataset)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_nativeListTransactions<'local>(
    mut env: JNIEnv<'local>,
    j_delta: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_transactions(&mut env, j_delta))
}

fn inner_list_transactions<'local>(
    env: &mut JNIEnv<'local>,
    j_delta: JObject<'local>,
) -> Result<JObject<'local>> {
    let txs: Vec<Transaction> = {
        let delta_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDatasetDelta>(&j_delta, NATIVE_DELTA) }?;
        block_on(delta_guard.inner.list_transactions())?
    };

    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for tx in txs.into_iter() {
        let jtx = convert_to_java_transaction(env, tx)?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&jtx)],
        )?;
    }
    Ok(array_list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_nativeGetInsertedRows<'local>(
    mut env: JNIEnv<'local>,
    j_delta: JObject<'local>,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_get_inserted_rows(&mut env, j_delta, stream_addr))
}

fn inner_get_inserted_rows<'local>(
    env: &mut JNIEnv,
    j_delta: JObject<'local>,
    stream_addr: jlong,
) -> Result<()> {
    let delta_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDatasetDelta>(&j_delta, NATIVE_DELTA) }?;

    let stream: DatasetRecordBatchStream = block_on(delta_guard.inner.get_inserted_rows())?;
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone())?;

    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_nativeGetUpdatedRows<'local>(
    mut env: JNIEnv<'local>,
    j_delta: JObject<'local>,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_get_updated_rows(&mut env, j_delta, stream_addr))
}

fn inner_get_updated_rows<'local>(
    env: &mut JNIEnv,
    j_delta: JObject<'local>,
    stream_addr: jlong,
) -> Result<()> {
    let delta_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDatasetDelta>(&j_delta, NATIVE_DELTA) }?;

    let stream: DatasetRecordBatchStream = block_on(delta_guard.inner.get_updated_rows())?;
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone())?;

    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_nativeGetDeletedRowIds<'local>(
    mut env: JNIEnv<'local>,
    j_delta: JObject<'local>,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_get_deleted_row_ids(&mut env, j_delta, stream_addr)
    )
}

fn inner_get_deleted_row_ids<'local>(
    env: &mut JNIEnv,
    j_delta: JObject<'local>,
    stream_addr: jlong,
) -> Result<()> {
    let delta_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDatasetDelta>(&j_delta, NATIVE_DELTA) }?;

    let stream: DatasetRecordBatchStream = block_on(delta_guard.inner.get_deleted_row_ids())?;
    let ffi_stream = to_ffi_arrow_array_stream(stream, RT.handle().clone())?;

    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_delta_DatasetDelta_releaseNativeDelta(
    mut env: JNIEnv,
    obj: JObject,
    handle: jlong,
) {
    ok_or_throw_without_return!(env, inner_release_native_delta(&mut env, obj, handle));
}

fn inner_release_native_delta(env: &mut JNIEnv, obj: JObject, _handle: jlong) -> Result<()> {
    let _: BlockingDatasetDelta = unsafe { env.take_rust_field(obj, NATIVE_DELTA) }?;
    Ok(())
}
