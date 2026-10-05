// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::FromJString;
use crate::{Error, JNIEnvExt, RT, block_on};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString};
use jni::sys::{JNI_TRUE, jboolean, jlong};
use lance::dataset::scanner::DatasetRecordBatchStream;
use lance::dataset::sql::SqlQueryBuilder;
use lance_io::ffi::to_ffi_arrow_array_stream;

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_SqlQuery_intoBatchRecords(
    mut env: JNIEnv,
    _class: JClass,
    java_dataset: JObject,
    sql: JString,
    table_name: JObject,
    with_row_id: jboolean,
    with_row_addr: jboolean,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_into_batch_records(
            &mut env,
            java_dataset,
            sql,
            table_name,
            with_row_id,
            with_row_addr,
            stream_addr,
        )
        .map_err(|e| Error::input_error(e.to_string()))
    )
}

fn inner_into_batch_records(
    env: &mut JNIEnv,
    java_dataset: JObject,
    sql: JString,
    table_name: JObject,
    with_row_id: jboolean,
    with_row_addr: jboolean,
    stream_addr: jlong,
) -> Result<()> {
    let builder = sql_builder(
        env,
        java_dataset,
        sql,
        table_name,
        with_row_id,
        with_row_addr,
    )?;

    let stream = block_on(async move {
        let query = builder.build().await?;
        query.into_stream().await
    })?;

    let ffi_stream =
        to_ffi_arrow_array_stream(DatasetRecordBatchStream::new(stream), RT.handle().clone())?;

    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

fn sql_builder(
    env: &mut JNIEnv,
    java_dataset: JObject,
    sql: JString,
    table_name: JObject,
    with_row_id: jboolean,
    with_row_addr: jboolean,
) -> Result<SqlQueryBuilder> {
    let sql_str = sql.extract(env)?;
    let table_str = env.get_string_opt(&table_name)?;

    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;

    let mut builder = dataset_guard
        .inner
        .sql(sql_str.as_str())
        .with_row_id(with_row_id == JNI_TRUE)
        .with_row_addr(with_row_addr == JNI_TRUE);

    if let Some(table) = table_str {
        builder = builder.table_name(table.as_str())
    }

    Ok(builder)
}
