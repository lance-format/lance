// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{BlockingDataset, NATIVE_DATASET};
use crate::block_on;
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::Schema as ArrowSchema;
use jni::JNIEnv;
use jni::objects::{JObject, JString};
use jni::sys::jlong;
use lance::dataset::{ColumnAlteration, NewColumnTransform};
use std::sync::Arc;

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
