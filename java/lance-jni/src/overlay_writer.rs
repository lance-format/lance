// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::array::{RecordBatch, StructArray};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi_and_data_type};
use arrow_schema::DataType;
use jni::{
    JNIEnv,
    objects::{JClass, JObject, JValue},
    sys::{jint, jlong},
};
use lance::dataset::OverlayWriter;
use lance::datatypes::Schema;
use lance_core::is_system_column;

use crate::blocking_dataset::clone_dataset;
use crate::traits::IntoJava;
use crate::{Error, JNIEnvExt, Result, block_on};

const NATIVE_WRITER: &str = "nativeOverlayWriterHandle";

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Fragment_nativeWriteOverlay<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,     // Fragment.class
    jdataset: JObject,  // Dataset
    fragment_id: jint,  // Fragment ID, interpreted as u32
    columns: JObject,   // List<String>
    allocator: JObject, // BufferAllocator
) -> JObject<'local> {
    ok_or_throw!(
        env,
        open_overlay_writer(&mut env, &jdataset, fragment_id, &columns, &allocator)
    )
}

fn open_overlay_writer<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: &JObject,
    fragment_id: jint,
    columns: &JObject,
    allocator: &JObject,
) -> Result<JObject<'local>> {
    let columns = env.get_strings(columns)?;
    if columns.is_empty() {
        return Err(Error::input_error(
            "writeOverlay requires at least one column to overlay".to_string(),
        ));
    }
    if let Some(column) = columns.iter().find(|column| is_system_column(column)) {
        return Err(Error::input_error(format!(
            "cannot overlay system column '{column}'"
        )));
    }

    let dataset = clone_dataset(env, jdataset)?;
    let fragment_id = fragment_id as u32;
    let fragment = dataset
        .inner
        .get_fragment(fragment_id as usize)
        .ok_or_else(|| Error::input_error(format!("Fragment not found: {fragment_id}")))?;
    let dataset_schema = fragment.dataset().schema();
    let fields = columns
        .iter()
        .map(|column| {
            dataset_schema
                .fields
                .iter()
                .find(|field| field.name == *column)
                .cloned()
                .ok_or_else(|| {
                    Error::input_error(format!("Unknown top-level overlay column '{column}'"))
                })
        })
        .collect::<Result<_>>()?;
    let schema = Schema {
        fields,
        metadata: dataset_schema.metadata.clone(),
    };
    let jwriter = env.new_object(
        "org/lance/fragment/OverlayWriter",
        "(Lorg/apache/arrow/memory/BufferAllocator;)V",
        &[JValue::Object(allocator)],
    )?;
    let writer = block_on(fragment.write_overlay(&schema)).map_err(lance::Error::from)?;
    unsafe { env.set_rust_field(&jwriter, NATIVE_WRITER, writer) }?;
    Ok(jwriter)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_fragment_OverlayWriter_nativeWriteBatch(
    mut env: JNIEnv,
    jwriter: JObject,      // OverlayWriter
    batch_address: jlong,  // memoryAddress of ArrowArray
    schema_address: jlong, // memoryAddress of ArrowSchema
) {
    ok_or_throw_without_return!(
        env,
        write_batch(&mut env, &jwriter, batch_address, schema_address)
    );
}

fn write_batch(
    env: &mut JNIEnv,
    jwriter: &JObject,
    batch_address: jlong,
    schema_address: jlong,
) -> Result<()> {
    let array = unsafe { FFI_ArrowArray::from_raw(batch_address as *mut FFI_ArrowArray) };
    let schema = unsafe { FFI_ArrowSchema::from_raw(schema_address as *mut FFI_ArrowSchema) };
    let data_type = DataType::try_from(&schema)?;
    let array_data = unsafe { from_ffi_and_data_type(array, data_type) }?;
    let batch = RecordBatch::from(StructArray::from(array_data));
    let mut writer = unsafe { env.get_rust_field::<_, _, OverlayWriter>(jwriter, NATIVE_WRITER) }?;
    block_on(writer.write_batch(&batch)).map_err(lance::Error::from)?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_fragment_OverlayWriter_nativeFinish<'local>(
    mut env: JNIEnv<'local>,
    jwriter: JObject, // OverlayWriter
) -> JObject<'local> {
    let writer: OverlayWriter = ok_or_throw!(
        env,
        unsafe { env.take_rust_field(jwriter, NATIVE_WRITER) }.map_err(Error::from)
    );
    let group = ok_or_throw!(
        env,
        block_on(writer.finish())
            .map_err(lance::Error::from)
            .map_err(Error::from)
    );
    match group {
        Some(ref group) => ok_or_throw!(env, group.into_java(&mut env)),
        None => JObject::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_fragment_OverlayWriter_nativeAbort(
    mut env: JNIEnv,
    jwriter: JObject, // OverlayWriter
) {
    let writer: OverlayWriter = ok_or_throw_without_return!(
        env,
        unsafe { env.take_rust_field(jwriter, NATIVE_WRITER) }.map_err(Error::from)
    );
    block_on(writer.abort());
}
