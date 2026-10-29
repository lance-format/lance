// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi};
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray, cast::AsArray, types::UInt64Type};
use arrow_schema::{DataType, Field, Schema};
use jni::{
    JNIEnv,
    objects::{JBooleanArray, JByteArray, JLongArray, JObject, JObjectArray, JString, JValue},
    sys::{jboolean, jbyteArray, jint, jintArray, jlong, jobjectArray},
};
use lance::blob::reference::{
    BLOB_REFERENCE_CAPABILITIES, BLOB_REFERENCE_OPERATIONS, BLOB_REFERENCE_PROTOCOL_VERSION,
    BlobMaterializationContext, BlobReadRange, BlobRefBatch, BlobRefBinding, BlobRefCapture,
    BlobRefReadSession, BlobRefReader, BlobSourceContext, NestedBlobRefBatch, blob_ref_field,
};
use lance_core::utils::address::RowAddress;
use tokio::sync::Mutex;

use crate::{
    block_on,
    error::Result,
    snapshot_retention::{dataset, java_bytes},
};

const NATIVE_HANDLE: &str = "nativeHandle";

struct ReaderState {
    reader: BlobRefReader,
    context: Arc<BlobMaterializationContext>,
}

#[derive(Clone)]
enum NativeHandle {
    Dataset(Arc<lance::Dataset>),
    Capture(Arc<Mutex<BlobRefCapture>>),
    Reader(Arc<ReaderState>),
    Stream(Arc<BlobRefReadSession>),
}

fn invalid(message: impl Into<String>) -> crate::Error {
    lance::Error::invalid_input(message).into()
}

fn handle(env: &mut JNIEnv, object: JObject) -> Result<NativeHandle> {
    // Clone before any blocking operation so cancel/close can access the handle.
    // close() holds this monitor while take_rust_field frees the native box.
    let _monitor = env.lock_obj(&object)?;
    let native = unsafe { env.get_rust_field::<_, _, NativeHandle>(&object, NATIVE_HANDLE) }
        .map_err(|error| {
            invalid(format!(
                "Blob reference resource is closed or unavailable: {error}"
            ))
        })?;
    let handle = native.clone();
    drop(native);
    Ok(handle)
}

fn attach<'a>(env: &mut JNIEnv<'a>, class: &str, handle: NativeHandle) -> Result<JObject<'a>> {
    let object = env.new_object(format!("org/lance/BlobReferences${class}"), "()V", &[])?;
    unsafe { env.set_rust_field(&object, NATIVE_HANDLE, handle) }?;
    Ok(object)
}

fn contexts(env: &mut JNIEnv, values: JObjectArray) -> Result<Vec<BlobSourceContext>> {
    let count = env.get_array_length(&values)?;
    let mut contexts = Vec::with_capacity(count as usize);
    for index in 0..count {
        let value = JByteArray::from(env.get_object_array_element(&values, index)?);
        contexts.push(BlobSourceContext::decode(&env.convert_byte_array(&value)?)?);
        env.delete_local_ref(value)?;
    }
    Ok(contexts)
}

fn unsigned_longs(env: &JNIEnv, values: JLongArray) -> Result<Vec<u64>> {
    let mut output = vec![0_i64; env.get_array_length(&values)? as usize];
    env.get_long_array_region(&values, 0, &mut output)?;
    Ok(output.into_iter().map(|value| value as u64).collect())
}

fn import_batch(array: jlong, schema: jlong) -> Result<RecordBatch> {
    if array == 0 || schema == 0 {
        return Err(invalid("Arrow array/schema addresses must not be null"));
    }
    // Both addresses come from freshly exported Java C Data structs. Move their
    // release callbacks to Rust; the Java wrappers can then close independently.
    let array = unsafe { FFI_ArrowArray::from_raw(array as *mut FFI_ArrowArray) };
    let schema = unsafe { FFI_ArrowSchema::from_raw(schema as *mut FFI_ArrowSchema) };
    let data = unsafe { from_ffi(array, &schema) }?;
    if !matches!(data.data_type(), DataType::Struct(_)) {
        return Err(invalid("Expected an Arrow record batch struct"));
    }
    Ok(RecordBatch::from(StructArray::from(data)))
}

fn export_batch(batch: RecordBatch, array: jlong, schema: jlong) -> Result<()> {
    if array == 0 || schema == 0 {
        return Err(invalid("Arrow array/schema addresses must not be null"));
    }
    let (ffi_array, ffi_schema) = to_ffi(&StructArray::from(batch).to_data())?;
    // Destinations are fresh, empty Java ArrowArray/ArrowSchema allocations.
    unsafe {
        std::ptr::write_unaligned(array as *mut FFI_ArrowArray, ffi_array);
        std::ptr::write_unaligned(schema as *mut FFI_ArrowSchema, ffi_schema);
    }
    Ok(())
}

fn capture_inputs(batch: &RecordBatch, column: &str) -> Result<(Field, ArrayRef, Vec<RowAddress>)> {
    let field = batch.schema().field_with_name(column)?.clone();
    let descriptors = batch
        .column_by_name(column)
        .ok_or_else(|| invalid(format!("Missing descriptor column {column}")))?
        .clone();
    let addresses = batch
        .column_by_name(lance_core::ROW_ADDR)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| invalid("Scan batch must contain physical uint64 _rowaddr"))?;
    if addresses.null_count() != 0 {
        return Err(invalid("_rowaddr must not contain nulls"));
    }
    let addresses = addresses
        .values()
        .iter()
        .copied()
        .map(RowAddress::from)
        .collect();
    Ok((field, descriptors, addresses))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_getProtocolVersion(
    _env: JNIEnv,
    _class: JObject,
) -> jint {
    BLOB_REFERENCE_PROTOCOL_VERSION as jint
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_getSupportedCapabilities(
    mut env: JNIEnv,
    _class: JObject,
) -> jintArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let array = env.new_int_array(BLOB_REFERENCE_CAPABILITIES.len() as jint)?;
            env.set_int_array_region(
                &array,
                0,
                &BLOB_REFERENCE_CAPABILITIES
                    .iter()
                    .map(|&capability| capability as jint)
                    .collect::<Vec<_>>(),
            )?;
            Ok(array.into_raw())
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_getSupportedOperations(
    mut env: JNIEnv,
    _class: JObject,
) -> jobjectArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let result = env.new_object_array(
                BLOB_REFERENCE_OPERATIONS.len() as jint,
                "java/lang/String",
                JObject::null(),
            )?;
            for (index, operation) in BLOB_REFERENCE_OPERATIONS.iter().enumerate() {
                let value = env.new_string(operation)?;
                env.set_object_array_element(&result, index as jint, value)?;
            }
            Ok(result.into_raw())
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_captureContextNative(
    mut env: JNIEnv,
    _class: JObject,
    source: JObject,
    table_identity: JString,
    access_context_key: JString,
    assume_immutable_external: jboolean,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let dataset = dataset(&mut env, source)?;
            let table_identity: String = env.get_string(&table_identity)?.into();
            let access_context_key: String = env.get_string(&access_context_key)?.into();
            let mut context = block_on(BlobSourceContext::from_dataset(
                &dataset,
                table_identity,
                access_context_key,
            ))?;
            if assume_immutable_external != 0 {
                context = context.with_external_immutability_assumption();
            }
            java_bytes(&env, &context.encode())
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_describeContextNative(
    mut env: JNIEnv,
    _class: JObject,
    source: JByteArray,
) -> jobjectArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let context = BlobSourceContext::decode(&env.convert_byte_array(source)?)?;
            let values = [
                Some(context.table_identity().to_owned()),
                Some(context.table_uri().to_owned()),
                context.branch().map(str::to_owned),
                Some(context.version().to_string()),
                Some(context.access_context_key().to_owned()),
            ];
            let result =
                env.new_object_array(values.len() as jint, "java/lang/String", JObject::null())?;
            for (index, value) in values.iter().enumerate() {
                if let Some(value) = value {
                    let value = env.new_string(value)?;
                    env.set_object_array_element(&result, index as jint, value)?;
                }
            }
            Ok(result.into_raw())
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_validateBatchNative(
    mut env: JNIEnv,
    _class: JObject,
    batch: JByteArray,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            java_bytes(
                &env,
                &BlobRefBatch::decode(&env.convert_byte_array(batch)?)?.encode()?,
            )
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_batchContextsNative(
    mut env: JNIEnv,
    _class: JObject,
    batch: JByteArray,
) -> jobjectArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let batch = BlobRefBatch::decode(&env.convert_byte_array(batch)?)?;
            let count = jint::try_from(batch.contexts().len())
                .map_err(|_| invalid("Blob context table exceeds JNI array capacity"))?;
            let result = env.new_object_array(count, "[B", JObject::null())?;
            for (index, context) in batch.contexts().iter().enumerate() {
                let value = env.byte_array_from_slice(&context.encode())?;
                env.set_object_array_element(&result, index as jint, &value)?;
                env.delete_local_ref(value)?;
            }
            Ok(result.into_raw())
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_exportBatchNative(
    mut env: JNIEnv,
    _class: JObject,
    batch: JByteArray,
    array: jlong,
    schema: jlong,
) {
    ok_or_throw_without_return!(
        env,
        (|| -> Result<_> {
            let batch = BlobRefBatch::decode(&env.convert_byte_array(batch)?)?;
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![blob_ref_field("blob_ref")])),
                vec![Arc::new(batch.array().clone())],
            )?;
            export_batch(batch, array, schema)
        })()
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_importBatchNative(
    mut env: JNIEnv,
    _class: JObject,
    sources: JObjectArray,
    array: jlong,
    schema: jlong,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let batch = import_batch(array, schema)?;
            if batch.num_columns() != 1 {
                return Err(invalid(
                    "Blob reference Arrow batch must contain exactly one column",
                ));
            }
            let values = batch
                .column(0)
                .as_struct_opt()
                .ok_or_else(|| invalid("Blob reference Arrow array must be a struct"))?;
            let batch = BlobRefBatch::try_from_arrow(
                contexts(&mut env, sources)?,
                batch.schema().field(0),
                values.clone(),
            )?;
            java_bytes(&env, &batch.encode()?)
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_remapBatchNative(
    mut env: JNIEnv,
    _class: JObject,
    batch: JByteArray,
    sources: JObjectArray,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let batch = BlobRefBatch::decode(&env.convert_byte_array(batch)?)?;
            let batch = batch.remap_contexts(contexts(&mut env, sources)?)?;
            java_bytes(&env, &batch.encode()?)
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_createCaptureNative<'a>(
    mut env: JNIEnv<'a>,
    _class: JObject,
    source: JObject,
    context: JByteArray,
    field_id: jint,
) -> JObject<'a> {
    ok_or_throw!(
        env,
        (|| -> Result<_> {
            let dataset = dataset(&mut env, source)?;
            let context = BlobSourceContext::decode(&env.convert_byte_array(context)?)?;
            let capture = BlobRefCapture::try_new(dataset, context, field_id)?;
            attach(
                &mut env,
                "Capture",
                NativeHandle::Capture(Arc::new(Mutex::new(capture))),
            )
        })()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_captureNative(
    mut env: JNIEnv,
    _class: JObject,
    capture: JObject,
    scan_source: JByteArray,
    column: JString,
    array: jlong,
    schema: jlong,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let NativeHandle::Capture(capture) = handle(&mut env, capture)? else {
                return Err(invalid("Expected a blob capture handle"));
            };
            let source = BlobSourceContext::decode(&env.convert_byte_array(scan_source)?)?;
            let column: String = env.get_string(&column)?.into();
            let batch = import_batch(array, schema)?;
            let (field, descriptors, addresses) = capture_inputs(&batch, &column)?;
            let descriptors = descriptors
                .as_struct_opt()
                .ok_or_else(|| invalid(format!("Missing descriptor struct column {column}")))?;
            let batch = block_on(async {
                capture
                    .lock()
                    .await
                    .capture(&source, &field, &addresses, descriptors)
                    .await
            })?;
            java_bytes(&env, &batch.encode()?)
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_captureNestedNative(
    mut env: JNIEnv,
    _class: JObject,
    capture: JObject,
    scan_source: JByteArray,
    column: JString,
    array: jlong,
    schema: jlong,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let NativeHandle::Capture(capture) = handle(&mut env, capture)? else {
                return Err(invalid("Expected a blob capture handle"));
            };
            let source = BlobSourceContext::decode(&env.convert_byte_array(scan_source)?)?;
            let column: String = env.get_string(&column)?.into();
            let batch = import_batch(array, schema)?;
            let (field, descriptors, addresses) = capture_inputs(&batch, &column)?;
            let nested = block_on(async {
                capture
                    .lock()
                    .await
                    .capture_nested(&source, &field, &addresses, &descriptors)
                    .await
            })?;
            java_bytes(&env, &nested.encode()?)
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_validateNestedBatchNative(
    mut env: JNIEnv,
    _class: JObject,
    encoded: JByteArray,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let batch = NestedBlobRefBatch::decode(&env.convert_byte_array(encoded)?)?;
            java_bytes(&env, &batch.encode()?)
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_importNestedBatchNative(
    mut env: JNIEnv,
    _class: JObject,
    references: JByteArray,
    array: jlong,
    schema: jlong,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let references = BlobRefBatch::decode(&env.convert_byte_array(references)?)?;
            let batch = NestedBlobRefBatch::try_new(references, import_batch(array, schema)?)?;
            java_bytes(&env, &batch.encode()?)
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_nestedReferencesNative(
    mut env: JNIEnv,
    _class: JObject,
    encoded: JByteArray,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let batch = NestedBlobRefBatch::decode(&env.convert_byte_array(encoded)?)?;
            java_bytes(&env, &batch.references().encode()?)
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_exportNestedIndicesNative(
    mut env: JNIEnv,
    _class: JObject,
    encoded: JByteArray,
    array: jlong,
    schema: jlong,
) {
    ok_or_throw_without_return!(
        env,
        (|| -> Result<_> {
            let batch = NestedBlobRefBatch::decode(&env.convert_byte_array(encoded)?)?;
            export_batch(batch.indices().clone(), array, schema)
        })()
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_metricsNative<'a>(
    mut env: JNIEnv<'a>,
    _class: JObject,
    reader: JObject,
) -> JObject<'a> {
    ok_or_throw!(
        env,
        (|| -> Result<_> {
            let NativeHandle::Reader(reader) = handle(&mut env, reader)? else {
                return Err(invalid("Expected a blob reader handle"));
            };
            let values: HashMap<String, u64> =
                serde_json::from_value(serde_json::to_value(reader.reader.metrics()?)?)?;
            let map = env.new_object("java/util/HashMap", "()V", &[])?;
            for (key, value) in values {
                let key = env.new_string(key)?;
                let value =
                    env.new_object("java/lang/Long", "(J)V", &[JValue::Long(value as i64)])?;
                env.call_method(
                    &map,
                    "put",
                    "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                    &[JValue::Object(&key), JValue::Object(&value)],
                )?;
                env.delete_local_ref(key)?;
                env.delete_local_ref(value)?;
            }
            Ok(map)
        })()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_bindDatasetNative<'a>(
    mut env: JNIEnv<'a>,
    _class: JObject,
    source: JObject,
) -> JObject<'a> {
    ok_or_throw!(
        env,
        (|| -> Result<_> {
            let dataset = dataset(&mut env, source)?;
            attach(&mut env, "DatasetBinding", NativeHandle::Dataset(dataset))
        })()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_createReaderNative<'a>(
    mut env: JNIEnv<'a>,
    _class: JObject,
    sources: JObjectArray,
    datasets: JObjectArray,
    io_bytes: jlong,
    materialization_bytes: jlong,
) -> JObject<'a> {
    ok_or_throw!(
        env,
        (|| -> Result<_> {
            let contexts = contexts(&mut env, sources)?;
            if contexts.len() != env.get_array_length(&datasets)? as usize {
                return Err(invalid(
                    "Blob source contexts and dataset bindings must have equal counts",
                ));
            }
            let mut bindings = Vec::with_capacity(contexts.len());
            for (index, context) in contexts.into_iter().enumerate() {
                let object = env.get_object_array_element(&datasets, index as jint)?;
                let NativeHandle::Dataset(dataset) = handle(&mut env, object)? else {
                    return Err(invalid("Expected a dataset binding handle"));
                };
                bindings.push(BlobRefBinding::try_new(context, dataset)?);
            }
            let context =
                BlobMaterializationContext::try_new(io_bytes as u64, materialization_bytes as u64)?;
            let reader = BlobRefReader::try_new(bindings, context.clone())?;
            attach(
                &mut env,
                "Reader",
                NativeHandle::Reader(Arc::new(ReaderState { reader, context })),
            )
        })()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_readNative<'a>(
    mut env: JNIEnv<'a>,
    _class: JObject,
    reader: JObject,
    batch: JByteArray,
    reference_indices: JLongArray,
    request_indices: JLongArray,
    has_ranges: JBooleanArray,
    offsets: JLongArray,
    lengths: JLongArray,
) -> JObject<'a> {
    ok_or_throw!(
        env,
        (|| -> Result<_> {
            let NativeHandle::Reader(reader) = handle(&mut env, reader)? else {
                return Err(invalid("Expected a blob reader handle"));
            };
            let batch = BlobRefBatch::decode(&env.convert_byte_array(batch)?)?;
            let reference_indices = unsigned_longs(&env, reference_indices)?;
            let request_indices = unsigned_longs(&env, request_indices)?;
            let offsets = unsigned_longs(&env, offsets)?;
            let lengths = unsigned_longs(&env, lengths)?;
            let mut has = vec![0; env.get_array_length(&has_ranges)? as usize];
            env.get_boolean_array_region(&has_ranges, 0, &mut has)?;
            if has.len() != offsets.len() || has.len() != lengths.len() {
                return Err(invalid(
                    "Blob range flags, offsets, and lengths must have equal counts",
                ));
            }
            let ranges = has
                .iter()
                .zip(offsets)
                .zip(lengths)
                .map(|((&has, offset), length)| {
                    (has != 0).then_some(BlobReadRange::new(offset, length))
                })
                .collect::<Vec<_>>();
            let requests = batch.read_requests(&reference_indices, &request_indices, &ranges)?;
            let stream = BlobRefReadSession::try_new(&reader.reader, requests)?;
            attach(
                &mut env,
                "ReadStream",
                NativeHandle::Stream(Arc::new(stream)),
            )
        })()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_nextNative(
    mut env: JNIEnv,
    _class: JObject,
    stream: JObject,
    array: jlong,
    schema: jlong,
) -> jboolean {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let NativeHandle::Stream(stream) = handle(&mut env, stream)? else {
                return Err(invalid("Expected a blob read stream handle"));
            };
            match block_on(stream.next_batch()).transpose()? {
                Some(batch) => {
                    export_batch(batch, array, schema)?;
                    Ok(1)
                }
                None => Ok(0),
            }
        })(),
        0
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_cancelNative(
    mut env: JNIEnv,
    _class: JObject,
    stream: JObject,
) {
    ok_or_throw_without_return!(
        env,
        (|| -> Result<_> {
            let NativeHandle::Stream(stream) = handle(&mut env, stream)? else {
                return Err(invalid("Expected a blob read stream handle"));
            };
            stream.cancel();
            Ok(())
        })()
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_closeNative(
    mut env: JNIEnv,
    _class: JObject,
    resource: JObject,
) {
    ok_or_throw_without_return!(
        env,
        (|| -> Result<_> {
            let handle =
                unsafe { env.take_rust_field::<_, _, NativeHandle>(resource, NATIVE_HANDLE) }?;
            if let NativeHandle::Stream(stream) = handle {
                block_on(stream.close());
            }
            Ok(())
        })()
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_reservedBytesNative(
    mut env: JNIEnv,
    _class: JObject,
    reader: JObject,
    io: jboolean,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let NativeHandle::Reader(reader) = handle(&mut env, reader)? else {
                return Err(invalid("Expected a blob reader handle"));
            };
            let reserved = if io != 0 {
                reader.context.io_reserved_bytes()
            } else {
                reader.context.reserved_bytes()
            };
            Ok(reserved? as jlong)
        })(),
        0
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_BlobReferences_snapshotIdentityNative(
    mut env: JNIEnv,
    _class: JObject,
    context: JByteArray,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let context = BlobSourceContext::decode(&env.convert_byte_array(context)?)?;
            java_bytes(&env, &context.snapshot_identity()?.encode())
        })(),
        std::ptr::null_mut()
    )
}
