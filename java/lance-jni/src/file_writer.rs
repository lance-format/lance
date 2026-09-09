// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex};

use crate::utils::to_rust_map;
use crate::{
    JNIEnvExt, block_on,
    error::{Error, Result},
    traits::IntoJava,
};
use arrow::{
    array::{RecordBatch, StructArray},
    ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi_and_data_type},
};
use arrow_schema::DataType;
use jni::objects::JMap;
use jni::{
    JNIEnv,
    objects::{JObject, JString},
    sys::jlong,
};
use lance::io::ObjectStore;
use lance_file::{
    version::LanceFileVersion,
    versions as file_versions,
    writer::{FileWriter, FileWriterOptions},
};
use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry};

pub const NATIVE_WRITER: &str = "nativeFileWriterHandle";

#[derive(Clone)]
pub struct BlockingFileWriter {
    pub(crate) inner: Arc<Mutex<FileWriter>>,
}

impl BlockingFileWriter {
    pub fn create(file_writer: FileWriter) -> Self {
        Self {
            inner: Arc::new(Mutex::new(file_writer)),
        }
    }
}

impl IntoJava for BlockingFileWriter {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        attach_native_writer(env, self)
    }
}

fn attach_native_writer<'local>(
    env: &mut JNIEnv<'local>,
    writer: BlockingFileWriter,
) -> Result<JObject<'local>> {
    let j_writer = create_java_writer_object(env)?;
    unsafe { env.set_rust_field(&j_writer, NATIVE_WRITER, writer) }?;
    Ok(j_writer)
}

fn create_java_writer_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let res = env.new_object("org/lance/file/LanceFileWriter", "()V", &[])?;
    Ok(res)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileWriter_openNative<'local>(
    mut env: JNIEnv<'local>,
    _writer_class: JObject,
    file_uri: JString,
    data_storage_version: JObject, // Optional<String>
    storage_options_obj: JObject,  // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_open(
            &mut env,
            file_uri,
            data_storage_version,
            storage_options_obj
        )
    )
}

fn inner_open<'local>(
    env: &mut JNIEnv<'local>,
    file_uri: JString,
    data_storage_version: JObject,
    storage_options_obj: JObject,
) -> Result<JObject<'local>> {
    let file_uri_str: String = env.get_string(&file_uri)?.into();
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let data_storage_version_opt = env.get_string_opt(&data_storage_version)?;
    let storage_options = to_rust_map(env, &jmap)?;

    let writer = block_on(async move {
        let object_params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_static_options(storage_options),
            )),
            ..Default::default()
        };
        let (obj_store, path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            &file_uri_str,
            &object_params,
        )
        .await?;
        let obj_store = Arc::new(obj_store);
        let obj_writer = obj_store.create(&path).await?;

        let version = data_storage_version_opt
            .map(|value| value.parse::<LanceFileVersion>())
            .transpose()?
            .unwrap_or_default()
            .resolve();
        file_versions::create_lazy_writer(version, obj_writer, FileWriterOptions::default())
    })?;

    let writer = BlockingFileWriter::create(writer);

    writer.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileWriter_closeNative<'local>(
    mut env: JNIEnv<'local>,
    writer: JObject,
) -> JObject<'local> {
    let maybe_err =
        unsafe { env.take_rust_field::<_, _, BlockingFileWriter>(writer, NATIVE_WRITER) };
    let writer = match maybe_err {
        Ok(writer) => Some(writer),
        // We were already closed, do nothing
        Err(jni::errors::Error::NullPtr(_)) => None,
        Err(err) => {
            Error::from(err).throw(&mut env);
            None
        }
    };
    if let Some(writer) = writer {
        match block_on(writer.inner.lock().unwrap().finish()) {
            Ok(_) => {}
            Err(e) => {
                Error::from(e).throw(&mut env);
            }
        }
    }
    JObject::null()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileWriter_nativeAddSchemaMetadata<'local>(
    mut env: JNIEnv<'local>,
    writer: JObject,
    schema_metadata: JObject, // Map<String, String>
) -> JObject<'local> {
    if let Err(e) = inner_add_schema_metadata(&mut env, writer, schema_metadata) {
        e.throw(&mut env);
        return JObject::null();
    }
    JObject::null()
}

fn inner_add_schema_metadata(
    env: &mut JNIEnv<'_>,
    writer: JObject,
    schema_metadata: JObject, // Map<String, String>
) -> Result<()> {
    let metadata_map = JMap::from_env(env, &schema_metadata)?;
    let metadata = to_rust_map(env, &metadata_map)?;
    let writer_guard =
        unsafe { env.get_rust_field::<_, _, BlockingFileWriter>(writer, NATIVE_WRITER) }?;
    let mut writer = writer_guard.inner.lock().unwrap();
    metadata.into_iter().for_each(|(k, v)| {
        writer.add_schema_metadata(k, v);
    });
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileWriter_writeNative<'local>(
    mut env: JNIEnv<'local>,
    writer: JObject,
    batch_address: jlong,
    schema_address: jlong,
) -> JObject<'local> {
    if let Err(e) = inner_write_batch(&mut env, writer, batch_address, schema_address) {
        e.throw(&mut env);
        return JObject::null();
    }
    JObject::null()
}

fn inner_write_batch(
    env: &mut JNIEnv<'_>,
    writer: JObject,
    batch_address: jlong,
    schema_address: jlong,
) -> Result<()> {
    let c_array_ptr = batch_address as *mut FFI_ArrowArray;
    let c_schema_ptr = schema_address as *mut FFI_ArrowSchema;

    let c_array = unsafe { FFI_ArrowArray::from_raw(c_array_ptr) };
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };

    let data_type = DataType::try_from(&c_schema)?;
    let array_data = unsafe { from_ffi_and_data_type(c_array, data_type) }?;
    let record_batch = RecordBatch::from(StructArray::from(array_data));

    let writer = unsafe { env.get_rust_field::<_, _, BlockingFileWriter>(writer, NATIVE_WRITER) }?;

    let mut writer = writer.inner.lock().unwrap();
    block_on(writer.write_batch(&record_batch))?;
    Ok(())
}
