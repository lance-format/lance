// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::utils::to_rust_map;
use crate::{
    JNIEnvExt, RT, block_on,
    error::{Error, Result},
    traits::IntoJava,
};
use arrow::{array::RecordBatchReader, ffi::FFI_ArrowSchema, ffi_stream::FFI_ArrowArrayStream};
use arrow_schema::SchemaRef;
use jni::objects::JMap;
use jni::{
    JNIEnv,
    objects::{JObject, JString},
    sys::{jint, jlong},
};
use lance::io::ObjectStore;
use lance_core::cache::LanceCache;
use lance_core::datatypes::{BlobHandling, OnMissing, Projection, Schema};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions, ReaderProjection};
use lance_file::versions as file_versions;
use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry};
use lance_io::{
    ReadBatchParams,
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use object_store::path::Path;

pub const NATIVE_READER: &str = "nativeFileReaderHandle";

#[derive(Clone, Debug)]
pub struct BlockingFileReader {
    pub(crate) inner: Arc<FileReader>,
}

impl BlockingFileReader {
    pub fn create(file_reader: Arc<FileReader>) -> Self {
        Self { inner: file_reader }
    }

    pub fn open_stream(
        &self,
        batch_size: u32,
        read_batch_params: ReadBatchParams,
        reader_projection: Option<ReaderProjection>,
        filter_expression: FilterExpression,
    ) -> Result<Box<dyn RecordBatchReader + Send + 'static>> {
        let reader = self.inner.clone();
        Ok(block_on(RT.spawn_blocking(move || {
            reader.read_stream_projected_blocking(
                read_batch_params,
                batch_size,
                reader_projection,
                filter_expression,
            )
        }))
        .unwrap()?)
    }

    pub fn schema(&self) -> Result<SchemaRef> {
        Ok(Arc::new(self.inner.schema().as_ref().into()))
    }

    pub fn num_rows(&self) -> u64 {
        self.inner.num_rows()
    }
}

impl IntoJava for BlockingFileReader {
    fn into_java<'local>(self, env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
        attach_native_reader(env, self)
    }
}

fn attach_native_reader<'local>(
    env: &mut JNIEnv<'local>,
    reader: BlockingFileReader,
) -> Result<JObject<'local>> {
    let j_reader = create_java_reader_object(env)?;
    unsafe { env.set_rust_field(&j_reader, NATIVE_READER, reader) }?;
    Ok(j_reader)
}

fn create_java_reader_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let res = env.new_object("org/lance/file/LanceFileReader", "()V", &[])?;
    Ok(res)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileReader_openNative<'local>(
    mut env: JNIEnv<'local>,
    _reader_class: JObject,
    file_uri: JString,
    storage_options_obj: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_open(&mut env, file_uri, storage_options_obj))
}

fn inner_open<'local>(
    env: &mut JNIEnv<'local>,
    file_uri: JString,
    storage_options_obj: JObject,
) -> Result<JObject<'local>> {
    let file_uri_str: String = env.get_string(&file_uri)?.into();
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options = to_rust_map(env, &jmap)?;
    let reader = block_on(async move {
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
        let config = SchedulerConfig::max_bandwidth(&obj_store);
        let scan_scheduler = ScanScheduler::new(obj_store, config);

        let file_scheduler = scan_scheduler
            .open_file(&Path::parse(&path)?, &CachedFileSize::unknown())
            .await?;
        FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
    })?;

    let reader = BlockingFileReader::create(Arc::new(reader));

    reader.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileReader_closeNative<'local>(
    mut env: JNIEnv<'local>,
    reader: JObject,
) -> JObject<'local> {
    let maybe_err =
        unsafe { env.take_rust_field::<_, _, BlockingFileReader>(reader, NATIVE_READER) };
    match maybe_err {
        Ok(_) => {}
        // We were already closed, do nothing
        Err(jni::errors::Error::NullPtr(_)) => {}
        Err(err) => Error::from(err).throw(&mut env),
    }
    JObject::null()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileReader_numRowsNative(
    mut env: JNIEnv<'_>,
    reader: JObject,
) -> jlong {
    match inner_num_rows(&mut env, reader) {
        Ok(num_rows) => num_rows,
        Err(e) => {
            e.throw(&mut env);
            0
        }
    }
}

// If the reader is closed, the native handle will be null and we will get a JniError::NullPtr
// error when we call get_rust_field.  Translate that into a more meaningful error.
fn unwrap_reader<T>(val: std::result::Result<T, jni::errors::Error>) -> Result<T> {
    match val {
        Ok(val) => Ok(val),
        Err(jni::errors::Error::NullPtr(_)) => Err(Error::io_error(
            "FileReader has already been closed".to_string(),
        )),
        err => Ok(err?),
    }
}

fn inner_num_rows(env: &mut JNIEnv<'_>, reader: JObject) -> Result<jlong> {
    let reader = unsafe { env.get_rust_field::<_, _, BlockingFileReader>(reader, NATIVE_READER) };
    let reader = unwrap_reader(reader)?;
    Ok(reader.num_rows() as i64)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileReader_populateSchemaNative(
    mut env: JNIEnv,
    reader: JObject,
    schema_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_populate_schema(&mut env, reader, schema_addr));
}

fn inner_populate_schema(env: &mut JNIEnv, reader: JObject, schema_addr: jlong) -> Result<()> {
    let reader = unsafe { env.get_rust_field::<_, _, BlockingFileReader>(reader, NATIVE_READER) };
    let reader = unwrap_reader(reader)?;
    let schema = reader.schema()?;
    let ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())?;
    unsafe { std::ptr::write_unaligned(schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_file_LanceFileReader_readAllNative(
    mut env: JNIEnv<'_>,
    reader: JObject,
    batch_size: jint,
    projected_names: JObject,
    selection_ranges: JObject,
    stream_addr: jlong,
    blob_read_mode: jint,
) {
    let result = (|| -> Result<()> {
        let mut read_parameter = ReadBatchParams::default();
        // We get reader here not from env.get_rust_field, because we need reader: MutexGuard<BlockingFileReader> has no relationship with the env lifecycle.
        // If we get reader from env.get_rust_field, we can't use env (can't borrow again) until we drop the reader.
        #[allow(unused_variables)]
        let reader = unsafe {
            let reader_ref = reader.as_ref();
            let ptr = env.get_field(reader_ref, NATIVE_READER, "J")?.j()?
                as *mut Mutex<BlockingFileReader>;
            let guard = env.lock_obj(reader_ref)?;
            if ptr.is_null() {
                return Err(Error::io_error(
                    "FileReader has already been closed".to_string(),
                ));
            }
            (*ptr).lock().unwrap()
        };

        let file_version = reader.inner.metadata().version();
        let base_schema = Schema::try_from(reader.schema()?.as_ref())?;

        let blob_handling = if blob_read_mode == 1 {
            BlobHandling::BlobsDescriptions
        } else {
            BlobHandling::AllBinary
        };

        let reader_projection = {
            let mut projection =
                Projection::empty(Arc::new(base_schema.clone())).with_blob_handling(blob_handling);

            if !projected_names.is_null() {
                let column_names: Vec<String> = env.get_strings(&projected_names)?;
                projection = projection.union_columns(&column_names, OnMissing::Error)?;
            } else {
                projection = projection.union_predicate(|_| true);
            }

            let transformed_schema = projection.to_bare_schema();

            let (field_ids, column_indices) =
                file_versions::data_file_columns(file_version, &base_schema);
            let field_id_to_column_index = field_ids
                .into_iter()
                .zip(column_indices)
                .filter_map(|(field_id, column_index)| {
                    (column_index >= 0).then_some((field_id as u32, column_index as u32))
                })
                .collect::<BTreeMap<_, _>>();

            Some(file_versions::reader_projection_from_field_ids(
                file_version,
                &transformed_schema,
                &field_id_to_column_index,
            )?)
        };

        if !selection_ranges.is_null() {
            let mut ranges: Vec<Range<u64>> = Vec::new();
            let jlist = env.get_list(&selection_ranges)?;
            let mut j_list_iter = jlist.iter(&mut env)?;

            loop {
                let item = j_list_iter.next(&mut env)?;
                if item.is_none() {
                    break; // End of the list
                }

                let item_obj = item.unwrap();
                if item_obj.is_null() {
                    continue;
                }

                let start_val = env
                    .call_method(&item_obj, "getStart", "()I", &[])
                    .and_then(|v| v.i())?;
                let end_val = env
                    .call_method(&item_obj, "getEnd", "()I", &[])
                    .and_then(|v| v.i())?;

                if start_val < 0 || end_val < 0 {
                    return Err(Error::input_error(format!(
                        "Invalid range values (negative): start={}, end={}",
                        start_val, end_val
                    )));
                }
                if start_val > end_val {
                    return Err(Error::input_error(format!(
                        "Invalid range (start > end): start={}, end={}",
                        start_val, end_val
                    )));
                }

                ranges.push(Range {
                    start: start_val as u64,
                    end: end_val as u64,
                });

                env.delete_local_ref(item_obj)?;
            }
            read_parameter = ReadBatchParams::Ranges(ranges.into_boxed_slice().into());
        }
        inner_read_all(
            &reader,
            batch_size,
            read_parameter,
            reader_projection,
            FilterExpression::no_filter(),
            stream_addr,
        )
    })();
    if let Err(e) = result {
        e.throw(&mut env);
    }
}

fn inner_read_all(
    reader: &BlockingFileReader,
    batch_size: jint,
    read_batch_params: ReadBatchParams,
    reader_projection: Option<ReaderProjection>,
    filter_expression: FilterExpression,
    stream_addr: jlong,
) -> Result<()> {
    let arrow_stream = reader.open_stream(
        batch_size as u32,
        read_batch_params,
        reader_projection,
        filter_expression,
    )?;
    let ffi_stream = FFI_ArrowArrayStream::new(arrow_stream);
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}
