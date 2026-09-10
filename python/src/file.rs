// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::blob::{PyDedicatedBlobWriter, PyPackedBlobWriter};
use crate::namespace::extract_namespace_arc;
use crate::{error::PythonErrorExt, rt};
use arrow::pyarrow::PyArrowType;
use arrow_array::{RecordBatch, RecordBatchReader, UInt32Array};
use arrow_schema::Schema as ArrowSchema;
use bytes::Bytes;
use futures::stream::StreamExt;
use lance::io::{ObjectStore, RecordBatchStream};
use lance_core::cache::LanceCache;
use lance_core::utils::path::LancePathExt;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{
    BufferDescriptor, CachedFileMetadata, FileReader, FileReaderOptions, FileStatistics,
};
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_file::{LanceEncodingsIo, version::LanceFileVersion, versions as file_versions};
use lance_io::object_store::{LanceNamespaceStorageOptionsProvider, ObjectStoreParams};
use lance_io::{
    ReadBatchParams,
    scheduler::{ScanScheduler, SchedulerConfig},
    traits::Writer,
    utils::CachedFileSize,
};
use object_store::{ObjectStoreExt, path::Path};
use pyo3::{
    Bound, IntoPyObjectExt, Py, PyErr, PyResult, Python,
    exceptions::{PyIOError, PyRuntimeError},
    pyclass, pyfunction, pymethods,
    types::PyAny,
};
use serde::Serialize;
use std::collections::HashMap;
use std::{pin::Pin, sync::Arc};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceBufferDescriptor {
    /// The byte offset of the buffer in the file
    pub position: u64,
    /// The size (in bytes) of the buffer
    pub size: u64,
}

impl LanceBufferDescriptor {
    fn new(inner: &BufferDescriptor) -> Self {
        Self {
            position: inner.position,
            size: inner.size,
        }
    }

    fn new_from_parts(position: u64, size: u64) -> Self {
        Self { position, size }
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug, Serialize)]
pub struct LancePageMetadata {
    /// The buffers in the page
    pub buffers: Vec<LanceBufferDescriptor>,
    /// A description of the encoding used to encode the page
    pub encoding: String,
}

impl LancePageMetadata {
    fn new(inner: &lance_file::format::pbfile::column_metadata::Page) -> Self {
        let buffers = inner
            .buffer_offsets
            .iter()
            .zip(inner.buffer_sizes.iter())
            .map(|(pos, size)| LanceBufferDescriptor::new_from_parts(*pos, *size))
            .collect();
        Self {
            buffers,
            encoding: lance_file::reader::describe_encoding(inner),
        }
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceColumnMetadata {
    /// The column-wide buffers
    pub column_buffers: Vec<LanceBufferDescriptor>,
    /// The data pages in the column
    pub pages: Vec<LancePageMetadata>,
}

impl LanceColumnMetadata {
    fn new(inner: &lance_file::format::pbfile::ColumnMetadata) -> Self {
        let column_buffers = inner
            .buffer_offsets
            .iter()
            .zip(inner.buffer_sizes.iter())
            .map(|(pos, size)| LanceBufferDescriptor::new_from_parts(*pos, *size))
            .collect();
        Self {
            column_buffers,
            pages: inner.pages.iter().map(LancePageMetadata::new).collect(),
        }
    }
}

/// Statistics summarize some of the file metadata for quick summary info
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceFileStatistics {
    /// Statistics about each of the columns in the file
    columns: Vec<LanceColumnStatistics>,
}

#[pymethods]
impl LanceFileStatistics {
    fn __repr__(&self) -> String {
        let column_reprs: Vec<String> = self.columns.iter().map(|col| col.__repr__()).collect();
        format!("FileStatistics(columns=[{}])", column_reprs.join(", "))
    }
}

/// Summary information describing a column
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceColumnStatistics {
    /// The number of pages in the column
    num_pages: usize,
    /// The total number of data & metadata bytes in the column
    ///
    /// This is the compressed on-disk size
    size_bytes: u64,
}

#[pymethods]
impl LanceColumnStatistics {
    fn __repr__(&self) -> String {
        format!(
            "ColumnStatistics(num_pages={}, size_bytes={})",
            self.num_pages, self.size_bytes
        )
    }
}

impl LanceFileStatistics {
    fn new(inner: &FileStatistics) -> Self {
        let columns = inner
            .columns
            .iter()
            .map(|column_stat| LanceColumnStatistics {
                num_pages: column_stat.num_pages,
                size_bytes: column_stat.size_bytes,
            })
            .collect();
        Self { columns }
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceFileMetadata {
    /// The schema of the file
    #[serde(skip)]
    pub schema: Option<Py<PyAny>>,
    /// The major version of the file
    pub major_version: u16,
    /// The minor version of the file
    pub minor_version: u16,
    /// The number of rows in the file
    pub num_rows: u64,
    /// The number of bytes in the data section of the file
    pub num_data_bytes: u64,
    /// The number of bytes in the column metadata section of the file
    pub num_column_metadata_bytes: u64,
    /// The number of bytes in the global buffer section of the file
    pub num_global_buffer_bytes: u64,
    /// The global buffers
    pub global_buffers: Vec<LanceBufferDescriptor>,
    /// The column metadata, an entry might be None if the metadata for a column
    /// was not loaded into memory when the file was opened.
    pub columns: Vec<Option<LanceColumnMetadata>>,
}

impl LanceFileMetadata {
    fn new(inner: &CachedFileMetadata, py: Python) -> Self {
        let arrow_schema = arrow_schema::Schema::from(inner.file_schema.as_ref());
        let schema = PyArrowType(arrow_schema).into_py_any(py).ok();
        Self {
            major_version: inner.major_version,
            minor_version: inner.minor_version,
            schema,
            num_rows: inner.num_rows,
            num_data_bytes: inner.num_data_bytes,
            num_column_metadata_bytes: inner.num_column_metadata_bytes,
            num_global_buffer_bytes: inner.num_global_buffer_bytes,
            global_buffers: inner
                .file_buffers
                .iter()
                .map(LanceBufferDescriptor::new)
                .collect(),
            columns: inner
                .column_metadatas
                .iter()
                .map(LanceColumnMetadata::new)
                .map(Some)
                .collect(),
        }
    }
}

#[pymethods]
impl LanceFileMetadata {
    pub fn __repr__(&self) -> PyResult<String> {
        serde_yaml::to_string(self).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

/// Summary of a completed Lance file write
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceFileWriteSummary {
    /// The number of rows written to the file
    pub num_rows: u64,
    /// The final size of the file, in bytes
    pub size_bytes: u64,
}

#[pymethods]
impl LanceFileWriteSummary {
    fn __repr__(&self) -> String {
        format!(
            "FileWriteSummary(num_rows={}, size_bytes={})",
            self.num_rows, self.size_bytes
        )
    }
}

#[pyclass]
pub struct LanceFileWriter {
    inner: Arc<Mutex<Box<FileWriter>>>,
}

impl LanceFileWriter {
    #[allow(clippy::too_many_arguments)]
    async fn open(
        uri_or_path: String,
        schema: Option<PyArrowType<ArrowSchema>>,
        data_cache_bytes: Option<u64>,
        version: Option<String>,
        storage_options: Option<HashMap<String, String>>,
        storage_options_provider: Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>>,
        keep_original_array: Option<bool>,
        max_page_bytes: Option<u64>,
    ) -> PyResult<Self> {
        let (object_store, path) = object_store_from_uri_or_path_with_provider(
            uri_or_path,
            storage_options,
            storage_options_provider,
        )
        .await?;
        Self::open_with_store(
            object_store,
            path,
            schema,
            data_cache_bytes,
            version,
            keep_original_array,
            max_page_bytes,
        )
        .await
    }

    async fn open_with_store(
        object_store: Arc<ObjectStore>,
        path: Path,
        schema: Option<PyArrowType<ArrowSchema>>,
        data_cache_bytes: Option<u64>,
        version: Option<String>,
        keep_original_array: Option<bool>,
        max_page_bytes: Option<u64>,
    ) -> PyResult<Self> {
        let object_writer = object_store.create(&path).await.infer_error()?;
        let version = version
            .map(|value| value.parse::<LanceFileVersion>())
            .transpose()
            .infer_error()?
            .unwrap_or_default()
            .resolve();
        let options = FileWriterOptions {
            data_cache_bytes,
            keep_original_array,
            max_page_bytes,
        };
        let inner = if let Some(schema) = schema {
            let lance_schema = lance_core::datatypes::Schema::try_from(&schema.0).infer_error()?;
            file_versions::create_writer(version, object_writer, lance_schema, options)
                .infer_error()
        } else {
            file_versions::create_lazy_writer(version, object_writer, options).infer_error()
        }?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Box::new(inner))),
        })
    }
}

#[pymethods]
impl LanceFileWriter {
    #[new]
    #[pyo3(signature=(path, schema=None, data_cache_bytes=None, version=None, storage_options=None, namespace_client=None, table_id=None, keep_original_array=None, max_page_bytes=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: String,
        schema: Option<PyArrowType<ArrowSchema>>,
        data_cache_bytes: Option<u64>,
        version: Option<String>,
        storage_options: Option<HashMap<String, String>>,
        namespace_client: Option<&Bound<'_, PyAny>>,
        table_id: Option<Vec<String>>,
        keep_original_array: Option<bool>,
        max_page_bytes: Option<u64>,
    ) -> PyResult<Self> {
        // Create storage options provider from namespace_client and table_id if both are provided
        let provider = if let (Some(ns_client), Some(tid)) = (&namespace_client, &table_id) {
            let ns_client = extract_namespace_arc(ns_client.py(), ns_client)?;
            Some(Arc::new(LanceNamespaceStorageOptionsProvider::new(
                ns_client,
                tid.clone(),
            ))
                as Arc<dyn lance_io::object_store::StorageOptionsProvider>)
        } else {
            None
        };

        rt().block_on(
            None,
            Self::open(
                path,
                schema,
                data_cache_bytes,
                version,
                storage_options,
                provider,
                keep_original_array,
                max_page_bytes,
            ),
        )?
    }

    pub fn write_batch(&self, batch: PyArrowType<RecordBatch>) -> PyResult<()> {
        rt().block_on(None, async {
            self.inner.lock().await.write_batch(&batch.0).await
        })?
        .infer_error()
    }

    /// Finish the file and return the row count and the final file size
    ///
    /// The size is reported by the object writer once the file has been closed
    /// and so it is accurate for object stores as well as local filesystems.
    pub fn finish(&self) -> PyResult<LanceFileWriteSummary> {
        let summary = rt()
            .block_on(None, async { self.inner.lock().await.finish().await })?
            .infer_error()?;
        Ok(LanceFileWriteSummary {
            num_rows: summary.num_rows,
            size_bytes: summary.size_bytes,
        })
    }

    pub fn add_global_buffer(&self, bytes: Vec<u8>) -> PyResult<u32> {
        rt().block_on(None, async {
            self.inner
                .lock()
                .await
                .add_global_buffer(Bytes::from(bytes))
                .await
        })?
        .infer_error()
    }

    pub fn add_schema_metadata(&self, key: String, value: String) -> PyResult<()> {
        rt().block_on(None, async {
            self.inner.lock().await.add_schema_metadata(key, value)
        })?;
        Ok(())
    }
}

impl Drop for LanceFileWriter {
    fn drop(&mut self) {
        rt().runtime.block_on(async {
            let mut inner = self.inner.lock().await;
            inner.abort().await;
        });
    }
}

pub async fn object_store_from_uri_or_path_no_options(
    uri_or_path: impl AsRef<str>,
) -> PyResult<(Arc<ObjectStore>, Path)> {
    object_store_from_uri_or_path(uri_or_path, None).await
}

pub async fn object_store_from_uri_or_path(
    uri_or_path: impl AsRef<str>,
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<(Arc<ObjectStore>, Path)> {
    object_store_from_uri_or_path_with_provider(uri_or_path, storage_options, None).await
}

pub async fn object_store_from_uri_or_path_with_provider(
    uri_or_path: impl AsRef<str>,
    storage_options: Option<HashMap<String, String>>,
    storage_options_provider: Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>>,
) -> PyResult<(Arc<ObjectStore>, Path)> {
    let object_store_registry = Arc::new(lance::io::ObjectStoreRegistry::default());

    let accessor = match (storage_options, storage_options_provider) {
        (Some(opts), Some(provider)) => Some(Arc::new(
            lance::io::StorageOptionsAccessor::with_initial_and_provider(opts, provider),
        )),
        (None, Some(provider)) => Some(Arc::new(lance::io::StorageOptionsAccessor::with_provider(
            provider,
        ))),
        (Some(opts), None) => Some(Arc::new(
            lance::io::StorageOptionsAccessor::with_static_options(opts),
        )),
        (None, None) => None,
    };

    let object_store_params = ObjectStoreParams {
        storage_options_accessor: accessor,
        ..Default::default()
    };

    let (object_store, path) = ObjectStore::from_uri_and_params(
        object_store_registry,
        uri_or_path.as_ref(),
        &object_store_params,
    )
    .await
    .infer_error()?;

    Ok((object_store, path))
}

#[pyclass]
pub struct LanceFileSession {
    object_store: Arc<ObjectStore>,
    base_path: Path,
}

impl LanceFileSession {
    pub async fn try_new(
        uri_or_path: String,
        storage_options: Option<HashMap<String, String>>,
        storage_options_provider: Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>>,
    ) -> PyResult<Self> {
        let (object_store, base_path) = object_store_from_uri_or_path_with_provider(
            uri_or_path,
            storage_options,
            storage_options_provider,
        )
        .await?;
        Ok(Self {
            object_store,
            base_path,
        })
    }
}

#[pymethods]
impl LanceFileSession {
    #[new]
    #[pyo3(signature=(uri_or_path, storage_options=None, namespace_client=None, table_id=None))]
    pub fn new(
        uri_or_path: String,
        storage_options: Option<HashMap<String, String>>,
        namespace_client: Option<&Bound<'_, PyAny>>,
        table_id: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // Create storage options provider from namespace_client and table_id if both are provided
        let provider = if let (Some(ns_client), Some(tid)) = (&namespace_client, &table_id) {
            let ns_client = extract_namespace_arc(ns_client.py(), ns_client)?;
            Some(Arc::new(LanceNamespaceStorageOptionsProvider::new(
                ns_client,
                tid.clone(),
            ))
                as Arc<dyn lance_io::object_store::StorageOptionsProvider>)
        } else {
            None
        };
        rt().block_on(None, Self::try_new(uri_or_path, storage_options, provider))?
    }

    #[pyo3(signature=(path, columns=None))]
    pub fn open_reader(
        &self,
        path: String,
        columns: Option<Vec<String>>,
    ) -> PyResult<LanceFileReader> {
        let path = self.base_path.child_path(&Path::from(path));
        rt().block_on(
            None,
            LanceFileReader::open_with_store(self.object_store.clone(), path, columns),
        )?
    }

    #[pyo3(signature=(
        path,
        schema=None,
        data_cache_bytes=None,
        version=None,
        keep_original_array=None,
        max_page_bytes=None
    ))]
    pub fn open_writer(
        &self,
        path: String,
        schema: Option<PyArrowType<ArrowSchema>>,
        data_cache_bytes: Option<u64>,
        version: Option<String>,
        keep_original_array: Option<bool>,
        max_page_bytes: Option<u64>,
    ) -> PyResult<LanceFileWriter> {
        let path = self.base_path.child_path(&Path::from(path));
        rt().block_on(
            None,
            LanceFileWriter::open_with_store(
                self.object_store.clone(),
                path,
                schema,
                data_cache_bytes,
                version,
                keep_original_array,
                max_page_bytes,
            ),
        )?
    }

    pub fn open_packed_blob_writer(
        &self,
        path: String,
        blob_id: u32,
    ) -> PyResult<PyPackedBlobWriter> {
        let path = self.base_path.child_path(&Path::from(path));
        rt().block_on(
            None,
            PyPackedBlobWriter::try_new(self.object_store.clone(), path, blob_id),
        )?
    }

    pub fn open_dedicated_blob_writer(
        &self,
        path: String,
        blob_id: u32,
    ) -> PyResult<PyDedicatedBlobWriter> {
        let path = self.base_path.child_path(&Path::from(path));
        rt().block_on(
            None,
            PyDedicatedBlobWriter::try_new(self.object_store.clone(), path, blob_id),
        )?
    }

    pub fn contains(&self, path: String) -> PyResult<bool> {
        let full_path = self.base_path.child_path(&Path::from(path));
        rt().block_on(None, async {
            self.object_store
                .exists(&full_path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        })?
    }

    pub fn list(&self, path: Option<String>) -> PyResult<Vec<String>> {
        use futures::stream::StreamExt;

        rt().block_on(None, async {
            // Construct the full path to list from
            let list_path = if let Some(prefix) = path {
                self.base_path.child_path(&Path::from(prefix))
            } else {
                self.base_path.clone()
            };

            // List all files under the specified path
            let stream = self.object_store.list(Some(list_path));
            let results: Vec<_> = stream.collect().await;

            let mut paths: Vec<String> = Vec::new();
            for meta_result in results {
                let meta = meta_result.map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                })?;

                // Strip the base_path prefix to make it relative
                // Use prefix_match which handles path separators correctly across platforms
                let relative_parts =
                    meta.location.prefix_match(&self.base_path).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Path '{}' does not start with base path '{}'",
                            meta.location.as_ref(),
                            self.base_path.as_ref()
                        ))
                    })?;
                let relative = Path::from_iter(relative_parts).as_ref().to_string();

                paths.push(relative);
            }

            Ok(paths)
        })?
    }

    /// Non-recursive, delimited list of a single directory level.
    ///
    /// Returns a tuple `(common_prefixes, objects)` of paths relative to the
    /// session's `base_path`, where `common_prefixes` are the immediate child
    /// "directories" and `objects` are the immediate child files. Unlike
    /// `list`, this does not recurse into the subtree.
    #[pyo3(signature=(path=None))]
    pub fn list_with_delimiter(
        &self,
        path: Option<String>,
    ) -> PyResult<(Vec<String>, Vec<String>)> {
        rt().block_on(None, async {
            let list_path = if let Some(prefix) = path {
                self.base_path.child_path(&Path::from(prefix))
            } else {
                self.base_path.clone()
            };

            let result = self
                .object_store
                .list_with_delimiter(Some(&list_path))
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

            // Strip the base_path prefix to make each path relative to the session.
            let relativize = |location: &Path| -> PyResult<String> {
                let relative_parts = location.prefix_match(&self.base_path).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Path '{}' does not start with base path '{}'",
                        location.as_ref(),
                        self.base_path.as_ref()
                    ))
                })?;
                Ok(Path::from_iter(relative_parts).as_ref().to_string())
            };

            let common_prefixes = result
                .common_prefixes
                .iter()
                .map(relativize)
                .collect::<PyResult<Vec<String>>>()?;
            let objects = result
                .objects
                .iter()
                .map(|meta| relativize(&meta.location))
                .collect::<PyResult<Vec<String>>>()?;

            Ok((common_prefixes, objects))
        })?
    }

    /// Upload a file from local filesystem to the object store
    ///
    /// Parameters
    /// ----------
    /// local_path : str
    ///     Local file path to upload
    /// remote_path : str
    ///     Remote path relative to session's base_path
    pub fn upload_file(&self, local_path: String, remote_path: String) -> PyResult<()> {
        rt().block_on(None, async {
            let local_file = tokio::fs::File::open(&local_path).await.map_err(|e| {
                PyIOError::new_err(format!("Failed to open local file {}: {}", local_path, e))
            })?;
            let mut reader = tokio::io::BufReader::new(local_file);
            let full_path = self.base_path.child_path(&Path::from(remote_path));

            let mut writer =
                self.object_store.create(&full_path).await.map_err(|e| {
                    PyIOError::new_err(format!("Failed to create remote file: {}", e))
                })?;

            tokio::io::copy(&mut reader, &mut writer)
                .await
                .map_err(|e| PyIOError::new_err(format!("Failed to upload file: {}", e)))?;
            Writer::shutdown(writer.as_mut())
                .await
                .map_err(|e| PyIOError::new_err(format!("Failed to finalize upload: {}", e)))?;

            Ok(())
        })?
    }

    /// Delete a file from the object store.
    ///
    /// The path is interpreted relative to the session's `base_path`, matching
    /// `contains`/`upload_file`/`download_file`. Deleting a missing object
    /// raises `OSError`, consistent with `download_file`.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path relative to `base_path` to delete.
    pub fn delete_file(&self, path: String) -> PyResult<()> {
        rt().block_on(None, async {
            let full_path = self.base_path.child_path(&Path::from(path));
            self.object_store
                .inner
                .delete(&full_path)
                .await
                .map_err(|e| PyIOError::new_err(format!("Failed to delete remote file: {}", e)))?;
            Ok(())
        })?
    }

    /// Read a byte range from a file in the object store.
    ///
    /// The path is interpreted relative to the session's `base_path`, matching
    /// the other session methods. This issues a single ranged GET. Reading a
    /// missing object raises `OSError`, consistent with `download_file`.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path relative to `base_path` to read from.
    /// offset : int
    ///     Byte offset at which to start reading.
    /// length : int
    ///     Number of bytes to read.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The requested byte range.
    pub fn read_range(&self, path: String, offset: usize, length: usize) -> PyResult<Vec<u8>> {
        rt().block_on(None, async {
            let full_path = self.base_path.child_path(&Path::from(path));
            let bytes = self
                .object_store
                .read_one_range(&full_path, offset..offset + length)
                .await
                .map_err(|e| {
                    PyIOError::new_err(format!("Failed to read range from remote file: {}", e))
                })?;
            Ok(bytes.to_vec())
        })?
    }

    /// Download a file from object store to local filesystem
    ///
    /// Parameters
    /// ----------
    /// remote_path : str
    ///     Remote path relative to session's base_path
    /// local_path : str
    ///     Local file path where the file will be saved
    pub fn download_file(&self, remote_path: String, local_path: String) -> PyResult<()> {
        rt().block_on(None, async {
            let full_path = self.base_path.child_path(&Path::from(remote_path));
            let get_result = self
                .object_store
                .inner
                .get(&full_path)
                .await
                .map_err(|e| PyIOError::new_err(format!("Failed to get remote file: {}", e)))?;

            let mut stream = get_result.into_stream();
            let mut writer = tokio::fs::File::create(&local_path).await.map_err(|e| {
                PyIOError::new_err(format!("Failed to create local file {}: {}", local_path, e))
            })?;
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(|e| {
                    PyIOError::new_err(format!("Failed to read chunk from remote: {}", e))
                })?;
                writer.write_all(&chunk).await.map_err(|e| {
                    PyIOError::new_err(format!("Failed to write chunk to local file: {}", e))
                })?;
            }

            writer
                .flush()
                .await
                .map_err(|e| PyIOError::new_err(format!("Failed to flush local file: {}", e)))?;

            Ok(())
        })?
    }
}

#[pyclass]
pub struct LanceFileReader {
    inner: Arc<FileReader>,
}

impl LanceFileReader {
    async fn open(
        uri_or_path: String,
        storage_options: Option<HashMap<String, String>>,
        storage_options_provider: Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let (object_store, path) = object_store_from_uri_or_path_with_provider(
            uri_or_path,
            storage_options,
            storage_options_provider,
        )
        .await?;
        Self::open_with_store(object_store, path, columns).await
    }

    async fn open_with_store(
        object_store: Arc<ObjectStore>,
        path: Path,
        columns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let scheduler =
            ScanScheduler::new(object_store, SchedulerConfig::new(2 * 1024 * 1024 * 1024));
        let file = scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .infer_error()?;
        let file_metadata = FileReader::read_all_metadata(&file)
            .await
            .map_err(|e| PyIOError::new_err(format!("Error reading file metadata: {}", e)))?;

        let mut base_projection = None;
        if let Some(columns) = columns {
            base_projection = Some(
                file_versions::reader_projection_from_column_names(
                    file_metadata.version(),
                    &file_metadata.file_schema,
                    &columns.iter().map(|s| s.as_str()).collect::<Vec<&str>>(),
                )
                .map_err(|e| PyIOError::new_err(format!("Error creating projection: {}", e)))?,
            );
        }

        let inner = FileReader::try_open_with_file_metadata(
            Arc::new(LanceEncodingsIo::new(file.clone())),
            path,
            base_projection,
            Arc::<DecoderPlugins>::default(),
            Arc::new(file_metadata),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await
        .infer_error()?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

struct LanceReaderAdapter(Pin<Box<dyn RecordBatchStream>>);

impl Iterator for LanceReaderAdapter {
    type Item = std::result::Result<RecordBatch, arrow::error::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = rt().block_on(None, self.0.next()).ok()?;
        batch.map(|b| b.map_err(|e| e.into()))
    }
}

impl RecordBatchReader for LanceReaderAdapter {
    fn schema(&self) -> std::sync::Arc<arrow_schema::Schema> {
        self.0.schema().clone()
    }
}

impl LanceFileReader {
    fn read_stream(
        &mut self,
        params: ReadBatchParams,
        batch_size: u32,
        batch_readahead: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let inner = self.inner.clone();
        let stream = rt().block_on(None, async move {
            inner
                .read_stream(
                    params,
                    batch_size,
                    batch_readahead,
                    FilterExpression::no_filter(),
                )
                .await
                .infer_error()
        })??;
        Ok(PyArrowType(Box::new(LanceReaderAdapter(stream))))
    }
}

#[pymethods]
impl LanceFileReader {
    #[new]
    #[pyo3(signature=(path, storage_options=None, namespace_client=None, table_id=None, columns=None))]
    pub fn new(
        path: String,
        storage_options: Option<HashMap<String, String>>,
        namespace_client: Option<&Bound<'_, PyAny>>,
        table_id: Option<Vec<String>>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // Create storage options provider from namespace_client and table_id if both are provided
        let provider = if let (Some(ns_client), Some(tid)) = (&namespace_client, &table_id) {
            let ns_client = extract_namespace_arc(ns_client.py(), ns_client)?;
            Some(Arc::new(LanceNamespaceStorageOptionsProvider::new(
                ns_client,
                tid.clone(),
            ))
                as Arc<dyn lance_io::object_store::StorageOptionsProvider>)
        } else {
            None
        };
        rt().block_on(None, Self::open(path, storage_options, provider, columns))?
    }

    pub fn read_all(
        &mut self,
        batch_size: u32,
        batch_readahead: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        self.read_stream(
            lance_io::ReadBatchParams::RangeFull,
            batch_size,
            batch_readahead,
        )
    }

    pub fn read_range(
        &mut self,
        offset: u64,
        num_rows: u64,
        batch_size: u32,
        batch_readahead: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        self.read_stream(
            lance_io::ReadBatchParams::Range((offset as usize)..(offset + num_rows) as usize),
            batch_size,
            batch_readahead,
        )
    }

    pub fn take_rows(
        &mut self,
        row_indices: Vec<u64>,
        batch_size: u32,
        batch_readahead: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let indices = row_indices
            .into_iter()
            .map(|idx| idx as u32)
            .collect::<Vec<_>>();
        let indices_arr = UInt32Array::from(indices);
        self.read_stream(
            lance_io::ReadBatchParams::Indices(indices_arr),
            batch_size,
            batch_readahead,
        )
    }

    pub fn metadata(&mut self, py: Python) -> LanceFileMetadata {
        let inner_meta = self.inner.metadata();
        LanceFileMetadata::new(inner_meta, py)
    }

    pub fn file_statistics(&self) -> LanceFileStatistics {
        let inner_stat = self.inner.file_statistics();
        LanceFileStatistics::new(&inner_stat)
    }

    pub fn read_global_buffer(&mut self, index: u32) -> PyResult<Vec<u8>> {
        let buffer_bytes = rt()
            .runtime
            .block_on(self.inner.read_global_buffer(index))
            .infer_error()?;
        Ok(buffer_bytes.to_vec())
    }

    pub fn num_rows(&mut self) -> u64 {
        self.inner.num_rows()
    }
}

#[pyfunction(name = "stable_version")]
pub fn stable_version() -> PyResult<String> {
    Ok(LanceFileVersion::Stable.resolve().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lance_file_statistics_repr_empty() {
        let stats = LanceFileStatistics { columns: vec![] };

        let repr_str = stats.__repr__();
        assert_eq!(repr_str, "FileStatistics(columns=[])");
    }

    #[test]
    fn test_lance_file_statistics_repr_single_column() {
        let stats = LanceFileStatistics {
            columns: vec![LanceColumnStatistics {
                num_pages: 5,
                size_bytes: 1024,
            }],
        };

        let repr_str = stats.__repr__();
        assert_eq!(
            repr_str,
            "FileStatistics(columns=[ColumnStatistics(num_pages=5, size_bytes=1024)])"
        );
    }

    #[test]
    fn test_lance_file_statistics_repr_multiple_columns() {
        let stats = LanceFileStatistics {
            columns: vec![
                LanceColumnStatistics {
                    num_pages: 5,
                    size_bytes: 1024,
                },
                LanceColumnStatistics {
                    num_pages: 3,
                    size_bytes: 512,
                },
            ],
        };

        let repr_str = stats.__repr__();
        assert_eq!(
            repr_str,
            "FileStatistics(columns=[ColumnStatistics(num_pages=5, size_bytes=1024), ColumnStatistics(num_pages=3, size_bytes=512)])"
        );
    }

    #[test]
    fn test_lance_column_statistics_repr() {
        let column_stats = LanceColumnStatistics {
            num_pages: 10,
            size_bytes: 2048,
        };

        let repr_str = column_stats.__repr__();
        assert_eq!(repr_str, "ColumnStatistics(num_pages=10, size_bytes=2048)");
    }
}
