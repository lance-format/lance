// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::str;
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

use arrow::array::AsArray;
use arrow::datatypes::UInt8Type;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::*;
use arrow_array::Array;
use arrow_array::{RecordBatch, RecordBatchReader, make_array};
use arrow_cast::cast_with_options;
use arrow_data::ArrayData;
use arrow_schema::{DataType, Schema as ArrowSchema};
use async_trait::async_trait;
use blob::LanceBlobFile;
use chrono::{Duration, TimeDelta, Utc};
use futures::{StreamExt, TryFutureExt};
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::bq::storage::RabitQuantizationMetadata;
use log::error;
use object_store::path::Path;
use pyo3::exceptions::{PyStopIteration, PyTypeError};
use pyo3::types::{PyBytes, PyInt, PyList, PyString, PyTuple};
use pyo3::{IntoPyObjectExt, prelude::*};
use pyo3::{
    PyResult,
    exceptions::{PyIOError, PyKeyError, PyValueError},
    intern,
    pybacked::PyBackedStr,
    pyclass,
    types::{IntoPyDict, PyDict},
};
use uuid::Uuid;

use lance::dataset::AutoCleanupParams;
use lance::dataset::cleanup::CleanupPolicyBuilder;
use lance::dataset::refs::{Ref, TagContents};
use lance::dataset::scanner::{
    AggregateExpr, ColumnOrdering, DatasetRecordBatchStream, ExecutionStatsCallback,
    MaterializationStyle, QueryFilter,
};
use lance::dataset::statistics::{DataStatistics, DatasetStatisticsExt};
use lance::dataset::{
    BatchInfo, BatchUDF, CommitBuilder, MergeStats, NewColumnTransform, UDFCheckpointStore,
    WriteDestination,
};
use lance::dataset::{ColumnAlteration, ProjectionRequest};
use lance::dataset::{
    Dataset as LanceDataset, DeleteBuilder, ExternalBlobMode,
    MergeInsertBuilder as LanceMergeInsertBuilder, ReadParams, UncommittedMergeInsert,
    UpdateBuilder, Version, WhenMatched, WhenNotMatched, WhenNotMatchedBySource, WriteMode,
    WriteParams,
    fragment::FileFragment as LanceFileFragment,
    progress::WriteFragmentProgress,
    scanner::Scanner as LanceScanner,
    transaction::{Operation, Transaction},
};
use lance::index::vector::utils::get_vector_type;
use lance::index::{DatasetIndexExt, IndexSegment, vector::VectorIndexParams};
use lance::{dataset::builder::DatasetBuilder, index::vector::IndexFileVersion};
use lance_arrow::as_fixed_size_list_array;
use lance_core::Error;
use lance_core::datatypes::BlobHandling;
use lance_datafusion::utils::reader_to_stream;
use lance_encoding::decoder::DecoderConfig;
use lance_file::reader::FileReaderOptions;
use lance_index::scalar::inverted::query::Occur;
use lance_index::scalar::inverted::query::{
    BooleanQuery, BoostQuery, FtsQuery, MatchQuery, MultiMatchQuery, Operator, PhraseQuery,
};
use lance_index::{
    FtsPrewarmOptions, IndexParams, IndexType, PrewarmOptions,
    optimize::OptimizeOptions,
    progress::{IndexBuildProgress, NoopIndexBuildProgress},
    scalar::{FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams},
    vector::{
        ApproxMode, DEFAULT_QUERY_PARALLELISM, Query as VectorQuery,
        hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, pq::PQBuildParams,
        sq::builder::SQBuildParams,
    },
};
use lance_io::object_store::{
    LanceNamespaceStorageOptionsProvider, ObjectStoreParams, StorageOptionsAccessor,
};
use lance_linalg::distance::MetricType;
use lance_table::format::{BasePath, Fragment, IndexMetadata};
use lance_table::io::commit::CommitHandler;
use lance_table::io::commit::external_manifest::ExternalManifestCommitHandler;

use crate::error::PythonErrorExt;
use crate::file::object_store_from_uri_or_path;
use crate::fragment::FileFragment;
use crate::indices::{PyIndexConfig, PyIndexDescription, PyIndexSegment};
use crate::namespace::extract_namespace_arc;
use crate::rt;
use crate::scanner::ScanStatistics;
use crate::schema::{LanceSchema, logical_schema_from_lance};
use crate::session::Session;
use crate::storage_options::PyStorageOptionsAccessor;
use crate::utils::PyLance;
use crate::{LanceReader, Scanner};
use lance::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;

use self::cleanup::CleanupStats;
use self::commit::PyCommitLock;
use self::io_stats::IoStats;

pub mod blob;
pub mod cleanup;
pub mod commit;
pub mod io_stats;
pub mod optimize;
pub mod stats;

const DEFAULT_NPROBES: usize = 1;
const LANCE_COMMIT_MESSAGE_KEY: &str = "__lance_commit_message";
const INDEX_PROGRESS_QUEUE_SIZE: usize = 1024;

fn read_blobs_to_python(
    py: Python<'_>,
    blobs: Vec<lance::dataset::ReadBlob>,
) -> Vec<(u64, Py<PyBytes>)> {
    blobs
        .into_iter()
        .map(|blob| (blob.row_address, PyBytes::new(py, &blob.data).unbind()))
        .collect()
}

fn configure_read_blobs_builder(
    mut builder: lance::dataset::ReadBlobsBuilder,
    io_buffer_size: Option<u64>,
    preserve_order: Option<bool>,
) -> lance::dataset::ReadBlobsBuilder {
    if let Some(bytes) = io_buffer_size {
        builder = builder.with_io_buffer_size_bytes(bytes);
    }
    if let Some(preserve) = preserve_order {
        builder = builder.preserve_order(preserve);
    }
    builder
}

fn stats_log_interval_from_millis(ms: u64) -> Option<std::time::Duration> {
    if ms == 0 {
        None
    } else {
        Some(std::time::Duration::from_millis(ms))
    }
}

/// Build a `ShardWriterConfig` from optional MemWAL writer keyword arguments.
///
/// Returns `None` when no keyword argument is supplied, so callers can tell
/// "use the defaults" apart from an explicit configuration.
#[allow(clippy::too_many_arguments)]
fn writer_config_from_kwargs(
    durable_write: Option<bool>,
    sync_indexed_write: Option<bool>,
    max_wal_buffer_size: Option<usize>,
    max_wal_flush_interval_ms: Option<u64>,
    max_memtable_size: Option<usize>,
    max_memtable_rows: Option<usize>,
    max_memtable_batches: Option<usize>,
    max_unflushed_memtable_bytes: Option<usize>,
    manifest_scan_batch_size: Option<usize>,
    async_index_buffer_rows: Option<usize>,
    async_index_interval_ms: Option<u64>,
    backpressure_log_interval_ms: Option<u64>,
    stats_log_interval_ms: Option<u64>,
    hnsw_params: Option<HashMap<String, HashMap<String, u32>>>,
) -> PyResult<Option<lance::dataset::mem_wal::ShardWriterConfig>> {
    use std::time::Duration;

    let mut config = lance::dataset::mem_wal::ShardWriterConfig::default();
    let mut any = false;
    if let Some(v) = durable_write {
        config = config.with_durable_write(v);
        any = true;
    }
    if let Some(v) = sync_indexed_write {
        config = config.with_sync_indexed_write(v);
        any = true;
    }
    if let Some(v) = max_wal_buffer_size {
        config = config.with_max_wal_buffer_size(v);
        any = true;
    }
    if let Some(v) = max_wal_flush_interval_ms {
        config = config.with_max_wal_flush_interval(Duration::from_millis(v));
        any = true;
    }
    if let Some(v) = max_memtable_size {
        config = config.with_max_memtable_size(v);
        any = true;
    }
    if let Some(v) = max_memtable_rows {
        config = config.with_max_memtable_rows(v);
        any = true;
    }
    if let Some(v) = max_memtable_batches {
        config = config.with_max_memtable_batches(v);
        any = true;
    }
    if let Some(v) = max_unflushed_memtable_bytes {
        config = config.with_max_unflushed_memtable_bytes(v);
        any = true;
    }
    if let Some(v) = manifest_scan_batch_size {
        config = config.with_manifest_scan_batch_size(v);
        any = true;
    }
    if let Some(v) = async_index_buffer_rows {
        config = config.with_async_index_buffer_rows(v);
        any = true;
    }
    if let Some(v) = async_index_interval_ms {
        config = config.with_async_index_interval(Duration::from_millis(v));
        any = true;
    }
    if let Some(v) = backpressure_log_interval_ms {
        config = config.with_backpressure_log_interval(Duration::from_millis(v));
        any = true;
    }
    if let Some(v) = stats_log_interval_ms {
        config = config.with_stats_log_interval(stats_log_interval_from_millis(v));
        any = true;
    }
    if let Some(overrides) = hnsw_params {
        // {index_name: {param: value}} -> HnswBuildParams, rejecting unknown
        // keys so typos surface instead of being silently ignored.
        for (index_name, params) in overrides {
            let mut build = HnswBuildParams::default();
            for (key, value) in params {
                match key.as_str() {
                    "num_edges" => build = build.num_edges(value as usize),
                    "ef_construction" => build = build.ef_construction(value as usize),
                    "max_level" => build = build.max_level(value as u16),
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown HNSW build param '{}' for index '{}'; \
                             expected one of 'num_edges', 'ef_construction', 'max_level'",
                            other, index_name
                        )));
                    }
                }
            }
            config = config.with_hnsw_params(index_name, build);
            any = true;
        }
    }
    Ok(any.then_some(config))
}

fn convert_reader(reader: &Bound<PyAny>) -> PyResult<Box<dyn RecordBatchReader + Send>> {
    let py = reader.py();
    if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        Ok(Box::new(
            rt().spawn(Some(py), async move { scanner.to_reader().await })?
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        ))
    } else {
        Ok(Box::new(ArrowArrayStreamReader::from_pyarrow_bound(
            reader,
        )?))
    }
}

#[pyclass(name = "_MergeInsertBuilder", module = "_lib", subclass)]
pub struct MergeInsertBuilder {
    builder: LanceMergeInsertBuilder,
    dataset: Py<Dataset>,
}

#[pymethods]
impl MergeInsertBuilder {
    #[new]
    #[pyo3(signature=(dataset, on=None))]
    pub fn new(dataset: &Bound<'_, PyAny>, on: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let dataset_py: Py<Dataset> = dataset.extract()?;
        let py = dataset.py();
        let ds = dataset_py.borrow(py).ds.clone();

        // Either a single string, which we put in a vector or an iterator
        // of strings, which we collect into a vector. If `on` is None, we
        // pass an empty vector and let the Rust builder fall back to the
        // schema's unenforced primary key (if configured).
        let on = if let Some(on_any) = on {
            on_any
                .cast::<PyString>()
                .map(|val| vec![val.to_string()])
                .or_else(|_| {
                    let iterator = on_any.try_iter().map_err(|_| {
                        PyTypeError::new_err(
                            "The `on` argument to merge_insert must be a str or iterable of str",
                        )
                    })?;
                    let mut keys = Vec::new();
                    for key in iterator {
                        keys.push(key?.cast::<PyString>()?.to_string());
                    }
                    PyResult::Ok(keys)
                })?
        } else {
            Vec::new()
        };

        let mut builder = LanceMergeInsertBuilder::try_new(ds, on)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        // We don't have do_nothing methods in python so we start with a blank slate
        builder
            .when_matched(WhenMatched::DoNothing)
            .when_not_matched(WhenNotMatched::DoNothing);

        Ok(Self {
            builder,
            dataset: dataset_py,
        })
    }

    #[pyo3(signature=(condition=None))]
    pub fn when_matched_update_all<'a>(
        mut slf: PyRefMut<'a, Self>,
        condition: Option<&str>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let new_val = if let Some(expr) = condition {
            let dataset = slf.dataset.borrow(slf.py());
            WhenMatched::update_if(&dataset.ds, expr)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
        } else {
            WhenMatched::UpdateAll
        };
        slf.builder.when_matched(new_val);
        Ok(slf)
    }

    pub fn when_matched_fail(mut slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.builder.when_matched(WhenMatched::Fail);
        Ok(slf)
    }

    pub fn when_matched_delete(mut slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.builder.when_matched(WhenMatched::Delete);
        Ok(slf)
    }

    pub fn when_not_matched_insert_all(mut slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.builder.when_not_matched(WhenNotMatched::InsertAll);
        Ok(slf)
    }

    #[pyo3(signature=(expr=None))]
    pub fn when_not_matched_by_source_delete<'a>(
        mut slf: PyRefMut<'a, Self>,
        expr: Option<&str>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let new_val = if let Some(expr) = expr {
            let dataset = slf.dataset.borrow(slf.py());
            WhenNotMatchedBySource::delete_if(&dataset.ds, expr)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
        } else {
            WhenNotMatchedBySource::Delete
        };
        slf.builder.when_not_matched_by_source(new_val);
        Ok(slf)
    }

    pub fn conflict_retries(
        mut slf: PyRefMut<'_, Self>,
        max_retries: u32,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.conflict_retries(max_retries);
        Ok(slf)
    }

    pub fn retry_timeout(
        mut slf: PyRefMut<'_, Self>,
        timeout: std::time::Duration,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.retry_timeout(timeout);
        Ok(slf)
    }

    pub fn use_index(mut slf: PyRefMut<'_, Self>, use_index: bool) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.use_index(use_index);
        Ok(slf)
    }

    pub fn execute(&mut self, new_data: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = new_data.py();
        let new_data = convert_reader(new_data)?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let (new_dataset, stats) = rt()
            .spawn(Some(py), job.execute_reader(new_data))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        let dataset = self.dataset.bind(py);

        dataset.borrow_mut().ds = new_dataset;

        Ok(Self::build_stats(&stats, py)?.into())
    }

    pub fn execute_uncommitted<'a>(
        &mut self,
        new_data: &Bound<'a, PyAny>,
    ) -> PyResult<(PyLance<Transaction>, Bound<'a, PyDict>)> {
        let py = new_data.py();
        let new_data = convert_reader(new_data)?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let UncommittedMergeInsert {
            transaction, stats, ..
        } = rt()
            .spawn(Some(py), job.execute_uncommitted(new_data))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        let stats = Self::build_stats(&stats, py)?;

        Ok((PyLance(transaction), stats))
    }

    #[pyo3(signature=(schema = None, verbose = false))]
    pub fn explain_plan(
        &mut self,
        schema: Option<PyArrowType<ArrowSchema>>,
        verbose: Option<bool>,
    ) -> PyResult<String> {
        let verbose = verbose.unwrap_or(false);
        let job = self
            .builder
            .clone()
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let schema_ref = schema.as_ref().map(|s| &s.0);
        rt().block_on(None, job.explain_plan(schema_ref, verbose))?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    pub fn analyze_plan(&mut self, new_data: &Bound<PyAny>) -> PyResult<String> {
        let new_data_reader = convert_reader(new_data)?;
        let new_data_stream = reader_to_stream(new_data_reader);
        let job = self
            .builder
            .clone()
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        rt().block_on(None, job.analyze_plan(new_data_stream))?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    /// Mark MemWAL generations as merged into the base table.
    ///
    /// Call this when executing a merge_insert that incorporates MemWAL
    /// flushed generation data. This updates the MemWAL generation tracking
    /// to prevent duplicate merges.
    pub fn mark_generations_as_merged<'a>(
        mut slf: PyRefMut<'a, Self>,
        generations: Vec<Bound<'a, crate::mem_wal::PyMergedGeneration>>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        use lance_index::mem_wal::MergedGeneration;

        let gens: Vec<MergedGeneration> = generations
            .iter()
            .map(|g| g.borrow().to_lance())
            .collect::<PyResult<_>>()?;
        slf.builder.mark_generations_as_merged(gens);
        Ok(slf)
    }
}

fn index_metadata_to_segment(metadata: IndexMetadata) -> PyResult<IndexSegment> {
    let fragment_bitmap = metadata.fragment_bitmap.ok_or_else(|| {
        PyValueError::new_err(format!(
            "Index metadata {} is missing fragment coverage",
            metadata.uuid
        ))
    })?;
    let index_details = metadata.index_details.ok_or_else(|| {
        PyValueError::new_err(format!(
            "Index metadata {} is missing index details",
            metadata.uuid
        ))
    })?;

    Ok(IndexSegment::new(
        metadata.uuid,
        fragment_bitmap.iter(),
        index_details,
        metadata.index_version,
    ))
}

fn extract_index_segments(segments: &Bound<'_, PyAny>) -> PyResult<Vec<IndexSegment>> {
    let mut extracted = Vec::new();
    for item in segments.try_iter()? {
        let item = item?;
        match item.extract::<PyRef<'_, PyIndexSegment>>() {
            Ok(segment) => extracted.push(segment.inner.clone()),
            Err(segment_err) => match item.extract::<PyLance<IndexMetadata>>() {
                Ok(metadata) => extracted.push(index_metadata_to_segment(metadata.0)?),
                Err(metadata_err) => {
                    return Err(PyTypeError::new_err(format!(
                        "commit_existing_index_segments expected IndexSegment or Index items; \
                         failed to read item as IndexSegment ({segment_err}) or Index ({metadata_err})"
                    )));
                }
            },
        }
    }
    Ok(extracted)
}

impl MergeInsertBuilder {
    fn build_stats<'a>(stats: &MergeStats, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("num_inserted_rows", stats.num_inserted_rows)?;
        dict.set_item("num_updated_rows", stats.num_updated_rows)?;
        dict.set_item("num_deleted_rows", stats.num_deleted_rows)?;
        Ok(dict)
    }
}

pub fn transforms_from_python(
    py: Python<'_>,
    transforms: &Bound<'_, PyAny>,
) -> PyResult<NewColumnTransform> {
    if let Ok(transforms) = transforms.cast::<PyDict>() {
        let expressions = transforms
            .iter()
            .map(|(k, v)| {
                let col = k.extract::<String>()?;
                let expr = v.extract::<String>()?;
                Ok((col, expr))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(NewColumnTransform::SqlExpressions(expressions))
    } else {
        let append_schema: PyArrowType<ArrowSchema> =
            transforms.getattr("output_schema")?.extract()?;
        let output_schema = Arc::new(append_schema.0);

        let result_checkpoint: Option<Py<PyAny>> = transforms.getattr("cache")?.extract()?;
        let result_checkpoint = result_checkpoint.map(|c| PyBatchUDFCheckpointWrapper { inner: c });

        let udf_obj = transforms.into_py_any(py)?;
        let mapper = move |batch: &RecordBatch| -> lance::Result<RecordBatch> {
            Python::attach(|py| {
                let py_batch: PyArrowType<RecordBatch> = PyArrowType(batch.clone());
                let result = udf_obj
                    .call_method1(py, "_call", (py_batch,))
                    .map_err(|err| {
                        lance::Error::invalid_input(format_python_error(err, py).unwrap())
                    })?;
                let result_batch: PyArrowType<RecordBatch> = result
                    .extract(py)
                    .map_err(|err: PyErr| lance::Error::invalid_input(err.to_string()))?;
                Ok(result_batch.0)
            })
        };

        Ok(NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: result_checkpoint
                .map(|c| Arc::new(c) as Arc<dyn UDFCheckpointStore>),
        }))
    }
}
impl FromPyObject<'_, '_> for PyLance<ColumnOrdering> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let ascending: bool = ob.getattr("ascending")?.extract()?;
        let nulls_first: bool = ob.getattr("nulls_first")?.extract()?;
        let column_name: String = ob.getattr("column_name")?.extract()?;
        Ok(Self(ColumnOrdering {
            ascending,
            nulls_first,
            column_name,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&ColumnOrdering> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceScanner")))
            .and_then(|cls| cls.getattr(intern!(py, "ColumnOrdering")))
            .expect("Failed to get RewrittenIndex class");

        let column_name = self.0.column_name.clone();
        let ascending = self.0.ascending;
        let nulls_first = self.0.nulls_first;
        cls.call1((column_name, ascending, nulls_first))
    }
}

/// Python binding for BasePath
#[pyclass(name = "DatasetBasePath", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct DatasetBasePath {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub is_dataset_root: bool,
}

#[pymethods]
impl DatasetBasePath {
    #[new]
    #[pyo3(signature = (path, name = None, is_dataset_root = false, id = None))]
    fn new(path: String, name: Option<String>, is_dataset_root: bool, id: Option<u32>) -> Self {
        Self {
            id: id.unwrap_or(0),
            name,
            path,
            is_dataset_root,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetBasePath(id={}, name={:?}, path={}, is_dataset_root={})",
            self.id, self.name, self.path, self.is_dataset_root
        )
    }
}

impl From<BasePath> for DatasetBasePath {
    fn from(base_path: BasePath) -> Self {
        Self {
            id: base_path.id,
            name: base_path.name,
            path: base_path.path,
            is_dataset_root: base_path.is_dataset_root,
        }
    }
}

impl From<DatasetBasePath> for BasePath {
    fn from(py_base_path: DatasetBasePath) -> Self {
        Self::new(
            py_base_path.id,
            py_base_path.path.clone(),
            py_base_path.name,
            py_base_path.is_dataset_root,
        )
    }
}

/// Lance Dataset that will be wrapped by another class in Python
#[pyclass(name = "_Dataset", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct Dataset {
    #[pyo3(get)]
    uri: String,
    pub(crate) ds: Arc<LanceDataset>,
}

#[pymethods]
impl Dataset {
    #[allow(clippy::too_many_arguments)]
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature=(uri, version=None, block_size=None, index_cache_size=None, metadata_cache_size=None, commit_handler=None, storage_options=None, manifest=None, metadata_cache_size_bytes=None, index_cache_size_bytes=None, read_params=None, session=None, namespace_client=None, table_id=None, namespace_client_managed_versioning=false, base_store_params=None))]
    fn new(
        py: Python,
        uri: String,
        version: Option<Bound<PyAny>>,
        block_size: Option<usize>,
        index_cache_size: Option<usize>,
        metadata_cache_size: Option<usize>,
        commit_handler: Option<Py<PyAny>>,
        storage_options: Option<HashMap<String, String>>,
        manifest: Option<&[u8]>,
        metadata_cache_size_bytes: Option<usize>,
        index_cache_size_bytes: Option<usize>,
        read_params: Option<&Bound<PyDict>>,
        session: Option<Session>,
        namespace_client: Option<&Bound<'_, PyAny>>,
        table_id: Option<Vec<String>>,
        namespace_client_managed_versioning: bool,
        base_store_params: Option<HashMap<String, HashMap<String, String>>>,
    ) -> PyResult<Self> {
        let mut params = ReadParams::default();
        if let Some(metadata_cache_size_bytes) = metadata_cache_size_bytes {
            params.metadata_cache_size_bytes(metadata_cache_size_bytes);
        } else if let Some(metadata_cache_size) = metadata_cache_size {
            #[allow(deprecated)]
            params.metadata_cache_size(metadata_cache_size);
        }
        if let Some(index_cache_size_bytes) = index_cache_size_bytes {
            params.index_cache_size_bytes(index_cache_size_bytes);
        } else if let Some(index_cache_size) = index_cache_size {
            // Convert from number of entries to bytes (assuming 20 MiB per entry)
            // This matches the Rust deprecated method's behavior
            let index_cache_size_bytes = index_cache_size * 20 * 1024 * 1024;
            params.index_cache_size_bytes(index_cache_size_bytes);
        }
        // Set up store options (block size)
        if let Some(block_size) = block_size {
            let mut store_params = params.store_options.take().unwrap_or_default();
            store_params.block_size = Some(block_size);
            params.store_options = Some(store_params);
        }
        if let Some(commit_handler) = commit_handler {
            let py_commit_lock = PyCommitLock::new(commit_handler);
            params.set_commit_lock(Arc::new(py_commit_lock));
        }

        // Handle read_params dict
        if let Some(read_params_dict) = read_params {
            let mut decoder_config = DecoderConfig::default();

            if let Some(cache_repetition_index) = read_params_dict
                .get_item("cache_repetition_index")
                .unwrap_or(None)
                .and_then(|v| v.extract::<bool>().ok())
            {
                decoder_config.cache_repetition_index = cache_repetition_index;
            }

            if let Some(validate_on_decode) = read_params_dict
                .get_item("validate_on_decode")
                .unwrap_or(None)
                .and_then(|v| v.extract::<bool>().ok())
            {
                decoder_config.validate_on_decode = validate_on_decode;
            }

            let file_reader_options = FileReaderOptions {
                decoder_config,
                ..Default::default()
            };
            params.file_reader_options = Some(file_reader_options);
        }

        let mut builder = DatasetBuilder::from_uri(&uri).with_read_params(params);

        if let Some(ver) = version {
            if let Ok(i) = ver.cast::<PyInt>() {
                let v: u64 = i.extract()?;
                builder = builder.with_version(v);
            } else if let Ok(v) = ver.cast::<PyString>() {
                let t: &str = &v.to_string_lossy();
                builder = builder.with_tag(t);
            } else {
                return Err(PyIOError::new_err(
                    "version must be an integer or a string.",
                ));
            };
        }
        // Save a copy of storage options for potential namespace-based credential refresh
        let initial_storage_options = storage_options.clone();

        if let Some(mut storage_options) = storage_options {
            if let Some(user_agent) = storage_options.get_mut("user_agent") {
                user_agent.push_str(&format!(" pylance/{}", env!("CARGO_PKG_VERSION")));
            } else {
                storage_options.insert(
                    "user_agent".to_string(),
                    format!("pylance/{}", env!("CARGO_PKG_VERSION")),
                );
            }

            builder = builder.with_storage_options(storage_options);
        }
        if let Some(manifest) = manifest {
            builder = builder.with_serialized_manifest(manifest).infer_error()?;
        }

        if let Some(session) = session {
            builder = builder.with_session(session.inner.clone());
        }

        // Set up namespace-based features if namespace_client and table_id are provided
        if let (Some(ns_client), Some(tid)) = (&namespace_client, &table_id) {
            let ns_client = extract_namespace_arc(py, ns_client)?;

            // Auto-create storage options provider from namespace client
            // when storage_options are present (meaning credentials came from namespace.describe_table)
            if initial_storage_options.is_some() {
                let provider: Arc<dyn lance_io::object_store::StorageOptionsProvider> = Arc::new(
                    LanceNamespaceStorageOptionsProvider::new(ns_client.clone(), tid.clone()),
                );
                // Create accessor with initial options and provider for credential refresh
                let accessor = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
                    initial_storage_options.clone().unwrap_or_default(),
                    provider,
                ));
                builder = builder.with_storage_options_accessor(accessor);
            }

            // Set up commit handler only if namespace manages versioning
            if namespace_client_managed_versioning {
                // The store derives the branch a request targets from the base
                // path it is handed, resolved against the table root.
                let external_store = LanceNamespaceExternalManifestStore::for_table_uri(
                    ns_client,
                    tid.clone(),
                    &uri,
                )
                .infer_error()?;
                let commit_handler: Arc<dyn CommitHandler> =
                    Arc::new(ExternalManifestCommitHandler {
                        external_manifest_store: Arc::new(external_store),
                    });
                builder = builder.with_commit_handler(commit_handler);
            }
        }

        if let Some(base_store_params) = base_store_params {
            for (base_path, opts) in base_store_params {
                let accessor = Arc::new(StorageOptionsAccessor::with_static_options(opts));
                let store_params = ObjectStoreParams {
                    storage_options_accessor: Some(accessor),
                    ..Default::default()
                };
                builder = builder.with_base_store_params(base_path, store_params);
            }
        }

        let dataset = rt().block_on(Some(py), builder.load())?;

        match dataset {
            Ok(ds) => Ok(Self {
                uri,
                ds: Arc::new(ds),
            }),
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }

    pub fn __copy__(&self) -> Self {
        self.clone()
    }

    #[getter(max_field_id)]
    fn max_field_id(self_: PyRef<'_, Self>) -> PyResult<i32> {
        Ok(self_.ds.manifest().max_field_id())
    }

    #[getter(schema)]
    fn schema<'py>(self_: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let logical_schema = logical_schema_from_lance(self_.ds.schema());
        logical_schema.to_pyarrow(self_.py())
    }

    #[getter(lance_schema)]
    fn lance_schema(self_: PyRef<'_, Self>) -> LanceSchema {
        LanceSchema(self_.ds.schema().clone())
    }

    fn replace_schema_metadata(&mut self, metadata: HashMap<String, String>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let metadata_options: HashMap<String, Option<String>> =
            metadata.into_iter().map(|(k, v)| (k, Some(v))).collect();
        rt().block_on(None, async {
            new_self
                .update_schema_metadata(metadata_options)
                .replace()
                .await
        })?
        .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn replace_field_metadata(
        &mut self,
        field_name: &str,
        metadata: HashMap<String, String>,
    ) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let future = async move {
            new_self
                .update_field_metadata()
                .replace(field_name, metadata)?
                .await?;
            Ok(new_self)
        };

        let new_self = rt()
            .block_on(None, future)?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[getter(data_storage_version)]
    fn data_storage_version(&self) -> PyResult<String> {
        Ok(self.ds.manifest().data_storage_format.version.clone())
    }

    #[getter(has_stable_row_ids)]
    fn has_stable_row_ids(&self) -> bool {
        self.ds.manifest().uses_stable_row_ids()
    }

    /// Get index statistics
    fn index_statistics(&self, index_name: String) -> PyResult<String> {
        rt().block_on(None, self.ds.index_statistics(&index_name))?
            .map_err(|err| match err {
                lance::Error::IndexNotFound { .. } => {
                    PyKeyError::new_err(format!("Index \"{}\" not found", index_name))
                }
                _ => PyIOError::new_err(format!(
                    "Failed to get index statistics for index {}: {}",
                    index_name, err
                )),
            })
    }

    fn serialized_manifest(&self, py: Python) -> Py<PyAny> {
        let manifest_bytes = self.ds.manifest().serialized();
        PyBytes::new(py, &manifest_bytes).into()
    }

    /// Get base paths from the manifest.
    ///
    /// Returns a dictionary mapping base_id to DatasetBasePath objects.
    fn base_paths(&self, py: Python) -> PyResult<Py<PyAny>> {
        let manifest = self.ds.manifest();
        let dict = pyo3::types::PyDict::new(py);

        for (base_id, base_path) in manifest.base_paths.iter() {
            let py_base_path = DatasetBasePath::from(base_path.clone());
            dict.set_item(base_id, py_base_path.into_py_any(py)?)?;
        }

        Ok(dict.into())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(columns=None, columns_with_transform=None, filter=None, search_filter=None, prefilter=None, limit=None, offset=None, nearest=None, batch_size=None, batch_size_bytes=None, io_buffer_size=None, batch_readahead=None, fragment_readahead=None, scan_in_order=None, fragments=None, index_segments=None, with_row_id=None, with_row_address=None, use_stats=None, substrait_filter=None, fast_search=None, full_text_query=None, late_materialization=None, blob_handling=None, use_scalar_index=None, include_deleted_rows=None, scan_stats_callback=None, strict_batch_size=None, order_by=None, disable_scoring_autoprojection=None, substrait_aggregate=None))]
    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        columns_with_transform: Option<Vec<(String, String)>>,
        filter: Option<String>,
        search_filter: Option<PySearchFilter>,
        prefilter: Option<bool>,
        limit: Option<i64>,
        offset: Option<i64>,
        nearest: Option<&Bound<PyDict>>,
        batch_size: Option<usize>,
        batch_size_bytes: Option<u64>,
        io_buffer_size: Option<u64>,
        batch_readahead: Option<usize>,
        fragment_readahead: Option<usize>,
        scan_in_order: Option<bool>,
        fragments: Option<Vec<FileFragment>>,
        index_segments: Option<Vec<String>>,
        with_row_id: Option<bool>,
        with_row_address: Option<bool>,
        use_stats: Option<bool>,
        substrait_filter: Option<Vec<u8>>,
        fast_search: Option<bool>,
        full_text_query: Option<&Bound<'_, PyAny>>,
        late_materialization: Option<Bound<PyAny>>,
        blob_handling: Option<Bound<PyAny>>,
        use_scalar_index: Option<bool>,
        include_deleted_rows: Option<bool>,
        scan_stats_callback: Option<&Bound<'_, PyAny>>,
        strict_batch_size: Option<bool>,
        order_by: Option<Vec<PyLance<ColumnOrdering>>>,
        disable_scoring_autoprojection: Option<bool>,
        substrait_aggregate: Option<Vec<u8>>,
    ) -> PyResult<Scanner> {
        let mut scanner: LanceScanner = self_.ds.scan();

        if with_row_id.unwrap_or(false) {
            scanner.with_row_id();
        }

        if with_row_address.unwrap_or(false) {
            scanner.with_row_address();
        }

        if let Some(false) = disable_scoring_autoprojection {
            return Err(PyValueError::new_err(
                "disable_scoring_autoprojection=False is no longer supported. \
                Scoring autoprojection has been removed. \
                Please include `_distance` or `_score` explicitly in your projection.",
            ));
        }
        // True or None: ignore — no autoprojection is now always the default

        match (columns, columns_with_transform) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Cannot specify both columns and columns_with_transform",
                ));
            }
            (Some(c), None) => {
                scanner
                    .project(&c)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            (None, Some(ct)) => {
                scanner
                    .project_with_transform(&ct)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            (None, None) => {}
        }
        if let Some(f) = filter {
            if substrait_filter.is_some() {
                return Err(PyValueError::new_err(
                    "cannot specify both a string filter and a substrait filter",
                ));
            }
            scanner
                .filter(f.as_str())
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(qf) = search_filter {
            scanner
                .filter_query(qf.inner)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(full_text_query) = full_text_query {
            let fts_query = if let Ok(full_text_query) = full_text_query.cast::<PyDict>() {
                let mut query = full_text_query
                    .get_item("query")?
                    .ok_or_else(|| PyKeyError::new_err("query must be specified"))?
                    .to_string();
                let columns = if let Some(columns) = full_text_query.get_item("columns")? {
                    if columns.is_none() {
                        None
                    } else {
                        Some(
                            columns
                                .cast::<PyList>()?
                                .iter()
                                .map(|c| c.extract::<String>())
                                .collect::<PyResult<Vec<String>>>()?,
                        )
                    }
                } else {
                    None
                };

                let is_phrase = query.len() >= 2 && query.starts_with('"') && query.ends_with('"');
                let is_multi_match = columns.as_ref().map(|cols| cols.len() > 1).unwrap_or(false);

                if is_phrase {
                    // Remove the surrounding quotes for phrase queries
                    query = query[1..query.len() - 1].to_string();
                }

                let query: FtsQuery = match (is_phrase, is_multi_match) {
                    (false, _) => MatchQuery::new(query).into(),
                    (true, false) => PhraseQuery::new(query).into(),
                    (true, true) => {
                        return Err(PyValueError::new_err(
                            "Phrase queries cannot be used with multiple columns.",
                        ));
                    }
                };
                let mut query = FullTextSearchQuery::new_query(query);
                if let Some(cols) = columns {
                    query = query.with_columns(&cols).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to set full text search columns: {}",
                            e
                        ))
                    })?;
                }
                query
            } else if let Ok(query) = full_text_query.cast::<PyFullTextQuery>() {
                let query = query.borrow();
                FullTextSearchQuery::new_query(query.inner.clone())
            } else {
                return Err(PyValueError::new_err(
                    "query must be a string or a Query object",
                ));
            };

            scanner
                .full_text_search(fts_query)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(f) = substrait_filter {
            scanner.filter_substrait(f.as_slice()).infer_error()?;
        }
        if let Some(prefilter) = prefilter {
            scanner.prefilter(prefilter);
        }

        scanner
            .limit(limit, offset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        if let Some(batch_size) = batch_size {
            scanner.batch_size(batch_size);
        }
        if let Some(batch_size_bytes) = batch_size_bytes {
            scanner.batch_size_bytes(batch_size_bytes);
        }
        if let Some(io_buffer_size) = io_buffer_size {
            scanner.io_buffer_size(io_buffer_size);
        }
        if let Some(batch_readahead) = batch_readahead {
            scanner.batch_readahead(batch_readahead);
        }

        if let Some(fragment_readahead) = fragment_readahead {
            scanner.fragment_readahead(fragment_readahead);
        }

        scanner.scan_in_order(scan_in_order.unwrap_or(true));

        if let Some(use_stats) = use_stats {
            scanner.use_stats(use_stats);
        }

        if let Some(true) = fast_search {
            scanner.fast_search();
        }

        if let Some(true) = include_deleted_rows {
            scanner.include_deleted_rows();
        }

        if let Some(fragments) = fragments {
            let fragments = fragments
                .into_iter()
                .map(|f| {
                    let file_fragment = LanceFileFragment::from(f);
                    file_fragment.into()
                })
                .collect();
            scanner.with_fragments(fragments);
        }

        if let Some(index_segments) = index_segments {
            let index_segments = index_segments
                .into_iter()
                .map(|segment| {
                    Uuid::parse_str(&segment).map_err(|err| {
                        PyValueError::new_err(format!(
                            "invalid index segment uuid '{segment}': {err}"
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            scanner
                .with_index_segments(index_segments)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        if let Some(scan_stats_callback) = scan_stats_callback {
            let callback = Self::make_scan_stats_callback(scan_stats_callback.clone())?;
            scanner.scan_stats_callback(callback);
        }

        if let Some(late_materialization) = late_materialization {
            if let Ok(style_as_bool) = late_materialization.extract::<bool>() {
                if style_as_bool {
                    scanner.materialization_style(MaterializationStyle::AllLate);
                } else {
                    scanner.materialization_style(MaterializationStyle::AllEarly);
                }
            } else if let Ok(columns) = late_materialization.extract::<Vec<String>>() {
                scanner.materialization_style(
                    MaterializationStyle::all_early_except(&columns, self_.ds.schema())
                        .infer_error()?,
                );
            } else {
                return Err(PyValueError::new_err(
                    "late_materialization must be a bool or a list of strings",
                ));
            }
        }

        if let Some(blob_handling) = blob_handling {
            let handling = if let Ok(handling) = blob_handling.extract::<String>() {
                match handling.as_str() {
                    "all_binary" => BlobHandling::AllBinary,
                    "blobs_descriptions" => BlobHandling::BlobsDescriptions,
                    "all_descriptions" => BlobHandling::AllDescriptions,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid blob_handling: {other}. Expected one of: all_binary, blobs_descriptions, all_descriptions"
                        )));
                    }
                }
            } else {
                return Err(PyTypeError::new_err("blob_handling must be a str"));
            };

            scanner.blob_handling(handling);
        }

        if let Some(use_scalar_index) = use_scalar_index {
            scanner.use_scalar_index(use_scalar_index);
        }

        if let Some(strict_batch_size) = strict_batch_size {
            scanner.strict_batch_size(strict_batch_size);
        }

        if let Some(nearest) = nearest {
            let default_k: usize = limit.unwrap_or(10) as usize;
            let (
                column,
                q,
                k,
                minimum_nprobes,
                maximum_nprobes,
                metric_type,
                refine_factor,
                use_index,
                ef,
                query_parallelism,
                approx_mode,
            ) = vector_query_params_from_dict(nearest, default_k)?;

            let (_, element_type) = get_vector_type(self_.ds.schema(), &column)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let scanner = match element_type {
                DataType::UInt8
                    if !matches!(
                        q.data_type(),
                        DataType::List(_) | DataType::FixedSizeList(_, _)
                    ) =>
                {
                    let q = arrow::compute::cast(&q, &DataType::UInt8).map_err(|e| {
                        PyValueError::new_err(format!("Failed to cast q to binary vector: {}", e))
                    })?;
                    let q = q.as_primitive::<UInt8Type>();
                    scanner.nearest(&column, q, k)
                }
                _ => scanner.nearest(&column, &q, k),
            };
            let distance_range: Option<(Option<f32>, Option<f32>)> =
                if let Some(dr) = nearest.get_item("distance_range")? {
                    if dr.is_none() {
                        None
                    } else {
                        let tuple = dr
                            .cast::<PyTuple>()
                            .map_err(|err| PyValueError::new_err(err.to_string()))?;
                        if tuple.len() != 2 {
                            return Err(PyValueError::new_err(
                                "distance_range must be a tuple of (lower_bound, upper_bound)",
                            ));
                        }
                        let lower_any = tuple.get_item(0)?;
                        let lower = if lower_any.is_none() {
                            None
                        } else {
                            Some(lower_any.extract()?)
                        };
                        let upper_any = tuple.get_item(1)?;
                        let upper = if upper_any.is_none() {
                            None
                        } else {
                            Some(upper_any.extract()?)
                        };
                        Some((lower, upper))
                    }
                } else {
                    None
                };

            scanner
                .map(|s| {
                    let mut s = s.minimum_nprobes(minimum_nprobes);
                    if let Some(maximum_nprobes) = maximum_nprobes {
                        s = s.maximum_nprobes(maximum_nprobes);
                    }
                    if let Some(factor) = refine_factor {
                        s = s.refine(factor);
                    }
                    if let Some(m) = metric_type {
                        s = s.distance_metric(m);
                    }
                    if let Some(ef) = ef {
                        s = s.ef(ef);
                    }
                    s = s.query_parallelism(query_parallelism);
                    s = s.approx_mode(approx_mode);
                    s.use_index(use_index);
                    if let Some((lower, upper)) = distance_range {
                        s.distance_range(lower, upper);
                    }
                    s
                })
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(orderings) = order_by {
            scanner
                .order_by(Some(orderings.into_iter().map(|o| o.0).collect()))
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(aggregate_bytes) = substrait_aggregate {
            scanner
                .aggregate(AggregateExpr::substrait(aggregate_bytes))
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        let scan = Arc::new(scanner);
        Ok(Scanner::new(scan))
    }

    #[pyo3(signature=(filter=None))]
    fn count_rows(&self, filter: Option<String>) -> PyResult<usize> {
        rt().block_on(None, self.ds.count_rows(filter))?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    #[pyo3(signature=(row_indices, columns = None, columns_with_transform = None))]
    fn take<'py>(
        self_: PyRef<'py, Self>,
        row_indices: Vec<u64>,
        columns: Option<Vec<String>>,
        columns_with_transform: Option<Vec<(String, String)>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let projection = match (columns, columns_with_transform) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Cannot specify both columns and columns_with_transform",
                ));
            }
            (Some(columns), None) => {
                Ok(ProjectionRequest::from_columns(columns, self_.ds.schema()))
            }
            (None, Some(sql_exprs)) => Ok(ProjectionRequest::from_sql(sql_exprs)),
            (None, None) => Ok(ProjectionRequest::from_schema(self_.ds.schema().clone())),
        }
        .infer_error()?;
        let batch = rt()
            .block_on(Some(self_.py()), self_.ds.take(&row_indices, projection))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        batch.to_pyarrow(self_.py())
    }

    #[pyo3(signature=(row_indices, columns = None, columns_with_transform = None))]
    fn take_rows<'py>(
        self_: PyRef<'py, Self>,
        row_indices: Vec<u64>,
        columns: Option<Vec<String>>,
        columns_with_transform: Option<Vec<(String, String)>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let projection = match (columns, columns_with_transform) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Cannot specify both columns and columns_with_transform",
                ));
            }
            (Some(columns), None) => {
                Ok(ProjectionRequest::from_columns(columns, self_.ds.schema()))
            }
            (None, Some(sql_exprs)) => Ok(ProjectionRequest::from_sql(sql_exprs)),
            (None, None) => Ok(ProjectionRequest::from_schema(self_.ds.schema().clone())),
        }
        .infer_error()?;

        let batch = rt()
            .block_on(
                Some(self_.py()),
                self_.ds.take_rows(&row_indices, projection),
            )?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        batch.to_pyarrow(self_.py())
    }

    fn take_blobs(
        self_: PyRef<'_, Self>,
        row_ids: Vec<u64>,
        blob_column: &str,
    ) -> PyResult<Vec<LanceBlobFile>> {
        let blobs = rt()
            .block_on(Some(self_.py()), self_.ds.take_blobs(&row_ids, blob_column))?
            .infer_error()?;
        Ok(blobs.into_iter().map(LanceBlobFile::from).collect())
    }

    fn take_blobs_by_addresses(
        self_: PyRef<'_, Self>,
        row_addresses: Vec<u64>,
        blob_column: &str,
    ) -> PyResult<Vec<LanceBlobFile>> {
        let blobs = rt()
            .block_on(
                Some(self_.py()),
                self_
                    .ds
                    .take_blobs_by_addresses(&row_addresses, blob_column),
            )?
            .infer_error()?;
        Ok(blobs.into_iter().map(LanceBlobFile::from).collect())
    }

    fn take_blobs_by_indices(
        self_: PyRef<'_, Self>,
        row_indices: Vec<u64>,
        blob_column: &str,
    ) -> PyResult<Vec<LanceBlobFile>> {
        let blobs = rt()
            .block_on(
                Some(self_.py()),
                self_.ds.take_blobs_by_indices(&row_indices, blob_column),
            )?
            .infer_error()?;
        Ok(blobs.into_iter().map(LanceBlobFile::from).collect())
    }

    #[pyo3(signature=(
        row_ids,
        blob_column,
        io_buffer_size=None,
        preserve_order=None
    ))]
    fn read_blobs(
        self_: PyRef<'_, Self>,
        row_ids: Vec<u64>,
        blob_column: &str,
        io_buffer_size: Option<u64>,
        preserve_order: Option<bool>,
    ) -> PyResult<Vec<(u64, Py<PyBytes>)>> {
        let builder = configure_read_blobs_builder(
            self_
                .ds
                .read_blobs(blob_column)
                .infer_error()?
                .with_row_ids(row_ids),
            io_buffer_size,
            preserve_order,
        );
        let blobs = rt()
            .block_on(Some(self_.py()), builder.execute())?
            .infer_error()?;
        Ok(read_blobs_to_python(self_.py(), blobs))
    }

    #[pyo3(signature=(
        row_addresses,
        blob_column,
        io_buffer_size=None,
        preserve_order=None
    ))]
    fn read_blobs_by_addresses(
        self_: PyRef<'_, Self>,
        row_addresses: Vec<u64>,
        blob_column: &str,
        io_buffer_size: Option<u64>,
        preserve_order: Option<bool>,
    ) -> PyResult<Vec<(u64, Py<PyBytes>)>> {
        let builder = configure_read_blobs_builder(
            self_
                .ds
                .read_blobs(blob_column)
                .infer_error()?
                .with_row_addresses(row_addresses),
            io_buffer_size,
            preserve_order,
        );
        let blobs = rt()
            .block_on(Some(self_.py()), builder.execute())?
            .infer_error()?;
        Ok(read_blobs_to_python(self_.py(), blobs))
    }

    #[pyo3(signature=(
        row_indices,
        blob_column,
        io_buffer_size=None,
        preserve_order=None
    ))]
    fn read_blobs_by_indices(
        self_: PyRef<'_, Self>,
        row_indices: Vec<u64>,
        blob_column: &str,
        io_buffer_size: Option<u64>,
        preserve_order: Option<bool>,
    ) -> PyResult<Vec<(u64, Py<PyBytes>)>> {
        let builder = configure_read_blobs_builder(
            self_
                .ds
                .read_blobs(blob_column)
                .infer_error()?
                .with_row_indices(row_indices),
            io_buffer_size,
            preserve_order,
        );
        let blobs = rt()
            .block_on(Some(self_.py()), builder.execute())?
            .infer_error()?;
        Ok(read_blobs_to_python(self_.py(), blobs))
    }

    #[pyo3(signature = (row_slices, columns = None, batch_readahead = 10))]
    fn take_scan(
        &self,
        py: Python<'_>,
        row_slices: Py<PyAny>,
        columns: Option<Vec<String>>,
        batch_readahead: usize,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let projection = if let Some(columns) = columns {
            Arc::new(
                self.ds
                    .schema()
                    .project_preserve_system_columns(&columns)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        } else {
            Arc::new(self.ds.schema().clone())
        };

        // Call into the Python iterable, only holding the GIL as necessary.
        let py_iter = row_slices.call_method0(py, "__iter__")?;
        let slice_iter = std::iter::from_fn(move || {
            Python::attach(|py| {
                match py_iter
                    .call_method0(py, "__next__")
                    .and_then(|range| range.extract::<(u64, u64)>(py))
                {
                    Ok((start, end)) => Some(Ok(start..end)),
                    Err(err) if err.is_instance_of::<PyStopIteration>(py) => None,
                    Err(err) => Some(Err(lance::Error::invalid_input_source(Box::new(err)))),
                }
            })
        });

        let slice_stream = futures::stream::iter(slice_iter).boxed();

        let stream = self.ds.take_scan(slice_stream, projection, batch_readahead);

        Ok(PyArrowType(Box::new(LanceReader::from_stream(stream))))
    }

    fn alter_columns(&mut self, alterations: &Bound<'_, PyList>) -> PyResult<()> {
        let alterations = alterations
            .iter()
            .map(|obj| {
                let obj = obj.cast::<PyDict>()?;
                let path: String = obj
                    .get_item("path")?
                    .ok_or_else(|| PyValueError::new_err("path is required"))?
                    .extract()?;
                let name: Option<String> =
                    obj.get_item("name")?.map(|n| n.extract()).transpose()?;
                let nullable: Option<bool> =
                    obj.get_item("nullable")?.map(|n| n.extract()).transpose()?;
                let data_type: Option<PyArrowType<DataType>> = obj
                    .get_item("data_type")?
                    .map(|n| n.extract())
                    .transpose()?;

                for key in obj.keys().iter().map(|k| k.extract::<String>()) {
                    let k = key?;
                    if k != "path" && k != "name" && k != "nullable" && k != "data_type" {
                        return Err(PyValueError::new_err(format!(
                            "Unknown key: {}. Valid keys are name, nullable, and data_type.",
                            k
                        )));
                    }
                }

                if name.is_none() && nullable.is_none() && data_type.is_none() {
                    return Err(PyValueError::new_err(
                        "At least one of name, nullable, or data_type must be specified",
                    ));
                }

                let mut alteration = ColumnAlteration::new(path);
                if let Some(name) = name {
                    alteration = alteration.rename(name);
                }
                if let Some(nullable) = nullable {
                    alteration = alteration.set_nullable(nullable);
                }
                if let Some(data_type) = data_type {
                    alteration = alteration.cast_to(data_type.0);
                }
                Ok(alteration)
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut new_self = self.ds.as_ref().clone();
        new_self = rt()
            .spawn(None, async move {
                new_self.alter_columns(&alterations).await.map(|_| new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn merge(
        &mut self,
        reader: PyArrowType<ArrowArrayStreamReader>,
        left_on: String,
        right_on: String,
    ) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let new_self = rt()
            .spawn(None, async move {
                new_self
                    .merge(reader.0, &left_on, &right_on)
                    .await
                    .map(|_| new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature=(predicate, conflict_retries=None, retry_timeout=None))]
    fn delete(
        &mut self,
        py: Python<'_>,
        predicate: String,
        conflict_retries: Option<u32>,
        retry_timeout: Option<std::time::Duration>,
    ) -> PyResult<Py<PyAny>> {
        let mut builder = DeleteBuilder::new(self.ds.clone(), predicate);

        if let Some(retries) = conflict_retries {
            builder = builder.conflict_retries(retries);
        }

        if let Some(timeout) = retry_timeout {
            builder = builder.retry_timeout(timeout);
        }

        let result = rt()
            .block_on(None, builder.execute())?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = result.new_dataset;
        let dict = PyDict::new(py);
        dict.set_item("num_deleted_rows", result.num_deleted_rows)?;
        Ok(dict.into())
    }

    #[pyo3(signature=(updates, predicate=None, conflict_retries=None, retry_timeout=None))]
    fn update(
        &mut self,
        updates: &Bound<'_, PyDict>,
        predicate: Option<&str>,
        conflict_retries: Option<u32>,
        retry_timeout: Option<std::time::Duration>,
    ) -> PyResult<Py<PyAny>> {
        let mut builder = UpdateBuilder::new(self.ds.clone());
        if let Some(predicate) = predicate {
            builder = builder
                .update_where(predicate)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        if let Some(retries) = conflict_retries {
            builder = builder.conflict_retries(retries);
        }

        if let Some(timeout) = retry_timeout {
            builder = builder.retry_timeout(timeout);
        }

        for (key, value) in updates {
            let column: PyBackedStr = key.cast::<PyString>()?.clone().try_into()?;
            let expr: PyBackedStr = value.cast::<PyString>()?.clone().try_into()?;

            builder = builder
                .set(column, &expr)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        let operation = builder
            .build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let new_self = rt()
            .block_on(None, operation.execute())?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        self.ds = new_self.new_dataset;
        let update_dict = PyDict::new(updates.py());
        let num_rows_updated = new_self.rows_updated;
        update_dict.set_item("num_rows_updated", num_rows_updated)?;
        Ok(update_dict.into())
    }

    fn count_deleted_rows(&self) -> PyResult<usize> {
        rt().block_on(None, self.ds.count_deleted_rows())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    #[pyo3(signature=(new_bases, transaction_properties=None))]
    fn add_bases(
        &mut self,
        new_bases: Vec<DatasetBasePath>,
        transaction_properties: Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        use lance_table::format::BasePath;
        let rust_bases: Vec<BasePath> = new_bases.into_iter().map(BasePath::from).collect();
        let new_dataset = rt()
            .block_on(None, self.ds.add_bases(rust_bases, transaction_properties))?
            .map_err(|err| match err {
                lance::Error::InvalidInput { .. } => PyValueError::new_err(err.to_string()),
                _ => PyIOError::new_err(err.to_string()),
            })?;

        self.ds = Arc::new(new_dataset);
        Ok(())
    }

    fn versions(self_: PyRef<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        let py = self_.py();
        let versions = self_.list_versions()?;
        let pyvers: Vec<Py<PyAny>> = versions
            .iter()
            .map(|v| {
                let dict = PyDict::new(py);
                dict.set_item("version", v.version).unwrap();
                dict.set_item(
                    "timestamp",
                    v.timestamp.timestamp_nanos_opt().unwrap_or_default(),
                )
                .unwrap();
                let tup: Vec<(&String, &String)> = v.metadata.iter().collect();
                dict.set_item("metadata", tup.into_py_dict(py)?).unwrap();
                dict.into_py_any(py)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(pyvers)
    }

    /// Fetches the currently checked out version of the dataset.
    fn version(&self) -> PyResult<u64> {
        Ok(self.ds.version().version)
    }

    fn latest_version(self_: PyRef<'_, Self>) -> PyResult<u64> {
        rt().block_on(Some(self_.py()), self_.ds.latest_version_id())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    /// Get the initial storage options used to open this dataset.
    ///
    /// This returns the options that were provided when the dataset was opened,
    /// without any refresh from the provider. Returns None if no storage options
    /// were provided.
    fn initial_storage_options(&self) -> Option<HashMap<String, String>> {
        self.ds.initial_storage_options().cloned()
    }

    /// Get the latest storage options, potentially refreshed from the provider.
    ///
    /// If a storage options provider was configured and credentials are expiring,
    /// this will refresh them. Returns the current valid storage options, or None
    /// if no storage options accessor is configured.
    fn latest_storage_options(self_: PyRef<'_, Self>) -> PyResult<Option<HashMap<String, String>>> {
        let result = rt()
            .block_on(Some(self_.py()), self_.ds.latest_storage_options())?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(result.map(|opts| opts.0))
    }

    /// Get the storage options accessor for this dataset.
    ///
    /// The accessor bundles static storage options and optional dynamic provider,
    /// handling caching and refresh logic internally.
    fn storage_options_accessor(&self) -> Option<PyStorageOptionsAccessor> {
        self.ds
            .storage_options_accessor()
            .map(PyStorageOptionsAccessor::new)
    }

    fn checkout_version(&self, version: Bound<PyAny>) -> PyResult<Self> {
        let reference = self.transform_ref(Some(version))?;
        self._checkout_version(reference)
    }

    /// Restore the current version
    #[pyo3(signature = (target_path, reference, storage_options=None))]
    fn shallow_clone(
        &mut self,
        py: Python,
        target_path: String,
        reference: Option<Bound<PyAny>>,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        // Perform a shallow clone of the dataset into the target path.
        // `version` can be a version number or a tag name.
        // `storage_options` will be forwarded to the object store params for the new dataset.
        let store_params = storage_options.as_ref().map(|opts| ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_static_options(opts.clone()),
            )),
            ..Default::default()
        });

        // Use a mutable clone of the inner dataset for operations that require &mut self
        let mut new_self = self.ds.as_ref().clone();
        let reference = self.transform_ref(reference)?;

        let ds = rt()
            .block_on(
                Some(py),
                new_self.shallow_clone(&target_path, reference, store_params),
            )?
            .map_err(|err: Error| PyIOError::new_err(err.to_string()))?;

        let uri = ds.uri().to_string();
        Ok(Self {
            ds: Arc::new(ds),
            uri,
        })
    }

    fn restore(&mut self) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        rt().block_on(None, new_self.restore())?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    /// Truncate the dataset by deleting all rows. The schema is preserved and a new version is created.
    fn truncate_table(&mut self) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        rt().block_on(None, new_self.truncate_table())?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    /// Cleanup old versions from the dataset
    #[pyo3(signature = (older_than_micros = None, retain_versions = None, delete_unverified = None, error_if_tagged_old_versions = None, delete_rate_limit = None))]
    fn cleanup_old_versions(
        &self,
        older_than_micros: Option<i64>,
        retain_versions: Option<usize>,
        delete_unverified: Option<bool>,
        error_if_tagged_old_versions: Option<bool>,
        delete_rate_limit: Option<u64>,
    ) -> PyResult<CleanupStats> {
        let cleanup_stats = rt()
            .block_on(None, async {
                let mut builder = CleanupPolicyBuilder::default();
                if let Some(v) = older_than_micros {
                    let older_than = Duration::microseconds(v);
                    builder = builder.before_timestamp(Utc::now() - older_than);
                }
                if let Some(v) = retain_versions {
                    builder = builder.retain_n_versions(self.ds.as_ref(), v).await?;
                }
                if let Some(v) = delete_unverified {
                    builder = builder.delete_unverified(v);
                }
                if let Some(v) = error_if_tagged_old_versions {
                    builder = builder.error_if_tagged_old_versions(v);
                }
                if let Some(v) = delete_rate_limit {
                    builder = builder.delete_rate_limit(v)?;
                }

                self.ds.cleanup_with_policy(builder.build()).await
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        Ok(CleanupStats {
            bytes_removed: cleanup_stats.bytes_removed,
            old_versions: cleanup_stats.old_versions,
            data_files_removed: cleanup_stats.data_files_removed,
            transaction_files_removed: cleanup_stats.transaction_files_removed,
            index_files_removed: cleanup_stats.index_files_removed,
            deletion_files_removed: cleanup_stats.deletion_files_removed,
        })
    }

    fn tags_ordered(self_: PyRef<'_, Self>, order: Option<String>) -> PyResult<Py<PyAny>> {
        let py = self_.py();
        let tags = self_.list_tags_ordered(order.as_deref())?;

        let pylist = PyList::empty(py);

        for (tag_name, tag_content) in tags {
            let dict = PyDict::new(py);
            dict.set_item("branch", tag_content.branch.clone())?;
            dict.set_item("version", tag_content.version)?;
            dict.set_item("created_at", tag_content.created_at)?;
            dict.set_item("updated_at", tag_content.updated_at)?;
            dict.set_item("manifest_size", tag_content.manifest_size)?;
            dict.set_item("metadata", tag_content.metadata.clone())?;

            pylist.append((tag_name.as_str(), dict))?;
        }

        Ok(pylist.unbind().as_any().clone())
    }

    fn tags(self_: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = self_.py();
        let tags = self_.list_tags()?;

        let pytags = PyDict::new(py);
        for (k, v) in tags.iter() {
            let dict = PyDict::new(py);
            dict.set_item("branch", v.branch.clone())?;
            dict.set_item("version", v.version)?;
            dict.set_item("created_at", v.created_at)?;
            dict.set_item("updated_at", v.updated_at)?;
            dict.set_item("manifest_size", v.manifest_size)?;
            dict.set_item("metadata", v.metadata.clone())?;
            pytags.set_item(k, dict.into_py_any(py)?)?;
        }
        pytags.into_py_any(py)
    }

    fn get_version(self_: PyRef<'_, Self>, tag: String) -> PyResult<u64> {
        let inner_result = rt().block_on(None, self_.ds.tags().get_version(&tag))?;

        inner_result.map_err(|err: lance::Error| match err {
            lance::Error::NotFound { .. } => {
                PyValueError::new_err(format!("Tag not found: {}", err))
            }
            lance::Error::RefNotFound { .. } => {
                PyValueError::new_err(format!("Ref not found: {}", err))
            }
            _ => PyIOError::new_err(format!("Storage error: {}", err)),
        })
    }

    fn create_tag(&mut self, tag: String, reference: Option<Bound<PyAny>>) -> PyResult<()> {
        let reference = self.transform_ref(reference)?;
        rt().block_on(
            None,
            self.ds.as_ref().tags().create(tag.as_str(), reference),
        )?
        .map_err(|err| match err {
            Error::NotFound { .. } => PyValueError::new_err(err.to_string()),
            Error::RefConflict { .. } => PyValueError::new_err(err.to_string()),
            Error::VersionNotFound { .. } => PyValueError::new_err(err.to_string()),
            _ => PyIOError::new_err(err.to_string()),
        })?;
        Ok(())
    }

    fn delete_tag(&mut self, tag: String) -> PyResult<()> {
        let new_self = self.ds.as_ref().clone();
        rt().block_on(None, new_self.tags().delete(tag.as_str()))?
            .map_err(|err| match err {
                lance::Error::NotFound { .. } => PyValueError::new_err(err.to_string()),
                lance::Error::RefNotFound { .. } => PyValueError::new_err(err.to_string()),
                _ => PyIOError::new_err(err.to_string()),
            })?;
        Ok(())
    }

    fn update_tag(&self, tag: String, reference: Option<Bound<PyAny>>) -> PyResult<()> {
        let reference = self.transform_ref(reference)?;
        rt().block_on(
            None,
            self.ds.as_ref().tags().update(tag.as_str(), reference),
        )?
        .infer_error()?;
        Ok(())
    }

    fn replace_tag_metadata(&self, tag: String, metadata: HashMap<String, String>) -> PyResult<()> {
        rt().block_on(
            None,
            self.ds.as_ref().tags().replace_metadata(&tag, metadata),
        )?
        .infer_error()?;
        Ok(())
    }

    /// Check out the latest version of the current branch
    fn checkout_latest(&mut self) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        rt().block_on(None, new_self.checkout_latest())?
            .map_err(|err| match err {
                Error::NotFound { .. } => PyValueError::new_err(err.to_string()),
                _ => PyIOError::new_err(err.to_string()),
            })?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    /// Create a branch from a specific version reference (version number or tag)
    #[pyo3(signature = (branch, reference=None, storage_options=None))]
    fn create_branch(
        &mut self,
        branch: String,
        reference: Option<Bound<PyAny>>,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let mut new_self = self.ds.as_ref().clone();
        let reference = self.transform_ref(reference)?;
        let store_params = storage_options.map(|opts| ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(
                lance::io::StorageOptionsAccessor::with_static_options(opts),
            )),
            ..Default::default()
        });
        let created = rt()
            .block_on(
                None,
                new_self.create_branch(branch.as_str(), reference, store_params),
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        // Update self to reflect any metadata changes; return the new branch dataset
        self.ds = Arc::new(new_self);
        Ok(Self {
            uri: created.uri().to_string(),
            ds: Arc::new(created),
        })
    }

    /// Delete a branch
    fn delete_branch(&mut self, branch: String) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        rt().block_on(None, new_self.delete_branch(branch.as_str()))?
            .map_err(|err| match err {
                Error::RefNotFound { .. } => PyValueError::new_err(err.to_string()),
                _ => PyIOError::new_err(err.to_string()),
            })?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    /// List branches as a Python dictionary mapping name -> metadata
    fn branches(self_: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = self_.py();
        let branches = rt()
            .block_on(None, self_.ds.branches().list())?
            .infer_error()?;
        let pybranches = PyDict::new(py);
        for (name, meta) in branches.iter() {
            let dict = PyDict::new(py);
            dict.set_item("parent_branch", meta.parent_branch.clone())?;
            dict.set_item("branch_identifier", meta.identifier.version_mapping.clone())?;
            dict.set_item("parent_version", meta.parent_version)?;
            dict.set_item("create_at", meta.create_at)?;
            dict.set_item("manifest_size", meta.manifest_size)?;
            dict.set_item("metadata", meta.metadata.clone())?;
            pybranches.set_item(name, dict.into_py_any(py)?)?;
        }
        Ok(pybranches.into())
    }

    fn replace_branch_metadata(
        &self,
        branch: String,
        metadata: HashMap<String, String>,
    ) -> PyResult<()> {
        rt().block_on(
            None,
            self.ds
                .as_ref()
                .branches()
                .replace_metadata(&branch, metadata),
        )?
        .infer_error()?;
        Ok(())
    }

    /// List branches ordered by parent_version
    fn branches_ordered(
        &self,
        py: Python<'_>,
        order: Option<&str>,
    ) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let ordering = match order {
            Some("asc") => Some(std::cmp::Ordering::Less),
            Some("desc") => Some(std::cmp::Ordering::Greater),
            Some(invalid) => {
                return Err(PyValueError::new_err(format!(
                    "Invalid sort order '{}'. Valid values are: asc, desc",
                    invalid
                )));
            }
            None => None,
        };
        let ordered = rt()
            .block_on(None, async {
                self.ds.branches().list_ordered(ordering).await
            })?
            .infer_error()?;
        let mut out: Vec<(String, Py<PyAny>)> = Vec::new();
        for (name, meta) in ordered.into_iter() {
            let dict = PyDict::new(py);
            dict.set_item("parent_branch", meta.parent_branch.clone())?;
            dict.set_item("branch_identifier", meta.identifier.version_mapping.clone())?;
            dict.set_item("parent_version", meta.parent_version)?;
            dict.set_item("create_at", meta.create_at)?;
            dict.set_item("manifest_size", meta.manifest_size)?;
            dict.set_item("metadata", meta.metadata.clone())?;
            out.push((name, dict.into_py_any(py)?));
        }
        Ok(out)
    }

    #[pyo3(signature = (**kwargs))]
    fn optimize_indices(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let mut options: OptimizeOptions = Default::default();
        if let Some(kwargs) = kwargs {
            if let Some(num_indices_to_merge) = kwargs.get_item("num_indices_to_merge")? {
                options.num_indices_to_merge = num_indices_to_merge.extract()?;
            }
            if let Some(index_names) = kwargs.get_item("index_names")? {
                options.index_names = Some(
                    index_names
                        .extract::<Vec<String>>()
                        .map_err(|err| PyValueError::new_err(err.to_string()))?,
                );
            }
        }
        rt().block_on(
            None,
            new_self
                .optimize_indices(&options)
                .map_err(|err| PyIOError::new_err(err.to_string())),
        )??;

        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (columns, index_type, name = None, replace = None, train = None, storage_options = None, kwargs = None))]
    #[allow(clippy::too_many_arguments)]
    fn create_index(
        &mut self,
        columns: Vec<PyBackedStr>,
        index_type: &str,
        name: Option<String>,
        replace: Option<bool>,
        train: Option<bool>,
        storage_options: Option<HashMap<String, String>>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<PyLance<IndexMetadata>> {
        let columns: Vec<&str> = columns.iter().map(|s| &**s).collect();
        let index_type = index_type.to_uppercase();
        let idx_type = match index_type.as_str() {
            "SCALAR" => IndexType::Scalar,
            "BTREE" => IndexType::Scalar,
            "BITMAP" => IndexType::Bitmap,
            "NGRAM" => IndexType::NGram,
            "ZONEMAP" => IndexType::ZoneMap,
            "BLOOMFILTER" => IndexType::BloomFilter,
            "LABEL_LIST" => IndexType::LabelList,
            "RTREE" => IndexType::RTree,
            "INVERTED" | "FTS" => IndexType::Inverted,
            "FM" => IndexType::Fm,
            "IVF_FLAT" | "IVF_PQ" | "IVF_SQ" | "IVF_RQ" | "IVF_HNSW_FLAT" | "IVF_HNSW_PQ"
            | "IVF_HNSW_SQ" => IndexType::Vector,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Index type '{index_type}' is not supported."
                )));
            }
        };

        log::info!("Creating index: type={}", index_type);
        let params: Box<dyn IndexParams> = match index_type.as_str() {
            "BTREE" => Box::new(ScalarIndexParams {
                index_type: "btree".to_string(),
                params: None,
            }),
            "BITMAP" => Box::new(ScalarIndexParams {
                index_type: "bitmap".to_string(),
                params: None,
            }),
            "NGRAM" => Box::new(ScalarIndexParams {
                index_type: "ngram".to_string(),
                params: None,
            }),
            "ZONEMAP" => Box::new(ScalarIndexParams {
                index_type: "zonemap".to_string(),
                params: None,
            }),
            "LABEL_LIST" => Box::new(ScalarIndexParams {
                index_type: "label_list".to_string(),
                params: None,
            }),
            "BLOOMFILTER" => Box::new(ScalarIndexParams {
                index_type: "bloomfilter".to_string(),
                params: None,
            }),
            "RTREE" => Box::new(ScalarIndexParams {
                index_type: "rtree".to_string(),
                params: None,
            }),
            "FM" => {
                let mut params_json = serde_json::Map::new();
                if let Some(kwargs) = kwargs
                    && let Some(num_segments) = kwargs.get_item("num_segments")?
                {
                    let n: u32 = num_segments.extract()?;
                    params_json.insert(
                        "num_segments".to_string(),
                        serde_json::Value::Number(n.into()),
                    );
                }
                let params = if params_json.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::Object(params_json).to_string())
                };
                Box::new(ScalarIndexParams {
                    index_type: "fm".to_string(),
                    params,
                })
            }
            "SCALAR" => {
                let Some(kwargs) = kwargs else {
                    return Err(PyValueError::new_err(
                        "SCALAR index type specified by no kwargs provided",
                    ));
                };
                let Some(config) = kwargs.get_item("config")? else {
                    return Err(PyValueError::new_err(
                        "SCALAR index type specified by no `config` in kwargs",
                    ));
                };
                let config: PyIndexConfig = config.extract()?;
                Box::new(ScalarIndexParams {
                    index_type: config.index_type.clone(),
                    params: Some(config.config.clone()),
                })
            }
            "INVERTED" | "FTS" => {
                let mut params = InvertedIndexParams::default();
                if let Some(kwargs) = kwargs {
                    if let Some(with_position) = kwargs.get_item("with_position")? {
                        params = params.with_position(with_position.extract()?);
                    }
                    if let Some(base_tokenizer) = kwargs.get_item("base_tokenizer")? {
                        params = params.base_tokenizer(base_tokenizer.extract()?);
                    }
                    if let Some(language) = kwargs.get_item("language")? {
                        let language: PyBackedStr =
                            language.cast::<PyString>()?.clone().try_into()?;
                        params = params.language(&language).map_err(|e| {
                            PyValueError::new_err(format!(
                                "can't set tokenizer language to {}: {:?}",
                                language, e
                            ))
                        })?;
                    }
                    if let Some(max_token_length) = kwargs.get_item("max_token_length")? {
                        params = params.max_token_length(max_token_length.extract()?);
                    }
                    if let Some(lower_case) = kwargs.get_item("lower_case")? {
                        params = params.lower_case(lower_case.extract()?);
                    }
                    if let Some(stem) = kwargs.get_item("stem")? {
                        params = params.stem(stem.extract()?);
                    }
                    if let Some(remove_stop_words) = kwargs.get_item("remove_stop_words")? {
                        params = params.remove_stop_words(remove_stop_words.extract()?);
                    }
                    if let Some(stop_words_file) = kwargs.get_item("custom_stop_words")? {
                        params = params.custom_stop_words(stop_words_file.extract()?);
                    }
                    if let Some(ascii_folding) = kwargs.get_item("ascii_folding")? {
                        params = params.ascii_folding(ascii_folding.extract()?);
                    }
                    if let Some(min_ngram_length) = kwargs.get_item("min_ngram_length")? {
                        params = params.ngram_min_length(min_ngram_length.extract()?);
                    }
                    if let Some(max_ngram_length) = kwargs.get_item("max_ngram_length")? {
                        params = params.ngram_max_length(max_ngram_length.extract()?);
                    }
                    if let Some(prefix_only) = kwargs.get_item("prefix_only")? {
                        params = params.ngram_prefix_only(prefix_only.extract()?);
                    }
                    if let Some(memory_limit) = kwargs.get_item("memory_limit")? {
                        params = params.memory_limit_mb(memory_limit.extract()?);
                    }
                    if let Some(num_workers) = kwargs.get_item("num_workers")? {
                        params = params.num_workers(num_workers.extract()?);
                    }
                }
                Box::new(params)
            }
            _ => {
                let column_type = match self.ds.schema().field(columns[0]) {
                    Some(f) => f.data_type().clone(),
                    None => {
                        return Err(PyValueError::new_err("Column not found in dataset schema."));
                    }
                };
                prepare_vector_index_params(&index_type, &column_type, storage_options, kwargs)?
            }
        };

        let replace = replace.unwrap_or(true);
        let train = train.unwrap_or(true); // Default to true for backward compatibility

        let mut progress_handler = Self::make_index_progress_handler_from_kwargs(kwargs)?;
        let mut new_self = self.ds.as_ref().clone();
        let mut builder = new_self
            .create_index_builder(&columns, idx_type, params.as_ref())
            .replace(replace)
            .train(train);
        if let Some(name) = name {
            builder = builder.name(name);
        }

        // Extract fragment_ids and index_uuid from kwargs
        let fragment_ids: Option<Vec<u32>> = if let Some(kwargs) = kwargs {
            kwargs
                .get_item("fragment_ids")?
                .and_then(|v| if v.is_none() { None } else { Some(v.extract()) })
                .transpose()?
        } else {
            None
        };

        let index_uuid: Option<Uuid> = if let Some(kwargs) = kwargs {
            kwargs
                .get_item("index_uuid")?
                .and_then(|v| {
                    if v.is_none() {
                        None
                    } else {
                        Some(v.extract::<String>())
                    }
                })
                .transpose()?
                .map(|s| {
                    Uuid::parse_str(&s).map_err(|e| {
                        PyValueError::new_err(format!("Invalid UUID string for index_uuid: {e}"))
                    })
                })
                .transpose()?
        } else {
            None
        };

        // Add fragment_ids and index_uuid support
        let has_fragment_ids = fragment_ids.is_some();
        if let Some(fragment_ids) = fragment_ids {
            builder = builder.fragments(fragment_ids);
        }
        if let Some(index_uuid) = index_uuid {
            builder = builder.index_uuid(index_uuid);
        }
        if let Some(progress_handler) = progress_handler.as_ref() {
            builder = builder.progress(progress_handler.progress.clone());
        }

        use std::future::IntoFuture;

        // Use execute_uncommitted if fragment_ids is provided, otherwise use execute
        let index_metadata = if has_fragment_ids {
            // For fragment-level indexing, use execute_uncommitted
            // Note: We don't update self.ds here as the index is not committed
            Self::run_index_future(builder.execute_uncommitted(), progress_handler.as_mut())?
                .infer_error()?
        } else {
            // For regular indexing, use the standard execute path
            let index_metadata =
                Self::run_index_future(builder.into_future(), progress_handler.as_mut())?
                    .infer_error()?;
            self.ds = Arc::new(new_self);
            index_metadata
        };

        Ok(PyLance(index_metadata))
    }

    fn merge_existing_index_segments(
        &self,
        segments: Vec<PyLance<IndexMetadata>>,
    ) -> PyResult<PyLance<IndexMetadata>> {
        let merged = rt()
            .block_on(
                None,
                self.ds
                    .merge_existing_index_segments(segments.into_iter().map(|s| s.0).collect()),
            )?
            .infer_error()?;
        Ok(PyLance(merged))
    }

    fn commit_existing_index_segments(
        &mut self,
        index_name: &str,
        column: &str,
        segments: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let segments = extract_index_segments(segments)?;
        rt().block_on(
            None,
            new_self.commit_existing_index_segments(index_name, column, segments),
        )?
        .infer_error()?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn drop_index(&mut self, name: &str) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        rt().block_on(None, new_self.drop_index(name))?
            .infer_error()?;
        self.ds = Arc::new(new_self);

        Ok(())
    }

    #[pyo3(signature = (name, *, with_position = false))]
    fn prewarm_index(&self, name: &str, with_position: bool) -> PyResult<()> {
        rt().block_on(None, async {
            if with_position {
                self.ds
                    .prewarm_index_with_options(
                        name,
                        &PrewarmOptions::Fts(FtsPrewarmOptions::new().with_position(true)),
                    )
                    .await
            } else {
                self.ds.prewarm_index(name).await
            }
        })?
        .infer_error()
    }

    #[pyo3(signature = (index_uuid, index_type, batch_readhead=None, progress_callback=None))]
    fn merge_index_metadata(
        &self,
        index_uuid: &str,
        index_type: &str,
        batch_readhead: Option<usize>,
        progress_callback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let parsed_uuid = Uuid::parse_str(index_uuid).map_err(|e| {
            PyValueError::new_err(format!("Invalid UUID string for index_uuid: {e}"))
        })?;
        let mut progress_handler =
            Self::make_index_progress_handler_from_callback(progress_callback)?;
        let progress: Arc<dyn IndexBuildProgress> = progress_handler
            .as_ref()
            .map(|handler| handler.progress.clone() as Arc<dyn IndexBuildProgress>)
            .unwrap_or_else(|| Arc::new(NoopIndexBuildProgress));

        Self::run_index_future(
            async {
                self.ds
                    .merge_index_metadata(
                        &parsed_uuid,
                        IndexType::try_from(index_type)?,
                        batch_readhead,
                        progress,
                    )
                    .await
            },
            progress_handler.as_mut(),
        )?
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn count_fragments(&self) -> usize {
        self.ds.count_fragments()
    }

    fn num_small_files(&self, max_rows_per_group: usize) -> PyResult<usize> {
        rt().block_on(None, self.ds.num_small_files(max_rows_per_group))
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    fn data_stats(&self) -> PyResult<PyLance<DataStatistics>> {
        rt().block_on(None, self.ds.calculate_data_stats())?
            .infer_error()
            .map(PyLance)
    }

    fn get_fragments(self_: PyRef<'_, Self>) -> PyResult<Vec<FileFragment>> {
        let core_fragments = self_.ds.get_fragments();

        let fragments: Vec<FileFragment> = core_fragments
            .iter()
            .map(|f| FileFragment::new(f.clone()))
            .collect::<Vec<_>>();
        Ok(fragments)
    }

    fn get_fragment(self_: PyRef<'_, Self>, fragment_id: usize) -> PyResult<Option<FileFragment>> {
        if let Some(fragment) = self_.ds.get_fragment(fragment_id) {
            Ok(Some(FileFragment::new(fragment)))
        } else {
            Ok(None)
        }
    }

    fn index_cache_entry_count(&self) -> PyResult<usize> {
        rt().block_on(None, self.ds.index_cache_entry_count())
    }

    fn index_cache_hit_rate(&self) -> PyResult<f32> {
        rt().block_on(None, self.ds.index_cache_hit_rate())
    }

    fn session(&self) -> Session {
        Session::new(self.ds.session())
    }

    /// Get a snapshot of current IO statistics without resetting counters
    fn io_stats_snapshot(&self) -> PyResult<IoStats> {
        let object_store = rt()
            .block_on(None, self.ds.object_store(None))?
            .infer_error()?;
        let stats = object_store.io_stats_snapshot();
        Ok(IoStats::from_lance(stats))
    }

    /// Get incremental IO statistics for this dataset
    fn io_stats_incremental(&self) -> PyResult<IoStats> {
        let object_store = rt()
            .block_on(None, self.ds.object_store(None))?
            .infer_error()?;
        let stats = object_store.io_stats_incremental();
        Ok(IoStats::from_lance(stats))
    }

    #[staticmethod]
    #[pyo3(signature = (dest, storage_options = None, ignore_not_found = None))]
    fn drop(
        dest: String,
        storage_options: Option<HashMap<String, String>>,
        ignore_not_found: Option<bool>,
    ) -> PyResult<()> {
        rt().spawn(None, async move {
            let (object_store, path) =
                object_store_from_uri_or_path(&dest, storage_options).await?;
            let result = object_store.remove_dir_all(path).await;

            match result {
                Ok(_) => Ok(()),
                Err(e) => {
                    let is_not_found = matches!(&e, lance_core::Error::NotFound { .. });

                    if let Some(true) = ignore_not_found {
                        if is_not_found {
                            Ok(())
                        } else {
                            Err(PyIOError::new_err(e.to_string()))
                        }
                    } else {
                        Err(PyIOError::new_err(e.to_string()))
                    }
                }
            }
        })?
    }

    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (dest, operation, read_version = None, commit_lock = None, storage_options = None, enable_v2_manifest_paths = None, detached = None, max_retries = None, commit_message = None, enable_stable_row_ids = None, namespace_client = None, table_id = None, namespace_client_managed_versioning = false, commit_timeout = None))]
    fn commit(
        dest: PyWriteDest,
        operation: PyLance<Operation>,
        read_version: Option<u64>,
        commit_lock: Option<&Bound<'_, PyAny>>,
        storage_options: Option<HashMap<String, String>>,
        enable_v2_manifest_paths: Option<bool>,
        detached: Option<bool>,
        max_retries: Option<u32>,
        commit_message: Option<String>,
        enable_stable_row_ids: Option<bool>,
        namespace_client: Option<&Bound<'_, PyAny>>,
        table_id: Option<Vec<String>>,
        namespace_client_managed_versioning: bool,
        commit_timeout: Option<std::time::Duration>,
    ) -> PyResult<Self> {
        let mut transaction = Transaction::new(read_version.unwrap_or_default(), operation.0, None);

        if let Some(commit_message) = commit_message {
            transaction.transaction_properties = Some(Arc::new(HashMap::from([(
                LANCE_COMMIT_MESSAGE_KEY.to_string(),
                commit_message,
            )])));
        }

        Self::commit_transaction(
            dest,
            PyLance(transaction),
            commit_lock,
            storage_options,
            enable_v2_manifest_paths,
            detached,
            max_retries,
            enable_stable_row_ids,
            namespace_client,
            table_id,
            namespace_client_managed_versioning,
            commit_timeout,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(deprecated)]
    #[staticmethod]
    #[pyo3(signature = (dest, transaction, commit_lock = None, storage_options = None, enable_v2_manifest_paths = None, detached = None, max_retries = None, enable_stable_row_ids = None, namespace_client = None, table_id = None, namespace_client_managed_versioning = false, commit_timeout = None))]
    fn commit_transaction(
        dest: PyWriteDest,
        transaction: PyLance<Transaction>,
        commit_lock: Option<&Bound<'_, PyAny>>,
        storage_options: Option<HashMap<String, String>>,
        enable_v2_manifest_paths: Option<bool>,
        detached: Option<bool>,
        max_retries: Option<u32>,
        enable_stable_row_ids: Option<bool>,
        namespace_client: Option<&Bound<'_, PyAny>>,
        table_id: Option<Vec<String>>,
        namespace_client_managed_versioning: bool,
        commit_timeout: Option<std::time::Duration>,
    ) -> PyResult<Self> {
        let accessor =
            crate::storage_options::create_accessor_from_storage_options(storage_options.clone())?;

        let object_store_params = if accessor.is_some() {
            Some(ObjectStoreParams {
                storage_options_accessor: accessor,
                ..Default::default()
            })
        } else {
            None
        };

        // Create commit_handler: prefer user-provided commit_lock, then namespace client-based handler
        // (only if namespace_client_managed_versioning is true)
        let commit_handler: Option<Arc<dyn CommitHandler>> =
            if let Some(commit_lock) = commit_lock.as_ref() {
                // User provided a commit_lock
                Some(
                    commit_lock
                        .into_py_any(commit_lock.py())
                        .map(|cl| Arc::new(PyCommitLock::new(cl)) as Arc<dyn CommitHandler>)?,
                )
            } else if namespace_client_managed_versioning
                && let (Some(ns_client), Some(tid)) = (namespace_client, table_id)
            {
                // Create ExternalManifestCommitHandler from namespace client and table_id
                // only when namespace manages versioning. The store derives the
                // branch a request targets from the base path it is handed,
                // resolved against the table root.
                let ns_client = extract_namespace_arc(ns_client.py(), ns_client)?;
                let external_store = LanceNamespaceExternalManifestStore::for_table_uri(
                    ns_client,
                    tid,
                    &dest.table_root_uri()?,
                )
                .infer_error()?;
                Some(Arc::new(ExternalManifestCommitHandler {
                    external_manifest_store: Arc::new(external_store),
                }) as Arc<dyn CommitHandler>)
            } else {
                None
            };

        let mut builder = CommitBuilder::new(dest.as_dest())
            .enable_v2_manifest_paths(enable_v2_manifest_paths.unwrap_or(true))
            .with_detached(detached.unwrap_or(false))
            .with_max_retries(max_retries.unwrap_or(20))
            .with_timeout(commit_timeout);

        if let Some(enable) = enable_stable_row_ids {
            builder = builder.use_stable_row_ids(enable);
        }

        if let Some(store_params) = object_store_params {
            builder = builder.with_store_params(store_params);
        }

        if let Some(commit_handler) = commit_handler {
            builder = builder.with_commit_handler(commit_handler);
        }

        let ds = rt()
            .block_on(
                commit_lock.map(|cl| cl.py()),
                builder.execute(transaction.0),
            )?
            .io_or_timeout_error()?;

        let uri = ds.uri().to_string();
        Ok(Self {
            ds: Arc::new(ds),
            uri,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(deprecated)]
    #[staticmethod]
    #[pyo3(signature = (dest, transactions, commit_lock = None, storage_options = None, enable_v2_manifest_paths = None, detached = None, max_retries = None, commit_timeout = None))]
    fn commit_batch(
        dest: PyWriteDest,
        transactions: Vec<PyLance<Transaction>>,
        commit_lock: Option<&Bound<'_, PyAny>>,
        storage_options: Option<HashMap<String, String>>,
        enable_v2_manifest_paths: Option<bool>,
        detached: Option<bool>,
        max_retries: Option<u32>,
        commit_timeout: Option<std::time::Duration>,
    ) -> PyResult<(Self, PyLance<Transaction>)> {
        let accessor =
            crate::storage_options::create_accessor_from_storage_options(storage_options.clone())?;

        let object_store_params = if accessor.is_some() {
            Some(ObjectStoreParams {
                storage_options_accessor: accessor,
                ..Default::default()
            })
        } else {
            None
        };

        let commit_handler = commit_lock
            .map(|commit_lock| {
                commit_lock
                    .into_py_any(commit_lock.py())
                    .map(|cl| Arc::new(PyCommitLock::new(cl)) as Arc<dyn CommitHandler>)
            })
            .transpose()?;

        let mut builder = CommitBuilder::new(dest.as_dest())
            .enable_v2_manifest_paths(enable_v2_manifest_paths.unwrap_or(true))
            .with_detached(detached.unwrap_or(false))
            .with_max_retries(max_retries.unwrap_or(20))
            .with_timeout(commit_timeout);

        if let Some(store_params) = object_store_params {
            builder = builder.with_store_params(store_params);
        }

        if let Some(commit_handler) = commit_handler {
            builder = builder.with_commit_handler(commit_handler);
        }

        let transactions = transactions
            .into_iter()
            .map(|transaction| transaction.0)
            .collect();

        let res = rt()
            .block_on(None, builder.execute_batch(transactions))?
            .io_or_timeout_error()?;
        let uri = res.dataset.uri().to_string();
        let ds = Self {
            ds: Arc::new(res.dataset),
            uri,
        };

        Ok((ds, PyLance(res.merged)))
    }

    fn validate(&self) -> PyResult<()> {
        rt().block_on(None, self.ds.validate())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    fn migrate_manifest_paths_v2(&mut self) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        rt().block_on(None, new_self.migrate_manifest_paths_v2())?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn drop_columns(&mut self, columns: Vec<String>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let columns: Vec<_> = columns.iter().map(|s| s.as_str()).collect();
        rt().block_on(None, new_self.drop_columns(&columns))?
            .map_err(|err| match err {
                lance::Error::InvalidInput { source, .. } => {
                    PyValueError::new_err(source.to_string())
                }
                _ => PyIOError::new_err(err.to_string()),
            })?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (reader, batch_size = None))]
    fn add_columns_from_reader(
        &mut self,
        reader: &Bound<'_, PyAny>,
        batch_size: Option<u32>,
    ) -> PyResult<()> {
        let batches = ArrowArrayStreamReader::from_pyarrow_bound(reader)?;

        let transforms = NewColumnTransform::Reader(Box::new(batches));

        let mut new_self = self.ds.as_ref().clone();
        let new_self = rt()
            .spawn(None, async move {
                new_self.add_columns(transforms, None, batch_size).await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);

        Ok(())
    }

    #[pyo3(signature = (transforms, read_columns = None, batch_size = None))]
    fn add_columns(
        &mut self,
        py: Python<'_>,
        transforms: &Bound<'_, PyAny>,
        read_columns: Option<Vec<String>>,
        batch_size: Option<u32>,
    ) -> PyResult<()> {
        let transforms = transforms_from_python(py, transforms)?;

        let mut new_self = self.ds.as_ref().clone();
        let new_self = rt()
            .spawn(None, async move {
                new_self
                    .add_columns(transforms, read_columns, batch_size)
                    .await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);

        Ok(())
    }

    /// Add NULL columns with only ArrowSchema.
    #[pyo3(signature = (schema))]
    fn add_columns_with_schema(&mut self, schema: PyArrowType<ArrowSchema>) -> PyResult<()> {
        let arrow_schema: &ArrowSchema = &schema.0;
        let transform = NewColumnTransform::AllNulls(Arc::new(arrow_schema.clone()));

        let mut new_self = self.ds.as_ref().clone();
        let new_self = rt()
            .spawn(None, async move {
                new_self.add_columns(transform, None, None).await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (index_name,partition_id, with_vector=false))]
    fn read_index_partition(
        &self,
        index_name: String,
        partition_id: usize,
        with_vector: bool,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let stream = rt()
            .block_on(
                None,
                self.ds
                    .read_index_partition(&index_name, partition_id, with_vector),
            )?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let reader = Box::new(LanceReader::from_stream(DatasetRecordBatchStream::new(
            stream,
        )));
        Ok(PyArrowType(reader))
    }

    #[pyo3(signature = (*, min_version=None, progress=None))]
    fn tracked_files(
        &self,
        min_version: Option<u64>,
        progress: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        use lance::dataset::files::{TrackedFilesOptions, TrackedFilesProgress};

        let progress_cb: Option<Box<dyn Fn(TrackedFilesProgress) + Send + Sync>> =
            if let Some(cb) = progress {
                if !cb.is_callable() {
                    return Err(PyValueError::new_err("progress must be callable"));
                }
                let cb = cb.clone().unbind();
                Some(Box::new(move |p: TrackedFilesProgress| {
                    Python::attach(|py| {
                        let total: Option<usize> = p.manifests_total;
                        match cb.call1(py, (p.manifests_processed, total)) {
                            Ok(_) => (),
                            Err(e) => {
                                log::error!("Error in tracked_files progress callback: {}", e);
                            }
                        }
                    });
                }))
            } else {
                None
            };

        let options = TrackedFilesOptions {
            min_version,
            progress: progress_cb,
        };
        let stream = rt().block_on(None, self.ds.tracked_files_with_options(options))?;
        let reader = Box::new(LanceReader::from_stream(DatasetRecordBatchStream::new(
            stream,
        )));
        Ok(PyArrowType(reader))
    }

    fn all_files(&self) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let stream = rt().block_on(None, self.ds.all_files())?;
        let reader = Box::new(LanceReader::from_stream(DatasetRecordBatchStream::new(
            stream,
        )));
        Ok(PyArrowType(reader))
    }

    #[pyo3(signature = (keys))]
    fn delete_config_keys(&mut self, keys: Vec<String>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let delete_updates: HashMap<String, Option<String>> =
            keys.into_iter().map(|k| (k, None)).collect();
        rt().block_on(None, async { new_self.update_config(delete_updates).await })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = ())]
    fn config(&self) -> HashMap<String, String> {
        self.ds.config().clone()
    }

    // Unified metadata APIs

    #[pyo3(signature = ())]
    fn get_table_metadata(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let new_self = self.ds.as_ref().clone();

        let table_metadata = new_self.metadata().clone();

        self.ds = Arc::new(new_self);

        let dict = PyDict::new(py);
        for (k, v) in table_metadata {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    #[pyo3(signature = ())]
    fn metadata(&self) -> HashMap<String, String> {
        self.ds.metadata().clone()
    }

    #[pyo3(signature = (values, *, replace = false))]
    fn update_metadata(
        &mut self,
        values: HashMap<String, Option<String>>,
        replace: bool,
    ) -> PyResult<HashMap<String, String>> {
        let mut new_self = self.ds.as_ref().clone();
        let result = if replace {
            rt().block_on(None, async {
                new_self.update_metadata(values).replace().await
            })?
        } else {
            rt().block_on(None, async { new_self.update_metadata(values).await })?
        }
        .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(result)
    }

    #[pyo3(signature = (values, *, replace = false))]
    fn update_config(
        &mut self,
        values: HashMap<String, Option<String>>,
        replace: bool,
    ) -> PyResult<HashMap<String, String>> {
        let mut new_self = self.ds.as_ref().clone();
        let new_config = if replace {
            rt().block_on(None, async {
                new_self.update_config(values).replace().await
            })?
        } else {
            rt().block_on(None, async { new_self.update_config(values).await })?
        }
        .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(new_config)
    }

    #[pyo3(signature = (values, *, replace = false))]
    fn update_schema_metadata(
        &mut self,
        values: HashMap<String, Option<String>>,
        replace: bool,
    ) -> PyResult<HashMap<String, String>> {
        let mut new_self = self.ds.as_ref().clone();
        let result = if replace {
            rt().block_on(None, async {
                new_self.update_schema_metadata(values).replace().await
            })?
        } else {
            rt().block_on(None, async {
                new_self.update_schema_metadata(values).await
            })?
        }
        .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(result)
    }

    fn schema_metadata(&self) -> PyResult<HashMap<String, String>> {
        Ok(self.ds.schema().metadata.clone())
    }

    #[pyo3(signature = (field_updates, *, replace = false))]
    fn update_field_metadata(
        &mut self,
        field_updates: HashMap<u32, HashMap<String, Option<String>>>,
        replace: bool,
    ) -> PyResult<()> {
        let ds_clone = self.ds.as_ref().clone();
        let new_self = rt()
            .block_on(None, async move {
                let mut new_self = ds_clone;
                let mut builder = new_self.update_field_metadata();
                for (field_id, metadata_updates) in field_updates {
                    let converted_updates: Vec<(String, Option<String>)> =
                        metadata_updates.into_iter().collect();
                    if replace {
                        builder = builder.replace(field_id as i32, converted_updates)?;
                    } else {
                        builder = builder.update(field_id as i32, converted_updates)?;
                    }
                }
                builder.await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (field_updates, *, replace = false))]
    fn update_field_metadata_by_path(
        &mut self,
        field_updates: HashMap<String, HashMap<String, Option<String>>>,
        replace: bool,
    ) -> PyResult<()> {
        let ds_clone = self.ds.as_ref().clone();
        let new_self = rt()
            .block_on(None, async move {
                let mut new_self = ds_clone;
                let mut builder = new_self.update_field_metadata();
                for (field_path, metadata_updates) in field_updates {
                    let converted_updates: Vec<(String, Option<String>)> =
                        metadata_updates.into_iter().collect();
                    if replace {
                        builder = builder.replace(field_path.as_str(), converted_updates)?;
                    } else {
                        builder = builder.update(field_path.as_str(), converted_updates)?;
                    }
                }
                builder.await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (index_name))]
    fn get_ivf_model(
        &self,
        py: Python<'_>,
        index_name: &str,
    ) -> PyResult<Py<crate::indices::PyIvfModel>> {
        use crate::indices::PyIvfModel;
        let ivf_model = crate::rt().block_on(Some(py), async {
            use lance::index::DatasetIndexInternalExt;
            use lance_index::metrics::NoOpMetricsCollector;

            // Load index metadata and find the requested index
            let idx_metas = self.ds.load_indices().await.infer_error()?;
            let idx_meta = idx_metas
                .iter()
                .find(|idx| idx.name == index_name)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("Index \"{}\" not found", index_name))
                })?;

            if idx_meta.fields.is_empty() {
                return Err(PyValueError::new_err("Index has no fields"));
            }

            let schema = self.ds.schema();
            let field = schema
                .field_by_id(idx_meta.fields[0])
                .ok_or_else(|| PyValueError::new_err("Failed to resolve index field"))?;
            let column_name = &field.name;

            let vindex = self
                .ds
                .open_vector_index(column_name, &idx_meta.uuid, &NoOpMetricsCollector)
                .await
                .infer_error()?;

            Ok::<lance_index::vector::ivf::storage::IvfModel, pyo3::PyErr>(
                vindex.ivf_model().clone(),
            )
        })??;

        Py::new(py, PyIvfModel { inner: ivf_model })
    }

    /// Read the transaction by specific version
    ///
    /// Returns None if the transaction file does not exist.
    #[pyo3(signature = (version))]
    fn read_transaction(&mut self, version: u64) -> PyResult<Option<PyLance<Transaction>>> {
        let new_self = self.ds.as_ref().clone();
        let transaction = rt()
            .block_on(None, new_self.read_transaction_by_version(version))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(transaction.map(PyLance))
    }

    #[pyo3(signature = (recent_transactions=10))]
    fn get_transactions(
        &mut self,
        recent_transactions: usize,
    ) -> PyResult<Vec<Option<PyLance<Transaction>>>> {
        let new_self = self.ds.as_ref().clone();
        let transactions = rt()
            .block_on(None, new_self.get_transactions(recent_transactions))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        Ok(transactions.into_iter().map(|t| t.map(PyLance)).collect())
    }

    #[pyo3(signature=(sql))]
    fn sql(&self, sql: String) -> PyResult<SqlQueryBuilder> {
        let builder = self.ds.sql(&sql);
        Ok(SqlQueryBuilder { builder })
    }

    #[pyo3(signature=())]
    fn describe_indices(&self, py: Python<'_>) -> PyResult<Vec<PyIndexDescription>> {
        let new_self = self.ds.as_ref().clone();
        let indices = rt()
            .block_on(Some(py), new_self.describe_indices(None))?
            .infer_error()?;
        Ok(indices
            .into_iter()
            .map(|desc| PyIndexDescription::new(desc.as_ref(), self.ds.as_ref()))
            .collect())
    }

    /// Create a delta builder to explore changes between dataset versions.
    #[pyo3(signature=())]
    fn delta(&self) -> PyResult<DatasetDeltaBuilder> {
        let ds = self.ds.as_ref().clone();
        let builder = ds.delta();
        Ok(DatasetDeltaBuilder { builder })
    }

    /// Initialize MemWAL on this dataset.
    ///
    /// Must be called once before any `mem_wal_writer()` calls. Append-only
    /// tables may omit primary-key metadata; primary keys are only required
    /// for primary-key lookup and last-write-wins deduplication workflows.
    ///
    /// At most one sharding mode may be selected: bucket sharding
    /// (`bucket_column` + `num_buckets`), identity sharding (`identity_column`),
    /// or `unsharded`. With none selected, shards are managed manually.
    ///
    /// Any writer-configuration keyword arguments are recorded as the default
    /// `ShardWriter` configuration for the MemWAL index.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        *,
        maintained_indexes=None,
        bucket_column=None,
        num_buckets=None,
        identity_column=None,
        unsharded=false,
        durable_write=None,
        sync_indexed_write=None,
        max_wal_buffer_size=None,
        max_wal_flush_interval_ms=None,
        max_memtable_size=None,
        max_memtable_rows=None,
        max_memtable_batches=None,
        max_unflushed_memtable_bytes=None,
        manifest_scan_batch_size=None,
        async_index_buffer_rows=None,
        async_index_interval_ms=None,
        backpressure_log_interval_ms=None,
        stats_log_interval_ms=None,
        hnsw_params=None,
    ))]
    fn initialize_mem_wal(
        &mut self,
        py: Python<'_>,
        maintained_indexes: Option<Vec<String>>,
        bucket_column: Option<String>,
        num_buckets: Option<u32>,
        identity_column: Option<String>,
        unsharded: bool,
        durable_write: Option<bool>,
        sync_indexed_write: Option<bool>,
        max_wal_buffer_size: Option<usize>,
        max_wal_flush_interval_ms: Option<u64>,
        max_memtable_size: Option<usize>,
        max_memtable_rows: Option<usize>,
        max_memtable_batches: Option<usize>,
        max_unflushed_memtable_bytes: Option<usize>,
        manifest_scan_batch_size: Option<usize>,
        async_index_buffer_rows: Option<usize>,
        async_index_interval_ms: Option<u64>,
        backpressure_log_interval_ms: Option<u64>,
        stats_log_interval_ms: Option<u64>,
        hnsw_params: Option<HashMap<String, HashMap<String, u32>>>,
    ) -> PyResult<()> {
        use lance::dataset::mem_wal::DatasetMemWalExt;

        let bucket = match (bucket_column.as_deref(), num_buckets) {
            (Some(_), Some(_)) => true,
            (None, None) => false,
            _ => {
                return Err(PyValueError::new_err(
                    "bucket sharding requires both `bucket_column` and `num_buckets`",
                ));
            }
        };
        let modes = [bucket, identity_column.is_some(), unsharded]
            .into_iter()
            .filter(|&m| m)
            .count();
        if modes > 1 {
            return Err(PyValueError::new_err(
                "at most one of bucket sharding, `identity_column`, or `unsharded` may be set",
            ));
        }

        let writer_config = writer_config_from_kwargs(
            durable_write,
            sync_indexed_write,
            max_wal_buffer_size,
            max_wal_flush_interval_ms,
            max_memtable_size,
            max_memtable_rows,
            max_memtable_batches,
            max_unflushed_memtable_bytes,
            manifest_scan_batch_size,
            async_index_buffer_rows,
            async_index_interval_ms,
            backpressure_log_interval_ms,
            stats_log_interval_ms,
            hnsw_params,
        )?;
        let maintained_indexes = maintained_indexes.unwrap_or_default();

        let mut ds = Arc::clone(&self.ds);
        let new_ds = rt()
            .block_on(Some(py), async move {
                let dataset = Arc::make_mut(&mut ds);
                let mut builder = dataset.initialize_mem_wal();
                if let (Some(column), Some(n)) = (bucket_column, num_buckets) {
                    builder = builder.bucket_sharding(column, n);
                } else if let Some(column) = identity_column {
                    builder = builder.identity_sharding(column);
                } else if unsharded {
                    builder = builder.unsharded();
                }
                builder = builder.maintained_indexes(maintained_indexes);
                if let Some(config) = writer_config {
                    builder = builder.writer_config_defaults(config);
                }
                builder.execute().await?;
                Ok::<Arc<LanceDataset>, lance_core::Error>(ds)
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.ds = new_ds;
        Ok(())
    }

    /// Return the MemWAL index details for this dataset, or `None` if MemWAL
    /// has not been initialized.
    ///
    /// The returned dict has `num_shards`, `maintained_indexes`,
    /// `writer_config_defaults`, and `sharding_specs`.
    fn mem_wal_index_details<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        use lance::dataset::mem_wal::DatasetMemWalExt;

        let ds = self.ds.clone();
        let details = rt()
            .block_on(Some(py), async move { ds.mem_wal_index_details().await })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let Some(details) = details else {
            return Ok(None);
        };

        let dict = PyDict::new(py);
        dict.set_item("num_shards", details.num_shards)?;
        dict.set_item("maintained_indexes", details.maintained_indexes)?;
        dict.set_item("writer_config_defaults", details.writer_config_defaults)?;

        let specs = PyList::empty(py);
        for spec in &details.sharding_specs {
            let spec_dict = PyDict::new(py);
            spec_dict.set_item("spec_id", spec.spec_id)?;
            let fields = PyList::empty(py);
            for field in &spec.fields {
                let field_dict = PyDict::new(py);
                field_dict.set_item("field_id", &field.field_id)?;
                field_dict.set_item("source_ids", field.source_ids.clone())?;
                field_dict.set_item("transform", field.transform.clone())?;
                field_dict.set_item("expression", field.expression.clone())?;
                field_dict.set_item("result_type", &field.result_type)?;
                field_dict.set_item("parameters", field.parameters.clone())?;
                fields.append(field_dict)?;
            }
            spec_dict.set_item("fields", fields)?;
            specs.append(spec_dict)?;
        }
        dict.set_item("sharding_specs", specs)?;
        Ok(Some(dict))
    }

    /// Get a ShardWriter for the specified shard.
    ///
    /// `initialize_mem_wal()` must be called before using this method.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        shard_id,
        *,
        durable_write=None,
        sync_indexed_write=None,
        max_wal_buffer_size=None,
        max_wal_flush_interval_ms=None,
        max_memtable_size=None,
        max_memtable_rows=None,
        max_memtable_batches=None,
        max_unflushed_memtable_bytes=None,
        manifest_scan_batch_size=None,
        async_index_buffer_rows=None,
        async_index_interval_ms=None,
        backpressure_log_interval_ms=None,
        stats_log_interval_ms=None,
        hnsw_params=None,
    ))]
    fn mem_wal_writer(
        &self,
        py: Python<'_>,
        shard_id: String,
        durable_write: Option<bool>,
        sync_indexed_write: Option<bool>,
        max_wal_buffer_size: Option<usize>,
        max_wal_flush_interval_ms: Option<u64>,
        max_memtable_size: Option<usize>,
        max_memtable_rows: Option<usize>,
        max_memtable_batches: Option<usize>,
        max_unflushed_memtable_bytes: Option<usize>,
        manifest_scan_batch_size: Option<usize>,
        async_index_buffer_rows: Option<usize>,
        async_index_interval_ms: Option<u64>,
        backpressure_log_interval_ms: Option<u64>,
        stats_log_interval_ms: Option<u64>,
        hnsw_params: Option<HashMap<String, HashMap<String, u32>>>,
    ) -> PyResult<crate::mem_wal::PyShardWriter> {
        use lance::dataset::mem_wal::DatasetMemWalExt;

        let uuid = uuid::Uuid::parse_str(&shard_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid shard_id UUID: {}", e)))?;

        let config = writer_config_from_kwargs(
            durable_write,
            sync_indexed_write,
            max_wal_buffer_size,
            max_wal_flush_interval_ms,
            max_memtable_size,
            max_memtable_rows,
            max_memtable_batches,
            max_unflushed_memtable_bytes,
            manifest_scan_batch_size,
            async_index_buffer_rows,
            async_index_interval_ms,
            backpressure_log_interval_ms,
            stats_log_interval_ms,
            hnsw_params,
        )?
        .unwrap_or_default();

        let ds = self.ds.clone();
        let writer = rt()
            .block_on(
                Some(py),
                async move { ds.mem_wal_writer(uuid, config).await },
            )?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(crate::mem_wal::PyShardWriter::new(
            writer,
            uuid,
            self.ds.clone(),
        ))
    }
}

#[pyclass(name = "SqlQuery", module = "_lib", subclass, skip_from_py_object)]
#[derive(Clone)]
pub struct SqlQuery {
    builder: lance::dataset::sql::SqlQueryBuilder,
}

#[pymethods]
impl SqlQuery {
    /// Execute the query and return a list of RecordBatches.
    ///
    /// This is an eager operation that will load all results into memory.
    /// This corresponds to `into_batch_records` in Rust.
    fn to_batch_records<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        use arrow::pyarrow::ToPyArrow;

        let builder = self.builder.clone();
        let batches = rt()
            .block_on(None, async move {
                let query = builder.build().await?;
                query.into_batch_records().await
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))? // Handles tokio::JoinError
            .map_err(|e| PyValueError::new_err(e.to_string()))?; // Handles lance::Error

        batches
            .iter()
            .map(|rb| rb.to_pyarrow(py))
            .collect::<PyResult<Vec<_>>>()
    }

    /// Execute the query and return a RecordBatchReader.
    ///
    /// This is a lazy operation that will stream results.
    fn to_stream_reader<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use crate::reader::LanceReader;
        use arrow::pyarrow::IntoPyArrow;
        use arrow_array::RecordBatchReader;
        use std::pin::Pin;

        let builder = self.builder.clone();
        let fut = Box::pin(async move {
            let query = builder.build().await?;
            let stream = query.into_stream().await?;
            Ok::<Pin<Box<dyn datafusion::execution::RecordBatchStream + Send>>, lance::Error>(
                stream,
            )
        });

        let stream = rt()
            .block_on(None, fut)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))?;

        let dataset_stream = DatasetRecordBatchStream::new(stream);
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(LanceReader::from_stream(dataset_stream));
        reader.into_pyarrow(py)
    }
}

#[pyclass(
    name = "SqlQueryBuilder",
    module = "_lib",
    subclass,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct SqlQueryBuilder {
    builder: lance::dataset::sql::SqlQueryBuilder,
}

#[pymethods]
impl SqlQueryBuilder {
    #[pyo3(signature = (table_name))]
    fn table_name(&self, table_name: &str) -> Self {
        Self {
            builder: self.builder.clone().table_name(table_name),
        }
    }

    #[pyo3(signature = (with_row_id))]
    fn with_row_id(&self, with_row_id: bool) -> Self {
        Self {
            builder: self.builder.clone().with_row_id(with_row_id),
        }
    }

    #[pyo3(signature = (with_row_addr))]
    fn with_row_addr(&self, with_row_addr: bool) -> Self {
        Self {
            builder: self.builder.clone().with_row_addr(with_row_addr),
        }
    }

    /// Build the SQL query.
    fn build(&self) -> PyResult<SqlQuery> {
        Ok(SqlQuery {
            builder: self.builder.clone(),
        })
    }
}

// -------------------- Delta API Bindings --------------------

#[pyclass(name = "DatasetDelta", module = "_lib", subclass)]
pub struct DatasetDelta {
    inner: lance::dataset::delta::DatasetDelta,
}

#[pymethods]
impl DatasetDelta {
    /// List transactions between begin_version+1 and end_version.
    fn list_transactions(
        &self,
    ) -> PyResult<Vec<PyLance<lance::dataset::transaction::Transaction>>> {
        let txs = rt()
            .block_on(None, self.inner.list_transactions())?
            .infer_error()?;
        Ok(txs.into_iter().map(PyLance).collect())
    }

    /// Get inserted rows between begin_version (exclusive) and end_version (inclusive) as a stream reader.
    fn get_inserted_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use arrow::pyarrow::IntoPyArrow;
        use arrow_array::RecordBatchReader;
        let stream = rt()
            .block_on(None, self.inner.get_inserted_rows())?
            .infer_error()?;
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(LanceReader::from_stream(stream));
        reader.into_pyarrow(py)
    }

    /// Get updated rows between begin_version (exclusive) and end_version (inclusive) as a stream reader.
    fn get_updated_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use arrow::pyarrow::IntoPyArrow;
        use arrow_array::RecordBatchReader;
        let stream = rt()
            .block_on(None, self.inner.get_updated_rows())?
            .infer_error()?;
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(LanceReader::from_stream(stream));
        reader.into_pyarrow(py)
    }
}

#[pyclass(
    name = "DatasetDeltaBuilder",
    module = "_lib",
    subclass,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct DatasetDeltaBuilder {
    builder: lance::dataset::delta::DatasetDeltaBuilder,
}

#[pymethods]
impl DatasetDeltaBuilder {
    #[pyo3(signature = (version))]
    fn compared_against_version(&self, version: u64) -> Self {
        Self {
            builder: self.builder.clone().compared_against_version(version),
        }
    }

    #[pyo3(signature = (begin_version))]
    fn with_begin_version(&self, begin_version: u64) -> Self {
        Self {
            builder: self.builder.clone().with_begin_version(begin_version),
        }
    }

    #[pyo3(signature = (end_version))]
    fn with_end_version(&self, end_version: u64) -> Self {
        Self {
            builder: self.builder.clone().with_end_version(end_version),
        }
    }

    fn build(&self) -> PyResult<DatasetDelta> {
        let delta = self.builder.clone().build().infer_error()?;
        Ok(DatasetDelta { inner: delta })
    }
}

#[derive(FromPyObject)]
pub enum PyWriteDest {
    Dataset(Dataset),
    Uri(PyBackedStr),
}

impl PyWriteDest {
    pub fn as_dest(&self) -> WriteDestination<'_> {
        match self {
            Self::Dataset(ds) => WriteDestination::Dataset(ds.ds.clone()),
            Self::Uri(uri) => WriteDestination::Uri(uri),
        }
    }

    /// The table root uri of this destination (a branch dataset resolves to
    /// its main location). Used to root the namespace manifest store.
    pub fn table_root_uri(&self) -> PyResult<String> {
        match self {
            Self::Dataset(ds) => Ok(ds.ds.branch_location().find_main().infer_error()?.uri),
            Self::Uri(uri) => Ok(uri.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum IndexProgressEventType {
    Start,
    Progress,
    Complete,
}

impl IndexProgressEventType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Progress => "progress",
            Self::Complete => "complete",
        }
    }
}

#[derive(Debug, Clone)]
struct IndexProgressEvent {
    event: IndexProgressEventType,
    stage: String,
    completed: Option<u64>,
    total: Option<u64>,
    unit: String,
}

#[derive(Debug, Default)]
struct ActiveIndexProgressStage {
    stage: Option<String>,
    completed: Option<u64>,
    total: Option<u64>,
    unit: String,
}

#[derive(Debug)]
struct PyIndexBuildProgress {
    sender: SyncSender<IndexProgressEvent>,
    active_stage: Mutex<ActiveIndexProgressStage>,
}

impl PyIndexBuildProgress {
    fn new(sender: SyncSender<IndexProgressEvent>) -> Self {
        Self {
            sender,
            active_stage: Mutex::new(ActiveIndexProgressStage::default()),
        }
    }

    /// Send a lifecycle event (start/complete) with blocking semantics.
    ///
    /// These events are rare relative to the channel capacity
    /// (`INDEX_PROGRESS_QUEUE_SIZE`), so blocking here will not stall
    /// the tokio runtime in practice.  Progress events use the
    /// non-blocking `try_send_progress_event` instead.
    fn send_event(&self, event: IndexProgressEvent) -> lance::Result<()> {
        self.sender.send(event).map_err(|_| {
            Error::invalid_input("Index progress callback receiver disconnected".to_string())
        })
    }

    fn try_send_progress_event(&self, event: IndexProgressEvent) -> lance::Result<()> {
        match self.sender.try_send(event) {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => Err(Error::invalid_input(
                "Index progress callback receiver disconnected".to_string(),
            )),
        }
    }
}

#[async_trait]
impl IndexBuildProgress for PyIndexBuildProgress {
    async fn stage_start(&self, stage: &str, total: Option<u64>, unit: &str) -> lance::Result<()> {
        {
            let mut active_stage = self.active_stage.lock().unwrap();
            *active_stage = ActiveIndexProgressStage {
                stage: Some(stage.to_string()),
                completed: Some(0),
                total,
                unit: unit.to_string(),
            };
        }

        self.send_event(IndexProgressEvent {
            event: IndexProgressEventType::Start,
            stage: stage.to_string(),
            completed: Some(0),
            total,
            unit: unit.to_string(),
        })
    }

    async fn stage_progress(&self, stage: &str, completed: u64) -> lance::Result<()> {
        let event = {
            let mut active_stage = self.active_stage.lock().unwrap();
            if active_stage.stage.is_none() {
                active_stage.stage = Some(stage.to_string());
            }
            active_stage.completed = Some(completed);
            IndexProgressEvent {
                event: IndexProgressEventType::Progress,
                stage: stage.to_string(),
                completed: Some(completed),
                total: active_stage.total,
                unit: active_stage.unit.clone(),
            }
        };

        self.try_send_progress_event(event)
    }

    async fn stage_complete(&self, stage: &str) -> lance::Result<()> {
        let event = {
            let mut active_stage = self.active_stage.lock().unwrap();
            let completed = active_stage.completed.or(active_stage.total);
            let total = active_stage.total;
            let unit = active_stage.unit.clone();
            *active_stage = ActiveIndexProgressStage::default();
            IndexProgressEvent {
                event: IndexProgressEventType::Complete,
                stage: stage.to_string(),
                completed,
                total,
                unit,
            }
        };

        self.send_event(event)
    }
}

struct IndexProgressDispatcher {
    callback: Py<PyAny>,
    index_progress_cls: Py<PyAny>,
    receiver: Receiver<IndexProgressEvent>,
}

impl IndexProgressDispatcher {
    fn new(
        py: Python<'_>,
        callback: Py<PyAny>,
        receiver: Receiver<IndexProgressEvent>,
    ) -> PyResult<Self> {
        let index_progress_cls = py
            .import("lance.progress")?
            .getattr("IndexProgress")?
            .unbind();
        Ok(Self {
            callback,
            index_progress_cls,
            receiver,
        })
    }

    fn drain(&mut self) -> PyResult<()> {
        while let Ok(event) = self.receiver.try_recv() {
            let is_complete = event.event == IndexProgressEventType::Complete;
            if let Err(err) = self.dispatch(event) {
                if is_complete {
                    // Complete events are purely informational — the stage's work
                    // is already done. Propagating a callback error here would
                    // abort the operation after the real work has succeeded.
                    log::warn!(
                        "Ignoring progress callback error on stage-complete event: {}",
                        err
                    );
                } else {
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    fn dispatch(&self, event: IndexProgressEvent) -> PyResult<()> {
        Python::attach(|py| {
            let progress = self.index_progress_cls.call1(
                py,
                (
                    event.event.as_str(),
                    event.stage,
                    event.completed,
                    event.total,
                    event.unit,
                ),
            )?;
            self.callback.call1(py, (progress,))?;
            Ok(())
        })
    }
}

struct IndexProgressHandler {
    progress: Arc<PyIndexBuildProgress>,
    dispatcher: IndexProgressDispatcher,
}

impl Dataset {
    fn transform_ref(&self, reference: Option<Bound<PyAny>>) -> PyResult<Ref> {
        if let Some(reference) = reference {
            if let Ok(i) = reference.cast::<PyInt>() {
                let version_number: u64 = i.extract()?;
                Ok(version_number.into())
            } else if let Ok(tag_name) = reference.cast::<PyString>() {
                let tag: &str = &tag_name.to_string_lossy();
                Ok(tag.into())
            } else if let Ok(tuple) = reference.cast::<PyTuple>() {
                if tuple.len() == 2 {
                    let (branch_name, version_number) =
                        tuple.extract::<(Option<String>, Option<u64>)>()?;
                    Ok((branch_name.as_deref(), version_number).into())
                } else {
                    Err(PyValueError::new_err(
                        "Version tuple should be Tuple[Optional[str], Optional[int]]",
                    ))
                }
            } else {
                Err(PyValueError::new_err(
                    "Version must be an int, a str or a (int, str) tuple.",
                ))
            }
        } else {
            Ok(Ref::Version(
                self.ds.manifest.branch.clone(),
                Some(self.ds.version().version),
            ))
        }
    }

    fn _checkout_version(&self, version: impl Into<Ref> + std::marker::Send) -> PyResult<Self> {
        let ds = rt()
            .block_on(None, self.ds.checkout_version(version))?
            .map_err(|err| match err {
                lance::Error::NotFound { .. } => PyValueError::new_err(err.to_string()),
                _ => PyIOError::new_err(err.to_string()),
            })?;

        Ok(Self {
            ds: Arc::new(ds),
            uri: self.uri.clone(),
        })
    }

    fn list_versions(&self) -> PyResult<Vec<Version>> {
        rt().block_on(None, self.ds.versions())?.infer_error()
    }

    fn list_tags(&self) -> PyResult<HashMap<String, TagContents>> {
        rt().block_on(None, self.ds.tags().list())?.infer_error()
    }

    fn list_tags_ordered(&self, order: Option<&str>) -> PyResult<Vec<(String, TagContents)>> {
        let ordering = match order {
            Some("asc") => Some(std::cmp::Ordering::Less),
            Some("desc") => Some(std::cmp::Ordering::Greater),
            Some(invalid_order) => {
                return Err(PyValueError::new_err(format!(
                    "Invalid sort order '{}'. Valid values are: asc, desc",
                    invalid_order
                )));
            }
            None => None,
        };
        rt().block_on(None, async {
            self.ds.tags().list_tags_ordered(ordering).await
        })?
        .infer_error()
    }

    fn make_scan_stats_callback(callback: Bound<'_, PyAny>) -> PyResult<ExecutionStatsCallback> {
        if !callback.is_callable() {
            return Err(PyValueError::new_err("Callback must be callable"));
        }

        let callback = callback.unbind();

        Ok(Arc::new(move |stats| {
            Python::attach(|py| {
                let stats = ScanStatistics::from_lance(stats);
                match callback.call1(py, (stats,)) {
                    Ok(_) => (),
                    Err(e) => {
                        // Don't fail scan if callback fails
                        error!("Error in scan stats callback: {}", e);
                    }
                }
            });
        }))
    }

    fn make_index_progress_handler(
        py: Python<'_>,
        callback: Py<PyAny>,
    ) -> PyResult<IndexProgressHandler> {
        if !callback.bind(py).is_callable() {
            return Err(PyValueError::new_err("Progress callback must be callable"));
        }

        let (sender, receiver) = mpsc::sync_channel(INDEX_PROGRESS_QUEUE_SIZE);
        let progress = Arc::new(PyIndexBuildProgress::new(sender));
        let dispatcher = IndexProgressDispatcher::new(py, callback, receiver)?;
        Ok(IndexProgressHandler {
            progress,
            dispatcher,
        })
    }

    fn make_index_progress_handler_from_kwargs(
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<IndexProgressHandler>> {
        let Some(kwargs) = kwargs else {
            return Ok(None);
        };

        let Some(callback) = get_dict_opt::<Py<PyAny>>(kwargs, "progress_callback")? else {
            return Ok(None);
        };

        Self::make_index_progress_handler(kwargs.py(), callback).map(Some)
    }

    fn make_index_progress_handler_from_callback(
        callback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<IndexProgressHandler>> {
        let Some(callback) = callback else {
            return Ok(None);
        };

        Self::make_index_progress_handler(callback.py(), callback.clone().unbind()).map(Some)
    }

    fn run_index_future<F>(
        future: F,
        progress_handler: Option<&mut IndexProgressHandler>,
    ) -> PyResult<F::Output>
    where
        F: std::future::Future + Send,
        F::Output: Send,
    {
        if let Some(progress_handler) = progress_handler {
            rt().block_on_pumping(None, future, || progress_handler.dispatcher.drain())
        } else {
            rt().block_on(None, future)
        }
    }
}

#[pyfunction(name = "_write_dataset")]
pub fn write_dataset(
    reader: &Bound<'_, PyAny>,
    dest: PyWriteDest,
    options: &Bound<'_, PyDict>,
) -> PyResult<Dataset> {
    let params = get_write_params(options, &dest.table_root_uri()?)?;
    let py = options.py();
    let ds = if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        let batches = rt()
            .block_on(Some(py), scanner.to_reader())?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        rt().block_on(
            Some(py),
            LanceDataset::write(batches, dest.as_dest(), params),
        )?
        .map_err(|err| PyIOError::new_err(err.to_string()))?
    } else {
        let batches = ArrowArrayStreamReader::from_pyarrow_bound(reader)?;
        rt().block_on(
            Some(py),
            LanceDataset::write(batches, dest.as_dest(), params),
        )?
        .map_err(|err| PyIOError::new_err(err.to_string()))?
    };
    Ok(Dataset {
        uri: ds.uri().to_string(),
        ds: Arc::new(ds),
    })
}

fn parse_write_mode(mode: &str) -> PyResult<WriteMode> {
    match mode.to_string().to_lowercase().as_str() {
        "create" => Ok(WriteMode::Create),
        "append" => Ok(WriteMode::Append),
        "overwrite" => Ok(WriteMode::Overwrite),
        _ => Err(PyValueError::new_err(format!("Invalid mode {mode}"))),
    }
}

pub fn get_commit_handler(options: &Bound<'_, PyDict>) -> PyResult<Option<Arc<dyn CommitHandler>>> {
    Ok(if options.is_none() {
        None
    } else if let Ok(Some(commit_handler)) = options.get_item("commit_handler") {
        Some(Arc::new(PyCommitLock::new(
            commit_handler.into_pyobject(options.py())?.into(),
        )))
    } else {
        None
    })
}

// Gets a value from the dictionary and attempts to extract it to
// the desired type.  If the value is None then it treats it as if
// it were never present in the dictionary.  If the value is not
// None it will try and parse it and parsing failures will be
// returned (e.g. a parsing failure is not considered `None`)
fn get_dict_opt<'py, D: FromPyObjectOwned<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<D>> {
    let value = dict.get_item(key)?;
    value
        .and_then(|v| {
            if v.is_none() {
                None
            } else {
                Some(v.extract::<D>().map_err(Into::into))
            }
        })
        .transpose()
}

/// `table_uri` is the destination table's root uri; it roots the namespace
/// manifest store when `namespace_client_managed_versioning` is requested.
#[allow(deprecated)]
pub fn get_write_params(
    options: &Bound<'_, PyDict>,
    table_uri: &str,
) -> PyResult<Option<WriteParams>> {
    let params = if options.is_none() {
        None
    } else {
        let mut p = WriteParams::default();
        if let Some(mode) = get_dict_opt::<String>(options, "mode")? {
            p.mode = parse_write_mode(mode.as_str())?;
        };
        if let Some(maybe_nrows) = get_dict_opt::<usize>(options, "max_rows_per_file")? {
            p.max_rows_per_file = maybe_nrows;
        }
        if let Some(maybe_nrows) = get_dict_opt::<usize>(options, "max_rows_per_group")? {
            p.max_rows_per_group = maybe_nrows;
        }
        if let Some(maybe_nbytes) = get_dict_opt::<usize>(options, "max_bytes_per_file")? {
            p.max_bytes_per_file = maybe_nbytes;
        }
        if let Some(data_storage_version) = get_dict_opt::<String>(options, "data_storage_version")?
        {
            p.data_storage_version = Some(data_storage_version.parse().infer_error()?);
        }
        if let Some(progress) = get_dict_opt::<Py<PyAny>>(options, "progress")? {
            p.progress = Arc::new(PyWriteProgress::new(progress.into_py_any(options.py())?));
        }

        let storage_options = get_dict_opt::<HashMap<String, String>>(options, "storage_options")?;

        // Extract namespace_client and table_id for storage options provider creation
        let namespace_client_opt = get_dict_opt::<Bound<PyAny>>(options, "namespace_client")?;
        let table_id_opt = get_dict_opt::<Vec<String>>(options, "table_id")?;

        if let Some(so) = storage_options.clone() {
            // If namespace_client and table_id are provided, create storage options provider from them
            if let (Some(ns_client), Some(table_id)) =
                (namespace_client_opt.as_ref(), table_id_opt.as_ref())
            {
                let ns_client = extract_namespace_arc(options.py(), ns_client)?;
                let provider: Arc<dyn lance_io::object_store::StorageOptionsProvider> = Arc::new(
                    LanceNamespaceStorageOptionsProvider::new(ns_client, table_id.clone()),
                );
                let accessor = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
                    so, provider,
                ));
                p.store_params = Some(ObjectStoreParams {
                    storage_options_accessor: Some(accessor),
                    ..Default::default()
                });
            } else {
                // No namespace, just use storage options directly
                let accessor = Arc::new(StorageOptionsAccessor::with_static_options(so));
                p.store_params = Some(ObjectStoreParams {
                    storage_options_accessor: Some(accessor),
                    ..Default::default()
                });
            }
        }

        if let Some(enable_stable_row_ids) = get_dict_opt::<bool>(options, "enable_stable_row_ids")?
        {
            p.enable_stable_row_ids = enable_stable_row_ids;
        }
        if let Some(enable_v2_manifest_paths) =
            get_dict_opt::<bool>(options, "enable_v2_manifest_paths")?
        {
            p.enable_v2_manifest_paths = enable_v2_manifest_paths;
        }

        if let Some(auto_cleanup) = get_dict_opt::<Bound<PyAny>>(options, "auto_cleanup_options")? {
            let mut auto_cleanup_params = AutoCleanupParams::default();

            auto_cleanup_params.interval = auto_cleanup
                .get_item("interval")
                .and_then(|i| i.extract::<usize>())
                .ok()
                .unwrap_or(auto_cleanup_params.interval);

            auto_cleanup_params.older_than = auto_cleanup
                .get_item("older_than_seconds")
                .and_then(|i| i.extract::<i64>())
                .ok()
                .map(TimeDelta::seconds)
                .unwrap_or(auto_cleanup_params.older_than);

            p.auto_cleanup = Some(auto_cleanup_params);
        } else {
            p.auto_cleanup = None;
        }

        p.commit_handler = get_commit_handler(options)?;

        // Handle initial_bases parameter (list of DatasetBasePath objects)
        if let Some(initial_bases_list) =
            get_dict_opt::<Vec<Bound<PyAny>>>(options, "initial_bases")?
        {
            let mut new_bases = Vec::new();

            for item in initial_bases_list.iter() {
                if let Ok(dataset_base_path) = item.extract::<DatasetBasePath>() {
                    new_bases.push(BasePath::from(dataset_base_path));
                } else {
                    return Err(PyValueError::new_err(
                        "initial_bases must contain DatasetBasePath objects only",
                    ));
                }
            }

            if !new_bases.is_empty() {
                p = p.with_initial_bases(new_bases);
            }
        }

        // Handle target_bases parameter (list of strings - base names or paths)
        if let Some(target_bases_list) = get_dict_opt::<Vec<String>>(options, "target_bases")?
            && !target_bases_list.is_empty()
        {
            p = p.with_target_base_names_or_paths(target_bases_list);
        }

        // Handle base_store_params: per-base storage options keyed by base path URI
        if let Some(base_store_params) =
            get_dict_opt::<HashMap<String, HashMap<String, String>>>(options, "base_store_params")?
        {
            for (base_path, opts) in base_store_params {
                let accessor = Arc::new(StorageOptionsAccessor::with_static_options(opts));
                let store_params = ObjectStoreParams {
                    storage_options_accessor: Some(accessor),
                    ..Default::default()
                };
                p = p.with_base_store_params(base_path, store_params);
            }
        }

        if let Some(allow_external) =
            get_dict_opt::<bool>(options, "allow_external_blob_outside_bases")?
        {
            p = p.with_allow_external_blob_outside_bases(allow_external);
        }
        if let Some(external_blob_mode) = get_dict_opt::<String>(options, "external_blob_mode")? {
            p = p.with_external_blob_mode(
                ExternalBlobMode::try_from(external_blob_mode.as_str()).infer_error()?,
            );
        }
        if let Some(max_bytes) = get_dict_opt::<usize>(options, "blob_pack_file_size_threshold")? {
            p = p.with_blob_pack_file_size_threshold(max_bytes);
        }

        // Handle properties
        if let Some(props) =
            get_dict_opt::<HashMap<String, String>>(options, "transaction_properties")?
        {
            let mut new_props = p
                .transaction_properties
                .as_ref()
                .map(|arc| (**arc).clone())
                .unwrap_or_default();
            for (key, value) in props {
                new_props.insert(key, value);
            }
            p.transaction_properties = Some(Arc::new(new_props));
        }

        // Handle namespace_client and table_id for managed versioning (external manifest store)
        // Only set if commit_handler is not already set by user and namespace_client_managed_versioning is true
        let namespace_client_managed_versioning =
            get_dict_opt::<bool>(options, "namespace_client_managed_versioning")?.unwrap_or(false);
        if p.commit_handler.is_none()
            && namespace_client_managed_versioning
            && let (Some(ns_client), Some(table_id)) =
                (namespace_client_opt.as_ref(), table_id_opt.as_ref())
        {
            // The store derives the branch a request targets from the base path
            // it is handed, resolved against the table root.
            let ns_client = extract_namespace_arc(options.py(), ns_client)?;
            let external_store = LanceNamespaceExternalManifestStore::for_table_uri(
                ns_client,
                table_id.clone(),
                table_uri,
            )
            .infer_error()?;
            let commit_handler: Arc<dyn CommitHandler> = Arc::new(ExternalManifestCommitHandler {
                external_manifest_store: Arc::new(external_store),
            });
            p.commit_handler = Some(commit_handler);
        }

        Some(p)
    };
    Ok(params)
}

fn prepare_vector_index_params(
    index_type: &str,
    column_type: &DataType,
    storage_options: Option<HashMap<String, String>>,
    kwargs: Option<&Bound<PyDict>>,
) -> PyResult<Box<dyn IndexParams>> {
    let mut m_type = MetricType::L2;
    let mut ivf_params = IvfBuildParams::default();
    let mut hnsw_params = HnswBuildParams::default();
    let mut pq_params = PQBuildParams::default();
    let mut sq_params = SQBuildParams::default();
    let mut rq_params = RQBuildParams::default();
    let mut index_file_version = IndexFileVersion::V3;
    let mut skip_transpose = false;

    if let Some(kwargs) = kwargs {
        // Parse metric type
        if let Some(mt) = kwargs.get_item("metric_type")? {
            m_type = MetricType::try_from(mt.to_string().to_lowercase().as_str())
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        // Parse sample rate
        if let Some(sample_rate) = kwargs.get_item("sample_rate")? {
            let sample_rate: usize = sample_rate.extract()?;
            ivf_params.sample_rate = sample_rate;
            pq_params.sample_rate = sample_rate;
            sq_params.sample_rate = sample_rate;
        }

        if let Some(max_iters) = kwargs.get_item("max_iters")? {
            let max_iters: usize = max_iters.extract()?;
            ivf_params.max_iters = max_iters;
            pq_params.max_iters = max_iters;
        }

        if let Some(streaming_sample_rate) = kwargs.get_item("streaming_sample_rate")? {
            ivf_params.streaming_sample_rate = Some(streaming_sample_rate.extract()?);
        }
        if let Some(streaming_coreset_rate) = kwargs.get_item("streaming_coreset_rate")? {
            ivf_params.streaming_coreset_rate = Some(streaming_coreset_rate.extract()?);
        }
        if let Some(streaming_refine_passes) = kwargs.get_item("streaming_refine_passes")? {
            ivf_params.streaming_refine_passes = streaming_refine_passes.extract()?;
        }

        // Parse IVF params
        if let Some(n) = kwargs.get_item("num_partitions")? {
            ivf_params.num_partitions = Some(n.extract()?)
        };

        if let Some(n) = kwargs.get_item("target_partition_size")? {
            ivf_params.target_partition_size = Some(n.extract()?)
        };

        if let Some(n) = kwargs.get_item("shuffle_partition_concurrency")? {
            ivf_params.shuffle_partition_concurrency = n.extract()?
        };

        if let Some(c) = kwargs.get_item("ivf_centroids")? {
            let batch = RecordBatch::from_pyarrow_bound(&c)?;
            if "_ivf_centroids" != batch.schema().field(0).name() {
                return Err(PyValueError::new_err(
                    "Expected '_ivf_centroids' as the first column name.",
                ));
            }

            let centroid_type = match column_type {
                DataType::List(field)
                    if matches!(field.data_type(), DataType::FixedSizeList(_, _)) =>
                {
                    field.data_type()
                }
                _ => column_type,
            };

            // It's important that the centroids are the same data type
            // as the vectors that will be indexed.
            let mut centroids: Arc<dyn Array> = batch.column(0).clone();
            if centroids.data_type() != centroid_type {
                centroids =
                    cast_with_options(centroids.as_ref(), centroid_type, &Default::default())
                        .map_err(|e| {
                            PyValueError::new_err(format!(
                                "Failed to cast centroids to vector type: {}",
                                e
                            ))
                        })?;
            }
            let centroids = as_fixed_size_list_array(centroids.as_ref());

            ivf_params.centroids = Some(Arc::new(centroids.clone()))
        };

        if let Some(f) = kwargs.get_item("precomputed_partitions_file")? {
            ivf_params.precomputed_partitions_file = Some(f.to_string());
        };

        if let Some(storage_options) = storage_options {
            ivf_params.storage_options = Some(storage_options);
        }

        match (
            kwargs.get_item("precomputed_shuffle_buffers")?,
            kwargs.get_item("precomputed_shuffle_buffers_path")?,
        ) {
            (Some(l), Some(p)) => {
                let path = Path::parse(p.to_string()).map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to parse precomputed_shuffle_buffers_path: {}",
                        e
                    ))
                })?;
                let list = l.cast::<PyList>()?.iter().map(|f| f.to_string()).collect();
                ivf_params.precomputed_shuffle_buffers = Some((path, list));
            }
            (None, None) => {}
            _ => {
                return Err(PyValueError::new_err(
                    "precomputed_shuffle_buffers and precomputed_shuffle_buffers_path must be specified together.",
                ));
            }
        }

        // Parse HNSW params
        if let Some(max_level) = kwargs.get_item("max_level")? {
            hnsw_params.max_level = max_level.extract()?;
        }

        if let Some(m) = kwargs.get_item("m")? {
            hnsw_params.m = m.extract()?;
        }

        if let Some(ef_c) = kwargs.get_item("ef_construction")? {
            hnsw_params.ef_construction = ef_c.extract()?;
        }

        // Parse PQ/RQ params
        if let Some(n) = kwargs.get_item("num_bits")? {
            let num_bits: u8 = n.extract()?;
            pq_params.num_bits = num_bits as usize;
            rq_params.num_bits = num_bits;
        };

        if let Some(n) = kwargs.get_item("num_sub_vectors")? {
            pq_params.num_sub_vectors = n.extract()?
        };

        if let Some(c) = kwargs.get_item("pq_codebook")? {
            let batch = RecordBatch::from_pyarrow_bound(&c)?;
            if "_pq_codebook" != batch.schema().field(0).name() {
                return Err(PyValueError::new_err(
                    "Expected '_pq_codebook' as the first column name.",
                ));
            }
            let codebook = as_fixed_size_list_array(batch.column(0));
            pq_params.codebook = Some(codebook.values().clone())
        };

        if let Some(r) = kwargs.get_item("rabitq_model")? {
            let json: String = r.extract()?;
            let meta: RabitQuantizationMetadata = serde_json::from_str(&json)
                .map_err(|e| PyValueError::new_err(format!("Invalid rabitq_model JSON: {e}")))?;
            rq_params.rotation = Some(meta);
        };

        if let Some(version) = kwargs.get_item("index_file_version")? {
            let version: String = version.extract()?;
            index_file_version = IndexFileVersion::try_from(&version)
                .map_err(|e| PyValueError::new_err(format!("Invalid index_file_version: {e}")))?;
        }

        if let Some(value) = kwargs.get_item("skip_transpose")? {
            skip_transpose = value.extract()?;
        }
    }

    let mut params = match index_type {
        "IVF_FLAT" => Ok(Box::new(VectorIndexParams::with_ivf_flat_params(
            m_type, ivf_params,
        ))),

        "IVF_PQ" => Ok(Box::new(VectorIndexParams::with_ivf_pq_params(
            m_type, ivf_params, pq_params,
        ))),

        "IVF_SQ" => Ok(Box::new(VectorIndexParams::with_ivf_sq_params(
            m_type, ivf_params, sq_params,
        ))),

        "IVF_RQ" => Ok(Box::new(VectorIndexParams::with_ivf_rq_params(
            m_type, ivf_params, rq_params,
        ))),

        "IVF_HNSW_FLAT" => Ok(Box::new(VectorIndexParams::ivf_hnsw(
            m_type,
            ivf_params,
            hnsw_params,
        ))),

        "IVF_HNSW_PQ" => Ok(Box::new(VectorIndexParams::with_ivf_hnsw_pq_params(
            m_type,
            ivf_params,
            hnsw_params,
            pq_params,
        ))),

        "IVF_HNSW_SQ" => Ok(Box::new(VectorIndexParams::with_ivf_hnsw_sq_params(
            m_type,
            ivf_params,
            hnsw_params,
            sq_params,
        ))),

        _ => Err(PyValueError::new_err(format!(
            "Index type '{index_type}' is not supported."
        ))),
    }?;
    params.version(index_file_version);
    params.skip_transpose(skip_transpose);
    if let Some(kwargs) = kwargs
        && let Some(acc) = kwargs.get_item("accelerator")?
    {
        params
            .runtime_hints
            .insert("lancedb.accelerator".to_string(), acc.to_string());
    }
    Ok(params)
}

#[pyclass(name = "_FragmentWriteProgress", module = "_lib")]
#[derive(Debug)]
pub struct PyWriteProgress {
    /// A Python object that implements the `WriteFragmentProgress` trait.
    py_obj: Py<PyAny>,
}

impl PyWriteProgress {
    fn new(obj: Py<PyAny>) -> Self {
        Self { py_obj: obj }
    }
}

#[async_trait]
impl WriteFragmentProgress for PyWriteProgress {
    async fn begin(&self, fragment: &Fragment) -> lance::Result<()> {
        let json_str = serde_json::to_string(fragment)?;

        Python::attach(|py| -> PyResult<()> {
            self.py_obj
                .call_method(py, "_do_begin", (json_str,), None)?;
            Ok(())
        })
        .map_err(|e| {
            lance::Error::invalid_input(format!(
                "Failed to call begin() on WriteFragmentProgress: {}",
                e
            ))
        })?;
        Ok(())
    }

    async fn complete(&self, fragment: &Fragment) -> lance::Result<()> {
        let json_str = serde_json::to_string(fragment)?;

        Python::attach(|py| -> PyResult<()> {
            self.py_obj
                .call_method(py, "_do_complete", (json_str,), None)?;
            Ok(())
        })
        .map_err(|e| {
            lance::Error::invalid_input(format!(
                "Failed to call complete() on WriteFragmentProgress: {}",
                e
            ))
        })?;
        Ok(())
    }
}

/// Formats a Python error just as it would in Python interpreter.
fn format_python_error(e: PyErr, py: Python) -> PyResult<String> {
    let sys_mod = py.import("sys")?;
    // the traceback is the third element of the tuple returned by sys.exc_info()
    let traceback = sys_mod.call_method0("exc_info")?.get_item(2)?;

    let tracback_mod = py.import("traceback")?;
    let fmt_func = tracback_mod.getattr("format_exception")?;
    let e_type = e.get_type(py).to_owned();
    let formatted = fmt_func.call1((e_type, &e, traceback))?;
    let lines: Vec<String> = formatted.extract()?;
    Ok(lines.join(""))
}

struct PyBatchUDFCheckpointWrapper {
    inner: Py<PyAny>,
}

impl PyBatchUDFCheckpointWrapper {
    fn batch_info_to_py(&self, info: &BatchInfo, py: Python) -> PyResult<Py<PyAny>> {
        self.inner
            .getattr(py, "BatchInfo")?
            .call1(py, (info.fragment_id, info.batch_index))
    }
}

impl UDFCheckpointStore for PyBatchUDFCheckpointWrapper {
    fn get_batch(&self, info: &BatchInfo) -> lance::Result<Option<RecordBatch>> {
        Python::attach(|py| {
            let info = self.batch_info_to_py(info, py)?;
            let batch = self.inner.call_method1(py, "get_batch", (info,))?;
            let batch: Option<PyArrowType<RecordBatch>> = batch.extract(py)?;
            Ok(batch.map(|b| b.0))
        })
        .map_err(|err: PyErr| {
            lance_core::Error::invalid_input(format!(
                "Failed to call get_batch() on UDFCheckpointer: {}",
                err
            ))
        })
    }

    fn get_fragment(&self, fragment_id: u32) -> lance::Result<Option<Fragment>> {
        let fragment_data = Python::attach(|py| {
            let fragment = self
                .inner
                .call_method1(py, "get_fragment", (fragment_id,))?;
            let fragment: Option<String> = fragment.extract(py)?;
            Ok(fragment)
        })
        .map_err(|err: PyErr| {
            lance_core::Error::invalid_input(format!(
                "Failed to call get_fragment() on UDFCheckpointer: {}",
                err
            ))
        })?;
        fragment_data
            .map(|data| {
                serde_json::from_str(&data).map_err(|err| {
                    lance_core::Error::invalid_input(format!(
                        "Failed to deserialize fragment data: {}",
                        err
                    ))
                })
            })
            .transpose()
    }

    fn insert_batch(&self, info: BatchInfo, batch: RecordBatch) -> lance::Result<()> {
        Python::attach(|py| {
            let info = self.batch_info_to_py(&info, py)?;
            let batch = PyArrowType(batch);
            self.inner.call_method1(py, "insert_batch", (info, batch))?;
            Ok(())
        })
        .map_err(|err: PyErr| {
            lance_core::Error::invalid_input(format!(
                "Failed to call insert_batch() on UDFCheckpointer: {}",
                err
            ))
        })
    }

    fn insert_fragment(&self, fragment: Fragment) -> lance_core::Result<()> {
        let data = serde_json::to_string(&fragment).map_err(|err| {
            lance_core::Error::io(format!("Failed to serialize fragment data: {}", err))
        })?;
        Python::attach(|py| {
            self.inner
                .call_method1(py, "insert_fragment", (fragment.id, data))?;
            Ok(())
        })
        .map_err(|err: PyErr| {
            lance_core::Error::invalid_input(format!(
                "Failed to call insert_fragment() on UDFCheckpointer: {}",
                err
            ))
        })
    }
}

#[pyclass(name = "PyFullTextQuery", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyFullTextQuery {
    pub(crate) inner: FtsQuery,
}

#[pymethods]
impl PyFullTextQuery {
    #[staticmethod]
    #[pyo3(signature = (query, column, boost=1.0, fuzziness=Some(0), max_expansions=50, operator="OR", prefix_length=0))]
    fn match_query(
        query: String,
        column: String,
        boost: f32,
        fuzziness: Option<u32>,
        max_expansions: usize,
        operator: &str,
        prefix_length: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MatchQuery::new(query)
                .with_column(Some(column))
                .with_boost(boost)
                .with_fuzziness(fuzziness)
                .with_max_expansions(max_expansions)
                .with_operator(
                    Operator::try_from(operator)
                        .map_err(|e| PyValueError::new_err(format!("Invalid operator: {}", e)))?,
                )
                .with_prefix_length(prefix_length)
                .into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (query, column, slop))]
    fn phrase_query(query: String, column: String, slop: u32) -> PyResult<Self> {
        Ok(Self {
            inner: PhraseQuery::new(query)
                .with_column(Some(column))
                .with_slop(slop)
                .into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (positive, negative,negative_boost=None))]
    fn boost_query(positive: Self, negative: Self, negative_boost: Option<f32>) -> PyResult<Self> {
        Ok(Self {
            inner: BoostQuery::new(positive.inner, negative.inner, negative_boost).into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (query, columns, boosts=None, operator="OR"))]
    fn multi_match_query(
        query: String,
        columns: Vec<String>,
        boosts: Option<Vec<f32>>,
        operator: &str,
    ) -> PyResult<Self> {
        let q = MultiMatchQuery::try_new(query, columns)
            .map_err(|e| PyValueError::new_err(format!("Invalid query: {}", e)))?;
        let q = if let Some(boosts) = boosts {
            q.try_with_boosts(boosts)
                .map_err(|e| PyValueError::new_err(format!("Invalid boosts: {}", e)))?
        } else {
            q
        };

        let op = Operator::try_from(operator)
            .map_err(|e| PyValueError::new_err(format!("Invalid operator: {}", e)))?;

        Ok(Self {
            inner: q.with_operator(op).into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (queries))]
    fn boolean_query(queries: Vec<(String, Self)>) -> PyResult<Self> {
        let mut sub_queries = Vec::with_capacity(queries.len());
        for (occur, q) in queries {
            let occur = Occur::try_from(occur.as_str())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            sub_queries.push((occur, q.inner));
        }

        Ok(Self {
            inner: BooleanQuery::new(sub_queries).into(),
        })
    }
}

type VectorQueryParams = (
    String,
    arrow_array::ArrayRef,
    usize,
    usize,
    Option<usize>,
    Option<MetricType>,
    Option<u32>,
    bool,
    Option<usize>,
    i32,
    ApproxMode,
);

fn extract_query_parallelism(value: &Bound<'_, PyAny>) -> PyResult<i32> {
    let query_parallelism = value.extract()?;
    if query_parallelism < -1 {
        Err(PyValueError::new_err("query_parallelism must be >= -1"))
    } else {
        Ok(query_parallelism)
    }
}

fn vector_query_query_parallelism_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<i32> {
    if let Some(query_parallelism) = dict.get_item("query_parallelism")?
        && !query_parallelism.is_none()
    {
        extract_query_parallelism(&query_parallelism)
    } else {
        Ok(DEFAULT_QUERY_PARALLELISM)
    }
}

fn vector_query_approx_mode_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<ApproxMode> {
    if let Some(approx_mode) = dict.get_item("approx_mode")?
        && !approx_mode.is_none()
    {
        match approx_mode.to_string().to_lowercase().as_str() {
            "fast" => Ok(ApproxMode::Fast),
            "normal" => Ok(ApproxMode::Normal),
            "accurate" => Ok(ApproxMode::Accurate),
            value => Err(PyValueError::new_err(format!(
                "approx_mode must be one of 'fast', 'normal', or 'accurate', got '{value}'"
            ))),
        }
    } else {
        Ok(ApproxMode::Normal)
    }
}

fn vector_query_params_from_dict(
    dict: &Bound<'_, PyDict>,
    default_k: usize,
) -> PyResult<VectorQueryParams> {
    let column = dict
        .get_item("column")?
        .ok_or_else(|| PyKeyError::new_err("Need column for nearest"))?
        .to_string();

    let qval = dict
        .get_item("q")?
        .ok_or_else(|| PyKeyError::new_err("Need q for nearest"))?;
    let data = ArrayData::from_pyarrow_bound(&qval)?;
    let key = make_array(data);

    let k: usize = if let Some(k) = dict.get_item("k")? {
        if k.is_none() {
            // Use limit if k is not specified, default to 10.
            default_k
        } else {
            k.extract()?
        }
    } else {
        default_k
    };

    let mut minimum_nprobes = DEFAULT_NPROBES;
    let mut maximum_nprobes: Option<usize> = None;

    if let Some(nprobes) = dict.get_item("nprobes")?
        && !nprobes.is_none()
    {
        let extracted: usize = nprobes.extract()?;
        minimum_nprobes = extracted;
        maximum_nprobes = Some(extracted);
    }

    if let Some(min_nprobes) = dict.get_item("minimum_nprobes")?
        && !min_nprobes.is_none()
    {
        minimum_nprobes = min_nprobes.extract()?;
    }

    if let Some(max_nprobes) = dict.get_item("maximum_nprobes")?
        && !max_nprobes.is_none()
    {
        maximum_nprobes = Some(max_nprobes.extract()?);
    }

    if let Some(maximum_nprobes_val) = maximum_nprobes
        && minimum_nprobes > maximum_nprobes_val
    {
        return Err(PyValueError::new_err(
            "minimum_nprobes must be <= maximum_nprobes",
        ));
    }

    if minimum_nprobes < 1 {
        return Err(PyValueError::new_err("minimum_nprobes must be >= 1"));
    }

    if let Some(maximum_nprobes_val) = maximum_nprobes
        && maximum_nprobes_val < 1
    {
        return Err(PyValueError::new_err("maximum_nprobes must be >= 1"));
    }

    let metric_type: Option<MetricType> = if let Some(metric) = dict.get_item("metric")? {
        if metric.is_none() {
            None
        } else {
            Some(
                MetricType::try_from(metric.to_string().to_lowercase().as_str())
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        }
    } else {
        None
    };

    // When refine factor is specified, a final Refine stage will be added to the I/O plan,
    // and use Flat index over the raw vectors to refine the results.
    // By default, `refine_factor` is None to not involve extra I/O exec node and random access.
    let refine_factor: Option<u32> = if let Some(rf) = dict.get_item("refine_factor")? {
        if rf.is_none() { None } else { rf.extract()? }
    } else {
        None
    };

    let use_index: bool = if let Some(idx) = dict.get_item("use_index")? {
        idx.extract()?
    } else {
        true
    };

    let ef: Option<usize> = if let Some(ef_obj) = dict.get_item("ef")? {
        if ef_obj.is_none() {
            None
        } else {
            ef_obj.extract()?
        }
    } else {
        None
    };

    let query_parallelism = vector_query_query_parallelism_from_dict(dict)?;
    let approx_mode = vector_query_approx_mode_from_dict(dict)?;

    Ok((
        column,
        key,
        k,
        minimum_nprobes,
        maximum_nprobes,
        metric_type,
        refine_factor,
        use_index,
        ef,
        query_parallelism,
        approx_mode,
    ))
}

#[pyclass(name = "PySearchFilter", from_py_object)]
#[derive(Debug, Clone)]
pub struct PySearchFilter {
    pub(crate) inner: QueryFilter,
}

#[pymethods]
impl PySearchFilter {
    /// Create a search filter from a full text query.
    #[staticmethod]
    #[pyo3(signature = (query))]
    fn from_full_text_query(query: PyFullTextQuery) -> PyResult<Self> {
        Ok(Self {
            inner: QueryFilter::Fts(FullTextSearchQuery::new_query(query.inner.clone())),
        })
    }

    /// Create a query filter from a vector search query dict.
    #[staticmethod]
    #[pyo3(signature = (query))]
    fn from_vector_search_query(query: &Bound<'_, PyDict>) -> PyResult<Self> {
        let default_k = 10;
        let (
            column,
            key,
            k,
            minimum_nprobes,
            maximum_nprobes,
            metric_type_opt,
            refine_factor,
            use_index,
            ef,
            query_parallelism,
            approx_mode,
        ) = vector_query_params_from_dict(query, default_k)?;

        let metric_type = Some(metric_type_opt.unwrap_or(MetricType::L2));

        let vector_query = VectorQuery {
            column,
            key,
            k,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes,
            maximum_nprobes,
            ef,
            refine_factor,
            metric_type,
            use_index,
            query_parallelism,
            dist_q_c: 0.0,
            approx_mode,
        };

        Ok(Self {
            inner: QueryFilter::Vector(vector_query),
        })
    }
}
