// Copyright 2023 Lance Developers.
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

//! Lance Columnar Data Format
//!
//! Lance columnar data format is an alternative to Parquet. It provides 100x faster for random access,
//! automatic versioning, optimized for computer vision, bioinformatics, spatial and ML data.
//! [Apache Arrow](https://arrow.apache.org/) and DuckDB compatible.

// Workaround for https://github.com/rust-lang/rust-clippy/issues/12039
// Remove after upgrading pyo3 to 0.23
#![allow(clippy::useless_conversion)]

use std::env;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{self, Ordering};

use ::arrow::pyarrow::PyArrowType;
use ::arrow_schema::Schema as ArrowSchema;
use ::lance::arrow::json::ArrowJsonExt;
use ::lance::datafusion::LanceTableProvider;
use ::lance::index::DatasetIndexExt;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
#[cfg(feature = "datagen")]
use datagen::register_datagen;
use dataset::blob::LanceBlobFile;
use dataset::cleanup::{
    CleanupCandidateFile, CleanupExplanation, CleanupReferencedBranch, CleanupStats,
};
use dataset::io_stats::IoStats;
use dataset::optimize::{
    PyCompaction, PyCompactionMetrics, PyCompactionPlan, PyCompactionTask, PyRewriteResult,
};
use dataset::{DatasetBasePath, MergeInsertBuilder, PyFullTextQuery, PySearchFilter};
use env_logger::{Builder, Env};
use file::{
    LanceBufferDescriptor, LanceColumnMetadata, LanceFileMetadata, LanceFileReader,
    LanceFileStatistics, LanceFileWriteSummary, LanceFileWriter, LancePageMetadata, stable_version,
};
use log::Level;
use pyo3::exceptions::PyIOError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyCapsule};
use scanner::ScanStatistics;
use session::Session;
use std::ffi::CString;
use std::ptr::NonNull;

pub(crate) mod arrow;
pub(crate) mod blob;
#[cfg(feature = "datagen")]
pub(crate) mod datagen;
pub(crate) mod dataset;
pub(crate) mod debug;
pub(crate) mod error;
pub(crate) mod executor;
pub(crate) mod file;
pub(crate) mod fragment;
pub(crate) mod fts;
pub(crate) mod indices;
pub(crate) mod mem_wal;
pub(crate) mod namespace;
pub(crate) mod object_store;
pub(crate) mod otel;
pub(crate) mod reader;
pub(crate) mod rowids;
pub(crate) mod scanner;
pub(crate) mod schema;
pub(crate) mod session;
pub(crate) mod storage_options;
pub(crate) mod tracing;
pub(crate) mod transaction;
pub(crate) mod utils;

pub use crate::arrow::{BFloat16, bfloat16_array};
use crate::file::LanceFileSession;
use crate::fragment::{write_fragments, write_fragments_transaction};
use crate::tracing::{PyTraceEvent, capture_trace_events, shutdown_tracing};
pub use crate::tracing::{TraceGuard, trace_to_chrome};
use crate::utils::Hnsw;
use crate::utils::KMeans;
pub use dataset::Dataset;
pub use dataset::serialize_row_addrs;
pub use dataset::write_dataset;
use fragment::{FileFragment, PyDeletionFile, PyRowDatasetVersionMeta, PyRowIdMeta};
pub use indices::register_indices;
pub use reader::LanceReader;
use rowids::{PyRowIdSequence, PyRowIdSequenceIterator};
pub use scanner::Scanner;

use crate::blob::{
    PyBlobDescriptor, PyBlobDescriptorArrayBuilder, PyDedicatedBlobWriter, PyPackedBlobWriter,
};
use crate::executor::BackgroundExecutor;

const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(not(feature = "datagen"))]
#[pyfunction]
pub fn is_datagen_supported() -> bool {
    false
}

// A fallback module for when datagen is not enabled
#[cfg(not(feature = "datagen"))]
fn register_datagen(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let datagen = PyModule::new(py, "datagen")?;
    datagen.add_wrapped(wrap_pyfunction!(is_datagen_supported))?;
    m.add_submodule(&datagen)?;
    Ok(())
}

fn create_background_executor() -> BackgroundExecutor {
    // TODO: make this runtime configurable (e.g. num threads)
    BackgroundExecutor::new()
}

static BACKGROUND_EXECUTOR: atomic::AtomicPtr<BackgroundExecutor> =
    atomic::AtomicPtr::new(std::ptr::null_mut());

static EXECUTOR_INSTALLED: atomic::AtomicBool = atomic::AtomicBool::new(false);

static ATFORK_INSTALLED: atomic::AtomicBool = atomic::AtomicBool::new(false);

pub fn rt() -> &'static BackgroundExecutor {
    loop {
        let ptr = BACKGROUND_EXECUTOR.load(Ordering::SeqCst);
        if !ptr.is_null() {
            // SAFETY: installed executors are leaked and remain valid for the
            // process lifetime. BackgroundExecutor uses shared access only.
            return unsafe { &*ptr };
        }
        if !EXECUTOR_INSTALLED.fetch_or(true, Ordering::SeqCst) {
            break;
        }
        std::thread::yield_now();
    }
    if !ATFORK_INSTALLED.fetch_or(true, Ordering::SeqCst) {
        install_atfork();
    }
    let new_ptr = Box::into_raw(Box::new(create_background_executor()));
    BACKGROUND_EXECUTOR.store(new_ptr, Ordering::SeqCst);
    // SAFETY: the executor is leaked and all of its operations take `&self`.
    unsafe { &*new_ptr }
}

/// After a fork() operation, force re-creation of the BackgroundExecutor. Note: this function
/// runs in "async-signal context" which means that we can't (safely) do much here.
extern "C" fn atfork_child() {
    BACKGROUND_EXECUTOR.store(std::ptr::null_mut(), Ordering::SeqCst);
    EXECUTOR_INSTALLED.store(false, Ordering::SeqCst);
}

#[cfg(not(windows))]
fn install_atfork() {
    unsafe { libc::pthread_atfork(None, None, Some(atfork_child)) };
}

#[cfg(windows)]
fn install_atfork() {}

pub fn init_logging(mut log_builder: Builder) {
    let logger = log_builder.build();

    let max_level = logger.filter();

    let trace_level = env::var("LANCE_TRACING").unwrap_or_default().to_lowercase();
    let trace_level = match trace_level.as_str() {
        "debug" => Level::Debug,
        "info" => Level::Info,
        "warn" => Level::Warn,
        "error" => Level::Error,
        "trace" => Level::Trace,
        _ => Level::Info,
    };

    tracing::initialize_tracing(trace_level);
    log::set_boxed_logger(Box::new(logger)).unwrap();
    log::set_max_level(max_level);
}

fn set_timestamp_precision(builder: &mut env_logger::Builder) {
    if let Ok(timestamp_precision) = env::var("LANCE_LOG_TS_PRECISION") {
        match timestamp_precision.as_str() {
            "ns" => {
                builder.format_timestamp_nanos();
            }
            "us" => {
                builder.format_timestamp_micros();
            }
            "ms" => {
                builder.format_timestamp_millis();
            }
            "s" => {
                builder.format_timestamp_secs();
            }
            _ => {
                // Can't log here because logging is not initialized yet
                println!(
                    "Invalid timestamp precision (valid values: ns, us, ms, s): {}, using default",
                    timestamp_precision
                );
            }
        };
    }
}

fn set_log_file_target(builder: &mut env_logger::Builder) {
    if let Ok(log_file_path) = env::var("LANCE_LOG_FILE") {
        let path = Path::new(&log_file_path);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            println!(
                "Failed to create parent directories for log file '{}': {}, using stderr",
                log_file_path, e
            );
            return;
        }

        // Try to open/create the log file
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => {
                builder.target(env_logger::Target::Pipe(Box::new(file)));
            }
            Err(e) => {
                println!(
                    "Failed to open log file '{}': {}, using stderr",
                    log_file_path, e
                );
            }
        }
    }
}

#[pymodule]
fn lance(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let env = Env::new()
        .filter_or("LANCE_LOG", "warn")
        .write_style("LANCE_LOG_STYLE");
    let mut log_builder = env_logger::Builder::from_env(env);
    set_timestamp_precision(&mut log_builder);
    set_log_file_target(&mut log_builder);
    init_logging(log_builder);

    m.add_class::<FFILanceTableProvider>()?;
    m.add_class::<Scanner>()?;
    m.add_class::<Dataset>()?;
    m.add_class::<DatasetBasePath>()?;
    m.add_class::<FileFragment>()?;
    m.add_class::<PyDeletionFile>()?;
    m.add_class::<PyRowIdMeta>()?;
    m.add_class::<PyRowIdSequence>()?;
    m.add_class::<PyRowIdSequenceIterator>()?;
    m.add_class::<PyRowDatasetVersionMeta>()?;
    m.add_class::<MergeInsertBuilder>()?;
    m.add_class::<LanceBlobFile>()?;
    m.add_class::<PyBlobDescriptor>()?;
    m.add_class::<PyBlobDescriptorArrayBuilder>()?;
    m.add_class::<PyPackedBlobWriter>()?;
    m.add_class::<PyDedicatedBlobWriter>()?;
    m.add_class::<LanceFileReader>()?;
    m.add_class::<LanceFileWriter>()?;
    m.add_class::<LanceFileWriteSummary>()?;
    m.add_class::<LanceFileSession>()?;
    m.add_class::<LanceFileMetadata>()?;
    m.add_class::<LanceFileStatistics>()?;
    m.add_class::<LanceColumnMetadata>()?;
    m.add_class::<LancePageMetadata>()?;
    m.add_class::<LanceBufferDescriptor>()?;
    m.add_class::<BFloat16>()?;
    m.add_class::<CleanupStats>()?;
    m.add_class::<CleanupCandidateFile>()?;
    m.add_class::<CleanupReferencedBranch>()?;
    m.add_class::<CleanupExplanation>()?;
    m.add_class::<IoStats>()?;
    m.add_class::<KMeans>()?;
    m.add_class::<Hnsw>()?;
    m.add_class::<PyCompactionTask>()?;
    m.add_class::<PyCompaction>()?;
    m.add_class::<PyCompactionPlan>()?;
    m.add_class::<PyRewriteResult>()?;
    m.add_class::<PyCompactionMetrics>()?;
    m.add_class::<ScanStatistics>()?;
    m.add_class::<Session>()?;
    m.add_class::<object_store::PyObjectStoreProvider>()?;
    m.add_class::<object_store::PyObjectStoreRegistry>()?;
    m.add_class::<PyTraceEvent>()?;
    m.add_class::<TraceGuard>()?;
    m.add_class::<fts::FtsToken>()?;
    m.add_class::<schema::LanceSchema>()?;
    m.add_class::<PyFullTextQuery>()?;
    m.add_class::<PySearchFilter>()?;
    m.add_class::<namespace::PyDirectoryNamespace>()?;
    m.add_class::<namespace::PyRestNamespace>()?;
    m.add_class::<namespace::PyRestAdapter>()?;
    m.add_class::<storage_options::PyStorageOptionsAccessor>()?;
    // MemWAL classes
    m.add_class::<mem_wal::PyCompactedSsTable>()?;
    m.add_class::<mem_wal::PyShardSnapshot>()?;
    m.add_class::<mem_wal::PyShardWriter>()?;
    m.add_class::<mem_wal::PyLsmScanner>()?;
    m.add_class::<mem_wal::PyExecutionPlan>()?;
    m.add_class::<mem_wal::PyLsmPointLookupPlanner>()?;
    m.add_class::<mem_wal::PyLsmVectorSearchPlanner>()?;
    m.add_wrapped(wrap_pyfunction!(mem_wal::py_evaluate_sharding_spec))?;
    m.add_wrapped(wrap_pyfunction!(mem_wal::py_write_pk_sidecar))?;
    m.add_wrapped(wrap_pyfunction!(bfloat16_array))?;
    m.add_wrapped(wrap_pyfunction!(write_dataset))?;
    m.add_wrapped(wrap_pyfunction!(serialize_row_addrs))?;
    m.add_wrapped(wrap_pyfunction!(write_fragments))?;
    m.add_wrapped(wrap_pyfunction!(write_fragments_transaction))?;
    m.add_wrapped(wrap_pyfunction!(schema_to_json))?;
    m.add_wrapped(wrap_pyfunction!(json_to_schema))?;
    m.add_wrapped(wrap_pyfunction!(schema::parse_field_path))?;
    m.add_wrapped(wrap_pyfunction!(schema::format_field_path))?;
    m.add_wrapped(wrap_pyfunction!(trace_to_chrome))?;
    m.add_wrapped(wrap_pyfunction!(capture_trace_events))?;
    m.add_wrapped(wrap_pyfunction!(shutdown_tracing))?;
    m.add_wrapped(wrap_pyfunction!(fts::tokenize))?;
    // OpenTelemetry metrics bridge
    m.add_class::<otel::PyMetricPoint>()?;
    m.add_class::<otel::PyMetricDescription>()?;
    m.add_wrapped(wrap_pyfunction!(otel::register_lance_metrics_recorder))?;
    m.add_wrapped(wrap_pyfunction!(otel::lance_metrics_catalog))?;
    m.add_wrapped(wrap_pyfunction!(otel::snapshot_lance_metrics))?;
    m.add_wrapped(wrap_pyfunction!(manifest_needs_migration))?;
    m.add_wrapped(wrap_pyfunction!(language_model_home))?;
    m.add_wrapped(wrap_pyfunction!(bytes_read_counter))?;
    m.add_wrapped(wrap_pyfunction!(iops_counter))?;
    m.add_wrapped(wrap_pyfunction!(simd_info))?;
    m.add_wrapped(wrap_pyfunction!(stable_version))?;
    // Debug functions
    m.add_wrapped(wrap_pyfunction!(debug::format_schema))?;
    m.add_wrapped(wrap_pyfunction!(debug::format_manifest))?;
    m.add_wrapped(wrap_pyfunction!(debug::format_fragment))?;
    m.add_wrapped(wrap_pyfunction!(debug::list_transactions))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    register_datagen(py, m)?;
    register_indices(py, m)?;
    Ok(())
}

#[pyfunction(name = "iops_counter")]
fn iops_counter() -> PyResult<u64> {
    Ok(::lance::io::iops_counter())
}

/// Returns a dict describing which SIMD tier the lance runtime dispatches to
/// on this host, plus the raw CPU feature flags it detected.
///
/// Mirrors `pyarrow.runtime_info()`: a cheap, transparent way to verify that
/// the host is hitting the expected SIMD tier (e.g., `"avx512_fp16"`,
/// `"avx2"`) when debugging vector-search performance.
///
/// Returns:
///   {
///     "tier": str,                      # e.g. "avx2", "avx_fma", "neon", "none"
///     "target_arch": str,               # e.g. "x86_64", "aarch64", "loongarch64"
///     "host_features": list[str],       # raw CPU feature flags (x86_64 only)
///   }
///
/// Examples:
///   >>> import lance
///   >>> info = lance.simd_info()
///   >>> sorted(info)
///   ['host_features', 'target_arch', 'tier']
///   >>> isinstance(info["tier"], str)
///   True
#[pyfunction]
pub fn simd_info(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let info = lance_core::utils::cpu::simd_info();
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("tier", info.tier.to_string())?;
    dict.set_item("target_arch", info.target_arch)?;
    dict.set_item("host_features", info.host_features)?;
    Ok(dict.into())
}

#[pyfunction(name = "bytes_read_counter")]
fn bytes_read_counter() -> PyResult<u64> {
    Ok(::lance::io::bytes_read_counter())
}

#[pyfunction(name = "_schema_to_json")]
fn schema_to_json(schema: PyArrowType<ArrowSchema>) -> PyResult<String> {
    schema.0.to_json().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to convert schema to json: {}", e))
    })
}

#[pyfunction(name = "_json_to_schema")]
fn json_to_schema(json: &str) -> PyResult<PyArrowType<ArrowSchema>> {
    let schema = ArrowSchema::from_json(json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to convert json to schema: {}, json={}",
            e, json
        ))
    })?;
    Ok(schema.into())
}

#[pyfunction]
pub fn language_model_home() -> PyResult<String> {
    let Some(p) = lance_index::scalar::inverted::language_model_home() else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Failed to get language model home",
        ));
    };
    let Some(pstr) = p.to_str() else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Failed to convert language model home to str",
        ));
    };
    Ok(String::from(pstr))
}

#[pyfunction]
#[pyo3(signature = (dataset,))]
fn manifest_needs_migration(dataset: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = dataset.py();
    let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
    let dataset_ref = &dataset.bind(py).borrow().ds;
    let indices = rt()
        .block_on(Some(py), dataset_ref.load_indices())?
        .map_err(|err| PyIOError::new_err(format!("Could not read dataset metadata: {}", err)))?;
    let (manifest, _) = rt()
        .block_on(Some(py), dataset_ref.latest_manifest())?
        .map_err(|err| PyIOError::new_err(format!("Could not read dataset metadata: {}", err)))?;
    Ok(::lance::io::commit::manifest_needs_migration(
        &manifest, &indices,
    ))
}

#[pyclass(
    name = "FFILanceTableProvider",
    module = "lance",
    subclass,
    skip_from_py_object
)]
#[derive(Clone)]
struct FFILanceTableProvider {
    dataset: Arc<::lance::Dataset>,
    with_row_id: bool,
    with_row_addr: bool,
}

#[pymethods]
impl FFILanceTableProvider {
    #[new]
    #[pyo3(signature = (dataset, *, with_row_id = false, with_row_addr = false))]
    fn new(dataset: &Bound<'_, PyAny>, with_row_id: bool, with_row_addr: bool) -> PyResult<Self> {
        let py = dataset.py();
        let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
        let dataset_ref = &dataset.bind(py).borrow().ds;
        // TODO: https://github.com/lance-format/lance/issues/3966 remove this workaround
        let _ = rt().block_on(Some(py), dataset_ref.load_indices())?;
        Ok(Self {
            dataset: dataset_ref.clone(),
            with_row_id,
            with_row_addr,
        })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let name = CString::new("datafusion_table_provider").unwrap();
        let a_lance_table_provider = Arc::new(LanceTableProvider::new(
            self.dataset.clone(),
            self.with_row_id,
            self.with_row_addr,
        ));

        let codec = ffi_logical_codec_from_pycapsule(session)?;
        let ffi_provider = FFI_TableProvider::new_with_ffi_codec(
            a_lance_table_provider,
            true,
            rt().get_runtime_handle(),
            codec,
        );
        PyCapsule::new(py, ffi_provider, Some(name.clone()))
    }
}

fn ffi_logical_codec_from_pycapsule(obj: Bound<PyAny>) -> PyResult<FFI_LogicalExtensionCodec> {
    let attr_name = "__datafusion_logical_extension_codec__";
    let capsule = if obj.hasattr(attr_name)? {
        obj.getattr(attr_name)?.call0()?
    } else {
        obj
    };

    let capsule = capsule.cast::<PyCapsule>()?;
    let data: NonNull<FFI_LogicalExtensionCodec> = capsule
        .pointer_checked(Some(c_str!("datafusion_logical_extension_codec")))?
        .cast();
    let codec = unsafe { data.as_ref() };

    Ok(codec.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_executor_is_a_shared_singleton() {
        let first: &'static BackgroundExecutor = rt();
        let second: &'static BackgroundExecutor = rt();
        assert!(std::ptr::eq(first, second));
    }
}
