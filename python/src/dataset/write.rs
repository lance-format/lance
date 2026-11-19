// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python dataset write input, options, and progress bindings.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::{FromPyArrow, PyArrowType};
use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use chrono::TimeDelta;
use lance::dataset::{
    AutoCleanupParams, BatchInfo, BatchUDF, Dataset as LanceDataset, ExternalBlobMode,
    NewColumnTransform, UDFCheckpointStore, WriteDestination, WriteMode, WriteParams,
    progress::WriteFragmentProgress,
};
use lance::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
use lance_file::writer::FileWriterOptions;
use lance_io::object_store::{
    LanceNamespaceStorageOptionsProvider, ObjectStoreParams, StorageOptionsAccessor,
};
use lance_table::format::{BasePath, Fragment};
use lance_table::io::commit::CommitHandler;
use lance_table::io::commit::external_manifest::ExternalManifestCommitHandler;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;

use crate::Scanner;
use crate::dataset::commit::PyCommitLock;
use crate::error::PythonErrorExt;
use crate::namespace::extract_namespace_arc;
use crate::rt;
use crate::session::Session;

use super::Dataset;
use super::base_path::DatasetBasePath;
use super::get_dict_opt;

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
        let data_cache_bytes = get_dict_opt::<u64>(options, "data_cache_bytes")?;
        let max_page_bytes = get_dict_opt::<u64>(options, "max_page_bytes")?;
        if data_cache_bytes.is_some() || max_page_bytes.is_some() {
            p.file_writer_options = Some(FileWriterOptions {
                data_cache_bytes,
                max_page_bytes,
                ..Default::default()
            });
        }
        if let Some(data_storage_version) = get_dict_opt::<String>(options, "data_storage_version")?
        {
            p.data_storage_version = Some(data_storage_version.parse().infer_error()?);
        }
        if let Some(progress) = get_dict_opt::<Py<PyAny>>(options, "progress")? {
            p.progress = Arc::new(PyWriteProgress::new(progress.into_py_any(options.py())?));
        }
        if let Some(session) = get_dict_opt::<Session>(options, "session")? {
            p.session = Some(session.inner.clone());
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

        // Handle target_all_bases parameter (bool: include primary storage)
        if let Some(target_all_bases) = get_dict_opt::<bool>(options, "target_all_bases")? {
            p = p.with_target_all_bases(target_all_bases);
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
