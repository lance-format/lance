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

use std::fmt::Write as _;
use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::{FromPyArrow, PyArrowType, ToPyArrow};
use arrow_array::RecordBatchReader;
use futures::TryFutureExt;
use lance::Error;
use lance::dataset::fragment::FileFragment as LanceFragment;
use lance::dataset::scanner::{ColumnOrdering, MaterializationStyle};
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{InsertBuilder, NewColumnTransform, WriteParams};
use lance_core::datatypes::BlobHandling;
use lance_io::utils::CachedFileSize;
use lance_table::format::overlay::DataOverlayFile;
use lance_table::format::{
    DataFile, DeletionFile, DeletionFileType, Fragment, RowDatasetVersionMeta, RowIdMeta,
};
use lance_table::io::deletion::deletion_file_path;
use object_store::path::Path;
use pyo3::basic::CompareOp;
use pyo3::types::PyTuple;
use pyo3::{exceptions::*, types::PyDict};
use pyo3::{intern, prelude::*};

use crate::dataset::{PyWriteDest, get_write_params, transforms_from_python};
use crate::error::PythonErrorExt;
use crate::schema::{LanceSchema, logical_schema_from_lance};
use crate::utils::{PyLance, export_vec, extract_vec};
use crate::{Dataset, Scanner, rt};

#[pyclass(name = "_Fragment", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct FileFragment {
    fragment: LanceFragment,
}

impl FileFragment {
    pub fn new(frag: LanceFragment) -> Self {
        Self { fragment: frag }
    }
}

#[pymethods]
impl FileFragment {
    fn __repr__(&self) -> PyResult<String> {
        let mut s = String::new();
        write!(
            s,
            "LanceFileFragment(id={}, data_files=[",
            self.fragment.id()
        )
        .unwrap();
        let file_path = self
            .fragment
            .metadata()
            .files
            .iter()
            .map(|f| format!("'{}'", f.path))
            .collect::<Vec<_>>()
            .join(", ");
        write!(s, "{}]", file_path).unwrap();
        if let Some(deletion) = &self.fragment.metadata().deletion_file {
            let path = deletion_file_path(&Default::default(), self.id() as u64, deletion);
            write!(s, ", deletion_file='{}'", path).unwrap();
        }
        write!(s, ")").unwrap();
        Ok(s)
    }

    #[staticmethod]
    #[pyo3(signature = (filename, dataset, fragment_id))]
    fn create_from_file(
        filename: &str,
        dataset: &Dataset,
        fragment_id: usize,
    ) -> PyResult<PyLance<Fragment>> {
        let metadata = rt().block_on(None, async {
            LanceFragment::create_from_file(filename, dataset.ds.as_ref(), fragment_id, None)
                .await
                .map_err(|err| PyIOError::new_err(err.to_string()))
        })??;
        Ok(PyLance(metadata))
    }

    #[staticmethod]
    #[pyo3(signature = (dataset, path, base_id=None))]
    fn create_data_file(
        dataset: &Dataset,
        path: &str,
        base_id: Option<u32>,
    ) -> PyResult<PyLance<DataFile>> {
        let ds = dataset.ds.clone();
        let data_file = rt()
            .block_on(None, ds.create_data_file(path, base_id))?
            .infer_error()?;
        Ok(PyLance(data_file))
    }

    #[staticmethod]
    #[pyo3(signature = (dataset_uri, fragment_id, reader, **kwargs))]
    fn create(
        dataset_uri: &str,
        fragment_id: Option<usize>,
        reader: &Bound<PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyLance<Fragment>> {
        let params = if let Some(kw_params) = kwargs {
            get_write_params(kw_params, dataset_uri)?
        } else {
            None
        };

        let batches = convert_reader(reader)?;

        reader.py().detach(|| {
            rt().runtime.block_on(async move {
                let metadata =
                    LanceFragment::create(dataset_uri, fragment_id.unwrap_or(0), batches, params)
                        .await
                        .map_err(|err| PyIOError::new_err(err.to_string()))?;

                Ok(PyLance(metadata))
            })
        })
    }

    fn id(&self) -> usize {
        self.fragment.id()
    }

    pub fn metadata(&self) -> PyLance<Fragment> {
        PyLance(self.fragment.metadata().clone())
    }

    #[pyo3(signature=(filter=None))]
    fn count_rows(&self, filter: Option<String>) -> PyResult<usize> {
        rt().runtime.block_on(async {
            self.fragment
                .count_rows(filter)
                .await
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature=(columns=None, with_row_address=None))]
    fn open_session(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        with_row_address: Option<bool>,
    ) -> PyResult<FragmentSession> {
        let dataset_schema = self_.fragment.dataset().schema();
        let projection = if let Some(columns) = columns {
            dataset_schema
                .project(&columns)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            dataset_schema.clone()
        };

        let fragment = self_.fragment.clone();
        let session = rt()
            .spawn(Some(self_.py()), async move {
                fragment
                    .open_session(&projection, with_row_address.unwrap_or(false))
                    .await
            })?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(FragmentSession {
            session: Arc::new(session),
        })
    }

    #[pyo3(signature=(row_indices, columns=None))]
    fn take<'py>(
        self_: PyRef<'py, Self>,
        row_indices: Vec<usize>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let dataset_schema = self_.fragment.dataset().schema();
        let projection = if let Some(columns) = columns {
            dataset_schema
                .project(&columns)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            dataset_schema.clone()
        };

        let indices = row_indices.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let fragment = self_.fragment.clone();
        let batch = rt()
            .spawn(Some(self_.py()), async move {
                fragment.take(&indices, &projection).await
            })?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        batch.to_pyarrow(self_.py())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(columns=None, columns_with_transform=None, batch_size=None, filter=None, limit=None, offset=None, with_row_id=None, with_row_address=None, batch_readahead=None, blob_handling=None, order_by=None, use_scalar_index=None, io_buffer_size=None, late_materialization=None, include_deleted_rows=None, batch_size_bytes=None, strict_batch_size=None))]
    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        columns_with_transform: Option<Vec<(String, String)>>,
        batch_size: Option<usize>,
        filter: Option<String>,
        limit: Option<i64>,
        offset: Option<i64>,
        with_row_id: Option<bool>,
        with_row_address: Option<bool>,
        batch_readahead: Option<usize>,
        blob_handling: Option<Bound<PyAny>>,
        order_by: Option<Vec<PyLance<ColumnOrdering>>>,
        use_scalar_index: Option<bool>,
        io_buffer_size: Option<u64>,
        late_materialization: Option<Bound<PyAny>>,
        include_deleted_rows: Option<bool>,
        batch_size_bytes: Option<u64>,
        strict_batch_size: Option<bool>,
    ) -> PyResult<Scanner> {
        let mut scanner = self_.fragment.scan();

        match (columns, columns_with_transform) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Cannot specify both `columns` and `columns_with_transform`",
                ));
            }
            (Some(cols), None) => {
                scanner
                    .project(&cols)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            (None, Some(cols_with_transform)) => {
                scanner
                    .project_with_transform(&cols_with_transform)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            (None, None) => {}
        }

        if let Some(batch_size) = batch_size {
            scanner.batch_size(batch_size);
        }
        if let Some(f) = filter {
            scanner
                .filter(&f)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        scanner
            .limit(limit, offset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        if with_row_id.unwrap_or(false) {
            scanner.with_row_id();
        }
        if with_row_address.unwrap_or(false) {
            scanner.with_row_address();
        }
        if let Some(batch_readahead) = batch_readahead {
            scanner.batch_readahead(batch_readahead);
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
        if let Some(orderings) = order_by {
            let col_orderings = Some(orderings.into_iter().map(|co| co.0).collect());
            scanner
                .order_by(col_orderings)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(io_buffer_size) = io_buffer_size {
            scanner.io_buffer_size(io_buffer_size);
        }
        if let Some(use_scalar_index) = use_scalar_index {
            scanner.use_scalar_index(use_scalar_index);
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
                    MaterializationStyle::all_early_except(&columns, self_.fragment.schema())
                        .infer_error()?,
                );
            } else {
                return Err(PyValueError::new_err(
                    "late_materialization must be a bool or a list of strings",
                ));
            }
        }
        if let Some(batch_size_bytes) = batch_size_bytes {
            scanner.batch_size_bytes(batch_size_bytes);
        }
        if let Some(true) = include_deleted_rows {
            scanner.include_deleted_rows();
        }
        if let Some(strict_batch_size) = strict_batch_size {
            scanner.strict_batch_size(strict_batch_size);
        }
        let scn = Arc::new(scanner);
        Ok(Scanner::new(scn))
    }

    #[pyo3(signature=(reader, batch_size=None))]
    fn add_columns_from_reader(
        &mut self,
        reader: &Bound<PyAny>,
        batch_size: Option<u32>,
    ) -> PyResult<(PyLance<Fragment>, LanceSchema)> {
        let batches = ArrowArrayStreamReader::from_pyarrow_bound(reader)?;

        let transforms = NewColumnTransform::Reader(Box::new(batches));

        let fragment = self.fragment.clone();
        let (fragment, schema) = rt()
            .spawn(None, async move {
                fragment.add_columns(transforms, None, batch_size).await
            })?
            .infer_error()?;

        Ok((PyLance(fragment), LanceSchema(schema)))
    }

    #[pyo3(signature=(transforms, read_columns=None, batch_size=None))]
    fn add_columns(
        &mut self,
        py: Python<'_>,
        transforms: &Bound<'_, PyAny>,
        read_columns: Option<Vec<String>>,
        batch_size: Option<u32>,
    ) -> PyResult<(PyLance<Fragment>, LanceSchema)> {
        let transforms = transforms_from_python(py, transforms)?;

        let fragment = self.fragment.clone();
        let (fragment, schema) = rt()
            .spawn(None, async move {
                fragment
                    .add_columns(transforms, read_columns, batch_size)
                    .await
            })?
            .infer_error()?;

        Ok((PyLance(fragment), LanceSchema(schema)))
    }

    fn merge(
        &mut self,
        reader: PyArrowType<ArrowArrayStreamReader>,
        left_on: String,
        right_on: String,
        max_field_id: i32,
    ) -> PyResult<(PyLance<Fragment>, LanceSchema)> {
        let mut fragment = self.fragment.clone();
        let (fragment, schema) = rt()
            .spawn(None, async move {
                fragment
                    .merge_columns(reader.0, &left_on, &right_on, max_field_id)
                    .await
            })?
            .infer_error()?;

        Ok((PyLance(fragment), LanceSchema(schema)))
    }

    fn update_columns(
        &mut self,
        reader: PyArrowType<ArrowArrayStreamReader>,
        left_on: String,
        right_on: String,
    ) -> PyResult<(PyLance<Fragment>, Vec<u32>)> {
        let mut fragment = self.fragment.clone();
        let (updated_fragment, fields_modified) = rt()
            .spawn(None, async move {
                fragment.update_columns(reader.0, &left_on, &right_on).await
            })?
            .infer_error()?;

        Ok((PyLance(updated_fragment), fields_modified))
    }

    fn delete(&self, predicate: &str) -> PyResult<Option<Self>> {
        let old_fragment = self.fragment.clone();
        let updated_fragment = rt()
            .block_on(None, async { old_fragment.delete(predicate).await })?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        match updated_fragment {
            Some(frag) => Ok(Some(Self::new(frag))),
            None => Ok(None),
        }
    }

    /// Delete rows by their local (within-fragment) physical row offsets.
    ///
    /// Adds the given 0-based offsets to this fragment's deletion vector and
    /// writes a new deletion file. Returns the updated fragment, or None if every
    /// row is now deleted. Unlike `delete(predicate)`, this deletes exactly the
    /// supplied rows without re-evaluating a filter -- useful when the caller
    /// already knows which rows to delete (e.g. offsets collected from a prior
    /// scan), avoiding a redundant predicate evaluation.
    fn delete_rows(&self, offsets: Vec<u32>) -> PyResult<Option<Self>> {
        let old_fragment = self.fragment.clone();
        // The core deletion path only errors once the deletion count reaches
        // physical_rows, so out-of-range offsets must be rejected here.
        let updated_fragment = rt()
            .block_on(None, async {
                let physical_rows = old_fragment.physical_rows().await?;
                if let Some(&offset) = offsets.iter().find(|&&o| o as usize >= physical_rows) {
                    return Err(Error::invalid_input(format!(
                        "delete_rows offset {offset} is out of range for fragment {} \
                         with {physical_rows} rows",
                        old_fragment.id()
                    )));
                }
                old_fragment.extend_deletions(offsets).await
            })?
            .infer_error()?;

        match updated_fragment {
            Some(frag) => Ok(Some(Self::new(frag))),
            None => Ok(None),
        }
    }

    fn schema<'py>(self_: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let schema = self_.fragment.dataset().schema();
        let logical_schema = logical_schema_from_lance(schema);
        logical_schema.to_pyarrow(self_.py())
    }

    /// Returns the data file objects associated with this fragment.
    fn data_files(self_: PyRef<'_, Self>) -> PyResult<Vec<PyLance<DataFile>>> {
        let data_files: Vec<_> = self_
            .fragment
            .metadata()
            .files
            .iter()
            .map(|f| PyLance(f.clone()))
            .collect();
        Ok(data_files)
    }

    fn deletion_file(&self) -> PyResult<Option<String>> {
        let deletion = self.fragment.metadata().deletion_file.clone();
        Ok(deletion
            .map(|d| deletion_file_path(&Default::default(), self.id() as u64, &d).to_string()))
    }

    #[getter]
    fn num_deletions(&self) -> PyResult<usize> {
        rt().block_on(None, self.fragment.count_deletions())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    #[getter]
    fn physical_rows(&self) -> PyResult<usize> {
        rt().block_on(None, self.fragment.physical_rows())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    fn validate(&self) -> PyResult<()> {
        rt().block_on(None, self.fragment.validate())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }
}

impl From<FileFragment> for LanceFragment {
    fn from(fragment: FileFragment) -> Self {
        fragment.fragment
    }
}

fn do_write_fragments(
    dest: PyWriteDest,
    reader: &Bound<PyAny>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Transaction> {
    let batches = convert_reader(reader)?;

    let params = match kwargs {
        Some(params) => get_write_params(params, &dest.table_root_uri()?)?.unwrap_or_default(),
        None => WriteParams::default(),
    };

    rt().block_on(
        Some(reader.py()),
        InsertBuilder::new(dest.as_dest())
            .with_params(&params)
            .execute_uncommitted_stream(batches),
    )?
    .map_err(|err| PyIOError::new_err(err.to_string()))
}

#[pyfunction(name = "_write_fragments")]
#[pyo3(signature = (dest, reader, **kwargs))]
pub fn write_fragments(
    dest: PyWriteDest,
    reader: &Bound<'_, PyAny>,
    kwargs: Option<&Bound<PyDict>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let written = do_write_fragments(dest, reader, kwargs)?;

    let get_fragments = |operation| match operation {
        Operation::Overwrite { fragments, .. } => Ok(fragments),
        Operation::Append { fragments, .. } => Ok(fragments),
        _ => Err(Error::internal("Unexpected operation")),
    };
    let fragments =
        get_fragments(written.operation).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    export_vec(reader.py(), &fragments)
}

#[pyfunction(name = "_write_fragments_transaction")]
#[pyo3(signature = (dest, reader, **kwargs))]
pub fn write_fragments_transaction<'py>(
    dest: PyWriteDest,
    reader: &'py Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let written = do_write_fragments(dest, reader, kwargs)?;

    PyLance(written).into_pyobject(reader.py())
}

fn convert_reader(reader: &Bound<PyAny>) -> PyResult<Box<dyn RecordBatchReader + Send + 'static>> {
    if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        let reader = rt().block_on(
            Some(reader.py()),
            scanner
                .to_reader()
                .map_err(|err| PyValueError::new_err(err.to_string())),
        )??;
        Ok(Box::new(reader))
    } else {
        Ok(Box::new(ArrowArrayStreamReader::from_pyarrow_bound(
            reader,
        )?))
    }
}

#[pyclass(name = "DeletionFile", module = "lance.fragment")]
pub struct PyDeletionFile(pub DeletionFile);

#[pymethods]
impl PyDeletionFile {
    #[new]
    fn new(
        read_version: u64,
        id: u64,
        file_type: &str,
        num_deleted_rows: usize,
        base_id: Option<u32>,
    ) -> PyResult<Self> {
        let file_type = match file_type {
            "array" => DeletionFileType::Array,
            "bitmap" => DeletionFileType::Bitmap,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "file_type must be either 'array' or 'bitmap', got '{}'",
                    file_type
                )));
            }
        };
        Ok(Self(DeletionFile {
            read_version,
            id,
            file_type,
            num_deleted_rows: Some(num_deleted_rows),
            base_id,
        }))
    }

    fn asdict(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(slf.py());

        dict.set_item(intern!(slf.py(), "read_version"), slf.0.read_version)?;
        dict.set_item(intern!(slf.py(), "id"), slf.0.id)?;
        dict.set_item(intern!(slf.py(), "file_type"), slf.file_type())?;
        dict.set_item(
            intern!(slf.py(), "num_deleted_rows"),
            slf.0.num_deleted_rows,
        )?;
        dict.set_item(intern!(slf.py(), "base_id"), slf.0.base_id)?;

        Ok(dict)
    }

    fn __repr__(&self) -> String {
        let mut repr = "DeletionFile(".to_string();
        write!(repr, "type='{}'", self.file_type()).unwrap();
        if let Some(num_deleted_rows) = self.0.num_deleted_rows {
            write!(repr, ", num_deleted_rows={}", num_deleted_rows).unwrap();
        }
        write!(repr, ")").unwrap();
        repr
    }

    #[getter]
    fn read_version(&self) -> u64 {
        self.0.read_version
    }

    #[getter]
    fn id(&self) -> u64 {
        self.0.id
    }

    #[getter]
    fn num_deleted_rows(&self) -> Option<usize> {
        self.0.num_deleted_rows
    }

    #[getter]
    fn file_type(&self) -> &str {
        match self.0.file_type {
            DeletionFileType::Array => "array",
            DeletionFileType::Bitmap => "bitmap",
        }
    }

    #[getter]
    fn base_id(&self) -> &Option<u32> {
        &self.0.base_id
    }

    #[pyo3(signature = (fragment_id, base_uri=None))]
    fn path(&self, fragment_id: u64, base_uri: Option<&str>) -> PyResult<String> {
        let base_path = if let Some(base_uri) = base_uri {
            Path::from_url_path(base_uri).map_err(|e| {
                PyValueError::new_err(format!("Invalid base URI: {}: {}", base_uri, e))
            })?
        } else {
            Path::default()
        };
        Ok(deletion_file_path(&base_path, fragment_id, &self.0).to_string())
    }

    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump CompactionPlan due to error: {}",
                err
            ))
        })
    }

    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let deletion_file = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!("Could not load DeletionFile due to error: {}", err))
        })?;
        Ok(Self(deletion_file))
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state])?.extract()?;
        let from_json = PyModule::import(py, "lance.fragment")?
            .getattr("DeletionFile")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for CompactionTask",
            )),
        }
    }
}

#[pyclass(name = "RowIdMeta", module = "lance.fragment")]
pub struct PyRowIdMeta(pub RowIdMeta);

#[pyclass(name = "RowDatasetVersionMeta", module = "lance.fragment")]
pub struct PyRowDatasetVersionMeta(pub RowDatasetVersionMeta);

#[pymethods]
impl PyRowIdMeta {
    fn asdict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pythonize::pythonize(py, &self.0)
            .map(|b| b.unbind())
            .map_err(|err| PyValueError::new_err(format!("Could not convert RowIdMeta: {}", err)))
    }

    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump CompactionPlan due to error: {}",
                err
            ))
        })
    }

    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let row_id_meta = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!("Could not load RowIdMeta due to error: {}", err))
        })?;
        Ok(Self(row_id_meta))
    }

    #[staticmethod]
    pub fn from_dict(dict: &Bound<'_, PyAny>) -> PyResult<Self> {
        let row_id_meta: RowIdMeta = pythonize::depythonize(dict)
            .map_err(|err| PyValueError::new_err(format!("Could not load RowIdMeta: {}", err)))?;
        Ok(Self(row_id_meta))
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state])?.extract()?;
        let from_json = PyModule::import(py, "lance.fragment")?
            .getattr("RowIdMeta")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for CompactionTask",
            )),
        }
    }
}

#[pymethods]
impl PyRowDatasetVersionMeta {
    fn asdict(&self) -> PyResult<Bound<'_, PyDict>> {
        Err(PyNotImplementedError::new_err(
            "PyRowDatasetVersionMeta.asdict is not yet supported.",
        ))
    }

    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not serialize RowDatasetVersionMeta due to error: {}",
                err
            ))
        })
    }

    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let dataset_version_meta = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not load RowDatasetVersionMeta due to error: {}",
                err
            ))
        })?;
        Ok(Self(dataset_version_meta))
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state])?.extract()?;
        let from_json = PyModule::import(py, "lance.fragment")?
            .getattr("RowDatasetVersionMeta")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for RowDatasetVersionMeta",
            )),
        }
    }
}

#[pyclass(
    name = "FragmentSession",
    module = "_lib",
    subclass,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct FragmentSession {
    session: Arc<lance::dataset::fragment::session::FragmentSession>,
}

#[pymethods]
impl FragmentSession {
    #[pyo3(signature=(indices))]
    pub fn take<'py>(self_: PyRef<'py, Self>, indices: Vec<u32>) -> PyResult<Bound<'py, PyAny>> {
        let session = self_.session.clone();
        let batch = rt()
            .spawn(
                Some(self_.py()),
                async move { session.take(&indices).await },
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        batch.to_pyarrow(self_.py())
    }
}

impl FromPyObject<'_, '_> for PyLance<Fragment> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let files = extract_vec(&ob.getattr("files")?)?;

        let deletion_file: Option<PyRef<PyDeletionFile>> =
            ob.getattr("deletion_file")?.extract()?;
        let deletion_file = deletion_file.map(|f| f.0.clone());

        let row_id_meta: Option<PyRef<PyRowIdMeta>> = ob.getattr("row_id_meta")?.extract()?;
        let row_id_meta = row_id_meta.map(|r| r.0.clone());
        let last_updated_at_version_meta: Option<PyRef<PyRowDatasetVersionMeta>> =
            ob.getattr("last_updated_at_version_meta")?.extract()?;
        let last_updated_at_version_meta = last_updated_at_version_meta.map(|r| r.0.clone());
        let created_at_version_meta: Option<PyRef<PyRowDatasetVersionMeta>> =
            ob.getattr("created_at_version_meta")?.extract()?;
        let created_at_version_meta = created_at_version_meta.map(|r| r.0.clone());

        Ok(Self(Fragment {
            id: ob.getattr("id")?.extract()?,
            files,
            deletion_file,
            physical_rows: ob.getattr("physical_rows")?.extract()?,
            row_id_meta,
            last_updated_at_version_meta,
            created_at_version_meta,
            // Round-tripped so overlays survive operations that pass existing
            // fragments back (a manual Delete/Update/Merge commit). Sorting
            // newest-last is deferred to the manifest reload after commit.
            overlays: extract_vec::<DataOverlayFile>(&ob.getattr("overlays")?)?,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&Fragment> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance.fragment"))
            .and_then(|m| m.getattr("FragmentMetadata"))
            .expect("FragmentMetadata class not found");

        let files = export_vec(py, &self.0.files)?;
        let deletion_file = self
            .0
            .deletion_file
            .as_ref()
            .map(|f| PyDeletionFile(f.clone()));
        let row_id_meta = self.0.row_id_meta.as_ref().map(|r| PyRowIdMeta(r.clone()));
        let last_updated_at_version_meta = self
            .0
            .last_updated_at_version_meta
            .as_ref()
            .map(|r| PyRowDatasetVersionMeta(r.clone()));
        let created_at_version_meta = self
            .0
            .created_at_version_meta
            .as_ref()
            .map(|r| PyRowDatasetVersionMeta(r.clone()));
        let overlays = export_vec(py, &self.0.overlays)?;

        cls.call1((
            self.0.id,
            files,
            self.0.physical_rows,
            deletion_file,
            row_id_meta,
            created_at_version_meta,
            last_updated_at_version_meta,
            overlays,
        ))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<Fragment> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyLance(&self.0).into_pyobject(py)
    }
}

impl FromPyObject<'_, '_> for PyLance<DataFile> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let file_size_bytes: Option<u64> = ob.getattr("file_size_bytes")?.extract()?;
        let file_size_bytes = CachedFileSize::new(file_size_bytes.unwrap_or(0));
        let fields: Vec<i32> = ob.getattr("fields")?.extract()?;
        let column_indices: Vec<i32> = ob.getattr("column_indices")?.extract()?;
        Ok(Self(DataFile {
            path: ob.getattr("path")?.extract()?,
            fields: fields.into(),
            column_indices: column_indices.into(),
            file_major_version: ob.getattr("file_major_version")?.extract()?,
            file_minor_version: ob.getattr("file_minor_version")?.extract()?,
            file_size_bytes,
            base_id: ob.getattr("base_id")?.extract()?,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&DataFile> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance.fragment"))
            .and_then(|m| m.getattr("DataFile"))
            .expect("DataFile class not found");

        let file_size_bytes = self.0.file_size_bytes.get().map(u64::from);
        cls.call1((
            &self.0.path,
            self.0.fields.to_vec(),
            self.0.column_indices.to_vec(),
            self.0.file_major_version,
            self.0.file_minor_version,
            file_size_bytes,
            self.0.base_id,
        ))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<DataFile> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyLance(&self.0).into_pyobject(py)
    }
}
