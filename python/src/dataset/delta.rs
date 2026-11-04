// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python dataset delta bindings.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::PythonErrorExt;
use crate::reader::LanceReader;
use crate::rt;
use crate::utils::PyLance;

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
    /// Get the row ids deleted between begin_version (exclusive) and end_version (inclusive) as a stream reader.
    ///
    /// Requires stable row ids on the dataset.
    fn get_deleted_row_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use arrow::pyarrow::IntoPyArrow;
        use arrow_array::RecordBatchReader;
        let stream = rt()
            .block_on(None, self.inner.get_deleted_row_ids())?
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

impl DatasetDeltaBuilder {
    pub(super) fn new(builder: lance::dataset::delta::DatasetDeltaBuilder) -> Self {
        Self { builder }
    }
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
