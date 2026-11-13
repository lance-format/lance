// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python SQL query bindings.

use std::collections::HashMap;

use arrow::pyarrow::FromPyArrow;
use arrow_array::{RecordBatch, make_array};
use arrow_data::ArrayData;
use datafusion::common::{ParamValues, ScalarValue};
use lance::dataset::scanner::DatasetRecordBatchStream;
use lance_core::datatypes::BlobHandling;
use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyDict, PyList, PySequence, PyString};

use crate::error::PythonErrorExt;
use crate::rt;

use super::convert_reader;

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

impl SqlQueryBuilder {
    pub(super) fn new(builder: lance::dataset::sql::SqlQueryBuilder) -> Self {
        Self { builder }
    }
}

#[pymethods]
impl SqlQueryBuilder {
    fn parameters(&self, parameters: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = parameters.py();
        let pa = py.import("pyarrow")?;
        let array_factory = pa.getattr("array")?;
        let to_scalar = |value: Bound<'_, PyAny>| -> PyResult<ScalarValue> {
            let array = array_factory.call1((PyList::new(py, [value])?,))?;
            let array = make_array(ArrayData::from_pyarrow_bound(&array)?);
            ScalarValue::try_from_array(array.as_ref(), 0)
                .map_err(|err| PyValueError::new_err(err.to_string()))
        };
        let parameters: ParamValues = if let Ok(values) = parameters.cast::<PyDict>() {
            values
                .iter()
                .map(|(name, value)| Ok((name.extract::<String>()?, to_scalar(value)?)))
                .collect::<PyResult<HashMap<_, _>>>()?
                .into()
        } else if parameters.is_instance(&pa.getattr("RecordBatch")?)? {
            let batch = RecordBatch::from_pyarrow_bound(parameters)?;
            if batch.num_rows() != 1 {
                return Err(PyValueError::new_err(format!(
                    "SQL parameters RecordBatch must have exactly one row, got {}",
                    batch.num_rows()
                )));
            }
            let mut values = HashMap::with_capacity(batch.num_columns());
            for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
                let value = ScalarValue::try_from_array(column.as_ref(), 0)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                if values.insert(field.name().clone(), value).is_some() {
                    return Err(PyValueError::new_err(format!(
                        "SQL parameters RecordBatch has duplicate column name {:?}",
                        field.name()
                    )));
                }
            }
            values.into()
        } else if let Ok(values) = parameters.cast::<PySequence>()
            && !parameters.is_instance_of::<PyString>()
            && !parameters.is_instance_of::<PyBytes>()
            && !parameters.is_instance_of::<PyByteArray>()
        {
            values
                .try_iter()?
                .map(|value| to_scalar(value?))
                .collect::<PyResult<Vec<_>>>()?
                .into()
        } else {
            return Err(PyTypeError::new_err(
                "SQL parameters must be a sequence of values, a dict, or a single-row pyarrow.RecordBatch",
            ));
        };
        Ok(Self {
            builder: self.builder.clone().with_parameters(parameters),
        })
    }

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

    #[pyo3(signature = (blob_handling))]
    fn blob_handling(&self, blob_handling: &str) -> PyResult<Self> {
        let blob_handling = match blob_handling {
            "all_binary" => BlobHandling::AllBinary,
            "blobs_descriptions" => BlobHandling::BlobsDescriptions,
            "all_descriptions" => BlobHandling::AllDescriptions,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid blob_handling: {other}. Expected one of: all_binary, blobs_descriptions, all_descriptions"
                )));
            }
        };
        Ok(Self {
            builder: self.builder.clone().blob_handling(blob_handling),
        })
    }

    /// Register an in-memory Arrow relation that the SQL query can join.
    ///
    /// The relation is exposed under `name` and can be referenced like any other
    /// table in the query. The `data` may be a pyarrow `Table` or
    /// `RecordBatchReader` (ingested the same way as `add`/`merge_insert`), and
    /// must be non-empty since the schema is derived from its batches.
    /// The input is fully materialized in memory before registration, so it
    /// must be bounded to available memory.
    ///
    /// `name` is parsed as a SQL identifier, so an unquoted name is lowercased
    /// (`IDs` registers the table `ids`) while a quoted one keeps its case
    /// (`"IDs"`). If two names resolve to the same table, the later registration
    /// wins. A name that resolves to the query's own `table_name` would hide the
    /// dataset and is rejected.
    ///
    /// Registered names are validated when the query is executed, not when
    /// `build()` returns, so a rejected name raises from `to_batch_records()`
    /// (as `ValueError`) or `to_stream_reader()` (as `OSError`).
    #[pyo3(signature = (name, data))]
    fn register_arrow(&self, name: &str, data: &Bound<PyAny>) -> PyResult<Self> {
        let reader = convert_reader(data)?;
        let batches = reader
            .collect::<std::result::Result<Vec<RecordBatch>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let builder = self
            .builder
            .clone()
            .register_arrow(name, batches)
            .infer_error()?;
        Ok(Self { builder })
    }

    #[pyo3(signature = (batch_size))]
    fn batch_size(&self, batch_size: usize) -> Self {
        Self {
            builder: self.builder.clone().batch_size(batch_size),
        }
    }

    #[pyo3(signature = (batch_size_bytes))]
    fn batch_size_bytes(&self, batch_size_bytes: u64) -> Self {
        Self {
            builder: self.builder.clone().batch_size_bytes(batch_size_bytes),
        }
    }

    /// Build the SQL query.
    fn build(&self) -> PyResult<SqlQuery> {
        Ok(SqlQuery {
            builder: self.builder.clone(),
        })
    }
}
