// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex};

use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{Array, cast::AsArray, make_array, types::UInt64Type};
use arrow_data::ArrayData;
use arrow_schema::Field;
use lance::blob::reference::{
    BLOB_REFERENCE_CAPABILITIES, BLOB_REFERENCE_OPERATIONS, BLOB_REFERENCE_PROTOCOL_VERSION,
    BlobMaterializationContext, BlobReadRange, BlobRefBatch, BlobRefBinding, BlobRefCapture,
    BlobRefReadSession, BlobRefReader, BlobSnapshotRetention, BlobSourceContext,
};
use lance_core::utils::address::RowAddress;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use tokio::sync::Mutex as AsyncMutex;

use crate::{
    error::PythonErrorExt,
    rt,
    snapshot_retention::{PySnapshotIdentity, dataset_inner},
};

/// Exact snapshot identity and authenticated worker context key, without credentials.
#[pyclass(
    name = "BlobSourceContext",
    module = "lance.blob",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyBlobSourceContext {
    inner: BlobSourceContext,
}

#[pymethods]
impl PyBlobSourceContext {
    /// Snapshot identity shared with ordinary scans, independent of blob reader settings.
    fn snapshot_identity(&self) -> PyResult<PySnapshotIdentity> {
        Ok(PySnapshotIdentity {
            inner: self.inner.snapshot_identity().infer_error()?,
        })
    }
    /// Capture metadata from a resolved LanceDataset, without payload reads or retention.
    ///
    /// table_identity must distinguish table incarnations and catalog/storage
    /// authorities. access_context_key must resolve authenticated storage and
    /// per-base routing independently on every worker. Neither may be empty.
    /// assume_immutable_external accepts the caller's external-content guarantee;
    /// it does not enforce object versions or conditional reads.
    #[staticmethod]
    #[pyo3(signature=(dataset, table_identity, access_context_key, *, assume_immutable_external=false))]
    fn from_dataset(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        table_identity: String,
        access_context_key: String,
        assume_immutable_external: bool,
    ) -> PyResult<Self> {
        let dataset = dataset_inner(dataset)?;
        let mut inner = rt()
            .block_on(Some(py), async move {
                BlobSourceContext::from_dataset(&dataset, table_identity, access_context_key).await
            })?
            .infer_error()?;
        if assume_immutable_external {
            inner = inner.with_external_immutability_assumption();
        }
        Ok(Self { inner })
    }

    /// Encode for exchange, spill, or a separate worker process.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
    }

    /// Reject unsupported versions and capabilities before scheduling reads.
    #[staticmethod]
    fn decode(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: BlobSourceContext::decode(bytes).infer_error()?,
        })
    }

    #[staticmethod]
    fn protocol_version() -> u32 {
        BLOB_REFERENCE_PROTOCOL_VERSION
    }

    #[staticmethod]
    fn supported_capabilities() -> Vec<u32> {
        BLOB_REFERENCE_CAPABILITIES.to_vec()
    }

    #[staticmethod]
    fn supported_operations() -> Vec<&'static str> {
        BLOB_REFERENCE_OPERATIONS.to_vec()
    }

    #[getter]
    fn table_identity(&self) -> &str {
        self.inner.table_identity()
    }
    #[getter]
    fn table_uri(&self) -> &str {
        self.inner.table_uri()
    }
    #[getter]
    fn branch(&self) -> Option<&str> {
        self.inner.branch()
    }
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version()
    }
    #[getter]
    fn access_context_key(&self) -> &str {
        self.inner.access_context_key()
    }
}

/// Nullable Arrow references and their portable source context table.
#[pyclass(name = "BlobRefBatch", module = "lance.blob", frozen)]
pub struct PyBlobRefBatch {
    inner: BlobRefBatch,
}

#[pymethods]
impl PyBlobRefBatch {
    #[staticmethod]
    fn decode(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: BlobRefBatch::decode(bytes).infer_error()?,
        })
    }

    /// Construct from a transported Arrow field/array and its context table.
    #[staticmethod]
    fn from_arrow(
        py: Python<'_>,
        contexts: Vec<Py<PyBlobSourceContext>>,
        field: &Bound<'_, PyAny>,
        array: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let field = Field::from_pyarrow_bound(field)?;
        let array = make_array(ArrayData::from_pyarrow_bound(array)?);
        let array = array
            .as_struct_opt()
            .ok_or_else(|| PyValueError::new_err("Blob reference array must be a struct"))?;
        let contexts = contexts
            .into_iter()
            .map(|context| context.borrow(py).inner.clone())
            .collect();
        Ok(Self {
            inner: BlobRefBatch::try_from_arrow(contexts, &field, array.clone()).infer_error()?,
        })
    }

    fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.inner.encode().infer_error()?))
    }
    #[getter]
    fn array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.array().to_data().to_pyarrow(py)
    }
    #[getter]
    fn field<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        lance::blob::reference::blob_ref_field("blob_ref").to_pyarrow(py)
    }
    #[getter]
    fn contexts(&self) -> Vec<PyBlobSourceContext> {
        self.inner
            .contexts()
            .iter()
            .cloned()
            .map(|inner| PyBlobSourceContext { inner })
            .collect()
    }
    fn __len__(&self) -> usize {
        self.inner.array().len()
    }

    /// Remap independently produced context slots before combining Arrow arrays.
    fn remap_contexts(
        &self,
        py: Python<'_>,
        contexts: Vec<Py<PyBlobSourceContext>>,
    ) -> PyResult<Self> {
        let contexts = contexts
            .into_iter()
            .map(|context| context.borrow(py).inner.clone())
            .collect();
        Ok(Self {
            inner: self.inner.remap_contexts(contexts).infer_error()?,
        })
    }
}

/// Reuse scanned Blob V2 descriptors; capture performs no payload reads or HEADs.
#[pyclass(name = "BlobRefCapture", module = "lance.blob")]
pub struct PyBlobRefCapture {
    inner: Arc<AsyncMutex<Option<BlobRefCapture>>>,
}

#[pymethods]
impl PyBlobRefCapture {
    #[new]
    fn new(
        dataset: &Bound<'_, PyAny>,
        context: &PyBlobSourceContext,
        field_id: i32,
    ) -> PyResult<Self> {
        let capture =
            BlobRefCapture::try_new(dataset_inner(dataset)?, context.inner.clone(), field_id)
                .infer_error()?;
        Ok(Self {
            inner: Arc::new(AsyncMutex::new(Some(capture))),
        })
    }

    /// Preserve the original scan field and physical uint64 _rowaddr array.
    fn capture(
        &self,
        py: Python<'_>,
        scan_source: &PyBlobSourceContext,
        field: &Bound<'_, PyAny>,
        row_addresses: &Bound<'_, PyAny>,
        descriptors: &Bound<'_, PyAny>,
    ) -> PyResult<PyBlobRefBatch> {
        let field = Field::from_pyarrow_bound(field)?;
        let addresses = make_array(ArrayData::from_pyarrow_bound(row_addresses)?);
        let addresses = addresses
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| PyValueError::new_err("row_addresses must be a uint64 Arrow array"))?;
        if addresses.null_count() != 0 {
            return Err(PyValueError::new_err(
                "row_addresses must not contain nulls",
            ));
        }
        let addresses = addresses
            .values()
            .iter()
            .copied()
            .map(RowAddress::from)
            .collect::<Vec<_>>();
        let descriptors = make_array(ArrayData::from_pyarrow_bound(descriptors)?);
        let descriptors = descriptors
            .as_struct_opt()
            .ok_or_else(|| PyValueError::new_err("descriptors must be a struct Arrow array"))?
            .clone();
        let scan_source = scan_source.inner.clone();
        let capture = self.inner.clone();
        let inner = rt()
            .block_on(Some(py), async move {
                capture
                    .lock()
                    .await
                    .as_mut()
                    .ok_or_else(|| lance::Error::invalid_input("BlobRefCapture is closed"))?
                    .capture(&scan_source, &field, &addresses, &descriptors)
                    .await
            })?
            .infer_error()?;
        Ok(PyBlobRefBatch { inner })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        rt().block_on(Some(py), async move {
            inner.lock().await.take();
        })
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __exit__(
        &self,
        py: Python<'_>,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.close(py)
    }
}

/// Native planned reader sharing finite, nonzero budgets across its read streams.
///
/// One oversized demand may run alone when idle; these are not hard per-value
/// caps. Returned Arrow arrays retain native reservations until released. Copies
/// made by callers require separate accounting. Close streams independently of
/// the factory; closing the factory only prevents new submissions.
#[pyclass(name = "BlobRefReader", module = "lance.blob")]
pub struct PyBlobRefReader {
    reader: Mutex<Option<Arc<BlobRefReader>>>,
    context: Arc<BlobMaterializationContext>,
}

#[pymethods]
impl PyBlobRefReader {
    #[new]
    #[pyo3(signature=(bindings, *, io_buffer_size_bytes, materialization_readahead_bytes))]
    fn new(
        py: Python<'_>,
        bindings: Vec<(Py<PyBlobSourceContext>, Bound<'_, PyAny>)>,
        io_buffer_size_bytes: u64,
        materialization_readahead_bytes: u64,
    ) -> PyResult<Self> {
        let bindings = bindings
            .into_iter()
            .map(|(context, dataset)| {
                BlobRefBinding::try_new(context.borrow(py).inner.clone(), dataset_inner(&dataset)?)
                    .infer_error()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let context = BlobMaterializationContext::try_new(
            io_buffer_size_bytes,
            materialization_readahead_bytes,
        )
        .infer_error()?;
        let reader = Arc::new(BlobRefReader::try_new(bindings, context.clone()).infer_error()?);
        Ok(Self {
            reader: Mutex::new(Some(reader)),
            context,
        })
    }

    /// Submit a finite request set. None ranges read whole values; tuples are (offset, length).
    #[pyo3(signature=(references, reference_indices, request_indices, ranges=None))]
    fn read(
        &self,
        references: &PyBlobRefBatch,
        reference_indices: Vec<u64>,
        request_indices: Vec<u64>,
        ranges: Option<Vec<Option<(u64, u64)>>>,
    ) -> PyResult<PyBlobRefReadStream> {
        let ranges = ranges
            .unwrap_or_else(|| vec![None; reference_indices.len()])
            .into_iter()
            .map(|range| range.map(|(offset, length)| BlobReadRange::new(offset, length)))
            .collect::<Vec<_>>();
        let requests = references
            .inner
            .read_requests(&reference_indices, &request_indices, &ranges)
            .infer_error()?;
        let reader = self
            .reader
            .lock()
            .map_err(|_| PyRuntimeError::new_err("BlobRefReader lock is poisoned"))?;
        let reader = reader
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("BlobRefReader is closed"))?;
        Ok(PyBlobRefReadStream {
            inner: Arc::new(BlobRefReadSession::try_new(reader, requests).infer_error()?),
        })
    }
    #[getter]
    fn reserved_bytes(&self) -> PyResult<u64> {
        self.context.reserved_bytes().infer_error()
    }
    #[getter]
    fn io_reserved_bytes(&self) -> PyResult<u64> {
        self.context.io_reserved_bytes().infer_error()
    }
    /// Close the factory. Existing streams and returned Arrow arrays retain their owners.
    fn close(&self) -> PyResult<()> {
        self.reader
            .lock()
            .map_err(|_| PyRuntimeError::new_err("BlobRefReader lock is poisoned"))?
            .take();
        Ok(())
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __exit__(
        &self,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.close()
    }
}

/// Iterator of one-request Arrow batches retaining native memory reservations.
#[pyclass(name = "BlobRefReadStream", module = "lance.blob")]
pub struct PyBlobRefReadStream {
    inner: Arc<BlobRefReadSession>,
}

#[pymethods]
impl PyBlobRefReadStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let inner = self.inner.clone();
        rt().block_on(Some(py), async move { inner.next_batch().await })?
            .transpose()
            .infer_error()?
            .map(|batch| batch.to_pyarrow(py))
            .transpose()
    }
    fn cancel(&self) {
        self.inner.cancel();
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        rt().block_on(Some(py), async move { inner.close().await })
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __exit__(
        &self,
        py: Python<'_>,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.close(py)
    }
}

/// Explicit-release exact-snapshot GC root requiring coordinated maintenance clients.
///
/// Acquire before exporting refs and release after every worker, retry, read, and
/// lazy consumer has stopped. Query tags have no expiry or renewal; maintenance owners have fixed expiration. Foreign dataset
/// roots are unsupported; external object lifetime remains the caller's contract.
/// Dropping the token does not release it. See the portable blob references guide
/// for deployment requirements and orphan recovery.
///
/// Persist the query ID and encoded context before acquisition, save the token
/// before exporting refs, and record release intent before release. Recover a lost
/// release response by retrying release with that saved token.
#[pyclass(name = "BlobSnapshotRetention", module = "lance.blob", frozen)]
pub struct PyBlobSnapshotRetention {
    inner: BlobSnapshotRetention,
}

#[pymethods]
impl PyBlobSnapshotRetention {
    /// Acquire before issuing references. retention_id must be unique to this query.
    #[staticmethod]
    fn acquire(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        context: &PyBlobSourceContext,
        retention_id: String,
    ) -> PyResult<Self> {
        let dataset = dataset_inner(dataset)?;
        let context = context.inner.clone();
        let inner = rt()
            .block_on(Some(py), async move {
                BlobSnapshotRetention::acquire(&dataset, context, retention_id).await
            })?
            .infer_error()?;
        Ok(Self { inner })
    }
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
    }
    #[staticmethod]
    fn decode(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: BlobSnapshotRetention::decode(bytes).infer_error()?,
        })
    }
    #[getter]
    fn context(&self) -> PyBlobSourceContext {
        PyBlobSourceContext {
            inner: self.inner.context().clone(),
        }
    }
    #[getter]
    fn tag_name(&self) -> &str {
        self.inner.tag_name()
    }
    /// Idempotent. Call after every worker, retry, and outstanding read has stopped.
    fn release(&self, py: Python<'_>, dataset: &Bound<'_, PyAny>) -> PyResult<()> {
        let dataset = dataset_inner(dataset)?;
        let inner = self.inner.clone();
        rt().block_on(Some(py), async move { inner.release(&dataset).await })?
            .infer_error()
    }
}
