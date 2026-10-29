// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance::dataset::{retention::SnapshotRetention, snapshot::SnapshotIdentity};
use pyo3::{prelude::*, types::PyBytes};

use crate::{dataset::Dataset, error::PythonErrorExt, rt};

pub(crate) fn dataset_inner(dataset: &Bound<'_, PyAny>) -> PyResult<Arc<lance::Dataset>> {
    Ok(dataset
        .getattr("_ds")?
        .extract::<PyRef<'_, Dataset>>()?
        .ds
        .clone())
}

/// Exact dataset snapshot identity independent of blob types and worker routing.
/// Encoding, decoding, and holding an identity do not protect files from cleanup.
#[pyclass(
    name = "SnapshotIdentity",
    module = "lance.retention",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySnapshotIdentity {
    pub(crate) inner: SnapshotIdentity,
}

#[pymethods]
impl PySnapshotIdentity {
    /// Capture the loaded snapshot. table_identity must distinguish table incarnations.
    #[staticmethod]
    fn from_dataset(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        table_identity: String,
    ) -> PyResult<Self> {
        let dataset = dataset_inner(dataset)?;
        let inner = rt()
            .block_on(Some(py), async move {
                SnapshotIdentity::from_dataset(&dataset, table_identity).await
            })?
            .infer_error()?;
        Ok(Self { inner })
    }
    /// Encode identity for another process or a durable acquisition-intent record.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
    }
    /// Validate a transported identity without acquiring retention.
    #[staticmethod]
    fn decode(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: SnapshotIdentity::decode(bytes).infer_error()?,
        })
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
    fn snapshot_id<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.snapshot_id())
    }
}

/// Explicit-release protection for ordinary scans and deferred blob reads.
///
/// Persist acquisition intent before acquire, save the token before issuing work,
/// and persist release intent after all consumers stop. Query tags do not expire.
/// Maintenance owners expire after 120 hours by default. Configure their lifetime
/// with dataset config ``lance.snapshot_retention.lock_expiration_hours``. Fixed
/// expiry requires bounded operations/storage requests and synchronized clocks;
/// it cannot fence an old process or cancel an already submitted storage request.
#[pyclass(name = "SnapshotRetention", module = "lance.retention", frozen)]
pub struct PySnapshotRetention {
    inner: SnapshotRetention,
}

#[pymethods]
impl PySnapshotRetention {
    /// Protect snapshot before issuing work. retention_id must identify one query lifetime.
    #[staticmethod]
    fn acquire(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        snapshot: &PySnapshotIdentity,
        retention_id: String,
    ) -> PyResult<Self> {
        let dataset = dataset_inner(dataset)?;
        let snapshot = snapshot.inner.clone();
        let inner = rt()
            .block_on(Some(py), async move {
                SnapshotRetention::acquire(&dataset, snapshot, retention_id).await
            })?
            .infer_error()?;
        Ok(Self { inner })
    }
    /// Serialize release ownership without acquiring or renewing protection.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
    }
    #[staticmethod]
    fn decode(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: SnapshotRetention::decode(bytes).infer_error()?,
        })
    }
    #[getter]
    fn snapshot(&self) -> PySnapshotIdentity {
        PySnapshotIdentity {
            inner: self.inner.snapshot().clone(),
        }
    }
    #[getter]
    fn tag_name(&self) -> &str {
        self.inner.tag_name()
    }
    /// Idempotently release after all readers, workers, and retries have stopped.
    fn release(&self, py: Python<'_>, dataset: &Bound<'_, PyAny>) -> PyResult<()> {
        let dataset = dataset_inner(dataset)?;
        let retention = self.inner.clone();
        rt().block_on(Some(py), async move { retention.release(&dataset).await })?
            .infer_error()
    }
}
