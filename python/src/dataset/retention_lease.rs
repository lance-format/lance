// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python binding for retention leases.

use chrono::{DateTime, Utc};
use lance::dataset::RetentionLease;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A retention lease: it keeps its leased version and every later version of
/// the dataset until the holder publishes a transaction for it, releases it,
/// or lets it expire.
///
/// Leases are obtained from :meth:`lance.LanceDataset.acquire_retention_lease`
/// and are picklable, so the holder can ship them to distributed workers.
#[pyclass(
    name = "RetentionLease",
    module = "lance",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyRetentionLease {
    pub(crate) inner: RetentionLease,
}

impl From<RetentionLease> for PyRetentionLease {
    fn from(inner: RetentionLease) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRetentionLease {
    /// The UUID a transaction must carry to publish this lease's result.
    #[getter]
    fn id(&self) -> String {
        self.inner.id().hyphenated().to_string()
    }

    /// The leased version: the version created by the acquisition commit.
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version()
    }

    /// When the lease was acquired, by the acquirer's clock.
    #[getter]
    fn created_at(&self) -> DateTime<Utc> {
        self.inner.created_at()
    }

    /// When the lease expires, by the clock of the acquirer or last renewer.
    #[getter]
    fn expires_at(&self) -> DateTime<Utc> {
        self.inner.expires_at()
    }

    /// Human-readable context stored with the lease, if any.
    #[getter]
    fn label(&self) -> Option<&str> {
        self.inner.label()
    }

    fn __repr__(&self) -> String {
        let label = match self.inner.label() {
            Some(label) => format!("{label:?}"),
            None => "None".to_string(),
        };
        format!(
            "RetentionLease(id='{}', version={}, created_at='{}', expires_at='{}', label={label})",
            self.inner.id(),
            self.inner.version(),
            self.inner.created_at().to_rfc3339(),
            self.inner.expires_at().to_rfc3339(),
        )
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let json = serde_json::to_string(&self.inner)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let ctor = py
            .import("lance")?
            .getattr("RetentionLease")?
            .getattr("_from_json")?
            .unbind();
        Ok((ctor, (json,)))
    }

    /// Reconstruct a lease from its serialized form; Rust validates the record.
    #[staticmethod]
    fn _from_json(json: &str) -> PyResult<Self> {
        serde_json::from_str::<RetentionLease>(json)
            .map(Self::from)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}
