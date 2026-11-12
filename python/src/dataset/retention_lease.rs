// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python binding for retention leases.

use std::sync::{Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use lance::dataset::{LeaseKeeper, LeaseOptions, LeaseStatus, RetentionLease};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::PythonErrorExt;
use crate::{Dataset, rt};

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

/// Reads a `lance.LeaseOptions` dataclass.
pub(crate) fn extract_lease_options(options: &Bound<'_, PyAny>) -> PyResult<LeaseOptions> {
    Ok(LeaseOptions {
        ttl: options.getattr("ttl")?.extract()?,
        label: options.getattr("label")?.extract()?,
    })
}

enum KeeperState {
    Running(Box<LeaseKeeper>),
    Stopped {
        lease: RetentionLease,
        status: LeaseStatus,
    },
}

impl KeeperState {
    fn lease(&self) -> RetentionLease {
        match self {
            Self::Running(keeper) => keeper.lease(),
            Self::Stopped { lease, .. } => lease.clone(),
        }
    }

    fn status(&self) -> LeaseStatus {
        match self {
            Self::Running(keeper) => keeper.status(),
            Self::Stopped { status, .. } => status.clone(),
        }
    }

    /// Stops renewing, returning the keeper if it was still running.
    fn stop(&mut self) -> Option<Box<LeaseKeeper>> {
        let stopped = Self::Stopped {
            lease: self.lease(),
            status: self.status(),
        };
        match std::mem::replace(self, stopped) {
            Self::Running(keeper) => Some(keeper),
            Self::Stopped { .. } => None,
        }
    }
}

/// Renews a retention lease in the background while long-running work uses
/// it.
///
/// The keeper never interrupts the work: a failed renewal is retried, and a
/// lease that ends anyway is reported through :attr:`lost_reason`. Whether the
/// work can still be published is decided when it commits. Publishing a
/// result for the lease (for example with
/// :meth:`lance.LanceDataset.commit_existing_index_segments` or
/// :meth:`lance.Transaction.for_lease`) releases it.
///
/// Used as a context manager, the keeper stops renewing on exit without
/// releasing the lease. Call :meth:`release` to abandon the work instead.
///
/// Parameters
/// ----------
/// dataset: LanceDataset
///     A dataset on the lease's branch, typically the leased dataset returned
///     by :meth:`lance.LanceDataset.acquire_retention_lease`.
/// lease: RetentionLease
///     The lease to renew. Each renewal extends it by its original time to
///     live, once half of the remaining time has passed.
///
/// Examples
/// --------
/// >>> import lance
/// >>> import pyarrow as pa
/// >>> dataset = lance.write_dataset(pa.table({"a": [1, 2]}), "example")
/// >>> lease, leased = dataset.acquire_retention_lease()
/// >>> keeper = lance.LeaseKeeper(leased, lease)
/// >>> # ... long-running work reading `leased` ...
/// >>> keeper.lost_reason is None
/// True
/// >>> keeper.release()
#[pyclass(name = "LeaseKeeper", module = "lance", frozen)]
pub struct PyLeaseKeeper {
    state: Mutex<KeeperState>,
}

impl PyLeaseKeeper {
    fn state(&self) -> MutexGuard<'_, KeeperState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[pymethods]
impl PyLeaseKeeper {
    #[new]
    fn new(dataset: &Bound<'_, PyAny>, lease: PyRef<'_, PyRetentionLease>) -> PyResult<Self> {
        let dataset = dataset.getattr("_ds")?.extract::<PyRef<'_, Dataset>>()?;
        // `LeaseKeeper::start` spawns its renewal task on the current runtime.
        let _runtime = rt().runtime.enter();
        let keeper = LeaseKeeper::start(&dataset.ds, lease.inner.clone());
        Ok(Self {
            state: Mutex::new(KeeperState::Running(Box::new(keeper))),
        })
    }

    /// The lease as of its latest renewal.
    #[getter]
    fn lease(&self) -> PyRetentionLease {
        self.state().lease().into()
    }

    /// Why the lease was lost, or ``None`` while it is held.
    ///
    /// A lost lease expired or was removed, so the keeper stopped renewing
    /// it. Publishing a result for it still commits normally, but cleanup and
    /// compaction may have removed what the lease retained, in which case the
    /// commit fails with a conflict.
    #[getter]
    fn lost_reason(&self) -> Option<String> {
        match self.state().status() {
            LeaseStatus::Held => None,
            LeaseStatus::Lost { reason } => Some(reason),
        }
    }

    /// Stop renewing and release the lease, for example when the work is
    /// abandoned. Not needed after publishing, since publication releases the
    /// lease.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the keeper has already stopped.
    fn release(&self, py: Python<'_>) -> PyResult<()> {
        let keeper = self.state().stop().ok_or_else(|| {
            PyValueError::new_err(
                "LeaseKeeper has already stopped; release its lease with \
                 LanceDataset.release_retention_lease(keeper.lease)",
            )
        })?;
        rt().block_on(Some(py), keeper.release())?.infer_error()
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Stop renewing without releasing the lease.
    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, PyTuple>) -> bool {
        self.state().stop();
        false
    }

    fn __repr__(&self) -> String {
        let state = self.state();
        let lease = state.lease();
        let status = match state.status() {
            LeaseStatus::Held => "held".to_string(),
            LeaseStatus::Lost { reason } => format!("lost: {reason}"),
        };
        format!(
            "LeaseKeeper(lease_id='{}', version={}, expires_at='{}', status={status:?})",
            lease.id(),
            lease.version(),
            lease.expires_at().to_rfc3339(),
        )
    }
}
