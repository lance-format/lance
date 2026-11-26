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

use std::fmt::Debug;
use std::sync::LazyLock;

use lance_table::io::commit::{CommitError, CommitLease, CommitLock};

use lance_core::Error;

use pyo3::{exceptions::PyIOError, prelude::*};

static PY_CONFLICT_ERROR: LazyLock<PyResult<Py<PyAny>>> = LazyLock::new(|| {
    Python::attach(|py| {
        py.import("lance.commit")
            .and_then(|commit| commit.getattr("CommitConflictError"))
            .map(|err| err.unbind())
    })
});

fn handle_error(py_err: PyErr, py: Python) -> CommitError {
    // The cached object is the class itself. `get_type()` would return its metaclass,
    // which no exception instance passes, and the conflict arm would never be taken.
    let conflict_err_type = match &*PY_CONFLICT_ERROR {
        Ok(err) => err.bind(py).clone(),
        Err(import_error) => {
            return CommitError::OtherError(Error::internal(format!(
                "Error importing from pylance {}",
                import_error
            )));
        }
    };

    if py_err.is_instance(py, &conflict_err_type) {
        // `retryable=False` means retrying cannot help, so it must not be retried.
        let retryable = py_err
            .value(py)
            .getattr("retryable")
            .and_then(|flag| flag.is_truthy())
            .unwrap_or(true);
        if retryable {
            CommitError::CommitConflict
        } else {
            CommitError::OtherError(Error::incompatible_transaction_source(
                py_err.to_string().into(),
            ))
        }
    } else {
        handler_error(py_err)
    }
}

/// Errors from releasing a lock are not conflicts: the commit may already have
/// landed, so they must not trigger a retry or report the commit as rejected.
fn handler_error(py_err: PyErr) -> CommitError {
    CommitError::OtherError(Error::internal(format!(
        "Error from commit handler: {}",
        py_err
    )))
}

pub struct PyCommitLock {
    inner: Py<PyAny>,
}

impl PyCommitLock {
    pub fn new(inner: Py<PyAny>) -> Self {
        Self { inner }
    }
}

impl Debug for PyCommitLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let repr = Python::attach(|py| {
            self.inner
                .call_method0(py, "__repr__")?
                .extract::<String>(py)
        })
        .ok();
        f.debug_struct("PyCommitLock")
            .field("inner", &repr)
            .finish()
    }
}

#[async_trait::async_trait]
impl CommitLock for PyCommitLock {
    type Lease = PyCommitLease;

    async fn lock(&self, version: u64) -> Result<Self::Lease, CommitError> {
        let lease = Python::attach(|py| -> Result<_, CommitError> {
            let lease = self
                .inner
                .call1(py, (version,))
                .map_err(|err| handle_error(err, py))?;
            lease
                .call_method0(py, "__enter__")
                .map_err(|err| handle_error(err, py))?;
            Ok(lease)
        })?;
        Ok(PyCommitLease { inner: lease })
    }
}

pub struct PyCommitLease {
    inner: Py<PyAny>,
}

#[async_trait::async_trait]
impl CommitLease for PyCommitLease {
    async fn release(&self, success: bool) -> Result<(), CommitError> {
        Python::attach(|py| {
            if success {
                self.inner
                    .call_method1(py, "__exit__", (py.None(), py.None(), py.None()))
                    .map_err(handler_error)
            } else {
                // Setting Python's pending error indicator would interfere
                // with calling __exit__; pass the exception directly instead.
                let err = PyIOError::new_err("commit failed");
                self.inner
                    .call_method1(
                        py,
                        "__exit__",
                        (err.get_type(py), err.value(py), err.traceback(py)),
                    )
                    .map_err(handler_error)
            }
        })?;
        Ok(())
    }
}
