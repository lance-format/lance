// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_io::storage::StorageOptionsAccessor;
use pyo3::prelude::*;

use crate::rt;

/// Python wrapper for StorageOptionsAccessor
///
/// This wraps a Rust StorageOptionsAccessor and exposes it to Python.
#[pyclass(name = "StorageOptionsAccessor", skip_from_py_object)]
#[derive(Clone)]
pub struct PyStorageOptionsAccessor {
    inner: Arc<StorageOptionsAccessor>,
}

impl PyStorageOptionsAccessor {
    pub fn new(accessor: Arc<StorageOptionsAccessor>) -> Self {
        Self { inner: accessor }
    }

    pub fn inner(&self) -> Arc<StorageOptionsAccessor> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyStorageOptionsAccessor {
    /// Create an accessor with only static options (no refresh capability)
    #[staticmethod]
    fn with_static_options(options: HashMap<String, String>) -> Self {
        Self {
            inner: Arc::new(StorageOptionsAccessor::with_static_options(options)),
        }
    }

    /// Get current valid storage options
    fn get_storage_options(&self, py: Python<'_>) -> PyResult<HashMap<String, String>> {
        let accessor = self.inner.clone();
        let options = rt()
            .block_on(Some(py), accessor.get_storage_options())?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(options.0)
    }

    /// Get the initial storage options without refresh
    fn initial_storage_options(&self) -> Option<HashMap<String, String>> {
        self.inner.initial_storage_options().cloned()
    }

    /// Get the accessor ID for equality/hashing
    fn accessor_id(&self) -> String {
        self.inner.accessor_id()
    }

    /// Check if this accessor has a dynamic provider
    fn has_provider(&self) -> bool {
        self.inner.has_provider()
    }

    /// Get the refresh offset in seconds
    fn refresh_offset_secs(&self) -> u64 {
        self.inner.refresh_offset().as_secs()
    }

    fn __repr__(&self) -> String {
        format!(
            "StorageOptionsAccessor(id={}, has_provider={})",
            self.inner.accessor_id(),
            self.inner.has_provider()
        )
    }
}

/// Create a StorageOptionsAccessor from storage options
///
/// This creates an accessor with static options only.
#[allow(dead_code)]
pub fn create_accessor_from_storage_options(
    storage_options: Option<HashMap<String, String>>,
) -> PyResult<Option<Arc<StorageOptionsAccessor>>> {
    match storage_options {
        Some(opts) => Ok(Some(Arc::new(StorageOptionsAccessor::with_static_options(
            opts,
        )))),
        None => Ok(None),
    }
}
