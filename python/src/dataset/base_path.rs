// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python bindings for dataset base paths.

use lance_table::format::BasePath;
use pyo3::prelude::*;

/// Python binding for BasePath
#[pyclass(name = "DatasetBasePath", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct DatasetBasePath {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub is_dataset_root: bool,
}

#[pymethods]
impl DatasetBasePath {
    #[new]
    #[pyo3(signature = (path, name = None, is_dataset_root = false, id = None))]
    fn new(path: String, name: Option<String>, is_dataset_root: bool, id: Option<u32>) -> Self {
        Self {
            id: id.unwrap_or(0),
            name,
            path,
            is_dataset_root,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetBasePath(id={}, name={:?}, path={}, is_dataset_root={})",
            self.id, self.name, self.path, self.is_dataset_root
        )
    }
}

impl From<BasePath> for DatasetBasePath {
    fn from(base_path: BasePath) -> Self {
        Self {
            id: base_path.id,
            name: base_path.name,
            path: base_path.path,
            is_dataset_root: base_path.is_dataset_root,
        }
    }
}

impl From<DatasetBasePath> for BasePath {
    fn from(py_base_path: DatasetBasePath) -> Self {
        Self::new(
            py_base_path.id,
            py_base_path.path.clone(),
            py_base_path.name,
            py_base_path.is_dataset_root,
        )
    }
}
