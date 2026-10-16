// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python bindings for Lance Namespace implementations

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use lance_namespace::LanceNamespace as LanceNamespaceTrait;
use lance_namespace::compat::merge_insert_request_from_json;
use lance_namespace::models::{
    AlterTableAddColumnsRequest, AlterTableAlterColumnsRequest, AlterTableBackfillColumnsRequest,
    AlterTableDropColumnsRequest, AlterTransactionRequest, AnalyzeTableQueryPlanRequest,
    CountTableRowsRequest, CreateMaterializedViewRequest, CreateTableIndexRequest,
    CreateTableTagRequest, CreateTableVersionRequest, CreateTableVersionResponse,
    DeleteFromTableRequest, DeleteTableTagRequest, DescribeTableIndexStatsRequest,
    DescribeTableRequest, DescribeTableResponse, DescribeTableVersionRequest,
    DescribeTableVersionResponse, DescribeTransactionRequest, DropTableIndexRequest,
    ExplainTableQueryPlanRequest, GetTableStatsRequest, GetTableTagVersionRequest,
    InsertIntoTableRequest, ListTableIndicesRequest, ListTableTagsRequest,
    ListTableVersionsRequest, ListTableVersionsResponse, ListTablesRequest, QueryTableRequest,
    RefreshMaterializedViewRequest, RestoreTableRequest, UpdateTableRequest,
    UpdateTableSchemaMetadataRequest, UpdateTableTagRequest,
};
use lance_namespace_impls::RestNamespaceBuilder;
use lance_namespace_impls::{ConnectBuilder, RestAdapter, RestAdapterConfig, RestAdapterHandle};
use lance_namespace_impls::{
    DirectoryNamespace, DirectoryNamespaceBuilder, DynamicContextProvider, OperationInfo,
    RestNamespace,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pythonize::{depythonize, pythonize};
use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::error::PythonErrorExt;
use crate::session::Session;

/// Python-implemented dynamic context provider.
///
/// Wraps a Python object that has a `provide_context(info: dict) -> dict` method.
/// For RestNamespace, context keys that start with `headers.` are converted to
/// HTTP headers by stripping the prefix.
pub struct PyDynamicContextProvider {
    provider: Py<PyAny>,
}

impl Clone for PyDynamicContextProvider {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            provider: self.provider.clone_ref(py),
        })
    }
}

impl PyDynamicContextProvider {
    /// Create a new Python context provider wrapper.
    pub fn new(provider: Py<PyAny>) -> Self {
        Self { provider }
    }
}

impl std::fmt::Debug for PyDynamicContextProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PyDynamicContextProvider")
    }
}

impl DynamicContextProvider for PyDynamicContextProvider {
    fn provide_context(&self, info: &OperationInfo) -> HashMap<String, String> {
        Python::attach(|py| {
            // Create Python dict for operation info
            let py_info = PyDict::new(py);
            if py_info.set_item("operation", &info.operation).is_err() {
                return HashMap::new();
            }
            if py_info.set_item("object_id", &info.object_id).is_err() {
                return HashMap::new();
            }

            // Call the provider's provide_context method
            let result = self
                .provider
                .call_method1(py, "provide_context", (py_info,));

            match result {
                Ok(headers_py) => {
                    // Convert Python dict to Rust HashMap
                    let bound_headers = headers_py.bind(py);
                    if let Ok(dict) = bound_headers.cast::<PyDict>() {
                        dict_to_hashmap(dict).unwrap_or_default()
                    } else {
                        log::warn!("Context provider did not return a dict");
                        HashMap::new()
                    }
                }
                Err(e) => {
                    log::error!("Failed to call context provider: {}", e);
                    HashMap::new()
                }
            }
        })
    }
}

/// Convert Python dict to HashMap<String, String>
fn dict_to_hashmap(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut map = HashMap::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let value_str: String = value.extract()?;
        map.insert(key_str, value_str);
    }
    Ok(map)
}

/// Python wrapper for DirectoryNamespace
#[pyclass(name = "PyDirectoryNamespace", module = "lance.lance")]
pub struct PyDirectoryNamespace {
    pub(crate) inner: Arc<DirectoryNamespace>,
}

#[pymethods]
impl PyDirectoryNamespace {
    /// Create a new DirectoryNamespace from properties
    ///
    /// # Arguments
    ///
    /// * `session` - Optional Lance session for sharing storage connections
    /// * `context_provider` - Optional object with `provide_context(info: dict) -> dict` method
    ///   for providing dynamic per-request context
    /// * `**properties` - Namespace configuration properties
    #[new]
    #[pyo3(signature = (session = None, context_provider = None, **properties))]
    fn new(
        session: Option<&Bound<'_, Session>>,
        context_provider: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = properties {
            props = dict_to_hashmap(dict)?;
        }

        let session_arc = session.map(|s| s.borrow().inner.clone());

        let mut builder =
            DirectoryNamespaceBuilder::from_properties(props, session_arc).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to create DirectoryNamespace: {}",
                    e
                ))
            })?;

        // Add context provider if provided
        if let Some(provider) = context_provider {
            let py_provider = PyDynamicContextProvider::new(provider.clone().unbind());
            builder = builder.context_provider(Arc::new(py_provider));
        }

        let namespace = crate::rt().block_on(None, builder.build())?.infer_error()?;

        Ok(Self {
            inner: Arc::new(namespace),
        })
    }

    /// Get the namespace ID
    fn namespace_id(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("PyDirectoryNamespace({})", self.namespace_id())
    }

    // Namespace operations

    fn list_namespaces<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_namespaces(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn describe_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_namespace(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn create_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn drop_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn namespace_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.namespace_exists(request))?
            .infer_error()?;
        Ok(())
    }

    // Table operations

    fn list_tables<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_tables(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn register_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.register_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn table_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.table_exists(request))?
            .infer_error()?;
        Ok(())
    }

    fn drop_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn deregister_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.deregister_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table(request, data))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn declare_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.declare_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn rename_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.rename_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    // Table version operations

    fn list_table_versions<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_versions(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn batch_delete_table_versions<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.batch_delete_table_versions(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Table branch operations

    fn create_table_branch<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_branch(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn list_table_branches<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_branches(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn delete_table_branch<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.delete_table_branch(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Data manipulation operations

    fn count_table_rows(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<i64> {
        let request: CountTableRowsRequest = depythonize(request)?;
        let count = crate::rt()
            .block_on(Some(py), self.inner.count_table_rows(request))?
            .infer_error()?;
        Ok(count)
    }

    fn insert_into_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: InsertIntoTableRequest = depythonize(request)?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.insert_into_table(request, data))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn merge_insert_into_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = merge_insert_request_from_json(depythonize(request)?).infer_error()?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.merge_insert_into_table(request, data))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn update_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: UpdateTableRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.update_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn delete_from_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DeleteFromTableRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.delete_from_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn query_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let request: QueryTableRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.query_table(request))?
            .infer_error()?;
        Ok(PyBytes::new(py, &response))
    }

    // Index operations

    fn create_table_index<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: CreateTableIndexRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_index(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn list_table_indices<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: ListTableIndicesRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_indices(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table_index_stats<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DescribeTableIndexStatsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table_index_stats(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Transaction operations

    fn describe_transaction<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DescribeTransactionRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_transaction(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn alter_transaction<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTransactionRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_transaction(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Additional index operations

    fn create_table_scalar_index<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: CreateTableIndexRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_scalar_index(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn drop_table_index<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DropTableIndexRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_table_index(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Additional table operations

    fn list_all_tables<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: ListTablesRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_all_tables(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn restore_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: RestoreTableRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.restore_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn update_table_schema_metadata<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: UpdateTableSchemaMetadataRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.update_table_schema_metadata(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn get_table_stats<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: GetTableStatsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.get_table_stats(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Query plan operations

    fn explain_table_query_plan(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<String> {
        let request: ExplainTableQueryPlanRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.explain_table_query_plan(request))?
            .infer_error()?;
        Ok(response)
    }

    fn analyze_table_query_plan(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<String> {
        let request: AnalyzeTableQueryPlanRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.analyze_table_query_plan(request))?
            .infer_error()?;
        Ok(response)
    }

    // Column alteration operations

    fn alter_table_add_columns<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTableAddColumnsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_table_add_columns(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn alter_table_alter_columns<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTableAlterColumnsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_table_alter_columns(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn alter_table_drop_columns<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTableDropColumnsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_table_drop_columns(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn alter_table_backfill_columns<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTableBackfillColumnsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_table_backfill_columns(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn refresh_materialized_view<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: RefreshMaterializedViewRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.refresh_materialized_view(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_materialized_view<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: CreateMaterializedViewRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_materialized_view(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Table tag operations

    fn list_table_tags<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: ListTableTagsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_tags(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn get_table_tag_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: GetTableTagVersionRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.get_table_tag_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table_tag<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: CreateTableTagRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_tag(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn delete_table_tag<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DeleteTableTagRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.delete_table_tag(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn update_table_tag<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: UpdateTableTagRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.update_table_tag(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Operation metrics methods

    /// Retrieve operation metrics as a dictionary.
    ///
    /// Returns a dict where keys are operation names (e.g., "list_tables", "describe_table")
    /// and values are the number of times each operation was called.
    ///
    /// Returns an empty dict if `ops_metrics_enabled` was false when creating the namespace.
    fn retrieve_ops_metrics(&self) -> HashMap<String, u64> {
        self.inner.retrieve_ops_metrics()
    }

    /// Reset all operation metrics counters to zero.
    ///
    /// Does nothing if `ops_metrics_enabled` was false when creating the namespace.
    fn reset_ops_metrics(&self) {
        self.inner.reset_ops_metrics()
    }
}

/// A generated request struct plus whatever the caller's Python object set
/// that the struct has no field for -- e.g. a newer Geneva tuning kwarg the
/// namespace schema hasn't caught up to yet.
///
/// Deserializing straight into `T` (plain `depythonize`) silently drops
/// those extras -- there's no `deny_unknown_fields` and no flatten
/// catch-all on the generated namespace request models, so a knob like
/// that just disappears before the request is even built. This combined
/// struct recovers it: `#[serde(flatten)]` on both fields makes serde
/// attribute every key either to `typed` (if `T` declares it) or to
/// `extra` (everything else), in one deserialization pass.
///
/// That's deliberately not "deserialize `T`, then diff its own
/// *serialization* against the raw object to see what's left over" --
/// the generated structs mark every optional field
/// `skip_serializing_if = "Option::is_none"`, so an unset field is
/// *absent* from `T`'s own serialized JSON but *present* (as `null`) in
/// the caller's raw object. A diff-based approach reads that mismatch as
/// "extra" and sends the field back as an explicit `null` on the wire --
/// which some namespace implementations reject for a field that's
/// optional (may be absent) but not nullable (may not be `null`).
/// Flatten-based deserialization doesn't have this problem: it's about
/// which keys `T::deserialize` recognizes, independent of how `T` itself
/// would serialize.
///
/// Also deliberately not "read the caller's object's `model_extra`
/// directly" -- by the time a request reaches here, the Python-level
/// `RestNamespace` wrapper (`python/lance/namespace.py`) has already
/// called `request.model_dump()` on it, so `py_request` is a plain dict,
/// not a live pydantic model; `model_extra` isn't available to read.
#[derive(Deserialize)]
struct WithExtraOptions<T> {
    #[serde(flatten)]
    typed: T,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

fn depythonize_with_extras<T: DeserializeOwned>(
    py_request: &Bound<'_, PyAny>,
) -> PyResult<(T, serde_json::Map<String, serde_json::Value>)> {
    let raw: serde_json::Value = depythonize(py_request)?;
    let combined: WithExtraOptions<T> = serde_json::from_value(raw)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((combined.typed, combined.extra))
}

/// Python wrapper for RestNamespace
#[pyclass(name = "PyRestNamespace", module = "lance.lance")]
pub struct PyRestNamespace {
    pub(crate) inner: Arc<RestNamespace>,
}

#[pymethods]
impl PyRestNamespace {
    /// Create a new RestNamespace from properties
    ///
    /// # Arguments
    ///
    /// * `context_provider` - Optional object with `provide_context(info: dict) -> dict` method
    ///   for providing dynamic per-request context. Context keys that start with `headers.`
    ///   are converted to HTTP headers by stripping the prefix. For example,
    ///   `{"headers.Authorization": "Bearer token"}` becomes the `Authorization` header.
    /// * `**properties` - Namespace configuration properties (uri, delimiter, header.*, etc.)
    #[new]
    #[pyo3(signature = (context_provider = None, **properties))]
    fn new(
        context_provider: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = properties {
            props = dict_to_hashmap(dict)?;
        }

        let mut builder = RestNamespaceBuilder::from_properties(props).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to create RestNamespace: {}",
                e
            ))
        })?;

        // Add context provider if provided
        if let Some(provider) = context_provider {
            let py_provider = PyDynamicContextProvider::new(provider.clone().unbind());
            builder = builder.context_provider(Arc::new(py_provider));
        }

        let namespace = builder.build();

        Ok(Self {
            inner: Arc::new(namespace),
        })
    }

    /// Get the namespace ID
    fn namespace_id(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("PyRestNamespace({})", self.namespace_id())
    }

    // Namespace operations

    fn list_namespaces<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_namespaces(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn drop_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn namespace_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.namespace_exists(request))?
            .infer_error()?;
        Ok(())
    }

    // Table operations

    fn list_tables<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_tables(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn register_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.register_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn table_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.table_exists(request))?
            .infer_error()?;
        Ok(())
    }

    fn drop_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn deregister_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.deregister_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table(request, data))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn declare_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.declare_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn rename_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.rename_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    // Table version operations

    fn list_table_versions<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_versions(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn batch_delete_table_versions<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.batch_delete_table_versions(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Table branch operations

    fn create_table_branch<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_branch(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn list_table_branches<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_branches(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn delete_table_branch<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.delete_table_branch(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Data manipulation operations

    fn count_table_rows(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<i64> {
        let request: CountTableRowsRequest = depythonize(request)?;
        let count = crate::rt()
            .block_on(Some(py), self.inner.count_table_rows(request))?
            .infer_error()?;
        Ok(count)
    }

    fn insert_into_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: InsertIntoTableRequest = depythonize(request)?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.insert_into_table(request, data))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn merge_insert_into_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = merge_insert_request_from_json(depythonize(request)?).infer_error()?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.merge_insert_into_table(request, data))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn update_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: UpdateTableRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.update_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn delete_from_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DeleteFromTableRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.delete_from_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn query_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let request: QueryTableRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.query_table(request))?
            .infer_error()?;
        Ok(PyBytes::new(py, &response))
    }

    // Index operations

    fn create_table_index<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: CreateTableIndexRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_index(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn list_table_indices<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: ListTableIndicesRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_indices(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table_index_stats<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DescribeTableIndexStatsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table_index_stats(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Transaction operations

    fn describe_transaction<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DescribeTransactionRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_transaction(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn alter_transaction<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTransactionRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_transaction(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Additional index operations

    fn create_table_scalar_index<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: CreateTableIndexRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_scalar_index(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn drop_table_index<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DropTableIndexRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_table_index(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Additional table operations

    fn list_all_tables<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: ListTablesRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_all_tables(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn restore_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: RestoreTableRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.restore_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn update_table_schema_metadata<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: UpdateTableSchemaMetadataRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.update_table_schema_metadata(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn get_table_stats<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: GetTableStatsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.get_table_stats(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Query plan operations

    fn explain_table_query_plan(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<String> {
        let request: ExplainTableQueryPlanRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.explain_table_query_plan(request))?
            .infer_error()?;
        Ok(response)
    }

    fn analyze_table_query_plan(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<String> {
        let request: AnalyzeTableQueryPlanRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.analyze_table_query_plan(request))?
            .infer_error()?;
        Ok(response)
    }

    // Column alteration operations

    fn alter_table_add_columns<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTableAddColumnsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_table_add_columns(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn alter_table_alter_columns<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTableAlterColumnsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_table_alter_columns(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn alter_table_drop_columns<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: AlterTableDropColumnsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.alter_table_drop_columns(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn alter_table_backfill_columns<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (typed, extra): (AlterTableBackfillColumnsRequest, _) =
            depythonize_with_extras(request)?;
        let response = crate::rt()
            .block_on(
                Some(py),
                self.inner
                    .alter_table_backfill_columns_with_extra(typed, extra),
            )?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn refresh_materialized_view<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (typed, extra): (RefreshMaterializedViewRequest, _) =
            depythonize_with_extras(request)?;
        let response = crate::rt()
            .block_on(
                Some(py),
                self.inner
                    .refresh_materialized_view_with_extra(typed, extra),
            )?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_materialized_view<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: CreateMaterializedViewRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_materialized_view(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Table tag operations

    fn list_table_tags<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: ListTableTagsRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_tags(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn get_table_tag_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: GetTableTagVersionRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.get_table_tag_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table_tag<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: CreateTableTagRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_tag(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn delete_table_tag<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: DeleteTableTagRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.delete_table_tag(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn update_table_tag<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: UpdateTableTagRequest = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.update_table_tag(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // Operation metrics methods

    /// Retrieve operation metrics as a dictionary.
    ///
    /// Returns a dict where keys are operation names (e.g., "list_tables", "describe_table")
    /// and values are the number of times each operation was called.
    ///
    /// Returns an empty dict if `ops_metrics_enabled` was false when creating the namespace.
    fn retrieve_ops_metrics(&self) -> HashMap<String, u64> {
        self.inner.retrieve_ops_metrics()
    }

    /// Reset all operation metrics counters to zero.
    ///
    /// Does nothing if `ops_metrics_enabled` was false when creating the namespace.
    fn reset_ops_metrics(&self) {
        self.inner.reset_ops_metrics()
    }
}

/// Get or create the DictWithModelDump class in Python.
/// This class acts like a dict but also has model_dump() method.
/// This allows it to work with both:
/// - depythonize (which expects a dict/Mapping)
/// - Python code that calls .model_dump() (like DirectoryNamespace wrapper)
fn get_dict_with_model_dump_class(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    // Use a module-level cache via __builtins__
    let builtins = py.import("builtins")?;
    if builtins.hasattr("_DictWithModelDump")? {
        return builtins.getattr("_DictWithModelDump");
    }

    // Create the class using exec
    let locals = PyDict::new(py);
    py.run(
        c"class DictWithModelDump(dict):
    def model_dump(self):
        return dict(self)",
        None,
        Some(&locals),
    )?;
    let class = locals.get_item("DictWithModelDump")?.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Failed to create DictWithModelDump class")
    })?;

    // Cache it
    builtins.setattr("_DictWithModelDump", &class)?;
    Ok(class)
}

/// Convert a Python namespace exception into a lance error, preserving the
/// namespace error identity when the exception is a `lance_namespace`
/// `LanceNamespaceError` carrying an error `code`, so callers can react to
/// e.g. TableNotFound the same way they do for native clients. Foreign
/// exceptions that happen to carry an integer `code` (e.g. SystemExit) must
/// not be reinterpreted, so the extraction is gated on the exception type.
fn namespace_error_from_py(method_name: &'static str, e: PyErr) -> lance_core::Error {
    Python::attach(|py| {
        let value = e.value(py);
        let is_namespace_error = py
            .import("lance_namespace.errors")
            .and_then(|module| module.getattr("LanceNamespaceError"))
            .and_then(|class| value.is_instance(&class))
            .unwrap_or(false);
        if is_namespace_error
            && let Ok(code) = value.getattr("code").and_then(|code| code.extract::<u32>())
        {
            return lance_namespace::error::NamespaceError::from_code(code, value.to_string())
                .into();
        }
        lance_core::Error::io(format!("Python error in {}: {}", method_name, e))
    })
}

/// Helper to call a Python namespace method with JSON serialization.
/// For methods that take a request and return a response.
/// Uses DictWithModelDump to pass a dict that also has model_dump() method,
/// making it compatible with both depythonize and Python wrappers.
async fn call_py_method<Req, Resp>(
    py_namespace: Arc<Py<PyAny>>,
    method_name: &'static str,
    request: Req,
) -> lance_core::Result<Resp>
where
    Req: serde::Serialize + Send + 'static,
    Resp: serde::de::DeserializeOwned + Send + 'static,
{
    let request_json = serde_json::to_string(&request).map_err(|e| {
        lance_core::Error::io(format!(
            "Failed to serialize request for {}: {}",
            method_name, e
        ))
    })?;

    let response_json = tokio::task::spawn_blocking(move || {
        Python::attach(|py| {
            let json_module = py.import("json")?;
            let request_dict = json_module.call_method1("loads", (&request_json,))?;

            // Wrap dict in DictWithModelDump so it works with both depythonize and .model_dump()
            let dict_class = get_dict_with_model_dump_class(py)?;
            let request_arg = dict_class.call1((request_dict,))?;

            // Call the Python method
            let result = py_namespace.call_method1(py, method_name, (request_arg,))?;

            // Convert response to dict, then to JSON
            // Pydantic models have model_dump() method
            let result_dict = if result.bind(py).hasattr("model_dump")? {
                result.call_method0(py, "model_dump")?
            } else {
                result
            };
            let response_json: String = json_module
                .call_method1("dumps", (result_dict,))?
                .extract()?;
            Ok::<_, PyErr>(response_json)
        })
    })
    .await
    .map_err(|e| lance_core::Error::io(format!("Task join error for {}: {}", method_name, e)))?
    .map_err(|e: PyErr| namespace_error_from_py(method_name, e))?;

    serde_json::from_str(&response_json).map_err(|e| {
        lance_core::Error::io(format!(
            "Failed to deserialize response from {}: {}",
            method_name, e
        ))
    })
}

/// Wrapper that allows any Python object implementing LanceNamespace protocol
/// to be used as a Rust LanceNamespace.
///
/// This is similar to JavaLanceNamespace in the Java bindings - it wraps a Python
/// object and calls back into Python when namespace methods are invoked.
///
/// We use `Arc<Py<PyAny>>` instead of `Py<PyAny>` directly because cloning `Py`
/// requires the GIL, but cloning `Arc` does not. This allows us to pass the
/// namespace reference to `spawn_blocking` without holding the GIL.
pub struct PyLanceNamespace {
    py_namespace: Arc<Py<PyAny>>,
    namespace_id: String,
}

impl PyLanceNamespace {
    /// Create a new PyLanceNamespace wrapper around a Python namespace object.
    pub fn new(_py: Python<'_>, py_namespace: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Get the namespace_id by calling the Python method
        let namespace_id = py_namespace
            .call_method0("namespace_id")?
            .extract::<String>()?;

        Ok(Self {
            py_namespace: Arc::new(py_namespace.clone().unbind()),
            namespace_id,
        })
    }

    /// Create an Arc<dyn LanceNamespace> from a Python namespace object.
    pub fn create_arc(
        py: Python<'_>,
        py_namespace: &Bound<'_, PyAny>,
    ) -> PyResult<Arc<dyn LanceNamespaceTrait>> {
        let wrapper = Self::new(py, py_namespace)?;
        Ok(Arc::new(wrapper))
    }
}

impl std::fmt::Debug for PyLanceNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PyLanceNamespace {{ id: {} }}", self.namespace_id)
    }
}

#[async_trait]
impl LanceNamespaceTrait for PyLanceNamespace {
    fn namespace_id(&self) -> String {
        self.namespace_id.clone()
    }

    async fn describe_table(
        &self,
        request: DescribeTableRequest,
    ) -> lance_core::Result<DescribeTableResponse> {
        call_py_method(self.py_namespace.clone(), "describe_table", request).await
    }

    async fn describe_table_version(
        &self,
        request: DescribeTableVersionRequest,
    ) -> lance_core::Result<DescribeTableVersionResponse> {
        call_py_method(self.py_namespace.clone(), "describe_table_version", request).await
    }

    async fn create_table_version(
        &self,
        request: CreateTableVersionRequest,
    ) -> lance_core::Result<CreateTableVersionResponse> {
        call_py_method(self.py_namespace.clone(), "create_table_version", request).await
    }

    async fn list_table_versions(
        &self,
        request: ListTableVersionsRequest,
    ) -> lance_core::Result<ListTableVersionsResponse> {
        call_py_method(self.py_namespace.clone(), "list_table_versions", request).await
    }
}

/// Extract an `Arc<dyn LanceNamespace>` from a Python namespace object.
///
/// This function handles the different ways a Python namespace can be provided:
/// 1. Direct PyO3 class (PyDirectoryNamespace or PyRestNamespace)
/// 2. Python wrapper class with `_inner` attribute that holds the PyO3 class
/// 3. Custom Python implementation (wrapped with PyLanceNamespace)
///
/// For Python wrapper classes (DirectoryNamespace, RestNamespace in namespace.py),
/// we check if it's the exact wrapper class by comparing type names. Subclasses
/// are wrapped with PyLanceNamespace to call through Python.
pub fn extract_namespace_arc(
    py: Python<'_>,
    namespace_client: &Bound<'_, PyAny>,
) -> PyResult<Arc<dyn LanceNamespaceTrait>> {
    // Direct PyO3 class
    if let Ok(dir_namespace_client) = namespace_client.cast::<PyDirectoryNamespace>() {
        return Ok(dir_namespace_client.borrow().inner.clone() as Arc<dyn LanceNamespaceTrait>);
    }
    if let Ok(rest_namespace_client) = namespace_client.cast::<PyRestNamespace>() {
        return Ok(rest_namespace_client.borrow().inner.clone() as Arc<dyn LanceNamespaceTrait>);
    }

    // Python wrapper class - check if it's the exact wrapper class
    if let Ok(inner) = namespace_client.getattr("_inner") {
        let type_name = namespace_client
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_default();

        if type_name == "DirectoryNamespace" {
            if let Ok(dir_namespace_client) = inner.cast::<PyDirectoryNamespace>() {
                return Ok(
                    dir_namespace_client.borrow().inner.clone() as Arc<dyn LanceNamespaceTrait>
                );
            }
        } else if type_name == "RestNamespace"
            && let Ok(rest_namespace_client) = inner.cast::<PyRestNamespace>()
        {
            return Ok(rest_namespace_client.borrow().inner.clone() as Arc<dyn LanceNamespaceTrait>);
        }
    }

    // Custom Python implementation or subclass - wrap with PyLanceNamespace
    PyLanceNamespace::create_arc(py, namespace_client)
}

/// Python wrapper for REST adapter server
#[pyclass(name = "PyRestAdapter", module = "lance.lance")]
pub struct PyRestAdapter {
    backend: Arc<dyn lance_namespace::LanceNamespace>,
    config: RestAdapterConfig,
    handle: Option<RestAdapterHandle>,
}

#[pymethods]
impl PyRestAdapter {
    /// Create a new REST adapter server with namespace configuration.
    /// Default port is 2333 per REST spec. Use port 0 to let OS assign an ephemeral port.
    /// Use `port` property after `start()` to get the actual port.
    #[new]
    #[pyo3(signature = (namespace_client_impl, namespace_client_properties, session = None, host = None, port = None))]
    fn new(
        namespace_client_impl: String,
        namespace_client_properties: Option<&Bound<'_, PyDict>>,
        session: Option<&Bound<'_, Session>>,
        host: Option<String>,
        port: Option<u16>,
    ) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = namespace_client_properties {
            props = dict_to_hashmap(dict)?;
        }

        let mut builder = ConnectBuilder::new(namespace_client_impl);
        for (k, v) in props {
            builder = builder.property(k, v);
        }

        if let Some(sess) = session {
            builder = builder.session(sess.borrow().inner.clone());
        }

        let backend = crate::rt()
            .block_on(None, builder.connect())?
            .infer_error()?;

        let mut config = RestAdapterConfig::default();
        if let Some(h) = host {
            config.host = h;
        }
        if let Some(p) = port {
            config.port = p;
        }

        Ok(Self {
            backend,
            config,
            handle: None,
        })
    }

    /// Get the actual port the server is listening on.
    /// Returns 0 if server is not started yet.
    #[getter]
    fn port(&self) -> u16 {
        self.handle.as_ref().map(|h| h.port()).unwrap_or(0)
    }

    /// Start the REST server in the background
    fn start(&mut self, py: Python) -> PyResult<()> {
        let adapter = RestAdapter::new(self.backend.clone(), self.config.clone());
        let handle = crate::rt()
            .block_on(Some(py), adapter.start())?
            .infer_error()?;

        self.handle = Some(handle);
        Ok(())
    }

    /// Stop the REST server
    fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.shutdown();
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        slf.stop();
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRestAdapter(host='{}', port={})",
            self.config.host, self.config.port
        )
    }
}
