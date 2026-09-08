// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::pyarrow::*;
use arrow_array::RecordBatchReader;
use lance::dataset::scanner::ExecutionSummaryCounts;
use pyo3::prelude::*;
use pyo3::pyclass;

use ::lance::dataset::scanner::Scanner as LanceScanner;
use pyo3::exceptions::PyValueError;

use crate::reader::LanceReader;
use crate::rt;
use crate::schema::logical_arrow_schema;

/// This will be wrapped by a python class to provide
/// additional functionality
#[pyclass(name = "_Scanner", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct Scanner {
    scanner: Arc<LanceScanner>,
}

impl Scanner {
    pub fn new(scanner: Arc<LanceScanner>) -> Self {
        Self { scanner }
    }

    pub(crate) async fn to_reader(&self) -> ::lance::Result<LanceReader> {
        LanceReader::try_new(self.scanner.clone()).await
    }
}

#[pyclass(name = "ScanStatistics", module = "_lib", get_all, skip_from_py_object)]
#[derive(Clone)]
/// Statistics about the scan.
pub struct ScanStatistics {
    /// Number of IO operations performed.  This may be slightly higher than
    /// the actual number due to coalesced I/O
    pub iops: usize,
    /// Number of requests made to the storage layer
    pub requests: usize,
    /// Number of bytes read from disk
    pub bytes_read: usize,
    /// Number of indices loaded
    pub indices_loaded: usize,
    /// Number of index partitions loaded
    pub parts_loaded: usize,
    /// Number of index comparisons performed
    pub index_comparisons: usize,
    /// Number of index cache page lookups that were served from memory
    pub index_cache_hits: usize,
    /// Number of index cache page lookups that had to load from storage
    pub index_cache_misses: usize,
    /// Additional metrics for more detailed statistics. These are subject to change in the future
    /// and should only be used for debugging purposes.
    pub all_counts: HashMap<String, usize>,
}

impl ScanStatistics {
    pub fn from_lance(stats: &ExecutionSummaryCounts) -> Self {
        Self {
            iops: stats.iops,
            requests: stats.requests,
            bytes_read: stats.bytes_read,
            indices_loaded: stats.indices_loaded,
            parts_loaded: stats.parts_loaded,
            index_comparisons: stats.index_comparisons,
            index_cache_hits: stats.index_cache_hits(),
            index_cache_misses: stats.index_cache_misses(),
            all_counts: stats.all_counts.clone(),
        }
    }
}

#[pymethods]
impl ScanStatistics {
    fn __repr__(&self) -> String {
        format!(
            "ScanStatistics(iops={}, requests={}, bytes_read={}, indices_loaded={}, parts_loaded={}, index_comparisons={}, index_cache_hits={}, index_cache_misses={}, all_counts={:?})",
            self.iops,
            self.requests,
            self.bytes_read,
            self.indices_loaded,
            self.parts_loaded,
            self.index_comparisons,
            self.index_cache_hits,
            self.index_cache_misses,
            self.all_counts
        )
    }
}

#[pymethods]
impl Scanner {
    #[getter(schema)]
    fn schema<'py>(self_: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let scanner = self_.scanner.clone();
        let schema = rt()
            .spawn(Some(self_.py()), async move { scanner.schema().await })?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let logical_schema = logical_arrow_schema(schema.as_ref());
        logical_schema.to_pyarrow(self_.py())
    }

    #[pyo3(signature = (*, verbose = false))]
    fn explain_plan(self_: PyRef<'_, Self>, verbose: bool) -> PyResult<String> {
        let scanner = self_.scanner.clone();
        let res = rt()
            .spawn(Some(self_.py()), async move {
                scanner.explain_plan(verbose).await
            })?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(res)
    }

    #[pyo3(signature = (*, count_rows = false))]
    fn analyze_plan(self_: PyRef<'_, Self>, count_rows: bool) -> PyResult<String> {
        let scanner = self_.scanner.clone();
        let res = rt()
            .spawn(Some(self_.py()), async move {
                if count_rows {
                    scanner.analyze_count_plan().await
                } else {
                    scanner.analyze_plan().await
                }
            })?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(res)
    }

    fn count_rows(self_: PyRef<'_, Self>) -> PyResult<u64> {
        let scanner = self_.scanner.clone();
        rt().spawn(Some(self_.py()), async move { scanner.count_rows().await })?
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn to_pyarrow(
        self_: PyRef<'_, Self>,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let scanner = self_.scanner.clone();
        let reader = rt()
            .spawn(Some(self_.py()), async move {
                LanceReader::try_new(scanner).await
            })?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(PyArrowType(Box::new(reader)))
    }
}
