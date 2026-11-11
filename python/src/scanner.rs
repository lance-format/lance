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

//! Python scanner configuration and search filter bindings.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::pyarrow::*;
use arrow_array::{RecordBatchReader, make_array};
use arrow_data::ArrayData;
use lance::dataset::scanner::{
    ColumnOrdering, ExecutionSummaryCounts, QueryFilter, RowAddrTreeMap,
};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::inverted::query::Occur;
use lance_index::scalar::inverted::query::{
    BooleanQuery, BoostQuery, CombinedFieldsQuery, FtsQuery, MatchQuery, MultiMatchQuery, Operator,
    PhraseQuery,
};
use lance_index::vector::{
    ApproxMode, DEFAULT_QUERY_PARALLELISM, ProbeParams, Query as VectorQuery,
};
use lance_linalg::distance::MetricType;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::types::{PyBytes, PyDict};

use ::lance::dataset::scanner::Scanner as LanceScanner;

use crate::error::PythonErrorExt;
use crate::reader::LanceReader;
use crate::rt;
use crate::schema::logical_arrow_schema;
use crate::utils::PyLance;

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
    /// Additional debugging timings in nanoseconds. Keys may change; nested and
    /// concurrent stages overlap and must not be summed as query wall time.
    pub all_times: HashMap<String, usize>,
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
            // Keep Python callbacks consistent with native execution summaries.
            all_times: stats.all_times.clone(),
        }
    }
}

#[pymethods]
impl ScanStatistics {
    fn __repr__(&self) -> String {
        format!(
            "ScanStatistics(iops={}, requests={}, bytes_read={}, indices_loaded={}, parts_loaded={}, index_comparisons={}, index_cache_hits={}, index_cache_misses={}, all_counts={:?}, all_times={:?})",
            self.iops,
            self.requests,
            self.bytes_read,
            self.indices_loaded,
            self.parts_loaded,
            self.index_comparisons,
            self.index_cache_hits,
            self.index_cache_misses,
            self.all_counts,
            self.all_times
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

impl FromPyObject<'_, '_> for PyLance<ColumnOrdering> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let ascending: bool = ob.getattr("ascending")?.extract()?;
        let nulls_first: bool = ob.getattr("nulls_first")?.extract()?;
        let column_name: String = ob.getattr("column_name")?.extract()?;
        Ok(Self(ColumnOrdering {
            ascending,
            nulls_first,
            column_name,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&ColumnOrdering> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceScanner")))
            .and_then(|cls| cls.getattr(intern!(py, "ColumnOrdering")))
            .expect("Failed to get RewrittenIndex class");

        let column_name = self.0.column_name.clone();
        let ascending = self.0.ascending;
        let nulls_first = self.0.nulls_first;
        cls.call1((column_name, ascending, nulls_first))
    }
}

/// Serialize row addresses into the payload the scanner's `row_addr_allowlist` /
/// `row_addr_blocklist` parameters accept.
///
/// Without this those parameters are unusable from Python: they take the roaring
/// `RowAddrTreeMap` encoding, which nothing else exposed here can produce. The
/// result stays plain bytes, so a mask may equally be built by another extension
/// module and handed in.
#[pyfunction(name = "_serialize_row_addrs")]
pub fn serialize_row_addrs(py: Python<'_>, addrs: Vec<u64>) -> PyResult<Py<PyBytes>> {
    let treemap = RowAddrTreeMap::from_iter(addrs);
    let mut buf = Vec::with_capacity(treemap.serialized_size());
    treemap.serialize_into(&mut buf).infer_error()?;
    Ok(PyBytes::new(py, &buf).unbind())
}

#[pyclass(name = "PyFullTextQuery", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyFullTextQuery {
    pub(crate) inner: FtsQuery,
}

#[pymethods]
impl PyFullTextQuery {
    #[staticmethod]
    #[pyo3(signature = (query, column, boost=1.0, fuzziness=Some(0), max_expansions=50, operator="OR", prefix_length=0, document_granularity=None))]
    #[allow(clippy::too_many_arguments)]
    fn match_query(
        query: String,
        column: String,
        boost: f32,
        fuzziness: Option<u32>,
        max_expansions: usize,
        operator: &str,
        prefix_length: u32,
        document_granularity: Option<&str>,
    ) -> PyResult<Self> {
        let mut query = MatchQuery::new(query)
            .with_column(Some(column))
            .with_boost(boost)
            .with_fuzziness(fuzziness)
            .with_max_expansions(max_expansions)
            .with_operator(
                Operator::try_from(operator)
                    .map_err(|e| PyValueError::new_err(format!("Invalid operator: {}", e)))?,
            )
            .with_prefix_length(prefix_length);
        if let Some(document_granularity) = document_granularity {
            query = query.with_document_granularity(
                DocumentGranularity::try_from(document_granularity)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            );
        }
        Ok(Self {
            inner: query.into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (query, column, slop, document_granularity=None))]
    fn phrase_query(
        query: String,
        column: String,
        slop: u32,
        document_granularity: Option<&str>,
    ) -> PyResult<Self> {
        let mut query = PhraseQuery::new(query)
            .with_column(Some(column))
            .with_slop(slop);
        if let Some(document_granularity) = document_granularity {
            query = query.with_document_granularity(
                DocumentGranularity::try_from(document_granularity)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            );
        }
        Ok(Self {
            inner: query.into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (positive, negative,negative_boost=None))]
    fn boost_query(positive: Self, negative: Self, negative_boost: Option<f32>) -> PyResult<Self> {
        Ok(Self {
            inner: BoostQuery::new(positive.inner, negative.inner, negative_boost).into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (query, columns, boosts=None, operator="OR"))]
    fn multi_match_query(
        query: String,
        columns: Vec<String>,
        boosts: Option<Vec<f32>>,
        operator: &str,
    ) -> PyResult<Self> {
        let q = MultiMatchQuery::try_new(query, columns)
            .map_err(|e| PyValueError::new_err(format!("Invalid query: {}", e)))?;
        let q = if let Some(boosts) = boosts {
            q.try_with_boosts(boosts)
                .map_err(|e| PyValueError::new_err(format!("Invalid boosts: {}", e)))?
        } else {
            q
        };

        let op = Operator::try_from(operator)
            .map_err(|e| PyValueError::new_err(format!("Invalid operator: {}", e)))?;

        Ok(Self {
            inner: q.with_operator(op).into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (query, columns, boosts=None, operator="OR"))]
    fn combined_fields_query(
        query: String,
        columns: Vec<String>,
        boosts: Option<Vec<f32>>,
        operator: &str,
    ) -> PyResult<Self> {
        let q = CombinedFieldsQuery::try_new(query, columns)
            .map_err(|e| PyValueError::new_err(format!("Invalid query: {}", e)))?;
        let q = if let Some(boosts) = boosts {
            q.try_with_boosts(boosts)
                .map_err(|e| PyValueError::new_err(format!("Invalid boosts: {}", e)))?
        } else {
            q
        };

        let op = Operator::try_from(operator)
            .map_err(|e| PyValueError::new_err(format!("Invalid operator: {}", e)))?;

        Ok(Self {
            inner: q.with_operator(op).into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (queries))]
    fn boolean_query(queries: Vec<(String, Self)>) -> PyResult<Self> {
        let mut sub_queries = Vec::with_capacity(queries.len());
        for (occur, q) in queries {
            let occur = Occur::try_from(occur.as_str())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            sub_queries.push((occur, q.inner));
        }

        Ok(Self {
            inner: BooleanQuery::new(sub_queries).into(),
        })
    }
}

pub(crate) type VectorQueryParams = (
    String,
    arrow_array::ArrayRef,
    usize,
    usize,
    Option<usize>,
    Option<MetricType>,
    Option<u32>,
    Option<u32>,
    bool,
    Option<usize>,
    i32,
    ApproxMode,
    f64,
);

fn extract_query_parallelism(value: &Bound<'_, PyAny>) -> PyResult<i32> {
    let query_parallelism = value.extract()?;
    if query_parallelism < -1 {
        Err(PyValueError::new_err("query_parallelism must be >= -1"))
    } else {
        Ok(query_parallelism)
    }
}

fn vector_query_query_parallelism_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<i32> {
    if let Some(query_parallelism) = dict.get_item("query_parallelism")?
        && !query_parallelism.is_none()
    {
        extract_query_parallelism(&query_parallelism)
    } else {
        Ok(DEFAULT_QUERY_PARALLELISM)
    }
}

fn vector_query_approx_mode_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<ApproxMode> {
    if let Some(approx_mode) = dict.get_item("approx_mode")?
        && !approx_mode.is_none()
    {
        match approx_mode.to_string().to_lowercase().as_str() {
            "fast" => Ok(ApproxMode::Fast),
            "normal" => Ok(ApproxMode::Normal),
            "accurate" => Ok(ApproxMode::Accurate),
            value => Err(PyValueError::new_err(format!(
                "approx_mode must be one of 'fast', 'normal', or 'accurate', got '{value}'"
            ))),
        }
    } else {
        Ok(ApproxMode::Normal)
    }
}

pub(crate) fn vector_query_params_from_dict(
    dict: &Bound<'_, PyDict>,
    default_k: usize,
) -> PyResult<VectorQueryParams> {
    let column = dict
        .get_item("column")?
        .ok_or_else(|| PyKeyError::new_err("Need column for nearest"))?
        .to_string();

    let qval = dict
        .get_item("q")?
        .ok_or_else(|| PyKeyError::new_err("Need q for nearest"))?;
    let data = ArrayData::from_pyarrow_bound(&qval)?;
    let key = make_array(data);

    let k: usize = if let Some(k) = dict.get_item("k")? {
        if k.is_none() {
            // Use limit if k is not specified, default to 10.
            default_k
        } else {
            k.extract()?
        }
    } else {
        default_k
    };

    let search_effort = dict
        .get_item("search_effort")?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<f64>())
        .transpose()?;
    let mut probes = ProbeParams {
        search_effort,
        ..Default::default()
    };

    if search_effort.is_none()
        && let Some(nprobes) = dict.get_item("nprobes")?
        && !nprobes.is_none()
    {
        probes.nprobes = Some(nprobes.extract()?);
    }

    if let Some(min_nprobes) = dict.get_item("minimum_nprobes")?
        && !min_nprobes.is_none()
    {
        probes.minimum_nprobes = Some(min_nprobes.extract()?);
    }

    if let Some(max_nprobes) = dict.get_item("maximum_nprobes")?
        && !max_nprobes.is_none()
    {
        probes.maximum_nprobes = Some(max_nprobes.extract()?);
    }
    let (minimum_nprobes, maximum_nprobes, search_effort) = probes.resolve();

    if let Some(maximum_nprobes_val) = maximum_nprobes
        && minimum_nprobes > maximum_nprobes_val
    {
        return Err(PyValueError::new_err(
            "minimum_nprobes must be <= maximum_nprobes",
        ));
    }

    if minimum_nprobes < 1 {
        return Err(PyValueError::new_err("minimum_nprobes must be >= 1"));
    }

    if let Some(maximum_nprobes_val) = maximum_nprobes
        && maximum_nprobes_val < 1
    {
        return Err(PyValueError::new_err("maximum_nprobes must be >= 1"));
    }

    let metric_type: Option<MetricType> = if let Some(metric) = dict.get_item("metric")? {
        if metric.is_none() {
            None
        } else {
            Some(
                MetricType::try_from(metric.to_string().to_lowercase().as_str())
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        }
    } else {
        None
    };

    // When refine factor is specified, a final Refine stage will be added to the I/O plan,
    // and use Flat index over the raw vectors to refine the results.
    // By default, `refine_factor` is None to not involve extra I/O exec node and random access.
    let refine_factor: Option<u32> = if let Some(rf) = dict.get_item("refine_factor")? {
        if rf.is_none() { None } else { rf.extract()? }
    } else {
        None
    };

    let quantized_refine_factor: Option<u32> =
        if let Some(factor) = dict.get_item("quantized_refine_factor")? {
            if factor.is_none() {
                None
            } else {
                let factor: u32 = factor.extract()?;
                if factor == 0 {
                    return Err(PyValueError::new_err(
                        "quantized_refine_factor must be greater than zero",
                    ));
                }
                Some(factor)
            }
        } else {
            None
        };

    let use_index: bool = if let Some(idx) = dict.get_item("use_index")? {
        idx.extract()?
    } else {
        true
    };

    let ef: Option<usize> = if let Some(ef_obj) = dict.get_item("ef")? {
        if ef_obj.is_none() {
            None
        } else {
            ef_obj.extract()?
        }
    } else {
        None
    };

    let query_parallelism = vector_query_query_parallelism_from_dict(dict)?;
    let approx_mode = vector_query_approx_mode_from_dict(dict)?;

    Ok((
        column,
        key,
        k,
        minimum_nprobes,
        maximum_nprobes,
        metric_type,
        refine_factor,
        quantized_refine_factor,
        use_index,
        ef,
        query_parallelism,
        approx_mode,
        search_effort,
    ))
}

#[pyclass(name = "PySearchFilter", from_py_object)]
#[derive(Debug, Clone)]
pub struct PySearchFilter {
    pub(crate) inner: QueryFilter,
}

#[pymethods]
impl PySearchFilter {
    /// Create a search filter from a full text query.
    #[staticmethod]
    #[pyo3(signature = (query))]
    fn from_full_text_query(query: PyFullTextQuery) -> PyResult<Self> {
        Ok(Self {
            inner: QueryFilter::Fts(FullTextSearchQuery::new_query(query.inner.clone())),
        })
    }

    /// Create a query filter from a vector search query dict.
    #[staticmethod]
    #[pyo3(signature = (query))]
    fn from_vector_search_query(query: &Bound<'_, PyDict>) -> PyResult<Self> {
        let default_k = 10;
        let (
            column,
            key,
            k,
            minimum_nprobes,
            maximum_nprobes,
            metric_type_opt,
            refine_factor,
            quantized_refine_factor,
            use_index,
            ef,
            query_parallelism,
            approx_mode,
            search_effort,
        ) = vector_query_params_from_dict(query, default_k)?;

        if quantized_refine_factor.is_some() {
            return Err(PyValueError::new_err(
                "quantized_refine_factor is only supported on dataset scanners",
            ));
        }

        // Leave an unset metric to the core, which takes it from the index the
        // search picks, as the `nearest` path does.
        let metric_type = metric_type_opt;

        let vector_query = VectorQuery {
            column,
            key,
            k,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes,
            maximum_nprobes,
            ef,
            refine_factor,
            metric_type,
            use_index,
            query_parallelism,
            dist_q_c: 0.0,
            approx_mode,
            search_effort,
        };
        vector_query
            .validate_search_effort()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(Self {
            inner: QueryFilter::Vector(vector_query),
        })
    }
}
