// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::*;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StructArray, make_array,
};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use lance::dataset::Dataset as LanceDataset;
use lance::dataset::mem_wal::scanner::{
    LsmDataSourceCollector, LsmPointLookupPlanner, LsmVectorSearchPlanner, SsTable,
    parse_filter_expr as parse_lsm_filter_expr,
};
use lance::dataset::mem_wal::write::{MemTableStats, ShardMemory, WriteStatsSnapshot};
use lance::dataset::mem_wal::{LsmScanner, ShardSnapshot, ShardWriter, evaluate_sharding_spec};
use lance_index::mem_wal::{
    CompactedSsTable as LanceCompactedSsTable, ShardingField, ShardingSpec,
};
use lance_linalg::distance::DistanceType;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::sync::Mutex as TokioMutex;
use uuid::Uuid;

use crate::dataset::Dataset as PyDataset;
use crate::rt;
use crate::schema::LanceSchema as PyLanceSchema;

/// Evaluate a MemWAL sharding spec against one PyArrow RecordBatch.
#[pyfunction(name = "_evaluate_sharding_spec", signature = (batch, spec, schema))]
pub fn py_evaluate_sharding_spec<'py>(
    py: Python<'py>,
    batch: PyArrowType<RecordBatch>,
    spec: &Bound<'_, PyAny>,
    schema: &PyLanceSchema,
) -> PyResult<Bound<'py, PyAny>> {
    let PyArrowType(batch) = batch;
    let spec = sharding_spec_from_py(spec)?;
    let result = evaluate_sharding_spec(&batch, &spec, &schema.0)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    result.to_pyarrow(py)
}

/// Write a primary-key dedup sidecar (`_pk_index/`) for an SSTable
/// dataset already written at `gen_path`, mirroring what production flush emits.
///
/// Test-support only: lets Python tests stage a *faithful* SSTable
/// (dataset + sidecar). Production always writes the sidecar during flush, so a
/// dataset-without-sidecar is not a state the system otherwise produces.
#[pyfunction(name = "_write_pk_sidecar", signature = (gen_path, data, pk_columns))]
pub fn py_write_pk_sidecar(
    py: Python<'_>,
    gen_path: String,
    data: &Bound<'_, PyAny>,
    pk_columns: Vec<String>,
) -> PyResult<()> {
    let reader = ArrowArrayStreamReader::from_pyarrow_bound(data)
        .map_err(|e| PyValueError::new_err(format!("Cannot read data as Arrow: {}", e)))?;
    let batches: Vec<RecordBatch> = reader
        .collect::<Result<_, _>>()
        .map_err(|e| PyIOError::new_err(format!("Failed to read batches: {}", e)))?;
    rt().block_on(Some(py), async move {
        let pk_refs: Vec<&str> = pk_columns.iter().map(String::as_str).collect();
        lance::dataset::mem_wal::scanner::write_pk_sidecar(&gen_path, &batches, &pk_refs).await
    })?
    .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))
}

fn sharding_spec_from_py(spec: &Bound<'_, PyAny>) -> PyResult<ShardingSpec> {
    let spec_id = get_py_value(spec, "spec_id")?.extract::<u32>()?;
    let fields_obj = get_py_value(spec, "fields")?;
    let mut fields = Vec::new();
    for field_obj in fields_obj.try_iter()? {
        fields.push(sharding_field_from_py(&field_obj?)?);
    }
    Ok(ShardingSpec { spec_id, fields })
}

fn sharding_field_from_py(field: &Bound<'_, PyAny>) -> PyResult<ShardingField> {
    Ok(ShardingField {
        field_id: get_py_value(field, "field_id")?.extract::<String>()?,
        source_ids: get_py_value(field, "source_ids")?.extract::<Vec<i32>>()?,
        transform: optional_string(get_py_value(field, "transform")?)?,
        expression: optional_string(get_py_value(field, "expression")?)?,
        result_type: get_py_value(field, "result_type")?.extract::<String>()?,
        parameters: get_py_value(field, "parameters")?.extract::<HashMap<String, String>>()?,
    })
}

fn get_py_value<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        dict.get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("Missing sharding spec field '{name}'")))
    } else {
        obj.getattr(name)
    }
}

fn optional_string(value: Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() {
        Ok(None)
    } else {
        value.extract::<String>().map(Some)
    }
}

/// Points to an SSTable compacted into the base table.
///
/// Used with `MergeInsertBuilder.mark_sstables_as_compacted()`.
#[pyclass(name = "_CompactedSsTable", module = "_lib")]
pub struct PyCompactedSsTable {
    pub shard_id: String,
    pub generation: u64,
}

#[pymethods]
impl PyCompactedSsTable {
    #[new]
    pub fn new(shard_id: String, generation: u64) -> Self {
        Self {
            shard_id,
            generation,
        }
    }

    #[getter]
    pub fn shard_id(&self) -> &str {
        &self.shard_id
    }

    #[getter]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn __repr__(&self) -> String {
        format!(
            "_CompactedSsTable(shard_id='{}', generation={})",
            self.shard_id, self.generation
        )
    }
}

impl PyCompactedSsTable {
    pub fn to_lance(&self) -> PyResult<LanceCompactedSsTable> {
        let uuid = Uuid::parse_str(&self.shard_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid shard_id UUID: {}", e)))?;
        Ok(LanceCompactedSsTable::new(uuid, self.generation))
    }
}

/// Snapshot of a MemWAL shard's state at a point in time.
///
/// Used to specify which SSTables to include when creating an
/// `_LsmScanner`. Supports a builder pattern for adding generations.
#[pyclass(name = "_ShardSnapshot", module = "_lib", skip_from_py_object)]
#[derive(Clone)]
pub struct PyShardSnapshot {
    pub inner: ShardSnapshot,
}

#[pymethods]
impl PyShardSnapshot {
    #[new]
    pub fn new(shard_id: String) -> PyResult<Self> {
        let uuid = Uuid::parse_str(&shard_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid shard_id UUID: {}", e)))?;
        Ok(Self {
            inner: ShardSnapshot::new(uuid),
        })
    }

    /// Set the sharding spec ID for this snapshot.
    pub fn with_spec_id(mut slf: PyRefMut<'_, Self>, spec_id: u32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_spec_id(spec_id);
        slf
    }

    /// Set the current (active) generation number.
    pub fn with_current_generation(
        mut slf: PyRefMut<'_, Self>,
        generation: u64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_current_generation(generation);
        slf
    }

    /// Add an SSTable by its generation number and storage path.
    pub fn with_sstable(
        mut slf: PyRefMut<'_, Self>,
        generation: u64,
        path: String,
    ) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_sstable(generation, path);
        slf
    }

    #[getter]
    pub fn shard_id(&self) -> String {
        self.inner.shard_id.to_string()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "_ShardSnapshot(shard_id='{}', current_gen={}, sstables={})",
            self.inner.shard_id,
            self.inner.current_generation,
            self.inner.sstables.len()
        )
    }
}

/// Long-lived stateful writer for a MemWAL shard.
///
/// Supports writing batches, querying statistics, creating LSM scanners,
/// and graceful shutdown. Supports the Python context manager protocol.
#[pyclass(name = "_ShardWriter", module = "_lib")]
pub struct PyShardWriter {
    inner: Arc<TokioMutex<Option<ShardWriter>>>,
    closed_state: Arc<TokioMutex<Option<ClosedShardWriterState>>>,
    shard_id: Uuid,
    dataset: Arc<LanceDataset>,
}

#[derive(Clone)]
struct ClosedShardWriterState {
    stats: WriteStatsSnapshot,
    memtable_stats: MemTableStats,
}

fn collect_record_batches(data: &Bound<'_, PyAny>) -> PyResult<Vec<RecordBatch>> {
    let reader = ArrowArrayStreamReader::from_pyarrow_bound(data)
        .map_err(|e| PyValueError::new_err(format!("Cannot read data as Arrow: {}", e)))?;
    reader
        .collect::<Result<_, _>>()
        .map_err(|e| PyIOError::new_err(format!("Failed to read batches: {}", e)))
}

#[pymethods]
impl PyShardWriter {
    /// Write data batches to the MemWAL.
    ///
    /// Accepts any PyArrow-compatible data source (RecordBatch, Table,
    /// or an Arrow stream reader).
    pub fn put(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let batches = collect_record_batches(data)?;

        if batches.is_empty() {
            return Ok(());
        }

        let inner = self.inner.clone();
        rt().block_on(Some(py), async move {
            let guard = inner.lock().await;
            match guard.as_ref() {
                Some(writer) => writer.put(batches).await.map(|_| ()),
                None => Err(lance_core::Error::invalid_input(
                    "ShardWriter is already closed",
                )),
            }
        })?
        .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))
    }

    /// Delete rows from the MemWAL by primary key.
    ///
    /// Accepts any PyArrow-compatible data source carrying the shard's primary
    /// key column(s). Rust core validates that primary keys exist and builds the
    /// tombstone rows.
    pub fn delete(&self, py: Python<'_>, keys: &Bound<'_, PyAny>) -> PyResult<()> {
        let batches = collect_record_batches(keys)?;

        if batches.is_empty() {
            return Ok(());
        }

        let inner = self.inner.clone();
        rt().block_on(Some(py), async move {
            let guard = inner.lock().await;
            match guard.as_ref() {
                Some(writer) => writer.delete(batches).await.map(|_| ()),
                None => Err(lance_core::Error::invalid_input(
                    "ShardWriter is already closed",
                )),
            }
        })?
        .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))
    }

    /// Flush pending data and close the writer.
    ///
    /// After close(), calling put() will raise an error.
    /// This is called automatically when using the context manager.
    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let closed_state = self.closed_state.clone();
        rt().block_on(Some(py), async move {
            let mut guard = inner.lock().await;
            if let Some(writer) = guard.take() {
                let stats_handle = writer.stats_handle();
                // Snapshot stats before close so the captured state reflects
                // what was written, not any internal bookkeeping done by close().
                let stats_snapshot = stats_handle.snapshot();
                let memtable_stats_before_close = writer.memtable_stats().await?;
                writer.close().await?;
                let closed_memtable_stats = closed_memtable_stats(memtable_stats_before_close);
                let mut closed_guard = closed_state.lock().await;
                *closed_guard = Some(ClosedShardWriterState {
                    stats: stats_snapshot,
                    memtable_stats: closed_memtable_stats,
                });
                Ok(())
            } else {
                Ok(())
            }
        })?
        .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))
    }

    /// Return a snapshot of current write statistics.
    ///
    /// Returns a dict with keys: put_count, put_time_ms, wal_flush_count,
    /// wal_flush_bytes, memtable_flush_count, memtable_flush_rows.
    pub fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let closed_state = self.closed_state.clone();
        let stats = rt()
            .block_on(Some(py), async move {
                let guard = inner.lock().await;
                if let Some(writer) = guard.as_ref() {
                    Ok(writer.stats())
                } else {
                    let closed_guard = closed_state.lock().await;
                    closed_guard
                        .as_ref()
                        .map(|state| state.stats.clone())
                        .ok_or_else(|| {
                            lance_core::Error::invalid_input("ShardWriter is already closed")
                        })
                }
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        write_stats_to_pydict(py, &stats)
    }

    /// Return current MemTable statistics.
    ///
    /// Returns a dict with keys: row_count, batch_count, estimated_size_bytes,
    /// index_bytes, frozen_bytes, generation.
    pub fn memtable_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let closed_state = self.closed_state.clone();
        let (stats, bytes) = rt()
            .block_on(Some(py), async move {
                let guard = inner.lock().await;
                match guard.as_ref() {
                    // Byte totals come from `memory()`, the lock-free view;
                    // `memtable_stats` carries none.
                    Some(w) => w
                        .memtable_stats()
                        .await
                        .map(|stats| (stats, Some(w.memory()))),
                    None => {
                        let closed_guard = closed_state.lock().await;
                        closed_guard
                            .as_ref()
                            // A closed writer holds nothing, so the byte totals
                            // are zero by construction rather than stale.
                            .map(|state| (state.memtable_stats.clone(), None))
                            .ok_or_else(|| {
                                lance_core::Error::invalid_input("ShardWriter is already closed")
                            })
                    }
                }
            })?
            .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))?;

        memtable_stats_to_pydict(py, &stats, bytes.as_ref())
    }

    /// Create an LSM scanner that includes the active MemTable for strong consistency.
    ///
    /// The scanner covers: base table + given SSTables + current active MemTable.
    #[pyo3(signature = (shard_snapshots=vec![]))]
    pub fn lsm_scanner(
        &self,
        py: Python<'_>,
        shard_snapshots: Vec<Bound<'_, PyShardSnapshot>>,
    ) -> PyResult<PyLsmScanner> {
        let mut snapshots: Vec<ShardSnapshot> = shard_snapshots
            .iter()
            .map(|s| s.borrow().inner.clone())
            .collect();

        let pk_columns = get_pk_columns(&self.dataset)?;
        let inner = self.inner.clone();
        let dataset = self.dataset.clone();
        let shard_id = self.shard_id;

        let (active_ref, writer_snapshot) = rt()
            .block_on(Some(py), async move {
                let guard = inner.lock().await;
                match guard.as_ref() {
                    Some(w) => {
                        let active_ref = w.active_memtable_ref().await?;
                        let writer_snapshot = w
                            .manifest()
                            .await?
                            .map(shard_snapshot_from_manifest)
                            .unwrap_or_else(|| ShardSnapshot::new(shard_id));
                        Ok((active_ref, writer_snapshot))
                    }
                    None => Err(lance_core::Error::invalid_input(
                        "ShardWriter is already closed",
                    )),
                }
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        snapshots.retain(|snapshot| snapshot.shard_id != shard_id);
        snapshots.push(writer_snapshot);

        let scanner = LsmScanner::new(dataset, snapshots, pk_columns)
            .with_active_memtable(shard_id, active_ref);

        Ok(PyLsmScanner {
            inner: Some(scanner),
        })
    }

    /// Return the shard ID as a UUID string.
    #[getter]
    pub fn shard_id(&self) -> String {
        self.shard_id.to_string()
    }

    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }
}

impl PyShardWriter {
    /// Create from a Rust ShardWriter and dataset reference.
    pub fn new(writer: ShardWriter, shard_id: Uuid, dataset: Arc<LanceDataset>) -> Self {
        Self {
            inner: Arc::new(TokioMutex::new(Some(writer))),
            closed_state: Arc::new(TokioMutex::new(None)),
            shard_id,
            dataset,
        }
    }
}

/// Python wrapper around a DataFusion physical execution plan.
#[pyclass(name = "_ExecutionPlan", module = "_lib", skip_from_py_object)]
#[derive(Clone)]
pub struct PyExecutionPlan {
    plan: Arc<dyn ExecutionPlan>,
    dataset_schema: Arc<ArrowSchema>,
}

impl PyExecutionPlan {
    pub fn new(plan: Arc<dyn ExecutionPlan>, dataset_schema: Arc<ArrowSchema>) -> Self {
        Self {
            plan,
            dataset_schema,
        }
    }
}

#[pymethods]
impl PyExecutionPlan {
    #[getter]
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.plan.schema().to_pyarrow(py)
    }

    #[getter]
    fn dataset_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.dataset_schema.to_pyarrow(py)
    }

    fn explain(&self) -> String {
        format!("{}", displayable(self.plan.as_ref()).indent(true))
    }

    fn to_batches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let plan = self.plan.clone();
        let batches = rt()
            .block_on(Some(py), async move {
                let ctx = SessionContext::new();
                collect(plan, ctx.task_ctx()).await
            })?
            .map_err(|e| PyIOError::new_err(format!("Plan execution failed: {}", e)))?;

        let py_batches: Vec<Bound<'py, PyAny>> = batches
            .into_iter()
            .map(|batch| {
                PyArrowType(batch)
                    .into_pyobject(py)
                    .map(|batch| batch.into_any())
                    .map_err(|e| PyIOError::new_err(e.to_string()))
            })
            .collect::<PyResult<_>>()?;
        PyList::new(py, py_batches)
    }

    fn to_reader<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let plan = self.plan.clone();
        let batches = rt()
            .block_on(Some(py), async move {
                let ctx = SessionContext::new();
                collect(plan, ctx.task_ctx()).await
            })?
            .map_err(|e| PyIOError::new_err(format!("Plan execution failed: {}", e)))?;

        let schema = self.plan.schema().clone();
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            schema,
        ));
        reader.into_pyarrow(py)
    }
}

/// LSM-aware scanner covering base table, SSTables, and active MemTable.
///
/// Provides deduplication by primary key, always returning the newest version
/// of each row across all LSM levels.
#[pyclass(name = "_LsmScanner", module = "_lib")]
pub struct PyLsmScanner {
    inner: Option<LsmScanner>,
}

#[pymethods]
impl PyLsmScanner {
    /// Create a scanner from dataset and shard snapshots (without active MemTable).
    #[staticmethod]
    pub fn from_snapshots(
        dataset: &Bound<'_, PyDataset>,
        shard_snapshots: Vec<Bound<'_, PyShardSnapshot>>,
    ) -> PyResult<Self> {
        let ds = dataset.borrow().ds.clone();
        let snapshots: Vec<ShardSnapshot> = shard_snapshots
            .iter()
            .map(|s| s.borrow().inner.clone())
            .collect();
        let pk_columns = get_pk_columns(&ds)?;
        Ok(Self {
            inner: Some(LsmScanner::new(ds, snapshots, pk_columns)),
        })
    }

    /// Select specific columns to return.
    pub fn project(
        mut slf: PyRefMut<'_, Self>,
        columns: Vec<String>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        let cols: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        slf.inner = Some(
            scanner
                .project(&cols)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        Ok(slf)
    }

    /// Set a SQL filter expression.
    pub fn filter(mut slf: PyRefMut<'_, Self>, expr: String) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        slf.inner = Some(
            scanner
                .filter(&expr)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        Ok(slf)
    }

    /// Limit the number of rows returned, optionally with an offset.
    #[pyo3(signature = (n=None, offset=None))]
    pub fn limit(
        mut slf: PyRefMut<'_, Self>,
        n: Option<usize>,
        offset: Option<usize>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        slf.inner = Some(
            scanner
                .limit(n.map(|n| n as i64), offset.map(|o| o as i64))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        Ok(slf)
    }

    /// Include the `_rowaddr` internal column in results.
    pub fn with_row_address(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        slf.inner = Some(scanner.with_row_address());
        Ok(slf)
    }

    /// Include the `_memtable_gen` internal column in results.
    pub fn with_memtable_gen(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        slf.inner = Some(scanner.with_memtable_gen());
        Ok(slf)
    }

    /// Execute the scan and return a single PyArrow RecordBatch.
    pub fn to_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let scanner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        let batch = rt()
            .block_on(Some(py), scanner.try_into_batch())
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        PyArrowType(batch).into_pyobject(py).map(|b| b.into_any())
    }

    /// Execute the scan and return all batches as a Python list.
    pub fn to_batches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let scanner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        let stream = rt()
            .block_on(Some(py), scanner.try_into_stream())
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let batches: Vec<RecordBatch> = rt()
            .block_on(Some(py), stream.try_collect())
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let py_batches: Vec<Bound<'py, PyAny>> = batches
            .into_iter()
            .map(|b| {
                PyArrowType(b)
                    .into_pyobject(py)
                    .map(|b| b.into_any())
                    .map_err(|e| PyIOError::new_err(e.to_string()))
            })
            .collect::<PyResult<_>>()?;
        PyList::new(py, py_batches)
    }

    /// Return the row count without loading all data.
    pub fn count_rows(&self, py: Python<'_>) -> PyResult<u64> {
        let scanner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        rt().block_on(Some(py), scanner.count_rows())
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }
}

/// Plans and executes primary key point lookups across all LSM levels.
///
/// More efficient than `_LsmScanner` for known-PK lookups due to bloom filter
/// optimizations and short-circuit evaluation.
#[pyclass(name = "_LsmPointLookupPlanner", module = "_lib")]
pub struct PyLsmPointLookupPlanner {
    planner: LsmPointLookupPlanner,
    dataset_schema: Arc<ArrowSchema>,
    pk_columns: Vec<String>,
}

#[pymethods]
impl PyLsmPointLookupPlanner {
    #[new]
    #[pyo3(signature = (dataset, shard_snapshots, pk_columns=None))]
    pub fn new(
        dataset: &Bound<'_, PyDataset>,
        shard_snapshots: Vec<Bound<'_, PyShardSnapshot>>,
        pk_columns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let ds = dataset.borrow().ds.clone();
        let snapshots: Vec<ShardSnapshot> = shard_snapshots
            .iter()
            .map(|s| s.borrow().inner.clone())
            .collect();
        let pk_cols = match pk_columns {
            Some(cols) => cols,
            None => get_pk_columns(&ds)?,
        };
        let base_schema = Arc::new(ArrowSchema::from(ds.schema()));
        let collector = LsmDataSourceCollector::new(ds.clone(), snapshots);
        let planner = LsmPointLookupPlanner::new(collector, pk_cols.clone(), base_schema.clone());
        Ok(Self {
            planner,
            dataset_schema: base_schema,
            pk_columns: pk_cols,
        })
    }

    /// Plan a single-row point lookup by primary key.
    ///
    /// For single-column primary keys, `pk_value` should be a PyArrow array
    /// with exactly one element. For composite primary keys, `pk_value` must
    /// be a StructArray with exactly one row and one field per PK column.
    #[pyo3(signature = (pk_value, columns=None))]
    pub fn plan_lookup(
        &self,
        py: Python<'_>,
        pk_value: PyArrowType<ArrayData>,
        columns: Option<Vec<String>>,
    ) -> PyResult<PyExecutionPlan> {
        let array = make_array(pk_value.0);
        let pk_values = scalar_values_from_pk_value(array.as_ref(), &self.pk_columns)?;
        let proj: Option<Vec<String>> = columns;
        let planner_ref = &self.planner;
        let plan = rt()
            .block_on(Some(py), async {
                planner_ref.plan_lookup(&pk_values, proj.as_deref()).await
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(PyExecutionPlan::new(plan, self.dataset_schema.clone()))
    }
}

/// Plans and executes vector KNN search across all LSM levels.
///
/// Only supports IVF-PQ vector indexes maintained in MemWAL.
/// Results include staleness filtering to return only the latest version of each row.
#[pyclass(name = "_LsmVectorSearchPlanner", module = "_lib")]
pub struct PyLsmVectorSearchPlanner {
    planner: LsmVectorSearchPlanner,
    vector_dim: usize,
    dataset_schema: Arc<ArrowSchema>,
}

#[pymethods]
impl PyLsmVectorSearchPlanner {
    #[new]
    #[pyo3(signature = (dataset, shard_snapshots, vector_column, pk_columns=None, distance_type=None, filter=None))]
    pub fn new(
        dataset: &Bound<'_, PyDataset>,
        shard_snapshots: Vec<Bound<'_, PyShardSnapshot>>,
        vector_column: String,
        pk_columns: Option<Vec<String>>,
        distance_type: Option<String>,
        filter: Option<String>,
    ) -> PyResult<Self> {
        let ds = dataset.borrow().ds.clone();
        let snapshots: Vec<ShardSnapshot> = shard_snapshots
            .iter()
            .map(|s| s.borrow().inner.clone())
            .collect();
        let pk_cols = match pk_columns {
            Some(cols) => cols,
            None => get_pk_columns(&ds)?,
        };
        let base_schema = Arc::new(ArrowSchema::from(ds.schema()));

        let dist_type = parse_distance_type(distance_type.as_deref().unwrap_or("l2"))?;

        let vector_dim = get_vector_dim(&ds, &vector_column)?;
        let filter = filter
            .as_deref()
            .map(|filter| {
                parse_lsm_filter_expr(base_schema.as_ref(), filter)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
            .transpose()?;

        let collector = LsmDataSourceCollector::new(ds.clone(), snapshots);
        let mut planner = LsmVectorSearchPlanner::new(
            collector,
            pk_cols,
            base_schema.clone(),
            vector_column,
            dist_type,
        )
        .with_dataset(ds);
        if let Some(filter) = filter {
            planner = planner.with_filter(Some(filter));
        }

        Ok(Self {
            planner,
            vector_dim,
            dataset_schema: base_schema,
        })
    }

    /// Plan a KNN vector search.
    ///
    /// `query` should be a flat PyArrow Float32Array with `vector_dim` elements.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, k=10, nprobes=20, columns=None, refine_base_table=false, overfetch_factor=1.0))]
    pub fn plan_search(
        &self,
        py: Python<'_>,
        query: PyArrowType<ArrayData>,
        k: usize,
        nprobes: usize,
        columns: Option<Vec<String>>,
        refine_base_table: bool,
        overfetch_factor: f64,
    ) -> PyResult<PyExecutionPlan> {
        let query_array = make_array(query.0);
        let float32_array = query_array
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                PyValueError::new_err(
                    "query must be a Float32Array. Use pa.array(values, type=pa.float32())",
                )
            })?;

        if float32_array.len() != self.vector_dim {
            return Err(PyValueError::new_err(format!(
                "Query vector has {} dimensions, expected {}",
                float32_array.len(),
                self.vector_dim
            )));
        }

        // Wrap the flat array into a FixedSizeListArray with one row
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let fsl = FixedSizeListArray::try_new(
            field,
            self.vector_dim as i32,
            Arc::new(float32_array.clone()),
            None,
        )
        .map_err(|e| PyValueError::new_err(format!("Cannot create query vector: {}", e)))?;

        let planner_ref = &self.planner;
        let plan = rt()
            .block_on(Some(py), async {
                planner_ref
                    .plan_search(
                        &fsl,
                        k,
                        nprobes,
                        columns.as_deref(),
                        refine_base_table,
                        overfetch_factor,
                    )
                    .await
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(PyExecutionPlan::new(plan, self.dataset_schema.clone()))
    }
}

/// Extract primary key column names from dataset schema.
pub(crate) fn get_pk_columns(ds: &LanceDataset) -> PyResult<Vec<String>> {
    let pk_fields = ds.schema().unenforced_primary_key();
    if pk_fields.is_empty() {
        return Err(PyValueError::new_err(
            "Dataset has no primary key. Set 'lance-schema:unenforced-primary-key' metadata \
             on the primary key field(s).",
        ));
    }
    Ok(pk_fields.iter().map(|f| f.name.clone()).collect())
}

/// Parse distance type string to DistanceType enum.
fn parse_distance_type(s: &str) -> PyResult<DistanceType> {
    match s.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok(DistanceType::L2),
        "cosine" => Ok(DistanceType::Cosine),
        "dot" | "inner_product" => Ok(DistanceType::Dot),
        "hamming" => Ok(DistanceType::Hamming),
        _ => Err(PyValueError::new_err(format!(
            "Unknown distance_type '{}'. Valid values: 'l2', 'cosine', 'dot', 'hamming'",
            s
        ))),
    }
}

/// Get the vector dimension from the dataset schema for a given column.
fn get_vector_dim(ds: &LanceDataset, column: &str) -> PyResult<usize> {
    let schema = ArrowSchema::from(ds.schema());
    let field = schema.field_with_name(column).map_err(|_| {
        PyValueError::new_err(format!("Column '{}' not found in dataset schema", column))
    })?;
    match field.data_type() {
        DataType::FixedSizeList(_, size) => Ok(*size as usize),
        other => Err(PyValueError::new_err(format!(
            "Column '{}' is not a FixedSizeList (got {:?}). \
             Vector columns must be FixedSizeList<float32>.",
            column, other
        ))),
    }
}

fn write_stats_to_pydict(py: Python<'_>, stats: &WriteStatsSnapshot) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("put_count", stats.put_count)?;
    dict.set_item("put_time_ms", stats.put_time.as_millis() as u64)?;
    dict.set_item("wal_flush_count", stats.wal_flush_count)?;
    dict.set_item("wal_flush_bytes", stats.wal_flush_bytes)?;
    dict.set_item("wal_flush_time_ms", stats.wal_flush_time.as_millis() as u64)?;
    dict.set_item("memtable_flush_count", stats.memtable_flush_count)?;
    dict.set_item("memtable_flush_rows", stats.memtable_flush_rows)?;
    dict.set_item(
        "memtable_flush_time_ms",
        stats.memtable_flush_time.as_millis() as u64,
    )?;
    Ok(dict.into_any().unbind())
}

fn memtable_stats_to_pydict(
    py: Python<'_>,
    stats: &MemTableStats,
    bytes: Option<&ShardMemory>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("row_count", stats.row_count)?;
    dict.set_item("batch_count", stats.batch_count)?;
    // Row data only, as this key has always meant; `index_bytes` below is the
    // rest of the active memtable's footprint.
    dict.set_item(
        "estimated_size_bytes",
        bytes.map_or(0, ShardMemory::row_bytes),
    )?;
    dict.set_item("index_bytes", bytes.map_or(0, ShardMemory::index_bytes))?;
    dict.set_item("generation", stats.generation)?;
    dict.set_item(
        "max_buffered_batch_position",
        stats.max_buffered_batch_position,
    )?;
    dict.set_item("durable_batch_count", stats.durable_batch_count)?;
    dict.set_item("global_offset", stats.global_offset)?;
    dict.set_item(
        "pending_wal_start_batch_position",
        stats.pending_wal_start_batch_position,
    )?;
    dict.set_item(
        "pending_wal_end_batch_position",
        stats.pending_wal_end_batch_position,
    )?;
    dict.set_item("pending_wal_batch_count", stats.pending_wal_batch_count)?;
    dict.set_item("pending_wal_row_count", stats.pending_wal_row_count)?;
    dict.set_item(
        "pending_wal_estimated_bytes",
        stats.pending_wal_estimated_bytes,
    )?;
    dict.set_item("frozen_count", stats.frozen_count)?;
    dict.set_item("frozen_bytes", bytes.map_or(0, ShardMemory::frozen_bytes))?;
    // Flushed generations still lingering out `frozen_memtable_grace`. Resident,
    // but no flush reclaims them, so they are not in `frozen_bytes`.
    dict.set_item("grace_bytes", bytes.map_or(0, ShardMemory::grace_bytes))?;
    // The whole footprint: what a process-wide budget meters, as opposed to
    // `frozen_bytes` + active, which is only what a flush can give back.
    dict.set_item(
        "retained_bytes",
        bytes.map_or(0, ShardMemory::retained_bytes),
    )?;
    Ok(dict.into_any().unbind())
}

fn scalar_values_from_pk_value(
    pk_value: &dyn Array,
    pk_columns: &[String],
) -> PyResult<Vec<ScalarValue>> {
    if pk_value.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "pk_value must contain exactly one row, got {}",
            pk_value.len()
        )));
    }

    if pk_columns.len() == 1 {
        let scalar = ScalarValue::try_from_array(pk_value, 0)
            .map_err(|e| PyValueError::new_err(format!("Cannot convert pk_value: {}", e)))?;
        return Ok(vec![scalar]);
    }

    let struct_array = pk_value.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
        PyValueError::new_err(format!(
            "Composite primary key lookup requires a StructArray with exactly one row and {} fields",
            pk_columns.len()
        ))
    })?;

    if struct_array.num_columns() != pk_columns.len() {
        return Err(PyValueError::new_err(format!(
            "Composite primary key lookup expected {} fields, got {}",
            pk_columns.len(),
            struct_array.num_columns()
        )));
    }

    let mut pk_values = Vec::with_capacity(pk_columns.len());
    for column_name in pk_columns {
        let column = struct_array.column_by_name(column_name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Composite primary key lookup requires field '{}' in pk_value",
                column_name
            ))
        })?;
        let scalar = ScalarValue::try_from_array(column.as_ref(), 0).map_err(|e| {
            PyValueError::new_err(format!("Cannot convert composite pk_value: {}", e))
        })?;
        pk_values.push(scalar);
    }
    Ok(pk_values)
}

fn shard_snapshot_from_manifest(manifest: lance_index::mem_wal::ShardManifest) -> ShardSnapshot {
    ShardSnapshot {
        shard_id: manifest.shard_id,
        spec_id: manifest.shard_spec_id,
        current_generation: manifest.current_generation,
        sstables: manifest
            .sstables
            .into_iter()
            .map(|sstable| SsTable {
                generation: sstable.generation,
                path: sstable.path,
            })
            .collect(),
    }
}

fn closed_memtable_stats(stats_before_close: MemTableStats) -> MemTableStats {
    // Close awaits every frozen memtable's flush, so nothing is owed afterwards
    // regardless of whether the active memtable had buffered batches.
    let stats_before_close = MemTableStats {
        frozen_count: 0,
        ..stats_before_close
    };

    if stats_before_close.batch_count == 0 {
        return stats_before_close;
    }

    // After a successful close every buffered batch is flushed and WAL-durable,
    // so the synthesized empty memtable starts at the writer's global end and the
    // durable cursor has caught up to it.
    let global_end = stats_before_close.global_offset + stats_before_close.batch_count;
    MemTableStats {
        row_count: 0,
        batch_count: 0,
        generation: stats_before_close.generation.saturating_add(1),
        max_buffered_batch_position: None,
        durable_batch_count: global_end,
        global_offset: global_end,
        pending_wal_start_batch_position: None,
        pending_wal_end_batch_position: None,
        pending_wal_batch_count: 0,
        pending_wal_row_count: 0,
        pending_wal_estimated_bytes: 0,
        frozen_count: 0,
    }
}
