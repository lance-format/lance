// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python merge-insert builder.

use std::sync::Arc;

use arrow::pyarrow::PyArrowType;
use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use lance::dataset::{
    MergeInsertAnalysis, MergeInsertBuilder as LanceMergeInsertBuilder, MergeInsertWriteMode,
    MergeInsertWriteStrategy, MergeStats, UncommittedMergeInsert, WhenMatched, WhenNotMatched,
    WhenNotMatchedBySource, transaction::Transaction,
};
use lance_datafusion::exec::{ExecutionMetricValue, ExecutionOutcome};
use lance_datafusion::utils::reader_to_stream;
use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::error::PythonErrorExt;
use crate::rt;
use crate::utils::PyLance;

use super::replayable::replayable_provider;
use super::{Dataset, convert_reader};

#[pyclass(name = "_MergeInsertBuilder", module = "_lib", subclass)]
pub struct MergeInsertBuilder {
    builder: LanceMergeInsertBuilder,
    dataset: Py<Dataset>,
}

#[pymethods]
impl MergeInsertBuilder {
    #[new]
    #[pyo3(signature=(dataset, on=None))]
    pub fn new(dataset: &Bound<'_, PyAny>, on: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let dataset_py: Py<Dataset> = dataset.extract()?;
        let py = dataset.py();
        let ds = dataset_py.borrow(py).ds.clone();

        // Either a single string, which we put in a vector or an iterator
        // of strings, which we collect into a vector. If `on` is None, we
        // pass an empty vector and let the Rust builder fall back to the
        // schema's unenforced primary key (if configured).
        let on = if let Some(on_any) = on {
            on_any
                .cast::<PyString>()
                .map(|val| vec![val.to_string()])
                .or_else(|_| {
                    let iterator = on_any.try_iter().map_err(|_| {
                        PyTypeError::new_err(
                            "The `on` argument to merge_insert must be a str or iterable of str",
                        )
                    })?;
                    let mut keys = Vec::new();
                    for key in iterator {
                        keys.push(key?.cast::<PyString>()?.to_string());
                    }
                    PyResult::Ok(keys)
                })?
        } else {
            Vec::new()
        };

        let mut builder = LanceMergeInsertBuilder::try_new(ds, on)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        // We don't have do_nothing methods in python so we start with a blank slate
        builder
            .when_matched(WhenMatched::DoNothing)
            .when_not_matched(WhenNotMatched::DoNothing);

        Ok(Self {
            builder,
            dataset: dataset_py,
        })
    }

    #[pyo3(signature=(condition=None))]
    pub fn when_matched_update_all<'a>(
        mut slf: PyRefMut<'a, Self>,
        condition: Option<&str>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let new_val = if let Some(expr) = condition {
            let dataset = slf.dataset.borrow(slf.py());
            WhenMatched::update_if(&dataset.ds, expr)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
        } else {
            WhenMatched::UpdateAll
        };
        slf.builder.when_matched(new_val);
        Ok(slf)
    }

    pub fn when_matched_fail(mut slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.builder.when_matched(WhenMatched::Fail);
        Ok(slf)
    }

    pub fn when_matched_delete(mut slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.builder.when_matched(WhenMatched::Delete);
        Ok(slf)
    }

    pub fn when_not_matched_insert_all(mut slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.builder.when_not_matched(WhenNotMatched::InsertAll);
        Ok(slf)
    }

    #[pyo3(signature=(expr=None))]
    pub fn when_not_matched_by_source_delete<'a>(
        mut slf: PyRefMut<'a, Self>,
        expr: Option<&str>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let new_val = if let Some(expr) = expr {
            let dataset = slf.dataset.borrow(slf.py());
            WhenNotMatchedBySource::delete_if(&dataset.ds, expr)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
        } else {
            WhenNotMatchedBySource::Delete
        };
        slf.builder.when_not_matched_by_source(new_val);
        Ok(slf)
    }

    pub fn conflict_retries(
        mut slf: PyRefMut<'_, Self>,
        max_retries: u32,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.conflict_retries(max_retries);
        Ok(slf)
    }

    pub fn retry_timeout(
        mut slf: PyRefMut<'_, Self>,
        timeout: std::time::Duration,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.retry_timeout(timeout);
        Ok(slf)
    }

    pub fn data_storage_version<'a>(
        mut slf: PyRefMut<'a, Self>,
        version: &str,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.builder
            .data_storage_version(version.parse().infer_error()?);
        Ok(slf)
    }

    pub fn use_index(mut slf: PyRefMut<'_, Self>, use_index: bool) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.use_index(use_index);
        Ok(slf)
    }

    pub fn write_mode<'a>(mut slf: PyRefMut<'a, Self>, mode: &str) -> PyResult<PyRefMut<'a, Self>> {
        let mode = match mode {
            "auto" => MergeInsertWriteMode::Auto,
            "rewrite_rows" => MergeInsertWriteMode::RewriteRows,
            "rewrite_columns" => MergeInsertWriteMode::RewriteColumns,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Invalid write_mode: {other}. Expected one of \
                     'auto', 'rewrite_rows', 'rewrite_columns'"
                )));
            }
        };
        slf.builder.write_mode(mode);
        Ok(slf)
    }

    pub fn target_bases(
        mut slf: PyRefMut<'_, Self>,
        bases: Vec<String>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.target_base_names_or_paths(bases);
        Ok(slf)
    }

    #[pyo3(signature = (include_primary = true))]
    pub fn target_all_bases(
        mut slf: PyRefMut<'_, Self>,
        include_primary: bool,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.target_all_bases(include_primary);
        Ok(slf)
    }

    pub fn execute(&mut self, new_data: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = new_data.py();
        let new_data = convert_reader(new_data)?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let (new_dataset, stats) = rt()
            .spawn(Some(py), job.execute_reader(new_data))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        let dataset = self.dataset.bind(py);

        dataset.borrow_mut().ds = new_dataset;

        Ok(Self::build_stats(&stats, py)?.into())
    }

    pub fn execute_uncommitted<'a>(
        &mut self,
        new_data: &Bound<'a, PyAny>,
    ) -> PyResult<(PyLance<Transaction>, Bound<'a, PyDict>)> {
        let py = new_data.py();
        let new_data = convert_reader(new_data)?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let UncommittedMergeInsert {
            transaction, stats, ..
        } = rt()
            .spawn(Some(py), job.execute_uncommitted(new_data))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        let stats = Self::build_stats(&stats, py)?;

        Ok((PyLance(transaction), stats))
    }

    pub fn execute_replayable(
        &mut self,
        schema: PyArrowType<ArrowSchema>,
        factory: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = factory.py();
        let provider = replayable_provider(Arc::new(schema.0), factory)?;
        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let (new_dataset, stats) = rt()
            .spawn(Some(py), job.execute_provider(provider))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.dataset.bind(py).borrow_mut().ds = new_dataset;
        Ok(Self::build_stats(&stats, py)?.into())
    }

    pub fn execute_dataset(&mut self, source: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = source.py();
        let source = source.extract::<PyRef<'_, Dataset>>()?.ds.clone();
        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let (dataset, stats) = rt()
            .spawn(Some(py), job.execute_dataset(source))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.dataset.bind(py).borrow_mut().ds = dataset;
        Ok(Self::build_stats(&stats, py)?.into())
    }

    /// Execute the merge insert from fully-materialized data.
    ///
    /// The data is read into memory and wrapped in an in-memory table, so retries
    /// never spill to disk and the source's statistics drive the join. Callers
    /// should only route in-memory inputs (e.g. a `pa.Table`) here.
    pub fn execute_batches(&mut self, new_data: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
        let py = new_data.py();
        let reader = convert_reader(new_data)?;
        let batches = reader
            .collect::<std::result::Result<Vec<RecordBatch>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let (new_dataset, stats) = rt()
            .spawn(Some(py), job.execute_batches(batches))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        let dataset = self.dataset.bind(py);
        dataset.borrow_mut().ds = new_dataset;

        Ok(Self::build_stats(&stats, py)?.into())
    }

    /// [`Self::execute_batches`] without committing; returns the transaction.
    pub fn execute_uncommitted_dataset<'a>(
        &mut self,
        source: &Bound<'a, PyAny>,
    ) -> PyResult<(PyLance<Transaction>, Bound<'a, PyDict>)> {
        let py = source.py();
        let source = source.extract::<PyRef<'_, Dataset>>()?.ds.clone();
        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let UncommittedMergeInsert {
            transaction, stats, ..
        } = rt()
            .spawn(Some(py), job.execute_uncommitted_dataset(source))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        let stats = Self::build_stats(&stats, py)?;
        Ok((PyLance(transaction), stats))
    }

    /// [`Self::execute_batches`] without committing; returns the transaction.
    pub fn execute_uncommitted_batches<'a>(
        &mut self,
        new_data: &Bound<'a, PyAny>,
    ) -> PyResult<(PyLance<Transaction>, Bound<'a, PyDict>)> {
        let py = new_data.py();
        let reader = convert_reader(new_data)?;
        let batches = reader
            .collect::<std::result::Result<Vec<RecordBatch>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let UncommittedMergeInsert {
            transaction, stats, ..
        } = rt()
            .spawn(Some(py), job.execute_uncommitted_batches(batches))?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;

        let stats = Self::build_stats(&stats, py)?;

        Ok((PyLance(transaction), stats))
    }

    #[pyo3(signature=(schema = None, verbose = false))]
    pub fn explain_plan(
        &mut self,
        schema: Option<PyArrowType<ArrowSchema>>,
        verbose: Option<bool>,
    ) -> PyResult<String> {
        let verbose = verbose.unwrap_or(false);
        let job = self
            .builder
            .clone()
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let schema_ref = schema.as_ref().map(|s| &s.0);
        rt().block_on(None, job.explain_plan(schema_ref, verbose))?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    pub fn analyze_plan(&mut self, new_data: &Bound<PyAny>) -> PyResult<String> {
        let new_data_reader = convert_reader(new_data)?;
        let new_data_stream = reader_to_stream(new_data_reader);
        let job = self
            .builder
            .clone()
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        rt().block_on(None, job.analyze_plan(new_data_stream))?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    pub fn analyze_plan_replayable(
        &mut self,
        schema: PyArrowType<ArrowSchema>,
        factory: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        let provider = replayable_provider(Arc::new(schema.0), factory)?;
        let job = self
            .builder
            .clone()
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        rt().spawn(Some(factory.py()), async move {
            job.analyze_plan_provider(provider).await
        })?
        .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    pub fn analyze_plan_dataset(&mut self, source: &Bound<'_, PyAny>) -> PyResult<String> {
        let py = source.py();
        let source = source.extract::<PyRef<'_, Dataset>>()?.ds.clone();
        let job = self
            .builder
            .clone()
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        rt().spawn(
            Some(py),
            async move { job.analyze_plan_dataset(source).await },
        )?
        .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    /// [`Self::analyze_plan`] for fully-materialized data.
    ///
    /// Routed to the same in-memory table `execute_batches` uses, so the reported
    /// plan is the one such a source actually runs.
    pub fn analyze_plan_batches(&mut self, new_data: &Bound<PyAny>) -> PyResult<String> {
        let reader = convert_reader(new_data)?;
        let batches = reader
            .collect::<std::result::Result<Vec<RecordBatch>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let job = self
            .builder
            .clone()
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        rt().block_on(None, job.analyze_plan_batches(batches))?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    pub fn _analyze_plan_report<'py>(
        &self,
        py: Python<'py>,
        new_data: &Bound<PyAny>,
        is_materialized: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let reader = convert_reader(new_data)?;
        let job = self
            .builder
            .clone()
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let report = if is_materialized {
            let batches = reader
                .collect::<std::result::Result<Vec<RecordBatch>, _>>()
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            rt().block_on(Some(py), job.analyze_plan_report_batches(batches))?
        } else {
            rt().block_on(Some(py), job.analyze_plan_report(reader_to_stream(reader)))?
        }
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Self::build_analysis_report(report, py)
    }

    pub fn _analyze_plan_report_dataset<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let source = source.extract::<PyRef<'_, Dataset>>()?.ds.clone();
        let job = self.builder.clone().try_build().infer_error()?;
        let report = rt()
            .spawn(Some(py), async move {
                job.analyze_plan_report_dataset(source).await
            })?
            .infer_error()?;
        Self::build_analysis_report(report, py)
    }

    pub fn _analyze_plan_report_replayable<'py>(
        &self,
        py: Python<'py>,
        schema: PyArrowType<ArrowSchema>,
        factory: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let provider = replayable_provider(Arc::new(schema.0), factory)?;
        let job = self.builder.clone().try_build().infer_error()?;
        let report = rt()
            .spawn(Some(py), async move {
                job.analyze_plan_report_provider(provider).await
            })?
            .infer_error()?;
        Self::build_analysis_report(report, py)
    }

    /// Mark MemWAL SSTables as compacted into the base table.
    ///
    /// Call this when executing a merge_insert that compacts MemWAL SSTables.
    /// This updates MemWAL compaction progress to prevent duplicate compactions.
    pub fn mark_sstables_as_compacted<'a>(
        mut slf: PyRefMut<'a, Self>,
        sstables: Vec<Bound<'a, crate::mem_wal::PyCompactedSsTable>>,
        dir_id: String,
    ) -> PyResult<PyRefMut<'a, Self>> {
        use lance_index::mem_wal::CompactedSsTable;

        let compacted_sstables: Vec<CompactedSsTable> = sstables
            .iter()
            .map(|sstable| sstable.borrow().to_lance())
            .collect::<PyResult<_>>()?;
        slf.builder
            .mark_sstables_as_compacted(dir_id, compacted_sstables);
        Ok(slf)
    }
}

impl MergeInsertBuilder {
    fn build_analysis_report(
        report: MergeInsertAnalysis,
        py: Python<'_>,
    ) -> PyResult<Bound<'_, PyDict>> {
        let wire = report.to_proto();
        let dict = Self::build_execution_report(report.execution, py)?;
        dict.set_item("operation_id", report.operation_id.to_string())?;
        dict.set_item(
            "write_mode",
            match report.write_mode {
                MergeInsertWriteStrategy::DeleteOnly => "delete_only",
                MergeInsertWriteStrategy::RewriteRows => "rewrite_rows",
                MergeInsertWriteStrategy::RewriteColumns => "rewrite_columns",
            },
        )?;
        dict.set_item("fragments_touched", report.fragments_touched)?;
        if let Some(stats) = report.stats {
            let values = Self::build_stats(&stats, py)?;
            values.set_item("bytes_written", stats.bytes_written)?;
            values.set_item("num_files_written", stats.num_files_written)?;
            values.set_item("num_skipped_duplicates", stats.num_skipped_duplicates)?;
            dict.set_item("stats", values)?;
        } else {
            dict.set_item("stats", py.None())?;
        }
        dict.set_item("joins", Self::build_joins(report.joins, py)?)?;
        dict.set_item("target_rows_read", report.target_rows_read)?;
        dict.set_item("fragments_read", report.fragments_read)?;
        dict.set_item("index_used", report.index_used)?;
        dict.set_item("join_index_segments", report.join_index_segments)?;
        dict.set_item(
            "join_duration_ns",
            report.join_duration.map(|v| v.as_nanos()),
        )?;
        dict.set_item("joined_rows", report.joined_rows)?;
        dict.set_item(
            "merge_write_duration_ns",
            report.merge_write_duration.map(|v| v.as_nanos()),
        )?;
        dict.set_item("mapped_target_rows", report.mapped_target_rows)?;
        for (name, usage) in [
            ("operator_spills", wire.operator_spills),
            ("replay_spills", wire.replay_spills),
            ("scratch_spills", wire.scratch_spills),
        ] {
            if let Some(usage) = usage {
                let values = PyDict::new(py);
                for (field, value) in [
                    ("spill_count", usage.spill_count),
                    ("bytes_written", usage.bytes_written),
                    ("bytes_read", usage.bytes_read),
                    ("write_duration_ns", usage.write_duration_ns),
                    ("read_duration_ns", usage.read_duration_ns),
                    ("rows", usage.rows),
                    ("accepted_bytes", usage.accepted_bytes),
                    ("charged_bytes", usage.charged_bytes),
                    ("peak_charged_bytes", usage.peak_charged_bytes),
                    ("live_files", usage.live_files),
                    ("files_created", usage.files_created),
                    ("quota_failures", usage.quota_failures),
                    ("cleanup_failures", usage.cleanup_failures),
                ] {
                    values.set_item(field, value)?;
                }
                dict.set_item(name, values)?;
            } else {
                dict.set_item(name, py.None())?;
            }
        }
        let subplans = PyList::empty(py);
        for subplan in report.subplans {
            let node = Self::build_execution_report(subplan.execution, py)?;
            node.set_item("id", subplan.id)?;
            node.set_item("purpose", subplan.purpose)?;
            node.set_item("joins", Self::build_joins(subplan.joins, py)?)?;
            subplans.append(node)?;
        }
        dict.set_item("subplans", subplans)?;
        Ok(dict)
    }

    fn build_execution_report(
        execution: lance_datafusion::exec::ExecutionAnalysis,
        py: Python<'_>,
    ) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("plan", execution.plan)?;
        dict.set_item(
            "outcome",
            match execution.outcome {
                ExecutionOutcome::Complete => "complete",
                ExecutionOutcome::Failed => "failed",
                ExecutionOutcome::Cancelled => "cancelled",
            },
        )?;
        dict.set_item("error", execution.error)?;
        dict.set_item("elapsed_ns", execution.elapsed.as_nanos())?;
        dict.set_item(
            "pool_peak_reserved_bytes",
            execution.pool_peak_reserved_bytes,
        )?;
        let counts = PyDict::new(py);
        let summary = execution.summary;
        for (name, value) in summary.all_counts {
            counts.set_item(name, value)?;
        }
        counts.set_item("iops", summary.iops)?;
        counts.set_item("requests", summary.requests)?;
        counts.set_item("bytes_read", summary.bytes_read)?;
        counts.set_item("indices_loaded", summary.indices_loaded)?;
        counts.set_item("parts_loaded", summary.parts_loaded)?;
        counts.set_item("index_comparisons", summary.index_comparisons)?;
        dict.set_item("counts", counts)?;
        dict.set_item("times_ns", summary.all_times)?;
        let operators = PyList::empty(py);
        for operator in execution.operators {
            let node = PyDict::new(py);
            node.set_item("path", operator.path)?;
            node.set_item("name", operator.name)?;
            let metrics = PyList::empty(py);
            for metric in operator.metrics {
                let value = PyDict::new(py);
                value.set_item("name", metric.name)?;
                value.set_item("partition", metric.partition)?;
                value.set_item("labels", metric.labels)?;
                let (kind, number) = match metric.value {
                    ExecutionMetricValue::Count(value) => ("count", value as u128),
                    ExecutionMetricValue::Gauge(value) => ("gauge", value as u128),
                    ExecutionMetricValue::PeakMemoryBytes(value) => {
                        ("peak_memory_bytes", value as u128)
                    }
                    ExecutionMetricValue::Time(value) => ("time_ns", value.as_nanos()),
                };
                value.set_item("kind", kind)?;
                value.set_item("value", number)?;
                metrics.append(value)?;
            }
            node.set_item("metrics", metrics)?;
            operators.append(node)?;
        }
        dict.set_item("operators", operators)?;
        Ok(dict)
    }

    fn build_joins(
        values: Vec<lance::dataset::MergeInsertJoin>,
        py: Python<'_>,
    ) -> PyResult<Bound<'_, PyList>> {
        let joins = PyList::empty(py);
        for join in values {
            let value = PyDict::new(py);
            value.set_item("path", join.path)?;
            value.set_item("subplan_id", join.subplan_id)?;
            value.set_item("join_type", format!("{:?}", join.join_type))?;
            match join.strategy {
                lance::dataset::MergeInsertJoinStrategy::Hash(mode) => {
                    value.set_item("strategy", "hash")?;
                    value.set_item("partition_mode", format!("{mode:?}"))?;
                }
                lance::dataset::MergeInsertJoinStrategy::SortMerge => {
                    value.set_item("strategy", "sort_merge")?;
                    value.set_item("partition_mode", py.None())?;
                }
            }
            joins.append(value)?;
        }
        Ok(joins)
    }

    fn build_stats<'a>(stats: &MergeStats, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("num_inserted_rows", stats.num_inserted_rows)?;
        dict.set_item("num_updated_rows", stats.num_updated_rows)?;
        dict.set_item("num_deleted_rows", stats.num_deleted_rows)?;
        Ok(dict)
    }
}
