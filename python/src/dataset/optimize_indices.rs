// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index optimization as plan, execute, merge and commit, for a driver that
//! runs the stages on different machines; `lance::index::optimize` documents
//! the model. Tasks, merges and results pickle through their JSON form.

use lance::index::optimize::{
    IndexSegmentResult, OptimizeIndexMerge, OptimizeIndexTask, OptimizeIndicesPlan,
    commit_optimize_indices, plan_optimize_indices,
};
use lance_index::optimize::OptimizeOptions;
use serde::{Serialize, de::DeserializeOwned};

use super::optimize::unwrap_dataset;
use super::*;

/// The `OptimizeOptions` of the keyword options of `optimize_indices`, read
/// as it always has: a key it does not know is ignored.
pub(crate) fn optimize_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<OptimizeOptions> {
    let mut options = OptimizeOptions::default();
    let Some(kwargs) = kwargs else {
        return Ok(options);
    };
    if let Some(num_indices_to_merge) = kwargs.get_item("num_indices_to_merge")? {
        options.num_indices_to_merge = num_indices_to_merge.extract()?;
    }
    if let Some(retrain) = kwargs.get_item("retrain")? {
        options.retrain = retrain.extract()?;
    }
    if let Some(index_names) = kwargs.get_item("index_names")? {
        options.index_names = Some(
            index_names
                .extract::<Vec<String>>()
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        );
    }
    if let Some(max_rows_per_task) = kwargs.get_item("max_rows_per_task")? {
        options.max_rows_per_task = max_rows_per_task.extract()?;
    }
    if let Some(max_rows_per_segment) = kwargs.get_item("max_rows_per_segment")? {
        options.max_rows_per_segment = max_rows_per_segment.extract()?;
    }
    Ok(options)
}

fn to_json<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|err| PyValueError::new_err(err.to_string()))
}

fn from_json<T: DeserializeOwned>(json: &str) -> PyResult<T> {
    serde_json::from_str(json).map_err(|err| PyValueError::new_err(err.to_string()))
}

/// A plan to optimize the indices of a dataset: the tasks to execute, then
/// the merges to execute over the task results.
#[pyclass(name = "OptimizeIndicesPlan", module = "lance.optimize")]
pub struct PyOptimizeIndicesPlan(OptimizeIndicesPlan);

#[pymethods]
impl PyOptimizeIndicesPlan {
    /// List[OptimizeIndexTask] : The build units, to run first.
    #[getter]
    fn tasks(&self) -> Vec<PyOptimizeIndexTask> {
        self.0
            .tasks
            .iter()
            .cloned()
            .map(PyOptimizeIndexTask)
            .collect()
    }

    /// List[OptimizeIndexMerge] : The merge units, to run once every task is
    /// done.
    #[getter]
    fn merges(&self) -> Vec<PyOptimizeIndexMerge> {
        self.0
            .merges
            .iter()
            .cloned()
            .map(PyOptimizeIndexMerge)
            .collect()
    }

    /// The JSON representation of the plan, not guaranteed to be stable
    /// across versions.
    fn json(&self) -> PyResult<String> {
        to_json(&self.0)
    }

    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        from_json(&json).map(Self)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        Ok((
            py.get_type::<Self>().getattr("from_json")?.unbind(),
            (self.json()?,),
        ))
    }
}

/// Builds one new, uncommitted segment of one index over a group of
/// fragments; pickle it to a worker, which executes it on its own dataset.
#[pyclass(name = "OptimizeIndexTask", module = "lance.optimize")]
pub struct PyOptimizeIndexTask(OptimizeIndexTask);

#[pymethods]
impl PyOptimizeIndexTask {
    /// str : The name of the index the task builds a segment of.
    #[getter]
    fn index_name(&self) -> String {
        self.0.reference.name.clone()
    }

    /// List[int] : The ids of the fragments to index.
    #[getter]
    fn fragments(&self) -> Vec<u32> {
        self.0.fragments.clone()
    }

    /// Build the segment over ``dataset``, at any version: the task reads the
    /// version it was planned from. Returns the segment to pass to the merges
    /// of the plan, or ``None`` when there was nothing to build.
    fn execute(
        &self,
        py: Python<'_>,
        dataset: Bound<PyAny>,
    ) -> PyResult<Option<PyIndexSegmentResult>> {
        let dataset = unwrap_dataset(dataset)?.borrow().clone();
        let result = rt()
            .block_on(Some(py), async move {
                self.0.execute(dataset.ds.as_ref()).await
            })?
            .infer_error()?;
        Ok(result.map(PyIndexSegmentResult))
    }

    /// The JSON representation of the task, not guaranteed to be stable
    /// across versions.
    fn json(&self) -> PyResult<String> {
        to_json(&self.0)
    }

    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        from_json(&json).map(Self)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        Ok((
            py.get_type::<Self>().getattr("from_json")?.unbind(),
            (self.json()?,),
        ))
    }
}

/// Combines the task results of its group with the committed segments the
/// plan assigned to it into one segment to commit; pickle it to a worker once
/// every task of the plan has run.
#[pyclass(name = "OptimizeIndexMerge", module = "lance.optimize")]
pub struct PyOptimizeIndexMerge(OptimizeIndexMerge);

#[pymethods]
impl PyOptimizeIndexMerge {
    /// int : The group whose task results the merge combines.
    #[getter]
    fn group(&self) -> u32 {
        self.0.group
    }

    /// Merge over ``dataset``, at any version, the ``task_results`` of the
    /// plan: those of other groups and ``None`` entries are ignored, so every
    /// merge may be given all of them. Returns the segment to commit, or
    /// ``None`` when the group has nothing to commit.
    fn execute(
        &self,
        py: Python<'_>,
        dataset: Bound<PyAny>,
        task_results: Vec<Option<PyIndexSegmentResult>>,
    ) -> PyResult<Option<PyIndexSegmentResult>> {
        let dataset = unwrap_dataset(dataset)?.borrow().clone();
        let task_results: Vec<IndexSegmentResult> = task_results
            .into_iter()
            .flatten()
            .map(|result| result.0)
            .collect();
        let result = rt()
            .block_on(Some(py), async move {
                self.0.execute(dataset.ds.as_ref(), &task_results).await
            })?
            .infer_error()?;
        Ok(result.map(PyIndexSegmentResult))
    }

    /// The JSON representation of the merge, not guaranteed to be stable
    /// across versions.
    fn json(&self) -> PyResult<String> {
        to_json(&self.0)
    }

    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        from_json(&json).map(Self)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        Ok((
            py.get_type::<Self>().getattr("from_json")?.unbind(),
            (self.json()?,),
        ))
    }
}

/// A segment built by a task or a merge, not yet in the manifest; pickle it
/// back to the driver.
#[pyclass(name = "IndexSegmentResult", module = "lance.optimize", from_py_object)]
#[derive(Clone)]
pub struct PyIndexSegmentResult(IndexSegmentResult);

#[pymethods]
impl PyIndexSegmentResult {
    /// int : The merge group the segment belongs to.
    #[getter]
    fn group(&self) -> u32 {
        self.0.group
    }

    /// The JSON representation of the result, not guaranteed to be stable
    /// across versions.
    fn json(&self) -> PyResult<String> {
        to_json(&self.0)
    }

    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        from_json(&json).map(Self)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        Ok((
            py.get_type::<Self>().getattr("from_json")?.unbind(),
            (self.json()?,),
        ))
    }
}

/// Index optimization as plan, execute, merge and commit, the stages
/// :py:meth:`lance.dataset.DatasetOptimizer.optimize_indices` runs in one
/// process: :py:meth:`plan`, execute every task of the plan, then every merge
/// over the task results, and :py:meth:`commit` the merge results.
#[pyclass(name = "OptimizeIndices", module = "lance.optimize")]
pub struct PyOptimizeIndices;

#[pymethods]
impl PyOptimizeIndices {
    /// Plan the optimization of the indices of ``dataset`` with the keyword
    /// ``options`` of :py:meth:`lance.dataset.DatasetOptimizer.optimize_indices`;
    /// without ``max_rows_per_task`` or ``max_rows_per_segment`` every index
    /// is one task.
    #[staticmethod]
    #[pyo3(signature = (dataset, options = None))]
    fn plan(
        py: Python<'_>,
        dataset: Bound<PyAny>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyOptimizeIndicesPlan> {
        let dataset = unwrap_dataset(dataset)?.borrow().clone();
        let options = optimize_options(options)?;
        let plan = rt()
            .block_on(Some(py), async move {
                plan_optimize_indices(dataset.ds.as_ref(), &options).await
            })?
            .infer_error()?;
        Ok(PyOptimizeIndicesPlan(plan))
    }

    /// Commit the merge ``results`` in one transaction, ``None`` entries
    /// ignored; whatever they do not cover stays unindexed for the next
    /// optimize. ``dataset`` is updated to the new version.
    #[staticmethod]
    fn commit(dataset: Bound<PyAny>, results: Vec<Option<PyIndexSegmentResult>>) -> PyResult<()> {
        let dataset_ref = unwrap_dataset(dataset)?;
        // Taken before the commit, so a handle another thread is using is
        // refused with nothing committed, not after the commit went through.
        let mut dataset_mut = dataset_ref.try_borrow_mut()?;
        let mut new_ds = dataset_mut.ds.as_ref().clone();
        let results: Vec<IndexSegmentResult> = results
            .into_iter()
            .flatten()
            .map(|result| result.0)
            .collect();
        rt().block_on(None, async {
            commit_optimize_indices(&mut new_ds, results, &OptimizeOptions::default()).await
        })?
        .infer_error()?;
        dataset_mut.ds = Arc::new(new_ds);
        Ok(())
    }
}
