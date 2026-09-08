// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use arrow::array::{Array, UInt64Array, make_array};
use arrow::compute::CastOptions;
use arrow::compute::kernels::cast::cast_with_options;
use arrow::datatypes::DataType;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_data::ArrayData;
use lance_table::format::RowIdMeta;
use lance_table::rowids::{RowIdSequence, read_row_ids, write_row_ids};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyRange, PyRangeMethods, PyTuple};
use pyo3::{IntoPyObjectExt, intern};

use crate::error::PythonErrorExt;
use crate::fragment::PyRowIdMeta;

/// The number of row ids shown in `RowIdSequence.__repr__` before eliding.
const REPR_PREVIEW_LEN: usize = 10;

/// A sequence of stable row ids belonging to a single fragment.
#[pyclass(name = "RowIdSequence", module = "lance.fragment")]
pub struct PyRowIdSequence(pub RowIdSequence);

#[pymethods]
impl PyRowIdSequence {
    #[new]
    fn new(row_ids: &Bound<'_, PyAny>) -> PyResult<Self> {
        let sequence = match contiguous_range(row_ids)? {
            // A `Range` segment is the most compact encoding, and taking it
            // directly avoids materializing the ids of a large range.
            Some(range) if range.is_empty() => RowIdSequence::new(),
            Some(range) => RowIdSequence::from(range),
            None => RowIdSequence::try_from_iter(extract_row_ids(row_ids)?).infer_error()?,
        };
        Ok(Self(sequence))
    }

    /// Read back the sequence stored inline in fragment row id metadata.
    #[staticmethod]
    fn from_inline_metadata(metadata: PyRef<'_, PyRowIdMeta>) -> PyResult<Self> {
        match &metadata.0 {
            RowIdMeta::Inline(data) => read_row_ids(data).infer_error().map(Self),
            RowIdMeta::External(_) => Err(PyNotImplementedError::new_err(
                "Row ids stored in an external file cannot be read into a RowIdSequence",
            )),
        }
    }

    /// Encode the sequence as row id metadata stored inline in the manifest.
    fn to_inline_metadata(&self) -> PyRowIdMeta {
        PyRowIdMeta(RowIdMeta::Inline(write_row_ids(&self.0).into()))
    }

    fn to_pyarrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let array = UInt64Array::from(self.0.iter().collect::<Vec<u64>>());
        array.into_data().to_pyarrow(py)
    }

    fn __len__(&self) -> usize {
        self.0.len() as usize
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyRowIdSequenceIterator>> {
        let row_ids: Vec<u64> = self.0.iter().collect();
        Py::new(py, PyRowIdSequenceIterator(row_ids.into_iter()))
    }

    fn __repr__(&self) -> String {
        let len = self.0.len();
        let preview = self
            .0
            .iter()
            .take(REPR_PREVIEW_LEN)
            .map(|row_id| row_id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if len > REPR_PREVIEW_LEN as u64 {
            format!("RowIdSequence([{}, ...], len={})", preview, len)
        } else {
            format!("RowIdSequence([{}])", preview)
        }
    }

    fn __richcmp__(
        &self,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let equal = self.0 == other.borrow().0;
        match op {
            CompareOp::Eq => equal.into_py_any(py),
            CompareOp::Ne => (!equal).into_py_any(py),
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let from_inline_metadata = PyModule::import(py, "lance.fragment")?
            .getattr("RowIdSequence")?
            .getattr("from_inline_metadata")?
            .extract()?;
        let metadata = Py::new(py, self.to_inline_metadata())?;
        let state = PyTuple::new(py, [metadata])?.extract()?;
        Ok((from_inline_metadata, state))
    }
}

/// The row ids are materialized up front because `RowIdSequence::iter` borrows
/// the sequence and a `#[pyclass]` cannot hold a borrowing iterator.
///
/// Refilling a bounded buffer from `RowIdSequence::slice` looks like the way to
/// avoid that, but it is quadratic: slicing takes an absolute offset, and for
/// the gapped `RangeWithHoles` and `RangeWithBitmap` encodings the slice skips
/// its prefix one element at a time. Making iteration both lazy and linear
/// needs a forward cursor over the private segments, which belongs in
/// `lance-table` rather than here. Bulk callers should use `to_pyarrow`.
#[pyclass(name = "RowIdSequenceIterator", module = "lance.fragment")]
pub struct PyRowIdSequenceIterator(std::vec::IntoIter<u64>);

#[pymethods]
impl PyRowIdSequenceIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<u64> {
        slf.0.next()
    }
}

/// Recognize a `range` with a step of one, whose row ids need no materialization.
///
/// Returns `None` for anything else, including strided and descending ranges,
/// which are handled by the general iterable path.
fn contiguous_range(ob: &Bound<'_, PyAny>) -> PyResult<Option<Range<u64>>> {
    let Ok(range) = ob.cast::<PyRange>() else {
        return Ok(None);
    };
    // Bounds are read as `isize`. A range reaching past that is well beyond any
    // dataset's row count, so it falls through to the element-wise path, which
    // reads each value as a u64, rather than failing.
    let (Ok(start), Ok(stop), Ok(step)) = (range.start(), range.stop(), range.step()) else {
        return Ok(None);
    };
    if step != 1 {
        return Ok(None);
    }
    if start < 0 {
        return Err(PyValueError::new_err(format!(
            "Row ids must be non-negative, but the range starts at {}",
            start
        )));
    }
    if stop <= start {
        return Ok(Some(0..0));
    }
    Ok(Some(start as u64..stop as u64))
}

fn extract_row_ids(ob: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let py = ob.py();
    if ob.hasattr(intern!(py, "__arrow_c_array__"))? {
        return row_ids_from_arrow(ob);
    }
    // A `pyarrow.ChunkedArray`, such as a `_rowid` column taken from a table.
    let chunks = intern!(py, "chunks");
    if ob.hasattr(chunks)? {
        let mut row_ids = Vec::new();
        for chunk in ob.getattr(chunks)?.try_iter()? {
            row_ids.extend(row_ids_from_arrow(&chunk?)?);
        }
        return Ok(row_ids);
    }

    let iter = ob.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "Row ids must be an iterable of integers or an Arrow array, but got {}",
            ob.get_type().name().map_or_else(
                |_| "an object of unknown type".to_string(),
                |name| name.to_string()
            )
        ))
    })?;
    iter.map(|row_id| row_id?.extract::<u64>()).collect()
}

fn row_ids_from_arrow(array: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let array = make_array(ArrayData::from_pyarrow_bound(array)?);
    if !array.data_type().is_integer() {
        return Err(PyTypeError::new_err(format!(
            "Row ids must be an array of integers, but got an array of type {}",
            array.data_type()
        )));
    }
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Row ids must not be null, but the array has {} null values",
            array.null_count()
        )));
    }

    // `safe: false` so that negative values raise instead of wrapping around
    // into the top of the row id space.
    let cast_options = CastOptions {
        safe: false,
        ..Default::default()
    };
    let array = cast_with_options(&array, &DataType::UInt64, &cast_options)
        .map_err(|err| PyValueError::new_err(format!("Row ids must fit in a uint64: {}", err)))?;
    Ok(array.as_primitive::<UInt64Type>().values().to_vec())
}
