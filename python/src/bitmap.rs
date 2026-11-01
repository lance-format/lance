// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::pyarrow::FromPyArrow;
use arrow_array::{cast::AsArray, make_array};
use arrow_cast::{CastOptions, cast_with_options};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::{IntoPyObjectExt, intern};
use roaring::RoaringBitmap;

/// A lazy, streaming iterator over a `Bitmap`'s values — yields one Python
/// `int` per `__next__` call rather than materializing them all up front.
#[pyclass(name = "BitmapIterator", module = "lance.bitmap")]
pub struct PyBitmapIter(roaring::bitmap::IntoIter);

#[pymethods]
impl PyBitmapIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<u32> {
        slf.0.next()
    }
}

/// A set of non-negative integers backed by a `RoaringBitmap`.
///
/// Cheap to clone (an `Arc` bump) and to pass into Lance APIs that accept a
/// bitmap, since no per-value Python object is created. Mutating methods
/// (`add`, `discard`, `update`, ...) copy-on-write: cloning a `Bitmap` and
/// mutating one copy never affects the other.
///
/// Implements the `collections.abc.MutableSet` interface — set algebra,
/// subset/superset comparisons, and in-place updates — and is registered as
/// one in `lance.bitmap`, so `isinstance(b, collections.abc.Set)` holds.
/// `isinstance(b, set)` does not: it is a distinct type, not a `set` subclass.
///
/// Anywhere a `Bitmap` accepts another set, it also accepts any iterable of
/// ints (a `set`, `list`, `range`, generator, ...), which is why `==` against
/// a plain list is true when the values match. Order and repeats in that
/// iterable are not significant.
#[pyclass(name = "Bitmap", module = "lance.bitmap", from_py_object)]
#[derive(Clone, Debug, Default)]
pub struct PyBitmap(pub Arc<RoaringBitmap>);

impl PyBitmap {
    pub fn new(bitmap: RoaringBitmap) -> Self {
        Self(Arc::new(bitmap))
    }
}

/// Whether a Python value is an integer.
///
/// Tested by `__index__` rather than `isinstance(value, int)` so that numpy
/// and other foreign integer scalars — which pyo3 also accepts — count.
pub(crate) fn is_int(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.hasattr(intern!(value.py(), "__index__"))
}

/// Convert one Python value to a bitmap value.
///
/// `u32::extract` alone raises a bare `OverflowError` that names neither the
/// limit nor the offending value, so an integer outside the range gets a
/// message that does. A non-integer keeps pyo3's own `TypeError`.
pub(crate) fn value_from_py(value: &Bound<'_, PyAny>) -> PyResult<u32> {
    if !is_int(value)? {
        return value.extract::<u32>();
    }
    value.extract::<u32>().map_err(|_| {
        PyValueError::new_err(format!(
            "Bitmap values must fit in an unsigned 32-bit integer, got {value}"
        ))
    })
}

/// Collect any Python iterable of ints into a bitmap.
pub(crate) fn bitmap_from_iterable(values: &Bound<'_, PyAny>) -> PyResult<RoaringBitmap> {
    values
        .try_iter()?
        .map(|item| value_from_py(&item?))
        .collect()
}

/// Read an integer pyarrow array's values into a `RoaringBitmap`, without
/// going through per-value Python objects — the cast reads the array's native
/// buffer and writes another one.
fn bitmap_from_pyarrow(ob: &Bound<'_, PyAny>) -> PyResult<RoaringBitmap> {
    let mut data = ArrayData::from_pyarrow_bound(ob)?;
    if data.null_count() > 0 {
        return Err(PyValueError::new_err(
            "Bitmap cannot be constructed from an array containing nulls",
        ));
    }
    // Buffers that arrive over the Arrow C data interface carry whatever
    // alignment the producer gave them (e.g. a sliced `pa.py_buffer`), while
    // `make_array` requires the type's native alignment and panics otherwise.
    // A panic can't cross the FFI boundary as a Python exception, so copy any
    // under-aligned buffer into an aligned allocation first.
    data.align_buffers();
    let array = make_array(data);
    if !array.data_type().is_integer() {
        // Reject before casting, not after: a float array would cast
        // "successfully" by truncating each value.
        return Err(PyValueError::new_err(format!(
            "Bitmap can only be constructed from an integer pyarrow array, got {}",
            array.data_type()
        )));
    }
    // `safe: false` makes a negative or too-large value an error naming that
    // value, rather than a silent null.
    let values = cast_with_options(
        &array,
        &DataType::UInt32,
        &CastOptions {
            safe: false,
            ..Default::default()
        },
    )
    .map_err(|e| {
        PyValueError::new_err(format!(
            "Bitmap values must fit in an unsigned 32-bit integer: {e}"
        ))
    })?;
    Ok(values
        .as_primitive::<arrow::datatypes::UInt32Type>()
        .values()
        .iter()
        .copied()
        .collect())
}

/// Coerce the right-hand side of a set operation.
///
/// A `Bitmap` lends its `Arc` rather than being copied. `None` means "not a
/// set of ints", which the operators surface as `NotImplemented` so Python can
/// try the reflected operation before raising.
fn operand(other: &Bound<'_, PyAny>) -> Option<Arc<RoaringBitmap>> {
    if let Ok(bitmap) = other.extract::<PyBitmap>() {
        return Some(bitmap.0);
    }
    bitmap_from_iterable(other).ok().map(Arc::new)
}

/// Coerce the argument of a named set method (`union`, `issubset`, ...), which
/// raises on a bad argument rather than deferring to Python, as `set` does.
fn required_operand(other: &Bound<'_, PyAny>) -> PyResult<Arc<RoaringBitmap>> {
    if let Ok(bitmap) = other.extract::<PyBitmap>() {
        return Ok(bitmap.0);
    }
    bitmap_from_iterable(other).map(Arc::new)
}

/// Run a set operator, yielding `NotImplemented` for an operand that isn't a
/// set of ints.
fn set_op(
    py: Python<'_>,
    other: &Bound<'_, PyAny>,
    op: impl FnOnce(&RoaringBitmap) -> RoaringBitmap,
) -> PyResult<Py<PyAny>> {
    match operand(other) {
        Some(other) => Ok(Py::new(py, PyBitmap::new(op(&other)))?.into_any()),
        None => Ok(py.NotImplemented()),
    }
}

#[pymethods]
impl PyBitmap {
    /// Construct a Bitmap from an iterable of non-negative ints (list, set,
    /// range, generator, ...) or an integer pyarrow Array/ChunkedArray.
    #[new]
    #[pyo3(signature = (values=None))]
    fn new_py(values: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(values) = values else {
            return Ok(Self::default());
        };
        if let Ok(existing) = values.extract::<Self>() {
            return Ok(existing);
        }
        // A pyarrow `Array` supports the `__arrow_c_array__` Arrow C Data
        // Interface export directly; a `ChunkedArray` doesn't (it only
        // supports the streaming `__arrow_c_stream__` form), so detect it by
        // its `combine_chunks()` method and flatten it to a single `Array`
        // first.
        if values.hasattr("combine_chunks")? {
            let combined = values.call_method0("combine_chunks")?;
            return Ok(Self::new(bitmap_from_pyarrow(&combined)?));
        }
        if values.hasattr("__arrow_c_array__")? {
            return Ok(Self::new(bitmap_from_pyarrow(values)?));
        }
        Ok(Self::new(bitmap_from_iterable(values)?))
    }

    fn __len__(&self) -> usize {
        self.0.len() as usize
    }

    fn __contains__(&self, value: &Bound<'_, PyAny>) -> bool {
        // A value that can't be a bitmap element simply isn't in it, matching
        // `set` — `"a" in {1}` is False, not a TypeError.
        value.extract::<u32>().is_ok_and(|v| self.0.contains(v))
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyBitmapIter>> {
        // Cloning the bitmap here is a native Rust-side copy of its compressed
        // containers, not a per-value Python allocation — the point is that
        // `PyBitmapIter` then streams values lazily instead of eagerly
        // building a Python list of boxed ints up front.
        Py::new(py, PyBitmapIter((*self.0).clone().into_iter()))
    }

    pub(crate) fn __repr__(&self) -> String {
        const MAX_VALUES_SHOWN: usize = 20;
        let len = self.0.len();
        let values: Vec<String> = self
            .0
            .iter()
            .take(MAX_VALUES_SHOWN)
            .map(|v| v.to_string())
            .collect();
        if (len as usize) > MAX_VALUES_SHOWN {
            format!("Bitmap({{{}, ...}}, len={})", values.join(", "), len)
        } else {
            format!("Bitmap({{{}}})", values.join(", "))
        }
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Some(other) = operand(other) else {
            // A value that isn't a set of ints (e.g. `5`) is simply unequal,
            // matching normal Python `==` semantics — it shouldn't raise just
            // because the type differs. Ordering against it has no meaning, so
            // defer to Python, which raises if nothing else handles it.
            return match op {
                CompareOp::Eq => false.into_py_any(py),
                CompareOp::Ne => true.into_py_any(py),
                _ => Ok(py.NotImplemented()),
            };
        };
        match op {
            CompareOp::Eq => (*self.0 == *other).into_py_any(py),
            CompareOp::Ne => (*self.0 != *other).into_py_any(py),
            CompareOp::Le => self.0.is_subset(&other).into_py_any(py),
            CompareOp::Lt => {
                (self.0.len() < other.len() && self.0.is_subset(&other)).into_py_any(py)
            }
            CompareOp::Ge => self.0.is_superset(&other).into_py_any(py),
            CompareOp::Gt => {
                (self.0.len() > other.len() && self.0.is_superset(&other)).into_py_any(py)
            }
        }
    }

    fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        set_op(py, other, |other| &*self.0 & other)
    }

    fn __rand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        set_op(py, other, |other| other & &*self.0)
    }

    fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        set_op(py, other, |other| &*self.0 | other)
    }

    fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        set_op(py, other, |other| other | &*self.0)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        set_op(py, other, |other| &*self.0 - other)
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        set_op(py, other, |other| other - &*self.0)
    }

    fn __xor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        set_op(py, other, |other| &*self.0 ^ other)
    }

    fn __rxor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        set_op(py, other, |other| other ^ &*self.0)
    }

    /// The values present in both this bitmap and `other`.
    fn intersection(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(&*self.0 & &*required_operand(other)?))
    }

    /// The values present in either this bitmap or `other`.
    fn union(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(&*self.0 | &*required_operand(other)?))
    }

    /// The values present in this bitmap but not in `other`.
    fn difference(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(&*self.0 - &*required_operand(other)?))
    }

    /// The values present in exactly one of this bitmap and `other`.
    fn symmetric_difference(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::new(&*self.0 ^ &*required_operand(other)?))
    }

    /// Whether every value in this bitmap is also in `other`.
    fn issubset(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.0.is_subset(&*required_operand(other)?))
    }

    /// Whether every value in `other` is also in this bitmap.
    fn issuperset(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.0.is_superset(&*required_operand(other)?))
    }

    /// Whether this bitmap and `other` share no values.
    fn isdisjoint(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.0.is_disjoint(&*required_operand(other)?))
    }

    /// A `Bitmap` with the same values. Mutating either leaves the other
    /// unchanged, so this shares the underlying buffer until one is written to.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Add a value, cloning the underlying bitmap first if it is shared with
    /// another `Bitmap`.
    fn add(&mut self, value: u32) {
        Arc::make_mut(&mut self.0).insert(value);
    }

    /// Remove a value if present, cloning the underlying bitmap first if it
    /// is shared with another `Bitmap`.
    fn discard(&mut self, value: u32) {
        Arc::make_mut(&mut self.0).remove(value);
    }

    /// Remove a value, raising `KeyError` if it isn't present.
    fn remove(&mut self, value: u32) -> PyResult<()> {
        if Arc::make_mut(&mut self.0).remove(value) {
            Ok(())
        } else {
            Err(PyKeyError::new_err(value))
        }
    }

    /// Remove and return the smallest value, raising `KeyError` if empty.
    fn pop(&mut self) -> PyResult<u32> {
        let bitmap = Arc::make_mut(&mut self.0);
        let value = bitmap
            .min()
            .ok_or_else(|| PyKeyError::new_err("pop from an empty Bitmap"))?;
        bitmap.remove(value);
        Ok(value)
    }

    /// Remove every value.
    fn clear(&mut self) {
        Arc::make_mut(&mut self.0).clear();
    }

    /// Add all values from an iterable, cloning the underlying bitmap first
    /// if it is shared with another `Bitmap`.
    fn update(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = required_operand(values)?;
        *Arc::make_mut(&mut self.0) |= &*other;
        Ok(())
    }

    fn __ior__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.update(other)
    }

    fn __iand__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = required_operand(other)?;
        *Arc::make_mut(&mut self.0) &= &*other;
        Ok(())
    }

    fn __isub__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = required_operand(other)?;
        *Arc::make_mut(&mut self.0) -= &*other;
        Ok(())
    }

    fn __ixor__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let other = required_operand(other)?;
        *Arc::make_mut(&mut self.0) ^= &*other;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (Py<PyAny>,))> {
        let mut buf = Vec::new();
        self.0
            .serialize_into(&mut buf)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize Bitmap: {e}")))?;
        let ctor = py
            .import("lance.bitmap")?
            .getattr("Bitmap")?
            .getattr("_from_bytes")?;
        let bytes = PyBytes::new(py, &buf);
        Ok((ctor.unbind(), (bytes.into(),)))
    }

    #[staticmethod]
    fn _from_bytes(data: &[u8]) -> PyResult<Self> {
        let bitmap = RoaringBitmap::deserialize_from(data)
            .map_err(|e| PyValueError::new_err(format!("Failed to deserialize Bitmap: {e}")))?;
        Ok(Self::new(bitmap))
    }
}
