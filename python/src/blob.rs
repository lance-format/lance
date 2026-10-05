// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{error::PythonErrorExt, rt};
use arrow::{
    array::{Array, ArrayRef, make_array},
    pyarrow::{FromPyArrow, ToPyArrow},
};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field};
use bytes::Bytes;
use lance::{
    BlobDescriptor, BlobDescriptorArrayBuilder, BlobRange, DedicatedBlobWriter, PackedBlobWriter,
};
use lance_arrow::iter_binary_array;
use pyo3::{
    Bound, PyErr, PyResult,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyAny, PyAnyMethods, PyDict, PyList, PyListMethods, PyModule, PyTypeMethods},
};
use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
};

fn with_writer<W, R>(
    inner: &Mutex<Option<W>>,
    writer_name: &str,
    operation: impl FnOnce(&W) -> R,
) -> PyResult<R> {
    let guard = inner.lock().map_err(|_| poisoned_writer(writer_name))?;
    let writer = guard.as_ref().ok_or_else(|| finished_writer(writer_name))?;
    Ok(operation(writer))
}

fn writer_mut<'a, W>(inner: &'a mut Mutex<Option<W>>, writer_name: &str) -> PyResult<&'a mut W> {
    inner
        .get_mut()
        .map_err(|_| poisoned_writer(writer_name))?
        .as_mut()
        .ok_or_else(|| finished_writer(writer_name))
}

fn take_writer<W>(inner: &mut Mutex<Option<W>>, writer_name: &str) -> PyResult<W> {
    inner
        .get_mut()
        .map_err(|_| poisoned_writer(writer_name))?
        .take()
        .ok_or_else(|| finished_writer(writer_name))
}

fn finished_writer(writer_name: &str) -> PyErr {
    PyValueError::new_err(format!("{writer_name} is already finished"))
}

fn poisoned_writer(writer_name: &str) -> PyErr {
    PyRuntimeError::new_err(format!("{writer_name} lock is poisoned"))
}

/// Reconstruct the PyArrow equivalent of [`BlobDescriptorArrayBuilder::field`].
///
/// Arrow's array bridge does not carry the enclosing extension field, so this
/// rebuilds the canonical six nullable blob-v2 children and
/// `ARROW:extension:name = lance.blob.v2` metadata.
fn descriptor_field_to_pyarrow<'py>(
    field: &Field,
    py: pyo3::Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = PyModule::import(py, "pyarrow")?;
    let child_fields = PyList::empty(py);
    for (name, type_fn) in [
        ("kind", "uint8"),
        ("data", "large_binary"),
        ("uri", "utf8"),
        ("blob_id", "uint32"),
        ("blob_size", "uint64"),
        ("position", "uint64"),
    ] {
        let data_type = pyarrow.getattr(type_fn)?.call0()?;
        let child = pyarrow.call_method1("field", (name, data_type, true))?;
        child_fields.append(child)?;
    }
    let data_type = pyarrow.call_method1("struct", (child_fields,))?;
    let metadata = PyDict::new(py);
    metadata.set_item("ARROW:extension:name", "lance.blob.v2")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("nullable", field.is_nullable())?;
    kwargs.set_item("metadata", metadata)?;
    pyarrow.call_method("field", (field.name().as_str(), data_type), Some(&kwargs))
}

/// Normalize inputs accepted by [`PyPackedBlobWriter::write_blobs`] into Arrow arrays.
///
/// BinaryArray, LargeBinaryArray, BinaryViewArray, FixedSizeBinaryArray, and
/// ChunkedArray values of any binary type are accepted. Chunk boundaries,
/// nulls, and empty values remain in the arrays; each row is later passed to
/// the core writer as an optional byte slice.
fn extract_blob_payloads(payloads: &Bound<'_, PyAny>) -> PyResult<Vec<ArrayRef>> {
    match ArrayData::from_pyarrow_bound(payloads) {
        Ok(data) => Ok(vec![validated_blob_payload(data, None)?]),
        Err(_) => {
            let pyarrow = PyModule::import(payloads.py(), "pyarrow")?;
            let chunked_array_type = pyarrow.getattr("ChunkedArray")?;
            if !payloads.is_instance(&chunked_array_type)? {
                return Err(PyValueError::new_err(format!(
                    "payloads must be a pyarrow BinaryArray, LargeBinaryArray, or ChunkedArray, got {}",
                    payloads.get_type().name()?
                )));
            }

            let chunked_data_type = DataType::from_pyarrow_bound(&payloads.getattr("type")?)?;
            if !chunked_data_type.is_binary() {
                return Err(PyValueError::new_err(format!(
                    "Packed blob payloads must have a Binary Arrow type, got {chunked_data_type}"
                )));
            }

            let chunks = payloads.getattr("chunks")?;
            let mut arrays = Vec::with_capacity(chunks.len()?);
            for (chunk_index, chunk) in chunks.try_iter()?.enumerate() {
                let data = ArrayData::from_pyarrow_bound(&chunk?)?;
                arrays.push(validated_blob_payload(data, Some(chunk_index))?);
            }
            Ok(arrays)
        }
    }
}

fn validated_blob_payload(data: ArrayData, chunk_index: Option<usize>) -> PyResult<ArrayRef> {
    let context = chunk_index
        .map(|index| format!("Packed blob payload chunk {index}"))
        .unwrap_or_else(|| "Packed blob payload array".to_string());
    if !data.data_type().is_binary() {
        return Err(PyValueError::new_err(format!(
            "{context} must have a Binary Arrow type, got {}",
            data.data_type()
        )));
    }
    if data.is_empty() {
        // PyArrow may export an empty slice without the values preceding its
        // nonzero first offset. Normalize it because an empty array never
        // observes those buffers, and Arrow validation would reject the slice.
        return Ok(make_array(ArrayData::new_empty(data.data_type())));
    }
    data.validate_full().map_err(|error| {
        PyValueError::new_err(format!("{context} contains invalid Arrow data: {error}"))
    })?;
    Ok(make_array(data))
}

async fn write_binary_payloads(writer: &mut PackedBlobWriter, payloads: &ArrayRef) -> PyResult<()> {
    let iter = iter_binary_array(payloads.as_ref()).map_err(|error| {
        PyValueError::new_err(format!(
            "Packed blob payloads must have a Binary Arrow type, got {}: {error}",
            payloads.data_type()
        ))
    })?;
    writer.write_packed_blobs(iter).await.infer_error()
}

#[pyclass(name = "BlobDescriptor", skip_from_py_object)]
#[derive(Clone)]
pub struct PyBlobDescriptor {
    inner: BlobDescriptor,
}

impl From<BlobDescriptor> for PyBlobDescriptor {
    fn from(inner: BlobDescriptor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBlobDescriptor {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "BlobDescriptorArrayBuilder", skip_from_py_object)]
pub struct PyBlobDescriptorArrayBuilder {
    field: arrow_schema::Field,
    inner: Option<BlobDescriptorArrayBuilder>,
}

impl PyBlobDescriptorArrayBuilder {
    fn inner_mut(&mut self) -> PyResult<&mut BlobDescriptorArrayBuilder> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("BlobDescriptorArrayBuilder is already finished"))
    }
}

#[pymethods]
impl PyBlobDescriptorArrayBuilder {
    #[new]
    pub fn new(column: String) -> Self {
        let inner = BlobDescriptorArrayBuilder::new(column);
        Self {
            field: inner.field().clone(),
            inner: Some(inner),
        }
    }

    #[getter]
    pub fn field<'py>(&self, py: pyo3::Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        descriptor_field_to_pyarrow(&self.field, py)
    }

    pub fn extend_packed(
        &mut self,
        blob_id: u32,
        offsets: Vec<u64>,
        sizes: Vec<u64>,
    ) -> PyResult<()> {
        if offsets.len() != sizes.len() {
            return Err(PyValueError::new_err(format!(
                "offsets and sizes must have the same length, got {} and {}",
                offsets.len(),
                sizes.len()
            )));
        }
        let ranges = offsets
            .into_iter()
            .zip(sizes)
            .map(|(offset, size)| BlobRange { offset, size })
            .collect::<Vec<_>>();
        self.inner_mut()?
            .extend_packed(blob_id, ranges)
            .infer_error()
    }

    pub fn append_dedicated(&mut self, blob_id: u32, size: u64) -> PyResult<()> {
        self.inner_mut()?
            .push_dedicated(blob_id, size)
            .infer_error()
    }

    pub fn append(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = value.extract::<pyo3::PyRef<'_, PyBlobDescriptor>>()?;
        self.inner_mut()?.push(value.inner.clone()).infer_error()
    }

    pub fn extend(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let iter = values.try_iter()?;
        for value in iter {
            let value = value?.extract::<pyo3::PyRef<'_, PyBlobDescriptor>>()?;
            self.inner_mut()?.push(value.inner.clone()).infer_error()?;
        }
        Ok(())
    }

    pub fn append_inline(&mut self, data: Vec<u8>) -> PyResult<()> {
        self.inner_mut()?
            .push_inline(Bytes::from(data))
            .infer_error()
    }

    pub fn append_null(&mut self) -> PyResult<()> {
        self.inner_mut()?.push_null().infer_error()
    }

    pub fn finish<'py>(&mut self, py: pyo3::Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.take().ok_or_else(|| {
            PyValueError::new_err("BlobDescriptorArrayBuilder is already finished")
        })?;
        let column = inner.finish().infer_error()?;
        column.array().to_data().to_pyarrow(py)
    }
}

#[pyclass(name = "PackedBlobWriter", skip_from_py_object)]
pub struct PyPackedBlobWriter {
    field: Option<Field>,
    inner: Mutex<Option<PackedBlobWriter>>,
}

impl PyPackedBlobWriter {
    pub(crate) async fn try_new(
        object_store: Arc<lance_io::object_store::ObjectStore>,
        data_file_path: object_store::path::Path,
        blob_id: u32,
    ) -> PyResult<Self> {
        let inner =
            PackedBlobWriter::try_new(object_store.as_ref().clone(), data_file_path, blob_id)
                .await
                .infer_error()?;
        Ok(Self {
            field: None,
            inner: Mutex::new(Some(inner)),
        })
    }

    fn with_inner<R>(&self, operation: impl FnOnce(&PackedBlobWriter) -> R) -> PyResult<R> {
        with_writer(&self.inner, "PackedBlobWriter", operation)
    }

    fn inner_mut(&mut self) -> PyResult<&mut PackedBlobWriter> {
        writer_mut(&mut self.inner, "PackedBlobWriter")
    }

    fn take_inner(&mut self) -> PyResult<PackedBlobWriter> {
        take_writer(&mut self.inner, "PackedBlobWriter")
    }
}

#[pymethods]
impl PyPackedBlobWriter {
    #[getter]
    pub fn blob_id(&self) -> PyResult<u32> {
        self.with_inner(PackedBlobWriter::blob_id)
    }

    #[getter]
    pub fn path(&self) -> PyResult<String> {
        self.with_inner(|writer| writer.path().to_string())
    }

    /// The descriptor field associated with the array returned by
    /// :meth:`finish_array`.
    ///
    /// The field uses the name passed to ``finish_array`` and carries the
    /// ``lance.blob.v2`` extension metadata. It is available only after
    /// ``finish_array`` succeeds; accessing it earlier raises ``ValueError``.
    #[getter]
    pub fn field<'py>(&self, py: pyo3::Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let field = self.field.as_ref().ok_or_else(|| {
            PyValueError::new_err("PackedBlobWriter field is available after finish_array")
        })?;
        descriptor_field_to_pyarrow(field, py)
    }

    /// Append one packed blob.
    ///
    /// Python ``bytes`` are borrowed without copying. Other compatible byte
    /// sequences use owned storage for the duration of the write.
    pub fn write_blob(&mut self, data: Cow<'_, [u8]>) -> PyResult<()> {
        rt().block_on(None, self.inner_mut()?.write_blob(data.as_ref()))?
            .infer_error()
    }

    /// Append a batch of packed blob payloads.
    ///
    /// Parameters
    /// ----------
    /// payloads : pyarrow.BinaryArray, pyarrow.LargeBinaryArray,
    ///     pyarrow.BinaryViewArray, pyarrow.FixedSizeBinaryArray, or
    ///     pyarrow.ChunkedArray
    ///     A binary Arrow array. Every chunk of a chunked array must be binary.
    ///     Each input row produces one descriptor row, in order, across chunks
    ///     and repeated calls. Null rows produce null descriptors; empty but
    ///     non-null byte strings produce valid zero-length blobs.
    ///
    /// Examples
    /// --------
    /// >>> import pyarrow as pa
    /// >>> payloads = pa.array([b"first", None, b""], type=pa.large_binary())
    /// >>> writer.write_blobs(payloads)
    /// >>> descriptors = writer.finish_array("blob")
    /// >>> len(descriptors)
    /// 3
    pub fn write_blobs(&mut self, payloads: &Bound<'_, PyAny>) -> PyResult<()> {
        let payloads = extract_blob_payloads(payloads)?;
        let result = {
            let writer = self.inner_mut()?;
            rt().block_on(None, async {
                for payloads in payloads {
                    write_binary_payloads(writer, &payloads).await?;
                }
                Ok(())
            })
        };
        match result {
            Ok(result) => result,
            Err(error) => {
                // KeyboardInterrupt drops the async batch future. Remove the core
                // writer as well so RAII cleanup runs and a completed prefix cannot
                // be reused as a new batch.
                self.take_inner()?;
                Err(error)
            }
        }
    }

    pub fn finish(&mut self) -> PyResult<Vec<PyBlobDescriptor>> {
        let inner = self.take_inner()?;
        let values = rt().block_on(None, inner.finish())?.infer_error()?;
        Ok(values.into_iter().map(Into::into).collect())
    }

    /// Finish the upload and return its blob descriptors as a PyArrow array.
    ///
    /// The returned ``pyarrow.StructArray`` has one row per payload previously
    /// passed to :meth:`write_blob` or :meth:`write_blobs`. The writer is consumed
    /// by this call. After it succeeds, :attr:`field` returns the matching
    /// extension field with ``field_name`` as its name.
    ///
    /// Parameters
    /// ----------
    /// field_name : str
    ///     Name for the descriptor field exposed by :attr:`field`.
    ///
    /// Returns
    /// -------
    /// pyarrow.StructArray
    ///     Row-aligned blob descriptors, including null rows from bulk input.
    ///
    /// Examples
    /// --------
    /// >>> import pyarrow as pa
    /// >>> writer.write_blobs(pa.array([b"value", None]))
    /// >>> descriptors = writer.finish_array("payload")
    /// >>> descriptors.is_null().to_pylist()
    /// [False, True]
    /// >>> writer.field.name
    /// 'payload'
    pub fn finish_array<'py>(
        &mut self,
        py: pyo3::Python<'py>,
        field_name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.take_inner()?;
        let values = rt().block_on(None, inner.finish())?.infer_error()?;
        let mut builder = BlobDescriptorArrayBuilder::new(field_name);
        builder.extend(values).infer_error()?;
        let column = builder.finish().infer_error()?;
        let (field, array) = column.into_parts();
        let array = array.to_data().to_pyarrow(py)?;
        self.field = Some(field);
        Ok(array)
    }
}

#[pyclass(name = "DedicatedBlobWriter", skip_from_py_object)]
pub struct PyDedicatedBlobWriter {
    inner: Mutex<Option<DedicatedBlobWriter>>,
}

impl PyDedicatedBlobWriter {
    pub(crate) async fn try_new(
        object_store: Arc<lance_io::object_store::ObjectStore>,
        data_file_path: object_store::path::Path,
        blob_id: u32,
    ) -> PyResult<Self> {
        let inner =
            DedicatedBlobWriter::try_new(object_store.as_ref().clone(), data_file_path, blob_id)
                .await
                .infer_error()?;
        Ok(Self {
            inner: Mutex::new(Some(inner)),
        })
    }

    fn with_inner<R>(&self, operation: impl FnOnce(&DedicatedBlobWriter) -> R) -> PyResult<R> {
        with_writer(&self.inner, "DedicatedBlobWriter", operation)
    }

    fn inner_mut(&mut self) -> PyResult<&mut DedicatedBlobWriter> {
        writer_mut(&mut self.inner, "DedicatedBlobWriter")
    }

    fn take_inner(&mut self) -> PyResult<DedicatedBlobWriter> {
        take_writer(&mut self.inner, "DedicatedBlobWriter")
    }
}

#[pymethods]
impl PyDedicatedBlobWriter {
    #[getter]
    pub fn blob_id(&self) -> PyResult<u32> {
        self.with_inner(DedicatedBlobWriter::blob_id)
    }

    #[getter]
    pub fn path(&self) -> PyResult<String> {
        self.with_inner(|writer| writer.path().to_string())
    }

    pub fn write(&mut self, data: Vec<u8>) -> PyResult<()> {
        rt().block_on(None, self.inner_mut()?.write(data))?
            .infer_error()
    }

    pub fn finish(&mut self) -> PyResult<PyBlobDescriptor> {
        let inner = self.take_inner()?;
        let value = rt().block_on(None, inner.finish())?.infer_error()?;
        Ok(value.into())
    }
}
