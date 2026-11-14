// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use pyo3::{
    Bound, Py, PyRef, PyResult, Python,
    exceptions::PyValueError,
    pyclass, pyfunction, pymethods,
    types::{PyByteArray, PyByteArrayMethods, PyBytes},
};

use lance::dataset::BlobFile as InnerBlobFile;

use crate::{error::PythonErrorExt, rt};

#[pyclass]
pub struct LanceBlobFile {
    pub(super) inner: Arc<InnerBlobFile>,
}

type PyReadBlobFile = (usize, Option<Py<PyBytes>>);

fn read_blob_files_to_python(
    py: Python<'_>,
    blobs: Vec<lance::blob::ReadBlobFile>,
) -> Vec<PyReadBlobFile> {
    blobs
        .into_iter()
        .map(|blob| {
            (
                blob.request_index,
                blob.data.map(|data| PyBytes::new(py, &data).unbind()),
            )
        })
        .collect()
}

fn configure_read_blob_files_builder(
    mut builder: lance::blob::ReadBlobFilesBuilder,
    io_buffer_size: Option<u64>,
    io_parallelism: Option<usize>,
    preserve_order: Option<bool>,
) -> lance::blob::ReadBlobFilesBuilder {
    if let Some(bytes) = io_buffer_size {
        builder = builder.with_io_buffer_size_bytes(bytes);
    }
    if let Some(parallelism) = io_parallelism {
        builder = builder.with_io_parallelism(parallelism);
    }
    if let Some(preserve) = preserve_order {
        builder = builder.preserve_order(preserve);
    }
    builder
}

#[pyfunction(signature=(
    blob_files,
    *,
    io_buffer_size=None,
    io_parallelism=None,
    preserve_order=None
))]
pub(crate) fn read_blob_files(
    py: Python<'_>,
    blob_files: Vec<Option<PyRef<'_, LanceBlobFile>>>,
    io_buffer_size: Option<u64>,
    io_parallelism: Option<usize>,
    preserve_order: Option<bool>,
) -> PyResult<Vec<PyReadBlobFile>> {
    let builder = configure_read_blob_files_builder(
        lance::blob::read_blob_files(
            blob_files
                .iter()
                .map(|blob_file| blob_file.as_ref().map(|blob_file| blob_file.inner.as_ref())),
        ),
        io_buffer_size,
        io_parallelism,
        preserve_order,
    );
    let blobs = rt().block_on(Some(py), builder.execute())?.infer_error()?;
    Ok(read_blob_files_to_python(py, blobs))
}

#[pyfunction(signature=(
    requests,
    *,
    io_buffer_size=None,
    io_parallelism=None,
    preserve_order=None
))]
pub(crate) fn read_blob_file_ranges(
    py: Python<'_>,
    requests: Vec<(Option<PyRef<'_, LanceBlobFile>>, u64, u64)>,
    io_buffer_size: Option<u64>,
    io_parallelism: Option<usize>,
    preserve_order: Option<bool>,
) -> PyResult<Vec<PyReadBlobFile>> {
    let builder = configure_read_blob_files_builder(
        lance::blob::read_blob_file_ranges(requests.iter().map(|(blob_file, offset, length)| {
            (
                blob_file.as_ref().map(|blob_file| blob_file.inner.as_ref()),
                lance::blob::BlobReadRange::new(*offset, *length),
            )
        })),
        io_buffer_size,
        io_parallelism,
        preserve_order,
    );
    let blobs = rt().block_on(Some(py), builder.execute())?.infer_error()?;
    Ok(read_blob_files_to_python(py, blobs))
}

#[pymethods]
impl LanceBlobFile {
    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        rt().block_on(Some(py), inner.close())?.infer_error()
    }

    pub fn is_closed(&self, py: Python<'_>) -> PyResult<bool> {
        let inner = self.inner.clone();
        rt().block_on(Some(py), inner.is_closed())
    }

    pub fn seek(&self, py: Python<'_>, position: u64) -> PyResult<()> {
        let inner = self.inner.clone();
        rt().block_on(Some(py), inner.seek(position))?.infer_error()
    }

    pub fn tell(&self, py: Python<'_>) -> PyResult<u64> {
        let inner = self.inner.clone();
        rt().block_on(Some(py), inner.tell())?.infer_error()
    }

    pub fn size(&self) -> u64 {
        self.inner.size()
    }

    pub fn readall<'a>(&'a self, py: Python<'a>) -> PyResult<Bound<'a, PyBytes>> {
        let inner = self.inner.clone();
        let data = rt().block_on(Some(py), inner.read())?.infer_error()?;
        Ok(PyBytes::new(py, &data))
    }

    /// Read a blob-local byte range without changing the current cursor.
    pub fn read_range<'a>(
        &'a self,
        py: Python<'a>,
        offset: u64,
        length: usize,
    ) -> PyResult<Bound<'a, PyBytes>> {
        let end = offset
            .checked_add(length as u64)
            .ok_or_else(|| PyValueError::new_err("offset + length overflowed"))?;
        let inner = self.inner.clone();
        let data = rt()
            .block_on(Some(py), inner.read_range(offset..end))?
            .infer_error()?;
        Ok(PyBytes::new(py, &data))
    }

    /// Read multiple blob-local `(offset, length)` ranges without changing the current cursor.
    pub fn read_ranges<'py>(
        &self,
        py: Python<'py>,
        ranges: Vec<(u64, u64)>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let ranges = ranges
            .into_iter()
            .enumerate()
            .map(|(i, (offset, length))| {
                let end = offset.checked_add(length).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Blob range request {i} offset + length overflowed u64: \
                         offset={offset}, length={length}"
                    ))
                })?;
                Ok(offset..end)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let inner = self.inner.clone();
        let data = rt()
            .block_on(Some(py), inner.read_ranges(&ranges))?
            .infer_error()?;
        Ok(data.iter().map(|bytes| PyBytes::new(py, bytes)).collect())
    }

    pub fn read_into(&self, dst: Bound<'_, PyByteArray>) -> PyResult<usize> {
        let inner = self.inner.clone();

        let data = rt()
            .block_on(Some(dst.py()), inner.read_up_to(dst.len()))?
            .infer_error()?;

        // Need to re-check length because the buffer could have been resized
        // by Python code while we were reading.
        if dst.len() < data.len() {
            Err(PyValueError::new_err("Buffer too small"))
        } else {
            // Safety: We've checked the buffer size above.  We've held the
            // GIL since then and so no other Python code could have modified
            // the buffer.
            unsafe {
                dst.as_bytes_mut()[0..data.len()].copy_from_slice(&data);
            }
            Ok(data.len())
        }
    }

    /// Sequential read-ahead size in bytes. `0` disables read-ahead.
    pub fn set_buffer_size(&self, py: Python<'_>, buffer_size: usize) -> PyResult<()> {
        let inner = self.inner.clone();
        rt().block_on(Some(py), inner.set_buffer_size(buffer_size))?
            .infer_error()
    }

    #[pyo3(name = "_range_submission_count")]
    pub fn range_submission_count(&self) -> usize {
        self.inner.range_submission_count()
    }
}

impl From<InnerBlobFile> for LanceBlobFile {
    fn from(inner: InnerBlobFile) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}
