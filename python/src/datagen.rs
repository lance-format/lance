use arrow::pyarrow::PyArrowType;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::Schema;
use lance_datagen::{BatchCount, ByteCount, RowCount};
use pyo3::{
    Bound, PyResult, Python, pyfunction,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

const DEFAULT_BATCH_SIZE_BYTES: u64 = 32 * 1024;
const DEFAULT_BATCH_COUNT: u32 = 4;

#[pyfunction]
pub fn is_datagen_supported() -> bool {
    true
}

/// Generate `batch_count` batches of random data for `schema`.
///
/// Batch size is set either by `rows_in_batch` (exact rows per batch) or
/// `bytes_in_batch` (approximate bytes per batch); the two are mutually
/// exclusive. When neither is given the byte-based default is used.
#[pyfunction]
#[pyo3(signature=(schema, batch_count=None, bytes_in_batch=None, rows_in_batch=None))]
pub fn rand_batches(
    schema: PyArrowType<Schema>,
    batch_count: Option<u32>,
    bytes_in_batch: Option<u64>,
    rows_in_batch: Option<u64>,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    if rows_in_batch.is_some() && bytes_in_batch.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rows_in_batch and bytes_in_batch are mutually exclusive",
        ));
    }
    let builder = lance_datagen::rand(&schema.0);
    let batch_count = BatchCount::from(batch_count.unwrap_or(DEFAULT_BATCH_COUNT));
    let reader: Box<dyn RecordBatchReader> = match rows_in_batch {
        Some(rows) => Box::new(builder.into_reader_rows(RowCount::from(rows), batch_count)),
        None => Box::new(
            builder
                .into_reader_bytes(
                    ByteCount::from(bytes_in_batch.unwrap_or(DEFAULT_BATCH_SIZE_BYTES)),
                    batch_count,
                    lance_datagen::RoundingBehavior::RoundUp,
                )
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Failed to generate batches: {}",
                        e
                    ))
                })?,
        ),
    };
    reader
        .map(|item| {
            item.map(PyArrowType::from).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Failed to generate batch: {}", e))
            })
        })
        .collect::<PyResult<Vec<PyArrowType<RecordBatch>>>>()
}

pub fn register_datagen(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let datagen = PyModule::new(py, "datagen")?;
    datagen.add_wrapped(wrap_pyfunction!(is_datagen_supported))?;
    datagen.add_wrapped(wrap_pyfunction!(rand_batches))?;
    m.add_submodule(&datagen)?;
    Ok(())
}
