// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::FromPyArrow;
use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use datafusion::catalog::{TableProvider, streaming::StreamingTable};
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::PartitionStream;
use futures::{TryStreamExt, stream};
use lance_datafusion::utils::reader_to_stream;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A Python factory is only accepted after the Python adapter has established
/// that it can return a fresh reader over the same source on every call.
pub(super) fn replayable_provider(
    schema: SchemaRef,
    factory: &Bound<'_, PyAny>,
) -> PyResult<Arc<dyn TableProvider>> {
    if !factory.is_callable() {
        return Err(PyTypeError::new_err(
            "Replayable merge source factory must be callable",
        ));
    }
    let partition = Arc::new(ReplayablePartition {
        schema: schema.clone(),
        factory: Arc::new(factory.clone().unbind()),
    });
    let provider = StreamingTable::try_new(schema, vec![partition])
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(Arc::new(provider))
}

struct ReplayablePartition {
    schema: SchemaRef,
    factory: Arc<Py<PyAny>>,
}

impl std::fmt::Debug for ReplayablePartition {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReplayablePartition")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl PartitionStream for ReplayablePartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _context: Arc<TaskContext>) -> SendableRecordBatchStream {
        let factory = self.factory.clone();
        let schema = self.schema.clone();
        let open_reader = async move {
            // The factory and Arrow C stream import can run Python or block on a
            // Lance scan. Keep both off the async worker, and release the GIL
            // before reader_to_stream drives the reader in background threads.
            let reader = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let source = factory.bind(py).call0().map_err(|error| {
                        DataFusionError::External(Box::new(error))
                            .context("Failed to open replayable merge source")
                    })?;
                    let reader =
                        ArrowArrayStreamReader::from_pyarrow_bound(&source).map_err(|error| {
                            DataFusionError::External(Box::new(error))
                                .context("Failed to import replayable merge source reader")
                        })?;
                    let actual_schema = reader.schema();
                    if actual_schema != schema {
                        return Err(DataFusionError::Execution(format!(
                            "Replayable merge source schema changed: expected {schema:?}, \
                             got {actual_schema:?} (including metadata)"
                        )));
                    }
                    Ok(reader)
                })
            })
            .await
            .map_err(|error| {
                DataFusionError::External(Box::new(error))
                    .context("Replayable merge source reader task failed")
            })??;
            Ok::<_, DataFusionError>(reader_to_stream(Box::new(reader)))
        };
        Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream::once(open_reader).try_flatten(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::ffi_stream::FFI_ArrowArrayStream;
    use arrow_array::{RecordBatch, record_batch};
    use arrow_schema::ArrowError;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::{PyCFunction, PyCapsule};

    use super::*;

    // Export C streams directly so the Rust tests do not require PyArrow.
    #[pyclass]
    struct ReaderSource(std::sync::Mutex<Option<Box<dyn RecordBatchReader + Send>>>);

    #[pymethods]
    impl ReaderSource {
        fn __arrow_c_stream__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
            let reader = self.0.lock().unwrap().take().unwrap();
            PyCapsule::new(
                py,
                FFI_ArrowArrayStream::new(reader),
                Some(c"arrow_array_stream".to_owned()),
            )
        }
    }

    struct TrackedReader {
        batch: Option<RecordBatch>,
        schema: SchemaRef,
        dropped: Arc<AtomicUsize>,
        drop_notify: Option<Arc<tokio::sync::Notify>>,
    }

    impl Iterator for TrackedReader {
        type Item = Result<RecordBatch, ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            self.batch.take().map(Ok)
        }
    }

    impl RecordBatchReader for TrackedReader {
        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }
    }

    impl Drop for TrackedReader {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
            if let Some(notify) = &self.drop_notify {
                notify.notify_one();
            }
        }
    }

    #[tokio::test]
    async fn test_factory_is_lazy_reexecuted_and_reader_released() {
        Python::initialize();
        let expected = record_batch!(("id", Int64, [1, 2])).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let partition = Python::attach(|py| {
            let calls = calls.clone();
            let dropped = dropped.clone();
            let batch = expected.clone();
            let factory = PyCFunction::new_closure(py, None, None, move |args, _| {
                // This would panic on a Tokio async worker. A Python factory can
                // legitimately call back into a synchronous Lance API.
                tokio::runtime::Handle::current().block_on(async {});
                calls.fetch_add(1, Ordering::SeqCst);
                let reader: Box<dyn RecordBatchReader + Send> = Box::new(TrackedReader {
                    batch: Some(batch.clone()),
                    schema: batch.schema(),
                    dropped: dropped.clone(),
                    drop_notify: None,
                });
                Py::new(args.py(), ReaderSource(std::sync::Mutex::new(Some(reader))))
            })
            .unwrap();
            ReplayablePartition {
                schema: expected.schema(),
                factory: Arc::new(factory.into_any().unbind()),
            }
        });
        for attempt in 0..2 {
            let mut stream = partition.execute(Arc::new(TaskContext::default()));
            assert_eq!(calls.load(Ordering::SeqCst), attempt);
            assert_eq!(stream.try_next().await.unwrap().unwrap(), expected);
            drop(stream);
            assert_eq!(calls.load(Ordering::SeqCst), attempt + 1);
            assert_eq!(dropped.load(Ordering::SeqCst), attempt + 1);
        }
    }

    #[tokio::test]
    async fn test_cancelled_factory_releases_reader_when_it_returns() {
        Python::initialize();
        let batch = record_batch!(("id", Int64, [1])).unwrap();
        let schema = batch.schema();
        let started = Arc::new(tokio::sync::Notify::new());
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let release_receiver = std::sync::Mutex::new(release_receiver);
        let dropped = Arc::new(tokio::sync::Notify::new());
        let partition = Python::attach(|py| {
            let started = started.clone();
            let dropped = dropped.clone();
            let factory = PyCFunction::new_closure(py, None, None, move |args, _| {
                started.notify_one();
                args.py()
                    .detach(|| {
                        release_receiver
                            .lock()
                            .unwrap()
                            .recv_timeout(std::time::Duration::from_secs(1))
                    })
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                let reader: Box<dyn RecordBatchReader + Send> = Box::new(TrackedReader {
                    batch: Some(batch.clone()),
                    schema: batch.schema(),
                    dropped: Arc::new(AtomicUsize::new(0)),
                    drop_notify: Some(dropped.clone()),
                });
                Py::new(args.py(), ReaderSource(std::sync::Mutex::new(Some(reader))))
            })
            .unwrap();
            ReplayablePartition {
                schema,
                factory: Arc::new(factory.into_any().unbind()),
            }
        });
        let mut stream = partition.execute(Arc::new(TaskContext::default()));
        let task = tokio::spawn(async move { stream.try_next().await });
        tokio::time::timeout(std::time::Duration::from_secs(1), started.notified())
            .await
            .expect("source factory should start");
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        // Blocking callbacks cannot be interrupted, but their eventual reader
        // must be released without being consumed once the stream is cancelled.
        release_sender.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), dropped.notified())
            .await
            .expect("cancelled factory result should release its reader");
    }

    #[tokio::test]
    async fn test_factory_error_retains_context() {
        Python::initialize();
        let partition = Python::attach(|py| {
            let factory = PyCFunction::new_closure(py, None, None, |_, _| -> PyResult<()> {
                Err(PyRuntimeError::new_err("source factory failed"))
            })
            .unwrap();
            ReplayablePartition {
                schema: Arc::new(arrow_schema::Schema::empty()),
                factory: Arc::new(factory.into_any().unbind()),
            }
        });
        let error = partition
            .execute(Arc::new(TaskContext::default()))
            .try_next()
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Failed to open replayable merge source")
        );
        assert!(error.to_string().contains("source factory failed"));
    }

    #[tokio::test]
    async fn test_reader_schema_metadata_must_match() {
        Python::initialize();
        let batch = record_batch!(("id", Int64, [1])).unwrap();
        let expected_schema = Arc::new(
            batch.schema().as_ref().clone().with_metadata(
                [("source_version".to_string(), "1".to_string())]
                    .into_iter()
                    .collect(),
            ),
        );
        let partition = Python::attach(|py| {
            let factory = PyCFunction::new_closure(py, None, None, move |args, _| {
                let reader: Box<dyn RecordBatchReader + Send> = Box::new(
                    arrow_array::RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                );
                Py::new(args.py(), ReaderSource(std::sync::Mutex::new(Some(reader))))
            })
            .unwrap();
            ReplayablePartition {
                schema: expected_schema,
                factory: Arc::new(factory.into_any().unbind()),
            }
        });
        let error = partition
            .execute(Arc::new(TaskContext::default()))
            .try_next()
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Replayable merge source schema changed")
        );
        assert!(error.to_string().contains("source_version"));
    }
}
