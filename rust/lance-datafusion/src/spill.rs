// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    io::{BufReader, BufWriter},
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema, SchemaRef};
use datafusion::{
    catalog::{TableProvider, streaming::StreamingTable},
    execution::{
        SendableRecordBatchStream, TaskContext,
        disk_manager::{DiskManager, RefCountedTempFile},
    },
    physical_plan::{stream::RecordBatchStreamAdapter, streaming::PartitionStream},
};
use datafusion_common::DataFusionError;
use futures::StreamExt;
use lance_arrow::memory::MemoryAccumulator;
use lance_core::error::LanceOptionExt;

fn disk_manager_accounting_lock(disk_manager: &Arc<DiskManager>) -> Arc<Mutex<()>> {
    static ACCOUNTING_LOCKS: OnceLock<Mutex<HashMap<usize, Weak<Mutex<()>>>>> = OnceLock::new();
    let locks = ACCOUNTING_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks.lock().unwrap_or_else(|error| error.into_inner());
    locks.retain(|_, lock| lock.strong_count() > 0);

    let key = Arc::as_ptr(disk_manager) as usize;
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return lock;
    }

    let lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    lock
}

/// Start a spill of Arrow data to a file that can be read later multiple times.
///
/// Up to `memory_limit` bytes of data can be buffered in memory before a spill
/// is created. If the memory limit is never reached before [`SpillSender::finish()`]
/// is called, then the data will simply be kept in memory and no spill will be
/// created.
///
/// `path` is the path to the file that may be created. It should not already
/// exist. It is the responsibility of the caller to delete the file after it is
/// no longer needed.
///
/// The [`SpillSender`] allows you to write batches to the spill.
///
/// The [`SpillReceiver`] can open a [`SendableRecordBatchStream`] that reads
/// batches from the spill. This can be opened before, during, or after batches
/// have been written to the spill.
///
/// Once [`SpillSender`] is dropped, the temporary file is deleted. This will
/// cause the [`SpillReceiver`] to return an error if it is still open.
pub fn create_replay_spill(
    path: std::path::PathBuf,
    schema: Arc<Schema>,
    memory_limit: usize,
) -> (SpillSender, SpillReceiver) {
    let initial_status = WriteStatus::default();
    let (status_sender, status_receiver) = tokio::sync::watch::channel(initial_status);
    let sender = SpillSender {
        memory_limit,
        managed_spill: None,
        path: path.clone(),
        schema: schema.clone(),
        state: SpillState::default(),
        status_sender,
    };

    let receiver = SpillReceiver {
        status_receiver,
        path,
        schema,
    };

    (sender, receiver)
}

/// Wrap a one-shot stream in a replayable provider using a job-local disk manager.
///
/// Up to `memory_limit` bytes are buffered before spilling, and scans can start
/// replaying batches while the source is still draining. Every spill file is
/// charged to `disk_manager`.
/// Sharing that manager with DataFusion execution enforces one aggregate quota
/// across replay and operator spill files. Dropping the provider aborts an
/// unfinished background drain of the source.
///
/// The manager must belong to one operation and must be discarded if any spill
/// reports an error. DataFusion 54.1.0 can retain a rejected operator charge;
/// operation-scoped ownership prevents that failed accounting from affecting a
/// later job.
///
/// # Examples
///
/// ```
/// # use std::sync::Arc;
/// # use arrow_array::{Int32Array, RecordBatch};
/// # use arrow_schema::{DataType, Field, Schema};
/// # use datafusion::execution::SendableRecordBatchStream;
/// # use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
/// # use futures::TryStreamExt;
/// # use lance_datafusion::exec::provider_to_stream;
/// # use datafusion::execution::disk_manager::DiskManager;
/// # use lance_datafusion::spill::spilling_table_provider_with_job_disk_manager;
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
/// let batch = RecordBatch::try_new(
///     schema.clone(),
///     vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
/// )?;
/// let source: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
///     schema,
///     futures::stream::iter(vec![Ok(batch)]),
/// ));
/// let disk_manager = Arc::new(
///     DiskManager::builder()
///         .with_max_temp_directory_size(1024 * 1024)
///         .build()?,
/// );
/// let provider =
///     spilling_table_provider_with_job_disk_manager(source, 0, disk_manager).await?;
/// let replayed: Vec<RecordBatch> = provider_to_stream(provider).await?.try_collect().await?;
/// assert_eq!(replayed.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
/// # Ok(())
/// # }
/// ```
pub async fn spilling_table_provider_with_job_disk_manager(
    mut source: SendableRecordBatchStream,
    memory_limit: usize,
    disk_manager: Arc<DiskManager>,
) -> Result<Arc<dyn TableProvider>, DataFusionError> {
    let schema = source.schema();
    let spill_file = disk_manager.create_tmp_file("writing replay spill")?;
    let spill_path = spill_file.path().to_owned();
    let (mut sender, receiver) = create_replay_spill(spill_path, schema.clone(), memory_limit);
    let accounting_lock = disk_manager_accounting_lock(&disk_manager);
    sender.managed_spill = Some(ManagedSpillFile {
        file: spill_file.clone(),
        disk_manager,
        accounting_lock,
    });

    // Drain the one-shot source into the spill once, in the background. The spill
    // tees to memory/disk so the first reader can consume batches as they arrive
    // while later readers replay the complete source.
    let drain_handle = tokio::task::spawn(async move {
        let mut errored = false;
        while let Some(res) = source.next().await {
            match res {
                Ok(batch) => {
                    if let Err(e) = sender.write(batch).await {
                        sender.send_error(e);
                        errored = true;
                        break;
                    }
                }
                Err(e) => {
                    sender.send_error(e);
                    errored = true;
                    break;
                }
            }
        }
        // Only finish on a clean drain. Calling finish() after an error would
        // overwrite the original (replayable) error with a generic one, losing
        // the source error's type (e.g. an external error from user code).
        if !errored && let Err(err) = sender.finish().await {
            sender.send_error(err);
        }
        sender
    });

    let partition = Arc::new(SpillPartition {
        schema: schema.clone(),
        receiver,
        _drain_handle: Arc::new(AbortOnDropHandle::new(drain_handle)),
        _spill_file: Arc::new(spill_file),
    });
    Ok(Arc::new(StreamingTable::try_new(schema, vec![partition])?))
}

struct AbortOnDropHandle<T> {
    handle: tokio::task::JoinHandle<T>,
}

impl<T> AbortOnDropHandle<T> {
    fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self { handle }
    }
}

impl<T> Drop for AbortOnDropHandle<T> {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// A [`PartitionStream`] backed by a replayable spill.
///
/// Each call to [`PartitionStream::execute`] opens a fresh stream over the spill,
/// so the partition can be scanned repeatedly. The spill file and the background
/// task draining the source are kept alive for as long as this partition exists.
struct SpillPartition {
    schema: SchemaRef,
    receiver: SpillReceiver,
    // Keeps the background drain task (which owns the `SpillSender`) alive. The
    // `SpillSender` must outlive the readers or they error out, so we hold the
    // handle rather than detaching it. It is declared before the spill file so
    // cancellation is requested before spill cleanup begins.
    _drain_handle: Arc<AbortOnDropHandle<SpillSender>>,
    // Keeps the DiskManager-owned spill file alive and accounts its disk usage.
    _spill_file: Arc<RefCountedTempFile>,
}

impl std::fmt::Debug for SpillPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpillPartition")
            .field("schema", &self.schema)
            .finish()
    }
}

impl PartitionStream for SpillPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let stream = self.receiver.read();
        let drain_handle = self._drain_handle.clone();
        let spill_file = self._spill_file.clone();
        let stream = stream.map(move |batch| {
            // The reader may outlive the provider and physical plan that
            // created it. Keep the drain and spill file alive until this
            // stream itself is dropped.
            let _keep_alive = (&drain_handle, &spill_file);
            batch
        });
        Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream))
    }
}

#[derive(Clone)]
pub struct SpillReceiver {
    status_receiver: tokio::sync::watch::Receiver<WriteStatus>,
    path: PathBuf,
    schema: Arc<Schema>,
}

impl SpillReceiver {
    /// Returns a stream of batches from the spill. The stream will emit
    /// batches as they are written to the spill. If the spill has already
    /// been finished, the stream will emit all batches in the spill.
    ///
    /// The stream will not complete until [`SpillSender::finish()`] is called.
    ///
    /// If the spill has been dropped, an error will be returned.
    pub fn read(&self) -> SendableRecordBatchStream {
        let rx = self.status_receiver.clone();
        let reader = SpillReader::new(rx, self.path.clone());

        let stream = futures::stream::try_unfold(reader, move |mut reader| async move {
            match reader.read().await {
                Ok(None) => Ok(None),
                Ok(Some(batch)) => Ok(Some((batch, reader))),
                Err(err) => Err(err),
            }
        });

        Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream))
    }
}

struct SpillReader {
    pub batches_read: usize,
    receiver: tokio::sync::watch::Receiver<WriteStatus>,
    state: SpillReaderState,
}

enum SpillReaderState {
    Buffered { spill_path: PathBuf },
    Reader { reader: AsyncStreamReader },
}

impl SpillReader {
    fn new(receiver: tokio::sync::watch::Receiver<WriteStatus>, spill_path: PathBuf) -> Self {
        Self {
            batches_read: 0,
            receiver,
            state: SpillReaderState::Buffered { spill_path },
        }
    }

    async fn wait_for_more_data(&mut self) -> Result<Option<Arc<[RecordBatch]>>, DataFusionError> {
        let status = self
            .receiver
            .wait_for(|status| {
                status.error.is_some()
                    || status.finished
                    || status.batches_written() > self.batches_read
            })
            .await
            .map_err(|_| {
                DataFusionError::Execution(
                    "Spill has been dropped before reader has finish.".into(),
                )
            })?;

        if let Some(error) = &status.error {
            let mut guard = error.lock().ok().expect_ok()?;
            return Err(DataFusionError::from(&mut (*guard)));
        }

        if let DataLocation::Buffered { batches } = &status.data_location {
            Ok(Some(batches.clone()))
        } else {
            Ok(None)
        }
    }

    async fn get_reader(&mut self) -> Result<&AsyncStreamReader, ArrowError> {
        if let SpillReaderState::Buffered { spill_path } = &self.state {
            let reader = AsyncStreamReader::open(spill_path.clone()).await?;
            // Skip batches we've already read before the writer started spilling.
            // The read batches were spilled to the file for the benefit of
            // future readers, as the spill is replay-able.
            for _ in 0..self.batches_read {
                reader.read().await?;
            }
            self.state = SpillReaderState::Reader { reader };
        }

        if let SpillReaderState::Reader { reader } = &mut self.state {
            Ok(reader)
        } else {
            unreachable!()
        }
    }

    async fn read(&mut self) -> Result<Option<RecordBatch>, DataFusionError> {
        let maybe_data = self.wait_for_more_data().await?;

        if let Some(batches) = maybe_data {
            if self.batches_read < batches.len() {
                let batch = batches[self.batches_read].clone();
                self.batches_read += 1;
                Ok(Some(batch))
            } else {
                Ok(None)
            }
        } else {
            let reader = self.get_reader().await?;
            let batch = reader.read().await?;
            if batch.is_some() {
                self.batches_read += 1;
            }
            Ok(batch)
        }
    }
}

/// The sender side of the spill. This is used to write batches to the spill.
///
/// Note: this must be kept alive until after the readers are done reading the
/// spill. Otherwise, they will return an error.
pub struct SpillSender {
    memory_limit: usize,
    managed_spill: Option<ManagedSpillFile>,
    schema: Arc<Schema>,
    path: PathBuf,
    state: SpillState,
    status_sender: tokio::sync::watch::Sender<WriteStatus>,
}

enum SpillState {
    Buffering {
        batches: Vec<RecordBatch>,
        memory_accumulator: MemoryAccumulator,
    },
    Spilling {
        writer: AsyncStreamWriter,
        batches_written: usize,
    },
    Finished {
        batches: Option<Arc<[RecordBatch]>>,
        batches_written: usize,
    },
    Errored {
        error: Arc<Mutex<SpillError>>,
    },
}

impl Default for SpillState {
    fn default() -> Self {
        Self::Buffering {
            batches: Vec::new(),
            memory_accumulator: MemoryAccumulator::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct WriteStatus {
    error: Option<Arc<Mutex<SpillError>>>,
    finished: bool,
    data_location: DataLocation,
}

impl WriteStatus {
    fn batches_written(&self) -> usize {
        match &self.data_location {
            DataLocation::Buffered { batches } => batches.len(),
            DataLocation::Spilled {
                batches_written, ..
            } => *batches_written,
        }
    }
}

#[derive(Clone, Debug)]
enum DataLocation {
    Buffered { batches: Arc<[RecordBatch]> },
    Spilled { batches_written: usize },
}

impl Default for DataLocation {
    fn default() -> Self {
        Self::Buffered {
            batches: Arc::new([]),
        }
    }
}

/// A DataFusion error that can be emitted multiple times. We provide the
/// Original error first, and subsequent conversions provide a copy with a
/// string representation of the original error.
#[derive(Debug)]
enum SpillError {
    Original(DataFusionError),
    Copy(DataFusionError),
}

impl From<DataFusionError> for SpillError {
    fn from(err: DataFusionError) -> Self {
        Self::Original(err)
    }
}

impl From<&mut SpillError> for DataFusionError {
    fn from(err: &mut SpillError) -> Self {
        match err {
            SpillError::Original(inner) => {
                let copy = Self::Execution(inner.to_string());
                let original = std::mem::replace(err, SpillError::Copy(copy));
                if let SpillError::Original(inner) = original {
                    inner
                } else {
                    unreachable!()
                }
            }
            SpillError::Copy(Self::Execution(message)) => Self::Execution(message.clone()),
            _ => unreachable!(),
        }
    }
}

impl From<&SpillState> for WriteStatus {
    fn from(state: &SpillState) -> Self {
        match state {
            SpillState::Buffering { batches, .. } => Self {
                finished: false,
                data_location: DataLocation::Buffered {
                    batches: batches.clone().into(),
                },
                error: None,
            },
            SpillState::Spilling {
                batches_written, ..
            } => Self {
                finished: false,
                data_location: DataLocation::Spilled {
                    batches_written: *batches_written,
                },
                error: None,
            },
            SpillState::Finished {
                batches_written,
                batches,
            } => {
                let data_location = if let Some(batches) = batches {
                    DataLocation::Buffered {
                        batches: batches.clone(),
                    }
                } else {
                    DataLocation::Spilled {
                        batches_written: *batches_written,
                    }
                };
                Self {
                    finished: true,
                    data_location,
                    error: None,
                }
            }
            SpillState::Errored { error } => Self {
                finished: true,
                data_location: DataLocation::default(), // Doesn't matter.
                error: Some(error.clone()),
            },
        }
    }
}

impl SpillSender {
    /// Write a batch to the spill.  
    ///  
    /// If there is room in the `memory_limit` then the batch is queued.  
    /// If `memory_limit` is first encountered then all queued batches, and this one,  
    /// will be written to disk as part of this call.  
    /// If we are already spilling then the batch will be written to disk as part of this  
    /// call.
    pub async fn write(&mut self, batch: RecordBatch) -> Result<(), DataFusionError> {
        if let SpillState::Finished { .. } = self.state {
            return Err(DataFusionError::Execution(
                "Spill has already been finished".to_string(),
            ));
        }

        if let SpillState::Errored { .. } = &self.state {
            return Err(DataFusionError::Execution(
                "Spill has sent an error".to_string(),
            ));
        }

        let (writer, batches_written) = match &mut self.state {
            SpillState::Buffering {
                batches,
                memory_accumulator,
            } => {
                memory_accumulator.record_batch(&batch);

                if memory_accumulator.total() > self.memory_limit {
                    let writer = AsyncStreamWriter::open(
                        self.path.clone(),
                        self.schema.clone(),
                        self.managed_spill.clone(),
                    )
                    .await?;
                    let batches_written = batches.len();
                    for batch in batches.drain(..) {
                        writer.write(batch).await?;
                    }
                    self.state = SpillState::Spilling {
                        writer,
                        batches_written,
                    };
                    if let SpillState::Spilling {
                        writer,
                        batches_written,
                    } = &mut self.state
                    {
                        (writer, batches_written)
                    } else {
                        unreachable!()
                    }
                } else {
                    batches.push(batch);
                    self.status_sender
                        .send_replace(WriteStatus::from(&self.state));
                    return Ok(());
                }
            }
            SpillState::Spilling {
                writer,
                batches_written,
            } => (writer, batches_written),
            _ => unreachable!(),
        };

        writer.write(batch).await?;
        *batches_written += 1;
        self.status_sender
            .send_replace(WriteStatus::from(&self.state));

        Ok(())
    }

    /// Send an error to the spill. This will be sent to all readers of the
    /// spill.
    pub fn send_error(&mut self, err: DataFusionError) {
        let error = Arc::new(Mutex::new(err.into()));
        self.state = SpillState::Errored { error };
        self.status_sender
            .send_replace(WriteStatus::from(&self.state));
    }

    /// Complete the spill write. This will finalize the Arrow IPC stream file.
    /// The file will remain available for reading until the spill is dropped.
    pub async fn finish(&mut self) -> Result<(), DataFusionError> {
        // We create a temporary state to get an owned copy of current state.
        // Since we hold an exclusive reference to `self`, no one should be
        // able to see this temporary state.
        let tmp_state = SpillState::Finished {
            batches_written: 0,
            batches: None,
        };
        match std::mem::replace(&mut self.state, tmp_state) {
            SpillState::Buffering { batches, .. } => {
                let batches_written = batches.len();
                self.state = SpillState::Finished {
                    batches_written,
                    batches: Some(batches.into()),
                };
                self.status_sender
                    .send_replace(WriteStatus::from(&self.state));
            }
            SpillState::Spilling {
                writer,
                batches_written,
            } => {
                writer.finish().await?;
                self.state = SpillState::Finished {
                    batches_written,
                    batches: None,
                };
                self.status_sender
                    .send_replace(WriteStatus::from(&self.state));
            }
            SpillState::Finished { .. } => {
                return Err(DataFusionError::Execution(
                    "Spill has already been finished".to_string(),
                ));
            }
            SpillState::Errored { .. } => {
                return Err(DataFusionError::Execution(
                    "Spill has sent an error".to_string(),
                ));
            }
        };

        Ok(())
    }
}

/// An async wrapper around [`StreamWriter`]. Each call uses [`tokio::task::spawn_blocking`]
/// to spawn a blocking task to write the batch.
struct AsyncStreamWriter {
    writer: Arc<Mutex<TrackedStreamWriter>>,
}

impl AsyncStreamWriter {
    pub async fn open(
        path: PathBuf,
        schema: Arc<Schema>,
        managed_spill: Option<ManagedSpillFile>,
    ) -> Result<Self, ArrowError> {
        let writer = tokio::task::spawn_blocking(move || {
            let file = if let Some(spill) = managed_spill.as_ref() {
                spill.file.inner().reopen().map_err(ArrowError::from)?
            } else {
                std::fs::File::create(&path).map_err(ArrowError::from)?
            };
            TrackedStreamWriter::try_new(file, &schema, managed_spill)
        })
        .await
        .unwrap()?;
        let writer = Arc::new(Mutex::new(writer));
        Ok(Self { writer })
    }

    pub async fn write(&self, batch: RecordBatch) -> Result<(), ArrowError> {
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut writer = writer.lock().unwrap();
            writer.write(&batch)
        })
        .await
        .unwrap()
    }

    pub async fn finish(self) -> Result<(), ArrowError> {
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut writer = writer.lock().unwrap();
            writer.finish()
        })
        .await
        .unwrap()
    }
}

#[derive(Clone)]
struct ManagedSpillFile {
    file: RefCountedTempFile,
    disk_manager: Arc<DiskManager>,
    accounting_lock: Arc<Mutex<()>>,
}

impl ManagedSpillFile {
    fn update_disk_usage(&mut self) -> Result<(), ArrowError> {
        // Keep replay-file checks and mutations atomic with respect to other
        // replay files. A mixed operator/replay failure aborts the owning job,
        // which then discards this operation-scoped disk manager.
        let _accounting_guard = self
            .accounting_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let new_file_usage = self
            .file
            .inner()
            .as_file()
            .metadata()
            .map_err(ArrowError::from)?
            .len();
        let old_file_usage = self.file.current_disk_usage();
        let used_without_file = self
            .disk_manager
            .used_disk_space()
            .checked_sub(old_file_usage)
            .ok_or_else(|| {
                ArrowError::ExternalError(Box::new(DataFusionError::Internal(format!(
                    "replay spill accounting is inconsistent: used_disk_space={} is less than current_file_disk_usage={old_file_usage}",
                    self.disk_manager.used_disk_space()
                ))))
            })?;
        if used_without_file
            .checked_add(new_file_usage)
            .is_none_or(|usage| usage > self.disk_manager.max_temp_directory_size())
        {
            return Err(ArrowError::ExternalError(Box::new(
                DataFusionError::ResourcesExhausted(format!(
                    "The used disk space during the spilling process has exceeded the allowable limit of {} bytes. Please try increasing the config: `datafusion.runtime.max_temp_directory_size`.",
                    self.disk_manager.max_temp_directory_size()
                )),
            )));
        }

        self.file
            .update_disk_usage()
            .map_err(|error| ArrowError::ExternalError(Box::new(error)))
    }
}

struct TrackedStreamWriter {
    writer: StreamWriter<BufWriter<std::fs::File>>,
    managed_spill: Option<ManagedSpillFile>,
}

impl TrackedStreamWriter {
    fn try_new(
        file: std::fs::File,
        schema: &Schema,
        managed_spill: Option<ManagedSpillFile>,
    ) -> Result<Self, ArrowError> {
        let writer = StreamWriter::try_new(BufWriter::new(file), schema)?;
        let mut tracked = Self {
            writer,
            managed_spill,
        };
        tracked.flush_and_update_disk_usage()?;
        Ok(tracked)
    }

    fn write(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        self.writer.write(batch)?;
        self.flush_and_update_disk_usage()
    }

    fn finish(&mut self) -> Result<(), ArrowError> {
        self.writer.finish()?;
        self.flush_and_update_disk_usage()
    }

    fn flush_and_update_disk_usage(&mut self) -> Result<(), ArrowError> {
        self.writer.flush()?;
        if let Some(spill) = self.managed_spill.as_mut() {
            spill.update_disk_usage()?;
        }
        Ok(())
    }
}

struct AsyncStreamReader {
    reader: Arc<Mutex<StreamReader<BufReader<std::fs::File>>>>,
}

impl AsyncStreamReader {
    pub async fn open(path: PathBuf) -> Result<Self, ArrowError> {
        let reader = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&path).map_err(ArrowError::from)?;
            let reader = BufReader::new(file);
            StreamReader::try_new(reader, None)
        })
        .await
        .unwrap()?;
        let reader = Arc::new(Mutex::new(reader));
        Ok(Self { reader })
    }

    pub async fn read(&self) -> Result<Option<RecordBatch>, ArrowError> {
        let reader = self.reader.clone();
        tokio::task::spawn_blocking(move || {
            let mut reader = reader.lock().unwrap();
            reader.next()
        })
        .await
        .unwrap()
        .transpose()
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field};
    use futures::{StreamExt, TryStreamExt, poll};
    use lance_core::utils::tempfile::{TempStdFile, TempStdPath};
    use tokio::sync::{Barrier, oneshot};

    use super::*;
    use crate::exec::provider_to_stream;

    struct DropSignal(Option<oneshot::Sender<()>>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    #[tokio::test]
    async fn test_spill() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batches = [
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
            )
            .unwrap(),
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![4, 5, 6]))],
            )
            .unwrap(),
        ];

        // Create a stream
        let path = TempStdFile::default();
        let (mut spill, receiver) = create_replay_spill(path.to_owned(), schema.clone(), 0);

        // We can open a reader prior to writing any data. No batches will be ready.
        let mut stream_before = receiver.read();
        let mut stream_before_next = stream_before.next();
        let poll_res = poll!(&mut stream_before_next);
        assert!(poll_res.is_pending());

        // If we write a batch, the existing reader can now receive it.
        spill.write(batches[0].clone()).await.unwrap();
        let stream_before_batch1 = stream_before_next
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_before_batch1, &batches[0]);
        let mut stream_before_next = stream_before.next();
        let poll_res = poll!(&mut stream_before_next);
        assert!(poll_res.is_pending());

        // We can also open a ready while the spill is being written to. We can
        // retrieve batches written so far immediately.
        let mut stream_during = receiver.read();
        let stream_during_batch1 = stream_during
            .next()
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_during_batch1, &batches[0]);
        let mut stream_during_next = stream_during.next();
        let poll_res = poll!(&mut stream_during_next);
        assert!(poll_res.is_pending());

        // Once we finish the spill, readers can get remaining batches and will
        // reach the end of the stream.
        spill.write(batches[1].clone()).await.unwrap();
        spill.finish().await.unwrap();

        let stream_before_batch2 = stream_before_next
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_before_batch2, &batches[1]);
        assert!(stream_before.next().await.is_none());

        let stream_during_batch2 = stream_during_next
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_during_batch2, &batches[1]);
        assert!(stream_during.next().await.is_none());

        // Can also start a reader after finishing.
        let stream_after = receiver.read();
        let stream_after_batches = stream_after.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(&stream_after_batches, &batches);

        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn test_spill_error() {
        // Create a spill
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let path = TempStdFile::default();
        let (mut spill, receiver) =
            create_replay_spill(path.as_ref().to_owned(), schema.clone(), 0);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        spill.write(batch.clone()).await.unwrap();

        let mut stream = receiver.read();
        let stream_batch = stream
            .next()
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_batch, &batch);

        spill.send_error(DataFusionError::ResourcesExhausted("🥱".into()));
        let stream_error = stream
            .next()
            .await
            .expect("Expected an error")
            .expect_err("Expected an error");
        assert!(matches!(
            stream_error,
            DataFusionError::ResourcesExhausted(message) if message == "🥱"
        ));

        // If we try to write after sending an error, it should return an error.
        let err = spill.write(batch).await;
        assert!(matches!(
            err,
            Err(DataFusionError::Execution(message)) if message == "Spill has sent an error"
        ));

        // If we try to finish after sending an error, it should return an error.
        let err = spill.finish().await;
        assert!(matches!(
            err,
            Err(DataFusionError::Execution(message)) if message == "Spill has sent an error"
        ));

        // If we try to read after sending an error, it should return an error.
        let mut stream = receiver.read();
        let stream_error = stream
            .next()
            .await
            .expect("Expected an error")
            .expect_err("Expected an error");
        assert!(matches!(
            stream_error,
            DataFusionError::Execution(message) if message.contains("🥱")
        ));

        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn test_spill_buffered() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let path = TempStdPath::default();
        let memory_limit = 1024 * 1024; // 1 MiB
        let (mut spill, receiver) = create_replay_spill(path.clone(), schema.clone(), memory_limit);

        // 0.5 MB batch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1; (512 * 1024) / 4]))],
        )
        .unwrap();
        spill.write(batch.clone()).await.unwrap();
        assert!(!std::fs::exists(&path).unwrap());

        spill.finish().await.unwrap();
        assert!(!std::fs::exists(&path).unwrap());

        let mut stream = receiver.read();
        let stream_batch = stream
            .next()
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_batch, &batch);

        assert!(!std::fs::exists(&path).unwrap());
    }

    #[tokio::test]
    async fn test_spill_buffered_transition() {
        // Starts as buffered, then spills, then finished.
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let path = TempStdPath::default();
        let memory_limit = 1024 * 1024; // 1 MiB
        let (mut spill, receiver) = create_replay_spill(path.clone(), schema.clone(), memory_limit);

        // 0.7 MB batch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1; (768 * 1024) / 4]))],
        )
        .unwrap();
        spill.write(batch.clone()).await.unwrap();
        assert!(!std::fs::exists(&path).unwrap());

        let mut stream = receiver.read();
        let stream_batch = stream
            .next()
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_batch, &batch);
        assert!(!std::fs::exists(&path).unwrap());

        // 0.5 MB batch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1; (512 * 1024) / 4]))],
        )
        .unwrap();
        spill.write(batch.clone()).await.unwrap();
        assert!(std::fs::exists(&path).unwrap());

        let stream_batch = stream
            .next()
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_batch, &batch);
        assert!(std::fs::exists(&path).unwrap());

        spill.finish().await.unwrap();

        assert!(stream.next().await.is_none());

        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn test_spilling_table_provider_honors_disk_limit() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1; 1024]))],
        )
        .unwrap();
        let source = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter([Ok(batch)]),
        ));

        let disk_manager = Arc::new(
            DiskManager::builder()
                .with_max_temp_directory_size(64)
                .build()
                .unwrap(),
        );
        let provider =
            spilling_table_provider_with_job_disk_manager(source, 0, disk_manager.clone())
                .await
                .unwrap();
        let error = provider_to_stream(provider)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("exceeded the allowable limit"),
            "unexpected replay quota error: {message}"
        );
        assert_eq!(
            disk_manager.used_disk_space(),
            0,
            "failed replay spills must release their disk reservation"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_quota_errors_release_disk_usage() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1; 1024]))],
        )
        .unwrap();
        let header_size = {
            let mut buffer = Vec::new();
            let writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
            drop(writer);
            u64::try_from(buffer.len()).unwrap()
        };
        let disk_manager = Arc::new(
            DiskManager::builder()
                .with_max_temp_directory_size(header_size * 2 - 1)
                .build()
                .unwrap(),
        );
        let barrier = Arc::new(Barrier::new(2));
        let source = |batch: RecordBatch| {
            let barrier = barrier.clone();
            Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::once(async move {
                    barrier.wait().await;
                    Ok(batch)
                }),
            )) as SendableRecordBatchStream
        };
        let first = spilling_table_provider_with_job_disk_manager(
            source(batch.clone()),
            0,
            disk_manager.clone(),
        )
        .await
        .unwrap();
        let second =
            spilling_table_provider_with_job_disk_manager(source(batch), 0, disk_manager.clone())
                .await
                .unwrap();
        let collect = |provider| async move {
            provider_to_stream(provider)
                .await
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
        };
        let (first_result, second_result) = tokio::join!(collect(first), collect(second));
        assert!(first_result.is_err());
        assert!(second_result.is_err());
        tokio::task::yield_now().await;
        assert_eq!(
            disk_manager.used_disk_space(),
            0,
            "concurrent replay quota failures must release their disk reservations"
        );
    }

    #[tokio::test]
    async fn test_mixed_spill_failure_isolated_from_next_job() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1; 1024]))],
        )
        .unwrap();
        let serialized_size = {
            let mut buffer = Vec::new();
            let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
            drop(writer);
            u64::try_from(buffer.len()).unwrap()
        };
        let source = |batch: RecordBatch| {
            Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter([Ok(batch)]),
            )) as SendableRecordBatchStream
        };

        let failed_job_manager = Arc::new(
            DiskManager::builder()
                .with_max_temp_directory_size(serialized_size)
                .build()
                .unwrap(),
        );
        let provider = spilling_table_provider_with_job_disk_manager(
            source(batch.clone()),
            0,
            failed_job_manager.clone(),
        )
        .await
        .unwrap();
        provider_to_stream(provider.clone())
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let mut operator_spill = failed_job_manager
            .create_tmp_file("testing mixed spill accounting")
            .unwrap();
        operator_spill.inner().as_file().set_len(1).unwrap();
        assert!(matches!(
            operator_spill.update_disk_usage(),
            Err(DataFusionError::ResourcesExhausted(_))
        ));
        drop(operator_spill);
        drop(provider);
        drop(failed_job_manager);

        // A quota error terminates the operation and discards its manager. The
        // next job starts with independent accounting even on DataFusion 54.1.0.
        let next_job_manager = Arc::new(
            DiskManager::builder()
                .with_max_temp_directory_size(serialized_size)
                .build()
                .unwrap(),
        );
        assert_eq!(next_job_manager.used_disk_space(), 0);
        let next_provider = spilling_table_provider_with_job_disk_manager(
            source(batch),
            0,
            next_job_manager.clone(),
        )
        .await
        .unwrap();
        provider_to_stream(next_provider.clone())
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        drop(next_provider);
        tokio::task::yield_now().await;
        assert_eq!(next_job_manager.used_disk_space(), 0);
    }

    #[tokio::test]
    async fn test_spilling_table_providers_share_disk_limit() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1; 1024]))],
        )
        .unwrap();
        let serialized_size = {
            let mut buffer = Vec::new();
            let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
            drop(writer);
            u64::try_from(buffer.len()).unwrap()
        };
        let disk_manager = Arc::new(
            DiskManager::builder()
                .with_max_temp_directory_size(serialized_size * 2 - 1)
                .build()
                .unwrap(),
        );

        let source = |batch: RecordBatch| {
            Box::pin(RecordBatchStreamAdapter::new(
                schema.clone(),
                futures::stream::iter([Ok(batch)]),
            )) as SendableRecordBatchStream
        };
        let first = spilling_table_provider_with_job_disk_manager(
            source(batch.clone()),
            0,
            disk_manager.clone(),
        )
        .await
        .unwrap();
        provider_to_stream(first.clone())
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        let second =
            spilling_table_provider_with_job_disk_manager(source(batch), 0, disk_manager.clone())
                .await
                .unwrap();
        let error = provider_to_stream(second)
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("exceeded the allowable limit"),
            "unexpected shared replay quota error: {error}"
        );
        drop(first);
    }

    #[tokio::test]
    async fn test_spilling_table_provider_aborts_drain_on_drop() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let (drop_sender, drop_receiver) = oneshot::channel();
        let drop_signal = DropSignal(Some(drop_sender));
        let pending_source = futures::stream::poll_fn(move |_context| {
            let _keep_alive = &drop_signal;
            std::task::Poll::<Option<Result<RecordBatch, DataFusionError>>>::Pending
        });
        let source = Box::pin(RecordBatchStreamAdapter::new(schema, pending_source));

        let disk_manager = Arc::new(
            DiskManager::builder()
                .with_max_temp_directory_size(1024)
                .build()
                .unwrap(),
        );
        let provider = spilling_table_provider_with_job_disk_manager(source, 0, disk_manager)
            .await
            .unwrap();
        tokio::task::yield_now().await;
        drop(provider);

        tokio::time::timeout(std::time::Duration::from_secs(1), drop_receiver)
            .await
            .expect("source drain was not aborted when the replay provider was dropped")
            .expect("source drain drop signal was canceled");
    }
}
