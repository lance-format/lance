// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::pin::Pin;
use std::task::Poll;
use std::{collections::VecDeque, sync::Arc, task::Context};

use arrow::compute::kernels;
use arrow_array::RecordBatch;
use datafusion_common::DataFusionError;
use datafusion_physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, ready};
use lance_core::error::DataFusionResult;
use lance_core::{Error, Result, utils::memory::MemoryReserver};
use lance_io::stream::chunker;

use crate::resources::{flat_batch_allowance, retain_batch_owner};

mod accounted;

use accounted::StrictBatchResources;

/// Insert a batch boundary every `max_chunk_size` rows without combining batches.
///
/// See [`lance_io::stream::chunker::break_stream`] for an example.
pub fn break_stream(
    stream: SendableRecordBatchStream,
    max_chunk_size: usize,
) -> Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>> {
    chunker::break_stream(stream.map_err(Into::into).boxed(), max_chunk_size)
}

/// Yield chunks of `chunk_size` rows, slicing input batches without copying
/// their buffers. The final chunk may have fewer rows.
///
/// See [`lance_io::stream::chunker::chunk_stream`] for an example.
pub fn chunk_stream(
    stream: SendableRecordBatchStream,
    chunk_size: usize,
) -> Pin<Box<dyn Stream<Item = Result<Vec<RecordBatch>>> + Send>> {
    chunker::chunk_stream(stream.map_err(Into::into).boxed(), chunk_size)
}

/// Preserve input batch boundaries while inserting the requested row boundaries.
///
/// The requested sizes must describe the complete input. Unlike
/// [`chunk_stream_with_sizes`], this does not combine adjacent input batches. It
/// only slices a batch when it crosses a requested boundary.
///
/// # Example
///
/// ```
/// # use datafusion_physical_plan::SendableRecordBatchStream;
/// # use lance_datafusion_exec::chunker::break_stream_with_sizes;
/// # fn split_stream(stream: SendableRecordBatchStream) {
/// let batches = break_stream_with_sizes(stream, vec![512, 512, 256]);
/// # drop(batches);
/// # }
/// ```
pub fn break_stream_with_sizes<I>(
    stream: SendableRecordBatchStream,
    output_sizes: I,
) -> Pin<Box<dyn Stream<Item = Result<Vec<RecordBatch>>> + Send>>
where
    I: IntoIterator<Item = usize>,
    I::IntoIter: Send + 'static,
{
    chunker::break_stream_with_sizes(stream.map_err(Into::into).boxed(), output_sizes)
}

/// Given a stream of record batches, yield chunks with the requested row counts.
///
/// The requested sizes must describe the complete input. An error is returned if
/// the input ends early, contains additional rows, or a requested size is zero.
/// Sizes are consumed lazily as chunks are requested.
///
/// # Example
///
/// ```
/// # use datafusion_physical_plan::SendableRecordBatchStream;
/// # use lance_datafusion_exec::chunker::chunk_stream_with_sizes;
/// # fn split_stream(stream: SendableRecordBatchStream) {
/// let chunks = chunk_stream_with_sizes(stream, vec![512, 512, 256]);
/// # drop(chunks);
/// # }
/// ```
pub fn chunk_stream_with_sizes<I>(
    stream: SendableRecordBatchStream,
    output_sizes: I,
) -> Pin<Box<dyn Stream<Item = Result<Vec<RecordBatch>>> + Send>>
where
    I: IntoIterator<Item = usize>,
    I::IntoIter: Send + 'static,
{
    chunker::chunk_stream_with_sizes(stream.map_err(Into::into).boxed(), output_sizes)
}

/// Given a stream of record batches, this will yield batches of a fixed size.
///
/// This stream _will_ combine record batches and so it can be fairly expensive as it will
/// likely force a copy of incoming data.  However, it can be useful when users require
/// precise batch sizing.
pub fn chunk_concat_stream(
    stream: SendableRecordBatchStream,
    chunk_size: usize,
) -> SendableRecordBatchStream {
    let schema = stream.schema();
    let schema_copy = schema.clone();
    let chunked = chunk_stream(stream, chunk_size);
    let chunk_concat = chunked
        .and_then(move |batches| {
            std::future::ready(
                // chunk_stream is zero-copy and so it gives us pieces of batches.  However, the btree
                // index needs 1 batch-per-page and so we concatenate here.
                kernels::concat::concat_batches(&schema, batches.iter()).map_err(|e| e.into()),
            )
        })
        .map_err(DataFusionError::from)
        .boxed();
    Box::pin(RecordBatchStreamAdapter::new(schema_copy, chunk_concat))
}

/// Given a stream of record batches, this will yield batches of a fixed size.
///
/// This stream _will_ combine record batches and so it can be fairly expensive as it will
/// likely force a copy of all incoming data.  However, it can be useful when users require
/// precise batch sizing.
pub struct StrictBatchSizeStream<S> {
    inner: Option<S>,
    batch_size: usize,
    buffered: VecDeque<RecordBatch>,
    buffered_rows: usize,
    resources: Option<StrictBatchResources>,
}

impl<S: Stream<Item = DataFusionResult<RecordBatch>> + Unpin> StrictBatchSizeStream<S> {
    pub fn new(inner: S, batch_size: usize) -> Self {
        Self {
            inner: Some(inner),
            batch_size,
            buffered: VecDeque::new(),
            buffered_rows: 0,
            resources: None,
        }
    }

    /// Use the operation pool for input directory storage and copies. Input
    /// backing must already be owned by an upstream producer in this mode.
    pub(crate) fn with_memory_reserver(mut self, pool: Arc<dyn MemoryReserver>) -> Result<Self> {
        let bytes = size_of::<Self>()
            .checked_add(256)
            .ok_or_else(|| Error::invalid_input("strict batch stream size overflow"))?;
        let control = pool.reserve("strict batch stream control", bytes)?;
        self.resources = Some(StrictBatchResources {
            pool,
            directory: None,
            concat: None,
            _stream_control: control,
        });
        Ok(self)
    }

    fn push_buffered(&mut self, batch: RecordBatch) -> Result<()> {
        if let Some(resources) = &mut self.resources
            && self.buffered.len() == self.buffered.capacity()
        {
            let capacity = self
                .buffered
                .capacity()
                .max(1)
                .checked_mul(2)
                .ok_or_else(|| Error::invalid_input("strict batch directory capacity overflow"))?;
            let bytes = capacity
                .checked_mul(size_of::<RecordBatch>())
                .ok_or_else(|| Error::invalid_input("strict batch directory size overflow"))?;
            let directory = resources
                .pool
                .reserve("strict batch input directory", bytes)?;
            let mut replacement = VecDeque::with_capacity(capacity);
            replacement.append(&mut self.buffered);
            self.buffered = replacement;
            resources.directory = Some(directory);
        }
        self.buffered_rows = self
            .buffered_rows
            .checked_add(batch.num_rows())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "strict batch buffered row count overflow: buffered_rows={}, input_rows={}",
                    self.buffered_rows,
                    batch.num_rows()
                ))
            })?;
        self.buffered.push_back(batch);
        Ok(())
    }

    fn split_front(&mut self, num_rows: usize) -> Result<RecordBatch> {
        let batch = self.buffered.pop_front().ok_or_else(|| {
            Error::internal(format!(
                "strict batch stream is missing {num_rows} buffered rows"
            ))
        })?;
        if batch.num_rows() <= num_rows {
            Ok(batch)
        } else {
            let owner = self
                .resources
                .as_ref()
                .map(|resources| {
                    let bytes = flat_batch_allowance(batch.schema_ref(), 0)?
                        .checked_mul(2)
                        .ok_or_else(|| {
                            Error::invalid_input("strict batch slice allowance overflow")
                        })?;
                    let control = resources
                        .pool
                        .reserve("strict batch slice control", bytes)?;
                    Ok::<Arc<dyn std::any::Any + Send + Sync>, Error>(Arc::new(control))
                })
                .transpose()?;
            let mut prefix = batch.slice(0, num_rows);
            let mut suffix = batch.slice(num_rows, batch.num_rows() - num_rows);
            if let Some(owner) = owner {
                prefix = retain_batch_owner(prefix, owner.clone())?;
                suffix = retain_batch_owner(suffix, owner)?;
            }
            self.buffered.push_front(suffix);
            Ok(prefix)
        }
    }

    fn take_buffered(&mut self, num_rows: usize) -> Result<Option<RecordBatch>> {
        self.buffered_rows = self.buffered_rows.checked_sub(num_rows).ok_or_else(|| {
            Error::internal(format!(
                "strict batch cannot take {num_rows} rows from {} buffered rows",
                self.buffered_rows
            ))
        })?;
        if self
            .buffered
            .front()
            .is_some_and(|batch| batch.num_rows() >= num_rows)
        {
            return self.split_front(num_rows).map(Some);
        }
        let count = self.buffered.len();
        let directory = self
            .resources
            .as_ref()
            .map(|resources| {
                let bytes = count
                    .checked_mul(size_of::<RecordBatch>())
                    .and_then(|n| n.checked_add(256))
                    .ok_or_else(|| {
                        Error::invalid_input("strict batch copy directory size overflow")
                    })?;
                resources.pool.reserve("strict batch copy directory", bytes)
            })
            .transpose()?;
        let mut batches = Vec::with_capacity(count);
        let mut rows_remaining = num_rows;
        while rows_remaining > 0 {
            let batch = self.split_front(rows_remaining)?;
            rows_remaining -= batch.num_rows();
            batches.push(batch);
        }
        let schema = batches
            .first()
            .ok_or_else(|| Error::internal("strict batch output is empty"))?
            .schema();
        let Some(resources) = &mut self.resources else {
            return Ok(Some(arrow::compute::concat_batches(&schema, &batches)?));
        };
        resources.start_concat(schema, batches, directory, num_rows)
    }

    fn clear(&mut self) {
        self.inner = None;
        self.buffered = VecDeque::new();
        self.buffered_rows = 0;
        if let Some(resources) = &mut self.resources {
            resources.concat = None;
            resources.directory = None;
        }
    }
}

/// Internal polling method for strict batch size enforcement.
///
/// # Use Case
/// When precise batch sizing is required (e.g., ML batch processing), this method guarantees
/// output batches exactly match batch_size until final partial batch. Maintains data integrity
/// across splits using row-aware splitting.
///
/// # Example
/// With batch_size=5 and input sequence:
/// - Fragment 1: 7 rows → splits into `[5,2]`
///   (queues 5, carries 2)
/// - Fragment 2: 4 rows → combines carried 2 + 4 = 6
///   splits into `[5,1]`
///
/// - Output batches: `[5]`, `[5]`, `[1]`
impl<S> Stream for StrictBatchSizeStream<S>
where
    S: Stream<Item = DataFusionResult<RecordBatch>> + Unpin,
{
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(resources) = &mut self.resources
                && let Some(concat) = &mut resources.concat
            {
                let result = ready!(concat.poll_unpin(cx));
                resources.concat = None;
                if result.is_err() {
                    self.clear();
                }
                return Poll::Ready(Some(result.map_err(Into::into)));
            }
            if self.buffered_rows >= self.batch_size
                || (self.inner.is_none() && self.buffered_rows > 0)
            {
                let rows = self.buffered_rows.min(self.batch_size);
                match self.take_buffered(rows) {
                    Ok(Some(batch)) => return Poll::Ready(Some(Ok(batch))),
                    Ok(None) => continue,
                    Err(error) => {
                        self.clear();
                        return Poll::Ready(Some(Err(error.into())));
                    }
                }
            }

            let Some(inner) = &mut self.inner else {
                self.clear();
                return Poll::Ready(None);
            };
            match ready!(Pin::new(inner).poll_next(cx)) {
                Some(Ok(batch)) if batch.num_rows() == 0 => continue,
                Some(Ok(batch)) => {
                    if let Err(error) = self.push_buffered(batch) {
                        self.clear();
                        return Poll::Ready(Some(Err(error.into())));
                    }
                }
                Some(Err(error)) => {
                    if self.resources.is_some() {
                        self.clear();
                    }
                    return Poll::Ready(Some(Err(error)));
                }
                None => self.inner = None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::DataFusionMemoryReserver;
    use arrow::datatypes::{Int32Type, Int64Type};
    use arrow_array::record_batch;
    use datafusion_execution::{
        TaskContext,
        memory_pool::{GreedyMemoryPool, MemoryPool},
        runtime_env::RuntimeEnvBuilder,
    };
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use futures::TryStreamExt;
    use lance_datagen::{BatchCount, RowCount, array};

    use crate::datagen::DatafusionDatagenExt;

    #[tokio::test]
    async fn test_chunk_concat_stream() {
        let batches = [10, 5, 13, 0]
            .map(|rows| record_batch!(("value", Int32, (0..rows).collect::<Vec<_>>())).unwrap());
        let schema = batches[0].schema();
        let stream = futures::stream::iter(batches.into_iter().map(Ok));
        let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream));
        let chunked = super::chunk_concat_stream(stream, 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 3);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 10);
        assert_eq!(chunked[2].num_rows(), 8);
    }

    #[tokio::test]
    async fn test_strict_batch_size_stream() {
        let batches = lance_datagen::gen_batch()
            .anon_col(array::step::<Int32Type>())
            .anon_col(array::step::<Int64Type>())
            .into_df_stream(RowCount::from(7), BatchCount::from(10));

        let stream = super::StrictBatchSizeStream::new(batches, 10);

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches.len(), 7);

        for batch in batches {
            assert_eq!(batch.num_rows(), 10);
        }
    }

    #[tokio::test]
    async fn test_strict_batch_size_stream_buffers_tiny_batches() {
        let batches = (0..100).map(|value| {
            arrow_array::record_batch!(("value", Int32, [value]))
                .map_err(datafusion_common::DataFusionError::from)
        });
        let stream = super::StrictBatchSizeStream::new(futures::stream::iter(batches), 100);

        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 100);
        let expected = (0..100).collect::<Vec<i32>>();
        assert_eq!(
            batches[0]["value"]
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .unwrap()
                .values()
                .as_ref(),
            expected.as_slice()
        );
    }
    #[rstest::rstest]
    #[case::split(false)]
    #[case::refusal(true)]
    #[tokio::test]
    async fn test_accounted_strict_batch_shares_aliases(#[case] refuse_concat: bool) {
        let cap = if refuse_concat { 4096 } else { 8192 };
        let pool = Arc::new(GreedyMemoryPool::new(cap));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool.clone())
            .build_arc()
            .unwrap();
        let context =
            DataFusionMemoryReserver::configure(TaskContext::default().with_runtime(runtime))
                .unwrap();
        let adapter = DataFusionMemoryReserver::from_context(&context)
            .unwrap()
            .unwrap();
        let charge = adapter.reserve("test input backing", 1024).unwrap();
        let baseline = pool.reserved();
        let backing = record_batch!(("value", Int32, [1, 2, 3, 4])).unwrap();
        let backing = retain_batch_owner(backing, Arc::new(charge)).unwrap();
        let empty = backing.slice(0, 0);
        if refuse_concat {
            let inputs = [Ok(empty), Ok(backing.slice(0, 2)), Ok(backing.slice(1, 2))];
            drop(backing);
            let outputs = StrictBatchSizeStream::new(futures::stream::iter(inputs), 4)
                .with_memory_reserver(adapter)
                .unwrap();
            let error = outputs.try_collect::<Vec<_>>().await.unwrap_err();
            assert!(
                error.to_string().contains("strict batch concat backing"),
                "{error}"
            );
            assert_eq!(pool.reserved(), baseline - 1024);
        } else {
            let inputs = [Ok(empty), Ok(backing)];
            let mut outputs = StrictBatchSizeStream::new(futures::stream::iter(inputs), 3)
                .with_memory_reserver(adapter)
                .unwrap();
            let first = outputs.try_next().await.unwrap().unwrap();
            let last = outputs.try_next().await.unwrap().unwrap();
            assert!(outputs.try_next().await.unwrap().is_none());
            drop((outputs, first));
            assert_eq!(pool.reserved(), baseline + 2 * (128 + 1024));
            assert_eq!(last.num_rows(), 1);
            drop(last);
            assert_eq!(pool.reserved(), baseline - 1024);
        }
        drop(context);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn test_accounted_strict_batch_drop_during_concat() {
        let pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool.clone())
            .build_arc()
            .unwrap();
        let context =
            DataFusionMemoryReserver::configure(TaskContext::default().with_runtime(runtime))
                .unwrap();
        let adapter = DataFusionMemoryReserver::from_context(&context)
            .unwrap()
            .unwrap();
        let tasks = adapter.task_group().unwrap();
        let baseline = pool.reserved();
        let input = record_batch!(("value", Int32, vec![1; 32 * 1024])).unwrap();
        let charge = adapter.reserve("test input backing", 128 * 1024).unwrap();
        let input = retain_batch_owner(input, Arc::new(charge)).unwrap();
        let mut stream = StrictBatchSizeStream::new(
            futures::stream::iter([Ok(input.clone()), Ok(input)]),
            48 * 1024,
        )
        .with_memory_reserver(adapter)
        .unwrap();
        let mut next = stream.try_next();
        if let Poll::Ready(result) = futures::poll!(&mut next) {
            drop(result.unwrap());
        }
        drop(next);
        drop(stream);
        tasks.wait().await;
        assert_eq!(tasks.peak(), 1);
        assert_eq!(pool.reserved(), baseline);
        drop(context);
        assert_eq!(pool.reserved(), 0);
    }
}
