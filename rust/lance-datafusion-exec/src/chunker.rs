// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::pin::Pin;
use std::task::Poll;
use std::{collections::VecDeque, task::Context};

use arrow::compute::kernels;
use arrow_array::RecordBatch;
use datafusion_common::DataFusionError;
use datafusion_physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{Stream, StreamExt, TryStreamExt, ready};
use lance_core::Result;
use lance_core::error::DataFusionResult;
use lance_io::stream::chunker;

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
    inner: S,
    batch_size: usize,
    buffered: VecDeque<RecordBatch>,
    buffered_rows: usize,
}

impl<S: Stream<Item = DataFusionResult<RecordBatch>> + Unpin> StrictBatchSizeStream<S> {
    pub fn new(inner: S, batch_size: usize) -> Self {
        Self {
            inner,
            batch_size,
            buffered: VecDeque::new(),
            buffered_rows: 0,
        }
    }

    fn take_buffered(&mut self, num_rows: usize) -> DataFusionResult<RecordBatch> {
        let mut batches = Vec::with_capacity(self.buffered.len());
        let mut rows_remaining = num_rows;

        while rows_remaining > 0 {
            let Some(batch) = self.buffered.pop_front() else {
                return Err(DataFusionError::Internal(format!(
                    "StrictBatchSizeStream has fewer than {num_rows} buffered rows"
                )));
            };
            if batch.num_rows() <= rows_remaining {
                rows_remaining -= batch.num_rows();
                batches.push(batch);
            } else {
                batches.push(batch.slice(0, rows_remaining));
                self.buffered
                    .push_front(batch.slice(rows_remaining, batch.num_rows() - rows_remaining));
                rows_remaining = 0;
            }
        }

        self.buffered_rows = self.buffered_rows.checked_sub(num_rows).ok_or_else(|| {
            DataFusionError::Internal(format!(
                "StrictBatchSizeStream cannot take {num_rows} rows from {} buffered rows",
                self.buffered_rows
            ))
        })?;

        if batches.len() == 1 {
            batches.pop().ok_or_else(|| {
                DataFusionError::Internal(
                    "StrictBatchSizeStream produced an empty output batch".to_string(),
                )
            })
        } else {
            let schema = batches.first().map(RecordBatch::schema).ok_or_else(|| {
                DataFusionError::Internal(
                    "StrictBatchSizeStream produced an empty output batch".to_string(),
                )
            })?;
            arrow::compute::concat_batches(&schema, &batches)
                .map_err(|error| DataFusionError::External(Box::new(error)))
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
            if self.buffered_rows >= self.batch_size {
                let batch_size = self.batch_size;
                return Poll::Ready(Some(self.take_buffered(batch_size)));
            }

            match ready!(Pin::new(&mut self.inner).poll_next(cx)) {
                Some(Ok(batch)) => {
                    if batch.num_rows() > 0 {
                        self.buffered_rows = self
                            .buffered_rows
                            .checked_add(batch.num_rows())
                            .ok_or_else(|| {
                                DataFusionError::Internal(format!(
                                    "StrictBatchSizeStream row count overflow while buffering {} rows",
                                    batch.num_rows()
                                ))
                            })?;
                        self.buffered.push_back(batch);
                    }
                }
                Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                None => {
                    return if self.buffered_rows == 0 {
                        Poll::Ready(None)
                    } else {
                        let buffered_rows = self.buffered_rows;
                        Poll::Ready(Some(self.take_buffered(buffered_rows)))
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{Int32Type, Int64Type};
    use arrow_array::record_batch;
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
}
