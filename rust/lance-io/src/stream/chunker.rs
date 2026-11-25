// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::VecDeque;
use std::pin::Pin;

use arrow_array::RecordBatch;
use futures::{Stream, StreamExt, TryStreamExt};
use lance_core::Result;

use super::BatchStream;

/// Wraps a [`BatchStream`] into a stream of RecordBatch chunks of
/// a given size.  This slices but does not copy any buffers.
struct BatchReaderChunker {
    /// The inner stream
    inner: BatchStream,
    /// The batches that have been read from the inner stream but not yet fully yielded
    buffered: VecDeque<RecordBatch>,
    /// The number of rows to yield in each chunk
    output_size: usize,
    /// The position within the first batch in the buffer to start yielding from
    i: usize,
}

impl BatchReaderChunker {
    fn new(inner: BatchStream, output_size: usize) -> Self {
        Self {
            inner,
            buffered: VecDeque::new(),
            output_size,
            i: 0,
        }
    }

    fn buffered_len(&self) -> usize {
        let buffer_total: usize = self.buffered.iter().map(|batch| batch.num_rows()).sum();
        buffer_total - self.i
    }

    async fn fill_buffer(&mut self, output_size: usize) -> Result<()> {
        while self.buffered_len() < output_size {
            match self.inner.next().await {
                Some(Ok(batch)) => self.buffered.push_back(batch),
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }
        Ok(())
    }

    async fn next(&mut self) -> Option<Result<Vec<RecordBatch>>> {
        self.next_sized(self.output_size).await
    }

    async fn next_sized(&mut self, output_size: usize) -> Option<Result<Vec<RecordBatch>>> {
        match self.fill_buffer(output_size).await {
            Ok(_) => {}
            Err(e) => return Some(Err(e)),
        };

        let mut batches = Vec::new();

        let mut rows_collected = 0;

        while rows_collected < output_size {
            if let Some(batch) = self.buffered.pop_front() {
                // Skip empty batch
                if batch.num_rows() == 0 {
                    continue;
                }

                let rows_remaining_in_batch = batch.num_rows() - self.i;
                let rows_to_take =
                    std::cmp::min(rows_remaining_in_batch, output_size - rows_collected);

                if rows_to_take == rows_remaining_in_batch {
                    // We're taking the whole batch, so we can just move it
                    let batch = if self.i == 0 {
                        batch
                    } else {
                        // We are taking the remainder of the batch, so we need to slice it
                        batch.slice(self.i, rows_to_take)
                    };
                    batches.push(batch);
                    self.i = 0;
                } else {
                    // Slice the batch without copying its buffers.
                    batches.push(batch.slice(self.i, rows_to_take));
                    // And then we need to push the remainder back onto the front of the queue
                    self.i += rows_to_take;
                    self.buffered.push_front(batch);
                }

                rows_collected += rows_to_take;
            } else {
                break;
            }
        }

        if batches.is_empty() {
            None
        } else {
            Some(Ok(batches))
        }
    }

    async fn next_at_most(&mut self, output_size: usize) -> Option<Result<Vec<RecordBatch>>> {
        loop {
            let batch = match self.buffered.pop_front() {
                Some(batch) => batch,
                None => match self.inner.next().await {
                    Some(Ok(batch)) => batch,
                    Some(Err(error)) => return Some(Err(error)),
                    None => return None,
                },
            };

            if batch.num_rows() == 0 {
                continue;
            }

            let rows_remaining_in_batch = batch.num_rows() - self.i;
            let rows_to_take = rows_remaining_in_batch.min(output_size);
            if rows_to_take == rows_remaining_in_batch {
                let batch = if self.i == 0 {
                    batch
                } else {
                    batch.slice(self.i, rows_to_take)
                };
                self.i = 0;
                return Some(Ok(vec![batch]));
            }

            let output = batch.slice(self.i, rows_to_take);
            self.i += rows_to_take;
            self.buffered.push_front(batch);
            return Some(Ok(vec![output]));
        }
    }
}

struct VariableBatchReaderChunker<I> {
    chunker: BatchReaderChunker,
    output_sizes: I,
    is_done: bool,
}

struct VariableBreakStreamState<I> {
    chunker: BatchReaderChunker,
    output_sizes: I,
    rows_remaining: Option<usize>,
    is_done: bool,
}

struct BreakStreamState {
    max_rows: usize,
    rows_seen: usize,
    batch: Option<RecordBatch>,
}

impl BreakStreamState {
    fn next(mut self) -> Option<(Result<RecordBatch>, Self)> {
        let batch = self.batch.take()?;
        let batch_rows = batch.num_rows();
        if batch_rows == 0 {
            return None;
        }
        let rows_to_boundary = self.max_rows - self.rows_seen;
        if batch_rows <= rows_to_boundary {
            return Some((Ok(batch), self));
        }

        let next = batch.slice(0, rows_to_boundary);
        self.batch = Some(batch.slice(rows_to_boundary, batch_rows - rows_to_boundary));
        self.rows_seen = 0;
        Some((Ok(next), self))
    }
}

/// Insert a batch boundary every `max_chunk_size` rows without combining batches.
///
/// The chunk size must be positive. Empty batches are skipped.
///
/// # Example
///
/// ```
/// # use lance_io::stream::{SendableRecordBatchStream, chunker::break_stream};
/// # fn split_stream(input: SendableRecordBatchStream) {
/// // With input batch lengths [3, 5, 8, 3, 5], the output lengths are [3, 5, 2, 6, 3, 1, 4].
/// let batches = break_stream(input, 10);
/// # drop(batches);
/// # }
/// ```
pub fn break_stream(stream: BatchStream, max_chunk_size: usize) -> BatchStream {
    let mut rows_already_seen = 0;
    stream
        .map_ok(move |batch| {
            let rows_seen = rows_already_seen;
            rows_already_seen = (rows_seen + batch.num_rows()) % max_chunk_size;
            let state = BreakStreamState {
                max_rows: max_chunk_size,
                rows_seen,
                batch: Some(batch),
            };

            futures::stream::unfold(state, move |state| std::future::ready(state.next()))
                .fuse()
                .boxed()
        })
        .try_flatten()
        .boxed()
}

/// Given a stream of record batches, this will yield batches of a fixed size.
///
/// In order to avoid copying data the batches will be converted into a stream of
/// `Vec<RecordBatch>` where each item is a `Vec` of batches whose total size is
/// `chunk_size`.
///
/// The final chunk may have fewer than `chunk_size` rows. Empty batches are
/// skipped.
///
/// # Example
///
/// ```
/// # use lance_io::stream::{SendableRecordBatchStream, chunker::chunk_stream};
/// # fn split_stream(input: SendableRecordBatchStream) {
/// let chunks = chunk_stream(input, 1024);
/// # drop(chunks);
/// # }
/// ```
pub fn chunk_stream(
    stream: BatchStream,
    chunk_size: usize,
) -> Pin<Box<dyn Stream<Item = Result<Vec<RecordBatch>>> + Send>> {
    let chunker = BatchReaderChunker::new(stream, chunk_size);
    futures::stream::unfold(chunker, |mut chunker| async move {
        match chunker.next().await {
            Some(Ok(batches)) => Some((Ok(batches), chunker)),
            Some(Err(e)) => Some((Err(e), chunker)),
            None => None,
        }
    })
    .fuse()
    .boxed()
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
/// # use lance_io::stream::SendableRecordBatchStream;
/// # use lance_io::stream::chunker::break_stream_with_sizes;
/// # fn split_stream(stream: SendableRecordBatchStream) {
/// let batches = break_stream_with_sizes(stream, vec![512, 512, 256]);
/// # drop(batches);
/// # }
/// ```
pub fn break_stream_with_sizes<I>(
    stream: BatchStream,
    output_sizes: I,
) -> Pin<Box<dyn Stream<Item = Result<Vec<RecordBatch>>> + Send>>
where
    I: IntoIterator<Item = usize>,
    I::IntoIter: Send + 'static,
{
    let state = VariableBreakStreamState {
        chunker: BatchReaderChunker::new(stream, 1),
        output_sizes: output_sizes.into_iter(),
        rows_remaining: None,
        is_done: false,
    };
    futures::stream::unfold(state, |mut state| async move {
        if state.is_done {
            return None;
        }

        if state.rows_remaining.is_none() {
            let Some(output_size) = state.output_sizes.next() else {
                return match state.chunker.next_at_most(1).await {
                    None => None,
                    Some(Ok(_)) => {
                        state.is_done = true;
                        Some((
                            Err(lance_core::Error::invalid_input(
                                "Input contained more rows than the requested chunk sizes",
                            )),
                            state,
                        ))
                    }
                    Some(Err(error)) => {
                        state.is_done = true;
                        Some((Err(error), state))
                    }
                };
            };
            if output_size == 0 {
                state.is_done = true;
                return Some((
                    Err(lance_core::Error::invalid_input(
                        "Requested chunk sizes must be greater than zero",
                    )),
                    state,
                ));
            }
            state.rows_remaining = Some(output_size);
        }

        let Some(rows_remaining) = state.rows_remaining else {
            state.is_done = true;
            return Some((
                Err(lance_core::Error::internal(
                    "Requested chunk boundary was not initialized",
                )),
                state,
            ));
        };
        match state.chunker.next_at_most(rows_remaining).await {
            Some(Ok(batches)) => {
                let actual_size = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
                let Some(rows_remaining) = rows_remaining.checked_sub(actual_size) else {
                    state.is_done = true;
                    return Some((
                        Err(lance_core::Error::internal(
                            "A boundary-preserving chunk exceeded its requested row count",
                        )),
                        state,
                    ));
                };
                state.rows_remaining = (rows_remaining > 0).then_some(rows_remaining);
                Some((Ok(batches), state))
            }
            Some(Err(error)) => {
                state.is_done = true;
                Some((Err(error), state))
            }
            None => {
                state.is_done = true;
                Some((
                    Err(lance_core::Error::invalid_input(format!(
                        "Input ended with {rows_remaining} rows remaining in a requested chunk"
                    ))),
                    state,
                ))
            }
        }
    })
    .boxed()
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
/// # use lance_io::stream::SendableRecordBatchStream;
/// # use lance_io::stream::chunker::chunk_stream_with_sizes;
/// # fn split_stream(stream: SendableRecordBatchStream) {
/// let chunks = chunk_stream_with_sizes(stream, vec![512, 512, 256]);
/// # drop(chunks);
/// # }
/// ```
pub fn chunk_stream_with_sizes<I>(
    stream: BatchStream,
    output_sizes: I,
) -> Pin<Box<dyn Stream<Item = Result<Vec<RecordBatch>>> + Send>>
where
    I: IntoIterator<Item = usize>,
    I::IntoIter: Send + 'static,
{
    let state = VariableBatchReaderChunker {
        chunker: BatchReaderChunker::new(stream, 1),
        output_sizes: output_sizes.into_iter(),
        is_done: false,
    };
    futures::stream::unfold(state, |mut state| async move {
        if state.is_done {
            return None;
        }

        let Some(output_size) = state.output_sizes.next() else {
            return match state.chunker.next_sized(1).await {
                None => None,
                Some(Ok(_)) => {
                    state.is_done = true;
                    Some((
                        Err(lance_core::Error::invalid_input(
                            "Input contained more rows than the requested chunk sizes",
                        )),
                        state,
                    ))
                }
                Some(Err(error)) => {
                    state.is_done = true;
                    Some((Err(error), state))
                }
            };
        };

        if output_size == 0 {
            state.is_done = true;
            return Some((
                Err(lance_core::Error::invalid_input(
                    "Requested chunk sizes must be greater than zero",
                )),
                state,
            ));
        }

        match state.chunker.next_sized(output_size).await {
            Some(Ok(batches)) => {
                let actual_size = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
                if actual_size == output_size {
                    Some((Ok(batches), state))
                } else {
                    state.is_done = true;
                    Some((
                        Err(lance_core::Error::invalid_input(format!(
                            "Input ended after {actual_size} rows while filling a requested {output_size}-row chunk"
                        ))),
                        state,
                    ))
                }
            }
            Some(Err(error)) => {
                state.is_done = true;
                Some((Err(error), state))
            }
            None => {
                state.is_done = true;
                Some((
                    Err(lance_core::Error::invalid_input(format!(
                        "Input ended before a requested {output_size}-row chunk could be filled"
                    ))),
                    state,
                ))
            }
        }
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use arrow_array::record_batch;
    use futures::{StreamExt, TryStreamExt};

    use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};

    #[tokio::test]
    async fn test_chunkers() {
        let batches = [10, 5, 13, 0]
            .map(|rows| record_batch!(("value", Int32, (0..rows).collect::<Vec<_>>())).unwrap());
        let schema = batches[0].schema();
        let make_stream = || -> SendableRecordBatchStream {
            let stream = futures::stream::iter(batches.clone().into_iter().map(Ok));
            Box::pin(RecordBatchStreamAdapter::new(schema.clone(), stream))
        };

        let chunked = super::chunk_stream(make_stream(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 3);
        assert_eq!(chunked[0].len(), 1);
        assert_eq!(chunked[0][0].num_rows(), 10);
        assert_eq!(chunked[1].len(), 2);
        assert_eq!(chunked[1][0].num_rows(), 5);
        assert_eq!(chunked[1][1].num_rows(), 5);
        assert_eq!(chunked[2].len(), 1);
        assert_eq!(chunked[2][0].num_rows(), 8);

        let sizes_consumed = Arc::new(AtomicUsize::new(0));
        let requested_sizes = [9, 10, 9].into_iter().inspect({
            let sizes_consumed = sizes_consumed.clone();
            move |_| {
                sizes_consumed.fetch_add(1, Ordering::SeqCst);
            }
        });
        let mut chunked = super::chunk_stream_with_sizes(make_stream(), requested_sizes);
        assert_eq!(sizes_consumed.load(Ordering::SeqCst), 0);
        let first_chunk = chunked.next().await.unwrap().unwrap();
        assert_eq!(sizes_consumed.load(Ordering::SeqCst), 1);
        let mut chunked = chunked.try_collect::<Vec<_>>().await.unwrap();
        chunked.insert(0, first_chunk);
        assert_eq!(sizes_consumed.load(Ordering::SeqCst), 3);
        assert_eq!(
            chunked
                .iter()
                .map(|batches| batches.iter().map(|batch| batch.num_rows()).sum::<usize>())
                .collect::<Vec<_>>(),
            vec![9, 10, 9]
        );

        let error = super::chunk_stream_with_sizes(make_stream(), vec![10, 17])
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("more rows than the requested chunk sizes")
        );

        let error = super::chunk_stream_with_sizes(make_stream(), vec![10, 19])
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("ended after 18 rows"));

        let sizes_consumed = Arc::new(AtomicUsize::new(0));
        let requested_sizes = [9, 10, 9].into_iter().inspect({
            let sizes_consumed = sizes_consumed.clone();
            move |_| {
                sizes_consumed.fetch_add(1, Ordering::SeqCst);
            }
        });
        let mut broken = super::break_stream_with_sizes(make_stream(), requested_sizes);
        assert_eq!(sizes_consumed.load(Ordering::SeqCst), 0);
        let first_batch = broken.next().await.unwrap().unwrap();
        assert_eq!(sizes_consumed.load(Ordering::SeqCst), 1);
        let mut broken = broken.try_collect::<Vec<_>>().await.unwrap();
        broken.insert(0, first_batch);
        assert_eq!(sizes_consumed.load(Ordering::SeqCst), 3);
        assert_eq!(
            broken
                .iter()
                .map(|batches| batches.iter().map(|batch| batch.num_rows()).sum::<usize>())
                .collect::<Vec<_>>(),
            vec![9, 1, 5, 4, 9]
        );

        let error = super::break_stream_with_sizes(make_stream(), vec![27])
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("more rows than the requested chunk sizes")
        );

        let error = super::break_stream_with_sizes(make_stream(), vec![29])
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("1 rows remaining"));

        let chunked = super::break_stream(make_stream(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(chunked.len(), 4);
        assert_eq!(chunked[0].num_rows(), 10);
        assert_eq!(chunked[1].num_rows(), 5);
        assert_eq!(chunked[2].num_rows(), 5);
        assert_eq!(chunked[3].num_rows(), 8);
    }
}
