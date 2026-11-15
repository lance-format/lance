// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep owned payload admission separate from the scheduler's delivery credit.

use super::*;
use futures::future::BoxFuture;

#[derive(Debug)]
pub(super) enum IoBuffer {
    Bytes(Bytes),
    Reserved(ReservedBuffer),
}

impl IoBuffer {
    pub(super) fn into_bytes(self) -> Result<Bytes> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::Reserved(_) => Err(Error::internal("expected ordinary I/O output")),
        }
    }

    fn into_reserved(self) -> Result<ReservedBuffer> {
        match self {
            Self::Reserved(buffer) => Ok(buffer),
            Self::Bytes(_) => Err(Error::internal("expected admitted I/O output")),
        }
    }
}

pub(super) async fn read(
    reader: Arc<dyn Reader>,
    range: Range<u64>,
    output: ReservedBuffer,
) -> Result<IoBuffer> {
    let pointer = output.as_ref().as_ptr() as usize;
    let capacity = output.capacity();
    IOPS_COUNTER.fetch_add(1, Ordering::Release);
    let result = reader
        .get_range_into(range.start as usize..range.end as usize, output)
        .await;
    BYTES_READ_COUNTER.fetch_add(range.end - range.start, Ordering::Release);
    let output = result?;
    if output.capacity() != capacity || output.as_ref().as_ptr() as usize != pointer {
        return Err(Error::invalid_input(
            "scheduled read must return the supplied allocation unchanged",
        ));
    }
    Ok(IoBuffer::Reserved(output))
}

struct CancelOnDrop {
    queue: Arc<IoQueue>,
    cancellation: Arc<RequestCancellation>,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.cancellation.cancel() {
            self.queue.cancel_pending_for_request(&self.cancellation);
        }
    }
}

impl FileScheduler {
    /// Schedule one read into an already admitted exclusive payload allocation.
    ///
    /// Uses this file's priority, backpressure policy, queue and I/O statistics.
    /// Unlike [`Self::submit_request`], this single IOP is neither coalesced nor
    /// split: callers choose the read geometry before admitting its allocation.
    /// The range length must equal the supplied buffer capacity. Empty reads do
    /// not enter the queue. No payload allocation or extra credit is acquired.
    ///
    /// Delivery refunds only scheduler credit. The returned buffer retains its
    /// allocation charge and can be recycled into the caller's workspace. Drop
    /// cancels queued/running work; a backend operation that outlives cancellation
    /// must retain its allocation until access stops, per [`Reader::get_range_into`].
    /// Opaque transport buffers/cache are outside this payload budget. Callers
    /// must reserve the entire dependent operation before scheduling reads.
    ///
    /// ```
    /// # async fn example(file: &lance_io::scheduler::FileScheduler) -> lance_core::Result<()> {
    /// use lance_core::utils::allocation::AllocationBudget;
    /// let budget = AllocationBudget::new(8192);
    /// let mut work = budget.try_reserve(8192).unwrap();
    /// let input = work.try_allocate(4096)?.unwrap();
    /// let input = file.submit_single_into(0..4096, 0, input).await?;
    /// input.recycle(&mut work)?;
    /// # Ok(()) }
    /// ```
    pub fn submit_single_into(
        &self,
        range: Range<u64>,
        priority: u64,
        output: ReservedBuffer,
    ) -> BoxFuture<'static, Result<ReservedBuffer>> {
        if range.end.checked_sub(range.start) != Some(output.capacity() as u64)
            || usize::try_from(range.end).is_err()
        {
            return futures::future::ready(Err(Error::invalid_input(
                "scheduled range does not match reserved payload or address space",
            )))
            .boxed();
        }
        if range.is_empty() {
            return futures::future::ready(Ok(output)).boxed();
        }
        let priority = ((self.base_priority as u128) << 64) + priority as u128;
        self.root.stats.record_request(std::slice::from_ref(&range));
        if let Some(stats) = &self.extra_stats {
            stats.record_request(std::slice::from_ref(&range));
        }
        let root = self.root.clone();
        let pending: BoxFuture<'static, Result<ReservedBuffer>> = match &root.io_queue {
            IoQueueType::Standard(queue) => {
                let cancellation = Arc::new(RequestCancellation::new());
                let guard = CancelOnDrop {
                    queue: queue.clone(),
                    cancellation: cancellation.clone(),
                };
                let (tx, rx) = oneshot::channel::<Response<ReservedBuffer>>();
                let callback_queue = queue.clone();
                let num_bytes = range.end - range.start;
                let bypass = self.bypass_backpressure;
                let account_bytes = !bypass && root.io_buffer_size != 0;
                queue.push(IoTask {
                    reader: self.reader.clone(),
                    to_read: range,
                    buffer: Some(output),
                    priority,
                    bypass_backpressure: bypass,
                    request_cancellation: Some(cancellation),
                    when_done: Box::new(move |result, was_admitted| {
                        if was_admitted {
                            callback_queue.on_iop_complete();
                        }
                        let response = Response {
                            data: Some(result.and_then(IoBuffer::into_reserved)),
                            io_queue: callback_queue,
                            priority,
                            num_reqs: usize::from(was_admitted),
                            num_bytes: if was_admitted && account_bytes {
                                num_bytes
                            } else {
                                0
                            },
                        };
                        // Dropping an undelivered response refunds both credits.
                        let _ = tx.send(response);
                    }),
                });
                async move {
                    let _guard = guard;
                    let mut response = rx
                        .await
                        .map_err(|_| Error::io("scheduled owned read ended without a response"))?;
                    response.data.take().ok_or_else(|| {
                        Error::internal("scheduled owned response already consumed")
                    })?
                }
                .boxed()
            }
            IoQueueType::Lite(queue) => {
                let reader = self.reader.clone();
                let requested = range.clone();
                match queue.clone().submit(
                    range,
                    priority,
                    Box::new(move || read(reader, requested, output).boxed()),
                    self.bypass_backpressure,
                ) {
                    Ok(task) => async move { task.await?.into_reserved() }.boxed(),
                    Err(error) => futures::future::ready(Err(error)).boxed(),
                }
            }
        };
        async move {
            // Keep the scheduler alive through completion, including when the
            // caller moves or drops its FileScheduler after submission.
            let _root = root;
            pending.await
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests;
