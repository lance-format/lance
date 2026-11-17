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

// The caller polls the backend directly, so cancellation must return both queue
// credits without relying on an IoTask completion callback. Payload ownership is
// independent: a detached backend still owns and charges its ReservedBuffer.
struct DirectReadCredit {
    queue: Arc<IoQueue>,
    priority: u128,
    num_bytes: u64,
}

impl DirectReadCredit {
    fn try_new(queue: &Arc<IoQueue>, priority: u128, bytes: u64, bypass: bool) -> Option<Self> {
        let mut state = queue.state.lock().unwrap_or_else(PoisonError::into_inner);
        // Never jump an already queued request, even if its priority would allow
        // this request to bypass byte backpressure.
        if state.done_scheduling
            || !state.pending_requests.is_empty()
            || !state.can_admit(priority, bytes, bypass)
        {
            return None;
        }
        state.admit(priority, bytes, bypass);
        let credit = Self {
            queue: queue.clone(),
            priority,
            num_bytes: if state.no_backpressure || bypass {
                0
            } else {
                bytes
            },
        };
        let event = state.scheduler_state_event();
        drop(state);
        emit_scheduler_state_event(event, &queue.stats);
        Some(credit)
    }
}

impl Drop for DirectReadCredit {
    fn drop(&mut self) {
        let event = {
            let mut state = self
                .queue
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            state.iops_avail += 1;
            state.bytes_avail += self.num_bytes as i64;
            state.priorities_in_flight.remove(self.priority);
            state.scheduler_state_event()
        };
        emit_scheduler_state_event(event, &self.queue.stats);
        self.queue.notify.notify_one();
    }
}

fn validate_range(range: &Range<u64>, output: &ReservedBuffer) -> Result<()> {
    if range.end.checked_sub(range.start) != Some(output.capacity() as u64)
        || usize::try_from(range.end).is_err()
    {
        return Err(Error::invalid_input(
            "scheduled range does not match reserved payload or address space",
        ));
    }
    Ok(())
}

impl FileScheduler {
    /// Fill a small group of independent admitted ranges in request order.
    ///
    /// At most 16 nonempty ranges and 256 KiB may share one backend dispatch.
    /// Every range first obtains its own normal scheduler credit without waiting
    /// or jumping queued requests. If that is not possible, release all tentative
    /// credits and use sequential [`Self::read_single_into`] calls. The lite
    /// scheduler also uses that path. No ranges are coalesced or copied.
    /// Cancellation and transport exclusions match the single-read contract.
    ///
    /// ```
    /// # async fn example(file: &lance_io::scheduler::FileScheduler) -> lance_core::Result<()> {
    /// use lance_core::utils::allocation::AllocationBudget;
    /// let budget = AllocationBudget::new(8);
    /// let mut work = budget.try_reserve(8).unwrap();
    /// let requests = vec![(0..4, work.try_allocate(4)?.unwrap()),
    ///                     (16..20, work.try_allocate(4)?.unwrap())];
    /// let outputs = file.read_many_into(requests, 0).await?;
    /// assert_eq!(outputs.len(), 2);
    /// # Ok(()) }
    /// ```
    pub fn read_many_into(
        &self,
        requests: Vec<(Range<u64>, ReservedBuffer)>,
        priority: u64,
    ) -> BoxFuture<'static, Result<Vec<ReservedBuffer>>> {
        let file = self.clone();
        async move {
            for (range, output) in &requests {
                validate_range(range, output)?;
            }
            let total = requests.iter().try_fold(0usize, |sum, (_, output)| {
                sum.checked_add(output.capacity())
            });
            let mut credits = Vec::new();
            if (2..=16).contains(&requests.len())
                && total.is_some_and(|bytes| bytes <= 256 * 1024)
                && requests.iter().all(|(range, _)| !range.is_empty())
                && let IoQueueType::Standard(queue) = &file.root.io_queue
            {
                let full_priority = ((file.base_priority as u128) << 64) + priority as u128;
                for (range, _) in &requests {
                    let Some(credit) = DirectReadCredit::try_new(
                        queue,
                        full_priority,
                        range.end - range.start,
                        file.bypass_backpressure,
                    ) else {
                        break;
                    };
                    credits.push(credit);
                }
            }
            if !requests.is_empty() && credits.len() == requests.len() {
                let identities: Vec<_> = requests
                    .iter()
                    .map(|(_, buffer)| (buffer.as_ref().as_ptr() as usize, buffer.capacity()))
                    .collect();
                for (range, _) in &requests {
                    file.record_owned_read(range);
                }
                IOPS_COUNTER.fetch_add(requests.len() as u64, Ordering::Release);
                let outputs = file
                    .reader
                    .get_ranges_into(
                        requests
                            .into_iter()
                            .map(|(range, buffer)| {
                                (range.start as usize..range.end as usize, buffer)
                            })
                            .collect(),
                    )
                    .await;
                BYTES_READ_COUNTER.fetch_add(
                    identities.iter().map(|(_, len)| *len as u64).sum(),
                    Ordering::Release,
                );
                let outputs = outputs?;
                if outputs.len() != identities.len()
                    || outputs
                        .iter()
                        .zip(identities)
                        .any(|(buffer, (pointer, len))| {
                            buffer.capacity() != len || buffer.as_ref().as_ptr() as usize != pointer
                        })
                {
                    return Err(Error::invalid_input(
                        "scheduled reads must return supplied allocations unchanged and in order",
                    ));
                }
                return Ok(outputs);
            }
            // Never await more credit while retaining only part of a group.
            drop(credits);
            let mut outputs = Vec::with_capacity(requests.len());
            for (range, output) in requests {
                outputs.push(file.read_single_into(range, priority, output).await?);
            }
            Ok(outputs)
        }
        .boxed()
    }

    /// Read into an admitted allocation when the returned future is polled.
    ///
    /// Uses the same priority, IOP limit and byte policy as
    /// [`Self::submit_single_into`]. An uncontended standard scheduler read runs
    /// in the caller's async task; otherwise it joins the existing queue. The
    /// lite scheduler uses its normal submission path. No scheduler credit or
    /// I/O is started before polling, so keeping an unpolled future cannot block
    /// another read. The future still owns the supplied payload allocation.
    ///
    /// Dropping a polled future refunds scheduler credit; a backend that keeps
    /// accessing the buffer must retain its allocation charge until it stops.
    /// The range, allocation, transport and dependent-operation requirements of
    /// [`Self::submit_single_into`] also apply here. Use that method when eager
    /// standard-queue submission is required.
    /// Poll this future on a Tokio runtime, as required by asynchronous readers.
    ///
    /// ```
    /// # async fn example(file: &lance_io::scheduler::FileScheduler) -> lance_core::Result<()> {
    /// use lance_core::utils::allocation::AllocationBudget;
    /// let budget = AllocationBudget::new(4096);
    /// let output = budget.try_reserve(4096).unwrap().allocate()?;
    /// let output = file.read_single_into(0..4096, 0, output).await?;
    /// assert_eq!(output.capacity(), 4096);
    /// # Ok(()) }
    /// ```
    pub fn read_single_into(
        &self,
        range: Range<u64>,
        priority: u64,
        output: ReservedBuffer,
    ) -> BoxFuture<'static, Result<ReservedBuffer>> {
        let file = self.clone();
        async move {
            validate_range(&range, &output)?;
            if range.is_empty() {
                return Ok(output);
            }
            let full_priority = ((file.base_priority as u128) << 64) + priority as u128;
            if let IoQueueType::Standard(queue) = &file.root.io_queue
                && let Some(_credit) = DirectReadCredit::try_new(
                    queue,
                    full_priority,
                    range.end - range.start,
                    file.bypass_backpressure,
                )
            {
                file.record_owned_read(&range);
                let result = read(file.reader.clone(), range.clone(), output).await;
                tracing::trace!(
                    file = file.reader.path().as_ref(),
                    bytes_read = range.end - range.start,
                    requests = 1,
                    range_start = range.start,
                    range_end = range.end,
                    "File I/O completed"
                );
                return result?.into_reserved();
            }
            file.submit_single_into(range, priority, output).await
        }
        .boxed()
    }

    fn record_owned_read(&self, range: &Range<u64>) {
        self.root.stats.record_request(std::slice::from_ref(range));
        if let Some(stats) = &self.extra_stats {
            stats.record_request(std::slice::from_ref(range));
        }
    }

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
        if let Err(error) = validate_range(&range, &output) {
            return futures::future::ready(Err(error)).boxed();
        }
        if range.is_empty() {
            return futures::future::ready(Ok(output)).boxed();
        }
        let priority = ((self.base_priority as u128) << 64) + priority as u128;
        self.record_owned_read(&range);
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
