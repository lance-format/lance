// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A process-wide memory budget for the data that queries read.
//!
//! A reader reserves the bytes a unit of work loads before requesting it and
//! releases them once the unit is decoded. A reader's units are admitted in
//! order, each once it fits under the limit; this is what bounds read-ahead.
//! Read-ahead leaves the last eighth of the limit to readers' first units, so
//! small reads are not queued behind read-ahead. A
//! reader that holds nothing is admitted even when its unit does not fit, by
//! taking the overdraft, which lets one query at a time exceed the limit by
//! what it needs to make progress. All readers of that query may use the overdraft,
//! because a downstream reader of a query cannot wait for its own upstream to
//! drain.
//!
//! This is deadlock-free: a reader only waits while it holds reservations that
//! drain without anyone being admitted, and the query holding the overdraft is
//! never refused, so it finishes and frees the overdraft for the next query.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

/// Identifies the query a reader belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryId(usize);

impl QueryId {
    /// The query identified by `owner`, an object that lives exactly as long
    /// as one execution (for example its task context).
    pub fn of<T: ?Sized>(owner: &Arc<T>) -> Self {
        Self(Arc::as_ptr(owner) as *const () as usize)
    }

    /// A fresh identity for a read that is not part of a plan execution.
    pub fn unique() -> Self {
        // Odd values never collide with the (aligned) addresses used by `of`.
        static NEXT: AtomicUsize = AtomicUsize::new(1);
        Self(NEXT.fetch_add(2, Ordering::Relaxed))
    }
}

type ResidentBytesFn = Box<dyn Fn() -> u64 + Send + Sync>;

/// Read-ahead may use all but `1 / FIRST_UNIT_SHARE` of the limit.
const FIRST_UNIT_SHARE: u64 = 8;

/// A memory limit shared by every reader of a session.
pub struct ReadMemoryBudget {
    limit: u64,
    /// Memory that the budget does not reserve but that counts against the
    /// limit, such as the index cache.
    resident_bytes: Option<ResidentBytesFn>,
    state: Mutex<BudgetState>,
    notify: Notify,
}

impl std::fmt::Debug for ReadMemoryBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadMemoryBudget")
            .field("limit", &self.limit)
            .field("reserved", &self.state.lock().unwrap().reserved)
            .finish()
    }
}

#[derive(Default)]
struct BudgetState {
    reserved: u64,
    /// The query holding the overdraft and the number of its reservations it covers.
    overdraft: Option<(QueryId, usize)>,
    peak_reserved: u64,
}

impl ReadMemoryBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            resident_bytes: None,
            state: Mutex::default(),
            notify: Notify::new(),
        }
    }

    /// Counts the memory reported by `resident_bytes` against the limit.
    pub fn with_resident_bytes(
        mut self,
        resident_bytes: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Self {
        self.resident_bytes = Some(Box::new(resident_bytes));
        self
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// Bytes currently reserved by readers.
    pub fn reserved_bytes(&self) -> u64 {
        self.state.lock().unwrap().reserved
    }

    /// The most bytes readers have held at once.
    pub fn peak_reserved_bytes(&self) -> u64 {
        self.state.lock().unwrap().peak_reserved
    }

    /// A reader of `query`. Each independent stream of reads (for example one
    /// scan) needs its own reader.
    pub fn reader(self: &Arc<Self>, query: QueryId) -> BudgetReader {
        BudgetReader(Arc::new(ReaderState {
            budget: self.clone(),
            query,
            queue: Mutex::default(),
        }))
    }
}

struct ReaderState {
    budget: Arc<ReadMemoryBudget>,
    query: QueryId,
    // Locked after the budget's state.
    queue: Mutex<ReaderQueue>,
}

/// A reader's own admissions, served strictly in order: if a later unit were
/// admitted first, the reader would hold memory it cannot release until the
/// earlier unit is admitted, and could wait on itself forever.
#[derive(Default)]
struct ReaderQueue {
    held: u64,
    next_ticket: u64,
    serving_ticket: u64,
    /// Tickets that left the queue before their turn.
    skipped_tickets: HashSet<u64>,
}

impl ReaderQueue {
    fn advance_past(&mut self, ticket: u64) {
        if ticket == self.serving_ticket {
            self.serving_ticket += 1;
        } else if ticket > self.serving_ticket {
            self.skipped_tickets.insert(ticket);
        }
        while self.skipped_tickets.remove(&self.serving_ticket) {
            self.serving_ticket += 1;
        }
    }
}

/// One stream of reads drawing on a [`ReadMemoryBudget`].
#[derive(Clone)]
pub struct BudgetReader(Arc<ReaderState>);

impl std::fmt::Debug for BudgetReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetReader")
            .field("query", &self.0.query)
            .finish_non_exhaustive()
    }
}

impl BudgetReader {
    /// The limit of the budget this reader draws on.
    pub fn limit(&self) -> u64 {
        self.0.budget.limit
    }

    /// Another reader for the same query, with its own line. Use one per
    /// independently consumed stream: a reader's units wait for the ones
    /// before them, so streams sharing a reader could wait for each other.
    pub fn sibling(&self) -> Self {
        self.0.budget.reader(self.0.query)
    }

    /// Reserves `bytes` the reader already holds, without waiting: memory that
    /// exists before it could be admitted, such as a decoded batch waiting to
    /// be emitted. It counts against the limit like any reservation.
    pub fn charge(&self, bytes: u64) -> Reservation {
        let mut state = self.0.budget.state.lock().unwrap();
        state.reserved += bytes;
        state.peak_reserved = state.peak_reserved.max(state.reserved);
        self.0.queue.lock().unwrap().held += bytes;
        Reservation {
            reader: self.clone(),
            bytes: AtomicU64::new(bytes),
            overdraft: false,
        }
    }

    /// Takes the reader's next place in line. A reader's units are admitted in
    /// the order their admissions are created, so create them in the order the
    /// reader consumes the units.
    pub fn admission(&self) -> Admission {
        let mut queue = self.0.queue.lock().unwrap();
        let ticket = queue.next_ticket;
        queue.next_ticket += 1;
        Admission {
            reader: self.clone(),
            ticket,
            queued: true,
        }
    }
}

/// A queued request for a unit of work's memory.
pub struct Admission {
    reader: BudgetReader,
    ticket: u64,
    queued: bool,
}

impl Admission {
    /// Waits until `bytes` may be loaded and reserves them.
    pub async fn reserve(mut self, bytes: u64) -> Reservation {
        let reader = self.reader.0.clone();
        let budget = &reader.budget;
        loop {
            // Created before checking so a release between the check and the
            // wait still wakes us.
            let notified = budget.notify.notified();
            let resident = budget.resident_bytes.as_ref().map_or(0, |f| f());
            {
                let mut state = budget.state.lock().unwrap();
                let mut queue = reader.queue.lock().unwrap();
                let total = resident
                    .saturating_add(state.reserved)
                    .saturating_add(bytes);
                // Read-ahead stops short of the last part of the limit, which
                // only a reader's first unit may use, so a small read is not
                // queued behind other readers' read-ahead.
                let limit = if queue.held == 0 {
                    budget.limit
                } else {
                    budget.limit - budget.limit / FIRST_UNIT_SHARE
                };
                let grant = if self.ticket != queue.serving_ticket {
                    None
                } else if total <= limit {
                    Some(false)
                } else if queue.held == 0 {
                    // The reader holds nothing, so it can only make progress by
                    // exceeding the limit: take the overdraft if it is free or
                    // already held by this query.
                    match state.overdraft {
                        None => Some(true),
                        Some((query, _)) if query == reader.query => Some(true),
                        Some(_) => None,
                    }
                } else {
                    None
                };
                if let Some(overdraft) = grant {
                    state.reserved += bytes;
                    state.peak_reserved = state.peak_reserved.max(state.reserved);
                    queue.held += bytes;
                    if overdraft {
                        let covered = state.overdraft.map_or(0, |(_, covered)| covered);
                        state.overdraft = Some((reader.query, covered + 1));
                    }
                    queue.advance_past(self.ticket);
                    self.queued = false;
                    drop(queue);
                    drop(state);
                    // The reader's next admission is now first in its line.
                    budget.notify.notify_waiters();
                    return Reservation {
                        reader: self.reader.clone(),
                        bytes: AtomicU64::new(bytes),
                        overdraft,
                    };
                }
            }
            notified.await;
        }
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        if !self.queued {
            return;
        }
        let reader = &self.reader.0;
        reader.queue.lock().unwrap().advance_past(self.ticket);
        reader.budget.notify.notify_waiters();
    }
}

/// Memory reserved by a reader; released when dropped.
pub struct Reservation {
    reader: BudgetReader,
    // Only changed under the budget's state lock.
    bytes: AtomicU64,
    overdraft: bool,
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// The reader holding this reservation.
    pub fn reader(&self) -> &BudgetReader {
        &self.reader
    }

    /// Changes the reservation to `bytes` without waiting, once the actual
    /// size of the unit is known.
    pub fn resize(&self, bytes: u64) {
        let reader = &self.reader.0;
        let mut state = reader.budget.state.lock().unwrap();
        let mut queue = reader.queue.lock().unwrap();
        let old = self.bytes.swap(bytes, Ordering::Relaxed);
        state.reserved = state.reserved - old + bytes;
        state.peak_reserved = state.peak_reserved.max(state.reserved);
        queue.held = queue.held - old + bytes;
        let shrunk = bytes < old;
        drop(queue);
        drop(state);
        if shrunk {
            reader.budget.notify.notify_waiters();
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let reader = &self.reader.0;
        let mut state = reader.budget.state.lock().unwrap();
        let bytes = self.bytes.load(Ordering::Relaxed);
        state.reserved -= bytes;
        reader.queue.lock().unwrap().held -= bytes;
        if self.overdraft {
            state.overdraft = match state.overdraft {
                Some((query, covered)) if covered > 1 => Some((query, covered - 1)),
                _ => None,
            };
        }
        drop(state);
        reader.budget.notify.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::FutureExt;
    use rand::{Rng, SeedableRng};

    use super::*;

    const WAIT: Duration = Duration::from_millis(50);

    fn budget(limit: u64) -> Arc<ReadMemoryBudget> {
        Arc::new(ReadMemoryBudget::new(limit))
    }

    /// Asserts that `reserve` is still waiting after a short while.
    async fn assert_waits(
        admission: Admission,
        bytes: u64,
    ) -> tokio::task::JoinHandle<Reservation> {
        let handle = tokio::spawn(admission.reserve(bytes));
        tokio::time::sleep(WAIT).await;
        assert!(!handle.is_finished(), "{bytes} bytes should wait");
        handle
    }

    #[tokio::test]
    async fn test_admits_what_fits() {
        let budget = budget(100);
        let reader = budget.reader(QueryId::unique());
        // The first unit may use the whole limit, read-ahead up to 7/8 of it.
        let a = reader.admission().reserve(60).await;
        let b = reader.admission().reserve(25).await;
        assert_eq!(budget.reserved_bytes(), 85);
        drop((a, b));
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[tokio::test]
    async fn test_read_ahead_waits_for_release() {
        let budget = budget(100);
        let reader = budget.reader(QueryId::unique());
        let first = reader.admission().reserve(80).await;
        let waiting = assert_waits(reader.admission(), 40).await;
        drop(first);
        let second = waiting.await.unwrap();
        assert_eq!(second.bytes(), 40);
        assert_eq!(budget.reserved_bytes(), 40);
    }

    #[tokio::test]
    async fn test_read_ahead_leaves_room_for_first_units() {
        let budget = budget(800);
        let heavy = budget.reader(QueryId::unique());
        let _held = heavy.admission().reserve(650).await;
        // Read-ahead may not go past 7/8 of the limit (700 bytes)...
        let read_ahead = assert_waits(heavy.admission(), 100).await;
        // ...but another reader's first unit still fits under the full limit.
        budget
            .reader(QueryId::unique())
            .admission()
            .reserve(100)
            .now_or_never()
            .expect("a first unit uses the room read-ahead leaves");
        read_ahead.abort();
    }

    #[tokio::test]
    async fn test_reader_holding_nothing_overdraws() {
        let budget = budget(100);
        let other = budget
            .reader(QueryId::unique())
            .admission()
            .reserve(90)
            .await;
        // Does not fit, but the reader holds nothing: it takes the overdraft.
        let reader = budget.reader(QueryId::unique());
        let overdrawn = reader.admission().reserve(500).await;
        assert_eq!(budget.reserved_bytes(), 590);
        drop((other, overdrawn));
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[tokio::test]
    async fn test_one_overdraft_at_a_time() {
        let budget = budget(100);
        let first = budget
            .reader(QueryId::unique())
            .admission()
            .reserve(500)
            .await;
        // Another query's reader that holds nothing must wait for the overdraft.
        let waiting = assert_waits(budget.reader(QueryId::unique()).admission(), 200).await;
        drop(first);
        let _second = waiting.await.unwrap();
        assert_eq!(budget.reserved_bytes(), 200);
    }

    #[tokio::test]
    async fn test_query_shares_its_overdraft() {
        let budget = budget(100);
        let query = QueryId::unique();
        let upstream = budget.reader(query).admission().reserve(500).await;
        // A downstream reader of the same query cannot wait for its upstream.
        let downstream = budget
            .reader(query)
            .admission()
            .reserve(300)
            .now_or_never()
            .expect("the query's own reader shares the overdraft");
        drop(upstream);
        // Still held by the downstream reservation.
        let waiting = assert_waits(budget.reader(QueryId::unique()).admission(), 200).await;
        drop(downstream);
        waiting.await.unwrap();
    }

    #[tokio::test]
    async fn test_reader_admits_in_order() {
        let budget = budget(100);
        let reader = budget.reader(QueryId::unique());
        let held = reader.admission().reserve(50).await;
        let first = reader.admission();
        let second = reader.admission();
        // The second unit fits but must not overtake the first, which does not.
        let second = tokio::spawn(second.reserve(10));
        let first = assert_waits(first, 80).await;
        assert!(!second.is_finished());
        drop(held);
        first.await.unwrap();
        second.await.unwrap();
    }

    #[tokio::test]
    async fn test_dropped_admission_frees_its_turn() {
        let budget = budget(100);
        let reader = budget.reader(QueryId::unique());
        let held = reader.admission().reserve(70).await;
        let cancelled = reader.admission();
        let next = reader.admission();
        let cancelled = tokio::spawn(cancelled.reserve(50));
        tokio::time::sleep(WAIT).await;
        cancelled.abort();
        let _ = cancelled.await;
        next.reserve(10)
            .now_or_never()
            .expect("the next unit is first in line");
        drop(held);
    }

    #[tokio::test]
    async fn test_resident_bytes_count_against_limit() {
        let resident = Arc::new(AtomicUsize::new(70));
        let budget = Arc::new(ReadMemoryBudget::new(100).with_resident_bytes({
            let resident = resident.clone();
            move || resident.load(Ordering::Relaxed) as u64
        }));
        let reader = budget.reader(QueryId::unique());
        let held = reader.admission().reserve(20).await;
        let waiting = assert_waits(reader.admission(), 20).await;
        resident.store(50, Ordering::Relaxed);
        // Resident memory shrinking is noticed at the next release.
        drop(held);
        waiting.await.unwrap();
    }

    #[tokio::test]
    async fn test_resize() {
        let budget = budget(100);
        let reader = budget.reader(QueryId::unique());
        let held = reader.admission().reserve(90).await;
        let waiting = assert_waits(reader.admission(), 50).await;
        held.resize(30);
        let _admitted = waiting.await.unwrap();
        held.resize(200);
        assert_eq!(budget.reserved_bytes(), 250);
    }

    /// Many queries with chained readers and units of every size, including
    /// units larger than the whole budget, must all finish.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_concurrent_readers_never_deadlock() {
        const LIMIT: u64 = 1000;
        let budget = budget(LIMIT);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut queries = Vec::new();
        for _ in 0..64 {
            let query = QueryId::unique();
            let readers: Vec<_> = (0..rng.random_range(1..=3))
                .map(|_| budget.reader(query))
                .collect();
            let units: Vec<Vec<u64>> = readers
                .iter()
                .map(|_| {
                    (0..rng.random_range(1..20))
                        .map(|_| match rng.random_range(0..10) {
                            0 => rng.random_range(LIMIT..3 * LIMIT),
                            _ => rng.random_range(1..LIMIT / 4),
                        })
                        .collect()
                })
                .collect();
            queries.push(tokio::spawn(run_chained_readers(readers, units)));
        }
        tokio::time::timeout(
            Duration::from_secs(60),
            futures::future::try_join_all(queries),
        )
        .await
        .expect("readers deadlocked")
        .unwrap();
        assert_eq!(budget.reserved_bytes(), 0);
    }

    /// Readers chained like operators of one plan. Each step admits a unit
    /// for every reader, upstream first, and each upstream unit stays held
    /// until its downstream reader has admitted its own unit. The units are
    /// then consumed after a random delay while the next step reads ahead.
    async fn run_chained_readers(readers: Vec<BudgetReader>, units: Vec<Vec<u64>>) {
        let mut rng = rand::rngs::StdRng::from_os_rng();
        let max_len = units.iter().map(Vec::len).max().unwrap();
        let mut consumers = Vec::new();
        for step in 0..max_len {
            let mut step_units = Vec::new();
            for (reader, units) in readers.iter().zip(&units) {
                if let Some(&bytes) = units.get(step) {
                    step_units.push(reader.admission().reserve(bytes).await);
                }
            }
            let consume_after = Duration::from_micros(rng.random_range(0..2000));
            consumers.push(tokio::spawn(async move {
                tokio::time::sleep(consume_after).await;
                drop(step_units);
            }));
        }
        futures::future::try_join_all(consumers).await.unwrap();
    }
}
