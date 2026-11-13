// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A memory budget for the data that a session's queries read.
//!
//! A reader reserves the bytes a unit of work loads before requesting it and
//! releases them once the unit is decoded. A reader's units are admitted in
//! order, each once it fits under the limit; this is what bounds read-ahead.
//! Read-ahead leaves the last eighth of the limit to readers' first units and
//! to small readers, so small reads are not queued behind large ones. A reader
//! that holds nothing is admitted even when its unit does not fit, by taking
//! the overdraft, which lets one query at a time exceed the limit by what it
//! needs to make progress. What a unit admitted through the overdraft decodes
//! to keeps the overdraft until the consumer takes it. All readers of that
//! query may use the overdraft, because a downstream reader of a query cannot
//! wait for its own upstream to drain.
//!
//! A reader that holds nothing and waits is stuck: its consumer waits on it.
//! While a reader whose unit fits under the limit is stuck, freed memory goes
//! to stuck readers rather than to read-ahead, oldest first among those it can
//! admit. The query holding the overdraft is never held back, since stuck
//! readers may be waiting for it to release the overdraft.
//!
//! A reader only waits for memory that drains as consumers take what they
//! read: its own reservations, other readers' reservations, or the overdraft,
//! which its query releases as its consumer takes the overdrawn unit. A query
//! whose consumer stops taking while it holds the overdraft therefore holds up
//! other queries' readers that hold nothing and whose units do not fit, and
//! while those wait, other readers' read-ahead.

use std::any::Any;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

/// Identifies the query a reader belongs to. It holds the object that
/// identifies the query, so no other query can take its place while a reader
/// of it is alive.
#[derive(Clone)]
pub struct QueryId(Arc<dyn Any + Send + Sync>);

impl QueryId {
    /// The query identified by `owner`, an object that lives exactly as long
    /// as one execution (for example its task context).
    pub fn of<T: Any + Send + Sync>(owner: &Arc<T>) -> Self {
        Self(owner.clone())
    }

    /// A fresh identity for a read that is not part of a plan execution.
    pub fn unique() -> Self {
        Self::of(&Arc::new(()))
    }

    fn addr(&self) -> *const () {
        Arc::as_ptr(&self.0) as *const ()
    }
}

impl PartialEq for QueryId {
    fn eq(&self, other: &Self) -> bool {
        self.addr() == other.addr()
    }
}

impl Eq for QueryId {}

impl std::fmt::Debug for QueryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QueryId({:p})", self.addr())
    }
}

/// Read-ahead may use all but `1 / SMALL_READ_SHARE` of the limit; the rest is
/// for readers' first units and for readers holding no more than it.
const SMALL_READ_SHARE: u64 = 8;

/// Reads are split into chunks of about `1 / DEFAULT_CHUNKS_PER_LIMIT` of the
/// limit, so many readers can read ahead at once and the overdraft is small.
const DEFAULT_CHUNKS_PER_LIMIT: u64 = 16;

/// Chunks are at least this large, since each costs an estimate and a task.
/// Under a smaller limit, every chunk is admitted through the overdraft.
const MIN_CHUNK_BYTES: u64 = 1024 * 1024;

/// A read that waits this long logs what holds the budget, once.
const LONG_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// A memory limit shared by every reader of a session.
pub struct ReadMemoryBudget {
    limit: u64,
    chunks_per_limit: u64,
    state: Mutex<BudgetState>,
    notify: Notify,
    /// Total time admissions spent waiting for memory, in nanoseconds.
    waited_nanos: AtomicU64,
}

/// What a [`ReadMemoryBudget`] has seen, for callers that report or size from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadBudgetStats {
    /// The most bytes reserved at once since the budget was created or since
    /// [`ReadMemoryBudget::reset_peak`].
    pub peak_reserved_bytes: u64,
    /// Total time admissions waited for memory, across all readers.
    pub waited: std::time::Duration,
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
    peak_reserved: u64,
    /// The query holding the overdraft and the number of its reservations it covers.
    overdraft: Option<(QueryId, usize)>,
    /// Readers waiting while they hold nothing, so their consumers wait on
    /// them, oldest first.
    stuck: VecDeque<StuckReader>,
    next_stuck_id: u64,
}

struct StuckReader {
    id: u64,
    query: QueryId,
    bytes: u64,
}

impl BudgetState {
    /// Whether a unit of `bytes` of `query`'s reader holding nothing can be
    /// admitted now: under the limit, or through the overdraft.
    fn admits_first_unit(&self, limit: u64, query: &QueryId, bytes: u64) -> bool {
        self.reserved.saturating_add(bytes) <= limit
            || self
                .overdraft
                .as_ref()
                .is_none_or(|(holder, _)| holder == query)
    }

    /// The oldest stuck reader that can be admitted now, which freed memory
    /// goes to before any other reader holding nothing.
    fn next_stuck(&self, limit: u64) -> Option<u64> {
        self.stuck
            .iter()
            .find(|stuck| self.admits_first_unit(limit, &stuck.query, stuck.bytes))
            .map(|stuck| stuck.id)
    }

    fn unstick(&mut self, id: u64) {
        self.stuck.retain(|stuck| stuck.id != id);
    }

    /// Whether read-ahead of `query` must leave freed memory to stuck readers:
    /// those whose unit fits under the limit once enough is freed. A unit
    /// larger than the limit waits only for the overdraft.
    fn holds_back(&self, limit: u64, query: &QueryId) -> bool {
        self.stuck.iter().any(|stuck| stuck.bytes <= limit)
            && !matches!(&self.overdraft, Some((holder, _)) if holder == query)
    }

    /// Counts one more reservation of `query` under the overdraft.
    fn cover_overdraft(&mut self, query: &QueryId) {
        let covered = self.overdraft.as_ref().map_or(0, |(_, covered)| *covered);
        self.overdraft = Some((query.clone(), covered + 1));
    }

    fn add(&mut self, bytes: u64) {
        self.reserved += bytes;
        self.peak_reserved = self.peak_reserved.max(self.reserved);
    }
}

impl ReadMemoryBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            chunks_per_limit: DEFAULT_CHUNKS_PER_LIMIT,
            state: Mutex::default(),
            notify: Notify::new(),
            waited_nanos: AtomicU64::new(0),
        }
    }

    /// The peak reserved bytes and the total time admissions waited.
    pub fn stats(&self) -> ReadBudgetStats {
        ReadBudgetStats {
            peak_reserved_bytes: self.state.lock().unwrap().peak_reserved,
            waited: std::time::Duration::from_nanos(self.waited_nanos.load(Ordering::Relaxed)),
        }
    }

    /// Starts a new peak from what is reserved now, so a caller can measure
    /// the peak of one stretch of work.
    pub fn reset_peak(&self) {
        let mut state = self.state.lock().unwrap();
        state.peak_reserved = state.reserved;
    }

    /// Splits reads into chunks of about `1 / chunks_per_limit` of the limit
    /// (at least 1 MiB) instead of 1/16. Fewer, larger chunks suit rows that
    /// are large next to the limit; more, smaller ones let more readers read
    /// ahead at once and keep the overdraft small.
    pub fn with_chunks_per_limit(self, chunks_per_limit: u64) -> Self {
        Self {
            chunks_per_limit: chunks_per_limit.max(1),
            ..self
        }
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// The size reads are split into, counting what they load and decode.
    pub fn chunk_bytes(&self) -> u64 {
        (self.limit / self.chunks_per_limit).max(MIN_CHUNK_BYTES)
    }

    /// Bytes currently reserved by readers.
    pub fn reserved_bytes(&self) -> u64 {
        self.state.lock().unwrap().reserved
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
    /// The chunk size of the budget this reader draws on.
    pub fn chunk_bytes(&self) -> u64 {
        self.0.budget.chunk_bytes()
    }

    /// Another reader for the same query, with its own line. Use one per
    /// independently consumed stream: a reader's units wait for the ones
    /// before them, so streams sharing a reader could wait for each other.
    pub fn sibling(&self) -> Self {
        self.0.budget.reader(self.0.query.clone())
    }

    /// Reserves `bytes` the reader already holds, without waiting: memory that
    /// exists before it could be admitted, such as a decoded batch waiting to
    /// be emitted. It counts against the limit like any reservation.
    pub fn charge(&self, bytes: u64) -> Reservation {
        self.charge_as(bytes, false)
    }

    fn charge_as(&self, bytes: u64, overdraft: bool) -> Reservation {
        let mut state = self.0.budget.state.lock().unwrap();
        self.charge_locked(&mut state, bytes, overdraft)
    }

    fn charge_locked(&self, state: &mut BudgetState, bytes: u64, overdraft: bool) -> Reservation {
        state.add(bytes);
        if overdraft {
            state.cover_overdraft(&self.0.query);
        }
        self.0.queue.lock().unwrap().held += bytes;
        Reservation {
            reader: self.clone(),
            bytes: AtomicU64::new(bytes),
            overdraft,
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
            stuck: None,
        }
    }
}

/// A queued request for a unit of work's memory.
pub struct Admission {
    reader: BudgetReader,
    ticket: u64,
    queued: bool,
    /// Its place among the budget's stuck readers, while it is stuck.
    stuck: Option<u64>,
}

impl Admission {
    /// Waits until `bytes` may be loaded and reserves them.
    pub async fn reserve(mut self, bytes: u64) -> Reservation {
        let reader = self.reader.0.clone();
        let budget = &reader.budget;
        let mut waiting_since: Option<std::time::Instant> = None;
        let mut logged_long_wait = false;
        loop {
            // Created before checking so a release between the check and the
            // wait still wakes us.
            let notified = budget.notify.notified();
            {
                let mut state = budget.state.lock().unwrap();
                let mut queue = reader.queue.lock().unwrap();
                let total = state.reserved.saturating_add(bytes);
                let grant = if self.ticket != queue.serving_ticket {
                    None
                } else if queue.held == 0 {
                    // The reader can only make progress with this unit: admit
                    // it under the limit, or else through the overdraft if it
                    // is free or already held by this query. Older stuck
                    // readers that can be admitted go first.
                    let older_first = state
                        .next_stuck(budget.limit)
                        .is_some_and(|id| self.stuck != Some(id));
                    (!older_first && state.admits_first_unit(budget.limit, &reader.query, bytes))
                        .then_some(total > budget.limit)
                } else {
                    // Read-ahead stops short of the last part of the limit,
                    // which only a reader's first unit or a small reader may
                    // use, so a small read is not queued behind other readers'
                    // read-ahead.
                    let small_share = budget.limit / SMALL_READ_SHARE;
                    let limit = if queue.held + bytes <= small_share {
                        budget.limit
                    } else {
                        budget.limit - small_share
                    };
                    (total <= limit && !state.holds_back(budget.limit, &reader.query))
                        .then_some(false)
                };
                if let Some(overdraft) = grant {
                    state.add(bytes);
                    queue.held += bytes;
                    if let Some(since) = waiting_since {
                        budget
                            .waited_nanos
                            .fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    if overdraft {
                        state.cover_overdraft(&reader.query);
                    }
                    queue.advance_past(self.ticket);
                    self.queued = false;
                    if let Some(id) = self.stuck.take() {
                        state.unstick(id);
                    }
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
                let stuck = self.ticket == queue.serving_ticket && queue.held == 0;
                match self.stuck {
                    None if stuck => {
                        let id = state.next_stuck_id;
                        state.next_stuck_id += 1;
                        state.stuck.push_back(StuckReader {
                            id,
                            query: reader.query.clone(),
                            bytes,
                        });
                        self.stuck = Some(id);
                    }
                    Some(id) if !stuck => {
                        state.unstick(id);
                        self.stuck = None;
                        budget.notify.notify_waiters();
                    }
                    _ => {}
                }
            }
            let since = *waiting_since.get_or_insert_with(std::time::Instant::now);
            if logged_long_wait {
                notified.await;
            } else if tokio::time::timeout(LONG_WAIT.saturating_sub(since.elapsed()), notified)
                .await
                .is_err()
            {
                logged_long_wait = true;
                let state = budget.state.lock().unwrap();
                let held = reader.queue.lock().unwrap().held;
                log::warn!(
                    "A read of {bytes} bytes has waited {:?} for the read memory budget: \
                     {} of {} bytes reserved, this reader holds {held}, {} readers holding \
                     nothing are waiting, overdraft held by {}",
                    since.elapsed(),
                    state.reserved,
                    budget.limit,
                    state.stuck.len(),
                    match &state.overdraft {
                        None => "no query",
                        Some((query, _)) if *query == reader.query => "this query",
                        Some(_) => "another query",
                    },
                );
            }
        }
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        if !self.queued {
            return;
        }
        let reader = &self.reader.0;
        if let Some(id) = self.stuck {
            reader.budget.state.lock().unwrap().unstick(id);
        }
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

    /// Moves `bytes` of this reservation to a new one, such as a batch decoded
    /// from this reservation's unit that outlives it, keeping at least `keep`
    /// here for memory this one still covers, and reserves without waiting
    /// whatever part it cannot move. Memory that came through the overdraft
    /// keeps it until released.
    pub fn split_off(&self, bytes: u64, keep: u64) -> Self {
        let reader = &self.reader;
        let mut state = reader.0.budget.state.lock().unwrap();
        let moved = self.bytes().saturating_sub(keep).min(bytes);
        self.bytes.fetch_sub(moved, Ordering::Relaxed);
        state.reserved -= moved;
        reader.0.queue.lock().unwrap().held -= moved;
        reader.charge_locked(&mut state, bytes, self.overdraft)
    }

    /// Changes the reservation to `bytes` without waiting, once the actual
    /// size of the unit is known.
    pub fn resize(&self, bytes: u64) {
        let reader = &self.reader.0;
        let mut state = reader.budget.state.lock().unwrap();
        let mut queue = reader.queue.lock().unwrap();
        let old = self.bytes.swap(bytes, Ordering::Relaxed);
        state.reserved -= old;
        state.add(bytes);
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
            state.overdraft = match state.overdraft.take() {
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

    /// A query's identity is not reused while it is held, even after its
    /// owner is dropped.
    #[test]
    fn test_query_id_outlives_its_owner() {
        let owner = Arc::new(0u64);
        let query = QueryId::of(&owner);
        assert_eq!(query, QueryId::of(&owner));
        drop(owner);
        let others: Vec<_> = (0..64).map(|_| Arc::new(0u64)).collect();
        assert!(others.iter().all(|other| QueryId::of(other) != query));
        assert_ne!(QueryId::unique(), QueryId::unique());
    }

    #[test]
    fn test_chunk_bytes() {
        assert_eq!(ReadMemoryBudget::new(64 << 20).chunk_bytes(), 4 << 20);
        let budget = ReadMemoryBudget::new(64 << 20).with_chunks_per_limit(4);
        assert_eq!(budget.chunk_bytes(), 16 << 20);
        assert_eq!(ReadMemoryBudget::new(4 << 20).chunk_bytes(), 1 << 20);
    }

    /// The budget reports the most it held and how long reads waited.
    #[tokio::test]
    async fn test_stats_report_the_peak_and_time_waited() {
        let budget = budget(1000);
        let reader = budget.reader(QueryId::unique());
        let first = reader.admission().reserve(500).await;
        let second = reader.admission().reserve(300).await;
        assert_eq!(budget.stats().peak_reserved_bytes, 800);
        assert_eq!(budget.stats().waited, Duration::ZERO);
        drop(second);
        budget.reset_peak();
        assert_eq!(budget.stats().peak_reserved_bytes, 500);

        // Another reader's read-ahead waits for room, and its wait is counted.
        let other = budget.reader(QueryId::unique());
        let _other_first = other.admission().reserve(1).await;
        let waiting = assert_waits(other.admission(), 400).await;
        drop(first);
        let _admitted = waiting.await.unwrap();
        assert!(budget.stats().waited > Duration::ZERO);
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
        // A reader holding little keeps reading ahead into that room.
        let small = budget.reader(QueryId::unique());
        let _first = small.admission().reserve(40).await;
        small
            .admission()
            .reserve(40)
            .now_or_never()
            .expect("a small reader reads ahead into the room large ones leave");
    }

    /// `reader` takes the overdraft and then holds `bytes` of it, so another
    /// query's reader holding nothing whose unit does not fit must wait.
    async fn hold_overdraft(reader: &BudgetReader, bytes: u64) -> Reservation {
        let overdraft = reader
            .admission()
            .reserve(reader.0.budget.limit() + 1)
            .await;
        overdraft.resize(bytes);
        overdraft
    }

    /// Memory freed while a reader holding nothing waits goes to it, not to
    /// another reader's read-ahead, even read-ahead that fits; a stuck reader
    /// that gives up stops holding read-ahead back.
    #[rstest::rstest]
    #[tokio::test]
    async fn test_stuck_reader_goes_before_read_ahead(#[values(false, true)] gives_up: bool) {
        let budget = budget(1000);
        let _overdraft = hold_overdraft(&budget.reader(QueryId::unique()), 1).await;
        let heavy = budget.reader(QueryId::unique());
        let held = heavy.admission().reserve(500).await;
        let other = budget.reader(QueryId::unique());
        let _other_held = other.admission().reserve(1).await;
        let stuck = assert_waits(budget.reader(QueryId::unique()).admission(), 600).await;
        // 10 bytes fit, but a reader holding nothing is waiting.
        let read_ahead = assert_waits(other.admission(), 10).await;
        if gives_up {
            stuck.abort();
            assert!(stuck.await.is_err());
        } else {
            drop(held);
            stuck.await.unwrap();
        }
        tokio::time::timeout(Duration::from_secs(10), read_ahead)
            .await
            .expect("read-ahead stayed held back")
            .unwrap();
        assert!(budget.state.lock().unwrap().stuck.is_empty());
    }

    /// Stuck readers are served oldest first among those that can be admitted,
    /// so one waiting for the overdraft does not hold up a later one that fits,
    /// nor, since its unit is larger than the limit, read-ahead. The overdraft
    /// holder's read-ahead is never held back.
    #[tokio::test]
    async fn test_stuck_readers_served_oldest_first() {
        let budget = budget(1000);
        let holder = budget.reader(QueryId::unique());
        let _overdraft = hold_overdraft(&holder, 500).await;
        let _needs_overdraft =
            assert_waits(budget.reader(QueryId::unique()).admission(), 1001).await;
        let fits_reader = budget.reader(QueryId::unique());
        let fits = fits_reader
            .admission()
            .reserve(300)
            .now_or_never()
            .expect("a unit that fits waited behind one that cannot be admitted");
        let _read_ahead = fits_reader
            .admission()
            .reserve(10)
            .now_or_never()
            .expect("a unit larger than the limit held back read-ahead");
        let older = assert_waits(budget.reader(QueryId::unique()).admission(), 300).await;
        let held_back = assert_waits(fits_reader.admission(), 10).await;
        holder
            .admission()
            .reserve(10)
            .now_or_never()
            .expect("the overdraft holder was held back");
        drop(fits);
        // The freed memory fits a newcomer too, but the older reader is first.
        let newcomer = budget.reader(QueryId::unique()).admission().reserve(300);
        assert!(newcomer.now_or_never().is_none(), "a newcomer went first");
        let _older = older.await.unwrap();
        held_back.abort();
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
        // What the overdrawn unit decodes to keeps the overdraft, and moves
        // out of the unit's reservation rather than adding to it.
        let output = first.split_off(50, 0);
        assert_eq!(budget.reserved_bytes(), 500);
        // Past what the unit can give up, the rest is reserved.
        let more = first.split_off(500, 100);
        assert_eq!(budget.reserved_bytes(), 650);
        drop((first, more));
        // Another query's reader that holds nothing must wait for the overdraft.
        let waiting = assert_waits(budget.reader(QueryId::unique()).admission(), 200).await;
        drop(output);
        let _second = waiting.await.unwrap();
        assert_eq!(budget.reserved_bytes(), 200);
    }

    #[tokio::test]
    async fn test_query_shares_its_overdraft() {
        let budget = budget(100);
        let query = QueryId::unique();
        let upstream = budget.reader(query.clone()).admission().reserve(500).await;
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

    /// Many queries with chained readers and units
    /// of every size, including units larger than the whole budget, must all
    /// finish.
    #[rstest::rstest]
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_concurrent_readers_never_deadlock(#[values(42, 7, 2026)] seed: u64) {
        const LIMIT: u64 = 1000;
        let budget = budget(LIMIT);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut queries = Vec::new();
        for _ in 0..64 {
            let query = QueryId::unique();
            let readers: Vec<_> = (0..rng.random_range(1..=3))
                .map(|_| budget.reader(query.clone()))
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
