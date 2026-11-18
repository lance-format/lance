// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in timing of k-means training, for benchmarks.
//!
//! With the `kmeans-profile` feature, two kinds of timers accumulate:
//!
//! - [`Phase`] timers wrap a step of the training driver, such as one node's
//!   Lloyd assignment, including the time the step waits on its parallel
//!   tasks. With one thread they give the exact time split between steps; with
//!   several, concurrent sibling sub-trees make them overlap.
//! - [`Work`] timers wrap code that runs on one thread without waiting: a
//!   parallel task body or a sequential stretch of the driver. Their sum is the
//!   time threads spent busy, so `threads * wall - busy` is the time threads
//!   spent idle: dispatching and stealing tasks, waiting at joins, and starved
//!   of work.
//!
//! Without the feature every timer is a zero-sized no-op.

/// A step of k-means training, timed on the thread that drives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Shuffle a node's rows and gather its training sample.
    NodeSample,
    /// Train one node's sub-clustering, Lloyd or e-means.
    NodeTrain,
    /// Lloyd: assign the node's sample to the centroids, one iteration.
    LloydAssign,
    /// Lloyd: recompute the centroids from the assignment, one iteration.
    LloydUpdate,
    /// E-means: assign one batch to the centroids.
    EMeansAssign,
    /// E-means: EMA update, revival and centroid read-off after one batch.
    EMeansUpdate,
    /// E-means: last-epoch-mean finalization.
    EMeansFinalize,
    /// Assign all of a node's rows to its sub-clusters.
    Membership,
    /// Compute leaf centroids as the means of their members.
    Leaves,
}

impl Phase {
    pub const ALL: [Self; 9] = [
        Self::NodeSample,
        Self::NodeTrain,
        Self::LloydAssign,
        Self::LloydUpdate,
        Self::EMeansAssign,
        Self::EMeansUpdate,
        Self::EMeansFinalize,
        Self::Membership,
        Self::Leaves,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::NodeSample => "node_sample",
            Self::NodeTrain => "node_train",
            Self::LloydAssign => "lloyd_assign",
            Self::LloydUpdate => "lloyd_update",
            Self::EMeansAssign => "emeans_assign",
            Self::EMeansUpdate => "emeans_update",
            Self::EMeansFinalize => "emeans_finalize",
            Self::Membership => "membership",
            Self::Leaves => "leaves",
        }
    }
}

/// What a thread is busy doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// Finding the nearest centroid of vectors.
    Assign,
    /// Summing members, moving centroids, reviving or splitting clusters.
    Update,
    /// Copying rows, shuffling and sampling.
    Gather,
    /// Partition bookkeeping.
    Other,
}

impl Work {
    pub const ALL: [Self; 4] = [Self::Assign, Self::Update, Self::Gather, Self::Other];

    pub fn name(self) -> &'static str {
        match self {
            Self::Assign => "assign",
            Self::Update => "update",
            Self::Gather => "gather",
            Self::Other => "other",
        }
    }
}

/// Accumulated time and number of timed calls.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Total {
    pub secs: f64,
    pub calls: u64,
}

/// Totals of every timer since the last [`reset`].
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub phases: Vec<(Phase, Total)>,
    pub work: Vec<(Work, Total)>,
}

#[cfg(feature = "kmeans-profile")]
mod imp {
    use std::cell::OnceCell;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use super::{Phase, Profile, Total, Work};

    const SLOTS: usize = Phase::ALL.len() + Work::ALL.len();

    /// One thread's totals. Each thread adds only to its own slots, so the hot
    /// per-vector timers never contend on a shared cache line.
    #[derive(Default)]
    struct Slots {
        nanos: [AtomicU64; SLOTS],
        calls: [AtomicU64; SLOTS],
        /// Open `Work` timers on this thread; nonzero while it is busy.
        active: AtomicU32,
    }

    static THREADS: Mutex<Vec<Arc<Slots>>> = Mutex::new(Vec::new());

    thread_local! {
        static LOCAL: OnceCell<Arc<Slots>> = const { OnceCell::new() };
    }

    fn with_slots(f: impl FnOnce(&Slots)) {
        LOCAL.with(|local| {
            f(local.get_or_init(|| {
                let slots = Arc::new(Slots::default());
                THREADS
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(slots.clone());
                slots
            }))
        });
    }

    #[must_use]
    pub struct Timer {
        slot: usize,
        start: Instant,
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            let nanos = self.start.elapsed().as_nanos() as u64;
            let is_work = self.slot >= Phase::ALL.len();
            with_slots(|slots| {
                slots.nanos[self.slot].fetch_add(nanos, Ordering::Relaxed);
                slots.calls[self.slot].fetch_add(1, Ordering::Relaxed);
                if is_work {
                    slots.active.fetch_sub(1, Ordering::Relaxed);
                }
            });
        }
    }

    pub fn phase(phase: Phase) -> Timer {
        Timer {
            slot: phase as usize,
            start: Instant::now(),
        }
    }

    pub fn work(work: Work) -> Timer {
        with_slots(|slots| {
            slots.active.fetch_add(1, Ordering::Relaxed);
        });
        Timer {
            slot: Phase::ALL.len() + work as usize,
            start: Instant::now(),
        }
    }

    pub fn busy_threads() -> usize {
        THREADS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|slots| slots.active.load(Ordering::Relaxed) > 0)
            .count()
    }

    fn total(threads: &[Arc<Slots>], slot: usize) -> Total {
        Total {
            secs: threads
                .iter()
                .map(|t| t.nanos[slot].load(Ordering::Relaxed))
                .sum::<u64>() as f64
                * 1e-9,
            calls: threads
                .iter()
                .map(|t| t.calls[slot].load(Ordering::Relaxed))
                .sum(),
        }
    }

    pub fn snapshot() -> Profile {
        let threads = THREADS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Profile {
            phases: Phase::ALL
                .iter()
                .map(|&phase| (phase, total(&threads, phase as usize)))
                .collect(),
            work: Work::ALL
                .iter()
                .map(|&work| (work, total(&threads, Phase::ALL.len() + work as usize)))
                .collect(),
        }
    }

    pub fn reset() {
        for slots in THREADS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
        {
            for slot in 0..SLOTS {
                slots.nanos[slot].store(0, Ordering::Relaxed);
                slots.calls[slot].store(0, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(not(feature = "kmeans-profile"))]
mod imp {
    use super::{Phase, Profile, Work};

    #[must_use]
    pub struct Timer;

    #[inline(always)]
    pub fn phase(_: Phase) -> Timer {
        Timer
    }

    #[inline(always)]
    pub fn work(_: Work) -> Timer {
        Timer
    }

    pub fn snapshot() -> Profile {
        Profile::default()
    }

    pub fn reset() {}

    pub fn busy_threads() -> usize {
        0
    }
}

pub use imp::Timer;
/// Threads inside a [`Work`] timer at this moment, for sampling utilization
/// over time; always zero without the `kmeans-profile` feature.
pub use imp::busy_threads;
/// Times one [`Phase`] until the returned timer drops.
pub use imp::phase;
/// Zeroes every timer.
pub use imp::reset;
/// Totals of every timer since the last [`reset`]; empty without the
/// `kmeans-profile` feature.
pub use imp::snapshot;
/// Times one stretch of [`Work`] until the returned timer drops.
pub use imp::work;
