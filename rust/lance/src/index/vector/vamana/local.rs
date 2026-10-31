// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Re-score reads of local data files made without Lance's I/O scheduler.
//!
//! A re-score reads a few dozen vectors of a few kilobytes each, which the
//! page cache almost always holds. Through the scheduler a query pays, for
//! every data file, an open on the blocking pool, a scheduler with a task of
//! its own and a queue, and under load those cost more than the reads. Here a
//! data file stays open in a pool the process shares, and each range is read
//! on the thread that asks for it with `preadv2(RWF_NOWAIT)`, which the kernel
//! serves only when the page cache holds the bytes. The ranges it refuses go
//! to the blocking pool together, in one trip per file.
//!
//! Only on 64-bit Linux, for local storage Lance itself reads by direct path,
//! for files on a filesystem known to answer the flag, and not when
//! `LANCE_VAMANA_READ_IN_PLACE` is set to anything but a true value. The call
//! goes to the kernel directly rather than through glibc's `preadv2`, which
//! wheels built for glibc 2.17 cannot link. The flag spares the asking thread
//! the device, not every wait: the permission hook of an on-access scanner
//! (fanotify) runs before the kernel looks at the flag, and a read can still
//! update the file's access time. The switch turns these reads off where
//! either matters, or where a sandbox kills a process for a system call it
//! does not know.
//!
//! The query's own counts and the object store's tracker get every read the
//! scheduler would have recorded. The scheduler's process-wide `iops_counter`
//! and `bytes_read_counter` cannot be added to from outside it, so these reads
//! are missing from them, as the reads of Lance's lite scheduler are.

use std::fs::File;
use std::ops::{AddAssign, Range};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, LazyLock, Weak};
use std::time::Duration;

use lance_core::utils::parse::parse_env_as_bool;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::IoStats;
use lance_io::utils::tracking_store::{IOTracker, IoMetricsGuard};
use moka::policy::EvictionPolicy;
use moka::sync::Cache;
use object_store::path::Path;
use tokio::task::JoinHandle;

use crate::{Error, Result};

/// The most one call reads on the thread that asked, summed over its files.
///
/// A copy out of the page cache costs the same on any thread, so the only
/// question is which. For a shortlist of twenty vectors, 10 to 77 kB from
/// `d = 128` to `d = 960`, it is the thread that asked, since a trip to the
/// blocking pool costs more than the copy. The search list of a large `k` is
/// copied on the blocking pool instead, so that one query does not hold a
/// runtime worker for long.
pub const IN_PLACE_BYTES: u64 = 1 << 20;

/// The most data files the process holds open, before its descriptor limit
/// lowers it.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const OPEN_FILES: u64 = 64;

/// How long a held file may go unread before it is closed, so that a file a
/// compaction replaced and a cleanup deleted stops holding its disk space.
const IDLE: Duration = Duration::from_secs(60);

/// How long after an open of a path, or a failed one, the path is not opened
/// again: a file the pool did not take in, or could not open, is read through
/// the scheduler meanwhile rather than opened on every miss.
const BACK_OFF: Duration = Duration::from_secs(10);

/// How long a directory on a filesystem off the list is remembered as such.
const ELSEWHERE: Duration = Duration::from_secs(600);

/// The most paths and directories the pool remembers an attempt or a verdict
/// for.
const REMEMBERED: u64 = 4096;

/// What a read of one range on the asking thread came to.
///
/// Off 64-bit Linux no read is made on the asking thread, so only the
/// refusal of the call is ever built there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    not(all(target_os = "linux", target_pointer_width = "64")),
    allow(dead_code)
)]
pub enum Nowait {
    /// Every byte was read.
    Served,
    /// Not this time: the page cache lacked some of the bytes, or a signal
    /// interrupted the call. The blocking read that follows asks again and
    /// reports any real error.
    Refused,
    /// The file's filesystem refuses the flag (kernels 4.6 to 4.13 refuse it
    /// everywhere), or a permission hook refuses the file: its later ranges
    /// go straight to the blocking pool, whose read reports a refusal that
    /// holds.
    FileRefuses,
    /// The kernel lacks the call, or a sandbox forbids it: no range of any
    /// file is read on the asking thread again.
    ProcessRefuses,
}

/// How a range is read on the asking thread: `len` bytes from `offset` into
/// `buf`, emptied first, which holds exactly those bytes on
/// [`Nowait::Served`].
pub type ReadNow = fn(&File, &mut Vec<u8>, usize, u64) -> Nowait;

/// How a data file is opened to be held: `None` for one on a filesystem not
/// known to answer `RWF_NOWAIT`, or whose filesystem cannot be told.
pub type OpenFile = fn(&str) -> std::io::Result<Option<File>>;

/// A local data file held open.
#[derive(Debug)]
pub struct HeldFile {
    file: File,
    /// Its size when it was opened, which the manifest naming its path has to
    /// give it too.
    size: u64,
    /// Set once its filesystem has refused `RWF_NOWAIT`.
    refuses_nowait: AtomicBool,
}

impl HeldFile {
    /// Its size when it was opened.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// The local data files a process holds open for re-scores, by path.
///
/// One pool for the process, because descriptors are the process's: a pool
/// for each dataset of each session would run out of them. It holds
/// [`OPEN_FILES`] files, or a sixteenth of the soft descriptor limit when that
/// is fewer. A file that misses is read through the scheduler while it is
/// opened in the background, so a miss costs no more than a read did without
/// the pool, and admission by frequency keeps reads spread over more files
/// than the pool holds from cycling it.
pub struct Descriptors {
    files: Cache<String, Arc<HeldFile>>,
    /// The paths opened, or tried, within the back-off: at most one open of a
    /// path is under way, and none soon after the last.
    attempted: Cache<String, ()>,
    /// The directories found on a filesystem off the list.
    elsewhere: Cache<String, ()>,
    open: OpenFile,
    /// Set when the kernel or a sandbox refused the call: from then on every
    /// read of a held file is made on the blocking pool.
    refused: AtomicBool,
}

static GLOBAL: LazyLock<Arc<Descriptors>> =
    LazyLock::new(|| Descriptors::new(capacity(), IDLE, BACK_OFF, open_listed));

static SWITCHED_ON: LazyLock<bool> =
    LazyLock::new(|| parse_env_as_bool("LANCE_VAMANA_READ_IN_PLACE", true));

impl Descriptors {
    /// The pool every re-score of the process shares.
    pub fn global() -> &'static Arc<Self> {
        &GLOBAL
    }

    /// A pool of at most `capacity` files opened by `open`, each closed once
    /// it goes unread for `idle`, and a path not opened again within
    /// `back_off` of its last open.
    ///
    /// A thread of the pool's own closes the files the pool lets go, off the
    /// runtime's workers, since closing a file a cleanup deleted frees its
    /// blocks then and there; and it runs the pool's evictions every quarter
    /// of `idle`, so that an idle file is closed while no query reads.
    pub fn new(capacity: u64, idle: Duration, back_off: Duration, open: OpenFile) -> Arc<Self> {
        let (closer, closing) = channel::<Arc<HeldFile>>();
        let files = Cache::builder()
            .max_capacity(capacity)
            .time_to_idle(idle)
            .eviction_listener(move |_path, file, _cause| {
                // Without the closing thread, the file closes here.
                let _ = closer.send(file);
            })
            .build();
        let remembered = |for_how_long| {
            Cache::builder()
                .max_capacity(REMEMBERED)
                .time_to_live(for_how_long)
                .eviction_policy(EvictionPolicy::lru())
                .build()
        };
        let pool = Arc::new(Self {
            files,
            attempted: remembered(back_off),
            elsewhere: remembered(ELSEWHERE),
            open,
            refused: AtomicBool::new(false),
        });
        let held = Arc::downgrade(&pool);
        let period = (idle / 4).max(Duration::from_millis(10));
        let spawned = std::thread::Builder::new()
            .name("lance-vamana-files".to_string())
            .spawn(move || close_files(&held, &closing, period));
        if let Err(error) = spawned {
            log::warn!("IVF_VAMANA closes the data files it holds only as re-scores run: {error}");
        }
        pool
    }

    /// The file the pool holds for `path`.
    pub fn get(&self, path: &str) -> Option<Arc<HeldFile>> {
        self.files.get(path)
    }

    /// Let go of the file held for `path`, which no longer holds what the
    /// manifest names.
    pub fn forget(&self, path: &str) {
        self.files.invalidate(path);
    }

    /// Whether the data file at `path` lies in a directory found on a
    /// filesystem off the list.
    pub fn elsewhere(&self, path: &str) -> bool {
        directory(path).is_some_and(|directory| self.elsewhere.contains_key(directory))
    }

    /// Open the data file at `path` on the blocking pool and hold it, unless
    /// the pool holds no files, or the path was opened or tried within the
    /// back-off. The handle is for a caller that has to know when the file is
    /// held.
    ///
    /// A file on a filesystem off the list marks its directory, whose files
    /// are not opened again. A failed open holds nothing and marks nothing,
    /// running out of descriptors being something that passes. Of two opens of
    /// one file, the first to finish is held.
    pub fn open_in_background(self: &Arc<Self>, path: String) -> Option<JoinHandle<()>> {
        if self.files.policy().max_capacity() == Some(0)
            || !self.attempted.entry(path.clone()).or_insert(()).is_fresh()
        {
            return None;
        }
        let pool = self.clone();
        Some(tokio::task::spawn_blocking(move || {
            let opened = (pool.open)(&path).and_then(|file| {
                file.map(|file| Ok((file.metadata()?.len(), file)))
                    .transpose()
            });
            match opened {
                Ok(Some((size, file))) => {
                    let held = HeldFile {
                        file,
                        size,
                        refuses_nowait: AtomicBool::new(false),
                    };
                    pool.files.entry(path).or_insert_with(|| Arc::new(held));
                }
                Ok(None) => {
                    if let Some(directory) = directory(&path) {
                        pool.elsewhere.insert(directory.to_string(), ());
                    }
                }
                Err(error) => {
                    log::debug!("IVF_VAMANA does not hold data file {path} open: {error}");
                }
            }
        }))
    }

    /// Run the pool's pending evictions and admissions now.
    #[cfg(test)]
    pub fn settle(&self) {
        self.files.run_pending_tasks();
        self.attempted.run_pending_tasks();
        self.elsewhere.run_pending_tasks();
    }

    /// How many files the pool holds.
    #[cfg(test)]
    pub fn held_files(&self) -> u64 {
        self.files.entry_count()
    }
}

/// The directory of the file at `path`.
fn directory(path: &str) -> Option<&str> {
    std::path::Path::new(path).parent()?.to_str()
}

/// Close the files `closing` brings, and run the evictions of `pool` every
/// `period`, until the pool is gone.
fn close_files(pool: &Weak<Descriptors>, closing: &Receiver<Arc<HeldFile>>, period: Duration) {
    loop {
        match closing.recv_timeout(period) {
            Ok(file) => drop(file),
            Err(RecvTimeoutError::Timeout) => match pool.upgrade() {
                Some(pool) => pool.files.run_pending_tasks(),
                None => return,
            },
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// Whether data files of `store` are read here: local storage Lance reads by
/// direct path (the condition of `ObjectStore::open`, which leaves
/// `file+uring` to its own reader), on 64-bit Linux, unless switched off.
pub fn reads_in_place(store: &ObjectStore) -> bool {
    cfg!(all(target_os = "linux", target_pointer_width = "64"))
        && store.scheme() == "file"
        && store.has_direct_local_paths()
        && *SWITCHED_ON
}

/// Whether `LANCE_VAMANA_READ_IN_PLACE` leaves these reads on.
#[cfg(test)]
pub fn switched_on() -> bool {
    *SWITCHED_ON
}

/// How one call's reads of local data files went.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalReads {
    /// Reads served on the thread that asked.
    pub in_place: usize,
    /// Reads made on the blocking pool.
    pub handed_off: usize,
    /// Trips to the blocking pool, at most one for each file.
    pub trips: usize,
}

impl AddAssign for LocalReads {
    fn add_assign(&mut self, other: Self) {
        self.in_place += other.in_place;
        self.handed_off += other.handed_off;
        self.trips += other.trips;
    }
}

/// What [`read`] needs besides the file and its ranges.
pub struct ReadContext<'a> {
    /// The file's store, whose sizes the scheduler would merge and split by
    /// and whose tracker it would record in.
    pub store: &'a ObjectStore,
    /// The file's path in `store`, as the tracker records it.
    pub path: &'a Path,
    pub io_stats: Option<&'a IoStats>,
    pub pool: &'a Descriptors,
    pub read_now: ReadNow,
    /// What the call may still read on the thread that asked, shared by all
    /// of its files.
    pub budget: &'a AtomicU64,
}

/// The bytes of the ranges [`read`] was asked for.
pub struct ReadBytes {
    buffers: Vec<Vec<u8>>,
    /// For each range asked for, the buffer that holds it and where in it.
    slices: Vec<(usize, Range<usize>)>,
    pub reads: LocalReads,
}

impl ReadBytes {
    /// The bytes of each range asked for, in the order asked.
    pub fn chunks(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.slices
            .iter()
            .map(|(buffer, range)| &self.buffers[*buffer][range.clone()])
    }
}

/// The bytes of `ranges` of `held`, which ascend without overlapping, read on
/// the thread that asks where the page cache serves them and within the
/// call's budget, and on the blocking pool otherwise.
///
/// Once a read of the file is refused, the rest of the file goes to the
/// blocking pool with it: the trip is certain then, and a read the page cache
/// misses starts reading the device on the asking thread before it is
/// refused. A refused read spends none of the budget.
///
/// The ranges are merged and split as `FileScheduler::submit_request` merges
/// and splits a request, and counted where it counts them: the split list
/// once in the query's `io_stats` as the reads start, and each piece of it in
/// the store's tracker, by the outcome of its own read, as Lance's local
/// reader records a read. A merged range is read whole even when it is split
/// for counting.
pub async fn read(
    held: Arc<HeldFile>,
    ranges: &[Range<u64>],
    context: &ReadContext<'_>,
) -> Result<ReadBytes> {
    let requests = Requests::of(
        ranges,
        context.store.block_size() as u64,
        context.store.max_iop_size(),
    );
    if let Some(io_stats) = context.io_stats {
        io_stats.record_request(&requests.split);
    }
    let tracker = context.store.io_tracker();
    let guards = requests
        .split
        .iter()
        .map(|_| tracker.begin_io("get"))
        .collect::<Vec<_>>();
    let mut reads = LocalReads::default();
    let mut buffers = vec![Vec::new(); requests.merged.len()];
    let mut handed_off = vec![false; requests.merged.len()];
    let mut nowait = !held.refuses_nowait.load(Ordering::Relaxed)
        && !context.pool.refused.load(Ordering::Relaxed);
    for (index, range) in requests.merged.iter().enumerate() {
        let bytes = range.end - range.start;
        let len = usize::try_from(bytes).map_err(|_| {
            Error::internal(format!(
                "IVF_VAMANA reads {bytes} bytes of a data file at once"
            ))
        })?;
        if nowait && spend(context.budget, bytes) {
            let outcome = (context.read_now)(&held.file, &mut buffers[index], len, range.start);
            if outcome == Nowait::Served {
                reads.in_place += 1;
                continue;
            }
            context.budget.fetch_add(bytes, Ordering::Relaxed);
            nowait = false;
            match outcome {
                Nowait::FileRefuses => held.refuses_nowait.store(true, Ordering::Relaxed),
                Nowait::ProcessRefuses => {
                    if !context.pool.refused.swap(true, Ordering::Relaxed) {
                        log::warn!(
                            "IVF_VAMANA cannot read data files without waiting here, the kernel \
                             or a sandbox refusing preadv2, and re-scores them on the blocking pool"
                        );
                    }
                }
                Nowait::Served | Nowait::Refused => {}
            }
        }
        handed_off[index] = true;
    }
    let wanted = handed_off
        .iter()
        .enumerate()
        .filter(|(_, handed)| **handed)
        .map(|(index, _)| (index, requests.merged[index].clone()))
        .collect::<Vec<_>>();
    let trip = if wanted.is_empty() {
        Ok(())
    } else {
        reads.trips = 1;
        reads.handed_off = wanted.len();
        match tokio::task::spawn_blocking(move || read_blocking(&held.file, wanted)).await {
            Ok(Ok(read)) => {
                for (index, buffer) in read {
                    buffers[index] = buffer;
                }
                Ok(())
            }
            Ok(Err(error)) => Err(Error::from(error)),
            Err(error) => Err(Error::io(format!(
                "IVF_VAMANA re-score read on the blocking pool failed: {error}"
            ))),
        }
    };
    record(tracker, context.path, &requests, guards, &handed_off, &trip);
    trip?;
    Ok(ReadBytes {
        buffers,
        slices: requests.slices,
        reads,
    })
}

/// Take `bytes` out of `budget`, if it holds that many.
fn spend(budget: &AtomicU64, bytes: u64) -> bool {
    budget
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
            left.checked_sub(bytes)
        })
        .is_ok()
}

/// Each piece of the split list in `tracker`, by the outcome of the read of
/// its merged range: served in place, or `trip` for the ranges `handed_off`
/// to the blocking pool, as Lance's local reader records each read it makes.
fn record(
    tracker: &IOTracker,
    path: &Path,
    requests: &Requests,
    guards: Vec<IoMetricsGuard>,
    handed_off: &[bool],
    trip: &Result<()>,
) {
    let served: Result<()> = Ok(());
    for ((guard, range), owner) in guards
        .into_iter()
        .zip(&requests.split)
        .zip(&requests.split_owners)
    {
        let bytes = range.end - range.start;
        let outcome = if handed_off[*owner] { trip } else { &served };
        guard.record(outcome, bytes);
        if outcome.is_ok() {
            tracker.record_read("get_range", path.clone(), bytes, Some(range.clone()));
        }
    }
}

/// The ranges of `wanted` read from `file`, each with its index, waiting for
/// the device: all of them asked for first, so that the device reads them at
/// once while they are copied one after another.
fn read_blocking(
    file: &File,
    wanted: Vec<(usize, Range<u64>)>,
) -> std::io::Result<Vec<(usize, Vec<u8>)>> {
    for (_, range) in &wanted {
        advise_willneed(file, range);
    }
    wanted
        .into_iter()
        .map(|(index, range)| {
            let len = usize::try_from(range.end - range.start).map_err(std::io::Error::other)?;
            let mut buffer = vec![0; len];
            read_at(file, &mut buffer, range.start)?;
            Ok((index, buffer))
        })
        .collect()
}

/// A file's ranges as the scheduler requests them.
#[derive(Debug, PartialEq)]
struct Requests {
    /// Ranges close enough together merged into one read.
    merged: Vec<Range<u64>>,
    /// The merged reads cut to the store's largest request: what the
    /// scheduler would issue and record.
    split: Vec<Range<u64>>,
    /// For each piece of `split`, the merged read it is cut from.
    split_owners: Vec<usize>,
    /// For each range asked for, the merged read that holds it and where in
    /// that read.
    slices: Vec<(usize, Range<usize>)>,
}

impl Requests {
    /// `ranges`, which ascend, merged when one starts within `block_size` of
    /// the end of the one before (the scheduler's `is_close_together`), then
    /// split into as few pieces of `max_iop_size` or less as hold them, the
    /// last taking what division leaves over.
    fn of(ranges: &[Range<u64>], block_size: u64, max_iop_size: u64) -> Self {
        let mut merged: Vec<Range<u64>> = Vec::new();
        let mut owners = Vec::with_capacity(ranges.len());
        for range in ranges {
            match merged.last_mut() {
                Some(last) if range.start <= last.end.saturating_add(block_size) => {
                    last.end = last.end.max(range.end);
                }
                _ => merged.push(range.clone()),
            }
            owners.push(merged.len() - 1);
        }
        let slices = ranges
            .iter()
            .zip(owners)
            .map(|(range, owner)| {
                let start = merged[owner].start;
                (
                    owner,
                    (range.start - start) as usize..(range.end - start) as usize,
                )
            })
            .collect();
        let max_iop_size = max_iop_size.max(1);
        let mut split = Vec::with_capacity(merged.len());
        let mut split_owners = Vec::with_capacity(merged.len());
        for (owner, read) in merged.iter().enumerate() {
            if read.is_empty() {
                split.push(read.clone());
                split_owners.push(owner);
                continue;
            }
            let pieces = (read.end - read.start).div_ceil(max_iop_size);
            let piece = (read.end - read.start) / pieces;
            for index in 0..pieces {
                let start = read.start + index * piece;
                let end = if index == pieces - 1 {
                    read.end
                } else {
                    start + piece
                };
                split.push(start..end);
                split_owners.push(owner);
            }
        }
        Self {
            merged,
            split,
            split_owners,
            slices,
        }
    }
}

/// `len` bytes of `file` from `offset`, read into `buf` only if the page
/// cache holds them all.
///
/// Only a full count is served. A short count is a range the page cache holds
/// part of, or one running past the end of the file; kernels 5.9 and 5.10
/// answer a miss with 0. Every error is a refusal rather than an error of its
/// own, because the blocking read that follows reports whatever is really
/// wrong.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub fn read_now(file: &File, buf: &mut Vec<u8>, len: usize, offset: u64) -> Nowait {
    use std::os::fd::AsRawFd;

    // The kernel takes the offset as a signed long, and -1 as "the file's
    // own position".
    let Ok(position) = libc::c_long::try_from(offset) else {
        return Nowait::Refused;
    };
    buf.clear();
    buf.reserve_exact(len);
    let iov = libc::iovec {
        iov_base: buf.as_mut_ptr().cast(),
        iov_len: len,
    };
    // The kernel's preadv2(fd, vec, vlen, pos_l, pos_h, flags); on a 64-bit
    // kernel pos_l holds the whole offset and pos_h is ignored. Every argument
    // is passed as a long, as the variadic syscall() reads them.
    let (vectors, position_high): (libc::c_long, libc::c_long) = (1, 0);
    // SAFETY: `iov` describes `len` bytes of `buf`'s spare capacity, which
    // nothing else refers to during the call and which outlives it, and the
    // kernel writes at most `len` bytes there; `file` keeps the descriptor
    // open for the call.
    let read = unsafe {
        libc::syscall(
            libc::SYS_preadv2,
            libc::c_long::from(file.as_raw_fd()),
            std::ptr::from_ref(&iov),
            vectors,
            position,
            position_high,
            libc::c_long::from(libc::RWF_NOWAIT),
        )
    };
    let errno = (read < 0)
        .then(|| std::io::Error::last_os_error().raw_os_error())
        .flatten();
    let outcome = answer(read, errno, len);
    if outcome == Nowait::Served {
        // SAFETY: the kernel wrote all `len` bytes.
        unsafe { buf.set_len(len) };
    }
    outcome
}

/// What a `preadv2` of `len` bytes that returned `read`, with `errno` set
/// when it failed, came to. A permission hook refuses one file with `EPERM`;
/// a sandbox that forbids the call answers `ENOSYS`, or `EPERM` for every
/// file, which then costs one refused call each.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn answer(read: libc::c_long, errno: Option<i32>, len: usize) -> Nowait {
    if read >= 0 {
        return if usize::try_from(read) == Ok(len) {
            Nowait::Served
        } else {
            Nowait::Refused
        };
    }
    match errno {
        Some(libc::EOPNOTSUPP | libc::EPERM) => Nowait::FileRefuses,
        Some(libc::ENOSYS) => Nowait::ProcessRefuses,
        _ => Nowait::Refused,
    }
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
pub fn read_now(_file: &File, _buf: &mut Vec<u8>, _len: usize, _offset: u64) -> Nowait {
    Nowait::ProcessRefuses
}

/// Local filesystems that serve a read without waiting when the page cache
/// holds it and otherwise refuse it or the flag: ext2 to ext4, xfs, btrfs,
/// tmpfs and overlayfs. A network or FUSE filesystem may ignore the flag and
/// wait on the network.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const LISTED_FILESYSTEMS: [u64; 5] = [0xEF53, 0x5846_5342, 0x9123_683E, 0x0102_1994, 0x794C_7630];

/// The file at `path`, opened to be held if it is on a listed filesystem; a
/// file whose filesystem cannot be told is not held.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn open_listed(path: &str) -> std::io::Result<Option<File>> {
    let file = File::open(path)?;
    Ok(filesystem(&file)
        .is_some_and(|magic| LISTED_FILESYSTEMS.contains(&magic))
        .then_some(file))
}

/// The magic number of the filesystem `file` is on.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn filesystem(file: &File) -> Option<u64> {
    use std::os::fd::AsRawFd;

    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `stat` has room for the struct fstatfs fills, and `file` keeps
    // the descriptor open for the call.
    if unsafe { libc::fstatfs(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: fstatfs returned 0, having filled `stat`.
    let stat = unsafe { stat.assume_init() };
    #[allow(
        clippy::unnecessary_cast,
        reason = "f_type is an i64 on glibc and a u64 on musl"
    )]
    let magic = stat.f_type as u64 & 0xFFFF_FFFF;
    Some(magic)
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
fn open_listed(_path: &str) -> std::io::Result<Option<File>> {
    Ok(None)
}

/// [`OPEN_FILES`], or a sixteenth of the soft descriptor limit when that is
/// fewer.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn capacity() -> u64 {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit fills `limit`.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return 0;
    }
    (limit.rlim_cur / 16).min(OPEN_FILES)
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
fn capacity() -> u64 {
    0
}

#[cfg(all(test, target_os = "linux", target_pointer_width = "64"))]
thread_local! {
    /// How many ranges this thread has asked the kernel to read ahead.
    static ADVISED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Ask the kernel to start reading `range` of `file` into the page cache.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn advise_willneed(file: &File, range: &Range<u64>) {
    use std::os::fd::AsRawFd;

    let (Ok(offset), Ok(len)) = (
        libc::off_t::try_from(range.start),
        libc::off_t::try_from(range.end - range.start),
    ) else {
        return;
    };
    #[cfg(test)]
    ADVISED.with(|advised| advised.set(advised.get() + 1));
    // SAFETY: advice about a descriptor `file` keeps open; it touches no
    // memory of ours. A refusal leaves the reads to wait one after another.
    unsafe { libc::posix_fadvise(file.as_raw_fd(), offset, len, libc::POSIX_FADV_WILLNEED) };
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
fn advise_willneed(_file: &File, _range: &Range<u64>) {}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

/// `buf` filled from `offset`, as Lance's local reader fills it on Windows.
#[cfg(windows)]
fn read_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;

    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => {
                let rest = buf;
                buf = &mut rest[read..];
                offset += read as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_at(_file: &File, _buf: &mut [u8], _offset: u64) -> std::io::Result<()> {
    // No file is held off 64-bit Linux, so only tests get here.
    Err(std::io::ErrorKind::Unsupported.into())
}

/// Whether the page cache serves `len` bytes of the file at `path`
/// without waiting, asked of the kernel directly and never through the
/// code under test, so that code which stopped reading in place cannot
/// talk a test into skipping the assertions that would catch it.
#[cfg(all(test, target_os = "linux", target_pointer_width = "64"))]
pub fn page_cache_serves(path: &str, len: usize) -> bool {
    use std::os::fd::AsRawFd;

    let file = File::open(path).unwrap();
    let mut buf = vec![0u8; len];
    let iov = libc::iovec {
        iov_base: buf.as_mut_ptr().cast(),
        iov_len: len,
    };
    let (vectors, zero): (libc::c_long, libc::c_long) = (1, 0);
    // SAFETY: `iov` describes `buf`, which outlives the call.
    let read = unsafe {
        libc::syscall(
            libc::SYS_preadv2,
            libc::c_long::from(file.as_raw_fd()),
            std::ptr::from_ref(&iov),
            vectors,
            zero,
            zero,
            libc::c_long::from(libc::RWF_NOWAIT),
        )
    };
    usize::try_from(read) == Ok(len)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use futures::FutureExt;
    use lance_core::utils::io_stats::IoStatsRecorder;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
    use lance_io::utils::CachedFileSize;

    use super::*;

    /// Bytes of a test file, each different from the one before.
    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|index| (index % 251) as u8).collect()
    }

    const FILE_BYTES: usize = 32768;

    /// Two vectors side by side, read as one, and one far from them.
    fn ranges() -> Vec<Range<u64>> {
        vec![0..512, 512..1024, 16384..16896]
    }

    /// A file of `pattern(FILE_BYTES)` named `name` in `dir`, and its path.
    fn write(dir: &TempStrDir, name: &str) -> String {
        let path = format!("{}/{name}", dir.as_str());
        std::fs::write(&path, pattern(FILE_BYTES)).unwrap();
        path
    }

    fn hold(path: &str) -> Arc<HeldFile> {
        let file = File::open(path).unwrap();
        let size = file.metadata().unwrap().len();
        Arc::new(HeldFile {
            file,
            size,
            refuses_nowait: AtomicBool::new(false),
        })
    }

    /// What [`read`] reads with, apart from the file: a local store, a pool
    /// of its own and a budget.
    struct Bench {
        store: ObjectStore,
        path: Path,
        pool: Arc<Descriptors>,
        budget: AtomicU64,
    }

    impl Bench {
        fn new(budget: u64) -> Self {
            Self {
                store: ObjectStore::local(),
                path: Path::from("data"),
                pool: Descriptors::new(4, IDLE, BACK_OFF, open_listed),
                budget: AtomicU64::new(budget),
            }
        }

        fn context<'a>(
            &'a self,
            io_stats: Option<&'a IoStats>,
            read_now: ReadNow,
        ) -> ReadContext<'a> {
            ReadContext {
                store: &self.store,
                path: &self.path,
                io_stats,
                pool: &self.pool,
                read_now,
                budget: &self.budget,
            }
        }
    }

    fn assert_bytes(bytes: &ReadBytes, ranges: &[Range<u64>]) {
        let file = pattern(FILE_BYTES);
        let expected = ranges
            .iter()
            .map(|range| &file[range.start as usize..range.end as usize]);
        assert!(bytes.chunks().eq(expected));
    }

    fn serve(file: &File, buf: &mut Vec<u8>, len: usize, offset: u64) -> Nowait {
        buf.clear();
        buf.resize(len, 0);
        read_at(file, buf, offset).unwrap();
        Nowait::Served
    }

    fn refuse(_file: &File, _buf: &mut Vec<u8>, _len: usize, _offset: u64) -> Nowait {
        Nowait::Refused
    }

    fn refuse_the_flag(_file: &File, _buf: &mut Vec<u8>, _len: usize, _offset: u64) -> Nowait {
        Nowait::FileRefuses
    }

    fn refuse_the_call(_file: &File, _buf: &mut Vec<u8>, _len: usize, _offset: u64) -> Nowait {
        Nowait::ProcessRefuses
    }

    /// Serves the file's first page only.
    fn serve_the_first_page(file: &File, buf: &mut Vec<u8>, len: usize, offset: u64) -> Nowait {
        if offset < 4096 {
            serve(file, buf, len, offset)
        } else {
            Nowait::Refused
        }
    }

    fn never(_file: &File, _buf: &mut Vec<u8>, _len: usize, _offset: u64) -> Nowait {
        panic!("a range was read on the asking thread");
    }

    /// Refuses the file's first page, and panics if asked again.
    fn refuse_once(refusal: Nowait, offset: u64) -> Nowait {
        assert!(offset < 4096, "asked again after a refusal");
        refusal
    }

    fn refused_once(_file: &File, _buf: &mut Vec<u8>, _len: usize, offset: u64) -> Nowait {
        refuse_once(Nowait::Refused, offset)
    }

    fn flag_refused_once(_file: &File, _buf: &mut Vec<u8>, _len: usize, offset: u64) -> Nowait {
        refuse_once(Nowait::FileRefuses, offset)
    }

    fn call_refused_once(_file: &File, _buf: &mut Vec<u8>, _len: usize, offset: u64) -> Nowait {
        refuse_once(Nowait::ProcessRefuses, offset)
    }

    /// Reads the page cache serves are made on the thread that asks: polled
    /// once outside any runtime, where a trip to the blocking pool would
    /// panic, the read is done.
    #[test]
    fn reads_the_page_cache_serves_stay_on_the_asking_thread() {
        let dir = TempStrDir::default();
        let held = hold(&write(&dir, "data"));
        let bench = Bench::new(IN_PLACE_BYTES);
        let bytes = read(held, &ranges(), &bench.context(None, serve))
            .now_or_never()
            .expect("a read served in place waits for nothing")
            .unwrap();
        assert_bytes(&bytes, &ranges());
        let reads = LocalReads {
            in_place: 2,
            handed_off: 0,
            trips: 0,
        };
        assert_eq!(bytes.reads, reads);
    }

    /// Ranges merged across a gap of less than a block each keep their own
    /// bytes, not the gap's.
    #[test]
    fn ranges_merged_across_a_gap_keep_their_own_bytes() {
        let dir = TempStrDir::default();
        let held = hold(&write(&dir, "data"));
        let bench = Bench::new(IN_PLACE_BYTES);
        let gapped = [0..512, 1024..1536, 16384..16896];
        let bytes = read(held, &gapped, &bench.context(None, serve))
            .now_or_never()
            .unwrap()
            .unwrap();
        assert_bytes(&bytes, &gapped);
        assert_eq!(bytes.reads.in_place, 2);
    }

    /// Reads refused on the asking thread are made on the blocking pool, all
    /// of a file's in one trip, and those served before stay where they were.
    #[rstest::rstest]
    #[case::all_refused(refuse, 0, 2)]
    #[case::one_refused(serve_the_first_page, 1, 1)]
    #[tokio::test]
    async fn refused_reads_go_to_the_blocking_pool_in_one_trip(
        #[case] read_now: ReadNow,
        #[case] in_place: usize,
        #[case] handed_off: usize,
    ) {
        let dir = TempStrDir::default();
        let held = hold(&write(&dir, "data"));
        let bench = Bench::new(IN_PLACE_BYTES);
        let bytes = read(held, &ranges(), &bench.context(None, read_now))
            .await
            .unwrap();
        assert_bytes(&bytes, &ranges());
        let reads = LocalReads {
            in_place,
            handed_off,
            trips: 1,
        };
        assert_eq!(bytes.reads, reads);
    }

    /// Once a read of a file is refused, however, the file's other ranges go
    /// to the blocking pool with it without being asked for, and the refused
    /// read spends none of the budget.
    #[rstest::rstest]
    #[case::this_time(refused_once)]
    #[case::by_the_filesystem(flag_refused_once)]
    #[case::by_the_kernel(call_refused_once)]
    #[tokio::test]
    async fn a_refusal_hands_the_rest_of_the_file_to_the_blocking_pool(#[case] read_now: ReadNow) {
        let dir = TempStrDir::default();
        let held = hold(&write(&dir, "data"));
        let bench = Bench::new(IN_PLACE_BYTES);
        let bytes = read(held, &ranges(), &bench.context(None, read_now))
            .await
            .unwrap();
        assert_bytes(&bytes, &ranges());
        let reads = LocalReads {
            in_place: 0,
            handed_off: 2,
            trips: 1,
        };
        assert_eq!(bytes.reads, reads);
        assert_eq!(bench.budget.load(Ordering::Relaxed), IN_PLACE_BYTES);
    }

    /// A refusal this time is not remembered: the file's next read asks
    /// again, and is served.
    #[tokio::test]
    async fn a_refusal_this_time_is_not_remembered() {
        let dir = TempStrDir::default();
        let held = hold(&write(&dir, "data"));
        let bench = Bench::new(IN_PLACE_BYTES);
        read(held.clone(), &ranges(), &bench.context(None, refuse))
            .await
            .unwrap();
        let again = read(held, &ranges(), &bench.context(None, serve))
            .await
            .unwrap();
        assert_eq!(again.reads.in_place, 2);
    }

    /// A file whose filesystem refuses the flag is not asked again: its
    /// later reads go straight to the blocking pool, while other files are
    /// still read in place.
    #[tokio::test]
    async fn a_filesystem_refusing_the_flag_is_remembered_for_its_file() {
        let dir = TempStrDir::default();
        let held = hold(&write(&dir, "data"));
        let bench = Bench::new(IN_PLACE_BYTES);
        let refused = LocalReads {
            in_place: 0,
            handed_off: 2,
            trips: 1,
        };
        let first = read(
            held.clone(),
            &ranges(),
            &bench.context(None, refuse_the_flag),
        )
        .await
        .unwrap();
        assert_eq!(first.reads, refused);
        let again = read(held, &ranges(), &bench.context(None, never))
            .await
            .unwrap();
        assert_bytes(&again, &ranges());
        assert_eq!(again.reads, refused);
        let other = hold(&write(&dir, "other"));
        let other = read(other, &ranges(), &bench.context(None, serve))
            .await
            .unwrap();
        assert_eq!(other.reads.in_place, 2);
    }

    /// A kernel or sandbox refusing the call is remembered by the pool: no
    /// file is read in place again, while a pool of its own still reads in
    /// place.
    #[tokio::test]
    async fn a_kernel_refusing_the_call_is_remembered_by_the_pool() {
        let dir = TempStrDir::default();
        let bench = Bench::new(IN_PLACE_BYTES);
        let refused = LocalReads {
            in_place: 0,
            handed_off: 2,
            trips: 1,
        };
        let first = read(
            hold(&write(&dir, "data")),
            &ranges(),
            &bench.context(None, refuse_the_call),
        )
        .await
        .unwrap();
        assert_eq!(first.reads, refused);
        let other = read(
            hold(&write(&dir, "other")),
            &ranges(),
            &bench.context(None, never),
        )
        .await
        .unwrap();
        assert_bytes(&other, &ranges());
        assert_eq!(other.reads, refused);
        let elsewhere = Bench::new(IN_PLACE_BYTES);
        let fresh = read(
            hold(&write(&dir, "fresh")),
            &ranges(),
            &elsewhere.context(None, serve),
        )
        .await
        .unwrap();
        assert_eq!(fresh.reads.in_place, 2);
    }

    /// The budget is the call's, not a file's: once its files have read it
    /// up in place, their other reads are made on the blocking pool.
    #[tokio::test]
    async fn the_budget_is_shared_by_a_calls_files() {
        let dir = TempStrDir::default();
        let bench = Bench::new(1024);
        let first = read(
            hold(&write(&dir, "data")),
            &ranges(),
            &bench.context(None, serve),
        )
        .await
        .unwrap();
        let spent = LocalReads {
            in_place: 1,
            handed_off: 1,
            trips: 1,
        };
        assert_eq!(first.reads, spent);
        let second = read(
            hold(&write(&dir, "other")),
            &ranges(),
            &bench.context(None, serve),
        )
        .await
        .unwrap();
        assert_bytes(&second, &ranges());
        let none_left = LocalReads {
            in_place: 0,
            handed_off: 2,
            trips: 1,
        };
        assert_eq!(second.reads, none_left);
    }

    /// A file that is not read in place spends none of the call's budget,
    /// which is left to the files that are.
    #[tokio::test]
    async fn a_file_not_read_in_place_spends_none_of_the_budget() {
        let dir = TempStrDir::default();
        let bench = Bench::new(1024);
        let refusing = hold(&write(&dir, "data"));
        refusing.refuses_nowait.store(true, Ordering::Relaxed);
        read(refusing, &ranges(), &bench.context(None, never))
            .await
            .unwrap();
        let served = read(
            hold(&write(&dir, "other")),
            &ranges(),
            &bench.context(None, serve),
        )
        .await
        .unwrap();
        assert_eq!(served.reads.in_place, 1);
    }

    /// A read the blocking pool cannot make is an error, and the tracker
    /// records no read for it.
    #[tokio::test]
    async fn a_read_past_the_end_of_the_file_is_an_error() {
        let dir = TempStrDir::default();
        let bench = Bench::new(IN_PLACE_BYTES);
        bench.store.io_tracker().incremental_stats();
        let past = [40000..40512];
        let error = read(
            hold(&write(&dir, "data")),
            &past,
            &bench.context(None, refuse),
        )
        .await
        .err()
        .unwrap();
        assert!(matches!(error, Error::IO { .. }), "{error:?}");
        assert_eq!(bench.store.io_tracker().incremental_stats().read_iops, 0);
    }

    /// When the trip fails, the tracker still records the reads served in
    /// place, each read by its own outcome, while the query's statistics
    /// count every request as it was made.
    #[tokio::test]
    async fn the_tracker_records_each_read_by_its_own_outcome() {
        let dir = TempStrDir::default();
        let bench = Bench::new(IN_PLACE_BYTES);
        let io_stats = IoStats::new();
        bench.store.io_tracker().incremental_stats();
        let ranges = [0..512, 40000..40512];
        let context = bench.context(Some(&io_stats), serve_the_first_page);
        read(hold(&write(&dir, "data")), &ranges, &context)
            .await
            .err()
            .unwrap();
        let tracked = bench.store.io_tracker().incremental_stats();
        assert_eq!((tracked.read_iops, tracked.read_bytes), (1, 512));
        assert_eq!(io_stats.snapshot().iops, 2);
    }

    /// Records the ranges of every request made through it.
    #[derive(Debug, Default)]
    struct Requested(std::sync::Mutex<Vec<Vec<Range<u64>>>>);

    impl IoStatsRecorder for Requested {
        fn record_request(&self, ranges: &[Range<u64>]) {
            self.0.lock().unwrap().push(ranges.to_vec());
        }
    }

    /// Ranges are merged and split as the scheduler merges and splits them:
    /// side by side, a block apart to the byte and a byte more, and far
    /// apart; and a read longer than the store's largest request is split
    /// into the fewest pieces, the last taking what division leaves over.
    #[tokio::test]
    async fn requests_are_those_the_scheduler_makes() {
        let store = Arc::new(ObjectStore::memory());
        let path = Path::from("data");
        let len = 1 << 20;
        store.put(&path, &vec![0; len]).await.unwrap();
        let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
        let block = store.block_size() as u64;
        for ranges in [
            vec![0..512, 512..1024],
            vec![0..512, 512 + block..1024 + block],
            vec![0..512, 513 + block..1025 + block],
            vec![0..512, 300_000..300_512],
            vec![0..512, 512..1024, 70_000..70_512, 600_000..600_512],
        ] {
            let requested = Arc::new(Requested::default());
            let file = scheduler
                .open_file(&path, &CachedFileSize::new(len as u64))
                .await
                .unwrap()
                .with_io_stats(requested.clone());
            file.submit_request(ranges.clone(), 0).await.unwrap();
            let requests = Requests::of(&ranges, block, store.max_iop_size());
            assert_eq!(
                *requested.0.lock().unwrap(),
                vec![requests.split],
                "{ranges:?}"
            );
        }
        for (read, split) in [
            (0..250, vec![0..83, 83..166, 166..250]),
            (0..200, vec![0..100, 100..200]),
            (0..100, vec![0..100]),
        ] {
            let requests = Requests::of(std::slice::from_ref(&read), 4096, 100);
            assert_eq!(requests.merged, vec![read]);
            assert_eq!(requests.split_owners, vec![0; split.len()]);
            assert_eq!(requests.split, split);
        }
    }

    /// A read counts in the query's statistics and in the store's tracker
    /// what the scheduler and Lance's local reader would: one request of the
    /// merged reads, and each read.
    #[tokio::test]
    async fn reads_are_counted_where_the_scheduler_counts_them() {
        let dir = TempStrDir::default();
        let bench = Bench::new(IN_PLACE_BYTES);
        let io_stats = IoStats::new();
        bench.store.io_tracker().incremental_stats();
        read(
            hold(&write(&dir, "data")),
            &ranges(),
            &bench.context(Some(&io_stats), serve),
        )
        .await
        .unwrap();
        let counted = io_stats.snapshot();
        assert_eq!(
            (counted.requests, counted.iops, counted.bytes_read),
            (1, 2, 1536)
        );
        let tracked = bench.store.io_tracker().incremental_stats();
        assert_eq!((tracked.read_iops, tracked.read_bytes), (2, 1536));
    }

    /// A read longer than the store's largest request is read whole and
    /// counted in the pieces the scheduler would request: from a sparse file
    /// a block longer than the largest request.
    #[tokio::test]
    async fn a_read_longer_than_the_largest_request_is_counted_in_pieces() {
        let bench = Bench::new(u64::MAX);
        let largest = bench.store.max_iop_size();
        if largest > 64 << 20 {
            return;
        }
        let dir = TempStrDir::default();
        let path = format!("{}/sparse", dir.as_str());
        let len = largest + 4096;
        File::create(&path).unwrap().set_len(len).unwrap();
        let io_stats = IoStats::new();
        bench.store.io_tracker().incremental_stats();
        let bytes = read(
            hold(&path),
            &[0..len],
            &bench.context(Some(&io_stats), serve),
        )
        .await
        .unwrap();
        assert_eq!(
            bytes.chunks().map(<[u8]>::len).collect::<Vec<_>>(),
            vec![len as usize]
        );
        let counted = io_stats.snapshot();
        assert_eq!(
            (counted.requests, counted.iops, counted.bytes_read),
            (1, 2, len)
        );
        let tracked = bench.store.io_tracker().incremental_stats();
        assert_eq!((tracked.read_iops, tracked.read_bytes), (2, len));
    }

    fn open_slowly(path: &str) -> std::io::Result<Option<File>> {
        std::thread::sleep(Duration::from_millis(100));
        File::open(path).map(Some)
    }

    fn open_elsewhere(_path: &str) -> std::io::Result<Option<File>> {
        Ok(None)
    }

    fn open_any(path: &str) -> std::io::Result<Option<File>> {
        File::open(path).map(Some)
    }

    static FAILED_OPENS: AtomicUsize = AtomicUsize::new(0);

    fn open_failing(_path: &str) -> std::io::Result<Option<File>> {
        FAILED_OPENS.fetch_add(1, Ordering::Relaxed);
        Err(std::io::Error::other("too many open files"))
    }

    /// Queries missing on one file while it is being opened open it once,
    /// and a file held is held with its size.
    #[tokio::test]
    async fn queries_missing_on_one_file_open_it_once() {
        let dir = TempStrDir::default();
        let path = write(&dir, "data");
        let pool = Descriptors::new(4, IDLE, BACK_OFF, open_slowly);
        let opening = pool.open_in_background(path.clone()).unwrap();
        assert!(pool.open_in_background(path.clone()).is_none());
        opening.await.unwrap();
        assert_eq!(pool.get(&path).unwrap().size(), FILE_BYTES as u64);
    }

    /// A failed open holds nothing, and its path is opened again once the
    /// back-off has passed, not before.
    #[tokio::test]
    async fn a_failed_open_is_tried_again_after_the_back_off() {
        let pool = Descriptors::new(4, IDLE, Duration::from_millis(100), open_failing);
        pool.open_in_background("data".to_string())
            .unwrap()
            .await
            .unwrap();
        assert!(pool.get("data").is_none());
        assert!(pool.open_in_background("data".to_string()).is_none());
        std::thread::sleep(Duration::from_millis(300));
        pool.open_in_background("data".to_string())
            .unwrap()
            .await
            .unwrap();
        assert_eq!(FAILED_OPENS.load(Ordering::Relaxed), 2);
    }

    /// A file on a filesystem off the list marks its directory, and is not
    /// held.
    #[tokio::test]
    async fn a_file_off_the_list_marks_its_directory() {
        let dir = TempStrDir::default();
        let pool = Descriptors::new(4, IDLE, BACK_OFF, open_elsewhere);
        let path = write(&dir, "data");
        pool.open_in_background(path.clone())
            .unwrap()
            .await
            .unwrap();
        assert!(pool.get(&path).is_none());
        assert!(pool.elsewhere(&path));
        assert!(pool.elsewhere(&format!("{}/other", dir.as_str())));
        assert!(!pool.elsewhere("/somewhere/else/data"));
    }

    /// A pool that may hold no files opens none.
    #[tokio::test]
    async fn a_pool_of_no_files_opens_none() {
        let dir = TempStrDir::default();
        let pool = Descriptors::new(0, IDLE, BACK_OFF, open_any);
        assert!(pool.open_in_background(write(&dir, "data")).is_none());
    }

    /// A full pool keeps the files read most rather than letting a newcomer
    /// push one out.
    #[tokio::test]
    async fn the_pool_keeps_its_hot_files() {
        let dir = TempStrDir::default();
        let paths = ["a", "b", "c"].map(|name| write(&dir, name));
        let pool = Descriptors::new(2, IDLE, BACK_OFF, open_any);
        // The pool counts how often files are read only once it is half
        // full, so each file is held and settled before it is read.
        for path in &paths[..2] {
            pool.open_in_background(path.clone())
                .unwrap()
                .await
                .unwrap();
            pool.settle();
        }
        for _ in 0..8 {
            for path in &paths[..2] {
                assert!(pool.get(path).is_some());
            }
        }
        pool.settle();
        pool.open_in_background(paths[2].clone())
            .unwrap()
            .await
            .unwrap();
        pool.settle();
        assert!(paths[..2].iter().all(|path| pool.get(path).is_some()));
        assert!(pool.get(&paths[2]).is_none());
    }

    /// A file left unread is closed by the pool's own thread, while nothing
    /// reads through the pool at all.
    #[tokio::test]
    async fn an_idle_file_is_closed_while_nothing_reads() {
        let dir = TempStrDir::default();
        let path = write(&dir, "data");
        let pool = Descriptors::new(2, Duration::from_millis(150), BACK_OFF, open_any);
        pool.open_in_background(path.clone())
            .unwrap()
            .await
            .unwrap();
        let held = Arc::downgrade(&pool.get(&path).unwrap());
        std::thread::sleep(Duration::from_millis(700));
        assert!(held.upgrade().is_none(), "the idle file is still open");
    }

    /// The real call on a file the page cache holds serves every read in
    /// place, polled once outside any runtime; on a filesystem that refuses
    /// the flag, every read goes to the blocking pool instead.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn the_kernel_serves_what_the_page_cache_holds() {
        let dir = TempStrDir::default();
        let path = write(&dir, "data");
        let bench = Bench::new(IN_PLACE_BYTES);
        if page_cache_serves(&path, FILE_BYTES) {
            let bytes = read(hold(&path), &ranges(), &bench.context(None, read_now))
                .now_or_never()
                .expect("a read the page cache serves waits for nothing")
                .unwrap();
            assert_bytes(&bytes, &ranges());
            assert_eq!(bytes.reads.in_place, 2);
        } else {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let bytes = runtime
                .block_on(read(hold(&path), &ranges(), &bench.context(None, read_now)))
                .unwrap();
            assert_bytes(&bytes, &ranges());
            assert_eq!(bytes.reads.handed_off, 2);
        }
    }

    /// Drop the pages of `file` the page cache holds, once they are on disk.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    fn drop_pages(file: &File) {
        use std::os::fd::AsRawFd;

        file.sync_all().unwrap();
        // SAFETY: advice about a descriptor `file` keeps open.
        let advised =
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        assert_eq!(advised, 0);
    }

    /// A read of pages the page cache has dropped is refused, this time and
    /// not for good, and made on the blocking pool. On a machine whose
    /// filesystem answers the flag and refuses such a read, which a sibling
    /// file is asked first: asking the file under test would start the
    /// readahead that puts its pages back.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[tokio::test]
    async fn a_read_of_dropped_pages_goes_to_the_blocking_pool() {
        let dir = TempStrDir::default();
        let probe = write(&dir, "probe");
        if !page_cache_serves(&probe, 4096) {
            return;
        }
        drop_pages(&File::open(&probe).unwrap());
        if page_cache_serves(&probe, 4096) {
            return;
        }
        let held = hold(&write(&dir, "data"));
        drop_pages(&held.file);
        let bench = Bench::new(IN_PLACE_BYTES);
        let bytes = read(held.clone(), &ranges(), &bench.context(None, read_now))
            .await
            .unwrap();
        assert_bytes(&bytes, &ranges());
        assert!(bytes.reads.handed_off > 0, "{:?}", bytes.reads);
        assert!(!held.refuses_nowait.load(Ordering::Relaxed));
    }

    /// The kernel is asked to read ahead every range that goes to the
    /// blocking pool, so that the device reads them at once.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn the_kernel_is_asked_to_read_ahead_what_goes_to_the_blocking_pool() {
        let dir = TempStrDir::default();
        let held = hold(&write(&dir, "data"));
        let before = ADVISED.with(std::cell::Cell::get);
        let read = read_blocking(&held.file, vec![(0, 0..512), (1, 16384..16896)]).unwrap();
        assert_eq!(ADVISED.with(std::cell::Cell::get) - before, 2);
        let file = pattern(FILE_BYTES);
        assert_eq!(read[0], (0, file[0..512].to_vec()));
        assert_eq!(read[1], (1, file[16384..16896].to_vec()));
    }

    /// An offset the kernel would read as a signed long gone negative is
    /// refused rather than asked.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn an_offset_past_a_long_is_refused() {
        let dir = TempStrDir::default();
        let file = File::open(write(&dir, "data")).unwrap();
        let mut buf = Vec::new();
        assert_eq!(read_now(&file, &mut buf, 8, u64::MAX), Nowait::Refused);
        assert!(buf.is_empty());
    }

    /// A read running past the end of the file is never served: the kernel
    /// answers with fewer bytes, or none at the end itself.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn a_read_running_past_the_end_is_not_served() {
        let dir = TempStrDir::default();
        let file = File::open(write(&dir, "data")).unwrap();
        for offset in [FILE_BYTES as u64 - 512, FILE_BYTES as u64] {
            let mut buf = Vec::new();
            assert_ne!(read_now(&file, &mut buf, 1024, offset), Nowait::Served);
            assert!(buf.is_empty());
        }
    }

    /// Only a full count is served; a short one, the 0 kernels 5.9 and 5.10
    /// answer a miss with, and every error but those that say the flag or
    /// the call is refused for good are refused this time.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn the_kernels_answers_are_classified() {
        for (read, errno, answered) in [
            (512, None, Nowait::Served),
            (511, None, Nowait::Refused),
            (0, None, Nowait::Refused),
            (-1, Some(libc::EAGAIN), Nowait::Refused),
            (-1, Some(libc::EINTR), Nowait::Refused),
            (-1, Some(libc::EINVAL), Nowait::Refused),
            (-1, None, Nowait::Refused),
            (-1, Some(libc::EOPNOTSUPP), Nowait::FileRefuses),
            (-1, Some(libc::EPERM), Nowait::FileRefuses),
            (-1, Some(libc::ENOSYS), Nowait::ProcessRefuses),
        ] {
            assert_eq!(answer(read, errno, 512), answered, "{read} {errno:?}");
        }
    }

    /// Only files of a plain local store are read here: not those of the
    /// io_uring store, which has a reader of its own, nor of a local store
    /// read through the object store API, nor of memory.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[tokio::test]
    async fn only_files_of_a_plain_local_store_are_read_here() {
        let dir = TempStrDir::default();
        let (local, _) = ObjectStore::from_uri(dir.as_str()).await.unwrap();
        assert_eq!(reads_in_place(&local), *SWITCHED_ON);
        for uri in [
            format!("file+uring://{}", dir.as_str()),
            format!("file-object-store://{}", dir.as_str()),
            "memory://".to_string(),
        ] {
            let (store, _) = ObjectStore::from_uri(&uri).await.unwrap();
            assert!(!reads_in_place(&store), "{uri}");
        }
    }

    /// A file is held only on a listed filesystem: procfs, which every Linux
    /// has, is not one; the test directory's filesystem is held when listed.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn only_files_on_listed_filesystems_are_held() {
        let dir = TempStrDir::default();
        let data = write(&dir, "data");
        for path in [data.as_str(), "/proc/self/stat"] {
            let magic = filesystem(&File::open(path).unwrap()).unwrap();
            let listed = LISTED_FILESYSTEMS.contains(&magic);
            assert_eq!(open_listed(path).unwrap().is_some(), listed, "{path}");
        }
        let proc = filesystem(&File::open("/proc/self/stat").unwrap()).unwrap();
        assert!(!LISTED_FILESYSTEMS.contains(&proc));
    }

    /// The filesystems listed are those libc names.
    #[cfg(all(target_os = "linux", target_pointer_width = "64", target_env = "gnu"))]
    #[test]
    fn the_listed_filesystems_are_those_libc_names() {
        let named = [
            libc::EXT4_SUPER_MAGIC,
            libc::XFS_SUPER_MAGIC,
            libc::BTRFS_SUPER_MAGIC,
            libc::TMPFS_MAGIC,
            libc::OVERLAYFS_SUPER_MAGIC,
        ]
        .map(|magic| magic as u64 & 0xFFFF_FFFF);
        assert_eq!(LISTED_FILESYSTEMS, named);
    }

    /// The pool holds a sixteenth of the soft descriptor limit, and no more
    /// than [`OPEN_FILES`] files.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn the_pool_holds_a_sixteenth_of_the_descriptor_limit() {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: getrlimit fills `limit`.
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
            0
        );
        assert_eq!(capacity(), (limit.rlim_cur / 16).min(64));
    }
}
