// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_file::reader::FileReaderOptions;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use super::FragmentOpenCache;

/// A per-scan cache of `ScanScheduler`s for non-default bases.
///
/// Files that live on an additional base (e.g. shallow clones) each need a
/// scheduler for their base's object store. Without this cache every opened
/// base file builds its own scheduler, so the scheduler back-pressure budget
/// scales with the number of files opened and can exhaust memory. Sharing one
/// scheduler per base for the lifetime of a scan bounds that budget by the
/// number of bases instead.
///
/// The cache lives on [`FragReadConfig`], so it shares the scan's lifetime and
/// its `io_buffer_size` — matching the primary scheduler threaded in through
/// [`FragReadConfig::scan_scheduler`] rather than a longer, dataset-wide scope.
#[derive(Clone)]
pub struct BaseSchedulers {
    /// `None` sizes each scheduler for the max bandwidth of its base's store.
    io_buffer_size: Option<u64>,
    schedulers: Arc<std::sync::Mutex<HashMap<u32, Arc<ScanScheduler>>>>,
}

impl std::fmt::Debug for BaseSchedulers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseSchedulers")
            .field("io_buffer_size", &self.io_buffer_size)
            .finish_non_exhaustive()
    }
}

impl BaseSchedulers {
    /// Create an empty cache whose schedulers will use `io_buffer_size`, the
    /// same budget as the scan's primary scheduler.
    pub fn new(io_buffer_size: u64) -> Self {
        Self {
            io_buffer_size: Some(io_buffer_size),
            schedulers: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Create an empty cache whose schedulers are sized for the max bandwidth
    /// of their base's object store, for scans whose primary scheduler uses
    /// [`SchedulerConfig::max_bandwidth`].
    pub fn max_bandwidth() -> Self {
        Self {
            io_buffer_size: None,
            schedulers: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Return the scheduler for `base_id`, building it once from `object_store`
    /// and reusing it for every later file read from the same base.
    pub(super) fn get_or_create(
        &self,
        base_id: u32,
        object_store: Arc<ObjectStore>,
    ) -> Arc<ScanScheduler> {
        let mut schedulers = self.schedulers.lock().unwrap();
        schedulers
            .entry(base_id)
            .or_insert_with(|| {
                let config = match self.io_buffer_size {
                    Some(io_buffer_size) => SchedulerConfig::new(io_buffer_size),
                    None => SchedulerConfig::max_bandwidth(&object_store),
                };
                ScanScheduler::new(object_store, config)
            })
            .clone()
    }

    /// Number of distinct bases a scheduler has been built for.
    #[doc(hidden)]
    pub fn num_cached_bases(&self) -> usize {
        self.schedulers.lock().unwrap().len()
    }
}

/// Options for projecting system columns, applying deletions, and scheduling reads.
#[derive(Debug, Default, Clone)]
pub struct FragReadConfig {
    // Add the row id column
    pub with_row_id: bool,
    // Add the row address column
    pub with_row_address: bool,
    /// Retain deleted rows in physical reads instead of filtering them out.
    ///
    /// Deleted rows have null `_rowid` and `_rowaddr` values when those columns
    /// are enabled. Payload and row-version columns are unchanged.
    /// [`FileFragment::open`](super::FileFragment::open) requires `with_row_id` or
    /// `with_row_address` when this is true. Defaults to false.
    pub make_deletions_null: bool,
    // Add the last updated at version column
    pub with_row_last_updated_at_version: bool,
    // Add the created at version column
    pub with_row_created_at_version: bool,
    /// The scan scheduler to use for reading data files.
    ///
    /// This should be specified if multiple readers are being used in
    /// an operation
    pub scan_scheduler: Option<Arc<ScanScheduler>>,
    /// The default scan priority to use for reading data files
    ///
    /// Only used if `scan_scheduler` is provided
    ///
    /// The overall priority for reads will be
    ///
    /// operation_priority: u32 | reader_priority: u32 | file_position: u64
    pub reader_priority: Option<u32>,
    /// File reader options to use when reading data files.
    pub file_reader_options: Option<FileReaderOptions>,
    /// Per-scan cache of schedulers for non-default bases.
    ///
    /// The scan sets this so every base file it opens shares one scheduler per
    /// base. When absent, a base file falls back to building its own scheduler.
    pub base_schedulers: Option<BaseSchedulers>,
    /// Read rows as if the fragment had no deletions: the deletion file is not
    /// loaded and deleted rows come back with their stored values. Only for a
    /// caller that knows the rows it asks for are live.
    pub ignore_deletions: bool,
    /// Per-scan cache of reader-open state, shared by the scan's fragments.
    ///
    /// Only used when opening with the projection the cache was built for.
    pub open_cache: Option<Arc<FragmentOpenCache>>,
}

impl FragReadConfig {
    pub fn with_row_id(mut self, value: bool) -> Self {
        self.with_row_id = value;
        self
    }

    pub fn with_ignore_deletions(mut self, value: bool) -> Self {
        self.ignore_deletions = value;
        self
    }

    pub fn with_row_address(mut self, value: bool) -> Self {
        self.with_row_address = value;
        self
    }

    /// Configure deletion markers instead of filtering deleted rows.
    ///
    /// See [`Self::make_deletions_null`]. For physical selections, use
    /// [`read_all`](super::FragmentReader::read_all),
    /// [`take_range`](super::FragmentReader::take_range), or
    /// [`read_ranges`](super::FragmentReader::read_ranges).
    /// [`read_range`](super::FragmentReader::read_range) interprets offsets
    /// as logical row offsets.
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table_runtime::{FileFragment, FragReadConfig};
    /// # async fn example(fragment: &FileFragment) -> Result<()> {
    /// let config = FragReadConfig::default()
    ///     .with_row_address(true)
    ///     .with_make_deletions_null(true);
    /// let reader = fragment.open(fragment.schema(), config).await?;
    /// // Read physical offsets, with null addresses marking deleted rows.
    /// let batches = reader.take_range(0..1, 1).await?;
    /// # let _ = batches;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_make_deletions_null(mut self, value: bool) -> Self {
        self.make_deletions_null = value;
        self
    }

    pub fn with_row_last_updated_at_version(mut self, value: bool) -> Self {
        self.with_row_last_updated_at_version = value;
        self
    }

    pub fn with_row_created_at_version(mut self, value: bool) -> Self {
        self.with_row_created_at_version = value;
        self
    }

    pub fn has_system_cols(&self) -> bool {
        self.with_row_id
            || self.with_row_address
            || self.with_row_last_updated_at_version
            || self.with_row_created_at_version
    }

    pub fn with_scan_scheduler(mut self, value: Arc<ScanScheduler>) -> Self {
        self.scan_scheduler = Some(value);
        self
    }

    pub fn with_reader_priority(mut self, value: u32) -> Self {
        self.reader_priority = Some(value);
        self
    }

    pub fn with_file_reader_options(mut self, value: FileReaderOptions) -> Self {
        self.file_reader_options = Some(value);
        self
    }

    pub fn with_base_schedulers(mut self, value: BaseSchedulers) -> Self {
        self.base_schedulers = Some(value);
        self
    }

    pub fn with_open_cache(mut self, value: Arc<FragmentOpenCache>) -> Self {
        self.open_cache = Some(value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_base_schedulers_shares_one_per_base() {
        let cache = BaseSchedulers::new(4 * 1024 * 1024);
        let store = Arc::new(ObjectStore::local());

        // Repeated resolutions of the same base reuse one scheduler, so opening
        // many files from a base does not multiply scheduler budgets.
        let first = cache.get_or_create(1, store.clone());
        let second = cache.get_or_create(1, store.clone());
        assert!(
            Arc::ptr_eq(&first, &second),
            "same base must reuse one scheduler"
        );

        // A different base gets its own scheduler.
        let other = cache.get_or_create(2, store.clone());
        assert!(
            !Arc::ptr_eq(&first, &other),
            "different bases must not share a scheduler"
        );

        // Clones of the cache (as threaded per fragment) share the same map.
        let cloned = cache.clone();
        let via_clone = cloned.get_or_create(1, store);
        assert!(
            Arc::ptr_eq(&first, &via_clone),
            "cache clones must share the per-scan scheduler map"
        );
        assert_eq!(cache.num_cached_bases(), 2);
    }
}
