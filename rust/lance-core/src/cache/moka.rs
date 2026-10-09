// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;

use crate::Result;
use crate::deepsize::Context;
use crate::error::CloneableError;

use super::backend::{CacheBackend, CacheEntry, CacheLoader};
use super::backend_metrics::BackendCounters;
use super::{
    CacheBackendDiagnostics, CacheBackendKind, CacheCodec, CacheLoadOutcome, CacheOccupancyByType,
    CacheOperationContext, CacheSnapshotMode, CacheTypeOccupancy, InternalCacheKey,
};

/// Internal record stored in the moka cache.
#[derive(Clone, Debug)]
struct MokaCacheEntry {
    entry: CacheEntry,
    size_bytes: usize,
    type_name: Option<&'static str>,
}

/// Per-entry key cost for eviction.
pub(super) fn key_footprint(_key: &InternalCacheKey) -> usize {
    std::mem::size_of::<InternalCacheKey>()
}

fn physical_size(key: &InternalCacheKey, size_bytes: usize) -> usize {
    key_footprint(key).saturating_add(size_bytes)
}

/// Number of physical bytes represented by one Moka weight unit.
///
/// Moka limits each entry's weight to `u32`, so capacities above 4 GiB need
/// coarser units to account for a single large entry without undercharging it.
fn weight_unit(capacity: usize) -> usize {
    capacity.div_ceil(u32::MAX as usize).max(1)
}

fn entry_weight(key: &InternalCacheKey, size_bytes: usize, weight_unit: usize) -> u32 {
    physical_size(key, size_bytes)
        .div_ceil(weight_unit)
        .try_into()
        .unwrap_or(u32::MAX)
}

/// Default [`CacheBackend`] backed by a [moka](https://crates.io/crates/moka) cache.
///
/// Nonzero capacities provide weighted eviction and concurrent-load deduplication.
/// A zero capacity invokes each loader independently without caching.
pub struct MokaCacheBackend {
    cache: moka::future::Cache<InternalCacheKey, MokaCacheEntry>,
    capacity: usize,
    weight_unit: usize,
    metrics: Arc<BackendCounters>,
}

/// Configuration for a [`MokaCacheBackend`].
///
/// Size-removal metrics are disabled by default because Moka's eviction
/// listener adds overhead to writes and eviction maintenance.
#[derive(Debug)]
pub struct MokaCacheBackendBuilder {
    capacity: usize,
    has_size_removal_metrics: bool,
}

impl MokaCacheBackendBuilder {
    /// Enable size-removal counts and accounted bytes through Moka's eviction listener.
    ///
    /// Size removals include rejected candidates as well as resident victims.
    /// The corresponding diagnostics fields are `None` unless this is enabled.
    pub fn with_size_removal_metrics(mut self) -> Self {
        self.has_size_removal_metrics = true;
        self
    }

    /// Build the configured backend.
    pub fn build(self) -> MokaCacheBackend {
        let metrics = Arc::new(BackendCounters::new(
            CacheBackendKind::Moka,
            self.has_size_removal_metrics,
        ));
        let weight_unit = weight_unit(self.capacity);
        let capacity_weight = self.capacity.div_ceil(weight_unit) as u64;
        let cache_builder = moka::future::Cache::builder()
            .max_capacity(capacity_weight)
            .weigher(move |key: &InternalCacheKey, entry: &MokaCacheEntry| {
                entry_weight(key, entry.size_bytes, weight_unit)
            });
        let cache_builder = if self.has_size_removal_metrics {
            let removal_metrics = metrics.clone();
            cache_builder.eviction_listener(move |key, entry: MokaCacheEntry, cause| {
                if cause == moka::notification::RemovalCause::Size {
                    // The same signal covers rejected candidates and resident victims.
                    let bytes = (entry_weight(key.as_ref(), entry.size_bytes, weight_unit) as u64)
                        .saturating_mul(weight_unit as u64);
                    removal_metrics.size_removal(1, bytes);
                }
            })
        } else {
            cache_builder
        };
        MokaCacheBackend {
            cache: cache_builder.build(),
            capacity: self.capacity,
            weight_unit,
            metrics,
        }
    }
}

impl std::fmt::Debug for MokaCacheBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MokaCacheBackend")
            .field("entry_count", &self.cache.entry_count())
            .finish()
    }
}

impl MokaCacheBackend {
    /// Configure a Moka backend with `capacity` weighted bytes.
    ///
    /// Size-removal metrics are disabled unless explicitly enabled:
    ///
    /// ```
    /// use lance_core::cache::MokaCacheBackend;
    /// let backend = MokaCacheBackend::builder(1024)
    ///     .with_size_removal_metrics()
    ///     .build();
    /// assert_eq!(backend.capacity(), 1024);
    /// ```
    pub fn builder(capacity: usize) -> MokaCacheBackendBuilder {
        MokaCacheBackendBuilder {
            capacity,
            has_size_removal_metrics: false,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::builder(capacity).build()
    }

    pub fn no_cache() -> Self {
        Self::with_capacity(0)
    }

    /// Configured weighted capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn weighted_size_bytes(&self) -> usize {
        self.cache
            .weighted_size()
            .saturating_mul(self.weight_unit as u64)
            .try_into()
            .unwrap_or(usize::MAX)
    }

    fn record_write(&self, key: &InternalCacheKey, size_bytes: usize) {
        let bytes = (entry_weight(key, size_bytes, self.weight_unit) as u64)
            .saturating_mul(self.weight_unit as u64);
        self.metrics.write(bytes);
        if self.capacity == 0 {
            self.metrics.disabled_rejection();
        }
        if key_footprint(key).checked_add(size_bytes).is_none()
            || physical_size(key, size_bytes).div_ceil(self.weight_unit) > u32::MAX as usize
        {
            self.metrics
                .weight_saturations
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    async fn insert_tagged(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        type_name: Option<&'static str>,
    ) {
        self.record_write(key, size_bytes);
        self.cache
            .insert(
                *key,
                MokaCacheEntry {
                    entry,
                    size_bytes,
                    type_name,
                },
            )
            .await;
    }

    async fn get_or_insert_tagged<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        type_name: Option<&'static str>,
    ) -> Result<(CacheEntry, bool)> {
        self.get_or_insert_tagged_outcome(key, loader, type_name)
            .await
            .map(|(entry, outcome)| (entry, outcome.was_loader_skipped()))
    }

    async fn get_or_insert_tagged_outcome<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        type_name: Option<&'static str>,
    ) -> Result<(CacheEntry, CacheLoadOutcome)> {
        if self.capacity == 0 {
            self.metrics.disabled_bypass();
            return loader
                .await
                .map(|(entry, _)| (entry, CacheLoadOutcome::Loaded));
        }

        let was_loaded = AtomicBool::new(false);
        let init = async {
            was_loaded.store(true, Ordering::Relaxed);
            loader
                .await
                .map(|(entry, size_bytes)| {
                    self.record_write(key, size_bytes);
                    MokaCacheEntry {
                        entry,
                        size_bytes,
                        type_name,
                    }
                })
                .map_err(CloneableError)
        };

        self.cache
            .try_get_with_by_ref(key, init)
            .await
            .map(|record| {
                let outcome = if was_loaded.load(Ordering::Relaxed) {
                    CacheLoadOutcome::Loaded
                } else {
                    CacheLoadOutcome::LoaderSkippedUnknown
                };
                (record.entry, outcome)
            })
            .map_err(|error| Arc::unwrap_or_clone(error).0)
    }

    fn occupancy_by_type(&self) -> CacheOccupancyByType {
        let mut types = BTreeMap::<&'static str, (u64, u64)>::new();
        let mut untagged_size_bytes = 0_u64;
        let mut untagged_num_entries = 0_u64;
        for (key, record) in self.cache.iter() {
            let weight = (entry_weight(key.as_ref(), record.size_bytes, self.weight_unit) as u64)
                .saturating_mul(self.weight_unit as u64);
            if let Some(type_name) = record.type_name {
                let totals = types.entry(type_name).or_default();
                totals.0 = totals.0.saturating_add(weight);
                totals.1 = totals.1.saturating_add(1);
            } else {
                untagged_size_bytes = untagged_size_bytes.saturating_add(weight);
                untagged_num_entries = untagged_num_entries.saturating_add(1);
            }
        }
        CacheOccupancyByType {
            types: types
                .into_iter()
                .map(
                    |(type_name, (size_bytes, num_entries))| CacheTypeOccupancy {
                        type_name: type_name.to_string(),
                        size_bytes,
                        num_entries,
                    },
                )
                .collect(),
            untagged_size_bytes,
            untagged_num_entries,
        }
    }
}

#[async_trait]
impl CacheBackend for MokaCacheBackend {
    async fn get(&self, key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        self.cache.get(key).await.map(|r| r.entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _codec: Option<CacheCodec>,
    ) {
        self.insert_tagged(key, entry, size_bytes, None).await;
    }

    async fn insert_with_context(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _codec: Option<CacheCodec>,
        context: CacheOperationContext,
    ) {
        self.insert_tagged(key, entry, size_bytes, context.type_name())
            .await;
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        _codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        self.get_or_insert_tagged(key, loader, None).await
    }

    async fn get_or_insert_with_context<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        _codec: Option<CacheCodec>,
        context: CacheOperationContext,
    ) -> Result<(CacheEntry, bool)> {
        self.get_or_insert_tagged(key, loader, context.type_name())
            .await
    }

    async fn get_or_insert_with_context_outcome<'a>(
        &self,
        key: &InternalCacheKey,
        loader: CacheLoader<'a>,
        _codec: Option<CacheCodec>,
        context: CacheOperationContext,
    ) -> Result<(CacheEntry, CacheLoadOutcome)> {
        self.get_or_insert_tagged_outcome(key, loader, context.type_name())
            .await
    }

    async fn clear(&self) {
        self.cache.invalidate_all();
        self.cache.run_pending_tasks().await;
    }

    async fn num_entries(&self) -> usize {
        self.cache.run_pending_tasks().await;
        self.cache.entry_count() as usize
    }

    async fn size_bytes(&self) -> usize {
        self.cache.run_pending_tasks().await;
        self.weighted_size_bytes()
    }

    fn capacity_bytes(&self) -> Option<usize> {
        Some(self.capacity)
    }

    fn diagnostics(&self) -> CacheBackendDiagnostics {
        self.metrics.snapshot(
            CacheBackendKind::Moka,
            self.capacity,
            self.approx_size_bytes(),
            self.approx_num_entries(),
        )
    }

    async fn diagnostics_with_types(
        &self,
        mode: CacheSnapshotMode,
    ) -> (CacheBackendDiagnostics, Option<CacheOccupancyByType>) {
        if mode == CacheSnapshotMode::Refreshed {
            self.cache.run_pending_tasks().await;
        }
        (self.diagnostics(), Some(self.occupancy_by_type()))
    }

    fn approx_num_entries(&self) -> usize {
        self.cache.entry_count() as usize
    }

    fn approx_size_bytes(&self) -> usize {
        // `weighted_size()` can be stale without `run_pending_tasks()`, which
        // is async and can't be called from this synchronous context.
        self.weighted_size_bytes()
    }

    fn deep_size_of_entries(
        &self,
        context: &mut Context,
        size_of_entry: &dyn Fn(&CacheEntry, &mut Context) -> Option<usize>,
    ) -> Option<usize> {
        Some(
            self.cache
                .iter()
                .map(|(key, record)| {
                    key_footprint(key.as_ref())
                        + size_of_entry(&record.entry, context).unwrap_or(record.size_bytes)
                })
                .sum(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;

    use futures::{FutureExt, poll};
    use tokio::sync::oneshot;

    use super::*;

    #[rstest::rstest]
    #[case::zero_capacity_success(MokaCacheBackend::with_capacity(0), false)]
    #[case::zero_capacity_error(MokaCacheBackend::with_capacity(0), true)]
    #[case::no_cache_success(MokaCacheBackend::no_cache(), false)]
    #[case::no_cache_error(MokaCacheBackend::no_cache(), true)]
    #[tokio::test]
    async fn zero_capacity_loaders_run_independently(
        #[case] backend: MokaCacheBackend,
        #[case] loader_fails: bool,
    ) {
        let key = InternalCacheKey::from_bytes([0; 16]);
        let first_entry: CacheEntry = Arc::new(1_u64);
        let second_entry: CacheEntry = Arc::new(2_u64);
        let (release_tx, release_rx) = oneshot::channel();
        let entry = first_entry.clone();
        let mut first = pin!(backend.get_or_insert(
            &key,
            Box::pin(async move {
                release_rx.await.unwrap();
                Ok((entry, 8))
            }),
            None,
        ));
        assert!(poll!(first.as_mut()).is_pending());

        let entry = second_entry.clone();
        let second = backend
            .get_or_insert(
                &key,
                Box::pin(async move {
                    if loader_fails {
                        Err(crate::Error::invalid_input("independent loader failed"))
                    } else {
                        Ok((entry, 8))
                    }
                }),
                None,
            )
            .now_or_never()
            .expect("a disabled cache must not wait for another loader on the same key");
        if loader_fails {
            let error = second.unwrap_err();
            assert!(matches!(&error, crate::Error::InvalidInput { .. }));
            assert!(error.to_string().contains("independent loader failed"));
        } else {
            let (entry, was_cached) = second.unwrap();
            assert!(Arc::ptr_eq(&entry, &second_entry));
            assert!(!was_cached);
        }

        release_tx.send(()).unwrap();
        let (entry, was_cached) = first.await.unwrap();
        assert!(Arc::ptr_eq(&entry, &first_entry));
        assert!(!was_cached);
        assert!(backend.get(&key, None).await.is_none());
        assert_eq!(backend.num_entries().await, 0);
        assert_eq!(backend.size_bytes().await, 0);
    }

    #[test]
    fn entry_weights_are_exact_at_byte_granularity() {
        let key = InternalCacheKey::from_bytes([0; 16]);
        assert_eq!(weight_unit(4096), 1);
        assert_eq!(entry_weight(&key, 7, 1), 23);
    }

    #[test]
    fn capacity_bytes_reports_configured_capacity() {
        assert_eq!(
            MokaCacheBackend::with_capacity(4096).capacity_bytes(),
            Some(4096)
        );
        assert_eq!(MokaCacheBackend::no_cache().capacity_bytes(), Some(0));
    }

    #[tokio::test]
    async fn size_methods_use_constant_time_weighted_accounting() {
        let backend = MokaCacheBackend::with_capacity(4096);
        let key = InternalCacheKey::from_bytes([0; 16]);
        let entry: CacheEntry = Arc::new(());
        let value_size = 7;
        let expected = physical_size(&key, value_size);

        backend.insert(&key, entry, value_size, None).await;

        assert_eq!(backend.size_bytes().await, expected);
        assert_eq!(backend.approx_size_bytes(), expected);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn entry_weights_scale_for_capacities_above_four_gibibytes() {
        let key = InternalCacheKey::from_bytes([0; 16]);
        let capacity = 6 * 1024 * 1024 * 1024;
        let weight_unit = weight_unit(capacity);
        assert_eq!(weight_unit, 2);

        let size_bytes = u32::MAX as usize + 1024;
        let expected = physical_size(&key, size_bytes).div_ceil(weight_unit);
        let weight = entry_weight(&key, size_bytes, weight_unit);
        assert_eq!(weight as usize, expected);
        assert_ne!(weight, u32::MAX);
    }

    #[cfg(target_pointer_width = "64")]
    #[tokio::test]
    async fn diagnostics_use_scaled_accounting_and_report_saturation() {
        let backend = MokaCacheBackend::builder(6 << 30)
            .with_size_removal_metrics()
            .build();
        let key = InternalCacheKey::from_bytes([0; 16]);
        let declared_size = u32::MAX as usize + 1024;
        let expected = entry_weight(&key, declared_size, backend.weight_unit) as u64
            * backend.weight_unit as u64;
        backend
            .insert(&key, Arc::new(()), declared_size, None)
            .await;
        let snapshot = backend
            .diagnostics_with_mode(super::super::CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(snapshot.write_bytes, Some(expected));
        assert_eq!(snapshot.size_bytes, Some(expected));
        assert_eq!(snapshot.weight_saturations, Some(0));

        backend.clear().await;
        backend.insert(&key, Arc::new(()), usize::MAX, None).await;
        let snapshot = backend
            .diagnostics_with_mode(super::super::CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(snapshot.weight_saturations, Some(1));
        assert_eq!(snapshot.size_removals, Some(1));
        assert_eq!(
            snapshot.size_removed_bytes,
            Some(u32::MAX as u64 * backend.weight_unit as u64)
        );
        assert_eq!(snapshot.resident_evictions, None);
    }

    #[tokio::test]
    async fn size_removal_metrics_are_opt_in_and_exclude_explicit_removals() {
        let key = InternalCacheKey::from_bytes([0; 16]);
        let disabled = MokaCacheBackend::with_capacity(256);
        disabled.insert(&key, Arc::new(()), 1024, None).await;
        let snapshot = disabled
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(
            (snapshot.size_removals, snapshot.size_removed_bytes),
            (None, None)
        );

        let enabled = MokaCacheBackend::builder(256)
            .with_size_removal_metrics()
            .build();
        enabled.insert(&key, Arc::new(()), 1024, None).await;
        let snapshot = enabled
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(
            (snapshot.size_removals, snapshot.size_removed_bytes),
            (Some(1), Some(1040))
        );
        enabled.insert(&key, Arc::new(()), 48, None).await;
        enabled.clear().await;
        let snapshot = enabled
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        assert_eq!(
            (snapshot.size_removals, snapshot.size_removed_bytes),
            (Some(1), Some(1040))
        );

        for id in 1..8 {
            enabled
                .insert(
                    &InternalCacheKey::from_bytes([id; 16]),
                    Arc::new(()),
                    48,
                    None,
                )
                .await;
        }
        let snapshot = enabled
            .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
            .await;
        let removals = snapshot.size_removals.unwrap();
        assert!(removals > 1);
        assert_eq!(
            snapshot.size_removed_bytes,
            Some(1040 + (removals - 1) * 64)
        );
    }
}

/// Registry identifier for the built-in Moka backend.
pub const MOKA_BACKEND_KIND: &str = "moka";

/// [`BackendBuildFn`](super::registry::BackendBuildFn) for [`MokaCacheBackend`].
///
/// Recognized options:
///   * `capacity` — total weighted capacity in bytes (`usize`).
///     This must be present and non-empty.
///   * `size_removal_metrics` — `true` enables size-removal metrics through
///     Moka's eviction listener; defaults to `false`.
///
/// Unknown options are rejected so typos surface immediately instead of
/// silently falling through to the default capacity.
pub(super) fn build_moka_backend(
    config: &super::registry::BackendConfig,
) -> Result<MokaCacheBackend> {
    let mut capacity: Option<usize> = None;
    let mut has_size_removal_metrics = false;
    for (key, value) in &config.options {
        match key.as_str() {
            "capacity" => {
                if value.is_empty() {
                    return Err(crate::Error::invalid_input(
                        "moka cache backend: capacity must not be empty",
                    ));
                } else {
                    capacity = Some(value.parse::<usize>().map_err(|err| {
                        crate::Error::invalid_input(format!(
                            "moka cache backend: cannot parse capacity {:?}: {}",
                            value, err
                        ))
                    })?);
                }
            }
            "size_removal_metrics" => {
                has_size_removal_metrics = value.parse::<bool>().map_err(|err| {
                    crate::Error::invalid_input(format!(
                        "moka cache backend: cannot parse size_removal_metrics {:?}: {}",
                        value, err
                    ))
                })?;
            }
            other => {
                return Err(crate::Error::invalid_input(format!(
                    "moka cache backend: unknown option {:?}",
                    other
                )));
            }
        }
    }
    let capacity = capacity.ok_or_else(|| {
        crate::Error::invalid_input(
            "moka cache backend: capacity is required; use moka://?capacity=<bytes>",
        )
    })?;
    let mut builder = MokaCacheBackend::builder(capacity);
    if has_size_removal_metrics {
        builder = builder.with_size_removal_metrics();
    }
    Ok(builder.build())
}

pub(super) fn build_moka(config: &super::registry::BackendConfig) -> Result<Arc<dyn CacheBackend>> {
    Ok(Arc::new(build_moka_backend(config)?))
}

#[cfg(test)]
mod moka_registry_tests {
    use super::super::backend_uri::{build_from_uri, parse_backend_uri};
    use super::super::registry::{BackendConfig, build_from_config, registry_test_lock};
    use super::*;

    #[test]
    fn test_moka_builds_from_config() {
        let _lock = registry_test_lock();
        let cfg = BackendConfig::new("moka")
            .unwrap()
            .with_option("capacity", "1048576");
        let backend = build_moka_backend(&cfg).unwrap();
        assert_eq!(backend.capacity(), 1048576);
        assert_eq!(backend.diagnostics().size_removals, None);
        let _backend = build_from_config(&cfg).unwrap();
    }

    #[test]
    fn test_moka_builds_from_uri() {
        let _lock = registry_test_lock();
        let cfg = parse_backend_uri("moka://?capacity=1048576").unwrap();
        let backend = build_moka_backend(&cfg).unwrap();
        assert_eq!(backend.capacity(), 1048576);
        assert_eq!(backend.diagnostics().size_removals, None);
        let _backend = build_from_uri("moka://?capacity=1048576").unwrap();
    }

    #[test]
    fn test_moka_size_removal_metrics_option() {
        let _lock = registry_test_lock();
        let cfg = parse_backend_uri("moka://?capacity=256&size_removal_metrics=true").unwrap();
        let backend = build_moka_backend(&cfg).unwrap();
        assert_eq!(backend.diagnostics().size_removals, Some(0));

        let cfg = parse_backend_uri("moka://?capacity=256&size_removal_metrics=false").unwrap();
        let backend = build_moka_backend(&cfg).unwrap();
        assert_eq!(backend.diagnostics().size_removals, None);
    }

    #[test]
    fn test_moka_rejects_invalid_size_removal_metrics_option() {
        let _lock = registry_test_lock();
        let cfg = BackendConfig::new("moka")
            .unwrap()
            .with_option("capacity", "256")
            .with_option("size_removal_metrics", "yes");
        let err = build_moka_backend(&cfg).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidInput { .. }));
        assert!(err.to_string().contains("size_removal_metrics"));
        assert!(err.to_string().contains("yes"));
    }

    #[test]
    fn test_moka_rejects_unknown_option() {
        let _lock = registry_test_lock();
        let cfg = BackendConfig::new("moka")
            .unwrap()
            .with_option("mystery", "1");
        let err = build_from_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("unknown option"));
    }

    #[test]
    fn test_moka_rejects_bad_capacity() {
        let _lock = registry_test_lock();
        let cfg = BackendConfig::new("moka")
            .unwrap()
            .with_option("capacity", "not-a-number");
        let err = build_from_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("cannot parse capacity"));
    }

    #[test]
    fn test_moka_rejects_missing_capacity() {
        let _lock = registry_test_lock();
        let cfg = BackendConfig::new("moka").unwrap();
        let err = build_from_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("capacity is required"));
    }

    #[test]
    fn test_moka_rejects_empty_capacity() {
        let _lock = registry_test_lock();
        let cfg = BackendConfig::new("moka")
            .unwrap()
            .with_option("capacity", "");
        let err = build_from_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("capacity must not be empty"));
    }
}
