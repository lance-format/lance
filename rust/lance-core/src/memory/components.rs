// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory that Lance components hold right now, read when the gauges are
//! refreshed.
//!
//! A component registers every instance it creates. The registry keeps a weak
//! reference only, so an instance counts while it lives and is forgotten once
//! it is dropped, and the paths that use the components do not change.

use std::sync::Arc;

/// A part of Lance whose memory `lance_memory_component_bytes` reports.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    /// Bytes an I/O scheduler admitted and its caller has not received yet.
    IoScheduler,
    /// Reservations in a DataFusion memory pool that Lance builds from its
    /// execution options.
    DataFusionPool,
    /// Memtables and pending WAL batches of a MemWAL shard writer.
    MemWal,
}

impl Component {
    #[cfg(feature = "metrics")]
    const ALL: [Self; 3] = [Self::IoScheduler, Self::DataFusionPool, Self::MemWal];

    /// The value of the `component` label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IoScheduler => "io_scheduler",
            Self::DataFusionPool => "datafusion_pool",
            Self::MemWal => "mem_wal",
        }
    }

    #[cfg(feature = "metrics")]
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// Counts `held_bytes(owner)` towards `component` for as long as `owner`
/// lives.
///
/// Does nothing unless observation is on and Lance is built with the `metrics`
/// feature. Only a weak reference to `owner` is kept. `held_bytes` runs when
/// the gauges are refreshed, on the refreshing thread and with no lock of the
/// registry held. It must not panic and must return quickly: a short lock is
/// fine, I/O or waiting on other work is not. Register each owner once, and
/// when one object passes its reservations on to another, register only one of
/// them, or their bytes count twice.
#[doc(hidden)]
#[inline]
pub fn register_component<T: ?Sized + Send + Sync + 'static>(
    component: Component,
    owner: &Arc<T>,
    held_bytes: fn(&T) -> u64,
) {
    #[cfg(feature = "metrics")]
    if super::observer::observing() {
        enabled::registry().register(component, owner, held_bytes);
    }
    #[cfg(not(feature = "metrics"))]
    let _ = (component, owner, held_bytes);
}

/// Sets the component gauge of every component registered in this process.
#[cfg(feature = "metrics")]
pub(super) fn publish() {
    enabled::registry().publish();
}

#[cfg(feature = "metrics")]
mod enabled {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicPtr, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

    use super::Component;
    use crate::memory::telemetry::METRIC_COMPONENT;

    /// Registrations kept before the dead ones are pruned on registration.
    /// Schedulers are created for every scan, so pruning on every
    /// registration would make registering linear in the live instances.
    pub(super) const MIN_PRUNE_AT: usize = 64;

    /// The registry and the process it was created in.
    pub(super) struct ProcessRegistry {
        pid: u32,
        registry: Registry,
    }

    static PROCESS_REGISTRY: AtomicPtr<ProcessRegistry> = AtomicPtr::new(std::ptr::null_mut());

    /// The registry of this process.
    pub(super) fn registry() -> &'static Registry {
        registry_for(&PROCESS_REGISTRY, std::process::id())
    }

    /// The registry of process `pid`, created on first use.
    ///
    /// A forked child inherits the registrations of its parent, whose owners
    /// are held by threads the child does not have, and possibly the lock of
    /// the registry held at the moment of the fork. The child therefore
    /// installs a registry of its own and never touches the inherited one,
    /// which is leaked, as the memory observer is.
    pub(super) fn registry_for(slot: &AtomicPtr<ProcessRegistry>, pid: u32) -> &'static Registry {
        loop {
            let current = slot.load(Ordering::Acquire);
            // SAFETY: non-null values of the slot come from `Box::into_raw`
            // below and are never freed, so a loaded pointer stays valid.
            if let Some(installed) = unsafe { current.as_ref() }
                && installed.pid == pid
            {
                return &installed.registry;
            }
            let fresh = Box::into_raw(Box::new(ProcessRegistry {
                pid,
                registry: Registry::new(),
            }));
            match slot.compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire) {
                // SAFETY: `fresh` came from `Box::into_raw` above and the slot
                // now owns it for the rest of the process.
                Ok(_) => return &unsafe { &*fresh }.registry,
                // SAFETY: another thread installed its registry first, so
                // `fresh` was never shared and can be freed here.
                Err(_) => drop(unsafe { Box::from_raw(fresh) }),
            }
        }
    }

    trait Probe: Send + Sync {
        /// The bytes the owner holds, or `None` once it is gone.
        fn held_bytes(&self) -> Option<u64>;
        fn is_live(&self) -> bool;
    }

    struct WeakProbe<T: ?Sized> {
        owner: Weak<T>,
        held_bytes: fn(&T) -> u64,
    }

    impl<T: ?Sized + Send + Sync> Probe for WeakProbe<T> {
        fn held_bytes(&self) -> Option<u64> {
            let owner = self.owner.upgrade()?;
            Some((self.held_bytes)(&owner))
        }

        fn is_live(&self) -> bool {
            self.owner.strong_count() > 0
        }
    }

    #[derive(Clone)]
    struct Registration {
        component: Component,
        /// Where the owner lives, to count an owner registered twice once.
        /// The weak reference keeps the allocation, so no other owner can
        /// take this address while the registration exists.
        address: usize,
        probe: Arc<dyn Probe>,
    }

    struct Registrations {
        list: Vec<Registration>,
        prune_at: usize,
        /// The components registered at least once, as [`Component::bit`]s.
        seen: u8,
    }

    impl Registrations {
        fn prune(&mut self) {
            self.list
                .retain(|registration| registration.probe.is_live());
            self.prune_at = MIN_PRUNE_AT.max(2 * self.list.len());
        }
    }

    pub(super) struct Registry {
        registrations: Mutex<Registrations>,
    }

    impl Registry {
        pub(super) const fn new() -> Self {
            Self {
                registrations: Mutex::new(Registrations {
                    list: Vec::new(),
                    prune_at: MIN_PRUNE_AT,
                    seen: 0,
                }),
            }
        }

        pub(super) fn register<T: ?Sized + Send + Sync + 'static>(
            &self,
            component: Component,
            owner: &Arc<T>,
            held_bytes: fn(&T) -> u64,
        ) {
            let registration = Registration {
                component,
                address: Arc::as_ptr(owner).cast::<()>() as usize,
                probe: Arc::new(WeakProbe {
                    owner: Arc::downgrade(owner),
                    held_bytes,
                }),
            };
            let mut registrations = self.lock();
            // Seen on registration, not when a refresh finds it alive: most
            // schedulers live between two refreshes.
            registrations.seen |= component.bit();
            registrations.list.push(registration);
            if registrations.list.len() >= registrations.prune_at {
                registrations.prune();
            }
        }

        pub(super) fn publish(&self) {
            let (list, seen) = {
                let mut registrations = self.lock();
                registrations.prune();
                (registrations.list.clone(), registrations.seen)
            };
            // The owners are read without the lock: reading one may take the
            // owner's own lock, and the last reference to an owner dropped
            // here runs its `Drop`.
            let mut totals = Component::ALL.map(|component| (component, 0_u64));
            let mut counted = HashSet::with_capacity(list.len());
            for registration in &list {
                if !counted.insert(registration.address) {
                    continue;
                }
                if let Some(bytes) = registration.probe.held_bytes()
                    && let Some((_, total)) = totals
                        .iter_mut()
                        .find(|(component, _)| *component == registration.component)
                {
                    *total = total.saturating_add(bytes);
                }
            }
            for (component, total) in totals {
                if seen & component.bit() != 0 {
                    metrics::gauge!(METRIC_COMPONENT, "component" => component.as_str())
                        .set(total as f64);
                }
            }
        }

        fn lock(&self) -> MutexGuard<'_, Registrations> {
            self.registrations
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
        }

        #[cfg(test)]
        pub(super) fn len(&self) -> usize {
            self.lock().list.len()
        }

        #[cfg(test)]
        pub(super) fn is_locked(&self) -> bool {
            self.registrations.try_lock().is_err()
        }
    }
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    use super::enabled::{MIN_PRUNE_AT, Registry, registry_for};
    use super::*;
    use crate::memory::telemetry::METRIC_COMPONENT;

    fn load(bytes: &AtomicU64) -> u64 {
        bytes.load(Ordering::Relaxed)
    }

    fn owner(bytes: u64) -> Arc<AtomicU64> {
        Arc::new(AtomicU64::new(bytes))
    }

    /// The component gauges `publish` sets, by label value.
    fn published(registry: &Registry) -> Vec<(String, f64)> {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || registry.publish());
        let mut gauges: Vec<(String, f64)> = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _unit, _description, value)| {
                let key = key.key();
                assert_eq!(key.name(), METRIC_COMPONENT);
                let labels: Vec<_> = key.labels().map(|l| (l.key(), l.value())).collect();
                let [("component", component)] = labels.as_slice() else {
                    panic!("unexpected labels {labels:?}");
                };
                let DebugValue::Gauge(value) = value else {
                    panic!("{component} is not a gauge: {value:?}");
                };
                (component.to_string(), value.0)
            })
            .collect();
        gauges.sort_by(|a, b| a.0.cmp(&b.0));
        gauges
    }

    fn gauges(expected: &[(&str, f64)]) -> Vec<(String, f64)> {
        expected
            .iter()
            .map(|(component, value)| (component.to_string(), *value))
            .collect()
    }

    #[test]
    fn test_live_owners_are_summed_per_component() {
        let registry = Registry::new();
        let (first, second, pool) = (owner(10), owner(5), owner(7));
        registry.register(Component::IoScheduler, &first, load);
        registry.register(Component::IoScheduler, &second, load);
        registry.register(Component::DataFusionPool, &pool, load);
        assert_eq!(
            published(&registry),
            gauges(&[("datafusion_pool", 7.0), ("io_scheduler", 15.0)])
        );
    }

    #[test]
    fn test_owners_are_read_when_published() {
        let registry = Registry::new();
        let scheduler = owner(10);
        registry.register(Component::IoScheduler, &scheduler, load);
        scheduler.store(25, Ordering::Relaxed);
        assert_eq!(published(&registry), gauges(&[("io_scheduler", 25.0)]));
    }

    #[test]
    fn test_a_component_without_live_owners_reads_zero() {
        let registry = Registry::new();
        let (kept, dropped) = (owner(10), owner(5));
        registry.register(Component::IoScheduler, &kept, load);
        registry.register(Component::MemWal, &dropped, load);
        assert_eq!(
            published(&registry),
            gauges(&[("io_scheduler", 10.0), ("mem_wal", 5.0)])
        );
        drop(dropped);
        assert_eq!(
            published(&registry),
            gauges(&[("io_scheduler", 10.0), ("mem_wal", 0.0)])
        );
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_an_owner_gone_before_the_first_refresh_reads_zero() {
        let registry = Registry::new();
        registry.register(Component::DataFusionPool, &owner(7), load);
        assert_eq!(published(&registry), gauges(&[("datafusion_pool", 0.0)]));
    }

    #[test]
    fn test_the_registry_does_not_keep_owners_alive() {
        let registry = Registry::new();
        let pool = owner(7);
        let watch = Arc::downgrade(&pool);
        registry.register(Component::DataFusionPool, &pool, load);
        drop(pool);
        assert_eq!(watch.strong_count(), 0);
    }

    #[test]
    fn test_an_owner_registered_twice_counts_once() {
        let registry = Registry::new();
        let pool = owner(7);
        registry.register(Component::DataFusionPool, &pool, load);
        registry.register(Component::DataFusionPool, &pool, load);
        assert_eq!(published(&registry), gauges(&[("datafusion_pool", 7.0)]));
    }

    #[test]
    fn test_dead_registrations_do_not_accumulate() {
        let registry = Registry::new();
        let kept = owner(3);
        registry.register(Component::IoScheduler, &kept, load);
        for _ in 0..1000 {
            registry.register(Component::IoScheduler, &owner(100), load);
            assert!(
                registry.len() <= MIN_PRUNE_AT,
                "{} registrations",
                registry.len()
            );
        }
        assert_eq!(published(&registry), gauges(&[("io_scheduler", 3.0)]));
    }

    #[test]
    fn test_a_forked_child_does_not_use_the_registry_it_inherits() {
        let slot = AtomicPtr::new(std::ptr::null_mut());
        let parent = registry_for(&slot, 1);
        let scheduler = owner(10);
        parent.register(Component::IoScheduler, &scheduler, load);
        assert!(std::ptr::eq(parent, registry_for(&slot, 1)));

        let child = registry_for(&slot, 2);
        assert!(!std::ptr::eq(parent, child));
        assert_eq!(published(child), gauges(&[]));
        assert!(std::ptr::eq(child, registry_for(&slot, 2)));
        assert_eq!(published(parent), gauges(&[("io_scheduler", 10.0)]));
    }

    #[test]
    fn test_owners_are_read_without_the_registry_lock() {
        static REGISTRY: Registry = Registry::new();
        fn lock_is_free(_: &()) -> u64 {
            u64::from(!REGISTRY.is_locked())
        }
        let held = Arc::new(());
        REGISTRY.register(Component::MemWal, &held, lock_is_free);
        assert_eq!(published(&REGISTRY), gauges(&[("mem_wal", 1.0)]));
    }

    #[test]
    fn test_nothing_registers_while_observation_is_off() {
        assert!(
            std::env::var_os(super::super::OBSERVE_ENV_VAR).is_none(),
            "this test needs the switch unset"
        );
        let scheduler = owner(10);
        register_component(Component::IoScheduler, &scheduler, load);
        assert_eq!(enabled::registry().len(), 0);
    }
}
