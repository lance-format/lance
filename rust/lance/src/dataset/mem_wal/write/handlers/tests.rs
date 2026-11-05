// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WAL append ordering across frozen and active stores.

use super::*;
use crate::dataset::mem_wal::write::test_support::*;

/// A background tick must append the **oldest** store that still owes the WAL,
/// never "whatever memtable is active".
///
/// A tick carries no store: it is enqueued by a timer and resolved when it is
/// handled. So a tick enqueued before a freeze is handled *after* it, and
/// resolving to the active memtable would append the incoming memtable's
/// batches ahead of the outgoing memtable's tail. WAL entry positions are
/// assigned in append-call order, replay walks them ascending, row positions
/// follow, and primary-key recency is "newest visible row position wins" — so
/// that inverts dedup after a crash, handing the key to the stale row. It
/// survives the crash that caused it, and a full scan cannot see it.
#[test]
fn test_next_pending_store_picks_the_oldest_owing_an_append() {
    let schema = create_test_schema();

    // A frozen store of 2 batches at coordinate 0, and the active store that
    // rotated in behind it at coordinate 2.
    let frozen = Arc::new(BatchStore::with_capacity(4));
    frozen.append(create_test_batch(&schema, 0, 1)).unwrap();
    frozen.append(create_test_batch(&schema, 1, 1)).unwrap();
    let active = Arc::new(BatchStore::with_capacity_at(4, 2));
    active.append(create_test_batch(&schema, 2, 1)).unwrap();

    let frozen_list = || std::iter::once(Arc::clone(&frozen));

    // Nothing durable: the frozen store owes the oldest append, so it wins —
    // even though the active memtable also has un-appended batches.
    let picked = next_pending_store(frozen_list(), Arc::clone(&active), 0).unwrap();
    assert!(
        Arc::ptr_eq(&picked, &frozen),
        "the outgoing memtable's tail must be appended before the incoming one's head"
    );

    // Still true partway through the frozen store.
    let picked = next_pending_store(frozen_list(), Arc::clone(&active), 1).unwrap();
    assert!(Arc::ptr_eq(&picked, &frozen));

    // Once the frozen store is fully durable, the active one is next.
    let picked = next_pending_store(frozen_list(), Arc::clone(&active), 2).unwrap();
    assert!(Arc::ptr_eq(&picked, &active));

    // Everything durable: nothing to do.
    assert!(next_pending_store(frozen_list(), Arc::clone(&active), 3).is_none());
}
