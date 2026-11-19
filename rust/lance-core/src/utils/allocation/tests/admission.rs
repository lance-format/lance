// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[case::zero(64, 0, true)]
#[case::exact(64, 64, true)]
#[case::too_large(64, 65, false)]
#[case::address_limit(usize::MAX, usize::MAX, true)]
#[tokio::test]
async fn queued_admission_validates_before_waiting(
    #[case] limit: usize,
    #[case] bytes: usize,
    #[case] valid: bool,
) {
    let budget = AllocationBudget::new(limit);
    let result = budget.reserve_when_available(bytes).now_or_never().unwrap();
    if valid {
        let work = result.unwrap();
        assert_eq!(work.len(), bytes);
        assert_eq!(budget.reserved(), bytes);
        drop(work);
    } else {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("exceeds allocation budget"));
    }
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn smaller_waiter_cannot_overtake_and_final_alias_controls_credit() {
    let budget = AllocationBudget::new(64);
    let held = budget.try_reserve(32).unwrap();
    let mut first = Box::pin(budget.reserve_when_available(64));
    let mut later = Box::pin(budget.reserve_when_available(32));
    assert!(first.as_mut().now_or_never().is_none());
    assert!(later.as_mut().now_or_never().is_none());
    assert_eq!(
        budget.reserved(),
        32,
        "waiters must not reserve partial credit"
    );
    drop(held);
    // Polling a later request first must not turn a refund into an overtaking.
    assert!(later.as_mut().now_or_never().is_none());
    let bytes = first
        .now_or_never()
        .expect("first request must receive its turn")
        .unwrap()
        .allocate()
        .unwrap()
        .freeze();
    let alias = bytes.slice(0..1);
    drop(bytes);
    assert_eq!(budget.reserved(), 64);
    assert!(later.as_mut().now_or_never().is_none());
    drop(alias);
    let work = later
        .now_or_never()
        .expect("final alias release must admit the waiter")
        .unwrap();
    assert_eq!(work.len(), 32);
    assert_eq!(budget.reserved(), 32);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn cancelling_head_admits_smaller_waiter_without_another_refund() {
    let budget = AllocationBudget::new(64);
    let held = budget.try_reserve(32).unwrap();
    let mut first = Box::pin(budget.reserve_when_available(64));
    let mut later = Box::pin(budget.reserve_when_available(32));
    assert!(first.as_mut().now_or_never().is_none());
    assert!(later.as_mut().now_or_never().is_none());
    let zero = budget
        .reserve_when_available(0)
        .now_or_never()
        .unwrap()
        .unwrap();
    let error = budget
        .reserve_when_available(65)
        .now_or_never()
        .unwrap()
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("exceeds allocation budget"));
    assert_eq!(budget.reserved(), 32);
    drop(first);
    let work = later
        .now_or_never()
        .expect("cancelled head releases its turn")
        .unwrap();
    assert_eq!(work.len(), 32);
    assert_eq!(budget.reserved(), 64);
    drop((zero, held, work));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn cancelling_middle_keeps_remaining_order() {
    let budget = AllocationBudget::new(64);
    let held = budget.try_reserve(64).unwrap();
    let mut first = Box::pin(budget.reserve_when_available(64));
    let mut cancelled = Box::pin(budget.reserve_when_available(32));
    let mut last = Box::pin(budget.reserve_when_available(32));
    assert!(first.as_mut().now_or_never().is_none());
    assert!(cancelled.as_mut().now_or_never().is_none());
    assert!(last.as_mut().now_or_never().is_none());
    drop(cancelled);
    drop(held);
    assert!(last.as_mut().now_or_never().is_none());
    let first_work = first
        .now_or_never()
        .expect("middle cancellation preserves the first turn")
        .unwrap();
    assert!(last.as_mut().now_or_never().is_none());
    drop(first_work);
    let last_work = last
        .now_or_never()
        .expect("remaining waiter must receive credit")
        .unwrap();
    assert_eq!(budget.reserved(), 32);
    drop(last_work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn nonblocking_competitor_cannot_turn_a_wake_into_credit() {
    let budget = AllocationBudget::new(64);
    let held = budget.try_reserve(64).unwrap();
    let mut waiting = Box::pin(budget.reserve_when_available(64));
    assert!(waiting.as_mut().now_or_never().is_none());
    drop(held);
    let competitor = budget.try_reserve_up_to(64).unwrap();
    assert!(waiting.as_mut().now_or_never().is_none());
    assert_eq!(budget.reserved(), 64);
    drop(competitor);
    let work = waiting
        .now_or_never()
        .expect("competitor refund must admit the waiter")
        .unwrap();
    assert_eq!(work.len(), 64);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn admitted_work_does_not_hold_the_queue_or_block_other_budgets() {
    let budget = AllocationBudget::new(64);
    let first = budget.reserve_when_available(32).await.unwrap();
    let second = budget
        .reserve_when_available(32)
        .now_or_never()
        .unwrap()
        .unwrap();
    let mut waiting = Box::pin(budget.reserve_when_available(1));
    assert!(waiting.as_mut().now_or_never().is_none());
    let other = AllocationBudget::new(1);
    let independent = other
        .reserve_when_available(1)
        .now_or_never()
        .unwrap()
        .unwrap();
    assert_eq!(budget.reserved(), 64);
    assert_eq!(other.reserved(), 1);
    drop(waiting);
    drop((first, second, independent));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(other.reserved(), 0);
}
