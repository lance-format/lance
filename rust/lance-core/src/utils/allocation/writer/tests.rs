// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_buffer::Buffer;
use rstest::rstest;

use super::*;
use crate::utils::allocation::AllocationBudget;

#[rstest]
#[case::empty(0)]
#[case::single_byte(1)]
#[case::unaligned(63)]
#[case::cache_line(64)]
#[case::beyond_cache_line(65)]
#[case::beyond_page(4097)]
fn exact_layout_initializes_only_appended_bytes(#[case] len: usize) {
    let budget = AllocationBudget::new(len);
    let mut work = budget.try_reserve(len).unwrap();
    let mut writer = work.try_allocate_writer(len).unwrap().unwrap();
    assert!(writer.initialized_mut().is_empty());
    assert_eq!(work.len(), 0);
    assert_eq!(budget.reserved(), len);
    let prefix = len.min(7);
    writer.extend_zeros(prefix).unwrap();
    let suffix: Vec<u8> = (prefix..len).map(|i| (i % 251 + 1) as u8).collect();
    writer.extend_from_slice(&suffix).unwrap();
    writer.extend_from_slice(&[]).unwrap();
    writer.extend_zeros(0).unwrap();
    if prefix != 0 {
        writer.initialized_mut()[0] = 255;
    }
    let mut expected = vec![0; prefix];
    expected.extend_from_slice(&suffix);
    if prefix != 0 {
        expected[0] = 255;
    }
    let bytes = writer.finish().unwrap();
    assert_eq!(bytes.as_ref(), expected);
    assert_eq!(bytes.as_ptr().align_offset(ALIGNMENT), 0);
    drop((bytes, work));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn refused_layout_preserves_the_original_workspace() {
    let budget = AllocationBudget::new(8);
    let mut work = budget.try_reserve(8).unwrap();
    assert!(work.try_allocate_writer(9).unwrap().is_none());
    let error = work.try_allocate_writer(usize::MAX).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("addressable allocation size"));
    assert_eq!(work.len(), 8);
    assert_eq!(budget.reserved(), 8);
}

#[test]
fn oversized_append_cannot_advance_or_change_the_initialized_prefix() {
    let budget = AllocationBudget::new(5);
    let mut work = budget.try_reserve(5).unwrap();
    let mut writer = work.try_allocate_writer(5).unwrap().unwrap();
    writer.extend_from_slice(b"ab").unwrap();
    for error in [
        writer.extend_from_slice(b"cdef").unwrap_err(),
        writer.extend_zeros(4).unwrap_err(),
        writer.extend_zeros(usize::MAX).unwrap_err(),
    ] {
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("exceeds reserved allocation length")
        );
    }
    assert_eq!(writer.initialized_mut(), b"ab");
    writer.extend_from_slice(b"cde").unwrap();
    assert_eq!(writer.finish().unwrap().as_ref(), b"abcde");
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::none(0)]
#[case::partial(2)]
#[case::one_byte_missing(4)]
fn incomplete_output_cannot_be_published(#[case] initialized: usize) {
    let budget = AllocationBudget::new(5);
    let mut work = budget.try_reserve(5).unwrap();
    let mut writer = work.try_allocate_writer(5).unwrap().unwrap();
    writer.extend_zeros(initialized).unwrap();
    let error = writer.finish().unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("initialized bytes, expected 5"));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn dropped_partial_writer_and_last_arrow_alias_refund_the_full_layout() {
    let budget = AllocationBudget::new(65);
    let mut work = budget.try_reserve(65).unwrap();
    let mut writer = work.try_allocate_writer(65).unwrap().unwrap();
    writer.extend_from_slice(b"partial").unwrap();
    let mut wait = Box::pin(budget.wait_for_capacity(65));
    assert!(futures::poll!(&mut wait).is_pending());
    drop(writer);
    wait.await.unwrap();
    assert_eq!(budget.reserved(), 0);

    let mut work = budget.try_reserve(65).unwrap();
    let mut writer = work.try_allocate_writer(65).unwrap().unwrap();
    writer.extend_zeros(64).unwrap();
    writer.extend_from_slice(&[91]).unwrap();
    let bytes = writer.finish().unwrap();
    let arrow = Buffer::from(bytes);
    let alias = arrow.slice_with_length(64, 1);
    drop((arrow, work));
    assert_eq!(budget.reserved(), 65);
    assert!(budget.try_reserve(1).is_none());
    // Finished owners are safe to move across the CPU/consumer boundary.
    std::thread::spawn(move || assert_eq!(alias.as_slice(), &[91]))
        .join()
        .unwrap();
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn unwinding_an_incomplete_writer_releases_its_credit() {
    let budget = AllocationBudget::new(64);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut work = budget.try_reserve(64).unwrap();
        let mut writer = work.try_allocate_writer(64).unwrap().unwrap();
        writer.extend_from_slice(b"prefix").unwrap();
        panic!("injected initializer panic");
    }));
    assert!(caught.is_err());
    assert_eq!(budget.reserved(), 0);
}
