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
fn exact_layout_initializes_only_appended_bytes(
    #[case] len: usize,
    #[values(false, true)] foreign: bool,
) {
    let budget = AllocationBudget::new(len);
    let mut work = budget.try_reserve(len).unwrap();
    let mut writer = work.try_allocate_writer(len).unwrap().unwrap();
    assert!(writer.initialized_mut().is_empty());
    assert_eq!(work.len(), 0);
    assert_eq!(budget.reserved(), len);
    let prefix = len.min(7);
    writer.extend_zeros(prefix).unwrap();
    let suffix: Vec<u8> = (prefix..len).map(|i| (i % 251 + 1) as u8).collect();
    if foreign {
        // SAFETY: initializes the complete bounded suffix without reading it.
        unsafe {
            writer.try_extend_with(suffix.len(), |output| {
                for (byte, value) in output.iter_mut().zip(&suffix) {
                    byte.write(*value);
                }
                Ok(output.len())
            })
        }
        .unwrap();
    } else {
        writer.extend_from_slice(&suffix).unwrap();
    }
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

#[rstest]
#[case::error(None)]
#[case::short(Some(1))]
#[case::long(Some(4))]
fn unsuccessful_foreign_write_preserves_readable_prefix(#[case] returned: Option<usize>) {
    let pool = AllocationBudget::new(5);
    let mut work = pool.try_reserve(5).unwrap();
    let mut writer = work.try_allocate_writer(5).unwrap().unwrap();
    writer.extend_from_slice(b"ab").unwrap();
    // SAFETY: no successful exact-length result, only bounded writes and no reads.
    let error = unsafe {
        writer.try_extend_with(3, |output| {
            output[0].write(99);
            returned.ok_or_else(|| Error::invalid_input("injected initializer error"))
        })
    }
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(if returned.is_some() {
        "expected 3"
    } else {
        "injected initializer error"
    }));
    assert_eq!(writer.initialized_mut(), b"ab");
    writer.extend_from_slice(b"cde").unwrap();
    let bytes = writer.finish().unwrap();
    assert_eq!(bytes.as_ref(), b"abcde");
    let alias = bytes.slice(4..5);
    drop(bytes);
    assert_eq!(pool.reserved(), 5);
    assert_eq!(alias.as_ref(), b"e");
    drop(alias);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn refused_foreign_write_does_not_call_initializer() {
    let pool = AllocationBudget::new(2);
    let mut work = pool.try_reserve(2).unwrap();
    let mut writer = work.try_allocate_writer(2).unwrap().unwrap();
    for len in [3, usize::MAX] {
        // SAFETY: the closure never returns and cannot expose any memory.
        let error = unsafe {
            writer.try_extend_with(len, |_| panic!("capacity must be checked before callback"))
        }
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("exceeds reserved allocation length")
        );
    }
    assert!(writer.initialized_mut().is_empty());
    writer.recycle(&mut work).unwrap();
    assert_eq!(work.len(), 2);
}

#[test]
fn panicking_foreign_writer_can_be_recycled_without_releasing_capacity() {
    let pool = AllocationBudget::new(8);
    let mut work = pool.try_reserve(8).unwrap();
    let mut writer = work.try_allocate_writer(8).unwrap().unwrap();
    writer.extend_from_slice(b"abc").unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: only a bounded write, and unwinding cannot publish the suffix.
        unsafe {
            writer.try_extend_with(5, |output| {
                output[0].write(7);
                panic!("injected foreign initializer panic");
            })
        }
        .unwrap();
    }));
    assert!(result.is_err());
    assert_eq!(writer.initialized_mut(), b"abc");
    writer.recycle(&mut work).unwrap();
    assert_eq!(work.len(), 8);
    assert_eq!(pool.reserved(), 8);
    assert!(pool.try_reserve(1).is_none());
    let other = AllocationBudget::new(3);
    let mut foreign = other.try_reserve(3).unwrap();
    let writer = work.try_allocate_writer(8).unwrap().unwrap();
    let error = writer.recycle(&mut foreign).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("different allocation budgets"));
    assert_eq!(foreign.len(), 3);
    assert_eq!(other.reserved(), 3);
    assert_eq!(pool.reserved(), 0);
}
