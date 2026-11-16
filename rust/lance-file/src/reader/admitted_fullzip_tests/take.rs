// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::scan::{GateIo, RangeIo};
use super::*;
use arrow_array::{UInt32Array, cast::AsArray};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

#[rstest]
#[tokio::test]
async fn sparse_take_matches_arrow_and_skips_pages(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(0, 2)] depth: usize,
) {
    let pages = [array(0, depth), array(1, depth), array(2, depth)];
    let n = pages[0].len() as u32;
    let indices = [0, 1, n - 1, 2 * n, 3 * n - 1];
    let expected = arrow_select::take::take(
        arrow_select::concat::concat(&pages.iter().map(AsRef::as_ref).collect::<Vec<_>>())
            .unwrap()
            .as_ref(),
        &UInt32Array::from(indices.to_vec()),
        None,
    )
    .unwrap();
    let (_fs, reader) = fixture_arrays(version, lite, pages.to_vec()).await;
    assert_eq!(reader.metadata().column_infos[0].page_infos.len(), 3);
    let skipped = reader.metadata().column_infos[0].page_infos[1]
        .buffer_offsets_and_sizes
        .clone();
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(65536);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    assert!(work.len() < 65536, "prepared input owns physical credit");
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.column(0).to_data(), expected.to_data());
    assert!(!reads.lock().unwrap().is_empty());
    for read in reads.lock().unwrap().iter() {
        for (offset, len) in skipped.iter() {
            assert!(
                read.end <= *offset || read.start >= offset + len,
                "read {read:?} touched a skipped page"
            );
        }
    }
    let alias = output.column(0).to_data().buffers()[0].slice_with_length(0, 1);
    drop((output, work));
    assert!(budget.reserved() > 1);
    assert!(budget.try_reserve(65536).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    // An independent next operation can acquire the same finite workspace.
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.column(0).to_data(), expected.to_data());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn sparse_take_preflights_all_selected_pages(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
) {
    let (_fs, reader) = fixture_arrays(
        version,
        false,
        vec![array(0, 0), Arc::new(LargeBinaryArray::new_null(8))],
    )
    .await;
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    for invalid in [vec![1, 1], vec![2, 0], vec![16]] {
        let error = reader
            .prepare_admitted_fullzip_take_in(&invalid, &mut work)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("FullZip"));
    }
    let error = reader
        .prepare_admitted_fullzip_take_in(&[0, 8], &mut work)
        .await
        .err()
        .unwrap();
    assert!(matches!(error, Error::NotSupported { .. }));
    // v2.2's all-null Constant page has no byte scalar. v2.3 emits a
    // planar sparse layout. Both must fail preflight without payload reads.
    let expected_reason = if version == ConcreteFileVersion::V2_2 {
        "admitted Constant pages require a stored byte scalar"
    } else {
        "expected a dense FullZip page"
    };
    assert!(error.to_string().contains(expected_reason));
    assert!(
        reads.lock().unwrap().is_empty(),
        "later selected layout must fail before earlier reads"
    );
    assert_eq!(work.len(), 8192);
    let input = reader
        .prepare_admitted_fullzip_take_in(&[0], &mut work)
        .await
        .unwrap()
        .unwrap();
    assert!(input.decode_in(&mut work).unwrap().is_some());
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn sparse_take_empty_and_refused_attempts_release(#[values(0, 2)] depth: usize) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_3, false, depth, 2).await;
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(65536);
    let mut zero = budget.try_reserve(0).unwrap();
    assert!(
        reader
            .prepare_admitted_fullzip_take_in(&[0], &mut zero)
            .await
            .unwrap()
            .is_none()
    );
    let input = reader
        .prepare_admitted_fullzip_take_in(&[], &mut zero)
        .await
        .unwrap()
        .unwrap();
    assert!(
        input.decode_in(&mut zero).unwrap().is_none(),
        "empty typed offsets are not free"
    );
    assert!(reads.lock().unwrap().is_empty());
    drop(zero);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[], &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(
        output.column(0).to_data(),
        array(0, depth).slice(0, 0).to_data()
    );
    assert!(work.len() < 65536);
    assert!(reads.lock().unwrap().is_empty());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[0, 2], &mut work)
        .await
        .unwrap()
        .unwrap();
    let held = work.try_split(work.len()).unwrap();
    assert!(input.decode_in(&mut work).unwrap().is_none());
    drop((held, work));
    assert_eq!(
        budget.reserved(),
        0,
        "failed decode/concat retains no partial output"
    );
}

#[rstest]
#[tokio::test]
async fn sparse_take_cancel_error_and_retry(#[values(false, true)] lite: bool) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, lite, 0, 2).await;
    let block = Arc::new(AtomicBool::new(true));
    let fail = Arc::new(AtomicBool::new(false));
    let entered = Arc::new(tokio::sync::Notify::new());
    let reader = reader.with_scheduler(Arc::new(GateIo {
        inner: reader.core.scheduler.clone(),
        block: block.clone(),
        fail: fail.clone(),
        entered: entered.clone(),
    }));
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    {
        let future = reader.prepare_admitted_fullzip_take_in(&[0, 8], &mut work);
        tokio::pin!(future);
        tokio::select! {
            _ = entered.notified() => (),
            _ = &mut future => panic!("gated input must stay pending"),
        }
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
    block.store(false, Ordering::Relaxed);
    fail.store(true, Ordering::Relaxed);
    let mut work = budget.try_reserve(8192).unwrap();
    let error = reader
        .prepare_admitted_fullzip_take_in(&[0, 8], &mut work)
        .await
        .err()
        .unwrap();
    assert!(matches!(error, Error::IO { .. }));
    assert!(
        error
            .to_string()
            .contains("injected admitted scan read failure")
    );
    drop(work);
    assert_eq!(budget.reserved(), 0);
    fail.store(false, Ordering::Relaxed);
    let mut work = budget.try_reserve(8192).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[0, 8], &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    let values = output.column(0).as_binary::<i64>();
    assert_eq!(values.value(0), array(0, 0).as_binary::<i64>().value(0));
    assert_eq!(values.value(1), array(1, 0).as_binary::<i64>().value(0));
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}
