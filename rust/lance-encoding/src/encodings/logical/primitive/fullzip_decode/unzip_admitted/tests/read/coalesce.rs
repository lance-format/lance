// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// Re-encode the writer-derived endpoints at each supported index width.
// The payload, row boundaries and expected values still come from the writer.
fn indexed_fixture(rows: usize, width: usize) -> Fixture {
    let values = (0..rows).map(|row| vec![row as u8; 16]).collect();
    let mut f = fixture_with_values(32, false, "flat", values);
    f.index = f
        .row_offsets
        .iter()
        .flat_map(|offset| {
            let value = (*offset as u64).to_le_bytes();
            assert!(width == 8 || (*offset as u64) < (1_u64 << (8 * width)));
            value[..width].to_vec()
        })
        .collect::<Vec<_>>()
        .into();
    f.scheduler.rep_index.as_mut().unwrap().bytes_per_value = width as u64;
    f
}

#[rstest]
#[case::one_byte(1, 0..4, true)]
#[case::two_bytes(2, 0..4, true)]
#[case::four_bytes(4, 0..4, true)]
#[case::eight_bytes(8, 0..4, true)]
#[case::at_cap(8, 0..511, true)]
#[case::shifted_at_cap(8, 1..512, true)]
#[case::above_cap(8, 0..512, false)]
#[tokio::test]
async fn index_span_bounds_reads_and_preserves_rows(
    #[case] width: usize,
    #[case] rows: Range<u64>,
    #[case] combined: bool,
) {
    let mut f = indexed_fixture(rows.end as usize, width);
    let source = source(&mut f);
    let (input, stage, decoded) = sizes(&f, rows.clone());
    let limit = (input + stage).max(stage + decoded);
    let budget = AllocationBudget::new(limit);
    let mut work = budget.try_reserve(limit).unwrap();
    let reads = Arc::new(Reads::default());
    let task = f
        .scheduler
        .try_read_unzip_range_in(
            rows.clone(),
            &mut work,
            reader(source, reads.clone(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let index = f.scheduler.rep_index.unwrap();
    let first = index.buf_position + rows.start * width as u64;
    let last = index.buf_position + rows.end * width as u64;
    let mut expected = if combined {
        vec![first..last + width as u64]
    } else {
        vec![first..first + width as u64, last..last + width as u64]
    };
    expected.push(
        37 + f.row_offsets[rows.start as usize] as u64
            ..37 + f.row_offsets[rows.end as usize] as u64,
    );
    assert_eq!(*reads.0.lock().unwrap(), expected);
    assert_eq!(
        work.len(),
        limit - stage,
        "index and input scratch recycled"
    );
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let arrow = page
        .data
        .try_to_arrow_in(&DataType::Binary, &mut work)
        .unwrap()
        .unwrap();
    let values = f.values[rows.start as usize..rows.end as usize]
        .iter()
        .map(Vec::as_slice)
        .collect();
    assert_eq!(arrow, BinaryArray::from_vec(values).to_data());
    drop((task, page, arrow, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn index_coalescing_does_not_require_more_credit(#[values(8, 39)] available: usize) {
    let mut f = indexed_fixture(4, 8);
    let source = source(&mut f);
    let budget = AllocationBudget::new(available);
    let mut work = budget.try_reserve(available).unwrap();
    let reads = Arc::new(Reads::default());
    // Both valid endpoints must be read using one reused entry. Payload refusal
    // then restores all credit, even one byte below the combined index span.
    assert!(
        f.scheduler
            .read_selected_range(0..4, &mut work, reader(source, reads.clone(), Fault::None))
            .await
            .unwrap()
            .is_none()
    );
    let index = f.scheduler.rep_index.unwrap().buf_position;
    assert_eq!(
        *reads.0.lock().unwrap(),
        vec![index..index + 8, index + 32..index + 40]
    );
    assert_eq!(work.len(), available);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn separate_endpoint_failures_and_cancellation_refund_owners(
    #[values(0, 1, 2)] at: usize,
    #[values(false, true)] cancel: bool,
) {
    let mut f = indexed_fixture(512, 8);
    let source = source(&mut f);
    let (input, stage, _) = sizes(&f, 0..512);
    let limit = input + stage;
    let budget = AllocationBudget::new(limit);
    let mut work = budget.try_reserve(limit).unwrap();
    let reads = Arc::new(Reads::default());
    let fault = if cancel {
        Fault::Pending(at)
    } else {
        Fault::Error(at)
    };
    let mut future = Box::pin(f.scheduler.try_read_unzip_range_in(
        0..512,
        &mut work,
        reader(source, reads.clone(), fault),
    ));
    if cancel {
        assert!(poll!(future.as_mut()).is_pending());
        assert_eq!(budget.reserved(), limit);
    } else {
        let Err(error) = future.as_mut().await else {
            panic!("expected read error")
        };
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("injected read error"));
    }
    assert_eq!(reads.0.lock().unwrap().len(), at + 1);
    drop(future);
    assert_eq!(budget.reserved(), work.len());
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
