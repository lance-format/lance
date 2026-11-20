// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[tokio::test]
async fn compact_page_read_recycles_copy_overlap_before_decode(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] fsst: bool,
    #[values("flat", "list", "deep")] kind: &str,
    #[values(0, 16, 4096)] gap: usize,
) {
    let mut f = fixture(bits, fsst, kind);
    let source = source_with_gap(&mut f, gap);
    let (input, stage, decoded) = sizes(&f, 0..4);
    let physical_read = input + gap + f.index.len();
    let limit = (physical_read + input)
        .max(input + stage)
        .max(stage + decoded);
    let budget = AllocationBudget::new(limit);
    let mut work = budget.try_reserve(limit).unwrap();
    let reads = Arc::new(Reads::default());
    let task = f
        .scheduler
        .try_read_unzip_range_in(0..4, &mut work, reader(source, reads.clone(), Fault::None))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        *reads.0.lock().unwrap(),
        vec![37..37 + physical_read as u64]
    );
    assert_eq!(
        work.len(),
        limit - stage,
        "whole backing and data copy retired before decode"
    );
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let array = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    let refs = f.values.iter().map(Vec::as_slice).collect();
    let expected = if bits == 32 {
        BinaryArray::from_vec(refs).to_data()
    } else {
        LargeBinaryArray::from_vec(refs).to_data()
    };
    assert_eq!(array, expected);
    let alias = array.slice(0, array.len());
    drop((array, page, task, work));
    assert!(budget.reserved() > 0);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::tight_credit("tight")]
#[case::distant_index("gap")]
#[case::large_page("large")]
#[case::partial_selection("partial")]
#[tokio::test]
async fn compact_read_fallback_preserves_original_progress(#[case] reason: &str) {
    let value_len = if reason == "large" { 131072 } else { 16384 };
    let mut f = fixture_with_values(
        32,
        false,
        "flat",
        vec![vec![b'a'; value_len], vec![], vec![b'b'; value_len], vec![]],
    );
    let gap = if reason == "gap" { 4097 } else { 4096 };
    let source = source_with_gap(&mut f, gap);
    let rows = if reason == "partial" { 1..3 } else { 0..4 };
    let (input, stage, decoded) = sizes(&f, rows.clone());
    let limit = if reason == "tight" {
        2 * input + gap + f.index.len() - 1
    } else {
        (input + stage).max(stage + decoded) + 8192
    };
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
    assert_eq!(
        *reads.0.lock().unwrap(),
        vec![
            index.buf_position + rows.start * index.bytes_per_value
                ..index.buf_position + (rows.end + 1) * index.bytes_per_value,
            37 + f.row_offsets[rows.start as usize] as u64
                ..37 + f.row_offsets[rows.end as usize] as u64,
        ]
    );
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let array = page
        .data
        .try_to_arrow_in(&DataType::Binary, &mut work)
        .unwrap()
        .unwrap();
    let values = f.values[rows.start as usize..rows.end as usize]
        .iter()
        .map(Vec::as_slice)
        .collect();
    assert_eq!(array, BinaryArray::from_vec(values).to_data());
    drop((array, page, task, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn compact_read_validates_stored_endpoints_without_retry(#[values(false, true)] first: bool) {
    let mut f = fixture(32, false, "flat");
    let mut source = source_with_gap(&mut f, 0);
    let index = f.scheduler.rep_index.unwrap();
    let width = index.bytes_per_value as usize;
    let offset = index.buf_position as usize + if first { 0 } else { 4 * width };
    Arc::make_mut(&mut source)[offset..offset + width]
        .copy_from_slice(&1_u64.to_le_bytes()[..width]);
    let budget = AllocationBudget::new(262144);
    let reads = Arc::new(Reads::default());
    let result = f
        .scheduler
        .try_read_decode_prefix_in(
            0..4,
            &budget,
            262144,
            reader(source, reads.clone(), Fault::None),
        )
        .await;
    let Err(error) = result else {
        panic!("expected stored endpoint corruption")
    };
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("repetition index endpoints"));
    assert_eq!(reads.0.lock().unwrap().len(), 1);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn compact_pending_read_charges_both_allocations_and_allows_peer(
    #[values(false, true)] cancel: bool,
) {
    let mut f = fixture(32, false, "flat");
    let source = source_with_gap(&mut f, 16);
    let (input, stage, decoded) = sizes(&f, 0..4);
    let physical_read = input + 16 + f.index.len();
    let limit = (physical_read + input)
        .max(input + stage)
        .max(stage + decoded);
    let budget = AllocationBudget::new(2 * limit);
    let mut work = budget.try_reserve(limit).unwrap();
    let reads = Arc::new(Reads::default());
    let fault = if cancel {
        Fault::Pending(0)
    } else {
        Fault::Error(0)
    };
    let mut future = Box::pin(f.scheduler.try_read_unzip_range_in(
        0..4,
        &mut work,
        reader(source.clone(), reads.clone(), fault),
    ));
    if cancel {
        assert!(poll!(future.as_mut()).is_pending());
        assert_eq!(budget.reserved(), limit);
        let mut peer = budget.try_reserve(limit).unwrap();
        assert!(budget.try_reserve(1).is_none());
        let task = f
            .scheduler
            .try_read_unzip_range_in(0..4, &mut peer, reader(source, Arc::default(), Fault::None))
            .await
            .unwrap()
            .unwrap();
        let page = task.try_decode_in(&mut peer).unwrap().unwrap();
        drop((page, task, peer));
        assert_eq!(budget.reserved(), limit);
    } else {
        let Err(error) = future.as_mut().await else {
            panic!("expected read error")
        };
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("injected read error"));
    }
    assert_eq!(reads.0.lock().unwrap().len(), 1);
    drop(future);
    assert_eq!(work.len(), limit - physical_read - input);
    assert_eq!(budget.reserved(), work.len());
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
