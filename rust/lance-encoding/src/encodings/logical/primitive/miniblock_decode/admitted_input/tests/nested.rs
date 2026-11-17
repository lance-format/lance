// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::ListArray;
use arrow_buffer::OffsetBuffer;
use arrow_schema::Field;

// Row 1 crosses three chunks, including one entirely composed of its continuation.
// Row 2 ends exactly at a chunk boundary. No level or index comes from the code tested.
fn nested_fixture(repeats: usize, large: bool) -> (PageInfo, Vec<u8>, ArrayRef) {
    let levels = [[1_u16, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]];
    let summaries = [(1_u64, 2_u64), (0, 4), (2, 0), (4, 0)];
    let mut chunks = Vec::new();
    let mut meta = Vec::new();
    let mut index = Vec::new();
    for _ in 0..repeats {
        for (rep, (ends, partial)) in levels.iter().zip(summaries) {
            let mut values = Vec::new();
            for offset in [20_u32, 21, 21, 23, 24] {
                values.extend_from_slice(&offset.to_le_bytes());
            }
            values.extend_from_slice(b"abcd");
            let mut chunk = 4_u16.to_le_bytes().to_vec();
            chunk.extend_from_slice(&8_u16.to_le_bytes());
            chunk.extend_from_slice(&24_u32.to_le_bytes()[..if large { 4 } else { 2 }]);
            chunk.resize(8, 0);
            chunk.extend_from_slice(bytemuck::cast_slice(rep));
            chunk.extend_from_slice(&values);
            let word = (((chunk.len() / 8 - 1) as u32) << 4) | 2;
            meta.extend_from_slice(&word.to_le_bytes()[..if large { 4 } else { 2 }]);
            chunks.extend_from_slice(&chunk);
            index.extend_from_slice(&ends.to_le_bytes());
            index.extend_from_slice(&partial.to_le_bytes());
        }
    }
    let buffers = [
        (0, meta.len() as u64),
        (meta.len() as u64, chunks.len() as u64),
        ((meta.len() + chunks.len()) as u64, index.len() as u64),
    ];
    let bytes = [meta, chunks, index].concat();
    let page = PageInfo {
        num_rows: (7 * repeats) as u64,
        priority: 0,
        buffer_offsets_and_sizes: Arc::from(buffers),
        encoding: PageEncoding::Structural(EncodingUtils::miniblock_layout(
            Some(EncodingUtils::flat(16, None)),
            None,
            EncodingUtils::variable(EncodingUtils::flat(32, None), None),
            1,
            1,
            None,
            &[
                DefinitionInterpretation::AllValidItem,
                DefinitionInterpretation::AllValidList,
            ],
            (16 * repeats) as u64,
            large,
        )),
    };
    let values: ArrayRef = Arc::new(BinaryArray::from_iter_values(
        (0..repeats * 4).flat_map(|_| [&b"a"[..], &b""[..], &b"bc"[..], &b"d"[..]]),
    ));
    let offsets = std::iter::once(0_i32)
        .chain(
            (0..repeats)
                .flat_map(|group| [2, 10, 12, 13, 14, 15, 16].map(|n| group as i32 * 16 + n)),
        )
        .collect::<Vec<_>>();
    let expected = Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Binary, false)),
        OffsetBuffer::new(offsets.into()),
        values,
        None,
    ));
    (page, bytes, expected)
}

#[rstest]
#[case::truncated(false, "two u64 words")]
#[case::overflow(true, "overflows")]
fn nested_index_extent_is_validated_before_io(#[case] overflow: bool, #[case] message: &str) {
    let (mut page, _, _) = nested_fixture(1, false);
    let mut buffers = page.buffer_offsets_and_sizes.to_vec();
    if overflow {
        buffers[2] = (u64::MAX, 64);
    } else {
        buffers[2].1 -= 1;
    }
    page.buffer_offsets_and_sizes = buffers.into();
    let PageEncoding::Structural(layout) = &page.encoding else {
        panic!("structural");
    };
    let Some(Layout::MiniBlockLayout(layout)) = &layout.layout else {
        panic!("MiniBlock");
    };
    let error = MiniBlockArrayScheduler::try_new(&page, layout, &DefaultDecompressionStrategy {})
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains(message));
}

#[rstest]
#[case::all(1, 0, 7)]
#[case::split_row(1, 1, 2)]
#[case::skip_preamble(1, 2, 3)]
#[case::ends_before_trailer(1, 0, 1)]
#[case::late_window(129, 897, 901)]
#[tokio::test]
async fn nested_input_preserves_complete_rows(
    #[case] repeats: usize,
    #[case] start: u64,
    #[case] end: u64,
    #[values(false, true)] large: bool,
) {
    let (page, bytes, expected) = nested_fixture(repeats, large);
    let scheduler = scheduler(&page);
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let reads = Mutex::new(Vec::new());
    let input = scheduler
        .prepare(
            start..end,
            expected.data_type().clone(),
            &mut work,
            |range, out| {
                reads.lock().unwrap().push(range.clone());
                std::future::ready(read(&bytes, range, out))
            },
        )
        .await
        .unwrap()
        .unwrap();
    assert!(
        reads
            .lock()
            .unwrap()
            .iter()
            .all(|range| range.end - range.start <= 4096)
    );
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(
        output.to_data(),
        expected
            .slice(start as usize, (end - start) as usize)
            .to_data()
    );
    let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
    drop((work, output));
    assert!(budget.reserved() > 1);
    assert!(budget.try_reserve(8192).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    assert!(budget.try_reserve(8192).is_some());
}

#[rstest]
#[case::unfinished_tail(3, 1, 1, "unfinished final row")]
#[case::overflow(3, 0, u64::MAX, "exceeds page rows")]
#[case::underflow(1, 1, 0, "underflow")]
#[case::missing_row(3, 0, 3, "exact row/item/data domain")]
#[tokio::test]
async fn corrupt_unselected_index_is_rejected_before_data(
    #[case] chunk: usize,
    #[case] word: usize,
    #[case] value: u64,
    #[case] message: &str,
) {
    let (page, mut bytes, expected) = nested_fixture(1, false);
    let pos = page.buffer_offsets_and_sizes[2].0 as usize + chunk * 16 + word * 8;
    bytes[pos..pos + 8].copy_from_slice(&value.to_le_bytes());
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let error = scheduler(&page)
        .prepare(
            0..1,
            expected.data_type().clone(),
            &mut work,
            |range, out| {
                assert!(
                    range.end <= page.buffer_offsets_and_sizes[1].0
                        || range.start >= page.buffer_offsets_and_sizes[2].0
                );
                std::future::ready(read(&bytes, range, out))
            },
        )
        .await
        .err()
        .unwrap();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains(message), "{error}");
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::row_boundary(0, 0, 0, "row boundaries")]
#[case::trailer(0, 2, 0, "row boundaries")]
#[tokio::test]
async fn selected_index_must_agree_with_actual_levels(
    #[case] chunk: usize,
    #[case] level: usize,
    #[case] value: u16,
    #[case] message: &str,
) {
    let (page, mut bytes, expected) = nested_fixture(1, false);
    let pos = page.buffer_offsets_and_sizes[1].0 as usize + chunk * 40 + 8 + level * 2;
    bytes[pos..pos + 2].copy_from_slice(&value.to_le_bytes());
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let input = scheduler(&page)
        .prepare(
            0..2,
            expected.data_type().clone(),
            &mut work,
            |range, out| std::future::ready(read(&bytes, range, out)),
        )
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains(message));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::before_second_index(1)]
#[case::during_payload(3)]
#[tokio::test]
async fn nested_input_cancellation_releases_overlapping_owners(#[case] pending_read: usize) {
    let (page, bytes, expected) = nested_fixture(1, false);
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let scheduler = scheduler(&page);
    {
        let mut calls = 0;
        let (sent, received) = tokio::sync::oneshot::channel();
        let mut sent = Some(sent);
        let future = scheduler.prepare(
            0..7,
            expected.data_type().clone(),
            &mut work,
            |range, out| {
                let pending = calls == pending_read;
                calls += 1;
                if pending {
                    sent.take().unwrap().send(()).unwrap();
                }
                let bytes = &bytes;
                async move {
                    if pending {
                        std::future::pending::<()>().await;
                    }
                    read(bytes, range, out)
                }
            },
        );
        tokio::pin!(future);
        tokio::select! {
            result = &mut future => panic!("read must be pending: {result:?}"),
            result = received => result.unwrap(),
        }
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::short(71, false)]
#[case::exact(72, true)]
#[tokio::test]
async fn paired_index_windows_are_preflighted_before_reads(
    #[case] size: usize,
    #[case] fits: bool,
) {
    let (page, bytes, expected) = nested_fixture(1, false);
    // Four 2-byte metadata words plus four 16-byte repetition entries.
    let budget = AllocationBudget::new(size);
    let mut work = budget.try_reserve(size).unwrap();
    let mut reads = 0;
    let input = scheduler(&page)
        .prepare(
            0..1,
            expected.data_type().clone(),
            &mut work,
            |range, out| {
                reads += 1;
                std::future::ready(read(&bytes, range, out))
            },
        )
        .await
        .unwrap();
    assert_eq!(input.is_some(), fits);
    assert_eq!(reads, if fits { 3 } else { 0 });
    if fits {
        assert_eq!(work.len(), size - 40);
    }
    drop((work, input));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn wrong_partial_length_is_not_a_valid_index() {
    let (page, mut bytes, expected) = nested_fixture(1, false);
    let position = page.buffer_offsets_and_sizes[2].0 as usize + 8;
    bytes[position..position + 8].copy_from_slice(&1_u64.to_le_bytes());
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let input = scheduler(&page)
        .prepare(
            0..2,
            expected.data_type().clone(),
            &mut work,
            |range, out| std::future::ready(read(&bytes, range, out)),
        )
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("trailer length"));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn admitted_input_does_not_borrow_more_credit_for_decode() {
    let (page, bytes, expected) = nested_fixture(1, false);
    // The row spans exactly three 40-byte chunks. No credit remains for levels.
    let budget = AllocationBudget::new(120);
    let mut work = budget.try_reserve(120).unwrap();
    let input = scheduler(&page)
        .prepare(
            1..2,
            expected.data_type().clone(),
            &mut work,
            |range, out| std::future::ready(read(&bytes, range, out)),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), 0);
    assert!(input.decode_in(&mut work).unwrap().is_none());
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
