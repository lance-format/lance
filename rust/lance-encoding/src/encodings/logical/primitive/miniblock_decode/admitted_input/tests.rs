// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::DefaultDecompressionStrategy,
    decoder::PageEncoding,
    format::{EncodingUtils, metadata::page_layout::Layout},
    repdef::DefinitionInterpretation,
};
use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use futures::FutureExt;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;
use std::sync::Mutex;

fn fixture(bits: u8, large: bool, nullable: bool, chunks: usize) -> (PageInfo, Vec<u8>, ArrayRef) {
    let width = usize::from(bits / 8);
    let mut payload = Vec::new();
    for offset in [0, 1, 1, 3, 4] {
        payload.extend_from_slice(&((5 * width + offset) as u64).to_le_bytes()[..width]);
    }
    payload.extend_from_slice(b"abcd");
    payload.resize(payload.len().next_multiple_of(width), 0);
    // The writer emits zero levels when there is no rep/def. Only value-buffer
    // lengths widen with the chunk format; structural lengths stay u16.
    let mut chunk = if nullable { 4_u16 } else { 0 }.to_le_bytes().to_vec();
    if nullable {
        chunk.extend_from_slice(&8_u16.to_le_bytes());
    }
    chunk.extend_from_slice(&(payload.len() as u32).to_le_bytes()[..if large { 4 } else { 2 }]);
    chunk.resize(chunk.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    if nullable {
        chunk.extend_from_slice(bytemuck::cast_slice(&[0_u16, 1, 0, 0]));
        chunk.resize(chunk.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    }
    chunk.extend_from_slice(&payload);
    chunk.resize(chunk.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    let word = (((chunk.len() / MINIBLOCK_ALIGNMENT - 1) as u32) << 4) | 2;
    let mut bytes = Vec::new();
    for _ in 0..chunks {
        bytes.extend_from_slice(&word.to_le_bytes()[..if large { 4 } else { 2 }]);
    }
    let meta = bytes.len();
    for _ in 0..chunks {
        bytes.extend_from_slice(&chunk);
    }
    let meaning = if nullable {
        DefinitionInterpretation::NullableItem
    } else {
        DefinitionInterpretation::AllValidItem
    };
    let page = PageInfo {
        num_rows: (chunks * 4) as u64,
        priority: 0,
        buffer_offsets_and_sizes: Arc::from([
            (0, meta as u64),
            (meta as u64, (chunk.len() * chunks) as u64),
        ]),
        encoding: PageEncoding::Structural(EncodingUtils::miniblock_layout(
            None,
            nullable.then(|| EncodingUtils::flat(16, None)),
            EncodingUtils::variable(EncodingUtils::flat(u64::from(bits), None), None),
            0,
            1,
            None,
            &[meaning],
            (chunks * 4) as u64,
            large,
        )),
    };
    let values = (0..chunks).flat_map(|_| {
        [
            Some(&b"a"[..]),
            if nullable { None } else { Some(&b""[..]) },
            Some(&b"bc"[..]),
            Some(&b"d"[..]),
        ]
    });
    let expected: ArrayRef = if bits == 32 {
        Arc::new(BinaryArray::from_iter(values))
    } else {
        Arc::new(LargeBinaryArray::from_iter(values))
    };
    (page, bytes, expected)
}

fn scheduler(page: &PageInfo) -> MiniBlockArrayScheduler {
    let PageEncoding::Structural(layout) = &page.encoding else {
        panic!("structural fixture");
    };
    let Some(Layout::MiniBlockLayout(layout)) = &layout.layout else {
        panic!("MiniBlock fixture");
    };
    MiniBlockArrayScheduler::try_new(page, layout, &DefaultDecompressionStrategy {}).unwrap()
}

fn read(bytes: &[u8], range: Range<u64>, mut out: ReservedBuffer) -> Result<ReservedBuffer> {
    out.as_mut()
        .copy_from_slice(&bytes[range.start as usize..range.end as usize]);
    Ok(out)
}

#[rstest]
#[case::one(1, 1, 3)]
#[case::cross_chunk(3, 3, 10)]
#[case::late_window(2101, 8399, 8403)]
#[tokio::test]
async fn flat_input_matches_arrow_with_bounded_index(
    #[case] chunks: usize,
    #[case] start: u64,
    #[case] end: u64,
    #[values(32, 64)] bits: u8,
    #[values(false, true)] large: bool,
    #[values(false, true)] nullable: bool,
) {
    let (page, bytes, expected) = fixture(bits, large, nullable, chunks);
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
    let ranges = reads.into_inner().unwrap();
    let boundary = page.buffer_offsets_and_sizes[1].0;
    assert!(
        ranges
            .iter()
            .filter(|r| r.start < boundary)
            .all(|r| r.end - r.start <= 4096)
    );
    let expected_chunks = (end - 1) / 4 - start / 4 + 1;
    assert_eq!(
        ranges.iter().filter(|r| r.start >= boundary).count() as u64,
        expected_chunks
    );
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(
        output.to_data(),
        expected
            .slice(start as usize, (end - start) as usize)
            .to_data()
    );
    drop(work);
    let alias = output
        .to_data()
        .buffers()
        .last()
        .unwrap()
        .slice_with_length(0, 1);
    drop(output);
    assert!(budget.reserved() > 0 && budget.reserved() <= 8192);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::no_metadata(0)]
#[case::short_metadata(3)]
#[case::short_inputs(63)]
#[case::short_output(65)]
#[tokio::test]
async fn flat_input_refusal_discards_whole_attempt(#[case] capacity: usize) {
    let (page, bytes, expected) = fixture(32, false, false, 2);
    let scheduler = scheduler(&page);
    // Two 32-byte chunks. Metadata is recycled before input admission.
    assert_eq!(page.buffer_offsets_and_sizes[1].1, 64);
    let budget = AllocationBudget::new(capacity);
    let mut work = budget.try_reserve(capacity).unwrap();
    let input = scheduler
        .prepare(
            0..8,
            expected.data_type().clone(),
            &mut work,
            |range, out| std::future::ready(read(&bytes, range, out)),
        )
        .await
        .unwrap();
    if capacity <= 63 {
        assert!(input.is_none());
    } else {
        assert!(input.unwrap().decode_in(&mut work).unwrap().is_none());
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::zero_nonfinal(0, 0)]
#[case::wrong_final(2, 3)]
#[case::size_suffix(2, 0xfff2)]
#[tokio::test]
async fn unselected_metadata_corruption_precedes_payload_reads(
    #[case] index: usize,
    #[case] word: u16,
) {
    let (page, mut bytes, expected) = fixture(32, false, false, 3);
    bytes[index * 2..index * 2 + 2].copy_from_slice(&word.to_le_bytes());
    let scheduler = scheduler(&page);
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let error = scheduler
        .prepare(
            0..1,
            expected.data_type().clone(),
            &mut work,
            |range, out| {
                assert!(range.end <= page.buffer_offsets_and_sizes[1].0);
                std::future::ready(read(&bytes, range, out))
            },
        )
        .await
        .err()
        .unwrap();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("chunk"));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::metadata(false)]
#[case::second_payload(true)]
#[tokio::test]
async fn cancelled_input_releases_all_owned_windows(#[case] payload: bool) {
    let (page, bytes, expected) = fixture(32, false, false, 2);
    let scheduler = scheduler(&page);
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let blocked = std::cell::Cell::new(false);
    let second_chunk = page.buffer_offsets_and_sizes[1].0 + 32;
    let mut pending = Box::pin(scheduler.prepare(
        0..8,
        expected.data_type().clone(),
        &mut work,
        |range, out| {
            let bytes = &bytes;
            let blocked = &blocked;
            async move {
                if !payload || range.start >= second_chunk {
                    blocked.set(true);
                    std::future::pending::<()>().await;
                }
                read(bytes, range, out)
            }
        },
    ));
    // The index traversal yields between windows as well as at blocked I/O.
    while !blocked.get() {
        assert!(pending.as_mut().now_or_never().is_none());
    }
    assert_eq!(budget.reserved(), 8192);
    drop(pending);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn malformed_page_descriptors_are_rejected_before_reads() {
    let (mut page, _, _) = fixture(32, false, false, 2);
    let PageEncoding::Structural(layout) = &page.encoding else {
        panic!("structural");
    };
    let Some(Layout::MiniBlockLayout(layout)) = &layout.layout else {
        panic!("miniblock");
    };
    let layout = layout.clone();
    for buffers in [
        vec![(0, 3), (3, 64)],
        vec![(u64::MAX, 4), (0, 64)],
        vec![(0, 4)],
    ] {
        page.buffer_offsets_and_sizes = buffers.into();
        let error =
            MiniBlockArrayScheduler::try_new(&page, &layout, &DefaultDecompressionStrategy {})
                .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("MiniBlock input"));
    }
}

#[tokio::test]
async fn chunk_header_count_is_checked_before_decode_output() {
    let (page, mut bytes, expected) = fixture(32, false, true, 2);
    let start = page.buffer_offsets_and_sizes[1].0 as usize;
    bytes[start..start + 2].copy_from_slice(&3_u16.to_le_bytes());
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let input = scheduler(&page)
        .prepare(
            0..1,
            expected.data_type().clone(),
            &mut work,
            |range, out| std::future::ready(read(&bytes, range, out)),
        )
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("level header"));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::one_byte_short(40, false)]
#[case::exact(41, true)]
#[tokio::test]
async fn original_workspace_covers_input_and_output_overlap(
    #[case] bytes_available: usize,
    #[case] fits: bool,
) {
    let (page, bytes, expected) = fixture(32, false, false, 1);
    // One 32-byte input chunk overlaps two i32 offsets and the selected byte.
    let budget = AllocationBudget::new(bytes_available);
    let mut work = budget.try_reserve(bytes_available).unwrap();
    let input = scheduler(&page)
        .prepare(0..1, DataType::Binary, &mut work, |range, out| {
            std::future::ready(read(&bytes, range, out))
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), bytes_available - 32);
    let output = input.decode_in(&mut work).unwrap();
    assert_eq!(output.is_some(), fits);
    drop(work);
    if let Some(output) = output {
        assert_eq!(output.to_data(), expected.slice(0, 1).to_data());
        assert_eq!(budget.reserved(), 9);
        let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
        drop(output);
        assert_eq!(budget.reserved(), 8);
        drop(alias);
    }
    assert_eq!(budget.reserved(), 0);
}
