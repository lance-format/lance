// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::OutOfLineBitpacking;
use crate::{
    buffer::LanceBuffer,
    compression::BlockCompressor,
    data::{BlockInfo, FixedWidthDataBlock},
    encodings::physical::{constant::ConstantEncoder, rle::BlockRleEncoder, value::ValueEncoder},
};
use futures::{FutureExt, poll};
use std::{cell::Cell, task::Poll};

#[derive(Clone, Copy)]
enum Children {
    Flat,
    RangeValues,
    RangeLengths,
    ConstantLengths,
    Metadata,
    PackedValues,
    PackedLengths,
}

fn fixture(children: Children, large: bool) -> (PageInfo, Vec<u8>, ArrayRef, Vec<u64>) {
    let mut value_codec: Box<dyn BlockCompressor> = Box::<ValueEncoder>::default();
    let mut length_codec: Box<dyn BlockCompressor> = Box::<ValueEncoder>::default();
    let metadata_values = matches!(children, Children::RangeValues | Children::Metadata);
    let runs = if metadata_values || matches!(children, Children::RangeLengths) {
        17
    } else if matches!(children, Children::PackedLengths) {
        2024 // 1000-value raw tail is larger than its preceding packed window
    } else {
        1025
    };
    match children {
        Children::RangeValues => value_codec = Box::new(RangeEncoder::new(64, 1, 1)),
        Children::Metadata => {
            value_codec = Box::new(RangeEncoder::new(64, 1, 1));
            length_codec = Box::new(ConstantEncoder::new(32, 2));
        }
        Children::RangeLengths => length_codec = Box::new(RangeEncoder::new(32, 1, 1)),
        Children::ConstantLengths => length_codec = Box::new(ConstantEncoder::new(32, 2)),
        #[cfg(feature = "bitpacking")]
        Children::PackedValues => value_codec = Box::new(OutOfLineBitpacking::new(5, 64)),
        #[cfg(feature = "bitpacking")]
        Children::PackedLengths => length_codec = Box::new(OutOfLineBitpacking::new(2, 32)),
        _ => {}
    }
    let counts = (0..runs)
        .flat_map(|i| {
            let value = if metadata_values {
                1 + i as u64
            } else {
                1 + (i as u64 * 17 % 31)
            };
            let length = match children {
                Children::RangeLengths => i + 1,
                Children::ConstantLengths | Children::Metadata => 2,
                _ => 1 + i % 3,
            };
            std::iter::repeat_n(value, length)
        })
        .collect::<Vec<_>>();
    let codec = BlockRleEncoder::try_new(64, value_codec, length_codec).unwrap();
    let (bytes, encoding) = codec
        .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            num_values: counts.len() as u64,
            data: LanceBuffer::reinterpret_vec(counts.clone()),
            block_info: BlockInfo::default(),
        }))
        .unwrap();
    let bytes = bytes.map(|b| b.to_vec()).unwrap_or_default();
    let (mut page, source) = payload::payload_page(&counts, encoding, bytes.len(), large);
    if bytes.is_empty() {
        page.buffer_offsets_and_sizes = Arc::from([]);
    }
    (page, bytes, source, counts)
}

#[rstest]
#[case::flat(Children::Flat)]
#[case::range_values(Children::RangeValues)]
#[case::range_lengths(Children::RangeLengths)]
#[case::constant_lengths(Children::ConstantLengths)]
#[case::metadata(Children::Metadata)]
#[case::packed_values(Children::PackedValues)]
#[case::packed_lengths(Children::PackedLengths)]
#[tokio::test]
async fn windowed_rle_preserves_selected_prefixes(
    #[case] children: Children,
    #[values(false, true)] large: bool,
) {
    if matches!(children, Children::PackedValues | Children::PackedLengths)
        && !cfg!(feature = "bitpacking")
    {
        return;
    }
    let (page, bytes, source, _) = fixture(children, large);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(16384);
    for rows in [
        0..5,
        page.num_rows - 5..page.num_rows,
        1..2,                             // a null parent selects no counts
        3..4,                             // a valid empty parent selects no counts
        page.num_rows - 1..page.num_rows, // empty parent at the terminal prefix
    ] {
        let reads = Cell::new(0);
        let mut work = budget.try_reserve(16384).unwrap();
        let input = scheduler
            .prepare(
                rows.clone(),
                source.data_type().clone(),
                &mut work,
                |r, mut buffer| {
                    assert!(buffer.capacity() <= 8192);
                    reads.set(reads.get() + buffer.capacity());
                    buffer
                        .as_mut()
                        .copy_from_slice(&bytes[(r.start - 17) as usize..(r.end - 17) as usize]);
                    std::future::ready(Ok(buffer))
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reads.get(), bytes.len()); // header and both streams once, no replay
        assert!(16384 - work.len() <= 4 * 8);
        let output = input.decode_in(&mut work).unwrap().unwrap();
        assert_eq!(
            output.to_data(),
            source
                .slice(rows.start as usize, (rows.end - rows.start) as usize)
                .to_data()
        );
        let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
        drop((output, work));
        assert!(budget.try_reserve(16384).is_none());
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[case::header(0, "frame exceeds")]
#[case::zero_length(1, "positive")]
#[case::wrong_length_total(2, "cardinality")]
#[case::zero_value(3, "positive")]
#[case::wrong_child_total(4, "child domain")]
#[tokio::test]
async fn late_rle_corruption_cannot_hide_outside_selection(
    #[case] kind: u8,
    #[case] message: &str,
) {
    let (page, mut bytes, source, _) = fixture(Children::Flat, false);
    let split = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let end = bytes.len();
    match kind {
        0 => bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes()),
        1 => bytes[end - 4..].copy_from_slice(&0_u32.to_le_bytes()),
        2 => bytes[end - 4..].copy_from_slice(&99_u32.to_le_bytes()),
        3 => bytes[split - 8..split].copy_from_slice(&0_u64.to_le_bytes()),
        4 => bytes[split - 8..split].copy_from_slice(&99_u64.to_le_bytes()),
        _ => unreachable!(),
    }
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(16384);
    let mut work = budget.try_reserve(16384).unwrap();
    let error = scheduler
        .prepare(
            0..1,
            source.data_type().clone(),
            &mut work,
            |r, mut buffer| {
                buffer
                    .as_mut()
                    .copy_from_slice(&bytes[(r.start - 17) as usize..(r.end - 17) as usize]);
                std::future::ready(Ok(buffer))
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains(message), "{error}");
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn rle_windows_share_one_admission_and_cancel_together() {
    let (page, bytes, source, counts) = fixture(Children::Flat, false);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    // One selected count needs two prefixes plus two independent 4KiB windows.
    let budget = AllocationBudget::new(8208);
    let mut short = budget.try_reserve(8207).unwrap();
    let reads = Cell::new(0);
    assert!(
        scheduler
            .prepare(
                0..1,
                source.data_type().clone(),
                &mut short,
                |r, mut buffer| {
                    reads.set(reads.get() + 1);
                    assert_eq!(r, 17..25); // only framing may be read before child preflight
                    buffer.as_mut().copy_from_slice(&bytes[..8]);
                    std::future::ready(Ok(buffer))
                }
            )
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(reads.get(), 1);
    assert_eq!(short.len(), 8207);
    drop(short);
    let mut work = budget.try_reserve(8208).unwrap();
    let prepared = scheduler.lists[0]
        .counts
        .prepare(
            0..1,
            counts.iter().sum(),
            &mut work,
            &mut |r, mut buffer: ReservedBuffer| {
                buffer
                    .as_mut()
                    .copy_from_slice(&bytes[(r.start - 17) as usize..(r.end - 17) as usize]);
                std::future::ready(Ok(buffer))
            },
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), 8192);
    assert_eq!(prepared.prefix(1).unwrap(), counts[0]);
    let Counts::Selected { prefixes, .. } = prepared else {
        unreachable!()
    };
    let alias = prefixes.slice_with_length(0, 1);
    drop((prefixes, work));
    assert_eq!(budget.reserved(), 16);
    drop(alias);
    let mut work = budget.try_reserve(8208).unwrap();
    let reads = Cell::new(0);
    let mut pending = scheduler
        .prepare(
            0..1,
            source.data_type().clone(),
            &mut work,
            |r, mut buffer| {
                let call = reads.get();
                reads.set(call + 1);
                let bytes = &bytes;
                async move {
                    if call == 2 {
                        std::future::pending::<()>().await;
                    }
                    buffer
                        .as_mut()
                        .copy_from_slice(&bytes[(r.start - 17) as usize..(r.end - 17) as usize]);
                    Ok(buffer)
                }
            },
        )
        .boxed_local();
    assert!(matches!(poll!(&mut pending), Poll::Pending));
    assert_eq!(reads.get(), 3); // header recycled, both child windows now live
    drop(pending);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
