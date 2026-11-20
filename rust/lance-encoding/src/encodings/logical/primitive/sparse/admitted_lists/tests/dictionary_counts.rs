// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::OutOfLineBitpacking;
use crate::{
    buffer::LanceBuffer,
    compression::BlockCompressor,
    data::{BlockInfo, FixedWidthDataBlock},
    encodings::physical::{
        constant::ConstantEncoder, dictionary::BlockDictionaryEncoder, value::ValueEncoder,
    },
};
use futures::{FutureExt, poll};
use std::{cell::Cell, task::Poll};

#[derive(Clone, Copy)]
enum Children {
    Flat,
    RangeItems,
    Metadata,
    PackedIndices,
    PackedItems,
    UnusedZero,
}

fn fixture(children: Children, large: bool) -> (PageInfo, Vec<u8>, ArrayRef) {
    let mut indices: Box<dyn BlockCompressor> = Box::<ValueEncoder>::default();
    let mut item_codec: Box<dyn BlockCompressor> = Box::<ValueEncoder>::default();
    let mut items = vec![1_u64, 3, 7, 13];
    match children {
        Children::RangeItems | Children::Metadata => {
            items = vec![1, 4, 7, 10];
            item_codec = Box::new(RangeEncoder::new(64, 1, 3));
            if matches!(children, Children::Metadata) {
                indices = Box::new(ConstantEncoder::new(32, 1));
            }
        }
        #[cfg(feature = "bitpacking")]
        Children::PackedIndices => indices = Box::new(OutOfLineBitpacking::new(2, 32)),
        #[cfg(feature = "bitpacking")]
        Children::PackedItems => {
            items = (0..1025).map(|i| 1 + i * 17 % 2047).collect();
            item_codec = Box::new(OutOfLineBitpacking::new(11, 64));
        }
        Children::UnusedZero => items[0] = 0,
        _ => {}
    }
    let counts = (0..2024)
        .map(|i| {
            let index = match children {
                Children::Metadata => 1,
                Children::UnusedZero => 1 + i % 3,
                _ => (i * 17 + i / 7) % items.len(),
            };
            items[index]
        })
        .collect::<Vec<_>>();
    let encoder = BlockDictionaryEncoder::try_new(64, items.into(), indices, item_codec).unwrap();
    let (bytes, encoding) = encoder
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
    (page, bytes, source)
}

#[rstest]
#[case::flat(Children::Flat)]
#[case::range_items(Children::RangeItems)]
#[case::metadata(Children::Metadata)]
#[case::packed_indices(Children::PackedIndices)]
#[case::packed_items(Children::PackedItems)]
#[case::unused_zero(Children::UnusedZero)]
#[tokio::test]
async fn dictionary_windows_preserve_selected_lists(
    #[case] children: Children,
    #[values(false, true)] large: bool,
) {
    if matches!(children, Children::PackedIndices | Children::PackedItems)
        && !cfg!(feature = "bitpacking")
    {
        return;
    }
    let (page, bytes, source) = fixture(children, large);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(65536);
    for rows in [
        0..5,
        page.num_rows - 5..page.num_rows,
        1..2,
        3..4,
        page.num_rows - 1..page.num_rows,
    ] {
        let mut work = budget.try_reserve(65536).unwrap();
        let reads = Cell::new(0);
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
        assert_eq!(reads.get(), bytes.len());
        assert!(65536 - work.len() <= 32); // table and every input window were recycled
        let output = input.decode_in(&mut work).unwrap().unwrap();
        assert_eq!(
            output.to_data(),
            source
                .slice(rows.start as usize, (rows.end - rows.start) as usize)
                .to_data()
        );
        let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
        drop((output, work));
        assert!(budget.try_reserve(65536).is_none());
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[case::frame_overflow(0, "frame")]
#[case::trailing_frame(1, "frame lengths")]
#[case::late_index(2, "index exceeds")]
#[case::zero_count(3, "positive")]
#[case::wrong_child_total(4, "child domain")]
#[case::metadata_payload(5, "metadata-only")]
#[tokio::test]
async fn dictionary_corruption_cannot_hide_outside_selection(
    #[case] kind: u8,
    #[case] message: &str,
) {
    let (mut page, mut bytes, source) = fixture(
        if kind == 5 {
            Children::RangeItems
        } else {
            Children::Flat
        },
        false,
    );
    let split = 16 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    match kind {
        0 => bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes()),
        1 => bytes[8..16].copy_from_slice(&33_u64.to_le_bytes()),
        2 => bytes[split - 4..split].copy_from_slice(&4_u32.to_le_bytes()),
        3 => bytes[split + 24..split + 32].copy_from_slice(&0_u64.to_le_bytes()),
        4 => bytes[split + 24..split + 32].copy_from_slice(&99_u64.to_le_bytes()),
        5 => {
            bytes[8..16].copy_from_slice(&1_u64.to_le_bytes());
            bytes.push(0);
            page.buffer_offsets_and_sizes = Arc::from([(17, bytes.len() as u64)]);
        }
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

#[rstest]
#[case::item_read(1)]
#[case::index_read(2)]
#[tokio::test]
async fn dictionary_table_and_window_share_credit_and_cancel(#[case] cancel_read: usize) {
    let (page, bytes, source) = fixture(Children::Flat, false);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    // 4 dictionary items + 2 selected prefixes + one 4KiB index window.
    let budget = AllocationBudget::new(4144);
    let mut work = budget.try_reserve(4143).unwrap();
    let reads = Cell::new(0);
    assert!(
        scheduler
            .prepare(
                0..1,
                source.data_type().clone(),
                &mut work,
                |r, mut buffer| {
                    reads.set(reads.get() + 1);
                    assert_eq!(r, 17..33);
                    buffer.as_mut().copy_from_slice(&bytes[..16]);
                    std::future::ready(Ok(buffer))
                }
            )
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(reads.get(), 1);
    assert_eq!(work.len(), 4143);
    drop(work);
    let mut work = budget.try_reserve(4144).unwrap();
    let input = scheduler
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
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), 4128);
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.to_data(), source.slice(0, 1).to_data());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(4144).unwrap();
    reads.set(0);
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
                    if call == cancel_read {
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
    assert_eq!(reads.get(), cancel_read + 1);
    drop(pending);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn maximum_dictionary_table_is_charged_and_recycled() {
    let counts = vec![1_u64, 4096, 2, 4096];
    let encoder = BlockDictionaryEncoder::try_new(
        64,
        (1..=4096).collect::<Vec<u64>>().into(),
        Box::<ValueEncoder>::default(),
        Box::<ValueEncoder>::default(),
    )
    .unwrap();
    let (bytes, encoding) = encoder
        .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            num_values: 4,
            data: LanceBuffer::reinterpret_vec(counts.clone()),
            block_info: BlockInfo::default(),
        }))
        .unwrap();
    let bytes = bytes.unwrap().to_vec();
    let (page, source) = payload::payload_page(&counts, encoding, bytes.len(), false);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(36880); // 4096*8 lookup + 16 prefixes + 4096 scratch
    for limit in [36879, 36880] {
        let mut work = budget.try_reserve(limit).unwrap();
        let reads = Cell::new(0);
        let input = scheduler
            .prepare(
                2..3,
                source.data_type().clone(),
                &mut work,
                |r, mut buffer| {
                    reads.set(reads.get() + buffer.capacity());
                    assert!(buffer.capacity() <= 4096);
                    buffer
                        .as_mut()
                        .copy_from_slice(&bytes[(r.start - 17) as usize..(r.end - 17) as usize]);
                    std::future::ready(Ok(buffer))
                },
            )
            .await
            .unwrap();
        if limit == 36879 {
            assert!(input.is_none());
            assert_eq!(reads.get(), 16);
            assert_eq!(work.len(), limit);
        } else {
            assert_eq!(reads.get(), bytes.len());
            assert_eq!(work.len(), limit - 16);
            let output = input.unwrap().decode_in(&mut work).unwrap().unwrap();
            assert_eq!(output.to_data(), source.slice(2, 1).to_data());
            drop(output);
        }
        drop(work);
        assert_eq!(budget.reserved(), 0);
    }
    let error = Counts::parse(
        Some(&EncodingUtils::dictionary(
            EncodingUtils::flat(32, None),
            EncodingUtils::flat(64, None),
            4097,
        )),
        1,
        &mut std::iter::empty(),
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("outside 1..=4096"));
}
