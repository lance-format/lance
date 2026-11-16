// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::OutOfLineBitpacking;
use crate::{
    buffer::LanceBuffer,
    compression::BlockCompressor,
    data::{BlockInfo, FixedWidthDataBlock},
    encodings::physical::{delta::DeltaEncoder, value::ValueEncoder},
};
use futures::{FutureExt, poll};
use std::{cell::RefCell, task::Poll};

pub(super) fn encode(values: &[u64], packed: bool, delta: bool) -> (Vec<u8>, CompressiveEncoding) {
    let child: Box<dyn BlockCompressor> = if packed {
        #[cfg(feature = "bitpacking")]
        {
            let max = if delta {
                values
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .max()
                    .unwrap_or(0)
            } else {
                values.iter().copied().max().unwrap_or(0)
            };
            Box::new(OutOfLineBitpacking::new(
                u64::from((64 - max.leading_zeros()).max(1)),
                64,
            ))
        }
        #[cfg(not(feature = "bitpacking"))]
        {
            panic!("test requires bitpacking")
        }
    } else {
        Box::new(ValueEncoder::default())
    };
    let codec: Box<dyn BlockCompressor> = if delta {
        Box::new(DeltaEncoder::new(64, values[0], child))
    } else {
        child
    };
    let (bytes, encoding) = codec
        .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            data: LanceBuffer::reinterpret_vec(values.to_vec()),
            num_values: values.len() as u64,
            block_info: BlockInfo::default(),
        }))
        .unwrap();
    (bytes.unwrap().to_vec(), encoding)
}

fn payload_page(
    values: &[u64],
    encoding: CompressiveEncoding,
    bytes: usize,
    large: bool,
) -> (PageInfo, ArrayRef) {
    let rows = values.len() * 2 + 1;
    let total: u64 = values.iter().sum();
    let mut page = fixture(1, positions(true, 0, 1, 0));
    page.num_rows = rows as u64;
    page.buffer_offsets_and_sizes = Arc::from([(17, bytes as u64)]);
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    layout.structure.as_mut().unwrap().codec = Some(pb23::structural_plane::Codec::Sparse(
        pb23::SparseStructure {
            layers: vec![
                list_layer(
                    rows as u64,
                    total,
                    0,
                    2,
                    values.len() as u64,
                    encoding,
                    positions(true, 1, 4, ((rows + 2) / 4) as u64),
                ),
                pb23::SparseStructuralLayer {
                    layer: Some(pb23::sparse_structural_layer::Layer::Validity(positions(
                        true, 0, 1, total,
                    ))),
                },
            ],
        },
    ));
    let mut offsets = vec![0_i64];
    for row in 0..rows {
        let count = if row % 2 == 0 {
            values.get(row / 2).copied().unwrap_or(0)
        } else {
            0
        };
        offsets.push(offsets.last().unwrap() + count as i64);
    }
    let field = Arc::new(Field::new("item", DataType::Binary, true));
    let child = new_null_array(&DataType::Binary, total as usize);
    let validity = Some(NullBuffer::from(
        (0..rows).map(|r| r % 4 != 1).collect::<Vec<_>>(),
    ));
    let source: ArrayRef = if large {
        Arc::new(LargeListArray::new(
            field,
            OffsetBuffer::new(offsets.into()),
            child,
            validity,
        ))
    } else {
        Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(
                offsets
                    .into_iter()
                    .map(|o| o as i32)
                    .collect::<Vec<_>>()
                    .into(),
            ),
            child,
            validity,
        ))
    };
    (page, source)
}

#[rstest]
#[tokio::test]
async fn count_payload_windows_preserve_selection_and_release(
    #[values(false, true)] delta: bool,
    #[values(false, true)] large: bool,
    #[values(1025, 8193)] count: usize,
) {
    for packed in [false, true] {
        if packed && !cfg!(feature = "bitpacking") {
            continue;
        }
        let values = (0..count)
            .map(|i| {
                if delta {
                    1 + i as u64 / 1024
                } else {
                    1 + (i as u64 * 17 % 31)
                }
            })
            .collect::<Vec<_>>();
        let (bytes, encoding) = encode(&values, packed, delta);
        let (page, source) = payload_page(&values, encoding, bytes.len(), large);
        let scheduler = AdmittedBytePageScheduler::try_from_page(
            &page,
            &DefaultDecompressionStrategy::default(),
        )
        .unwrap();
        let budget = AllocationBudget::new(9000);
        for range in [1..5, 2043..2049, page.num_rows - 2..page.num_rows] {
            let reads = RefCell::new(Vec::new());
            let mut work = budget.try_reserve(9000).unwrap();
            let input = scheduler
                .prepare_array_range_in(
                    range.clone(),
                    source.data_type().clone(),
                    &mut work,
                    |range, mut buffer| {
                        assert_eq!(buffer.capacity(), (range.end - range.start) as usize);
                        assert!(buffer.capacity() <= 8192);
                        reads.borrow_mut().push(range.clone());
                        buffer.as_mut().copy_from_slice(
                            &bytes[(range.start - 17) as usize..(range.end - 17) as usize],
                        );
                        std::future::ready(Ok(buffer))
                    },
                )
                .await
                .unwrap()
                .unwrap();
            if count > 1025 {
                assert!(reads.borrow().len() >= 2);
            }
            assert_eq!(
                reads.borrow().iter().map(|r| r.end - r.start).sum::<u64>(),
                bytes.len() as u64
            );
            assert!(9000 - work.len() <= (range.end - range.start + 1) as usize * 8);
            let output = input.decode_in(&mut work).unwrap().unwrap();
            let expected = arrow_select::take::take(
                source.as_ref(),
                &UInt32Array::from((range.start as u32..range.end as u32).collect::<Vec<_>>()),
                None,
            )
            .unwrap();
            assert_eq!(output.to_data(), expected.to_data());
            let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
            drop((output, work));
            assert!(budget.reserved() > 1);
            assert!(budget.try_reserve(9000).is_none());
            drop(alias);
            assert_eq!(budget.reserved(), 0);
        }
    }
}

#[rstest]
#[case::zero_count(false)]
#[case::bad_total(true)]
#[tokio::test]
async fn validates_unselected_suffix(#[case] total_mismatch: bool) {
    let mut values = vec![1_u64; 1025];
    let (bytes, encoding) = encode(&values, false, false);
    let (page, source) = payload_page(&values, encoding, bytes.len(), false);
    values[1024] = if total_mismatch { 2 } else { 0 };
    let (bytes, _) = encode(&values, false, false);
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let error = scheduler
        .prepare_array_range_in(
            0..1,
            source.data_type().clone(),
            &mut work,
            |range, mut buffer| {
                buffer.as_mut().copy_from_slice(
                    &bytes[(range.start - 17) as usize..(range.end - 17) as usize],
                );
                std::future::ready(Ok(buffer))
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains(if total_mismatch {
        "child domain"
    } else {
        "positive"
    }));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn scratch_shortage_and_cancel_release_complete_attempt() {
    let values = vec![1_u64; 1025];
    let (bytes, encoding) = encode(&values, false, false);
    let (page, source) = payload_page(&values, encoding, bytes.len(), false);
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(4112); // selected two prefixes plus one 4KiB window
    let mut work = budget.try_reserve(4111).unwrap();
    assert!(
        scheduler
            .prepare_array_range_in(0..1, source.data_type().clone(), &mut work, unexpected_read)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(work.len(), 4111);
    drop(work);
    let mut work = budget.try_reserve(4112).unwrap();
    let mut future = scheduler
        .prepare_array_range_in(
            0..1,
            source.data_type().clone(),
            &mut work,
            |_, buffer| async move {
                std::future::pending::<()>().await;
                Ok(buffer)
            },
        )
        .boxed();
    assert!(matches!(poll!(&mut future), Poll::Pending));
    assert_eq!(budget.reserved(), 4112);
    drop(future);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn nested_payload_counts_keep_each_selected_child_domain() {
    let (outer_bytes, outer_codec) = encode(&[2, 1, 4], false, false);
    let inner_values = [1, 3, 2, 1, 4, 2, 3];
    let (inner_bytes, inner_codec) = encode(&inner_values, false, false);
    let mut bytes = outer_bytes.clone();
    bytes.extend_from_slice(&inner_bytes);
    let mut page = fixture(2, positions(true, 0, 1, 0));
    page.num_rows = 3;
    page.buffer_offsets_and_sizes = Arc::from([
        (17, outer_bytes.len() as u64),
        (17 + outer_bytes.len() as u64, inner_bytes.len() as u64),
    ]);
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    layout.structure.as_mut().unwrap().codec = Some(pb23::structural_plane::Codec::Sparse(
        pb23::SparseStructure {
            layers: vec![
                list_layer(3, 7, 0, 1, 3, outer_codec, positions(true, 0, 1, 0)),
                list_layer(7, 16, 0, 1, 7, inner_codec, positions(true, 0, 1, 0)),
                pb23::SparseStructuralLayer {
                    layer: Some(pb23::sparse_structural_layer::Layer::Validity(positions(
                        true, 0, 1, 16,
                    ))),
                },
            ],
        },
    ));
    let inner: ArrayRef = Arc::new(LargeListArray::new(
        Arc::new(Field::new("leaf", DataType::Utf8, true)),
        OffsetBuffer::new(vec![0_i64, 1, 4, 6, 7, 11, 13, 16].into()),
        new_null_array(&DataType::Utf8, 16),
        None,
    ));
    let source: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("inner", inner.data_type().clone(), false)),
        OffsetBuffer::new(vec![0_i32, 2, 3, 7].into()),
        inner,
        None,
    ));
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(1024);
    for range in [0..1, 1..3, 2..3] {
        let mut work = budget.try_reserve(1024).unwrap();
        let input = scheduler
            .prepare_array_range_in(
                range.clone(),
                source.data_type().clone(),
                &mut work,
                |range, mut buffer| {
                    buffer.as_mut().copy_from_slice(
                        &bytes[(range.start - 17) as usize..(range.end - 17) as usize],
                    );
                    std::future::ready(Ok(buffer))
                },
            )
            .await
            .unwrap()
            .unwrap();
        let output = input.decode_in(&mut work).unwrap().unwrap();
        assert_eq!(
            output.to_data(),
            source
                .slice(range.start as usize, (range.end - range.start) as usize)
                .to_data()
        );
        drop((output, work));
        assert_eq!(budget.reserved(), 0);
    }
}

#[tokio::test]
async fn metadata_delta_counts_need_no_payload_buffer() {
    let values = [1_u64, 2, 4];
    let encoding = EncodingUtils::delta(64, 1, EncodingUtils::range(64, 1, 1));
    let (mut page, source) = payload_page(&values, encoding, 0, false);
    page.buffer_offsets_and_sizes = Arc::from([]);
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(256);
    let mut work = budget.try_reserve(256).unwrap();
    let input = scheduler
        .prepare_array_range_in(2..6, source.data_type().clone(), &mut work, unexpected_read)
        .await
        .unwrap()
        .unwrap();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.to_data(), source.slice(2, 4).to_data());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}
