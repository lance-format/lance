// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use futures::{FutureExt, poll};
use std::{cell::RefCell, task::Poll};

fn position_fixture(
    values: &[u64],
    domain: u64,
    packed: bool,
    delta: bool,
    large: bool,
    payload_counts: bool,
) -> (PageInfo, Vec<u8>, ArrayRef) {
    let (mut bytes, encoding) = payload::encode(values, packed, delta);
    let mut buffers = vec![(17, bytes.len() as u64)];
    let counts = (0..values.len())
        .map(|i| {
            if payload_counts {
                1 + (i as u64 * 13 % 19)
            } else {
                2
            }
        })
        .collect::<Vec<_>>();
    let count_codec = if payload_counts {
        let (data, codec) = payload::encode(&counts, false, false);
        buffers.push((17 + bytes.len() as u64, data.len() as u64));
        bytes.extend_from_slice(&data);
        codec
    } else {
        EncodingUtils::constant(Some(2_u64.to_le_bytes().to_vec().into()))
    };
    let total = counts.iter().sum();
    let mut list = list_layer(
        domain,
        total,
        0,
        1,
        values.len() as u64,
        count_codec,
        positions(true, 0, 1, 0),
    );
    let Some(pb23::sparse_structural_layer::Layer::List(layer)) = &mut list.layer else {
        unreachable!()
    };
    layer.non_empty_positions = Some(encoding);
    let mut page = fixture(1, positions(true, 0, 1, 0));
    page.num_rows = domain;
    page.buffer_offsets_and_sizes = buffers.into();
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    layout.structure.as_mut().unwrap().codec = Some(pb23::structural_plane::Codec::Sparse(
        pb23::SparseStructure {
            layers: vec![
                list,
                pb23::SparseStructuralLayer {
                    layer: Some(pb23::sparse_structural_layer::Layer::Validity(positions(
                        true, 0, 1, total,
                    ))),
                },
            ],
        },
    ));
    let mut offsets = vec![0_i64];
    let mut current = 0;
    for row in 0..domain {
        let count = if values.get(current) == Some(&row) {
            current += 1;
            counts[current - 1]
        } else {
            0
        };
        offsets.push(offsets.last().unwrap() + count as i64);
    }
    let field = Arc::new(Field::new("item", DataType::Binary, true));
    let child = new_null_array(&DataType::Binary, total as usize);
    let source: ArrayRef = if large {
        Arc::new(LargeListArray::new(
            field,
            OffsetBuffer::new(offsets.into()),
            child,
            None,
        ))
    } else {
        Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(
                offsets
                    .into_iter()
                    .map(|n| n as i32)
                    .collect::<Vec<_>>()
                    .into(),
            ),
            child,
            None,
        ))
    };
    (page, bytes, source)
}

#[rstest]
#[tokio::test]
async fn selected_positions_keep_global_ranks_and_full_capacity(
    #[values(false, true)] packed: bool,
    #[values(false, true)] delta: bool,
    #[values(false, true)] large: bool,
    #[values(false, true)] payload_counts: bool,
) {
    if packed && !cfg!(feature = "bitpacking") {
        return;
    }
    let values = (0..1025).map(|i| 2 + i * 3 + i / 7).collect::<Vec<_>>();
    let domain = values.last().unwrap() + 3;
    let (page, bytes, source) =
        position_fixture(&values, domain, packed, delta, large, payload_counts);
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(9000);
    for range in [0..2, 1..9, 1600..1607, domain - 3..domain] {
        let mut work = budget.try_reserve(9000).unwrap();
        let reads = RefCell::new(Vec::new());
        let input = scheduler
            .prepare_array_range_in(
                range.clone(),
                source.data_type().clone(),
                &mut work,
                |r, mut buffer| {
                    assert!(buffer.capacity() <= 8192);
                    reads.borrow_mut().push(r.clone());
                    buffer
                        .as_mut()
                        .copy_from_slice(&bytes[(r.start - 17) as usize..(r.end - 17) as usize]);
                    std::future::ready(Ok(buffer))
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            reads.borrow().iter().map(|r| r.end - r.start).sum::<u64>(),
            bytes.len() as u64
        );
        let selected = values.iter().filter(|&&v| range.contains(&v)).count();
        let retained = (range.end - range.start) as usize * 8
            + if payload_counts {
                (selected + 1) * 8
            } else {
                0
            };
        assert_eq!(9000 - work.len(), retained);
        let output = input.decode_in(&mut work).unwrap().unwrap();
        assert_eq!(
            output.to_data(),
            source
                .slice(range.start as usize, (range.end - range.start) as usize)
                .to_data()
        );
        let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
        drop((output, work));
        assert!(budget.reserved() > 1);
        assert!(budget.try_reserve(9000).is_none());
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[case::duplicate(0)]
#[case::decreasing(1)]
#[case::domain(2)]
#[case::null_parent(3)]
#[tokio::test]
async fn validates_positions_outside_selection(#[case] kind: u8) {
    let mut values = (0..1025).map(|i| 2 + i * 3).collect::<Vec<_>>();
    let (mut page, mut bytes, source) = position_fixture(&values, 4000, false, false, false, false);
    match kind {
        0 => values[1024] = values[1023],
        1 => values[1024] = values[1023] - 1,
        2 => values[1024] = 4000,
        _ => {
            let PageEncoding::Planar(layout) = &mut page.encoding else {
                unreachable!()
            };
            let pb23::structural_plane::Codec::Sparse(structure) =
                layout.structure.as_mut().unwrap().codec.as_mut().unwrap()
            else {
                unreachable!()
            };
            let Some(pb23::sparse_structural_layer::Layer::List(list)) =
                &mut structure.layers[0].layer
            else {
                unreachable!()
            };
            list.validity = Some(positions(true, values[1024], 1, 1));
        }
    }
    if kind < 3 {
        bytes = payload::encode(&values, false, false).0;
    }
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(9000);
    let mut work = budget.try_reserve(9000).unwrap();
    let error = scheduler
        .prepare_array_range_in(
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
    assert!(error.to_string().contains(match kind {
        0 | 1 => "strictly increasing",
        2 => "domain",
        _ => "null List parent",
    }));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn positions_shortage_and_cancel_preserve_admission() {
    let values = (0..1025).map(|i| 2 + i * 3).collect::<Vec<_>>();
    let (page, _, source) = position_fixture(&values, 4000, false, false, false, false);
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(4104); // one selected parent + 4KiB window
    let mut work = budget.try_reserve(4103).unwrap();
    assert!(
        scheduler
            .prepare_array_range_in(0..1, source.data_type().clone(), &mut work, unexpected_read)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(work.len(), 4103);
    drop(work);
    let mut work = budget.try_reserve(4104).unwrap();
    let mut pending = scheduler
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
    assert!(matches!(poll!(&mut pending), Poll::Pending));
    assert_eq!(budget.reserved(), 4104);
    drop(pending);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn selected_position_alias_keeps_unused_backing_capacity() {
    let (page, bytes, _) = position_fixture(&[2, 5, 9], 12, false, false, false, false);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(128);
    let mut work = budget.try_reserve(128).unwrap();
    let selected = scheduler.lists[0]
        .non_empty
        .prepare(
            0..4,
            &scheduler.lists[0].validity,
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
    assert_eq!(selected.rank(4).unwrap(), 1);
    let error = selected.rank(5).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("outside prepared selection"));
    let Positions::Selected { ref values, .. } = selected else {
        unreachable!()
    };
    assert_eq!(values.len(), 8);
    let alias = values.slice_with_length(0, 1);
    drop((selected, work));
    assert_eq!(budget.reserved(), 24); // min(4 selected rows, 3 total positions) * 8
    assert!(budget.try_reserve(128).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn nested_position_payloads_preserve_each_rank_domain(
    #[values(false, true)] valid_positions: bool,
) {
    let (mut page, mut bytes, _) = position_fixture(&[1, 4, 6], 8, false, false, false, false);
    let (inner_bytes, inner_encoding) = payload::encode(&[0, 3, 5], false, true);
    let mut buffers = page.buffer_offsets_and_sizes.to_vec();
    buffers.push((17 + bytes.len() as u64, inner_bytes.len() as u64));
    bytes.extend_from_slice(&inner_bytes);
    page.buffer_offsets_and_sizes = buffers.into();
    let mut inner_layer = list_layer(
        6,
        6,
        0,
        1,
        3,
        EncodingUtils::constant(Some(2_u64.to_le_bytes().to_vec().into())),
        positions(true, 0, 1, 0),
    );
    let Some(pb23::sparse_structural_layer::Layer::List(inner)) = &mut inner_layer.layer else {
        unreachable!()
    };
    inner.non_empty_positions = Some(inner_encoding);
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let pb23::structural_plane::Codec::Sparse(structure) =
        layout.structure.as_mut().unwrap().codec.as_mut().unwrap()
    else {
        unreachable!()
    };
    structure.layers.insert(1, inner_layer);
    if valid_positions {
        for (layer, domain) in structure.layers.iter_mut().take(2).zip([8, 6]) {
            let Some(pb23::sparse_structural_layer::Layer::List(list)) = &mut layer.layer else {
                unreachable!()
            };
            list.validity = Some(positions(false, 0, 1, domain));
        }
    }
    let inner: ArrayRef = Arc::new(LargeListArray::new(
        Arc::new(Field::new("leaf", DataType::Binary, true)),
        OffsetBuffer::new(vec![0_i64, 2, 2, 2, 4, 4, 6].into()),
        new_null_array(&DataType::Binary, 6),
        None,
    ));
    let source: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("item", inner.data_type().clone(), true)),
        OffsetBuffer::new(vec![0_i32, 0, 2, 2, 2, 4, 4, 6, 6].into()),
        inner,
        None,
    ));
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(512);
    for range in [0..1, 1..5, 4..7, 7..8] {
        let mut work = budget.try_reserve(512).unwrap();
        let input = scheduler
            .prepare_array_range_in(
                range.clone(),
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
async fn position_storage_overlaps_output_in_original_workspace() {
    let (page, bytes, source) = position_fixture(&[2, 5, 9], 12, false, false, false, false);
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    // Positions retain 24 bytes. Four parents need 20 offset bytes, two null
    // Binary children need 12 offset bytes and one validity byte.
    let budget = AllocationBudget::new(57);
    for size in [56, 57] {
        let mut work = budget.try_reserve(size).unwrap();
        let input = scheduler
            .prepare_array_range_in(
                0..4,
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
        assert_eq!(size - work.len(), 24);
        let output = input.decode_in(&mut work).unwrap();
        if size == 56 {
            assert!(output.is_none());
        } else {
            assert_eq!(
                output.as_ref().unwrap().to_data(),
                source.slice(0, 4).to_data()
            );
        }
        drop((output, work));
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[tokio::test]
async fn metadata_delta_positions_do_not_consume_following_count_buffer(
    #[values(false, true)] payload_counts: bool,
) {
    let (mut page, mut bytes, source) =
        position_fixture(&[0, 1, 3], 4, false, false, false, payload_counts);
    let codec = crate::encodings::physical::sequence::select_u64_sequence_codec(
        [0, 1, 3].into_iter(),
        &Default::default(),
        Default::default(),
    )
    .unwrap();
    assert!(!codec.has_payload());
    let encoding = codec.expected_encoding().clone();
    let sequence =
        crate::encodings::physical::sequence::SequenceDecoder::try_new(&encoding, 64, 3).unwrap();
    assert!(sequence.metadata().is_none());
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let pb23::structural_plane::Codec::Sparse(structure) =
        layout.structure.as_mut().unwrap().codec.as_mut().unwrap()
    else {
        unreachable!()
    };
    let Some(pb23::sparse_structural_layer::Layer::List(list)) = &mut structure.layers[0].layer
    else {
        unreachable!()
    };
    list.non_empty_positions = Some(encoding);
    drop(bytes.drain(..24));
    page.buffer_offsets_and_sizes = if payload_counts {
        Arc::from([(17, bytes.len() as u64)])
    } else {
        Arc::from([])
    };
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy::default())
            .unwrap();
    let budget = AllocationBudget::new(512);
    let mut work = budget.try_reserve(512).unwrap();
    let reads = RefCell::new(0);
    let input = scheduler
        .prepare_array_range_in(
            1..4,
            source.data_type().clone(),
            &mut work,
            |r, mut buffer| {
                assert!(payload_counts, "metadata positions have no input buffer");
                *reads.borrow_mut() += 1;
                buffer
                    .as_mut()
                    .copy_from_slice(&bytes[(r.start - 17) as usize..(r.end - 17) as usize]);
                std::future::ready(Ok(buffer))
            },
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(*reads.borrow(), usize::from(payload_counts));
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.to_data(), source.slice(1, 3).to_data());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}
