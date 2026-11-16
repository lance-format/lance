// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use futures::{FutureExt, poll};
use std::{cell::Cell, task::Poll};

fn payload_fixture(
    parents: &[u64],
    valid: &[bool],
    is_null: bool,
    metadata_parents: bool,
    packed: bool,
    delta: bool,
) -> (PageInfo, Vec<u8>, ArrayRef) {
    let (mut bytes, parent_codec) = if metadata_parents {
        (
            vec![],
            EncodingUtils::range(64, parents[0], parents[1] - parents[0]),
        )
    } else {
        payload::encode(parents, packed, delta)
    };
    let mut buffers = Vec::new();
    if !metadata_parents {
        buffers.push((17, bytes.len() as u64));
    }
    let positions = valid
        .iter()
        .enumerate()
        .filter_map(|(i, &value)| (value != is_null).then_some(i as u64))
        .collect::<Vec<_>>();
    let (validity_bytes, validity_codec) = payload::encode(&positions, packed, delta);
    buffers.push((17 + bytes.len() as u64, validity_bytes.len() as u64));
    bytes.extend_from_slice(&validity_bytes);
    let mut validity = super::positions(is_null, 0, 1, positions.len() as u64);
    validity.positions = Some(validity_codec);
    let mut list = list_layer(
        valid.len() as u64,
        parents.len() as u64 * 2,
        0,
        1,
        parents.len() as u64,
        EncodingUtils::constant(Some(2_u64.to_le_bytes().to_vec().into())),
        validity,
    );
    let Some(pb23::sparse_structural_layer::Layer::List(layer)) = &mut list.layer else {
        unreachable!()
    };
    layer.non_empty_positions = Some(parent_codec);
    let mut page = fixture(1, super::positions(true, 0, 1, 0));
    page.num_rows = valid.len() as u64;
    page.buffer_offsets_and_sizes = buffers.into();
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        &mut layout.structure.as_mut().unwrap().codec
    else {
        unreachable!()
    };
    structure.layers[0] = list;
    structure.layers[1].layer = Some(pb23::sparse_structural_layer::Layer::Validity(
        super::positions(true, 0, 1, parents.len() as u64 * 2),
    ));
    let mut offsets = vec![0_i32];
    for row in 0..valid.len() as u64 {
        offsets.push(offsets.last().unwrap() + if parents.contains(&row) { 2 } else { 0 });
    }
    let source = wrap(
        new_null_array(&DataType::Binary, parents.len() * 2),
        false,
        &offsets,
        valid.to_vec(),
    );
    (page, bytes, source)
}

#[rstest]
#[case::flat(false, false)]
#[case::packed(true, false)]
#[case::delta_flat(false, true)]
#[case::delta_packed(true, true)]
#[tokio::test]
async fn full_domain_merge_retains_only_selected_structure(
    #[case] packed: bool,
    #[case] delta: bool,
    #[values(false, true)] is_null: bool,
    #[values(false, true)] metadata_parents: bool,
) {
    if packed && !cfg!(feature = "bitpacking") {
        return;
    }
    let parents = (0..1025)
        .map(|i| {
            if metadata_parents {
                i * 4
            } else {
                i * 3 + i / 7
            }
        })
        .collect::<Vec<_>>();
    let valid = (0..4200)
        .map(|row| parents.contains(&row) || row % 7 != (row / 13) % 7)
        .collect::<Vec<_>>();
    let (page, bytes, source) =
        payload_fixture(&parents, &valid, is_null, metadata_parents, packed, delta);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(32768);
    for rows in [0..8, 2010..2020, 4195..4200] {
        let mut work = budget.try_reserve(32768).unwrap();
        let read_bytes = Cell::new(0);
        let input = scheduler
            .prepare(
                rows.clone(),
                source.data_type().clone(),
                &mut work,
                |range, mut buffer| {
                    assert!(buffer.capacity() <= 8192);
                    read_bytes.set(read_bytes.get() + buffer.capacity());
                    buffer.as_mut().copy_from_slice(
                        &bytes[(range.start - 17) as usize..(range.end - 17) as usize],
                    );
                    std::future::ready(Ok(buffer))
                },
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(read_bytes.get(), bytes.len());
        let rows_len = (rows.end - rows.start) as usize;
        assert_eq!(
            32768 - work.len(),
            rows_len.div_ceil(8) + if metadata_parents { 0 } else { rows_len * 8 }
        );
        let output = input.decode_in(&mut work).unwrap().unwrap();
        assert_eq!(
            output.to_data(),
            source.slice(rows.start as usize, rows_len).to_data()
        );
        let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
        drop((output, work));
        assert!(budget.reserved() > 1);
        assert!(budget.try_reserve(32768).is_none());
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[case::duplicate(0, "strictly increasing")]
#[case::decreasing(1, "strictly increasing")]
#[case::outside_domain(2, "exceeds List domain")]
#[case::null_parent(3, "null List parent")]
#[tokio::test]
async fn invalid_unselected_suffix_is_not_hidden(
    #[case] kind: usize,
    #[case] message: &str,
    #[values(false, true)] is_null: bool,
) {
    let parents = (0..1025).map(|i| i * 4).collect::<Vec<_>>();
    let valid = (0..4200).map(|i| i % 4 != 1).collect::<Vec<_>>();
    let (mut page, mut bytes, source) =
        payload_fixture(&parents, &valid, is_null, false, false, false);
    let (start, len) = page.buffer_offsets_and_sizes[1];
    let offset = (start - 17) as usize;
    let mut values = bytes[offset..offset + len as usize]
        .chunks_exact(8)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    match kind {
        0 => {
            let n = values.len();
            values[n - 1] = values[n - 2];
        }
        1 => {
            let n = values.len();
            values.swap(n - 1, n - 2);
        }
        2 => {
            *values.last_mut().unwrap() = 4200;
        }
        3 if is_null => {
            values.push(4096);
            values.sort_unstable();
        }
        3 => values.retain(|&v| v != 4096),
        _ => unreachable!(),
    }
    bytes.truncate(offset);
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    let mut buffers = page.buffer_offsets_and_sizes.to_vec();
    buffers[1].1 = values.len() as u64 * 8;
    page.buffer_offsets_and_sizes = buffers.into();
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        &mut layout.structure.as_mut().unwrap().codec
    else {
        unreachable!()
    };
    let Some(pb23::sparse_structural_layer::Layer::List(list)) = &mut structure.layers[0].layer
    else {
        unreachable!()
    };
    list.validity.as_mut().unwrap().num_positions = values.len() as u64;
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(16384);
    let mut work = budget.try_reserve(16384).unwrap();
    let error = scheduler
        .prepare(
            1..2,
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
    assert!(error.to_string().contains(message), "{error}");
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn simultaneous_windows_preflight_cancel_and_alias_lifetime() {
    let valid = (0..12).map(|i| ![0, 3, 7].contains(&i)).collect::<Vec<_>>();
    let (page, bytes, source) = payload_fixture(&[1, 4, 9], &valid, true, false, false, false);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    // 24 selected-position bytes + 1 bitmap byte + two 24-byte input windows.
    let budget = AllocationBudget::new(73);
    let mut short = budget.try_reserve(72).unwrap();
    assert!(
        scheduler
            .prepare(
                0..5,
                source.data_type().clone(),
                &mut short,
                unexpected_read
            )
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(short.len(), 72);
    drop(short);
    let mut work = budget.try_reserve(73).unwrap();
    let calls = Cell::new(0);
    let mut preparation = scheduler
        .prepare(
            0..5,
            source.data_type().clone(),
            &mut work,
            |range, mut buffer| {
                let call = calls.get();
                calls.set(call + 1);
                let bytes = &bytes;
                async move {
                    if call == 1 {
                        std::future::pending::<()>().await;
                    }
                    buffer.as_mut().copy_from_slice(
                        &bytes[(range.start - 17) as usize..(range.end - 17) as usize],
                    );
                    Ok(buffer)
                }
            },
        )
        .boxed_local();
    assert!(matches!(poll!(&mut preparation), Poll::Pending));
    assert_eq!(calls.get(), 2); // first cursor is live while the second read is pending
    assert_eq!(budget.reserved(), 73);
    drop(preparation);
    drop(work);
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(73).unwrap();
    let input = scheduler
        .prepare(
            0..5,
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
    assert_eq!(work.len(), 48); // both windows recycled; selected structure stays charged
    let Validity::Selected { bitmap, .. } = &input.lists[0].validity else {
        unreachable!()
    };
    let input_alias = bitmap.clone();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.to_data(), source.slice(0, 5).to_data());
    let output_alias = output.nulls().unwrap().clone();
    drop((output, work));
    assert_eq!(budget.reserved(), 26); // input bitmap + parent offsets and bitmap
    drop(output_alias);
    assert_eq!(budget.reserved(), 1);
    drop(input_alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::one_short(108, false)]
#[case::exact(109, true)]
#[tokio::test]
async fn retained_validity_overlaps_decoded_output(
    #[case] bytes_available: usize,
    #[case] fits: bool,
) {
    let valid = (0..12).map(|i| ![0, 3, 7].contains(&i)).collect::<Vec<_>>();
    let (page, bytes, source) = payload_fixture(&[1, 4, 9], &valid, true, false, false, false);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(bytes_available);
    let mut work = budget.try_reserve(bytes_available).unwrap();
    let input = scheduler
        .prepare(
            0..12,
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
    assert_eq!(bytes_available - work.len(), 26);
    let output = input.decode_in(&mut work).unwrap();
    assert_eq!(output.is_some(), fits);
    if let Some(output) = output {
        assert_eq!(output.to_data(), source.to_data());
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::nonempty(0)]
#[case::empty(1)]
#[case::later(2)]
#[tokio::test]
async fn nested_validity_uses_the_child_domain(#[case] row: u64) {
    let valid = (0..12).map(|i| ![0, 3, 7].contains(&i)).collect::<Vec<_>>();
    let (mut page, bytes, child) = payload_fixture(&[1, 4, 9], &valid, false, false, false, false);
    page.num_rows = 5;
    let PageEncoding::Planar(layout) = &mut page.encoding else {
        unreachable!()
    };
    let Some(pb23::structural_plane::Codec::Sparse(structure)) =
        &mut layout.structure.as_mut().unwrap().codec
    else {
        unreachable!()
    };
    structure.layers.insert(
        0,
        list_layer(
            5,
            12,
            0,
            2,
            3,
            EncodingUtils::constant(Some(4_u64.to_le_bytes().to_vec().into())),
            positions(true, 0, 1, 0),
        ),
    );
    let source = wrap(child, true, &[0, 4, 4, 8, 8, 12], vec![true; 5]);
    let scheduler = NullListScheduler::try_from_page(&page).unwrap().unwrap();
    let budget = AllocationBudget::new(512);
    let mut work = budget.try_reserve(512).unwrap();
    let input = scheduler
        .prepare(
            row..row + 1,
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
    assert_eq!(output.to_data(), source.slice(row as usize, 1).to_data());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}
