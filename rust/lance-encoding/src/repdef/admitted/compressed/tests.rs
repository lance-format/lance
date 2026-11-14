// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_buffer::Buffer;
use lance_core::{
    Error,
    utils::allocation::{AllocationBudget, Reservation},
};
use rstest::rstest;

use crate::{
    buffer::LanceBuffer,
    compression::BlockDecompressor,
    encodings::physical::value::ValueDecompressor,
    format::metadata,
    repdef::{CompositeRepDefUnraveler, DefinitionInterpretation as Meaning, RepDefUnraveler},
};

fn own(bytes: &[u8], work: &mut Reservation) -> LanceBuffer {
    let mut owner = work.try_allocate(bytes.len() + 1).unwrap().unwrap();
    owner.as_mut()[1..].copy_from_slice(bytes);
    LanceBuffer::from(Buffer::from(owner.freeze()).slice(1))
}

#[rstest]
fn invalid_decoded_levels_recycle_both_outputs(#[values(false, true)] bad_rep: bool) {
    let codec = ValueDecompressor::from_flat(&metadata::Flat {
        bits_per_value: 16,
        data: None,
    });
    let budget = AllocationBudget::new(18);
    let mut work = budget.try_reserve(18).unwrap();
    let rep_values = if bad_rep { [1_u16, 2] } else { [1, 0] };
    let def_values = if bad_rep { [0_u16, 0] } else { [0, 3] };
    let rep = own(bytemuck::cast_slice(&rep_values), &mut work);
    let def = own(bytemuck::cast_slice(&def_values), &mut work);
    assert_eq!(work.len(), 8);
    let error = RepDefUnraveler::try_from_encoded_in(
        Some((&codec, &rep)),
        Some((&codec, &def)),
        2,
        Arc::from([Meaning::NullableItem, Meaning::NullableList]),
        2,
        &mut work,
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("level exceeds"));
    assert_eq!(work.len(), 8);
    assert_eq!(budget.reserved(), 18);
    drop(rep);
    drop(def);
    assert_eq!(budget.reserved(), 8);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn invalid_second_frame_is_rejected_before_first_decode() {
    let codec = ValueDecompressor::from_flat(&metadata::Flat {
        bits_per_value: 16,
        data: None,
    });
    let budget = AllocationBudget::new(64);
    let mut work = budget.try_reserve(64).unwrap();
    let rep = own(bytemuck::cast_slice(&[1_u16, 0]), &mut work);
    let def = own(&[0, 0, 0], &mut work);
    let before = work.len();
    let error = RepDefUnraveler::try_from_encoded_in(
        Some((&codec, &rep)),
        Some((&codec, &def)),
        2,
        Arc::from([Meaning::NullableItem, Meaning::AllValidList]),
        2,
        &mut work,
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("expected 4"));
    assert_eq!(work.len(), before);
}

#[rstest]
fn one_or_no_encoded_stream(#[values(false, true)] has_def: bool) {
    let codec = ValueDecompressor::from_flat(&metadata::Flat {
        bits_per_value: 16,
        data: None,
    });
    let budget = AllocationBudget::new(14);
    let mut work = budget.try_reserve(14).unwrap();
    let input = own(bytemuck::cast_slice(&[0_u16, 1, 0]), &mut work);
    let source = RepDefUnraveler::try_from_encoded_in(
        None,
        has_def.then_some((&codec as &dyn BlockDecompressor, &input)),
        3,
        Arc::from([Meaning::NullableItem]),
        3,
        &mut work,
    )
    .unwrap()
    .unwrap();
    let mut source = CompositeRepDefUnraveler::new(vec![source]);
    let validity = source
        .try_unravel_validity_in(3, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(
        validity.as_ref().map(|v| v.iter().collect::<Vec<_>>()),
        has_def.then_some(vec![true, false, true])
    );
    drop(source);
    drop(input);
    drop(work);
    assert_eq!(budget.reserved(), usize::from(has_def));
    drop(validity);
    assert_eq!(budget.reserved(), 0);
}

#[cfg(feature = "bitpacking")]
mod bitpacked {
    use super::*;
    use crate::encodings::physical::bitpacking::{InlineBitpacking, OutOfLineBitpacking};
    use lance_bitpacking::BitPacking;

    fn padded(values: &[u16], width: usize) -> Vec<u16> {
        let mut input = [0_u16; 1024];
        input[..values.len()].copy_from_slice(values);
        let mut output = vec![0; width * 64];
        // SAFETY: one complete u16 chunk and exactly 1024*width/16 packed words.
        unsafe {
            u16::unchecked_pack(width, &input, &mut output);
        }
        output
    }

    #[test]
    fn two_streams_share_scratch_and_transfer_into_list_output() {
        let rep_bytes = padded(&[1, 0, 1, 1, 1, 0], 1);
        let def_bytes = padded(&[0, 1, 2, 3, 0, 0], 2);
        let input_bytes = 128 + 1 + 256 + 1;
        // During the second decode: 12 retained rep bytes + 12 def output +
        // 256 alignment bytes + 2048 full-tail scratch. First scratch is reused.
        let peak = 12 + 12 + 256 + 2048;
        let budget = AllocationBudget::new(input_bytes + peak);
        let mut work = budget.try_reserve(input_bytes + peak).unwrap();
        let rep = own(bytemuck::cast_slice(&rep_bytes), &mut work);
        let def = own(bytemuck::cast_slice(&def_bytes), &mut work);
        let rep_codec = OutOfLineBitpacking::new(1, 16);
        let def_codec = OutOfLineBitpacking::new(2, 16);
        let meaning = Arc::from([Meaning::NullableItem, Meaning::NullableAndEmptyableList]);
        let mut short = work.try_split(peak - 1).unwrap();
        assert!(
            RepDefUnraveler::try_from_encoded_in(
                Some((&rep_codec, &rep)),
                Some((&def_codec, &def)),
                6,
                Arc::clone(&meaning),
                4,
                &mut short,
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(short.len(), peak - 1);
        drop(short);
        drop(work);
        let mut work = budget.try_reserve(peak).unwrap();
        let source = RepDefUnraveler::try_from_encoded_in(
            Some((&rep_codec, &rep)),
            Some((&def_codec, &def)),
            6,
            meaning,
            4,
            &mut work,
        )
        .unwrap()
        .unwrap();
        assert_eq!(work.len(), peak - 24);
        drop(rep);
        drop(def);
        let mut source = CompositeRepDefUnraveler::new(vec![source]);
        let leaf = source
            .try_unravel_validity_in(4, &mut work)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(leaf.iter().collect::<Vec<_>>(), [true, false, true, true]);
        let (offsets, validity) = source
            .try_unravel_offsets_in::<i32>(&mut work)
            .unwrap()
            .unwrap();
        assert_eq!(offsets.as_ref(), &[0, 2, 2, 2, 4]);
        assert_eq!(
            validity.as_ref().unwrap().iter().collect::<Vec<_>>(),
            [true, false, true, true]
        );
        drop(work);
        assert_eq!(budget.reserved(), 24 + 1 + 20 + 1);
        drop(source);
        assert_eq!(budget.reserved(), 22);
        let alias = offsets.slice(0, 1);
        drop(offsets);
        drop(leaf);
        drop(validity);
        assert_eq!(budget.reserved(), 20);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn unsupported_second_codec_preserves_admission() {
        let codec = OutOfLineBitpacking::new(0, 16);
        let unsupported = InlineBitpacking::new(16);
        let input = LanceBuffer::empty();
        let budget = AllocationBudget::new(1024);
        let mut work = budget.try_reserve(1024).unwrap();
        let error = RepDefUnraveler::try_from_encoded_in(
            Some((&codec, &input)),
            Some((&unsupported, &input)),
            128,
            Arc::from([Meaning::NullableItem, Meaning::NullableList]),
            128,
            &mut work,
        )
        .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(
            error
                .to_string()
                .contains("no admitted structural-level decoder")
        );
        assert_eq!(work.len(), 1024);
    }
}
