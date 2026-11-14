// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::{DecompressionStrategy, DefaultDecompressionStrategy},
    format::EncodingUtils,
};

fn frame(values: &[u16], width: usize) -> Vec<u8> {
    let mut data = ((values.len() * 2) as u64).to_le_bytes().to_vec();
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    for _ in values {
        data.extend_from_slice(&1_u32.to_le_bytes()[..width]);
    }
    data
}

fn codec(width: usize) -> Box<dyn BlockDecompressor> {
    DefaultDecompressionStrategy::default()
        .create_block_decompressor(&EncodingUtils::rle(
            EncodingUtils::flat(16, None),
            EncodingUtils::flat((width * 8) as u64, None),
        ))
        .unwrap()
}

#[rstest]
fn joint_admission_composes_null_empty_lists(#[values(1, 2, 4)] width: usize) {
    let rep_bytes = frame(&[1, 0, 1, 1, 1, 0], width);
    let def_bytes = frame(&[0, 1, 2, 3, 0, 0], width);
    let input_bytes = rep_bytes.len() + def_bytes.len() + 2;
    // 24 bytes of levels stay charged alongside the 1-byte leaf validity,
    // five i32 offsets and 1-byte list validity. No compressed/run scratch.
    let budget = AllocationBudget::new(input_bytes + 46);
    let mut work = budget.try_reserve(input_bytes + 46).unwrap();
    let rep = own(&rep_bytes, &mut work);
    let def = own(&def_bytes, &mut work);
    let codec = codec(width);
    let meanings = Arc::from([Meaning::NullableItem, Meaning::NullableAndEmptyableList]);
    let mut short = work.try_split(23).unwrap();
    assert!(
        RepDefUnraveler::try_from_encoded_in(
            Some((codec.as_ref(), &rep)),
            Some((codec.as_ref(), &def)),
            6,
            Arc::clone(&meanings),
            4,
            &mut short,
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(short.len(), 23);
    drop(short);
    drop(work);
    let mut work = budget.try_reserve(46).unwrap();
    let levels = RepDefUnraveler::try_from_encoded_in(
        Some((codec.as_ref(), &rep)),
        Some((codec.as_ref(), &def)),
        6,
        meanings,
        4,
        &mut work,
    )
    .unwrap()
    .unwrap();
    assert_eq!(work.len(), 22);
    drop(rep);
    drop(def);
    let mut levels = CompositeRepDefUnraveler::new(vec![levels]);
    let leaf = levels
        .try_unravel_validity_in(4, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    let (offsets, validity) = levels
        .try_unravel_offsets_in::<i32>(&mut work)
        .unwrap()
        .unwrap();
    assert_eq!(leaf.iter().collect::<Vec<_>>(), [true, false, true, true]);
    assert_eq!(offsets.as_ref(), &[0, 2, 2, 2, 4]);
    assert_eq!(
        validity.as_ref().unwrap().iter().collect::<Vec<_>>(),
        [true, false, true, true]
    );
    assert_eq!(work.len(), 0);
    drop(work);
    assert_eq!(budget.reserved(), 46);
    drop(levels);
    assert_eq!(budget.reserved(), 22);
    let last = offsets.slice(0, 1);
    drop(offsets);
    drop(leaf);
    drop(validity);
    assert_eq!(budget.reserved(), 20);
    drop(last);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn malformed_second_stream_preserves_complete_credit(#[values(false, true)] bad_level: bool) {
    let rep_bytes = frame(&[1, 0], 1);
    let mut def_bytes = frame(&[0, if bad_level { 3 } else { 0 }], 1);
    if !bad_level {
        // A zero run is invalid framing, even though the overall buffer fits.
        *def_bytes.last_mut().unwrap() = 0;
    }
    let input_bytes = rep_bytes.len() + def_bytes.len() + 2;
    let budget = AllocationBudget::new(input_bytes + 8);
    let mut work = budget.try_reserve(input_bytes + 8).unwrap();
    let rep = own(&rep_bytes, &mut work);
    let def = own(&def_bytes, &mut work);
    let codec = codec(1);
    let error = RepDefUnraveler::try_from_encoded_in(
        Some((codec.as_ref(), &rep)),
        Some((codec.as_ref(), &def)),
        2,
        Arc::from([Meaning::NullableItem, Meaning::NullableList]),
        2,
        &mut work,
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(if bad_level {
        "level exceeds"
    } else {
        "length is zero"
    }));
    assert_eq!(work.len(), 8);
    assert_eq!(budget.reserved(), input_bytes + 8);
    drop(rep);
    drop(def);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
