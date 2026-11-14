// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::{EncodingUtils, metadata::compressive_encoding::Compression};

#[derive(Clone, Copy, Debug)]
enum Offsets {
    Constant,
    Range,
    Delta,
}

fn metadata_fixture(
    bits: u8,
    offsets: Offsets,
    large: bool,
    nested: bool,
    available: usize,
) -> Fixture {
    let encoding = match offsets {
        Offsets::Constant => EncodingUtils::constant(Some(bytes::Bytes::copy_from_slice(
            &0_u64.to_le_bytes()[..usize::from(bits / 8)],
        ))),
        Offsets::Range => EncodingUtils::range(u64::from(bits), 0, 3),
        Offsets::Delta => EncodingUtils::delta(
            u64::from(bits),
            0,
            EncodingUtils::constant(Some(bytes::Bytes::copy_from_slice(
                &3_u64.to_le_bytes()[..usize::from(bits / 8)],
            ))),
        ),
    };
    let encoding = EncodingUtils::variable(encoding, None);
    let Some(Compression::Variable(variable)) = encoding.compression else {
        panic!("expected Variable")
    };
    framed_fixture(
        if matches!(offsets, Offsets::Constant) {
            b""
        } else {
            b"abcdefghijkl"
        },
        BinaryMiniBlockDecompressor::from_variable(&variable).unwrap(),
        large,
        nested,
        available,
    )
}

#[rstest]
#[case::constant(Offsets::Constant)]
#[case::range(Offsets::Range)]
#[case::delta(Offsets::Delta)]
fn metadata_offsets_preserve_task_admission_and_structure(
    #[case] offsets: Offsets,
    #[values(32, 64)] bits: u8,
    #[values(false, true)] large: bool,
    #[values(false, true)] multiple: bool,
    #[values(false, true)] nested: bool,
) {
    let width = usize::from(bits / 8);
    let empty = matches!(offsets, Offsets::Constant);
    let all: &[&[u8]] = if empty {
        &[b"", b"", b"", b""]
    } else {
        &[b"abc", b"def", b"ghi", b"jkl"]
    };
    let selected = if multiple || nested { all } else { &all[1..3] };
    let payload_bytes = selected.iter().map(|v| v.len()).sum::<usize>();
    let output_bytes = (selected.len() + 1) * width + payload_bytes;
    let required = if multiple {
        11 * width + 2 * payload_bytes + if nested { 24 } else { 0 }
    } else {
        output_bytes + if nested { 48 } else { 0 }
    };
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = metadata_fixture(bits, offsets, large, nested, required);
    select(
        &mut task,
        if multiple {
            &[(0, 2), (2, 2)]
        } else if nested {
            &[(0, 4)]
        } else {
            &[(1, 2)]
        },
    );
    let mut short = work.try_split(required - 1).unwrap();
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), required - 1);
        assert_eq!(budget.reserved(), physical + required);
    }
    drop(short);
    drop(work);
    let mut work = budget.try_reserve(required).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let actual = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    let expected = if bits == 32 {
        BinaryArray::from_vec(selected.to_vec()).to_data()
    } else {
        LargeBinaryArray::from_vec(selected.to_vec()).to_data()
    };
    assert_eq!(actual, expected);
    assert_eq!(actual.get_buffer_memory_size(), output_bytes);
    assert_eq!(work.len(), if nested && !multiple { 24 } else { 0 });
    drop(task);
    drop(work);
    assert_eq!(
        budget.reserved(),
        output_bytes + if nested { 24 } else { 0 }
    );
    if nested {
        let mut structure = CompositeRepDefUnraveler::new(vec![page.repdef]);
        let mut reconstruction = budget.try_reserve(64).unwrap();
        let valid = structure
            .try_unravel_validity_in(4, &mut reconstruction)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(valid.iter().collect::<Vec<_>>(), [true, false, true, true]);
        let (offsets, validity) = structure
            .try_unravel_offsets_in::<i32>(&mut reconstruction)
            .unwrap()
            .unwrap();
        assert_eq!(offsets.as_ref(), [0, 2, 2, 2, 4]);
        assert_eq!(
            validity.unwrap().iter().collect::<Vec<_>>(),
            [true, false, true, true]
        );
    } else {
        drop(page.repdef);
    }
    drop(page.data);
    assert_eq!(budget.reserved(), output_bytes);
    drop(actual);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn later_metadata_error_recycles_structural_scratch(#[values(false, true)] large: bool) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = metadata_fixture(32, Offsets::Range, large, true, 1024);
    select(&mut task, &[(0, 2), (2, 2)]);
    // The second chunk claims one fewer value byte, while all framing stays valid.
    let mut bytes = task.instructions[1].1.data.to_vec();
    if large {
        bytes[6..10].copy_from_slice(&11_u32.to_le_bytes());
    } else {
        bytes[6..8].copy_from_slice(&11_u16.to_le_bytes());
    }
    let mut corrupt = work.try_allocate(bytes.len()).unwrap().unwrap();
    corrupt.as_mut().copy_from_slice(&bytes);
    task.instructions[1].1.data = LanceBuffer::from(Buffer::from(corrupt.freeze()));
    let available = work.len();
    let reserved = budget.reserved();
    for _ in 0..2 {
        let error = task.try_decode_in(&mut work).err().unwrap();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("value-buffer length"), "{error}");
        assert_eq!(work.len(), available);
        assert_eq!(budget.reserved(), reserved);
    }
    drop(task);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
