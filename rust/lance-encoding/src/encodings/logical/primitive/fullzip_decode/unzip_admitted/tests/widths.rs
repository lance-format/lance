// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[rstest]
#[case::nil(0, 0)]
#[case::rep8(1, 0)]
#[case::rep16(9, 0)]
#[case::rep32(17, 0)]
#[case::def8(0, 1)]
#[case::def16(0, 9)]
#[case::def32(0, 17)]
#[case::both8(1, 2)]
#[case::both16(1, 8)]
#[case::both32(1, 16)]
fn admitted_unzip_control_widths_preserve_values_and_structure(
    #[case] rep_bits: u8,
    #[case] def_bits: u8,
    #[values(32, 64)] offset_bits: u8,
) {
    let has_rep = rep_bits != 0;
    let has_def = def_bits != 0;
    // Literal row/leaf expectations, independent of the admitted traversal.
    let (events, meanings, rows, expected_values, expected_validity, expected_offsets) =
        match (has_rep, has_def) {
            (false, false) => (
                vec![(0_u32, 0_u32, b"ab".as_slice()), (0, 0, b"c")],
                vec![Meaning::AllValidItem],
                2,
                vec![b"ab".as_slice(), b"c"],
                vec![true, true],
                None,
            ),
            (true, false) => (
                vec![(1, 0, b"ab".as_slice()), (0, 0, b""), (1, 0, b"c")],
                vec![Meaning::AllValidItem, Meaning::AllValidList],
                2,
                vec![b"ab".as_slice(), b"", b"c"],
                vec![true, true, true],
                Some((vec![0, 2, 3], vec![true, true])),
            ),
            (false, true) => (
                vec![(0, 0, b"ab".as_slice()), (0, 1, b""), (0, 0, b"c")],
                vec![Meaning::NullableItem],
                3,
                vec![b"ab".as_slice(), b"", b"c"],
                vec![true, false, true],
                None,
            ),
            (true, true) => (
                vec![
                    (1, 0, b"ab".as_slice()),
                    (0, 1, b""),
                    (1, 3, b""),
                    (1, 0, b"c"),
                ],
                vec![Meaning::NullableItem, Meaning::NullableAndEmptyableList],
                3,
                vec![b"ab".as_slice(), b"", b"c"],
                vec![true, false, true],
                Some((vec![0, 2, 2, 3], vec![true, false, true])),
            ),
        };
    let parser = ControlWordParser::new(rep_bits, def_bits);
    let mut bytes = Vec::new();
    for (rep, def, value) in events {
        let control = (rep << def_bits) | def;
        bytes.extend_from_slice(&control.to_le_bytes()[..parser.bytes_per_word()]);
        if def == 0 {
            bytes.extend_from_slice(
                &(value.len() as u64).to_le_bytes()[..usize::from(offset_bits / 8)],
            );
            bytes.extend_from_slice(value);
        }
    }
    let mut f = fixture(offset_bits, false, "flat");
    let PerValueDecompressor::Variable(codec) = &f.scheduler.details.value_decompressor else {
        panic!("expected variable codec")
    };
    f.scheduler.details = Arc::new(super::super::super::FullZipDecodeDetails {
        value_decompressor: PerValueDecompressor::Variable(codec.clone()),
        ctrl_word_parser: parser,
        def_meaning: meanings.into(),
        max_rep: u16::from(has_rep),
        max_visible_def: u16::from(has_def),
    });
    let pool = AllocationBudget::new(8192);
    let mut work = pool.try_reserve(8192).unwrap();
    let mut input = work.try_allocate(bytes.len()).unwrap().unwrap();
    input.as_mut().copy_from_slice(&bytes);
    let input = [LanceBuffer::from(Buffer::from(input.freeze()))];
    let task = f
        .scheduler
        .try_unzip_range_in(&input, rows, 0..rows, &mut work)
        .unwrap()
        .unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let (ty, expected) = if offset_bits == 32 {
        (
            DataType::Binary,
            BinaryArray::from_vec(expected_values).to_data(),
        )
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(expected_values).to_data(),
        )
    };
    let arrow = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(arrow, expected);
    let mut structure = CompositeRepDefUnraveler::new(vec![page.repdef]);
    let bits = |validity: Option<arrow_buffer::NullBuffer>, len| {
        validity.map_or_else(|| vec![true; len], |v| v.iter().collect::<Vec<_>>())
    };
    let validity = structure
        .try_unravel_validity_in(expected_validity.len(), &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(bits(validity, expected_validity.len()), expected_validity);
    if let Some((expected_offsets, expected_validity)) = expected_offsets {
        let (offsets, validity) = structure
            .try_unravel_offsets_in::<i32>(&mut work)
            .unwrap()
            .unwrap();
        assert_eq!(&offsets[..], expected_offsets);
        assert_eq!(bits(validity, expected_validity.len()), expected_validity);
    }
    drop((input, task, page.data, arrow, structure, work));
    assert_eq!(pool.reserved(), 0);

    // A malformed unselected final item is rejected before output admission.
    bytes.truncate(bytes.len() - 2);
    let input = [LanceBuffer::from(bytes)];
    let mut no_credit = pool.try_reserve(0).unwrap();
    assert!(matches!(
        f.scheduler
            .try_unzip_range_in(&input, rows, 0..0, &mut no_credit),
        Err(Error::CorruptFile { .. })
    ));
    assert_eq!(pool.reserved(), 0);
}
