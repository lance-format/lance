// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    encodings::{
        logical::primitive::miniblock::{MiniBlockCompressionContext, MiniBlockCompressor},
        physical::fsst::{FsstMiniBlockDecompressor, FsstMiniBlockEncoder},
    },
    format::metadata::compressive_encoding::Compression,
};

#[rstest]
fn fsst_tasks_admit_expanded_values_levels_and_concat(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] large: bool,
    #[values(false, true)] multiple: bool,
    #[values(false, true)] nested: bool,
) {
    // The larger input's unsampled symbols expand enough to require a u32
    // header. Keep both fixtures above FSST's compression threshold, but do
    // not truncate that encoded length into a u16 header.
    let values = [
        vec![b'a'; 16384],
        vec![],
        vec![b'b'; if large { 32768 } else { 16384 }],
        b"d".to_vec(),
    ];
    let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let block = if bits == 32 {
        DataBlock::from_array(BinaryArray::from_vec(refs.clone()))
    } else {
        DataBlock::from_array(LargeBinaryArray::from_vec(refs.clone()))
    };
    let (encoded, description) = FsstMiniBlockEncoder::new(Some(1 << 20))
        .compress(MiniBlockCompressionContext::new(0, large, true), block)
        .unwrap();
    assert_eq!(encoded.chunks.len(), 1);
    assert_eq!(encoded.data.len(), 1);
    assert_eq!(
        encoded.chunks[0].buffer_sizes[0] as usize,
        encoded.data[0].len()
    );
    assert_eq!(encoded.data[0].len() > usize::from(u16::MAX), large);
    let Some(Compression::Fsst(description)) = description.compression else {
        panic!("expected FSST")
    };
    let header = u64::from_ne_bytes(description.symbol_table[..8].try_into().unwrap());
    assert_ne!(
        header & (1 << 24),
        0,
        "fixture must exercise FSST expansion"
    );
    let codec = FsstMiniBlockDecompressor::new(
        &description,
        Box::new(BinaryMiniBlockDecompressor::new(bits).unwrap()),
    );
    let selected = if multiple || nested {
        &refs[..]
    } else {
        &refs[1..3]
    };
    let width = usize::from(bits / 8);
    let payload = selected.iter().map(|v| v.len()).sum::<usize>();
    let output_bytes = (selected.len() + 1) * width + payload;
    let required = if multiple {
        11 * width + 2 * payload + if nested { 24 } else { 0 }
    } else {
        output_bytes + if nested { 48 } else { 0 }
    };
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = framed_fixture(&[&encoded.data[0]], codec, large, nested, required);
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
    drop((short, work));
    let mut work = budget.try_reserve(required).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    let (ty, expected) = if bits == 32 {
        (
            DataType::Binary,
            BinaryArray::from_vec(selected.to_vec()).to_data(),
        )
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(selected.to_vec()).to_data(),
        )
    };
    let array = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(array, expected);
    assert_eq!(array.get_buffer_memory_size(), output_bytes);
    drop((task, work));
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
    drop(array);
    assert_eq!(budget.reserved(), 0);
}
