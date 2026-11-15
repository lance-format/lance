// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::DataType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;
use crate::{
    compression::{
        DecompressionStrategy, DefaultDecompressionStrategy, PerValueDecodeBudget,
        VariablePerValueDecompressor,
    },
    data::DataBlock,
    decoder::DecodePageTask,
    encodings::{
        logical::primitive::{
            fullzip::{PerValueCompressor, PerValueDataBlock},
            fullzip_decode::{FullZipDecodeDetails, PerValueDecompressor},
        },
        physical::{
            binary::VariableEncoder,
            fsst::test_utils::{charge_per_value, per_value_fixture},
        },
    },
    repdef::{CompositeRepDefUnraveler, ControlWordParser, DefinitionInterpretation as Meaning},
};

fn levels(values: &[u16], work: &mut Reservation) -> ScalarBuffer<u16> {
    let mut storage = work
        .try_allocate(std::mem::size_of_val(values))
        .unwrap()
        .unwrap();
    storage
        .as_mut()
        .copy_from_slice(bytemuck::cast_slice(values));
    ScalarBuffer::new(Buffer::from(storage.freeze()), 0, values.len())
}

#[rstest]
#[case::fsst_passthrough("fsst_passthrough")]
#[case::fsst_compressed("fsst_compressed")]
#[case::raw("raw")]
fn fullzip_admits_value_and_structural_overlap(
    #[case] kind: &str,
    #[values(32, 64)] bits: u8,
    #[values(false, true)] nested: bool,
) {
    let (codec, input, values) = if kind == "raw" {
        let values = vec![b"ab".to_vec(), vec![], b"cd".to_vec(), b"e".to_vec()];
        let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let block = if bits == 32 {
            DataBlock::from_array(BinaryArray::from_vec(refs))
        } else {
            DataBlock::from_array(LargeBinaryArray::from_vec(refs))
        };
        let (encoded, description) = VariableEncoder::default().compress(block).unwrap();
        let PerValueDataBlock::Variable(input) = encoded else {
            panic!("expected raw variable values");
        };
        let codec = DefaultDecompressionStrategy::default()
            .create_variable_per_value_decompressor(&description)
            .unwrap();
        (codec, input, values)
    } else {
        let (codec, input, values, _) = per_value_fixture(bits, kind == "fsst_compressed");
        (codec, input, values)
    };
    let output_bytes = 5 * usize::from(bits / 8) + values.iter().map(Vec::len).sum::<usize>();
    let level_bytes = if nested { 24 } else { 0 };
    let needed = output_bytes + level_bytes;
    let physical = 1 + input.offsets.len() + 3 + input.data.len() + 37;
    let pool = AllocationBudget::new(physical + level_bytes + needed);
    let mut work = pool.try_reserve(physical + level_bytes + needed).unwrap();
    let (input, sibling, _) = charge_per_value(input, 1, &mut work);
    let rep = levels(if nested { &[1, 0, 1, 1, 1, 0] } else { &[] }, &mut work);
    let def = levels(if nested { &[0, 1, 2, 3, 0, 0] } else { &[] }, &mut work);
    let meanings: Arc<[Meaning]> = if nested {
        Arc::from([Meaning::NullableItem, Meaning::NullableAndEmptyableList])
    } else {
        Arc::from([Meaning::AllValidItem])
    };
    let codec: Arc<dyn VariablePerValueDecompressor> = Arc::from(codec);
    let details = Arc::new(FullZipDecodeDetails {
        value_decompressor: PerValueDecompressor::Variable(codec.clone()),
        max_rep: u16::from(nested),
        max_visible_def: max_visible_level(&meanings),
        def_meaning: meanings,
        ctrl_word_parser: ControlWordParser::new(u8::from(nested), if nested { 2 } else { 0 }),
    });
    let task = VariableFullZipDecodeTask {
        details,
        decompressor: codec,
        data: input.data,
        offsets: input.offsets,
        bits_per_offset: bits,
        num_visible_items: 4,
        rep,
        def,
    };
    let mut short = work.try_split(needed - 1).unwrap();
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), needed - 1);
        assert_eq!(pool.reserved(), physical + level_bytes + needed);
    }
    drop((short, work));
    let mut work = pool.try_reserve(needed).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert!(work.is_empty());
    let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from_vec(refs).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(refs).to_data(),
        )
    };
    let array = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(array, expected);
    assert_eq!(array.get_buffer_memory_size(), output_bytes);
    drop((task, work));
    assert_eq!(pool.reserved(), physical + needed);
    drop(sibling);
    assert_eq!(pool.reserved(), needed);
    if nested {
        let mut structure = CompositeRepDefUnraveler::new(vec![page.repdef]);
        let mut reconstruction = pool.try_reserve(64).unwrap();
        let validity = structure
            .try_unravel_validity_in(4, &mut reconstruction)
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            validity.iter().collect::<Vec<_>>(),
            [true, false, true, true]
        );
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
    assert_eq!(pool.reserved(), output_bytes);
    drop(array);
    assert_eq!(pool.reserved(), 0);
}

#[derive(Debug)]
struct RefuseOrFail(bool);

impl VariablePerValueDecompressor for RefuseOrFail {
    fn decompress(&self, _: VariableWidthBlock) -> Result<DataBlock> {
        panic!("ordinary decode must not be called")
    }
    fn plan_decode(&self, _: &VariableWidthBlock) -> Result<PerValueDecodeBudget> {
        Ok(PerValueDecodeBudget {
            output_bytes: 8,
            workspace_bytes: 8,
        })
    }
    fn try_decompress_in(
        &self,
        _: &VariableWidthBlock,
        _: &mut Reservation,
    ) -> Result<Option<DataBlock>> {
        if self.0 {
            Err(Error::corrupt_file_named("test", "injected codec error"))
        } else {
            Ok(None)
        }
    }
}

#[rstest]
fn fullzip_failure_recycles_structural_credit(#[values(false, true)] fail: bool) {
    let (_, input, _, _) = per_value_fixture(32, false);
    let pool = AllocationBudget::new(256);
    let mut work = pool.try_reserve(256).unwrap();
    let (input, sibling, _) = charge_per_value(input, 1, &mut work);
    let def = levels(&[0, 1, 0, 0], &mut work);
    let codec: Arc<dyn VariablePerValueDecompressor> = Arc::new(RefuseOrFail(fail));
    let task = VariableFullZipDecodeTask {
        details: Arc::new(FullZipDecodeDetails {
            value_decompressor: PerValueDecompressor::Variable(codec.clone()),
            def_meaning: Arc::from([Meaning::NullableItem]),
            ctrl_word_parser: ControlWordParser::new(0, 1),
            max_rep: 0,
            max_visible_def: 1,
        }),
        decompressor: codec,
        data: input.data,
        offsets: input.offsets,
        bits_per_offset: 32,
        num_visible_items: 4,
        rep: ScalarBuffer::from(Vec::<u16>::new()),
        def,
    };
    let available = work.len();
    for _ in 0..2 {
        let result = task.try_decode_in(&mut work);
        if fail {
            let Err(error) = result else {
                panic!("expected the injected codec error");
            };
            assert!(matches!(error, Error::CorruptFile { .. }));
            assert!(error.to_string().contains("injected codec error"));
        } else {
            assert!(result.unwrap().is_none());
        }
        assert_eq!(work.len(), available);
        assert_eq!(pool.reserved(), 256);
    }
    drop((task, sibling, work));
    assert_eq!(pool.reserved(), 0);
}
