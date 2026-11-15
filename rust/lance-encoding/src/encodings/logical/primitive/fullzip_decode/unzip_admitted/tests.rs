// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use arrow_schema::DataType;
use lance_core::{datatypes::Field, utils::allocation::AllocationBudget};
use rstest::rstest;

use super::*;
use crate::{
    compression::{BlockCompressor, CompressionStrategy, DefaultDecompressionStrategy},
    data::DataBlock,
    decoder::PageEncoding,
    encodings::{
        logical::primitive::{
            PrimitiveStructuralEncoder, fullzip::PerValueCompressor, miniblock::MiniBlockCompressor,
        },
        physical::{binary::VariableEncoder, fsst::FsstPerValueEncoder},
    },
    format::metadata::page_layout::Layout,
    repdef::{CompositeRepDefUnraveler, DefinitionInterpretation as Meaning, SerializedRepDefs},
    utils::bytepack::ByteUnpacker,
};

#[derive(Debug)]
struct Codec(bool);

impl CompressionStrategy for Codec {
    fn create_per_value(&self, _: &Field, _: &DataBlock) -> Result<Box<dyn PerValueCompressor>> {
        let raw = Box::new(VariableEncoder::default());
        Ok(if self.0 {
            Box::new(FsstPerValueEncoder::new(raw))
        } else {
            raw
        })
    }
    fn create_block_compressor(
        &self,
        _: &Field,
        _: &DataBlock,
    ) -> Result<Box<dyn BlockCompressor>> {
        panic!("FullZip writer must use per-value compression")
    }
    fn create_miniblock_compressor(
        &self,
        _: &Field,
        _: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        panic!("FullZip writer must not use MiniBlock compression")
    }
}

struct Fixture {
    scheduler: FullZipScheduler,
    zipped: LanceBuffer,
    index: LanceBuffer,
    row_offsets: Vec<usize>,
    values: Vec<Vec<u8>>,
    meanings: Vec<Meaning>,
    rep: Option<Vec<u16>>,
    def: Option<Vec<u16>>,
    value_starts: Vec<usize>,
    level_starts: Vec<usize>,
}

fn fixture(bits: u8, fsst: bool, kind: &str) -> Fixture {
    let values = vec![vec![b'a'; 16384], vec![], vec![b'b'; 16384], vec![]];
    fixture_with_values(bits, fsst, kind, values)
}

fn fixture_with_values(bits: u8, fsst: bool, kind: &str, values: Vec<Vec<u8>>) -> Fixture {
    let rows = values.len();
    if kind != "flat" {
        assert_eq!(rows, 4);
    }
    let (rep, def, meanings, value_starts, level_starts) = match kind {
        "flat" => (
            None,
            None,
            vec![Meaning::AllValidItem],
            (0..=rows).collect(),
            (0..=rows).collect(),
        ),
        "nullable" => (
            None,
            Some(vec![0, 1, 0, 0]),
            vec![Meaning::NullableItem],
            vec![0, 1, 2, 3, 4],
            vec![0, 1, 2, 3, 4],
        ),
        "list" => (
            Some(vec![1, 0, 1, 1, 1, 0]),
            Some(vec![0, 1, 3, 2, 0, 0]),
            vec![Meaning::NullableItem, Meaning::NullableAndEmptyableList],
            vec![0, 2, 2, 2, 4],
            vec![0, 2, 3, 4, 6],
        ),
        "deep" => (
            Some(vec![2, 0, 1, 1, 2, 2, 2, 0]),
            Some(vec![0, 1, 3, 2, 5, 4, 0, 0]),
            vec![
                Meaning::NullableItem,
                Meaning::NullableAndEmptyableList,
                Meaning::NullableAndEmptyableList,
            ],
            vec![0, 2, 2, 2, 4],
            vec![0, 4, 5, 6, 8],
        ),
        _ => unreachable!(),
    };
    let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (ty, block) = if bits == 32 {
        (
            DataType::Binary,
            DataBlock::from_array(BinaryArray::from_vec(refs)),
        )
    } else {
        (
            DataType::LargeBinary,
            DataBlock::from_array(LargeBinaryArray::from_vec(refs)),
        )
    };
    let field = Field::try_from(&arrow_schema::Field::new("values", ty, true)).unwrap();
    let page = PrimitiveStructuralEncoder::encode_full_zip(
        0,
        &field,
        &Codec(fsst),
        block,
        SerializedRepDefs::new(rep.clone(), def.clone(), meanings.clone()),
        0,
        rows as u64,
    )
    .unwrap();
    let PageEncoding::Structural(description) = page.description else {
        panic!("expected structural page")
    };
    let Some(Layout::FullZipLayout(layout)) = description.layout else {
        panic!("expected FullZip")
    };
    let locations = page
        .data
        .iter()
        .scan(0_u64, |offset, data| {
            let location = (*offset, data.len() as u64);
            *offset += data.len() as u64;
            Some(location)
        })
        .collect::<Vec<_>>();
    let scheduler = FullZipScheduler::try_new(
        &locations,
        0,
        rows as u64,
        &layout,
        &DefaultDecompressionStrategy::default(),
    )
    .unwrap();
    let row_offsets = ByteUnpacker::new(page.data[1].clone(), page.data[1].len() / (rows + 1))
        .map(|offset| usize::try_from(offset).unwrap())
        .collect();
    Fixture {
        scheduler,
        zipped: page.data[0].clone(),
        index: page.data[1].clone(),
        row_offsets,
        values,
        meanings,
        rep,
        def,
        value_starts,
        level_starts,
    }
}

// Row-aligned input slices share a larger physical owner, including every gap.
fn charge(f: &Fixture, work: &mut Reservation) -> (Vec<LanceBuffer>, Buffer, usize) {
    let physical = f.zipped.len() + 4 * 13 + 1;
    let mut allocation = work.try_allocate(physical).unwrap().unwrap();
    let mut ranges = Vec::new();
    let mut cursor = 1;
    for row in f.row_offsets.windows(2) {
        let value = &f.zipped[row[0]..row[1]];
        allocation.as_mut()[cursor..cursor + value.len()].copy_from_slice(value);
        ranges.push(cursor..cursor + value.len());
        cursor += value.len() + 13;
    }
    let owner = Buffer::from(allocation.freeze());
    let input = ranges
        .into_iter()
        .map(|r| LanceBuffer::from(owner.slice_with_length(r.start, r.len())))
        .collect();
    (input, owner.slice_with_length(physical - 1, 1), physical)
}

#[rstest]
fn fullzip_range_unzip_bounds_physical_overlap_and_semantics(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] fsst: bool,
    #[values("flat", "nullable", "list", "deep")] kind: &str,
    #[values(0..4, 1..3, 3..4, 4..4)] range: Range<u64>,
) {
    let f = fixture(bits, fsst, kind);
    let selected = f.value_starts[range.start as usize]..f.value_starts[range.end as usize];
    let level_range = f.level_starts[range.start as usize]..f.level_starts[range.end as usize];
    let levels =
        level_range.len() * 2 * (usize::from(f.rep.is_some()) + usize::from(f.def.is_some()));
    let final_values = (selected.len() + 1) * usize::from(bits / 8)
        + f.values[selected.clone()]
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    let decode_bytes = final_values + levels;
    let stage_bytes = f
        .scheduler
        .plan_unzip_range(std::slice::from_ref(&f.zipped), 4, range.clone())
        .unwrap();
    let valid_items = f.def.as_ref().map_or(selected.len(), |def| {
        def[level_range.clone()].iter().filter(|d| **d == 0).count()
    });
    let encoded_payload = f.row_offsets[range.end as usize]
        - f.row_offsets[range.start as usize]
        - level_range.len() * f.scheduler.details.ctrl_word_parser.bytes_per_word()
        - valid_items * usize::from(bits / 8);
    assert_eq!(
        stage_bytes,
        encoded_payload + (selected.len() + 1) * usize::from(bits / 8) + levels
    );
    let physical = f.zipped.len() + 4 * 13 + 1;
    let pool = AllocationBudget::new(physical + stage_bytes + decode_bytes);
    let mut work = pool
        .try_reserve(physical + stage_bytes + decode_bytes)
        .unwrap();
    let (input, sibling, actual) = charge(&f, &mut work);
    assert_eq!(actual, physical);
    assert_eq!(
        f.scheduler
            .plan_unzip_range(&input, 4, range.clone())
            .unwrap(),
        stage_bytes
    );
    let mut short = work.try_split(stage_bytes - 1).unwrap();
    for _ in 0..2 {
        assert!(
            f.scheduler
                .try_unzip_range_in(&input, 4, range.clone(), &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), stage_bytes - 1);
        assert_eq!(pool.reserved(), physical + stage_bytes + decode_bytes);
    }
    drop((work, short));
    let mut work = pool.try_reserve(stage_bytes + decode_bytes).unwrap();
    let task = f
        .scheduler
        .try_unzip_range_in(&input, 4, range, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), decode_bytes);
    let mut short = work.try_split(decode_bytes - 1).unwrap();
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), decode_bytes - 1);
    }
    drop((work, short));
    let mut work = pool.try_reserve(decode_bytes).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert!(work.is_empty());
    let refs = f.values[selected.clone()]
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from_vec(refs).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(refs).to_data(),
        )
    };
    let arrow = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(arrow, expected);
    drop((input, work));
    assert_eq!(pool.reserved(), physical + stage_bytes + decode_bytes);
    drop(sibling);
    assert_eq!(pool.reserved(), stage_bytes + decode_bytes);
    drop(task);
    assert_eq!(pool.reserved(), decode_bytes);
    let expected_repdef = RepDefUnraveler::new(
        f.rep.as_ref().map(|v| v[level_range.clone()].to_vec()),
        f.def.as_ref().map(|v| v[level_range].to_vec()),
        f.meanings.clone().into(),
        selected.len() as u64,
    );
    let mut expected_structure = CompositeRepDefUnraveler::new(vec![expected_repdef]);
    let mut structure = CompositeRepDefUnraveler::new(vec![page.repdef]);
    let mut reconstruction = pool.try_reserve(1024).unwrap();
    let validity = structure
        .try_unravel_validity_in(selected.len(), &mut reconstruction)
        .unwrap()
        .unwrap();
    let expected_validity = expected_structure.unravel_validity(selected.len()).unwrap();
    let bits = |validity: Option<arrow_buffer::NullBuffer>, len| {
        validity.map_or_else(|| vec![true; len], |v| v.iter().collect::<Vec<_>>())
    };
    assert_eq!(
        bits(validity, selected.len()),
        bits(expected_validity, selected.len())
    );
    for _ in f.meanings.iter().filter(|m| m.is_list()) {
        let (offsets, validity) = structure
            .try_unravel_offsets_in::<i32>(&mut reconstruction)
            .unwrap()
            .unwrap();
        let (expected_offsets, expected_validity) =
            expected_structure.unravel_offsets::<i32>().unwrap();
        assert_eq!(&offsets[..], &expected_offsets[..]);
        assert_eq!(
            bits(validity, offsets.len() - 1),
            bits(expected_validity, expected_offsets.len() - 1)
        );
    }
    drop((structure, reconstruction, page.data, arrow));
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn owned_raw_unzip_reuses_only_charged_compact_values(
    #[values(32, 64)] bits: u8,
    #[values("flat", "nullable", "list", "deep")] kind: &str,
    #[values(0..4, 1..3, 3..4, 4..4)] range: Range<u64>,
) {
    let f = fixture(bits, false, kind);
    let selected = f.value_starts[range.start as usize]..f.value_starts[range.end as usize];
    let level_range = f.level_starts[range.start as usize]..f.level_starts[range.end as usize];
    let levels =
        level_range.len() * 2 * (usize::from(f.rep.is_some()) + usize::from(f.def.is_some()));
    let stage = f
        .scheduler
        .plan_unzip_range(std::slice::from_ref(&f.zipped), 4, range.clone())
        .unwrap();
    let physical = f.zipped.len() + 4 * 13 + 1;
    let pool = AllocationBudget::new(physical + stage + levels);
    let mut work = pool.try_reserve(physical + stage + levels).unwrap();
    let (input, sibling, _) = charge(&f, &mut work);
    let task = f
        .scheduler
        .try_unzip_owned_in(&input, 4, range, &mut work)
        .unwrap()
        .unwrap();
    drop((input, sibling));
    assert_eq!(work.len(), levels, "no room for a second copy of values");
    let page = task.decode(&mut work).unwrap().unwrap();
    assert!(work.is_empty());
    assert_eq!(pool.reserved(), stage + levels);
    let refs = f.values[selected.clone()]
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from_vec(refs).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(refs).to_data(),
        )
    };
    let arrow = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(arrow, expected);
    // Even an empty Arrow slice retains the complete exact unzip allocation.
    let alias = arrow.slice(0, 0);
    drop((page, arrow, work));
    assert_eq!(pool.reserved(), stage);
    let peer = pool.try_reserve(physical + levels).unwrap();
    assert!(pool.try_reserve(1).is_none());
    drop(alias);
    assert_eq!(pool.reserved(), peer.len());
    drop(peer);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::insufficient_levels("nullable", false)]
#[case::compressed_values("flat", true)]
fn owned_unzip_refusal_releases_compact_input(#[case] kind: &str, #[case] fsst: bool) {
    let f = fixture(32, fsst, kind);
    let stage = f
        .scheduler
        .plan_unzip_range(std::slice::from_ref(&f.zipped), 4, 0..4)
        .unwrap();
    let physical = f.zipped.len() + 4 * 13 + 1;
    let pool = AllocationBudget::new(physical + stage);
    let mut work = pool.try_reserve(physical + stage).unwrap();
    let (input, sibling, _) = charge(&f, &mut work);
    let task = f
        .scheduler
        .try_unzip_owned_in(&input, 4, 0..4, &mut work)
        .unwrap()
        .unwrap();
    drop((input, sibling));
    assert!(work.is_empty());
    assert!(task.decode(&mut work).unwrap().is_none());
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::truncated_word("word", "control word")]
#[case::truncated_length("length", "length prefix")]
#[case::long_value("payload", "item length")]
#[case::invalid_level("level", "level exceeds")]
#[case::continuation("continuation", "inside a row")]
#[case::extra_row("rows", "more rows")]
#[case::short_rows("missing", "row count")]
fn fullzip_unzip_rejects_invalid_tail_before_admission(#[case] kind: &str, #[case] message: &str) {
    let mut f = fixture(64, false, "deep");
    let mut bytes = f.zipped.as_ref().to_vec();
    let mut rows = 4;
    match kind {
        "word" => {
            f.scheduler.details = Arc::new(super::super::FullZipDecodeDetails {
                ctrl_word_parser: ControlWordParser::new(8, 8),
                value_decompressor: super::super::PerValueDecompressor::Variable(Arc::new(
                    crate::encodings::physical::binary::VariableDecoder::default(),
                )),
                def_meaning: f.meanings.clone().into(),
                max_rep: 2,
                max_visible_def: 1,
            });
            bytes = vec![0];
        }
        "length" => {
            bytes.truncate(bytes.len() - 1);
        }
        "payload" => {
            let i = f.row_offsets[3] + 1;
            bytes[i..i + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        }
        "level" => {
            let i = f.row_offsets[3];
            bytes[i] = 255;
        }
        "continuation" => {
            bytes[0] = 0;
        }
        "rows" => {
            rows = 3;
        }
        "missing" => {
            rows = 5;
        }
        _ => unreachable!(),
    }
    let input = [LanceBuffer::from(bytes)];
    let pool = AllocationBudget::new(0);
    let mut work = pool.try_reserve(0).unwrap();
    for _ in 0..2 {
        let Err(error) = f
            .scheduler
            .try_unzip_range_in(&input, rows, 0..0, &mut work)
        else {
            panic!("invalid unselected tail must fail")
        };
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
        assert!(work.is_empty());
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn fullzip_unzip_empty_input_and_cancel_release_ownership() {
    let f = fixture(32, false, "flat");
    let pool = AllocationBudget::new(4);
    let mut work = pool.try_reserve(4).unwrap();
    let task = f
        .scheduler
        .try_unzip_range_in(&[], 0, 0..0, &mut work)
        .unwrap()
        .unwrap();
    assert!(work.is_empty());
    assert_eq!(pool.reserved(), 4);
    drop(task);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn fullzip_unzip_finite_consumers_reclaim_complete_stage(#[values(false, true)] fsst: bool) {
    let f = fixture(32, fsst, "deep");
    let needed = f
        .scheduler
        .plan_unzip_range(std::slice::from_ref(&f.zipped), 4, 0..1)
        .unwrap();
    let physical = f.zipped.len() + 4 * 13 + 1;
    let pool = AllocationBudget::new(physical + 2 * needed);
    let mut input_work = pool.try_reserve(physical).unwrap();
    let (input, sibling, _) = charge(&f, &mut input_work);
    drop((sibling, input_work));
    let consume = || {
        let mut work = pool.try_reserve(needed).unwrap();
        f.scheduler
            .try_unzip_range_in(&input, 4, 0..1, &mut work)
            .unwrap()
            .unwrap()
    };
    let (first, second) = std::thread::scope(|scope| {
        let a = scope.spawn(consume);
        let b = scope.spawn(consume);
        (a.join().unwrap(), b.join().unwrap())
    });
    assert!(pool.try_reserve(1).is_none());
    drop(first);
    let third = consume();
    assert_eq!(pool.reserved(), physical + 2 * needed);
    drop((input, second, third));
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn fullzip_unzip_invalid_request_preserves_credit(#[values(false, true)] invalid_width: bool) {
    let mut f = fixture(32, false, "flat");
    let range = if invalid_width {
        f.scheduler.bits_per_offset = 16;
        0..4
    } else {
        0..5
    };
    let pool = AllocationBudget::new(16);
    let mut work = pool.try_reserve(16).unwrap();
    let Err(error) =
        f.scheduler
            .try_unzip_range_in(std::slice::from_ref(&f.zipped), 4, range, &mut work)
    else {
        panic!("invalid request must be rejected");
    };
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("offset width or unzip row range")
    );
    assert_eq!(work.len(), 16);
    assert_eq!(pool.reserved(), 16);
}

mod read;
mod widths;
