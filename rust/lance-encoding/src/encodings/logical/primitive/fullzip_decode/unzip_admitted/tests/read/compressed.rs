// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::encodings::physical::block::{
    CompressedBufferEncoder, CompressionConfig, CompressionScheme,
};
use arrow_data::ArrayData;

#[derive(Debug)]
struct GeneralCodec(CompressionScheme);

impl CompressionStrategy for GeneralCodec {
    fn create_per_value(&self, _: &Field, _: &DataBlock) -> Result<Box<dyn PerValueCompressor>> {
        Ok(Box::new(CompressedBufferEncoder::try_new(
            CompressionConfig::new(self.0, Some(0)),
        )?))
    }
    fn create_block_compressor(
        &self,
        _: &Field,
        _: &DataBlock,
    ) -> Result<Box<dyn BlockCompressor>> {
        panic!("FullZip requires per-value compression")
    }
    fn create_miniblock_compressor(
        &self,
        _: &Field,
        _: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        panic!("FullZip requires per-value compression")
    }
}

fn compress(mut f: Fixture, scheme: CompressionScheme) -> Fixture {
    let bits = f.scheduler.bits_per_offset;
    let refs = f.values.iter().map(Vec::as_slice).collect::<Vec<_>>();
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
    let rows = f.row_offsets.len() - 1;
    let field = Field::try_from(&arrow_schema::Field::new("values", ty, true)).unwrap();
    let page = PrimitiveStructuralEncoder::encode_full_zip(
        0,
        &field,
        &GeneralCodec(scheme),
        block,
        SerializedRepDefs::new(f.rep.clone(), f.def.clone(), f.meanings.clone()),
        0,
        rows as u64,
    )
    .unwrap();
    let PageEncoding::Structural(description) = page.description else {
        panic!("structural page");
    };
    let Some(Layout::FullZipLayout(layout)) = description.layout else {
        panic!("FullZip required");
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
    f.scheduler = FullZipScheduler::try_new(
        &locations,
        0,
        rows as u64,
        &layout,
        &DefaultDecompressionStrategy::default(),
    )
    .unwrap();
    f.row_offsets = ByteUnpacker::new(page.data[1].clone(), page.data[1].len() / (rows + 1))
        .map(|offset| usize::try_from(offset).unwrap())
        .collect();
    f.zipped = page.data[0].clone();
    f.index = page.data[1].clone();
    f
}

// Price canonical final buffers from the independent Arrow oracle, including
// the selected child interval, not any source backing capacity or decoder plan.
fn final_bytes(data: &ArrayData) -> usize {
    let width = match data.data_type() {
        DataType::Binary | DataType::Utf8 | DataType::List(_) => 4,
        DataType::LargeBinary | DataType::LargeUtf8 | DataType::LargeList(_) => 8,
        _ => unreachable!(),
    };
    let offset = |row: usize| {
        let bytes = &data.buffers()[0][row * width..(row + 1) * width];
        if width == 4 {
            i32::from_ne_bytes(bytes.try_into().unwrap()) as usize
        } else {
            i64::from_ne_bytes(bytes.try_into().unwrap()) as usize
        }
    };
    let start = offset(data.offset());
    let end = offset(data.offset() + data.len());
    let payload = if data.child_data().is_empty() {
        end - start
    } else {
        final_bytes(&data.child_data()[0].slice(start, end - start))
    };
    (data.len() + 1) * width
        + payload
        + if data.null_count() == 0 {
            0
        } else {
            data.len().div_ceil(8)
        }
}

#[rstest]
#[tokio::test]
async fn compressed_array_needs_only_input_final_buffers_and_context(
    #[values(CompressionScheme::Zstd, CompressionScheme::Lz4)] scheme: CompressionScheme,
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    ty: DataType,
    #[values("flat", "list", "large_list")] shape: &str,
    #[values(0..12, 3..10, 1..3)] rows: Range<u64>,
) {
    let (f, expected) = if shape == "flat" {
        let bits = if matches!(ty, DataType::Binary | DataType::Utf8) {
            32
        } else {
            64
        };
        let f = fixture_with_values(
            bits,
            false,
            "nullable",
            (0..12)
                .map(|row| {
                    if row % 4 == 0 {
                        Vec::new()
                    } else {
                        format!("value {row} αβ").repeat(100).into_bytes()
                    }
                })
                .collect(),
        );
        let values = f
            .values
            .iter()
            .enumerate()
            .map(|(row, value)| (row % 3 != 1).then_some(value.as_slice()))
            .collect::<Vec<_>>();
        let expected = arrow_array::make_array(super::flat::expected(&ty, &values));
        (f, expected)
    } else {
        super::list::list_case(
            &ty,
            shape == "large_list",
            true,
            Meaning::NullableAndEmptyableList,
        )
    };
    let expected = expected.slice(rows.start as usize, (rows.end - rows.start) as usize);
    let output_bytes = final_bytes(&expected.to_data());
    let mut f = compress(f, scheme);
    let input_bytes = f.row_offsets[rows.end as usize] - f.row_offsets[rows.start as usize];
    let scratch = if scheme == CompressionScheme::Zstd {
        // SAFETY: pure pinned-library size query with no pointers or allocations.
        // This oracle does not consult the decoder planner.
        unsafe { ::zstd::zstd_safe::zstd_sys::ZSTD_estimateDCtxSize() }
    } else {
        0
    };
    let limit = input_bytes + output_bytes + scratch;
    let source = source(&mut f);
    let scheduler = Arc::new(f.scheduler);
    for shortage in [true, false] {
        let capacity = limit - usize::from(shortage);
        let pool = AllocationBudget::new(capacity);
        let mut work = pool.try_reserve(capacity).unwrap();
        let input = scheduler
            .clone()
            .prepare_array_range_in(
                rows.clone(),
                expected.data_type().clone(),
                &mut work,
                reader(source.clone(), Arc::default(), Fault::None),
            )
            .await
            .unwrap()
            .unwrap();
        let output = input.decode_in(&mut work).unwrap();
        if shortage {
            assert!(output.is_none());
            assert_eq!(work.len(), capacity);
            drop(work);
            assert_eq!(pool.reserved(), 0);
            continue;
        }
        let output = output.unwrap();
        assert_eq!(output.to_data(), expected.to_data());
        assert_eq!(work.len(), input_bytes + scratch);
        let data = output.to_data();
        let parent = data.buffers()[0].clone();
        let leaf = data.child_data().first().unwrap_or(&data);
        let alias = leaf.buffers()[0].slice_with_length(0, 1);
        let child_bytes = final_bytes(leaf);
        drop((data, output, work));
        assert_eq!(pool.reserved(), output_bytes);
        drop(parent);
        assert_eq!(pool.reserved(), child_bytes);
        assert!(pool.try_reserve(capacity).is_none());
        drop(alias);
        assert_eq!(pool.reserved(), 0);
        let mut work = pool.try_reserve(capacity).unwrap();
        let input = scheduler
            .clone()
            .prepare_array_range_in(
                rows.clone(),
                expected.data_type().clone(),
                &mut work,
                reader(source.clone(), Arc::default(), Fault::None),
            )
            .await
            .unwrap()
            .unwrap();
        drop((input.decode_in(&mut work).unwrap().unwrap(), work));
        assert_eq!(
            pool.reserved(),
            0,
            "final consumer release permits next operation"
        );
    }
}

#[rstest]
#[case::checksum(0)]
#[case::short_output(1)]
#[case::long_output(2)]
#[tokio::test]
async fn compressed_replay_failure_releases_unpublished_outputs(#[case] fault: u8) {
    let mut encoder = ::zstd::stream::Encoder::new(Vec::new(), 0).unwrap();
    encoder.include_checksum(true).unwrap();
    encoder.include_contentsize(false).unwrap();
    std::io::Write::write_all(&mut encoder, &[1, 2, 3, 4]).unwrap();
    let mut frame = (match fault {
        1 => 5_u64,
        2 => 3,
        _ => 4,
    })
    .to_le_bytes()
    .to_vec();
    frame.extend_from_slice(&encoder.finish().unwrap());
    if fault == 0 {
        *frame.last_mut().unwrap() ^= 1;
    }
    let prior = ::zstd::bulk::compress(&[17; 4097], 0).unwrap();
    // Raw FullZip framing carries independently constructed codec frames. The
    // second frame passes size planning and fails after an earlier output write.
    let mut f = fixture_with_values(64, false, "flat", vec![prior, frame]);
    Arc::get_mut(&mut f.scheduler.details)
        .unwrap()
        .value_decompressor = PerValueDecompressor::Variable(Arc::new(
        CompressedBufferEncoder::try_new(CompressionConfig::new(CompressionScheme::Zstd, Some(0)))
            .unwrap(),
    ));
    let source = source(&mut f);
    let scheduler = Arc::new(f.scheduler);
    let pool = AllocationBudget::new(1024 * 1024);
    for _ in 0..2 {
        let mut work = pool.try_reserve(pool.capacity()).unwrap();
        let input = scheduler
            .clone()
            .prepare_array_range_in(
                0..2,
                DataType::LargeBinary,
                &mut work,
                reader(source.clone(), Arc::default(), Fault::None),
            )
            .await
            .unwrap()
            .unwrap();
        let error = input.decode_in(&mut work).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(match fault {
            0 => "checksum",
            1 => "initializer produced 4 bytes, expected 5",
            _ => "Destination buffer is too small",
        }));
        drop(work);
        assert_eq!(pool.reserved(), 0);
    }
}

#[rstest]
#[tokio::test]
async fn compressed_frames_validate_without_output_credit(
    #[values(CompressionScheme::Zstd, CompressionScheme::Lz4)] scheme: CompressionScheme,
) {
    let mut f = fixture_with_values(64, false, "flat", vec![vec![1, 2, 3]]);
    Arc::get_mut(&mut f.scheduler.details)
        .unwrap()
        .value_decompressor = PerValueDecompressor::Variable(Arc::new(
        CompressedBufferEncoder::try_new(CompressionConfig::new(scheme, Some(0))).unwrap(),
    ));
    let capacity = f.zipped.len();
    let source = source(&mut f);
    let pool = AllocationBudget::new(capacity);
    let mut work = pool.try_reserve(capacity).unwrap();
    let input = Arc::new(f.scheduler)
        .prepare_array_range_in(
            0..1,
            DataType::LargeBinary,
            &mut work,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), 0);
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("length prefix"));
    assert_eq!(work.len(), capacity);
    drop(work);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn compressed_arrow_rejects_invalid_utf8(
    #[values(CompressionScheme::Zstd, CompressionScheme::Lz4)] scheme: CompressionScheme,
) {
    let f = fixture_with_values(32, false, "flat", vec![vec![0xff], b"valid".to_vec()]);
    let mut f = compress(f, scheme);
    let source = source(&mut f);
    let pool = AllocationBudget::new(1024 * 1024);
    let mut work = pool.try_reserve(pool.capacity()).unwrap();
    let input = Arc::new(f.scheduler)
        .prepare_array_range_in(
            0..2,
            DataType::Utf8,
            &mut work,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(error, Error::Arrow { .. }));
    assert!(error.to_string().to_lowercase().contains("utf"));
    drop(work);
    assert_eq!(pool.reserved(), 0);
}
