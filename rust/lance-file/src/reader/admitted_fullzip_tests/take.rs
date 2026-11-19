// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::scan::{GateIo, RangeIo};
use super::*;
use crate::reader::PreparedFullZipTake;
use arrow_array::{UInt32Array, cast::AsArray};
use lance_encoding::constants::COMPRESSION_META_KEY;
use lance_encoding::format::metadata::{CompressionScheme, compressive_encoding::Compression};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn small_raw_take_keeps_nested_values_and_alias_credit(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(0, 2)] depth: usize,
) {
    let expected = array(0, depth);
    let (_fs, reader) = fixture_arrays(version, false, vec![expected.clone()]).await;
    let indices = (0..expected.len() as u32).collect::<Vec<_>>();
    let budget = AllocationBudget::new(65536);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    assert!(
        input
            .decode_work_estimate()
            .is_some_and(|work| work <= 256 * 1024)
    );
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.column(0).to_data(), expected.to_data());
    let alias = output.column(0).slice(0, 1);
    drop((output, work));
    let held = budget.reserved();
    assert!(held > 0);
    let peer = budget.try_reserve(65536 - held).unwrap();
    assert!(budget.try_reserve(1).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), peer.len());
    drop(peer);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn null_placeholder_without_columns_preserves_row_count() {
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let output = PreparedFullZipTake::all_null(Arc::new(Schema::empty()), 7)
        .decode_in(&mut work)
        .unwrap()
        .unwrap();
    assert_eq!(output.num_rows(), 7);
    assert_eq!(output.num_columns(), 0);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn null_placeholders_release_prior_columns_on_refusal() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bytes", DataType::LargeBinary, true),
        Field::new("number", DataType::UInt64, true),
    ]));
    let budget = AllocationBudget::new(128);
    // Three rows: byte offsets/validity = 33, u64 values/validity = 25.
    let mut short = budget.try_reserve(57).unwrap();
    assert!(
        PreparedFullZipTake::all_null(schema.clone(), 3)
            .decode_in(&mut short)
            .unwrap()
            .is_none()
    );
    drop(short);
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(58).unwrap();
    let output = PreparedFullZipTake::all_null(schema.clone(), 3)
        .decode_in(&mut work)
        .unwrap()
        .unwrap();
    let expected = RecordBatch::try_new(
        schema.clone(),
        schema
            .fields()
            .iter()
            .map(|field| arrow_array::new_null_array(field.data_type(), 3))
            .collect(),
    )
    .unwrap();
    assert_eq!(output, expected);
    drop(work);
    assert_eq!(budget.reserved(), 58);
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::unsupported(false)]
#[case::required(true)]
fn invalid_null_placeholders_release_prior_columns(#[case] required: bool) {
    let field = if required {
        Field::new("bad", DataType::UInt64, false)
    } else {
        Field::new(
            "bad",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("bytes", DataType::LargeBinary, true),
        field,
    ]));
    let budget = AllocationBudget::new(128);
    let mut work = budget.try_reserve(128).unwrap();
    let error = PreparedFullZipTake::all_null(schema, 3)
        .decode_in(&mut work)
        .unwrap_err();
    if required {
        assert!(matches!(error, Error::Arrow { .. }));
        assert!(error.to_string().contains("non-nullable"));
    } else {
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("no admitted null"));
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn sparse_take_matches_arrow_and_skips_pages(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(0, 2)] depth: usize,
) {
    let pages = [array(0, depth), array(1, depth), array(2, depth)];
    let n = pages[0].len() as u32;
    let indices = [0, 1, n - 1, 2 * n, 3 * n - 1];
    let expected = arrow_select::take::take(
        arrow_select::concat::concat(&pages.iter().map(AsRef::as_ref).collect::<Vec<_>>())
            .unwrap()
            .as_ref(),
        &UInt32Array::from(indices.to_vec()),
        None,
    )
    .unwrap();
    let (_fs, reader) = fixture_arrays(version, lite, pages.to_vec()).await;
    assert_eq!(reader.metadata().column_infos[0].page_infos.len(), 3);
    let skipped = reader.metadata().column_infos[0].page_infos[1]
        .buffer_offsets_and_sizes
        .clone();
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(65536);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        input.decode_work_estimate(),
        None,
        "sparse runs need concat"
    );
    assert!(work.len() < 65536, "prepared input owns physical credit");
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.column(0).to_data(), expected.to_data());
    assert!(!reads.lock().unwrap().is_empty());
    for read in reads.lock().unwrap().iter() {
        for (offset, len) in skipped.iter() {
            assert!(
                read.end <= *offset || read.start >= offset + len,
                "read {read:?} touched a skipped page"
            );
        }
    }
    let alias = output.column(0).to_data().buffers()[0].slice_with_length(0, 1);
    drop((output, work));
    assert!(budget.reserved() > 1);
    assert!(budget.try_reserve(65536).is_none());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    // An independent next operation can acquire the same finite workspace.
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&indices, &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(output.column(0).to_data(), expected.to_data());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn sparse_take_preflights_all_selected_pages(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
) {
    let (_fs, reader) = fixture_arrays(version, false, vec![array(0, 0), array(1, 0)]).await;
    let reader = with_unsupported_page(reader, 0, 1);
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    for invalid in [vec![1, 1], vec![2, 0], vec![16]] {
        let error = reader
            .prepare_admitted_fullzip_take_in(&invalid, &mut work)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("FullZip"));
    }
    let error = reader
        .prepare_admitted_fullzip_take_in(&[0, 8], &mut work)
        .await
        .err()
        .unwrap();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("repetition indexes"));
    assert!(
        reads.lock().unwrap().is_empty(),
        "later selected layout must fail before earlier reads"
    );
    assert_eq!(work.len(), 8192);
    let input = reader
        .prepare_admitted_fullzip_take_in(&[0], &mut work)
        .await
        .unwrap()
        .unwrap();
    assert!(input.decode_in(&mut work).unwrap().is_some());
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn sparse_take_empty_and_refused_attempts_release(#[values(0, 2)] depth: usize) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_3, false, depth, 2).await;
    let reads = Arc::new(Mutex::new(Vec::new()));
    let reader = reader.with_scheduler(Arc::new(RangeIo {
        inner: reader.core.scheduler.clone(),
        reads: reads.clone(),
    }));
    let budget = AllocationBudget::new(65536);
    let mut zero = budget.try_reserve(0).unwrap();
    assert!(
        reader
            .prepare_admitted_fullzip_take_in(&[0], &mut zero)
            .await
            .unwrap()
            .is_none()
    );
    let input = reader
        .prepare_admitted_fullzip_take_in(&[], &mut zero)
        .await
        .unwrap()
        .unwrap();
    assert!(
        input.decode_in(&mut zero).unwrap().is_none(),
        "empty typed offsets are not free"
    );
    assert!(reads.lock().unwrap().is_empty());
    drop(zero);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[], &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(
        output.column(0).to_data(),
        array(0, depth).slice(0, 0).to_data()
    );
    assert!(work.len() < 65536);
    assert!(reads.lock().unwrap().is_empty());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(65536).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[0, 2], &mut work)
        .await
        .unwrap()
        .unwrap();
    let held = work.try_split(work.len()).unwrap();
    assert!(input.decode_in(&mut work).unwrap().is_none());
    drop((held, work));
    assert_eq!(
        budget.reserved(),
        0,
        "failed decode/concat retains no partial output"
    );
}

#[rstest]
#[tokio::test]
async fn sparse_take_cancel_error_and_retry(#[values(false, true)] lite: bool) {
    let (_fs, reader) = fixture(ConcreteFileVersion::V2_2, lite, 0, 2).await;
    let block = Arc::new(AtomicBool::new(true));
    let fail = Arc::new(AtomicBool::new(false));
    let entered = Arc::new(tokio::sync::Notify::new());
    let reader = reader.with_scheduler(Arc::new(GateIo {
        inner: reader.core.scheduler.clone(),
        block: block.clone(),
        fail: fail.clone(),
        entered: entered.clone(),
    }));
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    {
        let future = reader.prepare_admitted_fullzip_take_in(&[0, 8], &mut work);
        tokio::pin!(future);
        tokio::select! {
            _ = entered.notified() => (),
            _ = &mut future => panic!("gated input must stay pending"),
        }
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
    block.store(false, Ordering::Relaxed);
    fail.store(true, Ordering::Relaxed);
    let mut work = budget.try_reserve(8192).unwrap();
    let error = reader
        .prepare_admitted_fullzip_take_in(&[0, 8], &mut work)
        .await
        .err()
        .unwrap();
    assert!(matches!(error, Error::IO { .. }));
    assert!(
        error
            .to_string()
            .contains("injected admitted scan read failure")
    );
    drop(work);
    assert_eq!(budget.reserved(), 0);
    fail.store(false, Ordering::Relaxed);
    let mut work = budget.try_reserve(8192).unwrap();
    let input = reader
        .prepare_admitted_fullzip_take_in(&[0, 8], &mut work)
        .await
        .unwrap()
        .unwrap();
    let output = input.decode_in(&mut work).unwrap().unwrap();
    let values = output.column(0).as_binary::<i64>();
    assert_eq!(values.value(0), array(0, 0).as_binary::<i64>().value(0));
    assert_eq!(values.value(1), array(1, 0).as_binary::<i64>().value(0));
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test(flavor = "current_thread")]
async fn compressed_cross_page_take_reuses_unzip_workspace(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[values(false, true)] nested: bool,
    #[values("zstd", "lz4")] compression: &str,
) {
    let metadata = HashMap::from([
        (
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            STRUCTURAL_ENCODING_FULLZIP.to_string(),
        ),
        (COMPRESSION_META_KEY.to_string(), compression.to_string()),
    ]);
    // Independent deterministic bytes keep compressed input comparable to output.
    // Repeated strings would conceal the intermediate ownership cost.
    let mut seed = 17_u64;
    let mut payload_bytes = 0;
    let arrays = (0..2)
        .map(|_| {
            let values = (0..8)
                .map(|row| {
                    if row == 1 {
                        return None;
                    }
                    let bytes = if row == 2 {
                        Vec::new()
                    } else {
                        (0..65536)
                            .map(|_| {
                                seed ^= seed << 13;
                                seed ^= seed >> 7;
                                seed ^= seed << 17;
                                seed as u8
                            })
                            .collect::<Vec<_>>()
                    };
                    payload_bytes += bytes.len();
                    Some(bytes)
                })
                .collect::<Vec<_>>();
            let leaf: ArrayRef = Arc::new(LargeBinaryArray::from_iter(
                values.iter().map(|value| value.as_deref()),
            ));
            if nested {
                Arc::new(ListArray::new(
                    Arc::new(
                        Field::new("item", DataType::LargeBinary, true)
                            .with_metadata(metadata.clone()),
                    ),
                    OffsetBuffer::new(vec![0_i32, 2, 2, 2, 5, 8].into()),
                    leaf,
                    Some(NullBuffer::from(vec![true, true, false, true, true])),
                )) as ArrayRef
            } else {
                leaf
            }
        })
        .collect::<Vec<_>>();
    let expected =
        arrow_select::concat::concat(&arrays.iter().map(AsRef::as_ref).collect::<Vec<_>>())
            .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", expected.data_type().clone(), true).with_metadata(metadata),
    ]));
    let batches = arrays
        .into_iter()
        .map(|array| RecordBatch::try_new(schema.clone(), vec![array]).unwrap())
        .collect();
    let (_fs, reader) = fixture_batches(version, lite, batches, 1).await;
    let pages = &reader.metadata().column_infos[0].page_infos;
    assert_eq!(pages.len(), 2);
    for page in pages.iter() {
        let layout = match &page.encoding {
            PageEncoding::Structural(layout) => layout.clone(),
            PageEncoding::Planar(layout) => layout.dense_execution_layout().unwrap(),
            _ => panic!("dense writer fixture"),
        };
        let Some(Layout::FullZipLayout(layout)) = layout.layout else {
            panic!("stored FullZip required");
        };
        let Some(Compression::General(general)) = layout
            .value_compression
            .as_ref()
            .and_then(|encoding| encoding.compression.as_ref())
        else {
            panic!("stored general compression required");
        };
        assert_eq!(
            general.compression.as_ref().unwrap().scheme(),
            if compression == "zstd" {
                CompressionScheme::CompressionAlgorithmZstd
            } else {
                CompressionScheme::CompressionAlgorithmLz4
            }
        );
    }
    let indices = (0..expected.len() as u32).collect::<Vec<_>>();
    // Two copies of the final payload fit (decoded chunks + concat), with room
    // for Zstd context and structural metadata. A third full payload does not.
    let limit = payload_bytes * 2 + 128 * 1024;
    let budget = AllocationBudget::new(limit);
    for _ in 0..2 {
        let mut work = budget.try_reserve(limit).unwrap();
        let input = reader
            .prepare_admitted_fullzip_take_in(&indices, &mut work)
            .await
            .unwrap()
            .unwrap();
        let output = input
            .decode_in(&mut work)
            .unwrap()
            .expect("cross-page output must fit");
        assert_eq!(output.column(0).to_data(), expected.to_data());
        let alias = output.column(0).slice(0, 1);
        drop((output, work));
        let retained = budget.reserved();
        assert!(retained > 0 && retained < limit);
        let peer = budget.try_reserve(limit - retained).unwrap();
        assert!(budget.try_reserve(1).is_none());
        drop(alias);
        assert_eq!(budget.reserved(), peer.len());
        drop(peer);
        assert_eq!(budget.reserved(), 0);
    }
}
