// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::encodings::physical::block::{CompressionConfig, CompressionScheme};
use arrow_array::Array;

fn encoded(bits: u8, count: usize, raw: &[u8]) -> (Dictionary, Vec<u8>) {
    let bytes = ::lz4::block::compress(raw, None, true).unwrap();
    let encoding = EncodingUtils::wrapped(
        CompressionConfig {
            scheme: CompressionScheme::Lz4,
            level: None,
        },
        EncodingUtils::variable(EncodingUtils::flat(u64::from(bits), None), None),
    )
    .unwrap();
    (
        Dictionary::try_new(&encoding, 0..bytes.len() as u64, count as u64).unwrap(),
        bytes,
    )
}

#[rstest]
#[tokio::test]
async fn selected_output_fits_without_materializing_dictionary(
    #[values(32, 64)] bits: u8,
    #[values(8, 16, 32, 64)] index_bits: u64,
    #[values(128 * 1024, 1024 * 1024)] unselected: usize,
) {
    let values = vec![
        b"alpha".to_vec(),
        vec![b'x'; unselected],
        vec![b'y'; 8193],
        Vec::new(),
    ];
    let (_, raw) = fixture(bits, &values);
    let (dictionary, bytes) = encoded(bits, values.len(), &raw);
    const LIMIT: usize = 80 * 1024;
    assert!(raw.len() > LIMIT);
    let budget = AllocationBudget::new(LIMIT);
    let mut work = budget.try_reserve(LIMIT).unwrap();
    let selection = [3, 0, 2, 0];
    let indices = indices(&selection, index_bits, &mut work);
    let requests = Mutex::new(Vec::new());
    let output = dictionary
        .expand(&indices, bits, &mut work, &mut |r: Range<u64>, o| {
            assert!(r.end - r.start <= 4096);
            requests.lock().unwrap().push(r.clone());
            read(&bytes, r, o)
        })
        .await
        .unwrap()
        .unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let array = arrow_array::make_array(output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap());
    let selected = selection.iter().map(|i| values[*i as usize].as_slice());
    let expected: ArrayRef = if bits == 32 {
        Arc::new(BinaryArray::from_iter_values(selected))
    } else {
        Arc::new(LargeBinaryArray::from_iter_values(selected))
    };
    assert_eq!(array.to_data(), expected.to_data());
    // A compressed block must be traversed, including unselected bytes. There
    // are exactly two complete payload passes, plus three checked size reads.
    let requests = requests.into_inner().unwrap();
    assert_eq!(requests.iter().filter(|r| r.start == 4).count(), 2);
    assert_eq!(
        requests
            .iter()
            .filter(|r| r.end == bytes.len() as u64)
            .count(),
        2
    );
    drop((indices, output, work));
    let physical = 5 * usize::from(bits / 8) + 8203;
    assert_eq!(budget.reserved(), physical);
    let alias = array.to_data().buffers()[0].slice_with_length(0, 1);
    drop(array);
    assert_eq!(budget.reserved(), physical);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::short(1, false)]
#[case::exact(0, true)]
#[tokio::test]
async fn output_records_history_and_input_overlap(#[case] missing: usize, #[case] fits: bool) {
    let (_, raw) = fixture(32, &[b"abc".to_vec()]);
    let (dictionary, bytes) = encoded(32, 1, &raw);
    // One u16 index, one four-word record, two u32 output offsets, three
    // output bytes, the complete small history and its compressed input window.
    let capacity = 2 + 32 + 8 + 3 + raw.len() + bytes.len() - 4 - missing;
    let budget = AllocationBudget::new(capacity);
    let mut work = budget.try_reserve(capacity).unwrap();
    let indices = indices(&[0], 16, &mut work);
    let before = work.len();
    let output = dictionary
        .expand(&indices, 32, &mut work, &mut |r, o| read(&bytes, r, o))
        .await
        .unwrap();
    assert_eq!(output.is_some(), fits);
    if !fits {
        assert_eq!(work.len(), before);
    }
    drop((output, indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::no_rows(vec![])]
#[case::empty_values(vec![1, 1])]
#[tokio::test]
async fn empty_output_validates_whole_first_pass(#[case] selection: Vec<u64>) {
    let (_, raw) = fixture(64, &[vec![b'x'; 100000], vec![]]);
    let (dictionary, bytes) = encoded(64, 2, &raw);
    let budget = AllocationBudget::new(72000);
    let mut work = budget.try_reserve(72000).unwrap();
    let indices = indices(&selection, 8, &mut work);
    let mut passes = 0;
    let output = dictionary
        .expand(&indices, 64, &mut work, &mut |r: Range<u64>, o| {
            if r.start == 4 {
                passes += 1;
            }
            read(&bytes, r, o)
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(passes, 1);
    let array = arrow_array::make_array(
        output
            .try_to_arrow_in(&DataType::LargeBinary, &mut work)
            .unwrap()
            .unwrap(),
    );
    assert_eq!(
        array.to_data(),
        LargeBinaryArray::from_iter_values(selection.iter().map(|_| b"".as_slice())).to_data()
    );
    drop((array, output, indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::width(0, 64)]
#[case::value_start(4, 0)]
#[case::first_offset(8, 1)]
#[case::unselected_suffix(20, 1)]
#[case::out_of_bounds(20, 100)]
#[tokio::test]
async fn corrupt_unselected_offsets_fail(#[case] at: usize, #[case] value: u32) {
    let (_, mut raw) = fixture(32, &[b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec()]);
    raw[at..at + 4].copy_from_slice(&value.to_le_bytes());
    let (dictionary, bytes) = encoded(32, 3, &raw);
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    let indices = indices(&[0], 8, &mut work);
    let error = dictionary
        .expand(&indices, 32, &mut work, &mut |r, o| read(&bytes, r, o))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn second_pass_offset_change_is_rejected() {
    let (_, raw) = fixture(32, &[b"abc".to_vec(), b"def".to_vec()]);
    let (mut dictionary, _) = encoded(32, 2, &raw);
    // Literal-only frames keep both read domains identical, independent of the
    // native compressor's choice of matches for changed metadata.
    let literal = |raw: &[u8]| {
        let mut bytes = (raw.len() as u32).to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0xf0, (raw.len() - 15) as u8]);
        bytes.extend_from_slice(raw);
        bytes
    };
    let bytes = literal(&raw);
    dictionary.bytes = 0..bytes.len() as u64;
    let mut changed_raw = raw.clone();
    changed_raw[12..16].copy_from_slice(&2_u32.to_le_bytes());
    let changed = literal(&changed_raw);
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    let indices = indices(&[0], 8, &mut work);
    let mut pass = 0;
    let error = dictionary
        .expand(&indices, 32, &mut work, &mut |r: Range<u64>, o| {
            if r.start == 4 {
                pass += 1;
            }
            read(if pass < 2 { &bytes } else { &changed }, r, o)
        })
        .await
        .unwrap_err();
    assert!(error.to_string().contains("offsets changed"));
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn cancellation_during_second_pass_releases_output_and_scratch() {
    let (_, raw) = fixture(32, &[vec![b'x'; 100000], b"yes".to_vec()]);
    let (dictionary, bytes) = encoded(32, 2, &raw);
    let budget = AllocationBudget::new(72000);
    let mut work = budget.try_reserve(72000).unwrap();
    let indices = indices(&[1], 8, &mut work);
    let (sent, received) = tokio::sync::oneshot::channel();
    let mut sent = Some(sent);
    let mut pass = 0;
    let mut reader = |r: Range<u64>, o| {
        if r.start == 4 {
            pass += 1;
        }
        if pass == 2 {
            let sent = sent.take().unwrap();
            async move {
                let held = o;
                sent.send(()).unwrap();
                std::future::pending::<()>().await;
                Ok(held)
            }
            .boxed()
        } else {
            read(&bytes, r, o).boxed()
        }
    };
    let mut future = Box::pin(dictionary.expand(&indices, 32, &mut work, &mut reader));
    tokio::select! { r = &mut future => panic!("unexpected completion {r:?}"), r = received => r.unwrap() }
    assert_eq!(budget.reserved(), 72000);
    drop(future);
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn other_general_codecs_remain_explicitly_unsupported() {
    let encoding = EncodingUtils::wrapped(
        CompressionConfig {
            scheme: CompressionScheme::Zstd,
            level: None,
        },
        EncodingUtils::variable(EncodingUtils::flat(32, None), None),
    )
    .unwrap();
    let error = Dictionary::try_new(&encoding, 0..100, 2).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("raw or LZ4"));
}

#[rstest]
#[tokio::test]
async fn offset_table_spans_multiple_windows(#[values(32, 64)] bits: u8) {
    let values = (0..2053)
        .map(|i| vec![(i % 251) as u8; i % 17])
        .collect::<Vec<_>>();
    let (_, raw) = fixture(bits, &values);
    let (dictionary, bytes) = encoded(bits, values.len(), &raw);
    let budget = AllocationBudget::new(72000);
    let mut work = budget.try_reserve(72000).unwrap();
    let selection = [2052, 1021, 0, 512, 2052];
    let indices = indices(&selection, 16, &mut work);
    let output = dictionary
        .expand(&indices, 64, &mut work, &mut |r, o| read(&bytes, r, o))
        .await
        .unwrap()
        .unwrap();
    let array = arrow_array::make_array(
        output
            .try_to_arrow_in(&DataType::LargeBinary, &mut work)
            .unwrap()
            .unwrap(),
    );
    let expected = LargeBinaryArray::from_iter_values(
        selection.iter().map(|i| values[*i as usize].as_slice()),
    );
    assert_eq!(array.to_data(), expected.to_data());
    drop((indices, output, array, work));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn malformed_compressed_suffix_is_not_hidden_by_selection() {
    let (_, raw) = fixture(32, &[b"yes".to_vec(), vec![b'x'; 100000]]);
    let (mut dictionary, mut bytes) = encoded(32, 2, &raw);
    bytes.push(0); // Turns the final literals into a truncated nonfinal sequence.
    dictionary.bytes.end += 1;
    let budget = AllocationBudget::new(72000);
    let mut work = budget.try_reserve(72000).unwrap();
    let indices = indices(&[0], 8, &mut work);
    let error = dictionary
        .expand(&indices, 32, &mut work, &mut |r, o| read(&bytes, r, o))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::no_prefix_credit(3)]
#[case::no_history_credit(100)]
#[tokio::test]
async fn insufficient_scratch_refuses_before_payload_reads(#[case] available: usize) {
    let (_, raw) = fixture(32, &[vec![b'x'; 100000]]);
    let (dictionary, bytes) = encoded(32, 1, &raw);
    let budget = AllocationBudget::new(available + 1);
    let mut work = budget.try_reserve(available + 1).unwrap();
    let indices = indices(&[0], 8, &mut work);
    let mut reads = 0;
    let output = dictionary
        .expand(&indices, 32, &mut work, &mut |r: Range<u64>, o| {
            assert_eq!(r, 0..4);
            reads += 1;
            read(&bytes, r, o)
        })
        .await
        .unwrap();
    assert!(output.is_none());
    assert_eq!(reads, usize::from(available >= 4));
    assert_eq!(work.len(), available);
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}
