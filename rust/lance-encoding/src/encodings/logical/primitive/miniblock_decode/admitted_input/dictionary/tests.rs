// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::EncodingUtils;
use arrow_array::{BinaryArray, LargeBinaryArray};
use futures::FutureExt;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;
use std::sync::Mutex;

#[cfg(feature = "lz4")]
mod compressed;

fn fixture(bits: u8, values: &[Vec<u8>]) -> (Dictionary, Vec<u8>) {
    let width = usize::from(bits / 8);
    let start = (values.len() + 3) * width;
    let mut bytes = (u64::from(bits)).to_le_bytes()[..width].to_vec();
    bytes.extend_from_slice(&(start as u64).to_le_bytes()[..width]);
    let mut offset = 0_u64;
    bytes.extend_from_slice(&offset.to_le_bytes()[..width]);
    for value in values {
        offset += value.len() as u64;
        bytes.extend_from_slice(&offset.to_le_bytes()[..width]);
    }
    for value in values {
        bytes.extend_from_slice(value);
    }
    let dictionary = Dictionary::try_new(
        &EncodingUtils::variable(EncodingUtils::flat(u64::from(bits), None), None),
        0..bytes.len() as u64,
        values.len() as u64,
    )
    .unwrap();
    (dictionary, bytes)
}

fn indices(values: &[u64], bits: u64, work: &mut Reservation) -> FixedWidthDataBlock {
    let width = (bits / 8) as usize;
    let mut output = work.try_allocate(values.len() * width).unwrap().unwrap();
    for (slot, value) in output.as_mut().chunks_exact_mut(width).zip(values) {
        slot.copy_from_slice(&value.to_le_bytes()[..width]);
    }
    FixedWidthDataBlock {
        data: LanceBuffer::from_bytes(output.freeze(), width as u64),
        bits_per_value: bits,
        num_values: values.len() as u64,
        block_info: BlockInfo::new(),
    }
}

fn read(
    bytes: &[u8],
    range: Range<u64>,
    mut output: ReservedBuffer,
) -> std::future::Ready<Result<ReservedBuffer>> {
    output
        .as_mut()
        .copy_from_slice(&bytes[range.start as usize..range.end as usize]);
    std::future::ready(Ok(output))
}

#[rstest]
#[tokio::test]
async fn selected_values_ignore_large_unselected_payload_and_keep_shared_charge(
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
    let (dictionary, bytes) = fixture(bits, &values);
    let budget = AllocationBudget::new(32768);
    let mut work = budget.try_reserve(32768).unwrap();
    let selection = [3, 0, 2, 0];
    let indices = indices(&selection, index_bits, &mut work);
    let requests = Mutex::new(Vec::new());
    let output = dictionary
        .expand(&indices, bits, &mut work, &mut |range: Range<u64>, out| {
            requests.lock().unwrap().push(range.clone());
            read(&bytes, range, out)
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
    let requests = requests.into_inner().unwrap();
    let large_start = (values.len() + 3) * usize::from(bits / 8) + values[0].len();
    let large_end = large_start + unselected;
    assert!(requests.iter().all(|r| r.end - r.start <= 4096));
    assert!(
        requests
            .iter()
            .all(|r| r.end <= large_start as u64 || r.start >= large_end as u64)
    );
    drop((indices, output, work));
    let physical = (selection.len() + 1) * usize::from(bits / 8) + 8193 + 10;
    assert_eq!(budget.reserved(), physical);
    let offsets_alias = array.to_data().buffers()[0].slice_with_length(0, 1);
    let values_alias = array.to_data().buffers()[1].slice_with_length(0, 1);
    drop(array);
    assert_eq!(budget.reserved(), physical);
    drop(offsets_alias);
    assert_eq!(budget.reserved(), physical);
    drop(values_alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::short(31, false)]
#[case::exact(32, true)]
#[tokio::test]
async fn output_ranges_and_io_scratch_share_one_budget(
    #[case] available: usize,
    #[case] fits: bool,
) {
    let (dictionary, bytes) = fixture(32, &[b"abc".to_vec()]);
    let budget = AllocationBudget::new(available);
    let mut work = budget.try_reserve(available).unwrap();
    let indices = indices(&[0], 16, &mut work);
    // Two index bytes +16-byte selected range +8-byte output offsets +3 bytes
    // output +3-byte I/O scratch. The metadata phase has a smaller peak.
    let output = dictionary
        .expand(&indices, 32, &mut work, &mut |r, o| read(&bytes, r, o))
        .await
        .unwrap();
    assert_eq!(output.is_some(), fits);
    drop((output, indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::first(0, 1)]
#[case::backwards(1025, 0)]
#[case::outside(1025, 999999)]
#[tokio::test]
async fn validates_unselected_offset_windows(
    #[case] bad: usize,
    #[case] value: u64,
    #[values(32, 64)] bits: u8,
) {
    let (dictionary, mut bytes) = fixture(bits, &vec![b"v".to_vec(); 1025]);
    let width = usize::from(bits / 8);
    bytes[(2 + bad) * width..(3 + bad) * width].copy_from_slice(&value.to_le_bytes()[..width]);
    let budget = AllocationBudget::new(8192);
    let mut work = budget.try_reserve(8192).unwrap();
    let indices = indices(&[0], 32, &mut work);
    let error = dictionary
        .expand(&indices, bits, &mut work, &mut |r, o| read(&bytes, r, o))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("dictionary offsets"));
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn rejects_invalid_index_before_io_even_with_no_free_credit() {
    let (dictionary, _) = fixture(32, &[b"a".to_vec()]);
    let budget = AllocationBudget::new(1);
    let mut work = budget.try_reserve(1).unwrap();
    let indices = indices(&[1], 8, &mut work);
    let error = dictionary
        .expand(&indices, 32, &mut work, &mut |_, _| {
            std::future::ready(Err(Error::internal("must not read")))
        })
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("index exceeds"));
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn empty_selection_validates_metadata_without_reading_values(#[values(32, 64)] bits: u8) {
    let (dictionary, bytes) = fixture(bits, &[b"unused".to_vec()]);
    let budget = AllocationBudget::new(32);
    let mut work = budget.try_reserve(32).unwrap();
    let indices = indices(&[], 8, &mut work);
    let data_start = (4 * usize::from(bits / 8)) as u64;
    let output = dictionary
        .expand(&indices, bits, &mut work, &mut |range: Range<u64>, out| {
            assert!(range.end <= data_start);
            read(&bytes, range, out)
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.num_values(), 0);
    drop((indices, output, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::width(0, 16)]
#[case::values_start(4, 15)]
#[tokio::test]
async fn corrupt_header_releases_its_scratch(#[case] at: usize, #[case] value: u32) {
    let (dictionary, mut bytes) = fixture(32, &[b"a".to_vec()]);
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    let budget = AllocationBudget::new(64);
    let mut work = budget.try_reserve(64).unwrap();
    let indices = indices(&[0], 8, &mut work);
    let before = work.len();
    let error = dictionary
        .expand(&indices, 32, &mut work, &mut |r, o| read(&bytes, r, o))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("dictionary header"));
    assert_eq!(work.len(), before);
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn valid_expansion_beyond_one_arrow_array_requests_replanning() {
    // Only metadata is materialized. No multi-GiB fixture or payload read occurs.
    let value_len = 1_u32 << 30;
    let dictionary = Dictionary::try_new(
        &EncodingUtils::variable(EncodingUtils::flat(32, None), None),
        0..16 + u64::from(value_len),
        1,
    )
    .unwrap();
    let bytes = [32_u32, 16, 0, value_len]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let budget = AllocationBudget::new(128);
    let mut work = budget.try_reserve(128).unwrap();
    let indices = indices(&[0, 0], 8, &mut work);
    let before = work.len();
    let output = dictionary
        .expand(&indices, 32, &mut work, &mut |range: Range<u64>, out| {
            assert!(range.end <= 16);
            read(&bytes, range, out)
        })
        .await
        .unwrap();
    assert!(output.is_none());
    assert_eq!(work.len(), before);
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn cancellation_keeps_inflight_scratch_and_releases_the_attempt() {
    let (dictionary, bytes) = fixture(32, &[vec![b'x'; 8193]]);
    let budget = AllocationBudget::new(16384);
    let mut work = budget.try_reserve(16384).unwrap();
    let indices = indices(&[0], 8, &mut work);
    let dictionary_values = 16_u64;
    let (sent, received) = tokio::sync::oneshot::channel();
    let mut sent = Some(sent);
    let mut read = |range: Range<u64>, out| {
        if range.start >= dictionary_values {
            let sent = sent.take().unwrap();
            async move {
                let held = out;
                sent.send(()).unwrap();
                std::future::pending::<()>().await;
                Ok(held)
            }
            .boxed()
        } else {
            read(&bytes, range, out).boxed()
        }
    };
    let mut future = Box::pin(dictionary.expand(&indices, 32, &mut work, &mut read));
    tokio::select! { result = &mut future => panic!("unexpected completion: {result:?}"), result = received => result.unwrap() };
    assert_eq!(budget.reserved(), 16384);
    drop(future);
    drop((indices, work));
    assert_eq!(budget.reserved(), 0);
}
