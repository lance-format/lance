// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use futures::FutureExt;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

fn read(
    bytes: &[u8],
    range: Range<u64>,
    mut out: ReservedBuffer,
) -> std::future::Ready<Result<ReservedBuffer>> {
    assert!(range.end - range.start <= WINDOW as u64);
    out.as_mut()
        .copy_from_slice(&bytes[range.start as usize..range.end as usize]);
    std::future::ready(Ok(out))
}

async fn decode(bytes: &[u8]) -> Result<Vec<u8>> {
    let budget = AllocationBudget::new(HISTORY + WINDOW + 4);
    let mut work = budget.try_reserve(HISTORY + WINDOW + 4).unwrap();
    let mut output = Vec::new();
    let range = 0..bytes.len() as u64;
    let result = async {
        let plan = Plan::read(&range, &mut work, &mut |r, o| read(bytes, r, o))
            .await?
            .unwrap();
        plan.visit(
            &range,
            &mut work,
            &mut |r, o| read(bytes, r, o),
            |at, chunk| {
                assert_eq!(at, output.len());
                assert!(chunk.len() <= WINDOW);
                output.extend_from_slice(chunk);
                Ok(())
            },
        )
        .await
    }
    .await;
    if result.is_ok() {
        assert_eq!(work.len(), HISTORY + WINDOW + 4);
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
    result.map(|()| output)
}

#[rstest]
#[tokio::test]
async fn native_writer_roundtrip(
    #[values(0, 1, 5, 4095, 4096, 4097, 65535, 65536, 131077)] len: usize,
    #[values(false, true)] repeated: bool,
) {
    let mut state = 0x12345678_u32;
    let expected = (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if repeated { b'x' } else { state as u8 }
        })
        .collect::<Vec<_>>();
    let bytes = ::lz4::block::compress(&expected, None, true).unwrap();
    assert_eq!(::lz4::block::decompress(&bytes, None).unwrap(), expected);
    assert_eq!(decode(&bytes).await.unwrap(), expected);
}

fn extension(mut extra: usize, bytes: &mut Vec<u8>) {
    while extra >= 255 {
        bytes.push(255);
        extra -= 255;
    }
    bytes.push(extra as u8);
}

#[rstest]
#[case::overlap(1, 131073)]
#[case::two_byte_period(2, 131073)]
#[case::unaligned_period(3, 131073)]
#[case::dictionary_period(32, 131073)]
#[case::before_window(4095, 131073)]
#[case::whole_window(4096, 131073)]
#[case::after_window(4097, 131073)]
#[case::wrapped_seed(65534, 131073)]
#[case::farthest(65535, 8193)]
#[case::short_wrapped_match(65535, 19)]
#[tokio::test]
async fn hand_encoded_matches_cross_history_and_output_boundaries(
    #[case] offset: usize,
    #[case] length: usize,
) {
    let mut expected = (0..offset).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    let mut payload = vec![((offset.min(15) as u8) << 4) | 15];
    if offset >= 15 {
        extension(offset - 15, &mut payload);
    }
    payload.extend_from_slice(&expected);
    payload.extend_from_slice(&(offset as u16).to_le_bytes());
    extension(length - 19, &mut payload);
    for _ in 0..length {
        expected.push(expected[expected.len() - offset]);
    }
    // The low final token bits are unused, not a second match.
    payload.push(0x5f);
    payload.extend_from_slice(b"final");
    expected.extend_from_slice(b"final");
    let mut bytes = (expected.len() as u32).to_le_bytes().to_vec();
    bytes.extend_from_slice(&payload);
    assert_eq!(::lz4::block::decompress(&bytes, None).unwrap(), expected);
    assert_eq!(decode(&bytes).await.unwrap(), expected);
}

#[rstest]
#[case::zero_offset(13, vec![0x13, b'x', 0, 0, 0x50, 1, 2, 3, 4, 5])]
#[case::before_history(13, vec![0x13, b'x', 2, 0, 0x50, 1, 2, 3, 4, 5])]
#[case::literal_length(20, vec![0xf0, 255])]
#[case::literal_payload(5, vec![0x50, 1])]
#[case::match_length(30, vec![0x1f, b'x', 1, 0, 255])]
#[case::no_final_literals(8, vec![0x13, b'x', 1, 0])]
#[case::too_few_final_literals(13, vec![0x17, b'x', 1, 0, 0x10, b'y'])]
#[case::late_match(10, vec![0x10, b'x', 1, 0, 0x50, 1, 2, 3, 4, 5])]
#[case::size_mismatch(7, vec![0x50, 1, 2, 3, 4, 5])]
#[tokio::test]
async fn malformed_sequences_are_rejected(#[case] size: u32, #[case] payload: Vec<u8>) {
    let mut bytes = size.to_le_bytes().to_vec();
    bytes.extend(payload);
    assert!(matches!(
        decode(&bytes).await,
        Err(Error::CorruptFile { .. })
    ));
}

#[tokio::test]
async fn accepted_mutations_agree_with_native_decoder() {
    let expected = (0..256).map(|i| (i % 31) as u8).collect::<Vec<_>>();
    let original = ::lz4::block::compress(&expected, None, true).unwrap();
    for at in 4..original.len() {
        for bit in [1, 16, 128] {
            let mut bytes = original.clone();
            bytes[at] ^= bit;
            if let Ok(decoded) = decode(&bytes).await {
                assert_eq!(::lz4::block::decompress(&bytes, None).unwrap(), decoded);
            }
        }
    }
}

#[tokio::test]
async fn cancellation_releases_input_and_history() {
    let bytes = ::lz4::block::compress(&vec![7; HISTORY * 2], None, true).unwrap();
    let budget = AllocationBudget::new(HISTORY + WINDOW + 4);
    let mut work = budget.try_reserve(HISTORY + WINDOW + 4).unwrap();
    let range = 0..bytes.len() as u64;
    let plan = Plan::read(&range, &mut work, &mut |r, o| read(&bytes, r, o))
        .await
        .unwrap()
        .unwrap();
    let mut reader = |r: Range<u64>, out: ReservedBuffer| {
        let bytes = &bytes;
        async move {
            if r.start >= 4 {
                std::future::pending::<()>().await;
            }
            read(bytes, r, out).await
        }
    };
    assert!(
        plan.visit(&range, &mut work, &mut reader, |_, _| Ok(()))
            .now_or_never()
            .is_none()
    );
    assert!(work.len() < HISTORY + WINDOW + 4);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
