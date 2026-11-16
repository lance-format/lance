// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::EncodingUtils;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

#[cfg(feature = "bitpacking")]
#[rstest]
#[case::raw(19, false)]
#[case::padded(19, true)]
#[case::tie(80, false)]
#[case::raw_larger_than_chunk(1000, false)]
#[tokio::test]
async fn packed_tail_windows_match_complete_sequence(
    #[case] tail: usize,
    #[case] padded: bool,
    #[values(false, true)] delta: bool,
    #[values(false, true)] incremental: bool,
) {
    use crate::{
        buffer::LanceBuffer,
        compression::BlockCompressor,
        data::{BlockInfo, DataBlock, FixedWidthDataBlock},
        encodings::physical::bitpacking::OutOfLineBitpacking,
    };
    let values = (0..3072)
        .map(|i| 1 + i as u64 * 17 % 31)
        .collect::<Vec<_>>();
    let codec = OutOfLineBitpacking::new(5, 64);
    let (bytes, encoding) = codec
        .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: 64,
            num_values: 3072,
            data: LanceBuffer::reinterpret_vec(values.clone()),
            block_info: BlockInfo::default(),
        }))
        .unwrap();
    let mut bytes = bytes.unwrap().to_vec();
    if !padded {
        bytes.truncate(1280);
        for value in &values[2048..2048 + tail] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let mut expected = values[..2048 + tail].to_vec();
    let encoding = if delta {
        let mut sum = 7_u64;
        expected = std::iter::once(sum)
            .chain(expected.into_iter().map(|v| {
                sum += v;
                sum
            }))
            .collect();
        EncodingUtils::delta(64, 7, encoding)
    } else {
        encoding
    };
    let plan = WindowSequence::try_new(
        &encoding,
        expected.len() as u64,
        11..11 + bytes.len() as u64,
    )
    .unwrap();
    let budget = AllocationBudget::new(plan.scratch_bytes());
    let mut work = budget.try_reserve(plan.scratch_bytes()).unwrap();
    let mut output = Vec::new();
    let mut read = |range: Range<u64>, mut buffer: ReservedBuffer| {
        buffer
            .as_mut()
            .copy_from_slice(&bytes[(range.start - 11) as usize..(range.end - 11) as usize]);
        std::future::ready(Ok(buffer))
    };
    if incremental {
        let mut cursor = WindowCursor::new(&plan);
        while let Some(value) = cursor.next(&mut work, &mut read).await.unwrap() {
            output.push(value);
        }
        assert_eq!(cursor.next(&mut work, &mut read).await.unwrap(), None);
    } else {
        plan.visit(&mut work, &mut read, &mut |value| {
            output.push(value);
            Ok(())
        })
        .await
        .unwrap()
        .unwrap();
    }
    assert_eq!(output, expected);
    assert_eq!(work.len(), plan.scratch_bytes());
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::truncated(15)]
#[case::trailing(17)]
fn flat_framing_rejected_before_input(#[case] size: u64) {
    let error =
        WindowSequence::try_new(&EncodingUtils::flat(64, None), 2, 100..100 + size).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("length"));
}

#[rstest]
#[tokio::test]
async fn delta_overflow_and_short_read_do_not_escape_credit(
    #[values(false, true)] incremental: bool,
) {
    for short in [false, true] {
        let plan = WindowSequence::try_new(
            &EncodingUtils::delta(64, u64::MAX, EncodingUtils::flat(64, None)),
            2,
            0..8,
        )
        .unwrap();
        let budget = AllocationBudget::new(8);
        let mut work = budget.try_reserve(8).unwrap();
        let mut read = |_: Range<u64>, mut buffer: ReservedBuffer| {
            buffer.as_mut().copy_from_slice(&1_u64.to_le_bytes());
            if short {
                drop(buffer);
                std::future::ready(Ok(AllocationBudget::new(0)
                    .try_reserve(0)
                    .unwrap()
                    .allocate()
                    .unwrap()))
            } else {
                std::future::ready(Ok(buffer))
            }
        };
        let error = if incremental {
            let mut cursor = WindowCursor::new(&plan);
            assert_eq!(
                cursor.next(&mut work, &mut read).await.unwrap(),
                Some(u64::MAX)
            );
            cursor.next(&mut work, &mut read).await.unwrap_err()
        } else {
            plan.visit(&mut work, &mut read, &mut |_| Ok(()))
                .await
                .unwrap_err()
        };
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error
                .to_string()
                .contains(if short { "short" } else { "overflows" })
        );
        drop(work);
        assert_eq!(budget.reserved(), 0);
    }
}

#[rstest]
#[case::range(false, vec![1, 3, 5])]
#[case::delta_range(true, vec![7, 8, 11, 16])]
#[tokio::test]
async fn metadata_range_needs_no_input_or_scratch(
    #[case] delta: bool,
    #[case] expected: Vec<u64>,
    #[values(false, true)] incremental: bool,
) {
    let range = EncodingUtils::range(64, 1, 2);
    let encoding = if delta {
        EncodingUtils::delta(64, 7, range)
    } else {
        range
    };
    let decoder = SequenceDecoder::try_new(&encoding, 64, expected.len() as u64).unwrap();
    assert!(!decoder.requires_payload());
    if delta {
        assert!(decoder.metadata().is_none());
    }
    let plan = WindowSequence::try_new(&encoding, expected.len() as u64, 0..0).unwrap();
    assert_eq!(plan.scratch_bytes(), 0);
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let mut output = Vec::new();
    if incremental {
        let mut cursor = WindowCursor::new(&plan);
        while let Some(value) = cursor
            .next(&mut work, &mut |_, _: ReservedBuffer| {
                std::future::ready(Err(Error::internal("metadata must not read")))
            })
            .await
            .unwrap()
        {
            output.push(value);
        }
    } else {
        plan.visit(
            &mut work,
            &mut |_, _: ReservedBuffer| {
                std::future::ready(Err(Error::internal("metadata Range must not read")))
            },
            &mut |value| {
                output.push(value);
                Ok(())
            },
        )
        .await
        .unwrap()
        .unwrap();
    }
    assert_eq!(output, expected);
    assert_eq!(budget.reserved(), 0);
    let error = WindowSequence::try_new(&encoding, expected.len() as u64, 0..1).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("unexpected payload"));
}

#[tokio::test]
async fn metadata_delta_overflow_is_not_wrapped() {
    let encoding = EncodingUtils::delta(64, u64::MAX - 1, EncodingUtils::range(64, 1, 1));
    let plan = WindowSequence::try_new(&encoding, 3, 0..0).unwrap();
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let mut observed = Vec::new();
    let error = plan
        .visit(
            &mut work,
            &mut |_, _: ReservedBuffer| {
                std::future::ready(Err(Error::internal("metadata Delta must not read")))
            },
            &mut |value| {
                observed.push(value);
                Ok(())
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("prefix sum overflows"));
    assert_eq!(observed, vec![u64::MAX - 1, u64::MAX]);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn metadata_traversal_allows_competing_preparation_to_progress() {
    let plan = WindowSequence::try_new(&EncodingUtils::range(64, 0, 1), 4096, 0..0).unwrap();
    let budget = AllocationBudget::new(0);
    let mut work = budget.try_reserve(0).unwrap();
    let progressed = std::cell::Cell::new(false);
    let scan = async {
        let mut count = 0;
        plan.visit(
            &mut work,
            &mut |_, _: ReservedBuffer| std::future::ready(Err(Error::internal("unexpected read"))),
            &mut |value| {
                assert_eq!(value, count);
                count += 1;
                if count == 4096 {
                    assert!(progressed.get());
                }
                Ok(())
            },
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(count, 4096);
    };
    tokio::join!(biased; scan, async { progressed.set(true) });
    assert_eq!(budget.reserved(), 0);
}
