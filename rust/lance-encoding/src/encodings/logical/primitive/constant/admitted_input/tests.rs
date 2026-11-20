// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, BinaryArray, LargeBinaryArray, LargeStringArray, ListArray, StringArray};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::Field;
use futures::FutureExt;
use rstest::rstest;

use super::*;
use crate::{
    compression::DefaultDecompressionStrategy, decoder::PageEncoding,
    encodings::logical::primitive::AdmittedBytePageScheduler, repdef::RepDefBuilder,
};
use lance_core::utils::allocation::AllocationBudget;

fn fixture(
    scalar: &ArrayRef,
    rows: u64,
    rep: &[u16],
    def: &[u16],
    meanings: &[DefinitionInterpretation],
) -> (PageInfo, Vec<u8>) {
    let mut data = lance_arrow::scalar::encode_scalar_value_buffer(scalar).unwrap();
    let mut buffers = vec![(0, data.len() as u64)];
    if !rep.is_empty() || !def.is_empty() {
        for levels in [rep, def] {
            let start = data.len() as u64;
            for level in levels {
                data.extend_from_slice(&level.to_le_bytes());
            }
            buffers.push((start, data.len() as u64 - start));
        }
    }
    (
        PageInfo {
            num_rows: rows,
            priority: 0,
            encoding: PageEncoding::Structural(EncodingUtils::constant_layout(meanings, None)),
            buffer_offsets_and_sizes: buffers.into(),
        },
        data,
    )
}

fn read(
    data: &[u8],
    range: Range<u64>,
    mut output: ReservedBuffer,
) -> impl Future<Output = Result<ReservedBuffer>> {
    output
        .as_mut()
        .copy_from_slice(&data[range.start as usize..range.end as usize]);
    std::future::ready(Ok(output))
}

#[rstest]
#[case::binary(0)]
#[case::large_binary(1)]
#[case::utf8(2)]
#[case::large_utf8(3)]
#[tokio::test]
async fn reads_byte_scalar_and_selected_nullable_rows(#[case] kind: u8) {
    let source: ArrayRef = match kind {
        0 => Arc::new(BinaryArray::from(vec![
            Some(b"hello".as_slice()),
            None,
            Some(b"hello".as_slice()),
        ])),
        1 => Arc::new(LargeBinaryArray::from(vec![
            Some(b"hello".as_slice()),
            None,
            Some(b"hello".as_slice()),
        ])),
        2 => Arc::new(StringArray::from(vec![Some("hello"), None, Some("hello")])),
        _ => Arc::new(LargeStringArray::from(vec![
            Some("hello"),
            None,
            Some("hello"),
        ])),
    };
    let scalar = arrow_select::take::take(
        source.as_ref(),
        &arrow_array::UInt32Array::from(vec![0]),
        None,
    )
    .unwrap();
    let (page, bytes) = fixture(
        &scalar,
        3,
        &[],
        &[0, 1, 0],
        &[DefinitionInterpretation::NullableItem],
    );
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy {}).unwrap();
    for rows in [0..3, 1..2, 2..3] {
        let budget = AllocationBudget::new(1024);
        let mut work = budget.try_reserve(1024).unwrap();
        let input = scheduler
            .prepare_array_range_in(
                rows.clone(),
                source.data_type().clone(),
                &mut work,
                |range, output| read(&bytes, range, output),
            )
            .await
            .unwrap()
            .unwrap();
        let output = input.decode_in(&mut work).unwrap().unwrap();
        assert_eq!(
            output.to_data(),
            source
                .slice(rows.start as usize, (rows.end - rows.start) as usize)
                .to_data()
        );
        drop(work);
        assert!(budget.reserved() > 0);
        let alias = output.to_data().buffers()[0].slice_with_length(0, 1);
        drop(output);
        assert!(budget.reserved() > 0);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
    }
}

#[tokio::test]
async fn selects_list_null_empty_and_nullable_children() {
    let scalar: ArrayRef = Arc::new(LargeBinaryArray::from(vec![b"value".as_slice()]));
    let expected = ListArray::new(
        Arc::new(Field::new("item", DataType::LargeBinary, true)),
        OffsetBuffer::new(vec![0, 2, 2, 2, 4].into()),
        Arc::new(LargeBinaryArray::from(vec![
            Some(b"value".as_slice()),
            None,
            Some(b"value".as_slice()),
            Some(b"value".as_slice()),
        ])),
        Some(NullBuffer::from(vec![true, true, false, true])),
    );
    // Use the actual writer's level numbering while keeping Arrow values and
    // null/empty semantics as the independent decode oracle.
    let mut builder = RepDefBuilder::default();
    builder.add_offsets(expected.offsets().clone(), expected.nulls().cloned());
    builder.add_validity_bitmap(expected.values().nulls().unwrap().clone());
    let encoded = RepDefBuilder::serialize(vec![builder]);
    let (page, bytes) = fixture(
        &scalar,
        expected.len() as u64,
        encoded.repetition_levels.as_deref().unwrap(),
        encoded.definition_levels.as_deref().unwrap(),
        &encoded.def_meaning,
    );
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy {}).unwrap();
    for rows in [0..4, 1..3, 3..4] {
        let budget = AllocationBudget::new(2048);
        let mut work = budget.try_reserve(2048).unwrap();
        let input = scheduler
            .prepare_array_range_in(
                rows.clone(),
                expected.data_type().clone(),
                &mut work,
                |range, output| read(&bytes, range, output),
            )
            .await
            .unwrap()
            .unwrap();
        let output = input.decode_in(&mut work).unwrap().unwrap();
        assert_eq!(
            output.to_data(),
            expected
                .slice(rows.start as usize, (rows.end - rows.start) as usize)
                .to_data()
        );
        drop(output);
        drop(work);
        assert_eq!(budget.reserved(), 0);
    }
}

#[tokio::test]
async fn late_list_range_uses_bounded_repetition_scratch() {
    let scalar: ArrayRef = Arc::new(BinaryArray::from(vec![b"x".as_slice()]));
    let rep = [1, 0].repeat(5000);
    let (page, bytes) = fixture(
        &scalar,
        5000,
        &rep,
        &[],
        &[
            DefinitionInterpretation::AllValidItem,
            DefinitionInterpretation::AllValidList,
        ],
    );
    let ty = DataType::List(Arc::new(Field::new("item", DataType::Binary, false)));
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy {}).unwrap();
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    let mut reads = 0;
    let input = scheduler
        .prepare_array_range_in(4999..5000, ty, &mut work, |range, output| {
            assert!(output.capacity() <= 1024);
            reads += 1;
            read(&bytes, range, output)
        })
        .await
        .unwrap()
        .unwrap();
    assert!(reads > 10); // The 20KiB page cannot have been retained by a 1KiB workspace.
    let output = input.decode_in(&mut work).unwrap().unwrap();
    let list = output.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(
        list.value(0).to_data(),
        BinaryArray::from(vec![b"x".as_slice(), b"x".as_slice()]).to_data()
    );
    drop(output);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn refusal_and_cancel_release_prepared_input() {
    let scalar: ArrayRef = Arc::new(BinaryArray::from(vec![b"payload".as_slice()]));
    let (page, bytes) = fixture(
        &scalar,
        3,
        &[],
        &[0, 1, 0],
        &[DefinitionInterpretation::NullableItem],
    );
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy {}).unwrap();
    for capacity in [0, bytes.len() - 1, bytes.len(), 1024] {
        let budget = AllocationBudget::new(capacity);
        let mut work = budget.try_reserve(capacity).unwrap();
        let input = scheduler
            .prepare_array_range_in(0..3, DataType::Binary, &mut work, |range, output| {
                read(&bytes, range, output)
            })
            .await
            .unwrap();
        let result = input.map(|input| input.decode_in(&mut work).unwrap());
        assert_eq!(result.flatten().is_some(), capacity == 1024);
        drop(work);
        assert_eq!(budget.reserved(), 0);
    }
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    let mut future = Box::pin(scheduler.prepare_array_range_in(
        0..3,
        DataType::Binary,
        &mut work,
        |_, output| async move {
            std::future::pending::<()>().await;
            Ok(output)
        },
    ));
    assert!(future.as_mut().now_or_never().is_none());
    drop(future);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::header(0)]
#[case::offset(1)]
#[case::utf8(2)]
#[case::level(3)]
#[tokio::test]
async fn rejects_corrupt_scalar_or_selected_levels(#[case] kind: u8) {
    let scalar: ArrayRef = Arc::new(StringArray::from(vec!["hello"]));
    let (page, mut bytes) = fixture(
        &scalar,
        2,
        &[],
        &[0, 0],
        &[DefinitionInterpretation::NullableItem],
    );
    match kind {
        0 => bytes[0] = 9,
        1 => bytes[16] = 99,
        2 => bytes[20] = 255,
        _ => *bytes.last_mut().unwrap() = 255,
    }
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy {}).unwrap();
    let budget = AllocationBudget::new(1024);
    let mut work = budget.try_reserve(1024).unwrap();
    let input = scheduler
        .prepare_array_range_in(0..2, DataType::Utf8, &mut work, |range, output| {
            read(&bytes, range, output)
        })
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(
        error,
        Error::CorruptFile { .. } | Error::Arrow { .. } | Error::InvalidInput { .. }
    ));
    assert!(!error.to_string().is_empty());
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::range_overflow(0)]
#[case::odd_levels(1)]
#[case::missing_repetition(2)]
#[case::wrong_row_count(3)]
#[tokio::test]
async fn rejects_invalid_page_bounds_without_unadmitted_reads(#[case] kind: u8) {
    let scalar: ArrayRef = Arc::new(BinaryArray::from(vec![b"x".as_slice()]));
    let (mut page, bytes) = fixture(
        &scalar,
        2,
        &[1, 1],
        &[],
        &[
            DefinitionInterpretation::AllValidItem,
            DefinitionInterpretation::AllValidList,
        ],
    );
    let mut buffers = page.buffer_offsets_and_sizes.to_vec();
    match kind {
        0 => buffers[0] = (u64::MAX, 2),
        1 => buffers[1].1 = 3,
        2 => buffers[1].1 = 0,
        _ => page.num_rows = 3,
    }
    page.buffer_offsets_and_sizes = buffers.into();
    let scheduler =
        AdmittedBytePageScheduler::try_from_page(&page, &DefaultDecompressionStrategy {});
    if kind != 3 {
        let error = scheduler.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("Constant"));
        return;
    }
    let budget = AllocationBudget::new(128);
    let mut work = budget.try_reserve(128).unwrap();
    let error = scheduler
        .unwrap()
        .prepare_array_range_in(
            2..3,
            DataType::List(Arc::new(Field::new("item", DataType::Binary, false))),
            &mut work,
            |range, buffer| read(&bytes, range, buffer),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("row count"));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
