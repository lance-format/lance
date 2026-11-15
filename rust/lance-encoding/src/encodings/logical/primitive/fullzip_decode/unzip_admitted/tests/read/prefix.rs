// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::encodings::logical::primitive::DecodedFullZipPrefix;

use super::*;

fn envelope(f: &Fixture, rows: Range<u64>) -> usize {
    let (input, stage, decoded) = sizes(f, rows);
    (f.index.len() / 5).max(input + stage).max(stage + decoded)
}

fn validate(f: &Fixture, result: DecodedFullZipPrefix, budget: &Arc<AllocationBudget>) {
    let rows = result.rows;
    let page = result.page;
    let values = f.value_starts[rows.start as usize]..f.value_starts[rows.end as usize];
    let levels = f.level_starts[rows.start as usize]..f.level_starts[rows.end as usize];
    let (_, _, decoded) = sizes(f, rows);
    assert_eq!(budget.reserved(), decoded);
    let mut work = budget.try_reserve(256).unwrap();
    let refs = f.values[values.clone()]
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let (ty, expected) = if f.scheduler.bits_per_offset == 32 {
        (DataType::Binary, BinaryArray::from_vec(refs).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(refs).to_data(),
        )
    };
    let arrow = page.data.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(arrow, expected);
    let expected = RepDefUnraveler::new(
        f.rep.as_ref().map(|v| v[levels.clone()].to_vec()),
        f.def.as_ref().map(|v| v[levels].to_vec()),
        f.meanings.clone().into(),
        values.len() as u64,
    );
    let mut expected = CompositeRepDefUnraveler::new(vec![expected]);
    let mut actual = CompositeRepDefUnraveler::new(vec![page.repdef]);
    let bits = |validity: Option<arrow_buffer::NullBuffer>, len| {
        validity.map_or_else(|| vec![true; len], |v| v.iter().collect::<Vec<_>>())
    };
    assert_eq!(
        bits(
            actual
                .try_unravel_validity_in(values.len(), &mut work)
                .unwrap()
                .unwrap(),
            values.len()
        ),
        bits(
            expected.unravel_validity(values.len()).unwrap(),
            values.len()
        ),
    );
    for _ in f.meanings.iter().filter(|m| m.is_list()) {
        let (offsets, validity) = actual
            .try_unravel_offsets_in::<i32>(&mut work)
            .unwrap()
            .unwrap();
        let (expected_offsets, expected_validity) = expected.unravel_offsets::<i32>().unwrap();
        assert_eq!(&offsets[..], &expected_offsets[..]);
        assert_eq!(
            bits(validity, offsets.len() - 1),
            bits(expected_validity, expected_offsets.len() - 1)
        );
    }
    drop((actual, page.data, arrow, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn prefixes_complete_all_rows_under_one_fixed_budget(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] fsst: bool,
    #[values("flat", "nullable", "list", "deep")] kind: &str,
    #[values(false, true)] tight: bool,
    #[values(0, 1)] start: u64,
) {
    let mut f = fixture(bits, fsst, kind);
    let source = source(&mut f);
    let limit = if tight {
        (0..4).map(|row| envelope(&f, row..row + 1)).max().unwrap()
    } else {
        envelope(&f, start..4)
    };
    let budget = AllocationBudget::new(limit);
    let mut cursor = start;
    let mut batches = 0;
    while cursor < 4 {
        let reads = Arc::new(Reads::default());
        let result = f
            .scheduler
            .try_read_decode_prefix_in(
                cursor..4,
                &budget,
                limit,
                reader(source.clone(), reads.clone(), Fault::None),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.rows.start, cursor);
        assert!(result.rows.end > cursor && result.rows.end <= 4);
        assert!(budget.reserved() <= limit);
        // At most three finite size attempts (4, 2, 1 rows), never a credit wait.
        assert!(reads.0.lock().unwrap().len() <= 9);
        cursor = result.rows.end;
        validate(&f, result, &budget);
        batches += 1;
    }
    assert_eq!(cursor, 4);
    if tight && start == 0 {
        assert!(batches > 1);
    } else if !tight {
        assert_eq!(batches, 1);
    }
}

#[rstest]
#[tokio::test]
async fn compressed_prefix_retries_after_codec_output_refusal(
    #[values(32, 64)] bits: u8,
    #[values("flat", "nullable", "list", "deep")] kind: &str,
) {
    // FSST samples complete strings. Both long values must share a vocabulary
    // so sampling either one compresses the other instead of escaping most bytes.
    // Keep the original skewed fixture above: it validates a different boundary.
    let first = b"abcdefgh".repeat(2048);
    let mut second = first.clone();
    *second.last_mut().unwrap() = b'Z';
    let mut f = fixture_with_values(bits, true, kind, vec![first, vec![], second, vec![]]);
    let source = source(&mut f);
    let limit = (0..4).map(|row| envelope(&f, row..row + 1)).max().unwrap();
    let (input, stage, decoded) = sizes(&f, 0..4);
    assert!(input + stage <= limit);
    assert!(stage + decoded > limit);
    let budget = AllocationBudget::new(limit);
    let mut work = budget.try_reserve(limit).unwrap();
    let task = f
        .scheduler
        .try_read_unzip_range_in(
            0..4,
            &mut work,
            reader(source.clone(), Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), limit - stage);
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut work).unwrap().is_none());
        assert_eq!(work.len(), limit - stage);
        assert_eq!(budget.reserved(), limit);
    }
    drop((task, work));
    assert_eq!(budget.reserved(), 0);

    let mut cursor = 0;
    let mut batches = 0;
    while cursor < 4 {
        let reads = Arc::new(Reads::default());
        let result = f
            .scheduler
            .try_read_decode_prefix_in(
                cursor..4,
                &budget,
                limit,
                reader(source.clone(), reads.clone(), Fault::None),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.rows.start, cursor);
        assert!(result.rows.end > cursor && result.rows.end <= 4);
        assert!(reads.0.lock().unwrap().len() <= 9);
        if cursor == 0 {
            assert!(result.rows.end < 4);
            assert!(reads.0.lock().unwrap().len() > 3);
        }
        cursor = result.rows.end;
        validate(&f, result, &budget);
        batches += 1;
    }
    assert_eq!(cursor, 4);
    assert!(batches > 1);
}

#[rstest]
#[tokio::test]
async fn prefix_cancellation_and_errors_release_entire_attempt(
    #[values(0, 1, 2)] at: usize,
    #[values(false, true)] cancel: bool,
) {
    let mut f = fixture(32, true, "deep");
    let source = source(&mut f);
    let limit = envelope(&f, 0..4);
    let budget = AllocationBudget::new(limit);
    let reads = Arc::new(Reads::default());
    let fault = if cancel {
        Fault::Pending(at)
    } else {
        Fault::Error(at)
    };
    let mut future = Box::pin(f.scheduler.try_read_decode_prefix_in(
        0..4,
        &budget,
        limit,
        reader(source, reads.clone(), fault),
    ));
    if cancel {
        assert!(poll!(future.as_mut()).is_pending());
        assert_eq!(budget.reserved(), limit);
        assert!(budget.try_reserve(1).is_none());
    } else {
        let Err(error) = future.as_mut().await else {
            panic!("expected read error")
        };
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("injected read error"));
    }
    assert_eq!(reads.0.lock().unwrap().len(), at + 1);
    drop(future);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn retained_output_defers_next_operation_without_reading(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] fsst: bool,
) {
    let mut f = fixture(bits, fsst, "flat");
    let source = source(&mut f);
    let limit = envelope(&f, 0..4);
    let budget = AllocationBudget::new(limit);
    let result = f
        .scheduler
        .try_read_decode_prefix_in(
            0..4,
            &budget,
            limit,
            reader(source.clone(), Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let mut work = budget.try_reserve(0).unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let arrow = result
        .page
        .data
        .try_to_arrow_in(&ty, &mut work)
        .unwrap()
        .unwrap();
    let alias = arrow.slice(0, 1);
    drop((result, arrow, work));
    assert_eq!(budget.reserved(), sizes(&f, 0..4).2);
    let reads = Arc::new(Reads::default());
    assert!(
        f.scheduler
            .try_read_decode_prefix_in(
                0..4,
                &budget,
                limit,
                reader(source.clone(), reads.clone(), Fault::None),
            )
            .await
            .unwrap()
            .is_none()
    );
    assert!(reads.0.lock().unwrap().is_empty());
    drop(alias);
    assert_eq!(budget.reserved(), 0);
    let result = f
        .scheduler
        .try_read_decode_prefix_in(
            0..4,
            &budget,
            limit,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    validate(&f, result, &budget);
}

#[tokio::test]
async fn corrupt_later_row_is_not_hidden_by_smaller_prefix_retry() {
    let mut f = fixture(32, false, "flat");
    let mut source = source(&mut f);
    let offset = 37 + f.row_offsets[3];
    Arc::make_mut(&mut source)[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let limit = envelope(&f, 0..4);
    let budget = AllocationBudget::new(limit);
    let reads = Arc::new(Reads::default());
    let result = f
        .scheduler
        .try_read_decode_prefix_in(
            0..4,
            &budget,
            limit,
            reader(source, reads.clone(), Fault::None),
        )
        .await;
    let Err(error) = result else {
        panic!("expected corrupt selected input")
    };
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("item length"));
    assert_eq!(reads.0.lock().unwrap().len(), 3);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::empty(2..2, 1024)]
#[case::reversed(Range { start: 2, end: 1 }, 1024)]
#[case::outside(0..5, 1024)]
#[case::zero_workspace(0..4, 0)]
#[tokio::test]
async fn invalid_prefix_request_rejected_before_admission(
    #[case] rows: Range<u64>,
    #[case] bytes: usize,
) {
    let mut f = fixture(32, false, "flat");
    let source = source(&mut f);
    let reads = Arc::new(Reads::default());
    let budget = AllocationBudget::new(0);
    let result = f
        .scheduler
        .try_read_decode_prefix_in(
            rows,
            &budget,
            bytes,
            reader(source, reads.clone(), Fault::None),
        )
        .await;
    let Err(error) = result else {
        panic!("expected invalid prefix")
    };
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("nonempty page rows and workspace")
    );
    assert!(reads.0.lock().unwrap().is_empty());
    assert_eq!(budget.reserved(), 0);
}
