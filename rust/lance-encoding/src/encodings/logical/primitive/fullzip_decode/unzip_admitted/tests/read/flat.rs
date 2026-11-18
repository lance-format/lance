// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{LargeStringArray, StringArray};

use super::*;
use crate::{
    compression::{PerValueDecodeBudget, VariablePerValueDecompressor},
    data::VariableWidthBlock,
};

fn expected(data_type: &DataType, values: &[Option<&[u8]>]) -> arrow_data::ArrayData {
    match data_type {
        DataType::Binary => BinaryArray::from(values.to_vec()).to_data(),
        DataType::LargeBinary => LargeBinaryArray::from(values.to_vec()).to_data(),
        DataType::Utf8 => StringArray::from_iter(
            values
                .iter()
                .map(|value| value.map(|v| std::str::from_utf8(v).unwrap())),
        )
        .to_data(),
        DataType::LargeUtf8 => LargeStringArray::from_iter(
            values
                .iter()
                .map(|value| value.map(|v| std::str::from_utf8(v).unwrap())),
        )
        .to_data(),
        _ => unreachable!(),
    }
}

#[rstest]
#[tokio::test]
async fn flat_raw_uses_only_input_and_final_arrow_payload(
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    data_type: DataType,
    #[values("flat", "nullable")] kind: &str,
    #[values(0..4, 1..3, 1..2, 3..4)] rows: Range<u64>,
) {
    let bits = if matches!(data_type, DataType::Binary | DataType::Utf8) {
        32
    } else {
        64
    };
    let mut f = fixture_with_values(
        bits,
        false,
        kind,
        vec![
            "hello α".as_bytes().to_vec(),
            vec![],
            "world β".as_bytes().to_vec(),
            vec![],
        ],
    );
    let selected = f.values[rows.start as usize..rows.end as usize]
        .iter()
        .enumerate()
        .map(|(i, value)| {
            (kind != "nullable" || i + rows.start as usize != 1).then_some(value.as_slice())
        })
        .collect::<Vec<_>>();
    let expected = expected(&data_type, &selected);
    // Price final Arrow buffers from logical values, not the decoder's planner.
    let output = (selected.len() + 1) * usize::from(bits / 8)
        + selected
            .iter()
            .flatten()
            .map(|value| value.len())
            .sum::<usize>()
        + if selected.contains(&None) {
            selected.len().div_ceil(8)
        } else {
            0
        };
    let input_bytes = f.row_offsets[rows.end as usize] - f.row_offsets[rows.start as usize];
    let limit = input_bytes + output;
    let source = source(&mut f);
    let scheduler = Arc::new(f.scheduler);
    // One byte short must refuse without a partial Arrow result or leaked input.
    let short = AllocationBudget::new(limit - 1);
    let mut work = short.try_reserve(limit - 1).unwrap();
    let input = scheduler
        .clone()
        .prepare_array_range_in(
            rows.clone(),
            data_type.clone(),
            &mut work,
            reader(source.clone(), Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    assert!(input.decode_in(&mut work).unwrap().is_none());
    assert_eq!(work.len(), limit - 1);
    drop(work);
    assert_eq!(short.reserved(), 0);

    let budget = AllocationBudget::new(limit);
    let mut work = budget.try_reserve(limit).unwrap();
    let input = scheduler
        .clone()
        .prepare_array_range_in(
            rows.clone(),
            data_type.clone(),
            &mut work,
            reader(source.clone(), Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let array = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(array.to_data(), expected);
    assert_eq!(
        work.len(),
        input_bytes,
        "input recycled; no transient levels retained"
    );
    let data = array.to_data();
    let offsets = data.buffers()[0].clone();
    let values = data.buffers()[1].clone();
    let nulls = data.nulls().cloned();
    drop((data, array, work));
    assert_eq!(budget.reserved(), output);
    assert!(budget.try_reserve(limit).is_none());
    drop(offsets);
    assert_eq!(budget.reserved(), output);
    if nulls.is_some() {
        drop(values);
        assert_eq!(
            budget.reserved(),
            output,
            "validity alias owns the entire backing allocation"
        );
        drop(nulls);
    } else {
        drop(values);
    }
    assert_eq!(budget.reserved(), 0);
    let mut work = budget.try_reserve(limit).unwrap();
    let input = scheduler
        .prepare_array_range_in(
            rows,
            data_type,
            &mut work,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    drop((input.decode_in(&mut work).unwrap().unwrap(), work));
    assert_eq!(
        budget.reserved(),
        0,
        "consumer release permits a complete next operation"
    );
}

#[rstest]
#[case::bad_length(false, "length")]
#[case::bad_control(true, "control")]
#[case::extra_row(false, "extra")]
#[tokio::test]
async fn flat_raw_validates_framing_even_without_output_credit(
    #[case] nullable: bool,
    #[case] fault: &str,
) {
    let mut f = fixture(32, false, if nullable { "nullable" } else { "flat" });
    let input_bytes = f.zipped.len();
    let last = f.row_offsets[3];
    let mut source = source(&mut f);
    let bytes = Arc::make_mut(&mut source);
    match fault {
        "length" => bytes[37 + last..37 + last + 4].copy_from_slice(&u32::MAX.to_le_bytes()),
        "control" => bytes[37] = 2,
        // Split the final nonempty item into an empty row plus remaining payload.
        _ => bytes[37 + f.row_offsets[2]..37 + f.row_offsets[2] + 4].fill(0),
    }
    let budget = AllocationBudget::new(input_bytes);
    let mut work = budget.try_reserve(input_bytes).unwrap();
    let input = Arc::new(f.scheduler)
        .prepare_array_range_in(
            0..4,
            DataType::Binary,
            &mut work,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains(if fault == "control" {
        "control level"
    } else {
        "item length"
    }));
    assert_eq!(work.len(), input_bytes);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn flat_raw_arrow_rejects_invalid_utf8_and_releases_output() {
    let mut f = fixture_with_values(
        32,
        false,
        "flat",
        vec![vec![0xff], vec![], b"ok".to_vec(), vec![]],
    );
    let source = source(&mut f);
    let budget = AllocationBudget::new(4096);
    let mut work = budget.try_reserve(4096).unwrap();
    let input = Arc::new(f.scheduler)
        .prepare_array_range_in(
            0..4,
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
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn flat_raw_partial_validity_bytes_preserve_sliced_nulls() {
    let values = (0..19)
        .map(|row| {
            if row % 3 == 1 {
                vec![]
            } else {
                format!("value {row} α").into_bytes()
            }
        })
        .collect::<Vec<_>>();
    let selected = values[2..19]
        .iter()
        .enumerate()
        .map(|(i, v)| ((i + 2) % 3 != 1).then_some(v.as_slice()))
        .collect::<Vec<_>>();
    let expected = expected(&DataType::LargeUtf8, &selected);
    let output_bytes = 18 * 8 + selected.iter().flatten().map(|v| v.len()).sum::<usize>() + 3;
    let mut f = fixture_with_values(64, false, "nullable", values.clone());
    let source = source(&mut f);
    let input_bytes = f.row_offsets[19] - f.row_offsets[2];
    let budget = AllocationBudget::new(input_bytes + output_bytes);
    let mut work = budget.try_reserve(input_bytes + output_bytes).unwrap();
    let input = Arc::new(f.scheduler)
        .prepare_array_range_in(
            2..19,
            DataType::LargeUtf8,
            &mut work,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let array = input.decode_in(&mut work).unwrap().unwrap();
    assert_eq!(array.to_data(), expected);
    assert_eq!(array.slice(5, 9).to_data(), expected.slice(5, 9));
    let nulls = array.nulls().unwrap().clone();
    assert_eq!(
        nulls.buffer().as_slice()[2] & 0xfe,
        0,
        "unused validity bits remain zero"
    );
    drop((array, work));
    assert_eq!(budget.reserved(), output_bytes);
    drop(nulls);
    assert_eq!(budget.reserved(), 0);
}

#[derive(Debug)]
struct ValidatingIdentity(Arc<dyn VariablePerValueDecompressor>);

impl VariablePerValueDecompressor for ValidatingIdentity {
    fn is_identity(&self) -> bool {
        true
    }
    fn decompress(&self, _: VariableWidthBlock) -> Result<DataBlock> {
        panic!("ordinary decode must not be called")
    }
    fn plan_decode(&self, data: &VariableWidthBlock) -> Result<PerValueDecodeBudget> {
        self.0.plan_decode(data)
    }
    fn try_decompress_in(
        &self,
        _: &VariableWidthBlock,
        _: &mut Reservation,
    ) -> Result<Option<DataBlock>> {
        Err(Error::corrupt_file_named(
            "custom codec",
            "identity-specific validation",
        ))
    }
}

#[tokio::test]
async fn scheduling_identity_hint_does_not_bypass_custom_codec_validation() {
    let mut f = fixture(32, false, "flat");
    let details = Arc::get_mut(&mut f.scheduler.details).unwrap();
    let PerValueDecompressor::Variable(codec) = &details.value_decompressor else {
        unreachable!()
    };
    details.value_decompressor =
        PerValueDecompressor::Variable(Arc::new(ValidatingIdentity(codec.clone())));
    let source = source(&mut f);
    let budget = AllocationBudget::new(262144);
    let mut work = budget.try_reserve(262144).unwrap();
    let input = Arc::new(f.scheduler)
        .prepare_array_range_in(
            0..4,
            DataType::Binary,
            &mut work,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("identity-specific validation"));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
