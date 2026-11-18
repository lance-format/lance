// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{ArrayRef, LargeListArray, ListArray};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::Field as ArrowField;

use super::*;

fn list_case(
    data_type: &DataType,
    large: bool,
    nullable: bool,
    parent: Meaning,
) -> (Fixture, ArrayRef) {
    let bits = if matches!(data_type, DataType::Binary | DataType::Utf8) {
        32
    } else {
        64
    };
    let emptyable = matches!(
        parent,
        Meaning::EmptyableList | Meaning::NullableAndEmptyableList
    );
    let parent_nullable = matches!(
        parent,
        Meaning::NullableList | Meaning::NullableAndEmptyableList
    );
    // Logical values are the oracle. The writer receives separately constructed
    // levels; the output planner is not consulted to price or build expectations.
    let logical = [
        Some(vec![Some("a"), if nullable { None } else { Some("b") }]),
        Some(if emptyable { vec![] } else { vec![Some("β")] }),
        if parent_nullable {
            None
        } else {
            Some(vec![Some("γ")])
        },
        Some(vec![Some(""), Some("δ")]),
    ];
    let logical = (0..12)
        .map(|row| logical[row % 4].clone())
        .collect::<Vec<_>>();
    let mut values = Vec::new();
    let mut items = Vec::new();
    let mut reps = Vec::new();
    let mut defs = Vec::new();
    let mut starts = vec![0];
    let mut levels = vec![0];
    for row in &logical {
        match row {
            None => {
                reps.push(1);
                defs.push(u16::from(nullable) + 1);
            }
            Some(row) if row.is_empty() => {
                reps.push(1);
                defs.push(u16::from(nullable) + if parent_nullable { 2 } else { 1 });
            }
            Some(row) => {
                for (index, value) in row.iter().enumerate() {
                    reps.push(u16::from(index == 0));
                    defs.push(u16::from(value.is_none()));
                    items.push(value.map(str::as_bytes));
                    values.push(value.unwrap_or_default().as_bytes().to_vec());
                }
            }
        }
        starts.push(values.len());
        levels.push(reps.len());
    }
    let child = arrow_array::make_array(super::flat::expected(data_type, &items));
    let field = Arc::new(
        ArrowField::new("item", data_type.clone(), nullable).with_metadata(
            std::collections::HashMap::from([("source".to_string(), "oracle".to_string())]),
        ),
    );
    let validity = parent_nullable
        .then(|| NullBuffer::from(logical.iter().map(Option::is_some).collect::<Vec<_>>()));
    let expected: ArrayRef = if large {
        Arc::new(LargeListArray::new(
            field,
            OffsetBuffer::new(starts.iter().map(|v| *v as i64).collect::<Vec<_>>().into()),
            child,
            validity,
        ))
    } else {
        Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(starts.iter().map(|v| *v as i32).collect::<Vec<_>>().into()),
            child,
            validity,
        ))
    };
    let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let block = if bits == 32 {
        DataBlock::from_array(BinaryArray::from_vec(refs))
    } else {
        DataBlock::from_array(LargeBinaryArray::from_vec(refs))
    };
    let meanings = vec![
        if nullable {
            Meaning::NullableItem
        } else {
            Meaning::AllValidItem
        },
        parent,
    ];
    let field = Field::try_from(&ArrowField::new("values", data_type.clone(), nullable)).unwrap();
    let page = PrimitiveStructuralEncoder::encode_full_zip(
        0,
        &field,
        &Codec(false),
        block,
        SerializedRepDefs::new(Some(reps.clone()), Some(defs.clone()), meanings.clone()),
        0,
        logical.len() as u64,
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
        logical.len() as u64,
        &layout,
        &DefaultDecompressionStrategy::default(),
    )
    .unwrap();
    let row_offsets = ByteUnpacker::new(
        page.data[1].clone(),
        page.data[1].len() / (logical.len() + 1),
    )
    .map(|offset| usize::try_from(offset).unwrap())
    .collect();
    (
        Fixture {
            scheduler,
            zipped: page.data[0].clone(),
            index: page.data[1].clone(),
            row_offsets,
            values,
            meanings,
            rep: Some(reps),
            def: Some(defs),
            value_starts: starts,
            level_starts: levels,
        },
        expected,
    )
}

#[rstest]
#[tokio::test]
async fn raw_list_fits_input_and_final_nodes_only(
    #[values(
        DataType::Binary,
        DataType::LargeBinary,
        DataType::Utf8,
        DataType::LargeUtf8
    )]
    data_type: DataType,
    #[values(false, true)] large: bool,
    #[values(false, true)] nullable: bool,
    #[values(
        Meaning::AllValidList,
        Meaning::NullableList,
        Meaning::EmptyableList,
        Meaning::NullableAndEmptyableList
    )]
    parent: Meaning,
    #[values(0..4, 1..3, 3..10)] rows: Range<u64>,
) {
    let (mut f, expected) = list_case(&data_type, large, nullable, parent);
    let start = rows.start as usize;
    let end = rows.end as usize;
    let selected = f.value_starts[start]..f.value_starts[end];
    let child_nulls = f.def.as_ref().unwrap()[f.level_starts[start]..f.level_starts[end]]
        .iter()
        .filter(|d| **d == 1 && nullable)
        .count();
    let parent_nulls = expected.slice(start, end - start).null_count();
    let parent_bytes = (end - start + 1) * if large { 8 } else { 4 }
        + if parent_nulls == 0 {
            0
        } else {
            (end - start).div_ceil(8)
        };
    let child_bytes = (selected.len() + 1) * usize::from(f.scheduler.bits_per_offset / 8)
        + f.values[selected.clone()]
            .iter()
            .map(Vec::len)
            .sum::<usize>()
        + if child_nulls == 0 {
            0
        } else {
            selected.len().div_ceil(8)
        };
    let input_bytes = f.row_offsets[end] - f.row_offsets[start];
    let limit = input_bytes + parent_bytes + child_bytes;
    let source = source(&mut f);
    let scheduler = Arc::new(f.scheduler);
    for shortage in [true, false] {
        let capacity = limit - usize::from(shortage);
        let budget = AllocationBudget::new(capacity);
        let mut work = budget.try_reserve(capacity).unwrap();
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
            assert_eq!(budget.reserved(), 0);
            continue;
        }
        let output = output.unwrap();
        assert_eq!(
            output.to_data(),
            expected.slice(start, end - start).to_data()
        );
        assert_eq!(work.len(), input_bytes, "no transient level allocations");
        let data = output.to_data();
        let parent_alias = data
            .nulls()
            .map(|n| n.buffer().clone())
            .unwrap_or_else(|| data.buffers()[0].clone());
        let child = &data.child_data()[0];
        let child_alias = child
            .nulls()
            .map(|n| n.buffer().clone())
            .unwrap_or_else(|| child.buffers()[1].clone());
        drop((output, data, work));
        assert_eq!(budget.reserved(), parent_bytes + child_bytes);
        assert!(budget.try_reserve(capacity).is_none());
        drop(parent_alias);
        assert_eq!(budget.reserved(), child_bytes);
        drop(child_alias);
        assert_eq!(budget.reserved(), 0);
        let mut work = budget.try_reserve(capacity).unwrap();
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
            budget.reserved(),
            0,
            "next operation progresses after final consumer release"
        );
    }
}

#[rstest]
#[case::continuation_after_empty(2, false)]
#[case::continuation_after_null(3, false)]
#[case::hidden_continuation(1, false)]
#[case::invalid_utf8(0, true)]
#[tokio::test]
async fn raw_list_rejects_invalid_structure_and_releases_attempt(
    #[case] row: usize,
    #[case] utf8: bool,
) {
    let (mut f, expected) = list_case(
        &DataType::Utf8,
        false,
        true,
        Meaning::NullableAndEmptyableList,
    );
    let input_bytes = f.row_offsets[4];
    let at = 37 + f.row_offsets[row];
    let parser = f.scheduler.details.ctrl_word_parser;
    let ControlWordParser::BOTH8(shift, _) = parser else {
        panic!("fixture requires one-byte control")
    };
    let mut source = source(&mut f);
    let bytes = Arc::make_mut(&mut source);
    if utf8 {
        bytes[at + 1 + 4] = 0xff;
    } else {
        bytes[at] &= !(1 << shift);
    }
    // Structural corruption must be detected with no output credit. UTF-8 is
    // validated by Arrow after final buffers are admitted.
    let limit = if utf8 { 4096 } else { input_bytes };
    let budget = AllocationBudget::new(limit);
    let mut work = budget.try_reserve(limit).unwrap();
    let input = Arc::new(f.scheduler)
        .prepare_array_range_in(
            0..4,
            expected.data_type().clone(),
            &mut work,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    if utf8 {
        assert!(matches!(error, Error::Arrow { .. }));
        assert!(error.to_string().to_lowercase().contains("utf"));
    } else {
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("list continuation"));
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn raw_list_preserves_child_nullability_validation() {
    let (mut f, _) = list_case(
        &DataType::Binary,
        false,
        true,
        Meaning::NullableAndEmptyableList,
    );
    let source = source(&mut f);
    let declared = DataType::List(Arc::new(ArrowField::new("item", DataType::Binary, false)));
    let budget = AllocationBudget::new(4096);
    let mut work = budget.try_reserve(4096).unwrap();
    let input = Arc::new(f.scheduler)
        .prepare_array_range_in(
            0..4,
            declared,
            &mut work,
            reader(source, Arc::default(), Fault::None),
        )
        .await
        .unwrap()
        .unwrap();
    let error = input.decode_in(&mut work).unwrap_err();
    assert!(matches!(error, Error::Arrow { .. }));
    assert!(error.to_string().contains("non-nullable child"));
    drop(work);
    assert_eq!(budget.reserved(), 0);
}
