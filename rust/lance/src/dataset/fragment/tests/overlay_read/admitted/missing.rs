// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::schema_evolution::NewColumnTransform;

#[rstest]
#[tokio::test]
async fn missing_columns_compose_with_payload_overlays_and_deletions(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    let mut dataset = write_payload_dataset(version, true, true).await;
    let files = dataset
        .get_fragments()
        .iter()
        .map(|f| f.metadata.files.clone())
        .collect::<Vec<_>>();
    let added = Arc::new(ArrowSchema::new(vec![
        fullzip_field("late", DataType::LargeBinary),
        ArrowField::new("count_late", DataType::UInt64, true),
        ArrowField::new(
            "nested_late",
            DataType::List(Arc::new(ArrowField::new(
                "item",
                DataType::LargeBinary,
                true,
            ))),
            true,
        ),
        ArrowField::new(
            "tree_late",
            DataType::Struct(
                vec![
                    ArrowField::new("flag", DataType::Boolean, false),
                    ArrowField::new("text", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        ),
    ]));
    dataset
        .add_columns(NewColumnTransform::AllNulls(added.clone()), None, None)
        .await
        .unwrap();
    assert_eq!(
        dataset
            .get_fragments()
            .iter()
            .map(|f| f.metadata.files.clone())
            .collect::<Vec<_>>(),
        files
    );
    let field = dataset.schema().field("late").unwrap().id;
    for fragment in 0..2 {
        dataset = commit_overlay(
            dataset,
            &format!("missing-{fragment}"),
            fragment,
            &[field],
            OverlayCoverage::dense(bitmap([1, 3, 4])),
            vec![payload("late", 3, false, 0)],
            version,
        )
        .await;
    }
    dataset
        .delete("untouched IN ('other1', 'other4', 'other7', 'other10')")
        .await
        .unwrap();
    for fragment in 0..2 {
        let original = open(&dataset, fragment, dataset.schema(), lite).await;
        for markers in [false, true] {
            let mut reader = original.clone();
            reader.with_row_address();
            if markers {
                reader.with_make_deletions_null();
            }
            let budget = AllocationBudget::new(2 * 65536);
            for selected in [vec![0, 1, 2, 3, 4, 5], vec![1, 4], vec![0, 5], vec![]] {
                let indices = UInt32Array::from(selected.clone());
                let output = reader
                    .try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(output, reader.take_as_batch(&selected, None).await.unwrap());
                for name in ["count_late", "nested_late", "tree_late"] {
                    assert_eq!(
                        output[name].as_ref(),
                        arrow_array::new_null_array(
                            added.field_with_name(name).unwrap().data_type(),
                            output.num_rows()
                        )
                        .as_ref()
                    );
                }
                drop(output);
                assert_eq!(budget.reserved(), 0);
            }
            let indices = UInt32Array::from(vec![0, 1, 3, 4, 5]);
            let (left, right) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                tokio::join!(
                    reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
                    reader.try_take_admitted_fullzip(&indices, budget.try_reserve(65536).unwrap()),
                )
            })
            .await
            .expect("missing-column composition must progress with one CPU worker");
            let (left, right) = (left.unwrap().unwrap(), right.unwrap().unwrap());
            assert_eq!(left, right);
            drop((left, right));
            assert_eq!(budget.reserved(), 0);
        }
    }
}

#[rstest]
#[tokio::test]
async fn missing_only_projection_charges_empty_offsets_and_final_alias(
    #[values(false, true)] lite: bool,
) {
    let mut dataset = write_payload_dataset(LanceFileVersion::V2_3, false, false).await;
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "late",
                DataType::LargeBinary,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    let projection = dataset.schema().project(&["late"]).unwrap();
    for fragment in 0..2 {
        let reader = open(&dataset, fragment, &projection, lite).await;
        let budget = AllocationBudget::new(1024);
        for selected in [vec![0, 2, 5], vec![]] {
            let indices = UInt32Array::from(selected.clone());
            let offsets = (selected.len() + 1) * 8;
            let exact = offsets + selected.len().div_ceil(8);
            assert!(
                reader
                    .try_take_admitted_fullzip(&indices, budget.try_reserve(exact - 1).unwrap())
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_eq!(budget.reserved(), 0);
            let output = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(exact).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(output, reader.take_as_batch(&selected, None).await.unwrap());
            assert_eq!(budget.reserved(), exact);
            let alias = output.column(0).to_data().buffers()[0].slice_with_length(0, 1);
            drop(output);
            assert_eq!(budget.reserved(), offsets);
            assert!(budget.try_reserve(1024).is_none());
            drop(alias);
            assert_eq!(budget.reserved(), 0);
            let retry = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(exact).unwrap())
                .await
                .unwrap()
                .unwrap();
            drop(retry);
            assert_eq!(budget.reserved(), 0);
        }
        for indices in [
            UInt32Array::from(vec![1, 0]),
            UInt32Array::from(vec![1, 1]),
            UInt32Array::from(vec![6]),
        ] {
            let error = reader
                .try_take_admitted_fullzip(&indices, budget.try_reserve(1024).unwrap())
                .await
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("sorted unique"));
            assert_eq!(budget.reserved(), 0);
        }
    }
}
