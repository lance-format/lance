// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatchIterator;
use arrow_schema::Schema as ArrowSchema;
use lance_core::cache::LanceCache;
use lance_encoding::data::try_interleave_nested_binary_in;
use lance_file::{
    reader::{FileReader, FileReaderOptions, FullZipScanOutcome},
    version::LanceFileVersion,
};
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};

use super::*;
use crate::dataset::{Dataset, WriteParams};

#[rstest]
#[tokio::test]
async fn validity_resolution_composes_with_real_file_owners(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    const WORKSPACE: usize = 65536;
    let schema = Arc::new(ArrowSchema::new(
        ["base", "left", "right"]
            .into_iter()
            .map(|name| {
                ArrowField::new(name, DataType::Utf8, true).with_metadata(HashMap::from([(
                    "lance-encoding:structural-encoding".into(),
                    "fullzip".into(),
                )]))
            })
            .collect::<Vec<_>>(),
    ));
    let columns: Vec<ArrayRef> = ["base", "left", "right"]
        .map(|name| {
            Arc::new(StringArray::from_iter((0..12).map(|row| {
                let is_valid = match name {
                    "base" => !matches!(row % 6, 2 | 3),
                    "left" => row % 6 != 2,
                    _ => row % 6 != 4,
                };
                is_valid.then(|| format!("{name}{row}"))
            }))) as ArrayRef
        })
        .into();
    let input = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(input)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 6,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    for (fragment_index, fragment) in fragments.iter().enumerate() {
        let path = dataset
            .data_dir()
            .join(fragment.metadata.files[0].path.as_str());
        let scheduler = ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
                use_lite_scheduler: Some(lite),
            },
        );
        let file = scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            file,
            None,
            Default::default(),
            &Arc::new(LanceCache::no_cache()),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        let budget = AllocationBudget::new(WORKSPACE);
        let mut scan = reader
            .open_admitted_fullzip_scan(0..6, budget.clone(), WORKSPACE, 6)
            .await
            .unwrap();
        let output = scan
            .try_next_with_transform(|rows, batch, work| {
                assert_eq!(rows, 0..6);
                let selection = OverlayBatchSelection::Range(0..6);
                let left_coverage: RoaringBitmap = [0, 2, 4].into_iter().collect();
                let right_coverage: RoaringBitmap = [1, 2, 4].into_iter().collect();
                let Some(left_route) =
                    AdmittedOverlayRouting::try_new(&selection, &[&left_coverage], work)?
                else {
                    return Ok(None);
                };
                let Some(right_route) =
                    AdmittedOverlayRouting::try_new(&selection, &[&right_coverage], work)?
                else {
                    return Ok(None);
                };
                let Some(left) = try_interleave_nested_binary_in(
                    &[batch.column(1).as_ref()],
                    &[(0, 0), (0, 2), (0, 4)],
                    work,
                )?
                else {
                    return Ok(None);
                };
                let Some(right) = try_interleave_nested_binary_in(
                    &[batch.column(2).as_ref()],
                    &[(0, 1), (0, 2), (0, 4)],
                    work,
                )?
                else {
                    return Ok(None);
                };
                // Reuse real decoded validity buffers as ancestor masks. This tests
                // shared physical owners without claiming general struct page decode.
                let donor = |array: ArrayRef| {
                    StructArray::new(
                        vec![Arc::new(ArrowField::new("value", DataType::Utf8, true))].into(),
                        vec![array.clone()],
                        array.nulls().cloned(),
                    )
                };
                let left_struct = donor(left.clone());
                let right_struct = donor(right.clone());
                let fields: Fields = vec![
                    Arc::new(ArrowField::new("left", DataType::Utf8, true)),
                    Arc::new(ArrowField::new("right", DataType::Utf8, true)),
                ]
                .into();
                let base = StructArray::new(
                    fields.clone(),
                    vec![batch.column(0).clone(); 2],
                    batch.column(0).nulls().cloned(),
                );
                let inputs = [
                    OverlayStructValidityInput {
                        routing: &left_route,
                        file_order: &[9],
                        sources: &[&left_struct],
                    },
                    OverlayStructValidityInput {
                        routing: &right_route,
                        file_order: &[2],
                        sources: &[&right_struct],
                    },
                ];
                let Some(validity) = AdmittedStructValidityRouting::try_new(&base, &inputs, work)?
                else {
                    return Ok(None);
                };
                let Some(left) = left_route.try_interleave_in(batch.column(0), &[left], work)?
                else {
                    return Ok(None);
                };
                let Some(right) = right_route.try_interleave_in(batch.column(0), &[right], work)?
                else {
                    return Ok(None);
                };
                let children = StructArray::new(fields, vec![left, right], base.nulls().cloned());
                let Some(output) =
                    try_replace_struct_validities_in(&children, &[validity.selection(&[])], work)?
                else {
                    return Ok(None);
                };
                let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "payload",
                    output.data_type().clone(),
                    true,
                )]));
                Ok(Some(RecordBatch::try_new(schema, vec![output])?))
            })
            .await
            .unwrap();
        let FullZipScanOutcome::Batch { rows, batch } = output else {
            panic!("compound attempt must fit")
        };
        assert_eq!(rows, 0..6);
        let result = batch.column(0).as_struct();
        assert_eq!(
            result.nulls(),
            Some(&NullBuffer::from(vec![
                true, true, true, false, false, true
            ]))
        );
        let start = fragment_index * 6;
        let expected_left = StringArray::from(vec![
            Some(format!("left{start}")),
            Some(format!("base{}", start + 1)),
            None,
            None,
            Some(format!("left{}", start + 4)),
            Some(format!("base{}", start + 5)),
        ]);
        let expected_right = StringArray::from(vec![
            Some(format!("base{start}")),
            Some(format!("right{}", start + 1)),
            Some(format!("right{}", start + 2)),
            None,
            None,
            Some(format!("base{}", start + 5)),
        ]);
        assert_eq!(result.column(0).to_data(), expected_left.to_data());
        assert_eq!(result.column(1).to_data(), expected_right.to_data());
        let alias = result.nulls().unwrap().buffer().slice_with_length(0, 1);
        drop(batch);
        assert_eq!(
            budget.reserved(),
            1,
            "only the final mask owner should remain"
        );
        assert!(budget.try_reserve(WORKSPACE).is_none());
        drop(alias);
        assert_eq!(budget.reserved(), 0);
        assert!(budget.try_reserve(WORKSPACE).is_some());
    }
}
