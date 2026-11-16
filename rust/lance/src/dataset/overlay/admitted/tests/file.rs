// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{RecordBatchIterator, cast::AsArray};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use lance_core::cache::LanceCache;
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
async fn admitted_routing_composes_with_real_fragment_inputs(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] lite: bool,
) {
    const WORKSPACE: usize = 65536;
    let schema = Arc::new(ArrowSchema::new(
        ["base", "new", "old"]
            .into_iter()
            .map(|name| {
                ArrowField::new(name, arrow_schema::DataType::Utf8, true).with_metadata(
                    HashMap::from([(
                        "lance-encoding:structural-encoding".to_string(),
                        "fullzip".to_string(),
                    )]),
                )
            })
            .collect::<Vec<_>>(),
    ));
    let columns: Vec<ArrayRef> = ["base", "new", "old"]
        .into_iter()
        .map(|name| {
            Arc::new(StringArray::from_iter((0..12).map(|row| {
                (name != "new" || row % 3 != 2).then(|| format!("{name}{row}"))
            }))) as ArrayRef
        })
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
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
    for (fragment_index, fragment) in fragments.into_iter().enumerate() {
        let scheduler = ScanScheduler::new(
            dataset.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
                use_lite_scheduler: Some(lite),
            },
        );
        let path = dataset
            .data_dir()
            .join(fragment.metadata.files[0].path.as_str());
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
            .open_admitted_fullzip_scan(0..6, budget.clone(), WORKSPACE, 3)
            .await
            .unwrap();
        for start in [0_u64, 3] {
            let outcome = scan
                .try_next_with_transform(|rows, batch, workspace| {
                    let newer: RoaringBitmap = [2, 5].into_iter().collect();
                    let older: RoaringBitmap = [0, 2, 3, 5].into_iter().collect();
                    let coverages = [&newer, &older];
                    let selection =
                        OverlayBatchSelection::Range(rows.start as u32..rows.end as u32);
                    let before = workspace.len();
                    let Some(routing) =
                        AdmittedOverlayRouting::try_new(&selection, &coverages, workspace)?
                    else {
                        return Ok(None);
                    };
                    assert!(
                        workspace.len() < before,
                        "routing must use the read's reservation"
                    );
                    // This fixture decodes aligned candidate columns, then gathers
                    // the exact dense overlay rows requested by the production routing.
                    // It does not claim to have integrated sparse disk reads.
                    let mut fetched = Vec::new();
                    for (index, coverage) in coverages.iter().enumerate() {
                        let offsets = routing.fetch_offsets(index)?;
                        let indices = offsets
                            .iter()
                            .map(|offset| {
                                let physical = coverage.select(*offset).unwrap();
                                (0, (u64::from(physical) - rows.start) as usize)
                            })
                            .collect::<Vec<_>>();
                        let Some(array) = try_interleave_nested_binary_in(
                            &[batch.column(index + 1).as_ref()],
                            &indices,
                            workspace,
                        )?
                        else {
                            return Ok(None);
                        };
                        fetched.push(array);
                    }
                    let Some(output) =
                        routing.try_interleave_in(batch.column(0), &fetched, workspace)?
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
            let FullZipScanOutcome::Batch { rows, batch } = outcome else {
                panic!("input, routing and assembly must fit");
            };
            assert_eq!(rows, start..start + 3);
            let first = fragment_index * 6 + start as usize;
            let expected = StringArray::from(vec![
                Some(format!("old{first}")),
                Some(format!("base{}", first + 1)),
                None,
            ]);
            assert_eq!(batch.column(0).to_data(), expected.to_data());
            let alias = batch
                .column(0)
                .as_string::<i32>()
                .values()
                .slice_with_length(0, 1);
            drop(batch);
            assert!(
                budget.reserved() > 1,
                "thin output must retain its full allocation"
            );
            if rows.end < 6 {
                assert!(matches!(
                    scan.try_next().await.unwrap(),
                    FullZipScanOutcome::NeedsCapacity
                ));
                assert_eq!(scan.position(), rows.end);
            }
            drop(alias);
            assert_eq!(budget.reserved(), 0);
        }
        assert!(matches!(
            scan.try_next().await.unwrap(),
            FullZipScanOutcome::Finished
        ));
    }
}
