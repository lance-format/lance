// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION};

use super::*;
use crate::reader::PhysicalRowVersionSource;

#[derive(Debug)]
struct Versions {
    rows: u64,
}

impl PhysicalRowVersionSource for Versions {
    fn num_rows(&self) -> u64 {
        self.rows
    }
    fn select_range(&self, rows: Range<usize>) -> Result<Box<dyn Iterator<Item = u64> + '_>> {
        if rows.start > rows.end || rows.end as u64 > self.rows {
            return Err(Error::invalid_input("invalid version test range"));
        }
        Ok(Box::new(std::iter::repeat_n(2, rows.len())))
    }
}

#[rstest]
#[case::created(true, false)]
#[case::updated(false, true)]
#[case::both(true, true)]
#[tokio::test]
async fn version_columns_without_ids_retain_exact_owner(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[case] created: bool,
    #[case] updated: bool,
) {
    let expected = batches().remove(0);
    let (_fs, reader) = fixture_batches(version, lite, vec![expected.clone()], 1).await;
    let budget = AllocationBudget::new(4096);
    let mut scan = reader
        .open_admitted_fullzip_scan(1..7, budget.clone(), 4096, 3)
        .await
        .unwrap();
    if created {
        scan = scan.with_row_created_at_version(None).unwrap();
    }
    if updated {
        scan = scan.with_row_last_updated_at_version(None).unwrap();
    }
    let mut position = 1;
    loop {
        match scan.try_next().await.unwrap() {
            FullZipScanOutcome::Batch { rows, batch } => {
                assert_eq!(rows.start, position);
                position = rows.end;
                assert_batch(&batch.project(&[0, 1]).unwrap(), &rows, &expected);
                let versions = arrow_array::UInt64Array::from(vec![1; batch.num_rows()]);
                assert_eq!(
                    batch.num_columns(),
                    2 + usize::from(created) + usize::from(updated)
                );
                if created {
                    assert_eq!(batch[ROW_CREATED_AT_VERSION].to_data(), versions.to_data());
                }
                if updated {
                    assert_eq!(
                        batch[ROW_LAST_UPDATED_AT_VERSION].to_data(),
                        versions.to_data()
                    );
                }
                if created && updated {
                    assert!(Arc::ptr_eq(
                        &batch[ROW_CREATED_AT_VERSION],
                        &batch[ROW_LAST_UPDATED_AT_VERSION]
                    ));
                }
                let name = if created {
                    ROW_CREATED_AT_VERSION
                } else {
                    ROW_LAST_UPDATED_AT_VERSION
                };
                let alias = batch[name].slice(0, 1);
                let count = batch.num_rows();
                drop(batch);
                assert_eq!(budget.reserved(), count * 8);
                if position < 7 {
                    assert!(matches!(
                        scan.try_next().await.unwrap(),
                        FullZipScanOutcome::NeedsCapacity
                    ));
                    assert_eq!(scan.position(), position);
                }
                drop(alias);
                assert_eq!(budget.reserved(), 0);
            }
            FullZipScanOutcome::Finished => break,
            FullZipScanOutcome::NeedsCapacity => panic!("version columns must fit after drain"),
        }
    }
    assert_eq!(position, 7);
}

#[rstest]
#[tokio::test]
async fn version_credit_is_reserved_before_payload(#[values(false, true)] all_deleted: bool) {
    let (_fs, reader) = fixture_batches(
        ConcreteFileVersion::V2_3,
        false,
        vec![batches().remove(0)],
        1,
    )
    .await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(15);
    // Equal values from distinct snapshots must retain separate owner charges.
    let mut scan = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), 15, 7)
        .await
        .unwrap()
        .with_row_created_at_version(Some(Arc::new(Versions { rows: 7 })))
        .unwrap()
        .with_row_last_updated_at_version(Some(Arc::new(Versions { rows: 7 })))
        .unwrap();
    if all_deleted {
        scan = scan
            .with_deleted_rows(Arc::new((0_u32..7).collect::<DeletionVector>()))
            .unwrap();
    }
    let held = all_deleted.then(|| budget.try_reserve(15).unwrap());
    let outcome = scan.try_next().await.unwrap();
    if all_deleted {
        assert!(matches!(outcome, FullZipScanOutcome::Finished));
    } else {
        assert!(matches!(outcome, FullZipScanOutcome::NeedsCapacity));
    }
    assert_eq!(scan.position(), if all_deleted { 7 } else { 0 });
    assert_eq!(stats.snapshot().iops, 0);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::length("length", "sequence length")]
#[case::duplicate("duplicate", "already configured")]
#[case::started("started", "before the first read")]
#[case::merge("merge", "after merging")]
#[case::collision("collision", "conflicts with projected field")]
#[tokio::test]
async fn invalid_version_configuration_reads_no_payload(#[case] kind: &str, #[case] message: &str) {
    let mut batch = batches().remove(0);
    if kind == "collision" {
        let schema = Arc::new(Schema::new(vec![
            batch
                .schema()
                .field(0)
                .clone()
                .with_name(ROW_CREATED_AT_VERSION),
            batch.schema().field(1).clone(),
        ]));
        batch = RecordBatch::try_new(schema, batch.columns().to_vec()).unwrap();
    }
    let (_fs, reader) = fixture_batches(ConcreteFileVersion::V2_3, false, vec![batch], 1).await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(4096);
    let held = budget.try_reserve(4096).unwrap();
    let mut scan = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), 4096, 7)
        .await
        .unwrap();
    let error = match kind {
        "length" => scan
            .with_row_created_at_version(Some(Arc::new(Versions { rows: 6 })))
            .unwrap_err(),
        "duplicate" => scan
            .with_row_created_at_version(None)
            .unwrap()
            .with_row_created_at_version(None)
            .unwrap_err(),
        "started" => {
            assert!(matches!(
                scan.try_next().await.unwrap(),
                FullZipScanOutcome::NeedsCapacity
            ));
            scan.with_row_created_at_version(None).unwrap_err()
        }
        "merge" => {
            let other = reader
                .open_admitted_fullzip_scan(0..7, budget.clone(), 4096, 7)
                .await
                .unwrap();
            scan.with_row_created_at_version(None)
                .unwrap()
                .try_merge(other)
                .unwrap_err()
        }
        "collision" => scan.with_row_created_at_version(None).unwrap_err(),
        _ => unreachable!(),
    };
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 4096);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}
