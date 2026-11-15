// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::reader::PhysicalRowIdSource;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use lance_core::{ROW_ADDR, ROW_ID, utils::address::RowAddress};

use super::*;

#[derive(Debug)]
struct RowIds(Vec<u64>);

impl PhysicalRowIdSource for RowIds {
    fn num_rows(&self) -> u64 {
        self.0.len() as u64
    }

    fn select_range(&self, rows: Range<usize>) -> Result<Box<dyn Iterator<Item = u64> + '_>> {
        let values = self
            .0
            .get(rows)
            .ok_or_else(|| Error::invalid_input("invalid test row range"))?;
        Ok(Box::new(values.iter().copied()))
    }
}

fn stable_ids(count: usize) -> (Vec<u64>, Arc<dyn PhysicalRowIdSource>) {
    // Non-contiguous, non-monotonic values distinguish
    // physical offsets from stable identities across deletion/range boundaries.
    let ids = (0..count)
        .map(|i| {
            if i < 9 {
                900 + i as u64 * 3
            } else {
                u64::MAX - 7 - i as u64 * 19
            }
        })
        .collect::<Vec<_>>();
    let sequence = Arc::new(RowIds(ids.clone()));
    (ids, sequence)
}

#[derive(Debug)]
struct WrongCardinality {
    extra: bool,
}

impl PhysicalRowIdSource for WrongCardinality {
    fn num_rows(&self) -> u64 {
        7
    }

    fn select_range(&self, rows: Range<usize>) -> Result<Box<dyn Iterator<Item = u64> + '_>> {
        let count = if self.extra {
            rows.len() + 1
        } else {
            rows.len() - 1
        };
        Ok(Box::new(0..count as u64))
    }
}

#[rstest]
#[tokio::test]
async fn row_source_cardinality_errors_release_complete_attempt(
    #[values(false, true)] extra: bool,
) {
    let (_fs, reader) = fixture_batches(
        ConcreteFileVersion::V2_3,
        false,
        vec![batches().remove(0)],
        1,
    )
    .await;
    let budget = AllocationBudget::new(4096);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..3, budget.clone(), 4096, 3)
        .await
        .unwrap()
        .with_row_id(7, Some(Arc::new(WrongCardinality { extra })))
        .unwrap();
    let error = scan.try_next().await.unwrap_err();
    if extra {
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("exceeded physical selection"));
    } else {
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("ended before physical selection")
        );
    }
    assert_eq!(scan.position(), 0);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::address(false, true, false)]
#[case::stable(true, false, true)]
#[case::address_id(true, false, false)]
#[case::both_stable(true, true, true)]
#[case::both_address(true, true, false)]
#[tokio::test]
async fn row_columns_follow_physical_selection_and_owner_lifetime(
    #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)] version: ConcreteFileVersion,
    #[values(false, true)] lite: bool,
    #[case] has_id: bool,
    #[case] has_address: bool,
    #[case] is_stable: bool,
) {
    let input = batches();
    let expected = arrow_select::concat::concat_batches(&input[0].schema(), &input[..5]).unwrap();
    let labels = input[..5]
        .iter()
        .map(|batch| batch.project(&[0]).unwrap())
        .collect();
    let frames = input[..5]
        .iter()
        .map(|batch| batch.project(&[1]).unwrap())
        .collect();
    let (_left_fs, left) = fixture_batches(version, lite, labels, 1).await;
    let (_right_fs, right) = fixture_batches(version, !lite, frames, 1).await;
    let stats = IoStats::new();
    let left = left.with_io_stats(stats.recorder());
    let right = right.with_io_stats(stats.recorder());
    let offsets = [0, 1, 6, 7, 13, 14, 20, 27, 33, 34];
    let live = (0..expected.num_rows() as u64)
        .filter(|row| !offsets.contains(&(*row as u32)))
        .collect::<Vec<_>>();
    let deleted = Arc::new(DeletionVector::Bitmap(offsets.into_iter().collect()));
    let (ids, sequence) = stable_ids(expected.num_rows());
    let budget = AllocationBudget::new(4096);
    let a = left
        .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 8)
        .await
        .unwrap();
    let b = right
        .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 8)
        .await
        .unwrap();
    let mut scan = a.try_merge(b).unwrap().with_deleted_rows(deleted).unwrap();
    // Deliberately configure in reverse schema order.
    if has_address {
        scan = scan.with_row_address(23).unwrap();
    }
    if has_id {
        scan = scan.with_row_id(23, is_stable.then_some(sequence)).unwrap();
    }
    assert_eq!(stats.snapshot().iops, 0);
    for selection in [1..7, 9..live.len() as u64] {
        scan.advance_to_live_range(selection.clone()).unwrap();
        let mut actual = Vec::new();
        loop {
            match scan.try_next().await.unwrap() {
                FullZipScanOutcome::Batch { rows, batch } => {
                    assert_batch(&batch.project(&[0, 1]).unwrap(), &rows, &expected);
                    actual.extend(rows.clone());
                    let expected_addr = rows
                        .clone()
                        .map(|row| 23_u64 * (1_u64 << 32) + row)
                        .collect::<Vec<_>>();
                    if has_id {
                        let expected_id = if is_stable {
                            ids[rows.start as usize..rows.end as usize].to_vec()
                        } else {
                            expected_addr.clone()
                        };
                        assert_eq!(
                            batch[ROW_ID].as_primitive::<UInt64Type>().values().as_ref(),
                            expected_id
                        );
                        assert_eq!(batch.schema().field(2).name(), ROW_ID);
                    }
                    if has_address {
                        assert_eq!(
                            batch[ROW_ADDR]
                                .as_primitive::<UInt64Type>()
                                .values()
                                .as_ref(),
                            expected_addr
                        );
                    }
                    if has_id && has_address && !is_stable {
                        assert!(Arc::ptr_eq(&batch[ROW_ID], &batch[ROW_ADDR]));
                    }
                    let name = if has_id { ROW_ID } else { ROW_ADDR };
                    let alias = batch[name].slice(0, 1);
                    let sibling = has_address.then(|| batch[ROW_ADDR].clone());
                    let row_count = batch.num_rows();
                    assert!(budget.reserved() <= 4096);
                    drop(batch);
                    drop(sibling);
                    // One visible row pins the full physical ID allocation.
                    assert_eq!(budget.reserved(), row_count * 8);
                    let before = stats.snapshot().iops;
                    if actual.len() < (selection.end - selection.start) as usize {
                        assert!(matches!(
                            scan.try_next().await.unwrap(),
                            FullZipScanOutcome::NeedsCapacity
                        ));
                        assert_eq!(scan.position(), rows.end);
                    }
                    assert_eq!(stats.snapshot().iops, before);
                    drop(alias);
                    assert_eq!(budget.reserved(), 0);
                }
                FullZipScanOutcome::Finished => break,
                FullZipScanOutcome::NeedsCapacity => {
                    panic!("complete row with IDs must fit after drain")
                }
            }
        }
        assert_eq!(
            actual,
            live[selection.start as usize..selection.end as usize]
        );
    }
}

#[rstest]
#[tokio::test]
async fn row_column_credit_refuses_before_payload_io(
    #[values(false, true)] lite: bool,
    #[values(false, true)] all_deleted: bool,
) {
    let input = batches().remove(0);
    let (_, ids) = stable_ids(input.num_rows());
    let (_fs, reader) = fixture_batches(ConcreteFileVersion::V2_3, lite, vec![input], 1).await;
    let stats = IoStats::new();
    let reader = reader.with_io_stats(stats.recorder());
    // Two distinct UInt64 columns cannot fit even one row. No payload request
    // should be launched while repeatedly reducing the candidate prefix.
    let budget = AllocationBudget::new(15);
    let mut scan = reader
        .open_admitted_fullzip_scan(0..7, budget.clone(), 15, 7)
        .await
        .unwrap()
        .with_row_id(4, Some(ids))
        .unwrap()
        .with_row_address(4)
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
        assert_eq!(scan.position(), 7);
    } else {
        assert!(matches!(outcome, FullZipScanOutcome::NeedsCapacity));
        assert_eq!(scan.position(), 0);
    }
    drop(held);
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::short_sequence("short", "sequence length")]
#[case::fragment_mismatch("fragment", "same fragment")]
#[case::duplicate("duplicate", "already configured")]
#[case::tombstone("tombstone", "valid fragment ID")]
#[case::after_attempt("started", "before the first read")]
#[case::after_logical_selection("mapped", "before the first read")]
#[case::merge_after_configuration("merge", "after merging")]
#[case::field_collision("collision", "conflicts with projected field")]
#[tokio::test]
async fn invalid_row_column_configuration_reads_no_payload(
    #[case] kind: &str,
    #[case] message: &str,
) {
    let mut batch = batches().remove(0);
    if kind == "collision" {
        let schema = Arc::new(Schema::new(vec![
            batch.schema().field(0).clone().with_name(ROW_ID),
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
        "short" => scan.with_row_id(4, Some(stable_ids(6).1)).unwrap_err(),
        "fragment" => scan
            .with_row_address(4)
            .unwrap()
            .with_row_id(5, None)
            .unwrap_err(),
        "duplicate" => scan
            .with_row_id(4, None)
            .unwrap()
            .with_row_id(4, None)
            .unwrap_err(),
        "tombstone" => scan
            .with_row_address(RowAddress::TOMBSTONE_FRAG)
            .unwrap_err(),
        "started" => {
            assert!(matches!(
                scan.try_next().await.unwrap(),
                FullZipScanOutcome::NeedsCapacity
            ));
            scan.with_row_address(4).unwrap_err()
        }
        "mapped" => {
            let mut scan = reader
                .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 7)
                .await
                .unwrap();
            scan.advance_to_live_range(1..3).unwrap();
            scan.with_row_address(4).unwrap_err()
        }
        "merge" => {
            let second = reader
                .open_admitted_fullzip_scan(0..7, budget.clone(), 4096, 7)
                .await
                .unwrap();
            scan.with_row_address(4)
                .unwrap()
                .try_merge(second)
                .unwrap_err()
        }
        "collision" => scan.with_row_id(4, None).unwrap_err(),
        _ => unreachable!(),
    };
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 4096);
    drop(held);
    assert_eq!(budget.reserved(), 0);
}
