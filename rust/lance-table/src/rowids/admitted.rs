// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keep table row ID encodings above the file reader's admission boundary.

use std::ops::Range;

use lance_core::{Error, Result};
use lance_file::reader::{PhysicalRowIdSource, PhysicalRowVersionSource};

use super::RowIdSequence;
use super::version::RowDatasetVersionSequence;

impl PhysicalRowVersionSource for RowDatasetVersionSequence {
    fn num_rows(&self) -> u64 {
        self.len()
    }

    fn select_range(&self, rows: Range<usize>) -> Result<Box<dyn Iterator<Item = u64> + '_>> {
        if rows.start > rows.end || rows.end as u64 > self.len() {
            return Err(Error::invalid_input(
                "version range exceeds physical sequence length",
            ));
        }
        // Skip whole runs, not individual rows. Empty runs emit no value; the
        // span's cardinality determines positional length, not its stored labels.
        let mut skip = rows.start;
        let values = self
            .runs
            .iter()
            .flat_map(move |run| {
                let len = run.len();
                let offset = skip.min(len);
                skip -= offset;
                std::iter::repeat_n(run.version(), len - offset)
            })
            .take(rows.len());
        Ok(Box::new(values))
    }
}

impl PhysicalRowIdSource for RowIdSequence {
    fn num_rows(&self) -> u64 {
        self.len()
    }

    fn select_range(&self, rows: Range<usize>) -> Result<Box<dyn Iterator<Item = u64> + '_>> {
        if rows.start > rows.end || rows.end as u64 > self.len() {
            return Err(Error::invalid_input(format!(
                "row ID range {}..{} exceeds physical sequence length {}",
                rows.start,
                rows.end,
                self.len()
            )));
        }
        // select maintains a segment cursor over immutable metadata. Do not use
        // select_range_with_cursor here: it collects a new, unadmitted Vec.
        Ok(Box::new(self.select(rows)))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::{
        ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION,
        cache::LanceCache,
        utils::{allocation::AllocationBudget, deletion::DeletionVector},
    };
    use lance_file::{
        reader::{FileReader, FileReaderOptions, FullZipScanOutcome},
        testing::{FsFixture, write_lance_file},
        version::ConcreteFileVersion,
        writer::FileWriterOptions,
    };
    use lance_io::{
        scheduler::{ScanScheduler, SchedulerConfig},
        utils::CachedFileSize,
    };
    use rstest::rstest;

    use super::*;
    use crate::rowids::{segment::U64Segment, version::RowDatasetVersionRun};

    #[rstest]
    #[tokio::test]
    async fn admitted_row_ids_preserve_real_sequence_and_allocation(
        #[values(ConcreteFileVersion::V2_2, ConcreteFileVersion::V2_3)]
        version: ConcreteFileVersion,
        #[values(false, true)] lite: bool,
        #[values(false, true)] multiple_segments: bool,
        #[values("defaults", "mixed", "distinct", "shared")] versions: &str,
    ) {
        let expected_ids = (0..35_u64)
            .map(|row| {
                if multiple_segments && row >= 9 {
                    u64::MAX - 7 - row * 19
                } else {
                    900 + row * 3
                }
            })
            .collect::<Vec<_>>();
        let mut sequence = RowIdSequence::new();
        if multiple_segments {
            for chunk in expected_ids.chunks(7) {
                sequence.extend(RowIdSequence::try_from_iter(chunk.iter().copied()).unwrap());
            }
        } else {
            sequence = RowIdSequence::try_from_iter(expected_ids.iter().copied()).unwrap();
        }
        let source: Arc<dyn PhysicalRowIdSource> = Arc::new(sequence);
        assert_eq!(source.num_rows(), 35);
        for bounds in [35..35, 3..3, 6..17] {
            let actual = source
                .select_range(bounds.clone())
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(actual, expected_ids[bounds]);
        }
        for (start, end) in [(0, 36), (36, 36), (7, 3)] {
            let error = source.select_range(start..end).err().unwrap();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("physical sequence length"));
        }
        let created_values = [vec![3; 9], vec![8; 5], vec![2; 21]].concat();
        let updated_values = [vec![10; 4], vec![11; 15], vec![10; 16]].concat();
        let make_versions = |values: &[u64]| {
            let mut runs = vec![RowDatasetVersionRun {
                span: U64Segment::Range(0..0),
                version: 99,
            }];
            let mut start = 0;
            while start < values.len() {
                let end = start
                    + values[start..]
                        .iter()
                        .take_while(|v| **v == values[start])
                        .count();
                runs.push(RowDatasetVersionRun {
                    span: U64Segment::Range(start as u64..end as u64),
                    version: values[start],
                });
                runs.push(RowDatasetVersionRun {
                    span: U64Segment::Range(end as u64..end as u64),
                    version: 99,
                });
                start = end;
            }
            Arc::new(RowDatasetVersionSequence { runs }) as Arc<dyn PhysicalRowVersionSource>
        };
        let created_source = make_versions(&created_values);
        let updated_source = make_versions(&updated_values);
        for (source, values) in [
            (&created_source, &created_values),
            (&updated_source, &updated_values),
        ] {
            for bounds in [0..0, 35..35, 3..18, 8..34] {
                assert_eq!(
                    source
                        .select_range(bounds.clone())
                        .unwrap()
                        .collect::<Vec<_>>(),
                    values[bounds]
                );
            }
            for (start, end) in [(0, 36), (36, 36), (7, 3)] {
                let error = source.select_range(start..end).err().unwrap();
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(error.to_string().contains("physical sequence length"));
            }
        }
        let (created_source, updated_source, created_values, updated_values) = match versions {
            "defaults" => (None, None, vec![1; 35], vec![1; 35]),
            "mixed" => (None, Some(updated_source), vec![1; 35], updated_values),
            "shared" => (
                Some(created_source.clone()),
                Some(created_source),
                created_values.clone(),
                created_values,
            ),
            "distinct" => (
                Some(created_source),
                Some(updated_source),
                created_values,
                updated_values,
            ),
            _ => unreachable!(),
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("payload", DataType::Utf8, false).with_metadata(HashMap::from([(
                "lance-encoding:structural-encoding".to_string(),
                "fullzip".to_string(),
            )])),
        ]));
        let data = (0..5)
            .map(|page| {
                let payload = StringArray::from_iter_values(
                    (page * 7..page * 7 + 7).map(|row| format!("row-{row:03}")),
                );
                RecordBatch::try_new(schema.clone(), vec![Arc::new(payload)]).unwrap()
            })
            .collect::<Vec<_>>();
        let mut fs = FsFixture::default();
        fs.scheduler = ScanScheduler::new(
            fs.object_store.clone(),
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
                use_lite_scheduler: Some(lite),
            },
        );
        write_lance_file(
            RecordBatchIterator::new(data.into_iter().map(Ok), schema),
            &fs,
            version,
            FileWriterOptions {
                data_cache_bytes: Some(1),
                ..Default::default()
            },
        )
        .await;
        let scheduler = fs
            .scheduler
            .open_file(&fs.tmp_path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let reader = FileReader::try_open(
            scheduler,
            None,
            Default::default(),
            &Arc::new(LanceCache::no_cache()),
            FileReaderOptions::default(),
        )
        .await
        .unwrap();
        let offsets = [0, 1, 6, 7, 13, 14, 20, 27, 33, 34];
        let deleted = Arc::new(DeletionVector::Bitmap(offsets.into_iter().collect()));
        let live = (0..35_u64)
            .filter(|row| !offsets.contains(&(*row as u32)))
            .collect::<Vec<_>>();
        let budget = AllocationBudget::new(4096);
        let mut scan = reader
            .open_admitted_fullzip_scan(0..0, budget.clone(), 4096, 8)
            .await
            .unwrap()
            .with_deleted_rows(deleted)
            .unwrap()
            .with_row_created_at_version(created_source)
            .unwrap()
            .with_row_id(23, Some(source))
            .unwrap()
            .with_row_last_updated_at_version(updated_source)
            .unwrap()
            .with_row_address(23)
            .unwrap();
        for selection in [1..7, 9..live.len() as u64] {
            scan.advance_to_live_range(selection.clone()).unwrap();
            let mut actual = Vec::new();
            loop {
                match scan.try_next().await.unwrap() {
                    FullZipScanOutcome::Batch { rows, batch } => {
                        actual.extend(rows.clone());
                        let ids = UInt64Array::from(
                            expected_ids[rows.start as usize..rows.end as usize].to_vec(),
                        );
                        let addresses = UInt64Array::from_iter_values(
                            rows.clone().map(|row| 23_u64 * (1_u64 << 32) + row),
                        );
                        let payload = StringArray::from_iter_values(
                            rows.clone().map(|row| format!("row-{row:03}")),
                        );
                        assert_eq!(batch[ROW_ID].to_data(), ids.to_data());
                        assert_eq!(batch[ROW_ADDR].to_data(), addresses.to_data());
                        assert_eq!(batch["payload"].to_data(), payload.to_data());
                        let created = UInt64Array::from(
                            created_values[rows.start as usize..rows.end as usize].to_vec(),
                        );
                        let updated = UInt64Array::from(
                            updated_values[rows.start as usize..rows.end as usize].to_vec(),
                        );
                        assert_eq!(batch[ROW_CREATED_AT_VERSION].to_data(), created.to_data());
                        assert_eq!(
                            batch[ROW_LAST_UPDATED_AT_VERSION].to_data(),
                            updated.to_data()
                        );
                        assert_eq!(batch.schema().field(3).name(), ROW_LAST_UPDATED_AT_VERSION);
                        assert_eq!(batch.schema().field(4).name(), ROW_CREATED_AT_VERSION);
                        assert_eq!(
                            Arc::ptr_eq(
                                &batch[ROW_CREATED_AT_VERSION],
                                &batch[ROW_LAST_UPDATED_AT_VERSION]
                            ),
                            matches!(versions, "defaults" | "shared")
                        );
                        let alias = batch[ROW_CREATED_AT_VERSION].slice(0, 1);
                        let id_alias = batch[ROW_ID].slice(0, 1);
                        let count = batch.num_rows();
                        drop(batch);
                        assert_eq!(budget.reserved(), count * 16);
                        drop(id_alias);
                        assert_eq!(budget.reserved(), count * 8);
                        if actual.len() < (selection.end - selection.start) as usize {
                            assert!(matches!(
                                scan.try_next().await.unwrap(),
                                FullZipScanOutcome::NeedsCapacity
                            ));
                            assert_eq!(scan.position(), rows.end);
                        }
                        drop(alias);
                        assert_eq!(budget.reserved(), 0);
                    }
                    FullZipScanOutcome::Finished => break,
                    FullZipScanOutcome::NeedsCapacity => {
                        panic!("row payload and IDs must fit after drain")
                    }
                }
            }
            assert_eq!(
                actual,
                live[selection.start as usize..selection.end as usize]
            );
        }
    }
}
