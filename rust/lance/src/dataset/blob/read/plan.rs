// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
    sync::Arc,
};

use crate::dataset::blob::file::{BlobFile, BlobSource, BlobSourceKey};
use crate::dataset::blob::read::BlobReadRange;
use lance_core::{Error, Result};

/// One logical blob selected for planned reading.
#[derive(Debug)]
pub(in crate::dataset::blob) struct BlobEntry {
    pub(in crate::dataset::blob) selection_index: usize,
    pub(in crate::dataset::blob) row_address: u64,
    pub(in crate::dataset::blob) file: BlobFile,
    pub(in crate::dataset::blob) requested_range: Option<BlobReadRange>,
}

/// One resolved logical selection. Its position in the containing vector is
/// the caller's selection index.
#[derive(Debug)]
pub(super) struct ResolvedBlobSelection {
    pub(super) row_address: u64,
    pub(super) file: Option<BlobFile>,
    pub(super) requested_range: Option<BlobReadRange>,
}

/// Physical read input derived from one [`BlobEntry`].
#[derive(Debug)]
pub(super) struct PlannedBlobRead {
    pub(super) selection_index: usize,
    pub(super) row_address: u64,
    pub(super) requested_range: Option<BlobReadRange>,
    physical_range: Range<u64>,
}

/// A slice of one disjoint physical range submitted to the file scheduler.
#[derive(Debug)]
pub(super) struct PlannedBlobReadSlice {
    pub(super) read_index: usize,
    pub(super) physical_range_index: usize,
    pub(super) relative_range: Range<u64>,
}

/// One physical read paired with the backing source used to batch it.
#[derive(Debug)]
struct SourcePlannedBlobRead {
    source_key: BlobSourceKey,
    source: Arc<BlobSource>,
    read: PlannedBlobRead,
}

/// One per-source read plan emitted by `read_blobs`.
#[derive(Debug)]
pub(super) struct BlobReadPlan {
    source_key: BlobSourceKey,
    pub(super) source: Arc<BlobSource>,
    pub(super) reads: Vec<PlannedBlobRead>,
}

/// A payload-bounded group of per-source plans.
#[derive(Debug)]
pub(super) struct BlobReadBatch {
    pub(super) plans: Vec<BlobReadPlan>,
}

#[derive(Debug)]
struct CoalescedRangeInsertion {
    merged: Range<u64>,
    replaced_starts: Vec<u64>,
    replaced_bytes: u64,
}

impl CoalescedRangeInsertion {
    fn additional_bytes(&self) -> Result<u64> {
        let merged_bytes = self.merged.end - self.merged.start;
        merged_bytes.checked_sub(self.replaced_bytes).ok_or_else(|| {
            Error::internal(format!(
                "Coalesced blob range {:?} was smaller than its replaced ranges totaling {} bytes",
                self.merged, self.replaced_bytes
            ))
        })
    }
}

/// Physical ranges already charged to one source in the current batch.
#[derive(Debug, Default)]
struct CoalescedPhysicalRanges {
    ranges: BTreeMap<u64, u64>,
}

impl CoalescedPhysicalRanges {
    fn plan_insertion(
        &self,
        range: &Range<u64>,
        block_size: u64,
    ) -> Result<Option<CoalescedRangeInsertion>> {
        if range.is_empty() {
            return Ok(None);
        }

        let scan_start = self
            .ranges
            .range(..=range.start)
            .next_back()
            .map(|(start, _)| *start)
            .unwrap_or(range.start);
        let mut merged = range.clone();
        let mut replaced_starts = Vec::new();
        let mut replaced_bytes = 0_u64;
        for (&start, &end) in self.ranges.range(scan_start..) {
            if end.saturating_add(block_size) < merged.start {
                continue;
            }
            if start > merged.end.saturating_add(block_size) {
                break;
            }
            merged.start = merged.start.min(start);
            merged.end = merged.end.max(end);
            replaced_starts.push(start);
            replaced_bytes = replaced_bytes
                .checked_add(end - start)
                .ok_or_else(|| Error::internal("Coalesced blob range size overflow".to_string()))?;
        }

        Ok(Some(CoalescedRangeInsertion {
            merged,
            replaced_starts,
            replaced_bytes,
        }))
    }

    fn insert(&mut self, insertion: CoalescedRangeInsertion) {
        for start in insertion.replaced_starts {
            self.ranges.remove(&start);
        }
        self.ranges
            .insert(insertion.merged.start, insertion.merged.end);
    }
}

/// Exact scheduler-visible physical footprint of the current batch.
#[derive(Debug, Default)]
struct BlobReadBatchFootprint {
    sources: HashMap<BlobSourceKey, CoalescedPhysicalRanges>,
    physical_bytes: u64,
}

impl BlobReadBatchFootprint {
    fn try_insert(
        &mut self,
        planned: &SourcePlannedBlobRead,
        physical_buffer_size: u64,
        allow_oversized: bool,
    ) -> Result<bool> {
        let remaining = physical_buffer_size.saturating_sub(self.physical_bytes);
        let additional_bytes = {
            let source_ranges = self.sources.entry(planned.source_key.clone()).or_default();
            let insertion = source_ranges.plan_insertion(
                &planned.read.physical_range,
                planned.source.object_store.block_size() as u64,
            )?;
            let Some(insertion) = insertion else {
                return Ok(true);
            };
            let additional_bytes = insertion.additional_bytes()?;
            if !allow_oversized && additional_bytes > remaining {
                return Ok(false);
            }
            source_ranges.insert(insertion);
            additional_bytes
        };
        self.physical_bytes = self
            .physical_bytes
            .checked_add(additional_bytes)
            .ok_or_else(|| Error::internal("Blob read batch size overflow".to_string()))?;
        Ok(true)
    }
}

/// Split selected reads into request-order batches before grouping each batch
/// by physical source. Each batch accounts for the physical spans that the file
/// scheduler will read after coalescing nearby ranges.
pub(super) fn plan_blob_read_batches(
    mut entries: Vec<BlobEntry>,
    physical_buffer_size: u64,
) -> Result<Vec<BlobReadBatch>> {
    debug_assert!(physical_buffer_size > 0);
    entries.sort_by_key(|entry| entry.selection_index);
    let planned_reads = plan_blob_reads(entries)?;

    let mut batches = Vec::new();
    let mut current_reads = Vec::new();
    let mut footprint = BlobReadBatchFootprint::default();
    for planned in planned_reads {
        if !footprint.try_insert(&planned, physical_buffer_size, current_reads.is_empty())? {
            batches.push(into_blob_read_batch(std::mem::take(&mut current_reads)));
            footprint = BlobReadBatchFootprint::default();
            let inserted = footprint.try_insert(&planned, physical_buffer_size, true)?;
            debug_assert!(inserted, "an empty batch must accept one physical read");
        }
        current_reads.push(planned);
    }
    if !current_reads.is_empty() {
        batches.push(into_blob_read_batch(current_reads));
    }

    Ok(batches)
}

fn plan_blob_reads(entries: Vec<BlobEntry>) -> Result<Vec<SourcePlannedBlobRead>> {
    let mut planned_reads = Vec::with_capacity(entries.len());
    for entry in entries {
        let logical_range = match entry.requested_range {
            Some(range) => range.checked_range(entry.selection_index)?,
            None => 0..entry.file.size,
        };
        let physical_range = entry.file.read_phys_range(logical_range).map_err(|err| {
            Error::invalid_input(format!(
                "Blob range request {} for row address {} is invalid: {}",
                entry.selection_index, entry.row_address, err
            ))
        })?;
        let source_key = BlobSourceKey::new(&entry.file.source);
        planned_reads.push(SourcePlannedBlobRead {
            source_key,
            source: entry.file.source.clone(),
            read: PlannedBlobRead {
                selection_index: entry.selection_index,
                row_address: entry.row_address,
                requested_range: entry.requested_range,
                physical_range,
            },
        });
    }

    Ok(planned_reads)
}

fn into_blob_read_batch(planned_reads: Vec<SourcePlannedBlobRead>) -> BlobReadBatch {
    BlobReadBatch {
        plans: group_blob_read_plans(planned_reads),
    }
}

/// Group selected blobs by physical source and sort each group's ranges by
/// physical offset before handing them to the file scheduler.
fn group_blob_read_plans(planned_reads: Vec<SourcePlannedBlobRead>) -> Vec<BlobReadPlan> {
    let mut plan_indices = HashMap::<BlobSourceKey, usize>::new();
    let mut plans = Vec::<BlobReadPlan>::new();

    for planned in planned_reads {
        let plan_index = if let Some(plan_index) = plan_indices.get(&planned.source_key) {
            *plan_index
        } else {
            let plan_index = plans.len();
            plans.push(BlobReadPlan {
                source_key: planned.source_key.clone(),
                source: planned.source,
                reads: Vec::new(),
            });
            plan_indices.insert(planned.source_key, plan_index);
            plan_index
        };
        plans[plan_index].reads.push(planned.read);
    }

    plans.sort_by(|left, right| {
        left.source_key
            .store_prefix
            .cmp(&right.source_key.store_prefix)
            .then_with(|| left.source_key.path.cmp(&right.source_key.path))
    });

    for plan in &mut plans {
        plan.reads.sort_by(|left, right| {
            left.physical_range
                .start
                .cmp(&right.physical_range.start)
                .then_with(|| left.physical_range.end.cmp(&right.physical_range.end))
                .then_with(|| left.selection_index.cmp(&right.selection_index))
        });
    }

    plans
}

pub(super) fn plan_blob_read_plans(entries: Vec<BlobEntry>) -> Result<Vec<BlobReadPlan>> {
    Ok(group_blob_read_plans(plan_blob_reads(entries)?))
}

/// Merge overlapping physical reads before submitting them to [`FileScheduler`](lance_io::scheduler::FileScheduler).
///
/// `FileScheduler` can safely coalesce and split disjoint ranges. Original
/// overlapping ranges need to be mapped onto their union first so a split does
/// not advance past the start of a later nested range while reconstructing the
/// caller's buffers.
pub(super) fn plan_disjoint_blob_reads(
    reads: &[PlannedBlobRead],
) -> (Vec<Range<u64>>, Vec<PlannedBlobReadSlice>) {
    let mut non_empty_ranges = reads
        .iter()
        .enumerate()
        .filter(|(_, read)| !read.physical_range.is_empty())
        .map(|(read_index, read)| (read_index, read.physical_range.clone()))
        .collect::<Vec<_>>();
    non_empty_ranges.sort_by_key(|(read_index, range)| (range.start, range.end, *read_index));

    let mut physical_ranges = Vec::<Range<u64>>::with_capacity(non_empty_ranges.len());
    let mut slices = Vec::with_capacity(non_empty_ranges.len());
    for (read_index, range) in non_empty_ranges {
        let physical_range_index = match physical_ranges.last_mut() {
            Some(physical_range) if range.start <= physical_range.end => {
                physical_range.end = physical_range.end.max(range.end);
                physical_ranges.len() - 1
            }
            _ => {
                physical_ranges.push(range.clone());
                physical_ranges.len() - 1
            }
        };
        let physical_start = physical_ranges[physical_range_index].start;
        slices.push(PlannedBlobReadSlice {
            read_index,
            physical_range_index,
            relative_range: range.start - physical_start..range.end - physical_start,
        });
    }

    (physical_ranges, slices)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dataset::blob::file::{BlobFile, BlobSource};
    use crate::dataset::blob::read::{
        BlobReadRange, ReadBlobsExecution, execute_blob_read_batches_stream,
    };
    use crate::dataset::blob::test_util::{
        RecordingRangeObjectStore, recording_range_store,
        recording_range_store_with_url_and_block_size,
    };
    use bytes::Bytes;
    use futures::TryStreamExt;
    use lance_core::datatypes::BlobKind;
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use std::sync::Arc;

    fn recording_range_store_with_block_size(
        data: Bytes,
        block_size: usize,
    ) -> (Arc<ObjectStore>, Arc<RecordingRangeObjectStore>) {
        recording_range_store_with_url_and_block_size(
            data,
            "mock://recording/blob-range-tests",
            Some(block_size),
        )
    }

    #[test]
    fn test_blob_read_batches_bound_physical_bytes() {
        let (store, _) = recording_range_store(Bytes::from_static(b"abcdefghij"));
        let source = Arc::new(BlobSource::new(store, Path::from("blobs/test.bin")));
        let entries = vec![
            BlobEntry {
                selection_index: 1,
                row_address: 11,
                file: BlobFile::with_source(source.clone(), 4, 6, BlobKind::Packed, None),
                requested_range: Some(BlobReadRange::new(0, 5)),
            },
            BlobEntry {
                selection_index: 0,
                row_address: 10,
                file: BlobFile::with_source(source, 1, 3, BlobKind::Packed, None),
                requested_range: Some(BlobReadRange::new(0, 3)),
            },
        ];

        let batches = plan_blob_read_batches(entries, 4).unwrap();

        assert_eq!(batches.len(), 2);
        let selection_indices = batches
            .iter()
            .map(|batch| {
                batch
                    .plans
                    .iter()
                    .flat_map(|plan| plan.reads.iter())
                    .map(|read| read.selection_index)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(selection_indices, vec![vec![0], vec![1]]);
    }

    #[tokio::test]
    async fn test_blob_read_batches_bound_coalesced_physical_bytes() {
        const BLOCK_SIZE: usize = 64 * 1024;
        const RANGE_GAP: u64 = 32 * 1024;
        const REQUEST_COUNT: usize = 4;

        let mut data = vec![0_u8; 128 * 1024];
        for request_index in 0..REQUEST_COUNT {
            data[request_index * RANGE_GAP as usize] = request_index as u8 + 1;
        }
        let (store, inner) = recording_range_store_with_block_size(Bytes::from(data), BLOCK_SIZE);
        let source = Arc::new(BlobSource::new(store, Path::from("blobs/dense.pack")));
        let entries = (0..REQUEST_COUNT)
            .map(|request_index| BlobEntry {
                selection_index: request_index,
                row_address: request_index as u64,
                file: BlobFile::with_source(
                    source.clone(),
                    request_index as u64 * RANGE_GAP,
                    1,
                    BlobKind::Packed,
                    None,
                ),
                requested_range: Some(BlobReadRange::new(0, 1)),
            })
            .collect();

        let batches = plan_blob_read_batches(entries, REQUEST_COUNT as u64).unwrap();
        assert_eq!(batches.len(), REQUEST_COUNT);

        let results = execute_blob_read_batches_stream(
            batches,
            Arc::new(ReadBlobsExecution::new(Some(REQUEST_COUNT as u64))),
            REQUEST_COUNT,
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert_eq!(results.len(), REQUEST_COUNT);
        for (request_index, result) in results.iter().enumerate() {
            assert_eq!(result.selection_index, request_index);
            assert_eq!(result.data.as_ref(), &[request_index as u8 + 1]);
        }
        assert_eq!(
            inner.requested_blob_ranges(),
            (0..REQUEST_COUNT)
                .map(|request_index| {
                    let start = request_index as u64 * RANGE_GAP;
                    start..start + 1
                })
                .collect::<Vec<_>>()
        );
    }
}
