// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared Zone Training Utilities
//!
//! This module provides common infrastructure for building zone-based scalar indexes.
//! It handles chunking data streams into fixed-size zones while respecting fragment
//! boundaries and computing zone bounds that remain valid after row deletions.

use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use datafusion::execution::SendableRecordBatchStream;
use futures::{TryStreamExt, stream};
use lance_core::error::Error;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::{ROW_ADDR, Result};
use lance_select::RowAddrTreeMap;
use std::sync::Arc;

/// Minimum amount of row-based zone work dispatched to the CPU pool at once.
///
/// This is a floor, not a target: zones are never divided, so a split holding a
/// single zone larger than this is dispatched as-is. Only zones smaller than the
/// floor are grouped together, which is the point of batching.
const MIN_ROWS_PER_SPLIT: usize = 8192;

//
// Example: Suppose we have two fragments, each with 4 rows.
// Fragment 0: start = 0, length = 4  // covers rows 0, 1, 2, 3 in fragment 0
// The row addresses for fragment 0 are: 0, 1, 2, 3
// Fragment 1: start = 0, length = 4  // covers rows 0, 1, 2, 3 in fragment 1
// The row addresses for fragment 1 are: (1<<32), (1<<32)+1, (1<<32)+2, (1<<32)+3
//
// Deletion is 0 index based. We delete the 0th and 1st row in fragment 0,
// and the 1st and 2nd row in fragment 1,
// Fragment 0: start = 2, length = 2 // covers rows 2, 3 in fragment 0
// The row addresses for fragment 0 are: 2, 3
// Fragment 1: start = 0, length = 4  // covers rows 0, 3 in fragment 1
// The row addresses for fragment 1 are: (1<<32), (1<<32)+3
/// Zone bound within a fragment
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneBound {
    pub fragment_id: u64,
    // start is start row of the zone in the fragment, also known
    // as the local offset. To get the actual first row address,
    // use `(fragment_id << 32) | start`.
    pub start: u64,
    // length is the span of row offsets between the first and last row in the zone,
    // calculated as (last_row_offset - first_row_offset + 1). It is not the count
    // of physical rows, since deletions may create gaps within the span.
    pub length: usize,
}

/// Index-specific logic used while building zones.
pub trait ZoneProcessor {
    type ZoneStatistics;

    /// Process a slice of values that belongs to the current zone.
    fn process_chunk(&mut self, values: &ArrayRef) -> Result<()>;

    /// Emit statistics when the zone is full or the fragment changes.
    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics>;

    /// Reset state so the processor can handle the next zone.
    fn reset(&mut self) -> Result<()>;
}

/// Trainer that handles chunking, fragment boundaries, and zone flushing.
pub struct ZoneTrainer<P> {
    processor_factory: Arc<dyn Fn() -> Result<P> + Send + Sync>,
    zone_capacity: u64,
}

impl<P> ZoneTrainer<P>
where
    P: ZoneProcessor + 'static,
    P::ZoneStatistics: Send + 'static,
{
    /// Create a new trainer that buffers at most `zone_capacity` rows per zone.
    ///
    /// `processor_factory` builds a fresh processor per zone so that zone
    /// statistics can be computed independently; it must be pure CPU.
    pub fn new(
        processor_factory: impl Fn() -> Result<P> + Send + Sync + 'static,
        zone_capacity: u64,
    ) -> Result<Self> {
        if zone_capacity == 0 {
            return Err(Error::invalid_input(
                "zone capacity must be greater than zero",
            ));
        }
        processor_factory()?;
        Ok(Self {
            processor_factory: Arc::new(processor_factory),
            zone_capacity,
        })
    }

    /// Consume the `_rowaddr`-annotated stream, split it into zones, and let the
    /// processor compute zone statistics.
    ///
    /// The caller must provide record batches where the first column is the
    /// value array that the zone processor understands, and the schema includes
    /// the `_rowaddr` column with physical row addresses. Future zone-based
    /// indexes should maintain this ordering or extend the trainer to accept an
    /// explicit column index.
    pub async fn train(
        self,
        stream: SendableRecordBatchStream,
    ) -> Result<(Vec<P::ZoneStatistics>, RowAddrTreeMap)> {
        let zone_size = usize::try_from(self.zone_capacity).map_err(|_| {
            Error::invalid_input("zone capacity does not fit into usize on this platform")
        })?;

        let splits = stream::try_unfold(
            ZoneSplitAssembler::new(stream, zone_size),
            |mut assembler| async move {
                Ok(assembler
                    .next_split()
                    .await?
                    .map(|split| (split, assembler)))
            },
        );
        let processor_factory = self.processor_factory;
        let split_tasks = splits.map_ok(move |split| {
            let processor_factory = processor_factory.clone();
            spawn_cpu(move || process_zone_split(&*processor_factory, split))
        });
        let results = split_tasks.try_buffered(get_num_compute_intensive_cpus());
        futures::pin_mut!(results);

        let mut zones = Vec::new();
        let mut null_rows = RowAddrTreeMap::new();
        while let Some((mut split_zones, split_null_rows)) = results.try_next().await? {
            zones.append(&mut split_zones);
            null_rows |= &split_null_rows;
        }
        Ok((zones, null_rows))
    }
}

type Zone = Vec<RecordBatch>;
type ZoneSplit = Vec<Zone>;

/// Serial accumulator that lazily groups the training stream into CPU-sized splits.
///
/// Zones never span fragments and hold at most `zone_size` physical rows; a
/// zone may span several input batches, collected as zero-copy slices.
struct ZoneSplitAssembler {
    stream: SendableRecordBatchStream,
    zone_size: usize,
    input_batch: Option<RecordBatch>,
    input_offset: usize,
    /// Zero-copy slices of the zone currently being assembled.
    current_zone: Zone,
    /// Physical rows accumulated so far in the current zone.
    current_zone_rows: usize,
    current_fragment: Option<u64>,
    current_split: ZoneSplit,
    current_split_rows: usize,
}

impl ZoneSplitAssembler {
    fn new(stream: SendableRecordBatchStream, zone_size: usize) -> Self {
        Self {
            stream,
            zone_size,
            input_batch: None,
            input_offset: 0,
            current_zone: Vec::new(),
            current_zone_rows: 0,
            current_fragment: None,
            current_split: Vec::new(),
            current_split_rows: 0,
        }
    }

    /// Produces the next bounded unit of zone work without eagerly draining the input.
    ///
    /// Each call advances `stream` only until enough *complete* zones have been
    /// collected to reach `MIN_ROWS_PER_SPLIT`. That constant is a floor, not a
    /// target: a zone is never divided between splits, so a single zone larger
    /// than the floor is returned on its own and the split may hold far more rows
    /// than the floor. Within a split, zones stay in input order, contain at most
    /// `zone_size` physical rows, and never cross a fragment boundary.
    ///
    /// A large input batch may therefore be consumed over several calls. The
    /// unread suffix remains in `input_batch` and `input_offset`, which bounds
    /// both queued CPU work and retained intermediate state. At end of stream,
    /// the trailing partial zone is included in the final split. This method
    /// returns `Some` only for non-empty splits and returns `None` once all input
    /// has been consumed.
    async fn next_split(&mut self) -> Result<Option<ZoneSplit>> {
        loop {
            // Return once the split contains enough complete zones.
            if self.current_split_rows >= MIN_ROWS_PER_SPLIT {
                return Ok(Some(self.take_split()));
            }

            // Read a batch only when the current one is exhausted.
            if self.input_batch.is_none() {
                let Some(batch) = self.stream.try_next().await? else {
                    // Flush trailing work at EOF.
                    self.finish_zone()?;
                    return Ok((!self.current_split.is_empty()).then(|| self.take_split()));
                };
                if batch.num_rows() == 0 {
                    continue;
                }
                row_addr_column(&batch)?;
                self.input_batch = Some(batch);
                self.input_offset = 0;
            }

            let input_rows = self
                .input_batch
                .as_ref()
                .map(RecordBatch::num_rows)
                .ok_or_else(|| Error::internal("zone split assembler lost its input batch"))?;
            if self.input_offset == input_rows {
                self.input_batch = None;
                self.input_offset = 0;
                continue;
            }

            let fragment_id = {
                let batch = self
                    .input_batch
                    .as_ref()
                    .ok_or_else(|| Error::internal("zone split assembler lost its input batch"))?;
                row_addr_column(batch)?.value(self.input_offset) >> 32
            };
            // A zone cannot cross a fragment boundary.
            if self
                .current_fragment
                .is_some_and(|current| current != fragment_id)
            {
                self.finish_zone()?;
                continue;
            }
            self.current_fragment = Some(fragment_id);

            let batch = self
                .input_batch
                .as_ref()
                .ok_or_else(|| Error::internal("zone split assembler lost its input batch"))?;
            let row_addrs = row_addr_column(batch)?.values();
            let remaining_rows = batch.num_rows() - self.input_offset;
            let take = remaining_rows.min(self.zone_size - self.current_zone_rows);
            let limit = self.input_offset + take;
            // Stop at either the zone limit or the next fragment.
            let end = row_addrs[self.input_offset..limit]
                .iter()
                .position(|addr| (addr >> 32) != fragment_id)
                .map_or(limit, |run_len| self.input_offset + run_len);
            let len = end - self.input_offset;
            self.current_zone.push(batch.slice(self.input_offset, len));
            self.current_zone_rows += len;
            self.input_offset = end;
            if self.current_zone_rows == self.zone_size {
                self.finish_zone()?;
            }
        }
    }

    fn finish_zone(&mut self) -> Result<()> {
        if self.current_zone.is_empty() {
            return Ok(());
        }
        self.current_split_rows = self
            .current_split_rows
            .checked_add(self.current_zone_rows)
            .ok_or_else(|| {
                Error::invalid_input("zone split row count exceeds usize on this platform")
            })?;
        self.current_split
            .push(std::mem::take(&mut self.current_zone));
        self.current_zone_rows = 0;
        self.current_fragment = None;
        Ok(())
    }

    fn take_split(&mut self) -> ZoneSplit {
        self.current_split_rows = 0;
        std::mem::take(&mut self.current_split)
    }
}

fn row_addr_column(batch: &RecordBatch) -> Result<&UInt64Array> {
    batch
        .column_by_name(ROW_ADDR)
        .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "zone training batches must include a UInt64 `{ROW_ADDR}` column"
            ))
        })
}

/// Pure-CPU statistics computation for one complete zone; runs via `spawn_cpu`.
///
/// `zone` holds the zone's non-empty slices, all from one fragment. Returns
/// the zone statistics plus the exact addresses of the zone's null rows.
fn process_zone<P: ZoneProcessor>(
    processor_factory: &dyn Fn() -> Result<P>,
    zone: Zone,
    null_rows: &mut RowAddrTreeMap,
) -> Result<P::ZoneStatistics> {
    let mut processor = processor_factory()?;
    for batch in &zone {
        let values = batch.column(0);
        processor.process_chunk(values)?;

        // Record exact row addresses for null values in this slice.
        if values.null_count() > 0
            && let Some(nulls) = values.nulls()
        {
            let row_addrs = row_addr_column(batch)?;
            null_rows.extend(
                nulls
                    .iter()
                    .enumerate()
                    .filter(|(_, is_valid)| !is_valid)
                    .map(|(i, _)| row_addrs.value(i)),
            );
        }
    }

    // Derive the zone bound from the first and last row addresses. Zone length
    // (offset span, last - first + 1) is not the row count: deletions may
    // leave gaps between consecutive offsets.
    let (Some(first), Some(last)) = (zone.first(), zone.last()) else {
        return Err(Error::invalid_input(
            "zone task must contain at least one batch",
        ));
    };
    let first_addr = RowAddress::new_from_u64(row_addr_column(first)?.value(0));
    let last_addr = RowAddress::new_from_u64(row_addr_column(last)?.value(last.num_rows() - 1));
    let start = first_addr.row_offset() as u64;
    let end = last_addr.row_offset() as u64;
    if end < start {
        return Err(Error::invalid_input("zone row offsets are out of order"));
    }
    let bound = ZoneBound {
        fragment_id: first_addr.fragment_id() as u64,
        start,
        length: (end - start + 1) as usize,
    };
    let stats = processor.finish_zone(bound)?;
    Ok(stats)
}

/// Computes all zones in one CPU-sized split, preserving their input order.
fn process_zone_split<P: ZoneProcessor>(
    processor_factory: &dyn Fn() -> Result<P>,
    split: ZoneSplit,
) -> Result<(Vec<P::ZoneStatistics>, RowAddrTreeMap)> {
    let mut statistics = Vec::with_capacity(split.len());
    let mut null_rows = RowAddrTreeMap::new();
    for zone in split {
        statistics.push(process_zone(processor_factory, zone, &mut null_rows)?);
    }
    Ok((statistics, null_rows))
}

/// Shared search helper that loops over zones, records metrics, and
/// collects row address ranges for matching zones. The result is always
/// returned as `SearchResult::AtMost` because zone-level pruning can only
/// guarantee a superset of the true matches.
pub fn search_zones<T, F>(
    zones: &[T],
    metrics: &dyn crate::metrics::MetricsCollector,
    mut zone_matches: F,
) -> Result<crate::scalar::SearchResult>
where
    T: AsRef<ZoneBound>,
    F: FnMut(&T) -> Result<bool>,
{
    metrics.record_comparisons(zones.len());
    let mut row_addr_tree_map = RowAddrTreeMap::new();

    // For each zone, check if it might contain the queried value
    for zone in zones {
        if zone_matches(zone)? {
            let bound = zone.as_ref();
            // Calculate the range of row addresses for this zone
            let zone_start_addr = (bound.fragment_id << 32) + bound.start;
            let zone_end_addr = zone_start_addr + bound.length as u64;

            // Add all row addresses in this zone to the result
            row_addr_tree_map.insert_range(zone_start_addr..zone_end_addr);
        }
    }

    Ok(crate::scalar::SearchResult::at_most(row_addr_tree_map))
}

/// Helper that retrains zones from `stream` and appends them to the existing
/// statistics. Returns the combined zone list and the null-row bitmap for the
/// new data only — callers are responsible for merging with any existing bitmap.
pub async fn rebuild_zones<P>(
    existing: &[P::ZoneStatistics],
    trainer: ZoneTrainer<P>,
    stream: SendableRecordBatchStream,
) -> Result<(Vec<P::ZoneStatistics>, RowAddrTreeMap)>
where
    P: ZoneProcessor + 'static,
    P::ZoneStatistics: Clone + Send + 'static,
{
    let mut combined = existing.to_vec();
    let (mut new_zones, null_rows) = trainer.train(stream).await?;
    combined.append(&mut new_zones);
    Ok((combined, null_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{metrics::LocalMetricsCollector, scalar::SearchResult};
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::ROW_ADDR;
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq)]
    struct MockStats {
        sum: i32,
        bound: ZoneBound,
    }

    #[derive(Debug)]
    struct MockProcessor {
        current_sum: i32,
    }

    impl MockProcessor {
        fn new() -> Self {
            Self { current_sum: 0 }
        }
    }

    impl ZoneProcessor for MockProcessor {
        type ZoneStatistics = MockStats;

        fn process_chunk(&mut self, values: &ArrayRef) -> Result<()> {
            let arr = values.as_any().downcast_ref::<Int32Array>().unwrap();
            self.current_sum += arr.iter().map(|v| v.unwrap_or(0)).sum::<i32>();
            Ok(())
        }

        fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
            Ok(MockStats {
                sum: self.current_sum,
                bound,
            })
        }

        fn reset(&mut self) -> Result<()> {
            self.current_sum = 0;
            Ok(())
        }
    }

    fn batch(values: Vec<i32>, fragments: Vec<u64>, offsets: Vec<u64>) -> RecordBatch {
        let val_array = Arc::new(Int32Array::from(values));
        let row_addrs: Vec<u64> = fragments
            .into_iter()
            .zip(offsets)
            .map(|(frag, off)| (frag << 32) | off)
            .collect();
        let addr_array = Arc::new(UInt64Array::from(row_addrs));
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        RecordBatch::try_new(schema, vec![val_array, addr_array]).unwrap()
    }

    #[tokio::test]
    async fn splits_single_fragment() {
        // Single fragment with 10 rows, zone capacity = 4.
        // Expect three zones with lengths [4, 4, 2].
        let values = vec![1; 10];
        let offsets: Vec<u64> = (0..10).collect();
        let batch = batch(values, vec![0; 10], offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 4).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Three zones: offsets [0..=3], [4..=7], [8..=9]
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 4);
        assert_eq!(stats[1].bound.start, 4);
        assert_eq!(stats[1].bound.length, 4);
        assert_eq!(stats[2].bound.start, 8);
        assert_eq!(stats[2].bound.length, 2); // Last zone has only 2 rows
        assert_eq!(
            stats.iter().map(|s| s.sum).collect::<Vec<_>>(),
            vec![4, 4, 2]
        );
    }

    #[tokio::test]
    async fn flushes_on_fragment_boundary() {
        // Two fragments back to back, capacity is large enough that only fragment
        // boundaries cause zone flushes. Expect two zones (one per fragment).
        let values = vec![1, 1, 1, 2, 2, 2];
        let fragments = vec![0, 0, 0, 1, 1, 1];
        let offsets = vec![0, 1, 2, 0, 1, 2];
        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 10).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Two zones, one per fragment (capacity=10 is large enough)
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.length, 3); // Fragment 0: offsets 0,1,2 → length = 2-0+1 = 3
        assert_eq!(stats[1].bound.fragment_id, 1);
        assert_eq!(stats[1].bound.length, 3); // Fragment 1: offsets 0,1,2 → length = 2-0+1 = 3
    }

    #[tokio::test]
    async fn errors_on_out_of_order_offsets() {
        // Offsets go backwards (5 -> 3). Trainer should treat this as invalid input
        // rather than silently emitting a zero-length zone.
        let values = vec![1, 2, 3];
        let fragments = vec![0, 0, 0];
        let offsets = vec![5, 3, 4];
        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 10).unwrap();
        let err = trainer.train(stream).await.unwrap_err();
        assert!(
            format!("{}", err).contains("zone row offsets are out of order"),
            "unexpected error: {err:?}"
        );
    }

    #[tokio::test]
    async fn handles_empty_batches() {
        // Empty batches in the stream should be properly skipped without affecting zones.
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));

        let empty_batch = RecordBatch::new_empty(schema.clone());
        let valid_batch = batch(vec![1, 2, 3], vec![0, 0, 0], vec![0, 1, 2]);

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![
                Ok(empty_batch.clone()),
                Ok(valid_batch),
                Ok(empty_batch),
            ]),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 10).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // One zone containing the 3 valid rows (empty batches skipped)
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].sum, 6);
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.length, 3);
    }

    #[tokio::test]
    async fn handles_zone_capacity_one() {
        // Each row becomes its own zone when capacity is 1.
        let values = vec![10, 20, 30];
        let offsets = vec![0, 1, 2];
        let batch = batch(values.clone(), vec![0, 0, 0], offsets.clone());
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 1).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Three zones, one per row (capacity=1)
        assert_eq!(stats.len(), 3);
        for (i, stat) in stats.iter().enumerate() {
            assert_eq!(stat.bound.fragment_id, 0);
            assert_eq!(stat.bound.start, offsets[i]);
            assert_eq!(stat.bound.length, 1); // Each zone contains exactly one row
            assert_eq!(stat.sum, values[i]);
        }
    }

    #[tokio::test]
    async fn batches_tiny_zones_incrementally_into_row_sized_splits() {
        let num_rows = MIN_ROWS_PER_SPLIT * 2 + 17;
        let input = batch(
            vec![1; num_rows],
            vec![0; num_rows],
            (0..num_rows as u64).collect(),
        );
        let mut assembler = ZoneSplitAssembler::new(stream_from_batches(vec![input]), 1);

        let first_split = assembler.next_split().await.unwrap().unwrap();
        assert_eq!(
            first_split
                .iter()
                .flatten()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            MIN_ROWS_PER_SPLIT
        );
        assert_eq!(first_split.len(), MIN_ROWS_PER_SPLIT);
        assert_eq!(assembler.input_offset, MIN_ROWS_PER_SPLIT);

        let second_split = assembler.next_split().await.unwrap().unwrap();
        assert_eq!(second_split.len(), MIN_ROWS_PER_SPLIT);

        let trailing_split = assembler.next_split().await.unwrap().unwrap();
        assert_eq!(trailing_split.len(), 17);
        assert!(assembler.next_split().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn handles_large_capacity() {
        // When capacity >> data size, all data fits in one zone.
        let values = vec![1; 100];
        let offsets: Vec<u64> = (0..100).collect();
        let batch = batch(values, vec![0; 100], offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 10000).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // One zone containing all 100 rows (capacity is large enough)
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].sum, 100);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 100);
    }

    #[tokio::test]
    async fn rejects_zero_capacity() {
        let err = ZoneTrainer::new(|| Ok(MockProcessor::new()), 0)
            .err()
            .unwrap();
        assert!(
            err.to_string()
                .contains("zone capacity must be greater than zero")
        );
    }

    #[tokio::test]
    async fn handles_multiple_batches_same_fragment() {
        // Multiple batches from the same fragment should be properly accumulated into zones.
        let b1 = batch(vec![1, 1], vec![0, 0], vec![0, 1]);
        let b2 = batch(vec![1, 1], vec![0, 0], vec![2, 3]);
        let b3 = batch(vec![1, 1], vec![0, 0], vec![4, 5]);

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            b1.schema(),
            stream::iter(vec![Ok(b1), Ok(b2), Ok(b3)]),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 4).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Two zones: first 4 rows, then remaining 2 rows
        assert_eq!(stats.len(), 2);
        // First zone: offsets [0..=3]
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 4);
        assert_eq!(stats[0].sum, 4);
        // Second zone: offsets [4..=5]
        assert_eq!(stats[1].bound.fragment_id, 0);
        assert_eq!(stats[1].bound.start, 4);
        assert_eq!(stats[1].bound.length, 2);
        assert_eq!(stats[1].sum, 2);
    }

    #[tokio::test]
    async fn handles_multi_batch_with_fragment_change() {
        // Complex scenario: multiple batches with fragment changes mid-batch.
        // This tests that zones flush correctly at fragment boundaries.
        let b1 = batch(vec![1, 1], vec![0, 0], vec![0, 1]);
        // b2 has fragment change: starts with frag 0, switches to frag 1
        let b2 = batch(vec![1, 1, 2, 2], vec![0, 0, 1, 1], vec![2, 3, 0, 1]);

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            b1.schema(),
            stream::iter(vec![Ok(b1), Ok(b2)]),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 3).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Three zones: frag 0 full zone, frag 0 partial (flushed at boundary), frag 1
        assert_eq!(stats.len(), 3);

        // Zone 0: Fragment 0, offsets [0..=2] (fills capacity)
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 3);
        assert_eq!(stats[0].sum, 3);

        // Zone 1: Fragment 0, offset 3 (partial, flushed at fragment boundary)
        assert_eq!(stats[1].bound.fragment_id, 0);
        assert_eq!(stats[1].bound.start, 3);
        assert_eq!(stats[1].bound.length, 1);
        assert_eq!(stats[1].sum, 1);

        // Zone 2: Fragment 1, offsets [0..=1]
        assert_eq!(stats[2].bound.fragment_id, 1);
        assert_eq!(stats[2].bound.start, 0);
        assert_eq!(stats[2].bound.length, 2);
        assert_eq!(stats[2].sum, 4);
    }

    #[tokio::test]
    async fn handles_non_contiguous_offsets_after_deletion() {
        // CRITICAL: Test deletion scenario with non-contiguous row offsets.
        // This is the main reason for tracking first/last offsets.
        // Simulate a zone where rows 2, 3, 4, 6 have been deleted.
        let values = vec![1, 1, 1, 1, 1, 1]; // 6 actual rows
        let fragments = vec![0, 0, 0, 0, 0, 0];
        let offsets = vec![0, 1, 5, 7, 8, 9]; // Non-contiguous!

        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 4).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Should create 2 zones (capacity=4):
        // Zone 0: rows at offsets [0, 1, 5, 7] (4 rows)
        // Zone 1: rows at offsets [8, 9] (2 rows)
        assert_eq!(stats.len(), 2);

        // First zone: 4 rows, but offset span is [0..=7] so length=8 (due to gaps)
        assert_eq!(stats[0].sum, 4);
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 8); // Address span: 7 - 0 + 1

        // Second zone: 2 rows, offset span is [8..=9] so length=2
        assert_eq!(stats[1].sum, 2);
        assert_eq!(stats[1].bound.fragment_id, 0);
        assert_eq!(stats[1].bound.start, 8);
        assert_eq!(stats[1].bound.length, 2); // Address span: 9 - 8 + 1
    }

    #[tokio::test]
    async fn handles_deletion_with_large_gaps() {
        // Extreme deletion scenario: very large gaps between consecutive rows.
        let values = vec![1, 1, 1];
        let fragments = vec![0, 0, 0];
        let offsets = vec![0, 100, 200]; // Huge gaps!

        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 10).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // One zone with 3 rows, but offset span [0..=200] so length=201 due to large gaps
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].sum, 3);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 201); // Span: 200 - 0 + 1
    }

    #[tokio::test]
    async fn handles_non_contiguous_fragment_ids() {
        // CRITICAL: Test fragment IDs that are not consecutive (e.g., after fragment deletion).
        // Original code assumed fragment_id + 1, which would fail here.
        // Fragment IDs: 0, 5, 10 (non-consecutive!)
        let values = vec![1, 1, 2, 2, 3, 3];
        let fragments = vec![0, 0, 5, 5, 10, 10]; // Gaps in fragment IDs
        let offsets = vec![0, 1, 0, 1, 0, 1];

        let batch = batch(values, fragments, offsets);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 10).unwrap();
        let (stats, _) = trainer.train(stream).await.unwrap();

        // Should create 3 zones (one per fragment)
        assert_eq!(stats.len(), 3);

        // Fragment 0
        assert_eq!(stats[0].bound.fragment_id, 0);
        assert_eq!(stats[0].bound.start, 0);
        assert_eq!(stats[0].bound.length, 2);
        assert_eq!(stats[0].sum, 2);

        // Fragment 5 (not 1!)
        assert_eq!(stats[1].bound.fragment_id, 5);
        assert_eq!(stats[1].bound.start, 0);
        assert_eq!(stats[1].bound.length, 2);
        assert_eq!(stats[1].sum, 4);

        // Fragment 10 (not 2!)
        assert_eq!(stats[2].bound.fragment_id, 10);
        assert_eq!(stats[2].bound.start, 0);
        assert_eq!(stats[2].bound.length, 2);
        assert_eq!(stats[2].sum, 6);
    }

    #[test]
    fn search_zones_collects_row_ranges() {
        // Ensure the shared helper converts matching zones into the correct row-id
        // ranges (fragment upper bits + local offsets) while skipping non-matching
        // zones. This protects the helper if we modify how RowAddrTreeMap ranges are
        // inserted in the future.
        #[derive(Debug)]
        struct DummyZone {
            bound: ZoneBound,
            matches: bool,
        }

        impl AsRef<ZoneBound> for DummyZone {
            fn as_ref(&self) -> &ZoneBound {
                &self.bound
            }
        }

        let zones = vec![
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 2,
                },
                matches: true,
            },
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 1,
                    start: 5,
                    length: 3,
                },
                matches: false,
            },
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 2,
                    start: 10,
                    length: 1,
                },
                matches: true,
            },
        ];

        let metrics = LocalMetricsCollector::default();
        let result = search_zones(&zones, &metrics, |zone| Ok(zone.matches)).unwrap();
        let SearchResult::AtMost(map) = result else {
            panic!("search_zones should return AtMost for dummy zones");
        };

        // Fragment 0, offsets 0 and 1
        assert!(map.selected(0));
        assert!(map.selected(1));
        // Fragment 1 should be skipped entirely
        assert!(!map.selected((1_u64 << 32) + 5));
        assert!(!map.selected((1_u64 << 32) + 7));
        // Fragment 2 includes only the single offset 10
        assert!(map.selected((2_u64 << 32) + 10));
        assert!(!map.selected((2_u64 << 32) + 11));
    }

    #[test]
    fn search_zones_returns_empty_when_no_match() {
        #[derive(Debug)]
        struct DummyZone {
            bound: ZoneBound,
            matches: bool,
        }

        impl AsRef<ZoneBound> for DummyZone {
            fn as_ref(&self) -> &ZoneBound {
                &self.bound
            }
        }

        // Both zones are marked as non-matching. The helper should return an empty map.
        let zones = vec![
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 0,
                    start: 0,
                    length: 4,
                },
                matches: false,
            },
            DummyZone {
                bound: ZoneBound {
                    fragment_id: 1,
                    start: 10,
                    length: 2,
                },
                matches: false,
            },
        ];

        let metrics = LocalMetricsCollector::default();
        let result = search_zones(&zones, &metrics, |zone| Ok(zone.matches)).unwrap();
        let SearchResult::AtMost(map) = result else {
            panic!("expected AtMost result");
        };
        // No zones should be inserted when every predicate evaluates to false
        assert!(map.is_empty());
    }

    #[tokio::test]
    async fn rebuild_zones_appends_new_stats() {
        let existing = vec![MockStats {
            sum: 50,
            bound: ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 2,
            },
        }];

        let batch = batch(vec![3, 4], vec![1, 1], vec![0, 1]);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 2).unwrap();
        let (rebuilt, _) = rebuild_zones(&existing, trainer, stream).await.unwrap();
        // Existing zone should remain unchanged and new stats appended afterwards
        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt[0].sum, 50);
        assert_eq!(rebuilt[1].sum, 7);
        assert_eq!(rebuilt[1].bound.fragment_id, 1);
        assert_eq!(rebuilt[1].bound.start, 0);
        assert_eq!(rebuilt[1].bound.length, 2);
    }

    #[tokio::test]
    async fn rebuild_zones_handles_multi_fragment_stream() {
        let existing = vec![MockStats {
            sum: 10,
            bound: ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 1,
            },
        }];

        // Construct a stream with two fragments. Trainer should emit two zones that
        // get appended after the existing entries.
        let batch = batch(vec![5, 5, 6, 6], vec![1, 1, 2, 2], vec![0, 1, 0, 1]);
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::once(async { Ok(batch) }),
        ));

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 2).unwrap();
        let (rebuilt, _) = rebuild_zones(&existing, trainer, stream).await.unwrap();
        // Existing zone plus two new fragments should yield three total zones
        assert_eq!(rebuilt.len(), 3);
        assert_eq!(rebuilt[0].bound.fragment_id, 0);
        assert_eq!(rebuilt[1].bound.fragment_id, 1);
        assert_eq!(rebuilt[2].bound.fragment_id, 2);
        assert_eq!(rebuilt[1].sum, 10);
        assert_eq!(rebuilt[2].sum, 12);
    }

    /// Nullable variant of [`batch`] for tests that assert on null handling.
    fn nullable_batch(values: Vec<Option<i32>>, row_addrs: Vec<u64>) -> RecordBatch {
        let val_array = Arc::new(Int32Array::from(values));
        let addr_array = Arc::new(UInt64Array::from(row_addrs));
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        RecordBatch::try_new(schema, vec![val_array, addr_array]).unwrap()
    }

    fn stream_from_batches(batches: Vec<RecordBatch>) -> SendableRecordBatchStream {
        let schema = batches[0].schema();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        ))
    }

    /// Straightforward row-at-a-time re-implementation of the zone-splitting
    /// rules, used as the oracle for the randomized equivalence test.
    fn reference_zones(rows: &[(u64, Option<i32>)], capacity: usize) -> (Vec<MockStats>, Vec<u64>) {
        fn flush(current: &mut Vec<(u64, Option<i32>)>, zones: &mut Vec<MockStats>) {
            if current.is_empty() {
                return;
            }
            let (first, last) = (current[0].0, current[current.len() - 1].0);
            let (start, end) = (first & 0xFFFF_FFFF, last & 0xFFFF_FFFF);
            zones.push(MockStats {
                sum: current.iter().map(|(_, value)| value.unwrap_or(0)).sum(),
                bound: ZoneBound {
                    fragment_id: first >> 32,
                    start,
                    length: (end - start + 1) as usize,
                },
            });
            current.clear();
        }

        let mut zones = Vec::new();
        let mut nulls = Vec::new();
        let mut current: Vec<(u64, Option<i32>)> = Vec::new();
        for &(addr, value) in rows {
            if !current.is_empty() && current[current.len() - 1].0 >> 32 != addr >> 32 {
                flush(&mut current, &mut zones);
            }
            if value.is_none() {
                nulls.push(addr);
            }
            current.push((addr, value));
            if current.len() == capacity {
                flush(&mut current, &mut zones);
            }
        }
        flush(&mut current, &mut zones);
        (zones, nulls)
    }

    #[tokio::test]
    async fn matches_serial_reference_on_random_streams() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        for trial in 0..4u64 {
            let mut rng = SmallRng::seed_from_u64(trial);

            // Non-contiguous fragment ids, random row counts, deletion gaps in
            // the offsets, and ~25% null values.
            let mut rows: Vec<(u64, Option<i32>)> = Vec::new();
            for fragment_id in [0u64, 3, 4, 11, 100] {
                let num_rows = rng.random_range(0..60);
                let mut offset: u64 = rng.random_range(0..4);
                for _ in 0..num_rows {
                    let value = (!rng.random_bool(0.25)).then(|| rng.random_range(-100..100));
                    rows.push(((fragment_id << 32) | offset, value));
                    offset += rng.random_range(1..5);
                }
            }

            // Odd batch sizes so zones accumulate dozens of slices (batch size
            // 1 or 3 under capacity 8192) and single batches complete multiple
            // zones (batch size 100 under capacity 1 or 3).
            let batch_size = [1, 3, 7, 100][(trial % 4) as usize];
            let batches: Vec<RecordBatch> = rows
                .chunks(batch_size)
                .map(|chunk| {
                    nullable_batch(
                        chunk.iter().map(|(_, value)| *value).collect(),
                        chunk.iter().map(|(addr, _)| *addr).collect(),
                    )
                })
                .collect();
            if batches.is_empty() {
                continue;
            }

            for capacity in [1u64, 3, 8, 8192] {
                let (expected_zones, expected_nulls) = reference_zones(&rows, capacity as usize);
                let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), capacity).unwrap();
                let (zones, null_rows) = trainer
                    .train(stream_from_batches(batches.clone()))
                    .await
                    .unwrap();
                assert_eq!(zones, expected_zones, "trial={trial} capacity={capacity}");
                assert_eq!(
                    null_rows,
                    RowAddrTreeMap::from_iter(expected_nulls.iter().copied()),
                    "trial={trial} capacity={capacity}"
                );
            }
        }
    }

    #[tokio::test]
    async fn collects_null_row_addresses() {
        // Nulls straddle a zone boundary (capacity 4: offsets 3 and 4), straddle
        // a batch boundary (offsets 4 and 5), sit behind deletion gaps (offset 5
        // is followed by 7 and 9), and fill an entire zone (fragment 1).
        let b1 = nullable_batch(
            vec![Some(1), Some(2), Some(3), None, None],
            vec![0, 1, 2, 3, 4],
        );
        let b2 = nullable_batch(vec![None, Some(5), Some(6)], vec![5, 7, 9]);
        let b3 = nullable_batch(
            vec![None, None, None],
            vec![(1 << 32), (1 << 32) + 1, (1 << 32) + 2],
        );

        let trainer = ZoneTrainer::new(|| Ok(MockProcessor::new()), 4).unwrap();
        let (zones, null_rows) = trainer
            .train(stream_from_batches(vec![b1, b2, b3]))
            .await
            .unwrap();

        assert_eq!(zones.len(), 3);
        let expected_nulls = [3, 4, 5, 1 << 32, (1 << 32) + 1, (1 << 32) + 2];
        assert_eq!(null_rows, RowAddrTreeMap::from_iter(expected_nulls));
    }
}
