// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use lance_core::{Error, Result};

use super::{
    CANDIDATE_BATCH_SIZES, DecodeBatchScheduler, FilterExpression, RootScheduler, SchedulerContext,
    StructuralBatchDecodeStream, StructuralFieldDecoder, has_variable_width,
};
use crate::EncodingsIo;

pub(super) struct BudgetScanScheduler {
    pub scheduler: DecodeBatchScheduler,
    pub io: Arc<dyn EncodingsIo>,
}

impl BudgetScanScheduler {
    pub async fn prepare(&mut self, range: Range<u64>) -> Result<()> {
        let RootScheduler::Structural(root) = &mut self.scheduler.root_scheduler else {
            return Err(Error::not_supported(
                "budget scans require structural encoding",
            ));
        };
        root.release_before(range.start)?;
        // Initialization may load dictionaries as well as chunk metadata. Limit
        // it to demanded pages and release previous pages' scheduler references.
        root.initialize(
            Some(&[range]),
            &FilterExpression::no_filter(),
            &SchedulerContext::new(self.io.clone(), self.scheduler.cache.clone()),
        )
        .await
    }
}

/// A proposed output batch, without advancing the scan's drain position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedBatch {
    /// Number of rows to pass to [`StructuralBatchDecodeStream::drain_task`].
    pub num_rows: u32,
    /// Conservative retained output size; not a bound on I/O or decode scratch.
    pub decoded_bytes: u64,
}

impl StructuralBatchDecodeStream {
    /// Plan the next output batch, stopping at the first candidate above the target.
    ///
    /// Fixed-width output is planned without loading payload. Variable-width output
    /// probes successively larger prefixes, retaining at most the first rejected
    /// candidate rather than loading every candidate up front. This is an output
    /// target: compressed input granularity and transient decode memory still need
    /// a separate budget. If even the one-row bound exceeds the target, return
    /// that row so conservative estimates do not prevent progress.
    pub async fn plan_next_batch(&mut self, byte_target: u64) -> Result<Option<PlannedBatch>> {
        if byte_target == 0 {
            return Err(Error::invalid_input("batch byte target must be positive"));
        }
        if self.rows_remaining == 0 {
            return Ok(None);
        }
        let row_limit = self.rows_remaining.min(self.rows_per_batch as u64);
        let fixed_bytes = if has_variable_width(self.root_decoder.data_type()) {
            None
        } else {
            Some(self.root_decoder.plan_decoded_bytes(row_limit)?)
        };
        let mut previous = None;
        for (index, candidate) in CANDIDATE_BATCH_SIZES.into_iter().enumerate() {
            let num_rows = u64::from(candidate).min(row_limit) as u32;
            let decoded_bytes = match &fixed_bytes {
                Some(bytes) => bytes[index],
                None => self.plan_decoded_bytes(u64::from(num_rows)).await?[index],
            };
            let plan = PlannedBatch {
                num_rows,
                decoded_bytes,
            };
            if decoded_bytes > byte_target {
                return Ok(Some(previous.unwrap_or(plan)));
            }
            previous = Some(plan);
            if u64::from(num_rows) == row_limit {
                break;
            }
        }
        Ok(previous)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use arrow_array::{Array, ArrayRef, RecordBatch, StringArray, UInt8Array, UInt32Array};
    use arrow_schema::{Field, Schema as ArrowSchema};
    use bytes::{Bytes, BytesMut};
    use futures::{FutureExt, future::BoxFuture};
    use lance_core::cache::LanceCache;
    use lance_core::datatypes::Schema;
    use rstest::rstest;

    use super::*;
    use crate::decoder::{ColumnInfo, PageInfo};
    use crate::decoder::{
        DecoderConfig, DecoderPlugins, SchedulerDecoderConfig, open_structural_budget_scan,
    };
    use crate::encoder::{
        BatchEncoder, EncodedBatch, EncodingOptions, OutOfLineBuffers, encode_batch,
    };
    use crate::repdef::RepDefBuilder;
    use crate::testing::{TestEncoding, test_encoding_strategy};

    #[derive(Debug, Default)]
    struct ReadState {
        requests: AtomicUsize,
        live_bytes: AtomicUsize,
        peak_bytes: AtomicUsize,
        is_blocked: AtomicBool,
    }

    struct ReadBytes {
        data: Vec<u8>,
        state: Arc<ReadState>,
    }

    impl AsRef<[u8]> for ReadBytes {
        fn as_ref(&self) -> &[u8] {
            &self.data
        }
    }

    impl Drop for ReadBytes {
        fn drop(&mut self) {
            self.state
                .live_bytes
                .fetch_sub(self.data.len(), Ordering::Relaxed);
        }
    }

    #[derive(Debug)]
    struct TrackedIo {
        data: Bytes,
        state: Arc<ReadState>,
    }

    impl EncodingsIo for TrackedIo {
        fn submit_request(
            &self,
            ranges: Vec<Range<u64>>,
            _priority: u64,
        ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
            self.state.requests.fetch_add(1, Ordering::Relaxed);
            let buffers = ranges
                .into_iter()
                .map(|range| {
                    let data = self.data[range.start as usize..range.end as usize].to_vec();
                    let live = self
                        .state
                        .live_bytes
                        .fetch_add(data.len(), Ordering::Relaxed)
                        + data.len();
                    self.state.peak_bytes.fetch_max(live, Ordering::Relaxed);
                    Bytes::from_owner(ReadBytes {
                        data,
                        state: self.state.clone(),
                    })
                })
                .collect();
            let is_blocked = self.state.is_blocked.load(Ordering::Relaxed);
            async move {
                if is_blocked {
                    std::future::pending::<()>().await;
                }
                Ok(buffers)
            }
            .boxed()
        }
    }

    // FileScheduler can return disjoint slices of one coalesced read. The
    // physical allocation, including the gap, survives through Arrow's custom
    // allocation owner even though Arrow only reports the visible slice sizes.
    // A future input budget must charge that owner, not sum returned lengths or
    // refund the reservation when the I/O future finishes.
    #[rstest]
    #[case::small_gap(4096)]
    #[case::large_gap(65536)]
    fn input_lifetime_coalesced_backing_allocation(#[case] physical_bytes: usize) {
        use crate::buffer::LanceBuffer;
        use arrow_buffer::ScalarBuffer;

        let state = Arc::new(ReadState::default());
        state.live_bytes.store(physical_bytes, Ordering::Relaxed);
        let input = Bytes::from_owner(ReadBytes {
            data: vec![37; physical_bytes],
            state: state.clone(),
        });
        let slices = [input.slice(0..1), input.slice(physical_bytes - 1..)];
        let arrays = slices.map(|bytes| {
            UInt8Array::new(
                ScalarBuffer::new(LanceBuffer::from_bytes(bytes, 1).into_buffer(), 0, 1),
                None,
            )
        });
        assert_eq!(arrays.iter().map(|a| a.len()).sum::<usize>(), 2);
        assert!(
            arrays
                .iter()
                .map(Array::get_buffer_memory_size)
                .sum::<usize>()
                < physical_bytes
        );
        let last_owner = arrays[1].slice(0, 1);
        drop(input);
        drop(arrays);
        assert_eq!(last_owner.value(0), 37);
        assert_eq!(state.live_bytes.load(Ordering::Relaxed), physical_bytes);
        drop(last_owner);
        assert_eq!(state.live_bytes.load(Ordering::Relaxed), 0);
    }

    async fn encode_paged_fixture(batch: &RecordBatch, version: TestEncoding) -> EncodedBatch {
        let schema = Arc::new(Schema::try_from(batch.schema().as_ref()).unwrap());
        let strategy = test_encoding_strategy(version);
        let options = EncodingOptions::default();
        let mut encoder = BatchEncoder::try_new(&schema, strategy.as_ref(), &options)
            .unwrap()
            .field_encoders
            .remove(0);
        let mut data = BytesMut::new();
        let mut pages = Vec::new();
        // Structural pages follow flush boundaries, not max_page_bytes for a
        // single input array. Exercise a real encoder across many such boundaries.
        for start in (0..batch.num_rows()).step_by(16) {
            let num_rows = (batch.num_rows() - start).min(16);
            let array = encoder
                .prepare_array(batch.column(0).slice(start, num_rows))
                .unwrap();
            let mut external = OutOfLineBuffers::new(data.len() as u64, options.buffer_alignment);
            let mut tasks = encoder
                .maybe_encode(
                    array,
                    &mut external,
                    RepDefBuilder::default(),
                    start as u64,
                    num_rows as u64,
                )
                .unwrap();
            tasks.extend(encoder.flush(&mut external).unwrap());
            for buffer in external.take_buffers() {
                data.extend_from_slice(&buffer);
            }
            for task in tasks {
                let page = task.await.unwrap();
                let mut buffers = Vec::new();
                for buffer in page.data {
                    buffers.push((data.len() as u64, buffer.len() as u64));
                    data.extend_from_slice(&buffer);
                }
                pages.push(PageInfo {
                    num_rows: page.num_rows,
                    priority: page.row_number,
                    encoding: page.description,
                    buffer_offsets_and_sizes: buffers.into(),
                });
            }
        }
        let mut external = OutOfLineBuffers::new(data.len() as u64, options.buffer_alignment);
        let mut columns = encoder.finish(&mut external).await.unwrap();
        assert_eq!(columns.len(), 1);
        let column = columns.remove(0);
        assert!(column.column_buffers.is_empty());
        assert!(column.final_pages.is_empty());
        assert!(external.take_buffers().is_empty());
        EncodedBatch {
            data: data.freeze(),
            page_table: vec![Arc::new(ColumnInfo::new(
                0,
                pages.into(),
                vec![],
                column.encoding,
            ))],
            schema,
            top_level_columns: vec![0],
            num_rows: batch.num_rows() as u64,
        }
    }

    #[rstest]
    #[case::fixed(128, false, false, false)]
    #[case::variable(128, true, false, false)]
    #[case::longer_variable(2048, true, false, false)]
    #[case::cancel_input(128, true, true, false)]
    #[case::empty(0, false, false, false)]
    #[case::many_pages(256, true, false, true)]
    #[case::more_pages(1024, true, false, true)]
    #[tokio::test]
    async fn test_budget_scan_demand_and_release(
        #[case] num_rows: usize,
        #[case] is_variable: bool,
        #[case] is_cancelled: bool,
        #[case] is_full_drain: bool,
        #[values(TestEncoding::StructuralU16, TestEncoding::StructuralU32)] version: TestEncoding,
    ) {
        let values: ArrayRef = if is_variable {
            Arc::new(StringArray::from_iter_values(
                (0..num_rows).map(|i| format!("{i:08x}").repeat(32)),
            ))
        } else {
            Arc::new(UInt32Array::from_iter_values(0..num_rows as u32))
        };
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            values.data_type().clone(),
            false,
        )]));
        let expected = RecordBatch::try_new(schema.clone(), vec![values]).unwrap();
        let encoded = if is_full_drain {
            encode_paged_fixture(&expected, version).await
        } else {
            encode_batch(
                &expected,
                Arc::new(Schema::try_from(schema.as_ref()).unwrap()),
                test_encoding_strategy(version).as_ref(),
                &EncodingOptions::default(),
            )
            .await
            .unwrap()
        };
        let max_page_bytes = encoded
            .page_table
            .iter()
            .flat_map(|col| col.page_infos.iter())
            .map(|page| {
                page.buffer_offsets_and_sizes
                    .iter()
                    .map(|(_, size)| *size)
                    .sum::<u64>()
            })
            .max()
            .unwrap_or(0);
        if is_full_drain {
            assert!(encoded.page_table[0].page_infos.len() > 1);
        }
        let state = Arc::new(ReadState::default());
        let io = Arc::new(TrackedIo {
            data: encoded.data,
            state: state.clone(),
        });
        let mut scan = open_structural_budget_scan(
            encoded.page_table,
            encoded.num_rows,
            encoded.top_level_columns,
            encoded.schema,
            SchedulerDecoderConfig {
                decoder_plugins: Arc::new(DecoderPlugins::default()),
                batch_size: u32::MAX,
                io,
                cache: Arc::new(LanceCache::no_cache()),
                decoder_config: DecoderConfig::default(),
                batch_size_bytes: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            scan.rows_scheduled, 0,
            "opening must not schedule payload for the whole file"
        );
        let initial_requests = state.requests.load(Ordering::Relaxed);
        assert_eq!(
            initial_requests, 0,
            "opening must defer dictionaries and chunk metadata too"
        );
        let target = if is_variable { 512 } else { 16 };
        let error = scan.plan_next_batch(0).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("must be positive"));
        if num_rows == 0 {
            assert!(scan.plan_next_batch(target).await.unwrap().is_none());
            assert!(scan.drain_task(1).await.unwrap().is_none());
            assert_eq!(state.requests.load(Ordering::Relaxed), initial_requests);
        } else if is_cancelled {
            state.is_blocked.store(true, Ordering::Relaxed);
            assert!(scan.plan_next_batch(target).now_or_never().is_none());
        } else {
            let plan = scan.plan_next_batch(target).await.unwrap().unwrap();
            assert_eq!(plan.num_rows, if is_variable { 1 } else { 4 });
            if is_variable {
                assert!(
                    scan.rows_scheduled <= 4,
                    "stop at the first rejected prefix, independent of file length"
                );
            } else {
                assert_eq!(scan.rows_scheduled, 0);
                assert_eq!(state.requests.load(Ordering::Relaxed), initial_requests);
            }
            let requests = state.requests.load(Ordering::Relaxed);
            assert_eq!(scan.plan_next_batch(target).await.unwrap(), Some(plan));
            tokio::task::yield_now().await;
            assert_eq!(
                state.requests.load(Ordering::Relaxed),
                requests,
                "a paused consumer must not trigger additional reads"
            );
            let batch = scan
                .drain_task(plan.num_rows)
                .await
                .unwrap()
                .unwrap()
                .task
                .await
                .unwrap();
            assert_eq!(batch, expected.slice(0, plan.num_rows as usize));
            assert!(batch.column(0).get_buffer_memory_size() as u64 <= target);
            if is_full_drain {
                let mut drained = batch.num_rows();
                while let Some(plan) = scan.plan_next_batch(target).await.unwrap() {
                    let batch = scan
                        .drain_task(plan.num_rows)
                        .await
                        .unwrap()
                        .unwrap()
                        .task
                        .await
                        .unwrap();
                    assert_eq!(batch, expected.slice(drained, batch.num_rows()));
                    assert!(batch.column(0).get_buffer_memory_size() as u64 <= target);
                    drained += batch.num_rows();
                }
                assert_eq!(drained, num_rows);
                // These fixtures select one row and probe at most four. Budget
                // four independently owned input pages plus current page metadata,
                // regardless of how many pages the complete file contains.
                assert!(state.peak_bytes.load(Ordering::Relaxed) as u64 <= 5 * max_page_bytes);
            }
        }
        drop(scan);
        tokio::task::yield_now().await;
        assert_eq!(
            state.live_bytes.load(Ordering::Relaxed),
            0,
            "dropping the scan must release input buffers, including pending reads"
        );
    }
}
