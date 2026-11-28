// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use async_trait::async_trait;
use background_iterator::BackgroundIterator;
use datafusion_common::DataFusionError;
use datafusion_execution::RecordBatchStream;
use datafusion_physical_plan::{
    SendableRecordBatchStream,
    metrics::{
        Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, MetricValue, MetricsSet, Time,
    },
    stream::RecordBatchStreamAdapter,
};
use futures::{StreamExt, TryStreamExt};
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_io::stream::{
    RecordBatchStreamAdapter as LanceStreamAdapter, SendableRecordBatchStream as LanceBatchStream,
    infer_write_schema,
};

pub use lance_io::stream::background_iterator;

/// A trait for [`RecordBatch`] iterators, readers and streams
/// that can be converted to a concrete stream type [`SendableRecordBatchStream`].
///
/// This also can read the schema from the first batch
/// and then update the schema to reflect the dictionary columns.
#[async_trait]
pub trait StreamingWriteSource: Send {
    /// Infer the Lance schema from the first batch stream.
    ///
    /// This will peek the first batch to get the dictionaries for dictionary columns.
    ///
    /// NOTE: this does not validate the schema. For example, for appends the schema
    /// should be checked to make sure it matches the existing dataset schema before
    /// writing.
    async fn into_stream_and_schema(self) -> Result<(SendableRecordBatchStream, Schema)>
    where
        Self: Sized,
    {
        let (stream, arrow_schema, schema) =
            infer_write_schema(self.into_stream(), |stream| stream.schema()).await?;
        let adapter = RecordBatchStreamAdapter::new(arrow_schema, stream);
        Ok((Box::pin(adapter), schema))
    }

    /// Returns the arrow schema.
    fn arrow_schema(&self) -> SchemaRef;

    /// Convert to a stream.
    ///
    /// The conversion will be conducted in a background thread.
    fn into_stream(self) -> SendableRecordBatchStream;
}

impl StreamingWriteSource for ArrowArrayStreamReader {
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        RecordBatchReader::schema(self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        reader_to_stream(Box::new(self))
    }
}

impl<I> StreamingWriteSource for RecordBatchIterator<I>
where
    Self: Send,
    I: IntoIterator<Item = ::core::result::Result<RecordBatch, ArrowError>> + Send + 'static,
{
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        RecordBatchReader::schema(self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        reader_to_stream(Box::new(self))
    }
}

impl<T> StreamingWriteSource for Box<T>
where
    T: StreamingWriteSource,
{
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        T::arrow_schema(&**self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        T::into_stream(*self)
    }
}

impl StreamingWriteSource for Box<dyn RecordBatchReader + Send> {
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        RecordBatchReader::schema(self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        reader_to_stream(self)
    }
}

impl StreamingWriteSource for SendableRecordBatchStream {
    #[inline]
    fn arrow_schema(&self) -> SchemaRef {
        RecordBatchStream::schema(&**self)
    }

    #[inline]
    fn into_stream(self) -> SendableRecordBatchStream {
        self
    }
}

/// Convert reader to a stream.
///
/// The reader will be called in a background thread.
pub fn reader_to_stream(batches: Box<dyn RecordBatchReader + Send>) -> SendableRecordBatchStream {
    let arrow_schema = batches.arrow_schema();
    // Keep readers' Arrow errors available through DataFusion's ArrowError variant.
    let stream = RecordBatchStreamAdapter::new(
        arrow_schema,
        BackgroundIterator::new(batches)
            .fuse()
            .map_err(DataFusionError::from),
    );
    Box::pin(stream)
}

/// Convert a DataFusion stream to a Lance stream, preserving its schema and
/// converting errors as batches are read.
///
/// ```
/// # use datafusion_execution::SendableRecordBatchStream;
/// # use lance_datafusion_exec::utils::from_datafusion_stream;
/// # fn example(input: SendableRecordBatchStream) {
/// let stream = from_datafusion_stream(input);
/// # drop(stream);
/// # }
/// ```
pub fn from_datafusion_stream(stream: SendableRecordBatchStream) -> LanceBatchStream {
    let schema = stream.schema();
    Box::pin(LanceStreamAdapter::new(schema, stream.map_err(Into::into)))
}

pub trait MetricsExt {
    fn find_count(&self, name: &str) -> Option<Count>;
    fn iter_counts(&self) -> impl Iterator<Item = (impl AsRef<str>, &Count)>;
    fn iter_times(&self) -> impl Iterator<Item = (impl AsRef<str>, &Time)>;
    fn iter_gauges(&self) -> impl Iterator<Item = (impl AsRef<str>, &Gauge)>;
}

impl MetricsExt for MetricsSet {
    fn find_count(&self, metric_name: &str) -> Option<Count> {
        self.iter().find_map(|m| match m.value() {
            MetricValue::Count { name, count } => {
                if name == metric_name {
                    Some(count.clone())
                } else {
                    None
                }
            }
            _ => None,
        })
    }

    fn iter_counts(&self) -> impl Iterator<Item = (impl AsRef<str>, &Count)> {
        self.iter().filter_map(|m| match m.value() {
            MetricValue::Count { name, count } => Some((name, count)),
            _ => None,
        })
    }

    fn iter_times(&self) -> impl Iterator<Item = (impl AsRef<str>, &Time)> {
        self.iter().filter_map(|m| match m.value() {
            MetricValue::Time { name, time } => Some((name, time)),
            _ => None,
        })
    }

    fn iter_gauges(&self) -> impl Iterator<Item = (impl AsRef<str>, &Gauge)> {
        self.iter().filter_map(|m| match m.value() {
            MetricValue::Gauge { name, gauge } => Some((name, gauge)),
            _ => None,
        })
    }
}

pub trait ExecutionPlanMetricsSetExt {
    fn new_count(&self, name: &'static str, partition: usize) -> Count;
    fn new_time(&self, name: &'static str, partition: usize) -> Time;
    fn new_gauge(&self, name: &'static str, partition: usize) -> Gauge;
}

impl ExecutionPlanMetricsSetExt for ExecutionPlanMetricsSet {
    fn new_count(&self, name: &'static str, partition: usize) -> Count {
        let count = Count::new();
        MetricBuilder::new(self)
            .with_partition(partition)
            .build(MetricValue::Count {
                name: Cow::Borrowed(name),
                count: count.clone(),
            });
        count
    }

    fn new_time(&self, name: &'static str, partition: usize) -> Time {
        let time = Time::new();
        MetricBuilder::new(self)
            .with_partition(partition)
            .build(MetricValue::Time {
                name: Cow::Borrowed(name),
                time: time.clone(),
            });
        time
    }

    fn new_gauge(&self, name: &'static str, partition: usize) -> Gauge {
        let gauge = Gauge::new();
        MetricBuilder::new(self)
            .with_partition(partition)
            .build(MetricValue::Gauge {
                name: Cow::Borrowed(name),
                gauge: gauge.clone(),
            });
        gauge
    }
}

// Common metrics
pub const IOPS_METRIC: &str = "iops";
pub const REQUESTS_METRIC: &str = "requests";
pub const BYTES_READ_METRIC: &str = "bytes_read";
pub const INDICES_LOADED_METRIC: &str = "indices_loaded";
pub const PARTS_LOADED_METRIC: &str = "parts_loaded";
pub const PARTITIONS_RANKED_METRIC: &str = "partitions_ranked";
pub const INDEX_COMPARISONS_METRIC: &str = "index_comparisons";
pub const INDEX_CACHE_HITS_METRIC: &str = "index_cache_hits";
pub const INDEX_CACHE_MISSES_METRIC: &str = "index_cache_misses";
/// Recorded by no operator: the summary of an execution reports it, see
/// [`crate::exec::ExecutionSummaryCounts::peak_reserved_bytes`].
pub const PEAK_RESERVED_BYTES_METRIC: &str = "peak_reserved_bytes";
pub const FRAGMENTS_SCANNED_METRIC: &str = "fragments_scanned";
pub const RANGES_SCANNED_METRIC: &str = "ranges_scanned";
pub const ROWS_SCANNED_METRIC: &str = "rows_scanned";
pub const TASK_WAIT_TIME_METRIC: &str = "task_wait_time";
/// Time a scan's reads waited for room in the session's read memory budget.
pub const READ_BUDGET_WAIT_TIME_METRIC: &str = "read_budget_wait_time";
pub const DECODE_TIME_METRIC: &str = "decode_time";
pub const DELTAS_SEARCHED_METRIC: &str = "deltas_searched";
pub const PARTITIONS_SEARCHED_METRIC: &str = "partitions_searched";
/// Index segments whose candidate search was omitted because the completed
/// prefilter proved they hold no selectable row. Index loading and
/// preparation are not skipped: a vector delta is counted once at the
/// early-search barrier, after its partitions were loaded and its scoring
/// skipped; a full-text segment is counted after every segment's statistics
/// were prepared.
pub const INDEX_SEGMENTS_SKIPPED_METRIC: &str = "index_segments_skipped";
pub const FIND_PARTITIONS_ELAPSED_METRIC: &str = "find_partitions_elapsed";
pub const SCALAR_INDEX_SEARCH_TIME_METRIC: &str = "search_time";
pub const SCALAR_INDEX_SER_TIME_METRIC: &str = "serialization_time";
pub const SCALAR_INDEX_MIN_ALLOWED_METRIC: &str = "min_allowed";
pub const SCALAR_INDEX_MIN_BLOCKED_METRIC: &str = "min_blocked";
pub const SCALAR_INDEX_MAX_ALLOWED_METRIC: &str = "max_allowed";
pub const SCALAR_INDEX_MAX_BLOCKED_METRIC: &str = "max_blocked";

#[cfg(test)]
mod tests {
    use arrow_array::record_batch;
    use futures::stream;

    use super::*;

    #[tokio::test]
    async fn write_source_preserves_datafusion_error_after_first_batch() {
        let batch = record_batch!(("id", Int32, [1])).unwrap();
        let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            batch.schema(),
            stream::iter([
                Ok(batch.clone()),
                Err(DataFusionError::Plan("bad plan".into())),
            ]),
        ));

        let (mut stream, _) = input.into_stream_and_schema().await.unwrap();
        assert_eq!(stream.try_next().await.unwrap(), Some(batch));
        let error = stream.try_next().await.unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(message) if message == "bad plan"));
    }
}
