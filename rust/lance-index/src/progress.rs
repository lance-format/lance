// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use async_trait::async_trait;
use datafusion_physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::StreamExt;
use lance_core::Result;
use std::pin::Pin;
use std::sync::Arc;

/// Progress callback for index building and distributed index finalization.
///
/// Called at stage boundaries during index construction. For a single logical stream, stages are
/// sequential: `stage_complete` is always called before the next `stage_start`, so only one stage
/// is active at a time. Callers that orchestrate independent sub-builds in parallel may prefix
/// stage names (for example `segment_plan[0]/merge_partitions`) to represent separate logical
/// streams. Stage names are index-type-specific (e.g. "train_ivf", "shuffle", "merge_partitions"
/// for vector indices; "load_data", "sort_values" for scalar indices; merge/finalization stages
/// for distributed index construction).
///
/// Methods take `&self` to allow concurrent calls from within a single stage. Implementations
/// must be thread-safe.
#[async_trait]
pub trait IndexBuildProgress: std::fmt::Debug + Sync + Send {
    /// A named stage has started.
    ///
    /// `total` is the number of work units if known, and `unit` describes
    /// what is being counted (e.g. "partitions", "batches", "rows").
    async fn stage_start(&self, stage: &str, total: Option<u64>, unit: &str) -> Result<()>;

    /// Progress within the current stage.
    async fn stage_progress(&self, stage: &str, completed: u64) -> Result<()>;

    /// A named stage has completed.
    async fn stage_complete(&self, stage: &str) -> Result<()>;
}

#[derive(Debug, Clone, Default)]
pub struct NoopIndexBuildProgress;

#[async_trait]
impl IndexBuildProgress for NoopIndexBuildProgress {
    async fn stage_start(&self, _: &str, _: Option<u64>, _: &str) -> Result<()> {
        Ok(())
    }
    async fn stage_progress(&self, _: &str, _: u64) -> Result<()> {
        Ok(())
    }
    async fn stage_complete(&self, _: &str) -> Result<()> {
        Ok(())
    }
}

/// Helper to create a default noop progress instance.
pub fn noop_progress() -> Arc<dyn IndexBuildProgress> {
    Arc::new(NoopIndexBuildProgress)
}

/// Reports `stage` until `data` yields its first batch, and returns a stream
/// that yields every batch of `data`, that one included.
///
/// The stage is opened with [`IndexBuildProgress::stage_start`] (no total, unit
/// `rows`) and closed with [`IndexBuildProgress::stage_complete`], with no
/// progress reported in between. A sort reads all of its input before it
/// yields anything, so for a lazily sorted stream the stage covers reading and
/// sorting the data, and the stages that follow cover consuming the sorted
/// output. An empty stream completes the stage at once. If the first batch is
/// an error, the error is returned and the stage does not complete.
///
/// ```
/// # use datafusion_physical_plan::SendableRecordBatchStream;
/// # use lance_core::Result;
/// # use lance_index::progress::{IndexBuildProgress, stage_until_first_batch};
/// # async fn write_index(_: SendableRecordBatchStream) -> Result<()> { Ok(()) }
/// # async fn build(
/// #     sorted: SendableRecordBatchStream,
/// #     progress: &dyn IndexBuildProgress,
/// # ) -> Result<()> {
/// let sorted = stage_until_first_batch(progress, "sort_values", sorted).await?;
/// progress.stage_start("write_index", None, "files").await?;
/// write_index(sorted).await?;
/// progress.stage_complete("write_index").await?;
/// # Ok(())
/// # }
/// ```
pub async fn stage_until_first_batch(
    progress: &dyn IndexBuildProgress,
    stage: &str,
    data: SendableRecordBatchStream,
) -> Result<SendableRecordBatchStream> {
    progress.stage_start(stage, None, "rows").await?;
    let schema = data.schema();
    let mut data = data.peekable();
    if let Some(Err(err)) = Pin::new(&mut data).next_if(|batch| batch.is_err()).await {
        return Err(err.into());
    }
    progress.stage_complete(stage).await?;
    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, data)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::bitmap::test_util::index_store;
    use crate::scalar::label_list::LabelListIndexPlugin;
    use crate::scalar::registry::{BasicTrainer, VALUE_COLUMN_NAME};
    use crate::scalar::zonemap::ZoneMapIndexPlugin;
    use arrow_array::{
        Int32Array, RecordBatch,
        cast::AsArray,
        types::{Int32Type, Int64Type, UInt64Type},
    };
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion_common::DataFusionError;
    use futures::TryStreamExt;
    use lance_core::ROW_ADDR;
    use lance_datagen::{ArrayGenerator, BatchCount, RowCount, array, gen_batch};
    use rstest::rstest;

    lance_testing::define_stage_event_progress!(RecordingProgress, IndexBuildProgress, Result<()>);

    /// Yields `items`, noting `produced` in `progress` as each one is yielded,
    /// so the notes interleave with the stage events.
    fn noting_stream(
        progress: Arc<RecordingProgress>,
        schema: SchemaRef,
        items: Vec<datafusion_common::Result<RecordBatch>>,
    ) -> SendableRecordBatchStream {
        let items = futures::stream::iter(items).inspect(move |_| {
            progress
                .events
                .lock()
                .unwrap()
                .push(("produced".to_string(), String::new(), 0));
        });
        Box::pin(RecordBatchStreamAdapter::new(schema, items))
    }

    fn kinds(events: &[(String, String, u64)]) -> Vec<&str> {
        events.iter().map(|(kind, _, _)| kind.as_str()).collect()
    }

    #[rstest]
    #[case::batches(
        vec![Ok(1), Ok(2), Ok(3)],
        &["start", "produced", "complete", "produced", "produced"],
        Some(vec![1, 2, 3])
    )]
    #[case::empty(vec![], &["start", "complete"], Some(vec![]))]
    #[case::first_batch_fails(vec![Err("boom"), Ok(2)], &["start", "produced"], None)]
    #[tokio::test]
    async fn test_stage_until_first_batch(
        #[case] items: Vec<std::result::Result<i32, &'static str>>,
        #[case] expected_events: &[&str],
        #[case] expected_values: Option<Vec<i32>>,
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let items = items
            .into_iter()
            .map(|item| {
                item.map(|value| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(vec![value]))],
                    )
                    .unwrap()
                })
                .map_err(|message| DataFusionError::Execution(message.to_string()))
            })
            .collect();
        let progress = Arc::new(RecordingProgress::default());
        let data = noting_stream(progress.clone(), schema, items);

        let values = match stage_until_first_batch(progress.as_ref(), "sort_values", data).await {
            Ok(data) => Some(
                data.try_collect::<Vec<_>>()
                    .await
                    .unwrap()
                    .iter()
                    .map(|batch| batch.column(0).as_primitive::<Int32Type>().value(0))
                    .collect::<Vec<_>>(),
            ),
            Err(err) => {
                assert!(err.to_string().contains("boom"), "{err}");
                None
            }
        };

        let events = progress.recorded_events();
        assert_eq!(kinds(&events), expected_events);
        assert!(
            events
                .iter()
                .filter(|(kind, _, _)| kind != "produced")
                .all(|(_, stage, _)| stage == "sort_values"),
            "{events:?}"
        );
        assert_eq!(values, expected_values);
    }

    /// A trainer pulls all of its input inside its first stage. The stage test
    /// in `lance` cannot check this for these two: its scan reads ahead of the
    /// first batch, and its builds reach the zone map trainer only when they
    /// train nothing.
    #[rstest]
    #[case::zonemap(
        &ZoneMapIndexPlugin,
        array::step::<Int64Type>(),
        "train_zones",
        "write_zonemap_index"
    )]
    #[case::label_list(
        &LabelListIndexPlugin,
        array::rand_list(&DataType::Int64, false),
        "sort_values",
        "write_label_list_index"
    )]
    #[tokio::test]
    async fn test_trainer_reads_its_input_in_its_first_stage(
        #[case] trainer: &dyn BasicTrainer,
        #[case] values: Box<dyn ArrayGenerator>,
        #[case] read_stage: &str,
        #[case] write_stage: &str,
    ) {
        let batches = gen_batch()
            .col(VALUE_COLUMN_NAME, values)
            .col(ROW_ADDR, array::step::<UInt64Type>())
            .into_reader_rows(RowCount::from(10), BatchCount::from(3))
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let schema = batches[0].schema();
        let request = trainer.new_training_request("{}", schema.field(0)).unwrap();
        let progress = Arc::new(RecordingProgress::default());
        let data = noting_stream(
            progress.clone(),
            schema,
            batches.into_iter().map(Ok).collect(),
        );
        let (_tmpdir, store) = index_store();

        trainer
            .train_index(data, store.as_ref(), request, None, progress.clone())
            .await
            .unwrap();

        let tags = progress
            .recorded_events()
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        assert_eq!(
            tags,
            [
                format!("start:{read_stage}"),
                "produced:".to_string(),
                "produced:".to_string(),
                "produced:".to_string(),
                format!("complete:{read_stage}"),
                format!("start:{write_stage}"),
                format!("complete:{write_stage}"),
            ]
        );
    }
}
