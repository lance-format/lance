// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    fmt,
    sync::{Arc, Mutex},
    time::Duration,
};

use arrow_array::{Int32Array, RecordBatch, cast::AsArray, types::Int32Type};
use datafusion::{
    common::Result as DataFusionResult,
    execution::{
        TaskContext,
        memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation},
        runtime_env::RuntimeEnvBuilder,
    },
    physical_plan::ExecutionPlan,
};
use futures::TryStreamExt;
use lance::{
    Dataset,
    dataset::{WriteParams, builder::DatasetBuilder},
    session::Session,
};
use lance_core::utils::{memory::MemoryReserver, tempfile::TempStrDir};
use lance_datafusion::{exec::StrictBatchSizeExec, resources::DataFusionMemoryReserver};
use lance_file::version::LanceFileVersion;
use lance_io::read_budget::ReadMemoryBudget;
use rstest::rstest;

const ROWS: usize = 32 * 1024;

#[derive(Debug, Clone)]
struct Admission {
    stage: String,
    requested: usize,
    before: usize,
    accepted: bool,
}

#[derive(Debug, Default)]
struct Activity {
    admissions: Vec<Admission>,
    peak: usize,
}

#[derive(Debug)]
struct RecordingPool {
    inner: GreedyMemoryPool,
    activity: Mutex<Activity>,
}

impl RecordingPool {
    fn new(cap: usize) -> Self {
        Self {
            inner: GreedyMemoryPool::new(cap),
            activity: Mutex::default(),
        }
    }

    fn record(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
        action: impl FnOnce() -> DataFusionResult<()>,
    ) -> DataFusionResult<()> {
        let mut activity = self.activity.lock().unwrap();
        let before = self.inner.reserved();
        let result = action();
        activity.admissions.push(Admission {
            stage: reservation.consumer().name().to_owned(),
            requested: additional,
            before,
            accepted: result.is_ok(),
        });
        activity.peak = activity.peak.max(self.inner.reserved());
        result
    }
}

impl fmt::Display for RecordingPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("recorded scan pool")
    }
}

impl MemoryPool for RecordingPool {
    fn name(&self) -> &str {
        "recorded scan pool"
    }
    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.record(reservation, additional, || {
            self.inner.grow(reservation, additional);
            Ok(())
        })
        .unwrap();
    }
    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        let _activity = self.activity.lock().unwrap();
        self.inner.shrink(reservation, shrink);
    }
    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> DataFusionResult<()> {
        self.record(reservation, additional, || {
            self.inner.try_grow(reservation, additional)
        })
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
}

fn value(row: usize) -> Option<i32> {
    (!row.is_multiple_of(257)).then_some(row as i32)
}

async fn fixture(directory: &TempStrDir) -> (Dataset, Arc<Session>) {
    let batches = (0..2)
        .map(|fragment| {
            let column = Int32Array::from_iter((fragment * ROWS..(fragment + 1) * ROWS).map(value));
            Ok(RecordBatch::try_from_iter([("id", Arc::new(column) as _)]).unwrap())
        })
        .collect::<Vec<_>>();
    let schema = batches[0].as_ref().unwrap().schema();
    Dataset::write(
        arrow_array::RecordBatchIterator::new(batches, schema),
        directory.as_str(),
        Some(WriteParams {
            max_rows_per_file: ROWS,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let session = Arc::new(Session::default().with_read_memory_budget(ReadMemoryBudget::new(1)));
    let dataset = DatasetBuilder::from_uri(directory.as_str())
        .with_session(session.clone())
        .load()
        .await
        .unwrap();
    assert_eq!(dataset.fragments().len(), 2);
    (dataset, session)
}

fn context(pool: Arc<dyn MemoryPool>) -> Arc<TaskContext> {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool)
        .build_arc()
        .unwrap();
    Arc::new(
        DataFusionMemoryReserver::configure(TaskContext::default().with_runtime(runtime)).unwrap(),
    )
}

async fn plan(dataset: &Dataset, rows: usize, fragment: Option<usize>) -> Arc<dyn ExecutionPlan> {
    let mut scanner = dataset.scan();
    scanner
        .batch_size(rows)
        .batch_readahead(1)
        .fragment_readahead(1)
        .strict_batch_size(true)
        .with_strict_batch_memory_accounting(true);
    if let Some(fragment) = fragment {
        scanner.with_fragments(vec![dataset.fragments()[fragment].clone()]);
    }
    let plan = scanner.create_plan().await.unwrap();
    assert!(
        plan.downcast_ref::<StrictBatchSizeExec>().is_some(),
        "{plan:?}"
    );
    plan
}

fn verify(batch: &RecordBatch, start: usize, rows: usize) {
    assert_eq!(batch.num_rows(), rows);
    assert!(
        batch["id"]
            .as_primitive::<Int32Type>()
            .iter()
            .eq((start..start + rows).map(value))
    );
}

#[tokio::test]
async fn strict_batch_read_overdraft_obeys_hard_pool_and_retries_after_release() {
    const CAP: usize = 256 * 1024;
    let directory = TempStrDir::default();
    let (dataset, session) = fixture(&directory).await;
    let policy = session.read_budget().unwrap();
    let pool = Arc::new(RecordingPool::new(CAP));
    let context = context(pool.clone());
    let tasks = DataFusionMemoryReserver::from_context(&context)
        .unwrap()
        .unwrap()
        .task_group()
        .unwrap();
    let direct = MemoryConsumer::new("direct DataFusion consumer").register(context.memory_pool());
    direct.try_grow(64).unwrap();
    let baseline = pool.reserved();

    let pipeline = plan(&dataset, ROWS + 1, None).await;
    let mut output = pipeline.execute(0, context.clone()).unwrap();
    let error = tokio::time::timeout(Duration::from_secs(5), output.try_next())
        .await
        .unwrap()
        .unwrap_err();
    assert!(
        error.to_string().contains("strict scan decoded backing"),
        "{error}"
    );
    let (refused, peak) = {
        let activity = pool.activity.lock().unwrap();
        let refused = activity
            .admissions
            .iter()
            .position(|event| !event.accepted)
            .unwrap();
        assert!(
            activity.admissions[..refused]
                .iter()
                .any(|event| event.stage == "strict batch input directory" && event.accepted)
        );
        assert!(
            activity
                .admissions
                .iter()
                .any(|event| event.stage == "direct DataFusion consumer")
        );
        (activity.admissions[refused].clone(), activity.peak)
    };
    assert!(refused.before >= baseline + ROWS * 4);
    assert!(refused.before + refused.requested > CAP);
    assert!(peak <= CAP);
    assert!(policy.stats().peak_reserved_bytes > policy.limit());

    drop((output, pipeline));
    tasks.wait().await;
    assert_eq!(policy.reserved_bytes(), 0);
    assert_eq!(pool.reserved(), baseline);
    let retry_start = pool.activity.lock().unwrap().admissions.len();
    let retry = plan(&dataset, ROWS + 1, Some(1)).await;
    let mut output = retry.execute(0, context.clone()).unwrap();
    let batch = output.try_next().await.unwrap().unwrap();
    verify(&batch, ROWS, ROWS);
    assert!(output.try_next().await.unwrap().is_none());
    assert!(
        pool.activity.lock().unwrap().admissions[retry_start..]
            .iter()
            .any(|event| event.stage == refused.stage
                && event.requested == refused.requested
                && event.accepted)
    );
    drop((batch, output, retry, direct, context));
    tasks.wait().await;
    assert_eq!(policy.reserved_bytes(), 0);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::zero_copy(false)]
#[case::concat(true)]
#[tokio::test]
async fn strict_batch_last_slice_retains_one_backing(#[case] concat: bool) {
    let directory = TempStrDir::default();
    let (dataset, session) = fixture(&directory).await;
    let pool = Arc::new(RecordingPool::new(4 * 1024 * 1024));
    let context = context(pool.clone());
    let tasks = DataFusionMemoryReserver::from_context(&context)
        .unwrap()
        .unwrap()
        .task_group()
        .unwrap();
    let rows = if concat { ROWS + ROWS / 2 } else { ROWS };
    let plan = plan(&dataset, rows, (!concat).then_some(0)).await;
    let mut output = plan.execute(0, context.clone()).unwrap();
    let batch = output.try_next().await.unwrap().unwrap();
    verify(&batch, 0, rows);
    let first_slice = batch.slice(0, 1);
    let last_slice = batch.slice(rows - 1, 1);
    drop(batch);

    if concat {
        let suffix = output.try_next().await.unwrap().unwrap();
        verify(&suffix, rows, 2 * ROWS - rows);
        drop(suffix);
    }
    assert!(output.try_next().await.unwrap().is_none());
    drop((output, plan, context, dataset, session));
    tasks.wait().await;

    let activity = pool.activity.lock().unwrap();
    let control = activity
        .admissions
        .iter()
        .filter(|event| event.stage == "inline allocation control")
        .map(|event| event.requested)
        .collect::<Vec<_>>();
    assert_eq!(control.len(), usize::from(concat));
    // The two Arrow buffers are aligned and both row counts are multiples of
    // 512, so neither buffer needs alignment padding. This expectation does
    // not use the pool balance or an observed backing admission.
    let expected =
        rows * size_of::<i32>() + rows.div_ceil(8) + 1024 + control.iter().sum::<usize>();
    drop(activity);
    assert_eq!(pool.reserved(), expected);
    assert!(first_slice["id"].is_null(0));
    drop(first_slice);
    assert_eq!(pool.reserved(), expected);
    verify(&last_slice, rows - 1, 1);
    drop(last_slice);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn strict_batch_scanner_creates_accounted_context() {
    let directory = TempStrDir::default();
    let (dataset, session) = fixture(&directory).await;
    let mut scanner = dataset.scan();
    scanner
        .batch_size(ROWS + 1)
        .batch_readahead(1)
        .fragment_readahead(1)
        .strict_batch_size(true)
        .with_strict_batch_memory_accounting(true);
    let mut output = scanner.try_into_stream().await.unwrap();
    verify(&output.try_next().await.unwrap().unwrap(), 0, ROWS + 1);
    verify(
        &output.try_next().await.unwrap().unwrap(),
        ROWS + 1,
        ROWS - 1,
    );
    assert!(output.try_next().await.unwrap().is_none());
    assert_eq!(session.read_budget().unwrap().reserved_bytes(), 0);
}
