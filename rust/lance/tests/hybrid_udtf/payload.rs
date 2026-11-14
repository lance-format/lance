// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use futures::stream::BoxStream;
use lance::dataset::NewColumnTransform;
use lance_file::reader::FileReaderOptions;
use lance_io::object_store::WrappingObjectStore;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result as StoreResult,
};
use tokio::sync::{Notify, Semaphore};

async fn payload_dataset(stable: bool) -> Dataset {
    let mut data = dataset(stable, false).await;
    // A separate file lets the controlled store identify payload I/O precisely:
    // neither candidate branch needs this column.
    data.add_columns(
        NewColumnTransform::SqlExpressions(vec![(
            "payload".into(),
            "repeat(cast(id AS string), CASE WHEN id = 2 THEN 8192 ELSE 256 END)".into(),
        )]),
        None,
        None,
    )
    .await
    .unwrap();
    data
}

#[rstest]
#[case::stable(true)]
#[case::addresses(false)]
#[tokio::test]
async fn hybrid_payload_batch_bytes(#[case] stable: bool) {
    let data = payload_dataset(stable).await;
    let baseline = context(Arc::new(data.clone()), 4, 1);
    let sql = format!(
        "SELECT payload, _relevance_score, _rowid FROM {}",
        relation(json!({}))
    );
    let expected = query(&baseline, &sql).await;
    let budget = 512;
    let ctx = context(
        Arc::new(data.with_file_reader_options(Some(FileReaderOptions {
            batch_size_bytes: Some(budget),
            read_chunk_size: 1024,
            ..Default::default()
        }))),
        4,
        1,
    );
    let plan = ctx
        .sql(&sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let readers = final_payload_readers(&plan);
    assert_eq!(readers.len(), 1);
    let reader = readers[0].downcast_ref::<FilteredReadExec>().unwrap();
    let options = reader.options().file_reader_options.as_ref().unwrap();
    assert_eq!(options.batch_size_bytes, Some(budget));
    assert_eq!(options.read_chunk_size, 1024);
    // Inspect the production final reader before any outer SQL operator can
    // coalesce its output. Splitting follows decode, so this is not a RAM cap.
    let batches = collect(readers[0].clone(), ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert!(batches.iter().any(|batch| {
        batch.num_rows() == 1 && batch.get_array_memory_size() > 2 * budget as usize
    }));
    for batch in &batches {
        assert!(batch.num_rows() == 1 || batch.get_array_memory_size() <= 2 * budget as usize);
    }
    let actual = query(&ctx, &sql).await;
    assert_eq!(actual, expected);
    let payload = arrow_cast::cast(&actual["payload"], &DataType::Utf8).unwrap();
    assert_eq!(payload.as_string::<i32>().value(2), "2".repeat(8192));
}

#[derive(Debug)]
struct PayloadReads {
    paths: Vec<String>,
    gate: Semaphore,
    changed: Notify,
    started: AtomicUsize,
    active: AtomicUsize,
    fail: AtomicBool,
}

struct ActiveRead<'a>(&'a PayloadReads);

impl Drop for ActiveRead<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
        self.0.changed.notify_one();
    }
}

impl PayloadReads {
    async fn wait(&self, predicate: impl Fn() -> bool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let changed = self.changed.notified();
                if predicate() {
                    break;
                }
                changed.await;
            }
        })
        .await
        .expect("payload read state did not converge");
    }

    async fn read(&self, location: &Path) -> StoreResult<Option<ActiveRead<'_>>> {
        if !self
            .paths
            .iter()
            .any(|path| location.as_ref().ends_with(path))
        {
            return Ok(None);
        }
        self.active.fetch_add(1, Ordering::SeqCst);
        let active = ActiveRead(self);
        self.started.fetch_add(1, Ordering::SeqCst);
        self.changed.notify_one();
        if let Ok(permit) = self.gate.acquire().await {
            permit.forget();
        }
        if self.fail.load(Ordering::SeqCst) {
            return Err(object_store::Error::Generic {
                store: "hybrid payload test",
                source: std::io::Error::other("injected payload read failure").into(),
            });
        }
        Ok(Some(active))
    }
}

#[derive(Debug)]
struct PayloadWrapper(Arc<PayloadReads>);

impl WrappingObjectStore for PayloadWrapper {
    fn wrap(&self, _: &str, target: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(PayloadStore {
            target,
            reads: self.0.clone(),
        })
    }

    fn wrap_paginated(
        &self,
        _: &str,
        original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(original)
    }
}

#[derive(Debug)]
struct PayloadStore {
    target: Arc<dyn ObjectStore>,
    reads: Arc<PayloadReads>,
}

impl std::fmt::Display for PayloadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PayloadStore")
    }
}

// Delegate storage unchanged except for the payload-only read barrier, following
// the existing controlled-store fixtures. RAII counts cancelled reads too.
#[async_trait::async_trait]
impl ObjectStore for PayloadStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> StoreResult<PutResult> {
        self.target.put_opts(path, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        opts: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.target.put_multipart_opts(path, opts).await
    }
    async fn get_opts(&self, path: &Path, opts: GetOptions) -> StoreResult<GetResult> {
        let _active = if opts.head {
            None
        } else {
            self.reads.read(path).await?
        };
        self.target.get_opts(path, opts).await
    }
    async fn get_ranges(&self, path: &Path, ranges: &[Range<u64>]) -> StoreResult<Vec<Bytes>> {
        let _active = self.reads.read(path).await?;
        self.target.get_ranges(path, ranges).await
    }
    fn delete_stream(
        &self,
        paths: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        self.target.delete_stream(paths)
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.target.list(prefix)
    }
    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        self.target.list_with_offset(prefix, offset)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }
    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> StoreResult<()> {
        self.target.copy_opts(from, to, opts).await
    }
    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> StoreResult<()> {
        self.target.rename_opts(from, to, opts).await
    }
}

#[rstest]
#[case::cancel_stable(true, "cancel")]
#[case::cancel_addresses(false, "cancel")]
#[case::error_stable(true, "error")]
#[case::error_addresses(false, "error")]
#[case::complete(true, "complete")]
#[tokio::test]
async fn hybrid_payload_lifecycle(#[case] stable: bool, #[case] outcome: &str) {
    let data = payload_dataset(stable).await;
    let payload_id = data.schema().field("payload").unwrap().id;
    let files = data
        .fragments()
        .iter()
        .flat_map(|fragment| &fragment.files)
        .filter(|file| file.fields.contains(&payload_id))
        .collect::<Vec<_>>();
    assert_eq!(files.len(), 2);
    assert!(
        files
            .iter()
            .all(|file| file.fields.as_ref() == [payload_id])
    );
    let reads = Arc::new(PayloadReads {
        paths: files.iter().map(|file| file.path.clone()).collect(),
        gate: Semaphore::new(0),
        changed: Notify::new(),
        started: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        fail: AtomicBool::new(outcome == "error"),
    });
    let ctx = context(
        Arc::new(data.with_object_store_wrappers([
            Arc::new(PayloadWrapper(reads.clone())) as Arc<dyn WrappingObjectStore>
        ])),
        4,
        1,
    );
    let no_payload = query(
        &ctx,
        &format!(
            "SELECT _rowid, _relevance_score FROM {}",
            relation(json!({}))
        ),
    )
    .await;
    assert_eq!(no_payload.num_rows(), 3);
    assert_eq!(reads.started.load(Ordering::SeqCst), 0);
    let plan = ctx
        .sql(&format!(
            "SELECT payload, _relevance_score FROM {}",
            relation(json!({}))
        ))
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let readers = final_payload_readers(&plan);
    assert_eq!(readers.len(), 1);
    // RowStreamRead retains its source, so expiration observes the actual
    // payload task lifetime rather than just the outer ExecutionPlan handle.
    let source = Arc::downgrade(
        readers[0]
            .downcast_ref::<FilteredReadExec>()
            .unwrap()
            .row_stream_input()
            .unwrap(),
    );
    drop(readers);
    let pool = ctx.runtime_env().memory_pool.clone();
    let mut pending = Box::pin(collect(plan.clone(), ctx.task_ctx()));
    tokio::select! {
        result = &mut pending => panic!("query completed before payload barrier: {result:?}"),
        () = reads.wait(|| reads.active.load(Ordering::SeqCst) > 0) => {}
    }
    assert!(reads.started.load(Ordering::SeqCst) > 0);
    if outcome == "cancel" {
        drop(pending);
    } else {
        // Ordered readers may await an earlier request before surfacing a later
        // error. Release submitted I/O; only cancellation asserts pre-drain drop.
        reads.gate.close();
        let result = tokio::time::timeout(Duration::from_secs(5), pending)
            .await
            .unwrap();
        if outcome == "error" {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("injected payload read failure")
            );
        } else {
            assert_eq!(
                result
                    .unwrap()
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>(),
                3
            );
        }
    }
    drop(plan);
    drop(ctx);
    // Prove task cancellation before releasing storage: detached tasks that
    // simply finish after the barrier opens must not satisfy this assertion.
    tokio::time::timeout(Duration::from_secs(5), async {
        while source.upgrade().is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("payload tasks retained their source while storage was blocked");
    // Already-submitted storage requests may drain after the row-batch tasks
    // disappear; cancellation does not promise synchronous I/O preemption.
    reads.gate.close();
    tokio::time::timeout(Duration::from_secs(5), async {
        while reads.active.load(Ordering::SeqCst) != 0 || source.upgrade().is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("payload tasks retained their source after completion");
    let settled = reads.started.load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(reads.started.load(Ordering::SeqCst), settled);
    assert_eq!(reads.active.load(Ordering::SeqCst), 0);
    assert_eq!(pool.reserved(), 0);
}
