// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use futures::stream::BoxStream;
use lance_io::object_store::WrappingObjectStore;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result as StoreResult,
};
use std::{
    fmt::Display,
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::sync::{Notify, Semaphore};

#[derive(Debug)]
struct ReadGate {
    armed: AtomicBool,
    started: Notify,
    release: Semaphore,
}

#[derive(Debug)]
struct GateWrapper(Arc<ReadGate>);

impl WrappingObjectStore for GateWrapper {
    fn wrap(&self, _prefix: &str, target: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        Arc::new(GatedStore {
            target,
            gate: self.0.clone(),
        })
    }

    fn wrap_paginated(
        &self,
        _prefix: &str,
        original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(original)
    }
}

#[derive(Debug)]
struct GatedStore {
    target: Arc<dyn ObjectStore>,
    gate: Arc<ReadGate>,
}

impl Display for GatedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GatedStore({})", self.target)
    }
}

#[async_trait::async_trait]
impl ObjectStore for GatedStore {
    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> StoreResult<PutResult> {
        self.target.put_opts(path, payload, options).await
    }
    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        self.target.put_multipart_opts(path, options).await
    }
    async fn get_opts(&self, path: &Path, options: GetOptions) -> StoreResult<GetResult> {
        if path.as_ref().ends_with(".lance") && self.gate.armed.swap(false, Ordering::SeqCst) {
            self.gate.started.notify_one();
            self.gate.release.acquire().await.unwrap().forget();
        }
        self.target.get_opts(path, options).await
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
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }
    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> StoreResult<()> {
        self.target.copy_opts(from, to, options).await
    }
    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> StoreResult<()> {
        self.target.rename_opts(from, to, options).await
    }
}

#[tokio::test]
async fn dataset_cursor_cancels_real_fragment_open_without_skipping_output() {
    let dataset = write_payload_dataset(LanceFileVersion::V2_3, false, false).await;
    let gate = Arc::new(ReadGate {
        armed: AtomicBool::new(true),
        started: Notify::new(),
        release: Semaphore::new(0),
    });
    let wrapped = dataset.with_object_store_wrappers([
        Arc::new(GateWrapper(gate.clone())) as Arc<dyn WrappingObjectStore>
    ]);
    let budget = AllocationBudget::new(65536);
    let mut scan = wrapped
        .try_scan_admitted_fullzip(
            wrapped.schema().clone(),
            Default::default(),
            6,
            65536,
            budget.clone(),
        )
        .unwrap();
    let position = scan.position();
    let mut next = Box::pin(scan.try_next());
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::select! {
            result = &mut next => panic!("gated read must remain pending: {result:?}"),
            () = gate.started.notified() => {}
        }
    })
    .await
    .unwrap();
    drop(next);
    assert_eq!(scan.position(), position);
    // A scheduler may still own the in-flight request. Let that real work end.
    gate.release.add_permits(1);
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let Outcome::Batch {
            fragment_id,
            rows,
            batch,
        } = scan.try_next().await.unwrap()
        else {
            panic!("retry must produce the original prefix");
        };
        assert_eq!(fragment_id, 0);
        assert_eq!(rows, 0..6);
        let reader = open(&dataset, 0, dataset.schema(), false).await;
        assert_eq!(
            batch,
            reader
                .take_as_batch(&[0, 1, 2, 3, 4, 5], None)
                .await
                .unwrap()
        );
        drop(batch);
        budget.wait_for_capacity(65536).await.unwrap();
    })
    .await
    .unwrap();
    assert_eq!(budget.reserved(), 0);
    assert_eq!(
        scan.position(),
        Some(AdmittedScanPosition {
            fragment_id: 1,
            row_offset: 0
        })
    );
}
