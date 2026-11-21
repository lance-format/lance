// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Listing concurrency for the managed subtrees cleanup sweeps.
//!
//! Cleanup lists `_versions`, `_transactions`, `data`, `_blobs`, `_indices`,
//! `_deletions`, and `_fri` independently. Consuming those streams serially
//! makes the pass cost one LIST round trip per subtree. Nothing in the
//! assertions below depends on how long listing takes, only on whether more
//! than one subtree is in flight at a time, so they stay valid if the listing
//! strategy is revisited.

use super::*;

/// Counts how many `list` streams cleanup is consuming at the same time.
#[derive(Debug, Default)]
struct InFlightLists {
    current: AtomicUsize,
    max: AtomicUsize,
    started: AtomicUsize,
}

impl InFlightLists {
    fn enter(self: &Arc<Self>) -> ListGuard {
        let current = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(current, Ordering::SeqCst);
        self.started.fetch_add(1, Ordering::SeqCst);
        ListGuard {
            lists: Arc::clone(self),
        }
    }

    fn max(&self) -> usize {
        self.max.load(Ordering::SeqCst)
    }

    fn started(&self) -> usize {
        self.started.load(Ordering::SeqCst)
    }
}

#[derive(Debug)]
struct ListGuard {
    lists: Arc<InFlightLists>,
}

impl Drop for ListGuard {
    fn drop(&mut self) {
        self.lists.current.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A store that holds each `list` stream open for `latency` before its first
/// item and records the peak number of streams open at once.
///
/// Without the delay the local backend may interleave listings by accident, so
/// the delay makes a serial merge observably serial: it cannot begin the next
/// subtree until the previous stream is exhausted.
#[derive(Debug)]
struct ListingLatencyStore {
    inner: Arc<dyn object_store::ObjectStore>,
    lists: Arc<InFlightLists>,
    latency: Duration,
}

impl std::fmt::Display for ListingLatencyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ListingLatencyStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for ListingLatencyStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, bytes, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let inner = self.inner.list(prefix);
        let lists = Arc::clone(&self.lists);
        let latency = self.latency;
        // `unfold` keeps the guard in the stream's state so the slot is held
        // until the subtree is exhausted, including when the stream is dropped
        // early by an error.
        futures::stream::unfold(Some((inner, None::<ListGuard>)), move |state| {
            let lists = Arc::clone(&lists);
            async move {
                let (mut inner, guard) = state?;
                let guard = match guard {
                    Some(guard) => guard,
                    None => {
                        let guard = lists.enter();
                        tokio::time::sleep(latency).await;
                        guard
                    }
                };
                inner
                    .next()
                    .await
                    .map(|item| (item, Some((inner, Some(guard)))))
            }
        })
        .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[derive(Debug)]
struct ListingLatencyWrapper {
    lists: Arc<InFlightLists>,
    latency: Duration,
}

impl WrappingObjectStore for ListingLatencyWrapper {
    fn wrap(
        &self,
        _storage_prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(ListingLatencyStore {
            inner: original,
            lists: Arc::clone(&self.lists),
            latency: self.latency,
        })
    }

    // The delay has to sit inside every listing, so a pushed-down paginated
    // listing would go around it.
    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}

/// Cleanup must overlap the managed subtree listings, not drain them one at a
/// time. A serial merge peaks at one open listing no matter how many subtrees
/// exist.
#[tokio::test]
async fn managed_subtree_listings_overlap() {
    const LATENCY: Duration = Duration::from_millis(50);
    let lists = Arc::new(InFlightLists::default());
    let dir = TempStrDir::default();
    let path_prefix = if dir.as_str().starts_with('/') {
        ""
    } else {
        "/"
    };
    let path = format!("file-object-store://{path_prefix}{}/my_db", dir.as_str());
    let store_params = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(ListingLatencyWrapper {
            lists: Arc::clone(&lists),
            latency: LATENCY,
        })),
        ..Default::default()
    };

    // Writing through the wrapper would pay the listing delay, so build the
    // dataset through the plain store and reopen it through the wrapper.
    let rows = lance_datagen::gen_batch()
        .col(
            "id",
            lance_datagen::array::step::<arrow_array::types::Int32Type>(),
        )
        .into_reader_rows(4.into(), 4.into());
    Dataset::write(rows, &path, Some(WriteParams::default()))
        .await
        .unwrap();
    let dataset = DatasetBuilder::from_uri(&path)
        .with_read_params(ReadParams {
            store_options: Some(store_params),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();

    // Explain lists every managed subtree and deletes nothing, so this
    // exercises the sweep without a cleanup-time side effect.
    dataset
        .cleanup(CleanupPolicyBuilder::default().build())
        .explain()
        .await
        .unwrap();

    assert!(
        lists.started() >= 2,
        "expected several managed subtree listings, saw {}",
        lists.started()
    );
    assert!(
        lists.max() >= 2,
        "managed subtree listings were serial: only {} was open at a time across {} listings",
        lists.max(),
        lists.started()
    );
}
