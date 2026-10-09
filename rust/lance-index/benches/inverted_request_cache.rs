// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Isolate warm FTS request-open overhead from query scoring and cloud latency.

use std::{
    collections::HashMap,
    hint::black_box,
    sync::{Arc, Mutex},
    time::Duration,
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::{FutureExt, future::try_join_all};
use lance_core::cache::LanceCache;
use lance_core::utils::address::RowAddress;
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_index::Index;
use lance_index::scalar::inverted::{
    DocSet, FTS_FORMAT_VERSION_KEY, InvertedIndex, InvertedIndexParams, InvertedIndexPlugin,
    METADATA_FILE, POSTING_BLOCK_SIZE_KEY, PostingListBuilder, TOKEN_SET_FORMAT_KEY, TokenSet,
    TokenSetFormat,
    builder::{InnerBuilder, PositionRecorder},
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{IndexStore, registry::ScalarIndexPlugin};
use lance_io::object_store::{ObjectStore, WrappingObjectStore};
use object_store::{list::PaginatedListStore, path::Path};

#[derive(Debug)]
struct RequestWrapper;

impl WrappingObjectStore for RequestWrapper {
    fn wrap(
        &self,
        _prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(ProxyObjectStore::new(
            original,
            Arc::new(Mutex::new(ProxyObjectStorePolicy::new())),
        ))
    }

    fn wrap_paginated(
        &self,
        _prefix: &str,
        _original: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        None
    }
}

struct Fixture {
    object_store: ObjectStore,
    metadata_cache: Arc<LanceCache>,
    index_cache: LanceCache,
    file_sizes: HashMap<String, u64>,
}

impl Fixture {
    async fn new(partitions: u64, with_file_sizes: bool) -> Self {
        let object_store = ObjectStore::memory();
        let metadata_cache = Arc::new(LanceCache::with_capacity(64 * 1024 * 1024));
        let index_cache = LanceCache::with_capacity(64 * 1024 * 1024);
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(object_store.clone()),
            Path::from("index"),
            metadata_cache.clone(),
        ));
        let params = InvertedIndexParams::default();
        let format = params.resolved_format_version();
        let mut file_sizes = HashMap::new();
        for id in 0..partitions {
            let mut builder = InnerBuilder::new_with_format_version_and_block_size(
                id,
                false,
                TokenSetFormat::default(),
                format,
                params.posting_block_size(),
            );
            let mut tokens = TokenSet::default();
            tokens.add("alpha".to_owned());
            let mut docs = DocSet::default();
            let mut posting = PostingListBuilder::new_with_posting_tail_codec_and_block_size(
                false,
                format.posting_tail_codec(),
                params.posting_block_size(),
            );
            for row in 0..32 {
                let doc = docs.append(RowAddress::new_from_parts(id as u32, row).into(), 1);
                posting.add(doc, PositionRecorder::Count(1));
            }
            builder.set_tokens(tokens);
            builder.set_docs(docs);
            builder.set_posting_lists(vec![posting]);
            for file in builder.write(store.as_ref()).await.unwrap() {
                file_sizes.insert(file.path, file.size_bytes);
            }
        }
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        let file = writer
            .finish_with_metadata(HashMap::from([
                ("params".to_owned(), serde_json::to_string(&params).unwrap()),
                (
                    "partitions".to_owned(),
                    serde_json::to_string(&(0..partitions).collect::<Vec<_>>()).unwrap(),
                ),
                (
                    TOKEN_SET_FORMAT_KEY.to_owned(),
                    TokenSetFormat::default().to_string(),
                ),
                (
                    FTS_FORMAT_VERSION_KEY.to_owned(),
                    format.index_version().to_string(),
                ),
                (
                    POSTING_BLOCK_SIZE_KEY.to_owned(),
                    params.posting_block_size().to_string(),
                ),
            ]))
            .await
            .unwrap();
        file_sizes.insert(file.path, file.size_bytes);
        let index = InvertedIndex::load(store.clone(), None, &index_cache)
            .await
            .unwrap();
        index.prewarm().await.unwrap();
        InvertedIndexPlugin
            .put_in_cache(store, &index_cache, index)
            .await
            .unwrap();
        if !with_file_sizes {
            file_sizes.clear();
        }
        Self {
            object_store,
            metadata_cache,
            index_cache,
            file_sizes,
        }
    }

    async fn open_request(&self) {
        let mut object_store = self.object_store.clone();
        object_store.apply_wrapper(&RequestWrapper);
        let store = Arc::new(
            LanceIndexStore::new(
                Arc::new(object_store),
                Path::from("index"),
                self.metadata_cache.clone(),
            )
            .with_file_sizes(self.file_sizes.clone()),
        );
        let index = InvertedIndexPlugin
            .get_or_insert_in_cache(
                store,
                None,
                &self.index_cache,
                async { panic!("prewarmed index metadata should remain cached") }.boxed(),
            )
            .await
            .unwrap();
        black_box(index);
    }
}

fn request_cache(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("inverted_request_open");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    for partitions in [1, 32, 256] {
        for file_sizes in [false, true] {
            let fixture = Arc::new(runtime.block_on(Fixture::new(partitions, file_sizes)));
            for concurrency in [1, 64] {
                group.throughput(Throughput::Elements(concurrency));
                let scenario = format!("p{partitions}_sizes{file_sizes}");
                group.bench_with_input(
                    BenchmarkId::new(scenario, concurrency),
                    &concurrency,
                    |b, &concurrency| {
                        b.to_async(&runtime).iter(|| async {
                            let tasks = (0..concurrency).map(|_| {
                                let fixture = fixture.clone();
                                tokio::spawn(async move { fixture.open_request().await })
                            });
                            try_join_all(tasks).await.unwrap();
                            // Test-util builds retain individual I/O records. Bound
                            // their lifetime to a batch on both compared revisions.
                            black_box(fixture.object_store.io_stats_incremental());
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, request_cache);
criterion_main!(benches);
