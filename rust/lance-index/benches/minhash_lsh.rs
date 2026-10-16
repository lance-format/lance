// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_index::Index;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pb;
use lance_index::pbold;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::minhash_lsh::{
    MinHashLshIndex, MinHashLshIndexBuilder, MinHashLshIndexParams, SIGNATURE_VERSION,
};
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_io::object_store::ObjectStore;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use object_store::path::Path;

fn bench_minhash_lsh(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("minhash_lsh");
    group.sample_size(10);

    for (name, num_hashes, num_bands, dense_prefix, is_dense) in [
        ("sparse_256", 256, 256, 0, false),
        ("dense_prefix_256", 256, 256, 8, false),
        ("sparse_64", 128, 64, 0, false),
        ("sparse_32", 128, 32, 0, false),
        ("default_16", 128, 16, 0, false),
        ("dense_64", 64, 64, 0, true),
    ] {
        let tempdir = tempfile::tempdir().unwrap();
        let path = Path::from_filesystem_path(tempdir.path()).unwrap();
        let store = runtime.block_on(async {
            Arc::new(LanceIndexStore::new(
                Arc::new(ObjectStore::local()),
                path,
                Arc::new(LanceCache::no_cache()),
            ))
        });
        let params = MinHashLshIndexParams {
            num_hashes,
            num_bands,
            ..Default::default()
        };
        let query = (0..40)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let texts: Vec<String> = (0..20_000)
            .map(|row| {
                if is_dense || row < dense_prefix {
                    query.clone()
                } else {
                    (0..40)
                        .map(|word| {
                            if word < 4 {
                                format!("word{}", (row % 37) + word)
                            } else {
                                format!("unique{row}_{word}")
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(texts)) as ArrayRef,
                Arc::new(UInt64Array::from_iter_values(0..20_000)) as ArrayRef,
            ],
        )
        .unwrap();
        runtime.block_on(async {
            let input = RecordBatchStreamAdapter::new(schema, stream::iter(vec![Ok(batch)]));
            MinHashLshIndexBuilder::try_new(params.clone())
                .unwrap()
                .train(Box::pin(input), store.as_ref())
                .await
                .unwrap();
        });
        let details = prost_types::Any::from_msg(&pb::MinHashLshIndexDetails {
            num_hashes: params.num_hashes,
            num_bands: params.num_bands,
            shingle_size: params.shingle_size,
            tokenizer: Some(pbold::InvertedIndexDetails::try_from(&params.tokenizer).unwrap()),
            signature_version: SIGNATURE_VERSION,
        })
        .unwrap();
        let cold_cache = LanceCache::no_cache();
        let warm_cache = LanceCache::with_capacity(64 << 20);
        let (cold, warm) = runtime.block_on(async {
            let cold = MinHashLshIndex::load(store.clone(), &details, None, &cold_cache)
                .await
                .unwrap();
            let warm = MinHashLshIndex::load(store.clone(), &details, None, &warm_cache)
                .await
                .unwrap();
            warm.prewarm().await.unwrap();
            (cold, warm)
        });
        let signature = warm.query_signature(&query).unwrap();
        let all = RowAddrMask::all_rows();
        let hits = runtime
            .block_on(warm.search_signature(&signature, 1, &all, &NoOpMetricsCollector))
            .unwrap();
        // With several hashes per band this sparse corpus may have no collisions.
        // These cases still measure the overhead of an empty bucket scan.
        if num_bands >= 64 || is_dense || dense_prefix > 0 {
            assert!(!hits.is_empty());
        }
        let empty = RowAddrMask::from_allowed(RowAddrTreeMap::default());
        for (case, index, mask) in [
            ("cold_query", &cold, &all),
            ("prewarmed_query", &warm, &all),
            // Rejected rows force a full cursor walk even when limit is one.
            ("prewarmed_cursor_walk", &warm, &empty),
        ] {
            group.bench_with_input(BenchmarkId::new(case, name), &index, |b, index| {
                b.to_async(&runtime).iter(|| async {
                    black_box(
                        index
                            .search_signature(&signature, 1, mask, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    )
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_minhash_lsh);
criterion_main!(benches);
