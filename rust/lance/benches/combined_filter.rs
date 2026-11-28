// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{env, hint::black_box};

use criterion::{Criterion, criterion_group, criterion_main};
use lance::dataset::builder::DatasetBuilder;

fn bench_combined_filter(c: &mut Criterion) {
    let Ok(uri) = env::var("LANCE_FILTER_BENCH_URI") else {
        return;
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let dataset = runtime.block_on(async {
        DatasetBuilder::from_uri(uri)
            .with_index_cache_size_bytes(1024 * 1024 * 1024)
            .load()
            .await
            .unwrap()
    });
    let cases = [
        ("score", "score >= 50"),
        ("category", "label = 'technology'"),
        ("tags", "array_has_any(tags, ['featured', 'important'])"),
        (
            "combined",
            "score >= 50 AND label = 'technology' AND array_has_any(tags, ['featured', 'important'])",
        ),
    ];
    let mut group = c.benchmark_group("combined_filter_dataset");
    for (name, predicate) in cases {
        let count = runtime
            .block_on(dataset.count_rows(Some(predicate.to_string())))
            .unwrap();
        if let ("combined", Ok(expected)) = (name, env::var("LANCE_FILTER_BENCH_EXPECT_COMBINED")) {
            assert_eq!(count, expected.parse::<usize>().unwrap());
        }
        group.bench_function(name, |b| {
            b.to_async(&runtime).iter(|| async {
                let result = dataset
                    .count_rows(Some(predicate.to_string()))
                    .await
                    .unwrap();
                assert_eq!(result, count);
                black_box(result)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_combined_filter);
criterion_main!(benches);
