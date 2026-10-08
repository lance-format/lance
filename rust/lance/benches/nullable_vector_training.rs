// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::vector::{has_vectors_to_train, utils::maybe_sample_training_data};
use lance_arrow::FixedSizeListArrayExt;
use rand::{Rng, SeedableRng, rngs::SmallRng};

fn bench_nullable_vector_training(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let vectors = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values((0..8192 * 2048).map(|_| rng.random::<f32>())),
        2048,
    )
    .unwrap();
    // This matches Coyo: the schema allows nulls, but every vector is present.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        vectors.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
    let dataset = rt
        .block_on(Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri,
            Some(WriteParams {
                max_rows_per_file: 1024,
                ..Default::default()
            }),
        ))
        .unwrap();

    let mut group = c.benchmark_group("nullable_vector_training");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    group.bench_function("preflight_256_of_8192", |b| {
        b.to_async(&rt).iter(|| async {
            assert!(has_vectors_to_train(&dataset, "vector", 256).await.unwrap());
        });
    });
    group.bench_function("sample_512_of_8192", |b| {
        b.to_async(&rt).iter(|| async {
            let sample = maybe_sample_training_data(&dataset, "vector", 512, None)
                .await
                .unwrap();
            assert_eq!(sample.len(), 512);
            assert_eq!(sample.null_count(), 0);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_nullable_vector_training);
criterion_main!(benches);
