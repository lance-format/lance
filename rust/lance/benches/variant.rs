// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compare equivalent field extraction from JSONB, unshredded Variant, and
//! shredded Variant. The typed-child case is valid only for the zero-fallback
//! fixture; it illustrates the opportunity for future residual-aware pushdown.
//!
//! Run with:
//! `cargo bench -p lance --bench variant --features variant --profile release-with-debug`

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::{InsertBuilder, WriteParams};
use parquet_variant_compute::json_to_variant;

const ROWS: usize = 10_000;

async fn fixture(fallback_percent: usize) -> Vec<(&'static str, Dataset, &'static str)> {
    let documents: Vec<String> = (0..ROWS)
        .map(|row| {
            let flag = if row % 100 < fallback_percent {
                "7"
            } else {
                "true"
            };
            let mut fields = vec![format!("\"id\":{row}"), format!("\"flag\":{flag}")];
            fields.extend((0..30).map(|field| {
                format!("\"field_{field}\":\"value-{row}-{field}-with-extra-payload\"")
            }));
            format!("{{{}}}", fields.join(","))
        })
        .collect();
    let strings: ArrayRef = Arc::new(StringArray::from(documents));
    let json_field = Field::new("event", DataType::Utf8, false)
        .with_metadata([("ARROW:extension:name".to_owned(), "arrow.json".to_owned())].into());
    let json_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![json_field])),
        vec![strings.clone()],
    )
    .unwrap();
    let variant = json_to_variant(&strings).unwrap();
    let variant_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![variant.field("event")])),
        vec![Arc::new(variant.into_inner())],
    )
    .unwrap();
    let params = WriteParams {
        max_rows_per_file: 2500,
        ..Default::default()
    };
    let json = InsertBuilder::new("memory://")
        .with_params(&params)
        .execute(vec![json_batch])
        .await
        .unwrap();
    let plain = InsertBuilder::new("memory://")
        .with_params(&params)
        .execute(vec![variant_batch.clone()])
        .await
        .unwrap();
    let target = DataType::Struct(
        vec![
            Field::new("id", DataType::Int64, true),
            Field::new("flag", DataType::Boolean, true),
        ]
        .into(),
    );
    let shredded = InsertBuilder::new("memory://")
        .with_params(&params)
        .with_variant_shredding("event", target)
        .execute(vec![variant_batch])
        .await
        .unwrap();
    let typed = shredded
        .scan()
        .project_with_transform(&[("id", "event.typed_value.id.typed_value")])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(typed["id"].null_count(), ROWS * fallback_percent / 100);
    let mut cases = vec![
        ("jsonb", json, "json_get_int(event, 'id')"),
        ("variant", plain, "variant_get_int(event, 'id')"),
        (
            "shredded_variant",
            shredded.clone(),
            "variant_get_int(event, 'id')",
        ),
    ];
    if fallback_percent == 0 {
        cases.push((
            "typed_child_no_fallback_only",
            shredded,
            "event.typed_value.id.typed_value",
        ));
    }
    let expected = Int64Array::from_iter_values(0..ROWS as i64);
    for (_, dataset, expression) in &cases {
        let result = dataset
            .scan()
            .project_with_transform(&[("id", *expression)])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result["id"].as_ref(), &expected);
    }
    cases
}

fn bench_variant(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    for fallback_percent in [0, 10] {
        let cases = rt.block_on(fixture(fallback_percent));
        let mut group = c.benchmark_group(format!(
            "variant_projection/fallback_{fallback_percent}_percent"
        ));
        group.throughput(Throughput::Elements(ROWS as u64));
        for (name, dataset, expression) in cases {
            group.bench_with_input(BenchmarkId::from_parameter(name), &dataset, |b, dataset| {
                b.to_async(&rt).iter(|| async {
                    let result = dataset
                        .scan()
                        .project_with_transform(&[("id", expression)])
                        .unwrap()
                        .try_into_batch()
                        .await
                        .unwrap();
                    std::hint::black_box(result);
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_variant);
criterion_main!(benches);
