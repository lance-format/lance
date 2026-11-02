// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compare planning costs for ordinary predicates and separated memberships.
//! Run with `cargo bench -p lance --bench membership_planning --profile release-with-debug`.

use std::hint::black_box;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion::logical_expr::{col, lit};
use lance_datafusion::planner::Planner;

fn membership_planning(c: &mut Criterion) {
    let schema = Arc::new(Schema::new(
        (0..16)
            .map(|i| Field::new(format!("c{i}"), DataType::Int64, true))
            .collect::<Vec<_>>(),
    ));
    let planner = Planner::new(schema);
    let mut group = c.benchmark_group("membership_planning");
    for (name, expr) in [
        (
            "distinct_columns",
            (0..16_i64)
                .map(|i| col(format!("c{i}")).eq(lit(i)))
                .reduce(|a, b| a.and(b))
                .unwrap(),
        ),
        (
            "ordinary_ranges",
            (0..16_i64)
                .map(|i| col(format!("c{i}")).gt(lit(i)))
                .reduce(|a, b| a.and(b))
                .unwrap(),
        ),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || expr.clone(),
                |expr| black_box(planner.optimize_expr(expr).unwrap()),
                BatchSize::SmallInput,
            );
        });
    }
    for size in [50_i64, 1024, 100_000] {
        let wide = col("c0").in_list((0..size).map(lit).collect(), false);
        for (name, expr) in [
            (
                "single_membership",
                wide.clone().and(col("c1").eq(lit(1_i64))),
            ),
            (
                "redundant_membership",
                col("c0")
                    .in_list(vec![lit(1_i64), lit(2_i64)], false)
                    .and(col("c1").eq(lit(1_i64)))
                    .and(wide),
            ),
        ] {
            group.bench_with_input(BenchmarkId::new(name, size), &expr, |b, expr| {
                b.iter_batched(
                    || expr.clone(),
                    |expr| black_box(planner.optimize_expr(expr).unwrap()),
                    BatchSize::SmallInput,
                );
            });
        }
    }
    for count in [32, 128, 1024] {
        let mut level: Vec<_> = (0..count)
            .map(|i| (col("c0") + lit(i as i64)).in_list(vec![lit(1_i64)], false))
            .collect();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| pair[0].clone().and(pair[1].clone()))
                .collect();
        }
        let expr = level.pop().unwrap();
        group.bench_with_input(
            BenchmarkId::new("computed_memberships", count),
            &expr,
            |b, expr| {
                b.iter_batched(
                    || expr.clone(),
                    |expr| black_box(planner.optimize_expr(expr).unwrap()),
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, membership_planning);
criterion_main!(benches);
