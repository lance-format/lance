// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hint::black_box;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance_core::datatypes::Schema;

/// Benchmark `Schema::project` on a wide schema, projecting every column.
///
/// Projecting C columns out of F top-level fields used to cost O(C * F): each
/// requested column linearly scanned the top-level fields to resolve its name
/// and then scanned the candidates built so far to merge duplicates (#9710).
/// Both scans are now hash-based, so the cost is O(F + C).
fn project_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_project");
    for n in [1_000usize, 10_000] {
        let arrow_schema = ArrowSchema::new(
            (0..n)
                .map(|i| ArrowField::new(format!("c{i}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        );
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let names: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(schema.project(black_box(&names)).unwrap()));
        });
    }
    group.finish();
}

criterion_group!(benches, project_wide);
criterion_main!(benches);
