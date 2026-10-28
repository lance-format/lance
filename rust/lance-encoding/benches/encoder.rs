// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, hint::black_box, sync::Arc};

use arrow_array::{ArrayRef, BooleanArray, ListArray, RecordBatch, StringArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lance_encoding::encoder::{EncodingOptions, encode_batch};

pub mod common;
use common::{BenchEncoding, encoding_strategy};

fn encode_batch_sync(rt: &tokio::runtime::Runtime, data: &RecordBatch) {
    let lance_schema =
        Arc::new(lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap());
    let encoding_strategy = encoding_strategy(BenchEncoding::StructuralU32);

    rt.block_on(encode_batch(
        data,
        lance_schema,
        encoding_strategy.as_ref(),
        &EncodingOptions::default(),
    ))
    .unwrap();
}

fn bench_encode_compressed(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("encode_compressed");

    const NUM_ROWS: usize = 5_000_000;
    const NUM_COLUMNS: usize = 10;

    // Generate compressible string data - high cardinality but compressible
    // (unique values to avoid dictionary encoding, repeated prefix for compression)
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..NUM_ROWS).map(|i| format!("prefix_that_compresses_well_{}", i)),
    ));

    for compression in ["zstd", "lz4"] {
        let mut metadata = HashMap::new();
        metadata.insert(
            "lance-encoding:compression".to_string(),
            compression.to_string(),
        );
        // Disable dictionary encoding to ensure we hit the compression path
        metadata.insert(
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        );
        // Force miniblock encoding (the path that benefits from compressor caching)
        metadata.insert(
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        );
        let fields: Vec<Field> = (0..NUM_COLUMNS)
            .map(|i| {
                Field::new(format!("s{}", i), DataType::Utf8, false).with_metadata(metadata.clone())
            })
            .collect();
        let columns: Vec<Arc<dyn arrow_array::Array>> =
            (0..NUM_COLUMNS).map(|_| array.clone()).collect();
        let schema = Arc::new(Schema::new(fields));
        let data = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
        // V2_2+ required for general compression
        let encoding_strategy = encoding_strategy(BenchEncoding::StructuralU32);

        group.throughput(criterion::Throughput::Elements(
            (NUM_ROWS * NUM_COLUMNS) as u64,
        ));
        group.bench_function(
            format!("{}_strings_{}cols", compression, NUM_COLUMNS),
            |b| {
                b.iter(|| {
                    rt.block_on(encode_batch(
                        &data,
                        lance_schema.clone(),
                        encoding_strategy.as_ref(),
                        &EncodingOptions::default(),
                    ))
                    .unwrap()
                })
            },
        );
    }
}

fn make_boolean_list_batch(array: ArrayRef, nullable: bool) -> RecordBatch {
    let item_field = Arc::new(Field::new("item", DataType::Boolean, true));
    let field = Field::new("values", DataType::List(item_field), nullable);
    let schema = Arc::new(Schema::new(vec![field]));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

fn dense_boolean_list(num_rows: usize, booleans_per_list: usize) -> ArrayRef {
    let mut offsets = Vec::with_capacity(num_rows + 1);
    let mut values = Vec::with_capacity(num_rows * booleans_per_list);
    offsets.push(0i32);

    for _ in 0..num_rows {
        values.extend((0..booleans_per_list).map(|idx| idx % 2 == 0));
        offsets.push(values.len() as i32);
    }

    Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Boolean, true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(BooleanArray::from(values)),
        None,
    ))
}

fn sparse_boolean_list(
    num_rows: usize,
    num_non_empty: usize,
    booleans_per_list: usize,
) -> ArrayRef {
    let step = num_rows / num_non_empty;
    let mut offsets = Vec::with_capacity(num_rows + 1);
    let mut values = Vec::with_capacity(num_non_empty * booleans_per_list);
    offsets.push(0i32);

    let mut next_non_empty = step / 2;
    for row in 0..num_rows {
        if row == next_non_empty {
            values.extend((0..booleans_per_list).map(|idx| idx % 2 == 0));
            next_non_empty += step;
        }
        offsets.push(values.len() as i32);
    }

    Arc::new(ListArray::new(
        Arc::new(Field::new("item", DataType::Boolean, true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(BooleanArray::from(values)),
        None,
    ))
}

fn bench_encode_structural_pages(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("encode_structural_pages");

    const NUM_ROWS: usize = 200_000;
    const BOOLEANS_PER_LIST: usize = 8;

    let dense = make_boolean_list_batch(dense_boolean_list(NUM_ROWS, BOOLEANS_PER_LIST), false);
    let sparse =
        make_boolean_list_batch(sparse_boolean_list(NUM_ROWS, 10, BOOLEANS_PER_LIST), false);

    group.throughput(criterion::Throughput::Elements(NUM_ROWS as u64));
    group.bench_function("dense_boolean_list", |b| {
        b.iter(|| encode_batch_sync(&rt, &dense))
    });
    group.bench_function("sparse_boolean_list", |b| {
        b.iter(|| encode_batch_sync(&rt, &sparse))
    });
}

fn bench_variable_offsets(c: &mut Criterion) {
    const NUM_ROWS: usize = 262_144;
    let metadata = HashMap::from([
        (
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        ),
        (
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        ),
        ("lance-encoding:compression".to_string(), "none".to_string()),
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Utf8, false).with_metadata(metadata),
    ]));
    let lance_schema = Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("variable_offsets");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));
    let corpora: [(&str, ArrayRef); 2] = [
        (
            "range",
            Arc::new(StringArray::from_iter_values(
                (0..NUM_ROWS).map(|index| format!("row_{index:012}")),
            )),
        ),
        (
            "delta",
            Arc::new(StringArray::from_iter_values(
                (0..NUM_ROWS).map(|index| "x".repeat(4 + index % 64)),
            )),
        ),
    ];
    for (workload, array) in corpora {
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        for (version, encoding) in [
            ("v2.2", BenchEncoding::StructuralU32),
            ("v2.3", BenchEncoding::StructuralSparse),
        ] {
            let strategy = encoding_strategy(encoding);
            let options = EncodingOptions::default();
            let encoded_bytes = runtime
                .block_on(encode_batch(
                    &batch,
                    lance_schema.clone(),
                    strategy.as_ref(),
                    &options,
                ))
                .unwrap()
                .data
                .len();
            group.bench_function(
                format!("{workload}/{version}/{encoded_bytes}B"),
                |bencher| {
                    bencher.iter(|| {
                        black_box(
                            runtime
                                .block_on(encode_batch(
                                    &batch,
                                    lance_schema.clone(),
                                    strategy.as_ref(),
                                    &options,
                                ))
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(lance_testing::pprof::PProfProfiler::new(100, lance_testing::pprof::Output::Flamegraph(None)));
    targets = bench_encode_compressed, bench_encode_structural_pages, bench_variable_offsets);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_encode_compressed, bench_encode_structural_pages, bench_variable_offsets);

criterion_main!(benches);
