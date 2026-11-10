// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmarks of the stable-partition row map: file size (label encoding),
//! point translation, batch translation and sweep throughput.
//!
//! The setup writes a 2M-row map with 1000 destinations and ~1/8 deleted
//! rows, then prints the achieved bits per row: the label column is a
//! low-cardinality u16 column, so the Lance page dictionary path should land
//! well under the nominal 10 bits (ceil(log2 1000)) per row.

// The size probes report to the human running the bench; log targets nothing
// useful here.
#![allow(clippy::print_stdout)]

use std::hint::black_box;
use std::sync::{Arc, OnceLock};

use criterion::{Criterion, criterion_group, criterion_main};
use lance_core::cache::LanceCache;
use lance_core::utils::stable_partition::rank_label_prefix;
use lance_core::utils::tempfile::TempDir;
use lance_index::frag_reuse::row_map::{RowMapReader, RowMapWriter, SourceRows};
use lance_index::scalar::IndexStore;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_io::object_store::ObjectStore;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;

const TOTAL_ROWS: u64 = 2_000_000;
const NUM_DESTINATIONS: u32 = 1000;

struct Fixture {
    _tempdir: TempDir,
    reader: RowMapReader,
    runtime: tokio::runtime::Runtime,
}

static FIXTURE: OnceLock<Fixture> = OnceLock::new();

fn fixture() -> &'static Fixture {
    FIXTURE.get_or_init(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let tempdir = TempDir::default();
        // The store's IO scheduler spawns onto the ambient tokio runtime.
        let _guard = runtime.enter();
        let (object_store, path) = runtime
            .block_on(ObjectStore::from_uri(tempdir.obj_path().as_ref()))
            .unwrap();
        // 2.1+ has the miniblock dictionary path; 2.0 stores the labels as
        // plain u16 and roughly triples the file.
        let store: Arc<dyn IndexStore> = Arc::new(LanceIndexStore::with_format_version(
            object_store,
            path,
            Arc::new(LanceCache::with_capacity(128 * 1024 * 1024)),
            lance_file::version::ConcreteFileVersion::V2_1,
        ));

        let mut rng = StdRng::seed_from_u64(97);
        let mut deleted = RoaringBitmap::new();
        for offset in 0..TOTAL_ROWS {
            if rng.random_ratio(1, 8) {
                deleted.insert(offset as u32);
            }
        }
        let live_rows = TOTAL_ROWS - deleted.len();
        let sources = vec![SourceRows {
            physical_rows: TOTAL_ROWS,
            deleted: Some(deleted),
        }];

        let reader = runtime.block_on(async {
            let writer = store
                .new_index_file("row_map.lance", RowMapWriter::schema())
                .await
                .unwrap();
            let mut writer = RowMapWriter::try_new(writer, sources, NUM_DESTINATIONS).unwrap();
            let mut pending = Vec::with_capacity(64 * 1024);
            for _ in 0..live_rows {
                pending.push(rng.random_range(0..NUM_DESTINATIONS) as u16);
                if pending.len() == pending.capacity() {
                    writer.append_labels(&pending).await.unwrap();
                    pending.clear();
                }
            }
            writer.append_labels(&pending).await.unwrap();
            let (file, counts) = writer.finish().await.unwrap();
            let counts_bytes = counts.encode().len();
            println!(
                "row map: {TOTAL_ROWS} rows, {NUM_DESTINATIONS} destinations -> \
                 {} bytes file ({:.2} bits/row, nominal 10), {counts_bytes} bytes counts",
                file.size_bytes,
                (file.size_bytes.saturating_sub(counts_bytes as u64)) as f64 * 8.0
                    / TOTAL_ROWS as f64,
            );
            // Size probe with destination locality: each 64K block draws its
            // labels from a small window of destinations, the realistic shape
            // when source scan order correlates with the clustering key. The
            // uniform-random file above is the worst case (every block sees
            // all destinations, dictionary indices need the nominal width).
            let local_rows = 512u64 * 1024;
            let writer = store
                .new_index_file("row_map_local.lance", RowMapWriter::schema())
                .await
                .unwrap();
            let mut writer = RowMapWriter::try_new(
                writer,
                vec![SourceRows {
                    physical_rows: local_rows,
                    deleted: None,
                }],
                NUM_DESTINATIONS,
            )
            .unwrap();
            let mut pending = Vec::with_capacity(64 * 1024);
            for row in 0..local_rows {
                let window = (row / (64 * 1024)) * 16 % u64::from(NUM_DESTINATIONS);
                pending.push(
                    ((window + rng.random_range(0..16)) % u64::from(NUM_DESTINATIONS)) as u16,
                );
                if pending.len() == pending.capacity() {
                    writer.append_labels(&pending).await.unwrap();
                    pending.clear();
                }
            }
            writer.append_labels(&pending).await.unwrap();
            let (local_file, local_counts) = writer.finish().await.unwrap();
            let local_counts_bytes = local_counts.encode().len();
            println!(
                "row map (16-destination block locality): {local_rows} rows -> \
                 {} bytes file ({:.2} bits/row), {local_counts_bytes} bytes counts",
                local_file.size_bytes,
                (local_file
                    .size_bytes
                    .saturating_sub(local_counts_bytes as u64)) as f64
                    * 8.0
                    / local_rows as f64,
            );

            RowMapReader::open(store.open_index_file("row_map.lance").await.unwrap())
                .await
                .unwrap()
        });

        Fixture {
            _tempdir: tempdir,
            reader,
            runtime,
        }
    })
}

fn bench_point_translate(c: &mut Criterion) {
    let fixture = fixture();
    let mut rng = StdRng::seed_from_u64(3);
    c.bench_function("row_map/translate_point", |b| {
        b.iter(|| {
            let row = rng.random_range(0..TOTAL_ROWS);
            black_box(
                fixture
                    .runtime
                    .block_on(fixture.reader.translate(row))
                    .unwrap(),
            )
        })
    });
}

fn bench_translate_many(c: &mut Criterion) {
    let fixture = fixture();
    let mut rng = StdRng::seed_from_u64(5);
    let rows: Vec<u64> = (0..1024).map(|_| rng.random_range(0..TOTAL_ROWS)).collect();
    c.bench_function("row_map/translate_many_1k", |b| {
        b.iter(|| {
            black_box(
                fixture
                    .runtime
                    .block_on(fixture.reader.translate_many(&rows))
                    .unwrap(),
            )
        })
    });
}

fn bench_sweep(c: &mut Criterion) {
    let fixture = fixture();
    let mut group = c.benchmark_group("row_map");
    group.sample_size(10);
    group.throughput(criterion::Throughput::Elements(TOTAL_ROWS));
    group.bench_function("sweep_full", |b| {
        b.iter(|| {
            let mut live = 0u64;
            fixture
                .runtime
                .block_on(fixture.reader.sweep(0..TOTAL_ROWS, |_, translated| {
                    live += u64::from(translated.is_some());
                    Ok(())
                }))
                .unwrap();
            black_box(live)
        })
    });
    group.finish();
}

fn bench_rank_in_block(c: &mut Criterion) {
    // The in-memory cost of one worst-case point lookup: ranking a label over
    // a full 64K block.
    let mut rng = StdRng::seed_from_u64(11);
    let values: Vec<u16> = (0..64 * 1024)
        .map(|_| rng.random_range(0..NUM_DESTINATIONS) as u16)
        .collect();
    c.bench_function("row_map/rank_label_prefix_64k", |b| {
        b.iter(|| black_box(rank_label_prefix(&values, None, values.len(), 500)))
    });
}

criterion_group!(
    benches,
    bench_point_translate,
    bench_translate_many,
    bench_sweep,
    bench_rank_in_block
);
criterion_main!(benches);
