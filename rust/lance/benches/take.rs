// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{
    BinaryArray, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    UInt32Array, cast::AsArray, types::Int32Type,
};
use arrow_schema::{DataType, Field, FieldRef, Schema as ArrowSchema};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use futures::StreamExt;
use lance::dataset::ProjectionRequest;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::LanceCache;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::LanceEncodingsIo;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::LanceFileVersion;
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};
use object_store::path::Path;
use rand::seq::{IteratorRandom, index};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::time::Duration;

const BATCH_SIZE: u64 = 1024;

fn random_ranges(num_rows: u64, file_size: u64, n: usize) -> Vec<u64> {
    let mut rng = rand::rng();
    let mut ranges = Vec::with_capacity(n);
    for i in 0..n {
        ranges.push(rng.random_range(1..num_rows));
        ranges[i] = ((ranges[i] / file_size) << 32) | (ranges[i] % file_size);
    }

    ranges
}

fn bench_random_take_with_dataset(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches = 1024;

    for file_size in [1024 * 1024, 1024] {
        dataset_take(
            c,
            &rt,
            file_size,
            num_batches,
            LanceFileVersion::V2_0,
            "V2_0",
        );
        dataset_take(
            c,
            &rt,
            file_size,
            num_batches,
            LanceFileVersion::V2_1,
            "V2_1",
        );
    }
}

fn dataset_take(
    c: &mut Criterion,
    rt: &tokio::runtime::Runtime,
    file_size: usize,
    num_batches: usize,
    version: LanceFileVersion,
    version_name: &str,
) {
    let dataset = rt.block_on(create_dataset(
        "memory://test.lance",
        version,
        num_batches as i32,
        file_size as i32,
    ));
    let schema = Arc::new(dataset.schema().clone());

    for num_rows in [1, 10, 100, 1000] {
        c.bench_function(
            &format!(
                "{version_name} Random Take Dataset({file_size} file size, {num_batches} batches, {num_rows} rows per take)",
            ), |b| {
                b.to_async(rt).iter(|| async {
                    let rows =
                        random_ranges(num_batches as u64 * BATCH_SIZE, file_size as u64, num_rows);
                    let batch = dataset
                        .take_rows(&rows, ProjectionRequest::Schema(schema.clone()))
                        .await
                        .unwrap_or_else(|_| panic!("rows: {:?}", rows));
                    assert_eq!(batch.num_rows(), num_rows);
                })
            },
        );
    }
}

fn bench_random_single_take_with_file_reader(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches: u64 = 1024;
    let file_size: u64 = num_batches * BATCH_SIZE + 1;
    let rows_gen = Box::new(|num_batches, file_size, num_rows| {
        let rows = random_ranges(num_batches * BATCH_SIZE, file_size, num_rows);
        let mut rows_list: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        for row in rows {
            rows_list.push(vec![row as u32]);
        }
        rows_list
    });

    file_reader_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_0,
        "V2_0 Single",
        rows_gen.clone(),
    );
    file_reader_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_1,
        "V2_1 Single",
        rows_gen,
    );
}

fn bench_random_batch_take_with_file_reader(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches: u64 = 1024;
    let file_size: u64 = num_batches * BATCH_SIZE + 1;
    let rows_gen = Box::new(|num_batches, file_size, num_rows| {
        let rows = random_ranges(num_batches * BATCH_SIZE, file_size, num_rows);
        let mut rows: Vec<u32> = rows.iter().map(|&x| x as u32).collect();
        rows.sort();
        vec![rows]
    });

    file_reader_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_0,
        "V2_0 Batch",
        rows_gen.clone(),
    );
    file_reader_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_1,
        "V2_1 Batch",
        rows_gen,
    );
}

fn file_reader_take(
    c: &mut Criterion,
    rt: &tokio::runtime::Runtime,
    file_size: u64,
    num_batches: u64,
    version: LanceFileVersion,
    version_name: &str,
    rows_gen: Box<dyn Fn(u64, u64, usize) -> Vec<Vec<u32>>>,
) {
    let (dataset, file_path) = rt.block_on(async {
        // Make sure there is only one fragment.
        let dataset = create_dataset(
            "memory://test.lance",
            version,
            num_batches as i32,
            file_size as i32,
        )
        .await;

        assert_eq!(dataset.get_fragments().len(), 1);
        let fragments = dataset.get_fragments();
        let fragment = fragments.first().unwrap();
        assert_eq!(fragment.num_data_files(), 1);
        let file = fragment.metadata().files.first().unwrap();
        let file_path = dataset.data_dir().join(file.path.as_str());

        (dataset, file_path)
    });

    // Bench random take.
    for num_rows in [1, 10, 100, 1000] {
        c.bench_function(&format!(
            "{version_name} Random Take FileReader({file_size} file size, {num_batches} batches, {num_rows} rows per take)"
        ), |b| {
            b.to_async(rt).iter(|| async {
                let file_reader = create_file_reader(&dataset, &file_path).await;

                let rows_list = rows_gen(num_batches, file_size, num_rows);
                for rows in rows_list {
                    let rows = ReadBatchParams::Indices(UInt32Array::from(rows));
                    let stream = file_reader
                        .read_stream(
                            rows,
                            1024,
                            16,
                            FilterExpression::no_filter(),
                        )
                        .await
                        .unwrap();
                    stream.fold(Vec::new(), |mut acc, item| async move {
                        acc.push(item);
                        acc
                    }).await;
                }
            })
        });
    }
}

async fn create_file_reader(dataset: &Dataset, file_path: &Path) -> FileReader {
    // Create file reader v2.
    let object_store = dataset.object_store(None).await.unwrap();
    let scheduler = ScanScheduler::new(object_store, SchedulerConfig::new(2 * 1024 * 1024 * 1024));
    let file = scheduler
        .open_file(file_path, &CachedFileSize::unknown())
        .await
        .unwrap();
    let file_metadata = FileReader::read_all_metadata(&file).await.unwrap();

    FileReader::try_open_with_file_metadata(
        Arc::new(LanceEncodingsIo::new(file.clone())),
        file_path.clone(),
        None,
        Arc::<DecoderPlugins>::default(),
        Arc::new(file_metadata),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

fn bench_random_single_take_with_file_fragment(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches: u64 = 1024;
    // Make sure there is only one fragment.
    let file_size: u64 = num_batches * BATCH_SIZE + 1;
    let rows_gen = Box::new(|num_batches, file_size, num_rows| {
        let rows = random_ranges(num_batches * BATCH_SIZE, file_size, num_rows);
        let mut rows_list: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        for row in rows {
            rows_list.push(vec![row as u32]);
        }
        rows_list
    });

    fragment_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_0,
        "V2_0 Single",
        rows_gen.clone(),
    );
    fragment_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_1,
        "V2_1 Single",
        rows_gen,
    );
}

fn bench_random_batch_take_with_file_fragment(c: &mut Criterion) {
    // default tokio runtime
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_batches: u64 = 1024;
    // Make sure there is only one fragment.
    let file_size: u64 = num_batches * BATCH_SIZE + 1;
    let rows_gen = Box::new(|num_batches, file_size, num_rows| {
        let rows = random_ranges(num_batches * BATCH_SIZE, file_size, num_rows);
        let mut rows: Vec<u32> = rows.iter().map(|&x| x as u32).collect();
        rows.sort();
        vec![rows]
    });

    fragment_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_0,
        "V2_0 Batch",
        rows_gen.clone(),
    );
    fragment_take(
        c,
        &rt,
        file_size,
        num_batches,
        LanceFileVersion::V2_1,
        "V2_1 Batch",
        rows_gen,
    );
}

fn fragment_take(
    c: &mut Criterion,
    rt: &tokio::runtime::Runtime,
    file_size: u64,
    num_batches: u64,
    version: LanceFileVersion,
    version_name: &str,
    rows_gen: Box<dyn Fn(u64, u64, usize) -> Vec<Vec<u32>>>,
) {
    let dataset = rt.block_on(create_dataset(
        "memory://test.lance",
        version,
        num_batches as i32,
        file_size as i32,
    ));

    assert_eq!(dataset.get_fragments().len(), 1);
    let fragments = dataset.get_fragments();
    let fragment = fragments.first().unwrap();

    // Bench random take.
    for num_rows in [1, 10, 100, 1000] {
        c.bench_function(&format!(
            "{version_name} Random Take Fragment({file_size} file size, {num_batches} batches, {num_rows} rows per take)"
        ), |b| {
            b.to_async(rt).iter(|| async {
                let rows_list = rows_gen(num_batches, file_size, num_rows);
                for rows in rows_list {
                    let _ = fragment.take(rows.as_slice(), dataset.schema()).await;
                }
            })
        });
    }
}

/// Compare selection kernels, including conversion, sorting, and consumption.
fn bench_sample_offsets(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_offsets");
    let sparse = [1_000_000, 10_000_000, 100_000_000]
        .into_iter()
        .flat_map(|population| [0, 10, 1024, 8192].map(|size| (population, size)));
    let dense = [50_000, 100_000, 100_001].map(|size| (100_000, size));
    for (population, requested) in sparse.chain(dense) {
        for direct in [false, true] {
            let mut rng = StdRng::seed_from_u64(42);
            let kernel = if direct { "direct" } else { "reservoir" };
            group.bench_function(
                BenchmarkId::new(kernel, format!("{requested}_of_{population}")),
                |b| {
                    b.iter(|| {
                        let offsets: Vec<u64> = if direct {
                            let size = requested.min(population);
                            if size == 0 {
                                Vec::new()
                            } else if size == population {
                                (0..population as u64).collect()
                            } else {
                                let mut offsets: Vec<u64> =
                                    index::sample(&mut rng, population, size)
                                        .into_iter()
                                        .map(|offset| offset as u64)
                                        .collect();
                                offsets.sort_unstable();
                                offsets
                            }
                        } else {
                            let mut offsets =
                                (0..population as u64).choose_multiple(&mut rng, requested);
                            offsets.sort_unstable();
                            offsets
                        };
                        black_box(offsets)
                    })
                },
            );
        }
    }
    group.finish();
}

/// Benchmarks Dataset::sample(), which is used during IVF training.
fn bench_sample(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // 100 batches * 1024 rows = 102,400 rows total, spread across multiple fragments
    let num_batches = 100;
    let file_size = 10 * BATCH_SIZE as usize; // 10,240 rows per fragment → 10 fragments
    let dataset = rt.block_on(create_dataset(
        "memory://sample_bench.lance",
        LanceFileVersion::V2_1,
        num_batches,
        file_size as i32,
    ));
    let total_rows = num_batches as u64 * BATCH_SIZE;
    let schema = dataset.schema().clone();

    for sample_size in [1024, 8192] {
        c.bench_function(
            &format!("sample({sample_size} of {total_rows} rows)"),
            |b| {
                b.to_async(&rt).iter(|| {
                    let schema = schema.clone();
                    let dataset = dataset.clone();
                    async move {
                        dataset.sample(sample_size, &schema, None).await.unwrap();
                    }
                })
            },
        );
    }
}

fn bench_sample_large(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("sample_dataset");
    for num_batches in [1024, 10_240] {
        let mut dataset = rt.block_on(create_dataset(
            "memory://",
            LanceFileVersion::Stable,
            num_batches,
            102_400,
        ));
        let projection = dataset.schema().project(&["i"]).unwrap();
        let fragment_ids: Vec<u32> = dataset
            .get_fragments()
            .iter()
            .step_by(2)
            .map(|fragment| fragment.id() as u32)
            .collect();
        for deleted in [false, true] {
            if deleted {
                rt.block_on(dataset.delete("i % 10 = 0")).unwrap();
            }
            for selected in [false, true] {
                let fragments = selected.then_some(fragment_ids.as_slice());
                let live_rows = rt.block_on(async {
                    if selected {
                        let mut rows = 0;
                        for fragment in dataset.get_fragments().iter().step_by(2) {
                            rows += fragment.count_rows(None).await.unwrap();
                        }
                        rows
                    } else {
                        dataset.count_rows(None).await.unwrap()
                    }
                });
                for size in [10, 1024] {
                    group.bench_function(
                        format!(
                            "{size}_of_{}_live_{live_rows}_selected_{selected}_deleted_{deleted}",
                            num_batches as u64 * BATCH_SIZE
                        ),
                        |b| {
                            b.to_async(&rt).iter(|| async {
                                black_box(
                                    dataset.sample(size, &projection, fragments).await.unwrap(),
                                )
                            })
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

/// Sparse logical takes with a deletion vector in every fragment. Dataset opening
/// is outside the timed region so cold and warm results measure the same operation.
/// "cold" uses a fresh Lance session; the filesystem cache remains warm.
fn bench_sparse_take_with_deletions(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("sparse_take_with_deletions");
    group.sample_size(20);
    for num_fragments in [8usize, 128, 1024] {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        rt.block_on(async {
            let data = arrow_array::record_batch!((
                "i",
                Int32,
                (0..(num_fragments * 4) as i32).collect::<Vec<_>>()
            ))
            .unwrap();
            let mut dataset = Dataset::write(
                RecordBatchIterator::new([Ok(data.clone())], data.schema()),
                uri,
                Some(WriteParams {
                    max_rows_per_file: 4,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            dataset.delete("i % 4 = 0").await.unwrap();
            assert_eq!(dataset.get_fragments().len(), num_fragments);
        });
        let offset = (num_fragments * 3 - 1) as u64;
        for cache in ["cold", "warm"] {
            let open = || rt.block_on(DatasetBuilder::from_uri(uri).load()).unwrap();
            let take = |dataset: &Dataset| {
                rt.block_on(async {
                    dataset
                        .take(
                            &[offset],
                            ProjectionRequest::from_columns(["i"], dataset.schema()),
                        )
                        .await
                        .unwrap()
                })
            };
            let dataset = open();
            let object_store = rt.block_on(dataset.object_store(None)).unwrap();
            if cache == "warm" {
                take(&dataset);
            }
            object_store.io_stats_incremental();
            let batch = take(&dataset);
            assert_eq!(batch.num_rows(), 1);
            assert_eq!(
                batch["i"].as_primitive::<Int32Type>().value(0),
                (num_fragments * 4 - 1) as i32,
                "{cache} take with {num_fragments} fragments returned the wrong value"
            );
            let stats = object_store.io_stats_incremental();
            let deletion_reads: Vec<_> = stats
                .requests
                .iter()
                .filter(|r| {
                    r.path.parts().any(|part| part.as_ref() == "_deletions") && r.method != "head"
                })
                .collect();
            let deletion_bytes: u64 = deletion_reads
                .iter()
                .map(|r| {
                    r.range
                        .as_ref()
                        .map(|range| range.end - range.start)
                        .unwrap_or_else(|| {
                            let name = r.path.filename().unwrap();
                            std::fs::metadata(dir.path().join("_deletions").join(name))
                                .unwrap()
                                .len()
                        })
                })
                .sum();
            let expected_deletion_reads = usize::from(cache == "cold");
            assert_eq!(
                deletion_reads.len(),
                expected_deletion_reads,
                "{cache} take with {num_fragments} fragments: {} deletion reads, {deletion_bytes} deletion bytes",
                deletion_reads.len()
            );
            if cache == "cold" {
                assert!(
                    deletion_bytes > 0,
                    "{cache} take with {num_fragments} fragments: {} deletion reads, {deletion_bytes} deletion bytes",
                    deletion_reads.len()
                );
            } else {
                assert_eq!(
                    deletion_bytes,
                    0,
                    "{cache} take with {num_fragments} fragments: {} deletion reads, {deletion_bytes} deletion bytes",
                    deletion_reads.len()
                );
            }
            let id = BenchmarkId::new(cache, num_fragments);
            group.bench_function(id, |b| {
                if cache == "cold" {
                    b.iter_batched_ref(
                        open,
                        |dataset| {
                            take(dataset);
                        },
                        BatchSize::PerIteration,
                    );
                } else {
                    b.iter_batched_ref(
                        || {
                            object_store.io_stats_incremental();
                            dataset.clone()
                        },
                        |dataset| {
                            take(dataset);
                        },
                        BatchSize::PerIteration,
                    );
                }
            });
        }
    }
    group.finish();
}

async fn create_dataset(
    path: &str,
    data_storage_version: LanceFileVersion,
    num_batches: i32,
    file_size: i32,
) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("f", DataType::Float32, false),
        Field::new("s", DataType::Binary, false),
        Field::new(
            "fsl",
            DataType::FixedSizeList(
                FieldRef::new(Field::new("item", DataType::Float32, true)),
                2,
            ),
            false,
        ),
        Field::new("blob", DataType::Binary, false),
    ]));
    let batch_size = BATCH_SIZE as i32;
    let batch_schema = schema.clone();
    let batches = (0..num_batches).map(move |i| {
        RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    i * batch_size..(i + 1) * batch_size,
                )),
                Arc::new(Float32Array::from_iter_values(
                    (i * batch_size..(i + 1) * batch_size)
                        .map(|x| x as f32)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(BinaryArray::from_iter_values(
                    (i * batch_size..(i + 1) * batch_size)
                        .map(|x| format!("blob-{}", x).into_bytes()),
                )),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from_iter_values(
                            (i * batch_size..(i + 2) * batch_size)
                                .map(|x| (batch_size + (x - batch_size) / 2) as f32),
                        ),
                        2,
                    )
                    .unwrap(),
                ),
                Arc::new(BinaryArray::from_iter_values(
                    (i * batch_size..(i + 1) * batch_size)
                        .map(|x| format!("blob-{}", x).into_bytes()),
                )),
            ],
        )
        .unwrap()
    });

    let write_params = WriteParams {
        max_rows_per_file: file_size as usize,
        max_rows_per_group: batch_size as usize,
        mode: WriteMode::Create,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(batches.map(Ok), schema.clone());
    Dataset::write(reader, path, Some(write_params))
        .await
        .unwrap()
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(10000)
        .warm_up_time(Duration::from_secs_f32(3.0))
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_random_take_with_dataset, bench_random_single_take_with_file_fragment, bench_random_single_take_with_file_reader, bench_random_batch_take_with_file_fragment, bench_random_batch_take_with_file_reader, bench_sample, bench_sample_offsets, bench_sample_large, bench_sparse_take_with_deletions);
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_random_take_with_dataset, bench_random_single_take_with_file_fragment, bench_random_single_take_with_file_reader, bench_random_batch_take_with_file_fragment, bench_random_batch_take_with_file_reader, bench_sample, bench_sample_offsets, bench_sample_large, bench_sparse_take_with_deletions);
criterion_main!(benches);
