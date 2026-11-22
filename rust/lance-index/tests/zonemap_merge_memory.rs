// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Checks that ZoneMap merge admission bounds the memory actually allocated:
//! a loaded segment holds no more than its `loaded_size_bytes`, and merging
//! loaded segments allocates no more than `estimate_zonemap_merge_bytes` on
//! top of them. The transient memory of loading is reported, not bounded.
//!
//! This test binary installs a counting global allocator, so it holds a
//! single test that runs its cases one after another.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{
    ArrayRef, BinaryArray, BooleanArray, Float64Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::ROW_ADDR;
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempObjDir;
use lance_index::scalar::IndexStore;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::ScalarIndexPlugin;
use lance_index::scalar::zonemap::{
    ZoneMapIndex, ZoneMapIndexBuilder, ZoneMapIndexBuilderParams, ZoneMapIndexPlugin,
    estimate_zonemap_merge_bytes, merge_zonemap_indices_by_coverage,
};
use lance_io::object_store::ObjectStore;
use roaring::RoaringBitmap;

struct CountingAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

// SAFETY: every call is forwarded to the system allocator unchanged; only
// the byte counters are updated around it.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded with the caller's layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded with the caller's layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded with the caller's pointer and layout.
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded with the caller's pointer, layout and size.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            // Both blocks are live while the contents are copied.
            grew(new_size);
            CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// One source: its fragments, each a column of values.
struct Source {
    fragments: Vec<(u32, ArrayRef)>,
}

fn store(dir: &TempObjDir) -> Arc<LanceIndexStore> {
    Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        (**dir).clone(),
        Arc::new(LanceCache::no_cache()),
    ))
}

async fn write_source(source: &Source, rows_per_zone: u64, store: &dyn IndexStore) {
    let data_type = source.fragments[0].1.data_type().clone();
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", data_type.clone(), true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let batches = source
        .fragments
        .iter()
        .map(|(fragment_id, values)| {
            let row_addrs = UInt64Array::from_iter_values(
                (0..values.len() as u64).map(|offset| (u64::from(*fragment_id) << 32) | offset),
            );
            Ok(
                RecordBatch::try_new(schema.clone(), vec![values.clone(), Arc::new(row_addrs)])
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let stream = Box::pin(RecordBatchStreamAdapter::new(schema, stream::iter(batches)));
    let mut builder =
        ZoneMapIndexBuilder::try_new(ZoneMapIndexBuilderParams::new(rows_per_zone), data_type)
            .unwrap();
    builder.train(stream).await.unwrap();
    builder.write_index(store).await.unwrap();
}

/// What one case measured, in bytes.
struct Measured {
    /// Held by the loaded sources once loading finished.
    retained: u64,
    /// Sum of the sources' `loaded_size_bytes`.
    loaded_size: u64,
    /// Peak while loading, beyond the starting point (not bounded).
    load_peak: u64,
    /// Peak while merging, beyond what the loaded sources hold.
    merge_peak: u64,
    /// `estimate_zonemap_merge_bytes` of the loaded sources.
    merge_estimate: u64,
}

/// Loads the sources from `stores` and merges them, measuring both steps.
async fn measure(stores: &[Arc<LanceIndexStore>], coverages: &[RoaringBitmap]) -> Measured {
    let dest_dir = TempObjDir::default();
    let dest = store(&dest_dir);
    let cache = LanceCache::no_cache();
    let details = prost_types::Any::default();

    let baseline = CURRENT.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    let mut loaded = Vec::new();
    for store in stores {
        let index = ZoneMapIndexPlugin
            .load_index(store.clone(), &details, 1, None, &cache)
            .await
            .unwrap();
        loaded.push(index);
    }
    let after_load = CURRENT.load(Ordering::SeqCst);
    let load_peak = PEAK.load(Ordering::SeqCst).saturating_sub(baseline) as u64;
    let retained = after_load.saturating_sub(baseline) as u64;

    let sources = loaded
        .iter()
        .zip(coverages)
        .map(|(index, coverage)| {
            (
                index.as_any().downcast_ref::<ZoneMapIndex>().unwrap(),
                coverage,
            )
        })
        .collect::<Vec<_>>();
    let indices = sources.iter().map(|(index, _)| *index).collect::<Vec<_>>();
    let loaded_size = indices.iter().map(|index| index.loaded_size_bytes()).sum();
    let merge_estimate = estimate_zonemap_merge_bytes(&indices);
    drop(indices);

    let before_merge = CURRENT.load(Ordering::SeqCst);
    PEAK.store(before_merge, Ordering::SeqCst);
    merge_zonemap_indices_by_coverage(&sources, dest.as_ref())
        .await
        .unwrap();
    let merge_peak = PEAK.load(Ordering::SeqCst).saturating_sub(before_merge) as u64;
    Measured {
        retained,
        loaded_size,
        load_peak,
        merge_peak,
        merge_estimate,
    }
}

fn int64_fragment(rows: usize, null_every: Option<usize>) -> ArrayRef {
    Arc::new(Int64Array::from_iter((0..rows).map(
        |row| match null_every {
            Some(every) if row % every == 0 => None,
            _ => Some(row as i64 * 7 % 1_000_003),
        },
    )))
}

struct Case {
    name: &'static str,
    rows_per_zone: u64,
    sources: Vec<Source>,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "int64 without nulls",
            rows_per_zone: 64,
            sources: vec![
                Source {
                    fragments: (0..10).map(|f| (f, int64_fragment(20_000, None))).collect(),
                },
                Source {
                    fragments: (10..20)
                        .map(|f| (f, int64_fragment(20_000, None)))
                        .collect(),
                },
            ],
        },
        Case {
            name: "int64 with sparse nulls",
            rows_per_zone: 256,
            sources: vec![
                Source {
                    fragments: (0..8)
                        .map(|f| (f, int64_fragment(50_000, Some(997))))
                        .collect(),
                },
                Source {
                    fragments: (8..16)
                        .map(|f| (f, int64_fragment(50_000, Some(13))))
                        .collect(),
                },
            ],
        },
        Case {
            // One null per container: the null positions serialize small.
            name: "one null per roaring container",
            rows_per_zone: 8192,
            sources: vec![Source {
                fragments: vec![(0, int64_fragment(1_000_000, Some(65_536)))],
            }],
        },
        Case {
            // Many fragments with one null each: tree entries dominate.
            name: "many fragments with one null each",
            rows_per_zone: 4,
            sources: vec![
                Source {
                    fragments: (0..1500).map(|f| (f, int64_fragment(3, Some(3)))).collect(),
                },
                Source {
                    fragments: (1500..3000)
                        .map(|f| (f, int64_fragment(3, Some(3))))
                        .collect(),
                },
            ],
        },
        Case {
            name: "all nulls",
            rows_per_zone: 1024,
            sources: vec![Source {
                fragments: (0..4)
                    .map(|f| (f, int64_fragment(100_000, Some(1))))
                    .collect(),
            }],
        },
        Case {
            name: "int32",
            rows_per_zone: 16,
            sources: vec![Source {
                fragments: vec![(
                    0,
                    Arc::new(Int32Array::from_iter_values(0..100_000)) as ArrayRef,
                )],
            }],
        },
        Case {
            name: "float64 with NaN and nulls",
            rows_per_zone: 16,
            sources: vec![Source {
                fragments: vec![(
                    0,
                    Arc::new(Float64Array::from_iter((0..100_000).map(|row| {
                        match row % 5 {
                            0 => None,
                            1 => Some(f64::NAN),
                            _ => Some(row as f64),
                        }
                    }))) as ArrayRef,
                )],
            }],
        },
        Case {
            name: "boolean",
            rows_per_zone: 16,
            sources: vec![Source {
                fragments: vec![(
                    0,
                    Arc::new(BooleanArray::from_iter(
                        (0..100_000).map(|row| (row % 7 != 0).then_some(row % 2 == 0)),
                    )) as ArrayRef,
                )],
            }],
        },
        Case {
            name: "utf8 short (8 bytes)",
            rows_per_zone: 16,
            sources: (0..2)
                .map(|source| Source {
                    fragments: (source * 4..source * 4 + 4)
                        .map(|f| (f, utf8_fragment(f, 25_000, 8, None)))
                        .collect(),
                })
                .collect(),
        },
        Case {
            name: "utf8 long (1 KiB)",
            rows_per_zone: 16,
            sources: (0..2)
                .map(|source| Source {
                    fragments: (source * 2..source * 2 + 2)
                        .map(|f| (f, utf8_fragment(f, 5_000, 1024, None)))
                        .collect(),
                })
                .collect(),
        },
        Case {
            name: "utf8 with nulls and all-null zones",
            rows_per_zone: 16,
            sources: (0..2)
                .map(|source| Source {
                    fragments: (source * 4..source * 4 + 4)
                        .map(|f| (f, utf8_fragment(f, 25_000, 24, Some(5))))
                        .collect(),
                })
                .collect(),
        },
        Case {
            name: "large utf8 (64 bytes)",
            rows_per_zone: 16,
            sources: vec![Source {
                fragments: (0..4)
                    .map(|f| {
                        let values = utf8_values(f, 25_000, 64, None);
                        (f, Arc::new(LargeStringArray::from(values)) as ArrayRef)
                    })
                    .collect(),
            }],
        },
        Case {
            name: "binary (256 bytes)",
            rows_per_zone: 16,
            sources: vec![Source {
                fragments: (0..4)
                    .map(|f| {
                        let values = utf8_values(f, 10_000, 256, None);
                        let values = values
                            .iter()
                            .map(|value| value.as_deref().map(str::as_bytes))
                            .collect::<Vec<_>>();
                        (f, Arc::new(BinaryArray::from(values)) as ArrayRef)
                    })
                    .collect(),
            }],
        },
    ]
}

/// `rows` strings of `width` bytes for fragment `fragment`; with
/// `null_every`, every such row is null and so is every row of each 40th
/// run of 32 rows, which leaves some zones of 16 rows all null.
fn utf8_values(
    fragment: u32,
    rows: usize,
    width: usize,
    null_every: Option<usize>,
) -> Vec<Option<String>> {
    (0..rows)
        .map(|row| {
            let null = null_every.is_some_and(|every| row % every == 0 || (row / 32) % 40 == 7);
            (!null).then(|| {
                let key = format!("{fragment:04}-{:07}", row * 7919 % 1_000_003);
                let mut value = key.repeat(width.div_ceil(key.len()));
                value.truncate(width);
                value
            })
        })
        .collect()
}

fn utf8_fragment(fragment: u32, rows: usize, width: usize, null_every: Option<usize>) -> ArrayRef {
    Arc::new(StringArray::from(utf8_values(
        fragment, rows, width, null_every,
    )))
}

#[tokio::test]
async fn test_merge_admission_bounds_measured_memory() {
    let mut report = Vec::new();
    for case in cases() {
        let dirs = case
            .sources
            .iter()
            .map(|_| TempObjDir::default())
            .collect::<Vec<_>>();
        let stores = dirs.iter().map(store).collect::<Vec<_>>();
        let mut coverages = Vec::new();
        for (source, store) in case.sources.iter().zip(&stores) {
            write_source(source, case.rows_per_zone, store.as_ref()).await;
            coverages.push(RoaringBitmap::from_iter(
                source.fragments.iter().map(|(fragment_id, _)| *fragment_id),
            ));
        }
        let m = measure(&stores, &coverages).await;
        let admitted = m.loaded_size + m.merge_estimate;
        let used = m.retained + m.merge_peak;
        report.push(format!(
            "{}: loaded size {} >= retained {} ({:.2}x); merge estimate {} >= merge peak {} \
             ({:.2}x); admitted {admitted} >= used {used} ({:.2}x); load transient peak {}",
            case.name,
            m.loaded_size,
            m.retained,
            m.loaded_size as f64 / m.retained.max(1) as f64,
            m.merge_estimate,
            m.merge_peak,
            m.merge_estimate as f64 / m.merge_peak.max(1) as f64,
            admitted as f64 / used.max(1) as f64,
            m.load_peak,
        ));
        assert!(
            m.retained <= m.loaded_size && m.merge_peak <= m.merge_estimate,
            "{}: measured memory exceeds the admission\n{}",
            case.name,
            report.join("\n")
        );
    }
    // Visible with --nocapture; records how loose the bound is.
    #[allow(clippy::print_stderr)]
    for line in &report {
        eprintln!("{line}");
    }
}
