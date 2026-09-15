// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// TODO:
// - [x] Create base cases with HashMap
// - [x] Create on-disk size measurement
// - [x] Create different cases for the index. Ideal, 25% deletions, 80% deletions + compaction.
// - [ ] Create a benchmark for the get method
//   - [x] Average over all valid values
//   - [ ] Time to get a value that is not in the index
// - [ ] Create a benchmark for the new method (building the in-memory index)
// Optional:
// - [ ] Create in-memory size measurement (if possible)

// Questions:
// How can I write out the file? Where should I put it?
// How can I take a argument to set the size of the index?

use std::{collections::HashMap, io::Write, ops::Range, sync::Arc};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_io::ReadBatchParams;
use lance_table::format::pb;
use lance_table::rowids::FragmentRowIdIndex;
use lance_table::{
    rowids::{RowIdIndex, RowIdSequence, read_row_ids, write_row_ids},
    utils::stream::{RowIdAndDeletesConfig, apply_row_id_and_deletes},
};
use prost::Message;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

fn make_sequence(row_id_range: Range<u64>, deletions: usize) -> RowIdSequence {
    let mut sequence = RowIdSequence::from(row_id_range);

    // Delete every other row
    let delete_ids = sequence
        .iter()
        .step_by(2)
        .take(deletions)
        .collect::<Vec<_>>();
    sequence.delete(delete_ids);

    sequence
}

fn make_frag_sequences(
    num_rows: u64,
    num_frags: u64,
    percent_deletion: f32,
) -> Vec<(u32, Arc<RowIdSequence>)> {
    let rows_per_frag = num_rows / num_frags;
    let mut start = 0;
    (0..num_frags)
        .map(|i| {
            let sequence = make_sequence(
                start..(start + rows_per_frag),
                (rows_per_frag as f32 * percent_deletion) as usize,
            );
            start += rows_per_frag;
            (i as u32, Arc::new(sequence))
        })
        .collect()
}

// For range of values
// https://bheisler.github.io/criterion.rs/book/user_guide/benchmarking_with_inputs.html

fn num_rows() -> u64 {
    std::env::var("BENCH_NUM_ROWS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(1_000_000)
}

struct SizeStats {
    structure: String,
    percent_deletions: f32,
    size: u64,
}

struct SizeStatsFile {
    file: Option<std::fs::File>,
}

impl SizeStatsFile {
    fn new() -> Self {
        if let Ok(path) = std::env::var("BENCH_SIZE_STATS_FILE") {
            let mut file = std::fs::File::create(path).unwrap();
            // Header row
            writeln!(file, "structure,percent_deletions,size").unwrap();
            Self { file: Some(file) }
        } else {
            Self { file: None }
        }
    }

    fn write_row(&mut self, stats: SizeStats) {
        if let Some(file) = &mut self.file {
            writeln!(
                file,
                "\"{}\",{},{}",
                stats.structure, stats.percent_deletions, stats.size
            )
            .unwrap();
        }
    }
}

fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_id_index_creation");
    let mut stats_file = SizeStatsFile::new();

    for percent_deletions in [0.0, 0.25, 0.5] {
        let sequences = make_frag_sequences(num_rows(), 100, percent_deletions);

        let fragment_indices: Vec<FragmentRowIdIndex> = sequences
            .iter()
            .map(|(frag_id, sequence)| FragmentRowIdIndex {
                fragment_id: *frag_id,
                row_id_sequence: sequence.clone(),
                deletion_vector: Arc::new(DeletionVector::default()),
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("BuildIndex", percent_deletions),
            &percent_deletions,
            |b, _| {
                b.iter(|| {
                    let _index = RowIdIndex::new(&fragment_indices).unwrap();
                });
            },
        );

        // Measure size of index
        {
            let mut size = 0;
            for (_frag_id, sequence) in &sequences {
                size += write_row_ids(sequence).len() as u64;
            }
            let stats = SizeStats {
                structure: "RowIdIndex".to_string(),
                percent_deletions,
                size,
            };
            stats_file.write_row(stats);
        }

        // TODO: we should compare tombstoned vs compacted. We don't mind the
        // regression in the tombstoned case, but we want to see the improvement
        // in the compacted case.

        // TODO: collect size of sequences when serialized

        // TODO: also show building a BTreeMap and HashMap

        let flat_data = sequences
            .iter()
            .map(|(frag_id, sequence)| {
                let row_ids = sequence.iter().collect::<Vec<_>>();
                let row_addresses = (0..sequence.len())
                    .map(|i| RowAddress::new_from_parts(*frag_id, i as u32))
                    .map(u64::from)
                    .collect::<Vec<_>>();
                (row_ids, row_addresses)
            })
            .collect::<Vec<_>>();

        // Size of flat data is just 16 bytes per row
        let size = flat_data
            .iter()
            .map(|(ids, _addresses)| ids.len() * 16)
            .sum::<usize>() as u64;
        let stats = SizeStats {
            structure: "FlatData".to_string(),
            percent_deletions,
            size,
        };
        stats_file.write_row(stats);

        group.bench_with_input(
            BenchmarkId::new("BuildHashMap", percent_deletions),
            &percent_deletions,
            |b, _| {
                b.iter(|| {
                    let mut index = HashMap::new();
                    index.extend(flat_data.iter().flat_map(|(ids, addresses)| {
                        ids.iter().copied().zip(addresses.iter().copied())
                    }));
                });
            },
        );
    }

    group.finish();
}

fn bench_get_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_id_index_get_single");

    for percent_deletions in [0.0, 0.02, 0.25, 0.5, 0.8] {
        let sequences = make_frag_sequences(num_rows(), 100, percent_deletions);

        let fragment_indices: Vec<FragmentRowIdIndex> = sequences
            .iter()
            .map(|(frag_id, sequence)| FragmentRowIdIndex {
                fragment_id: *frag_id,
                row_id_sequence: sequence.clone(),
                deletion_vector: Arc::new(DeletionVector::default()),
            })
            .collect();

        let index = RowIdIndex::new(&fragment_indices).unwrap();

        let mut i = 0;
        let total_rows: u64 = num_rows();
        let mut next_id = || {
            let id = i;
            i += 241861;
            i %= total_rows;
            id
        };

        group.bench_with_input(
            BenchmarkId::new("GetIndex", percent_deletions),
            &percent_deletions,
            |b, _| {
                b.iter(|| {
                    let _ = index.get(next_id());
                });
            },
        );

        let flat_data = sequences
            .iter()
            .map(|(frag_id, sequence)| {
                let row_ids = sequence.iter().collect::<Vec<_>>();
                let row_addresses = (0..sequence.len())
                    .map(|i| RowAddress::new_from_parts(*frag_id, i as u32))
                    .map(u64::from)
                    .collect::<Vec<_>>();
                (row_ids, row_addresses)
            })
            .collect::<Vec<_>>();

        let index =
            {
                let mut index = HashMap::new();
                index.extend(flat_data.iter().flat_map(|(ids, addresses)| {
                    ids.iter().copied().zip(addresses.iter().copied())
                }));
                index
            };

        group.bench_with_input(
            BenchmarkId::new("GetHashMap", percent_deletions),
            &percent_deletions,
            |b, _| {
                b.iter(|| {
                    for i in 0..num_rows() {
                        let _ = index.get(&i);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_apply_row_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_row_id");

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )])),
        vec![Arc::new(UInt64Array::from(
            (0..num_rows()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();

    let config = RowIdAndDeletesConfig {
        params: ReadBatchParams::default(),
        with_row_id: true,
        with_row_addr: false,
        with_row_last_updated_at_version: false,
        with_row_created_at_version: false,
        deletion_vector: None,
        row_id_sequence: None,
        last_updated_at_sequence: None,
        created_at_sequence: None,
        make_deletions_null: false,
        total_num_rows: num_rows() as u32,
    };

    group.bench_function("ApplyRowId", |b| {
        let batch = batch.clone();
        b.iter(|| {
            let _ = apply_row_id_and_deletes(batch.clone(), 0, 0, &config);
        });
    });

    group.finish();
}

/// A fragment count large enough for the probe gate, small enough to build in
/// seconds. `BENCH_SHOT_FRAGMENTS=19960` reproduces the real table.
fn shot_fragments() -> usize {
    std::env::var("BENCH_SHOT_FRAGMENTS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(2000)
}

/// Rows per fragment of the shot table (17.4B rows over 19,960 fragments).
const SHOT_ROWS_PER_FRAGMENT: u64 = 870_000;

/// Mixes a position into a pseudo-random byte cheaply enough to fill gigabits.
fn hash_byte(position: u64, seed: u64) -> u8 {
    (position ^ seed)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(23)
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .to_le_bytes()[7]
}

/// `start..end` as a bitmap segment keeping about `density_pct` of its slots.
fn bitmap_segment(start: u64, end: u64, density_pct: u8, seed: u64) -> pb::U64Segment {
    let len = (end - start) as usize;
    let mut data = vec![0u8; len.div_ceil(8)];
    for (byte_index, byte) in data.iter_mut().enumerate() {
        let mut bits = 0u8;
        for bit in 0..8 {
            if hash_byte((byte_index * 8 + bit) as u64, seed) % 100 < density_pct {
                bits |= 1 << bit;
            }
        }
        *byte = bits;
    }
    if !len.is_multiple_of(8) {
        data[len / 8] &= (1u8 << (len % 8)) - 1;
    }
    pb::U64Segment {
        segment: Some(pb::u64_segment::Segment::RangeWithBitmap(
            pb::u64_segment::RangeWithBitmap {
                start,
                end,
                bitmap: data,
            },
        )),
    }
}

/// `start..end` minus about `holes_pct` percent of its slots, as sorted holes.
fn holes_segment(start: u64, end: u64, holes_pct: u8, seed: u64) -> pb::U64Segment {
    let holes: Vec<u32> = (0..(end - start))
        .filter(|offset| hash_byte(*offset, seed) % 100 < holes_pct)
        .map(|offset| offset as u32)
        .collect();
    let offsets = holes.iter().flat_map(|hole| hole.to_le_bytes()).collect();
    pb::U64Segment {
        segment: Some(pb::u64_segment::Segment::RangeWithHoles(
            pb::u64_segment::RangeWithHoles {
                start,
                end,
                holes: Some(pb::EncodedU64Array {
                    array: Some(pb::encoded_u64_array::Array::U32Array(
                        pb::encoded_u64_array::U32Array {
                            base: start,
                            offsets,
                        },
                    )),
                }),
            },
        )),
    }
}

fn range_segment(start: u64, end: u64) -> pb::U64Segment {
    pb::U64Segment {
        segment: Some(pb::u64_segment::Segment::Range(pb::u64_segment::Range {
            start,
            end,
        })),
    }
}

/// A live row id from `segment`, or `None` when the sampled slot is a hole.
fn sample_segment(segment: &pb::U64Segment, rng: &mut SmallRng) -> Option<u64> {
    use pb::u64_segment::Segment::*;
    match segment.segment.as_ref().unwrap() {
        Range(range) => Some(rng.random_range(range.start..range.end)),
        RangeWithBitmap(bitmap) => {
            let offset = rng.random_range(0..(bitmap.end - bitmap.start)) as usize;
            (bitmap.bitmap[offset / 8] & (1 << (offset % 8)) != 0)
                .then_some(bitmap.start + offset as u64)
        }
        RangeWithHoles(holes) => {
            let offset = rng.random_range(0..(holes.end - holes.start)) as u32;
            let Some(pb::encoded_u64_array::Array::U32Array(array)) =
                holes.holes.as_ref().unwrap().array.as_ref()
            else {
                unreachable!()
            };
            let is_hole = array
                .offsets
                .chunks_exact(4)
                .any(|hole| u32::from_le_bytes(hole.try_into().unwrap()) == offset);
            (!is_hole).then_some(holes.start + offset as u64)
        }
        _ => unreachable!(),
    }
}

/// Fragments shaped like the shot table after deletes and compaction: about
/// half keep one `Range`; the rest were compacted in groups of 16, and each
/// output fragment holds two dense (82%) slices taken from different source
/// fragments, so spans interleave and one id sits under up to 16 fragments.
/// One compacted slice in three carries only 1% holes instead of a bitmap.
/// Returns the fragments and a pool of live row ids to query.
fn shot_table_like(
    num_fragments: usize,
    rows_per_fragment: u64,
    seed: u64,
) -> (Vec<FragmentRowIdIndex>, Vec<u64>) {
    const GROUP: usize = 16;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sequences: Vec<Vec<pb::U64Segment>> = Vec::with_capacity(num_fragments);
    for group_start in (0..num_fragments).step_by(GROUP) {
        let group_end = (group_start + GROUP).min(num_fragments);
        let is_compacted = rng.random_range(0..100) < 44;
        if !is_compacted || group_end - group_start < 2 {
            for fragment in group_start..group_end {
                let start = fragment as u64 * rows_per_fragment;
                sequences.push(vec![range_segment(start, start + rows_per_fragment)]);
            }
            continue;
        }
        let half = rows_per_fragment / 2;
        let mut slices: Vec<(u64, u64)> = (group_start..group_end)
            .flat_map(|fragment| {
                let start = fragment as u64 * rows_per_fragment;
                [
                    (start, start + half),
                    (start + half, start + rows_per_fragment),
                ]
            })
            .collect();
        slices.shuffle(&mut rng);
        for pair in slices.chunks_exact(2) {
            let mut pair = [pair[0], pair[1]];
            pair.sort_unstable();
            let segments = pair
                .iter()
                .map(|(start, end)| {
                    let seed = rng.random();
                    if rng.random_range(0..3) == 0 {
                        holes_segment(*start, *end, 1, seed)
                    } else {
                        bitmap_segment(*start, *end, 82, seed)
                    }
                })
                .collect();
            sequences.push(segments);
        }
    }

    let mut pool = Vec::with_capacity(100_000);
    while pool.len() < 100_000 {
        let segments = &sequences[rng.random_range(0..sequences.len())];
        let segment = &segments[rng.random_range(0..segments.len())];
        if let Some(id) = sample_segment(segment, &mut rng) {
            pool.push(id);
        }
    }

    let fragments = sequences
        .into_iter()
        .enumerate()
        .map(|(fragment_id, segments)| {
            let bytes = pb::RowIdSequence { segments }.encode_to_vec();
            FragmentRowIdIndex {
                fragment_id: fragment_id as u32,
                row_id_sequence: Arc::new(read_row_ids(&bytes).unwrap()),
                deletion_vector: Arc::new(DeletionVector::default()),
            }
        })
        .collect();
    (fragments, pool)
}

fn bench_shot_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_id_index_shot_table");
    group.sample_size(10);
    let (fragments, pool) = shot_table_like(shot_fragments(), SHOT_ROWS_PER_FRAGMENT, 7);

    group.bench_function("build", |b| {
        b.iter(|| RowIdIndex::new(&fragments).unwrap());
    });

    let index = RowIdIndex::new(&fragments).unwrap();
    for n in [1usize, 1000, 100_000] {
        let ids = &pool[..n];
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("get_many", n), &ids, |b, ids| {
            b.iter(|| {
                let out = index.get_many(ids).unwrap();
                assert!(out.iter().all(Option::is_some));
            });
        });
    }
    group.throughput(criterion::Throughput::Elements(1));
    let mut next = 0usize;
    group.bench_function("get", |b| {
        b.iter(|| {
            next = (next + 1) % pool.len();
            assert!(index.get(pool[next]).unwrap().is_some());
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config=Criterion::default().with_profiler(lance_testing::pprof::PProfProfiler::new(100, lance_testing::pprof::Output::Flamegraph(None)));
    targets=bench_creation, bench_get_single, bench_apply_row_id, bench_shot_table);
#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_creation,
    bench_get_single,
    bench_apply_row_id,
    bench_shot_table
);
criterion_main!(benches);
