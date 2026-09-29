// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Parity tests for the lazy full-precision scan of a layered index: stage 1
//! bounds every row of the sign plane, stage 2 reranks the survivors against
//! gathered ex rows, and together they must reproduce the eager scan's heap
//! (row ids, distance bits, tie outcomes) and prune counters exactly.

use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::Schema;
use bytes::Bytes;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::cache::{CacheCodec, CacheDecode};
use lance_linalg::distance::DistanceType;
use rstest::rstest;

use super::RQBuildParams;
use super::builder::RabitQuantizer;
use super::dist_table_quant::{
    DistTableDequant, quantize_dist_table_into, quantize_dist_table_u16_into,
};
use super::layered::{PlaneBatch, RQPrecision, SignBounds, plane_columns};
use super::storage::{
    ExRows, RabitPruneCounters, RabitQuantizationStorage, SignStage, StagePruneCounts, SurvivorRow,
    select_full_survivors, take_captured_prune_counters,
};
use super::transform::{ERROR_FACTORS_COLUMN, RQTransformer};
use crate::vector::graph::OrderedNode;
use crate::vector::quantizer::{Quantization, QuantizerStorage};
use crate::vector::storage::{
    DistCalculator, DistanceCalculatorOptions, QueryResidual, RabitRawQueryContext, VectorStore,
};
use crate::vector::transform::Transformer;
use crate::vector::{ApproxMode, CENTROID_DIST_COLUMN, PART_ID_COLUMN};

type Heap = BinaryHeap<OrderedNode<u64>>;

/// Deterministic generator so every case is reproducible from its seed.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Uniform in `[-scale, scale)`.
    fn symmetric(&mut self, scale: f32) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }

    fn vector(&mut self, dim: usize, scale: f32) -> Vec<f32> {
        (0..dim).map(|_| self.symmetric(scale)).collect()
    }
}

/// One layered partition as the plane cache holds it.
struct Partition {
    sign: RecordBatch,
    hi: RecordBatch,
    lo: RecordBatch,
    row_ids: Vec<u64>,
    rotated_centroid: Vec<f32>,
    dist_q_c: f32,
}

impl Partition {
    fn len(&self) -> usize {
        self.row_ids.len()
    }

    /// The first `rows` rows, as fresh (unsliced) arrays.
    fn prefix(&self, rows: usize) -> Self {
        let indices = UInt32Array::from((0..rows as u32).collect::<Vec<_>>());
        Self {
            sign: self.sign.take(&indices).unwrap(),
            hi: self.hi.take(&indices).unwrap(),
            lo: self.lo.take(&indices).unwrap(),
            row_ids: self.row_ids[..rows].to_vec(),
            rotated_centroid: self.rotated_centroid.clone(),
            dist_q_c: self.dist_q_c,
        }
    }

    /// The eager loader's storage: all three planes, sign columns first.
    fn full_batch(&self) -> RecordBatch {
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for batch in [&self.sign, &self.hi, &self.lo] {
            fields.extend(batch.schema().fields().iter().cloned());
            columns.extend(batch.columns().iter().cloned());
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }
}

struct Fixture {
    rq: RabitQuantizer,
    /// Metric the storages are opened with; `Cosine` exercises the raw-query
    /// mapping to L2.
    distance_type: DistanceType,
    bits: u8,
    query: ArrayRef,
    raw_query: RabitRawQueryContext,
}

impl Fixture {
    fn new(dim: usize, bits: u8, distance_type: DistanceType, rng: &mut SplitMix64) -> Self {
        let sample = FixedSizeListArray::try_new_from_values(
            Float32Array::from(rng.vector(64 * dim, 1.0)),
            dim as i32,
        )
        .unwrap();
        let rq = RabitQuantizer::build(
            &sample,
            DistanceType::L2,
            &RQBuildParams::new(bits).with_layered(true),
        )
        .unwrap();
        let query = rng.vector(dim, 1.0);
        Self::with_query(rq, bits, distance_type, query)
    }

    fn with_query(
        rq: RabitQuantizer,
        bits: u8,
        distance_type: DistanceType,
        query: Vec<f32>,
    ) -> Self {
        let query: ArrayRef = Arc::new(Float32Array::from(query));
        let raw_query = rq.metadata_ref().prepare_raw_query_context(&query).unwrap();
        Self {
            rq,
            distance_type,
            bits,
            query,
            raw_query,
        }
    }

    fn dim(&self) -> usize {
        self.query.len()
    }

    fn rotated_dim(&self) -> usize {
        self.rq.metadata_ref().rotated_dim()
    }

    fn transform_metric(&self) -> DistanceType {
        match self.distance_type {
            DistanceType::Dot => DistanceType::Dot,
            _ => DistanceType::L2,
        }
    }

    /// Quantize `residuals` (row-major, `dim` wide) of a partition centered
    /// at `centroid`, with row ids `row_id_base..`: the transformer's batch,
    /// the rotated centroid and the query's centroid distance.
    fn transform(
        &self,
        residuals: &[f32],
        centroid: &[f32],
        row_id_base: u64,
    ) -> (RecordBatch, Vec<f32>, f32) {
        let dim = self.dim();
        let rows = residuals.len() / dim;
        let query = self.query.as_any().downcast_ref::<Float32Array>().unwrap();
        let query = query.values();
        let metric = self.transform_metric();
        let centroid_dists: Vec<f32> = residuals
            .chunks_exact(dim)
            .map(|residual| match metric {
                DistanceType::Dot => {
                    1.0 - residual
                        .iter()
                        .zip(centroid)
                        .map(|(r, c)| (r + c) * c)
                        .sum::<f32>()
                }
                _ => residual.iter().map(|r| r * r).sum(),
            })
            .collect();
        let row_ids: Vec<u64> = (row_id_base..row_id_base + rows as u64).collect();
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(residuals.to_vec()),
            dim as i32,
        )
        .unwrap();
        let input = RecordBatch::try_from_iter(vec![
            ("vector", Arc::new(vectors) as ArrayRef),
            (
                lance_core::ROW_ID,
                Arc::new(UInt64Array::from(row_ids)) as ArrayRef,
            ),
            (
                PART_ID_COLUMN,
                Arc::new(UInt32Array::from(vec![0; rows])) as ArrayRef,
            ),
            (
                CENTROID_DIST_COLUMN,
                Arc::new(Float32Array::from(centroid_dists)) as ArrayRef,
            ),
        ])
        .unwrap();
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(centroid.to_vec()),
            dim as i32,
        )
        .unwrap();
        let rotated_centroid = self
            .rq
            .quantize_split(&centroids)
            .unwrap()
            .rotated_residuals
            .unwrap();
        let batch = RQTransformer::new(self.rq.clone(), metric, centroids, "vector")
            .unwrap()
            .transform(&input)
            .unwrap();
        let dist_q_c = match metric {
            DistanceType::Dot => 1.0 - query.iter().zip(centroid).map(|(q, c)| q * c).sum::<f32>(),
            _ => query
                .iter()
                .zip(centroid)
                .map(|(q, c)| (q - c).powi(2))
                .sum(),
        };
        (batch, rotated_centroid, dist_q_c)
    }

    /// Encode a layered partition and split it into its cache planes.
    fn encode(&self, residuals: &[f32], centroid: &[f32], row_id_base: u64) -> Partition {
        let (batch, rotated_centroid, dist_q_c) = self.transform(residuals, centroid, row_id_base);
        let plane = |plane: u8| {
            let schema = batch.schema();
            let indices: Vec<usize> = plane_columns(plane, SignBounds::default())
                .iter()
                .map(|name| schema.index_of(name).unwrap())
                .collect();
            batch.project(&indices).unwrap()
        };
        let sign = plane(0);
        let row_ids = sign[lance_core::ROW_ID]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();
        Partition {
            sign,
            hi: plane(1),
            lo: plane(2),
            row_ids,
            rotated_centroid,
            dist_q_c,
        }
    }

    fn random_partition(&self, rows: usize, row_id_base: u64, rng: &mut SplitMix64) -> Partition {
        let centroid = rng.vector(self.dim(), 0.2);
        let residuals = rng.vector(rows * self.dim(), 1.0);
        self.encode(&residuals, &centroid, row_id_base)
    }

    fn residual<'a>(&'a self, partition: &'a Partition) -> Option<QueryResidual<'a>> {
        Some(QueryResidual::RabitRawQuery {
            rotated_centroid: Some(&partition.rotated_centroid),
            query: Some(&self.raw_query),
        })
    }

    fn full_storage(&self, partition: &Partition) -> RabitQuantizationStorage {
        RabitQuantizationStorage::try_from_batch(
            partition.full_batch(),
            self.rq.metadata_ref(),
            self.distance_type,
            None,
        )
        .unwrap()
    }

    fn sign_storage(&self, partition: &Partition) -> RabitQuantizationStorage {
        RabitQuantizationStorage::try_from_sign_plane_for_full(
            partition.sign.clone(),
            self.rq.metadata_ref(),
            self.distance_type,
        )
        .unwrap()
    }
}

fn full_options(approx_mode: ApproxMode) -> DistanceCalculatorOptions {
    DistanceCalculatorOptions {
        approx_mode,
        rq_precision: RQPrecision::Full,
    }
}

#[derive(Debug, Clone, Copy)]
struct Scan {
    k: usize,
    lower: Option<f32>,
    upper: Option<f32>,
    approx_mode: ApproxMode,
}

impl Scan {
    fn top_k(k: usize) -> Self {
        Self {
            k,
            lower: None,
            upper: None,
            approx_mode: ApproxMode::Normal,
        }
    }
}

/// Where stage 1 takes its heap threshold from. Every variant but the
/// negative controls' `Fixed` is a legal (possibly stale) heap top.
#[derive(Debug, Clone, Copy)]
enum Threshold {
    Unbounded,
    Live,
    /// The live top as of this many partitions ago.
    Stale(usize),
    /// A value just above the live top.
    NextUp,
    /// The top published when the heap first filled.
    FirstFull,
    /// A caller-chosen threshold, used only to show illegal ones break parity.
    Fixed(f32),
}

/// How the survivors' ex rows are served to stage 2.
#[derive(Debug, Clone, Copy)]
enum Gather {
    /// Whole planes; the ex index is the sign-plane offset.
    Identity,
    /// Arrow `take` of the survivors.
    Take,
    /// A persistent plane-cache round trip with a row-selective decode.
    Codec,
}

const GATHERS: [Gather; 3] = [Gather::Identity, Gather::Take, Gather::Codec];

fn codec_rows(plane: &RecordBatch, rows: &[u32]) -> RecordBatch {
    let codec = CacheCodec::from_impl::<PlaneBatch>();
    let mut encoded = Vec::new();
    codec
        .serialize(
            &(Arc::new(PlaneBatch(plane.clone())) as Arc<dyn std::any::Any + Send + Sync>),
            &mut encoded,
        )
        .unwrap();
    let encoded = Bytes::from(encoded);
    let read =
        |range: std::ops::Range<usize>| -> lance_core::Result<Bytes> { Ok(encoded.slice(range)) };
    let CacheDecode::Hit(decoded) = codec.deserialize_rows(&read, rows) else {
        panic!("row-selective plane decode failed for rows {rows:?}")
    };
    decoded.downcast::<PlaneBatch>().unwrap().0.clone()
}

fn filter_mask(partition: &Partition, filter: Option<&HashSet<u64>>) -> Option<Vec<bool>> {
    filter.map(|filter| {
        partition
            .row_ids
            .iter()
            .map(|row_id| filter.contains(row_id))
            .collect()
    })
}

fn stage_one(
    fixture: &Fixture,
    partition: &Partition,
    sign: &RabitQuantizationStorage,
    approx_mode: ApproxMode,
) -> std::result::Result<SignStage, &'static str> {
    let mut scratch = Vec::new();
    let calc = sign.dist_calculator_with_scratch(
        fixture.query.clone(),
        partition.dist_q_c,
        fixture.residual(partition),
        &mut scratch,
        full_options(approx_mode),
    );
    let mut stage = SignStage::default();
    calc.full_sign_stage_with_scratch(
        &mut stage,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut Vec::new(),
    )?;
    Ok(stage)
}

/// The eager scan exactly as `FlatIndex` drives it; returns the heap and the
/// per-partition prune counters it reported.
fn eager_scan(
    fixture: &Fixture,
    partitions: &[&Partition],
    scan: Scan,
    filter: Option<&HashSet<u64>>,
) -> (Heap, Vec<RabitPruneCounters>) {
    take_captured_prune_counters();
    let mut heap = BinaryHeap::new();
    for partition in partitions {
        let storage = fixture.full_storage(partition);
        let mut scratch = Vec::new();
        let calc = storage.dist_calculator_with_scratch(
            fixture.query.clone(),
            partition.dist_q_c,
            fixture.residual(partition),
            &mut scratch,
            full_options(scan.approx_mode),
        );
        let (mut dists, mut u16s, mut u8s, mut u32s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        match filter {
            None => calc.accumulate_topk_with_scratch(
                scan.k,
                scan.lower,
                scan.upper,
                |id| storage.row_id(id),
                &mut heap,
                &mut dists,
                &mut u16s,
                &mut u8s,
                &mut u32s,
            ),
            Some(filter) => calc.accumulate_filtered_topk_with_scratch(
                scan.k,
                scan.lower,
                scan.upper,
                storage
                    .row_ids()
                    .enumerate()
                    .map(|(id, &row_id)| (id as u32, row_id)),
                |row_id| filter.contains(&row_id),
                &mut heap,
                &mut dists,
                &mut u16s,
                &mut u8s,
                &mut u32s,
            ),
        }
    }
    (heap, take_captured_prune_counters())
}

/// Knobs of one lazy replay that the tests vary.
#[derive(Debug, Clone, Copy)]
struct Lazy {
    threshold: Threshold,
    gather: Gather,
}

/// The lazy two-stage scan over the same partitions, in the same probe
/// order and on one heap, as the IVF orchestrator drives it.
fn lazy_scan(
    fixture: &Fixture,
    partitions: &[&Partition],
    scan: Scan,
    filter: Option<&HashSet<u64>>,
    lazy: Lazy,
) -> (Heap, Vec<RabitPruneCounters>, Vec<StagePruneCounts>) {
    take_captured_prune_counters();
    let mut heap: Heap = BinaryHeap::new();
    let mut returned = Vec::new();
    let mut stage1 = Vec::new();
    let heap_top = |heap: &Heap| {
        (scan.k > 0 && heap.len() >= scan.k).then(|| heap.peek().map(|node| node.dist.0))
    };
    // history[j] is the published threshold after j partitions were scored.
    let mut history = vec![None];
    let mut first_full = None;
    for (probe, partition) in partitions.iter().enumerate() {
        let sign = fixture.sign_storage(partition);
        let stage = stage_one(fixture, partition, &sign, scan.approx_mode).unwrap();
        let accept = filter_mask(partition, filter);
        let live = heap_top(&heap).flatten();
        let threshold = match lazy.threshold {
            Threshold::Unbounded => None,
            Threshold::Live => live,
            Threshold::Stale(staleness) => history[probe.saturating_sub(staleness)],
            Threshold::NextUp => live.map(f32::next_up),
            Threshold::FirstFull => first_full,
            // Published like a legal threshold, only once the heap is full.
            Threshold::Fixed(threshold) => live.map(|_| threshold),
        };
        let mut survivors = Vec::new();
        let mut counts = StagePruneCounts::default();
        select_full_survivors(
            &stage.lower_bounds,
            accept.as_deref(),
            scan.upper,
            threshold,
            &mut survivors,
            &mut counts,
        );
        if !matches!(lazy.threshold, Threshold::Fixed(_)) {
            let mut needed = Vec::new();
            select_full_survivors(
                &stage.lower_bounds,
                accept.as_deref(),
                scan.upper,
                live,
                &mut needed,
                &mut StagePruneCounts::default(),
            );
            let fetched: HashSet<u32> = survivors.iter().copied().collect();
            assert!(
                needed.iter().all(|offset| fetched.contains(offset)),
                "a legal threshold must fetch every row the live heap can still accept ({lazy:?})"
            );
        }
        let (hi, lo, ex_indices): (RecordBatch, RecordBatch, Vec<u32>) = match lazy.gather {
            Gather::Identity => (
                partition.hi.clone(),
                partition.lo.clone(),
                survivors.clone(),
            ),
            Gather::Take => {
                let indices = UInt32Array::from(survivors.clone());
                (
                    partition.hi.take(&indices).unwrap(),
                    partition.lo.take(&indices).unwrap(),
                    (0..survivors.len() as u32).collect(),
                )
            }
            Gather::Codec => (
                codec_rows(&partition.hi, &survivors),
                codec_rows(&partition.lo, &survivors),
                (0..survivors.len() as u32).collect(),
            ),
        };
        let ex_rows =
            ExRows::from_plane_batches(&hi, &lo, fixture.rotated_dim(), fixture.bits).unwrap();
        let mut scratch = Vec::new();
        let calc = sign
            .dist_calculator_with_ex_rows(
                fixture.query.clone(),
                partition.dist_q_c,
                fixture.residual(partition),
                &mut scratch,
                full_options(scan.approx_mode),
                ex_rows,
            )
            .unwrap();
        let rows: Vec<SurvivorRow> = survivors
            .iter()
            .zip(ex_indices)
            .map(|(&offset, ex_index)| SurvivorRow {
                ex_index,
                row_id: partition.row_ids[offset as usize],
                binary_ip: stage.binary_ips[offset as usize],
                lower_bound: stage.lower_bounds[offset as usize],
            })
            .collect();
        let was_full = heap_top(&heap).is_some();
        let mut published = None;
        let counters = calc.accumulate_survivor_rows(
            scan.k,
            scan.lower,
            scan.upper,
            &rows,
            counts,
            &mut heap,
            |top| {
                assert!(published.is_none(), "the first-full event fires once");
                published = Some(top);
            },
        );
        assert_eq!(
            published.is_some(),
            !was_full && heap_top(&heap).is_some(),
            "first-full fires exactly when this partition fills the heap"
        );
        if first_full.is_none() {
            first_full = published;
        }
        if scan.k > 0 && partition.len() > 0 {
            returned.push(counters);
            stage1.push(counts);
        }
        history.push(heap_top(&heap).flatten());
    }
    let captured = take_captured_prune_counters();
    assert_eq!(
        returned, captured,
        "returned counters are the reported ones"
    );
    (heap, captured, stage1)
}

fn to_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn heap_bits(heap: Heap) -> Vec<(u64, u32)> {
    heap.into_vec()
        .into_iter()
        .map(|node| (node.id, node.dist.0.to_bits()))
        .collect()
}

#[track_caller]
fn assert_lazy_matches_eager(
    fixture: &Fixture,
    partitions: &[&Partition],
    scan: Scan,
    filter: Option<&HashSet<u64>>,
    lazy: Lazy,
) -> HeapPrunes {
    let (eager, eager_counters) = eager_scan(fixture, partitions, scan, filter);
    let (actual, lazy_counters, stage1) = lazy_scan(fixture, partitions, scan, filter, lazy);
    assert_eq!(
        heap_bits(actual),
        heap_bits(eager),
        "heap differs: {scan:?} {lazy:?}"
    );
    assert_eq!(
        lazy_counters, eager_counters,
        "prune counters differ: {scan:?} {lazy:?}"
    );
    let stage1_heap: usize = stage1.iter().map(|counts| counts.pruned_heap).sum();
    let total_heap: usize = lazy_counters
        .iter()
        .map(|counters| counters.pruned_heap)
        .sum();
    HeapPrunes {
        stage1: stage1_heap,
        stage2: total_heap - stage1_heap,
    }
}

/// Heap prunes of a lazy replay by stage; stage 2 prunes the rows a stale
/// threshold over-fetched.
#[derive(Debug, Default, Clone, Copy)]
struct HeapPrunes {
    stage1: usize,
    stage2: usize,
}

fn legal_thresholds(rng: &mut SplitMix64) -> [Threshold; 5] {
    [
        Threshold::Unbounded,
        Threshold::Live,
        Threshold::Stale(1 + rng.below(8)),
        Threshold::NextUp,
        Threshold::FirstFull,
    ]
}

/// Stage 1 reproduces the eager full store's binary inner products and native
/// lower bounds bit for bit, across SIMD block boundaries, padded dims, every
/// layered width, both LUT precisions and the exact-dequantization fallback.
#[rstest]
fn sign_stage_matches_full_store(#[values(64, 96, 1024)] dim: usize, #[values(5, 7, 9)] bits: u8) {
    const ROWS: [usize; 14] = [0, 1, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 2158];
    let mut rng = SplitMix64(dim as u64 * 131 + bits as u64);
    let fixture = Fixture::new(dim, bits, DistanceType::L2, &mut rng);
    let all_rows = fixture.random_partition(ROWS[ROWS.len() - 1], 0, &mut rng);
    let query = fixture
        .query
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .values()
        .to_vec();
    // A query this large makes the affine LUT reconstruction overflow while
    // the table itself stays finite, which forces the exact per-row fallback.
    // The window depends on the (unseeded) rotation, so sweep scales densely
    // from 1e30 to 1e38 in quarter decades.
    let exact_fixture = (120..=152)
        .map(|quarter_decades| 10f32.powf(quarter_decades as f32 / 4.0))
        .map(|scale| {
            Fixture::with_query(
                fixture.rq.clone(),
                bits,
                DistanceType::L2,
                query.iter().map(|q| q * scale).collect(),
            )
        })
        .find(|fixture| {
            let dist_table = &fixture.raw_query.dist_table;
            dist_table.iter().all(|value| value.is_finite())
                && quantize_dist_table_into(dist_table, &mut Vec::new()) == DistTableDequant::Exact
        })
        .expect("some query scale must force the exact dequantization fallback");
    assert_eq!(
        quantize_dist_table_u16_into(&exact_fixture.raw_query.dist_table, &mut Vec::new()),
        DistTableDequant::Exact
    );

    for rows in ROWS {
        let partition = all_rows.prefix(rows);
        for fixture in [&fixture, &exact_fixture] {
            let full = fixture.full_storage(&partition);
            let sign = fixture.sign_storage(&partition);
            assert_eq!(sign.len(), rows);
            for approx_mode in [ApproxMode::Normal, ApproxMode::Accurate] {
                for residual in [fixture.residual(&partition), None] {
                    let (mut full_scratch, mut sign_scratch) = (Vec::new(), Vec::new());
                    let options = full_options(approx_mode);
                    let eager = full.dist_calculator_with_scratch(
                        fixture.query.clone(),
                        partition.dist_q_c,
                        residual,
                        &mut full_scratch,
                        options,
                    );
                    let calc = sign.dist_calculator_with_scratch(
                        fixture.query.clone(),
                        partition.dist_q_c,
                        residual,
                        &mut sign_scratch,
                        options,
                    );
                    let mut stage = SignStage::default();
                    calc.full_sign_stage_with_scratch(
                        &mut stage,
                        &mut Vec::new(),
                        &mut Vec::new(),
                        &mut Vec::new(),
                    )
                    .unwrap();
                    let expected_ips = eager.binary_inner_products();
                    assert_eq!(
                        to_bits(&stage.binary_ips),
                        to_bits(&expected_ips),
                        "binary inner products differ: rows={rows} mode={approx_mode:?}"
                    );
                    let expected_bounds: Vec<f32> = expected_ips
                        .iter()
                        .enumerate()
                        .map(|(row, &ip)| eager.raw_query_lower_bound(row, ip).unwrap())
                        .collect();
                    assert_eq!(
                        to_bits(&stage.lower_bounds),
                        to_bits(&expected_bounds),
                        "lower bounds differ: rows={rows} mode={approx_mode:?}"
                    );
                }
            }
        }
    }
}

/// Property test: random probe sequences of layered partitions, all legal
/// threshold schedules and all three ways of serving the ex rows produce the
/// eager heap and prune counters bit for bit.
#[rstest]
fn lazy_replay_equals_dense_heap(
    #[values(5, 7, 9)] bits: u8,
    #[values(DistanceType::L2, DistanceType::Dot, DistanceType::Cosine)] metric: DistanceType,
) {
    const SEEDS: u64 = 200;
    const POOL_ROWS: [usize; 9] = [0, 1, 17, 31, 32, 33, 64, 100, 150];
    let metric_seed = match metric {
        DistanceType::L2 => 1,
        DistanceType::Dot => 2,
        _ => 3,
    };
    let mut rng = SplitMix64(bits as u64 * 7919 + metric_seed);
    let dim = if bits == 7 { 96 } else { 64 };
    let fixture = Fixture::new(dim, bits, metric, &mut rng);
    let pool: Vec<Partition> = POOL_ROWS
        .iter()
        .enumerate()
        .map(|(part, &rows)| fixture.random_partition(rows, part as u64 * 10_000, &mut rng))
        .collect();
    // Range bounds drawn from the spread of full-precision scores.
    let mut scores: Vec<f32> = pool
        .iter()
        .filter(|partition| partition.len() > 0)
        .flat_map(|partition| {
            let storage = fixture.full_storage(partition);
            storage
                .dist_calculator_with_scratch(
                    fixture.query.clone(),
                    partition.dist_q_c,
                    fixture.residual(partition),
                    &mut Vec::new(),
                    full_options(ApproxMode::Normal),
                )
                .distance_all(partition.len())
        })
        .collect();
    scores.sort_by(f32::total_cmp);
    let quantile = |q: f64| scores[(q * (scores.len() - 1) as f64) as usize];

    let mut prunes = HeapPrunes::default();
    for seed in 0..SEEDS {
        let mut rng = SplitMix64(seed ^ 0xA5A5_0000);
        let mut order: Vec<usize> = (0..pool.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let partitions: Vec<&Partition> = order[..1 + rng.below(6)]
            .iter()
            .map(|&part| &pool[part])
            .collect();
        let total: usize = partitions.iter().map(|partition| partition.len()).sum();
        let approx_mode = if rng.below(2) == 0 {
            ApproxMode::Normal
        } else {
            ApproxMode::Accurate
        };
        let (lower, upper) = match rng.below(4) {
            0 => (None, None),
            1 => (Some(quantile(0.1)), None),
            2 => (None, Some(quantile(0.6))),
            _ => (Some(quantile(0.05)), Some(quantile(0.7))),
        };
        let filter: Option<HashSet<u64>> = (rng.below(3) == 0).then(|| {
            partitions
                .iter()
                .flat_map(|partition| partition.row_ids.iter().copied())
                .filter(|_| rng.below(3) != 0)
                .collect()
        });
        for k in [0, 1, 7, 100, total + 1] {
            let scan = Scan {
                k,
                lower,
                upper,
                approx_mode,
            };
            for threshold in legal_thresholds(&mut rng) {
                let gather = GATHERS[rng.below(GATHERS.len())];
                let replay = assert_lazy_matches_eager(
                    &fixture,
                    &partitions,
                    scan,
                    filter.as_ref(),
                    Lazy { threshold, gather },
                );
                prunes.stage1 += replay.stage1;
                prunes.stage2 += replay.stage2;
            }
        }
    }
    assert!(
        prunes.stage1 > 0 && prunes.stage2 > 0,
        "the replays must prune in stage 1 and re-prune stale over-fetch in stage 2: {prunes:?}"
    );
}

/// Duplicate rows (identical codes and factors) tie within a partition,
/// across a 32-row SIMD block boundary and across partitions; the incumbent
/// keeps its slot in both scans, and the outcome depends on probe order.
#[test]
fn ties_keep_incumbents_in_probe_order() {
    let mut rng = SplitMix64(0x7135);
    let fixture = Fixture::new(64, 7, DistanceType::L2, &mut rng);
    let dim = fixture.dim();
    let centroid = rng.vector(dim, 0.2);
    let mut residuals = rng.vector(70 * dim, 1.0);
    // Rows 31 and 32 straddle the first SIMD block boundary; both copy row 5.
    let duplicate = residuals[5 * dim..6 * dim].to_vec();
    residuals[31 * dim..32 * dim].copy_from_slice(&duplicate);
    residuals[32 * dim..33 * dim].copy_from_slice(&duplicate);
    let first = fixture.encode(&residuals, &centroid, 0);
    let second = fixture.encode(&residuals, &centroid, 1_000);

    let forward = [&first, &second];
    let backward = [&second, &first];
    let mut order_sensitive = false;
    for k in 1..=24 {
        for probes in [&forward[..], &backward[..]] {
            for gather in GATHERS {
                for threshold in [Threshold::Live, Threshold::Stale(1), Threshold::FirstFull] {
                    assert_lazy_matches_eager(
                        &fixture,
                        probes,
                        Scan::top_k(k),
                        None,
                        Lazy { threshold, gather },
                    );
                }
            }
        }
        let ids = |probes: &[&Partition]| {
            let mut ids: Vec<u64> = eager_scan(&fixture, probes, Scan::top_k(k), None)
                .0
                .into_iter()
                .map(|node| node.id)
                .collect();
            ids.sort_unstable();
            ids
        };
        order_sensitive |= ids(&forward[..]) != ids(&backward[..]);
    }
    assert!(
        order_sensitive,
        "swapping identical partitions must change which tied rows are kept"
    );
}

/// Negative controls: the parity above depends on carrying the original
/// partition's binary inner products and on taking the threshold only from
/// the full-score heap.
#[test]
fn illegal_inputs_break_parity() {
    let mut rng = SplitMix64(0x4e67);
    let fixture = Fixture::new(64, 7, DistanceType::L2, &mut rng);

    // Recomputing a row's inner product from a compacted sign plane moves it
    // between the quantized SIMD LUT and the float tail LUT.
    let partition = fixture.random_partition(33, 0, &mut rng);
    let sign = fixture.sign_storage(&partition);
    let stage = stage_one(&fixture, &partition, &sign, ApproxMode::Normal).unwrap();
    let ex_rows = ExRows::from_plane_batches(
        &partition.hi,
        &partition.lo,
        fixture.rotated_dim(),
        fixture.bits,
    )
    .unwrap();
    let mut scratch = Vec::new();
    let calc = sign
        .dist_calculator_with_ex_rows(
            fixture.query.clone(),
            partition.dist_q_c,
            fixture.residual(&partition),
            &mut scratch,
            full_options(ApproxMode::Normal),
            ex_rows,
        )
        .unwrap();
    let changed_scores = (0..partition.len())
        .filter(|&row| {
            let single = partition_rows(&partition, &[row as u32]);
            let recomputed = stage_one(
                &fixture,
                &single,
                &fixture.sign_storage(&single),
                ApproxMode::Normal,
            )
            .unwrap()
            .binary_ips[0];
            let carried =
                calc.distance_with_binary_inner_product(row as u32, stage.binary_ips[row]);
            let rescored = calc.distance_with_binary_inner_product(row as u32, recomputed);
            carried.to_bits() != rescored.to_bits()
        })
        .count();
    assert!(
        changed_scores > 0,
        "recomputed inner products must change some full-precision score"
    );

    // A threshold below the live heap top prunes a row the eager scan keeps.
    let probes: Vec<Partition> = (0..4)
        .map(|part| fixture.random_partition(100, part * 1_000, &mut rng))
        .collect();
    let probes: Vec<&Partition> = probes.iter().collect();
    let scan = Scan::top_k(10);
    let (last, entrant_bound, live, eager_bits) = (1..probes.len())
        .rev()
        .find_map(|last| {
            let prefix = &probes[..=last];
            let (eager, _) = eager_scan(&fixture, prefix, scan, None);
            let kept: HashSet<u64> = eager.iter().map(|node| node.id).collect();
            let partition = prefix[last];
            let stage = stage_one(
                &fixture,
                partition,
                &fixture.sign_storage(partition),
                scan.approx_mode,
            )
            .unwrap();
            let entrant_bound = partition
                .row_ids
                .iter()
                .zip(&stage.lower_bounds)
                .filter(|(row_id, _)| kept.contains(row_id))
                .map(|(_, &bound)| bound)
                .min_by(f32::total_cmp)?;
            let (before, _) = eager_scan(&fixture, &prefix[..last], scan, None);
            let live = before.peek().unwrap().dist.0;
            Some((last, entrant_bound, live, heap_bits(eager)))
        })
        .expect("some later probe must place a row in the final top-k");
    assert!(
        entrant_bound < live,
        "an entrant was reranked below the live top"
    );
    let (tightened, _, _) = lazy_scan(
        &fixture,
        &probes[..=last],
        scan,
        None,
        Lazy {
            threshold: Threshold::Fixed(entrant_bound),
            gather: Gather::Take,
        },
    );
    assert_ne!(heap_bits(tightened), eager_bits);

    // The k-th best 1-bit estimate is not a full-score threshold.
    let mut broke = 0;
    for k in 1..=30 {
        let scan = Scan::top_k(k);
        let (eager, _) = eager_scan(&fixture, &probes, scan, None);
        let mut estimates: Vec<f32> = probes
            .iter()
            .flat_map(|partition| {
                fixture
                    .full_storage(partition)
                    .dist_calculator_with_scratch(
                        fixture.query.clone(),
                        partition.dist_q_c,
                        fixture.residual(partition),
                        &mut Vec::new(),
                        full_options(ApproxMode::Fast),
                    )
                    .distance_all(partition.len())
            })
            .collect();
        estimates.sort_by(f32::total_cmp);
        let (estimated, _, _) = lazy_scan(
            &fixture,
            &probes,
            scan,
            None,
            Lazy {
                threshold: Threshold::Fixed(estimates[k - 1]),
                gather: Gather::Take,
            },
        );
        broke += usize::from(heap_bits(estimated) != heap_bits(eager));
    }
    assert!(
        broke > 0,
        "a 1-bit estimate threshold must lose some true top-k row"
    );
}

/// The rows of `partition` at `rows`, re-materialized as their own partition.
fn partition_rows(partition: &Partition, rows: &[u32]) -> Partition {
    let indices = UInt32Array::from(rows.to_vec());
    Partition {
        sign: partition.sign.take(&indices).unwrap(),
        hi: partition.hi.take(&indices).unwrap(),
        lo: partition.lo.take(&indices).unwrap(),
        row_ids: rows
            .iter()
            .map(|&row| partition.row_ids[row as usize])
            .collect(),
        rotated_centroid: partition.rotated_centroid.clone(),
        dist_q_c: partition.dist_q_c,
    }
}

/// Prefiltered partitions follow the eager filtered scan; rows the filter
/// rejects are never selected.
#[rstest]
#[case::all(|_: usize, _: usize| true)]
#[case::none(|_: usize, _: usize| false)]
#[case::alternating(|offset: usize, _: usize| offset.is_multiple_of(2))]
#[case::single_row(|offset: usize, _: usize| offset == 3)]
#[case::every_17th(|offset: usize, _: usize| offset.is_multiple_of(17))]
#[case::simd_tail_only(|offset: usize, rows: usize| offset >= rows - rows % 32)]
fn filters_match_eager_filtered_scan(#[case] accept: fn(usize, usize) -> bool) {
    let mut rng = SplitMix64(0xF117);
    let fixture = Fixture::new(64, 9, DistanceType::L2, &mut rng);
    let partitions: Vec<Partition> = [70, 33, 5, 100]
        .into_iter()
        .enumerate()
        .map(|(part, rows)| fixture.random_partition(rows, part as u64 * 1_000, &mut rng))
        .collect();
    let probes: Vec<&Partition> = partitions.iter().collect();
    let filter: HashSet<u64> = partitions
        .iter()
        .flat_map(|partition| {
            let rows = partition.len();
            partition
                .row_ids
                .iter()
                .enumerate()
                .filter(move |(offset, _)| accept(*offset, rows))
                .map(|(_, &row_id)| row_id)
        })
        .collect();
    for partition in &partitions {
        let stage = stage_one(
            &fixture,
            partition,
            &fixture.sign_storage(partition),
            ApproxMode::Normal,
        )
        .unwrap();
        let mask = filter_mask(partition, Some(&filter)).unwrap();
        let mut survivors = Vec::new();
        let mut counts = StagePruneCounts::default();
        select_full_survivors(
            &stage.lower_bounds,
            Some(mask.as_slice()),
            None,
            None,
            &mut survivors,
            &mut counts,
        );
        assert!(survivors.iter().all(|&offset| mask[offset as usize]));
        assert_eq!(
            counts.candidates,
            mask.iter().filter(|&&accepted| accepted).count()
        );
    }
    for k in [1, 10, 400] {
        for gather in GATHERS {
            for threshold in [Threshold::Live, Threshold::Stale(2), Threshold::Unbounded] {
                assert_lazy_matches_eager(
                    &fixture,
                    &probes,
                    Scan::top_k(k),
                    Some(&filter),
                    Lazy { threshold, gather },
                );
            }
        }
    }
}

/// Query distance ranges, including an upper bound below every lower bound
/// (no survivors), a lower bound above every score (every exact score is
/// rejected) and a heap that never fills.
#[test]
fn range_bounds_match_eager() {
    let mut rng = SplitMix64(0x5A9E);
    let fixture = Fixture::new(64, 5, DistanceType::L2, &mut rng);
    let partitions: Vec<Partition> = [40, 65, 12]
        .into_iter()
        .enumerate()
        .map(|(part, rows)| fixture.random_partition(rows, part as u64 * 1_000, &mut rng))
        .collect();
    let probes: Vec<&Partition> = partitions.iter().collect();
    let total: usize = partitions.iter().map(Partition::len).sum();
    let mut bounds = Vec::new();
    let mut scores = Vec::new();
    for partition in &partitions {
        bounds.extend(
            stage_one(
                &fixture,
                partition,
                &fixture.sign_storage(partition),
                ApproxMode::Normal,
            )
            .unwrap()
            .lower_bounds,
        );
        scores.extend(
            fixture
                .full_storage(partition)
                .dist_calculator_with_scratch(
                    fixture.query.clone(),
                    partition.dist_q_c,
                    fixture.residual(partition),
                    &mut Vec::new(),
                    full_options(ApproxMode::Normal),
                )
                .distance_all(partition.len()),
        );
    }
    let min_bound = bounds.iter().copied().min_by(f32::total_cmp).unwrap();
    let max_score = scores.iter().copied().max_by(f32::total_cmp).unwrap();
    let median = {
        let mut sorted = scores.clone();
        sorted.sort_by(f32::total_cmp);
        sorted[sorted.len() / 2]
    };
    let ranges = [
        (None, None),
        (Some(median), None),
        (None, Some(median)),
        (Some(median - 1.0), Some(median + 1.0)),
        (None, Some(min_bound.next_down())),
        (Some(max_score.next_up()), None),
    ];
    for (lower, upper) in ranges {
        for k in [1, 10, total + 5] {
            let scan = Scan {
                k,
                lower,
                upper,
                approx_mode: ApproxMode::Normal,
            };
            for gather in GATHERS {
                for threshold in [Threshold::Live, Threshold::Stale(1), Threshold::NextUp] {
                    assert_lazy_matches_eager(
                        &fixture,
                        &probes,
                        scan,
                        None,
                        Lazy { threshold, gather },
                    );
                }
            }
        }
    }
    let (heap, counters) = eager_scan(
        &fixture,
        &probes,
        Scan {
            k: 10,
            lower: None,
            upper: Some(min_bound.next_down()),
            approx_mode: ApproxMode::Normal,
        },
        None,
    );
    assert!(heap.is_empty());
    assert!(
        counters
            .iter()
            .all(|c| c.pruned_upper_bound == c.candidates)
    );
    let (heap, counters) = eager_scan(
        &fixture,
        &probes,
        Scan {
            k: 10,
            lower: Some(max_score.next_up()),
            upper: None,
            approx_mode: ApproxMode::Normal,
        },
        None,
    );
    assert!(heap.is_empty());
    assert!(
        counters
            .iter()
            .all(|c| c.exact_rejected == c.exact && c.exact > 0)
    );
}

/// NaN bounds are never pruned and infinite ones are pruned as upper-bound
/// rows, in both scans.
#[test]
fn non_finite_bounds_match_eager() {
    let mut rng = SplitMix64(0x0A11);
    let fixture = Fixture::new(64, 7, DistanceType::L2, &mut rng);
    let partitions: Vec<Partition> = [70, 40]
        .into_iter()
        .enumerate()
        .map(|(part, rows)| {
            let mut partition = fixture.random_partition(rows, part as u64 * 1_000, &mut rng);
            let errors = partition.sign[ERROR_FACTORS_COLUMN]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .iter()
                .enumerate()
                .map(|(row, &error)| match row % 11 {
                    3 => f32::NAN,
                    7 => f32::NEG_INFINITY,
                    _ => error,
                })
                .collect::<Vec<_>>();
            partition.sign = partition
                .sign
                .replace_column_by_name(ERROR_FACTORS_COLUMN, Arc::new(Float32Array::from(errors)))
                .unwrap();
            partition
        })
        .collect();
    let probes: Vec<&Partition> = partitions.iter().collect();
    let stage = stage_one(
        &fixture,
        probes[0],
        &fixture.sign_storage(probes[0]),
        ApproxMode::Normal,
    )
    .unwrap();
    assert!(stage.lower_bounds[3].is_nan());
    assert_eq!(stage.lower_bounds[7], f32::INFINITY);
    for k in [1, 5, 50] {
        for gather in GATHERS {
            for threshold in [Threshold::Live, Threshold::Stale(1), Threshold::FirstFull] {
                assert_lazy_matches_eager(
                    &fixture,
                    &probes,
                    Scan::top_k(k),
                    None,
                    Lazy { threshold, gather },
                );
            }
        }
    }
    let mut survivors = Vec::new();
    let mut counts = StagePruneCounts::default();
    select_full_survivors(
        &stage.lower_bounds,
        None,
        None,
        Some(f32::NAN),
        &mut survivors,
        &mut counts,
    );
    assert_eq!(counts.pruned_heap, 0, "a NaN threshold prunes nothing");
    assert_eq!(
        survivors.len() + counts.pruned_upper_bound,
        stage.lower_bounds.len()
    );
}

/// Survivor selection boundaries match the eager per-row step's categories.
#[test]
fn select_full_survivors_boundaries() {
    let bounds = [
        1.0,
        2.0,
        2.0,
        3.0,
        f32::NAN,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::MAX,
        2.5,
    ];
    let mut survivors = vec![99];
    let mut counts = StagePruneCounts {
        candidates: 7,
        ..Default::default()
    };
    // lb == T is a heap prune, lb == U an upper-bound prune; NaN survives.
    select_full_survivors(
        &bounds,
        None,
        Some(3.0),
        Some(2.0),
        &mut survivors,
        &mut counts,
    );
    assert_eq!(survivors, vec![0, 4, 5]);
    assert_eq!(
        counts,
        StagePruneCounts {
            candidates: 9,
            pruned_upper_bound: 3,
            pruned_heap: 3,
        }
    );
    // Without a query upper bound the eager scan still compares to f32::MAX.
    select_full_survivors(&bounds, None, None, Some(2.0), &mut survivors, &mut counts);
    assert_eq!(survivors, vec![0, 4, 5]);
    assert_eq!(counts.pruned_upper_bound, 2);
    assert_eq!(counts.pruned_heap, 4);
    // Upper-bound pruning takes precedence over the heap threshold.
    select_full_survivors(
        &bounds,
        None,
        Some(2.0),
        Some(1.0),
        &mut survivors,
        &mut counts,
    );
    assert_eq!(survivors, vec![4, 5]);
    assert_eq!(counts.pruned_upper_bound, 6);
    assert_eq!(counts.pruned_heap, 1);
    let accept = [true, false, true, true, false, true, true, true, false];
    select_full_survivors(
        &bounds,
        Some(&accept[..]),
        Some(3.0),
        None,
        &mut survivors,
        &mut counts,
    );
    assert_eq!(survivors, vec![0, 2, 5]);
    assert_eq!(
        counts,
        StagePruneCounts {
            candidates: 6,
            pruned_upper_bound: 3,
            pruned_heap: 0,
        }
    );
    select_full_survivors(&[], None, None, None, &mut survivors, &mut counts);
    assert!(survivors.is_empty());
    assert_eq!(counts, StagePruneCounts::default());
}

/// Stage 1 refuses calculators that cannot gate on the native lower bound,
/// and the sign-only constructors refuse inputs they cannot serve.
#[test]
fn gating_off_reports_reason() {
    let mut rng = SplitMix64(0x6A7E);
    let fixture = Fixture::new(64, 7, DistanceType::L2, &mut rng);
    let partition = fixture.random_partition(40, 0, &mut rng);
    let sign = fixture.sign_storage(&partition);
    assert_eq!(
        stage_one(&fixture, &partition, &sign, ApproxMode::Fast).unwrap_err(),
        "approx_mode_fast"
    );

    let mut residual_metadata = fixture.rq.metadata_ref().clone();
    residual_metadata.query_estimator = super::storage::RabitQueryEstimator::ResidualQuery;
    let residual_sign = RabitQuantizationStorage::try_from_sign_plane_for_full(
        partition.sign.clone(),
        &residual_metadata,
        fixture.distance_type,
    )
    .unwrap();
    let mut scratch = Vec::new();
    let calc = residual_sign.dist_calculator_with_scratch(
        fixture.query.clone(),
        partition.dist_q_c,
        None,
        &mut scratch,
        full_options(ApproxMode::Normal),
    );
    assert_eq!(
        calc.full_sign_stage_with_scratch(
            &mut SignStage::default(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new()
        )
        .unwrap_err(),
        "residual_query_estimator"
    );

    // A native index written without error factors scores every row.
    let mut native_metadata = fixture.rq.metadata_ref().clone();
    native_metadata.layered = false;
    let native_rq = RabitQuantizer::from_metadata(&native_metadata, DistanceType::L2).unwrap();
    let native_fixture = Fixture::with_query(
        RabitQuantizer::try_from(native_rq).unwrap(),
        7,
        DistanceType::L2,
        fixture
            .query
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec(),
    );
    let native_centroid = rng.vector(native_fixture.dim(), 0.2);
    let (native_batch, native_rotated_centroid, native_dist_q_c) = native_fixture.transform(
        &rng.vector(40 * native_fixture.dim(), 1.0),
        &native_centroid,
        0,
    );
    let native_storage = RabitQuantizationStorage::try_from_batch(
        native_batch.drop_column(ERROR_FACTORS_COLUMN).unwrap(),
        &native_metadata,
        DistanceType::L2,
        None,
    )
    .unwrap();
    let mut native_scratch = Vec::new();
    let calc = native_storage.dist_calculator_with_scratch(
        native_fixture.query.clone(),
        native_dist_q_c,
        Some(QueryResidual::RabitRawQuery {
            rotated_centroid: Some(&native_rotated_centroid),
            query: Some(&native_fixture.raw_query),
        }),
        &mut native_scratch,
        full_options(ApproxMode::Normal),
    );
    assert_eq!(
        calc.full_sign_stage_with_scratch(
            &mut SignStage::default(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new()
        )
        .unwrap_err(),
        "missing_error_factors"
    );

    let no_errors = partition.sign.drop_column(ERROR_FACTORS_COLUMN).unwrap();
    let error = RabitQuantizationStorage::try_from_sign_plane_for_full(
        no_errors,
        fixture.rq.metadata_ref(),
        fixture.distance_type,
    )
    .unwrap_err();
    assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
    assert!(error.to_string().contains(ERROR_FACTORS_COLUMN), "{error}");
    let error = RabitQuantizationStorage::try_from_sign_plane_for_full(
        partition.sign.clone(),
        &native_metadata,
        fixture.distance_type,
    )
    .unwrap_err();
    assert!(error.to_string().contains("layered multi-bit"), "{error}");

    // Gathered rows must match the index layout and the scan precision.
    let error = ExRows::from_plane_batches(&partition.hi, &partition.lo, fixture.rotated_dim(), 9)
        .unwrap_err();
    assert!(error.to_string().contains("byte width mismatch"), "{error}");
    let short_lo = partition.lo.slice(0, 10);
    let error =
        ExRows::from_plane_batches(&partition.hi, &short_lo, fixture.rotated_dim(), 7).unwrap_err();
    assert!(error.to_string().contains("disagree in length"), "{error}");
    let ex_rows =
        ExRows::from_plane_batches(&partition.hi, &partition.lo, fixture.rotated_dim(), 7).unwrap();
    for options in [
        full_options(ApproxMode::Fast),
        DistanceCalculatorOptions {
            approx_mode: ApproxMode::Normal,
            rq_precision: RQPrecision::High,
        },
    ] {
        let error = sign
            .dist_calculator_with_ex_rows(
                fixture.query.clone(),
                partition.dist_q_c,
                fixture.residual(&partition),
                &mut Vec::new(),
                options,
                ex_rows,
            )
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("gathered ex rows require"),
            "{error}"
        );
    }
}
