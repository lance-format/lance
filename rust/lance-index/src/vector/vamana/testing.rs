// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared by the Vamana tests.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt64Array,
};
use arrow_schema::{Field, Schema};
use async_trait::async_trait;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{ROW_ID, ROW_ID_FIELD, Result};
use lance_linalg::distance::DistanceType;
use lance_select::RowAddrMask;

use super::graph::{Adjacency, PartitionGraph};
use super::index::Vamana;
use super::params::VamanaBuildParams;
use super::search::{Comparisons, SearchScratch, greedy_search_with_lookahead};
use crate::prefilter::PreFilter;
use crate::vector::SQ_CODE_COLUMN;
use crate::vector::bq::storage::{
    RABIT_CODE_COLUMN, RabitQuantizationMetadata, RabitQuantizationStorage,
};
use crate::vector::bq::transform::{
    ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, SCALE_FACTORS_COLUMN,
};
use crate::vector::flat::index::FlatMetadata;
use crate::vector::flat::storage::{FLAT_COLUMN, FlatFloatStorage};
use crate::vector::graph::{OrderedFloat, OrderedNode};
use crate::vector::pq::ProductQuantizer;
use crate::vector::pq::storage::ProductQuantizationStorage;
use crate::vector::quantizer::{Quantization, QuantizerStorage};
use crate::vector::sq::ScalarQuantizer;
use crate::vector::sq::storage::ScalarQuantizationStorage;
use crate::vector::storage::{DistCalculator, StorageBuilder, VectorStore};

/// recall@10 of walks from `start` through `graph`, one per query of `queries`
/// (row-major, `dimension` wide), against brute force over `storage`, and the
/// distances the walks measured.
pub fn walk_recall_at_ten(
    storage: &impl VectorStore,
    graph: &impl Adjacency,
    start: u32,
    queries: &[f32],
    dimension: usize,
    search_list_size: usize,
) -> (f64, u64) {
    const K: usize = 10;
    let mut scratch = SearchScratch::new(storage.len());
    let walked = Comparisons::default();
    let mut hits = 0;
    let mut expected = 0;
    for query in queries.chunks_exact(dimension) {
        let calculator = storage.dist_calculator(Arc::new(Float32Array::from(query.to_vec())), 0.0);
        let mut exact = (0..storage.len() as u32)
            .map(|id| OrderedNode::new(id, OrderedFloat(calculator.distance(id))))
            .collect::<Vec<_>>();
        exact.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        let exact = exact
            .iter()
            .take(K)
            .map(|node| node.id)
            .collect::<HashSet<_>>();
        let found = greedy_search_with_lookahead(
            graph,
            &calculator,
            start,
            search_list_size,
            0,
            &mut scratch,
            &walked,
        )
        .unwrap();
        hits += found
            .candidates
            .iter()
            .take(K)
            .filter(|node| exact.contains(&node.id))
            .count();
        expected += K;
    }
    (hits as f64 / expected as f64, walked.get())
}

/// An FNV-1a hash of every out-edge list of `graph`, in vertex order.
pub fn graph_fingerprint(graph: &PartitionGraph) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |value: u32| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for vertex in 0..graph.len() as u32 {
        let neighbors = graph.neighbors(vertex);
        eat(neighbors.len() as u32);
        neighbors.iter().for_each(|neighbor| eat(*neighbor));
    }
    hash
}

/// A store of each kind a graph is built over.
pub enum Store {
    Flat(FlatFloatStorage),
    Scalar(ScalarQuantizationStorage),
    Product(ProductQuantizationStorage),
}

/// A fixed multiplicative congruential sequence in `[0, 1)`, so every run sees
/// the same points.
pub fn scattered_values(count: usize) -> Vec<f32> {
    let mut state = 12345u64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f32 / (1u64 << 31) as f32
        })
        .collect()
}

pub fn vectors_of(values: Vec<f32>, dimension: usize) -> FixedSizeListArray {
    FixedSizeListArray::try_new_from_values(Float32Array::from(values), dimension as i32).unwrap()
}

/// A flat store whose row ids are its positions.
pub fn flat_storage(
    values: Vec<f32>,
    dimension: usize,
    distance_type: DistanceType,
) -> FlatFloatStorage {
    FlatFloatStorage::new(vectors_of(values, dimension), distance_type)
}

/// A flat store over `vectors` named by `row_ids`.
pub fn flat_storage_with_row_ids(
    vectors: FixedSizeListArray,
    row_ids: Vec<u64>,
    distance_type: DistanceType,
) -> FlatFloatStorage {
    let dim = vectors.value_length() as usize;
    let batch = RecordBatch::try_from_iter(vec![
        (ROW_ID, Arc::new(UInt64Array::from(row_ids)) as ArrayRef),
        (FLAT_COLUMN, Arc::new(vectors) as ArrayRef),
    ])
    .unwrap();
    FlatFloatStorage::try_from_batch(batch, &FlatMetadata { dim }, distance_type, None).unwrap()
}

/// An 8-bit scalar-quantized store over `vectors` named by `row_ids`.
pub fn sq_storage(
    vectors: &FixedSizeListArray,
    row_ids: Vec<u64>,
    distance_type: DistanceType,
) -> ScalarQuantizationStorage {
    let quantizer = ScalarQuantizer::build(vectors, distance_type, &Default::default()).unwrap();
    let codes = quantizer.quantize(vectors).unwrap();
    let batch = RecordBatch::try_from_iter(vec![
        (ROW_ID, Arc::new(UInt64Array::from(row_ids)) as ArrayRef),
        (SQ_CODE_COLUMN, codes),
    ])
    .unwrap();
    ScalarQuantizationStorage::try_new(8, distance_type, quantizer.bounds(), [batch], None).unwrap()
}

/// A product-quantized `L2` store over `vectors`, named by their positions:
/// one sub-vector to every two dimensions, `num_bits` to each code, and the
/// first `2^num_bits` vectors for centroids, so every run codes alike.
pub fn pq_storage(vectors: &FixedSizeListArray, num_bits: u32) -> ProductQuantizationStorage {
    let dimension = vectors.value_length() as usize;
    let quantizer = ProductQuantizer::new(
        dimension / 2,
        num_bits,
        dimension,
        vectors.slice(0, 1 << num_bits),
        DistanceType::L2,
    );
    let schema = Schema::new(vec![
        Field::new("vec", vectors.data_type().clone(), true),
        ROW_ID_FIELD.clone(),
    ]);
    let row_ids = UInt64Array::from_iter_values(0..vectors.len() as u64);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(vectors.clone()), Arc::new(row_ids)],
    )
    .unwrap();
    StorageBuilder::new("vec".to_owned(), DistanceType::L2, quantizer, None)
        .unwrap()
        .build(vec![batch])
        .unwrap()
}

/// A one-bit RaBitQ store of `num_rows` arbitrary codes over 64 dimensions:
/// enough to be refused, not to be searched.
pub fn rq_storage(num_rows: usize) -> RabitQuantizationStorage {
    const DIMENSION: usize = 64;
    let mut metadata: RabitQuantizationMetadata =
        serde_json::from_str(r#"{"num_bits":1,"packed":true}"#).unwrap();
    metadata.rotate_mat = Some(
        FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0; DIMENSION * DIMENSION]),
            DIMENSION as i32,
        )
        .unwrap(),
    );
    let code_bytes = metadata.binary_code_bytes();
    let codes = FixedSizeListArray::try_new_from_values(
        UInt8Array::from(vec![0u8; num_rows * code_bytes]),
        code_bytes as i32,
    )
    .unwrap();
    let factors = |offset: f32| {
        Arc::new(Float32Array::from_iter_values(
            (0..num_rows).map(|row| row as f32 + offset),
        )) as ArrayRef
    };
    let batch = RecordBatch::try_from_iter(vec![
        (
            ROW_ID,
            Arc::new(UInt64Array::from_iter_values(0..num_rows as u64)) as ArrayRef,
        ),
        (RABIT_CODE_COLUMN, Arc::new(codes) as ArrayRef),
        (ADD_FACTORS_COLUMN, factors(0.0)),
        (SCALE_FACTORS_COLUMN, factors(0.5)),
        (ERROR_FACTORS_COLUMN, factors(0.25)),
    ])
    .unwrap();
    RabitQuantizationStorage::try_from_batch(batch, &metadata, DistanceType::L2, None).unwrap()
}

/// Ten vertices on a line named `10 * position + 3`, each pointing at its
/// neighbors on it, except that vertex 0 points at 2 before 1: an order no
/// prune writes, so a list carried over can be told from a rebuilt one.
///
/// With 4 and 5 gone, 3 and 6 each inherit only the other dead vertex, so
/// 6..=9 are cut off from the rest.
pub fn torn_line() -> (Vamana, FixedSizeListArray) {
    let adjacency = (0..10u32)
        .map(|position| match position {
            0 => vec![2, 1],
            _ => [position.checked_sub(1), Some(position + 1)]
                .into_iter()
                .flatten()
                .filter(|neighbor| *neighbor < 10)
                .collect(),
        })
        .collect::<Vec<_>>();
    let params = VamanaBuildParams {
        max_degree: 4,
        search_list_size: 8,
        alpha: 1.2,
    };
    let graph = PartitionGraph::try_new(
        params.max_degree,
        (0..10).map(|position| 10 * position + 3).collect(),
        &adjacency,
    )
    .unwrap();
    (
        Vamana::from_graph(&graph, 4, vec![], params).unwrap(),
        vectors_of((0..10).map(|i| i as f32).collect(), 1),
    )
}

/// Eight vertices on a line at one out-edge each, named `10 * position + 3`.
/// The medoid, 5, leads through 6 and 7 to 0, which points into the ring
/// 1 -> 2 -> 3 -> 4 -> 1 of the other survivors. Vertex 3 points at 4 though 2
/// is nearer: a choice no prune makes, so a graph carried over can be told
/// from a rebuilt one.
///
/// With 5, 6 and 7 gone, nothing points at 0. Inserted again, 0 points at 1,
/// whose one edge, to 2, is nearer than 0: 0 stays cut off.
pub fn cut_off_after_relink() -> (Vamana, FixedSizeListArray) {
    let params = VamanaBuildParams {
        max_degree: 1,
        search_list_size: 8,
        alpha: 1.2,
    };
    let graph = PartitionGraph::try_new(
        params.max_degree,
        (0..8).map(|position| 10 * position + 3).collect(),
        &[
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![1],
            vec![6],
            vec![7],
            vec![0],
        ],
    )
    .unwrap();
    (
        Vamana::from_graph(&graph, 5, vec![], params).unwrap(),
        vectors_of(vec![0.0, 2.0, 3.0, 10.0, 30.0, 50.0, 40.0, 1.0], 1),
    )
}

/// A prefilter over a fixed mask, ready from the start.
pub struct MaskFilter(pub Arc<RowAddrMask>);

#[async_trait]
impl PreFilter for MaskFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.0.clone()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.0.selected_indices(row_ids)
    }
}

/// The `k` rows nearest `query` by `storage`'s distance among those `accept`
/// passes, as `(row id, distance)`, nearest first.
pub fn exact_nearest(
    storage: &impl VectorStore,
    query: ArrayRef,
    k: usize,
    accept: impl Fn(u64, f32) -> bool,
) -> Vec<(u64, f32)> {
    let calculator = storage.dist_calculator(query, 0.0);
    let mut all = (0..storage.len() as u32)
        .map(|position| (storage.row_id(position), calculator.distance(position)))
        .filter(|(row_id, distance)| accept(*row_id, *distance))
        .collect::<Vec<_>>();
    all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

/// The share of `expected` that `found` holds.
pub fn recall(found: &[u64], expected: &[(u64, f32)]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let found = found.iter().collect::<std::collections::HashSet<_>>();
    expected
        .iter()
        .filter(|(row_id, _)| found.contains(row_id))
        .count() as f64
        / expected.len() as f64
}
