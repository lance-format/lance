// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{ArrayRef, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use itertools::Itertools;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;

use super::build::build_partition;
use super::entry_points::train_entry_points;
use super::graph::{Adjacency, LoadedGraph, PartitionGraph, neighbor_item_field};
use super::incremental::{self, build_incremental};
use super::params::{VAMANA_FORMAT_VERSION, VamanaBuildParams, VamanaMetadata, VamanaQueryParams};
use super::search::{Comparisons, StopRule};
use super::walk::{Admission, WalkParams, Walked, walk, with_scratch};
use super::{NEIGHBORS_COL, ROW_ID_COL, VAMANA_METADATA_KEY, VAMANA_SEED, VAMANA_TYPE, remap};
use crate::metrics::MetricsCollector;
use crate::prefilter::PreFilter;
use crate::vector::VECTOR_RESULT_SCHEMA;
use crate::vector::bq::storage::RabitQuantizationStorage;
use crate::vector::graph::{OrderedFloat, OrderedNode};
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::v3::subindex::IvfSubIndex;

/// Below this share of a partition's rows passing the prefilter, the passing
/// rows are scored directly rather than walked to, as HNSW does.
const FLAT_FILTER_PERCENT: usize = 10;

/// How much a re-walk may widen its list when a filtered walk came back short.
const MAX_FILTER_WIDENING: usize = 10;

/// One IVF partition's Vamana graph: vertex `i` is position `i` of the
/// partition's storage.
#[derive(Debug, Clone)]
pub struct Vamana {
    metadata: VamanaMetadata,
    graph: LoadedGraph,
    /// What [`IvfSubIndex::to_batch`] returns, and what `graph` views.
    batch: RecordBatch,
}

/// The candidates a search kept, by storage position, nearest first.
#[derive(Debug, Default)]
pub struct Searched {
    pub nodes: Vec<OrderedNode>,
    pub comparisons: u64,
}

impl Vamana {
    /// A graph over no vertices: what an empty partition holds.
    pub fn empty(params: VamanaBuildParams) -> Self {
        Self {
            metadata: VamanaMetadata {
                format_version: VAMANA_FORMAT_VERSION,
                params,
                medoid: 0,
                entry_points: Vec::new(),
                num_vertices: 0,
            },
            graph: LoadedGraph::empty(),
            batch: RecordBatch::new_empty(Self::schema()),
        }
    }

    pub fn from_graph(
        graph: &PartitionGraph,
        medoid: u32,
        entry_points: Vec<u32>,
        params: VamanaBuildParams,
    ) -> Result<Self> {
        if graph.is_empty() {
            return Ok(Self::empty(params));
        }
        let metadata = VamanaMetadata {
            format_version: VAMANA_FORMAT_VERSION,
            params,
            medoid,
            entry_points,
            num_vertices: graph.len() as u32,
        };
        let schema = Schema::new(Self::schema().fields().clone()).with_metadata(HashMap::from([(
            VAMANA_METADATA_KEY.to_string(),
            serde_json::to_string(&metadata)?,
        )]));
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(UInt64Array::from(graph.row_ids().to_vec())),
                Arc::new(graph.to_list_array()?),
            ],
        )?;
        Self::try_from_batch(batch, metadata)
    }

    fn try_from_batch(batch: RecordBatch, metadata: VamanaMetadata) -> Result<Self> {
        let num_vertices = batch.num_rows();
        if metadata.format_version != VAMANA_FORMAT_VERSION {
            return Err(Error::index(format!(
                "Vamana partition format version {} is not supported, this build reads {}",
                metadata.format_version, VAMANA_FORMAT_VERSION
            )));
        }
        if metadata.num_vertices as usize != num_vertices {
            return Err(Error::index(format!(
                "Vamana metadata records {} vertices but the batch holds {num_vertices}",
                metadata.num_vertices
            )));
        }
        if metadata.medoid as usize >= num_vertices
            || metadata
                .entry_points
                .iter()
                .any(|entry| *entry as usize >= num_vertices)
        {
            return Err(Error::index(format!(
                "Vamana medoid {} or an entry point is outside a graph of {num_vertices} vertices",
                metadata.medoid
            )));
        }
        if metadata
            .entry_points
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::index(format!(
                "Vamana entry points must ascend without repeats, got {:?}",
                metadata.entry_points
            )));
        }
        metadata.params.validate()?;
        let row_ids = batch
            .column_by_name(ROW_ID_COL)
            .and_then(|column| column.as_primitive_opt::<UInt64Type>())
            .ok_or_else(|| Error::index(format!("Vamana batch lacks a UInt64 {ROW_ID_COL}")))?;
        let neighbors = batch
            .column_by_name(NEIGHBORS_COL)
            .and_then(|column| column.as_list_opt::<i32>())
            .ok_or_else(|| Error::index(format!("Vamana batch lacks a list {NEIGHBORS_COL}")))?;
        let graph = LoadedGraph::try_new(row_ids, neighbors, metadata.params.max_degree)?;
        Ok(Self {
            metadata,
            graph,
            batch,
        })
    }

    pub fn len(&self) -> usize {
        self.graph.num_vertices()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn metadata(&self) -> &VamanaMetadata {
        &self.metadata
    }

    pub fn params(&self) -> &VamanaBuildParams {
        &self.metadata.params
    }

    pub fn graph(&self) -> &LoadedGraph {
        &self.graph
    }

    /// Walk this partition's graph towards the query `calculator` measures.
    pub fn walk(
        &self,
        calculator: &impl DistCalculator,
        params: &WalkParams,
        admission: Option<Admission<'_>>,
    ) -> Result<Walked> {
        with_scratch(|scratch| {
            walk(
                &self.graph,
                calculator,
                &self.metadata.entry_points,
                self.metadata.medoid,
                params,
                admission,
                scratch,
            )
        })
    }

    /// Up to `keep` vertices nearest `query` by the storage's distance that
    /// pass `prefilter` and lie within the query's bounds, nearest first.
    ///
    /// A filtered walk steers by every vertex and keeps the admitted ones on
    /// the side, so a filter that only removes deleted rows costs no more than
    /// an unfiltered walk. When fewer than half of the walk's own list pass,
    /// its neighborhood holds too few admitted rows to answer from, as HNSW's
    /// filtered walk would find by walking further: rows that pass a mask
    /// rarely are then scored directly, and otherwise the walk is repeated
    /// with its list widened by the share that does not pass. A mask that
    /// passes rows only far from the query can leave even the wider walk
    /// short of them, and its passing rows are then scored directly too. A
    /// walk whose list reached past the upper bound has already measured
    /// every row in range it could, and is neither repeated nor followed by
    /// scoring. `stop` applies to an unfiltered walk only.
    pub fn search_nodes(
        &self,
        query: ArrayRef,
        keep: usize,
        params: &VamanaQueryParams,
        storage: &impl VectorStore,
        prefilter: &dyn PreFilter,
        stop: Option<StopRule>,
    ) -> Result<Searched> {
        if keep == 0 {
            return Ok(Searched::default());
        }
        let lower = OrderedFloat(params.lower_bound.unwrap_or(f32::MIN));
        let upper = OrderedFloat(params.upper_bound.unwrap_or(f32::MAX));
        let bounded = params.lower_bound.is_some() || params.upper_bound.is_some();
        let in_bounds = |distance: f32| {
            let distance = OrderedFloat(distance);
            distance >= lower && distance < upper
        };

        // A store that does not match the graph, such as one remapped through
        // a fragment-reuse index while the graph was not, is scored directly.
        let num_vertices = self.len();
        if storage.len() != num_vertices {
            let passing = if prefilter.is_empty() {
                (0..storage.len() as u64).collect()
            } else {
                prefilter.filter_row_ids(Box::new(storage.row_ids()))
            };
            return Ok(score(storage, query, keep, params, &passing, in_bounds));
        }
        if num_vertices == 0 {
            return Ok(Searched::default());
        }

        let calculator = storage.dist_calculator(query.clone(), params.dist_q_c);
        let mut walk_params = WalkParams::new(params.ef.max(keep));
        if prefilter.is_empty() && !bounded {
            walk_params.stop = stop;
            let walked = self.walk(&calculator, &walk_params, None)?;
            let mut nodes = walked.list;
            nodes.truncate(keep);
            return Ok(Searched {
                nodes,
                comparisons: walked.comparisons,
            });
        }

        let mask = (!prefilter.is_empty()).then(|| prefilter.mask());
        let mut accept = |position: u32, distance: f32| {
            in_bounds(distance)
                && mask
                    .as_ref()
                    .is_none_or(|mask| mask.selected(storage.row_id(position)))
        };
        let walked = self.walk(
            &calculator,
            &walk_params,
            Some(Admission {
                keep,
                accept: &mut accept,
            }),
        )?;
        let passing_in_list = walked
            .list
            .iter()
            .filter(|node| accept(node.id, node.dist.0))
            .count();
        // A list whose back reached the upper bound holds every vertex below
        // it that the walk measured, so a wider walk would find no more.
        let reaches_upper =
            |list: &[OrderedNode]| list.last().is_some_and(|last| last.dist >= upper);
        if walked.admitted.len() >= keep && passing_in_list * 2 >= walked.list.len()
            || reaches_upper(&walked.list)
        {
            return Ok(Searched {
                nodes: walked.admitted,
                comparisons: walked.comparisons,
            });
        }

        let search_list_size = walk_params.search_list_size;
        let (widened, passing) = if mask.is_some() {
            let passing = prefilter.filter_row_ids(Box::new(storage.row_ids()));
            if passing.len() * 100 < num_vertices * FLAT_FILTER_PERCENT
                || passing.len() <= search_list_size
            {
                let mut scored = score(storage, query, keep, params, &passing, in_bounds);
                scored.comparisons += walked.comparisons;
                return Ok(scored);
            }
            (
                search_list_size.saturating_mul(num_vertices) / passing.len(),
                Some(passing),
            )
        } else {
            (
                search_list_size.saturating_mul(walked.list.len()) / passing_in_list.max(1),
                None,
            )
        };
        let widened = widened
            .min(search_list_size.saturating_mul(MAX_FILTER_WIDENING))
            .min(num_vertices);
        if widened <= search_list_size {
            return Ok(Searched {
                nodes: walked.admitted,
                comparisons: walked.comparisons,
            });
        }
        walk_params.search_list_size = widened;
        let rewalked = self.walk(
            &calculator,
            &walk_params,
            Some(Admission {
                keep,
                accept: &mut accept,
            }),
        )?;
        let comparisons = walked.comparisons + rewalked.comparisons;
        if rewalked.admitted.len() < keep
            && !reaches_upper(&rewalked.list)
            && let Some(passing) = passing
        {
            let mut scored = score(storage, query, keep, params, &passing, in_bounds);
            scored.comparisons += comparisons;
            return Ok(scored);
        }
        Ok(Searched {
            nodes: rewalked.admitted,
            comparisons,
        })
    }
}

/// Refuse a store a graph cannot be built over.
///
/// Under dot the distance `1 - dot` goes below zero, where an `alpha` above 1
/// tightens the prune instead of relaxing it. A RaBitQ store cannot measure
/// from one stored vector to another, which every prune does. Hamming is
/// refused with dot: only L2 and cosine have been built and measured.
pub(crate) fn check_store(store: &impl VectorStore) -> Result<()> {
    let distance_type = store.distance_type();
    if !matches!(distance_type, DistanceType::L2 | DistanceType::Cosine) {
        return Err(Error::not_supported(format!(
            "Vamana supports the l2 and cosine distances, not {distance_type}"
        )));
    }
    if store.as_any().is::<RabitQuantizationStorage>() {
        return Err(Error::not_supported(
            "Vamana cannot be built over RaBitQ codes, which cannot measure the distance \
             between two stored vectors"
                .to_string(),
        ));
    }
    Ok(())
}

/// The nearest `keep` of the storage positions `passing` within the bounds,
/// scored one by one.
fn score(
    storage: &impl VectorStore,
    query: ArrayRef,
    keep: usize,
    params: &VamanaQueryParams,
    passing: &[u64],
    in_bounds: impl Fn(f32) -> bool,
) -> Searched {
    let calculator = storage.dist_calculator(query, params.dist_q_c);
    let mut heap = BinaryHeap::<OrderedNode>::with_capacity(keep + 1);
    for position in passing {
        let distance = calculator.distance(*position as u32);
        if !in_bounds(distance) {
            continue;
        }
        let node = OrderedNode::new(*position as u32, OrderedFloat(distance));
        if heap.len() < keep {
            heap.push(node);
        } else if heap
            .peek()
            .is_some_and(|farthest| node.dist < farthest.dist)
        {
            heap.pop();
            heap.push(node);
        }
    }
    Searched {
        nodes: heap.into_sorted_vec(),
        comparisons: passing.len() as u64,
    }
}

/// The walk scratch is not counted: it belongs to the thread that walks, not
/// to a partition, and keeps a byte per vertex of the largest partition that
/// thread has walked.
impl DeepSizeOf for Vamana {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.batch.get_array_memory_size()
            + self.metadata.entry_points.capacity() * std::mem::size_of::<u32>()
    }
}

impl IvfSubIndex for Vamana {
    type QueryParams = VamanaQueryParams;
    type BuildParams = VamanaBuildParams;

    fn load(data: RecordBatch) -> Result<Self> {
        if data.num_rows() == 0 {
            return Ok(Self::empty(VamanaBuildParams::default()));
        }
        let metadata = data
            .schema_ref()
            .metadata()
            .get(VAMANA_METADATA_KEY)
            .ok_or_else(|| Error::index(format!("{VAMANA_METADATA_KEY} not found")))?;
        let metadata: VamanaMetadata = serde_json::from_str(metadata).map_err(|e| {
            Error::index(format!(
                "Failed to decode Vamana metadata: {e}, json: {metadata}"
            ))
        })?;
        Self::try_from_batch(data, metadata)
    }

    fn name() -> &'static str {
        VAMANA_TYPE
    }

    fn metadata_key() -> &'static str {
        VAMANA_METADATA_KEY
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(ROW_ID_COL, DataType::UInt64, false),
            Field::new(NEIGHBORS_COL, DataType::List(neighbor_item_field()), false),
        ]))
    }

    fn search(
        &self,
        query: ArrayRef,
        k: usize,
        params: Self::QueryParams,
        storage: &impl VectorStore,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        if params.ef < k {
            return Err(Error::index(
                "ef must be greater than or equal to k".to_string(),
            ));
        }
        let searched = self.search_nodes(query, k, &params, storage, prefilter.as_ref(), None)?;
        metrics.record_comparisons(searched.comparisons as usize);
        let (row_ids, distances): (Vec<_>, Vec<_>) = searched
            .nodes
            .into_iter()
            .map(|node| (storage.row_id(node.id), node.dist.0))
            .unique_by(|(row_id, _)| *row_id)
            .unzip();
        Ok(RecordBatch::try_new(
            VECTOR_RESULT_SCHEMA.clone(),
            vec![
                Arc::new(Float32Array::from(distances)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )?)
    }

    fn index_vectors(storage: &impl VectorStore, params: Self::BuildParams) -> Result<Self> {
        params.validate()?;
        check_store(storage)?;
        if storage.is_empty() {
            return Ok(Self::empty(params));
        }
        let built = build_partition(storage, &params, VAMANA_SEED, &Comparisons::default())?;
        let entry_points = train_entry_points(storage, VAMANA_SEED)?;
        Self::from_graph(&built.graph, built.medoid, entry_points, params)
    }

    fn may_extend(vertices: usize, matched: usize, num_rows: usize) -> bool {
        incremental::may_extend(vertices, matched, num_rows)
    }

    fn index_vectors_incremental(
        base: &Self,
        storage: &impl VectorStore,
        base_rows: usize,
        params: Self::BuildParams,
    ) -> Result<Self> {
        build_incremental(base, storage, base_rows, params)
    }

    fn remap(&self, mapping: &RowAddrRemap, store: &impl VectorStore) -> Result<Self> {
        remap::remap(self, mapping, store)
    }

    fn to_batch(&self) -> Result<RecordBatch> {
        Ok(self.batch.clone())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::FixedSizeListArray;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_select::{RowAddrMask, RowAddrTreeMap};
    use rstest::rstest;

    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::prefilter::NoFilter;
    use crate::vector::flat::storage::{FlatBinStorage, FlatFloatStorage};
    use crate::vector::vamana::incremental::build_incremental;
    use crate::vector::vamana::testing::{
        MaskFilter, exact_nearest, flat_storage_with_row_ids, recall, rq_storage, scattered_values,
        sq_storage, vectors_of,
    };

    const DIMENSION: usize = 8;

    fn params() -> VamanaBuildParams {
        VamanaBuildParams {
            max_degree: 16,
            search_list_size: 40,
            alpha: 1.2,
        }
    }

    /// `num_rows` scattered points named `10 * position + 3`, and `queries`
    /// more from the same distribution.
    fn fixture(num_rows: usize, queries: usize) -> (FixedSizeListArray, Vec<u64>, Vec<ArrayRef>) {
        let values = scattered_values((num_rows + queries) * DIMENSION)
            .into_iter()
            .map(|value| value - 0.5)
            .collect::<Vec<_>>();
        let (indexed, rest) = values.split_at(num_rows * DIMENSION);
        let queries = rest
            .as_chunks::<DIMENSION>()
            .0
            .iter()
            .map(|query| Arc::new(Float32Array::from(query.to_vec())) as ArrayRef)
            .collect();
        (
            vectors_of(indexed.to_vec(), DIMENSION),
            (0..num_rows as u64)
                .map(|position| 10 * position + 3)
                .collect(),
            queries,
        )
    }

    fn flat_fixture(num_rows: usize, queries: usize) -> (FlatFloatStorage, Vec<ArrayRef>) {
        let (vectors, row_ids, queries) = fixture(num_rows, queries);
        (
            flat_storage_with_row_ids(vectors, row_ids, DistanceType::L2),
            queries,
        )
    }

    fn query_params(ef: usize) -> VamanaQueryParams {
        VamanaQueryParams {
            ef,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
        }
    }

    fn search(
        index: &Vamana,
        storage: &impl VectorStore,
        query: &ArrayRef,
        k: usize,
        params: VamanaQueryParams,
        prefilter: Arc<dyn PreFilter>,
    ) -> Vec<(u64, f32)> {
        let batch = index
            .search(
                query.clone(),
                k,
                params,
                storage,
                prefilter,
                &NoOpMetricsCollector,
            )
            .unwrap();
        let distances =
            batch[crate::vector::DIST_COL].as_primitive::<arrow_array::types::Float32Type>();
        let row_ids = batch[lance_core::ROW_ID].as_primitive::<UInt64Type>();
        row_ids
            .values()
            .iter()
            .zip(distances.values())
            .map(|(row_id, distance)| (*row_id, *distance))
            .collect()
    }

    #[test]
    fn a_built_index_round_trips_through_its_batch() {
        let (storage, queries) = flat_fixture(500, 5);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let batch = index.to_batch().unwrap();
        assert_eq!(batch.num_rows(), 500);
        assert_eq!(batch.schema().fields(), Vamana::schema().fields());
        let metadata: VamanaMetadata =
            serde_json::from_str(&batch.schema().metadata()[VAMANA_METADATA_KEY]).unwrap();
        assert_eq!(&metadata, index.metadata());
        assert!(!metadata.entry_points.is_empty());

        // A batch read from an index file carries the file's own keys too.
        let mut file_metadata = batch.schema().metadata().clone();
        file_metadata.insert("lance:index".to_string(), "{}".to_string());
        let from_file = batch
            .clone()
            .with_schema(Arc::new(
                batch.schema().as_ref().clone().with_metadata(file_metadata),
            ))
            .unwrap();
        let loaded = Vamana::load(from_file).unwrap();
        assert_eq!(loaded.metadata(), index.metadata());
        assert_eq!(
            loaded.graph().row_ids(),
            storage.row_ids().copied().collect::<Vec<_>>()
        );
        let reloaded = Vamana::load(loaded.to_batch().unwrap()).unwrap();
        assert_eq!(
            reloaded.to_batch().unwrap().schema().metadata()["lance:index"],
            "{}"
        );
        for query in &queries {
            assert_eq!(
                search(
                    &index,
                    &storage,
                    query,
                    10,
                    query_params(20),
                    Arc::new(NoFilter)
                ),
                search(
                    &reloaded,
                    &storage,
                    query,
                    10,
                    query_params(20),
                    Arc::new(NoFilter)
                ),
            );
        }
    }

    #[test]
    fn an_empty_partition_writes_and_reads_nothing() {
        let (storage, queries) = flat_fixture(0, 1);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        assert!(index.is_empty());
        let batch = index.to_batch().unwrap();
        assert_eq!(batch.num_rows(), 0);
        // An empty partition's metadata is written as an empty string.
        let loaded = Vamana::load(batch).unwrap();
        assert!(loaded.is_empty());
        assert!(
            search(
                &loaded,
                &storage,
                &queries[0],
                10,
                query_params(20),
                Arc::new(NoFilter)
            )
            .is_empty()
        );
    }

    #[test]
    fn malformed_batches_are_rejected() {
        let (storage, _) = flat_fixture(50, 0);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let batch = index.to_batch().unwrap();
        let with_metadata = |edit: &dyn Fn(&mut VamanaMetadata)| {
            let mut metadata = index.metadata().clone();
            edit(&mut metadata);
            let schema = batch
                .schema()
                .as_ref()
                .clone()
                .with_metadata(HashMap::from([(
                    VAMANA_METADATA_KEY.to_string(),
                    serde_json::to_string(&metadata).unwrap(),
                )]));
            RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap()
        };
        let cases: Vec<(RecordBatch, &str)> = vec![
            (
                RecordBatch::try_new(Vamana::schema(), batch.columns().to_vec()).unwrap(),
                "not found",
            ),
            (
                with_metadata(&|m| m.num_vertices = 49),
                "records 49 vertices",
            ),
            (with_metadata(&|m| m.medoid = 50), "outside a graph of 50"),
            (
                with_metadata(&|m| m.entry_points = vec![3, 60]),
                "outside a graph of 50",
            ),
            (
                with_metadata(&|m| m.entry_points = vec![5, 3]),
                "ascend without repeats",
            ),
            (
                with_metadata(&|m| m.entry_points = vec![3, 3]),
                "ascend without repeats",
            ),
            (with_metadata(&|m| m.format_version = 2), "format version 2"),
            (
                with_metadata(&|m| m.params.max_degree = 2),
                "more than max_degree 2",
            ),
        ];
        for (malformed, message) in cases {
            let error = Vamana::load(malformed).unwrap_err();
            assert!(matches!(error, Error::Index { .. }), "{error}");
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    #[test]
    fn search_finds_the_nearest_rows_at_their_exact_distances() {
        let (storage, queries) = flat_fixture(1000, 50);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let mut total = 0.0;
        for query in &queries {
            let expected = exact_nearest(&storage, query.clone(), 10, |_, _| true);
            let found = search(
                &index,
                &storage,
                query,
                10,
                query_params(40),
                Arc::new(NoFilter),
            );
            assert_eq!(found.len(), 10);
            assert!(found.windows(2).all(|pair| pair[0].1 <= pair[1].1));
            let calculator = storage.dist_calculator(query.clone(), 0.0);
            for (row_id, distance) in &found {
                assert_eq!(*distance, calculator.distance(((row_id - 3) / 10) as u32));
            }
            total += recall(
                &found.iter().map(|(row_id, _)| *row_id).collect::<Vec<_>>(),
                &expected,
            );
        }
        let recall = total / queries.len() as f64;
        assert!(recall >= 0.95, "recall {recall}");
    }

    /// Built and walked over 8-bit codes, judged against the float vectors.
    #[test]
    fn search_over_scalar_codes_finds_the_nearest_rows() {
        let (vectors, row_ids, queries) = fixture(1000, 50);
        let flat = flat_storage_with_row_ids(vectors.clone(), row_ids.clone(), DistanceType::L2);
        let sq = sq_storage(&vectors, row_ids, DistanceType::L2);
        let index = Vamana::index_vectors(&sq, params()).unwrap();
        let mut total = 0.0;
        for query in &queries {
            let expected = exact_nearest(&flat, query.clone(), 10, |_, _| true);
            let found = search(&index, &sq, query, 10, query_params(40), Arc::new(NoFilter));
            total += recall(
                &found.iter().map(|(row_id, _)| *row_id).collect::<Vec<_>>(),
                &expected,
            );
        }
        let recall = total / queries.len() as f64;
        assert!(recall >= 0.9, "recall {recall}");
    }

    #[test]
    fn ef_below_k_is_an_error() {
        let (storage, queries) = flat_fixture(50, 1);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let error = index
            .search(
                queries[0].clone(),
                10,
                query_params(9),
                &storage,
                Arc::new(NoFilter),
                &NoOpMetricsCollector,
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("ef must be greater than or equal to k"),
            "{error}"
        );
    }

    fn mask_filter(mask: RowAddrMask) -> Arc<dyn PreFilter> {
        Arc::new(MaskFilter(Arc::new(mask)))
    }

    /// A prefilter that blocks deleted rows walks; one that passes few rows
    /// scores them directly; one that passes many but leaves the walk short
    /// widens it. Each must return exactly the passing rows, nearest first.
    #[rstest]
    #[case::blocks_a_third(3, false)]
    #[case::passes_one_in_fifty(50, true)]
    #[case::passes_one_in_eight(8, true)]
    fn a_prefiltered_search_returns_only_passing_rows(#[case] every: u64, #[case] allow: bool) {
        let (storage, queries) = flat_fixture(1000, 20);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let chosen = move |row_id: u64| ((row_id - 3) / 10).is_multiple_of(every);
        let listed = storage
            .row_ids()
            .copied()
            .filter(|row_id| chosen(*row_id))
            .collect::<RowAddrTreeMap>();
        let (mask, passes): (RowAddrMask, Box<dyn Fn(u64) -> bool>) = if allow {
            (RowAddrMask::from_allowed(listed), Box::new(chosen))
        } else {
            (
                RowAddrMask::from_block(listed),
                Box::new(move |row_id| !chosen(row_id)),
            )
        };
        let prefilter = mask_filter(mask);
        let mut total = 0.0;
        for query in &queries {
            let expected = exact_nearest(&storage, query.clone(), 10, |row_id, _| passes(row_id));
            let found = search(
                &index,
                &storage,
                query,
                10,
                query_params(20),
                prefilter.clone(),
            );
            assert_eq!(found.len(), 10);
            assert!(found.iter().all(|(row_id, _)| passes(*row_id)), "{found:?}");
            total += recall(
                &found.iter().map(|(row_id, _)| *row_id).collect::<Vec<_>>(),
                &expected,
            );
        }
        let recall = total / queries.len() as f64;
        // Rows that pass rarely are scored directly, which is exact.
        let bar = if every >= 50 { 1.0 } else { 0.9 };
        assert!(recall >= bar, "recall {recall}");
    }

    /// A filter passing a third of the rows, spread evenly, leaves most of a
    /// narrow walk's own list blocked for many queries. Those walk again,
    /// wider, rather than answer from the few rows the first walk admitted,
    /// and the second walk costs fewer distances than scoring every passing
    /// row would.
    #[test]
    fn a_filter_blocking_most_of_the_list_widens_the_walk() {
        let (storage, queries) = flat_fixture(1000, 20);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let chosen = |row_id: u64| ((row_id - 3) / 10).is_multiple_of(3);
        let listed = storage
            .row_ids()
            .copied()
            .filter(|row_id| chosen(*row_id))
            .collect::<RowAddrTreeMap>();
        let passing = storage.row_ids().filter(|row_id| chosen(**row_id)).count() as u64;
        let prefilter = mask_filter(RowAddrMask::from_allowed(listed));
        let (mut walked_again, mut total) = (0, 0.0);
        for query in &queries {
            let search_nodes = |prefilter: &dyn PreFilter| {
                index
                    .search_nodes(
                        query.clone(),
                        10,
                        &query_params(10),
                        &storage,
                        prefilter,
                        None,
                    )
                    .unwrap()
            };
            let unfiltered = search_nodes(&NoFilter);
            let filtered = search_nodes(prefilter.as_ref());
            let second_walk = filtered.comparisons - unfiltered.comparisons;
            assert!(second_walk < passing, "{second_walk} more distances");
            // Fewer than half of the first list passing always widens the walk.
            let passing_in_list = unfiltered
                .nodes
                .iter()
                .filter(|node| chosen(storage.row_id(node.id)))
                .count();
            if passing_in_list * 2 < unfiltered.nodes.len() {
                assert!(second_walk > 0, "{passing_in_list} of the list passing");
            }
            walked_again += usize::from(second_walk > 0);
            let found = filtered
                .nodes
                .iter()
                .map(|node| storage.row_id(node.id))
                .collect::<Vec<_>>();
            assert!(found.iter().all(|row_id| chosen(*row_id)), "{found:?}");
            total += recall(
                &found,
                &exact_nearest(&storage, query.clone(), 10, |row_id, _| chosen(row_id)),
            );
        }
        assert!(
            walked_again * 4 >= queries.len(),
            "{walked_again} of {} queries walked again",
            queries.len()
        );
        let recall = total / queries.len() as f64;
        assert!(recall >= 0.9, "recall {recall}");
    }

    /// Rows passing only far from the query are out of even a widened walk's
    /// reach; they are scored directly, which is exact.
    #[test]
    fn a_filter_passing_only_far_rows_is_answered_exactly() {
        let line = vectors_of((0..400).map(|i| i as f32).collect(), 1);
        let row_ids = (0..400)
            .map(|position| 10 * position + 3)
            .collect::<Vec<u64>>();
        let storage = flat_storage_with_row_ids(line, row_ids.clone(), DistanceType::L2);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let far = row_ids[240..].iter().copied().collect::<RowAddrTreeMap>();
        let prefilter = mask_filter(RowAddrMask::from_allowed(far));
        let query = Arc::new(Float32Array::from(vec![0.0])) as ArrayRef;
        let expected = exact_nearest(&storage, query.clone(), 10, |row_id, _| {
            row_id >= row_ids[240]
        });
        assert_eq!(expected[0].0, row_ids[240]);
        let found = search(&index, &storage, &query, 10, query_params(20), prefilter);
        assert_eq!(found, expected);
    }

    /// An upper bound the walk's list reaches past leaves nothing for a wider
    /// walk to find, with a mask or without: the walk is neither repeated nor
    /// followed by scoring the passing rows directly.
    #[test]
    fn an_upper_bound_the_list_reaches_past_costs_no_second_walk() {
        let (storage, queries) = flat_fixture(1000, 10);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let is_even = |row_id: u64| ((row_id - 3) / 10).is_multiple_of(2);
        // Half the rows pass: too many to score directly, too few to answer
        // the bounded walk with `keep` rows.
        let evens = mask_filter(RowAddrMask::from_allowed(
            storage
                .row_ids()
                .copied()
                .filter(|row_id| is_even(*row_id))
                .collect::<RowAddrTreeMap>(),
        ));
        for query in &queries {
            let search_nodes = |params: &VamanaQueryParams, prefilter: &dyn PreFilter| {
                index
                    .search_nodes(query.clone(), 20, params, &storage, prefilter, None)
                    .unwrap()
            };
            let unbounded = search_nodes(&query_params(40), &NoFilter);
            let bounded_params = VamanaQueryParams {
                upper_bound: Some(unbounded.nodes[15].dist.0),
                ..query_params(40)
            };
            let bounded = search_nodes(&bounded_params, &NoFilter);
            assert_eq!(bounded.comparisons, unbounded.comparisons);
            assert_eq!(bounded.nodes, unbounded.nodes[..15]);

            let masked = search_nodes(&bounded_params, evens.as_ref());
            assert_eq!(masked.comparisons, unbounded.comparisons);
            let expected = unbounded.nodes[..15]
                .iter()
                .filter(|node| is_even(storage.row_id(node.id)))
                .cloned()
                .collect::<Vec<_>>();
            assert_eq!(masked.nodes, expected);
        }
    }

    /// A bounded walk whose list stops short of the upper bound, with fewer
    /// rows admitted than asked for, walks again wider. The wider list reaches
    /// past the bound, so it holds every row below it that the walk measured,
    /// and the passing rows are not scored directly.
    #[test]
    fn a_bounded_walk_short_of_its_bound_walks_again_and_scores_nothing() {
        let (storage, queries) = flat_fixture(1000, 10);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let is_even = |row_id: u64| ((row_id - 3) / 10).is_multiple_of(2);
        let evens = mask_filter(RowAddrMask::from_allowed(
            storage
                .row_ids()
                .copied()
                .filter(|row_id| is_even(*row_id))
                .collect::<RowAddrTreeMap>(),
        ));
        for query in &queries {
            let walk = |ef: usize| {
                index
                    .search_nodes(
                        query.clone(),
                        ef,
                        &query_params(ef),
                        &storage,
                        &NoFilter,
                        None,
                    )
                    .unwrap()
            };
            let (narrow, wide) = (walk(40), walk(80));
            let upper = exact_nearest(&storage, query.clone(), 50, |_, _| true)[49].1;
            assert!(narrow.nodes[39].dist.0 < upper);
            assert!(wide.nodes[79].dist.0 >= upper);
            let expected = exact_nearest(&storage, query.clone(), 40, |row_id, distance| {
                is_even(row_id) && distance < upper
            });
            assert!(expected.len() < 40, "{} passing rows below", expected.len());

            let bounded = VamanaQueryParams {
                upper_bound: Some(upper),
                ..query_params(40)
            };
            let masked = index
                .search_nodes(query.clone(), 40, &bounded, &storage, evens.as_ref(), None)
                .unwrap();
            assert_eq!(masked.comparisons, narrow.comparisons + wide.comparisons);
            let found = masked
                .nodes
                .iter()
                .map(|node| (storage.row_id(node.id), node.dist.0))
                .collect::<Vec<_>>();
            assert_eq!(found, expected);
        }
    }

    /// Stores a graph cannot be built over are refused by every path that
    /// builds one.
    #[test]
    fn stores_a_graph_cannot_be_built_over_are_refused() {
        fn refused(result: Result<Vamana>, what: &str) {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::NotSupported { .. }), "{error}");
            assert!(error.to_string().contains(what), "{error}");
        }
        let (vectors, row_ids, _) = fixture(40, 0);
        refused(
            Vamana::index_vectors(
                &flat_storage_with_row_ids(vectors.clone(), row_ids.clone(), DistanceType::Dot),
                params(),
            ),
            "not dot",
        );
        let bits = FixedSizeListArray::try_new_from_values(
            arrow_array::UInt8Array::from(vec![7u8; 40 * 4]),
            4,
        )
        .unwrap();
        refused(
            Vamana::index_vectors(&FlatBinStorage::new(bits, DistanceType::Hamming), params()),
            "not hamming",
        );
        refused(Vamana::index_vectors(&rq_storage(40), params()), "RaBitQ");

        let l2 = flat_storage_with_row_ids(vectors.clone(), row_ids.clone(), DistanceType::L2);
        let index = Vamana::index_vectors(&l2, params()).unwrap();
        let dot = flat_storage_with_row_ids(vectors, row_ids, DistanceType::Dot);
        refused(
            index.remap(&RowAddrRemap::direct(HashMap::new()), &dot),
            "not dot",
        );
        refused(build_incremental(&index, &dot, 40, params()), "not dot");
    }

    #[test]
    fn a_prefilter_that_passes_nothing_returns_nothing() {
        let (storage, queries) = flat_fixture(300, 1);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        let prefilter = mask_filter(RowAddrMask::allow_nothing());
        assert!(
            search(
                &index,
                &storage,
                &queries[0],
                10,
                query_params(20),
                prefilter
            )
            .is_empty()
        );
    }

    /// Every row in `[lower, upper)` and no other, the bounds being the
    /// distances of the 6th and 16th nearest rows: both ends are exact.
    #[test]
    fn a_bounded_search_returns_exactly_the_rows_within_its_bounds() {
        let (storage, queries) = flat_fixture(1000, 10);
        let index = Vamana::index_vectors(&storage, params()).unwrap();
        for query in &queries {
            let unbounded = exact_nearest(&storage, query.clone(), 40, |_, _| true);
            let (lower, upper) = (unbounded[5].1, unbounded[15].1);
            let bounded = VamanaQueryParams {
                lower_bound: Some(lower),
                upper_bound: Some(upper),
                ..query_params(40)
            };
            let found = search(&index, &storage, query, 20, bounded, Arc::new(NoFilter));
            assert_eq!(found, unbounded[5..15].to_vec());
        }
    }

    /// A store remapped behind the graph's back holds fewer rows than the
    /// graph has vertices, and an empty graph indexes none of a store's rows;
    /// either store is scored directly rather than walked.
    #[test]
    fn a_store_that_does_not_match_the_graph_is_scored_directly() {
        let (vectors, row_ids, queries) = fixture(300, 3);
        let full = flat_storage_with_row_ids(vectors.clone(), row_ids.clone(), DistanceType::L2);
        let empty = Vamana::empty(params());
        for query in &queries {
            let found = search(
                &empty,
                &full,
                query,
                10,
                query_params(20),
                Arc::new(NoFilter),
            );
            assert_eq!(found, exact_nearest(&full, query.clone(), 10, |_, _| true));
        }
        let index = Vamana::index_vectors(&full, params()).unwrap();
        let kept = (0..300u32)
            .filter(|position| position % 3 != 0)
            .collect::<Vec<_>>();
        let shorter = flat_storage_with_row_ids(
            arrow_select::take::take(
                &vectors,
                &arrow_array::UInt32Array::from(kept.clone()),
                None,
            )
            .unwrap()
            .as_fixed_size_list()
            .clone(),
            kept.iter()
                .map(|position| row_ids[*position as usize])
                .collect(),
            DistanceType::L2,
        );
        for query in &queries {
            let expected = exact_nearest(&shorter, query.clone(), 10, |_, _| true);
            let found = search(
                &index,
                &shorter,
                query,
                10,
                query_params(20),
                Arc::new(NoFilter),
            );
            assert_eq!(found, expected);
        }
    }
}
