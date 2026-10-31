// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A merged partition's graph, extended from the one its old rows were in.
//!
//! An index merge assembles each partition from the rows of the merged
//! segments, the first segment's first, and from the new rows, and builds its
//! sub index anew. A Vamana partition extends the first segment's graph
//! instead: the graph's vertices are matched with the leading rows of the
//! merged storage, and the rows past them are inserted.

use arrow::compute::filter;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use lance_index::scalar::OldIndexDataFilter;
use lance_index::vector::VectorIndex;
use lance_index::vector::quantizer::Quantization;
use lance_index::vector::storage::VectorStore;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::vector::vamana::ROW_ID_COL;

use super::super::builder::ExistingIndex;
use super::super::ivf::v2::IVFIndex;
use crate::{Error, Result};

/// The graph of a partition in the first merged segment: its sub index batch,
/// the row ids of its vertices whose rows the segment still contributes, in
/// vertex order, and the rows of the merged storage it was read for.
pub struct MergeBase {
    partition: usize,
    batch: RecordBatch,
    row_ids: Vec<u64>,
    storage_rows: usize,
}

/// The graph of `partition` in the first of `sources`, the segments whose rows
/// open the merged partition, for `S` to extend over the merged partition's
/// storage of `storage_rows` rows. `None` when no segment's rows open the
/// partition, when `S` builds it anew however many of the graph's rows open
/// it, or when the graph leaves nothing to reuse (see [`surviving_row_ids`]).
pub async fn merge_base<S: IvfSubIndex + 'static, Q: Quantization + 'static>(
    sources: &[ExistingIndex],
    partition: usize,
    storage_rows: usize,
) -> Result<Option<MergeBase>> {
    let Some(source) = sources.first() else {
        return Ok(None);
    };
    let index = source
        .index
        .as_any()
        .downcast_ref::<IVFIndex<S, Q>>()
        .ok_or(Error::invalid_input("existing index is not IVF index"))?;
    // A partition a split added has no rows in the old segments.
    if partition >= index.ivf_model().num_partitions() {
        return Ok(None);
    }
    // No more vertices match rows of the storage than survive the merge's
    // filter, nor than the storage holds: a graph not extended even were
    // every vertex to match is not read, and one the filter leaves too few
    // vertices of is not held through the build.
    let vertices = index.ivf_model().partition_size(partition);
    let may_extend =
        |surviving: usize| S::may_extend(vertices, surviving.min(storage_rows), storage_rows);
    if !may_extend(vertices) {
        return Ok(None);
    }
    // Resolved before the graph is read, as `take_partition_batches` resolves
    // it before decoding a partition.
    let old_data_filter = source.old_data_filter().await?;
    let batch = index.read_sub_index_batch(partition, None, None).await?;
    let Some(row_ids) = surviving_row_ids(&batch, old_data_filter, partition)? else {
        return Ok(None);
    };
    if !may_extend(row_ids.len()) {
        return Ok(None);
    }
    Ok(Some(MergeBase {
        partition,
        batch,
        row_ids,
        storage_rows,
    }))
}

/// The row ids of `graph`'s vertices that `old_data_filter`, the filter the
/// merge applied to the segment's rows, keeps, in vertex order: a row
/// rewritten since comes back among the new rows with a new vector, under its
/// old stable row id or, rewritten in place, at its old address. `None` when
/// no row is left, or when the graph has no UInt64 row ids: such a graph does
/// not load, and the merge rebuilds it rather than fail.
fn surviving_row_ids(
    graph: &RecordBatch,
    old_data_filter: Option<&OldIndexDataFilter>,
    partition: usize,
) -> Result<Option<Vec<u64>>> {
    let Some(row_ids) = graph
        .column_by_name(ROW_ID_COL)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
    else {
        log::warn!(
            "Rebuilding the Vamana graph of partition {partition} in an index merge: its old \
             graph has no UInt64 {ROW_ID_COL}"
        );
        return Ok(None);
    };
    let row_ids = match old_data_filter {
        Some(old_data) => filter(row_ids, &old_data.filter_row_ids(row_ids))?
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec(),
        None => row_ids.values().to_vec(),
    };
    Ok((!row_ids.is_empty()).then_some(row_ids))
}

/// `storage`'s sub index, extended from `base` over the leading rows of
/// `storage` that are `base`'s, in order. A row renamed or dropped since
/// `base` was built ends them, so that no row past them is matched with a
/// vertex by its row id. A graph this build cannot load is rebuilt, as every
/// merge did before merges read graphs.
pub fn extend<S: IvfSubIndex>(
    base: MergeBase,
    storage: &impl VectorStore,
    params: S::BuildParams,
) -> Result<S> {
    // The old segments' rows were translated through any fragment reuse index
    // when their partitions were loaded, so that building the storage drops
    // none of the rows the merge counted.
    debug_assert_eq!(
        storage.len(),
        base.storage_rows,
        "the storage of merged partition {} against the rows its old graph was read for",
        base.partition
    );
    let base_rows = base
        .row_ids
        .iter()
        .zip(storage.row_ids())
        .take_while(|(old, merged)| old == merged)
        .count();
    let old = match S::load(base.batch) {
        Ok(old) => old,
        Err(error) => {
            log::warn!(
                "Rebuilding the Vamana graph of partition {} in an index merge: its old graph \
                 does not load: {error}",
                base.partition
            );
            return S::index_vectors(storage, params);
        }
    };
    log::debug!(
        "Matching the old Vamana graph of partition {} with the first {base_rows} of its {} rows",
        base.partition,
        storage.len()
    );
    S::index_vectors_incremental(&old, storage, base_rows, params)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Float32Type;
    use arrow_array::{FixedSizeListArray, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_index::vector::vamana::graph::Adjacency;
    use lance_index::vector::vamana::{
        VAMANA_METADATA_KEY, Vamana, VamanaBuildParams, VamanaMetadata,
    };
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_seed;
    use roaring::RoaringBitmap;

    use super::*;

    const DIMENSION: usize = 8;

    fn params() -> VamanaBuildParams {
        VamanaBuildParams {
            max_degree: 8,
            search_list_size: 16,
            alpha: 1.2,
        }
    }

    /// The first `rows` of one set of random vectors, numbered from 0.
    fn storage(rows: usize) -> FlatFloatStorage {
        let values = generate_random_array_with_seed::<Float32Type>(60 * DIMENSION, [7; 32]);
        let vectors = FixedSizeListArray::try_new_from_values(values, DIMENSION as i32).unwrap();
        FlatFloatStorage::new(vectors.slice(0, rows), DistanceType::L2)
    }

    /// A graph's vertices, their out-edges and its metadata.
    fn contents(graph: &Vamana) -> (Vec<u64>, Vec<Vec<u32>>, VamanaMetadata) {
        let row_ids = graph.graph().row_ids().to_vec();
        let neighbors = (0..row_ids.len() as u32)
            .map(|vertex| graph.graph().neighbors(vertex).to_vec())
            .collect();
        (row_ids, neighbors, graph.metadata().clone())
    }

    /// The graph of the first 50 rows as a merge of 60 rows reads it, its
    /// format version moved by `format_shift`.
    fn base(format_shift: u32) -> MergeBase {
        let old = Vamana::index_vectors(&storage(50), params()).unwrap();
        let mut metadata = old.metadata().clone();
        metadata.format_version += format_shift;
        let batch = old.to_batch().unwrap();
        let mut schema_metadata = batch.schema().metadata().clone();
        schema_metadata.insert(
            VAMANA_METADATA_KEY.to_string(),
            serde_json::to_string(&metadata).unwrap(),
        );
        let schema = batch
            .schema()
            .as_ref()
            .clone()
            .with_metadata(schema_metadata);
        MergeBase {
            partition: 0,
            batch: RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap(),
            row_ids: (0..50).collect(),
            storage_rows: 60,
        }
    }

    /// An old graph this build cannot load, as one of another format, leaves
    /// the merge to build the partition from scratch rather than fail, while
    /// one it loads is extended into a graph no fresh build draws.
    #[test]
    fn an_old_graph_that_does_not_load_is_built_anew() {
        let merged = storage(60);
        let fresh = contents(&Vamana::index_vectors(&merged, params()).unwrap());
        let extended = extend::<Vamana>(base(0), &merged, params()).unwrap();
        assert_ne!(contents(&extended), fresh);
        let rebuilt = extend::<Vamana>(base(1), &merged, params()).unwrap();
        assert_eq!(contents(&rebuilt), fresh);
    }

    /// A merge reuses the graph's row ids the segment's filter keeps, in
    /// vertex order. A graph whose row ids are missing or not UInt64 is left
    /// to a rebuild rather than failing the merge, as is one with no row left.
    #[test]
    fn a_graph_without_row_ids_is_left_to_a_rebuild() {
        let graph = Vamana::index_vectors(&storage(10), params())
            .unwrap()
            .to_batch()
            .unwrap();
        let all = Some((0..10).collect::<Vec<u64>>());
        assert_eq!(surviving_row_ids(&graph, None, 0).unwrap(), all);
        // The rows live in fragment 0.
        let fragments = |to_keep: u32| OldIndexDataFilter::Fragments {
            to_keep: RoaringBitmap::from_iter([to_keep]),
            to_remove: RoaringBitmap::new(),
        };
        assert_eq!(
            surviving_row_ids(&graph, Some(&fragments(0)), 0).unwrap(),
            all
        );
        assert_eq!(
            surviving_row_ids(&graph, Some(&fragments(1)), 0).unwrap(),
            None
        );

        let at = graph.schema().index_of(ROW_ID_COL).unwrap();
        let others = (0..graph.num_columns())
            .filter(|column| *column != at)
            .collect::<Vec<_>>();
        let missing = graph.project(&others).unwrap();
        assert_eq!(surviving_row_ids(&missing, None, 0).unwrap(), None);
        let mut fields = graph.schema().fields().to_vec();
        fields[at] = Arc::new(Field::new(ROW_ID_COL, DataType::Int64, false));
        let mut columns = graph.columns().to_vec();
        columns[at] = Arc::new(Int64Array::from_iter_values(0..10));
        let signed = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        assert_eq!(surviving_row_ids(&signed, None, 0).unwrap(), None);
    }
}
