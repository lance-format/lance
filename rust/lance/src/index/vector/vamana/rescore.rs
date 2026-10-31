// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{Field, Schema};
use lance_core::ROW_ID_FIELD;
use lance_index::vector::flat::compute_distance;
use lance_index::vector::{DIST_COL, Query, VECTOR_RESULT_SCHEMA};
use lance_io::scheduler::IoStats;
use lance_linalg::distance::DistanceType;

use super::vectors::{Vectors, read_vectors};
use crate::dataset::Dataset;
use crate::{Error, Result};

/// The nearest `keep` of `row_ids` to `query` within its bounds, nearest
/// first, at their exact `distance_type` distances from the vectors
/// `dataset` stores in `column`, read as [`read_vectors`] reads them and
/// counted in `io_stats`.
///
/// The distances come from [`compute_distance`], as those of brute force and
/// refine do, so these rows rank among theirs; rows at equal distances rank
/// by row id.
pub async fn rescore(
    dataset: &Arc<Dataset>,
    column: &str,
    distance_type: DistanceType,
    query: &Query,
    row_ids: Vec<u64>,
    keep: usize,
    io_stats: Option<IoStats>,
) -> Result<RecordBatch> {
    if keep == 0 {
        return Ok(RecordBatch::new_empty(VECTOR_RESULT_SCHEMA.clone()));
    }
    let Vectors {
        row_ids,
        vectors,
        through_take,
        local,
    } = read_vectors(dataset, column, &row_ids, io_stats.as_ref()).await?;
    if row_ids.is_empty() {
        return Ok(RecordBatch::new_empty(VECTOR_RESULT_SCHEMA.clone()));
    }
    log::debug!(
        "IVF_VAMANA rescores {} rows of column {column}, {through_take} of them read through \
         take; of its reads of held files, {} were made in place and {} on the blocking pool in \
         {} trips",
        row_ids.len(),
        local.in_place,
        local.handed_off,
        local.trips
    );
    let row_ids = UInt64Array::from(row_ids);
    let schema = Schema::new(vec![
        Field::new(column, vectors.data_type().clone(), true),
        ROW_ID_FIELD.clone(),
    ]);
    let vectors = RecordBatch::try_new(Arc::new(schema), vec![vectors, Arc::new(row_ids.clone())])?;
    let scored = compute_distance(query.key.clone(), distance_type, column, vectors).await?;
    let distances = scored
        .column_by_name(DIST_COL)
        .ok_or_else(|| Error::internal(format!("compute_distance returned no {DIST_COL}")))?
        .as_primitive::<Float32Type>();
    let mut nearest = distances
        .iter()
        .zip(row_ids.values())
        .filter_map(|(distance, row_id)| Some((distance?, *row_id)))
        .filter(|(distance, _)| {
            !distance.is_nan()
                && query.lower_bound.is_none_or(|lower| *distance >= lower)
                && query.upper_bound.is_none_or(|upper| *distance < upper)
        })
        .collect::<Vec<_>>();
    nearest.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    nearest.truncate(keep);
    let (distances, row_ids): (Vec<f32>, Vec<u64>) = nearest.into_iter().unzip();
    Ok(RecordBatch::try_new(
        VECTOR_RESULT_SCHEMA.clone(),
        vec![
            Arc::new(Float32Array::from(distances)),
            Arc::new(UInt64Array::from(row_ids)),
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use arrow_array::types::UInt64Type;
    use arrow_array::{ArrayRef, FixedSizeListArray, Int32Array, RecordBatchIterator};
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use lance_core::{ROW_ADDR, ROW_ID};
    use lance_file::writer::FileWriterOptions;
    use lance_index::vector::DEFAULT_QUERY_PARALLELISM;

    use super::*;
    use crate::dataset::transaction::Operation;
    use crate::dataset::{WriteDestination, WriteParams};

    const ROWS: usize = 64;
    const DIMENSION: i32 = 128;

    /// The vectors of `rows`, each `value(row)` repeated `DIMENSION` times,
    /// except the one at `null`, which is null.
    fn vectors(rows: Range<usize>, value: impl Fn(usize) -> f32, null: Option<usize>) -> ArrayRef {
        let nulls = null.map(|null| NullBuffer::from_iter(rows.clone().map(|row| row != null)));
        let values = rows
            .flat_map(|row| std::iter::repeat_n(value(row), DIMENSION as usize))
            .collect::<Vec<_>>();
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let vectors = FixedSizeListArray::try_new(
            item,
            DIMENSION,
            Arc::new(Float32Array::from(values)),
            nulls,
        )
        .unwrap();
        Arc::new(vectors)
    }

    /// One fragment of `ROWS` [`vectors`], written 32 to a page.
    async fn write(value: impl Fn(usize) -> f32, null: Option<usize>) -> Arc<Dataset> {
        let batch = RecordBatch::try_from_iter([
            (
                "id",
                Arc::new(Int32Array::from_iter_values(0..ROWS as i32)) as ArrayRef,
            ),
            ("vec", vectors(0..ROWS, value, null)),
        ])
        .unwrap();
        let schema = batch.schema();
        let batches = (0..ROWS)
            .step_by(32)
            .map(|start| Ok(batch.slice(start, 32)))
            .collect::<Vec<_>>();
        let params = WriteParams {
            file_writer_options: Some(FileWriterOptions {
                data_cache_bytes: Some(2 * 16 * 1024),
                ..Default::default()
            }),
            ..Default::default()
        };
        let dataset = Dataset::write(
            RecordBatchIterator::new(batches, schema),
            "memory://",
            Some(params),
        )
        .await
        .unwrap();
        Arc::new(dataset)
    }

    /// A query for the vector of `DIMENSION` values all `value`.
    fn query(value: f32) -> Query {
        Query {
            column: "vec".to_string(),
            key: Arc::new(Float32Array::from(vec![value; DIMENSION as usize])),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            search_effort: 0.5,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
            adaptive_stop: None,
        }
    }

    fn row_ids(batch: &RecordBatch) -> Vec<u64> {
        batch[ROW_ID].as_primitive::<UInt64Type>().values().to_vec()
    }

    /// A row whose vector holds no number, or that has no vector, is left out,
    /// and the others rank by their exact distances.
    #[tokio::test]
    async fn rows_whose_vectors_are_not_numbers_are_left_out() {
        let dataset = write(|row| if row == 9 { f32::NAN } else { row as f32 }, Some(20)).await;
        let every_row = (0..ROWS as u64).collect::<Vec<_>>();
        let nearest = rescore(
            &dataset,
            "vec",
            DistanceType::L2,
            &query(0.0),
            every_row,
            ROWS,
            None,
        )
        .await
        .unwrap();
        let expected = (0..ROWS as u64)
            .filter(|row| *row != 9 && *row != 20)
            .collect::<Vec<_>>();
        assert_eq!(row_ids(&nearest), expected);
    }

    /// Rows at equal distances rank by row id, whichever way their vectors
    /// were read: here an overlay gives the row of the smaller id its vector,
    /// so that vector is read through take, after the other's is read at its
    /// byte offset.
    #[tokio::test]
    async fn rows_at_equal_distances_rank_by_row_id() {
        let dataset = write(|row| if row == 40 { 1000.0 } else { row as f32 }, None).await;
        let schema = dataset.schema().project(&["vec"]).unwrap();
        let mut overlay = dataset
            .get_fragment(0)
            .unwrap()
            .write_overlay(&schema)
            .await
            .unwrap();
        let newer = RecordBatch::try_from_iter([
            (ROW_ADDR, Arc::new(UInt64Array::from(vec![3])) as ArrayRef),
            ("vec", vectors(3..4, |_| 1000.0, None)),
        ])
        .unwrap();
        overlay.write_batch(&newer).await.unwrap();
        let group = overlay.finish().await.unwrap().unwrap();
        let read_version = dataset.version().version;
        let dataset = Arc::new(
            Dataset::commit(
                WriteDestination::Dataset(dataset),
                Operation::DataOverlay {
                    groups: vec![group],
                },
                Some(read_version),
                None,
                None,
                Arc::new(Default::default()),
                false,
            )
            .await
            .unwrap(),
        );
        let read = read_vectors(&dataset, "vec", &[3, 40], None).await.unwrap();
        assert_eq!((read.row_ids, read.through_take), (vec![40, 3], 1));
        let nearest = rescore(
            &dataset,
            "vec",
            DistanceType::L2,
            &query(1000.0),
            vec![3, 40],
            1,
            None,
        )
        .await
        .unwrap();
        assert_eq!(row_ids(&nearest), vec![3]);
    }
}
