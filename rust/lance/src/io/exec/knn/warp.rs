// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! WARP candidate selection (WARPselect) over a dataset's vector index segments.

use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{Array, cast::AsArray, types::Float32Type};
use arrow_schema::DataType;
use futures::future::try_join_all;
use lance_core::utils::tokio::spawn_cpu;
use lance_index::metrics::MetricsCollector;
use lance_index::vector::VectorIndex;
use lance_index::vector::warp::{centroid_similarities, missing_similarity, probe_partitions};
use lance_linalg::distance::dot_f32;
use lance_table::format::IndexMetadata;

use crate::dataset::Dataset;
use crate::index::loading::DatasetIndexInternalExt;
use crate::{Error, Result};

/// WARPselect output for one index segment.
// TODO: remove once WARP candidate generation reads the selection.
#[allow(dead_code)]
#[derive(Debug)]
pub struct SegmentSelection {
    pub index: Arc<dyn VectorIndex>,
    /// `Γ[i][k] = <c_k, q_i>` for query token `i` and partition `k`; WARP's
    /// score decomposition reuses it as each probed token's centroid term.
    pub centroid_similarities: Vec<Vec<f32>>,
    /// `P_i`: partitions query token `i` probes in this segment, best first.
    pub probed_partitions: Vec<Vec<u32>>,
}

/// WARPselect output for one multivector query.
// TODO: remove once WARP candidate generation reads the selection.
#[allow(dead_code)]
#[derive(Debug)]
pub struct WarpSelection {
    pub segments: Vec<SegmentSelection>,
    /// `m_i`: the similarity imputed for documents query token `i` never
    /// observes, estimated over the partitions of every segment.
    pub missing_similarities: Vec<f32>,
}

/// Per-query WARP search parameters.
#[derive(Debug, Clone, Copy)]
pub struct WarpParams {
    /// Partitions each query token probes per index segment (`n_probe`).
    pub nprobe: usize,
    /// The rank `t'` whose estimated similarity is imputed for unobserved
    /// documents.
    pub imputation_rank: usize,
}

/// Run WARPselect for a flattened multivector query against index segments.
///
/// Each segment is probed with its own `nprobe` partitions, as IVF search
/// treats delta segments. `m_i` instead walks the partitions of all segments
/// together, because it estimates the `imputation_rank`-th most similar token
/// in the whole indexed corpus.
///
/// Only centroids and partition sizes are read; they live in the IVF model
/// loaded (and cached) when a segment is opened, so no partition data is
/// fetched from storage.
pub async fn warp_select(
    dataset: &Dataset,
    column: &str,
    segments: &[IndexMetadata],
    query_tokens: &dyn Array,
    dim: usize,
    params: WarpParams,
    metrics: &dyn MetricsCollector,
) -> Result<WarpSelection> {
    let WarpParams {
        nprobe,
        imputation_rank,
    } = params;
    if nprobe == 0 {
        return Err(Error::invalid_input(
            "WARP nprobes must be at least 1, got 0",
        ));
    }
    if imputation_rank == 0 {
        return Err(Error::invalid_input(
            "WARP imputation_rank must be at least 1, got 0",
        ));
    }
    if dim == 0 || !query_tokens.len().is_multiple_of(dim) || query_tokens.is_empty() {
        return Err(Error::invalid_input(format!(
            "WARP query on column {column} has {} values, which is not a positive multiple \
             of the vector dimension {dim}",
            query_tokens.len()
        )));
    }

    if segments.is_empty() {
        return Err(Error::invalid_input(format!(
            "WARP search on column {column} needs at least one index segment"
        )));
    }

    // Multivector indices are cosine-only, so compare against unit query tokens.
    let query_tokens = cast(query_tokens, &DataType::Float32)?;
    let query_tokens = query_tokens
        .as_primitive::<Float32Type>()
        .values()
        .chunks_exact(dim)
        .map(|token| {
            let norm = dot_f32(token, token).sqrt();
            if norm > 0.0 {
                token.iter().map(|value| value / norm).collect()
            } else {
                token.to_vec()
            }
        })
        .collect::<Vec<Vec<f32>>>();

    let indices = try_join_all(
        segments
            .iter()
            .map(|segment| dataset.open_vector_index(column, &segment.uuid, metrics)),
    )
    .await?;
    let column = column.to_string();
    let segment_uuids = segments
        .iter()
        .map(|segment| segment.uuid)
        .collect::<Vec<_>>();

    spawn_cpu(move || {
        let segments = indices
            .into_iter()
            .zip(segment_uuids)
            .map(|(index, uuid)| {
                let ivf = index.ivf_model();
                let centroids = ivf.centroids_array().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "WARP search on column {column} needs IVF centroids, but index segment \
                         {uuid} has none"
                    ))
                })?;
                let centroid_similarities = query_tokens
                    .iter()
                    .map(|token| centroid_similarities(token, centroids))
                    .collect::<Result<Vec<_>>>()?;
                let probed_partitions = centroid_similarities
                    .iter()
                    .map(|similarities| probe_partitions(similarities, nprobe))
                    .collect();
                Ok(SegmentSelection {
                    index,
                    centroid_similarities,
                    probed_partitions,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // The centroid model's `lengths` are not populated for current index
        // files; the partition storage holds the real sizes.
        let partition_sizes = segments
            .iter()
            .flat_map(|segment| {
                (0..segment.index.total_partitions())
                    .map(|partition| segment.index.partition_size(partition) as u32)
            })
            .collect::<Vec<_>>();
        let missing_similarities = (0..query_tokens.len())
            .map(|token| {
                let similarities = segments
                    .iter()
                    .flat_map(|segment| segment.centroid_similarities[token].iter().copied())
                    .collect::<Vec<_>>();
                missing_similarity(&similarities, &partition_sizes, imputation_rank)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(WarpSelection {
            segments,
            missing_similarities,
        })
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{
        FixedSizeListArray, Float32Array, ListArray, RecordBatch, RecordBatchIterator,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::IndexType;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::optimize::OptimizeOptions;
    use lance_linalg::distance::DistanceType;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;

    const DIM: usize = 8;
    const TOKENS_PER_ROW: usize = 4;

    fn multivector_batch(num_rows: usize, rng: &mut StdRng) -> RecordBatch {
        let values = Float32Array::from_iter_values(
            (0..num_rows * TOKENS_PER_ROW * DIM).map(|_| rng.random_range(-1.0..1.0)),
        );
        let tokens = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();
        let field = Arc::new(Field::new("item", tokens.data_type().clone(), true));
        let rows = ListArray::new(
            field,
            OffsetBuffer::from_lengths(std::iter::repeat_n(TOKENS_PER_ROW, num_rows)),
            Arc::new(tokens),
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            rows.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(rows)]).unwrap()
    }

    async fn write(dataset: Option<&mut Dataset>, batch: RecordBatch) -> Option<Dataset> {
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let params = WriteParams {
            max_rows_per_file: 100,
            mode: WriteMode::Append,
            ..Default::default()
        };
        match dataset {
            Some(dataset) => {
                dataset.append(reader, Some(params)).await.unwrap();
                None
            }
            None => Some(
                Dataset::write(reader, "memory://", Some(params))
                    .await
                    .unwrap(),
            ),
        }
    }

    /// Partitions are probed per segment, but `m_i` walks every segment's
    /// partitions together with their sizes.
    #[tokio::test]
    async fn test_warp_select_across_segments() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut dataset = write(None, multivector_batch(300, &mut rng)).await.unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                None,
                &VectorIndexParams::ivf_flat(4, DistanceType::Cosine),
                true,
            )
            .await
            .unwrap();
        write(Some(&mut dataset), multivector_batch(200, &mut rng)).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert!(dataset.get_fragments().len() > 1);
        let segments = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(segments.len(), 2);

        let num_query_tokens = 3;
        let query = Float32Array::from_iter_values(
            (0..num_query_tokens * DIM).map(|_| rng.random_range(-1.0..1.0)),
        );
        let imputation_rank = 700;
        let selection = warp_select(
            &dataset,
            "vector",
            &segments,
            &query,
            DIM,
            WarpParams {
                nprobe: 2,
                imputation_rank,
            },
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();

        assert_eq!(selection.segments.len(), 2);
        let total_tokens = selection
            .segments
            .iter()
            .flat_map(|segment| {
                (0..segment.index.total_partitions())
                    .map(|partition| segment.index.partition_size(partition))
            })
            .sum::<usize>();
        assert_eq!(total_tokens, 500 * TOKENS_PER_ROW);

        for (token, query_token) in query.values().chunks_exact(DIM).enumerate() {
            let norm = dot_f32(query_token, query_token).sqrt();
            let unit = query_token.iter().map(|v| v / norm).collect::<Vec<_>>();

            let mut walk = Vec::new();
            for segment in &selection.segments {
                let ivf = segment.index.ivf_model();
                let expected =
                    centroid_similarities(&unit, ivf.centroids_array().unwrap()).unwrap();
                assert_eq!(segment.centroid_similarities[token], expected);
                assert_eq!(
                    segment.probed_partitions[token],
                    probe_partitions(&expected, 2)
                );
                walk.extend(
                    expected
                        .into_iter()
                        .enumerate()
                        .map(|(partition, similarity)| {
                            (similarity, segment.index.partition_size(partition))
                        }),
                );
            }
            // Expected m_i from a full sort of the combined partitions.
            walk.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut population = 0;
            let expected = walk
                .iter()
                .find(|(_, size)| {
                    population += *size;
                    population >= imputation_rank
                })
                .unwrap()
                .0;
            assert_eq!(selection.missing_similarities[token], expected);
        }
    }

    #[rstest::rstest]
    #[case::zero_nprobe(0, 10, 8, 16)]
    #[case::zero_imputation_rank(2, 0, 8, 16)]
    #[case::ragged_query(2, 10, 8, 15)]
    #[case::no_segments(2, 10, 8, 16)]
    #[tokio::test]
    async fn test_warp_select_rejects_invalid_input(
        #[case] nprobe: usize,
        #[case] imputation_rank: usize,
        #[case] dim: usize,
        #[case] query_len: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(7);
        let dataset = write(None, multivector_batch(10, &mut rng)).await.unwrap();
        let query = Float32Array::from(vec![0.5; query_len]);
        let err = warp_select(
            &dataset,
            "vector",
            &[],
            &query,
            dim,
            WarpParams {
                nprobe,
                imputation_rank,
            },
            &NoOpMetricsCollector,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }
}
