// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

use std::collections::HashMap;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt8Array,
};
use arrow_schema::{ArrowError, DataType, Field as ArrowField};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::ROW_ID;
use lance_core::utils::tempfile::TempStrDir;
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::kmeans::{KMeansParams, train_kmeans};
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::{DistanceType, MetricType};
use uuid::Uuid;

use crate::dataset::WriteParams;
use crate::index::vector::{StageParams, VectorIndexParams};
use crate::io::exec::QUERY_INDEX_COL;
use crate::io::exec::filtered_read::FilteredReadExec;
use crate::utils::test::assert_plan_node_equals;

// Creates a dataset with 5 batches where each batch has 80 rows
//
// The dataset has the following columns:
//
//  i   - i32      : [0, 1, ..., 399]
//  s   - &str     : ["s-0", "s-1", ..., "s-399"]
//  vec - [f32; 32]: [[0, 1, ... 31], [32, ..., 63], ... [..., (80 * 5 * 32) - 1]]
//
// An IVF-PQ index with 2 partitions is trained on this data
pub struct TestVectorDataset {
    pub tmp_dir: TempStrDir,
    pub schema: Arc<ArrowSchema>,
    pub dataset: Dataset,
    dimension: u32,
}

impl TestVectorDataset {
    pub async fn new(data_storage_version: LanceFileVersion, stable_row_ids: bool) -> Result<Self> {
        Self::new_with_dimension(data_storage_version, stable_row_ids, 32).await
    }

    pub async fn new_with_dimension(
        data_storage_version: LanceFileVersion,
        stable_row_ids: bool,
        dimension: u32,
    ) -> Result<Self> {
        let path = TempStrDir::default();

        // Make sure the schema has metadata so it tests all paths that re-construct the schema along the way
        let metadata: HashMap<String, String> = vec![("dataset".to_string(), "vector".to_string())]
            .into_iter()
            .collect();

        let schema = Arc::new(ArrowSchema::new_with_metadata(
            vec![
                ArrowField::new("i", DataType::Int32, true),
                ArrowField::new("s", DataType::Utf8, true),
                ArrowField::new(
                    "vec",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        dimension as i32,
                    ),
                    true,
                ),
            ],
            metadata,
        ));

        let batches: Vec<RecordBatch> = (0..5)
            .map(|i| {
                let vector_values: Float32Array = (0..dimension * 80).map(|v| v as f32).collect();
                let vectors =
                    FixedSizeListArray::try_new_from_values(vector_values, dimension as i32)
                        .unwrap();
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 80..(i + 1) * 80)),
                        Arc::new(StringArray::from_iter_values(
                            (i * 80..(i + 1) * 80).map(|v| format!("s-{}", v)),
                        )),
                        Arc::new(vectors),
                    ],
                )
            })
            .collect::<std::result::Result<Vec<_>, ArrowError>>()?;

        let params = WriteParams {
            max_rows_per_group: 10,
            max_rows_per_file: 200,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        let dataset = Dataset::write(reader, &path, Some(params)).await?;

        Ok(Self {
            tmp_dir: path,
            schema,
            dataset,
            dimension,
        })
    }

    pub async fn make_vector_index(&mut self) -> Result<()> {
        self.make_vector_index_with_metric(MetricType::L2).await
    }

    pub async fn make_vector_index_with_metric(&mut self, metric: MetricType) -> Result<()> {
        let mut params = VectorIndexParams::ivf_pq(2, 8, 2, metric, 2);
        // Two partitions over 400 vectors only holds at a sample rate this
        // fixture can cover: the trained count is capped at the vectors
        // available per centroid.
        if let Some(StageParams::Ivf(ivf)) = params.stages.first_mut() {
            ivf.sample_rate = 200;
        }
        self.dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".to_string()),
                &params,
                true,
            )
            .await?;
        Ok(())
    }

    pub async fn make_ivf_hnsw_index(&mut self) -> Result<()> {
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            MetricType::L2,
            IvfBuildParams::new(2),
            HnswBuildParams::default(),
            SQBuildParams::default(),
        );
        self.dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("idx".to_string()),
                &params,
                true,
            )
            .await?;
        Ok(())
    }

    pub async fn make_segmented_vector_index(&mut self) -> Result<Vec<Uuid>> {
        let batch = self
            .dataset
            .scan()
            .project(&["vec"])
            .unwrap()
            .try_into_batch()
            .await?;
        let vectors = batch
            .column_by_name("vec")
            .expect("vector column should exist")
            .as_fixed_size_list();
        let values = vectors.values().as_primitive::<Float32Type>();
        let centroids = train_kmeans::<Float32Type>(
            values,
            KMeansParams::new(None, 10, 1, DistanceType::L2),
            self.dimension as usize,
            2,
            2,
        )
        .unwrap()
        .centroids
        .as_primitive::<Float32Type>()
        .clone();
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(centroids, self.dimension as i32).unwrap(),
        );
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            IvfBuildParams::try_with_centroids(2, centroids).unwrap(),
        );
        let fragment_ids = self
            .dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();

        let mut segments = Vec::with_capacity(fragment_ids.len());
        for fragment_id in fragment_ids {
            let mut builder =
                self.dataset
                    .create_index_builder(&["vec"], IndexType::Vector, &params);
            builder = builder.name("idx".to_string()).fragments(vec![fragment_id]);
            segments.push(builder.execute_uncommitted().await?);
        }

        let segment_ids = segments
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();
        self.dataset
            .commit_existing_index_segments("idx", "vec", segments)
            .await?;
        Ok(segment_ids)
    }

    pub async fn make_scalar_index(&mut self) -> Result<()> {
        self.dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await?;
        Ok(())
    }

    pub async fn make_segmented_scalar_index(&mut self) -> Result<()> {
        let params = ScalarIndexParams::default();
        let mut segments = Vec::with_capacity(self.dataset.get_fragments().len());
        for fragment in self.dataset.get_fragments() {
            segments.push(
                self.dataset
                    .create_index_builder(&["i"], IndexType::BTree, &params)
                    .name("i_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await?,
            );
        }
        self.dataset
            .commit_existing_index_segments("i_idx", "i", segments)
            .await?;
        Ok(())
    }

    fn fts_index_params() -> InvertedIndexParams {
        // These scanner tests search for the token "s" (from the `s-{N}`
        // column values) to exercise fragment/append coverage, and "s" is
        // in the full English stop-word list. Keep the token searchable;
        // stop-word behavior itself is covered by the tokenizer tests.
        InvertedIndexParams::default()
            .with_position(true)
            .remove_stop_words(false)
    }

    pub async fn make_fts_index(&mut self) -> Result<()> {
        let params = Self::fts_index_params();
        self.dataset
            .create_index(&["s"], IndexType::Inverted, None, &params, true)
            .await?;
        Ok(())
    }

    pub async fn make_segmented_fts_index(&mut self) -> Result<()> {
        let params = Self::fts_index_params();
        let fragments = self.dataset.get_fragments();
        let mut segments = Vec::with_capacity(fragments.len());
        for fragment in fragments {
            let segment = self
                .dataset
                .create_index_builder(&["s"], IndexType::Inverted, &params)
                .name("s_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await?;
            segments.push(segment);
        }
        self.dataset
            .commit_existing_index_segments("s_idx", "s", segments)
            .await
    }

    pub async fn append_new_data(&mut self) -> Result<()> {
        self.append_data_with_range(400, 410).await
    }

    pub async fn append_data_with_range(&mut self, start: i32, end: i32) -> Result<()> {
        let count = (end - start) as usize;
        let vector_values: Float32Array = (0..count)
            .flat_map(|i| vec![i as f32; self.dimension as usize].into_iter())
            .collect();
        let new_vectors =
            FixedSizeListArray::try_new_from_values(vector_values, self.dimension as i32).unwrap();
        let new_data: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter_values(start..end)),
            Arc::new(StringArray::from_iter_values(
                (start..end).map(|v| format!("s-{}", v)),
            )),
            Arc::new(new_vectors),
        ];
        let reader = RecordBatchIterator::new(
            vec![RecordBatch::try_new(self.schema.clone(), new_data).unwrap()]
                .into_iter()
                .map(Ok),
            self.schema.clone(),
        );
        self.dataset.append(reader, None).await?;
        Ok(())
    }
}

pub async fn make_binary_vector_dataset() -> Result<(TempStrDir, Dataset)> {
    let tmp_dir = TempStrDir::default();
    let dim = 4;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "bin",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::UInt8, true)),
                dim,
            ),
            false,
        ),
    ]));

    let vectors = FixedSizeListArray::try_new_from_values(
        UInt8Array::from(vec![
            0b0000_1111u8,
            0,
            0,
            0, //
            0b0000_0011u8,
            0,
            0,
            0, //
            0u8,
            0,
            0,
            0,
        ]),
        dim,
    )?;
    let ids = Int32Array::from(vec![0, 1, 2]);

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(vectors)])?;
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(reader, &tmp_dir, None).await?;
    let dataset = Dataset::open(&tmp_dir).await?;
    Ok((tmp_dir, dataset))
}

pub fn batch_row_ids(batch: &RecordBatch) -> Vec<u64> {
    batch
        .column_by_name(ROW_ID)
        .unwrap()
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec()
}

pub async fn assert_scan_slice_matches(
    dataset: &Dataset,
    fragments: Option<&[Fragment]>,
    offset: usize,
    limit: usize,
) -> Result<()> {
    let mut full_scan = dataset.scan();
    if let Some(fragments) = fragments {
        full_scan.with_fragments(fragments.to_vec());
    }
    let full = full_scan.try_into_batch().await?;
    let expected_offset = offset.min(full.num_rows());
    let expected_len = limit.min(full.num_rows() - expected_offset);
    let expected = full.slice(expected_offset, expected_len);

    let mut scan = dataset.scan();
    if let Some(fragments) = fragments {
        scan.with_fragments(fragments.to_vec());
    }
    scan.limit(Some(limit as i64), Some(offset as i64))?;
    let actual = scan.try_into_batch().await?;
    assert_eq!(actual, expected, "offset={offset}, limit={limit}");
    Ok(())
}

pub fn batch_knn_two_queries() -> (FixedSizeListArray, Vec<f32>) {
    let query_values = (32..96).map(|v| v as f32).collect::<Vec<_>>();
    let queries =
        FixedSizeListArray::try_new_from_values(Float32Array::from(query_values.clone()), 32)
            .unwrap();
    (queries, query_values)
}

pub fn assert_query_index_field(batch: &RecordBatch) {
    let schema = batch.schema();
    let field = schema.field(0);
    assert_eq!(field.name(), QUERY_INDEX_COL);
    assert_eq!(field.data_type(), &DataType::Int32);
    assert!(!field.is_nullable());
}

/// Assert that the plan when formatted matches the expected string.
///
/// Within expected, you can use `...` to match any number of characters.
pub async fn assert_plan_equals(
    dataset: &Dataset,
    plan: impl Fn(&mut Scanner) -> Result<&mut Scanner>,
    expected: &str,
) -> Result<()> {
    let mut scan = dataset.scan();
    // Pin target_parallelism=1 so EnforceDistribution produces deterministic plans
    // regardless of the machine's CPU count.
    scan.target_parallelism(1);
    plan(&mut scan)?;
    let exec_plan = scan.create_plan().await?;
    assert_plan_node_equals(exec_plan, expected).await
}

pub fn find_filtered_read(plan: &dyn ExecutionPlan) -> Option<&FilteredReadExec> {
    if let Some(f) = plan.downcast_ref::<FilteredReadExec>() {
        return Some(f);
    }
    for child in plan.children() {
        if let Some(f) = find_filtered_read(child.as_ref()) {
            return Some(f);
        }
    }
    None
}
