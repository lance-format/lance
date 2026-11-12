// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};

pub async fn write_vector_segment_metadata(
    dataset: &Dataset,
    index_name: &str,
    field_id: i32,
    uuid: Uuid,
    fragment_bitmap: impl IntoIterator<Item = u32>,
    payload: &[u8],
) -> IndexMetadata {
    let index_path = dataset
        .indices_dir()
        .join(uuid.to_string())
        .join(INDEX_FILE_NAME);
    dataset
        .object_store
        .as_ref()
        .put(&index_path, payload)
        .await
        .unwrap();
    IndexMetadata {
        uuid,
        name: index_name.to_string(),
        fields: vec![field_id],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap.into_iter().collect()),
        index_details: Some(Arc::new(vector_index_details_default())),
        index_version: IndexType::Vector.version(),
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(vec![lance_table::format::IndexFile {
            path: INDEX_FILE_NAME.to_string(),
            size_bytes: payload.len() as u64,
            file_metadata_size_bytes: None,
        }]),
        clustering_id: None,
        metadata: Default::default(),
    }
}

pub fn list_io_stats(stats: &IoStats) -> IoStats {
    let requests = stats
        .requests
        .iter()
        .filter(|request| request.method == "list")
        .cloned()
        .collect::<Vec<_>>();
    IoStats {
        read_iops: requests.len() as u64,
        requests,
        ..Default::default()
    }
}

pub fn segment_from_metadata(metadata: &IndexMetadata) -> IndexSegment {
    metadata
        .clone()
        .into_index_segment()
        .expect("test segment metadata should convert to an index segment")
}

pub async fn write_fragmented_vector_dataset(uri: &str, dimension: i32) -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        ),
    ]));
    let batches = (0..5)
        .map(|i| {
            let vector_values: Float32Array = (0..dimension * 80)
                .map(|value| value as f32 + (i * 1000) as f32)
                .collect();
            let vectors =
                FixedSizeListArray::try_new_from_values(vector_values, dimension).unwrap();
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(i * 80..(i + 1) * 80)),
                    Arc::new(vectors),
                ],
            )
        })
        .collect::<std::result::Result<Vec<_>, arrow_schema::ArrowError>>()
        .unwrap();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_group: 10,
            max_rows_per_file: 80,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

pub async fn write_skewed_fragmented_vector_dataset(uri: &str, dimension: i32) -> Dataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension,
            ),
            false,
        ),
    ]));
    let first_fragment_rows = 20_000;
    let second_fragment_rows = 100;
    // A tiny per-row perturbation keeps the rows distinct, so the oversized
    // partition can actually be split by k-means.
    let first_values = (0..first_fragment_rows * dimension as usize)
        .map(|i| (i / dimension as usize) as f32 * 0.0001)
        .collect::<Vec<f32>>();
    let second_values = vec![100.0f32; second_fragment_rows * dimension as usize];
    let first_vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(first_values), dimension)
            .unwrap();
    let second_vectors =
        FixedSizeListArray::try_new_from_values(Float32Array::from(second_values), dimension)
            .unwrap();
    let batches = vec![
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..first_fragment_rows as i32)),
                Arc::new(first_vectors),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    first_fragment_rows as i32..(first_fragment_rows + second_fragment_rows) as i32,
                )),
                Arc::new(second_vectors),
            ],
        )
        .unwrap(),
    ];
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_group: 1024,
            max_rows_per_file: first_fragment_rows,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

pub async fn create_segmented_vector_index(
    dataset: &mut Dataset,
    index_name: &str,
    column: &str,
    dimension: i32,
) -> Vec<Uuid> {
    let batch = dataset
        .scan()
        .project(&[column])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = batch
        .column_by_name(column)
        .expect("vector column should exist")
        .as_fixed_size_list();
    let values = vectors.values().as_primitive::<Float32Type>();
    let centroids = train_kmeans::<Float32Type>(
        values,
        KMeansParams::new(None, 10, 1, DistanceType::L2),
        dimension as usize,
        2,
        2,
    )
    .unwrap()
    .centroids
    .as_primitive::<Float32Type>()
    .clone();
    let centroids =
        Arc::new(FixedSizeListArray::try_new_from_values(centroids, dimension).unwrap());
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(2, centroids).unwrap(),
    );
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let columns = [column];

    let mut segments = Vec::with_capacity(fragment_ids.len());
    for fragment_id in fragment_ids {
        let mut builder = dataset.create_index_builder(&columns, IndexType::Vector, &params);
        builder = builder
            .name(index_name.to_string())
            .fragments(vec![fragment_id]);
        segments.push(builder.execute_uncommitted().await.unwrap());
    }

    let segment_ids = segments
        .iter()
        .map(|segment| segment.uuid)
        .collect::<Vec<_>>();
    dataset
        .commit_existing_index_segments(index_name, column, segments)
        .await
        .unwrap();
    segment_ids
}

/// Two-fragment, two-int-column dataset for the scalar-segment covering
/// tests below. Column "id" is keyed, "carried" plays the covered column.
pub async fn write_two_int_column_dataset(uri: &str, rows_per_fragment: i32) -> Dataset {
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("carried", array::step::<Int32Type>())
        .into_reader_rows(
            RowCount::from(rows_per_fragment as u64),
            BatchCount::from(2),
        );
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment as usize,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// Nothing writes a covering declaration yet, so the declaration below is
/// hand-constructed on top of real single-field segments: merge plumbing
/// never reads covered-column storage, so a declaration naming a real,
/// uninvolved column is a faithful stand-in for a genuinely covered segment.
pub fn make_covered(
    segment: IndexMetadata,
    keyed_field_id: i32,
    carried_field_id: i32,
) -> IndexMetadata {
    IndexMetadata {
        fields: vec![keyed_field_id, carried_field_id],
        covering_fields: vec![carried_field_id],
        ..segment
    }
}

/// Build one covered segment per named fragment, merge them, and assert the
/// merged declaration survived and is committable.
///
/// The per-index-type cases below differ only in how their dataset and
/// params are built, so the merge and the assertions live here.
pub async fn assert_merge_preserves_covering(
    dataset: &mut Dataset,
    column: &str,
    index_type: IndexType,
    params: &dyn IndexParams,
    fragment_ids: Vec<u32>,
    keyed_field_id: i32,
    carried_field_id: i32,
) {
    let index_name = format!("covered_{column}");
    let mut covered_segments = Vec::new();
    for fragment_id in fragment_ids {
        let segment = dataset
            .create_index_builder(&[column], index_type, params)
            .name(index_name.clone())
            .fragments(vec![fragment_id])
            .execute_uncommitted()
            .await
            .unwrap();
        covered_segments.push(make_covered(segment, keyed_field_id, carried_field_id));
    }

    let merged = dataset
        .merge_existing_index_segments(covered_segments)
        .await
        .unwrap();

    merged
        .validate_covering_fields()
        .expect("merged covering declaration must be committable");
    assert_eq!(merged.fields, vec![keyed_field_id, carried_field_id]);
    assert_eq!(
        merged.covering_fields,
        vec![carried_field_id],
        "merge_segments must neither drop the covering declaration nor \
         produce an uncommittable one"
    );
}

pub fn all_fragment_ids(dataset: &Dataset) -> Vec<u32> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect()
}

#[cfg(feature = "geo")]
pub async fn compact_into_one_fragment(
    dataset: &mut Dataset,
    rows_per_fragment: i32,
    fragments: i32,
) {
    compact_files(
        dataset,
        crate::utils::test::geo::deferred_compaction(rows_per_fragment, fragments),
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        dataset.get_fragments().len(),
        1,
        "the fragments must become one rewrite group"
    );
}

/// Rows the merged index holds.
#[cfg(feature = "geo")]
pub async fn merged_rtree_item_count(dataset: &Dataset, merged: &IndexMetadata) -> u64 {
    let index =
        crate::index::scalar::open_scalar_index(dataset, "geometry", merged, &NoOpMetricsCollector)
            .await
            .unwrap();
    index.statistics().unwrap()["num_items"].as_u64().unwrap()
}

pub fn sample_vector_field() -> Field {
    let dimensions = 16;
    let column_name = "vec";
    Field::new(
        column_name,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimensions,
        ),
        false,
    )
}

/// IVF_FLAT params carrying `partitions` precomputed centroids and no
/// partition count. The centroids are trained on a fresh sample of the
/// distribution `vector_reader` draws from, so the partitions they cut are
/// balanced; random centroids could leave one under the join threshold.
pub fn ivf_flat_from_centroids(partitions: usize) -> VectorIndexParams {
    let dimension = 16;
    let sample = generate_random_array(partitions * 256 * dimension as usize);
    let centroids = train_kmeans::<Float32Type>(
        &sample,
        KMeansParams::new(None, 10, 1, DistanceType::L2),
        dimension as usize,
        partitions,
        256,
    )
    .unwrap()
    .centroids
    .as_primitive::<Float32Type>()
    .clone();
    let centroids =
        Arc::new(FixedSizeListArray::try_new_from_values(centroids, dimension).unwrap());
    VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams {
            centroids: Some(centroids),
            ..Default::default()
        },
    )
}

/// Partition counts for a trained logical vector index, one per segment.
/// Whether a segment covers no rows, which is how a definition reads back
/// from the manifest.
pub fn segment_covers_nothing(index: &IndexMetadata) -> bool {
    index
        .fragment_bitmap
        .as_ref()
        .is_some_and(roaring::RoaringBitmap::is_empty)
}

pub async fn trained_partitions(dataset: &Dataset) -> Vec<usize> {
    dataset
        .open_logical_vector_index("vector", "vector_idx")
        .await
        .unwrap()
        .as_ivf()
        .unwrap()
        .num_partitions_per_segment()
        .into_iter()
        .map(|(_, partitions)| partitions)
        .collect()
}

/// A reader over one batch of `rows` random 16-dimensional vectors.
pub fn vector_reader(rows: usize) -> impl arrow_array::RecordBatchReader + Send + 'static {
    let dimensions = 16;
    let field = Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimensions,
        ),
        false,
    );
    let schema = Arc::new(Schema::new(vec![field]));
    let values = generate_random_array(rows * dimensions as usize);
    let vectors = arrow_array::FixedSizeListArray::try_new_from_values(values, dimensions).unwrap();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
    RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema)
}

/// Build a dataset of `rows` random vectors, one fragment.
pub async fn small_vector_dataset(dir: &std::path::Path, rows: usize) -> Dataset {
    Dataset::write(vector_reader(rows), dir.to_str().unwrap(), None)
        .await
        .unwrap()
}

/// A dataset of `rows` rows where only the first `non_null` hold a vector,
/// each random.
pub async fn partly_null_vector_dataset(
    dir: &std::path::Path,
    rows: usize,
    non_null: usize,
) -> Dataset {
    let dimensions = 16;
    let field = Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimensions,
        ),
        true,
    );
    let schema = Arc::new(Schema::new(vec![field]));
    let mut builder = arrow_array::builder::FixedSizeListBuilder::new(
        arrow_array::builder::Float32Builder::new(),
        dimensions,
    );
    let values = generate_random_array(non_null * dimensions as usize);
    for row in 0..rows {
        if row < non_null {
            let start = row * dimensions as usize;
            builder
                .values()
                .append_slice(&values.values()[start..start + dimensions as usize]);
            builder.append(true);
        } else {
            for _ in 0..dimensions {
                builder.values().append_null();
            }
            builder.append(false);
        }
    }
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(builder.finish())]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    Dataset::write(reader, dir.to_str().unwrap(), None)
        .await
        .unwrap()
}

/// Append `rows` more random vectors as a new fragment.
pub async fn append_vectors(dir: &std::path::Path, rows: usize) -> Dataset {
    Dataset::write(
        vector_reader(rows),
        dir.to_str().unwrap(),
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// A table of `rows`, each holding `vectors_per_row` vectors.
pub async fn multivector_dataset(
    dir: &std::path::Path,
    rows: usize,
    vectors_per_row: usize,
) -> Dataset {
    multivector_dataset_with(dir, &vec![vectors_per_row; rows]).await
}

/// A multivector table whose row `i` holds `lengths[i]` vectors, so uneven
/// and empty lists can be built as easily as uniform ones.
pub async fn multivector_dataset_with(dir: &std::path::Path, lengths: &[usize]) -> Dataset {
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, ListBuilder};

    let dimensions = 16;
    let field = Field::new(
        "vector",
        DataType::List(Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimensions,
            ),
            true,
        ))),
        true,
    );
    let schema = Arc::new(Schema::new(vec![field]));

    let mut builder =
        ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), dimensions));
    for (row, &length) in lengths.iter().enumerate() {
        for vector in 0..length {
            for value in 0..dimensions {
                builder
                    .values()
                    .values()
                    .append_value((row + vector + value as usize) as f32 + 0.5);
            }
            builder.values().append(true);
        }
        builder.append(true);
    }
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(builder.finish())]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
    Dataset::write(reader, dir.to_str().unwrap(), None)
        .await
        .unwrap()
}

/// Helper function to check if an index is being used in a query plan
pub fn assert_index_usage(plan: &str, column_name: &str, should_use_index: bool, context: &str) {
    let index_used = if column_name == "text" {
        // For inverted index, look for MatchQuery which indicates FTS index usage
        plan.contains("MatchQuery")
    } else {
        // For btree/bitmap, look for MaterializeIndex which indicates scalar index usage
        plan.contains("ScalarIndexQuery")
    };

    if should_use_index {
        assert!(
            index_used,
            "Query plan should use index {}: {}",
            context, plan
        );
    } else {
        assert!(
            !index_used,
            "Query plan should NOT use index {}: {}",
            context, plan
        );
    }
}

// Helper function to validate indices after each clone iteration
pub async fn validate_indices_after_clone(
    dataset: &Dataset,
    round: usize,
    expected_scalar_rows: usize,
    dimensions: u32,
) {
    // Verify cloned dataset has indices
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices.len(),
        2,
        "Round {}: Cloned dataset should have 2 indices",
        round
    );
    let index_names: HashSet<String> = indices.iter().map(|idx| idx.name.clone()).collect();
    assert!(
        index_names.contains("vector_idx"),
        "Round {}: Should contain vector_idx",
        round
    );
    assert!(
        index_names.contains("category_idx"),
        "Round {}: Should contain category_idx",
        round
    );

    // Test basic data access without using indices for now
    // This ensures the dataset is accessible and data is intact
    // In chain cloning, each round adds 50 rows to the previous dataset
    // Round 1: 300 (original), Round 2: 350 (300 + 50 from round 1), Round 3: 400 (350 + 50 from round 2)
    let expected_total_rows = 300 + (round - 1) * 50;
    let total_rows = dataset.count_rows(None).await.unwrap();
    assert_eq!(
        total_rows, expected_total_rows,
        "Round {}: Should have {} rows after clone (chain cloning accumulates data)",
        round, expected_total_rows
    );

    // Verify vector search
    let query_vector = generate_random_array(dimensions as usize);
    let search_results = dataset
        .scan()
        .nearest("vector", &query_vector, 5)
        .unwrap()
        .limit(Some(5), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert!(
        search_results.num_rows() > 0,
        "Round {}: Vector search should return results immediately after clone",
        round
    );

    // Test basic scalar query to verify data integrity
    let scalar_results = dataset
        .scan()
        .filter("category = 'category_0'")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    assert_eq!(
        expected_scalar_rows,
        scalar_results.num_rows(),
        "Round {}: Scalar query should return {} results",
        round,
        expected_scalar_rows
    );
}

/// Helper to assert that all indices have file sizes populated
pub async fn assert_all_indices_have_files(dataset: &Dataset, context: &str) {
    let indices = dataset.load_indices().await.unwrap();
    for index in indices.iter() {
        // Skip system indices (mem_wal, frag_reuse) which don't have files
        if index.name == lance_index::mem_wal::MEM_WAL_INDEX_NAME
            || index.name == lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME
        {
            continue;
        }
        assert!(
            index.files.is_some(),
            "{}: Index '{}' should have files field populated",
            context,
            index.name
        );
        let files = index.files.as_ref().unwrap();
        assert!(
            !files.is_empty(),
            "{}: Index '{}' should have at least one file",
            context,
            index.name
        );
        for file in files {
            assert!(
                file.size_bytes > 0,
                "{}: Index '{}' file '{}' should have positive size",
                context,
                index.name,
                file.path
            );
        }
    }
}

pub fn two_column_reader() -> impl arrow_array::RecordBatchReader + Send + 'static {
    lance_datagen::gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("payload", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(10), BatchCount::from(1))
}

#[derive(Debug, Clone, Copy)]
pub enum UnreadableIndexKind {
    NewerVersion,
    UnknownType,
    ReservedField,
}

/// Make `index_name` unreadable to this build, and give it the full fragment
/// coverage a real index of that name would have.
pub async fn hide_index_as(dataset: &mut Dataset, index_name: &str, kind: UnreadableIndexKind) {
    let current = dataset.load_indices_by_name(index_name).await.unwrap();
    assert_eq!(current.len(), 1);
    let mut unreadable = current.clone();
    match kind {
        UnreadableIndexKind::NewerVersion => {
            // Derive the unsupported version from what this build's
            // plugin actually reports as its max supported version,
            // rather than from the freshly created index's own version:
            // BTree now varies that baseline by whether the dataset uses
            // stable row ids (see BTREE_ROW_ID_DOMAIN_VERSION), so
            // `current + 1` no longer reliably lands past what this
            // build can read.
            let max_supported =
                crate::index::scalar::IndexDetails(current[0].index_details.clone().unwrap())
                    .index_version()
                    .unwrap();
            unreadable[0].index_version = (max_supported + 1) as i32;
        }
        UnreadableIndexKind::UnknownType => {
            unreadable[0].index_details = Some(Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/example.ForeignIndexDetails".to_string(),
                value: Vec::new(),
            }));
        }
        UnreadableIndexKind::ReservedField => {
            unreadable[0].fields = vec![lance_table::format::ROW_ID_FIELD_ID];
        }
    }
    unreadable[0].fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: unreadable,
            removed_indices: current,
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
}

/// Raise `index_name` past the version this build can read.
pub async fn hide_index_from_this_build(dataset: &mut Dataset, index_name: &str) {
    hide_index_as(dataset, index_name, UnreadableIndexKind::NewerVersion).await;
}

/// The readable companion every fixture below carries over `payload`.
pub const READABLE_INDEX: &str = "payload_idx";

/// A dataset carrying an unreadable BTree index over `id` beside an ordinary
/// readable BTree index over `payload`.
///
/// The readable companion is what makes the filter's selectivity visible:
/// with a single entry, "hid the one it cannot read" and "hid everything"
/// produce the same answer to every assertion in this module.
///
/// Both indices are committed untrained and given their coverage by hand.
/// Nothing here ever reads them, and training one would take a non-spillable
/// 40 MB reservation out of the session's shared 150 MB pool to sort ten
/// rows - three of those in flight at once is all the pool has room for.
pub async fn dataset_with_an_unreadable_index(
    uri: &str,
    index_name: &str,
    kind: UnreadableIndexKind,
) -> Dataset {
    // Stable row ids so the BTree indices this fixture creates are
    // address-domain (index_version 1): `hide_index_as`'s NewerVersion
    // kind computes an unsupported version as `current + 1`, which only
    // exceeds what this build supports starting from that baseline. On a
    // dataset without stable row ids, BTree is created at index_version 0
    // instead (see BTREE_ROW_ID_DOMAIN_VERSION), and 0 + 1 is this
    // build's own max supported version, not an unsupported one.
    let mut dataset = Dataset::write(
        two_column_reader(),
        uri,
        Some(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name(index_name.to_string())
        .train(false)
        .await
        .unwrap();
    hide_index_as(&mut dataset, index_name, kind).await;

    dataset
        .create_index_builder(&["payload"], IndexType::BTree, &btree_params)
        .name(READABLE_INDEX.to_string())
        .train(false)
        .await
        .unwrap();
    dataset
}

/// A dataset carrying an index whose version is newer than this build.
pub async fn dataset_with_an_index_from_a_newer_build(uri: &str, index_name: &str) -> Dataset {
    dataset_with_an_unreadable_index(uri, index_name, UnreadableIndexKind::NewerVersion).await
}

/// Indices the manifest itself carries, bypassing the usable-index filter.
pub async fn raw_manifest_indices(dataset: &Dataset) -> Vec<IndexMetadata> {
    lance_table::io::manifest::read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap()
}

/// The manifest entry named `name`, whole. Compare these, not names: a
/// carried-forward index that kept its name but lost its coverage or gained
/// a new uuid is exactly the corruption this suite exists to catch.
pub async fn manifest_index(dataset: &Dataset, name: &str) -> IndexMetadata {
    raw_manifest_indices(dataset)
        .await
        .into_iter()
        .find(|idx| idx.name == name)
        .unwrap_or_else(|| panic!("no index named {name} in the manifest"))
}

/// Sorted: a commit that replaces an entry appends the replacement, so
/// comparing in manifest order would break on which operation ran rather
/// than on what it did. Order is not meaningless in general - delta merging
/// selects a suffix of it - but no test here asserts on it.
pub async fn manifest_index_names(dataset: &Dataset) -> Vec<String> {
    let mut names = raw_manifest_indices(dataset)
        .await
        .into_iter()
        .map(|idx| idx.name)
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[derive(Debug)]
pub struct RequestIndexStoreWrapper {
    policy: Arc<Mutex<ProxyObjectStorePolicy>>,
    pub revoked: Arc<AtomicBool>,
    pub requests: Arc<Mutex<Vec<String>>>,
}

impl RequestIndexStoreWrapper {
    pub fn new(index_path: String) -> Self {
        let revoked = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let mut policy = ProxyObjectStorePolicy::new();
        let policy_revoked = revoked.clone();
        let policy_requests = requests.clone();
        policy.set_before_policy(
            "observe_index_reads",
            Arc::new(move |method, path| {
                if path.as_ref().contains(&index_path) {
                    policy_requests.lock().unwrap().push(method.to_owned());
                    if policy_revoked.load(Ordering::Relaxed) {
                        return Err(object_store::Error::Generic {
                            store: "RequestIndexStoreWrapper",
                            source: "request credentials have been revoked".into(),
                        }
                        .into());
                    }
                }
                Ok(())
            }),
        );
        Self {
            policy: Arc::new(Mutex::new(policy)),
            revoked,
            requests,
        }
    }
}

impl WrappingObjectStore for RequestIndexStoreWrapper {
    fn wrap(
        &self,
        _storage_prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(ProxyObjectStore::new(original, self.policy.clone()))
    }

    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}
