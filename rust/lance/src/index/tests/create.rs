// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::support::*;
use super::*;

#[tokio::test]
async fn test_recreate_index() {
    const DIM: i32 = 8;
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
            true,
        ),
        Field::new(
            "o",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
            true,
        ),
    ]));
    let data = generate_random_array(2048 * DIM as usize);
    let batches: Vec<RecordBatch> = vec![
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(FixedSizeListArray::try_new_from_values(data.clone(), DIM).unwrap()),
                Arc::new(FixedSizeListArray::try_new_from_values(data, DIM).unwrap()),
            ],
        )
        .unwrap(),
    ];

    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    let params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 2);
    dataset
        .create_index(&["v"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset
        .create_index(&["o"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    // Create index again
    dataset
        .create_index(&["v"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    // Can not overwrite an index on different columns.
    assert!(
        dataset
            .create_index(
                &["v"],
                IndexType::Vector,
                Some("o_idx".to_string()),
                &params,
                true,
            )
            .await
            .is_err()
    );
}

#[tokio::test]
async fn test_unknown_index_type_does_not_block_queries_or_optimization() {
    let reader = gen_batch()
        .col("vector", array::rand_vec::<Float32Type>(Dimension::from(8)))
        .col("number", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(64), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
    dataset
        .create_index(
            &["number"],
            IndexType::BTree,
            Some("number_idx".to_string()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    let appended = gen_batch()
        .col("vector", array::rand_vec::<Float32Type>(Dimension::from(8)))
        .col("number", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(32), BatchCount::from(1));
    dataset.append(appended, None).await.unwrap();
    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("number_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 32);

    let field_id = dataset.schema().field("vector").unwrap().id;
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let mut foreign_segment = write_vector_segment_metadata(
        &dataset,
        "foreign_idx",
        field_id,
        Uuid::new_v4(),
        fragment_ids,
        b"opaque external index",
    )
    .await;
    foreign_segment.index_details = Some(Arc::new(prost_types::Any {
        type_url: "type.googleapis.com/example.MyVectorIndexDetails".to_string(),
        value: Vec::new(),
    }));
    foreign_segment.index_version = 1;
    dataset
        .commit_existing_index_segments("foreign_idx", "vector", vec![foreign_segment])
        .await
        .unwrap();

    assert!(
        dataset
            .load_indices_by_name("foreign_idx")
            .await
            .unwrap()
            .is_empty(),
        "an index with no reader must not enter the usable-index view"
    );
    assert_eq!(
        load_all_indices_by_name(&dataset, "foreign_idx")
            .await
            .unwrap()
            .len(),
        1,
        "hiding an unusable index must not erase its manifest metadata"
    );

    let query = Float32Array::from(vec![0.5_f32; 8]);
    let mut scanner = dataset.scan();
    scanner.nearest("vector", &query, 5).unwrap();
    assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 5);

    dataset.optimize_indices(&Default::default()).await.unwrap();
    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("number_idx").await.unwrap()).unwrap();
    assert_eq!(stats["num_unindexed_rows"], 0);
    assert_eq!(
        load_all_indices_by_name(&dataset, "foreign_idx")
            .await
            .unwrap()
            .len(),
        1,
        "optimizing supported indices must preserve the opaque segment"
    );
}

#[tokio::test]
async fn test_create_index_too_small_for_pq() {
    let test_dir = TempStrDir::default();
    let dimensions = 1536;

    let field = Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimensions,
        ),
        false,
    );

    let schema = Arc::new(Schema::new(vec![field]));
    let float_arr = generate_random_array(100 * dimensions as usize);

    let vectors =
        arrow_array::FixedSizeListArray::try_new_from_values(float_arr, dimensions).unwrap();
    let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vectors)]).unwrap();
    let reader = RecordBatchIterator::new(
        vec![record_batch.clone()].into_iter().map(Ok),
        schema.clone(),
    );

    let test_uri = &test_dir;
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    let params = VectorIndexParams::ivf_pq(1, 8, 96, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .expect("a table too small to train on still accepts an index");

    // The definition is there and covers nothing: 100 rows cannot train a
    // 256-code quantizer, so there is nothing to cover yet.
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert!(segment_covers_nothing(&indices[0]));
    // Every fragment is still unindexed, which is what optimize will pick
    // up once the column can train.
    let unindexed = dataset.unindexed_fragments("vector_idx").await.unwrap();
    assert_eq!(unindexed.len(), dataset.get_fragments().len());
}

/// More partitions requested than the data supports trains fewer of them.
///
/// A partition wants a codebook's worth of vectors, so 300 vectors support
/// one partition however many are asked for.
#[tokio::test]
async fn test_create_index_with_more_partitions_than_rows() {
    let test_dir = tempfile::tempdir().unwrap();
    let rows = 300;
    let mut dataset = small_vector_dataset(test_dir.path(), rows).await;

    let params = VectorIndexParams::ivf_pq(1000, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .expect("more partitions than rows should reduce partitions, not fail");

    // Trained, not degraded: 300 rows clear the 256-code PQ floor.
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert!(!segment_covers_nothing(&indices[0]));

    // 300 / 256 = 1, not the 1000 requested.
    assert_eq!(trained_partitions(&dataset).await, vec![1]);

    // The target recorded is the 300 vectors the one partition holds, not
    // 300 / 1000, which would have optimize split it apart.
    let details = dataset.load_indices().await.unwrap()[0]
        .index_details
        .clone()
        .unwrap();
    assert_eq!(target_partition_size_from_details(&details), Some(300));
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    assert_eq!(trained_partitions(&dataset).await, vec![1]);
}

/// A table indexed while empty trains on the first append-mode optimize
/// that has data behind it.
///
/// Writing rows never touches the index, so the optimize is where the
/// definition becomes a real index — and append is the mode scheduled
/// maintenance uses, so it has to be the mode that gets there.
#[tokio::test]
async fn test_append_mode_trains_a_definition_once_data_arrives() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 0).await;

    let params = VectorIndexParams::ivf_pq(2, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    let mut dataset = append_vectors(test_dir.path(), 3000).await;
    assert_eq!(dataset.count_rows(None).await.unwrap(), 3000);
    let indices = dataset.load_indices().await.unwrap();
    assert!(
        segment_covers_nothing(&indices[0]),
        "writing rows leaves the index a definition"
    );

    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices.len(),
        1,
        "training supersedes the definition rather than adding a delta to it"
    );
    assert!(
        !segment_covers_nothing(&indices[0]),
        "3,000 vectors clear the floor, so the append trains the column"
    );
    // Sized from the 3,000 vectors present, not the 2 the request named.
    assert_eq!(trained_partitions(&dataset).await, vec![1]);
}

/// The cap reduces a requested count only while the data cannot support it.
#[tokio::test]
async fn test_create_index_partition_cap_follows_the_data() {
    // 8 partitions want 8 * 256 = 2048 vectors: 1000 falls inside the band
    // where the count is reduced, 3000 clears it.
    for (rows, expected) in [(1000, 3), (3000, 8)] {
        let test_dir = tempfile::tempdir().unwrap();
        let mut dataset = small_vector_dataset(test_dir.path(), rows).await;

        let params = VectorIndexParams::ivf_pq(8, 8, 4, DistanceType::L2, 1);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        assert_eq!(
            trained_partitions(&dataset).await,
            vec![expected],
            "{rows} vectors"
        );
    }
}

/// The cap counts the vectors IVF fits a centroid on, not codebook entries.
#[tokio::test]
async fn test_partition_cap_ignores_the_codebook_size() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 300).await;

    // A 4-bit codebook holds 16 entries, but a partition still wants the
    // 256 vectors IVF training samples for one centroid.
    let params = VectorIndexParams::ivf_pq(8, 4, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // 300 / 256 = 1.
    assert_eq!(trained_partitions(&dataset).await, vec![1]);
}

/// Sampling fewer vectors per centroid makes more partitions supportable.
#[tokio::test]
async fn test_partition_cap_follows_the_configured_sample_rate() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 300).await;

    let params = VectorIndexParams::with_ivf_pq_params(
        DistanceType::L2,
        IvfBuildParams {
            num_partitions: Some(8),
            sample_rate: 64,
            ..Default::default()
        },
        PQBuildParams {
            num_sub_vectors: 4,
            num_bits: 8,
            ..Default::default()
        },
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // 300 / 64 = 4.
    assert_eq!(trained_partitions(&dataset).await, vec![4]);
}

/// A definition records no partition count, so the index the table grows
/// into is sized from the data present when it finally trains. A count
/// asked for on a table too small to train it does not survive the wait.
#[tokio::test]
async fn test_deferred_index_sizes_from_the_data_it_trains_on() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 100).await;

    let params = VectorIndexParams::ivf_pq(8, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert!(
        segment_covers_nothing(&indices[0]),
        "100 vectors cannot train a 256-code quantizer, so nothing is covered yet"
    );

    // 3100 vectors would clear 8 * 256, but no count was recorded, so
    // training derives one from the data: 3100 / 8192 -> 1.
    let mut dataset = append_vectors(test_dir.path(), 3000).await;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    assert_eq!(trained_partitions(&dataset).await, vec![1]);
}

/// Appending to an index that is still a definition, on a table that still
/// cannot train, is a no-op rather than an error, and stays one when
/// repeated.
#[tokio::test]
async fn test_append_to_a_definition_that_still_cannot_train() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 100).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // Twice, so the second call sees whatever the first one left behind.
    for attempt in 0..2 {
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap_or_else(|error| {
                panic!("append {attempt} on a definition must not fail: {error}")
            });
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert!(
            segment_covers_nothing(&indices[0]),
            "100 vectors still cannot train a 256-code quantizer"
        );
    }
}

/// Compaction leaves an index that covers nothing alone. It has no file to
/// remap, and an empty fragment bitmap cannot intersect a rewrite group, so
/// the remapper skips it instead of opening a file that is not there.
#[tokio::test]
async fn test_compaction_skips_a_definition_only_index() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 100).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // A second fragment, so compaction has two to rewrite into one.
    let mut dataset = append_vectors(test_dir.path(), 100).await;
    assert_eq!(dataset.get_fragments().len(), 2);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1000,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 200);
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert!(
        segment_covers_nothing(&indices[0]),
        "the index should still be a definition covering nothing"
    );
}

/// The floor counts vectors, not rows.
///
/// A column that is mostly null has plenty of rows and nothing to train on,
/// so it has to degrade rather than hand too few vectors to the quantizer.
#[tokio::test]
async fn test_create_index_counts_vectors_not_rows() {
    let test_dir = tempfile::tempdir().unwrap();
    // 1000 rows clear the floor; the 100 actual vectors do not.
    let mut dataset = partly_null_vector_dataset(test_dir.path(), 1000, 100).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .expect("a mostly-null column must not be handed to the quantizer");

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert!(
        segment_covers_nothing(&indices[0]),
        "100 vectors cannot train a 256-code quantizer"
    );
}

/// A vector query works while the index is still only a definition.
///
/// The index covers no fragments, so every row is unindexed and the search
/// has to answer from the table itself rather than opening an index file
/// that was never written.
#[tokio::test]
async fn test_vector_query_against_a_definition_only_index() {
    let test_dir = tempfile::tempdir().unwrap();
    let rows = 100;
    let mut dataset = small_vector_dataset(test_dir.path(), rows).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();
    assert!(segment_covers_nothing(
        &dataset.load_indices().await.unwrap()[0]
    ));

    let query = vec![0.0_f32; 16];
    let results = dataset
        .scan()
        .nearest("vector", &Float32Array::from(query), 5)
        .unwrap()
        .try_into_batch()
        .await
        .expect("a query must not open an index file that was never written");
    assert_eq!(results.num_rows(), 5);
}

/// `fast_search` against an index that is still a definition finds nothing.
///
/// It restricts the search to what the index covers, and a definition
/// covers no rows, so there is nothing to return rather than a fallback
/// scan of the table.
#[tokio::test]
async fn test_fast_search_against_a_definition_only_index() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 100).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    let query = Float32Array::from(vec![0.0_f32; 16]);
    let results = dataset
        .scan()
        .nearest("vector", &query, 5)
        .unwrap()
        .fast_search()
        .try_into_batch()
        .await
        .expect("fast_search must not fail on an index that covers nothing");
    assert_eq!(results.num_rows(), 0);
}

/// An index type without a codebook trains on a table a codebook could not.
///
/// Only PQ needs a vector per code; RQ needs a pair of values and flat
/// storage needs none, so 100 vectors are enough for these to cover the
/// table rather than degrade to a definition.
#[rstest]
#[case::ivf_flat(VectorIndexParams::ivf_flat(1, DistanceType::L2))]
#[case::ivf_rq(VectorIndexParams::ivf_rq(1, 8, DistanceType::L2))]
#[tokio::test]
async fn test_create_index_without_a_codebook_trains_on_a_small_table(
    #[case] params: VectorIndexParams,
) {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 100).await;

    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .expect("an index with no codebook has no row floor to clear");

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert!(
        !segment_covers_nothing(&indices[0]),
        "should have trained and covered the table"
    );
}

/// A multivector row holds a list, so the floor counts the vectors in the
/// lists rather than the rows that carry them.
#[tokio::test]
async fn test_multivector_floor_counts_vectors_not_rows() {
    let sparse = {
        let mut lengths = vec![0_usize; 100];
        lengths[0] = 10;
        lengths
    };
    // Lists per row, whether that trains, and the count that decides it.
    let cases: [(Vec<usize>, bool, &str); 4] = [
        (vec![10; 100], true, "100 rows of 10 vectors give 1,000"),
        (vec![2; 10], false, "10 rows of 2 vectors give 20"),
        (sparse, false, "one row holding 10 vectors still gives 10"),
        (vec![0; 100], false, "empty lists give none"),
    ];

    for (lengths, trains, why) in cases {
        let test_dir = tempfile::tempdir().unwrap();
        let mut dataset = multivector_dataset_with(test_dir.path(), &lengths).await;

        // Multivector columns are cosine-only.
        let params = VectorIndexParams::ivf_pq(1, 8, 4, DistanceType::Cosine, 1);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "{why}: a table below the floor takes the index rather than failing: {error}"
                )
            });

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1, "{why}");
        let covers_nothing = segment_covers_nothing(&indices[0]);
        assert_eq!(!covers_nothing, trains, "{why}");
    }
}

/// The partition cap counts a multivector row's whole list too.
///
/// 1,000 vectors support three partitions; counting the 100 rows instead
/// would allow only one.
#[tokio::test]
async fn test_partition_cap_counts_multivector_lists() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = multivector_dataset(test_dir.path(), 100, 10).await;

    let params = VectorIndexParams::ivf_pq(8, 8, 4, DistanceType::Cosine, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // 1,000 / 256 = 3, capped from the 8 requested.
    assert_eq!(trained_partitions(&dataset).await, vec![3]);
}

/// The partition cap counts vectors, so blanks cannot inflate it.
///
/// 3,000 rows holding 300 vectors support one partition, not the eight a
/// row count would appear to allow.
#[tokio::test]
async fn test_partition_cap_ignores_null_rows() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = partly_null_vector_dataset(test_dir.path(), 3000, 300).await;

    let params = VectorIndexParams::ivf_pq(8, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    // 300 / 256 = 1.
    assert_eq!(trained_partitions(&dataset).await, vec![1]);
}

/// Deleting every row leaves the index defined.
///
/// The definition is the user's declaration, not a property of the data, so
/// emptying the table must not withdraw it — otherwise reloading a table
/// silently drops the indexes it was created with.
#[tokio::test]
async fn test_vector_index_survives_deleting_all_rows() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 1024).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 8, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);

    dataset.delete("true").await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), 0);

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1, "the index definition must survive");

    // And it is still there after a reopen, so it lives in the manifest
    // rather than in whatever the session happened to hold.
    let reopened = Dataset::open(test_dir.path().to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(reopened.load_indices().await.unwrap().len(), 1);
}

/// Retraining an index that is still only a definition does nothing.
///
/// A definition has no segment to open, and the retrain path resolves that
/// before it reaches for one.
#[tokio::test]
async fn test_retrain_an_index_that_is_still_a_definition() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 100).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 4, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    dataset
        .optimize_indices(&OptimizeOptions::retrain())
        .await
        .expect("retraining a definition must not look for a segment to open");

    // Still 100 vectors, so still a definition.
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert!(segment_covers_nothing(&indices[0]));
}

/// Updating every row leaves the index defined, and optimizing still runs.
#[tokio::test]
async fn test_vector_index_survives_updating_all_rows() {
    let test_dir = tempfile::tempdir().unwrap();
    let mut dataset = small_vector_dataset(test_dir.path(), 1024).await;

    let params = VectorIndexParams::ivf_pq(1, 8, 8, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    let update = crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
        .set("vector", "vector")
        .unwrap()
        .build()
        .unwrap();
    let updated = update.execute().await.unwrap();
    let mut dataset = updated.new_dataset.as_ref().clone();

    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .expect("optimizing after a full update must not fail");
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
}

/// A table with no rows at all takes an index, which is the case every
/// other database allows and the one a fresh or reloaded table is in.
#[tokio::test]
async fn test_create_vector_index_on_empty_table() {
    let test_dir = tempfile::tempdir().unwrap();
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
    let reader = RecordBatchIterator::new(
        Vec::<std::result::Result<RecordBatch, arrow_schema::ArrowError>>::new(),
        schema.clone(),
    );
    let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let params = VectorIndexParams::ivf_pq(1, 8, 96, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .expect("an empty table still accepts an index");

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert!(segment_covers_nothing(&indices[0]));

    // Reopening has to find the same definition: it is carried by the
    // manifest, not by a file on disk.
    let reopened = Dataset::open(test_dir.path().to_str().unwrap())
        .await
        .unwrap();
    let indices = reopened.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(indices[0].name, "vector_idx");
}

#[tokio::test]
async fn test_create_bitmap_index() {
    let test_dir = TempStrDir::default();
    let field = Field::new("tag", DataType::Utf8, false);
    let schema = Arc::new(Schema::new(vec![field]));
    let array = StringArray::from_iter_values((0..128).map(|i| ["a", "b", "c"][i % 3]));
    let record_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
    let reader = RecordBatchIterator::new(
        vec![record_batch.clone()].into_iter().map(Ok),
        schema.clone(),
    );

    let test_uri = &test_dir;
    let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();
    dataset
        .create_index(
            &["tag"],
            IndexType::Bitmap,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let index = dataset
        .open_generic_index("tag", &indices[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(index.index_type(), IndexType::Bitmap);
}

#[rstest]
#[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
#[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
#[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
#[tokio::test]
async fn test_create_empty_scalar_index(
    #[case] column_name: &str,
    #[case] index_type: IndexType,
    #[case] params: Box<dyn IndexParams>,
) {
    use lance_datagen::{BatchCount, ByteCount, RowCount, array};

    // Create dataset with scalar and text columns (no vector column needed)
    let reader = lance_datagen::gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("text", array::rand_utf8(ByteCount::from(10), false))
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

    // Create an empty index with train=false
    // Test using IntoFuture - can await directly without calling .execute()
    dataset
        .create_index_builder(&[column_name], index_type, params.as_ref())
        .name("index".to_string())
        .train(false)
        .await
        .unwrap();

    // Verify we can get index statistics
    let stats = dataset.index_statistics("index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        stats["num_indexed_rows"], 0,
        "Empty index should have zero indexed rows"
    );
    let descriptions = dataset.describe_indices(None).await.unwrap();
    assert_eq!(descriptions.len(), 1);
    let current_fragment_count = u64::try_from(dataset.fragments().len()).unwrap();
    let fragment_bitmap_size_bytes = u64::try_from(
        dataset.load_indices().await.unwrap()[0]
            .fragment_bitmap_serialized_size()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        descriptions[0].fragment_coverage(),
        Some(&IndexFragmentCoverage {
            covered_fragment_count: 0,
            current_fragment_count,
            missing_fragment_count: current_fragment_count,
            stale_fragment_count: 0,
            fragment_bitmap_size_bytes,
        }),
        "an empty built-in index has known zero coverage"
    );

    // Append new data using lance_datagen
    let append_reader = lance_datagen::gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("text", array::rand_utf8(ByteCount::from(10), false))
        .into_reader_rows(RowCount::from(50), BatchCount::from(1));

    dataset.append(append_reader, None).await.unwrap();

    // Critical test: Verify the empty index is still present after append
    let indices_after_append = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_append.len(),
        1,
        "Index should be retained after append for index type {:?}",
        index_type
    );

    let stats = dataset.index_statistics("index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        stats["num_indexed_rows"], 0,
        "Empty index should still have zero indexed rows after append"
    );

    // Test optimize_indices with empty index
    dataset.optimize_indices(&Default::default()).await.unwrap();

    // Verify the index still exists after optimization
    let indices_after_optimize = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_optimize.len(),
        1,
        "Index should still exist after optimization"
    );

    // Check index statistics after optimization
    let stats = dataset.index_statistics("index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        stats["num_unindexed_rows"], 0,
        "Empty index should indexed all rows"
    );
}

/// Test that scalar indices are retained after deleting all data from a table.
///
/// This test verifies that when we:
/// 1. Create a table with data
/// 2. Add a scalar index with train=true
/// 3. Delete all data in the table
/// The index remains available on the table.
#[rstest]
#[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
#[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
#[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
#[tokio::test]
async fn test_scalar_index_retained_after_delete_all(
    #[case] column_name: &str,
    #[case] index_type: IndexType,
    #[case] params: Box<dyn IndexParams>,
) {
    use lance_datagen::{BatchCount, ByteCount, RowCount, array};

    // Create dataset with initial data
    let reader = lance_datagen::gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("text", array::rand_utf8(ByteCount::from(10), false))
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

    // Create index with train=true (normal index with data)
    dataset
        .create_index_builder(&[column_name], index_type, params.as_ref())
        .name("index".to_string())
        .train(true)
        .await
        .unwrap();

    // Verify index was created and has indexed rows
    let stats = dataset.index_statistics("index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        stats["num_indexed_rows"], 100,
        "Index should have indexed all 100 rows"
    );

    // Verify index is being used in queries before delete
    let plan = if column_name == "text" {
        // Use full-text search for inverted index
        dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("test".to_string()))
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap()
    } else {
        // Use equality filter for btree/bitmap indices
        dataset
            .scan()
            .filter(format!("{} = 50", column_name).as_str())
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap()
    };
    // Verify index is being used before delete
    assert_index_usage(&plan, column_name, true, "before delete");

    let indexes = dataset.load_indices().await.unwrap();
    let original_index = indexes[0].clone();

    // Delete all rows from the table
    dataset.delete("true").await.unwrap();

    // Verify table is empty
    let row_count = dataset.count_rows(None).await.unwrap();
    assert_eq!(row_count, 0, "Table should be empty after delete all");

    // Critical test: Verify the index still exists after deleting all data
    let indices_after_delete = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_delete.len(),
        1,
        "Index should be retained after deleting all data"
    );
    assert_eq!(
        indices_after_delete[0].name, "index",
        "Index name should remain the same after delete"
    );

    // Critical test: Verify the fragment bitmap is empty after delete
    let index_after_delete = &indices_after_delete[0];
    let effective_bitmap = index_after_delete
        .effective_fragment_bitmap(&dataset.fragment_bitmap)
        .unwrap();
    assert!(
        effective_bitmap.is_empty(),
        "Effective bitmap should be empty after deleting all data"
    );
    assert_eq!(
        index_after_delete.fragment_bitmap, original_index.fragment_bitmap,
        "Fragment bitmap should remain the same after delete"
    );

    // Verify we can still get index statistics
    let stats = dataset.index_statistics("index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        stats["num_indexed_rows"], 0,
        "Index should now report zero indexed rows after delete all"
    );
    assert_eq!(
        stats["num_unindexed_rows"], 0,
        "Index should report zero unindexed rows after delete all"
    );
    assert_eq!(
        stats["num_indexed_fragments"], 0,
        "Index should report zero indexed fragments after delete all"
    );
    assert_eq!(
        stats["num_unindexed_fragments"], 0,
        "Index should report zero unindexed fragments after delete all"
    );

    // Verify index is NOT being used in queries after delete (empty bitmap)
    if column_name == "text" {
        // The retained index has no live fragments, so an empty snapshot
        // needs no index scan to return an empty FTS result.
        let mut scan = dataset.scan();
        scan.project(&[column_name])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("test".to_string()))
            .unwrap();
        let plan_after_delete = scan.explain_plan(false).await.unwrap();
        assert_index_usage(
            &plan_after_delete,
            column_name,
            false,
            "after delete (empty bitmap)",
        );
        assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 0);
    } else {
        // Use equality filter for btree/bitmap indices
        let _plan_after_delete = dataset
            .scan()
            .filter(format!("{} = 50", column_name).as_str())
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        // Verify index is NOT being used after delete (empty bitmap)
        assert_index_usage(
            &_plan_after_delete,
            column_name,
            false,
            "after delete (empty bitmap)",
        );
    }

    // Test that we can append new data and the index is still there
    let append_reader = lance_datagen::gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("text", array::rand_utf8(ByteCount::from(10), false))
        .into_reader_rows(RowCount::from(50), BatchCount::from(1));

    dataset.append(append_reader, None).await.unwrap();

    // Verify index still exists after append
    let indices_after_append = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_append.len(),
        1,
        "Index should still exist after appending to empty table"
    );
    let stats = dataset.index_statistics("index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        stats["num_indexed_rows"], 0,
        "Index should now report zero indexed rows after data is added"
    );

    // Test optimize_indices after delete all
    dataset.optimize_indices(&Default::default()).await.unwrap();

    // Verify index still exists after optimization
    let indices_after_optimize = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_optimize.len(),
        1,
        "Index should still exist after optimization following delete all"
    );

    // Verify we can still get index statistics after optimization
    let stats = dataset.index_statistics("index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        stats["num_indexed_rows"],
        dataset.count_rows(None).await.unwrap(),
        "Index should now cover all newly added rows after optimization"
    );
}

/// Test that scalar indices are retained after updating rows in a table.
///
/// This test verifies that when we:
/// 1. Create a table with data
/// 2. Add a scalar index with train=true
/// 3. Update rows in the table
/// The index remains available on the table.
#[rstest]
#[case::btree("i", IndexType::BTree, Box::new(ScalarIndexParams::default()))]
#[case::bitmap("i", IndexType::Bitmap, Box::new(ScalarIndexParams::default()))]
#[case::inverted("text", IndexType::Inverted, Box::new(InvertedIndexParams::default()))]
#[tokio::test]
async fn test_scalar_index_retained_after_update(
    #[case] column_name: &str,
    #[case] index_type: IndexType,
    #[case] params: Box<dyn IndexParams>,
) {
    use crate::dataset::UpdateBuilder;
    use lance_datagen::{BatchCount, ByteCount, RowCount, array};

    // Create dataset with initial data
    let reader = lance_datagen::gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("text", array::rand_utf8(ByteCount::from(10), false))
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, "memory://test", None).await.unwrap();

    // Create index with train=true (normal index with data)
    dataset
        .create_index_builder(&[column_name], index_type, params.as_ref())
        .name("index".to_string())
        .train(true)
        .await
        .unwrap();

    // Verify index was created and has indexed rows
    let stats = dataset.index_statistics("index").await.unwrap();
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(
        stats["num_indexed_rows"], 100,
        "Index should have indexed all 100 rows"
    );

    // Verify index is being used in queries before update
    let plan = if column_name == "text" {
        // Use full-text search for inverted index
        dataset
            .scan()
            .project(&[column_name])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("test".to_string()))
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap()
    } else {
        // Use equality filter for btree/bitmap indices
        dataset
            .scan()
            .filter(format!("{} = 50", column_name).as_str())
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap()
    };
    // Verify index is being used before update
    assert_index_usage(&plan, column_name, true, "before update");

    // Update some rows - update first 50 rows
    let update_result = UpdateBuilder::new(Arc::new(dataset))
        .set("i", "i + 1000")
        .unwrap()
        .set("text", "'updated_' || text")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();

    let mut dataset = update_result.new_dataset.as_ref().clone();

    // Verify row count remains the same
    let row_count = dataset.count_rows(None).await.unwrap();
    assert_eq!(row_count, 100, "Row count should remain 100 after update");

    // Critical test: Verify the index still exists after updating data
    let indices_after_update = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_update.len(),
        1,
        "Index should be retained after updating rows"
    );

    // Critical test: Verify the effective fragment bitmap is empty after update
    let indices = dataset.load_indices().await.unwrap();
    let index = &indices[0];
    let effective_bitmap = index
        .effective_fragment_bitmap(&dataset.fragment_bitmap)
        .unwrap();
    assert!(
        effective_bitmap.is_empty(),
        "Effective fragment bitmap should be empty after updating all data"
    );

    // Verify we can still get index statistics
    let stats_after_update = dataset.index_statistics("index").await.unwrap();
    let stats_after_update: serde_json::Value = serde_json::from_str(&stats_after_update).unwrap();

    // The index should still be available
    assert_eq!(
        stats_after_update["num_indexed_rows"], 0,
        "Index statistics should be zero after update, as it is not re-trained"
    );

    // Verify index behavior in queries after update (empty bitmap)
    if column_name == "text" {
        // Inverted indexes will still appear to be used in FTS queries even with empty bitmaps.
        // This is because FTS queries require an index to exist, and the query execution
        // will handle the empty bitmap appropriately by falling back to scanning unindexed data.
        // TODO: once metrics are working on FTS queries, we can check the
        // analyze plan output instead for actual index usage statistics.
        let _plan_after_update = dataset
            .scan()
            .project(&[column_name])
            .unwrap()
            .full_text_search(FullTextSearchQuery::new("test".to_string()))
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert_index_usage(
            &_plan_after_update,
            column_name,
            true, // FTS indices always appear in the plan, even with empty bitmaps
            "after update (empty bitmap)",
        );
    } else {
        // Use equality filter for btree/bitmap indices
        let _plan_after_update = dataset
            .scan()
            .filter(format!("{} = 50", column_name).as_str())
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        // With immutable bitmaps, index is still used even with empty effective bitmap
        // The prefilter will handle non-existent fragments
        assert_index_usage(
            &_plan_after_update,
            column_name,
            false,
            "after update (empty effective bitmap)",
        );
    }

    // Test that we can optimize indices after update
    dataset.optimize_indices(&Default::default()).await.unwrap();

    // Verify index still exists after optimization
    let indices_after_optimize = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices_after_optimize.len(),
        1,
        "Index should still exist after optimization following update"
    );

    let stats_after_optimization = dataset.index_statistics("index").await.unwrap();
    let stats_after_optimization: serde_json::Value =
        serde_json::from_str(&stats_after_optimization).unwrap();

    // The index should still be available
    assert_eq!(
        stats_after_optimization["num_unindexed_rows"], 0,
        "Index should have zero unindexed rows after optimization"
    );
}
