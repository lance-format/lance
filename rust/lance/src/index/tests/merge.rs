// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::support::*;
use super::*;

/// Premise guard, not a regression test: confirms that the broken shape
/// (`fields` truncated to the keyed field while `covering_fields` is
/// inherited unchanged) is actually rejected by `validate_covering_fields`,
/// and that the fixed shape (both fields carried through) is accepted.
/// Both shapes are hand-built here, so this exercises no `merge_segments`
/// code and would keep passing even if every impl regressed to the broken
/// shape. The real regression coverage is
/// `test_merge_existing_index_segments_preserves_covering_bitmap`,
/// `_btree`, `_vector`, and `_rtree` below, which merge real segments
/// through the actual `merge_segments` implementations.
#[test]
fn test_merged_covered_metadata_is_committable() {
    let source = IndexMetadata {
        uuid: Uuid::new_v4(),
        name: "covered".to_string(),
        fields: vec![7, 11],
        covering_fields: vec![11],
        dataset_version: 1,
        fragment_bitmap: Some(RoaringBitmap::from_iter([0u32])),
        index_details: None,
        index_version: 0,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };

    // The shape every merge_segments impl builds: keyed field only, carried
    // inherited from the source.
    let field_id = *source.fields.first().unwrap();
    let merged = IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: vec![field_id],
        ..source.clone()
    };
    assert!(
        merged.validate_covering_fields().is_err(),
        "this is the broken shape; if it validates, the premise has changed"
    );

    // What the fix must produce instead.
    let fixed = IndexMetadata {
        uuid: Uuid::new_v4(),
        ..source
    };
    fixed
        .validate_covering_fields()
        .expect("merged covered metadata must be committable");
    assert_eq!(fixed.fields, vec![7, 11]);
    assert_eq!(fixed.covering_fields, vec![11]);
}

/// Two shapes of the same bug, one per `merge_segments` family:
///
///   - **struct update** (bitmap, inverted, zonemap, label_list,
///     bloomfilter, fmindex): `fields` was truncated to `vec![field_id]`
///     while `..segments[0].clone()` re-supplied `covering_fields`,
///     producing an *uncommittable* merged segment.
///   - **explicit** (btree, ngram): `fields: vec![field_id],
///     covering_fields: vec![]` was built outright, *silently dropping* the
///     declaration instead.
#[rstest]
#[case::struct_update(BuiltinIndexType::Bitmap, IndexType::Bitmap)]
#[case::explicit(BuiltinIndexType::BTree, IndexType::BTree)]
#[tokio::test]
async fn test_merge_existing_index_segments_preserves_covering_scalar(
    #[case] builtin: BuiltinIndexType,
    #[case] index_type: IndexType,
) {
    let test_dir = TempStrDir::default();
    let mut dataset = write_two_int_column_dataset(&test_dir, 20).await;
    let id_field_id = dataset.schema().field("id").unwrap().id;
    let carried_field_id = dataset.schema().field("carried").unwrap().id;
    let fragment_ids = all_fragment_ids(&dataset);

    let params = ScalarIndexParams::for_builtin(builtin);
    assert_merge_preserves_covering(
        &mut dataset,
        "id",
        index_type,
        &params,
        fragment_ids,
        id_field_id,
        carried_field_id,
    )
    .await;
}

/// Segments that disagree on which columns they carry cannot be folded
/// into one coherent output declaration: there is no single
/// `covering_fields` that would describe the merged segment truthfully.
#[tokio::test]
async fn test_merge_existing_index_segments_rejects_mismatched_covering_fields() {
    let test_dir = TempStrDir::default();
    let mut dataset = write_two_int_column_dataset(&test_dir, 20).await;
    let id_field_id = dataset.schema().field("id").unwrap().id;
    let carried_field_id = dataset.schema().field("carried").unwrap().id;

    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
    let mut mismatched_segments = Vec::new();
    for (i, fragment) in dataset.get_fragments().into_iter().enumerate() {
        let segment = dataset
            .create_index_builder(&["id"], IndexType::Bitmap, &params)
            .name("mismatched_bitmap".to_string())
            .fragments(vec![fragment.id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
        // First segment declares `carried` covered; the second declares
        // nothing covered -- a genuine disagreement, not just a
        // reordering.
        if i == 0 {
            mismatched_segments.push(make_covered(segment, id_field_id, carried_field_id));
        } else {
            mismatched_segments.push(segment);
        }
    }

    let err = dataset
        .merge_existing_index_segments(mismatched_segments)
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("requires segments with identical fields"),
        "unexpected error: {err}"
    );
}

/// Vector shape: `crate::index::vector::ivf::merge_segments` already
/// preserves both `fields` and `covering_fields` via struct update, but
/// `merge_existing_index_segments`'s own tail used to unconditionally
/// overwrite `fields` to `vec![field_id]` afterwards, reproducing the
/// struct-update bug for every index type it dispatches to.
#[tokio::test]
async fn test_merge_existing_index_segments_preserves_covering_vector() {
    const DIMENSION: i32 = 4;
    let test_dir = TempStrDir::default();
    let mut dataset = write_fragmented_vector_dataset(&test_dir, DIMENSION).await;
    let vector_field_id = dataset.schema().field("vector").unwrap().id;
    let id_field_id = dataset.schema().field("id").unwrap().id;

    let batch = dataset
        .scan()
        .project(&["vector"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let vectors = batch
        .column_by_name("vector")
        .expect("vector column should exist")
        .as_fixed_size_list();
    let values = vectors.values().as_primitive::<Float32Type>();
    let centroids = train_kmeans::<Float32Type>(
        values,
        KMeansParams::new(None, 10, 1, DistanceType::L2),
        DIMENSION as usize,
        2,
        2,
    )
    .unwrap()
    .centroids
    .as_primitive::<Float32Type>()
    .clone();
    let centroids =
        Arc::new(FixedSizeListArray::try_new_from_values(centroids, DIMENSION).unwrap());
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(2, centroids).unwrap(),
    );

    let fragment_ids = all_fragment_ids(&dataset).into_iter().take(2).collect();

    assert_merge_preserves_covering(
        &mut dataset,
        "vector",
        IndexType::Vector,
        &params,
        fragment_ids,
        vector_field_id,
        id_field_id,
    )
    .await;
}

/// Struct-update shape, sweep-found in `rtree.rs`: same bug as the
/// bitmap/inverted/zonemap/label_list/bloomfilter/fmindex family, not
/// called out in the original brief but reachable from the same
/// `all_rtree` dispatch branch this function tests.
#[cfg(feature = "geo")]
#[tokio::test]
async fn test_merge_existing_index_segments_preserves_covering_rtree() {
    const ROWS_PER_FRAGMENT: i32 = 20;
    const FRAGMENTS: i32 = 2;

    let (schema, batches) = crate::utils::test::geo::batches(ROWS_PER_FRAGMENT, FRAGMENTS);
    let test_dir = TempStrDir::default();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT as usize,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let id_field_id = dataset.schema().field("id").unwrap().id;
    let geometry_field_id = dataset.schema().field("geometry").unwrap().id;
    let fragment_ids = all_fragment_ids(&dataset);

    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::RTree);
    assert_merge_preserves_covering(
        &mut dataset,
        "geometry",
        IndexType::RTree,
        &params,
        fragment_ids,
        geometry_field_id,
        id_field_id,
    )
    .await;
}

/// RTree segments staged before a deferred compaction: merged coverage moves
/// to the compacted fragment only if the segments cover the whole group.
#[cfg(feature = "geo")]
#[rstest]
#[case::whole_group(true)]
#[case::part_of_the_group(false)]
#[tokio::test]
async fn test_rtree_merge_coverage_across_a_deferred_compaction(#[case] whole_group: bool) {
    const ROWS_PER_FRAGMENT: i32 = 10;
    const FRAGMENTS: i32 = 3;

    let test_dir = TempStrDir::default();
    let (mut dataset, params) = crate::utils::test::geo::dataset_with_committed_rtree_index(
        test_dir.as_str(),
        ROWS_PER_FRAGMENT,
        FRAGMENTS,
    )
    .await;
    let mut covered = all_fragment_ids(&dataset);
    if !whole_group {
        covered.pop();
    }
    let staged =
        crate::utils::test::geo::stage_rtree_segments(&mut dataset, &params, covered).await;
    compact_into_one_fragment(&mut dataset, ROWS_PER_FRAGMENT, FRAGMENTS).await;
    let surviving = all_fragment_ids(&dataset);

    let merged = dataset.merge_existing_index_segments(staged).await.unwrap();
    let coverage = merged
        .fragment_bitmap
        .as_ref()
        .expect("a merged segment records what it covers");

    if !whole_group {
        assert!(
            coverage.is_empty(),
            "a straddling merge claimed {coverage:?}; the rewritten fragment holds \
             unindexed rows, which would then be dropped from results instead of scanned"
        );
        return;
    }
    assert_eq!(
        coverage.iter().collect::<Vec<_>>(),
        surviving,
        "coverage must name the fragment the rewrite produced"
    );
    // The claim is only correct if the index holds every row of the fragment.
    assert_eq!(
        merged_rtree_item_count(&dataset, &merged).await,
        (ROWS_PER_FRAGMENT * FRAGMENTS) as u64,
        "the merged index must hold every row of the fragment it claims"
    );
}
