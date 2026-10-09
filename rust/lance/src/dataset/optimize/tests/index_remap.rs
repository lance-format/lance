// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the index remap contract area.

use super::*;

#[tokio::test]
async fn test_stable_row_indices() {
    // Validate behavior of indices after compaction with stable row ids.
    let mut data_gen = BatchGenerator::new()
        .col(Box::new(
            RandomVector::new().vec_width(16).named("vec".to_owned()),
        ))
        .col(Box::new(IncrementingInt32::new().named("i".to_owned())));
    let mut dataset = Dataset::write(
        data_gen.batch(500),
        "memory://test/table",
        Some(WriteParams {
            enable_stable_row_ids: true,
            max_rows_per_file: 100, // 5 files
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Delete first 110 rows so rowids != final rowaddrs
    // First 100 rows deletes first file. Next 10 deletes part of second
    // file, so we will trigger the with deletions code path.
    dataset.delete("i < 110").await.unwrap();

    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("scalar".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vector".into()),
            &params,
            false,
        )
        .await
        .unwrap();

    async fn index_by_name(dataset: &Dataset) -> HashMap<String, Uuid> {
        let mut by_name = HashMap::new();
        for index in dataset.load_indices().await.unwrap().iter() {
            if let Some(existing) = by_name.insert(index.name.clone(), index.uuid) {
                panic!(
                    "index name {:?} is not unique: segments {existing} and {} share it",
                    index.name, index.uuid
                );
            }
        }
        by_name
    }
    let indices = index_by_name(&dataset).await;

    async fn vector_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();

        let query = Float32Array::from(vec![0.0f32; 16]);
        scanner
            .nearest("vec", &query, 10)
            .unwrap()
            .project(&["i"])
            .unwrap();

        scanner.try_into_batch().await.unwrap()
    }

    async fn scalar_query(dataset: &Dataset) -> RecordBatch {
        let mut scanner = dataset.scan();

        scanner.filter("i = 100").unwrap().project(&["i"]).unwrap();

        scanner.try_into_batch().await.unwrap()
    }

    let before_vec_result = vector_query(&dataset).await;
    let before_scalar_result = scalar_query(&dataset).await;

    let options = CompactionOptions {
        target_rows_per_fragment: 180,
        ..Default::default()
    };
    let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

    // The vector index is row-id-domain: stable row ids survive the
    // rewrite on their own, so it follows its data for free and keeps its
    // uuid. The scalar index defaults to BTree, which is address-domain
    // under stable row ids and supports remap, so eager compaction remaps
    // it in place (a new uuid, but full coverage of the live fragments) --
    // unlike an index whose type cannot remap (ZoneMap, BloomFilter, FM;
    // see `test_addr_domain_index_after_compaction_with_stable_row_ids`).
    let current_indices = index_by_name(&dataset).await;
    assert_eq!(current_indices["vector"], indices["vector"]);
    assert_ne!(current_indices["scalar"], indices["scalar"]);

    let live_fragments: RoaringBitmap = dataset.fragments().iter().map(|f| f.id as u32).collect();
    let scalar_index = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .find(|index| index.name == "scalar")
        .unwrap()
        .clone();
    assert_eq!(
        scalar_index.effective_fragment_bitmap(&live_fragments),
        Some(live_fragments)
    );

    let after_vec_result = vector_query(&dataset).await;
    assert_eq!(before_vec_result, after_vec_result);

    let after_scalar_result = scalar_query(&dataset).await;
    assert_eq!(before_scalar_result, after_scalar_result);
}
/// Regression test for https://github.com/lance-format/lance/issues/8076
///
/// Zone map, bloom filter, and FM indices report matches as physical row addresses, so
/// compaction invalidates it even under stable row ids. Reusing it for the rewritten
/// fragments made a filtered scan fail with an internal error (a fragment referenced
/// by the index no longer existed) or, once translation tolerated that, silently drop
/// every match.
///
/// NGram also reports matches as physical row addresses, but -- unlike these
/// three -- it supports eager remap (`ScalarIndexPlugin::can_remap`), so an
/// eager compaction keeps it covering the rewritten fragments instead of
/// losing coverage of them; see
/// `test_defer_index_remap_ngram_with_stable_row_ids` for that case instead.
#[rstest]
#[case::zone_map(BuiltinIndexType::ZoneMap, IndexType::ZoneMap, "i", "i > 0", 199)]
#[case::bloom_filter(BuiltinIndexType::BloomFilter, IndexType::BloomFilter, "i", "i = 0", 1)]
#[case::fm(
    BuiltinIndexType::Fm,
    IndexType::Fm,
    "text",
    "contains(text, 'needle')",
    100
)]
#[tokio::test]
async fn test_addr_domain_index_after_compaction_with_stable_row_ids(
    #[case] builtin: BuiltinIndexType,
    #[case] index_type: IndexType,
    #[case] indexed_column: &str,
    #[case] query: &str,
    #[case] expected_rows: usize,
    #[values(false, true)] defer_index_remap: bool,
) {
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "text",
            lance_datagen::array::cycle_utf8_literals(&["needle", "haystack"]),
        )
        .into_ram_dataset_with_params(
            FragmentCount::from(2),
            FragmentRowCount::from(100),
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 100,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    dataset
        .create_index(
            &[indexed_column],
            index_type,
            Some("addr_idx".to_string()),
            &ScalarIndexParams::for_builtin(builtin),
            false,
        )
        .await
        .unwrap();

    let options = CompactionOptions {
        defer_index_remap,
        ..Default::default()
    };
    if defer_index_remap && index_type == IndexType::Fm {
        // FM discards the fragment reuse index at load.
        let error = plan_compaction(&dataset, &options).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("`addr_idx`"), "{error}");
        return;
    }
    compact_files(&mut dataset, options, None).await.unwrap();

    let live_fragments: RoaringBitmap = dataset.fragments().iter().map(|f| f.id as u32).collect();
    let index = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .find(|index| index.name == "addr_idx")
        .expect("index must survive compaction")
        .clone();
    let covered = index.effective_fragment_bitmap(&live_fragments);
    if defer_index_remap {
        // Readers translate its addresses through the fragment reuse index.
        assert_eq!(covered, Some(live_fragments));
    } else {
        // The index only knows the pre-compaction fragments, so it must not claim
        // to cover the fragment they were rewritten into.
        assert!(
            covered.is_none_or(|covered| covered.is_empty()),
            "compaction must not point an address-domain index at the fragments it wrote"
        );
    }

    let mut scanner = dataset.scan();
    scanner.filter(query).unwrap();
    let matched = scanner.try_into_batch().await.unwrap();
    assert_eq!(matched.num_rows(), expected_rows);
}
#[tokio::test]
async fn test_remap_index_after_compaction() {
    let mut data_gen = BatchGenerator::new()
        .col(Box::new(
            RandomVector::new().vec_width(128).named("vec".to_owned()),
        ))
        .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(6_000),
        "memory://test/table",
        Some(WriteParams {
            max_rows_per_file: 1_000, // 6 files
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Create a index to be remapped
    let index_name = Some("scalar".into());
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            index_name.clone(),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };

    // Remap without a frag reuse index should yield unsupported
    let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
        panic!("scalar index must be available");
    };

    let result = remapping::remap_column_index(&mut dataset, &["i"], index_name.clone()).await;
    assert!(matches!(result, Err(Error::NotSupported { .. })));

    let plan = plan_compaction(&dataset, &options).await.unwrap();

    // Commit each rewrite task separately to simulate 3 compaction runs
    // being accumulated in the fragment reuse index
    for task in plan.tasks().iter() {
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();
    }

    // Load and verify the fragment reuse index content
    let Some(frag_reuse_index_meta) = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
    else {
        panic!("Fragment reuse index must be available");
    };
    let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
        .await
        .unwrap();
    let frag_reuse_index =
        open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
            .await
            .unwrap();

    assert_eq!(frag_reuse_index.details.versions.len(), plan.tasks().len());

    // Check auto-remap
    let mut all_fragment_bitmap = RoaringBitmap::new();
    dataset.fragments().iter().for_each(|f| {
        all_fragment_bitmap.insert(f.id as u32);
    });
    let Some(scalar_index_before_remap) = dataset.load_index_by_name("scalar").await.unwrap()
    else {
        panic!("scalar index must be available");
    };
    assert_eq!(
        scalar_index_before_remap.fragment_bitmap.unwrap(),
        all_fragment_bitmap
    );

    // Trigger index remap
    remapping::remap_column_index(&mut dataset, &["i"], index_name.clone())
        .await
        .unwrap();

    // Compare against original index
    let indices = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap();
    let Some(remapped_scalar_index) = indices.into_iter().find(|idx| idx.name == "scalar") else {
        panic!("scalar index must be available");
    };
    assert_ne!(remapped_scalar_index.uuid, scalar_index.uuid);
    assert_eq!(
        remapped_scalar_index.fragment_bitmap.unwrap(),
        all_fragment_bitmap
    );
}
#[tokio::test]
async fn test_read_bitmap_index_with_defer_index_remap() {
    // Create a dataset with categorical values
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
        )
        .col(
            "category",
            lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
        )
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    // Get initial counts for each category
    let count1 = dataset
        .count_rows(Some("category = 1".to_owned()))
        .await
        .unwrap();
    let count2 = dataset
        .count_rows(Some("category = 2".to_owned()))
        .await
        .unwrap();
    let count3 = dataset
        .count_rows(Some("category = 3".to_owned()))
        .await
        .unwrap();

    // Create a bitmap index on the category column
    let index_name = Some("category_idx".into());
    dataset
        .create_index(
            &["category"],
            IndexType::Bitmap,
            index_name.clone(),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let original_index = indices
        .iter()
        .find(|idx| idx.name == "category_idx")
        .unwrap();

    // Run compaction with deferred index remapping
    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // Verify the index UUID is unchanged (it should not be remapped yet)
    let Some(current_index) = dataset.load_index_by_name("category_idx").await.unwrap() else {
        panic!("category index must be available");
    };
    assert_eq!(current_index.uuid, original_index.uuid);

    // Verify that scans still work correctly and return the same counts
    assert_eq!(
        dataset
            .count_rows(Some("category = 1".to_owned()))
            .await
            .unwrap(),
        count1
    );
    assert_eq!(
        dataset
            .count_rows(Some("category = 2".to_owned()))
            .await
            .unwrap(),
        count2
    );
    assert_eq!(
        dataset
            .count_rows(Some("category = 3".to_owned()))
            .await
            .unwrap(),
        count3
    );

    // Verify that after index creation and compaction, scan uses bitmap index scan
    let mut scanner = dataset.scan();
    scanner.filter("category = 1").unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ScalarIndexQuery: query=[category = 1]@category_idx(Bitmap)"),
        "Expected index query in plan: {}",
        plan
    );
}
/// A zone map is built over `_rowaddr`, so a rewrite invalidates every
/// address it stores. Stable row ids do not save it: they spare the
/// row-id-domain indices a remap, not this one.
///
/// Without a deferred remap there is nothing to repair those addresses with,
/// so the rewritten fragments are dropped from its coverage and the scanner
/// falls back to a full scan. Answers stay correct; the index stops earning
/// its keep.
#[tokio::test]
async fn test_zonemap_loses_coverage_compacting_stable_row_ids() {
    let mut dataset = zonemap_stable_row_id_dataset("memory://").await;

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 512,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_none(),
        "an eager compaction writes no fragment-reuse index"
    );
    assert!(
        zonemap_coverage(&dataset).await.is_empty(),
        "coverage should have been dropped rather than followed"
    );
    assert_zonemap_answers_match_scan(&dataset).await;
}
/// When a remap-capable index (BTree) shares a stable-row-id table with a
/// ZoneMap, an eager compaction must still remap the BTree in place and
/// drop the ZoneMap's coverage of the rewritten fragments -- the same
/// outcome as each index alone -- even though `DatasetIndexRemapper`
/// never opens the ZoneMap to find that out (it isn't
/// `supports_eager_remap`, so it's excluded up front rather than opened
/// and found to report `RemapResult::Drop`).
#[tokio::test]
async fn test_eager_compaction_remaps_btree_and_drops_zonemap_coverage_together() {
    let mut dataset = zonemap_stable_row_id_dataset("memory://").await;
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let id_idx_before = dataset.load_indices_by_name("id_idx").await.unwrap();
    assert_eq!(id_idx_before.len(), 1);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 512,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_none(),
        "an eager compaction writes no fragment-reuse index"
    );

    let id_idx_after = dataset.load_indices_by_name("id_idx").await.unwrap();
    assert_eq!(id_idx_after.len(), 1);
    assert_ne!(
        id_idx_before[0].uuid, id_idx_after[0].uuid,
        "the BTree must be remapped onto the rewritten fragments"
    );
    assert!(
        zonemap_coverage(&dataset).await.is_empty(),
        "the ZoneMap's coverage should have been dropped rather than remapped"
    );

    assert_zonemap_answers_match_scan(&dataset).await;
    let mut by_id = dataset.scan();
    by_id.filter("id = 7").unwrap();
    by_id.project(&["id"]).unwrap();
    assert_eq!(
        by_id.try_into_batch().await.unwrap()["id"]
            .as_primitive::<Int32Type>()
            .values(),
        &[7],
    );
}
/// The deletion sits at a non-zero offset of fragment 0 and the matches span
/// every fragment, so an address the index fails to translate loses rows.
#[rstest]
#[case::zone_map(AddressIndex::ZoneMap)]
#[case::bloom_filter(AddressIndex::BloomFilter)]
#[tokio::test]
async fn test_read_address_index_with_defer_index_remap(
    #[case] address_index: AddressIndex,
    #[values(false, true)] enable_stable_row_ids: bool,
) {
    let dir = TempStrDir::default();
    let batch = arrow_array::record_batch!(
        ("id", Int32, (0..12).collect::<Vec<_>>()),
        (
            "value",
            Int64,
            [
                Some(0),
                None,
                Some(20),
                Some(30),
                Some(40),
                None,
                Some(60),
                Some(70),
                Some(80),
                None,
                Some(100),
                Some(110)
            ]
        )
    )
    .unwrap();
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 4,
            max_rows_per_group: 4,
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);
    address_index.create(&mut dataset, false).await;
    dataset.delete("id = 1").await.unwrap();

    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 512,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_removed, 3);
    assert_eq!(metrics.fragments_added, 1);
    assert!(dataset.frag_reuse_index().await.unwrap().is_some());
    assert_index_answers(&dataset, &address_index.cases(false)).await;

    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    assert_index_answers(&dataset, &address_index.cases(false)).await;

    let merged = dataset
        .merge_existing_index_segments(dataset.load_indices_by_name("addr_idx").await.unwrap())
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("addr_idx", address_index.column(), vec![merged])
        .await
        .unwrap();
    assert_index_answers(&dataset, &address_index.cases(false)).await;

    let appended = arrow_array::record_batch!(
        ("id", Int32, [12, 13, 14, 15]),
        ("value", Int64, [None, Some(130), None, Some(150)])
    )
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();
    dataset
        .optimize_indices(&lance_index::optimize::OptimizeOptions::merge(1))
        .await
        .unwrap();
    assert_index_answers(&dataset, &address_index.cases(true)).await;

    address_index.create(&mut dataset, true).await;
    assert_index_answers(&dataset, &address_index.cases(true)).await;
}
#[rstest]
#[tokio::test]
async fn test_read_fm_index_with_defer_index_remap(#[values(false, true)] has_deletions: bool) {
    let dir = TempStrDir::default();
    let batch = arrow_array::record_batch!(
        ("id", Int32, (0..12).collect::<Vec<_>>()),
        (
            "text",
            Utf8,
            [
                Some("needle alpha"),
                None,
                Some("needle deleted"),
                Some("haystack"),
                Some("needle"),
                Some("haystack"),
                Some("needle"),
                Some("haystack"),
                Some("needle"),
                None,
                Some("needle needle"),
                Some("haystack")
            ]
        )
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 4,
            max_rows_per_group: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);
    dataset
        .create_index(
            &["text"],
            IndexType::Fm,
            Some("addr_idx".to_string()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::Fm),
            false,
        )
        .await
        .unwrap();
    if has_deletions {
        dataset.delete("id = 2").await.unwrap();
    }
    let cases = [
        (
            "contains(text, 'needle')",
            (0..12)
                .step_by(2)
                .filter(|id| !has_deletions || *id != 2)
                .collect(),
        ),
        (
            "contains(text, 'deleted')",
            if has_deletions { vec![] } else { vec![2] },
        ),
        ("contains(text, 'haystack')", vec![3, 5, 7, 11]),
        ("contains(text, 'absent')", vec![]),
    ];
    assert_index_answers(&dataset, &cases).await;

    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_removed, 3);
    assert_eq!(metrics.fragments_added, 1);
    let fri = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("deferred compaction must record a fragment reuse index");
    assert_eq!(fri.index_version, 0);
    assert_index_answers(&dataset, &cases).await;

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_index_answers(&reopened, &cases).await;
}

#[tokio::test]
async fn test_read_btree_index_with_defer_index_remap() {
    // Create a dataset with an incremental ID column
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
        )
        .col("id", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(110), FragmentRowCount::from(1000))
        .await
        .unwrap();

    // Get initial counts for some ID ranges
    let count_low = dataset
        .count_rows(Some("id < 1000".to_owned()))
        .await
        .unwrap();
    let count_mid = dataset
        .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
        .await
        .unwrap();
    let count_high = dataset
        .count_rows(Some("id >= 5000".to_owned()))
        .await
        .unwrap();

    // Create a btree index on the id column
    let index_name = Some("id_idx".into());
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            index_name.clone(),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let original_index = indices.iter().find(|idx| idx.name == "id_idx").unwrap();

    // Run compaction with deferred index remapping
    let options = CompactionOptions {
        target_rows_per_fragment: 50_000,
        defer_index_remap: true,
        ..Default::default()
    };

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // Verify the index UUID is unchanged (it should not be remapped yet)
    let Some(current_index) = dataset.load_index_by_name("id_idx").await.unwrap() else {
        panic!("id index must be available");
    };
    assert_eq!(current_index.uuid, original_index.uuid);

    // Verify that scans still work correctly and return the same counts
    assert_eq!(
        dataset
            .count_rows(Some("id < 1000".to_owned()))
            .await
            .unwrap(),
        count_low
    );
    assert_eq!(
        dataset
            .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
            .await
            .unwrap(),
        count_mid
    );
    assert_eq!(
        dataset
            .count_rows(Some("id >= 5000".to_owned()))
            .await
            .unwrap(),
        count_high
    );

    // Verify that after index creation and compaction, scan uses btree index scan
    let mut scanner = dataset.scan();
    scanner.filter("id >= 2000 and id < 3000").unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ScalarIndexQuery: query=[id >= 2000 && id < 3000]@id_idx(BTree)"),
        "Expected scalar index query in plan: {}",
        plan
    );
}
#[rstest]
#[case(IndexRemapMode::Compact)]
#[case(IndexRemapMode::Direct)]
#[tokio::test]
async fn test_btree_index_remap_after_compaction(#[case] index_remap_mode: IndexRemapMode) {
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(32)),
        )
        .col("id", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    // Delete rows scattered across fragments so the remap must drop some old
    // addresses and shift the survivors.
    dataset.delete("id % 10 == 0").await.unwrap();

    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let count_low = dataset
        .count_rows(Some("id < 1000".to_owned()))
        .await
        .unwrap();
    let count_mid = dataset
        .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
        .await
        .unwrap();
    let count_high = dataset
        .count_rows(Some("id >= 5000".to_owned()))
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 50_000,
        index_remap_mode,
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // The index was remapped inline and must still drive scans.
    let mut scanner = dataset.scan();
    scanner.filter("id >= 2000 and id < 3000").unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ScalarIndexQuery: query=[id >= 2000 && id < 3000]@id_idx(BTree)"),
        "Expected scalar index query in plan: {}",
        plan
    );

    // Counts resolved through the remapped index match the pre-compaction
    // values in both remap modes.
    assert_eq!(
        dataset
            .count_rows(Some("id < 1000".to_owned()))
            .await
            .unwrap(),
        count_low
    );
    assert_eq!(
        dataset
            .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
            .await
            .unwrap(),
        count_mid
    );
    assert_eq!(
        dataset
            .count_rows(Some("id >= 5000".to_owned()))
            .await
            .unwrap(),
        count_high
    );
}
#[rstest]
#[case(IndexRemapMode::Compact)]
#[case(IndexRemapMode::Direct)]
#[tokio::test]
async fn test_inverted_index_remap_after_compaction(#[case] index_remap_mode: IndexRemapMode) {
    use arrow_array::cast::AsArray;

    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col("doc", lance_datagen::array::random_sentence(1, 100, false))
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    dataset
        .create_index(
            &["doc"],
            IndexType::Inverted,
            Some("doc_idx".into()),
            &InvertedIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let original_uuid = dataset
        .load_index_by_name("doc_idx")
        .await
        .unwrap()
        .unwrap()
        .uuid;

    // Sample a few words from a real document to drive full-text searches.
    let words: Vec<String> = {
        let mut scanner = dataset.scan();
        scanner
            .project(&["doc"])
            .unwrap()
            .limit(Some(1), None)
            .unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut words: Vec<String> = batches[0]["doc"]
            .as_string::<i32>()
            .value(0)
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        words.sort();
        words.dedup();
        words.truncate(3);
        words
    };
    assert!(!words.is_empty(), "sampled document must contain words");

    // Delete rows scattered across fragments so the remap must drop some old
    // addresses and shift the survivors.
    dataset.delete("id % 10 == 0").await.unwrap();

    // Capture the post-deletion full-text-search counts (resolved through the
    // index + deletion vectors) before compaction physically remaps.
    let mut before = Vec::new();
    for word in &words {
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(word.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        before.push(scanner.count_rows().await.unwrap());
    }

    // Inline remap (defer_index_remap = false): compaction physically rebuilds
    // the inverted index through the configured remap mode.
    let options = CompactionOptions {
        target_rows_per_fragment: 50_000,
        index_remap_mode,
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // The index was physically remapped inline, so its uuid must change.
    assert_ne!(
        dataset
            .load_index_by_name("doc_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid,
        original_uuid,
        "inverted index must be physically remapped inline (mode {index_remap_mode:?})"
    );

    // The remapped index must still drive full-text search.
    let mut scanner = dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new(words[0].clone()))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("MatchQuery"),
        "Expected inverted index scan in plan: {}",
        plan
    );

    // Counts resolved through the remapped index match the pre-compaction
    // values in both remap modes.
    for (word, expected) in words.iter().zip(before) {
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(word.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(
            scanner.count_rows().await.unwrap(),
            expected,
            "full-text count for {word:?} changed after compaction (mode {index_remap_mode:?})"
        );
    }
}
#[tokio::test]
async fn test_read_inverted_index_with_defer_index_remap() {
    // Generate random words using lance-datagen
    let mut words_gen = lance_datagen::array::random_sentence(1, 100, true);
    let doc_col = words_gen
        .generate_default(lance_datagen::RowCount::from(6000))
        .unwrap();

    let batch = RecordBatch::try_new(
        Schema::new(vec![Field::new("doc", DataType::LargeUtf8, false)]).into(),
        vec![doc_col.clone()],
    )
    .unwrap();
    let schema_ref = batch.schema();
    let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
    let mut dataset = Dataset::write(
        stream,
        "memory://test/table",
        Some(WriteParams {
            max_rows_per_file: 1_000, // 6 files
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Get initial counts for some word searches
    // Extract some test words from the generated documents
    let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
    let sample_words: Vec<String> = large_string_array
        .value(0)
        .split_whitespace()
        .take(10)
        .map(|s| s.to_string())
        .collect();
    let test_word1 = &sample_words[0];
    let test_word2 = &sample_words[1];
    let test_word3 = &sample_words[2];

    // Create an inverted index on the doc column
    let index_name = Some("doc_idx".into());
    dataset
        .create_index(
            &["doc"],
            IndexType::Inverted,
            index_name.clone(),
            &InvertedIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let original_index = indices.iter().find(|idx| idx.name == "doc_idx").unwrap();

    // Run compaction with deferred index remapping
    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // Verify the index UUID is unchanged (it should not be remapped yet)
    let Some(current_index) = dataset.load_index_by_name("doc_idx").await.unwrap() else {
        panic!("doc index must be available");
    };
    assert_eq!(current_index.uuid, original_index.uuid);

    // Initial scan
    let mut scanner = dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let count1 = scanner.count_rows().await.unwrap();
    scanner = dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new(test_word2.clone()))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let count2 = scanner.count_rows().await.unwrap();
    scanner = dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new(test_word3.clone()))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let count3 = scanner.count_rows().await.unwrap();

    // Verify that after index creation and compaction, scan uses inverted index scan
    let mut scanner = dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("MatchQuery"),
        "Expected inverted index scan in plan: {}",
        plan
    );
    assert!(
        !plan.contains("LanceScan"),
        "Expected no fragment scan in plan: {}",
        plan
    );

    // Reindex to the latest
    dataset
        .create_index(
            &["doc"],
            IndexType::Inverted,
            index_name.clone(),
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Verify that scans still work correctly and return the same counts
    let mut scanner = dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    assert_eq!(scanner.count_rows().await.unwrap(), count1);
    scanner = dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new(test_word2.clone()))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    assert_eq!(scanner.count_rows().await.unwrap(), count2);
    scanner = dataset.scan();
    scanner
        .full_text_search(FullTextSearchQuery::new(test_word3.clone()))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    assert_eq!(scanner.count_rows().await.unwrap(), count3);
}
/// Deferred compaction that materializes deletions must not corrupt an
/// inverted (FTS) index read through the fragment-reuse index. The index's
/// posting lists reference doc_ids positionally; if the load-time remap
/// dropped the deleted rows it would renumber the doc_ids and desync the
/// posting lists (out-of-bounds `num_tokens`, wrong/stale row ids). The
/// tombstone-preserve-positions load path must keep results correct in the
/// FRI window and after the physical remap + trim.
#[tokio::test]
async fn test_read_inverted_index_with_defer_index_remap_and_deletions() {
    // Enough surviving docs for several compressed posting-list blocks
    // (BLOCK_SIZE = 128), split across several fragments so compaction has
    // real work — but no larger.
    const ROWS: i32 = 1200;
    const DELETED: i32 = 400;

    // Every row contains "lance", so the term matches all live rows; `id`
    // tells us exactly which rows survive.
    let ids = Int32Array::from_iter_values(0..ROWS);
    let docs = LargeStringArray::from_iter_values((0..ROWS).map(|_| "lance apple orange"));
    let batch = RecordBatch::try_new(
        Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("doc", DataType::LargeUtf8, false),
        ])
        .into(),
        vec![Arc::new(ids) as ArrayRef, Arc::new(docs) as ArrayRef],
    )
    .unwrap();
    let schema_ref = batch.schema();
    let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
    let mut dataset = Dataset::write(
        stream,
        "memory://test/table",
        Some(WriteParams {
            max_rows_per_file: 200, // 6 fragments
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    dataset
        .create_index(
            &["doc"],
            IndexType::Inverted,
            Some("doc_idx".into()),
            &InvertedIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // Delete a prefix, then deferred-compact so the deletions are
    // materialized into the fragment-reuse index the index is read through.
    dataset.delete(&format!("id < {DELETED}")).await.unwrap();
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some(),
        "deferred compaction must leave a fragment-reuse index"
    );

    // FTS "lance" → sorted surviving ids. Projecting `id` forces a take, so
    // a stale row address would error or return a wrong/dead row.
    async fn search_ids(dataset: &Dataset) -> Vec<i32> {
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new("lance".to_owned()))
            .unwrap();
        scanner.project::<&str>(&["id"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    let expected = (DELETED..ROWS).collect::<Vec<_>>();

    // FRI window: index read through the reuse index.
    let during = search_ids(&dataset).await;
    assert_eq!(
        during, expected,
        "FRI-window FTS must return exactly the surviving rows (no resurrection, no loss, no stale rows)"
    );

    // Physical remap + trim: must still be correct.
    remapping::remap_column_index(&mut dataset, &["doc"], Some("doc_idx".into()))
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    let after = search_ids(&dataset).await;
    assert_eq!(
        after, expected,
        "FTS must stay correct after physical remap + fragment-reuse trim"
    );
}
#[tokio::test]
async fn test_read_ngram_index_with_defer_index_remap() {
    // Generate random words using lance-datagen
    let mut words_gen = lance_datagen::array::random_sentence(1, 100, true);
    let doc_col = words_gen
        .generate_default(lance_datagen::RowCount::from(6000))
        .unwrap();

    let batch = RecordBatch::try_new(
        Schema::new(vec![Field::new("doc", DataType::LargeUtf8, false)]).into(),
        vec![doc_col.clone()],
    )
    .unwrap();
    let schema_ref = batch.schema();
    let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
    let mut dataset = Dataset::write(
        stream,
        "memory://test/table",
        Some(WriteParams {
            max_rows_per_file: 1_000, // 6 files
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Get initial counts for some word searches
    // Extract some test words from the generated documents
    let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
    let sample_words: Vec<String> = large_string_array
        .value(0)
        .split_whitespace()
        .take(10)
        .map(|s| s.to_string())
        .collect();
    let test_word1 = &sample_words[0];
    let test_word2 = &sample_words[1];
    let test_word3 = &sample_words[2];

    // Create an inverted index on the doc column
    let index_name = Some("doc_idx".into());
    dataset
        .create_index(
            &["doc"],
            IndexType::NGram,
            index_name.clone(),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let original_index = indices.iter().find(|idx| idx.name == "doc_idx").unwrap();

    // Initial scan
    let count1 = dataset
        .count_rows(Some(format!("contains(doc, '{}')", test_word1)))
        .await
        .unwrap();
    let count2 = dataset
        .count_rows(Some(format!("contains(doc, '{}')", test_word2)))
        .await
        .unwrap();
    let count3 = dataset
        .count_rows(Some(format!("contains(doc, '{}')", test_word3)))
        .await
        .unwrap();

    // Run compaction with deferred index remapping
    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // Verify the index UUID is unchanged (it should not be remapped yet)
    let Some(current_index) = dataset.load_index_by_name("doc_idx").await.unwrap() else {
        panic!("doc index must be available");
    };
    assert_eq!(current_index.uuid, original_index.uuid);

    // Verify that scans still work correctly and return the same counts
    assert_eq!(
        dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word1)))
            .await
            .unwrap(),
        count1
    );
    assert_eq!(
        dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word2)))
            .await
            .unwrap(),
        count2
    );
    assert_eq!(
        dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word3)))
            .await
            .unwrap(),
        count3
    );

    // Verify that after index creation and compaction, scan uses inverted index scan
    let mut scanner = dataset.scan();
    scanner
        .filter(&format!("contains(doc, '{}')", test_word1))
        .unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ScalarIndexQuery: query=[contains(doc, Utf8"),
        "Expected scalar index query in plan: {}",
        plan
    );
}
#[tokio::test]
async fn test_compaction_preserves_legacy_nullable_label_list_version() {
    let dir =
        crate::utils::test::copy_test_data_to_tmp("v3.0.1/nullable_label_list/data.lance").unwrap();
    let mut dataset = Dataset::open(&dir.path_str()).await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    let original = dataset.load_indices_by_name("labels_idx").await.unwrap();
    assert_eq!(original[0].index_version, 0);
    let predicate = "NOT array_has_any(labels, [1])";
    // Version 0 cannot recover list-level NULLs. Keep its existing reader
    // behavior rather than advertise the current NULL-aware format.
    let legacy_count = dataset.count_rows(Some(predicate.into())).await.unwrap();
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);
    let remapped = dataset.load_indices_by_name("labels_idx").await.unwrap();
    assert_eq!(remapped[0].index_version, 0);
    assert_eq!(remapped[0].index_details, original[0].index_details);
    assert_ne!(remapped[0].uuid, original[0].uuid);
    assert_eq!(
        dataset.count_rows(Some(predicate.into())).await.unwrap(),
        legacy_count
    );
    for (predicate, expected) in [("array_has_any(labels, [1])", 2), ("labels IS NULL", 2)] {
        for indexed in [false, true] {
            let mut scan = dataset.scan();
            scan.use_scalar_index(indexed).filter(predicate).unwrap();
            assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), expected);
        }
    }
}
#[tokio::test]
async fn test_read_label_list_index_with_defer_index_remap() {
    // Create a dataset with list data for labels
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
        )
        .col(
            "labels",
            lance_datagen::array::rand_list_any(
                lance_datagen::array::cycle::<Int64Type>(vec![1, 2, 3, 4, 5]),
                false,
            ),
        )
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    // Get initial counts for different label values
    let count1 = dataset
        .count_rows(Some("array_has_any(labels, [1])".to_owned()))
        .await
        .unwrap();
    let count2 = dataset
        .count_rows(Some("array_has_any(labels, [5])".to_owned()))
        .await
        .unwrap();
    let count3 = dataset
        .count_rows(Some("array_has_any(labels, [10])".to_owned()))
        .await
        .unwrap();

    // Create a label list index on the labels column
    let index_name = Some("labels_idx".into());
    dataset
        .create_index(
            &["labels"],
            IndexType::LabelList,
            index_name.clone(),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let indices = dataset.load_indices().await.unwrap();
    let original_index = indices.iter().find(|idx| idx.name == "labels_idx").unwrap();

    // Run compaction with deferred index remapping
    let options = CompactionOptions {
        target_rows_per_fragment: 2000,
        defer_index_remap: true,
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(metrics.fragments_added > 0);

    // Verify that the index UUID remains unchanged
    let indices = dataset.load_indices().await.unwrap();
    let current_index = indices.iter().find(|idx| idx.name == "labels_idx").unwrap();
    assert_eq!(current_index.uuid, original_index.uuid);

    // Verify that scans still work correctly and return the same counts
    assert_eq!(
        dataset
            .count_rows(Some("array_has_any(labels, [1])".to_owned()))
            .await
            .unwrap(),
        count1
    );
    assert_eq!(
        dataset
            .count_rows(Some("array_has_any(labels, [5])".to_owned()))
            .await
            .unwrap(),
        count2
    );
    assert_eq!(
        dataset
            .count_rows(Some("array_has_any(labels, [10])".to_owned()))
            .await
            .unwrap(),
        count3
    );

    // Verify that after index creation and compaction, scan uses label list index scan
    let mut scanner = dataset.scan();
    scanner.filter("array_has_any(labels, [1])").unwrap();
    scanner.project::<String>(&[]).unwrap().with_row_id();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains(
            "ScalarIndexQuery: query=[array_has_any(labels, List([1]))]@labels_idx(LabelList)",
        ),
        "Expected scalar index query in plan: {}",
        plan
    );
}
#[tokio::test]
async fn test_bitmap_index_defer_compaction_with_deletions() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "category",
            lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
        )
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();
    dataset
        .create_index(
            &["category"],
            IndexType::Bitmap,
            Some("category_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset.delete("id < 1500").await.unwrap();
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME),
        "deferred compaction must record a frag-reuse index"
    );

    let mut scanner = dataset.scan();
    scanner.filter("category = 3").unwrap();
    scanner.project(&["id"]).unwrap();
    let batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let mut returned = 0;
    for b in &batches {
        for id in b["id"].as_primitive::<Int32Type>().values() {
            assert!(
                *id >= 1500,
                "bitmap returned deleted id {id} in the FRI window"
            );
            returned += 1;
        }
    }
    assert!(returned > 0, "expected surviving category=3 rows");
}
