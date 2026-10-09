// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the frag reuse contract area.

use super::*;
use crate::dataset::optimize::commit::reserve_fragment_ids;
use crate::dataset::optimize::execution::{can_use_binary_copy, live_row_addrs};

// Regression test for https://github.com/lancedb/lance/issues/6161
// When FragReuseIndexDetails exceeds 204800 bytes it is written to an external
// file. Previously the file was silently dropped (temp file deleted) because
// tokio::io::AsyncWriteExt::shutdown was called instead of
// lance_io::traits::Writer::shutdown, which persists the temp file.
#[tokio::test]
async fn test_defer_index_remap_large_external_file() {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    // Create ~150 fragments × 1000 rows to produce a FragReuseIndexDetails
    // that exceeds the 204800-byte inline threshold (~302 KB serialized).
    let num_fragments = 150usize;
    let rows_per_fragment = 1000usize;
    let total_rows = num_fragments * rows_per_fragment;

    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));

    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            vec![Ok(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..total_rows as i32)) as ArrayRef],
            )
            .unwrap())],
            schema.clone(),
        ),
        test_uri,
        Some(WriteParams {
            max_rows_per_file: rows_per_fragment,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), num_fragments);

    // An FRI is only written for compactions that touch indexed data, so
    // index the column being compacted.
    create_scalar_index(&mut dataset, "i", false).await;

    // Delete a few rows from each fragment so compaction has something to do.
    dataset.delete("i % 1000 = 0").await.unwrap();

    compact_files(
        &mut dataset,
        CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    // Loading the FragReuseIndex details must succeed even when the details
    // were written to an external file.
    let frag_reuse_meta = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("fragment reuse index must exist after compaction");

    load_frag_reuse_index_details(&dataset, &frag_reuse_meta)
        .await
        .expect("loading large frag reuse index details must not fail");
}
/// A shallow clone stamps its index metadata with a `base_id`, and the
/// entry's external `details.binpb` stays in the SOURCE dataset's indices
/// directory. Loading the details on the clone must resolve the file
/// through the entry's base instead of the clone's own dataset root,
/// which never contained the file.
#[tokio::test]
async fn test_shallow_clone_loads_external_frag_reuse_details() {
    // On disk: base-path resolution must reach the source dataset's
    // store, which memory:// fixtures cannot demonstrate.
    let source_dir = TempStrDir::default();
    let clone_dir = TempStrDir::default();
    let clone_uri = format!("{}/clone", clone_dir);

    use crate::index::frag_reuse::build_frag_reuse_index_metadata;
    use lance_index::frag_reuse::{FragDigest, FragReuseIndexDetails, FragReuseVersion};

    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            vec![Ok(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..8)) as ArrayRef],
            )
            .unwrap())],
            schema.clone(),
        ),
        &source_dir,
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // An FRI entry whose details spill to the external file: real
    // compactions produce run-compressed bitmaps far below the
    // 204800-byte inline threshold, so synthesize enough reuse versions
    // to cross it.
    let digest = |id: u64| FragDigest {
        id,
        physical_rows: 4,
        num_deleted_rows: 0,
    };
    let mut versions = Vec::new();
    for i in 0..3500u64 {
        let old_id = 1_000 + i;
        let mut addrs = RoaringTreemap::new();
        for offset in 0..4u64 {
            addrs.insert((old_id << 32) + offset);
        }
        let mut serialized = Vec::new();
        addrs.serialize_into(&mut serialized).unwrap();
        versions.push(FragReuseVersion {
            dataset_version: i + 1,
            groups: vec![FragReuseGroup {
                changed_row_addrs: serialized,
                old_frags: vec![digest(old_id)],
                new_frags: vec![digest(100_000 + i)],
            }],
        });
    }
    let details = FragReuseIndexDetails { versions };
    let bitmap: RoaringBitmap = (0..3500u32).map(|i| 100_000 + i).collect();
    let entry = build_frag_reuse_index_metadata(&dataset, None, details, bitmap)
        .await
        .unwrap();
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![entry],
                    removed_indices: vec![],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

    let source_meta = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("fragment reuse index must exist");
    // Sanity: the details actually went external.
    let proto = source_meta
        .index_details
        .as_ref()
        .unwrap()
        .to_msg::<lance_table::format::pb::FragmentReuseIndexDetails>()
        .unwrap();
    assert!(matches!(
        proto.content,
        Some(lance_table::format::pb::fragment_reuse_index_details::Content::External(_))
    ));
    let source_details = load_frag_reuse_index_details(&dataset, &source_meta)
        .await
        .unwrap();
    assert_eq!(source_details.versions.len(), 3500);

    let version = dataset.manifest.version;
    let clone = dataset
        .shallow_clone(clone_uri.as_str(), version, None)
        .await
        .unwrap();
    let clone_meta = clone
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("the clone must carry the fragment reuse index");
    assert!(clone_meta.base_id.is_some());

    // Before the fix this failed with a not-found error: the loader
    // looked for details.binpb under the clone's own indices dir.
    let clone_details = load_frag_reuse_index_details(&clone, &clone_meta)
        .await
        .expect("the clone must load external FRI details through its base");
    assert_eq!(clone_details, source_details);

    // The dataset-level open path goes through the same loader.
    let index = clone
        .open_frag_reuse_index(&NoOpMetricsCollector)
        .await
        .unwrap()
        .expect("the clone must open the fragment reuse index");
    assert_eq!(index.details.versions, source_details.versions);

    // Cached by the source's file: one read, then none, unless the cache is off.
    let dir_name = |dir: &str| {
        std::path::Path::new(dir)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    };
    let (source_name, clone_name) = (dir_name(source_dir.as_str()), dir_name(clone_dir.as_str()));
    for cached in [true, false] {
        let session = if cached {
            crate::session::Session::default()
        } else {
            crate::session::Session::with_index_cache_backend(
                Arc::new(lance_core::cache::MokaCacheBackend::no_cache()),
                128 * 1024 * 1024,
                Default::default(),
            )
        };
        let fresh = crate::dataset::builder::DatasetBuilder::from_uri(clone_uri.as_str())
            .with_session(Arc::new(session))
            .load()
            .await
            .unwrap();
        let store = fresh.object_store(clone_meta.base_id).await.unwrap();
        store.io_stats_incremental();
        for load in 0..2 {
            let details = load_frag_reuse_index_details(&fresh, &clone_meta)
                .await
                .unwrap();
            assert_eq!(details, source_details);
            let reads: Vec<String> = store
                .io_stats_incremental()
                .requests
                .iter()
                .map(|request| request.path.to_string())
                .filter(|path| {
                    path.ends_with(lance_index::frag_reuse::FRAG_REUSE_DETAILS_FILE_NAME)
                })
                .collect();
            if cached && load > 0 {
                assert!(reads.is_empty(), "{reads:?}");
            } else {
                assert_eq!(reads.len(), 1, "{reads:?}");
                assert!(
                    reads[0].contains(&source_name) && !reads[0].contains(&clone_name),
                    "{reads:?}"
                );
            }
        }
    }
}
/// An address-domain index is remapped by reading its persisted pages'
/// stored addresses directly, never composed with a fragment reuse
/// index's pending moves. Eagerly remapping one while a prior deferred
/// compaction's FRI is still undrained would write those untranslated,
/// pre-FRI addresses into the rebuilt index, and the manifest would then
/// claim it fully covers fragments the FRI alone can still resolve --
/// silently losing the rows the FRI has to translate. Eager compaction
/// must refuse instead, until the FRI is drained.
#[rstest]
#[case::btree_physical(IndexType::BTree, false)]
#[case::btree_stable(IndexType::BTree, true)]
#[case::inverted_physical(IndexType::Inverted, false)]
#[tokio::test]
async fn test_eager_remap_refuses_undrained_fri_over_address_domain_index(
    #[case] index_type: IndexType,
    #[case] enable_stable_row_ids: bool,
    #[values(false, true)] is_update: bool,
    #[values(false, true)] is_optimized_before_compaction: bool,
) {
    let batch = arrow_array::record_batch!(
        ("id", Int32, [0, 1, 2, 3]),
        ("value", Int32, [Some(0), Some(1), None, Some(3)]),
        (
            "doc",
            Utf8,
            [Some("needle"), Some("needle"), None, Some("needle")]
        )
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        "memory://",
        Some(WriteParams {
            enable_stable_row_ids,
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let column = if index_type == IndexType::BTree {
        dataset
            .create_index(
                &["value"],
                index_type,
                Some("test_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        "value"
    } else {
        dataset
            .create_index(
                &["doc"],
                index_type,
                Some("test_idx".into()),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        "doc"
    };
    let deferred_options = CompactionOptions {
        defer_index_remap: true,
        ..Default::default()
    };

    // Persisted index addresses now require the FRI, even after optimize_indices.
    compact_files(&mut dataset, deferred_options.clone(), None)
        .await
        .unwrap();
    if is_optimized_before_compaction {
        dataset.optimize_indices(&Default::default()).await.unwrap();
    }
    if is_update {
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 0")
            .unwrap()
            .set(
                column,
                if index_type == IndexType::BTree {
                    "value + 10"
                } else {
                    "'needle updated'"
                },
            )
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = result.new_dataset.as_ref().clone();
    } else {
        dataset.delete("id = 0").await.unwrap();
    }
    assert!(
        !dataset
            .frag_reuse_index()
            .await
            .unwrap()
            .unwrap()
            .is_empty()
    );

    let assert_answers = async |dataset: &Dataset| {
        let expected = if is_update { vec![0, 1, 3] } else { vec![1, 3] };
        let filter = format!("{column} IS NOT NULL");
        assert_eq!(
            values_matching(dataset, "id", &filter, false).await,
            expected
        );
        if index_type == IndexType::BTree {
            assert_eq!(
                values_matching(dataset, "id", "value >= 0", true).await,
                expected
            );
            assert_eq!(
                values_matching(dataset, "id", "value IS NULL", true).await,
                vec![2]
            );
            assert_index_is_used(dataset, "value >= 0", "test_idx").await;
        } else {
            let mut scanner = dataset.scan();
            scanner
                .full_text_search(FullTextSearchQuery::new("needle".to_string()))
                .unwrap()
                .project(&["id"])
                .unwrap();
            assert!(
                scanner
                    .explain_plan(false)
                    .await
                    .unwrap()
                    .contains("MatchQuery")
            );
            let mut ids = scanner.try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values()
                .to_vec();
            ids.sort_unstable();
            assert_eq!(ids, expected);
        }
    };
    assert_answers(&dataset).await;

    let version = dataset.version().version;
    let error = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(error.to_string().contains("defer_index_remap"), "{error}");
    assert_eq!(dataset.version().version, version);

    let error = DefaultCompactionPlanner::new(CompactionOptions::default())
        .unwrap()
        .plan_groups(
            &dataset,
            vec![CompactionGroup {
                fragments: dataset.fragments().as_ref().clone(),
                output_clustering_id: None,
                output_metadata: HashMap::new(),
            }],
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(error.to_string().contains("defer_index_remap"), "{error}");

    // Distributed callers can commit with options different from the plan's.
    let plan = plan_compaction(&dataset, &deferred_options).await.unwrap();
    let mut completed = Vec::new();
    for task in plan.compaction_tasks() {
        completed.push(task.execute(&dataset).await.unwrap());
    }
    assert!(!completed.is_empty());
    let error = commit_compaction(
        &mut dataset,
        completed.clone(),
        Arc::new(DatasetIndexRemapperOptions::default()),
        &CompactionOptions::default(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(error.to_string().contains("defer_index_remap"), "{error}");
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_answers(&dataset).await;

    // Keeping the second compaction deferred must preserve every indexed row.
    commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &deferred_options,
    )
    .await
    .unwrap();
    assert_answers(&dataset).await;

    // Explicitly remap and trim the history to allow eager compaction again.
    remapping::remap_column_index(&mut dataset, &[column], Some("test_idx".into()))
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(
        dataset
            .frag_reuse_index()
            .await
            .unwrap()
            .is_none_or(|fri| fri.is_empty())
    );
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert_answers(&dataset).await;
}
/// A table whose only index can never consume an eager remap (ZoneMap
/// always reports `RemapResult::Drop`) still has to honor row-address
/// moves its tasks recorded under `defer_index_remap` at plan/execution
/// time, even when the commit uses different, eager options --
/// distributed mode may commit with options other than it planned with
/// (see `records_stable_row_id_moves`). Discarding them instead of
/// folding them into a fragment reuse index would silently lose track of
/// rows no bitmap-follow path ever sees again.
#[tokio::test]
async fn test_recorded_moves_over_unremappable_index_still_commit_as_fri() {
    let mut dataset = stable_row_id_dataset("memory://", 3).await;
    dataset
        .create_index(
            &["i"],
            IndexType::ZoneMap,
            Some("i_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
            false,
        )
        .await
        .unwrap();

    let plan = plan_compaction(
        &dataset,
        &CompactionOptions {
            target_rows_per_fragment: 1_000,
            defer_index_remap: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        let result = task.execute(&dataset).await.unwrap();
        assert!(result.row_addrs.is_some());
        results.push(result);
    }
    assert!(!results.is_empty());

    // Committed with the default (eager) options, unlike the deferred
    // ones the tasks were planned and executed with.
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &CompactionOptions::default(),
    )
    .await
    .unwrap();

    assert!(
        dataset
            .load_index_by_name(lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some(),
        "recorded addresses have no eager consumer and must be committed as an FRI"
    );
}
#[tokio::test]
async fn test_recording_refuses_moves_it_cannot_pair() {
    let dataset = stable_row_id_dataset("memory://", 2).await;
    let options = CompactionOptions {
        defer_index_remap: true,
        ..Default::default()
    };
    let mut fragments = dataset.manifest.fragments.as_ref().clone();
    fragments.reverse();
    let error = rewrite_files(
        Cow::Borrowed(&dataset),
        TaskData {
            fragments: fragments.clone(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        },
        &options,
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(
        error.to_string().contains("not in ascending id order"),
        "{error}"
    );

    fragments.reverse();
    let mut rewritten = fragments.clone();
    rewritten[0].physical_rows = Some(9);
    let error = live_row_addrs(&dataset, &fragments, &rewritten)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("leave 20 live rows, but the rewrite wrote 19"),
        "{error}"
    );
}
#[tokio::test]
async fn test_defer_index_remap() {
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

    // Create another same dataset to mimic behavior without deferred index remap
    let mut data_gen2 = BatchGenerator::new()
        .col(Box::new(
            RandomVector::new().vec_width(128).named("vec".to_owned()),
        ))
        .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

    let mut dataset2 = Dataset::write(
        data_gen2.batch(6_000),
        "memory://test/table",
        Some(WriteParams {
            max_rows_per_file: 1_000, // 6 files
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Delete some rows to create deletions
    dataset.delete("i < 500").await.unwrap();
    dataset2.delete("i < 500").await.unwrap();

    // Create the same scalar index on both datasets so deferred and immediate
    // remapping are compared under the same conditions.
    create_scalar_index(&mut dataset, "i", false).await;
    create_scalar_index(&mut dataset2, "i", false).await;

    // Verify the initial state - no fragment reuse index should exist
    let initial_indices = dataset.load_indices().await.unwrap();
    assert_eq!(initial_indices.len(), 1);
    assert_eq!(initial_indices[0].name, "scalar");

    // Store the original scalar index UUID for comparison
    let original_scalar_uuid = initial_indices[0].uuid;

    // Plan and execute compaction manually
    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };
    let options2 = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: false,
        ..Default::default()
    };

    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let plan2 = plan_compaction(&dataset2, &options2).await.unwrap();

    let mut expected_all_old_frag_ids = Vec::new();
    let mut expected_all_new_frag_ids = Vec::new();
    let mut expected_all_new_frag_bitmap = RoaringBitmap::new();
    let mut expected_all_row_id_map = HashMap::new();
    let mut deferred_results = Vec::new();
    let mut immediate_results = Vec::new();

    for (task, task2) in plan.tasks().iter().zip(plan2.tasks()) {
        let deferred_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
            .await
            .unwrap();
        let immediate_result = rewrite_files(Cow::Borrowed(&dataset2), task2.clone(), &options2)
            .await
            .unwrap();

        // Both should produce row_addrs (address-style row IDs)
        assert!(deferred_result.row_addrs.is_some());
        assert!(!deferred_result.row_addrs.as_ref().unwrap().is_empty());
        assert!(!deferred_result.row_addrs.as_ref().unwrap().is_empty());
        assert!(!deferred_result.original_fragments.is_empty());
        assert!(!deferred_result.new_fragments.is_empty());

        assert!(immediate_result.row_addrs.is_some());
        assert!(!immediate_result.original_fragments.is_empty());
        assert!(!immediate_result.new_fragments.is_empty());

        // Both should capture the same row addresses
        assert_eq!(deferred_result.row_addrs, immediate_result.row_addrs);

        deferred_results.push(deferred_result);
        immediate_results.push(immediate_result);
    }

    // Reserve fragment IDs for immediate results to build expected values
    {
        let frags: Vec<&mut Fragment> = immediate_results
            .iter_mut()
            .flat_map(|r| r.new_fragments.iter_mut())
            .collect();
        reserve_fragment_ids(&dataset2, frags.into_iter())
            .await
            .unwrap();
    }

    // Build expected values by transposing using the immediate results
    for immediate_result in &immediate_results {
        let row_addrs_bytes = immediate_result.row_addrs.as_ref().unwrap();
        let row_addrs =
            RoaringTreemap::deserialize_from(&mut Cursor::new(row_addrs_bytes)).unwrap();
        let transposed = transpose_row_addrs(
            row_addrs,
            &immediate_result.original_fragments,
            &immediate_result.new_fragments,
        );
        expected_all_row_id_map.extend(transposed);
        immediate_result.new_fragments.iter().for_each(|frag| {
            expected_all_new_frag_bitmap.insert(frag.id as u32);
        });
        expected_all_new_frag_ids.extend(
            immediate_result
                .new_fragments
                .iter()
                .map(|s| s.id)
                .collect::<Vec<_>>(),
        );
        expected_all_old_frag_ids.extend(
            immediate_result
                .original_fragments
                .iter()
                .map(|s| s.id)
                .collect::<Vec<_>>(),
        );
    }

    // Now commit the first compaction (using deferred results)
    let first_metrics = commit_compaction(
        &mut dataset,
        deferred_results.clone(),
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    // Verify compaction happened
    assert!(first_metrics.fragments_removed > 0);
    assert!(first_metrics.fragments_added > 0);

    // Load and verify the fragment reuse index content
    let Some(frag_reuse_index_meta) = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
    else {
        panic!("Fragment reuse index must be available");
    };

    assert_eq!(
        frag_reuse_index_meta.fragment_bitmap.clone().unwrap(),
        expected_all_new_frag_bitmap
    );
    let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
        .await
        .unwrap();
    let frag_reuse_index =
        open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
            .await
            .unwrap();
    let stats = CompactFragReuseIndexHandle(Arc::new(frag_reuse_index.clone()))
        .statistics()
        .unwrap();
    assert_eq!(
        serde_json::to_string(&stats).unwrap(),
        dataset
            .index_statistics(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
    );

    // Verify the index has one version with the correct dataset version
    let compaction_version = &frag_reuse_index.details.versions[0];
    assert_eq!(frag_reuse_index.details.versions.len(), 1);
    assert_eq!(
        compaction_version.dataset_version,
        frag_reuse_index_meta.dataset_version
    );

    // Verify the index compaction version information matches the RewriteResults
    let mut compacted_all_old_frag_digests = Vec::new();
    let mut compacted_all_new_frag_digests = Vec::new();
    let mut transposed_map = HashMap::new();
    for group in compaction_version.groups.iter() {
        let changed_row_addr_bytes = &group.changed_row_addrs;
        let mut cursor = Cursor::new(&changed_row_addr_bytes);
        let changed_row_addrs = RoaringTreemap::deserialize_from(&mut cursor).unwrap();
        compacted_all_old_frag_digests.extend(group.old_frags.clone());
        compacted_all_new_frag_digests.extend(group.new_frags.clone());

        let group_transposed_map =
            transpose_row_ids_from_digest(changed_row_addrs, &group.old_frags, &group.new_frags);
        transposed_map.extend(group_transposed_map);
    }
    assert_eq!(transposed_map, expected_all_row_id_map);
    assert_eq!(
        compacted_all_old_frag_digests
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        expected_all_old_frag_ids
    );
    assert_eq!(
        compacted_all_new_frag_digests
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>(),
        expected_all_new_frag_ids
    );

    // Verify the scalar index UUID is unchanged (it should not be remapped yet)
    let Some(current_scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
        panic!("scalar index must be available");
    };
    assert_eq!(current_scalar_index.uuid, original_scalar_uuid);
}
#[tokio::test]
async fn test_defer_index_remap_skips_fri_when_no_indexed_data() {
    // A deferred compaction touching no indexed data must write no FRI --
    // such a version is un-drainable (remap no-ops, trim retains it forever).
    let mut data_gen =
        BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_owned())));

    let mut dataset = Dataset::write(
        data_gen.batch(600),
        "memory://test/noindex",
        Some(WriteParams {
            max_rows_per_file: 100, // 6 small files -> compaction has work
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // No index at all: nothing covers any fragment.
    assert!(dataset.load_indices().await.unwrap().is_empty());
    let fragments_before = dataset.get_fragments().len();
    assert!(fragments_before > 1, "need multiple fragments to compact");

    let options = CompactionOptions {
        target_rows_per_fragment: 100_000,
        defer_index_remap: true,
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    // Compaction actually ran...
    assert!(
        dataset.get_fragments().len() < fragments_before,
        "compaction should have merged fragments"
    );
    // ...but no fragment-reuse index was created.
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_none(),
        "deferred compaction with no indexed data must not create an FRI"
    );
}
#[tokio::test]
async fn test_defer_index_remap_multiple_compactions() {
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

    // FRI is written only for compactions touching indexed data; index "i" so
    // the successive deferred compactions build a chained fragment-reuse index.
    create_scalar_index(&mut dataset, "i", false).await;

    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };

    let mut prepublish_versions = Vec::new();
    for i in 0..10 {
        dataset
            .delete(&format!("i < {}", 500 * (i + 1)))
            .await
            .unwrap();
        let read_version = dataset.manifest.version;
        compact_files(&mut dataset, options.clone(), None)
            .await
            .unwrap();

        if dataset.manifest.version > read_version {
            prepublish_versions.push(dataset.manifest.version - 1);
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

        // Verify the index has one version with the correct dataset version
        assert_eq!(
            frag_reuse_index
                .details
                .versions
                .iter()
                .map(|v| v.dataset_version)
                .collect::<Vec<_>>(),
            prepublish_versions
        );
    }
}
/// With stable row ids, groups are recorded even when none touches indexed data.
#[rstest]
#[case::indexed_and_unindexed_groups(false)]
#[case::only_unindexed_groups_with_stable_row_ids(true)]
#[tokio::test]
async fn test_defer_index_remap_mixed_records_all_groups(#[case] enable_stable_row_ids: bool) {
    // All-or-nothing: a compaction touching any indexed data records the full
    // FRI, including the unindexed group (a per-group filter would drop it).
    let mut data_gen =
        BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_owned())));
    let mut dataset = Dataset::write(
        data_gen.batch(300),
        "memory://test/mixed",
        Some(WriteParams {
            max_rows_per_file: 100, // 3 fragments
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Index the initial fragments, then append more that stay unindexed.
    if enable_stable_row_ids {
        dataset
            .create_index(
                &["i"],
                IndexType::ZoneMap,
                Some("scalar".into()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
                false,
            )
            .await
            .unwrap();
    } else {
        create_scalar_index(&mut dataset, "i", false).await;
    }
    Dataset::write(
        data_gen.batch(300),
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Some(WriteParams {
            max_rows_per_file: 100, // 3 more, unindexed
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.checkout_latest().await.unwrap();

    // Fragments not covered by the scalar index are the "unindexed" ones.
    let indexed: HashSet<u32> = dataset
        .load_index_by_name("scalar")
        .await
        .unwrap()
        .unwrap()
        .fragment_bitmap
        .unwrap()
        .iter()
        .collect();
    let unindexed_frags: Vec<u64> = dataset
        .fragments()
        .iter()
        .map(|f| f.id)
        .filter(|id| !indexed.contains(&(*id as u32)))
        .collect();
    assert!(
        !unindexed_frags.is_empty(),
        "expected some unindexed fragments"
    );

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 100_000,
            defer_index_remap: true,
            excluded_fragment_ids: if enable_stable_row_ids {
                indexed.iter().copied().collect()
            } else {
                vec![]
            },
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    // All-or-nothing: because indexed fragments were compacted, the FRI is
    // written AND records the unindexed group too (a per-group filter would
    // have dropped it).
    let fri_meta = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("mixed compaction must write an FRI");
    let details = load_frag_reuse_index_details(&dataset, &fri_meta)
        .await
        .unwrap();
    let recorded_old: HashSet<u64> = details
        .versions
        .iter()
        .flat_map(|v| v.old_frag_ids())
        .collect();
    for f in &unindexed_frags {
        assert!(
            recorded_old.contains(f),
            "unindexed fragment {f} must be recorded in the FRI (all-or-nothing)"
        );
    }
}
#[tokio::test]
async fn test_deferred_compaction_not_split_by_frag_reuse_index() {
    // The fragment-reuse index is a system index and must be excluded from
    // compaction bin planning; otherwise its covered fragment is isolated and
    // the small fragments never coalesce back to one.
    let data = sample_data();
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let options = CompactionOptions {
        defer_index_remap: true,
        ..Default::default()
    };

    // Two small fragments -> deferred compaction folds them into one,
    // creating the fragment-reuse index.
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 400))], data.schema());
    let mut dataset = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 200,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Index "a" so the deferred compaction records an FRI (only written for
    // compactions touching indexed data). The FRI is a system index and must
    // still not split later compaction bins -- the property this test guards.
    create_scalar_index(&mut dataset, "a", false).await;
    compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some()
    );

    // Append two more small fragments, then compact again.
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(400, 400))], data.schema());
    let mut dataset = Dataset::write(
        reader,
        test_uri,
        Some(WriteParams {
            max_rows_per_file: 200,
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);

    // Reindex so every fragment is data-indexed -- then the FRI (a system
    // index, correctly excluded from bin planning) is the only thing that
    // could split the bin.
    create_scalar_index(&mut dataset, "a", true).await;

    compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(
        dataset.get_fragments().len(),
        1,
        "FRI (a system index) must not split the compaction bin; all fragments coalesce"
    );
}
#[tokio::test]
async fn test_concurrent_compaction_reindex_compaction_commit_first() {
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

    // Create an index
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

    // Write some more data for reindexing
    Dataset::write(
        data_gen.batch(6_000),
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Some(WriteParams {
            max_rows_per_file: 1_000, // 6 files
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    dataset.checkout_latest().await.unwrap();
    let mut dataset_clone = dataset.clone();

    // First commit a compaction with deferred remap
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

    // Concurrent reindex should succeed
    dataset_clone
        .create_index(
            &["i"],
            IndexType::Scalar,
            index_name.clone(),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // Check new index does not cover the compacted files
    dataset.checkout_latest().await.unwrap();

    let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
        panic!("scalar index must be available");
    };
    let index_frags = scalar_index
        .fragment_bitmap
        .unwrap()
        .iter()
        .collect::<HashSet<_>>();
    assert_eq!(
        index_frags,
        dataset
            .fragments()
            .iter()
            .map(|f| f.id as u32)
            .collect::<HashSet<_>>()
    )
}
#[tokio::test]
async fn test_concurrent_compaction_reindex_reindex_commit_first() {
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

    // Create an index
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

    // Write some more data for reindexing
    Dataset::write(
        data_gen.batch(6_000),
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Some(WriteParams {
            max_rows_per_file: 1_000, // 6 files
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    dataset.checkout_latest().await.unwrap();
    let mut dataset_clone = dataset.clone();

    // Concurrent reindex should succeed
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            index_name.clone(),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();

    // First commit a compaction with deferred remap
    compact_files(
        &mut dataset_clone,
        CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    // Check new index is auto-remapped
    dataset.checkout_latest().await.unwrap();
    let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
        panic!("scalar index must be available");
    };
    let index_frags = scalar_index
        .fragment_bitmap
        .unwrap()
        .iter()
        .collect::<HashSet<_>>();
    assert_eq!(
        index_frags,
        dataset
            .fragments()
            .iter()
            .map(|f| f.id as u32)
            .collect::<HashSet<_>>()
    )
}
#[tokio::test]
async fn test_cleanup_after_a_stale_remap_keeps_later_compactions() {
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
        )
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    // Index "i" so the deferred compaction touches indexed data and writes an FRI.
    create_scalar_index(&mut dataset, "i", false).await;

    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };

    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let tasks = plan.tasks();

    // Only compact the first task, record the state of the dataset
    let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
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

    let mut dataset_clone = dataset.clone();

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
    assert_eq!(frag_reuse_details.versions.len(), 1);

    // First commit the remaining 2 compaction tasks.
    let rewrite_result2 = rewrite_files(Cow::Borrowed(&dataset), tasks[1].clone(), &options)
        .await
        .unwrap();
    let rewritten_frags2 = rewrite_result2
        .original_fragments
        .iter()
        .map(|f| f.id)
        .collect::<Vec<_>>();
    commit_compaction(
        &mut dataset,
        Vec::from([rewrite_result2]),
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    // Get the new fragment IDs from the frag_reuse_index after commit
    let frag_reuse_index_meta2 = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .unwrap();
    let frag_reuse_details2 = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta2)
        .await
        .unwrap();
    let new_frags2 = frag_reuse_details2.versions.last().unwrap().new_frag_ids();

    let rewrite_result3 = rewrite_files(Cow::Borrowed(&dataset), tasks[2].clone(), &options)
        .await
        .unwrap();
    let rewritten_frags3 = rewrite_result3
        .original_fragments
        .iter()
        .map(|f| f.id)
        .collect::<Vec<_>>();
    commit_compaction(
        &mut dataset,
        Vec::from([rewrite_result3]),
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    // Get the new fragment IDs from the frag_reuse_index after commit
    let frag_reuse_index_meta3 = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .unwrap();
    let frag_reuse_details3 = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta3)
        .await
        .unwrap();
    let new_frags3 = frag_reuse_details3.versions.last().unwrap().new_frag_ids();

    // dataset_clone only knows the first reuse version. Its remap commits at the
    // latest version, so the cleanup trims only that version.
    remapping::remap_column_index(&mut dataset_clone, &["i"], Some("scalar".into()))
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset_clone).await.unwrap();

    // Load and verify the fragment reuse index content
    dataset.checkout_latest().await.unwrap();
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
    assert_eq!(frag_reuse_details.versions.len(), 2);
    assert_eq!(
        frag_reuse_details.versions[0].old_frag_ids(),
        rewritten_frags2
    );
    assert_eq!(frag_reuse_details.versions[0].new_frag_ids(), new_frags2);
    assert_eq!(
        frag_reuse_details.versions[1].old_frag_ids(),
        rewritten_frags3
    );
    assert_eq!(frag_reuse_details.versions[1].new_frag_ids(), new_frags3);
}
#[tokio::test]
async fn test_concurrent_cleanup_and_compaction_rebase_compaction() {
    let mut dataset = lance_datagen::gen_batch()
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
        )
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    // Index "i" so the deferred compaction touches indexed data and writes an FRI.
    create_scalar_index(&mut dataset, "i", false).await;

    let options = CompactionOptions {
        target_rows_per_fragment: 2_000,
        defer_index_remap: true,
        ..Default::default()
    };

    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let tasks = plan.tasks();

    // Only compact the first task, record the state of the dataset
    let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
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

    let mut dataset_clone = dataset.clone();

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
    assert_eq!(frag_reuse_details.versions.len(), 1);

    // Catch the index up to the compaction (on `dataset` only; `dataset_clone`
    // keeps the un-caught-up index for the concurrent rewrite below), then
    // clean up: with the index caught up the trim removes the first version.
    remapping::remap_column_index(&mut dataset, &["i"], Some("scalar".into()))
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();

    // Load and verify the fragment reuse index content
    dataset.checkout_latest().await.unwrap();
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
    assert_eq!(frag_reuse_details.versions.len(), 0);

    // Concurrently commit a rewrite
    // After rebase it should only contain the latest reuse version
    let rewrite_result2 = rewrite_files(Cow::Borrowed(&dataset_clone), tasks[1].clone(), &options)
        .await
        .unwrap();
    let rewritten_frags2 = rewrite_result2
        .original_fragments
        .iter()
        .map(|f| f.id)
        .collect::<Vec<_>>();
    commit_compaction(
        &mut dataset_clone,
        Vec::from([rewrite_result2]),
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    // Load and verify the fragment reuse index content
    dataset.checkout_latest().await.unwrap();
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
    assert_eq!(frag_reuse_details.versions.len(), 1);
    assert_eq!(
        frag_reuse_details.versions[0].old_frag_ids(),
        rewritten_frags2
    );
    // Verify new fragment IDs are non-zero (allocated by commit_compaction)
    let new_frags2 = frag_reuse_details.versions[0].new_frag_ids();
    assert!(new_frags2.iter().all(|id| *id != 0));
}
/// Two external entries of the same size each load their own details from one session.
#[tokio::test]
async fn test_reuse_details_cache_keeps_entries_of_one_size_apart() {
    use crate::index::frag_reuse::build_frag_reuse_index_metadata;
    use crate::utils::test::{inline_padding_capacity, padding_reuse_versions};
    let dir = TempStrDir::default();
    let dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_dataset(
            dir.as_str(),
            FragmentCount::from(1),
            FragmentRowCount::from(10),
        )
        .await
        .unwrap();
    let count = inline_padding_capacity() as u64 + 1;
    let mut entries = Vec::new();
    for stamp in [1, 2] {
        let mut versions = padding_reuse_versions(count);
        for version in &mut versions {
            version.dataset_version = stamp;
        }
        let details = lance_index::frag_reuse::FragReuseIndexDetails { versions };
        let entry =
            build_frag_reuse_index_metadata(&dataset, None, details.clone(), RoaringBitmap::new())
                .await
                .unwrap();
        entries.push((entry, details));
    }
    let sizes: Vec<_> = entries
        .iter()
        .map(|(entry, _)| {
            match entry
                .index_details
                .as_ref()
                .unwrap()
                .to_msg::<lance_table::format::pb::FragmentReuseIndexDetails>()
                .unwrap()
                .content
            {
                Some(lance_table::format::pb::fragment_reuse_index_details::Content::External(
                    file,
                )) => file.size,
                other => panic!("expected external details, got {other:?}"),
            }
        })
        .collect();
    assert_eq!(sizes[0], sizes[1]);
    for (entry, details) in &entries {
        let loaded = load_frag_reuse_index_details(&dataset, entry)
            .await
            .unwrap();
        assert_eq!(loaded.as_ref(), details);
    }
}
/// Versions written out of stamp order load from memory as a read of the file decodes them.
#[tokio::test]
async fn test_reuse_details_cache_holds_what_a_read_decodes() {
    use crate::index::frag_reuse::build_frag_reuse_index_metadata;
    use crate::utils::test::{inline_padding_capacity, padding_reuse_versions};
    let dir = TempStrDir::default();
    let dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_dataset(
            dir.as_str(),
            FragmentCount::from(1),
            FragmentRowCount::from(10),
        )
        .await
        .unwrap();
    let mut versions = padding_reuse_versions(inline_padding_capacity() as u64 + 1);
    let count = versions.len() as u64;
    for (i, version) in versions.iter_mut().enumerate() {
        version.dataset_version = count - i as u64;
    }
    let entry = build_frag_reuse_index_metadata(
        &dataset,
        None,
        lance_index::frag_reuse::FragReuseIndexDetails { versions },
        RoaringBitmap::new(),
    )
    .await
    .unwrap();
    assert!(stores_reuse_details_externally(&entry));

    let cached = load_frag_reuse_index_details(&dataset, &entry)
        .await
        .unwrap();
    let uncached = crate::dataset::builder::DatasetBuilder::from_uri(dir.as_str())
        .with_session(Arc::new(crate::session::Session::with_index_cache_backend(
            Arc::new(lance_core::cache::MokaCacheBackend::no_cache()),
            128 * 1024 * 1024,
            Default::default(),
        )))
        .load()
        .await
        .unwrap();
    let read = load_frag_reuse_index_details(&uncached, &entry)
        .await
        .unwrap();
    assert_eq!(cached, read);
    assert!(
        cached
            .versions
            .is_sorted_by_key(|version| version.dataset_version)
    );
}
/// An uncontended commit publishes right after its reservation, so its reuse
/// details are written once and no provisional copy is left behind. With a
/// cache, the history check reads nothing back; without one it reads the files
/// and reaches the same outcome.
#[rstest]
#[tokio::test]
async fn test_deferred_compaction_writes_reuse_details_once(
    #[values(true, false)] external_before: bool,
    #[values(true, false)] cached: bool,
) {
    use crate::utils::test::{commit_padding_reuse_history, inline_padding_capacity};
    let dir = TempStrDir::default();
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_dataset_with_params(
            dir.as_str(),
            FragmentCount::from(4),
            FragmentRowCount::from(100),
            Some(WriteParams {
                max_rows_per_file: 100,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    create_scalar_index(&mut dataset, "i", false).await;
    // Scattered deletions make the compaction's own version larger than a padding one.
    dataset.delete("i % 3 = 1").await.unwrap();
    let capacity = inline_padding_capacity();
    let history = commit_padding_reuse_history(
        &mut dataset,
        if external_before {
            capacity + 1
        } else {
            capacity
        },
    )
    .await;
    assert_eq!(stores_reuse_details_externally(&history), external_before);

    let session = if cached {
        crate::session::Session::default()
    } else {
        crate::session::Session::with_index_cache_backend(
            Arc::new(lance_core::cache::MokaCacheBackend::no_cache()),
            128 * 1024 * 1024,
            Default::default(),
        )
    };
    let options = CompactionOptions {
        target_rows_per_fragment: 200,
        defer_index_remap: true,
        excluded_fragment_ids: fragments_except(&dataset, &[0, 1]),
        ..Default::default()
    };
    let result = plan_one_task(&dataset, &options).await;
    // Committed through a session that has read nothing yet.
    let mut dataset = crate::dataset::builder::DatasetBuilder::from_uri(dir.as_str())
        .with_session(Arc::new(session))
        .load()
        .await
        .unwrap();
    let dirs_before = reuse_details_dirs(dir.as_str());
    dataset.object_store.io_stats_incremental();
    commit_one(&mut dataset, result, &options).await.unwrap();
    let stats = dataset.object_store.io_stats_incremental();

    assert_eq!(reuse_details_requests(&stats, "put"), 1);
    // With a cache, only the stored history is read, once.
    let reads = reuse_details_requests(&stats, "get");
    if cached {
        assert_eq!(reads, usize::from(external_before));
    } else {
        assert!(reads > usize::from(external_before));
    }
    let entry = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .unwrap();
    assert!(stores_reuse_details_externally(&entry));
    let written: Vec<_> = reuse_details_dirs(dir.as_str())
        .difference(&dirs_before)
        .cloned()
        .collect();
    assert_eq!(written, vec![entry.uuid.to_string()]);
    let details = load_frag_reuse_index_details(&dataset, &entry)
        .await
        .unwrap();
    assert_eq!(entry.fragment_bitmap, Some(details.new_frag_bitmap()));
    let watermark = dataset.manifest.version - 1;
    assert_eq!(entry.dataset_version, watermark);
    assert_eq!(
        frag_reuse_watermarks(&dataset).await.last(),
        Some(&watermark)
    );
}
/// The second commit must conflict whether or not its session saw the first.
#[rstest]
#[tokio::test]
async fn test_concurrent_compactions_with_defer_index_remap(
    #[values(
        ReuseHistoryChange::Created,
        ReuseHistoryChange::Replaced,
        ReuseHistoryChange::Removed
    )]
    change: ReuseHistoryChange,
    #[values(false, true)] separate_session: bool,
    #[values(false, true)] enable_stable_row_ids: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = indexed_dataset(
        dir.as_str(),
        BuiltinIndexType::ZoneMap,
        enable_stable_row_ids,
    )
    .await;
    if !matches!(change, ReuseHistoryChange::Created) {
        compact_files(
            &mut dataset,
            deferred_compaction_excluding(&[0, 1, 2, 3]),
            None,
        )
        .await
        .unwrap();
    }
    let untouched: Vec<u32> = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .filter(|id| *id > 3)
        .collect();
    let options_a = deferred_compaction_excluding(&[&[2, 3], untouched.as_slice()].concat());
    let options_b = deferred_compaction_excluding(&[&[0, 1], untouched.as_slice()].concat());
    let before = values_by_address(&dataset).await;
    let result_a = plan_one_task(&dataset, &options_a).await;
    let result_b = plan_one_task(&dataset, &options_b).await;
    let open = async || {
        if separate_session {
            Dataset::open(dir.as_str()).await.unwrap()
        } else {
            dataset.clone()
        }
    };
    let mut committer_a = open().await;
    let mut committer_b = open().await;

    if matches!(change, ReuseHistoryChange::Removed) {
        if enable_stable_row_ids {
            // Stable row ids keep their history, so there is no removal to race.
            let error = committer_a
                .drop_index(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            let latest = Dataset::open(dir.as_str()).await.unwrap();
            assert_eq!(frag_reuse_watermarks(&latest).await.len(), 1);
            return;
        }
        committer_a.drop_index(FRAG_REUSE_INDEX_NAME).await.unwrap();
    } else {
        commit_one(&mut committer_a, result_a, &options_a)
            .await
            .unwrap();
    }
    let error = commit_one(&mut committer_b, result_b.clone(), &options_b)
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    let expected_versions = match change {
        ReuseHistoryChange::Created => 1,
        ReuseHistoryChange::Replaced => 2,
        ReuseHistoryChange::Removed => 0,
    };
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        frag_reuse_watermarks(&latest).await.len(),
        expected_versions
    );
    if !matches!(change, ReuseHistoryChange::Removed) {
        assert_mappings_intact(dir.as_str(), &before).await;
    }

    let mut recommitter = Dataset::open(dir.as_str()).await.unwrap();
    commit_one(&mut recommitter, result_b, &options_b)
        .await
        .unwrap();
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        frag_reuse_watermarks(&latest).await.len(),
        expected_versions + 1
    );
    if !matches!(change, ReuseHistoryChange::Removed) {
        assert_mappings_intact(dir.as_str(), &before).await;
    }
}
/// The race lands after the first attempt's check, so only the retry can catch it.
#[rstest]
#[tokio::test]
async fn test_deferred_compaction_rechecks_the_reuse_history_on_every_attempt(
    #[values(false, true)] enable_stable_row_ids: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = indexed_dataset(
        dir.as_str(),
        BuiltinIndexType::ZoneMap,
        enable_stable_row_ids,
    )
    .await;
    let handler = Arc::new(RaceBeforeRewrite::default());
    dataset.commit_handler = handler.clone();
    let before = values_by_address(&dataset).await;
    let options = deferred_compaction_excluding(&[2, 3, 4, 5]);
    let result = plan_one_task(&dataset, &options).await;

    handler.arm(&dataset, vec![Race::CompactElsewhere { excluded: &[0, 1] }]);
    let error = commit_one(&mut dataset, result.clone(), &options)
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    assert!(handler.races.lock().unwrap().is_empty());
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await.len(), 1);
    assert_mappings_intact(dir.as_str(), &before).await;

    let mut recommitter = Dataset::open(dir.as_str()).await.unwrap();
    commit_one(&mut recommitter, result, &options)
        .await
        .unwrap();
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await.len(), 2);
    assert_mappings_intact(dir.as_str(), &before).await;
}
/// Without any index, stable row ids still record moves: of two disjoint
/// deferred compactions from separate sessions, the second retries, and the
/// retry keeps both.
#[tokio::test]
async fn test_deferred_compactions_in_separate_sessions_keep_both_moves() {
    let dir = TempStrDir::default();
    let dataset = stable_row_id_dataset(dir.as_str(), 4).await;
    let before = rows_by_addr(&dataset).await;
    let options_a = deferred_compaction_excluding(&[2, 3]);
    let options_b = deferred_compaction_excluding(&[0, 1]);
    let result_a = plan_one_task(&Dataset::open(dir.as_str()).await.unwrap(), &options_a).await;
    let result_b = plan_one_task(&Dataset::open(dir.as_str()).await.unwrap(), &options_b).await;
    let mut committer_a = Dataset::open(dir.as_str()).await.unwrap();
    let mut committer_b = Dataset::open(dir.as_str()).await.unwrap();
    commit_one(&mut committer_a, result_a, &options_a)
        .await
        .unwrap();
    let error = commit_one(&mut committer_b, result_b.clone(), &options_b)
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await.len(), 1);
    assert_moves_keep_row_identity(&before, &latest).await;

    let mut recommitter = Dataset::open(dir.as_str()).await.unwrap();
    commit_one(&mut recommitter, result_b, &options_b)
        .await
        .unwrap();
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await.len(), 2);
    assert_moves_keep_row_identity(&before, &latest).await;
}
#[rstest]
#[tokio::test]
async fn test_cleanup_racing_a_deferred_compaction_keeps_its_mappings(
    #[values(false, true)] separate_session: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = indexed_dataset(dir.as_str(), BuiltinIndexType::BTree, false).await;
    let before = values_by_address(&dataset).await;
    let options = deferred_compaction_excluding(&fragments_except(&dataset, &[0, 1]));
    compact_files(&mut dataset, options, None).await.unwrap();
    remapping::remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
        .await
        .unwrap();
    let mut cleaner = if separate_session {
        Dataset::open(dir.as_str()).await.unwrap()
    } else {
        dataset.clone()
    };
    let options = deferred_compaction_excluding(&fragments_except(&dataset, &[2, 3]));
    compact_files(&mut dataset, options, None).await.unwrap();

    let error = cleanup_frag_reuse_index(&mut cleaner).await.unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await.len(), 2);
    assert_mappings_intact(dir.as_str(), &before).await;

    cleaner.checkout_latest().await.unwrap();
    cleanup_frag_reuse_index(&mut cleaner).await.unwrap();
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        frag_reuse_watermarks(&latest).await,
        frag_reuse_watermarks(&dataset).await[1..]
    );
    assert_index_answers_like_a_scan(&latest).await;
}
#[rstest]
#[tokio::test]
async fn test_deferred_compaction_over_a_partial_cleanup_keeps_the_remaining_mappings(
    #[values(false, true)] separate_session: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = indexed_dataset(dir.as_str(), BuiltinIndexType::BTree, false).await;
    let options = deferred_compaction_excluding(&fragments_except(&dataset, &[0, 1]));
    compact_files(&mut dataset, options, None).await.unwrap();
    remapping::remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
        .await
        .unwrap();
    let before = values_by_address(&dataset).await;
    let options = deferred_compaction_excluding(&fragments_except(&dataset, &[2, 3]));
    compact_files(&mut dataset, options, None).await.unwrap();
    let history = frag_reuse_watermarks(&dataset).await;
    assert_eq!(history.len(), 2);
    let open = async || {
        if separate_session {
            Dataset::open(dir.as_str()).await.unwrap()
        } else {
            dataset.clone()
        }
    };
    let mut stale = open().await;
    let options = deferred_compaction_excluding(&fragments_except(&stale, &[4, 5]));
    let result = plan_one_task(&stale, &options).await;

    let mut cleaner = open().await;
    cleanup_frag_reuse_index(&mut cleaner).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&cleaner).await, history[1..]);
    commit_one(&mut stale, result, &options).await.unwrap();
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        frag_reuse_watermarks(&latest).await,
        vec![history[1], latest.manifest.version - 1]
    );
    assert_mappings_intact(dir.as_str(), &before).await;
}
#[tokio::test]
async fn test_distributed_deferred_compaction_keeps_mappings_recorded_before_its_handle() {
    let dir = TempStrDir::default();
    let mut dataset = indexed_dataset(dir.as_str(), BuiltinIndexType::BTree, false).await;
    let options = deferred_compaction_excluding(&fragments_except(&dataset, &[0, 1]));
    compact_files(&mut dataset, options, None).await.unwrap();
    let before = values_by_address(&dataset).await;
    let options_b = deferred_compaction_excluding(&fragments_except(&dataset, &[4, 5]));
    let result_b = plan_one_task(&dataset, &options_b).await;

    remapping::remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(frag_reuse_watermarks(&dataset).await.is_empty());
    let options = deferred_compaction_excluding(&fragments_except(&dataset, &[2, 3]));
    compact_files(&mut dataset, options, None).await.unwrap();

    let mut driver = Dataset::open(dir.as_str()).await.unwrap();
    commit_one(&mut driver, result_b, &options_b).await.unwrap();
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await.len(), 2);
    assert_mappings_intact(dir.as_str(), &before).await;
}
/// A compaction without `defer_index_remap` records nothing, whatever
/// history the table has, through `compact_files` and through the
/// distributed API: the history and the feature flags stay as they were,
/// every row keeps its stable row id, and a zone map stops covering the
/// rewritten fragments and still answers as a scan does.
#[rstest]
#[tokio::test]
async fn test_compaction_records_only_when_deferred(
    #[values(History::Absent, History::Empty, History::Recorded)] history: History,
    #[values(false, true)] indexed: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = stable_row_id_dataset(dir.as_str(), 6).await;
    dataset.delete("i IN (1, 2, 5, 13, 21)").await.unwrap();
    match history {
        History::Absent => {}
        History::Empty => commit_empty_frag_reuse_index(&mut dataset).await,
        History::Recorded => {
            let options = deferred_compaction_excluding(&[0, 1, 2, 3]);
            compact_files(&mut dataset, options, None).await.unwrap();
        }
    }
    if indexed {
        dataset
            .create_index(
                &["i"],
                IndexType::ZoneMap,
                Some("i_idx".into()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
                false,
            )
            .await
            .unwrap();
    }
    let watermarks = frag_reuse_watermarks(&dataset).await;
    let flags = flag_words(&dataset, FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS);
    let fragments_before = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<RoaringBitmap>();
    let before = rows_by_addr(&dataset).await;

    let mut eager = Dataset::open(dir.as_str()).await.unwrap();
    let options = eager_compaction_excluding(&[0, 1, 2, 3]);
    compact_files(&mut eager, options, None).await.unwrap();
    let planner = Dataset::open(dir.as_str()).await.unwrap();
    let options = eager_compaction_excluding(&[0, 1]);
    let result = plan_one_task(&planner, &options).await;
    assert!(result.row_addrs.is_none());
    let mut committer = Dataset::open(dir.as_str()).await.unwrap();
    commit_one(&mut committer, result, &CompactionOptions::default())
        .await
        .unwrap();

    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await, watermarks);
    assert_eq!(
        latest.frag_reuse_index().await.unwrap().is_some(),
        !matches!(history, History::Absent)
    );
    assert_eq!(
        flag_words(&latest, FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS),
        flags
    );
    assert_rows_keep_their_ids(&before, &latest).await;
    if indexed {
        let coverage = latest
            .load_indices_by_name("i_idx")
            .await
            .unwrap()
            .iter()
            .filter_map(|index| index.fragment_bitmap.clone())
            .fold(RoaringBitmap::new(), |coverage, bitmap| coverage | bitmap);
        let untouched = latest
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<RoaringBitmap>()
            & fragments_before;
        assert_eq!(coverage, untouched);
        assert_index_answers_like_a_scan(&latest).await;
    }
}
/// A second recorded compaction keeps row identity however the rewrite
/// copies rows, also for a group mixing a fragment the first one wrote with
/// one never rewritten: every row read before either compaction translates
/// through the whole history to the row with the same stable row id and
/// value, read from a cold handle.
#[rstest]
#[case::binary_copy(SecondRewrite::BinaryCopy, false)]
#[case::binary_copy_over_recorded_output(SecondRewrite::BinaryCopy, true)]
#[case::scan(SecondRewrite::Scan, false)]
#[case::scan_over_recorded_output(SecondRewrite::Scan, true)]
#[case::spilled_binary_copy(SecondRewrite::SpilledBinaryCopy, false)]
#[case::spilled_scan(SecondRewrite::SpilledScan, false)]
#[case::spilled_scan_over_recorded_output(SecondRewrite::SpilledScan, true)]
#[tokio::test]
async fn test_repeated_recording_keeps_row_identity(
    #[case] rewrite: SecondRewrite,
    #[case] crosses_recorded_coverage: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = stable_row_id_dataset(dir.as_str(), 4).await;
    let spills = matches!(
        rewrite,
        SecondRewrite::SpilledBinaryCopy | SecondRewrite::SpilledScan
    );
    let scans = matches!(rewrite, SecondRewrite::Scan | SecondRewrite::SpilledScan);
    // Fragment 0 makes row ids and addresses diverge.
    dataset.delete("i IN (1, 2)").await.unwrap();
    let mut before = rows_by_addr(&dataset).await;
    let options = deferred_compaction_excluding(&[0, 1]);
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let task = plan.compaction_tasks().next().unwrap();
    let first_sources = task
        .task
        .fragments
        .iter()
        .map(|fragment| fragment.id)
        .collect::<Vec<_>>();
    assert_eq!(first_sources, vec![2, 3]);
    let result = task.execute(&dataset).await.unwrap();
    commit_compaction(
        &mut dataset,
        vec![result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    let cat = [
        "needle", "haystack", "needle", "haystack", "needle", "haystack", "needle", "haystack",
        "needle", "haystack",
    ];
    for values in [
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    ] {
        let batch =
            arrow_array::record_batch!(("i", Int32, values.to_vec()), ("cat", Utf8, cat.to_vec()))
                .unwrap();
        dataset
            .append(
                arrow_array::RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();
    }
    if spills {
        dataset
            .update_config([
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
    }
    for (address, row) in rows_by_addr(&dataset).await {
        if address >> 32 >= 5 {
            before.insert(address, row);
        }
    }

    let mut second = Dataset::open(dir.as_str()).await.unwrap();
    let newest = newest_fragment(&second);
    assert_eq!(newest, 6);
    let sources = if crosses_recorded_coverage {
        vec![newest - 2, newest - 1, newest]
    } else {
        vec![newest - 1, newest]
    };
    if scans {
        let removed_value = if crosses_recorded_coverage { 25 } else { 105 };
        second
            .delete(&format!("i = {removed_value}"))
            .await
            .unwrap();
        before.retain(|_, (_, value)| *value != removed_value);
    }

    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        compaction_mode: Some(if scans {
            CompactionMode::TryBinaryCopy
        } else {
            CompactionMode::ForceBinaryCopy
        }),
        ..deferred_compaction_excluding(&fragments_except(&second, &sources))
    };
    let source_fragments = second
        .manifest
        .fragments
        .iter()
        .filter(|fragment| sources.contains(&(fragment.id as u32)))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        can_use_binary_copy(&second, &options, &source_fragments).await,
        !scans
    );
    let before_second = rows_by_addr(&second).await;
    let metrics = compact_files(&mut second, options, None).await.unwrap();
    assert_eq!(metrics.fragments_removed, sources.len());

    let latest = Dataset::open(dir.as_str()).await.unwrap();
    let frag_reuse = latest.frag_reuse_index().await.unwrap().unwrap();
    assert_eq!(frag_reuse.details.versions.len(), 2);
    let recorded = &frag_reuse.details.versions[1];
    assert_eq!(recorded.groups.len(), 1);
    let recorded_sources = recorded.groups[0]
        .old_frags
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect::<Vec<_>>();
    assert_eq!(recorded_sources, sources);
    assert_eq!(
        flag_words(&latest, FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS),
        (true, true)
    );
    if spills {
        for id in recorded.new_frag_ids() {
            let fragment = latest
                .manifest
                .fragments
                .iter()
                .find(|f| f.id == id)
                .unwrap();
            assert!(matches!(fragment.row_id_meta, Some(RowIdMeta::Column)));
            let carrier = if scans { 0 } else { 1 };
            assert_eq!(fragment.files.len(), carrier + 1);
            assert_eq!(
                fragment.row_lineage_file(ROW_ID_FIELD_ID).unwrap(),
                Some(&fragment.files[carrier])
            );
        }
    }
    assert_moves_keep_row_identity(&before, &latest).await;
    assert_moves_keep_row_identity(&before_second, &latest).await;
}
/// A zone map keeps answering through the index, over every fragment and as
/// a forced scan does, after a recorded compaction, a second one whose group
/// crosses the first one's output, and index maintenance.
#[rstest]
#[tokio::test]
async fn test_zonemap_answers_through_repeated_recording(
    #[values(
        ZoneMapMaintenance::None,
        ZoneMapMaintenance::OptimizeIndices,
        ZoneMapMaintenance::MergeSegments,
        ZoneMapMaintenance::Cleanup,
        ZoneMapMaintenance::AppendAndOptimize
    )]
    maintenance: ZoneMapMaintenance,
) {
    let dir = TempStrDir::default();
    let mut dataset = indexed_dataset(dir.as_str(), BuiltinIndexType::ZoneMap, true).await;
    let before = values_by_address(&dataset).await;
    let options = deferred_compaction_excluding(&[0, 1, 2, 3]);
    compact_files(&mut dataset, options, None).await.unwrap();
    assert_index_serves_every_fragment(dir.as_str()).await;

    let mut second = Dataset::open(dir.as_str()).await.unwrap();
    let sources = [3, newest_fragment(&second)];
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        ..deferred_compaction_excluding(&fragments_except(&second, &sources))
    };
    let metrics = compact_files(&mut second, options, None).await.unwrap();
    assert_eq!(metrics.fragments_removed, 2);
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await.len(), 2);
    assert_mappings_intact(dir.as_str(), &before).await;
    assert_index_serves_every_fragment(dir.as_str()).await;

    let mut maintainer = Dataset::open(dir.as_str()).await.unwrap();
    match maintenance {
        ZoneMapMaintenance::None => {}
        ZoneMapMaintenance::OptimizeIndices => {
            maintainer
                .optimize_indices(&lance_index::optimize::OptimizeOptions::default())
                .await
                .unwrap();
        }
        ZoneMapMaintenance::MergeSegments => {
            let segments = maintainer.load_indices_by_name("i_idx").await.unwrap();
            let merged = maintainer
                .merge_existing_index_segments(segments)
                .await
                .unwrap();
            maintainer
                .commit_existing_index_segments("i_idx", "i", vec![merged])
                .await
                .unwrap();
        }
        ZoneMapMaintenance::Cleanup => {
            cleanup_frag_reuse_index(&mut maintainer).await.unwrap();
        }
        ZoneMapMaintenance::AppendAndOptimize => {
            let batch = arrow_array::record_batch!(("i", Int32, [1000, 1001, 1002])).unwrap();
            maintainer
                .append(
                    RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                    None,
                )
                .await
                .unwrap();
            maintainer
                .optimize_indices(&lance_index::optimize::OptimizeOptions::default())
                .await
                .unwrap();
        }
    }
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(frag_reuse_watermarks(&latest).await.len(), 2);
    assert_index_serves_every_fragment(dir.as_str()).await;
}
/// Unrecorded tasks planned before the table started recording commit
/// after it without recording: the history keeps only the recorded
/// compaction, and every row keeps its stable row id.
#[tokio::test]
async fn test_unrecorded_tasks_commit_after_recording_starts() {
    let dir = TempStrDir::default();
    let dataset = stable_row_id_dataset(dir.as_str(), 4).await;
    // The stale task keeps the leading pair, while the recorded compaction
    // rewrites the trailing pair and leaves the leading sources intact.
    let options = eager_compaction_excluding(&[]);
    let stale = execute_first_fragment_pair(&dataset, &options).await;
    assert!(stale.row_addrs.is_none());

    let mut recorder = Dataset::open(dir.as_str()).await.unwrap();
    compact_files(&mut recorder, deferred_compaction_excluding(&[0, 1]), None)
        .await
        .unwrap();
    let recorded = Dataset::open(dir.as_str()).await.unwrap();
    let watermarks = frag_reuse_watermarks(&recorded).await;
    assert_eq!(watermarks.len(), 1);
    let before = rows_by_addr(&recorded).await;

    let mut committer = Dataset::open(dir.as_str()).await.unwrap();
    commit_one(&mut committer, stale, &options).await.unwrap();
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert!(
        latest
            .manifest
            .fragments
            .iter()
            .all(|fragment| fragment.id > 3),
        "{:?}",
        latest.manifest.fragments
    );
    assert_eq!(frag_reuse_watermarks(&latest).await, watermarks);
    assert_rows_keep_their_ids(&before, &latest).await;
}
/// The first recorded compaction lands over an unrecorded rewrite of other
/// fragments, whether it sees that rewrite through this session's cache or
/// a transaction file, and records only its own sources.
#[rstest]
#[tokio::test]
async fn test_first_recorded_compaction_lands_over_an_unrecorded_rewrite(
    #[values(false, true)] separate_session: bool,
) {
    let dir = TempStrDir::default();
    let dataset = stable_row_id_dataset(dir.as_str(), 4).await;
    let open = async || {
        if separate_session {
            Dataset::open(dir.as_str()).await.unwrap()
        } else {
            dataset.clone()
        }
    };
    let before = rows_by_addr(&dataset).await;
    let options = deferred_compaction_excluding(&[]);
    let recorded = execute_first_fragment_pair(&dataset, &options).await;
    let mut committer = open().await;

    let mut eager = open().await;
    compact_files(&mut eager, eager_compaction_excluding(&[0, 1]), None)
        .await
        .unwrap();
    let rewritten = Dataset::open(dir.as_str()).await.unwrap();
    assert!(rewritten.frag_reuse_index().await.unwrap().is_none());

    commit_one(&mut committer, recorded, &options)
        .await
        .unwrap();
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_recorded_fragments(&before, &latest, &[0, 1]).await;
}
/// An unrecorded rewrite of other fragments landing while the first
/// recorded compaction commits is rebased over, and the recorded
/// compaction lands with only its own sources.
#[tokio::test]
async fn test_first_recorded_compaction_rebases_over_an_unrecorded_rewrite() {
    let dir = TempStrDir::default();
    let mut dataset = stable_row_id_dataset(dir.as_str(), 4).await;
    let handler = Arc::new(RaceBeforeRewrite::default());
    dataset.commit_handler = handler.clone();
    let before = rows_by_addr(&dataset).await;
    let options = deferred_compaction_excluding(&[]);
    let result = execute_first_fragment_pair(&dataset, &options).await;

    handler.arm(
        &dataset,
        vec![Race::CompactEagerlyElsewhere { excluded: &[0, 1] }],
    );
    commit_one(&mut dataset, result, &options).await.unwrap();
    assert!(handler.races.lock().unwrap().is_empty());
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_recorded_fragments(&before, &latest, &[0, 1]).await;
}
/// The first recorded compaction landing while an unrecorded one of other
/// fragments commits is rebased over, and the unrecorded one lands without
/// recording.
#[tokio::test]
async fn test_unrecorded_compaction_rebases_over_the_first_recording() {
    let dir = TempStrDir::default();
    let mut dataset = stable_row_id_dataset(dir.as_str(), 4).await;
    let handler = Arc::new(RaceBeforeRewrite::default());
    dataset.commit_handler = handler.clone();
    let before = rows_by_addr(&dataset).await;
    let options = eager_compaction_excluding(&[]);
    let result = execute_first_fragment_pair(&dataset, &options).await;
    assert!(result.row_addrs.is_none());

    handler.arm(&dataset, vec![Race::CompactElsewhere { excluded: &[0, 1] }]);
    commit_one(&mut dataset, result, &options).await.unwrap();
    assert!(handler.races.lock().unwrap().is_empty());
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_recorded_fragments(&before, &latest, &[2, 3]).await;
}
/// A recorded and an unrecorded compaction of the same fragments still
/// conflict, in either order: the second is refused as retryable, and the
/// history holds exactly what the first recorded.
#[rstest]
#[tokio::test]
async fn test_recorded_and_unrecorded_compactions_of_the_same_fragments_conflict(
    #[values(false, true)] recorded_first: bool,
) {
    let dir = TempStrDir::default();
    let dataset = stable_row_id_dataset(dir.as_str(), 4).await;
    let deferred = deferred_compaction_excluding(&[0, 1]);
    let eager = eager_compaction_excluding(&[0, 1]);
    let recorded = (plan_one_task(&dataset, &deferred).await, &deferred);
    let unrecorded = (plan_one_task(&dataset, &eager).await, &eager);
    let (first, second) = if recorded_first {
        (recorded, unrecorded)
    } else {
        (unrecorded, recorded)
    };

    let mut committer = Dataset::open(dir.as_str()).await.unwrap();
    commit_one(&mut committer, first.0, first.1).await.unwrap();
    // The tasks' read version anchors the conflict check.
    let mut late = Dataset::open(dir.as_str()).await.unwrap();
    let error = commit_one(&mut late, second.0, second.1).await.unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    let latest = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        frag_reuse_watermarks(&latest).await.len(),
        usize::from(recorded_first)
    );
}
/// The same compaction with `defer_index_remap` keeps the zone map covering
/// its data: the FRI records the address moves and the index applies them on
/// load, then bakes them in when the segments are merged.
#[tokio::test]
async fn test_defer_index_remap_zonemap_with_stable_row_ids() {
    let mut dataset = zonemap_stable_row_id_dataset("memory://").await;
    let coverage_before = zonemap_coverage(&dataset).await;

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

    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some(),
        "a deferred compaction over an address-domain index must write an FRI"
    );

    let coverage_after = zonemap_coverage(&dataset).await;
    assert_ne!(coverage_after, coverage_before);
    assert_eq!(
        coverage_after.len(),
        1,
        "coverage followed to the new fragment"
    );
    assert_zonemap_answers_match_scan(&dataset).await;
    assert_zonemap_is_used(&dataset).await;

    // Draining the FRI into the index must not change any answer.
    let merged = dataset
        .merge_existing_index_segments(dataset.load_indices_by_name("value_idx").await.unwrap())
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("value_idx", "value", vec![merged])
        .await
        .unwrap();

    assert_zonemap_answers_match_scan(&dataset).await;
    assert_zonemap_is_used(&dataset).await;
}
/// NGram stores row addresses too, so -- like the zone map above -- it
/// coexists with the FRI a deferred compaction writes on a stable-row-id
/// dataset, answering through it on load and after it is drained.
#[tokio::test]
async fn test_defer_index_remap_ngram_with_stable_row_ids() {
    async fn assert_ngram_answers(dataset: &Dataset) {
        let filter = "contains(cat, 'cat-1')";
        for use_scalar_index in [false, true] {
            let mut scanner = dataset.scan();
            scanner.filter(filter).unwrap();
            scanner.project(&["id"]).unwrap();
            scanner.use_scalar_index(use_scalar_index);
            let mut ids = scanner.try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values()
                .to_vec();
            ids.sort_unstable();
            assert_eq!(ids, vec![1, 10, 11], "use_scalar_index={use_scalar_index}");
        }
        let mut scanner = dataset.scan();
        scanner.filter(filter).unwrap();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery") && plan.contains("cat_idx(NGram)"),
            "Expected NGram index query in plan: {plan}"
        );
    }

    let mut dataset = zonemap_stable_row_id_dataset("memory://").await;
    dataset
        .create_index(
            &["cat"],
            IndexType::NGram,
            Some("cat_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::NGram),
            false,
        )
        .await
        .unwrap();
    assert_ngram_answers(&dataset).await;

    compact_files(
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
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some(),
        "a deferred compaction over an address-domain index must write an FRI"
    );
    assert_ngram_answers(&dataset).await;

    // Draining the FRI into the index must not change any answer.
    let merged = dataset
        .merge_existing_index_segments(dataset.load_indices_by_name("cat_idx").await.unwrap())
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("cat_idx", "cat", vec![merged])
        .await
        .unwrap();
    assert_ngram_answers(&dataset).await;
}
/// An FRI and a row-id-domain index cannot coexist on a stable-row-id
/// dataset: the FRI is applied to every index on load, and a stable row id
/// is numerically indistinguishable from an address into fragment 0, so it
/// would silently rewrite the Inverted index's valid row ids.
///
/// Compaction refuses to create that pair, at the plan boundary and again at
/// the commit boundary.
#[tokio::test]
async fn test_defer_index_remap_rejected_while_row_id_index_present() {
    let mut dataset = zonemap_stable_row_id_dataset("memory://").await;
    dataset
        .create_index(
            &["cat"],
            IndexType::Inverted,
            Some("id_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::Inverted),
            false,
        )
        .await
        .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 512,
        defer_index_remap: true,
        ..Default::default()
    };

    let plan_err = plan_compaction(&dataset, &options).await.unwrap_err();
    assert!(matches!(plan_err, Error::InvalidInput { .. }));
    let msg = plan_err.to_string();
    assert!(msg.contains("defer_index_remap"), "{msg}");
    assert!(msg.contains("id_idx"), "{msg}");

    // Nothing is rewritten, and the zone map keeps the coverage it had.
    let version_before = dataset.manifest.version;
    let compact_err = compact_files(&mut dataset, options, None)
        .await
        .unwrap_err();
    assert!(matches!(compact_err, Error::InvalidInput { .. }));
    assert_eq!(dataset.manifest.version, version_before);
    assert_eq!(zonemap_coverage(&dataset).await.len(), 3);
}
/// The mirror image: once an FRI is live, a row-id-domain index cannot be
/// added, because the FRI would be applied to it on load.
#[tokio::test]
async fn test_create_row_id_domain_index_rejected_under_frag_reuse() {
    let mut dataset = zonemap_stable_row_id_dataset("memory://").await;
    compact_files(
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
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some()
    );

    let err = dataset
        .create_index(
            &["cat"],
            IndexType::Inverted,
            Some("id_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::Inverted),
            false,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput { .. }));
    let msg = err.to_string();
    assert!(msg.contains("fragment reuse index"), "{msg}");

    // Address-domain indices are still allowed: the FRI repairs them.
    dataset
        .create_index(
            &["id"],
            IndexType::BloomFilter,
            Some("id_bloom".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter),
            false,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_btree".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();
    assert_zonemap_answers_match_scan(&dataset).await;
    assert_id_idx_answers_every_row(&dataset, "id_btree").await;
}
/// Index creation and a deferred compaction race, and each one's guard looks
/// at a snapshot taken before the other committed.
///
/// The compaction plans and rewrites while no row-id-domain index exists, so
/// its plan-time guard passes. An Inverted index is then created and commits: no
/// FRI exists yet, so its guard passes too. The compaction finally commits
/// from the handle it planned on, which still cannot see the Inverted index,
/// and the rewrite rebases on top of the `CreateIndex`.
///
/// The result is the combination both guards exist to prevent: stable row
/// ids, an FRI, and a row-id-domain index whose row ids the FRI will rewrite
/// on load.
///
/// Fails two ways, both real. In a debug build the invariant's
/// `debug_assert` in `manifest_build` trips first. In a release build there
/// is nothing to stop it: the commit returns `Ok` and the invalid
/// combination is durably committed, which the final assertion here catches.
#[tokio::test]
async fn test_defer_index_remap_racing_row_id_index_creation() {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let mut dataset = zonemap_stable_row_id_dataset(test_uri).await;

    let options = CompactionOptions {
        target_rows_per_fragment: 512,
        defer_index_remap: true,
        ..Default::default()
    };

    // Plan and rewrite against a dataset that carries no row-id-domain index.
    let completed = execute_compaction_plan(&dataset, &options).await;

    // A concurrent writer adds one and commits first. No FRI exists yet, so
    // nothing refuses it.
    let mut concurrent = Dataset::open(test_uri).await.unwrap();
    concurrent
        .create_index(
            &["cat"],
            IndexType::Inverted,
            Some("id_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::Inverted),
            false,
        )
        .await
        .unwrap();
    assert!(concurrent.manifest.version > dataset.manifest.version);

    // The compaction commits from its own, now-stale handle.
    let commit_result = commit_compaction(
        &mut dataset,
        completed,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await;

    // Only the rebase can see both sides, so it is what must refuse.
    let commit_err = commit_result
        .expect_err("the rewrite must conflict with the concurrently created row-id-domain index");
    assert!(
        matches!(commit_err, Error::RetryableCommitConflict { .. }),
        "expected RetryableCommitConflict, got: {commit_err:?}"
    );

    // And the invalid combination must not be on disk.
    let after = Dataset::open(test_uri).await.unwrap();
    assert!(after.manifest.uses_stable_row_ids());
    let has_frag_reuse = after
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .is_some();
    let row_id_indices = row_id_domain_indices(&after).await.unwrap();
    assert!(
        !has_frag_reuse || row_id_indices.is_empty(),
        "committed the invalid combination: stable row ids + fragment reuse index + \
             row-id-domain index(es) {row_id_indices:?}"
    );
    assert_id_idx_answers_every_row(&after, "id_idx").await;
}
/// The mirror ordering: the rewrite commits its FRI first, and the index
/// creation -- planned against a version with no FRI -- rebases on top of it.
/// The rebase is again the only place that sees both.
#[tokio::test]
async fn test_row_id_index_creation_racing_defer_index_remap() {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let mut dataset = zonemap_stable_row_id_dataset(test_uri).await;

    // A second handle pinned to the pre-compaction version.
    let mut concurrent = Dataset::open(test_uri).await.unwrap();

    compact_files(
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
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some()
    );

    // `create_index` sees no FRI at its own version, so its guard passes and
    // the transaction reaches the rebase.
    //
    // NGram: now address-domain like BTree/Bitmap, so this no longer
    // illustrates a row-id-domain index racing an FRI. It still conflicts
    // here, though, through a narrower check: `conflict_resolver.rs`'s
    // deferred-remap branch carries an extra, NGram-specific coverage check
    // (mirroring `ngram_requires_rebuild` in append.rs) that retries whenever
    // a new NGram index's coverage overlaps fragments a concurrent
    // deferred-remap rewrite just consumed, regardless of domain.
    let err = concurrent
        .create_index(
            &["cat"],
            IndexType::NGram,
            Some("id_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::NGram),
            false,
        )
        .await
        .expect_err("creating a row-id-domain index must conflict with the committed FRI");
    assert!(
        matches!(
            err,
            Error::RetryableCommitConflict { .. } | Error::InvalidInput { .. }
        ),
        "expected a conflict or a refusal, got: {err:?}"
    );

    let after = Dataset::open(test_uri).await.unwrap();
    assert!(
        row_id_domain_indices(&after).await.unwrap().is_empty(),
        "a row-id-domain index landed alongside the fragment reuse index"
    );
    assert_zonemap_answers_match_scan(&after).await;
}
/// The mirror ordering only conflicts today by accident: the rebase rejects
/// a `CreateIndex` whose coverage overlaps the rewritten fragments. An index
/// built over fragments the compaction left alone does not overlap, so
/// nothing stops it -- and the `Rewrite` transaction read back from its file
/// reports no fragment-reuse index at all, because the proto has no field
/// for one, so the rebase cannot even tell an FRI landed.
#[tokio::test]
async fn test_row_id_index_creation_racing_defer_index_remap_disjoint_coverage() {
    let test_dir = TempStrDir::default();
    let test_uri = test_dir.as_str();
    let mut dataset = zonemap_stable_row_id_dataset(test_uri).await;

    let concurrent = Dataset::open(test_uri).await.unwrap();

    // Compact only fragments 0 and 1, leaving 2 untouched.
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 512,
            defer_index_remap: true,
            excluded_fragment_ids: vec![2],
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
            .is_some()
    );

    // A btree over only the untouched fragment. Its coverage is disjoint from
    // the rewrite, so the incidental overlap rule does not fire.
    let mut concurrent = concurrent;
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let result =
        crate::index::CreateIndexBuilder::new(&mut concurrent, &["id"], IndexType::BTree, &params)
            .name("id_idx".into())
            .fragments(vec![2])
            .await;

    let after = Dataset::open(test_uri).await.unwrap();
    let has_frag_reuse = after
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .is_some();
    let row_id_indices = row_id_domain_indices(&after).await.unwrap();
    assert!(
        !has_frag_reuse || row_id_indices.is_empty(),
        "committed the invalid combination: stable row ids + fragment reuse index + \
             row-id-domain index(es) {row_id_indices:?} (create returned {result:?})"
    );
}
/// The commits use `RenameCommitHandler` so that how the rebase sees the first
/// one does not depend on the platform.
#[rstest]
#[tokio::test]
async fn test_index_creation_racing_deferred_compaction_with_stable_row_ids(
    // Inverted: every other scalar index type now always creates an
    // address-domain segment on a stable-row-id dataset (safe alongside an
    // FRI by construction), so only Inverted still illustrates this race.
    #[values(IndexType::Inverted, IndexType::ZoneMap)] index_type: IndexType,
    #[values(false, true)] index_commits_first: bool,
    #[values(false, true)] is_disjoint: bool,
    #[values(
        FirstCommitSeenAs::InMemory,
        FirstCommitSeenAs::InlineTransaction,
        FirstCommitSeenAs::SeparateSession
    )]
    first_commit_seen_as: FirstCommitSeenAs,
) {
    let is_in_memory = first_commit_seen_as == FirstCommitSeenAs::InMemory;
    let is_separate_session = first_commit_seen_as == FirstCommitSeenAs::SeparateSession;
    // How the rebase sees the compaction does not matter: an index rebasing
    // past it finds the moves recorded in the latest manifest's reuse index.
    let expected = match (index_type, index_commits_first) {
        (IndexType::Inverted, true) => RaceOutcome::ConflictThenRejected,
        (IndexType::Inverted, false) => RaceOutcome::Rejected,
        _ => RaceOutcome::Committed,
    };
    let dir = TempStrDir::default();
    let mut dataset = stable_row_id_dataset(dir.as_str(), 3).await;
    dataset.commit_handler = Arc::new(RenameCommitHandler);
    dataset.delete("i IN (1, 2)").await.unwrap();
    let before = rows_by_addr(&dataset).await;
    let mut other = if is_separate_session {
        let mut other = Dataset::open(dir.as_str()).await.unwrap();
        other.commit_handler = Arc::new(RenameCommitHandler);
        other
    } else {
        dataset.clone()
    };
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        defer_index_remap: true,
        excluded_fragment_ids: if is_disjoint { vec![2] } else { vec![] },
        ..Default::default()
    };
    let params = ScalarIndexParams::for_builtin(if index_type == IndexType::Inverted {
        BuiltinIndexType::Inverted
    } else {
        BuiltinIndexType::ZoneMap
    });
    // Inverted only applies to the text `cat` column; ZoneMap stays on the
    // numeric `i` column the filters below query.
    let index_column = if index_type == IndexType::Inverted {
        "cat"
    } else {
        "i"
    };
    let create_index = async |dataset: &mut Dataset| {
        let columns = [index_column];
        let builder = dataset
            .create_index_builder(&columns, index_type, &params)
            .name("i_idx".into());
        if is_disjoint {
            builder.fragments(vec![2]).await.map(|_| ())
        } else {
            builder.await.map(|_| ())
        }
    };

    let result = if index_commits_first {
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let mut results = Vec::new();
        for task in plan.compaction_tasks() {
            results.push(task.execute(&dataset).await.unwrap());
        }
        create_index(&mut other).await.unwrap();
        let index_key = TransactionKey {
            version: other.version().version,
        };
        let committed_index = dataset.metadata_cache.get_with_key(&index_key).await;
        if is_separate_session {
            assert!(committed_index.is_none());
        } else {
            assert!(committed_index.is_some());
            let listed = uncached_listed_manifest(&dataset, index_key.version).await;
            if is_in_memory {
                let listed_key = ManifestKey {
                    version: listed.version,
                    e_tag: listed.e_tag.as_deref(),
                };
                dataset
                    .metadata_cache
                    .insert_with_key(&listed_key, other.manifest.clone())
                    .await;
            }
        }
        let result = commit_compaction(
            &mut dataset,
            results.clone(),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .map(|_| ());
        let seen = dataset
            .metadata_cache
            .get_with_key(&index_key)
            .await
            .expect("the rebase caches the index commit it read");
        if let Some(committed_index) = committed_index {
            assert_eq!(Arc::ptr_eq(&seen, &committed_index), is_in_memory);
        }
        if matches!(expected, RaceOutcome::ConflictThenRejected) {
            assert!(
                matches!(result, Err(Error::RetryableCommitConflict { .. })),
                "{result:?}"
            );
            dataset.checkout_latest().await.unwrap();
            commit_compaction(
                &mut dataset,
                results,
                Arc::new(DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
            .map(|_| ())
        } else {
            result
        }
    } else {
        compact_files(&mut dataset, options, None).await.unwrap();
        let rewrite_key = TransactionKey {
            version: dataset.version().version,
        };
        let cached_rewrite = other.metadata_cache.get_with_key(&rewrite_key).await;
        if is_separate_session {
            assert!(cached_rewrite.is_none());
            let from_storage = Dataset::open(dir.as_str())
                .await
                .unwrap()
                .read_transaction_by_version(rewrite_key.version)
                .await
                .unwrap()
                .unwrap();
            assert!(matches!(
                from_storage.operation,
                Operation::Rewrite {
                    frag_reuse_index: None,
                    ..
                }
            ));
        } else {
            assert!(matches!(
                cached_rewrite.unwrap().operation,
                Operation::Rewrite {
                    frag_reuse_index: Some(_),
                    ..
                }
            ));
            let listed = uncached_listed_manifest(&other, rewrite_key.version).await;
            if is_in_memory {
                let listed_key = ManifestKey {
                    version: listed.version,
                    e_tag: listed.e_tag.as_deref(),
                };
                other
                    .metadata_cache
                    .insert_with_key(&listed_key, dataset.manifest.clone())
                    .await;
            }
        }
        let result = create_index(&mut other).await;
        let seen = other
            .metadata_cache
            .get_with_key(&rewrite_key)
            .await
            .expect("the rebase caches the rewrite it read");
        assert_eq!(
            matches!(
                seen.operation,
                Operation::Rewrite {
                    frag_reuse_index: Some(_),
                    ..
                }
            ),
            is_in_memory
        );
        if matches!(expected, RaceOutcome::ConflictThenRejected) {
            assert!(
                matches!(result, Err(Error::RetryableCommitConflict { .. })),
                "{result:?}"
            );
            other.checkout_latest().await.unwrap();
            create_index(&mut other).await
        } else {
            result
        }
    };

    let latest = Dataset::open(dir.as_str()).await.unwrap();
    let names = load_all_indices(&latest)
        .await
        .unwrap()
        .iter()
        .map(|index| index.name.clone())
        .collect::<HashSet<_>>();
    match expected {
        RaceOutcome::Committed => {
            result.unwrap();
            assert!(names.contains(FRAG_REUSE_INDEX_NAME) && names.contains("i_idx"));
            assert_moves_keep_row_identity(&before, &latest).await;
            for use_scalar_index in [false, true] {
                assert_eq!(
                    values_matching(&latest, "i", "i > 3", use_scalar_index).await,
                    (4..30).collect::<Vec<_>>()
                );
            }
            assert_index_is_used(&latest, "i > 3", "i_idx").await;
        }
        RaceOutcome::Rejected | RaceOutcome::ConflictThenRejected => {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(error.to_string().contains("i_idx"), "{error}");
            assert_ne!(
                names.contains(FRAG_REUSE_INDEX_NAME),
                names.contains("i_idx"),
                "{names:?}"
            );
            if names.contains(FRAG_REUSE_INDEX_NAME) {
                assert_moves_keep_row_identity(&before, &latest).await;
            } else {
                assert_eq!(rows_by_addr(&latest).await, before);
            }
        }
    }
}
/// A deferred commit over a plan that groups covered and uncovered fragments
/// must refuse to cross a physical index boundary even when this build cannot
/// open the covering segment.
///
/// The commit guard reloads live non-system segment coverage before
/// publication; it must see every physical boundary, not only the ones in the
/// usable-index view. Filtering out a segment this build cannot open would
/// let the deferred rewrite delete old covered fragment ids during bitmap
/// repair without adding the merged fragment, stranding a capable reader on a
/// flat scan of that range.
#[tokio::test]
async fn test_deferred_compaction_rejects_crossing_an_unsupported_index_boundary() {
    const DIM: u32 = 16;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(64))
        .await
        .unwrap();

    // A single physical vector segment covering only the first two fragments,
    // so a plan rewriting all four fragments together partially crosses it.
    let centroids = FixedSizeListArray::try_new_from_values(
        Float32Array::from_iter_values(std::iter::repeat_n(0.0, DIM as usize)),
        DIM as i32,
    )
    .unwrap();
    let params = VectorIndexParams::with_ivf_flat_params(
        DistanceType::L2,
        IvfBuildParams::try_with_centroids(1, Arc::new(centroids)).unwrap(),
    );
    let fragment_ids = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    let segment = CreateIndexBuilder::new(&mut dataset, &["vec"], IndexType::Vector, &params)
        .name("vec_idx".to_string())
        .fragments(fragment_ids[..2].to_vec())
        .execute_uncommitted()
        .await
        .unwrap();
    dataset
        .commit_existing_index_segments("vec_idx", "vec", vec![segment])
        .await
        .unwrap();

    // Raise the segment's version so this build can no longer open it, keeping
    // its physical coverage over the first two fragments intact.
    let committed = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .filter(|index| index.name == "vec_idx")
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(committed.len(), 1);
    let mut hidden = committed.clone();
    hidden[0].index_version = IndexType::max_vector_version() as i32 + 1;
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: hidden.clone(),
            removed_indices: committed,
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
    assert!(
        dataset
            .load_indices_by_name("vec_idx")
            .await
            .unwrap()
            .is_empty(),
        "the segment must be hidden from the usable-index view"
    );

    // A public/custom plan can group the covered fragments (0,1) with the
    // uncovered ones (2,3) into one rewrite task - something the default
    // planner never does, because it keeps covered and uncovered fragments in
    // separate candidate bins. Execute that plan and commit it deferred, so
    // publication runs the guard without any remapper.
    let deferred_options = CompactionOptions {
        defer_index_remap: true,
        ..Default::default()
    };
    let task = CompactionTask {
        task: TaskData {
            fragments: dataset.fragments().as_ref().clone(),
            output_clustering_id: None,
            output_metadata: HashMap::new(),
        },
        read_version: dataset.manifest.version,
        options: deferred_options.clone(),
    };
    let rewrite_result = task.execute(&dataset).await.unwrap();
    let completed_tasks = vec![rewrite_result];
    assert_eq!(completed_tasks[0].original_fragments.len(), 4);

    let version_before_commit = dataset.manifest.version;
    let error = commit_compaction(
        &mut dataset,
        completed_tasks,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &deferred_options,
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("physical coverage boundary"));
    assert_eq!(dataset.manifest.version, version_before_commit);
}
