// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the lineage contract area.

use super::*;
use crate::dataset::optimize::execution::{
    PlannedRowLineage, append_row_lineage_columns, can_use_binary_copy, place_planned_row_lineage,
    place_row_lineage_in_files,
};
use crate::dataset::optimize::planning::pins_dead_user_columns_for_lineage;
use crate::dataset::rowids::{RowLineage, RowLineagePlan, RowLineageSpill, plan_row_lineage_spill};

#[tokio::test]
async fn test_compaction_balances_non_divisible_stable_task() {
    let test_dir = TempStrDir::default();
    let data = sample_data().slice(0, 121);
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let mut dataset = Dataset::write(
        reader,
        &test_dir,
        Some(WriteParams {
            max_rows_per_file: 121,
            data_storage_version: Some(LanceFileVersion::Stable),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.delete("a < 20").await.unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 10,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks.len(), 1);
    assert_eq!(plan.tasks[0].fragments.len(), 1);

    let metrics = compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 1);
    assert_eq!(metrics.fragments_added, 10);
    assert_eq!(
        dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.metadata.physical_rows.unwrap())
            .collect::<Vec<_>>(),
        [vec![11], vec![10; 9]].concat()
    );

    let second_metrics = compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(second_metrics, CompactionMetrics::default());
}
/// A sequence type over the budget in only some output fragments sends the
/// task down the per-fragment path: once the lineage is cut at the written
/// sizes, a fragment whose row ids fit keeps them inline, and only the ones
/// over the budget get a lineage file of their own.
#[tokio::test]
async fn per_fragment_plan_spills_only_the_fragments_over_budget() {
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset_with_params(
            FragmentCount::from(1),
            FragmentRowCount::from(10),
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset
        .update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "100"),
        ])
        .await
        .unwrap();
    // A range encodes to a few bytes at any length. The scattered ids, a
    // permutation of 500..1000 with no runs, take about two bytes a row,
    // so any 250 of them are well over the 100-byte budget.
    let scattered = (0..500_u64)
        .map(|i| 500 + (i * 7919) % 500)
        .collect::<Vec<_>>();
    let row_ids = [
        RowIdSequence::from(0..500),
        RowIdSequence::from(scattered.as_slice()),
    ];
    let lineages = row_ids
        .into_iter()
        .map(|row_ids| {
            let created_at = RowDatasetVersionSequence::from_uniform_row_count(row_ids.len(), 1);
            RowLineage {
                row_ids,
                last_updated_at: created_at.clone(),
                created_at,
            }
        })
        .collect::<Vec<_>>();
    let plan = plan_row_lineage_spill(100, &lineages);
    assert_eq!(plan, RowLineagePlan::PerFragment);
    // Written as [250, 500, 250] rather than the planned [500, 500], so
    // the middle fragment joins the tail of the range to the head of the
    // scattered ids.
    let mut fragments = [250, 500, 250]
        .into_iter()
        .enumerate()
        .map(|(id, rows)| {
            Fragment::new(id as u64)
                .with_file(
                    format!("{id}.lance"),
                    vec![0],
                    vec![0],
                    ConcreteFileVersion::V2_0,
                    None,
                )
                .with_physical_rows(rows)
        })
        .collect::<Vec<_>>();

    place_planned_row_lineage(
        &dataset,
        &mut fragments,
        PlannedRowLineage {
            lineages,
            planned_rows: vec![500, 500],
            plan,
        },
    )
    .await
    .unwrap();

    assert!(
        matches!(fragments[0].row_id_meta, Some(RowIdMeta::Inline(_))),
        "a fragment whose row ids fit must keep them inline, got {:?}",
        fragments[0].row_id_meta
    );
    assert_eq!(fragments[0].files.len(), 1);
    for fragment in &fragments[1..] {
        assert!(
            matches!(fragment.row_id_meta, Some(RowIdMeta::Column)),
            "fragment {}'s row ids are over the budget, got {:?}",
            fragment.id,
            fragment.row_id_meta
        );
        assert_eq!(fragment.files.len(), 2);
        assert_eq!(fragment.files[1].fields.as_ref(), [ROW_ID_FIELD_ID]);
    }
    let mut placed_row_ids = Vec::with_capacity(1000);
    for fragment in &fragments {
        let row_ids = match &fragment.row_id_meta {
            Some(RowIdMeta::Inline(bytes)) => read_row_ids(bytes).unwrap(),
            _ => read_spilled_row_ids(&dataset, fragment).await.unwrap(),
        };
        placed_row_ids.extend(row_ids.iter());
    }
    assert_eq!(
        placed_row_ids,
        (0..500)
            .chain(scattered.iter().copied())
            .collect::<Vec<_>>()
    );
}
/// The writer sizes a column's pages by the memory its arrays hold, so a
/// lineage array sharing one buffer with the whole task would make it cut
/// a tiny page for every batch.
#[tokio::test]
async fn append_row_lineage_columns_gives_each_batch_its_own_buffer() {
    let batch = arrow_array::record_batch!(("i", Int32, [0, 1, 2, 3])).unwrap();
    let schema = batch.schema();
    let batches = vec![batch; 3]
        .into_iter()
        .map(Ok::<_, datafusion::error::DataFusionError>);
    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::iter(batches),
    ));
    let spill = RowLineageSpill {
        row_ids: true,
        ..Default::default()
    };
    let fields = spill.schema_fields().unwrap();

    let output = append_row_lineage_columns(input, &fields, vec![(0..12).collect()])
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(output.len(), 3);
    for (index, batch) in output.iter().enumerate() {
        let start = index as u64 * 4;
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
        assert_eq!(
            row_ids.values().to_vec(),
            (start..start + 4).collect::<Vec<_>>()
        );
        assert!(
            row_ids.get_buffer_memory_size() <= batch.num_rows() * 8,
            "batch {index}'s row ids hold {} bytes for {} rows",
            row_ids.get_buffer_memory_size(),
            batch.num_rows()
        );
    }
}
#[rstest]
#[case::without_index(&[], false)]
#[case::zonemap(&[BuiltinIndexType::ZoneMap], false)]
#[case::bloom_filter(&[BuiltinIndexType::BloomFilter], false)]
#[case::btree(&[BuiltinIndexType::BTree], false)]
#[case::mixed(&[BuiltinIndexType::ZoneMap, BuiltinIndexType::BTree], false)]
#[case::zonemap_all_deleted(&[BuiltinIndexType::ZoneMap], true)]
#[case::bloom_filter_all_deleted(&[BuiltinIndexType::BloomFilter], true)]
#[tokio::test]
async fn test_row_addrs_only_used_with_remappable_index(
    #[case] index_types: &[BuiltinIndexType],
    #[case] is_fully_deleted: bool,
    #[values(IndexRemapMode::Direct, IndexRemapMode::Compact)] index_remap_mode: IndexRemapMode,
) {
    let mut dataset = lance_datagen::gen_batch()
        .col("a", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(128))
        .await
        .unwrap();

    for builtin in index_types {
        dataset
            .create_index(
                &["a"],
                IndexType::Scalar,
                Some(builtin.as_str().to_string()),
                &ScalarIndexParams::for_builtin(builtin.clone()),
                false,
            )
            .await
            .unwrap();
    }

    if is_fully_deleted {
        // Stored coverage may retain retired fragment IDs after DELETE.
        // Appended rows must stay visible before and after compaction.
        dataset.delete("true").await.unwrap();
        let mut scanner = dataset.scan();
        scanner.filter("a = 128").unwrap();
        assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 0);

        let data = arrow_array::record_batch!(("a", Int32, vec![-1; 256])).unwrap();
        let reader = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        dataset
            .append(
                reader,
                Some(WriteParams {
                    max_rows_per_file: 128,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }

    let options = CompactionOptions {
        target_rows_per_fragment: 256,
        index_remap_mode,
        ..Default::default()
    };
    let original_indices = dataset.load_indices().await.unwrap();
    let task = TaskData {
        fragments: dataset.fragments().as_ref().clone(),
        output_clustering_id: None,
        output_metadata: HashMap::new(),
    };
    let mut result = rewrite_files(Cow::Borrowed(&dataset), task, &options)
        .await
        .unwrap();
    assert_eq!(result.row_addrs.is_some(), !index_types.is_empty());

    if let Some(row_addrs_bytes) = &result.row_addrs {
        let row_addrs =
            RoaringTreemap::deserialize_from(&mut Cursor::new(row_addrs_bytes)).unwrap();
        assert_eq!(row_addrs.len(), 256);
        // The captured addresses are contiguous per-fragment ranges, so the
        // persisted blob must be run-optimized: O(fragments) bytes, not
        // O(rows). Without run containers this serializes at ~2 bytes per
        // address, so under one byte per address proves the run form was written.
        assert!(
            row_addrs_bytes.len() < row_addrs.len() as usize,
            "serialized row addrs ({} bytes for {} addresses) should be \
                 run-optimized before persisting",
            row_addrs_bytes.len(),
            row_addrs.len()
        );
    }
    if !index_types.contains(&BuiltinIndexType::BTree) {
        // Invalid bytes prove commit skips decoding and building the map
        // when no index on the table supports remapping.
        result.row_addrs = Some(b"not a roaring treemap".to_vec());
    }
    commit_compaction(
        &mut dataset,
        vec![result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 1);
    let current_indices = dataset.load_indices().await.unwrap();
    for original in original_indices.iter() {
        let current = current_indices
            .iter()
            .find(|index| index.name == original.name)
            .unwrap();
        if original.name == BuiltinIndexType::BTree.as_str() {
            assert_ne!(current.uuid, original.uuid);
        } else {
            assert_eq!(current.uuid, original.uuid);
            assert_eq!(current.fragment_bitmap, original.fragment_bitmap);
            assert!(
                current
                    .effective_fragment_bitmap(dataset.fragment_bitmap.as_ref())
                    .unwrap()
                    .is_empty()
            );
        }
    }
    let mut scanner = dataset.scan();
    scanner.filter("a = 128").unwrap();
    assert_eq!(
        scanner.try_into_batch().await.unwrap().num_rows(),
        usize::from(!is_fully_deleted)
    );

    if is_fully_deleted {
        let mut scanner = dataset.scan();
        scanner.filter("a = -1").unwrap();
        assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 256);

        dataset.optimize_indices(&Default::default()).await.unwrap();
        let mut scanner = dataset.scan();
        scanner.filter("a = -1").unwrap();
        assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 256);
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices[0].fragment_bitmap.as_ref(),
            Some(dataset.fragment_bitmap.as_ref())
        );
    }
}
/// Eager (non-deferred) compaction on a stable-row-id dataset captures row
/// addresses only when something will consume them: an index this build
/// can remap in place (see `commit_compaction`'s `needs_remapping`).
/// A dataset with no index, only row-id-domain ones (whose ids survive
/// the rewrite on their own), or only an address-domain index whose type
/// can never remap (ZoneMap, BloomFilter, FM) must not pay for a capture
/// nothing will use -- on a large compaction, that capture is an
/// old-to-new move for every rewritten row.
#[rstest::rstest]
#[case::no_index(IndexPresence::None, false)]
#[case::remap_capable_index(IndexPresence::RemapCapable, true)]
#[case::address_domain_only_index(IndexPresence::AddressDomainOnly, false)]
#[tokio::test]
async fn test_stable_row_ids_eager_compaction_captures_row_addrs_only_when_needed(
    #[case] index_presence: IndexPresence,
    #[case] expect_capture: bool,
) {
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 9_000))], data.schema());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 3_000,
            enable_stable_row_ids: true,
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    match index_presence {
        IndexPresence::None => {}
        // Defaults to BTree, which is address-domain under stable row ids.
        IndexPresence::RemapCapable => create_scalar_index(&mut dataset, "a", false).await,
        IndexPresence::AddressDomainOnly => {
            dataset
                .create_index(
                    &["a"],
                    IndexType::ZoneMap,
                    Some("a_idx".into()),
                    &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
                    false,
                )
                .await
                .unwrap();
        }
    }

    let options = CompactionOptions {
        target_rows_per_fragment: 9_000,
        defer_index_remap: false,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 1);

    let result = rewrite_files(Cow::Borrowed(&dataset), plan.tasks()[0].clone(), &options)
        .await
        .unwrap();
    assert_eq!(
        result.row_addrs.is_some(),
        expect_capture,
        "eager stable-row-id compaction must capture row addresses exactly when an \
             eager-remap-capable index needs them to remap"
    );
}
/// `DefaultCompactionPlanner::plan` computes `has_eager_remappable_index`
/// once and stashes it on `CompactionPlan::options`; `rewrite_files` must
/// trust that cached value instead of reloading index metadata itself,
/// and must fall back to a live reload only when a caller hands it
/// options that were never produced by the planner.
#[tokio::test]
async fn test_rewrite_files_trusts_planners_cached_eager_remap_capability() {
    let data = sample_data();
    let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 9_000))], data.schema());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 3_000,
            enable_stable_row_ids: true,
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let options = CompactionOptions {
        target_rows_per_fragment: 9_000,
        defer_index_remap: false,
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 1);
    assert_eq!(
        plan.options().has_eager_remappable_index,
        Some(false),
        "no index existed at plan time, so the planner should have cached a negative answer"
    );

    // Diverge from the planned state: a remap-capable index now exists,
    // but the plan's cached answer still says it doesn't.
    create_scalar_index(&mut dataset, "a", false).await;

    let stale_cached_result = rewrite_files(
        Cow::Borrowed(&dataset),
        plan.tasks()[0].clone(),
        plan.options(),
    )
    .await
    .unwrap();
    assert!(
        stale_cached_result.row_addrs.is_none(),
        "rewrite_files must trust the plan's cached capability instead of reloading, even \
             though reloading now would (incorrectly, for this plan) say an index needs it"
    );

    // Options a caller builds directly, bypassing the planner, carry no
    // cached answer and so fall back to a live reload that does see the
    // new index.
    assert_eq!(options.has_eager_remappable_index, None);
    let live_reload_result =
        rewrite_files(Cow::Borrowed(&dataset), plan.tasks()[0].clone(), &options)
            .await
            .unwrap();
    assert!(
        live_reload_result.row_addrs.is_some(),
        "without a cached answer, rewrite_files must fall back to loading index metadata live"
    );
}
/// Deleting in fragment 0 makes stable row ids and addresses diverge there, so
/// confusing the two misplaces rows.
#[tokio::test]
async fn test_defer_index_remap_records_moves_with_stable_row_ids() {
    let mut dataset = stable_row_id_dataset("memory://", 5).await;
    // Fragment 3 is emptied, so it leaves the manifest before the compaction.
    dataset
        .delete("i IN (1, 2, 5, 13) OR (i >= 30 AND i < 40)")
        .await
        .unwrap();
    let before = rows_by_addr(&dataset).await;
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        defer_index_remap: true,
        excluded_fragment_ids: vec![4],
        ..Default::default()
    };
    let metrics = compact_files(&mut dataset, options.clone(), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 3);

    let frag_reuse = dataset.frag_reuse_index().await.unwrap().unwrap();
    assert_eq!(frag_reuse.details.versions.len(), 1);
    for deleted in [1, 2, 5, 13] {
        assert_eq!(
            frag_reuse.row_addr_remap().get(written_addr(deleted)),
            Some(None),
            "row {deleted}"
        );
    }
    for not_rewritten in [30, 39, 40, 49] {
        assert_eq!(
            frag_reuse.row_addr_remap().get(written_addr(not_rewritten)),
            None,
            "row {not_rewritten}"
        );
    }
    assert_moves_keep_row_identity(&before, &dataset).await;

    let version = dataset.version().version;
    compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(dataset.version().version, version);
}
#[rstest]
#[case::binary_copy(CompactionMode::ForceBinaryCopy, false)]
#[case::deletions_fall_back_to_a_scan(CompactionMode::TryBinaryCopy, true)]
#[tokio::test]
async fn test_recorded_moves_keep_row_identity(
    #[case] compaction_mode: CompactionMode,
    #[case] has_deletions: bool,
) {
    let mut dataset = stable_row_id_dataset("memory://", 3).await;
    if has_deletions {
        dataset.delete("i IN (0, 4, 15)").await.unwrap();
    }
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        defer_index_remap: true,
        compaction_mode: Some(compaction_mode),
        ..Default::default()
    };
    assert_eq!(
        can_use_binary_copy(&dataset, &options, &dataset.manifest.fragments).await,
        !has_deletions
    );
    let before = rows_by_addr(&dataset).await;
    compact_files(&mut dataset, options, None).await.unwrap();
    assert_moves_keep_row_identity(&before, &dataset).await;
}
/// The second compaction rechunks row ids from where the first one spilled them.
#[rstest]
#[case::in_data_file(CompactionMode::TryBinaryCopy, 0)]
#[case::in_lineage_file(CompactionMode::ForceBinaryCopy, 1)]
#[tokio::test]
async fn test_recorded_moves_keep_row_identity_with_spilled_lineage(
    #[case] first_compaction_mode: CompactionMode,
    #[case] lineage_carrier: usize,
) {
    let mut dataset = stable_row_id_dataset("memory://", 3).await;
    dataset
        .update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
        ])
        .await
        .unwrap();
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        defer_index_remap: true,
        compaction_mode: Some(CompactionMode::TryBinaryCopy),
        ..Default::default()
    };
    // Binary copy refuses deletion files.
    if first_compaction_mode != CompactionMode::ForceBinaryCopy {
        dataset.delete("i IN (0, 4, 15)").await.unwrap();
    }
    let unrecorded_version = dataset.version().version;
    let before = rows_by_addr(&dataset).await;
    let first_options = CompactionOptions {
        compaction_mode: Some(first_compaction_mode),
        ..options.clone()
    };
    compact_files(&mut dataset, first_options, None)
        .await
        .unwrap();
    let fragment = &dataset.manifest.fragments[0];
    assert!(matches!(fragment.row_id_meta, Some(RowIdMeta::Column)));
    assert_eq!(fragment.files.len(), lineage_carrier + 1);
    assert_eq!(
        fragment.row_lineage_file(ROW_ID_FIELD_ID).unwrap(),
        Some(&fragment.files[lineage_carrier])
    );
    assert_eq!(
        flag_words(&dataset, FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS),
        (true, true)
    );
    assert_eq!(flag_words(&dataset, FLAG_SPILLED_ROW_LINEAGE), (true, true));
    assert_moves_keep_row_identity(&before, &dataset).await;

    let appended = arrow_array::record_batch!(("i", Int32, [100, 101])).unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();
    dataset.delete("i = 20").await.unwrap();
    let before = rows_by_addr(&dataset).await;
    compact_files(&mut dataset, options, None).await.unwrap();
    let frag_reuse = dataset.frag_reuse_index().await.unwrap().unwrap();
    assert_eq!(frag_reuse.details.versions.len(), 2);
    assert_moves_keep_row_identity(&before, &dataset).await;
    let recorded_version = dataset.version().version;

    let mut restored = dataset.checkout_version(unrecorded_version).await.unwrap();
    restored.restore().await.unwrap();
    assert_eq!(
        flag_words(&restored, FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS),
        (false, false)
    );
    let mut restored = restored.checkout_version(recorded_version).await.unwrap();
    restored.restore().await.unwrap();
    assert_eq!(
        flag_words(&restored, FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS),
        (true, true)
    );
    assert_moves_keep_row_identity(&before, &restored).await;

    let error = restored
        .drop_index(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert_eq!(
        flag_words(&restored, FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS),
        (true, true)
    );
    assert_eq!(
        flag_words(&restored, FLAG_SPILLED_ROW_LINEAGE),
        (true, true)
    );
}
/// Tasks that recorded their moves publish them however the commit is
/// configured, also onto a history the table already holds; tasks that did
/// not record leave that history as it is.
#[rstest]
#[case::options_at_commit(true, true)]
#[case::options_omitted_at_commit(true, false)]
#[case::eager(false, false)]
#[tokio::test]
async fn test_distributed_compaction_with_stable_row_ids(
    #[case] defer_index_remap: bool,
    #[case] commit_with_options: bool,
    #[values(false, true)] has_history: bool,
) {
    let dir = TempStrDir::default();
    let mut dataset = stable_row_id_dataset(dir.as_str(), if has_history { 7 } else { 5 }).await;
    dataset.delete("i IN (11, 12, 45)").await.unwrap();
    if has_history {
        let options = deferred_compaction_excluding(&[0, 1, 2, 3, 4]);
        compact_files(&mut dataset, options, None).await.unwrap();
    }
    let watermarks = frag_reuse_watermarks(&dataset).await;
    let options = CompactionOptions {
        target_rows_per_fragment: 1_000,
        defer_index_remap,
        excluded_fragment_ids: vec![2],
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks().len(), 2);

    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        let worker = Dataset::open(dir.as_str()).await.unwrap();
        results.push(task.execute(&worker).await.unwrap());
        // Recording reads fragment metadata, not a row-id index of the table.
        let row_id_index = RowIdIndexKey {
            version: worker.manifest.version,
            with_deletions: true,
            e_tag: worker.manifest_location.e_tag.as_deref(),
        };
        assert!(
            worker
                .index_cache
                .get_with_key(&row_id_index)
                .await
                .is_none()
        );
        get_row_id_index(&worker).await.unwrap();
        assert!(
            worker
                .index_cache
                .get_with_key(&row_id_index)
                .await
                .is_some()
        );
    }
    assert!(
        results
            .iter()
            .all(|result| result.row_addrs.is_some() == defer_index_remap)
    );
    // Replayed results may lack source row counts, which a recording commit fills in.
    if defer_index_remap {
        for fragment in results
            .iter_mut()
            .flat_map(|result| result.original_fragments.iter_mut())
        {
            fragment.physical_rows = None;
            if let Some(deletion_file) = fragment.deletion_file.as_mut() {
                deletion_file.num_deleted_rows = None;
            }
        }
    }

    let mut committer = Dataset::open(dir.as_str()).await.unwrap();
    let appended = arrow_array::record_batch!(("i", Int32, [1000, 1001])).unwrap();
    committer
        .append(
            RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();
    let before = rows_by_addr(&committer).await;
    let commit_options = if commit_with_options {
        options
    } else {
        CompactionOptions::default()
    };
    commit_compaction(
        &mut committer,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &commit_options,
    )
    .await
    .unwrap();
    if defer_index_remap {
        assert_eq!(
            frag_reuse_watermarks(&committer).await.len(),
            watermarks.len() + 1
        );
        assert_moves_keep_row_identity(&before, &committer).await;
    } else {
        assert_eq!(frag_reuse_watermarks(&committer).await, watermarks);
        assert_eq!(
            committer.frag_reuse_index().await.unwrap().is_some(),
            has_history
        );
        assert_rows_keep_their_ids(&before, &committer).await;
    }
}

/// A byte limit can close a compaction output file before its planned row
/// count, and the writer then spreads the remaining rows over files of
/// other sizes. The lineage placed inline has to follow the written row
/// counts, and a written total that differs from the planned one is an
/// error rather than lineage shifted onto the wrong rows.
#[rstest]
#[case::extra_file(
        &[10, 10],
        &[9, 6, 5],
        RowLineageSpill { row_ids: true, ..Default::default() },
        None
    )]
#[case::shifted(
        &[101, 100, 100],
        &[100, 101, 100],
        RowLineageSpill { row_ids: true, ..Default::default() },
        None
    )]
#[case::shifted_all_inline(
        &[101, 100, 100],
        &[100, 101, 100],
        RowLineageSpill::default(),
        None
    )]
#[case::mismatched_totals(
        &[10, 10],
        &[10, 9],
        RowLineageSpill { row_ids: true, ..Default::default() },
        Some("wrote 19 rows")
    )]
fn place_row_lineage_in_files_follows_written_sizes(
    #[case] planned_rows: &[u64],
    #[case] written_rows: &[usize],
    #[case] spill: RowLineageSpill,
    #[case] expected_error: Option<&str>,
) {
    let total_rows = planned_rows.iter().sum::<u64>();
    // Every row has a version of its own, so a sequence cut at the wrong
    // row cannot pass for the right one.
    let versions = (1..=total_rows).collect::<Vec<_>>();
    let mut lineages = Vec::with_capacity(planned_rows.len());
    let mut start = 0_u64;
    for rows in planned_rows {
        let end = start + rows;
        let created_at =
            RowDatasetVersionSequence::from_versions(&versions[start as usize..end as usize]);
        lineages.push(RowLineage {
            row_ids: RowIdSequence::from(start..end),
            last_updated_at: created_at.clone(),
            created_at,
        });
        start = end;
    }
    let columns = spill.take_columns(&mut lineages);
    if spill.row_ids {
        assert_eq!(columns, vec![(0..total_rows).collect::<Vec<_>>()]);
    } else {
        assert!(columns.is_empty(), "{columns:?}");
    }
    // Each written file carries the user column and the spilled columns.
    let fields = std::iter::once(0)
        .chain(spill.field_ids())
        .collect::<Vec<_>>();
    let column_indices = (0..fields.len() as i32).collect::<Vec<_>>();
    let mut fragments = written_rows
        .iter()
        .enumerate()
        .map(|(id, rows)| {
            Fragment::new(id as u64)
                .with_file(
                    format!("{id}.lance"),
                    fields.clone(),
                    column_indices.clone(),
                    ConcreteFileVersion::V2_0,
                    None,
                )
                .with_physical_rows(*rows)
        })
        .collect::<Vec<_>>();

    let placed = place_row_lineage_in_files(&mut fragments, lineages, planned_rows, spill);
    if let Some(expected) = expected_error {
        let error = placed.unwrap_err();
        let Error::Internal { message, .. } = &error else {
            panic!("expected an internal error, got {error}");
        };
        assert!(message.contains(expected), "{message}");
        return;
    }
    placed.unwrap();
    let mut placed_row_ids = Vec::with_capacity(versions.len());
    let mut placed_created_at = Vec::with_capacity(versions.len());
    let mut placed_last_updated_at = Vec::with_capacity(versions.len());
    for fragment in &fragments {
        let physical_rows = fragment.physical_rows.unwrap() as u64;
        match &fragment.row_id_meta {
            Some(RowIdMeta::Column) if spill.row_ids => {}
            Some(RowIdMeta::Inline(bytes)) if !spill.row_ids => {
                let row_ids = read_row_ids(bytes).unwrap();
                assert_eq!(row_ids.len(), physical_rows);
                placed_row_ids.extend(row_ids.iter());
            }
            other => panic!(
                "fragment {} has row id meta {other:?} under {spill:?}",
                fragment.id
            ),
        }
        for (meta, collected) in [
            (&fragment.created_at_version_meta, &mut placed_created_at),
            (
                &fragment.last_updated_at_version_meta,
                &mut placed_last_updated_at,
            ),
        ] {
            let sequence = meta.as_ref().unwrap().load_sequence().unwrap();
            assert_eq!(sequence.len(), physical_rows);
            collected.extend(sequence.versions());
        }
    }
    if !spill.row_ids {
        assert_eq!(placed_row_ids, (0..total_rows).collect::<Vec<_>>());
    }
    assert_eq!(placed_created_at, versions);
    assert_eq!(placed_last_updated_at, versions);
}

/// A fragment compacts on its own when one of its files is kept only for
/// the row lineage it carries next to user columns that are gone. Neither
/// layout compaction writes may qualify, or every compaction would plan
/// another. Field -2 is the tombstone and -3..=-5 are the lineage columns;
/// the schema holds fields 0 and 1, and 5 and 6 were dropped.
#[rstest]
#[case::in_file_carrier(vec![vec![0, 1, -3, -4, -5]], false)]
#[case::lineage_only_file(vec![vec![0, 1], vec![-3, -4, -5]], false)]
#[case::tombstoned_carrier(vec![vec![-2, -2, -3, -4, -5], vec![0, 1]], true)]
#[case::carrier_of_dropped_columns(vec![vec![5, 6, -3, -4, -5], vec![0, 1]], true)]
#[case::dead_file_without_lineage(vec![vec![-2, 6], vec![0, 1]], false)]
fn compaction_reclaims_lineage_carriers_of_dead_user_columns(
    #[case] files: Vec<Vec<i32>>,
    #[case] expected: bool,
) {
    let mut fragment = Fragment::new(0);
    for (index, fields) in files.into_iter().enumerate() {
        let column_indices = (0..fields.len() as i32).collect();
        fragment.add_file(
            format!("{index}.lance"),
            fields,
            column_indices,
            ConcreteFileVersion::V2_2,
            None,
        );
    }
    let schema_field_ids = HashSet::from([0, 1]);
    assert_eq!(
        pins_dead_user_columns_for_lineage(&fragment, &schema_field_ids),
        expected
    );
}
