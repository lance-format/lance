// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::Int32Array;
use arrow_schema::{DataType, Field, Schema};
use uuid::Uuid;

fn segment(aliases: BTreeMap<u32, u32>) -> IndexMetadata {
    IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: vec![0],
        covering_fields: vec![],
        name: "idx".into(),
        dataset_version: 1,
        fragment_bitmap: None,
        index_details: None,
        index_version: 1,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
        local_fragment_aliases: aliases,
    }
}

fn address(fragment: u32, offset: u32) -> u64 {
    u64::from(RowAddress::new_from_parts(fragment, offset))
}

#[test]
fn test_alias_translation_is_single_hop_and_precedes_downstream_remapping() {
    let index = segment([(17, 42), (42, 43)].into());
    let downstream = LocalFragmentRemapper::compose(&segment([(42, 60)].into()), None);
    let remapper = LocalFragmentRemapper::compose(&index, downstream).unwrap();
    assert_eq!(remapper.remap_row_id(address(17, 5)), Some(address(60, 5)));
    assert_eq!(remapper.remap_row_id(address(42, 5)), Some(address(43, 5)));
    assert_eq!(remapper.remap_row_id(address(3, 1)), Some(address(3, 1)));
    assert!(alias_cache_namespace(&index).is_some());
    assert!(alias_cache_namespace(&segment(Default::default())).is_none());
}

#[test]
fn test_alias_translation_rewrites_an_address_column() {
    let index = segment([(7, 42)].into());
    let remapper = LocalFragmentRemapper::compose(&index, None).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new("address", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(UInt64Array::from(vec![address(7, 4)])),
        ],
    )
    .unwrap();
    let translated = remapper.remap_row_ids_record_batch(batch, 1).unwrap();
    assert_eq!(
        translated.column(1).as_primitive::<UInt64Type>().value(0),
        address(42, 4)
    );
}

mod integration {
    use super::*;
    use arrow_array::{RecordBatchIterator, StringArray};
    use futures::TryStreamExt;
    use lance_core::Error;
    use lance_index::IndexType;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::ScalarIndexParams;
    use lance_table::feature_flags::FLAG_LOCAL_FRAGMENT_ALIASES;
    use lance_table::format::Fragment;
    use roaring::RoaringBitmap;

    use crate::dataset::optimize::remapping::remap_column_index;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::scanner::ColumnOrdering;
    use crate::dataset::transaction::{Operation, RewriteGroup, Transaction};
    use crate::dataset::write::{CommitBuilder, WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::frag_reuse_reader::tests as reader_tests;
    use crate::index::loading::DatasetIndexInternalExt;

    fn test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, false),
            Field::new("s", DataType::Utf8, false),
        ]))
    }

    fn make_batch_with_prefix(values: impl IntoIterator<Item = i32>, prefix: &str) -> RecordBatch {
        let vals: Vec<i32> = values.into_iter().collect();
        let labels: Vec<String> = vals.iter().map(|v| format!("{prefix}-{v}")).collect();
        RecordBatch::try_new(
            test_schema(),
            vec![
                Arc::new(Int32Array::from(vals)),
                Arc::new(StringArray::from(labels)),
            ],
        )
        .unwrap()
    }

    fn make_batch(values: impl IntoIterator<Item = i32>) -> RecordBatch {
        make_batch_with_prefix(values, "row")
    }

    async fn append_batch(dataset: &mut Dataset, batch: RecordBatch) {
        let reader = RecordBatchIterator::new(vec![Ok(batch)], test_schema());
        dataset
            .append(
                reader,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }

    async fn append_values(dataset: &mut Dataset, values: impl IntoIterator<Item = i32>) {
        append_batch(dataset, make_batch(values)).await;
    }

    async fn query_rows(
        dataset: &Dataset,
        filter: Option<&str>,
        use_scalar_index: bool,
    ) -> Vec<(i32, String)> {
        let mut scan = dataset.scan();
        scan.project(&["i", "s"]).unwrap();
        scan.use_scalar_index(use_scalar_index);
        if let Some(expr) = filter {
            scan.filter(expr).unwrap();
        }
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut rows: Vec<(i32, String)> = batches
            .iter()
            .flat_map(|b| {
                let ints = b.column(0).as_primitive::<arrow_array::types::Int32Type>();
                let strs = b.column(1).as_string::<i32>();
                (0..b.num_rows()).map(|idx| (ints.value(idx), strs.value(idx).to_string()))
            })
            .collect();
        rows.sort();
        rows
    }

    async fn assert_indexed_matches_full_scan(dataset: &Dataset, filter: &str) {
        let mut plan_scan = dataset.scan();
        plan_scan.project(&["i", "s"]).unwrap();
        plan_scan.filter(filter).unwrap();
        let plan = plan_scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery") || plan.contains("MaterializeIndex"),
            "expected scalar index plan for `{filter}`, got:\n{plan}"
        );
        let indexed = query_rows(dataset, Some(filter), true).await;
        let full = query_rows(dataset, Some(filter), false).await;
        assert_eq!(indexed, full, "mismatch for filter `{filter}`");
    }

    fn read_index_dir_bytes(uri: &str, uuid: Uuid) -> BTreeMap<String, Vec<u8>> {
        let dir = std::path::Path::new(uri)
            .join("_indices")
            .join(uuid.to_string());
        let mut files = BTreeMap::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().into_owned();
            files.insert(name, std::fs::read(entry.path()).unwrap());
        }
        assert!(!files.is_empty(), "expected index files in {dir:?}");
        files
    }

    /// Builds a dataset whose live fragments are `{2, 3}` (values `0..5` and `5..10`)
    /// with a BTree index `"i_idx"` whose files physically store fragment ids `{0, 1}`
    /// and whose manifest metadata maps `local_fragment_aliases = {0 -> 2, 1 -> 3}`.
    async fn dataset_with_aliased_btree(uri: &str, enable_stable_row_ids: bool) -> Dataset {
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch(0..5))], test_schema()),
            uri,
            Some(WriteParams {
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        append_values(&mut dataset, 5..10).await;

        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                Some("i_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let original = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == "i_idx")
            .unwrap()
            .clone();
        assert_eq!(
            original.fragment_bitmap,
            Some(RoaringBitmap::from_iter([0u32, 1]))
        );

        // Append identical rows as fragments 2 and 3, then drop fragments 0 and 1.
        append_values(&mut dataset, 0..5).await;
        append_values(&mut dataset, 5..10).await;
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::Delete {
                        updated_fragments: vec![],
                        deleted_fragment_ids: vec![0, 1],
                        predicate: "false".to_string(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        // Re-register the existing BTree files (storing fragment ids 0 and 1)
        // against committed fragments 2 and 3 via `local_fragment_aliases`.
        let current = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == "i_idx")
            .cloned();
        let mut aliased = original;
        aliased.dataset_version = dataset.manifest.version;
        aliased.fragment_bitmap = Some(RoaringBitmap::from_iter([2u32, 3]));
        aliased.local_fragment_aliases = BTreeMap::from([(0, 2), (1, 3)]);
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![aliased],
                        removed_indices: current.into_iter().collect(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        dataset
    }

    async fn query_i_values(dataset: &Dataset, filter: Option<&str>) -> Vec<i32> {
        let mut scan = dataset.scan();
        scan.project(&["i"]).unwrap();
        if let Some(expr) = filter {
            scan.filter(expr).unwrap();
        }
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut values: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        values.sort_unstable();
        values
    }

    async fn ordered_i_values(dataset: &Dataset, limit: Option<i64>) -> Vec<i32> {
        let mut scan = dataset.scan();
        scan.project(&["i"]).unwrap();
        scan.order_by(Some(vec![ColumnOrdering::asc_nulls_last("i".to_string())]))
            .unwrap();
        if let Some(limit) = limit {
            scan.limit(Some(limit), None).unwrap();
        }
        let batches: Vec<RecordBatch> = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect()
    }

    async fn idx_meta(dataset: &Dataset, name: &str) -> IndexMetadata {
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == name)
            .unwrap()
            .clone()
    }

    async fn commit_stable_partition(
        dataset: Dataset,
        source_ids: &[u64],
        dest_base_id: u64,
    ) -> Dataset {
        let old_fragments: Vec<Fragment> = source_ids
            .iter()
            .map(|id| {
                dataset
                    .fragments()
                    .iter()
                    .find(|f| f.id == *id)
                    .unwrap()
                    .clone()
            })
            .collect();
        let (transition, destinations) =
            reader_tests::prepare_partition(&dataset, source_ids, dest_base_id).await;
        let read_version = dataset.manifest.version;
        let frag_reuse_index = Some(
            crate::index::frag_reuse::frag_reuse_entry_appending(&dataset, vec![transition])
                .await
                .unwrap(),
        );
        CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index,
                },
                None,
            ))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_aliased_btree_query_ordered_scan_flags_and_cache_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let dataset = dataset_with_aliased_btree(uri, false).await;

        assert_ne!(
            dataset.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );

        // Reopen from disk to confirm persistence and manifest decoding.
        let reopened = Dataset::open(uri).await.unwrap();
        let meta = idx_meta(&reopened, "i_idx").await;
        assert_eq!(
            meta.local_fragment_aliases,
            BTreeMap::from([(0, 2), (1, 3)])
        );

        assert_indexed_matches_full_scan(&reopened, "i = 3").await;
        assert_indexed_matches_full_scan(&reopened, "i >= 7").await;
        assert_indexed_matches_full_scan(&reopened, "i IN (1, 8)").await;
        assert_eq!(
            query_rows(&reopened, Some("i = 0"), true).await,
            vec![(0, "row-0".to_string())]
        );
        assert_eq!(
            ordered_i_values(&reopened, Some(6)).await,
            vec![0, 1, 2, 3, 4, 5]
        );

        // Cache isolation across valid alias maps sharing the same index UUID,
        // index files, and session cache: rebase the same segment onto fragments
        // {4, 5} which hold the same indexed keys (`i`) with distinct `s` payloads.
        let files_before_rebase = read_index_dir_bytes(uri, meta.uuid);
        let v1_version = reopened.manifest.version;
        let mut v2 = reopened.clone();
        append_batch(&mut v2, make_batch_with_prefix(0..5, "v2")).await;
        append_batch(&mut v2, make_batch_with_prefix(5..10, "v2")).await;
        v2.apply_commit(
            Transaction::new(
                v2.manifest.version,
                Operation::Delete {
                    updated_fragments: vec![],
                    deleted_fragment_ids: vec![2, 3],
                    predicate: "false".to_string(),
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
        let prev = idx_meta(&v2, "i_idx").await;
        let mut rebased = prev.clone();
        rebased.dataset_version = v2.manifest.version;
        rebased.fragment_bitmap = Some(RoaringBitmap::from_iter([4u32, 5]));
        rebased.local_fragment_aliases = BTreeMap::from([(0, 4), (1, 5)]);
        v2.apply_commit(
            Transaction::new(
                v2.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![rebased],
                    removed_indices: vec![prev],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

        // Both snapshots share the exact same index UUID and session cache, and
        // both must return semantically valid rows matching their respective full scans.
        assert_eq!(
            read_index_dir_bytes(uri, meta.uuid),
            files_before_rebase,
            "metadata-only rebase must not rewrite index files"
        );
        assert_eq!(
            query_rows(&v2, Some("i = 0"), true).await,
            vec![(0, "v2-0".to_string())]
        );
        assert_indexed_matches_full_scan(&v2, "i >= 7").await;
        let v1_checked_out = v2.checkout_version(v1_version).await.unwrap();
        assert_eq!(
            query_rows(&v1_checked_out, Some("i = 0"), true).await,
            vec![(0, "row-0".to_string())]
        );
        assert_indexed_matches_full_scan(&v1_checked_out, "i >= 7").await;
    }

    #[tokio::test]
    async fn test_aliased_btree_inline_compaction_clears_aliases_and_flag() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = dataset_with_aliased_btree(uri, false).await;

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        let meta = idx_meta(&dataset, "i_idx").await;
        assert!(meta.local_fragment_aliases.is_empty());
        assert_eq!(
            dataset.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            dataset.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );

        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(query_i_values(&reopened, Some("i = 2")).await, vec![2]);
        assert_eq!(
            query_i_values(&reopened, Some("i >= 6")).await,
            vec![6, 7, 8, 9]
        );
        assert_eq!(
            ordered_i_values(&reopened, None).await,
            (0..10).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_aliased_btree_v0_fri_compaction_and_remap() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = dataset_with_aliased_btree(uri, false).await;

        // Delete rows before compaction so row offsets shift in the compacted fragment.
        dataset.delete("i = 2 OR i = 7").await.unwrap();

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Aliases are retained while deferred compaction is active through V0 FRI.
        let reopened = Dataset::open(uri).await.unwrap();
        let meta = idx_meta(&reopened, "i_idx").await;
        assert_eq!(
            meta.local_fragment_aliases,
            BTreeMap::from([(0, 2), (1, 3)])
        );
        assert_indexed_matches_full_scan(&reopened, "i = 1").await;
        assert_indexed_matches_full_scan(&reopened, "i = 2").await;
        assert_indexed_matches_full_scan(&reopened, "i >= 6").await;
        assert_eq!(
            ordered_i_values(&reopened, None).await,
            vec![0, 1, 3, 4, 5, 6, 8, 9]
        );

        // calculate_included_frags must return the post-FRI committed fragment id, not {0, 1}.
        let scalar = reopened
            .open_scalar_index("i", &meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let included = scalar.calculate_included_frags().await.unwrap();
        let live_frags: RoaringBitmap = reopened.fragments().iter().map(|f| f.id as u32).collect();
        assert_eq!(included, live_frags);

        // Remapping materializes committed addresses and clears aliases + feature flag.
        remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
            .await
            .unwrap();
        let after = idx_meta(&dataset, "i_idx").await;
        assert!(after.local_fragment_aliases.is_empty());
        assert_eq!(
            dataset.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            dataset.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        let final_ds = Dataset::open(uri).await.unwrap();
        assert_indexed_matches_full_scan(&final_ds, "i = 4").await;
        assert_indexed_matches_full_scan(&final_ds, "i >= 5").await;
    }

    #[tokio::test]
    async fn test_aliased_btree_tagged_fri_compaction_and_remap() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = dataset_with_aliased_btree(uri, false).await;

        // Delete rows before compaction so row offsets shift under Tagged FRI.
        dataset.delete("i = 1 OR i = 6").await.unwrap();

        // Transition fragments {2, 3} -> {10, 11} via tagged FRI, then defer-compact them.
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::ReserveFragments { num_fragments: 20 },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let mut dataset = commit_stable_partition(dataset, &[2, 3], 10).await;
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Queries must compose local_fragment_aliases ahead of Tagged FRI (BatchRowIdRemapper).
        let reopened = Dataset::open(uri).await.unwrap();
        let meta = idx_meta(&reopened, "i_idx").await;
        assert_eq!(
            meta.local_fragment_aliases,
            BTreeMap::from([(0, 2), (1, 3)])
        );
        assert_indexed_matches_full_scan(&reopened, "i = 0").await;
        assert_indexed_matches_full_scan(&reopened, "i = 1").await;
        assert_indexed_matches_full_scan(&reopened, "i = 9").await;
        assert_indexed_matches_full_scan(&reopened, "i >= 4").await;
        assert_eq!(
            ordered_i_values(&reopened, None).await,
            vec![0, 2, 3, 4, 5, 7, 8, 9]
        );

        // calculate_included_frags must also translate through alias -> Tagged FRI.
        let scalar = reopened
            .open_scalar_index("i", &meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let included = scalar.calculate_included_frags().await.unwrap();
        let live_frags: RoaringBitmap = reopened.fragments().iter().map(|f| f.id as u32).collect();
        assert_eq!(included, live_frags);

        // Tagged remap must rewrite the index with committed addresses and clear aliases + flag.
        remap_column_index(&mut dataset, &["i"], Some("i_idx".into()))
            .await
            .unwrap();
        let after = idx_meta(&dataset, "i_idx").await;
        assert_ne!(after.uuid, meta.uuid);
        assert!(after.local_fragment_aliases.is_empty());
        assert_eq!(
            dataset.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            dataset.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        let final_ds = Dataset::open(uri).await.unwrap();
        assert_indexed_matches_full_scan(&final_ds, "i = 3").await;
        assert_eq!(
            ordered_i_values(&final_ds, None).await,
            vec![0, 2, 3, 4, 5, 7, 8, 9]
        );
    }

    #[tokio::test]
    async fn test_aliased_btree_optimize_indices_merges_and_clears_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = dataset_with_aliased_btree(uri, false).await;

        append_values(&mut dataset, 10..15).await;
        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();

        let meta = idx_meta(&dataset, "i_idx").await;
        assert!(meta.local_fragment_aliases.is_empty());
        assert_eq!(
            dataset.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            dataset.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(query_i_values(&reopened, Some("i = 2")).await, vec![2]);
        assert_eq!(
            query_i_values(&reopened, Some("i >= 12")).await,
            vec![12, 13, 14]
        );
        assert_eq!(
            ordered_i_values(&reopened, None).await,
            (0..15).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_aliased_btree_calculate_included_frags_and_bitmap_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = dataset_with_aliased_btree(uri, false).await;

        let meta = idx_meta(&dataset, "i_idx").await;
        let scalar = dataset
            .open_scalar_index("i", &meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            scalar.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter([2u32, 3])
        );

        // Force bitmap recovery by committing the aliased segment with `fragment_bitmap = None`
        // (which triggers `migrate_indices` during commit) and verify committed ids are recovered.
        let mut missing_bitmap = meta.clone();
        missing_bitmap.fragment_bitmap = None;
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![missing_bitmap],
                        removed_indices: vec![meta],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let recovered = idx_meta(&dataset, "i_idx").await;
        assert_eq!(
            recovered.fragment_bitmap,
            Some(RoaringBitmap::from_iter([2u32, 3])),
            "bitmap recovery must persist committed fragment ids, not provisional ids"
        );
    }

    #[tokio::test]
    async fn test_aliased_unsupported_segments_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch(0..5))], test_schema()),
            uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // 1. Logical-row-id Bitmap (index_version = 0 on stable_row_ids table) is rejected.
        dataset
            .create_index(
                &["s"],
                IndexType::Bitmap,
                Some("s_bitmap".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let bitmap = idx_meta(&dataset, "s_bitmap").await;
        let mut aliased_bitmap = bitmap.clone();
        aliased_bitmap.index_version = 0;
        aliased_bitmap.local_fragment_aliases = BTreeMap::from([(9, 0)]);
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![aliased_bitmap.clone()],
                        removed_indices: vec![bitmap],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let err = dataset
            .open_scalar_index("s", &aliased_bitmap.uuid, &NoOpMetricsCollector)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");

        // 2. Logical-row-id BTree (index_version = 0 on stable_row_ids table) is rejected.
        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                Some("i_btree".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let btree = idx_meta(&dataset, "i_btree").await;
        let mut logical_btree = btree.clone();
        logical_btree.index_version = 0;
        logical_btree.local_fragment_aliases = BTreeMap::from([(9, 0)]);
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![logical_btree.clone()],
                        removed_indices: vec![btree],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let err = dataset
            .open_scalar_index("i", &logical_btree.uuid, &NoOpMetricsCollector)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");

        // 3. Querying through a logical-row-id aliased scalar index fails clearly.
        let mut scan = dataset.scan();
        scan.filter("s = 'row-0'").unwrap();
        let scan_err = match scan.try_into_stream().await {
            Ok(stream) => stream.try_collect::<Vec<_>>().await.unwrap_err(),
            Err(e) => e,
        };
        assert!(
            matches!(scan_err, Error::NotSupported { .. })
                || scan_err.to_string().contains("logical row ids"),
            "expected NotSupported for logical-row-id aliased query, got: {scan_err:?}"
        );
    }

    #[tokio::test]
    async fn test_aliased_bitmap_and_zonemap_indexes_work() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch(0..5))], test_schema()),
            uri,
            None,
        )
        .await
        .unwrap();
        append_values(&mut dataset, 5..10).await;

        dataset
            .create_index(
                &["s"],
                IndexType::Bitmap,
                Some("s_bitmap".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["i"],
                IndexType::ZoneMap,
                Some("i_zonemap".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let orig_bitmap = idx_meta(&dataset, "s_bitmap").await;
        let orig_zonemap = idx_meta(&dataset, "i_zonemap").await;

        // Replace fragments {0, 1} with {2, 3} holding identical data, and alias both indexes.
        append_values(&mut dataset, 0..5).await;
        append_values(&mut dataset, 5..10).await;
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::Delete {
                        updated_fragments: vec![],
                        deleted_fragment_ids: vec![0, 1],
                        predicate: "false".to_string(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        let mut aliased_bitmap = orig_bitmap;
        aliased_bitmap.dataset_version = dataset.manifest.version;
        aliased_bitmap.fragment_bitmap = Some(RoaringBitmap::from_iter([2u32, 3]));
        aliased_bitmap.local_fragment_aliases = BTreeMap::from([(0, 2), (1, 3)]);

        let mut aliased_zonemap = orig_zonemap;
        aliased_zonemap.dataset_version = dataset.manifest.version;
        aliased_zonemap.fragment_bitmap = Some(RoaringBitmap::from_iter([2u32, 3]));
        aliased_zonemap.local_fragment_aliases = BTreeMap::from([(0, 2), (1, 3)]);

        let existing = dataset.load_indices().await.unwrap();
        let removed: Vec<IndexMetadata> = existing
            .iter()
            .filter(|idx| idx.name == "s_bitmap" || idx.name == "i_zonemap")
            .cloned()
            .collect();
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![aliased_bitmap, aliased_zonemap],
                        removed_indices: removed,
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        let reopened = Dataset::open(uri).await.unwrap();
        assert_indexed_matches_full_scan(&reopened, "s = 'row-2'").await;
        assert_indexed_matches_full_scan(&reopened, "s = 'row-8'").await;
        assert_indexed_matches_full_scan(&reopened, "i >= 7").await;

        // ZoneMap::calculate_included_frags must return committed fragment ids {2, 3}.
        let zm_meta = idx_meta(&reopened, "i_zonemap").await;
        let zm = reopened
            .open_scalar_index("i", &zm_meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            zm.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter([2u32, 3])
        );
    }

    async fn dataset_with_zonemap_and_bloomfilter(uri: &str, aliased: bool) -> Dataset {
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch(0..5))], test_schema()),
            uri,
            None,
        )
        .await
        .unwrap();
        append_values(&mut dataset, 5..10).await;

        dataset
            .create_index(
                &["i"],
                IndexType::ZoneMap,
                Some("i_zm".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["s"],
                IndexType::BloomFilter,
                Some("s_bf".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        if !aliased {
            return dataset;
        }

        let orig_zm = idx_meta(&dataset, "i_zm").await;
        let orig_bf = idx_meta(&dataset, "s_bf").await;

        append_values(&mut dataset, 0..5).await;
        append_values(&mut dataset, 5..10).await;
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::Delete {
                        updated_fragments: vec![],
                        deleted_fragment_ids: vec![0, 1],
                        predicate: "false".to_string(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        let mut aliased_zm = orig_zm;
        aliased_zm.dataset_version = dataset.manifest.version;
        aliased_zm.fragment_bitmap = Some(RoaringBitmap::from_iter([2u32, 3]));
        aliased_zm.local_fragment_aliases = BTreeMap::from([(0, 2), (1, 3)]);

        let mut aliased_bf = orig_bf;
        aliased_bf.dataset_version = dataset.manifest.version;
        aliased_bf.fragment_bitmap = Some(RoaringBitmap::from_iter([2u32, 3]));
        aliased_bf.local_fragment_aliases = BTreeMap::from([(0, 2), (1, 3)]);

        let existing = dataset.load_indices().await.unwrap();
        let removed: Vec<IndexMetadata> = existing
            .iter()
            .filter(|idx| idx.name == "i_zm" || idx.name == "s_bf")
            .cloned()
            .collect();
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![aliased_zm, aliased_bf],
                        removed_indices: removed,
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        dataset
    }

    #[tokio::test]
    async fn test_zonemap_and_bloomfilter_coverage_across_aliases_and_fri() {
        // 1. No aliases and no FRI.
        let dir1 = tempfile::tempdir().unwrap();
        let ds1 = dataset_with_zonemap_and_bloomfilter(dir1.path().to_str().unwrap(), false).await;
        for (col, name) in [("i", "i_zm"), ("s", "s_bf")] {
            let meta = idx_meta(&ds1, name).await;
            let idx = ds1
                .open_scalar_index(col, &meta.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            assert_eq!(
                idx.calculate_included_frags().await.unwrap(),
                RoaringBitmap::from_iter([0u32, 1])
            );
        }
        assert_indexed_matches_full_scan(&ds1, "i >= 7").await;
        assert_indexed_matches_full_scan(&ds1, "s = 'row-3'").await;

        // 2. Aliases without FRI.
        let dir2 = tempfile::tempdir().unwrap();
        let uri2 = dir2.path().to_str().unwrap();
        let mut ds2 = dataset_with_zonemap_and_bloomfilter(uri2, true).await;
        for (col, name) in [("i", "i_zm"), ("s", "s_bf")] {
            let meta = idx_meta(&ds2, name).await;
            let idx = ds2
                .open_scalar_index(col, &meta.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            assert_eq!(
                idx.calculate_included_frags().await.unwrap(),
                RoaringBitmap::from_iter([2u32, 3])
            );
        }
        assert_indexed_matches_full_scan(&ds2, "i >= 7").await;
        assert_indexed_matches_full_scan(&ds2, "s = 'row-8'").await;

        // 3. V0 FRI with aliases (must not panic on RowAddrSelection::Full).
        compact_files(
            &mut ds2,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let ds2_reopened = Dataset::open(uri2).await.unwrap();
        let live_v0: RoaringBitmap = ds2_reopened
            .fragments()
            .iter()
            .map(|f| f.id as u32)
            .collect();
        for (col, name) in [("i", "i_zm"), ("s", "s_bf")] {
            let meta = idx_meta(&ds2_reopened, name).await;
            let idx = ds2_reopened
                .open_scalar_index(col, &meta.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            assert_eq!(idx.calculate_included_frags().await.unwrap(), live_v0);
        }
        assert_indexed_matches_full_scan(&ds2_reopened, "i >= 7").await;
        assert_indexed_matches_full_scan(&ds2_reopened, "s = 'row-3'").await;

        // 4. Tagged FRI with aliases and row-offset shifts.
        let dir3 = tempfile::tempdir().unwrap();
        let uri3 = dir3.path().to_str().unwrap();
        let mut ds3 = dataset_with_zonemap_and_bloomfilter(uri3, true).await;
        ds3.delete("i = 2 OR i = 7").await.unwrap();
        ds3.apply_commit(
            Transaction::new(
                ds3.manifest.version,
                Operation::ReserveFragments { num_fragments: 20 },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
        let mut ds3 = commit_stable_partition(ds3, &[2, 3], 10).await;
        compact_files(
            &mut ds3,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let ds3_reopened = Dataset::open(uri3).await.unwrap();
        let live_tagged: RoaringBitmap = ds3_reopened
            .fragments()
            .iter()
            .map(|f| f.id as u32)
            .collect();
        for (col, name) in [("i", "i_zm"), ("s", "s_bf")] {
            let meta = idx_meta(&ds3_reopened, name).await;
            let idx = ds3_reopened
                .open_scalar_index(col, &meta.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            assert_eq!(idx.calculate_included_frags().await.unwrap(), live_tagged);
        }
        assert_indexed_matches_full_scan(&ds3_reopened, "i >= 6").await;
        assert_indexed_matches_full_scan(&ds3_reopened, "i = 2").await;
        assert_indexed_matches_full_scan(&ds3_reopened, "s = 'row-8'").await;
        assert_indexed_matches_full_scan(&ds3_reopened, "s = 'row-7'").await;
    }

    #[tokio::test]
    async fn test_aliased_btree_mixed_identity_and_non_transitive_maps() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();

        // Write fragments 0 (0..5) and 1 (5..10), create BTree storing {0, 1}.
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch(0..5))], test_schema()),
            uri,
            None,
        )
        .await
        .unwrap();
        append_values(&mut dataset, 5..10).await;
        dataset
            .create_index(
                &["i"],
                IndexType::BTree,
                Some("i_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let original = idx_meta(&dataset, "i_idx").await;

        // Case 1: mixed identity (0 stays 0) + aliased (1 -> 2).
        append_values(&mut dataset, 5..10).await; // fragment 2
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::Delete {
                        updated_fragments: vec![],
                        deleted_fragment_ids: vec![1],
                        predicate: "false".to_string(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let current = idx_meta(&dataset, "i_idx").await;
        let mut mixed = original.clone();
        mixed.dataset_version = dataset.manifest.version;
        mixed.fragment_bitmap = Some(RoaringBitmap::from_iter([0u32, 2]));
        mixed.local_fragment_aliases = BTreeMap::from([(1, 2)]);
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![mixed],
                        removed_indices: vec![current],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        assert_indexed_matches_full_scan(&dataset, "i = 2").await;
        assert_indexed_matches_full_scan(&dataset, "i = 8").await;
        assert_indexed_matches_full_scan(&dataset, "i >= 3").await;

        // Case 2: non-transitive chain {0 -> 1, 1 -> 2} on a dataset whose live fragments
        // 1 and 2 have the same indexed keys (`i = 0..5`) with distinct `s` payloads.
        let dir2 = tempfile::tempdir().unwrap();
        let uri2 = dir2.path().to_str().unwrap();
        let mut ds2 = Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch_with_prefix(0..5, "f0"))], test_schema()),
            uri2,
            None,
        )
        .await
        .unwrap(); // frag 0: i=0..5, s="f0-*"
        append_batch(&mut ds2, make_batch_with_prefix(0..5, "f1")).await; // frag 1: i=0..5, s="f1-*"
        ds2.create_index(
            &["i"],
            IndexType::BTree,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
        let orig2 = idx_meta(&ds2, "i_idx").await;
        append_batch(&mut ds2, make_batch_with_prefix(0..5, "f2")).await; // frag 2: i=0..5, s="f2-*"
        ds2.apply_commit(
            Transaction::new(
                ds2.manifest.version,
                Operation::Delete {
                    updated_fragments: vec![],
                    deleted_fragment_ids: vec![0],
                    predicate: "false".to_string(),
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
        let cur2 = idx_meta(&ds2, "i_idx").await;
        let mut chain = orig2;
        chain.dataset_version = ds2.manifest.version;
        chain.fragment_bitmap = Some(RoaringBitmap::from_iter([1u32, 2]));
        chain.local_fragment_aliases = BTreeMap::from([(0, 1), (1, 2)]);
        ds2.apply_commit(
            Transaction::new(
                ds2.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![chain],
                    removed_indices: vec![cur2],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
        // If translation were wrongly transitive (0 -> 1 -> 2), rows from frag 1 ("f1-*")
        // would be missing and frag 2 ("f2-*") would appear twice!
        assert_indexed_matches_full_scan(&ds2, "i = 2").await;
        assert_eq!(
            query_rows(&ds2, Some("i = 2"), true).await,
            vec![(2, "f1-2".to_string()), (2, "f2-2".to_string())]
        );
    }

    #[tokio::test]
    async fn test_aliased_btree_clone_restore_and_commit_validation() {
        let dir = tempfile::tempdir().unwrap();
        let src_uri = dir.path().join("src");
        let clone_uri = dir.path().join("clone");
        let mut dataset = dataset_with_aliased_btree(src_uri.to_str().unwrap(), false).await;
        let aliased_version = dataset.manifest.version;

        // 1. Shallow clone preserves local_fragment_aliases and FLAG_LOCAL_FRAGMENT_ALIASES.
        let cloned = dataset
            .shallow_clone(clone_uri.to_str().unwrap(), aliased_version, None)
            .await
            .unwrap();
        assert_ne!(
            cloned.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_ne!(
            cloned.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            idx_meta(&cloned, "i_idx").await.local_fragment_aliases,
            BTreeMap::from([(0, 2), (1, 3)])
        );
        assert_indexed_matches_full_scan(&cloned, "i = 4").await;
        assert_indexed_matches_full_scan(&cloned, "i >= 7").await;

        // 2. Restore after clearing aliases (via inline compaction) brings back aliases + flag.
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            dataset.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        let mut restored = dataset.checkout_version(aliased_version).await.unwrap();
        restored.restore().await.unwrap();
        assert_ne!(
            restored.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_ne!(
            restored.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            idx_meta(&restored, "i_idx").await.local_fragment_aliases,
            BTreeMap::from([(0, 2), (1, 3)])
        );
        assert_indexed_matches_full_scan(&restored, "i = 1").await;

        // 3. Commit-time validation rejects duplicate destinations and reserved TOMBSTONE_FRAG.
        let current = idx_meta(&restored, "i_idx").await;
        let mut dup = current.clone();
        dup.local_fragment_aliases = BTreeMap::from([(0, 2), (1, 2)]);
        let err = restored
            .apply_commit(
                Transaction::new(
                    restored.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![dup],
                        removed_indices: vec![current.clone()],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("more than one"), "{err:?}");

        let mut tombstone = current.clone();
        tombstone.local_fragment_aliases = BTreeMap::from([(0, RowAddress::TOMBSTONE_FRAG)]);
        let err = restored
            .apply_commit(
                Transaction::new(
                    restored.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![tombstone],
                        removed_indices: vec![current],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("reserved"), "{err:?}");
    }

    async fn dataset_with_aliased_scalar_index(
        uri: &str,
        column: &str,
        index_type: IndexType,
        name: &str,
    ) -> Dataset {
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch(0..5))], test_schema()),
            uri,
            None,
        )
        .await
        .unwrap();
        append_values(&mut dataset, 5..10).await;

        dataset
            .create_index(
                &[column],
                index_type,
                Some(name.into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let original = idx_meta(&dataset, name).await;

        append_values(&mut dataset, 0..5).await;
        append_values(&mut dataset, 5..10).await;
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::Delete {
                        updated_fragments: vec![],
                        deleted_fragment_ids: vec![0, 1],
                        predicate: "false".to_string(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        let current = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == name)
            .cloned();
        let mut aliased = original;
        aliased.dataset_version = dataset.manifest.version;
        aliased.fragment_bitmap = Some(RoaringBitmap::from_iter([2u32, 3]));
        aliased.local_fragment_aliases = BTreeMap::from([(0, 2), (1, 3)]);
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![aliased],
                        removed_indices: current.into_iter().collect(),
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();

        dataset
    }

    #[tokio::test]
    async fn test_aliased_bitmap_non_transitive_compaction_does_not_double_translate() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch_with_prefix(0..5, "f0"))], test_schema()),
            uri,
            None,
        )
        .await
        .unwrap();
        append_batch(&mut ds, make_batch_with_prefix(0..5, "f1")).await;
        ds.create_index(
            &["i"],
            IndexType::Bitmap,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
        let orig = idx_meta(&ds, "i_idx").await;
        append_batch(&mut ds, make_batch_with_prefix(0..5, "f2")).await;
        ds.apply_commit(
            Transaction::new(
                ds.manifest.version,
                Operation::Delete {
                    updated_fragments: vec![],
                    deleted_fragment_ids: vec![0],
                    predicate: "false".to_string(),
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
        let cur = idx_meta(&ds, "i_idx").await;
        let mut chain = orig;
        chain.dataset_version = ds.manifest.version;
        chain.fragment_bitmap = Some(RoaringBitmap::from_iter([1u32, 2]));
        chain.local_fragment_aliases = BTreeMap::from([(0, 1), (1, 2)]);
        ds.apply_commit(
            Transaction::new(
                ds.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![chain],
                    removed_indices: vec![cur],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();

        assert_indexed_matches_full_scan(&ds, "i = 2").await;
        assert_eq!(
            query_rows(&ds, Some("i = 2"), true).await,
            vec![(2, "f1-2".to_string()), (2, "f2-2".to_string())]
        );

        compact_files(
            &mut ds,
            CompactionOptions {
                target_rows_per_fragment: 20,
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        assert!(
            idx_meta(&ds, "i_idx")
                .await
                .local_fragment_aliases
                .is_empty()
        );
        assert_indexed_matches_full_scan(&ds, "i = 2").await;
        assert_eq!(
            query_rows(&ds, Some("i = 2"), true).await,
            vec![(2, "f1-2".to_string()), (2, "f2-2".to_string())]
        );
    }

    #[tokio::test]
    async fn test_aliased_ngram_optimize_indices_preserves_postings() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut ds = dataset_with_aliased_scalar_index(uri, "s", IndexType::NGram, "s_ngram").await;

        assert_indexed_matches_full_scan(&ds, "s LIKE '%row-3%'").await;
        append_values(&mut ds, 10..15).await;
        ds.optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();

        assert!(
            idx_meta(&ds, "s_ngram")
                .await
                .local_fragment_aliases
                .is_empty()
        );
        assert_indexed_matches_full_scan(&ds, "s LIKE '%row-3%'").await;
        assert_indexed_matches_full_scan(&ds, "s LIKE '%row-12%'").await;
    }

    #[tokio::test]
    async fn test_aliased_zonemap_optimize_indices_rebuilds_committed_zones() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut ds = dataset_with_aliased_scalar_index(uri, "i", IndexType::ZoneMap, "i_zm").await;

        assert_indexed_matches_full_scan(&ds, "i = 3").await;
        append_values(&mut ds, 10..15).await;
        ds.optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();

        assert!(
            idx_meta(&ds, "i_zm")
                .await
                .local_fragment_aliases
                .is_empty()
        );
        assert_indexed_matches_full_scan(&ds, "i = 3").await;
        assert_indexed_matches_full_scan(&ds, "i = 12").await;
    }

    #[tokio::test]
    async fn test_aliased_bloomfilter_optimize_indices_rebuilds_committed_zones() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        let mut ds =
            dataset_with_aliased_scalar_index(uri, "s", IndexType::BloomFilter, "s_bf").await;

        assert_indexed_matches_full_scan(&ds, "s = 'row-3'").await;
        append_values(&mut ds, 10..15).await;
        ds.optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();

        assert!(
            idx_meta(&ds, "s_bf")
                .await
                .local_fragment_aliases
                .is_empty()
        );
        assert_indexed_matches_full_scan(&ds, "s = 'row-3'").await;
        assert_indexed_matches_full_scan(&ds, "s = 'row-12'").await;
    }
}
