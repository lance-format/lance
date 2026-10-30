// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
use crate::dataset::write::CommitBuilder;
use crate::index::frag_reuse_reader::tests as reader_tests;
use arrow_array::types::Int32Type;
use chrono::TimeDelta;
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::scalar::ScalarIndexParams;
use lance_table::format::Fragment;
use lance_table::format::IndexMetadata;
use lance_table::format::pb::fragment_reuse_index_details as pb_fri;
use lance_table::transaction::RewriteGroup;
use prost::Message;

impl MockDatasetFixture {
    /// The `_fri/<map_id>/` directories currently on disk.
    async fn list_fri_map_dirs(&self) -> Vec<String> {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let (os, path) =
            ObjectStore::from_uri_and_params(registry, &self.dataset_path, &self.os_params())
                .await
                .unwrap();
        let mut ids = std::collections::HashSet::new();
        let mut stream = os.read_dir_all(&path.clone().join("_fri"), None);
        loop {
            match stream.try_next().await {
                Ok(Some(meta)) => {
                    let relative = remove_prefix(&meta.location, &path);
                    if let Some(map_id) = relative.parts().nth(1) {
                        ids.insert(map_id.as_ref().to_string());
                    }
                }
                Ok(None) => break,
                Err(Error::NotFound { .. }) => break,
                Err(e) => panic!("{e}"),
            }
        }
        let mut ids: Vec<String> = ids.into_iter().collect();
        ids.sort();
        ids
    }
}

async fn reserve_fragments(dataset: &mut Dataset, num_fragments: u32) {
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::ReserveFragments { num_fragments },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
}

/// A persisted dataset (mock store, recorded mtimes) with one
/// committed stable-partition rewrite; returns the map id of its
/// row map.
async fn make_tagged(fixture: &MockDatasetFixture) -> (Dataset, String) {
    let data = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_reader_rows(
            lance_datagen::RowCount::from(4),
            lance_datagen::BatchCount::from(2),
        );
    Dataset::write(
        data,
        &fixture.dataset_path,
        Some(WriteParams {
            store_params: Some(fixture.os_params()),
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            mode: WriteMode::Create,
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut dataset = *fixture.open().await.unwrap();
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let (transition, destinations) = reader_tests::prepare(&dataset).await;
    let Some(pb_fri::transition::Mapping::StablePartition(mapping)) = &transition.mapping else {
        unreachable!()
    };
    let map_id = mapping.map_id.clone();
    let read_version = dataset.manifest.version;
    let frag_reuse_index = Some(
        crate::index::frag_reuse::frag_reuse_entry_appending(&dataset, vec![transition])
            .await
            .unwrap(),
    );
    let dataset = CommitBuilder::new(Arc::new(dataset))
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
        .unwrap();
    (dataset, map_id)
}

/// D1: a map referenced by a retained manifest survives cleanup, no
/// matter how old it is.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn referenced_map_is_kept() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let (_, map_id) = make_tagged(&fixture).await;
    MockClock::set_system_time(TimeDelta::try_days(10).unwrap().to_std().unwrap());
    fixture
        .run_cleanup(utc_now() - TimeDelta::try_seconds(1).unwrap())
        .await
        .unwrap();
    assert_eq!(fixture.list_fri_map_dirs().await, vec![map_id]);
}

#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn version_cleanup_keeps_maps_of_rescued_branch_roots() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let (mut dataset, map_id) = make_tagged(&fixture).await;
    let root_version = dataset.version().version;
    let mut branch = fixture
        .create_branch_and_load(&mut dataset, "child", (None, None))
        .await
        .unwrap();
    branch
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut branch).await.unwrap();
    assert!(
        branch
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_none()
    );
    branch
        .cleanup(expire_below(branch.version().version, true))
        .execute()
        .await
        .unwrap();
    fixture.overwrite_some_data().await.unwrap();
    let latest = fixture.open().await.unwrap().version().version;

    fixture
        .run_cleanup_with_policy(expire_below(latest, true))
        .await
        .unwrap();

    assert_eq!(fixture.list_fri_map_dirs().await, vec![map_id]);
    fixture
        .open()
        .await
        .unwrap()
        .checkout_version(root_version)
        .await
        .unwrap();
    assert_eq!(branch.count_rows(None).await.unwrap(), 8);
}

#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn parent_cleanup_keeps_a_row_map_named_only_by_a_branch_history() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let (mut dataset, map_id) = make_tagged(&fixture).await;
    let mut branch = fixture
        .create_branch_and_load(&mut dataset, "child", (None, None))
        .await
        .unwrap();
    // With its own data and index the branch rescues no parent
    // version, so only its tagged history names the parent's row map.
    branch.delete("true").await.unwrap();
    branch
        .append(
            lance_datagen::gen_batch()
                .col("i", lance_datagen::array::step::<Int32Type>())
                .into_reader_rows(
                    lance_datagen::RowCount::from(4),
                    lance_datagen::BatchCount::from(1),
                ),
            None,
        )
        .await
        .unwrap();
    branch
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let branch_entry = branch
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .unwrap();
    // Writers leave base_id unset on tagged index entries.
    assert_ne!(branch_entry.index_version, 0);
    assert_eq!(branch_entry.base_id, None);
    branch
        .cleanup(expire_below(branch.version().version, true))
        .execute()
        .await
        .unwrap();

    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    let latest = dataset.version().version;
    // The branch recorded the parent uri without the trailing slash.
    let respelled = fixture
        .load_dataset(&format!("{}/", fixture.dataset_path))
        .await
        .unwrap();
    cleanup_old_versions(&respelled, expire_below(latest, true))
        .await
        .unwrap();

    assert_eq!(fixture.list_fri_map_dirs().await, vec![map_id]);
}

/// D2: a map referenced only by expiring manifests goes away in the
/// same pass that removes them (the trim released it from the entry
/// first).
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn map_released_by_trim_is_collected_with_its_manifests() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let (mut dataset, map_id) = make_tagged(&fixture).await;
    assert_eq!(fixture.list_fri_map_dirs().await, vec![map_id]);

    // Drain the index onto the destinations, then trim: the entry is
    // deleted, so only old manifests reference the map.
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_none()
    );

    // Young files referenced by expiring manifests still wait for the
    // unverified retention window unless verified; the reference from
    // the removed manifest is that verification.
    MockClock::set_system_time(TimeDelta::try_days(10).unwrap().to_std().unwrap());
    fixture
        .run_cleanup(utc_now() - TimeDelta::try_seconds(1).unwrap())
        .await
        .unwrap();
    assert!(fixture.list_fri_map_dirs().await.is_empty());
}

/// D3 + D4: an unreferenced map (a failed commit's orphan) is kept
/// while young and collected once it ages past the unverified
/// retention threshold.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn orphan_map_ages_out() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let (dataset, committed_map_id) = make_tagged(&fixture).await;

    // A row map written at day 10 whose commit never lands.
    MockClock::set_system_time(TimeDelta::try_days(10).unwrap().to_std().unwrap());
    let (orphan_transition, _) = reader_tests::prepare(&dataset).await;
    let Some(pb_fri::transition::Mapping::StablePartition(mapping)) = &orphan_transition.mapping
    else {
        unreachable!()
    };
    let orphan_map_id = mapping.map_id.clone();
    let mut expected = vec![committed_map_id.clone(), orphan_map_id.clone()];
    expected.sort();
    assert_eq!(fixture.list_fri_map_dirs().await, expected);

    // D3: unreferenced but young, kept.
    fixture
        .run_cleanup(utc_now() + TimeDelta::try_seconds(1).unwrap())
        .await
        .unwrap();
    assert_eq!(fixture.list_fri_map_dirs().await, expected);

    // D4: unreferenced and old, collected; the referenced map stays.
    MockClock::set_system_time(TimeDelta::try_days(20).unwrap().to_std().unwrap());
    fixture
        .run_cleanup(utc_now() + TimeDelta::try_seconds(1).unwrap())
        .await
        .unwrap();
    assert_eq!(fixture.list_fri_map_dirs().await, vec![committed_map_id]);
}

/// D5: reference resolution decodes external entry content too. The
/// integration tests above cover inline entries; this drives the
/// resolver directly over an entry whose details live in an external
/// file.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn reference_resolution_reads_external_entries() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let (dataset, _) = make_tagged(&fixture).await;

    let map_id = Uuid::new_v4().to_string();
    let digest = |id: u64| pb_fri::FragmentDigest {
        id,
        physical_rows: 4,
        num_deleted_rows: 0,
    };
    let content = {
        let mut content = Vec::new();
        let transition = pb_fri::Transition {
            sources: vec![digest(100)],
            destinations: vec![digest(101)],
            mapping: Some(pb_fri::transition::Mapping::StablePartition(
                pb_fri::StablePartition {
                    map_id: map_id.clone(),
                    map_size_bytes: 1,
                    base_id: None,
                },
            )),
        };
        content.extend(reader_tests::field(2, &transition.encode_to_vec()));
        content
    };
    let uuid = Uuid::new_v4();
    let details_path = dataset
        .indices_dir()
        .join(uuid.to_string())
        .join("frag_reuse_details.binpb");
    let mut writer = dataset.object_store.create(&details_path).await.unwrap();
    use tokio::io::AsyncWriteExt;
    writer.write_all(&content).await.unwrap();
    writer.shutdown().await.unwrap();
    let entry = IndexMetadata {
        uuid,
        fields: vec![],
        covering_fields: vec![],
        name: FRAG_REUSE_INDEX_NAME.into(),
        dataset_version: dataset.manifest.version,
        fragment_bitmap: None,
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: reader_tests::field(
                2,
                &lance_table::format::pb::ExternalFile {
                    path: "frag_reuse_details.binpb".into(),
                    offset: 0,
                    size: content.len() as u64,
                }
                .encode_to_vec(),
            ),
        })),
        index_version: 1,
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };

    let task = CleanupTask::new(
        &dataset.cleanup(CleanupPolicy::default()),
        CleanupAction::Execute,
    )
    .unwrap();
    let mut inspection = CleanupInspection::default();
    inspection
        .frag_reuse_entries
        .insert(uuid, (entry.clone(), true));
    assert!(task.resolve_frag_reuse_map_ids(&mut inspection).await);
    assert!(
        inspection
            .referenced_files
            .frag_reuse_map_ids
            .contains(&map_id)
    );

    // An uninterpretable entry (a future index_version) disables
    // `_fri` collection for the whole run instead of guessing.
    let future_entry = IndexMetadata {
        index_version: 2,
        ..entry.clone()
    };
    let mut inspection = CleanupInspection::default();
    inspection
        .frag_reuse_entries
        .insert(uuid, (future_entry, true));
    assert!(!task.resolve_frag_reuse_map_ids(&mut inspection).await);

    // So does an unknown envelope-level record inside a supported
    // index_version: it may reference row maps this build cannot
    // see, so no map may be treated as unreferenced this run.
    let mut unknown_content = content.clone();
    unknown_content.extend(reader_tests::field(9, b"future envelope record"));
    let unknown_entry = IndexMetadata {
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: reader_tests::field(1, &unknown_content),
        })),
        ..entry
    };
    let mut inspection = CleanupInspection::default();
    inspection
        .frag_reuse_entries
        .insert(uuid, (unknown_entry, true));
    assert!(
        !task.resolve_frag_reuse_map_ids(&mut inspection).await,
        "an unknown envelope record must disable _fri collection"
    );
}

/// E: the full lifecycle, asserting every stage. A stable-partition
/// commit installs the tagged entry and its row map; an index rebuild
/// over the destinations simulates the drain and drops the old
/// segment; the trim then deletes the fully drained entry; a young
/// cleanup keeps the map (still referenced by retained manifests);
/// an aged cleanup removes those manifests and the map with them,
/// leaving a working table.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn end_to_end_lifecycle() {
    use crate::index::frag_reuse::decode_frag_reuse_ledger;
    use lance_table::system_index::frag_reuse::ledger::Mapping;

    let fixture = MockDatasetFixture::try_new().unwrap();
    let (mut dataset, map_id) = make_tagged(&fixture).await;

    // Stage 1: the tagged entry holds one stable-partition
    // transition referencing the on-disk row map, and the old index
    // segment still hangs off the sources.
    let entry = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
        .expect("the stable-partition commit installs the entry");
    assert_eq!(entry.index_version, 1);
    let ledger = decode_frag_reuse_ledger(&dataset, &entry).await.unwrap();
    assert_eq!(ledger.transitions().len(), 1);
    let Mapping::StablePartition(partition) = ledger.transitions()[0].mapping() else {
        panic!("expected a stable-partition transition");
    };
    assert_eq!(partition.map_id, map_id);
    assert_eq!(fixture.list_fri_map_dirs().await, vec![map_id.clone()]);
    let old_segment = dataset
        .load_index_by_name("i_idx")
        .await
        .unwrap()
        .expect("fixture index");

    // Stage 2: the trim retains the entry while the segment still
    // derives from the sources.
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some()
    );

    // Stage 3: drain -- rebuild the index over the destinations,
    // dropping the old segment.
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let rebuilt = dataset
        .load_index_by_name("i_idx")
        .await
        .unwrap()
        .expect("rebuilt index");
    assert_ne!(rebuilt.uuid, old_segment.uuid);

    // Stage 4: the trim now deletes the fully drained entry.
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_none()
    );

    // Stage 5: a young cleanup keeps the map -- the manifests that
    // reference it are still retained.
    fixture
        .run_cleanup(utc_now() - TimeDelta::try_days(7).unwrap())
        .await
        .unwrap();
    assert_eq!(fixture.list_fri_map_dirs().await, vec![map_id]);

    // Stage 6: once aged, the referencing manifests expire and the
    // map goes with them; the table keeps working.
    MockClock::set_system_time(TimeDelta::try_days(10).unwrap().to_std().unwrap());
    fixture
        .run_cleanup(utc_now() - TimeDelta::try_seconds(1).unwrap())
        .await
        .unwrap();
    assert!(fixture.list_fri_map_dirs().await.is_empty());
    let reopened = fixture.open().await.unwrap();
    assert_eq!(reopened.count_rows(None).await.unwrap(), 8);
    assert_eq!(
        reopened
            .count_rows(Some("i >= 4".to_string()))
            .await
            .unwrap(),
        4
    );
}

/// Commit a fresh `i_idx` delta segment built over the current
/// (translated) table state; `covered` optionally narrows the
/// committed bitmap to an under-claim.
async fn commit_delta_segment(dataset: &mut Dataset, covered: Option<&[u32]>) -> Uuid {
    let params = ScalarIndexParams::default();
    let mut delta =
        crate::index::CreateIndexBuilder::new(dataset, &["i"], IndexType::BTree, &params)
            .name("i_idx_delta".into())
            .execute_uncommitted()
            .await
            .unwrap();
    delta.name = "i_idx".into();
    if let Some(covered) = covered {
        delta.fragment_bitmap = Some(covered.iter().copied().collect());
    }
    let uuid = delta.uuid;
    dataset
        .apply_commit(
            Transaction::new(
                dataset.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![delta],
                    removed_indices: vec![],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    uuid
}

async fn stored_segments(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
    lance_table::io::manifest::read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap()
    .into_iter()
    .filter(|idx| idx.name == name)
    .collect()
}

/// Draining through `optimize_indices` merge on a tagged table. A
/// provenance-only segment still owning translated coverage is opened
/// through the translating loader, so the merged file holds live
/// addresses. The merged segment commits the UNION of the selected
/// segments' stored bitmaps (provenance, retired sources included):
/// the tagged reader treats a segment's bitmap as provenance, stops
/// translating at a live fragment, and derives coverage of a
/// destination whenever every contributing source is present, so the
/// merged segment keeps the translated destination coverage instead
/// of degrading to scan fallback, and the transition it still names
/// stays retained.
///
/// A provenance-only segment whose translated coverage is fully taken
/// over by siblings is excluded from the query listing by
/// direct-coverage-wins and cannot be selected for a merge; it is
/// dead weight that the prune step removes
/// (`delta_drain_stages_partial_then_full`).
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn merge_keeps_translated_coverage_on_tagged_tables() {
    use lance_index::optimize::OptimizeOptions;

    let fixture = MockDatasetFixture::try_new().unwrap();
    let (mut dataset, _map_id) = make_tagged(&fixture).await;
    // A partial delta: the old segment keeps exclusive translated
    // coverage of destination 11, so it stays openable.
    commit_delta_segment(&mut dataset, Some(&[10])).await;
    assert_eq!(stored_segments(&dataset, "i_idx").await.len(), 2);
    let all_rows = dataset.count_rows(None).await.unwrap();
    let scan_counts = counts_without_index(&dataset).await;

    dataset
        .optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();

    // One merged segment; its stored bitmap is provenance: the retired
    // sources of the old segment plus the delta's direct coverage.
    let merged = stored_segments(&dataset, "i_idx").await;
    assert_eq!(merged.len(), 1, "{merged:?}");
    let stored = merged[0].fragment_bitmap.clone().unwrap();
    assert!(
        stored.contains(0) && stored.contains(1) && stored.contains(10),
        "{stored:?}"
    );

    // The reader derives coverage of BOTH destinations from that
    // provenance, so nothing falls back to scan.
    let derived = derived_coverage(&dataset, "i_idx").await;
    assert_eq!(
        derived,
        dataset.fragment_bitmap.as_ref().clone(),
        "the merged segment must answer for every live fragment"
    );
    assert_index_used(&dataset).await;
    assert_eq!(dataset.count_rows(None).await.unwrap(), all_rows);
    assert_eq!(counts_with_index(&dataset).await, scan_counts);

    // Trim keeps the transition: the merged segment's provenance still
    // names its sources and destination 11 has no direct coverage.
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some(),
        "the merged segment still depends on the transition"
    );
    assert_eq!(stored_segments(&dataset, "i_idx").await.len(), 1);
    assert_eq!(counts_with_index(&dataset).await, scan_counts);
}

/// A merge that selects only SOME of the segments contributing to a
/// destination must not strip the unselected sibling of its coverage.
/// Two per-fragment segments S1 (fragment 0) and S2 (fragment 1) both
/// contribute to destinations 10 and 11 after the recluster. Merging
/// S2 with newly appended data produces M whose provenance still names
/// fragment 1, so the reader keeps resolving 10 and 11 from {S1, M}.
/// With a `stored ∩ live` bitmap M would drop fragment 1 and S1 would
/// lose both destinations.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn merge_keeps_unselected_sibling_coverage_on_tagged_tables() {
    use lance_index::optimize::OptimizeOptions;

    let fixture = MockDatasetFixture::try_new().unwrap();
    let (mut dataset, _map_id) = make_tagged_per_fragment(&fixture).await;
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 2);
    let s1 = segments
        .iter()
        .find(|s| s.fragment_bitmap.as_ref().unwrap().contains(0))
        .unwrap()
        .uuid;

    // New data so a one-segment merge has something to fold in.
    let appended = arrow_array::record_batch!(("i", Int32, [8, 9, 10, 11])).unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
            Some(WriteParams {
                store_params: Some(fixture.os_params()),
                commit_handler: Some(Arc::new(RenameCommitHandler)),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let all_rows = dataset.count_rows(None).await.unwrap();
    let scan_counts = counts_without_index(&dataset).await;

    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();

    let after = stored_segments(&dataset, "i_idx").await;
    assert_eq!(after.len(), 2, "{after:?}");
    assert!(after.iter().any(|s| s.uuid == s1), "S1 was not selected");
    // Both destinations are still derived for S1 and for the merged
    // segment, and every live fragment is covered.
    let derived = dataset.load_indices().await.unwrap();
    for segment in derived.iter().filter(|s| s.name == "i_idx") {
        let coverage = segment.fragment_bitmap.as_ref().unwrap();
        assert!(
            coverage.contains(10) && coverage.contains(11),
            "segment {} lost translated coverage: {coverage:?}",
            segment.uuid
        );
    }
    assert_eq!(
        derived_coverage(&dataset, "i_idx").await,
        dataset.fragment_bitmap.as_ref().clone()
    );
    assert_index_used(&dataset).await;
    assert_eq!(dataset.count_rows(None).await.unwrap(), all_rows);
    assert_eq!(counts_with_index(&dataset).await, scan_counts);
}

/// Sorted per-value row counts through the scalar index.
async fn counts_with_index(dataset: &Dataset) -> Vec<usize> {
    let mut counts = Vec::new();
    for value in 0..12 {
        let mut scan = dataset.scan();
        scan.filter(&format!("i = {value}")).unwrap();
        counts.push(scan.try_into_batch().await.unwrap().num_rows());
    }
    counts
}

/// The same counts with the scalar index disabled: the ground truth.
async fn counts_without_index(dataset: &Dataset) -> Vec<usize> {
    let mut counts = Vec::new();
    for value in 0..12 {
        let mut scan = dataset.scan();
        scan.filter(&format!("i = {value}")).unwrap();
        scan.use_scalar_index(false);
        counts.push(scan.try_into_batch().await.unwrap().num_rows());
    }
    counts
}

/// Union of the coverage the tagged reader derives for `name`.
async fn derived_coverage(dataset: &Dataset, name: &str) -> roaring::RoaringBitmap {
    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|s| s.name == name)
        .filter_map(|s| s.fragment_bitmap.clone())
        .fold(roaring::RoaringBitmap::new(), |acc, b| acc | b)
}

/// A filtered scan plans through the scalar index (no scan fallback).
async fn assert_index_used(dataset: &Dataset) {
    let mut scan = dataset.scan();
    scan.filter("i >= 4").unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
}

/// `make_tagged` with `i_idx` built as one segment per source fragment.
async fn make_tagged_per_fragment(fixture: &MockDatasetFixture) -> (Dataset, String) {
    let data = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_reader_rows(
            lance_datagen::RowCount::from(4),
            lance_datagen::BatchCount::from(2),
        );
    Dataset::write(
        data,
        &fixture.dataset_path,
        Some(WriteParams {
            store_params: Some(fixture.os_params()),
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            mode: WriteMode::Create,
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let mut dataset = *fixture.open().await.unwrap();
    let params = ScalarIndexParams::default();
    let fragment_ids: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();
    let mut segments = Vec::new();
    for fragment in fragment_ids {
        segments.push(
            crate::index::CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
                .name("i_idx".into())
                .fragments(vec![fragment])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments("i_idx", "i", segments)
        .await
        .unwrap();
    reserve_fragments(&mut dataset, 20).await;
    let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
    let (transition, destinations) = reader_tests::prepare(&dataset).await;
    let Some(pb_fri::transition::Mapping::StablePartition(mapping)) = &transition.mapping else {
        unreachable!()
    };
    let map_id = mapping.map_id.clone();
    let read_version = dataset.manifest.version;
    let frag_reuse_index = Some(
        crate::index::frag_reuse::frag_reuse_entry_appending(&dataset, vec![transition])
            .await
            .unwrap(),
    );
    let dataset = CommitBuilder::new(Arc::new(dataset))
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
        .unwrap();
    (dataset, map_id)
}

/// The staged drain the merge tests were after, via delta segments:
/// partial direct coverage retains the transition; completing the
/// coverage lets one maintenance run prune the superseded segment,
/// release the transition, and (aged) collect the row map.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn delta_drain_stages_partial_then_full() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    let (mut dataset, map_id) = make_tagged(&fixture).await;
    let all_rows = dataset.count_rows(None).await.unwrap();

    // Stage 1: only destination 10 gains direct coverage; the
    // transition and the old segment must both stay.
    commit_delta_segment(&mut dataset, Some(&[10])).await;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    assert_eq!(stored_segments(&dataset, "i_idx").await.len(), 2);
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_some(),
        "destination 11 lacks direct coverage, so the transition stays"
    );
    assert_eq!(fixture.list_fri_map_dirs().await, vec![map_id.clone()]);

    // Stage 2: destination 11 gains direct coverage too. One
    // maintenance run prunes the superseded old segment and releases
    // the transition.
    let second = commit_delta_segment(&mut dataset, Some(&[11])).await;
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();
    let segments = stored_segments(&dataset, "i_idx").await;
    assert_eq!(segments.len(), 2, "the two deltas remain");
    assert!(segments.iter().any(|s| s.uuid == second));
    assert!(
        dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .is_none(),
        "with the old segment pruned, nothing needs the transition"
    );
    assert_eq!(dataset.count_rows(None).await.unwrap(), all_rows);

    // Stage 3: the released row map ages out with its manifests.
    MockClock::set_system_time(TimeDelta::try_days(10).unwrap().to_std().unwrap());
    fixture
        .run_cleanup(utc_now() - TimeDelta::try_seconds(1).unwrap())
        .await
        .unwrap();
    assert!(fixture.list_fri_map_dirs().await.is_empty());
    let reopened = fixture.open().await.unwrap();
    assert_eq!(reopened.count_rows(None).await.unwrap(), all_rows);
}

/// D-guard: a v0 dataset records no tagged entries and has no `_fri`
/// directory; cleanup behaves exactly as before.
#[tokio::test]
async fn v0_dataset_is_untouched() {
    let fixture = MockDatasetFixture::try_new().unwrap();
    fixture.create_some_data().await.unwrap();
    fixture.overwrite_some_data().await.unwrap();
    MockClock::set_system_time(TimeDelta::try_days(10).unwrap().to_std().unwrap());
    let removed = fixture
        .run_cleanup(utc_now() - TimeDelta::try_days(8).unwrap())
        .await
        .unwrap();
    assert_eq!(removed.old_versions, 1);
    assert!(fixture.list_fri_map_dirs().await.is_empty());
}
