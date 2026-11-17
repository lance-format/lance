// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction tests for the overlays contract area.

use super::*;
use crate::dataset::optimize::commit::normalize_source_fragments;

#[rstest]
#[case::default_target(LanceFileVersion::V2_2, None)]
#[case::explicit_target(LanceFileVersion::V2_2, Some(LanceFileVersion::V2_2))]
#[case::non_default_target(LanceFileVersion::V2_0, Some(LanceFileVersion::V2_2))]
#[tokio::test]
async fn test_compaction_target_preserves_overlays(
    #[case] default_version: LanceFileVersion,
    #[case] target: Option<LanceFileVersion>,
    #[values(
        CompactionMode::Reencode,
        CompactionMode::TryBinaryCopy,
        CompactionMode::ForceBinaryCopy
    )]
    mode: CompactionMode,
) {
    let batch = record_batch!(
        ("id", Int32, [0, 1, 2, 3]),
        ("val", Int32, [10, 20, 30, 40])
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(RecordBatch::new_empty(batch.schema()))], batch.schema()),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(default_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    // The first fragment is clean; eligibility must inspect every fragment.
    let mut dataset = commit_overlay(
        dataset,
        1,
        &[1],
        OverlayCoverage::dense(bitmap([0, 1])),
        vec![i32_array([Some(999), None])],
    )
    .await;
    let expected = BTreeMap::from([(0, Some(10)), (1, Some(20)), (2, Some(999)), (3, None)]);
    assert_eq!(id_val_map(&dataset).await, expected);
    let manifest = dataset.manifest.clone();
    let options = CompactionOptions {
        target_rows_per_fragment: 4,
        data_storage_version: target,
        compaction_mode: Some(mode),
        ..Default::default()
    };
    let result = compact_files(&mut dataset, options, None).await;
    if mode == CompactionMode::ForceBinaryCopy {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("binary copy is not supported"));
        assert_eq!(dataset.manifest, manifest);
    } else {
        let metrics = result.unwrap();
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);
        assert!(dataset.manifest.fragments[0].overlays.is_empty());
        assert_eq!(
            dataset.manifest.fragments[0].files[0]
                .file_version()
                .unwrap(),
            ConcreteFileVersion::V2_2
        );
    }
    assert_eq!(id_val_map(&dataset).await, expected);
    assert_eq!(
        dataset.manifest.data_storage_format.lance_file_format(),
        default_version.resolve()
    );
    dataset.validate().await.unwrap();
}
#[tokio::test]
async fn test_max_overlays_triggers_full_compaction() {
    // Fragment 0 gets 3 overlays; fragment 1 stays clean.
    let dataset = create_base_dataset("memory://").await;
    let mut dataset = commit_n_overlays(dataset, 3).await;
    assert_eq!(
        dataset.get_fragment(0).unwrap().metadata().overlays.len(),
        3
    );

    // Threshold 2: only fragment 0 (3 > 2) is compacted.
    let metrics = compact_files(&mut dataset, overlay_only_options(2), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 1);
    assert_eq!(metrics.fragments_added, 1);

    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 2);
    // The compacted fragment is a fresh single-data-file fragment with no
    // overlays; fragment 1 is untouched.
    let compacted = fragments
        .iter()
        .find(|f| f.id() != 1)
        .expect("a new fragment id was assigned");
    assert!(compacted.metadata().overlays.is_empty());
    assert_eq!(compacted.metadata().files.len(), 1);

    // The overlaid values were materialized: id i in 0..3 -> 1000 + i.
    let values = id_val_map(&dataset).await;
    let expected: BTreeMap<i32, Option<i32>> = (0..12)
        .map(|id| {
            let v = if id < 3 { 1000 + id } else { id * 10 };
            (id, Some(v))
        })
        .collect();
    assert_eq!(values, expected);
}
#[tokio::test]
async fn test_below_threshold_is_a_noop() {
    let dataset = create_base_dataset("memory://").await;
    let mut dataset = commit_n_overlays(dataset, 2).await;

    // 2 overlays, threshold 2: `overlays > max` is false, so no compaction.
    let metrics = compact_files(&mut dataset, overlay_only_options(2), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 0);
    assert_eq!(metrics.fragments_added, 0);
    assert_eq!(
        dataset.get_fragment(0).unwrap().metadata().overlays.len(),
        2
    );
}
#[tokio::test]
async fn test_overlay_compaction_materializes_deletions() {
    let dataset = create_base_dataset("memory://").await;
    let mut dataset = commit_n_overlays(dataset, 3).await;
    // Delete a row from the overlaid fragment (id 2 is at offset 2).
    dataset.delete("id = 2").await.unwrap();
    assert!(
        dataset
            .get_fragment(0)
            .unwrap()
            .metadata()
            .deletion_file
            .is_some()
    );

    compact_files(&mut dataset, overlay_only_options(2), None)
        .await
        .unwrap();

    // The deletion was materialized: no deletion file remains and id 2 is gone.
    for fragment in dataset.get_fragments() {
        assert!(fragment.metadata().deletion_file.is_none());
        assert!(fragment.metadata().overlays.is_empty());
    }
    let values = id_val_map(&dataset).await;
    assert!(!values.contains_key(&2));
    // Surviving overlaid cells still carry their materialized values.
    assert_eq!(values.get(&0), Some(&Some(1000)));
    assert_eq!(values.get(&1), Some(&Some(1001)));
}
#[rstest]
#[case::eager_remap(false)]
#[case::deferred_remap(true)]
#[tokio::test]
async fn test_overlay_compaction_reconciles_stale_index(#[case] defer_index_remap: bool) {
    let mut dataset = create_base_dataset("memory://").await;
    // Index `val` before any overlay -> the index is stale once val is overlaid.
    dataset
        .create_index(
            &["val"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let indexed_version = dataset.version().version;

    // Overlay val[0] 0 -> 100 (committed after the index) and push fragment 0
    // over the overlay limit.
    let mut dataset = commit_n_overlays(dataset, 3).await;

    let val_index_before = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .find(|i| i.fields == vec![1])
        .expect("val index present")
        .clone();
    assert!(
        val_index_before
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .contains(0)
    );

    let options = CompactionOptions {
        defer_index_remap,
        ..overlay_only_options(2)
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    // The stale val index no longer covers the compacted fragment, so its
    // rows fall back to a flat scan instead of serving stale values. A
    // deferred remap leaves the old fragment id in the bitmap for the
    // reuse index to translate later, so it must be gone too.
    let indices = dataset.load_indices().await.unwrap();
    let val_index = indices
        .iter()
        .find(|i| i.fields == vec![1])
        .expect("val index present");
    let compacted_id = dataset
        .get_fragments()
        .iter()
        .map(|f| f.id() as u32)
        .find(|id| *id != 1)
        .unwrap();
    assert!(
        !val_index
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .contains(compacted_id),
        "stale index must drop the compacted fragment from its coverage"
    );
    assert!(
        !val_index.fragment_bitmap.as_ref().unwrap().contains(0),
        "stale index must drop the source fragment from its coverage"
    );

    // The indexed query is correct: the materialized value is found and the
    // stale pre-overlay value is gone.
    let mut scanner = dataset.scan();
    scanner
        .filter("val = 1000")
        .unwrap()
        .project(&["id"])
        .unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids.value(0), 0);

    let mut scanner = dataset.scan();
    scanner.filter("val = 0").unwrap().project(&["id"]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 0, "stale value 0 must no longer match");

    // Pages a fresh reader of the latest snapshot warms through the reuse
    // index must not serve the snapshot from before the compaction.
    let historical = dataset.checkout_version(indexed_version).await.unwrap();
    let latest = historical
        .checkout_version(dataset.version().version)
        .await
        .unwrap();
    latest.prewarm_index("val_idx").await.unwrap();
    let mut scanner = historical.scan();
    scanner.filter("val = 0").unwrap();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 1);
}
/// Round 6: unlike `migrate_fragments`, the reuse-record normalization
/// must keep a fully-deleted fragment: the rewrite group still removes
/// it from the manifest and the transition still consumes it (zero live
/// rows), so a silently shortened source list would desynchronize the
/// digests, the group and the manifest.
#[tokio::test]
async fn test_normalize_source_fragments_keeps_fully_deleted() {
    use lance_table::format::{DeletionFile, DeletionFileType};
    let dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(4))
        .await
        .unwrap();
    let mut fragment = Fragment::new(7);
    fragment.physical_rows = Some(4);
    fragment.deletion_file = Some(DeletionFile {
        read_version: 1,
        id: 1,
        file_type: DeletionFileType::Array,
        num_deleted_rows: Some(4),
        base_id: None,
        inline: None,
    });
    let normalized = normalize_source_fragments(&dataset, &[fragment])
        .await
        .unwrap();
    assert_eq!(normalized.len(), 1);
    assert_eq!(normalized[0].id, 7);
    assert_eq!(normalized[0].physical_rows, Some(4));
    assert_eq!(
        normalized[0]
            .deletion_file
            .as_ref()
            .unwrap()
            .num_deleted_rows,
        Some(4)
    );
}
