// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A user index built at a version before a stable-partition rewrite and
//! committed after it. The committed rewrite is read back from its
//! transaction file, where `frag_reuse` is never serialized, so the resolver
//! cannot see from the transaction that remapping was deferred; the latest
//! manifest's tagged entry records the lineage and proves it. The index then
//! lands with its provenance and translates, instead of a retryable conflict
//! that throws the finished build away. The deferred rules still apply: a
//! segment covering only part of a rewrite group, or an NGram segment, must
//! be rebuilt over the destinations; a segment off the lineage keeps the
//! eager rule.

use super::*;
use crate::dataset::cleanup::CleanupPolicyBuilder;
use crate::dataset::{
    InsertBuilder, MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched,
    WriteMode, WriteParams,
};

/// Two fragments of four rows: `i` 0..8, a constant `text` and a payload
/// `w` (so a partial-schema in-place rewrite is expressible).
async fn fixture(uri: &str) -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(uri, FragmentCount::from(2), FragmentRowCount::from(4))
        .await
        .unwrap()
}

/// A segment built by a session that still reads the pre-rewrite version;
/// that session is returned so it can commit later, from that read version.
async fn stage(
    dataset: &Dataset,
    column: &str,
    index_type: IndexType,
    fragments: Vec<u32>,
) -> (Dataset, IndexMetadata) {
    let mut stale = dataset.clone();
    let params = ScalarIndexParams::for_builtin(index_type.try_into().unwrap());
    let segment = crate::index::CreateIndexBuilder::new(&mut stale, &[column], index_type, &params)
        .name("idx".to_string())
        .fragments(fragments)
        .execute_uncommitted()
        .await
        .unwrap();
    (stale, segment)
}

/// Sorted `i` values under a predicate, with or without the scalar index.
async fn i_values(dataset: &Dataset, predicate: Option<&str>, use_index: bool) -> Vec<i32> {
    let mut scan = dataset.scan();
    if let Some(predicate) = predicate {
        scan.filter(predicate).unwrap();
    }
    scan.use_scalar_index(use_index);
    let batch = scan.try_into_batch().await.unwrap();
    let mut values: Vec<i32> = batch["i"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect();
    values.sort_unstable();
    values
}

/// The index answers `predicate` through the plan and agrees with the
/// index-disabled scan, for the predicate and for the whole table.
async fn assert_index_serves(dataset: &Dataset, predicate: &str) -> Vec<i32> {
    let plan = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    let indexed = i_values(dataset, Some(predicate), true).await;
    assert_eq!(indexed, i_values(dataset, Some(predicate), false).await);
    assert_eq!(
        i_values(dataset, None, true).await,
        i_values(dataset, None, false).await
    );
    indexed
}

fn stored_bitmap(indices: &[IndexMetadata]) -> RoaringBitmap {
    indices
        .iter()
        .find(|idx| idx.name == "idx")
        .unwrap()
        .fragment_bitmap
        .clone()
        .unwrap()
}

enum IndexCommitMode {
    Immediate,
    Staged,
    LatestHandle,
    Uri,
}

#[rstest::rstest]
#[case::indexed_payload("text", true)]
#[case::preserved_join_key("i", false)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_build_withdraws_columns_rewritten_before_cleanup(
    #[case] indexed_column: &str,
    #[case] is_withdrawn: bool,
    #[values(false, true)] is_tagged: bool,
    #[values(
        IndexCommitMode::Immediate,
        IndexCommitMode::Staged,
        IndexCommitMode::LatestHandle,
        IndexCommitMode::Uri
    )]
    commit_mode: IndexCommitMode,
) {
    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let mut stale = if is_tagged {
        make_tagged(dataset).await
    } else {
        dataset
    };
    let read_version = stale.version().version;
    let needs_retained_snapshot = matches!(
        commit_mode,
        IndexCommitMode::LatestHandle | IndexCommitMode::Uri
    );
    if needs_retained_snapshot {
        stale.tags().create("build", read_version).await.unwrap();
    }
    let fragments = stale.fragments().iter().map(|f| f.id as u32).collect();
    let segment = if !matches!(commit_mode, IndexCommitMode::Immediate) {
        Some(
            stage(&stale, indexed_column, IndexType::BTree, fragments)
                .await
                .1,
        )
    } else {
        None
    };

    // Only one fragment is rewritten. Its join key keeps its original file,
    // while the text column moves to a new file under the same field id.
    let source =
        arrow_array::record_batch!(("i", Int32, [0]), ("text", Utf8, ["patched"])).unwrap();
    let (latest, _) = MergeInsertBuilder::try_new(Arc::new(stale.clone()), vec!["i".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    let rewrite_version = latest.version().version;
    let latest = crate::dataset::CommitBuilder::new(latest)
        .execute(Transaction::new(
            rewrite_version,
            Operation::ReserveFragments { num_fragments: 1 },
            None,
        ))
        .await
        .unwrap();
    let stats = latest
        .cleanup(
            CleanupPolicyBuilder::default()
                .versions((1..=rewrite_version).collect())
                .unwrap()
                .error_if_tagged_old_versions(false)
                .build(),
        )
        .execute()
        .await
        .unwrap();
    assert_eq!(
        stats.old_versions,
        rewrite_version - u64::from(needs_retained_snapshot)
    );
    assert_eq!(stale.version().version, read_version);

    if let Some(segment) = segment {
        if needs_retained_snapshot {
            let transaction = Transaction::new(
                read_version,
                Operation::CreateIndex {
                    new_indices: vec![segment],
                    removed_indices: vec![],
                },
                None,
            );
            let builder = match commit_mode {
                IndexCommitMode::LatestHandle => {
                    crate::dataset::CommitBuilder::new(Arc::new(latest))
                }
                IndexCommitMode::Uri => crate::dataset::CommitBuilder::new(dir.as_str()),
                _ => unreachable!(),
            };
            builder.execute(transaction).await.unwrap();
        } else {
            stale
                .commit_existing_index_segments("idx", indexed_column, vec![segment])
                .await
                .unwrap();
        }
    } else {
        stale
            .create_index(
                &[indexed_column],
                IndexType::BTree,
                Some("idx".to_string()),
                &ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
    }

    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    let expected = if is_tagged { [10u32, 11] } else { [0u32, 1] };
    assert_eq!(
        stored_bitmap(&stored),
        if is_withdrawn {
            RoaringBitmap::from_iter([expected[1]])
        } else {
            RoaringBitmap::from_iter(expected)
        }
    );
    let predicates = if is_withdrawn {
        ["text = 'document'", "text = 'patched'"]
    } else {
        ["i = 0", "i = 7"]
    };
    for predicate in predicates {
        let expected_rows = i_values(&dataset, Some(predicate), false).await;
        assert_eq!(
            assert_index_serves(&dataset, predicate).await,
            expected_rows
        );
    }
    assert_eq!(
        i_values(&dataset, Some("text = 'patched'"), false).await,
        vec![0]
    );
}

#[rstest::rstest]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_build_after_cleanup_withdraws_rewritten_retired_provenance(
    #[values(false, true)] has_cleanup: bool,
    #[values(false, true)] is_staged: bool,
) {
    let dir = TempStrDir::default();
    let mut stale = fixture(dir.as_str()).await;
    let segment = if is_staged {
        Some(stage(&stale, "text", IndexType::BTree, vec![0, 1]).await.1)
    } else {
        None
    };
    let source =
        arrow_array::record_batch!(("i", Int32, [0]), ("text", Utf8, ["patched"])).unwrap();
    let (latest, _) = MergeInsertBuilder::try_new(Arc::new(stale.clone()), vec!["i".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    // The changed column no longer shares any fragment IDs with the build.
    let latest = make_tagged(latest.as_ref().clone()).await;
    let partition_version = latest.version().version;
    let latest = crate::dataset::CommitBuilder::new(Arc::new(latest))
        .execute(Transaction::new(
            partition_version,
            Operation::ReserveFragments { num_fragments: 1 },
            None,
        ))
        .await
        .unwrap();
    if has_cleanup {
        let stats = latest
            .cleanup(
                CleanupPolicyBuilder::default()
                    .versions((1..=partition_version).collect())
                    .unwrap()
                    .build(),
            )
            .execute()
            .await
            .unwrap();
        assert_eq!(stats.old_versions, partition_version);
    }
    if let Some(segment) = segment {
        stale
            .commit_existing_index_segments("idx", "text", vec![segment])
            .await
            .unwrap();
    } else {
        stale
            .create_index(
                &["text"],
                IndexType::BTree,
                Some("idx".into()),
                &ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();
    }
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        i_values(&dataset, Some("text = 'patched'"), false).await,
        vec![0]
    );
    for predicate in ["text = 'patched'", "text = 'document'"] {
        assert_eq!(
            i_values(&dataset, Some(predicate), true).await,
            i_values(&dataset, Some(predicate), false).await,
            "has_cleanup={has_cleanup}, is_staged={is_staged}, predicate={predicate}"
        );
    }
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert_eq!(
        stored_bitmap(&stored),
        if has_cleanup {
            RoaringBitmap::new()
        } else {
            RoaringBitmap::from_iter([1u32])
        }
    );
}

#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn multi_segment_fmindex_after_cleanup_returns_empty_definition_and_scans() {
    let dir = TempStrDir::default();
    let mut stale = fixture(dir.as_str()).await;
    let source =
        arrow_array::record_batch!(("i", Int32, [0]), ("text", Utf8, ["patched"])).unwrap();
    let (latest, _) = MergeInsertBuilder::try_new(Arc::new(stale.clone()), vec!["i".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    let latest = make_tagged(latest.as_ref().clone()).await;
    let partition_version = latest.version().version;
    let latest = crate::dataset::CommitBuilder::new(Arc::new(latest))
        .execute(Transaction::new(
            partition_version,
            Operation::ReserveFragments { num_fragments: 1 },
            None,
        ))
        .await
        .unwrap();
    let stats = latest
        .cleanup(
            CleanupPolicyBuilder::default()
                .versions((1..=partition_version).collect())
                .unwrap()
                .build(),
        )
        .execute()
        .await
        .unwrap();
    assert_eq!(stats.old_versions, partition_version);

    let commit_version = latest.version().version + 1;
    let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::Fm)
        .with_params(&serde_json::json!({ "num_segments": 2 }));
    let committed = stale
        .create_index(&["text"], IndexType::Fm, Some("idx".into()), &params, false)
        .await
        .unwrap();

    assert_eq!(stale.version().version, commit_version);
    assert_eq!(committed.name, "idx");
    assert!(
        committed
            .fragment_bitmap
            .as_ref()
            .is_some_and(RoaringBitmap::is_empty),
        "all retired coverage must be withdrawn"
    );
    assert!(
        stale.load_indices_by_name("idx").await.unwrap().is_empty(),
        "the empty definition is not query-visible"
    );
    let stored = crate::index::loading::load_all_indices(&stale)
        .await
        .unwrap();
    let definitions = stored
        .iter()
        .filter(|index| index.name == "idx")
        .collect::<Vec<_>>();
    assert_eq!(definitions.len(), 2);
    assert!(definitions.iter().all(|index| {
        index
            .fragment_bitmap
            .as_ref()
            .is_some_and(RoaringBitmap::is_empty)
    }));
    assert!(definitions.iter().any(|index| index.uuid == committed.uuid));

    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.version().version, commit_version);
    assert_eq!(
        i_values(&dataset, Some("contains(text, 'patched')"), false).await,
        vec![0]
    );
    assert_eq!(
        i_values(&dataset, Some("contains(text, 'document')"), false).await,
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    for predicate in ["contains(text, 'patched')", "contains(text, 'document')"] {
        assert_eq!(
            i_values(&dataset, Some(predicate), true).await,
            i_values(&dataset, Some(predicate), false).await,
            "{predicate}"
        );
    }
}

#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_built_before_a_stable_partition_lands_translating() {
    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let (mut stale, segment) = stage(&dataset, "i", IndexType::BTree, vec![0, 1]).await;
    let tagged = make_tagged(dataset).await;

    stale
        .commit_existing_index_segments("idx", "i", vec![segment.clone()])
        .await
        .unwrap();
    assert_eq!(stale.manifest.version, tagged.manifest.version + 1);

    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert_eq!(
        stored.iter().find(|idx| idx.name == "idx").unwrap().uuid,
        segment.uuid
    );
    assert_eq!(
        stored_bitmap(&stored),
        RoaringBitmap::from_iter([0u32, 1]),
        "provenance: the retired sources it was built from"
    );
    let derived = dataset.load_indices().await.unwrap();
    assert_eq!(
        stored_bitmap(&derived),
        RoaringBitmap::from_iter([10u32, 11]),
        "it translates to both destinations"
    );
    assert_eq!(assert_index_serves(&dataset, "i = 3").await, vec![3]);
    assert_eq!(assert_index_serves(&dataset, "i = 6").await, vec![6]);
}

/// The segment covers fragment 0 but not fragment 1, and the rewrite
/// consumed both: it lands with that provenance and claims nothing for the
/// partition (the reader scans it), which a later optimize repairs.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_covering_part_of_a_partition_lands_claiming_nothing() {
    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let (mut stale, segment) = stage(&dataset, "i", IndexType::BTree, vec![0]).await;
    let tagged = make_tagged(dataset).await;

    stale
        .commit_existing_index_segments("idx", "i", vec![segment.clone()])
        .await
        .unwrap();
    assert_eq!(stale.manifest.version, tagged.manifest.version + 1);
    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert_eq!(stored_bitmap(&stored), RoaringBitmap::from_iter([0u32]));
    let derived = dataset.load_indices().await.unwrap();
    assert!(
        derived
            .iter()
            .find(|idx| idx.name == "idx")
            .and_then(|idx| idx.fragment_bitmap.as_ref())
            .is_none_or(|bitmap| bitmap.is_empty()),
        "nothing claimed for the partition"
    );
    assert_eq!(
        i_values(&dataset, Some("i = 3"), true).await,
        i_values(&dataset, Some("i = 3"), false).await
    );
}

/// An NGram segment over retired sources is refused by the committed-segment
/// check, so the conflict asks for the rebuild up front.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn ngram_index_built_before_a_stable_partition_retries() {
    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let (mut stale, segment) = stage(&dataset, "text", IndexType::NGram, vec![0, 1]).await;
    make_tagged(dataset).await;

    let error = stale
        .commit_existing_index_segments("idx", "text", vec![segment])
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
}

/// A segment over a fragment the rewrite never touched lands untouched: the
/// group is off the segment's bitmap, so neither rule fires.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_off_the_rewritten_fragments_lands_directly() {
    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let batch = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
        .into_batch_rows(lance_datagen::RowCount::from(4))
        .unwrap();
    let dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap();
    assert_eq!(dataset.fragments().len(), 3);
    let (mut stale, segment) = stage(&dataset, "i", IndexType::BTree, vec![2]).await;

    let mut dataset = dataset;
    reserve(&mut dataset, 20).await;
    let old_fragments: Vec<Fragment> = dataset
        .fragments()
        .iter()
        .filter(|f| f.id < 2)
        .cloned()
        .collect();
    let (transition, destinations) = prepare_partition(&dataset, &[0, 1], 10).await;
    let version = dataset.manifest.version;
    let tagged = commit_sp(
        &dataset,
        version,
        tagged_rewrite(&dataset, old_fragments, destinations, vec![transition]).await,
    )
    .await
    .unwrap();

    stale
        .commit_existing_index_segments("idx", "i", vec![segment])
        .await
        .unwrap();
    assert_eq!(stale.manifest.version, tagged.manifest.version + 1);
    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert_eq!(stored_bitmap(&stored), RoaringBitmap::from_iter([2u32]));
    let derived = dataset.load_indices().await.unwrap();
    assert_eq!(stored_bitmap(&derived), RoaringBitmap::from_iter([2u32]));
    assert_eq!(assert_index_serves(&dataset, "i = 9").await, vec![9]);
}

/// Between the build and the commit, a payload rewrite landed on a
/// destination. A stale index on that payload loses the transition's
/// coverage until `optimize_indices` rebuilds it; an index on the preserved
/// join key still translates to the live destinations.
#[rstest::rstest]
#[case::payload("text", true)]
#[case::join_key("i", false)]
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_built_before_a_stable_partition_and_a_column_rewrite_preserves_valid_coverage(
    #[case] indexed_column: &str,
    #[case] withdrawn: bool,
) {
    use crate::dataset::{MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched};
    use lance_index::optimize::OptimizeOptions;

    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let (mut stale, segment) = stage(&dataset, indexed_column, IndexType::BTree, vec![0, 1]).await;
    let tagged = make_tagged(dataset).await;

    // Patch `text` of the row `i = 3` (fragment 11) in place. The payload
    // index becomes stale, while the preserved join key remains indexed.
    let schema = Arc::new(ArrowSchema::from(
        &tagged.schema().project(&["i", "text"]).unwrap(),
    ));
    let source = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![3])),
            Arc::new(arrow_array::StringArray::from(vec!["patched"])),
        ],
    )
    .unwrap();
    let (tagged, _) = MergeInsertBuilder::try_new(Arc::new(tagged), vec!["i".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    assert_eq!(
        tagged.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![10, 11],
        "rewritten in place"
    );

    stale
        .commit_existing_index_segments("idx", indexed_column, vec![segment.clone()])
        .await
        .unwrap();
    let mut dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::loading::load_all_indices(&dataset)
        .await
        .unwrap();
    assert_eq!(
        stored.iter().find(|idx| idx.name == "idx").unwrap().uuid,
        segment.uuid
    );
    assert_eq!(
        stored_bitmap(&stored),
        if withdrawn {
            RoaringBitmap::new()
        } else {
            RoaringBitmap::from_iter([0u32, 1])
        }
    );
    if !withdrawn {
        let derived = dataset.load_indices().await.unwrap();
        assert_eq!(
            stored_bitmap(&derived),
            RoaringBitmap::from_iter([10u32, 11])
        );
        assert_eq!(assert_index_serves(&dataset, "i = 3").await, vec![3]);
    }
    assert_eq!(
        i_values(&dataset, Some("text = 'patched'"), true).await,
        i_values(&dataset, Some("text = 'patched'"), false).await
    );
    assert_eq!(
        i_values(&dataset, None, true).await,
        (0..8).collect::<Vec<_>>()
    );

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let derived = dataset.load_indices().await.unwrap();
    assert_eq!(
        derived
            .iter()
            .filter(|idx| idx.name == "idx")
            .filter_map(|idx| idx.fragment_bitmap.clone())
            .fold(RoaringBitmap::new(), |acc, b| acc | b),
        RoaringBitmap::from_iter([10u32, 11])
    );
    let predicate = if withdrawn {
        "text = 'patched'"
    } else {
        "i = 3"
    };
    assert_eq!(assert_index_serves(&dataset, predicate).await, vec![3]);
}

/// The same race with `alter_columns`: a cast of the indexed column landed
/// between the build and the commit (admitted, no committed index covered
/// it yet). A cast rewrites the column under a new field id, so the merge
/// arm sees the field the index keys on removed from the schema and the
/// commit retries against the new field.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_built_before_a_stable_partition_and_a_cast_retries() {
    use crate::dataset::ColumnAlteration;
    use arrow_schema::DataType;

    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let (mut stale, segment) = stage(&dataset, "i", IndexType::BTree, vec![0, 1]).await;
    let mut tagged = make_tagged(dataset).await;
    tagged
        .alter_columns(&[ColumnAlteration::new("i".into()).cast_to(DataType::Int64)])
        .await
        .unwrap();
    assert_eq!(
        tagged.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![10, 11],
        "cast in place"
    );

    let error = stale
        .commit_existing_index_segments("idx", "i", vec![segment])
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
}
