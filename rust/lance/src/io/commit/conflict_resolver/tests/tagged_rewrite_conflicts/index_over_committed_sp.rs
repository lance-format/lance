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
use crate::dataset::{InsertBuilder, WriteMode, WriteParams};

/// Two fragments of four rows: `i` 0..8 and a constant `text`.
async fn fixture(uri: &str) -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "text",
            lance_datagen::array::fill_utf8("document".to_string()),
        )
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
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
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
/// consumed both: it would claim partial coverage of the destinations, so
/// the build must be redone over them.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_covering_part_of_a_partition_retries() {
    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let (mut stale, segment) = stage(&dataset, "i", IndexType::BTree, vec![0]).await;
    let tagged = make_tagged(dataset).await;

    let error = stale
        .commit_existing_index_segments("idx", "i", vec![segment])
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, tagged.manifest.version);
    assert!(
        crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .iter()
            .all(|idx| idx.name != "idx")
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
        dataset,
        version,
        tagged_rewrite(old_fragments, destinations, vec![transition]),
    )
    .await
    .unwrap();

    stale
        .commit_existing_index_segments("idx", "i", vec![segment])
        .await
        .unwrap();
    assert_eq!(stale.manifest.version, tagged.manifest.version + 1);
    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    assert_eq!(stored_bitmap(&stored), RoaringBitmap::from_iter([2u32]));
    let derived = dataset.load_indices().await.unwrap();
    assert_eq!(stored_bitmap(&derived), RoaringBitmap::from_iter([2u32]));
    assert_eq!(assert_index_serves(&dataset, "i = 9").await, vec![9]);
}
