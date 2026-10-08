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
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
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
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    assert_eq!(stored_bitmap(&stored), RoaringBitmap::from_iter([2u32]));
    let derived = dataset.load_indices().await.unwrap();
    assert_eq!(stored_bitmap(&derived), RoaringBitmap::from_iter([2u32]));
    assert_eq!(assert_index_serves(&dataset, "i = 9").await, vec![9]);
}

/// Between the build and the commit, an in-place rewrite of the indexed
/// column landed on a destination (rows did not move, so the rewrite is
/// admitted while no committed index covers the column). The index still
/// lands: the resolver withdraws the transition's sources from it, so it
/// claims nothing and the rows are scanned until `optimize_indices`
/// rebuilds it.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn index_built_before_a_stable_partition_and_a_column_rewrite_lands_withdrawn() {
    use crate::dataset::{MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched};
    use lance_index::optimize::OptimizeOptions;

    let dir = TempStrDir::default();
    let dataset = fixture(dir.as_str()).await;
    let (mut stale, segment) = stage(&dataset, "i", IndexType::BTree, vec![0, 1]).await;
    let tagged = make_tagged(dataset).await;

    // Patch `text` of the row `i = 3` (fragment 11) in place; the source
    // carries the key column, so `i` is rewritten in place as well.
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
        .commit_existing_index_segments("idx", "i", vec![segment.clone()])
        .await
        .unwrap();
    let mut dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    assert_eq!(
        stored.iter().find(|idx| idx.name == "idx").unwrap().uuid,
        segment.uuid
    );
    assert!(
        stored_bitmap(&stored).is_empty(),
        "{:?}",
        stored_bitmap(&stored)
    );
    assert_eq!(
        i_values(&dataset, Some("i = 3"), true).await,
        i_values(&dataset, Some("i = 3"), false).await
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
    assert_eq!(assert_index_serves(&dataset, "i = 3").await, vec![3]);
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
