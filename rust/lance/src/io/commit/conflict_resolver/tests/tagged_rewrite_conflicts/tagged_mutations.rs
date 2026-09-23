// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Normal mutations on a table carrying a tagged fragment reuse history.
//!
//! A delete, a row-rewriting update, a merge insert, a config update, an
//! overwrite, a restore, a schema change and an in-place rewrite of a column
//! no translating segment indexes all commit: the history and every
//! segment's stored provenance are untouched (or, for an overwrite, gone with
//! the old fragments), and the reader applies the current deletion vectors
//! after translation. An in-place rewrite of a column a translating segment
//! indexes withdraws that segment's coverage of the transition (the rows are
//! scanned) instead of being refused. Each test tags the table
//! first (stable partition of F0, F1 into F10 evens and F11 odds, with `i_idx`
//! built before), mutates through the public API, reopens from the URI and
//! checks indexed queries against index-disabled scans, that the segment is
//! still listed and still translating, and that the history is untouched.

use super::*;
use crate::dataset::{
    MergeInsertBuilder, MergeInsertWriteMode, UpdateBuilder, WhenMatched, WhenNotMatched,
};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

/// Two fragments of four rows each, columns `i` (indexed key), `v` and `w`
/// (payloads, initially equal to `i`), tagged by a stable partition.
async fn tagged_two_column_fixture(uri: &str) -> Dataset {
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col("v", lance_datagen::array::step::<Int32Type>())
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(uri, FragmentCount::from(2), FragmentRowCount::from(4))
        .await
        .unwrap();
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
    let dataset = make_tagged(dataset).await;
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![10, 11]
    );
    dataset
}

/// Sorted `(i, v)` pairs, optionally under a predicate, with or without the
/// scalar index.
async fn rows(dataset: &Dataset, predicate: Option<&str>, use_index: bool) -> Vec<(i32, i32)> {
    let mut scan = dataset.scan();
    if let Some(predicate) = predicate {
        scan.filter(predicate).unwrap();
    }
    scan.use_scalar_index(use_index);
    let batch = scan.try_into_batch().await.unwrap();
    let i = batch["i"].as_primitive::<Int32Type>();
    let v = batch["v"].as_primitive::<Int32Type>();
    let mut pairs: Vec<(i32, i32)> = i
        .values()
        .iter()
        .zip(v.values().iter())
        .map(|(i, v)| (*i, *v))
        .collect();
    pairs.sort_unstable();
    pairs
}

/// The indexed plan is used for `predicate`, and indexed and index-disabled
/// results agree for it and for the full table.
async fn assert_index_agrees_with_scan(dataset: &Dataset, predicate: &str) -> Vec<(i32, i32)> {
    let plan = dataset
        .scan()
        .filter(predicate)
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    let indexed = rows(dataset, Some(predicate), true).await;
    assert_eq!(
        indexed,
        rows(dataset, Some(predicate), false).await,
        "{predicate}"
    );
    assert_eq!(
        rows(dataset, None, true).await,
        rows(dataset, None, false).await
    );
    indexed
}

fn user_segment(indices: &[IndexMetadata]) -> &IndexMetadata {
    indices.iter().find(|idx| idx.name == "i_idx").unwrap()
}

fn fri_entry(indices: &[IndexMetadata]) -> &IndexMetadata {
    indices
        .iter()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .unwrap()
}

/// The stored segment and the history are exactly what they were before the
/// mutation, and the segment still translates: its derived coverage is the
/// live destinations, not its stored provenance.
async fn assert_segment_and_history_untouched(
    dataset: &Dataset,
    before: &[IndexMetadata],
    live: &[u32],
) {
    let stored = crate::index::load_all_indices(dataset).await.unwrap();
    let segment = user_segment(&stored);
    assert_eq!(segment.uuid, user_segment(before).uuid);
    assert_eq!(
        segment.fragment_bitmap,
        user_segment(before).fragment_bitmap
    );
    assert_eq!(
        segment.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1]),
        "stored provenance stays the retired sources"
    );
    assert_eq!(fri_entry(&stored).uuid, fri_entry(before).uuid);
    assert_eq!(fri_entry(&stored).index_version, 1);
    let derived = dataset.load_indices().await.unwrap();
    assert_eq!(
        user_segment(&derived).fragment_bitmap.as_ref().unwrap(),
        &live.iter().copied().collect::<RoaringBitmap>(),
        "derived coverage follows the live destinations"
    );
}

#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_partial_row_delete() {
    let dir = TempStrDir::default();
    let mut dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    dataset.delete("i = 5").await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.fragments().len(), 2);
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 5").await,
        vec![]
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 4").await,
        vec![(4, 4)]
    );
    assert_eq!(
        rows(&dataset, None, true).await,
        [0, 1, 2, 3, 4, 6, 7].map(|i| (i, i))
    );
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// Deleting every row of a destination drops the fragment. The translating
/// segment's stored bitmap names retired sources, so measuring it against the
/// live fragments would read as empty: it must be kept, not dropped as an
/// index with no coverage.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_whole_fragment_delete() {
    let dir = TempStrDir::default();
    let mut dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    // F10 holds the even values.
    dataset.delete("i % 2 = 0").await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![11]
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 2").await,
        vec![]
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 3").await,
        vec![(3, 3)]
    );
    assert_eq!(
        rows(&dataset, None, true).await,
        [1, 3, 5, 7].map(|i| (i, i))
    );
    assert_segment_and_history_untouched(&dataset, &before, &[11]).await;
}

#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_row_rewriting_update() {
    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    let result = UpdateBuilder::new(Arc::new(dataset))
        .update_where("i = 2")
        .unwrap()
        .set("v", "999")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    assert_eq!(result.rows_updated, 1);
    let dataset = fresh_session(dir.as_str()).await;
    // The rewritten row lives in a new fragment; the destinations stay live.
    assert_eq!(dataset.fragments().len(), 3);
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 2").await,
        vec![(2, 999)]
    );
    assert_eq!(
        rows(&dataset, None, true).await,
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| (i, if i == 2 { 999 } else { i }))
    );
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_merge_insert_rewriting_rows() {
    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    // A full-schema source: `Auto` rewrites rows (delete + new fragment).
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let source = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![3, 100])),
            Arc::new(Int32Array::from(vec![333, 1000])),
            Arc::new(Int32Array::from(vec![3, 100])),
        ],
    )
    .unwrap();
    let (dataset, stats) = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["i".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    assert_eq!((stats.num_updated_rows, stats.num_inserted_rows), (1, 1));
    let dataset = fresh_session(dataset.uri()).await;
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 3").await,
        vec![(3, 333)]
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 100").await,
        vec![(100, 1000)]
    );
    assert_eq!(rows(&dataset, None, true).await.len(), 9);
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// A scalar index on `v`, built before or after the table is tagged: before,
/// its bitmap is the retired sources and it covers the destinations only
/// through the history; after, it names the destinations directly.
async fn create_v_index(dataset: &mut Dataset) {
    dataset
        .create_index(
            &["v"],
            IndexType::Scalar,
            Some("v_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
}

/// A user index is installed on a tagged table like on any other: it covers
/// the live fragments directly, and the entry is carried through untouched.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_installs_a_new_user_index() {
    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();
    let mut dataset = fresh_session(dir.as_str()).await;
    create_v_index(&mut dataset).await;
    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    let v_idx = stored.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(
        v_idx.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11])
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "v = 3").await,
        vec![(3, 3)]
    );
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// A user index built before a stable partition and committed after it is
/// rebased across the rewrite by the conflict resolver: it lands with its
/// provenance (the retired sources) and translates, so it answers for the
/// destinations without a rebuild.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn user_index_built_before_a_stable_partition_lands_translating() {
    let dir = TempStrDir::default();
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col("v", lance_datagen::array::step::<Int32Type>())
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(
            dir.as_str(),
            FragmentCount::from(2),
            FragmentRowCount::from(4),
        )
        .await
        .unwrap();
    let read_version = dataset.manifest.version;
    let built = crate::index::CreateIndexBuilder::new(
        &mut dataset,
        &["v"],
        IndexType::Scalar,
        &ScalarIndexParams::default(),
    )
    .name("v_idx".into())
    .execute_uncommitted()
    .await
    .unwrap();
    assert_eq!(
        built.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1])
    );
    let mut dataset = make_tagged(dataset).await;
    dataset
        .apply_commit(
            Transaction::new(
                read_version,
                Operation::CreateIndex {
                    new_indices: vec![built],
                    removed_indices: vec![],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    let v_idx = stored.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(
        v_idx.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1]),
        "the provenance is kept"
    );
    let listed = dataset.load_indices().await.unwrap();
    let derived = listed.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(
        derived.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11]),
        "and translates to the destinations"
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "v = 3").await,
        vec![(3, 3)]
    );
    assert_eq!(
        rows(&dataset, None, true).await,
        rows(&dataset, None, false).await
    );
}

/// A withdrawal that empties a segment beside a serving sibling of the same
/// name removes that segment from the manifest in the same commit: the
/// sibling keeps serving, the rewritten rows are scanned, an older snapshot
/// still lists and serves the retired segment, and once no retained version
/// references it the ordinary index cleanup reclaims its files.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_retires_an_emptied_segment_beside_a_serving_sibling() {
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use lance_index::optimize::OptimizeOptions;

    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    // Append a fragment and extend `i_idx` with a delta segment over it.
    let batch = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .col("v", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .col("w", lance_datagen::array::step_custom::<Int32Type>(8, 1))
        .into_batch_rows(lance_datagen::RowCount::from(4))
        .unwrap();
    let mut dataset = InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let segments: Vec<IndexMetadata> = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == "i_idx")
        .cloned()
        .collect();
    assert_eq!(segments.len(), 2, "{segments:?}");
    let appended = dataset
        .fragments()
        .iter()
        .map(|f| f.id as u32)
        .find(|id| *id > 11)
        .unwrap();
    let translating = segments
        .iter()
        .find(|idx| idx.fragment_bitmap.as_ref().unwrap() == &RoaringBitmap::from_iter([0u32, 1]))
        .unwrap()
        .clone();
    let delta = segments
        .iter()
        .find(|idx| idx.fragment_bitmap.as_ref().unwrap() == &RoaringBitmap::from_iter([appended]))
        .unwrap()
        .clone();
    let version_before = dataset.manifest.version;

    // Row i = 2 (w = 2) lives in F10: rewriting `i` there withdraws the
    // translating segment's whole transition, emptying it.
    rewrite_in_place(dataset, "i", 2, 222).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let remaining: Vec<IndexMetadata> = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .filter(|idx| idx.name == "i_idx")
        .cloned()
        .collect();
    assert_eq!(remaining.len(), 1, "{remaining:?}");
    assert_eq!(remaining[0].uuid, delta.uuid);
    assert_eq!(remaining[0].fragment_bitmap, delta.fragment_bitmap);
    assert_eq!(rows(&dataset, Some("i = 222"), true).await, vec![(222, 2)]);
    assert_eq!(rows(&dataset, Some("i = 2"), true).await, vec![]);
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 9").await,
        vec![(9, 9)]
    );
    assert_eq!(
        rows(&dataset, None, true).await,
        rows(&dataset, None, false).await
    );

    // The older snapshot still lists and serves the retired segment.
    let old = dataset.checkout_version(version_before).await.unwrap();
    assert!(
        crate::index::load_all_indices(&old)
            .await
            .unwrap()
            .iter()
            .any(|idx| idx.uuid == translating.uuid)
    );
    assert_eq!(
        assert_index_agrees_with_scan(&old, "i = 2").await,
        vec![(2, 2)]
    );

    // Once no retained version references it, cleanup reclaims its files.
    let index_dir = |uuid: &Uuid| {
        std::path::Path::new(dir.as_str())
            .join("_indices")
            .join(uuid.to_string())
    };
    assert!(index_dir(&translating.uuid).exists());
    assert!(index_dir(&delta.uuid).exists());
    cleanup_old_versions(
        &dataset,
        CleanupPolicyBuilder::default()
            .before_timestamp(chrono::Utc::now() + chrono::Duration::days(1))
            .delete_unverified(true)
            .build(),
    )
    .await
    .unwrap();
    assert!(
        !index_dir(&translating.uuid).exists(),
        "the retired segment's files are reclaimed"
    );
    assert!(
        index_dir(&delta.uuid).exists(),
        "the serving sibling's files stay"
    );
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 9").await,
        vec![(9, 9)]
    );
}

/// A packed struct is rewritten as one physical column, but an index on its
/// child `s.x` records the child's field id: the withdrawal expands the
/// rewritten parent to its descendants, so the child index loses the
/// transition's coverage and the indexed query reads the patched value from
/// the scan.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_withdraws_a_child_index_when_its_packed_parent_is_rewritten() {
    use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, StructArray};
    use arrow_schema::{DataType, Field, Fields};
    use lance_encoding::constants::PACKED_STRUCT_META_KEY;
    use lance_file::version::LanceFileVersion;

    async fn i_values(dataset: &Dataset, predicate: &str, use_index: bool) -> Vec<i32> {
        let mut scan = dataset.scan();
        scan.filter(predicate).unwrap();
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

    let children = Fields::from(vec![Field::new("x", DataType::Int32, false)]);
    let mut packed = Field::new("s", DataType::Struct(children.clone()), false);
    packed.set_metadata([(PACKED_STRUCT_META_KEY.to_string(), "true".to_string())].into());
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("w", DataType::Int32, false),
        packed.clone(),
    ]));
    let values = Arc::new(Int32Array::from((0..8).collect::<Vec<_>>())) as ArrayRef;
    let struct_values = Arc::new(StructArray::new(
        children.clone(),
        vec![values.clone()],
        None,
    )) as ArrayRef;
    let batch =
        RecordBatch::try_new(schema.clone(), vec![values.clone(), values, struct_values]).unwrap();
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema.clone()),
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 4,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["s.x"],
            IndexType::BTree,
            Some("x_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let dataset = make_tagged(dataset).await;
    let x_idx = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .find(|idx| idx.name == "x_idx")
        .cloned()
        .unwrap();
    assert_eq!(
        x_idx.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1])
    );

    // Row w = 3 lives in F11 (odd rows); its packed struct is patched whole.
    let source = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("w", DataType::Int32, false),
            packed,
        ])),
        vec![
            Arc::new(Int32Array::from(vec![3])) as ArrayRef,
            Arc::new(StructArray::new(
                children,
                vec![Arc::new(Int32Array::from(vec![333])) as ArrayRef],
                None,
            )) as ArrayRef,
        ],
    )
    .unwrap();
    MergeInsertBuilder::try_new(Arc::new(dataset), vec!["w".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let x_idx = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .find(|idx| idx.name == "x_idx")
        .cloned()
        .unwrap();
    assert!(
        x_idx.fragment_bitmap.as_ref().unwrap().is_empty(),
        "the transition's sources are withdrawn from the child index: {:?}",
        x_idx.fragment_bitmap
    );
    assert_eq!(i_values(&dataset, "s.x = 333", false).await, vec![3]);
    assert_eq!(i_values(&dataset, "s.x = 333", true).await, vec![3]);
    assert_eq!(i_values(&dataset, "s.x = 3", true).await, Vec::<i32>::new());
    assert_eq!(
        i_values(&dataset, "s.x >= 0", true).await,
        (0..8).collect::<Vec<_>>()
    );
}

/// The rebase path of the packed-struct rule: an index on the child `s.x`
/// is built before the tagging rewrite and before the packed parent `s` is
/// rewritten in place, and commits after both. Rebasing it over the rewrite
/// must expand the rewritten parent to its descendants, exactly as the
/// manifest build does for a rewrite that lands after the index; otherwise
/// the index keeps the transition's coverage and answers `s.x = 3` with the
/// row whose value is now 333.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn child_index_built_before_a_packed_parent_rewrite_lands_withdrawn() {
    use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, StructArray};
    use arrow_schema::{DataType, Field, Fields};
    use lance_encoding::constants::PACKED_STRUCT_META_KEY;
    use lance_file::version::LanceFileVersion;

    async fn i_values(dataset: &Dataset, predicate: &str, use_index: bool) -> Vec<i32> {
        let mut scan = dataset.scan();
        scan.filter(predicate).unwrap();
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

    let children = Fields::from(vec![Field::new("x", DataType::Int32, false)]);
    let mut packed = Field::new("s", DataType::Struct(children.clone()), false);
    packed.set_metadata([(PACKED_STRUCT_META_KEY.to_string(), "true".to_string())].into());
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("w", DataType::Int32, false),
        packed.clone(),
    ]));
    let values = Arc::new(Int32Array::from((0..8).collect::<Vec<_>>())) as ArrayRef;
    let struct_values = Arc::new(StructArray::new(
        children.clone(),
        vec![values.clone()],
        None,
    )) as ArrayRef;
    let batch =
        RecordBatch::try_new(schema.clone(), vec![values.clone(), values, struct_values]).unwrap();
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema.clone()),
        dir.as_str(),
        Some(WriteParams {
            max_rows_per_file: 4,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Built against F0 and F1, committed later.
    let read_version = dataset.manifest.version;
    let built = crate::index::CreateIndexBuilder::new(
        &mut dataset,
        &["s.x"],
        IndexType::BTree,
        &ScalarIndexParams::default(),
    )
    .name("x_idx".into())
    .execute_uncommitted()
    .await
    .unwrap();
    assert_eq!(
        built.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1])
    );

    // Meanwhile: the table is tagged (F0, F1 -> F10, F11) and the packed
    // struct of the row w = 3 (in F11) is patched whole.
    let dataset = make_tagged(dataset).await;
    let source = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            Field::new("w", DataType::Int32, false),
            packed,
        ])),
        vec![
            Arc::new(Int32Array::from(vec![3])) as ArrayRef,
            Arc::new(StructArray::new(
                children,
                vec![Arc::new(Int32Array::from(vec![333])) as ArrayRef],
                None,
            )) as ArrayRef,
        ],
    )
    .unwrap();
    MergeInsertBuilder::try_new(Arc::new(dataset), vec!["w".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();

    // The build lands by rebasing over the tagging rewrite (provenance
    // kept) and the packed rewrite (the transition's coverage withdrawn).
    let mut dataset = fresh_session(dir.as_str()).await;
    dataset
        .apply_commit(
            Transaction::new(
                read_version,
                Operation::CreateIndex {
                    new_indices: vec![built],
                    removed_indices: vec![],
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let x_idx = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .find(|idx| idx.name == "x_idx")
        .cloned()
        .unwrap();
    assert!(
        x_idx.fragment_bitmap.as_ref().unwrap().is_empty(),
        "the transition's sources are withdrawn from the rebased child index: {:?}",
        x_idx.fragment_bitmap
    );
    assert_eq!(i_values(&dataset, "s.x = 333", false).await, vec![3]);
    assert_eq!(i_values(&dataset, "s.x = 333", true).await, vec![3]);
    assert_eq!(i_values(&dataset, "s.x = 3", true).await, Vec::<i32>::new());
    assert_eq!(
        i_values(&dataset, "s.x >= 0", true).await,
        (0..8).collect::<Vec<_>>()
    );
}

/// Patch `column` of the row `w = key` (`w` equals `i` and no index covers
/// it) to `value` in place: a partial-schema source keyed on an unindexed
/// column, in `RewriteColumns` mode, so only `column` and the key are
/// rewritten.
async fn rewrite_in_place(
    dataset: Dataset,
    column: &str,
    key: i32,
    value: i32,
) -> Result<Arc<Dataset>> {
    let schema = Arc::new(ArrowSchema::from(
        &dataset.schema().project(&["w", column]).unwrap(),
    ));
    let source = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![key])),
            Arc::new(Int32Array::from(vec![value])),
        ],
    )
    .unwrap();
    MergeInsertBuilder::try_new(Arc::new(dataset), vec!["w".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .map(|(dataset, _)| dataset)
}

/// Rewriting a column in place that no translating segment indexes is an
/// ordinary in-place update: the `i` segment is untouched and keeps
/// translating, and the patched value is what the scan reads.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_in_place_rewrite_of_an_unindexed_column() {
    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    rewrite_in_place(dataset, "v", 3, 333).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![10, 11],
        "the row was patched in place"
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 3").await,
        vec![(3, 333)]
    );
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// Rewriting in place a column that a translating segment indexes commits.
/// The segment cannot withdraw the rewritten destination from its bitmap
/// (it holds the retired sources), so the whole transition's sources are
/// withdrawn: `v_idx` claims nothing, the new value is answered by the
/// scan, and `i_idx`, which does not index `v`, is untouched.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_withdraws_a_translated_index_after_an_in_place_rewrite() {
    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    // Built before the rewrite: `v_idx` translates like `i_idx`. Rebuild
    // the history so the fixture's assumptions (two destinations) hold.
    let mut untagged = dataset.checkout_version(2).await.unwrap();
    untagged.restore().await.unwrap();
    create_v_index(&mut untagged).await;
    let dataset = make_tagged(untagged).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();
    let v_before = before
        .iter()
        .find(|idx| idx.name == "v_idx")
        .unwrap()
        .clone();
    assert_eq!(
        v_before.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1])
    );

    rewrite_in_place(dataset, "v", 3, 333).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    let v_after = stored.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(v_after.uuid, v_before.uuid, "the segment stays listed");
    assert!(
        v_after.fragment_bitmap.as_ref().unwrap().is_empty(),
        "the transition's sources were withdrawn: {:?}",
        v_after.fragment_bitmap
    );
    let derived = dataset.load_indices().await.unwrap();
    assert!(
        derived
            .iter()
            .find(|idx| idx.name == "v_idx")
            .and_then(|idx| idx.fragment_bitmap.as_ref())
            .is_none_or(|bitmap| bitmap.is_empty()),
        "v_idx claims nothing"
    );
    assert_eq!(rows(&dataset, Some("v = 333"), true).await, vec![(3, 333)]);
    assert_eq!(rows(&dataset, Some("v = 3"), true).await, vec![]);
    assert_eq!(
        rows(&dataset, None, true).await,
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| (i, if i == 3 { 333 } else { i }))
    );
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// The withdrawal is exact when the history can be walked, and the commit
/// path decodes an external history for that: with two stable partitions
/// (F0, F1 into F10, F11 and F2, F3 into F20, F21) and `v_idx` built over
/// all four sources, rewriting `v` on F11 withdraws {0, 1} only; the segment
/// keeps translating to F20, F21.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_withdraws_exactly_through_an_external_history() {
    use lance_table::format::pb::fragment_reuse_index_details::InlineContent;

    let dir = TempStrDir::default();
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col("v", lance_datagen::array::step::<Int32Type>())
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(
            dir.as_str(),
            FragmentCount::from(4),
            FragmentRowCount::from(4),
        )
        .await
        .unwrap();
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
    create_v_index(&mut dataset).await;
    reserve(&mut dataset, 40).await;
    let (first, mut destinations) = prepare_partition(&dataset, &[0, 1], 10).await;
    let (second, more) = prepare_partition(&dataset, &[2, 3], 20).await;
    destinations.extend(more);
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![first, second],
    }
    .encode_to_vec();
    crate::index::frag_reuse_reader::tests::install(&mut dataset, content, destinations, true)
        .await;
    let indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    persist_fixture(&mut dataset, indices).await;
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![10, 11, 20, 21]
    );
    assert_eq!(rows(&dataset, Some("v = 3"), true).await, vec![(3, 3)]);
    assert_eq!(rows(&dataset, Some("v = 13"), true).await, vec![(13, 13)]);

    // Row i = 3 lives in F11 (odd values of the first partition).
    rewrite_in_place(dataset, "v", 3, 333).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    let v_idx = stored.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(
        v_idx.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([2u32, 3]),
        "only the first partition's sources were withdrawn"
    );
    let derived = dataset.load_indices().await.unwrap();
    assert_eq!(
        derived
            .iter()
            .find(|idx| idx.name == "v_idx")
            .and_then(|idx| idx.fragment_bitmap.clone()),
        Some(RoaringBitmap::from_iter([20u32, 21])),
        "still translating to the second partition"
    );
    assert_eq!(rows(&dataset, Some("v = 333"), true).await, vec![(3, 333)]);
    assert_eq!(rows(&dataset, Some("v = 13"), true).await, vec![(13, 13)]);
    assert_eq!(
        rows(&dataset, None, true).await,
        (0..16)
            .map(|i| (i, if i == 3 { 333 } else { i }))
            .collect::<Vec<_>>()
    );
    let i_idx = stored.iter().find(|idx| idx.name == "i_idx").unwrap();
    assert_eq!(
        i_idx.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([0u32, 1, 2, 3]),
        "an index that does not index v is untouched"
    );
}

/// A history the commit path cannot decode fails the commit rather than
/// guessing at what to withdraw.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_with_corrupt_history_refuses_the_commit() {
    let dir = TempStrDir::default();
    let mut dataset = tagged_two_column_fixture(dir.as_str()).await;
    let mut indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    let entry = indices
        .iter_mut()
        .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
        .unwrap();
    entry.index_details = Some(Arc::new(prost_types::Any {
        type_url: "/lance.table.FragmentReuseIndexDetails".into(),
        value: vec![0x0a, 0x03, 0xff, 0xff],
    }));
    persist_fixture(&mut dataset, indices).await;
    let dataset = fresh_session(dir.as_str()).await;
    let version = dataset.manifest.version;

    let error = rewrite_in_place(dataset, "v", 3, 333).await.unwrap_err();
    assert!(error.to_string().contains("FRI details"), "{error}");
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, version, "nothing was committed");
}

/// A segment built after the rewrite names the destinations directly, so an
/// in-place rewrite of its column is withdrawn from its bitmap as on an
/// untagged table: the patched fragment leaves the `v` index and is
/// scanned, the other destination is still served.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_prunes_in_place_rewrite_from_a_direct_index() {
    let dir = TempStrDir::default();
    let mut dataset = tagged_two_column_fixture(dir.as_str()).await;
    create_v_index(&mut dataset).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();
    let v_before = before.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(
        v_before.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11])
    );

    rewrite_in_place(dataset, "v", 333).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    let v_after = stored.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(v_after.uuid, v_before.uuid);
    assert_eq!(
        v_after.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32]),
        "the patched destination (odd values, i = 3) left the v index"
    );
    assert_eq!(rows(&dataset, Some("v = 333"), true).await, vec![(3, 333)]);
    assert_eq!(rows(&dataset, Some("v = 3"), true).await, vec![]);
    assert_eq!(rows(&dataset, Some("v = 4"), true).await, vec![(4, 4)]);
    assert_eq!(
        rows(&dataset, None, true).await,
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| (i, if i == 3 { 333 } else { i }))
    );
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// Sorted `i` values under a predicate, with or without the scalar index;
/// for tables whose `v` is no longer Int32.
async fn i_values(dataset: &Dataset, predicate: Option<&str>, use_index: bool) -> Vec<i32> {
    let mut scan = dataset.scan();
    if let Some(predicate) = predicate {
        scan.filter(predicate).unwrap();
    }
    scan.use_scalar_index(use_index);
    scan.project(&["i"]).unwrap();
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

/// Adding a column writes new data files and rewrites none; dropping an
/// unindexed column projects the schema. Both keep the `i` segment
/// translating.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_add_and_drop_columns() {
    let dir = TempStrDir::default();
    let mut dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    dataset
        .add_columns(
            crate::dataset::NewColumnTransform::SqlExpressions(vec![("z".into(), "i * 2".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    dataset.drop_columns(&["w"]).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        dataset
            .schema()
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        vec!["i", "v", "z"]
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 6").await,
        vec![(6, 6)]
    );
    assert_eq!(i_values(&dataset, Some("z = 12"), true).await, vec![6]);
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// Casting an unindexed column rewrites its data file in place (a `Merge`
/// carrying the new file): admitted, the `i` segment still translates. (A
/// cast of an indexed column is refused by schema evolution itself, before
/// any commit, so the merge arm of the in-place rule is pinned at the
/// manifest level.)
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_alter_columns_follows_the_in_place_rewrite_rule() {
    use crate::dataset::ColumnAlteration;
    use arrow_schema::DataType;

    let dir = TempStrDir::default();
    let mut dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    dataset
        .alter_columns(&[ColumnAlteration::new("v".into()).cast_to(DataType::Int64)])
        .await
        .unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        dataset.schema().field("v").unwrap().data_type(),
        DataType::Int64
    );
    assert_eq!(i_values(&dataset, Some("i = 6"), true).await, vec![6]);
    assert_eq!(
        i_values(&dataset, Some("i = 6"), true).await,
        i_values(&dataset, Some("i = 6"), false).await
    );
    assert_eq!(i_values(&dataset, Some("v = 5"), false).await, vec![5]);
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// An overwrite replaces the table: every index, the tagged entry included,
/// is gone with the old fragments, and the sticky flag keeps the tagged
/// record form for the rewrites to come.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_overwrite() {
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};

    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    let schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![100, 101])),
            Arc::new(Int32Array::from(vec![100, 101])),
            Arc::new(Int32Array::from(vec![100, 101])),
        ],
    )
    .unwrap();
    InsertBuilder::new(Arc::new(dataset))
        .with_params(&WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap();

    let dataset = fresh_session(dir.as_str()).await;
    assert!(
        crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .is_empty()
    );
    assert_ne!(
        dataset.manifest.reader_feature_flags
            & lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX,
        0,
        "the tagged record form is sticky"
    );
    assert_eq!(
        rows(&dataset, None, true).await,
        vec![(100, 100), (101, 101)]
    );
}

/// Restore copies a whole earlier manifest: back to the untagged version the
/// index names live fragments directly, and forward to the tagged version
/// it translates again through the row maps, which no cleanup removed.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_restore() {
    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    let tagged_version = dataset.manifest.version;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    // Version 2 is the indexed, untagged table (1 = data, 2 = i_idx).
    let mut untagged = dataset.checkout_version(2).await.unwrap();
    untagged.restore().await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, tagged_version + 1);
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![0, 1]
    );
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    assert!(stored.iter().all(|idx| idx.name == "i_idx"), "{stored:?}");
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 3").await,
        vec![(3, 3)]
    );

    let mut tagged = dataset.checkout_version(tagged_version).await.unwrap();
    tagged.restore().await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, tagged_version + 2);
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 3").await,
        vec![(3, 3)]
    );
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// Stable row ids replace the row addresses a tagged history translates:
/// the migration is refused while the entry exists.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_refuses_stable_row_id_migration() {
    let dir = TempStrDir::default();
    let mut dataset = tagged_two_column_fixture(dir.as_str()).await;
    let version = dataset.manifest.version;
    let error = dataset.migrate_to_stable_row_ids().await.unwrap_err();
    assert!(
        error.to_string().contains("stable row id migration"),
        "{error}"
    );
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, version, "nothing was committed");
    assert!(!dataset.manifest.uses_stable_row_ids());
}

#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn tagged_table_accepts_config_update() {
    let dir = TempStrDir::default();
    let mut dataset = tagged_two_column_fixture(dir.as_str()).await;
    let before = crate::index::load_all_indices(&dataset).await.unwrap();

    dataset.update_config([("tenant", "blue")]).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        dataset.manifest.config.get("tenant").map(String::as_str),
        Some("blue")
    );
    assert_eq!(
        assert_index_agrees_with_scan(&dataset, "i = 6").await,
        vec![(6, 6)]
    );
    assert_segment_and_history_untouched(&dataset, &before, &[10, 11]).await;
}

/// Every commit attempt prepares the index list against the manifest it
/// builds on. A rewrite that read version V and lands after another writer
/// installed an index on the rewritten column at V+1 withdraws that index
/// too: the retry prepared from the list current at the retry, not from
/// what it saw at first.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn retry_withdraws_an_index_created_concurrently_on_the_rewritten_column() {
    let dir = TempStrDir::default();
    let stale = tagged_two_column_fixture(dir.as_str()).await;
    let read_version = stale.manifest.version;

    // Another writer indexes `v` on the live fragments meanwhile.
    let mut other = fresh_session(dir.as_str()).await;
    create_v_index(&mut other).await;
    assert_eq!(other.manifest.version, read_version + 1);
    let stored = crate::index::load_all_indices(&other).await.unwrap();
    let v_idx = stored.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(
        v_idx.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32, 11])
    );

    // Row i = 3 lives in F11; the rewrite commits from the stale handle and
    // is rebased over the index creation.
    rewrite_in_place(stale, "v", 3, 333).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(dataset.manifest.version, read_version + 2);
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    let v_idx = stored.iter().find(|idx| idx.name == "v_idx").unwrap();
    assert_eq!(
        v_idx.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([10u32]),
        "the concurrently created index lost the rewritten fragment"
    );
    assert_eq!(rows(&dataset, Some("v = 333"), true).await, vec![(3, 333)]);
    assert_eq!(rows(&dataset, Some("v = 3"), true).await, vec![]);
    assert_eq!(
        rows(&dataset, None, true).await,
        rows(&dataset, None, false).await
    );
    assert_segment_and_history_untouched(
        &dataset,
        &crate::index::load_all_indices(&other).await.unwrap(),
        &[10, 11],
    )
    .await;
}

/// A detached commit prepares the index list the same way as the main
/// chain: an in-place rewrite of an indexed column withdraws the coverage
/// in the detached manifest it produces.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn detached_in_place_rewrite_on_a_tagged_table_withdraws_like_the_main_chain() {
    let dir = TempStrDir::default();
    let dataset = tagged_two_column_fixture(dir.as_str()).await;
    let base_version = dataset.manifest.version;
    let base = dataset.clone();

    // Commit the rewrite once on the main chain to obtain its operation
    // (the rewritten data files it points at stay on disk).
    rewrite_in_place(dataset, "i", 3, 333).await.unwrap();
    let committed = fresh_session(dir.as_str()).await;
    let transaction = committed
        .read_transaction()
        .await
        .unwrap()
        .expect("the rewrite recorded its transaction");
    assert!(matches!(transaction.operation, Operation::Update { .. }));

    // The same operation, committed detached on top of the base version.
    let detached = Dataset::commit_detached(
        Arc::new(base),
        transaction.operation,
        Some(base_version),
        None,
        None,
        Arc::new(Session::default()),
        false,
    )
    .await
    .unwrap();
    assert!(lance_table::format::is_detached_version(
        detached.manifest.version
    ));
    let stored = crate::index::load_all_indices(&detached).await.unwrap();
    let segment = user_segment(&stored);
    assert!(
        segment.fragment_bitmap.as_ref().unwrap().is_empty(),
        "the translating segment's coverage of the rewritten transition was withdrawn: {:?}",
        segment.fragment_bitmap
    );
    assert_eq!(fri_entry(&stored).index_version, 1);
    assert_eq!(i_values(&detached, Some("i = 333"), true).await, vec![333]);
    assert_eq!(
        i_values(&detached, None, true).await,
        i_values(&detached, None, false).await
    );
}

/// Sorted `(i, vector)` rows of a nearest-neighbour search over every row,
/// with or without the vector index: the two must agree on row identity,
/// multiplicity and the values read back.
async fn knn_rows(dataset: &Dataset, query: &[f32], use_index: bool) -> Vec<(i32, Vec<f32>)> {
    use arrow_array::types::Float32Type;
    let total = dataset.count_rows(None).await.unwrap();
    let mut scan = dataset.scan();
    scan.nearest("v", &arrow_array::Float32Array::from(query.to_vec()), total)
        .unwrap();
    scan.use_index(use_index);
    scan.project(&["i", "v"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let i = batch["i"].as_primitive::<Int32Type>();
    let v = batch["v"].as_fixed_size_list();
    let mut rows: Vec<(i32, Vec<f32>)> = (0..batch.num_rows())
        .map(|row| {
            (
                i.value(row),
                v.value(row).as_primitive::<Float32Type>().values().to_vec(),
            )
        })
        .collect();
    rows.sort_by_key(|row| row.0);
    assert_eq!(rows.len(), total, "every row is returned");
    rows
}

/// Rewrite the vector `v` of the row keyed `w = key` in place.
async fn rewrite_vector_in_place(dataset: Dataset, key: i32, value: [f32; 4]) -> Arc<Dataset> {
    use arrow_array::{FixedSizeListArray, Float32Array};
    use arrow_schema::{DataType, Field};
    let schema = Arc::new(ArrowSchema::from(
        &dataset.schema().project(&["w", "v"]).unwrap(),
    ));
    let vector = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        4,
        Arc::new(Float32Array::from(value.to_vec())),
        None,
    )
    .unwrap();
    let source = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![key])), Arc::new(vector)],
    )
    .unwrap();
    MergeInsertBuilder::try_new(Arc::new(dataset), vec!["w".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .map(|(dataset, _)| dataset)
        .unwrap()
}

/// A vector fixture: `frag_count` fragments of four rows with columns `i`,
/// `v` (a random 4-vector) and `w` (= `i`), and an exact IVF_FLAT index on
/// `v` over every fragment, built before any rewrite.
async fn vector_fixture(uri: &str, frag_count: u32) -> Dataset {
    use arrow_array::types::Float32Type;
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col("v", lance_datagen::array::rand_vec::<Float32Type>(4.into()))
        .col("w", lance_datagen::array::step::<Int32Type>())
        .into_dataset(
            uri,
            FragmentCount::from(frag_count),
            FragmentRowCount::from(4),
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["v"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &crate::index::vector::VectorIndexParams::ivf_flat(
                1,
                lance_linalg::distance::DistanceType::L2,
            ),
            false,
        )
        .await
        .unwrap();
    dataset
}

async fn vec_segment_bitmap(dataset: &Dataset) -> Option<RoaringBitmap> {
    let stored = crate::index::load_all_indices(dataset).await.unwrap();
    stored
        .iter()
        .find(|idx| idx.name == "vec_idx")
        .and_then(|idx| idx.fragment_bitmap.clone())
}

/// Vector twin of the exact withdrawal: with two stable partitions and the
/// index built over all four sources, rewriting `v` on F11 withdraws {0, 1}
/// and the segment keeps translating to F20, F21. The nearest-neighbour
/// search agrees with the index-disabled scan on every row, before and
/// after `optimize_indices`.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn vector_index_translating_remainder_agrees_with_the_scan() {
    use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
    let dir = TempStrDir::default();
    let mut dataset = vector_fixture(dir.as_str(), 4).await;
    reserve(&mut dataset, 40).await;
    let (first, mut destinations) = prepare_partition(&dataset, &[0, 1], 10).await;
    let (second, more) = prepare_partition(&dataset, &[2, 3], 20).await;
    destinations.extend(more);
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![first, second],
    }
    .encode_to_vec();
    crate::index::frag_reuse_reader::tests::install(&mut dataset, content, destinations, true)
        .await;
    let indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    persist_fixture(&mut dataset, indices).await;
    let dataset = fresh_session(dir.as_str()).await;
    let query = [0.25f32, 0.5, 0.75, 1.0];
    assert_eq!(
        knn_rows(&dataset, &query, true).await,
        knn_rows(&dataset, &query, false).await
    );

    // Row w = 3 lives in F11.
    rewrite_vector_in_place(dataset, 3, [9.0, 9.0, 9.0, 9.0]).await;
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        vec_segment_bitmap(&dataset).await,
        Some(RoaringBitmap::from_iter([2u32, 3])),
        "the first partition's sources were withdrawn, the second still translates"
    );
    let indexed = knn_rows(&dataset, &query, true).await;
    assert_eq!(indexed, knn_rows(&dataset, &query, false).await);
    assert_eq!(indexed[3], (3, vec![9.0, 9.0, 9.0, 9.0]));

    let mut dataset = fresh_session(dir.as_str()).await;
    dataset.optimize_indices(&Default::default()).await.unwrap();
    let dataset = fresh_session(dir.as_str()).await;
    let indexed = knn_rows(&dataset, &query, true).await;
    assert_eq!(indexed, knn_rows(&dataset, &query, false).await);
    assert_eq!(indexed[3], (3, vec![9.0, 9.0, 9.0, 9.0]));
    let stored = crate::index::load_all_indices(&dataset).await.unwrap();
    for segment in stored.iter().filter(|idx| idx.name == "vec_idx") {
        let bitmap = segment.fragment_bitmap.as_ref().unwrap();
        assert!(
            !bitmap.contains(0) && !bitmap.contains(1),
            "optimize must not restore the withdrawn provenance: {bitmap:?}"
        );
    }
}

/// The partial-withdrawal shape the reader would otherwise load as it is:
/// the index was built over the two sources of the only partition and a
/// third fragment the history never touched. Rewriting `v` on F11 withdraws
/// the sources, and the live-only remainder {2} goes with them (the file
/// still holds the withdrawn rows and would serve them raw), so the search
/// scans until `optimize_indices` rebuilds the segment. The search agrees
/// with the index-disabled scan at every step.
#[tokio::test]
#[serial_test::serial(frag_reuse_maintenance)]
async fn vector_index_live_only_remainder_is_withdrawn_whole_and_agrees_with_the_scan() {
    use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
    let dir = TempStrDir::default();
    let mut dataset = vector_fixture(dir.as_str(), 3).await;
    reserve(&mut dataset, 40).await;
    let untouched = dataset.fragments()[2].clone();
    let (transition, destinations) = prepare_partition(&dataset, &[0, 1], 10).await;
    let live: Vec<Fragment> = std::iter::once(untouched)
        .chain(destinations.iter().cloned())
        .collect();
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    crate::index::frag_reuse_reader::tests::install(&mut dataset, content, destinations, false)
        .await;
    // Fragment 2 is untouched by the partition and stays live.
    Arc::make_mut(&mut dataset.manifest).fragments = live.into();
    dataset.fragment_bitmap = Arc::new(
        dataset
            .manifest
            .fragments
            .iter()
            .map(|f| f.id as u32)
            .collect(),
    );
    let indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    persist_fixture(&mut dataset, indices).await;
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![2, 10, 11]
    );
    assert_eq!(
        vec_segment_bitmap(&dataset).await,
        Some(RoaringBitmap::from_iter([0u32, 1, 2]))
    );
    let query = [0.25f32, 0.5, 0.75, 1.0];
    assert_eq!(
        knn_rows(&dataset, &query, true).await,
        knn_rows(&dataset, &query, false).await
    );

    // Row w = 3 lives in F11.
    rewrite_vector_in_place(dataset, 3, [9.0, 9.0, 9.0, 9.0]).await;
    let dataset = fresh_session(dir.as_str()).await;
    assert_eq!(
        vec_segment_bitmap(&dataset).await,
        Some(RoaringBitmap::new()),
        "the live-only remainder went with the withdrawn sources"
    );
    let indexed = knn_rows(&dataset, &query, true).await;
    assert_eq!(indexed, knn_rows(&dataset, &query, false).await);
    assert_eq!(indexed[3], (3, vec![9.0, 9.0, 9.0, 9.0]));

    let plan = dataset
        .scan()
        .nearest("v", &arrow_array::Float32Array::from(query.to_vec()), 4)
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(
        !plan.contains("ANN"),
        "an emptied segment is not searched: {plan}"
    );
    // Rebuilding the emptied segment through `optimize_indices` is the
    // maintenance change's job (its `withdrawn_vector_index_is_rebuilt_by_optimize`).
}
