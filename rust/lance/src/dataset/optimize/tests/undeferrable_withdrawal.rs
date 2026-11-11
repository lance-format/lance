// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A deferred compaction on a tagged fragment reuse history leaves every
//! covering segment's retired provenance in place for the reader to translate.
//! A segment whose type cannot be translated in batches
//! (`segment_supports_batch_remapping` answers `Some(false)`, today a legacy
//! pre-(0,3) vector index file) would be left addressing provenance it can
//! never translate, so the Rewrite commit withdraws its coverage of the
//! rewritten sources in the same manifest version.
//!
//! The fixture is one table with both kinds of segment: a legacy-format
//! IVF_PQ index (undeferrable) and a BTree index (deferrable) over the same
//! fragments, on a table made tagged by a stable-partition commit before the
//! indices exist.

use super::*;
use crate::dataset::CommitBuilder;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::optimize::commit::commit_compaction;
use crate::index::frag_reuse::frag_reuse_entry_appending;
use crate::index::frag_reuse_reader::tests::prepare_partition;
use crate::index::vector::IndexFileVersion;
use lance_io::object_store::ObjectStoreParams;
use lance_table::system_index::frag_reuse::metadata::uses_tagged_fri;
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use uuid::Uuid;

const INDEX_ROWS_PER_FRAGMENT: u32 = 64;
const VECTOR_INDEX: &str = "vector_idx";
const BTREE_INDEX: &str = "i_idx";

/// Four fragments of `i` plus a 4-dimensional vector column.
async fn fixture() -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "vector",
            lance_datagen::array::rand_vec::<Float32Type>(4.into()),
        )
        .into_ram_dataset(
            FragmentCount::from(4),
            FragmentRowCount::from(INDEX_ROWS_PER_FRAGMENT),
        )
        .await
        .unwrap()
}

/// Makes `dataset` tagged by committing a stable partition of its first two
/// fragments, then indexes `i` (BTree, deferrable) and `vector` (a legacy
/// pre-(0,3) IVF_PQ file, undeferrable) over everything that is then live.
///
/// The partition commits BEFORE the indices exist, so no segment's coverage
/// depends on it: the indices are built over the destinations and the
/// withdrawal under test is the DIRECT case, sources still live in the
/// manifest it is computed against.
async fn tagged_with_both_index_kinds() -> Dataset {
    let mut dataset = make_tagged(fixture().await).await;
    create_legacy_vector_index(&mut dataset).await;
    create_btree_index(&mut dataset).await;
    dataset
}

/// A legacy (pre-(0,3)) IVF_PQ file: the one index kind
/// `segment_supports_batch_remapping` reports as undeferrable at this pin.
async fn create_legacy_vector_index(dataset: &mut Dataset) {
    let mut params =
        VectorIndexParams::ivf_pq(1, 4, 2, lance_linalg::distance::DistanceType::L2, 10);
    params.version(IndexFileVersion::Legacy);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some(VECTOR_INDEX.into()),
            &params,
            true,
        )
        .await
        .unwrap();
}

async fn create_btree_index(dataset: &mut Dataset) {
    dataset
        .create_index(
            &["i"],
            IndexType::Scalar,
            Some(BTREE_INDEX.into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
}

/// The stored (not reader-derived) metadata of every non-system segment, by
/// name: uuid and declared fragment coverage. Read straight from the manifest
/// so a withdrawal is observed as the commit wrote it.
async fn stored_segments(dataset: &Dataset) -> Vec<(String, Uuid, RoaringBitmap)> {
    load_all_indices(dataset)
        .await
        .unwrap()
        .iter()
        .filter(|index| !is_system_index(index))
        .map(|index| {
            (
                index.name.clone(),
                index.uuid,
                index.fragment_bitmap.clone().unwrap_or_default(),
            )
        })
        .collect()
}

fn coverage_of(segments: &[(String, Uuid, RoaringBitmap)], name: &str) -> RoaringBitmap {
    segments
        .iter()
        .filter(|(index_name, _, _)| index_name == name)
        .fold(RoaringBitmap::new(), |coverage, (_, _, bitmap)| {
            coverage | bitmap
        })
}

/// The coverage the READER derives for `name` after translating the history.
async fn derived_coverage(dataset: &Dataset, name: &str) -> RoaringBitmap {
    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|index| index.name == name)
        .fold(RoaringBitmap::new(), |coverage, index| {
            coverage | index.fragment_bitmap.clone().unwrap_or_default()
        })
}

fn deferred_options(target_rows_per_fragment: usize) -> CompactionOptions {
    CompactionOptions {
        target_rows_per_fragment,
        defer_index_remap: true,
        ..Default::default()
    }
}

/// Runs the plan's tasks, keeping only the positions in `keep` (all of them
/// when `keep` is `None`): the omitted tasks are the "failed" ones, never
/// executed and never committed.
async fn execute_tasks(
    dataset: &Dataset,
    plan: &CompactionPlan,
    options: &CompactionOptions,
    keep: Option<&[usize]>,
) -> Vec<RewriteResult> {
    let mut results = Vec::new();
    for (position, task) in plan.tasks().iter().enumerate() {
        if keep.is_some_and(|keep| !keep.contains(&position)) {
            continue;
        }
        results.push(
            rewrite_files(Cow::Borrowed(dataset), task.clone(), options)
                .await
                .unwrap(),
        );
    }
    results
}

/// Plans a deferred compaction and commits `keep` of its tasks (all of them
/// when `keep` is `None`), returning how many were committed.
async fn run_deferred_compaction(
    dataset: &mut Dataset,
    options: &CompactionOptions,
    keep: Option<&[usize]>,
) -> usize {
    let plan = plan_compaction(dataset, options).await.unwrap();
    let results = execute_tasks(dataset, &plan, options, keep).await;
    let committed = results.len();
    if committed > 0 {
        commit_compaction(
            dataset,
            results,
            Arc::new(DatasetIndexRemapperOptions::default()),
            options,
        )
        .await
        .unwrap();
    }
    committed
}

/// Asserts a KNN answer against brute force, tolerating ties at the cutoff:
/// `k` rows come back, none of them further than the k-th smallest
/// brute-force distance, and every row strictly closer than that cutoff is
/// present. Which of several equidistant rows fills the last slot is not
/// pinned by anything, so it is not asserted.
async fn assert_knn_matches_brute_force(
    dataset: &Dataset,
    query: &Float32Array,
    k: usize,
    rows: &[i32],
    plan: &str,
) {
    let scored = brute_force_distances(dataset, query).await;
    assert_eq!(rows.len(), k, "a KNN query must return k rows: {plan}");
    let cutoff = scored[k - 1].0;
    let by_row: HashMap<i32, f32> = scored.iter().map(|(d, i)| (*i, *d)).collect();
    for row in rows {
        let distance = by_row[row];
        assert!(
            distance <= cutoff,
            "row {row} at distance {distance} is beyond the k-th brute-force \
             distance {cutoff}: {plan}"
        );
    }
    for (distance, row) in &scored {
        if *distance < cutoff {
            assert!(
                rows.contains(row),
                "row {row} at distance {distance} is strictly closer than the \
                 cutoff {cutoff} and must be returned: {plan}"
            );
        }
    }
}

/// Every live row's squared distance to `query`, ascending, computed without
/// any index.
async fn brute_force_distances(dataset: &Dataset, query: &Float32Array) -> Vec<(f32, i32)> {
    let batch = dataset
        .scan()
        .project(&["i", "vector"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let values = batch["i"].as_primitive::<Int32Type>();
    let vectors = batch["vector"].as_fixed_size_list();
    let mut scored: Vec<(f32, i32)> = (0..batch.num_rows())
        .map(|row| {
            let vector = vectors.value(row);
            let vector = vector.as_primitive::<Float32Type>();
            let distance: f32 = query
                .values()
                .iter()
                .zip(vector.values().iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (distance, values.value(row))
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored
}

/// A query vector taken from a row that is live after the compaction.
async fn query_vector(dataset: &Dataset, i: i32) -> Float32Array {
    let batch = dataset
        .scan()
        .filter(&format!("i = {i}"))
        .unwrap()
        .project(&["vector"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 1, "row {i} is not live");
    batch["vector"]
        .as_fixed_size_list()
        .value(0)
        .as_primitive::<Float32Type>()
        .clone()
}

/// The KNN answer the planner produces for `dataset`, with the plan text so a
/// test can assert which path served it.
async fn indexed_knn(dataset: &Dataset, query: &Float32Array, k: usize) -> (Vec<i32>, String) {
    let mut scan = dataset.scan();
    scan.nearest("vector", query, k).unwrap();
    scan.project(&["i"]).unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut rows: Vec<i32> = batch["i"]
        .as_primitive::<Int32Type>()
        .iter()
        .map(|value| value.unwrap())
        .collect();
    rows.sort_unstable();
    (rows, plan)
}

/// The whole contract on one commit: in the SAME committed manifest version
/// the undeferrable segment's bitmap no longer names the rewritten sources
/// (nor the destinations), the deferrable segment still translates to the
/// destinations through the history, both index definitions remain, and no
/// other segment changed. The queries afterwards: KNN on the withdrawn
/// column equals brute force without any retired or duplicated row, and the
/// BTree lookup stays correct.
#[tokio::test]
async fn deferred_compaction_withdraws_the_undeferrable_segments_coverage() {
    let mut dataset = tagged_with_both_index_kinds().await;
    let before = stored_segments(&dataset).await;
    let vector_before = coverage_of(&before, VECTOR_INDEX);
    let btree_before = coverage_of(&before, BTREE_INDEX);
    assert_eq!(
        vector_before, btree_before,
        "the fixture indexes the same fragments with both kinds"
    );
    let version_before = dataset.manifest.version;

    // Compact everything live into one fragment.
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    let sources: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let committed = run_deferred_compaction(&mut dataset, &options, None).await;
    assert!(committed > 0, "the fixture must have something to compact");
    // The withdrawal rides the rewrite commit: the Rewrite is the LAST
    // version this compaction wrote (the fragment-id reservation before it is
    // the pre-existing address-style step), and the narrowed bitmap is already
    // in it.
    assert!(dataset.manifest.version > version_before);
    assert!(matches!(
        dataset
            .read_transaction()
            .await
            .unwrap()
            .map(|transaction| transaction.operation),
        Some(Operation::Rewrite { .. })
    ));
    let destinations: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();

    let after = stored_segments(&dataset).await;
    // The undeferrable segment gave up the rewritten sources and claims no
    // destination either.
    let vector_after = coverage_of(&after, VECTOR_INDEX);
    assert!(
        vector_after.is_disjoint(&sources),
        "the legacy vector segment still covers rewritten sources {:?}",
        (&vector_after & &sources).iter().collect::<Vec<_>>()
    );
    assert!(
        vector_after.is_disjoint(&destinations),
        "a withdrawal never claims the rewrite's outputs: {:?}",
        (&vector_after & &destinations).iter().collect::<Vec<_>>()
    );
    // The deferrable segment kept its provenance and translates to the
    // destinations.
    assert_eq!(
        coverage_of(&after, BTREE_INDEX),
        btree_before,
        "a deferrable segment's stored provenance is untouched"
    );
    assert_eq!(
        derived_coverage(&dataset, BTREE_INDEX).await,
        destinations,
        "the BTree's coverage must translate to the rewrite's outputs"
    );
    assert!(
        derived_coverage(&dataset, VECTOR_INDEX).await.is_empty(),
        "the withdrawn segment must derive no coverage"
    );
    // Both definitions remain, so a later optimize can restore coverage.
    for name in [VECTOR_INDEX, BTREE_INDEX] {
        assert!(
            after.iter().any(|(index_name, _, _)| index_name == name),
            "index {name} must keep its definition"
        );
    }
    // No other segment changed: every uuid present before is still present
    // with its coverage, except the withdrawn vector segments.
    let withdrawn: Vec<Uuid> = before
        .iter()
        .filter(|(name, _, _)| name == VECTOR_INDEX)
        .map(|(_, uuid, _)| *uuid)
        .collect();
    for (name, uuid, bitmap) in &before {
        if withdrawn.contains(uuid) {
            continue;
        }
        assert!(
            after
                .iter()
                .any(|(_, after_uuid, after_bitmap)| after_uuid == uuid && after_bitmap == bitmap),
            "segment {uuid} of index {name} changed without being withdrawn"
        );
    }

    // Queries: the withdrawn column answers exactly, by scanning. Asserted
    // positively on the flat node the fallback produces, with the
    // substring check on the index node kept as a secondary guard.
    let query = query_vector(&dataset, 7).await;
    let (rows, plan) = indexed_knn(&dataset, &query, 5).await;
    assert!(
        plan.contains("KNNVectorDistance"),
        "a withdrawn segment must be answered by the flat distance scan: \
         {plan}"
    );
    assert!(
        !plan.contains("ANN"),
        "a withdrawn segment must not serve an ANN query: {plan}"
    );
    assert!(
        derived_coverage(&dataset, VECTOR_INDEX).await.is_empty(),
        "the withdrawn segment's effective coverage must be empty"
    );
    assert_knn_matches_brute_force(&dataset, &query, 5, &rows, &plan).await;
    // And the BTree lookup stays correct and indexed.
    let mut lookup = dataset.scan();
    lookup.filter("i = 7").unwrap().project(&["i"]).unwrap();
    let lookup_plan = lookup.explain_plan(false).await.unwrap();
    assert!(lookup_plan.contains("ScalarIndexQuery"), "{lookup_plan}");
    let batch = lookup.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch["i"].as_primitive::<Int32Type>().value(0), 7);
}

/// A deferred compaction of fragments the undeferrable index does not cover
/// withdraws nothing: every index's metadata is byte-for-byte what it was.
#[tokio::test]
async fn a_compaction_off_the_undeferrable_coverage_withdraws_nothing() {
    let mut dataset = tagged_with_both_index_kinds().await;
    // Append two fragments AFTER the indices were built, so no segment covers
    // them, then compact only those.
    let batch = {
        let mut scan = dataset.scan();
        scan.project(&["i", "vector"]).unwrap();
        scan.try_into_batch().await.unwrap()
    };
    let appended = RecordBatch::try_new(
        batch.schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(
                10_000..10_000 + batch.num_rows() as i32,
            )) as ArrayRef,
            batch["vector"].clone(),
        ],
    )
    .unwrap();
    let schema = appended.schema();
    dataset
        .append(
            RecordBatchIterator::new([Ok(appended)], schema),
            Some(WriteParams {
                max_rows_per_file: INDEX_ROWS_PER_FRAGMENT as usize,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let indexed: RoaringBitmap = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .filter(|index| !is_system_index(index))
        .filter_map(|index| index.fragment_bitmap.clone())
        .fold(RoaringBitmap::new(), |covered, bitmap| covered | bitmap);
    let unindexed: Vec<u32> = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .filter(|id| !indexed.contains(*id))
        .collect();
    assert!(
        unindexed.len() >= 2,
        "the appended fragments must be unindexed"
    );

    let before = stored_segments(&dataset).await;
    // Hold every covered fragment out of the plan, so the compaction touches
    // only the uncovered appended ones.
    let options = CompactionOptions {
        excluded_fragment_ids: indexed.iter().collect(),
        ..deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize)
    };
    let committed = run_deferred_compaction(&mut dataset, &options, None).await;
    assert!(
        committed > 0,
        "the appended fragments must actually be compacted"
    );

    assert_eq!(
        stored_segments(&dataset).await,
        before,
        "a compaction touching no covered fragment must not change any index metadata"
    );
}

/// Committing only part of the plan withdraws only the committed tasks'
/// sources; the omitted task's sources stay covered. The task that was never
/// passed to the commit is the "failed" case: its fragments keep their
/// coverage because nothing retired them.
#[tokio::test]
async fn a_partial_commit_withdraws_only_the_committed_tasks_sources() {
    let mut dataset = tagged_with_both_index_kinds().await;
    let before = stored_segments(&dataset).await;
    let vector_before = coverage_of(&before, VECTOR_INDEX);

    // One task per pair of fragments, so a subset is meaningful.
    let options = deferred_options(2 * INDEX_ROWS_PER_FRAGMENT as usize);
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert!(
        plan.tasks().len() >= 2,
        "need at least two tasks to commit a subset, got {}",
        plan.tasks().len()
    );
    let committed_sources: RoaringBitmap = plan.tasks()[0]
        .fragments
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let omitted_sources: RoaringBitmap = plan.tasks()[1..]
        .iter()
        .flat_map(|task| task.fragments.iter())
        .map(|fragment| fragment.id as u32)
        .collect();
    assert!(!committed_sources.is_empty() && !omitted_sources.is_empty());

    let results = execute_tasks(&dataset, &plan, &options, Some(&[0])).await;
    assert_eq!(results.len(), 1);
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    let vector_after = coverage_of(&stored_segments(&dataset).await, VECTOR_INDEX);
    assert!(
        vector_after.is_disjoint(&committed_sources),
        "the committed task's sources must be withdrawn"
    );
    assert_eq!(
        &vector_after & &omitted_sources,
        &vector_before & &omitted_sources,
        "an omitted (failed) task's sources keep their coverage"
    );
    assert!(
        !(&vector_after & &omitted_sources).is_empty(),
        "the omitted task's sources must still be covered"
    );
}

/// The ordinary eager default is unchanged: `defer_index_remap = false` on
/// this table refuses up front (the tagged history requires the deferred
/// path), without withdrawing anything.
#[tokio::test]
async fn the_eager_default_is_unchanged_and_withdraws_nothing() {
    let mut dataset = tagged_with_both_index_kinds().await;
    let before = stored_segments(&dataset).await;
    let options = CompactionOptions {
        target_rows_per_fragment: 4 * INDEX_ROWS_PER_FRAGMENT as usize,
        ..Default::default()
    };
    let error = compact_files(&mut dataset, options, None)
        .await
        .expect_err("an eager compaction of a tagged history must be refused");
    assert!(error.to_string().contains("defer_index_remap"), "{error}");
    assert_eq!(
        stored_segments(&dataset).await,
        before,
        "a refused compaction must not withdraw coverage"
    );
}

/// `None` from the capability predicate (details absent or uninterpretable)
/// is NOT turned into a withdrawal: the commit leaves such a segment's
/// coverage exactly as it was, and the reader's own refusal (no derived
/// coverage, scan fallback) remains the behaviour. Pins the current refusal.
#[tokio::test]
async fn unknown_index_details_are_refused_rather_than_withdrawn() {
    let mut dataset = tagged_with_both_index_kinds().await;
    // Strip the BTree segment's details in the manifest, which is the
    // `None` state: `segment_supports_batch_remapping` cannot classify it.
    let stripped: Vec<IndexMetadata> = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .map(|index| {
            let mut index = index.clone();
            if index.name == BTREE_INDEX {
                index.index_details = None;
            }
            index
        })
        .collect();
    let btree = stripped
        .iter()
        .find(|index| index.name == BTREE_INDEX)
        .unwrap()
        .clone();
    let coverage_before = btree.fragment_bitmap.clone().unwrap();
    let read_version = dataset.manifest.version;
    let removed_indices: Vec<IndexMetadata> = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .filter(|index| index.name == BTREE_INDEX)
        .cloned()
        .collect();
    dataset = CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::CreateIndex {
                new_indices: vec![btree.clone()],
                removed_indices,
            },
            None,
        ))
        .await
        .unwrap();
    assert!(
        load_all_indices(&dataset)
            .await
            .unwrap()
            .iter()
            .any(|index| index.name == BTREE_INDEX && index.index_details.is_none()),
        "the fixture must leave the BTree segment without details"
    );

    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    run_deferred_compaction(&mut dataset, &options, None).await;

    let after = coverage_of(&stored_segments(&dataset).await, BTREE_INDEX);
    assert_eq!(
        after, coverage_before,
        "an unclassifiable segment's coverage is left alone, not withdrawn"
    );
    // The reader refuses it instead: no derived coverage, so queries scan.
    assert!(
        derived_coverage(&dataset, BTREE_INDEX).await.is_empty(),
        "the reader must derive no coverage for an unclassifiable segment"
    );
}

/// Concurrency, order (i): a CreateIndex of an undeferrable (legacy vector)
/// index commits FIRST, then a deferred Rewrite planned before that index
/// existed is rebased over it. The rebased Rewrite must withdraw the NEW
/// index's coverage of its own sources: the classification is redone against
/// the manifest the retrying attempt builds on, not against the snapshot the
/// compaction was planned from.
#[tokio::test]
async fn a_rewrite_rebased_over_a_committed_undeferrable_index_withdraws_its_coverage() {
    // Tagged, with only the BTree in place when the compaction is planned.
    let mut dataset = fixture().await;
    dataset = make_tagged(dataset).await;
    create_btree_index(&mut dataset).await;

    // Plan and execute the compaction at this version, before the vector
    // index exists.
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let sources: RoaringBitmap = plan
        .tasks()
        .iter()
        .flat_map(|task| task.fragments.iter())
        .map(|fragment| fragment.id as u32)
        .collect();
    assert!(!sources.is_empty());
    let results = execute_tasks(&dataset, &plan, &options, None).await;

    // A concurrent writer commits the legacy vector index over exactly those
    // fragments, so the rebased rewrite meets a segment it cannot translate.
    let mut concurrent = dataset.clone();
    create_legacy_vector_index(&mut concurrent).await;
    let vector_before = coverage_of(&stored_segments(&concurrent).await, VECTOR_INDEX);
    assert!(
        !(&vector_before & &sources).is_empty(),
        "the concurrent index must cover the rewrite's sources"
    );

    // The compaction handle is still at the pre-index version; the commit
    // path rebases it over the CreateIndex.
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    let after = stored_segments(&dataset).await;
    let vector_after = coverage_of(&after, VECTOR_INDEX);
    assert!(
        after.iter().any(|(name, _, _)| name == VECTOR_INDEX),
        "the concurrently created index must survive the rebase"
    );
    assert!(
        vector_after.is_disjoint(&sources),
        "the rebased rewrite must withdraw the new index's coverage of its sources: {:?}",
        (&vector_after & &sources).iter().collect::<Vec<_>>()
    );
    assert!(
        derived_coverage(&dataset, VECTOR_INDEX).await.is_empty(),
        "the withdrawn segment must derive no coverage"
    );
    // The deferrable index is unaffected and still translates.
    let destinations: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        derived_coverage(&dataset, BTREE_INDEX).await,
        destinations,
        "the BTree keeps translating through the history"
    );
    assert_query_exactness(&dataset).await;
}

/// Concurrency, order (ii): the deferred Rewrite commits FIRST, then a
/// CreateIndex of an undeferrable (legacy vector) index built at the
/// pre-rewrite version is rebased over the committed rewrite. The new segment
/// must not land keeping coverage of the sources the rewrite retired, since
/// it can never translate them.
#[tokio::test]
async fn an_undeferrable_index_rebased_over_a_committed_rewrite_drops_retired_coverage() {
    let mut dataset = fixture().await;
    dataset = make_tagged(dataset).await;
    create_btree_index(&mut dataset).await;

    // The index build reads this version and keeps its handle.
    let mut stale = dataset.clone();
    let pre_rewrite_fragments: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let segment = {
        let mut params =
            VectorIndexParams::ivf_pq(1, 4, 2, lance_linalg::distance::DistanceType::L2, 10);
        params.version(IndexFileVersion::Legacy);
        CreateIndexBuilder::new(&mut stale, &["vector"], IndexType::Vector, &params)
            .name(VECTOR_INDEX.to_string())
            .execute_uncommitted()
            .await
            .unwrap()
    };
    assert_eq!(
        segment.fragment_bitmap.as_ref().unwrap(),
        &pre_rewrite_fragments,
        "the staged segment must cover the pre-rewrite fragments"
    );

    // The compaction commits first, retiring every source.
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    run_deferred_compaction(&mut dataset, &options, None).await;
    let live: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert!(
        live.is_disjoint(&pre_rewrite_fragments),
        "the compaction must retire every source"
    );

    // Now the stale handle commits its segment; the resolver rebases it.
    let committed = CommitBuilder::new(Arc::new(stale))
        .execute(Transaction::new(
            pre_rewrite_version(&segment),
            Operation::CreateIndex {
                new_indices: vec![segment.clone()],
                removed_indices: vec![],
            },
            None,
        ))
        .await;
    // The rebase must LAND, not refuse: the whole point is that the segment
    // keeps its live coverage and gives up only what it cannot translate. A
    // refusal here would make every assertion below vacuous, so it fails the
    // test rather than being tolerated.
    let committed = committed.expect("the rebased index build must land");
    let landed = stored_segments(&committed).await;
    assert!(
        landed.iter().any(|(name, _, _)| name == VECTOR_INDEX),
        "the index definition must land so a later optimize can rebuild it"
    );
    let vector_landed = coverage_of(&landed, VECTOR_INDEX);
    assert!(
        vector_landed.is_disjoint(&pre_rewrite_fragments),
        "the rebased index must not keep coverage of retired sources it cannot translate: {:?}",
        (&vector_landed & &pre_rewrite_fragments)
            .iter()
            .collect::<Vec<_>>()
    );
    assert!(
        derived_coverage(&committed, VECTOR_INDEX).await.is_empty(),
        "a segment that kept nothing translatable must derive no coverage"
    );
    assert_query_exactness(&committed).await;
}

/// A whole-fragment DELETE is not a rewrite: the fragment's rows are gone,
/// not moved, so an undeferrable segment rebased over it keeps every bit of
/// coverage it can still serve. Only a fragment the history records as
/// consumed is withdrawn.
///
/// Without the provenance constraint the deleted fragment enters the
/// retired-provenance path, the segment is selected, and the live-only
/// remainder rule then empties its whole coverage.
#[tokio::test]
async fn an_undeferrable_index_rebased_over_a_committed_delete_keeps_its_coverage() {
    let mut dataset = fixture().await;
    dataset = make_tagged(dataset).await;

    // The index build reads this version and keeps its handle; its coverage
    // names every fragment live here.
    let mut stale = dataset.clone();
    let covered: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let segment = {
        let mut params =
            VectorIndexParams::ivf_pq(1, 4, 2, lance_linalg::distance::DistanceType::L2, 10);
        params.version(IndexFileVersion::Legacy);
        CreateIndexBuilder::new(&mut stale, &["vector"], IndexType::Vector, &params)
            .name(VECTOR_INDEX.to_string())
            .execute_uncommitted()
            .await
            .unwrap()
    };
    assert_eq!(segment.fragment_bitmap.as_ref().unwrap(), &covered);

    // A concurrent writer deletes one fragment's rows WHOLE, so the fragment
    // leaves the manifest without any transition recording it.
    let deleted = dataset.fragments().last().unwrap().id as u32;
    let bounds = {
        let mut scan = dataset.scan();
        scan.project(&["i"]).unwrap();
        scan.with_row_address();
        let batch = scan.try_into_batch().await.unwrap();
        let addresses = batch[lance_core::ROW_ADDR].as_primitive::<UInt64Type>();
        let values = batch["i"].as_primitive::<Int32Type>();
        let mut rows: Vec<i32> = (0..batch.num_rows())
            .filter(|row| (addresses.value(*row) >> 32) as u32 == deleted)
            .map(|row| values.value(row))
            .collect();
        rows.sort_unstable();
        rows
    };
    assert!(!bounds.is_empty(), "the fragment must hold rows");
    dataset
        .delete(&format!(
            "i >= {} AND i <= {}",
            bounds[0],
            bounds[bounds.len() - 1]
        ))
        .await
        .unwrap();
    let live: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert!(
        !live.contains(deleted),
        "the delete must drop fragment {deleted} whole"
    );
    assert_eq!(
        live,
        &covered - RoaringBitmap::from_iter([deleted]),
        "nothing but {deleted} may leave the manifest"
    );

    // Now the stale handle commits its segment; the resolver rebases it.
    let committed = CommitBuilder::new(Arc::new(stale))
        .execute(Transaction::new(
            pre_rewrite_version(&segment),
            Operation::CreateIndex {
                new_indices: vec![segment.clone()],
                removed_indices: vec![],
            },
            None,
        ))
        .await
        .expect("the rebased index build must land");

    let landed = stored_segments(&committed).await;
    let vector_landed = coverage_of(&landed, VECTOR_INDEX);
    // The staged bitmap lands exactly as it was built: no withdrawal ran, so
    // the segment keeps every fragment, including the deleted one the rebase
    // does not prune here (the reader filters its coverage to the live set).
    assert_eq!(
        vector_landed,
        covered,
        "a plain delete must withdraw nothing; the segment kept {:?} of the \
         staged {:?}",
        vector_landed.iter().collect::<Vec<_>>(),
        covered.iter().collect::<Vec<_>>()
    );
    assert!(
        (&vector_landed & &live) == live,
        "every fragment the delete did not touch must still be covered"
    );
    assert_query_exactness(&committed).await;
}

/// A `CreateIndex` whose read version IS the manifest version was not
/// rebased over anything, so no withdrawal is computed: a segment covering a
/// fragment that left the manifest before the build started keeps its stored
/// bitmap exactly as it was staged.
#[tokio::test]
async fn a_create_index_at_the_current_version_withdraws_nothing() {
    let mut dataset = fixture().await;
    dataset = make_tagged(dataset).await;
    let live: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();

    // Stage a segment that also names a fragment the partition retired, so
    // the withdrawal would have something to take if it ran at all.
    let retired = dataset
        .manifest
        .max_fragment_id()
        .unwrap()
        .checked_add(1)
        .unwrap() as u32;
    let mut segment = {
        let mut params =
            VectorIndexParams::ivf_pq(1, 4, 2, lance_linalg::distance::DistanceType::L2, 10);
        params.version(IndexFileVersion::Legacy);
        CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
            .name(VECTOR_INDEX.to_string())
            .execute_uncommitted()
            .await
            .unwrap()
    };
    let staged = &live | RoaringBitmap::from_iter([retired]);
    segment.fragment_bitmap = Some(staged.clone());

    let committed = CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![segment],
                removed_indices: vec![],
            },
            None,
        ))
        .await
        .expect("a CreateIndex at the current version must land");
    assert_eq!(
        coverage_of(&stored_segments(&committed).await, VECTOR_INDEX),
        staged,
        "a CreateIndex that was never rebased must land its bitmap unchanged"
    );
}

/// A PARTIAL compaction costs a rebased undeferrable segment only the
/// fragments the rewrite actually consumed: the ones it left alone are still
/// servable and keep their coverage.
///
/// The rewrite is already committed when the withdrawal is computed, so its
/// sources are outside the manifest's live set. Measuring the remainder rule
/// against that post-rewrite liveness reads the untouched fragments as a
/// live-only remainder the history does not mention and clears them, which
/// empties the segment. The withdrawal must instead be measured against the
/// liveness of the transaction's own read snapshot for exactly those
/// sources, which is what the forward commit order sees.
#[tokio::test]
async fn a_partial_rewrite_costs_a_rebased_index_only_the_rewritten_source() {
    Box::pin(partial_rewrite_coverage()).await;
}

/// As [`a_partial_rewrite_costs_a_rebased_index_only_the_rewritten_source`],
/// but the CreateIndex is committed through a handle that has already moved
/// past the rewrite while the transaction still names the pre-rewrite read
/// version (what `CommitBuilder` on a URI does: it loads the latest version).
///
/// The withdrawal must be measured against the read-version snapshot the
/// transaction names, not against whatever version the committing handle
/// happens to hold. Measuring against a newer handle makes the rewrite's
/// sources absent from it, so nothing restores their pre-rewrite liveness and
/// the untouched fragments are read as a live-only remainder and cleared.
#[tokio::test]
async fn a_rebased_index_is_measured_against_its_read_version_not_the_handle() {
    Box::pin(partial_rewrite_coverage_committed_from_latest()).await;
}

async fn partial_rewrite_coverage() -> RoaringBitmap {
    Box::pin(partial_rewrite_coverage_from(CommittingHandle::ReadVersion)).await
}

/// Which handle commits the rebased CreateIndex.
#[derive(Clone, Copy, PartialEq)]
enum CommittingHandle {
    /// The build's own handle, still at the transaction's read version.
    ReadVersion,
    /// A handle reloaded at the latest version, past the rewrite, as
    /// `CommitBuilder` on a URI produces.
    Latest,
}

async fn partial_rewrite_coverage_committed_from_latest() -> RoaringBitmap {
    Box::pin(partial_rewrite_coverage_from(CommittingHandle::Latest)).await
}

/// Builds a tagged table, stages an undeferrable segment over every live
/// fragment, and commits a deferred compaction of the lineage's two recorded
/// destinations before the staged segment is committed, so the resolver
/// rebases it. `handle` selects which dataset handle commits the rebased
/// CreateIndex. Returns the segment's stored coverage; the withdrawal of the
/// rewritten sources and the survival of everything else are asserted here.
async fn partial_rewrite_coverage_from(handle: CommittingHandle) -> RoaringBitmap {
    let mut dataset = fixture().await;
    // Tagged through a partition of the FIRST fragment, whose two
    // destinations the lineage records: only a recorded group may be rebased
    // over at all (`check_create_index` refuses an unrecorded one).
    let partitioned = dataset.fragments()[0].id;
    dataset = make_tagged_partitioning(dataset, &[partitioned]).await;
    // The partition's destinations carry reserved ids above every original
    // fragment's, and they are the only live fragments the lineage records.
    let destinations: Vec<u32> = {
        let recorded = load_all_indices(&dataset)
            .await
            .unwrap()
            .iter()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
            .expect("the tagged entry must exist")
            .fragment_bitmap
            .clone()
            .expect("the tagged entry must record its lineage");
        dataset
            .fragments()
            .iter()
            .map(|fragment| fragment.id as u32)
            .filter(|id| recorded.contains(*id))
            .collect()
    };
    assert_eq!(destinations.len(), 2, "the partition produces two outputs");
    let (companion, rewritten) = (destinations[0], destinations[1]);
    // Everything else the segment covers and the rewrite leaves alone.
    let remainder: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .filter(|id| *id != companion && *id != rewritten)
        .collect();
    assert!(!remainder.is_empty(), "there must be untouched coverage");
    let covered: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();

    // The index build reads this version and keeps its handle.
    let mut stale = dataset.clone();
    let segment = {
        let mut params =
            VectorIndexParams::ivf_pq(1, 4, 2, lance_linalg::distance::DistanceType::L2, 10);
        params.version(IndexFileVersion::Legacy);
        CreateIndexBuilder::new(&mut stale, &["vector"], IndexType::Vector, &params)
            .name(VECTOR_INDEX.to_string())
            .execute_uncommitted()
            .await
            .unwrap()
    };
    assert_eq!(
        segment.fragment_bitmap.as_ref().unwrap(),
        &covered,
        "the staged segment must cover every fragment live at its read version"
    );

    let options = CompactionOptions {
        excluded_fragment_ids: dataset
            .fragments()
            .iter()
            .map(|fragment| fragment.id as u32)
            .filter(|id| *id != companion && *id != rewritten)
            .collect(),
        // Pinned so candidacy does not depend on the fixture's file sizes:
        // both fragments are under the row target and under this floor, so
        // the planner bins exactly the two.
        min_bytes_per_file: Some(WriteParams::default().max_bytes_per_file),
        ..deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize)
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let planned: RoaringBitmap = plan
        .tasks()
        .iter()
        .flat_map(|task| task.fragments.iter())
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        planned,
        RoaringBitmap::from_iter([companion, rewritten]),
        "the compaction must rewrite the two recorded destinations only"
    );

    let index_transaction = Transaction::new(
        pre_rewrite_version(&segment),
        Operation::CreateIndex {
            new_indices: vec![segment.clone()],
            removed_indices: vec![],
        },
        None,
    );
    let results = execute_tasks(&dataset, &plan, &options, None).await;
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();
    // The index build handle is still at the pre-rewrite version, so the
    // commit path rebases its segment over the committed rewrite.
    let committing = match handle {
        CommittingHandle::ReadVersion => stale,
        // Past the rewrite already, while the transaction still names the
        // pre-rewrite read version.
        CommittingHandle::Latest => {
            let latest = dataset.clone();
            assert!(
                latest.manifest.version > index_transaction.read_version,
                "the committing handle must be newer than the read version"
            );
            latest
        }
    };
    let committed = CommitBuilder::new(Arc::new(committing))
        .execute(index_transaction)
        .await
        .expect("the rebased index build must land");

    let stored = coverage_of(&stored_segments(&committed).await, VECTOR_INDEX);
    let live: RoaringBitmap = committed
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert!(
        !stored.contains(rewritten) && !stored.contains(companion),
        "the rewritten sources must be withdrawn: {:?}",
        stored.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        &stored & &remainder,
        remainder,
        "the fragments the rewrite left alone are still servable and must \
         keep their coverage; the segment kept {:?} of the untouched {:?}",
        stored.iter().collect::<Vec<_>>(),
        remainder.iter().collect::<Vec<_>>()
    );
    assert!(
        stored.is_subset(&live),
        "a withdrawal never leaves a non-live fragment behind: {:?}",
        (&stored - &live).iter().collect::<Vec<_>>()
    );
    // Row-level exactness only: this fixture's legacy IVF_PQ file is too
    // small for its own ANN path, and the contract under test is which
    // coverage survives, not how the index answers.
    let scan = committed
        .scan()
        .project(&["i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let mut rows: Vec<i32> = scan["i"]
        .as_primitive::<Int32Type>()
        .iter()
        .map(|value| value.unwrap())
        .collect();
    rows.sort_unstable();
    let deduped = {
        let mut deduped = rows.clone();
        deduped.dedup();
        deduped
    };
    assert_eq!(rows, deduped, "a rewrite must not duplicate a row");
    stored
}

/// `retire_emptied_segments` sees only the transaction's `new_indices` in the
/// CreateIndex arm, so a same-name sibling already in the manifest is
/// invisible to it: an emptied NEW segment is kept as the rebuild record even
/// though the sibling still serves the index. This pins the whole resulting
/// manifest for that case.
///
/// The outcome is harmless: the sibling keeps its coverage and its
/// definition, the new segment stays as an empty record, and the reader
/// derives the sibling's coverage alone (an empty bitmap contributes
/// nothing), so queries are served by the sibling. Nothing usable is lost,
/// so the arm's final-set decision is left alone.
#[tokio::test]
async fn an_emptied_create_index_segment_is_kept_beside_its_sibling() {
    let mut dataset = fixture().await;
    dataset = make_tagged(dataset).await;

    // The sibling: a legacy vector segment over one fragment the rewrite
    // below never touches, committed before the staged build starts.
    let recorded: RoaringBitmap = {
        let entry = load_all_indices(&dataset)
            .await
            .unwrap()
            .iter()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
            .expect("the tagged entry must exist")
            .fragment_bitmap
            .clone()
            .expect("the tagged entry must record its lineage");
        dataset
            .fragments()
            .iter()
            .map(|fragment| fragment.id as u32)
            .filter(|id| entry.contains(*id))
            .collect()
    };
    assert_eq!(recorded.len(), 2, "the partition produces two outputs");
    let untouched: u32 = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .find(|id| !recorded.contains(*id))
        .expect("a fragment off the lineage must exist");

    // Two legacy segments of the index name under test, built here and not
    // committed: `create_index` refuses a duplicate name, so the sibling is
    // installed directly into the manifest instead. Each has its own index
    // files, since a commit to the name reads every segment it keeps.
    let (template, sibling) = {
        let mut scratch = dataset.clone();
        let mut params =
            VectorIndexParams::ivf_pq(1, 4, 2, lance_linalg::distance::DistanceType::L2, 10);
        params.version(IndexFileVersion::Legacy);
        let template =
            CreateIndexBuilder::new(&mut scratch, &["vector"], IndexType::Vector, &params)
                .name(VECTOR_INDEX.to_string())
                .execute_uncommitted()
                .await
                .unwrap();
        let sibling =
            CreateIndexBuilder::new(&mut scratch, &["vector"], IndexType::Vector, &params)
                .name(VECTOR_INDEX.to_string())
                .execute_uncommitted()
                .await
                .unwrap();
        (template, sibling)
    };

    // The sibling: a segment of the index name under test, already in the
    // manifest, covering only the fragment the rewrite never touches. It is
    // installed BEFORE the staged build reads the table, so the rebase walks
    // ordinary transaction files afterwards.
    let sibling_coverage = RoaringBitmap::from_iter([untouched]);
    let sibling_uuid = sibling.uuid;
    {
        let mut sibling = sibling;
        sibling.fragment_bitmap = Some(sibling_coverage.clone());
        let mut indices = load_all_indices(&dataset).await.unwrap().to_vec();
        indices.push(sibling);
        crate::index::frag_reuse_reader::tests::persist_fixture(&mut dataset, indices).await;
    }

    // The staged build reads the version that already has the sibling, and
    // covers the recorded destinations the rewrite then retires: every bit of
    // its coverage is withdrawn, so it lands emptied.
    let stale = dataset.clone();
    let mut segment = template;
    segment.fragment_bitmap = Some(recorded.clone());
    segment.dataset_version = dataset.manifest.version;
    assert_ne!(segment.uuid, sibling_uuid, "it must be a second segment");

    // The rewrite of the recorded destinations commits first.
    let options = CompactionOptions {
        excluded_fragment_ids: dataset
            .fragments()
            .iter()
            .map(|fragment| fragment.id as u32)
            .filter(|id| !recorded.contains(*id))
            .collect(),
        min_bytes_per_file: Some(WriteParams::default().max_bytes_per_file),
        ..deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize)
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let rewritten: RoaringBitmap = plan
        .tasks()
        .iter()
        .flat_map(|task| task.fragments.iter())
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        rewritten, recorded,
        "the compaction must rewrite the recorded destinations only"
    );
    let results = execute_tasks(&dataset, &plan, &options, None).await;
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    // Now the stale handle commits its segment; the resolver rebases it.
    let committed = CommitBuilder::new(Arc::new(stale))
        .execute(Transaction::new(
            pre_rewrite_version(&segment),
            Operation::CreateIndex {
                new_indices: vec![segment.clone()],
                removed_indices: vec![],
            },
            None,
        ))
        .await
        .expect("the rebased index build must land");

    // The complete resulting manifest for this index name.
    let landed: Vec<(Uuid, RoaringBitmap)> = stored_segments(&committed)
        .await
        .into_iter()
        .filter(|(name, _, _)| name == VECTOR_INDEX)
        .map(|(_, uuid, bitmap)| (uuid, bitmap))
        .collect();
    assert_eq!(
        landed,
        vec![
            (sibling_uuid, sibling_coverage.clone()),
            (segment.uuid, RoaringBitmap::new()),
        ],
        "the sibling must keep its coverage and the emptied new segment must \
         stay as the rebuild record"
    );
    let definitions = load_all_indices(&committed).await.unwrap();
    for (uuid, _) in &landed {
        assert!(
            definitions
                .iter()
                .any(|index| index.uuid == *uuid && index.index_details.is_some()),
            "segment {uuid} must keep its definition"
        );
    }
    // The reader derives the sibling's coverage alone: an empty bitmap claims
    // nothing, so the untouched fragment is still served by the index.
    assert_eq!(
        derived_coverage(&committed, VECTOR_INDEX).await,
        sibling_coverage,
        "the sibling must still serve the fragment it covers"
    );
    // Row-level exactness: a rewrite moves rows, it never adds or drops any.
    let scan = committed
        .scan()
        .project(&["i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let mut rows: Vec<i32> = scan["i"]
        .as_primitive::<Int32Type>()
        .iter()
        .map(|value| value.unwrap())
        .collect();
    rows.sort_unstable();
    let deduped = {
        let mut deduped = rows.clone();
        deduped.dedup();
        deduped
    };
    assert_eq!(rows, deduped, "a rewrite must not duplicate a row");
    assert_eq!(
        rows.len(),
        4 * INDEX_ROWS_PER_FRAGMENT as usize,
        "every row must survive"
    );
}

/// The read version a staged segment was built at: its coverage names the
/// fragments of that version, and the commit must be submitted from it so the
/// resolver actually rebases.
fn pre_rewrite_version(segment: &IndexMetadata) -> u64 {
    segment.dataset_version
}

/// Makes `dataset` tagged with a stable partition of its first two fragments,
/// before any user index exists.
async fn make_tagged(mut dataset: Dataset) -> Dataset {
    let reserve = Transaction::new(
        dataset.manifest.version,
        Operation::ReserveFragments { num_fragments: 2 },
        None,
    );
    dataset
        .apply_commit(reserve, &Default::default(), &Default::default())
        .await
        .unwrap();
    let dest_base_id = dataset.manifest.max_fragment_id().unwrap() - 1;
    let old_fragments: Vec<Fragment> = dataset.fragments()[..2].to_vec();
    let source_ids: Vec<u64> = old_fragments.iter().map(|fragment| fragment.id).collect();
    let (transition, new_fragments) = prepare_partition(&dataset, &source_ids, dest_base_id).await;
    let read_version = dataset.manifest.version;
    let frag_reuse_index = Some(
        frag_reuse_entry_appending(&dataset, vec![transition])
            .await
            .unwrap(),
    );
    let dataset = CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments,
                    new_fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index,
            },
            None,
        ))
        .await
        .unwrap();
    let stored = load_all_indices(&dataset).await.unwrap();
    assert!(
        uses_tagged_fri(
            &dataset.manifest,
            stored
                .iter()
                .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        ),
        "the stable partition must make the table tagged"
    );
    dataset
}

/// Every live row is returned exactly once by a KNN query and by a scan, with
/// the KNN answer equal to brute force: no retired row, no duplicate.
async fn assert_query_exactness(dataset: &Dataset) {
    let scan = dataset
        .scan()
        .project(&["i"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let mut live: Vec<i32> = scan["i"]
        .as_primitive::<Int32Type>()
        .iter()
        .map(|value| value.unwrap())
        .collect();
    live.sort_unstable();
    let deduped = {
        let mut deduped = live.clone();
        deduped.dedup();
        deduped
    };
    assert_eq!(live, deduped, "the scan returned a duplicate row");
    let query = query_vector(dataset, live[live.len() / 2]).await;
    let (rows, plan) = indexed_knn(dataset, &query, 5).await;
    assert_knn_matches_brute_force(dataset, &query, 5, &rows, &plan).await;
}

/// An object store whose reads of one suffix can be failed on demand, so a
/// transient storage fault can be told apart from a missing object.
#[derive(Debug)]
struct FailingReadWrapper {
    /// Reads of a path ending in this suffix fail while `failing` is set.
    suffix: String,
    failing: Arc<AtomicBool>,
}

impl lance_io::object_store::WrappingObjectStore for FailingReadWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(FailingReadStore {
            inner: original,
            suffix: self.suffix.clone(),
            failing: self.failing.clone(),
        })
    }

    // The wrapper fails paths, so a pushed-down listing must not walk past it.
    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}

#[derive(Debug)]
struct FailingReadStore {
    inner: Arc<dyn object_store::ObjectStore>,
    suffix: String,
    failing: Arc<AtomicBool>,
}

impl FailingReadStore {
    fn fails(&self, location: &object_store::path::Path) -> bool {
        self.failing.load(Ordering::SeqCst) && location.as_ref().ends_with(&self.suffix)
    }

    /// A generic error, deliberately NOT `NotFound`: the point is that a fault
    /// that says nothing about the object's existence must not be read as a
    /// format verdict.
    fn injected_error() -> object_store::Error {
        object_store::Error::Generic {
            store: "failing-read-test-store",
            source: "injected transient read failure".to_string().into(),
        }
    }
}

impl std::fmt::Display for FailingReadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailingReadStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for FailingReadStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        if self.fails(location) {
            return Err(Self::injected_error());
        }
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        if self.fails(location) {
            return Err(Self::injected_error());
        }
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&object_store::path::Path>,
        offset: &object_store::path::Path,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, opts).await
    }

    async fn rename_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::RenameOptions,
    ) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, opts).await
    }
}

const FTS_INDEX: &str = "doc_idx";

/// A table on disk with `i`, a vector column and a `doc` column, made tagged
/// and indexed with a healthy (non-legacy) inverted index on `doc`.
///
/// On disk rather than in RAM because the test opens it through a wrapping
/// object store, which `into_ram_dataset` cannot carry.
async fn tagged_with_fts_index(path: &str) -> Dataset {
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "vector",
            lance_datagen::array::rand_vec::<Float32Type>(4.into()),
        )
        .col("doc", lance_datagen::array::random_sentence(1, 10, false))
        .into_dataset(
            path,
            FragmentCount::from(4),
            FragmentRowCount::from(INDEX_ROWS_PER_FRAGMENT),
        )
        .await
        .unwrap();
    dataset = make_tagged(dataset).await;
    dataset
        .create_index(
            &["doc"],
            IndexType::Inverted,
            Some(FTS_INDEX.into()),
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    dataset
}

/// Reopens `path` through a store that fails reads of `suffix` while the
/// returned flag is set.
async fn open_with_failing_reads(path: &str, suffix: String) -> (Dataset, Arc<AtomicBool>) {
    let failing = Arc::new(AtomicBool::new(false));
    let dataset = DatasetBuilder::from_uri(path)
        .with_store_params(ObjectStoreParams {
            object_store_wrapper: Some(Arc::new(FailingReadWrapper {
                suffix,
                failing: failing.clone(),
            })),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    (dataset, failing)
}

/// The relative path of a segment's inverted metadata file, as the index
/// store reads it.
async fn fts_metadata_suffix(dataset: &Dataset, name: &str) -> String {
    let index = dataset
        .load_index_by_name(name)
        .await
        .unwrap()
        .expect("the index must exist");
    format!(
        "{}/{}",
        index.uuid,
        lance_index::scalar::inverted::METADATA_FILE
    )
}

/// A word that occurs in the table, to drive full-text search.
async fn sample_word(dataset: &Dataset) -> String {
    let batch = dataset
        .scan()
        .project(&["doc"])
        .unwrap()
        .limit(Some(1), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let doc = batch["doc"].as_string::<i32>().value(0).to_string();
    doc.split_whitespace()
        .next()
        .expect("a document must have a word")
        .to_string()
}

/// The number of rows full-text search returns for `word`, with the plan so a
/// test can assert which path served it.
async fn fts_count(dataset: &Dataset, word: &str) -> (usize, String) {
    let mut scan = dataset.scan();
    scan.full_text_search(FullTextSearchQuery::new(word.to_string()))
        .unwrap();
    scan.project::<String>(&[]).unwrap().with_row_id();
    let plan = scan.explain_plan(false).await.unwrap();
    let count = scan.count_rows().await.unwrap();
    (count as usize, plan)
}

/// [`fts_count`] with the queried column pinned, so the query is answerable
/// even when no inverted index derives coverage and the scanner therefore
/// cannot discover the column from the index list.
async fn fts_count_on_column(dataset: &Dataset, word: &str, column: &str) -> (usize, String) {
    let mut scan = dataset.scan();
    scan.full_text_search(
        FullTextSearchQuery::new(word.to_string())
            .with_column(column.to_string())
            .unwrap(),
    )
    .unwrap();
    scan.project::<String>(&[]).unwrap().with_row_id();
    let plan = scan.explain_plan(false).await.unwrap();
    let count = scan.count_rows().await.unwrap();
    (count as usize, plan)
}

/// W1: a transient read failure on the inverted metadata file is NOT a legacy
/// format verdict. The deferred compaction attempt must not persist a
/// narrowed bitmap from it, and nothing may be cached: once the fault clears,
/// the segment classifies as deferrable and keeps its provenance.
#[tokio::test]
async fn a_read_failure_does_not_classify_an_inverted_index_as_legacy() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_read_failure", dir.as_str());
    let seed = tagged_with_fts_index(&path).await;
    let suffix = fts_metadata_suffix(&seed, FTS_INDEX).await;
    let fts_before = coverage_of(&stored_segments(&seed).await, FTS_INDEX);
    assert!(
        !fts_before.is_empty(),
        "the fixture's FTS segment must cover fragments"
    );
    let word = sample_word(&seed).await;
    let (count_before, plan_before) = fts_count(&seed, &word).await;
    assert!(
        plan_before.contains("MatchQuery"),
        "the fixture's FTS index must serve the query: {plan_before}"
    );
    drop(seed);

    let (mut dataset, failing) = open_with_failing_reads(&path, suffix.clone()).await;
    let sources: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert!(
        !(&fts_before & &sources).is_empty(),
        "the FTS segment must cover what the compaction rewrites, or the \
         withdrawal under test could not happen at all"
    );
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);

    // Reads of the metadata file fail with a generic (not NotFound) error.
    failing.store(true, Ordering::SeqCst);
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let results = execute_tasks(&dataset, &plan, &options, None).await;
    let outcome = commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await;
    failing.store(false, Ordering::SeqCst);

    // Whatever the existing error rules did with the fault (surfaced it, or
    // retried until it committed), no coverage may have been withdrawn from
    // the FTS segment on the strength of an unreadable file. This is the
    // assertion that fails on the collapsed `is_ok()`.
    let reopened = DatasetBuilder::from_uri(&path).load().await.unwrap();
    assert_eq!(
        coverage_of(&stored_segments(&reopened).await, FTS_INDEX),
        fts_before,
        "a read fault must not narrow the FTS segment's provenance (commit {})",
        if outcome.is_ok() { "landed" } else { "failed" }
    );

    // Nothing was cached: the SAME handle, whose index cache saw the fault,
    // classifies the segment as deferrable now that reads succeed. Observed
    // through the selector, which lists exactly the `Some(false)` segments.
    let index = dataset
        .load_index_by_name(FTS_INDEX)
        .await
        .unwrap()
        .expect("the FTS index definition must survive");
    let covered = index.fragment_bitmap.clone().unwrap();
    assert!(
        crate::index::frag_reuse::undeferrable_rewrite_segments(&dataset, &covered)
            .await
            .unwrap()
            .is_empty(),
        "a read failure must not be cached as the legacy verdict"
    );

    // With reads restored the compaction runs to completion and the FTS
    // segment keeps its provenance and still serves the query.
    let mut dataset = DatasetBuilder::from_uri(&path).load().await.unwrap();
    if dataset.manifest.version == reopened.manifest.version {
        run_deferred_compaction(&mut dataset, &options, None).await;
    }
    let fts_final = coverage_of(&stored_segments(&dataset).await, FTS_INDEX);
    assert_eq!(
        fts_final, fts_before,
        "the healthy FTS segment's stored provenance is never withdrawn"
    );
    let destinations: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        derived_coverage(&dataset, FTS_INDEX).await,
        destinations,
        "the FTS segment must translate to the rewrite's outputs"
    );
    let (count_after, plan_after) = fts_count(&dataset, &word).await;
    assert!(
        plan_after.contains("MatchQuery"),
        "the retained FTS segment must still serve the query: {plan_after}"
    );
    assert_eq!(
        count_after, count_before,
        "full-text search must stay exact after the deferred compaction"
    );
}

/// W1 control: a positively MISSING metadata file is the legacy layout, and
/// the deferred compaction does withdraw that segment's coverage of the
/// rewritten sources.
#[tokio::test]
async fn a_missing_inverted_metadata_file_is_the_legacy_verdict() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_missing_metadata", dir.as_str());
    let dataset = tagged_with_fts_index(&path).await;
    let fts_before = coverage_of(&stored_segments(&dataset).await, FTS_INDEX);
    assert!(!fts_before.is_empty());
    let sources: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert!(
        !(&fts_before & &sources).is_empty(),
        "the FTS segment must cover what the compaction rewrites"
    );

    // Delete the metadata file: now the object genuinely is not there.
    let index = dataset
        .load_index_by_name(FTS_INDEX)
        .await
        .unwrap()
        .unwrap();
    let metadata_path = dataset
        .indices_dir()
        .join(index.uuid.to_string())
        .join(lance_index::scalar::inverted::METADATA_FILE);
    dataset.object_store.delete(&metadata_path).await.unwrap();
    // Reopened so the classification reads storage rather than the handle's
    // caches, as a fresh writer would.
    let mut dataset = DatasetBuilder::from_uri(&path).load().await.unwrap();

    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    let committed = run_deferred_compaction(&mut dataset, &options, None).await;
    assert!(committed > 0, "the compaction must commit");

    let fts_after = coverage_of(&stored_segments(&dataset).await, FTS_INDEX);
    assert!(
        fts_after.is_disjoint(&sources),
        "a legacy inverted segment's coverage of the rewritten sources is withdrawn: {:?}",
        (&fts_after & &sources).iter().collect::<Vec<_>>()
    );
    assert!(
        stored_segments(&dataset)
            .await
            .iter()
            .any(|(name, _, _)| name == FTS_INDEX),
        "the index definition must survive the withdrawal"
    );
}

/// W2: an undeferrable segment can address a rewritten fragment one hop away.
///
/// `P` was partitioned into `F1` and `F2` while the segment covered `P`, so
/// the segment's provenance names `P` and nothing else; a later rewrite of
/// `F1` alone never names `P`. The selector must still classify the segment
/// and the commit must withdraw `P`.
///
/// The manifest this test rewrites is the shape a build BEFORE this change
/// left behind: its Rewrite commits withdrew nothing, so an undeferrable
/// segment could survive a partition still claiming the retired source. This
/// build can no longer produce that state (the partition's own Rewrite
/// withdraws a direct hit, and a rebased `CreateIndex` strips retired
/// provenance), so the older manifest is written directly through the
/// A fault reading one segment's format marker must not take the whole table
/// down. The marker probe runs inside `load_indices`, which every query plan
/// calls, so propagating the error would fail vector search, scalar lookups
/// and plain scans alike. The segment is excluded for the duration of the
/// fault and nothing is cached, so the SAME handle serves it again afterwards.
#[tokio::test]
async fn a_read_failure_on_one_segment_does_not_fail_every_query() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_read_failure_queries", dir.as_str());
    let mut seed = tagged_with_fts_index(&path).await;
    // A second index, on another column, to prove an unrelated query is not
    // collateral damage of the faulting segment.
    create_btree_index(&mut seed).await;
    let suffix = fts_metadata_suffix(&seed, FTS_INDEX).await;
    let word = sample_word(&seed).await;
    drop(seed);

    // Compact first, so the segment's provenance is retired and the reader
    // must translate it: that is the only state in which the format marker is
    // probed at all, and therefore the only state the fault can reach.
    {
        let mut writer = DatasetBuilder::from_uri(&path).load().await.unwrap();
        let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
        run_deferred_compaction(&mut writer, &options, None).await;
    }

    // A handle whose index cache has never seen a successful probe, so the
    // fault is what the probe meets.
    let (dataset, failing) = open_with_failing_reads(&path, suffix).await;
    let fts_before = coverage_of(&stored_segments(&dataset).await, FTS_INDEX);
    failing.store(true, Ordering::SeqCst);
    // The index list itself resolves, so planning works at all.
    dataset
        .load_indices()
        .await
        .expect("a read fault on one segment must not fail the index listing");
    // A plain scan is unaffected.
    let scanned = dataset
        .scan()
        .project(&["i"])
        .unwrap()
        .try_into_batch()
        .await
        .expect("a read fault on an index must not fail a scan")
        .num_rows();
    assert!(scanned > 0);
    // The unrelated BTree index still plans and answers.
    let mut lookup = dataset.scan();
    lookup.filter("i = 1").unwrap().project(&["i"]).unwrap();
    let lookup_plan = lookup
        .explain_plan(false)
        .await
        .expect("a read fault on the FTS segment must not fail a BTree query");
    assert!(lookup_plan.contains("ScalarIndexQuery"), "{lookup_plan}");
    let looked_up = lookup
        .try_into_batch()
        .await
        .expect("a read fault on the FTS segment must not fail a BTree query")
        .num_rows();
    assert_eq!(looked_up, 1, "the BTree query must answer during the fault");
    // The faulting segment is excluded rather than trusted, and the flat
    // fallback answers.
    let (count_during, plan_during) = fts_count_on_column(&dataset, &word, "doc").await;
    assert!(
        plan_during.contains("FlatMatchQuery"),
        "the unreadable segment must be excluded, not trusted: {plan_during}"
    );
    assert!(count_during > 0, "the word must match rows");

    // Nothing about the fault was cached: once reads succeed the SAME handle
    // (no reopen, same manifest snapshot) re-probes the segment, so the
    // listing carries its coverage again and the index serves the query
    // instead of the flat fallback.
    failing.store(false, Ordering::SeqCst);
    let destinations: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        derived_coverage(&dataset, FTS_INDEX).await,
        destinations,
        "an unverifiable segment must be re-probed, not cached as excluded"
    );
    let (_, plan_after) = fts_count_on_column(&dataset, &word, "doc").await;
    assert!(
        !plan_after.contains("FlatMatchQuery"),
        "the re-probed segment must serve the query once reads succeed: {plan_after}"
    );
    // The selector likewise classifies it as deferrable, so nothing is
    // withdrawn.
    assert!(
        crate::index::frag_reuse::undeferrable_rewrite_segments(&dataset, &fts_before)
            .await
            .unwrap()
            .is_empty(),
        "a read fault must not be cached as the legacy verdict"
    );
    assert_eq!(
        coverage_of(&stored_segments(&dataset).await, FTS_INDEX),
        fts_before,
        "a read fault never narrows stored provenance"
    );
}

/// Words the fixture's analyzer drops, so a test never drives full-text
/// search with a term the index cannot match.
const FIXTURE_STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// Whether `plan` contains `node` as a plan node of its own.
///
/// Node names nest as substrings (`FlatMatchQuery:` contains `MatchQuery:`),
/// so "which node served this query" is only answerable at line starts.
fn plan_has_node(plan: &str, node: &str) -> bool {
    plan.lines().any(|line| line.trim_start().starts_with(node))
}

/// A non-stopword occurring in `column`, with the number of rows containing
/// it counted by brute force over the table.
///
/// The generator joins lowercase words with single spaces and adds no
/// punctuation, so whitespace tokens are exactly what the analyzer indexes
/// and the count is ground truth rather than an approximation.
async fn frequent_word_and_brute_force_count(dataset: &Dataset, column: &str) -> (String, usize) {
    let batch = dataset
        .scan()
        .project(&[column])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let docs = batch[column].as_string::<i32>();
    let mut rows_per_word: HashMap<&str, usize> = HashMap::new();
    for row in 0..docs.len() {
        for word in docs
            .value(row)
            .split_whitespace()
            .filter(|word| !FIXTURE_STOP_WORDS.contains(word))
            .collect::<HashSet<_>>()
        {
            *rows_per_word.entry(word).or_default() += 1;
        }
    }
    let (word, count) = rows_per_word
        .into_iter()
        .max_by_key(|(word, count)| (*count, *word))
        .expect("the fixture's documents must contain a non-stopword");
    (word.to_string(), count)
}

/// A probe fault excludes the segment for the duration of the fault only, so
/// the plan the first (degraded) query produced must never be reused once
/// reads succeed. The degraded answer is exact but slow, and a cached
/// "excluded" verdict would make every later query on the handle pay that
/// flat scan forever; equally, a cached plan built while the segment was
/// visible must not survive a fault. The second query on the SAME handle
/// therefore re-probes, serves from the inverted index, and derives the
/// segment's full coverage again.
#[tokio::test]
async fn a_transient_probe_outage_never_poisons_the_cached_query_plan() {
    Box::pin(transient_probe_outage_never_poisons_the_cached_query_plan()).await;
}

async fn transient_probe_outage_never_poisons_the_cached_query_plan() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_probe_outage_plan_cache", dir.as_str());
    let seed = tagged_with_fts_index(&path).await;
    let suffix = fts_metadata_suffix(&seed, FTS_INDEX).await;
    let (word, brute_force) = frequent_word_and_brute_force_count(&seed, "doc").await;
    assert!(
        brute_force > 0,
        "the chosen word {word} must match rows in the fixture"
    );
    drop(seed);

    // The segment's provenance must be retired for the format marker to be
    // probed at all; that probe is the only thing the fault can reach.
    {
        let mut writer = DatasetBuilder::from_uri(&path).load().await.unwrap();
        let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
        run_deferred_compaction(&mut writer, &options, None).await;
    }

    // A fresh handle, so the first probe this handle ever makes meets the
    // fault. The wrapper fails every read of the marker while armed, for as
    // long as it is armed.
    let (dataset, failing) = open_with_failing_reads(&path, suffix).await;
    failing.store(true, Ordering::SeqCst);
    let (count_during, plan_during) = fts_count_on_column(&dataset, &word, "doc").await;
    assert_eq!(
        count_during, brute_force,
        "the excluded segment's rows must still be served by the flat scan: \
         {plan_during}"
    );
    assert!(
        plan_has_node(&plan_during, "FlatMatchQuery:"),
        "the unreadable segment must be excluded in favour of a flat scan: \
         {plan_during}"
    );
    assert!(
        !plan_has_node(&plan_during, "MatchQuery:"),
        "an unreadable segment must not serve the query as an index: \
         {plan_during}"
    );

    // Same handle, same manifest snapshot: with reads restored the segment is
    // re-probed, so the index serves the query and the degraded plan is gone.
    failing.store(false, Ordering::SeqCst);
    let (count_after, plan_after) = fts_count_on_column(&dataset, &word, "doc").await;
    assert_eq!(
        count_after, brute_force,
        "the index answer must equal brute force: {plan_after}"
    );
    assert!(
        plan_has_node(&plan_after, "MatchQuery:"),
        "the re-probed segment must serve the query from the inverted index, \
         not from the outage's plan: {plan_after}"
    );
    assert!(
        !plan_has_node(&plan_after, "FlatMatchQuery:"),
        "the flat fallback must not outlive the fault: {plan_after}"
    );
    let destinations: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        derived_coverage(&dataset, FTS_INDEX).await,
        destinations,
        "the re-probed segment must derive its full coverage again"
    );
}

/// The commit-time selector treats a marker probe fault as "not
/// classified": the deferred compaction commits, withdraws nothing from the
/// segment it could not read, and the verdict is not cached, so once reads
/// succeed the same handle classifies the segment again. A withdrawal is
/// durable, so it must never rest on a transient fault.
/// Reads the fault must absorb to confine itself to the FIRST of the two
/// marker probes the maintenance plan build makes for one segment: the probe
/// inside the build's own `load_indices`, and the one
/// `missing_coverage_reason` makes for the segment that probe excluded. The
/// reader retries a failed metadata read once, so each probe spends two reads
/// and the budget is the first probe's pair.
const PLAN_BUILD_FIRST_PROBE_READS: usize = 4;

/// Reopens `path` through a store that fails reads of `suffix` only while a
/// shared budget of failures is left, so a test can clear the fault after a
/// known number of probes instead of on a wall-clock edge.
///
/// The maintenance plan build probes the marker twice for the same segment --
/// once in its own `load_indices`, once in `missing_coverage_reason` -- and
/// only a fault confined to the first probe produces the state this test is
/// about.
async fn open_with_budgeted_failing_reads(
    path: &str,
    suffix: String,
    budget: usize,
) -> (Dataset, Arc<AtomicUsize>) {
    let remaining = Arc::new(AtomicUsize::new(budget));
    let dataset = DatasetBuilder::from_uri(path)
        .with_store_params(ObjectStoreParams {
            object_store_wrapper: Some(Arc::new(BudgetedFailingReadWrapper {
                suffix,
                remaining: remaining.clone(),
            })),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    (dataset, remaining)
}

#[derive(Debug)]
struct BudgetedFailingReadWrapper {
    suffix: String,
    remaining: Arc<AtomicUsize>,
}

impl lance_io::object_store::WrappingObjectStore for BudgetedFailingReadWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(BudgetedFailingReadStore {
            inner: original,
            suffix: self.suffix.clone(),
            remaining: self.remaining.clone(),
        })
    }

    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}

#[derive(Debug)]
struct BudgetedFailingReadStore {
    inner: Arc<dyn object_store::ObjectStore>,
    suffix: String,
    remaining: Arc<AtomicUsize>,
}

impl BudgetedFailingReadStore {
    /// Consumes one unit of the budget when `location` matches.
    fn fails(&self, location: &object_store::path::Path) -> bool {
        if !location.as_ref().ends_with(&self.suffix) {
            return false;
        }
        self.remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok()
    }
}

impl std::fmt::Display for BudgetedFailingReadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BudgetedFailingReadStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for BudgetedFailingReadStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        if self.fails(location) {
            return Err(FailingReadStore::injected_error());
        }
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        if self.fails(location) {
            return Err(FailingReadStore::injected_error());
        }
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&object_store::path::Path>,
        offset: &object_store::path::Path,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, opts).await
    }

    async fn rename_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::RenameOptions,
    ) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, opts).await
    }
}

/// The query path only ever builds a plan for segments the listing kept, so a
/// probe fault there costs the segment its coverage for that one call. The
/// MAINTENANCE path plans over the STORED segments, so a fault that drops the
/// segment from the plan build's own listing sends it to
/// `missing_coverage_reason` instead -- and that second probe, made after the
/// fault cleared, returns a definitive `NoDerivedCoverage`. Caching a plan
/// assembled from those two inconsistent views freezes "no coverage" for the
/// whole snapshot: the remap skips the segment and every later query on the
/// handle is refused or flat-scanned, all because of a fault that is already
/// over. A segment absent from an unverifiable listing is therefore reported
/// as unverifiable without re-probing: the caller hears the fault and the
/// plan is not cached.
#[tokio::test]
async fn a_probe_outage_during_maintenance_planning_does_not_poison_the_cached_query_plan() {
    Box::pin(probe_outage_during_maintenance_planning_does_not_poison_the_cached_query_plan())
        .await;
}

async fn probe_outage_during_maintenance_planning_does_not_poison_the_cached_query_plan() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_maintenance_plan_cache", dir.as_str());
    let seed = tagged_with_fts_index(&path).await;
    let suffix = fts_metadata_suffix(&seed, FTS_INDEX).await;
    let (word, brute_force) = frequent_word_and_brute_force_count(&seed, "doc").await;
    assert!(
        brute_force > 0,
        "the chosen word {word} must match rows in the fixture"
    );
    drop(seed);

    // Retire the segment's provenance, so the plan build probes the marker at
    // all and the segment is a translation candidate rather than identity.
    {
        let mut writer = DatasetBuilder::from_uri(&path).load().await.unwrap();
        let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
        run_deferred_compaction(&mut writer, &options, None).await;
    }

    // A fresh handle with exactly one failure left in the budget: the plan
    // build's own `load_indices` probe fails (the segment leaves its listing,
    // uncached), and the `missing_coverage_reason` probe that follows succeeds
    // and answers "deferrable" -- the inconsistency this test is about.
    let (mut dataset, remaining) =
        open_with_budgeted_failing_reads(&path, suffix, PLAN_BUILD_FIRST_PROBE_READS).await;
    let before: Vec<(String, Uuid, RoaringBitmap)> = stored_segments(&dataset).await;
    let fts_before = coverage_of(&before, FTS_INDEX);
    assert!(!fts_before.is_empty());
    let segment_uuid = before
        .iter()
        .find(|(name, _, _)| name == FTS_INDEX)
        .map(|(_, uuid, _)| *uuid)
        .expect("the fixture must have an FTS segment");

    // The maintenance plan build. The listing it planned from had lost the
    // segment to the fault, so the caller hears that nothing was remapped --
    // a definitive reason re-probed against that incomplete listing would be
    // a silent skip -- and no verdict is left cached behind.
    let error = crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect_err("a segment absent from an unverifiable listing must not be a skipped success");
    let message = error.to_string();
    assert!(
        message.contains(&segment_uuid.to_string()),
        "the error must name the segment the caller asked about: {message}"
    );
    assert_eq!(
        remaining.load(Ordering::SeqCst),
        0,
        "the maintenance plan build must have met the injected fault"
    );

    // Reads have been healthy since the second probe. The same handle must
    // re-derive the segment's coverage and serve the query from the index.
    let destinations: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        derived_coverage(&dataset, FTS_INDEX).await,
        destinations,
        "a plan built from an unverifiable listing must not be cached: the \
         segment still covers every destination once reads succeed"
    );
    let (count_after, plan_after) = fts_count_on_column(&dataset, &word, "doc").await;
    assert_eq!(
        count_after, brute_force,
        "the index answer must equal brute force: {plan_after}"
    );
    assert!(
        plan_has_node(&plan_after, "MatchQuery:"),
        "the segment must serve the query from the inverted index, not from a \
         plan cached during the outage: {plan_after}"
    );
    assert!(
        !plan_has_node(&plan_after, "FlatMatchQuery:"),
        "the flat fallback must not outlive the fault: {plan_after}"
    );
}

/// A caller that names one index and asks for it to be remapped must hear
/// that the segment could not be classified, not a success log. `Withdrawn`
/// and `Unsupported` are verdicts about the segment and skipping is right for
/// them; an unreadable format marker is a fault on the store, and reporting
/// "queries are unaffected" for it hides a remap that silently did nothing
/// from a caller that can simply retry.
#[tokio::test]
async fn an_explicit_remap_reports_an_unverifiable_segment_as_an_error() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_explicit_remap_unverifiable", dir.as_str());
    let seed = tagged_with_fts_index(&path).await;
    let suffix = fts_metadata_suffix(&seed, FTS_INDEX).await;
    let (word, brute_force) = frequent_word_and_brute_force_count(&seed, "doc").await;
    drop(seed);

    // Retire the segment's provenance, so the remap has translation work and
    // the format marker is probed at all.
    {
        let mut writer = DatasetBuilder::from_uri(&path).load().await.unwrap();
        let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
        run_deferred_compaction(&mut writer, &options, None).await;
    }

    let (mut dataset, failing) = open_with_failing_reads(&path, suffix).await;
    let before = stored_segments(&dataset).await;
    let fts_before = coverage_of(&before, FTS_INDEX);
    assert!(!fts_before.is_empty());
    let segment_uuid = before
        .iter()
        .find(|(name, _, _)| name == FTS_INDEX)
        .map(|(_, uuid, _)| *uuid)
        .expect("the fixture must have an FTS segment");

    failing.store(true, Ordering::SeqCst);
    let error = crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect_err("an unverifiable segment must not be reported as a skipped success");
    let message = error.to_string();
    assert!(
        message.contains(&segment_uuid.to_string()),
        "the error must name the segment the caller asked about: {message}"
    );
    assert!(
        message.contains("format marker"),
        "the error must carry the read diagnostic, so the caller can tell a \
         fault from a verdict: {message}"
    );

    // Nothing was published while the probe was failing.
    assert_eq!(
        stored_segments(&dataset).await,
        before,
        "a refused remap must leave every segment's provenance untouched"
    );

    // The same remap, on the same handle, once reads recover.
    failing.store(false, Ordering::SeqCst);
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect("the retry must succeed once the marker is readable");
    let (count_after, plan_after) = fts_count_on_column(&dataset, &word, "doc").await;
    assert_eq!(
        count_after, brute_force,
        "the remapped index must answer exactly: {plan_after}"
    );
    assert!(
        plan_has_node(&plan_after, "MatchQuery:"),
        "the remapped segment must serve the query from the index: {plan_after}"
    );

    // A verdict ABOUT the segment is still a skip, not an error: the segment
    // the remap just rewrote needs no translation, so the second pass finds
    // nothing to do and says so by returning success.
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect("a segment with nothing to translate must still be a skip, not an error");
}

#[tokio::test]
async fn a_read_failure_defers_the_withdrawal_instead_of_failing_the_commit() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_read_failure_commit", dir.as_str());
    let seed = tagged_with_fts_index(&path).await;
    let suffix = fts_metadata_suffix(&seed, FTS_INDEX).await;
    let word = sample_word(&seed).await;
    let fts_before = coverage_of(&stored_segments(&seed).await, FTS_INDEX);
    assert!(!fts_before.is_empty());
    drop(seed);

    // A handle whose index cache has never probed this segment, so the fault
    // is what the probe meets.
    let (mut dataset, failing) = open_with_failing_reads(&path, suffix).await;
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let results = execute_tasks(&dataset, &plan, &options, None).await;
    assert!(!results.is_empty(), "the compaction must have a task");

    // With the read failing the selector cannot classify the segment, so the
    // commit lands and withdraws nothing.
    failing.store(true, Ordering::SeqCst);
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .expect("a marker read fault must not fail the compaction commit");
    assert_eq!(
        coverage_of(&stored_segments(&dataset).await, FTS_INDEX),
        fts_before,
        "an unclassifiable segment must keep its provenance; a withdrawal is \
         durable and must not rest on a transient fault"
    );

    // Nothing was cached from the fault: with reads restored the SAME handle
    // classifies this (healthy, deferrable) segment, so the selector lists
    // nothing to withdraw and the segment keeps translating.
    failing.store(false, Ordering::SeqCst);
    assert!(
        crate::index::frag_reuse::undeferrable_rewrite_segments(&dataset, &fts_before)
            .await
            .unwrap()
            .is_empty(),
        "a read fault must not be cached as the legacy verdict"
    );
    let destinations: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        derived_coverage(&dataset, FTS_INDEX).await,
        destinations,
        "the healthy segment must translate to the rewrite's outputs"
    );
    let (_, plan_after) = fts_count_on_column(&dataset, &word, "doc").await;
    assert!(
        !plan_after.contains("FlatMatchQuery"),
        "the retained segment must serve the query once reads succeed: \
         {plan_after}"
    );
}
/// crate-internal manifest writer the fragment-reuse reader tests use.
#[tokio::test]
async fn a_retired_source_reached_by_the_rewrite_is_withdrawn_one_hop_away() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_retired_provenance", dir.as_str());

    // (1) Four fragments, an FTS index and a BTree index, all three built
    // while every source fragment is live.
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col("doc", lance_datagen::array::random_sentence(1, 10, false))
        .into_dataset(
            &path,
            FragmentCount::from(4),
            FragmentRowCount::from(INDEX_ROWS_PER_FRAGMENT),
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["doc"],
            IndexType::Inverted,
            Some(FTS_INDEX.into()),
            &InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    create_btree_index(&mut dataset).await;
    let indexed_sources: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let fts_entry = dataset
        .load_index_by_name(FTS_INDEX)
        .await
        .unwrap()
        .expect("the FTS index must exist");
    assert_eq!(
        fts_entry.fragment_bitmap.as_ref().unwrap(),
        &indexed_sources,
        "the FTS segment must cover every source fragment"
    );
    let btree_stored_before = coverage_of(&stored_segments(&dataset).await, BTREE_INDEX);
    let word = sample_word(&dataset).await;
    let (_, plan_before) = fts_count_on_column(&dataset, &word, "doc").await;
    assert!(
        !plan_before.contains("FlatMatchQuery"),
        "the fixture's FTS index, not a flat scan, must serve the query: {plan_before}"
    );

    // One fragment of rows no index covers, appended BEFORE the partition so
    // its id precedes the destinations': the planner only bins ADJACENT
    // fragments, and step (4) narrows the plan to this one plus F1.
    append_unindexed_fragment(&mut dataset).await;
    let companion: u32 = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .max()
        .unwrap();

    // (2) A deferred (tagged) partition of P alone, through the real commit
    // path: P retires into the two destinations F1 and F2.
    let partitioned = dataset.fragments()[0].id;
    dataset = make_tagged_partitioning(dataset, &[partitioned]).await;
    let live_after_partition: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let destinations: Vec<u32> = live_after_partition
        .iter()
        .filter(|id| !indexed_sources.contains(*id) && *id != companion)
        .collect();
    assert_eq!(
        destinations.len(),
        2,
        "the partition must produce exactly two destinations, got {destinations:?}"
    );
    let (f1, f2) = (destinations[0], destinations[1]);
    assert_eq!(
        f1,
        companion + 1,
        "F1 must sit next to the companion fragment, or the planner bins neither"
    );
    assert!(
        !live_after_partition.contains(partitioned as u32),
        "P must be retired by the partition"
    );

    // (3) Reproduce the older build's manifest: the FTS segment stores
    // exactly {P}, a retired fragment with no live destination of its own,
    // while the BTree segment keeps the provenance a deferrable segment is
    // left with. Everything else, including the tagged entry, is untouched.
    let retired_only = RoaringBitmap::from_iter([partitioned as u32]);
    let older_build_indices: Vec<IndexMetadata> = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .map(|index| {
            let mut index = index.clone();
            if index.name == FTS_INDEX {
                index.fragment_bitmap = Some(retired_only.clone());
            }
            index
        })
        .collect();
    crate::index::frag_reuse_reader::tests::persist_fixture(&mut dataset, older_build_indices)
        .await;
    // The segment must be undeferrable for the withdrawal to apply at all;
    // a missing inverted metadata file is the legacy verdict (see
    // `a_missing_inverted_metadata_file_is_the_legacy_verdict`).
    let fts_uuid = dataset
        .load_index_by_name(FTS_INDEX)
        .await
        .unwrap()
        .unwrap()
        .uuid;
    let metadata_path = dataset
        .indices_dir()
        .join(fts_uuid.to_string())
        .join(lance_index::scalar::inverted::METADATA_FILE);
    dataset.object_store.delete(&metadata_path).await.unwrap();

    let mut dataset = DatasetBuilder::from_uri(&path).load().await.unwrap();
    let stored = stored_segments(&dataset).await;
    assert_eq!(
        coverage_of(&stored, FTS_INDEX),
        retired_only,
        "the fixture must leave the FTS segment covering the retired source only"
    );
    assert_eq!(
        coverage_of(&stored, BTREE_INDEX),
        btree_stored_before,
        "the deferrable segment's stored provenance is untouched by the partition"
    );
    assert_eq!(
        derived_coverage(&dataset, BTREE_INDEX).await,
        &live_after_partition - RoaringBitmap::from_iter([companion]),
        "the BTree must translate to every live fragment it covers"
    );

    // A handle opened and queried BEFORE the rewrite, to prove the
    // withdrawal is observed through `checkout_latest` rather than served
    // from this handle's warmed caches.
    let mut warmed = DatasetBuilder::from_uri(&path).load().await.unwrap();
    let (count_before, _) = fts_count_on_column(&warmed, &word, "doc").await;
    assert!(count_before > 0, "the word must match some row");

    // (4) Rewrite F1 only, through the real deferred compaction commit path:
    // every other live fragment is held out, so the one task is F1 with the
    // uncovered companion. The rewrite never names P.
    let options = CompactionOptions {
        excluded_fragment_ids: dataset
            .fragments()
            .iter()
            .map(|fragment| fragment.id as u32)
            .filter(|id| *id != f1 && *id != companion)
            .collect(),
        ..deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize)
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    let rewritten: RoaringBitmap = plan
        .tasks()
        .iter()
        .flat_map(|task| task.fragments.iter())
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        rewritten,
        RoaringBitmap::from_iter([companion, f1]),
        "the compaction must rewrite F1 and the uncovered companion only"
    );
    assert!(
        !rewritten.contains(partitioned as u32),
        "the rewrite must never name the retired source P itself"
    );
    let committed = run_deferred_compaction(&mut dataset, &options, None).await;
    assert!(committed > 0, "the compaction must commit");

    // (5) The retired source P is gone from the undeferrable segment, even
    // though the rewrite never named it.
    let after = stored_segments(&dataset).await;
    let fts_after = coverage_of(&after, FTS_INDEX);
    assert!(
        !fts_after.contains(partitioned as u32),
        "the undeferrable segment still covers the retired source {partitioned} \
         the rewrite reached through the history: {:?}",
        fts_after.iter().collect::<Vec<_>>()
    );
    // Withdrawing P ends this segment's acceleration of the sibling
    // destination F2 as well: its remaining coverage would have been the
    // live-only remainder the helper clears, so the bitmap is empty.
    assert!(
        fts_after.is_empty(),
        "the withdrawal must clear the segment whole: {:?}",
        fts_after.iter().collect::<Vec<_>>()
    );
    assert!(
        after.iter().any(|(name, _, _)| name == FTS_INDEX),
        "the FTS definition must survive so a later optimize can rebuild it"
    );
    assert!(
        derived_coverage(&dataset, FTS_INDEX).await.is_empty(),
        "the withdrawn segment must derive no coverage"
    );

    // Full-text search is still exact, now served by a flat scan: a rewrite
    // moves rows, it never adds or drops any.
    let (count_after, plan_after) = fts_count_on_column(&dataset, &word, "doc").await;
    assert!(
        plan_after.contains("FlatMatchQuery"),
        "a withdrawn segment must not serve the full-text query; it must fall \
         back to a flat scan: {plan_after}"
    );
    assert_eq!(
        count_after, count_before,
        "full-text search must stay exact after the withdrawal: {plan_after}"
    );

    // The deferrable control keeps its translated coverage of the untouched
    // destination F2 and still serves a filtered query.
    let live_after_rewrite: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert!(
        live_after_rewrite.contains(f2),
        "F2 must be untouched by the rewrite"
    );
    assert_eq!(
        coverage_of(&after, BTREE_INDEX),
        btree_stored_before,
        "a deferrable segment's stored provenance is never withdrawn"
    );
    let btree_derived = derived_coverage(&dataset, BTREE_INDEX).await;
    assert!(
        btree_derived.contains(f2),
        "the BTree must keep translating to the untouched destination F2: {:?}",
        btree_derived.iter().collect::<Vec<_>>()
    );
    let mut lookup = dataset.scan();
    lookup.filter("i = 1").unwrap().project(&["i"]).unwrap();
    let lookup_plan = lookup.explain_plan(false).await.unwrap();
    assert!(lookup_plan.contains("ScalarIndexQuery"), "{lookup_plan}");
    let batch = lookup.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch["i"].as_primitive::<Int32Type>().value(0), 1);

    // The handle that was warm before the rewrite sees the same withdrawal.
    warmed.checkout_latest().await.unwrap();
    assert_eq!(
        coverage_of(&stored_segments(&warmed).await, FTS_INDEX),
        fts_after,
        "a warmed handle must not resurrect the withdrawn coverage"
    );
    assert!(derived_coverage(&warmed, FTS_INDEX).await.is_empty());
    let (warmed_after, warmed_plan) = fts_count_on_column(&warmed, &word, "doc").await;
    assert!(warmed_plan.contains("FlatMatchQuery"), "{warmed_plan}");
    assert_eq!(warmed_after, count_after);
}

/// [`make_tagged`] over a chosen set of source fragments, so a test can
/// partition exactly one fragment into its two destinations.
async fn make_tagged_partitioning(mut dataset: Dataset, source_ids: &[u64]) -> Dataset {
    let reserve = Transaction::new(
        dataset.manifest.version,
        Operation::ReserveFragments { num_fragments: 2 },
        None,
    );
    dataset
        .apply_commit(reserve, &Default::default(), &Default::default())
        .await
        .unwrap();
    let dest_base_id = dataset.manifest.max_fragment_id().unwrap() - 1;
    let old_fragments: Vec<Fragment> = dataset
        .fragments()
        .iter()
        .filter(|fragment| source_ids.contains(&fragment.id))
        .cloned()
        .collect();
    assert_eq!(old_fragments.len(), source_ids.len());
    let (transition, new_fragments) = prepare_partition(&dataset, source_ids, dest_base_id).await;
    let read_version = dataset.manifest.version;
    let frag_reuse_index = Some(
        frag_reuse_entry_appending(&dataset, vec![transition])
            .await
            .unwrap(),
    );
    let dataset = CommitBuilder::new(Arc::new(dataset))
        .execute(Transaction::new(
            read_version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments,
                    new_fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index,
            },
            None,
        ))
        .await
        .unwrap();
    let stored = load_all_indices(&dataset).await.unwrap();
    assert!(
        uses_tagged_fri(
            &dataset.manifest,
            stored
                .iter()
                .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        ),
        "the stable partition must make the table tagged"
    );
    dataset
}

/// Appends one fragment of fresh rows no index covers, so a compaction plan
/// can be narrowed to one indexed fragment plus this one.
async fn append_unindexed_fragment(dataset: &mut Dataset) {
    let batch = {
        let mut scan = dataset.scan();
        scan.project(&["i", "doc"]).unwrap();
        scan.limit(Some(INDEX_ROWS_PER_FRAGMENT as i64), None)
            .unwrap();
        scan.try_into_batch().await.unwrap()
    };
    let appended = RecordBatch::try_new(
        batch.schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(
                10_000..10_000 + batch.num_rows() as i32,
            )) as ArrayRef,
            batch["doc"].clone(),
        ],
    )
    .unwrap();
    let schema = appended.schema();
    dataset
        .append(
            RecordBatchIterator::new([Ok(appended)], schema),
            Some(WriteParams {
                max_rows_per_file: INDEX_ROWS_PER_FRAGMENT as usize,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
}

/// A staged full-text build may be committed straight through
/// `commit_existing_index_segments`, without the finalizing merge. The replay
/// carries no recorded file list, so a layout decision taken from that list
/// alone reads the absent root metadata as the legacy layout and withdraws the
/// segment's coverage: the query then falls back to a flat scan whose bare
/// tokenizer never reaches the configured stemmer, silently missing every row
/// the index would have matched.
#[rstest::rstest]
#[case::late_merge(false, false)]
#[case::late_direct_commit(false, true)]
#[case::early_finalized_control(true, false)]
#[tokio::test]
async fn staged_fts_preserves_stemming_with_direct_commit(
    #[case] finalize_first: bool,
    #[case] direct_commit: bool,
) {
    let dir = TempStrDir::default();
    let path = dir.as_str();
    let dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "vector",
            lance_datagen::array::rand_vec::<Float32Type>(4.into()),
        )
        .col(
            "doc",
            lance_datagen::array::fill_utf8("running".to_string()),
        )
        .into_dataset(
            path,
            FragmentCount::from(4),
            FragmentRowCount::from(INDEX_ROWS_PER_FRAGMENT),
        )
        .await
        .unwrap();
    let mut dataset = make_tagged(dataset).await;
    create_btree_index(&mut dataset).await;
    let sources: Vec<u32> = dataset.fragment_bitmap.iter().collect();
    let params = InvertedIndexParams::default().stem(true);
    let mut staged = CreateIndexBuilder::new(&mut dataset, &["doc"], IndexType::Inverted, &params)
        .name(FTS_INDEX.to_string())
        .fragments(sources)
        .execute_uncommitted()
        .await
        .unwrap();
    // The staged build has no published layout yet, so it is never selected
    // for an undeferrable withdrawal, and asking twice must not memoize a
    // verdict from the absent root metadata.
    for _ in 0..2 {
        assert!(
            crate::index::frag_reuse::undeferrable_segments_covering(
                &dataset,
                std::iter::once(&staged),
                dataset.fragment_bitmap.as_ref(),
                None,
                dataset.fragment_bitmap.as_ref(),
            )
            .await
            .unwrap()
            .is_empty()
        );
    }
    if finalize_first {
        staged = dataset
            .merge_existing_index_segments(vec![staged])
            .await
            .unwrap();
    }
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    run_deferred_compaction(&mut dataset, &options, None).await;
    let destinations = dataset.fragment_bitmap.as_ref().clone();
    let merged = if direct_commit {
        staged
    } else {
        dataset
            .merge_existing_index_segments(vec![staged])
            .await
            .unwrap()
    };
    dataset
        .commit_existing_index_segments(FTS_INDEX, "doc", vec![merged])
        .await
        .unwrap();
    let fresh = DatasetBuilder::from_uri(path).load().await.unwrap();
    for handle in [&dataset, &fresh] {
        let (count, plan) = fts_count_on_column(handle, "run", "doc").await;
        assert_eq!(
            count,
            4 * INDEX_ROWS_PER_FRAGMENT as usize,
            "configured stemming must survive staged replay: {plan}"
        );
        assert!(plan_has_node(&plan, "MatchQuery:"), "{plan}");
        assert!(!plan_has_node(&plan, "FlatMatchQuery:"), "{plan}");
        assert!(!coverage_of(&stored_segments(handle).await, FTS_INDEX).is_empty());
        assert_eq!(derived_coverage(handle, FTS_INDEX).await, destinations);
    }
}

/// A finalize interrupted after copying some partitions, before it publishes
/// the root metadata, leaves root files beside the staging directory. Reading
/// that layout as legacy caches the verdict, so the retried direct commit
/// withdraws the segment's compacted coverage and the query falls back to a
/// flat scan whose bare tokenizer never reaches the configured stemmer.
#[derive(Debug)]
struct StopBeforeFtsRootMetadata;

#[async_trait::async_trait]
impl lance_index::progress::IndexBuildProgress for StopBeforeFtsRootMetadata {
    async fn stage_start(&self, stage: &str, _: Option<u64>, _: &str) -> lance_core::Result<()> {
        if stage == "write_merged_metadata" {
            return Err(lance_core::Error::io("interrupted before root metadata"));
        }
        Ok(())
    }
    async fn stage_progress(&self, _: &str, _: u64) -> lance_core::Result<()> {
        Ok(())
    }
    async fn stage_complete(&self, _: &str) -> lance_core::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn direct_staged_fts_retry_after_interrupted_finalize() {
    let dir = TempStrDir::default();
    let path = dir.as_str();
    let dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "vector",
            lance_datagen::array::rand_vec::<Float32Type>(4.into()),
        )
        .col(
            "doc",
            lance_datagen::array::fill_utf8("running".to_string()),
        )
        .into_dataset(
            path,
            FragmentCount::from(4),
            FragmentRowCount::from(INDEX_ROWS_PER_FRAGMENT),
        )
        .await
        .unwrap();
    let mut dataset = make_tagged(dataset).await;
    create_btree_index(&mut dataset).await;
    let sources: Vec<u32> = dataset.fragment_bitmap.iter().collect();
    let params = InvertedIndexParams::default().stem(true);
    let staged = CreateIndexBuilder::new(&mut dataset, &["doc"], IndexType::Inverted, &params)
        .name(FTS_INDEX.to_string())
        .fragments(sources)
        .execute_uncommitted()
        .await
        .unwrap();
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    run_deferred_compaction(&mut dataset, &options, None).await;

    let index_dir = dataset.indices_dir().join(staged.uuid.to_string());
    let store = <lance_index::scalar::lance_format::LanceIndexStore as crate::dataset::index::LanceIndexStoreExt>::from_dataset_for_new(
        &dataset,
        &staged.uuid,
    )
    .unwrap();
    let error = lance_index::scalar::inverted::builder::merge_index_files_with_options(
        dataset.object_store.as_ref(),
        &index_dir,
        Arc::new(store),
        lance_index::scalar::inverted::builder::FinalizeOptions::default(),
        Arc::new(StopBeforeFtsRootMetadata),
    )
    .await
    .expect_err("the finalize must stop before publishing its root marker");
    assert!(
        error
            .to_string()
            .contains("interrupted before root metadata")
    );
    let files = lance_table::format::list_index_files_with_sizes(&dataset.object_store, &index_dir)
        .await
        .unwrap();
    assert!(files.iter().any(|f| f.path.starts_with("staging/")));
    assert!(files.iter().any(|f| !f.path.starts_with("staging/")));
    assert!(
        !files
            .iter()
            .any(|f| f.path == lance_index::scalar::inverted::METADATA_FILE)
    );

    dataset
        .commit_existing_index_segments(FTS_INDEX, "doc", vec![staged])
        .await
        .unwrap();
    let fresh = DatasetBuilder::from_uri(path).load().await.unwrap();
    let (same_count, same_plan) = fts_count_on_column(&dataset, "run", "doc").await;
    let (fresh_count, fresh_plan) = fts_count_on_column(&fresh, "run", "doc").await;
    assert_eq!(
        (same_count, fresh_count),
        (
            4 * INDEX_ROWS_PER_FRAGMENT as usize,
            4 * INDEX_ROWS_PER_FRAGMENT as usize
        ),
        "retry must preserve stemming on both handles:\n{same_plan}\n{fresh_plan}"
    );
    for plan in [same_plan, fresh_plan] {
        assert!(plan_has_node(&plan, "MatchQuery:"), "{plan}");
        assert!(!plan_has_node(&plan, "FlatMatchQuery:"), "{plan}");
    }
}

/// Commits a second segment of `FTS_INDEX` over everything live, as index
/// maintenance does: built uncommitted under its own name, then renamed into
/// the logical index.
async fn commit_fts_delta_segment(dataset: &mut Dataset) -> Uuid {
    let params = InvertedIndexParams::default();
    let mut delta = CreateIndexBuilder::new(dataset, &["doc"], IndexType::Inverted, &params)
        .name("doc_idx_delta".to_string())
        .execute_uncommitted()
        .await
        .unwrap();
    delta.name = FTS_INDEX.to_string();
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

/// Pruning superseded segments reads each segment's coverage through the
/// translating listing. A segment whose format marker cannot be read leaves
/// that listing without its translated coverage, so its valid coverage looks
/// empty, the subset check passes and the prune COMMITS its removal -- losing
/// a healthy segment to a fault that says nothing about it. The prune must
/// refuse while the listing cannot be verified, and prune once it can. The
/// refusal is scoped to the logical index, so a registered healthy sibling of
/// the unverifiable segment is refused with it, and resolves once reads do.
/// The remap here meets the unreadable segment itself first, so it is the
/// pre-existing per-segment refusal control; the sibling orientation, where
/// the unreadable segment is visited second, is
/// `an_explicit_remap_refuses_a_readable_segment_beside_an_unreadable_sibling`.
#[tokio::test]
async fn a_read_failure_refuses_to_prune_a_superseded_segment() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_prune_read_failure", dir.as_str());
    let seed = tagged_with_fts_index(&path).await;
    let suffix = fts_metadata_suffix(&seed, FTS_INDEX).await;
    let (word, brute_force) = frequent_word_and_brute_force_count(&seed, "doc").await;
    assert!(
        brute_force > 0,
        "the chosen word {word} must match rows in the fixture"
    );
    drop(seed);

    // The compaction retires the original segment's provenance, so its only
    // valid coverage is what the ledger translates; the delta then covers
    // every destination directly, making the original superseded.
    let mut writer = DatasetBuilder::from_uri(&path).load().await.unwrap();
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    run_deferred_compaction(&mut writer, &options, None).await;
    let delta_uuid = commit_fts_delta_segment(&mut writer).await;
    let destinations: RoaringBitmap = writer
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    drop(writer);

    // A handle whose index cache has never seen a successful probe, so the
    // fault is what the prune's own listing meets.
    let (mut dataset, failing) = open_with_failing_reads(&path, suffix).await;
    let before = stored_segments(&dataset).await;
    assert_eq!(
        before
            .iter()
            .filter(|(name, _, _)| name == FTS_INDEX)
            .count(),
        2,
        "the fixture must hold a superseded segment and its replacement"
    );
    // The superseded segment, as the manifest registers it: the one the
    // delta did not replace.
    let superseded: IndexMetadata = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .find(|index| index.name == FTS_INDEX && index.uuid != delta_uuid)
        .expect("the fixture must hold the superseded segment")
        .clone();

    failing.store(true, Ordering::SeqCst);
    let error = crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
        .await
        .expect_err("an unverifiable listing must not justify removing a segment");
    let message = error.to_string();
    assert!(
        message.contains(FTS_INDEX) && message.contains("could not be read"),
        "the error must name the index whose file could not be read: {message}"
    );
    // The other mutation site reading the same listing: the coverage the
    // optimizer replaces a segment by. An unreadable marker must refuse here
    // too, rather than report the segment as covering nothing.
    let coverage_error =
        crate::index::append::tagged_segment_coverage(&dataset, &[&superseded], None)
            .await
            .expect_err("an unverifiable listing must not report empty coverage");
    let coverage_message = coverage_error.to_string();
    assert!(
        coverage_message.contains(FTS_INDEX) && coverage_message.contains("could not be read"),
        "the error must name the index whose file could not be read: {coverage_message}"
    );
    // The refusal is scoped to the logical index, not to the unreadable file:
    // the healthy delta segment reads fine on its own, but it shares the
    // sibling group whose coverage decision cannot be verified.
    let healthy_sibling: IndexMetadata = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .find(|index| index.name == FTS_INDEX && index.uuid == delta_uuid)
        .expect("the fixture must hold the healthy delta segment")
        .clone();
    let sibling_error =
        crate::index::append::tagged_segment_coverage(&dataset, &[&healthy_sibling], None)
            .await
            .expect_err(
                "a healthy sibling of an unverifiable segment shares its logical \
                 group, so its coverage must be refused too",
            );
    let sibling_message = sibling_error.to_string();
    assert!(
        sibling_message.contains(FTS_INDEX) && sibling_message.contains("could not be read"),
        "the error must name the index whose file could not be read: {sibling_message}"
    );
    // Manifest order puts the unreadable segment first here, so the remap
    // meets it directly: the pre-existing per-segment refusal. The sibling
    // orientation is covered by
    // `an_explicit_remap_refuses_a_readable_segment_beside_an_unreadable_sibling`.
    let remap_error = crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect_err("an explicit remap must not publish a sibling of an unverifiable segment");
    assert!(
        remap_error.to_string().contains(FTS_INDEX),
        "the error must name the index that could not be verified: {remap_error}"
    );
    dataset.checkout_latest().await.unwrap();
    assert_eq!(
        stored_segments(&dataset).await,
        before,
        "a refused prune or remap must leave the manifest's index list untouched"
    );

    // Once reads recover the same call answers instead of refusing: the
    // segment is reported as covering nothing because the delta covers every
    // destination DIRECTLY (direct coverage wins), which is the verdict that
    // makes it superseded -- and is exactly what an unreadable marker must
    // not be mistaken for.
    failing.store(false, Ordering::SeqCst);
    let coverage = crate::index::append::tagged_segment_coverage(&dataset, &[&superseded], None)
        .await
        .expect("the coverage must resolve once the marker is readable")
        .expect("a tagged table must report derived coverage");
    assert!(
        coverage.contains_key(&superseded.uuid),
        "every queried segment must be accounted for: {coverage:?}"
    );
    let sibling_coverage =
        crate::index::append::tagged_segment_coverage(&dataset, &[&healthy_sibling], None)
            .await
            .expect("the sibling's coverage must resolve once the marker is readable")
            .expect("a tagged table must report derived coverage");
    assert!(
        sibling_coverage.contains_key(&healthy_sibling.uuid),
        "every queried segment must be accounted for: {sibling_coverage:?}"
    );
    assert_eq!(
        derived_coverage(&dataset, FTS_INDEX).await,
        destinations,
        "the logical index still covers every destination through the delta"
    );

    // And the remap the fault refused succeeds, publishing the sibling it
    // could not verify before.
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect("the remap must succeed once the marker is readable");

    // The same prune, once reads recover, removes the superseded segment.
    crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
        .await
        .expect("the prune must succeed once the marker is readable");
    let after = stored_segments(&dataset).await;
    assert_eq!(
        after
            .iter()
            .filter(|(name, _, _)| name == FTS_INDEX)
            .count(),
        1,
        "the superseded segment must be pruned once its coverage is verifiable"
    );
    let (count_after, plan_after) = fts_count_on_column(&dataset, &word, "doc").await;
    assert_eq!(
        count_after, brute_force,
        "the pruned index must still answer exactly: {plan_after}"
    );
    assert!(
        plan_has_node(&plan_after, "MatchQuery:"),
        "the surviving segment must serve the query from the index: {plan_after}"
    );
    assert!(
        !plan_has_node(&plan_after, "FlatMatchQuery:"),
        "every destination must stay covered after the prune: {plan_after}"
    );
}

/// The explicit remap classifies and commits one segment at a time, in
/// manifest order. When the segment whose index file cannot be read is the
/// NEWER directly covering delta, the older translating segment is visited
/// first and, classified from a listing that lacks the delta's direct
/// coverage, would be translated and published over destinations the delta
/// already covers: after reads recover every match is answered twice. The
/// remap must refuse the whole logical index before publishing anything,
/// and answer exactly, on the same and on a fresh handle, once it can verify.
#[tokio::test]
async fn an_explicit_remap_refuses_a_readable_segment_beside_an_unreadable_sibling() {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_remap_unreadable_sibling", dir.as_str());
    let seed = tagged_with_fts_index(&path).await;
    let (word, brute_force) = frequent_word_and_brute_force_count(&seed, "doc").await;
    assert!(
        brute_force > 0,
        "the chosen word {word} must match rows in the fixture"
    );
    drop(seed);

    // The compaction retires the original segment's provenance, so it only
    // covers the destinations the ledger translates for it; the delta then
    // covers every destination directly.
    let mut writer = DatasetBuilder::from_uri(&path).load().await.unwrap();
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    run_deferred_compaction(&mut writer, &options, None).await;
    let delta_uuid = commit_fts_delta_segment(&mut writer).await;
    drop(writer);

    // The fault is on the DELTA: the segment the remap reaches second.
    let suffix = format!(
        "{delta_uuid}/{}",
        lance_index::scalar::inverted::METADATA_FILE
    );
    let (mut dataset, failing) = open_with_failing_reads(&path, suffix).await;
    let before = stored_segments(&dataset).await;
    assert_eq!(
        before
            .iter()
            .filter(|(name, _, _)| name == FTS_INDEX)
            .count(),
        2,
        "the fixture must hold a translating segment and its directly covering delta"
    );
    let registered: Vec<IndexMetadata> = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .filter(|index| index.name == FTS_INDEX)
        .cloned()
        .collect();
    assert_ne!(
        registered[0].uuid, delta_uuid,
        "the fixture relies on the older translating segment being visited \
         first, before the unreadable delta"
    );

    failing.store(true, Ordering::SeqCst);
    let remap_error = crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect_err(
        "an explicit remap must not publish a readable segment while its \
         sibling's coverage cannot be verified",
    );
    let remap_message = remap_error.to_string();
    assert!(
        remap_message.contains(FTS_INDEX) && remap_message.contains("could not"),
        "the error must name the index whose coverage could not be verified: {remap_message}"
    );
    dataset.checkout_latest().await.unwrap();
    assert_eq!(
        stored_segments(&dataset).await,
        before,
        "a refused remap must publish nothing: the older segment must not be \
         replaced over destinations the delta covers directly"
    );

    // The same handle, once reads recover: nothing of the fault is cached.
    failing.store(false, Ordering::SeqCst);
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect("the remap must succeed once every sibling is readable");
    let destinations: RoaringBitmap = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        derived_coverage(&dataset, FTS_INDEX).await,
        destinations,
        "the logical index must cover every destination after the remap"
    );
    let (count, plan) = fts_count_on_column(&dataset, &word, "doc").await;
    assert_eq!(
        count, brute_force,
        "the logical index must answer exactly after the remap: {plan}"
    );
    assert!(
        plan_has_node(&plan, "MatchQuery:"),
        "the remapped segments must serve the query from the index: {plan}"
    );
    assert!(
        !plan_has_node(&plan, "FlatMatchQuery:"),
        "every destination must stay covered after the remap: {plan}"
    );

    let fresh = DatasetBuilder::from_uri(&path).load().await.unwrap();
    let (fresh_count, fresh_plan) = fts_count_on_column(&fresh, &word, "doc").await;
    assert_eq!(
        fresh_count, brute_force,
        "a reopened handle must answer exactly after the remap: {fresh_plan}"
    );
    assert!(
        plan_has_node(&fresh_plan, "MatchQuery:"),
        "a reopened handle must serve the query from the index: {fresh_plan}"
    );
    assert!(
        !plan_has_node(&fresh_plan, "FlatMatchQuery:"),
        "a reopened handle must find every destination covered: {fresh_plan}"
    );
}

/// A coverage decision for one logical index is decided within that index's
/// own sibling group (direct-coverage-wins, sibling exclusions), so an
/// unreadable segment of ANOTHER index says nothing about it. Maintenance of
/// the healthy index must proceed while the faulting one refuses, and the
/// table-wide prune stays fail-closed. Nothing is cached across the fault:
/// the same handle recovers without a reopen.
#[tokio::test]
async fn an_unreadable_index_does_not_block_maintenance_of_another_index() {
    let dir = TempStrDir::default();
    let path = format!("{}/two_indices_one_unreadable", dir.as_str());
    let mut seed = tagged_with_fts_index(path.as_str()).await;
    create_btree_index(&mut seed).await;
    // Both logical indices now carry translating coverage, so both go through
    // the derived listing the fault degrades.
    let options = deferred_options(4 * INDEX_ROWS_PER_FRAGMENT as usize);
    run_deferred_compaction(&mut seed, &options, None).await;
    let suffix = fts_metadata_suffix(&seed, FTS_INDEX).await;
    let (word, brute_force) = frequent_word_and_brute_force_count(&seed, "doc").await;
    assert!(
        brute_force > 0,
        "the chosen word {word} must match rows in the fixture"
    );
    drop(seed);

    // A handle whose index cache has never seen a successful probe of the FTS
    // segment, so the fault is what every decision below meets.
    let (mut dataset, failing) = open_with_failing_reads(&path, suffix).await;
    let segments = load_all_indices(&dataset).await.unwrap();
    let btree_before: IndexMetadata = segments
        .iter()
        .find(|index| index.name == BTREE_INDEX)
        .expect("the fixture must hold a BTree segment")
        .clone();
    let fts_segment: IndexMetadata = segments
        .iter()
        .find(|index| index.name == FTS_INDEX)
        .expect("the fixture must hold an FTS segment")
        .clone();

    failing.store(true, Ordering::SeqCst);
    // (a) The BTree index's own segments were all verified, so its remap
    // proceeds and the index still serves a filtered query.
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["i"],
        Some(BTREE_INDEX.into()),
    )
    .await
    .expect("an unreadable FTS segment must not block the BTree index's remap");
    let btree_segment: IndexMetadata = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .find(|index| index.name == BTREE_INDEX)
        .expect("the BTree index must survive its own remap")
        .clone();
    assert_ne!(
        btree_segment.uuid, btree_before.uuid,
        "the BTree remap must publish a new segment, not skip"
    );
    let mut lookup = dataset.scan();
    lookup.filter("i = 1").unwrap().project(&["i"]).unwrap();
    let lookup_plan = lookup.explain_plan(false).await.unwrap();
    assert!(
        lookup_plan.contains("ScalarIndexQuery"),
        "the BTree index must still serve the filtered query: {lookup_plan}"
    );
    let looked_up = lookup.try_into_batch().await.unwrap();
    assert_eq!(looked_up.num_rows(), 1, "{lookup_plan}");
    assert_eq!(
        looked_up["i"].as_primitive::<Int32Type>().value(0),
        1,
        "{lookup_plan}"
    );

    // (b) The same listing answers the BTree segment's coverage.
    let coverage = crate::index::append::tagged_segment_coverage(&dataset, &[&btree_segment], None)
        .await
        .expect("a verified index's coverage must resolve during another index's fault")
        .expect("a tagged table must report derived coverage");
    assert!(
        coverage.contains_key(&btree_segment.uuid),
        "every queried segment must be accounted for: {coverage:?}"
    );

    let before = stored_segments(&dataset).await;
    // (c)/(d) The decision that DOES depend on the unreadable segment refuses,
    // naming the index the caller asked about.
    let remap_error = crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect_err("an unreadable segment must not be reported as a skipped success");
    assert!(
        remap_error.to_string().contains(FTS_INDEX),
        "the error must name the index that could not be verified: {remap_error}"
    );
    let coverage_error =
        crate::index::append::tagged_segment_coverage(&dataset, &[&fts_segment], None)
            .await
            .expect_err("an unverifiable segment must not report empty coverage");
    assert!(
        coverage_error.to_string().contains(FTS_INDEX),
        "the error must name the index that could not be verified: {coverage_error}"
    );

    // (e) The prune commits one table-wide decision, so any uncertain group
    // refuses it and the manifest's index list is untouched.
    let prune_error = crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
        .await
        .expect_err("an unverifiable group must not justify a table-wide prune");
    assert!(
        prune_error.to_string().contains(FTS_INDEX),
        "the error must name the index that could not be verified: {prune_error}"
    );
    dataset.checkout_latest().await.unwrap();
    assert_eq!(
        stored_segments(&dataset).await,
        before,
        "a refused prune must leave the manifest's index list untouched"
    );

    // Nothing about the fault was cached: the SAME handle (no reopen) recovers
    // every refused decision once reads succeed.
    failing.store(false, Ordering::SeqCst);
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .expect("the remap must succeed once the marker is readable");
    let fts_segment: IndexMetadata = load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .find(|index| index.name == FTS_INDEX)
        .expect("the fixture must hold an FTS segment")
        .clone();
    let fts_coverage =
        crate::index::append::tagged_segment_coverage(&dataset, &[&fts_segment], None)
            .await
            .expect("the coverage must resolve once the marker is readable")
            .expect("a tagged table must report derived coverage");
    assert!(
        fts_coverage.contains_key(&fts_segment.uuid),
        "every queried segment must be accounted for: {fts_coverage:?}"
    );
    let (count_after, plan_after) = fts_count_on_column(&dataset, &word, "doc").await;
    assert_eq!(
        count_after, brute_force,
        "the recovered index must answer exactly: {plan_after}"
    );
    assert!(
        plan_has_node(&plan_after, "MatchQuery:"),
        "the re-probed segment must serve the query from the index: {plan_after}"
    );
    assert!(
        !plan_has_node(&plan_after, "FlatMatchQuery:"),
        "every destination must stay covered: {plan_after}"
    );
    crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
        .await
        .expect("the prune must succeed once the marker is readable");
}

/// An attempt window over reads of one suffix: the first `start` matching
/// reads succeed, the next [`PHASED_FAULT_READS`] fail, and the rest succeed.
/// Unlike [`BudgetedFailingReadWrapper`], whose budget always starts at the
/// first matching read, this places the fault at a later probe of the same
/// marker so an earlier probe answers and a later one does not.
#[derive(Debug)]
struct PhasedFailingReadWrapper {
    suffix: String,
    /// Counts every matching read, so a test can report how far the window got.
    attempts: Arc<AtomicUsize>,
    start: usize,
}

impl lance_io::object_store::WrappingObjectStore for PhasedFailingReadWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(PhasedFailingReadStore {
            inner: original,
            suffix: self.suffix.clone(),
            attempts: self.attempts.clone(),
            start: self.start,
        })
    }

    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
}

#[derive(Debug)]
struct PhasedFailingReadStore {
    inner: Arc<dyn object_store::ObjectStore>,
    suffix: String,
    attempts: Arc<AtomicUsize>,
    start: usize,
}

impl PhasedFailingReadStore {
    /// Counts one attempt when `location` matches, and fails it when that
    /// attempt falls inside the window.
    fn fails(&self, location: &object_store::path::Path) -> bool {
        if !location.as_ref().ends_with(&self.suffix) {
            return false;
        }
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        (self.start..self.start + PHASED_FAULT_READS).contains(&attempt)
    }
}

impl std::fmt::Display for PhasedFailingReadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PhasedFailingReadStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for PhasedFailingReadStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &object_store::path::Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &object_store::path::Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        if self.fails(location) {
            return Err(FailingReadStore::injected_error());
        }
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &object_store::path::Path,
        ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<bytes::Bytes>> {
        if self.fails(location) {
            return Err(FailingReadStore::injected_error());
        }
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<
            'static,
            object_store::Result<object_store::path::Path>,
        >,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&object_store::path::Path>,
        offset: &object_store::path::Path,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, opts).await
    }

    async fn rename_opts(
        &self,
        from: &object_store::path::Path,
        to: &object_store::path::Path,
        opts: object_store::RenameOptions,
    ) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, opts).await
    }
}

/// How many consecutive marker reads [`PhasedFailingReadWrapper`] fails: one
/// outage long enough to span the narrowing step's own derivation.
const PHASED_FAULT_READS: usize = 4;

/// The attempt the fault starts at: the guard's verification of the listing
/// answers, and the narrowing that follows meets the outage.
const PHASED_FAULT_START: usize = 2;

/// A `start` past every marker read the operation makes, so the wrapper is
/// installed but never fires: the healthy control of the same fixture.
const PHASED_FAULT_NEVER: usize = 1000;

/// Builds the partial-coverage fixture at `path` and returns the delta's uuid
/// and the word/brute-force pair to assert against.
///
/// The compaction leaves TWO destinations and the finalized delta covers only
/// ONE of them. That asymmetry is what the narrowing under test needs: the
/// older segment still translates provenance into the other destination, so
/// it keeps nonempty derived coverage and stays eligible for the remap
/// instead of being skipped as fully superseded.
async fn partially_covered_fts_delta(path: &str) -> (Uuid, String, usize) {
    let seed = tagged_with_fts_index(path).await;
    let (word, brute_force) = frequent_word_and_brute_force_count(&seed, "doc").await;
    assert!(brute_force > 0, "the word {word} must match rows");
    drop(seed);

    let mut writer = DatasetBuilder::from_uri(path).load().await.unwrap();
    let options = deferred_options(2 * INDEX_ROWS_PER_FRAGMENT as usize);
    run_deferred_compaction(&mut writer, &options, None).await;
    let destinations: Vec<u32> = writer
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    assert_eq!(
        destinations.len(),
        2,
        "the fixture needs two destinations so the delta can cover one: {destinations:?}"
    );

    // Built over one destination only, and finalized (root metadata written)
    // before registration, so the reader recognizes it as a directly covering
    // sibling rather than an unpublished staging layout.
    let params = InvertedIndexParams::default();
    let staged = CreateIndexBuilder::new(&mut writer, &["doc"], IndexType::Inverted, &params)
        .name("doc_idx_partial_delta".to_string())
        .fragments(vec![destinations[0]])
        .execute_uncommitted()
        .await
        .unwrap();
    let mut delta = writer
        .merge_existing_index_segments(vec![staged])
        .await
        .unwrap();
    assert!(
        delta
            .files
            .as_ref()
            .unwrap()
            .iter()
            .any(|file| file.path == lance_index::scalar::inverted::METADATA_FILE),
        "the fixture delta must be finalized before registration"
    );
    delta.name = FTS_INDEX.to_string();
    let delta_uuid = delta.uuid;
    writer
        .apply_commit(
            Transaction::new(
                writer.manifest.version,
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
    drop(writer);

    // A plain handle, no fault and no cache surgery: the premise the faulted
    // run depends on must hold before the fault is installed.
    let plain = DatasetBuilder::from_uri(path).load().await.unwrap();
    let fragments: RoaringBitmap = plain
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let segments = stored_segments(&plain).await;
    let fts: Vec<(Uuid, RoaringBitmap)> = segments
        .iter()
        .filter(|(name, _, _)| name == FTS_INDEX)
        .map(|(_, uuid, bitmap)| (*uuid, bitmap.clone()))
        .collect();
    assert_eq!(fts.len(), 2, "the fixture must hold two FTS segments");
    let delta_coverage = fts
        .iter()
        .find(|(uuid, _)| *uuid == delta_uuid)
        .map(|(_, bitmap)| bitmap.clone())
        .expect("the registered delta must be in the manifest");
    assert!(
        !delta_coverage.is_empty() && delta_coverage.is_subset(&fragments),
        "the delta must cover a nonempty subset of the destinations: \
         {delta_coverage:?} vs {fragments:?}"
    );
    assert_ne!(
        delta_coverage, fragments,
        "a delta covering everything would supersede the older segment \
         outright, skipping the narrowing under test"
    );
    // The older segment translates its retired provenance into the
    // destination the delta does not hold, so it is only PARTIALLY
    // superseded and the remap still visits it.
    let older_derived = plain
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|index| index.name == FTS_INDEX && index.uuid != delta_uuid)
        .fold(RoaringBitmap::new(), |coverage, index| {
            coverage | index.fragment_bitmap.clone().unwrap_or_default()
        });
    assert!(
        !older_derived.is_empty(),
        "the older segment must still derive coverage, or it would be \
         skipped as fully superseded before the narrowing"
    );
    assert_eq!(
        derived_coverage(&plain, FTS_INDEX).await,
        fragments,
        "together the two segments must derive every destination"
    );
    let (count, plan) = fts_count_on_column(&plain, &word, "doc").await;
    assert_eq!(
        count, brute_force,
        "the fixture must answer exactly before any fault: {plan}"
    );

    (delta_uuid, word, brute_force)
}

/// The explicit remap verifies a derived listing before publishing anything,
/// then narrows the segment's coverage to what the reader attributes to it.
/// Both steps must read the SAME listing. The fixture keeps the older
/// segment only PARTIALLY superseded -- a finalized delta covers one of two
/// destinations -- so the older segment stays eligible and the narrowing is
/// reached. When the narrowing derives its own listing, a marker outage
/// between the two steps drops the directly covering sibling from it, nothing
/// cedes that sibling's destination, and the published replacement overlaps
/// it: every match is answered twice, on the same handle, on a fresh one,
/// after a successful retry, and in a materialized scan.
#[tokio::test]
async fn an_explicit_remap_narrows_coverage_from_the_listing_it_verified() {
    // Both caches off, so the narrowing step would really re-derive instead
    // of reusing a cached listing: the fault only shows through a fresh read.
    Box::pin(assert_partial_coverage_remap_is_exact(
        PHASED_FAULT_START,
        true,
    ))
    .await;
}

/// The same fixture and wrapper with the window placed past every marker
/// read: reads never fail, so an exact answer here shows the faulted case
/// above fails on overlapping publication rather than on the fixture, the
/// fault budget or the disabled caches.
#[tokio::test]
async fn a_partially_covered_remap_is_exact_with_healthy_uncached_reads() {
    Box::pin(assert_partial_coverage_remap_is_exact(
        PHASED_FAULT_NEVER,
        true,
    ))
    .await;
}

/// The same healthy run with the default caches, the ordinary configuration.
#[tokio::test]
async fn a_partially_covered_remap_is_exact_with_healthy_cached_reads() {
    Box::pin(assert_partial_coverage_remap_is_exact(
        PHASED_FAULT_NEVER,
        false,
    ))
    .await;
}

/// Remaps the partial-coverage fixture through the phased wrapper at `start`
/// and asserts nothing overlaps and every indexed answer is exact.
async fn assert_partial_coverage_remap_is_exact(start: usize, no_cache: bool) {
    let dir = TempStrDir::default();
    let path = format!("{}/fts_remap_listing_check_to_use", dir.as_str());
    let (delta_uuid, word, brute_force) = Box::pin(partially_covered_fts_delta(&path)).await;

    let suffix = format!(
        "{delta_uuid}/{}",
        lance_index::scalar::inverted::METADATA_FILE
    );
    let attempts = Arc::new(AtomicUsize::new(0));
    let mut builder = DatasetBuilder::from_uri(&path).with_store_params(ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(PhasedFailingReadWrapper {
            suffix,
            attempts: attempts.clone(),
            start,
        })),
        ..Default::default()
    });
    if no_cache {
        builder = builder
            .with_index_cache_size_bytes(0)
            .with_metadata_cache_size_bytes(0);
    }
    let mut dataset = builder.load().await.unwrap();

    // Whatever the outcome -- a refusal or a remap narrowed from the verified
    // listing -- what is published must never overlap.
    let remap_result = crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await;
    // The first remap need not consume the whole window, so the counter is
    // driven past it before the retry: otherwise the retry would meet
    // leftover injected errors and prove nothing about recovered reads.
    let met = attempts.swap(usize::MAX / 2, Ordering::SeqCst);
    let exhausted = met >= start + PHASED_FAULT_READS;
    let context = format!(
        "start={start}, no_cache={no_cache}, attempts={met}, \
         exhausted={exhausted}, remap={remap_result:?}"
    );

    let segments = stored_segments(&dataset).await;
    let fts: Vec<(Uuid, RoaringBitmap)> = segments
        .iter()
        .filter(|(name, _, _)| name == FTS_INDEX)
        .map(|(_, uuid, bitmap)| (*uuid, bitmap.clone()))
        .collect();
    for i in 0..fts.len() {
        for j in i + 1..fts.len() {
            assert!(
                fts[i].1.is_disjoint(&fts[j].1),
                "segments {} and {} must not claim the same destinations: \
                 {:?} ({context})",
                fts[i].0,
                fts[j].0,
                (&fts[i].1 & &fts[j].1).iter().collect::<Vec<_>>()
            );
        }
    }

    let (same_count, same_plan) = fts_count_on_column(&dataset, &word, "doc").await;
    let fresh = DatasetBuilder::from_uri(&path).load().await.unwrap();
    let (fresh_count, fresh_plan) = fts_count_on_column(&fresh, &word, "doc").await;
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["doc"],
        Some(FTS_INDEX.into()),
    )
    .await
    .unwrap_or_else(|error| {
        panic!("the retry must succeed once reads recover: {error} ({context})")
    });
    let (retried_count, retried_plan) = fts_count_on_column(&dataset, &word, "doc").await;

    let mut scan = fresh.scan();
    scan.full_text_search(
        FullTextSearchQuery::new(word.to_string())
            .with_column("doc".to_string())
            .unwrap(),
    )
    .unwrap();
    scan.project(&["i"]).unwrap();
    let materialized = scan.try_into_batch().await.unwrap().num_rows();

    assert_eq!(
        (same_count, fresh_count, retried_count, materialized),
        (brute_force, brute_force, brute_force, brute_force),
        "an overlapping replacement answers every match twice ({context})\
         \n{same_plan}\n{fresh_plan}"
    );
    for plan in [&same_plan, &fresh_plan, &retried_plan] {
        assert!(
            plan_has_node(plan, "MatchQuery:") && !plan_has_node(plan, "FlatMatchQuery:"),
            "every destination must stay served from the index ({context}): {plan}"
        );
    }
}
