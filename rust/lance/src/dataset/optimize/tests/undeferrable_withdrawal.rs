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
use std::sync::atomic::{AtomicBool, Ordering};
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

    // A legacy segment's metadata to clone the sibling from, built here and
    // discarded: `create_index` refuses a duplicate name, so the sibling is
    // installed directly into the manifest instead.
    let template = {
        let mut scratch = dataset.clone();
        let mut params =
            VectorIndexParams::ivf_pq(1, 4, 2, lance_linalg::distance::DistanceType::L2, 10);
        params.version(IndexFileVersion::Legacy);
        CreateIndexBuilder::new(&mut scratch, &["vector"], IndexType::Vector, &params)
            .name(VECTOR_INDEX.to_string())
            .execute_uncommitted()
            .await
            .unwrap()
    };

    // The sibling: a segment of the index name under test, already in the
    // manifest, covering only the fragment the rewrite never touches. It is
    // installed BEFORE the staged build reads the table, so the rebase walks
    // ordinary transaction files afterwards.
    let sibling_coverage = RoaringBitmap::from_iter([untouched]);
    let sibling_uuid = Uuid::new_v4();
    {
        let mut sibling = template.clone();
        sibling.uuid = sibling_uuid;
        sibling.fragment_bitmap = Some(sibling_coverage.clone());
        let mut indices = load_all_indices(&dataset).await.unwrap().to_vec();
        indices.push(sibling);
        crate::index::frag_reuse_reader::tests::persist_fixture(&mut dataset, indices).await;
    }

    // The staged build reads the version that already has the sibling, and
    // covers the recorded destinations the rewrite then retires: every bit of
    // its coverage is withdrawn, so it lands emptied.
    let stale = dataset.clone();
    // Keeps the template's uuid, so its index files are the ones the
    // capability probe actually opens; the SIBLING is the synthetic copy.
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

/// The commit-time selector treats a marker probe fault as "not
/// classified": the deferred compaction commits, withdraws nothing from the
/// segment it could not read, and the verdict is not cached, so once reads
/// succeed the same handle classifies the segment again. A withdrawal is
/// durable, so it must never rest on a transient fault.
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
