// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The fragment reuse contract, checked for every index type and history form:
//! whatever the history records, an indexed query returns exactly the rows a
//! query computed from the table's current contents returns.
//!
//! The oracle never consults an index or the history: it is a model of the
//! live rows that only row-changing steps update, and every check first
//! asserts that a full scan of the table still matches it. Each step is
//! checked through the handle that ran it and through a cold handle with a
//! fresh session, so a cache keyed without the history's identity cannot hide
//! behind a warm entry.
//!
//! Several steps leave an index's declared coverage narrower than the entries
//! its files physically hold (a whole-fragment delete, an in-place update, a
//! staged build committed after a concurrent delete). Readers must treat
//! coverage as an allow list rather than a description of the files.

mod index_kind;
mod test_dataset;

use lance_core::Error;
use rstest::rstest;

use index_kind::IndexKind;
use test_dataset::{HistoryForm, ROWS_PER_FRAGMENT, TestDataset};

async fn run_contract(kind: IndexKind, form: HistoryForm) {
    let mut table = TestDataset::new(kind, form).await;
    table.create_index().await;
    table.check("build").await;

    table.delete_where(|row| row.i % 5 == 0).await;
    table.check("partial delete").await;

    table.delete_whole_fragment(0).await;
    table.check("whole-fragment delete").await;

    table.staged_merge_racing_a_delete().await;
    table.check("staged merge racing a delete").await;

    table.rewrite().await;
    table.check("first rewrite").await;

    table.append(ROWS_PER_FRAGMENT).await;
    table.delete_where(|row| row.i % 7 == 3).await;
    table.check("append and delete over the rewrite").await;

    table.update_in_place(&[11, 13, 17, 33]).await;
    table.check("in-place update").await;

    table.rewrite().await;
    table.check("chained rewrite").await;

    table.eager_compaction_over_pending_history().await;
    table.check("eager compaction over a pending history").await;

    table.optimize().await;
    table.check("optimize_indices").await;

    table.remap().await;
    table.check("remap").await;

    table.trim().await;
    table.check("trim").await;
}

/// The shortest path to an eager compaction over a pending history: the
/// index still holds the addresses it was built with, the history translates
/// them once, and the eager compaction moves the translated rows again.
async fn run_eager_after_deferred(kind: IndexKind) {
    let mut table = TestDataset::new(kind, HistoryForm::V0).await;
    table.append(2 * ROWS_PER_FRAGMENT).await;
    table.create_index().await;
    // Six fragments become three, so the eager compaction rewrites rows the
    // history has already moved once.
    table
        .deferred_compaction(2 * ROWS_PER_FRAGMENT as usize)
        .await;
    assert_eq!(table.dataset().fragments().len(), 3);
    table.check("deferred compaction").await;

    table.eager_compaction_over_pending_history().await;
    table.check("eager compaction over a pending history").await;
}

#[rstest]
#[tokio::test]
async fn test_eager_compaction_over_a_pending_history_matches_the_table(
    #[values(
        IndexKind::BTree,
        IndexKind::Bitmap,
        IndexKind::ZoneMap,
        IndexKind::BloomFilter,
        IndexKind::LabelList,
        IndexKind::ListElement,
        IndexKind::NGram,
        IndexKind::Inverted,
        IndexKind::IvfFlat
    )]
    kind: IndexKind,
) {
    // Boxed for CI clippy `large_futures`: the commit cache grows this
    // compaction future past 16 KiB.
    Box::pin(run_eager_after_deferred(kind)).await;
}

#[rstest]
#[tokio::test]
async fn test_indexed_queries_match_the_table_through_fragment_reuse(
    #[values(
        IndexKind::BTree,
        IndexKind::Bitmap,
        IndexKind::ZoneMap,
        IndexKind::BloomFilter,
        IndexKind::LabelList,
        IndexKind::ListElement,
        IndexKind::NGram,
        IndexKind::IvfFlat
    )]
    kind: IndexKind,
    #[values(
        HistoryForm::V0,
        HistoryForm::TaggedOrdered,
        HistoryForm::TaggedPartition
    )]
    form: HistoryForm,
) {
    Box::pin(run_contract(kind, form)).await;
}

#[tokio::test]
async fn test_full_text_search_matches_the_table_through_v0_fragment_reuse() {
    Box::pin(run_contract(IndexKind::Inverted, HistoryForm::V0)).await;
}

#[rstest]
#[case::ordered(HistoryForm::TaggedOrdered)]
#[case::partition(HistoryForm::TaggedPartition)]
#[ignore = "https://github.com/lance-format/lance/issues/10623"]
#[tokio::test]
async fn test_full_text_search_matches_the_table_through_tagged_fragment_reuse(
    #[case] form: HistoryForm,
) {
    Box::pin(run_contract(IndexKind::Inverted, form)).await;
}

/// A listelement index on a stable-row-ID table, where the history is a v0
/// one loaded through `load_index`: through an eager compaction over a
/// pending history, and through history cleanup.
async fn run_list_element_on_stable_row_ids() {
    let mut table = TestDataset::new(IndexKind::ListElement, HistoryForm::V0WithStableRowIds).await;
    // Deleting before the compaction shifts the survivors' offsets when it
    // packs them, so translation must map offsets, not just fragment ids.
    table.delete_where(|row| row.i % 5 == 0).await;
    table.append(ROWS_PER_FRAGMENT).await;
    table.create_index().await;
    table.check("build").await;

    let before = TestDataset::fragment_ids(table.dataset()).await;
    table
        .deferred_compaction(2 * ROWS_PER_FRAGMENT as usize)
        .await;
    let after = TestDataset::fragment_ids(table.dataset()).await;
    assert!(
        before.iter().any(|(i, fragment)| after[i] != *fragment),
        "no covered live row moved, so the checks would not exercise translation"
    );
    table.check("deferred compaction").await;
    let history = table.history_len().await;
    assert!(history > 0);

    // The index still holds pre-compaction addresses, and an eager remap
    // does not compose them with the pending history, so it is refused.
    let version = table.dataset().version().version;
    let error = table.try_eager_compaction().await.unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(error.to_string().contains("pending remaps"), "{error}");
    assert_eq!(table.dataset().version().version, version);
    table.check("refused eager compaction").await;

    // The index has not caught up, so nothing may be trimmed yet.
    table.trim().await;
    assert_eq!(table.history_len().await, history, "trim before remap");
    table.check("trim before remap").await;

    table.optimize().await;
    table.check("optimize_indices").await;
    table.remap().await;
    table.check("remap").await;

    table.trim().await;
    assert!(
        table.history_len().await < history,
        "trim after remap left the history at {history} versions"
    );
    table.check("trim after remap").await;

    let metrics = table.try_eager_compaction().await.unwrap();
    assert!(
        metrics.fragments_removed > 0,
        "the eager compaction did nothing"
    );
    table.check("eager compaction after cleanup").await;
}

#[tokio::test]
async fn test_list_element_on_stable_row_ids_through_eager_compaction_and_cleanup() {
    // Boxed for CI clippy `large_futures`.
    Box::pin(run_list_element_on_stable_row_ids()).await;
}
