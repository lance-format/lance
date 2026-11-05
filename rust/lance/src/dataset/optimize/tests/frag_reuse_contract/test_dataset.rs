// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A table paired with a model of its live rows and of which of them the
//! index serves.
//!
//! Every mutation that changes row contents applies the same change to the
//! model, deriving the table's change from the model where it can so the two
//! cannot diverge by construction. Rewrites and index maintenance leave the
//! rows alone: that they leave every answer unchanged is the property under
//! test. A deferred rewrite also leaves coverage alone, since translating the
//! index's addresses through the history is what fragment reuse is for.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt64Type};
use lance_core::utils::tempfile::TempStrDir;
use lance_core::{Error, ROW_ADDR};
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::optimize::OptimizeOptions;
use lance_table::format::Fragment;

use super::index_kind::{IndexKind, Row};
use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
use crate::dataset::optimize::remapping::remap_column_index;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::transaction::{Operation, RewriteGroup, Transaction};
use crate::dataset::write::CommitBuilder;
use crate::dataset::{
    MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched, WriteParams,
};
use crate::index::frag_reuse::frag_reuse_entry_appending;
use crate::index::frag_reuse_reader::tests::prepare_partition;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::session::Session;

pub(super) const ROWS_PER_FRAGMENT: i32 = 8;
const INDEX_NAME: &str = "idx";

/// The record form a fragment rewrite takes.
#[derive(Debug, Clone, Copy)]
pub(super) enum HistoryForm {
    /// A deferred compaction on an untagged table records a v0 reuse version.
    V0,
    /// Deferred compaction on a tagged table records ordered-compaction
    /// transitions. A stable partition committed before the index exists
    /// makes the table tagged.
    TaggedOrdered,
    /// Every rewrite is a stable partition.
    TaggedPartition,
}

pub(super) struct TestDataset {
    dataset: Dataset,
    kind: IndexKind,
    form: HistoryForm,
    /// The live rows by `i`: the oracle every check compares against.
    model: BTreeMap<i32, Row>,
    /// The live rows by `i` the index is not expected to serve, so a query
    /// scans their fragments instead. Without this, a history that silently
    /// dropped coverage would still pass every row check through the scan.
    uncovered: BTreeSet<i32>,
    /// Rows the index reaches only through one stable-partition transition.
    /// The index's bitmap names the transition's sources, not its
    /// destinations, so an in-place update of any of these rows withdraws
    /// the whole transition and uncovers all of them.
    translated_together: Vec<BTreeSet<i32>>,
    /// The `i` the next appended row takes, so appended rows never reuse a
    /// deleted row's identity.
    next_i: i32,
    _dir: TempStrDir,
}

impl TestDataset {
    /// Four fragments of written rows, in the shape `form` needs before the
    /// index is created.
    pub(super) async fn new(kind: IndexKind, form: HistoryForm) -> Self {
        let dir = TempStrDir::default();
        let next_i = 4 * ROWS_PER_FRAGMENT;
        let model: BTreeMap<i32, Row> = (0..next_i).map(|i| (i, Row::written(i))).collect();
        let batch = Row::to_batch(model.values());
        let schema = batch.schema();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            dir.as_str(),
            Some(write_params()),
        )
        .await
        .unwrap();
        let mut test_dataset = Self {
            dataset,
            kind,
            form,
            model,
            uncovered: (0..next_i).collect(),
            translated_together: Vec::new(),
            next_i,
            _dir: dir,
        };
        if matches!(form, HistoryForm::TaggedOrdered) {
            test_dataset.stable_partition().await;
        }
        test_dataset
    }

    pub(super) fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    pub(super) async fn create_index(&mut self) {
        let params = self.kind.params();
        self.dataset
            .create_index(
                &[self.kind.column()],
                self.kind.index_type(),
                Some(INDEX_NAME.to_string()),
                params.as_ref(),
                false,
            )
            .await
            .unwrap();
        self.uncovered.clear();
        self.translated_together.clear();
    }

    pub(super) async fn delete_where(&mut self, predicate: fn(&Row) -> bool) {
        let ids: Vec<i32> = self
            .model
            .values()
            .filter(|row| predicate(row))
            .map(|row| row.i)
            .collect();
        self.delete_ids(&ids).await;
    }

    /// Deletes every row of the live fragment at `position`, whatever its
    /// layout after earlier rewrites.
    pub(super) async fn delete_whole_fragment(&mut self, position: usize) {
        let fragment = self.dataset.fragments()[position].clone();
        let mut scan = self.dataset.scan();
        scan.with_fragments(vec![fragment]).project(&["i"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch["i"].as_primitive::<Int32Type>().values().to_vec();
        self.delete_ids(&ids).await;
    }

    async fn delete_ids(&mut self, ids: &[i32]) {
        for i in ids {
            assert!(self.model.remove(i).is_some(), "row {i} is not live");
            self.uncovered.remove(i);
        }
        let ids: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        self.dataset
            .delete(&format!("i IN ({})", ids.join(", ")))
            .await
            .unwrap();
    }

    pub(super) async fn append(&mut self, num_rows: i32) {
        let rows: Vec<Row> = (self.next_i..self.next_i + num_rows)
            .map(Row::written)
            .collect();
        self.next_i += num_rows;
        let batch = Row::to_batch(&rows);
        let schema = batch.schema();
        self.dataset
            .append(
                RecordBatchIterator::new([Ok(batch)], schema),
                Some(write_params()),
            )
            .await
            .unwrap();
        self.uncovered.extend(rows.iter().map(|row| row.i));
        self.model.extend(rows.into_iter().map(|row| (row.i, row)));
    }

    /// Rewrites every indexed column of the live rows among `ids` in place,
    /// keeping their addresses. Ids no longer live are left alone, as the
    /// merge's `DoNothing` does.
    pub(super) async fn update_in_place(&mut self, ids: &[i32]) {
        let updated: Vec<Row> = ids.iter().map(|&i| Row::updated(i)).collect();
        let mut merge =
            MergeInsertBuilder::try_new(Arc::new(self.dataset.clone()), vec!["i".into()]).unwrap();
        merge
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .write_mode(MergeInsertWriteMode::RewriteColumns);
        // Leaving `pad` out of the patch is what makes the update a column
        // rewrite that keeps row addresses.
        let patch = Row::to_batch(&updated).project(&[0, 2, 3, 4, 5]).unwrap();
        let schema = patch.schema();
        let (dataset, _) = merge
            .try_build()
            .unwrap()
            .execute_reader(RecordBatchIterator::new([Ok(patch)], schema))
            .await
            .unwrap();
        self.dataset = dataset.as_ref().clone();
        for row in updated {
            if let Some(live) = self.model.get_mut(&row.i) {
                *live = row;
                self.uncovered.insert(live.i);
            }
        }
        let updated_ids: BTreeSet<i32> = ids.iter().copied().collect();
        for rows in &self.translated_together {
            if !rows.is_disjoint(&updated_ids) {
                self.uncovered
                    .extend(rows.iter().filter(|i| self.model.contains_key(i)));
            }
        }
        // The update withdraws its fragments from the index's coverage, not
        // just the rows it rewrote.
        self.widen_uncovered_to_fragments().await;
    }

    /// Rewrites every live fragment in the record form of `self.form`.
    pub(super) async fn rewrite(&mut self) {
        match self.form {
            HistoryForm::V0 | HistoryForm::TaggedOrdered => {
                self.deferred_compaction(CompactionOptions::default().target_rows_per_fragment)
                    .await;
            }
            HistoryForm::TaggedPartition => self.stable_partition().await,
        }
        let entry = self
            .dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .expect("a deferred rewrite of indexed fragments records its moves");
        let is_tagged = entry.index_version >= 1;
        assert_eq!(
            is_tagged,
            !matches!(self.form, HistoryForm::V0),
            "{:?} recorded index_version {}",
            self.form,
            entry.index_version
        );
    }

    pub(super) async fn deferred_compaction(&mut self, target_rows_per_fragment: usize) {
        let options = CompactionOptions {
            target_rows_per_fragment,
            defer_index_remap: true,
            ..Default::default()
        };
        compact_files(&mut self.dataset, options, None)
            .await
            .unwrap();
        self.widen_uncovered_to_fragments().await;
    }

    /// Partitions every live fragment into two destinations through the
    /// commit path, recording a stable-partition transition.
    async fn stable_partition(&mut self) {
        let reserve = Transaction::new(
            self.dataset.manifest.version,
            Operation::ReserveFragments { num_fragments: 2 },
            None,
        );
        self.dataset
            .apply_commit(reserve, &Default::default(), &Default::default())
            .await
            .unwrap();
        let dest_base_id = self.dataset.manifest.max_fragment_id().unwrap() - 1;
        let old_fragments: Vec<Fragment> = self.dataset.fragments().iter().cloned().collect();
        let source_ids: Vec<u64> = old_fragments.iter().map(|fragment| fragment.id).collect();
        let (transition, new_fragments) =
            prepare_partition(&self.dataset, &source_ids, dest_base_id).await;
        // The partition rewrites every live fragment, so this transition
        // subsumes any earlier one.
        self.translated_together = vec![
            self.model
                .keys()
                .filter(|i| !self.uncovered.contains(i))
                .copied()
                .collect(),
        ];
        let read_version = self.dataset.manifest.version;
        let frag_reuse_index = Some(
            frag_reuse_entry_appending(&self.dataset, vec![transition])
                .await
                .unwrap(),
        );
        self.dataset = CommitBuilder::new(Arc::new(self.dataset.clone()))
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
        self.widen_uncovered_to_fragments().await;
    }

    /// Stages a merge of the index's segments, lets a concurrent writer
    /// delete a whole fragment, append rows and index them, then commits the
    /// staged segment on the retry: its files still hold the deleted
    /// fragment's entries.
    pub(super) async fn staged_merge_racing_a_delete(&mut self) {
        let kind = self.kind;
        let segments = self.dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        // Some index types refuse to merge segments over retired fragments or
        // under a tagged history; a refusal leaves the table as it was.
        let staged = match self.dataset.merge_existing_index_segments(segments).await {
            Ok(staged) => staged,
            Err(Error::NotSupported { .. } | Error::InvalidInput { .. }) => return,
            Err(error) => panic!("{kind:?}: staging a merge failed: {error}"),
        };

        // `self` plays the concurrent writer, so the model follows its
        // changes; the handle that staged the merge falls behind.
        let mut stale = self.dataset.clone();
        self.delete_whole_fragment(0).await;
        self.append(ROWS_PER_FRAGMENT).await;
        self.dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        self.uncovered.clear();
        // The retry below prunes the deleted fragment from the staged
        // segment's coverage and leaves the appended segment in place, so
        // every live row stays covered.

        let conflict = stale
            .commit_existing_index_segments(INDEX_NAME, kind.column(), vec![staged.clone()])
            .await
            .unwrap_err();
        assert!(
            matches!(conflict, Error::RetryableCommitConflict { .. }),
            "{conflict}"
        );
        self.dataset
            .commit_existing_index_segments(INDEX_NAME, kind.column(), vec![staged])
            .await
            .unwrap();
    }

    /// Compacts eagerly while the history still has moves the index has not
    /// caught up with. Remapping must compose the pending translation with
    /// the compaction's own moves; refusing the compaction is also correct.
    pub(super) async fn eager_compaction_over_pending_history(&mut self) {
        self.append(ROWS_PER_FRAGMENT).await;
        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            defer_index_remap: false,
            ..Default::default()
        };
        match compact_files(&mut self.dataset, options, None).await {
            Ok(_) => self.widen_uncovered_to_fragments().await,
            // A refused compaction leaves the table, and the appended rows
            // unindexed, as they were.
            Err(Error::InvalidInput { .. } | Error::NotSupported { .. }) => {}
            Err(error) => panic!("{:?}: eager compaction failed: {error}", self.kind),
        }
    }

    pub(super) async fn optimize(&mut self) {
        self.dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        self.uncovered.clear();
    }

    pub(super) async fn remap(&mut self) {
        remap_column_index(
            &mut self.dataset,
            &[self.kind.column()],
            Some(INDEX_NAME.to_string()),
        )
        .await
        .unwrap();
        self.translated_together.clear();
    }

    pub(super) async fn trim(&mut self) {
        cleanup_frag_reuse_index(&mut self.dataset).await.unwrap();
    }

    /// The fragment holding each live row, by `i`.
    async fn fragment_ids(dataset: &Dataset) -> BTreeMap<i32, u64> {
        let mut scan = dataset.scan();
        scan.project(&["i", ROW_ADDR]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch["i"].as_primitive::<Int32Type>().values();
        let addresses = batch[ROW_ADDR].as_primitive::<UInt64Type>().values();
        ids.iter()
            .zip(addresses.iter())
            .map(|(&i, &address)| (i, address >> 32))
            .collect()
    }

    /// Uncovers every row sharing a fragment with an uncovered row. Coverage
    /// is a set of whole fragments, so a rewrite that merges covered and
    /// uncovered rows into one fragment cannot keep the covered ones served
    /// by the index without the index missing the others.
    async fn widen_uncovered_to_fragments(&mut self) {
        let fragment_ids = Self::fragment_ids(&self.dataset).await;
        let uncovered_fragments: BTreeSet<u64> =
            self.uncovered.iter().map(|i| fragment_ids[i]).collect();
        self.uncovered = fragment_ids
            .into_iter()
            .filter(|(_, fragment_id)| uncovered_fragments.contains(fragment_id))
            .map(|(i, _)| i)
            .collect();
    }

    /// Asserts that a full scan returns the model, that every query of the
    /// index kind returns the model's answer, and that the index leaves out
    /// exactly the fragments holding rows the model expects it not to serve,
    /// through this handle and through a cold handle with a fresh session, so
    /// a cache keyed without the history's identity cannot hide behind a warm
    /// entry.
    pub(super) async fn check(&self, step: &str) {
        let Self { kind, form, .. } = self;
        let cold = DatasetBuilder::from_uri(self.dataset.uri())
            .with_session(Arc::new(Session::default()))
            .load()
            .await
            .unwrap();
        let handles = [(&self.dataset, "warm"), (&cold, "cold")];
        let model: Vec<&Row> = self.model.values().collect();
        for (handle, which) in handles {
            let mut scanned = Row::from_batch(&handle.scan().try_into_batch().await.unwrap());
            scanned.sort_by_key(|row| row.i);
            assert_eq!(
                scanned.iter().collect::<Vec<_>>(),
                model,
                "{kind:?} under {form:?} after {step}: full scan through the {which} handle"
            );
        }
        for query in kind.queries() {
            let expected = query.expected(self.model.values());
            for (handle, which) in handles {
                let context = format!(
                    "{kind:?} under {form:?} after {step}: {query:?} through the {which} handle"
                );
                let actual = query
                    .run(handle)
                    .await
                    .unwrap_or_else(|error| panic!("{context}: {error}"));
                assert_eq!(actual, expected, "{context}");
            }
        }
        let fragment_ids = Self::fragment_ids(&self.dataset).await;
        let expected_unindexed: BTreeSet<u64> =
            self.uncovered.iter().map(|i| fragment_ids[i]).collect();
        for (handle, which) in handles {
            let unindexed: BTreeSet<u64> = handle
                .unindexed_fragments(INDEX_NAME)
                .await
                .unwrap()
                .iter()
                .map(|fragment| fragment.id)
                .collect();
            assert_eq!(
                unindexed, expected_unindexed,
                "{kind:?} under {form:?} after {step}: fragments not covered by {INDEX_NAME} \
                 through the {which} handle (expected left out of the index: \
                 {expected_unindexed:?}, actually left out: {unindexed:?})"
            );
        }
    }
}

fn write_params() -> WriteParams {
    WriteParams {
        max_rows_per_file: ROWS_PER_FRAGMENT as usize,
        ..Default::default()
    }
}
