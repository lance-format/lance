// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemWAL Index operations.
//!
//! The index data structures and the helpers that read and update the index's
//! `IndexMetadata` entry live in [`lance_table::system_index::mem_wal`]; this
//! module holds the dataset-level operations built on top of them.

pub(crate) use lance_table::system_index::mem_wal::{
    load_mem_wal_index_details, new_mem_wal_index_meta, open_mem_wal_index,
};

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use lance_index::mem_wal::{CompactedSsTable, MEM_WAL_INDEX_NAME, MemWalIndexDetails};
    use lance_table::format::IndexMetadata;
    use lance_table::system_index::mem_wal::update_mem_wal_index_compacted_sstables;
    use std::sync::Arc;
    use uuid::Uuid;

    use crate::index::DatasetIndexExt;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use crate::dataset::mem_wal::DatasetMemWalExt;
    use crate::dataset::transaction::{Operation, Transaction};
    use crate::dataset::write::merge_insert::{
        MergeInsertBuilder, MergeInsertWriteMode, WhenMatched, WhenNotMatched,
    };
    use crate::dataset::{CommitBuilder, InsertBuilder, WriteParams};
    use arrow_array::cast::AsArray;
    use arrow_array::record_batch;
    use arrow_array::types::Int32Type;

    async fn test_dataset() -> crate::Dataset {
        let write_params = WriteParams {
            max_rows_per_file: 10,
            ..Default::default()
        };
        let data = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::Int32, true),
            ])),
            vec![
                Arc::new(Int32Array::from_iter_values(0..10_i32)),
                Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(0, 10))),
            ],
        )
        .unwrap();
        InsertBuilder::new("memory://test_mem_wal")
            .with_params(&write_params)
            .execute(vec![data])
            .await
            .unwrap()
    }

    /// A dataset with `__lance_mem_wal` already installed, as MemWAL
    /// initialization leaves it. Recording compaction progress requires the
    /// system index to exist, so any test that commits progress needs this
    /// rather than a bare [`test_dataset`].
    async fn test_dataset_with_mem_wal() -> crate::Dataset {
        let dataset = test_dataset().await;
        let mem_wal_index =
            new_mem_wal_index_meta(dataset.manifest.version, MemWalIndexDetails::default())
                .unwrap();
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![mem_wal_index],
                removed_indices: vec![],
            },
            None,
        );
        CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap()
    }

    /// UpdateMemWalState touches indexes, not data, so it must carry the
    /// fragment list forward. The operation builds its manifest from scratch,
    /// and an unpopulated fragment list is published as an empty one.
    #[tokio::test]
    async fn test_update_mem_wal_state_preserves_fragments() {
        let dataset = test_dataset_with_mem_wal().await;
        let rows_before = dataset.count_rows(None).await.unwrap();
        let fragments_before: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
        assert!(rows_before > 0, "precondition: the table holds rows");

        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(Uuid::new_v4(), 1)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        assert_eq!(
            dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
            fragments_before,
            "UpdateMemWalState dropped fragments"
        );
        assert_eq!(
            dataset.count_rows(None).await.unwrap(),
            rows_before,
            "UpdateMemWalState dropped rows"
        );
    }

    /// Test that UpdateMemWalState with lower generation than committed fails without retry.
    /// Per spec: If committed_generation >= to_commit_generation, abort without retry.
    #[tokio::test]
    async fn test_update_mem_wal_state_conflict_lower_generation_no_retry() {
        let dataset = test_dataset_with_mem_wal().await;
        let shard = Uuid::new_v4();

        // First commit UpdateMemWalState with generation 10
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Try to commit UpdateMemWalState with generation 5 (lower than 10)
        // This should fail with non-retryable conflict
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            matches!(result, Err(crate::Error::IncompatibleTransaction { .. })),
            "Expected non-retryable IncompatibleTransaction for lower generation, got {:?}",
            result
        );
    }

    /// Test that UpdateMemWalState with equal generation as committed fails without retry.
    #[tokio::test]
    async fn test_update_mem_wal_state_conflict_equal_generation_no_retry() {
        let dataset = test_dataset_with_mem_wal().await;
        let shard = Uuid::new_v4();

        // First commit UpdateMemWalState with generation 10
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Try to commit UpdateMemWalState with generation 10 (equal)
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            matches!(result, Err(crate::Error::IncompatibleTransaction { .. })),
            "Expected non-retryable IncompatibleTransaction for equal generation, got {:?}",
            result
        );
    }

    /// Test that UpdateMemWalState with higher generation than committed is retryable.
    /// Per spec: If committed_generation < to_commit_generation, retry is allowed.
    #[tokio::test]
    async fn test_update_mem_wal_state_conflict_higher_generation_retryable() {
        let dataset = test_dataset_with_mem_wal().await;
        let shard = Uuid::new_v4();

        // First commit UpdateMemWalState with generation 5
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Try to commit UpdateMemWalState with generation 10 (higher than 5)
        // This should fail with retryable conflict
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            matches!(result, Err(crate::Error::RetryableCommitConflict { .. })),
            "Expected retryable conflict for higher generation, got {:?}",
            result
        );
    }

    /// Test that UpdateMemWalState on different shards don't conflict.
    #[tokio::test]
    async fn test_update_mem_wal_state_different_shards_no_conflict() {
        let dataset = test_dataset_with_mem_wal().await;
        let shard1 = Uuid::new_v4();
        let shard2 = Uuid::new_v4();

        // First commit UpdateMemWalState for shard1
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard1, 10)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Commit UpdateMemWalState for shard2 based on old version
        // This should succeed because different shards don't conflict
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard2, 5)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            result.is_ok(),
            "Expected success for different shards, got {:?}",
            result
        );

        // Verify both shards are in the index
        let dataset = result.unwrap();
        let mem_wal_idx = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .unwrap()
            .clone();
        let details = load_mem_wal_index_details(mem_wal_idx).unwrap();
        assert_eq!(details.compacted_sstables.len(), 2);
    }

    /// Test that CreateIndex of MemWalIndex can be rebased against UpdateMemWalState.
    /// The compacted_sstables from UpdateMemWalState should be included in CreateIndex.
    #[tokio::test]
    async fn test_create_index_rebase_against_update_mem_wal_state() {
        let dataset = test_dataset_with_mem_wal().await;
        let shard = Uuid::new_v4();

        // First commit UpdateMemWalState with generation 10
        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // CreateIndex of MemWalIndex based on old version (before UpdateMemWalState)
        // This should succeed and combine the compaction progress.
        let details = MemWalIndexDetails {
            num_shards: 1,
            ..Default::default()
        };
        let mem_wal_index = new_mem_wal_index_meta(dataset.manifest.version - 1, details).unwrap();

        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::CreateIndex {
                new_indices: vec![mem_wal_index],
                removed_indices: vec![],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            result.is_ok(),
            "Expected CreateIndex to succeed with rebase, got {:?}",
            result
        );

        // Verify the compacted_sstables from UpdateMemWalState were included in CreateIndex
        let dataset = result.unwrap();
        let mem_wal_idx = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .unwrap()
            .clone();
        let details = load_mem_wal_index_details(mem_wal_idx).unwrap();
        assert_eq!(details.compacted_sstables.len(), 1);
        assert_eq!(details.compacted_sstables[0].shard_id, shard);
        assert_eq!(details.compacted_sstables[0].generation, 10);
        assert_eq!(details.num_shards, 1); // Config from CreateIndex preserved
    }

    /// Test that UpdateMemWalState against CreateIndex of MemWalIndex checks generations.
    #[tokio::test]
    async fn test_update_mem_wal_state_against_create_index_lower_generation() {
        let dataset = test_dataset().await;
        let shard = Uuid::new_v4();

        // First commit CreateIndex of MemWalIndex with compacted_sstables
        let details = MemWalIndexDetails {
            compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
            ..Default::default()
        };
        let mem_wal_index = new_mem_wal_index_meta(dataset.manifest.version, details).unwrap();

        let txn1 = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![mem_wal_index],
                removed_indices: vec![],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn1)
            .await
            .unwrap();

        // Try UpdateMemWalState with lower generation
        let txn2 = Transaction::new(
            dataset.manifest.version - 1, // Based on old version
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
            },
            None,
        );
        let result = CommitBuilder::new(Arc::new(dataset)).execute(txn2).await;

        assert!(
            matches!(result, Err(crate::Error::IncompatibleTransaction { .. })),
            "Expected non-retryable IncompatibleTransaction when UpdateMemWalState generation is lower than CreateIndex, got {:?}",
            result
        );
    }

    /// A table with `generation` folded into base.
    async fn compacted_dataset(shard: Uuid, generation: u64) -> crate::Dataset {
        let dataset = test_dataset_with_mem_wal().await;
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, generation)],
            },
            None,
        );
        CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap()
    }

    /// An index segment spanning `fragments`, as a completed build leaves.
    fn index_over(name: &str, fragments: &[u32]) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version: 1,
            fragment_bitmap: Some(roaring::RoaringBitmap::from_iter(fragments.iter().copied())),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    async fn catch_up_generation(dataset: &crate::Dataset, index: &str) -> Option<u64> {
        let meta = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .unwrap()
            .clone();
        load_mem_wal_index_details(meta)
            .unwrap()
            .index_catchup
            .into_iter()
            .find(|entry| entry.index_name == index)
            .and_then(|entry| entry.caught_up_generations.first().map(|g| g.generation))
    }

    /// The commit path, not the derivation in isolation: `commit_transaction`
    /// has to load the read version's indices and hand them down.
    #[tokio::test]
    async fn an_index_covering_the_table_earns_catch_up_on_commit() {
        let shard = Uuid::new_v4();
        let dataset = compacted_dataset(shard, 5).await;

        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![index_over("idx", &[0])],
                removed_indices: vec![],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        assert_eq!(catch_up_generation(&dataset, "idx").await, Some(5));
    }

    /// A repair with nothing to rebuild still has to commit.
    ///
    /// Coverage is derived at commit time, so an index that already spans the
    /// table records its position only if there is a commit to record it on.
    /// That is the ordinary case after a remap, or after a compaction that
    /// advanced a generation without changing which fragments exist: the
    /// optimize finds no unindexed fragments and has no new segment to publish.
    /// Skipping the commit there leaves the position missing forever, the
    /// scheduler repeating the same repair, and the last SSTable unretirable.
    #[tokio::test]
    async fn a_no_work_repair_still_records_derived_catch_up() {
        use lance_index::optimize::OptimizeOptions;

        let shard = Uuid::new_v4();
        let dataset = compacted_dataset(shard, 7).await;
        // Already spans the table, so the optimize below has nothing to build.
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![index_over("idx", &[0])],
                removed_indices: vec![],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        // Compaction advances without adding a fragment, so the index still
        // covers and the optimize stays a no-op.
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 9)],
            },
            None,
        );
        let mut dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        dataset
            .optimize_indices(&OptimizeOptions::append().index_names(vec!["idx".to_string()]))
            .await
            .unwrap();

        assert_eq!(catch_up_generation(&dataset, "idx").await, Some(9));

        // And once it is current, the next pass must not commit again:
        // periodic maintenance would otherwise mint a version forever.
        let after_repair = dataset.manifest.version;
        dataset
            .optimize_indices(&OptimizeOptions::append().index_names(vec!["idx".to_string()]))
            .await
            .unwrap();
        assert_eq!(dataset.manifest.version, after_repair);
    }

    /// A no-op optimize on a table that is not on the protocol must stay a
    /// no-op: the early return is what keeps ordinary tables from committing an
    /// empty version on every maintenance pass.
    #[tokio::test]
    async fn a_no_work_optimize_with_nothing_compacted_commits_nothing() {
        use lance_index::optimize::OptimizeOptions;

        let dataset = test_dataset_with_mem_wal().await;
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![index_over("idx", &[0])],
                removed_indices: vec![],
            },
            None,
        );
        let mut dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();
        let before = dataset.manifest.version;

        dataset
            .optimize_indices(&OptimizeOptions::append().index_names(vec!["idx".to_string()]))
            .await
            .unwrap();

        assert_eq!(dataset.manifest.version, before);
    }

    /// A table carrying compaction progress but no catch-up entry earns one
    /// from an ordinary commit. This is how a table written before catch-up was
    /// maintained heals itself: nothing has to be run against it.
    #[tokio::test]
    async fn a_table_with_no_catchup_entry_earns_one() {
        let dataset = test_dataset_with_mem_wal().await;
        let shard = Uuid::new_v4();
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![index_over("idx", &[0])],
                removed_indices: vec![],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        assert_eq!(catch_up_generation(&dataset, "idx").await, Some(5));
    }

    /// What a commit earns is fixed by the version it read, and a rebase does
    /// not move it. The builder inspected a one-fragment table with generation
    /// 5 folded in; by the time it commits, an append has landed. It still
    /// earns 5 -- judged against the table it never saw, it would earn nothing
    /// and the SSTables would be retained forever.
    #[tokio::test]
    async fn credit_is_anchored_to_the_read_version_across_a_rebase() {
        let shard = Uuid::new_v4();
        let dataset = compacted_dataset(shard, 5).await;
        let read_version = dataset.manifest.version;

        let data = RecordBatch::try_new(
            Arc::new(Schema::from(dataset.schema())),
            vec![
                Arc::new(Int32Array::from_iter_values(10..20_i32)),
                Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(0, 10))),
            ],
        )
        .unwrap();
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: crate::dataset::WriteMode::Append,
                max_rows_per_file: 10,
                ..Default::default()
            })
            .execute(vec![data])
            .await
            .unwrap();
        assert!(
            dataset.get_fragments().len() > 1,
            "the append must add a fragment the index has not seen"
        );

        // Built against `read_version`, and only ever saw fragment 0.
        let txn = Transaction::new(
            read_version,
            Operation::CreateIndex {
                new_indices: vec![index_over("idx", &[0])],
                removed_indices: vec![],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        assert_eq!(catch_up_generation(&dataset, "idx").await, Some(5));
    }

    /// A user index build that races a compaction commit is rejected outright
    /// rather than rebased -- only the system index may rebase against
    /// `UpdateMemWalState`. Anything scheduling catch-up work has to expect the
    /// build to be thrown away and retried, so a busy shard needs the two kept
    /// apart rather than merely retried.
    #[tokio::test]
    async fn a_user_index_build_cannot_rebase_past_a_compaction_commit() {
        let shard = Uuid::new_v4();
        let dataset = compacted_dataset(shard, 5).await;
        let read_version = dataset.manifest.version;

        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 9)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        let txn = Transaction::new(
            read_version,
            Operation::CreateIndex {
                new_indices: vec![index_over("idx", &[0])],
                removed_indices: vec![],
            },
            None,
        );
        let err = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("incompatible"),
            "expected an incompatible-transaction error, got {err}"
        );
    }

    /// One `__lance_mem_wal` entry carrying `details`, as a real table has.
    fn indices_with(details: MemWalIndexDetails) -> Vec<IndexMetadata> {
        vec![new_mem_wal_index_meta(1, details).unwrap()]
    }

    fn compacted_generation(indices: &[IndexMetadata], shard: Uuid) -> Option<u64> {
        load_mem_wal_index_details(indices[0].clone())
            .unwrap()
            .compacted_sstables
            .iter()
            .find(|sstable| sstable.shard_id == shard)
            .map(|sstable| sstable.generation)
    }

    #[test]
    fn test_update_compacted_sstables() {
        let shard1 = Uuid::new_v4();
        let shard2 = Uuid::new_v4();
        let mut indices = indices_with(MemWalIndexDetails::default());

        update_mem_wal_index_compacted_sstables(
            &mut indices,
            1,
            vec![CompactedSsTable::new(shard1, 5)],
        )
        .unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(compacted_generation(&indices, shard1), Some(5));

        // Advancing an existing shard.
        update_mem_wal_index_compacted_sstables(
            &mut indices,
            2,
            vec![CompactedSsTable::new(shard1, 10)],
        )
        .unwrap();
        assert_eq!(compacted_generation(&indices, shard1), Some(10));

        // A second shard is independent.
        update_mem_wal_index_compacted_sstables(
            &mut indices,
            3,
            vec![CompactedSsTable::new(shard2, 3)],
        )
        .unwrap();
        assert_eq!(compacted_generation(&indices, shard1), Some(10));
        assert_eq!(compacted_generation(&indices, shard2), Some(3));
    }

    /// A stale proposal must fail the whole transaction: accepting it while
    /// keeping generation 10 would publish the stale worker's rows under a
    /// marker they did not produce.
    #[test]
    fn a_lower_generation_rejects_the_whole_update() {
        let shard = Uuid::new_v4();
        let mut indices = indices_with(MemWalIndexDetails {
            compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
            ..Default::default()
        });
        let before = indices.clone();

        let err = update_mem_wal_index_compacted_sstables(
            &mut indices,
            2,
            vec![CompactedSsTable::new(shard, 8)],
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("Stale SSTable compaction"),
            "{err}"
        );
        assert_eq!(
            indices[0].uuid, before[0].uuid,
            "a rejected update must leave the index list untouched"
        );
        assert_eq!(compacted_generation(&indices, shard), Some(10));
    }

    /// Equal is also refused: it reports nothing new, and accepting it would let
    /// a retry publish a second set of row mutations under the same marker.
    #[test]
    fn an_equal_generation_rejects() {
        let shard = Uuid::new_v4();
        let mut indices = indices_with(MemWalIndexDetails {
            compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
            ..Default::default()
        });

        let err = update_mem_wal_index_compacted_sstables(
            &mut indices,
            2,
            vec![CompactedSsTable::new(shard, 10)],
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("Stale SSTable compaction"),
            "{err}"
        );
    }

    #[test]
    fn duplicate_shards_in_one_update_reject() {
        let shard = Uuid::new_v4();
        let mut indices = indices_with(MemWalIndexDetails::default());

        let err = update_mem_wal_index_compacted_sstables(
            &mut indices,
            1,
            vec![
                CompactedSsTable::new(shard, 5),
                CompactedSsTable::new(shard, 6),
            ],
        )
        .unwrap_err();

        assert!(err.to_string().contains("Duplicate shard"), "{err}");
    }

    /// Absent metadata must not be materialized: default details describe a
    /// table with no MemWAL shards, so the recorded generation would name a
    /// shard nothing can corroborate.
    #[test]
    fn a_missing_mem_wal_index_rejects() {
        let mut indices: Vec<IndexMetadata> = Vec::new();

        let err = update_mem_wal_index_compacted_sstables(
            &mut indices,
            1,
            vec![CompactedSsTable::new(Uuid::new_v4(), 5)],
        )
        .unwrap_err();

        assert!(err.to_string().contains("does not exist"), "{err}");
        assert!(indices.is_empty(), "nothing should have been created");
    }

    /// Recording progress replaces the entry where it sits, so the index list
    /// keeps its order.
    #[test]
    fn recording_progress_keeps_the_system_index_position() {
        let shard = Uuid::new_v4();
        let mut indices = indices_with(MemWalIndexDetails::default());
        // A neighbour to show the entry is replaced in place, not moved.
        indices.push(IndexMetadata {
            name: "other_index".to_string(),
            ..indices[0].clone()
        });
        update_mem_wal_index_compacted_sstables(
            &mut indices,
            2,
            vec![CompactedSsTable::new(shard, 5)],
        )
        .unwrap();

        assert_eq!(indices[0].name, MEM_WAL_INDEX_NAME);
        assert_eq!(indices[1].name, "other_index");
    }

    /// Recording progress must not disturb any other saved MemWAL field.
    #[test]
    fn recording_progress_preserves_unrelated_mem_wal_state() {
        let shard = Uuid::new_v4();
        let other_shard = Uuid::new_v4();
        let mut writer_config_defaults = HashMap::new();
        writer_config_defaults.insert("target_size".to_string(), "64MB".to_string());
        let details = MemWalIndexDetails {
            snapshot_ts_millis: 12_345,
            num_shards: 4,
            maintained_indexes: vec!["vector_idx".to_string()],
            compacted_sstables: vec![CompactedSsTable::new(other_shard, 7)],
            writer_config_defaults: writer_config_defaults.clone(),
            ..Default::default()
        };
        let mut indices = indices_with(details);

        update_mem_wal_index_compacted_sstables(
            &mut indices,
            2,
            vec![CompactedSsTable::new(shard, 5)],
        )
        .unwrap();

        let after = load_mem_wal_index_details(indices[0].clone()).unwrap();
        assert_eq!(after.snapshot_ts_millis, 12_345);
        assert_eq!(after.num_shards, 4);
        assert_eq!(after.maintained_indexes, vec!["vector_idx".to_string()]);
        assert_eq!(after.writer_config_defaults, writer_config_defaults);
        // The untouched shard keeps its generation.
        assert_eq!(compacted_generation(&indices, other_shard), Some(7));
    }

    /// The hole this guards is a worker that refreshed to HEAD before
    /// submitting: there is no intervening transaction, so the conflict
    /// resolver sees nothing to reject and only apply-time validation stands
    /// between a stale generation and a published commit.
    ///
    /// The row mutation is the point: were the stale generation accepted, this
    /// commit would add a fragment while the recorded generation stayed at 10,
    /// so the rows and the number describing them would disagree.
    #[tokio::test]
    async fn a_stale_generation_against_latest_head_publishes_nothing() {
        let shard = Uuid::new_v4();
        let dataset = test_dataset_with_mem_wal().await;
        let version = dataset.manifest.version;

        // Record generation 10.
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                version,
                Operation::UpdateMemWalState {
                    dir_id: String::new(),
                    compacted_sstables: vec![CompactedSsTable::new(shard, 10)],
                },
                None,
            ))
            .await
            .unwrap();

        let version_before = dataset.manifest.version;
        let fragments_before = dataset.get_fragments().len();

        // Reuse written data so schema validation cannot mask the stale generation.
        let mut new_fragment = dataset.fragments()[0].clone();
        new_fragment.id = 999;

        // Built against the LATEST version, so there is nothing stale for the
        // conflict resolver to catch, and carrying a real row mutation.
        let stale = Transaction::new(
            version_before,
            Operation::Update {
                removed_fragment_ids: vec![],
                updated_fragments: vec![],
                new_fragments: vec![new_fragment],
                fields_modified: vec![],
                mem_wal_dir_id: Some(String::new()),
                compacted_sstables: vec![CompactedSsTable::new(shard, 5)],
                fields_for_preserving_frag_bitmap: vec![],
                update_mode: None,
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            None,
        );
        let mut dataset = dataset;
        let result = CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(stale)
            .await;

        // Asserting the reason, not just the failure: the commit must be
        // refused by apply-time generation validation, not by something
        // incidental about the fragment.
        let err = result.expect_err("a stale generation must fail the whole commit");
        assert!(matches!(err, crate::Error::InvalidInput { .. }));
        assert!(
            err.to_string().contains("Stale SSTable compaction"),
            "expected stale-generation rejection, got {err}"
        );

        // Neither the marker nor the fragment may have been published.
        dataset.checkout_latest().await.unwrap();
        let latest = dataset;
        assert_eq!(
            latest.manifest.version, version_before,
            "no new table version may be published"
        );
        assert_eq!(
            latest.get_fragments().len(),
            fragments_before,
            "the row mutation must not be published"
        );
        let details = load_mem_wal_index_details(
            latest
                .load_indices()
                .await
                .unwrap()
                .iter()
                .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
                .unwrap()
                .clone(),
        )
        .unwrap();
        assert_eq!(
            details.compacted_sstables[0].generation, 10,
            "the recorded generation must be unchanged"
        );
    }

    /// The system index holds the catch-up positions the WAL pod retires SSTables against.
    /// Erasing it through the ordinary index API would leave the table claiming
    /// nothing was ever compacted while the SSTables are already gone.

    #[test]
    fn test_empty_compacted_sstables_noop() {
        let mut indices = Vec::new();

        // Empty update should be a no-op, even with no MemWAL index present.
        update_mem_wal_index_compacted_sstables(&mut indices, 1, vec![]).unwrap();

        assert!(indices.is_empty());
    }

    /// Regression: a committed `__mem_wal` (legitimately `fragment_bitmap:
    /// None`) must not break `describe_indices` — the path behind lancedb's
    /// `list_indices`/`wait_for_index`. It's described as zero indexed rows,
    /// like `__frag_reuse`.
    #[tokio::test]
    async fn test_describe_indices_includes_mem_wal_system_index() {
        use crate::index::DatasetIndexExt;
        use lance_index::IndexType;
        use lance_index::scalar::ScalarIndexParams;

        let mut dataset = test_dataset_with_mem_wal().await;

        // A real user index that describe_indices must keep returning.
        dataset
            .create_index(
                &["a"],
                IndexType::Scalar,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Commit a __mem_wal index, as WAL provisioning does in production.
        let shard = Uuid::new_v4();
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::UpdateMemWalState {
                dir_id: String::new(),
                compacted_sstables: vec![CompactedSsTable::new(shard, 1)],
            },
            None,
        );
        let dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap();

        // The system index is present with no fragment_bitmap (by design).
        let mem_wal = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|i| i.name == MEM_WAL_INDEX_NAME)
            .unwrap()
            .clone();
        assert!(mem_wal.fragment_bitmap.is_none());

        // describe_indices describes the bitmap-less __mem_wal alongside the
        // real index instead of erroring.
        let descriptions = dataset.describe_indices(None).await.unwrap();
        let mem_wal_desc = descriptions
            .iter()
            .find(|d| d.name() == MEM_WAL_INDEX_NAME)
            .expect("__mem_wal must be described, not skipped");
        assert_eq!(
            mem_wal_desc.index_type(),
            "MemWal",
            "system index type must resolve via infer_system_index_type"
        );
        assert_eq!(
            mem_wal_desc.rows_indexed(),
            0,
            "a bitmap-less system index indexes zero rows"
        );
        assert_eq!(
            descriptions.len(),
            2,
            "both the real scalar index and __mem_wal must be listed"
        );
    }

    const DIR_A: &str = "a1b2c3d4";
    const DIR_B: &str = "c3d4e5f6";

    /// Commit a MemWAL index with `dir_id`, as an initialization would.
    async fn enable(dataset: crate::Dataset, dir_id: &str) -> crate::Dataset {
        let details = MemWalIndexDetails {
            dir_id: dir_id.to_string(),
            ..Default::default()
        };
        let index = new_mem_wal_index_meta(dataset.manifest.version, details).unwrap();
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![index],
                removed_indices: vec![],
            },
            None,
        );
        CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap()
    }

    /// Remove the MemWAL index, as a disable would.
    async fn disable(dataset: crate::Dataset) -> crate::Dataset {
        let mem_wal = dataset
            .load_index_by_name(MEM_WAL_INDEX_NAME)
            .await
            .unwrap()
            .expect("MemWAL index present");
        let txn = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![],
                removed_indices: vec![mem_wal],
            },
            None,
        );
        CommitBuilder::new(Arc::new(dataset))
            .execute(txn)
            .await
            .unwrap()
    }

    fn progress(read_version: u64, dir_id: &str, compacted: Vec<CompactedSsTable>) -> Transaction {
        Transaction::new(
            read_version,
            Operation::UpdateMemWalState {
                compacted_sstables: compacted,
                dir_id: dir_id.to_string(),
            },
            None,
        )
    }

    async fn commit(dataset: &crate::Dataset, txn: Transaction) -> crate::Result<crate::Dataset> {
        CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(txn)
            .await
    }

    #[tokio::test]
    async fn test_fence_commits_while_its_mem_wal_is_present() {
        let dataset = enable(test_dataset().await, DIR_A).await;
        let fenced = commit(&dataset, progress(dataset.manifest.version, DIR_A, vec![]))
            .await
            .unwrap();
        assert_eq!(fenced.manifest.version, dataset.manifest.version + 1);
        assert!(
            fenced.mem_wal_index_details().await.unwrap().is_some(),
            "a fence keeps the MemWAL index"
        );
    }

    #[tokio::test]
    async fn test_fence_built_on_head_without_mem_wal_is_stale() {
        let dataset = disable(enable(test_dataset().await, DIR_A).await).await;
        let error = commit(&dataset, progress(dataset.manifest.version, DIR_A, vec![]))
            .await
            .expect_err("the MemWAL was removed");
        assert!(error.is_stale_mem_wal(), "unexpected: {error}");
    }

    #[tokio::test]
    async fn test_fence_rebased_over_removal_is_stale() {
        let enabled = enable(test_dataset().await, DIR_A).await;
        let read_version = enabled.manifest.version;
        let disabled = disable(enabled).await;
        let error = commit(&disabled, progress(read_version, DIR_A, vec![]))
            .await
            .expect_err("the fence rebases over the removal");
        assert!(error.is_stale_mem_wal(), "unexpected: {error}");
    }

    #[tokio::test]
    async fn test_progress_for_a_replaced_mem_wal_is_stale() {
        let shard = Uuid::new_v4();
        let enabled = enable(test_dataset().await, DIR_A).await;
        let read_version = enabled.manifest.version;
        let reenabled = enable(disable(enabled).await, DIR_B).await;

        // Built directly on the re-enabled HEAD, as a job that checks out latest
        // between chunks would.
        let on_head = commit(
            &reenabled,
            progress(
                reenabled.manifest.version,
                DIR_A,
                vec![CompactedSsTable::new(shard, 3)],
            ),
        )
        .await
        .expect_err("progress names the replaced MemWAL");
        assert!(on_head.is_stale_mem_wal(), "unexpected: {on_head}");

        // Rebased over the disable and re-enable.
        let rebased = commit(
            &reenabled,
            progress(read_version, DIR_A, vec![CompactedSsTable::new(shard, 3)]),
        )
        .await
        .expect_err("progress names the replaced MemWAL");
        assert!(rebased.is_stale_mem_wal(), "unexpected: {rebased}");

        let details = reenabled.mem_wal_index_details().await.unwrap().unwrap();
        assert!(
            details.compacted_sstables.is_empty(),
            "the new MemWAL recorded the old one's progress"
        );
    }

    #[tokio::test]
    async fn test_progress_rebases_over_same_mem_wal_progress() {
        let dataset = enable(test_dataset().await, DIR_A).await;
        let read_version = dataset.manifest.version;
        let dataset = commit(
            &dataset,
            progress(
                read_version,
                DIR_A,
                vec![CompactedSsTable::new(Uuid::new_v4(), 2)],
            ),
        )
        .await
        .unwrap();
        commit(
            &dataset,
            progress(
                read_version,
                DIR_A,
                vec![CompactedSsTable::new(Uuid::new_v4(), 5)],
            ),
        )
        .await
        .expect("progress for another shard of the same MemWAL rebases");
    }

    fn compaction_update(
        compacted_sstables: Vec<CompactedSsTable>,
        mem_wal_dir_id: Option<&str>,
    ) -> Operation {
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![],
            new_fragments: vec![],
            fields_modified: vec![],
            compacted_sstables,
            mem_wal_dir_id: mem_wal_dir_id.map(str::to_string),
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: None,
            inserted_rows_filter: None,
            updated_fragment_offsets: None,
        }
    }

    #[tokio::test]
    async fn test_compaction_update_checks_its_mem_wal() {
        let shard = Uuid::new_v4();
        let dataset = enable(
            disable(enable(test_dataset().await, DIR_A).await).await,
            DIR_B,
        )
        .await;
        let version = dataset.manifest.version;

        let stale = commit(
            &dataset,
            Transaction::new(
                version,
                compaction_update(vec![CompactedSsTable::new(shard, 1)], Some(DIR_A)),
                None,
            ),
        )
        .await
        .expect_err("a compaction from the replaced MemWAL");
        assert!(stale.is_stale_mem_wal(), "unexpected: {stale}");

        // An earlier chunk of the same compaction records no progress, and is
        // refused all the same.
        let stale_chunk = commit(
            &dataset,
            Transaction::new(version, compaction_update(vec![], Some(DIR_A)), None),
        )
        .await
        .expect_err("a progress-less chunk from the replaced MemWAL");
        assert!(stale_chunk.is_stale_mem_wal(), "unexpected: {stale_chunk}");

        let unnamed = commit(
            &dataset,
            Transaction::new(
                version,
                compaction_update(vec![CompactedSsTable::new(shard, 1)], None),
                None,
            ),
        )
        .await
        .expect_err("compaction progress must name its MemWAL");
        assert!(
            unnamed.to_string().contains("mem_wal_dir_id"),
            "unexpected: {unnamed}"
        );

        commit(
            &dataset,
            Transaction::new(
                version,
                compaction_update(vec![CompactedSsTable::new(shard, 1)], Some(DIR_B)),
                None,
            ),
        )
        .await
        .expect("a compaction from the current MemWAL commits");
    }

    /// Every merge-insert executor carries the MemWAL named by
    /// `mark_sstables_as_compacted` into its commit, including the in-place
    /// column rewrite and chunks that record no progress.
    #[rstest::rstest]
    #[case::rewrite_rows(MergeInsertWriteMode::RewriteRows)]
    #[case::rewrite_columns(MergeInsertWriteMode::RewriteColumns)]
    #[tokio::test]
    async fn test_merge_insert_checks_its_mem_wal(#[case] write_mode: MergeInsertWriteMode) {
        let initial = record_batch!(
            ("id", Int32, [1, 2, 3, 4]),
            ("value", Int32, [10, 20, 30, 40]),
            ("other", Int32, [100, 200, 300, 400])
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            })
            .execute(vec![initial])
            .await
            .unwrap();
        let dataset = Arc::new(enable(disable(enable(dataset, DIR_A).await).await, DIR_B).await);

        let merge = |dir_id: &str, compacted: Vec<CompactedSsTable>, value: i32| {
            let dataset = dataset.clone();
            let dir_id = dir_id.to_string();
            async move {
                let source = record_batch!(("id", Int32, [1]), ("value", Int32, [value])).unwrap();
                let uncommitted = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".into()])
                    .unwrap()
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::DoNothing)
                    .use_index(false)
                    .write_mode(write_mode)
                    .mark_sstables_as_compacted(dir_id, compacted)
                    .try_build()
                    .unwrap()
                    .execute_uncommitted_batches(vec![source])
                    .await
                    .unwrap();
                CommitBuilder::new(dataset)
                    .execute(uncommitted.transaction)
                    .await
            }
        };

        let stale = merge(DIR_A, vec![], 999)
            .await
            .expect_err("a progress-less chunk from the replaced MemWAL");
        assert!(stale.is_stale_mem_wal(), "unexpected: {stale}");

        let merged = merge(DIR_B, vec![CompactedSsTable::new(Uuid::new_v4(), 1)], 11)
            .await
            .expect("progress for the current MemWAL commits");
        let batch = merged
            .scan()
            .filter("id = 1")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            batch["value"].as_primitive::<Int32Type>().values(),
            &[11],
            "only the current MemWAL's merge lands"
        );
    }

    #[tokio::test]
    async fn test_enable_rejected_when_a_mem_wal_is_present() {
        let enabled = enable(test_dataset().await, DIR_A).await;

        // Built on the HEAD that already carries a MemWAL, so no rebase runs and
        // only the manifest-build check can refuse it.
        let index = new_mem_wal_index_meta(
            enabled.manifest.version,
            MemWalIndexDetails {
                dir_id: DIR_B.to_string(),
                ..Default::default()
            },
        )
        .unwrap();
        let error = commit(
            &enabled,
            Transaction::new(
                enabled.manifest.version,
                Operation::CreateIndex {
                    new_indices: vec![index],
                    removed_indices: vec![],
                },
                None,
            ),
        )
        .await
        .expect_err("a second enable must not replace the first");
        // A same-name index is folded in as a replacement, so this reads as a
        // rewrite naming another MemWAL.
        assert!(
            error.is_stale_mem_wal() || error.to_string().contains("already initialized"),
            "unexpected: {error}"
        );
        let details = enabled.mem_wal_index_details().await.unwrap().unwrap();
        assert_eq!(details.dir_id, DIR_A);
    }
}
