// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// Lineage overview with annotated base versions:
// - branch1 is created from main@v1
// - branch4 is created from main@v2 (after main receives a second write)
// - dev/branch2 is created from branch1@latest
// - feature/nathan/branch3 is created from dev/branch2@latest
//
// ASCII lineage with versions:
//    main:v1 ──▶ branch1:v1 ──▶ dev/branch2:v2 ──▶ feature/nathan/branch3:v3
//        │
//    (main:v2) ──▶ branch4:v2
//
// Cleanup policy focus (unless explicitly overridden in a test):
// - retain_n_versions = 1: keep the latest manifest per branch
// - referenced branches: when enabled, protect parent files referenced by descendants
// - retained source versions keep their complete snapshots, including indices
// - file counts reported per branch:
//   manifest: number of manifest files under _versions
//   data: .lance files under data directory
//   tx: .txn files under _transactions, including retained source versions
//   delete: deletion files count under _deletions
//   index: index files count under _indices
//
// Note: branch2 is stored as "dev/branch2"; comments may refer to it as branch2 for brevity.
// Important: auto_cleanup_hook uses policy derived from manifest config; it does not flip
// clean_referenced_branches unless tests call cleanup_old_versions with a custom policy.
struct LineageSetup {
    main: BranchDatasetFixture,
    branch1: BranchDatasetFixture,
    branch2: BranchDatasetFixture,
    branch3: BranchDatasetFixture,
    branch4: BranchDatasetFixture,
}

impl LineageSetup {
    /// Assert all branches and main are unchanged since last refresh.
    pub async fn assert_all_unchanged(&mut self) {
        self.main.assert_not_changed().await.unwrap();
        self.branch1.assert_not_changed().await.unwrap();
        self.branch2.assert_not_changed().await.unwrap();
        self.branch3.assert_not_changed().await.unwrap();
        self.branch4.assert_not_changed().await.unwrap();
    }

    /// Assert specified branches are unchanged.
    pub async fn assert_unchanged(&mut self, branches: &[&str]) {
        for &b in branches {
            match b {
                "main" => self.main.assert_not_changed().await.unwrap(),
                "branch1" => self.branch1.assert_not_changed().await.unwrap(),
                "branch2" => self.branch2.assert_not_changed().await.unwrap(),
                "branch3" => self.branch3.assert_not_changed().await.unwrap(),
                "branch4" => self.branch4.assert_not_changed().await.unwrap(),
                _ => panic!("unknown branch: {}", b),
            }
        }
    }

    pub async fn enable_auto_cleanup(&mut self) -> Result<()> {
        let updates = [
            ("lance.auto_cleanup.interval", "1"),
            ("lance.auto_cleanup.retain_versions", "1"),
            ("lance.auto_cleanup.referenced_branch", "true"),
        ];
        self.main.dataset.update_config(updates).await?;
        self.branch1.dataset.update_config(updates).await?;
        self.branch2.dataset.update_config(updates).await?;
        self.branch3.dataset.update_config(updates).await?;
        self.branch4.dataset.update_config(updates).await?;
        self.main.refresh().await?;
        self.branch1.refresh().await?;
        self.branch2.refresh().await?;
        self.branch3.refresh().await?;
        self.branch4.refresh().await?;
        Ok(())
    }

    pub async fn disable_auto_cleanup(&mut self) -> Result<()> {
        let updates = [
            ("lance.auto_cleanup.interval", None),
            ("lance.auto_cleanup.retain_versions", None),
            ("lance.auto_cleanup.older_than", None),
        ];
        self.main.dataset.update_config(updates).await?;
        self.branch1.dataset.update_config(updates).await?;
        self.branch2.dataset.update_config(updates).await?;
        self.branch3.dataset.update_config(updates).await?;
        self.branch4.dataset.update_config(updates).await?;
        self.main.refresh().await?;
        self.branch1.refresh().await?;
        self.branch2.refresh().await?;
        self.branch3.refresh().await?;
        self.branch4.refresh().await?;
        Ok(())
    }
}

// Build the lineage and configure per-branch auto-cleanup to retain latest version.
async fn build_lineage_datasets() -> Result<LineageSetup> {
    let fixture = Arc::new(MockDatasetFixture::try_new()?);

    MockClock::set_system_time(TimeDelta::try_seconds(1).unwrap().to_std().unwrap());

    // Create main (initial write) with id and text columns for inverted index
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field};
    let ids = Int32Array::from_iter_values(0..50i32);
    let texts = StringArray::from_iter_values((0..50i32).map(|i| format!("text_{}", i)));
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(texts)]).unwrap();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

    Dataset::write(
        reader,
        &fixture.dataset_path,
        Some(WriteParams {
            mode: WriteMode::Create,
            store_params: Some(fixture.os_params()),
            ..Default::default()
        }),
    )
    .await?;
    let mut main = BranchDatasetFixture::new(fixture.clone(), fixture.load().await?);
    // Initial index creation and refresh counts
    main.create_text_index().await?;
    main.write_data().await?;

    // Create branch1 from main@v1, then do an initial append + deterministic delete
    let mut branch1 = BranchDatasetFixture::new(
        fixture.clone(),
        fixture
            .create_branch_and_load(&mut main.dataset, "branch1", (None, None))
            .await?,
    );
    branch1.write_data().await?;

    // Create branch2 from branch1@latest
    let mut branch2 = BranchDatasetFixture::new(
        fixture.clone(),
        fixture
            .create_branch_and_load(&mut branch1.dataset, "dev/branch2", ("branch1", None))
            .await?,
    );
    branch2.write_data().await?;

    // Create branch3 from branch2@latest, initial append + delete
    let mut branch3 = BranchDatasetFixture::new(
        fixture.clone(),
        fixture
            .create_branch_and_load(
                &mut branch2.dataset,
                "feature/nathan/branch3",
                ("dev/branch2", None),
            )
            .await?,
    );
    branch3.write_data().await?;

    // Create branch4 from a new version in main
    main.write_data().await?;
    let mut branch4 = BranchDatasetFixture::new(
        fixture.clone(),
        fixture
            .create_branch_and_load(&mut main.dataset, "branch4", (None, None))
            .await?,
    );
    branch4.write_data().await?;

    let mut lineage = LineageSetup {
        main,
        branch1,
        branch2,
        branch3,
        branch4,
    };

    lineage.disable_auto_cleanup().await?;
    Ok(lineage)
}

// BranchDatasetFixture combines dataset with branch-specific state and file counting.
// It provides:
// - Shared fixture for temporary directory and mock store
// - Dataset holding for stateful operations (checkout, write, etc.)
// - File counting for cleanup verification
struct BranchDatasetFixture {
    fixture: Arc<MockDatasetFixture>,
    dataset: Dataset,
    counts: FileCounts,
}

impl BranchDatasetFixture {
    fn new(fixture: Arc<MockDatasetFixture>, dataset: Dataset) -> Self {
        Self {
            fixture,
            dataset,
            counts: FileCounts {
                num_manifest_files: 0,
                num_data_files: 0,
                num_tx_files: 0,
                num_delete_files: 0,
                num_index_files: 0,
                num_bytes: 0,
            },
        }
    }

    // Create a full-text index (Inverted) on the "text" column once.
    // We only create this on main during dataset creation. Branches inherit the index configuration.
    async fn create_text_index(&mut self) -> Result<()> {
        use crate::index::DatasetIndexExt;
        use lance_index::IndexType;
        use lance_index::scalar::InvertedIndexParams;
        let params = InvertedIndexParams::default();
        self.dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await?;
        Ok(())
    }

    // Append a batch, then read exactly one row and delete that row; finally optimize indices.
    async fn append_delete_and_optimize_index(&mut self) -> Result<()> {
        // Append a small batch with id and text columns
        self.write_batch(5).await?;
        // Delete the last row to create a deletion file
        self.delete_last_row().await?;
        // Optimize indices after write and delete
        use lance_index::optimize::OptimizeOptions;
        self.dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await?;
        Ok(())
    }

    // Append a batch with id and text columns.
    async fn write_batch(&mut self, rows: i32) -> Result<()> {
        use crate::dataset::WriteParams;
        use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
        use arrow_schema::{DataType, Field};

        let ids = Int32Array::from_iter_values(0..rows);
        let texts = StringArray::from_iter_values((0..rows).map(|i| format!("text_{}", i)));
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(texts)]).unwrap();
        let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        self.dataset
            .append(
                reader,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    store_params: Some(self.fixture.os_params()),
                    ..Default::default()
                }),
            )
            .await?;
        self.dataset.checkout_latest().await?;
        Ok(())
    }

    // Delete the last row to generate a deletion file.
    async fn delete_last_row(&mut self) -> Result<()> {
        let batch = self.dataset.scan().with_row_id().try_into_batch().await?;
        if batch.num_rows() > 0 {
            let row_id_col = batch.column_by_name(lance_core::ROW_ID).unwrap();
            let uint64_array = row_id_col.as_any().downcast_ref::<UInt64Array>().unwrap();
            let max_row_id = compute::max(uint64_array).unwrap_or(0);
            self.dataset
                .delete(&format!("_rowid = {}", max_row_id))
                .await?;
        }
        Ok(())
    }

    // Update counters by listing authoritative branch directories instead of reading the latest manifest.
    async fn refresh(&mut self) -> Result<()> {
        use futures::TryStreamExt;
        let branch_path = self.dataset.base.clone();

        // Count files in a directory, filtering by optional extension(s).
        async fn count_dir(os: &ObjectStore, dir: &Path, exts: Option<&[&str]>) -> Result<usize> {
            let mut count = 0usize;
            let mut s = os.read_dir_all(dir, None);
            while let Some(meta) = s.try_next().await? {
                match exts {
                    Some(exts) => {
                        if let Some(e) = meta.location.extension()
                            && exts.contains(&e)
                        {
                            count += 1;
                        }
                    }
                    None => count += 1,
                }
            }
            Ok(count)
        }

        let manifest_dir = branch_path.clone().join("_versions");
        self.counts.num_manifest_files = count_dir(
            &self.dataset.object_store,
            &manifest_dir,
            Some(&["manifest"]),
        )
        .await
        .unwrap_or(0);

        // Transactions: count files under _transactions (extension .txn)
        let txn_dir = branch_path.clone().join("_transactions");
        self.counts.num_tx_files = count_dir(&self.dataset.object_store, &txn_dir, Some(&["txn"]))
            .await
            .unwrap_or(0);

        // Indices: count files under _indices
        let idx_dir = branch_path.clone().join(crate::dataset::INDICES_DIR);
        self.counts.num_index_files = count_dir(&self.dataset.object_store, &idx_dir, None)
            .await
            .unwrap_or(0);

        // Deletions: count files under _deletions (extensions .arrow / .bin)
        let del_dir = branch_path.clone().join("_deletions");
        self.counts.num_delete_files = count_dir(
            &self.dataset.object_store,
            &del_dir,
            Some(&["arrow", "bin"]),
        )
        .await
        .unwrap_or(0);

        // Data files: count .lance files under data/
        let data_dir = branch_path.clone().join(crate::dataset::DATA_DIR);
        self.counts.num_data_files =
            count_dir(&self.dataset.object_store, &data_dir, Some(&["lance"]))
                .await
                .unwrap_or(0);

        Ok(())
    }

    async fn count_data(&self) -> Result<usize> {
        use futures::TryStreamExt;
        let mut count = 0usize;
        let mut s = self.dataset.scan().try_into_stream().await?;
        while let Some(_batch) = s.try_next().await? {
            count += 1;
        }
        Ok(count)
    }

    // Strict equality assertion for all counters.
    async fn assert_not_changed(&mut self) -> Result<()> {
        let pre_counts = self.counts;
        let pre_data_count = self.count_data().await?;

        self.refresh().await?;
        assert_eq!(
            self.counts.num_manifest_files,
            pre_counts.num_manifest_files
        );
        assert_eq!(self.counts.num_data_files, pre_counts.num_data_files);
        assert_eq!(self.counts.num_tx_files, pre_counts.num_tx_files);
        assert_eq!(self.counts.num_delete_files, pre_counts.num_delete_files);
        assert_eq!(self.counts.num_index_files, pre_counts.num_index_files);
        assert_eq!(self.count_data().await?, pre_data_count);
        Ok(())
    }

    // Append, delete top row, and optimize indices.
    async fn write_data(&mut self) -> Result<()> {
        self.append_delete_and_optimize_index().await?;
        self.refresh().await
    }

    // Compact files for a given branch and optimize indices to stabilize index files.
    async fn compact(&mut self) -> Result<()> {
        use crate::dataset::optimize::{CompactionOptions, compact_files};
        compact_files(&mut self.dataset, CompactionOptions::default(), None).await?;
        self.refresh().await
    }

    async fn run_cleanup(&mut self) -> Result<RemovalStats> {
        let policy = CleanupPolicyBuilder::default()
            .error_if_tagged_old_versions(false)
            .retain_n_versions(&self.dataset, 1)
            .await?
            .build();
        self.run_cleanup_inner(policy).await
    }

    async fn run_cleanup_with_referenced_branches(&mut self) -> Result<RemovalStats> {
        let policy = CleanupPolicyBuilder::default()
            .error_if_tagged_old_versions(false)
            .clean_referenced_branches(true)
            .retain_n_versions(&self.dataset, 1)
            .await?
            .build();
        self.run_cleanup_inner(policy).await
    }

    async fn explain_cleanup_with_referenced_branches(&mut self) -> Result<CleanupExplanation> {
        let policy = CleanupPolicyBuilder::default()
            .error_if_tagged_old_versions(false)
            .clean_referenced_branches(true)
            .retain_n_versions(&self.dataset, 1)
            .await?
            .build();
        self.dataset.checkout_latest().await?;
        self.dataset.cleanup(policy).explain().await
    }

    async fn run_cleanup_inner(&mut self, policy: CleanupPolicy) -> Result<RemovalStats> {
        let pre_count = self.count_data().await?;
        self.dataset.checkout_latest().await?;
        let stats = cleanup_old_versions(&self.dataset, policy).await;
        self.refresh().await?;
        // Assert data could be read again and did't change
        assert_eq!(self.count_data().await?, pre_count);
        stats
    }
}

// ===================== Tests =====================
#[tokio::test]
async fn cleanup_lineage_branch1() {
    let mut setup = build_lineage_datasets().await.unwrap();

    setup.branch1.write_data().await.unwrap();
    setup.branch1.run_cleanup().await.unwrap();
    // Branch2 and branch3 hold references from branch1:
    // - 1 manifest file
    // - 1 data file
    // - 1 deletion file
    // - 4 index files
    // The left is the counts for the latest version of appending
    assert_eq!(setup.branch1.counts.num_manifest_files, 2);
    assert_eq!(setup.branch1.counts.num_data_files, 2);
    assert_eq!(setup.branch1.counts.num_tx_files, 2);
    assert_eq!(setup.branch1.counts.num_delete_files, 2);
    assert_eq!(setup.branch1.counts.num_index_files, 14);
    setup.assert_all_unchanged().await;

    setup.branch1.compact().await.unwrap();
    setup.branch1.run_cleanup().await.unwrap();
    // Branch2 and branch3 hold references from branch1:
    // - 1 manifest file
    // - 1 data file
    // - 1 deletion file
    // - 4 index files
    // The left (1, 1, 1, 0, 4) is the counts for the latest version of compaction
    assert_eq!(setup.branch1.counts.num_manifest_files, 2);
    assert_eq!(setup.branch1.counts.num_data_files, 2);
    assert_eq!(setup.branch1.counts.num_tx_files, 2);
    assert_eq!(setup.branch1.counts.num_delete_files, 1);
    assert_eq!(setup.branch1.counts.num_index_files, 14);
    setup.assert_all_unchanged().await;

    // Now we clean the referenced files of branch1 by branch2 and branch3
    setup.branch2.compact().await.unwrap();
    setup.branch3.compact().await.unwrap();
    setup.branch3.run_cleanup().await.unwrap();
    setup.branch2.run_cleanup().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version of compaction
    assert_eq!(setup.branch2.counts.num_manifest_files, 1);
    assert_eq!(setup.branch2.counts.num_data_files, 1);
    assert_eq!(setup.branch2.counts.num_tx_files, 1);
    assert_eq!(setup.branch2.counts.num_delete_files, 0);
    assert_eq!(setup.branch2.counts.num_index_files, 7);
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version of compaction
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 1);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 0);
    assert_eq!(setup.branch3.counts.num_index_files, 7);
    setup.branch1.run_cleanup().await.unwrap();

    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version of compaction
    assert_eq!(setup.branch1.counts.num_manifest_files, 1);
    assert_eq!(setup.branch1.counts.num_data_files, 1);
    assert_eq!(setup.branch1.counts.num_tx_files, 1);
    assert_eq!(setup.branch1.counts.num_delete_files, 0);
    assert_eq!(setup.branch1.counts.num_index_files, 7);
    setup.assert_all_unchanged().await;
}

#[tokio::test]
async fn cleanup_lineage_branch3() {
    let mut setup = build_lineage_datasets().await.unwrap();

    setup.branch3.write_data().await.unwrap();
    setup.branch3.run_cleanup().await.unwrap();
    // Two writes produced:
    // - 2 data files
    // - 2 deletion files
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 2);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 2);
    assert_eq!(setup.branch3.counts.num_index_files, 7);
    setup
        .assert_unchanged(&["branch1", "branch2", "branch4", "main"])
        .await;

    setup.branch2.compact().await.unwrap();
    setup.branch2.run_cleanup().await.unwrap();
    // Branch3 hold references from branch2:
    // - 1 manifest file
    // - 1 data file
    // - 1 deletion file
    // The left is the counts for the latest version of compaction
    assert_eq!(setup.branch2.counts.num_manifest_files, 2);
    assert_eq!(setup.branch2.counts.num_data_files, 2);
    assert_eq!(setup.branch2.counts.num_tx_files, 2);
    assert_eq!(setup.branch2.counts.num_delete_files, 1);
    assert_eq!(setup.branch2.counts.num_index_files, 14);

    setup.branch3.compact().await.unwrap();
    setup.branch3.run_cleanup().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 1);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 0);
    assert_eq!(setup.branch3.counts.num_index_files, 7);
    setup
        .assert_unchanged(&["branch1", "branch2", "branch4", "main"])
        .await;

    setup.branch2.compact().await.unwrap();
    setup.branch2.run_cleanup().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version
    assert_eq!(setup.branch2.counts.num_manifest_files, 1);
    assert_eq!(setup.branch2.counts.num_data_files, 1);
    assert_eq!(setup.branch2.counts.num_tx_files, 1);
    assert_eq!(setup.branch2.counts.num_delete_files, 0);
    assert_eq!(setup.branch2.counts.num_index_files, 7);
}

#[tokio::test]
async fn cleanup_lineage_branch4() {
    // Setup shared lineage and per-branch auto-clean config
    let mut setup = build_lineage_datasets().await.unwrap();

    setup.branch4.write_data().await.unwrap();
    setup.branch4.run_cleanup().await.unwrap();
    // Two writes produced:
    // - 2 data files
    // - 2 deletion files
    assert_eq!(setup.branch4.counts.num_manifest_files, 1);
    assert_eq!(setup.branch4.counts.num_data_files, 2);
    assert_eq!(setup.branch4.counts.num_tx_files, 1);
    assert_eq!(setup.branch4.counts.num_delete_files, 2);
    assert_eq!(setup.branch4.counts.num_index_files, 7);
    setup.assert_all_unchanged().await;

    setup.main.compact().await.unwrap();
    setup.main.run_cleanup().await.unwrap();
    // Branch1-branch2 hold references from main:
    // - 1 manifest file
    // - 2 data files
    // - 1 deletion file
    // - 4 index files
    // Branch4 holds references from main:
    // - 1 manifest file
    // - 3 data files
    // - 1 deletion file
    // - 4 index files
    // The left(1, 1, 1, 0, 0) is the counts for the latest version of compaction
    assert_eq!(setup.main.counts.num_manifest_files, 3);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 3);
    assert_eq!(setup.main.counts.num_delete_files, 2);
    assert_eq!(setup.main.counts.num_index_files, 21);

    setup.branch4.compact().await.unwrap();
    setup.branch4.run_cleanup().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts of one version
    assert_eq!(setup.branch4.counts.num_manifest_files, 1);
    assert_eq!(setup.branch4.counts.num_data_files, 1);
    assert_eq!(setup.branch4.counts.num_tx_files, 1);
    assert_eq!(setup.branch4.counts.num_delete_files, 0);
    assert_eq!(setup.branch4.counts.num_index_files, 7);
    setup.assert_all_unchanged().await;

    setup.main.run_cleanup().await.unwrap();
    // Branch1-branch2 hold references from main:
    // - 1 manifest file
    // - 2 data files
    // - 1 deletion file
    // - 4 index files
    // The left(1, 1, 1, 0, 4) is the counts for the latest version of compaction
    assert_eq!(setup.main.counts.num_manifest_files, 2);
    assert_eq!(setup.main.counts.num_data_files, 3);
    assert_eq!(setup.main.counts.num_tx_files, 2);
    assert_eq!(setup.main.counts.num_delete_files, 1);
    assert_eq!(setup.main.counts.num_index_files, 14);
}

#[tokio::test]
async fn cleanup_lineage_main() {
    // Setup shared lineage and per-branch auto-clean config
    let mut setup = build_lineage_datasets().await.unwrap();

    setup.main.write_data().await.unwrap();
    setup.main.run_cleanup().await.unwrap();
    // Branch1-branch2 hold references from main:
    // - 1 manifest file
    // - 2 data files
    // - 1 deletion file
    // - 4 index files(only for branch1)
    // Branch4 holds references from main:
    // - 1 manifest file
    // - 3 data files
    // - 1 deletion file
    // - 4 index files
    // The left(1, 1, 1, 1, 4) is the counts for the latest version of compaction
    assert_eq!(setup.main.counts.num_manifest_files, 3);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 3);
    assert_eq!(setup.main.counts.num_delete_files, 3);
    assert_eq!(setup.main.counts.num_index_files, 21);
    setup.assert_all_unchanged().await;

    setup.main.compact().await.unwrap();
    setup.main.run_cleanup().await.unwrap();
    // Cleanup the deletion file
    // Produce 1 datafile and cleanup 1
    assert_eq!(setup.main.counts.num_manifest_files, 3);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 3);
    assert_eq!(setup.main.counts.num_delete_files, 2);
    assert_eq!(setup.main.counts.num_index_files, 21);
    setup.assert_all_unchanged().await;

    setup.branch1.write_data().await.unwrap();
    setup.branch1.compact().await.unwrap();
    setup.branch2.write_data().await.unwrap();
    setup.branch2.compact().await.unwrap();
    setup.branch2.run_cleanup().await.unwrap();
    // Branch3 holds references from branch2:
    // - 1 manifest file
    // - 1 data files
    // - 1 deletion file
    // Branch3 holds reference from branch1:
    // - 1 manifest file
    // - 1 data files
    // - 2 deletion files
    // - 4 index files
    assert_eq!(setup.branch2.counts.num_manifest_files, 2);
    assert_eq!(setup.branch2.counts.num_data_files, 2);
    assert_eq!(setup.branch2.counts.num_tx_files, 2);
    assert_eq!(setup.branch2.counts.num_delete_files, 1);
    assert_eq!(setup.branch2.counts.num_index_files, 14);
    setup.branch1.run_cleanup().await.unwrap();
    // The retained branch root still keeps its own index files.
    assert_eq!(setup.branch1.counts.num_manifest_files, 2);
    assert_eq!(setup.branch1.counts.num_data_files, 2);
    assert_eq!(setup.branch1.counts.num_tx_files, 2);
    assert_eq!(setup.branch1.counts.num_delete_files, 1);
    assert_eq!(setup.branch1.counts.num_index_files, 14);

    setup.main.run_cleanup().await.unwrap();
    // Branch3 holds references from main:
    // - 1 manifest file
    // - 1 data files
    // - 1 deletion file
    // Branch4 holds references from main:
    // - 1 manifest file
    // - 3 data files
    // - 2 deletion files
    // - 4 index files
    assert_eq!(setup.main.counts.num_manifest_files, 3);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 3);
    assert_eq!(setup.main.counts.num_delete_files, 2);
    assert_eq!(setup.main.counts.num_index_files, 21);

    setup.branch3.write_data().await.unwrap();
    setup.branch3.compact().await.unwrap();
    setup.branch3.run_cleanup().await.unwrap();
    // Only the counts for the latest version
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 1);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 0);
    assert_eq!(setup.branch3.counts.num_index_files, 7);

    setup.main.run_cleanup().await.unwrap();
    // Cleanup doesn't take effects if we don't clean branch2 and branch1 first
    assert_eq!(setup.main.counts.num_manifest_files, 3);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 3);
    assert_eq!(setup.main.counts.num_delete_files, 2);
    assert_eq!(setup.main.counts.num_index_files, 21);

    // Cleanup doesn't take effect if we don't clean branch2 first
    setup.branch1.run_cleanup().await.unwrap();
    assert_eq!(setup.branch1.counts.num_manifest_files, 2);
    assert_eq!(setup.branch1.counts.num_data_files, 2);
    assert_eq!(setup.branch1.counts.num_tx_files, 2);
    assert_eq!(setup.branch1.counts.num_delete_files, 1);
    assert_eq!(setup.branch1.counts.num_index_files, 14);

    setup.branch2.run_cleanup().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version
    assert_eq!(setup.branch2.counts.num_manifest_files, 1);
    assert_eq!(setup.branch2.counts.num_data_files, 1);
    assert_eq!(setup.branch2.counts.num_tx_files, 1);
    assert_eq!(setup.branch2.counts.num_delete_files, 0);
    assert_eq!(setup.branch2.counts.num_index_files, 7);

    setup.branch1.run_cleanup().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version
    assert_eq!(setup.branch1.counts.num_manifest_files, 1);
    assert_eq!(setup.branch1.counts.num_data_files, 1);
    assert_eq!(setup.branch1.counts.num_tx_files, 1);
    assert_eq!(setup.branch1.counts.num_delete_files, 0);
    assert_eq!(setup.branch1.counts.num_index_files, 7);

    setup.main.run_cleanup().await.unwrap();
    // Branch4 holds references from main:
    // - 1 manifest file
    // - 3 data files
    // - 2 deletion files
    // - 4 index files
    assert_eq!(setup.main.counts.num_manifest_files, 2);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 2);
    assert_eq!(setup.main.counts.num_delete_files, 2);
    assert_eq!(setup.main.counts.num_index_files, 14);

    setup.branch4.write_data().await.unwrap();
    setup.branch4.compact().await.unwrap();
    setup.branch4.run_cleanup().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version
    assert_eq!(setup.branch4.counts.num_manifest_files, 1);
    assert_eq!(setup.branch4.counts.num_data_files, 1);
    assert_eq!(setup.branch4.counts.num_tx_files, 1);
    assert_eq!(setup.branch4.counts.num_delete_files, 0);
    assert_eq!(setup.branch4.counts.num_index_files, 7);

    setup.main.run_cleanup().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts for the latest version
    assert_eq!(setup.main.counts.num_manifest_files, 1);
    assert_eq!(setup.main.counts.num_data_files, 1);
    assert_eq!(setup.main.counts.num_tx_files, 1);
    assert_eq!(setup.main.counts.num_delete_files, 0);
    assert_eq!(setup.main.counts.num_index_files, 7);
}

#[tokio::test]
async fn auto_clean_referenced_branches_from_branch2() {
    // Setup shared lineage and per-branch auto-clean config
    let mut setup = build_lineage_datasets().await.unwrap();

    setup.branch3.write_data().await.unwrap();
    setup.enable_auto_cleanup().await.unwrap();
    setup
        .branch2
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch3.refresh().await.unwrap();
    // Branch3 holds references from branch2:
    // - 1 manifest file
    // - 1 data file
    // - 1 deletion file
    assert_eq!(setup.branch2.counts.num_manifest_files, 2);
    assert_eq!(setup.branch2.counts.num_data_files, 1);
    assert_eq!(setup.branch2.counts.num_tx_files, 2);
    assert_eq!(setup.branch2.counts.num_delete_files, 1);
    assert_eq!(setup.branch2.counts.num_index_files, 7);
    // After auto-clean: branch3
    // 2 appends produced 2 data files
    // 2 deletes produced 2 deletion files
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 2);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 2);
    assert_eq!(setup.branch3.counts.num_index_files, 7);
    setup
        .assert_unchanged(&["branch1", "branch4", "main"])
        .await;

    setup.disable_auto_cleanup().await.unwrap();
    setup.branch2.write_data().await.unwrap();
    setup.branch2.compact().await.unwrap();
    setup.branch3.compact().await.unwrap();
    setup.enable_auto_cleanup().await.unwrap();
    setup
        .branch2
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch3.refresh().await.unwrap();
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts of one version
    assert_eq!(setup.branch2.counts.num_manifest_files, 1);
    assert_eq!(setup.branch2.counts.num_data_files, 1);
    assert_eq!(setup.branch2.counts.num_tx_files, 1);
    assert_eq!(setup.branch2.counts.num_delete_files, 0);
    assert_eq!(setup.branch2.counts.num_index_files, 7);
    // Only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts of one version
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 1);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 0);
    assert_eq!(setup.branch3.counts.num_index_files, 7);
    setup
        .assert_unchanged(&["branch1", "branch4", "main"])
        .await;
}

#[tokio::test]
async fn auto_clean_referenced_branches_from_main() {
    let mut setup = build_lineage_datasets().await.unwrap();

    setup.enable_auto_cleanup().await.unwrap();
    setup.main.write_data().await.unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    // Branch3, branch2 and branch1 hold references from main:
    // - 1 manifest file
    // - 2 data files
    // - 1 deletion file
    // Branch4 holds references from main:
    // - 1 manifest file
    // - 3 data files
    // - 1 deletion file
    // - 4 index files
    assert_eq!(setup.main.counts.num_manifest_files, 3);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 3);
    assert_eq!(setup.main.counts.num_delete_files, 3);
    assert_eq!(setup.main.counts.num_index_files, 21);

    setup.main.compact().await.unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    // Branch3, branch2 and branch1 hold references from main:
    // - 1 manifest file
    // - 2 data files
    // - 1 deletion file
    // Branch4 holds references from main:
    // - 1 manifest file
    // - 3 data files
    // - 1 deletion file
    assert_eq!(setup.main.counts.num_manifest_files, 3);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 3);
    assert_eq!(setup.main.counts.num_delete_files, 2);
    assert_eq!(setup.main.counts.num_index_files, 21);

    setup.branch4.compact().await.unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch4.refresh().await.unwrap();
    // Branch3, branch2 and branch1 hold references from main:
    // - 1 manifest file
    // - 2 data files
    // - 1 deletion file
    assert_eq!(setup.main.counts.num_manifest_files, 2);
    assert_eq!(setup.main.counts.num_data_files, 3);
    assert_eq!(setup.main.counts.num_tx_files, 2);
    assert_eq!(setup.main.counts.num_delete_files, 1);
    assert_eq!(setup.main.counts.num_index_files, 14);
    // (1, 1, 1, 0, 4) is the counts of one version
    assert_eq!(setup.branch4.counts.num_manifest_files, 1);
    assert_eq!(setup.branch4.counts.num_data_files, 1);
    assert_eq!(setup.branch4.counts.num_tx_files, 1);
    assert_eq!(setup.branch4.counts.num_delete_files, 0);
    assert_eq!(setup.branch4.counts.num_index_files, 7);

    setup.branch1.write_data().await.unwrap();
    setup.branch1.compact().await.unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch1.refresh().await.unwrap();
    // Branch3 and branch2 still hold references from main:
    // - 1 manifest file
    // - 2 data files
    // - 1 deletion file
    assert_eq!(setup.main.counts.num_manifest_files, 2);
    assert_eq!(setup.main.counts.num_data_files, 3);
    assert_eq!(setup.main.counts.num_tx_files, 2);
    assert_eq!(setup.main.counts.num_delete_files, 1);
    assert_eq!(setup.main.counts.num_index_files, 14);
    // Branch3 and branch2 still hold references from branch1:
    // - 1 manifest file
    // - 1 data files
    // - 1 deletion file
    assert_eq!(setup.branch1.counts.num_manifest_files, 2);
    assert_eq!(setup.branch1.counts.num_data_files, 2);
    assert_eq!(setup.branch1.counts.num_tx_files, 2);
    assert_eq!(setup.branch1.counts.num_delete_files, 1);
    assert_eq!(setup.branch1.counts.num_index_files, 14);

    setup.branch2.write_data().await.unwrap();
    setup.branch2.compact().await.unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch2.refresh().await.unwrap();
    // Branch3 still holds references from main:
    // - 1 manifest file
    // - 2 data files
    // - 1 deletion file
    assert_eq!(setup.main.counts.num_manifest_files, 2);
    assert_eq!(setup.main.counts.num_data_files, 3);
    assert_eq!(setup.main.counts.num_tx_files, 2);
    assert_eq!(setup.main.counts.num_delete_files, 1);
    assert_eq!(setup.main.counts.num_index_files, 14);
    // Branch3 still holds references from branch1:
    // - 1 manifest file
    // - 1 data files
    // - 1 deletion file
    assert_eq!(setup.branch1.counts.num_manifest_files, 2);
    assert_eq!(setup.branch1.counts.num_data_files, 2);
    assert_eq!(setup.branch1.counts.num_tx_files, 2);
    assert_eq!(setup.branch1.counts.num_delete_files, 1);
    assert_eq!(setup.branch1.counts.num_index_files, 14);
    // Branch3 still holds references from branch2:
    // - 1 manifest file
    // - 1 data files
    // - 1 deletion file
    assert_eq!(setup.branch2.counts.num_manifest_files, 2);
    assert_eq!(setup.branch2.counts.num_data_files, 2);
    assert_eq!(setup.branch2.counts.num_tx_files, 2);
    assert_eq!(setup.branch2.counts.num_delete_files, 1);
    assert_eq!(setup.branch2.counts.num_index_files, 14);

    setup.branch3.write_data().await.unwrap();
    setup.branch3.compact().await.unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch1.refresh().await.unwrap();
    setup.branch2.refresh().await.unwrap();
    setup.branch3.refresh().await.unwrap();
    // For all branches, only the latest manifest is retained.
    // (1, 1, 1, 0, 4) is the counts of one version
    assert_eq!(setup.main.counts.num_manifest_files, 1);
    assert_eq!(setup.main.counts.num_data_files, 1);
    assert_eq!(setup.main.counts.num_tx_files, 1);
    assert_eq!(setup.main.counts.num_delete_files, 0);
    assert_eq!(setup.main.counts.num_index_files, 7);
    assert_eq!(setup.branch1.counts.num_manifest_files, 1);
    assert_eq!(setup.branch1.counts.num_data_files, 1);
    assert_eq!(setup.branch1.counts.num_tx_files, 1);
    assert_eq!(setup.branch1.counts.num_delete_files, 0);
    assert_eq!(setup.branch1.counts.num_index_files, 7);
    assert_eq!(setup.branch2.counts.num_manifest_files, 1);
    assert_eq!(setup.branch2.counts.num_data_files, 1);
    assert_eq!(setup.branch2.counts.num_tx_files, 1);
    assert_eq!(setup.branch2.counts.num_delete_files, 0);
    assert_eq!(setup.branch2.counts.num_index_files, 7);
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 1);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 0);
    assert_eq!(setup.branch3.counts.num_index_files, 7);
    setup.assert_unchanged(&["branch4"]).await;
}

#[tokio::test]
async fn explain_cleanup_with_referenced_branches_matches_cleanup() {
    let mut setup = build_lineage_datasets().await.unwrap();

    setup.enable_auto_cleanup().await.unwrap();
    setup.main.write_data().await.unwrap();
    setup.main.compact().await.unwrap();
    setup.branch4.compact().await.unwrap();
    setup.branch1.write_data().await.unwrap();
    setup.branch1.compact().await.unwrap();
    setup.branch2.write_data().await.unwrap();
    setup.branch2.compact().await.unwrap();
    setup.branch3.write_data().await.unwrap();
    setup.branch3.compact().await.unwrap();

    setup.main.refresh().await.unwrap();
    setup.branch1.refresh().await.unwrap();
    setup.branch2.refresh().await.unwrap();
    setup.branch3.refresh().await.unwrap();
    setup.branch4.refresh().await.unwrap();
    let main_counts_before = setup.main.counts;
    let branch1_counts_before = setup.branch1.counts;
    let branch2_counts_before = setup.branch2.counts;
    let branch3_counts_before = setup.branch3.counts;
    let branch4_counts_before = setup.branch4.counts;

    let explanation = setup
        .main
        .explain_cleanup_with_referenced_branches()
        .await
        .unwrap();

    setup.main.refresh().await.unwrap();
    setup.branch1.refresh().await.unwrap();
    setup.branch2.refresh().await.unwrap();
    setup.branch3.refresh().await.unwrap();
    setup.branch4.refresh().await.unwrap();
    assert_eq!(setup.main.counts, main_counts_before);
    assert_eq!(setup.branch1.counts, branch1_counts_before);
    assert_eq!(setup.branch2.counts, branch2_counts_before);
    assert_eq!(setup.branch3.counts, branch3_counts_before);
    assert_eq!(setup.branch4.counts, branch4_counts_before);

    let removed = setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();

    assert!(!explanation.referenced_branches.is_empty());
    assert!(
        explanation
            .referenced_branches
            .iter()
            .any(|branch| branch.cleanup_candidate)
    );
    assert_eq!(explanation.stats, removed);
    setup.branch1.refresh().await.unwrap();
    setup.branch2.refresh().await.unwrap();
    setup.branch3.refresh().await.unwrap();
    setup.branch4.refresh().await.unwrap();
    assert_eq!(setup.main.counts.num_manifest_files, 1);
    assert_eq!(setup.branch1.counts.num_manifest_files, 1);
    assert_eq!(setup.branch2.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch4.counts.num_manifest_files, 1);
}

#[tokio::test]
async fn auto_clean_referenced_branches_with_tags() {
    let mut setup = build_lineage_datasets().await.unwrap();

    setup
        .branch3
        .dataset
        .tags()
        .create("branch3-tag", setup.branch3.dataset.version().version)
        .await
        .unwrap();
    setup
        .main
        .dataset
        .tags()
        .create("main-tag", setup.main.dataset.version().version)
        .await
        .unwrap();

    setup.branch1.compact().await.unwrap();
    setup.branch2.compact().await.unwrap();
    setup.branch3.compact().await.unwrap();
    setup.branch4.compact().await.unwrap();
    setup.main.compact().await.unwrap();
    setup.enable_auto_cleanup().await.unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch1.refresh().await.unwrap();
    setup.branch2.refresh().await.unwrap();
    setup.branch3.refresh().await.unwrap();
    setup.branch4.refresh().await.unwrap();
    // Two tags hold two manifest references
    // Main tag holds 1 tx file, 3 data files, 2 deletion files and 4 index files
    assert_eq!(setup.main.counts.num_manifest_files, 3);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 3);
    assert_eq!(setup.main.counts.num_delete_files, 2);
    assert_eq!(setup.main.counts.num_index_files, 21);
    // Branch3 tag holds branch1 with 1 tx file, 1 data files, 1 deletion files and 4 index files
    assert_eq!(setup.branch2.counts.num_manifest_files, 2);
    assert_eq!(setup.branch2.counts.num_data_files, 2);
    assert_eq!(setup.branch2.counts.num_tx_files, 2);
    assert_eq!(setup.branch2.counts.num_delete_files, 1);
    assert_eq!(setup.branch2.counts.num_index_files, 14);
    // Branch3 tag holds branch2 with 1 tx file, 1 data files, 1 deletion files and 4 index files
    assert_eq!(setup.branch2.counts.num_manifest_files, 2);
    assert_eq!(setup.branch2.counts.num_data_files, 2);
    assert_eq!(setup.branch2.counts.num_tx_files, 2);
    assert_eq!(setup.branch2.counts.num_delete_files, 1);
    assert_eq!(setup.branch2.counts.num_index_files, 14);
    assert_eq!(setup.branch4.counts.num_manifest_files, 1);
    assert_eq!(setup.branch4.counts.num_data_files, 1);
    assert_eq!(setup.branch4.counts.num_tx_files, 1);
    assert_eq!(setup.branch4.counts.num_delete_files, 0);
    assert_eq!(setup.branch4.counts.num_index_files, 7);

    setup
        .branch3
        .dataset
        .tags()
        .delete("branch3-tag")
        .await
        .unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch1.refresh().await.unwrap();
    setup.branch2.refresh().await.unwrap();
    setup.branch3.refresh().await.unwrap();
    setup.branch4.refresh().await.unwrap();
    // 1 manifest file referenced by branch3-tag is cleaned
    assert_eq!(setup.main.counts.num_manifest_files, 2);
    assert_eq!(setup.main.counts.num_data_files, 4);
    assert_eq!(setup.main.counts.num_tx_files, 2);
    assert_eq!(setup.main.counts.num_delete_files, 2);
    assert_eq!(setup.main.counts.num_index_files, 14);
    assert_eq!(setup.branch1.counts.num_manifest_files, 1);
    assert_eq!(setup.branch1.counts.num_data_files, 1);
    assert_eq!(setup.branch1.counts.num_tx_files, 1);
    assert_eq!(setup.branch1.counts.num_delete_files, 0);
    assert_eq!(setup.branch1.counts.num_index_files, 7);
    assert_eq!(setup.branch2.counts.num_manifest_files, 1);
    assert_eq!(setup.branch2.counts.num_data_files, 1);
    assert_eq!(setup.branch2.counts.num_tx_files, 1);
    assert_eq!(setup.branch2.counts.num_delete_files, 0);
    assert_eq!(setup.branch2.counts.num_index_files, 7);
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 1);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 0);
    assert_eq!(setup.branch3.counts.num_index_files, 7);
    assert_eq!(setup.branch4.counts.num_manifest_files, 1);
    assert_eq!(setup.branch4.counts.num_data_files, 1);
    assert_eq!(setup.branch4.counts.num_tx_files, 1);
    assert_eq!(setup.branch4.counts.num_delete_files, 0);
    assert_eq!(setup.branch4.counts.num_index_files, 7);

    setup.main.dataset.tags().delete("main-tag").await.unwrap();
    setup
        .main
        .run_cleanup_with_referenced_branches()
        .await
        .unwrap();
    setup.branch2.refresh().await.unwrap();
    setup.branch3.refresh().await.unwrap();
    setup.branch4.refresh().await.unwrap();
    // All cleaned up
    assert_eq!(setup.main.counts.num_manifest_files, 1);
    assert_eq!(setup.main.counts.num_data_files, 1);
    assert_eq!(setup.main.counts.num_tx_files, 1);
    assert_eq!(setup.main.counts.num_delete_files, 0);
    assert_eq!(setup.main.counts.num_index_files, 7);
    assert_eq!(setup.branch2.counts.num_manifest_files, 1);
    assert_eq!(setup.branch2.counts.num_data_files, 1);
    assert_eq!(setup.branch2.counts.num_tx_files, 1);
    assert_eq!(setup.branch2.counts.num_delete_files, 0);
    assert_eq!(setup.branch2.counts.num_index_files, 7);
    assert_eq!(setup.branch3.counts.num_manifest_files, 1);
    assert_eq!(setup.branch3.counts.num_data_files, 1);
    assert_eq!(setup.branch3.counts.num_tx_files, 1);
    assert_eq!(setup.branch3.counts.num_delete_files, 0);
    assert_eq!(setup.branch3.counts.num_index_files, 7);
    assert_eq!(setup.branch4.counts.num_manifest_files, 1);
    assert_eq!(setup.branch4.counts.num_data_files, 1);
    assert_eq!(setup.branch4.counts.num_tx_files, 1);
    assert_eq!(setup.branch4.counts.num_delete_files, 0);
    assert_eq!(setup.branch4.counts.num_index_files, 7);
}
