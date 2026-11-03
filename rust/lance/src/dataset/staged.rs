// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Append-only transactions using existing writers and Transaction V2.

use std::sync::Arc;

use arrow_array::RecordBatch;
use lance_datafusion::utils::StreamingWriteSource;
use lance_table::format::{DataFile, IndexMetadata, Manifest, is_detached_version};
use lance_table::io::commit::ManifestLocation;
use lance_table::transaction::action::{
    Action, AddDataFile, AddFragment, CompositeOperation, FieldRef, Ref, UserStep,
};
use uuid::Uuid;

use super::transaction::{Operation, Transaction};
use super::{CommitBuilder, InsertBuilder, ManifestWriteConfig, Scanner, WriteMode, WriteParams};
use crate::index::load_all_indices;
use crate::{Dataset, Error, Result};

/// An experimental append-only transaction committed as one dataset version.
///
/// Statements see a fixed committed base plus their earlier staged writes.
/// After issuing a scanner, a transaction with staged writes rejects any
/// intervening version. A transaction with no staged writes returns its original
/// snapshot without checking the latest version. Blind appends may rebase using
/// Transaction V2 conflict checks.
/// Concurrent named operations conflict with staged writes. Commit retries reuse
/// the prepared files without re-executing statements.
///
/// Failed staging preserves the last valid snapshot. Aborting or dropping the
/// transaction leaves files intact for earlier scanners. Orphan cleanup must not
/// overlap active transactions or readers: it can remove staged files or borrowed
/// blob objects. A later commit can then publish references to missing files.
/// Files from failed writes are reclaimed by orphan GC.
///
/// Commit consumes the transaction, including on an error. An error after
/// submission may mean publication succeeded. Do not resubmit its files or
/// delete them; the shared commit layer verifies outcomes when possible.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use arrow_array::RecordBatch;
/// # async fn example(dataset: &Dataset, batches: Vec<RecordBatch>) -> Result<()> {
/// let mut tx = dataset.begin_transaction().await?;
/// tx.append(batches).await?;
/// let rows = tx.scan()?.try_into_batch().await?;
/// let committed = tx.commit().await?;
/// # Ok(())
/// # }
/// ```
pub struct StagedTransaction {
    base: Arc<Dataset>,
    indices: Arc<Vec<IndexMetadata>>,
    id: String,
    state: Option<StagedState>,
}

struct StagedState {
    steps: Vec<UserStep>,
    snapshot: Arc<Dataset>,
    next_fragment: u32,
    requires_original_version: bool,
}

/// Unpublished index metadata retained by a staged snapshot. Its presence also
/// prevents direct publication through this Dataset. The index list must survive
/// cache eviction because no stored manifest can reload it.
#[derive(Debug)]
pub struct StagedSnapshotState {
    pub(crate) indices: Arc<Vec<IndexMetadata>>,
}

impl Dataset {
    pub(crate) fn ensure_published(&self) -> Result<()> {
        if self.staged_snapshot.is_some() {
            return Err(Error::invalid_input(
                "Cannot publish through a staged snapshot; commit its owning transaction instead",
            ));
        }
        Ok(())
    }

    /// Make an immutable staged snapshot without publishing any metadata.
    pub(crate) fn with_staged_manifest(
        &self,
        manifest: Manifest,
        indices: Vec<IndexMetadata>,
    ) -> Self {
        let identity = format!("private-{}", Uuid::new_v4());
        // Staged snapshots can share a version but contain different metadata.
        let location = ManifestLocation {
            version: manifest.version,
            path: self.base.clone().join("_staged").join(identity.as_str()),
            size: None,
            naming_scheme: self.manifest_location.naming_scheme,
            e_tag: Some(identity.clone()),
            identity: Some(identity),
        };
        let mut dataset = self.clone();
        if dataset.manifest.base_paths != manifest.base_paths {
            dataset.base_object_stores = Default::default();
        }
        dataset.fragment_bitmap =
            Arc::new(manifest.fragments.iter().map(|f| f.id as u32).collect());
        dataset.manifest = Arc::new(manifest);
        dataset.manifest_location = location;
        dataset.staged_snapshot = Some(Arc::new(StagedSnapshotState {
            indices: Arc::new(indices),
        }));
        dataset
    }

    /// Begin an experimental append-only transaction from this committed snapshot.
    ///
    /// See [`StagedTransaction`] for the transaction's behavior and an example.
    pub async fn begin_transaction(&self) -> Result<StagedTransaction> {
        self.ensure_published()?;
        if is_detached_version(self.manifest.version) {
            return Err(Error::not_supported(
                "Staged transactions require a mainline snapshot",
            ));
        }
        let indices = load_all_indices(self).await?;
        let snapshot =
            self.with_staged_manifest(self.manifest.as_ref().clone(), indices.as_ref().clone());
        Ok(StagedTransaction {
            base: Arc::new(self.clone()),
            indices,
            id: Uuid::new_v4().to_string(),
            state: Some(StagedState {
                steps: Vec::new(),
                snapshot: Arc::new(snapshot),
                next_fragment: 0,
                requires_original_version: false,
            }),
        })
    }
}

impl StagedTransaction {
    /// Transaction UUID used to verify an ambiguous commit outcome.
    ///
    /// After an unknown publication outcome, compare this with transactions
    /// returned by [`Dataset::read_transaction_by_version`]. Do not resubmit the
    /// prepared files merely because acknowledgement was lost.
    ///
    /// See [`StagedTransaction`] for an example.
    pub fn id(&self) -> &str {
        &self.id
    }

    fn active(&self) -> Result<&StagedState> {
        self.state
            .as_ref()
            .ok_or_else(|| Error::invalid_input("Staged transaction has been aborted"))
    }

    /// Scan an immutable staged snapshot using the ordinary Lance scanner.
    ///
    /// Earlier scanners retain their snapshot across later appends and logical abort.
    /// Once a scanner is issued, publishing staged writes requires an unchanged
    /// base version. A read-only commit returns the original snapshot.
    /// The version check preserves row IDs and addresses returned by the scanner.
    ///
    /// See [`StagedTransaction`] for an example.
    pub fn scan(&mut self) -> Result<Scanner> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| Error::invalid_input("Staged transaction has been aborted"))?;
        state.requires_original_version = true;
        Ok(state.snapshot.scan())
    }

    /// Append batches to the staged snapshot.
    ///
    /// An error leaves the previous staged snapshot usable. Use
    /// [`Self::append_stream`] for input larger than memory.
    ///
    /// See [`StagedTransaction`] for an example.
    pub async fn append(&mut self, batches: Vec<RecordBatch>) -> Result<()> {
        self.append_with_params(
            batches,
            &WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            },
        )
        .await
    }

    /// Stage batches with the existing writer's file, encoding, progress and blob options.
    ///
    /// `mode` must be [`WriteMode::Append`]. As with ordinary appends, the writer
    /// inherits the dataset's stable row ID setting regardless of
    /// `enable_stable_row_ids`. `store_params` configures external blob source reads;
    /// the destination store and session come from the dataset.
    ///
    /// Per-append `commit_handler`, `auto_cleanup`, `transaction_properties`,
    /// `session`, `base_store_params`, `initial_bases`, and target-base options
    /// are unsupported and must be `None`. `enable_v2_manifest_paths` and
    /// `skip_auto_cleanup` must retain their defaults; publication is controlled
    /// by [`Self::commit`]. Rejected settings leave the previous snapshot intact.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::{WriteMode, WriteParams};
    /// # use arrow_array::RecordBatch;
    /// # async fn example(dataset: &Dataset, batches: Vec<RecordBatch>) -> Result<()> {
    /// let mut tx = dataset.begin_transaction().await?;
    /// let params = WriteParams {
    ///     mode: WriteMode::Append,
    ///     max_rows_per_file: 10_000,
    ///     ..Default::default()
    /// };
    /// tx.append_with_params(batches, &params).await?;
    /// tx.commit().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn append_with_params(
        &mut self,
        batches: Vec<RecordBatch>,
        params: &WriteParams,
    ) -> Result<()> {
        self.validate_append_params(params)?;
        let written = InsertBuilder::new(self.active()?.snapshot.clone())
            .with_params(params)
            .execute_uncommitted(batches)
            .await?;
        self.stage_append(written)
    }

    /// Append a streaming source without collecting it into memory.
    ///
    /// A cancelled or failed write may leave orphan files, but does not replace
    /// the last valid staged snapshot. See [`StagedTransaction`] for an example of
    /// the transaction lifecycle; pass a [`StreamingWriteSource`] here instead
    /// of batches.
    pub async fn append_stream(&mut self, source: impl StreamingWriteSource) -> Result<()> {
        self.append_stream_with_params(
            source,
            &WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            },
        )
        .await
    }

    /// Stage a streaming source with the same options and restrictions as
    /// [`Self::append_with_params`]. Validation precedes consumption of the source.
    ///
    /// See [`Self::append_with_params`] for an example; pass a
    /// [`StreamingWriteSource`] instead of batches.
    pub async fn append_stream_with_params(
        &mut self,
        source: impl StreamingWriteSource,
        params: &WriteParams,
    ) -> Result<()> {
        self.validate_append_params(params)?;
        let written = InsertBuilder::new(self.active()?.snapshot.clone())
            .with_params(params)
            .execute_uncommitted_stream(source)
            .await?;
        self.stage_append(written)
    }

    fn validate_append_params(&self, params: &WriteParams) -> Result<()> {
        self.active()?;
        if !matches!(params.mode, WriteMode::Append) {
            return Err(Error::invalid_input(format!(
                "Staged append requires mode Append, got {:?}",
                params.mode
            )));
        }
        for (field, unsupported) in [
            ("commit_handler", params.commit_handler.is_some()),
            ("auto_cleanup", params.auto_cleanup.is_some()),
            (
                "transaction_properties",
                params.transaction_properties.is_some(),
            ),
            ("session", params.session.is_some()),
            ("base_store_params", params.base_store_params.is_some()),
            ("initial_bases", params.initial_bases.is_some()),
            ("target_bases", params.target_bases.is_some()),
            (
                "target_base_names_or_paths",
                params.target_base_names_or_paths.is_some(),
            ),
            ("target_all_bases", params.target_all_bases.is_some()),
            ("enable_v2_manifest_paths", !params.enable_v2_manifest_paths),
            ("skip_auto_cleanup", params.skip_auto_cleanup),
        ] {
            if unsupported {
                return Err(Error::invalid_input(format!(
                    "WriteParams.{field} cannot be set per staged append; use its default"
                )));
            }
        }
        Ok(())
    }

    fn transaction(&self, steps: Vec<UserStep>) -> Transaction {
        let mut transaction = Transaction::new(
            self.base.manifest.version,
            Operation::CompositeOperation(CompositeOperation::new(steps)),
            None,
        );
        transaction.uuid.clone_from(&self.id);
        transaction
    }

    fn stage_append(&mut self, written: Transaction) -> Result<()> {
        let Operation::Append { fragments } = written.operation else {
            return Err(Error::internal(
                "Uncommitted append returned a different operation",
            ));
        };
        let previous = self.active()?;
        let mut next_fragment = previous.next_fragment;
        let mut actions = Vec::new();
        for fragment in fragments {
            let token = next_fragment;
            next_fragment = next_fragment
                .checked_add(1)
                .ok_or_else(|| Error::invalid_input("Staged fragment token capacity exceeded"))?;
            let physical_rows = fragment
                .physical_rows
                .ok_or_else(|| Error::internal("New fragment has no physical row count"))?;
            actions.push(Action::AddFragment(AddFragment {
                id: Ref::Local(token),
                physical_rows: physical_rows as u64,
                row_id_meta: fragment.row_id_meta,
                last_updated_at_version_meta: fragment.last_updated_at_version_meta,
                created_at_version_meta: fragment.created_at_version_meta,
                data_change: true,
            }));
            for file in fragment.files {
                if file.base_id.is_some() {
                    return Err(Error::not_supported(
                        "Staged append to a secondary storage base is not supported",
                    ));
                }
                let field_ids = file
                    .fields
                    .iter()
                    .copied()
                    .map(FieldRef::Committed)
                    .collect();
                actions.push(Action::AddDataFile(AddDataFile {
                    fragment: Ref::Local(token),
                    file: DataFile {
                        fields: Arc::from([]),
                        ..file
                    },
                    field_ids,
                    data_change: true,
                    base: None,
                }));
            }
        }
        let mut steps = previous.steps.clone();
        steps.push(UserStep::new("append", actions));
        let config = ManifestWriteConfig::default().to_build_config();
        let (manifest, indices) = self.transaction(steps.clone()).build_manifest(
            Some(&self.base.manifest),
            self.indices.as_ref().clone(),
            "",
            &config,
        )?;
        let candidate = StagedState {
            steps,
            snapshot: Arc::new(self.base.with_staged_manifest(manifest, indices)),
            next_fragment,
            requires_original_version: previous.requires_original_version,
        };
        // Keep the previous snapshot if validation fails.
        self.state = Some(candidate);
        Ok(())
    }

    /// Publish all statements as one version; consume the transaction on failure.
    ///
    /// Transactions with no staged writes return their original snapshot, even
    /// after a scan or a concurrent commit. Read-only commits leave the dataset's
    /// current version unchanged. See [`StagedTransaction`] for an example and
    /// handling of unknown commit outcomes.
    pub async fn commit(mut self) -> Result<Dataset> {
        let state = self
            .state
            .take()
            .ok_or_else(|| Error::invalid_input("Staged transaction has been aborted"))?;
        if state.steps.is_empty() {
            return Ok(self.base.as_ref().clone());
        }
        // Earlier scanners may still need files that commit-triggered cleanup would remove.
        let mut builder = CommitBuilder::new(self.base.clone())
            .with_experimental_composite_operations(true)
            .with_skip_auto_cleanup(true);
        if state.requires_original_version {
            builder = builder.with_required_base_version(self.base.manifest.version);
        }
        builder.execute(self.transaction(state.steps)).await
    }

    /// Abort logically. Repeated aborts are harmless; future statements fail.
    ///
    /// No objects are deleted, including borrowed Managed blobs and files held
    /// by old scanners. Unpublished objects remain eligible for orphan GC.
    ///
    /// See [`StagedTransaction`] for an example.
    pub fn abort(&mut self) {
        self.state = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::write::WriteProgressFn;
    use crate::index::DatasetIndexExt;
    use arrow_array::record_batch;
    use datafusion::error::DataFusionError;
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
    use rstest::rstest;

    #[rstest]
    #[case::batch(false)]
    #[case::stream(true)]
    #[tokio::test]
    async fn append_params_preserve_layout_and_format(
        #[case] streaming: bool,
        #[values(false, true)] stable: bool,
        #[values(None, Some(false), Some(true))] stable_option: Option<bool>,
    ) {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                data_storage_version: Some(LanceFileVersion::V2_0),
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        let progress = Arc::new(std::sync::Mutex::new(Vec::new()));
        let reported = progress.clone();
        let mut params = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 2,
            max_rows_per_group: 1,
            data_storage_version: Some(LanceFileVersion::V2_1),
            write_progress: Some(WriteProgressFn::new(move |stats| {
                reported.lock().unwrap().push(stats);
            })),
            ..Default::default()
        };
        if let Some(stable_option) = stable_option {
            params.enable_stable_row_ids = stable_option;
        }
        let batch = record_batch!(("id", Int32, [1, 2, 3, 4, 5])).unwrap();
        let ordinary = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                data_storage_version: Some(LanceFileVersion::V2_0),
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let ordinary = InsertBuilder::new(Arc::new(ordinary))
            .with_params(&WriteParams {
                write_progress: None,
                ..params.clone()
            })
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        if streaming {
            let source = RecordBatchStreamAdapter::new(
                batch.schema(),
                futures::stream::iter(vec![Ok(batch)]),
            );
            tx.append_stream_with_params(Box::pin(source) as SendableRecordBatchStream, &params)
                .await
                .unwrap();
        } else {
            tx.append_with_params(vec![batch], &params).await.unwrap();
        }
        let first = tx.scan().unwrap();
        let snapshot = tx.active().unwrap().snapshot.clone();
        assert_eq!(
            snapshot
                .manifest
                .fragments
                .iter()
                .map(|f| f.physical_rows.unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 2, 1]
        );
        for fragment in snapshot.manifest.fragments.iter().skip(1) {
            assert_eq!(
                fragment.files[0].file_version().unwrap(),
                ConcreteFileVersion::V2_1
            );
        }
        assert_eq!(progress.lock().unwrap().last().unwrap().rows_written, 5);
        tx.append(vec![record_batch!(("id", Int32, [6])).unwrap()])
            .await
            .unwrap();
        assert_eq!(first.try_into_batch().await.unwrap().num_rows(), 6);
        let committed = tx.commit().await.unwrap();
        assert_eq!(committed.version().version, dataset.version().version + 1);
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(reopened.manifest.fragments, committed.manifest.fragments);
        assert_eq!(
            reopened.manifest.data_storage_format,
            dataset.manifest.data_storage_format
        );
        assert_eq!(reopened.manifest.uses_stable_row_ids(), stable);
        let ordinary = InsertBuilder::new(Arc::new(ordinary))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [6])).unwrap()])
            .await
            .unwrap();
        assert_eq!(
            reopened.manifest.uses_stable_row_ids(),
            ordinary.manifest.uses_stable_row_ids()
        );
        assert_eq!(reopened.manifest.next_row_id, ordinary.manifest.next_row_id);
        assert_eq!(
            reopened.manifest.data_storage_format,
            ordinary.manifest.data_storage_format
        );
        assert_eq!(
            reopened
                .scan()
                .with_row_id()
                .try_into_batch()
                .await
                .unwrap(),
            ordinary
                .scan()
                .with_row_id()
                .try_into_batch()
                .await
                .unwrap()
        );
        assert_eq!(
            reopened.scan().try_into_batch().await.unwrap(),
            record_batch!(("id", Int32, [0, 1, 2, 3, 4, 5, 6])).unwrap()
        );
        if stable {
            assert_eq!(
                reopened
                    .count_rows(Some("_rowid = id".into()))
                    .await
                    .unwrap(),
                7
            );
        }
    }

    #[rstest]
    #[case::create("mode_create")]
    #[case::overwrite("mode_overwrite")]
    #[case::handler("commit_handler")]
    #[case::cleanup("auto_cleanup")]
    #[case::properties("transaction_properties")]
    #[case::session("session")]
    #[case::base_stores("base_store_params")]
    #[case::initial_bases("initial_bases")]
    #[case::target_ids("target_bases")]
    #[case::target_names("target_base_names_or_paths")]
    #[case::all_bases("target_all_bases")]
    #[case::manifest_paths("enable_v2_manifest_paths")]
    #[case::skip_cleanup("skip_auto_cleanup")]
    #[tokio::test]
    async fn unsupported_append_params_do_not_write(
        #[case] field: &str,
        #[values(false, true)] streaming: bool,
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        let previous = tx.active().unwrap().snapshot.clone();
        let mut params = WriteParams {
            mode: WriteMode::Append,
            enable_stable_row_ids: stable,
            write_progress: Some(WriteProgressFn::new(|_| {
                panic!("invalid params reached writer")
            })),
            ..Default::default()
        };
        match field {
            "mode_create" => params.mode = WriteMode::Create,
            "mode_overwrite" => params.mode = WriteMode::Overwrite,
            "commit_handler" => params.commit_handler = Some(dataset.commit_handler.clone()),
            "auto_cleanup" => params.auto_cleanup = Some(Default::default()),
            "transaction_properties" => params.transaction_properties = Some(Default::default()),
            "session" => params.session = Some(dataset.session.clone()),
            "base_store_params" => params.base_store_params = Some(Default::default()),
            "initial_bases" => params.initial_bases = Some(vec![]),
            "target_bases" => params.target_bases = Some(vec![0]),
            "target_base_names_or_paths" => {
                params.target_base_names_or_paths = Some(vec![dataset.uri().to_string()])
            }
            "target_all_bases" => params.target_all_bases = Some(true),
            "enable_v2_manifest_paths" => params.enable_v2_manifest_paths = false,
            "skip_auto_cleanup" => params.skip_auto_cleanup = true,
            _ => unreachable!(),
        }
        let data_dir = format!("{}/data", dir.as_str());
        let files_before = std::fs::read_dir(&data_dir).unwrap().count();
        let error = if streaming {
            let source = RecordBatchStreamAdapter::new(
                record_batch!(("id", Int32, [2])).unwrap().schema(),
                futures::stream::poll_fn(|_| panic!("invalid params consumed the source")),
            );
            tx.append_stream_with_params(Box::pin(source) as SendableRecordBatchStream, &params)
                .await
        } else {
            tx.append_with_params(vec![record_batch!(("id", Int32, [2])).unwrap()], &params)
                .await
        }
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(if field.starts_with("mode_") {
            "mode"
        } else {
            field
        }));
        assert_eq!(std::fs::read_dir(&data_dir).unwrap().count(), files_before);
        assert!(Arc::ptr_eq(&previous, &tx.active().unwrap().snapshot));
        tx.append(vec![record_batch!(("id", Int32, [2])).unwrap()])
            .await
            .unwrap();
        assert_eq!(
            tx.commit().await.unwrap().count_rows(None).await.unwrap(),
            3
        );
    }

    #[rstest]
    #[case::batch(false)]
    #[case::stream(true)]
    #[tokio::test]
    async fn append_params_use_writer_validation(#[case] streaming: bool) {
        let dataset = InsertBuilder::new("memory://")
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        let previous = tx.active().unwrap().snapshot.clone();
        let params = WriteParams {
            mode: WriteMode::Append,
            file_writer_options: Some(lance_file::writer::FileWriterOptions {
                max_page_bytes: Some(0),
                ..Default::default()
            }),
            ..Default::default()
        };
        let batch = record_batch!(("id", Int32, [1])).unwrap();
        let error = if streaming {
            let source = RecordBatchStreamAdapter::new(
                batch.schema(),
                futures::stream::iter(vec![Ok(batch)]),
            );
            tx.append_stream_with_params(Box::pin(source) as SendableRecordBatchStream, &params)
                .await
        } else {
            tx.append_with_params(vec![batch], &params).await
        }
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("max_page_bytes must be greater than 0")
        );
        assert!(Arc::ptr_eq(&previous, &tx.active().unwrap().snapshot));
        tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        assert_eq!(
            tx.commit().await.unwrap().count_rows(None).await.unwrap(),
            2
        );
    }

    #[rstest]
    #[case::physical(false)]
    #[case::stable(true)]
    #[tokio::test]
    async fn appends_are_private_and_publish_once(#[case] stable: bool) {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                max_rows_per_file: 1,
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let version = dataset.version().version;
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
            .await
            .unwrap();
        let first = tx.scan().unwrap();
        tx.append(vec![record_batch!(("id", Int32, [4, 5])).unwrap()])
            .await
            .unwrap();
        assert_eq!(first.try_into_batch().await.unwrap().num_rows(), 4);
        assert_eq!(
            tx.scan()
                .unwrap()
                .filter("id >= 2")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            4
        );
        assert_eq!(
            Dataset::open(dir.as_str())
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            2
        );
        let committed = tx.commit().await.unwrap();
        assert_eq!(committed.version().version, version + 1);
        assert_eq!(
            Dataset::open(dir.as_str())
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            6
        );
    }

    #[rstest]
    #[case::physical(false)]
    #[case::stable(true)]
    #[tokio::test]
    async fn read_only_commit_returns_original_snapshot_after_concurrent_append(
        #[case] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let original_version = dataset.version().version;
        let mut tx = dataset.begin_transaction().await.unwrap();
        let reader = tx.scan().unwrap();
        assert_eq!(
            tx.scan().unwrap().try_into_batch().await.unwrap(),
            record_batch!(("id", Int32, [0])).unwrap()
        );
        let concurrent = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        let committed = tx.commit().await.unwrap();
        assert_eq!(committed.version().version, original_version);
        let expected = record_batch!(("id", Int32, [0])).unwrap();
        assert_eq!(committed.scan().try_into_batch().await.unwrap(), expected);
        assert_eq!(reader.try_into_batch().await.unwrap(), expected);
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(reopened.version().version, concurrent.version().version);
        assert_eq!(
            reopened.scan().try_into_batch().await.unwrap(),
            record_batch!(("id", Int32, [0, 1])).unwrap()
        );
    }

    #[rstest]
    #[case::blind(false)]
    #[case::read_dependent(true)]
    #[tokio::test]
    async fn concurrent_append_obeys_read_contract(
        #[case] read: bool,
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut first = dataset.begin_transaction().await.unwrap();
        let mut second = dataset.begin_transaction().await.unwrap();
        first
            .append(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        second
            .append(vec![record_batch!(("id", Int32, [2])).unwrap()])
            .await
            .unwrap();
        let old = if read {
            Some(second.scan().unwrap())
        } else {
            None
        };
        first.commit().await.unwrap();
        if let Some(old) = old {
            assert_eq!(old.try_into_batch().await.unwrap().num_rows(), 2);
        }
        let result = second.commit().await;
        if read {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::RetryableCommitConflict { .. }));
            assert!(
                error
                    .to_string()
                    .contains("Staged reads require base version")
            );
        } else {
            assert_eq!(
                result
                    .unwrap()
                    .scan()
                    .try_into_batch()
                    .await
                    .unwrap()
                    .num_rows(),
                3
            );
        }
        assert_eq!(
            Dataset::open(dir.as_str())
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            if read { 2 } else { 3 }
        );
        if stable {
            assert_eq!(
                Dataset::open(dir.as_str())
                    .await
                    .unwrap()
                    .scan()
                    .filter("_rowid = id")
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap()
                    .num_rows(),
                if read { 2 } else { 3 }
            );
        }
    }

    #[tokio::test]
    async fn failed_write_preserves_snapshot_and_abort_preserves_scanner() {
        let dir = TempStrDir::default();
        let batch = record_batch!(("id", Int32, [0, 1])).unwrap();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![batch.clone()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![batch.clone()]).await.unwrap();
        let source = RecordBatchStreamAdapter::new(
            batch.schema(),
            futures::stream::iter([
                Ok(batch.clone()),
                Err(DataFusionError::Execution("injected stream failure".into())),
            ]),
        );
        let error = tx
            .append_stream(Box::pin(source) as SendableRecordBatchStream)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("injected stream failure"));
        assert_eq!(
            tx.scan()
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            4
        );
        tx.append(vec![batch]).await.unwrap();
        let scanner = tx.scan().unwrap();
        tx.abort();
        tx.abort();
        let Err(error) = tx.scan() else {
            panic!("aborted transaction scanned");
        };
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("aborted"));
        assert_eq!(scanner.try_into_batch().await.unwrap().num_rows(), 6);
        assert_eq!(
            Dataset::open(dir.as_str())
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            2
        );
    }
    #[rstest]
    #[case::lost_conflict(crate::utils::test::AmbiguousFailure::LandAndConflict, true)]
    #[case::lost_io(crate::utils::test::AmbiguousFailure::LandAndError, true)]
    #[case::rejected(crate::utils::test::AmbiguousFailure::FailOutright, false)]
    #[tokio::test]
    async fn publication_uses_existing_outcome_verification(
        #[case] failure: crate::utils::test::AmbiguousFailure,
        #[case] landed: bool,
    ) {
        let dir = TempStrDir::default();
        let handler = Arc::new(crate::utils::test::AmbiguousCommitHandler::default());
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                commit_handler: Some(handler.clone()),
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        let identity = tx.id().to_owned();
        let reader = tx.scan().unwrap();
        handler.fail_next(failure);
        let result = tx.commit().await;
        assert_eq!(result.is_ok(), landed);
        if let Err(error) = result {
            assert!(
                error
                    .to_string()
                    .contains("simulated outright commit failure")
            );
        }
        assert_eq!(reader.try_into_batch().await.unwrap().num_rows(), 2);
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(
            reopened.scan().try_into_batch().await.unwrap().num_rows(),
            if landed { 2 } else { 1 }
        );
        if landed {
            assert_eq!(
                reopened
                    .read_transaction_by_version(2)
                    .await
                    .unwrap()
                    .unwrap()
                    .uuid,
                identity
            );
        }
    }

    #[tokio::test]
    async fn invalid_action_preserves_previous_snapshot() {
        let dataset = InsertBuilder::new("memory://")
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        let before = tx.active().unwrap().snapshot.clone();
        let mut written = InsertBuilder::new(before.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(vec![record_batch!(("id", Int32, [2])).unwrap()])
            .await
            .unwrap();
        let Operation::Append { fragments } = &mut written.operation else {
            panic!("expected append");
        };
        fragments[0].files[0].fields = Arc::from([i32::MAX]);
        let error = tx.stage_append(written).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("field"));
        assert!(Arc::ptr_eq(&tx.active().unwrap().snapshot, &before));
        tx.append(vec![record_batch!(("id", Int32, [3])).unwrap()])
            .await
            .unwrap();
        let committed = tx.commit().await.unwrap();
        assert_eq!(
            committed.scan().try_into_batch().await.unwrap().num_rows(),
            3
        );
        assert_eq!(before.scan().try_into_batch().await.unwrap().num_rows(), 2);
    }

    #[tokio::test]
    async fn unknown_publication_retains_readable_files() {
        let dir = TempStrDir::default();
        let handler = Arc::new(crate::utils::test::AmbiguousCommitHandler::default());
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                commit_handler: Some(handler.clone()),
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        let identity = tx.id().to_owned();
        let reader = tx.scan().unwrap();
        handler.fail_next(crate::utils::test::AmbiguousFailure::LandAndError);
        handler
            .fail_resolve
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let error = tx.commit().await.unwrap_err();
        assert!(error.is_commit_status_unknown());
        assert!(
            error
                .to_string()
                .contains("simulated ambiguous commit failure")
        );
        handler
            .fail_resolve
            .store(false, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(reader.try_into_batch().await.unwrap().num_rows(), 2);
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(
            reopened.scan().try_into_batch().await.unwrap().num_rows(),
            2
        );
        assert_eq!(
            reopened
                .read_transaction_by_version(2)
                .await
                .unwrap()
                .unwrap()
                .uuid,
            identity
        );
    }

    #[tokio::test]
    async fn existing_index_reads_staged_appends_and_old_snapshot() {
        let dir = TempStrDir::default();
        let mut dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        dataset
            .create_index(
                &["id"],
                lance_index::IndexType::BTree,
                None,
                &lance_index::scalar::ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
            .await
            .unwrap();
        let mut old = tx.scan().unwrap();
        old.filter("id >= 1").unwrap();
        tx.append(vec![record_batch!(("id", Int32, [4, 5])).unwrap()])
            .await
            .unwrap();
        assert_eq!(old.try_into_batch().await.unwrap().num_rows(), 3);
        assert_eq!(
            tx.scan()
                .unwrap()
                .filter("id >= 1")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            5
        );
        tx.commit().await.unwrap();
        assert_eq!(
            Dataset::open(dir.as_str())
                .await
                .unwrap()
                .scan()
                .filter("id >= 1")
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            5
        );
    }

    #[tokio::test]
    async fn cancelled_statement_and_active_stream_survive_abort() {
        let dataset = InsertBuilder::new("memory://")
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        let batch = record_batch!(("id", Int32, [1])).unwrap();
        tx.append(vec![batch.clone()]).await.unwrap();
        let previous = tx.active().unwrap().snapshot.clone();
        let source = RecordBatchStreamAdapter::new(batch.schema(), futures::stream::pending());
        let mut pending = Box::pin(tx.append_stream(Box::pin(source) as SendableRecordBatchStream));
        assert!(futures::poll!(&mut pending).is_pending());
        drop(pending);
        assert!(Arc::ptr_eq(&previous, &tx.active().unwrap().snapshot));
        tx.append(vec![batch]).await.unwrap();
        let stream = tx.scan().unwrap().try_into_stream().await.unwrap();
        tx.abort();
        let count = futures::TryStreamExt::try_fold(stream, 0usize, |count, batch| async move {
            Ok(count + batch.num_rows())
        })
        .await
        .unwrap();
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn active_reader_does_not_pin_aged_staged_files_against_gc() {
        mock_instant::thread_local::MockClock::set_system_time(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap(),
        );
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        let reader = tx.scan().unwrap();
        let staged_path = tx
            .active()
            .unwrap()
            .snapshot
            .manifest
            .fragments
            .last()
            .unwrap()
            .files[0]
            .path
            .clone();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(
                std::path::Path::new(dir.as_str())
                    .join("data")
                    .join(staged_path.as_str()),
            )
            .unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(
            std::time::SystemTime::now() - std::time::Duration::from_secs(8 * 86400),
        ))
        .unwrap();
        dataset
            .cleanup_old_versions(chrono::Duration::zero(), Some(false), Some(false))
            .await
            .unwrap();
        let error = reader.try_into_batch().await.unwrap_err();
        assert!(matches!(error.find_root(), Error::IO { .. }), "{error:?}");
        assert!(error.to_string().contains(staged_path.as_str()), "{error}");
        tx.abort();
        assert_eq!(
            Dataset::open(dir.as_str())
                .await
                .unwrap()
                .scan()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            1
        );
    }

    #[rstest]
    #[case::main(false)]
    #[case::detached(true)]
    #[tokio::test]
    async fn staged_snapshot_cannot_publish_after_a_racer(#[case] detached: bool) {
        let dir = TempStrDir::default();
        let base = Arc::new(
            InsertBuilder::new(dir.as_str())
                .execute(vec![record_batch!(("a", Int32, [1, 2])).unwrap()])
                .await
                .unwrap(),
        );
        let params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let prepared = InsertBuilder::new(base.clone())
            .with_params(&params)
            .execute_uncommitted(vec![record_batch!(("a", Int32, [10, 11])).unwrap()])
            .await
            .unwrap();
        let (manifest, indices) = prepared
            .build_manifest(
                Some(&base.manifest),
                vec![],
                "",
                &ManifestWriteConfig::default().to_build_config(),
            )
            .unwrap();
        let private = Arc::new(base.with_staged_manifest(manifest, indices));
        let old_reader = private.scan();
        assert_eq!(private.scan().try_into_batch().await.unwrap().num_rows(), 4);
        assert!(load_all_indices(&private).await.unwrap().is_empty());
        let second = base.with_staged_manifest(private.manifest.as_ref().clone(), vec![]);
        assert_ne!(
            private.manifest_location.e_tag,
            second.manifest_location.e_tag
        );
        InsertBuilder::new(base)
            .with_params(&params)
            .execute(vec![record_batch!(("a", Int32, [20, 21])).unwrap()])
            .await
            .unwrap();
        let prepared = InsertBuilder::new(private.clone())
            .with_params(&params)
            .execute_uncommitted(vec![record_batch!(("a", Int32, [30])).unwrap()])
            .await
            .unwrap();
        let error = CommitBuilder::new(private)
            .with_detached(detached)
            .execute(prepared)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("staged snapshot"));
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(
            reopened.scan().try_into_batch().await.unwrap().num_rows(),
            4
        );
        assert_eq!(
            reopened.count_rows(Some("a >= 20".into())).await.unwrap(),
            2
        );
        assert_eq!(old_reader.try_into_batch().await.unwrap().num_rows(), 4);
    }
}
