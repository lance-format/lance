// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Append-only staged transactions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::RecordBatch;
use lance_datafusion::utils::StreamingWriteSource;
use lance_file::version::ConcreteFileVersion;
use lance_table::format::{
    DataFile, Fragment, IndexMetadata, Manifest, RowIdMeta, is_detached_version,
};
use lance_table::io::commit::{CommitHandler, ManifestLocation};
use lance_table::rowids::read_row_ids;
use lance_table::transaction::action::{
    Action, AddDataFile, AddFragment, AddFragmentParts, ApplyState, CompositeOperation, FieldRef,
    Ref, UserStep,
};
use uuid::Uuid;

use super::transaction::{Operation, Transaction};
use super::{CommitBuilder, InsertBuilder, ManifestWriteConfig, Scanner, WriteMode, WriteParams};
use crate::datatypes::Schema;
use crate::index::loading::load_all_indices;
use crate::{Dataset, Error, Result};

/// Version carried by every uncommitted staged dataset.
///
/// `base_version + 1` is not safe. Another writer can commit that version
/// first, and a scan prepared against the uncommitted snapshot would then
/// describe the other writer's rows.
///
/// Staged snapshots use `u64::MAX` instead. It sits at the top of the detached
/// range, and detached commits draw again when they land on it. A transaction
/// or manifest copied from the snapshot keeps this version, so commit rejects it.
pub const STAGED_VERSION: u64 = u64::MAX;

/// An experimental append transaction committed as one dataset version.
///
/// Each append sees the base plus the appends before it. Appends that have not
/// been scanned can rebase over concurrent commits. [`Self::scan`] pins the
/// base, and [`Self::commit`] then requires that version to still be the
/// latest, so the row ids and addresses the scanner returned stay valid. That
/// commit gets one attempt. A commit with no staged appends returns the base
/// snapshot and does not check for a newer version. Retries reuse the files
/// already written.
///
/// A failed append leaves the previous snapshot in place. Aborting or dropping
/// the transaction deletes nothing, so an earlier scanner can finish. Nothing
/// protects the uncommitted files from cleanup. `delete_unverified` removes
/// them, and any cleanup removes a staged file older than seven days. A later
/// commit would then point at missing files. The same cleanup reclaims files
/// from failed writes. A blob object that a staged row borrows is unprotected
/// too. Cleanup that drops its last committed reference leaves the later
/// commit pointing at a missing object.
///
/// A staged commit skips auto cleanup, because an earlier scanner may still be
/// reading files that cleanup would remove. A table written only through
/// staged commits needs an explicit cleanup call.
///
/// [`Self::commit`] consumes the transaction even when it fails. An error
/// after submission can mean the commit landed. Do not resubmit it or delete
/// its files. Compare [`Self::id`] with the dataset history first.
///
/// ```
/// # use lance::{Dataset, Result};
/// # use arrow_array::RecordBatch;
/// # async fn example(dataset: &Dataset, batches: Vec<RecordBatch>) -> Result<()> {
/// let mut tx = dataset.begin_transaction().await?;
/// tx.append(batches).await?;
/// let rows = tx.scan()?.try_into_batch().await?;
/// let committed = tx.commit().await?;
/// # let _ = (rows, committed);
/// # Ok(())
/// # }
/// ```
pub struct StagedTransaction {
    base: Arc<Dataset>,
    id: String,
    state: StagedState,
}

/// Publication options for a [`StagedTransaction`].
///
/// Retry settings apply only while the transaction can rebase over concurrent
/// commits. After [`StagedTransaction::scan`] has pinned the base, publication
/// gets one attempt.
pub struct StagedCommitBuilder {
    transaction: StagedTransaction,
    commit: CommitBuilder<'static>,
    transaction_properties: Option<HashMap<String, String>>,
}

impl StagedCommitBuilder {
    /// Set how many times publication retries when it can rebase over
    /// concurrent commits.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.commit = self.commit.with_max_retries(max_retries);
        self
    }

    /// Set how long publication keeps retrying.
    pub fn with_retry_timeout(mut self, retry_timeout: Duration) -> Self {
        self.commit = self.commit.with_retry_timeout(retry_timeout);
        self
    }

    /// Bound the whole publication by `timeout`. `None` means no bound.
    ///
    /// A zero timeout is rejected when publication runs.
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.commit = self.commit.with_timeout(timeout);
        self
    }

    /// Publish through `commit_handler` instead of the handler the dataset was
    /// opened with.
    pub fn with_commit_handler(mut self, commit_handler: Arc<dyn CommitHandler>) -> Self {
        self.commit = self.commit.with_commit_handler(commit_handler);
        self
    }

    /// Record `transaction_properties` on the committed transaction.
    pub fn with_transaction_properties(
        mut self,
        transaction_properties: HashMap<String, String>,
    ) -> Self {
        self.transaction_properties = Some(transaction_properties);
        self
    }

    /// Publish every staged append as one version.
    ///
    /// See [`StagedTransaction::commit`] for a transaction with no staged
    /// writes, and for a commit whose outcome is unknown.
    pub async fn execute(self) -> Result<Dataset> {
        let StagedTransaction { base, id, state } = self.transaction;
        if state.steps.is_empty() {
            return Ok(base.as_ref().clone());
        }
        let mut commit = self.commit;
        if state.requires_original_version {
            commit = commit.with_expected_latest_version(base.manifest.version);
        }
        let mut transaction = Transaction::new(
            base.manifest.version,
            Operation::CompositeOperation(CompositeOperation::new(state.steps)),
            None,
        );
        transaction.uuid = id;
        transaction.transaction_properties = self.transaction_properties.map(Arc::new);
        commit.execute(transaction).await
    }
}

struct StagedState {
    steps: Vec<UserStep>,
    working: ApplyState,
    staged_dataset: Option<Arc<Dataset>>,
    /// Last dataset snapshot whose schema matches the working state. Starts at
    /// `base` and advances when [`StagedTransaction::staged_dataset`] builds a
    /// snapshot.
    append_base: Arc<Dataset>,
    /// The next append must write against a rebuilt snapshot because a staged
    /// 2.2 or newer data file promoted a legacy blob field.
    append_context_dirty: bool,
    next_local_fragment_id: u32,
    /// Scan already handed out row ids, or a step cannot be replayed onto a
    /// newer version.
    requires_original_version: bool,
}

/// Refuse a transaction prepared against an uncommitted staged dataset.
pub fn ensure_prepared_on_published(transaction: &Transaction) -> Result<()> {
    if transaction.read_version == STAGED_VERSION {
        return Err(Error::invalid_input(format!(
            "Transaction {} was prepared on an uncommitted staged dataset. Prepare it on a committed dataset",
            transaction.uuid
        )));
    }
    Ok(())
}

impl Dataset {
    pub(crate) fn ensure_published(&self) -> Result<()> {
        if self.manifest.version == STAGED_VERSION {
            return Err(Error::invalid_input(
                "Operation requires a committed dataset. Commit the uncommitted staged dataset's owning staged transaction first",
            ));
        }
        Ok(())
    }

    fn with_staged_manifest(&self, mut manifest: Manifest, indices: Vec<IndexMetadata>) -> Self {
        manifest.version = STAGED_VERSION;
        // Not a committed manifest. A fresh location keeps the snapshot out of
        // the committed manifest cache.
        let identity = format!("private-{}", Uuid::new_v4());
        let location = ManifestLocation {
            version: manifest.version,
            path: self.base.clone().join("_staged").join(identity.as_str()),
            size: None,
            naming_scheme: self.manifest_location.naming_scheme,
            e_tag: Some(identity.clone()),
            identity: Some(identity),
            last_modified: None,
        };
        let mut dataset = self.clone();
        dataset.set_manifest(Arc::new(manifest), location);
        dataset.staged_indices = Some(Arc::new(indices));
        dataset
    }

    /// Begin an experimental append transaction from this committed version.
    ///
    /// See [`StagedTransaction`].
    pub async fn begin_transaction(&self) -> Result<StagedTransaction> {
        self.ensure_published()?;
        if is_detached_version(self.manifest.version) {
            return Err(Error::not_supported(format!(
                "Staged transactions require a mainline dataset version, got detached version {}",
                self.manifest.version
            )));
        }
        let indices = load_all_indices(self).await?;
        let working = ApplyState::at_read_version(
            self.manifest.clone(),
            Arc::unwrap_or_clone(indices),
            &ManifestWriteConfig::default().to_build_config(),
        )?;
        let base = Arc::new(self.clone());
        Ok(StagedTransaction {
            id: Uuid::new_v4().to_string(),
            state: StagedState {
                steps: Vec::new(),
                working,
                staged_dataset: None,
                append_base: base.clone(),
                append_context_dirty: false,
                next_local_fragment_id: 0,
                requires_original_version: false,
            },
            base,
        })
    }
}

impl StagedTransaction {
    /// Transaction id used to check an ambiguous commit.
    ///
    /// Save this before calling [`Self::commit`] or [`Self::commit_with`], which
    /// consume the transaction. After publication returns an unknown outcome,
    /// compare the saved id with [`Dataset::read_transaction_by_version`]. Do
    /// not write the same files again only because the acknowledgement was lost.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Committed version this transaction started from.
    pub fn base_version(&self) -> u64 {
        self.base.manifest.version
    }

    fn staged_dataset(&mut self) -> Result<Arc<Dataset>> {
        if let Some(cached) = self.state.staged_dataset.as_ref() {
            return Ok(cached.clone());
        }
        let (manifest, indices) = self.state.working.build_manifest()?;
        let built = Arc::new(self.base.with_staged_manifest(manifest, indices));
        self.state.staged_dataset = Some(built.clone());
        self.state.append_base = built.clone();
        self.state.append_context_dirty = false;
        Ok(built)
    }

    fn append_dataset(&mut self) -> Result<Arc<Dataset>> {
        if !self.state.append_context_dirty {
            return Ok(self.state.append_base.clone());
        }
        self.staged_dataset()
    }

    /// Scan the uncommitted snapshot.
    ///
    /// Calling this method pins a non-empty commit to [`Self::base_version`],
    /// even if the returned scanner is never polled.
    pub fn scan(&mut self) -> Result<Scanner> {
        let dataset = self.staged_dataset()?;
        self.state.requires_original_version = true;
        Ok(dataset.scan())
    }

    /// Append batches.
    ///
    /// An error leaves the previous snapshot usable. Use [`Self::append_stream`]
    /// when the input does not fit in memory.
    pub async fn append(&mut self, batches: Vec<RecordBatch>) -> Result<()> {
        self.append_with_params(batches, &append_params()).await
    }

    /// Append batches with the writer's file, encoding, progress, and blob options.
    ///
    /// `mode` must be [`WriteMode::Append`]. The writer inherits the dataset's
    /// stable row id setting and ignores `enable_stable_row_ids`. `store_params`
    /// configures reads of external blob sources. The destination store and
    /// session come from the dataset.
    ///
    /// `commit_handler`, `auto_cleanup`, `transaction_properties`, `session`,
    /// `base_store_params`, and `initial_bases` must stay unset. `target_bases`
    /// routes files to committed bases. `enable_v2_manifest_paths` and
    /// `skip_auto_cleanup` must stay at their defaults. Publication is
    /// [`Self::commit`]. A rejected setting leaves the previous snapshot intact.
    pub async fn append_with_params(
        &mut self,
        batches: Vec<RecordBatch>,
        params: &WriteParams,
    ) -> Result<()> {
        self.validate_append_params(params)?;
        let target = self.append_dataset()?;
        let written = InsertBuilder::new(target)
            .with_params(params)
            .execute_uncommitted(batches)
            .await?;
        self.stage_append(written)
    }

    /// Append a stream without collecting it first.
    ///
    /// A cancelled or failed write can leave orphan files. It does not replace
    /// the last valid snapshot.
    pub async fn append_stream(&mut self, source: impl StreamingWriteSource) -> Result<()> {
        self.append_stream_with_params(source, &append_params())
            .await
    }

    /// Append a stream with the same options and restrictions as
    /// [`Self::append_with_params`]. The params are checked before the source
    /// is read.
    pub async fn append_stream_with_params(
        &mut self,
        source: impl StreamingWriteSource,
        params: &WriteParams,
    ) -> Result<()> {
        self.validate_append_params(params)?;
        let target = self.append_dataset()?;
        let written = InsertBuilder::new(target)
            .with_params(params)
            .execute_uncommitted_stream(source)
            .await?;
        self.stage_append(written)
    }

    fn validate_append_params(&self, params: &WriteParams) -> Result<()> {
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
            ("enable_v2_manifest_paths", !params.enable_v2_manifest_paths),
            ("skip_auto_cleanup", params.skip_auto_cleanup),
        ] {
            if unsupported {
                return Err(Error::invalid_input(format!(
                    "WriteParams.{field} cannot be set on one staged append. Leave it at the default"
                )));
            }
        }
        Ok(())
    }

    fn stage_append(&mut self, written: Transaction) -> Result<()> {
        let fragments = match written.operation {
            Operation::Append { fragments } => fragments,
            other => {
                return Err(Error::internal(format!(
                    "Uncommitted append {} for staged transaction {} returned {}",
                    written.uuid,
                    self.id,
                    other.name()
                )));
            }
        };
        if fragments.is_empty() {
            return Ok(());
        }
        let start = self.state.next_local_fragment_id;
        let next_local_fragment_id = local_fragment_id(start, fragments.len())?;
        let mut actions = Vec::new();
        for (offset, fragment) in fragments.into_iter().enumerate() {
            let local_id = local_fragment_id(start, offset)?;
            actions.extend(actions_for_appended_fragment(
                Ref::Local(local_id),
                fragment,
            )?);
        }
        let step = UserStep::new("append", actions);
        let pins_base_version = step_pins_base_version(&step, &self.base.manifest)?;
        let promotes_legacy_blob =
            step_promotes_legacy_blob(&self.state.append_base.manifest.schema, &step.actions);
        let next_working = self.state.working.apply_step_checked(&step)?;
        self.state.working = next_working;
        self.state.steps.push(step);
        self.state.staged_dataset = None;
        self.state.append_context_dirty |= promotes_legacy_blob;
        self.state.next_local_fragment_id = next_local_fragment_id;
        self.state.requires_original_version |= pins_base_version;
        Ok(())
    }

    /// Set retries, timeout, properties, or a commit handler, then publish.
    pub fn commit_with(self) -> StagedCommitBuilder {
        // An earlier scanner may still need files that commit-triggered cleanup
        // would remove.
        let commit = CommitBuilder::new(self.base.clone())
            .with_experimental_composite_operations(true)
            .with_skip_auto_cleanup(true);
        StagedCommitBuilder {
            transaction: self,
            commit,
            transaction_properties: None,
        }
    }

    /// Publish every append as one version.
    ///
    /// The transaction is consumed on failure. No staged appends returns the
    /// original snapshot, including after a scan or a concurrent commit.
    pub async fn commit(self) -> Result<Dataset> {
        self.commit_with().execute().await
    }

    /// Abandon this transaction without publishing.
    ///
    /// No objects are deleted. A scanner that already started can finish, and
    /// unpublished objects stay eligible for orphan cleanup.
    pub fn abort(self) {}
}

fn append_params() -> WriteParams {
    WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    }
}

fn local_fragment_id(start: u32, added_fragments: usize) -> Result<u32> {
    let added = u32::try_from(added_fragments).map_err(|_| {
        Error::invalid_input(format!(
            "Staged append adds {added_fragments} fragments, which does not fit in a local fragment id"
        ))
    })?;
    start.checked_add(added).ok_or_else(|| {
        Error::invalid_input(format!(
            "Staged append already holds {start} local fragment ids and this step adds {added_fragments}"
        ))
    })
}

/// Lower one fragment from an uncommitted append into actions.
///
/// The fragment is taken apart field by field. A new field fails to compile
/// until it is carried onto [`AddFragment`] or refused here.
fn actions_for_appended_fragment(id: Ref, fragment: Fragment) -> Result<Vec<Action>> {
    let Fragment {
        id: _,
        files,
        overlays,
        deletion_file,
        row_id_meta,
        physical_rows,
        last_updated_at_version_meta,
        created_at_version_meta,
        clustering_id,
        metadata,
    } = fragment;
    let first_path = files.first().map_or("<none>", |file| file.path.as_str());
    if !overlays.is_empty() {
        return Err(unstageable_fragment("overlays", first_path));
    }
    if deletion_file.is_some() {
        return Err(unstageable_fragment("deletion_file", first_path));
    }
    if clustering_id.is_some() {
        return Err(unstageable_fragment("clustering_id", first_path));
    }
    if !metadata.is_empty() {
        return Err(unstageable_fragment("metadata", first_path));
    }
    let physical_rows = physical_rows.ok_or_else(|| {
        Error::invalid_input(format!(
            "Fragment written for append must record physical rows. Its first data file is {first_path}"
        ))
    })?;
    let physical_rows = u64::try_from(physical_rows).map_err(|_| {
        Error::invalid_input(format!(
            "Fragment written for append has {physical_rows} physical rows, which does not fit in u64. Its first data file is {first_path}"
        ))
    })?;
    let mut actions = vec![Action::AddFragment(AddFragment::new_unchecked(
        AddFragmentParts {
            id,
            physical_rows,
            row_id_meta,
            last_updated_at_version_meta,
            created_at_version_meta,
            data_change: true,
        },
    ))];
    for file in files {
        actions.push(Action::AddDataFile(data_file_action(id, file)?));
    }
    Ok(actions)
}

fn unstageable_fragment(part: &str, first_path: &str) -> Error {
    Error::not_supported(format!(
        "Staged append cannot stage a new fragment that carries {part}. Its first data file is {first_path}"
    ))
}

fn data_file_action(fragment: Ref, file: DataFile) -> Result<AddDataFile> {
    let field_ids = file
        .fields
        .iter()
        .copied()
        .map(FieldRef::Committed)
        .collect();
    let base = file
        .base_id
        .map(|base_id| Ref::Committed(u64::from(base_id)));
    let file = DataFile {
        fields: Arc::<[i32]>::from([]),
        base_id: None,
        ..file
    };
    AddDataFile::try_new_unchecked(fragment, file, field_ids, base, true)
}

fn step_pins_base_version(step: &UserStep, manifest: &Manifest) -> Result<bool> {
    for action in &step.actions {
        let Action::AddFragment(add_fragment) = action else {
            continue;
        };
        if preassigned_row_ids_pin_base(manifest, add_fragment.parts().row_id_meta.as_ref())? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// A spilled row-id column is not readable here, so a rebase could mint the
/// same ids again. Inline ids at or past the counter are rejected by apply.
/// Inline ids below the counter were already reserved and can move forward.
fn preassigned_row_ids_pin_base(
    manifest: &Manifest,
    row_id_meta: Option<&RowIdMeta>,
) -> Result<bool> {
    if !manifest.uses_stable_row_ids() {
        return Ok(false);
    }
    match row_id_meta {
        None => Ok(false),
        Some(RowIdMeta::Column) => Ok(true),
        Some(RowIdMeta::Inline(encoded)) => {
            let sequence = read_row_ids(encoded.bytes())?;
            Ok(sequence
                .row_id_range()
                .is_some_and(|range| *range.end() >= manifest.next_row_id))
        }
    }
}

/// A 2.2 or newer file promotes an unpromoted legacy blob field when the
/// manifest is assembled, so the next append must write against the rebuilt
/// schema.
fn step_promotes_legacy_blob(schema: &Schema, actions: &[Action]) -> bool {
    if !schema
        .fields_pre_order()
        .any(|field| field.is_blob() && !field.is_blob_v2())
    {
        return false;
    }
    actions.iter().any(|action| {
        let Action::AddDataFile(add_file) = action else {
            return false;
        };
        if !add_file
            .file()
            .file_version()
            .is_ok_and(file_can_promote_legacy_blob)
        {
            return false;
        }
        add_file
            .field_ids()
            .iter()
            .any(|field_ref| match *field_ref {
                FieldRef::Committed(field_id) => schema
                    .field_by_id(field_id)
                    .is_some_and(|field| field.is_blob() && !field.is_blob_v2()),
                FieldRef::Local(_) => false,
            })
    })
}

/// 2.2 is the first file version that can promote a legacy blob field when the
/// manifest is assembled.
fn file_can_promote_legacy_blob(version: ConcreteFileVersion) -> bool {
    match version {
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => false,
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::index::DatasetIndexRemapperOptions;
    use crate::dataset::optimize::{
        CompactionOptions, commit_compaction, compact_files, plan_compaction,
    };
    use crate::dataset::write::WriteProgressFn;
    use crate::dataset::{DeleteBuilder, MergeInsertBuilder, WhenMatched, WhenNotMatched};
    use crate::index::DatasetIndexExt;
    use crate::io::exec::filtered_read::FilteredReadExec;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, record_batch};
    use datafusion::error::DataFusionError;
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_table::format::RowIdMeta;
    use lance_table::rowids::{RowIdSequence, write_row_ids};
    use rstest::rstest;

    fn id_batch(ids: impl IntoIterator<Item = i32>) -> RecordBatch {
        let ids = Arc::new(Int32Array::from_iter_values(ids)) as ArrayRef;
        RecordBatch::try_from_iter([("id", ids)]).unwrap()
    }

    async fn ids(dataset: &Dataset) -> Vec<i32> {
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
        ids.sort_unstable();
        ids
    }

    fn fragment_ids(dataset: &Dataset) -> Vec<u64> {
        dataset
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect()
    }

    async fn table_with_staged_append(uri: &str) -> (Dataset, StagedTransaction) {
        let dataset = InsertBuilder::new(uri)
            .with_params(&WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            })
            .execute(vec![id_batch(0..4)])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![id_batch([10, 11])]).await.unwrap();
        (dataset, tx)
    }

    async fn snapshot_from_scan_plan(tx: &mut StagedTransaction) -> Arc<Dataset> {
        let mut plans = vec![tx.scan().unwrap().create_plan().await.unwrap()];
        loop {
            let plan = plans.pop().expect("scan must contain a FilteredReadExec");
            if let Some(read) = plan.downcast_ref::<FilteredReadExec>() {
                return read.dataset().clone();
            }
            plans.extend(plan.children().into_iter().cloned());
        }
    }

    fn merge_small_fragments() -> CompactionOptions {
        CompactionOptions {
            target_rows_per_fragment: 100,
            ..Default::default()
        }
    }

    #[rstest]
    #[case::batch(false)]
    #[case::stream(true)]
    #[tokio::test]
    async fn test_append_params_preserve_layout_and_format(
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
        let snapshot = tx.staged_dataset().unwrap();
        assert_eq!(
            snapshot
                .manifest
                .fragments
                .iter()
                .map(|fragment| fragment.physical_rows.unwrap())
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
            .with_params(&append_params())
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
    #[case::manifest_paths("enable_v2_manifest_paths")]
    #[case::skip_cleanup("skip_auto_cleanup")]
    #[tokio::test]
    async fn test_unsupported_append_params_do_not_write(
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
        let previous = tx.staged_dataset().unwrap();
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
        assert!(Arc::ptr_eq(&previous, &tx.staged_dataset().unwrap()));
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
    async fn test_append_params_use_writer_validation(#[case] streaming: bool) {
        let dataset = InsertBuilder::new("memory://")
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        let previous = tx.staged_dataset().unwrap();
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
        assert!(Arc::ptr_eq(&previous, &tx.staged_dataset().unwrap()));
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
    async fn test_appends_are_private_until_one_commit(
        #[case] stable: bool,
        #[values(2, 10, 50)] append_count: i32,
    ) {
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
        for statement in 2..=append_count {
            tx.append(vec![
                record_batch!(("id", Int32, [2 * statement, 2 * statement + 1])).unwrap(),
            ])
            .await
            .unwrap();
        }
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
            2 * append_count as usize
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
            2 + 2 * append_count as usize
        );
    }

    #[rstest]
    #[case::physical(false)]
    #[case::stable(true)]
    #[tokio::test]
    async fn test_read_only_commit_returns_the_original_snapshot_after_a_concurrent_append(
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
            .with_params(&append_params())
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
    async fn test_concurrent_append_obeys_the_read_contract(
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
        let staged = second.staged_dataset().unwrap();
        assert_eq!(fragment_ids(&staged), [0, 1]);
        if stable {
            assert_eq!(
                staged
                    .scan()
                    .filter("_rowid = 1")
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap(),
                record_batch!(("id", Int32, [2])).unwrap()
            );
        }
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
        } else {
            let committed = result.unwrap();
            assert_eq!(fragment_ids(&committed), [0, 1, 2]);
            assert_eq!(
                committed
                    .scan()
                    .filter(&format!("_rowaddr >= {}", 2u64 << 32))
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap(),
                record_batch!(("id", Int32, [2])).unwrap()
            );
            assert_eq!(
                committed.scan().try_into_batch().await.unwrap().num_rows(),
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
    async fn test_failed_write_preserves_the_snapshot_and_abort_preserves_the_scanner() {
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
    async fn test_publication_uses_the_existing_outcome_check(
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

    #[rstest]
    #[case::bad_field(
        |fragment: &mut Fragment| fragment.files[0].fields = Arc::from([i32::MAX]),
        "names field"
    )]
    #[case::missing_file(
        |fragment: &mut Fragment| fragment.files.clear(),
        "does not contain non-nullable field"
    )]
    #[tokio::test]
    async fn test_rejected_append_keeps_the_previous_snapshot(
        #[case] corrupt: fn(&mut Fragment),
        #[case] rejection: &str,
    ) {
        let dataset = InsertBuilder::new("memory://")
            .execute(vec![id_batch([0])])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![id_batch([1])]).await.unwrap();
        let before = tx.staged_dataset().unwrap();
        let mut written = InsertBuilder::new(before.clone())
            .with_params(&append_params())
            .execute_uncommitted(vec![id_batch([2])])
            .await
            .unwrap();
        let Operation::Append { fragments } = &mut written.operation else {
            panic!("expected append");
        };
        corrupt(&mut fragments[0]);
        let error = tx.stage_append(written).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(rejection), "{error}");
        assert!(Arc::ptr_eq(&tx.staged_dataset().unwrap(), &before));
        assert_eq!(tx.state.steps.len(), 1);
        tx.append(vec![id_batch([3])]).await.unwrap();
        let after = tx.staged_dataset().unwrap();
        assert_eq!(fragment_ids(&after), [0, 1, 2]);
        let committed = tx.commit().await.unwrap();
        assert_eq!(ids(&committed).await, [0, 1, 3]);
        assert_eq!(ids(&before).await, [0, 1]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_two_appends_allocate_distinct_local_fragment_ids_and_commit_together(
        #[values(false, true)] stable: bool,
    ) {
        let dataset = InsertBuilder::new("memory://")
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
        tx.append(vec![record_batch!(("id", Int32, [2])).unwrap()])
            .await
            .unwrap();
        assert_eq!(tx.state.steps.len(), 2);
        assert_eq!(tx.state.next_local_fragment_id, 2);
        let staged = tx.staged_dataset().unwrap();
        assert_eq!(fragment_ids(&staged), [0, 1, 2]);
        assert_eq!(ids(&staged).await, [0, 1, 2]);
        let committed = tx.commit().await.unwrap();
        assert_eq!(fragment_ids(&committed), [0, 1, 2]);
        assert_eq!(ids(&committed).await, [0, 1, 2]);
    }

    #[tokio::test]
    async fn test_unknown_publication_keeps_the_files_readable() {
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
    async fn test_an_existing_index_reads_staged_appends_and_the_old_snapshot() {
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
    async fn test_a_cancelled_append_and_an_open_scan_survive_abort() {
        let dataset = InsertBuilder::new("memory://")
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        let batch = record_batch!(("id", Int32, [1])).unwrap();
        tx.append(vec![batch.clone()]).await.unwrap();
        let previous = tx.staged_dataset().unwrap();
        let source = RecordBatchStreamAdapter::new(batch.schema(), futures::stream::pending());
        let mut pending = Box::pin(tx.append_stream(Box::pin(source) as SendableRecordBatchStream));
        assert!(futures::poll!(&mut pending).is_pending());
        drop(pending);
        assert!(Arc::ptr_eq(&previous, &tx.staged_dataset().unwrap()));
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
    async fn test_an_open_reader_does_not_keep_aged_staged_files() {
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
            .staged_dataset()
            .unwrap()
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
        let committed = tx.commit().await.unwrap();
        let error = committed.scan().try_into_batch().await.unwrap_err();
        assert!(matches!(error.find_root(), Error::IO { .. }), "{error:?}");
        assert!(error.to_string().contains(staged_path.as_str()), "{error}");
    }

    #[derive(Debug, Clone, Copy)]
    enum Statement {
        Append,
        DetachedCommit,
        Delete,
        MergeInsert,
        Compaction,
        CreateIndex,
    }

    #[rstest]
    #[case::append(Statement::Append)]
    #[case::detached_commit(Statement::DetachedCommit)]
    #[case::delete(Statement::Delete)]
    #[case::merge_insert(Statement::MergeInsert)]
    #[case::compaction(Statement::Compaction)]
    #[case::create_index(Statement::CreateIndex)]
    #[tokio::test]
    async fn test_a_staged_snapshot_cannot_publish_a_statement(#[case] statement: Statement) {
        let dir = TempStrDir::default();
        let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
        let staged = snapshot_from_scan_plan(&mut tx).await;
        let mut handle = staged.as_ref().clone();

        let outcome = match statement {
            Statement::Append => InsertBuilder::new(staged)
                .with_params(&append_params())
                .execute(vec![id_batch([60])])
                .await
                .map(drop),
            Statement::DetachedCommit => {
                let on_committed = InsertBuilder::new(Arc::new(dataset.clone()))
                    .with_params(&append_params())
                    .execute_uncommitted(vec![id_batch([60])])
                    .await
                    .unwrap();
                CommitBuilder::new(staged)
                    .with_detached(true)
                    .execute(on_committed)
                    .await
                    .map(drop)
            }
            Statement::Delete => handle.delete("id = 10").await.map(drop),
            Statement::MergeInsert => MergeInsertBuilder::try_new(staged, vec!["id".into()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .try_build()
                .unwrap()
                .execute_batches(vec![id_batch([10, 60])])
                .await
                .map(drop),
            Statement::Compaction => compact_files(&mut handle, merge_small_fragments(), None)
                .await
                .map(drop),
            Statement::CreateIndex => handle
                .create_index(
                    &["id"],
                    lance_index::IndexType::BTree,
                    None,
                    &lance_index::scalar::ScalarIndexParams::default(),
                    false,
                )
                .await
                .map(drop),
        };
        let error = outcome.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("uncommitted staged dataset"),
            "{error}"
        );

        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(reopened.version_id(), dataset.version_id());
        assert_eq!(ids(&reopened).await, vec![0, 1, 2, 3]);
        assert!(reopened.load_indices().await.unwrap().is_empty());
        let committed = tx.commit().await.unwrap();
        assert_eq!(ids(&committed).await, vec![0, 1, 2, 3, 10, 11]);
    }

    #[derive(Debug, Clone, Copy)]
    enum Prepared {
        Append,
        AppendInBatch,
        Delete,
        DetachedDelete,
        MergeInsert,
        Compaction,
        IndexSegment,
    }

    #[rstest]
    #[case::append(Prepared::Append)]
    #[case::append_in_batch(Prepared::AppendInBatch)]
    #[case::delete(Prepared::Delete)]
    #[case::detached_delete(Prepared::DetachedDelete)]
    #[case::merge_insert(Prepared::MergeInsert)]
    #[case::compaction(Prepared::Compaction)]
    #[case::index_segment(Prepared::IndexSegment)]
    #[tokio::test]
    async fn test_work_prepared_on_a_staged_snapshot_cannot_be_committed(
        #[case] prepared: Prepared,
    ) {
        let dir = TempStrDir::default();
        let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
        let staged = snapshot_from_scan_plan(&mut tx).await;
        let racer = Arc::new(
            InsertBuilder::new(Arc::new(dataset))
                .with_params(&append_params())
                .execute(vec![id_batch([50])])
                .await
                .unwrap(),
        );
        let mut committed_handle = racer.as_ref().clone();

        let outcome = match prepared {
            Prepared::Append | Prepared::AppendInBatch => {
                let on_staged = InsertBuilder::new(staged)
                    .with_params(&append_params())
                    .execute_uncommitted(vec![id_batch([60])])
                    .await
                    .unwrap();
                let builder = CommitBuilder::new(racer.clone());
                if matches!(prepared, Prepared::Append) {
                    builder.execute(on_staged).await.map(drop)
                } else {
                    let on_committed = InsertBuilder::new(racer.clone())
                        .with_params(&append_params())
                        .execute_uncommitted(vec![id_batch([61])])
                        .await
                        .unwrap();
                    builder
                        .execute_batch(vec![on_committed, on_staged])
                        .await
                        .map(drop)
                }
            }
            Prepared::Delete | Prepared::DetachedDelete => {
                let delete = DeleteBuilder::new(staged, "id = 10")
                    .execute_uncommitted()
                    .await
                    .unwrap();
                CommitBuilder::new(racer.clone())
                    .with_detached(matches!(prepared, Prepared::DetachedDelete))
                    .execute(delete.transaction)
                    .await
                    .map(drop)
            }
            Prepared::MergeInsert => {
                let upsert = MergeInsertBuilder::try_new(staged, vec!["id".into()])
                    .unwrap()
                    .when_matched(WhenMatched::UpdateAll)
                    .when_not_matched(WhenNotMatched::InsertAll)
                    .try_build()
                    .unwrap()
                    .execute_uncommitted_batches(vec![id_batch([10, 60])])
                    .await
                    .unwrap();
                CommitBuilder::new(racer.clone())
                    .execute(upsert.transaction)
                    .await
                    .map(drop)
            }
            Prepared::Compaction => {
                let options = merge_small_fragments();
                let plan = plan_compaction(&staged, &options).await.unwrap();
                let mut rewrites = Vec::new();
                for task in plan.compaction_tasks() {
                    rewrites.push(task.execute(&staged).await.unwrap());
                }
                assert!(!rewrites.is_empty());
                commit_compaction(
                    &mut committed_handle,
                    rewrites,
                    Arc::new(DatasetIndexRemapperOptions::default()),
                    &options,
                )
                .await
                .map(drop)
            }
            Prepared::IndexSegment => {
                let mut handle = staged.as_ref().clone();
                let segment = handle
                    .create_index_builder(
                        &["id"],
                        lance_index::IndexType::BTree,
                        &lance_index::scalar::ScalarIndexParams::default(),
                    )
                    .execute_uncommitted()
                    .await
                    .unwrap();
                committed_handle
                    .commit_existing_index_segments("id_idx", "id", vec![segment])
                    .await
            }
        };
        let refusal = match prepared {
            Prepared::Compaction => format!("d{STAGED_VERSION}.manifest"),
            Prepared::IndexSegment => format!("future dataset version {STAGED_VERSION}"),
            Prepared::Append
            | Prepared::AppendInBatch
            | Prepared::Delete
            | Prepared::DetachedDelete
            | Prepared::MergeInsert => "prepared on an uncommitted staged dataset".to_owned(),
        };
        let error = outcome.unwrap_err();
        assert!(error.to_string().contains(&refusal), "{error}");

        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(reopened.version_id(), racer.version_id());
        assert_eq!(ids(&reopened).await, vec![0, 1, 2, 3, 50]);
        assert!(reopened.load_indices().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_a_handle_rebuilt_from_a_staged_manifest_stays_staged() {
        use prost::Message;

        let dir = TempStrDir::default();
        let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
        let staged = snapshot_from_scan_plan(&mut tx).await;
        let serialized = lance_table::format::pb::Manifest::from(staged.manifest()).encode_to_vec();
        let rebuilt = Arc::new(
            crate::dataset::builder::DatasetBuilder::from_uri(dir.as_str())
                .with_serialized_manifest(&serialized)
                .unwrap()
                .load()
                .await
                .unwrap(),
        );
        let policy = crate::dataset::cleanup::CleanupPolicyBuilder::default()
            .before_timestamp(crate::utils::temporal::utc_now() + chrono::TimeDelta::days(1))
            .build();
        let prepared_on_rebuilt = InsertBuilder::new(rebuilt.clone())
            .with_params(&append_params())
            .execute_uncommitted(vec![id_batch([61])])
            .await
            .unwrap();
        for error in [
            rebuilt.cleanup_with_policy(policy).await.unwrap_err(),
            InsertBuilder::new(rebuilt.clone())
                .with_params(&append_params())
                .execute(vec![id_batch([60])])
                .await
                .unwrap_err(),
            CommitBuilder::new(Arc::new(dataset.clone()))
                .execute(prepared_on_rebuilt)
                .await
                .unwrap_err(),
        ] {
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains("uncommitted staged dataset"),
                "{error}"
            );
        }

        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(reopened.version_id(), dataset.version_id());
        assert_eq!(ids(&reopened).await, vec![0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn test_checkout_latest_replaces_staged_index_metadata() {
        let dir = TempStrDir::default();
        let mut dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
            .await
            .unwrap();
        let old_reader = tx.scan().unwrap();
        let mut refreshed = tx.staged_dataset().unwrap().as_ref().clone();
        let index = dataset
            .create_index(
                &["id"],
                lance_index::IndexType::BTree,
                None,
                &lance_index::scalar::ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        refreshed.checkout_latest().await.unwrap();
        let indices = refreshed.load_indices().await.unwrap();
        assert_eq!(
            indices.iter().map(|index| index.uuid).collect::<Vec<_>>(),
            vec![index.uuid]
        );
        refreshed.ensure_published().unwrap();
        assert_eq!(refreshed.count_rows(None).await.unwrap(), 2);
        assert_eq!(old_reader.try_into_batch().await.unwrap().num_rows(), 4);
        let error = tx.staged_dataset().unwrap().ensure_published().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("staged"));
    }

    #[tokio::test]
    async fn test_a_staged_snapshot_rejects_history_questions() {
        let dir = TempStrDir::default();
        let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
        let base_version = dataset.version_id();
        let reader = tx.scan().unwrap();
        let staged = snapshot_from_scan_plan(&mut tx).await;
        let committed = InsertBuilder::new(Arc::new(dataset))
            .with_params(&append_params())
            .execute(vec![id_batch([50])])
            .await
            .unwrap();

        let now = crate::utils::temporal::utc_now();
        let deltas = [
            staged.delta().compared_against_version(base_version),
            staged
                .delta()
                .with_begin_version(base_version)
                .with_end_version(committed.version_id()),
            staged.delta().with_begin_date(now).with_end_date(now),
            staged.delta().with_begin_date(now),
        ];
        let mut errors = vec![
            staged.read_transaction().await.unwrap_err(),
            staged.is_stale().await.unwrap_err(),
            staged.has_successor_version().await.unwrap_err(),
        ];
        errors.extend(deltas.map(|delta| delta.build().err().unwrap()));
        for error in errors {
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains("uncommitted staged dataset"),
                "{error}"
            );
        }

        let rows = reader.try_into_batch().await.unwrap();
        assert_eq!(
            rows["id"].as_primitive::<Int32Type>().values(),
            &[0, 1, 2, 3, 10, 11]
        );
    }

    #[tokio::test]
    async fn test_cleanup_rejects_a_staged_snapshot() {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [2])).unwrap()])
            .await
            .unwrap();
        let staged = tx.staged_dataset().unwrap();
        let policy = crate::dataset::cleanup::CleanupPolicyBuilder::default()
            .before_timestamp(crate::utils::temporal::utc_now() + chrono::TimeDelta::days(1))
            .build();
        let operation = staged.cleanup(policy.clone());
        for error in [
            operation.explain().await.unwrap_err(),
            operation.execute().await.unwrap_err(),
            staged.cleanup_with_policy(policy).await.unwrap_err(),
            staged
                .cleanup_old_versions(chrono::TimeDelta::zero(), None, None)
                .await
                .unwrap_err(),
        ] {
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains("uncommitted staged dataset"),
                "{error}"
            );
        }

        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(reopened.version_id(), dataset.version_id());
        let committed = tx.commit().await.unwrap();
        assert_eq!(
            committed.scan().try_into_batch().await.unwrap(),
            record_batch!(("id", Int32, [0, 1, 2])).unwrap()
        );
    }

    #[tokio::test]
    async fn test_blind_appends_build_no_snapshot_until_a_read() {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0])).unwrap()])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        assert!(tx.state.staged_dataset.is_none());

        tx.append(vec![record_batch!(("id", Int32, [1])).unwrap()])
            .await
            .unwrap();
        tx.append(vec![record_batch!(("id", Int32, [2])).unwrap()])
            .await
            .unwrap();
        assert!(tx.state.staged_dataset.is_none());

        let _ = tx.scan().unwrap();
        let first_snapshot = tx.state.staged_dataset.clone().unwrap();
        let _ = tx.scan().unwrap();
        let second_snapshot = tx.state.staged_dataset.clone().unwrap();
        assert!(Arc::ptr_eq(&first_snapshot, &second_snapshot));

        tx.append(vec![record_batch!(("id", Int32, [3])).unwrap()])
            .await
            .unwrap();
        assert!(tx.state.staged_dataset.is_none());

        let committed = tx.commit().await.unwrap();
        committed.validate().await.unwrap();
        assert_eq!(ids(&committed).await, [0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn test_blind_append_preserves_base_indices_across_concurrent_append() {
        let dir = TempStrDir::default();
        let mut dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![
                record_batch!(("id", Int32, [0, 1]), ("val", Int32, [10, 20])).unwrap(),
            ])
            .await
            .unwrap();
        dataset
            .create_index(
                &["val"],
                lance_index::IndexType::BTree,
                Some("val_idx".into()),
                &lance_index::scalar::ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let base_version = dataset.version_id();

        let mut tx = dataset.begin_transaction().await.unwrap();
        assert_eq!(tx.base_version(), base_version);
        assert!(tx.state.staged_dataset.is_none());
        assert!(!tx.state.requires_original_version);

        tx.append(vec![
            record_batch!(("id", Int32, [2, 3]), ("val", Int32, [30, 40])).unwrap(),
        ])
        .await
        .unwrap();
        assert_eq!(tx.base_version(), base_version);
        assert!(tx.state.staged_dataset.is_none());
        assert!(!tx.state.requires_original_version);

        let mut racer = dataset.begin_transaction().await.unwrap();
        racer
            .append(vec![
                record_batch!(("id", Int32, [4]), ("val", Int32, [50])).unwrap(),
            ])
            .await
            .unwrap();
        let after_racer = racer.commit().await.unwrap();

        let committed = tx.commit().await.unwrap();
        committed.validate().await.unwrap();
        assert_eq!(committed.version_id(), after_racer.version_id() + 1);
        assert_eq!(ids(&committed).await, [0, 1, 2, 3, 4]);
        let index_names = committed
            .load_indices()
            .await
            .unwrap()
            .iter()
            .map(|index| index.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(index_names, ["val_idx"]);
    }

    #[tokio::test]
    async fn test_merge_insert_prepared_on_a_staged_snapshot_is_refused() {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            })
            .execute(vec![
                record_batch!(("id", Int32, [1, 2]), ("b", Int32, [10, 20])).unwrap(),
            ])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![
            record_batch!(("id", Int32, [3]), ("b", Int32, [30])).unwrap(),
        ])
        .await
        .unwrap();
        let leaked = tx.staged_dataset().unwrap();
        let patch = record_batch!(("id", Int32, [1]), ("b", Int32, [99])).unwrap();
        let mut builder = MergeInsertBuilder::try_new(leaked, vec!["id".into()]).unwrap();
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing);
        let uncommitted = builder
            .try_build()
            .unwrap()
            .execute_uncommitted_batches(vec![patch])
            .await
            .unwrap();
        let error = CommitBuilder::new(Arc::new(dataset))
            .execute(uncommitted.transaction)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("prepared on an uncommitted staged dataset")
        );
    }

    #[derive(Debug, Clone, Copy)]
    enum Uncommitted {
        StagedSnapshot,
        DetachedVersion,
    }

    #[rstest]
    #[case::staged_snapshot(Uncommitted::StagedSnapshot)]
    #[case::detached_version(Uncommitted::DetachedVersion)]
    #[tokio::test]
    async fn test_begin_transaction_requires_a_committed_mainline_snapshot(
        #[case] uncommitted: Uncommitted,
    ) {
        let dir = TempStrDir::default();
        let (dataset, mut tx) = table_with_staged_append(dir.as_str()).await;
        let error = match uncommitted {
            Uncommitted::StagedSnapshot => snapshot_from_scan_plan(&mut tx)
                .await
                .begin_transaction()
                .await
                .err()
                .unwrap(),
            Uncommitted::DetachedVersion => {
                let base = Arc::new(dataset);
                let written = InsertBuilder::new(base.clone())
                    .with_params(&append_params())
                    .execute_uncommitted(vec![id_batch([60])])
                    .await
                    .unwrap();
                CommitBuilder::new(base)
                    .with_detached(true)
                    .execute(written)
                    .await
                    .unwrap()
                    .begin_transaction()
                    .await
                    .err()
                    .unwrap()
            }
        };
        match uncommitted {
            Uncommitted::StagedSnapshot => {
                assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
                assert!(
                    error.to_string().contains("uncommitted staged dataset"),
                    "{error}"
                );
            }
            Uncommitted::DetachedVersion => {
                assert!(matches!(error, Error::NotSupported { .. }), "{error}");
                assert!(
                    error.to_string().contains("mainline dataset version"),
                    "{error}"
                );
            }
        }
    }

    #[rstest]
    #[case::deletion_file(
        |fragment: &mut Fragment| {
            fragment.deletion_file = Some(lance_table::format::DeletionFile {
                read_version: 1,
                id: 7,
                file_type: lance_table::format::DeletionFileType::Array,
                num_deleted_rows: Some(1),
                base_id: None,
                inline: None,
            })
        },
        "deletion_file"
    )]
    #[case::overlays(
        |fragment: &mut Fragment| {
            let mut overlay_file = fragment.files[0].clone();
            overlay_file.path = "overlay.lance".to_string();
            fragment.overlays.push(lance_table::format::overlay::DataOverlayFile {
                data_file: overlay_file,
                coverage: lance_table::format::overlay::OverlayCoverage::Shared(Arc::new(
                    roaring::RoaringBitmap::from_iter([0_u32]),
                )),
                committed_version: 1,
            })
        },
        "overlays"
    )]
    #[case::clustering_id(
        |fragment: &mut Fragment| fragment.clustering_id = Some(1),
        "clustering_id"
    )]
    #[case::metadata(
        |fragment: &mut Fragment| {
            fragment.metadata.insert("shard".into(), "42".into());
        },
        "metadata"
    )]
    #[tokio::test]
    async fn test_append_refuses_a_fragment_its_actions_cannot_express(
        #[case] carry: fn(&mut Fragment),
        #[case] carried_part: &str,
    ) {
        let dataset = InsertBuilder::new("memory://")
            .execute(vec![id_batch([0])])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![id_batch([1])]).await.unwrap();
        let before = tx.staged_dataset().unwrap();
        let mut written = InsertBuilder::new(before.clone())
            .with_params(&append_params())
            .execute_uncommitted(vec![id_batch([2])])
            .await
            .unwrap();
        let Operation::Append { fragments } = &mut written.operation else {
            panic!("expected append");
        };
        carry(&mut fragments[0]);
        let error = tx.stage_append(written).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains(carried_part), "{error}");
        assert!(Arc::ptr_eq(&tx.staged_dataset().unwrap(), &before));
        assert_eq!(tx.state.steps.len(), 1);
        tx.append(vec![id_batch([3])]).await.unwrap();
        let committed = tx.commit().await.unwrap();
        assert_eq!(ids(&committed).await, [0, 1, 3]);
    }

    #[tokio::test]
    async fn test_unreserved_row_ids_leave_the_previous_snapshot() {
        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            })
            .execute(vec![id_batch([0])])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![id_batch([1])]).await.unwrap();
        let before = tx.staged_dataset().unwrap();
        let mut written = InsertBuilder::new(before.clone())
            .with_params(&append_params())
            .execute_uncommitted(vec![id_batch([2])])
            .await
            .unwrap();
        let Operation::Append { fragments } = &mut written.operation else {
            panic!("expected append");
        };
        let assigned = dataset.manifest.next_row_id;
        let sequence = RowIdSequence::from(assigned..assigned + 1);
        fragments[0].row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&sequence).into()));
        let error = tx.stage_append(written).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("reserved"), "{error}");
        assert!(Arc::ptr_eq(&tx.staged_dataset().unwrap(), &before));
        assert!(!tx.state.requires_original_version);
        tx.append(vec![id_batch([3])]).await.unwrap();
        let committed = tx.commit().await.unwrap();
        assert_eq!(ids(&committed).await, [0, 1, 3]);
    }

    #[tokio::test]
    async fn test_append_routes_files_to_a_committed_base() {
        let dir = lance_core::utils::tempfile::TempDir::default();
        let registered_base = dir.std_path().join("registered");
        std::fs::create_dir_all(&registered_base).unwrap();
        let primary_uri = format!("file://{}", dir.std_path().join("primary").display());
        let dataset = InsertBuilder::new(primary_uri.as_str())
            .with_params(&WriteParams {
                initial_bases: Some(vec![lance_table::format::BasePath {
                    id: 1,
                    name: Some("registered".to_string()),
                    is_dataset_root: true,
                    path: format!("file://{}", registered_base.display()),
                }]),
                ..Default::default()
            })
            .execute(vec![id_batch([0])])
            .await
            .unwrap();
        assert_eq!(dataset.manifest.fragments[0].files[0].base_id, None);

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append_with_params(
            vec![id_batch([1, 2])],
            &WriteParams {
                target_bases: Some(vec![1]),
                ..append_params()
            },
        )
        .await
        .unwrap();
        let staged = tx.staged_dataset().unwrap();
        assert_eq!(staged.manifest.fragments[1].files[0].base_id, Some(1));
        assert_eq!(ids(&staged).await, [0, 1, 2]);

        tx.commit().await.unwrap();
        let reopened = Dataset::open(primary_uri.as_str()).await.unwrap();
        assert_eq!(reopened.manifest.fragments[1].files[0].base_id, Some(1));
        assert_eq!(ids(&reopened).await, [0, 1, 2]);
    }

    #[rstest]
    #[case::v2_1(LanceFileVersion::V2_1)]
    #[case::v2_2(LanceFileVersion::V2_2)]
    #[tokio::test]
    async fn test_repeated_appends_without_legacy_blob_never_rebuild_snapshot(
        #[case] file_version: LanceFileVersion,
    ) {
        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                data_storage_version: Some(file_version),
                ..Default::default()
            })
            .execute(vec![id_batch([0])])
            .await
            .unwrap();
        let mut tx = dataset.begin_transaction().await.unwrap();
        assert!(!tx.state.append_context_dirty);
        tx.append(vec![id_batch([1])]).await.unwrap();
        assert!(!tx.state.append_context_dirty);
        assert!(tx.state.staged_dataset.is_none());
        assert!(Arc::ptr_eq(&tx.state.append_base, &tx.base));

        tx.append(vec![id_batch([2])]).await.unwrap();
        assert!(!tx.state.append_context_dirty);
        assert!(tx.state.staged_dataset.is_none());
        assert!(Arc::ptr_eq(&tx.state.append_base, &tx.base));

        let committed = tx.commit().await.unwrap();
        assert_eq!(ids(&committed).await, [0, 1, 2]);
    }

    #[tokio::test]
    async fn test_legacy_blob_promotion_rebuilds_append_schema_once() {
        use arrow_array::LargeBinaryArray;
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_arrow::BLOB_META_KEY;

        let blob_meta = HashMap::from([(BLOB_META_KEY.to_string(), "true".to_string())]);
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("blob", DataType::LargeBinary, true).with_metadata(blob_meta),
        ]));
        let batch = |id: i32, payload: &[u8]| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![id])) as ArrayRef,
                    Arc::new(LargeBinaryArray::from(vec![Some(payload)])) as ArrayRef,
                ],
            )
            .unwrap()
        };
        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute(vec![batch(0, b"v21")])
            .await
            .unwrap();
        assert!(!dataset.schema().field("blob").unwrap().is_blob_v2());

        let v22_params = WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append_with_params(vec![batch(1, b"first-v22")], &v22_params)
            .await
            .unwrap();
        assert!(tx.state.append_context_dirty);
        assert!(tx.state.staged_dataset.is_none());

        // The second 2.2 append rebuilds the snapshot once so the writer sees
        // the promoted blob v2 schema, and does not leave the context dirty
        // because the schema no longer has an unpromoted legacy blob field.
        tx.append_with_params(vec![batch(2, b"second-v22")], &v22_params)
            .await
            .unwrap();
        assert!(!tx.state.append_context_dirty);
        assert!(tx.state.staged_dataset.is_none());
        assert!(
            tx.state
                .append_base
                .schema()
                .field("blob")
                .unwrap()
                .is_blob_v2()
        );
        let promoted_base = tx.state.append_base.clone();

        // A third 2.2 append reuses the promoted append_base without building
        // another snapshot.
        tx.append_with_params(vec![batch(3, b"third-v22")], &v22_params)
            .await
            .unwrap();
        assert!(!tx.state.append_context_dirty);
        assert!(tx.state.staged_dataset.is_none());
        assert!(Arc::ptr_eq(&tx.state.append_base, &promoted_base));

        let committed = tx.commit().await.unwrap();
        assert!(committed.schema().field("blob").unwrap().is_blob_v2());
        assert_eq!(ids(&committed).await, [0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn test_commit_options_apply_to_the_single_publication() {
        let dir = TempStrDir::default();
        let (dataset, tx) = table_with_staged_append(dir.as_str()).await;
        let identity = tx.id().to_owned();
        let properties = HashMap::from([("job".to_string(), "nightly".to_string())]);
        let committed = tx
            .commit_with()
            .with_max_retries(3)
            .with_retry_timeout(Duration::from_secs(5))
            .with_transaction_properties(properties.clone())
            .execute()
            .await
            .unwrap();
        assert_eq!(committed.version_id(), dataset.version_id() + 1);
        let recorded = Dataset::open(dir.as_str())
            .await
            .unwrap()
            .read_transaction_by_version(committed.version_id())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(recorded.uuid, identity);
        assert_eq!(
            recorded.transaction_properties.as_deref(),
            Some(&properties)
        );
    }

    #[tokio::test]
    async fn test_a_zero_commit_timeout_publishes_nothing() {
        let dir = TempStrDir::default();
        let (dataset, tx) = table_with_staged_append(dir.as_str()).await;
        let error = tx
            .commit_with()
            .with_timeout(Some(Duration::ZERO))
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("timeout"), "{error}");
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(reopened.version_id(), dataset.version_id());
        assert_eq!(ids(&reopened).await, [0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn test_staged_commit_skips_auto_cleanup() {
        use mock_instant::thread_local::MockClock;

        MockClock::set_system_time(Duration::from_secs(1));
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                auto_cleanup: Some(crate::dataset::AutoCleanupParams {
                    interval: 1,
                    older_than: chrono::TimeDelta::zero(),
                }),
                ..Default::default()
            })
            .execute(vec![id_batch([0])])
            .await
            .unwrap();
        MockClock::set_system_time(Duration::from_secs(2));
        let mut dataset = Arc::new(dataset);
        for id in 1..=2 {
            let mut tx = dataset.begin_transaction().await.unwrap();
            tx.append(vec![id_batch([id])]).await.unwrap();
            dataset = Arc::new(tx.commit().await.unwrap());
            MockClock::set_system_time(Duration::from_secs(2 + id as u64));
        }
        assert_eq!(dataset.version_id(), 3);
        assert_eq!(ids(&dataset.checkout_version(1).await.unwrap()).await, [0]);
        assert_eq!(
            ids(&dataset.checkout_version(2).await.unwrap()).await,
            [0, 1]
        );
    }
}
