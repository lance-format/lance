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
mod tests;
