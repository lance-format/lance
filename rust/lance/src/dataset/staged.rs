// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Staged transactions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::RecordBatch;
use lance_datafusion::utils::StreamingWriteSource;
use lance_file::version::ConcreteFileVersion;
use lance_index::{IndexParams, IndexType};
use lance_table::format::{
    BasePath, DataFile, Fragment, IndexMetadata, Manifest, RowIdMeta, is_detached_version,
};
use lance_table::io::commit::{CommitHandler, ManifestLocation};
use lance_table::rowids::read_row_ids;
use lance_table::transaction::action::{
    Action, AddBase, AddDataFile, AddFragment, AddFragmentParts, AddIndexSegment,
    AddIndexSegmentParts, ApplyState, CompositeOperation, FieldRef, Ref, UserStep,
};
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::transaction::{Operation, Transaction};
use super::{CommitBuilder, InsertBuilder, ManifestWriteConfig, Scanner, WriteMode, WriteParams};
use crate::datatypes::Schema;
use crate::index::CreateIndexBuilder;
use crate::index::create::resolve_staged_index_name;
use crate::index::loading::load_all_indices;
use crate::index::local_fragment_addresses::{
    ensure_index_can_rebase_local_fragments, ensure_segment_can_rebase_local_fragments,
};
use crate::index::scalar::{IndexDetails, load_scalar_segment_params};
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

/// An experimental staged transaction committed as one dataset version.
///
/// Each step sees the base plus the steps before it:
/// - [`Self::add_bases`] registers new storage bases.
/// - [`Self::append`] / [`Self::append_stream`] stage new fragments.
/// - [`Self::create_index`] builds a new index over all visible fragments
///   (committed and staged).
/// - [`Self::extend_index`] adds coverage for uncovered fragments to an
///   existing index while preserving its existing segments.
/// - [`Self::scan`] queries the uncommitted snapshot (including staged index
///   segments) and pins publication to [`Self::base_version`].
/// - [`Self::commit`] / [`Self::commit_with`] publish all staged steps as one
///   dataset version, or [`Self::abort`] abandons them without publishing.
///
/// Steps that have not been scanned can rebase over concurrent commits.
/// [`Self::scan`] pins the base, and [`Self::commit`] then requires that
/// version to still be the latest, so the row ids and addresses the scanner
/// returned stay valid. That commit gets one attempt. A commit with no staged
/// steps returns the base snapshot and does not check for a newer version.
/// Retries reuse the data and index files already written.
///
/// A failed step leaves the previous snapshot in place. Aborting or dropping
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
/// # use lance_index::IndexType;
/// # use lance_index::scalar::ScalarIndexParams;
/// # async fn example(dataset: &Dataset, batches: Vec<RecordBatch>) -> Result<()> {
/// let mut tx = dataset.begin_transaction().await?;
/// tx.append(batches).await?;
/// tx.create_index(&["id"], IndexType::BTree, None, &ScalarIndexParams::default()).await?;
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

    /// Publish every staged step as one version.
    ///
    /// See [`StagedTransaction::commit`] for a transaction with no staged
    /// steps, and for a commit whose outcome is unknown.
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
    /// 2.2 or newer data file promoted a legacy blob field or a base was added.
    append_context_dirty: bool,
    next_local_fragment_id: u32,
    next_local_base_id: u32,
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
        self.with_private_manifest(manifest, indices)
    }

    fn with_private_manifest(&self, manifest: Manifest, indices: Vec<IndexMetadata>) -> Self {
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

    /// Begin an experimental staged transaction from this committed version.
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
                next_local_base_id: 0,
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
    /// routes files to committed bases, while `target_base_names_or_paths` and
    /// `target_all_bases` can also route to bases added earlier in the same
    /// transaction. `enable_v2_manifest_paths` and `skip_auto_cleanup` must
    /// stay at their defaults. Publication is [`Self::commit`]. A rejected
    /// setting leaves the previous snapshot intact.
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
        let next_local_fragment_id = next_local_id("fragment", start, fragments.len())?;
        let mut actions = Vec::new();
        for (offset, fragment) in fragments.into_iter().enumerate() {
            let local_id = next_local_id("fragment", start, offset)?;
            actions.extend(actions_for_appended_fragment(
                Ref::Local(local_id),
                fragment,
                |base_id| self.base_reference(base_id),
            )?);
        }
        let step = UserStep::new("append", actions);
        // Managed and external Blob v2 descriptors bake staging base ids into
        // immutable data files, so pin the base version once a base was added.
        let pins_base_version = step_pins_base_version(&step, &self.base.manifest)?
            || (self.state.next_local_base_id > 0
                && schema_has_blob(&self.state.append_base.manifest.schema));
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

    /// Add new storage bases in this transaction.
    ///
    /// Later appends in the same transaction can write to these bases using
    /// `WriteParams::target_base_names_or_paths` or
    /// `WriteParams::target_all_bases`, and staged indexes can cover fragments
    /// stored in them. Each base must have `id == 0`, and its final base id is
    /// assigned on commit. Per base store credentials are not yet supported on
    /// staged appends because `WriteParams::base_store_params` must stay unset.
    pub fn add_bases(&mut self, new_bases: Vec<BasePath>) -> Result<()> {
        if new_bases.is_empty() {
            return Ok(());
        }
        let start = self.state.next_local_base_id;
        let next_local_base_id = next_local_id("base", start, new_bases.len())?;
        let mut actions = Vec::with_capacity(new_bases.len());
        for (offset, base) in new_bases.into_iter().enumerate() {
            if base.id != 0 {
                return Err(Error::invalid_input(format!(
                    "Staged base registration requires id 0, got {} for '{}'",
                    base.id, base.path
                )));
            }
            let local = next_local_id("base", start, offset)?;
            actions.push(Action::AddBase(AddBase::new(local, base)));
        }
        let step = UserStep::new("add_bases", actions);
        let next_working = self.state.working.apply_step_checked(&step)?;
        self.state.working = next_working;
        self.state.steps.push(step);
        self.state.staged_dataset = None;
        self.state.append_context_dirty = true;
        self.state.next_local_base_id = next_local_base_id;
        Ok(())
    }

    fn base_reference(&self, base_id: u32) -> Result<Ref> {
        if self.base.manifest.base_paths.contains_key(&base_id) {
            return Ok(Ref::Committed(u64::from(base_id)));
        }
        let base = self
            .state
            .append_base
            .manifest
            .base_paths
            .get(&base_id)
            .ok_or_else(|| {
                Error::internal(format!(
                    "Prepared file references unknown storage base {base_id}"
                ))
            })?;
        self.state
            .steps
            .iter()
            .flat_map(|step| &step.actions)
            .find_map(|action| match action {
                Action::AddBase(added) if added.base().path == base.path => {
                    Some(Ref::Local(added.local()))
                }
                _ => None,
            })
            .ok_or_else(|| {
                Error::internal(format!(
                    "Storage base {base_id} has no staged AddBase action"
                ))
            })
    }

    /// Build a new index over all fragments visible in this transaction
    /// (committed and staged).
    ///
    /// Fails if an index with the same name already exists. Use
    /// [`Self::extend_index`] to add coverage to an existing index.
    ///
    /// Committed fragments are indexed without aliases. Staged fragments store
    /// their local fragment id in the index files and record
    /// [`IndexMetadata::local_fragment_aliases`] on commit so retries do not
    /// rewrite index files. Index types that cannot rebase through
    /// `local_fragment_aliases` are rejected when staged fragments are present.
    pub async fn create_index(
        &mut self,
        columns: &[&str],
        index_type: IndexType,
        name: Option<String>,
        params: &dyn IndexParams,
    ) -> Result<()> {
        let snapshot = self.staged_dataset()?;
        let indices = load_all_indices(&snapshot).await?;
        let index_name = resolve_staged_index_name(
            snapshot.schema(),
            &indices,
            columns,
            name.as_deref(),
            index_type,
            params,
        )?;
        let plans = self.unindexed_segment_plans(&index_name, &indices, &snapshot)?;
        if plans.is_empty() {
            return Err(Error::invalid_input(format!(
                "Index '{index_name}' has no fragments to cover in this transaction"
            )));
        }
        self.stage_index_step(
            "create_index",
            &snapshot,
            columns,
            index_type,
            params,
            &index_name,
            false,
            plans,
        )
        .await
    }

    /// Extend an existing index over uncovered fragments visible in this
    /// transaction (committed and staged) without rebuilding existing segments.
    ///
    /// Reuses the existing index's column and params. Returns `Ok(())` without
    /// staging a step if every visible fragment is already covered. Fails if
    /// the index does not exist, is a vector, covering, or system index, or if
    /// staged fragments need indexing and the index type cannot rebase through
    /// `local_fragment_aliases`.
    pub async fn extend_index(&mut self, name: &str) -> Result<()> {
        let snapshot = self.staged_dataset()?;
        let indices = load_all_indices(&snapshot).await?;
        let existing = indices
            .iter()
            .find(|idx| idx.name == name)
            .ok_or_else(|| Error::index(format!("Index '{name}' does not exist")))?;
        if lance_table::system_index::is_system_index(existing) {
            return Err(Error::not_supported(format!(
                "Staged extend_index cannot extend system index '{name}'"
            )));
        }
        if !existing.covering_fields.is_empty() {
            return Err(Error::not_supported(format!(
                "Staged extend_index cannot extend covering index '{name}'"
            )));
        }
        if existing
            .index_details
            .as_ref()
            .is_some_and(|details| IndexDetails(details.clone()).is_vector())
        {
            return Err(Error::not_supported(format!(
                "Staged extend_index does not support vector index '{name}'"
            )));
        }
        let plans = self.unindexed_segment_plans(name, &indices, &snapshot)?;
        if plans.is_empty() {
            return Ok(());
        }
        let field_id = existing.keyed_field().ok_or_else(|| {
            Error::invalid_input(format!(
                "Index '{name}' must have one keyed field, got {:?}",
                existing.keyed_fields()
            ))
        })?;
        let column = snapshot.schema().field_path(field_id)?;
        let params = load_scalar_segment_params(&snapshot, existing).await?;
        self.stage_index_step(
            "extend_index",
            &snapshot,
            &[column.as_str()],
            IndexType::Scalar,
            &params,
            name,
            true,
            plans,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn stage_index_step(
        &mut self,
        step_name: &'static str,
        snapshot: &Dataset,
        columns: &[&str],
        index_type: IndexType,
        params: &dyn IndexParams,
        index_name: &str,
        copy_of_existing: bool,
        plans: Vec<SegmentPlan>,
    ) -> Result<()> {
        match index_type {
            IndexType::FragmentReuse => {
                return Err(Error::invalid_input(
                    "Fragment reuse index can only be created through compaction",
                ));
            }
            IndexType::MemWal => {
                return Err(Error::not_supported(
                    "Staged create_index cannot build the MemWAL system index",
                ));
            }
            _ => {}
        }
        if plans.iter().any(SegmentPlan::has_local_fragments) {
            ensure_index_can_rebase_local_fragments(index_type, params)?;
        }
        let mut actions = Vec::with_capacity(plans.len());
        for mut plan in plans {
            let has_local_fragments = plan.has_local_fragments();
            plan.fragments.sort_unstable_by_key(|fragment| fragment.id);
            let expected_coverage: RoaringBitmap = plan
                .fragments
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect();
            let mut manifest = Manifest::new_from_previous(
                &self.base.manifest,
                snapshot.manifest.schema.clone(),
                Arc::new(plan.fragments),
            );
            manifest.version = self.base.manifest.version;
            manifest.reader_feature_flags = snapshot.manifest.reader_feature_flags;
            manifest.writer_feature_flags = snapshot.manifest.writer_feature_flags;
            manifest.next_row_id = snapshot.manifest.next_row_id;
            manifest.base_paths = snapshot.manifest.base_paths.clone();
            // Keep the committed base version so CreateIndexBuilder does not
            // reject this private handle as an uncommitted staged snapshot, and
            // pass no existing indices because index_name was already resolved.
            let mut index_dataset = self.base.with_private_manifest(manifest, Vec::new());
            let mut builder =
                CreateIndexBuilder::new(&mut index_dataset, columns, index_type, params)
                    .name(index_name.to_string());
            if copy_of_existing {
                builder = builder.copy_of_existing();
            }
            let segment = builder.execute_uncommitted().await?;
            if has_local_fragments {
                ensure_segment_can_rebase_local_fragments(&self.base.manifest, &segment)?;
            }
            if segment.fragment_bitmap.as_ref() != Some(&expected_coverage) {
                return Err(Error::not_supported(format!(
                    "Staged index '{}' did not cover all selected fragments because they contained too few rows to train",
                    segment.name
                )));
            }
            let fields = segment
                .keyed_fields()
                .iter()
                .copied()
                .map(FieldRef::Committed)
                .collect();
            actions.push(Action::AddIndexSegment(AddIndexSegment::try_new_unchecked(
                AddIndexSegmentParts {
                    uuid: segment.uuid,
                    name: segment.name,
                    fields,
                    covering_fields: segment
                        .covering_fields
                        .into_iter()
                        .map(FieldRef::Committed)
                        .collect(),
                    index_details: segment.index_details,
                    index_version: segment.index_version,
                    covered_fragments: Some(plan.covered_fragments),
                    files: segment.files.unwrap_or_default(),
                    base: None,
                    created_at: segment.created_at,
                    dataset_version: None,
                    data_change: false,
                },
            )?));
        }
        let step = UserStep::new(step_name, actions);
        let next_working = self.state.working.apply_step_checked(&step)?;
        self.state.working = next_working;
        self.state.steps.push(step);
        self.state.staged_dataset = None;
        Ok(())
    }

    fn unindexed_segment_plans(
        &self,
        index_name: &str,
        indices: &[IndexMetadata],
        snapshot: &Dataset,
    ) -> Result<Vec<SegmentPlan>> {
        let mut covered_ids = RoaringBitmap::new();
        let mut covers_all_committed = false;
        for segment in indices.iter().filter(|idx| idx.name == index_name) {
            match &segment.fragment_bitmap {
                Some(bitmap) => covered_ids |= bitmap,
                None => covers_all_committed = true,
            }
        }
        let mut local_fragment_ids =
            HashMap::with_capacity(self.state.next_local_fragment_id as usize);
        for local_id in 0..self.state.next_local_fragment_id {
            let snapshot_id = self.state.working.resolve_fragment(Ref::Local(local_id))?;
            local_fragment_ids.insert(snapshot_id, local_id);
        }
        let mut committed = SegmentPlan::default();
        let mut local = SegmentPlan::default();
        for fragment in snapshot.manifest.fragments.iter() {
            let already_covered = u32::try_from(fragment.id)
                .ok()
                .is_some_and(|id| covered_ids.contains(id));
            if already_covered {
                continue;
            }
            if let Some(&local_id) = local_fragment_ids.get(&fragment.id) {
                let mut local_fragment = fragment.clone();
                local_fragment.id = u64::from(local_id);
                local.fragments.push(local_fragment);
                local.covered_fragments.push(Ref::Local(local_id));
            } else if !covers_all_committed {
                u32::try_from(fragment.id).map_err(|_| {
                    Error::invalid_input(format!(
                        "Committed fragment {} does not fit in an index fragment id",
                        fragment.id
                    ))
                })?;
                committed.fragments.push(fragment.clone());
                committed
                    .covered_fragments
                    .push(Ref::Committed(fragment.id));
            }
        }
        let mut plans = Vec::with_capacity(2);
        if !committed.fragments.is_empty() {
            plans.push(committed);
        }
        if !local.fragments.is_empty() {
            plans.push(local);
        }
        Ok(plans)
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

    /// Publish every staged step as one version.
    ///
    /// The transaction is consumed on failure. A transaction with no staged
    /// steps returns the original snapshot, including after a scan or a
    /// concurrent commit.
    pub async fn commit(self) -> Result<Dataset> {
        self.commit_with().execute().await
    }

    /// Abandon this transaction without publishing.
    ///
    /// No objects are deleted. A scanner that already started can finish, and
    /// unpublished objects stay eligible for orphan cleanup.
    pub fn abort(self) {}
}

#[derive(Default)]
struct SegmentPlan {
    fragments: Vec<Fragment>,
    covered_fragments: Vec<Ref>,
}

impl SegmentPlan {
    fn has_local_fragments(&self) -> bool {
        self.covered_fragments
            .iter()
            .any(|reference| matches!(reference, Ref::Local(_)))
    }
}

fn append_params() -> WriteParams {
    WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    }
}

fn next_local_id(kind: &str, start: u32, count: usize) -> Result<u32> {
    let added = u32::try_from(count).map_err(|_| {
        Error::invalid_input(format!(
            "Staged step adds {count} {kind}s, which does not fit in a local {kind} id"
        ))
    })?;
    start.checked_add(added).ok_or_else(|| {
        Error::invalid_input(format!(
            "Staged transaction already holds {start} local {kind} ids and this step adds {count}"
        ))
    })
}

/// Lower one fragment from an uncommitted append into actions.
///
/// The fragment is taken apart field by field. A new field fails to compile
/// until it is carried onto [`AddFragment`] or refused here.
fn actions_for_appended_fragment(
    id: Ref,
    fragment: Fragment,
    resolve_base: impl Fn(u32) -> Result<Ref>,
) -> Result<Vec<Action>> {
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
        actions.push(Action::AddDataFile(data_file_action(
            id,
            file,
            &resolve_base,
        )?));
    }
    Ok(actions)
}

fn unstageable_fragment(part: &str, first_path: &str) -> Error {
    Error::not_supported(format!(
        "Staged append cannot stage a new fragment that carries {part}. Its first data file is {first_path}"
    ))
}

fn data_file_action(
    fragment: Ref,
    file: DataFile,
    resolve_base: &impl Fn(u32) -> Result<Ref>,
) -> Result<AddDataFile> {
    let field_ids = file
        .fields
        .iter()
        .copied()
        .map(FieldRef::Committed)
        .collect();
    let base = file.base_id.map(resolve_base).transpose()?;
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

fn schema_has_blob(schema: &Schema) -> bool {
    schema
        .fields_pre_order()
        .any(|field| field.is_blob() || field.is_blob_v2())
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
    use crate::index::vector::VectorIndexParams;
    use crate::io::exec::filtered_read::FilteredReadExec;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, record_batch};
    use datafusion::error::DataFusionError;
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::{BuiltinIndexType, InvertedIndexParams, ScalarIndexParams};
    use lance_linalg::distance::MetricType;
    use lance_table::feature_flags::FLAG_LOCAL_FRAGMENT_ALIASES;
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

    #[rstest]
    #[case::uncontended(false)]
    #[case::reallocated(true)]
    #[tokio::test]
    async fn test_staged_create_index_follows_fragment_allocation(
        #[case] racer: bool,
        #[values(IndexType::BTree, IndexType::Scalar)] index_type: IndexType,
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [7, 8, 9])).unwrap()])
            .await
            .unwrap();
        tx.create_index(&["id"], index_type, Some("by_id".into()), &params)
            .await
            .unwrap();
        if racer {
            let mut other = dataset.begin_transaction().await.unwrap();
            other
                .append(vec![record_batch!(("id", Int32, [2])).unwrap()])
                .await
                .unwrap();
            other.commit().await.unwrap();
        }
        let committed = tx.commit().await.unwrap();
        let indices = load_all_indices(&committed).await.unwrap();
        let by_id: Vec<_> = indices
            .iter()
            .filter(|index| index.name == "by_id")
            .collect();
        assert_eq!(by_id.len(), 2);
        let committed_seg = by_id
            .iter()
            .find(|index| index.local_fragment_aliases.is_empty())
            .expect("committed segment");
        assert_eq!(
            committed_seg.fragment_bitmap,
            Some(RoaringBitmap::from_iter([0]))
        );
        let local_seg = by_id
            .iter()
            .find(|index| !index.local_fragment_aliases.is_empty())
            .expect("local segment");
        let covered = local_seg.fragment_bitmap.as_ref().unwrap();
        let committed_id = covered.iter().next().unwrap();
        assert_eq!(covered.len(), 1);
        assert_eq!(committed_id, if racer { 2 } else { 1 });
        assert_eq!(
            local_seg.local_fragment_aliases.get(&0),
            Some(&committed_id)
        );
        assert_eq!(local_seg.results_are_row_addrs(), stable);
        assert_ne!(
            committed.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_ne!(
            committed.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            committed
                .count_indexed_rows("by_id", "id >= 0", None, Some(&[0, committed_id]))
                .await
                .unwrap(),
            5
        );
        assert_eq!(
            committed
                .count_indexed_rows("by_id", "id >= 7", None, Some(&[committed_id]))
                .await
                .unwrap(),
            3
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_staged_create_index_extends_existing_index_and_clears_aliases_on_maintenance(
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let mut dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let params = ScalarIndexParams::default();
        dataset
            .create_index(&["id"], IndexType::BTree, None, &params, false)
            .await
            .unwrap();

        let mut first = dataset.begin_transaction().await.unwrap();
        first
            .append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
            .await
            .unwrap();
        first
            .append(vec![record_batch!(("id", Int32, [4, 5])).unwrap()])
            .await
            .unwrap();
        first.extend_index("id_idx").await.unwrap();

        let mut second = dataset.begin_transaction().await.unwrap();
        second
            .append(vec![record_batch!(("id", Int32, [6, 7])).unwrap()])
            .await
            .unwrap();
        second.extend_index("id_idx").await.unwrap();
        second
            .append(vec![record_batch!(("id", Int32, [8, 9])).unwrap()])
            .await
            .unwrap();
        second.extend_index("id_idx").await.unwrap();

        first.commit().await.unwrap();
        let mut committed = second.commit().await.unwrap();
        let segments = committed.load_indices_by_name("id_idx").await.unwrap();
        assert_eq!(segments.len(), 4);
        let indexed_ids: Vec<u32> = committed.manifest.fragments[1..]
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect();
        assert_eq!(indexed_ids, vec![1, 2, 3, 4]);
        assert_eq!(
            committed
                .count_indexed_rows("id_idx", "id >= 2", None, Some(&indexed_ids))
                .await
                .unwrap(),
            8
        );

        compact_files(
            &mut committed,
            CompactionOptions {
                target_rows_per_fragment: 100,
                excluded_fragment_ids: vec![0, 3, 4],
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        let after_compact = Dataset::open(dir.as_str()).await.unwrap();
        let compacted_seg = load_all_indices(&after_compact)
            .await
            .unwrap()
            .iter()
            .find(|segment| segment.fragment_bitmap == Some(RoaringBitmap::from_iter([5])))
            .expect("compacted segment")
            .clone();
        assert!(compacted_seg.local_fragment_aliases.is_empty());
        assert_ne!(
            after_compact.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            after_compact
                .count_indexed_rows("id_idx", "id >= 2", None, Some(&[3, 4, 5]))
                .await
                .unwrap(),
            8
        );

        committed
            .optimize_indices(&OptimizeOptions::merge(4))
            .await
            .unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(
            reopened.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(
            reopened.manifest.writer_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        let all_ids: Vec<u32> = reopened
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect();
        assert_eq!(
            reopened
                .count_indexed_rows("id_idx", "id >= 2", None, Some(&all_ids))
                .await
                .unwrap(),
            8
        );
    }

    #[tokio::test]
    async fn test_staged_create_index_rejects_unsupported_inputs() {
        let mut dataset = InsertBuilder::new("memory://")
            .execute(vec![
                record_batch!(("id", Int32, [0, 1]), ("text", Utf8, ["alpha", "beta"])).unwrap(),
            ])
            .await
            .unwrap();
        let params = ScalarIndexParams::default();
        dataset
            .create_index(&["id"], IndexType::BTree, None, &params, false)
            .await
            .unwrap();
        let inverted_scalar = ScalarIndexParams::new("inverted".to_string());
        let inverted_params = InvertedIndexParams::default();
        let minhash_params = ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh);
        let fm_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Fm);
        let vector_params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 2);
        let mut tx = dataset.begin_transaction().await.unwrap();
        // Existing index already covers all committed fragments and nothing is staged yet:
        // extend_index is a no-op.
        tx.extend_index("id_idx").await.unwrap();
        assert!(tx.state.steps.is_empty());
        // Extending a non-existent index fails with Error::Index.
        let missing_index = tx.extend_index("no_such_index").await.unwrap_err();
        assert!(
            matches!(missing_index, Error::Index { .. }),
            "{missing_index}"
        );
        // Creating an index whose name already exists fails with Error::Index.
        let duplicate_name = tx
            .create_index(&["id"], IndexType::BTree, None, &params)
            .await
            .unwrap_err();
        assert!(
            matches!(duplicate_name, Error::Index { .. }),
            "{duplicate_name}"
        );

        let batch =
            record_batch!(("id", Int32, [2, 3]), ("text", Utf8, ["gamma", "delta"])).unwrap();
        tx.append(vec![batch]).await.unwrap();
        for error in [
            tx.create_index(&["text"], IndexType::Inverted, None, &inverted_params)
                .await
                .unwrap_err(),
            tx.create_index(&["text"], IndexType::Scalar, None, &inverted_scalar)
                .await
                .unwrap_err(),
            tx.create_index(&["id"], IndexType::MinHashLsh, None, &minhash_params)
                .await
                .unwrap_err(),
            tx.create_index(&["id"], IndexType::Fm, None, &fm_params)
                .await
                .unwrap_err(),
            tx.create_index(&["id"], IndexType::IvfPq, None, &vector_params)
                .await
                .unwrap_err(),
        ] {
            assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        }
        let missing_col = tx
            .create_index(&["missing"], IndexType::BTree, None, &params)
            .await
            .unwrap_err();
        assert!(matches!(missing_col, Error::Index { .. }), "{missing_col}");
        // Failed create_index / extend_index attempts leave the staged transaction intact.
        assert_eq!(tx.state.steps.len(), 1);
        assert!(!tx.state.requires_original_version);

        tx.extend_index("id_idx").await.unwrap();
        assert_eq!(tx.state.steps.len(), 2);
        // Calling extend_index again when coverage is already complete is a no-op.
        tx.extend_index("id_idx").await.unwrap();
        assert_eq!(tx.state.steps.len(), 2);

        let committed = tx.commit().await.unwrap();
        assert_eq!(ids(&committed).await, [0, 1, 2, 3]);
    }

    #[rstest]
    #[case::bitmap(IndexType::Bitmap, BuiltinIndexType::Bitmap, "id = 7", vec![7])]
    #[case::zonemap(IndexType::ZoneMap, BuiltinIndexType::ZoneMap, "id >= 7", vec![7, 8, 9, 10])]
    #[case::bloomfilter(IndexType::BloomFilter, BuiltinIndexType::BloomFilter, "id = 9", vec![9])]
    #[tokio::test]
    async fn test_staged_create_index_scalar_plugins_rebase_and_compact(
        #[case] index_type: IndexType,
        #[case] builtin: BuiltinIndexType,
        #[case] filter: &str,
        #[case] expected: Vec<i32>,
    ) {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let params = ScalarIndexParams::for_builtin(builtin);
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [7, 8])).unwrap()])
            .await
            .unwrap();
        tx.append(vec![record_batch!(("id", Int32, [9, 10])).unwrap()])
            .await
            .unwrap();
        tx.create_index(&["id"], index_type, Some("by_id".into()), &params)
            .await
            .unwrap();

        // Concurrent append shifts the staged fragments from ids 1, 2 to ids 2, 3.
        let mut racer = dataset.begin_transaction().await.unwrap();
        racer
            .append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
            .await
            .unwrap();
        racer.commit().await.unwrap();

        let mut committed = tx.commit().await.unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        let indices = load_all_indices(&reopened).await.unwrap();
        let by_id: Vec<_> = indices.iter().filter(|i| i.name == "by_id").collect();
        assert_eq!(by_id.len(), 2);
        let committed_seg = by_id
            .iter()
            .find(|i| i.local_fragment_aliases.is_empty())
            .expect("committed segment");
        assert_eq!(
            committed_seg.fragment_bitmap,
            Some(RoaringBitmap::from_iter([0]))
        );
        let local_seg = by_id
            .iter()
            .find(|i| !i.local_fragment_aliases.is_empty())
            .expect("local segment");
        assert_eq!(local_seg.local_fragment_aliases.get(&0), Some(&2));
        assert_eq!(local_seg.local_fragment_aliases.get(&1), Some(&3));

        async fn query_ids(dataset: &Dataset, filter: &str, use_index: bool) -> Vec<i32> {
            let mut scan = dataset.scan();
            scan.project(&["id"])
                .unwrap()
                .filter(filter)
                .unwrap()
                .use_scalar_index(use_index);
            if use_index {
                let plan = scan.explain_plan(true).await.unwrap();
                assert!(
                    plan.contains("ScalarIndexQuery") || plan.contains("MaterializeIndex"),
                    "expected scalar index in plan, got:\n{plan}"
                );
            }
            let batch = scan.try_into_batch().await.unwrap();
            let mut vals: Vec<i32> = batch["id"].as_primitive::<Int32Type>().values().to_vec();
            vals.sort_unstable();
            vals
        }

        assert_eq!(query_ids(&reopened, filter, true).await, expected);
        assert_eq!(query_ids(&reopened, filter, false).await, expected);

        compact_files(
            &mut committed,
            CompactionOptions {
                target_rows_per_fragment: 100,
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        // Bitmap remaps inline on compaction; ZoneMap/BloomFilter retain their
        // aliased files until optimize_indices rebuilds them. Queries must stay
        // accurate in both states.
        let after_compact = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(query_ids(&after_compact, filter, true).await, expected);

        committed
            .optimize_indices(&OptimizeOptions::merge(2))
            .await
            .unwrap();
        let after = Dataset::open(dir.as_str()).await.unwrap();
        let after_indices = load_all_indices(&after).await.unwrap();
        let by_id: Vec<_> = after_indices.iter().filter(|i| i.name == "by_id").collect();
        assert!(!by_id.is_empty());
        assert!(by_id.iter().all(|s| s.local_fragment_aliases.is_empty()));
        assert_eq!(
            after.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        assert_eq!(query_ids(&after, filter, true).await, expected);
    }

    #[tokio::test]
    async fn test_staged_create_index_ngram_rebase_and_preview() {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![
                record_batch!(("text", Utf8, ["alpha", "bravo"])).unwrap(),
            ])
            .await
            .unwrap();
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::NGram);
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![
            record_batch!(("text", Utf8, ["charlie", "delta force"])).unwrap(),
        ])
        .await
        .unwrap();
        tx.create_index(
            &["text"],
            IndexType::NGram,
            Some("text_idx".into()),
            &params,
        )
        .await
        .unwrap();

        // Preview the uncommitted staged index through tx.scan() and verify
        // the plan uses the scalar index rather than a full-scan fallback.
        let mut preview = tx.scan().unwrap();
        preview
            .project(&["text"])
            .unwrap()
            .filter("contains(text, 'delta')")
            .unwrap();
        let preview_plan = preview.explain_plan(true).await.unwrap();
        assert!(
            preview_plan.contains("ScalarIndexQuery") || preview_plan.contains("MaterializeIndex"),
            "expected scalar index in staged preview plan, got:\n{preview_plan}"
        );
        let preview_batch = preview.try_into_batch().await.unwrap();
        assert_eq!(preview_batch.num_rows(), 1);

        let committed = tx.commit().await.unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        let mut indexed = reopened.scan();
        indexed
            .project(&["text"])
            .unwrap()
            .filter("contains(text, 'delta')")
            .unwrap();
        let batch = indexed.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(committed.count_rows(None).await.unwrap(), 4);
    }

    #[tokio::test]
    async fn test_staged_create_index_abort_leaves_orphan_files_cleanable() {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let params = ScalarIndexParams::default();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
            .await
            .unwrap();
        tx.create_index(&["id"], IndexType::BTree, Some("by_id".into()), &params)
            .await
            .unwrap();

        let indices_dir = std::path::Path::new(dir.as_str()).join("_indices");
        let staged_entries: Vec<_> = std::fs::read_dir(&indices_dir)
            .unwrap()
            .collect::<std::io::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(staged_entries.len(), 2);

        tx.abort();

        let policy = crate::dataset::cleanup::CleanupPolicyBuilder::default()
            .before_timestamp(crate::utils::temporal::utc_now() + chrono::TimeDelta::days(1))
            .delete_unverified(true)
            .build();
        let stats = dataset.cleanup_with_policy(policy).await.unwrap();
        assert!(stats.bytes_removed > 0);
        let remaining_files = std::fs::read_dir(&indices_dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .flat_map(|entry| std::fs::read_dir(entry.path()).into_iter().flatten())
            .count();
        assert_eq!(remaining_files, 0);

        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1]);
        assert!(load_all_indices(&reopened).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_staged_create_index_committed_only_and_partial_coverage() {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let params = ScalarIndexParams::default();

        // Committed-only staged create_index (no staged append).
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.create_index(&["id"], IndexType::BTree, Some("by_id".into()), &params)
            .await
            .unwrap();
        let committed = tx.commit().await.unwrap();
        assert_eq!(
            committed.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
            0
        );
        let segs = committed.load_indices_by_name("by_id").await.unwrap();
        assert_eq!(segs.len(), 1);
        assert!(segs[0].local_fragment_aliases.is_empty());
        assert_eq!(segs[0].fragment_bitmap, Some(RoaringBitmap::from_iter([0])));

        // Append committed fragment 1 without indexing it -> partial committed coverage.
        let with_uncovered = InsertBuilder::new(Arc::new(committed))
            .with_params(&append_params())
            .execute(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
            .await
            .unwrap();

        // Stage local fragment 0 and call extend_index: should build one committed
        // segment for fragment 1 (no aliases) and one local segment for local 0.
        let mut tx2 = with_uncovered.begin_transaction().await.unwrap();
        tx2.append(vec![record_batch!(("id", Int32, [4, 5])).unwrap()])
            .await
            .unwrap();
        tx2.extend_index("by_id").await.unwrap();
        // Append another local fragment and extend the same index again in the same tx.
        tx2.append(vec![record_batch!(("id", Int32, [6, 7])).unwrap()])
            .await
            .unwrap();
        tx2.extend_index("by_id").await.unwrap();
        let final_ds = tx2.commit().await.unwrap();
        let all_segs = final_ds.load_indices_by_name("by_id").await.unwrap();
        assert_eq!(all_segs.len(), 4);
        let mut covered = RoaringBitmap::new();
        for seg in &all_segs {
            covered |= seg.fragment_bitmap.as_ref().unwrap();
        }
        assert_eq!(covered, RoaringBitmap::from_iter([0, 1, 2, 3]));
        assert_eq!(
            final_ds
                .count_indexed_rows("by_id", "id >= 0", None, Some(&[0, 1, 2, 3]))
                .await
                .unwrap(),
            8
        );
    }

    #[tokio::test]
    async fn test_staged_create_then_append_then_extend_in_same_transaction_and_committed_only_inverted_vector()
     {
        use arrow_array::types::Float32Type;
        use arrow_array::{FixedSizeListArray, StringArray};
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

        let dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("text", DataType::Utf8, false),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    4,
                ),
                false,
            ),
        ]));
        let make_batch = |ids: &[i32]| {
            let texts: Vec<String> = ids.iter().map(|i| format!("doc {i}")).collect();
            let vecs = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                ids.iter().map(|&i| {
                    let f = i as f32;
                    Some([Some(f), Some(f + 1.0), Some(f + 2.0), Some(f + 3.0)])
                }),
                4,
            );
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
                    Arc::new(StringArray::from(texts)) as ArrayRef,
                    Arc::new(vecs) as ArrayRef,
                ],
            )
            .unwrap()
        };
        let initial_ids: Vec<i32> = (0..16).collect();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![make_batch(&initial_ids)])
            .await
            .unwrap();

        let btree_params = ScalarIndexParams::default();
        let inverted_params = InvertedIndexParams::default();
        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);

        // 1. In a single transaction with no local appends yet, build BTree, Inverted,
        // and Vector indexes over the committed fragment (no aliases required).
        // 2. Then append a local fragment and extend the BTree index created earlier
        // in this same transaction.
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".into()),
            &btree_params,
        )
        .await
        .unwrap();
        tx.create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_idx".into()),
            &inverted_params,
        )
        .await
        .unwrap();
        tx.create_index(
            &["vec"],
            IndexType::IvfFlat,
            Some("vec_idx".into()),
            &vector_params,
        )
        .await
        .unwrap();

        tx.append(vec![make_batch(&[100, 101])]).await.unwrap();
        tx.extend_index("id_idx").await.unwrap();

        let committed = tx.commit().await.unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        let id_segs = reopened.load_indices_by_name("id_idx").await.unwrap();
        assert_eq!(id_segs.len(), 2);
        let text_segs = reopened.load_indices_by_name("text_idx").await.unwrap();
        assert_eq!(text_segs.len(), 1);
        assert!(text_segs[0].local_fragment_aliases.is_empty());
        let vec_segs = reopened.load_indices_by_name("vec_idx").await.unwrap();
        assert_eq!(vec_segs.len(), 1);
        assert!(vec_segs[0].local_fragment_aliases.is_empty());
        assert_eq!(
            reopened
                .count_indexed_rows("id_idx", "id >= 100", None, Some(&[0, 1]))
                .await
                .unwrap(),
            2
        );
        assert_eq!(committed.count_rows(None).await.unwrap(), 18);
    }

    #[tokio::test]
    async fn test_staged_create_index_label_list_and_rtree_two_segments() {
        use arrow_array::ListArray;
        use arrow_buffer::OffsetBuffer;
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use geo_types::line_string;
        use geoarrow_array::GeoArrowArray;
        use geoarrow_array::builder::LineStringBuilder;
        use geoarrow_schema::{Dimension, LineStringType};

        let dir = TempStrDir::default();
        let line_type = LineStringType::new(Dimension::XY, Default::default());
        let tags_field = ArrowField::new(
            "tags",
            DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
            false,
        );
        let schema = Arc::new(ArrowSchema::new(vec![
            tags_field,
            line_type.clone().to_field("geom", true),
        ]));
        let make_batch = |tag_rows: &[&[&str]], coords: &[(f64, f64)]| {
            let mut values = Vec::new();
            let mut offsets = vec![0i32];
            for row in tag_rows {
                values.extend(row.iter().copied());
                offsets.push(values.len() as i32);
            }
            let list_arr = ListArray::new(
                Arc::new(ArrowField::new("item", DataType::Utf8, true)),
                OffsetBuffer::new(offsets.into()),
                Arc::new(arrow_array::StringArray::from(values)),
                None,
            );
            let mut line_builder = LineStringBuilder::new(line_type.clone());
            for &(x, y) in coords {
                line_builder
                    .push_line_string(Some(&line_string![(x: x, y: y), (x: x + 1.0, y: y + 1.0)]))
                    .unwrap();
            }
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(list_arr) as ArrayRef,
                    line_builder.finish().to_array_ref(),
                ],
            )
            .unwrap()
        };

        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![make_batch(
                &[&["red", "blue"], &["green"]],
                &[(0.0, 0.0), (5.0, 5.0)],
            )])
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![make_batch(
            &[&["yellow", "red"], &["purple"]],
            &[(10.0, 10.0), (20.0, 20.0)],
        )])
        .await
        .unwrap();
        let label_params = ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList);
        let rtree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::RTree);
        tx.create_index(
            &["tags"],
            IndexType::LabelList,
            Some("tags_idx".into()),
            &label_params,
        )
        .await
        .unwrap();
        tx.create_index(
            &["geom"],
            IndexType::RTree,
            Some("geom_idx".into()),
            &rtree_params,
        )
        .await
        .unwrap();

        // Concurrent append shifts local fragment from 1 to 2.
        let mut racer = dataset.begin_transaction().await.unwrap();
        racer
            .append(vec![make_batch(&[&["orange"]], &[(50.0, 50.0)])])
            .await
            .unwrap();
        racer.commit().await.unwrap();

        tx.commit().await.unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(
            reopened
                .load_indices_by_name("tags_idx")
                .await
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            reopened
                .load_indices_by_name("geom_idx")
                .await
                .unwrap()
                .len(),
            2
        );

        let mut tag_scan = reopened.scan();
        tag_scan
            .project(&["tags"])
            .unwrap()
            .filter("array_has_any(tags, ['yellow'])")
            .unwrap();
        let tag_plan = tag_scan.explain_plan(true).await.unwrap();
        assert!(
            tag_plan.contains("ScalarIndexQuery") || tag_plan.contains("MaterializeIndex"),
            "expected LabelList index in plan:\n{tag_plan}"
        );
        assert_eq!(tag_scan.try_into_batch().await.unwrap().num_rows(), 1);

        let sql = "SELECT geom FROM dataset WHERE ST_Intersects(geom, ST_GeomFromText('LINESTRING (10 11, 11 10)'))";
        let explain = reopened
            .sql(&format!("EXPLAIN {sql}"))
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        assert!(
            format!("{explain:?}").contains("ScalarIndexQuery"),
            "expected RTree index in plan:\n{explain:?}"
        );
        let geo_batches = reopened
            .sql(sql)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let geo_rows: usize = geo_batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(geo_rows, 1);
    }

    #[tokio::test]
    async fn test_staged_create_index_mixed_conflicts_with_concurrent_legacy_compaction() {
        let dir = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![record_batch!(("id", Int32, [0, 1])).unwrap()])
            .await
            .unwrap();
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&append_params())
            .execute(vec![record_batch!(("id", Int32, [2, 3])).unwrap()])
            .await
            .unwrap();
        let params = ScalarIndexParams::default();

        // Build mixed index over committed {0, 1} + local {0}.
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![record_batch!(("id", Int32, [20, 21])).unwrap()])
            .await
            .unwrap();
        tx.create_index(&["id"], IndexType::BTree, Some("by_id".into()), &params)
            .await
            .unwrap();

        // Legacy compaction commits ReserveFragments + Rewrite, which is not a
        // CompositeOperation, so conflict resolution fails closed rather than
        // publishing stale committed-fragment coverage.
        let mut compactor = Dataset::open(dir.as_str()).await.unwrap();
        compact_files(&mut compactor, merge_small_fragments(), None)
            .await
            .unwrap();

        let err = tx.commit().await.unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );

        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1, 2, 3]);
        assert!(load_all_indices(&reopened).await.unwrap().is_empty());
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

    #[rstest]
    #[case::uncontended_identity(false)]
    #[case::contended_retry(true)]
    #[tokio::test]
    async fn test_staged_create_index_on_empty_dataset_identity_and_retry(#[case] racer: bool) {
        let dir = TempStrDir::default();
        let empty_batch = id_batch(std::iter::empty());
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![empty_batch])
            .await
            .unwrap();
        assert!(dataset.manifest.fragments.is_empty());

        let params = ScalarIndexParams::default();
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![id_batch([7, 8])]).await.unwrap();
        tx.create_index(&["id"], IndexType::BTree, Some("by_id".into()), &params)
            .await
            .unwrap();

        if racer {
            let mut other = dataset.begin_transaction().await.unwrap();
            other.append(vec![id_batch([1, 2])]).await.unwrap();
            other.commit().await.unwrap();
        }

        let committed = tx.commit().await.unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();

        let recorded = reopened
            .read_transaction_by_version(committed.version_id())
            .await
            .unwrap()
            .expect("committed transaction must be recorded");
        assert!(matches!(
            recorded.operation,
            Operation::CompositeOperation(_)
        ));

        let segments = reopened.load_indices_by_name("by_id").await.unwrap();
        assert_eq!(segments.len(), 1);
        let segment = &segments[0];
        if racer {
            assert_eq!(segment.fragment_bitmap, Some(RoaringBitmap::from_iter([1])));
            assert_eq!(segment.local_fragment_aliases.get(&0), Some(&1));
            assert_ne!(
                reopened.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
                0
            );
            assert_eq!(
                reopened
                    .count_indexed_rows("by_id", "id >= 7", None, Some(&[1]))
                    .await
                    .unwrap(),
                2
            );
        } else {
            assert_eq!(segment.fragment_bitmap, Some(RoaringBitmap::from_iter([0])));
            assert!(segment.local_fragment_aliases.is_empty());
            assert_eq!(
                reopened.manifest.reader_feature_flags & FLAG_LOCAL_FRAGMENT_ALIASES,
                0
            );
            assert_eq!(
                reopened
                    .count_indexed_rows("by_id", "id >= 7", None, Some(&[0]))
                    .await
                    .unwrap(),
                2
            );
        }
    }

    #[tokio::test]
    async fn test_staged_extend_index_all_rebaseable_scalar_plugins() {
        use arrow_array::StringArray;
        use arrow_array::builder::{ListBuilder, StringBuilder};
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use geo_types::line_string;
        use geoarrow_array::GeoArrowArray;
        use geoarrow_array::builder::LineStringBuilder;
        use geoarrow_schema::{Dimension, LineStringType};

        let dir = TempStrDir::default();
        let line_type = LineStringType::new(Dimension::XY, Default::default());
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("text", DataType::Utf8, false),
            ArrowField::new(
                "tags",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                false,
            ),
            line_type.clone().to_field("geom", true),
        ]));
        type PluginRow<'a> = (i32, &'a str, &'a [&'a str], (f64, f64));
        let make_batch = |rows: &[PluginRow<'_>]| {
            let ids: Vec<i32> = rows.iter().map(|r| r.0).collect();
            let texts: Vec<&str> = rows.iter().map(|r| r.1).collect();
            let mut tags = ListBuilder::new(StringBuilder::new());
            let mut geom_builder = LineStringBuilder::new(line_type.clone());
            for row in rows {
                for &tag in row.2 {
                    tags.values().append_value(tag);
                }
                tags.append(true);
                let (x, y) = row.3;
                geom_builder
                    .push_line_string(Some(&line_string![(x: x, y: y), (x: x + 1.0, y: y + 1.0)]))
                    .unwrap();
            }
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids)) as ArrayRef,
                    Arc::new(StringArray::from(texts)) as ArrayRef,
                    Arc::new(tags.finish()) as ArrayRef,
                    geom_builder.finish().to_array_ref(),
                ],
            )
            .unwrap()
        };

        // Committed fragment 0.
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![make_batch(&[
                (1, "alpha_one", &["red"], (1.0, 1.0)),
                (2, "bravo_two", &["blue"], (2.0, 2.0)),
            ])])
            .await
            .unwrap();

        // Create all 7 rebaseable scalar index plugins on committed fragment 0.
        let btree_params = ScalarIndexParams::default();
        let bitmap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap);
        let zonemap_params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap);
        let bloom_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter);
        let ngram_params = ScalarIndexParams::for_builtin(BuiltinIndexType::NGram);
        let labellist_params = ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList);
        let rtree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::RTree);

        let mut tx = dataset.begin_transaction().await.unwrap();
        for (col, idx_type, name, params) in [
            ("id", IndexType::BTree, "id_btree", &btree_params),
            ("id", IndexType::Bitmap, "id_bitmap", &bitmap_params),
            ("id", IndexType::ZoneMap, "id_zonemap", &zonemap_params),
            ("id", IndexType::BloomFilter, "id_bloom", &bloom_params),
            ("text", IndexType::NGram, "text_ngram", &ngram_params),
            ("tags", IndexType::LabelList, "tags_ll", &labellist_params),
            ("geom", IndexType::RTree, "geom_rtree", &rtree_params),
        ] {
            tx.create_index(&[col], idx_type, Some(name.into()), params)
                .await
                .unwrap();
        }
        let indexed = tx.commit().await.unwrap();

        // Append uncovered committed fragment 1.
        let with_frag1 = InsertBuilder::new(Arc::new(indexed))
            .with_params(&append_params())
            .execute(vec![make_batch(&[
                (10, "charlie_ten", &["green"], (10.0, 10.0)),
                (11, "delta_eleven", &["yellow"], (11.0, 11.0)),
            ])])
            .await
            .unwrap();

        // Stage local fragment 0 and extend all 7 indexes in one transaction.
        let mut tx2 = with_frag1.begin_transaction().await.unwrap();
        tx2.append(vec![make_batch(&[
            (20, "echo_twenty", &["purple"], (20.0, 20.0)),
            (21, "foxtrot_twentyone", &["orange"], (21.0, 21.0)),
        ])])
        .await
        .unwrap();
        for name in [
            "id_btree",
            "id_bitmap",
            "id_zonemap",
            "id_bloom",
            "text_ngram",
            "tags_ll",
            "geom_rtree",
        ] {
            tx2.extend_index(name).await.unwrap();
        }

        // Concurrent racer claims fragment 2 so our local fragment 0 rebases to 3.
        let mut racer = with_frag1.begin_transaction().await.unwrap();
        racer
            .append(vec![make_batch(&[(
                99,
                "racer_row",
                &["gray"],
                (99.0, 99.0),
            )])])
            .await
            .unwrap();
        racer.commit().await.unwrap();

        tx2.commit().await.unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();

        for name in [
            "id_btree",
            "id_bitmap",
            "id_zonemap",
            "id_bloom",
            "text_ngram",
            "tags_ll",
            "geom_rtree",
        ] {
            let segs = reopened.load_indices_by_name(name).await.unwrap();
            assert_eq!(segs.len(), 3, "index {name} should have 3 segments");
            let mut covered = RoaringBitmap::new();
            for seg in &segs {
                covered |= seg.fragment_bitmap.as_ref().unwrap();
            }
            assert_eq!(
                covered,
                RoaringBitmap::from_iter([0, 1, 3]),
                "index {name} coverage mismatch"
            );
            let local_seg = segs
                .iter()
                .find(|s| !s.local_fragment_aliases.is_empty())
                .unwrap_or_else(|| panic!("index {name} missing local segment"));
            assert_eq!(
                local_seg.local_fragment_aliases.get(&0),
                Some(&3),
                "index {name} alias mismatch"
            );
        }

        // Verify queries across committed + extended committed + rebased local fragments.
        for (filter, expected) in [
            ("id = 1 OR id = 10 OR id = 20", vec![1, 10, 20]),
            ("contains(text, 'twenty')", vec![20, 21]),
            (
                "array_has_any(tags, ['red', 'green', 'purple'])",
                vec![1, 10, 20],
            ),
        ] {
            let mut scan = reopened.scan();
            scan.project(&["id"]).unwrap().filter(filter).unwrap();
            let batch = scan.try_into_batch().await.unwrap();
            let mut got: Vec<i32> = batch["id"].as_primitive::<Int32Type>().values().to_vec();
            got.sort_unstable();
            assert_eq!(got, expected, "filter `{filter}` mismatch");
        }
    }

    #[tokio::test]
    async fn test_staged_non_rebaseable_and_vector_indexes_matrix() {
        use arrow_array::builder::{ListBuilder, StringBuilder};
        use arrow_array::types::Float32Type;
        use arrow_array::{FixedSizeListArray, StringArray};
        use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
        use lance_index::scalar::FullTextSearchQuery;
        use lance_index::scalar::inverted::DocumentGranularity;

        let dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("text", DataType::Utf8, false),
            ArrowField::new(
                "tags",
                DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
                false,
            ),
            ArrowField::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    8,
                ),
                false,
            ),
        ]));
        let make_batch = |ids: &[i32]| {
            let texts: Vec<String> = ids
                .iter()
                .map(|i| {
                    if *i == 19 {
                        "zebra".into()
                    } else {
                        format!("alpha{i}")
                    }
                })
                .collect();
            let mut tags = ListBuilder::new(StringBuilder::new());
            for &i in ids {
                tags.values().append_value(format!("label_{i}"));
                tags.append(true);
            }
            let vecs = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                ids.iter().map(|&i| {
                    let f = i as f32;
                    Some([
                        Some(f),
                        Some(f + 1.0),
                        Some(f + 2.0),
                        Some(f + 3.0),
                        Some(f + 4.0),
                        Some(f + 5.0),
                        Some(f + 6.0),
                        Some(f + 7.0),
                    ])
                }),
                8,
            );
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
                    Arc::new(StringArray::from(texts)) as ArrayRef,
                    Arc::new(tags.finish()) as ArrayRef,
                    Arc::new(vecs) as ArrayRef,
                ],
            )
            .unwrap()
        };

        let initial: Vec<i32> = (0..16).collect();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![make_batch(&initial)])
            .await
            .unwrap();

        let fm_params = ScalarIndexParams::for_builtin(BuiltinIndexType::Fm);
        let minhash_params = ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh);
        let row_fts_params = InvertedIndexParams::default();
        let list_fts_params =
            InvertedIndexParams::default().document_granularity(DocumentGranularity::ListElement);
        let ivf_flat_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        let ivf_rq_params = VectorIndexParams::ivf_rq(1, 1, MetricType::L2);

        // Committed-only create_index works for Fm, MinHashLsh, Inverted
        // (including default naming like tags_list_element_idx), and vector
        // indexes.
        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.create_index(&["text"], IndexType::Fm, Some("text_fm".into()), &fm_params)
            .await
            .unwrap();
        tx.create_index(
            &["text"],
            IndexType::MinHashLsh,
            Some("text_minhash".into()),
            &minhash_params,
        )
        .await
        .unwrap();
        tx.create_index(&["text"], IndexType::Inverted, None, &row_fts_params)
            .await
            .unwrap();
        tx.create_index(&["tags"], IndexType::Inverted, None, &list_fts_params)
            .await
            .unwrap();
        tx.create_index(&["vec"], IndexType::IvfFlat, None, &ivf_flat_params)
            .await
            .unwrap();
        tx.create_index(
            &["vec"],
            IndexType::IvfRq,
            Some("vec_rq".into()),
            &ivf_rq_params,
        )
        .await
        .unwrap();

        assert!(matches!(
            tx.create_index(&["id"], IndexType::FragmentReuse, None, &fm_params)
                .await
                .unwrap_err(),
            Error::InvalidInput { .. }
        ));
        assert!(matches!(
            tx.create_index(&["id"], IndexType::MemWal, None, &fm_params)
                .await
                .unwrap_err(),
            Error::NotSupported { .. }
        ));

        let committed = tx.commit().await.unwrap();

        let with_frag1 = InsertBuilder::new(Arc::new(committed))
            .with_params(&append_params())
            .execute(vec![make_batch(&[16, 17, 18, 19])])
            .await
            .unwrap();

        let mut tx2 = with_frag1.begin_transaction().await.unwrap();
        for name in [
            "text_fm",
            "text_minhash",
            "text_idx",
            "tags_list_element_idx",
        ] {
            tx2.extend_index(name)
                .await
                .unwrap_or_else(|err| panic!("committed-only extend_index({name}) failed: {err}"));
        }
        for vec_name in ["vec_idx", "vec_rq"] {
            let err = tx2.extend_index(vec_name).await.unwrap_err();
            assert!(
                matches!(err, Error::NotSupported { .. }),
                "expected NotSupported for {vec_name}, got {err}"
            );
            assert!(err.to_string().contains("vector index"), "{err}");
        }
        let extended = tx2.commit().await.unwrap();
        for name in [
            "text_fm",
            "text_minhash",
            "text_idx",
            "tags_list_element_idx",
        ] {
            let segs = extended.load_indices_by_name(name).await.unwrap();
            assert_eq!(segs.len(), 2, "index {name} should have 2 segments");
            let mut covered = RoaringBitmap::new();
            for seg in &segs {
                covered |= seg.fragment_bitmap.as_ref().unwrap();
            }
            assert_eq!(covered, RoaringBitmap::from_iter([0, 1]));
        }

        let mut fts_scan = extended.scan();
        fts_scan
            .full_text_search(FullTextSearchQuery::new("zebra".to_string()))
            .unwrap();
        let fts_batch = fts_scan.try_into_batch().await.unwrap();
        assert_eq!(fts_batch.num_rows(), 1);

        let mut tx3 = extended.begin_transaction().await.unwrap();
        tx3.append(vec![make_batch(&[20, 21])]).await.unwrap();
        for name in [
            "text_fm",
            "text_minhash",
            "text_idx",
            "tags_list_element_idx",
            "vec_idx",
            "vec_rq",
        ] {
            let err = tx3.extend_index(name).await.unwrap_err();
            assert!(
                matches!(err, Error::NotSupported { .. }),
                "expected NotSupported for local extend_index({name}), got {err}"
            );
        }
    }

    #[tokio::test]
    async fn test_staged_extend_index_rejects_mixed_result_domain() {
        fn copy_dir(source: &std::path::Path, destination: &std::path::Path) {
            std::fs::create_dir_all(destination).unwrap();
            for entry in std::fs::read_dir(source).unwrap() {
                let entry = entry.unwrap();
                let dest = destination.join(entry.file_name());
                if entry.file_type().unwrap().is_dir() {
                    copy_dir(&entry.path(), &dest);
                } else {
                    std::fs::copy(entry.path(), dest).unwrap();
                }
            }
        }

        let dir = TempStrDir::default();
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/v14.0.0-beta.15/stable_row_id_btree.lance");
        copy_dir(&source, std::path::Path::new(dir.as_str()));

        let dataset = Dataset::open(dir.as_str()).await.unwrap();
        let original = dataset.load_indices_by_name("value_idx").await.unwrap();
        assert!(
            original
                .iter()
                .all(|segment| !segment.results_are_row_addrs())
        );

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![
            record_batch!(("value", Int32, [Some(20), None])).unwrap(),
        ])
        .await
        .unwrap();
        let err = tx.extend_index("value_idx").await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(
            err.to_string()
                .contains("mixes row-ID and row-address segments"),
            "{err}"
        );

        // Failed extend_index leaves the staged append intact and existing
        // indexed queries still work after committing the append.
        let committed = tx.commit().await.unwrap();
        let mut scan = committed.scan();
        scan.filter("value >= 0").unwrap();
        assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 15);
    }

    #[rstest]
    #[tokio::test]
    async fn test_staged_add_bases_with_append_and_scan_publishes_together(
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let existing = TempStrDir::default();
        let first = TempStrDir::default();
        let second = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                initial_bases: Some(vec![BasePath::new(
                    7,
                    existing.to_string(),
                    Some("existing".into()),
                    true,
                )]),
                ..Default::default()
            })
            .execute(vec![id_batch([0, 1])])
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            first.to_string(),
            Some("first".into()),
            true,
        )])
        .unwrap();
        tx.append_with_params(
            vec![id_batch([2, 3])],
            &WriteParams {
                target_base_names_or_paths: Some(vec!["first".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();
        let early_reader = tx.scan().unwrap();

        // Register an unnamed second base and append a stream targeting it by path.
        tx.add_bases(vec![BasePath::new(0, second.to_string(), None, true)])
            .unwrap();
        let stream_batch = id_batch([4, 5]);
        let stream = RecordBatchStreamAdapter::new(
            stream_batch.schema(),
            futures::stream::iter(vec![Ok(stream_batch)]),
        );
        tx.append_stream_with_params(
            Box::pin(stream) as SendableRecordBatchStream,
            &WriteParams {
                target_base_names_or_paths: Some(vec![second.to_string()]),
                ..append_params()
            },
        )
        .await
        .unwrap();

        // Append into the pre-existing committed base as well.
        tx.append_with_params(
            vec![id_batch([6])],
            &WriteParams {
                target_bases: Some(vec![7]),
                ..append_params()
            },
        )
        .await
        .unwrap();

        assert_eq!(early_reader.try_into_batch().await.unwrap().num_rows(), 4);
        assert_eq!(
            ids(&tx.staged_dataset().unwrap()).await,
            [0, 1, 2, 3, 4, 5, 6]
        );

        let committed = tx.commit().await.unwrap();
        assert_eq!(committed.version_id(), dataset.version_id() + 1);
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(reopened.manifest.base_paths.len(), 3);
        let first_id = reopened
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("first"))
            .unwrap()
            .id;
        let second_id = reopened
            .manifest
            .base_paths
            .values()
            .find(|b| b.path == second.as_str())
            .unwrap()
            .id;
        let base_ids: Vec<Option<u32>> = reopened
            .manifest
            .fragments
            .iter()
            .map(|f| f.files[0].base_id)
            .collect();
        assert_eq!(base_ids, [None, Some(first_id), Some(second_id), Some(7)]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_staged_add_bases_with_append_and_index_commits_and_queries(
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let secondary = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![id_batch([0, 1])])
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            secondary.to_string(),
            Some("secondary".into()),
            true,
        )])
        .unwrap();
        tx.append_with_params(
            vec![id_batch([10, 11, 12])],
            &WriteParams {
                target_base_names_or_paths: Some(vec!["secondary".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();
        tx.append(vec![id_batch([20, 21])]).await.unwrap();
        tx.create_index(
            &["id"],
            IndexType::BTree,
            Some("by_id".into()),
            &lance_index::scalar::ScalarIndexParams::default(),
        )
        .await
        .unwrap();

        // Staged scan sees both the secondary base and primary rows through the staged index.
        let staged_filtered = tx
            .scan()
            .unwrap()
            .filter("id >= 10")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(staged_filtered.num_rows(), 5);

        let committed = tx.commit().await.unwrap();
        assert_eq!(committed.version_id(), dataset.version_id() + 1);
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1, 10, 11, 12, 20, 21]);
        let secondary_id = reopened
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("secondary"))
            .unwrap()
            .id;
        assert_eq!(
            reopened.manifest.fragments[1].files[0].base_id,
            Some(secondary_id)
        );
        assert_eq!(reopened.manifest.fragments[2].files[0].base_id, None);
        assert_eq!(
            reopened
                .count_indexed_rows("by_id", "id >= 10", None, None)
                .await
                .unwrap(),
            5
        );
        let mut scan = reopened.scan();
        scan.filter("id = 11 OR id = 21").unwrap();
        let rows = scan.try_into_batch().await.unwrap();
        let mut got = rows["id"].as_primitive::<Int32Type>().values().to_vec();
        got.sort_unstable();
        assert_eq!(got, [11, 21]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_staged_add_bases_with_append_and_index_rebases_over_concurrent_writes(
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let our_base = TempStrDir::default();
        let racer_base = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![id_batch([0, 1])])
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            our_base.to_string(),
            Some("ours".into()),
            true,
        )])
        .unwrap();
        tx.append_with_params(
            vec![id_batch([10, 11, 12])],
            &WriteParams {
                target_base_names_or_paths: Some(vec!["ours".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();
        tx.create_index(
            &["id"],
            IndexType::BTree,
            Some("by_id".into()),
            &lance_index::scalar::ScalarIndexParams::default(),
        )
        .await
        .unwrap();

        // A concurrent transaction registers a disjoint base and appends into it.
        let mut racer = dataset.begin_transaction().await.unwrap();
        racer
            .add_bases(vec![BasePath::new(
                0,
                racer_base.to_string(),
                Some("theirs".into()),
                true,
            )])
            .unwrap();
        racer
            .append_with_params(
                vec![id_batch([2, 3])],
                &WriteParams {
                    target_base_names_or_paths: Some(vec!["theirs".into()]),
                    ..append_params()
                },
            )
            .await
            .unwrap();
        let after_racer = racer.commit().await.unwrap();
        let racer_base_id = after_racer
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("theirs"))
            .unwrap()
            .id;

        // Our transaction rebases both its base_id and its fragment_id + index aliases.
        let committed = tx.commit().await.unwrap();
        assert_eq!(committed.version_id(), after_racer.version_id() + 1);
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1, 2, 3, 10, 11, 12]);
        let our_base_id = reopened
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("ours"))
            .unwrap()
            .id;
        assert_ne!(our_base_id, racer_base_id);
        assert_eq!(reopened.manifest.fragments.len(), 3);
        assert_eq!(
            reopened.manifest.fragments[1].files[0].base_id,
            Some(racer_base_id)
        );
        assert_eq!(
            reopened.manifest.fragments[2].files[0].base_id,
            Some(our_base_id)
        );
        let committed_frag_id = reopened.manifest.fragments[2].id as u32;
        assert_eq!(committed_frag_id, 2);
        let indices = load_all_indices(&reopened).await.unwrap();
        let by_id: Vec<_> = indices.iter().filter(|idx| idx.name == "by_id").collect();
        assert_eq!(by_id.len(), 2);
        let local_seg = by_id
            .iter()
            .find(|idx| !idx.local_fragment_aliases.is_empty())
            .expect("local segment");
        assert_eq!(
            local_seg
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![committed_frag_id]
        );
        assert_eq!(
            reopened
                .count_indexed_rows("by_id", "id >= 10", None, Some(&[committed_frag_id]))
                .await
                .unwrap(),
            3
        );
    }

    #[rstest]
    #[case::same_name(true)]
    #[case::same_path(false)]
    #[tokio::test]
    async fn test_staged_add_bases_conflicts_with_concurrent_same_name_or_path(
        #[case] collide_name: bool,
    ) {
        let dir = TempStrDir::default();
        let path_a = TempStrDir::default();
        let path_b = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![id_batch([0])])
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            path_a.to_string(),
            Some("shared".into()),
            true,
        )])
        .unwrap();
        tx.append_with_params(
            vec![id_batch([1])],
            &WriteParams {
                target_base_names_or_paths: Some(vec!["shared".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();

        let mut racer = dataset.begin_transaction().await.unwrap();
        let racer_base = if collide_name {
            BasePath::new(0, path_b.to_string(), Some("shared".into()), true)
        } else {
            BasePath::new(0, path_a.to_string(), Some("other".into()), true)
        };
        racer.add_bases(vec![racer_base]).unwrap();
        racer.commit().await.unwrap();

        let error = tx.commit().await.unwrap_err();
        assert!(
            matches!(
                error,
                Error::RetryableCommitConflict { .. } | Error::CommitConflict { .. }
            ),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::nonzero_id(5, "fresh", true, "id 0")]
    #[case::duplicate_committed_name(0, "existing", true, "already exists")]
    #[case::duplicate_committed_path(0, "fresh", false, "already exists")]
    #[tokio::test]
    async fn test_staged_add_bases_validation_preserves_previous_snapshot(
        #[case] base_id: u32,
        #[case] name: &str,
        #[case] fresh_path: bool,
        #[case] expected_msg: &str,
    ) {
        let dir = TempStrDir::default();
        let existing = TempStrDir::default();
        let fresh = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                initial_bases: Some(vec![BasePath::new(
                    7,
                    existing.to_string(),
                    Some("existing".into()),
                    true,
                )]),
                ..Default::default()
            })
            .execute(vec![id_batch([0])])
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.append(vec![id_batch([1])]).await.unwrap();
        let before = tx.staged_dataset().unwrap();

        let path = if fresh_path {
            fresh.to_string()
        } else {
            existing.to_string()
        };
        let error = tx
            .add_bases(vec![BasePath::new(base_id, path, Some(name.into()), true)])
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(expected_msg), "{error}");
        assert!(Arc::ptr_eq(&tx.staged_dataset().unwrap(), &before));

        let committed = tx.commit().await.unwrap();
        assert_eq!(ids(&committed).await, [0, 1]);
        assert_eq!(committed.manifest.base_paths.len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_staged_add_bases_with_index_survives_compaction_and_optimize(
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let secondary = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![id_batch([0, 1])])
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            secondary.to_string(),
            Some("secondary".into()),
            true,
        )])
        .unwrap();
        tx.append_with_params(
            vec![id_batch([10, 11, 12])],
            &WriteParams {
                target_base_names_or_paths: Some(vec!["secondary".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();
        tx.create_index(
            &["id"],
            IndexType::BTree,
            Some("by_id".into()),
            &lance_index::scalar::ScalarIndexParams::default(),
        )
        .await
        .unwrap();

        let mut committed = tx.commit().await.unwrap();
        compact_files(&mut committed, merge_small_fragments(), None)
            .await
            .unwrap();
        committed
            .optimize_indices(&lance_index::optimize::OptimizeOptions::merge(2))
            .await
            .unwrap();

        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1, 10, 11, 12]);
        let indices = load_all_indices(&reopened).await.unwrap();
        let by_id: Vec<_> = indices.iter().filter(|idx| idx.name == "by_id").collect();
        assert_eq!(by_id.len(), 1);
        assert!(by_id[0].local_fragment_aliases.is_empty());
        assert_eq!(
            reopened
                .count_indexed_rows("by_id", "id >= 10", None, None)
                .await
                .unwrap(),
            3
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_staged_add_bases_with_extend_index_and_concurrent_rebase(
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let our_base = TempStrDir::default();
        let racer_base = TempStrDir::default();
        let mut dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![id_batch([0, 1])])
            .await
            .unwrap();
        let params = ScalarIndexParams::default();
        dataset
            .create_index(&["id"], IndexType::BTree, None, &params, false)
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            our_base.to_string(),
            Some("ours".into()),
            true,
        )])
        .unwrap();
        tx.append_with_params(
            vec![id_batch([10, 11, 12])],
            &WriteParams {
                target_base_names_or_paths: Some(vec!["ours".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();
        tx.extend_index("id_idx").await.unwrap();

        // Racer commits a disjoint base and fragment first so our base_id and
        // fragment_id both shift on retry.
        let mut racer = dataset.begin_transaction().await.unwrap();
        racer
            .add_bases(vec![BasePath::new(
                0,
                racer_base.to_string(),
                Some("theirs".into()),
                true,
            )])
            .unwrap();
        racer
            .append_with_params(
                vec![id_batch([2, 3])],
                &WriteParams {
                    target_base_names_or_paths: Some(vec!["theirs".into()]),
                    ..append_params()
                },
            )
            .await
            .unwrap();
        racer.commit().await.unwrap();

        tx.commit().await.unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1, 2, 3, 10, 11, 12]);
        let our_base_id = reopened
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("ours"))
            .unwrap()
            .id;
        assert_eq!(
            reopened.manifest.fragments[2].files[0].base_id,
            Some(our_base_id)
        );
        let segs = reopened.load_indices_by_name("id_idx").await.unwrap();
        assert_eq!(segs.len(), 2);
        let local_seg = segs
            .iter()
            .find(|s| !s.local_fragment_aliases.is_empty())
            .expect("extended local segment");
        assert_eq!(local_seg.local_fragment_aliases.get(&0), Some(&2));
        assert_eq!(
            reopened
                .count_indexed_rows("id_idx", "id >= 10", None, Some(&[2]))
                .await
                .unwrap(),
            3
        );
    }

    #[tokio::test]
    async fn test_staged_add_bases_alone_target_all_duplicate_and_abort() {
        let dir = TempStrDir::default();
        let base_a = TempStrDir::default();
        let base_b = TempStrDir::default();
        let base_c = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .execute(vec![id_batch([0])])
            .await
            .unwrap();

        // Empty add_bases is a no-op and does not create a new version.
        let mut empty_tx = dataset.begin_transaction().await.unwrap();
        empty_tx.add_bases(Vec::new()).unwrap();
        let unchanged = empty_tx.commit().await.unwrap();
        assert_eq!(unchanged.version_id(), dataset.version_id());

        // Abort drops staged bases without modifying the dataset.
        let mut abort_tx = dataset.begin_transaction().await.unwrap();
        abort_tx
            .add_bases(vec![BasePath::new(
                0,
                base_a.to_string(),
                Some("aborted".into()),
                true,
            )])
            .unwrap();
        abort_tx.abort();
        assert!(
            Dataset::open(dir.as_str())
                .await
                .unwrap()
                .manifest
                .base_paths
                .is_empty()
        );

        // Base-only commit persists the base with no appends or indexes.
        let mut base_only_tx = dataset.begin_transaction().await.unwrap();
        base_only_tx
            .add_bases(vec![BasePath::new(
                0,
                base_a.to_string(),
                Some("base_a".into()),
                true,
            )])
            .unwrap();
        let after_base_only = base_only_tx.commit().await.unwrap();
        assert_eq!(after_base_only.version_id(), dataset.version_id() + 1);
        assert_eq!(after_base_only.manifest.base_paths.len(), 1);

        // Duplicate name or path across two add_bases calls in the same tx fails
        // while keeping the first base usable with target_all_bases.
        let mut tx = after_base_only.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            base_b.to_string(),
            Some("base_b".into()),
            true,
        )])
        .unwrap();
        let dup_name = tx
            .add_bases(vec![BasePath::new(
                0,
                base_c.to_string(),
                Some("base_b".into()),
                true,
            )])
            .unwrap_err();
        assert!(matches!(dup_name, Error::InvalidInput { .. }), "{dup_name}");
        let dup_path = tx
            .add_bases(vec![BasePath::new(
                0,
                base_b.to_string(),
                Some("base_c".into()),
                true,
            )])
            .unwrap_err();
        assert!(matches!(dup_path, Error::InvalidInput { .. }), "{dup_path}");

        tx.append_with_params(
            vec![id_batch([1, 2, 3, 4])],
            &WriteParams {
                max_rows_per_file: 2,
                target_all_bases: Some(false),
                ..append_params()
            },
        )
        .await
        .unwrap();
        let committed = tx.commit().await.unwrap();
        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1, 2, 3, 4]);
        assert_eq!(reopened.manifest.base_paths.len(), 2);
        let base_ids: std::collections::BTreeSet<u32> = reopened.manifest.fragments[1..]
            .iter()
            .map(|f| f.files[0].base_id.expect("expected secondary base_id"))
            .collect();
        let registered_ids: std::collections::BTreeSet<u32> =
            reopened.manifest.base_paths.keys().copied().collect();
        assert!(
            base_ids.is_subset(&registered_ids) && !base_ids.is_empty(),
            "target_all_bases=Some(false) should route fragments to registered bases"
        );
        assert_eq!(committed.version_id(), after_base_only.version_id() + 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_staged_add_bases_batch_and_concurrent_rebase(
        #[values(false, true)] stable: bool,
    ) {
        let dir = TempStrDir::default();
        let base_a = TempStrDir::default();
        let base_b = TempStrDir::default();
        let racer_base = TempStrDir::default();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            })
            .execute(vec![id_batch([0])])
            .await
            .unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![
            BasePath::new(0, base_a.to_string(), Some("base_a".into()), true),
            BasePath::new(0, base_b.to_string(), Some("base_b".into()), true),
        ])
        .unwrap();
        tx.append_with_params(
            vec![id_batch([1, 2])],
            &WriteParams {
                target_base_names_or_paths: Some(vec!["base_a".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();
        tx.append_with_params(
            vec![id_batch([3, 4])],
            &WriteParams {
                target_base_names_or_paths: Some(vec!["base_b".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();

        let mut racer = dataset.begin_transaction().await.unwrap();
        racer
            .add_bases(vec![BasePath::new(
                0,
                racer_base.to_string(),
                Some("racer".into()),
                true,
            )])
            .unwrap();
        racer
            .append_with_params(
                vec![id_batch([5])],
                &WriteParams {
                    target_base_names_or_paths: Some(vec!["racer".into()]),
                    ..append_params()
                },
            )
            .await
            .unwrap();
        let after_racer = racer.commit().await.unwrap();
        let racer_id = after_racer
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("racer"))
            .unwrap()
            .id;

        let committed = tx.commit().await.unwrap();
        assert_eq!(committed.version_id(), after_racer.version_id() + 1);

        let reopened = Dataset::open(dir.as_str()).await.unwrap();
        assert_eq!(ids(&reopened).await, [0, 1, 2, 3, 4, 5]);
        assert_eq!(reopened.manifest.base_paths.len(), 3);
        let a_id = reopened
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("base_a"))
            .unwrap()
            .id;
        let b_id = reopened
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("base_b"))
            .unwrap()
            .id;
        assert_ne!(a_id, b_id);
        assert_ne!(a_id, racer_id);
        assert_ne!(b_id, racer_id);
        let base_ids: Vec<Option<u32>> = reopened
            .manifest
            .fragments
            .iter()
            .map(|f| f.files[0].base_id)
            .collect();
        assert_eq!(base_ids, [None, Some(racer_id), Some(a_id), Some(b_id)]);
    }

    #[rstest]
    #[case::uncontended(false)]
    #[case::concurrent_conflict(true)]
    #[tokio::test]
    async fn test_staged_add_bases_with_managed_blob_v2(#[case] concurrent: bool) {
        use crate::blob::{BlobArrayBuilder, blob_field};
        use crate::dataset::blob::test_util::managed_blob_schema;
        use lance_arrow::{
            ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
            BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME,
        };
        use lance_core::datatypes::BlobKind;

        let dir = TempStrDir::default();
        let staged_base = TempStrDir::default();
        let racer_base = TempStrDir::default();
        let field = blob_field("blob", true).with_metadata(HashMap::from([
            (ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string()),
            (
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
                "8".to_string(),
            ),
            (
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
                "24".to_string(),
            ),
        ]));
        let schema = Arc::new(managed_blob_schema(vec![field]));

        let mut init_blobs = BlobArrayBuilder::new(1);
        init_blobs.push_bytes(b"seed").unwrap();
        let init_batch =
            RecordBatch::try_new(schema.clone(), vec![init_blobs.finish().unwrap()]).unwrap();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            })
            .execute(vec![init_batch])
            .await
            .unwrap();

        let payload = b"staged managed blob payload value";
        let mut blobs = BlobArrayBuilder::new(1);
        blobs.push_bytes(payload).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            staged_base.to_string(),
            Some("blobs".into()),
            true,
        )])
        .unwrap();
        tx.append_with_params(
            vec![batch],
            &WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                target_base_names_or_paths: Some(vec!["blobs".into()]),
                ..append_params()
            },
        )
        .await
        .unwrap();

        if concurrent {
            let mut racer = dataset.begin_transaction().await.unwrap();
            racer
                .add_bases(vec![BasePath::new(
                    0,
                    racer_base.to_string(),
                    Some("racer".into()),
                    true,
                )])
                .unwrap();
            racer.commit().await.unwrap();

            let err = tx.commit().await.unwrap_err();
            assert!(
                matches!(err, Error::RetryableCommitConflict { .. }),
                "{err:?}"
            );
            return;
        }

        tx.commit().await.unwrap();
        let reopened = Arc::new(Dataset::open(dir.as_str()).await.unwrap());
        let base_id = reopened
            .manifest
            .base_paths
            .values()
            .find(|b| b.name.as_deref() == Some("blobs"))
            .unwrap()
            .id;
        assert_eq!(
            reopened.manifest.fragments[1].files[0].base_id,
            Some(base_id)
        );
        let values = reopened
            .take_blobs_by_indices(&[0, 1], "blob")
            .await
            .unwrap();
        assert_eq!(values[1].as_ref().unwrap().kind(), BlobKind::Managed);
        assert_eq!(
            values[1].as_ref().unwrap().read().await.unwrap().as_ref(),
            payload
        );
    }

    #[rstest]
    #[case::uncontended(false)]
    #[case::concurrent_conflict(true)]
    #[tokio::test]
    async fn test_staged_add_bases_with_external_blob_v2_in_primary_file(#[case] concurrent: bool) {
        use crate::blob::{BlobArrayBuilder, blob_field};
        use arrow_schema::Schema as ArrowSchema;
        use lance_core::datatypes::BlobKind;

        let dir = TempStrDir::default();
        let ext_base = lance_core::utils::tempfile::TempDir::default();
        let racer_base = TempStrDir::default();
        let payload = b"staged external blob payload value";
        let ext_file = ext_base.std_path().join("payload.bin");
        std::fs::write(&ext_file, payload).unwrap();
        let ext_base_uri = format!("file://{}", ext_base.std_path().display());
        let ext_file_uri = format!("file://{}", ext_file.display());

        let schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
        let mut init_blobs = BlobArrayBuilder::new(1);
        init_blobs.push_bytes(b"seed").unwrap();
        let init_batch =
            RecordBatch::try_new(schema.clone(), vec![init_blobs.finish().unwrap()]).unwrap();
        let dataset = InsertBuilder::new(dir.as_str())
            .with_params(&WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            })
            .execute(vec![init_batch])
            .await
            .unwrap();

        let mut blobs = BlobArrayBuilder::new(1);
        blobs.push_uri(ext_file_uri).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();

        let mut tx = dataset.begin_transaction().await.unwrap();
        tx.add_bases(vec![BasePath::new(
            0,
            ext_base_uri,
            Some("ext".into()),
            false,
        )])
        .unwrap();
        // Write into the primary data file while the external descriptor points
        // to the staged base.
        tx.append_with_params(
            vec![batch],
            &WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..append_params()
            },
        )
        .await
        .unwrap();

        if concurrent {
            let mut racer = dataset.begin_transaction().await.unwrap();
            racer
                .add_bases(vec![BasePath::new(
                    0,
                    racer_base.to_string(),
                    Some("racer".into()),
                    true,
                )])
                .unwrap();
            racer.commit().await.unwrap();

            let err = tx.commit().await.unwrap_err();
            assert!(
                matches!(err, Error::RetryableCommitConflict { .. }),
                "{err:?}"
            );
            return;
        }

        tx.commit().await.unwrap();
        let reopened = Arc::new(Dataset::open(dir.as_str()).await.unwrap());
        assert_eq!(reopened.manifest.fragments[1].files[0].base_id, None);
        let values = reopened
            .take_blobs_by_indices(&[0, 1], "blob")
            .await
            .unwrap();
        assert_eq!(values[1].as_ref().unwrap().kind(), BlobKind::External);
        assert_eq!(
            values[1].as_ref().unwrap().read().await.unwrap().as_ref(),
            payload
        );
    }
}
