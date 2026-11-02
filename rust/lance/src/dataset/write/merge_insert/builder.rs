// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{
    MergeInsertBuilder, MergeInsertJob, MergeInsertParams, MergeInsertWriteMode, PlanFileVersion,
    SourceDedupeBehavior, WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
};
use crate::dataset::write::merge_insert::job::merge_insert_execution_options;
use crate::{
    Dataset,
    dataset::{ExternalBlobMode, WriteMode, WriteParams, versions},
};
use lance_core::{Error, Result};
use lance_datafusion::exec::new_session_context;
use lance_file::version::LanceFileVersion;
use lance_index::mem_wal::CompactedSsTable;
use std::{sync::Arc, time::Duration};

impl MergeInsertBuilder {
    /// Creates a new builder
    ///
    /// By default this will build a job that has the same semantics as find-or-create
    ///  - matching rows will be kept as-is
    ///  - new rows in the new data will be inserted
    ///  - rows in the old data that do not match will be left as-is
    ///
    /// Use the methods on this builder to customize that behavior
    pub fn try_new(dataset: Arc<Dataset>, on: Vec<String>) -> Result<Self> {
        // Determine the join keys to use. If `on` is empty, fall back to the
        // schema's unenforced primary key (if configured).
        let resolved_on = if on.is_empty() {
            let schema = dataset.schema();
            let pk_fields = schema.unenforced_primary_key();

            if pk_fields.is_empty() {
                return Err(Error::invalid_input(
                    "A merge insert operation requires join keys: specify `on` columns explicitly or configure a primary key in the dataset schema",
                ));
            }

            pk_fields
                .iter()
                .map(|field| schema.field_path(field.id))
                .collect::<Result<Vec<_>>>()?
        } else {
            // Resolve column names using case-insensitive matching to handle
            // lowercased column names from SQL parsing or user input
            on.iter()
                .map(|col| {
                    dataset
                        .schema()
                        .field_case_insensitive(col)
                        .map(|f| f.name.clone())
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "Merge insert key column '{}' does not exist in schema",
                                col
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?
        };

        Ok(Self {
            dataset,
            params: MergeInsertParams {
                on: resolved_on,
                when_matched: WhenMatched::DoNothing,
                insert_not_matched: true,
                delete_not_matched_by_source: WhenNotMatchedBySource::Keep,
                conflict_retries: 10,
                spill_for_retry: true,
                retry_timeout: Duration::from_secs(30),
                compacted_sstables: Vec::new(),
                mem_wal_dir_id: None,
                skip_auto_cleanup: false,
                use_index: true,
                write_mode: MergeInsertWriteMode::default(),
                source_dedupe_behavior: SourceDedupeBehavior::Fail,
                commit_retries: None,
                target_bases: None,
                target_base_names_or_paths: None,
                target_all_bases: None,
                data_storage_version: None,
            },
            write_params: Arc::new(WriteParams::default()),
        })
    }

    /// Specify what should happen when a target row matches a row in the source
    pub fn when_matched(&mut self, behavior: WhenMatched) -> &mut Self {
        self.params.when_matched = behavior;
        self
    }

    /// Specify what should happen when a source row has no match in the target
    ///
    /// These are typically "new rows"
    pub fn when_not_matched(&mut self, behavior: WhenNotMatched) -> &mut Self {
        self.params.insert_not_matched = match behavior {
            WhenNotMatched::DoNothing => false,
            WhenNotMatched::InsertAll => true,
        };
        self
    }

    /// Specify what should happen when a target row has no match in the source
    ///
    /// These are typically "old rows"
    pub fn when_not_matched_by_source(&mut self, behavior: WhenNotMatchedBySource) -> &mut Self {
        self.params.delete_not_matched_by_source = behavior;
        self
    }

    /// Set number of times to retry the operation if there is contention.
    ///
    /// If this is set > 0, then the operation will keep a copy of the input data
    /// either in memory or on disk (depending on the size of the data) and will
    /// retry the operation if there is contention.
    ///
    /// Default is 10.
    pub fn conflict_retries(&mut self, retries: u32) -> &mut Self {
        self.params.conflict_retries = retries;
        self
    }

    /// Controls whether a one-shot stream source is spilled so it can be replayed
    /// across retries.
    ///
    /// When the source is a one-shot stream (e.g. [`MergeInsertJob::execute`]) and
    /// `conflict_retries > 0`, the source is buffered in memory and spilled to disk
    /// so each retry can re-read it. Set this to `false` to skip that buffering and
    /// fail fast with a contention error instead of writing the stream to disk.
    ///
    /// This has no effect on re-scannable sources (materialized batches via
    /// [`MergeInsertJob::execute_batches`], or a [`TableProvider`] via
    /// [`MergeInsertJob::execute_provider`]). Arbitrary providers may still use
    /// bounded replay within an indexed attempt to preserve one logical snapshot.
    ///
    /// Default is true.
    ///
    /// [`TableProvider`]: datafusion::catalog::TableProvider
    pub fn spill_for_retry(&mut self, spill: bool) -> &mut Self {
        self.params.spill_for_retry = spill;
        self
    }

    /// Set the timeout used to limit retries.
    ///
    /// This is the maximum time to spend on the operation before giving up. At
    /// least one attempt will be made, regardless of how long it takes to complete.
    /// Subsequent attempts will be cancelled once this timeout is reached. If
    /// the timeout has been reached during the first attempt, the operation
    /// will be cancelled immediately.
    ///
    /// The default is 30 seconds.
    pub fn retry_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.params.retry_timeout = timeout;
        self
    }

    pub fn skip_auto_cleanup(&mut self, skip: bool) -> &mut Self {
        self.params.skip_auto_cleanup = skip;
        self
    }

    /// Controls whether to use indices for the merge operation.
    ///
    /// When set to false, forces a full table scan even if an index exists on the join key.
    /// This can be useful for benchmarking or when the optimizer chooses a suboptimal path.
    ///
    /// Default is true (use index if available).
    pub fn use_index(&mut self, use_index: bool) -> &mut Self {
        self.params.use_index = use_index;
        self
    }

    /// Selects how the merged rows are written to disk.
    ///
    /// For a partial-schema update the two modes have different cost shapes:
    /// [`MergeInsertWriteMode::RewriteColumns`] never reads or writes the columns
    /// the source omits, but its replacement column file covers every row of each
    /// fragment it touches, so the bytes written barely fall as fewer rows match.
    /// [`MergeInsertWriteMode::RewriteRows`] instead scales with the matched rows.
    /// Patching columns wins once the fraction of rows matched exceeds roughly the
    /// fraction of each row's bytes the source columns occupy: for a KB-scale
    /// update of a MB-per-row table that is nearly always, and for a table whose
    /// columns are all narrow it may never be.
    ///
    /// The per-fragment matched row count that decides this is only known once
    /// the join has run, so the caller picks rather than the planner guessing.
    ///
    /// Default is [`MergeInsertWriteMode::Auto`], which rewrites whole rows.
    pub fn write_mode(&mut self, mode: MergeInsertWriteMode) -> &mut Self {
        self.params.write_mode = mode;
        self
    }

    /// Specify how to handle duplicate source rows.
    ///
    /// Default is `Fail`, which errors when multiple source rows match one target row.
    /// Use `FirstSeen` to keep the first encountered row for each non-null join key,
    /// including unmatched keys that will be inserted, and skip subsequent rows.
    /// Join keys containing NULL are not deduplicated because merge insert uses SQL
    /// equality, where NULL does not equal NULL.
    ///
    /// If the source contains duplicates and `FirstSeen` behavior doesn't match your needs,
    /// sort the source data before passing it to the merge insert operation.
    pub fn source_dedupe_behavior(&mut self, behavior: SourceDedupeBehavior) -> &mut Self {
        self.params.source_dedupe_behavior = behavior;
        self
    }

    /// Mark MemWAL SSTables as compacted when this commit succeeds.
    ///
    /// This updates `compacted_sstables` in the MemWAL index atomically with
    /// the data commit.
    ///
    /// **For multi-pass compaction, call this only on the final successful
    /// data-changing pass.** Intermediate passes must not carry compaction
    /// progress. Lance cannot tell whether a caller has another pass planned,
    /// so it cannot enforce this: if a delete pass carried the marker and the
    /// process then died before the matching upsert, the recorded generation
    /// would claim rows were copied in that never were.
    ///
    /// `dir_id` names the MemWAL the SSTables belong to; the commit fails with
    /// [`Error::StaleMemWal`] if it was since disabled or replaced. Pass an
    /// empty `sstables` on passes that record no progress so they are checked too.
    pub fn mark_sstables_as_compacted(
        &mut self,
        dir_id: impl Into<String>,
        sstables: Vec<CompactedSsTable>,
    ) -> &mut Self {
        self.params.mem_wal_dir_id = Some(dir_id.into());
        self.params.compacted_sstables.extend(sstables);
        self
    }

    /// Set the number of inner commit retries for manifest version conflicts.
    /// Different from `conflict_retries` which handles semantic conflicts.
    /// Default: 20
    pub fn commit_retries(&mut self, retries: u32) -> &mut Self {
        self.params.commit_retries = Some(retries);
        self
    }

    /// Set the exact V2 data file version for rows written by this merge.
    ///
    /// If omitted, the dataset's default write version is used. The default
    /// remains unchanged. Targets cannot cross the V1/V2 boundary.
    ///
    /// ```
    /// # use lance::{Dataset, Result, dataset::MergeInsertBuilder};
    /// # use lance_file::version::LanceFileVersion;
    /// # use std::sync::Arc;
    /// # fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])?
    ///     .data_storage_version(LanceFileVersion::V2_2)
    ///     .try_build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn data_storage_version(&mut self, version: LanceFileVersion) -> &mut Self {
        self.params.data_storage_version = Some(PlanFileVersion(version.resolve()));
        self
    }

    /// Set the write parameters for the data files this merge insert writes,
    /// as [`InsertBuilder::with_params`](crate::dataset::InsertBuilder::with_params)
    /// does for an insert.
    ///
    /// Every path that writes rows applies them: file size limits, file writer
    /// options, store parameters, progress, the Blob pack size and external
    /// Blob handling. With `allow_external_blob_outside_bases`, an external
    /// Blob URI the source supplies outside the dataset's registered bases is
    /// stored as an absolute reference instead of being refused.
    ///
    /// The file version and the bases new files go to are set on this builder
    /// ([`Self::data_storage_version`], [`Self::target_bases`] and its
    /// siblings), so [`Self::try_build`] refuses parameters that set them.
    /// `mode` must stay at its default, `WriteMode::Create`: merge insert
    /// decides how its files are written, and another mode would change the
    /// schema they are written with (`Overwrite`) or start index seeds the
    /// merge does not commit (`Append`). `try_build` also refuses
    /// `ExternalBlobMode::Ingest`, which merge insert does not support. Fields
    /// that only shape a commit or a new dataset, such as
    /// `transaction_properties` or `enable_stable_row_ids`, do not apply.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::{MergeInsertBuilder, WhenMatched, WriteParams};
    /// # async fn test(dataset: Arc<Dataset>) -> Result<()> {
    /// let job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])?
    ///     .when_matched(WhenMatched::UpdateAll)
    ///     .with_params(&WriteParams {
    ///         allow_external_blob_outside_bases: true,
    ///         ..Default::default()
    ///     })
    ///     .try_build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_params(&mut self, params: &WriteParams) -> &mut Self {
        self.write_params = Arc::new(params.clone());
        self
    }

    /// Write new fragments produced by this merge insert to these base IDs.
    ///
    /// New data files are distributed across the target bases round-robin,
    /// the same way a normal write with [`WriteParams::target_bases`] routes
    /// them. The IDs must be registered in the dataset manifest, or
    /// [`crate::dataset::write::PRIMARY_BASE_ID`] (0) to include the dataset's primary
    /// storage in the rotation (e.g. `vec![0, 1, 2]` spreads across primary
    /// plus bases 1 and 2). Data files that patch existing fragments and
    /// deletion files are always written to the dataset's primary storage.
    ///
    /// Cannot be combined with [`Self::target_base_names_or_paths`].
    pub fn target_bases(&mut self, base_ids: Vec<u32>) -> &mut Self {
        self.params.target_bases = Some(base_ids);
        self
    }

    /// Like [`Self::target_bases`], but referencing bases by name or path URI.
    ///
    /// References are resolved against the base paths registered in the
    /// dataset manifest when the merge insert executes. An entry equal to the
    /// dataset's URI includes the dataset's primary storage in the rotation.
    ///
    /// Cannot be combined with [`Self::target_bases`].
    pub fn target_base_names_or_paths(&mut self, refs: Vec<String>) -> &mut Self {
        self.params.target_base_names_or_paths = Some(refs);
        self
    }

    /// Write new fragments produced by this merge insert to every base
    /// registered in the dataset manifest, resolved when the merge executes.
    /// When `include_primary` is true the dataset's primary storage
    /// participates in the rotation as the first slot.
    ///
    /// Cannot be combined with [`Self::target_bases`] or
    /// [`Self::target_base_names_or_paths`].
    pub fn target_all_bases(&mut self, include_primary: bool) -> &mut Self {
        self.params.target_all_bases = Some(include_primary);
        self
    }

    /// Crate a merge insert job
    pub fn try_build(&mut self) -> Result<MergeInsertJob> {
        let write_version = self.params.write_version(&self.dataset);
        versions::validate_write_version(
            self.dataset
                .manifest
                .data_storage_format
                .lance_file_format(),
            write_version,
        )?;
        if !self.params.insert_not_matched
            && self.params.when_matched == WhenMatched::DoNothing
            && self.params.delete_not_matched_by_source == WhenNotMatchedBySource::Keep
        {
            return Err(Error::invalid_input(
                "The merge insert job is not configured to change the data in any way",
            ));
        }
        if self.params.target_bases.is_some() && self.params.target_base_names_or_paths.is_some() {
            return Err(Error::invalid_input(
                "Cannot specify both target_base_names_or_paths and target_bases. Use one or the other.",
            ));
        }
        if self.params.target_all_bases.is_some()
            && (self.params.target_bases.is_some()
                || self.params.target_base_names_or_paths.is_some())
        {
            return Err(Error::invalid_input(
                "Cannot specify target_all_bases together with target_bases or target_base_names_or_paths.",
            ));
        }
        validate_merge_insert_write_params(&self.write_params)?;
        let mut params = self.params.clone();
        params.data_storage_version = Some(PlanFileVersion(write_version));
        let spill_execution_options = merge_insert_execution_options();
        // DataFusion 54.1.0 can retain a rejected spill charge. Keep its manager
        // operation-scoped so any execution error discards that accounting and
        // cannot reduce the quota of a later merge job.
        let spill_session_context = new_session_context(&spill_execution_options);
        Ok(MergeInsertJob {
            dataset: self.dataset.clone(),
            params,
            spill_session_context,
            spill_execution_options,
            write_params: self.write_params.clone(),
        })
    }
}

/// Refuse write parameters a merge insert cannot honor: the write mode, the
/// file version and the target bases are the job's own, and the job has no
/// ingest path for external Blob URIs.
fn validate_merge_insert_write_params(params: &WriteParams) -> Result<()> {
    // The fragment writers prepare their schema and index seeds by mode; merge
    // insert has always written with `Create` and commits that result.
    if !matches!(params.mode, WriteMode::Create) {
        return Err(Error::invalid_input(format!(
            "Merge insert write params set mode={:?}; leave mode at its default, WriteMode::Create, since merge insert decides how its files are written",
            params.mode
        )));
    }
    if let Some(version) = params.data_storage_version {
        return Err(Error::invalid_input(format!(
            "Merge insert write params set data_storage_version={version:?}; use MergeInsertBuilder::data_storage_version instead",
        )));
    }
    if params.target_bases.is_some()
        || params.target_base_names_or_paths.is_some()
        || params.target_all_bases.is_some()
        || params.initial_bases.is_some()
    {
        return Err(Error::invalid_input(format!(
            "Merge insert write params set target_bases={:?}, target_base_names_or_paths={:?}, target_all_bases={:?} and initial_bases={}; route new files with MergeInsertBuilder::target_bases, target_base_names_or_paths or target_all_bases instead",
            params.target_bases,
            params.target_base_names_or_paths,
            params.target_all_bases,
            params.initial_bases.as_ref().map_or(0, Vec::len),
        )));
    }
    if params.external_blob_mode == ExternalBlobMode::Ingest {
        return Err(Error::invalid_input(
            "Merge insert write params set external_blob_mode=\"ingest\", which merge insert does not support",
        ));
    }
    Ok(())
}
