// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::num::NonZeroU32;

use arrow_array::RecordBatch;
use futures::StreamExt;
use lance_core::datatypes::{OnMissing, OnTypeMismatch};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result, datatypes::Schema};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::{DataFile, Fragment};
use lance_table::utils::stream::ReadBatchFutStream;

use super::Dataset;
use super::fragment::FragmentReader;
use super::scanner::get_default_batch_size;
use super::versions;
use super::write::{GenericWriter, cleanup_data_fragments};
use crate::dataset::FileFragment;

mod blanks;
use blanks::{BlankPlan, add_blanks, add_blanks_with_plans, blank_plans, compact_blank_source};

/// Update or insert a new column.
///
/// To use, call [`Updater::next`] to get the next [`RecordBatch`] as input,
/// then call [`Updater::update`] to update the batch. Repeat until
/// [`Updater::next`] returns `None`.
///
/// `write_schema` dictates the schema of the new file, while `final_schema` is
/// the schema of the full fragment after the update. These are optional and if
/// not specified, the updater will infer the write schema from the first batch
/// of results and will append them to the current schema to get the final schema.
pub struct Updater {
    fragment: FileFragment,

    /// The reader over the [`Fragment`]
    input_stream: ReadBatchFutStream,

    /// The last batch read from the file, with deleted rows removed
    last_input: Option<RecordBatch>,

    /// A batch discovered by `finish` while checking that the input stream is
    /// exhausted. It has not been handed to the caller yet.
    prefetched_input: Option<RecordBatch>,

    writer: Option<Box<dyn GenericWriter>>,

    /// The final schema of the fragment after the update.
    final_schema: Option<Schema>,

    /// The schema the new files will be written in. This only contains new columns.
    write_schema: Option<Schema>,

    allow_external_blob_outside_bases: bool,

    write_version: ConcreteFileVersion,

    finished: bool,

    deletion_restorer: DeletionRestorer,

    /// A live output row that can supply fixed-width values for later blank-only
    /// batches. This keeps trailing deleted runs bounded without inventing values.
    blank_source: Option<RecordBatch>,
}

impl Updater {
    /// Create a new updater with source reader, and destination writer.
    ///
    /// The `schemas` parameter is a tuple of the write schema (just the new fields)
    /// and the final schema (all the fields).
    ///
    /// If the schemas are not known, they can be None and will be inferred from
    /// the first batch of results.
    pub(super) async fn try_new(
        fragment: FileFragment,
        reader: FragmentReader,
        deletion_vector: DeletionVector,
        schemas: Option<(Schema, Schema)>,
        batch_size: Option<u32>,
        write_version: ConcreteFileVersion,
    ) -> Result<Self> {
        if batch_size == Some(0) {
            return Err(Error::invalid_input(format!(
                "batch_size must be greater than zero, got 0 for fragment {}",
                fragment.id()
            )));
        }
        let (write_schema, final_schema) = if let Some((write_schema, final_schema)) = schemas {
            (Some(write_schema), Some(final_schema))
        } else {
            (None, None)
        };

        let source_version = fragment
            .metadata()
            .referenced_lance_files()
            .next()
            .map(|file| file.file_version())
            .transpose()?
            .unwrap_or_else(|| {
                fragment
                    .dataset()
                    .manifest()
                    .data_storage_format
                    .lance_file_format()
            });
        let legacy_batch_size =
            versions::row_group_size_for_rewrite(source_version, &fragment).await?;

        let batch_size = match (&legacy_batch_size, batch_size) {
            // If this is a v1 dataset we must use the row group size of the file
            (Some(legacy_batch_size), _) => *legacy_batch_size,
            // If this is a v2 dataset, let the user pick the batch size
            (None, Some(user_specified_batch_size)) => user_specified_batch_size,
            // Otherwise, default to 1024 if the user didn't specify anything
            (None, None) => {
                let default_batch_size = get_default_batch_size().unwrap_or(1024);
                u32::try_from(default_batch_size).map_err(|_| {
                    Error::invalid_input(format!(
                        "Fragment Updater: default batch size {default_batch_size} exceeds the u32 row-address limit"
                    ))
                })?
            }
        };
        // Blame whoever owns the zero: for a legacy file the size came from the file's
        // own first row group, so an empty one is a file-integrity problem, not a bad
        // argument.
        let batch_size = NonZeroU32::new(batch_size).ok_or_else(|| {
            if legacy_batch_size.is_some() {
                Error::corrupt_file_named(
                    "row group metadata",
                    format!(
                        "the first row group of fragment {} reports zero rows, so the updater \
                         has no batch size to rewrite it with",
                        fragment.id()
                    ),
                )
            } else {
                Error::invalid_input("Fragment Updater: batch size must be greater than zero")
            }
        })?;

        let input_stream = reader.read_all(batch_size.get()).await?;

        Ok(Self {
            fragment,
            input_stream,
            last_input: None,
            prefetched_input: None,
            writer: None,
            write_schema,
            final_schema,
            allow_external_blob_outside_bases: false,
            write_version,
            finished: false,
            deletion_restorer: DeletionRestorer::new(
                deletion_vector,
                legacy_batch_size,
                batch_size,
                write_version,
            ),
            blank_source: None,
        })
    }

    pub fn fragment(&self) -> &FileFragment {
        &self.fragment
    }

    pub fn dataset(&self) -> &Dataset {
        self.fragment.dataset()
    }

    /// Returns the next [`RecordBatch`] as input for updater.
    ///
    /// Every batch this hands out must be passed back to [`Self::update`] before the
    /// next call: the deletion restorer advances there, so skipping it would leave
    /// deleted rows unaccounted for and fail the stream at its end.
    pub async fn next(&mut self) -> Result<Option<&RecordBatch>> {
        if self.finished {
            return Ok(None);
        }
        if self.last_input.is_some() {
            return Err(Error::invalid_input(
                "Fragment Updater: the previous input batch must be updated before reading the next batch",
            ));
        }
        if let Some(batch) = self.prefetched_input.take() {
            self.last_input = Some(batch);
            return Ok(self.last_input.as_ref());
        }
        let batch = self.input_stream.next().await;
        match batch {
            None => {
                if !self.deletion_restorer.is_exhausted() {
                    // The stream cannot supply rows the restorer still needs. In
                    // practice that means the deletion vector points at rows the
                    // stream never produced — an id past the fragment's physical row
                    // count, or fewer rows read than the fragment claims to have.
                    //
                    // Deferred blanks can also be outstanding here, but only if no
                    // batch after the deferral had a live row, i.e. the whole fragment
                    // is deleted; `write_deletions` drops such a fragment before it
                    // reaches an updater, so that path is defensive. A legacy file
                    // cannot defer at all — its fully deleted batch is refused
                    // earlier, by `add_blanks`.
                    //
                    // Don't name a count: the deletion-vector case owes no blanks yet,
                    // so a number here would read as zero rows owed.
                    return Err(Error::not_supported(format!(
                        "Fragment Updater: the input stream for fragment {} ended while \
                         deleted rows were still unaccounted for, run compaction to \
                         materialize deletions first",
                        self.fragment.id(),
                    )));
                }
                self.finished = true;
                Ok(None)
            }
            Some(batch) => {
                self.last_input = Some(batch.await?);
                Ok(self.last_input.as_ref())
            }
        }
    }

    /// Create a new Writer for new columns.
    ///
    /// After it is called, this Fragment contains the metadata of the new DataFile,
    /// containing the columns, even the data has not written yet.
    ///
    /// It is the caller's responsibility to close the [`FileWriter`].
    ///
    /// Internal use only.
    async fn new_writer(&mut self, schema: Schema) -> Result<Box<dyn GenericWriter>> {
        versions::open_update_writer(
            self.write_version,
            self.dataset(),
            &schema,
            self.allow_external_blob_outside_bases,
        )
        .await
    }

    /// Allow trusted existing external blob references to pass through an update rewrite.
    /// Callers must separately validate any newly supplied references before writing.
    pub(super) fn allow_external_blob_outside_bases(&mut self) {
        self.allow_external_blob_outside_bases = true;
    }

    async fn write_batch(&mut self, batch: RecordBatch) -> Result<()> {
        if self.writer.is_none() {
            let write_schema = self
                .write_schema
                .as_ref()
                .ok_or_else(|| Error::internal("Fragment Updater: missing write schema"))?
                .clone();
            self.writer = Some(self.new_writer(write_schema).await?);
        }

        self.writer
            .as_mut()
            .ok_or_else(|| Error::internal("Fragment Updater: missing writer"))?
            .write(&[batch])
            .await
    }

    /// Update one batch.
    pub async fn update(&mut self, batch: RecordBatch) -> Result<()> {
        let Some(last) = self.last_input.as_ref() else {
            return Err(Error::invalid_input(
                "Fragment Updater: no input data is available before update".to_string(),
            ));
        };

        if last.num_rows() != batch.num_rows() {
            return Err(Error::invalid_input(format!(
                "Fragment Updater: new batch has different size with the source batch: {} != {}",
                last.num_rows(),
                batch.num_rows()
            )));
        };

        if self.write_schema.is_none() {
            // Infer from the caller's batch even when it has no rows. A leading
            // deleted run may not produce anything writable until a later batch.
            let output_schema = batch.schema();
            let mut final_schema = self.fragment.schema().merge(output_schema.as_ref())?;
            final_schema.set_field_id(Some(self.fragment.dataset().manifest.max_field_id()));
            final_schema.validate()?;
            let write_schema = final_schema.project_by_schema(
                output_schema.as_ref(),
                OnMissing::Error,
                OnTypeMismatch::Error,
            )?;
            self.final_schema = Some(final_schema);
            self.write_schema = Some(write_schema);
        }

        let current_blank_source = (batch.num_rows() > 0).then(|| batch.slice(0, 1));
        let write_schema = self
            .write_schema
            .as_ref()
            .ok_or_else(|| Error::internal("Fragment Updater: missing write schema"))?
            .clone();

        // Add back deletions that belong to this input batch. Blanks deferred by
        // earlier empty batches are emitted separately below so each write stays
        // bounded by the updater's chosen batch size.
        let restored = self.deletion_restorer.restore(batch, Some(&write_schema))?;

        // A slice would keep all of the source batch's buffers alive until this
        // fragment finishes. Materialize the single row only when a later batch can
        // still need it. The temporary slice above lives only for this update.
        // Legacy files never defer, so they never need a source at all.
        let retained_blank_source = if self.deletion_restorer.can_defer()
            && !self.deletion_restorer.is_exhausted()
            && self.blank_source.is_none()
        {
            current_blank_source
                .as_ref()
                .map(|source| {
                    compact_blank_source(
                        source,
                        self.deletion_restorer.storage_version,
                        Some(&write_schema),
                    )
                })
                .transpose()?
        } else {
            None
        };
        let blank_source = retained_blank_source
            .as_ref()
            .or(self.blank_source.as_ref())
            .cloned();

        if let Some(blank_source) = blank_source
            .as_ref()
            .filter(|_| self.deletion_restorer.has_pending_blanks())
        {
            let plans = blank_plans(
                blank_source,
                self.deletion_restorer.storage_version,
                Some(&write_schema),
            )?;
            while let Some(blanks) = self
                .deletion_restorer
                .take_pending_blanks_with_plans(blank_source, &plans)?
            {
                self.write_batch(blanks).await?;
            }
        }

        if let Some(restored) = restored {
            self.write_batch(restored).await?;
        }
        if self.deletion_restorer.is_exhausted() {
            self.blank_source = None;
        } else if retained_blank_source.is_some() {
            self.blank_source = retained_blank_source;
        }
        // `last_input` is the state token pairing one successful `next` with one
        // successful `update`. Clearing it rejects duplicate updates and lets the
        // next input batch advance the restorer exactly once.
        self.last_input = None;

        Ok(())
    }

    /// Finish updating this fragment, and returns the updated [`Fragment`].
    pub async fn finish(&mut self) -> Result<Fragment> {
        if self.last_input.is_some() {
            return Err(Error::invalid_input(
                "Fragment Updater: cannot finish while the last input batch has not been updated",
            ));
        }
        // Probe the stream before blaming the deletion vector. A caller that simply
        // stopped polling also leaves deleted rows unaccounted for, and telling it to run
        // compaction would be the wrong remedy; which of the two conditions it hits
        // otherwise depends only on where the deletions happen to fall. Some callers know
        // their update fits in one batch and call `finish` without polling `next` once
        // more, so preserve that usage rather than requiring a terminal `next`. Keep a
        // batch found by the probe so the error is recoverable and repeated `finish`
        // calls cannot discard the stream one batch at a time.
        if !self.finished {
            if self.prefetched_input.is_none()
                && let Some(batch) = self.input_stream.next().await
            {
                self.prefetched_input = Some(batch.await?);
            }
            if self.prefetched_input.is_some() {
                return Err(Error::invalid_input(
                    "Fragment Updater: cannot finish while unread input batches remain; call next and update until the input stream ends",
                ));
            }
        }

        if !self.deletion_restorer.is_exhausted() {
            // The stream really is exhausted and blanks are still owed, which is the same
            // condition -- and the same remedy -- as the check in [`Self::next`].
            return Err(Error::not_supported(format!(
                "Fragment Updater: cannot finish fragment {} while deleted rows are still \
                 unaccounted for, run compaction to materialize deletions first",
                self.fragment.id(),
            )));
        }

        // Every data file in a fragment must cover the same physical rows, and the
        // restorer has counted exactly the rows handed to the writer. Compare the two
        // rather than trust the deletion vector to have been well formed: an id at or
        // past the fragment's last row would otherwise be restored as if it named a
        // real row, and the file would be committed one row long.
        //
        // The guards above already reject every way a caller can end up short, so this
        // one is only reachable from a deletion vector that names rows the fragment does
        // not have -- state `Dataset::delete` cannot produce and only a corrupt or
        // hand-written deletion file can. It therefore has no end-to-end test by
        // construction; `test_restore_deletes_overshoots_when_deletion_vector_runs_one_past_the_end`
        // pins the miscount it detects at the restorer level.
        //
        // Only the metadata count that `FileFragment::physical_rows` itself trusts is
        // usable here. Datasets written before the writer version existed could record
        // a wrong `physical_rows`, and the reader ignores it for exactly that reason, so
        // comparing against it would reject sound updates. Those fragments keep the
        // weaker checks above.
        if self.dataset().manifest.writer_version.is_some()
            && let Some(physical_rows) = self.fragment.metadata.physical_rows
        {
            let restored = self.deletion_restorer.current_row_id as usize;
            if restored != physical_rows {
                return Err(Error::internal(format!(
                    "Fragment Updater: fragment {} has {physical_rows} physical rows but the \
                     update produced {restored}; the fragment's deletion vector may name rows \
                     it does not have",
                    self.fragment.id(),
                )));
            }
        }

        // Only now is the stream known to be both exhausted and fully accounted for.
        // Marking it finished any earlier would make a following `next` report a clean end
        // of stream while blanks were still owed.
        self.finished = true;

        if let Some(writer) = self.writer.as_mut() {
            let (_, data_file) = writer.finish().await?;
            self.fragment.metadata.files.push(data_file);
        }
        // A finished writer is no longer an unfinished resource to clean up, and
        // dropping it makes repeated `finish` calls idempotent instead of appending
        // the same data file twice.
        self.writer = None;

        Ok(self.fragment.metadata().clone())
    }

    /// Clean up any data file and blob sidecars created by the current unfinished writer.
    pub(super) async fn cleanup_unfinished_writer(&mut self) {
        let Some(writer) = self.writer.take() else {
            return;
        };
        let (path, base_id) = writer.data_file_path();
        let path = path.to_string();
        drop(writer);

        if path.is_empty() {
            return;
        }

        let mut fragment = Fragment::new(self.fragment.id() as u64);
        // cleanup_data_fragments only needs path/base_id to remove the unfinished
        // data file and any blob sidecars. Build a minimal synthetic fragment so
        // we can reuse the shared cleanup path without fabricating full metadata.
        fragment.files.push(DataFile::new(
            path,
            vec![],
            vec![],
            self.write_version,
            None,
            base_id,
        ));
        cleanup_data_fragments(
            &self.dataset().object_store,
            &self.dataset().base,
            None,
            &[fragment],
        )
        .await;
    }

    /// Get the final schema of the fragment after the update.
    ///
    /// This may be None if the schema is not known. This can happen if it was
    /// not specified up front and the first batch of results has not yet been
    /// processed.
    pub fn schema(&self) -> Option<&Schema> {
        self.final_schema.as_ref()
    }
}

/// Restores deleted rows.
///
/// All data files in a fragment must have the same # of rows (including deleted rows)
/// When we run the update process the next/update methods don't actually calculate on
/// deleted rows.  This means the updated batches will have fewer rows than the original
/// data files.  This struct restores the deleted rows, inserting arbitrary values into the
/// batches where the deleted rows should be.
///
/// To do this we scan through the deletion vector in sorted order, merging deleted rows
/// in as appropriate.
///
/// Any method returning an error leaves the restorer mid-batch: the deletion vector
/// has been walked past rows that never made it into an output batch. Drop it and
/// start over rather than calling it again.
struct DeletionRestorer {
    current_row_id: u32,

    /// Number of rows in each batch, only used in legacy files for validation
    legacy_batch_size: Option<u32>,

    /// Maximum number of rows materialized in one output batch.
    max_output_rows: NonZeroU32,

    deletion_vector_iter: Option<Box<dyn Iterator<Item = u32> + Send>>,

    last_deleted_row_id: Option<u32>,

    /// Blank rows owed to batches that had no live row to copy a placeholder from
    ///
    /// See [`Self::restore`] for why they are deferred instead of materialized.
    /// Only ever non-zero for non-legacy files, which are the only ones that defer.
    pending_blank_rows: u32,

    /// The file version blanks will be written in, which decides whether a column
    /// can take a null blank. See [`blank_plan`].
    storage_version: ConcreteFileVersion,
}

impl DeletionRestorer {
    fn new(
        deletion_vector: DeletionVector,
        legacy_batch_size: Option<u32>,
        max_output_rows: NonZeroU32,
        storage_version: ConcreteFileVersion,
    ) -> Self {
        let deletion_vector_iter =
            (!deletion_vector.is_empty()).then(|| deletion_vector.into_sorted_iter());
        Self {
            current_row_id: 0,
            legacy_batch_size,
            max_output_rows,
            deletion_vector_iter,
            last_deleted_row_id: None,
            pending_blank_rows: 0,
            storage_version,
        }
    }

    fn is_exhausted(&self) -> bool {
        self.deletion_vector_iter.is_none() && self.pending_blank_rows == 0
    }

    fn has_pending_blanks(&self) -> bool {
        self.pending_blank_rows != 0
    }

    /// Whether blanks owed by a batch with no live row can be deferred to a later one.
    ///
    /// Legacy files cannot: their output batches have to reproduce the original row
    /// group size, which moving rows between batches would break.
    fn can_defer(&self) -> bool {
        self.legacy_batch_size.is_none()
    }

    /// Given a batch of `num_rows`, walk through the deletion vector, and figure out where blanks
    /// should be inserted.
    ///
    /// For example, if self.current_row_id is 10 and the deletion vector is [11, 12, 19, 25] and
    /// num_rows is 7 then this function will at least return [1, 2] and the batch will at least
    /// span row ids 10..18.
    ///
    /// Then, in the example we need to choose whether the returned batch should include
    /// row 19 (and have 10 rows) or not (and have 9 rows).  This is only a concern in v1 files
    /// where we want to match the original row group size (which is the batch size).  If the
    /// batch size is 9 then we do not include 19 and return as above.
    ///
    /// If the maximum output size is 10 then we include 19 and return [1, 2, 9].
    ///
    /// In v2 files we stop at the configured output batch size. Any remaining deleted rows
    /// are handled by later input batches or emitted as bounded blank-only batches.
    fn deleted_batch_offsets_in_range(&mut self, mut num_rows: u32) -> Result<Vec<u32>> {
        let mut deleted = Vec::new();
        let first_row_id = self.current_row_id;
        let max_output_rows = self.max_output_rows.get();
        debug_assert!(num_rows <= max_output_rows);
        // The last row id (exclusive) in the batch
        let mut last_row_id = first_row_id.checked_add(num_rows).ok_or_else(|| {
            Error::internal(format!(
                "Fragment Updater: row range overflow for first row {first_row_id} and {num_rows} live rows"
            ))
        })?;
        // If there are zero deleted rows then the range covered will be first_row_id..last_row_id
        let Some(deletion_vector_iter) = self.deletion_vector_iter.as_mut() else {
            return Ok(deleted);
        };

        // Now we need to walk through our deletion vector and figure out where to insert blanks
        // Take the stashed id rather than peeking at it: leaving a consumed id in the
        // field relies on the early return above to never read it again. `or_else` has
        // to stay lazy — `or` would pull from the iterator even when a stash is waiting,
        // silently dropping a deleted row.
        let mut next_deleted_id = self
            .last_deleted_row_id
            .take()
            .or_else(|| deletion_vector_iter.next());
        loop {
            if let Some(next_deleted_id) = next_deleted_id {
                if next_deleted_id > last_row_id
                    || (next_deleted_id == last_row_id && num_rows == max_output_rows)
                {
                    // Either the next deleted id is out of range or it is the next row but
                    // we are full.  Either way, stash it and return
                    self.last_deleted_row_id = Some(next_deleted_id);
                    return Ok(deleted);
                }
                // Otherwise, the deleted row is in range, and we have space in our batch
                // and so we include it
                deleted.push(next_deleted_id.checked_sub(first_row_id).ok_or_else(|| {
                    Error::internal(format!(
                        "Fragment Updater: deletion row id {next_deleted_id} is before the current row {first_row_id}"
                    ))
                })?);
                last_row_id = last_row_id.checked_add(1).ok_or_else(|| {
                    Error::internal("Fragment Updater: deleted row range overflow")
                })?;
                num_rows = num_rows.checked_add(1).ok_or_else(|| {
                    Error::internal("Fragment Updater: restored batch row count overflow")
                })?;
            } else {
                // Deleted row ids iterator is exhausted
                self.deletion_vector_iter = None;
                // `is_exhausted` reads these two together, so a stash left behind here
                // would make it report exhaustion while a deleted row is still owed.
                debug_assert!(self.last_deleted_row_id.is_none());
                return Ok(deleted);
            }
            next_deleted_id = deletion_vector_iter.next();
        }
    }

    /// Restore the deleted rows for one batch of live rows.
    ///
    /// A blank is the cheapest value its column can hold (see [`add_blanks`]), and
    /// for a column that has no such value it is a copy of the batch's first live
    /// row, so a batch with no live rows may have nothing to copy from. That happens
    /// when a deleted run starts at physical row 0: there is no preceding batch for
    /// [`Self::deleted_batch_offsets_in_range`] to append the run to, so the run
    /// arrives as an empty batch carrying every one of its offsets.
    ///
    /// Rather than invent placeholder values for an arbitrary schema, we remember
    /// how many blanks we owe. [`Self::take_pending_blanks_with_plans`] later
    /// materializes them in bounded batches using a live row as the source for
    /// fixed-width values.
    fn restore(
        &mut self,
        batch: RecordBatch,
        write_schema: Option<&Schema>,
    ) -> Result<Option<RecordBatch>> {
        // Holds by construction today — deferring is the only thing that sets
        // pending_blank_rows and it is gated on non-legacy — so this documents the
        // invariant the legacy row-count check below depends on rather than guarding
        // against a state we can reach.
        debug_assert!(self.pending_blank_rows == 0 || self.can_defer());

        // Because of deleted rows, the number of row ids in the batch might not
        // match the length.
        let num_live_rows = u32::try_from(batch.num_rows()).map_err(|_| {
            Error::internal(format!(
                "Fragment Updater: input batch has {} rows, exceeding the u32 row-address limit",
                batch.num_rows()
            ))
        })?;
        let deleted_batch_offsets = self.deleted_batch_offsets_in_range(num_live_rows)?;

        // Legacy files must reproduce the original row group size, which deferring
        // would break, so they keep reporting the pre-existing error instead.
        if batch.num_rows() == 0 && self.can_defer() {
            let deferred = u32::try_from(deleted_batch_offsets.len()).map_err(|_| {
                Error::internal(format!(
                    "Fragment Updater: {} deferred blanks exceed the u32 row-address limit",
                    deleted_batch_offsets.len()
                ))
            })?;
            self.pending_blank_rows =
                self.pending_blank_rows
                    .checked_add(deferred)
                    .ok_or_else(|| {
                        Error::internal("Fragment Updater: pending blank row count overflow")
                    })?;
            self.current_row_id = self
                .current_row_id
                .checked_add(deferred)
                .ok_or_else(|| Error::internal("Fragment Updater: restored row id overflow"))?;
            return Ok(None);
        }

        let batch = add_blanks(
            batch,
            &deleted_batch_offsets,
            self.storage_version,
            write_schema,
        )?;

        if let Some(batch_size) = self.legacy_batch_size {
            // validation just in case, when the input has a fixed batch size then the
            // output should have the same fixed batch size (except the last batch)
            let is_last = self.is_exhausted();
            if batch.num_rows() != batch_size as usize && !is_last {
                return Err(Error::internal(format!(
                    "Fragment Updater: batch size mismatch: {} != {}",
                    batch.num_rows(),
                    batch_size
                )));
            }
        }

        let restored_rows = u32::try_from(batch.num_rows()).map_err(|_| {
            Error::internal(format!(
                "Fragment Updater: restored batch has {} rows, exceeding the u32 row-address limit",
                batch.num_rows()
            ))
        })?;
        self.current_row_id = self
            .current_row_id
            .checked_add(restored_rows)
            .ok_or_else(|| Error::internal("Fragment Updater: restored row id overflow"))?;
        Ok(Some(batch))
    }

    /// Materialize one bounded chunk of blanks deferred by empty input batches.
    ///
    /// `source` supplies values for layouts that have no cheaper synthetic blank.
    /// `plans` must have been produced by [`blank_plans`] for `source`.
    fn take_pending_blanks_with_plans(
        &mut self,
        source: &RecordBatch,
        plans: &[BlankPlan],
    ) -> Result<Option<RecordBatch>> {
        if self.pending_blank_rows == 0 {
            return Ok(None);
        }
        debug_assert_eq!(source.num_rows(), 1);

        let chunk_size = self.pending_blank_rows.min(self.max_output_rows.get());
        let batch_offsets = (0..chunk_size).collect::<Vec<_>>();
        let with_source = add_blanks_with_plans(source.clone(), &batch_offsets, plans)?;
        // Offsets `0..n` put the blanks first and append the whole source, so exactly
        // one row -- the source itself -- has to come back off the end. Check the
        // coupling rather than only documenting it in `add_blanks`.
        debug_assert_eq!(with_source.num_rows(), chunk_size as usize + 1);
        self.pending_blank_rows -= chunk_size;
        Ok(Some(with_source.slice(0, chunk_size as usize)))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, num::NonZeroU32, sync::Arc};

    use arrow::{array::AsArray, datatypes::Int32Type};
    use arrow_array::{
        Array, ArrayRef, BinaryArray, BinaryViewArray, DictionaryArray, FixedSizeListArray,
        GenericListArray, Int32Array, LargeBinaryArray, LargeStringArray, MapArray,
        OffsetSizeTrait, RecordBatch, StringArray, StringViewArray, StructArray, UInt64Array,
    };
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use lance_arrow::json::{
        ARROW_JSON_EXT_NAME, JSON_EXT_NAME, convert_json_columns, decode_json, encode_json,
    };
    use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
    use lance_core::datatypes::Schema as LanceSchema;
    use lance_datagen::RowCount;
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;

    use super::{
        BlankPlan, DeletionRestorer, Error, add_blanks, blank_plans, compact_blank_source,
    };

    /// For tests that do not care which version they run against: the blank a
    /// nullable column gets is the same for every version that can store nulls.
    const ANY_VERSION: ConcreteFileVersion = ConcreteFileVersion::V2_1;

    fn batch_of(field: Field, array: ArrayRef) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![array]).unwrap()
    }

    fn nonzero(rows: u32) -> NonZeroU32 {
        NonZeroU32::new(rows).unwrap()
    }

    /// Materialize every blank the restorer still owes, planning them from `source`.
    ///
    /// Production code plans once per input batch and drains the debt in chunks; tests
    /// only ever have one chunk in flight, so this keeps them to a single call.
    fn take_pending_blanks(
        restorer: &mut DeletionRestorer,
        source: &RecordBatch,
        write_schema: Option<&LanceSchema>,
    ) -> Option<RecordBatch> {
        let plans = blank_plans(source, restorer.storage_version, write_schema).unwrap();
        restorer
            .take_pending_blanks_with_plans(source, &plans)
            .unwrap()
    }

    /// A field carrying the `arrow.json` extension metadata PyArrow's `pa.json_()` emits.
    fn arrow_json_field(name: &str, data_type: DataType, nullable: bool) -> Field {
        Field::new(name, data_type, nullable).with_metadata(HashMap::from([(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        )]))
    }

    /// A field carrying Lance's internal `lance.json` (JSONB) extension metadata.
    fn lance_json_field(name: &str, nullable: bool) -> Field {
        Field::new(name, DataType::LargeBinary, nullable).with_metadata(HashMap::from([(
            ARROW_EXT_NAME_KEY.to_string(),
            JSON_EXT_NAME.to_string(),
        )]))
    }

    /// Row 20 is the case the two variants disagree on: a legacy file has to stop at
    /// its row group size and leave 20 for the next batch, while a v2 file has room
    /// for it in this one.
    #[rstest]
    #[case::legacy(Some(10), 10)]
    #[case::v2(None, 1024)]
    fn test_restore_deletes(#[case] legacy_batch_size: Option<u32>, #[case] max_output_rows: u32) {
        let mut restorer = super::DeletionRestorer::new(
            vec![11, 12, 19, 20, 25].into_iter().collect(),
            legacy_batch_size,
            nonzero(max_output_rows),
            ANY_VERSION,
        );

        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(10))
            .unwrap();
        // First batch is rows ids 0..9 so nothing is restored
        let restored = restorer.restore(batch.clone(), None).unwrap().unwrap();
        assert_eq!(restored, batch);

        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(7))
            .unwrap();
        // Next batch is rows ids 10..16 so we need to restore 11, 12
        // 19, and maybe 20 (depends on batch size)
        let restored = restorer.restore(batch, None).unwrap().unwrap();
        let values = restored.column(0).as_primitive::<Int32Type>();
        assert_eq!(values.value(0), 0);
        assert_eq!(values.value(1), 0);
        assert_eq!(values.value(2), 0);
        assert_eq!(values.value(3), 1);
        assert_eq!(values.value(4), 2);
        assert_eq!(values.value(5), 3);
        assert_eq!(values.value(6), 4);
        assert_eq!(values.value(7), 5);
        assert_eq!(values.value(8), 6);
        assert_eq!(values.value(9), 0);
        if legacy_batch_size.is_some() {
            assert_eq!(values.len(), 10);
        } else {
            assert_eq!(values.value(10), 0);
            assert_eq!(values.len(), 11);
        }
    }

    /// Rows 0..10 are deleted, so the first read batch has no live rows. A legacy file
    /// cannot defer the blanks and reports the error instead;
    /// [`test_restore_deletes_leading_empty_batch`] covers the v2 side. Row 15 is here
    /// only to keep the deletion vector identical between the two tests — legacy fails
    /// on the first batch, so 15 is never reached.
    #[test]
    fn test_restore_deletes_leading_empty_batch_legacy() {
        let mut restorer = super::DeletionRestorer::new(
            (0..10).chain([15]).collect(),
            Some(10),
            nonzero(10),
            ANY_VERSION,
        );

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();

        // Assert the source, not just is_err: the batch-size check further down
        // returns Internal, and the two are different failures.
        let err = restorer.restore(empty, None).unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
    }

    /// The v2 side of the same deletion vector: blanks owed by a batch with no live
    /// row are deferred to a later batch that has one to copy.
    #[test]
    fn test_restore_deletes_leading_empty_batch() {
        let mut restorer = super::DeletionRestorer::new(
            (0..10).chain([15]).collect(),
            None,
            nonzero(10),
            ANY_VERSION,
        );

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();

        // Nothing is written for the fully deleted batch itself.
        assert!(restorer.restore(empty.clone(), None).unwrap().is_none());
        assert!(!restorer.is_exhausted());

        // A second empty batch must carry the debt through untouched: row 15 is
        // out of its range, so it defers nothing of its own.
        let restored = restorer.restore(empty, None).unwrap();
        assert!(restored.is_none());
        assert!(!restorer.is_exhausted());

        // The next batch supplies a source for the 10 deferred blanks, while row 15
        // contributes one new blank after its five live rows. That last one pins the
        // offset calculation after the deferred rows were already counted.
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();
        let blank_source = batch.slice(0, 1);
        let restored = restorer.restore(batch, None).unwrap().unwrap();

        let pending = take_pending_blanks(&mut restorer, &blank_source, None).unwrap();
        assert_eq!(pending.num_rows(), 10);
        let pending_values = pending.column(0).as_primitive::<Int32Type>();
        for value in pending_values.values() {
            assert_eq!(*value, 0);
        }

        assert_eq!(restored.num_rows(), 6);
        let values = restored.column(0).as_primitive::<Int32Type>();
        for i in 0..5 {
            assert_eq!(values.value(i), i as i32);
        }
        assert_eq!(values.value(5), 0);
        assert!(restorer.is_exhausted());
    }

    /// The debt itself has to keep the restorer from reporting exhaustion, not just
    /// the deletion vector. With no row past the deleted run the iterator empties on
    /// the first call, so only `pending_blank_rows` can hold `is_exhausted` back —
    /// and it must, or `Updater::next` would accept a data file short by ten rows.
    #[test]
    fn test_restore_deletes_owes_blanks_after_vector_drains() {
        let mut restorer =
            super::DeletionRestorer::new((0..10).collect(), None, nonzero(10), ANY_VERSION);

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();

        assert!(restorer.restore(empty, None).unwrap().is_none());
        assert!(!restorer.is_exhausted());

        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();
        let blank_source = batch.slice(0, 1);
        assert_eq!(
            restorer.restore(batch, None).unwrap().unwrap().num_rows(),
            5
        );
        assert_eq!(
            take_pending_blanks(&mut restorer, &blank_source, None)
                .unwrap()
                .num_rows(),
            10
        );
        assert!(restorer.is_exhausted());
    }

    #[test]
    fn test_restore_deletes_chunks_pending_blanks() {
        let mut restorer =
            super::DeletionRestorer::new((0..12).collect(), None, nonzero(4), ANY_VERSION);
        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();

        for _ in 0..3 {
            assert!(restorer.restore(empty.clone(), None).unwrap().is_none());
        }
        assert!(!restorer.is_exhausted());

        let source = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(1))
            .unwrap();
        for _ in 0..3 {
            let blanks = take_pending_blanks(&mut restorer, &source, None).unwrap();
            assert_eq!(blanks.num_rows(), 4);
        }
        assert!(take_pending_blanks(&mut restorer, &source, None).is_none());
        assert!(restorer.is_exhausted());
    }

    #[test]
    fn test_restore_deletes_caps_trailing_run() {
        let mut restorer =
            super::DeletionRestorer::new((5..25).collect(), None, nonzero(10), ANY_VERSION);
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();
        let source = batch.slice(0, 1);

        let restored = restorer.restore(batch, None).unwrap().unwrap();
        assert_eq!(restored.num_rows(), 10);

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();
        assert!(restorer.restore(empty.clone(), None).unwrap().is_none());
        assert_eq!(
            take_pending_blanks(&mut restorer, &source, None)
                .unwrap()
                .num_rows(),
            10
        );
        assert!(restorer.restore(empty, None).unwrap().is_none());
        assert_eq!(
            take_pending_blanks(&mut restorer, &source, None)
                .unwrap()
                .num_rows(),
            5
        );
        assert!(restorer.is_exhausted());
    }

    /// A deletion vector naming the row one past the fragment's last row is consumed as
    /// if it were real, so the restorer reports exhaustion having counted one row more
    /// than the fragment has. [`Updater::finish`] compares its count against
    /// `physical_rows` precisely because `is_exhausted` cannot see this.
    #[test]
    fn test_restore_deletes_overshoots_when_deletion_vector_runs_one_past_the_end() {
        let mut restorer = super::DeletionRestorer::new(
            [25].into_iter().collect(),
            None,
            nonzero(10),
            ANY_VERSION,
        );

        let mut written = 0;
        for rows in [10u64, 10, 5] {
            let batch = lance_datagen::gen_batch()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(rows))
                .unwrap();
            written += restorer.restore(batch, None).unwrap().unwrap().num_rows();
        }

        // 25 physical rows went in and 26 came out: row id 25 became a blank of its own.
        assert_eq!(written, 26);
        assert_eq!(restorer.current_row_id, 26);
        assert!(
            restorer.is_exhausted(),
            "the miscount is invisible to is_exhausted, which is why finish() counts rows"
        );
    }

    /// A deletion vector naming a row the fragment does not have leaves the restorer
    /// unexhausted with no blanks owed: the id stays stashed, so the iterator is never
    /// drained. `Updater::next` relies on this to refuse rather than write a data file
    /// missing that row, and the error must not claim a blank count for it.
    #[test]
    fn test_restore_deletes_not_exhausted_when_deletion_vector_overruns() {
        let mut restorer = super::DeletionRestorer::new(
            [100].into_iter().collect(),
            None,
            nonzero(1024),
            ANY_VERSION,
        );

        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        // Row 100 is past this batch, so it is stashed rather than consumed and
        // nothing is restored. No blanks are owed either — which is why the error in
        // `Updater::next` cannot name a count.
        assert_eq!(
            restorer.restore(batch, None).unwrap().unwrap().num_rows(),
            5
        );
        assert!(!restorer.is_exhausted());
    }

    /// Deferred blanks are counted into `current_row_id` when they are deferred, so
    /// consuming them must not count them again. A later deleted row is what makes
    /// the double count observable: it lands at the wrong offset once the restorer
    /// thinks the fragment is further along than it is.
    #[test]
    fn test_restore_deletes_does_not_double_count_deferred_blanks() {
        let mut restorer = super::DeletionRestorer::new(
            (0..10).chain([22]).collect(),
            None,
            nonzero(10),
            ANY_VERSION,
        );

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();
        assert!(restorer.restore(empty, None).unwrap().is_none());

        // Rows 10..20 are live, so this batch pays off the ten blanks and nothing
        // else: row 22 is past its range and stays stashed.
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(10))
            .unwrap();
        let blank_source = batch.slice(0, 1);
        assert_eq!(
            restorer.restore(batch, None).unwrap().unwrap().num_rows(),
            10
        );
        assert_eq!(
            take_pending_blanks(&mut restorer, &blank_source, None)
                .unwrap()
                .num_rows(),
            10
        );
        assert!(!restorer.is_exhausted());

        // Row 22 falls inside this batch's range, but only if current_row_id sits at
        // 20. Counting the deferred blanks twice would have pushed it to 30, putting
        // row 22 behind the batch and dropping its blank.
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();
        let restored = restorer.restore(batch, None).unwrap().unwrap();

        // Physical rows 20..25 arrive with row 22 deleted, so the blank lands third.
        assert_eq!(restored.num_rows(), 6);
        let values = restored.column(0).as_primitive::<Int32Type>();
        assert_eq!(values.value(0), 0);
        assert_eq!(values.value(1), 1);
        assert_eq!(values.value(2), 0);
        for i in 2..5 {
            assert_eq!(values.value(1 + i), i as i32);
        }
        assert!(restorer.is_exhausted());
    }

    #[test]
    fn test_add_blanks() {
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(10))
            .unwrap();

        let with_blanks = add_blanks(batch.clone(), &[5, 7], ANY_VERSION, None).unwrap();

        assert_eq!(with_blanks.num_rows(), 12);
        let values = with_blanks.column(0).as_primitive::<Int32Type>();
        for i in 0..5 {
            assert_eq!(values.value(i), i as i32);
        }
        assert_eq!(values.value(5), 0);
        assert_eq!(values.value(6), 5);
        assert_eq!(values.value(7), 0);
        for i in 8..12 {
            assert_eq!(values.value(i), (i - 2) as i32);
        }

        let with_blanks = add_blanks(batch, &[0, 11], ANY_VERSION, None).unwrap();
        let values = with_blanks.column(0).as_primitive::<Int32Type>();
        assert_eq!(values.value(0), 0);
        for i in 1..11 {
            assert_eq!(values.value(i), (i - 1) as i32);
        }
        assert_eq!(values.value(11), 0);
    }

    /// The ways a caller can hand `add_blanks` offsets it cannot satisfy. The
    /// message keyword matters as much as the variant: most of these return
    /// `Internal`, so matching only the variant would let one check stand in for
    /// the other.
    #[rstest]
    #[case::empty_batch(0, &[0, 1, 2], "missing too many rows in merge")]
    #[case::non_increasing(5, &[3, 1], "strictly increasing")]
    #[case::equal_offsets(5, &[1, 1], "strictly increasing")]
    #[case::past_end(5, &[100], "more live rows before it, but")]
    // Rejected at the second offset with only three live rows left, so this is the
    // only case that exercises the `- batch_pos` term: without it the remaining
    // count reads as five and this offset slips through.
    #[case::past_end_after_live_rows(5, &[2, 7], "more live rows before it, but")]
    fn test_add_blanks_rejects_invalid_offsets(
        #[case] num_rows: u64,
        #[case] batch_offsets: &[u32],
        #[case] expected_message: &str,
    ) {
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(num_rows))
            .unwrap();

        let err = add_blanks(batch, batch_offsets, ANY_VERSION, None).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(expected_message),
            "expected {expected_message:?} in {message:?}"
        );
    }

    /// An offset equal to the batch length is the trailing-deletion shape: every
    /// live row comes first, then the blanks. It has to be accepted, which is what
    /// pins the bounds check to `>` rather than `>=`.
    #[test]
    fn test_add_blanks_at_end_of_batch() {
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        let with_blanks = add_blanks(batch, &[5], ANY_VERSION, None).unwrap();

        assert_eq!(with_blanks.num_rows(), 6);
        let values = with_blanks.column(0).as_primitive::<Int32Type>();
        for i in 0..5 {
            assert_eq!(values.value(i), i as i32);
        }
        assert_eq!(values.value(5), 0);
    }

    /// The retained source must not pin the payload of the batch it came from. A
    /// variable-width column is never read out of it -- blanks use the filler -- so it
    /// is replaced outright rather than merely narrowed to row zero.
    #[test]
    fn test_compact_blank_source_releases_unused_payload() {
        let large_value = vec![7_u8; 8192];
        let batch = batch_of(
            Field::new("payload", DataType::Binary, false),
            Arc::new(BinaryArray::from(vec![&b"x"[..], large_value.as_slice()])),
        );

        let source = compact_blank_source(&batch.slice(0, 1), ANY_VERSION, None).unwrap();
        let payload = source.column(0).as_binary::<i32>();
        assert_eq!(payload.len(), 1);
        assert_eq!(payload.value(0), b"");
        assert_eq!(payload.value_data().len(), 0);
    }

    /// Fixed-width columns are the ones blanks actually copy from, so their row zero
    /// has to survive compaction. Dictionaries survive too, but only the values their
    /// one remaining key references.
    #[test]
    fn test_compact_blank_source_keeps_values_blanks_copy() {
        let dictionary = DictionaryArray::<Int32Type>::new(
            Int32Array::from(vec![1, 0]),
            Arc::new(StringArray::from(vec!["alpha", "beta"])),
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("n", DataType::Int32, false),
                Field::new("d", dictionary.data_type().clone(), false),
            ])),
            vec![Arc::new(Int32Array::from(vec![7, 8])), Arc::new(dictionary)],
        )
        .unwrap();

        let source = compact_blank_source(&batch.slice(0, 1), ANY_VERSION, None).unwrap();
        assert_eq!(source.column(0).as_primitive::<Int32Type>().value(0), 7);
        let compacted = source.column(1).as_dictionary::<Int32Type>();
        assert_eq!(compacted.values().len(), 1);
        assert_eq!(compacted.values().as_string::<i32>().value(0), "beta");
    }

    #[test]
    fn test_dictionary_blanks_share_values() {
        let dictionary_values: ArrayRef = Arc::new(StringArray::from(vec!["alpha", "beta"]));
        let dictionary = DictionaryArray::<Int32Type>::new(
            Int32Array::from(vec![0, 1]),
            dictionary_values.clone(),
        );
        let batch = batch_of(
            Field::new("d", dictionary.data_type().clone(), false),
            Arc::new(dictionary),
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let values = with_blanks.column(0).as_dictionary::<Int32Type>();
        assert_eq!(values.keys().values(), &[0, 0, 1]);
        assert_eq!(values.values().len(), 2);
        assert!(Arc::ptr_eq(values.values(), &dictionary_values));
    }

    /// Both list widths go through one generic offset walk, so each instantiation
    /// needs its own coverage.
    /// Both list widths go through one generic offset walk, so each instantiation needs
    /// its own coverage. `nullable` picks which blank the column gets: a null entry when
    /// it may hold one, an empty entry when it may not. Either way the child array is
    /// reused untouched -- that is the whole point of rebuilding only the offsets.
    fn assert_list_blanks_reuse_child_values<O: OffsetSizeTrait>(nullable: bool) {
        let values: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));
        let child = Arc::new(Field::new("item", DataType::Int32, false));
        let lists = GenericListArray::<O>::try_new(
            child,
            OffsetBuffer::from_lengths([2, 1]),
            values.clone(),
            None,
        )
        .unwrap();
        let batch = batch_of(
            Field::new("l", lists.data_type().clone(), nullable),
            Arc::new(lists),
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let lists = with_blanks.column(0).as_list::<O>();
        // The blank is the zero-length entry between the two live ones.
        assert_eq!(lists.offsets().lengths().collect::<Vec<_>>(), vec![2, 0, 1]);
        assert!(Arc::ptr_eq(lists.values(), &values));
        if nullable {
            assert_eq!(lists.null_count(), 1);
            assert!(lists.is_null(1), "a nullable list blank must be null");
        } else {
            assert_eq!(
                lists.null_count(),
                0,
                "a non-nullable list blank must be empty, not null"
            );
        }
    }

    #[rstest]
    #[case::nullable(true)]
    #[case::non_nullable(false)]
    fn test_list_blanks_reuse_child_values(#[case] nullable: bool) {
        assert_list_blanks_reuse_child_values::<i32>(nullable);
    }

    #[rstest]
    #[case::nullable(true)]
    #[case::non_nullable(false)]
    fn test_large_list_blanks_reuse_child_values(#[case] nullable: bool) {
        assert_list_blanks_reuse_child_values::<i64>(nullable);
    }

    #[rstest]
    #[case::nullable(true)]
    #[case::non_nullable(false)]
    fn test_map_blanks_reuse_entries(#[case] nullable: bool) {
        let entry_fields = Fields::from(vec![
            Field::new("keys", DataType::Int32, false),
            Field::new("values", DataType::Int32, false),
        ]);
        let entries = StructArray::new(
            entry_fields.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ],
            None,
        );
        let entries_field = Arc::new(Field::new("entries", DataType::Struct(entry_fields), false));
        let maps = MapArray::try_new(
            entries_field.clone(),
            OffsetBuffer::from_lengths([2, 1]),
            entries.clone(),
            None,
            false,
        )
        .unwrap();
        let batch = batch_of(
            Field::new("m", DataType::Map(entries_field, false), nullable),
            Arc::new(maps),
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let maps = with_blanks.column(0).as_map();
        assert_eq!(maps.value_offsets(), &[0, 2, 2, 3]);
        assert_eq!(maps.entries(), &entries);
        if nullable {
            assert_eq!(maps.null_count(), 1);
            assert!(maps.is_null(1), "a nullable map blank must be null");
        } else {
            assert_eq!(
                maps.null_count(),
                0,
                "a non-nullable map blank must be empty, not null"
            );
        }
    }

    /// A map whose *entries* field is declared nullable violates the Arrow spec, but
    /// Lance's `Field` validation lets it through, so a stored column can have this shape.
    /// The blank path rebuilds with `MapArray::try_new`, which rejects a nullable entries
    /// field, so the plan must fall back to `Take` (the pre-existing behavior) instead of
    /// newly failing with "MapArray entries cannot contain nulls" on a deleted fragment.
    #[test]
    fn test_nullable_map_entries_field_copies_row_zero() {
        let entry_fields = Fields::from(vec![
            Field::new("keys", DataType::Int32, false),
            Field::new("values", DataType::Int32, false),
        ]);
        let entries = StructArray::new(
            entry_fields.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![10, 20, 30])),
            ],
            None,
        );
        // Deliberately nullable entries field. `MapArray::try_new` would reject it, so
        // build the array unchecked the way Lance's readers surface such a column.
        let entries_field = Arc::new(Field::new("entries", DataType::Struct(entry_fields), true));
        let map_type = DataType::Map(entries_field, false);
        let offsets = OffsetBuffer::<i32>::from_lengths([2, 1]);
        // SAFETY: offsets are monotonic and cover the entries; the only spec violation is
        // the nullable entries field, which is exactly the shape under test.
        let map_data = unsafe {
            arrow::array::ArrayData::builder(map_type.clone())
                .len(2)
                .add_buffer(offsets.into_inner().into_inner())
                .add_child_data(entries.to_data())
                .build_unchecked()
        };
        let maps = MapArray::from(map_data);
        let batch = batch_of(Field::new("m", map_type, true), Arc::new(maps));

        let plans = blank_plans(&batch, ANY_VERSION, None).unwrap();
        assert!(
            matches!(plans.as_slice(), [BlankPlan::Take]),
            "a nullable entries field must fall back to Take"
        );

        // Previously this errored with "MapArray entries cannot contain nulls". Take
        // copies row zero into the blank, so its span repeats row zero's.
        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();
        let maps = with_blanks.column(0).as_map();
        assert_eq!(maps.len(), 3);
        assert_eq!(maps.value_offsets(), &[0, 2, 4, 5]);
    }

    #[test]
    fn test_fixed_size_list_blanks_shrink_variable_width_children() {
        let child = Arc::new(Field::new("item", DataType::Binary, false));
        let values: ArrayRef = Arc::new(BinaryArray::from(vec![
            &b"aa"[..],
            &b"bb"[..],
            &b"cc"[..],
            &b"dd"[..],
        ]));
        let lists = FixedSizeListArray::try_new(child.clone(), 2, values, None).unwrap();
        let batch = batch_of(
            Field::new("l", DataType::FixedSizeList(child, 2), true),
            Arc::new(lists),
        );

        let with_blanks = add_blanks(batch, &[1, 3], ANY_VERSION, None).unwrap();

        let lists = with_blanks.column(0).as_fixed_size_list();
        assert_eq!(lists.len(), 4);
        assert_eq!(lists.null_count(), 2);
        assert!(lists.is_null(1));
        assert!(lists.is_null(3));

        let values = lists.values().as_binary::<i32>();
        assert_eq!(values.null_count(), 0);
        assert_eq!(values.value(0), b"aa");
        assert_eq!(values.value(1), b"bb");
        assert_eq!(values.value(2), b"");
        assert_eq!(values.value(3), b"");
        assert_eq!(values.value(4), b"cc");
        assert_eq!(values.value(5), b"dd");
        assert_eq!(values.value(6), b"");
        assert_eq!(values.value(7), b"");
        assert_eq!(values.value_data().len(), 8);
    }

    /// A non-nullable fixed-size list cannot take a null blank, so `blank_is_null` is
    /// false and the blank keeps row zero's validity. Its child may still be nullable,
    /// and then the child slots take nulls rather than empty values.
    #[test]
    fn test_non_nullable_fixed_size_list_blanks_keep_row_zero_validity() {
        let child = Arc::new(Field::new("item", DataType::Binary, true));
        let values: ArrayRef = Arc::new(BinaryArray::from(vec![
            &b"aa"[..],
            &b"bb"[..],
            &b"cc"[..],
            &b"dd"[..],
        ]));
        let lists = FixedSizeListArray::try_new(child.clone(), 2, values, None).unwrap();
        let batch = batch_of(
            Field::new("l", DataType::FixedSizeList(child, 2), false),
            Arc::new(lists),
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let lists = with_blanks.column(0).as_fixed_size_list();
        assert_eq!(lists.len(), 3);
        // Row zero is valid, so the blank that mirrors it is valid too.
        assert_eq!(lists.null_count(), 0);
        let values = lists.values().as_binary::<i32>();
        assert_eq!(values.value(0), b"aa");
        assert_eq!(values.value(1), b"bb");
        assert!(values.is_null(2), "a nullable child blank must be null");
        assert!(values.is_null(3));
        assert_eq!(values.value(4), b"cc");
        assert_eq!(values.value(5), b"dd");
        assert_eq!(values.value_data().len(), 8);
    }

    /// A fixed-size list whose child is fixed width collapses to `Take`, so its blank is
    /// a copy of row zero -- values and validity both. That is not the same as the
    /// `FixedSizeList` plan would produce: it would null the parent instead.
    #[test]
    fn test_fixed_width_fixed_size_list_blanks_copy_row_zero() {
        let child = Arc::new(Field::new("item", DataType::Int32, false));
        let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let lists = FixedSizeListArray::try_new(child.clone(), 2, values, None).unwrap();
        let batch = batch_of(
            Field::new("l", DataType::FixedSizeList(child, 2), true),
            Arc::new(lists),
        );

        let plans = blank_plans(&batch, ANY_VERSION, None).unwrap();
        assert!(
            matches!(plans.as_slice(), [BlankPlan::Take]),
            "a fixed-width child makes the whole column cheapest to take"
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let lists = with_blanks.column(0).as_fixed_size_list();
        assert_eq!(lists.len(), 3);
        assert_eq!(
            lists.null_count(),
            0,
            "the collapse copies row zero rather than nulling the parent"
        );
        assert_eq!(
            lists.values().as_primitive::<Int32Type>().values(),
            &[1, 2, 1, 2, 3, 4]
        );
    }

    /// A fixed-size list of structs is the one shape whose FSL child plan is `Struct`,
    /// which is the only path that runs the child validity loop in
    /// `apply_fixed_size_list_blank_plan`. No built-in Lance type has this shape, so it
    /// is covered here directly rather than through a writer round-trip.
    #[test]
    fn test_fixed_size_list_of_struct_blanks() {
        let struct_fields = Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("n", DataType::Int32, false),
        ]);
        // Two structs per list row, two list rows.
        let child = StructArray::new(
            struct_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            ],
            None,
        );
        let child_field = Arc::new(Field::new("item", DataType::Struct(struct_fields), true));
        let lists =
            FixedSizeListArray::try_new(child_field.clone(), 2, Arc::new(child), None).unwrap();
        let batch = batch_of(
            Field::new("l", DataType::FixedSizeList(child_field, 2), true),
            Arc::new(lists),
        );

        // A variable-width struct member keeps the child plan a `Struct`, so the FSL plan
        // does not collapse to `Take`.
        let plans = blank_plans(&batch, ANY_VERSION, None).unwrap();
        assert!(
            matches!(
                plans.as_slice(),
                [BlankPlan::FixedSizeList { child, .. }]
                    if matches!(child.as_ref(), BlankPlan::Struct(_))
            ),
            "expected FixedSizeList<Struct> to keep a Struct child plan"
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();
        let lists = with_blanks.column(0).as_fixed_size_list();
        assert_eq!(lists.len(), 3);
        assert!(lists.is_valid(0));
        assert!(lists.is_null(1), "a nullable FSL blank must be null");
        assert!(lists.is_valid(2));

        let structs = lists.values().as_struct();
        let names = structs.column(0).as_string::<i32>();
        let ns = structs.column(1).as_primitive::<Int32Type>();
        assert_eq!(names.value(0), "a");
        assert_eq!(names.value(1), "b");
        assert!(
            names.is_null(2) && names.is_null(3),
            "the blank's nullable struct member is nulled"
        );
        assert_eq!(names.value(4), "c");
        assert_eq!(names.value(5), "d");
        // `n` is non-nullable fixed width, so the blank copies row zero's two slots (1, 2).
        assert_eq!(ns.values(), &[1, 2, 1, 2, 3, 4]);
    }

    #[test]
    fn test_nullable_variable_width_blanks_are_null() {
        let batch = batch_of(
            Field::new("s", DataType::Utf8, true),
            Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])),
        );

        let with_blanks = add_blanks(batch, &[1, 4], ANY_VERSION, None).unwrap();

        assert_eq!(with_blanks.num_rows(), 5);
        let values = with_blanks.column(0).as_string::<i32>();
        assert_eq!(values.null_count(), 2);
        assert!(values.is_null(1));
        assert!(values.is_null(4));
        assert_eq!(values.value(0), "alpha");
        assert_eq!(values.value(2), "beta");
        assert_eq!(values.value(3), "gamma");
        assert_eq!(values.value_data().len(), "alphabetagamma".len());
    }

    #[test]
    fn test_non_nullable_variable_width_blanks_are_empty() {
        let batch = batch_of(
            Field::new("b", DataType::Binary, false),
            Arc::new(BinaryArray::from(vec![&[1u8, 2, 3][..], &[4u8, 5, 6][..]])),
        );

        let with_blanks = add_blanks(batch, &[1, 3], ANY_VERSION, None).unwrap();

        let values = with_blanks.column(0).as_binary::<i32>();
        assert_eq!(values.null_count(), 0);
        assert_eq!(values.value(0), &[1, 2, 3]);
        assert_eq!(values.value(1), b"");
        assert_eq!(values.value(2), &[4, 5, 6]);
        assert_eq!(values.value(3), b"");
        assert_eq!(values.value_data().len(), 6);
    }

    #[test]
    fn test_write_schema_controls_blank_nullability() {
        let batch = batch_of(
            // Arrow producers commonly mark every input field nullable.
            Field::new("b", DataType::Binary, true),
            Arc::new(BinaryArray::from(vec![&[1u8, 2, 3][..]])),
        );
        let write_schema =
            LanceSchema::try_from(&Schema::new(vec![Field::new("b", DataType::Binary, false)]))
                .unwrap();

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, Some(&write_schema)).unwrap();

        let values = with_blanks.column(0).as_binary::<i32>();
        assert_eq!(values.null_count(), 0);
        assert_eq!(values.value(1), b"");
    }

    /// The cheap-blank optimization is gated on v2.0+. Legacy v1 keeps the original
    /// behavior of copying row zero into every blank; v2.0+ gives a nullable column a
    /// null blank instead.
    #[rstest]
    #[case::v1_copies_row_zero(ConcreteFileVersion::V1, false)]
    #[case::v2_0(ConcreteFileVersion::V2_0, true)]
    #[case::v2_1(ConcreteFileVersion::V2_1, true)]
    fn test_blank_optimization_gated_on_v2(
        #[case] storage_version: ConcreteFileVersion,
        #[case] expect_null: bool,
    ) {
        let batch = batch_of(
            Field::new("b", DataType::LargeBinary, true),
            Arc::new(LargeBinaryArray::from(vec![&[1u8, 2][..]])),
        );

        let with_blanks = add_blanks(batch, &[1], storage_version, None).unwrap();

        let values = with_blanks.column(0).as_binary::<i64>();
        if expect_null {
            assert_eq!(values.null_count(), 1);
            assert!(values.is_null(1));
        } else {
            // v1 falls back to take(0): the blank copies row zero, not null or empty.
            assert_eq!(values.null_count(), 0);
            assert_eq!(values.value(1), &[1u8, 2]);
        }
    }

    #[rstest]
    #[case::v2_0(ConcreteFileVersion::V2_0)]
    #[case::v2_1(ConcreteFileVersion::V2_1)]
    fn test_struct_blanks_shrink_variable_width_children(
        #[case] storage_version: ConcreteFileVersion,
    ) {
        let fields = Fields::from(vec![
            Field::new("payload", DataType::Binary, false),
            Field::new("n", DataType::Int32, false),
        ]);
        let struct_array = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(BinaryArray::from(vec![&[1u8; 8][..], &[2u8; 8][..]])) as ArrayRef,
                Arc::new(Int32Array::from(vec![10, 20])),
            ],
            None,
        );
        let batch = batch_of(
            Field::new("s", DataType::Struct(fields), true),
            Arc::new(struct_array),
        );

        let with_blanks = add_blanks(batch, &[1, 3], storage_version, None).unwrap();

        let structs = with_blanks.column(0).as_struct();
        assert_eq!(structs.null_count(), 0);
        let payload = structs.column(0).as_binary::<i32>();
        assert_eq!(payload.value(0), &[1u8; 8]);
        assert_eq!(payload.value(1), b"");
        assert_eq!(payload.value(2), &[2u8; 8]);
        assert_eq!(payload.value(3), b"");
        assert_eq!(payload.value_data().len(), 16);
        let n = structs.column(1).as_primitive::<Int32Type>();
        assert_eq!(n.values(), &[10, 10, 20, 10]);
    }

    /// A blob v2 column that pins `position`/`size` as required cannot use the empty inline
    /// descriptor: that blank nulls every non-`data` child, but a non-nullable child cannot
    /// be nulled, and taking it from row zero would leave `position`/`size` set while `uri`
    /// is null -- a descriptor the blob writer rejects. The column falls back to copying row
    /// zero, the pre-optimization blank, which is always a valid descriptor.
    ///
    /// The two-field shape keeps the optimization;
    /// `test_non_nullable_blob_blanks_add_no_sidecar_bytes` pins that end to end.
    #[test]
    fn test_blob_blank_with_required_position_and_size_copies_row_zero() {
        let fields = Fields::from(vec![
            Field::new("data", DataType::LargeBinary, true),
            Field::new("uri", DataType::Utf8, true),
            Field::new("position", DataType::UInt64, false),
            Field::new("size", DataType::UInt64, false),
        ]);
        // A valid external descriptor: `uri` set, `data` absent, `position`/`size` present.
        let struct_array = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![None::<&[u8]>])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("ref://blob")])),
                Arc::new(UInt64Array::from(vec![7])),
                Arc::new(UInt64Array::from(vec![7])),
            ],
            None,
        );
        let blob_field = Field::new("blob", DataType::Struct(fields), false).with_metadata(
            HashMap::from([(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]),
        );
        let batch = batch_of(blob_field, Arc::new(struct_array));

        let plans = blank_plans(&batch, ANY_VERSION, None).unwrap();
        assert!(
            matches!(plans.as_slice(), [BlankPlan::Take]),
            "a required `position`/`size` forces the whole column to copy row zero"
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let structs = with_blanks.column(0).as_struct();
        assert_eq!(structs.null_count(), 0);
        // The blank is a copy of row zero's external descriptor.
        let data = structs.column(0).as_binary::<i64>();
        assert!(data.is_null(0) && data.is_null(1), "`data` stays absent");
        let uri = structs.column(1).as_string::<i32>();
        assert_eq!(uri.value(0), "ref://blob");
        assert_eq!(
            uri.value(1),
            "ref://blob",
            "the blank copies row zero's uri"
        );
        assert_eq!(
            structs
                .column(2)
                .as_primitive::<arrow::datatypes::UInt64Type>()
                .values(),
            &[7, 7]
        );
        assert_eq!(
            structs
                .column(3)
                .as_primitive::<arrow::datatypes::UInt64Type>()
                .values(),
            &[7, 7]
        );
    }

    /// A nested child takes a null rather than an empty value when its own Arrow field
    /// allows one. Both cost nothing to store, but an empty value is a real value a
    /// reader may interpret: a blob v2 column is a struct of `data` and `uri`, and an
    /// empty `uri` reads as an external reference to nowhere instead of "no blob".
    /// `test_add_columns_blob_blanks_round_trip` covers that end to end.
    #[test]
    fn test_nullable_struct_children_take_null_blanks() {
        let fields = Fields::from(vec![
            Field::new("data", DataType::LargeBinary, true),
            Field::new("uri", DataType::Utf8, true),
        ]);
        let struct_array = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![&b"payload"[..]])) as ArrayRef,
                Arc::new(StringArray::from(vec![None::<&str>])),
            ],
            None,
        );
        let batch = batch_of(
            Field::new("blob", DataType::Struct(fields), true),
            Arc::new(struct_array),
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let structs = with_blanks.column(0).as_struct();
        // The struct itself stays valid, mirroring row zero.
        assert_eq!(structs.null_count(), 0);
        let data = structs.column(0).as_binary::<i64>();
        assert_eq!(data.value(0), b"payload");
        assert!(data.is_null(1), "the blank payload must be null, not empty");
        assert!(
            structs.column(1).as_string::<i32>().is_null(1),
            "the blank uri must be null, not an empty reference"
        );
    }

    #[test]
    fn test_write_schema_controls_nested_blank_nullability() {
        let input_children = Fields::from(vec![Field::new("payload", DataType::Binary, true)]);
        let struct_array = StructArray::new(
            input_children.clone(),
            vec![Arc::new(BinaryArray::from(vec![&b"payload"[..]]))],
            None,
        );
        let batch = batch_of(
            Field::new("s", DataType::Struct(input_children), true),
            Arc::new(struct_array),
        );

        let write_children = Fields::from(vec![Field::new("payload", DataType::Binary, false)]);
        let write_schema = LanceSchema::try_from(&Schema::new(vec![Field::new(
            "s",
            DataType::Struct(write_children),
            true,
        )]))
        .unwrap();

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, Some(&write_schema)).unwrap();

        let payload = with_blanks
            .column(0)
            .as_struct()
            .column(0)
            .as_binary::<i32>();
        assert_eq!(payload.null_count(), 0);
        assert_eq!(payload.value(0), b"payload");
        assert_eq!(payload.value(1), b"");
    }

    #[test]
    fn test_struct_blanks_preserve_parent_nulls() {
        let fields = Fields::from(vec![
            Field::new("payload", DataType::Binary, false),
            Field::new("n", DataType::Int32, false),
        ]);
        let struct_array = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(BinaryArray::from(vec![&b"a"[..], &b"b"[..]])) as ArrayRef,
                Arc::new(Int32Array::from(vec![10, 20])),
            ],
            Some(NullBuffer::from(vec![true, false])),
        );
        let batch = batch_of(
            Field::new("s", DataType::Struct(fields), true),
            Arc::new(struct_array),
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let structs = with_blanks.column(0).as_struct();
        assert_eq!(structs.null_count(), 1);
        assert!(structs.is_valid(0));
        assert!(structs.is_valid(1), "the blank struct row must be valid");
        assert!(
            structs.is_null(2),
            "the original parent null must be preserved"
        );
    }

    /// The shape that used to overflow: a long deleted run behind a large value.
    ///
    /// This asserts the mechanism -- blanks add no payload bytes, so the values buffer
    /// stays the size of the live rows alone -- rather than the overflow itself.
    /// Reproducing an `i32` offset overflow needs `live_rows * value_len > i32::MAX`,
    /// over 2 GiB of allocation, which is not something a unit test should do. Keeping
    /// the buffer flat is what makes the overflow unreachable at any scale.
    #[test]
    fn test_heavy_deletion_blanks_cost_no_payload_bytes() {
        const LIVE_ROWS: usize = 410;
        const BLANKS: usize = 9583;
        const VALUE_LEN: usize = 8 * 1024;

        let value = vec![7u8; VALUE_LEN];
        let array: ArrayRef = Arc::new(BinaryArray::from_iter_values(
            (0..LIVE_ROWS).map(|_| value.as_slice()),
        ));
        let live_payload = array
            .to_data()
            .buffers()
            .iter()
            .map(|b| b.len())
            .sum::<usize>();
        let batch = batch_of(Field::new("b", DataType::Binary, true), array);
        let offsets = (LIVE_ROWS as u32..(LIVE_ROWS + BLANKS) as u32).collect::<Vec<_>>();

        let with_blanks = add_blanks(batch, &offsets, ANY_VERSION, None).unwrap();

        assert_eq!(with_blanks.num_rows(), LIVE_ROWS + BLANKS);
        let values = with_blanks.column(0).as_binary::<i32>();
        assert_eq!(values.null_count(), BLANKS);
        // Copying row zero into every blank would have made this
        // `(LIVE_ROWS + BLANKS) * VALUE_LEN`, roughly 78 MiB instead of 3 MiB.
        assert_eq!(values.value_data().len(), LIVE_ROWS * VALUE_LEN);
        let out_payload = with_blanks
            .column(0)
            .to_data()
            .buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum::<usize>();
        assert!(
            out_payload < live_payload * 102 / 100,
            "blanks grew the column from {live_payload} to {out_payload} bytes"
        );
    }

    /// View types keep their data buffers no matter how they are selected, so the
    /// blank only has to avoid inventing a value.
    #[rstest]
    #[case::utf8_view(DataType::Utf8View)]
    #[case::binary_view(DataType::BinaryView)]
    fn test_view_type_blanks(#[case] data_type: DataType) {
        let array: ArrayRef = match data_type {
            DataType::Utf8View => Arc::new(StringViewArray::from(vec!["alpha"])),
            _ => Arc::new(BinaryViewArray::from(vec![&b"alpha"[..]])),
        };

        let empty = add_blanks(
            batch_of(Field::new("v", data_type.clone(), false), array.clone()),
            &[1],
            ANY_VERSION,
            None,
        )
        .unwrap();
        assert_eq!(empty.column(0).null_count(), 0);
        assert_eq!(empty.column(0).len(), 2);

        let nulled = add_blanks(
            batch_of(Field::new("v", data_type, true), array),
            &[1],
            ANY_VERSION,
            None,
        )
        .unwrap();
        assert_eq!(nulled.column(0).null_count(), 1);
        assert!(nulled.column(0).is_null(1));
    }

    /// A zero-width fixed-size list has no child rows to select, so the child array
    /// passes through untouched and only the parent validity is rebuilt.
    #[test]
    fn test_zero_width_fixed_size_list_blanks() {
        let child = Arc::new(Field::new("item", DataType::Binary, false));
        let values: ArrayRef = Arc::new(BinaryArray::from(Vec::<&[u8]>::new()));
        let lists =
            FixedSizeListArray::try_new_with_length(child.clone(), 0, values, None, 2).unwrap();
        let batch = batch_of(
            Field::new("l", DataType::FixedSizeList(child, 0), true),
            Arc::new(lists),
        );

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();

        let lists = with_blanks.column(0).as_fixed_size_list();
        assert_eq!(lists.len(), 3);
        assert_eq!(lists.values().len(), 0);
        assert!(lists.is_null(1));
    }

    /// The write schema is indexed by top-level name. Resolving the name as a field
    /// path instead would mishandle the characters that syntax reserves: a backtick is
    /// legal in a Lance field name -- only `.` is rejected -- but it either fails to
    /// parse or, worse, parses as a quote and points at a different field.
    #[test]
    fn test_blank_plans_look_up_names_that_are_not_field_paths() {
        for name in ["payload`raw", "`payload`"] {
            // Arrow producers commonly mark every input field nullable; the dataset's
            // non-nullable declaration is what has to win.
            let batch = batch_of(
                Field::new(name, DataType::Binary, true),
                Arc::new(BinaryArray::from(vec![&[1u8, 2, 3][..]])),
            );
            let write_schema = LanceSchema::try_from(&Schema::new(vec![Field::new(
                name,
                DataType::Binary,
                false,
            )]))
            .unwrap();

            let with_blanks = add_blanks(batch, &[1], ANY_VERSION, Some(&write_schema))
                .unwrap_or_else(|error| {
                    panic!("add_blanks rejected the column named {name:?}: {error}")
                });

            let values = with_blanks.column(0).as_binary::<i32>();
            assert_eq!(values.null_count(), 0, "{name}");
            assert_eq!(values.value(1), b"", "{name}");
        }
    }

    #[test]
    fn test_blank_plans_reject_field_missing_from_write_schema() {
        let batch = batch_of(
            Field::new("b", DataType::Binary, true),
            Arc::new(BinaryArray::from(vec![&[1u8][..]])),
        );
        let write_schema =
            LanceSchema::try_from(&Schema::new(vec![Field::new("c", DataType::Binary, true)]))
                .unwrap();

        let err = add_blanks(batch, &[1], ANY_VERSION, Some(&write_schema)).unwrap_err();
        assert!(matches!(err, Error::Internal { .. }), "{err:?}");
        assert!(
            err.to_string().contains("missing from the write schema"),
            "{err}"
        );
    }

    /// A JSON column stores its documents as text (`arrow.json`) or as encoded JSONB
    /// (`lance.json`), and the writer re-encodes or decodes every value it is handed.
    /// The generic zero-payload blank is an empty value in both cases, which only works
    /// because `jsonb` reads empty input as the `null` document. Pin that: if it ever
    /// became an error instead, `add_columns` would start failing on any fragment with
    /// deleted rows and a non-nullable JSON column.
    #[rstest]
    #[case::arrow_json_utf8(arrow_json_field("j", DataType::Utf8, false))]
    #[case::arrow_json_large_utf8(arrow_json_field("j", DataType::LargeUtf8, false))]
    #[case::lance_json(lance_json_field("j", false))]
    fn test_non_nullable_json_blanks_round_trip_as_null_documents(#[case] field: Field) {
        let document = r#"{"name": "Alice"}"#;
        let array: ArrayRef = match field.data_type() {
            DataType::Utf8 => Arc::new(StringArray::from(vec![document])),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec![document])),
            _ => Arc::new(LargeBinaryArray::from(vec![
                encode_json(document).unwrap().as_slice(),
            ])),
        };
        let is_jsonb = matches!(field.data_type(), DataType::LargeBinary);

        let with_blanks = add_blanks(batch_of(field, array), &[1], ANY_VERSION, None).unwrap();

        assert_eq!(with_blanks.column(0).null_count(), 0);
        let encoded = if is_jsonb {
            with_blanks.column(0).clone()
        } else {
            // This is the step that re-encodes the blank, so it is where an empty
            // string would be rejected.
            convert_json_columns(&with_blanks)
                .unwrap()
                .column(0)
                .clone()
        };
        let values = encoded.as_binary::<i64>();
        assert_eq!(decode_json(values.value(0)), r#"{"name":"Alice"}"#);
        assert_eq!(decode_json(values.value(1)), "null");
    }

    #[rstest]
    #[case::arrow_json(arrow_json_field("j", DataType::Utf8, true))]
    #[case::lance_json(lance_json_field("j", true))]
    fn test_nullable_json_blanks_are_null(#[case] field: Field) {
        let document = r#"{"name": "Alice"}"#;
        let array: ArrayRef = match field.data_type() {
            DataType::Utf8 => Arc::new(StringArray::from(vec![document])),
            _ => Arc::new(LargeBinaryArray::from(vec![
                encode_json(document).unwrap().as_slice(),
            ])),
        };

        let with_blanks = add_blanks(batch_of(field, array), &[1], ANY_VERSION, None).unwrap();

        assert_eq!(with_blanks.column(0).null_count(), 1);
        assert!(with_blanks.column(0).is_null(1));
        // A null passes through the conversion untouched.
        convert_json_columns(&with_blanks).unwrap();
    }

    /// v1 no longer runs the cheap-blank path at all, so even a JSON column just copies
    /// row zero rather than getting the empty-document treatment v2.0+ would give it.
    #[test]
    fn test_legacy_v1_blanks_copy_row_zero() {
        let field = arrow_json_field("j", DataType::Utf8, true);
        let document = r#"{"name": "Alice"}"#;
        let batch = batch_of(field, Arc::new(StringArray::from(vec![document])));

        let with_blanks = add_blanks(batch, &[1], ConcreteFileVersion::V1, None).unwrap();

        assert_eq!(with_blanks.column(0).null_count(), 0);
        assert_eq!(with_blanks.column(0).as_string::<i32>().value(1), document);
    }

    /// A JSON child of a struct is planned like any other variable-width child, so its
    /// blank is the empty document rather than a copy of row zero.
    #[test]
    fn test_json_inside_struct_is_shrunk() {
        let document = r#"{"name": "Alice"}"#;
        let fields = Fields::from(vec![
            arrow_json_field("j", DataType::Utf8, false),
            Field::new("n", DataType::Int32, false),
        ]);
        let struct_array = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(StringArray::from(vec![document])) as ArrayRef,
                Arc::new(Int32Array::from(vec![10])),
            ],
            None,
        );
        let batch = batch_of(
            Field::new("s", DataType::Struct(fields), true),
            Arc::new(struct_array),
        );

        let plans = blank_plans(&batch, ANY_VERSION, None).unwrap();
        assert!(matches!(plans.as_slice(), [BlankPlan::Struct(_)]));

        let with_blanks = add_blanks(batch, &[1], ANY_VERSION, None).unwrap();
        let converted = convert_json_columns(&with_blanks).unwrap();
        let json = converted.column(0).as_struct().column(0).as_binary::<i64>();
        assert_eq!(decode_json(json.value(0)), r#"{"name":"Alice"}"#);
        assert_eq!(decode_json(json.value(1)), "null");
    }
}
