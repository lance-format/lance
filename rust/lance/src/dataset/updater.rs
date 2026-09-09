// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{RecordBatch, UInt32Array};
use futures::StreamExt;
use lance_core::datatypes::{OnMissing, OnTypeMismatch};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result, datatypes::Schema};
use lance_table::format::{DataFile, Fragment};
use lance_table::utils::stream::ReadBatchFutStream;

use super::Dataset;
use super::fragment::FragmentReader;
use super::scanner::get_default_batch_size;
use super::versions;
use super::write::{GenericWriter, cleanup_data_fragments};
use crate::dataset::FileFragment;
use crate::dataset::utils::SchemaAdapter;

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

    writer: Option<Box<dyn GenericWriter>>,

    /// The final schema of the fragment after the update.
    final_schema: Option<Schema>,

    /// The schema the new files will be written in. This only contains new columns.
    write_schema: Option<Schema>,

    /// The adapter to convert the logical data to physical data.
    schema_adapter: Option<SchemaAdapter>,

    allow_external_blob_outside_bases: bool,

    finished: bool,

    deletion_restorer: DeletionRestorer,
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
    ) -> Result<Self> {
        let (write_schema, final_schema) = if let Some((write_schema, final_schema)) = schemas {
            (Some(write_schema), Some(final_schema))
        } else {
            (None, None)
        };

        let storage_version = fragment
            .dataset()
            .manifest()
            .data_storage_format
            .lance_file_format();
        let legacy_batch_size =
            versions::row_group_size_for_rewrite(storage_version, &fragment).await?;

        let batch_size = match (&legacy_batch_size, batch_size) {
            // If this is a v1 dataset we must use the row group size of the file
            (Some(legacy_batch_size), _) => *legacy_batch_size,
            // If this is a v2 dataset, let the user pick the batch size
            (None, Some(user_specified_batch_size)) => user_specified_batch_size,
            // Otherwise, default to 1024 if the user didn't specify anything
            (None, None) => get_default_batch_size().unwrap_or(1024) as u32,
        };

        let input_stream = reader.read_all(batch_size).await?;

        Ok(Self {
            fragment,
            input_stream,
            last_input: None,
            writer: None,
            write_schema,
            final_schema,
            // The schema adapter needs the data schema, not the logical schema, so it can't be
            // created until after the first batch is read.
            schema_adapter: None,
            allow_external_blob_outside_bases: false,
            finished: false,
            deletion_restorer: DeletionRestorer::new(deletion_vector, legacy_batch_size),
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
        let data_storage_version = self
            .dataset()
            .manifest()
            .data_storage_format
            .lance_file_format();

        versions::open_update_writer(
            data_storage_version,
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

        // Add back in deleted rows
        let batch = self.deletion_restorer.restore(batch)?;

        if self.writer.is_none() {
            if self.write_schema.is_none() {
                // Need to infer the schema.
                let output_schema = batch.schema();
                let mut final_schema = self.fragment.schema().merge(output_schema.as_ref())?;
                final_schema.set_field_id(Some(self.fragment.dataset().manifest.max_field_id()));
                self.final_schema = Some(final_schema);
                self.final_schema.as_ref().unwrap().validate()?;
                self.write_schema = Some(self.final_schema.as_ref().unwrap().project_by_schema(
                    output_schema.as_ref(),
                    OnMissing::Error,
                    OnTypeMismatch::Error,
                )?);
            }

            self.writer = Some(
                self.new_writer(self.write_schema.as_ref().unwrap().clone())
                    .await?,
            );
        }

        let schema_adapter = if let Some(schema_adapter) = self.schema_adapter.as_ref() {
            schema_adapter
        } else {
            self.schema_adapter = Some(SchemaAdapter::new(batch.schema()));
            self.schema_adapter.as_ref().unwrap()
        };

        let batch = schema_adapter.to_physical_batch(batch)?;

        let writer = self.writer.as_mut().unwrap();

        writer.write(&[batch]).await?;

        Ok(())
    }

    /// Finish updating this fragment, and returns the updated [`Fragment`].
    pub async fn finish(&mut self) -> Result<Fragment> {
        if let Some(writer) = self.writer.as_mut() {
            let (_, data_file) = writer.finish().await?;
            self.fragment.metadata.files.push(data_file);
        }

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
        let storage_version = self
            .dataset()
            .manifest()
            .data_storage_format
            .lance_file_format();
        // cleanup_data_fragments only needs path/base_id to remove the unfinished
        // data file and any blob sidecars. Build a minimal synthetic fragment so
        // we can reuse the shared cleanup path without fabricating full metadata.
        fragment.files.push(DataFile::new(
            path,
            vec![],
            vec![],
            storage_version,
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

    deletion_vector_iter: Option<Box<dyn Iterator<Item = u32> + Send>>,

    last_deleted_row_id: Option<u32>,

    /// Blank rows owed to batches that had no live row to copy a placeholder from
    ///
    /// See [`Self::restore`] for why they are deferred instead of materialized.
    /// Only ever non-zero for non-legacy files, which are the only ones that defer.
    pending_blank_rows: u32,
}

impl DeletionRestorer {
    fn new(deletion_vector: DeletionVector, legacy_batch_size: Option<u32>) -> Self {
        Self {
            current_row_id: 0,
            legacy_batch_size,
            deletion_vector_iter: Some(deletion_vector.into_sorted_iter()),
            last_deleted_row_id: None,
            pending_blank_rows: 0,
        }
    }

    fn is_exhausted(&self) -> bool {
        self.deletion_vector_iter.is_none() && self.pending_blank_rows == 0
    }

    fn is_full(batch_size: Option<u32>, num_rows: u32) -> bool {
        if let Some(legacy_batch_size) = batch_size {
            // We should never encounter the case that `batch_size < num_rows` because
            // that would mean we have a v1 writer and it generated a batch with more rows
            // than expected
            debug_assert!(legacy_batch_size >= num_rows);
            legacy_batch_size == num_rows
        } else {
            false
        }
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
    /// If the batch size is 10 (or unset) then we do include 19 and the return will be [1, 2, 9]
    ///
    /// In v2 files, since the batch size will be unset, we will always include as many deleted
    /// rows at the end as we can.
    fn deleted_batch_offsets_in_range(&mut self, mut num_rows: u32) -> Vec<u32> {
        let mut deleted = Vec::new();
        let first_row_id = self.current_row_id;
        // The last row id (exclusive) in the batch
        let mut last_row_id = first_row_id + num_rows;
        // If there are zero deleted rows then the range covered will be first_row_id..last_row_id
        if self.deletion_vector_iter.is_none() {
            return deleted;
        }
        let deletion_vector_iter = self.deletion_vector_iter.as_mut().unwrap();

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
                    || (next_deleted_id == last_row_id
                        && Self::is_full(self.legacy_batch_size, num_rows))
                {
                    // Either the next deleted id is out of range or it is the next row but
                    // we are full.  Either way, stash it and return
                    self.last_deleted_row_id = Some(next_deleted_id);
                    return deleted;
                }
                // Otherwise, the deleted row is in range, and we have space in our batch
                // and so we include it
                deleted.push(next_deleted_id - first_row_id);
                last_row_id += 1;
                num_rows += 1;
            } else {
                // Deleted row ids iterator is exhausted
                self.deletion_vector_iter = None;
                // `is_exhausted` reads these two together, so a stash left behind here
                // would make it report exhaustion while a deleted row is still owed.
                debug_assert!(self.last_deleted_row_id.is_none());
                return deleted;
            }
            next_deleted_id = deletion_vector_iter.next();
        }
    }

    /// Restore the deleted rows for one batch of live rows.
    ///
    /// Blanks are materialized by copying the batch's first live row (see
    /// [`add_blanks`]), so a batch with no live rows has nothing to copy from. That
    /// happens when a deleted run starts at physical row 0: there is no preceding
    /// batch for [`Self::deleted_batch_offsets_in_range`] to append the run to, so
    /// the run arrives as an empty batch carrying every one of its offsets.
    ///
    /// Rather than invent placeholder values for an arbitrary schema, we remember
    /// how many blanks we owe and prepend them to the next batch that does have a
    /// live row. Deleted rows sort before the live rows that follow them, so the
    /// physical row order is preserved either way.
    fn restore(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        // Holds by construction today — deferring is the only thing that sets
        // pending_blank_rows and it is gated on non-legacy — so this documents the
        // invariant the legacy row-count check below depends on rather than guarding
        // against a state we can reach.
        debug_assert!(self.pending_blank_rows == 0 || self.legacy_batch_size.is_none());

        // Because of deleted rows, the number of row ids in the batch might not
        // match the length.
        let deleted_batch_offsets = self.deleted_batch_offsets_in_range(batch.num_rows() as u32);

        // Legacy files must reproduce the original row group size, which deferring
        // would break, so they keep reporting the pre-existing error instead.
        if batch.num_rows() == 0 && self.legacy_batch_size.is_none() {
            let deferred = deleted_batch_offsets.len() as u32;
            self.pending_blank_rows += deferred;
            self.current_row_id += deferred;
            return Ok(batch);
        }

        let pending_blank_rows = self.pending_blank_rows;
        let batch_offsets = if pending_blank_rows == 0 {
            deleted_batch_offsets
        } else {
            // The deferred blanks take the front of the batch, pushing the offsets
            // computed for this batch back by that many rows.
            let mut batch_offsets =
                Vec::with_capacity(pending_blank_rows as usize + deleted_batch_offsets.len());
            batch_offsets.extend(0..pending_blank_rows);
            batch_offsets.extend(
                deleted_batch_offsets
                    .iter()
                    .map(|offset| offset + pending_blank_rows),
            );
            batch_offsets
        };

        let batch = add_blanks(batch, &batch_offsets)?;

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

        // The deferred blanks were counted when they were deferred.
        self.current_row_id += batch.num_rows() as u32 - pending_blank_rows;
        self.pending_blank_rows = 0;
        Ok(batch)
    }
}

/// Add blank rows where there are deleted rows
///
/// `batch_offsets` must be strictly increasing, and no offset may require more
/// live rows before it than the batch has left: an offset is the position a blank
/// takes in the output, so either kind of violation asks for an impossible number
/// of live rows in between.
///
/// Blanks copy the batch's first row, so the batch must have at least one row.
/// [`DeletionRestorer::restore`] defers blanks past an empty batch to keep that
/// true; only legacy files, which cannot defer, can still reach the error below.
pub(crate) fn add_blanks(batch: RecordBatch, batch_offsets: &[u32]) -> Result<RecordBatch> {
    // Fast early return
    if batch_offsets.is_empty() {
        return Ok(batch);
    }

    if batch.num_rows() == 0 {
        return Err(Error::not_supported(
            "Fragment Updater: missing too many rows in merge, run compaction to materialize \
             deletions first",
        ));
    }

    let num_live_rows = batch.num_rows() as u32;
    let mut selection_vector = Vec::<u32>::with_capacity(batch.num_rows() + batch_offsets.len());
    let mut batch_pos = 0;
    let mut next_id = 0;
    for (idx, batch_offset) in batch_offsets.iter().enumerate() {
        // A non-increasing offset panics in debug and wraps in release; reject it
        // up front so the error names the real problem.
        let num_rows = batch_offset.checked_sub(next_id).ok_or_else(|| {
            Error::internal(format!(
                "Fragment Updater: blank offsets must be strictly increasing, but offset \
                 {batch_offset} (entry {idx} of {}) is below the expected minimum {next_id}",
                batch_offsets.len()
            ))
        })?;
        // An offset needing more live rows than remain would index past the batch:
        // `take` runs unchecked below, so catch it here rather than letting it
        // panic inside arrow or, worse, read the wrong rows.
        if num_rows > num_live_rows - batch_pos {
            return Err(Error::internal(format!(
                "Fragment Updater: blank offset {batch_offset} (entry {idx} of \
                 {}) needs {num_rows} more live rows before it, but {} of the batch's \
                 {num_live_rows} are still unused",
                batch_offsets.len(),
                num_live_rows - batch_pos
            )));
        }
        selection_vector.extend(batch_pos..batch_pos + num_rows);
        // For simplicity, we just use the first value for deleted rows
        // TODO: optimize this to use small value for each column.
        selection_vector.push(0);
        next_id = *batch_offset + 1;
        batch_pos += num_rows;
    }
    selection_vector.extend(batch_pos..num_live_rows);
    let selection_vector = UInt32Array::from(selection_vector);

    let arrays = batch
        .columns()
        .iter()
        .map(|array| {
            arrow::compute::take(array.as_ref(), &selection_vector, None)
                .map_err(|e| Error::arrow(format!("Failed to add blanks: {}", e)))
        })
        .collect::<Result<Vec<_>>>()?;

    let batch = RecordBatch::try_new(batch.schema(), arrays)?;

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use arrow::{array::AsArray, datatypes::Int32Type};
    use lance_datagen::RowCount;
    use rstest::rstest;

    use super::{Error, add_blanks};

    #[test]
    fn test_restore_deletes() {
        for batch_size in &[None, Some(10)] {
            let mut restorer = super::DeletionRestorer::new(
                vec![11, 12, 19, 20, 25].into_iter().collect(),
                *batch_size,
            );

            let batch = lance_datagen::gen_batch()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(10))
                .unwrap();
            // First batch is rows ids 0..9 so nothing is restored
            let restored = restorer.restore(batch.clone()).unwrap();
            assert_eq!(restored, batch);

            let batch = lance_datagen::gen_batch()
                .col("x", lance_datagen::array::step::<Int32Type>())
                .into_batch_rows(RowCount::from(7))
                .unwrap();
            // Next batch is rows ids 10..16 so we need to restore 11, 12
            // 19, and maybe 20 (depends on batch size)
            let restored = restorer.restore(batch).unwrap();
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
            if *batch_size == Some(10) {
                assert_eq!(values.len(), 10);
            } else {
                assert_eq!(values.value(10), 0);
                assert_eq!(values.len(), 11);
            }
        }
    }

    /// Rows 0..10 are deleted, so the first read batch has no live rows. A legacy file
    /// cannot defer the blanks and reports the error instead;
    /// [`test_restore_deletes_leading_empty_batch`] covers the v2 side. Row 15 is here
    /// only to keep the deletion vector identical between the two tests — legacy fails
    /// on the first batch, so 15 is never reached.
    #[test]
    fn test_restore_deletes_leading_empty_batch_legacy() {
        let mut restorer = super::DeletionRestorer::new((0..10).chain([15]).collect(), Some(10));

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();

        // Assert the source, not just is_err: the batch-size check further down
        // returns Internal, and the two are different failures.
        let err = restorer.restore(empty).unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
    }

    /// The v2 side of the same deletion vector: blanks owed by a batch with no live
    /// row are deferred to a later batch that has one to copy.
    #[test]
    fn test_restore_deletes_leading_empty_batch() {
        let mut restorer = super::DeletionRestorer::new((0..10).chain([15]).collect(), None);

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();

        // Nothing is written for the fully deleted batch itself.
        assert_eq!(restorer.restore(empty.clone()).unwrap().num_rows(), 0);
        assert!(!restorer.is_exhausted());

        // A second empty batch must carry the debt through untouched: row 15 is
        // out of its range, so it defers nothing of its own.
        let restored = restorer.restore(empty).unwrap();
        assert_eq!(restored.num_rows(), 0);
        assert!(!restorer.is_exhausted());

        // The next batch covers row ids 10..15, so it owes the 10 deferred blanks
        // in front of its own rows and one more for row 15 at the end. That last
        // one is what pins the offset shift: without it the offsets would not be
        // increasing.
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();
        let restored = restorer.restore(batch).unwrap();

        assert_eq!(restored.num_rows(), 16);
        let values = restored.column(0).as_primitive::<Int32Type>();
        // Blanks copy the batch's first live row rather than inventing a value,
        // which is what lets a non-nullable column through.
        for i in 0..10 {
            assert_eq!(values.value(i), 0);
        }
        for i in 0..5 {
            assert_eq!(values.value(10 + i), i as i32);
        }
        assert_eq!(values.value(15), 0);
        assert!(restorer.is_exhausted());
    }

    /// The debt itself has to keep the restorer from reporting exhaustion, not just
    /// the deletion vector. With no row past the deleted run the iterator empties on
    /// the first call, so only `pending_blank_rows` can hold `is_exhausted` back —
    /// and it must, or `Updater::next` would accept a data file short by ten rows.
    #[test]
    fn test_restore_deletes_owes_blanks_after_vector_drains() {
        let mut restorer = super::DeletionRestorer::new((0..10).collect(), None);

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();

        assert_eq!(restorer.restore(empty).unwrap().num_rows(), 0);
        assert!(!restorer.is_exhausted());

        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();
        assert_eq!(restorer.restore(batch).unwrap().num_rows(), 15);
        assert!(restorer.is_exhausted());
    }

    /// A deletion vector naming a row the fragment does not have leaves the restorer
    /// unexhausted with no blanks owed: the id stays stashed, so the iterator is never
    /// drained. `Updater::next` relies on this to refuse rather than write a data file
    /// missing that row, and the error must not claim a blank count for it.
    #[test]
    fn test_restore_deletes_not_exhausted_when_deletion_vector_overruns() {
        let mut restorer = super::DeletionRestorer::new([100].into_iter().collect(), None);

        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();

        // Row 100 is past this batch, so it is stashed rather than consumed and
        // nothing is restored. No blanks are owed either — which is why the error in
        // `Updater::next` cannot name a count.
        assert_eq!(restorer.restore(batch).unwrap().num_rows(), 5);
        assert!(!restorer.is_exhausted());
    }

    /// Deferred blanks are counted into `current_row_id` when they are deferred, so
    /// consuming them must not count them again. A later deleted row is what makes
    /// the double count observable: it lands at the wrong offset once the restorer
    /// thinks the fragment is further along than it is.
    #[test]
    fn test_restore_deletes_does_not_double_count_deferred_blanks() {
        let mut restorer = super::DeletionRestorer::new((0..10).chain([22]).collect(), None);

        let empty = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(0))
            .unwrap();
        assert_eq!(restorer.restore(empty).unwrap().num_rows(), 0);

        // Rows 10..20 are live, so this batch pays off the ten blanks and nothing
        // else: row 22 is past its range and stays stashed.
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(10))
            .unwrap();
        assert_eq!(restorer.restore(batch).unwrap().num_rows(), 20);
        assert!(!restorer.is_exhausted());

        // Row 22 falls inside this batch's range, but only if current_row_id sits at
        // 20. Counting the deferred blanks twice would have pushed it to 30, putting
        // row 22 behind the batch and dropping its blank.
        let batch = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_batch_rows(RowCount::from(5))
            .unwrap();
        let restored = restorer.restore(batch).unwrap();

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

        let with_blanks = add_blanks(batch.clone(), &[5, 7]).unwrap();

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

        let with_blanks = add_blanks(batch, &[0, 11]).unwrap();
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

        let err = add_blanks(batch, batch_offsets).unwrap_err();
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

        let with_blanks = add_blanks(batch, &[5]).unwrap();

        assert_eq!(with_blanks.num_rows(), 6);
        let values = with_blanks.column(0).as_primitive::<Int32Type>();
        for i in 0..5 {
            assert_eq!(values.value(i), i as i32);
        }
        assert_eq!(values.value(5), 0);
    }
}
