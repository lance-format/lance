// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read the files and row metadata that make up a table fragment.

mod columns;
mod deletions;
mod file_reader;
mod read_config;
#[doc(hidden)]
pub mod read_defaults;
mod reader;
pub mod session;
mod updater;
pub(crate) mod write;

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::future::try_join_all;
use futures::{future, join};
use lance_core::datatypes::Schema;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::{DataFile, DeletionFile, Fragment};
use lance_table::io::deletion::deletion_file_path;

use crate::Table;
use crate::io::deletion::read_table_deletion_file;

pub(crate) use columns::discard_staged_file;
#[doc(hidden)]
pub use columns::{duplicate_field_path, relax_nullability, validate_write_version};
pub use file_reader::GenericFileReader;
#[doc(hidden)]
pub use file_reader::V1FragmentReader;
pub use read_config::{BaseSchedulers, FragReadConfig};
pub use reader::FragmentReader;
pub use session::FragmentSession;
#[doc(hidden)]
pub use updater::Updater;
pub use write::FragmentCreateBuilder;
#[doc(hidden)]
pub use write::generate_random_filename;

/// A fragment bound to the table snapshot that supplies its schema and storage.
#[derive(Debug, Clone)]
pub struct FileFragment {
    table: Arc<Table>,
    metadata: Fragment,
}

impl FileFragment {
    /// Bind fragment metadata to a table snapshot for reading.
    pub fn new(table: Arc<Table>, metadata: Fragment) -> Self {
        Self { table, metadata }
    }

    /// Borrow the snapshot that supplies the schema and resolves file locations.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// The table schema associated with this fragment.
    pub fn schema(&self) -> &Schema {
        self.table.schema()
    }

    /// Returns the fragment's metadata.
    pub fn metadata(&self) -> &Fragment {
        &self.metadata
    }

    /// The id of this [`FileFragment`].
    pub fn id(&self) -> usize {
        self.metadata.id as usize
    }

    /// The number of data files in this fragment.
    pub fn num_data_files(&self) -> usize {
        self.metadata.files.len()
    }

    /// Gets the data file for a given field
    pub fn data_file_for_field(&self, field_id: u32) -> Option<&DataFile> {
        self.metadata
            .files
            .iter()
            .find(|f| f.fields.contains(&(field_id as i32)))
    }

    /// Edit staged metadata before publishing a new fragment.
    #[doc(hidden)]
    pub fn metadata_mut(&mut self) -> &mut Fragment {
        &mut self.metadata
    }

    /// Count live rows, falling back to file metadata if counts are missing.
    pub async fn count_rows(&self) -> Result<usize> {
        self.live_rows().await
    }

    /// Get the number of rows after applying the fragment's deletions.
    pub async fn live_rows(&self) -> Result<usize> {
        let (physical_rows, deleted_rows) =
            future::try_join(self.physical_rows(), self.count_deletions()).await?;
        Ok(physical_rows - deleted_rows)
    }

    /// Get the number of rows that have been deleted in this fragment.
    pub async fn count_deletions(&self) -> Result<usize> {
        match &self.metadata().deletion_file {
            Some(DeletionFile {
                num_deleted_rows: Some(num_deleted),
                ..
            }) => Ok(*num_deleted),
            _ => {
                let deletion_vector = self.get_deletion_vector().await?;
                if let Some(deletion_vector) = deletion_vector {
                    Ok(deletion_vector.len())
                } else {
                    Ok(0)
                }
            }
        }
    }

    /// Get the number of physical rows in the fragment synchronously
    ///
    /// Fails if the fragment does not have the physical row count in the metadata.  This method should
    /// only be called in new workflows which are not run on old versions of Lance.
    pub fn fast_physical_rows(&self) -> Result<usize> {
        if self.table.manifest.writer_version.is_some() {
            let Some(physical_rows) = self.metadata.physical_rows else {
                return Err(Error::internal(format!(
                    "The method fast_physical_rows was called on a fragment that does not have the physical row count in the metadata. Fragment id: {}",
                    self.id()
                )));
            };
            Ok(physical_rows)
        } else {
            Err(Error::internal(format!(
                "The method fast_physical_rows was called on a fragment that does not have the physical row count in the metadata. Fragment id: {}",
                self.id()
            )))
        }
    }

    /// Get the number of deleted rows in the fragment synchronously
    ///
    /// Fails if the fragment does not have deletion count in the metadata.  This method should only
    /// be called in new workflows which are not run on old versions of Lance.
    pub fn fast_num_deletions(&self) -> Result<usize> {
        match &self.metadata().deletion_file {
            Some(DeletionFile {
                num_deleted_rows: Some(num_deleted),
                ..
            }) => Ok(*num_deleted),
            None => Ok(0),
            _ => Err(Error::internal(format!(
                "The method fast_num_deletions was called on a fragment that does not have the deletion count in the metadata. Fragment id: {}",
                self.id()
            ))),
        }
    }

    /// Get the number of logical rows (physical rows - deleted rows) in the fragment synchronously
    ///
    /// Fails if the fragment does not have the physical row count or deletion count in the metadata.  This method should only
    /// be called in new workflows which are not run on old versions of Lance.
    pub fn fast_logical_rows(&self) -> Result<usize> {
        let num_physical_rows = self.fast_physical_rows()?;
        let num_deleted_rows = self.fast_num_deletions()?;
        Ok(num_physical_rows - num_deleted_rows)
    }

    /// Get the number of physical rows in this [`FileFragment`].
    ///
    /// Physical rows include deleted rows. If there are no deleted rows, this is
    /// equal to the number of logical rows in the fragment.
    ///
    /// When the dataset manifest contains writer-version metadata, this method
    /// trusts [`Fragment::physical_rows`] when it is present. This includes
    /// file-less fragments, which can represent rows whose columns are all null.
    /// Use [`Self::metadata`] to inspect the persisted [`Fragment`] metadata.
    ///
    /// Legacy manifests without writer-version metadata may contain an incorrect
    /// cached count, so this method ignores it and reads the count from a data
    /// file. If such a fragment has no data files, this method returns
    /// [`Error::NotFound`]. See [`Self::validate`] for the corresponding metadata
    /// checks.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table) -> Result<()> {
    /// for fragment in table.get_fragments() {
    ///     let physical_rows = fragment.physical_rows().await?;
    ///     if fragment.num_data_files() == 0 {
    ///         assert_eq!(fragment.metadata().physical_rows, Some(physical_rows));
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn physical_rows(&self) -> Result<usize> {
        // Early versions that did not write the writer version also could write
        // incorrect `physical_rows` values. So if we don't have a writer version,
        // we should not use the cached value. On write, we update the values
        // in the manifest, fixing the issue for future reads.
        // See: https://github.com/lance-format/lance/issues/1531
        if self.table.manifest.writer_version.is_some()
            && let Some(physical_rows) = self.metadata.physical_rows
        {
            return Ok(physical_rows);
        }

        // Without a trustworthy cached count, a data file must supply the length.
        if self.metadata.files.is_empty() {
            return Err(Error::not_found(format!(
                "Fragment {} does not contain any data",
                self.id()
            )));
        };

        // Just open any file. All of them should have same size.
        let some_file = &self.metadata.files[0];
        let reader = self
            .open_reader(some_file, None, &FragReadConfig::default())
            .await?
            .ok_or_else(|| {
                Error::internal(format!(
                    "The data file {} did not have any fields contained in the dataset schema",
                    some_file.path
                ))
            })?;

        Ok(reader.len() as usize)
    }

    /// Validate this [`FileFragment`] and its persisted [`Fragment`] metadata.
    ///
    /// Verifies:
    /// * All field ids in the fragment are distinct.
    /// * Within each data file, field ids are in increasing order.
    /// * All data files holding user data exist and have the same length. A
    ///   file kept only for the spilled row lineage it carries is not opened;
    ///   [`crate::rowids::validate_stable_row_ids`] reads that lineage back, which checks its length.
    /// * Field ids are distinct between data files.
    /// * The deletion file exists and has row offsets in the correct range.
    /// * [`Fragment::physical_rows`], when present, matches the data-file length.
    /// * [`DeletionFile::num_deleted_rows`], when present, matches the deletion-vector length.
    ///
    /// When no file holds user data, the fragment is accepted if the manifest contains
    /// writer-version metadata, [`Fragment::physical_rows`] is present, and the
    /// remaining metadata is valid. The cached count supplies the expected length
    /// for deletion metadata validation. A missing count returns
    /// [`Error::CorruptFile`].
    ///
    /// For legacy manifests without writer-version metadata, a file-less fragment
    /// returns [`Error::CorruptFile`] even when [`Fragment::physical_rows`] is
    /// present, because the cached count cannot be trusted. See
    /// [`Self::physical_rows`] for the corresponding row-count lookup behavior.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table) -> Result<()> {
    /// for fragment in table.get_fragments() {
    ///     fragment.validate().await?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn validate(&self) -> Result<()> {
        let mut seen_fields = HashSet::new();
        for data_file in &self.metadata.files {
            let last = -1;
            for field_id in data_file.fields.iter() {
                // Negative ids are not schema fields: the tombstone marks a
                // field superseded by a later data file, and the others are
                // hidden system columns such as spilled row lineage. None has
                // an ordering, and a tombstone can repeat.
                if *field_id < 0 {
                    continue;
                }
                if *field_id <= last {
                    return Err(Error::corrupt_file(
                        self.table
                            .data_file_dir(data_file)?
                            .join(data_file.path.as_str()),
                        format!(
                            "Field id {} is not in increasing order in fragment {:#?}",
                            field_id, self
                        ),
                    ));
                }

                if !seen_fields.insert(field_id) {
                    return Err(Error::corrupt_file(
                        self.table
                            .data_file_dir(data_file)?
                            .join(data_file.path.as_str()),
                        format!(
                            "Field id {} is duplicated in fragment {:#?}",
                            field_id, self
                        ),
                    ));
                }
            }
        }

        let mut saw_v1 = false;
        let mut saw_v2 = false;
        for data_file in self.metadata.referenced_lance_files() {
            match data_file.file_version()? {
                ConcreteFileVersion::V1 => saw_v1 = true,
                ConcreteFileVersion::V2_0
                | ConcreteFileVersion::V2_1
                | ConcreteFileVersion::V2_2
                | ConcreteFileVersion::V2_3 => saw_v2 = true,
            }
            if saw_v1 && saw_v2 {
                return Err(Error::corrupt_file(
                    self.table
                        .data_file_dir(data_file)?
                        .join(data_file.path.as_str()),
                    "Fragment mixes V1 and V2 data files",
                ));
            }
        }

        for data_file in &self.metadata.files {
            data_file.validate(&self.table.data_file_dir(data_file)?)?;
        }

        // A file that holds no field of the dataset schema is not opened when
        // it holds no user field at all, or when the fragment keeps it for a
        // spilled row lineage sequence it carries -- a data file whose user
        // columns were all dropped or replaced after compaction wrote the
        // lineage next to them. The sequences it carries are checked against
        // `physical_rows` when they are validated. Any other file is opened,
        // so a file listing only user fields the schema does not have is
        // still reported.
        let schema_field_ids = self
            .table
            .schema()
            .fields_pre_order()
            .map(|field| field.id)
            .collect::<HashSet<_>>();
        let spilled_field_ids = self.metadata.spilled_row_lineage_field_ids();
        let user_data_files = self.metadata.files.iter().filter(|data_file| {
            let fields = &data_file.fields;
            let holds_schema_field = fields.iter().any(|id| schema_field_ids.contains(id));
            let holds_user_field = fields.iter().any(|id| *id >= 0);
            let holds_spilled_lineage = fields.iter().any(|id| spilled_field_ids.contains(id));
            holds_schema_field || (holds_user_field && !holds_spilled_lineage)
        });
        let get_lengths = user_data_files.clone().map(|data_file| async move {
            let data_file_dir = self.table.data_file_dir(data_file)?;
            let reader = self
                .open_reader(data_file, None, &FragReadConfig::default())
                .await?
                .ok_or_else(|| {
                    Error::corrupt_file(
                        data_file_dir.clone().join(data_file.path.as_str()),
                        "did not have any fields in common with the dataset schema",
                    )
                })?;
            Result::Ok(reader.len() as usize)
        });
        let get_lengths = try_join_all(get_lengths);

        let deletion_vector = self.get_deletion_vector();

        let (get_lengths, deletion_vector) = join!(get_lengths, deletion_vector);

        let get_lengths = get_lengths?;
        let expected_length = if let Some(first_length) = get_lengths.first() {
            for (length, data_file) in get_lengths.iter().zip(user_data_files) {
                if length != first_length {
                    let path = self
                        .table
                        .data_file_dir(data_file)?
                        .join(data_file.path.as_str());
                    return Err(Error::corrupt_file(
                        path,
                        format!(
                            "data file has incorrect length. Expected: {} Got: {}",
                            first_length, length
                        ),
                    ));
                }
            }
            if let Some(physical_rows) = self.metadata.physical_rows
                && physical_rows != *first_length
            {
                return Err(Error::corrupt_file(
                    self.table.state.base.clone(),
                    format!(
                        "Fragment metadata has incorrect physical_rows. Fragment: {} Actual: {} Metadata: {}",
                        self.id(),
                        first_length,
                        physical_rows
                    ),
                ));
            }
            *first_length
        } else {
            // No user data file supplied a length, so validation must use a
            // trustworthy persisted count, including when only lineage remains.
            if self.table.manifest.writer_version.is_none() {
                return Err(Error::corrupt_file(
                    self.table.state.base.clone(),
                    format!(
                        "Fragment {} has no user data files; physical_rows={:?} cannot be trusted because manifest.writer_version is missing",
                        self.id(),
                        self.metadata.physical_rows
                    ),
                ));
            }
            self.metadata.physical_rows.ok_or_else(|| {
                Error::corrupt_file(
                    self.table.state.base.clone(),
                    format!(
                        "Fragment {} has no user data files and is missing physical_rows",
                        self.id()
                    ),
                )
            })?
        };

        if let Some(deletion_vector) = deletion_vector? {
            if let Some(num_deletions) = self
                .metadata
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows
                && num_deletions != deletion_vector.len()
            {
                return Err(Error::corrupt_file(
                    deletion_file_path(
                        &self.table.state.base,
                        self.metadata.id,
                        self.metadata.deletion_file.as_ref().unwrap(),
                    ),
                    format!(
                        "deletion vector length does not match metadata. Metadata: {} Deletion vector: {}",
                        num_deletions,
                        deletion_vector.len()
                    ),
                ));
            }

            for offset in deletion_vector.iter() {
                if offset as usize >= expected_length {
                    let deletion_file_meta = self.metadata.deletion_file.as_ref().unwrap();
                    return Err(Error::corrupt_file(
                        deletion_file_path(
                            &self.table.state.base,
                            self.metadata.id,
                            deletion_file_meta,
                        ),
                        format!(
                            "deletion vector contains an offset that is out of range. Offset: {} Fragment length: {}",
                            offset, expected_length
                        ),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Open a [`FragmentSession`], which manages a short-lived session of [`FileFragment`].
    ///
    /// This API works well for users making repeated requests over the same projected schema.
    pub async fn open_session(
        &self,
        projection: &Schema,
        with_row_address: bool,
    ) -> Result<FragmentSession> {
        FragmentSession::open(self, projection, with_row_address).await
    }

    /// Take rows from this fragment based on the offset in the file.
    ///
    /// This will always return the same number of rows as the input indices.
    /// If indices are out-of-bounds, this will return an error.
    ///
    /// The projection may include the `_rowid` and `_rowaddr` system columns
    /// (for example from [`Schema::project_preserve_system_columns`]). They
    /// are returned in the order they appear in the projection.
    pub async fn take(&self, indices: &[u32], projection: &Schema) -> Result<RecordBatch> {
        // System columns are produced by the reader rather than read from data files.
        let mut with_row_id = false;
        let mut with_row_address = false;
        let mut data_fields = Vec::with_capacity(projection.fields.len());
        for field in &projection.fields {
            match field.name.as_str() {
                ROW_ID => with_row_id = true,
                ROW_ADDR => with_row_address = true,
                name if lance_core::is_system_column(name) => {
                    return Err(Error::invalid_input(format!(
                        "System column {name} is not supported by fragment take"
                    )));
                }
                _ => data_fields.push(field.clone()),
            }
        }
        let data_projection = Schema {
            fields: data_fields,
            metadata: projection.metadata.clone(),
        };

        // Re-map the indices to row ids using the deletion vector
        let deletion_vector = self.get_deletion_vector().await?;
        let row_ids = if let Some(deletion_vector) = deletion_vector {
            // Sorting the deletion vector costs O(M log M), where M is the
            // deletion vector length. Each index is then resolved with a
            // binary search over the sorted deletions, for a total of
            // O(N log M + M log M) where N = indices.len().
            let mut sorted_deleted_ids = deletion_vector
                .as_ref()
                .clone()
                .into_iter()
                .collect::<Vec<_>>();
            sorted_deleted_ids.sort();

            Cow::Owned(resolve_actual_row_ids(indices, &sorted_deleted_ids)?)
        } else {
            Cow::Borrowed(indices)
        };

        // Then call take rows. Keep an internal address for empty projections
        // so the batch retains its row count.
        let read_config = FragReadConfig::default()
            .with_row_id(with_row_id)
            .with_row_address(with_row_address || data_projection.fields.is_empty());
        let batch = self
            .take_rows(&row_ids, &data_projection, read_config)
            .await?;

        // Restore the requested column order and drop any internal address.
        let batch_schema = batch.schema();
        let column_indices = projection
            .fields
            .iter()
            .map(|field| batch_schema.index_of(&field.name))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let batch = batch.project(&column_indices)?;
        if projection.fields.is_empty() {
            return Ok(batch);
        }

        // Convert Lance JSON columns (LargeBinary/JSONB) back to Arrow JSON (Utf8)
        // for user-facing output.
        if batch
            .schema()
            .fields()
            .iter()
            .any(|f| lance_arrow::json::is_json_field(f) || lance_arrow::json::has_json_fields(f))
        {
            Ok(lance_arrow::json::convert_lance_json_to_arrow(&batch)?)
        } else {
            Ok(batch)
        }
    }

    /// Get the deletion vector for this fragment, using the cache if available.
    pub async fn get_deletion_vector(&self) -> Result<Option<Arc<DeletionVector>>> {
        let Some(deletion_file) = self.metadata.deletion_file.as_ref() else {
            return Ok(None);
        };

        let deletion_vector =
            read_table_deletion_file(&self.table, self.id() as u64, deletion_file).await?;

        Ok(Some(deletion_vector))
    }

    /// Take rows based on internal local row offsets
    ///
    /// If the row offsets are out-of-bounds, this will return an error. But if the
    /// row offset is marked deleted, it will be ignored. Thus, the number of rows
    /// returned may be less than the number of row offsets provided.
    ///
    /// To recover the original row addresses from the returned RecordBatch, set the
    /// `with_row_address` parameter to true. This will add a column named `_rowaddr`
    /// to the RecordBatch at the end.
    #[doc(hidden)]
    pub async fn take_rows(
        &self,
        row_offsets: &[u32],
        projection: &Schema,
        read_config: FragReadConfig,
    ) -> Result<RecordBatch> {
        let reader = self.open(projection, read_config).await?;

        reader.take_rows(row_offsets).await
    }
}

impl From<FileFragment> for Fragment {
    fn from(fragment: FileFragment) -> Self {
        fragment.metadata
    }
}

///
/// The live row with logical index `i` sits at physical row `i + k`, where `k`
/// is the number of deleted rows before it. The `j`-th deletion (0-based) has
/// `deleted[j] - j` live rows before it, a count that never decreases with
/// `j`, so it comes before the live row exactly when `deleted[j] - j <= i`.
/// One binary search over the sorted deletions finds `k`. For example, if
/// deleted rows are [2, 3, 5], the counts are [2, 2, 3], and row 4 has all
/// three deletions before it, so it is at physical row 7.
pub(crate) fn resolve_actual_row_ids(
    row_ids: &[u32],
    sorted_deleted_ids: &[u32],
) -> Result<Vec<u32>> {
    let mut physical_offsets = Vec::with_capacity(row_ids.len());
    for &row_id in row_ids {
        let logical_index = row_id as usize;
        // Deletion ids are unique and ascending, so `deleted[j] >= j`.
        let (mut lo, mut hi) = (0, sorted_deleted_ids.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if sorted_deleted_ids[mid] as usize - mid <= logical_index {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let physical_offset = u32::try_from(lo)
            .ok()
            .and_then(|deleted_rows| row_id.checked_add(deleted_rows))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Logical row index {row_id} plus {lo} preceding deletions exceeds maximum physical row offset {}",
                    u32::MAX
                ))
            })?;
        physical_offsets.push(physical_offset);
    }
    Ok(physical_offsets)
}

#[cfg(test)]
mod tests;
