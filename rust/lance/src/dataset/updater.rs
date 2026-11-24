// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use lance_core::{Result, datatypes::Schema};
use lance_file::writer::FileWriterOptions;
use lance_table::format::Fragment;
use lance_table_runtime::fragment::Updater as RuntimeUpdater;

use super::{Dataset, FileFragment};

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
    inner: RuntimeUpdater,
}

impl Updater {
    pub(super) fn from_runtime(fragment: FileFragment, inner: RuntimeUpdater) -> Self {
        Self { fragment, inner }
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
        Box::pin(self.inner.next()).await
    }

    pub(super) fn set_file_writer_options(&mut self, options: FileWriterOptions) {
        self.inner.set_file_writer_options(options);
    }

    pub(super) fn allow_external_blob_outside_bases(&mut self) {
        self.inner.allow_external_blob_outside_bases();
    }

    /// Update one batch.
    ///
    /// An error leaves the updater in an unspecified state: the deletion restorer may
    /// already have advanced past this batch's deleted rows, so retrying the same batch
    /// would count them twice. After an error, drop the updater and start over rather
    /// than calling `update` or `finish` again.
    pub async fn update(&mut self, batch: RecordBatch) -> Result<()> {
        Box::pin(self.inner.update(batch)).await
    }

    /// Update one input batch with output split across several batches.
    ///
    /// The batches together must have as many rows as the batch returned by
    /// [`Self::next`], and must share column types. They reach the writer without
    /// being concatenated, so the output for one input batch may hold more
    /// variable-width data than one array with 32-bit offsets can address. Legacy
    /// files are the exception: each restored batch has to reproduce a row group, so
    /// their batches are concatenated first.
    ///
    /// Errors leave the updater in the same unspecified state as [`Self::update`].
    pub async fn update_batches(&mut self, batches: Vec<RecordBatch>) -> Result<()> {
        Box::pin(self.inner.update_batches(batches)).await
    }

    /// Finish updating this fragment and return its new metadata.
    pub async fn finish(&mut self) -> Result<Fragment> {
        let fragment = Box::pin(self.inner.finish()).await?;
        *self.fragment.metadata_mut() = fragment.clone();
        Ok(fragment)
    }

    pub(super) async fn cleanup_unfinished_writer(&mut self) {
        Box::pin(self.inner.cleanup_unfinished_writer()).await;
    }

    /// Get the final schema of the fragment after the update.
    ///
    /// This may be None if the schema is not known. This can happen if it was
    /// not specified up front and the first batch of results has not yet been
    /// processed.
    pub fn schema(&self) -> Option<&Schema> {
        self.inner.schema()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use lance_file::version::LanceFileVersion;

    use crate::Error;
    use crate::dataset::{Dataset, WriteParams};

    fn batch_of(field: Field, array: ArrayRef) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![array]).unwrap()
    }

    async fn int_dataset(
        uri: &str,
        rows: i32,
        version: LanceFileVersion,
        max_rows_per_group: usize,
    ) -> Dataset {
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..rows))],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri,
            Some(WriteParams {
                data_storage_version: Some(version),
                max_rows_per_group,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    fn int_output(name: &str, values: impl IntoIterator<Item = i32>) -> RecordBatch {
        batch_of(
            Field::new(name, DataType::Int32, false),
            Arc::new(Int32Array::from_iter_values(values)),
        )
    }

    #[tokio::test]
    async fn test_update_batches_rejects_mismatched_batches() {
        let dataset = int_dataset(
            "memory://update-batches-mismatch",
            4,
            LanceFileVersion::V2_2,
            1024,
        )
        .await;
        let fragment = dataset.get_fragment(0).unwrap();
        let mut updater = fragment
            .updater::<String>(None, None, None, None)
            .await
            .unwrap();
        updater.next().await.unwrap().unwrap();

        let error = updater.update_batches(vec![]).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("empty batch list"), "{error}");

        let output = int_output("o", 0..4);
        let error = updater
            .update_batches(vec![output.slice(0, 1), output.slice(1, 1)])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("4 != 2"), "{error}");

        let strings = batch_of(
            Field::new("o", DataType::Utf8, false),
            Arc::new(StringArray::from(vec!["a", "b"])),
        );
        let error = updater
            .update_batches(vec![output.slice(0, 2), strings])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("column types"), "{error}");
    }

    /// Each restored legacy batch must match the file's row group size. Restoring the
    /// two halves of the first row group separately would fail that check, since the
    /// deleted row 7 keeps the restorer from treating either half as the last batch.
    #[tokio::test]
    async fn test_update_batches_concatenates_legacy_row_groups() {
        let mut dataset = int_dataset(
            "memory://update-batches-legacy",
            8,
            LanceFileVersion::Legacy,
            4,
        )
        .await;
        dataset.delete("i = 7").await.unwrap();
        let fragment = dataset.get_fragment(0).unwrap();
        let mut updater = fragment
            .updater::<String>(None, None, None, None)
            .await
            .unwrap();

        assert_eq!(updater.next().await.unwrap().unwrap().num_rows(), 4);
        let output = int_output("o", 0..4);
        updater
            .update_batches(vec![output.slice(0, 2), output.slice(2, 2)])
            .await
            .unwrap();

        assert_eq!(updater.next().await.unwrap().unwrap().num_rows(), 3);
        updater.update(int_output("o", 4..7)).await.unwrap();
        assert!(updater.next().await.unwrap().is_none());

        let updated = updater.finish().await.unwrap();
        assert_eq!(updated.files.len(), 2);
    }
}
