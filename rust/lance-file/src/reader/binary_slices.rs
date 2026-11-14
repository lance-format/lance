// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use lance_encoding::encodings::{
    logical::primitive::MiniBlockScheduler, physical::binary::PreparedBinaryValue,
};

use super::FileReader;

impl FileReader {
    /// Prepare byte reads for one flat binary value without loading its payload.
    ///
    /// `column_index` is a physical column index and `row` is relative to that
    /// column, including ragged files. The handle exposes the exact value length
    /// and reuses validated metadata for repeated byte reads. Unsupported
    /// encodings return `None`, requesting ordinary decoding by the caller.
    /// Handles retain this reader's I/O scope; do not share request-specific
    /// handles through a global cache. No persisted format changes.
    ///
    /// ```
    /// # use lance_file::reader::FileReader;
    /// # async fn example(reader: &FileReader) -> lance_core::Result<()> {
    /// if let Some(value) = reader.try_prepare_binary_value(0, 2).await? {
    ///     let tail = value.read_slices(&[value.len().saturating_sub(8)..value.len()]).await?;
    ///     assert!(tail[0].len() <= 8);
    /// }
    /// # Ok(()) }
    /// ```
    pub async fn try_prepare_binary_value(
        &self,
        column_index: usize,
        row: u64,
    ) -> Result<Option<PreparedBinaryValue>> {
        let column = self
            .metadata
            .column_infos
            .get(column_index)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "binary byte read column index {column_index} exceeds {} physical columns",
                    self.metadata.column_infos.len()
                ))
            })?;
        let mut page_row = row;
        for page in column.page_infos.iter() {
            if page_row < page.num_rows {
                return MiniBlockScheduler::try_prepare_binary_value(
                    page,
                    page_row,
                    &self.core.scheduler,
                )
                .await;
            }
            page_row -= page.num_rows;
        }
        Err(Error::invalid_input(format!(
            "binary byte read row {row} is outside physical column {column_index}"
        )))
    }
}
