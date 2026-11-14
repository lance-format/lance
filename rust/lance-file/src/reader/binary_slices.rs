// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use bytes::Bytes;
use lance_core::{Error, Result};
use lance_encoding::encodings::logical::primitive::MiniBlockScheduler;

use super::FileReader;

impl FileReader {
    /// Read selected byte ranges within one flat binary value.
    ///
    /// `column_index` is a physical column index and `row` is relative to that
    /// column, including for files whose columns have different lengths. Byte
    /// ranges are relative to the value, may overlap, and preserve input order.
    /// Empty ranges produce empty buffers. This optional path returns `None`
    /// for unsupported encodings; callers must retain ordinary decoding as a
    /// fallback. It does not change the stored format or initialize a scan.
    ///
    /// Returned bytes may share coalesced I/O backing. Their visible lengths
    /// alone do not establish retained allocation sizes.
    ///
    /// ```
    /// # use lance_file::reader::FileReader;
    /// # async fn example(reader: &FileReader) -> lance_core::Result<()> {
    /// let selected = reader.try_read_binary_value_slices(0, 2, &[4..12]).await?;
    /// if let Some(selected) = selected {
    ///     assert_eq!(selected[0].len(), 8);
    /// }
    /// # Ok(()) }
    /// ```
    pub async fn try_read_binary_value_slices(
        &self,
        column_index: usize,
        row: u64,
        slices: &[Range<u64>],
    ) -> Result<Option<Vec<Bytes>>> {
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
                return MiniBlockScheduler::try_read_binary_value_slices(
                    page,
                    page_row,
                    slices,
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
