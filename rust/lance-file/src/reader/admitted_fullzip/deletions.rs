// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::{Error, Result, utils::deletion::DeletionVector};

use super::FullZipScan;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum DeletionMode {
    #[default]
    Filter,
    MarkNull,
}

impl FullZipScan {
    /// Preserve physical rows and mark deleted `_rowid`/`_rowaddr` values null.
    ///
    /// Configure generated row IDs or addresses first, after merging files and
    /// before any read attempt or logical selection. Both enabled identifier
    /// columns share one admitted validity buffer; payload and version columns
    /// keep their stored values. Even fully deleted ranges require payload I/O.
    /// Snapshot identity/domain rules match [`Self::with_deleted_rows`]. This
    /// replaces any previously configured filtering snapshot.
    ///
    /// Use physical selections with [`Self::advance_to_range`]. Live-row range
    /// mapping is rejected in this mode rather than changing the requested row
    /// domain. This does not adopt the cursor in ordinary fragment/scanner reads.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_core::{Result, utils::deletion::DeletionVector};
    /// use lance_file::reader::FullZipScan;
    /// fn mark(scan: FullZipScan, deleted: Arc<DeletionVector>) -> Result<FullZipScan> {
    ///     scan.with_row_address(7)?.with_deleted_rows_as_nulls(deleted)
    /// }
    /// ```
    pub fn with_deleted_rows_as_nulls(self, deleted: Arc<DeletionVector>) -> Result<Self> {
        if self.row_ids.is_none() {
            return Err(Error::invalid_input(
                "FullZip deletion markers require a generated row ID or row address",
            ));
        }
        let mut scan = self.with_deleted_rows(deleted)?;
        scan.deletion_mode = DeletionMode::MarkNull;
        Ok(scan)
    }
}
