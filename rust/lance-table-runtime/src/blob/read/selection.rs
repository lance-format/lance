// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, RecordBatch, StructArray};
use lance_core::datatypes::{BlobVersion, parse_field_path};
use lance_core::{Error, ROW_ADDR, Result};

use super::plan::ResolvedBlobSelection;
use crate::Table;
use crate::blob::file::BlobFile;
use crate::blob::locate::{
    blob_version_from_descriptions, collect_blob_files_v1, collect_blob_v2_descriptor_files,
};
use crate::take::{
    MissingRowPolicy, TakeOptions, TakeProjection, TakeSelection, row_offsets_to_row_addresses,
};

impl Table {
    /// Take lazy blob handles by stable row ID, or by address on tables without
    /// stable IDs. A missing or deleted row returns an error.
    ///
    /// The result follows request order, preserving duplicates. Null blobs are
    /// returned as `None` and valid empty blobs have a handle of size zero.
    ///
    /// ```
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table, row_id: u64) -> lance_core::Result<()> {
    /// for blob in table.take_blobs(&[row_id], "images").await?.into_iter().flatten() {
    ///     let bytes = blob.read().await?;
    /// #   let _ = bytes;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs(
        &self,
        row_ids: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        self.take_blob_handles(TakeSelection::RowIds(row_ids), column.as_ref())
            .await
    }

    /// Take lazy blob handles by physical row address. Addresses encode a
    /// fragment ID and a physical row offset. Deleted or invalid addresses
    /// return an error.
    ///
    /// ```
    /// # use lance_core::utils::address::RowAddress;
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table, fragment_id: u32) -> lance_core::Result<()> {
    /// let address = RowAddress::new_from_parts(fragment_id, 0);
    /// let handles = table.take_blobs_by_addresses(&[address.into()], "images").await?;
    /// # let _ = handles;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs_by_addresses(
        &self,
        row_addrs: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        self.take_blob_handles(TakeSelection::RowAddresses(row_addrs), column.as_ref())
            .await
    }

    /// Take lazy blob handles by zero-based position among the snapshot's live rows.
    ///
    /// ```
    /// # use lance_table_runtime::Table;
    /// # async fn example(table: &Table) -> lance_core::Result<()> {
    /// let handles = table.take_blobs_by_indices(&[0, 1], "images").await?;
    /// assert_eq!(handles.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs_by_indices(
        &self,
        indices: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        let addresses = row_offsets_to_row_addresses(&self.get_fragments(), indices).await?;
        self.take_blobs_by_addresses(&addresses, column).await
    }

    async fn take_blob_handles(
        &self,
        selection: TakeSelection<'_>,
        column: &str,
    ) -> Result<Vec<Option<BlobFile>>> {
        let field_id = validate_blob_column(self, column)?;
        let collected = collect_blob_selection(self, field_id, column, selection).await?;
        Ok(collected
            .into_iter()
            .map(|selection| selection.file)
            .collect())
    }
}

/// Validate that `column` exists and is a blob column, returning its field id.
pub(super) fn validate_blob_column(table: &Table, column: &str) -> Result<u32> {
    let schema = table.schema();
    let blob_field = schema
        .field(column)
        .ok_or_else(|| Error::field_not_found(column, schema.field_paths()))?;
    if !blob_field.is_blob() {
        return Err(Error::invalid_input_source(
            format!("the column '{}' is not a blob column", column).into(),
        ));
    }
    Ok(blob_field.id as u32)
}

/// Resolve every row address and non-null payload in the selection.
pub(super) async fn collect_blob_selection(
    table: &Table,
    blob_field_id: u32,
    column: &str,
    selection: TakeSelection<'_>,
) -> Result<Vec<ResolvedBlobSelection>> {
    let projection = table.schema().project(&[column])?;
    let projection = TakeProjection::try_new(Arc::new(table.clone()), &projection)?;
    let description_and_addr = table
        .take_with_options(
            selection,
            &projection,
            TakeOptions {
                with_row_address: true,
                missing_row_policy: MissingRowPolicy::Error,
                ..Default::default()
            },
        )
        .await?;

    if description_and_addr.num_rows() == 0 {
        return Ok(Vec::new());
    }

    let descriptions = leaf_descriptor_struct(&description_and_addr, column)?;
    let row_addrs = description_and_addr
        .column_by_name(ROW_ADDR)
        .ok_or_else(|| Error::internal("Blob descriptor take did not return row addresses"))?
        .as_primitive::<UInt64Type>();

    let files = match blob_version_from_descriptions(descriptions)? {
        BlobVersion::V1 => {
            collect_blob_files_v1(table, blob_field_id, descriptions, row_addrs).await
        }
        BlobVersion::V2 => {
            collect_blob_v2_descriptor_files(table, blob_field_id, descriptions, row_addrs.values())
                .await
        }
    }?;
    if files.len() != row_addrs.len() {
        return Err(Error::internal(format!(
            "Resolved blob file count {} did not match row address count {}",
            files.len(),
            row_addrs.len()
        )));
    }
    Ok(row_addrs
        .values()
        .iter()
        .copied()
        .zip(files)
        .map(|(row_address, file)| ResolvedBlobSelection {
            row_address,
            file,
            requested_range: None,
        })
        .collect())
}

/// Walk into the descriptor `RecordBatch` at `column` and return the leaf
/// descriptor `StructArray`, descending through nested struct children for
/// dotted paths.
fn leaf_descriptor_struct<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a StructArray> {
    let current = leaf_descriptor_array(batch, column)?;
    current
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            Error::invalid_input_source(
                format!(
                    "Blob column '{}' expected descriptor struct but got {}",
                    column,
                    current.data_type()
                )
                .into(),
            )
        })
}

fn leaf_descriptor_array<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a dyn Array> {
    let path = parse_field_path(column)?;
    let mut current: &dyn Array = batch
        .column_by_name(&path[0])
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob column '{}' was not found in descriptor batch",
                column
            ))
        })?
        .as_ref();
    for segment in &path[1..] {
        let struct_array = current
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                Error::invalid_input_source(
                    format!(
                        "Blob column path '{}' expected struct before segment '{}' but got {}",
                        column,
                        segment,
                        current.data_type()
                    )
                    .into(),
                )
            })?;
        current = struct_array
            .column_by_name(segment)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Blob column path '{}' missing segment '{}'",
                    column, segment
                ))
            })?
            .as_ref();
    }
    Ok(current)
}
