// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{FileFragment, FragReadConfig, FragmentReader, resolve_actual_row_ids};
use arrow_array::RecordBatch;
use lance_core::Result;
use lance_core::datatypes::Schema;
use std::borrow::Cow;

/// A [`FragmentSession`] manages a short-lived session of [`FileFragment`], allowing us to maintain
/// internal states instead of creating new ones each time.
///
/// This API works well for users making repeated requests over the same projected schema.
#[derive(Debug)]
pub struct FragmentSession {
    reader: FragmentReader,
    sorted_deleted_ids: Option<Vec<u32>>,
}

impl FragmentSession {
    /// Open a reusable reader for logical takes with the supplied projection.
    pub async fn open(
        fragment: &FileFragment,
        projection: &Schema,
        with_row_address: bool,
    ) -> Result<Self> {
        let reader = fragment
            .open(
                projection,
                FragReadConfig::default().with_row_address(with_row_address),
            )
            .await?;

        let sorted_deleted_ids = fragment.get_deletion_vector().await?.map(|dv| {
            let mut ids = dv.as_ref().clone().into_iter().collect::<Vec<_>>();
            ids.sort();
            ids
        });

        Ok(Self {
            reader,
            sorted_deleted_ids,
        })
    }

    /// Take logical row positions in input order, skipping deleted positions.
    pub async fn take(&self, indices: &[u32]) -> Result<RecordBatch> {
        // Remap row ids if needed
        let row_ids = if let Some(sorted_deleted_ids) = &self.sorted_deleted_ids {
            Cow::Owned(resolve_actual_row_ids(indices, sorted_deleted_ids)?)
        } else {
            Cow::Borrowed(indices)
        };

        // Then call take rows
        let batch = self.take_rows(&row_ids).await?;

        // Convert Lance JSON columns (LargeBinary/JSONB) back to Arrow JSON (Utf8)
        // for user-facing output, mirroring FileFragment::take.
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

    /// Take physical row offsets, omitting rows that were deleted.
    #[doc(hidden)]
    pub async fn take_rows(&self, row_offsets: &[u32]) -> Result<RecordBatch> {
        self.reader.take_rows(row_offsets).await
    }
}
