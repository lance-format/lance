// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Set-valued capture resolution, with bounded positional lookup buffers.

use arrow_array::{UInt64Array, cast::AsArray, types::UInt64Type};
use futures::TryStreamExt;
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_table::format::ROW_ID_FIELD_ID;
use lance_table::rowids::{RowIdSequence, segment::U64Segment};
use roaring::RoaringTreemap;

use super::PreparedLookup;
use crate::Dataset;

// Captures can cover millions of rows. Only this many keys and positional
// matches may be expanded at once, independent of the capture's total length.
const CAPTURE_LOOKUP_BATCH_SIZE: usize = 8192;

fn row_id_set(sequence: &RowIdSequence) -> RoaringTreemap {
    let mut ids = RoaringTreemap::new();
    for segment in sequence.segments() {
        match segment {
            U64Segment::Range(range) => {
                ids.insert_range(range.clone());
            }
            _ => ids.extend(segment.iter()),
        }
    }
    ids
}

impl Dataset {
    pub(crate) async fn resolve_captured_row_ids(
        &self,
        sequence: &RowIdSequence,
    ) -> Result<RoaringTreemap> {
        let mut addresses = RoaringTreemap::new();
        if sequence.is_empty() {
            return Ok(addresses);
        }
        if !self.manifest.uses_stable_row_ids() {
            return Ok(row_id_set(sequence));
        }
        let cell = self.lookup_cache.plan(self, ROW_ID)?;
        let plan = cell
            .get_or_try_init(|| PreparedLookup::new(self, ROW_ID, ROW_ID_FIELD_ID))
            .await?;

        // Only a scan needs membership for the whole capture. Keep both the
        // request and the unresolved IDs compressed, rather than one entry in
        // a positional map for each selected row.
        let requested = (!plan.uncovered.is_empty()).then(|| row_id_set(sequence));
        let mut remaining = requested.clone();
        if remaining.is_none() || !plan.ranges.is_empty() || !plan.segments.is_empty() {
            let mut ids = sequence.iter();
            loop {
                let mut batch = Vec::with_capacity(CAPTURE_LOOKUP_BATCH_SIZE);
                batch.extend(ids.by_ref().take(CAPTURE_LOOKUP_BATCH_SIZE));
                if batch.is_empty() {
                    break;
                }
                let keys = UInt64Array::from(batch);
                let matches = plan.lookup_indexed(self, ROW_ID, &keys).await?;
                for (&row_id, address) in keys.values().iter().zip(matches) {
                    if let Some(address) = address {
                        addresses.insert(address.into());
                        if let Some(remaining) = &mut remaining {
                            remaining.remove(row_id);
                        }
                    } else if remaining.is_none() {
                        return Err(Error::internal(format!(
                            "Captured row ID {row_id} has no live address in dataset version {}",
                            self.version().version
                        )));
                    }
                }
            }
        }

        if let (Some(requested), Some(mut remaining)) = (requested, remaining) {
            let mut scan = self.scan();
            scan.with_fragments(plan.uncovered.clone())
                .with_row_address()
                .use_scalar_index(false)
                .batch_size(CAPTURE_LOOKUP_BATCH_SIZE)
                .batch_readahead(1)
                .project_with_transform(&[(VALUE_COLUMN_NAME, ROW_ID)])?;
            let mut stream = Box::pin(scan.try_into_stream()).await?;
            while let Some(batch) = stream.try_next().await? {
                let ids = batch
                    .column_by_name(VALUE_COLUMN_NAME)
                    .ok_or_else(|| Error::internal("Capture scan omitted the row ID column"))?
                    .as_primitive::<UInt64Type>();
                let row_addresses = batch
                    .column_by_name(ROW_ADDR)
                    .ok_or_else(|| Error::internal("Capture scan omitted the row address column"))?
                    .as_primitive::<UInt64Type>();
                for (row_id, address) in ids.iter().zip(row_addresses.iter()) {
                    let (Some(row_id), Some(address)) = (row_id, address) else {
                        continue;
                    };
                    if requested.contains(row_id) {
                        // The plan partitions fragments between indexed/inline
                        // lookup and this single scan. Another live occurrence
                        // therefore names a different physical row.
                        if !remaining.remove(row_id) {
                            return Err(Error::invalid_input(format!(
                                "Captured row ID {row_id} is not unique: multiple live addresses in dataset version {}",
                                self.version().version
                            )));
                        }
                        addresses.insert(address);
                    }
                }
            }
            if let Some(row_id) = remaining.iter().next() {
                return Err(Error::internal(format!(
                    "Captured row ID {row_id} has no live address in dataset version {}",
                    self.version().version
                )));
            }
        }
        Ok(addresses)
    }
}
