// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in exact physical take with base, overlays and output in one admission.

use crate::dataset::overlay::{OverlayBatchSelection, PreparedOverlayBatch};
use lance_core::utils::{allocation::Reservation, tokio::spawn_cpu};

use super::*;

mod deletions;

impl FragmentReader {
    /// Take sorted unique physical offsets within one already-admitted workspace.
    ///
    /// This opt-in path uses this fragment's actual file readers, field-ID mapping,
    /// overlay coverage planner and newest-first sources. Base and sparse overlay
    /// inputs prepare asynchronously, then decode/concatenate/interleave together
    /// in one CPU job. Every payload owner remains charged until its final alias
    /// drops; no dependent stage waits for more credit or uses ordinary decoding.
    /// `None` means discard/replan the complete attempt, never EOF/partial rows.
    /// No implicit prefix reduction or cursor is involved. Cancellation releases
    /// owners when their actual I/O/CPU work ends.
    ///
    /// Currently requires dense FullZip or byte-scalar Constant v2.2/v2.3 pages
    /// with byte/string leaves optionally nested in List/LargeList, with disjoint
    /// top-level fields across base files.
    /// Flat all-null byte/string pages require metadata-only validity. V2_3 List
    /// chains with metadata-proven null leaves support metadata or
    /// Flat/Bitpacked/Delta parent validity, nonempty positions and counts.
    /// RLE counts also support windowable unsigned value/run-length children.
    /// Dictionary counts charge the bounded lookup table and windowed children.
    /// Payloads use bounded input windows and selected charged storage.
    /// Other sparse codecs return `NotSupported` when selected.
    /// Stored layout determines decoder dispatch.
    /// Deleted physical offsets are removed before base/overlay reads using the
    /// reader's immutable deletion snapshot. A reduced selection's offsets share
    /// this workspace; output contains only surviving rows in input order. Empty
    /// results still charge their typed offset buffers. Deletion metadata itself
    /// is borrowed, not admitted here. This is not logical-offset remapping.
    /// Struct ancestors, deletion markers, generated system columns,
    /// missing-column synthesis and legacy readers are explicitly unsupported.
    /// No query/scanner default is changed. The caller accounts for the borrowed
    /// selection's physical storage; metadata/transport/cache/allocator overhead
    /// remain outside this payload bound. Performance is not qualified.
    ///
    /// ```
    /// use arrow_array::{RecordBatch, UInt32Array};
    /// use lance_core::{Result, utils::allocation::AllocationBudget};
    /// use lance::dataset::fragment::FragmentReader;
    /// async fn take(reader: &FragmentReader, indices: &UInt32Array)
    ///     -> Result<Option<RecordBatch>> {
    ///     let budget = AllocationBudget::new(65536);
    ///     let work = budget.try_reserve(65536).unwrap();
    ///     reader.try_take_admitted_fullzip(indices, work).await
    /// }
    /// ```
    pub async fn try_take_admitted_fullzip(
        &self,
        indices: &UInt32Array,
        mut workspace: Reservation,
    ) -> Result<Option<RecordBatch>> {
        let mut read_selection =
            OverlayReadSelection::from_sorted_indices(indices, self.num_physical_rows as u64)?;
        if self.num_system_cols() != 0 || self.make_deletions_null {
            return Err(Error::not_supported(
                "admitted fragment take does not yet compose deletion markers or generated columns",
            ));
        }
        // Validate the original selection above: deletion pruning must not hide
        // duplicates, null indices or physical offsets outside this fragment.
        let Some(live_indices) =
            deletions::try_filter_in(indices, self.deletion_vec.as_deref(), &mut workspace)?
        else {
            return Ok(None);
        };
        if live_indices.len() != indices.len() {
            read_selection = OverlayReadSelection::from_sorted_indices(
                &live_indices,
                self.num_physical_rows as u64,
            )?;
        }
        let indices = live_indices;
        let schema = Arc::new(self.output_schema.clone());
        // Resolve file-to-output placement before payload reads. Shared struct
        // spines need admitted ancestor reconstruction, never merge_streams here.
        let mut file_columns = Vec::new();
        let mut covered = vec![false; schema.fields().len()];
        for reader in &self.readers {
            if reader.projection().fields.is_empty() {
                continue;
            }
            let mut columns = Vec::with_capacity(reader.projection().fields.len());
            for field in &reader.projection().fields {
                let column = schema.index_of(&field.name).map_err(Error::from)?;
                if covered[column] {
                    return Err(Error::not_supported(
                        "admitted fragment take requires disjoint top-level base fields",
                    ));
                }
                covered[column] = true;
                columns.push(column);
            }
            file_columns.push((reader, columns));
        }
        if covered.is_empty() || covered.iter().any(|covered| !covered) {
            return Err(Error::not_supported(
                "admitted fragment take requires stored payload for every output field",
            ));
        }
        let selection = OverlayBatchSelection::Indices(indices.clone());
        let plans = if let Some(overlay) = &self.overlay {
            resolve_overlays(
                &overlay.planner,
                &read_selection,
                &overlay.fragment,
                &overlay.read_config,
            )
            .await?
        } else {
            Vec::new()
        };
        let Some(overlays) =
            PreparedOverlayBatch::try_prepare_in(&selection, &plans, &schema, &mut workspace)
                .await?
        else {
            return Ok(None);
        };
        let mut base_inputs = Vec::with_capacity(file_columns.len());
        for (reader, columns) in file_columns {
            let Some(input) = reader
                .prepare_admitted_fullzip_take(
                    indices.values(),
                    reader.projection().clone(),
                    &mut workspace,
                )
                .await?
            else {
                return Ok(None);
            };
            base_inputs.push((input, columns));
        }
        spawn_cpu(move || {
            // Keep a copied selection's physical owner in the same admitted
            // operation through output construction, alongside decoded inputs.
            let rows = indices.len();
            let mut columns = vec![None; schema.fields().len()];
            for (input, placement) in base_inputs {
                let Some(batch) = input.decode_in(&mut workspace)? else {
                    return Ok(None);
                };
                if batch.num_rows() != rows || batch.num_columns() != placement.len() {
                    return Err(Error::invalid_input(
                        "admitted base file output differs from its selection/projection",
                    ));
                }
                for (array, column) in batch.columns().iter().zip(placement) {
                    columns[column] = Some(array.clone());
                }
            }
            let columns = columns
                .into_iter()
                .map(|column| {
                    column
                        .ok_or_else(|| Error::internal("admitted base field was not reconstructed"))
                })
                .collect::<Result<Vec<_>>>()?;
            let base = RecordBatch::try_new(schema, columns)?;
            overlays.decode_and_merge_in(base, &mut workspace)
        })
        .await
    }
}
