// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in exact physical take with base, overlays and output in one admission.

use crate::dataset::overlay::{OverlayBatchSelection, PreparedOverlayBatch};
use lance_core::utils::{allocation::Reservation, tokio::spawn_cpu};

use super::*;

mod deletions;
mod row_ids;
mod row_versions;
mod scan;
pub use scan::{AdmittedFragmentScan, AdmittedFragmentScanOutcome};

impl FragmentReader {
    /// Take sorted unique physical offsets within one already-admitted workspace.
    ///
    /// This opt-in path uses this fragment's actual file readers, field-ID mapping,
    /// overlay coverage planner and newest-first sources. Base and sparse overlay
    /// inputs prepare asynchronously, then decode/concatenate/interleave together
    /// in one CPU job. Small shallow raw batches without overlays or generated
    /// columns can decode inline, with cooperative yielding. Every payload owner
    /// remains charged until its final alias
    /// drops; no dependent stage waits for more credit or uses ordinary decoding.
    /// `None` means discard/replan the complete attempt, never EOF/partial rows.
    /// No implicit prefix reduction or cursor is involved. Cancellation releases
    /// owners when their actual I/O/CPU work ends.
    ///
    /// Currently requires dense FullZip or byte-scalar Constant v2.2/v2.3 pages
    /// with byte/string leaves optionally nested in List/LargeList, with disjoint
    /// top-level fields across base files. MiniBlock byte/string and nested
    /// List/LargeList pages also support two-word repetition indexes, without page
    /// dictionaries or deeper index formats.
    /// Flat all-null byte/string pages require metadata-only validity. V2_3 List
    /// chains with metadata-proven null leaves support metadata or
    /// Flat/Bitpacked/Delta parent validity, nonempty positions and counts.
    /// RLE counts also support windowable unsigned value/run-length children.
    /// Dictionary counts charge the bounded lookup table and windowed children.
    /// Payloads use bounded input windows and selected charged storage.
    /// Other sparse codecs return `NotSupported` when selected.
    /// Stored layout determines decoder dispatch.
    /// By default, deleted physical offsets are removed before base/overlay
    /// reads using the reader's immutable deletion snapshot. Reduced offsets share
    /// this workspace; output contains only surviving rows in input order. Empty
    /// results still charge their typed offset buffers. Deletion metadata itself
    /// is borrowed, not admitted here. This is not logical-offset remapping.
    /// With deletion markers enabled, physical rows and payload remain intact;
    /// deleted rows have null identifiers. Their shared bitmap is admitted with
    /// the output values. Markers require `_rowid` or `_rowaddr`.
    /// `_rowid` and `_rowaddr` use selected physical offsets. Stable IDs use
    /// the reader's inline row-ID sequence snapshot; spilled ID columns are not
    /// supported. Shared sequence metadata is outside this payload budget, but
    /// generated values reserve credit alongside inputs and retain it through
    /// their final Arrow alias. Non-stable IDs and addresses share one buffer.
    /// Row-version outputs use inline reader snapshots (or version 1 when absent),
    /// with credit reserved before payload reads. Deletion markers do not null
    /// version values. Shared inline metadata is outside this payload budget;
    /// spilled version columns are unsupported even if already cached by open.
    /// Missing top-level columns use typed all-null placeholders with child and
    /// bitmap buffers charged to the same workspace. Missing children within a
    /// stored struct still require unsupported ancestor reconstruction.
    /// Struct ancestors and legacy readers are explicitly unsupported.
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
        if self.make_deletions_null && !self.with_row_id && !self.with_row_addr {
            return Err(Error::invalid_input(
                "deletion markers require with_row_id or with_row_address",
            ));
        }
        // Validate the original selection above: deletion pruning must not hide
        // duplicates, null indices or physical offsets outside this fragment.
        let live_indices = if self.make_deletions_null {
            indices.clone()
        } else {
            let Some(live_indices) =
                deletions::try_filter_in(indices, self.deletion_vec.as_deref(), &mut workspace)?
            else {
                return Ok(None);
            };
            live_indices
        };
        if live_indices.len() != indices.len() {
            read_selection = OverlayReadSelection::from_sorted_indices(
                &live_indices,
                self.num_physical_rows as u64,
            )?;
        }
        let indices = live_indices;
        let schema = Arc::new(self.output_schema.clone());
        let Some(row_versions) =
            row_versions::PreparedRowVersions::try_prepare(self, indices.len(), &mut workspace)?
        else {
            return Ok(None);
        };
        let Some(row_ids) =
            row_ids::PreparedRowIds::try_prepare(self, &indices, &mut workspace).await?
        else {
            return Ok(None);
        };
        // Resolve file-to-output placement before payload reads. Shared struct
        // spines need admitted ancestor reconstruction, never merge_streams here.
        let mut file_columns = Vec::new();
        let mut covered = vec![false; schema.fields().len()];
        row_ids.mark_columns(&mut covered);
        row_versions.mark_columns(&mut covered);
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
                "admitted fragment take requires stored payload or generated columns for every output field",
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
        // Avoid a thread handoff for small known-linear decodes. The estimate
        // charges structural passes and per-row overhead, not compressed size.
        // Unknown codecs, concatenation and output assembly stay on the pool.
        const MAX_INLINE_WORK: usize = 256 * 1024;
        let inline = plans.is_empty()
            && !self.with_row_id
            && !self.with_row_addr
            && !self.with_row_created_at_version
            && !self.with_row_last_updated_at_version
            && base_inputs
                .iter()
                .try_fold(0usize, |total, (input, _)| {
                    total.checked_add(input.decode_work_estimate()?)
                })
                .is_some_and(|work| work <= MAX_INLINE_WORK);
        let decode = move || {
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
            row_ids.fill_columns(&indices, &mut columns)?;
            row_versions.fill_columns(&indices, &mut columns)?;
            let columns = columns
                .into_iter()
                .map(|column| {
                    column
                        .ok_or_else(|| Error::internal("admitted base field was not reconstructed"))
                })
                .collect::<Result<Vec<_>>>()?;
            let base = RecordBatch::try_new(schema, columns)?;
            overlays.decode_and_merge_in(base, &mut workspace)
        };
        if inline {
            // Repeated ready reads must still give other async tasks a turn.
            tokio::task::consume_budget().await;
            decode()
        } else {
            spawn_cpu(decode).await
        }
    }
}
