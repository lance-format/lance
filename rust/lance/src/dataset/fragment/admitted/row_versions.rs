// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Generate selected lineage values without a row-sized metadata expansion.

use std::sync::Arc;

use arrow_array::{ArrayRef, UInt32Array};
use lance_core::{
    Error, ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION, Result,
    utils::allocation::Reservation,
};
use lance_table::rowids::version::{RowDatasetVersionMeta, RowDatasetVersionSequence};

use super::{FragmentReader, row_ids::row_array};

pub(super) struct PreparedRowVersions {
    columns: [Option<usize>; 2],
    sequences: [Option<Arc<RowDatasetVersionSequence>>; 2],
    is_shared: bool,
    credit: Reservation,
}

impl PreparedRowVersions {
    pub(super) fn try_prepare(
        reader: &FragmentReader,
        count: usize,
        workspace: &mut Reservation,
    ) -> Result<Option<Self>> {
        let columns = [
            reader
                .with_row_created_at_version
                .then(|| reader.output_schema.index_of(ROW_CREATED_AT_VERSION))
                .transpose()?,
            reader
                .with_row_last_updated_at_version
                .then(|| reader.output_schema.index_of(ROW_LAST_UPDATED_AT_VERSION))
                .transpose()?,
        ];
        let sequences = [
            reader.created_at_sequence.clone(),
            reader.last_updated_at_sequence.clone(),
        ];
        for ((column, sequence), meta) in columns.iter().zip(&sequences).zip([
            &reader.fragment.metadata.created_at_version_meta,
            &reader.fragment.metadata.last_updated_at_version_meta,
        ]) {
            if column.is_none() {
                continue;
            }
            if !reader.fragment.dataset.manifest.uses_stable_row_ids() {
                return Err(Error::invalid_input(
                    "admitted row versions require stable row IDs",
                ));
            }
            // Ordinary open may have cached a spilled sequence. Its whole-column
            // input is not admitted here, so a cache hit must not bypass this gate.
            if matches!(meta, Some(RowDatasetVersionMeta::Column)) {
                return Err(Error::not_supported(
                    "admitted fragment versions do not yet read spilled version columns",
                ));
            }
            if let Some(sequence) = sequence {
                let length = sequence.runs.iter().try_fold(0u64, |total, run| {
                    total.checked_add(run.len() as u64).ok_or_else(|| {
                        Error::invalid_input("row version sequence length overflows")
                    })
                })?;
                if length != reader.num_physical_rows as u64 {
                    return Err(Error::invalid_input(
                        "row version sequence length differs from fragment physical length",
                    ));
                }
            }
        }
        let is_shared = match (&sequences[0], &sequences[1]) {
            (None, None) => true,
            (Some(left), Some(right)) => Arc::ptr_eq(left, right),
            _ => false,
        };
        let buffers = columns.iter().flatten().count()
            - usize::from(is_shared && columns.iter().all(Option::is_some));
        let bytes = count
            .checked_mul(size_of::<u64>())
            .and_then(|bytes| bytes.checked_mul(buffers))
            .ok_or_else(|| {
                Error::invalid_input("generated fragment versions exceed address space")
            })?;
        let Some(credit) = workspace.try_split(bytes) else {
            return Ok(None);
        };
        Ok(Some(Self {
            columns,
            sequences,
            is_shared,
            credit,
        }))
    }

    pub(super) fn mark_columns(&self, covered: &mut [bool]) {
        for column in self.columns.into_iter().flatten() {
            covered[column] = true;
        }
    }

    pub(super) fn fill_columns(
        mut self,
        indices: &UInt32Array,
        output: &mut [Option<ArrayRef>],
    ) -> Result<()> {
        let mut first: Option<ArrayRef> = None;
        for (column, sequence) in self.columns.into_iter().zip(&self.sequences) {
            let Some(column) = column else {
                continue;
            };
            let array = if self.is_shared
                && let Some(first) = &first
            {
                first.clone()
            } else if let Some(sequence) = sequence {
                row_array(
                    &mut self.credit,
                    indices.len(),
                    selected_versions(sequence, indices),
                    None,
                )?
            } else {
                row_array(
                    &mut self.credit,
                    indices.len(),
                    std::iter::repeat_n(1, indices.len()),
                    None,
                )?
            };
            first = Some(array.clone());
            output[column] = Some(array);
        }
        Ok(())
    }
}

fn selected_versions<'a>(
    sequence: &'a RowDatasetVersionSequence,
    indices: &'a UInt32Array,
) -> impl Iterator<Item = u64> + 'a {
    let mut runs = sequence.runs.iter();
    let mut positions = indices.values().iter();
    let mut end = 0u64;
    let mut version = 0;
    // Sorted physical offsets and checked total length are established before
    // dispatch. Skip whole runs (including empty runs), not every skipped row.
    // No run-offset index or expanded version array is constructed.
    std::iter::from_fn(move || {
        let position = u64::from(*positions.next()?);
        while position >= end {
            let run = runs.next()?;
            end += run.len() as u64;
            version = run.version();
        }
        Some(version)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_table::rowids::{segment::U64Segment, version::RowDatasetVersionRun};

    #[test]
    fn sparse_versions_skip_empty_runs_and_large_physical_gaps() {
        let boundary = u64::from(u32::MAX);
        let sequence = RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..0),
                    version: 99,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..boundary),
                    version: 7,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(boundary..boundary),
                    version: 98,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(boundary..boundary + 1),
                    version: 11,
                },
            ],
        };
        let indices = UInt32Array::from(vec![0, u32::MAX - 1, u32::MAX]);
        assert_eq!(
            selected_versions(&sequence, &indices).collect::<Vec<_>>(),
            vec![7, 7, 11]
        );
        assert_eq!(
            selected_versions(&sequence, &UInt32Array::from(Vec::<u32>::new())).count(),
            0
        );
    }
}
