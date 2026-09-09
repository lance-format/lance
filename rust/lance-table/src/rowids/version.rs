// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row version tracking for cross-version diff functionality
//!
//! This module provides data structures and functionality to track the latest
//! update version for each row in a Lance dataset, enabling efficient
//! cross-version diff operations.

use std::{ops::Range, sync::Arc};

use lance_core::Error;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;
use prost::Message;
use serde::de::Deserializer;
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

use crate::format::{ExternalFile, Fragment, pb};
use crate::rowids::segment::U64Segment;
use crate::rowids::{RowIdSequence, read_row_ids};

/// A run of identical versions over a contiguous span of row positions.
///
/// Span is expressed as a U64Segment over row offsets (0..N within a fragment),
/// not over row IDs. This keeps the encoding aligned with RowIdSequence order
/// and enables zipped iteration without building a map.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub struct RowDatasetVersionRun {
    pub span: U64Segment,
    pub version: u64,
}

impl RowDatasetVersionRun {
    /// Number of rows covered by this run.
    pub fn len(&self) -> usize {
        self.span.len()
    }

    /// Whether this run covers no rows.
    pub fn is_empty(&self) -> bool {
        self.span.is_empty()
    }

    /// The version value of this run.
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// Sequence of dataset versions
///
/// Stores version runs aligned to the positional order of RowIdSequence.
/// Provides sequential iterators and optional lightweight indexing for
/// efficient random access.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf, Default)]
pub struct RowDatasetVersionSequence {
    pub runs: Vec<RowDatasetVersionRun>,
}

/// A reusable cursor for reading ranges from a version sequence in one pass.
///
/// The cursor caches the current run length. Readers normally request adjacent
/// batches, so this avoids rebuilding all run offsets and rescanning the run
/// prefix for every batch. A backwards selection lazily builds a run-offset
/// index. Once built, non-adjacent selections use it in either direction.
#[derive(Debug, Default)]
pub(crate) struct RowDatasetVersionCursor {
    run_index: usize,
    offset_in_run: usize,
    position: usize,
    run_len: Option<usize>,
    run_offsets: Option<Vec<usize>>,
    indexed_total_len: usize,
    #[cfg(test)]
    indexed_seek_count: usize,
}

impl RowDatasetVersionCursor {
    fn seek_indexed(
        &mut self,
        sequence: &RowDatasetVersionSequence,
        position: usize,
    ) -> Result<()> {
        if self.run_offsets.is_none() {
            let mut total_len = 0;
            let run_offsets = sequence
                .runs
                .iter()
                .map(|run| {
                    let offset = total_len;
                    total_len += run.len();
                    offset
                })
                .collect();
            self.run_offsets = Some(run_offsets);
            self.indexed_total_len = total_len;
        }

        if position >= self.indexed_total_len {
            return Err(Error::internal(format!(
                "version column position {} out of range (total_len={})",
                position, self.indexed_total_len
            )));
        }

        let run_offsets = self.run_offsets.as_ref().unwrap();
        let mut run_index = match run_offsets.binary_search(&position) {
            Ok(run_index) => run_index,
            Err(run_index) => run_index - 1,
        };
        while run_index + 1 < run_offsets.len() && run_offsets[run_index + 1] <= position {
            run_index += 1;
        }
        self.run_index = run_index;
        self.offset_in_run = position - run_offsets[run_index];
        self.position = position;
        self.run_len = None;
        #[cfg(test)]
        {
            self.indexed_seek_count += 1;
        }
        Ok(())
    }

    fn current_run<'a>(
        &mut self,
        sequence: &'a RowDatasetVersionSequence,
    ) -> Option<(&'a RowDatasetVersionRun, usize)> {
        loop {
            let run = sequence.runs.get(self.run_index)?;
            let run_len = *self.run_len.get_or_insert_with(|| match &run.span {
                // Version runs are normally positional ranges. Keep this hot
                // path local instead of using the general segment length path.
                U64Segment::Range(range) => (range.end - range.start) as usize,
                span => span.len(),
            });
            if self.offset_in_run < run_len {
                return Some((run, run_len));
            }
            self.run_index += 1;
            self.offset_in_run = 0;
            self.run_len = None;
        }
    }

    /// Append the versions in `selection` to `versions`.
    pub(crate) fn extend_range(
        &mut self,
        sequence: &RowDatasetVersionSequence,
        selection: Range<usize>,
        versions: &mut Vec<u64>,
    ) -> Result<()> {
        if selection.is_empty() {
            return Ok(());
        }
        if selection.start < self.position
            || (self.run_offsets.is_some() && selection.start != self.position)
        {
            self.seek_indexed(sequence, selection.start)?;
        }

        while self.position < selection.start {
            let Some((_, run_len)) = self.current_run(sequence) else {
                return Err(Error::internal(format!(
                    "version column position {} out of range (total_len={})",
                    selection.start, self.position
                )));
            };
            let advance = (selection.start - self.position).min(run_len - self.offset_in_run);
            self.offset_in_run += advance;
            self.position += advance;
        }

        while self.position < selection.end {
            let Some((run, run_len)) = self.current_run(sequence) else {
                return Err(Error::internal(format!(
                    "version column position {} out of range (total_len={})",
                    self.position, self.position
                )));
            };
            let count = (selection.end - self.position).min(run_len - self.offset_in_run);
            versions.extend(std::iter::repeat_n(run.version(), count));
            self.offset_in_run += count;
            self.position += count;
        }
        Ok(())
    }
}

impl RowDatasetVersionSequence {
    /// Create a new empty version sequence
    pub fn new() -> Self {
        Self { runs: Vec::new() }
    }

    /// Create a version sequence with a single uniform run of `row_count` rows.
    pub fn from_uniform_row_count(row_count: u64, version: u64) -> Self {
        if row_count == 0 {
            return Self::new();
        }
        let run = RowDatasetVersionRun {
            span: U64Segment::Range(0..row_count),
            version,
        };
        Self { runs: vec![run] }
    }

    /// Number of rows tracked by this sequence (sum of run lengths).
    pub fn len(&self) -> u64 {
        self.runs.iter().map(|s| s.len() as u64).sum()
    }

    /// Empty if there are no runs or all runs are empty.
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty() || self.runs.iter().all(|s| s.is_empty())
    }

    /// Returns a forward iterator over versions, expanding runs lazily.
    pub fn versions(&self) -> VersionsIter<'_> {
        VersionsIter::new(&self.runs)
    }

    /// Create a reusable cursor for sequential range reads.
    pub(crate) fn cursor(&self) -> RowDatasetVersionCursor {
        RowDatasetVersionCursor::default()
    }

    /// Random access: get the version at global row position `index`.
    pub fn version_at(&self, index: usize) -> Option<u64> {
        let mut offset = 0usize;
        for run in &self.runs {
            let len = run.len();
            if index < offset + len {
                return Some(run.version());
            }
            offset += len;
        }
        None
    }

    /// Get the version associated with a specific row id.
    /// This reconstructs the positional offset from RowIdSequence and then
    /// performs `version_at` lookup.
    pub fn get_version_for_row_id(&self, row_ids: &RowIdSequence, row_id: u64) -> Option<u64> {
        let mut offset = 0usize;
        for seg in &row_ids.0 {
            if seg.range().is_some_and(|r| r.contains(&row_id))
                && let Some(local) = seg.position(row_id)
            {
                return self.version_at(offset + local);
            }
            offset += seg.len();
        }
        None
    }

    /// Convenience: collect row IDs with version strictly greater than `threshold`.
    pub fn rows_with_version_greater_than(
        &self,
        row_ids: &RowIdSequence,
        threshold: u64,
    ) -> Vec<u64> {
        row_ids
            .iter()
            .zip(self.versions())
            .filter_map(|(rid, v)| if v > threshold { Some(rid) } else { None })
            .collect()
    }

    /// Delete rows by positional offsets (e.g., from a deletion vector)
    pub fn mask(&mut self, positions: impl IntoIterator<Item = u32>) -> Result<()> {
        let mut local_positions: Vec<u32> = Vec::new();
        let mut positions_iter = positions.into_iter();
        let mut curr_position = positions_iter.next();
        let mut offset: usize = 0;
        let mut cutoff: usize = 0;

        for run in self.runs.iter_mut() {
            cutoff += run.span.len();
            while let Some(position) = curr_position {
                if position as usize >= cutoff {
                    break;
                }
                local_positions.push(position - offset as u32);
                curr_position = positions_iter.next();
            }

            if !local_positions.is_empty() {
                run.span.mask(local_positions.as_slice());
                local_positions.clear();
            }
            offset = cutoff;
        }

        self.runs.retain(|r| !r.span.is_empty());
        Ok(())
    }
}

/// Iterator over versions expanding runs lazily.
pub struct VersionsIter<'a> {
    runs: &'a [RowDatasetVersionRun],
    run_idx: usize,
    remaining_in_run: usize,
    current_version: u64,
}

impl<'a> VersionsIter<'a> {
    fn new(runs: &'a [RowDatasetVersionRun]) -> Self {
        let mut it = Self {
            runs,
            run_idx: 0,
            remaining_in_run: 0,
            current_version: 0,
        };
        it.advance_run();
        it
    }

    fn advance_run(&mut self) {
        if self.run_idx < self.runs.len() {
            let run = &self.runs[self.run_idx];
            self.remaining_in_run = run.len();
            self.current_version = run.version();
        } else {
            self.remaining_in_run = 0;
        }
    }
}

impl<'a> Iterator for VersionsIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_in_run == 0 {
            // Move to next run
            self.run_idx += 1;
            if self.run_idx >= self.runs.len() {
                return None;
            }
            self.advance_run();
        }
        self.remaining_in_run = self.remaining_in_run.saturating_sub(1);
        Some(self.current_version)
    }
}

/// Metadata about the location of dataset version sequence data
/// Following the same pattern as RowIdMeta
///
/// When stored inline, identical byte sequences are shared across fragments
/// via `Arc<[u8]>` to reduce manifest memory for large tables.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub enum RowDatasetVersionMeta {
    /// Small sequences stored inline in the fragment metadata
    Inline(Arc<[u8]>),
    /// Large sequences stored in external files
    External(ExternalFile),
}

// Custom Serialize: convert Arc<[u8]> to slice for transparent JSON output
impl Serialize for RowDatasetVersionMeta {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        #[serde(untagged)]
        enum Helper<'a> {
            Inline { inline: &'a [u8] },
            External { external: &'a ExternalFile },
        }

        match self {
            Self::Inline(data) => Helper::Inline {
                inline: data.as_ref(),
            }
            .serialize(serializer),
            Self::External(file) => Helper::External { external: file }.serialize(serializer),
        }
    }
}

// Custom Deserialize: read Vec<u8> and convert to Arc<[u8]>
impl<'de> Deserialize<'de> for RowDatasetVersionMeta {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Helper {
            Inline { inline: Vec<u8> },
            External { external: ExternalFile },
        }

        match Helper::deserialize(deserializer)? {
            Helper::Inline { inline } => Ok(Self::Inline(Arc::from(inline))),
            Helper::External { external } => Ok(Self::External(external)),
        }
    }
}

impl RowDatasetVersionMeta {
    /// Create inline metadata from a version sequence
    pub fn from_sequence(sequence: &RowDatasetVersionSequence) -> lance_core::Result<Self> {
        let bytes = write_dataset_versions(sequence);
        Ok(Self::Inline(Arc::from(bytes)))
    }

    /// Create external metadata reference
    pub fn from_external_file(path: String, offset: u64, size: u64) -> Self {
        Self::External(ExternalFile { path, offset, size })
    }

    /// Load the version sequence from this metadata
    pub fn load_sequence(&self) -> lance_core::Result<RowDatasetVersionSequence> {
        match self {
            Self::Inline(data) => read_dataset_versions(data),
            Self::External(_file) => {
                todo!("External file loading not yet implemented")
            }
        }
    }
}

/// Helper function to convert RowDatasetVersionMeta to protobuf format for last_updated_at
pub fn last_updated_at_version_meta_to_pb(
    meta: &Option<RowDatasetVersionMeta>,
) -> Option<pb::data_fragment::LastUpdatedAtVersionSequence> {
    meta.as_ref().map(|m| match m {
        RowDatasetVersionMeta::Inline(data) => {
            pb::data_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(
                data.to_vec(),
            )
        }
        RowDatasetVersionMeta::External(file) => {
            pb::data_fragment::LastUpdatedAtVersionSequence::ExternalLastUpdatedAtVersions(
                pb::ExternalFile {
                    path: file.path.clone(),
                    offset: file.offset,
                    size: file.size,
                },
            )
        }
    })
}

/// Helper function to convert RowDatasetVersionMeta to protobuf format for created_at
pub fn created_at_version_meta_to_pb(
    meta: &Option<RowDatasetVersionMeta>,
) -> Option<pb::data_fragment::CreatedAtVersionSequence> {
    meta.as_ref().map(|m| match m {
        RowDatasetVersionMeta::Inline(data) => {
            pb::data_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(data.to_vec())
        }
        RowDatasetVersionMeta::External(file) => {
            pb::data_fragment::CreatedAtVersionSequence::ExternalCreatedAtVersions(
                pb::ExternalFile {
                    path: file.path.clone(),
                    offset: file.offset,
                    size: file.size,
                },
            )
        }
    })
}

/// Serialize a dataset version sequence to a buffer (following RowIdSequence pattern)
pub fn write_dataset_versions(sequence: &RowDatasetVersionSequence) -> Vec<u8> {
    // Convert to protobuf sequence
    let pb_sequence = pb::RowDatasetVersionSequence {
        runs: sequence
            .runs
            .iter()
            .map(|run| pb::RowDatasetVersionRun {
                span: Some(pb::U64Segment::from(run.span.clone())),
                version: run.version,
            })
            .collect(),
    };

    pb_sequence.encode_to_vec()
}

/// Deserialize a dataset version sequence from bytes (following RowIdSequence pattern)
pub fn read_dataset_versions(data: &[u8]) -> lance_core::Result<RowDatasetVersionSequence> {
    let pb_sequence = pb::RowDatasetVersionSequence::decode(data).map_err(|e| {
        Error::internal(format!("Failed to decode RowDatasetVersionSequence: {}", e))
    })?;

    let segments = pb_sequence
        .runs
        .into_iter()
        .map(|pb_run| {
            let positions_pb = pb_run.span.ok_or_else(|| {
                Error::internal("Missing positions in RowDatasetVersionRun".to_string())
            })?;
            let segment = U64Segment::try_from(positions_pb)?;
            Ok(RowDatasetVersionRun {
                span: segment,
                version: pb_run.version,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RowDatasetVersionSequence { runs: segments })
}

/// Re-chunk a sequence of dataset version runs into new chunk sizes (aligned with RowIdSequence rechunking)
pub fn rechunk_version_sequences(
    sequences: impl IntoIterator<Item = RowDatasetVersionSequence>,
    chunk_sizes: impl IntoIterator<Item = u64>,
    allow_incomplete: bool,
) -> Result<Vec<RowDatasetVersionSequence>> {
    let chunk_sizes_vec: Vec<u64> = chunk_sizes.into_iter().collect();
    let total_chunks = chunk_sizes_vec.len();
    let mut chunked_sequences: Vec<RowDatasetVersionSequence> = Vec::with_capacity(total_chunks);

    let mut run_iter = sequences
        .into_iter()
        .flat_map(|sequence| sequence.runs.into_iter())
        .peekable();

    let too_few_segments_error = |chunk_index: usize, expected_chunk_size: u64, remaining: u64| {
        Error::invalid_input(format!(
            "Got too few version runs for chunk {}. Expected chunk size: {}, remaining needed: {}",
            chunk_index, expected_chunk_size, remaining
        ))
    };

    let too_many_segments_error = |processed_chunks: usize, total_chunk_sizes: usize| {
        Error::invalid_input(format!(
            "Got too many version runs for the provided chunk lengths. Processed {} chunks out of {} expected",
            processed_chunks, total_chunk_sizes
        ))
    };

    let mut segment_offset = 0_u64;

    for (chunk_index, chunk_size) in chunk_sizes_vec.iter().enumerate() {
        let chunk_size = *chunk_size;
        let mut out_seq = RowDatasetVersionSequence::new();
        let mut remaining = chunk_size;

        while remaining > 0 {
            let remaining_in_segment = run_iter
                .peek()
                .map_or(0, |run| run.span.len() as u64 - segment_offset);

            if remaining_in_segment == 0 {
                if run_iter.next().is_some() {
                    segment_offset = 0;
                    continue;
                } else if allow_incomplete {
                    break;
                } else {
                    return Err(too_few_segments_error(chunk_index, chunk_size, remaining));
                }
            }

            match remaining_in_segment.cmp(&remaining) {
                std::cmp::Ordering::Greater => {
                    let run = run_iter.peek().unwrap();
                    let seg = run.span.slice(segment_offset as usize, remaining as usize);
                    out_seq.runs.push(RowDatasetVersionRun {
                        span: seg,
                        version: run.version,
                    });
                    segment_offset += remaining;
                    remaining = 0;
                }
                std::cmp::Ordering::Equal | std::cmp::Ordering::Less => {
                    let run = run_iter.next().ok_or_else(|| {
                        too_few_segments_error(chunk_index, chunk_size, remaining)
                    })?;
                    let seg = run
                        .span
                        .slice(segment_offset as usize, remaining_in_segment as usize);
                    out_seq.runs.push(RowDatasetVersionRun {
                        span: seg,
                        version: run.version,
                    });
                    segment_offset = 0;
                    remaining -= remaining_in_segment;
                }
            }
        }

        chunked_sequences.push(out_seq);
    }

    if run_iter.peek().is_some() {
        return Err(too_many_segments_error(
            chunked_sequences.len(),
            total_chunks,
        ));
    }

    Ok(chunked_sequences)
}

/// Build version metadata for a fragment if it has physical rows and no existing metadata.
pub fn build_version_meta(
    fragment: &Fragment,
    current_version: u64,
) -> Option<RowDatasetVersionMeta> {
    if let Some(physical_rows) = fragment.physical_rows
        && physical_rows > 0
    {
        // Verify row_id_meta exists (sanity check for stable row IDs)
        if fragment.row_id_meta.is_none() {
            panic!("Can not find row id meta, please make sure you have enabled stable row id.")
        }

        // Use physical_rows directly as the authoritative row count
        // This is correct even for compacted fragments where row_id_meta might
        // have been partially copied
        let version_sequence = RowDatasetVersionSequence::from_uniform_row_count(
            physical_rows as u64,
            current_version,
        );

        return Some(RowDatasetVersionMeta::from_sequence(&version_sequence).unwrap());
    }
    None
}

/// Refresh row-level latest update version metadata for a full fragment rewrite-column update.
///
/// This sets a uniform version sequence for all rows in the fragment to `current_version`.
pub fn refresh_row_latest_update_meta_for_full_frag_rewrite_cols(
    fragment: &mut Fragment,
    current_version: u64,
) -> Result<()> {
    let row_count = if let Some(pr) = fragment.physical_rows {
        pr as u64
    } else if let Some(row_id_meta) = fragment.row_id_meta.as_ref() {
        match row_id_meta {
            crate::format::RowIdMeta::Inline(data) => {
                let sequence = read_row_ids(data).unwrap();
                sequence.len()
            }
            // Follow existing behavior: external sequence not yet supported here
            crate::format::RowIdMeta::External(_file) => 0,
        }
    } else {
        0
    };

    if row_count > 0 {
        let version_seq =
            RowDatasetVersionSequence::from_uniform_row_count(row_count, current_version);
        let version_meta = RowDatasetVersionMeta::from_sequence(&version_seq)?;
        fragment.last_updated_at_version_meta = Some(version_meta);
    }

    Ok(())
}

/// Refresh row-level latest update version metadata for a partial fragment rewrite-column update.
///
/// `updated_offsets` are local row offsets (within the fragment) that have been updated.
/// Existing version metadata is preserved and only the updated positions are set to `current_version`.
/// If no existing metadata is present, positions default to `prev_version`.
pub fn refresh_row_latest_update_meta_for_partial_frag_rewrite_cols(
    fragment: &mut Fragment,
    updated_offsets: &[usize],
    current_version: u64,
    prev_version: u64,
) -> Result<()> {
    // Determine row count for fragment
    let row_count_u64: u64 = if let Some(pr) = fragment.physical_rows {
        pr as u64
    } else if let Some(row_id_meta) = fragment.row_id_meta.as_ref() {
        match row_id_meta {
            crate::format::RowIdMeta::Inline(data) => {
                let sequence = read_row_ids(data).unwrap();
                sequence.len()
            }
            crate::format::RowIdMeta::External(_file) => {
                // Preserve original behavior for external sequences
                todo!("External file loading not yet implemented")
            }
        }
    } else {
        0
    };

    if row_count_u64 > 0 {
        // Build base version vector from existing meta or previous dataset version
        let mut base_versions: Vec<u64> = Vec::with_capacity(row_count_u64 as usize);
        if let Some(meta) = fragment.last_updated_at_version_meta.as_ref() {
            if let Ok(base_seq) = meta.load_sequence() {
                for pos in 0..(row_count_u64 as usize) {
                    base_versions.push(base_seq.version_at(pos).unwrap_or(prev_version));
                }
            } else {
                base_versions.resize(row_count_u64 as usize, prev_version);
            }
        } else {
            base_versions.resize(row_count_u64 as usize, prev_version);
        }

        // Apply updates to updated positions
        for &pos in updated_offsets {
            if pos < base_versions.len() {
                base_versions[pos] = current_version;
            }
        }

        // Compress into runs
        let mut runs: Vec<RowDatasetVersionRun> = Vec::new();
        if !base_versions.is_empty() {
            let mut start = 0usize;
            let mut curr_ver = base_versions[0];
            for (idx, &ver) in base_versions.iter().enumerate().skip(1) {
                if ver != curr_ver {
                    runs.push(RowDatasetVersionRun {
                        span: U64Segment::Range(start as u64..idx as u64),
                        version: curr_ver,
                    });
                    start = idx;
                    curr_ver = ver;
                }
            }
            runs.push(RowDatasetVersionRun {
                span: U64Segment::Range(start as u64..base_versions.len() as u64),
                version: curr_ver,
            });
        }
        let new_seq = RowDatasetVersionSequence { runs };
        let new_meta = RowDatasetVersionMeta::from_sequence(&new_seq)?;
        fragment.last_updated_at_version_meta = Some(new_meta);
    }

    Ok(())
}

// Protobuf conversion implementations
impl TryFrom<pb::data_fragment::LastUpdatedAtVersionSequence> for RowDatasetVersionMeta {
    type Error = Error;

    fn try_from(value: pb::data_fragment::LastUpdatedAtVersionSequence) -> Result<Self> {
        match value {
            pb::data_fragment::LastUpdatedAtVersionSequence::InlineLastUpdatedAtVersions(data) => {
                Ok(Self::Inline(Arc::from(data)))
            }
            pb::data_fragment::LastUpdatedAtVersionSequence::ExternalLastUpdatedAtVersions(
                file,
            ) => Ok(Self::External(ExternalFile {
                path: file.path,
                offset: file.offset,
                size: file.size,
            })),
        }
    }
}

impl TryFrom<pb::data_fragment::CreatedAtVersionSequence> for RowDatasetVersionMeta {
    type Error = Error;

    fn try_from(value: pb::data_fragment::CreatedAtVersionSequence) -> Result<Self> {
        match value {
            pb::data_fragment::CreatedAtVersionSequence::InlineCreatedAtVersions(data) => {
                Ok(Self::Inline(Arc::from(data)))
            }
            pb::data_fragment::CreatedAtVersionSequence::ExternalCreatedAtVersions(file) => {
                Ok(Self::External(ExternalFile {
                    path: file.path,
                    offset: file.offset,
                    size: file.size,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_random_access() {
        let seq = RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..3),
                    version: 1,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..2),
                    version: 2,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..1),
                    version: 3,
                },
            ],
        };
        assert_eq!(seq.version_at(0), Some(1));
        assert_eq!(seq.version_at(2), Some(1));
        assert_eq!(seq.version_at(3), Some(2));
        assert_eq!(seq.version_at(4), Some(2));
        assert_eq!(seq.version_at(5), Some(3));
        assert_eq!(seq.version_at(6), None);
    }

    #[test]
    fn test_serialization_round_trip() {
        let seq = RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..4),
                    version: 42,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..3),
                    version: 99,
                },
            ],
        };
        let bytes = write_dataset_versions(&seq);
        let seq2 = read_dataset_versions(&bytes).unwrap();
        assert_eq!(seq2.runs.len(), 2);
        assert_eq!(seq2.len(), 7);
        assert_eq!(seq2.version_at(0), Some(42));
        assert_eq!(seq2.version_at(5), Some(99));
    }

    #[test]
    fn test_get_version_for_row_id() {
        let seq = RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..2),
                    version: 8,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..2),
                    version: 9,
                },
            ],
        };
        let rows = RowIdSequence::from(10..14); // row ids: 10,11,12,13
        assert_eq!(seq.get_version_for_row_id(&rows, 10), Some(8));
        assert_eq!(seq.get_version_for_row_id(&rows, 11), Some(8));
        assert_eq!(seq.get_version_for_row_id(&rows, 12), Some(9));
        assert_eq!(seq.get_version_for_row_id(&rows, 13), Some(9));
        assert_eq!(seq.get_version_for_row_id(&rows, 99), None);
    }

    #[test]
    fn test_version_cursor_ranges_gaps_and_rewind() {
        let seq = RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..3),
                    version: 10,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(3..3),
                    version: 99,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(3..5),
                    version: 20,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(5..9),
                    version: 30,
                },
            ],
        };
        let expected = [10, 10, 10, 20, 20, 30, 30, 30, 30];
        let mut cursor = seq.cursor();
        let mut actual = Vec::new();

        cursor.extend_range(&seq, 0..2, &mut actual).unwrap();
        cursor.extend_range(&seq, 2..5, &mut actual).unwrap();
        cursor.extend_range(&seq, 7..9, &mut actual).unwrap();
        assert_eq!(actual, [10, 10, 10, 20, 20, 30, 30]);

        actual.clear();
        cursor.extend_range(&seq, 1..6, &mut actual).unwrap();
        assert_eq!(actual, expected[1..6]);

        cursor.extend_range(&seq, 6..6, &mut actual).unwrap();
        assert_eq!(actual, expected[1..6]);
    }

    #[test]
    fn test_version_cursor_descending_ranges_use_indexed_seek() {
        const RUNS: usize = 10_000;
        let seq = RowDatasetVersionSequence {
            runs: (0..RUNS)
                .map(|position| RowDatasetVersionRun {
                    span: U64Segment::Range(position as u64..position as u64 + 1),
                    version: position as u64,
                })
                .collect(),
        };
        let mut cursor = seq.cursor();
        let mut actual = Vec::with_capacity(RUNS);

        for position in (0..RUNS).rev() {
            cursor
                .extend_range(&seq, position..position + 1, &mut actual)
                .unwrap();
        }

        assert_eq!(actual, (0..RUNS as u64).rev().collect::<Vec<_>>());
        assert_eq!(cursor.run_offsets.as_ref().unwrap().len(), RUNS);
    }

    #[test]
    fn test_version_cursor_alternating_ranges_use_indexed_seek_both_directions() {
        const RUNS: usize = 10_000;
        let seq = RowDatasetVersionSequence {
            runs: (0..RUNS)
                .map(|position| RowDatasetVersionRun {
                    span: U64Segment::Range(position as u64..position as u64 + 1),
                    version: position as u64,
                })
                .collect(),
        };
        let mut cursor = seq.cursor();
        let mut actual = Vec::with_capacity(RUNS);

        for selection_index in 0..RUNS {
            let position = if selection_index % 2 == 0 {
                RUNS - 1
            } else {
                0
            };
            cursor
                .extend_range(&seq, position..position + 1, &mut actual)
                .unwrap();
        }

        let expected = (0..RUNS)
            .map(|selection_index| {
                if selection_index % 2 == 0 {
                    (RUNS - 1) as u64
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(cursor.run_offsets.as_ref().unwrap().len(), RUNS);
        assert_eq!(cursor.indexed_seek_count, RUNS - 1);
    }

    #[test]
    fn test_version_cursor_reports_out_of_bounds() {
        let seq = RowDatasetVersionSequence::from_uniform_row_count(4, 7);
        let mut cursor = seq.cursor();
        let mut actual = Vec::new();
        let error = cursor.extend_range(&seq, 4..5, &mut actual).unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("position 4 out of range (total_len=4)")
        );
        assert!(actual.is_empty());
    }

    #[test]
    fn test_version_cursor_non_range_span() {
        let non_range_span = U64Segment::from_slice(&[0, 2, 4, 6, 8]);
        assert!(!matches!(non_range_span, U64Segment::Range(_)));
        let seq = RowDatasetVersionSequence {
            runs: vec![
                RowDatasetVersionRun {
                    span: non_range_span,
                    version: 4,
                },
                RowDatasetVersionRun {
                    span: U64Segment::Range(0..3),
                    version: 8,
                },
            ],
        };
        let mut actual = Vec::new();
        seq.cursor().extend_range(&seq, 0..8, &mut actual).unwrap();
        assert_eq!(actual, [4, 4, 4, 4, 4, 8, 8, 8]);
    }
}
