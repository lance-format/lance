// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The data plane of a fragment reuse history: reading the mapping data a
//! history references but does not hold in memory.
//!
//! A tagged history's stable-partition transitions keep their row maps in
//! external files. This module never reads them itself (lance-table does no
//! object-store IO); a [`MappingSource`] the caller provides does, when a
//! [`TranslationPlan`](super::plan::TranslationPlan) bound to it translates
//! addresses that pass through such a transition. Formats whose mappings
//! are fully decoded (legacy) never call it.

use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::UInt16Array;
use async_trait::async_trait;
use lance_core::utils::address::RowAddress;
use lance_core::utils::stable_partition::CountsMatrix;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use crate::format::pb::fragment_reuse_index_details as pb;

/// Reads the row maps of stable-partition transitions.
///
/// A row map labels every physical row of a transition's sources, in
/// source order, with the destination it moved to (null for a deleted row);
/// its counts matrix gives, per block of rows, how many rows each
/// destination received before the block. Implemented in lance-index over
/// the dataset's object stores; tests use an in-memory one.
#[async_trait]
pub trait MappingSource: Send + Sync + Debug {
    /// The counts matrix of the row map `mapping` references.
    async fn counts(&self, mapping: &MappingRef) -> Result<Arc<CountsMatrix>>;

    /// The labels of the source rows `rows` (positions in the transition's
    /// concatenated source order), one per row. `rows` always spans whole
    /// blocks of the counts matrix.
    async fn read(&self, mapping: &MappingRef, rows: Range<u64>) -> Result<UInt16Array>;
}

/// An external row map a history references: the file of one
/// stable-partition transition. Row maps are immutable once written, so a
/// reference identifies its content.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MappingRef {
    map_id: String,
    base_id: Option<u32>,
    size_bytes: u64,
}

impl MappingRef {
    /// A reference to row map `map_id` (its directory under `_fri`) of
    /// `size_bytes`, stored under the base `base_id` (the dataset root when
    /// `None`). Histories create references; this constructor exists for
    /// tests of a [`MappingSource`].
    #[doc(hidden)]
    pub fn new(map_id: impl Into<String>, base_id: Option<u32>, size_bytes: u64) -> Self {
        Self {
            map_id: map_id.into(),
            base_id,
            size_bytes,
        }
    }

    pub(super) fn of(partition: &pb::StablePartition) -> Self {
        Self::new(
            partition.map_id.clone(),
            partition.base_id,
            partition.map_size_bytes,
        )
    }

    /// The name of the row map's directory under the base's `_fri` directory.
    pub fn map_id(&self) -> &str {
        &self.map_id
    }

    /// The base the row map is stored under; `None` is the dataset root.
    pub fn base_id(&self) -> Option<u32> {
        self.base_id
    }

    /// The size of the row map file in bytes, as recorded by its writer.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

/// The most blocks one [`MappingSource::read`] spans. Blocks a batch touches
/// are read in runs of adjacent blocks, so a dense batch reads few large
/// ranges and a sparse one never reads a block it does not need; the bound
/// keeps the labels held at once to a few MiB (a block is 64K labels).
const MAX_READ_BLOCKS: usize = 32;

/// Translate `rows`, addresses in the sources of one stable-partition
/// transition, through its row map. One result per input, in order; `None`
/// for a row deleted at the source.
pub(super) async fn translate_stable_partition(
    source: &dyn MappingSource,
    mapping: &MappingRef,
    sources: &[pb::FragmentDigest],
    destinations: &[pb::FragmentDigest],
    rows: &[u64],
) -> Result<Vec<Option<u64>>> {
    let layout = SourceLayout::try_new(sources, destinations)?;
    let mut requests = Vec::with_capacity(rows.len());
    for (position, &row) in rows.iter().enumerate() {
        requests.push((layout.position_of(row)?, position));
    }
    let mut output = vec![None; rows.len()];
    if requests.is_empty() {
        return Ok(output);
    }
    let counts = source.counts(mapping).await?;
    layout.validate(&counts, destinations, mapping)?;
    requests.sort_unstable_by_key(|&(row, _)| row);
    let mut remaining = requests.as_slice();
    while let Some(&(first, _)) = remaining.first() {
        let first_block = counts.block_of(first);
        let last_block = run_end(&counts, first_block, remaining);
        let run = counts.block_range(first_block).start..counts.block_range(last_block).end;
        let labels = source.read(mapping, run.clone()).await?;
        if labels.len() as u64 != run.end - run.start {
            return Err(corrupt(
                mapping,
                format!("a read of rows {run:?} returned {} labels", labels.len()),
            ));
        }
        for block in first_block..=last_block {
            let range = counts.block_range(block);
            let end = remaining.partition_point(|&(row, _)| row < range.end);
            let (requested, rest) = remaining.split_at(end);
            remaining = rest;
            let block_labels = labels.slice(
                (range.start - run.start) as usize,
                (range.end - range.start) as usize,
            );
            // One sweep per touched block: the counters at a block boundary
            // plus the labels before a row give its destination offset.
            let mut counters = counts.counters_at_block(block);
            let mut requested = requested.iter().peekable();
            for (offset, label) in block_labels.iter().enumerate() {
                let translated = match label {
                    Some(label) => {
                        let counter = counters.get_mut(label as usize).ok_or_else(|| {
                            corrupt(mapping, format!("invalid row-map label {label}"))
                        })?;
                        let destination_offset = *counter;
                        *counter = counter
                            .checked_add(1)
                            .ok_or_else(|| corrupt(mapping, "row-map count overflow"))?;
                        Some(u64::from(RowAddress::new_from_parts(
                            destinations[label as usize].id as u32,
                            destination_offset,
                        )))
                    }
                    None => None,
                };
                let row = range.start + offset as u64;
                while let Some(&&(requested_row, position)) = requested.peek() {
                    if requested_row != row {
                        break;
                    }
                    output[position] = translated;
                    requested.next();
                }
            }
            if counters != counts.counters_at_block(block + 1) {
                return Err(corrupt(
                    mapping,
                    format!("row-map labels disagree with counts in block {block}"),
                ));
            }
        }
    }
    Ok(output)
}

/// The last block of the run of adjacent blocks, starting at `block`, that
/// the sorted `requests` touch, bounded to [`MAX_READ_BLOCKS`].
fn run_end(counts: &CountsMatrix, block: usize, requests: &[(u64, usize)]) -> usize {
    let mut last = block;
    let mut cursor = requests.partition_point(|&(row, _)| row < counts.block_range(block).end);
    while cursor < requests.len() && last + 1 - block < MAX_READ_BLOCKS {
        let next = counts.block_of(requests[cursor].0);
        if next != last + 1 {
            break;
        }
        last = next;
        let end = counts.block_range(next).end;
        cursor += requests[cursor..].partition_point(|&(row, _)| row < end);
    }
    last
}

/// Where each source fragment's rows start in the row map's concatenated
/// source order.
struct SourceLayout {
    offsets: HashMap<u32, (u64, u64)>,
    total_rows: u64,
}

impl SourceLayout {
    fn try_new(
        sources: &[pb::FragmentDigest],
        destinations: &[pb::FragmentDigest],
    ) -> Result<Self> {
        let mut total_rows = 0_u64;
        let mut offsets = HashMap::with_capacity(sources.len());
        for source in sources {
            if source.id >= u64::from(RowAddress::TOMBSTONE_FRAG)
                || source.num_deleted_rows > source.physical_rows
                || source.physical_rows > u64::from(u32::MAX)
                || offsets
                    .insert(source.id as u32, (total_rows, source.physical_rows))
                    .is_some()
            {
                return Err(Error::invalid_input(format!(
                    "invalid or duplicate source fragment {} with {} rows",
                    source.id, source.physical_rows
                )));
            }
            total_rows = total_rows
                .checked_add(source.physical_rows)
                .ok_or_else(|| Error::invalid_input("source physical row count overflow"))?;
        }
        let destination_fragments: RoaringBitmap =
            destinations.iter().map(|f| f.id as u32).collect();
        if destinations.len() > u16::MAX as usize
            || destination_fragments.len() != destinations.len() as u64
            || destinations.iter().any(|f| {
                f.id >= u64::from(RowAddress::TOMBSTONE_FRAG)
                    || f.physical_rows > u64::from(u32::MAX)
                    || f.num_deleted_rows != 0
            })
        {
            return Err(Error::invalid_input(
                "invalid stable-partition destination layout",
            ));
        }
        Ok(Self {
            offsets,
            total_rows,
        })
    }

    fn position_of(&self, row: u64) -> Result<u64> {
        let address = RowAddress::from(row);
        let &(base, rows) = self.offsets.get(&address.fragment_id()).ok_or_else(|| {
            Error::invalid_input(format!(
                "address {address} is outside stable-partition sources"
            ))
        })?;
        if u64::from(address.row_offset()) >= rows {
            return Err(Error::invalid_input(format!(
                "address {address} exceeds source length {rows}"
            )));
        }
        Ok(base + u64::from(address.row_offset()))
    }

    /// The row map must describe exactly the transition's digests: a map
    /// that does not would translate rows to wrong addresses.
    fn validate(
        &self,
        counts: &CountsMatrix,
        destinations: &[pb::FragmentDigest],
        mapping: &MappingRef,
    ) -> Result<()> {
        if counts.total_rows() != self.total_rows
            || counts.num_destinations() as usize != destinations.len()
        {
            return Err(corrupt(
                mapping,
                "row-map dimensions differ from transition digests",
            ));
        }
        for (label, destination) in destinations.iter().enumerate() {
            if u64::from(counts.total(label as u16)) != destination.physical_rows {
                return Err(corrupt(
                    mapping,
                    format!("row-map total differs for destination {}", destination.id),
                ));
            }
        }
        Ok(())
    }
}

fn corrupt(mapping: &MappingRef, message: impl Into<String>) -> Error {
    Error::corrupt_file_named(&format!("row map {}", mapping.map_id()), message.into())
}
