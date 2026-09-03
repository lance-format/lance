// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stable-partition row mapping for reordered rewrites.
//!
//! A reordered rewrite (e.g. reclustering) reads `n` source fragments in scan
//! order and distributes their live rows across `m` destination fragments,
//! preserving relative row order within each destination (a stable partition).
//! Unlike compaction, the destination of a row cannot be derived from row
//! order alone, so the rewrite records one label per physical source row:
//!
//! * The label of the `i`-th physical source row (concatenated scan order) is
//!   the index of its destination in the rewrite's ordered destination list.
//! * A deleted source row is labeled NULL: it moved nowhere.
//!
//! Ordering:
//!
//! The labels do not carry destination offsets; offsets are reconstructed by
//! counting, which is only correct when the rewrite obeys all of:
//!
//! * Labels are recorded in source physical-row order: fragments in scan
//!   order, offsets ascending within each fragment.
//! * Each destination receives its rows in that same source order, and rows
//!   are never re-sorted within a destination afterwards.
//! * The ordered destination list is fixed for the whole rewrite.
//!
//! A rewrite that routes rows through parallel writers must merge each
//! destination's output back into source order before recording this
//! mapping; otherwise the derived offsets address the wrong rows.
//!
//! This contract is deliberate scope, not an oversight: the format
//! represents stable partitions only. A rewrite that sorts rows *within* a
//! destination is not representable here — with source rows `[a, b]`
//! written to one destination as `[b, a]`, both labels are equal, and
//! counting would assign the offsets backwards. Such a rewrite needs a
//! per-row final-offset (permutation) encoding, which would be a separate
//! format rather than a relaxation of this one.
//!
//! Because each destination is filled in source scan order, the destination
//! row offset of a live row equals the number of earlier source rows with the
//! same label. This module provides that arithmetic without materializing an
//! O(rows) address map:
//!
//! * [`CountsMatrix`] — for every fixed-size block of source rows, the
//!   cumulative per-label row count through the end of that block, as a
//!   dense `num_blocks * m` grid. A block is the unit of storage, IO and
//!   caching for the label column. The grid size is exact and data
//!   independent (~1.5 MB for 50M rows across 500 destinations, ~61 MB at a
//!   1B-row rewrite across 1000); the encoded header names the
//!   representation so sparser encodings can be added later without breaking
//!   readers.
//! * [`translate_in_block`] — point lookup: destination offset = cumulative
//!   count before the row's block (one array read) + the label's rank within
//!   the block prefix. Touches one block of labels.
//! * [`SweepTranslator`] — sequential translation: per-label counters seeded
//!   from a block boundary, then `offset = counter[label]++` per row. O(1)
//!   per row, used for bulk remapping and whole-unit decodes.
//!
//! Label storage and file IO live in `lance-index`; this module is pure
//! arithmetic over decoded label blocks and the counts matrix.

use crate::deepsize::{Context, DeepSizeOf};
use crate::{Error, Result};
use arrow_buffer::NullBuffer;

/// Default number of source rows per block: the counts granularity and the
/// unit of label IO.
pub const DEFAULT_BLOCK_ROWS: u32 = 64 * 1024;

/// Labels are u16 destination indices, so one rewrite addresses at most
/// 65536 destinations.
pub const MAX_DESTINATIONS: u32 = u16::MAX as u32 + 1;

const COUNTS_MAGIC: &[u8; 4] = b"LSPC";
const COUNTS_VERSION: u32 = 1;
/// magic + version + repr + num_destinations + block_rows + total_rows
const COUNTS_HEADER_BYTES: usize = 4 + 4 + 4 + 4 + 4 + 8;
/// Representation tag in the encoded header. Only the dense grid is written
/// today; other tag values are reserved for future representations (e.g.
/// per-destination sparse postings) and rejected with a clear error by
/// current readers.
const REPR_DENSE: u32 = 0;

fn corrupt(message: impl Into<String>) -> Error {
    Error::corrupt_file_named("stable_partition_counts", message)
}

/// Cumulative per-destination row counts at every block boundary.
///
/// Row `b` of the grid holds, for each destination label `d`, the number of
/// source rows labeled `d` in blocks `0..=b`. The final row therefore holds
/// the total row count of every destination. Deleted (NULL-labeled) rows are
/// not counted; the deleted count of a block is implied by
/// `block length - sum of per-label deltas`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountsMatrix {
    /// Number of destinations `m`; labels are `0..m`.
    num_destinations: u32,
    /// Source rows per block. Only the final block may be shorter.
    block_rows: u32,
    /// Total physical source rows, including deleted rows.
    total_rows: u64,
    /// `num_blocks * num_destinations` cumulative counts, row-major by block.
    cumulative: Vec<u32>,
}

impl CountsMatrix {
    pub fn num_destinations(&self) -> u32 {
        self.num_destinations
    }

    pub fn block_rows(&self) -> u32 {
        self.block_rows
    }

    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    pub fn num_blocks(&self) -> usize {
        self.cumulative.len() / self.num_destinations as usize
    }

    /// The block containing physical source row `row`.
    pub fn block_of(&self, row: u64) -> usize {
        (row / u64::from(self.block_rows)) as usize
    }

    /// The physical source row range covered by `block`.
    pub fn block_range(&self, block: usize) -> std::ops::Range<u64> {
        let start = block as u64 * u64::from(self.block_rows);
        let end = (start + u64::from(self.block_rows)).min(self.total_rows);
        start..end
    }

    fn block_row(&self, block: usize) -> &[u32] {
        let m = self.num_destinations as usize;
        &self.cumulative[block * m..(block + 1) * m]
    }

    /// Rows labeled `label` in blocks before `block`: the destination offset
    /// of the first `label`-row inside `block`.
    pub fn count_before(&self, block: usize, label: u16) -> u32 {
        if block == 0 {
            0
        } else {
            self.block_row(block - 1)[label as usize]
        }
    }

    /// Per-label counters at the start of `block`, seeding a
    /// [`SweepTranslator`].
    pub fn counters_at_block(&self, block: usize) -> Vec<u32> {
        if block == 0 {
            vec![0; self.num_destinations as usize]
        } else {
            self.block_row(block - 1).to_vec()
        }
    }

    /// Total rows of destination `label`; the final cumulative row.
    pub fn total(&self, label: u16) -> u32 {
        let blocks = self.num_blocks();
        if blocks == 0 {
            0
        } else {
            self.block_row(blocks - 1)[label as usize]
        }
    }

    /// Total live (labeled) rows across all destinations.
    pub fn total_live_rows(&self) -> u64 {
        let blocks = self.num_blocks();
        if blocks == 0 {
            0
        } else {
            self.block_row(blocks - 1)
                .iter()
                .map(|&c| u64::from(c))
                .sum()
        }
    }

    /// Deleted (NULL-labeled) rows in blocks `0..=block`.
    pub fn deleted_through(&self, block: usize) -> u64 {
        let rows_through = self.block_range(block).end;
        let live_through: u64 = self.block_row(block).iter().map(|&c| u64::from(c)).sum();
        rows_through - live_through
    }

    /// Check internal consistency: per-label counts must be non-decreasing
    /// across blocks and each block's live rows must fit its row range.
    /// O(num_blocks * num_destinations), no label IO. Called when a row map
    /// is opened, so a corrupt counts buffer fails loudly instead of
    /// translating rows to wrong addresses.
    pub fn validate(&self) -> Result<()> {
        let mut prev_live = 0u64;
        for block in 0..self.num_blocks() {
            if block > 0 {
                let (prev, cur) = (self.block_row(block - 1), self.block_row(block));
                if prev.iter().zip(cur).any(|(p, c)| c < p) {
                    return Err(corrupt(format!(
                        "cumulative counts decrease at block {block}"
                    )));
                }
            }
            let live_through: u64 = self.block_row(block).iter().map(|&c| u64::from(c)).sum();
            let range = self.block_range(block);
            let block_len = range.end - range.start;
            if live_through - prev_live > block_len {
                return Err(corrupt(format!(
                    "block {block} accounts for more live rows than its {block_len} rows"
                )));
            }
            prev_live = live_through;
        }
        Ok(())
    }

    /// Serialize to the on-disk form stored in the row map file's global
    /// buffer: a fixed header naming the representation, then the cumulative
    /// counts as little-endian u32.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(COUNTS_HEADER_BYTES + self.cumulative.len() * 4);
        buf.extend_from_slice(COUNTS_MAGIC);
        buf.extend_from_slice(&COUNTS_VERSION.to_le_bytes());
        buf.extend_from_slice(&REPR_DENSE.to_le_bytes());
        buf.extend_from_slice(&self.num_destinations.to_le_bytes());
        buf.extend_from_slice(&self.block_rows.to_le_bytes());
        buf.extend_from_slice(&self.total_rows.to_le_bytes());
        for count in &self.cumulative {
            buf.extend_from_slice(&count.to_le_bytes());
        }
        buf
    }

    /// Decode the on-disk form, checking structure exactly: magic, version, a
    /// supported representation, shape legality and precise payload length.
    /// Content invariants (monotone counts, per-block row budgets) are
    /// checked by [`Self::validate`].
    pub fn decode(buf: &[u8]) -> Result<Self> {
        let header: &[u8; COUNTS_HEADER_BYTES] = buf
            .get(..COUNTS_HEADER_BYTES)
            .and_then(|header| header.try_into().ok())
            .ok_or_else(|| {
                corrupt(format!(
                    "counts buffer is {} bytes, shorter than the {COUNTS_HEADER_BYTES}-byte header",
                    buf.len()
                ))
            })?;
        if &header[0..4] != COUNTS_MAGIC {
            return Err(corrupt("counts buffer has a bad magic number"));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != COUNTS_VERSION {
            return Err(corrupt(format!("unsupported counts version {version}")));
        }
        let repr = u32::from_le_bytes(header[8..12].try_into().unwrap());
        if repr != REPR_DENSE {
            return Err(corrupt(format!(
                "unsupported counts representation {repr}; this reader only supports the dense grid ({REPR_DENSE})"
            )));
        }
        let num_destinations = u32::from_le_bytes(header[12..16].try_into().unwrap());
        let block_rows = u32::from_le_bytes(header[16..20].try_into().unwrap());
        let total_rows = u64::from_le_bytes(header[20..28].try_into().unwrap());
        Self::check_shape(num_destinations, block_rows)?;
        let num_blocks = total_rows.div_ceil(u64::from(block_rows));
        num_blocks
            .checked_mul(u64::from(num_destinations))
            .and_then(|cells| cells.checked_mul(4))
            .filter(|&bytes| bytes == (buf.len() - COUNTS_HEADER_BYTES) as u64)
            .ok_or_else(|| {
                corrupt(format!(
                    "counts buffer of {} bytes does not match {num_blocks} blocks of {num_destinations} destinations",
                    buf.len()
                ))
            })?;
        let mut cumulative =
            Vec::with_capacity((num_blocks * u64::from(num_destinations)) as usize);
        for chunk in buf[COUNTS_HEADER_BYTES..].chunks_exact(4) {
            cumulative.push(u32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self {
            num_destinations,
            block_rows,
            total_rows,
            cumulative,
        })
    }

    fn check_shape(num_destinations: u32, block_rows: u32) -> Result<()> {
        if num_destinations == 0 || num_destinations > MAX_DESTINATIONS {
            return Err(Error::invalid_input(format!(
                "stable partition needs 1..={MAX_DESTINATIONS} destinations, got {num_destinations}"
            )));
        }
        if block_rows == 0 {
            return Err(Error::invalid_input(
                "stable partition block_rows must be positive",
            ));
        }
        Ok(())
    }

    fn check_label(&self, label: u16) -> Result<()> {
        if u32::from(label) >= self.num_destinations {
            return Err(Error::invalid_input(format!(
                "label {label} is outside the {} destinations of this stable partition",
                self.num_destinations
            )));
        }
        Ok(())
    }
}

impl DeepSizeOf for CountsMatrix {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.cumulative.deep_size_of_children(context)
    }
}

/// Streaming builder for a [`CountsMatrix`]: feed one label per physical
/// source row in scan order.
#[derive(Debug)]
pub struct CountsMatrixBuilder {
    num_destinations: u32,
    block_rows: u32,
    running: Vec<u32>,
    cumulative: Vec<u32>,
    total_rows: u64,
    rows_in_block: u32,
}

impl CountsMatrixBuilder {
    pub fn try_new(num_destinations: u32, block_rows: u32) -> Result<Self> {
        CountsMatrix::check_shape(num_destinations, block_rows)?;
        Ok(Self {
            num_destinations,
            block_rows,
            running: vec![0; num_destinations as usize],
            cumulative: Vec::new(),
            total_rows: 0,
            rows_in_block: 0,
        })
    }

    /// Record the label of the next physical source row; `None` marks a
    /// deleted row.
    pub fn push(&mut self, label: Option<u16>) -> Result<()> {
        if let Some(label) = label {
            if u32::from(label) >= self.num_destinations {
                return Err(Error::invalid_input(format!(
                    "label {label} is outside the {} destinations of this stable partition",
                    self.num_destinations
                )));
            }
            let counter = &mut self.running[label as usize];
            *counter = counter.checked_add(1).ok_or_else(|| {
                Error::invalid_input(format!(
                    "destination {label} exceeds the u32 row-offset range"
                ))
            })?;
        }
        self.total_rows += 1;
        self.rows_in_block += 1;
        if self.rows_in_block == self.block_rows {
            self.cumulative.extend_from_slice(&self.running);
            self.rows_in_block = 0;
        }
        Ok(())
    }

    pub fn finish(mut self) -> CountsMatrix {
        if self.rows_in_block > 0 {
            self.cumulative.extend_from_slice(&self.running);
        }
        CountsMatrix {
            num_destinations: self.num_destinations,
            block_rows: self.block_rows,
            total_rows: self.total_rows,
            cumulative: self.cumulative,
        }
    }
}

/// Rank of `label` among the first `upto` labels of a block: the number of
/// valid entries equal to `label` in `values[..upto]`.
pub fn rank_label_prefix(
    values: &[u16],
    validity: Option<&NullBuffer>,
    upto: usize,
    label: u16,
) -> u32 {
    match validity {
        // NULL slots hold arbitrary values after an encoding round trip, so
        // they must be masked out of the count.
        Some(validity) if validity.null_count() > 0 => values[..upto]
            .iter()
            .enumerate()
            .filter(|&(i, &value)| value == label && validity.is_valid(i))
            .count() as u32,
        _ => values[..upto]
            .iter()
            .filter(|&&value| value == label)
            .count() as u32,
    }
}

/// Translate one physical source row using its decoded label block.
///
/// `values`/`validity` are the labels of `block` (exactly the rows of
/// [`CountsMatrix::block_range`]); `pos_in_block` addresses the row within
/// them. Returns `None` when the row was deleted at the source, otherwise
/// `(destination index, destination row offset)`.
pub fn translate_in_block(
    counts: &CountsMatrix,
    block: usize,
    values: &[u16],
    validity: Option<&NullBuffer>,
    pos_in_block: usize,
) -> Result<Option<(u16, u32)>> {
    let block_range = counts.block_range(block);
    let block_len = (block_range.end - block_range.start) as usize;
    if values.len() != block_len {
        return Err(Error::invalid_input(format!(
            "block {block} holds {block_len} labels but {} were supplied",
            values.len()
        )));
    }
    if pos_in_block >= block_len {
        return Err(Error::invalid_input(format!(
            "row {pos_in_block} is outside block {block} of {block_len} rows"
        )));
    }
    if validity.is_some_and(|validity| !validity.is_valid(pos_in_block)) {
        return Ok(None);
    }
    let label = values[pos_in_block];
    counts.check_label(label)?;
    let offset = counts.count_before(block, label)
        + rank_label_prefix(values, validity, pos_in_block, label);
    Ok(Some((label, offset)))
}

/// Sequential translator: seed per-label counters from a block boundary, then
/// feed every physical source row's label in order.
#[derive(Debug)]
pub struct SweepTranslator {
    counters: Vec<u32>,
}

impl SweepTranslator {
    /// Counters positioned at the first row of `start_block`.
    pub fn new(counts: &CountsMatrix, start_block: usize) -> Result<Self> {
        if start_block >= counts.num_blocks().max(1) {
            return Err(Error::invalid_input(format!(
                "start block {start_block} is outside the {} blocks of this stable partition",
                counts.num_blocks()
            )));
        }
        Ok(Self {
            counters: counts.counters_at_block(start_block),
        })
    }

    /// Translate the next row. `None` in, `None` out for deleted rows.
    #[inline]
    pub fn advance(&mut self, label: Option<u16>) -> Result<Option<(u16, u32)>> {
        let Some(label) = label else {
            return Ok(None);
        };
        let counter = self.counters.get_mut(label as usize).ok_or_else(|| {
            Error::invalid_input(format!(
                "label {label} is outside the destinations of this stable partition"
            ))
        })?;
        let offset = *counter;
        *counter += 1;
        Ok(Some((label, offset)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// ~1/8 deleted rows, labels uniform over `m`.
    fn random_labels(rng: &mut StdRng, rows: usize, m: u32) -> Vec<Option<u16>> {
        (0..rows)
            .map(|_| {
                if rng.random_ratio(1, 8) {
                    None
                } else {
                    Some(rng.random_range(0..m) as u16)
                }
            })
            .collect()
    }

    fn build(labels: &[Option<u16>], m: u32, block_rows: u32) -> CountsMatrix {
        let mut builder = CountsMatrixBuilder::try_new(m, block_rows).unwrap();
        for &label in labels {
            builder.push(label).unwrap();
        }
        builder.finish()
    }

    /// Reference translation: scan all labels with per-destination counters.
    fn reference(labels: &[Option<u16>]) -> Vec<Option<(u16, u32)>> {
        let mut counters = std::collections::HashMap::new();
        labels
            .iter()
            .map(|label| {
                label.map(|label| {
                    let counter = counters.entry(label).or_insert(0u32);
                    let offset = *counter;
                    *counter += 1;
                    (label, offset)
                })
            })
            .collect()
    }

    fn block_slices(
        labels: &[Option<u16>],
        counts: &CountsMatrix,
        block: usize,
    ) -> (Vec<u16>, Option<NullBuffer>) {
        let range = counts.block_range(block);
        let block_labels = &labels[range.start as usize..range.end as usize];
        // NULL slots get a garbage value on purpose: after an encoding round
        // trip their contents are undefined and must not affect ranks.
        let values = block_labels
            .iter()
            .map(|label| label.unwrap_or(0xBEEF_u64 as u16))
            .collect();
        let has_nulls = block_labels.iter().any(Option::is_none);
        let validity =
            has_nulls.then(|| NullBuffer::from_iter(block_labels.iter().map(Option::is_some)));
        (values, validity)
    }

    #[test]
    fn test_counts_matrix_hand_example() {
        // 10 rows, block_rows=4 -> blocks of 4, 4, 2. m=3, and destination 1
        // receives nothing in blocks 0 and 2.
        let labels = [
            Some(0),
            Some(2),
            None,
            Some(2), // block 0: d0=1, d2=2, 1 deleted
            Some(1),
            Some(1),
            Some(0),
            Some(2), // block 1: cum d0=2, d1=2, d2=3
            None,
            Some(0), // block 2: cum d0=3, d1=2, d2=3, 2 deleted total
        ];
        let counts = build(&labels, 3, 4);
        counts.validate().unwrap();
        assert_eq!(counts.num_blocks(), 3);
        assert_eq!(counts.total_rows(), 10);
        assert_eq!(counts.block_range(2), 8..10);

        assert_eq!(counts.count_before(0, 2), 0);
        assert_eq!(counts.count_before(1, 2), 2);
        assert_eq!(counts.count_before(2, 0), 2);
        // Destination 1 received nothing in blocks 0 and 2: lookups around
        // the gap see the count of its last change (or 0).
        assert_eq!(counts.count_before(1, 1), 0);
        assert_eq!(counts.count_before(3, 1), 2);
        assert_eq!(
            (counts.total(0), counts.total(1), counts.total(2)),
            (3, 2, 3)
        );
        assert_eq!(counts.total_live_rows(), 8);
        assert_eq!(counts.deleted_through(0), 1);
        assert_eq!(counts.deleted_through(2), 2);
        assert_eq!(counts.counters_at_block(1), vec![1, 0, 2]);

        // Exact multiple of block_rows: no partial block.
        let exact = build(&labels[..8], 3, 4);
        assert_eq!(exact.num_blocks(), 2);
        assert_eq!(exact.total_rows(), 8);

        let empty = build(&[], 3, 4);
        assert_eq!(empty.num_blocks(), 0);
        assert_eq!(empty.total(1), 0);
        assert_eq!(empty.total_live_rows(), 0);
        empty.validate().unwrap();
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let mut rng = StdRng::seed_from_u64(7);
        let labels = random_labels(&mut rng, 1000, 5);
        let counts = build(&labels, 5, 64);
        let decoded = CountsMatrix::decode(&counts.encode()).unwrap();
        assert_eq!(decoded, counts);
        decoded.validate().unwrap();

        let empty = build(&[], 5, 64);
        assert_eq!(CountsMatrix::decode(&empty.encode()).unwrap(), empty);

        // Structural corruption is caught at decode: too short, bad magic,
        // truncated payload, unknown representation tag.
        assert!(CountsMatrix::decode(&[]).is_err());
        let mut bad_magic = counts.encode();
        bad_magic[0] = b'X';
        assert!(CountsMatrix::decode(&bad_magic).is_err());
        let mut truncated = counts.encode();
        truncated.pop();
        assert!(CountsMatrix::decode(&truncated).is_err());
        let mut bad_repr = counts.encode();
        bad_repr[8] = 9;
        assert!(CountsMatrix::decode(&bad_repr).is_err());

        // Content corruption is caught by validate: shrink a later block's
        // cumulative count below an earlier one.
        let mut decreasing = counts;
        let m = decreasing.num_destinations as usize;
        decreasing.cumulative[m] = 0;
        assert!(
            CountsMatrix::decode(&decreasing.encode())
                .unwrap()
                .validate()
                .is_err()
        );
    }

    #[test]
    fn test_builder_rejects_bad_shapes_and_labels() {
        assert!(CountsMatrixBuilder::try_new(0, 64).is_err());
        assert!(CountsMatrixBuilder::try_new(MAX_DESTINATIONS + 1, 64).is_err());
        assert!(CountsMatrixBuilder::try_new(1, 0).is_err());
        let mut builder = CountsMatrixBuilder::try_new(3, 4).unwrap();
        assert!(builder.push(Some(3)).is_err());
        assert!(builder.push(Some(2)).is_ok());
    }

    #[test]
    fn test_point_and_sweep_match_reference() {
        let m = 5u32;
        let mut rng = StdRng::seed_from_u64(42);
        let labels = random_labels(&mut rng, 1000, m);
        let counts = build(&labels, m, 64);
        counts.validate().unwrap();
        let expected = reference(&labels);

        // Point lookups, block by block, against the reference map.
        for block in 0..counts.num_blocks() {
            let (values, validity) = block_slices(&labels, &counts, block);
            let range = counts.block_range(block);
            for pos in 0..(range.end - range.start) as usize {
                let translated =
                    translate_in_block(&counts, block, &values, validity.as_ref(), pos).unwrap();
                assert_eq!(
                    translated,
                    expected[range.start as usize + pos],
                    "row {}",
                    range.start as usize + pos
                );
            }
        }

        // A sweep from the start and one from every block boundary.
        for start_block in 0..counts.num_blocks() {
            let mut sweep = SweepTranslator::new(&counts, start_block).unwrap();
            let start_row = counts.block_range(start_block).start as usize;
            for (row, &label) in labels.iter().enumerate().skip(start_row) {
                assert_eq!(
                    sweep.advance(label).unwrap(),
                    expected[row],
                    "row {row} from block {start_block}"
                );
            }
        }
    }

    #[test]
    fn test_translate_in_block_input_checks() {
        let labels = [Some(0), Some(1), None, Some(0)];
        let counts = build(&labels, 2, 4);
        let (values, validity) = block_slices(&labels, &counts, 0);
        // Wrong slice length and out-of-range row are rejected.
        assert!(translate_in_block(&counts, 0, &values[..3], None, 0).is_err());
        assert!(translate_in_block(&counts, 0, &values, validity.as_ref(), 4).is_err());
        // A label outside the destination list is data corruption.
        assert!(translate_in_block(&counts, 0, &[0, 9, 0, 0], None, 1).is_err());
        assert!(SweepTranslator::new(&counts, 1).is_err());
        let mut sweep = SweepTranslator::new(&counts, 0).unwrap();
        assert!(sweep.advance(Some(9)).is_err());
    }
}
