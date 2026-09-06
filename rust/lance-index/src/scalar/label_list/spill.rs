// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Spillable aggregation for LabelList index builds.
//!
//! A LabelList index is a bitmap index over unnested list elements, so its keys
//! are derived after the scan and the scanner cannot pre-sort by them (hence
//! `TrainingOrdering::None`). That rules out the run-length streaming writer
//! plain bitmap indexes use, which requires value-sorted input. This builder
//! establishes the ordering itself: it accumulates into a sorted map, writes a
//! sorted spill file whenever the estimated size exceeds a byte budget, and
//! leaves the caller with a set of sorted files to k-way merge, reduced to a
//! bounded fan-in first so that the number of open cursors does not grow with
//! the number of spills. Modelled on
//! `NGramIndexBuilder`, which solves the same problem for the same reason;
//! see [`MAX_MERGE_FAN_IN`] for where the two diverge.
//!
//! # Aggregation-state contract
//!
//! `LANCE_LABEL_LIST_SPILL_BYTES` limits the estimated memory held by the mutable
//! label-to-row-set aggregation map. After an insertion brings the estimate to or
//! above the configured budget, the map is flushed and cleared. The map may
//! therefore exceed the budget by the most recent insertion, and the estimate is
//! deliberately conservative rather than an allocator-exact measurement.
//!
//! This is not an end-to-end build-memory limit. Memory outside the aggregation
//! budget includes scan batches, the `list_nulls` row set, source indexes and
//! their caches during update or segment merge, merge cursor batches, the bitmap
//! currently being merged, and the output writer. Concurrent builds add their
//! own working state as well.
//!
//! [`MAX_MERGE_FAN_IN`] separately bounds the number of cursor batches held by a
//! merge, not their total bytes: a batch can contain large serialized bitmaps.
//! `MAX_BUFFERED_BYTES` separately limits the output writer's buffered keys and
//! bitmaps. Neither is charged to the aggregation budget.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::DataType;
use datafusion_common::ScalarValue;
use lance_core::cache::LanceCache;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempDir;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_select::RowAddrTreeMap;
use tracing::warn;

use crate::scalar::IndexStore;
use crate::scalar::bitmap::{
    BitmapBatchWriter, BitmapIndex, drain_sorted_bitmap_cursors, merge_index_maps,
    new_bitmap_batch_writer, open_sorted_bitmap_cursors,
};
use crate::scalar::btree::OrderableScalarValue;
use crate::scalar::lance_format::LanceIndexStore;

/// Default estimated aggregation-state budget before spilling.
///
/// This prevents the mutable label-to-row-set map from growing without limit as
/// label cardinality increases. It is not a limit on total build memory; see the
/// module-level aggregation-state contract.
const DEFAULT_SPILL_BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Fixed cost charged for each distinct label, on top of the key's own size.
///
/// Dominated by the inner `BTreeMap<u32, RowAddrSelection>` inside the label's
/// `RowAddrTreeMap`: Rust's B=6 gives a leaf capacity of 11, and the whole node
/// is allocated on the first insert, so a label with a single row address pays
/// `16 + 11 * (4 + size_of::<RowAddrSelection>())` ~= 324 bytes for one entry.
/// On top of that come the outer map's value slot (~25 bytes amortised over a
/// full node), the roaring `containers` Vec, and its first container's store.
/// Rounded up from ~410 to leave room for allocator rounding, which none of the
/// above measures.
///
/// Covers the label's *first* fragment, since the leaf node and roaring bitmap
/// above are what its first row address allocates. Every fragment after that is
/// charged [`FRAGMENT_OVERHEAD_BYTES`].
///
/// Deliberately generous, and measured rather than guessed: LabelList's problem
/// case is a very large number of labels with few rows each, so under-charging
/// the per-label overhead is exactly how a byte budget silently fails to
/// constrain the accumulator. `deep_size_of` is not usable here -- it walks
/// logical contents, not allocation capacity, and reports about half the real
/// cost for this shape.
/// `test_label_overhead_covers_a_single_row_labels_allocations` pins the
/// arithmetic so a change to `RowAddrSelection` cannot silently invalidate it.
const LABEL_OVERHEAD_BYTES: usize = 512;

/// Cost charged for a row address landing in a fragment the label already has.
///
/// Roaring's sparse store is a `Vec<u16>`, so two bytes per value, and a `Vec`
/// grows by doubling, so its capacity can be twice its length -- four bytes is
/// therefore the ceiling, not an average. A dense set costs far less: it becomes
/// a bitmap store, a fixed 8 KiB per 65,536-value block, or about an eighth of a
/// byte per row. The estimate errs toward spilling early rather than late.
const ROW_ADDR_COST_BYTES: usize = 4;

/// Cost charged for a row address landing in a fragment the label has not seen.
///
/// A `RowAddrTreeMap` is a `BTreeMap<u32, RowAddrSelection>` keyed by fragment,
/// so a new fragment is not a cheap increment: it takes a slot in that map,
/// roughly 29 bytes amortised over a full leaf node, and allocates a whole new
/// `RoaringBitmap` -- its `containers` `Vec`, plus that container's own store.
/// Measured at 70 bytes per fragment by `deep_size_of`, which counts neither
/// allocation capacity nor malloc overhead; scaled by the same factor
/// [`LABEL_OVERHEAD_BYTES`] uses over its own measurement, and rounded.
///
/// This is the difference between a label whose rows sit in one fragment and one
/// spread across thousands. Charging [`ROW_ADDR_COST_BYTES`] for both under-counts
/// the spread case by more than an order of magnitude, which would make the
/// aggregation-state budget ineffective.
const FRAGMENT_OVERHEAD_BYTES: usize = 192;

/// Name of the spill file the index being updated is streamed into. Distinct
/// from the numbered build spills and from the intermediate merge files so the
/// three never collide.
const EXISTING_INDEX_FILE_NAME: &str = "label-list-existing.lance";

/// Prefix for the intermediate files written when the spill count exceeds
/// [`MAX_MERGE_FAN_IN`].
const MERGE_FILE_PREFIX: &str = "label-list-merge-";

/// Maximum number of spill files merged in a single k-way pass.
///
/// Without a cap, the number of spill files is `total_bytes / budget_bytes`, and
/// merging them all at once holds one `MERGE_ROWS_PER_CHUNK` batch open per file.
/// Capping fan-in makes the cursor count independent of the spill count, at the
/// cost of extra passes over the spill. It does not impose a byte limit on those
/// cursor batches.
///
/// The bound is in rows, not bytes: 64 files x 512 serialized bitmaps. That is
/// small whenever the bitmaps are (the high-cardinality case, which is also the
/// case that produces many files), but a column with a few very popular labels
/// can serialize those to megabytes each, and a run of them landing in one chunk
/// is not bounded by `budget_bytes`. Making the chunk byte-aware is the fix for
/// that; it belongs in `BitmapShardCursor`, which every merge shares.
///
/// A pass rewrites the spilled bytes exactly once and divides the file count by
/// the fan-in, so reducing any realistic spill takes one or two passes.
/// `NGramIndexBuilder` instead merges each flush back into a single per-worker
/// file, which holds fan-in at one but rewrites the whole spill on every flush
/// -- quadratic in the number of flushes, where this is logarithmic.
const MAX_MERGE_FAN_IN: usize = 64;

/// Read the configured budget, rejecting a value that cannot be honoured.
///
/// Rejected rather than defaulted: someone who sets this knob is limiting
/// aggregation state deliberately, and silently substituting
/// [`DEFAULT_SPILL_BUDGET_BYTES`] would use a different limit without warning.
/// Zero is rejected by [`LabelListSpillBuilder::new_local`], which every budget
/// reaches.
pub(super) fn default_spill_budget_bytes() -> Result<usize> {
    match std::env::var("LANCE_LABEL_LIST_SPILL_BYTES") {
        Ok(raw) => parse_spill_budget_bytes(&raw),
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_SPILL_BUDGET_BYTES),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::invalid_input(
            "LANCE_LABEL_LIST_SPILL_BYTES is not valid unicode; expected a whole number of bytes"
                .to_string(),
        )),
    }
}

/// Split out of [`default_spill_budget_bytes`] so that the rejection is testable
/// without mutating the process environment, which is unsound to do while other
/// tests are running.
fn parse_spill_budget_bytes(raw: &str) -> Result<usize> {
    raw.trim().parse().map_err(|_| {
        Error::invalid_input(format!(
            "LANCE_LABEL_LIST_SPILL_BYTES must be a whole number of bytes, got '{raw}'"
        ))
    })
}

/// The sorted spill files produced by a [`LabelListSpillBuilder`], together
/// with the store they live in.
pub(super) struct LabelListSpills {
    store: Arc<dyn IndexStore>,
    value_type: DataType,
    files: Vec<String>,
    /// Serial number for the next intermediate merge file, so that repeated
    /// fan-in reduction passes never reuse a name.
    next_merge_id: usize,
    /// Kept alive so the spill files outlive the builder that wrote them.
    _tmpdir: Option<TempDir>,
}

impl LabelListSpills {
    #[cfg(test)]
    pub(super) fn files(&self) -> &[String] {
        &self.files
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Add the index being updated as one more sorted merge input.
    ///
    /// It is streamed out through its `index_map`, one bitmap at a time, so the
    /// bitmap payload for every key is not materialized at once. The source
    /// `index_map` and any bitmaps retained by its cache remain outside the
    /// aggregation-state budget.
    /// `NGramIndexBuilder::merge_old_index` feeds the previous index into its
    /// merge the same way.
    ///
    /// This rewrites the index into scratch rather than merging from the index
    /// file in place, which would save a full write and read. Two things block
    /// that: [`open_sorted_bitmap_cursors`] opens every input from one store,
    /// and the old index lives in the index store rather than local scratch; and
    /// cursors require key-sorted files, which LabelList indexes written before
    /// spill-based builds are not. The rewrite is also not where this path's cost
    /// sits -- `merge_index_maps` issues one single-row read per key of the old
    /// index, which dominates either way.
    pub(super) async fn add_existing_index(&mut self, index: &Arc<BitmapIndex>) -> Result<()> {
        let file_name = EXISTING_INDEX_FILE_NAME.to_string();
        let mut writer =
            new_bitmap_batch_writer(self.store.as_ref(), &file_name, index.value_type()).await?;
        merge_index_maps(std::slice::from_ref(index), None, &mut writer, None).await?;
        writer.finish().await?;
        self.files.push(file_name);
        Ok(())
    }

    /// Merge every spill file into `writer` as one ascending `(key, bitmap)`
    /// stream, unioning the row sets of labels that spilled more than once.
    ///
    /// Reduces the file set to at most [`MAX_MERGE_FAN_IN`] entries first, so the
    /// number of cursors open at once is independent of how many times the
    /// builder spilled. This bounds cursor count, not cursor bytes or total
    /// operation memory.
    pub(super) async fn merge_into(&mut self, writer: &mut BitmapBatchWriter) -> Result<()> {
        while self.files.len() > MAX_MERGE_FAN_IN {
            let inputs = std::mem::take(&mut self.files);
            for group in inputs.chunks(MAX_MERGE_FAN_IN) {
                // A trailing group of one is already a sorted file; merging it
                // into a new one would only copy it.
                if let [only] = group {
                    self.files.push(only.clone());
                    continue;
                }
                let merged = self.merge_group(group).await?;
                self.files.push(merged);
            }
        }

        let (mut cursors, mut heap, _) =
            open_sorted_bitmap_cursors(self.store.as_ref(), &self.files).await?;
        drain_sorted_bitmap_cursors(&mut cursors, &mut heap, writer, None).await
    }

    /// Merge `group` into one new sorted spill file and delete the inputs.
    async fn merge_group(&mut self, group: &[String]) -> Result<String> {
        let file_name = format!("{MERGE_FILE_PREFIX}{}.lance", self.next_merge_id);
        self.next_merge_id += 1;

        let mut writer =
            new_bitmap_batch_writer(self.store.as_ref(), &file_name, &self.value_type).await?;
        let (mut cursors, mut heap, _) =
            open_sorted_bitmap_cursors(self.store.as_ref(), group).await?;
        drain_sorted_bitmap_cursors(&mut cursors, &mut heap, &mut writer, None).await?;
        writer.finish().await?;
        // Release the readers before unlinking what they were reading.
        drop(cursors);

        // The merged file supersedes its inputs, so dropping them now holds
        // scratch usage at roughly one copy of the spill rather than one per pass.
        for name in group {
            if let Err(error) = self.store.delete_index_file(name).await {
                warn!(
                    "Failed to delete intermediate label list spill file '{}': {}. \
                     This does not affect the built index, but the spill file \
                     may need manual cleanup.",
                    name, error
                );
            }
        }

        Ok(file_name)
    }
}

/// Accumulates `(label, row address)` pairs into sorted spill files while
/// limiting the estimated memory held by the in-memory aggregation map.
pub(super) struct LabelListSpillBuilder {
    spill_store: Arc<dyn IndexStore>,
    tmpdir: Option<TempDir>,
    value_type: DataType,
    budget_bytes: usize,
    state: BTreeMap<OrderableScalarValue, RowAddrTreeMap>,
    /// Running estimate of the memory held by `state`.
    estimated_bytes: usize,
    spill_files: Vec<String>,
}

impl LabelListSpillBuilder {
    /// Build against a caller-supplied store. Used by tests; production callers
    /// want [`Self::new_local`] so that spilling stays off object storage.
    #[cfg(test)]
    pub(super) fn new(
        spill_store: Arc<dyn IndexStore>,
        value_type: DataType,
        budget_bytes: usize,
    ) -> Self {
        Self {
            spill_store,
            tmpdir: None,
            value_type,
            budget_bytes,
            state: BTreeMap::new(),
            estimated_bytes: 0,
            spill_files: Vec::new(),
        }
    }

    /// Build against a local temporary directory.
    ///
    /// Spill files are scratch: written once, read back once, then removed with
    /// the directory when the resulting [`LabelListSpills`] drops. Putting them
    /// in the index store would put them on object storage, where the round trip
    /// costs far more than the local scratch disk the indexer is already
    /// provisioned with. `NGramIndexBuilder` spills the same way for the same
    /// reason.
    ///
    /// The directory comes from `std::env::temp_dir()`, so `TMPDIR` chooses it,
    /// and it needs real capacity: a build spills roughly one copy of the
    /// label-to-row-set map, and an update spills that plus a full copy of the
    /// index being updated (see [`LabelListSpills::add_existing_index`]). Where
    /// `/tmp` is a memory-backed tmpfs -- common on container images -- scratch
    /// still consumes system memory and defeats the purpose of moving aggregation
    /// state out of RAM, so point `TMPDIR` at disk.
    pub(super) fn new_local(value_type: DataType, budget_bytes: usize) -> Result<Self> {
        if budget_bytes == 0 {
            return Err(Error::invalid_input(
                "LabelList spill budget must be at least one byte, got 0".to_string(),
            ));
        }
        let tmpdir = TempDir::try_new()?;
        let spill_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));
        Ok(Self {
            spill_store,
            tmpdir: Some(tmpdir),
            value_type,
            budget_bytes,
            state: BTreeMap::new(),
            estimated_bytes: 0,
            spill_files: Vec::new(),
        })
    }

    fn spill_filename(id: usize) -> String {
        format!("label-list-spill-{id}.lance")
    }

    pub(super) async fn insert(&mut self, key: ScalarValue, row_addr: u64) -> Result<()> {
        let fragment = RowAddress::from(row_addr).fragment_id();

        // `entry` takes the key by value and descends once. A `get_mut` miss
        // followed by `insert` walks the tree twice, on the path that dominates
        // the high-cardinality case this builder exists for.
        match self.state.entry(OrderableScalarValue(key)) {
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let bitmap = entry.get_mut();
                let new_fragment = bitmap.get(&fragment).is_none();
                bitmap.insert(row_addr);
                self.estimated_bytes += if new_fragment {
                    FRAGMENT_OVERHEAD_BYTES + ROW_ADDR_COST_BYTES
                } else {
                    ROW_ADDR_COST_BYTES
                };
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                self.estimated_bytes +=
                    entry.key().0.size() + LABEL_OVERHEAD_BYTES + ROW_ADDR_COST_BYTES;
                let mut bitmap = RowAddrTreeMap::default();
                bitmap.insert(row_addr);
                entry.insert(bitmap);
            }
        }

        if self.estimated_bytes >= self.budget_bytes {
            self.flush().await?;
        }
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        if self.state.is_empty() {
            return Ok(());
        }
        let file_name = Self::spill_filename(self.spill_files.len());
        let mut writer =
            new_bitmap_batch_writer(self.spill_store.as_ref(), &file_name, &self.value_type)
                .await?;

        // `BTreeMap` iteration is already in key order, which is exactly what
        // the downstream k-way merge requires of each spill file.
        for (key, bitmap) in std::mem::take(&mut self.state) {
            writer.emit(key.0, &bitmap).await?;
        }
        writer.finish().await?;

        self.estimated_bytes = 0;
        self.spill_files.push(file_name);
        Ok(())
    }

    pub(super) async fn finish(mut self) -> Result<LabelListSpills> {
        self.flush().await?;
        Ok(LabelListSpills {
            store: self.spill_store,
            value_type: self.value_type,
            files: self.spill_files,
            next_merge_id: 0,
            _tmpdir: self.tmpdir,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use lance_select::RowAddrTreeMap;
    use rstest::rstest;

    use crate::scalar::bitmap::test_util::{self, row_addrs};

    /// Read every spill file back in file order, which is the order the
    /// downstream k-way merge relies on being sorted.
    async fn read_all_sorted(
        store: &dyn IndexStore,
        files: &[String],
    ) -> Vec<(String, RowAddrTreeMap)> {
        let mut out = Vec::new();
        for name in files {
            out.extend(
                test_util::read_key_bitmaps(store, name)
                    .await
                    .into_iter()
                    .map(|(key, bitmap)| {
                        (
                            key.expect("spill files under test have no null keys"),
                            bitmap,
                        )
                    }),
            );
        }
        out
    }

    /// A budget of one byte forces a flush on essentially every key, which is
    /// the cheapest way to exercise the multi-spill path deterministically.
    #[tokio::test]
    async fn test_spills_when_over_budget() {
        let (_tmpdir, store) = test_util::index_store();
        let mut builder = LabelListSpillBuilder::new(store.clone(), DataType::Utf8, 1);

        for (key, row) in [("b", 1u64), ("a", 2), ("b", 3), ("c", 4)] {
            builder
                .insert(ScalarValue::Utf8(Some(key.to_string())), row)
                .await
                .unwrap();
        }

        let spills = builder.finish().await.unwrap();
        let files = spills.files();
        assert!(
            files.len() > 1,
            "expected multiple spill files, got {files:?}"
        );

        // Every individual spill file must be sorted by key on its own.
        for name in files {
            let contents = read_all_sorted(store.as_ref(), std::slice::from_ref(name)).await;
            let keys: Vec<&String> = contents.iter().map(|(k, _)| k).collect();
            let mut sorted = keys.clone();
            sorted.sort();
            assert_eq!(keys, sorted, "spill file {name} is not sorted by key");
        }
    }

    #[tokio::test]
    async fn test_single_spill_when_under_budget() {
        let (_tmpdir, store) = test_util::index_store();
        let mut builder = LabelListSpillBuilder::new(store.clone(), DataType::Utf8, 1 << 30);

        for (key, row) in [("b", 1u64), ("a", 2), ("b", 3)] {
            builder
                .insert(ScalarValue::Utf8(Some(key.to_string())), row)
                .await
                .unwrap();
        }

        let spills = builder.finish().await.unwrap();
        assert_eq!(spills.files().len(), 1);

        let contents = read_all_sorted(store.as_ref(), spills.files()).await;
        let keys: Vec<String> = contents.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(keys, vec!["a", "b"], "keys are unioned within a spill");
        assert_eq!(
            row_addrs(&contents[1].1),
            vec![1, 3],
            "duplicate keys must union their row sets"
        );
    }

    #[tokio::test]
    async fn test_no_spill_files_when_nothing_inserted() {
        let (_tmpdir, store) = test_util::index_store();
        let builder = LabelListSpillBuilder::new(store, DataType::Utf8, 1 << 30);
        assert!(builder.finish().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_merge_spills_unions_duplicate_keys() {
        let (_tmpdir, store) = test_util::index_store();
        let mut builder = LabelListSpillBuilder::new(store.clone(), DataType::Utf8, 1);
        for (key, row) in [("b", 1u64), ("a", 2), ("b", 3), ("c", 4)] {
            builder
                .insert(ScalarValue::Utf8(Some(key.to_string())), row)
                .await
                .unwrap();
        }
        let mut spills = builder.finish().await.unwrap();
        assert!(spills.files().len() > 1);

        let (_out_tmpdir, out_store) = test_util::index_store();
        let mut writer =
            new_bitmap_batch_writer(out_store.as_ref(), "merged.lance", &DataType::Utf8)
                .await
                .unwrap();
        spills.merge_into(&mut writer).await.unwrap();
        writer.finish().await.unwrap();

        let merged = read_all_sorted(out_store.as_ref(), &["merged.lance".to_string()]).await;
        let keys: Vec<String> = merged.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(
            keys,
            vec!["a", "b", "c"],
            "each key must appear exactly once, in ascending order"
        );
        assert_eq!(row_addrs(&merged[1].1), vec![1, 3]);
    }

    /// The per-fragment charge must cover what touching a new fragment allocates:
    /// a slot in the label's inner map plus a whole new roaring bitmap. Pinned
    /// alongside the per-label arithmetic for the same reason -- the aggregation
    /// estimate is useful only while these constants track the structures they
    /// stand for.
    #[test]
    fn test_fragment_overhead_covers_a_new_fragments_allocations() {
        const BTREE_LEAF_CAPACITY: usize = 11;
        // One slot in the label's fragment map, amortised over a full leaf node.
        let inner_slot = (2 * std::mem::size_of::<usize>()
            + BTREE_LEAF_CAPACITY
                * (std::mem::size_of::<u32>()
                    + std::mem::size_of::<lance_select::RowAddrSelection>()))
            / BTREE_LEAF_CAPACITY;
        assert!(
            FRAGMENT_OVERHEAD_BYTES > inner_slot,
            "FRAGMENT_OVERHEAD_BYTES={FRAGMENT_OVERHEAD_BYTES} must exceed the map \
             slot alone ({inner_slot}); a new fragment also allocates a RoaringBitmap"
        );
    }

    /// A label whose rows are spread across many fragments must be charged for
    /// them. The estimate used to charge ROW_ADDR_COST_BYTES per row address
    /// however it landed, so one hot label over N fragments was under-counted by
    /// more than an order of magnitude and ran the aggregation map past its
    /// configured target.
    #[tokio::test]
    async fn test_budget_accounts_for_a_label_spread_across_fragments() {
        use lance_core::deepsize::DeepSizeOf;

        const FRAGMENTS: u64 = 5_000;

        let (_tmpdir, store) = test_util::index_store();
        // A budget nothing can reach, so `estimated_bytes` is the running
        // accounting for everything the map still holds.
        let mut builder = LabelListSpillBuilder::new(store, DataType::Utf8, usize::MAX);
        for fragment in 0..FRAGMENTS {
            builder
                .insert(
                    ScalarValue::Utf8(Some("hot".to_string())),
                    (fragment << 32) | 1,
                )
                .await
                .unwrap();
        }

        let real: usize = builder
            .state
            .values()
            .map(|bitmap| bitmap.deep_size_of())
            .sum();
        assert!(
            real > 0,
            "the fixture must actually hold something to measure"
        );
        // `deep_size_of` counts neither allocation capacity nor malloc overhead,
        // so it is a floor on the true cost. The estimate must clear it.
        assert!(
            builder.estimated_bytes >= real,
            "estimated {} bytes for a label across {FRAGMENTS} fragments, but it \
             holds at least {real}",
            builder.estimated_bytes
        );
    }

    /// The per-label charge must cover what a single-row label really allocates,
    /// the largest part of which is the inner `BTreeMap`'s leaf node -- allocated
    /// at full capacity to hold one entry. Pinned so that a change to
    /// `RowAddrSelection`'s size cannot silently invalidate the budget.
    #[test]
    fn test_label_overhead_covers_a_single_row_labels_allocations() {
        // Rust's BTreeMap uses B=6, so a leaf holds 2B-1 = 11 entries and the
        // whole node is allocated on the first insert.
        const BTREE_LEAF_CAPACITY: usize = 11;
        let inner_leaf_node = 2 * std::mem::size_of::<usize>()
            + BTREE_LEAF_CAPACITY
                * (std::mem::size_of::<u32>()
                    + std::mem::size_of::<lance_select::RowAddrSelection>());
        let outer_value_slot = std::mem::size_of::<RowAddrTreeMap>();
        assert!(
            LABEL_OVERHEAD_BYTES >= inner_leaf_node + outer_value_slot,
            "LABEL_OVERHEAD_BYTES={LABEL_OVERHEAD_BYTES} must cover the inner leaf \
             node ({inner_leaf_node}) plus the outer value slot \
             ({outer_value_slot}); a budget that under-charges these does not \
             constrain the accumulator"
        );
    }

    /// A malformed budget must be rejected rather than silently replaced by the
    /// default: someone who set the knob specifically to limit aggregation state
    /// should not unknowingly get `DEFAULT_SPILL_BUDGET_BYTES` instead.
    #[rstest]
    #[case::size_suffix("64M")]
    #[case::not_a_number("abc")]
    #[case::negative("-1")]
    #[case::empty("")]
    fn test_spill_budget_rejects_a_malformed_value(#[case] raw: &str) {
        let error = parse_spill_budget_bytes(raw).expect_err("must be rejected");
        assert!(
            error.to_string().contains("LANCE_LABEL_LIST_SPILL_BYTES"),
            "the error must name the variable so it can be fixed, got: {error}"
        );
    }

    #[test]
    fn test_spill_budget_accepts_a_plain_byte_count() {
        assert_eq!(parse_spill_budget_bytes(" 4096 ").unwrap(), 4096);
    }

    /// Zero constrains no aggregation state, so it is an error rather than
    /// something to clamp -- and it does not merely spill often: without the
    /// guard it reaches `lance-io` and panics there.
    #[test]
    fn test_spill_builder_rejects_a_zero_budget() {
        let Err(error) = LabelListSpillBuilder::new_local(DataType::Utf8, 0) else {
            panic!("a zero budget must be rejected")
        };
        assert!(
            error.to_string().contains("at least one byte"),
            "got: {error}"
        );
    }

    /// More spill files than the merge may open at once must be reduced to a
    /// bounded fan-in first, otherwise the number of resident cursor batches
    /// grows with the number of spills.
    #[tokio::test]
    async fn test_merge_reduces_fan_in_to_a_bound() {
        let (_tmpdir, store) = test_util::index_store();
        let mut builder = LabelListSpillBuilder::new(store.clone(), DataType::Utf8, 1);

        // A one-byte budget flushes on every insert, so this is one spill file
        // per key -- comfortably past MAX_MERGE_FAN_IN.
        let keys: Vec<String> = (0..MAX_MERGE_FAN_IN * 3)
            .map(|i| format!("label-{i:04}"))
            .collect();
        for (row, key) in keys.iter().enumerate() {
            builder
                .insert(ScalarValue::Utf8(Some(key.clone())), row as u64)
                .await
                .unwrap();
        }

        let mut spills = builder.finish().await.unwrap();
        assert!(
            spills.files().len() > MAX_MERGE_FAN_IN,
            "fixture must spill more files than the merge may open at once, got {}",
            spills.files().len()
        );

        let (_out_tmpdir, out_store) = test_util::index_store();
        let mut writer =
            new_bitmap_batch_writer(out_store.as_ref(), "merged.lance", &DataType::Utf8)
                .await
                .unwrap();
        spills.merge_into(&mut writer).await.unwrap();
        writer.finish().await.unwrap();

        assert!(
            spills.files().len() <= MAX_MERGE_FAN_IN,
            "the final merge opened {} files at once, above the {MAX_MERGE_FAN_IN} bound",
            spills.files().len()
        );

        // Each pass deletes the files it consumed, so scratch holds one copy of
        // the spill rather than one per pass.
        let live: Vec<String> = store
            .list_files_with_sizes()
            .await
            .unwrap()
            .into_iter()
            .map(|file| file.path)
            .collect();
        assert_eq!(
            live.len(),
            spills.files().len(),
            "consumed spill files must be deleted, found {live:?}"
        );

        let merged = read_all_sorted(out_store.as_ref(), &["merged.lance".to_string()]).await;
        let merged_keys: Vec<String> = merged.iter().map(|(key, _)| key.clone()).collect();
        assert_eq!(
            merged_keys, keys,
            "every key must survive the multi-pass merge, in ascending order"
        );
        for (row, (_, bitmap)) in merged.iter().enumerate() {
            assert_eq!(row_addrs(bitmap), vec![row as u64]);
        }
    }

    /// Nulls are legitimate label values: a null element inside a list survives
    /// unnesting as a null key, so the merge must carry it like any other.
    #[tokio::test]
    async fn test_merge_spills_preserves_null_keys() {
        let (_tmpdir, store) = test_util::index_store();
        let mut builder = LabelListSpillBuilder::new(store.clone(), DataType::Utf8, 1);
        builder
            .insert(ScalarValue::Utf8(Some("a".to_string())), 1)
            .await
            .unwrap();
        builder.insert(ScalarValue::Utf8(None), 2).await.unwrap();
        builder.insert(ScalarValue::Utf8(None), 3).await.unwrap();
        let mut spills = builder.finish().await.unwrap();

        let (_out_tmpdir, out_store) = test_util::index_store();
        let mut writer =
            new_bitmap_batch_writer(out_store.as_ref(), "merged.lance", &DataType::Utf8)
                .await
                .unwrap();
        spills.merge_into(&mut writer).await.unwrap();
        writer.finish().await.unwrap();

        let reader = out_store.open_index_file("merged.lance").await.unwrap();
        let batch = reader.read_range(0..reader.num_rows(), None).await.unwrap();
        let keys: Vec<ScalarValue> = (0..batch.num_rows())
            .map(|idx| ScalarValue::try_from_array(batch.column(0), idx).unwrap())
            .collect();
        assert_eq!(
            keys,
            vec![
                ScalarValue::Utf8(None),
                ScalarValue::Utf8(Some("a".to_string()))
            ],
            "null sorts first and must survive the merge"
        );

        let bitmaps = batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .unwrap();
        let null_rows = RowAddrTreeMap::deserialize_from(bitmaps.value(0)).unwrap();
        assert_eq!(row_addrs(&null_rows), vec![2, 3]);
    }
}
