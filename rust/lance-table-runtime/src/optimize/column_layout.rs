// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-fragment column-layout statistics.
//!
//! An `add_columns` call that writes data files writes one per fragment, so a
//! wide fragment can end up split across many small files. These stats, from
//! [`Table::column_layout_stats`], make that layout visible. The default
//! compaction planner considers repacking a fragment's columns when its
//! `live_file_count` is above
//! [`max_data_files_per_fragment`](super::CompactionOptions::max_data_files_per_fragment);
//! a custom
//! [`CompactionPlanner`](super::CompactionPlanner) can use
//! the rest.

use std::collections::HashSet;

use lance_table::format::overlay::TOMBSTONE_FIELD_ID;

use crate::Table;

/// Column-layout statistics for a single fragment.
///
/// This is an experimental API. It may change at any time.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentColumnLayoutStats {
    /// The fragment these stats describe.
    pub fragment_id: u64,
    /// Number of data files holding at least one column of the dataset schema.
    /// A file holding only tombstones, dropped columns, or the fragment's
    /// spilled row lineage holds no schema column and is not counted.
    pub live_file_count: usize,
    /// Recorded size in bytes of each data file, in the fragment's file order.
    /// `None` when the manifest has no size for the file.
    pub file_sizes: Vec<Option<u64>>,
    /// Number of fields of the dataset schema each data file holds, in the
    /// fragment's file order. A file counted in `live_file_count` holds at
    /// least one.
    pub fields_per_file: Vec<usize>,
    /// The share of the fragment's field slots that hold no live data: slots
    /// tombstoned by a column update or repack, or left by a dropped column.
    /// The reserved ids of spilled row lineage are not counted on either side.
    /// A fragment rewrite reclaims these slots.
    pub dead_slot_ratio: f64,
    /// Number of overlay files attached to the fragment.
    pub overlay_count: usize,
}

impl Table {
    /// Per-fragment column-layout stats, in manifest fragment order.
    ///
    /// Reads only fragment metadata, never a data file.
    ///
    /// This is an experimental API. It may change at any time.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance_table_runtime::Table;
    /// # fn example(dataset: &Table) {
    /// for stats in dataset.column_layout_stats() {
    ///     println!(
    ///         "fragment {}: {} live files",
    ///         stats.fragment_id, stats.live_file_count
    ///     );
    /// }
    /// # }
    /// ```
    pub fn column_layout_stats(&self) -> Vec<FragmentColumnLayoutStats> {
        let schema_ids: HashSet<i32> = self
            .schema()
            .fields_pre_order()
            .map(|field| field.id)
            .collect();
        self.manifest
            .fragments
            .iter()
            .map(|fragment| {
                let fields_per_file: Vec<usize> = fragment
                    .files
                    .iter()
                    .map(|file| {
                        file.fields
                            .iter()
                            .filter(|id| schema_ids.contains(id))
                            .count()
                    })
                    .collect();
                // Field slots of user columns, live or dead; the other negative
                // ids are spilled row lineage and are left out. Dead slots are
                // counted directly rather than as user minus live, since
                // `Schema::validate` lets a schema carry a lineage id.
                let user_ids = || {
                    fragment
                        .files
                        .iter()
                        .flat_map(|file| file.fields.iter())
                        .filter(|id| **id >= 0 || **id == TOMBSTONE_FIELD_ID)
                };
                let user_slots = user_ids().count();
                let dead_slots = user_ids().filter(|id| !schema_ids.contains(id)).count();
                FragmentColumnLayoutStats {
                    fragment_id: fragment.id,
                    live_file_count: fields_per_file.iter().filter(|count| **count > 0).count(),
                    file_sizes: fragment
                        .files
                        .iter()
                        .map(|file| file.file_size_bytes.get().map(|size| size.get()))
                        .collect(),
                    fields_per_file,
                    dead_slot_ratio: if user_slots == 0 {
                        0.0
                    } else {
                        dead_slots as f64 / user_slots as f64
                    },
                    overlay_count: fragment.overlays.len(),
                }
            })
            .collect()
    }
}
