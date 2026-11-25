// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod writer;

// Deciding which rows an overlay makes stale needs only fragment and index metadata,
// so it lives at the table layer; this module resolves the reads that consume it.
pub use lance_table::format::overlay::staleness::{
    collect_overlay_stale_frags, collect_overlay_stale_rows_for_segment, overlaid_fragments,
};
