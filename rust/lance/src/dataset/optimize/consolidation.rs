// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared size policy for byte-aware fragment consolidation.

use lance_table::format::Fragment;
use std::collections::HashSet;

use crate::dataset::WriteParams;

/// Read the main-file size when it is a reliable consolidation signal.
///
/// Multi-file, deleted, overlaid, and obsolete-column layouts need a rewrite
/// estimate rather than a sum of source sizes. Return unknown for those layouts.
/// Blob sidecars are intentionally excluded: the output-file cap applies to the
/// main data file, not to its referenced payloads. No object-store I/O is issued.
///
/// `schema_field_ids` contains the field IDs from the current dataset schema.
///
/// ```
/// use std::collections::HashSet;
/// use std::num::NonZero;
///
/// use lance::dataset::optimize::consolidation::fragment_file_bytes;
/// use lance_file::version::ConcreteFileVersion;
/// use lance_table::format::Fragment;
///
/// let fragment = Fragment::new(0).with_file(
///     "data.lance",
///     vec![7],
///     vec![0],
///     ConcreteFileVersion::V2_2,
///     NonZero::new(1024),
/// );
/// let schema_field_ids = HashSet::from([7]);
/// assert_eq!(fragment_file_bytes(&fragment, &schema_field_ids), Some(1024));
/// ```
pub fn fragment_file_bytes(fragment: &Fragment, schema_field_ids: &HashSet<i32>) -> Option<u64> {
    if fragment.files.len() != 1
        || fragment.deletion_file.is_some()
        || !fragment.overlays.is_empty()
    {
        return None;
    }
    let file = &fragment.files[0];
    if file
        .fields
        .iter()
        .any(|id| *id >= 0 && !schema_field_ids.contains(id))
    {
        return None;
    }
    file.file_size_bytes.get().map(|size| size.get())
}

/// Whether a fragment can benefit from consolidation with its neighbors.
///
/// A fragment is small only while below both its row target and 95% of the
/// effective main-file cap. The 5% tolerance accommodates writer boundaries;
/// it deliberately leaves nearly full files alone. Unknown sizes retain row-only
/// eligibility. This policy never overrides deletion or overlay rewrite rules.
///
/// ```
/// use lance::dataset::optimize::consolidation::needs_consolidation;
/// assert!(!needs_consolidation(10, Some(1024), 1000, Some(1024)));
/// assert!(needs_consolidation(10, None, 1000, Some(1024)));
/// ```
pub fn needs_consolidation(
    rows: u64,
    file_bytes: Option<u64>,
    target_rows: usize,
    max_bytes_per_file: Option<usize>,
) -> bool {
    let cap =
        max_bytes_per_file.unwrap_or_else(|| WriteParams::default().max_bytes_per_file) as u64;
    const FULLNESS_TOLERANCE_DENOMINATOR: u64 = 20;
    let near_full = cap.saturating_sub(cap / FULLNESS_TOLERANCE_DENOMINATOR);
    rows < target_rows as u64 && file_bytes.is_none_or(|bytes| bytes < near_full)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullness_and_unknown_sizes() {
        assert!(needs_consolidation(10, Some(949), 1000, Some(1000)));
        assert!(!needs_consolidation(10, Some(950), 1000, Some(1000)));
        assert!(!needs_consolidation(10, Some(1100), 1000, Some(1000)));
        assert!(needs_consolidation(10, None, 1000, Some(1000)));
        assert!(!needs_consolidation(1000, None, 1000, Some(1000)));
        assert!(needs_consolidation(10, Some(1000), 1000, Some(2000)));
        let default_cap = WriteParams::default().max_bytes_per_file as u64;
        assert!(!needs_consolidation(10, Some(default_cap), 1000, None));
    }
}
