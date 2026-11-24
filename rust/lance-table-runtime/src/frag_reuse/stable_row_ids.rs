// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use lance_table::feature_flags::FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
use lance_table::format::{IndexMetadata, Manifest};
use lance_table::system_index::frag_reuse::{
    FRAG_REUSE_INDEX_NAME, is_frag_reuse_index_entry, metadata::uses_tagged_fri,
};

/// An empty fragment reuse index counts.
pub fn has_frag_reuse_with_stable_row_ids(manifest: &Manifest, indices: &[IndexMetadata]) -> bool {
    manifest.uses_stable_row_ids() && indices.iter().any(is_frag_reuse_index_entry)
}

pub fn apply_frag_reuse_with_stable_row_ids_flag(
    manifest: &mut Manifest,
    indices: &[IndexMetadata],
) {
    if has_frag_reuse_with_stable_row_ids(manifest, indices) {
        manifest.reader_feature_flags |= FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
        manifest.writer_feature_flags |= FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
    } else {
        manifest.reader_feature_flags &= !FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
        manifest.writer_feature_flags &= !FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS;
    }
}

/// Verify the table-format boundary for fragment reuse with stable row ids.
/// Returns whether the user index implementations must also be checked.
pub fn validate_stable_row_id_fri_shape(
    manifest: &Manifest,
    indices: &[IndexMetadata],
    is_supported: bool,
) -> Result<bool> {
    if !has_frag_reuse_with_stable_row_ids(manifest, indices) {
        return Ok(false);
    }
    let version = manifest.version;
    if !is_supported {
        return Err(Error::not_supported(format!(
            "Cannot commit version {version}: the dataset uses stable row IDs and carries a \
             fragment reuse index, which this build of Lance does not support"
        )));
    }
    let frag_reuse_entries: Vec<_> = indices
        .iter()
        .filter(|index| is_frag_reuse_index_entry(index))
        .collect();
    if frag_reuse_entries.len() != 1 {
        return Err(Error::invalid_input(format!(
            "Cannot commit version {version}: the dataset carries {} fragment reuse index \
             entries, but exactly one is allowed",
            frag_reuse_entries.len()
        )));
    }
    // A tagged history (index_version 1) is allowed beside stable row ids as
    // long as every index stores row addresses, which the check below holds
    // every index to.
    let frag_reuse_version = frag_reuse_entries[0].index_version;
    if !matches!(frag_reuse_version, 0 | 1) {
        return Err(Error::invalid_input(format!(
            "Cannot commit version {version}: the dataset uses stable row IDs and carries a \
             fragment reuse index with index_version {frag_reuse_version}, which this build of \
             Lance does not support"
        )));
    }
    Ok(true)
}

/// Check the format and build constraints before recording a compaction
/// on a stable-row-ID table. User index compatibility is checked separately.
pub fn validate_stable_row_id_recording_format(
    manifest: &Manifest,
    indices: &[IndexMetadata],
    is_supported: bool,
) -> Result<()> {
    if !is_supported {
        return Err(Error::not_supported(
            "Cannot compact with defer_index_remap: the dataset uses stable row IDs, and \
             recording their row-address moves in a fragment reuse index is not supported by \
             this build of Lance",
        ));
    }
    // The tagged-history flag is sticky, so it can outlive a trim and a
    // migration to stable row ids.
    let frag_reuse_entry = indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME);
    if uses_tagged_fri(manifest, frag_reuse_entry) {
        return Err(Error::not_supported(
            "Cannot compact with defer_index_remap: the dataset uses stable row IDs, and its \
             manifest requires a tagged fragment reuse history, which stable row IDs do not \
             support",
        ));
    }
    Ok(())
}
