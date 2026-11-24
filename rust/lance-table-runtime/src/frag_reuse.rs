// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fragment reuse history and its table storage.

use lance_table::format::{IndexMetadata, Manifest};
use uuid::Uuid;

mod clone;
mod coverage;
mod deletions;
pub mod history;
mod rewrite;
mod stable_row_ids;
mod storage;
mod trim;

pub use clone::{CloneBaseRemap, collect_tagged_row_map_paths, relocate_tagged_entry_for_clone};
pub use coverage::{undeferrable_rewrite_segments, undeferrable_segments_covering};
pub use deletions::{live_source_row_addrs, normalize_source_fragments};
pub use rewrite::{
    AddedRecords, build_frag_reuse_rewrite_entry, legacy_version_transitions, records_added_since,
};
pub use stable_row_ids::{
    apply_frag_reuse_with_stable_row_ids_flag, has_frag_reuse_with_stable_row_ids,
    validate_stable_row_id_fri_shape,
};
pub use storage::{
    FRAG_REUSE_INLINE_DETAILS_LIMIT, build_frag_reuse_index_metadata,
    build_tagged_frag_reuse_entry, decode_frag_reuse_ledger, decode_frag_reuse_ledger_from_content,
    encode_length_delimited_field, extract_raw_frag_reuse_content, frag_reuse_entry_appending,
    load_frag_reuse_index_details, load_frag_reuse_records, load_raw_frag_reuse_content,
    preload_external_frag_reuse_history, spill_frag_reuse_history,
};
pub use trim::{
    SUPERSEDED_PRUNE_PROPERTY, SUPERSEDED_PRUNE_REBASED_TO_NOOP, TAGGED_TRIM_REBASED_TO_NOOP,
    TaggedTrimOutcome, derive_tagged_trim, is_superseded_prune_transaction,
    is_tagged_trim_operation,
};

/// Cache key for fragment reuse history with `index_version = 0` and the
/// mappings decoded from it.
///
/// A writer can change the history without changing its UUID. Include the
/// history contents and storage location in the key so we don't reuse an old
/// mapping. Referenced files are immutable once published.
pub fn v0_cache_identity(manifest: &Manifest, entry: &IndexMetadata) -> Uuid {
    fn update_sized(hasher: &mut blake3::Hasher, bytes: &[u8]) {
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"lance.frag_reuse.v0_cache_identity");
    hasher.update(entry.uuid.as_bytes());
    if let Some(base_id) = entry.base_id {
        hasher.update(&base_id.to_le_bytes());
        if let Some(base) = manifest.base_paths.get(&base_id) {
            hasher.update(&[u8::from(base.is_dataset_root)]);
            update_sized(&mut hasher, base.path.as_bytes());
        }
    }
    if let Some(details) = &entry.index_details {
        update_sized(&mut hasher, details.type_url.as_bytes());
        update_sized(&mut hasher, &details.value);
    }
    let mut identity = [0; 16];
    identity.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
    Uuid::from_bytes(identity)
}
