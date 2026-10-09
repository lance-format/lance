// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! System indices: table-level structure persisted as indices.
//!
//! Unlike normal indices, whose internals stay opaque behind
//! [`crate::format::IndexMetadata::index_details`], the table format genuinely
//! interprets the contents of these indices (fragment remapping, row
//! visibility). They therefore live at the table layer.
//!
//! The `Index`-trait adapters for these structs live in `lance-index`, which
//! re-exports the structs defined here.

pub mod frag_reuse;
pub mod mem_wal;

use crate::format::IndexMetadata;
use frag_reuse::FRAG_REUSE_INDEX_NAME;
use mem_wal::MEM_WAL_INDEX_NAME;

/// Whether `index_meta` describes one of the system indices defined in this module.
pub fn is_system_index(index_meta: &IndexMetadata) -> bool {
    index_meta.name == FRAG_REUSE_INDEX_NAME || index_meta.name == MEM_WAL_INDEX_NAME
}
