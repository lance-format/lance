// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::Result;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::commit::CommitLifecycleHook;
use crate::{CommitIndexHook, Table, WriteSeedHook};

use super::IndexRemapper;

/// A user index segment and its coverage in the planner's fragment ID space.
pub struct CompactionIndexSegment<'a> {
    pub index: &'a IndexMetadata,
    pub coverage: RoaringBitmap,
}

/// Index-specific policy for planning and executing compaction.
#[async_trait]
pub trait CompactionIndexHook: Send + Sync {
    /// Return disjoint groups of compatible segments that may be atomically
    /// replaced. Unlisted segments keep their individual coverage boundaries.
    async fn merge_index_segments(
        &self,
        table: &Table,
        segments: &[CompactionIndexSegment<'_>],
        candidate_windows: &[RoaringBitmap],
    ) -> Vec<Vec<Uuid>>;

    /// Whether this segment constrains a rewrite that cannot replace segments
    /// atomically. The default preserves every segment boundary.
    fn requires_whole_segment(&self, _table: &Table, _index: &IndexMetadata) -> bool {
        true
    }

    /// Whether eager remapping needs the old row addresses on this snapshot.
    /// For a stable-row-ID table, only an address-domain index that supports
    /// eager remapping needs them.
    async fn needs_row_addrs(&self, table: &Table) -> Result<bool>;

    /// Check whether user indices can follow deferred moves on a stable-row-ID
    /// table. The runtime checks the manifest's format and build support first.
    fn validate_stable_row_id_recording(&self, indices: &[IndexMetadata]) -> Result<()>;
}

/// Creates a remapper when an eager compaction commit needs one.
///
/// The runtime calls the factory before materializing the old-to-new address
/// map. A factory may return `None` if the index implementation has no work.
#[async_trait]
pub trait IndexRemapperFactory: Send + Sync {
    /// Whether committing with this factory can merge compatible segments.
    fn supports_vector_segment_replacement(&self) -> bool {
        false
    }

    async fn create_remapper(&self, table: &Table) -> Result<Option<Box<dyn IndexRemapper>>>;
}

/// Engine callbacks bound locally when planning, executing or committing.
///
/// These are never serialized with a task. Without engine bindings, planning
/// preserves segment boundaries and excludes indexed fragments that require an
/// eager remap. Fragment repacks do not need a remapper.
#[derive(Default)]
pub struct CompactionHooks<'a> {
    pub index: Option<&'a dyn CompactionIndexHook>,
    pub remapper: Option<&'a dyn IndexRemapperFactory>,
    pub commit_index: Option<Arc<dyn CommitIndexHook>>,
    pub commit_lifecycle: Option<&'a dyn CommitLifecycleHook>,
    pub write_seeds: Option<Arc<dyn WriteSeedHook>>,
}
