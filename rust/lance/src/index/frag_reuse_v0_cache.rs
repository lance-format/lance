// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-segment cache identity under a v0 fragment reuse history.

use std::borrow::Cow;

use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::deepsize::DeepSizeOf;
use lance_index::frag_reuse::FragReuseIndexDetails;
use lance_table::format::IndexMetadata;
use lance_table::format::pb::fragment_reuse_index_details as pb_fri;
use prost::Message;
use uuid::Uuid;

use crate::dataset::index::frag_reuse::index_postdates_rewrite;

/// A content digest of each version of a v0 history, computed once per history.
pub(super) struct V0HistoryDigests {
    versions: Vec<[u8; 32]>,
}

impl V0HistoryDigests {
    pub(super) fn new(details: &FragReuseIndexDetails) -> Self {
        let versions = details
            .versions
            .iter()
            .map(|version| {
                *blake3::hash(&pb_fri::Version::from(version).encode_to_vec()).as_bytes()
            })
            .collect();
        Self { versions }
    }
}

impl DeepSizeOf for V0HistoryDigests {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.versions.len() * 32
    }
}

pub(super) struct V0HistoryDigestsKey<'a> {
    pub(super) history: &'a Uuid,
}

impl CacheKey for V0HistoryDigestsKey<'_> {
    type ValueType = V0HistoryDigests;

    fn key(&self) -> Cow<'_, str> {
        self.history.to_string().into()
    }

    fn type_name() -> &'static str {
        "V0HistoryDigests"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-v0-history-digests", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.history.as_bytes());
    }
}

/// `segment`'s cache id: a hash of the versions that can still move its rows,
/// or `None` when there are none. A version the segment was built after and
/// covers no source of cannot. Disjoint coverage alone is not enough:
/// deferred overlay compaction withdraws coverage while the rows stay.
pub(super) fn v0_segment_cache_id(
    history: Uuid,
    details: &FragReuseIndexDetails,
    digests: &V0HistoryDigests,
    segment: &IndexMetadata,
) -> Option<Uuid> {
    if segment.fragment_bitmap.is_none() {
        return Some(history);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"lance.frag_reuse.v0_segment_cache_id");
    let mut on_path = false;
    for (version, digest) in details.versions.iter().zip(&digests.versions) {
        // A version that cannot be classified counts as on the path.
        if !index_postdates_rewrite(version, segment).unwrap_or(false) {
            on_path = true;
            hasher.update(digest);
        }
    }
    on_path.then(|| {
        let mut identity = [0; 16];
        identity.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
        Uuid::from_bytes(identity)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::index::frag_reuse::tests::{index_covering, reuse_version};

    #[test]
    fn v0_segment_cache_id_tracks_versions_on_the_path() {
        let history = Uuid::new_v4();
        let id = |versions: Vec<_>, segment: &IndexMetadata| {
            let details = FragReuseIndexDetails { versions };
            v0_segment_cache_id(history, &details, &V0HistoryDigests::new(&details), segment)
        };
        // Fragments 4 and 5 were rewritten into 6 at version 10.
        let rewrite = reuse_version(10, &[4, 5], &[6]);
        let unrelated = reuse_version(12, &[1], &[7]);
        let stale = index_covering(5, &[6]);

        // Built after the whole history: plain namespace.
        assert_eq!(id(vec![rewrite.clone()], &index_covering(11, &[6])), None);
        assert_eq!(
            id(
                vec![rewrite.clone(), unrelated.clone()],
                &index_covering(13, &[0])
            ),
            None
        );
        let on_path = id(vec![rewrite.clone()], &stale);
        assert!(on_path.is_some());
        assert_eq!(id(vec![rewrite.clone()], &stale), on_path);
        // Built before the rewrite: keyed even when disjoint.
        assert!(id(vec![rewrite.clone()], &index_covering(5, &[0])).is_some());
        // Trimming an older version leaves the key unchanged.
        let after_rewrite = index_covering(11, &[0]);
        assert!(id(vec![unrelated.clone()], &after_rewrite).is_some());
        assert_eq!(
            id(vec![rewrite.clone(), unrelated.clone()], &after_rewrite),
            id(vec![unrelated.clone()], &after_rewrite)
        );
        // A later version changes a stale segment's key.
        assert_ne!(
            id(vec![rewrite.clone(), unrelated.clone()], &stale),
            on_path
        );
        // A different version at the same position changes it too.
        let mut moved = unrelated.clone();
        moved.groups[0].new_frags[0].physical_rows = 99;
        assert_ne!(
            id(vec![rewrite.clone(), moved], &stale),
            id(vec![rewrite.clone(), unrelated], &stale)
        );
        let no_bitmap = IndexMetadata {
            fragment_bitmap: None,
            ..index_covering(5, &[])
        };
        assert_eq!(id(vec![rewrite.clone()], &no_bitmap), Some(history));
        // A group split by the segment's coverage counts as on the path.
        assert!(id(vec![rewrite], &index_covering(11, &[4])).is_some());
    }
}
