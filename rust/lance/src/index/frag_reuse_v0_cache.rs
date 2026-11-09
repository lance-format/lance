// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-segment cache identity under a v0 fragment reuse history.

use std::borrow::Cow;

use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::deepsize::DeepSizeOf;
use lance_index::frag_reuse::{
    FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::dataset::index::frag_reuse::index_postdates_rewrite;

/// What [`v0_segment_cache_id`] needs from a v0 history, computed once per
/// history: a content digest of each version in history order.
pub(super) struct V0HistoryDigests {
    versions: Vec<[u8; 32]>,
}

impl V0HistoryDigests {
    pub(super) fn new(details: &FragReuseIndexDetails) -> Self {
        Self {
            versions: details.versions.iter().map(version_digest).collect(),
        }
    }
}

/// Hashes every field of `version` in place: converting to protobuf and
/// encoding would copy each group's row-address bitmap twice. Lengths separate
/// the variable-sized fields, and the destructuring fails to compile when a
/// field is added, so a new field cannot silently drop out of the digest.
fn version_digest(version: &FragReuseVersion) -> [u8; 32] {
    let FragReuseVersion {
        dataset_version,
        groups,
    } = version;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&dataset_version.to_le_bytes());
    hasher.update(&(groups.len() as u64).to_le_bytes());
    for group in groups {
        let FragReuseGroup {
            changed_row_addrs,
            old_frags,
            new_frags,
        } = group;
        hasher.update(&(changed_row_addrs.len() as u64).to_le_bytes());
        hasher.update(changed_row_addrs);
        for frags in [old_frags, new_frags] {
            hasher.update(&(frags.len() as u64).to_le_bytes());
            for frag in frags {
                let FragDigest {
                    id,
                    physical_rows,
                    num_deleted_rows,
                } = frag;
                hasher.update(&id.to_le_bytes());
                hasher.update(&(*physical_rows as u64).to_le_bytes());
                hasher.update(&(*num_deleted_rows as u64).to_le_bytes());
            }
        }
    }
    *hasher.finalize().as_bytes()
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

/// The part of a v0 history that can change one segment's decoded rows, as a
/// cache identity, or `None` when no part can.
///
/// A v0 remapper rewrites every stored row while the segment is decoded, but a
/// version the segment was built after (and covers none of the sources of)
/// moves none of its rows, so decoding with or without it gives the same rows.
/// The segment is keyed by the contents of every other version: one built or
/// remapped after the whole history stays in the plain namespace, and
/// trimming versions older than the segment leaves its key unchanged.
///
/// Disjoint coverage is not enough (unlike the trim gate): coverage can be
/// withdrawn while the index data still holds the withdrawn fragments' rows,
/// which the remapper still translates. A segment with no fragment bitmap
/// cannot be classified and keeps the whole history's identity.
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
        // A version that cannot be classified is treated as on the path.
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
    use roaring::RoaringBitmap;

    // Every field reaches the digest, and the length prefixes keep content
    // from shifting across a field or group boundary unnoticed.
    #[test]
    fn version_digest_covers_every_field() {
        let frag = |id| FragDigest {
            id,
            physical_rows: 100,
            num_deleted_rows: 0,
        };
        let group = |addrs: &[u8], old: Vec<FragDigest>, new: Vec<FragDigest>| FragReuseGroup {
            changed_row_addrs: addrs.to_vec(),
            old_frags: old,
            new_frags: new,
        };
        let base = FragReuseVersion {
            dataset_version: 10,
            groups: vec![group(&[1, 2], vec![frag(4)], vec![frag(6)])],
        };
        let mut variants = vec![base.clone()];
        let mut edit = |change: &dyn Fn(&mut FragReuseVersion)| {
            let mut version = base.clone();
            change(&mut version);
            variants.push(version);
        };
        edit(&|v| v.dataset_version = 11);
        edit(&|v| v.groups[0].changed_row_addrs = vec![1, 3]);
        edit(&|v| v.groups[0].old_frags[0].id = 5);
        edit(&|v| v.groups[0].new_frags[0].physical_rows = 99);
        edit(&|v| v.groups[0].new_frags[0].num_deleted_rows = 1);
        edit(&|v| {
            let moved = v.groups[0].new_frags.pop().unwrap();
            v.groups[0].old_frags.push(moved);
        });
        edit(&|v| {
            v.groups = vec![
                group(&[1], vec![frag(4)], vec![]),
                group(&[2], vec![], vec![frag(6)]),
            ]
        });
        let digests: std::collections::HashSet<_> = variants.iter().map(version_digest).collect();
        assert_eq!(digests.len(), variants.len());
    }

    // One row per branch of the per-segment v0 key.
    #[test]
    fn v0_segment_cache_id_tracks_versions_on_the_path() {
        fn version(dataset_version: u64, old: &[u64], new: &[u64]) -> FragReuseVersion {
            let digest = |id: u64| FragDigest {
                id,
                physical_rows: 100,
                num_deleted_rows: 0,
            };
            FragReuseVersion {
                dataset_version,
                groups: vec![FragReuseGroup {
                    changed_row_addrs: Vec::new(),
                    old_frags: old.iter().copied().map(digest).collect(),
                    new_frags: new.iter().copied().map(digest).collect(),
                }],
            }
        }
        fn segment(dataset_version: u64, covered: Option<&[u32]>) -> IndexMetadata {
            IndexMetadata {
                uuid: Uuid::new_v4(),
                fields: vec![0],
                covering_fields: vec![],
                name: "idx".into(),
                dataset_version,
                fragment_bitmap: covered.map(|ids| RoaringBitmap::from_iter(ids.iter().copied())),
                index_details: None,
                index_version: 0,
                created_at: None,
                base_id: None,
                files: None,
                clustering_id: None,
                metadata: Default::default(),
            }
        }
        let history = Uuid::new_v4();
        let id = |versions: Vec<FragReuseVersion>, segment: &IndexMetadata| {
            let details = FragReuseIndexDetails { versions };
            v0_segment_cache_id(history, &details, &V0HistoryDigests::new(&details), segment)
        };
        // A compaction at version 10 rewrote fragments 4 and 5 into 6; the
        // stale segment's coverage moved onto 6 with it.
        let rewrite = version(10, &[4, 5], &[6]);
        let unrelated = version(12, &[1], &[7]);
        let stale = segment(5, Some(&[6]));

        // Built or remapped after the whole history: plain namespace.
        assert_eq!(id(vec![rewrite.clone()], &segment(11, Some(&[6]))), None);
        assert_eq!(
            id(
                vec![rewrite.clone(), unrelated.clone()],
                &segment(13, Some(&[0]))
            ),
            None
        );
        let on_path = id(vec![rewrite.clone()], &stale);
        assert!(on_path.is_some());
        assert_eq!(id(vec![rewrite.clone()], &stale), on_path);
        // Disjoint coverage does not exempt a segment built before the
        // rewrite: the coverage may have been withdrawn while the index data
        // still holds the rewritten rows.
        assert!(id(vec![rewrite.clone()], &segment(5, Some(&[0]))).is_some());
        // Trimming a version older than the segment leaves its key unchanged.
        let after_rewrite = segment(11, Some(&[0]));
        assert!(id(vec![unrelated.clone()], &after_rewrite).is_some());
        assert_eq!(
            id(vec![rewrite.clone(), unrelated.clone()], &after_rewrite),
            id(vec![unrelated.clone()], &after_rewrite)
        );
        // A stale segment counts every later version.
        assert_ne!(id(vec![rewrite.clone(), unrelated], &stale), on_path);
        assert_eq!(id(vec![rewrite.clone()], &segment(5, None)), Some(history));
        // A rewrite group only half inside the segment cannot be classified
        // and counts as on the path.
        assert!(id(vec![rewrite], &segment(11, Some(&[4]))).is_some());
    }
}
