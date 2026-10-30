// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-segment cache identity under a v0 fragment reuse history.

use std::borrow::Cow;

use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::deepsize::DeepSizeOf;
use lance_index::frag_reuse::{
    FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
};
use lance_index::metrics::NoOpMetricsCollector;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::Dataset;
use crate::dataset::index::frag_reuse::{is_index_remap_caught_up, reuse_chain_frag_bitmap};
use crate::index::frag_reuse::{OpenPurpose, fri_cache_id, open_row_id_remapping_with_plan};

/// What [`v0_segment_cache_id`] needs from a v0 history, computed once per
/// history: every fragment the history touches, and a content digest of each
/// version in history order.
pub(super) struct V0HistoryDigests {
    chain: RoaringBitmap,
    versions: Vec<[u8; 32]>,
}

impl V0HistoryDigests {
    pub(super) fn new(details: &FragReuseIndexDetails) -> Self {
        let versions = details.versions.iter().map(version_digest).collect();
        Self {
            chain: reuse_chain_frag_bitmap(&details.versions),
            versions,
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
        self.chain.serialized_size() + self.versions.len() * 32
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
/// A v0 remapper is applied while the segment is decoded, but a version the
/// segment has caught up on moves none of its rows: that is the condition
/// under which the trim may drop the version (`is_index_remap_caught_up`), so
/// decoding with or without it gives the same rows. Keying on the contents of
/// the remaining versions keeps a segment whose fragments the history never
/// touched in the plain namespace through compactions and trims. A segment
/// the history does touch is keyed by every version it has not caught up on,
/// which the gate decides against the fragments of the whole history, so a
/// later version changes its key until it is remapped. A segment with no
/// fragment bitmap cannot be classified and keeps the whole history's
/// identity.
pub(super) fn v0_segment_cache_id(
    history: Uuid,
    details: &FragReuseIndexDetails,
    digests: &V0HistoryDigests,
    segment: &IndexMetadata,
) -> Option<Uuid> {
    let Some(covered) = &segment.fragment_bitmap else {
        return Some(history);
    };
    // Caught up on every version; skips the per-version gate below.
    if covered.is_disjoint(&digests.chain) {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"lance.frag_reuse.v0_segment_cache_id");
    let mut on_path = false;
    for (version, digest) in details.versions.iter().zip(&digests.versions) {
        // A version the gate cannot classify is treated as on the path.
        let caught_up = is_index_remap_caught_up(version, segment, &digests.chain).unwrap_or(false);
        if !caught_up {
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

/// The FRI part of a segment's vector cache keys, resolved the way the open
/// path resolves it (see [`fri_cache_id`]). `None` when it cannot be resolved
/// here.
pub(super) async fn segment_cache_id(
    dataset: &Dataset,
    uuid: &Uuid,
    purpose: OpenPurpose,
) -> Option<Option<Uuid>> {
    let index = dataset
        .load_index_with_purpose(uuid, purpose)
        .await
        .ok()??;
    let resolved =
        open_row_id_remapping_with_plan(dataset, &index, None, purpose, &NoOpMetricsCollector)
            .await
            .ok()?;
    Some(fri_cache_id(&resolved).copied())
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert_eq!(id(vec![rewrite.clone()], &segment(5, Some(&[0]))), None);
        assert_eq!(id(vec![rewrite.clone()], &segment(11, Some(&[6]))), None);
        let on_path = id(vec![rewrite.clone()], &stale);
        assert!(on_path.is_some());
        assert_eq!(id(vec![rewrite.clone()], &stale), on_path);
        // An untouched segment stays in the plain namespace as the history
        // grows.
        assert_eq!(
            id(
                vec![rewrite.clone(), unrelated.clone()],
                &segment(5, Some(&[0]))
            ),
            None
        );
        assert_eq!(id(vec![unrelated.clone()], &stale), None);
        // A touched segment that has not been remapped counts every later
        // version: the gate checks it against the whole history's fragments.
        assert_ne!(id(vec![rewrite.clone(), unrelated], &stale), on_path);
        assert_eq!(id(vec![rewrite.clone()], &segment(5, None)), Some(history));
        // A rewrite group only half inside the segment is rejected by the
        // gate and counts as on the path.
        assert!(id(vec![rewrite], &segment(11, Some(&[4]))).is_some());
    }
}
