// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, io::Cursor, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch, UInt64Array};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::row_addr_remap::{GroupInputWithLayout, RowAddrRemap};
use lance_core::{Error, Result};
use lance_select::RowAddrTreeMap;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::format::pb::fragment_reuse_index_details::InlineContent;
use crate::format::{ExternalFile, Fragment, pb};

pub const FRAG_REUSE_INDEX_NAME: &str = "__lance_frag_reuse";
pub const FRAG_REUSE_DETAILS_FILE_NAME: &str = "details.binpb";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragDigest {
    pub id: u64,
    pub physical_rows: usize,
    pub num_deleted_rows: usize,
}

impl From<&FragDigest> for pb::fragment_reuse_index_details::FragmentDigest {
    fn from(digest: &FragDigest) -> Self {
        Self {
            id: digest.id,
            physical_rows: digest.physical_rows as u64,
            num_deleted_rows: digest.num_deleted_rows as u64,
        }
    }
}

impl From<&Fragment> for FragDigest {
    fn from(fragment: &Fragment) -> Self {
        Self {
            id: fragment.id,
            physical_rows: fragment
                .physical_rows
                .expect("Fragment doesn't have physical rows recorded"),
            num_deleted_rows: fragment
                .deletion_file
                .as_ref()
                .and_then(|d| d.num_deleted_rows)
                .unwrap_or(0),
        }
    }
}

impl TryFrom<pb::fragment_reuse_index_details::FragmentDigest> for FragDigest {
    type Error = Error;

    fn try_from(digest: pb::fragment_reuse_index_details::FragmentDigest) -> Result<Self> {
        Ok(Self {
            id: digest.id,
            physical_rows: digest.physical_rows as usize,
            num_deleted_rows: digest.num_deleted_rows as usize,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseGroup {
    pub changed_row_addrs: Vec<u8>,
    pub old_frags: Vec<FragDigest>,
    pub new_frags: Vec<FragDigest>,
}

impl From<&FragReuseGroup> for pb::fragment_reuse_index_details::Group {
    fn from(group: &FragReuseGroup) -> Self {
        Self {
            changed_row_addrs: group.changed_row_addrs.clone(),
            old_fragments: group.old_frags.iter().map(|f| f.into()).collect(),
            new_fragments: group.new_frags.iter().map(|f| f.into()).collect(),
        }
    }
}

impl TryFrom<pb::fragment_reuse_index_details::Group> for FragReuseGroup {
    type Error = Error;

    fn try_from(group: pb::fragment_reuse_index_details::Group) -> Result<Self> {
        Ok(Self {
            changed_row_addrs: group.changed_row_addrs,
            old_frags: group
                .old_fragments
                .into_iter()
                .map(FragDigest::try_from)
                .collect::<Result<_>>()?,
            new_frags: group
                .new_fragments
                .into_iter()
                .map(FragDigest::try_from)
                .collect::<Result<_>>()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseVersion {
    pub dataset_version: u64,
    pub groups: Vec<FragReuseGroup>,
}

impl From<&FragReuseVersion> for pb::fragment_reuse_index_details::Version {
    fn from(version: &FragReuseVersion) -> Self {
        Self {
            dataset_version: version.dataset_version,
            groups: version.groups.iter().map(|g| g.into()).collect(),
        }
    }
}

impl TryFrom<pb::fragment_reuse_index_details::Version> for FragReuseVersion {
    type Error = Error;

    fn try_from(version: pb::fragment_reuse_index_details::Version) -> Result<Self> {
        Ok(Self {
            dataset_version: version.dataset_version,
            groups: version
                .groups
                .into_iter()
                .map(FragReuseGroup::try_from)
                .collect::<Result<_>>()?,
        })
    }
}

impl FragReuseVersion {
    pub fn old_frag_ids(&self) -> Vec<u64> {
        self.groups
            .iter()
            .flat_map(|g| g.old_frags.iter().map(|f| f.id))
            .collect::<Vec<_>>()
    }

    pub fn new_frag_ids(&self) -> Vec<u64> {
        self.groups
            .iter()
            .flat_map(|g| g.new_frags.iter().map(|f| f.id))
            .collect::<Vec<_>>()
    }

    pub fn new_frag_bitmap(&self) -> RoaringBitmap {
        RoaringBitmap::from_iter(self.new_frag_ids().iter().map(|&id| id as u32))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub enum FragReuseIndexDetailsContentType {
    Inline(FragReuseIndexDetails),
    External(ExternalFile),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
pub struct FragReuseIndexDetails {
    pub versions: Vec<FragReuseVersion>,
}

impl From<&FragReuseIndexDetails> for InlineContent {
    fn from(details: &FragReuseIndexDetails) -> Self {
        let mut versions: Vec<pb::fragment_reuse_index_details::Version> =
            details.versions.iter().map(|m| m.into()).collect();
        // sort from oldest to latest version
        versions.sort_by_key(|v| v.dataset_version);
        Self { versions }
    }
}

impl TryFrom<InlineContent> for FragReuseIndexDetails {
    type Error = Error;

    fn try_from(content: InlineContent) -> Result<Self> {
        Ok(Self {
            versions: content
                .versions
                .into_iter()
                .map(|m| m.try_into())
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl FragReuseIndexDetails {
    pub fn new_frag_bitmap(&self) -> RoaringBitmap {
        RoaringBitmap::from_iter(
            self.versions
                .iter()
                .flat_map(|v| v.new_frag_ids().into_iter().map(|id| id as u32)),
        )
    }
}

/// An index that stores materialized row ID maps.
///
/// This type is retained for API and serde compatibility. Dataset loading uses
/// [`CompactFragReuseIndex`] so persisted FRI details are not expanded into a
/// hash-map entry for every affected row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FragReuseIndex {
    pub uuid: Uuid,
    pub row_id_maps: Vec<HashMap<u64, Option<u64>>>,
    pub details: FragReuseIndexDetails,
}

impl DeepSizeOf for FragReuseIndex {
    fn deep_size_of_children(&self, cx: &mut Context) -> usize {
        self.row_id_maps.deep_size_of_children(cx) + self.details.deep_size_of_children(cx)
    }
}

impl FragReuseIndex {
    pub fn new(
        uuid: Uuid,
        row_id_maps: Vec<HashMap<u64, Option<u64>>>,
        details: FragReuseIndexDetails,
    ) -> Self {
        Self {
            uuid,
            row_id_maps,
            details,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.row_id_maps.iter().all(HashMap::is_empty)
    }

    pub fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        let mut mapped = Some(row_id);
        for row_id_map in &self.row_id_maps {
            if let Some(current) = mapped {
                mapped = row_id_map.get(&current).copied().unwrap_or(mapped);
            }
        }
        mapped
    }

    pub fn remap_row_ids_in_place(&self, row_ids: &mut [Option<u64>]) {
        for row_id_map in &self.row_id_maps {
            for row_id in row_ids.iter_mut() {
                if let Some(current) = *row_id
                    && let Some(mapped) = row_id_map.get(&current)
                {
                    *row_id = *mapped;
                }
            }
        }
    }

    pub fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        RowAddrTreeMap::from_iter(
            row_addrs
                .row_addrs()
                .unwrap()
                .filter_map(|addr| self.remap_row_id(u64::from(addr))),
        )
    }

    pub fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        RoaringTreemap::from_iter(row_ids.iter().filter_map(|addr| self.remap_row_id(addr)))
    }

    pub fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        remap_row_ids_record_batch(batch, row_id_idx, |row_ids| {
            self.remap_row_ids_in_place(row_ids)
        })
    }

    pub fn remap_row_ids_array(&self, array: ArrayRef) -> PrimitiveArray<UInt64Type> {
        remap_row_ids_array(array, |row_ids| self.remap_row_ids_in_place(row_ids))
    }

    pub fn remap_fragment_bitmap(&self, fragment_bitmap: &mut RoaringBitmap) -> Result<()> {
        remap_fragment_bitmap(&self.details, fragment_bitmap)
    }
}

/// A compact row-address remap chain for deferred compactions.
///
/// Each FRI version retains rewritten-row bitmaps and fragment layouts. Queries
/// use bitmap rank plus the ordered new-fragment ranges instead of storing one
/// hash-map entry per affected row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactFragReuseIndex {
    pub uuid: Uuid,
    row_addr_remap: RowAddrRemap,
    pub details: FragReuseIndexDetails,
}

impl DeepSizeOf for CompactFragReuseIndex {
    fn deep_size_of_children(&self, cx: &mut Context) -> usize {
        self.row_addr_remap.deep_size_of_children(cx) + self.details.deep_size_of_children(cx)
    }
}

impl CompactFragReuseIndex {
    #[doc(hidden)]
    pub fn from_row_id_maps(
        uuid: Uuid,
        row_id_maps: Vec<HashMap<u64, Option<u64>>>,
        details: FragReuseIndexDetails,
    ) -> Self {
        Self {
            uuid,
            row_addr_remap: RowAddrRemap::chained(
                row_id_maps.into_iter().map(RowAddrRemap::direct),
            ),
            details,
        }
    }

    /// Build a queryable index directly from serialized FRI details without
    /// expanding each affected row into a hash map.
    pub fn try_new(uuid: Uuid, details: FragReuseIndexDetails) -> Result<Self> {
        let mut version_remaps = Vec::with_capacity(details.versions.len());
        for (version_idx, version) in details.versions.iter().enumerate() {
            let mut groups = Vec::with_capacity(version.groups.len());
            for (group_idx, group) in version.groups.iter().enumerate() {
                let changed_row_addrs = RoaringTreemap::deserialize_from(Cursor::new(
                    &group.changed_row_addrs,
                ))
                .map_err(|error| {
                    Error::index(format!(
                        "failed to deserialize changed row addresses for FRI version {version_idx}, group {group_idx}: {error}"
                    ))
                })?;
                let old_frags = group
                    .old_frags
                    .iter()
                    .map(|frag| fragment_layout(frag, "old", version_idx, group_idx))
                    .collect::<Result<Vec<_>>>()?;
                let new_frags = group
                    .new_frags
                    .iter()
                    .map(|frag| fragment_layout(frag, "new", version_idx, group_idx))
                    .collect::<Result<Vec<_>>>()?;
                groups.push(GroupInputWithLayout {
                    rewritten_old_row_addrs: changed_row_addrs,
                    old_frags,
                    new_frags,
                });
            }
            let remap = RowAddrRemap::compact_with_layout(groups).map_err(|error| {
                Error::index(format!(
                    "failed to build compact remap for FRI version {version_idx}: {error}"
                ))
            })?;
            version_remaps.push(remap);
        }

        Ok(Self {
            uuid,
            row_addr_remap: RowAddrRemap::chained(version_remaps),
            details,
        })
    }

    /// The ordered remap chain used by index and transaction remapping paths.
    pub fn row_addr_remap(&self) -> &RowAddrRemap {
        &self.row_addr_remap
    }

    /// Returns whether the index contains no row-address remapping.
    pub fn is_empty(&self) -> bool {
        self.row_addr_remap.is_empty()
    }

    pub fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        self.row_addr_remap.get(row_id).unwrap_or(Some(row_id))
    }

    /// Apply all FRI versions to row addresses in place. `None` values remain
    /// deleted and missing mappings pass through unchanged.
    pub fn remap_row_ids_in_place(&self, row_ids: &mut [Option<u64>]) {
        self.row_addr_remap.remap_in_place(row_ids);
    }

    pub fn remap_row_addrs_tree_map(&self, row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
        RowAddrTreeMap::from_iter(row_addrs.row_addrs().unwrap().filter_map(|addr| {
            let addr_as_u64 = u64::from(addr);
            self.remap_row_id(addr_as_u64)
        }))
    }

    pub fn remap_row_ids_roaring_tree_map(&self, row_ids: &RoaringTreemap) -> RoaringTreemap {
        RoaringTreemap::from_iter(row_ids.iter().filter_map(|addr| self.remap_row_id(addr)))
    }

    /// Remap a record batch that contains a row_id column at index `row_id_idx`.
    /// Every other column (there may be one, as for scalar indexes -- `(value,
    /// row_id)` -- or several, as for a covered vector index's storage --
    /// `(row_id, code, <covering...>)`) is row-aligned to the surviving,
    /// remapped row ids by the same take.
    pub fn remap_row_ids_record_batch(
        &self,
        batch: RecordBatch,
        row_id_idx: usize,
    ) -> Result<RecordBatch> {
        remap_row_ids_record_batch(batch, row_id_idx, |row_ids| {
            self.remap_row_ids_in_place(row_ids)
        })
    }

    pub fn remap_row_ids_array(&self, array: ArrayRef) -> PrimitiveArray<UInt64Type> {
        remap_row_ids_array(array, |row_ids| self.remap_row_ids_in_place(row_ids))
    }

    pub fn remap_fragment_bitmap(&self, fragment_bitmap: &mut RoaringBitmap) -> Result<()> {
        remap_fragment_bitmap(&self.details, fragment_bitmap)
    }
}

fn remap_row_ids_record_batch(
    batch: RecordBatch,
    row_id_idx: usize,
    remap: impl FnOnce(&mut [Option<u64>]),
) -> Result<RecordBatch> {
    // Every column but `_rowid` is carried through row-aligned, however many there are.
    // A covered index's storage batch holds its covering ("included") columns next to the
    // codes, so this cannot assume the two-column `[_rowid, <codes>]` shape.
    let row_ids = batch.column(row_id_idx).as_primitive::<UInt64Type>();
    let mut remapped_row_ids = row_ids
        .values()
        .iter()
        .copied()
        .map(Some)
        .collect::<Vec<_>>();
    remap(&mut remapped_row_ids);
    let (keep_indices, new_row_ids): (Vec<u64>, Vec<u64>) = remapped_row_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, new_id)| new_id.map(|new_id| (idx as u64, new_id)))
        .unzip();
    let keep_indices = UInt64Array::from_iter_values(keep_indices);
    let new_row_ids: ArrayRef = Arc::new(UInt64Array::from_iter_values(new_row_ids));

    let columns = batch
        .columns()
        .iter()
        .enumerate()
        .map(|(idx, column)| {
            if idx == row_id_idx {
                Ok(new_row_ids.clone())
            } else {
                Ok(arrow::compute::take(column, &keep_indices, None)?)
            }
        })
        .collect::<Result<Vec<ArrayRef>>>()?;
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

fn remap_row_ids_array(
    array: ArrayRef,
    remap: impl FnOnce(&mut [Option<u64>]),
) -> PrimitiveArray<UInt64Type> {
    let primitive_array = array
        .as_any()
        .downcast_ref::<PrimitiveArray<UInt64Type>>()
        .expect("expected row IDs to be uint64 array");
    let mut remapped = (0..primitive_array.len())
        .map(|i| {
            if primitive_array.is_null(i) {
                None
            } else {
                Some(primitive_array.value(i))
            }
        })
        .collect::<Vec<_>>();
    remap(&mut remapped);
    PrimitiveArray::from(remapped)
}

fn remap_fragment_bitmap(
    details: &FragReuseIndexDetails,
    fragment_bitmap: &mut RoaringBitmap,
) -> Result<()> {
    for version in details.versions.iter() {
        for group in version.groups.iter() {
            let mut removed = 0;
            for old_frag in group.old_frags.iter() {
                if fragment_bitmap.remove(old_frag.id as u32) {
                    removed += 1;
                }
            }

            if removed > 0 {
                if removed != group.old_frags.len() {
                    // Straddle: the index covered only part of this rewrite
                    // group. Caused by the bug fixed in
                    // <https://github.com/lance-format/lance/pull/6610>.
                    // We've already removed the indexed old_frags from the
                    // bitmap above; deliberately do NOT insert new_frags,
                    // since the merged fragment also contains rows that
                    // were never indexed. Affected rows fall through to
                    // flat scan until the next optimize_indices. The fix
                    // is persisted on the next write via build_manifest.
                    tracing::warn!(
                        "Healing straddling fragment-reuse rewrite group in index bitmap: \
                             group {:?} was only partially indexed ({} of {} old fragments). \
                             Affected rows will use flat scan until the next optimize_indices.",
                        group.old_frags,
                        removed,
                        group.old_frags.len(),
                    );
                    continue;
                }

                for new_frag in group.new_frags.iter() {
                    fragment_bitmap.insert(new_frag.id as u32);
                }
            }
        }
    }
    Ok(())
}

fn fragment_layout(
    frag: &FragDigest,
    role: &str,
    version_idx: usize,
    group_idx: usize,
) -> Result<(u32, u32)> {
    let fragment_id = u32::try_from(frag.id).map_err(|_| {
        Error::index(format!(
            "FRI version {version_idx}, group {group_idx} has {role} fragment id {} outside the row-address range",
            frag.id
        ))
    })?;
    let physical_rows = u32::try_from(frag.physical_rows).map_err(|_| {
        Error::index(format!(
            "FRI version {version_idx}, group {group_idx} has {role} fragment {fragment_id} with physical_rows={} outside the row-address range",
            frag.physical_rows
        ))
    })?;
    Ok((fragment_id, physical_rows))
}

#[cfg(test)]
mod tests {

    use super::*;
    use rstest::rstest;

    fn addr(fragment_id: u32, offset: u32) -> u64 {
        u64::from(lance_core::utils::address::RowAddress::new_from_parts(
            fragment_id,
            offset,
        ))
    }

    fn serialize_changed(addrs: impl IntoIterator<Item = u64>) -> Vec<u8> {
        let changed = RoaringTreemap::from_iter(addrs);
        let mut bytes = Vec::with_capacity(changed.serialized_size());
        changed.serialize_into(&mut bytes).unwrap();
        bytes
    }

    fn digest(id: u64, physical_rows: usize) -> FragDigest {
        FragDigest {
            id,
            physical_rows,
            num_deleted_rows: 0,
        }
    }

    #[test]
    fn test_compact_fri_tristate_one_to_many_and_chain() {
        let details = FragReuseIndexDetails {
            versions: vec![
                FragReuseVersion {
                    dataset_version: 1,
                    groups: vec![
                        // One old fragment is split into two new fragments.
                        FragReuseGroup {
                            changed_row_addrs: serialize_changed([
                                addr(1, 0),
                                addr(1, 2),
                                addr(1, 3),
                            ]),
                            old_frags: vec![digest(1, 4)],
                            new_frags: vec![digest(10, 1), digest(11, 2)],
                        },
                        // A separate rewrite group deletes an entire fragment.
                        FragReuseGroup {
                            changed_row_addrs: serialize_changed([]),
                            old_frags: vec![digest(3, 2)],
                            new_frags: vec![],
                        },
                    ],
                },
                FragReuseVersion {
                    dataset_version: 2,
                    groups: vec![FragReuseGroup {
                        changed_row_addrs: serialize_changed([addr(10, 0), addr(11, 1)]),
                        old_frags: vec![digest(10, 1), digest(11, 2)],
                        new_frags: vec![digest(20, 2)],
                    }],
                },
            ],
        };
        let details = FragReuseIndexDetails::try_from(InlineContent::from(&details)).unwrap();
        let fri = CompactFragReuseIndex::try_new(Uuid::new_v4(), details).unwrap();

        // Surviving rows follow both versions in oldest-to-newest order.
        assert_eq!(fri.remap_row_id(addr(1, 0)), Some(addr(20, 0)));
        assert_eq!(fri.remap_row_id(addr(1, 3)), Some(addr(20, 1)));
        // Deletes can happen in either the first or a later version.
        assert_eq!(fri.remap_row_id(addr(1, 1)), None);
        assert_eq!(fri.remap_row_id(addr(1, 2)), None);
        assert_eq!(fri.remap_row_id(addr(3, 0)), None);
        // Uncovered fragments and out-of-range offsets retain the existing
        // missing-map pass-through semantics.
        assert_eq!(fri.remap_row_id(addr(2, 0)), Some(addr(2, 0)));
        assert_eq!(fri.remap_row_id(addr(1, 4)), Some(addr(1, 4)));

        let mut batch = vec![
            Some(addr(1, 0)),
            Some(addr(1, 1)),
            Some(addr(1, 2)),
            Some(addr(1, 3)),
            Some(addr(2, 0)),
            None,
        ];
        fri.remap_row_ids_in_place(&mut batch);
        assert_eq!(
            batch,
            vec![
                Some(addr(20, 0)),
                None,
                None,
                Some(addr(20, 1)),
                Some(addr(2, 0)),
                None,
            ]
        );
    }

    #[test]
    fn test_compact_fri_rejects_invalid_changed_row_bitmap() {
        let details = FragReuseIndexDetails {
            versions: vec![FragReuseVersion {
                dataset_version: 1,
                groups: vec![FragReuseGroup {
                    changed_row_addrs: vec![1, 2, 3],
                    old_frags: vec![digest(1, 1)],
                    new_frags: vec![digest(2, 1)],
                }],
            }],
        };
        let error = CompactFragReuseIndex::try_new(Uuid::new_v4(), details).unwrap_err();
        assert!(matches!(error, Error::Index { .. }));
        assert!(
            error
                .to_string()
                .contains("failed to deserialize changed row addresses for FRI version 0, group 0"),
            "{error}"
        );
    }

    #[rstest]
    #[case::unknown_fragment(
        vec![addr(2, 0)],
        vec![digest(1, 1)],
        "from fragments [2] not in its old fragments"
    )]
    #[case::offset_out_of_range(
        vec![addr(1, 1)],
        vec![digest(1, 1)],
        "row offset outside old fragment 1 with physical_rows=1"
    )]
    #[case::duplicate_old_fragment(
        vec![addr(1, 0)],
        vec![digest(1, 1), digest(1, 1)],
        "old fragment 1 more than once"
    )]
    fn test_compact_fri_preserves_layout_validation(
        #[case] changed_addrs: Vec<u64>,
        #[case] old_frags: Vec<FragDigest>,
        #[case] expected_message: &str,
    ) {
        let details = FragReuseIndexDetails {
            versions: vec![FragReuseVersion {
                dataset_version: 1,
                groups: vec![FragReuseGroup {
                    changed_row_addrs: serialize_changed(changed_addrs),
                    old_frags,
                    new_frags: vec![digest(10, 1)],
                }],
            }],
        };

        let error = CompactFragReuseIndex::try_new(Uuid::new_v4(), details).unwrap_err();
        assert!(matches!(error, Error::Index { .. }));
        let message = error.to_string();
        assert!(message.contains("FRI version 0"), "{message}");
        assert!(message.contains("rewrite group 0"), "{message}");
        assert!(message.contains(expected_message), "{message}");
    }

    #[tokio::test]
    async fn test_serialize_deserialize_index_details() {
        // Create sample FragReuseVersions with different dataset versions
        let version1 = FragReuseVersion {
            dataset_version: 2,
            groups: vec![FragReuseGroup {
                changed_row_addrs: vec![1, 2, 3],
                old_frags: vec![FragDigest {
                    id: 1,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }],
                new_frags: vec![
                    FragDigest {
                        id: 2,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                    FragDigest {
                        id: 3,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                ],
            }],
        };

        let version2 = FragReuseVersion {
            dataset_version: 1,
            groups: vec![FragReuseGroup {
                changed_row_addrs: vec![4, 5, 6],
                old_frags: vec![FragDigest {
                    id: 2,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }],
                new_frags: vec![
                    FragDigest {
                        id: 4,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                    FragDigest {
                        id: 5,
                        physical_rows: 1,
                        num_deleted_rows: 0,
                    },
                ],
            }],
        };

        // Create FragReuseIndexDetails with versions in reverse order
        let details = FragReuseIndexDetails {
            versions: vec![version1, version2],
        };

        // Convert to protobuf format
        let inline_content: InlineContent = (&details).into();

        // Convert back to FragReuseIndexDetails
        let roundtrip_details = FragReuseIndexDetails::try_from(inline_content).unwrap();

        // Verify the roundtrip
        assert_eq!(roundtrip_details.versions.len(), 2);

        // Verify versions are sorted by dataset_version (oldest to latest)
        assert_eq!(roundtrip_details.versions[0].dataset_version, 1);
        assert_eq!(
            roundtrip_details.versions[0].groups[0].changed_row_addrs,
            vec![4, 5, 6]
        );
        assert_eq!(
            roundtrip_details.versions[0].groups[0].new_frags,
            vec![
                FragDigest {
                    id: 4,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                },
                FragDigest {
                    id: 5,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }
            ]
        );
        assert_eq!(
            roundtrip_details.versions[0].groups[0].old_frags,
            vec![FragDigest {
                id: 2,
                physical_rows: 1,
                num_deleted_rows: 0,
            }]
        );

        assert_eq!(roundtrip_details.versions[1].dataset_version, 2);
        assert_eq!(
            roundtrip_details.versions[1].groups[0].changed_row_addrs,
            vec![1, 2, 3]
        );
        assert_eq!(
            roundtrip_details.versions[1].groups[0].new_frags,
            vec![
                FragDigest {
                    id: 2,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                },
                FragDigest {
                    id: 3,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                }
            ]
        );
        assert_eq!(
            roundtrip_details.versions[1].groups[0].old_frags,
            vec![FragDigest {
                id: 1,
                physical_rows: 1,
                num_deleted_rows: 0,
            }]
        );
    }
}
