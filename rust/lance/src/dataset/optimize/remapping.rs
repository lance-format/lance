// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for remapping row ids. Necessary before stable row ids.
//!

use crate::Result;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexExt;
use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
use crate::{Dataset, index};
use async_trait::async_trait;
use lance_core::Error;
use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_index::frag_reuse::{FRAG_REUSE_INDEX_NAME, FragDigest};
use lance_table::format::{Fragment, IndexFile, IndexMetadata};
use lance_table::io::manifest::read_manifest_indexes;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// The result of remapping an index
#[derive(Debug, Clone, PartialEq)]
pub enum RemapResult {
    // Index could not be remapped, drop it
    Drop,
    // No remapping is needed, keep the index as-is
    Keep(Uuid),
    // Index was remapped, return the new index
    Remapped(RemappedIndex),
}

/// A remapped index
#[derive(Debug, Clone, PartialEq)]
pub struct RemappedIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
    pub index_details: prost_types::Any,
    pub index_version: u32,
    /// List of files in the index with their sizes.
    pub files: Option<Vec<IndexFile>>,
}

/// When compaction runs the row ids will change.  This typically means that
/// indices will need to be remapped.  The details of how this happens are not
/// a part of the compaction process and so a trait is defined here to allow
/// for inversion of control.
#[async_trait]
pub trait IndexRemapper: Send + Sync {
    async fn remap_indices(
        &self,
        index_map: RowAddrRemap,
        affected_fragment_ids: &[u64],
    ) -> Result<Vec<RemappedIndex>>;
}

/// Options for creating an [IndexRemapper]
///
/// Currently we don't have any options but we may need options in the future and so we
/// want to keep a placeholder
#[async_trait]
pub trait IndexRemapperOptions: Send + Sync {
    /// Creates a remapper when the dataset has indices that need row address remapping.
    ///
    /// Returns `None` when no remappable indices exist, allowing compaction to avoid
    /// materializing an unused row address map.
    async fn create_remapper(&self, dataset: &Dataset) -> Result<Option<Box<dyn IndexRemapper>>>;
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct IgnoreRemap {}

#[async_trait]
impl IndexRemapper for IgnoreRemap {
    async fn remap_indices(&self, _: RowAddrRemap, _: &[u64]) -> Result<Vec<RemappedIndex>> {
        Ok(Vec::new())
    }
}

#[async_trait]
impl IndexRemapperOptions for IgnoreRemap {
    async fn create_remapper(&self, _: &Dataset) -> Result<Option<Box<dyn IndexRemapper>>> {
        Ok(None)
    }
}

/// Iterator that yields row_addrs that are in the given fragments but not in
/// the given row_addrs iterator.
struct MissingAddrs<'a, I: Iterator<Item = u64>> {
    row_addrs: I,
    expected_row_addr: u64,
    current_fragment_idx: usize,
    last: Option<u64>,
    fragments: &'a Vec<FragDigest>,
}

impl<'a, I: Iterator<Item = u64>> MissingAddrs<'a, I> {
    /// row_addrs must be sorted in the same order in which the rows would be
    /// found by scanning fragments in the order they are presented in.
    /// fragments is not guaranteed to be sorted by id.
    fn new(row_addrs: I, fragments: &'a Vec<FragDigest>) -> Self {
        assert!(!fragments.is_empty());
        let first_frag = &fragments[0];
        Self {
            row_addrs,
            expected_row_addr: first_frag.id * RowAddress::FRAGMENT_SIZE,
            current_fragment_idx: 0,
            last: None,
            fragments,
        }
    }
}

impl<I: Iterator<Item = u64>> Iterator for MissingAddrs<'_, I> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_fragment_idx >= self.fragments.len() {
                return None;
            }
            let val = if let Some(last) = self.last {
                self.last = None;
                last
            } else {
                // If we've exhausted row_addrs but we aren't done then use 0 which
                // is guaranteed to not match because that would mean that row_addrs
                // was empty and we check for that earlier.
                self.row_addrs.next().unwrap_or(0)
            };

            let current_fragment = &self.fragments[self.current_fragment_idx];
            let frag = val / RowAddress::FRAGMENT_SIZE;
            let expected_row_addr = self.expected_row_addr;
            self.expected_row_addr += 1;

            let current_physical_rows = current_fragment.physical_rows;
            if (self.expected_row_addr % RowAddress::FRAGMENT_SIZE) == current_physical_rows as u64
            {
                self.current_fragment_idx += 1;
                if self.current_fragment_idx < self.fragments.len() {
                    self.expected_row_addr =
                        self.fragments[self.current_fragment_idx].id * RowAddress::FRAGMENT_SIZE;
                }
            }
            if frag != current_fragment.id {
                self.last = Some(val);
                return Some(expected_row_addr);
            }
            if val != expected_row_addr {
                self.last = Some(val);
                return Some(expected_row_addr);
            }
        }
    }
}

pub fn transpose_row_addrs(
    row_addrs: RoaringTreemap,
    old_fragments: &[Fragment],
    new_fragments: &[Fragment],
) -> HashMap<u64, Option<u64>> {
    let old_frag_digests: Vec<FragDigest> = old_fragments.iter().map(|frag| frag.into()).collect();
    let new_frag_digests: Vec<FragDigest> = new_fragments.iter().map(|frag| frag.into()).collect();
    transpose_row_ids_from_digest(row_addrs, &old_frag_digests, &new_frag_digests)
}

pub fn transpose_row_ids_from_digest(
    row_addrs: RoaringTreemap,
    old_fragments: &Vec<FragDigest>,
    new_fragments: &[FragDigest],
) -> HashMap<u64, Option<u64>> {
    let new_addrs = new_fragments.iter().flat_map(|frag| {
        (0..frag.physical_rows as u32).map(|offset| {
            Some(u64::from(RowAddress::new_from_parts(
                frag.id as u32,
                offset,
            )))
        })
    });
    // The hashmap will have an entry for each row addr to map plus all rows that
    // were deleted.
    let expected_size = row_addrs.len() as usize
        + old_fragments
            .iter()
            .map(|frag| frag.num_deleted_rows)
            .sum::<usize>();
    // We expect row addrs to be unique, so we should already not get many collisions.
    // The default hasher is designed to be resistance to DoS attacks, which is
    // more than we need for this use case.
    let mut mapping: HashMap<u64, Option<u64>> = HashMap::with_capacity(expected_size);
    mapping.extend(row_addrs.iter().zip(new_addrs));
    MissingAddrs::new(row_addrs.into_iter(), old_fragments).for_each(|addr| {
        mapping.insert(addr, None);
    });
    mapping
}

/// Remap a given index using the fragment reuse index if possible.
/// If the frag reuse index does not exist, the operation fails with [Error::NotSupported]
/// If the frag reuse index exists but is empty, the operation succeeds without a commit.
async fn remap_index(dataset: &mut Dataset, index_id: &Uuid) -> Result<()> {
    let indices = dataset.load_indices().await?;
    let frag_reuse_index_meta = match indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME) {
        None => Err(Error::not_supported_source(
            "Fragment reuse index not found, cannot remap an index post compaction".into(),
        )),
        Some(frag_reuse_index_meta) => Ok(frag_reuse_index_meta),
    }?;

    let frag_reuse_details = load_frag_reuse_index_details(dataset, frag_reuse_index_meta).await?;
    let frag_reuse_index =
        open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref()).await?;

    if frag_reuse_index.is_empty() {
        return Ok(());
    }

    // Read the index's on-disk metadata once. Its stored row addresses are at
    // this baseline; we compose all reuse versions into a single remap so the
    // index file is rebuilt and committed exactly once, rather than once per
    // version (the reuse index can accumulate many versions before remap runs).
    let curr_index_meta = read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await?
    .into_iter()
    .find(|idx| idx.uuid == *index_id)
    .ok_or_else(|| {
        Error::index(format!(
            "index {index_id} not found in manifest; it may have been concurrently dropped"
        ))
    })?;

    // Compose the coverage (fragment bitmap) remap across every reuse version in
    // one pass. Chaining is automatic: a version inserts its new fragments,
    // which a later version then sees as its old fragments. `data_predates_version`
    // is evaluated against the fixed baseline (there are no intermediate
    // commits), and the new-fragment branch handles a bitmap that was already
    // coverage-remapped + persisted before the data was remapped (e.g. while
    // remapping a *sibling* index).
    let baseline_version = curr_index_meta.dataset_version;
    let has_unknown_coverage = curr_index_meta.fragment_bitmap.is_none();
    let (should_remap, mut bitmap_after_remap) = match curr_index_meta.fragment_bitmap.clone() {
        Some(mut index_frag_bitmap) => {
            let mut should_remap = false;
            for version in frag_reuse_index.details.versions.iter() {
                let data_predates_version = baseline_version < version.dataset_version;
                for group in version.groups.iter() {
                    let mut old_frag_in_index = 0;
                    for old_frag in group.old_frags.iter() {
                        if index_frag_bitmap.remove(old_frag.id as u32) {
                            old_frag_in_index += 1;
                        }
                    }

                    if old_frag_in_index > 0 {
                        if old_frag_in_index != group.old_frags.len() {
                            // this should never happen because we always commit a full rewrite group
                            // and we always reindex either the entire group or nothing.
                            // We use invalid input to be consistent with
                            // dataset::transaction::recalculate_fragment_bitmap
                            return Err(Error::invalid_input(format!(
                                "The compaction plan included a rewrite group that was a split of indexed and non-indexed data: {:?}",
                                group.old_frags
                            )));
                        }
                        index_frag_bitmap.extend(group.new_frags.iter().map(|f| f.id as u32));
                        should_remap = true;
                    } else if data_predates_version
                        && group
                            .new_frags
                            .iter()
                            .any(|new_frag| index_frag_bitmap.contains(new_frag.id as u32))
                    {
                        // The bitmap was already coverage-remapped onto this
                        // group's new fragments and persisted before the data was
                        // remapped, so the old fragments are gone from the bitmap
                        // but the index data still needs remapping.
                        should_remap = true;
                    }
                }
            }
            (should_remap, Some(index_frag_bitmap))
        }
        None => (true, None),
    };

    if !should_remap {
        return Ok(());
    }

    // Apply the compact version chain directly while rebuilding the index. The
    // remapper passes intermediate moved addresses into later FRI versions and
    // leaves missing mappings unchanged, so no composed per-row map is needed.
    // This also handles the sibling-coverage-remap case: remapping is driven by
    // the row addresses stored in the index, not by its already-advanced bitmap.
    let remap_result =
        index::remap_index(dataset, index_id, frag_reuse_index.row_addr_remap()).await?;

    // Remapping advances the index watermark for fragment-reuse cleanup, but it
    // does not incorporate overlays committed after the source index was built.
    // Exclude those fragments so queries scan their current values instead.
    if let Some(fragment_bitmap) = &mut bitmap_after_remap {
        // Fields whose value this index materializes: its indexed key fields plus every
        // covered (included) column expanded to its leaf subtree (an overlay lists leaf
        // ids, while `covering_fields` records a parent struct id).
        let relevant_field_ids =
            Transaction::index_dependent_leaf_ids(&curr_index_meta, dataset.schema());
        for fragment in dataset.manifest.fragments.iter() {
            // An overlay refreshing an indexed or covered field makes the index storage
            // stale for that fragment: it still holds the pre-overlay copy of the value.
            let has_newer_relevant_overlay = fragment.overlays.iter().any(|overlay| {
                overlay.committed_version > curr_index_meta.dataset_version
                    && overlay
                        .data_file
                        .fields
                        .iter()
                        .any(|field_id| relevant_field_ids.contains(field_id))
            });
            if has_newer_relevant_overlay {
                fragment_bitmap.remove(fragment.id as u32);
            }
        }
    }
    let new_dataset_version = if has_unknown_coverage {
        curr_index_meta.dataset_version
    } else {
        dataset.manifest.version
    };

    let new_index_meta = match remap_result {
        // Nothing to commit: either the composed remap emptied the index (every
        // row deleted), matching the prior per-version behavior, or
        // `index::remap_index` withdrew a covered index it cannot carry payload
        // through. Either way the existing entry is left untouched.
        //
        // The withdrawal case is unreachable here today: the only caller is
        // `remap_column_index`, which refuses a covered index first. Compaction
        // reaches that withdrawal through `DatasetIndexRemapper`, which handles
        // `RemapResult::Drop` in `dataset/index.rs` rather than through here.
        RemapResult::Drop => return Ok(()),
        RemapResult::Keep(new_id) => IndexMetadata {
            uuid: new_id,
            name: curr_index_meta.name.clone(),
            fields: curr_index_meta.fields.clone(),
            covering_fields: curr_index_meta.covering_fields.clone(),
            dataset_version: new_dataset_version,
            fragment_bitmap: bitmap_after_remap,
            index_details: curr_index_meta.index_details.clone(),
            index_version: curr_index_meta.index_version,
            created_at: curr_index_meta.created_at,
            base_id: None,
            files: curr_index_meta.files.clone(),
        },
        RemapResult::Remapped(remapped_index) => IndexMetadata {
            uuid: remapped_index.new_id,
            name: curr_index_meta.name.clone(),
            fields: curr_index_meta.fields.clone(),
            covering_fields: curr_index_meta.covering_fields.clone(),
            dataset_version: new_dataset_version,
            fragment_bitmap: bitmap_after_remap,
            index_details: Some(Arc::new(remapped_index.index_details)),
            index_version: remapped_index.index_version as i32,
            created_at: curr_index_meta.created_at,
            base_id: None,
            files: remapped_index.files,
        },
    };

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![new_index_meta],
            removed_indices: vec![curr_index_meta],
        },
        None,
    );

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(())
}

pub async fn remap_column_index(
    dataset: &mut Dataset,
    columns: &[&str],
    name: Option<String>,
) -> Result<()> {
    if columns.len() != 1 {
        return Err(Error::index(
            "Only support remapping index on 1 column at the moment".to_string(),
        ));
    }

    let column = columns[0];
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::index(format!(
            "RemapIndex: column '{column}' does not exist"
        )));
    };

    let indices = dataset.load_indices().await?;
    let index_name = name.unwrap_or(format!("{column}_idx"));
    let index = match indices.iter().find(|i| i.name == index_name) {
        None => {
            return Err(Error::index(format!(
                "Index with name {} not found",
                index_name
            )));
        }
        Some(index) => {
            // The real question is "does this index belong to this column",
            // i.e. its one keyed field is `field.id`. Carried fields are
            // irrelevant here, same as in `index::remap_index`.
            if index.keyed_field() != Some(field.id) {
                Err(Error::index(format!(
                    "Index name {} already exists with fields {:?} (carried fields {:?}); \
                     expected a single keyed field {}",
                    index_name, index.fields, index.covering_fields, field.id
                )))
            } else if !index.covering_fields.is_empty() {
                // Same rule as `optimize_indices`, and for the same reason: no
                // index type carries the declared payload through a remap, so the
                // result would still claim values its storage does not hold. The
                // caller named this index, so refuse out loud -- compaction
                // withdraws instead only because it must not block a table-level
                // operation over one index it cannot remap.
                Err(Error::index(format!(
                    "Remapping index '{}' is not supported: it declares covering \
                     fields {:?}, which no index builder writes or preserves yet",
                    index_name, index.covering_fields,
                )))
            } else {
                Ok(index)
            }
        }
    }?;

    remap_index(dataset, &index.uuid).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_matches_transpose() {
        use lance_core::utils::row_addr_remap::GroupInputWithLayout;
        // Ascending old fragments (compaction's scan order), with deletions.
        let old = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 2,
            },
            FragDigest {
                id: 1,
                physical_rows: 4,
                num_deleted_rows: 1,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];
        // 9 rewritten rows (offsets that survived in each old fragment).
        let rewritten = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 3),
            (3, 0),
            (3, 1),
            (3, 2),
        ];
        let addrs = RoaringTreemap::from_iter(
            rewritten
                .iter()
                .map(|(f, o)| u64::from(RowAddress::new_from_parts(*f, *o))),
        );
        // 9 rewritten rows split across two new fragments.
        let new = vec![
            FragDigest {
                id: 10,
                physical_rows: 4,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 11,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
        ];

        let expected = transpose_row_ids_from_digest(addrs.clone(), &old, &new);
        let compact = RowAddrRemap::compact_with_layout([GroupInputWithLayout {
            rewritten_old_row_addrs: addrs,
            old_frags: old
                .iter()
                .map(|f| (f.id as u32, f.physical_rows as u32))
                .collect(),
            new_frags: new
                .iter()
                .map(|f| (f.id as u32, f.physical_rows as u32))
                .collect(),
        }])
        .unwrap();

        // Every real address in the old fragments must map identically.
        for f in &old {
            for o in 0..f.physical_rows as u32 {
                let a = u64::from(RowAddress::new_from_parts(f.id as u32, o));
                assert_eq!(
                    compact.get(a),
                    expected.get(&a).copied(),
                    "mismatch at ({}, {})",
                    f.id,
                    o
                );
            }
        }
        // A fragment outside the group is unaffected by both.
        let outside = u64::from(RowAddress::new_from_parts(99, 0));
        assert_eq!(compact.get(outside), expected.get(&outside).copied());
    }

    #[test]
    fn test_missing_indices() {
        // Sanity test to make sure MissingIds works.  Does not test actual functionality so
        // feel free to remove if it becomes inconvenient
        let frags = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];
        let rows = [(0, 1), (0, 3), (0, 4), (3, 0), (3, 2)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());

        let missing = MissingAddrs::new(rows, &frags).collect::<Vec<_>>();
        let expected_missing = [(0, 0), (0, 2), (3, 1)]
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(missing, expected_missing);
    }

    #[test]
    fn test_missing_ids() {
        // test with missing first row
        // test with missing last row
        // test fragment ids out of order

        let fragments = vec![
            FragDigest {
                id: 0,
                physical_rows: 5,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 3,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
            FragDigest {
                id: 1,
                physical_rows: 3,
                num_deleted_rows: 0,
            },
        ];

        // Written as pairs of (fragment_id, offset)
        let row_addrs = vec![
            (0, 1),
            (0, 3),
            (0, 4),
            (3, 0),
            (3, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ];
        let row_addrs = row_addrs
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into());
        let result = MissingAddrs::new(row_addrs, &fragments).collect::<Vec<_>>();

        let expected = vec![(0, 0), (0, 2), (3, 1)];
        let expected = expected
            .into_iter()
            .map(|(frag, offset)| RowAddress::new_from_parts(frag, offset).into())
            .collect::<Vec<u64>>();
        assert_eq!(result, expected);
    }

    /// A *physical* remap through the real production entry point,
    /// `remap_column_index`, must refuse a covered index rather than quietly do
    /// nothing. No index type carries the declared payload through a remap, and
    /// the caller named this index, so a silent no-op would hand back an index
    /// that covers nothing with no indication why.
    ///
    /// Asserting on committed metadata here would prove nothing: the flow is
    /// `remap_column_index` -> the private `remap_index` -> `index::remap_index`,
    /// which withdraws a covered index before the fully-deleted `Keep` check, and
    /// the `Drop` arm returns without committing. The `Keep`/`Remapped` arms are
    /// therefore unreachable for a covered index, so a "declaration preserved"
    /// assertion would pass even if those arms stopped preserving it. Assert the
    /// refusal instead.
    #[tokio::test]
    async fn test_remap_column_index_refuses_a_covered_index() {
        use crate::dataset::index::DatasetIndexRemapperOptions;
        use crate::dataset::optimize::{
            CompactionOptions, commit_compaction, plan_compaction, rewrite_files,
        };
        use crate::utils::test::covering;
        use lance_core::utils::tempfile::TempStrDir;
        use std::borrow::Cow;

        let test_uri = TempStrDir::default();
        // Two fragments, so compaction has something to merge.
        let mut dataset = covering::write_vector_payload_dataset(&test_uri).await;
        covering::append_vector_payload_rows(&mut dataset, covering::ROWS_PER_FRAGMENT).await;
        covering::create_ivf_pq_index(&mut dataset, "vec").await;

        let (_, id_field_id) = covering::declare_covering(&mut dataset, "vec", "payload").await;
        let index_name = dataset.load_indices().await.unwrap()[0].name.clone();

        // Delete some (not all) rows so the remap has real work to do and does
        // not take the all-fragments-deleted `RemapResult::Keep` shortcut.
        dataset.delete("payload < 100").await.unwrap();

        let options = CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert!(
            !plan.tasks().is_empty(),
            "compaction plan must have work to do, or this test proves nothing"
        );
        for task in plan.tasks().iter() {
            let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
                .await
                .unwrap();
            commit_compaction(
                &mut dataset,
                Vec::from([rewrite_result]),
                Arc::new(DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
            .unwrap();
        }

        let before = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == index_name)
            .cloned()
            .expect("precondition: the covered index exists");

        // `remap_column_index` is user-directed -- the caller named this index --
        // so it must refuse rather than no-op. Refusing is also what makes this
        // test meaningful: the `Keep`/`Remapped` arms below are unreachable for a
        // covered index, so asserting on the committed metadata instead would
        // pass whether or not those arms preserved `covering_fields`.
        let error = remap_column_index(&mut dataset, &["vec"], Some(index_name.clone()))
            .await
            .expect_err("remapping a covered index must be refused");
        assert!(
            error.to_string().contains("declares covering fields"),
            "unexpected message: {error}"
        );

        // Refused, not half-applied.
        let after = dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .find(|idx| idx.name == index_name)
            .cloned()
            .expect("a refused remap must leave the index in place");
        assert_eq!(
            after.uuid, before.uuid,
            "a refused remap replaced the index"
        );
        assert_eq!(after.covering_fields, vec![id_field_id]);
        assert_eq!(after.fragment_bitmap, before.fragment_bitmap);
    }
}
