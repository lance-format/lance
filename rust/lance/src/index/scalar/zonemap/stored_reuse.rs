// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stored zones of committed segments as an input of the ordinary zone map
//! build.
//!
//! Before the seed and scan pipeline starts, a build of a named index looks
//! for committed segments of the same index that already summarize some of
//! the requested fragments. A fragment offered by exactly one such segment,
//! whose stored zones are exactly the fragment's target zones, takes its
//! statistics whole from that segment; every other fragment goes on to its
//! write seed or a scan as before. Nothing here changes an address, invents
//! a count, or touches the commit.

use std::collections::{HashMap, HashSet};

use arrow_schema::DataType;
use lance_index::pbold;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::zonemap::{
    ReusedZones, StoredZones, ZONEMAP_INDEX_FILENAME, ZoneMapIndexBuilderParams,
    ZoneMapIndexVersion,
};
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_table::system_index::frag_reuse::metadata::uses_tagged_fri;
use roaring::RoaringBitmap;

use crate::index::loading::{load_all_indices_by_name, load_stored_indices};
use crate::index::provenance::has_overlay_newer_than;
use crate::index::segment::segment_has_zonemap_details;
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Largest stored size of a segment's zone map file that a build loads to
/// reuse its zones. A larger segment, or one whose size the manifest does
/// not record, is not opened; its fragments are summarized from seeds or a
/// scan. This bounds what is read, not the memory the decoded zones take.
pub(super) const STORED_ZONE_ADMISSION_BYTES: u64 = 256 * 1024 * 1024;

#[cfg(test)]
tokio::task_local! {
    /// Overrides [`STORED_ZONE_ADMISSION_BYTES`] for one test.
    pub static TEST_STORED_ZONE_ADMISSION_BYTES: u64;
}

fn admission_limit() -> u64 {
    #[cfg(test)]
    if let Ok(limit) = TEST_STORED_ZONE_ADMISSION_BYTES.try_with(|limit| *limit) {
        return limit;
    }
    STORED_ZONE_ADMISSION_BYTES
}

/// What the build took from committed segments.
pub(super) struct StoredZoneReuse {
    /// The accumulator the seed and scan pipeline continues.
    pub zones: ReusedZones,
    /// Fragments whose statistics came whole from a committed segment.
    pub fragments: HashSet<u64>,
    /// Segments loaded and used.
    pub sources_used: usize,
    /// Segments not opened because of their stored size.
    pub sources_skipped: usize,
    /// Segments whose version or configuration does not match the build, or
    /// whose file did not decode.
    pub sources_unusable: usize,
    /// Stored bytes of the segment files loaded.
    pub stored_bytes: u64,
    /// In-memory bytes those files decoded to.
    pub decoded_bytes: usize,
}

impl StoredZoneReuse {
    pub fn none(keep_null_positions: bool) -> Self {
        Self {
            zones: ReusedZones::new(keep_null_positions),
            fragments: HashSet::new(),
            sources_used: 0,
            sources_skipped: 0,
            sources_unusable: 0,
            stored_bytes: 0,
            decoded_bytes: 0,
        }
    }
}

/// The requested fragments each candidate offers. A segment offers fragment
/// `f` when `f` is live (it was requested at this version), in the
/// segment's coverage and in the bitmap the segment stores (a fragment that
/// is covered only because a reuse history projects old fragments onto a
/// compaction destination is not), and no overlay on the indexed field of
/// `f` is newer than the segment. A fragment offered by two or more
/// segments is offered by none.
fn offered_fragments(
    dataset: &Dataset,
    candidates: &[(&IndexMetadata, RoaringBitmap)],
    fragments: &[Fragment],
    field_id: i32,
) -> Vec<RoaringBitmap> {
    let indexed = HashSet::from([field_id]);
    let schema = dataset.schema();
    let mut offered = vec![RoaringBitmap::new(); candidates.len()];
    for fragment in fragments {
        let Ok(id) = u32::try_from(fragment.id) else {
            continue;
        };
        let mut offering = candidates
            .iter()
            .enumerate()
            .filter(|(_, (segment, coverage))| {
                coverage.contains(id)
                    && !has_overlay_newer_than(fragment, segment.dataset_version, &indexed, schema)
            })
            .map(|(position, _)| position);
        if let (Some(only), None) = (offering.next(), offering.next()) {
            offered[only].insert(id);
        }
    }
    offered
}

/// Why `segment` cannot feed this build, decided from its metadata alone
/// so that no file is opened for it.
fn metadata_mismatch(
    segment: &IndexMetadata,
    params: &ZoneMapIndexBuilderParams,
) -> Option<String> {
    if segment.index_version != ZoneMapIndexVersion::CURRENT as i32 {
        return Some(format!("index version {}", segment.index_version));
    }
    let details = segment
        .index_details
        .as_ref()
        .and_then(|details| details.to_msg::<pbold::ZoneMapIndexDetails>().ok())?;
    if let Some(rows_per_zone) = details.rows_per_zone
        && rows_per_zone != params.rows_per_zone()
    {
        return Some(format!("rows_per_zone {rows_per_zone}"));
    }
    if params.store_null_bitmap() && details.has_null_bitmap == Some(false) {
        return Some("no stored null positions".to_string());
    }
    None
}

/// The stored size of `segment`'s zone map file, when the manifest records it.
fn stored_size(segment: &IndexMetadata) -> Option<u64> {
    segment
        .files
        .as_ref()?
        .iter()
        .find(|file| file.path == ZONEMAP_INDEX_FILENAME)
        .map(|file| file.size_bytes)
        .filter(|size| *size > 0)
}

/// Take the statistics of as many of `fragments` as committed segments of
/// `index_name` can supply whole. Returns nothing reused when there is no
/// name, on tables with stable row ids, and on tables whose fragment reuse
/// history is tagged. Segments are loaded one at a time and each is
/// released before the next is read.
pub(super) async fn reuse_stored_zones(
    dataset: &Dataset,
    index_name: Option<&str>,
    column: &str,
    data_type: &DataType,
    params: &ZoneMapIndexBuilderParams,
    fragments: &[Fragment],
) -> Result<StoredZoneReuse> {
    let keep_null_positions = params.store_null_bitmap();
    let mut reuse = StoredZoneReuse::none(keep_null_positions);
    let Some(index_name) = index_name else {
        return Ok(reuse);
    };
    if dataset.manifest.uses_stable_row_ids() || fragments.is_empty() {
        return Ok(reuse);
    }
    let stored = load_stored_indices(dataset).await?;
    let fri = stored
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME);
    if uses_tagged_fri(&dataset.manifest, fri) {
        return Ok(reuse);
    }
    let Some(field_id) = dataset.schema().field(column).map(|field| field.id) else {
        return Ok(reuse);
    };
    let stored_bitmaps: HashMap<_, _> = stored
        .iter()
        .filter_map(|index| Some((index.uuid, index.fragment_bitmap.as_ref()?)))
        .collect();
    let live = dataset.fragment_bitmap.as_ref();
    let segments = load_all_indices_by_name(dataset, index_name).await?;
    let candidates: Vec<(&IndexMetadata, RoaringBitmap)> = segments
        .iter()
        .filter(|segment| {
            segment.keyed_field() == Some(field_id) && segment_has_zonemap_details(segment)
        })
        .filter_map(|segment| {
            let coverage = segment.effective_fragment_bitmap(live)?;
            let stored = stored_bitmaps.get(&segment.uuid)?;
            Some((segment, coverage & *stored))
        })
        .collect();
    if candidates.is_empty() {
        return Ok(reuse);
    }
    let offered = offered_fragments(dataset, &candidates, fragments, field_id);
    let by_id: HashMap<u64, &Fragment> = fragments
        .iter()
        .map(|fragment| (fragment.id, fragment))
        .collect();
    let limit = admission_limit();
    for ((segment, _), offered) in candidates.iter().zip(offered) {
        if offered.is_empty() {
            continue;
        }
        if let Some(reason) = metadata_mismatch(segment, params) {
            log::debug!(
                "zone map segment {} cannot supply stored zones: {reason}",
                segment.uuid
            );
            reuse.sources_unusable += 1;
            continue;
        }
        let Some(size) = stored_size(segment).filter(|size| *size <= limit) else {
            log::debug!(
                "zone map segment {} is not loaded for reuse: its stored size is unknown or \
                 above {limit} bytes",
                segment.uuid
            );
            reuse.sources_skipped += 1;
            continue;
        };
        let version = ZoneMapIndexVersion::from_metadata(segment.index_version as u32)?;
        let store = LanceIndexStore::from_dataset_for_existing(dataset, segment).await?;
        // A file that does not decode as this build's zones is not a source;
        // a storage failure is the build's failure.
        let source = match StoredZones::load(&store, version, &offered).await {
            Ok(source) => source,
            Err(error @ (Error::IO { .. } | Error::NotFound { .. })) => return Err(error),
            Err(error) => {
                log::warn!(
                    "zone map segment {} cannot supply stored zones: its file did not decode: \
                     {error}",
                    segment.uuid
                );
                reuse.sources_unusable += 1;
                continue;
            }
        };
        reuse.stored_bytes += size;
        reuse.decoded_bytes += source.decoded_bytes();
        if let Some(reason) =
            source.incompatibility(data_type, params.rows_per_zone(), keep_null_positions)
        {
            log::debug!(
                "zone map segment {} cannot supply stored zones: {reason}",
                segment.uuid
            );
            reuse.sources_unusable += 1;
            continue;
        }
        reuse.sources_used += 1;
        for id in &offered {
            let fragment = by_id[&u64::from(id)];
            let physical_rows = super::physical_rows(dataset, fragment).await?;
            if reuse
                .zones
                .reuse_stored(&source, fragment.id, physical_rows)?
            {
                reuse.fragments.insert(fragment.id);
            }
        }
        drop(source);
    }
    Ok(reuse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(version: u64) -> IndexMetadata {
        IndexMetadata {
            uuid: uuid::Uuid::new_v4(),
            name: "x".to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version: version,
            fragment_bitmap: None,
            index_details: None,
            index_version: ZoneMapIndexVersion::CURRENT as i32,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn a_fragment_offered_twice_is_offered_by_none() {
        use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
        use arrow_array::types::Int32Type;
        let dataset = lance_datagen::gen_batch()
            .col("x", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(4), FragmentRowCount::from(4))
            .await
            .unwrap();
        let field_id = dataset.schema().field("x").unwrap().id;
        let (a, b) = (segment(1), segment(1));
        let candidates = vec![
            (&a, RoaringBitmap::from_iter([0u32, 1, 2])),
            (&b, RoaringBitmap::from_iter([2u32, 3])),
        ];
        let fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
        let offered = offered_fragments(&dataset, &candidates, &fragments, field_id);
        assert_eq!(offered[0], RoaringBitmap::from_iter([0u32, 1]));
        assert_eq!(offered[1], RoaringBitmap::from_iter([3u32]));
        // A fragment that is not requested is offered by nobody.
        let offered = offered_fragments(&dataset, &candidates, &fragments[..1], field_id);
        assert_eq!(offered[0], RoaringBitmap::from_iter([0u32]));
        assert!(offered[1].is_empty());
    }

    #[test]
    fn admission_reads_the_zone_map_file_size() {
        use lance_table::format::IndexFile;
        let mut source = segment(1);
        assert_eq!(stored_size(&source), None);
        source.files = Some(vec![IndexFile {
            path: "other.lance".to_string(),
            size_bytes: 10,
            file_metadata_size_bytes: None,
        }]);
        assert_eq!(stored_size(&source), None);
        source.files.as_mut().unwrap().push(IndexFile {
            path: ZONEMAP_INDEX_FILENAME.to_string(),
            size_bytes: 0,
            file_metadata_size_bytes: None,
        });
        assert_eq!(stored_size(&source), None);
        source.files.as_mut().unwrap()[1].size_bytes = 42;
        assert_eq!(stored_size(&source), Some(42));
    }
}
