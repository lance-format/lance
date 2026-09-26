// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::index::frag_reuse_reader::SegmentPlanParts;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance_core::Error;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::deepsize::DeepSizeOf;
use lance_index::frag_reuse::{
    CompactFragReuseIndex, CompactFragReuseIndexHandle, FRAG_REUSE_DETAILS_FILE_NAME,
    FRAG_REUSE_INDEX_NAME, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
};
use lance_index::scalar::{BatchRowIdRemapper, MetricsCollector, RowIdRemapper};
use lance_table::format::IndexMetadata;
use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
use prost::Message;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

/// The remapper resolved for one index open.
///
/// The FRI version picks the interface and the segment's need picks the
/// behavior: `V0` feeds the pre-existing synchronous consumers exactly as
/// before tagged histories existed; `V1Identity` carries no remapper at all
/// (the segment's rows are untouched, so the plugin's original load path
/// applies); `V1Translate` feeds the additive `*_with_remapping` entry points
/// that may await row-map reads.
#[derive(Clone)]
pub(crate) enum ResolvedRemapping {
    /// A v0 FRI mapping served by the compact in-memory handle.
    V0(Arc<dyn RowIdRemapper>),
    /// A tagged history under which this segment's rows are unchanged.
    V1Identity,
    /// A tagged-history mapping whose payload may need asynchronous reads.
    V1Translate {
        remapper: Arc<dyn BatchRowIdRemapper>,
        /// The segment's translation identity (see
        /// [`super::frag_reuse_reader::FragmentReuseIndex::translation_fingerprint`]).
        fingerprint: [u8; 32],
    },
}

impl std::fmt::Debug for ResolvedRemapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V0(_) => f.debug_tuple("V0").finish_non_exhaustive(),
            Self::V1Identity => f.debug_tuple("V1Identity").finish(),
            Self::V1Translate { remapper, .. } => {
                f.debug_tuple("V1Translate").field(remapper).finish()
            }
        }
    }
}

/// The FRI identity that belongs in an index's cache namespace, if any.
///
/// Only a v0 history is applied while the index is decoded (its remapper is
/// handed to the plugin's load path), so only then does the FRI entry's UUID
/// identify cached content. Under a tagged history the FRI UUID is
/// deliberately left out: every rewrite and trim mints a new one, even for
/// transitions a segment never touches, and the translated namespace from
/// [`scoped_index_cache`] already carries the segment's own translation
/// identity.
pub(crate) fn fri_cache_id(resolved: &Option<(Uuid, ResolvedRemapping)>) -> Option<&Uuid> {
    match resolved {
        Some((uuid, ResolvedRemapping::V0(_))) => Some(uuid),
        _ => None,
    }
}

/// Scope the dataset-level index cache for one resolved remapping.
///
/// This owns the single cache-scoping rule, which classifies cached objects
/// by what they depend on rather than by snapshot:
///
/// * Content decoded without translation (an identity segment under a
///   tagged history, a v0 history, no history) depends only on the index
///   file, so it lives in the plain per-index namespace and an append or an
///   unrelated rewrite never cold-starts it.
/// * Content that embeds translated addresses (pages, postings, partitions
///   loaded through a `V1Translate` remapper) depends on the segment's
///   translation state, so it lives under a namespace named by the segment's
///   translation fingerprint. The entry goes cold exactly when that state
///   changes: a transition on the segment's path trimmed or replaced, a
///   fragment on its path dropped from the manifest, a sibling taking direct
///   ownership of one of its destinations. An append, a row-level delete, or
///   a rewrite of fragments the segment never covered leaves the fingerprint,
///   and the entry, in place.
///
/// Soundness rests on the fingerprint covering every input of
/// `remap_row_ids_excluding` (see `translation_fingerprint`); cached
/// translated objects hold the `FragmentReuseIndex` they were filled with, so
/// a stale one must never be served under a changed state.
pub(crate) fn scoped_index_cache(
    dataset: &Dataset,
    resolved: &Option<(Uuid, ResolvedRemapping)>,
) -> crate::session::index_caches::DSIndexCache {
    crate::session::index_caches::DSIndexCache(match resolved {
        Some((_, ResolvedRemapping::V1Translate { fingerprint, .. })) => dataset
            .index_cache
            .with_key_prefix(&translated_namespace(fingerprint)),
        _ => dataset.index_cache.0.clone(),
    })
}

/// The cache namespace of translated content for one translation identity.
fn translated_namespace(fingerprint: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut prefix = String::with_capacity("fri-xlat/".len() + 2 * fingerprint.len());
    prefix.push_str("fri-xlat/");
    for byte in fingerprint {
        write!(prefix, "{byte:02x}").expect("writing to a String cannot fail");
    }
    prefix
}

/// The translation inputs one segment needs under a tagged history.
#[derive(Clone, Debug)]
pub(crate) enum SegmentRemappingPlan {
    /// The segment's stored coverage cannot intersect any rewritten path.
    Identity,
    /// The rewritten query coverage plus the fragments owned by other
    /// selected sibling segments of the same logical index, and the
    /// translation identity derived from them.
    Translate {
        coverage: RoaringBitmap,
        excluded_fragments: RoaringBitmap,
        fingerprint: [u8; 32],
    },
    /// Committed metadata exists but the filtered listing carries no query
    /// coverage for this segment (it was skipped or lost its bitmap).
    MissingCoverage,
}

/// Snapshot-level plan of every committed segment's translation inputs.
///
/// Which rows a segment owns is decided once per manifest snapshot, from one
/// pass over the same `load_indices` output every per-open resolution used to
/// re-scan. Openers only look their segment up by UUID.
#[derive(Clone, Debug)]
pub(crate) struct FriQueryPlan {
    pub(crate) segments: HashMap<Uuid, SegmentRemappingPlan>,
}

impl DeepSizeOf for FriQueryPlan {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.segments
            .values()
            .map(|segment| match segment {
                SegmentRemappingPlan::Translate {
                    coverage,
                    excluded_fragments,
                    fingerprint,
                } => {
                    coverage.serialized_size()
                        + excluded_fragments.serialized_size()
                        + fingerprint.len()
                }
                _ => 0,
            })
            .sum::<usize>()
            + self.segments.len() * std::mem::size_of::<(Uuid, SegmentRemappingPlan)>()
    }
}

#[derive(Clone)]
pub(crate) struct FriQueryPlanKey<'a> {
    pub(crate) fri_uuid: &'a Uuid,
}

impl CacheKey for FriQueryPlanKey<'_> {
    type ValueType = FriQueryPlan;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.fri_uuid.to_string().into()
    }

    fn type_name() -> &'static str {
        "FriQueryPlan"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-query-plan", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.fri_uuid.as_bytes());
    }
}

/// Build or fetch the snapshot's FRI query plan.
///
/// Cached in the tagged (manifest-path scoped) namespace; concurrent opens
/// coalesce on one build. Everything only needed to BUILD the plan (notably
/// `load_indices` and its tagged coverage post-processing) runs inside the
/// loader, so warm opens never recompute coverage.
async fn fri_query_plan(
    dataset: &Dataset,
    fri: &IndexMetadata,
    stored: &[IndexMetadata],
    mapping: &Arc<super::frag_reuse_reader::FragmentReuseIndex>,
) -> lance_core::Result<Arc<FriQueryPlan>> {
    dataset
        .index_cache
        .with_key_prefix(dataset.manifest_location.path.as_ref())
        .get_or_insert_with_key(
            FriQueryPlanKey {
                fri_uuid: &fri.uuid,
            },
            || async {
                // The filtered listing carries the rewritten query coverage;
                // `stored` keeps provenance from before that rewrite. Callers
                // hold metadata returned by load_indices and cannot supply
                // this distinction.
                let indices = dataset.load_indices().await?;
                let stored_by_uuid: HashMap<Uuid, &IndexMetadata> =
                    stored.iter().map(|entry| (entry.uuid, entry)).collect();
                // Group the filtered listing by logical index name; the
                // backtrack derives each member's sibling exclusions from the
                // stored provenance of the whole group in one pass, so
                // "direct coverage wins" is owned by one algorithm.
                let mut filtered_by_uuid: HashMap<Uuid, &IndexMetadata> =
                    HashMap::with_capacity(indices.len());
                let mut groups: HashMap<&str, Vec<Uuid>> = HashMap::new();
                for entry in indices.iter() {
                    filtered_by_uuid.insert(entry.uuid, entry);
                    if entry.name != FRAG_REUSE_INDEX_NAME {
                        groups
                            .entry(entry.name.as_str())
                            .or_default()
                            .push(entry.uuid);
                    }
                }
                let mut parts_by_uuid: HashMap<Uuid, SegmentPlanParts> = HashMap::new();
                for members in groups.into_values() {
                    let provenance: Vec<RoaringBitmap> = members
                        .iter()
                        .map(|uuid| {
                            stored_by_uuid
                                .get(uuid)
                                .and_then(|source| source.fragment_bitmap.clone())
                                .unwrap_or_default()
                        })
                        .collect();
                    for (uuid, parts) in members.iter().zip(mapping.segment_plans(&provenance)) {
                        parts_by_uuid.insert(*uuid, parts);
                    }
                }
                let mut segments = HashMap::with_capacity(stored.len());
                for source in stored.iter() {
                    let plan = if !mapping.may_need_translation(source.fragment_bitmap.as_ref()) {
                        SegmentRemappingPlan::Identity
                    } else if let Some(entry) = filtered_by_uuid.get(&source.uuid)
                        && let Some(bitmap) = &entry.fragment_bitmap
                    {
                        let coverage = bitmap & dataset.fragment_bitmap.as_ref();
                        // Other selected segments own their direct coverage.
                        // Drop paths entering those fragments before later
                        // mappings can merge them with this segment's
                        // contribution.
                        let parts = parts_by_uuid.remove(&source.uuid).unwrap_or_default();
                        let fingerprint = mapping.translation_fingerprint(
                            &coverage,
                            &parts.excluded,
                            &parts.path,
                        );
                        SegmentRemappingPlan::Translate {
                            coverage,
                            excluded_fragments: parts.excluded,
                            fingerprint,
                        }
                    } else {
                        SegmentRemappingPlan::MissingCoverage
                    };
                    segments.insert(source.uuid, plan);
                }
                Ok(FriQueryPlan { segments })
            },
        )
        .await
}

/// Resolve the FRI remapper shared by scalar and vector index loading.
pub(super) async fn open_row_id_remapping(
    dataset: &Dataset,
    index: &IndexMetadata,
    metrics: &dyn MetricsCollector,
) -> lance_core::Result<Option<(Uuid, ResolvedRemapping)>> {
    open_row_id_remapping_with_plan(dataset, index, None, metrics).await
}

/// [`open_row_id_remapping`] for a segment whose plan the caller supplies
/// (a staged segment planned by its merge); `None` looks the segment
/// up in the snapshot plan, which only knows committed segments.
pub(super) async fn open_row_id_remapping_with_plan(
    dataset: &Dataset,
    index: &IndexMetadata,
    staged: Option<&SegmentRemappingPlan>,
    metrics: &dyn MetricsCollector,
) -> lance_core::Result<Option<(Uuid, ResolvedRemapping)>> {
    // The cheap cached stored listing decides the generation; the filtered
    // listing (whose tagged post-processing recomputes coverage) is only
    // consulted inside the once-per-snapshot plan build.
    let stored = super::load_all_indices(dataset).await?;
    let Some(fri) = stored
        .iter()
        .find(|entry| entry.name == FRAG_REUSE_INDEX_NAME)
    else {
        return Ok(None);
    };
    if fri.index_version == 0 {
        return Ok(dataset.open_frag_reuse_index(metrics).await?.map(|legacy| {
            (
                legacy.uuid,
                ResolvedRemapping::V0(Arc::new(CompactFragReuseIndexHandle(legacy))),
            )
        }));
    }
    if fri.index_version != 1 {
        return Err(Error::not_supported(format!(
            "FRI index_version {} is unsupported. Please upgrade to a newer version",
            fri.index_version
        )));
    }
    // Everything below is v1-only code: legacy-only scopes must never get here.
    lance_index::scalar::check_batch_remapping_entry()?;
    let mapping = super::frag_reuse_reader::FragmentReuseIndex::open(dataset, fri).await?;
    let snapshot_plan;
    let plan = match staged {
        Some(plan) => plan,
        None => {
            snapshot_plan = fri_query_plan(dataset, fri, &stored, &mapping).await?;
            match snapshot_plan.segments.get(&index.uuid) {
                Some(plan) => plan,
                None => {
                    return Err(Error::not_supported(format!(
                        "FRI remapping requires committed segment metadata for {}; a staged \
                         segment must be opened with its own plan",
                        index.uuid
                    )));
                }
            }
        }
    };
    match plan {
        SegmentRemappingPlan::Identity => Ok(Some((fri.uuid, ResolvedRemapping::V1Identity))),
        SegmentRemappingPlan::MissingCoverage => Err(Error::not_supported(format!(
            "FRI query coverage is unavailable for segment {}",
            index.uuid
        ))),
        SegmentRemappingPlan::Translate {
            coverage,
            excluded_fragments,
            fingerprint,
        } => Ok(Some((
            fri.uuid,
            ResolvedRemapping::V1Translate {
                remapper: Arc::new(super::frag_reuse_remapping::QueryRowIdRemapper::new(
                    mapping,
                    coverage.clone(),
                    excluded_fragments.clone(),
                )),
                fingerprint: *fingerprint,
            },
        ))),
    }
}

/// Load fragment reuse index details from index metadata
pub async fn load_frag_reuse_index_details(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> lance_core::Result<Arc<FragReuseIndexDetails>> {
    if index.index_version != 0 {
        return Err(Error::not_supported(format!(
            "This operation requires interpreting FRI index_version {}; tagged FRI maintenance is not supported by this client. Upgrade to a client supporting this operation",
            index.index_version
        )));
    }
    let details_any = index.index_details.clone();
    if details_any.is_none()
        || !details_any
            .as_ref()
            .unwrap()
            .type_url
            .ends_with("FragmentReuseIndexDetails")
    {
        return Err(Error::index(
            "Index details is not for the fragment reuse index",
        ));
    }

    let proto = details_any.unwrap().to_msg::<FragmentReuseIndexDetails>()?;
    match &proto.content {
        None => Err(Error::index("Index details content is not found")),
        Some(Content::Inline(content)) => {
            Ok(Arc::new(FragReuseIndexDetails::try_from(content.clone())?))
        }
        Some(Content::External(external_file)) => {
            // the file content will be cached in the index cache later
            // so we do not put it to the file cache
            let data = read_fri_external_file(dataset, index, external_file).await?;

            let pb_sequence = InlineContent::decode(data)?;
            Ok(Arc::new(FragReuseIndexDetails::try_from(pb_sequence)?))
        }
    }
}

/// Resolve an FRI entry's external details bytes, honoring the entry's base:
/// a shallow-cloned entry's `details.binpb` lives in the SOURCE dataset, so
/// the path and store come from the entry's `base_id` (like every other
/// base-aware index file) instead of the current dataset root.
async fn read_fri_external_file(
    dataset: &Dataset,
    index: &IndexMetadata,
    file: &ExternalFile,
) -> lance_core::Result<bytes::Bytes> {
    let end = file
        .offset
        .checked_add(file.size)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::corrupt_file_named("FRI details", "external FRI range overflow"))?;
    let path = dataset
        .indice_files_dir(index)?
        .join(index.uuid.to_string())
        .join(file.path.as_str());
    dataset
        .object_store_for_index(index)
        .await?
        .open(&path)
        .await?
        .get_range(file.offset as usize..end)
        .await
        .map_err(Error::from)
}

/// open fragment reuse index based on its metadata details
pub(crate) async fn open_frag_reuse_index(
    uuid: Uuid,
    details: &FragReuseIndexDetails,
) -> lance_core::Result<CompactFragReuseIndex> {
    CompactFragReuseIndex::try_new(uuid, details.clone())
}

pub(crate) async fn build_new_frag_reuse_index(
    dataset: &mut Dataset,
    frag_reuse_groups: Vec<FragReuseGroup>,
    new_fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    let new_version = FragReuseVersion {
        dataset_version: dataset.manifest.version,
        groups: frag_reuse_groups,
    };

    let index_meta = dataset.load_indices().await.map(|indices| {
        indices
            .iter()
            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            .cloned()
    })?;

    let new_index_details = match &index_meta {
        None => FragReuseIndexDetails {
            versions: Vec::from([new_version]),
        },
        Some(index_meta) => {
            let current_details = load_frag_reuse_index_details(dataset, index_meta).await?;
            let mut versions = current_details.versions.clone();
            versions.push(new_version);
            FragReuseIndexDetails { versions }
        }
    };

    build_frag_reuse_index_metadata(
        dataset,
        index_meta.as_ref(),
        new_index_details,
        new_fragment_bitmap,
    )
    .await
}

pub(crate) async fn build_frag_reuse_index_metadata(
    dataset: &Dataset,
    index_meta: Option<&IndexMetadata>,
    new_index_details: FragReuseIndexDetails,
    new_fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    let index_id = uuid::Uuid::new_v4();
    let new_index_details_proto = InlineContent::from(&new_index_details);
    let proto = if new_index_details_proto.encoded_len() > 204800 {
        let file_path = dataset
            .indices_dir()
            .join(index_id.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME);
        let mut writer = dataset.object_store.create(&file_path).await?;
        writer
            .write_all(new_index_details_proto.encode_to_vec().as_slice())
            .await?;
        writer.shutdown().await?;
        let external_file = ExternalFile {
            path: FRAG_REUSE_DETAILS_FILE_NAME.to_owned(),
            offset: 0,
            size: new_index_details_proto.encoded_len() as u64,
        };
        FragmentReuseIndexDetails {
            content: Some(Content::External(external_file)),
        }
    } else {
        FragmentReuseIndexDetails {
            content: Some(Content::Inline(new_index_details_proto)),
        }
    };

    Ok(IndexMetadata {
        uuid: index_id,
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(new_fragment_bitmap),
        index_details: Some(Arc::new(prost_types::Any::from_msg(&proto)?)),
        index_version: index_meta.map_or(0, |index_meta| index_meta.index_version),
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        // Fragment reuse index is inline (no files)
        files: None,
    })
}
