// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
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
use lance_table::transaction::{FragmentReuseRewrite, RewriteGroup};
use prost::Message;
use roaring::RoaringBitmap;
use std::collections::{HashMap, HashSet};
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
    V1Translate(Arc<dyn BatchRowIdRemapper>),
}

impl std::fmt::Debug for ResolvedRemapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V0(_) => f.debug_tuple("V0").finish_non_exhaustive(),
            Self::V1Identity => f.debug_tuple("V1Identity").finish(),
            Self::V1Translate(remapper) => f.debug_tuple("V1Translate").field(remapper).finish(),
        }
    }
}

/// Scope the dataset-level index cache for one resolved remapping.
///
/// This owns the single cache-scoping rule: a tagged history (FRI
/// index_version != 0) rewrites what each segment covers per manifest
/// snapshot, so its entries are keyed under the manifest path; v0 and
/// FRI-less datasets keep the pre-existing keys.
pub(crate) fn scoped_index_cache(
    dataset: &Dataset,
    resolved: &Option<(Uuid, ResolvedRemapping)>,
) -> crate::session::index_caches::DSIndexCache {
    crate::session::index_caches::DSIndexCache(match resolved {
        Some((_, ResolvedRemapping::V1Identity | ResolvedRemapping::V1Translate(_))) => dataset
            .index_cache
            .with_key_prefix(dataset.manifest_location.path.as_ref()),
        _ => dataset.index_cache.0.clone(),
    })
}

/// The translation inputs one segment needs under a tagged history.
#[derive(Clone, Debug)]
pub(crate) enum SegmentRemappingPlan {
    /// The segment's stored coverage cannot intersect any rewritten path.
    Identity,
    /// The rewritten query coverage plus the fragments owned by other
    /// selected sibling segments of the same logical index.
    Translate {
        coverage: RoaringBitmap,
        excluded_fragments: RoaringBitmap,
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
                } => coverage.serialized_size() + excluded_fragments.serialized_size(),
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
                let mut excluded_by_uuid: HashMap<Uuid, RoaringBitmap> = HashMap::new();
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
                        excluded_by_uuid.insert(*uuid, parts.excluded);
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
                        let excluded_fragments =
                            excluded_by_uuid.remove(&source.uuid).unwrap_or_default();
                        SegmentRemappingPlan::Translate {
                            coverage,
                            excluded_fragments,
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
    let plan = fri_query_plan(dataset, fri, &stored, &mapping).await?;
    match plan.segments.get(&index.uuid) {
        None => Err(Error::not_supported(format!(
            "FRI remapping requires committed segment metadata for {}",
            index.uuid
        ))),
        Some(SegmentRemappingPlan::Identity) => Ok(Some((fri.uuid, ResolvedRemapping::V1Identity))),
        Some(SegmentRemappingPlan::MissingCoverage) => Err(Error::not_supported(format!(
            "FRI query coverage is unavailable for segment {}",
            index.uuid
        ))),
        Some(SegmentRemappingPlan::Translate {
            coverage,
            excluded_fragments,
        }) => Ok(Some((
            fri.uuid,
            ResolvedRemapping::V1Translate(Arc::new(
                super::frag_reuse_remapping::QueryRowIdRemapper::new(
                    mapping,
                    coverage.clone(),
                    excluded_fragments.clone(),
                ),
            )),
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
            let data = read_fri_external_file(dataset, index, external_file.clone()).await?;

            let pb_sequence = InlineContent::decode(data)?;
            Ok(Arc::new(FragReuseIndexDetails::try_from(pb_sequence)?))
        }
    }
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

/// One length-delimited protobuf field, the unit both the inline details
/// payload and appended transitions are spliced with.
fn encode_length_delimited_field(tag: u32, bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len() + 8);
    prost::encoding::encode_key(tag, prost::encoding::WireType::LengthDelimited, &mut output);
    prost::encoding::encode_varint(bytes.len() as u64, &mut output);
    output.extend_from_slice(bytes);
    output
}

/// Resolve an FRI entry's external details bytes, honoring the entry's base:
/// a shallow-cloned entry's `details.binpb` (like its row maps) lives in the
/// SOURCE dataset, so the path and store come from the entry's `base_id`,
/// exactly as the reader resolves them. This is the writer stack's single
/// external-resolution point (`FragReuseLedger::decode`'s `read_external`
/// callback shape); full unification with the reader-side loader in
/// `frag_reuse_reader` happens when the stack flattens.
async fn read_fri_external_file(
    dataset: &Dataset,
    index: &IndexMetadata,
    file: ExternalFile,
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

/// Decode a committed FRI entry into its transition ledger, resolving
/// external content through [`read_fri_external_file`]. The entry's original
/// `Any` is decoded directly, so the envelope is parsed exactly once, by the
/// ledger. Works for v0 entries too: legacy versions decode as lifted
/// transitions.
pub(crate) async fn decode_frag_reuse_ledger(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> lance_core::Result<lance_table::system_index::frag_reuse::ledger::FragReuseLedger> {
    let details = entry
        .index_details
        .as_ref()
        .ok_or_else(|| Error::index("Index details is not for the fragment reuse index"))?;
    lance_table::system_index::frag_reuse::ledger::FragReuseLedger::decode(
        entry.index_version,
        details,
        |file| read_fri_external_file(dataset, entry, file),
    )
    .await
}

/// The row-map ids of the stable-partition transitions in the dataset's
/// committed FRI entry (empty when the entry is absent or v0). The conflict
/// resolver diffs these identities between a rewrite's read version and the
/// current manifest to detect a concurrent reordered rewrite that no
/// transaction file can reveal. Identity, not count: a concurrent trim plus
/// a concurrent stable-partition append can net a zero count change while
/// still introducing an unvalidated transition.
pub(crate) async fn stable_partition_map_ids(
    dataset: &Dataset,
) -> lance_core::Result<HashSet<String>> {
    let stored = super::load_all_indices(dataset).await?;
    let Some(entry) = stored.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME) else {
        return Ok(HashSet::new());
    };
    let ledger = decode_frag_reuse_ledger(dataset, entry).await?;
    Ok(ledger
        .transitions()
        .iter()
        .filter_map(|transition| match transition.mapping() {
            lance_table::system_index::frag_reuse::ledger::Mapping::StablePartition(partition) => {
                Some(partition.map_id.clone())
            }
            _ => None,
        })
        .collect())
}

/// Extract a committed FRI entry's `FragmentReuseIndexDetails` content bytes
/// verbatim, resolving an external reference but never reinterpreting the
/// content: existing legacy versions and transitions keep their exact wire
/// form when an operation carries them forward. Works for index_version 0 and
/// 1 alike (a 0 -> 1 lift is the same bytes reinterpreted under version 1),
/// unlike [`load_frag_reuse_index_details`], which decodes v0 semantics.
pub(crate) async fn load_raw_frag_reuse_content(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> lance_core::Result<Vec<u8>> {
    use bytes::Buf;
    use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};

    let corrupt = |message: &str| Error::corrupt_file_named("FRI details", message);
    let details = index
        .index_details
        .as_ref()
        .filter(|details| details.type_url.ends_with("FragmentReuseIndexDetails"))
        .ok_or_else(|| Error::index("Index details is not for the fragment reuse index"))?;
    let mut wire = bytes::Bytes::copy_from_slice(&details.value);
    let mut content: Option<(u32, bytes::Bytes)> = None;
    while wire.has_remaining() {
        let (tag, wire_type) = decode_key(&mut wire).map_err(|e| corrupt(&e.to_string()))?;
        if wire_type == WireType::LengthDelimited {
            let length = decode_varint(&mut wire).map_err(|e| corrupt(&e.to_string()))?;
            if length > wire.remaining() as u64 {
                return Err(corrupt("FRI details field length exceeds payload"));
            }
            let payload = wire.split_to(length as usize);
            if matches!(tag, 1 | 2) && content.replace((tag, payload)).is_some() {
                return Err(corrupt("multiple FRI content fields"));
            }
        } else {
            skip_field(wire_type, tag, &mut wire, DecodeContext::default())
                .map_err(|e| corrupt(&e.to_string()))?;
        }
    }
    match content {
        None => Err(corrupt("missing FRI content")),
        Some((1, inline)) => Ok(inline.to_vec()),
        Some((_, external)) => {
            let external_file =
                ExternalFile::decode(external).map_err(|e| corrupt(&e.to_string()))?;
            let expected = external_file.size;
            let data = read_fri_external_file(dataset, index, external_file).await?;
            if data.len() as u64 != expected {
                return Err(corrupt(&format!(
                    "external FRI size mismatch: expected {expected}, received {}",
                    data.len()
                )));
            }
            Ok(data.to_vec())
        }
    }
}

/// Assemble the tagged FRI entry a stable-partition rewrite commits, and
/// return it with the `dataset_version` of the entry it appended onto.
///
/// The current entry's content bytes are carried over verbatim (a v0 entry is
/// lifted to index_version 1 by reinterpretation, not re-encoding) and each
/// new transition is appended as another `InlineContent.transitions` element.
/// Before anything is spilled or committed, the binding between the rewrite
/// groups and the transitions is validated (sources and destinations must
/// match the groups' old and new fragments one to one, in order), row counts
/// must be conserved, and the whole assembled content must decode as a valid
/// ledger.
pub(crate) async fn build_frag_reuse_rewrite_entry(
    dataset: &Dataset,
    frag_reuse_rewrite: &FragmentReuseRewrite,
    groups: &[RewriteGroup],
) -> lance_core::Result<(IndexMetadata, Option<u64>)> {
    // The spec excludes tagged histories on stable-row-id tables: the FRI is
    // address-based, and under stable row ids a rewrite's rows keep their
    // ids, so there is no address translation to record. The planner blocks
    // the deferred-compaction combination already; this covers a hand-built
    // rewrite committed directly.
    if dataset.manifest.uses_stable_row_ids() {
        return Err(Error::not_supported(
            "Tagged fragment reuse histories are address-based and excluded on \
             stable-row-id datasets; this rewrite cannot carry transition intent here",
        ));
    }

    let transitions = &frag_reuse_rewrite.transitions;
    if transitions.is_empty() {
        return Err(Error::invalid_input(
            "a fragment-reuse rewrite carries no transitions",
        ));
    }

    // Bind the covered rewrite groups to the transitions, one to one and in
    // order. A group is covered when its old fragments appear among the
    // transitions' sources; a group straddling covered and uncovered sources
    // is rejected (see `ordered_rewrite_groups`).
    let source_ids: HashSet<u64> = transitions
        .iter()
        .flat_map(|transition| transition.sources.iter().map(|source| source.id))
        .collect();
    let mut covered_groups = Vec::with_capacity(transitions.len());
    for group in groups {
        let covered = group
            .old_fragments
            .iter()
            .filter(|frag| source_ids.contains(&frag.id))
            .count();
        if covered == 0 {
            continue;
        }
        if covered != group.old_fragments.len() {
            return Err(Error::invalid_input(
                "a rewrite group mixes transition-covered and order-preserving source fragments",
            ));
        }
        covered_groups.push(group);
    }
    if covered_groups.len() != transitions.len() {
        return Err(Error::invalid_input(format!(
            "the fragment-reuse rewrite lists {} transitions but {} rewrite groups are \
             covered by their sources",
            transitions.len(),
            covered_groups.len()
        )));
    }
    // Destination ids must be finalized before assembly: the commit path's
    // `fragments_with_ids` treats id 0 as unassigned and renumbers it, which
    // would strand the recorded destination id (and, with a live fragment 0,
    // translate into unrelated rows). The flow reserves fresh ids through
    // ReserveFragments; enforce that invariant here instead of trusting the
    // caller: no id 0, and no id that is already live in the manifest (the
    // transition's sources are live and are covered by the same rule).
    let live_fragments: HashSet<u64> = dataset.fragments().iter().map(|frag| frag.id).collect();

    for (group, transition) in covered_groups.iter().zip(transitions.iter()) {
        if group.old_fragments.len() != transition.sources.len() {
            return Err(Error::invalid_input(format!(
                "a transition lists {} sources but its rewrite group holds {} old fragments",
                transition.sources.len(),
                group.old_fragments.len()
            )));
        }
        for (frag, digest) in group.old_fragments.iter().zip(transition.sources.iter()) {
            let physical_rows = frag.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "source fragment {} has no physical row count",
                    frag.id
                ))
            })? as u64;
            let num_deleted_rows = frag
                .deletion_file
                .as_ref()
                .and_then(|deletion| deletion.num_deleted_rows)
                .unwrap_or(0) as u64;
            if digest.id != frag.id
                || digest.physical_rows != physical_rows
                || digest.num_deleted_rows != num_deleted_rows
            {
                return Err(Error::invalid_input(format!(
                    "transition source digest {:?} does not match old fragment {} \
                     ({physical_rows} physical rows, {num_deleted_rows} deleted)",
                    digest, frag.id
                )));
            }
        }
        if group.new_fragments.len() != transition.destinations.len() {
            return Err(Error::invalid_input(format!(
                "a transition lists {} destinations but its rewrite group holds {} new fragments",
                transition.destinations.len(),
                group.new_fragments.len()
            )));
        }
        for (frag, digest) in group
            .new_fragments
            .iter()
            .zip(transition.destinations.iter())
        {
            if frag.id == 0 {
                return Err(Error::invalid_input(
                    "destination fragment id 0 is unassigned (the commit path renumbers it); \
                     reserve ids with ReserveFragments and assign them before the rewrite is \
                     assembled",
                ));
            }
            if live_fragments.contains(&frag.id) {
                return Err(Error::invalid_input(format!(
                    "destination fragment id {} is already live in the dataset; a rewrite \
                     destination must use a freshly reserved id",
                    frag.id
                )));
            }
            // The digest claiming zero deletions is not enough: check the
            // fragment itself, or a destination carrying a deletion file
            // would commit a digest that undercounts its physical rows'
            // liveness and later fail (or falsely pass) translation.
            if frag.deletion_file.is_some() {
                return Err(Error::invalid_input(format!(
                    "destination fragment {} carries a deletion file; rewrite destinations                      must be written without deletions",
                    frag.id
                )));
            }
            let physical_rows = frag.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "destination fragment {} has no physical row count",
                    frag.id
                ))
            })? as u64;
            if digest.id != frag.id
                || digest.physical_rows != physical_rows
                || digest.num_deleted_rows != 0
            {
                return Err(Error::invalid_input(format!(
                    "transition destination digest {:?} does not match new fragment {} \
                     ({physical_rows} physical rows)",
                    digest, frag.id
                )));
            }
        }
        // Conservation: every live source row lands in exactly one
        // destination. The digests were just bound to the actual fragments,
        // so this checks the fragments themselves.
        let live_source_rows: u64 = transition
            .sources
            .iter()
            .map(|digest| digest.physical_rows.saturating_sub(digest.num_deleted_rows))
            .sum();
        let destination_rows: u64 = transition
            .destinations
            .iter()
            .map(|digest| digest.physical_rows)
            .sum();
        if live_source_rows != destination_rows {
            return Err(Error::invalid_input(format!(
                "a transition does not conserve rows: {live_source_rows} live source rows, \
                 {destination_rows} destination rows"
            )));
        }
        // TODO(row-map totals): also validate the transition's row-map label
        // totals against the destination digests by tail-reading the map
        // file's counts buffer (RowMapReader keeps per-destination totals);
        // today that costs one object-store read per transition, so the
        // ledger's digest conservation stands in for it at commit time.
    }

    // Carry the current entry's content bytes over verbatim.
    let stored = super::load_all_indices(dataset).await?;
    let existing = stored.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME);
    let (mut content, base_bitmap, base_entry_version) = match existing {
        None => (Vec::new(), RoaringBitmap::new(), None),
        Some(entry) => {
            if !matches!(entry.index_version, 0 | 1) {
                return Err(Error::not_supported(format!(
                    "Cannot append a stable-partition transition to FRI index_version {}; \
                     upgrade to a newer version of Lance",
                    entry.index_version
                )));
            }
            (
                load_raw_frag_reuse_content(dataset, entry).await?,
                entry.fragment_bitmap.clone().unwrap_or_default(),
                Some(entry.dataset_version),
            )
        }
    };
    for transition in transitions {
        // Another `InlineContent.transitions` (field 2) element; repeated
        // protobuf fields concatenate, so appending preserves the existing
        // wire form untouched.
        content.extend_from_slice(&encode_length_delimited_field(
            2,
            &transition.encode_to_vec(),
        ));
    }

    // Commit-side validation of the assembled entry: lineage order, digest
    // conservation, single content field, mapping presence, unknown-mapping
    // detection. Runs on the inline form before any spill.
    let assembled = prost_types::Any {
        type_url: "/lance.table.FragmentReuseIndexDetails".into(),
        value: encode_length_delimited_field(1, &content),
    };
    let ledger = lance_table::system_index::frag_reuse::ledger::FragReuseLedger::decode(
        1,
        &assembled,
        |_| async {
            Err(Error::invalid_input(
                "the assembled FRI content is inline; no external read is possible",
            ))
        },
    )
    .await?;
    // The decode above uses READER semantics, which deliberately skip
    // transitions with unknown mappings instead of failing; a writer must
    // not maintain a history it cannot fully interpret (spec: "Writers must
    // reject operations that require interpreting or maintaining unsupported
    // mappings").
    if ledger.has_unsupported_transitions() {
        return Err(Error::not_supported(
            "the fragment reuse history contains mappings this writer cannot maintain;              upgrade to a newer version of Lance before rewriting this table",
        ));
    }
    // Every stable-partition transition must own its row map: a reused
    // map_id would let maintenance of one transition delete or overwrite the
    // map file another live transition still references (object-store
    // writes are not create-if-absent, so nothing else catches the clash).
    let mut seen_map_ids = HashSet::new();
    for transition in ledger.transitions() {
        if let lance_table::system_index::frag_reuse::ledger::Mapping::StablePartition(partition) =
            transition.mapping()
            && !seen_map_ids.insert(partition.map_id.clone())
        {
            return Err(Error::invalid_input(format!(
                "stable-partition row-map id {} is referenced by more than one transition;                  each transition must own its row map",
                partition.map_id
            )));
        }
    }

    // Provenance: the previous coverage plus every fragment this rewrite's
    // transitions touch, retired sources deliberately included.
    let mut fragment_bitmap = base_bitmap;
    for transition in transitions {
        for digest in transition
            .sources
            .iter()
            .chain(transition.destinations.iter())
        {
            // In-range by construction: the ledger decode above already
            // validated every digest id against the row-address fragment
            // bound (the one authoritative enforcement point).
            fragment_bitmap.insert(digest.id as u32);
        }
    }

    let index_id = Uuid::new_v4();
    let details_value = if content.len() > 204800 {
        let file_path = dataset
            .indices_dir()
            .join(index_id.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME);
        let mut writer = dataset.object_store.create(&file_path).await?;
        writer.write_all(&content).await?;
        writer.shutdown().await?;
        let external_file = ExternalFile {
            path: FRAG_REUSE_DETAILS_FILE_NAME.to_owned(),
            offset: 0,
            size: content.len() as u64,
        };
        encode_length_delimited_field(2, &external_file.encode_to_vec())
    } else {
        assembled.value
    };

    let entry = IndexMetadata {
        uuid: index_id,
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: details_value,
        })),
        index_version: 1,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        // The row-map files live in their own directories referenced from the
        // transitions, not under this entry's uuid.
        files: None,
    };
    Ok((entry, base_entry_version))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::frag_reuse_reader::tests as reader_tests;
    use crate::utils::test::DatagenExt;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{Int32Array, RecordBatch};
    use lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
    use lance_table::format::Fragment;
    use lance_table::format::pb::fragment_reuse_index_details::{
        FragmentDigest, StablePartition, Transition, transition,
    };
    use lance_table::system_index::frag_reuse::FragDigest;
    use lance_table::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping};
    use lance_table::system_index::frag_reuse::metadata::is_tagged;
    use lance_table::transaction::{Operation, Transaction};
    use roaring::RoaringTreemap;

    async fn sorted_values(dataset: &Dataset) -> Vec<i32> {
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let mut values: Vec<i32> = batch["i"]
            .as_primitive::<Int32Type>()
            .iter()
            .map(|value| value.unwrap())
            .collect();
        values.sort_unstable();
        values
    }

    async fn reserve_fragments(dataset: &mut Dataset, num_fragments: u32) {
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::ReserveFragments { num_fragments },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
    }

    fn stored_fri(indices: &[IndexMetadata]) -> IndexMetadata {
        indices
            .iter()
            .find(|idx| idx.name == FRAG_REUSE_INDEX_NAME)
            .cloned()
            .unwrap()
    }

    async fn decode_entry(dataset: &Dataset, entry: &IndexMetadata) -> FragReuseLedger {
        decode_frag_reuse_ledger(dataset, entry).await.unwrap()
    }

    /// One atomic Rewrite carries the whole recluster: fragments swapped, the
    /// tagged entry installed, provenance bitmaps untouched, reads identical.
    #[tokio::test]
    async fn stable_partition_rewrite_commits_atomically() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 20).await;
        let before = sorted_values(&dataset).await;
        assert_eq!(before, (0..8).collect::<Vec<_>>());

        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        let read_version = dataset.manifest.version;
        let committed = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: old_fragments.clone(),
                        new_fragments: destinations.clone(),
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition.clone()],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap();
        let mut dataset = committed;

        // The table became tagged in the same commit.
        let flag = FLAG_FRAGMENT_REUSE_INDEX;
        assert_eq!(dataset.manifest.reader_feature_flags & flag, flag);
        assert_eq!(dataset.manifest.writer_feature_flags & flag, flag);
        let live_ids: Vec<u64> = dataset.fragments().iter().map(|frag| frag.id).collect();
        assert_eq!(live_ids, vec![10, 11]);
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        assert!(is_tagged(&entry));
        assert_eq!(entry.index_version, 1);
        assert_eq!(
            entry.fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([0u32, 1, 10, 11])
        );
        // The scalar index keeps its retired source ids as provenance; the
        // tagged reader depends on the stored bitmaps staying untouched.
        let scalar = stored.iter().find(|idx| idx.name == "i_idx").unwrap();
        assert_eq!(
            scalar.fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([0u32, 1])
        );
        let ledger = decode_entry(&dataset, &entry).await;
        assert_eq!(ledger.transitions().len(), 1);

        // Reads are row-identical, unfiltered and through the translated
        // index -- asserted on the returned VALUES, not just counts, so a
        // wrong-but-live translation cannot pass.
        assert_eq!(sorted_values(&dataset).await, before);
        assert_eq!(filtered_values(&dataset, "i = 3").await, vec![3]);
        assert_eq!(filtered_values(&dataset, "i >= 4").await, vec![4, 5, 6, 7]);

        // A second stable-partition rewrite passes the tagged gate and
        // appends onto the v1 entry, preserving its bytes verbatim.
        let first_content = load_raw_frag_reuse_content(&dataset, &entry).await.unwrap();
        reserve_fragments(&mut dataset, 20).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (mut transition, mut destinations) = reader_tests::prepare(&dataset).await;
        for (i, fragment) in destinations.iter_mut().enumerate() {
            fragment.id = 20 + i as u64;
            transition.destinations[i].id = 20 + i as u64;
        }
        let read_version = dataset.manifest.version;
        let dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap();
        let live_ids: Vec<u64> = dataset.fragments().iter().map(|frag| frag.id).collect();
        assert_eq!(live_ids, vec![20, 21]);
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        assert_eq!(entry.index_version, 1);
        let second_content = load_raw_frag_reuse_content(&dataset, &entry).await.unwrap();
        assert!(second_content.starts_with(&first_content));
        let ledger = decode_entry(&dataset, &entry).await;
        assert_eq!(ledger.transitions().len(), 2);
        let scalar = stored.iter().find(|idx| idx.name == "i_idx").unwrap();
        assert_eq!(
            scalar.fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([0u32, 1])
        );
        assert_eq!(sorted_values(&dataset).await, before);
        assert_eq!(
            dataset.count_rows(Some("i = 3".to_string())).await.unwrap(),
            1
        );
    }

    /// A 0 -> 1 lift reinterprets the committed v0 bytes without re-encoding
    /// them: the assembled content starts with the exact previous wire form.
    #[tokio::test]
    async fn lift_preserves_v0_content_bytes() {
        let mut dataset = reader_tests::fixture().await;

        // A committed v0 entry with one legacy compaction (100 -> 110).
        let mut addrs = RoaringTreemap::new();
        for offset in 0..4u64 {
            addrs.insert((100 << 32) + offset);
        }
        let mut changed_row_addrs = Vec::new();
        addrs.serialize_into(&mut changed_row_addrs).unwrap();
        let digest = |id: u64| FragDigest {
            id,
            physical_rows: 4,
            num_deleted_rows: 0,
        };
        let details = FragReuseIndexDetails {
            versions: vec![FragReuseVersion {
                dataset_version: 1,
                groups: vec![FragReuseGroup {
                    changed_row_addrs,
                    old_frags: vec![digest(100)],
                    new_frags: vec![digest(110)],
                }],
            }],
        };
        let v0_entry = build_frag_reuse_index_metadata(
            &dataset,
            None,
            details.clone(),
            RoaringBitmap::from_iter([110u32]),
        )
        .await
        .unwrap();
        assert_eq!(v0_entry.index_version, 0);
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![v0_entry],
                        removed_indices: vec![],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let v0_entry = stored_fri(&stored);
        let v0_content = load_raw_frag_reuse_content(&dataset, &v0_entry)
            .await
            .unwrap();
        assert_eq!(v0_content, InlineContent::from(&details).encode_to_vec());

        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        let frag_reuse_rewrite = FragmentReuseRewrite {
            transitions: vec![transition.clone()],
            base_entry_version: None,
        };
        let groups = vec![RewriteGroup {
            old_fragments,
            new_fragments: destinations,
        }];
        let (entry, base_entry_version) =
            build_frag_reuse_rewrite_entry(&dataset, &frag_reuse_rewrite, &groups)
                .await
                .unwrap();
        assert_eq!(base_entry_version, Some(v0_entry.dataset_version));
        assert_eq!(entry.index_version, 1);
        let lifted = load_raw_frag_reuse_content(&dataset, &entry).await.unwrap();
        assert!(lifted.starts_with(&v0_content));
        assert_eq!(
            &lifted[v0_content.len()..],
            encode_length_delimited_field(2, &transition.encode_to_vec()).as_slice()
        );
        assert_eq!(
            entry.fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([0u32, 1, 10, 11, 110])
        );
    }

    #[tokio::test]
    async fn binding_mismatch_rejected() {
        let dataset = reader_tests::fixture().await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        let groups = |old: Vec<Fragment>, new: Vec<Fragment>| {
            vec![RewriteGroup {
                old_fragments: old,
                new_fragments: new,
            }]
        };
        let sp = |transitions: Vec<Transition>| FragmentReuseRewrite {
            transitions,
            base_entry_version: None,
        };

        // A group straddling covered and uncovered sources.
        let mut with_extra = old_fragments.clone();
        let mut foreign = Fragment::new(99);
        foreign.physical_rows = Some(4);
        with_extra.push(foreign);
        let error = build_frag_reuse_rewrite_entry(
            &dataset,
            &sp(vec![transition.clone()]),
            &groups(with_extra, destinations.clone()),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("mixes"), "{error}");

        // No group covered by the transition's sources.
        let error = build_frag_reuse_rewrite_entry(&dataset, &sp(vec![transition.clone()]), &[])
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("covered by their sources"),
            "{error}"
        );

        // A source digest that disagrees with its fragment.
        let mut tampered = transition.clone();
        tampered.sources[0].physical_rows += 1;
        let error = build_frag_reuse_rewrite_entry(
            &dataset,
            &sp(vec![tampered]),
            &groups(old_fragments.clone(), destinations.clone()),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("source digest"), "{error}");

        // Destinations out of order relative to the row map's label space.
        let mut reversed = destinations.clone();
        reversed.reverse();
        let error = build_frag_reuse_rewrite_entry(
            &dataset,
            &sp(vec![transition.clone()]),
            &groups(old_fragments, reversed),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("destination digest"), "{error}");
    }

    #[tokio::test]
    async fn conservation_violation_rejected() {
        let dataset = reader_tests::fixture().await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (mut transition, mut destinations) = reader_tests::prepare(&dataset).await;
        // Drop the second destination consistently from the digests and the
        // group: the binding holds, but half the live rows have no home.
        transition.destinations.truncate(1);
        destinations.truncate(1);
        let error = build_frag_reuse_rewrite_entry(
            &dataset,
            &FragmentReuseRewrite {
                transitions: vec![transition],
                base_entry_version: None,
            },
            &[RewriteGroup {
                old_fragments,
                new_fragments: destinations,
            }],
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("does not conserve rows"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn ledger_invalid_assembly_rejected() {
        let dataset = reader_tests::fixture().await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (mut transition, destinations) = reader_tests::prepare(&dataset).await;
        // The binding never opens the mapping; the ledger validation does.
        let Some(transition::Mapping::StablePartition(mapping)) = &mut transition.mapping else {
            unreachable!()
        };
        mapping.map_id = "not-a-uuid".to_string();
        let error = build_frag_reuse_rewrite_entry(
            &dataset,
            &FragmentReuseRewrite {
                transitions: vec![transition],
                base_entry_version: None,
            },
            &[RewriteGroup {
                old_fragments,
                new_fragments: destinations,
            }],
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("map_id"), "{error}");
    }

    #[tokio::test]
    async fn oversized_assembly_spills_to_external_file() {
        let dataset = reader_tests::fixture().await;
        // Enough synthetic transitions to exceed the 200KB inline threshold.
        // The binding only checks transitions against their groups, so the
        // fragments need not exist in the dataset.
        let mut transitions = Vec::new();
        let mut groups = Vec::new();
        for i in 0..4000u64 {
            let fragment = |id: u64| {
                let mut fragment = Fragment::new(id);
                fragment.physical_rows = Some(4);
                fragment
            };
            let digest = |id: u64| FragmentDigest {
                id,
                physical_rows: 4,
                num_deleted_rows: 0,
            };
            transitions.push(Transition {
                sources: vec![digest(1_000 + i)],
                destinations: vec![digest(100_000 + i)],
                mapping: Some(transition::Mapping::StablePartition(StablePartition {
                    map_id: Uuid::new_v4().to_string(),
                    map_size_bytes: 1,
                    base_id: None,
                })),
            });
            groups.push(RewriteGroup {
                old_fragments: vec![fragment(1_000 + i)],
                new_fragments: vec![fragment(100_000 + i)],
            });
        }
        let expected: Vec<u8> = transitions
            .iter()
            .flat_map(|transition| encode_length_delimited_field(2, &transition.encode_to_vec()))
            .collect();
        assert!(expected.len() > 204800);

        let (entry, base_entry_version) = build_frag_reuse_rewrite_entry(
            &dataset,
            &FragmentReuseRewrite {
                transitions,
                base_entry_version: None,
            },
            &groups,
        )
        .await
        .unwrap();
        assert_eq!(base_entry_version, None);
        // The details reference an external file whose bytes are the content.
        let details = entry.index_details.as_ref().unwrap();
        let proto = FragmentReuseIndexDetails::decode(details.value.as_slice()).unwrap();
        let Some(Content::External(external)) = proto.content else {
            panic!("expected external content, got {proto:?}");
        };
        assert_eq!(external.path, FRAG_REUSE_DETAILS_FILE_NAME);
        assert_eq!(external.size, expected.len() as u64);
        assert_eq!(
            load_raw_frag_reuse_content(&dataset, &entry).await.unwrap(),
            expected
        );
    }

    /// Task-chain end to end: a deferred compaction after the atomic
    /// stable-partition commit appends an ordered-compaction transition to
    /// the tagged entry (not a legacy version), and index queries translate
    /// through the two-hop chain (stable partition, then compaction).
    #[tokio::test]
    async fn deferred_compaction_chains_onto_stable_partition() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 20).await;
        let before = sorted_values(&dataset).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        let read_version = dataset.manifest.version;
        let mut dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap();
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let sp_entry = stored_fri(&stored);
        let sp_content = load_raw_frag_reuse_content(&dataset, &sp_entry)
            .await
            .unwrap();

        let metrics = crate::dataset::optimize::compact_files(
            &mut dataset,
            crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 100,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        assert_eq!(entry.index_version, 1);
        assert!(is_tagged(&entry));
        // Appended, not re-encoded: the stable-partition record is intact
        // byte for byte and the compaction rides a transition, not a legacy
        // version.
        let content = load_raw_frag_reuse_content(&dataset, &entry).await.unwrap();
        assert!(content.starts_with(&sp_content));
        let ledger = decode_entry(&dataset, &entry).await;
        assert_eq!(ledger.transitions().len(), 2);
        assert!(matches!(
            ledger.transitions()[0].mapping(),
            Mapping::StablePartition(_)
        ));
        assert!(matches!(
            ledger.transitions()[1].mapping(),
            Mapping::OrderedCompaction(_)
        ));
        // Lineage: the compaction consumed the stable partition's
        // destinations.
        assert!(ledger.consumer(10).is_some());
        assert!(ledger.consumer(11).is_some());
        assert!(ledger.consumer(0).is_some());
        // Index provenance is still the original coverage.
        let scalar = stored.iter().find(|idx| idx.name == "i_idx").unwrap();
        assert_eq!(
            scalar.fragment_bitmap.as_ref().unwrap(),
            &RoaringBitmap::from_iter([0u32, 1])
        );

        // Reads translate through both hops -- asserted on the returned
        // VALUES, not just counts, so a wrong-but-live translation cannot
        // pass.
        assert_eq!(sorted_values(&dataset).await, before);
        assert_eq!(filtered_values(&dataset, "i = 3").await, vec![3]);
        assert_eq!(filtered_values(&dataset, "i >= 4").await, vec![4, 5, 6, 7]);
    }

    async fn filtered_values(dataset: &Dataset, predicate: &str) -> Vec<i32> {
        let mut scan = dataset.scan();
        scan.filter(predicate).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let mut values: Vec<i32> = batch["i"]
            .as_primitive::<Int32Type>()
            .iter()
            .map(|value| value.unwrap())
            .collect();
        values.sort_unstable();
        values
    }

    /// A pre-assembled tagged entry without `frag_reuse_rewrite` intent has
    /// bypassed assembly and validation; the commit chokepoint rejects it
    /// even though the empty-conflicts finish path passes it through
    /// unchanged. With intent, the same commit works (the atomic e2e above).
    #[tokio::test]
    async fn tagged_entry_without_intent_rejected_at_commit() {
        let mut dataset = reader_tests::fixture().await;
        let entry = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: FRAG_REUSE_INDEX_NAME.to_string(),
            fields: vec![],
            covering_fields: vec![],
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(RoaringBitmap::from_iter([0u32, 1])),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "/lance.table.FragmentReuseIndexDetails".into(),
                value: encode_length_delimited_field(1, &[]),
            })),
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
        };
        let error = dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::Rewrite {
                        groups: vec![],
                        rewritten_indices: vec![],
                        frag_reuse_index: Some(entry),
                        frag_reuse_rewrite: None,
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("must be assembled"), "{error}");
    }

    /// The spec excludes tagged histories on stable-row-id tables; a
    /// hand-built rewrite carrying transition intent is refused at assembly,
    /// before anything is written or committed.
    #[tokio::test]
    async fn frag_reuse_rewrite_rejected_on_stable_row_id_dataset() {
        let dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset_with_params(
                crate::utils::test::FragmentCount::from(2),
                crate::utils::test::FragmentRowCount::from(4),
                Some(crate::dataset::WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 4,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert!(dataset.manifest.uses_stable_row_ids());
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(Transaction::new(
                dataset.manifest.version,
                Operation::Rewrite {
                    groups: vec![],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("stable-row-id"), "{error}");
    }

    /// FIX 1: the assembly's ledger decode uses reader semantics, which skip
    /// transitions with unknown mappings; the writer must refuse to maintain
    /// such a history instead of silently carrying it forward.
    #[tokio::test]
    async fn unknown_mapping_in_history_rejects_rewrite() {
        let mut dataset = reader_tests::fixture().await;
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        // A transition carrying a field this writer does not know: the
        // reader-side decode skips it and flags the history unsupported.
        let mut unknown_raw = transition.encode_to_vec();
        unknown_raw.extend_from_slice(&reader_tests::field(9, b"future-mapping-payload"));
        let content = reader_tests::field(2, &unknown_raw);
        reader_tests::install(&mut dataset, content, destinations, false).await;
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        reader_tests::persist_fixture(&mut dataset, indices).await;

        reserve_fragments(&mut dataset, 30).await;
        let version = dataset.latest_version_id().await.unwrap();
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let source_ids: Vec<u64> = old_fragments.iter().map(|f| f.id).collect();
        let (new_transition, new_destinations) =
            reader_tests::prepare_partition(&dataset, &source_ids, 20).await;
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(Transaction::new(
                version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: new_destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![new_transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("cannot maintain"), "{error}");
        // Nothing was committed.
        assert_eq!(dataset.latest_version_id().await.unwrap(), version);
    }

    /// FIX 2: a shallow clone's index metadata is base-stamped and its
    /// external `details.binpb` lives in the SOURCE dataset; the first
    /// rewrite on the clone must resolve it through the entry's base and
    /// carry the lifted legacy content verbatim.
    #[tokio::test]
    async fn shallow_clone_resolves_external_details_from_source_base() {
        // On disk: base-path resolution must reach the SOURCE dataset's
        // store, which in-memory fixtures cannot demonstrate.
        let source_dir = lance_core::utils::tempfile::TempStrDir::default();
        let target_dir = lance_core::utils::tempfile::TempStrDir::default();
        let target_uri = format!("{}/clone", target_dir.as_str());
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_dataset(
                source_dir.as_str(),
                crate::utils::test::FragmentCount::from(2),
                crate::utils::test::FragmentRowCount::from(4),
            )
            .await
            .unwrap();
        // A v0 entry whose details spill to an external file (>200KB).
        let digest = |id: u64| FragDigest {
            id,
            physical_rows: 4,
            num_deleted_rows: 0,
        };
        let mut versions = Vec::new();
        for i in 0..3500u64 {
            let old_id = 1_000 + i;
            let mut addrs = RoaringTreemap::new();
            for offset in 0..4u64 {
                addrs.insert((old_id << 32) + offset);
            }
            let mut serialized = Vec::new();
            addrs.serialize_into(&mut serialized).unwrap();
            versions.push(FragReuseVersion {
                dataset_version: i + 1,
                groups: vec![FragReuseGroup {
                    changed_row_addrs: serialized,
                    old_frags: vec![digest(old_id)],
                    new_frags: vec![digest(100_000 + i)],
                }],
            });
        }
        let details = FragReuseIndexDetails { versions };
        let bitmap: RoaringBitmap = (0..3500u32).map(|i| 100_000 + i).collect();
        let entry = build_frag_reuse_index_metadata(&dataset, None, details, bitmap)
            .await
            .unwrap();
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![entry],
                        removed_indices: vec![],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let source_entry = stored_fri(&stored);
        let source_content = load_raw_frag_reuse_content(&dataset, &source_entry)
            .await
            .unwrap();
        assert!(source_content.len() > 204800);

        let version = dataset.manifest.version;
        let mut clone = dataset
            .shallow_clone(target_uri.as_str(), version, None)
            .await
            .unwrap();
        let stored = crate::index::load_all_indices(&clone).await.unwrap();
        let cloned_entry = stored_fri(&stored);
        assert!(cloned_entry.base_id.is_some());

        // First rewrite on the clone: assembly must read the source's
        // external details through the entry's base.
        reserve_fragments(&mut clone, 30).await;
        let read_version = clone.manifest.version;
        let old_fragments: Vec<Fragment> = clone.fragments().iter().cloned().collect();
        let (transition, destinations) = reader_tests::prepare(&clone).await;
        let clone = crate::dataset::write::CommitBuilder::new(Arc::new(clone))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap();

        let stored = crate::index::load_all_indices(&clone).await.unwrap();
        let entry = stored_fri(&stored);
        assert_eq!(entry.index_version, 1);
        // The assembled entry carries the source's legacy content verbatim.
        let content = load_raw_frag_reuse_content(&clone, &entry).await.unwrap();
        assert!(content.starts_with(&source_content));
        let ledger = decode_entry(&clone, &entry).await;
        assert_eq!(ledger.transitions().len(), 3501);
    }

    /// Round-4 addendum: every stable-partition transition must own its row
    /// map. A caller bug reusing an existing map_id would let maintenance of
    /// one transition destroy the map another live transition references, so
    /// the assembly rejects the duplicate by name and nothing commits.
    #[tokio::test]
    async fn duplicate_row_map_id_rejected() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        let Some(transition::Mapping::StablePartition(first_mapping)) = &transition.mapping else {
            unreachable!()
        };
        let first_map_id = first_mapping.map_id.clone();
        let read_version = dataset.manifest.version;
        let mut dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap();

        reserve_fragments(&mut dataset, 30).await;
        let version = dataset.latest_version_id().await.unwrap();
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (mut reused, new_destinations) =
            reader_tests::prepare_partition(&dataset, &[10, 11], 40).await;
        let Some(transition::Mapping::StablePartition(mapping)) = &mut reused.mapping else {
            unreachable!()
        };
        mapping.map_id = first_map_id.clone();
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(Transaction::new(
                version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: new_destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![reused],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(&first_map_id), "{error}");
        assert_eq!(dataset.latest_version_id().await.unwrap(), version);
    }

    /// Round 5 item 4: deferred compaction of fragments no index covers and
    /// no lineage reaches commits a plain rewrite on a tagged table -- no
    /// new transition, the entry untouched -- instead of growing the history
    /// with records no reader ever has to translate.
    #[tokio::test]
    async fn uncovered_deferred_compaction_commits_plain_rewrite() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 20).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        let read_version = dataset.manifest.version;
        let dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap();

        // Two two-row fragments outside every bitmap: with a target of four
        // rows they are the only compaction candidates.
        let schema = Arc::new(arrow_schema::Schema::from(dataset.schema()));
        let dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(100..102))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();
        let mut dataset = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![
                RecordBatch::try_new(
                    schema,
                    vec![Arc::new(Int32Array::from_iter_values(102..104))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();

        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry_before = stored_fri(&stored);
        let before = sorted_values(&dataset).await;

        let metrics = crate::dataset::optimize::compact_files(
            &mut dataset,
            crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 4,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        // The entry is byte-for-byte the one from before the compaction.
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        assert_eq!(entry.uuid, entry_before.uuid);
        let ledger = decode_entry(&dataset, &entry).await;
        assert_eq!(ledger.transitions().len(), 1);
        assert_eq!(sorted_values(&dataset).await, before);
        assert_eq!(filtered_values(&dataset, "i = 3").await, vec![3]);
        assert_eq!(
            filtered_values(&dataset, "i >= 100").await,
            (100..104).collect::<Vec<_>>()
        );
    }

    /// Round 5 item 7: detached commits skip the rebase pipeline, so nothing
    /// would assemble or validate transition intent, and a detached manifest
    /// is outside the version chain where an appended history has meaning.
    /// Both intent-carrying and tagged-entry-carrying rewrites are refused.
    #[tokio::test]
    async fn detached_commit_rejects_transition_intent_and_tagged_entries() {
        let dataset = reader_tests::fixture().await;
        let version = dataset.manifest.version;
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
            .with_detached(true)
            .execute(Transaction::new(
                version,
                Operation::Rewrite {
                    groups: vec![],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("Detached commits"), "{error}");

        let entry = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: FRAG_REUSE_INDEX_NAME.to_string(),
            fields: vec![],
            covering_fields: vec![],
            dataset_version: version,
            fragment_bitmap: Some(RoaringBitmap::from_iter([0u32, 1])),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "/lance.table.FragmentReuseIndexDetails".into(),
                value: encode_length_delimited_field(1, &[]),
            })),
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
        };
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .with_detached(true)
            .execute(Transaction::new(
                version,
                Operation::Rewrite {
                    groups: vec![],
                    rewritten_indices: vec![],
                    frag_reuse_index: Some(entry),
                    frag_reuse_rewrite: None,
                },
                None,
            ))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("Detached commits"), "{error}");
    }

    /// Round 5 item 9: a destination fragment carrying a deletion file is
    /// rejected by the binding even when its digest claims zero deletions.
    #[tokio::test]
    async fn destination_with_deletion_file_rejected() {
        let dataset = reader_tests::fixture().await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (transition, mut destinations) = reader_tests::prepare(&dataset).await;
        destinations[0].deletion_file = Some(lance_table::format::DeletionFile {
            read_version: 1,
            id: 1,
            file_type: lance_table::format::DeletionFileType::Array,
            num_deleted_rows: Some(1),
            base_id: None,
        });
        let error = build_frag_reuse_rewrite_entry(
            &dataset,
            &FragmentReuseRewrite {
                transitions: vec![transition],
                base_entry_version: None,
            },
            &[RewriteGroup {
                old_fragments,
                new_fragments: destinations,
            }],
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("carries a deletion file"),
            "{error}"
        );
    }

    /// Round 6 item 1: the commit path renumbers destination id 0
    /// (`fragments_with_ids` treats it as unassigned), which would strand
    /// the recorded destination id; assembly must reject it up front.
    #[tokio::test]
    async fn unassigned_destination_id_rejected_at_commit() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (mut transition, mut destinations) = reader_tests::prepare(&dataset).await;
        destinations[0].id = 0;
        transition.destinations[0].id = 0;
        let read_version = dataset.manifest.version;
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("unassigned"), "{error}");
        assert_eq!(dataset.latest_version_id().await.unwrap(), read_version);
    }

    /// Round 6 item 1: a destination id colliding with a fragment that is
    /// live in the manifest (here one of the rewrite's own sources) is
    /// rejected; destinations must use freshly reserved ids.
    #[tokio::test]
    async fn live_destination_id_rejected_at_commit() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (mut transition, mut destinations) = reader_tests::prepare(&dataset).await;
        destinations[0].id = 1;
        transition.destinations[0].id = 1;
        let read_version = dataset.manifest.version;
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("already live"), "{error}");
    }

    async fn indexed_three_fragment_dataset() -> Dataset {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(
                crate::utils::test::FragmentCount::from(3),
                crate::utils::test::FragmentRowCount::from(4),
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["i"],
                lance_index::IndexType::Scalar,
                Some("i_idx".into()),
                &lance_index::scalar::ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        dataset
    }

    /// Tag the table by rewriting fragment 0 only, leaving 1 and 2 as
    /// indexed compaction candidates.
    async fn tag_fragment_zero(mut dataset: Dataset) -> Dataset {
        reserve_fragments(&mut dataset, 40).await;
        let old_fragments: Vec<Fragment> = dataset
            .fragments()
            .iter()
            .filter(|f| f.id == 0)
            .cloned()
            .collect();
        let (transition, destinations) = reader_tests::prepare_partition(&dataset, &[0], 10).await;
        let read_version = dataset.manifest.version;
        crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap()
    }

    /// Round 6 item 2: a source fragment with a deletion file whose count
    /// was never materialized in the manifest. The digests and the rewrite
    /// group must be built from the same normalized metadata, so the commit
    /// succeeds and conservation uses the real deleted count.
    #[tokio::test]
    async fn deferred_compaction_materializes_missing_deletion_counts() {
        let mut dataset = indexed_three_fragment_dataset().await;
        dataset.delete("i = 5").await.unwrap();
        // Strip the materialized count, as a legacy writer may leave it.
        let mut fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
        let deletion = fragments
            .iter_mut()
            .find(|f| f.id == 1)
            .unwrap()
            .deletion_file
            .as_mut()
            .unwrap();
        assert!(deletion.num_deleted_rows.is_some());
        deletion.num_deleted_rows = None;
        Arc::make_mut(&mut dataset.manifest).fragments = Arc::new(fragments);
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        reader_tests::persist_fixture(&mut dataset, indices).await;
        assert!(
            dataset
                .fragments()
                .iter()
                .find(|f| f.id == 1)
                .unwrap()
                .deletion_file
                .as_ref()
                .unwrap()
                .num_deleted_rows
                .is_none()
        );

        let mut dataset = tag_fragment_zero(dataset).await;

        // Deferred compaction over everything, including the fragment with
        // the unmaterialized deletion count.
        let metrics = crate::dataset::optimize::compact_files(
            &mut dataset,
            crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 100,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(metrics.fragments_removed, 4);
        assert_eq!(metrics.fragments_added, 2);

        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        let ledger = decode_entry(&dataset, &entry).await;
        assert_eq!(ledger.transitions().len(), 3);
        assert!(ledger.consumer(1).is_some());
        // Conservation held with the real deleted count: 12 rows minus one.
        assert_eq!(dataset.count_rows(None).await.unwrap(), 11);
        assert_eq!(filtered_values(&dataset, "i = 5").await, Vec::<i32>::new());
        assert_eq!(filtered_values(&dataset, "i = 4").await, vec![4]);
    }

    /// Round 6 item 2: heavy deletions (three of four rows) flow through
    /// the same normalized digests; conservation and translation stay
    /// correct. A source with EVERY row deleted cannot be produced through
    /// the public API (the delete path drops fully-deleted fragments from
    /// the manifest); `normalize_source_fragments` keeping such a fragment
    /// is covered by a unit test next to it in `optimize`.
    #[tokio::test]
    async fn deferred_compaction_consumes_heavily_deleted_source() {
        let mut dataset = indexed_three_fragment_dataset().await;
        dataset.delete("i >= 9").await.unwrap();
        let mut dataset = tag_fragment_zero(dataset).await;

        crate::dataset::optimize::compact_files(
            &mut dataset,
            crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 100,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        let ledger = decode_entry(&dataset, &entry).await;
        assert!(ledger.consumer(2).is_some());
        assert_eq!(dataset.count_rows(None).await.unwrap(), 9);
        assert_eq!(filtered_values(&dataset, "i = 8").await, vec![8]);
        assert_eq!(filtered_values(&dataset, "i >= 9").await, Vec::<i32>::new());
    }
}
