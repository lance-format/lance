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
use lance_table::transaction::{RewriteGroup, StablePartitionRewrite};
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
            let file_path = dataset
                .indices_dir()
                .join(index.uuid.to_string())
                .join(external_file.path.clone());

            // the file content will be cached in the index cache later
            // so we do not put it to the file cache
            let range = external_file.offset as usize
                ..(external_file.offset as usize + external_file.size as usize);
            let data = dataset
                .object_store
                .open(&file_path)
                .await?
                .get_range(range)
                .await?;

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

/// Decode a committed FRI entry into its transition ledger, resolving any
/// external content up front so the ledger's `read_external` never fires.
/// Works for v0 entries too: legacy versions decode as lifted transitions.
pub(crate) async fn decode_frag_reuse_ledger(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> lance_core::Result<lance_table::system_index::frag_reuse::ledger::FragReuseLedger> {
    let content = load_raw_frag_reuse_content(dataset, entry).await?;
    let inline = prost_types::Any {
        type_url: "/lance.table.FragmentReuseIndexDetails".into(),
        value: encode_length_delimited_field(1, &content),
    };
    lance_table::system_index::frag_reuse::ledger::FragReuseLedger::decode(
        entry.index_version,
        &inline,
        |_| async {
            Err(Error::invalid_input(
                "re-wrapped FRI content is inline; no external read is possible",
            ))
        },
    )
    .await
}

/// The number of stable-partition transitions in the dataset's committed FRI
/// entry (0 when the entry is absent or v0). The conflict resolver diffs this
/// count between a rewrite's read version and the current manifest to detect
/// a concurrent reordered rewrite that no transaction file can reveal.
pub(crate) async fn stable_partition_transition_count(
    dataset: &Dataset,
) -> lance_core::Result<usize> {
    let stored = super::load_all_indices(dataset).await?;
    let Some(entry) = stored.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME) else {
        return Ok(0);
    };
    let ledger = decode_frag_reuse_ledger(dataset, entry).await?;
    Ok(ledger
        .transitions()
        .iter()
        .filter(|transition| {
            matches!(
                transition.mapping(),
                lance_table::system_index::frag_reuse::ledger::Mapping::StablePartition(_)
            )
        })
        .count())
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
            let file_path = dataset
                .indices_dir()
                .join(index.uuid.to_string())
                .join(external_file.path.clone());
            let range = external_file.offset as usize
                ..(external_file.offset as usize + external_file.size as usize);
            let data = dataset
                .object_store
                .open(&file_path)
                .await?
                .get_range(range)
                .await?;
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
pub(crate) async fn build_stable_partition_rewrite_entry(
    dataset: &Dataset,
    stable_partition: &StablePartitionRewrite,
    groups: &[RewriteGroup],
) -> lance_core::Result<(IndexMetadata, Option<u64>)> {
    let transitions = &stable_partition.transitions;
    if transitions.is_empty() {
        return Err(Error::invalid_input(
            "a stable-partition rewrite carries no transitions",
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
                "a rewrite group mixes stable-partition and order-preserving source fragments",
            ));
        }
        covered_groups.push(group);
    }
    if covered_groups.len() != transitions.len() {
        return Err(Error::invalid_input(format!(
            "the stable-partition rewrite lists {} transitions but {} rewrite groups are \
             covered by their sources",
            transitions.len(),
            covered_groups.len()
        )));
    }
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
    lance_table::system_index::frag_reuse::ledger::FragReuseLedger::decode(
        1,
        &assembled,
        |_| async {
            Err(Error::invalid_input(
                "the assembled FRI content is inline; no external read is possible",
            ))
        },
    )
    .await?;

    // Provenance: the previous coverage plus every fragment this rewrite's
    // transitions touch, retired sources deliberately included.
    let mut fragment_bitmap = base_bitmap;
    for transition in transitions {
        for digest in transition
            .sources
            .iter()
            .chain(transition.destinations.iter())
        {
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
    use crate::index::frag_reuse_reader::tests as reader_tests;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
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
                    stable_partition: Some(StablePartitionRewrite {
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

        // Reads are row-identical, unfiltered and through the translated index.
        assert_eq!(sorted_values(&dataset).await, before);
        assert_eq!(
            dataset.count_rows(Some("i = 3".to_string())).await.unwrap(),
            1
        );
        assert_eq!(
            dataset
                .count_rows(Some("i >= 4".to_string()))
                .await
                .unwrap(),
            4
        );

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
                    stable_partition: Some(StablePartitionRewrite {
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
        let stable_partition = StablePartitionRewrite {
            transitions: vec![transition.clone()],
            base_entry_version: None,
        };
        let groups = vec![RewriteGroup {
            old_fragments,
            new_fragments: destinations,
        }];
        let (entry, base_entry_version) =
            build_stable_partition_rewrite_entry(&dataset, &stable_partition, &groups)
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
        let sp = |transitions: Vec<Transition>| StablePartitionRewrite {
            transitions,
            base_entry_version: None,
        };

        // A group straddling covered and uncovered sources.
        let mut with_extra = old_fragments.clone();
        let mut foreign = Fragment::new(99);
        foreign.physical_rows = Some(4);
        with_extra.push(foreign);
        let error = build_stable_partition_rewrite_entry(
            &dataset,
            &sp(vec![transition.clone()]),
            &groups(with_extra, destinations.clone()),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("mixes"), "{error}");

        // No group covered by the transition's sources.
        let error =
            build_stable_partition_rewrite_entry(&dataset, &sp(vec![transition.clone()]), &[])
                .await
                .unwrap_err();
        assert!(
            error.to_string().contains("covered by their sources"),
            "{error}"
        );

        // A source digest that disagrees with its fragment.
        let mut tampered = transition.clone();
        tampered.sources[0].physical_rows += 1;
        let error = build_stable_partition_rewrite_entry(
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
        let error = build_stable_partition_rewrite_entry(
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
        let error = build_stable_partition_rewrite_entry(
            &dataset,
            &StablePartitionRewrite {
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
        let error = build_stable_partition_rewrite_entry(
            &dataset,
            &StablePartitionRewrite {
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

        let (entry, base_entry_version) = build_stable_partition_rewrite_entry(
            &dataset,
            &StablePartitionRewrite {
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
                    stable_partition: Some(StablePartitionRewrite {
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

        // Reads translate through both hops.
        assert_eq!(sorted_values(&dataset).await, before);
        assert_eq!(
            dataset.count_rows(Some("i = 3".to_string())).await.unwrap(),
            1
        );
        assert_eq!(
            dataset
                .count_rows(Some("i >= 4".to_string()))
                .await
                .unwrap(),
            4
        );
    }
}
