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
use lance_io::object_store::{ObjectStore, ObjectStoreRegistry};
use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
use lance_table::format::{Fragment, IndexMetadata, Manifest};
use lance_table::transaction::{FragmentReuseRewrite, RewriteGroup};
use object_store::path::Path;
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
    let proto = if new_index_details_proto.encoded_len() > FRAG_REUSE_INLINE_DETAILS_LIMIT {
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
// The sp-x-sp manifest diff caller went away when it was replaced by
// merge-through-reassembly; the maintenance stack (trim derivation, tagged
// remap planning) and the commit-path tests call it now.
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

/// Decode already-loaded FRI content bytes into a transition ledger. Used by
/// maintenance paths that hold the entry's verbatim content (the trim's
/// splice inputs and outputs, cleanup's reference resolution) and by tests.
pub(crate) async fn decode_frag_reuse_ledger_from_content(
    index_version: i32,
    content: &[u8],
) -> lance_core::Result<lance_table::system_index::frag_reuse::ledger::FragReuseLedger> {
    let inline = prost_types::Any {
        type_url: "/lance.table.FragmentReuseIndexDetails".into(),
        value: encode_length_delimited_field(1, content),
    };
    lance_table::system_index::frag_reuse::ledger::FragReuseLedger::decode(
        index_version,
        &inline,
        |_| async {
            Err(Error::invalid_input(
                "re-wrapped FRI content is inline; no external read is possible",
            ))
        },
    )
    .await
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
    let details = index
        .index_details
        .as_ref()
        .filter(|details| details.type_url.ends_with("FragmentReuseIndexDetails"))
        .ok_or_else(|| Error::index("Index details is not for the fragment reuse index"))?;
    extract_raw_frag_reuse_content(details, |file| read_fri_external_file(dataset, index, file))
        .await
}

/// [`load_raw_frag_reuse_content`] with the external read abstracted: callers
/// outside a live `Dataset` (the shallow-clone relocation reads the SOURCE
/// dataset's entry before the clone exists; the parent's cleanup reads a
/// BRANCH manifest's entry) supply their own resolution.
pub(crate) async fn extract_raw_frag_reuse_content<F, Fut>(
    details: &prost_types::Any,
    read_external: F,
) -> lance_core::Result<Vec<u8>>
where
    F: FnOnce(ExternalFile) -> Fut,
    Fut: std::future::Future<Output = lance_core::Result<bytes::Bytes>>,
{
    use bytes::Buf;
    use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};

    let corrupt = |message: &str| Error::corrupt_file_named("FRI details", message);
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
            let data = read_external(external_file).await?;
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

/// The source fragment's deleted row count as the manifest sees it right
/// now. Materialized metadata is free; a deletion file whose count was never
/// materialized is read once (rare -- the delete path materializes counts).
async fn current_deleted_rows(dataset: &Dataset, frag: &Fragment) -> lance_core::Result<u64> {
    match &frag.deletion_file {
        None => Ok(0),
        Some(deletion) => match deletion.num_deleted_rows {
            Some(count) => Ok(count as u64),
            None => Ok(
                crate::io::deletion::read_dataset_deletion_file(dataset, frag.id, deletion)
                    .await?
                    .len() as u64,
            ),
        },
    }
}

/// Validate job-side deletion folding with exact accounting.
///
/// A concurrent delete landed on a source after the rewrite job snapshotted
/// it; instead of recomputing, the job translated each newly deleted source
/// row through the transition's own mapping and wrote destination deletion
/// vectors for exactly those positions. Three equations, all equalities:
///
/// 1. per source, the rows of its CURRENT deletion vector that were already
///    dead at rewrite time (a null row-map label for stable partition, an
///    address absent from the survivor bitmap for ordered compaction) must
///    number exactly the digest's rewrite-time deleted count -- this also
///    pins the arithmetic that a delta row can never predate the mapping,
///    and implies the remaining rows number exactly the delta;
/// 2. the destination deletion vectors must hold exactly the summed delta
///    rows -- no extras, no misses;
/// 3. per destination, the translated delta positions must equal the
///    supplied deletion vector as a SET -- a count match with a position
///    mismatch is one row wrongly dead plus one row resurrected.
///
/// AUTHORITY: source deletion state is read from the CURRENT MANIFEST
/// fragments, never from the job-supplied rewrite group, which establishes
/// group identity and ordering only. Trusting the group here would let a
/// crafted group deletion vector resurrect one row while wrongly deleting
/// another with all counts matching.
///
/// The translation is the reader stack's own [`MappingReader`]
/// (`OrderedCompactionMapping` over the transition's survivor bitmap, or
/// `StablePartitionMapping` over the row map file, opened exactly as the
/// reader opens it). Deleted offsets are translated in bounded batches, and
/// the expected destination positions accumulate as one `RoaringBitmap` per
/// destination, so memory stays bounded by the batch size plus the bitmaps.
/// IO: none at all for ordered compaction (the bitmap is already in the
/// transition details); for stable partition, only the label blocks each
/// batch touches, plus the reader's one tail read for the counts matrix.
async fn validate_folded_deletions(
    dataset: &Dataset,
    group: &RewriteGroup,
    transition: &lance_table::format::pb::fragment_reuse_index_details::Transition,
    manifest_by_id: &HashMap<u64, &Fragment>,
    source_deltas: &[u64],
) -> lance_core::Result<()> {
    use lance_core::utils::fragment_reuse::{MappingReader, OrderedCompactionMapping};
    use lance_index::frag_reuse::stable_partition::{MAPPING_FILE, StablePartitionMapping};
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_table::format::pb::fragment_reuse_index_details::transition::Mapping;

    /// Deleted offsets translated per request; bounds the translation
    /// buffers and, for stable partition, the label blocks in flight.
    const FOLDING_BATCH_ROWS: usize = 16 * 1024;

    // One common reader over either mapping kind.
    let mapping: Arc<dyn MappingReader> = match &transition.mapping {
        Some(Mapping::OrderedCompaction(ordered)) => {
            let mut cursor = std::io::Cursor::new(&ordered.changed_row_addrs);
            let bitmap = roaring::RoaringTreemap::deserialize_from(&mut cursor)
                .map_err(|e| Error::invalid_input(e.to_string()))?;
            let layout =
                |digests: &[lance_table::format::pb::fragment_reuse_index_details::FragmentDigest]| {
                    digests
                        .iter()
                        .map(|digest| (digest.id as u32, digest.physical_rows as u32))
                        .collect()
                };
            let remap = lance_core::utils::row_addr_remap::RowAddrRemap::compact_with_layout([
                lance_core::utils::row_addr_remap::GroupInputWithLayout {
                    rewritten_old_row_addrs: bitmap,
                    old_frags: layout(&transition.sources),
                    new_frags: layout(&transition.destinations),
                },
            ])
            .map_err(|e| Error::invalid_input(e.to_string()))?;
            Arc::new(OrderedCompactionMapping::new(Arc::new(remap)))
        }
        Some(Mapping::StablePartition(reference)) => {
            // Open the row map exactly as the reader does (base-aware).
            let base = match reference.base_id {
                None => dataset.base.clone(),
                Some(id) => dataset
                    .manifest
                    .base_paths
                    .get(&id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "mapping {} references missing base {id}",
                            reference.map_id
                        ))
                    })?
                    .extract_path(dataset.session.store_registry())?,
            };
            let directory = base.join("_fri").join(reference.map_id.as_str());
            let store = dataset.object_store(reference.base_id).await?;
            let metadata_cache = dataset.metadata_cache.file_metadata_cache(&directory);
            let store =
                LanceIndexStore::new(store, directory, Arc::new(metadata_cache)).with_file_sizes(
                    HashMap::from([(MAPPING_FILE.to_string(), reference.map_size_bytes)]),
                );
            Arc::new(StablePartitionMapping::try_new(
                Arc::new(store),
                transition.sources.clone(),
                transition.destinations.clone(),
            )?)
        }
        None => {
            return Err(Error::invalid_input(
                "a transition carrying folded deletions has no mapping",
            ));
        }
    };

    // Expected destination deletions, one bitmap per destination.
    let destination_slot: HashMap<u64, usize> = transition
        .destinations
        .iter()
        .enumerate()
        .map(|(slot, digest)| (digest.id, slot))
        .collect();
    let mut expected: Vec<RoaringBitmap> =
        vec![RoaringBitmap::new(); transition.destinations.len()];
    let mut expected_total = 0u64;

    for digest in transition.sources.iter() {
        // AUTHORITY: the current manifest fragment, not the group copy.
        let Some(frag) = manifest_by_id.get(&digest.id) else {
            return Err(Error::invalid_input(format!(
                "source fragment {} is no longer in the current dataset",
                digest.id
            )));
        };
        let mut offsets: Vec<u32> = match &frag.deletion_file {
            None => Vec::new(),
            Some(deletion) => {
                crate::io::deletion::read_dataset_deletion_file(dataset, frag.id, deletion)
                    .await?
                    .iter()
                    .collect()
            }
        };
        // Sorted for block locality in the stable-partition reads.
        offsets.sort_unstable();
        let mut rewrite_time_deleted = 0u64;
        for chunk in offsets.chunks(FOLDING_BATCH_ROWS) {
            let mut addresses = Vec::with_capacity(chunk.len());
            for &offset in chunk {
                if u64::from(offset) >= digest.physical_rows {
                    return Err(Error::invalid_input(format!(
                        "source fragment {} has a deleted row offset {offset} beyond its \
                         {} physical rows",
                        frag.id, digest.physical_rows
                    )));
                }
                addresses.push((digest.id << 32) | u64::from(offset));
            }
            for translation in mapping.remap_row_ids(&addresses).await? {
                match translation {
                    None => rewrite_time_deleted += 1,
                    Some(address) => {
                        let slot = destination_slot.get(&(address >> 32)).ok_or_else(|| {
                            Error::invalid_input(format!(
                                "the mapping translates to fragment {} which is not a \
                                     destination of this transition",
                                address >> 32
                            ))
                        })?;
                        expected[*slot].insert(address as u32);
                        expected_total += 1;
                    }
                }
            }
        }
        // Equation 1: rows dead at rewrite time, exactly the digest's count.
        if rewrite_time_deleted != digest.num_deleted_rows {
            return Err(Error::invalid_input(format!(
                "scan-time deletion accounting failed for source fragment {}: \
                 {rewrite_time_deleted} deleted rows predate the rewrite mapping but the \
                 digest records {} rewrite-time deletions",
                digest.id, digest.num_deleted_rows
            )));
        }
    }

    // The supplied destination deletion vectors, per destination.
    let mut supplied: Vec<RoaringBitmap> = Vec::with_capacity(group.new_fragments.len());
    let mut supplied_total = 0u64;
    for frag in group.new_fragments.iter() {
        let bitmap: RoaringBitmap = match &frag.deletion_file {
            None => RoaringBitmap::new(),
            Some(deletion) => {
                crate::io::deletion::read_dataset_deletion_file(dataset, frag.id, deletion)
                    .await?
                    .iter()
                    .collect()
            }
        };
        supplied_total += bitmap.len();
        supplied.push(bitmap);
    }
    // Equation 2: cardinality, exactly the summed deltas.
    let expected_delta: u64 = source_deltas.iter().sum();
    if supplied_total != expected_delta {
        return Err(Error::invalid_input(format!(
            "destination deletion vectors hold {supplied_total} rows but the sources \
             gained {expected_delta} deletions since the scan"
        )));
    }
    debug_assert_eq!(expected_total, expected_delta);
    // Equation 3: positions, per destination, as sets.
    for (slot, digest) in transition.destinations.iter().enumerate() {
        if expected[slot] != supplied[slot] {
            let missing = (&expected[slot] - &supplied[slot]).iter().next();
            let extra = (&supplied[slot] - &expected[slot]).iter().next();
            return Err(Error::invalid_input(format!(
                "folded deletions do not land on the translated positions for destination \
                 fragment {}: missing translated offset {missing:?}, unexpected deletion \
                 offset {extra:?}",
                digest.id
            )));
        }
    }
    Ok(())
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
    let manifest_by_id: HashMap<u64, &Fragment> = dataset
        .fragments()
        .iter()
        .map(|frag| (frag.id, frag))
        .collect();

    for (group, transition) in covered_groups.iter().zip(transitions.iter()) {
        if group.old_fragments.len() != transition.sources.len() {
            return Err(Error::invalid_input(format!(
                "a transition lists {} sources but its rewrite group holds {} old fragments",
                transition.sources.len(),
                group.old_fragments.len()
            )));
        }
        // Regime detection: the per-source delta between the scan-time
        // digest and the CURRENT manifest. All deltas zero -> regime A,
        // exactly the historical validation (metadata-only here, except the
        // rare read of a deletion vector whose count was never
        // materialized). Any delta positive -> the folding regime: a concurrent delete landed on a
        // source after the job snapshotted it, and the job answered by
        // folding those rows into destination deletion vectors instead of
        // recomputing; the fold is then validated row by row against the
        // transition's own row map (see `validate_folded_deletions`). A
        // source missing from the manifest contributes no delta: that
        // commit fails later exactly as before (double consumption or
        // fragment liveness).
        let mut source_deltas: Vec<u64> = Vec::with_capacity(transition.sources.len());
        for digest in transition.sources.iter() {
            let current = match manifest_by_id.get(&digest.id) {
                None => digest.num_deleted_rows,
                Some(frag) => current_deleted_rows(dataset, frag).await?,
            };
            if current < digest.num_deleted_rows {
                return Err(Error::invalid_input(format!(
                    "source fragment {} records {current} deleted rows in the manifest, below \
                     the {} its scan-time digest carries; a deletion vector cannot shrink",
                    digest.id, digest.num_deleted_rows
                )));
            }
            source_deltas.push(current - digest.num_deleted_rows);
        }
        let folding = source_deltas.iter().any(|&delta| delta > 0);

        for (source_index, (frag, digest)) in group
            .old_fragments
            .iter()
            .zip(transition.sources.iter())
            .enumerate()
        {
            // Materializing a data overlay breaks the reuse premise that a
            // rewrite moves addresses, never values: the destination holds
            // the overlaid values physically (and carries no overlay for
            // the runtime staleness guard to see), while an index built
            // before the overlay keeps the source in its bitmap as
            // provenance and would serve the destination through
            // translation. v0 handles this by dropping the DESTINATION ids
            // from stale bitmaps, which is a no-op here because tagged
            // provenance never contains them. Until provenance-side
            // invalidation exists, refuse to record a transition over
            // overlaid sources.
            if !frag.overlays.is_empty() {
                return Err(Error::not_supported(format!(
                    "source fragment {} carries data overlay files; a rewrite recording \
                     fragment reuse transitions would materialize the overlaid values while \
                     indices keep translated coverage over the old addresses. Compact the \
                     overlays away or rebuild the covering indices eagerly before this \
                     rewrite",
                    frag.id
                )));
            }
            let physical_rows = frag.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "source fragment {} has no physical row count",
                    frag.id
                ))
            })? as u64;
            let num_deleted_rows = if folding {
                // One authoritative accessor in the folding regime, so an
                // unmaterialized count cannot be classified one way and
                // validated another.
                current_deleted_rows(dataset, frag).await?
            } else {
                frag.deletion_file
                    .as_ref()
                    .and_then(|deletion| deletion.num_deleted_rows)
                    .unwrap_or(0) as u64
            };
            if !folding {
                // Regime A: exactly the historical binding.
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
            } else {
                // Regime B: the digest keeps the scan-time snapshot; the
                // group must carry the source's CURRENT metadata.
                if digest.id != frag.id || digest.physical_rows != physical_rows {
                    return Err(Error::invalid_input(format!(
                        "transition source digest {:?} does not match old fragment {} \
                         ({physical_rows} physical rows)",
                        digest, frag.id
                    )));
                }
                let current = digest.num_deleted_rows + source_deltas[source_index];
                if num_deleted_rows != current {
                    return Err(Error::invalid_input(format!(
                        "deletion folding requires the rewrite group to carry the source's \
                         current metadata: fragment {} shows {num_deleted_rows} deleted rows \
                         but the manifest records {current}",
                        frag.id
                    )));
                }
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
            // liveness and later fail (or falsely pass) translation. Under
            // the folding regime destinations MAY carry deletion vectors --
            // they hold exactly the delta rows, validated position by
            // position below.
            if !folding && frag.deletion_file.is_some() {
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
        if folding {
            validate_folded_deletions(dataset, group, transition, &manifest_by_id, &source_deltas)
                .await?;
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

    let entry = build_tagged_frag_reuse_entry(dataset, content, fragment_bitmap).await?;
    Ok((entry, base_entry_version))
}

/// Package assembled tagged FRI content bytes into a fresh manifest entry,
/// spilling to an external details file above the inline threshold. The
/// content must already be validated (a decodable ledger); this only encodes.
pub(crate) async fn build_tagged_frag_reuse_entry(
    dataset: &Dataset,
    content: Vec<u8>,
    fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    let index_id = Uuid::new_v4();
    let details_value = encode_tagged_frag_reuse_details(
        &dataset.object_store,
        dataset.indices_dir(),
        index_id,
        content,
    )
    .await?;

    Ok(IndexMetadata {
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
    })
}

/// Encode validated tagged FRI content bytes as an `index_details` value,
/// spilling to `<indices_dir>/<index_id>/details.binpb` above the inline
/// threshold.
async fn encode_tagged_frag_reuse_details(
    object_store: &ObjectStore,
    indices_dir: Path,
    index_id: Uuid,
    content: Vec<u8>,
) -> lance_core::Result<Vec<u8>> {
    if content.len() > FRAG_REUSE_INLINE_DETAILS_LIMIT {
        let file_path = indices_dir
            .join(index_id.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME);
        let mut writer = object_store.create(&file_path).await?;
        writer.write_all(&content).await?;
        writer.shutdown().await?;
        let external_file = ExternalFile {
            path: FRAG_REUSE_DETAILS_FILE_NAME.to_owned(),
            offset: 0,
            size: content.len() as u64,
        };
        Ok(encode_length_delimited_field(
            2,
            &external_file.encode_to_vec(),
        ))
    } else {
        Ok(encode_length_delimited_field(1, &content))
    }
}

/// Details above this size spill to an external `details.binpb` file.
pub(crate) const FRAG_REUSE_INLINE_DETAILS_LIMIT: usize = 204800;

/// `InlineContent.Transition.stable_partition` (proto field 4).
const STABLE_PARTITION_FIELD: u32 = 4;
/// `StablePartition.base_id` (proto field 3, varint).
const STABLE_PARTITION_BASE_ID_FIELD: u32 = 3;

/// Rewrite each stable-partition mapping's `base_id` through `remap`, leaving
/// every other byte in its exact wire form. The rewrite is a surgical
/// wire-level patch (no protobuf decode/re-encode), so unknown NESTED fields
/// -- extensions of `StablePartition` or `FragmentDigest` this writer does
/// not know, exactly the shape `base_id` itself was added in -- survive
/// byte-identically instead of being silently stripped. An unknown envelope
/// record is still refused: it may participate in base resolution in ways
/// this writer cannot see, so a history it cannot fully parse must not be
/// relocated (mirrors the trim's obligation). Unknown fields INSIDE a
/// transition are refused earlier, by the ledger gate the caller runs.
fn relocate_stable_partition_bases(
    content: &[u8],
    remap: impl Fn(Option<u32>) -> lance_core::Result<Option<u32>>,
) -> lance_core::Result<Vec<u8>> {
    use bytes::Buf;
    use prost::encoding::{WireType, decode_key, decode_varint};

    let corrupt = |message: String| Error::corrupt_file_named("FRI details", message);
    let total = content.len();
    let mut buf = bytes::Bytes::copy_from_slice(content);
    let mut output = Vec::with_capacity(content.len());
    while buf.has_remaining() {
        let start = total - buf.remaining();
        let (tag, wire_type) = decode_key(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        if wire_type != WireType::LengthDelimited || !matches!(tag, 1 | 2) {
            return Err(Error::not_supported(format!(
                "the tagged FRI history carries an unknown record (field {tag}); \
                 upgrade to a newer version of Lance before cloning this table"
            )));
        }
        let length = decode_varint(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        if length > buf.remaining() as u64 {
            return Err(corrupt(format!(
                "field {tag} length {length} exceeds remaining {} bytes",
                buf.remaining()
            )));
        }
        let payload = buf.split_to(length as usize);
        let end = total - buf.remaining();
        if tag == 2
            && let Some(patched) = patch_transition_stable_partition(&payload, &remap)?
        {
            output.extend_from_slice(&encode_length_delimited_field(2, &patched));
        } else {
            // Legacy versions and transitions without a stable-partition
            // mapping (ordered compaction) reference no bases; verbatim.
            output.extend_from_slice(&content[start..end]);
        }
    }
    Ok(output)
}

/// Patch a transition's stable-partition submessage in place, copying every
/// other field's bytes verbatim (whatever their field number or wire type).
/// Returns `None` when the transition carries no stable-partition mapping.
fn patch_transition_stable_partition(
    raw: &[u8],
    remap: &impl Fn(Option<u32>) -> lance_core::Result<Option<u32>>,
) -> lance_core::Result<Option<Vec<u8>>> {
    use bytes::Buf;
    use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};

    let corrupt = |message: String| Error::corrupt_file_named("FRI details", message);
    let total = raw.len();
    let mut buf = bytes::Bytes::copy_from_slice(raw);
    let mut output = Vec::with_capacity(raw.len() + 8);
    let mut found = false;
    while buf.has_remaining() {
        let start = total - buf.remaining();
        let (tag, wire_type) = decode_key(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        if tag == STABLE_PARTITION_FIELD {
            if wire_type != WireType::LengthDelimited {
                return Err(corrupt("stable partition mapping must be a message".into()));
            }
            let length = decode_varint(&mut buf).map_err(|e| corrupt(e.to_string()))?;
            if length > buf.remaining() as u64 {
                return Err(corrupt(format!(
                    "stable partition length {length} exceeds remaining {} bytes",
                    buf.remaining()
                )));
            }
            let payload = buf.split_to(length as usize);
            if found {
                // The ledger gate rejects this before relocation runs.
                return Err(corrupt(
                    "transition contains multiple stable-partition mappings".into(),
                ));
            }
            found = true;
            let patched = patch_stable_partition_base(&payload, remap)?;
            output.extend_from_slice(&encode_length_delimited_field(
                STABLE_PARTITION_FIELD,
                &patched,
            ));
        } else {
            if wire_type == WireType::LengthDelimited {
                let length = decode_varint(&mut buf).map_err(|e| corrupt(e.to_string()))?;
                if length > buf.remaining() as u64 {
                    return Err(corrupt(format!(
                        "field {tag} length {length} exceeds remaining {} bytes",
                        buf.remaining()
                    )));
                }
                buf.advance(length as usize);
            } else {
                skip_field(wire_type, tag, &mut buf, DecodeContext::default())
                    .map_err(|e| corrupt(e.to_string()))?;
            }
            let end = total - buf.remaining();
            output.extend_from_slice(&raw[start..end]);
        }
    }
    Ok(found.then_some(output))
}

/// Drop the submessage's `base_id` field wherever it occurs (proto3
/// last-one-wins yields the effective current value) and re-emit the
/// remapped value at the end; every other byte is preserved verbatim.
fn patch_stable_partition_base(
    raw: &[u8],
    remap: &impl Fn(Option<u32>) -> lance_core::Result<Option<u32>>,
) -> lance_core::Result<Vec<u8>> {
    use bytes::Buf;
    use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};

    let corrupt = |message: String| Error::corrupt_file_named("FRI details", message);
    let total = raw.len();
    let mut buf = bytes::Bytes::copy_from_slice(raw);
    let mut output = Vec::with_capacity(raw.len() + 8);
    let mut existing = None;
    while buf.has_remaining() {
        let start = total - buf.remaining();
        let (tag, wire_type) = decode_key(&mut buf).map_err(|e| corrupt(e.to_string()))?;
        if tag == STABLE_PARTITION_BASE_ID_FIELD {
            if wire_type != WireType::Varint {
                return Err(corrupt("stable partition base_id must be a varint".into()));
            }
            let value = decode_varint(&mut buf).map_err(|e| corrupt(e.to_string()))?;
            existing =
                Some(u32::try_from(value).map_err(|_| {
                    corrupt(format!("stable partition base_id {value} overflows u32"))
                })?);
        } else {
            if wire_type == WireType::LengthDelimited {
                let length = decode_varint(&mut buf).map_err(|e| corrupt(e.to_string()))?;
                if length > buf.remaining() as u64 {
                    return Err(corrupt(format!(
                        "field {tag} length {length} exceeds remaining {} bytes",
                        buf.remaining()
                    )));
                }
                buf.advance(length as usize);
            } else {
                skip_field(wire_type, tag, &mut buf, DecodeContext::default())
                    .map_err(|e| corrupt(e.to_string()))?;
            }
            let end = total - buf.remaining();
            output.extend_from_slice(&raw[start..end]);
        }
    }
    if let Some(base_id) = remap(existing)? {
        prost::encoding::encode_key(
            STABLE_PARTITION_BASE_ID_FIELD,
            WireType::Varint,
            &mut output,
        );
        prost::encoding::encode_varint(u64::from(base_id), &mut output);
    }
    Ok(output)
}

/// Relocate a tagged FRI entry for a shallow clone.
///
/// The source entry's content is resolved base-aware (a chained clone's entry
/// or details file may live in one of the source's own bases), its
/// stable-partition references are restamped through the clone's base mapping
/// (`Manifest::shallow_clone` carries the source's `base_paths` ids over
/// verbatim and adds `new_base_id` for the source's own base, so `None`
/// becomes `Some(new_base_id)` and `Some(id)` stays `id`), and the relocated
/// content is written into the CLONE under a fresh uuid with `base_id: None`.
/// Row-map files are NOT copied: the relocated references keep resolving them
/// from the bases they live in.
///
/// A history this writer cannot fully interpret (unsupported transitions or
/// unknown records) is refused with `NotSupported` before anything is
/// written, per the spec's writer obligation.
///
/// Returns the relocated entry and, when the details spilled to an external
/// file in the target, that file's path so a failed clone commit can delete
/// it again.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn relocate_tagged_entry_for_shallow_clone(
    source_store: &ObjectStore,
    source_base: &Path,
    source_manifest: &Manifest,
    store_registry: Arc<ObjectStoreRegistry>,
    entry: &IndexMetadata,
    new_base_id: u32,
    target_store: &ObjectStore,
    target_base: &Path,
) -> lance_core::Result<(IndexMetadata, Option<Path>)> {
    let details = entry
        .index_details
        .as_ref()
        .filter(|details| details.type_url.ends_with("FragmentReuseIndexDetails"))
        .ok_or_else(|| Error::index("Index details is not for the fragment reuse index"))?;
    // Base-aware resolution of the entry's external details without a live
    // `Dataset`: the same rule as `read_fri_external_file`, against the
    // source manifest's `base_paths`.
    let registry = store_registry.clone();
    let content = extract_raw_frag_reuse_content(details, |file| async move {
        let end = file
            .offset
            .checked_add(file.size)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                Error::corrupt_file_named("FRI details", "external FRI range overflow")
            })?;
        let (store, indices_dir) = match entry.base_id {
            None => (None, source_base.clone().join(crate::dataset::INDICES_DIR)),
            Some(id) => {
                let base_path = source_manifest.base_paths.get(&id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "base_path id {} not found for index {}",
                        id, entry.uuid
                    ))
                })?;
                let path = base_path.extract_path(registry.clone())?;
                let dir = if base_path.is_dataset_root {
                    path.join(crate::dataset::INDICES_DIR)
                } else {
                    path
                };
                // Foreign bases are opened with default store params:
                // per-base source credentials are not plumbed through the
                // clone commit path. Documented limitation shared with deep
                // clone's base reads (see
                // <https://github.com/lance-format/lance/issues/6093>).
                let (store, _) = ObjectStore::from_uri_and_params(
                    registry,
                    &base_path.path,
                    &Default::default(),
                )
                .await?;
                (Some(store), dir)
            }
        };
        let path = indices_dir
            .join(entry.uuid.to_string())
            .join(file.path.as_str());
        let store = store.as_deref().unwrap_or(source_store);
        store
            .open(&path)
            .await?
            .get_range(file.offset as usize..end)
            .await
            .map_err(Error::from)
    })
    .await?;

    // Full ledger validation before interpreting anything; reader semantics
    // skip unknown mappings, and a writer must not relocate a history it
    // cannot fully interpret.
    let ledger = decode_frag_reuse_ledger_from_content(entry.index_version, &content).await?;
    if ledger.has_unsupported_transitions() {
        return Err(Error::not_supported(
            "the fragment reuse history contains mappings this writer cannot relocate; \
             upgrade to a newer version of Lance before cloning this table",
        ));
    }

    let relocated = relocate_stable_partition_bases(&content, |base_id| match base_id {
        // The source's own base gets the id the clone assigned to it.
        None => Ok(Some(new_base_id)),
        // The clone carries the source's `base_paths` entries over under the
        // same ids, so a foreign-base reference keeps its id; it must exist.
        Some(id) => {
            if source_manifest.base_paths.contains_key(&id) {
                Ok(Some(id))
            } else {
                Err(Error::corrupt_file_named(
                    "FRI details",
                    format!("stable partition mapping references missing base {id}"),
                ))
            }
        }
    })?;
    // The relocated history must still decode as a well-formed ledger.
    decode_frag_reuse_ledger_from_content(entry.index_version, &relocated).await?;

    let index_id = Uuid::new_v4();
    let target_indices_dir = target_base.clone().join(crate::dataset::INDICES_DIR);
    let spilled = (relocated.len() > FRAG_REUSE_INLINE_DETAILS_LIMIT).then(|| {
        target_indices_dir
            .clone()
            .join(index_id.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME)
    });
    let details_value =
        encode_tagged_frag_reuse_details(target_store, target_indices_dir, index_id, relocated)
            .await?;
    Ok((
        IndexMetadata {
            uuid: index_id,
            name: entry.name.clone(),
            fields: entry.fields.clone(),
            covering_fields: entry.covering_fields.clone(),
            dataset_version: entry.dataset_version,
            fragment_bitmap: entry.fragment_bitmap.clone(),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "/lance.table.FragmentReuseIndexDetails".into(),
                value: details_value,
            })),
            index_version: entry.index_version,
            created_at: entry.created_at,
            // The relocated entry lives in the clone.
            base_id: None,
            files: entry.files.clone(),
        },
        spilled,
    ))
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
        // A v0 entry is carried exactly as before relocation existed: same
        // uuid and details bytes, base-stamped to the source, never lifted.
        assert_eq!(cloned_entry.index_version, 0);
        assert_eq!(cloned_entry.uuid, source_entry.uuid);
        assert_eq!(cloned_entry.index_details, source_entry.index_details);
        assert_eq!(cloned_entry.base_id, Some(0));

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

    /// Shallow-cloning tagged histories. All fixtures are on-disk: base-path
    /// resolution must reach the SOURCE dataset's store, which `memory://`
    /// fixtures cannot demonstrate (each URI is its own store).
    mod shallow_clone {
        use super::*;
        use futures::TryStreamExt;

        /// An on-disk two-fragment dataset with a committed scalar index.
        async fn disk_fixture(uri: &str) -> Dataset {
            let mut dataset = lance_datagen::gen_batch()
                .col("i", lance_datagen::array::step::<Int32Type>())
                .into_dataset(
                    uri,
                    crate::utils::test::FragmentCount::from(2),
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

        /// Commit one stable-partition rewrite over `source_ids`, writing its
        /// row map into the dataset's own `_fri/`.
        async fn commit_stable_partition(
            dataset: Dataset,
            source_ids: &[u64],
            dest_base_id: u64,
        ) -> Dataset {
            let old_fragments: Vec<Fragment> = source_ids
                .iter()
                .map(|id| {
                    dataset
                        .fragments()
                        .iter()
                        .find(|f| f.id == *id)
                        .unwrap()
                        .clone()
                })
                .collect();
            let (transition, destinations) =
                reader_tests::prepare_partition(&dataset, source_ids, dest_base_id).await;
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

        /// The `_fri/<map_id>/` directories under the dataset's own base.
        async fn list_fri_map_dirs(dataset: &Dataset) -> Vec<String> {
            let prefix = dataset.base.clone().join("_fri");
            let mut ids = HashSet::new();
            let mut stream = dataset.object_store.read_dir_all(&prefix, None);
            loop {
                match stream.try_next().await {
                    Ok(Some(meta)) => {
                        let relative = meta
                            .location
                            .as_ref()
                            .strip_prefix(prefix.as_ref())
                            .unwrap()
                            .trim_start_matches('/');
                        ids.insert(relative.split('/').next().unwrap().to_string());
                    }
                    Ok(None) => break,
                    Err(Error::NotFound { .. }) => break,
                    Err(e) => panic!("{e}"),
                }
            }
            let mut ids: Vec<String> = ids.into_iter().collect();
            ids.sort();
            ids
        }

        /// The stable-partition `base_id`s of the entry's transitions, in
        /// lineage order.
        fn stable_partition_bases(ledger: &FragReuseLedger) -> Vec<Option<u32>> {
            ledger
                .transitions()
                .iter()
                .filter_map(|transition| match transition.mapping() {
                    Mapping::StablePartition(partition) => Some(partition.base_id),
                    _ => None,
                })
                .collect()
        }

        /// Commit a v0 entry big enough (>200KB) that its lifted tagged form
        /// spills details to an external file. Returns the v0 content bytes.
        async fn commit_big_v0_entry(dataset: &mut Dataset) -> Vec<u8> {
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
            let entry = build_frag_reuse_index_metadata(dataset, None, details, bitmap)
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
            let stored = crate::index::load_all_indices(dataset).await.unwrap();
            load_raw_frag_reuse_content(dataset, &stored_fri(&stored))
                .await
                .unwrap()
        }

        /// Test 1: cloning a tagged table relocates the row-map references
        /// into the clone's base mapping instead of rejecting. Queries on the
        /// clone translate through row maps read from the SOURCE's `_fri/`.
        #[rstest::rstest]
        #[case::inline(false)]
        #[case::external(true)]
        #[tokio::test]
        async fn clone_relocates_row_map_references(#[case] external: bool) {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_uri = format!("{}/clone", clone_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            let v0_content = if external {
                commit_big_v0_entry(&mut dataset).await
            } else {
                Vec::new()
            };
            reserve_fragments(&mut dataset, 20).await;
            let before = sorted_values(&dataset).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            let stored = crate::index::load_all_indices(&dataset).await.unwrap();
            let source_entry = stored_fri(&stored);
            assert_eq!(source_entry.index_version, 1);
            let source_content = load_raw_frag_reuse_content(&dataset, &source_entry)
                .await
                .unwrap();
            assert_eq!(source_content.len() > 204800, external);

            let version = dataset.manifest.version;
            let clone = dataset
                .shallow_clone(clone_uri.as_str(), version, None)
                .await
                .unwrap();

            // The relocated entry lives in the clone under a fresh uuid; the
            // rest of its metadata is carried over.
            let stored = crate::index::load_all_indices(&clone).await.unwrap();
            let entry = stored_fri(&stored);
            assert!(is_tagged(&entry));
            assert_ne!(entry.uuid, source_entry.uuid);
            assert_eq!(entry.base_id, None);
            assert_eq!(entry.index_version, 1);
            assert_eq!(entry.fragment_bitmap, source_entry.fragment_bitmap);
            assert_eq!(entry.dataset_version, source_entry.dataset_version);
            let proto = FragmentReuseIndexDetails::decode(
                entry.index_details.as_ref().unwrap().value.as_slice(),
            )
            .unwrap();
            if external {
                // Spilled to the CLONE's own `_indices/<new uuid>/`.
                assert!(matches!(proto.content, Some(Content::External(_))));
            } else {
                assert!(matches!(proto.content, Some(Content::Inline(_))));
            }

            // The source's own base got the clone's freshly assigned id;
            // records without base references are carried verbatim.
            let clone_content = load_raw_frag_reuse_content(&clone, &entry).await.unwrap();
            assert!(clone_content.starts_with(&v0_content));
            assert_ne!(clone_content, source_content);
            let ledger = decode_entry(&clone, &entry).await;
            assert_eq!(ledger.transitions().len(), if external { 3501 } else { 1 });
            assert_eq!(stable_partition_bases(&ledger), vec![Some(0)]);
            assert!(clone.manifest.base_paths[&0].is_dataset_root);

            // The row-map files were NOT copied: the clone has no `_fri/` of
            // its own and reads the source's.
            assert!(list_fri_map_dirs(&clone).await.is_empty());
            assert_eq!(list_fri_map_dirs(&dataset).await.len(), 1);
            assert_eq!(sorted_values(&clone).await, before);
            assert_eq!(filtered_values(&clone, "i = 3").await, vec![3]);
            assert_eq!(filtered_values(&clone, "i >= 4").await, vec![4, 5, 6, 7]);
            // The source is untouched and still answers identically.
            assert_eq!(sorted_values(&dataset).await, before);
            assert_eq!(filtered_values(&dataset, "i = 3").await, vec![3]);
        }

        /// Test 2: a new stable-partition rewrite on the CLONE appends onto
        /// the relocated entry, mixing a source-base mapping with a
        /// clone-base one; both translation hops resolve correctly.
        #[tokio::test]
        async fn stable_partition_on_clone_mixes_bases() {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_uri = format!("{}/clone", clone_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            reserve_fragments(&mut dataset, 40).await;
            let before = sorted_values(&dataset).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            let source_map_dirs = list_fri_map_dirs(&dataset).await;

            let version = dataset.manifest.version;
            let mut clone = dataset
                .shallow_clone(clone_uri.as_str(), version, None)
                .await
                .unwrap();
            let stored = crate::index::load_all_indices(&clone).await.unwrap();
            let relocated_content = load_raw_frag_reuse_content(&clone, &stored_fri(&stored))
                .await
                .unwrap();

            reserve_fragments(&mut clone, 40).await;
            let clone = commit_stable_partition(clone, &[10, 11], 20).await;

            let stored = crate::index::load_all_indices(&clone).await.unwrap();
            let entry = stored_fri(&stored);
            assert_eq!(entry.base_id, None);
            // Appended onto the relocated entry, its bytes verbatim.
            let content = load_raw_frag_reuse_content(&clone, &entry).await.unwrap();
            assert!(content.starts_with(&relocated_content));
            let ledger = decode_entry(&clone, &entry).await;
            assert_eq!(ledger.transitions().len(), 2);
            assert_eq!(stable_partition_bases(&ledger), vec![Some(0), None]);

            // The new row map lives in the clone's own `_fri/`; the first
            // hop's map is still only in the source's.
            let clone_map_dirs = list_fri_map_dirs(&clone).await;
            assert_eq!(clone_map_dirs.len(), 1);
            assert!(!source_map_dirs.contains(&clone_map_dirs[0]));
            assert_eq!(list_fri_map_dirs(&dataset).await, source_map_dirs);

            // Queries translate through both hops (source-base then
            // clone-base row map).
            assert_eq!(sorted_values(&clone).await, before);
            assert_eq!(filtered_values(&clone, "i = 3").await, vec![3]);
            assert_eq!(filtered_values(&clone, "i >= 4").await, vec![4, 5, 6, 7]);
        }

        /// Test 3: clone of a clone. The first hop's mapping keeps its
        /// carried-over base id, the second hop's own-base mapping gets the
        /// freshly assigned one.
        #[tokio::test]
        async fn chained_clone_remaps_both_hops() {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone1_uri = format!("{}/clone1", clone_dir.as_str());
            let clone2_uri = format!("{}/clone2", clone_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            reserve_fragments(&mut dataset, 40).await;
            let before = sorted_values(&dataset).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;

            let version = dataset.manifest.version;
            let mut clone1 = dataset
                .shallow_clone(clone1_uri.as_str(), version, None)
                .await
                .unwrap();
            reserve_fragments(&mut clone1, 40).await;
            let mut clone1 = commit_stable_partition(clone1, &[10, 11], 20).await;

            let version = clone1.manifest.version;
            let clone2 = clone1
                .shallow_clone(clone2_uri.as_str(), version, None)
                .await
                .unwrap();

            // Both hops' bases are present under their expected ids.
            assert_eq!(clone2.manifest.base_paths.len(), 2);
            assert!(
                clone2.manifest.base_paths[&0]
                    .path
                    .ends_with(source_dir.as_str())
            );
            assert!(clone2.manifest.base_paths[&1].path.ends_with("clone1"));

            let stored = crate::index::load_all_indices(&clone2).await.unwrap();
            let entry = stored_fri(&stored);
            assert_eq!(entry.base_id, None);
            let ledger = decode_entry(&clone2, &entry).await;
            assert_eq!(stable_partition_bases(&ledger), vec![Some(0), Some(1)]);

            // Nothing was copied; both row maps are read from their homes.
            assert!(list_fri_map_dirs(&clone2).await.is_empty());
            assert_eq!(sorted_values(&clone2).await, before);
            assert_eq!(filtered_values(&clone2, "i = 3").await, vec![3]);
            assert_eq!(filtered_values(&clone2, "i >= 4").await, vec![4, 5, 6, 7]);
        }

        /// Test 4a: trim on the clone splices retained records verbatim, so a
        /// retained source-base mapping keeps its stamped base id while a
        /// drained record is dropped.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn trim_on_clone_preserves_relocated_bases() {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_uri = format!("{}/clone", clone_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;

            // A v0 legacy version over synthetic fragments: after the clone
            // it is trimmable (disjoint from every index) while the
            // stable-partition transition is still pinned by `i_idx`.
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
                details,
                RoaringBitmap::from_iter([110u32]),
            )
            .await
            .unwrap();
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

            reserve_fragments(&mut dataset, 20).await;
            let before = sorted_values(&dataset).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;

            let version = dataset.manifest.version;
            let mut clone = dataset
                .shallow_clone(clone_uri.as_str(), version, None)
                .await
                .unwrap();
            let stored = crate::index::load_all_indices(&clone).await.unwrap();
            let pre_trim_entry = stored_fri(&stored);
            let pre_trim_content = load_raw_frag_reuse_content(&clone, &pre_trim_entry)
                .await
                .unwrap();

            crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut clone)
                .await
                .unwrap();
            let stored = crate::index::load_all_indices(&clone).await.unwrap();
            let entry = stored_fri(&stored);
            assert_ne!(entry.uuid, pre_trim_entry.uuid);
            let trimmed_content = load_raw_frag_reuse_content(&clone, &entry).await.unwrap();
            // The legacy version is gone; the retained transition is the
            // exact byte suffix of the relocated entry, base stamp included.
            assert!(pre_trim_content.ends_with(&trimmed_content));
            assert!(trimmed_content.len() < pre_trim_content.len());
            let ledger = decode_entry(&clone, &entry).await;
            assert_eq!(ledger.transitions().len(), 1);
            assert_eq!(stable_partition_bases(&ledger), vec![Some(0)]);
            assert_eq!(sorted_values(&clone).await, before);
            assert_eq!(filtered_values(&clone, "i = 3").await, vec![3]);
        }

        /// Test 4b: the clone's `_fri/` garbage collection enumerates only
        /// its own directory, so draining and cleaning the clone never
        /// deletes the source's row-map files.
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn clone_fri_gc_spares_source_row_maps() {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_uri = format!("{}/clone", clone_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            reserve_fragments(&mut dataset, 40).await;
            let before = sorted_values(&dataset).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            let source_map_dirs = list_fri_map_dirs(&dataset).await;
            assert_eq!(source_map_dirs.len(), 1);

            let version = dataset.manifest.version;
            let mut clone = dataset
                .shallow_clone(clone_uri.as_str(), version, None)
                .await
                .unwrap();
            reserve_fragments(&mut clone, 40).await;
            let mut clone = commit_stable_partition(clone, &[10, 11], 20).await;
            assert_eq!(list_fri_map_dirs(&clone).await.len(), 1);

            // Drain: rebuild the index over the final destinations, trim the
            // entry away entirely.
            clone
                .create_index(
                    &["i"],
                    lance_index::IndexType::Scalar,
                    Some("i_idx".into()),
                    &lance_index::scalar::ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
            crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut clone)
                .await
                .unwrap();
            let stored = crate::index::load_all_indices(&clone).await.unwrap();
            assert!(!stored.iter().any(|idx| idx.name == FRAG_REUSE_INDEX_NAME));

            // Expire every old clone version and collect aggressively.
            crate::dataset::cleanup::cleanup_old_versions(
                &clone,
                crate::dataset::cleanup::CleanupPolicyBuilder::default()
                    .before_timestamp(
                        chrono::Utc::now() + chrono::TimeDelta::try_seconds(10).unwrap(),
                    )
                    .delete_unverified(true)
                    .error_if_tagged_old_versions(false)
                    .build(),
            )
            .await
            .unwrap();

            // The clone's own released map is collected; the source's map is
            // outside the clone's `_fri/` and survives untouched.
            assert!(list_fri_map_dirs(&clone).await.is_empty());
            assert_eq!(list_fri_map_dirs(&dataset).await, source_map_dirs);
            assert_eq!(sorted_values(&dataset).await, before);
            assert_eq!(filtered_values(&dataset, "i = 3").await, vec![3]);
            assert_eq!(sorted_values(&clone).await, before);
        }

        /// Test 5: a history this writer cannot fully interpret is not
        /// relocated: the clone is rejected whether the opacity sits inside a
        /// transition (unknown mapping fields) or beside the known records
        /// (unknown envelope field).
        #[rstest::rstest]
        #[case::unknown_transition_field(false)]
        #[case::unknown_envelope_record(true)]
        #[tokio::test]
        async fn clone_rejects_uninterpretable_history(#[case] envelope: bool) {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_uri = format!("{}/clone", clone_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            let (transition, destinations) = reader_tests::prepare(&dataset).await;
            let content = if envelope {
                let mut content = reader_tests::field(2, &transition.encode_to_vec());
                content.extend(reader_tests::field(3, b"future record"));
                content
            } else {
                let mut raw = transition.encode_to_vec();
                raw.extend(reader_tests::field(9, b"future mapping payload"));
                reader_tests::field(2, &raw)
            };
            reader_tests::install(&mut dataset, content, destinations, false).await;
            let indices = crate::index::load_all_indices(&dataset)
                .await
                .unwrap()
                .as_ref()
                .clone();
            reader_tests::persist_fixture(&mut dataset, indices).await;

            let version = dataset.manifest.version;
            let error = dataset
                .shallow_clone(clone_uri.as_str(), version, None)
                .await
                .unwrap_err();
            assert!(matches!(error, Error::NotSupported { .. }), "{error}");
            assert!(
                error.to_string().to_lowercase().contains("upgrade"),
                "{error}"
            );
            // Nothing was created at the target.
            assert!(Dataset::open(clone_uri.as_str()).await.is_err());
            assert_eq!(dataset.manifest.version, version);
        }

        /// The `_indices/<uuid>/` directories under the dataset's own base.
        async fn list_index_dirs(dataset: &Dataset) -> Vec<String> {
            let prefix = dataset.base.clone().join(crate::dataset::INDICES_DIR);
            let mut ids = HashSet::new();
            let mut stream = dataset.object_store.read_dir_all(&prefix, None);
            loop {
                match stream.try_next().await {
                    Ok(Some(meta)) => {
                        let relative = meta
                            .location
                            .as_ref()
                            .strip_prefix(prefix.as_ref())
                            .unwrap()
                            .trim_start_matches('/');
                        ids.insert(relative.split('/').next().unwrap().to_string());
                    }
                    Ok(None) => break,
                    Err(Error::NotFound { .. }) => break,
                    Err(e) => panic!("{e}"),
                }
            }
            let mut ids: Vec<String> = ids.into_iter().collect();
            ids.sort();
            ids
        }

        /// Owner-required regression: the PARENT's cleanup must not delete
        /// `_fri/` row maps a branch still translates through.
        ///
        /// 1. Tagged source (stable-partition mapping), `create_branch` a
        ///    child (the branch carries a relocated entry stamped to the
        ///    parent base).
        /// 2. The source rebuilds its index and trims its FRI history away.
        /// 3. The source's cleanup expires the older versions.
        /// 4. The child's indexed queries still succeed, opening the
        ///    source's row map. Retaining the branch-point manifest alone is
        ///    not enough: the map ids must be collected into the referenced
        ///    set explicitly.
        ///
        /// The control case runs the same drain without a branch: nothing
        /// references the map and cleanup collects it.
        #[rstest::rstest]
        #[case::branch_pins_the_map(true)]
        #[case::no_branch_releases_the_map(false)]
        #[tokio::test]
        #[serial_test::serial(frag_reuse_maintenance)]
        async fn parent_cleanup_spares_row_maps_a_branch_translates_through(
            #[case] with_branch: bool,
        ) {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let mut parent = disk_fixture(source_dir.as_str()).await;
            reserve_fragments(&mut parent, 20).await;
            let before = sorted_values(&parent).await;
            let source_ids: Vec<u64> = parent.fragments().iter().map(|f| f.id).collect();
            let mut parent = commit_stable_partition(parent, &source_ids, 10).await;
            let map_dirs = list_fri_map_dirs(&parent).await;
            assert_eq!(map_dirs.len(), 1);

            let child_uri = if with_branch {
                let version = parent.manifest.version;
                let child = parent.create_branch("child", version, None).await.unwrap();
                // The branch's relocated entry reads the parent's row map in
                // place, through the base stamped at branch creation.
                let stored = crate::index::load_all_indices(&child).await.unwrap();
                let entry = stored_fri(&stored);
                assert_eq!(entry.base_id, None);
                let ledger = decode_entry(&child, &entry).await;
                assert_eq!(stable_partition_bases(&ledger), vec![Some(0)]);
                Some(child.uri().to_string())
            } else {
                None
            };

            // The parent drains: index rebuilt over the destinations, tagged
            // history trimmed away entirely.
            parent
                .create_index(
                    &["i"],
                    lance_index::IndexType::Scalar,
                    Some("i_idx".into()),
                    &lance_index::scalar::ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
            crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut parent)
                .await
                .unwrap();
            let stored = crate::index::load_all_indices(&parent).await.unwrap();
            assert!(!stored.iter().any(|idx| idx.name == FRAG_REUSE_INDEX_NAME));

            // Expire every old parent version.
            crate::dataset::cleanup::cleanup_old_versions(
                &parent,
                crate::dataset::cleanup::CleanupPolicyBuilder::default()
                    .before_timestamp(
                        chrono::Utc::now() + chrono::TimeDelta::try_seconds(10).unwrap(),
                    )
                    .delete_unverified(true)
                    .error_if_tagged_old_versions(false)
                    .build(),
            )
            .await
            .unwrap();

            if let Some(child_uri) = child_uri {
                // The branch pins the row map, and the child (opened fresh,
                // no warm caches) still translates through it.
                assert_eq!(list_fri_map_dirs(&parent).await, map_dirs);
                let child = Dataset::open(child_uri.as_str()).await.unwrap();
                assert_eq!(sorted_values(&child).await, before);
                assert_eq!(filtered_values(&child, "i = 3").await, vec![3]);
                assert_eq!(filtered_values(&child, "i >= 4").await, vec![4, 5, 6, 7]);
            } else {
                // Control: nothing references the map; the same cleanup
                // collects it.
                assert!(list_fri_map_dirs(&parent).await.is_empty());
            }
        }

        /// The relocation must not silently strip nested unknown fields:
        /// `base_id` is patched at the wire level, so extensions inside the
        /// stable-partition submessage or a fragment digest (exactly the
        /// shape `base_id` itself was added in) survive byte-identically.
        #[tokio::test]
        async fn clone_preserves_unknown_nested_wire_fields() {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_uri = format!("{}/clone", clone_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            let before = sorted_values(&dataset).await;
            let (transition, destinations) = reader_tests::prepare(&dataset).await;

            // Re-encode the transition by hand, planting unknown subfields
            // inside the first source digest and the stable-partition
            // submessage.
            let Some(transition::Mapping::StablePartition(partition)) = &transition.mapping else {
                unreachable!()
            };
            let mut partition_bytes = partition.encode_to_vec();
            partition_bytes.extend(reader_tests::field(9, b"stable-partition-extension"));
            let build_transition = |partition_bytes: &[u8]| {
                let mut raw = Vec::new();
                for (position, digest) in transition.sources.iter().enumerate() {
                    let mut digest_bytes = digest.encode_to_vec();
                    if position == 0 {
                        digest_bytes.extend(reader_tests::field(9, b"digest-extension"));
                    }
                    raw.extend(reader_tests::field(1, &digest_bytes));
                }
                for digest in transition.destinations.iter() {
                    raw.extend(reader_tests::field(2, &digest.encode_to_vec()));
                }
                raw.extend(reader_tests::field(4, partition_bytes));
                reader_tests::field(2, &raw)
            };
            let content = build_transition(&partition_bytes);
            reader_tests::install(&mut dataset, content, destinations, false).await;
            let indices = crate::index::load_all_indices(&dataset)
                .await
                .unwrap()
                .as_ref()
                .clone();
            reader_tests::persist_fixture(&mut dataset, indices).await;

            let version = dataset.manifest.version;
            let clone = dataset
                .shallow_clone(clone_uri.as_str(), version, None)
                .await
                .unwrap();

            // Byte-exact expectation: everything identical except the
            // stable-partition submessage gains `base_id = 0` (field 3,
            // varint) appended after its preserved unknown field.
            let mut expected_partition = partition_bytes.clone();
            expected_partition.extend([0x18, 0x00]);
            let expected = build_transition(&expected_partition);
            let stored = crate::index::load_all_indices(&clone).await.unwrap();
            let entry = stored_fri(&stored);
            let clone_content = load_raw_frag_reuse_content(&clone, &entry).await.unwrap();
            assert_eq!(clone_content, expected);

            // The relocated history still resolves and translates.
            let ledger = decode_entry(&clone, &entry).await;
            assert_eq!(stable_partition_bases(&ledger), vec![Some(0)]);
            assert_eq!(sorted_values(&clone).await, before);
            assert_eq!(filtered_values(&clone, "i = 3").await, vec![3]);
        }

        /// Parity with `deep_clone`: shallow-cloning onto a URI where a
        /// dataset already lives fails before anything is written there, so
        /// a tagged clone cannot pollute a live dataset's `_indices/` with
        /// staged details.
        #[tokio::test]
        async fn clone_onto_existing_dataset_rejected() {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let target_dir = lance_core::utils::tempfile::TempStrDir::default();
            let target_uri = format!("{}/occupied", target_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            // External details: without the pre-check, the relocation would
            // stage a spilled details file into the occupied target.
            commit_big_v0_entry(&mut dataset).await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;

            let occupied = lance_datagen::gen_batch()
                .col("j", lance_datagen::array::step::<Int32Type>())
                .into_dataset(
                    target_uri.as_str(),
                    crate::utils::test::FragmentCount::from(1),
                    crate::utils::test::FragmentRowCount::from(4),
                )
                .await
                .unwrap();
            let occupied_version = occupied.manifest.version;
            assert!(list_index_dirs(&occupied).await.is_empty());

            let version = dataset.manifest.version;
            let error = dataset
                .shallow_clone(target_uri.as_str(), version, None)
                .await
                .unwrap_err();
            assert!(
                matches!(error, Error::DatasetAlreadyExists { .. }),
                "{error}"
            );
            // Nothing was staged in the occupied dataset.
            assert!(list_index_dirs(&occupied).await.is_empty());
            let reopened = Dataset::open(target_uri.as_str()).await.unwrap();
            assert_eq!(reopened.manifest.version, occupied_version);
        }

        /// The target preflight only lets a definitive "no dataset here"
        /// resolution permit the clone: any other resolver failure (storage,
        /// auth, corrupt listing) aborts before anything is written, instead
        /// of treating an uninspectable target as absent.
        #[tokio::test]
        async fn clone_preflight_propagates_resolver_errors() {
            use lance_table::io::commit::{
                CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme, ManifestWriter,
                RenameCommitHandler,
            };

            /// Delegates everything except latest-location resolution, which
            /// fails with a non-NotFound error for the poisoned target path.
            #[derive(Debug)]
            struct FailingResolver {
                inner: RenameCommitHandler,
                fail_suffix: &'static str,
            }

            #[async_trait::async_trait]
            impl CommitHandler for FailingResolver {
                async fn resolve_latest_location(
                    &self,
                    base_path: &Path,
                    object_store: &ObjectStore,
                ) -> lance_core::Result<ManifestLocation> {
                    if base_path.as_ref().ends_with(self.fail_suffix) {
                        return Err(Error::io("injected resolver outage"));
                    }
                    self.inner
                        .resolve_latest_location(base_path, object_store)
                        .await
                }

                async fn commit(
                    &self,
                    manifest: &mut Manifest,
                    indices: Option<Vec<IndexMetadata>>,
                    base_path: &Path,
                    object_store: &ObjectStore,
                    manifest_writer: ManifestWriter,
                    naming_scheme: ManifestNamingScheme,
                    transaction: Option<lance_table::format::Transaction>,
                ) -> std::result::Result<ManifestLocation, CommitError> {
                    self.inner
                        .commit(
                            manifest,
                            indices,
                            base_path,
                            object_store,
                            manifest_writer,
                            naming_scheme,
                            transaction,
                        )
                        .await
                }
            }

            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_uri = format!("{}/flaky-target", clone_dir.as_str());
            // Build the tagged source normally, then reopen it with the
            // failing resolver installed as its commit handler.
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let dataset = commit_stable_partition(dataset, &source_ids, 10).await;
            let version = dataset.manifest.version;
            drop(dataset);
            let mut dataset =
                crate::dataset::builder::DatasetBuilder::from_uri(source_dir.as_str())
                    .with_read_params(crate::dataset::ReadParams {
                        commit_handler: Some(Arc::new(FailingResolver {
                            inner: RenameCommitHandler,
                            fail_suffix: "flaky-target",
                        })),
                        ..Default::default()
                    })
                    .load()
                    .await
                    .unwrap();

            let error = dataset
                .shallow_clone(clone_uri.as_str(), version, None)
                .await
                .unwrap_err();
            assert!(matches!(error, Error::IO { .. }), "{error}");
            assert!(
                error.to_string().contains("injected resolver outage"),
                "{error}"
            );
            // The clone did not proceed: nothing was written to the target.
            assert!(!std::path::Path::new(clone_uri.as_str()).exists());
        }

        /// A clone commit that conclusively fails must delete the details
        /// file it staged in the target, alongside its transaction file.
        /// Driven through the internal commit path: the public
        /// `shallow_clone` rejects an occupied target before staging, so the
        /// racing case is simulated by committing the same clone twice.
        #[tokio::test]
        async fn failed_clone_commit_cleans_staged_details() {
            let source_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_dir = lance_core::utils::tempfile::TempStrDir::default();
            let clone_uri = format!("{}/clone", clone_dir.as_str());
            let mut dataset = disk_fixture(source_dir.as_str()).await;
            commit_big_v0_entry(&mut dataset).await;
            reserve_fragments(&mut dataset, 20).await;
            let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
            let mut dataset = commit_stable_partition(dataset, &source_ids, 10).await;

            let version = dataset.manifest.version;
            let clone = dataset
                .shallow_clone(clone_uri.as_str(), version, None)
                .await
                .unwrap();
            // Exactly the relocated entry's spilled details dir.
            let staged = list_index_dirs(&clone).await;
            assert_eq!(staged.len(), 1);

            let transaction = Transaction::new(
                version,
                Operation::Clone {
                    is_shallow: true,
                    ref_name: None,
                    ref_version: version,
                    ref_path: dataset.uri().to_string(),
                    branch_name: None,
                },
                None,
            );
            let error = crate::io::commit::commit_new_dataset(
                &dataset.object_store,
                None,
                dataset.commit_handler.as_ref(),
                &clone.base,
                &transaction,
                &Default::default(),
                clone.manifest_location.naming_scheme,
                dataset.metadata_cache.as_ref(),
                dataset.session.store_registry(),
            )
            .await
            .unwrap_err();
            assert!(
                matches!(error, Error::DatasetAlreadyExists { .. }),
                "{error}"
            );
            // The retry's staged details file was deleted again; only the
            // committed clone's remains.
            assert_eq!(list_index_dirs(&clone).await, staged);
        }
    }
    /// Round 8 (a): a stable-partition rewrite over a source carrying data
    /// overlay files is refused. Materialization would bake the overlaid
    /// values into the destination while indices keep translated coverage
    /// over the old addresses, and the destination carries no overlay for
    /// the runtime staleness guard to see.
    #[tokio::test]
    async fn overlaid_source_rejects_stable_partition_rewrite() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        // Attach an overlay to source fragment 0 by manifest surgery: how
        // the overlay got there is irrelevant to the rule under test.
        let mut fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
        fragments[0]
            .overlays
            .push(lance_table::format::overlay::DataOverlayFile {
                data_file: lance_table::format::DataFile::new_legacy_from_fields(
                    "overlay-0.lance",
                    vec![0],
                    None,
                ),
                coverage: lance_table::format::overlay::OverlayCoverage::dense(
                    RoaringBitmap::from_iter([0u32]),
                ),
                committed_version: dataset.manifest.version,
            });
        Arc::make_mut(&mut dataset.manifest).fragments = Arc::new(fragments);
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        reader_tests::persist_fixture(&mut dataset, indices).await;

        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let version = dataset.latest_version_id().await.unwrap();
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(Transaction::new(
                version,
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
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("overlay"), "{error}");
        assert_eq!(dataset.latest_version_id().await.unwrap(), version);
    }

    /// Round 8 (b): deferred compaction on a tagged table refuses a source
    /// carrying data overlay files instead of recording a transition. The
    /// task is hand-built the way a distributed driver would replay one.
    #[tokio::test]
    async fn overlaid_source_rejects_tagged_compaction() {
        let dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(
                crate::utils::test::FragmentCount::from(2),
                crate::utils::test::FragmentRowCount::from(4),
            )
            .await
            .unwrap();
        let schema = Arc::new(arrow_schema::Schema::from(dataset.schema()));
        let mut dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(100..104))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();
        let appended_id = dataset.fragments().last().unwrap().id;
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
        // Overlay on the appended fragment, attached before tagging (a
        // tagged table cannot accept new overlays through the gate).
        let mut fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
        fragments
            .iter_mut()
            .find(|f| f.id == appended_id)
            .unwrap()
            .overlays
            .push(lance_table::format::overlay::DataOverlayFile {
                data_file: lance_table::format::DataFile::new_legacy_from_fields(
                    "overlay-f.lance",
                    vec![0],
                    None,
                ),
                coverage: lance_table::format::overlay::OverlayCoverage::dense(
                    RoaringBitmap::from_iter([0u32]),
                ),
                committed_version: dataset.manifest.version,
            });
        Arc::make_mut(&mut dataset.manifest).fragments = Arc::new(fragments);
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        reader_tests::persist_fixture(&mut dataset, indices).await;

        // Tag through a rewrite of the clean fragments 0 and 1.
        reserve_fragments(&mut dataset, 40).await;
        let old_fragments: Vec<Fragment> = dataset
            .fragments()
            .iter()
            .filter(|f| f.id < 2)
            .cloned()
            .collect();
        let (transition, sp_destinations) =
            reader_tests::prepare_partition(&dataset, &[0, 1], 10).await;
        let read_version = dataset.manifest.version;
        let mut dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: sp_destinations,
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

        // A replayed deferred-compaction task over the overlaid fragment.
        let overlaid = dataset
            .fragments()
            .iter()
            .find(|f| f.id == appended_id)
            .unwrap()
            .clone();
        assert!(!overlaid.overlays.is_empty());
        let txn = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(vec![
                RecordBatch::try_new(
                    schema,
                    vec![Arc::new(Int32Array::from_iter_values(100..104))],
                )
                .unwrap(),
            ])
            .await
            .unwrap();
        let Operation::Append {
            fragments: new_fragments,
        } = txn.operation
        else {
            unreachable!()
        };
        let mut row_addrs = RoaringTreemap::new();
        for offset in 0..4u64 {
            row_addrs.insert((appended_id << 32) + offset);
        }
        let mut serialized = Vec::new();
        row_addrs.serialize_into(&mut serialized).unwrap();
        let task = crate::dataset::optimize::RewriteResult {
            metrics: Default::default(),
            new_fragments,
            read_version: dataset.manifest.version,
            original_fragments: vec![overlaid],
            row_addrs: Some(serialized),
        };
        let fragments_before: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
        let error = crate::dataset::optimize::commit_compaction(
            &mut dataset,
            vec![task],
            Arc::new(crate::dataset::optimize::IgnoreRemap::default()),
            &crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 100,
                defer_index_remap: true,
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("overlay"), "{error}");
        // No rewrite landed: the fragment list is untouched (only the
        // internal id reservation may have advanced the version).
        let fragments_after: Vec<u64> = dataset
            .checkout_version(dataset.latest_version_id().await.unwrap())
            .await
            .unwrap()
            .fragments()
            .iter()
            .map(|f| f.id)
            .collect();
        assert_eq!(fragments_after, fragments_before);
    }

    /// Post-scan-deletion folding harness: fixture, reserved ids, a prepared
    /// transition, then an optional delta delete AFTER the row map was
    /// built. `old_fragments` carries the CURRENT metadata, as the folding
    /// regime requires.
    struct FoldingHarness {
        dataset: Dataset,
        read_version: u64,
        old_fragments: Vec<Fragment>,
        destinations: Vec<Fragment>,
        transition: Transition,
    }

    async fn folding_harness(delta_delete: Option<&str>) -> FoldingHarness {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        if let Some(predicate) = delta_delete {
            dataset.delete(predicate).await.unwrap();
        }
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        FoldingHarness {
            read_version: dataset.manifest.version,
            dataset,
            old_fragments,
            destinations,
            transition,
        }
    }

    /// The job folds delta rows: writes a destination deletion vector for
    /// the given offsets, leaving the digest at its scan-time zero.
    async fn fold_destination(harness: &mut FoldingHarness, destination: usize, offsets: &[u32]) {
        let deletion_vector: lance_core::utils::deletion::DeletionVector =
            offsets.iter().copied().collect();
        let file = lance_table::io::deletion::write_deletion_file(
            &harness.dataset.base,
            harness.destinations[destination].id,
            harness.read_version,
            &deletion_vector,
            &harness.dataset.object_store,
        )
        .await
        .unwrap()
        .unwrap();
        harness.destinations[destination].deletion_file = Some(file);
    }

    async fn commit_fold(harness: FoldingHarness) -> lance_core::Result<Dataset> {
        crate::dataset::write::CommitBuilder::new(Arc::new(harness.dataset))
            .execute(Transaction::new(
                harness.read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: harness.old_fragments,
                        new_fragments: harness.destinations,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![harness.transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
    }

    /// Round 10 (1): the genuine concurrency sequence. The job snapshots
    /// at V and builds its row map; a concurrent delete commits V+1; the
    /// job's commit from V is rejected by the conflict resolver; the job
    /// refolds the delta into destination deletion vectors (the recompute
    /// that lives job-side) and re-commits from V+1, passing exact
    /// accounting. Scans and translated index queries both exclude the
    /// folded rows; the ledger digests keep the scan-time counts.
    #[tokio::test]
    async fn folded_delta_deletions_commit_and_read_correctly() {
        // Fixture values 0..8; fragment 0 holds 0..4. Parity partition:
        // destination 0 takes evens, destination 1 takes odds, in scan
        // order. Deleting values 1 and 2 after the scan translates to
        // destination 1 offset 0 and destination 0 offset 1.
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let job_read_version = dataset.manifest.version;
        let scan_time_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        dataset.delete("i = 1 OR i = 2").await.unwrap();

        // Committing from the job's snapshot is a conflict: the delete
        // touched a rewritten fragment.
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(Transaction::new(
                job_read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: scan_time_fragments,
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
            .unwrap_err();
        assert!(
            matches!(error, Error::RetryableCommitConflict { .. }),
            "{error}"
        );

        // Job-side retry: fold the delta and re-commit from the current
        // version.
        let mut harness = FoldingHarness {
            read_version: dataset.manifest.version,
            old_fragments: dataset.fragments().iter().cloned().collect(),
            destinations,
            transition,
            dataset,
        };
        fold_destination(&mut harness, 1, &[0]).await;
        fold_destination(&mut harness, 0, &[1]).await;
        let dataset = commit_fold(harness).await.unwrap();

        assert_eq!(sorted_values(&dataset).await, vec![0, 3, 4, 5, 6, 7]);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
        assert_eq!(filtered_values(&dataset, "i = 1").await, Vec::<i32>::new());
        assert_eq!(filtered_values(&dataset, "i = 2").await, Vec::<i32>::new());
        assert_eq!(filtered_values(&dataset, "i = 3").await, vec![3]);
        assert_eq!(filtered_values(&dataset, "i >= 4").await, vec![4, 5, 6, 7]);

        // Digest semantics unchanged: scan-time counts, zero-deletion
        // destinations, same row map.
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        let ledger = decode_entry(&dataset, &entry).await;
        assert_eq!(ledger.transitions().len(), 1);
        let recorded = &ledger.transitions()[0];
        assert!(recorded.sources().iter().all(|d| d.num_deleted_rows == 0));
        assert!(
            recorded
                .destinations()
                .iter()
                .all(|d| d.num_deleted_rows == 0)
        );
    }

    /// Round 10 (2): no delta means regime A, and regime A still rejects a
    /// destination carrying a deletion vector exactly as before.
    #[tokio::test]
    async fn regime_a_still_rejects_destination_deletion_files() {
        let mut harness = folding_harness(None).await;
        fold_destination(&mut harness, 0, &[0]).await;
        let version = harness.read_version;
        let dataset = harness.dataset.clone();
        let error = commit_fold(harness).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("carries a deletion file"),
            "{error}"
        );
        assert_eq!(dataset.latest_version_id().await.unwrap(), version);
    }

    /// Round 10 (3): folding only one of the two delta rows breaks the
    /// cardinality equation.
    #[tokio::test]
    async fn folded_count_mismatch_rejected() {
        let mut harness = folding_harness(Some("i = 1 OR i = 2")).await;
        fold_destination(&mut harness, 1, &[0]).await;
        let error = commit_fold(harness).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("hold 1 rows but the sources gained 2"),
            "{error}"
        );
    }

    /// Round 10 (4): right cardinality, one wrong offset -- one row wrongly
    /// dead plus one row resurrected -- breaks the set equation.
    #[tokio::test]
    async fn folded_wrong_position_rejected() {
        let mut harness = folding_harness(Some("i = 1 OR i = 2")).await;
        fold_destination(&mut harness, 1, &[0]).await;
        // Value 2 translates to destination 0 offset 1; offset 2 is wrong.
        fold_destination(&mut harness, 0, &[2]).await;
        let error = commit_fold(harness).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("translated positions"),
            "{error}"
        );
    }

    /// Round 10 (5a): a current deletion vector smaller than the scan-time
    /// digest is corrupt (deletion vectors cannot shrink).
    #[tokio::test]
    async fn shrunken_source_deletion_vector_rejected() {
        let mut dataset = reader_tests::fixture().await;
        dataset.delete("i = 0 OR i = 1").await.unwrap();
        reserve_fragments(&mut dataset, 30).await;
        let (transition, destinations) = reader_tests::prepare(&dataset).await;
        assert_eq!(transition.sources[0].num_deleted_rows, 2);
        // Corruption stand-in: replace the source deletion vector with a
        // smaller one.
        let deletion_vector: lance_core::utils::deletion::DeletionVector =
            [0u32].into_iter().collect();
        let file = lance_table::io::deletion::write_deletion_file(
            &dataset.base,
            0,
            dataset.manifest.version,
            &deletion_vector,
            &dataset.object_store,
        )
        .await
        .unwrap()
        .unwrap();
        let mut fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
        fragments[0].deletion_file = Some(file);
        Arc::make_mut(&mut dataset.manifest).fragments = Arc::new(fragments);
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        reader_tests::persist_fixture(&mut dataset, indices).await;

        let harness = FoldingHarness {
            read_version: dataset.latest_version_id().await.unwrap(),
            old_fragments: dataset.fragments().iter().cloned().collect(),
            destinations,
            transition,
            dataset,
        };
        let error = commit_fold(harness).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("cannot shrink"), "{error}");
    }

    /// Round 10 (5b): the null-label accounting is an exact equality. A
    /// current deletion vector that gained rows but lost a scan-time
    /// deletion has fewer null-labeled rows than the digest records, even
    /// though its total makes the delta look positive; the same equality is
    /// what forbids a delta row from ever carrying a null label.
    #[tokio::test]
    async fn scan_time_deletion_accounting_rejected() {
        let mut dataset = reader_tests::fixture().await;
        // Scan-time deletion: value 0 (fragment 0, offset 0) is a null
        // label in the row map and counts 1 in the digest.
        dataset.delete("i = 0").await.unwrap();
        reserve_fragments(&mut dataset, 30).await;
        let (transition, mut destinations) = reader_tests::prepare(&dataset).await;
        assert_eq!(transition.sources[0].num_deleted_rows, 1);
        // Corrupted current vector: drops the scan-time offset 0, adds
        // offsets 1 and 2 (count 2, delta +1, so regime B engages).
        let deletion_vector: lance_core::utils::deletion::DeletionVector =
            [1u32, 2].into_iter().collect();
        let file = lance_table::io::deletion::write_deletion_file(
            &dataset.base,
            0,
            dataset.manifest.version,
            &deletion_vector,
            &dataset.object_store,
        )
        .await
        .unwrap()
        .unwrap();
        let mut fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
        fragments[0].deletion_file = Some(file);
        Arc::make_mut(&mut dataset.manifest).fragments = Arc::new(fragments);
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        reader_tests::persist_fixture(&mut dataset, indices).await;

        let read_version = dataset.latest_version_id().await.unwrap();
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        // Fold something plausible for the +1 delta so the accounting
        // equation, not the cardinality one, is what fires.
        let deletion_vector: lance_core::utils::deletion::DeletionVector =
            [0u32].into_iter().collect();
        let file = lance_table::io::deletion::write_deletion_file(
            &dataset.base,
            destinations[1].id,
            read_version,
            &deletion_vector,
            &dataset.object_store,
        )
        .await
        .unwrap()
        .unwrap();
        destinations[1].deletion_file = Some(file);

        let harness = FoldingHarness {
            read_version,
            old_fragments,
            destinations,
            transition,
            dataset,
        };
        let error = commit_fold(harness).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("scan-time deletion accounting"),
            "{error}"
        );
    }

    /// Copy the live rows of `source_ids` into one new fragment carrying
    /// `dest_id`, and build the ordered-compaction transition a tagged
    /// compaction would record for it (survivor bitmap over the sources'
    /// live addresses, digests as of NOW).
    async fn oc_job(dataset: &Dataset, source_ids: &[u64], dest_id: u64) -> (Transition, Fragment) {
        let source_fragments: Vec<Fragment> = source_ids
            .iter()
            .map(|id| {
                dataset
                    .fragments()
                    .iter()
                    .find(|f| f.id == *id)
                    .unwrap()
                    .clone()
            })
            .collect();
        let batch = {
            let mut scan = dataset.scan();
            scan.with_fragments(source_fragments.clone());
            scan.try_into_batch().await.unwrap()
        };
        let txn = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(vec![batch])
            .await
            .unwrap();
        let Operation::Append { mut fragments } = txn.operation else {
            unreachable!()
        };
        assert_eq!(fragments.len(), 1);
        fragments[0].id = dest_id;
        let destination = fragments.pop().unwrap();

        let mut survivors = RoaringTreemap::new();
        let mut sources = Vec::new();
        for frag in &source_fragments {
            let deleted: Option<RoaringBitmap> = dataset
                .get_fragment(frag.id as usize)
                .unwrap()
                .get_deletion_vector()
                .await
                .unwrap()
                .map(|v| v.iter().collect());
            let physical = frag.physical_rows.unwrap() as u64;
            for offset in 0..physical {
                if !deleted.as_ref().is_some_and(|d| d.contains(offset as u32)) {
                    survivors.insert((frag.id << 32) | offset);
                }
            }
            sources.push(FragmentDigest {
                id: frag.id,
                physical_rows: physical,
                num_deleted_rows: deleted.as_ref().map_or(0, |d| d.len()),
            });
        }
        let mut changed_row_addrs = Vec::new();
        survivors.serialize_into(&mut changed_row_addrs).unwrap();
        let transition = Transition {
            sources,
            destinations: vec![FragmentDigest {
                id: destination.id,
                physical_rows: destination.physical_rows.unwrap() as u64,
                num_deleted_rows: 0,
            }],
            mapping: Some(transition::Mapping::OrderedCompaction(
                lance_table::format::pb::fragment_reuse_index_details::OrderedCompaction {
                    changed_row_addrs,
                },
            )),
        };
        (transition, destination)
    }

    async fn write_dv(
        dataset: &Dataset,
        fragment: &mut Fragment,
        read_version: u64,
        offsets: &[u32],
    ) {
        let deletion_vector: lance_core::utils::deletion::DeletionVector =
            offsets.iter().copied().collect();
        let file = lance_table::io::deletion::write_deletion_file(
            &dataset.base,
            fragment.id,
            read_version,
            &deletion_vector,
            &dataset.object_store,
        )
        .await
        .unwrap()
        .unwrap();
        fragment.deletion_file = Some(file);
    }

    /// Round 10b (1): ordered-compaction folding through the genuine
    /// concurrency sequence. The compaction copies its sources at V, a
    /// concurrent delete commits V+1, the commit from V is rejected by the
    /// conflict resolver, and the job folds the delta into the destination
    /// deletion vector at its compaction-translated offsets and re-commits
    /// from V+1 -- exact accounting with no extra validation IO.
    #[tokio::test]
    async fn oc_folded_delta_deletions_commit_and_read_correctly() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let job_read_version = dataset.manifest.version;
        let scan_time_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        // Values 0..8 in fragments 0 and 1; the compaction output preserves
        // scan order, so value v lands at destination offset v.
        let (transition, mut destination) = oc_job(&dataset, &[0, 1], 20).await;
        dataset.delete("i = 1 OR i = 6").await.unwrap();

        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(Transaction::new(
                job_read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: scan_time_fragments,
                        new_fragments: vec![destination.clone()],
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
            .unwrap_err();
        assert!(
            matches!(error, Error::RetryableCommitConflict { .. }),
            "{error}"
        );

        let read_version = dataset.manifest.version;
        write_dv(&dataset, &mut destination, read_version, &[1, 6]).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: vec![destination],
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

        assert_eq!(sorted_values(&dataset).await, vec![0, 2, 3, 4, 5, 7]);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
        assert_eq!(filtered_values(&dataset, "i = 1").await, Vec::<i32>::new());
        assert_eq!(filtered_values(&dataset, "i = 6").await, Vec::<i32>::new());
        assert_eq!(filtered_values(&dataset, "i = 5").await, vec![5]);
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        let ledger = decode_entry(&dataset, &entry).await;
        assert_eq!(ledger.transitions().len(), 1);
        assert!(matches!(
            ledger.transitions()[0].mapping(),
            Mapping::OrderedCompaction(_)
        ));
        assert!(
            ledger.transitions()[0]
                .sources()
                .iter()
                .all(|d| d.num_deleted_rows == 0)
        );
    }

    /// Round 10b (2): no delta on an ordered-compaction rewrite keeps
    /// regime A, and the destination deletion vector stays rejected.
    #[tokio::test]
    async fn oc_regime_a_still_rejects_destination_deletion_files() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let (transition, mut destination) = oc_job(&dataset, &[0, 1], 20).await;
        let read_version = dataset.manifest.version;
        write_dv(&dataset, &mut destination, read_version, &[0]).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: vec![destination],
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
        assert!(
            error.to_string().contains("carries a deletion file"),
            "{error}"
        );
    }

    /// Round 10b (3): the rewrite-time accounting equation for ordered
    /// compaction -- deleted offsets absent from the survivor bitmap must
    /// exactly match the digest count. A corrupted current vector that
    /// dropped a rewrite-time deletion breaks it.
    #[tokio::test]
    async fn oc_scan_time_deletion_accounting_rejected() {
        let mut dataset = reader_tests::fixture().await;
        // A rewrite-time deletion: value 0 is absent from the survivor
        // bitmap and counts 1 in the digest.
        dataset.delete("i = 0").await.unwrap();
        reserve_fragments(&mut dataset, 30).await;
        let (transition, mut destination) = oc_job(&dataset, &[0, 1], 20).await;
        assert_eq!(transition.sources[0].num_deleted_rows, 1);
        // Corrupted current vector: drops offset 0, adds offsets 1 and 2
        // (count 2, delta +1, regime B engages).
        let mut fragments: Vec<Fragment> = dataset.fragments().as_ref().clone();
        let base_version = dataset.manifest.version;
        {
            let frag = &mut fragments[0];
            let deletion_vector: lance_core::utils::deletion::DeletionVector =
                [1u32, 2].into_iter().collect();
            let file = lance_table::io::deletion::write_deletion_file(
                &dataset.base,
                frag.id,
                base_version,
                &deletion_vector,
                &dataset.object_store,
            )
            .await
            .unwrap()
            .unwrap();
            frag.deletion_file = Some(file);
        }
        Arc::make_mut(&mut dataset.manifest).fragments = Arc::new(fragments);
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        reader_tests::persist_fixture(&mut dataset, indices).await;

        let read_version = dataset.latest_version_id().await.unwrap();
        write_dv(&dataset, &mut destination, read_version, &[1]).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: vec![destination],
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
        assert!(
            error.to_string().contains("scan-time deletion accounting"),
            "{error}"
        );
    }

    /// Round 10b (4): right cardinality, wrong destination offset under an
    /// ordered-compaction fold breaks the set equation.
    #[tokio::test]
    async fn oc_folded_wrong_position_rejected() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        let (transition, mut destination) = oc_job(&dataset, &[0, 1], 20).await;
        dataset.delete("i = 1 OR i = 6").await.unwrap();
        let read_version = dataset.manifest.version;
        // Value 6 translates to destination offset 6; offset 5 is wrong.
        write_dv(&dataset, &mut destination, read_version, &[1, 5]).await;
        let old_fragments: Vec<Fragment> = dataset.fragments().iter().cloned().collect();
        let error = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments: vec![destination],
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
        assert!(
            error.to_string().contains("translated positions"),
            "{error}"
        );
    }

    /// Round 10b (5): a mixed rewrite -- one stable-partition group and one
    /// ordered-compaction group in the same commit, both folding -- is
    /// validated independently per transition and both records land.
    #[tokio::test]
    async fn mixed_sp_and_oc_folding_validate_independently() {
        let mut dataset = reader_tests::fixture().await;
        reserve_fragments(&mut dataset, 30).await;
        // Stable partition over fragment 0 (values 0..4, parity split:
        // destination 10 evens, destination 11 odds).
        let (sp_transition, sp_destinations) =
            reader_tests::prepare_partition(&dataset, &[0], 10).await;
        // Ordered compaction over fragment 1 (values 4..8, scan order:
        // value v at destination offset v - 4).
        let (oc_transition, mut oc_destination) = oc_job(&dataset, &[1], 21).await;

        dataset.delete("i = 1 OR i = 6").await.unwrap();
        let read_version = dataset.manifest.version;
        // Value 1: stable-partition label odd, first odd row -> destination
        // 11 offset 0. Value 6: compaction offset 2.
        let mut sp_destinations = sp_destinations;
        write_dv(&dataset, &mut sp_destinations[1], read_version, &[0]).await;
        write_dv(&dataset, &mut oc_destination, read_version, &[2]).await;

        let sp_old: Vec<Fragment> = dataset
            .fragments()
            .iter()
            .filter(|f| f.id == 0)
            .cloned()
            .collect();
        let oc_old: Vec<Fragment> = dataset
            .fragments()
            .iter()
            .filter(|f| f.id == 1)
            .cloned()
            .collect();
        let dataset = crate::dataset::write::CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::Rewrite {
                    groups: vec![
                        RewriteGroup {
                            old_fragments: sp_old,
                            new_fragments: sp_destinations,
                        },
                        RewriteGroup {
                            old_fragments: oc_old,
                            new_fragments: vec![oc_destination],
                        },
                    ],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                    frag_reuse_rewrite: Some(FragmentReuseRewrite {
                        transitions: vec![sp_transition, oc_transition],
                        base_entry_version: None,
                    }),
                },
                None,
            ))
            .await
            .unwrap();

        assert_eq!(sorted_values(&dataset).await, vec![0, 2, 3, 4, 5, 7]);
        assert_eq!(filtered_values(&dataset, "i = 1").await, Vec::<i32>::new());
        assert_eq!(filtered_values(&dataset, "i = 6").await, Vec::<i32>::new());
        assert_eq!(filtered_values(&dataset, "i = 7").await, vec![7]);
        let stored = crate::index::load_all_indices(&dataset).await.unwrap();
        let entry = stored_fri(&stored);
        let ledger = decode_entry(&dataset, &entry).await;
        assert_eq!(ledger.transitions().len(), 2);
        let (sp_count, oc_count) = ledger
            .transitions()
            .iter()
            .fold((0, 0), |(sp, oc), t| match t.mapping() {
                Mapping::StablePartition(_) => (sp + 1, oc),
                Mapping::OrderedCompaction(_) => (sp, oc + 1),
            });
        assert_eq!((sp_count, oc_count), (1, 1));
    }

    /// Round 10b P1 regression: the source deletion state is read from the
    /// manifest, never from the job-supplied group. A crafted group whose
    /// deletion vector swaps the deleted offset (digest {}, manifest {1},
    /// group {2}) with a destination vector folded for the wrong row keeps
    /// every count equal; only manifest authority catches the position lie
    /// (row 1 resurrected, row 2 wrongly dead).
    #[tokio::test]
    async fn job_supplied_source_deletion_state_cannot_resurrect_rows() {
        let mut harness = folding_harness(Some("i = 1")).await;
        // Forge the group copy of fragment 0: same count, different offset.
        let forged: lance_core::utils::deletion::DeletionVector = [2u32].into_iter().collect();
        let file = lance_table::io::deletion::write_deletion_file(
            &harness.dataset.base,
            0,
            harness.read_version,
            &forged,
            &harness.dataset.object_store,
        )
        .await
        .unwrap()
        .unwrap();
        let slot = harness
            .old_fragments
            .iter()
            .position(|f| f.id == 0)
            .unwrap();
        harness.old_fragments[slot].deletion_file = Some(file);
        // Fold as if value 2 were the delta: destination 0 offset 1.
        fold_destination(&mut harness, 0, &[1]).await;
        let error = commit_fold(harness).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("translated positions"),
            "{error}"
        );
    }
}
