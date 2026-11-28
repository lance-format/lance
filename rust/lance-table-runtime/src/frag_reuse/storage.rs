// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use bytes::Buf;
use lance_core::Error;
use lance_io::object_store::ObjectStore;
use lance_table::format::IndexMetadata;
use lance_table::format::pb::fragment_reuse_index_details::{
    self as pb_fri, Content, InlineContent,
};
use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
use lance_table::system_index::frag_reuse::{
    FRAG_REUSE_DETAILS_FILE_NAME, FRAG_REUSE_INDEX_NAME, FragReuseGroup, FragReuseIndexDetails,
    FragReuseVersion, PreloadedFragReuseHistory,
};
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::Table;
use crate::caches::FragReuseDetailsKey;
use crate::commit::load_all_indices;

/// `dataset_version` stamps both the new reuse version and the entry, which must agree.
pub async fn build_new_frag_reuse_index(
    dataset: &Table,
    frag_reuse_groups: Vec<FragReuseGroup>,
    new_fragment_bitmap: RoaringBitmap,
    dataset_version: u64,
) -> lance_core::Result<IndexMetadata> {
    let new_version = FragReuseVersion {
        // `finish_rewrite` restamps it if the rewrite publishes on a later version.
        dataset_version,
        groups: frag_reuse_groups,
    };

    let indices = dataset.load_index_metadata().await?;
    let index_meta = indices.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME);

    let (new_index_details, fragment_bitmap) = match index_meta {
        None => (
            FragReuseIndexDetails {
                versions: Vec::from([new_version]),
            },
            new_fragment_bitmap,
        ),
        Some(index_meta) => {
            let current_details = load_frag_reuse_index_details(dataset, index_meta).await?;
            // Every version's new fragments, as a rebuild in `finish_rewrite` or a cleanup
            // publishes, so the entry is the same whether or not its commit is restamped.
            let fragment_bitmap = current_details.new_frag_bitmap() | new_fragment_bitmap;
            let mut versions = current_details.versions.clone();
            versions.push(new_version);
            (FragReuseIndexDetails { versions }, fragment_bitmap)
        }
    };

    let mut entry =
        build_frag_reuse_index_metadata(dataset, index_meta, new_index_details, fragment_bitmap)
            .await?;
    entry.dataset_version = dataset_version;
    Ok(entry)
}

/// Load fragment reuse index details from index metadata
pub async fn load_frag_reuse_index_details(
    dataset: &Table,
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
            let (store, path) = fri_external_location(dataset, index, external_file).await?;
            dataset
                .index_cache()
                .get_or_insert_with_key(
                    FragReuseDetailsKey {
                        store_identity: &store.store_prefix,
                        path: &path,
                        offset: external_file.offset,
                        size: external_file.size,
                    },
                    || async {
                        let data = read_fri_external_range(&store, &path, external_file).await?;
                        FragReuseIndexDetails::try_from(InlineContent::decode(data)?)
                    },
                )
                .await
        }
    }
}

/// Where an FRI entry's external details live, honoring the entry's base:
/// a shallow-cloned entry's `details.binpb` lives in the SOURCE dataset, so
/// the path and store come from the entry's `base_id` (like every other
/// base-aware index file) instead of the current dataset root.
async fn fri_external_location(
    dataset: &Table,
    index: &IndexMetadata,
    file: &ExternalFile,
) -> lance_core::Result<(Arc<ObjectStore>, Path)> {
    let path = dataset
        .view()
        .indice_files_dir(index)?
        .join(index.uuid.to_string())
        .join(file.path.as_str());
    Ok((dataset.object_store(index.base_id).await?, path))
}

async fn read_fri_external_range(
    store: &ObjectStore,
    path: &Path,
    file: &ExternalFile,
) -> lance_core::Result<bytes::Bytes> {
    let end = file
        .offset
        .checked_add(file.size)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::corrupt_file_named("FRI details", "external FRI range overflow"))?;
    store
        .open(path)
        .await?
        .get_range(file.offset as usize..end)
        .await
        .map_err(Error::from)
}

/// An FRI entry's raw external details bytes, read from storage.
async fn read_fri_external_file(
    dataset: &Table,
    index: &IndexMetadata,
    file: &ExternalFile,
) -> lance_core::Result<bytes::Bytes> {
    let (store, path) = fri_external_location(dataset, index, file).await?;
    read_fri_external_range(&store, &path, file).await
}

pub async fn build_frag_reuse_index_metadata(
    dataset: &Table,
    index_meta: Option<&IndexMetadata>,
    mut new_index_details: FragReuseIndexDetails,
    new_fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    // The encoding orders versions by stamp; the cached copy must match a read of the file.
    new_index_details
        .versions
        .sort_by_key(|version| version.dataset_version);
    let index_id = uuid::Uuid::new_v4();
    let new_index_details_proto = InlineContent::from(&new_index_details);
    let proto = if new_index_details_proto.encoded_len() > FRAG_REUSE_INLINE_DETAILS_LIMIT {
        let file_path = dataset
            .indices_dir()
            .join(index_id.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME);
        let mut writer = dataset.state.object_store.create(&file_path).await?;
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

    let entry = IndexMetadata {
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
        clustering_id: None,
        metadata: Default::default(),
        local_fragment_aliases: Default::default(),
    };
    // Spares the commit's history check from reading the file back.
    if let Some(Content::External(file)) = &proto.content {
        let (store, path) = fri_external_location(dataset, &entry, file).await?;
        dataset
            .index_cache()
            .insert_with_key(
                &FragReuseDetailsKey {
                    store_identity: &store.store_prefix,
                    path: &path,
                    offset: file.offset,
                    size: file.size,
                },
                Arc::new(new_index_details),
            )
            .await;
    }
    Ok(entry)
}

/// One length-delimited protobuf field, the unit both the inline details
/// payload and appended transitions are spliced with.
pub fn encode_length_delimited_field(tag: u32, bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len() + 8);
    prost::encoding::encode_key(tag, prost::encoding::WireType::LengthDelimited, &mut output);
    prost::encoding::encode_varint(bytes.len() as u64, &mut output);
    output.extend_from_slice(bytes);
    output
}

/// Decode a committed FRI entry into its transition ledger, resolving
/// external content through the index's registered base. The entry's original
/// `Any` is decoded directly, so the envelope is parsed exactly once, by the
/// ledger. Works for v0 entries too: legacy versions decode as lifted
/// transitions.
// The sp-x-sp manifest diff caller went away when it was replaced by
// merge-through-reassembly; the maintenance stack (trim derivation, tagged
// remap planning) and the commit-path tests call it now.
pub async fn decode_frag_reuse_ledger(
    dataset: &Table,
    entry: &IndexMetadata,
) -> lance_core::Result<lance_table::system_index::frag_reuse::ledger::FragReuseLedger> {
    let details = entry
        .index_details
        .as_ref()
        .ok_or_else(|| Error::index("Index details is not for the fragment reuse index"))?;
    lance_table::system_index::frag_reuse::ledger::FragReuseLedger::decode(
        entry.index_version,
        details,
        |file| async move { read_fri_external_file(dataset, entry, &file).await },
    )
    .await
}

/// Decode already-loaded FRI content bytes into a transition ledger. Used by
/// maintenance paths that hold the entry's verbatim content (the trim's
/// splice inputs and outputs, cleanup's reference resolution) and by tests.
pub async fn decode_frag_reuse_ledger_from_content(
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
pub async fn load_raw_frag_reuse_content(
    dataset: &Table,
    index: &IndexMetadata,
) -> lance_core::Result<Vec<u8>> {
    let details = index
        .index_details
        .as_ref()
        .filter(|details| details.type_url.ends_with("FragmentReuseIndexDetails"))
        .ok_or_else(|| Error::index("Index details is not for the fragment reuse index"))?;
    extract_raw_frag_reuse_content(details, |file| async move {
        read_fri_external_file(dataset, index, &file).await
    })
    .await
}

/// [`load_raw_frag_reuse_content`] with the external read abstracted: callers
/// outside a live `Table` (the shallow-clone relocation reads the SOURCE
/// dataset's entry before the clone exists; the parent's cleanup reads a
/// BRANCH manifest's entry) supply their own resolution.
pub async fn extract_raw_frag_reuse_content<F, Fut>(
    details: &prost_types::Any,
    read_external: F,
) -> lance_core::Result<Vec<u8>>
where
    F: FnOnce(ExternalFile) -> Fut,
    Fut: std::future::Future<Output = lance_core::Result<bytes::Bytes>>,
{
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

/// The complete fragment reuse index entry a rewrite installs when it
/// appends `transitions`: `dataset`'s entry (its records carried over
/// verbatim, a v0 entry lifted to the tagged format) plus the new
/// transitions. This is what `Operation::Rewrite::frag_reuse_index` carries
/// for a rewrite on a tagged history.
///
/// `dataset` must be the snapshot at the transaction's read version: the
/// commit path diffs the entry against the entry at that version to find
/// what this rewrite adds, and refuses an entry whose `dataset_version`
/// differs from the read version. A caller whose handle has moved on (a
/// distributed compaction driver at V+N committing tasks planned at V)
/// checks the read version out first.
///
/// The result is validated as a ledger (lineage order, digest conservation,
/// mapping presence, no mapping this writer cannot maintain) but not yet
/// against the rewrite groups: the commit path works out which transitions
/// the entry adds relative to the read version's entry, binds them to the
/// groups, merges them onto the entry current at commit and finalizes the
/// result (spilling content above the inline limit to an external file).
/// The returned entry keeps its content inline whatever its size; it is
/// in-memory intent, never the entry that reaches a manifest.
pub async fn frag_reuse_entry_appending(
    dataset: &Table,
    transitions: Vec<pb_fri::Transition>,
) -> lance_core::Result<IndexMetadata> {
    if dataset.manifest.uses_stable_row_ids() {
        return Err(Error::not_supported(
            "Tagged fragment reuse histories are address-based and excluded on \
             stable-row-id datasets; this rewrite cannot carry transition intent here",
        ));
    }
    let stored = load_all_indices(dataset, None).await?;
    let existing = stored.iter().find(|idx| idx.name == FRAG_REUSE_INDEX_NAME);
    let (mut content, mut fragment_bitmap) = match existing {
        None => (Vec::new(), RoaringBitmap::new()),
        Some(entry) => {
            if !matches!(entry.index_version, 0 | 1) {
                return Err(Error::not_supported(format!(
                    "Cannot append a transition to FRI index_version {}; upgrade to a newer \
                     version of Lance",
                    entry.index_version
                )));
            }
            (
                load_raw_frag_reuse_content(dataset, entry).await?,
                entry.fragment_bitmap.clone().unwrap_or_default(),
            )
        }
    };
    for transition in &transitions {
        content.extend_from_slice(&encode_length_delimited_field(
            2,
            &transition.encode_to_vec(),
        ));
    }
    let details = prost_types::Any {
        type_url: "/lance.table.FragmentReuseIndexDetails".into(),
        value: encode_length_delimited_field(1, &content),
    };
    let ledger = lance_table::system_index::frag_reuse::ledger::FragReuseLedger::decode(
        1,
        &details,
        |_| async {
            Err(Error::invalid_input(
                "the assembled FRI content is inline; no external read is possible",
            ))
        },
    )
    .await?;
    if ledger.has_unsupported_transitions() {
        return Err(Error::not_supported(
            "the fragment reuse history contains mappings this writer cannot maintain; \
             upgrade to a newer version of Lance before rewriting this table",
        ));
    }
    for transition in &transitions {
        for digest in transition
            .sources
            .iter()
            .chain(transition.destinations.iter())
        {
            fragment_bitmap.insert(digest.id as u32);
        }
    }
    Ok(IndexMetadata {
        uuid: Uuid::new_v4(),
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(details)),
        index_version: 1,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
        local_fragment_aliases: Default::default(),
    })
}

/// The records a fragment reuse entry holds, whatever its version: its
/// legacy compaction versions and its tagged transitions.
pub async fn load_frag_reuse_records(
    dataset: &Table,
    entry: &IndexMetadata,
) -> lance_core::Result<InlineContent> {
    let content = load_raw_frag_reuse_content(dataset, entry).await?;
    decode_records(&content)
}

pub(super) fn decode_records(content: &[u8]) -> lance_core::Result<InlineContent> {
    InlineContent::decode(content).map_err(|err| {
        Error::corrupt_file_named(
            "FRI details",
            format!("the fragment reuse content does not decode: {err}"),
        )
    })
}

/// Package assembled tagged FRI content bytes into a fresh manifest entry,
/// spilling to an external details file above the inline threshold. The
/// content must already be validated (a decodable ledger); this only encodes.
pub async fn build_tagged_frag_reuse_entry(
    dataset: &Table,
    content: Vec<u8>,
    fragment_bitmap: RoaringBitmap,
) -> lance_core::Result<IndexMetadata> {
    let index_id = Uuid::new_v4();
    let details_value = encode_tagged_frag_reuse_details(
        &dataset.state.object_store,
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
        clustering_id: None,
        metadata: Default::default(),
        local_fragment_aliases: Default::default(),
    })
}

/// Encode validated tagged FRI content bytes as an `index_details` value,
/// spilling to `<indices_dir>/<index_id>/details.binpb` above the inline
/// threshold.
pub(super) async fn encode_tagged_frag_reuse_details(
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
pub const FRAG_REUSE_INLINE_DETAILS_LIMIT: usize = 204800;

/// The current fragment reuse history, read for an action-based commit whose
/// manifest build cannot read the external details file it lives in. `None`
/// when the table has no history or keeps it inline, which the build reads
/// from the entry itself.
pub async fn preload_external_frag_reuse_history(
    dataset: &Table,
) -> lance_core::Result<Option<PreloadedFragReuseHistory>> {
    let indices = load_all_indices(dataset, None).await?;
    let Some(entry) = indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
    else {
        return Ok(None);
    };
    let is_external = entry.index_details.as_ref().is_some_and(|details| {
        FragmentReuseIndexDetails::decode(details.value.as_slice())
            .is_ok_and(|details| matches!(details.content, Some(Content::External(_))))
    });
    if !is_external {
        return Ok(None);
    }
    let content = load_raw_frag_reuse_content(dataset, entry).await?;
    Ok(Some(PreloadedFragReuseHistory {
        entry: entry.uuid,
        content: content.into(),
    }))
}

/// Move a fragment reuse history an action-based commit assembled inline to
/// an external details file once it outgrows the inline limit. The manifest
/// build does no IO, so it always assembles the history inline.
pub async fn spill_frag_reuse_history(
    dataset: &Table,
    indices: &mut [IndexMetadata],
) -> lance_core::Result<()> {
    for entry in indices
        .iter_mut()
        .filter(|index| index.name == FRAG_REUSE_INDEX_NAME)
    {
        let Some(details) = entry.index_details.as_ref() else {
            continue;
        };
        if details.value.len() <= FRAG_REUSE_INLINE_DETAILS_LIMIT {
            continue;
        }
        let content = extract_raw_frag_reuse_content(details, |_| async {
            Err(Error::internal(
                "a fragment reuse history over the inline limit is already external",
            ))
        })
        .await?;
        let value = encode_tagged_frag_reuse_details(
            &dataset.state.object_store,
            dataset.indices_dir(),
            entry.uuid,
            content,
        )
        .await?;
        entry.index_details = Some(Arc::new(prost_types::Any {
            type_url: details.type_url.clone(),
            value,
        }));
    }
    Ok(())
}
