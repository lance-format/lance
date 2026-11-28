// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::Arc;

use futures::TryStreamExt;
use lance_core::Error;
use lance_io::object_store::{ObjectStore, ObjectStoreRegistry};
use lance_table::format::{IndexMetadata, Manifest};
use lance_table::system_index::frag_reuse::FRAG_REUSE_DETAILS_FILE_NAME;
use lance_table::system_index::frag_reuse::ledger::Mapping;
use lance_table::system_index::frag_reuse::stable_partition::MAPPING_FILE;
use object_store::path::Path;
use uuid::Uuid;

use crate::Table;
use crate::paths::{FRAG_REUSE_DIR, INDICES_DIR};

use super::storage::{
    FRAG_REUSE_INLINE_DETAILS_LIMIT, decode_frag_reuse_ledger,
    decode_frag_reuse_ledger_from_content, encode_length_delimited_field,
    encode_tagged_frag_reuse_details, extract_raw_frag_reuse_content,
};

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

/// How a clone restamps the stable-partition `base_id`s it relocates.
pub enum CloneBaseRemap {
    /// Shallow clone: row-map files stay where they are.
    /// `Manifest::shallow_clone` carries the source's `base_paths` ids over
    /// verbatim and adds `new_base_id` for the source's own base, so `None`
    /// becomes `Some(new_base_id)` and `Some(id)` stays `id`.
    Shallow { new_base_id: u32 },
    /// Deep clone: every referenced row map is copied into the clone's own
    /// `_fri/` (see [`collect_tagged_row_map_paths`]), so every reference
    /// becomes local: any stamp clears to unset.
    Deep,
}

/// Relocate a tagged FRI entry for a clone.
///
/// The source entry's content is resolved base-aware (a chained clone's entry
/// or details file may live in one of the source's own bases), its
/// stable-partition references are restamped through the clone's base mapping
/// (per [`CloneBaseRemap`]), and the relocated content is written into the
/// CLONE under a fresh uuid with `base_id: None`.
///
/// A history this writer cannot fully interpret (unsupported transitions or
/// unknown records) is refused with `NotSupported` before anything is
/// written, per the spec's writer obligation.
///
/// Returns the relocated entry and, when the details spilled to an external
/// file in the target, that file's path so a failed clone commit can delete
/// it again.
#[allow(clippy::too_many_arguments)]
pub async fn relocate_tagged_entry_for_clone(
    source_store: &ObjectStore,
    source_base: &Path,
    source_manifest: &Manifest,
    store_registry: Arc<ObjectStoreRegistry>,
    entry: &IndexMetadata,
    base_remap: CloneBaseRemap,
    target_store: &ObjectStore,
    target_base: &Path,
) -> lance_core::Result<(IndexMetadata, Option<Path>)> {
    let details = entry
        .index_details
        .as_ref()
        .filter(|details| details.type_url.ends_with("FragmentReuseIndexDetails"))
        .ok_or_else(|| Error::index("Index details is not for the fragment reuse index"))?;
    // Base-aware resolution of the entry's external details without a live
    // `Table`: the same rule as `read_fri_external_file`, against the
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
            None => (None, source_base.clone().join(INDICES_DIR)),
            Some(id) => {
                let base_path = source_manifest.base_paths.get(&id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "base_path id {} not found for index {}",
                        id, entry.uuid
                    ))
                })?;
                let path = base_path.extract_path(registry.clone())?;
                let dir = if base_path.is_dataset_root {
                    path.join(INDICES_DIR)
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

    let relocated =
        relocate_stable_partition_bases(&content, |base_id| match (&base_remap, base_id) {
            // Deep clone materializes every referenced map into the clone's
            // own `_fri/`, so every stamp clears to unset.
            (CloneBaseRemap::Deep, _) => Ok(None),
            // The source's own base gets the id the clone assigned to it.
            (CloneBaseRemap::Shallow { new_base_id }, None) => Ok(Some(*new_base_id)),
            // The clone carries the source's `base_paths` entries over under
            // the same ids, so a foreign-base reference keeps its id; it
            // must exist.
            (CloneBaseRemap::Shallow { .. }, Some(id)) => {
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
    let target_indices_dir = target_base.clone().join(INDICES_DIR);
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
            clustering_id: None,
            metadata: Default::default(),
        },
        spilled,
    ))
}

/// The copy work a deep clone owes a tagged FRI entry: every file of every
/// `_fri/<map_id>/` row map the history references, resolved through the
/// manifest's `base_paths` (a chained shallow-then-deep clone finds maps
/// living in ancestor bases), as `(relative_path, base_root)` pairs for
/// `deep_clone`'s copy loop. Map ids are UUIDs minted at rewrite time, so
/// the copies land in the target's fresh `_fri/` namespace collision-free.
///
/// A history this writer cannot fully interpret (an unknown mapping kind or
/// an unknown envelope record) is refused with `NotSupported` before
/// anything is copied: an unknown record may reference row maps this writer
/// cannot enumerate, and copying an incomplete set would fabricate a
/// silently broken "independent" table. This is the same obligation the
/// shallow-clone relocation honors, with the same error.
pub async fn collect_tagged_row_map_paths(
    dataset: &Table,
    entry: &IndexMetadata,
) -> lance_core::Result<Vec<(String, Path)>> {
    let ledger = decode_frag_reuse_ledger(dataset, entry).await?;
    if ledger.has_unsupported_transitions() {
        return Err(Error::not_supported(
            "the fragment reuse history contains mappings this writer cannot relocate; \
             upgrade to a newer version of Lance before cloning this table",
        ));
    }

    let mut seen: HashSet<(Option<u32>, &str)> = HashSet::new();
    let mut paths = Vec::new();
    for transition in ledger.transitions() {
        let Mapping::StablePartition(reference) = transition.mapping() else {
            continue;
        };
        if !seen.insert((reference.base_id, reference.map_id.as_str())) {
            continue;
        }
        // The same base-aware resolution the reader opens the map with.
        let base_root = match reference.base_id {
            None => dataset.state.base.clone(),
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
                .extract_path(dataset.state.session.store_registry())?,
        };
        let map_dir = base_root
            .clone()
            .join(FRAG_REUSE_DIR)
            .join(reference.map_id.as_str());
        let mut mapping_size = None;
        // Deep clone copies these paths through the source table's store.
        let mut stream = dataset.state.object_store.read_dir_all(&map_dir, None);
        loop {
            match stream.try_next().await {
                Ok(Some(meta)) => {
                    let relative = meta
                        .location
                        .as_ref()
                        .strip_prefix(map_dir.as_ref())
                        .ok_or_else(|| {
                            Error::internal(format!(
                                "listing {} returned {} outside it",
                                map_dir, meta.location
                            ))
                        })?
                        .trim_start_matches('/');
                    if relative == MAPPING_FILE {
                        mapping_size = Some(meta.size);
                    }
                    paths.push((
                        format!("{FRAG_REUSE_DIR}/{}/{relative}", reference.map_id),
                        base_root.clone(),
                    ));
                }
                Ok(None) => break,
                Err(Error::NotFound { .. }) => break,
                Err(error) => return Err(error),
            }
        }
        // The copy must produce a working clone: the map's mapping file has
        // to be there with the size the history records for it (the reader
        // opens the file by that size). Refuse the clone now rather than
        // hand the target a history whose first translating query fails.
        match mapping_size {
            Some(size) if size == reference.map_size_bytes => {}
            Some(size) => {
                return Err(Error::corrupt_file_named(
                    "FRI details",
                    format!(
                        "row map {} has a {MAPPING_FILE} of {size} bytes under its base, \
                         the history records {} bytes",
                        reference.map_id, reference.map_size_bytes
                    ),
                ));
            }
            None => {
                return Err(Error::corrupt_file_named(
                    "FRI details",
                    format!(
                        "row map {} has no {MAPPING_FILE} under its base",
                        reference.map_id
                    ),
                ));
            }
        }
    }
    Ok(paths)
}
