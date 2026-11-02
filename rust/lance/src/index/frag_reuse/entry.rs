// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading and writing the FRI entry. Every entry is built from a
//! [`FragReuseHistory`] here, so the inline-or-spill choice, the fragment
//! bitmap and the `index_version` follow one rule.

use std::sync::Arc;

use lance_core::{Error, Result};
use lance_index::frag_reuse::{
    FRAG_REUSE_DETAILS_FILE_NAME, FRAG_REUSE_INDEX_NAME, FragReuseIndexDetails,
};
use lance_io::object_store::ObjectStore;
use lance_table::format::IndexMetadata;
use lance_table::format::pb::ExternalFile;
use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
use lance_table::system_index::frag_reuse::history::{FragReuseHistory, length_delimited};
use object_store::path::Path;
use prost::Message;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use super::{fri_external_location, read_fri_external_file};
use crate::Dataset;
use crate::session::index_caches::{FragReuseContent, FragReuseContentKey, FragReuseDetailsKey};

/// Details above this size spill to an external `details.binpb` file.
const FRAG_REUSE_INLINE_DETAILS_LIMIT: usize = 204800;

const DETAILS_TYPE_URL: &str = "/lance.table.FragmentReuseIndexDetails";
/// `FragmentReuseIndexDetails.inline`.
const INLINE_CONTENT_FIELD: u32 = 1;
/// `FragmentReuseIndexDetails.external`.
const EXTERNAL_CONTENT_FIELD: u32 = 2;

/// The history a committed FRI entry holds, resolving external details
/// through the entry's base.
pub async fn load_frag_reuse_history(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> Result<FragReuseHistory> {
    let details = entry
        .index_details
        .as_ref()
        .ok_or_else(|| Error::index("Index details is not for the fragment reuse index"))?;
    let (history, dropped) =
        FragReuseHistory::decode_stored(entry.index_version, details, |file| async move {
            read_fri_external_file(dataset, entry, &file).await
        })
        .await?;
    if !dropped.is_empty() {
        log::warn!(
            "fragment reuse entry {} records invalid versions {dropped:?}; they are dropped, and \
             rows they translated are scanned until their indices are rebuilt",
            entry.uuid
        );
    }
    Ok(history)
}

/// The history a caller's uncommitted FRI entry proposes. Unlike stored
/// history it is never repaired: an invalid proposal is the caller's error.
pub async fn load_proposed_frag_reuse_history(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> Result<FragReuseHistory> {
    let details = entry
        .index_details
        .as_ref()
        .ok_or_else(|| Error::invalid_input("the proposed fragment reuse entry has no details"))?;
    FragReuseHistory::decode(entry.index_version, details, |file| async move {
        read_fri_external_file(dataset, entry, &file).await
    })
    .await
    .map_err(|error| match error {
        Error::CorruptFile { source, .. } => Error::invalid_input(format!(
            "the proposed fragment reuse entry is invalid: {source}"
        )),
        other => other,
    })
}

/// The current FRI entry's history, or an empty history in the format of
/// `index_version` when the table has none.
pub async fn current_frag_reuse_history(
    dataset: &Dataset,
    index_version: i32,
) -> Result<(Option<IndexMetadata>, FragReuseHistory)> {
    let current = super::super::load_all_indices(dataset)
        .await?
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        .cloned();
    let history = match &current {
        Some(entry) => load_frag_reuse_history(dataset, entry).await?,
        None => FragReuseHistory::empty(index_version)?,
    };
    Ok((current, history))
}

/// An FRI entry and the external details file it wrote, if any.
pub struct EncodedEntry {
    pub entry: IndexMetadata,
    /// The spilled `details.binpb`, which a caller that fails to commit the
    /// entry may delete.
    pub spilled: Option<Path>,
}

/// Encode `history` as a new entry of `dataset`, spilling the details under
/// the dataset's indices directory when they exceed the inline limit.
pub async fn encode_frag_reuse_entry(
    dataset: &Dataset,
    history: &FragReuseHistory,
    dataset_version: u64,
) -> Result<IndexMetadata> {
    let encoded = encode_frag_reuse_entry_at(
        &dataset.object_store,
        dataset.indices_dir(),
        history,
        dataset_version,
    )
    .await?;
    // Spares the commit's history checks from reading the file back.
    if let Some(details) = &encoded.entry.index_details
        && let Some(file) = external_file(details)?
    {
        let (store, path) = fri_external_location(dataset, &encoded.entry, &file).await?;
        dataset
            .index_cache
            .insert_with_key(
                &FragReuseContentKey {
                    store_identity: &store.store_prefix,
                    path: &path,
                    offset: file.offset,
                    size: file.size,
                },
                Arc::new(FragReuseContent(history.content())),
            )
            .await;
        if !history.is_tagged() {
            let legacy = FragReuseIndexDetails::try_from(
                InlineContent::decode(history.content()).map_err(|e| {
                    Error::corrupt_file_named(
                        "FRI details",
                        format!("history does not decode: {e}"),
                    )
                })?,
            )?;
            dataset
                .index_cache
                .insert_with_key(
                    &FragReuseDetailsKey {
                        store_identity: &store.store_prefix,
                        path: &path,
                        offset: file.offset,
                        size: file.size,
                    },
                    Arc::new(legacy),
                )
                .await;
        }
    }
    Ok(encoded.entry)
}

/// [`encode_frag_reuse_entry`] into an arbitrary store, for a clone writing
/// into its target before the target dataset exists.
pub async fn encode_frag_reuse_entry_at(
    store: &ObjectStore,
    indices_dir: Path,
    history: &FragReuseHistory,
    dataset_version: u64,
) -> Result<EncodedEntry> {
    let uuid = Uuid::new_v4();
    let content = history.content();
    let (details, spilled) = if content.len() > FRAG_REUSE_INLINE_DETAILS_LIMIT {
        let path = indices_dir
            .join(uuid.to_string())
            .join(FRAG_REUSE_DETAILS_FILE_NAME);
        let mut writer = store.create(&path).await?;
        writer.write_all(&content[..]).await?;
        writer.shutdown().await?;
        let file = ExternalFile {
            path: FRAG_REUSE_DETAILS_FILE_NAME.to_owned(),
            offset: 0,
            size: content.len() as u64,
        };
        (
            length_delimited(EXTERNAL_CONTENT_FIELD, &file.encode_to_vec()),
            Some(path),
        )
    } else {
        (length_delimited(INLINE_CONTENT_FIELD, &content), None)
    };
    Ok(EncodedEntry {
        entry: entry_with_details(uuid, history, details, dataset_version),
        spilled,
    })
}

/// Encode `history` with its details always inline, for an entry that only
/// carries intent to the commit path and is never itself committed.
pub fn encode_inline_frag_reuse_entry(
    history: &FragReuseHistory,
    dataset_version: u64,
) -> IndexMetadata {
    let details = length_delimited(INLINE_CONTENT_FIELD, &history.content());
    entry_with_details(Uuid::new_v4(), history, details, dataset_version)
}

fn entry_with_details(
    uuid: Uuid,
    history: &FragReuseHistory,
    details: Vec<u8>,
    dataset_version: u64,
) -> IndexMetadata {
    IndexMetadata {
        uuid,
        name: FRAG_REUSE_INDEX_NAME.to_string(),
        fields: vec![],
        covering_fields: vec![],
        dataset_version,
        fragment_bitmap: Some(history.fragment_bitmap()),
        index_details: Some(Arc::new(prost_types::Any {
            type_url: DETAILS_TYPE_URL.into(),
            value: details,
        })),
        index_version: history.index_version(),
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    }
}

fn external_file(details: &prost_types::Any) -> Result<Option<ExternalFile>> {
    let details =
        lance_table::format::pb::FragmentReuseIndexDetails::decode(details.value.as_slice())
            .map_err(|e| Error::corrupt_file_named("FRI details", e.to_string()))?;
    Ok(match details.content {
        Some(lance_table::format::pb::fragment_reuse_index_details::Content::External(file)) => {
            Some(file)
        }
        _ => None,
    })
}
