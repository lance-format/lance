// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The IO around the FRI entry codec: reading the external history file
//! [`FragReuseHistory::decode`] needs and writing the one
//! [`FragReuseHistory::encode`] spills.

use std::borrow::Cow;
use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::{Error, Result};
use lance_index::frag_reuse::{
    FRAG_REUSE_DETAILS_FILE_NAME, FRAG_REUSE_INDEX_NAME, FragReuseIndexDetails,
};
use lance_io::object_store::ObjectStore;
use lance_table::format::IndexMetadata;
use lance_table::format::pb::ExternalFile;
use lance_table::format::pb::fragment_reuse_index_details::InlineContent;
use lance_table::system_index::frag_reuse::history::{
    EncodedHistory, FragReuseHistory, HistoryFormat,
};
use object_store::path::Path;
use prost::Message;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::Dataset;
use crate::session::index_caches::{FragReuseContent, FragReuseContentKey, FragReuseDetailsKey};

/// The bytes of `entry`'s external history file, read through
/// `read_external`, which must return exactly the referenced byte range;
/// `None` when the history is inline.
pub async fn read_external_history<F, Fut>(
    entry: &IndexMetadata,
    read_external: F,
) -> Result<Option<Bytes>>
where
    F: FnOnce(ExternalFile) -> Fut,
    Fut: Future<Output = Result<Bytes>>,
{
    match FragReuseHistory::locate(entry)? {
        Some(file) => Ok(Some(read_external(file).await?)),
        None => Ok(None),
    }
}

/// [`read_external_history`] through the entry's base, once per session.
async fn read_entry_external_history(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> Result<Option<Bytes>> {
    read_external_history(entry, |file| async move {
        read_fri_external_file(dataset, entry, &file).await
    })
    .await
}

/// The history a committed FRI entry holds, resolving external details
/// through the entry's base.
pub async fn load_frag_reuse_history(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> Result<FragReuseHistory> {
    let external = read_entry_external_history(dataset, entry).await?;
    FragReuseHistory::decode(entry, external)
}

/// The history a caller's uncommitted FRI entry proposes (see
/// [`FragReuseHistory::decode_proposed`]).
pub async fn load_proposed_frag_reuse_history(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> Result<FragReuseHistory> {
    // Locating and reading the external file decodes the entry's details
    // and checks their range before `decode_proposed` sees the entry, so a
    // proposal with bad details can fail here first. It is the caller's
    // error either way.
    let external =
        read_entry_external_history(dataset, entry)
            .await
            .map_err(|error| match error {
                Error::CorruptFile { source, .. } => Error::invalid_input(format!(
                    "the proposed fragment reuse entry is invalid: {source}"
                )),
                other => other,
            })?;
    FragReuseHistory::decode_proposed(entry, external)
}

/// The history of `indices`' FRI entry, if the table has one.
pub async fn stored_frag_reuse_history(
    dataset: &Dataset,
    indices: &[IndexMetadata],
) -> Result<Option<FragReuseHistory>> {
    match indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
    {
        Some(entry) => Ok(Some(load_frag_reuse_history(dataset, entry).await?)),
        None => Ok(None),
    }
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
    if let Some(file) = FragReuseHistory::locate(&encoded.entry)? {
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
        if history.format() == HistoryFormat::Legacy {
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
    let EncodedHistory { entry, external } = history.encode(dataset_version);
    let spilled = match external {
        Some(content) => {
            let path = indices_dir
                .join(entry.uuid.to_string())
                .join(FRAG_REUSE_DETAILS_FILE_NAME);
            let mut writer = store.create(&path).await?;
            writer.write_all(&content[..]).await?;
            writer.shutdown().await?;
            Some(path)
        }
        None => None,
    };
    Ok(EncodedEntry { entry, spilled })
}

/// Where an FRI entry's external details live, honoring the entry's base:
/// a shallow-cloned entry's `details.binpb` lives in the SOURCE dataset, so
/// the path and store come from the entry's `base_id` (like every other
/// base-aware index file) instead of the current dataset root.
pub(super) async fn fri_external_location(
    dataset: &Dataset,
    index: &IndexMetadata,
    file: &ExternalFile,
) -> lance_core::Result<(Arc<ObjectStore>, Path)> {
    let path = dataset
        .indice_files_dir(index)?
        .join(index.uuid.to_string())
        .join(file.path.as_str());
    Ok((dataset.object_store_for_index(index).await?, path))
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

/// An FRI entry's raw external details bytes, read once per session.
pub(super) async fn read_fri_external_file(
    dataset: &Dataset,
    index: &IndexMetadata,
    file: &ExternalFile,
) -> lance_core::Result<bytes::Bytes> {
    let (store, path) = fri_external_location(dataset, index, file).await?;
    let content = dataset
        .index_cache
        .get_or_insert_with_key(
            FragReuseContentKey {
                store_identity: &store.store_prefix,
                path: &path,
                offset: file.offset,
                size: file.size,
            },
            || async {
                read_fri_external_range(&store, &path, file)
                    .await
                    .map(FragReuseContent)
            },
        )
        .await?;
    Ok(content.0.clone())
}

impl Dataset {
    /// The fragment reuse history of this snapshot, or `None` when the table
    /// has none.
    ///
    /// The history is decoded once per session for each fragment reuse
    /// entry, and shared by every version and index segment that sees it.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn test(dataset: &Dataset) -> Result<()> {
    /// if let Some(history) = dataset.frag_reuse_history().await? {
    ///     let transitions = history.ledger().transitions();
    ///     assert!(transitions.iter().all(|transition| !transition.sources().is_empty()));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn frag_reuse_history(&self) -> Result<Option<FragReuseHistory>> {
        let indices = crate::index::loading::load_all_indices(self).await?;
        match indices
            .iter()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        {
            Some(entry) => Ok(Some(
                cached_frag_reuse_history(self, entry)
                    .await?
                    .as_ref()
                    .clone(),
            )),
            None => Ok(None),
        }
    }
}

/// [`load_frag_reuse_history`], once per session for each entry.
pub async fn cached_frag_reuse_history(
    dataset: &Dataset,
    entry: &IndexMetadata,
) -> Result<Arc<FragReuseHistory>> {
    let store = dataset.object_store_for_index(entry).await?;
    let key = FragReuseHistoryKey {
        entry: super::v0_cache_identity(dataset, entry),
        index_version: entry.index_version,
        store: &store.store_prefix,
        directory: dataset.indice_files_dir(entry)?.to_string(),
    };
    dataset
        .index_cache
        .get_or_insert_with_key(key, || load_frag_reuse_history(dataset, entry))
        .await
}

/// Keys a decoded history by its entry. The entry identity covers the
/// entry's uuid and stored details: released version 0 writers could change
/// a history without minting a new uuid.
struct FragReuseHistoryKey<'a> {
    entry: Uuid,
    index_version: i32,
    store: &'a str,
    directory: String,
}

impl CacheKey for FragReuseHistoryKey<'_> {
    type ValueType = FragReuseHistory;

    fn key(&self) -> Cow<'_, str> {
        format!(
            "{}:{}:{}:{}",
            self.entry, self.index_version, self.store, self.directory
        )
        .into()
    }

    fn type_name() -> &'static str {
        "FragmentReuseHistory"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-history", 2)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.entry.as_bytes());
        builder.write_i32(self.index_version);
        builder.write_str(self.store);
        builder.write_str(&self.directory);
    }
}
