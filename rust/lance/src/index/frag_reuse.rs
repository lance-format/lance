// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance_core::Error;
use lance_index::frag_reuse::{
    CompactFragReuseIndex, CompactFragReuseIndexHandle, FRAG_REUSE_DETAILS_FILE_NAME,
    FRAG_REUSE_INDEX_NAME, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
};
use lance_index::scalar::{MetricsCollector, RowIdRemapping};
use lance_table::format::IndexMetadata;
use lance_table::format::pb::fragment_reuse_index_details::{Content, InlineContent};
use lance_table::format::pb::{ExternalFile, FragmentReuseIndexDetails};
use prost::Message;
use roaring::RoaringBitmap;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

/// Resolve the FRI remapper shared by scalar and vector index loading.
pub(super) async fn open_row_id_remapping(
    dataset: &Dataset,
    index: &IndexMetadata,
    metrics: &dyn MetricsCollector,
) -> lance_core::Result<Option<(Uuid, RowIdRemapping)>> {
    let indices = dataset.load_indices().await?;
    let Some(fri) = indices
        .iter()
        .find(|entry| entry.name == FRAG_REUSE_INDEX_NAME)
    else {
        return Ok(None);
    };
    if fri.index_version == 0 {
        return Ok(dataset.open_frag_reuse_index(metrics).await?.map(|legacy| {
            (
                legacy.uuid,
                RowIdRemapping::InMemory(Arc::new(CompactFragReuseIndexHandle(legacy))),
            )
        }));
    }
    let mapping = super::frag_reuse_query::QueryFragReuseIndex::open(dataset, fri).await?;
    // Resolve provenance here, before query coverage is rewritten. Callers may
    // hold metadata returned by load_indices and cannot supply this distinction.
    let stored = super::load_all_indices(dataset).await?;
    let source = stored
        .iter()
        .find(|entry| entry.uuid == index.uuid)
        .ok_or_else(|| {
            Error::not_supported(format!(
                "FRI remapping requires committed segment metadata for {}",
                index.uuid
            ))
        })?;
    if !mapping.needs_translation(source.fragment_bitmap.as_ref()) {
        let identity =
            CompactFragReuseIndex::try_new(fri.uuid, FragReuseIndexDetails { versions: vec![] })?;
        return Ok(Some((
            fri.uuid,
            RowIdRemapping::InMemory(Arc::new(CompactFragReuseIndexHandle(Arc::new(identity)))),
        )));
    }
    let coverage = indices
        .iter()
        .find(|entry| entry.uuid == index.uuid)
        .and_then(|entry| entry.fragment_bitmap.clone())
        .ok_or_else(|| {
            Error::not_supported(format!(
                "FRI query coverage is unavailable for segment {}",
                index.uuid
            ))
        })?
        & dataset.fragment_bitmap.as_ref();
    Ok(Some((
        fri.uuid,
        RowIdRemapping::External(Arc::new(super::frag_reuse_query::QueryRowIdRemapper::new(
            mapping, coverage,
        ))),
    )))
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
