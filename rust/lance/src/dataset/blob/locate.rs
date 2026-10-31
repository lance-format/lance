// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{Array, StructArray};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, uri_to_url};
use object_store::path::Path;

use super::file::{BlobFile, BlobSource, BlobSourceKey, shared_blob_source};
use crate::dataset::Dataset;
use lance_core::datatypes::{BLOB_DESC_FIELDS, BlobKind, BlobV2Layout, BlobVersion};
use lance_core::utils::blob::blob_path;
use lance_core::{Error, Result, utils::address::RowAddress};

/// Base-aware physical location metadata used while resolving blob reads.
///
/// This is cached per fragment so repeated rows from the same fragment do not
/// recompute the object store, data directory, and data file key.
#[derive(Clone)]
pub(super) struct BlobReadLocation {
    pub(super) base_id: Option<u32>,
    object_store: Arc<ObjectStore>,
    pub(super) data_file_dir: Path,
    pub(super) data_file_key: String,
    data_file_path: Path,
}

fn blob_descriptor_fields_match(
    actual: &arrow_schema::Fields,
    expected: &arrow_schema::Fields,
) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| {
                actual.name() == expected.name() && actual.data_type() == expected.data_type()
            })
}

pub(super) fn blob_version_from_descriptions(descriptions: &StructArray) -> Result<BlobVersion> {
    let fields = descriptions.fields();
    if blob_descriptor_fields_match(fields, &BLOB_DESC_FIELDS) {
        return Ok(BlobVersion::V1);
    }
    if BlobV2Layout::classify(fields) == Some(BlobV2Layout::Descriptor) {
        return Ok(BlobVersion::V2);
    }
    Err(Error::invalid_input_source(format!(
        "Unrecognized blob descriptions schema: expected v1 descriptor or v2 descriptor layout but got {fields:?}",
    )
    .into()))
}

pub(super) struct BlobV2DescriptorColumns<'a> {
    pub(super) descriptions: &'a StructArray,
    pub(super) kinds: &'a arrow::array::PrimitiveArray<UInt8Type>,
    pub(super) positions: &'a arrow::array::PrimitiveArray<UInt64Type>,
    pub(super) sizes: &'a arrow::array::PrimitiveArray<UInt64Type>,
    pub(super) blob_ids: &'a arrow::array::PrimitiveArray<UInt32Type>,
    pub(super) blob_uris: &'a arrow::array::GenericStringArray<i32>,
}

impl<'a> BlobV2DescriptorColumns<'a> {
    pub(super) fn new(descriptions: &'a StructArray) -> Self {
        Self {
            descriptions,
            kinds: descriptions.column(0).as_primitive::<UInt8Type>(),
            positions: descriptions.column(1).as_primitive::<UInt64Type>(),
            sizes: descriptions.column(2).as_primitive::<UInt64Type>(),
            blob_ids: descriptions.column(3).as_primitive::<UInt32Type>(),
            blob_uris: descriptions.column(4).as_string::<i32>(),
        }
    }

    pub(super) fn managed_base_id(&self, idx: usize) -> Option<u32> {
        (self.kinds.value(idx) == BlobKind::ManagedWithBase as u8).then(|| self.blob_ids.value(idx))
    }

    pub(super) fn is_null_blob(&self, idx: usize) -> bool {
        self.descriptions.is_null(idx) || self.kinds.is_null(idx)
    }
}

/// Resolve blob v1 descriptors without dropping null selection slots.
pub(super) fn collect_blob_files_v1(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &arrow::array::PrimitiveArray<UInt64Type>,
) -> Result<Vec<Option<BlobFile>>> {
    if descriptions.len() != row_addrs.len() {
        return Err(Error::internal(format!(
            "Blob descriptor count {} did not match row address count {}",
            descriptions.len(),
            row_addrs.len()
        )));
    }
    let positions = descriptions.column(0).as_primitive::<UInt64Type>();
    let sizes = descriptions.column(1).as_primitive::<UInt64Type>();
    let mut source_cache = HashMap::<BlobSourceKey, Arc<BlobSource>>::new();
    let mut files = Vec::with_capacity(row_addrs.len());
    for selection_index in 0..row_addrs.len() {
        if descriptions.is_null(selection_index)
            || positions.is_null(selection_index)
            || sizes.is_null(selection_index)
        {
            files.push(None);
            continue;
        }

        let position = positions.value(selection_index);
        let size = sizes.value(selection_index);
        // V1 encodes valid empty blobs as (0, 0) and smuggles null
        // repetition/definition levels through a non-zero zero-sized position.
        if size == 0 && position != 0 {
            files.push(None);
            continue;
        }

        let row_addr = row_addrs.value(selection_index);
        let frag_id = RowAddress::from(row_addr).fragment_id();
        let frag = dataset.get_fragment(frag_id as usize).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob row address {} references missing fragment {}",
                row_addr, frag_id
            ))
        })?;
        let data_file = frag.data_file_for_field(blob_field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "Blob field {} has no data file in fragment {} for row address {}",
                blob_field_id, frag_id, row_addr
            ))
        })?;
        let data_file_path = dataset.data_dir().join(data_file.path.as_str());
        files.push(Some(BlobFile::with_source(
            shared_blob_source(
                &mut source_cache,
                dataset.object_store.clone(),
                &data_file_path,
            ),
            position,
            size,
            BlobKind::Inline,
            None,
        )));
    }
    Ok(files)
}

/// Resolve blob v2 descriptors without dropping null selection slots.
pub(super) async fn collect_blob_files_v2(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &arrow::array::PrimitiveArray<UInt64Type>,
) -> Result<Vec<Option<BlobFile>>> {
    collect_blob_v2_descriptor_files(dataset, blob_field_id, descriptions, row_addrs.values()).await
}

/// Resolve blob v2 descriptors to lazy handles without materializing their payloads.
pub(in crate::dataset) async fn collect_blob_v2_descriptor_files(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    descriptions: &StructArray,
    row_addrs: &[u64],
) -> Result<Vec<Option<BlobFile>>> {
    if descriptions.len() != row_addrs.len() {
        return Err(Error::internal(format!(
            "Blob descriptor count {} did not match row address count {}",
            descriptions.len(),
            row_addrs.len()
        )));
    }
    let columns = BlobV2DescriptorColumns::new(descriptions);
    let mut files = Vec::with_capacity(row_addrs.len());
    let mut read_context = BlobV2ReadContext::new(dataset, blob_field_id);
    for (selection_index, row_addr) in row_addrs.iter().enumerate() {
        files.push(
            read_context
                .collect_file(&columns, selection_index, *row_addr)
                .await?,
        );
    }

    Ok(files)
}

/// Resolved metadata shared by lazy handles and portable reference capture.
pub(super) struct ResolvedBlob {
    object_store: Arc<ObjectStore>,
    path: Path,
    position: u64,
    size: Option<u64>,
    kind: BlobKind,
    uri: Option<String>,
    base_id: Option<u32>,
}

impl ResolvedBlob {
    async fn into_file(
        self,
        source_cache: &mut HashMap<BlobSourceKey, Arc<BlobSource>>,
    ) -> Result<BlobFile> {
        let size = match self.size {
            Some(size) => size,
            None => self.object_store.size(&self.path).await?,
        };
        Ok(BlobFile::with_source(
            shared_blob_source(source_cache, self.object_store, &self.path),
            self.position,
            size,
            self.kind,
            self.uri,
        ))
    }

    pub(super) fn into_reference(
        self,
        dataset: &Dataset,
        context: &crate::blob::reference::BlobSourceContext,
    ) -> Result<crate::blob::reference::BlobRef> {
        let uri = if self.kind == BlobKind::External && self.base_id.is_none() {
            self.uri
                .ok_or_else(|| Error::internal("Resolved external blob has no URI"))?
        } else {
            let root = match self.base_id {
                Some(id) => dataset
                    .manifest()
                    .base_paths
                    .get(&id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!("Blob references unknown base_id {id}"))
                    })?
                    .path
                    .clone(),
                None => dataset.refs.root()?.uri,
            };
            object_uri(dataset, &root, &self.path)?
        };
        context.reference(uri, self.position, self.size, self.kind, self.base_id)
    }
}

fn object_uri(dataset: &Dataset, root: &str, path: &Path) -> Result<String> {
    let root_path = ObjectStore::extract_path_from_uri(dataset.session.store_registry(), root)?;
    let relative = path.prefix_match(&root_path).ok_or_else(|| {
        Error::invalid_input(format!(
            "Resolved blob path {path} is outside source root {root}"
        ))
    })?;
    let mut url = uri_to_url(root)?;
    // Start with the source URI so provider-specific authorities (including
    // Azure filesystem/account names) survive object-path normalization.
    url.path_segments_mut()
        .map_err(|_| Error::invalid_input(format!("Blob source URI cannot contain paths: {root}")))?
        .pop_if_empty()
        .extend(relative.map(|part| part.as_ref().to_string()));
    Ok(url.into())
}

pub(super) struct BlobV2ReadContext {
    dataset: Arc<Dataset>,
    blob_field_id: u32,
    fragment_cache: HashMap<u32, BlobReadLocation>,
    store_cache: HashMap<u32, Arc<ObjectStore>>,
    external_base_path_cache: HashMap<u32, Path>,
    source_cache: HashMap<BlobSourceKey, Arc<BlobSource>>,
}

impl BlobV2ReadContext {
    pub(super) fn new(dataset: &Arc<Dataset>, blob_field_id: u32) -> Self {
        Self {
            dataset: dataset.clone(),
            blob_field_id,
            fragment_cache: HashMap::new(),
            store_cache: HashMap::new(),
            external_base_path_cache: HashMap::new(),
            source_cache: HashMap::new(),
        }
    }

    pub(super) async fn collect_file(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        row_addr: u64,
    ) -> Result<Option<BlobFile>> {
        match self.resolve(columns, idx, row_addr).await? {
            Some(blob) => Ok(Some(blob.into_file(&mut self.source_cache).await?)),
            None => Ok(None),
        }
    }

    pub(super) async fn capture_reference(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        row_addr: u64,
        context: &crate::blob::reference::BlobSourceContext,
    ) -> Result<Option<crate::blob::reference::BlobRef>> {
        if columns.is_null_blob(idx) {
            return Ok(None);
        }
        let kind = BlobKind::try_from(columns.kinds.value(idx))?;
        if kind == BlobKind::External && columns.blob_ids.value(idx) == 0 {
            // The URI is already absolute. Opening a store is unnecessary for
            // transport and can perform network I/O, such as S3 region discovery.
            let size = columns.sizes.value(idx);
            return context
                .reference(
                    columns.blob_uris.value(idx).to_string(),
                    columns.positions.value(idx),
                    (size > 0).then_some(size),
                    kind,
                    None,
                )
                .map(Some);
        }
        self.resolve(columns, idx, row_addr)
            .await?
            .map(|resolved| resolved.into_reference(&self.dataset, context))
            .transpose()
    }

    pub(super) async fn resolve(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        row_addr: u64,
    ) -> Result<Option<ResolvedBlob>> {
        if columns.is_null_blob(idx) {
            return Ok(None);
        }

        let kind = BlobKind::try_from(columns.kinds.value(idx))?;
        let blob = match kind {
            BlobKind::Inline => self.resolve_inline(columns, idx, row_addr).await?,
            BlobKind::Dedicated => self.resolve_dedicated(columns, idx, row_addr).await?,
            BlobKind::Packed => self.resolve_packed(columns, idx, row_addr).await?,
            BlobKind::External => self.resolve_external(columns, idx).await?,
            BlobKind::Managed | BlobKind::ManagedWithBase => {
                self.resolve_managed(columns, idx, row_addr).await?
            }
        };

        Ok(Some(blob))
    }

    pub(super) async fn blob_read_location(&mut self, row_addr: u64) -> Result<BlobReadLocation> {
        resolve_blob_read_location(
            &self.dataset,
            self.blob_field_id,
            row_addr,
            &mut self.fragment_cache,
            &mut self.store_cache,
        )
        .await
    }

    async fn resolve_inline(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        row_addr: u64,
    ) -> Result<ResolvedBlob> {
        let position = columns.positions.value(idx);
        let size = columns.sizes.value(idx);
        let location = self.blob_read_location(row_addr).await?;
        Ok(ResolvedBlob {
            object_store: location.object_store,
            path: location.data_file_path,
            position,
            size: Some(size),
            kind: BlobKind::Inline,
            uri: None,
            base_id: location.base_id,
        })
    }

    async fn resolve_dedicated(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        row_addr: u64,
    ) -> Result<ResolvedBlob> {
        let blob_id = columns.blob_ids.value(idx);
        let size = columns.sizes.value(idx);
        let location = self.blob_read_location(row_addr).await?;
        let path = blob_path(&location.data_file_dir, &location.data_file_key, blob_id);
        Ok(ResolvedBlob {
            object_store: location.object_store,
            path,
            position: 0,
            size: Some(size),
            kind: BlobKind::Dedicated,
            uri: None,
            base_id: location.base_id,
        })
    }

    async fn resolve_packed(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        row_addr: u64,
    ) -> Result<ResolvedBlob> {
        let blob_id = columns.blob_ids.value(idx);
        let size = columns.sizes.value(idx);
        let position = columns.positions.value(idx);
        let location = self.blob_read_location(row_addr).await?;
        let path = blob_path(&location.data_file_dir, &location.data_file_key, blob_id);
        Ok(ResolvedBlob {
            object_store: location.object_store,
            path,
            position,
            size: Some(size),
            kind: BlobKind::Packed,
            uri: None,
            base_id: location.base_id,
        })
    }

    pub(super) async fn resolve_managed(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
        row_addr: u64,
    ) -> Result<ResolvedBlob> {
        let uri = columns.blob_uris.value(idx);
        let position = columns.positions.value(idx);
        let size = columns.sizes.value(idx);
        let relative = lance_core::utils::blob::validate_managed_reference(uri, position, size)?;
        let base_id = resolve_managed_base_id(
            &self.dataset,
            self.blob_field_id,
            row_addr,
            columns.managed_base_id(idx),
        )?;
        let (root, store) = if let Some(id) = base_id {
            let root = if let Some(root) = self.external_base_path_cache.get(&id) {
                root.clone()
            } else {
                let root = self.dataset.blob_base_path(Some(id))?;
                self.external_base_path_cache.insert(id, root.clone());
                root
            };
            let store = if let Some(store) = self.store_cache.get(&id) {
                store.clone()
            } else {
                let store = self.dataset.object_store(Some(id)).await?;
                self.store_cache.insert(id, store.clone());
                store
            };
            (root, store)
        } else {
            (self.dataset.base.clone(), self.dataset.object_store.clone())
        };
        let path = join_base_and_relative_path(&root, relative.as_ref())?;
        Ok(ResolvedBlob {
            object_store: store,
            path,
            position,
            size: Some(size),
            kind: BlobKind::Managed,
            uri: Some(uri.to_string()),
            base_id,
        })
    }

    async fn resolve_external(
        &mut self,
        columns: &BlobV2DescriptorColumns<'_>,
        idx: usize,
    ) -> Result<ResolvedBlob> {
        let uri_or_path = columns.blob_uris.value(idx).to_string();
        let position = columns.positions.value(idx);
        let size = columns.sizes.value(idx);
        let base_id = columns.blob_ids.value(idx);
        let (object_store, path) = if base_id == 0 {
            let registry = self.dataset.session.store_registry();
            let params = self
                .dataset
                .store_params
                .as_ref()
                .map(|p| Arc::new((**p).clone()))
                .unwrap_or_else(|| Arc::new(ObjectStoreParams::default()));
            ObjectStore::from_uri_and_params(registry, &uri_or_path, &params).await?
        } else {
            let object_store = if let Some(store) = self.store_cache.get(&base_id) {
                store.clone()
            } else {
                let store = self.dataset.object_store(Some(base_id)).await?;
                self.store_cache.insert(base_id, store.clone());
                store
            };
            let base_root = if let Some(path) = self.external_base_path_cache.get(&base_id) {
                path.clone()
            } else {
                let base = self
                    .dataset
                    .manifest
                    .base_paths
                    .get(&base_id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "External blob references unknown base_id {}",
                            base_id
                        ))
                    })?;
                let path = base.extract_path(self.dataset.session.store_registry())?;
                self.external_base_path_cache.insert(base_id, path.clone());
                path
            };
            let path = join_base_and_relative_path(&base_root, &uri_or_path)?;
            (object_store, path)
        };
        Ok(ResolvedBlob {
            object_store,
            path,
            position,
            size: (size > 0).then_some(size),
            kind: BlobKind::External,
            uri: Some(uri_or_path),
            base_id: (base_id != 0).then_some(base_id),
        })
    }
}

pub(super) fn join_base_and_relative_path(base: &Path, relative_path: &str) -> Result<Path> {
    let relative = Path::parse(relative_path).map_err(|e| {
        Error::invalid_input(format!(
            "Invalid relative external blob path '{}': {}",
            relative_path, e
        ))
    })?;
    Ok(Path::from_iter(base.parts().chain(relative.parts())))
}

/// Resolve the physical read location for a blob row in a base-aware way.
///
/// Given a `row_addr`, this helper locates the owning fragment and the blob field's
/// data file, then returns the concrete object store and paths needed to read blob
/// bytes correctly under multi-base datasets.
///
/// It uses two caller-provided caches:
/// - `fragment_cache` memoizes per-fragment path metadata (`data_file_dir`,
///   `data_file_path`, and `data_file_key`) plus the resolved store.
/// - `store_cache` memoizes `base_id -> ObjectStore` so multiple fragments that
///   share the same base do not repeat async store resolution.
async fn resolve_blob_read_location(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    row_addr: u64,
    fragment_cache: &mut HashMap<u32, BlobReadLocation>,
    store_cache: &mut HashMap<u32, Arc<ObjectStore>>,
) -> Result<BlobReadLocation> {
    let frag_id = RowAddress::from(row_addr).fragment_id();
    if let Some(location) = fragment_cache.get(&frag_id) {
        return Ok(location.clone());
    }

    let frag = dataset
        .get_fragment(frag_id as usize)
        .ok_or_else(|| Error::internal("Fragment not found".to_string()))?;
    let data_file = frag
        .data_file_for_field(blob_field_id)
        .ok_or_else(|| Error::internal("Data file not found for blob field".to_string()))?;
    let data_file_dir = dataset.data_file_dir(data_file)?;
    let data_file_path = data_file_dir.clone().join(data_file.path.as_str());
    let data_file_key = data_file_key_from_path(data_file.path.as_str()).to_string();

    let object_store = if let Some(base_id) = data_file.base_id {
        if let Some(store) = store_cache.get(&base_id) {
            store.clone()
        } else {
            let store = dataset.object_store(Some(base_id)).await?;
            store_cache.insert(base_id, store.clone());
            store
        }
    } else {
        dataset.object_store.clone()
    };

    let location = BlobReadLocation {
        base_id: data_file.base_id,
        object_store,
        data_file_dir,
        data_file_key,
        data_file_path,
    };
    fragment_cache.insert(frag_id, location.clone());
    Ok(location)
}

pub(super) fn resolve_managed_base_id(
    dataset: &Dataset,
    field_id: u32,
    row_addr: u64,
    explicit: Option<u32>,
) -> Result<Option<u32>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    let fragment = dataset
        .get_fragment(RowAddress::from(row_addr).fragment_id() as usize)
        .ok_or_else(|| Error::internal("Managed descriptor fragment is missing"))?;
    let file = fragment
        .data_file_for_field(field_id)
        .ok_or_else(|| Error::internal("Managed descriptor data file is missing"))?;
    Ok(file.base_id)
}

pub(in crate::dataset) fn data_file_key_from_path(path: &str) -> &str {
    let filename = path.rsplit('/').next().unwrap_or(path);
    filename.strip_suffix(".lance").unwrap_or(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dataset::blob::test_util::{BlobTestFixture, create_multi_base_blob_v2_fixture};
    use crate::dataset::blob::write::INLINE_MAX;
    use arrow_array::{ArrayRef, StringArray, StructArray, UInt8Array, UInt32Array, UInt64Array};
    use arrow_schema::{DataType, Field};
    use lance_core::Error;
    use lance_core::datatypes::BlobKind;
    use std::sync::Arc;

    #[test]
    fn test_blob_version_rejects_malformed_v2_descriptor_layout() {
        let descriptions = StructArray::try_new(
            vec![
                Field::new("kind", DataType::UInt8, false),
                Field::new("position", DataType::UInt64, false),
                Field::new("size", DataType::UInt32, false),
                Field::new("blob_id", DataType::UInt32, false),
                Field::new("blob_uri", DataType::Utf8, false),
            ]
            .into(),
            vec![
                Arc::new(UInt8Array::from(vec![BlobKind::Inline as u8])),
                Arc::new(UInt64Array::from(vec![0])),
                Arc::new(UInt32Array::from(vec![0])),
                Arc::new(UInt32Array::from(vec![0])),
                Arc::new(StringArray::from(vec![""])),
            ],
            None,
        )
        .unwrap();

        let error = blob_version_from_descriptions(&descriptions).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("expected v1 descriptor or v2 descriptor layout")
        );
    }

    #[tokio::test]
    async fn test_collect_blob_files_v1_rejects_missing_fragment() {
        let fixture = BlobTestFixture::new().await;
        let blob_field_id =
            fixture.dataset.schema().project(&["blobs"]).unwrap().fields[0].id as u32;
        let descriptions = StructArray::from(vec![
            (
                Arc::new(Field::new("position", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![1])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("size", DataType::UInt64, false)),
                Arc::new(UInt64Array::from(vec![3])) as ArrayRef,
            ),
        ]);
        let row_addrs = UInt64Array::from(vec![(999_u64 << 32) | 7]);

        let err = collect_blob_files_v1(&fixture.dataset, blob_field_id, &descriptions, &row_addrs)
            .unwrap_err();

        assert!(err.to_string().contains("references missing fragment"));
    }

    #[test]
    fn test_data_file_key_from_path() {
        assert_eq!(data_file_key_from_path("data/abc.lance"), "abc");
        assert_eq!(data_file_key_from_path("abc.lance"), "abc");
        assert_eq!(data_file_key_from_path("nested/path/xyz"), "xyz");
    }

    #[tokio::test]
    async fn test_take_blob_v2_from_non_default_base_packed() {
        let fixture =
            create_multi_base_blob_v2_fixture(vec![0x5A; INLINE_MAX + 4096], None, true).await;

        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&[0], "blob")
            .await
            .unwrap();

        assert_eq!(blobs.len(), 1);
        let blob = blobs[0].as_ref().unwrap();
        assert_eq!(blob.kind(), BlobKind::Packed);
        assert_eq!(
            blob.read().await.unwrap().as_ref(),
            fixture.expected.as_slice()
        );
    }

    #[tokio::test]
    async fn test_take_blob_v2_from_data_only_base() {
        let fixture =
            create_multi_base_blob_v2_fixture(vec![0x6B; INLINE_MAX + 2048], None, false).await;

        let blobs = fixture
            .dataset
            .take_blobs_by_indices(&[0], "blob")
            .await
            .unwrap();

        assert_eq!(blobs.len(), 1);
        let blob = blobs[0].as_ref().unwrap();
        assert_eq!(blob.kind(), BlobKind::Packed);
        assert_eq!(
            blob.read().await.unwrap().as_ref(),
            fixture.expected.as_slice()
        );
    }
}
