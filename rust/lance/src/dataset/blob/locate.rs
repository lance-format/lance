// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::{Array, StructArray};
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use object_store::path::Path;

use super::file::{BlobFile, BlobSource, BlobSourceKey, shared_blob_source};
use crate::dataset::Dataset;
use lance_core::datatypes::{BLOB_DESC_FIELDS, BlobKind, BlobV2Layout, BlobVersion};
use lance_core::utils::blob::{blob_path, validate_managed_reference};
use lance_core::{Error, Result, utils::address::RowAddress};

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

/// Where the payload of one blob v2 descriptor row is stored.
///
/// Parsed from the stable descriptor columns (`kind`, `position`, `size`,
/// `blob_id`, `blob_uri`) so callers never interpret the raw columns, whose
/// meaning depends on `kind`. Resolve it to readable bytes with
/// [`BlobLocator::locate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StoredBlob<'a> {
    /// A zero-length payload; reading it requires no IO.
    Empty,
    /// A range of the data file that holds the descriptor.
    Inline { position: u64, size: u64 },
    /// A range of a sidecar file shared by several rows of the data file.
    Packed {
        blob_id: u32,
        position: u64,
        size: u64,
    },
    /// A whole sidecar file owned by a single row.
    Dedicated { blob_id: u32, size: u64 },
    /// An object outside the dataset. `uri` is absolute when `base_id` is 0
    /// and relative to that registered base otherwise. A `size` of 0 means
    /// the object length is discovered when located.
    External {
        base_id: u32,
        uri: &'a str,
        position: u64,
        size: u64,
    },
    /// An object whose lifecycle is managed by the dataset. `uri` is relative
    /// to `base_id`, or to the descriptor's data file base when `base_id` is
    /// `None` (see [`BlobLocator::managed_base`]).
    Managed {
        base_id: Option<u32>,
        uri: &'a str,
        position: u64,
        size: u64,
    },
}

impl<'a> StoredBlob<'a> {
    /// Parse row `row` of a blob v2 descriptor array, returning `None` for a
    /// null blob.
    pub(super) fn parse(descriptions: &'a StructArray, row: usize) -> Result<Option<Self>> {
        let kinds = descriptions.column(0).as_primitive::<UInt8Type>();
        if descriptions.is_null(row) || kinds.is_null(row) {
            return Ok(None);
        }
        let position = descriptions
            .column(1)
            .as_primitive::<UInt64Type>()
            .value(row);
        let size = descriptions
            .column(2)
            .as_primitive::<UInt64Type>()
            .value(row);
        let blob_id = descriptions
            .column(3)
            .as_primitive::<UInt32Type>()
            .value(row);
        let uri = descriptions.column(4).as_string::<i32>().value(row);
        let kind = BlobKind::try_from(kinds.value(row))?;
        Ok(Some(match kind {
            // Writers store empty payloads as an inline (0, 0) range.
            BlobKind::Inline if position == 0 && size == 0 => Self::Empty,
            BlobKind::Inline => Self::Inline { position, size },
            BlobKind::Packed => Self::Packed {
                blob_id,
                position,
                size,
            },
            BlobKind::Dedicated => Self::Dedicated { blob_id, size },
            BlobKind::External => Self::External {
                base_id: blob_id,
                uri,
                position,
                size,
            },
            BlobKind::Managed | BlobKind::ManagedWithBase => {
                validate_managed_reference(uri, position, size)?;
                Self::Managed {
                    base_id: (kind == BlobKind::ManagedWithBase).then_some(blob_id),
                    uri,
                    position,
                    size,
                }
            }
        }))
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

/// Resolve blob v2 descriptors to lazy handles without materializing their payloads.
pub(in crate::dataset) async fn collect_blob_v2_descriptor_files(
    dataset: &Dataset,
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
    let mut locator = BlobLocator::new(dataset, blob_field_id);
    let mut files = Vec::with_capacity(row_addrs.len());
    for (selection_index, row_addr) in row_addrs.iter().enumerate() {
        let file = match StoredBlob::parse(descriptions, selection_index)? {
            Some(blob) => Some(locator.locate(&blob, *row_addr).await?),
            None => None,
        };
        files.push(file);
    }
    Ok(files)
}

/// The data file that holds a blob field within one fragment.
#[derive(Clone)]
struct BlobReadLocation {
    base_id: Option<u32>,
    data_file_dir: Path,
    data_file_key: String,
    data_file_path: Path,
}

/// Resolves [`StoredBlob`]s of one blob field to physical locations.
///
/// Rows are identified by row address, which selects the fragment data file
/// that inline and sidecar payloads, and implicitly based Managed objects, are
/// relative to. Fragment locations, base roots, object stores, and blob
/// sources are cached, so a locator should be reused across the rows of a
/// batch.
pub(super) struct BlobLocator<'a> {
    dataset: &'a Dataset,
    field_id: u32,
    fragments: HashMap<u32, BlobReadLocation>,
    base_roots: HashMap<u32, Path>,
    stores: HashMap<u32, Arc<ObjectStore>>,
    sources: HashMap<BlobSourceKey, Arc<BlobSource>>,
}

impl<'a> BlobLocator<'a> {
    pub(super) fn new(dataset: &'a Dataset, field_id: u32) -> Self {
        Self {
            dataset,
            field_id,
            fragments: HashMap::new(),
            base_roots: HashMap::new(),
            stores: HashMap::new(),
            sources: HashMap::new(),
        }
    }

    /// Resolve `blob`, stored at `row_addr`, to a lazy handle on its payload.
    ///
    /// External blobs with an unknown size issue a request for the object length.
    pub(super) async fn locate(
        &mut self,
        blob: &StoredBlob<'_>,
        row_addr: u64,
    ) -> Result<BlobFile> {
        match *blob {
            StoredBlob::Empty => self.locate_inline(row_addr, 0, 0).await,
            StoredBlob::Inline { position, size } => {
                self.locate_inline(row_addr, position, size).await
            }
            StoredBlob::Packed {
                blob_id,
                position,
                size,
            } => {
                self.locate_sidecar(row_addr, blob_id, position, size, BlobKind::Packed)
                    .await
            }
            StoredBlob::Dedicated { blob_id, size } => {
                self.locate_sidecar(row_addr, blob_id, 0, size, BlobKind::Dedicated)
                    .await
            }
            StoredBlob::External {
                base_id,
                uri,
                position,
                size,
            } => {
                let (store, path) = if base_id == 0 {
                    let registry = self.dataset.session.store_registry();
                    let params = self
                        .dataset
                        .store_params
                        .as_ref()
                        .map(|p| Arc::new((**p).clone()))
                        .unwrap_or_else(|| Arc::new(ObjectStoreParams::default()));
                    ObjectStore::from_uri_and_params(registry, uri, &params).await?
                } else {
                    let root = self.base_root(Some(base_id))?;
                    let store = self.store(Some(base_id)).await?;
                    (store, join_base_and_relative_path(&root, uri)?)
                };
                let size = if size > 0 {
                    size
                } else {
                    store.size(&path).await?
                };
                Ok(self.file(store, &path, position, size, BlobKind::External, Some(uri)))
            }
            StoredBlob::Managed {
                base_id,
                uri,
                position,
                size,
            } => {
                let base_id = self.managed_base(base_id, row_addr)?;
                let root = self.base_root(base_id)?;
                let store = self.store(base_id).await?;
                let path = join_base_and_relative_path(&root, uri)?;
                Ok(self.file(store, &path, position, size, BlobKind::Managed, Some(uri)))
            }
        }
    }

    /// The base that a Managed blob's `uri` is relative to: the descriptor's
    /// explicit base if it has one, otherwise the base of the data file at
    /// `row_addr`.
    pub(super) fn managed_base(
        &mut self,
        explicit: Option<u32>,
        row_addr: u64,
    ) -> Result<Option<u32>> {
        match explicit {
            Some(base_id) => Ok(Some(base_id)),
            None => Ok(self.data_file(row_addr)?.base_id),
        }
    }

    /// The Packed or Dedicated sidecar `blob_id` of the data file at
    /// `row_addr`, as that data file's base and the sidecar path relative to
    /// the base root.
    pub(super) fn sidecar(&mut self, row_addr: u64, blob_id: u32) -> Result<(Option<u32>, Path)> {
        let location = self.data_file(row_addr)?.clone();
        let path = blob_path(&location.data_file_dir, &location.data_file_key, blob_id);
        let root = self.base_root(location.base_id)?;
        let relative = path
            .prefix_match(&root)
            .ok_or_else(|| Error::internal(format!("Blob source {path} is outside base {root}")))?;
        Ok((location.base_id, Path::from_iter(relative)))
    }

    async fn locate_inline(&mut self, row_addr: u64, position: u64, size: u64) -> Result<BlobFile> {
        let location = self.data_file(row_addr)?.clone();
        let store = self.store(location.base_id).await?;
        Ok(self.file(
            store,
            &location.data_file_path,
            position,
            size,
            BlobKind::Inline,
            None,
        ))
    }

    async fn locate_sidecar(
        &mut self,
        row_addr: u64,
        blob_id: u32,
        position: u64,
        size: u64,
        kind: BlobKind,
    ) -> Result<BlobFile> {
        let location = self.data_file(row_addr)?.clone();
        let store = self.store(location.base_id).await?;
        let path = blob_path(&location.data_file_dir, &location.data_file_key, blob_id);
        Ok(self.file(store, &path, position, size, kind, None))
    }

    fn file(
        &mut self,
        store: Arc<ObjectStore>,
        path: &Path,
        position: u64,
        size: u64,
        kind: BlobKind,
        uri: Option<&str>,
    ) -> BlobFile {
        let source = shared_blob_source(&mut self.sources, store, path);
        BlobFile::with_source(source, position, size, kind, uri.map(str::to_string))
    }

    fn data_file(&mut self, row_addr: u64) -> Result<&BlobReadLocation> {
        let frag_id = RowAddress::from(row_addr).fragment_id();
        if !self.fragments.contains_key(&frag_id) {
            let frag = self.dataset.get_fragment(frag_id as usize).ok_or_else(|| {
                Error::internal(format!(
                    "Blob row address {row_addr} references missing fragment {frag_id}"
                ))
            })?;
            let data_file = frag.data_file_for_field(self.field_id).ok_or_else(|| {
                Error::internal(format!(
                    "Blob field {} has no data file in fragment {frag_id}",
                    self.field_id
                ))
            })?;
            let data_file_dir = self.dataset.data_file_dir(data_file)?;
            let location = BlobReadLocation {
                base_id: data_file.base_id,
                data_file_path: data_file_dir.clone().join(data_file.path.as_str()),
                data_file_key: data_file_key_from_path(data_file.path.as_str()).to_string(),
                data_file_dir,
            };
            self.fragments.insert(frag_id, location);
        }
        Ok(&self.fragments[&frag_id])
    }

    fn base_root(&mut self, base_id: Option<u32>) -> Result<Path> {
        let Some(base_id) = base_id else {
            return Ok(self.dataset.base.clone());
        };
        if let Some(root) = self.base_roots.get(&base_id) {
            return Ok(root.clone());
        }
        let root = self.dataset.blob_base_path(Some(base_id))?;
        self.base_roots.insert(base_id, root.clone());
        Ok(root)
    }

    async fn store(&mut self, base_id: Option<u32>) -> Result<Arc<ObjectStore>> {
        let Some(base_id) = base_id else {
            return Ok(self.dataset.object_store.clone());
        };
        if let Some(store) = self.stores.get(&base_id) {
            return Ok(store.clone());
        }
        let store = self.dataset.object_store(Some(base_id)).await?;
        self.stores.insert(base_id, store.clone());
        Ok(store)
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
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field};
    use lance_core::Error;
    use lance_core::datatypes::{BLOB_V2_DESC_FIELDS, BlobKind};
    use lance_core::utils::tempfile::TempStdDir;
    use rstest::rstest;
    use std::sync::Arc;

    fn descriptor(
        kind: Option<BlobKind>,
        position: u64,
        size: u64,
        blob_id: u32,
        uri: &str,
        valid: bool,
    ) -> StructArray {
        // Older descriptor schemas mark null blobs with a nullable `kind` child.
        let fields = BLOB_V2_DESC_FIELDS
            .iter()
            .map(|field| field.as_ref().clone().with_nullable(true))
            .collect::<Vec<_>>();
        StructArray::try_new(
            fields.into(),
            vec![
                Arc::new(UInt8Array::from(vec![kind.map(|kind| kind as u8)])),
                Arc::new(UInt64Array::from(vec![position])),
                Arc::new(UInt64Array::from(vec![size])),
                Arc::new(UInt32Array::from(vec![blob_id])),
                Arc::new(StringArray::from(vec![uri])),
            ],
            Some(NullBuffer::from(vec![valid])),
        )
        .unwrap()
    }

    #[rstest]
    #[case::null_row(Some(BlobKind::Inline), 1, 2, 0, "", false, None)]
    #[case::null_kind(None, 1, 2, 0, "", true, None)]
    #[case::empty(Some(BlobKind::Inline), 0, 0, 0, "", true, Some(StoredBlob::Empty))]
    #[case::inline_empty_range(
        Some(BlobKind::Inline),
        4,
        0,
        0,
        "",
        true,
        Some(StoredBlob::Inline { position: 4, size: 0 })
    )]
    #[case::inline(
        Some(BlobKind::Inline),
        4,
        8,
        0,
        "",
        true,
        Some(StoredBlob::Inline { position: 4, size: 8 })
    )]
    #[case::packed(
        Some(BlobKind::Packed),
        4,
        8,
        3,
        "",
        true,
        Some(StoredBlob::Packed { blob_id: 3, position: 4, size: 8 })
    )]
    #[case::dedicated(
        Some(BlobKind::Dedicated),
        0,
        8,
        3,
        "",
        true,
        Some(StoredBlob::Dedicated { blob_id: 3, size: 8 })
    )]
    #[case::external(
        Some(BlobKind::External),
        4,
        8,
        2,
        "a/b.bin",
        true,
        Some(StoredBlob::External { base_id: 2, uri: "a/b.bin", position: 4, size: 8 })
    )]
    #[case::managed(
        Some(BlobKind::Managed),
        4,
        8,
        0,
        "_blobs/a.blob",
        true,
        Some(StoredBlob::Managed { base_id: None, uri: "_blobs/a.blob", position: 4, size: 8 })
    )]
    #[case::managed_with_base(
        Some(BlobKind::ManagedWithBase),
        0,
        0,
        2,
        "_blobs/a.blob",
        true,
        Some(StoredBlob::Managed { base_id: Some(2), uri: "_blobs/a.blob", position: 0, size: 0 })
    )]
    fn test_stored_blob_parse(
        #[case] kind: Option<BlobKind>,
        #[case] position: u64,
        #[case] size: u64,
        #[case] blob_id: u32,
        #[case] uri: &str,
        #[case] valid: bool,
        #[case] expected: Option<StoredBlob<'static>>,
    ) {
        let descriptions = descriptor(kind, position, size, blob_id, uri, valid);
        assert_eq!(StoredBlob::parse(&descriptions, 0).unwrap(), expected);
    }

    #[rstest]
    #[case::non_canonical_uri("_blobs/../a.blob", 0, 1)]
    #[case::range_overflow("_blobs/a.blob", u64::MAX, 1)]
    fn test_stored_blob_parse_rejects_invalid_managed(
        #[case] uri: &str,
        #[case] position: u64,
        #[case] size: u64,
    ) {
        let descriptions = descriptor(Some(BlobKind::Managed), position, size, 0, uri, true);
        let error = StoredBlob::parse(&descriptions, 0).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }

    #[tokio::test]
    async fn test_blob_locator_locates_each_kind() {
        let fixture = create_multi_base_blob_v2_fixture(vec![1], None, false).await;
        let dataset = &fixture.dataset;
        let field_id = dataset.schema().field("blob").unwrap().id as u32;
        let data_file = dataset.get_fragment(0).unwrap().metadata.files[0].clone();
        let data_dir = dataset.data_file_dir(&data_file).unwrap();
        let base_root = dataset.blob_base_path(Some(1)).unwrap();
        let sidecar = blob_path(&data_dir, data_file_key_from_path(&data_file.path), 7);
        let external_dir = TempStdDir::default();
        let external_file = external_dir.join("payload.bin");
        std::fs::write(&external_file, b"abcd").unwrap();
        let external_uri = format!("file://{}", external_file.display());
        let data_file_path = data_dir.clone().join(data_file.path.as_str());
        let row_addr = 0;

        let cases = [
            (
                StoredBlob::Empty,
                BlobKind::Inline,
                data_file_path.clone(),
                0,
                0,
                None,
            ),
            (
                StoredBlob::Inline {
                    position: 3,
                    size: 5,
                },
                BlobKind::Inline,
                data_file_path.clone(),
                3,
                5,
                None,
            ),
            (
                StoredBlob::Packed {
                    blob_id: 7,
                    position: 3,
                    size: 5,
                },
                BlobKind::Packed,
                sidecar.clone(),
                3,
                5,
                None,
            ),
            (
                StoredBlob::Dedicated {
                    blob_id: 7,
                    size: 5,
                },
                BlobKind::Dedicated,
                sidecar.clone(),
                0,
                5,
                None,
            ),
            (
                StoredBlob::External {
                    base_id: 1,
                    uri: "a/b.bin",
                    position: 3,
                    size: 5,
                },
                BlobKind::External,
                base_root.clone().join("a").join("b.bin"),
                3,
                5,
                Some("a/b.bin"),
            ),
            (
                StoredBlob::External {
                    base_id: 0,
                    uri: &external_uri,
                    position: 1,
                    size: 0,
                },
                BlobKind::External,
                Path::from_absolute_path(&external_file).unwrap(),
                1,
                4,
                Some(external_uri.as_str()),
            ),
            (
                StoredBlob::Managed {
                    base_id: None,
                    uri: "_blobs/a.blob",
                    position: 3,
                    size: 5,
                },
                BlobKind::Managed,
                base_root.clone().join("_blobs").join("a.blob"),
                3,
                5,
                Some("_blobs/a.blob"),
            ),
            (
                StoredBlob::Managed {
                    base_id: Some(1),
                    uri: "_blobs/a.blob",
                    position: 3,
                    size: 5,
                },
                BlobKind::Managed,
                base_root.clone().join("_blobs").join("a.blob"),
                3,
                5,
                Some("_blobs/a.blob"),
            ),
        ];
        let mut locator = BlobLocator::new(dataset, field_id);
        for (blob, kind, path, position, size, uri) in cases {
            let file = locator.locate(&blob, row_addr).await.unwrap();
            assert_eq!(file.kind(), kind, "{blob:?}");
            assert_eq!(file.data_path(), &path, "{blob:?}");
            assert_eq!(file.position(), position, "{blob:?}");
            assert_eq!(file.size(), size, "{blob:?}");
            assert_eq!(file.uri(), uri, "{blob:?}");
        }

        assert_eq!(locator.managed_base(None, row_addr).unwrap(), Some(1));
        assert_eq!(locator.managed_base(Some(2), row_addr).unwrap(), Some(2));
        let (sidecar_base, relative) = locator.sidecar(row_addr, 7).unwrap();
        assert_eq!(sidecar_base, Some(1));
        assert_eq!(
            Path::from_iter(base_root.parts().chain(relative.parts())),
            sidecar
        );
    }

    #[tokio::test]
    async fn test_blob_locator_rejects_unknown_base() {
        let fixture = create_multi_base_blob_v2_fixture(vec![1], None, false).await;
        let field_id = fixture.dataset.schema().field("blob").unwrap().id as u32;
        let mut locator = BlobLocator::new(&fixture.dataset, field_id);
        for blob in [
            StoredBlob::External {
                base_id: 9,
                uri: "a.bin",
                position: 0,
                size: 1,
            },
            StoredBlob::Managed {
                base_id: Some(9),
                uri: "a.blob",
                position: 0,
                size: 1,
            },
        ] {
            let error = locator.locate(&blob, 0).await.unwrap_err();
            assert!(error.to_string().contains("unknown base_id 9"), "{error}");
        }
    }

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
