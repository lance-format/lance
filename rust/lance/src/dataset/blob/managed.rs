// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use arrow_array::{ArrayRef, StructArray};
use arrow_schema::Field as ArrowField;
use lance_datafusion::utils::from_datafusion_stream;
use object_store::path::Path;

use super::locate::join_base_and_relative_path;
use crate::dataset::Dataset;
use lance_core::{Error, Result};
use lance_io::object_store::{ObjectStore, ObjectStoreRegistry};
use lance_table_runtime::blob::field_contains_blob;

pub(super) async fn managed_references(
    dataset: &Dataset,
    mut scan: crate::dataset::scanner::Scanner,
    include_registered_sidecars: bool,
) -> Result<HashSet<(Option<u32>, String)>> {
    let names = dataset
        .schema()
        .fields
        .iter()
        .filter(|field| field_contains_blob(field))
        .map(|field| field.name.as_str())
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Ok(HashSet::new());
    }
    scan.project(&names)?.with_row_address();
    let stream = from_datafusion_stream(scan.try_into_stream().await?.into());
    lance_table_runtime::blob::managed_references(
        &dataset.table_view().to_table(),
        stream,
        include_registered_sidecars,
    )
    .await
}

/// Where a managed blob base sits relative to the dataset being cleaned.
#[derive(Clone)]
enum BaseResolution {
    Resolved(Path),
    /// Opened store is not the dataset being cleaned.
    Foreign,
    /// The base did not open.
    Unresolved,
}

/// Blob paths this cleanup owner may still reference.
///
/// Keep a resolved path in `data/` or `_blobs/`. A base in `owner_base_ids`
/// with no such path joins onto the owner's root. An opened foreign store
/// contributes no path.
pub(in crate::dataset) async fn managed_paths(
    dataset: &Dataset,
    owner: &Dataset,
    owner_base_ids: &BTreeSet<Option<u32>>,
    scan: crate::dataset::scanner::Scanner,
) -> Result<HashSet<Path>> {
    let references = managed_references(dataset, scan, false).await?;
    let mut paths = HashSet::with_capacity(references.len());
    let mut resolution_by_base = HashMap::new();
    let primary = primary_root(dataset, owner);
    for (base_id, uri) in references {
        let root = match base_id {
            Some(id) => {
                registered_root(dataset, owner, id, owner_base_ids, &mut resolution_by_base).await?
            }
            None => primary.clone(),
        };
        match &root {
            BaseResolution::Resolved(root) => {
                let path = join_base_and_relative_path(root, &uri)?;
                if crate::dataset::cleanup::in_blob_deletion_scope(&owner.table_state.base, &path) {
                    paths.insert(path);
                    continue;
                }
            }
            BaseResolution::Foreign => continue,
            BaseResolution::Unresolved => {}
        }
        if owner_base_ids.contains(&base_id) {
            paths.insert(join_base_and_relative_path(&owner.table_state.base, &uri)?);
        }
    }
    Ok(paths)
}

fn primary_root(dataset: &Dataset, owner: &Dataset) -> BaseResolution {
    let registry = dataset.session.store_registry();
    if may_be_same_store(
        registry.as_ref(),
        &owner.table_state.object_store,
        dataset.table_state.object_store.scheme(),
        &dataset.table_state.object_store.store_prefix,
    ) {
        BaseResolution::Resolved(dataset.table_state.base.clone())
    } else {
        BaseResolution::Foreign
    }
}

async fn registered_root(
    dataset: &Dataset,
    owner: &Dataset,
    base_id: u32,
    owner_base_ids: &BTreeSet<Option<u32>>,
    resolution_by_base: &mut HashMap<u32, BaseResolution>,
) -> Result<BaseResolution> {
    if let Some(root) = resolution_by_base.get(&base_id) {
        return Ok(root.clone());
    }
    let registry = dataset.session.store_registry();
    let root = match dataset.object_store(Some(base_id)).await {
        Ok(base_store) => {
            if !may_be_same_store(
                registry.as_ref(),
                &owner.table_state.object_store,
                base_store.scheme(),
                &base_store.store_prefix,
            ) {
                BaseResolution::Foreign
            } else {
                let root = dataset
                    .manifest
                    .base_paths
                    .get(&base_id)
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Managed blob references unknown base_id {base_id}"
                        ))
                    })?
                    .extract_path(registry)?;
                BaseResolution::Resolved(root)
            }
        }
        // An inherited base may be unavailable here. Fall back to lineage.
        Err(_) if owner_base_ids.contains(&Some(base_id)) => BaseResolution::Unresolved,
        Err(error) => return Err(error),
    };
    resolution_by_base.insert(base_id, root.clone());
    Ok(root)
}

/// Whether `owner` may be the store named by `scheme` and `prefix`.
///
/// `prefix` comes from the opened store and includes an Azure account.
/// An unknown provider or a prefix that is not `scheme$location` matches.
pub(in crate::dataset) fn may_be_same_store(
    registry: &ObjectStoreRegistry,
    owner: &ObjectStore,
    scheme: &str,
    prefix: &str,
) -> bool {
    if owner.store_prefix == prefix {
        return true;
    }
    let local = |scheme: &str| matches!(scheme, "file" | "file-object-store" | "file+uring");
    if local(owner.scheme()) && local(scheme) {
        return true;
    }
    if local(owner.scheme()) || local(scheme) {
        return false;
    }
    let Some(owner_provider) = registry.get_provider(owner.scheme()) else {
        return true;
    };
    let Some(provider) = registry.get_provider(scheme) else {
        return true;
    };
    if !Arc::ptr_eq(&owner_provider, &provider) {
        return false;
    }
    match (
        prefix_location(&owner.store_prefix, owner.scheme()),
        prefix_location(prefix, scheme),
    ) {
        (Some(owner_location), Some(location)) => owner_location == location,
        _ => true,
    }
}

fn prefix_location<'a>(prefix: &'a str, scheme: &str) -> Option<&'a str> {
    prefix.strip_prefix(scheme)?.strip_prefix('$')
}

/// Preserve Managed references and adopt Packed/Dedicated objects in place.
/// Only Inline payloads are copied. Source bases are made explicit when needed
/// to preserve their identity at the destination.
/// Convert scanned descriptors into prepared write values without reading
/// out-of-line payloads. `row_addrs` must identify each descriptor's source
/// row in `dataset`; `field_id` identifies its blob leaf. Inline values are
/// read from their source data file. The caller must retain the source snapshot
/// until the new references have been committed.
///
/// `field` supplies the destination name, nullability and metadata. The returned
/// array has the prepared Blob v2 layout and can be interleaved with values from
/// [`BlobDescriptorArrayBuilder`](crate::blob::BlobDescriptorArrayBuilder) before writing.
///
/// `target_base_id` is the destination data file's base in the same dataset:
/// `None` for primary storage, or `Some(id)` for a registered base. Write the
/// returned array only to that base, using the same base registry. Reusing
/// primary-storage objects in a registered base requires a registered base
/// whose URI matches the dataset URI; otherwise this function returns an error.
///
/// ```
/// # use std::sync::Arc;
/// # use arrow_array::StructArray;
/// # use arrow_schema::Field;
/// # use lance::Dataset;
/// # async fn preserve(dataset: Arc<Dataset>, values: &StructArray, rows: &[u64], field: &Field) -> lance::Result<()> {
/// let prepared = lance::dataset::preserve_managed_descriptors(
///     &dataset, 0, values, rows, field, None,
/// ).await?;
/// # let _ = prepared;
/// # Ok(()) }
/// ```
pub async fn preserve_managed_descriptors(
    dataset: &Arc<Dataset>,
    field_id: u32,
    values: &StructArray,
    row_addrs: &[u64],
    field: &ArrowField,
    target_base_id: Option<u32>,
) -> Result<ArrayRef> {
    let table = dataset.table_view().to_table();
    Box::pin(lance_table_runtime::blob::preserve_managed_descriptors(
        &table,
        field_id,
        values,
        row_addrs,
        field,
        target_base_id,
    ))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dataset::blob::test_util::managed_blob_schema;
    use crate::{
        Dataset,
        blob::{BlobArrayBuilder, blob_field},
        dataset::WriteParams,
    };
    use arrow_array::RecordBatch;
    use arrow_array::RecordBatchIterator;
    use arrow_schema::Schema;
    use lance_arrow::{
        ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
        BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_V2_EXT_NAME,
    };
    use lance_core::datatypes::BlobKind;
    use lance_core::{Error, utils::tempfile::TempStrDir};
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::storage_options::StorageOptionsAccessor;
    use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
    use lance_table::feature_flags::FLAG_MANAGED_BLOBS;
    use lance_table::format::BasePath;
    use rstest::rstest;
    use std::collections::HashMap;
    use std::sync::Arc;
    use uuid::Uuid;

    #[rstest]
    #[case::inline(6, BlobKind::Inline)]
    #[case::packed(16, BlobKind::Packed)]
    #[case::dedicated(32, BlobKind::Dedicated)]
    #[tokio::test]
    async fn managed_write_read(
        #[case] size: usize,
        #[case] kind: BlobKind,
        #[values(false, true)] enabled: bool,
        #[values(false, true)] maximum_base_id: bool,
        #[values(false, true)] primary_write: bool,
        #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let payload = vec![42u8; size];
        let mut blobs = BlobArrayBuilder::new(3);
        blobs.push_bytes(&payload).unwrap();
        blobs.push_bytes(b"").unwrap();
        blobs.push_null().unwrap();
        let field = blob_field("blob", true).with_metadata(HashMap::from([
            (ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string()),
            (
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY.to_string(),
                "8".to_string(),
            ),
            (
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY.to_string(),
                "24".to_string(),
            ),
        ]));
        let schema = Arc::new(if enabled {
            managed_blob_schema(vec![field])
        } else {
            Schema::new(vec![field])
        });
        let batch = RecordBatch::try_new(schema.clone(), vec![blobs.finish().unwrap()]).unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)], schema),
                &test_dir,
                Some(WriteParams {
                    data_storage_version: Some(version),
                    initial_bases: maximum_base_id.then(|| {
                        vec![BasePath::new(
                            u32::MAX,
                            test_dir.to_string(),
                            Some("payload".to_string()),
                            true,
                        )]
                    }),
                    target_bases: (maximum_base_id && !primary_write).then_some(vec![u32::MAX]),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        let flag = FLAG_MANAGED_BLOBS;
        let expected_flags = if enabled { flag } else { 0 };
        assert_eq!(dataset.manifest.reader_feature_flags & flag, expected_flags);
        assert_eq!(dataset.manifest.writer_feature_flags & flag, expected_flags);
        for (_, uri) in managed_references(&dataset, dataset.scan(), false)
            .await
            .unwrap()
        {
            assert!(uri.starts_with("_blobs/"), "{uri}");
            assert_eq!(uri.split('/').count(), 2);
            Uuid::parse_str(uri.trim_start_matches("_blobs/").trim_end_matches(".blob")).unwrap();
        }
        let id = (maximum_base_id && !primary_write).then_some(u32::MAX);
        assert_eq!(
            dataset.manifest.base_paths.len(),
            usize::from(maximum_base_id)
        );
        for fragment in dataset.get_fragments() {
            assert_eq!(fragment.metadata().files[0].base_id, id);
        }
        let values = dataset
            .take_blobs_by_indices(&[0, 1, 2], "blob")
            .await
            .unwrap();
        let kind = if enabled && kind != BlobKind::Inline {
            BlobKind::Managed
        } else {
            kind
        };
        assert_eq!(values[0].as_ref().unwrap().kind(), kind);
        assert_eq!(
            values[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            payload
        );
        assert!(values[1].as_ref().unwrap().read().await.unwrap().is_empty());
        assert!(values[2].is_none());
        if kind == BlobKind::Managed
            && let Some(id) = id
        {
            let mut missing_base = dataset.as_ref().clone();
            let manifest = Arc::make_mut(&mut missing_base.manifest);
            // The data file lives at the dataset root even when the writer
            // selected an explicit base. Isolate the descriptor's base lookup.
            for fragment in Arc::make_mut(&mut manifest.fragments) {
                for file in fragment.referenced_lance_files_mut() {
                    file.base_id = None;
                }
            }
            manifest.base_paths.clear();
            let error = Arc::new(missing_base)
                .take_blobs_by_indices(&[0], "blob")
                .await
                .err()
                .unwrap();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(
                error.to_string().contains(&format!("unknown base_id {id}")),
                "{error}"
            );
        }
    }

    async fn opened_store(uri: &str, options: &[(&str, &str)]) -> Arc<ObjectStore> {
        let options = options
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                options,
            ))),
            ..Default::default()
        };
        ObjectStore::from_uri_and_params(Arc::new(ObjectStoreRegistry::default()), uri, &params)
            .await
            .unwrap()
            .0
    }

    #[tokio::test]
    async fn local_scheme_alias_names_one_store() {
        let registry = ObjectStoreRegistry::default();
        let owner = opened_store("file:///tmp/orders", &[]).await;
        let base = opened_store("file-object-store:///tmp/orders/_blobs", &[]).await;
        assert!(super::may_be_same_store(
            &registry,
            &owner,
            base.scheme(),
            &base.store_prefix
        ));
    }

    #[cfg(feature = "aws")]
    #[tokio::test]
    async fn distinct_buckets_are_different_stores() {
        let registry = ObjectStoreRegistry::default();
        let options = [
            ("region", "us-east-1"),
            ("aws_access_key_id", "test"),
            ("aws_secret_access_key", "test"),
        ];
        let owner = opened_store("s3://bucket-a/orders", &options).await;
        let same_bucket = opened_store("s3://bucket-a/orders/_blobs", &options).await;
        let other_bucket = opened_store("s3://bucket-b/orders/_blobs", &options).await;
        let dynamo_alias = opened_store("s3+ddb://bucket-a/orders/_blobs", &options).await;
        assert!(super::may_be_same_store(
            &registry,
            &owner,
            same_bucket.scheme(),
            &same_bucket.store_prefix
        ));
        assert!(!super::may_be_same_store(
            &registry,
            &owner,
            other_bucket.scheme(),
            &other_bucket.store_prefix
        ));
        assert!(super::may_be_same_store(
            &registry,
            &owner,
            dynamo_alias.scheme(),
            &dynamo_alias.store_prefix
        ));
    }

    #[cfg(feature = "azure")]
    #[tokio::test]
    async fn azure_account_comes_from_storage_options() {
        let registry = ObjectStoreRegistry::default();
        let owner = opened_store("az://container/orders", &[("account_name", "account-a")]).await;
        let same_account = opened_store(
            "az://container/orders/_blobs",
            &[("account_name", "account-a")],
        )
        .await;
        let other_account = opened_store(
            "az://container/orders/_blobs",
            &[("account_name", "account-b")],
        )
        .await;
        let scheme_alias = opened_store(
            "abfss://container/orders/_blobs",
            &[("account_name", "account-a")],
        )
        .await;
        assert_eq!(owner.store_prefix, "az$container@account-a");
        assert_eq!(same_account.store_prefix, "az$container@account-a");
        assert_eq!(other_account.store_prefix, "az$container@account-b");
        assert!(super::may_be_same_store(
            &registry,
            &owner,
            same_account.scheme(),
            &same_account.store_prefix
        ));
        assert!(!super::may_be_same_store(
            &registry,
            &owner,
            other_account.scheme(),
            &other_account.store_prefix
        ));
        assert!(super::may_be_same_store(
            &registry,
            &owner,
            scheme_alias.scheme(),
            &scheme_alias.store_prefix
        ));
    }
}
