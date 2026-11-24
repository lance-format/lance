// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashSet, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_buffer::ArrowNativeType;
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField};
use lance_arrow::FieldExt;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, uri_to_url};
use object_store::path::Path;
use roaring::RoaringBitmap;
use url::Url;

use crate::blob::{blob_v2_shape_error, validate_prepared_blob_array};
use crate::session::UnregisteredExternalBlobPolicy;
use lance_core::datatypes::BlobV2Layout;
use lance_core::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResolvedExternalBase {
    pub base_id: u32,
    pub relative_path: String,
}

#[derive(Clone, Debug)]
pub struct ExternalBaseCandidate {
    pub base_id: u32,
    pub store_prefix: String,
    pub base_path: Path,
    pub store_params: ObjectStoreParams,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ExternalBaseResolver {
    candidates: Vec<ExternalBaseCandidate>,
    pub(crate) registered_base_ids: RoaringBitmap,
    pub(crate) unregistered_external_blob_policy: UnregisteredExternalBlobPolicy,
    store_registry: Arc<ObjectStoreRegistry>,
}

impl ExternalBaseResolver {
    pub(crate) fn new(
        candidates: Vec<ExternalBaseCandidate>,
        store_registry: Arc<ObjectStoreRegistry>,
    ) -> Self {
        Self {
            registered_base_ids: candidates.iter().map(|base| base.base_id).collect(),
            candidates,
            store_registry,
            unregistered_external_blob_policy: UnregisteredExternalBlobPolicy::Allow,
        }
    }

    pub(super) async fn resolve_external_uri(
        &self,
        uri: &str,
    ) -> Result<Option<ResolvedExternalBase>> {
        let url = uri_to_url(uri)?;
        // A missing provider cannot resolve any registered base. Let controlled
        // reference writes report the session policy instead of a registry error.
        if self.unregistered_external_blob_policy == UnregisteredExternalBlobPolicy::Deny
            && self.store_registry.get_provider(url.scheme()).is_none()
        {
            return Ok(None);
        }
        let uri_path = ObjectStore::extract_path_from_uri(self.store_registry.clone(), uri)?;

        let mut best_match: Option<(usize, ResolvedExternalBase)> = None;
        let provider = self
            .store_registry
            .get_provider(url.scheme())
            .ok_or_else(|| Error::invalid_input(format!("Unknown scheme: {}", url.scheme())))?;
        for candidate in &self.candidates {
            let Some(uri_store_prefix) = provider.calculate_object_store_prefix_for_match(
                &url,
                candidate.store_params.storage_options(),
            )?
            else {
                continue;
            };
            if candidate.store_prefix != uri_store_prefix {
                continue;
            }
            let Some(relative_parts) = uri_path.prefix_match(&candidate.base_path) else {
                continue;
            };
            let relative_path = Path::from_iter(relative_parts);
            if relative_path.as_ref().is_empty() {
                continue;
            }
            let prefix_len = candidate.base_path.parts().count();
            if best_match
                .as_ref()
                .map(|(current_len, _)| prefix_len > *current_len)
                .unwrap_or(true)
            {
                best_match = Some((
                    prefix_len,
                    ResolvedExternalBase {
                        base_id: candidate.base_id,
                        relative_path: relative_path.to_string(),
                    },
                ));
            }
        }

        Ok(best_match.map(|(_, matched)| matched))
    }
}

pub(super) fn arrow_field_contains_blob_v2(field: &ArrowField) -> bool {
    if field.is_blob_v2() {
        return true;
    }
    match field.data_type() {
        ArrowDataType::Struct(children) => children
            .iter()
            .any(|child| arrow_field_contains_blob_v2(child)),
        ArrowDataType::List(child) | ArrowDataType::LargeList(child) => {
            arrow_field_contains_blob_v2(child)
        }
        _ => false,
    }
}

pub(super) fn collect_external_blob_uris(
    field: &ArrowField,
    array: &ArrayRef,
    selected_rows: &[bool],
    field_path: &str,
    external_uris: &mut Vec<(String, String)>,
) -> Result<()> {
    if !arrow_field_contains_blob_v2(field) {
        return Ok(());
    }
    if array.len() != selected_rows.len() {
        return Err(Error::internal(format!(
            "Blob field '{}' row count {} did not match selection length {}",
            field_path,
            array.len(),
            selected_rows.len()
        )));
    }

    if field.is_blob_v2() {
        let struct_array = array.as_struct();
        match BlobV2Layout::classify(struct_array.fields()) {
            Some(BlobV2Layout::Prepared) => {
                validate_prepared_blob_array(field, array)?;
                return Ok(());
            }
            Some(BlobV2Layout::Logical) => {}
            _ => {
                return Err(blob_v2_shape_error(
                    field,
                    &[BlobV2Layout::Logical, BlobV2Layout::Prepared],
                ));
            }
        }
        let data_column = struct_array
            .column_by_name("data")
            .ok_or_else(|| Error::invalid_input("Blob struct missing `data` field"))?
            .as_binary::<i64>();
        let uri_column = struct_array
            .column_by_name("uri")
            .ok_or_else(|| Error::invalid_input("Blob struct missing `uri` field"))?
            .as_string::<i32>();
        let position_column = struct_array
            .column_by_name("position")
            .map(|column| column.as_primitive::<UInt64Type>());
        let size_column = struct_array
            .column_by_name("size")
            .map(|column| column.as_primitive::<UInt64Type>());
        for (row_idx, is_selected) in selected_rows.iter().copied().enumerate() {
            if !is_selected || struct_array.is_null(row_idx) {
                continue;
            }
            let has_data = data_column.is_valid(row_idx);
            let has_uri = uri_column.is_valid(row_idx);
            let has_position = position_column
                .as_ref()
                .is_some_and(|column| column.is_valid(row_idx));
            let has_size = size_column
                .as_ref()
                .is_some_and(|column| column.is_valid(row_idx));
            if has_position != has_size {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {row_idx} must set both `position` and `size`, or neither",
                    field_path
                )));
            }
            if has_position && !has_uri {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {row_idx} sets `position` and `size` but `uri` is null",
                    field_path
                )));
            }
            if has_data == has_uri {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {row_idx} must set exactly one of `data` and `uri`",
                    field_path
                )));
            }
            if has_size
                && size_column
                    .as_ref()
                    .is_some_and(|column| column.value(row_idx) == 0)
            {
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' row {row_idx} external range `size` must be greater than zero",
                    field_path
                )));
            }
            if has_uri {
                external_uris.push((
                    field_path.to_string(),
                    uri_column.value(row_idx).to_string(),
                ));
            }
        }
        return Ok(());
    }

    match field.data_type() {
        ArrowDataType::Struct(children) => {
            let struct_array = array.as_struct();
            let child_selection = selected_rows
                .iter()
                .copied()
                .enumerate()
                .map(|(row_idx, is_selected)| is_selected && struct_array.is_valid(row_idx))
                .collect::<Vec<_>>();
            for (child_field, child_array) in children.iter().zip(struct_array.columns()) {
                let child_path = format!("{}.{}", field_path, child_field.name());
                collect_external_blob_uris(
                    child_field,
                    child_array,
                    &child_selection,
                    &child_path,
                    external_uris,
                )?;
            }
        }
        ArrowDataType::List(child) => {
            let list_array = array.as_list::<i32>();
            let mut child_selection = vec![false; list_array.values().len()];
            for (row_idx, is_selected) in selected_rows.iter().copied().enumerate() {
                if is_selected && list_array.is_valid(row_idx) {
                    let start = list_array.value_offsets()[row_idx].as_usize();
                    let end = list_array.value_offsets()[row_idx + 1].as_usize();
                    child_selection[start..end].fill(true);
                }
            }
            let child_path = format!("{}.{}", field_path, child.name());
            collect_external_blob_uris(
                child,
                list_array.values(),
                &child_selection,
                &child_path,
                external_uris,
            )?;
        }
        ArrowDataType::LargeList(child) => {
            let list_array = array.as_list::<i64>();
            let mut child_selection = vec![false; list_array.values().len()];
            for (row_idx, is_selected) in selected_rows.iter().copied().enumerate() {
                if is_selected && list_array.is_valid(row_idx) {
                    let start = list_array.value_offsets()[row_idx].as_usize();
                    let end = list_array.value_offsets()[row_idx + 1].as_usize();
                    child_selection[start..end].fill(true);
                }
            }
            let child_path = format!("{}.{}", field_path, child.name());
            collect_external_blob_uris(
                child,
                list_array.values(),
                &child_selection,
                &child_path,
                external_uris,
            )?;
        }
        _ => {}
    }
    Ok(())
}

/// Validate external blob references supplied by selected input rows.
///
/// Existing rows can contain trusted absolute references that were accepted by an earlier write.
/// Update paths use this check before allowing those fallback values through the writer, so newly
/// matched values must still resolve beneath a registered external base.
#[doc(hidden)]
pub async fn validate_external_blob_references(
    resolver: &ExternalBaseResolver,
    batch: &RecordBatch,
    selected_rows: &[bool],
) -> Result<()> {
    let mut external_uris = Vec::new();
    for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
        collect_external_blob_uris(
            field,
            array,
            selected_rows,
            field.name(),
            &mut external_uris,
        )?;
    }

    let mut validated_uris = HashSet::new();
    for (field_path, uri) in external_uris {
        if validated_uris.insert(uri.clone())
            && resolver.resolve_external_uri(&uri).await?.is_none()
        {
            return Err(Error::invalid_input(format!(
                "External blob URI '{}' in field '{}' is outside registered external bases (dataset root is not allowed)",
                uri, field_path
            )));
        }
    }
    Ok(())
}

pub(super) fn normalize_external_absolute_uri(uri: &str) -> Result<String> {
    let url = Url::parse(uri).map_err(|_| {
        Error::invalid_input(format!(
            "External URI '{}' is outside registered external bases and is not a valid absolute URI",
            uri
        ))
    })?;
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "azure")]
    use std::collections::HashMap;

    #[cfg(feature = "azure")]
    use lance_table::format::BasePath;

    use super::*;
    use arrow_array::types::{UInt8Type, UInt32Type};
    use arrow_schema::Schema;
    use async_trait::async_trait;
    use lance_core::{
        ROW_ADDR,
        datatypes::{BlobHandling, BlobKind, OnMissing, Projection},
        utils::tempfile::TempDir,
    };
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::ObjectStoreProvider;
    #[cfg(feature = "azure")]
    use lance_io::object_store::StorageOptionsAccessor;
    use rstest::rstest;

    use crate::blob::{
        BlobArrayBuilder, BlobMaterializationContext, blob_field,
        materialize_blob_v2_binary_batch_with_admission,
    };
    use crate::{InsertBuilder, WriteParams};

    #[derive(Debug)]
    struct ReferenceProbeProvider;

    #[async_trait]
    impl ObjectStoreProvider for ReferenceProbeProvider {
        async fn new_store(&self, _uri: Url, _params: &ObjectStoreParams) -> Result<ObjectStore> {
            panic!("resolving an external reference must not open a store");
        }

        fn extract_path(&self, uri: &Url) -> Result<Path> {
            Path::from_url_path(uri.path())
                .map_err(|error| Error::invalid_input(format!("Invalid source path: {error}")))
        }

        fn calculate_object_store_prefix(
            &self,
            _uri: &Url,
            _options: Option<&std::collections::HashMap<String, String>>,
        ) -> Result<String> {
            Ok("probe".to_string())
        }
    }

    #[cfg(feature = "azure")]
    fn azure_store_params(account_name: &str) -> ObjectStoreParams {
        ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([
                    ("account_name".to_string(), account_name.to_string()),
                    ("account_key".to_string(), "dGVzdA==".to_string()),
                ]),
            ))),
            ..Default::default()
        }
    }

    #[cfg(feature = "azure")]
    #[tokio::test]
    async fn test_external_base_resolver_uses_candidate_store_params() {
        let store_registry = Arc::new(ObjectStoreRegistry::default());
        let base_a = BasePath::new(
            1,
            "az://container/path-a".to_string(),
            Some("base-a".to_string()),
            false,
        );
        let base_b = BasePath::new(
            2,
            "az://container/path-b".to_string(),
            Some("base-b".to_string()),
            false,
        );

        let base_a_params = azure_store_params("account-a");
        let base_b_params = azure_store_params("account-b");

        let (store_a, extracted_a) =
            ObjectStore::from_uri_and_params(store_registry.clone(), &base_a.path, &base_a_params)
                .await
                .unwrap();
        let (store_b, extracted_b) =
            ObjectStore::from_uri_and_params(store_registry.clone(), &base_b.path, &base_b_params)
                .await
                .unwrap();

        let resolver = ExternalBaseResolver::new(
            vec![
                ExternalBaseCandidate {
                    base_id: base_a.id,
                    store_prefix: store_a.store_prefix.clone(),
                    base_path: extracted_a,
                    store_params: base_a_params,
                },
                ExternalBaseCandidate {
                    base_id: base_b.id,
                    store_prefix: store_b.store_prefix.clone(),
                    base_path: extracted_b,
                    store_params: base_b_params,
                },
            ],
            store_registry,
        );

        let resolved_a = resolver
            .resolve_external_uri("az://container/path-a/file.bin")
            .await
            .unwrap()
            .unwrap();
        let resolved_b = resolver
            .resolve_external_uri("az://container/path-b/file.bin")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(resolved_a.base_id, 1);
        assert_eq!(resolved_a.relative_path, "file.bin");
        assert_eq!(resolved_b.base_id, 2);
        assert_eq!(resolved_b.relative_path, "file.bin");
    }

    #[rstest]
    #[case::absolute("/tmp/external-base/payload")]
    #[case::relative("external-base/payload")]
    #[tokio::test]
    async fn test_external_reference_matches_local_paths_when_provider_is_enabled(
        #[case] uri: &str,
    ) {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let path = ObjectStore::extract_path_from_uri(registry.clone(), uri).unwrap();
        let base = path.parent().unwrap();
        let mut resolver = ExternalBaseResolver::new(
            vec![ExternalBaseCandidate {
                base_id: 2,
                store_prefix: "file".to_string(),
                base_path: base,
                store_params: ObjectStoreParams::default(),
            }],
            registry,
        );
        resolver.unregistered_external_blob_policy = UnregisteredExternalBlobPolicy::Deny;
        let matched = resolver.resolve_external_uri(uri).await.unwrap().unwrap();
        assert_eq!(matched.base_id, 2);
        assert_eq!(matched.relative_path, "payload");
    }

    #[tokio::test]
    async fn test_external_reference_preserves_registered_matches_and_input_errors() {
        let registry = Arc::new(
            ObjectStoreRegistry::default()
                .without_providers(["file", "file-object-store", "file+uring"])
                .unwrap(),
        );
        registry.insert("probe", Arc::new(ReferenceProbeProvider));
        let mut resolver = ExternalBaseResolver::new(
            vec![ExternalBaseCandidate {
                base_id: 2,
                store_prefix: "probe".to_string(),
                base_path: Path::from("base"),
                store_params: ObjectStoreParams::default(),
            }],
            registry,
        );
        resolver.unregistered_external_blob_policy = UnregisteredExternalBlobPolicy::Deny;
        let matched = resolver
            .resolve_external_uri("probe://bucket/base/payload")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(matched.base_id, 2);
        assert_eq!(matched.relative_path, "payload");
        let error = resolver
            .resolve_external_uri("probe://bucket/base/%FF")
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("Invalid source path"), "{error}");
        assert!(!error.to_string().contains("Remote tables"), "{error}");
        resolver.unregistered_external_blob_policy = UnregisteredExternalBlobPolicy::Allow;
        let error = resolver
            .resolve_external_uri("file:///etc/passwd")
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("Unknown scheme: file"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_blob_v2_external_outside_base_allowed() {
        let dataset_dir = TempDir::default();
        let external_dir = TempDir::default();
        let external_path = external_dir.std_path().join("external.bin");
        std::fs::write(&external_path, b"outside").unwrap();
        let external_uri = format!("file://{}", external_path.display());

        let mut blob_builder = BlobArrayBuilder::new(1);
        blob_builder.push_uri(external_uri.clone()).unwrap();
        let blob_array: arrow_array::ArrayRef = blob_builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![blob_field("blob", true)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array]).unwrap();
        let params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            allow_external_blob_outside_bases: true,
            ..Default::default()
        };
        let table = InsertBuilder::new(dataset_dir.path_str().as_str())
            .with_params(&params)
            .execute(vec![batch])
            .await
            .unwrap();

        let projection = table
            .schema()
            .project_preserve_system_columns(&["blob", ROW_ADDR])
            .unwrap();
        let descriptor_batch = table.take(&[0], &projection).await.unwrap();
        {
            let desc = descriptor_batch.column_by_name("blob").unwrap().as_struct();
            assert_eq!(
                desc.column(0).as_primitive::<UInt8Type>().value(0),
                BlobKind::External as u8
            );
            assert_eq!(desc.column(2).as_primitive::<UInt64Type>().value(0), 0);
            assert_eq!(desc.column(3).as_primitive::<UInt32Type>().value(0), 0);
            let expected_uri = normalize_external_absolute_uri(&external_uri).unwrap();
            assert_eq!(desc.column(4).as_string::<i32>().value(0), expected_uri);
        }

        let output_schema = Projection::empty(Arc::new(table.schema().clone()))
            .union_columns(["blob"], OnMissing::Error)
            .unwrap()
            .with_blob_handling(BlobHandling::AllBinary)
            .to_schema();
        let descriptor_bytes =
            u64::try_from(descriptor_batch.get_array_memory_size()).unwrap_or(u64::MAX);
        let context = BlobMaterializationContext::new(None, Some(1));
        let materialized = materialize_blob_v2_binary_batch_with_admission(
            &table,
            &output_schema,
            descriptor_batch,
            &context,
            context.admission(),
        )
        .await
        .unwrap();
        let reserved = context
            .budget
            .as_ref()
            .unwrap()
            .state
            .lock()
            .unwrap()
            .reserved;
        assert_eq!(
            reserved,
            descriptor_bytes + b"outside".len() as u64 + 2 * std::mem::size_of::<i64>() as u64
        );
        assert_eq!(
            materialized
                .batch()
                .column_by_name("blob")
                .unwrap()
                .as_binary::<i64>()
                .value(0),
            b"outside"
        );
        drop(materialized);
        assert_eq!(
            context
                .budget
                .as_ref()
                .unwrap()
                .state
                .lock()
                .unwrap()
                .reserved,
            0
        );

        let blobs = table.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(blobs.len(), 1);
        assert_eq!(
            blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"outside"
        );
    }
}
