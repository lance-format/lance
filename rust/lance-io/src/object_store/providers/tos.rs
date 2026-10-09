// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use object_store::ObjectStore as OSObjectStore;
use opendal::{Operator, services::Tos};
use url::Url;

use crate::object_store::dynamic_opendal::DynamicOpenDalStore;
use crate::object_store::opendal_store::OpendalStore;
use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE,
    DirectoryOperations, ObjectStore, ObjectStoreParams, ObjectStoreProvider, StorageOptions,
};
use lance_core::error::{Error, Result};

#[derive(Default, Debug)]
pub struct TosStoreProvider;

impl TosStoreProvider {
    fn tos_env_options_from_iter<I, K, V>(vars: I) -> HashMap<String, String>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let vars = vars
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect::<Vec<_>>();
        let mut config_map = HashMap::new();

        for prefix in ["VOLCENGINE_", "TOS_"] {
            for (key, value) in &vars {
                if let Some(stripped_key) = key.strip_prefix(prefix) {
                    config_map.insert(stripped_key.to_ascii_lowercase(), value.clone());
                }
            }
        }

        config_map
    }

    fn base_tos_options(
        base_path: &Url,
        storage_options: &StorageOptions,
    ) -> Result<HashMap<String, String>> {
        let bucket = base_path
            .host_str()
            .ok_or_else(|| Error::invalid_input("TOS URL must contain bucket name"))?
            .to_string();

        let prefix = base_path.path().trim_start_matches('/').to_string();

        let mut config_map = Self::tos_env_options_from_iter(std::env::vars());

        config_map.extend(storage_options.0.clone());

        config_map.insert("bucket".to_string(), bucket);
        if prefix.is_empty() {
            config_map.remove("root");
        } else {
            config_map.insert("root".to_string(), "/".to_string());
        }

        Ok(config_map)
    }

    /// Normalize TOS storage options, resolving aliases for well-known keys
    /// while passing through all other options so that OpenDAL can use them.
    fn normalize_tos_config(options: &HashMap<String, String>) -> Result<HashMap<String, String>> {
        let mut config_map = options.clone();

        let alias_groups: &[(&str, &[&str])] = &[
            ("endpoint", &["tos_endpoint"]),
            ("region", &["tos_region"]),
            ("access_key_id", &["tos_access_key_id"]),
            ("secret_access_key", &["tos_secret_access_key"]),
            ("security_token", &["tos_security_token"]),
        ];

        for (canonical, aliases) in alias_groups {
            for alias in *aliases {
                if let Some(value) = config_map.remove(*alias) {
                    config_map.insert(canonical.to_string(), value);
                    break;
                }
            }
        }

        if !config_map.contains_key("endpoint") {
            return Err(Error::invalid_input(
                "TOS endpoint is required. Please provide 'tos_endpoint' in storage options or set TOS_ENDPOINT environment variable",
            ));
        }

        Ok(config_map)
    }

    fn build_tos_store(config_map: HashMap<String, String>) -> Result<OpendalStore> {
        let operator = Operator::from_iter::<Tos>(config_map)
            .map_err(|e| Error::invalid_input(format!("Failed to create TOS operator: {:?}", e)))?;

        Ok(OpendalStore::new(operator))
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for TosStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());

        let base_options = Self::base_tos_options(&base_path, &storage_options)?;
        let accessor = params.get_accessor();

        let (inner, directory_operations): (Arc<dyn OSObjectStore>, Arc<dyn DirectoryOperations>) =
            if let Some(accessor) = accessor.filter(|a| a.has_provider()) {
                let store = Arc::new(
                    DynamicOpenDalStore::new(
                        format!("tos:{}", base_path),
                        base_options,
                        accessor,
                        Self::normalize_tos_config,
                        Self::build_tos_store,
                    )
                    .with_protected_keys(["bucket", "root"]),
                );
                (store.clone(), store)
            } else {
                let store = Arc::new(Self::build_tos_store(Self::normalize_tos_config(
                    &base_options,
                )?)?);
                (store.clone(), store)
            };

        let mut url = base_path;
        if !url.path().ends_with('/') {
            url.set_path(&format!("{}/", url.path()));
        }

        Ok(ObjectStore {
            scheme: "tos".to_string(),
            inner,
            directory_operations: Some(directory_operations),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: params.use_constant_size_upload_parts,
            list_is_lexically_ordered: params.list_is_lexically_ordered.unwrap_or(true),
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count: storage_options.download_retry_count(),
            io_tracker: Default::default(),
            store_prefix: self.calculate_object_store_prefix(&url, params.storage_options())?,
            // Listed in full: no paginated lister covers OpenDAL yet.
            paginated_lister: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use http::{Method, Request, Response};
    use object_store::path::Path;
    use object_store::{ObjectStoreExt as _, memory::InMemory};
    use opendal::{Buffer, HttpBody, HttpTransport, HttpTransporter, OperationContext, Operator};
    use rstest::rstest;
    use serde_json::json;
    use std::collections::BTreeSet;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::{Tos, TosStoreProvider};
    use crate::object_store::dynamic_opendal::DynamicOpenDalStore;
    use crate::object_store::test_utils::StaticMockStorageOptionsProvider;
    use crate::object_store::{ObjectStore, opendal_store::OpendalStore};
    use crate::object_store::{ObjectStoreProvider, StorageOptionsAccessor};
    use url::Url;

    #[derive(Clone, Default)]
    struct HierarchicalTos {
        keys: Arc<Mutex<BTreeSet<String>>>,
        deleted: Arc<Mutex<Vec<String>>>,
        fail_delete: Option<String>,
    }

    impl HttpTransport for HierarchicalTos {
        async fn fetch(&self, request: Request<Buffer>) -> opendal::Result<Response<HttpBody>> {
            let mut keys = self.keys.lock().unwrap();
            let (status, body) = if request.method() == Method::GET {
                let url = Url::parse(&request.uri().to_string()).unwrap();
                let prefix = url
                    .query_pairs()
                    .find(|(key, _)| key == "prefix")
                    .unwrap()
                    .1;
                let contents = keys.iter().filter(|key| key.starts_with(prefix.as_ref())).map(|key| {
                    json!({"Key": key, "Size": 0, "LastModified": "2026-10-09T00:00:00Z"})
                }).collect::<Vec<_>>();
                (200, json!({"Contents": contents, "IsTruncated": false}))
            } else {
                assert_eq!(
                    request.method(),
                    Method::DELETE,
                    "directory deletion must use single-object requests"
                );
                let key =
                    opendal::raw::percent_decode_path(request.uri().path().trim_start_matches('/'));
                self.deleted.lock().unwrap().push(key.clone());
                let directory = format!("{}/", key.trim_end_matches('/'));
                if self.fail_delete.as_deref() == Some(key.as_str()) {
                    (
                        403,
                        json!({"Code": "AccessDenied", "Message": "Deletion denied"}),
                    )
                } else if keys.contains(&directory)
                    && keys
                        .iter()
                        .any(|child| child.starts_with(&directory) && child != &directory)
                {
                    (
                        409,
                        json!({"Code": "CannotDelete", "Message": "Directory is not empty"}),
                    )
                } else {
                    keys.remove(&key);
                    keys.remove(&directory);
                    (204, json!({}))
                }
            };
            let buffer = Buffer::from(body.to_string());
            let size = buffer.len() as u64;
            Ok(Response::builder()
                .status(status)
                .body(HttpBody::new(
                    futures::stream::iter([Ok(buffer)]),
                    Some(size),
                ))
                .unwrap())
        }
    }

    async fn hierarchical_store(mock: HierarchicalTos) -> Arc<ObjectStore> {
        let operator = Operator::from_iter::<Tos>(HashMap::from([
            ("bucket".to_string(), "test-bucket".to_string()),
            (
                "endpoint".to_string(),
                "https://tos-cn-beijing.volces.com".to_string(),
            ),
            ("region".to_string(), "cn-beijing".to_string()),
            ("access_key_id".to_string(), "test-key".to_string()),
            ("secret_access_key".to_string(), "test-secret".to_string()),
        ]))
        .unwrap()
        .with_context(OperationContext::new().with_http_transport(HttpTransporter::new(mock)));
        let native = Arc::new(OpendalStore::new(operator));
        let (mut store, _) = ObjectStore::from_uri("memory://").await.unwrap();
        let store_mut = Arc::get_mut(&mut store).unwrap();
        store_mut.inner = native.clone();
        store_mut.directory_operations = Some(native);
        store
    }

    #[rstest]
    #[case::hierarchical("dataset", true)]
    #[case::flat("dataset", false)]
    #[case::reserved_character("dataset/run~1", true)]
    #[case::literal_percent_escape("dataset/run%25231", true)]
    #[tokio::test]
    async fn test_remove_dir_all_preserves_directory_semantics(
        #[case] base_url: &str,
        #[case] hierarchical: bool,
    ) {
        let base = Path::from_url_path(base_url).unwrap();
        let prefix = opendal::raw::percent_decode_path(base.as_ref());
        let mut keys = BTreeSet::from([
            format!("{prefix}/data/part.lance"),
            format!("{prefix}/_versions/1.manifest"),
            format!("{prefix}-other/keep"),
        ]);
        if hierarchical {
            keys.extend([
                format!("{prefix}/"),
                format!("{prefix}/data/"),
                format!("{prefix}/_versions/"),
                format!("{prefix}/empty/"),
                format!("{prefix}/empty/nested/"),
            ]);
        }
        let mock = HierarchicalTos {
            keys: Arc::new(Mutex::new(keys)),
            ..Default::default()
        };
        let store = hierarchical_store(mock.clone()).await;
        store.remove_dir_all(base.clone()).await.unwrap();
        assert_eq!(
            *mock.keys.lock().unwrap(),
            BTreeSet::from([format!("{prefix}-other/keep")])
        );
        // The listing contains zero-byte files, which must not be treated as directories.
        {
            let deleted = mock.deleted.lock().unwrap();
            assert_eq!(deleted.len(), if hierarchical { 7 } else { 2 });
            if hierarchical {
                assert_eq!(deleted.last().unwrap(), &prefix);
            }
        }
        store.remove_dir_all(base).await.unwrap();
    }

    #[rstest]
    #[case::empty_directory(true)]
    #[case::missing_directory(false)]
    #[tokio::test]
    async fn test_remove_empty_or_missing_directory(#[case] exists: bool) {
        let mock = HierarchicalTos::default();
        if exists {
            mock.keys.lock().unwrap().insert("empty/".to_string());
        }
        hierarchical_store(mock.clone())
            .await
            .remove_dir_all("empty")
            .await
            .unwrap();
        assert!(mock.keys.lock().unwrap().is_empty());
        assert_eq!(mock.deleted.lock().unwrap().len(), usize::from(exists));
    }

    #[rstest]
    #[case::file("dataset/file")]
    #[case::directory("dataset")]
    #[tokio::test]
    async fn test_remove_dir_all_propagates_deletion_failure(#[case] fail_delete: &str) {
        let keys = BTreeSet::from(["dataset/".to_string(), "dataset/file".to_string()]);
        let mock = HierarchicalTos {
            keys: Arc::new(Mutex::new(keys.clone())),
            fail_delete: Some(fail_delete.to_string()),
            ..Default::default()
        };
        let error = hierarchical_store(mock.clone())
            .await
            .remove_dir_all("dataset")
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::IO { .. }));
        assert!(error.to_string().contains("Deletion denied"), "{error}");
        if fail_delete == "dataset/file" {
            assert_eq!(*mock.keys.lock().unwrap(), keys);
            assert_eq!(*mock.deleted.lock().unwrap(), ["dataset/file"]);
        } else {
            assert_eq!(
                *mock.keys.lock().unwrap(),
                BTreeSet::from(["dataset/".to_string()])
            );
            assert_eq!(*mock.deleted.lock().unwrap(), ["dataset/file", "dataset"]);
        }
    }

    #[tokio::test]
    async fn test_remove_dir_all_respects_wrapped_listing_and_deletion() {
        let mock = HierarchicalTos {
            keys: Arc::new(Mutex::new(BTreeSet::from([
                "dataset/".to_string(),
                "dataset/hidden".to_string(),
            ]))),
            ..Default::default()
        };
        let mut store = hierarchical_store(mock.clone()).await;
        let wrapped = Arc::new(InMemory::new());
        wrapped
            .put(&Path::from("dataset/visible"), bytes::Bytes::new().into())
            .await
            .unwrap();
        Arc::get_mut(&mut store).unwrap().inner = wrapped.clone();
        store.remove_dir_all("dataset").await.unwrap();
        assert!(matches!(
            wrapped.head(&Path::from("dataset/visible")).await,
            Err(object_store::Error::NotFound { .. })
        ));
        assert!(mock.deleted.lock().unwrap().is_empty());
        assert!(mock.keys.lock().unwrap().contains("dataset/hidden"));
    }

    #[test]
    fn test_tos_store_path() {
        let provider = TosStoreProvider;

        let url = Url::parse("tos://bucket/path/to/file").unwrap();
        let path = provider.extract_path(&url).unwrap();
        let expected_path = object_store::path::Path::from("path/to/file");
        assert_eq!(path, expected_path);
    }

    #[test]
    fn test_tos_env_options_normalize_supported_prefixes() {
        let config = TosStoreProvider::tos_env_options_from_iter([
            ("VOLCENGINE_ENDPOINT", "https://tos-cn-beijing.volces.com"),
            ("TOS_ACCESS_KEY_ID", "tos-akid"),
            ("TOS_SECRET_ACCESS_KEY", "tos-secret"),
        ]);

        assert_eq!(
            config.get("endpoint").unwrap(),
            "https://tos-cn-beijing.volces.com"
        );
        assert_eq!(config.get("access_key_id").unwrap(), "tos-akid");
        assert_eq!(config.get("secret_access_key").unwrap(), "tos-secret");
    }

    #[test]
    fn test_tos_alias_options_override_canonical_env_options() {
        let config = TosStoreProvider::normalize_tos_config(&HashMap::from([
            (
                "endpoint".to_string(),
                "https://env.example.com".to_string(),
            ),
            (
                "tos_endpoint".to_string(),
                "https://user.example.com".to_string(),
            ),
            ("region".to_string(), "env-region".to_string()),
            ("tos_region".to_string(), "user-region".to_string()),
            ("access_key_id".to_string(), "env-akid".to_string()),
            ("tos_access_key_id".to_string(), "user-akid".to_string()),
            ("secret_access_key".to_string(), "env-secret".to_string()),
            (
                "tos_secret_access_key".to_string(),
                "user-secret".to_string(),
            ),
            ("security_token".to_string(), "env-token".to_string()),
            ("tos_security_token".to_string(), "user-token".to_string()),
            ("bucket".to_string(), "bucket".to_string()),
        ]))
        .unwrap();

        assert_eq!(config.get("endpoint").unwrap(), "https://user.example.com");
        assert_eq!(config.get("region").unwrap(), "user-region");
        assert_eq!(config.get("access_key_id").unwrap(), "user-akid");
        assert_eq!(config.get("secret_access_key").unwrap(), "user-secret");
        assert_eq!(config.get("security_token").unwrap(), "user-token");
        assert!(!config.contains_key("tos_endpoint"));
        assert!(!config.contains_key("tos_secret_access_key"));
        assert!(!config.contains_key("tos_security_token"));
    }

    #[test]
    fn test_tos_url_bucket_and_root_are_authoritative() {
        let storage_options = crate::object_store::StorageOptions(HashMap::from([
            (
                "tos_endpoint".to_string(),
                "https://tos-cn-beijing.volces.com".to_string(),
            ),
            ("bucket".to_string(), "storage-options-bucket".to_string()),
            ("root".to_string(), "/storage-options-root".to_string()),
        ]));
        let base_options = TosStoreProvider::base_tos_options(
            &Url::parse("tos://url-bucket/path").unwrap(),
            &storage_options,
        )
        .unwrap();
        let config = TosStoreProvider::normalize_tos_config(&base_options).unwrap();

        assert_eq!(config.get("bucket").unwrap(), "url-bucket");
        assert_eq!(config.get("root").unwrap(), "/");

        let base_options = TosStoreProvider::base_tos_options(
            &Url::parse("tos://url-bucket").unwrap(),
            &storage_options,
        )
        .unwrap();
        let config = TosStoreProvider::normalize_tos_config(&base_options).unwrap();

        assert_eq!(config.get("bucket").unwrap(), "url-bucket");
        assert!(!config.contains_key("root"));
    }

    #[tokio::test]
    async fn test_dynamic_opendal_tos_store_uses_provider_credentials() {
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            StaticMockStorageOptionsProvider {
                options: HashMap::from([
                    (
                        "tos_endpoint".to_string(),
                        "https://tos-cn-beijing.volces.com".to_string(),
                    ),
                    ("tos_region".to_string(), "cn-beijing".to_string()),
                    ("tos_access_key_id".to_string(), "akid".to_string()),
                    ("tos_secret_access_key".to_string(), "secret".to_string()),
                    ("tos_security_token".to_string(), "token".to_string()),
                ]),
            },
        )));

        let base_options = TosStoreProvider::base_tos_options(
            &Url::parse("tos://url-bucket/path").unwrap(),
            &crate::object_store::StorageOptions(HashMap::new()),
        )
        .unwrap();

        let store = DynamicOpenDalStore::new(
            "tos",
            base_options,
            accessor,
            TosStoreProvider::normalize_tos_config,
            TosStoreProvider::build_tos_store,
        )
        .with_protected_keys(["bucket", "root"]);

        let current_store = store
            .current_store()
            .await
            .expect("dynamic OpenDAL TOS store should build");

        assert!(current_store.to_string().contains("Opendal"));
    }
}
