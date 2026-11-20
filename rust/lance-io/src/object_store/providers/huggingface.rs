// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use object_store::ObjectStore as OSObjectStore;
use object_store::path::Path;
use opendal::{Operator, services::Huggingface};
use url::Url;

use crate::object_store::dynamic_opendal::DynamicOpenDalStore;
use crate::object_store::opendal_store::OpendalStore;
use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, ParsedHfUrl, StorageOptions, parse_hf_url,
    resolve_hf_revision, validate_hf_revision_option,
};
use lance_core::error::{Error, Result};

use super::opendal::finish_opendal_operator;

/// Hugging Face object store provider backed by OpenDAL.
#[derive(Default, Debug)]
pub struct HuggingfaceStoreProvider;

fn build_hf_base_options(
    repo_type: &str,
    repo_id: &str,
    revision: Option<&str>,
    storage_options: &StorageOptions,
) -> HashMap<String, String> {
    let mut options = storage_options.0.clone();
    options.insert("repo_type".to_string(), repo_type.to_string());
    options.insert("repo_id".to_string(), repo_id.to_string());
    if let Some(revision) = revision {
        options.insert("revision".to_string(), revision.to_string());
    }
    options
}

fn normalize_download_mode(download_mode: String) -> Result<String> {
    match download_mode.to_lowercase().as_str() {
        "xet" => Ok("xet".to_string()),
        "http" => Ok("http".to_string()),
        _ => Err(Error::invalid_input(format!(
            "Invalid Huggingface download_mode: {download_mode}. Expected one of: xet, http"
        ))),
    }
}

fn normalize_hf_config(options: &HashMap<String, String>) -> Result<HashMap<String, String>> {
    let mut config_map = HashMap::new();

    let repo_type = options
        .get("repo_type")
        .cloned()
        .ok_or_else(|| Error::invalid_input("Huggingface repo_type is required"))?;
    let repo_id = options
        .get("repo_id")
        .cloned()
        .ok_or_else(|| Error::invalid_input("Huggingface repo_id is required"))?;

    config_map.insert("repo_type".to_string(), repo_type);
    config_map.insert("repo_id".to_string(), repo_id);

    if let Some(revision) = options
        .get("hf_revision")
        .or_else(|| options.get("revision"))
    {
        let key = if options.contains_key("hf_revision") {
            "hf_revision"
        } else {
            "revision"
        };
        // A provider refresh can supply an empty revision. Reject it instead of
        // silently falling back to the repository's default branch.
        validate_hf_revision_option(key, revision)?;
        config_map.insert("revision".to_string(), revision.clone());
    }

    if let Some(root) = options
        .get("hf_root")
        .cloned()
        .or_else(|| options.get("root").cloned())
        && !root.is_empty()
    {
        config_map.insert("root".to_string(), root);
    }

    if let Some(token) = options
        .get("hf_token")
        .cloned()
        .or_else(|| options.get("token").cloned())
        && !token.is_empty()
    {
        config_map.insert("token".to_string(), token);
    }

    let download_mode = options
        .get("hf_download_mode")
        .filter(|download_mode| !download_mode.is_empty())
        .cloned()
        .or_else(|| {
            options
                .get("download_mode")
                .filter(|download_mode| !download_mode.is_empty())
                .cloned()
        })
        .unwrap_or_else(|| "http".to_string());
    config_map.insert(
        "download_mode".to_string(),
        normalize_download_mode(download_mode)?,
    );

    if let Some(enabled) = options
        .get("hf_enable_resolve_cache")
        .or_else(|| options.get("enable_resolve_cache"))
    {
        config_map.insert("enable_resolve_cache".to_string(), enabled.clone());
    }

    if let Some((_, max_retries)) = options
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("client_max_retries"))
    {
        config_map.insert("client_max_retries".to_string(), max_retries.clone());
    }

    Ok(config_map)
}

fn build_hf_store(config_map: HashMap<String, String>) -> Result<OpendalStore> {
    let storage_options = StorageOptions(config_map);
    let max_retries = storage_options.client_max_retries();
    let config_map = &storage_options.0;
    let repo_type = config_map
        .get("repo_type")
        .ok_or_else(|| Error::invalid_input("Huggingface repo_type is required"))?;
    let repo_id = config_map
        .get("repo_id")
        .ok_or_else(|| Error::invalid_input("Huggingface repo_id is required"))?;

    let mut builder = Huggingface::default().repo_type(repo_type).repo_id(repo_id);
    if let Some(revision) = config_map.get("revision") {
        builder = builder.revision(revision);
    }
    if let Some(root) = config_map.get("root") {
        builder = builder.root(root);
    }
    if let Some(token) = config_map.get("token") {
        builder = builder.token(token);
    }
    if let Some(download_mode) = config_map.get("download_mode") {
        builder = builder.download_mode(download_mode);
    }
    if let Some(enabled) = config_map.get("enable_resolve_cache") {
        let enabled = enabled.parse::<bool>().map_err(|_| {
            Error::invalid_input(format!(
                "Invalid Huggingface enable_resolve_cache: {enabled:?}. Expected true or false"
            ))
        })?;
        builder = builder.enable_resolve_cache(enabled);
    }

    let operator = Operator::new(builder).map_err(|e| {
        Error::invalid_input(format!("Failed to create Huggingface operator: {:?}", e))
    })?;
    let operator = finish_opendal_operator(operator, max_retries);

    Ok(OpendalStore::new(operator))
}

#[async_trait::async_trait]
impl ObjectStoreProvider for HuggingfaceStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let ParsedHfUrl {
            repo_type,
            repo_id,
            revision,
            ..
        } = parse_hf_url(&base_path)?;

        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let storage_options =
            StorageOptions::new(params.storage_options().cloned().unwrap_or_default());
        let revision = resolve_hf_revision(revision.as_deref(), Some(&storage_options.0))?;
        let download_retry_count = storage_options.download_retry_count();

        let mut base_options =
            build_hf_base_options(&repo_type, &repo_id, revision.as_deref(), &storage_options);
        if !base_options.contains_key("hf_token") && !base_options.contains_key("token") {
            if let Ok(token) = std::env::var("HF_TOKEN") {
                base_options.insert("hf_token".to_string(), token);
            } else if let Ok(token) = std::env::var("HUGGINGFACE_TOKEN") {
                base_options.insert("hf_token".to_string(), token);
            }
        }

        let accessor = params.get_accessor();
        let inner: Arc<dyn OSObjectStore> =
            if let Some(accessor) = accessor.filter(|a| a.has_provider()) {
                Arc::new(
                    DynamicOpenDalStore::new(
                        format!("hf:{}", base_path),
                        base_options,
                        accessor,
                        normalize_hf_config,
                        build_hf_store,
                    )
                    .with_protected_keys([
                        "repo_type",
                        "repo_id",
                        "revision",
                        "hf_revision",
                    ]),
                )
            } else {
                Arc::new(build_hf_store(normalize_hf_config(&base_options)?)?)
            };

        Ok(ObjectStore {
            scheme: "hf".to_string(),
            inner,
            external_blob_inner: None,
            direct_local_io: false,
            _external_blob_origin: None,
            local_dir_operations: None,
            local_fsync: false,
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: params.use_constant_size_upload_parts,
            list_is_lexically_ordered: params.list_is_lexically_ordered.unwrap_or(true),
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count,
            scheduler_error_mode: Default::default(),
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
            // Listed in full: no paginated lister covers OpenDAL yet.
            paginated_lister: None,
        })
    }

    fn extract_path(&self, url: &Url) -> Result<Path> {
        let parsed = parse_hf_url(url)?;
        Path::from_url_path(&parsed.relative_path).map_err(|_| {
            Error::invalid_input(format!("Invalid path in Huggingface URL: {}", url.path()))
        })
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        let parsed = parse_hf_url(url)?;
        let revision = resolve_hf_revision(parsed.revision.as_deref(), storage_options)?;
        let mut prefix = format!("{}${}@{}", url.scheme(), url.authority(), parsed.repo_id);
        if let Some(revision) = revision {
            prefix.push('@');
            prefix.push_str(&revision);
        }
        Ok(prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    use bytes::Bytes;
    use rstest::rstest;
    use serial_test::serial;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::object_store::StorageOptionsAccessor;
    use crate::object_store::dynamic_opendal::DynamicOpenDalStore;
    use crate::object_store::test_utils::StaticMockStorageOptionsProvider;

    /// Restore `HF_ENDPOINT` when dropped so the local mock cannot leak into
    /// other tests.
    struct RestoreHfEndpoint {
        original: Option<OsString>,
    }

    impl RestoreHfEndpoint {
        fn set(value: &str) -> Self {
            let original = std::env::var_os("HF_ENDPOINT");
            // SAFETY: `#[serial(hf_endpoint_env)]` keeps the tests that use this
            // guard from changing the environment at the same time. `set_var`
            // also requires that no other thread reads or writes the environment
            // outside `std::env` meanwhile, which this test binary does not
            // guarantee; see the same caveat on the GooseFS test helper.
            unsafe { std::env::set_var("HF_ENDPOINT", value) };
            Self { original }
        }
    }

    impl Drop for RestoreHfEndpoint {
        fn drop(&mut self) {
            // SAFETY: see `RestoreHfEndpoint::set`.
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var("HF_ENDPOINT", value),
                    None => std::env::remove_var("HF_ENDPOINT"),
                }
            }
        }
    }

    #[test]
    fn parse_basic_url() {
        let url = Url::parse("hf://datasets/acme/repo/path/to/table.lance").unwrap();
        let parsed = parse_hf_url(&url).unwrap();
        assert_eq!(
            parsed,
            ParsedHfUrl {
                repo_type: "dataset".to_string(),
                repo_id: "acme/repo".to_string(),
                revision: None,
                relative_path: "path/to/table.lance".to_string(),
            }
        );
    }

    #[rstest]
    #[case::dataset("hf://datasets/acme/repo@0123/path/to/table.lance", "dataset")]
    #[case::model("hf://models/acme/repo@0123/path/to/table.lance", "model")]
    #[case::space("hf://spaces/acme/repo@0123/path/to/table.lance", "space")]
    fn parse_url_with_revision(#[case] uri: &str, #[case] repo_type: &str) {
        let url = Url::parse(uri).unwrap();
        let parsed = parse_hf_url(&url).unwrap();
        assert_eq!(parsed.repo_type, repo_type);
        assert_eq!(parsed.repo_id, "acme/repo");
        assert_eq!(parsed.revision.as_deref(), Some("0123"));
        assert_eq!(parsed.relative_path, "path/to/table.lance");
    }

    #[test]
    fn parse_legacy_url_without_repo_type() {
        let url = Url::parse("hf://acme/repo@0123/path/to/file").unwrap();
        let parsed = parse_hf_url(&url).unwrap();
        assert_eq!(
            parsed,
            ParsedHfUrl {
                repo_type: "model".to_string(),
                repo_id: "acme/repo".to_string(),
                revision: Some("0123".to_string()),
                relative_path: "path/to/file".to_string(),
            }
        );
    }

    #[test]
    fn parse_at_in_path_is_not_a_revision() {
        let url = Url::parse("hf://datasets/acme/repo@0123/path/to/file@name").unwrap();
        let parsed = parse_hf_url(&url).unwrap();
        assert_eq!(parsed.revision.as_deref(), Some("0123"));
        assert_eq!(parsed.relative_path, "path/to/file@name");

        let url = Url::parse("hf://datasets/acme/repo/path/to/@name").unwrap();
        let parsed = parse_hf_url(&url).unwrap();
        assert_eq!(parsed.revision, None);
        assert_eq!(parsed.relative_path, "path/to/@name");
    }

    #[rstest]
    #[case::empty_at_end("hf://datasets/acme/repo@/path")]
    #[case::extra_at("hf://datasets/acme/repo@@rev/path")]
    #[case::encoded_space("hf://datasets/acme/repo@bad%20rev/path")]
    #[case::encoded_slash("hf://datasets/acme/repo@refs%2Fpr%2F1/path")]
    fn parse_rejects_malformed_revision(#[case] uri: &str) {
        let url = Url::parse(uri).unwrap();
        let err = parse_hf_url(&url).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(
            err.to_string().contains("revision"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_invalid_url_errors() {
        let url = Url::parse("hf://datasets/only-owner").unwrap();
        let err = parse_hf_url(&url).unwrap_err();
        assert!(err.to_string().contains("repository name"));
    }

    #[rstest]
    #[case::matching_hf(Some("pinned"), Some("pinned"), None, Some("pinned"))]
    #[case::matching_plain(None, None, Some("pinned"), Some("pinned"))]
    #[case::url_wins(Some("pinned"), None, None, Some("pinned"))]
    #[case::prefixed_option_precedence(None, Some("prefixed"), Some("plain"), Some("prefixed"))]
    fn resolve_revision(
        #[case] url_revision: Option<&str>,
        #[case] hf_revision: Option<&str>,
        #[case] revision: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        let mut options = HashMap::new();
        if let Some(value) = hf_revision {
            options.insert("hf_revision".to_string(), value.to_string());
        }
        if let Some(value) = revision {
            options.insert("revision".to_string(), value.to_string());
        }
        assert_eq!(
            resolve_hf_revision(url_revision, Some(&options))
                .unwrap()
                .as_deref(),
            expected
        );
    }

    #[rstest]
    #[case::prefixed("hf_revision")]
    #[case::unprefixed("revision")]
    fn resolve_rejects_conflicting_option(#[case] key: &str) {
        let options = HashMap::from([(key.to_string(), "other".to_string())]);
        let err = resolve_hf_revision(Some("pinned"), Some(&options)).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains(key), "unexpected error: {err}");
        assert!(
            err.to_string().contains("pinned"),
            "unexpected error: {err}"
        );
    }

    #[rstest]
    #[case::prefixed("hf_revision")]
    #[case::unprefixed("revision")]
    fn resolve_rejects_empty_option(#[case] key: &str) {
        let options = HashMap::from([(key.to_string(), String::new())]);
        let err = resolve_hf_revision(None, Some(&options)).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains(key), "unexpected error: {err}");
    }

    #[test]
    fn storage_option_revision_takes_precedence_without_url_pin() {
        let config = normalize_hf_config(&build_hf_base_options(
            "dataset",
            "acme/repo",
            resolve_hf_revision(
                None,
                Some(&HashMap::from([
                    ("hf_revision".to_string(), "prefixed".to_string()),
                    ("revision".to_string(), "plain".to_string()),
                ])),
            )
            .unwrap()
            .as_deref(),
            &StorageOptions(HashMap::new()),
        ))
        .unwrap();

        assert_eq!(config.get("revision").unwrap(), "prefixed");
    }

    #[test]
    fn normalize_prefers_hf_revision_over_revision() {
        let config = normalize_hf_config(&HashMap::from([
            ("repo_type".to_string(), "dataset".to_string()),
            ("repo_id".to_string(), "acme/repo".to_string()),
            ("hf_revision".to_string(), "prefixed".to_string()),
            ("revision".to_string(), "plain".to_string()),
        ]))
        .unwrap();

        assert_eq!(config.get("revision").unwrap(), "prefixed");
    }

    #[test]
    fn storage_options_cannot_override_url_repo_identity() {
        let config = normalize_hf_config(&build_hf_base_options(
            "dataset",
            "acme/repo",
            Some("stable"),
            &StorageOptions(HashMap::from([
                ("repo_type".to_string(), "model".to_string()),
                ("repo_id".to_string(), "other/repo".to_string()),
                ("hf_revision".to_string(), "stable".to_string()),
            ])),
        ))
        .unwrap();

        assert_eq!(config.get("repo_type").unwrap(), "dataset");
        assert_eq!(config.get("repo_id").unwrap(), "acme/repo");
        assert_eq!(config.get("revision").unwrap(), "stable");
    }

    #[test]
    fn normalize_rejects_empty_revision_option() {
        let err = normalize_hf_config(&HashMap::from([
            ("repo_type".to_string(), "dataset".to_string()),
            ("repo_id".to_string(), "acme/repo".to_string()),
            ("hf_revision".to_string(), String::new()),
        ]))
        .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("hf_revision"));
    }

    #[test]
    fn storage_option_download_mode_takes_hf_prefix_precedence() {
        let config = normalize_hf_config(&build_hf_base_options(
            "dataset",
            "acme/repo",
            None,
            &StorageOptions(HashMap::from([
                ("download_mode".to_string(), "xet".to_string()),
                ("hf_download_mode".to_string(), "http".to_string()),
            ])),
        ))
        .unwrap();

        assert_eq!(config.get("download_mode").unwrap(), "http");
    }

    #[test]
    fn storage_option_download_mode_defaults_to_http() {
        let config = normalize_hf_config(&build_hf_base_options(
            "dataset",
            "acme/repo",
            None,
            &StorageOptions(HashMap::new()),
        ))
        .unwrap();

        assert_eq!(config.get("download_mode").unwrap(), "http");
    }

    #[rstest]
    #[case::default(None, None, None)]
    #[case::prefixed_enabled(Some("true"), None, Some("true"))]
    #[case::prefixed_disabled(Some("false"), None, Some("false"))]
    #[case::unprefixed_enabled(None, Some("true"), Some("true"))]
    #[case::unprefixed_disabled(None, Some("false"), Some("false"))]
    #[case::prefixed_takes_precedence(Some("false"), Some("true"), Some("false"))]
    fn storage_option_resolve_cache(
        #[case] prefixed: Option<&str>,
        #[case] unprefixed: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        let mut options = HashMap::new();
        if let Some(value) = prefixed {
            options.insert("hf_enable_resolve_cache".to_string(), value.to_string());
        }
        if let Some(value) = unprefixed {
            options.insert("enable_resolve_cache".to_string(), value.to_string());
        }
        let config = normalize_hf_config(&build_hf_base_options(
            "dataset",
            "acme/repo",
            None,
            &StorageOptions(options),
        ))
        .unwrap();

        assert_eq!(
            config.get("enable_resolve_cache").map(String::as_str),
            expected
        );
        build_hf_store(config).unwrap();
    }

    #[rstest]
    #[case::prefixed("hf_enable_resolve_cache", "invalid")]
    #[case::unprefixed("enable_resolve_cache", "invalid")]
    #[case::empty("hf_enable_resolve_cache", "")]
    #[case::uppercase("hf_enable_resolve_cache", "TRUE")]
    #[case::numeric("hf_enable_resolve_cache", "1")]
    #[tokio::test]
    async fn new_store_rejects_invalid_resolve_cache(#[case] key: &str, #[case] value: &str) {
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([(key.to_string(), value.to_string())]),
            ))),
            ..Default::default()
        };
        let err = HuggingfaceStoreProvider
            .new_store(
                Url::parse("hf://datasets/acme/repo/table.lance").unwrap(),
                &params,
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("enable_resolve_cache"));
        assert!(err.to_string().contains("Expected true or false"));
    }

    #[rstest]
    #[case::prefixed("hf_revision")]
    #[case::unprefixed("revision")]
    #[tokio::test]
    async fn new_store_rejects_conflicting_revision_option(#[case] key: &str) {
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([(key.to_string(), "other".to_string())]),
            ))),
            ..Default::default()
        };
        let err = HuggingfaceStoreProvider
            .new_store(
                Url::parse("hf://datasets/acme/repo@pinned/table.lance").unwrap(),
                &params,
            )
            .await
            .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains(key), "unexpected error: {err}");
        assert!(
            err.to_string().contains("pinned"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn storage_option_preserves_client_max_retries() {
        let config = normalize_hf_config(&build_hf_base_options(
            "dataset",
            "acme/repo",
            None,
            &StorageOptions(HashMap::from([(
                "CLIENT_MAX_RETRIES".to_string(),
                "5".to_string(),
            )])),
        ))
        .unwrap();

        assert_eq!(config.get("client_max_retries").unwrap(), "5");
    }

    #[test]
    fn storage_option_download_mode_rejects_invalid_value() {
        let err = normalize_hf_config(&build_hf_base_options(
            "dataset",
            "acme/repo",
            None,
            &StorageOptions(HashMap::from([(
                "hf_download_mode".to_string(),
                "invalid".to_string(),
            )])),
        ))
        .unwrap_err();

        assert!(
            err.to_string().contains("download_mode"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn extract_path_returns_relative() {
        let url = Url::parse("hf://datasets/acme/repo/sub/dir/table.lance").unwrap();
        let provider = HuggingfaceStoreProvider;
        let path = provider.extract_path(&url).unwrap();
        assert_eq!(path.to_string(), "sub/dir/table.lance");
    }

    #[test]
    fn extract_path_ignores_revision_and_preserves_escapes() {
        let url = Url::parse("hf://datasets/acme/repo@0123/sub%20dir/file%2Bname%40suffix.lance")
            .unwrap();
        let provider = HuggingfaceStoreProvider;
        let path = provider.extract_path(&url).unwrap();
        assert_eq!(path.to_string(), "sub dir/file+name@suffix.lance");
    }

    #[test]
    fn calculate_prefix_uses_repo_id() {
        let provider = HuggingfaceStoreProvider;
        let url = Url::parse("hf://datasets/acme/repo/path").unwrap();
        let prefix = provider.calculate_object_store_prefix(&url, None).unwrap();
        assert_eq!(prefix, "hf$datasets@acme/repo");
    }

    #[test]
    fn calculate_prefix_includes_url_revision() {
        let provider = HuggingfaceStoreProvider;
        let first = provider
            .calculate_object_store_prefix(
                &Url::parse("hf://datasets/acme/repo@commit-one/data").unwrap(),
                None,
            )
            .unwrap();
        let second = provider
            .calculate_object_store_prefix(
                &Url::parse("hf://datasets/acme/repo@commit-two/data").unwrap(),
                None,
            )
            .unwrap();

        assert_eq!(first, "hf$datasets@acme/repo@commit-one");
        assert_eq!(second, "hf$datasets@acme/repo@commit-two");
        assert_ne!(first, second);
    }

    #[test]
    fn calculate_prefix_includes_option_revision() {
        let provider = HuggingfaceStoreProvider;
        let options = HashMap::from([("revision".to_string(), "commit-one".to_string())]);
        let prefix = provider
            .calculate_object_store_prefix(
                &Url::parse("hf://datasets/acme/repo/data").unwrap(),
                Some(&options),
            )
            .unwrap();
        assert_eq!(prefix, "hf$datasets@acme/repo@commit-one");
    }

    #[test]
    fn calculate_prefix_accepts_matching_url_and_option_revision() {
        let provider = HuggingfaceStoreProvider;
        let options = HashMap::from([
            ("hf_revision".to_string(), "commit-one".to_string()),
            ("revision".to_string(), "commit-one".to_string()),
        ]);
        let prefix = provider
            .calculate_object_store_prefix(
                &Url::parse("hf://datasets/acme/repo@commit-one/data").unwrap(),
                Some(&options),
            )
            .unwrap();
        assert_eq!(prefix, "hf$datasets@acme/repo@commit-one");
    }

    #[test]
    fn calculate_prefix_rejects_conflicting_option_revision() {
        let provider = HuggingfaceStoreProvider;
        let options = HashMap::from([("hf_revision".to_string(), "other".to_string())]);
        let err = provider
            .calculate_object_store_prefix(
                &Url::parse("hf://datasets/acme/repo@commit-one/data").unwrap(),
                Some(&options),
            )
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("hf_revision"));
    }

    #[tokio::test]
    async fn test_dynamic_opendal_hf_store_uses_provider_token() {
        let parsed = parse_hf_url(&Url::parse("hf://datasets/acme/repo/path").unwrap()).unwrap();
        let accessor = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            StaticMockStorageOptionsProvider {
                options: HashMap::from([
                    ("hf_token".to_string(), "dynamic-token".to_string()),
                    ("hf_enable_resolve_cache".to_string(), "true".to_string()),
                ]),
            },
        )));

        let store = DynamicOpenDalStore::new(
            "hf",
            build_hf_base_options(
                &parsed.repo_type,
                &parsed.repo_id,
                parsed.revision.as_deref(),
                &StorageOptions(HashMap::new()),
            ),
            accessor,
            normalize_hf_config,
            build_hf_store,
        );

        let current_store = store
            .current_store()
            .await
            .expect("dynamic OpenDAL HuggingFace store should build");

        assert!(current_store.to_string().contains("Opendal"));
    }

    #[tokio::test]
    #[serial(hf_endpoint_env)]
    async fn dynamic_refresh_cannot_override_url_revision() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(
                r"^/datasets/acme/repo/resolve/pinned/payload\.bin$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"pinned".to_vec()))
            .mount(&server)
            .await;
        let _endpoint = RestoreHfEndpoint::set(&server.uri());

        let accessor = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            StaticMockStorageOptionsProvider {
                options: HashMap::from([
                    ("hf_revision".to_string(), "revised".to_string()),
                    ("revision".to_string(), "other".to_string()),
                    ("hf_download_mode".to_string(), "http".to_string()),
                ]),
            },
        )));
        let params = ObjectStoreParams {
            storage_options_accessor: Some(accessor),
            ..Default::default()
        };
        let store = HuggingfaceStoreProvider
            .new_store(
                Url::parse("hf://datasets/acme/repo@pinned/payload.bin").unwrap(),
                &params,
            )
            .await
            .unwrap();

        assert_eq!(store.store_prefix, "hf$datasets@acme/repo@pinned");
        assert_eq!(
            store
                .read_one_all(&Path::from("payload.bin"))
                .await
                .unwrap(),
            Bytes::from_static(b"pinned")
        );
    }
}
