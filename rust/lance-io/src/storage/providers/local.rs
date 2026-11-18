// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::storage::BackendConfig;
#[cfg(any(windows, test))]
use crate::storage::backend::object_store::from_error;
use std::{collections::HashMap, sync::Arc};

use crate::storage::LocalDirOperations;
use crate::storage::path::Path;
use crate::storage::{
    DEFAULT_LOCAL_BLOCK_SIZE, DEFAULT_LOCAL_IO_PARALLELISM, StorageOptions, StorageParams,
    StorageProvider,
};
use ::object_store::local::LocalFileSystem;
use lance_core::Error;
use lance_core::error::Result;
#[cfg(any(windows, test))]
use std::io::ErrorKind;
use url::Url;

#[derive(Default, Debug)]
pub struct FileStoreProvider;

#[cfg(any(windows, test))]
#[derive(Debug)]
struct FileSystemDirOperations {
    local_file_system: LocalFileSystem,
}

#[cfg(any(windows, test))]
#[async_trait::async_trait]
impl LocalDirOperations for FileSystemDirOperations {
    fn directory_to_filesystem(&self, path: &Path) -> Result<std::path::PathBuf> {
        // This maps directory prefixes, which may themselves end in '#<digits>'.
        // A valid placeholder filename bypasses LocalFileSystem's file-name guard
        // while retaining its rooted path and Windows filename conversions.
        let placeholder = self
            .local_file_system
            .path_to_filesystem(&crate::storage::backend::object_store::to_path(
                &path.clone().join("__lance_directory__"),
            )?)
            .map_err(from_error)?;
        placeholder
            .parent()
            .map(std::path::Path::to_path_buf)
            .ok_or_else(|| Error::io(format!("Cannot resolve local directory prefix {path}")))
    }

    async fn remove_dir_all(&self, path: &Path) -> Result<()> {
        let local_path = self
            .local_file_system
            .path_to_filesystem(&crate::storage::backend::object_store::to_path(path)?)
            .map_err(from_error)?;
        let storage_path = path.to_string();
        tokio::task::spawn_blocking(move || {
            std::fs::remove_dir_all(local_path).map_err(|error| match error.kind() {
                ErrorKind::NotFound => Error::not_found(storage_path),
                _ => Error::from(error),
            })
        })
        .await
        .map_err(|error| Error::io(format!("recursive directory removal task failed: {error}")))?
    }

    async fn canonicalize(&self, path: &Path) -> Result<Option<std::path::PathBuf>> {
        let local_path = self
            .local_file_system
            .path_to_filesystem(&crate::storage::backend::object_store::to_path(path)?)
            .map_err(from_error)?;
        crate::storage::canonicalize_local_io_result(tokio::fs::canonicalize(local_path).await)
    }
}

#[cfg(windows)]
mod windows {
    use std::path::PathBuf;

    use super::*;

    #[derive(Debug)]
    pub(super) struct UncPath {
        pub(super) root: PathBuf,
        pub(super) relative_path: Path,
        pub(super) store_prefix: String,
    }

    pub(super) fn extract_unc_path(url: &Url) -> Result<Option<UncPath>> {
        if url.scheme() != "file" {
            return Ok(None);
        }

        let Some(host) = url.host_str().filter(|host| *host != "localhost") else {
            return Ok(None);
        };
        let encoded_path = url.path().strip_prefix('/').unwrap_or(url.path());
        let (encoded_share, relative_path) =
            encoded_path.split_once('/').unwrap_or((encoded_path, ""));
        if encoded_share.is_empty() {
            return Err(Error::invalid_input(format!(
                "UNC URL '{}' is missing a share name",
                url
            )));
        }

        let share = Path::from_url_path(encoded_share).map_err(|error| {
            Error::invalid_input(format!(
                "Failed to parse share name from UNC URL '{}': {}",
                url, error
            ))
        })?;
        if share.parts_count() != 1 || share.as_ref().contains('\\') {
            return Err(Error::invalid_input(format!(
                "UNC URL '{}' has an invalid share name",
                url
            )));
        }

        Ok(Some(UncPath {
            root: PathBuf::from(format!(r"\\{}\{}", host, share)),
            relative_path: Path::from_url_path(relative_path).map_err(|error| {
                Error::invalid_input(format!(
                    "Failed to parse path '{}' from UNC URL '{}': {}",
                    relative_path, url, error
                ))
            })?,
            store_prefix: format!("{}${}/{}", url.scheme(), host, encoded_share),
        }))
    }
}

impl FileStoreProvider {
    pub(super) fn default_backend() -> BackendConfig {
        Self::backend(
            LocalFileSystem::new(),
            None,
            "file",
            DEFAULT_LOCAL_BLOCK_SIZE,
            false,
        )
    }

    fn backend(
        inner: LocalFileSystem,
        local_dir_operations: Option<Arc<dyn LocalDirOperations>>,
        scheme: &str,
        block_size: usize,
        local_fsync: bool,
    ) -> BackendConfig {
        BackendConfig {
            store_prefix: None,
            backend: crate::storage::backend::object_store::from_object_store(Arc::new(inner)),
            direct_local_io: local_dir_operations.is_none(),
            local_io: if local_dir_operations.is_some() {
                crate::storage::LocalIo::Backend
            } else {
                match scheme {
                    "file-object-store" => crate::storage::LocalIo::Backend,
                    "file+uring" => crate::storage::LocalIo::Uring,
                    _ => crate::storage::LocalIo::Standard,
                }
            },
            is_local: true,
            // This scheme exercises cloud IO policies against a local backend.
            is_remote: scheme == "file-object-store",
            max_single_copy_size: None,
            local_dir_operations,
            local_fsync,
            scheme: scheme.to_owned(),
            block_size,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: false,
            io_parallelism: DEFAULT_LOCAL_IO_PARALLELISM,
            // Listed in full: reading a directory is one local walk whatever the page size,
            // so there is no request for a page to be pushed into.
        }
    }
}

#[async_trait::async_trait]
impl StorageProvider for FileStoreProvider {
    fn is_local(&self) -> bool {
        true
    }
    async fn new_backend(&self, base_path: Url, params: &StorageParams) -> Result<BackendConfig> {
        let block_size = params.block_size.unwrap_or(DEFAULT_LOCAL_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());
        let local_fsync = storage_options.local_fsync()?;

        #[cfg(windows)]
        let (inner, local_dir_operations) = match windows::extract_unc_path(&base_path)? {
            Some(unc_path) => {
                let inner = LocalFileSystem::new_with_prefix(unc_path.root)
                    .map_err(from_error)?
                    .with_fsync(local_fsync);
                let operations = FileSystemDirOperations {
                    local_file_system: inner.clone(),
                };
                (
                    inner,
                    Some(Arc::new(operations) as Arc<dyn LocalDirOperations>),
                )
            }
            None => (LocalFileSystem::new().with_fsync(local_fsync), None),
        };
        #[cfg(not(windows))]
        let inner = LocalFileSystem::new().with_fsync(local_fsync);
        #[cfg(not(windows))]
        let local_dir_operations = None;

        Ok(Self::backend(
            inner,
            local_dir_operations,
            base_path.scheme(),
            block_size,
            local_fsync,
        ))
    }

    fn extract_path(&self, url: &Url) -> Result<Path> {
        #[cfg(windows)]
        if let Some(unc_path) = windows::extract_unc_path(url)? {
            return Ok(unc_path.relative_path);
        }
        if let Ok(file_path) = url.to_file_path()
            && let Ok(path) = Path::from_absolute_path(&file_path)
        {
            return Ok(path);
        }

        Path::from_url_path(url.path()).map_err(|e| {
            Error::invalid_input(format!("Failed to parse path '{}': {}", url.path(), e))
        })
    }

    fn calculate_storage_prefix(
        &self,
        url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        #[cfg(windows)]
        if let Some(unc_path) = windows::extract_unc_path(url)? {
            return Ok(unc_path.store_prefix);
        }

        Ok(url.scheme().to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{create_dir_all, write};
    use std::path::Path as StdPath;

    use crate::storage::{StorageOptionsAccessor, uri_to_url};
    use rstest::rstest;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use tempfile::tempdir;

    use super::*;
    use crate::storage::Storage;

    #[rstest]
    #[case::direct("file", false)]
    #[case::backend("file-object-store", true)]
    #[cfg_attr(target_os = "linux", case::uring("file+uring", false))]
    #[tokio::test]
    async fn test_local_cloud_policy(#[case] scheme: &str, #[case] cloud: bool) {
        let (store, _) = Storage::from_uri(&format!("{scheme}:///tmp"))
            .await
            .unwrap();
        assert!(store.is_local());
        assert_eq!(store.is_cloud(), cloud);
    }

    #[rstest]
    #[case::direct("file")]
    #[case::storage("file-object-store")]
    #[tokio::test]
    async fn test_local_fsync_storage_option(#[case] scheme: &str) {
        let url = Url::parse(&format!("{scheme}:///")).unwrap();
        let default_store = crate::storage::StorageRegistry::default()
            .new_store(url.clone(), &StorageParams::default())
            .await
            .unwrap();
        assert!(!default_store.local_fsync);

        let params = StorageParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("local_fsync".to_owned(), "true".to_owned())]),
            ))),
            ..Default::default()
        };
        let store = crate::storage::StorageRegistry::default()
            .new_store(url, &params)
            .await
            .unwrap();
        assert!(store.local_fsync);

        let directory = tempdir().unwrap();
        let file_path = directory.path().join("data");
        let object_path = Path::from_absolute_path(&file_path).unwrap();
        store.put(&object_path, b"data").await.unwrap();
        assert_eq!(std::fs::read(file_path).unwrap(), b"data");
    }

    #[tokio::test]
    async fn test_invalid_local_fsync_storage_option() {
        let params = StorageParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("local_fsync".to_owned(), "sometimes".to_owned())]),
            ))),
            ..Default::default()
        };
        let error = crate::storage::StorageRegistry::default()
            .new_store(Url::parse("file:///").unwrap(), &params)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("local_fsync"), "{error}");
        assert!(error.to_string().contains("sometimes"), "{error}");
    }

    fn rooted_local_store(root: &StdPath) -> Storage {
        let inner = LocalFileSystem::new_with_prefix(root).unwrap();
        let local_dir_operations = Arc::new(FileSystemDirOperations {
            local_file_system: inner.clone(),
        });
        let mut config = BackendConfig::new(
            crate::storage::backend::object_store::from_object_store(Arc::new(inner)),
            "file",
        );
        config.local_dir_operations = Some(local_dir_operations);
        config.io_parallelism = DEFAULT_LOCAL_IO_PARALLELISM;
        crate::utils::testing::TestStorageProvider(config)
            .build(Url::parse("file:///").unwrap(), &StorageParams::default())
    }

    #[tokio::test]
    async fn test_rooted_remove_dir_all_removes_tree() {
        let sandbox = tempdir().unwrap();
        let root = sandbox.path().join("share");
        let dataset = root.join("dataset");
        create_dir_all(dataset.join("nested")).unwrap();
        write(dataset.join("nested/data"), "delete").unwrap();

        rooted_local_store(&root)
            .remove_dir_all(Path::from("dataset"))
            .await
            .unwrap();

        assert!(!dataset.exists(), "recursive deletion must remove the tree");
    }

    #[tokio::test]
    async fn test_rooted_staging_listing_uses_local_root() {
        let sandbox = tempdir().unwrap();
        let root = sandbox.path().join("share");
        let dataset = root.join("dataset");
        create_dir_all(dataset.join("nested#1")).unwrap();
        write(dataset.join("nested#1/file#1"), "staging").unwrap();
        let store = rooted_local_store(&root);
        let mut files = store.read_local_staging_files(&Path::from("dataset"));
        let file = futures::TryStreamExt::try_next(&mut files)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            file.location,
            Path::parse("dataset/nested#1/file#1").unwrap()
        );
        assert_eq!(file.size, 7);
        store.delete(&file.location).await.unwrap();
        assert!(!dataset.join("nested#1/file#1").exists());
        assert!(
            futures::TryStreamExt::try_next(&mut files)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_rooted_canonicalize_uses_local_root() {
        let sandbox = tempdir().unwrap();
        let root = sandbox.path().join("share");
        let dataset = root.join("dataset");
        create_dir_all(&dataset).unwrap();
        let store = rooted_local_store(&root);

        let canonical = store
            .canonicalize_local_path(&Path::from("dataset"))
            .await
            .unwrap()
            .expect("rooted local paths should canonicalize");

        assert_eq!(
            canonical,
            std::fs::canonicalize(&dataset).unwrap(),
            "canonicalization must use the store's filesystem root"
        );
        assert!(!store.has_direct_local_paths());
        #[cfg(unix)]
        {
            symlink(&dataset, root.join("alias")).unwrap();
            assert_eq!(
                store
                    .canonicalize_local_path(&Path::from("alias"))
                    .await
                    .unwrap(),
                Some(canonical)
            );
        }
        assert!(
            store
                .canonicalize_local_path(&Path::from("missing"))
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            Storage::memory()
                .canonicalize_local_path(&Path::from("dataset"))
                .await
                .unwrap()
                .is_none()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_rooted_remove_dir_all_does_not_follow_directory_symlink() {
        let sandbox = tempdir().unwrap();
        let root = sandbox.path().join("share");
        let dataset = root.join("dataset");
        let outside = sandbox.path().join("outside");
        create_dir_all(&dataset).unwrap();
        create_dir_all(&outside).unwrap();
        let sentinel = outside.join("sentinel");
        write(&sentinel, "keep").unwrap();
        symlink(&outside, dataset.join("link")).unwrap();

        rooted_local_store(&root)
            .remove_dir_all(Path::from("dataset"))
            .await
            .unwrap();

        assert!(
            sentinel.exists(),
            "recursive deletion must not follow links"
        );
        assert!(!dataset.exists(), "recursive deletion must remove the tree");
    }

    #[test]
    fn test_file_store_path() {
        let provider = FileStoreProvider;

        let cases = [
            ("file:///", ""),
            ("file:///usr/local/bin", "usr/local/bin"),
            ("file-object-store:///path/to/file", "path/to/file"),
            ("file:///path/to/foo/../bar", "path/to/bar"),
        ];

        for (uri, expected_path) in cases {
            let url = uri_to_url(uri).unwrap();
            let path = provider.extract_path(&url).unwrap();
            assert_eq!(path.as_ref(), expected_path, "uri: '{}'", uri);
        }
    }

    #[test]
    fn test_calculate_storage_prefix() {
        let provider = FileStoreProvider;
        assert_eq!(
            "file",
            provider
                .calculate_storage_prefix(&Url::parse("file:///etc").unwrap(), None)
                .unwrap()
        );
    }

    #[test]
    fn test_calculate_storage_prefix_for_file_storage() {
        let provider = FileStoreProvider;
        assert_eq!(
            "file-object-store",
            provider
                .calculate_storage_prefix(&Url::parse("file-object-store:///etc").unwrap(), None)
                .unwrap()
        );
    }

    #[test]
    #[cfg(windows)]
    fn test_file_store_path_windows() {
        let provider = FileStoreProvider;

        let cases = [
            (
                "C:\\Users\\ADMINI~1\\AppData\\Local\\",
                "C:/Users/ADMINI~1/AppData/Local",
            ),
            (
                "C:\\Users\\ADMINI~1\\AppData\\Local\\..\\",
                "C:/Users/ADMINI~1/AppData",
            ),
            (
                "file-object-store:///C:/Users/ADMINI~1/AppData/Local",
                "C:/Users/ADMINI~1/AppData/Local",
            ),
            (
                "file:///C:/Users/RUNNER~1/AppData/Local/Temp/tmpm49j_w0f",
                "C:/Users/RUNNER~1/AppData/Local/Temp/tmpm49j_w0f",
            ),
            (
                "file://192.168.0.1/My%20Share/data/my-dataset.lance",
                "data/my-dataset.lance",
            ),
        ];

        for (uri, expected_path) in cases {
            let url = uri_to_url(uri).unwrap();
            let path = provider.extract_path(&url).unwrap();
            assert_eq!(path.as_ref(), expected_path);
        }
    }

    #[test]
    #[cfg(windows)]
    fn test_unc_share_path() {
        let url = Url::parse("file://server/My%20Share/data/my-dataset.lance").unwrap();
        let unc_path = windows::extract_unc_path(&url).unwrap().unwrap();

        assert_eq!(
            unc_path.root,
            std::path::PathBuf::from(r"\\server\My Share")
        );
        assert_eq!(unc_path.relative_path.as_ref(), "data/my-dataset.lance");
        assert_eq!(unc_path.store_prefix, "file$server/My%20Share");

        let storage_url =
            Url::parse("file-object-store://server/My%20Share/data/my-dataset.lance").unwrap();
        assert!(windows::extract_unc_path(&storage_url).unwrap().is_none());
    }
}
