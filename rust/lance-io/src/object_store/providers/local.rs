// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

#[cfg(any(windows, test))]
use crate::object_store::LocalDirOperations;
use crate::object_store::{
    DEFAULT_LOCAL_BLOCK_SIZE, DEFAULT_LOCAL_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, StorageOptions,
};
use lance_core::Error;
use lance_core::error::Result;
use object_store::{local::LocalFileSystem, path::Path};
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
    async fn remove_dir_all(&self, path: &Path) -> Result<()> {
        let local_path = self.local_file_system.path_to_filesystem(path)?;
        let object_store_path = path.to_string();
        tokio::task::spawn_blocking(move || {
            std::fs::remove_dir_all(local_path).map_err(|error| match error.kind() {
                ErrorKind::NotFound => Error::not_found(object_store_path),
                _ => Error::from(error),
            })
        })
        .await
        .map_err(|error| Error::io(format!("recursive directory removal task failed: {error}")))?
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

#[async_trait::async_trait]
impl ObjectStoreProvider for FileStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_LOCAL_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());
        let download_retry_count = storage_options.download_retry_count();

        #[cfg(windows)]
        let (inner, local_dir_operations) = match windows::extract_unc_path(&base_path)? {
            Some(unc_path) => {
                let inner = LocalFileSystem::new_with_prefix(unc_path.root)?;
                let operations = FileSystemDirOperations {
                    local_file_system: inner.clone(),
                };
                (
                    inner,
                    Some(Arc::new(operations) as Arc<dyn LocalDirOperations>),
                )
            }
            None => (LocalFileSystem::new(), None),
        };
        #[cfg(not(windows))]
        let inner = LocalFileSystem::new();
        #[cfg(not(windows))]
        let local_dir_operations = None;

        Ok(ObjectStore {
            inner: Arc::new(inner),
            local_dir_operations,
            scheme: base_path.scheme().to_owned(),
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: false,
            io_parallelism: DEFAULT_LOCAL_IO_PARALLELISM,
            download_retry_count,
            scheduler_error_mode: Default::default(),
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
            // Listed in full: reading a directory is one local walk whatever the page size,
            // so there is no request for a page to be pushed into.
            paginated_lister: None,
        })
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

    fn calculate_object_store_prefix(
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

    use crate::object_store::uri_to_url;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use tempfile::tempdir;

    use super::*;

    fn rooted_local_store(root: &StdPath) -> ObjectStore {
        let inner = LocalFileSystem::new_with_prefix(root).unwrap();
        let local_dir_operations = Arc::new(FileSystemDirOperations {
            local_file_system: inner.clone(),
        });
        ObjectStore {
            inner: Arc::new(inner),
            local_dir_operations: Some(local_dir_operations),
            scheme: "file".to_owned(),
            block_size: DEFAULT_LOCAL_BLOCK_SIZE,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: false,
            io_parallelism: DEFAULT_LOCAL_IO_PARALLELISM,
            download_retry_count: 0,
            scheduler_error_mode: Default::default(),
            io_tracker: Default::default(),
            store_prefix: "file$rooted-test".to_owned(),
            paginated_lister: None,
        }
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
    fn test_calculate_object_store_prefix() {
        let provider = FileStoreProvider;
        assert_eq!(
            "file",
            provider
                .calculate_object_store_prefix(&Url::parse("file:///etc").unwrap(), None)
                .unwrap()
        );
    }

    #[test]
    fn test_calculate_object_store_prefix_for_file_object_store() {
        let provider = FileStoreProvider;
        assert_eq!(
            "file-object-store",
            provider
                .calculate_object_store_prefix(
                    &Url::parse("file-object-store:///etc").unwrap(),
                    None
                )
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

        let object_store_url =
            Url::parse("file-object-store://server/My%20Share/data/my-dataset.lance").unwrap();
        assert!(
            windows::extract_unc_path(&object_store_url)
                .unwrap()
                .is_none()
        );
    }
}
