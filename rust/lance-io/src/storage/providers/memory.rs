// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::storage::BackendConfig;
use std::{collections::HashMap, sync::Arc};

use crate::storage::path::Path;
use crate::storage::{
    DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_LOCAL_BLOCK_SIZE, StorageParams, StorageProvider,
};
use ::object_store::memory::InMemory;
use lance_core::error::Result;
use url::Url;

/// Provides a fresh in-memory object store for each call to `new_backend`.
#[derive(Default, Debug)]
pub struct MemoryStoreProvider;

impl MemoryStoreProvider {
    pub(super) fn default_backend() -> BackendConfig {
        let block_size = DEFAULT_LOCAL_BLOCK_SIZE;
        BackendConfig {
            store_prefix: None,
            backend: crate::storage::backend::object_store::from_object_store(Arc::new(
                InMemory::new(),
            )),
            local_dir_operations: None,
            local_fsync: false,
            direct_local_io: false,
            local_io: crate::storage::LocalIo::Backend,
            is_local: false,
            is_remote: false,
            max_single_copy_size: None,
            scheme: String::from("memory"),
            block_size,
            use_constant_size_upload_parts: false,
            list_is_lexically_ordered: true,
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            // Listed in full: the store is already in memory, so a page costs no less than
            // the directory does.
        }
    }
}

#[async_trait::async_trait]
impl StorageProvider for MemoryStoreProvider {
    async fn new_backend(&self, _base_path: Url, params: &StorageParams) -> Result<BackendConfig> {
        let mut config = Self::default_backend();
        config.block_size = params.block_size.unwrap_or(DEFAULT_LOCAL_BLOCK_SIZE);
        Ok(config)
    }

    fn extract_path(&self, url: &Url) -> Result<Path> {
        let mut output = String::new();
        if let Some(domain) = url.domain() {
            output.push_str(domain);
        }
        output.push_str(url.path());
        // The in-memory store uses the Path directly as a key with no HTTP layer,
        // so there is no re-encoding step and thus no double-encoding to avoid.
        // Path::from also tolerates the empty segments that local temp paths embed.
        Ok(Path::from(output))
    }

    fn calculate_storage_prefix(
        &self,
        _url: &Url,
        _storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        Ok("memory".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_store_path() {
        let provider = MemoryStoreProvider;

        let url = Url::parse("memory://path/to/file").unwrap();
        let path = provider.extract_path(&url).unwrap();
        let expected_path = Path::from("path/to/file");
        assert_eq!(path, expected_path);
    }

    #[test]
    fn test_calculate_storage_prefix() {
        let provider = MemoryStoreProvider;
        assert_eq!(
            "memory",
            provider
                .calculate_storage_prefix(&Url::parse("memory://etc").unwrap(), None)
                .unwrap()
        );
    }
}
