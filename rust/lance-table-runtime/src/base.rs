// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use futures::TryStreamExt;
use lance_core::{Error, Result};
use lance_io::object_store::{
    ChainedWrappingObjectStore, ObjectStore, ObjectStoreParams, StorageOptions,
    StorageOptionsAccessor, WrappingObjectStore, parse_base_scoped_key,
};
use lance_table::format::{BasePath, DeletionFile, IndexMetadata};
use lance_table::io::{commit::VERSIONS_DIR, deletion::DELETIONS_DIR};
use object_store::path::Path;

use crate::paths::{DATA_DIR, INDICES_DIR, TRANSACTIONS_DIR};
use crate::refs::Refs;
use crate::{Table, TableState, TableView};

impl TableState {
    pub fn store_params_for_base(
        &self,
        base_path: Option<&lance_table::format::BasePath>,
    ) -> ObjectStoreParams {
        // Base-specific bindings are exact ObjectStoreParams keyed by
        // `BasePath.path` and are used as-is. Otherwise the dataset-level
        // default params are resolved for the base scope: `base_<id>.<key>`
        // storage options overlay the shared defaults for that base.
        if let Some(params) = base_path.and_then(|base_path| {
            self.base_store_params
                .as_ref()
                .and_then(|params| params.get(&base_path.path))
        }) {
            return params.clone();
        }
        let default_params = self.store_params.as_deref().cloned().unwrap_or_default();
        match default_params.scoped_to_base(base_path.map(|base_path| base_path.id)) {
            Cow::Owned(scoped_params) => scoped_params,
            Cow::Borrowed(_) => default_params,
        }
    }

    /// Returns the initial storage options without triggering any refresh.
    ///
    /// For the latest refreshed options, use [`Self::latest_storage_options`].
    pub fn initial_storage_options(&self) -> Option<&HashMap<String, String>> {
        self.store_params
            .as_ref()
            .and_then(|params| params.storage_options())
    }

    /// Returns the storage options provider used when opening this dataset, if any.
    pub fn storage_options_provider(
        &self,
    ) -> Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>> {
        self.store_params
            .as_ref()
            .and_then(|params| params.storage_options_accessor.as_ref())
            .and_then(|accessor| accessor.provider().cloned())
    }

    /// Returns the unified storage options accessor for this dataset, if any.
    ///
    /// The accessor handles both static and dynamic storage options with automatic
    /// caching and refresh. Use [`StorageOptionsAccessor::get_storage_options`] to
    /// get the latest options.
    pub fn storage_options_accessor(&self) -> Option<Arc<StorageOptionsAccessor>> {
        self.store_params
            .as_ref()
            .and_then(|params| params.get_accessor())
    }

    /// Returns the latest (possibly refreshed) storage options.
    ///
    /// If a dynamic storage options provider is configured, this will return
    /// the cached options if still valid, or fetch fresh options if expired.
    ///
    /// For the initial static options without refresh, use [`Self::initial_storage_options`].
    ///
    /// # Returns
    ///
    /// - `Ok(Some(options))` - Storage options are available (static or refreshed)
    /// - `Ok(None)` - No storage options were configured for this dataset
    /// - `Err(...)` - Error occurred while fetching/refreshing options from provider
    pub async fn latest_storage_options(&self) -> Result<Option<StorageOptions>> {
        // First check if we have an accessor (handles both static and dynamic options)
        if let Some(accessor) = self.storage_options_accessor() {
            let options = accessor.get_storage_options().await?;
            return Ok(Some(options));
        }

        // Fallback to initial storage options if no accessor
        Ok(self.initial_storage_options().cloned().map(StorageOptions))
    }

    pub fn data_dir(&self) -> Path {
        self.base.clone().join(DATA_DIR)
    }

    pub fn indices_dir(&self) -> Path {
        self.base.clone().join(INDICES_DIR)
    }

    pub fn transactions_dir(&self) -> Path {
        self.base.clone().join(TRANSACTIONS_DIR)
    }

    pub fn deletions_dir(&self) -> Path {
        self.base.clone().join(DELETIONS_DIR)
    }

    pub fn versions_dir(&self) -> Path {
        self.base.clone().join(VERSIONS_DIR)
    }

    /// Replace the primary store and discard base stores derived from the previous binding.
    pub fn set_object_store(
        &mut self,
        object_store: Arc<ObjectStore>,
        store_params: Option<ObjectStoreParams>,
    ) {
        self.object_store = object_store;
        self.clear_base_object_stores();
        if let Some(store_params) = store_params {
            self.store_params = Some(Box::new(store_params));
        }
    }

    /// Discard resolved additional stores when their scope or bindings change.
    pub fn clear_base_object_stores(&mut self) {
        self.base_object_stores = Default::default();
    }
}

impl TableView<'_> {
    #[doc(hidden)]
    pub fn blob_base_path(&self, base_id: Option<u32>) -> Result<Path> {
        match base_id {
            Some(id) => self
                .manifest
                .base_paths
                .get(&id)
                .ok_or_else(|| {
                    Error::invalid_input(format!("Blob references unknown base_id {id}"))
                })?
                .extract_path(self.state.session.store_registry()),
            None => Ok(self.state.base.clone()),
        }
    }

    async fn base_object_store(&self, base_id: u32) -> Result<Arc<ObjectStore>> {
        let base_path = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
            Error::invalid_input(format!("Dataset base path with ID {} not found", base_id))
        })?;
        if base_path.path == self.state.uri
            && !self
                .state
                .base_store_params
                .as_ref()
                .is_some_and(|params| params.contains_key(&base_path.path))
            && self.state.store_params.as_ref().is_none_or(|params| {
                matches!(params.scoped_to_base(Some(base_id)), Cow::Borrowed(_))
            })
        {
            return Ok(self.state.object_store.clone());
        }
        let store_params = self.state.store_params_for_base(Some(base_path));

        let cell = {
            let mut stores = self.state.base_object_stores.lock().unwrap();
            stores.entry(base_id).or_default().clone()
        };
        let store = cell
            .get_or_try_init(|| async {
                // Wrappers define a request or execution scope. Keep the
                // fully resolved store in this dataset's OnceCell, but do not
                // also put it in the global registry: provider-local state
                // such as GCS AIMD token buckets must not cross that scope.
                let (store, _) = if store_params.object_store_wrapper.is_some() {
                    ObjectStore::from_uri_and_params_uncached(
                        self.state.session.store_registry(),
                        &base_path.path,
                        &store_params,
                    )
                    .await?
                } else {
                    ObjectStore::from_uri_and_params(
                        self.state.session.store_registry(),
                        &base_path.path,
                        &store_params,
                    )
                    .await?
                };
                Ok::<_, Error>(store)
            })
            .await?;
        Ok(store.clone())
    }

    /// Resolve the data directory for a given base_id.
    ///
    /// If `base_id` is `None`, returns the default data directory.
    pub fn data_file_dir_for_base(&self, base_id: Option<u32>) -> Result<Path> {
        match base_id {
            Some(base_id) => {
                let base_path = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                    Error::invalid_input(format!("base_path id {} not found", base_id))
                })?;
                let path = base_path.extract_path(self.state.session.store_registry())?;
                if base_path.is_dataset_root {
                    Ok(path.join(DATA_DIR))
                } else {
                    Ok(path)
                }
            }
            None => Ok(self.state.base.clone().join(DATA_DIR)),
        }
    }

    pub fn dataset_dir_for_deletion(&self, deletion_file: &DeletionFile) -> Result<Path> {
        match deletion_file.base_id.as_ref() {
            Some(base_id) => {
                let base_paths = &self.manifest.base_paths;
                let base_path = base_paths.get(base_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "base_path id {} not found for deletion_file {:?}",
                        base_id, deletion_file
                    ))
                })?;

                if !base_path.is_dataset_root {
                    return Err(Error::internal(format!(
                        "base_path id {} is not a dataset root for deletion_file {:?}",
                        base_id, deletion_file
                    )));
                }
                base_path.extract_path(self.state.session.store_registry())
            }
            None => Ok(self.state.base.clone()),
        }
    }

    /// Get the indices directory for a specific index, considering its base_id
    pub fn indice_files_dir(&self, index: &IndexMetadata) -> Result<Path> {
        match index.base_id.as_ref() {
            Some(base_id) => {
                let base_paths = &self.manifest.base_paths;
                let base_path = base_paths.get(base_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "base_path id {} not found for index {}",
                        base_id, index.uuid
                    ))
                })?;
                let path = base_path.extract_path(self.state.session.store_registry())?;
                if base_path.is_dataset_root {
                    Ok(path.join(INDICES_DIR))
                } else {
                    // For non-dataset-root base paths, we assume the path already points to the indices directory
                    Ok(path)
                }
            }
            None => Ok(self.state.base.clone().join(INDICES_DIR)),
        }
    }

    /// Resolve the object store for the primary dataset or an additional base.
    ///
    /// Pass `None` to get the primary dataset object store. Pass `Some(base_id)`
    /// when resolving a file whose metadata references an additional base.
    pub async fn object_store(&self, base_id: Option<u32>) -> Result<Arc<ObjectStore>> {
        match base_id {
            Some(base_id) => self.base_object_store(base_id).await,
            None => Ok(self.state.object_store.clone()),
        }
    }

    /// Clone this dataset with extra object store wrappers applied to all read stores.
    ///
    /// The returned dataset uses the wrappers for the already-open primary object
    /// store and appends the same wrappers to the dataset-level and base-specific
    /// object store params used when additional base stores are opened later.
    pub fn with_object_store_wrappers(
        &self,
        wrappers: impl IntoIterator<Item = Arc<dyn WrappingObjectStore>>,
    ) -> Table {
        let wrappers = wrappers.into_iter().collect::<Vec<_>>();
        if wrappers.is_empty() {
            return self.to_table();
        }

        let mut cloned = self.to_table();
        // Each wrapper application defines a new store lifetime. Keep base
        // stores alive within the derived dataset without sharing stateful
        // provider layers (such as an AIMD throttle) with other scopes.
        cloned.state.clear_base_object_stores();
        let mut object_store = self.state.object_store.as_ref().clone();
        for wrapper in &wrappers {
            object_store.apply_wrapper(wrapper.as_ref());
        }
        cloned.state.object_store = Arc::new(object_store);
        cloned.refs = Refs::new(
            cloned.state.object_store.clone(),
            cloned.state.commit_handler.clone(),
            cloned.branch_location(),
        );

        let store_params = self
            .state
            .store_params
            .as_deref()
            .cloned()
            .unwrap_or_default();
        cloned.state.store_params = Some(Box::new(append_object_store_wrappers(
            store_params,
            &wrappers,
        )));
        cloned.state.base_store_params =
            self.state
                .base_store_params
                .as_ref()
                .map(|base_store_params| {
                    Arc::new(
                        base_store_params
                            .iter()
                            .map(|(base_path, store_params)| {
                                (
                                    base_path.clone(),
                                    append_object_store_wrappers(store_params.clone(), &wrappers),
                                )
                            })
                            .collect(),
                    )
                });
        cloned
    }
}

fn append_object_store_wrappers(
    mut store_params: ObjectStoreParams,
    wrappers: &[Arc<dyn WrappingObjectStore>],
) -> ObjectStoreParams {
    let mut all_wrappers = Vec::with_capacity(
        store_params.object_store_wrapper.as_ref().map_or(0, |_| 1) + wrappers.len(),
    );
    if let Some(wrapper) = store_params.object_store_wrapper.take() {
        all_wrappers.push(wrapper);
    }
    all_wrappers.extend(wrappers.iter().cloned());
    store_params.object_store_wrapper = match all_wrappers.len() {
        0 => None,
        1 => all_wrappers.pop(),
        _ => Some(Arc::new(ChainedWrappingObjectStore::new(all_wrappers))),
    };
    store_params
}

/// Log base-scoped storage options (`base_<id>.<key>`) whose id does not
/// match any registered base path. Unregistered entries are ignored during
/// resolution. The open path logs at debug, since options may legitimately be
/// vended for bases the loaded version does not register; the write path logs
/// at warn, since ids are already assigned there and an unmatched id is much
/// more likely a mistake.
pub fn log_unregistered_base_scoped_options(
    store_params: Option<&ObjectStoreParams>,
    base_paths: &HashMap<u32, BasePath>,
    level: log::Level,
) {
    if !log::log_enabled!(level) {
        return;
    }
    let Some(options) = store_params.and_then(|params| params.storage_options()) else {
        return;
    };
    let unregistered = options
        .keys()
        .filter_map(|key| parse_base_scoped_key(key).map(|(id, _)| id))
        .filter(|id| !base_paths.contains_key(id))
        .collect::<BTreeSet<_>>();
    if !unregistered.is_empty() {
        log::log!(
            level,
            "Ignoring base-scoped storage options for unregistered base path ids: {:?}",
            unregistered
        );
    }
}

/// Whether anything at all lives under `prefix`. Stops at the first entry, so this stays
/// cheap even on a storage root holding millions of objects.
pub async fn has_any_entry(object_store: &ObjectStore, prefix: &Path) -> Result<bool> {
    match object_store.list(Some(prefix.clone())).try_next().await {
        Ok(entry) => Ok(entry.is_some()),
        // Local filesystems report a missing directory as an error where object stores
        // return an empty listing. Neither has anything to protect.
        Err(e) if e.is_not_found() => Ok(false),
        Err(e) => Err(e),
    }
}
