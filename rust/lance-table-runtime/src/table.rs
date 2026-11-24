// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_core::{
    Error, Result,
    datatypes::{Projectable, Schema},
};
use lance_file::reader::FileReaderOptions;
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, StorageOptions, StorageOptionsAccessor, WrappingObjectStore,
};
use lance_table::feature_flags::ensure_can_read_manifest;
use lance_table::format::{BasePath, DataFile, Fragment, Manifest};
use lance_table::io::{
    commit::{CommitHandler, ManifestLocation, VERSIONS_DIR},
    manifest::read_manifest,
};
use lance_table::transaction::Transaction;
use object_store::path::Path;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::base::{has_any_entry, log_unregistered_base_scoped_options};
use crate::branch_location::BranchLocation;
use crate::caches::DSMetadataCache;
use crate::refs::{self, BranchIdentifier, BranchReservation, Branches, Refs, Tags};
use crate::{FileFragment, TableBuilder, TableSession};

/// A loaded Lance table snapshot with its manifest and reference handles.
///
/// Open a table with [crate::TableBuilder] to inspect the latest version or
/// resolve a specific version, branch, or tag.
#[derive(Clone)]
pub struct Table {
    pub(crate) state: TableState,
    /// The checked-out manifest.
    pub manifest: Arc<Manifest>,
    /// Branch and tag operations bound to this table.
    pub refs: Refs,
}

impl Projectable for Table {
    fn schema(&self) -> &Schema {
        self.schema()
    }
}

impl std::fmt::Debug for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("uri", &self.state.uri)
            .field("base", &self.state.base)
            .field("version", &self.manifest.version)
            .finish()
    }
}

/// Runtime state of a checked-out table snapshot, shared between embedders.
#[doc(hidden)]
#[derive(Clone)]
pub struct TableState {
    pub object_store: Arc<ObjectStore>,
    pub commit_handler: Arc<dyn CommitHandler>,
    pub uri: String,
    pub base: Path,
    pub manifest_location: ManifestLocation,
    pub(crate) session: Arc<TableSession>,
    // Keep the incarnation opened by this handle; the name may later be reused.
    pub branch_identifier: Arc<BranchIdentifier>,
    pub fragment_bitmap: Arc<RoaringBitmap>,
    pub metadata_cache: Arc<DSMetadataCache>,
    pub file_reader_options: Option<Arc<FileReaderOptions>>,
    pub store_params: Option<Box<ObjectStoreParams>>,
    pub base_store_params: Option<Arc<HashMap<String, ObjectStoreParams>>>,
    pub(crate) base_object_stores: BaseObjectStores,
}

/// The cell coalesces concurrent first resolutions into one build.
type BaseObjectStores = Arc<Mutex<HashMap<u32, Arc<OnceCell<Arc<ObjectStore>>>>>>;

impl TableState {
    /// Initialize runtime bindings and derived metadata for a loaded snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        object_store: Arc<ObjectStore>,
        base: Path,
        uri: String,
        manifest: &Manifest,
        manifest_location: ManifestLocation,
        session: Arc<TableSession>,
        metadata_cache: Arc<DSMetadataCache>,
        commit_handler: Arc<dyn CommitHandler>,
        branch_identifier: Arc<BranchIdentifier>,
        file_reader_options: Option<FileReaderOptions>,
        store_params: Option<ObjectStoreParams>,
        base_store_params: Option<Arc<HashMap<String, ObjectStoreParams>>>,
    ) -> Self {
        Self {
            object_store,
            base,
            uri,
            manifest_location,
            session,
            commit_handler,
            branch_identifier,
            fragment_bitmap: Arc::new(manifest.fragments.iter().map(|f| f.id as u32).collect()),
            metadata_cache,
            file_reader_options: file_reader_options.map(Arc::new),
            store_params: store_params.map(Box::new),
            base_store_params,
            base_object_stores: Default::default(),
        }
    }

    /// Replace the session associated with the runtime bindings.
    pub fn set_session(&mut self, session: Arc<TableSession>) {
        self.session = session;
    }

    /// Replace the manifest and location together and refresh derived metadata.
    pub fn set_manifest(
        &mut self,
        current: &mut Arc<Manifest>,
        manifest: Arc<Manifest>,
        manifest_location: ManifestLocation,
    ) {
        self.update_fragment_and_base_state(current, &manifest);
        *current = manifest;
        self.manifest_location = manifest_location;
    }

    fn update_fragment_and_base_state(&mut self, current: &Manifest, new: &Manifest) {
        if new.base_paths != current.base_paths {
            self.clear_base_object_stores();
        }
        self.fragment_bitmap = Arc::new(new.fragments.iter().map(|f| f.id as u32).collect());
    }
}

/// A borrowed snapshot for embedders that expose their own manifest and refs.
#[doc(hidden)]
pub struct TableView<'a> {
    pub(crate) state: &'a TableState,
    pub(crate) manifest: &'a Arc<Manifest>,
    pub(crate) refs: &'a Refs,
}

impl<'a> TableView<'a> {
    /// Borrow the state, manifest and refs of the same checked-out snapshot.
    pub fn new(state: &'a TableState, manifest: &'a Arc<Manifest>, refs: &'a Refs) -> Self {
        Self {
            state,
            manifest,
            refs,
        }
    }

    /// Materialize an owned snapshot with the same storage and cache bindings.
    pub fn to_table(&self) -> Table {
        Table::from_parts(self.state.clone(), self.manifest.clone(), self.refs.clone())
    }

    /// View the files of another manifest of this table through its base paths.
    pub fn with_manifest_view(&self, manifest: Arc<Manifest>) -> Table {
        let mut state = self.state.clone();
        state.update_fragment_and_base_state(self.manifest, &manifest);
        Table::from_parts(state, manifest, self.refs.clone())
    }

    fn already_checked_out(&self, location: &ManifestLocation, branch_name: Option<&str>) -> bool {
        // The ETag is an opaque object-generation token, not a content hash.
        // Comparing the token still prevents reusing this Dataset's manifest
        // after the physical object was replaced, for example by a recent
        // drop/recreate at the same URI and version.
        self.manifest.branch.as_deref() == branch_name
            && self.manifest.version == location.version
            && self.state.manifest_location.naming_scheme == location.naming_scheme
            && location.e_tag.as_ref().is_some_and(|e_tag| {
                self.state
                    .manifest_location
                    .e_tag
                    .as_ref()
                    .is_some_and(|current_e_tag| e_tag == current_e_tag)
            })
    }

    async fn checkout_by_ref(
        &self,
        version_number: Option<u64>,
        branch: Option<&str>,
    ) -> Result<Option<Table>> {
        let standardized_branch = branch.and_then(refs::standardize_branch);
        // Reject malformed names at the boundary (mirroring the branch CRUD
        // paths) so they fail as InvalidRef instead of tripping the wrong-chain
        // check below
        if let Some(branch_name) = standardized_branch.as_deref()
            && !Branches::is_main_branch(Some(branch_name))
        {
            refs::check_valid_branch(branch_name)?;
        }
        let branch_identifier = self
            .branches()
            .get_identifier(standardized_branch.as_deref())
            .await?;
        if standardized_branch == self.manifest.branch
            && self.state.branch_identifier.as_ref() != &branch_identifier
        {
            return Err(Error::RefNotFound {
                message: format!(
                    "branch {} was deleted and recreated",
                    refs::normalize_branch(branch)
                ),
            });
        }

        let new_location = self
            .branch_location()
            .find_branch(standardized_branch.as_deref())?;

        let manifest_location = if let Some(version_number) = version_number {
            self.state
                .commit_handler
                .resolve_version_location(
                    &new_location.path,
                    version_number,
                    &self.state.object_store.inner,
                )
                .await?
        } else {
            self.state
                .commit_handler
                .resolve_latest_location(&new_location.path, &self.state.object_store)
                .await?
        };

        if self.already_checked_out(&manifest_location, standardized_branch.as_deref()) {
            self.ensure_branch_exists().await?;
            return Ok(None);
        }

        let manifest = Table::get_manifest(
            self.state.object_store.as_ref(),
            &manifest_location,
            &new_location.uri,
            self.state.session.as_ref(),
        )
        .await?;

        // The resolved manifest must belong to the requested branch. A mismatch
        // means the commit handler resolved against a different chain (for
        // example an external manifest store that ignores branch-qualified
        // paths); error loudly rather than hand back another branch's data.
        if manifest.branch.as_deref() != standardized_branch.as_deref() {
            return Err(Error::internal(format!(
                "checkout of branch '{}' at version {} resolved a manifest belonging to branch '{}'",
                refs::normalize_branch(branch),
                manifest.version,
                refs::normalize_branch(manifest.branch.as_deref()),
            )));
        }

        let table = Table::checkout_manifest(
            self.state.object_store.clone(),
            new_location.path,
            new_location.uri,
            manifest,
            manifest_location,
            self.state.session.clone(),
            self.state.commit_handler.clone(),
            self.state.file_reader_options.as_deref().cloned(),
            self.state.store_params.as_deref().cloned(),
            self.state.base_store_params.clone(),
        )
        .await?;
        if table.state.branch_identifier.as_ref() != &branch_identifier {
            return Err(Error::RefNotFound {
                message: format!(
                    "branch {} was deleted and recreated during checkout",
                    refs::normalize_branch(branch)
                ),
            });
        }
        Ok(Some(table))
    }

    /// Resolve the requested snapshot. A matching manifest needs no replacement.
    pub async fn checkout_version(&self, version: impl Into<refs::Ref>) -> Result<Option<Table>> {
        let reference: refs::Ref = version.into();
        match reference {
            refs::Ref::Version(branch, version_number) => {
                self.checkout_by_ref(version_number, branch.as_deref())
                    .await
            }
            refs::Ref::VersionNumber(version_number) => {
                self.checkout_by_ref(Some(version_number), self.manifest.branch.as_deref())
                    .await
            }
            refs::Ref::Tag(tag_name) => {
                let tag_contents = self.tags().get(tag_name.as_str()).await?;
                self.checkout_by_ref(Some(tag_contents.version), tag_contents.branch.as_deref())
                    .await
            }
        }
    }

    pub async fn latest_manifest(&self) -> Result<(Arc<Manifest>, ManifestLocation)> {
        self.ensure_branch_exists().await?;
        let location = self
            .state
            .commit_handler
            .resolve_latest_location(&self.state.base, &self.state.object_store)
            .await?;

        if self.already_checked_out(&location, self.manifest.branch.as_deref()) {
            ensure_can_read_manifest(self.manifest)?;
            self.ensure_branch_exists().await?;
            return Ok((self.manifest.clone(), self.state.manifest_location.clone()));
        }
        let manifest = Table::get_manifest(
            &self.state.object_store,
            &location,
            &self.state.uri,
            &self.state.session,
        )
        .await?;
        self.ensure_branch_exists().await?;
        Ok((manifest, location))
    }

    pub fn branch_location(&self) -> BranchLocation {
        BranchLocation {
            path: self.state.base.clone(),
            uri: self.state.uri.clone(),
            branch: self.manifest.branch.clone(),
        }
    }

    pub async fn branch_identifier(&self) -> Result<BranchIdentifier> {
        self.ensure_branch_exists().await?;
        Ok(self.state.branch_identifier.as_ref().clone())
    }

    pub async fn ensure_branch_exists(&self) -> Result<()> {
        if let Some(branch) = self.manifest.branch.as_deref() {
            let current = self.branches().get(branch).await?;
            if &current.identifier != self.state.branch_identifier.as_ref() {
                return Err(Error::RefNotFound {
                    message: format!("branch {branch} was deleted and recreated"),
                });
            }
        }
        Ok(())
    }

    /// Get the currently checked-out version details.
    ///
    /// This constructs a full [Version], including summary metadata derived
    /// from the loaded manifest fragments.
    pub fn version(&self) -> Version {
        let mut version = Version::from(self.manifest.as_ref());
        // `Version::from` only sees the manifest itself, not where it was read
        // from. This dataset already knows the on-disk size, so report it here
        // too; otherwise `get_version()` would disagree with `versions()`,
        // which takes the size from the manifest location.
        version.manifest_size = self.manifest_location().size;
        version
    }

    /// Return whether the dataset has a newer committed version.
    pub async fn is_stale(&self) -> Result<bool> {
        let latest_version = self.state.latest_version_id().await?;
        Ok(latest_version != self.manifest.version)
    }

    /// Return whether the immediate attached successor manifest exists.
    ///
    /// This is a fast contiguous-history probe. It does not resolve the latest
    /// version and may return `false` if intermediate manifests have been
    /// removed. Callers that need a general freshness check should use
    /// [`Self::is_stale`].
    #[doc(hidden)]
    pub async fn has_successor_version(&self) -> Result<bool> {
        let Some(next_version) = self.manifest.version.checked_add(1) else {
            return Ok(false);
        };
        if lance_table::format::is_detached_version(next_version) {
            return Ok(false);
        }

        let exists = self
            .state
            .commit_handler
            .version_exists(
                &self.state.base,
                next_version,
                self.state.object_store.inner.as_ref(),
                self.state.manifest_location.naming_scheme,
            )
            .await?;
        Ok(exists)
    }

    /// Look up the fragment with `id` in the manifest.
    ///
    /// `Manifest::fragments` is kept sorted by id, so this binary searches
    /// rather than scanning. Two kinds of manifest predate that invariant and
    /// are still readable: those written before fragments were forced into id
    /// order (Lance 0.10 and earlier), and those with duplicate fragment ids
    /// (Lance 0.16 and earlier). Neither is rejected on read, so the search
    /// result is checked and a scan takes over when it does not match --
    /// returning some other fragment's data would be silent corruption.
    pub fn find_fragment(&self, id: u64) -> Option<&'a Fragment> {
        if !u32::try_from(id).is_ok_and(|id| self.state.fragment_bitmap.contains(id)) {
            return None;
        }
        let fragments = self.manifest.fragments.as_slice();
        let index = fragments.partition_point(|fragment| fragment.id < id);
        match fragments.get(index) {
            Some(fragment) if fragment.id == id => Some(fragment),
            _ => fragments.iter().find(|fragment| fragment.id == id),
        }
    }

    /// Atomically reserve a branch name against an existing version.
    pub async fn reserve_branch(
        &self,
        branch: &str,
        version: impl Into<refs::Ref>,
    ) -> Result<BranchReservation> {
        refs::check_new_branch(branch)?;

        let (source_branch, version_number) = self.resolve_reference(version.into()).await?;
        let branch_location = self.branch_location().find_branch(Some(branch))?;
        let branch_file = refs::branch_contents_path(&self.refs.root()?.path, branch);
        if self.state.object_store.exists(&branch_file).await? {
            return Err(Error::RefConflict {
                message: format!("branch {branch} already exists"),
            });
        }
        let versions_path = branch_location.path.clone().join(VERSIONS_DIR);
        if has_any_entry(&self.state.object_store, &versions_path).await? {
            return Err(Error::RefConflict {
                message: format!(
                    "Cannot create branch {branch}: {versions_path} contains leftover version data. \
                     Use force_delete_branch to clean up the unused branch before retrying"
                ),
            });
        }

        // Claim the name before the clone can write into another creator's branch.
        let contents = self
            .branches()
            .create(branch, version_number, source_branch.as_deref())
            .await?;
        Ok(BranchReservation {
            branch: branch.to_string(),
            source_branch,
            source_version: version_number,
            identifier: contents.identifier,
        })
    }

    pub async fn resolve_reference(&self, reference: refs::Ref) -> Result<(Option<String>, u64)> {
        match reference {
            refs::Ref::Version(branch, version_number) => {
                if let Some(version_number) = version_number {
                    Ok((branch, version_number))
                } else {
                    let branch_location = self.branch_location().find_branch(branch.as_deref())?;
                    let version_number = self
                        .state
                        .commit_handler
                        .resolve_latest_location(&branch_location.path, &self.state.object_store)
                        .await?
                        .version;
                    Ok((branch, version_number))
                }
            }
            refs::Ref::VersionNumber(version_number) => {
                Ok((self.manifest.branch.clone(), version_number))
            }
            refs::Ref::Tag(tag_name) => {
                let tag_contents = self.tags().get(tag_name.as_str()).await?;
                Ok((tag_contents.branch, tag_contents.version))
            }
        }
    }

    /// Get the currently checked-out version id.
    ///
    /// This is a cheap accessor that reads the id directly from the loaded
    /// manifest without constructing the full [Version] summary.
    pub fn version_id(&self) -> u64 {
        self.manifest.version
    }

    /// Get the schema of the dataset
    pub fn schema(&self) -> &'a Schema {
        &self.manifest.schema
    }

    /// Get the full manifest of the dataset version.
    pub fn manifest(&self) -> &'a Manifest {
        self.manifest
    }

    pub fn manifest_location(&self) -> &'a ManifestLocation {
        &self.state.manifest_location
    }

    pub fn count_fragments(&self) -> usize {
        self.manifest.fragments.len()
    }

    pub fn fragments(&self) -> &'a Arc<Vec<Fragment>> {
        &self.manifest.fragments
    }

    /// Iterate over manifest fragments without allocating data readers.
    pub fn iter_fragments(&self) -> impl Iterator<Item = &'a Fragment> + use<'a> {
        self.manifest.fragments.iter()
    }

    /// Ids of the fragments in this version.
    pub fn fragment_bitmap(&self) -> &'a Arc<RoaringBitmap> {
        &self.state.fragment_bitmap
    }

    pub fn tags(&self) -> Tags<'a> {
        self.refs.tags()
    }

    pub fn branches(&self) -> Branches<'a> {
        self.refs.branches()
    }
}

impl TableState {
    /// Get all versions, in ascending order.
    ///
    /// Every manifest in the history is read. Use [`Self::versions_with_options`] to
    /// read only a page, or [`Self::count_versions`] to avoid reading manifests at all.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        self.versions_with_options(0, None, false).await
    }

    /// Get one page of versions in the current version history.
    ///
    /// Versions are ordered ascending by version, or descending when `descending`
    /// is true. `offset` is the number of versions to skip and `limit` is the
    /// maximum number to return, so `versions_with_options(0, None, false)` is the
    /// whole history. `None` means no limit, `Some(0)` returns an empty page without
    /// listing or reading any manifest, and an `offset` past the end of the history
    /// returns an empty page. Detached versions are never part of the history.
    ///
    /// Only the manifests in the returned page are read, concurrently within the
    /// store's effective IO window.
    ///
    /// With `descending` on a store whose listing is lexically ordered and whose
    /// manifests are V2 paths, listing stops after `offset + limit` manifest entries
    /// have been observed, so a page near the newest version does not enumerate the
    /// history. Every other case (ascending pages, V1 manifest paths, or a listing
    /// that is not lexically ordered) enumerates every manifest location before
    /// selecting the page. These are listing-entry bounds: a store may still
    /// materialize more keys, or issue more requests, than the entries it yields.
    pub async fn versions_with_options(
        &self,
        offset: usize,
        limit: Option<usize>,
        descending: bool,
    ) -> Result<Vec<Version>> {
        let limit = limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok(Vec::new());
        }

        let locations: Vec<ManifestLocation> = if descending {
            let mut stream =
                self.commit_handler
                    .list_manifest_locations(&self.base, &self.object_store, true);
            let mut skipped = 0;
            let mut page = Vec::new();
            while page.len() < limit {
                let Some(location) = stream.try_next().await? else {
                    break;
                };
                if skipped < offset {
                    skipped += 1;
                } else {
                    page.push(location);
                }
            }
            page
        } else {
            let mut all: Vec<ManifestLocation> = self
                .commit_handler
                .list_manifest_locations(&self.base, &self.object_store, false)
                .try_collect()
                .await?;
            all.sort_unstable_by_key(|location| location.version);
            all.into_iter().skip(offset).take(limit).collect()
        };

        stream::iter(locations)
            .map(|location| async move {
                let manifest_size = location.size;
                let manifest =
                    read_manifest(&self.object_store, &location.path, manifest_size).await?;
                let mut version = Version::from(&manifest);
                version.manifest_size = manifest_size;
                Ok::<_, Error>(version)
            })
            .buffered(self.object_store.io_parallelism())
            .try_collect()
            .await
    }

    /// Get the number of versions in the current version history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest.
    pub async fn count_versions(&self) -> Result<u64> {
        self.commit_handler
            .list_manifest_locations(&self.base, &self.object_store, false)
            .try_fold(0_u64, |count, _| async move { Ok(count + 1) })
            .await
    }

    /// List lightweight references to all attached versions in the current branch's history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest. The references are sorted by version in ascending order.
    /// Detached manifests are excluded; see [`Self::list_detached_manifests`].
    ///
    /// Use [`Self::latest_version_id`] instead when only the latest version is needed.
    pub async fn version_refs(&self) -> Result<Vec<VersionRef>> {
        let mut versions: Vec<_> = self
            .commit_handler
            .list_manifest_locations(&self.base, &self.object_store, false)
            .map_ok(|location| VersionRef {
                version: location.version,
            })
            .try_collect()
            .await?;
        versions.sort_unstable_by_key(|version| version.version);
        Ok(versions)
    }

    /// List all detached manifest locations.
    ///
    /// Detached manifests are versions that are not part of the main version history.
    /// They are created by `commit_detached` and can be used for staging changes.
    ///
    pub async fn list_detached_manifests(&self) -> Result<Vec<ManifestLocation>> {
        self.commit_handler
            .list_detached_manifest_locations(&self.base, &self.object_store)
            .try_collect()
            .await
    }

    /// Get the latest version of the dataset
    /// This is meant to be a fast path for checking if a dataset has changed. This is why
    /// we don't return the full version struct.
    pub async fn latest_version_id(&self) -> Result<u64> {
        Ok(self
            .commit_handler
            .resolve_latest_location(&self.base, &self.object_store)
            .await?
            .version)
    }
}

/// Dataset Version
#[derive(Deserialize, Serialize, Debug)]
pub struct Version {
    /// version number
    pub version: u64,

    /// Timestamp of dataset creation in UTC.
    pub timestamp: DateTime<Utc>,

    /// Key-value pairs of metadata.
    pub metadata: BTreeMap<String, String>,

    /// Size of the manifest file for this version, in bytes, if known.
    ///
    /// On wide tables (many columns and/or fragments) the manifest is rewritten
    /// in full on every commit; exposing its size per version makes metadata
    /// growth observable via [`Table::versions`].
    #[serde(default)]
    pub manifest_size: Option<u64>,
}

/// A lightweight reference to an attached dataset version, which could be used to uniquely identify a version.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct VersionRef {
    /// Version number within the current branch's history.
    pub version: u64,
}

/// Convert Manifest to Data Version.
impl From<&Manifest> for Version {
    fn from(m: &Manifest) -> Self {
        Self {
            version: m.version,
            timestamp: m.timestamp(),
            metadata: m.summary().into(),
            manifest_size: None,
        }
    }
}

/// The transaction that produced a version of the dataset, along with the
/// version's commit timestamp.
///
/// Returned by [`Table::read_version_transaction`], which reads this
/// information directly from storage without checking out the version.
#[derive(Debug, Clone)]
pub struct VersionTransaction {
    /// Version number.
    pub version: u64,

    /// Timestamp the version was committed, in UTC.
    pub timestamp: DateTime<Utc>,

    /// The transaction that produced this version, if one was recorded.
    pub transaction: Option<Transaction>,
}

impl Table {
    /// Open the latest snapshot at `uri` with the default reader options.
    ///
    /// Use [`TableBuilder`] to select a version or set storage options.
    ///
    /// ```no_run
    /// # async fn example() -> lance_core::Result<()> {
    /// let table = lance_table_runtime::Table::open("table.lance").await?;
    /// println!("{}", table.version_id());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn open(uri: &str) -> Result<Self> {
        TableBuilder::from_uri(uri).load().await
    }

    /// Assemble a snapshot using the matching parts returned by into_parts.
    #[doc(hidden)]
    pub fn from_parts(state: TableState, manifest: Arc<Manifest>, refs: Refs) -> Self {
        Self {
            state,
            manifest,
            refs,
        }
    }

    /// Transfer the runtime state, manifest and refs without copying their contents.
    #[doc(hidden)]
    pub fn into_parts(self) -> (TableState, Arc<Manifest>, Refs) {
        (self.state, self.manifest, self.refs)
    }

    pub(crate) fn view(&self) -> TableView<'_> {
        TableView::new(&self.state, &self.manifest, &self.refs)
    }

    /// Return the session used by this snapshot.
    pub fn session(&self) -> Arc<TableSession> {
        self.state.session.clone()
    }

    /// Open a file-reading view using the registered bases of another manifest.
    ///
    /// For a fully bound snapshot of another version, use checkout_version.
    #[doc(hidden)]
    pub fn with_manifest_view(&self, manifest: Arc<Manifest>) -> Self {
        self.view().with_manifest_view(manifest)
    }

    /// Resolve the parameters for an additional base or the primary store.
    pub fn store_params_for_base(&self, base: Option<&BasePath>) -> ObjectStoreParams {
        self.state.store_params_for_base(base)
    }

    /// Get the registered bases in the current snapshot, keyed by manifest id.
    pub fn base_paths(&self) -> &HashMap<u32, BasePath> {
        &self.manifest.base_paths
    }

    /// Open a snapshot around a manifest that has already been loaded.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub async fn checkout_manifest(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        uri: String,
        manifest: Arc<Manifest>,
        manifest_location: ManifestLocation,
        session: Arc<TableSession>,
        commit_handler: Arc<dyn CommitHandler>,
        file_reader_options: Option<FileReaderOptions>,
        store_params: Option<ObjectStoreParams>,
        base_store_params: Option<Arc<HashMap<String, ObjectStoreParams>>>,
    ) -> Result<Self> {
        let refs = Refs::new(
            object_store.clone(),
            commit_handler.clone(),
            BranchLocation {
                path: base_path.clone(),
                uri: uri.clone(),
                branch: manifest.branch.clone(),
            },
        );
        let branch_identifier = Arc::new(
            refs.branches()
                .get_identifier(manifest.branch.as_deref())
                .await?,
        );
        log_unregistered_base_scoped_options(
            store_params.as_ref(),
            &manifest.base_paths,
            log::Level::Debug,
        );
        let metadata_cache = Arc::new(session.metadata_cache().for_dataset(&uri));
        let state = TableState::new(
            object_store,
            base_path,
            uri,
            &manifest,
            manifest_location,
            session,
            metadata_cache,
            commit_handler,
            branch_identifier,
            file_reader_options,
            store_params,
            base_store_params,
        );
        Ok(Self::from_parts(state, manifest, refs))
    }

    /// Check out the latest version of the dataset
    pub async fn checkout_latest(&mut self) -> Result<()> {
        let (manifest, manifest_location) = self.latest_manifest().await?;
        self.state
            .set_manifest(&mut self.manifest, manifest, manifest_location);
        Ok(())
    }

    /// Check out a dataset version with a ref.
    ///
    /// In reference contexts, `"main"` is an alias for the default branch and
    /// is equivalent to `None`.
    pub async fn checkout_version(&self, version: impl Into<refs::Ref>) -> Result<Self> {
        Ok(self
            .view()
            .checkout_version(version)
            .await?
            .unwrap_or_else(|| self.clone()))
    }

    /// Check out the latest version of the branch.
    ///
    /// Use `"main"` to check out the latest version of the default branch.
    pub async fn checkout_branch(&self, branch: &str) -> Result<Self> {
        self.checkout_version((branch, None)).await
    }

    pub async fn latest_manifest(&self) -> Result<(Arc<Manifest>, ManifestLocation)> {
        self.view().latest_manifest().await
    }

    pub async fn branch_identifier(&self) -> Result<BranchIdentifier> {
        self.view().branch_identifier().await
    }

    /// Return whether the dataset has a newer committed version.
    pub async fn is_stale(&self) -> Result<bool> {
        self.view().is_stale().await
    }

    /// Return whether the immediate attached successor manifest exists.
    ///
    /// This is a fast contiguous-history probe. It does not resolve the latest
    /// version and may return `false` if intermediate manifests have been
    /// removed. Callers that need a general freshness check should use
    /// [`Self::is_stale`].
    #[doc(hidden)]
    pub async fn has_successor_version(&self) -> Result<bool> {
        self.view().has_successor_version().await
    }

    /// Get the currently checked-out version details.
    ///
    /// This constructs a full [Version], including summary metadata derived
    /// from the loaded manifest fragments.
    pub fn version(&self) -> Version {
        self.view().version()
    }

    pub fn branch_location(&self) -> BranchLocation {
        self.view().branch_location()
    }

    pub fn tags(&self) -> Tags<'_> {
        self.view().tags()
    }

    pub fn branches(&self) -> Branches<'_> {
        self.view().branches()
    }

    /// Get the currently checked-out version id.
    ///
    /// This is a cheap accessor that reads the id directly from the loaded
    /// manifest without constructing the full [Version] summary.
    pub fn version_id(&self) -> u64 {
        self.view().version_id()
    }

    /// Get the schema of the dataset
    pub fn schema(&self) -> &Schema {
        self.view().schema()
    }

    /// Get the full manifest of the dataset version.
    pub fn manifest(&self) -> &Manifest {
        self.view().manifest()
    }

    pub fn manifest_location(&self) -> &ManifestLocation {
        self.view().manifest_location()
    }

    pub fn count_fragments(&self) -> usize {
        self.view().count_fragments()
    }

    pub fn fragments(&self) -> &Arc<Vec<Fragment>> {
        self.view().fragments()
    }

    /// Iterate over manifest fragments without allocating data readers.
    pub fn iter_fragments(&self) -> impl Iterator<Item = &Fragment> {
        self.view().iter_fragments()
    }

    /// Get readable fragments in manifest order. They share this snapshot.
    ///
    /// ```
    /// # use lance_table_runtime::{FragReadConfig, Table};
    /// # async fn example(table: &Table) -> lance_core::Result<()> {
    /// for fragment in table.get_fragments() {
    ///     let _reader = fragment.open(table.schema(), FragReadConfig::default()).await?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_fragments(&self) -> Vec<FileFragment> {
        let table = Arc::new(self.clone());
        self.iter_fragments()
            .map(|fragment| FileFragment::new(table.clone(), fragment.clone()))
            .collect()
    }

    /// Get a readable fragment by its persisted id.
    ///
    /// Returns `None` when the id is absent from this snapshot.
    pub fn get_fragment(&self, id: usize) -> Option<FileFragment> {
        let fragment = self.view().find_fragment(id as u64)?.clone();
        Some(FileFragment::new(Arc::new(self.clone()), fragment))
    }

    /// Count live rows in this snapshot.
    ///
    /// Persisted counts are used when available and trustworthy. Older
    /// fragments can require I/O to read file metadata and deletions.
    pub async fn count_rows(&self) -> Result<usize> {
        let trust_metadata = self.manifest.writer_version.is_some();
        let table = Arc::new(self.clone());
        let mut num_rows = 0;
        let mut need_io = Vec::new();
        for fragment in self.iter_fragments() {
            if trust_metadata
                && !fragment.files.is_empty()
                && let Some(rows) = fragment.num_rows()
            {
                num_rows += rows;
            } else {
                need_io.push(FileFragment::new(table.clone(), fragment.clone()));
            }
        }
        stream::iter(need_io)
            .map(|fragment| async move { fragment.live_rows().await })
            .buffer_unordered(16)
            .try_fold(num_rows, |total, rows| async move { Ok(total + rows) })
            .await
    }

    /// Ids of the fragments in this version.
    pub fn fragment_bitmap(&self) -> &Arc<RoaringBitmap> {
        self.view().fragment_bitmap()
    }

    /// Atomically reserve a branch name against an existing version.
    pub async fn reserve_branch(
        &self,
        branch: &str,
        version: impl Into<refs::Ref>,
    ) -> Result<BranchReservation> {
        self.view().reserve_branch(branch, version).await
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    pub async fn read_transaction(&self) -> Result<Option<Transaction>> {
        self.view().read_transaction().await
    }

    /// Read the transaction (if any) and commit timestamp of a version of the
    /// dataset. `version` is a version number on this dataset's current branch.
    ///
    /// When `version` is the version this handle is checked out at, the
    /// in-memory manifest is used and the transaction is read directly from
    /// storage, so no manifest download is needed. Otherwise the version's
    /// manifest is read transiently: no historical `Dataset` is constructed,
    /// no `IndexSection` is decoded, and no session cache is written, so
    /// scanning many historical versions does not fill the shared caches. A
    /// manifest that the session cache already holds (for example after
    /// `checkout_version`) is reused instead of being downloaded again.
    ///
    /// Neither path consults or populates the session's transaction cache
    /// (unlike [`Self::read_transaction`]): that cache is keyed only by
    /// version number, which is not safe to reuse across a dataset dropped
    /// and recreated at the same URI within the same session (see #8916).
    ///
    /// Returns an error if the version does not exist (for example, if it has
    /// been cleaned up).
    ///
    /// # Example
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table_runtime::Table;
    /// # async fn example(dataset: &Table) -> Result<()> {
    /// let record = dataset.read_version_transaction(5).await?;
    /// let committed_at = record.timestamp;
    /// let operation = record.transaction.as_ref().map(|t| t.operation.name());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_version_transaction(&self, version: u64) -> Result<VersionTransaction> {
        self.view().read_version_transaction(version).await
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    ///
    /// Does not populate the session caches; see
    /// [`Self::read_version_transaction`].
    ///
    /// # Example
    ///
    /// ```
    /// # use lance_core::Result;
    /// # use lance_table_runtime::Table;
    /// # async fn example(dataset: &Table) -> Result<()> {
    /// let transaction = dataset.read_transaction_by_version(5).await?;
    /// let operation = transaction.as_ref().map(|t| t.operation.name());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_transaction_by_version(&self, version: u64) -> Result<Option<Transaction>> {
        self.view().read_transaction_by_version(version).await
    }

    /// Read the transactions for up to the given number of versions, starting
    /// at the checked-out version and following its branch history backwards.
    pub async fn get_transactions(
        &self,
        recent_transactions: usize,
    ) -> Result<Vec<Option<Transaction>>> {
        self.view().get_transactions(recent_transactions).await
    }

    /// Get all versions, in ascending order.
    ///
    /// Every manifest in the history is read. Use [`Self::versions_with_options`] to
    /// read only a page, or [`Self::count_versions`] to avoid reading manifests at all.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        self.state.versions().await
    }

    /// Get one page of versions in the current version history.
    ///
    /// Versions are ordered ascending by version, or descending when `descending`
    /// is true. `offset` is the number of versions to skip and `limit` is the
    /// maximum number to return, so `versions_with_options(0, None, false)` is the
    /// whole history. `None` means no limit, `Some(0)` returns an empty page without
    /// listing or reading any manifest, and an `offset` past the end of the history
    /// returns an empty page. Detached versions are never part of the history.
    ///
    /// Only the manifests in the returned page are read, concurrently within the
    /// store's effective IO window.
    ///
    /// With `descending` on a store whose listing is lexically ordered and whose
    /// manifests are V2 paths, listing stops after `offset + limit` manifest entries
    /// have been observed, so a page near the newest version does not enumerate the
    /// history. Every other case (ascending pages, V1 manifest paths, or a listing
    /// that is not lexically ordered) enumerates every manifest location before
    /// selecting the page. These are listing-entry bounds: a store may still
    /// materialize more keys, or issue more requests, than the entries it yields.
    pub async fn versions_with_options(
        &self,
        offset: usize,
        limit: Option<usize>,
        descending: bool,
    ) -> Result<Vec<Version>> {
        self.state
            .versions_with_options(offset, limit, descending)
            .await
    }

    /// Get the number of versions in the current version history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest.
    pub async fn count_versions(&self) -> Result<u64> {
        self.state.count_versions().await
    }

    /// List lightweight references to all attached versions in the current branch's history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest. The references are sorted by version in ascending order.
    /// Detached manifests are excluded; see [`Self::list_detached_manifests`].
    ///
    /// Use [`Self::latest_version_id`] instead when only the latest version is needed.
    pub async fn version_refs(&self) -> Result<Vec<VersionRef>> {
        self.state.version_refs().await
    }

    /// List all detached manifest locations.
    ///
    /// Detached manifests are versions that are not part of the main version history.
    /// They are created by `commit_detached` and can be used for staging changes.
    ///
    pub async fn list_detached_manifests(&self) -> Result<Vec<ManifestLocation>> {
        self.state.list_detached_manifests().await
    }

    /// Get the latest version of the dataset
    /// This is meant to be a fast path for checking if a dataset has changed. This is why
    /// we don't return the full version struct.
    pub async fn latest_version_id(&self) -> Result<u64> {
        self.state.latest_version_id().await
    }

    /// Resolve the object store for the primary dataset or an additional base.
    ///
    /// Pass `None` to get the primary dataset object store. Pass `Some(base_id)`
    /// when resolving a file whose metadata references an additional base.
    pub async fn object_store(&self, base_id: Option<u32>) -> Result<Arc<ObjectStore>> {
        self.view().object_store(base_id).await
    }

    /// Clone this dataset with extra object store wrappers applied to all read stores.
    ///
    /// The returned dataset uses the wrappers for the already-open primary object
    /// store and appends the same wrappers to the dataset-level and base-specific
    /// object store params used when additional base stores are opened later.
    pub fn with_object_store_wrappers(
        &self,
        wrappers: impl IntoIterator<Item = Arc<dyn WrappingObjectStore>>,
    ) -> Self {
        self.view().with_object_store_wrappers(wrappers)
    }

    /// Returns the initial storage options without triggering any refresh.
    ///
    /// For the latest refreshed options, use [`Self::latest_storage_options`].
    pub fn initial_storage_options(&self) -> Option<&HashMap<String, String>> {
        self.state.initial_storage_options()
    }

    /// Returns the storage options provider used when opening this dataset, if any.
    pub fn storage_options_provider(
        &self,
    ) -> Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>> {
        self.state.storage_options_provider()
    }

    /// Returns the unified storage options accessor for this dataset, if any.
    ///
    /// The accessor handles both static and dynamic storage options with automatic
    /// caching and refresh. Use [`StorageOptionsAccessor::get_storage_options`] to
    /// get the latest options.
    pub fn storage_options_accessor(&self) -> Option<Arc<StorageOptionsAccessor>> {
        self.state.storage_options_accessor()
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
        self.state.latest_storage_options().await
    }

    pub fn data_dir(&self) -> Path {
        self.state.data_dir()
    }

    pub(crate) fn data_file_dir(&self, data_file: &DataFile) -> Result<Path> {
        self.view().data_file_dir_for_base(data_file.base_id)
    }

    pub fn indices_dir(&self) -> Path {
        self.state.indices_dir()
    }

    pub fn transactions_dir(&self) -> Path {
        self.state.transactions_dir()
    }

    pub fn deletions_dir(&self) -> Path {
        self.state.deletions_dir()
    }

    pub fn versions_dir(&self) -> Path {
        self.state.versions_dir()
    }

    /// Get the fully qualified URI for this snapshot.
    pub fn uri(&self) -> &str {
        &self.state.uri
    }

    /// The runtime parameters used to open the primary store.
    pub fn store_params(&self) -> Option<&ObjectStoreParams> {
        self.state.store_params.as_deref()
    }

    /// File reader defaults carried by this snapshot.
    pub fn file_reader_options(&self) -> Option<&FileReaderOptions> {
        self.state.file_reader_options.as_deref()
    }

    /// The commit handler used to resolve this table's manifests.
    pub fn commit_handler(&self) -> Arc<dyn CommitHandler> {
        self.state.commit_handler.clone()
    }
}
