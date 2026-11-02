// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shard manifest storage with bit-reversed versioned naming.
//!
//! Shard manifests are stored as versioned protobuf files using bit-reversed
//! naming scheme to distribute files across object store keyspace.
//!
//! ## File Layout
//!
//! ```text
//! _mem_wal/{shard_id}/manifest/
//!   ├── {bit_reversed_version}.binpb  # Versioned manifest files
//!   └── version_hint.json             # Best-effort version hint
//! ```
//!
//! ## Write Protocol
//!
//! 1. Compute next version number
//! 2. Write manifest to `{bit_reversed_version}.binpb` using PUT-IF-NOT-EXISTS
//! 3. Best-effort update `version_hint.json` (failure is acceptable)
//!
//! ## Read Protocol
//!
//! 1. Start from the version `version_hint.json` names, or, when the hint is
//!    missing or unreadable or that version is gone, from the newest listed
//!    version
//! 2. Use HEAD requests to check existence of subsequent versions
//! 3. Continue until a version is not found
//! 4. Return the last found version
//!
//! ## Pruning
//!
//! A writer builds each version on its cached position, and the existing next
//! version is what rejects a writer that lost its shard. A writer re-reads any
//! position older than one hour, and
//! [`ShardManifestStore::prune_versions_before`] never deletes a version younger
//! than one day, so the version a stale writer could target still exists and
//! its write collides. Pruning deletes oldest first, so the versions left are
//! always contiguous and a lookup from any of them reaches the latest.

use object_store::ObjectStoreExt;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use lance_core::{Error, Result};
use lance_index::mem_wal::{ShardManifest, ShardStatus};
use lance_io::object_store::ObjectStore;
use lance_table::format::pb;
use log::{info, warn};
use object_store::path::Path;
use prost::Message;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use super::util::{manifest_filename, parse_bit_reversed_filename, shard_manifest_path};

/// How long a writer builds on its position without re-reading storage.
const MAX_POSITION_AGE: Duration = Duration::from_secs(60 * 60);

/// The youngest version pruning deletes. It exceeds [`MAX_POSITION_AGE`] by a
/// margin for a stalled write and for skew between this clock and the store's.
const MIN_PRUNE_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Version hint file structure.
#[derive(Debug, Serialize, Deserialize)]
struct VersionHint {
    version: u64,
}

/// Store for reading and writing shard manifests.
///
/// Handles versioned manifest files with bit-reversed naming scheme
/// and PUT-IF-NOT-EXISTS atomicity.
#[derive(Debug)]
pub struct ShardManifestStore {
    object_store: Arc<ObjectStore>,
    shard_id: Uuid,
    manifest_dir: Path,
    manifest_scan_batch_size: usize,
    max_position_age: Duration,
    min_prune_age: Duration,
    /// This store's position: the version it may build its next write on, and
    /// what [`Self::latest`] serves.
    ///
    /// Set by a landed write and by [`Self::refresh_latest`] — the epoch holder
    /// is the sole permitted writer, so what it wrote or last refreshed to
    /// stays latest until a PUT-IF-NOT-EXISTS collision proves otherwise. A
    /// plain [`Self::latest`] scan never sets it, so a reader that polls keeps
    /// observing the writer instead of pinning the first manifest it saw.
    latest: RwLock<Option<Position>>,
}

/// A manifest this store knows was the latest at `confirmed_at`.
#[derive(Debug)]
struct Position {
    manifest: ShardManifest,
    /// Taken before the read or write that confirmed it, so the age is never
    /// understated.
    confirmed_at: Instant,
}

impl ShardManifestStore {
    /// Create a new manifest store for the given shard.
    ///
    /// # Arguments
    ///
    /// * `object_store` - Object store for reading/writing manifests
    /// * `base_path` - Base path within the object store (from ObjectStore::from_uri)
    /// * `dir_id` - The MemWAL's `dir_id` (empty for the legacy `_mem_wal`)
    /// * `shard_id` - Shard UUID
    /// * `manifest_scan_batch_size` - Batch size for parallel HEAD requests when scanning versions
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_path: &Path,
        dir_id: &str,
        shard_id: Uuid,
        manifest_scan_batch_size: usize,
    ) -> Self {
        let manifest_dir = shard_manifest_path(base_path, dir_id, &shard_id);
        Self {
            object_store,
            shard_id,
            manifest_dir,
            manifest_scan_batch_size,
            max_position_age: MAX_POSITION_AGE,
            min_prune_age: MIN_PRUNE_AGE,
            latest: RwLock::new(None),
        }
    }

    #[cfg(test)]
    fn with_ages(mut self, max_position_age: Duration, min_prune_age: Duration) -> Self {
        self.max_position_age = max_position_age;
        self.min_prune_age = min_prune_age;
        self
    }

    /// The cached manifest, if this store has written one.
    fn cached(&self) -> Option<ShardManifest> {
        self.latest
            .read()
            .expect("manifest cache lock")
            .as_ref()
            .map(|position| position.manifest.clone())
    }

    /// The cached manifest, if it was confirmed within the max position age.
    fn fresh_cached(&self) -> Option<ShardManifest> {
        self.latest
            .read()
            .expect("manifest cache lock")
            .as_ref()
            .filter(|position| position.confirmed_at.elapsed() <= self.max_position_age)
            .map(|position| position.manifest.clone())
    }

    /// Publish `manifest`, confirmed latest at `confirmed_at`, as the latest.
    /// Only ever called after a durable write or a read of storage.
    ///
    /// Never regresses: the flush task and the tailer's cursor updates share one
    /// handle, so two writes can win their CAS in one order and return to their
    /// callers in the other. Re-confirming the same version renews it.
    fn cache(&self, manifest: &ShardManifest, confirmed_at: Instant) {
        let mut latest = self.latest.write().expect("manifest cache lock");
        let is_newer = latest.as_ref().is_none_or(|c| {
            manifest.version > c.manifest.version
                || (manifest.version == c.manifest.version && confirmed_at > c.confirmed_at)
        });
        if is_newer {
            *latest = Some(Position {
                manifest: manifest.clone(),
                confirmed_at,
            });
        }
    }

    /// Drop the cache on a write collision — the one signal that another
    /// writer may have moved the shard past us.
    fn invalidate(&self) {
        *self.latest.write().expect("manifest cache lock") = None;
    }

    /// The latest manifest as far as this store knows: its own position when it
    /// has one, otherwise a scan of storage.
    ///
    /// Cheap, and deliberately not authoritative — it can sit behind a peer's
    /// commit, and a scan here does *not* become this store's position, so a
    /// reader that polls keeps observing the writer. To observe a peer, or to
    /// take a position to write from, use [`Self::refresh_latest`].
    ///
    /// Returns `None` if no manifest exists (new shard).
    pub async fn latest(&self) -> Result<Option<ShardManifest>> {
        match self.cached() {
            Some(cached) => Ok(Some(cached)),
            None => self.scan_latest().await,
        }
    }

    /// Read the latest manifest from storage and adopt it as this store's
    /// position.
    ///
    /// The adopting half matters: a claim reads uncached precisely because it
    /// must see another process, and what it finds is the version its own write
    /// then builds on. Callers that only want to *look* want [`Self::latest`],
    /// which leaves this store's position alone.
    ///
    /// Returns `None` if no manifest exists (new shard).
    #[instrument(name = "manifest_refresh_latest", level = "debug", skip_all, fields(shard_id = %self.shard_id))]
    pub async fn refresh_latest(&self) -> Result<Option<ShardManifest>> {
        let started = Instant::now();
        let latest = self.scan_latest().await?;
        if let Some(manifest) = &latest {
            self.cache(manifest, started);
        }
        Ok(latest)
    }

    /// Scan storage for the latest manifest, touching no local state.
    async fn scan_latest(&self) -> Result<Option<ShardManifest>> {
        let version = self.find_latest_version().await?;
        if version == 0 {
            return Ok(None);
        }

        self.read_version(version).await.map(Some)
    }

    /// Read a specific manifest version.
    pub async fn read_version(&self, version: u64) -> Result<ShardManifest> {
        let filename = manifest_filename(version);
        let path = self.manifest_dir.clone().join(filename.as_str());

        let data = self.object_store.inner.get(&path).await.map_err(|e| {
            Error::io(format!(
                "Failed to read manifest version {} for shard {}: {}",
                version, self.shard_id, e
            ))
        })?;

        let bytes = data
            .bytes()
            .await
            .map_err(|e| Error::io(format!("Failed to read manifest bytes: {}", e)))?;

        let pb_manifest = pb::ShardManifest::decode(bytes)
            .map_err(|e| Error::io(format!("Failed to decode manifest protobuf: {}", e)))?;

        ShardManifest::try_from(pb_manifest)
    }

    /// Write an initial manifest for a newly-created shard.
    ///
    /// `shard_field_values` maps field_id to raw Arrow scalar bytes.
    /// Initial manifests use writer epoch 0. A writer that claims the shard
    /// will write a new manifest with epoch 1 before appending WAL entries.
    ///
    /// A shard that already has a manifest is not written to: once version 1
    /// is pruned, writing it again would succeed and hide the shard's history.
    pub async fn initialize_shard(
        &self,
        shard_spec_id: u32,
        shard_field_values: HashMap<String, Vec<u8>>,
    ) -> Result<ShardManifest> {
        let manifest = ShardManifest {
            shard_id: self.shard_id,
            version: 1,
            shard_spec_id,
            shard_field_values,
            writer_epoch: 0,
            replay_after_wal_entry_position: 0,
            wal_entry_position_last_seen: 0,
            current_generation: 1,
            sstables: vec![],
            sealed_by: None,
            status: ShardStatus::Active,
        };

        let existing = match self.refresh_latest().await? {
            Some(existing) => existing,
            None => match self.write(&manifest).await {
                Ok(_) => return Ok(manifest),
                Err(error) => match self.refresh_latest().await? {
                    Some(existing) => existing,
                    None => return Err(error),
                },
            },
        };
        if existing.shard_spec_id == manifest.shard_spec_id
            && existing.shard_field_values == manifest.shard_field_values
        {
            Ok(existing)
        } else {
            Err(Error::invalid_input(format!(
                "shard {} is already initialized with spec {} and different field values than spec {}",
                self.shard_id, existing.shard_spec_id, manifest.shard_spec_id
            )))
        }
    }

    /// Write a new manifest version atomically.
    ///
    /// Uses storage-appropriate strategy:
    /// - Local: Write to temp file + atomic rename for fencing
    /// - Cloud: PUT-IF-NOT-EXISTS (S3 conditional write)
    ///
    /// Returns the version that was written.
    ///
    /// Callers derive `manifest.version` from a manifest they just read, which
    /// is what keeps the sequence gap-free — the cache treats a landed write as
    /// proof of the tip, and `find_latest_version` stops at the first absent
    /// batch, so a gap hides every version past it. Whoever holds that
    /// predecessor checks the successor; see [`Self::commit_update`].
    ///
    /// A version at or below this store's position is reported as the collision
    /// it is, so callers retry.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RetryableCommitConflict`] if another writer already
    /// holds this version.
    #[instrument(name = "manifest_write", level = "debug", skip_all, fields(shard_id = %self.shard_id, version = manifest.version, epoch = manifest.writer_epoch))]
    pub(crate) async fn write(&self, manifest: &ShardManifest) -> Result<u64> {
        let version = manifest.version;
        if self.cached().is_some_and(|c| version <= c.version) {
            // Someone already took it — our own position proves it exists.
            // Report the collision so callers retry rather than fail.
            self.invalidate();
            return Err(self.version_taken(version));
        }
        let filename = manifest_filename(version);
        let path = self.manifest_dir.clone().join(filename.as_str());

        let pb_manifest = pb::ShardManifest::from(manifest);
        let bytes = pb_manifest.encode_to_vec();

        let started = Instant::now();
        self.object_store
            .put_if_absent(&path, Bytes::from(bytes).into())
            .await
            .inspect_err(|_| self.invalidate())
            .map_err(|error| {
                if matches!(
                    error,
                    object_store::Error::AlreadyExists { .. }
                        | object_store::Error::Precondition { .. }
                ) {
                    self.version_taken(version)
                } else {
                    Error::io(format!(
                        "Failed to write manifest version {} for shard {}: {}",
                        version, self.shard_id, error
                    ))
                }
            })?;

        // The write landed, so this is now the latest.
        self.cache(manifest, started);

        // Best-effort update version hint (failures are logged as warnings)
        self.write_version_hint(version).await;

        Ok(version)
    }

    /// The error for a version another writer already holds. `commit_update`
    /// matches on the variant to decide whether to retry.
    fn version_taken(&self, version: u64) -> Error {
        Error::retryable_commit_conflict_source(
            version,
            format!(
                "Manifest version {} already exists for shard {}",
                version, self.shard_id
            )
            .into(),
        )
    }

    /// Find the latest manifest version.
    ///
    /// Starts from the hinted version, or from the newest listed one when the
    /// hint is missing or unreadable or its version was pruned, then scans
    /// forward with HEAD requests until a version is not found.
    async fn find_latest_version(&self) -> Result<u64> {
        let mut latest_found = match self.read_version_hint().await {
            Some(hint) if hint > 0 && self.version_exists(hint).await? => hint,
            _ => self.list_versions().await?.last().copied().unwrap_or(0),
        };

        // Parallel scan forward with batches of HEAD requests
        let batch_size = self.manifest_scan_batch_size;
        loop {
            let mut futures = FuturesUnordered::new();
            for offset in 0..batch_size {
                let version = latest_found + 1 + offset as u64;
                futures.push(async move { (version, self.version_exists(version).await) });
            }

            let mut found_any = false;
            while let Some((version, result)) = futures.next().await {
                if result? && version > latest_found {
                    latest_found = version;
                    found_any = true;
                }
            }

            if !found_any {
                break;
            }
        }

        Ok(latest_found)
    }

    /// Check if a manifest version exists using HEAD request.
    async fn version_exists(&self, version: u64) -> Result<bool> {
        let filename = manifest_filename(version);
        let path = self.manifest_dir.clone().join(filename.as_str());

        match self.object_store.inner.head(&path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(Error::io(format!(
                "HEAD request failed for version {}: {}",
                version, e
            ))),
        }
    }

    /// Read the version hint file: `None` when it is absent, unparsable or
    /// unreadable. The hint is best-effort; without it the lookup lists the
    /// versions instead.
    async fn read_version_hint(&self) -> Option<u64> {
        let path = self.manifest_dir.clone().join("version_hint.json");
        let read = async { self.object_store.inner.get(&path).await?.bytes().await };
        let bytes = match read.await {
            Ok(bytes) => bytes,
            Err(object_store::Error::NotFound { .. }) => return None,
            Err(e) => {
                warn!(
                    "Failed to read version hint for shard {}: {}",
                    self.shard_id, e
                );
                return None;
            }
        };
        serde_json::from_slice::<VersionHint>(&bytes)
            .ok()
            .map(|hint| hint.version)
    }

    /// Write the version hint file (best-effort, failures logged but ignored).
    async fn write_version_hint(&self, version: u64) {
        let path = self.manifest_dir.clone().join("version_hint.json");
        let hint = VersionHint { version };

        match serde_json::to_vec(&hint) {
            Ok(bytes) => {
                if let Err(e) = self
                    .object_store
                    .inner
                    .put(&path, Bytes::from(bytes).into())
                    .await
                {
                    warn!(
                        "Failed to write version hint for shard {}: {}",
                        self.shard_id, e
                    );
                }
            }
            Err(e) => {
                warn!("Failed to serialize version hint: {}", e);
            }
        }
    }

    /// List all manifest versions, in ascending order.
    pub async fn list_versions(&self) -> Result<Vec<u64>> {
        Ok(self
            .list_version_times()
            .await?
            .into_iter()
            .map(|(version, _)| version)
            .collect())
    }

    /// List all manifest versions with when each was written, in ascending
    /// version order.
    async fn list_version_times(&self) -> Result<Vec<(u64, DateTime<Utc>)>> {
        let mut versions = Vec::new();
        let mut listing = self.object_store.inner.list(Some(&self.manifest_dir));
        while let Some(item) = listing.next().await {
            let meta = item.map_err(|e| {
                Error::io(format!(
                    "Failed to list manifest versions for shard {}: {}",
                    self.shard_id, e
                ))
            })?;
            if let Some(filename) = meta.location.filename()
                && filename.ends_with(".binpb")
                && let Some(version) = parse_bit_reversed_filename(filename)
            {
                versions.push((version, meta.last_modified));
            }
        }
        versions.sort_unstable();
        Ok(versions)
    }

    /// Delete the manifest versions below `keep_from` that are at least one day
    /// old and return how many were deleted. `keep_from` may not be past the
    /// latest version, so the latest always survives.
    ///
    /// Versions are deleted oldest first, one at a time, stopping at the first
    /// failure or the first version younger than a day, so the versions left
    /// stay contiguous even mid-prune. See the module docs for why the age
    /// keeps a stale writer fenced.
    pub async fn prune_versions_before(&self, keep_from: u64) -> Result<usize> {
        let latest = self.find_latest_version().await?;
        if keep_from > latest {
            return Err(Error::invalid_input(format!(
                "cannot prune manifest versions before {keep_from} of shard {}: the latest is {latest}",
                self.shard_id
            )));
        }
        let min_age = chrono::Duration::from_std(self.min_prune_age)
            .expect("the minimum prune age fits in a chrono duration");
        let written_by = Utc::now() - min_age;
        let pruned: Vec<Path> = self
            .list_version_times()
            .await?
            .into_iter()
            .take_while(|(version, written)| *version < keep_from && *written <= written_by)
            .map(|(version, _)| {
                self.manifest_dir
                    .clone()
                    .join(manifest_filename(version).as_str())
            })
            .collect();
        let count = pruned.len();
        for path in pruned {
            match self.object_store.inner.delete(&path).await {
                Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
                Err(e) => {
                    return Err(Error::io(format!(
                        "Failed to prune manifest version {path} for shard {}: {e}",
                        self.shard_id
                    )));
                }
            }
        }
        Ok(count)
    }

    /// Get the shard ID.
    pub fn shard_id(&self) -> Uuid {
        self.shard_id
    }

    // ========================================================================
    // Epoch-based Writer Fencing
    // ========================================================================

    /// Claim a shard by incrementing its writer epoch.
    ///
    /// This establishes single-writer semantics by:
    /// 1. Loading the current manifest (or creating initial state)
    /// 2. Incrementing the writer epoch
    /// 3. Atomically writing the new manifest
    ///
    /// On version conflict, re-reads the manifest and only retries when
    /// the latest writer_epoch is strictly less than the epoch we were
    /// targeting — meaning the version was bumped by something other than
    /// a real claim (a tailer cursor update or a concurrent
    /// `initialize_shard` writing epoch 0). If the latest writer_epoch
    /// is equal to or greater than our target, the target epoch is
    /// already claimed and this call fails. This preserves the
    /// no-epoch-war guarantee for real claimants while tolerating benign
    /// version bumps.
    ///
    /// # Returns
    ///
    /// A tuple of `(epoch, ShardManifest)` where the manifest is the
    /// claimed state (may be freshly created or loaded and epoch-bumped).
    ///
    /// # Errors
    ///
    /// Returns an error if another writer claimed an equal-or-higher
    /// epoch than our target, or if the manifest stays contended past
    /// the retry budget.
    #[instrument(name = "manifest_claim_epoch", level = "info", skip_all, fields(shard_id = %self.shard_id, shard_spec_id))]
    pub async fn claim_epoch(&self, shard_spec_id: u32) -> Result<(u64, ShardManifest)> {
        const MAX_CLAIM_RETRIES: usize = 16;
        let mut last_write_err: Option<Error> = None;
        for _ in 0..MAX_CLAIM_RETRIES {
            // Refreshing, not reading: a claim exists to discover another
            // writer's epoch, and the tip it finds is what our write builds on.
            let current = self.refresh_latest().await?;

            // A sealed shard is mid-drop (drop-table 2PC). Refuse the claim
            // with a distinguishable error rather than minting a new epoch,
            // so a caller that skips its own status check still cannot
            // resurrect a shard being dropped. Sophon's reconcile keys on
            // the "sealed" marker in this message to tell it apart from an
            // ordinary epoch fence.
            if let Some(m) = &current
                && m.status == ShardStatus::Sealed
            {
                return Err(Error::invalid_input(format!(
                    "shard {} is sealed; refusing claim (drop in flight)",
                    self.shard_id
                )));
            }

            let (next_version, next_epoch, base_manifest) = match current {
                Some(m) => (m.next_version(), m.writer_epoch + 1, Some(m)),
                None => (1, 1, None),
            };

            let new_manifest = if let Some(base) = base_manifest {
                ShardManifest {
                    version: next_version,
                    writer_epoch: next_epoch,
                    ..base
                }
            } else {
                ShardManifest {
                    shard_id: self.shard_id,
                    version: next_version,
                    shard_spec_id,
                    shard_field_values: HashMap::new(),
                    writer_epoch: next_epoch,
                    replay_after_wal_entry_position: 0,
                    wal_entry_position_last_seen: 0,
                    current_generation: 1,
                    sstables: vec![],
                    sealed_by: None,
                    status: ShardStatus::Active,
                }
            };

            match self.write(&new_manifest).await {
                Ok(_) => {
                    info!(
                        "Claimed shard {} with epoch {} (version {})",
                        self.shard_id, next_epoch, next_version
                    );
                    return Ok((next_epoch, new_manifest));
                }
                Err(write_err) => {
                    let latest_epoch = self
                        .refresh_latest()
                        .await?
                        .map(|m| m.writer_epoch)
                        .unwrap_or(0);
                    if latest_epoch >= next_epoch {
                        return Err(Error::io(format!(
                            "Failed to claim shard {} (version {}): another writer claimed epoch {} (>= our target {}): {}",
                            self.shard_id, next_version, latest_epoch, next_epoch, write_err
                        )));
                    }
                    last_write_err = Some(write_err);
                }
            }
        }

        Err(Error::io(format!(
            "Failed to claim shard {} after {} retries due to manifest contention: {}",
            self.shard_id,
            MAX_CLAIM_RETRIES,
            last_write_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        )))
    }

    /// Check if the given epoch has been fenced by a newer writer.
    ///
    /// Loads the current manifest and compares epochs. If the stored epoch
    /// is higher than the local epoch, the writer has been fenced.
    #[instrument(name = "manifest_check_fenced", level = "debug", skip_all, fields(shard_id = %self.shard_id, local_epoch))]
    pub async fn check_fenced(&self, local_epoch: u64) -> Result<()> {
        // Refreshed: a fence is another process's write, which our own
        // position can never show us.
        let current = self.refresh_latest().await?;
        Self::check_fenced_against(&current, local_epoch, self.shard_id)
    }

    /// Check fencing against a pre-read manifest (avoids redundant read).
    fn check_fenced_against(
        manifest: &Option<ShardManifest>,
        local_epoch: u64,
        shard_id: Uuid,
    ) -> Result<()> {
        match manifest {
            Some(m) if m.writer_epoch > local_epoch => Err(Error::fenced_by_peer(format!(
                "local epoch {} < stored epoch {} for shard {}",
                local_epoch, m.writer_epoch, shard_id
            ))),
            _ => Ok(()),
        }
    }

    /// Update the manifest with retry on version conflict.
    ///
    /// This method:
    /// 1. Reads the latest manifest
    /// 2. Checks if fenced (fails immediately if so)
    /// 3. Calls `prepare_fn` to create the new manifest
    /// 4. Attempts to write
    /// 5. On version conflict, retries from step 1
    ///
    /// # Arguments
    ///
    /// * `local_epoch` - The writer's epoch (for fencing check)
    /// * `prepare_fn` - Function that takes current manifest and returns new
    ///   manifest. Its `version` must be `current.next_version()`; anything
    ///   else is rejected, so the sequence cannot develop a gap.
    ///
    /// # Returns
    ///
    /// The successfully written manifest.
    ///
    /// # Concurrency
    ///
    /// Each losing CAS clears the store's shared position, so commits that
    /// overlap within one CAS round-trip all fall back to a scan and retry —
    /// roughly `n^2/2` scans for `n` of them. Commits spaced further apart than
    /// that window cost nothing: the winner leaves its position warm for the
    /// next one.
    ///
    /// `MAX_RETRIES` therefore bounds how many commits can overlap on one
    /// handle: the unluckiest loses every round, so past ten concurrent commits
    /// it exhausts its budget and returns the conflict instead of landing.
    /// Reaching that needs eleven commit sources inside a single CAS, which no
    /// current caller comes close to. Worth revisiting if one funnels many
    /// independent writers through a single [`Self`].
    #[instrument(name = "manifest_commit_update", level = "debug", skip_all, fields(shard_id = %self.shard_id, local_epoch))]
    pub async fn commit_update<F>(&self, local_epoch: u64, prepare_fn: F) -> Result<ShardManifest>
    where
        F: Fn(&ShardManifest) -> ShardManifest,
    {
        self.try_commit_update(local_epoch, |current| Ok(prepare_fn(current)))
            .await
    }

    /// Like [`Self::commit_update`], with validation against the manifest used
    /// by each CAS attempt, including retries after a concurrent commit.
    pub(crate) async fn try_commit_update<F>(
        &self,
        local_epoch: u64,
        prepare_fn: F,
    ) -> Result<ShardManifest>
    where
        F: Fn(&ShardManifest) -> Result<ShardManifest>,
    {
        const MAX_RETRIES: usize = 10;

        for attempt in 0..MAX_RETRIES {
            // Step 1: take a position to build on. A cold cache — a fresh
            // store, or a retry after losing a race — must go to storage and
            // adopt what it finds, or the write below has no baseline. So must
            // an old one: pruning may have deleted the successor that would
            // reject a write built on it.
            let current = match self.fresh_cached() {
                Some(cached) => cached,
                None => self
                    .refresh_latest()
                    .await?
                    .ok_or_else(|| Error::io("Shard manifest not found"))?,
            };

            // Step 2: Check fencing
            Self::check_fenced_against(&Some(current.clone()), local_epoch, self.shard_id)?;

            // Step 3: Prepare new manifest
            let new_manifest = prepare_fn(&current)?;

            // Check the successor against `current`, the manifest the closure
            // actually built on. The store's position is shared and moves
            // under concurrent commits — a peer's failed CAS can clear it
            // between here and the write — so it cannot judge this.
            if new_manifest.version != current.next_version() {
                return Err(Error::invalid_input(format!(
                    "manifest version {} is not the successor of {} for shard {}: the version sequence must stay gap-free",
                    new_manifest.version, current.version, self.shard_id
                )));
            }

            // Validate epoch matches
            if new_manifest.writer_epoch != local_epoch {
                return Err(Error::invalid_input(format!(
                    "Manifest epoch {} doesn't match local epoch {}",
                    new_manifest.writer_epoch, local_epoch
                )));
            }

            // Step 4: Try to commit
            match self.write(&new_manifest).await {
                Ok(_) => {
                    return Ok(new_manifest);
                }
                Err(e) => {
                    // Check if it's a version conflict (can retry) vs other error
                    let is_version_conflict = matches!(e, Error::RetryableCommitConflict { .. });

                    if is_version_conflict && attempt < MAX_RETRIES - 1 {
                        continue;
                    }

                    return Err(e);
                }
            }
        }

        Err(Error::io(format!(
            "Failed to update manifest for shard {} after {} attempts",
            self.shard_id, MAX_RETRIES
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
    use std::sync::Mutex;
    use tempfile::TempDir;

    async fn create_local_store() -> (Arc<ObjectStore>, Path, TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        (store, path, temp_dir)
    }

    fn create_test_manifest(shard_id: Uuid, version: u64, epoch: u64) -> ShardManifest {
        ShardManifest {
            shard_id,
            version,
            shard_spec_id: 0,
            shard_field_values: HashMap::new(),
            writer_epoch: epoch,
            replay_after_wal_entry_position: 0,
            wal_entry_position_last_seen: 0,
            current_generation: 1,
            sstables: vec![],
            sealed_by: None,
            status: ShardStatus::Active,
        }
    }

    /// A warm cache must not hide a successor's claim from `check_fenced`.
    #[tokio::test]
    async fn check_fenced_sees_a_peer_through_a_warm_cache() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let incumbent = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        let successor = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        // Claiming writes a manifest, warming the cache.
        let (epoch, _) = incumbent.claim_epoch(0).await.unwrap();
        assert!(incumbent.cached().is_some(), "the claim write must cache");

        successor.claim_epoch(0).await.unwrap();

        assert!(
            incumbent.check_fenced(epoch).await.is_err(),
            "a cached manifest must not hide a successor's epoch"
        );
    }

    /// Reads must not cache, or a reader-only handle (a WAL tailer, a
    /// drop-reconcile probe) would never observe the writer.
    #[tokio::test]
    async fn a_reader_only_store_never_caches() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        let reader = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let (epoch, _) = writer.claim_epoch(0).await.unwrap();
        assert_eq!(
            reader.latest().await.unwrap().unwrap().current_generation,
            1
        );
        assert!(
            reader.cached().is_none(),
            "a read must not populate the cache"
        );

        writer
            .commit_update(epoch, |c| ShardManifest {
                version: c.version + 1,
                current_generation: 5,
                ..c.clone()
            })
            .await
            .unwrap();

        assert_eq!(
            reader.latest().await.unwrap().unwrap().current_generation,
            5,
            "a reader must see the writer's later commits"
        );
    }

    /// A losing `commit_update` re-reads from storage, so it converges instead
    /// of spinning on the version it lost on.
    #[tokio::test]
    async fn commit_update_recovers_from_a_stale_cache() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let ours = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        let peer = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let (epoch, _) = ours.claim_epoch(0).await.unwrap();
        ours.latest().await.unwrap().unwrap();

        // A same-epoch commit through another handle stales our cache.
        peer.commit_update(epoch, |c| ShardManifest {
            version: c.version + 1,
            current_generation: 7,
            ..c.clone()
        })
        .await
        .unwrap();

        let updated = ours
            .commit_update(epoch, |c| ShardManifest {
                version: c.version + 1,
                wal_entry_position_last_seen: 42,
                ..c.clone()
            })
            .await
            .unwrap();

        // Built on the peer's version, not on the stale cached one.
        assert_eq!(updated.current_generation, 7);
        assert_eq!(updated.wal_entry_position_last_seen, 42);
        assert_eq!(
            ours.refresh_latest().await.unwrap().unwrap().version,
            updated.version
        );
    }

    /// A write whose CAS won earlier but returned later must not publish its
    /// older manifest over the newer one.
    #[tokio::test]
    async fn a_late_write_never_regresses_the_cache() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let older = create_test_manifest(shard_id, 1, 1);
        let newer = create_test_manifest(shard_id, 2, 1);
        manifest_store.write(&older).await.unwrap();
        manifest_store.write(&newer).await.unwrap();

        // The straggler resolving after the newer write already cached.
        manifest_store.cache(&older, Instant::now());

        assert_eq!(
            manifest_store.latest().await.unwrap().unwrap().version,
            2,
            "the cache must hold the newest version this store wrote"
        );
    }

    /// The cached read and the storage read agree after a write.
    #[tokio::test]
    async fn latest_serves_the_written_manifest() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let mut manifest = create_test_manifest(shard_id, 1, 1);
        manifest_store.write(&manifest).await.unwrap();
        manifest.version = 2;
        manifest.current_generation = 9;
        manifest_store.write(&manifest).await.unwrap();

        let cached = manifest_store.latest().await.unwrap().unwrap();
        let durable = manifest_store.refresh_latest().await.unwrap().unwrap();
        assert_eq!(cached.version, 2);
        assert_eq!(cached.current_generation, 9);
        assert_eq!(cached, durable);
    }

    #[tokio::test]
    async fn test_latest_empty() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let result = manifest_store.latest().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_write_and_read_manifest() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let manifest = create_test_manifest(shard_id, 1, 1);
        manifest_store.write(&manifest).await.unwrap();

        let loaded = manifest_store.latest().await.unwrap().unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.writer_epoch, 1);
        assert_eq!(loaded.shard_id, shard_id);
    }

    #[tokio::test]
    async fn test_multiple_versions() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        // Write multiple versions
        for version in 1..=5 {
            let manifest = create_test_manifest(shard_id, version, version);
            manifest_store.write(&manifest).await.unwrap();
        }

        // Should find latest
        let loaded = manifest_store.latest().await.unwrap().unwrap();
        assert_eq!(loaded.version, 5);
        assert_eq!(loaded.writer_epoch, 5);

        // List should return all versions
        let versions = manifest_store.list_versions().await.unwrap();
        assert_eq!(versions, vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn test_read_specific_version() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        for version in 1..=3 {
            let manifest = create_test_manifest(shard_id, version, version * 10);
            manifest_store.write(&manifest).await.unwrap();
        }

        let v2 = manifest_store.read_version(2).await.unwrap();
        assert_eq!(v2.version, 2);
        assert_eq!(v2.writer_epoch, 20);
    }

    #[tokio::test]
    async fn test_put_if_not_exists() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let manifest1 = create_test_manifest(shard_id, 1, 1);
        manifest_store.write(&manifest1).await.unwrap();

        // Second write to same version should fail
        let manifest2 = create_test_manifest(shard_id, 1, 2);
        let result = manifest_store.write(&manifest2).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_initialize_shard_writes_v1_with_epoch_zero() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let mut field_values = HashMap::new();
        field_values.insert("user_bucket".to_string(), 7i32.to_le_bytes().to_vec());

        let manifest = manifest_store
            .initialize_shard(3, field_values.clone())
            .await
            .unwrap();
        assert_eq!(manifest.shard_id, shard_id);
        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.writer_epoch, 0);
        assert_eq!(manifest.shard_spec_id, 3);
        assert_eq!(manifest.shard_field_values, field_values);

        let loaded = manifest_store.latest().await.unwrap().unwrap();
        assert_eq!(loaded, manifest);
    }

    #[tokio::test]
    async fn test_initialize_shard_idempotent_on_match() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let mut field_values = HashMap::new();
        field_values.insert("k".to_string(), b"v".to_vec());

        let first = manifest_store
            .initialize_shard(1, field_values.clone())
            .await
            .unwrap();
        let second = manifest_store
            .initialize_shard(1, field_values)
            .await
            .unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn test_claim_epoch_after_cursor_update() {
        // After a tailer cursor update bumps the manifest version without
        // claiming an epoch, the next claim_epoch should observe the new
        // state and produce the next epoch — this guards against treating
        // a cursor update as a real claimant.
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let (first_epoch, first) = manifest_store.claim_epoch(0).await.unwrap();
        assert_eq!(first_epoch, 1);
        assert_eq!(first.version, 1);

        let mut cursor_update = first.clone();
        cursor_update.version += 1;
        cursor_update.wal_entry_position_last_seen = 42;
        manifest_store.write(&cursor_update).await.unwrap();

        let (second_epoch, second) = manifest_store.claim_epoch(0).await.unwrap();
        assert_eq!(second_epoch, 2);
        assert_eq!(second.version, 3);
        assert_eq!(second.wal_entry_position_last_seen, 42);
    }

    #[tokio::test]
    async fn test_claim_epoch_refuses_sealed_manifest() {
        // A `Sealed` manifest is the drop-table 2PC in-doubt marker:
        // `claim_epoch` must refuse it with a distinguishable error rather
        // than mint a new epoch, so a sealed shard can't be resurrected even
        // by a caller that skips its own status check. Rolling the status
        // back to `Active` makes the shard claimable again (reversible).
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let (epoch, claimed) = manifest_store.claim_epoch(0).await.unwrap();
        assert_eq!(claimed.status, ShardStatus::Active);

        // Seal it (drop-table prepare).
        let sealed = ShardManifest {
            version: claimed.version + 1,
            sealed_by: None,
            status: ShardStatus::Sealed,
            ..claimed
        };
        manifest_store.write(&sealed).await.unwrap();

        // The claim is refused with the distinguishable "sealed" error —
        // and the manifest is left untouched (no new epoch minted).
        let err = manifest_store.claim_epoch(0).await.unwrap_err();
        assert!(
            err.to_string().contains("sealed"),
            "expected a distinguishable sealed-refusal error, got: {err}"
        );
        let after = manifest_store.latest().await.unwrap().unwrap();
        assert_eq!(after.writer_epoch, sealed.writer_epoch, "no epoch minted");
        assert_eq!(after.status, ShardStatus::Sealed);

        // Roll back to Active (drop-table abort) → re-claimable.
        let active = ShardManifest {
            version: sealed.version + 1,
            sealed_by: None,
            status: ShardStatus::Active,
            ..sealed
        };
        manifest_store.write(&active).await.unwrap();
        let (next_epoch, reclaimed) = manifest_store.claim_epoch(0).await.unwrap();
        assert!(next_epoch > epoch, "rolled-back shard mints the next epoch");
        assert_eq!(reclaimed.status, ShardStatus::Active);
    }

    #[tokio::test]
    async fn test_initialize_shard_rejects_conflict_with_mismatch() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        manifest_store
            .initialize_shard(1, HashMap::new())
            .await
            .unwrap();

        let mut other = HashMap::new();
        other.insert("k".to_string(), b"v".to_vec());
        let result = manifest_store.initialize_shard(1, other).await;
        assert!(
            result.is_err(),
            "second initialize_shard with different fields must fail"
        );
    }

    /// A commit closure that names the wrong version fails loudly instead of
    /// having its intent rewritten underneath it.
    #[tokio::test]
    async fn commit_update_rejects_a_closure_that_skips_a_version() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let ours = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let (epoch, claimed) = ours.claim_epoch(0).await.unwrap();

        let error = ours
            .commit_update(epoch, |c| ShardManifest {
                version: 99,
                current_generation: 7,
                ..c.clone()
            })
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("is not the successor of"), "{}", error);
        assert_eq!(
            ours.refresh_latest().await.unwrap().unwrap().version,
            claimed.version,
            "the rejected commit left the shard alone"
        );

        // The same edit with the right version commits.
        let committed = ours
            .commit_update(epoch, |c| ShardManifest {
                version: c.next_version(),
                current_generation: 7,
                ..c.clone()
            })
            .await
            .unwrap();
        assert_eq!(committed.version, claimed.next_version());
        assert_eq!(committed.current_generation, 7);
    }

    /// The two reads differ in one thing that matters: `refresh_latest` adopts
    /// what it finds as this store's position, `latest` does not. Getting that
    /// backwards either pins pollers or rejects valid writes.
    #[tokio::test]
    async fn only_refresh_latest_adopts_a_position() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        let observer = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let (epoch, _) = writer.claim_epoch(0).await.unwrap();

        // A plain read leaves the observer positionless, so it keeps going to
        // storage and keeps seeing the writer.
        assert!(observer.latest().await.unwrap().is_some());
        assert!(
            observer.cached().is_none(),
            "`latest` must not take a position"
        );

        writer
            .commit_update(epoch, |c| ShardManifest {
                version: c.next_version(),
                current_generation: 9,
                ..c.clone()
            })
            .await
            .unwrap();
        assert_eq!(
            observer.latest().await.unwrap().unwrap().current_generation,
            9,
            "a poller must observe the writer's later commits"
        );

        // Refreshing takes a position, which is what lets a claim write from it.
        let refreshed = observer.refresh_latest().await.unwrap().unwrap();
        assert_eq!(
            observer.cached().map(|c| c.version),
            Some(refreshed.version),
            "`refresh_latest` must take a position"
        );
    }

    /// The store's position moves under concurrent commits — a peer's failed
    /// CAS clears it — so it cannot judge a closure's output. Every commit
    /// must land, and none may be lost.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_commits_on_one_handle_all_land() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let shared = Arc::new(ShardManifestStore::new(store, &base_path, "", shard_id, 2));
        let (epoch, claimed) = shared.claim_epoch(0).await.unwrap();

        const COMMITS: u64 = 8;
        let mut tasks = Vec::new();
        for _ in 0..COMMITS {
            let shared = shared.clone();
            tasks.push(tokio::spawn(async move {
                shared
                    .commit_update(epoch, |c| ShardManifest {
                        version: c.next_version(),
                        current_generation: c.current_generation + 1,
                        ..c.clone()
                    })
                    .await
            }));
        }

        let mut failures = Vec::new();
        for task in tasks {
            if let Err(error) = task.await.unwrap() {
                failures.push(error.to_string());
            }
        }
        assert!(failures.is_empty(), "commits failed: {:#?}", failures);

        let tip = shared.refresh_latest().await.unwrap().unwrap();
        assert_eq!(
            tip.current_generation,
            claimed.current_generation + COMMITS,
            "every commit must be reflected; a lost update means one was \
             built on stale state and overwrote an intervening one"
        );
        assert_eq!(tip.version, claimed.version + COMMITS);
    }
    /// A version this store's own position proves is taken must read as a
    /// collision, so `commit_update` retries instead of failing.
    #[tokio::test]
    async fn write_reports_a_taken_version_as_a_collision() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let ours = ShardManifestStore::new(store, &base_path, "", shard_id, 2);

        let (_, claimed) = ours.claim_epoch(0).await.unwrap();

        let mut replay = claimed.clone();
        replay.current_generation = 42;
        let error = ours.write(&replay).await.unwrap_err();
        assert!(
            matches!(error, Error::RetryableCommitConflict { .. }),
            "the variant is what commit_update retries on: {:?}",
            error
        );
        assert!(
            ours.cached().is_none(),
            "a collision must drop the position so the retry re-reads"
        );
    }

    /// Writes versions `1..=last` of one shard, the hint naming `last`.
    async fn write_versions(store: &ShardManifestStore, shard_id: Uuid, last: u64) {
        for version in 1..=last {
            store
                .write(&create_test_manifest(shard_id, version, 1))
                .await
                .unwrap();
        }
    }

    fn hint_path(base_path: &Path, shard_id: &Uuid) -> Path {
        shard_manifest_path(base_path, "", shard_id).join("version_hint.json")
    }

    async fn write_hint(store: &ObjectStore, base_path: &Path, shard_id: &Uuid, version: u64) {
        let hint = serde_json::to_vec(&VersionHint { version }).unwrap();
        store
            .inner
            .put(&hint_path(base_path, shard_id), Bytes::from(hint).into())
            .await
            .unwrap();
    }

    /// `store`, with every `method` call on a path `fails` matches refused.
    fn failing_store(
        store: &ObjectStore,
        method: &'static str,
        fails: impl Fn(&Path) -> bool + Send + Sync + 'static,
    ) -> Arc<ObjectStore> {
        let policy = Arc::new(Mutex::new(ProxyObjectStorePolicy::new()));
        policy.lock().unwrap().set_before_policy(
            method,
            Arc::new(move |called: &str, path: &Path| {
                if called == method && fails(path) {
                    return Err(object_store::Error::Generic {
                        store: "test",
                        source: format!("{method} refused").into(),
                    }
                    .into());
                }
                Ok(())
            }),
        );
        let mut proxied = store.clone();
        proxied.inner = Arc::new(ProxyObjectStore::new(store.inner.clone(), policy));
        Arc::new(proxied)
    }

    /// Once old versions are pruned, the latest is still found with no hint
    /// and with a hint naming a pruned version, and a claim builds on it.
    #[tokio::test]
    async fn the_latest_version_survives_pruned_history() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
            .with_ages(Duration::ZERO, Duration::ZERO);
        write_versions(&writer, shard_id, 20).await;
        assert_eq!(writer.prune_versions_before(20).await.unwrap(), 19);
        assert_eq!(writer.list_versions().await.unwrap(), vec![20]);

        write_hint(&store, &base_path, &shard_id, 3).await;
        let reader = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        assert_eq!(reader.refresh_latest().await.unwrap().unwrap().version, 20);

        store
            .inner
            .delete(&hint_path(&base_path, &shard_id))
            .await
            .unwrap();
        let claimant = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        let (epoch, claimed) = claimant.claim_epoch(0).await.unwrap();
        assert_eq!((claimed.version, epoch), (21, 2));
    }

    /// Pruning keeps the latest version and refuses a bound past it.
    #[tokio::test]
    async fn pruning_never_removes_the_latest_version() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
            .with_ages(Duration::ZERO, Duration::ZERO);
        write_versions(&writer, shard_id, 5).await;

        assert!(writer.prune_versions_before(6).await.is_err());
        assert_eq!(writer.list_versions().await.unwrap(), vec![1, 2, 3, 4, 5]);
        assert_eq!(writer.prune_versions_before(3).await.unwrap(), 2);
        assert_eq!(writer.list_versions().await.unwrap(), vec![3, 4, 5]);
    }

    /// A writer that lost its shard without noticing is still rejected after
    /// pruning: either the successor that rejects it is too young to prune, or
    /// its position is too old to build on and it re-reads.
    #[rstest::rstest]
    #[case::successor_too_young_to_prune(MAX_POSITION_AGE, MIN_PRUNE_AGE, 5)]
    #[case::stale_position_is_reread(Duration::ZERO, Duration::ZERO, 0)]
    #[tokio::test]
    async fn pruning_never_unfences_a_stale_writer(
        #[case] max_position_age: Duration,
        #[case] min_prune_age: Duration,
        #[case] versions_left_below_tip: usize,
    ) {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let stale = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
            .with_ages(max_position_age, min_prune_age);
        let peer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
            .with_ages(max_position_age, min_prune_age);
        let (stale_epoch, _) = stale.claim_epoch(0).await.unwrap();
        let (peer_epoch, _) = peer.claim_epoch(0).await.unwrap();
        for generation in 3..=6 {
            peer.commit_update(peer_epoch, |c| ShardManifest {
                version: c.next_version(),
                current_generation: generation,
                ..c.clone()
            })
            .await
            .unwrap();
        }
        peer.prune_versions_before(6).await.unwrap();
        assert_eq!(
            peer.list_versions().await.unwrap().len(),
            versions_left_below_tip + 1
        );

        let result = stale
            .commit_update(stale_epoch, |c| ShardManifest {
                version: c.next_version(),
                current_generation: 99,
                ..c.clone()
            })
            .await;
        assert!(
            matches!(result, Err(Error::Fenced { .. })),
            "the stale writer must be rejected: {result:?}"
        );
        let reader = ShardManifestStore::new(store, &base_path, "", shard_id, 2);
        let tip = reader.refresh_latest().await.unwrap().unwrap();
        assert_eq!(
            (tip.version, tip.writer_epoch, tip.current_generation),
            (6, peer_epoch, 6)
        );
    }

    /// A listing that fails is an error, never an empty shard a claim would
    /// reset.
    #[tokio::test]
    async fn a_failed_listing_is_an_error() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        write_versions(&writer, shard_id, 3).await;
        store
            .inner
            .delete(&hint_path(&base_path, &shard_id))
            .await
            .unwrap();

        let failing = failing_store(&store, "list", |_| true);
        let reader = ShardManifestStore::new(failing, &base_path, "", shard_id, 2);
        assert!(reader.refresh_latest().await.is_err());
    }

    /// A prune that fails partway deletes the versions below the failure and
    /// leaves no gap above it, so a lookup from the oldest survivor still
    /// reaches the latest.
    #[tokio::test]
    async fn a_failed_prune_leaves_the_versions_contiguous() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        write_versions(&writer, shard_id, 6).await;

        let third =
            shard_manifest_path(&base_path, "", &shard_id).join(manifest_filename(3).as_str());
        let failing = failing_store(&store, "delete", move |path| *path == third);
        let pruner = ShardManifestStore::new(failing, &base_path, "", shard_id, 2)
            .with_ages(Duration::ZERO, Duration::ZERO);
        assert!(pruner.prune_versions_before(6).await.is_err());
        assert_eq!(writer.list_versions().await.unwrap(), vec![3, 4, 5, 6]);

        write_hint(&store, &base_path, &shard_id, 3).await;
        let reader = ShardManifestStore::new(store, &base_path, "", shard_id, 2);
        assert_eq!(reader.refresh_latest().await.unwrap().unwrap().version, 6);
    }

    /// Initializing a shard whose version 1 was pruned writes nothing and
    /// returns its latest.
    #[tokio::test]
    async fn initializing_a_pruned_shard_keeps_its_tip() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
            .with_ages(Duration::ZERO, Duration::ZERO);
        write_versions(&writer, shard_id, 6).await;
        writer.prune_versions_before(6).await.unwrap();

        let retry = ShardManifestStore::new(store, &base_path, "", shard_id, 2);
        let existing = retry.initialize_shard(0, HashMap::new()).await.unwrap();
        assert_eq!(existing.version, 6);
        assert_eq!(writer.list_versions().await.unwrap(), vec![6]);
    }

    /// A hint that cannot be read falls back to listing like a missing one,
    /// so a pruned history still resolves to the latest version.
    #[tokio::test]
    async fn an_unreadable_hint_falls_back_to_listing() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2)
            .with_ages(Duration::ZERO, Duration::ZERO);
        write_versions(&writer, shard_id, 3).await;
        writer.prune_versions_before(3).await.unwrap();

        let failing = failing_store(&store, "get_opts", |path| {
            path.as_ref().ends_with("version_hint.json")
        });
        let reader = ShardManifestStore::new(failing, &base_path, "", shard_id, 2);
        assert_eq!(reader.refresh_latest().await.unwrap().unwrap().version, 3);
    }

    /// A HEAD that fails is not a version that is absent, and the scan must not
    /// read it as the end of the sequence: the answer becomes a position.
    #[tokio::test]
    async fn a_failed_head_is_not_read_as_the_end_of_the_sequence() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();

        // The durable tip is v3, written by a peer that claimed epoch 2.
        let peer = ShardManifestStore::new(store.clone(), &base_path, "", shard_id, 2);
        for version in 1..=3u64 {
            let epoch = if version == 3 { 2 } else { 1 };
            peer.write(&create_test_manifest(shard_id, version, epoch))
                .await
                .unwrap();
        }

        // The hint is written after the manifest and is best-effort, so lagging
        // by one is the ordinary state during any commit.
        let hint_path = shard_manifest_path(&base_path, "", &shard_id).join("version_hint.json");
        store
            .inner
            .put(
                &hint_path,
                Bytes::from(serde_json::to_vec(&VersionHint { version: 2 }).unwrap()).into(),
            )
            .await
            .unwrap();

        // A store whose HEAD on v3 gets a transient 503.
        let policy = Arc::new(Mutex::new(ProxyObjectStorePolicy::new()));
        let v3_file = manifest_filename(3);
        policy.lock().unwrap().set_before_policy(
            "503",
            Arc::new(move |method: &str, path: &Path| {
                if method == "get_opts" && path.as_ref().ends_with(v3_file.as_str()) {
                    return Err(object_store::Error::Generic {
                        store: "test",
                        source: "503 slow down".into(),
                    }
                    .into());
                }
                Ok(())
            }),
        );
        let mut proxied = (*store).clone();
        proxied.inner = Arc::new(ProxyObjectStore::new(store.inner.clone(), policy.clone()));
        let ours = ShardManifestStore::new(Arc::new(proxied), &base_path, "", shard_id, 2);

        let err = ours.refresh_latest().await.unwrap_err();
        assert!(
            err.to_string().contains("503"),
            "the scan must surface the HEAD failure, got: {err}"
        );
        assert!(ours.cached().is_none(), "a failed scan takes no position");
        assert!(
            ours.check_fenced(1).await.is_err(),
            "a fence check that could not read the tip must not report clear"
        );

        // Once the blip clears, the scan sees the durable tip.
        policy.lock().unwrap().clear_before_policy("503");
        assert_eq!(ours.latest().await.unwrap().unwrap().version, 3);
    }
}
