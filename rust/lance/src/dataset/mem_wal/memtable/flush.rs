// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemTable flush to persistent storage.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use bytes::Bytes;
use datafusion::physical_plan::SendableRecordBatchStream;
use lance_core::cache::LanceCache;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use lance_file::version::ConcreteFileVersion;
use lance_index::mem_wal::{ShardManifest, SsTable};
use lance_index::scalar::ScalarIndexParams;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_table::format::IndexMetadata;
use lance_table::io::commit::write_manifest_file_to_path;
use lance_table::io::deletion::write_deletion_file;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use roaring::RoaringBitmap;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use super::super::index::{FlushContext, FlushOutcome, GenerationWrite, IndexStore, MemIndexSpec};
use super::super::memtable::MemTable;
use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::mem_wal::manifest::ShardManifestStore;
use crate::dataset::mem_wal::scanner::SsTableWarmer;
use crate::dataset::mem_wal::scanner::exec::{compute_pk_hash, validate_pk_types};
use crate::dataset::mem_wal::util::{derived_store_params, generate_random_hash, sstable_path};
use crate::dataset::write::InsertBuilder;
use crate::index::CreateIndexBuilder;
use crate::session::Session;

#[derive(Debug, Clone)]
pub struct FlushResult {
    pub sstable: SsTable,
    pub rows_flushed: usize,
    pub covered_wal_entry_position: u64,
}

/// Build the within-generation deletion vector for forward-written flush data.
///
/// `batches` are in on-disk (insert) order, so the newest version of each
/// primary key is at the largest offset: the last occurrence of a PK hash is
/// kept and every earlier occurrence is marked deleted. Keys are hashed
/// (collisions accepted, consistent with the read path).
fn compute_dedup_deletions(batches: &[RecordBatch], pk_indices: &[usize]) -> RoaringBitmap {
    let mut deleted = RoaringBitmap::new();
    let mut latest: HashMap<u64, u32> = HashMap::new();
    let mut offset: u32 = 0;
    for batch in batches {
        for row in 0..batch.num_rows() {
            let pk_hash = compute_pk_hash(batch, pk_indices, row);
            if let Some(previous) = latest.insert(pk_hash, offset) {
                // An earlier (older) occurrence of this PK is now superseded.
                deleted.insert(previous);
            }
            offset += 1;
        }
    }
    deleted
}

pub struct MemTableFlusher {
    object_store: Arc<ObjectStore>,
    base_path: Path,
    base_uri: String,
    shard_id: Uuid,
    manifest_store: Arc<ShardManifestStore>,
    /// When present, each new generation is warmed before it is committed, so
    /// the first query sees zero cold reads. `None` => no warming.
    warmer: Option<Arc<dyn SsTableWarmer>>,
    /// Store params the base dataset was opened with, reused for the flusher's
    /// own opens + writes. Used verbatim only for the base's own URI; generation
    /// URIs go through [`derived_store_params`]. `None` opens by URI alone.
    store_params: Option<ObjectStoreParams>,
    /// Session for those opens, sharing the base's store registry. `None` opens
    /// with a fresh session.
    session: Option<Arc<Session>>,
}

/// What a flushed generation holds, for the manifest entry recording it.
///
/// Read off the memtable being flushed, which is frozen. That matters: an
/// appending store bumps these counters before it publishes the batch, so only
/// a sealed one agrees with what a scan of it will see.
#[derive(Clone, Copy)]
struct FlushedSize {
    in_memory_bytes: Option<u64>,
    physical_rows: Option<u64>,
    primary_key_bytes: Option<u64>,
}

impl FlushedSize {
    /// Zero reads as unmeasured. An empty memtable is refused before a flush
    /// gets here, so a flushed generation always holds rows and a zero can only
    /// mean the accounting failed.
    fn of(memtable: &MemTable) -> Self {
        Self {
            // `row_bytes`, not the store's retained heap: the window being
            // written is what a reader of this generation gets back.
            in_memory_bytes: Some(memtable.batch_store().row_bytes() as u64).filter(|b| *b > 0),
            physical_rows: Some(memtable.row_count() as u64).filter(|r| *r > 0),
            // Zero means the table has no primary key, which is not a
            // measurement of one.
            primary_key_bytes: Some(memtable.pk_bytes() as u64).filter(|b| *b > 0),
        }
    }

    fn sstable(self, generation: u64, path: String) -> SsTable {
        SsTable {
            generation,
            path,
            in_memory_bytes: self.in_memory_bytes,
            physical_rows: self.physical_rows,
            primary_key_bytes: self.primary_key_bytes,
        }
    }
}

/// Rows per training batch handed to an index builder at flush.
const TRAINING_BATCH_SIZE: usize = 8192;

impl MemTableFlusher {
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        base_uri: impl Into<String>,
        shard_id: Uuid,
        manifest_store: Arc<ShardManifestStore>,
    ) -> Self {
        Self {
            object_store,
            base_path,
            base_uri: base_uri.into(),
            shard_id,
            manifest_store,
            warmer: None,
            store_params: None,
            session: None,
        }
    }

    /// Attach the warmer fired pre-commit for each new generation.
    pub fn with_warmer(mut self, warmer: Option<Arc<dyn SsTableWarmer>>) -> Self {
        self.warmer = warmer;
        self
    }

    /// Set the store params + session used for derived-URI opens. Injected by
    /// `mem_wal_writer` from the base `Dataset`.
    pub fn with_storage_context(
        mut self,
        store_params: Option<ObjectStoreParams>,
        session: Option<Arc<Session>>,
    ) -> Self {
        self.store_params = store_params;
        self.session = session;
        self
    }

    /// Open the base table, reusing the injected store params verbatim — they
    /// were resolved for exactly this URI, so a path-bound `object_store`
    /// binding still points where it should.
    async fn open_base(&self) -> Result<Dataset> {
        self.open_uri(&self.base_uri, self.store_params.clone())
            .await
    }

    /// Open an SSTable under `_mem_wal/`. The params must be adapted
    /// first: a path-bound store binding would redirect the open at the base
    /// table (see [`derived_store_params`]).
    async fn open_generation(&self, uri: &str) -> Result<Dataset> {
        self.open_uri(uri, self.store_params.as_ref().map(derived_store_params))
            .await
    }

    /// Open `uri` with the injected session, or by URI alone when nothing was
    /// injected.
    async fn open_uri(
        &self,
        uri: &str,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Dataset> {
        let mut builder = DatasetBuilder::from_uri(uri);
        if let Some(params) = store_params {
            builder = builder.with_store_params(params);
        }
        if let Some(session) = &self.session {
            builder = builder.with_session(session.clone());
        }
        builder.load().await
    }

    /// Warm a just-written generation before it is committed. Best-effort: a
    /// failure is logged and the flush proceeds — warming is never a commit
    /// gate. No-op without a warmer. `uri` must be the resolved reader path
    /// (`path_to_uri(gen_path)`) so warmed entries key-match later queries.
    async fn warm_generation(&self, uri: &str) {
        let Some(warmer) = &self.warmer else {
            return;
        };
        if let Err(e) = warmer.warm(uri).await {
            warn!("pre-commit warm failed for generation {uri}; committing cold: {e}");
        }
    }

    /// Construct a full URI for a path within the base dataset.
    fn path_to_uri(&self, path: &Path) -> String {
        let path_str = path.as_ref();
        let base_str = self.base_path.as_ref();

        let relative = if let Some(stripped) = path_str.strip_prefix(base_str) {
            stripped.trim_start_matches('/')
        } else {
            path_str
        };

        let base = self.base_uri.trim_end_matches('/');
        if relative.is_empty() {
            base.to_string()
        } else {
            format!("{}/{}", base, relative)
        }
    }

    fn generation_target(&self, memtable: &MemTable) -> Result<(String, Path, Option<String>)> {
        let generation = memtable.generation();
        if let Some(target) = memtable.target() {
            if target.generation != generation {
                return Err(Error::internal(format!(
                    "memtable generation {} carries data target generation {}",
                    generation, target.generation
                )));
            }
            return Ok((
                target.generation_dir.clone(),
                target.generation_path(&self.base_path, &self.shard_id),
                Some(target.data_file_name.clone()),
            ));
        }

        // Blob-free memtables do not reserve a storage identity during puts.
        // Preserve the existing flush-time random generation and file naming
        // for that path.
        let random_hash = generate_random_hash();
        Ok((
            format!("{}_gen_{}", random_hash, generation),
            sstable_path(&self.base_path, &self.shard_id, &random_hash, generation),
            None,
        ))
    }

    /// Storage file version of the shard's base dataset. SSTables
    /// (data fragments and index files) are written at this same version so the
    /// whole shard stays on one format (e.g. a 2.2 base => 2.2 SSTables).
    ///
    /// Falls back to the default selector's exact version when no base dataset exists at
    /// `base_uri` (e.g. flusher unit tests that run without a committed base).
    /// In production MemWAL is always initialized on a real dataset, so the base
    /// version is inherited; other open errors are propagated.
    async fn base_storage_version(&self) -> Result<ConcreteFileVersion> {
        match self.open_base().await {
            Ok(dataset) => Ok(dataset.manifest().data_storage_format.lance_file_format()),
            Err(Error::DatasetNotFound { .. }) => Ok(lance_file::version::stable_file_version()),
            Err(e) => Err(e),
        }
    }

    /// Flush the MemTable to storage (data files, indexes, bloom filter).
    ///
    /// `covered_wal_entry_position` is stamped into the manifest's
    /// `replay_after_wal_entry_position` so post-restart replay skips the
    /// WAL entries this generation captures. Pass 0 only for shards that
    /// have not yet appended any WAL entry — non-zero positions are
    /// 1-based (see `FIRST_WAL_ENTRY_POSITION`).
    #[instrument(name = "mt_flush_storage", level = "info", skip_all, fields(shard_id = %self.shard_id, epoch, generation = memtable.generation(), row_count = memtable.row_count()))]
    pub async fn flush(
        &self,
        memtable: &MemTable,
        epoch: u64,
        covered_wal_entry_position: u64,
        durable: usize,
    ) -> Result<FlushResult> {
        self.manifest_store.check_fenced(epoch).await?;

        if memtable.row_count() == 0 {
            return Err(Error::invalid_input("Cannot flush empty MemTable"));
        }

        if !memtable.all_flushed_to_wal(durable) {
            return Err(Error::invalid_input(
                "MemTable has unflushed fragments - WAL flush required first",
            ));
        }

        let generation = memtable.generation();
        let size = FlushedSize::of(memtable);
        let (gen_folder_name, gen_path, preassigned_data_file_name) =
            self.generation_target(memtable)?;

        info!(
            "Flushing MemTable generation {} to {} ({} rows, {} batches)",
            generation,
            gen_path,
            memtable.row_count(),
            memtable.batch_count()
        );

        let (rows_flushed, deleted) = self
            .write_data_file(
                &gen_path,
                memtable,
                preassigned_data_file_name.as_deref(),
                self.base_storage_version().await?,
            )
            .await?;

        // Persist the within-generation deletion vector so the
        // SSTable exposes newest-per-PK on every read path.
        if !deleted.is_empty() {
            let uri = self.path_to_uri(&gen_path);
            let dataset = self.open_generation(&uri).await?;
            self.finalize_generation(&dataset, &deleted, None).await?;
        }

        let bloom_path = gen_path.clone().join("bloom_filter.bin");
        self.write_bloom_filter(&bloom_path, memtable.bloom_filter())
            .await?;

        // Write the standalone primary-key dedup sidecar. A primary key needs
        // no secondary index, so this is required on the plain-flush path too —
        // the LSM scanner opens it to dedup the generation. (`flush_with_indexes`
        // writes it on the indexed path.) No-op when the memtable has no PK.
        self.create_pk_index(&gen_path, memtable.indexes()).await?;

        // Warm before commit (zero cold window); no-op without a warmer.
        let warm_uri = self.path_to_uri(&gen_path);
        self.warm_generation(&warm_uri).await;

        let new_manifest = self
            .update_manifest(
                epoch,
                generation,
                &gen_folder_name,
                covered_wal_entry_position,
                size,
            )
            .await?;

        info!(
            "Flushed SSTable {} for shard {} (manifest version {})",
            generation, self.shard_id, new_manifest.version
        );

        Ok(FlushResult {
            sstable: size.sstable(generation, gen_folder_name),
            rows_flushed,
            covered_wal_entry_position,
        })
    }

    /// Write the data file in insert (forward) order.
    ///
    /// Returns the total number of rows written and the within-generation
    /// deletion vector marking every older duplicate of each primary key (see
    /// [`compute_dedup_deletions`]). Forward order keeps the data file, the
    /// incrementally-built indexes, and the deletion-vector offsets in one
    /// position space (newest = largest offset) with no remap.
    #[instrument(name = "mt_write_data_file", level = "debug", skip_all, fields(path = %path))]
    async fn write_data_file(
        &self,
        path: &Path,
        memtable: &MemTable,
        preassigned_data_file_name: Option<&str>,
        storage_version: ConcreteFileVersion,
    ) -> Result<(usize, RoaringBitmap)> {
        use arrow_array::RecordBatchIterator;

        use crate::dataset::WriteParams;

        if memtable.row_count() == 0 {
            return Ok((0, RoaringBitmap::new()));
        }

        let batches = memtable.scan_batches().await?;
        if batches.is_empty() {
            return Ok((0, RoaringBitmap::new()));
        }
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        // Build the deletion vector before `batches` is moved into the writer.
        let pk_columns: Vec<String> = memtable
            .lance_schema()
            .unenforced_primary_key()
            .iter()
            .map(|f| f.name.clone())
            .collect();
        let deleted = if pk_columns.is_empty() {
            RoaringBitmap::new()
        } else {
            let schema = batches[0].schema();
            // Match the read-path contract (create_dedup_plan): unsupported PK
            // types must error here rather than hit compute_pk_hash's
            // debug-format fallback, which can collapse distinct keys.
            validate_pk_types(schema.as_ref(), &pk_columns)?;
            let pk_indices = pk_columns
                .iter()
                .map(|c| {
                    schema.index_of(c).map_err(|_| {
                        Error::invalid_input(format!(
                            "Primary key column '{}' not found in flush schema",
                            c
                        ))
                    })
                })
                .collect::<Result<Vec<usize>>>()?;
            compute_dedup_deletions(&batches, &pk_indices)
        };

        let uri = self.path_to_uri(path);
        if let Some(preassigned_data_file_name) = preassigned_data_file_name {
            match self.open_generation(&uri).await {
                Ok(dataset) => {
                    let fragments = dataset.get_fragments();
                    let is_expected = fragments.len() == 1
                        && fragments[0].metadata().physical_rows == Some(total_rows)
                        && fragments[0]
                            .metadata()
                            .files
                            .iter()
                            .any(|file| file.path == preassigned_data_file_name);
                    if is_expected {
                        return Ok((total_rows, deleted));
                    }
                    return Err(Error::io(format!(
                        "generation {} already exists but does not describe the expected file {} and {} rows",
                        path, preassigned_data_file_name, total_rows
                    )));
                }
                Err(Error::DatasetNotFound { .. }) => {}
                Err(error) => return Err(error),
            }
        }

        let reader =
            RecordBatchIterator::new(batches.into_iter().map(Ok), memtable.schema().clone());

        // Use very large max_rows_per_file to ensure 1 fragment per SSTable.
        // Inherit the base dataset's storage version so the SSTable
        // matches it (a 2.2 base also fixes the v2.1 miniblock 32 KiB chunk cap
        // that the dense HNSW graph List columns overflow at scale).
        let write_params = WriteParams {
            max_rows_per_file: usize::MAX,
            data_storage_version: Some(storage_version.to_selector()),
            // Write the generation through the base's store params + session so it
            // uses the same store the base was opened with. Adapted for the
            // generation URI: a path-bound store binding would send this write at
            // the base table's own path (see [`derived_store_params`]).
            store_params: self.store_params.as_ref().map(derived_store_params),
            session: self.session.clone(),
            ..Default::default()
        };
        let mut builder = InsertBuilder::new(uri.as_str()).with_params(&write_params);
        if let Some(preassigned_data_file_name) = preassigned_data_file_name {
            builder = builder.with_preassigned_data_file_name(preassigned_data_file_name);
        }
        builder.execute_stream(reader).await?;

        Ok((total_rows, deleted))
    }

    /// Persist the within-generation deletion vector (and any indexes) onto the
    /// just-written generation by rewriting its manifest in place.
    ///
    /// The generation dataset is brand-new and not yet published in the shard
    /// manifest, so overwriting its v1 manifest is safe. A no-op when there is
    /// neither a deletion vector nor an index to record.
    async fn finalize_generation(
        &self,
        dataset: &Dataset,
        deleted: &RoaringBitmap,
        indexes: Option<Vec<IndexMetadata>>,
    ) -> Result<()> {
        let indexes = indexes.filter(|i| !i.is_empty());
        if deleted.is_empty() && indexes.is_none() {
            return Ok(());
        }

        let mut manifest = dataset.manifest().clone();
        let manifest_path = dataset.manifest_location().path.clone();

        if !deleted.is_empty() {
            let dv = DeletionVector::from(deleted.clone());
            let deletion_file = write_deletion_file(
                &dataset.base,
                0, // 1 fragment per SSTable
                dataset.version().version,
                &dv,
                dataset.object_store.as_ref(),
            )
            .await?;
            let fragments = Arc::make_mut(&mut manifest.fragments);
            if let Some(fragment) = fragments.first_mut() {
                fragment.deletion_file = deletion_file;
            }
        }

        // Clear stale section offsets from the v1 manifest since the rewritten
        // file has a different layout (added index/deletion metadata).
        manifest.index_section = None;
        manifest.transaction_section = None;
        manifest.transaction_file = None;
        write_manifest_file_to_path(
            &self.object_store,
            &mut manifest,
            indexes,
            &manifest_path,
            None,
        )
        .await
        .map_err(|e| Error::io(format!("Failed to write generation manifest: {}", e)))?;
        Ok(())
    }

    async fn write_bloom_filter(
        &self,
        path: &Path,
        bloom: &lance_core::utils::bloomfilter::sbbf::Sbbf,
    ) -> Result<()> {
        let data = bloom.to_bytes();
        self.object_store
            .inner
            .put(path, Bytes::from(data).into())
            .await
            .map_err(|e| Error::io(format!("Failed to write bloom filter: {}", e)))?;
        Ok(())
    }

    /// Flush the MemTable to storage with indexes.
    ///
    /// See [`MemTableFlusher::flush`] for `covered_wal_entry_position`
    /// semantics.
    #[instrument(name = "mt_flush_with_indexes", level = "info", skip_all, fields(shard_id = %self.shard_id, epoch, generation = memtable.generation(), row_count = memtable.row_count(), index_count = index_specs.len()))]
    pub async fn flush_with_indexes(
        &self,
        memtable: &MemTable,
        epoch: u64,
        index_specs: &[MemIndexSpec],
        covered_wal_entry_position: u64,
        durable: usize,
    ) -> Result<FlushResult> {
        self.manifest_store.check_fenced(epoch).await?;

        if memtable.row_count() == 0 {
            return Err(Error::invalid_input("Cannot flush empty MemTable"));
        }

        if !memtable.all_flushed_to_wal(durable) {
            return Err(Error::invalid_input(
                "MemTable has unflushed fragments - WAL flush required first",
            ));
        }

        let generation = memtable.generation();
        let size = FlushedSize::of(memtable);
        let (gen_folder_name, gen_path, preassigned_data_file_name) =
            self.generation_target(memtable)?;

        info!(
            "Flushing MemTable generation {} with indexes to {} ({} rows, {} batches)",
            generation,
            gen_path,
            memtable.row_count(),
            memtable.batch_count()
        );

        let storage_version = self.base_storage_version().await?;
        let (total_rows, deleted) = self
            .write_data_file(
                &gen_path,
                memtable,
                preassigned_data_file_name.as_deref(),
                storage_version,
            )
            .await?;

        // Dataset::write already committed the data; the indexes join it in
        // one manifest below.
        let uri = self.path_to_uri(&gen_path);
        let mut dataset = self.open_generation(&uri).await?;
        let all_indexes = self
            .create_indexes(
                &mut dataset,
                &gen_path,
                index_specs,
                memtable.indexes(),
                total_rows,
                crate::dataset::versions::index_file_version(storage_version),
            )
            .await?;
        info!(
            generation,
            index_count = all_indexes.len(),
            "created indexes on SSTable"
        );

        // Write the standalone primary-key dedup index (sidecar, not a manifest
        // index — the block-list opens it directly by path).
        self.create_pk_index(&gen_path, memtable.indexes()).await?;

        // Write a single manifest that records the fragments, the
        // within-generation deletion vector, and all indexes, overwriting the
        // data-only v1 manifest created by Dataset::write.
        self.finalize_generation(&dataset, &deleted, Some(all_indexes))
            .await?;

        let bloom_path = gen_path.clone().join("bloom_filter.bin");
        self.write_bloom_filter(&bloom_path, memtable.bloom_filter())
            .await?;

        // Warm before commit (zero cold window); no-op without a warmer.
        let warm_uri = self.path_to_uri(&gen_path);
        self.warm_generation(&warm_uri).await;

        let new_manifest = self
            .update_manifest(
                epoch,
                generation,
                &gen_folder_name,
                covered_wal_entry_position,
                size,
            )
            .await?;

        info!(
            "Flushed SSTable {} for shard {} (manifest version {})",
            generation, self.shard_id, new_manifest.version
        );

        Ok(FlushResult {
            sstable: size.sstable(generation, gen_folder_name),
            rows_flushed: memtable.row_count(),
            covered_wal_entry_position,
        })
    }

    /// Build every index in `index_specs` the memtable holds into the flushed
    /// generation, one at a time.
    ///
    /// Returns index metadata without committing; the caller writes a single
    /// manifest with all of it.
    async fn create_indexes(
        &self,
        dataset: &mut Dataset,
        gen_path: &Path,
        index_specs: &[MemIndexSpec],
        mem_indexes: Option<&IndexStore>,
        total_rows: usize,
        storage_version: ConcreteFileVersion,
    ) -> Result<Vec<IndexMetadata>> {
        let Some(store) = mem_indexes else {
            return Ok(vec![]);
        };

        let mut created = Vec::new();
        for spec in index_specs {
            let Some(index) = store.get_index(&spec.name) else {
                debug!(index = %spec.name, "memtable does not hold this index; not flushed");
                continue;
            };
            let generation = GenerationWrite {
                path: gen_path,
                object_store: &self.object_store,
                dataset,
                total_rows,
                name: &spec.name,
                storage_version,
            };
            let training_data = match index
                .flush(&FlushContext {
                    batch_size: TRAINING_BATCH_SIZE,
                    generation: Some(&generation),
                })
                .await?
            {
                FlushOutcome::Skip => {
                    debug!(index = %spec.name, "index holds nothing to flush");
                    continue;
                }
                FlushOutcome::Wrote(index_meta) => {
                    // Queries find a generation's index by name and field.
                    if index_meta.name != spec.name || index_meta.fields != spec.field_ids {
                        return Err(Error::internal(format!(
                            "index '{}' on fields {:?} recorded its file as '{}' on fields {:?}",
                            spec.name, spec.field_ids, index_meta.name, index_meta.fields
                        )));
                    }
                    created.push(*index_meta);
                    continue;
                }
                FlushOutcome::TrainingData(stream) => Some(stream),
                FlushOutcome::BuildFromGeneration => None,
            };
            created.push(build_scalar_index(dataset, spec, training_data).await?);
        }
        Ok(created)
    }

    /// Write the standalone primary-key dedup index for this generation.
    ///
    /// Unlike user indexes, this is a **sidecar**: it is not registered in the
    /// manifest. The block-list opens it directly by path
    /// ([`pk_index_path`]) and probes it with `Equals`. Single-column primary
    /// keys index the typed value; composite keys index the order-preserving
    /// `Binary` encoded tuple (see [`super::super::index::encode_pk_tuple`]).
    /// Row positions line up 1:1 with the forward-written data file, so they are
    /// the SSTable row ids directly. No-op without a primary-key index.
    async fn create_pk_index(
        &self,
        gen_path: &Path,
        mem_indexes: Option<&IndexStore>,
    ) -> Result<()> {
        use datafusion::physical_plan::SendableRecordBatchStream;
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use lance_index::scalar::btree::train_btree_index;
        use lance_index::scalar::lance_format::LanceIndexStore;

        use crate::dataset::mem_wal::util::pk_index_path;

        let Some(registry) = mem_indexes else {
            return Ok(());
        };
        let batches = registry.pk_training_batches(TRAINING_BATCH_SIZE)?;
        if batches.is_empty() {
            return Ok(());
        }

        let schema = batches[0].schema();
        let store = LanceIndexStore::new(
            self.object_store.clone(),
            pk_index_path(gen_path),
            Arc::new(LanceCache::no_cache()),
        );
        let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches.into_iter().map(Ok)),
        ));
        train_btree_index(stream, &store, TRAINING_BATCH_SIZE as u64, None, None).await?;
        Ok(())
    }

    /// Update the shard manifest with the new SSTable.
    async fn update_manifest(
        &self,
        epoch: u64,
        generation: u64,
        gen_path: &str,
        covered_wal_entry_position: u64,
        size: FlushedSize,
    ) -> Result<ShardManifest> {
        let gen_path = gen_path.to_string();

        self.manifest_store
            .commit_update(epoch, |current| {
                let mut sstables = current.sstables.clone();
                sstables.push(size.sstable(generation, gen_path.clone()));

                ShardManifest {
                    version: current.next_version(),
                    replay_after_wal_entry_position: covered_wal_entry_position,
                    wal_entry_position_last_seen: current
                        .wal_entry_position_last_seen
                        .max(covered_wal_entry_position),
                    current_generation: generation + 1,
                    sstables,
                    ..current.clone()
                }
            })
            .await
    }
}

/// Message driving the background memtable-flush task.
pub enum TriggerMemTableFlush {
    /// Flush a frozen memtable to Lance storage.
    Flush {
        /// The frozen memtable to flush.
        memtable: Arc<MemTable>,
        /// The indexes the memtable was built with. Not read from the writer,
        /// whose set changes with the schema while this memtable keeps the old one.
        index_specs: Arc<[MemIndexSpec]>,
        /// Optional channel to notify when flush completes.
        done: Option<tokio::sync::oneshot::Sender<Result<FlushResult>>>,
    },
}

impl std::fmt::Debug for TriggerMemTableFlush {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flush {
                memtable,
                index_specs,
                done,
            } => f
                .debug_struct("TriggerMemTableFlush::Flush")
                .field("memtable_gen", &memtable.generation())
                .field("memtable_rows", &memtable.row_count())
                .field("index_count", &index_specs.len())
                .field("has_done", &done.is_some())
                .finish(),
        }
    }
}

/// Build a scalar index on the generation from `training_data`, or from its
/// rows without any.
async fn build_scalar_index(
    dataset: &mut Dataset,
    spec: &MemIndexSpec,
    training_data: Option<SendableRecordBatchStream>,
) -> Result<IndexMetadata> {
    let index_type = spec.plugin.flush_index_type();
    if !index_type.is_scalar() {
        return Err(Error::invalid_input(format!(
            "index '{}' asked the flush to build a {index_type} index, but the flush builds \
             only scalar types; plugin '{}' must write it and return FlushOutcome::Wrote",
            spec.name,
            spec.plugin.name()
        )));
    }
    let columns: Vec<&str> = spec.columns.iter().map(String::as_str).collect();
    let params = ScalarIndexParams::default();
    let mut builder =
        CreateIndexBuilder::new(dataset, &columns, index_type, &params).name(spec.name.clone());
    if let Some(stream) = training_data {
        // Memtable positions are the flushed file's row ids, so no remap.
        builder = builder.preprocessed_stream(stream, spec.plugin.training_criteria());
    }
    builder.execute_uncommitted().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::index::test_plugin::{Deviation, wrapped};
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_index::scalar::inverted::INVERTED_INDEX_VERSION_V2;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn create_local_store() -> (Arc<ObjectStore>, Path, String, TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        (store, path, uri, temp_dir)
    }

    /// A local store with one claimed shard and a flusher over it.
    ///
    /// Every flush test in this module repeats this setup.
    struct FlushFixture {
        manifest_store: Arc<ShardManifestStore>,
        flusher: MemTableFlusher,
        epoch: u64,
        _temp_dir: TempDir,
    }

    impl FlushFixture {
        async fn new() -> Self {
            let (store, base_path, base_uri, temp_dir) = create_local_store().await;
            let shard_id = Uuid::new_v4();
            let manifest_store = Arc::new(ShardManifestStore::new(
                store.clone(),
                &base_path,
                shard_id,
                2,
            ));
            let (epoch, _) = manifest_store.claim_epoch(0).await.unwrap();
            let flusher =
                MemTableFlusher::new(store, base_path, base_uri, shard_id, manifest_store.clone());
            Self {
                manifest_store,
                flusher,
                epoch,
                _temp_dir: temp_dir,
            }
        }

        /// Flush `memtable` and read the entry it wrote back **off storage**.
        ///
        /// Through `read_version`, not `latest`: the store caches the manifest
        /// it just wrote, so `latest` returns that same Rust value and would
        /// pass even for a field that never reached the protobuf.
        async fn flush_and_read_back(&self, memtable: &MemTable, durable: usize) -> SsTable {
            let result = self
                .flusher
                .flush(memtable, self.epoch, 1, durable)
                .await
                .unwrap();
            let version = self.manifest_store.latest().await.unwrap().unwrap().version;
            self.manifest_store
                .read_version(version)
                .await
                .unwrap()
                .sstables
                .into_iter()
                .find(|sstable| sstable.generation == result.sstable.generation)
                .expect("the flush recorded its generation")
        }
    }

    /// The shard schema these tests build their memtables against.
    fn test_lance_schema() -> lance_core::datatypes::Schema {
        lance_core::datatypes::Schema::try_from(create_test_schema().as_ref()).unwrap()
    }

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    /// Schema with `id` marked as the unenforced primary key, so the flush
    /// computes a within-generation deletion vector.
    fn create_pk_schema() -> Arc<ArrowSchema> {
        let mut id_metadata = std::collections::HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);
        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, num_rows: usize) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
                Arc::new(StringArray::from_iter_values(
                    (0..num_rows).map(|i| format!("name_{}", i)),
                )),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_flusher_requires_wal_flush() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));

        // Claim shard
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();
        memtable
            .insert(create_test_batch(&schema, 10))
            .await
            .unwrap();

        // Nothing is durable yet, so the L0 flush must refuse.
        let durable = 0;
        assert!(!memtable.all_flushed_to_wal(durable));

        let flusher = MemTableFlusher::new(store, base_path, base_uri, shard_id, manifest_store);
        let result = flusher.flush(&memtable, epoch, 0, 0).await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unflushed fragments")
        );
    }

    #[tokio::test]
    async fn test_flusher_empty_memtable() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));

        // Claim shard
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let schema = create_test_schema();
        let memtable = MemTable::new(schema, 1, vec![]).unwrap();

        let flusher = MemTableFlusher::new(store, base_path, base_uri, shard_id, manifest_store);
        let result = flusher.flush(&memtable, epoch, 0, 0).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty MemTable"));
    }

    #[tokio::test]
    async fn test_flusher_success() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));

        // Claim shard
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();
        let frag_id = memtable
            .insert(create_test_batch(&schema, 10))
            .await
            .unwrap();

        // Simulate WAL flush
        let durable = frag_id + 1;
        assert!(memtable.all_flushed_to_wal(durable));

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path,
            base_uri,
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher.flush(&memtable, epoch, 1, durable).await.unwrap();

        assert_eq!(result.sstable.generation, 1);
        assert_eq!(result.rows_flushed, 10);
        assert_eq!(result.covered_wal_entry_position, 1);

        // Verify manifest was updated
        let updated_manifest = manifest_store.latest().await.unwrap().unwrap();
        assert_eq!(updated_manifest.version, 2);
        assert_eq!(updated_manifest.replay_after_wal_entry_position, 1);
        assert_eq!(updated_manifest.current_generation, 2);
        assert_eq!(updated_manifest.sstables.len(), 1);
    }

    /// A flushed generation records what it holds, read back off storage.
    ///
    /// The read goes through the persisted protobuf on purpose: a consumer
    /// decides whether to open a generation from these numbers, so a field that
    /// never reached storage is the failure worth catching.
    #[tokio::test]
    async fn flushed_sstable_records_what_it_holds() {
        let fixture = FlushFixture::new().await;
        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();
        let rows = 10;
        let frag_id = memtable
            .insert(create_test_batch(&schema, rows))
            .await
            .unwrap();
        let accounted = memtable.batch_store().row_bytes() as u64;

        let entry = fixture.flush_and_read_back(&memtable, frag_id + 1).await;

        assert_eq!(entry.physical_rows, Some(rows as u64));
        assert_eq!(entry.in_memory_bytes, Some(accounted));
        // The MemTable's own accounting, so above the raw payload (4 bytes per
        // `id`) and not the encoded file size.
        let payload = rows as u64 * std::mem::size_of::<i32>() as u64;
        assert!(
            entry.in_memory_bytes.unwrap() > payload,
            "recorded {:?} should exceed the raw payload {payload}",
            entry.in_memory_bytes
        );
        // No primary key here: absent, not zero, which a consumer would read as
        // costing nothing.
        assert_eq!(entry.primary_key_bytes, None);
    }

    /// With a primary key, its size is recorded too -- the term neither of the
    /// others carries, since a narrow key over many rows and a wide key over
    /// few weigh the same in `in_memory_bytes`.
    #[tokio::test]
    async fn flushed_sstable_records_its_primary_key_size() {
        let fixture = FlushFixture::new().await;
        let schema = create_pk_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![0]).unwrap();
        let rows = 10;
        let frag_id = memtable
            .insert(create_test_batch(&schema, rows))
            .await
            .unwrap();

        let entry = fixture.flush_and_read_back(&memtable, frag_id + 1).await;

        // `id` is a non-nullable Int32, so the key columns hold at least four
        // bytes a row and cannot reach the whole MemTable's size.
        let key_bytes = entry.primary_key_bytes.expect("a keyed table records it");
        assert!(
            key_bytes >= rows as u64 * std::mem::size_of::<i32>() as u64,
            "recorded {key_bytes} is below the raw key payload"
        );
        assert!(
            key_bytes < entry.in_memory_bytes.unwrap(),
            "keys ({key_bytes}) cannot outweigh the whole MemTable ({:?})",
            entry.in_memory_bytes
        );
    }

    /// A `SsTableWarmer` that counts calls and optionally fails.
    #[derive(Debug)]
    struct CountingWarmer {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl SsTableWarmer for CountingWarmer {
        async fn warm(&self, _path: &str) -> Result<()> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.fail {
                Err(Error::io("simulated warm failure".to_string()))
            } else {
                Ok(())
            }
        }
    }

    /// Warming is a best-effort optimization, never a commit gate: a warmer that
    /// errors pre-commit must still let the flush commit the generation. The
    /// warm fires exactly once on the pre-commit path.
    #[tokio::test]
    async fn test_flusher_commits_when_warm_fails() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();
        let frag_id = memtable
            .insert(create_test_batch(&schema, 10))
            .await
            .unwrap();
        let durable = frag_id + 1;

        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let warmer: Arc<dyn SsTableWarmer> = Arc::new(CountingWarmer {
            calls: calls.clone(),
            fail: true,
        });

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path,
            base_uri,
            shard_id,
            manifest_store.clone(),
        )
        .with_warmer(Some(warmer));
        // Flush must succeed despite the warmer erroring.
        let result = flusher.flush(&memtable, epoch, 1, durable).await.unwrap();

        assert_eq!(result.sstable.generation, 1);
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "pre-commit warm fires exactly once"
        );
        let updated = manifest_store.latest().await.unwrap().unwrap();
        assert_eq!(
            updated.sstables.len(),
            1,
            "generation still committed after a failed warm"
        );
    }

    /// Flushing a generation with within-generation duplicate PKs writes a
    /// deletion vector so the SSTable dataset exposes newest-per-PK on scan.
    #[tokio::test]
    async fn test_flush_writes_dedup_deletion_vector() {
        use futures::TryStreamExt;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let schema = create_pk_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![0]).unwrap();
        // Append order (newest last): id=1 a->a2, id=2 b, id=3 c->c2.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 1, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "a2", "c2"])),
            ],
        )
        .unwrap();
        let frag_id = memtable.insert(batch).await.unwrap();
        let durable = frag_id + 1;

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path,
            base_uri.clone(),
            shard_id,
            manifest_store,
        );
        let result = flusher.flush(&memtable, epoch, 1, durable).await.unwrap();
        assert_eq!(result.rows_flushed, 5, "all physical rows are written");

        // Scanning the SSTable must honor the deletion vector and
        // return only the newest version of each PK.
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri.trim_end_matches('/'),
            shard_id,
            result.sstable.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        let batches: Vec<RecordBatch> = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let mut rows = std::collections::HashMap::new();
        for b in &batches {
            let ids = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let names = b
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..b.num_rows() {
                rows.insert(ids.value(i), names.value(i).to_string());
            }
        }

        assert_eq!(
            rows.len(),
            3,
            "deletion vector should leave newest-per-PK, got {:?}",
            rows
        );
        assert_eq!(rows.get(&1), Some(&"a2".to_string()));
        assert_eq!(rows.get(&2), Some(&"b".to_string()));
        assert_eq!(rows.get(&3), Some(&"c2".to_string()));
    }

    /// Flushing a memtable with a primary-key index writes a standalone sidecar
    /// BTree at `{gen}/_pk_index` that the block-list can reopen by path and
    /// probe by value — including for a within-gen-superseded PK (existence,
    /// not visibility).
    #[tokio::test]
    async fn sstable_pk_index_sidecar_is_probeable() {
        use lance_core::cache::LanceCache;
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::registry::IndexPluginRegistry;
        use lance_index::scalar::lance_format::LanceIndexStore;
        use lance_index::scalar::{SargableQuery, SearchResult};

        use super::super::super::index::IndexStore;
        use crate::dataset::mem_wal::util::pk_index_path;
        use datafusion::common::ScalarValue;

        let (store, base_path, _base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        // Primary-key index on `id`, no user indexes.
        let schema = create_pk_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![0]).unwrap();
        let mut registry = IndexStore::new();
        registry.enable_pk_index(&[("id".to_string(), 0)]);
        memtable.set_indexes(registry);

        // id=1 updated in-gen (a -> a2); id=2 unique.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 1])),
                Arc::new(StringArray::from(vec!["a", "b", "a2"])),
            ],
        )
        .unwrap();
        let frag_id = memtable.insert(batch).await.unwrap();
        let durable = frag_id + 1;

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            _base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &[], 1, durable)
            .await
            .unwrap();

        // Reopen the sidecar directly by path (the block-list's route).
        let gen_path = base_path
            .clone()
            .join("_mem_wal")
            .join(shard_id.to_string())
            .join(result.sstable.path.as_str());
        let index_store = Arc::new(LanceIndexStore::new(
            store.clone(),
            pk_index_path(&gen_path),
            Arc::new(LanceCache::no_cache()),
        ));
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("BTree").unwrap();
        let details =
            prost_types::Any::from_msg(&lance_index::pbold::BTreeIndexDetails::default()).unwrap();
        // The sidecar is always freshly built by `to_training_batches`, never a
        // legacy segment, so it's always at the current (row-address-domain) version.
        let index = plugin
            .load_index(
                index_store,
                &details,
                lance_index::scalar::btree::BTREE_INDEX_VERSION,
                None,
                &LanceCache::no_cache(),
            )
            .await
            .unwrap();

        let contains = |id: i32| {
            let index = index.clone();
            async move {
                let result = index
                    .search(
                        &SargableQuery::Equals(ScalarValue::Int32(Some(id))),
                        &NoOpMetricsCollector,
                    )
                    .await
                    .unwrap();
                match result {
                    SearchResult::Exact(s) | SearchResult::AtMost(s) | SearchResult::AtLeast(s) => {
                        !s.is_empty()
                    }
                }
            }
        };
        // Both PKs present (id=1 even though its first version was superseded);
        // an absent PK is not.
        assert!(contains(1).await);
        assert!(contains(2).await);
        assert!(!contains(99).await);
    }

    /// Regression: production dispatches a PK-only flush (a primary key, no
    /// secondary index) to `flush`, not `flush_with_indexes`. `flush` must still
    /// write the PK dedup sidecar, otherwise cross-generation dedup fails with
    /// `page_lookup.lance not found`.
    #[tokio::test]
    async fn plain_flush_writes_pk_sidecar() {
        use lance_core::cache::LanceCache;
        use lance_index::metrics::NoOpMetricsCollector;
        use lance_index::registry::IndexPluginRegistry;
        use lance_index::scalar::lance_format::LanceIndexStore;
        use lance_index::scalar::{SargableQuery, SearchResult};

        use super::super::super::index::IndexStore;
        use crate::dataset::mem_wal::util::pk_index_path;
        use datafusion::common::ScalarValue;

        let (store, base_path, _base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        // Primary-key index on `id`, no user indexes.
        let schema = create_pk_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![0]).unwrap();
        let mut registry = IndexStore::new();
        registry.enable_pk_index(&[("id".to_string(), 0)]);
        memtable.set_indexes(registry);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let frag_id = memtable.insert(batch).await.unwrap();
        let durable = frag_id + 1;

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            _base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        // The plain-flush path — what the writer dispatches to with no indexes.
        let result = flusher.flush(&memtable, epoch, 1, durable).await.unwrap();

        let gen_path = base_path
            .clone()
            .join("_mem_wal")
            .join(shard_id.to_string())
            .join(result.sstable.path.as_str());
        let index_store = Arc::new(LanceIndexStore::new(
            store.clone(),
            pk_index_path(&gen_path),
            Arc::new(LanceCache::no_cache()),
        ));
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("BTree").unwrap();
        let details =
            prost_types::Any::from_msg(&lance_index::pbold::BTreeIndexDetails::default()).unwrap();
        // The sidecar is always freshly built by `to_training_batches`, never a
        // legacy segment, so it's always at the current (row-address-domain) version.
        let index = plugin
            .load_index(
                index_store,
                &details,
                lance_index::scalar::btree::BTREE_INDEX_VERSION,
                None,
                &LanceCache::no_cache(),
            )
            .await
            .unwrap();

        let contains = |id: i32| {
            let index = index.clone();
            async move {
                let result = index
                    .search(
                        &SargableQuery::Equals(ScalarValue::Int32(Some(id))),
                        &NoOpMetricsCollector,
                    )
                    .await
                    .unwrap();
                match result {
                    SearchResult::Exact(s) | SearchResult::AtMost(s) | SearchResult::AtLeast(s) => {
                        !s.is_empty()
                    }
                }
            }
        };
        assert!(contains(1).await);
        assert!(contains(2).await);
        assert!(!contains(99).await);
    }

    /// Covers `finalize_generation` writing both a deletion vector *and*
    /// indexes into the same manifest — the deletion-only and index-only
    /// paths are exercised by sibling tests.
    #[tokio::test]
    async fn test_flush_with_indexes_and_dedup_deletion_vector() {
        use super::super::super::index::IndexStore;
        use crate::index::DatasetIndexExt;
        use futures::TryStreamExt;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        // BTree on the non-PK `name` column so the index sees the dedup set.
        let index_specs = vec![MemIndexSpec::btree("name_btree", 1, "name")];

        let schema = create_pk_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![0]).unwrap();
        let registry =
            IndexStore::from_specs(&index_specs, &test_lance_schema(), 100_000, 1_000).unwrap();
        memtable.set_indexes(registry);

        // Duplicate PKs in append order: id=1 a->a2, id=2 b, id=3 c->c2.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 1, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "a2", "c2"])),
            ],
        )
        .unwrap();
        let frag_id = memtable.insert(batch).await.unwrap();
        let durable = frag_id + 1;

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_specs, 1, durable)
            .await
            .unwrap();
        assert_eq!(result.rows_flushed, 5, "all physical rows are written");

        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri.trim_end_matches('/'),
            shard_id,
            result.sstable.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        assert_eq!(
            dataset.version().version,
            1,
            "SSTable dataset must be a single-version dataset"
        );

        // Index half of the combined manifest.
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "name_btree");

        // Deletion-vector half: scan returns newest-per-PK.
        let batches: Vec<RecordBatch> = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let mut rows = std::collections::HashMap::new();
        for b in &batches {
            let ids = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let names = b
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..b.num_rows() {
                rows.insert(ids.value(i), names.value(i).to_string());
            }
        }
        assert_eq!(
            rows.len(),
            3,
            "deletion vector should leave newest-per-PK, got {:?}",
            rows
        );
        assert_eq!(rows.get(&1), Some(&"a2".to_string()));
        assert_eq!(rows.get(&2), Some(&"b".to_string()));
        assert_eq!(rows.get(&3), Some(&"c2".to_string()));

        // The BTree on `name` must not surface a stale value: a hit for the
        // pre-update "a" would mean the indexed path ignored the deletion
        // vector.
        let stale_hits = dataset
            .scan()
            .filter("name = 'a'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            stale_hits.num_rows(),
            0,
            "older name 'a' for id=1 must be filtered out by the deletion vector"
        );
        let fresh_hits = dataset
            .scan()
            .filter("name = 'a2'")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(fresh_hits.num_rows(), 1);
    }

    #[tokio::test]
    async fn test_flusher_with_btree_index() {
        use super::super::super::index::IndexStore;
        use crate::index::DatasetIndexExt;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));

        // Claim shard
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        // Create index config for the 'id' column (field_id = 0)
        let index_specs = vec![MemIndexSpec::btree("id_btree", 0, "id")];

        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();

        // Set up in-memory index registry so preprocessed data path is used
        let registry =
            IndexStore::from_specs(&index_specs, &test_lance_schema(), 100_000, 1_000).unwrap();
        memtable.set_indexes(registry);

        let frag_id = memtable
            .insert(create_test_batch(&schema, 10))
            .await
            .unwrap();

        // Simulate WAL flush
        let durable = frag_id + 1;

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_specs, 1, durable)
            .await
            .unwrap();

        assert_eq!(result.sstable.generation, 1);
        assert_eq!(result.rows_flushed, 10);

        // Verify the SSTable dataset is a single-version dataset with the BTree index
        let gen_uri = format!("{}/_mem_wal/{}/{}", base_uri, shard_id, result.sstable.path);
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        assert_eq!(
            dataset.version().version,
            1,
            "SSTable dataset must be a single-version dataset"
        );
        let indices = dataset.load_indices().await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "id_btree");

        // Verify query results are correct
        // The test data has ids 0-9, so querying for id = 5 should return 1 row
        let batch = dataset
            .scan()
            .filter("id = 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let id_col = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(id_col.value(0), 5);

        // Verify the query plan uses the BTree index
        let mut scan = dataset.scan();
        scan.filter("id = 5").unwrap();
        scan.prefilter(true);
        let plan = scan.create_plan().await.unwrap();
        crate::utils::test::assert_plan_node_equals(
            plan,
            "LanceRead: ...full_filter=id = Int32(5)...
  ScalarIndexQuery: query=[id = 5]@id_btree(BTree)",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_flusher_with_hnsw_index() {
        use super::super::super::index::IndexStore;
        use crate::index::DatasetIndexExt;
        use arrow_array::{FixedSizeListArray, Float32Array};
        use lance_linalg::distance::DistanceType;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));

        // Claim shard
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        let vector_dim = 8;
        let num_vectors = 300;

        let vector_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    vector_dim as i32,
                ),
                false,
            ),
        ]));

        // Generate random-ish vectors.
        let vectors: Vec<f32> = (0..num_vectors * vector_dim)
            .map(|i| ((i as f32 * 0.1).sin() + (i as f32 * 0.05).cos()) * 0.5)
            .collect();
        let vectors_array = Float32Array::from(vectors);

        // Create HNSW index config (field_id = 1 for vector column)
        let index_specs = vec![MemIndexSpec::hnsw(
            "vector_hnsw",
            1,
            "vector",
            DistanceType::L2,
        )];

        let mut memtable = MemTable::new(vector_schema.clone(), 1, vec![]).unwrap();
        let registry =
            IndexStore::from_specs(&index_specs, &test_lance_schema(), num_vectors, 100).unwrap();
        memtable.set_indexes(registry);

        // Create test batch with vectors
        let ids = Int32Array::from_iter_values(0..num_vectors as i32);
        // Use the field from the schema to ensure nullability matches
        let inner_field = Arc::new(Field::new("item", DataType::Float32, false));
        let vectors_fsl_data = FixedSizeListArray::try_new(
            inner_field,
            vector_dim as i32,
            Arc::new(vectors_array),
            None,
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            vector_schema.clone(),
            vec![Arc::new(ids), Arc::new(vectors_fsl_data)],
        )
        .unwrap();

        let frag_id = memtable.insert(batch).await.unwrap();

        // Simulate WAL flush
        let durable = frag_id + 1;

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_specs, 1, durable)
            .await
            .unwrap();

        assert_eq!(result.sstable.generation, 1);
        assert_eq!(result.rows_flushed, num_vectors);

        // Verify the SSTable dataset is a single-version dataset with the HNSW index
        let gen_uri = format!("{}/_mem_wal/{}/{}", base_uri, shard_id, result.sstable.path);
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        assert_eq!(
            dataset.version().version,
            1,
            "SSTable dataset must be a single-version dataset"
        );
        let indices = dataset.load_indices().await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "vector_hnsw");

        // End-to-end query: pick a row from the SSTable dataset, query for
        // it, and verify the index path returns it as the nearest neighbor.
        // This exercises the on-disk HNSW + SQ8 format including the IVF
        // partition routing and the storage_metadata ScalarQuantizationMetadata
        // deserialization.
        let scanned: Vec<RecordBatch> = {
            use futures::TryStreamExt;
            dataset
                .scan()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap()
        };
        let total_scanned: usize = scanned.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_scanned, num_vectors);

        // Query with the first vector in the dataset; it must come back as
        // the nearest neighbor with distance ~0.
        let first_vec_values: Vec<f32> = (0..vector_dim)
            .map(|i| ((i as f32 * 0.1).sin() + (i as f32 * 0.05).cos()) * 0.5)
            .collect();
        let query = Float32Array::from(first_vec_values);
        let mut scan = dataset.scan();
        scan.nearest("vector", &query, 5).unwrap();
        scan.fast_search();
        let batch = scan.try_into_batch().await.unwrap();
        assert!(batch.num_rows() > 0, "query returned no rows");
        let dist_col = batch
            .column_by_name("_distance")
            .expect("_distance column missing")
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!(
            dist_col.value(0) < 1e-3,
            "expected near-zero distance for self-match, got {}",
            dist_col.value(0)
        );

        // Verify the query plan uses the HNSW vector index
        let mut scan = dataset.scan();
        scan.nearest("vector", &query, 5).unwrap();
        scan.fast_search();
        let plan = scan.create_plan().await.unwrap();
        let plan_str = format!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
        assert!(
            plan_str.contains("ANNSubIndex: name=vector_hnsw, k=5"),
            "query plan must use HNSW index, got: {plan_str}"
        );
        assert!(
            plan_str.contains("ANNIvfPartition:"),
            "query plan must use IVF partition, got: {plan_str}"
        );
    }

    #[tokio::test]
    async fn test_flusher_with_fts_index() {
        use super::super::super::index::IndexStore;
        use crate::index::DatasetIndexExt;
        use arrow_array::StringArray;
        use arrow_schema::{DataType, Field, Schema as ArrowSchema};
        use std::sync::Arc;

        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));

        // Claim shard
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();

        // Create schema with text column
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]));

        // Create FTS index config (field_id = 1 for text column)
        let index_specs = vec![MemIndexSpec::fts("text_fts", 1, "text")];

        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();

        // Set up in-memory index registry
        let registry =
            IndexStore::from_specs(&index_specs, &test_lance_schema(), 100_000, 1_000).unwrap();
        memtable.set_indexes(registry);

        // Create test batch with text data
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    "hello world",
                    "quick brown fox",
                    "lazy dog jumps",
                ])),
            ],
        )
        .unwrap();

        let frag_id = memtable.insert(batch).await.unwrap();

        // Simulate WAL flush
        let durable = frag_id + 1;

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_specs, 1, durable)
            .await
            .unwrap();

        assert_eq!(result.sstable.generation, 1);
        assert_eq!(result.rows_flushed, 3);

        // Verify the SSTable dataset is a single-version dataset with the FTS index
        let gen_uri = format!("{}/_mem_wal/{}/{}", base_uri, shard_id, result.sstable.path);
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        assert_eq!(
            dataset.version().version,
            1,
            "SSTable dataset must be a single-version dataset"
        );
        let indices = dataset.load_indices().await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "text_fts");
        assert_eq!(indices[0].index_version, INVERTED_INDEX_VERSION_V2 as i32);

        // Verify FTS query returns correct results
        // Searching for "hello" should find the first document
        use lance_index::scalar::FullTextSearchQuery;
        let batch = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let id_col = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(
            id_col.value(0),
            1,
            "Should find document with 'hello world'"
        );

        // Searching for "fox" should find the second document
        let batch = dataset
            .scan()
            .full_text_search(FullTextSearchQuery::new("fox".to_owned()))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let id_col = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(
            id_col.value(0),
            2,
            "Should find document with 'quick brown fox'"
        );

        // Verify the query plan uses the FTS index
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new("hello".to_owned()))
            .unwrap();
        let plan = scan.create_plan().await.unwrap();
        crate::utils::test::assert_plan_node_equals(
            plan,
            "ProjectionExec: expr=[id@2 as id, text@3 as text, _score@1 as _score]
  LanceRead: ..., source=stream(_rowid)
    MatchQuery: column=text, query=[hello]",
        )
        .await
        .unwrap();
    }

    /// Flush one memtable holding `batch`, maintaining `spec`, and return the
    /// indexes the generation records.
    async fn flush_one(spec: MemIndexSpec, batch: RecordBatch) -> Result<Vec<IndexMetadata>> {
        use crate::index::DatasetIndexExt;

        let (generation, _dir) = flush_generation(spec, batch).await?;
        Ok(generation.load_indices().await?.as_ref().clone())
    }

    /// The generation written by flushing `batch` with the index `spec`, and the
    /// directory holding it.
    async fn flush_generation(
        spec: MemIndexSpec,
        batch: RecordBatch,
    ) -> Result<(Dataset, TempDir)> {
        use crate::dataset::mem_wal::index::IndexStore;

        let (store, base_path, base_uri, temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = Arc::new(ShardManifestStore::new(
            store.clone(),
            &base_path,
            shard_id,
            2,
        ));
        let (epoch, _manifest) = manifest_store.claim_epoch(0).await.unwrap();
        let specs = vec![spec];
        let lance_schema =
            lance_core::datatypes::Schema::try_from(batch.schema().as_ref()).unwrap();
        let mut memtable = MemTable::new(batch.schema(), 1, vec![]).unwrap();
        memtable.set_indexes(IndexStore::from_specs(&specs, &lance_schema, 1000, 16).unwrap());
        let durable = memtable.insert(batch).await.unwrap() + 1;
        let result =
            MemTableFlusher::new(store, base_path, base_uri.clone(), shard_id, manifest_store)
                .flush_with_indexes(&memtable, epoch, &specs, 1, durable)
                .await?;
        let generation = Dataset::open(&format!(
            "{}/_mem_wal/{}/{}",
            base_uri.trim_end_matches('/'),
            shard_id,
            result.sstable.path
        ))
        .await?;
        Ok((generation, temp_dir))
    }

    /// An index holding nothing writes no index: an empty list-element
    /// full-text index is not rebuilt as a whole-row one.
    #[tokio::test]
    async fn an_empty_index_is_left_out_of_the_generation() {
        use arrow_array::Array;
        use arrow_array::builder::{ListBuilder, StringBuilder};
        use lance_index::scalar::InvertedIndexParams;
        use lance_index::scalar::inverted::DocumentGranularity;

        let mut tags = ListBuilder::new(StringBuilder::new());
        tags.append(false);
        tags.append(true);
        let tags = tags.finish();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("tags", tags.data_type().clone(), true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(tags)],
        )
        .unwrap();
        let spec = MemIndexSpec::fts_with_params(
            "tags_fts",
            1,
            "tags",
            InvertedIndexParams::default().document_granularity(DocumentGranularity::ListElement),
        );
        assert!(flush_one(spec, batch).await.unwrap().is_empty());
    }

    /// The flush builds only scalar indexes; a vector plugin asking it to is
    /// refused by name.
    #[tokio::test]
    async fn a_vector_plugin_asking_the_flush_to_build_it_is_an_error() {
        use arrow_array::{FixedSizeListArray, Float32Array};
        use lance_linalg::distance::DistanceType;

        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("vector", DataType::FixedSizeList(item.clone(), 2), false),
        ]));
        let vectors = FixedSizeListArray::try_new(
            item,
            2,
            Arc::new(Float32Array::from_iter_values((0..20).map(|i| i as f32))),
            None,
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from((0..10).collect::<Vec<_>>())),
                Arc::new(vectors),
            ],
        )
        .unwrap();
        let spec = wrapped(
            MemIndexSpec::hnsw("vector_hnsw", 1, "vector", DistanceType::L2),
            Deviation::AsksForABuild,
        );
        let error = flush_one(spec, batch).await.unwrap_err();
        assert!(
            error.to_string().contains("vector_hnsw") && error.to_string().contains("Wrote"),
            "{error}"
        );
    }

    fn text_batch(texts: &[&str]) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(0..texts.len() as i32)),
                Arc::new(StringArray::from(texts.to_vec())),
            ],
        )
        .unwrap()
    }

    /// A failed index flush fails the generation, and an index recorded under
    /// a name or fields other than its own is refused.
    #[rstest::rstest]
    #[case::fails(Deviation::FailsToFlush, "flush failed")]
    #[case::another_name(Deviation::RecordsAnotherName, "another_name")]
    #[tokio::test]
    async fn an_index_that_does_not_flush_as_itself_fails_the_flush(
        #[case] deviation: Deviation,
        #[case] message: &str,
    ) {
        let spec = wrapped(MemIndexSpec::fts("text_fts", 1, "text"), deviation);
        let error = flush_one(spec, text_batch(&["hello world"]))
            .await
            .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }

    /// A flushed full-text index with word positions answers a phrase query.
    #[tokio::test]
    async fn a_full_text_index_with_positions_answers_phrases_once_flushed() {
        use lance_index::scalar::FullTextSearchQuery;
        use lance_index::scalar::InvertedIndexParams;
        use lance_index::scalar::inverted::query::{FtsQuery, PhraseQuery};

        let spec = MemIndexSpec::fts_with_params(
            "text_fts",
            1,
            "text",
            InvertedIndexParams::default().with_position(true),
        );
        let (generation, _dir) =
            flush_generation(spec, text_batch(&["quick brown fox", "brown quick"]))
                .await
                .unwrap();
        let found = generation
            .scan()
            .full_text_search(FullTextSearchQuery::new_query(FtsQuery::Phrase(
                PhraseQuery::new("quick brown".to_string()).with_column(Some("text".to_string())),
            )))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = found
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(ids, vec![0]);
    }
}
