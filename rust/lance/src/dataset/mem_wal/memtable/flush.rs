// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MemTable flush to persistent storage.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use bytes::Bytes;
use lance_core::cache::LanceCache;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use lance_index::IndexType;
use lance_index::mem_wal::{FlushedGeneration, ShardManifest};
use lance_index::scalar::{IndexStore, ScalarIndexParams};
use lance_io::object_store::ObjectStore;
use lance_table::format::IndexMetadata;
use lance_table::io::commit::write_manifest_file_to_path;
use lance_table::io::deletion::write_deletion_file;
use log::info;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use roaring::RoaringBitmap;
use tracing::instrument;
use uuid::Uuid;

use super::super::index::MemIndexConfig;
use super::super::memtable::MemTable;
use crate::Dataset;
use crate::dataset::mem_wal::manifest::ShardManifestStore;
use crate::dataset::mem_wal::scanner::exec::{compute_pk_hash, validate_pk_types};
use crate::dataset::mem_wal::util::{flushed_memtable_path, generate_random_hash};

#[derive(Debug, Clone)]
pub struct FlushResult {
    pub generation: FlushedGeneration,
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
}

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
    ) -> Result<FlushResult> {
        self.manifest_store.check_fenced(epoch).await?;

        if memtable.row_count() == 0 {
            return Err(Error::invalid_input("Cannot flush empty MemTable"));
        }

        if !memtable.all_flushed_to_wal() {
            return Err(Error::invalid_input(
                "MemTable has unflushed fragments - WAL flush required first",
            ));
        }

        let random_hash = generate_random_hash();
        let generation = memtable.generation();
        let gen_folder_name = format!("{}_gen_{}", random_hash, generation);
        let gen_path =
            flushed_memtable_path(&self.base_path, &self.shard_id, &random_hash, generation);

        info!(
            "Flushing MemTable generation {} to {} ({} rows, {} batches)",
            generation,
            gen_path,
            memtable.row_count(),
            memtable.batch_count()
        );

        let (rows_flushed, deleted) = self.write_data_file(&gen_path, memtable).await?;

        // Persist the within-generation deletion vector so the flushed
        // generation exposes newest-per-PK on every read path.
        if !deleted.is_empty() {
            let uri = self.path_to_uri(&gen_path);
            let dataset = Dataset::open(&uri).await?;
            self.finalize_generation(&dataset, &deleted, None).await?;
        }

        let bloom_path = gen_path.clone().join("bloom_filter.bin");
        self.write_bloom_filter(&bloom_path, memtable.bloom_filter())
            .await?;

        let new_manifest = self
            .update_manifest(
                epoch,
                generation,
                &gen_folder_name,
                covered_wal_entry_position,
            )
            .await?;

        info!(
            "Flushed generation {} for shard {} (manifest version {})",
            generation, self.shard_id, new_manifest.version
        );

        Ok(FlushResult {
            generation: FlushedGeneration {
                generation,
                path: gen_folder_name,
            },
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
        let reader =
            RecordBatchIterator::new(batches.into_iter().map(Ok), memtable.schema().clone());

        // Use very large max_rows_per_file to ensure 1 fragment per flushed memtable
        let write_params = WriteParams {
            max_rows_per_file: usize::MAX,
            ..Default::default()
        };
        Dataset::write(reader, &uri, Some(write_params)).await?;

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
                0, // 1 fragment per flushed generation
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
    #[instrument(name = "mt_flush_with_indexes", level = "info", skip_all, fields(shard_id = %self.shard_id, epoch, generation = memtable.generation(), row_count = memtable.row_count(), index_count = index_configs.len()))]
    pub async fn flush_with_indexes(
        &self,
        memtable: &MemTable,
        epoch: u64,
        index_configs: &[MemIndexConfig],
        covered_wal_entry_position: u64,
    ) -> Result<FlushResult> {
        self.manifest_store.check_fenced(epoch).await?;

        if memtable.row_count() == 0 {
            return Err(Error::invalid_input("Cannot flush empty MemTable"));
        }

        if !memtable.all_flushed_to_wal() {
            return Err(Error::invalid_input(
                "MemTable has unflushed fragments - WAL flush required first",
            ));
        }

        let random_hash = generate_random_hash();
        let generation = memtable.generation();
        let gen_folder_name = format!("{}_gen_{}", random_hash, generation);
        let gen_path =
            flushed_memtable_path(&self.base_path, &self.shard_id, &random_hash, generation);

        info!(
            "Flushing MemTable generation {} with indexes to {} ({} rows, {} batches)",
            generation,
            gen_path,
            memtable.row_count(),
            memtable.batch_count()
        );

        let (total_rows, deleted) = self.write_data_file(&gen_path, memtable).await?;

        // Open the dataset once for all index building. Dataset::write already
        // created a v1 manifest with the fragment data.
        let uri = self.path_to_uri(&gen_path);
        let mut dataset = Dataset::open(&uri).await?;

        // Collect all index metadata without committing individually.
        // We write a single manifest containing both data and all indexes.
        let mut all_indexes: Vec<IndexMetadata> = Vec::new();

        let btree_indexes = self
            .create_indexes(&mut dataset, index_configs, memtable.indexes())
            .await?;
        if !btree_indexes.is_empty() {
            info!(
                "Created {} BTree indexes on flushed generation {}",
                btree_indexes.len(),
                generation
            );
        }
        all_indexes.extend(btree_indexes);

        if let Some(registry) = memtable.indexes() {
            for config in index_configs {
                if let MemIndexConfig::Hnsw(hnsw_config) = config
                    && let Some(mem_index) = registry.get_hnsw(&hnsw_config.name)
                {
                    let mut index_meta = self
                        .create_hnsw_index(&gen_path, hnsw_config, mem_index)
                        .await?;

                    let schema = dataset.schema();
                    let field_idx = schema
                        .field(&hnsw_config.column)
                        .map(|f| f.id)
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "HNSW index '{}' references column '{}' which is not in the dataset schema",
                                hnsw_config.name, hnsw_config.column
                            ))
                        })?;
                    index_meta.fields = vec![field_idx];
                    index_meta.dataset_version = dataset.version().version;
                    let fragment_ids: roaring::RoaringBitmap =
                        dataset.fragment_bitmap.as_ref().clone();
                    index_meta.fragment_bitmap = Some(fragment_ids);
                    all_indexes.push(index_meta);

                    info!(
                        "Created HNSW index '{}' on flushed generation {}",
                        hnsw_config.name, generation
                    );
                }
            }

            let fts_indexes = self
                .create_fts_indexes(
                    &dataset,
                    &gen_path,
                    index_configs,
                    memtable.indexes(),
                    total_rows,
                )
                .await?;
            all_indexes.extend(fts_indexes);
        }

        // Write a single manifest that records the fragments, the
        // within-generation deletion vector, and all indexes, overwriting the
        // data-only v1 manifest created by Dataset::write.
        self.finalize_generation(&dataset, &deleted, Some(all_indexes))
            .await?;

        let bloom_path = gen_path.clone().join("bloom_filter.bin");
        self.write_bloom_filter(&bloom_path, memtable.bloom_filter())
            .await?;

        let new_manifest = self
            .update_manifest(
                epoch,
                generation,
                &gen_folder_name,
                covered_wal_entry_position,
            )
            .await?;

        info!(
            "Flushed generation {} for shard {} (manifest version {})",
            generation, self.shard_id, new_manifest.version
        );

        Ok(FlushResult {
            generation: FlushedGeneration {
                generation,
                path: gen_folder_name,
            },
            rows_flushed: memtable.row_count(),
            covered_wal_entry_position,
        })
    }

    /// Create BTree indexes on the flushed dataset (uncommitted).
    ///
    /// Returns index metadata without committing to the dataset manifest.
    /// The caller is responsible for writing a single manifest with all indexes.
    async fn create_indexes(
        &self,
        dataset: &mut Dataset,
        index_configs: &[MemIndexConfig],
        mem_indexes: Option<&super::super::index::IndexStore>,
    ) -> Result<Vec<IndexMetadata>> {
        use arrow_array::RecordBatchIterator;

        use crate::index::CreateIndexBuilder;

        let btree_configs: Vec<_> = index_configs
            .iter()
            .filter_map(|c| match c {
                MemIndexConfig::BTree(cfg) => Some(cfg),
                MemIndexConfig::Hnsw(_) => None,
                MemIndexConfig::Fts(_) => None,
            })
            .collect();

        if btree_configs.is_empty() {
            return Ok(vec![]);
        }

        let mut created_indexes = Vec::new();

        for btree_cfg in btree_configs {
            let params = ScalarIndexParams::default();
            let mut builder = CreateIndexBuilder::new(
                dataset,
                &[btree_cfg.column.as_str()],
                IndexType::BTree,
                &params,
            )
            .name(btree_cfg.name.clone());

            if let Some(registry) = mem_indexes
                && let Some(btree_index) = registry.get_btree(&btree_cfg.name)
            {
                // Forward-written data: index row positions line up 1:1 with
                // the data file, no remap needed.
                let training_batches = btree_index.to_training_batches(8192)?;
                if !training_batches.is_empty() {
                    let schema = training_batches[0].schema();
                    let reader =
                        RecordBatchIterator::new(training_batches.into_iter().map(Ok), schema);
                    builder = builder.preprocessed_data(Box::new(reader));
                }
            }

            let index_meta = builder.execute_uncommitted().await?;
            created_indexes.push(index_meta);
        }

        Ok(created_indexes)
    }

    /// Create FTS (Full-Text Search) indexes from in-memory data (uncommitted).
    ///
    /// Writes the FTS index files and returns index metadata without committing.
    /// The caller is responsible for writing a single manifest with all indexes.
    async fn create_fts_indexes(
        &self,
        dataset: &Dataset,
        gen_path: &Path,
        index_configs: &[MemIndexConfig],
        mem_indexes: Option<&super::super::index::IndexStore>,
        total_rows: usize,
    ) -> Result<Vec<IndexMetadata>> {
        use lance_index::pbold;
        use lance_index::scalar::inverted::current_fts_format_version;
        use lance_index::scalar::lance_format::LanceIndexStore;

        let fts_configs: Vec<_> = index_configs
            .iter()
            .filter_map(|c| match c {
                MemIndexConfig::Fts(cfg) => Some(cfg),
                _ => None,
            })
            .collect();

        if fts_configs.is_empty() {
            return Ok(vec![]);
        }

        let Some(registry) = mem_indexes else {
            return Ok(vec![]);
        };

        let mut created_indexes = Vec::new();

        for fts_cfg in fts_configs {
            let Some(fts_index) = registry.get_fts(&fts_cfg.name) else {
                continue;
            };

            if fts_index.is_empty() {
                continue;
            }

            let partition_id = uuid::Uuid::new_v4().as_u64_pair().0;

            let mut inner_builder = fts_index.to_index_builder(partition_id, total_rows)?;

            let index_uuid = uuid::Uuid::new_v4();
            let index_dir = gen_path
                .clone()
                .join("_indices")
                .join(index_uuid.to_string());
            let index_store = LanceIndexStore::new(
                self.object_store.clone(),
                index_dir.clone(),
                Arc::new(LanceCache::no_cache()),
            );

            inner_builder.write(&index_store).await?;

            self.write_fts_metadata(&index_store, partition_id, fts_cfg)
                .await?;

            let details = pbold::InvertedIndexDetails::try_from(&fts_cfg.params)?;
            let index_details = prost_types::Any::from_msg(&details)
                .map_err(|e| Error::io(format!("Failed to serialize index details: {}", e)))?;

            let schema = dataset.schema();
            let field_idx = schema.field(&fts_cfg.column).map(|f| f.id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "FTS index '{}' references column '{}' which is not in the dataset schema",
                    fts_cfg.name, fts_cfg.column
                ))
            })?;

            let fragment_ids: roaring::RoaringBitmap = dataset.fragment_bitmap.as_ref().clone();

            let index_meta = IndexMetadata {
                uuid: index_uuid,
                name: fts_cfg.name.clone(),
                fields: vec![field_idx],
                dataset_version: dataset.version().version,
                fragment_bitmap: Some(fragment_ids),
                index_details: Some(Arc::new(index_details)),
                index_version: current_fts_format_version().index_version() as i32,
                created_at: None,
                base_id: None,
                files: None,
            };
            created_indexes.push(index_meta);

            info!(
                "Created FTS index '{}' on column '{}' (direct flush)",
                fts_cfg.name, fts_cfg.column
            );
        }

        Ok(created_indexes)
    }

    /// Write FTS index metadata file.
    async fn write_fts_metadata(
        &self,
        index_store: &lance_index::scalar::lance_format::LanceIndexStore,
        partition_id: u64,
        config: &super::super::index::FtsIndexConfig,
    ) -> Result<()> {
        use arrow_array::{RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        use lance_index::scalar::inverted::TokenSetFormat;

        // Create metadata with params and partitions in schema metadata (this is what InvertedIndex expects)
        let params_json = serde_json::to_string(&config.params)?;
        let partitions_json = serde_json::to_string(&[partition_id])?;
        let token_set_format = TokenSetFormat::default().to_string();

        let schema = Arc::new(
            Schema::new(vec![Field::new("_placeholder", DataType::Utf8, true)]).with_metadata(
                [
                    ("params".to_string(), params_json),
                    ("partitions".to_string(), partitions_json),
                    ("token_set_format".to_string(), token_set_format),
                ]
                .into(),
            ),
        );

        // Create a minimal batch (schema metadata is what matters)
        let placeholder_array = Arc::new(StringArray::from(vec![None::<&str>]));
        let batch = RecordBatch::try_new(schema.clone(), vec![placeholder_array])?;

        let mut writer = index_store.new_index_file("metadata.lance", schema).await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;

        Ok(())
    }

    /// Create an HNSW + SQ8 index from the in-memory HNSW.
    ///
    /// Writes:
    /// - `auxiliary.idx`: SQ8-quantized vector storage (`_rowid`, `__sq_code`).
    ///   Bounds learned from the full memtable in one pass.
    /// - `index.idx`: HNSW graph (`__vector_id`, `__neighbors`, `_distance`).
    ///
    /// Both files use a single placeholder IVF partition so they conform to
    /// the existing Lance `IVF_HNSW_SQ` reader path.
    ///
    /// # Arguments
    /// * `gen_path` - Path to the flushed generation folder
    /// * `config` - HNSW index configuration
    /// * `mem_index` - In-memory HNSW index (snapshotted, not consumed)
    async fn create_hnsw_index(
        &self,
        gen_path: &Path,
        config: &super::super::index::HnswIndexConfig,
        mem_index: &super::super::index::HnswMemIndex,
    ) -> Result<IndexMetadata> {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Float32Type;
        use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch as ArrowRecordBatch};
        use arrow_schema::Schema as ArrowSchema;
        use lance_arrow::FixedSizeListArrayExt;
        use lance_core::ROW_ID;
        use lance_file::writer::FileWriter;
        use lance_index::pb;
        use lance_index::vector::DISTANCE_TYPE_KEY;
        use lance_index::vector::SQ_CODE_COLUMN;
        use lance_index::vector::hnsw::HNSW;
        use lance_index::vector::ivf::storage::IVF_METADATA_KEY;
        use lance_index::vector::sq::ScalarQuantizer;
        use lance_index::vector::storage::STORAGE_METADATA_KEY;
        use lance_index::vector::v3::subindex::IvfSubIndex;
        use lance_index::{
            INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, INDEX_METADATA_SCHEMA_KEY,
            IndexMetadata as IndexMetaSchema,
        };
        use prost::Message;
        use std::ops::Range;
        use std::sync::Arc;

        let index_uuid = uuid::Uuid::new_v4();
        let index_dir = gen_path
            .clone()
            .join("_indices")
            .join(index_uuid.to_string());

        let distance_type = mem_index.distance_type();
        let dim = mem_index.dim();
        if dim == 0 {
            return Err(Error::invalid_input(
                "HnswMemIndex has no inserted vectors; nothing to flush",
            ));
        }
        // Forward-written data: HNSW row ids line up 1:1 with the data file, so
        // no position reversal (pass `None`).
        let Some((hnsw, flat_storage_batch)) = mem_index.to_lance_hnsw(None)? else {
            return Err(Error::invalid_input(
                "HnswMemIndex is empty; nothing to flush",
            ));
        };

        // Train SQ8 on the full memtable in one pass: learn global min/max
        // from every flushed vector, then quantize all rows in one shot.
        let row_id_col = flat_storage_batch
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::invalid_input("_rowid missing from HNSW storage batch"))?
            .clone();
        let flat_col = flat_storage_batch
            .column_by_name(lance_index::vector::flat::storage::FLAT_COLUMN)
            .ok_or_else(|| Error::invalid_input("flat column missing from HNSW storage batch"))?
            .clone();
        let flat_fsl = flat_col.as_fixed_size_list();
        let mut sq = ScalarQuantizer::new(8, dim);
        let bounds: Range<f64> = sq.update_bounds::<Float32Type>(flat_fsl)?;
        let sq_codes = sq.transform::<Float32Type>(flat_fsl as &dyn arrow_array::Array)?;

        let storage_schema = ArrowSchema::new(vec![
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new(
                SQ_CODE_COLUMN,
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::UInt8,
                        true,
                    )),
                    dim as i32,
                ),
                true,
            ),
        ]);
        let storage_batch = ArrowRecordBatch::try_new(
            Arc::new(storage_schema.clone()),
            vec![row_id_col, sq_codes],
        )?;

        // Single-partition IVF for both the storage and graph files. We need
        // *some* centroid because the on-disk read path routes every query
        // through `IvfModel::find_partitions` before HNSW search; that call
        // unwraps `centroids`. With one partition the centroid value is
        // irrelevant for routing — every query goes to partition 0 — so use
        // a zero vector.
        let zero_centroid_values = Float32Array::from(vec![0.0f32; dim]);
        let zero_centroid_fsl =
            FixedSizeListArray::try_new_from_values(zero_centroid_values, dim as i32)?;
        let mut storage_ivf =
            lance_index::vector::ivf::storage::IvfModel::new(zero_centroid_fsl.clone(), None);
        storage_ivf.add_partition(storage_batch.num_rows() as u32);

        let storage_path = index_dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
        let mut storage_writer = FileWriter::try_new(
            self.object_store.create(&storage_path).await?,
            (&storage_schema).try_into()?,
            Default::default(),
        )?;
        storage_writer.write_batch(&storage_batch).await?;

        let storage_ivf_pb = pb::Ivf::try_from(&storage_ivf)?;
        storage_writer.add_schema_metadata(DISTANCE_TYPE_KEY, distance_type.to_string());
        let ivf_buffer_pos = storage_writer
            .add_global_buffer(storage_ivf_pb.encode_to_vec().into())
            .await?;
        storage_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());

        // The reader needs the SQ metadata in two forms: a single
        // ScalarQuantizationMetadata under SQ_METADATA_KEY (whole-file path),
        // and a JSON array of per-partition ScalarQuantizationMetadata strings
        // under STORAGE_METADATA_KEY (per-partition path). With one partition
        // we serialize the same value twice.
        let sq_meta = lance_index::vector::sq::storage::ScalarQuantizationMetadata {
            dim,
            num_bits: 8,
            bounds,
        };
        let sq_meta_json = serde_json::to_string(&sq_meta)?;
        storage_writer.add_schema_metadata(
            STORAGE_METADATA_KEY,
            serde_json::to_string(&[&sq_meta_json])?,
        );
        storage_writer.add_schema_metadata(
            lance_index::vector::sq::storage::SQ_METADATA_KEY,
            sq_meta_json,
        );
        storage_writer.finish().await?;

        // Write the HNSW graph batch to index.idx. The graph file uses the
        // same single-partition IVF model with zero centroid for the same
        // reason as the storage file.
        let hnsw_batch = hnsw.to_batch()?;
        let hnsw_metadata_json = hnsw_batch
            .schema_ref()
            .metadata()
            .get(lance_index::vector::hnsw::builder::HNSW_METADATA_KEY)
            .cloned()
            .unwrap_or_default();
        let index_schema: ArrowSchema = HNSW::schema().as_ref().clone();
        let index_path = index_dir.clone().join(INDEX_FILE_NAME);
        let mut index_writer = FileWriter::try_new(
            self.object_store.create(&index_path).await?,
            (&index_schema).try_into()?,
            Default::default(),
        )?;
        index_writer.write_batch(&hnsw_batch).await?;

        let mut index_ivf =
            lance_index::vector::ivf::storage::IvfModel::new(zero_centroid_fsl, None);
        index_ivf.add_partition(hnsw_batch.num_rows() as u32);
        let index_ivf_pb = pb::Ivf::try_from(&index_ivf)?;
        // The on-disk type string matches Lance's index loader vocabulary —
        // an HNSW sub-index over SQ8-quantized vector storage, registered
        // under the same name as the standard IVF_HNSW_SQ path even though
        // our IVF layer is a single-partition placeholder.
        let index_metadata = IndexMetaSchema {
            index_type: "IVF_HNSW_SQ".to_string(),
            distance_type: distance_type.to_string(),
        };
        index_writer.add_schema_metadata(
            INDEX_METADATA_SCHEMA_KEY,
            serde_json::to_string(&index_metadata)?,
        );
        let ivf_buffer_pos = index_writer
            .add_global_buffer(index_ivf_pb.encode_to_vec().into())
            .await?;
        index_writer.add_schema_metadata(IVF_METADATA_KEY, ivf_buffer_pos.to_string());
        // Per-partition HNSW metadata: a JSON array with one entry.
        index_writer.add_schema_metadata(
            HNSW::metadata_key(),
            serde_json::to_string(&[hnsw_metadata_json])?,
        );
        index_writer.finish().await?;

        let index_details = Some(Arc::new(prost_types::Any {
            type_url: "type.googleapis.com/lance.index.VectorIndexDetails".to_string(),
            value: vec![],
        }));
        let index_meta = IndexMetadata {
            uuid: index_uuid,
            name: config.name.clone(),
            fields: vec![0], // updated by caller
            dataset_version: 0,
            fragment_bitmap: None,
            index_details,
            base_id: None,
            created_at: Some(chrono::Utc::now()),
            index_version: 1,
            files: None,
        };

        Ok(index_meta)
    }

    /// Update the shard manifest with the new flushed generation.
    async fn update_manifest(
        &self,
        epoch: u64,
        generation: u64,
        gen_path: &str,
        covered_wal_entry_position: u64,
    ) -> Result<ShardManifest> {
        let gen_path = gen_path.to_string();

        self.manifest_store
            .commit_update(epoch, |current| {
                let mut flushed_generations = current.flushed_generations.clone();
                flushed_generations.push(FlushedGeneration {
                    generation,
                    path: gen_path.clone(),
                });

                ShardManifest {
                    version: current.version + 1,
                    replay_after_wal_entry_position: covered_wal_entry_position,
                    wal_entry_position_last_seen: current
                        .wal_entry_position_last_seen
                        .max(covered_wal_entry_position),
                    current_generation: generation + 1,
                    flushed_generations,
                    ..current.clone()
                }
            })
            .await
    }
}

/// Message to trigger flush of a frozen memtable to Lance storage.
pub struct TriggerMemTableFlush {
    /// The frozen memtable to flush.
    pub memtable: Arc<MemTable>,
    /// Optional channel to notify when flush completes.
    pub done: Option<tokio::sync::oneshot::Sender<Result<FlushResult>>>,
}

impl std::fmt::Debug for TriggerMemTableFlush {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriggerMemTableFlush")
            .field("memtable_gen", &self.memtable.generation())
            .field("memtable_rows", &self.memtable.row_count())
            .field("has_done", &self.done.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn create_local_store() -> (Arc<ObjectStore>, Path, String, TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        (store, path, uri, temp_dir)
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

        // Not flushed to WAL yet
        assert!(!memtable.all_flushed_to_wal());

        let flusher = MemTableFlusher::new(store, base_path, base_uri, shard_id, manifest_store);
        let result = flusher.flush(&memtable, epoch, 0).await;

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
        let result = flusher.flush(&memtable, epoch, 0).await;

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
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);
        assert!(memtable.all_flushed_to_wal());

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path,
            base_uri,
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher.flush(&memtable, epoch, 1).await.unwrap();

        assert_eq!(result.generation.generation, 1);
        assert_eq!(result.rows_flushed, 10);
        assert_eq!(result.covered_wal_entry_position, 1);

        // Verify manifest was updated
        let updated_manifest = manifest_store.read_latest().await.unwrap().unwrap();
        assert_eq!(updated_manifest.version, 2);
        assert_eq!(updated_manifest.replay_after_wal_entry_position, 1);
        assert_eq!(updated_manifest.current_generation, 2);
        assert_eq!(updated_manifest.flushed_generations.len(), 1);
    }

    /// Flushing a generation with within-generation duplicate PKs writes a
    /// deletion vector so the flushed dataset exposes newest-per-PK on scan.
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
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path,
            base_uri.clone(),
            shard_id,
            manifest_store,
        );
        let result = flusher.flush(&memtable, epoch, 1).await.unwrap();
        assert_eq!(result.rows_flushed, 5, "all physical rows are written");

        // Scanning the flushed generation must honor the deletion vector and
        // return only the newest version of each PK.
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri.trim_end_matches('/'),
            shard_id,
            result.generation.path
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

    /// Covers `finalize_generation` writing both a deletion vector *and*
    /// indexes into the same manifest — the deletion-only and index-only
    /// paths are exercised by sibling tests.
    #[tokio::test]
    async fn test_flush_with_indexes_and_dedup_deletion_vector() {
        use super::super::super::index::{BTreeIndexConfig, IndexStore};
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
        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "name_btree".to_string(),
            field_id: 1,
            column: "name".to_string(),
        })];

        let schema = create_pk_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![0]).unwrap();
        let registry = IndexStore::from_configs(&index_configs, 100_000, 1_000).unwrap();
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
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_configs, 1)
            .await
            .unwrap();
        assert_eq!(result.rows_flushed, 5, "all physical rows are written");

        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri.trim_end_matches('/'),
            shard_id,
            result.generation.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        assert_eq!(
            dataset.version().version,
            1,
            "flushed dataset must be a single-version dataset"
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
        use super::super::super::index::{BTreeIndexConfig, IndexStore};
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
        let index_configs = vec![MemIndexConfig::BTree(BTreeIndexConfig {
            name: "id_btree".to_string(),
            field_id: 0,
            column: "id".to_string(),
        })];

        let schema = create_test_schema();
        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();

        // Set up in-memory index registry so preprocessed data path is used
        let registry = IndexStore::from_configs(&index_configs, 100_000, 1_000).unwrap();
        memtable.set_indexes(registry);

        let frag_id = memtable
            .insert(create_test_batch(&schema, 10))
            .await
            .unwrap();

        // Simulate WAL flush
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_configs, 1)
            .await
            .unwrap();

        assert_eq!(result.generation.generation, 1);
        assert_eq!(result.rows_flushed, 10);

        // Verify the flushed dataset is a single-version dataset with the BTree index
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri, shard_id, result.generation.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        assert_eq!(
            dataset.version().version,
            1,
            "flushed dataset must be a single-version dataset"
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
        let index_configs = vec![MemIndexConfig::hnsw(
            "vector_hnsw".to_string(),
            1,
            "vector".to_string(),
            DistanceType::L2,
        )];

        let mut memtable = MemTable::new(vector_schema.clone(), 1, vec![]).unwrap();
        let registry = IndexStore::from_configs(&index_configs, num_vectors, 100).unwrap();
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
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_configs, 1)
            .await
            .unwrap();

        assert_eq!(result.generation.generation, 1);
        assert_eq!(result.rows_flushed, num_vectors);

        // Verify the flushed dataset is a single-version dataset with the HNSW index
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri, shard_id, result.generation.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        assert_eq!(
            dataset.version().version,
            1,
            "flushed dataset must be a single-version dataset"
        );
        let indices = dataset.load_indices().await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "vector_hnsw");

        // End-to-end query: pick a row from the flushed dataset, query for
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
        use super::super::super::index::{FtsIndexConfig, IndexStore};
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
        let index_configs = vec![MemIndexConfig::Fts(FtsIndexConfig::new(
            "text_fts".to_string(),
            1,
            "text".to_string(),
        ))];

        let mut memtable = MemTable::new(schema.clone(), 1, vec![]).unwrap();

        // Set up in-memory index registry
        let registry = IndexStore::from_configs(&index_configs, 100_000, 1_000).unwrap();
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
        memtable.mark_wal_flushed(&[frag_id], 1, &[0]);

        let flusher = MemTableFlusher::new(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            shard_id,
            manifest_store.clone(),
        );
        let result = flusher
            .flush_with_indexes(&memtable, epoch, &index_configs, 1)
            .await
            .unwrap();

        assert_eq!(result.generation.generation, 1);
        assert_eq!(result.rows_flushed, 3);

        // Verify the flushed dataset is a single-version dataset with the FTS index
        let gen_uri = format!(
            "{}/_mem_wal/{}/{}",
            base_uri, shard_id, result.generation.path
        );
        let dataset = Dataset::open(&gen_uri).await.unwrap();
        assert_eq!(
            dataset.version().version,
            1,
            "flushed dataset must be a single-version dataset"
        );
        let indices = dataset.load_indices().await.unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, "text_fts");

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
  Take: ...
    CoalesceBatchesExec: ...
      MatchQuery: column=text, query=hello",
        )
        .await
        .unwrap();
    }
}
