// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset API extensions for MemWAL.
//!
//! # Limitations
//!
//! A MemWAL maintains the indexes its spec names, or every index the table has
//! when it names none. The set is re-read whenever a MemTable is built, so an
//! index created later is picked up without the spec being touched. Naming one
//! that never existed is rejected; one the dataset no longer has is skipped
//! when a shard opens. The sharding spec is not replaceable, because the
//! generations already written were homed under it.
//!
//! A writer that is already open keeps the set it opened with until something
//! calls [`DatasetMemWalExt::refresh_mem_wal_index_configs`].
//!
//! A rename reaches sealed generations and replay, but not the active MemTable,
//! which keeps its original names until its writer reopens.
//!
//! # Upgrading
//!
//! Generations are read by field id. Ones flushed before that carry positional
//! ids instead, which mispair against a table whose ids have gaps, and nothing
//! records which scheme a generation used. Compact generations into base before
//! upgrading. Durable WAL entries are unaffected: they carry no ids.
//!
//! # Known gaps
//!
//! A schema change started on a handle opened before the MemWAL was installed
//! still commits: the refusals below read MemWAL state from the caller's
//! handle, and the conflict resolver accepts both commits in either order.
//!
//! Adding a non-nullable column is not refused, which leaves rows in older
//! generations with no value for it.
//!
//! A generation frozen before an index joined the maintained set is flushed
//! without a vector or full-text index for it, and search over both is
//! index-only, so its rows answer neither until compaction folds them into the
//! base table.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_index::mem_wal::{MEM_WAL_INDEX_NAME, MemWalIndexDetails, ShardingField, ShardingSpec};
use lance_index::metrics::NoOpMetricsCollector;
use tracing::warn;
use uuid::Uuid;

use crate::Dataset;
use crate::dataset::CommitBuilder;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexExt;
use crate::index::DatasetIndexInternalExt;
use crate::index::mem_wal::{load_mem_wal_index_details, new_mem_wal_index_meta};

use super::index::{
    MemIndexRegistry, MemIndexSpec, ResolveContext, unsupported_index_type, validate_index_specs,
};
use super::scanner::sstable_cache::open_sstable;
use super::scanner::{DatasetCache, ShardSnapshot};
use super::schema_with_tombstone;
use super::util::derived_store_params;
use super::write::ShardWriter;
use super::{SealFence, ShardWriterConfig};

/// Spec id of the sole sharding spec installed by [`InitializeMemWalBuilder`].
const SHARDING_SPEC_ID: u32 = 1;

/// Spec id used by manually managed shards, which have no sharding spec.
const MANUAL_SHARD_SPEC_ID: u32 = 0;

/// Field id, within the sharding spec, of the derived shard-routing value.
const SHARDING_FIELD_ID: &str = "bucket";

/// Result type of the derived shard-routing value.
const SHARDING_RESULT_TYPE: &str = "int32";

/// Transform name for [`InitializeMemWalBuilder::bucket_sharding`]. Matches
/// Iceberg's `bucket(col, N)` partition transform name.
const BUCKET_TRANSFORM: &str = "bucket";

/// Transform name for [`InitializeMemWalBuilder::unsharded`]: every row maps to
/// a single shard.
const UNSHARDED_TRANSFORM: &str = "unsharded";

/// Transform name for [`InitializeMemWalBuilder::identity_sharding`]: the shard
/// value is the raw value of the source column.
const IDENTITY_TRANSFORM: &str = "identity";

/// Parameter key holding the bucket count `N` on the bucket transform.
const NUM_BUCKETS_PARAM: &str = "num_buckets";

/// Inclusive upper bound for `num_buckets`. Bounds the number of distinct
/// MemWAL shards a single bucket spec can address, which caps how many shard
/// manifests the dataset has to manage.
const MAX_NUM_BUCKETS: u32 = 1024;

/// Resolve the shard identity before a dataset-level writer creates or claims
/// a manifest.
fn resolve_writer_shard_spec_id(
    details: &MemWalIndexDetails,
    configured_spec_id: u32,
) -> Result<u32> {
    match details.sharding_specs.as_slice() {
        [] if configured_spec_id == MANUAL_SHARD_SPEC_ID => Ok(MANUAL_SHARD_SPEC_ID),
        [] => Err(Error::invalid_input(format!(
            "shard_spec_id {configured_spec_id} is invalid for manual sharding; expected {MANUAL_SHARD_SPEC_ID}"
        ))),
        [spec] if configured_spec_id == MANUAL_SHARD_SPEC_ID => Ok(spec.spec_id),
        [spec] if configured_spec_id == spec.spec_id => Ok(configured_spec_id),
        [spec] => Err(Error::invalid_input(format!(
            "shard_spec_id {configured_spec_id} does not match the MemWAL index sharding spec id {}",
            spec.spec_id
        ))),
        specs => Err(Error::not_supported(format!(
            "opening a MemWAL writer requires at most one sharding spec, found {}",
            specs.len()
        ))),
    }
}

/// How writes are partitioned into MemWAL shards.
#[derive(Debug)]
enum Sharding {
    /// No sharding spec is recorded; shards are managed manually.
    Manual,
    /// A single shard; every row is routed to it.
    Unsharded,
    /// Hash-bucket a shard key into `num_buckets` shards.
    Bucket { column: String, num_buckets: u32 },
    /// Shard by the raw value of `column` (identity transform).
    Identity { column: String },
}

/// Builder for initializing MemWAL on a [`Dataset`].
///
/// Created by [`DatasetMemWalExt::initialize_mem_wal`]. Choose a sharding
/// strategy and the indexes to maintain, then call [`execute`](Self::execute).
///
/// # Example
///
/// ```ignore
/// use lance::dataset::mem_wal::DatasetMemWalExt;
///
/// dataset
///     .initialize_mem_wal()
///     .bucket_sharding("id", 16)
///     .maintained_indexes(["id_btree"])
///     .execute()
///     .await?;
/// ```
#[must_use = "InitializeMemWalBuilder does nothing unless `.execute()` is awaited"]
pub struct InitializeMemWalBuilder<'a> {
    dataset: &'a mut Dataset,
    sharding: Sharding,
    maintained_indexes: Option<Vec<String>>,
    mem_index_registry: MemIndexRegistry,
    writer_config_defaults: HashMap<String, String>,
}

impl<'a> InitializeMemWalBuilder<'a> {
    fn new(dataset: &'a mut Dataset) -> Self {
        Self {
            dataset,
            sharding: Sharding::Manual,
            maintained_indexes: None,
            mem_index_registry: MemIndexRegistry::default(),
            writer_config_defaults: HashMap::new(),
        }
    }

    /// Route every row to a single MemWAL shard.
    pub fn unsharded(mut self) -> Self {
        self.sharding = Sharding::Unsharded;
        self
    }

    /// Hash-bucket `column` into `num_buckets` shards.
    ///
    /// `column` must name a scalar dataset column that can be hash-bucketed.
    /// `num_buckets` must be in `[1, 1024]`. These constraints are validated
    /// by [`execute`](Self::execute).
    pub fn bucket_sharding(mut self, column: impl Into<String>, num_buckets: u32) -> Self {
        self.sharding = Sharding::Bucket {
            column: column.into(),
            num_buckets,
        };
        self
    }

    /// Shard by the raw value of `column` (the identity transform).
    ///
    /// Each distinct value of `column` becomes its own shard; use this when the
    /// data is already partitioned by that column. For primary-key tables, the
    /// caller is responsible for ensuring every primary key maps consistently
    /// to a single value of `column`. `column` must be a scalar column that
    /// exists on the dataset; it is validated by [`execute`](Self::execute).
    pub fn identity_sharding(mut self, column: impl Into<String>) -> Self {
        self.sharding = Sharding::Identity {
            column: column.into(),
        };
        self
    }

    /// Set the base-table indexes to maintain in MemTables, replacing any
    /// previously set list.
    ///
    /// Each name must reference an existing index the MemWAL can maintain;
    /// [`execute`](Self::execute) enforces both. The primary key btree, when
    /// present, is maintained implicitly and must not be listed.
    pub fn maintained_indexes<I, S>(mut self, indexes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.maintained_indexes = Some(indexes.into_iter().map(Into::into).collect());
        self
    }

    /// Validate the maintained set against these plugins, the ones the writers
    /// will open with.
    pub fn mem_index_registry(mut self, registry: MemIndexRegistry) -> Self {
        self.mem_index_registry = registry;
        self
    }

    /// Record `config` as the default `ShardWriter` configuration.
    ///
    /// Every tunable field except `index_overrides`, whose values are Rust
    /// objects, is persisted into the MemWAL index so that all writers —
    /// across processes and restarts — start from the same defaults. Shard identity (`shard_id`, `shard_spec_id`) is not a
    /// configuration default and is not recorded. These remain defaults only:
    /// an individual writer may still override any value at runtime in its own
    /// (non-persisted) `ShardWriterConfig`.
    ///
    /// Merges into any defaults already set; a key set via
    /// [`add_writer_config_default`](Self::add_writer_config_default) afterwards wins.
    pub fn writer_config_defaults(mut self, config: ShardWriterConfig) -> Self {
        self.writer_config_defaults
            .extend(writer_config_to_defaults(&config));
        self
    }

    /// Record a single arbitrary writer-configuration default.
    ///
    /// Use this for keys not covered by
    /// [`writer_config_defaults`](Self::writer_config_defaults).
    pub fn add_writer_config_default(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.writer_config_defaults.insert(key.into(), value.into());
        self
    }

    /// Initialize MemWAL on the dataset, committing the MemWAL system index.
    ///
    /// Fails if any maintained index does not exist or cannot be maintained by
    /// the MemWAL, if the selected sharding configuration is invalid, or if
    /// MemWAL is already initialized.
    ///
    /// Validated against the dataset as it stands here; see the module-level
    /// limitations for changes made afterwards.
    pub async fn execute(self) -> Result<()> {
        let Self {
            dataset,
            sharding,
            maintained_indexes,
            mem_index_registry,
            writer_config_defaults,
        } = self;

        // Resolve (and validate) the sharding choice before any I/O.
        let (sharding_specs, num_shards) = resolve_sharding(dataset, sharding)?;

        dataset.schema().verify_primary_key()?;

        let indices = dataset.load_indices().await?;
        if indices.iter().any(|idx| idx.name == MEM_WAL_INDEX_NAME) {
            return Err(Error::invalid_input(
                "MemWAL is already initialized on this dataset.",
            ));
        }

        // Gate the commit, not just a preflight a caller may skip: a set the
        // writer cannot open leaves the table unwritable.
        let maintain_all_indexes = maintained_indexes.is_none();
        let maintained_indexes = maintained_indexes.unwrap_or_default();
        if !maintain_all_indexes {
            validate_maintained_indexes_with(dataset, &maintained_indexes, &mem_index_registry)
                .await?;
        }

        let details = MemWalIndexDetails {
            num_shards,
            sharding_specs,
            maintained_indexes,
            maintain_all_indexes,
            writer_config_defaults,
            ..Default::default()
        };

        let index_meta = new_mem_wal_index_meta(dataset.manifest.version, details)?;
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![index_meta],
                removed_indices: vec![],
            },
            None,
        );

        let new_dataset = CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(transaction)
            .await?;
        *dataset = new_dataset;

        Ok(())
    }
}

/// Resolve a [`Sharding`] choice into the sharding specs and shard count to
/// persist in [`MemWalIndexDetails`].
fn resolve_sharding(dataset: &Dataset, sharding: Sharding) -> Result<(Vec<ShardingSpec>, u32)> {
    match sharding {
        Sharding::Manual => Ok((Vec::new(), 0)),
        Sharding::Unsharded => Ok((vec![unsharded_sharding_spec()], 1)),
        Sharding::Bucket {
            column,
            num_buckets,
        } => Ok((
            vec![bucket_sharding_spec(dataset, &column, num_buckets)?],
            num_buckets,
        )),
        Sharding::Identity { column } => Ok((vec![identity_sharding_spec(dataset, &column)?], 0)),
    }
}

/// Build the sharding spec for [`InitializeMemWalBuilder::unsharded`].
fn unsharded_sharding_spec() -> ShardingSpec {
    ShardingSpec {
        spec_id: SHARDING_SPEC_ID,
        fields: vec![ShardingField {
            field_id: SHARDING_FIELD_ID.to_string(),
            source_ids: Vec::new(),
            transform: Some(UNSHARDED_TRANSFORM.to_string()),
            expression: None,
            result_type: SHARDING_RESULT_TYPE.to_string(),
            parameters: HashMap::new(),
        }],
    }
}

/// Build the sharding spec for [`InitializeMemWalBuilder::bucket_sharding`].
fn bucket_sharding_spec(dataset: &Dataset, column: &str, num_buckets: u32) -> Result<ShardingSpec> {
    if num_buckets == 0 || num_buckets > MAX_NUM_BUCKETS {
        return Err(Error::invalid_input(format!(
            "bucket_sharding: num_buckets must be in [1, {}], got {}",
            MAX_NUM_BUCKETS, num_buckets
        )));
    }

    let source_field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "bucket_sharding: column '{}' not found on the dataset",
            column
        ))
    })?;

    let data_type = source_field.data_type();
    if !is_bucket_sharding_supported_type(&data_type) {
        return Err(Error::invalid_input(format!(
            "bucket_sharding: column '{}' has type {:?}, which cannot be used as a shard key",
            column, data_type
        )));
    }

    Ok(ShardingSpec {
        spec_id: SHARDING_SPEC_ID,
        fields: vec![ShardingField {
            field_id: SHARDING_FIELD_ID.to_string(),
            source_ids: vec![source_field.id],
            transform: Some(BUCKET_TRANSFORM.to_string()),
            expression: None,
            result_type: SHARDING_RESULT_TYPE.to_string(),
            parameters: HashMap::from([(NUM_BUCKETS_PARAM.to_string(), num_buckets.to_string())]),
        }],
    })
}

/// Build the sharding spec for [`InitializeMemWalBuilder::identity_sharding`].
fn identity_sharding_spec(dataset: &Dataset, column: &str) -> Result<ShardingSpec> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "identity_sharding: column '{}' not found on the dataset",
            column
        ))
    })?;

    let data_type = field.data_type();
    let result_type = scalar_result_type(&data_type).ok_or_else(|| {
        Error::invalid_input(format!(
            "identity_sharding: column '{}' has type {:?}, which cannot be used as a shard key",
            column, data_type
        ))
    })?;

    Ok(ShardingSpec {
        spec_id: SHARDING_SPEC_ID,
        fields: vec![ShardingField {
            field_id: SHARDING_FIELD_ID.to_string(),
            source_ids: vec![field.id],
            transform: Some(IDENTITY_TRANSFORM.to_string()),
            expression: None,
            result_type: result_type.to_string(),
            parameters: HashMap::new(),
        }],
    })
}

fn is_bucket_sharding_supported_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
    )
}

/// The Arrow type name for a scalar column usable as a shard key, or `None`
/// for types that cannot be a shard key.
fn scalar_result_type(data_type: &DataType) -> Option<&'static str> {
    Some(match data_type {
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::UInt8 => "uint8",
        DataType::UInt16 => "uint16",
        DataType::UInt32 => "uint32",
        DataType::UInt64 => "uint64",
        DataType::Utf8 => "utf8",
        DataType::LargeUtf8 => "large_utf8",
        DataType::Boolean => "boolean",
        _ => return None,
    })
}

/// Extract the tunable defaults from a [`ShardWriterConfig`] into the persisted
/// string map. Shard identity (`shard_id`, `shard_spec_id`) is not a default.
/// `Duration` knobs are recorded in milliseconds with a `_ms` key suffix.
fn writer_config_to_defaults(config: &ShardWriterConfig) -> HashMap<String, String> {
    let mut defaults = HashMap::from([
        (
            "durable_write".to_string(),
            config.durable_write.to_string(),
        ),
        (
            "max_wal_buffer_size".to_string(),
            config.max_wal_buffer_size.to_string(),
        ),
        (
            "max_wal_persist_retries".to_string(),
            config.max_wal_persist_retries.to_string(),
        ),
        (
            "wal_persist_retry_base_delay_ms".to_string(),
            config.wal_persist_retry_base_delay.as_millis().to_string(),
        ),
        (
            "max_memtable_size".to_string(),
            config.max_memtable_size.to_string(),
        ),
        (
            "max_memtable_rows".to_string(),
            config.max_memtable_rows.to_string(),
        ),
        (
            "max_memtable_batches".to_string(),
            config.max_memtable_batches.to_string(),
        ),
        (
            "manifest_scan_batch_size".to_string(),
            config.manifest_scan_batch_size.to_string(),
        ),
        (
            "max_unflushed_memtable_bytes".to_string(),
            config.max_unflushed_memtable_bytes.to_string(),
        ),
        (
            "backpressure_log_interval_ms".to_string(),
            config.backpressure_log_interval.as_millis().to_string(),
        ),
        (
            "enable_memtable".to_string(),
            config.enable_memtable.to_string(),
        ),
    ]);
    if let Some(interval) = config.max_wal_flush_interval {
        defaults.insert(
            "max_wal_flush_interval_ms".to_string(),
            interval.as_millis().to_string(),
        );
    }
    if let Some(interval) = config.stats_log_interval {
        defaults.insert(
            "stats_log_interval_ms".to_string(),
            interval.as_millis().to_string(),
        );
    }
    // Per-index HNSW build params are recorded under `hnsw.<index>.<field>` keys.
    for (index_name, params) in &config.hnsw_params {
        defaults.insert(format!("hnsw.{index_name}.num_edges"), params.m.to_string());
        defaults.insert(
            format!("hnsw.{index_name}.ef_construction"),
            params.ef_construction.to_string(),
        );
        defaults.insert(
            format!("hnsw.{index_name}.max_level"),
            params.max_level.to_string(),
        );
    }
    defaults
}

/// Extension trait for Dataset to support MemWAL operations.
#[async_trait]
pub trait DatasetMemWalExt {
    /// Begin initializing MemWAL on this dataset.
    ///
    /// Returns an [`InitializeMemWalBuilder`]; configure the sharding strategy
    /// and maintained indexes, then call [`InitializeMemWalBuilder::execute`].
    fn initialize_mem_wal(&mut self) -> InitializeMemWalBuilder<'_>;

    /// Return the MemWAL index details for this dataset, if MemWAL is initialized.
    async fn mem_wal_index_details(&self) -> Result<Option<MemWalIndexDetails>> {
        Ok(None)
    }

    /// Replace the set of base-table indexes the MemTables maintain, and only
    /// that: everything else the MemWAL index carries is preserved. The
    /// sharding spec is not changeable, because the generations already written
    /// were homed under it and nothing re-homes them.
    ///
    /// A named set is validated against the table before the commit. It takes
    /// effect for MemTables built after it; a writer already open picks it up
    /// through [`Self::refresh_mem_wal_index_configs`].
    async fn update_mem_wal_maintained_indexes(
        &mut self,
        indexes: Option<Vec<String>>,
    ) -> Result<()> {
        self.update_mem_wal_maintained_indexes_with(indexes, &MemIndexRegistry::default())
            .await
    }

    /// As [`Self::update_mem_wal_maintained_indexes`], validating a named set
    /// against `registry`: the plugins the writers will be opened with.
    async fn update_mem_wal_maintained_indexes_with(
        &mut self,
        _indexes: Option<Vec<String>>,
        _registry: &MemIndexRegistry,
    ) -> Result<()> {
        Err(Error::not_supported(
            "update_mem_wal_maintained_indexes on this dataset type",
        ))
    }

    /// List current MemWAL shard IDs from object storage directory listing.
    async fn list_mem_wal_latest_shard_ids(&self) -> Result<Vec<Uuid>> {
        Ok(Vec::new())
    }

    /// Prewarm the SSTables of the given MemWAL shards into this
    /// dataset's session caches.
    ///
    /// For every SSTable in `snapshots`, opens the generation's
    /// on-disk dataset (populating the session's metadata/index caches, and the
    /// optional `cache` of opened `Arc<Dataset>`s) and prewarms each of its
    /// indexes. Opens run concurrently.
    ///
    /// The caller chooses how to enumerate the shards — list the `_mem_wal`
    /// directory (e.g. [`Self::list_mem_wal_latest_shard_ids`] then read each
    /// shard manifest), or read the MemWAL index shard snapshots — and passes
    /// the resulting [`ShardSnapshot`]s here. Prewarming is purely a cache
    /// optimization; correctness never depends on it, so passing a generation
    /// that has since been retired is harmless.
    async fn prewarm_mem_wal(
        &self,
        _snapshots: &[ShardSnapshot],
        _cache: Option<&Arc<dyn DatasetCache>>,
    ) -> Result<()> {
        Ok(())
    }

    /// Get a ShardWriter for the specified shard.
    ///
    /// Automatically loads index configurations from the MemWalIndex
    /// and creates the appropriate in-memory indexes.
    /// The default `config.shard_spec_id` is resolved to the index's sole
    /// automatic sharding spec; an explicit id must match it. A manually
    /// sharded index accepts only id `0`.
    ///
    /// # Arguments
    ///
    /// * `shard_id` - UUID identifying this shard
    /// * `config` - Writer configuration (durability, buffer sizes, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let writer = dataset.mem_wal_writer(
    ///     Uuid::new_v4(),
    ///     ShardWriterConfig::default(),
    /// ).await?;
    /// writer.put(vec![batch1, batch2]).await?;
    /// ```
    async fn mem_wal_writer(
        &self,
        shard_id: Uuid,
        config: ShardWriterConfig,
    ) -> Result<ShardWriter>;

    /// Re-resolve the maintained set from this dataset and install it on
    /// `writer`, sealing so the next MemTable carries it.
    ///
    /// This is how a change reaches a writer that is already open: a replaced
    /// `maintained_indexes`, or, for a table that maintains all of them, an
    /// index created since the writer opened. Read the dataset at the version to
    /// up -- a stale handle resolves the set it was already holding.
    ///
    /// `Ok(None)` when the set is unchanged, which is every tick that finds
    /// nothing new.
    async fn refresh_mem_wal_index_configs(
        &self,
        writer: &ShardWriter,
    ) -> Result<Option<SealFence>>;
}

/// Prewarm every index of `dataset` into its session caches. A no-op when the
/// dataset has no indexes; duplicate index names are warmed once.
async fn prewarm_all_indexes(dataset: &Dataset) -> Result<()> {
    let indices = dataset.load_indices().await?;
    let mut seen = std::collections::HashSet::new();
    for index in indices.iter() {
        if seen.insert(index.name.as_str()) {
            dataset.prewarm_index(&index.name).await?;
        }
    }
    Ok(())
}

#[async_trait]
impl DatasetMemWalExt for Dataset {
    fn initialize_mem_wal(&mut self) -> InitializeMemWalBuilder<'_> {
        InitializeMemWalBuilder::new(self)
    }

    async fn mem_wal_index_details(&self) -> Result<Option<MemWalIndexDetails>> {
        // Stored list, not the derived listing: a history this writer cannot decode must
        // not block add_columns.
        let Some(index_meta) = crate::index::load_all_indices(self)
            .await?
            .iter()
            .find(|idx| idx.name == MEM_WAL_INDEX_NAME)
            .cloned()
        else {
            return Ok(None);
        };

        load_mem_wal_index_details(index_meta).map(Some)
    }

    async fn update_mem_wal_maintained_indexes_with(
        &mut self,
        indexes: Option<Vec<String>>,
        registry: &MemIndexRegistry,
    ) -> Result<()> {
        let Some(existing_meta) = self.load_index_by_name(MEM_WAL_INDEX_NAME).await? else {
            return Err(Error::invalid_input(
                "MemWAL is not initialized on this dataset.",
            ));
        };
        let details = load_mem_wal_index_details(existing_meta.clone())?;
        let maintain_all_indexes = indexes.is_none();
        let indexes = indexes.unwrap_or_default();
        if details.maintain_all_indexes == maintain_all_indexes
            && details.maintained_indexes == indexes
        {
            return Ok(());
        }

        if !maintain_all_indexes {
            validate_maintained_indexes_with(self, &indexes, registry).await?;
        }

        let details = MemWalIndexDetails {
            maintained_indexes: indexes,
            maintain_all_indexes,
            ..details
        };
        let index_meta = new_mem_wal_index_meta(self.manifest.version, details)?;
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![index_meta],
                removed_indices: vec![existing_meta],
            },
            None,
        );

        let new_dataset = CommitBuilder::new(Arc::new(self.clone()))
            .execute(transaction)
            .await?;
        *self = new_dataset;

        Ok(())
    }

    async fn list_mem_wal_latest_shard_ids(&self) -> Result<Vec<Uuid>> {
        let prefix = super::util::mem_wal_path(&self.branch_location().path);
        let object_store = self.object_store(None).await?;
        let list_result = object_store
            .inner
            .list_with_delimiter(Some(&prefix))
            .await
            .map_err(|e| {
                Error::io(format!(
                    "failed to list MemWAL shard directories at {}: {}",
                    prefix, e
                ))
            })?;
        let mut ids = Vec::new();
        for shard_prefix in list_result.common_prefixes {
            if let Some(name) = shard_prefix.filename()
                && let Ok(shard_id) = Uuid::parse_str(name)
            {
                ids.push(shard_id);
            }
        }
        ids.sort();
        Ok(ids)
    }

    async fn prewarm_mem_wal(
        &self,
        snapshots: &[ShardSnapshot],
        cache: Option<&Arc<dyn DatasetCache>>,
    ) -> Result<()> {
        let session = self.session();
        // Every open below targets a generation URI, never the base's own.
        let store_params = self.store_params().map(derived_store_params);
        // Resolve SSTable paths exactly as the LSM collector does, so the
        // session/cache entries we warm key-match the paths later lookups open.
        let base_path = self.uri().trim_end_matches('/').to_string();
        let opens = snapshots
            .iter()
            .flat_map(|snapshot| {
                let shard_id = snapshot.shard_id;
                let base_path = &base_path;
                let session = &session;
                let store_params = &store_params;
                snapshot.sstables.iter().map(move |sstable| {
                    let path = format!("{}/_mem_wal/{}/{}", base_path, shard_id, sstable.path);
                    async move {
                        let dataset =
                            open_sstable(&path, Some(session), store_params.as_ref(), cache, None)
                                .await?;
                        prewarm_all_indexes(&dataset).await
                    }
                })
            })
            .collect::<Vec<_>>();
        futures::future::try_join_all(opens).await?;
        Ok(())
    }

    async fn mem_wal_writer(
        &self,
        shard_id: Uuid,
        mut config: ShardWriterConfig,
    ) -> Result<ShardWriter> {
        let details = require_mem_wal_details(self).await?;
        config.shard_spec_id = resolve_writer_shard_spec_id(&details, config.shard_spec_id)?;
        let index_specs = maintained_index_specs(self, &details, &config).await?;

        // Set shard_id in config
        config.shard_id = shard_id;

        // Inject the dataset's store params + session so the flusher opens the
        // base + generations with the same store the base was resolved with.
        config.store_params = self.store_params().cloned();
        config.session = Some(self.session());

        // Reuse the dataset's own object store + base path; `ObjectStore::from_uri`
        // would discard the store params the dataset was opened with, signing WAL
        // writes with the ambient identity. Mirrors `list_mem_wal_latest_shard_ids`.
        let base_uri = self.uri();
        let store = self.object_store(None).await?;
        let base_path = self.branch_location().path;

        // Create ShardWriter
        ShardWriter::open(
            store,
            base_path,
            base_uri,
            config,
            Arc::new(super::arrow_schema_with_field_ids(self.schema())),
            index_specs,
        )
        .await
    }

    async fn refresh_mem_wal_index_configs(
        &self,
        writer: &ShardWriter,
    ) -> Result<Option<SealFence>> {
        let details = require_mem_wal_details(self).await?;
        let index_specs = maintained_index_specs(self, &details, writer.config()).await?;
        writer.replace_index_configs(index_specs).await
    }
}

/// Whether an index kind this writer cannot mirror is fatal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OnUnsupportedIndex {
    /// The caller named this index, so refusing is the only honest answer: it
    /// asked for something the fresh tier cannot give it.
    Reject,
    /// The set is "every index on the table", so this one arrived by existing.
    /// Skipping keeps the table writable when a kind the writer cannot mirror
    /// is introduced.
    Skip,
}

impl ShardWriter {
    /// Moves this writer onto `dataset`'s current schema and maintained indexes.
    ///
    /// Pass the base table as read after the schema change. Returns `Ok(None)`
    /// when the writer is already up to date.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::mem_wal::ShardWriter;
    /// # async fn doc(writer: &ShardWriter, dataset: &Dataset) -> Result<()> {
    /// writer.evolve_to(dataset).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn evolve_to(&self, dataset: &Dataset) -> Result<Option<SealFence>> {
        let details = require_mem_wal_details(dataset).await?;
        let index_specs = maintained_index_specs(dataset, &details, self.config()).await?;
        self.evolve_schema(
            Arc::new(super::arrow_schema_with_field_ids(dataset.schema())),
            index_specs,
        )
        .await
    }
}

/// Whether an index the set names but the dataset does not have is fatal.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OnMissingIndex {
    /// Validating a set before it is installed: a name that resolves to nothing
    /// is the operator's mistake, and the only moment it can still be corrected.
    Reject,
    /// Opening a shard against a set installed earlier: the index may have been
    /// dropped since, or carried away with the column it covered. Rejecting
    /// here would refuse every read on the table for something that costs only
    /// the fresh tier's copy of one index.
    Skip,
}

/// Loads the MemWAL index details, failing if MemWAL is not initialized.
async fn require_mem_wal_details(dataset: &Dataset) -> Result<MemWalIndexDetails> {
    let index = dataset
        .open_mem_wal_index(&NoOpMetricsCollector)
        .await?
        .ok_or_else(|| {
            Error::invalid_input(
                "MemWAL is not initialized on this dataset. Call initialize_mem_wal() first.",
            )
        })?;
    Ok(index.details.clone())
}

/// The index specs a writer on `dataset` maintains, built with the writer's
/// own plugins and settings.
///
/// Every path that gives a writer its indexes uses this, so they all agree.
async fn maintained_index_specs(
    dataset: &Dataset,
    details: &MemWalIndexDetails,
    config: &ShardWriterConfig,
) -> Result<Vec<MemIndexSpec>> {
    let (index_names, on_unsupported) = resolve_maintained_indexes(dataset, details).await?;
    build_index_specs(
        dataset,
        &index_names,
        &writer_overrides(config)?,
        &config.mem_index_registry,
        OnMissingIndex::Skip,
        on_unsupported,
    )
    .await
}

/// The indexes a MemTable should carry, and how to treat one this writer cannot
/// mirror.
///
/// `maintain_all_indexes` resolves against the table's indexes every time this
/// runs, so an index created after the spec was written is picked up and a
/// dropped one falls out. System indexes are not table data and are never
/// mirrored.
async fn resolve_maintained_indexes(
    dataset: &Dataset,
    details: &MemWalIndexDetails,
) -> Result<(Vec<String>, OnUnsupportedIndex)> {
    if !details.maintain_all_indexes {
        return Ok((
            details.maintained_indexes.clone(),
            OnUnsupportedIndex::Reject,
        ));
    }
    let names = dataset
        .load_indices()
        .await?
        .iter()
        .filter(|index| !lance_index::is_system_index(index))
        .map(|index| index.name.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok((names, OnUnsupportedIndex::Skip))
}

/// The writer's per-index settings, as the values plugins read. Naming one
/// index in both `hnsw_params` and `index_overrides` is an error.
fn writer_overrides(
    config: &ShardWriterConfig,
) -> Result<HashMap<String, Arc<dyn Any + Send + Sync>>> {
    let mut overrides = config.index_overrides.clone();
    for (name, params) in &config.hnsw_params {
        if overrides.contains_key(name) {
            return Err(Error::invalid_input(format!(
                "index '{name}' has both HNSW parameters and other writer settings"
            )));
        }
        overrides.insert(name.clone(), Arc::new(params.clone()));
    }
    Ok(overrides)
}

/// Build the in-memory index specs for `index_names`.
///
/// Shared by [`DatasetMemWalExt::mem_wal_writer`] and
/// [`validate_maintained_indexes`], so a set that validates is one the writer
/// can build.
async fn build_index_specs(
    dataset: &Dataset,
    index_names: &[String],
    overrides: &HashMap<String, Arc<dyn Any + Send + Sync>>,
    registry: &MemIndexRegistry,
    on_missing: OnMissingIndex,
    on_unsupported: OnUnsupportedIndex,
) -> Result<Vec<MemIndexSpec>> {
    // The shard schema is the base schema plus `_tombstone`, as
    // `ShardWriter::open` extends it.
    let base_schema = super::arrow_schema_with_field_ids(dataset.schema());
    let shard_arrow = schema_with_tombstone(&base_schema);
    let shard_schema = LanceSchema::try_from(shard_arrow.as_ref())?;
    let shard_pk_columns: Vec<String> = shard_schema
        .unenforced_primary_key()
        .iter()
        .map(|field| field.name.clone())
        .collect();

    let mut index_specs = Vec::with_capacity(index_names.len());
    let mut listed = std::collections::HashSet::new();
    for index_name in index_names {
        // A name listed twice is one index.
        if !listed.insert(index_name.as_str()) {
            continue;
        }
        // A maintained index can split into multiple physical segments
        // (e.g. `optimize_indices(append)` deltas), which the singular
        // `load_index_by_name` rejects. Every segment carries the same
        // type and params, so take the first match.
        let index_meta = dataset
            .load_indices_by_name(index_name)
            .await?
            .into_iter()
            .next();

        // An index the maintained set names and the dataset does not have:
        // dropped outright, or carried away with the column it covered. Serve
        // without it -- the base index is gone for everyone, so the fresh tier
        // has nothing to keep in step with. See `OnMissingIndex::Skip`.
        let Some(index_meta) = index_meta else {
            if on_missing == OnMissingIndex::Reject {
                return Err(Error::invalid_input(format!(
                    "Index '{}' from maintained_indexes not found on dataset",
                    index_name
                )));
            }
            log::warn!(
                "index '{}' is named by maintained_indexes but is not on the dataset; \
                 the fresh tier will not maintain it",
                index_name
            );
            continue;
        };

        let type_url = index_meta
            .index_details
            .as_ref()
            .map(|d| d.type_url.as_str())
            .unwrap_or("");
        let Some(plugin) = registry.plugin_for_details_url(type_url).cloned() else {
            // Nobody named this index: it arrived because the table has it and
            // the set is "everything". A kind no plugin maintains must not make
            // the table unwritable, or introducing one upstream would break
            // every table maintaining all of them.
            if on_unsupported == OnUnsupportedIndex::Skip {
                log::warn!(
                    "index '{}' has a type no registered plugin maintains ({}); \
                     the fresh tier will not maintain it",
                    index_name,
                    type_url
                );
                continue;
            }
            return Err(unsupported_index_type(index_name, type_url, registry));
        };

        let columns = match index_meta
            .fields
            .iter()
            .map(|field_id| column_path(dataset.schema(), index_name, *field_id))
            .collect::<Result<Vec<_>>>()
        {
            Ok(columns) => columns,
            Err(error) => {
                skip_or_fail(on_unsupported, index_name, error)?;
                continue;
            }
        };

        // Not skipped under maintain-all: resolving reads the base index, and a
        // read that fails once must not drop the index for the writer's life.
        let ctx = ResolveContext::new(
            index_name,
            dataset,
            &index_meta,
            &shard_schema,
            &columns,
            overrides.get(index_name).map(|o| o.as_ref()),
        );
        let resolved = plugin.resolve(&ctx).await?;
        if ctx.overrides_ignored() {
            warn!(
                index = index_name,
                plugin = plugin.name(),
                "writer settings for this index were ignored: its plugin reads none"
            );
        }

        let field_ids =
            match resolved.field_ids {
                // The index's own columns keep its own field ids.
                None if resolved.columns == columns => Ok(index_meta.fields.clone()),
                None => resolved
                    .columns
                    .iter()
                    .map(|column| {
                        shard_schema.field(column).map(|field| field.id).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "index '{index_name}' resolved to column '{column}', which is not \
                             in the shard schema"
                        ))
                    })
                    })
                    .collect::<Result<Vec<_>>>(),
                Some(field_ids) => Ok(field_ids),
            };
        let field_ids = match field_ids {
            Ok(field_ids) => field_ids,
            Err(error) => {
                skip_or_fail(on_unsupported, index_name, error)?;
                continue;
            }
        };

        let spec = MemIndexSpec {
            name: index_name.clone(),
            field_ids,
            columns: resolved.columns,
            plugin,
            params: resolved.params,
        };

        // Nobody named this one, so a table must not become unwritable for
        // merely having an index the writer cannot build.
        if on_unsupported == OnUnsupportedIndex::Skip
            && let Err(error) = validate_index_specs(
                std::slice::from_ref(&spec),
                shard_arrow.as_ref(),
                &shard_schema,
                &shard_pk_columns,
            )
        {
            skip_or_fail(on_unsupported, index_name, error)?;
            continue;
        }
        index_specs.push(spec);
    }
    Ok(index_specs)
}

/// Under maintain-all, skip an index the writer cannot build so the table stays
/// writable; otherwise refuse it.
fn skip_or_fail(on_unsupported: OnUnsupportedIndex, index_name: &str, error: Error) -> Result<()> {
    if on_unsupported == OnUnsupportedIndex::Reject {
        return Err(error);
    }
    log::warn!(
        "index '{index_name}' is not one the fresh tier can maintain ({error}); it will not be \
         maintained"
    );
    Ok(())
}

/// The path a query uses for field `field_id`: its name at the top level, and
/// its full path from the root when nested.
fn column_path(schema: &LanceSchema, index_name: &str, field_id: i32) -> Result<String> {
    let ancestry = schema.field_ancestry_by_id(field_id).ok_or_else(|| {
        Error::invalid_input(format!(
            "index '{index_name}' names field {field_id}, which is not in the dataset schema"
        ))
    })?;
    match ancestry.as_slice() {
        [field] => Ok(field.name.clone()),
        _ => schema.field_path_minimal(field_id),
    }
}

/// Whether the MemWAL can maintain `index_names` on `dataset`.
///
/// Applies the same rules [`ShardWriter::open`] does, so a set that passes here
/// is a set the writer can open. [`InitializeMemWalBuilder::execute`] runs it
/// before committing; it is public so a caller inferring a set can ask the same
/// question first. A type url alone cannot decide this: a plugin also checks
/// the column, as HNSW needs a `FixedSizeList<Float32>` one.
///
/// All-or-nothing: it reports the first index it cannot maintain rather than
/// returning a usable subset, so a caller inferring a set surfaces the error
/// instead of dropping an index it believes is maintained.
///
/// Judges `dataset` as given; see the module-level limitations.
///
/// A plugin may read the base index while resolving it.
pub async fn validate_maintained_indexes(dataset: &Dataset, index_names: &[String]) -> Result<()> {
    validate_maintained_indexes_with(dataset, index_names, &MemIndexRegistry::default()).await
}

/// [`validate_maintained_indexes`] against `registry`, the plugins the writers
/// will open with.
pub async fn validate_maintained_indexes_with(
    dataset: &Dataset,
    index_names: &[String],
    registry: &MemIndexRegistry,
) -> Result<()> {
    // Validation does not depend on a writer's settings.
    let index_specs = build_index_specs(
        dataset,
        index_names,
        &HashMap::new(),
        registry,
        OnMissingIndex::Reject,
        OnUnsupportedIndex::Reject,
    )
    .await?;

    // The shard schema is base + `_tombstone`, as `ShardWriter::open` extends
    // it; field ids and the primary key resolve against that, not the base.
    let base_schema = super::arrow_schema_with_field_ids(dataset.schema());
    let schema = schema_with_tombstone(&base_schema);
    let lance_schema = LanceSchema::try_from(schema.as_ref())?;
    let pk_columns: Vec<String> = lance_schema
        .unenforced_primary_key()
        .iter()
        .map(|field| field.name.clone())
        .collect();

    validate_index_specs(&index_specs, schema.as_ref(), &lance_schema, &pk_columns)
}

#[cfg(test)]
mod tests {
    use super::super::index::HnswParams;
    use super::super::scanner::SsTableCache;
    use super::*;

    use arrow_array::{
        Array, Int32Array, ListArray, RecordBatch, RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_index::IndexType;
    use lance_index::scalar::inverted::DocumentGranularity;
    use lance_index::scalar::{InvertedIndexParams, ScalarIndexParams};
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use rstest::rstest;

    use crate::dataset::WriteParams;

    fn id_v_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("v", DataType::Int32, true),
        ]))
    }

    /// A dataset of 256 rows with an IVF vector index `vector_idx` over a
    /// `FixedSizeList<item_type>` column.
    async fn dataset_with_vector_index(uri: &str, item_type: DataType) -> Dataset {
        use crate::index::vector::VectorIndexParams;
        use arrow_array::ArrayRef;
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, Float64Builder};
        use lance_linalg::distance::DistanceType;

        const ROWS: i32 = 256;
        const DIM: i32 = 4;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", item_type.clone(), true)), DIM),
                true,
            ),
        ]));

        let vectors: ArrayRef = match item_type {
            DataType::Float32 => {
                let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), DIM);
                for row in 0..ROWS {
                    for d in 0..DIM {
                        builder.values().append_value((row * DIM + d) as f32);
                    }
                    builder.append(true);
                }
                Arc::new(builder.finish())
            }
            DataType::Float64 => {
                let mut builder = FixedSizeListBuilder::new(Float64Builder::new(), DIM);
                for row in 0..ROWS {
                    for d in 0..DIM {
                        builder.values().append_value((row * DIM + d) as f64);
                    }
                    builder.append(true);
                }
                Arc::new(builder.finish())
            }
            other => panic!("unhandled vector item type {other:?}"),
        };

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from((0..ROWS).collect::<Vec<_>>())),
                vectors,
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new([Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &VectorIndexParams::ivf_flat(1, DistanceType::L2),
                true,
            )
            .await
            .unwrap();
        dataset
    }

    fn id_v_batch(schema: &Arc<ArrowSchema>, ids: &[i32]) -> RecordBatch {
        let vs: Vec<i32> = ids.iter().map(|i| i * 10).collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(Int32Array::from(vs)),
            ],
        )
        .unwrap()
    }

    /// A dataset with the `id`/`v` schema holding `ids`, ready for MemWAL.
    async fn id_v_dataset(uri: &str, ids: &[i32]) -> Dataset {
        let schema = id_v_schema();
        let batches: Vec<_> = if ids.is_empty() {
            Vec::new()
        } else {
            vec![Ok(id_v_batch(&schema, ids))]
        };
        let reader = RecordBatchIterator::new(batches, schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap()
    }

    fn sharding_spec(spec_id: u32) -> ShardingSpec {
        ShardingSpec {
            spec_id,
            fields: Vec::new(),
        }
    }

    #[rstest]
    #[case::manual_accepts_manual(Vec::new(), MANUAL_SHARD_SPEC_ID, MANUAL_SHARD_SPEC_ID)]
    #[case::automatic_resolves_default(vec![sharding_spec(1)], MANUAL_SHARD_SPEC_ID, 1)]
    #[case::automatic_accepts_explicit_match(vec![sharding_spec(1)], 1, 1)]
    fn test_writer_shard_spec_resolution_accepts_valid_identity(
        #[case] sharding_specs: Vec<ShardingSpec>,
        #[case] configured_spec_id: u32,
        #[case] expected_spec_id: u32,
    ) {
        let details = MemWalIndexDetails {
            sharding_specs,
            ..Default::default()
        };

        let resolved = resolve_writer_shard_spec_id(&details, configured_spec_id).unwrap();
        assert_eq!(resolved, expected_spec_id);
    }

    #[rstest]
    #[case::manual_rejects_automatic(Vec::new(), 1, "manual sharding", false)]
    #[case::automatic_rejects_other(vec![sharding_spec(1)], 2, "sharding spec id 1", false)]
    #[case::multiple_specs_are_unsupported(
        vec![sharding_spec(1), sharding_spec(2)],
        0,
        "found 2",
        true
    )]
    fn test_writer_shard_spec_resolution_rejects_invalid_identity(
        #[case] sharding_specs: Vec<ShardingSpec>,
        #[case] configured_spec_id: u32,
        #[case] expected_message: &str,
        #[case] is_not_supported: bool,
    ) {
        let details = MemWalIndexDetails {
            sharding_specs,
            ..Default::default()
        };

        let error = resolve_writer_shard_spec_id(&details, configured_spec_id).unwrap_err();
        if is_not_supported {
            assert!(matches!(&error, Error::NotSupported { .. }));
        } else {
            assert!(matches!(&error, Error::InvalidInput { .. }));
        }
        assert!(error.to_string().contains(expected_message), "{error}");
    }

    #[tokio::test]
    async fn test_validate_maintained_indexes_rejects_non_f32_vector_column() {
        // A `FixedSizeList<Float64>` vector index is a valid durable index whose
        // type url resolves to `Hnsw` like any other, but the memtable HNSW needs
        // Float32 — so committing it would leave the table unwritable.
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let dataset = dataset_with_vector_index(&uri, DataType::Float64).await;

        let index_meta = dataset
            .load_indices_by_name("vector_idx")
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(
            MemIndexRegistry::default()
                .plugin_for_details_url(
                    index_meta.index_details.as_ref().unwrap().type_url.as_str()
                )
                .map(|plugin| plugin.name()),
            Some("Hnsw"),
            "the type url cannot see the column type"
        );

        let error = validate_maintained_indexes(&dataset, &["vector_idx".to_string()])
            .await
            .expect_err("a Float64 vector column is not maintainable");
        assert!(
            error.to_string().contains("FixedSizeList<Float32>"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn test_validate_maintained_indexes_accepts_f32_vector_column() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let dataset = dataset_with_vector_index(&uri, DataType::Float32).await;

        validate_maintained_indexes(&dataset, &["vector_idx".to_string()])
            .await
            .expect("a Float32 vector column is maintainable");
    }

    /// An index that covers nothing is maintainable: its recorded details
    /// state the distance type, so there is no file to open.
    ///
    /// A table can register WAL before it holds enough vectors to train, and
    /// validation refusing the definition would leave the index outside the
    /// maintained set for the life of the table — the set is a snapshot, so
    /// training it later does not add it back.
    #[tokio::test]
    async fn test_validate_maintained_indexes_accepts_a_definition() {
        use crate::index::vector::VectorIndexParams;
        use lance_linalg::distance::DistanceType;

        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        )]));
        let reader = RecordBatchIterator::new(vec![], schema.clone());
        let mut dataset = Dataset::write(reader, &uri, Some(WriteParams::default()))
            .await
            .unwrap();
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &VectorIndexParams::ivf_pq(4, 8, 2, DistanceType::Cosine, 1),
                true,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert!(
            indices[0]
                .fragment_bitmap
                .as_ref()
                .is_some_and(roaring::RoaringBitmap::is_empty),
            "an empty table trains nothing, so the index covers no rows"
        );

        validate_maintained_indexes(&dataset, &["vector_idx".to_string()])
            .await
            .expect("a definition is maintainable");
    }

    #[tokio::test]
    async fn test_validate_maintained_indexes_accepts_btree() {
        // Guards the shard-schema plumbing: validation resolves field ids against
        // base + `_tombstone`, so a scalar index on an ordinary column must pass.
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut dataset = id_v_dataset(&uri, &[1, 2, 3]).await;
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        validate_maintained_indexes(&dataset, &["id_idx".to_string()])
            .await
            .expect("a BTree index on an Int32 column is maintainable");
    }

    #[tokio::test]
    async fn test_validate_maintained_indexes_rejects_unmaintainable_kind() {
        // A zone map is a valid durable index the memtable cannot build. The
        // error names it, so a caller validating a set knows which to drop.
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut dataset = id_v_dataset(&uri, &[1, 2, 3]).await;
        dataset
            .create_index(
                &["v"],
                IndexType::ZoneMap,
                Some("v_zonemap".to_string()),
                &ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::ZoneMap),
                true,
            )
            .await
            .unwrap();

        let error = validate_maintained_indexes(&dataset, &["v_zonemap".to_string()])
            .await
            .expect_err("the memtable cannot build a zone map index");
        assert!(
            error.to_string().contains("v_zonemap"),
            "the error must name the index: {error}"
        );
    }

    #[tokio::test]
    async fn test_validate_maintained_indexes_rejects_unknown_name() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let dataset = id_v_dataset(&uri, &[1]).await;

        let error = validate_maintained_indexes(&dataset, &["nope".to_string()])
            .await
            .expect_err("an index that does not exist cannot be maintained");
        assert!(
            error.to_string().contains("not found"),
            "unexpected: {error}"
        );
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_rejects_unmaintainable_index() {
        // Initialization persists the set, so it must apply the writer's rules
        // itself: a Float64 vector index committed here leaves the table unwritable.
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut dataset = dataset_with_vector_index(&uri, DataType::Float64).await;

        let error = dataset
            .initialize_mem_wal()
            .unsharded()
            .maintained_indexes(["vector_idx"])
            .execute()
            .await
            .expect_err("a Float64 vector column is not maintainable");
        assert!(
            error.to_string().contains("FixedSizeList<Float32>"),
            "unexpected error: {error}"
        );
        assert!(
            dataset.mem_wal_index_details().await.unwrap().is_none(),
            "a rejected maintained set must not be committed"
        );
    }

    /// "Maintain everything" is an intent, not a list: an index built after the
    /// spec is maintained without anyone updating it, and a dropped one falls
    /// out. The set is resolved from the table each time it is asked for.
    #[tokio::test]
    async fn test_maintain_all_picks_up_an_index_built_later() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut dataset = id_v_dataset(&uri, &[]).await;

        // Installed with no indexes on the table at all.
        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .unwrap();
        let details = dataset
            .mem_wal_index_details()
            .await
            .unwrap()
            .expect("initialized");
        assert!(
            details.maintain_all_indexes,
            "an unnamed set is the intent to maintain everything"
        );
        assert!(
            details.maintained_indexes.is_empty(),
            "the intent is persisted, not a snapshot of the table's indexes"
        );
        let (resolved, _) = resolve_maintained_indexes(&dataset, &details)
            .await
            .unwrap();
        assert!(resolved.is_empty(), "the table has no indexes yet");

        // An index built afterwards is maintained with no further call.
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        let (resolved, on_unsupported) = resolve_maintained_indexes(&dataset, &details)
            .await
            .unwrap();
        assert_eq!(
            resolved,
            vec!["id_idx".to_string()],
            "an index built after the spec is maintained without updating it"
        );
        assert_eq!(
            on_unsupported,
            OnUnsupportedIndex::Skip,
            "nobody named these, so a kind the writer cannot mirror is skipped"
        );

        // A named set is the opposite on both counts.
        dataset
            .update_mem_wal_maintained_indexes(Some(Vec::new()))
            .await
            .unwrap();
        let named = dataset
            .mem_wal_index_details()
            .await
            .unwrap()
            .expect("initialized");
        assert!(!named.maintain_all_indexes);
        let (resolved, on_unsupported) =
            resolve_maintained_indexes(&dataset, &named).await.unwrap();
        assert!(
            resolved.is_empty(),
            "an empty named set maintains nothing, even though the table has an index"
        );
        assert_eq!(on_unsupported, OnUnsupportedIndex::Reject);
    }

    /// The maintained set moves in both directions, and nothing else in the
    /// index moves with it.
    ///
    /// The sharding spec, the shard count, what has been compacted and the
    /// writer defaults all live in the same message, so a careless update would
    /// take them with it. An unchanged set must not commit at all, and a
    /// rejected one must leave the committed set alone.
    #[tokio::test]
    async fn test_update_mem_wal_maintained_indexes_moves_only_that_set() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut dataset = id_v_dataset(&uri, &[1, 2]).await;
        for (columns, name) in [(&["id"][..], "id_idx"), (&["v"][..], "v_idx")] {
            dataset
                .create_index(
                    columns,
                    IndexType::BTree,
                    Some(name.to_string()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
        }
        dataset
            .initialize_mem_wal()
            .unsharded()
            .maintained_indexes(["id_idx", "v_idx"])
            .execute()
            .await
            .unwrap();
        let before = dataset
            .mem_wal_index_details()
            .await
            .unwrap()
            .expect("initialized");

        // Narrowing.
        dataset
            .update_mem_wal_maintained_indexes(Some(vec!["id_idx".to_string()]))
            .await
            .expect("dropping one from the set is allowed");
        let after = dataset
            .mem_wal_index_details()
            .await
            .unwrap()
            .expect("still initialized");
        assert_eq!(after.maintained_indexes, vec!["id_idx".to_string()]);
        assert_eq!(after.sharding_specs, before.sharding_specs);
        assert_eq!(after.num_shards, before.num_shards);
        assert_eq!(after.compacted_sstables, before.compacted_sstables);
        assert_eq!(after.writer_config_defaults, before.writer_config_defaults);

        // Unchanged: no commit at all.
        let version = dataset.manifest.version;
        dataset
            .update_mem_wal_maintained_indexes(Some(vec!["id_idx".to_string()]))
            .await
            .unwrap();
        assert_eq!(
            dataset.manifest.version, version,
            "an unchanged set must not commit"
        );

        // Refused, and the committed set is left alone.
        let error = dataset
            .update_mem_wal_maintained_indexes(Some(vec!["nope".to_string()]))
            .await
            .expect_err("an unknown index must be refused");
        assert!(error.to_string().contains("nope"), "unexpected: {error}");
        assert_eq!(
            dataset
                .mem_wal_index_details()
                .await
                .unwrap()
                .expect("still initialized")
                .maintained_indexes,
            vec!["id_idx".to_string()],
        );

        // Growing again, then emptying: both are sets, not an uninstall.
        dataset
            .update_mem_wal_maintained_indexes(Some(vec![
                "id_idx".to_string(),
                "v_idx".to_string(),
            ]))
            .await
            .expect("adding one back is allowed");
        dataset
            .update_mem_wal_maintained_indexes(Some(Vec::new()))
            .await
            .expect("an empty set is allowed");
        assert!(
            dataset
                .mem_wal_index_details()
                .await
                .unwrap()
                .expect("still initialized after emptying the set")
                .maintained_indexes
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_update_mem_wal_maintained_indexes_requires_an_initialized_mem_wal() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut dataset = id_v_dataset(&uri, &[1]).await;

        let error = dataset
            .update_mem_wal_maintained_indexes(Some(vec!["id_idx".to_string()]))
            .await
            .expect_err("no MemWAL to update");
        assert!(
            error.to_string().contains("not initialized"),
            "unexpected: {error}"
        );
        assert!(dataset.mem_wal_index_details().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_initialize_mem_wal_rejects_unknown_index_name() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut dataset = id_v_dataset(&uri, &[1]).await;

        let error = dataset
            .initialize_mem_wal()
            .unsharded()
            .maintained_indexes(["nope"])
            .execute()
            .await
            .expect_err("maintained_indexes must reference existing indexes");
        assert!(
            error.to_string().contains("nope") && error.to_string().contains("not found"),
            "unexpected error: {error}"
        );
        assert!(dataset.mem_wal_index_details().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_prewarm_mem_wal_opens_and_warms_indexes() {
        // `prewarm_mem_wal` opens each SSTable (into the base
        // dataset's session + the supplied cache) and warms its indexes. We
        // place an SSTable dataset with a BTree index at the
        // canonical `{base}/_mem_wal/{shard}/{folder}` path, prewarm it via a
        // snapshot, and assert the generation is cached and its index loadable.
        let tmp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let schema = id_v_schema();

        // Base dataset (1-row sentinel).
        let reader = RecordBatchIterator::new([Ok(id_v_batch(&schema, &[-1]))], schema.clone());
        let base = Dataset::write(reader, &base_uri, Some(WriteParams::default()))
            .await
            .unwrap();

        // SSTable with a BTree index on `id`.
        let shard_id = Uuid::new_v4();
        let folder = "deadbeef_gen_1";
        let gen_uri = format!("{}/_mem_wal/{}/{}", base_uri, shard_id, folder);
        let reader =
            RecordBatchIterator::new([Ok(id_v_batch(&schema, &[1, 2, 3]))], schema.clone());
        let mut gen_ds = Dataset::write(reader, &gen_uri, Some(WriteParams::default()))
            .await
            .unwrap();
        gen_ds
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_sstable(1, folder.to_string());

        let cache: Arc<dyn DatasetCache> = Arc::new(SsTableCache::new(4));
        base.prewarm_mem_wal(std::slice::from_ref(&snapshot), Some(&cache))
            .await
            .expect("prewarm must open the generation and warm its index");

        // The generation is resident in the cache (same session), with its
        // index loadable — a later lookup that opens this path is a pure hit.
        let warmed = cache
            .get_or_open(&gen_uri, Some(base.session()), base.store_params().cloned())
            .await
            .unwrap();
        assert_eq!(warmed.load_indices().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_prewarm_mem_wal_empty_is_noop() {
        // No snapshots / no SSTables: prewarm is a clean no-op.
        let tmp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let schema = id_v_schema();
        let reader = RecordBatchIterator::new([Ok(id_v_batch(&schema, &[-1]))], schema.clone());
        let base = Dataset::write(reader, &base_uri, Some(WriteParams::default()))
            .await
            .unwrap();

        base.prewarm_mem_wal(&[], None).await.unwrap();

        let empty = ShardSnapshot::new(Uuid::new_v4()).with_current_generation(1);
        base.prewarm_mem_wal(std::slice::from_ref(&empty), None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_mem_wal_writer_uses_automatic_sharding_spec() {
        let uri = "memory://";
        let schema = id_v_schema();
        let reader = RecordBatchIterator::new([Ok(id_v_batch(&schema, &[1]))], schema.clone());
        let mut dataset = Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap();
        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .unwrap();

        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
            .await
            .unwrap();
        let manifest = writer.manifest().await.unwrap().unwrap();

        assert_eq!(manifest.shard_spec_id, SHARDING_SPEC_ID);
        writer.close().await.unwrap();
    }

    /// An index the MemTable cannot mirror does not make the table unwritable.
    ///
    /// Nobody named it -- it arrived because the table has it and the set is
    /// "everything" -- so a vector index over `Float64`, which resolves to HNSW
    /// and is then refused on its element type, is skipped rather than failing
    /// every writer open.
    #[tokio::test]
    async fn test_maintain_all_skips_an_index_it_cannot_build() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().display());
        let mut dataset = dataset_with_vector_index(&uri, DataType::Float64).await;
        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .unwrap();
        let shard_id = Uuid::new_v4();
        dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
            .await
            .expect("automatic intent must skip an index the MemTable cannot mirror");
    }

    /// An index created after a writer opened still reaches that writer.
    ///
    /// This is the whole point of persisting the intent rather than a resolved
    /// list: nobody has to name the index, and nobody has to reopen the writer.
    #[tokio::test]
    async fn test_refresh_installs_an_index_created_after_the_writer_opened() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut dataset = id_v_dataset(&uri, &[1]).await;
        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .unwrap();

        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
            .await
            .unwrap();
        assert!(
            dataset
                .refresh_mem_wal_index_configs(&writer)
                .await
                .unwrap()
                .is_none(),
            "the table has no indexes, so there is nothing to install"
        );

        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .refresh_mem_wal_index_configs(&writer)
            .await
            .unwrap()
            .expect("the new index changes the set");

        writer
            .put(vec![id_v_batch(&id_v_schema(), &[2])])
            .await
            .unwrap();
        let refs = writer.in_memory_memtable_refs().await.unwrap();
        assert!(
            refs.active.index_store.get_btree("id_idx").is_some(),
            "the memtable opened after the refresh carries the new index"
        );

        assert!(
            dataset
                .refresh_mem_wal_index_configs(&writer)
                .await
                .unwrap()
                .is_none(),
            "a second pass over an unchanged table seals nothing"
        );
        writer.close().await.unwrap();
    }

    /// A full-text index on a field inside a list of structs is maintained.
    #[tokio::test]
    async fn test_a_full_text_index_inside_a_list_can_be_maintained() {
        let children = arrow_schema::Fields::from(vec![Field::new("name", DataType::Utf8, true)]);
        let values = StructArray::new(
            children.clone(),
            vec![Arc::new(StringArray::from(vec!["alpha"]))],
            None,
        );
        let item = Arc::new(Field::new("item", DataType::Struct(children), true));
        let tags = ListArray::new(
            item,
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 1])),
            Arc::new(values),
            None,
        );
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "tags",
            tags.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(tags)]).unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, "memory://nested_fts", None)
            .await
            .unwrap();
        let params =
            InvertedIndexParams::default().document_granularity(DocumentGranularity::ListElement);
        dataset
            .create_index(
                &["tags.name"],
                IndexType::Inverted,
                Some("tags_fts".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();
        validate_maintained_indexes(&dataset, &["tags_fts".to_string()])
            .await
            .unwrap();
    }

    /// Maintaining every index skips one on a nested field, so the table stays
    /// writable, and a same-named top-level field's index keeps its own field
    /// id.
    #[rstest]
    #[case::nested_only(false)]
    #[case::beside_a_top_level_namesake(true)]
    #[tokio::test]
    async fn test_maintain_all_skips_a_nested_index(#[case] with_top_level_code: bool) {
        use arrow_array::StructArray;
        use arrow_schema::Fields;

        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().display());
        let attrs_fields = Fields::from(vec![Field::new("code", DataType::Int32, true)]);
        let mut fields = vec![
            Field::new("id", DataType::Int32, false),
            Field::new("attrs", DataType::Struct(attrs_fields.clone()), true),
        ];
        if with_top_level_code {
            fields.push(Field::new("code", DataType::Int32, true));
        }
        let schema = Arc::new(ArrowSchema::new(fields));
        let rows = |ids: std::ops::Range<i32>| {
            let attrs = StructArray::new(
                attrs_fields.clone(),
                vec![Arc::new(Int32Array::from_iter_values(
                    ids.clone().map(|i| i % 7),
                ))],
                Some(ids.clone().map(|i| i % 3 != 0).collect()),
            );
            let mut columns: Vec<arrow_array::ArrayRef> = vec![
                Arc::new(Int32Array::from_iter_values(ids.clone())),
                Arc::new(attrs),
            ];
            if with_top_level_code {
                columns.push(Arc::new(Int32Array::from_iter_values(ids.map(|i| i % 5))));
            }
            RecordBatch::try_new(schema.clone(), columns).unwrap()
        };
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(rows(0..30))], schema.clone()),
            &uri,
            Some(WriteParams::default()),
        )
        .await
        .unwrap();
        let mut indexes = vec![("attrs.code", "nested_code_btree")];
        if with_top_level_code {
            indexes.push(("code", "code_btree"));
        }
        for (column, name) in &indexes {
            dataset
                .create_index(
                    &[*column],
                    IndexType::BTree,
                    Some(name.to_string()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
        }
        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .unwrap();

        let expected: Vec<String> = if with_top_level_code {
            vec!["code_btree".to_string()]
        } else {
            Vec::new()
        };
        let shard_id = Uuid::new_v4();
        // Closing flushes the memtable, so the second round reopens over it.
        for round in 0..2 {
            let writer = dataset
                .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
                .await
                .unwrap();
            assert_eq!(
                writer.maintained_index_names().await,
                expected,
                "round {round}"
            );
            writer
                .put(vec![rows(100 * (round + 1)..100 * (round + 1) + 20)])
                .await
                .unwrap();
            writer.close().await.unwrap();
        }
        if with_top_level_code {
            let specs = maintained_index_specs(
                &dataset,
                &require_mem_wal_details(&dataset).await.unwrap(),
                &ShardWriterConfig::new(shard_id),
            )
            .await
            .unwrap();
            assert_eq!(specs.len(), 1);
            assert_eq!(specs[0].columns, vec!["code".to_string()]);
            assert_eq!(
                specs[0].field_ids,
                vec![dataset.schema().field("code").unwrap().id],
                "bound to the top-level field, not the nested one of the same name"
            );
        }
    }

    use super::super::index::BTreeMemIndexPlugin;

    /// Claims a kind no built-in plugin maintains (the base table's zone map)
    /// and maintains and flushes it as a B-tree, on the named column if any.
    #[derive(Debug)]
    struct UnclaimedKindAsBTree(Option<&'static str>);

    #[async_trait::async_trait]
    impl super::super::index::MemIndexPlugin for UnclaimedKindAsBTree {
        fn name(&self) -> &str {
            "UnclaimedKindAsBTree"
        }
        fn details_message(&self) -> &str {
            "ZoneMapIndexDetails"
        }
        fn flush_index_type(&self) -> IndexType {
            IndexType::BTree
        }
        fn training_criteria(&self) -> lance_index::scalar::registry::TrainingCriteria {
            BTreeMemIndexPlugin.training_criteria()
        }
        async fn resolve(
            &self,
            ctx: &super::super::index::ResolveContext<'_>,
        ) -> Result<super::super::index::ResolvedIndex> {
            let columns = match self.0 {
                Some(column) => vec![column.to_string()],
                None => ctx.columns.to_vec(),
            };
            Ok(super::super::index::ResolvedIndex::plain(columns))
        }
        fn validate(&self, ctx: &super::super::index::MemIndexBuildContext<'_>) -> Result<()> {
            BTreeMemIndexPlugin.validate(ctx)
        }
        fn create(
            &self,
            ctx: &super::super::index::MemIndexBuildContext<'_>,
        ) -> Result<Arc<dyn super::super::index::MemIndex>> {
            BTreeMemIndexPlugin.create(ctx)
        }
    }

    async fn id_v_dataset_with_zone_map_and_btree(uri: &str) -> Dataset {
        let mut dataset = id_v_dataset(uri, &[1, 2, 3]).await;
        for (column, kind, name) in [
            ("id", IndexType::ZoneMap, "id_zone_map"),
            ("v", IndexType::BTree, "v_btree"),
        ] {
            dataset
                .create_index(
                    &[column],
                    kind,
                    Some(name.to_string()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
        }
        dataset
    }

    /// Maintaining every index skips one no plugin claims.
    #[tokio::test]
    async fn test_maintain_all_skips_an_index_no_plugin_claims() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().display());
        let mut dataset = id_v_dataset_with_zone_map_and_btree(&uri).await;
        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .unwrap();
        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
            .await
            .unwrap();
        assert_eq!(
            writer.maintained_index_names().await,
            vec!["v_btree".to_string()]
        );
        writer.close().await.unwrap();
    }

    /// A named set is refused until a registered plugin claims each index.
    #[tokio::test]
    async fn test_a_named_set_is_validated_against_the_writers_plugins() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().display());
        let mut dataset = id_v_dataset_with_zone_map_and_btree(&uri).await;
        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .unwrap();
        let named = Some(vec!["id_zone_map".to_string()]);

        let error = dataset
            .update_mem_wal_maintained_indexes(named.clone())
            .await
            .unwrap_err();
        assert!(
            matches!(error, lance_core::Error::InvalidInput { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains("id_zone_map"), "{error}");

        let registry = MemIndexRegistry::default()
            .with_plugin(Arc::new(UnclaimedKindAsBTree(None)))
            .unwrap();
        dataset
            .update_mem_wal_maintained_indexes_with(named, &registry)
            .await
            .unwrap();

        let shard_id = Uuid::new_v4();
        let added = ShardWriterConfig::new(shard_id)
            .with_mem_index_plugin(Arc::new(UnclaimedKindAsBTree(None)))
            .unwrap();
        assert!(
            added
                .mem_index_registry
                .plugin_for_details_url("ZoneMapIndexDetails")
                .is_some()
        );
        let config = ShardWriterConfig::new(shard_id).with_mem_index_registry(registry);
        let writer = dataset.mem_wal_writer(shard_id, config).await.unwrap();
        assert_eq!(
            writer.maintained_index_names().await,
            vec!["id_zone_map".to_string()]
        );
        writer.close().await.unwrap();
    }

    /// A plugin that resolves an index to other columns gets their field ids
    /// from the shard schema; a column the schema lacks skips the index under
    /// maintain-all and refuses a named set.
    #[tokio::test]
    async fn test_an_index_resolved_to_other_columns_takes_their_field_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().display());
        let dataset = id_v_dataset_with_zone_map_and_btree(&uri).await;
        let specs = |column: &'static str, on_unsupported: OnUnsupportedIndex| {
            let registry = MemIndexRegistry::default()
                .with_plugin(Arc::new(UnclaimedKindAsBTree(Some(column))))
                .unwrap();
            let dataset = &dataset;
            async move {
                build_index_specs(
                    dataset,
                    &["id_zone_map".to_string()],
                    &HashMap::new(),
                    &registry,
                    OnMissingIndex::Reject,
                    on_unsupported,
                )
                .await
            }
        };

        let moved = specs("v", OnUnsupportedIndex::Reject).await.unwrap();
        assert_eq!(moved[0].columns, vec!["v".to_string()]);
        assert_eq!(
            moved[0].field_ids,
            vec![dataset.schema().field("v").unwrap().id]
        );

        assert!(
            specs("nope", OnUnsupportedIndex::Skip)
                .await
                .unwrap()
                .is_empty()
        );
        let error = specs("nope", OnUnsupportedIndex::Reject).await.unwrap_err();
        assert!(error.to_string().contains("'nope'"), "{error}");
    }

    /// [`dataset_with_vector_index`] with a MemWAL that maintains every index.
    async fn vector_table_maintaining_all(uri: &str) -> Dataset {
        let mut dataset = dataset_with_vector_index(uri, DataType::Float32).await;
        dataset
            .initialize_mem_wal()
            .unsharded()
            .execute()
            .await
            .unwrap();
        dataset
    }

    /// Settings of a type the plugin does not read stop the writer from
    /// opening; settings for a kind that reads none are ignored, and the index
    /// is still maintained.
    #[tokio::test]
    async fn test_an_index_override_is_checked_only_by_a_plugin_that_reads_one() {
        let tmp = tempfile::tempdir().unwrap();
        let mut dataset =
            vector_table_maintaining_all(&format!("{}/base", tmp.path().display())).await;
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        let shard_id = Uuid::new_v4();

        let config = ShardWriterConfig::new(shard_id).with_index_override("vector_idx", 7u32);
        let Err(error) = dataset.mem_wal_writer(shard_id, config).await else {
            panic!("the writer must not open");
        };
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("vector_idx"), "{error}");

        let config = ShardWriterConfig::new(shard_id).with_index_override("id_idx", 7u32);
        let writer = dataset.mem_wal_writer(shard_id, config).await.unwrap();
        assert!(
            writer
                .maintained_index_names()
                .await
                .contains(&"id_idx".to_string())
        );
        writer.close().await.unwrap();
    }

    /// An index named in both `hnsw_params` and `index_overrides` is refused.
    #[tokio::test]
    async fn test_an_index_overridden_twice_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let dataset = vector_table_maintaining_all(&format!("{}/base", tmp.path().display())).await;
        let shard_id = Uuid::new_v4();
        let config = ShardWriterConfig::new(shard_id)
            .with_hnsw_params("vector_idx", HnswBuildParams::default())
            .with_index_override("vector_idx", HnswBuildParams::default());
        let Err(error) = dataset.mem_wal_writer(shard_id, config).await else {
            panic!("the writer must not open");
        };
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("vector_idx"), "{error}");
        assert!(error.to_string().contains("both"), "{error}");
    }

    /// HNSW settings given through `with_index_override` are the ones the index
    /// is built with.
    #[tokio::test]
    async fn test_index_overrides_reach_the_plugin() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().display());
        let dataset = dataset_with_vector_index(&uri, DataType::Float32).await;
        let settings = HnswBuildParams::default().num_edges(7).ef_construction(33);
        let config = ShardWriterConfig::new(Uuid::new_v4())
            .with_index_override("vector_idx", settings.clone());
        let specs = build_index_specs(
            &dataset,
            &["vector_idx".to_string()],
            &writer_overrides(&config).unwrap(),
            &MemIndexRegistry::default(),
            OnMissingIndex::Reject,
            OnUnsupportedIndex::Reject,
        )
        .await
        .unwrap();
        let params = specs[0].params::<HnswParams>().unwrap();
        assert_eq!(params.build_params, settings);
    }

    /// A maintained set naming one index twice maintains it once.
    #[tokio::test]
    async fn test_an_index_named_twice_is_maintained_once() {
        let tmp = tempfile::tempdir().unwrap();
        let uri = format!("{}/base", tmp.path().display());
        let mut dataset = id_v_dataset(&uri, &[1, 2]).await;
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .initialize_mem_wal()
            .unsharded()
            .maintained_indexes(["id_idx", "id_idx"])
            .execute()
            .await
            .unwrap();
        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(shard_id, ShardWriterConfig::new(shard_id))
            .await
            .unwrap();
        assert_eq!(
            writer.maintained_index_names().await,
            vec!["id_idx".to_string()]
        );
        writer.close().await.unwrap();
    }
}
