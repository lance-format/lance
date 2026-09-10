// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::future::Future;
use std::sync::{Arc, Mutex, OnceLock};

use arc_swap::ArcSwap;
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::future::BoxFuture;
use lance_core::{
    Result,
    cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache, UnsizedCacheKey},
    deepsize::{Context, DeepSizeOf},
};

use crate::progress::IndexBuildProgress;
use crate::registry::IndexPluginRegistry;
use crate::scalar::RowIdRemapper;
use crate::scalar::{CreatedIndex, IndexStore, ScalarIndex, expression::ScalarQueryParser};
// Re-export training types that were previously defined here
pub use crate::scalar::{TrainingCriteria, TrainingOrdering};

pub const VALUE_COLUMN_NAME: &str = "value";

/// A trait object for plugin-specific training parameters and data requirements.
///
/// Returned by [`BasicTrainer::new_training_request`]. The caller uses
/// [`criteria`](Self::criteria) to prepare the training data stream, then passes
/// the request back to [`BasicTrainer::train_index`], which may downcast
/// it to the plugin-specific concrete type to recover parsed parameters.
pub trait TrainingRequest: std::any::Any + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn criteria(&self) -> &TrainingCriteria;
}

/// A default training request impl for indexes that don't need any parameters
pub(crate) struct DefaultTrainingRequest {
    criteria: TrainingCriteria,
}

impl DefaultTrainingRequest {
    pub fn new(criteria: TrainingCriteria) -> Self {
        Self { criteria }
    }
}

impl TrainingRequest for DefaultTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

/// Implemented by indexes that can train on a stream of column data.
///
/// The training process has two stages. In the first stage, the caller provides
/// index parameters and receives a [`TrainingRequest`] that describes what criteria
/// the training data must satisfy (e.g. sort order, row-ID availability). In the
/// second stage, the caller prepares the data accordingly and calls
/// [`train_index`](Self::train_index).
///
/// Any scalar index plugin that builds from a column data stream should implement
/// this trait.
#[async_trait]
pub trait BasicTrainer: Send + Sync {
    /// Creates a new training request from the given parameters.
    ///
    /// The returned request specifies the criteria the training data must satisfy.
    /// It is the caller's responsibility to prepare data that meets those criteria
    /// before calling [`train_index`](Self::train_index).
    fn new_training_request(&self, params: &str, field: &Field)
    -> Result<Box<dyn TrainingRequest>>;

    /// Train a new index from a prepared data stream.
    ///
    /// The provided data must fulfill all the criteria returned by
    /// [`new_training_request`](Self::new_training_request). It is the caller's
    /// responsibility to ensure this.
    ///
    /// Returns index details describing the index. These details may be useful for
    /// planning and must be provided when loading the index. It is the caller's
    /// responsibility to store them.
    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex>;
}

/// A trait for scalar index plugins
#[async_trait]
pub trait ScalarIndexPlugin: Send + Sync + std::fmt::Debug {
    /// Returns this plugin's [`BasicTrainer`] implementation, if any.
    ///
    /// Training an index can be a complex process.  For example, a btree index might
    /// be trained using a shuffler from a distributed OLAP system such as
    /// Spark or Ray.  A vector index can be trained by sampling the column to create
    /// a kmeans model and then streaming the vectors to assign partitions.  Encapsulating
    /// the entire set of possible approaches is beyond what this trait can model.
    /// This is especially true because this is a low-level crate with no concept of a table
    /// or a dataset.
    ///
    /// However, in many cases, an index can be trained on a (potentially sorted) stream
    /// of column data.  There is also significant utility in being able to provide users
    /// with a simple generic "create an index" API.
    ///
    /// This method is a compromise.  Indexes that support training on a stream of column
    /// data should override this to return `Some(self)`.  Indexes that need their own
    /// individualized training approaches should return `None` and provide their own
    /// methods for training.
    ///
    /// An index can take both approaches.  Providing a simple (but maybe less
    /// efficient) stream-based trainer while also providing more specialized index
    /// creation methods elsewhere.
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        None
    }

    /// A short name for the index
    ///
    /// This is a friendly name for display purposes and also can be used as an alias for
    /// the index type URL.  If multiple plugins have the same name, then the first one
    /// found will be used.
    ///
    /// By convention this is MixedCase with no spaces.  When used as an alias, it will be
    /// compared case-insensitively.
    fn name(&self) -> &str;

    /// Returns true if the index returns an exact answer (e.g. not AtMost)
    fn provides_exact_answer(&self) -> bool;

    /// The version of the index plugin
    ///
    /// We assume that indexes are not forwards compatible.  If an index was written with a
    /// newer version than this, it cannot be read
    fn version(&self) -> u32;

    /// Returns a new query parser for the index
    ///
    /// Can return None if this index cannot participate in query optimization
    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>>;

    /// Load an index from storage
    ///
    /// The index details should match the details that were returned when the index was
    /// originally trained.
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>>;

    /// Look up a previously-opened index in the cache.
    ///
    /// `cache` is already per-index namespaced by the caller, so a plugin's key
    /// only needs to disambiguate entries within a single index.
    ///
    /// The default implementation reads an in-memory `Arc<dyn ScalarIndex>` entry.
    /// Plugins whose index has a serializable representation should override this
    /// (together with [`put_in_cache`](Self::put_in_cache)) to store that
    /// representation under a sized [`CacheKey`] with
    /// a codec, and reconstruct the index here. `index_store` and
    /// `frag_reuse_index` are provided so the override can rebuild the index
    /// without re-reading metadata.
    async fn get_from_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        _frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Option<Arc<dyn ScalarIndex>>> {
        let Some(entry) = cache.get_unsized_with_key(&ScalarIndexCacheKey).await else {
            return Ok(None);
        };
        Ok(entry.index_for_store(&index_store))
    }

    /// Store a freshly-opened index in the cache.
    ///
    /// `cache` is already per-index namespaced; see
    /// [`get_from_cache`](Self::get_from_cache).
    ///
    /// The default implementation stores the `Arc<dyn ScalarIndex>` in-memory.
    async fn put_in_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        cache: &LanceCache,
        index: Arc<dyn ScalarIndex>,
    ) -> Result<()> {
        cache
            .insert_unsized_with_key(
                &ScalarIndexCacheKey,
                Arc::new(StoreBoundScalarIndexCacheEntry::new(index_store, index)),
            )
            .await;
        Ok(())
    }

    /// Open an index through the cache, awaiting `load` only on a miss.
    ///
    /// The default read-through / write-back over
    /// [`get_from_cache`](Self::get_from_cache) and
    /// [`put_in_cache`](Self::put_in_cache) does not coalesce concurrent cold
    /// opens; plugins with a serializable form should override with
    /// [`single_flight_open`] so one shared load populates the sized state key.
    async fn get_or_insert_in_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
        load: ScalarIndexLoad<'_>,
    ) -> Result<Arc<dyn ScalarIndex>> {
        if let Some(index) = self
            .get_from_cache(index_store.clone(), frag_reuse_index, cache)
            .await?
        {
            return Ok(index);
        }
        let index = load.await?;
        self.put_in_cache(index_store, cache, index.clone()).await?;
        Ok(index)
    }

    /// Optional hook allowing a plugin to provide statistics without loading the index.
    async fn load_statistics(
        &self,
        _index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
    ) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }

    /// Optional hook that plugins can use if they need to be aware of the registry
    fn attach_registry(&self, _registry: Arc<IndexPluginRegistry>) {}

    /// Returns a JSON string representation of the provided index details
    ///
    /// These details will be user-visible and should be considered part of the public
    /// API.  As a result, efforts should be made to ensure the information is backwards
    /// compatible and avoid breaking changes.
    fn details_as_json(&self, _details: &prost_types::Any) -> Result<serde_json::Value> {
        // Return an empty JSON object as the default implementation
        Ok(serde_json::json!({}))
    }

    /// Optionally create a seed writer for the given column.
    ///
    /// A seed writer observes column values during data file writes, accumulates
    /// compact statistics in memory, and serializes them as a global buffer
    /// embedded in the data file footer. The buffer is later harvested during
    /// index updates to skip a full column scan.
    ///
    /// All parameters needed to construct the writer must be derivable from
    /// `index_details` — this method must not perform any I/O. Return `Ok(None)`
    /// if this index type does not support seed writing.
    async fn create_seed_writer(
        &self,
        _field_path: &str,
        _data_type: &DataType,
        _index_details: &prost_types::Any,
    ) -> Result<Option<Box<dyn super::seed::IndexSeedWriter>>> {
        Ok(None)
    }

    /// Returns true if this index type may have seed buffers embedded in data
    /// files for the given index configuration.
    ///
    /// When false the caller can skip opening data files to look for seeds
    /// entirely, avoiding I/O for index types or configurations that never
    /// write seeds.
    fn might_use_seeds(&self, _index_details: &prost_types::Any) -> bool {
        false
    }

    /// Attempt to update `reference_index` using pre-harvested `seeds` instead
    /// of re-scanning column data.
    ///
    /// Each [`FragmentSeed`](super::seed::FragmentSeed) carries the raw bytes
    /// written by the corresponding [`IndexSeedWriter`](super::seed::IndexSeedWriter)
    /// and the original `metadata_value` stored in the data file, which the plugin
    /// can use for compatibility validation (e.g. confirming `rows_per_zone`).
    ///
    /// Return `Ok(Some(created))` if the seed-based update succeeded, or
    /// `Ok(None)` to signal that the caller should fall back to a full column scan.
    async fn update_from_seeds(
        &self,
        _seeds: Vec<super::seed::FragmentSeed>,
        _reference_index: Arc<dyn ScalarIndex>,
        _index_details: &prost_types::Any,
        _dest_store: &dyn IndexStore,
    ) -> Result<Option<CreatedIndex>> {
        Ok(None)
    }
}

/// A boxed, `Send` future performing the storage-level load of a scalar index
/// (compat checks, `load_index`, metrics).
///
/// Passed to [`ScalarIndexPlugin::get_or_insert_in_cache`], which awaits it at
/// most once on a cache miss and drops it un-awaited on a warm hit.
pub type ScalarIndexLoad<'a> = BoxFuture<'a, Result<Arc<dyn ScalarIndex>>>;

/// Single-flight open helper for plugins with a serializable form.
///
/// Concurrent cold opens of the same index coalesce onto one `load`: it runs
/// once, `to_state` converts the opened index to its sized state, and the state
/// is cached under `state_key` (persisted via the key's
/// [`CacheCodec`](lance_core::cache::CacheCodec)). Every caller — warm hits
/// included — then rebuilds the index from the shared state via `from_state`, an
/// IO-free reconstruct.
///
/// `to_state` / `from_state` mirror the plugin's
/// [`put_in_cache`](ScalarIndexPlugin::put_in_cache) /
/// [`get_from_cache`](ScalarIndexPlugin::get_from_cache), keeping the state
/// representation defined in one place.
pub async fn single_flight_open<K, ToState, FromState>(
    cache: &LanceCache,
    state_key: K,
    load: ScalarIndexLoad<'_>,
    to_state: ToState,
    from_state: FromState,
) -> Result<Arc<dyn ScalarIndex>>
where
    K: CacheKey + Send,
    K::ValueType: DeepSizeOf + Send + Sync + 'static,
    ToState: FnOnce(&dyn ScalarIndex) -> Result<K::ValueType> + Send,
    FromState: FnOnce(Arc<K::ValueType>) -> Result<Arc<dyn ScalarIndex>> + Send,
{
    let state = cache
        .get_or_insert_with_key(state_key, move || async move {
            let index = load.await?;
            to_state(index.as_ref())
        })
        .await?;
    from_state(state)
}

pub(crate) async fn single_flight_store_bound_open<Rebind, RebindFuture>(
    index_store: Arc<dyn IndexStore>,
    cache: &LanceCache,
    load: ScalarIndexLoad<'_>,
    rebind: Rebind,
) -> Result<Arc<dyn ScalarIndex>>
where
    Rebind: FnOnce(Arc<dyn ScalarIndex>) -> RebindFuture + Send,
    RebindFuture: Future<Output = Result<Option<Arc<dyn ScalarIndex>>>> + Send,
{
    let pending_load = Arc::new(Mutex::new(Some(load)));
    let cache_load = pending_load.clone();
    let loaded_index = Arc::new(OnceLock::new());
    let cache_loaded_index = loaded_index.clone();
    let cache_index_store = index_store.clone();
    let entry = cache
        .get_or_insert_unsized_with_key(ScalarIndexCacheKey, move || async move {
            let load = take_scalar_index_load(&cache_load)?.ok_or_else(|| {
                lance_core::Error::internal(
                    "store-bound scalar index cache loader was already consumed",
                )
            })?;
            let index = load.await?;
            cache_loaded_index.get_or_init(|| index.clone());
            Ok(Arc::new(StoreBoundScalarIndexCacheEntry::new(
                cache_index_store,
                index,
            )))
        })
        .await?;

    // Another request can replace the shared slot after our load is published
    // but before this caller resumes. Keep the result of our own load so that
    // this request always receives the storage binding it opened.
    if let Some(index) = loaded_index.get() {
        return Ok(index.clone());
    }

    let binding = entry.binding.load_full();
    if index_store.is_same_storage_binding(binding.index_store.as_ref()) {
        return Ok(binding.index.clone());
    }

    // Reader-free state can be rebound independently for each request. Do not
    // let a slow or cancelled binding delay unrelated requests. Publish only
    // if the source binding is still current, so a delayed caller cannot roll
    // the cache back after another request has replaced it.
    if let Some(index) = rebind(binding.index.clone()).await? {
        let previous = entry.binding.compare_and_swap(
            &binding,
            Arc::new(StoreBoundScalarIndexBinding {
                index_store: index_store.clone(),
                index: index.clone(),
            }),
        );
        if !Arc::ptr_eq(&previous, &binding)
            && index_store.is_same_storage_binding(previous.index_store.as_ref())
        {
            return Ok(previous.index.clone());
        }
        return Ok(index);
    }

    // The cache slot stays stable across rotations. Serialize replacements and
    // recheck after locking so same-binding waiters reuse the first reload.
    let _replacement_guard = entry.replacement_guard.lock().await;
    if let Some(index) = entry.index_for_store(&index_store) {
        return Ok(index);
    }

    let load = take_scalar_index_load(&pending_load)?.ok_or_else(|| {
        lance_core::Error::internal("store-bound scalar index load has no retained result")
    })?;
    let index = load.await?;
    entry.replace(index_store, index.clone());
    Ok(index)
}

fn take_scalar_index_load<'a>(
    pending_load: &Arc<Mutex<Option<ScalarIndexLoad<'a>>>>,
) -> Result<Option<ScalarIndexLoad<'a>>> {
    pending_load
        .lock()
        .map_err(|_| {
            lance_core::Error::internal("store-bound scalar index cache loader mutex was poisoned")
        })
        .map(|mut pending_load| pending_load.take())
}

/// A live scalar index together with the store binding used to open it.
#[derive(DeepSizeOf)]
struct StoreBoundScalarIndexBinding {
    index_store: Arc<dyn IndexStore>,
    index: Arc<dyn ScalarIndex>,
}

/// A stable cache slot for one live, store-bound scalar index.
pub struct StoreBoundScalarIndexCacheEntry {
    binding: ArcSwap<StoreBoundScalarIndexBinding>,
    replacement_guard: tokio::sync::Mutex<()>,
}

impl DeepSizeOf for StoreBoundScalarIndexCacheEntry {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.binding.load_full().deep_size_of_children(context)
    }
}

impl StoreBoundScalarIndexCacheEntry {
    fn new(index_store: Arc<dyn IndexStore>, index: Arc<dyn ScalarIndex>) -> Self {
        Self {
            binding: ArcSwap::from_pointee(StoreBoundScalarIndexBinding { index_store, index }),
            replacement_guard: tokio::sync::Mutex::new(()),
        }
    }

    fn index_for_store(&self, index_store: &Arc<dyn IndexStore>) -> Option<Arc<dyn ScalarIndex>> {
        let binding = self.binding.load();
        index_store
            .is_same_storage_binding(binding.index_store.as_ref())
            .then(|| binding.index.clone())
    }

    fn replace(&self, index_store: Arc<dyn IndexStore>, index: Arc<dyn ScalarIndex>) {
        self.binding.store(Arc::new(StoreBoundScalarIndexBinding {
            index_store,
            index,
        }));
    }

    /// Return a shared handle to the cached scalar index.
    pub fn index(&self) -> Arc<dyn ScalarIndex> {
        self.binding.load().index.clone()
    }
}

/// In-memory cache key for a live, store-bound scalar index.
///
/// Used by the default [`ScalarIndexPlugin::get_from_cache`] /
/// [`ScalarIndexPlugin::put_in_cache`] implementations. The cache is already
/// per-index namespaced by the caller, so a constant key suffices. The entry
/// cannot be serialized, so this is an [`UnsizedCacheKey`] with no codec —
/// plugins that want a persistable cache entry override those methods with a
/// sized key.
pub struct ScalarIndexCacheKey;

impl UnsizedCacheKey for ScalarIndexCacheKey {
    type ValueType = StoreBoundScalarIndexCacheEntry;

    fn key(&self) -> Cow<'_, str> {
        Cow::Borrowed("scalar_index")
    }

    fn type_name() -> &'static str {
        "ScalarIndex"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.registry.scalar-index-key", 2)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_variant(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, pin::Pin};

    use arrow_schema::Schema;
    use futures::FutureExt;
    use lance_core::cache::{
        CacheBackend, CacheCodec, CacheEntry, InternalCacheKey, MokaCacheBackend,
    };
    use lance_io::object_store::ObjectStore;
    use tokio::sync::Notify;

    use crate::scalar::inverted::{
        InvertedIndex, InvertedIndexParams, METADATA_FILE, TOKEN_SET_FORMAT_KEY, TokenSetFormat,
    };
    use crate::scalar::lance_format::LanceIndexStore;

    /// Pause the cold caller after publication, allowing a warm caller to rotate
    /// the same entry before the cold caller receives its result.
    #[derive(Debug)]
    struct PauseColdReturn {
        inner: MokaCacheBackend,
        published: Notify,
        resume: Notify,
    }

    #[async_trait]
    impl CacheBackend for PauseColdReturn {
        async fn get(
            &self,
            key: &InternalCacheKey,
            codec: Option<CacheCodec>,
        ) -> Option<CacheEntry> {
            self.inner.get(key, codec).await
        }

        async fn insert(
            &self,
            key: &InternalCacheKey,
            entry: CacheEntry,
            size_bytes: usize,
            codec: Option<CacheCodec>,
        ) {
            self.inner.insert(key, entry, size_bytes, codec).await;
        }

        async fn get_or_insert<'a>(
            &self,
            key: &InternalCacheKey,
            loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
            codec: Option<CacheCodec>,
        ) -> Result<(CacheEntry, bool)> {
            let result = self.inner.get_or_insert(key, loader, codec).await?;
            if !result.1 {
                self.published.notify_one();
                self.resume.notified().await;
            }
            Ok(result)
        }

        async fn clear(&self) {
            self.inner.clear().await;
        }

        async fn num_entries(&self) -> usize {
            self.inner.num_entries().await
        }

        async fn size_bytes(&self) -> usize {
            self.inner.size_bytes().await
        }
    }

    async fn empty_index_bindings() -> [(Arc<dyn IndexStore>, Arc<dyn ScalarIndex>); 2] {
        let object_store = ObjectStore::memory();
        let metadata_cache = Arc::new(LanceCache::with_capacity(1024 * 1024));
        let store_a: Arc<dyn IndexStore> = Arc::new(LanceIndexStore::new(
            Arc::new(object_store.clone()),
            "index".into(),
            metadata_cache.clone(),
        ));
        let store_b: Arc<dyn IndexStore> = Arc::new(LanceIndexStore::new(
            Arc::new(object_store),
            "index".into(),
            metadata_cache,
        ));
        assert!(!store_a.is_same_storage_binding(store_b.as_ref()));
        let mut writer = store_a
            .new_index_file(METADATA_FILE, Arc::new(Schema::empty()))
            .await
            .unwrap();
        writer
            .finish_with_metadata(HashMap::from([
                ("partitions".to_owned(), "[]".to_owned()),
                (
                    "params".to_owned(),
                    serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
                ),
                (
                    TOKEN_SET_FORMAT_KEY.to_owned(),
                    TokenSetFormat::default().to_string(),
                ),
            ]))
            .await
            .unwrap();
        let index_cache = LanceCache::no_cache();
        let index_a = InvertedIndex::load(store_a.clone(), None, &index_cache)
            .await
            .unwrap();
        let index_b = InvertedIndex::load(store_b.clone(), None, &index_cache)
            .await
            .unwrap();
        [(store_a, index_a), (store_b, index_b)]
    }

    #[tokio::test]
    async fn test_cold_scalar_open_keeps_its_binding_after_concurrent_rotation() {
        let [(store_a, index_a), (store_b, index_b)] = empty_index_bindings().await;
        let backend = Arc::new(PauseColdReturn {
            inner: MokaCacheBackend::with_capacity(1024 * 1024),
            published: Notify::new(),
            resume: Notify::new(),
        });
        let cache = LanceCache::with_backend(backend.clone());
        let cold_cache = cache.clone();
        let cold_index = index_a.clone();
        let cold = tokio::spawn(async move {
            single_flight_store_bound_open(
                store_a,
                &cold_cache,
                async move { Ok(cold_index) }.boxed(),
                |_| async { panic!("cold caller must keep its own loaded index") },
            )
            .await
        });
        backend.published.notified().await;
        let replacement = index_b.clone();
        let warm = single_flight_store_bound_open(
            store_b,
            &cache,
            async { panic!("warm caller must rebind the cached index") }.boxed(),
            |_| async move { Ok(Some(replacement)) },
        )
        .await
        .unwrap();
        assert!(Arc::ptr_eq(&warm, &index_b));
        backend.resume.notify_one();
        let cold = cold.await.unwrap().unwrap();
        assert!(Arc::ptr_eq(&cold, &index_a));
        let cached = cache
            .get_unsized_with_key(&ScalarIndexCacheKey)
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&cached.index(), &index_b));
    }

    #[tokio::test]
    async fn test_independent_scalar_rebinds_do_not_wait_for_each_other() {
        let [(store_a, index_a), (store_b, index_b)] = empty_index_bindings().await;
        let [(store_c, index_c), _] = empty_index_bindings().await;
        let cache = LanceCache::with_capacity(1024 * 1024);
        let entry = Arc::new(StoreBoundScalarIndexCacheEntry::new(store_a, index_a));
        cache
            .insert_unsized_with_key(&ScalarIndexCacheKey, entry.clone())
            .await;
        let started = Arc::new(Notify::new());
        let resume = Arc::new(Notify::new());
        let slow_started = started.clone();
        let slow_resume = resume.clone();
        let slow_cache = cache.clone();
        let slow_index = index_b.clone();
        let slow = tokio::spawn(async move {
            single_flight_store_bound_open(
                store_b,
                &slow_cache,
                async { panic!("warm request must not reload metadata") }.boxed(),
                |_| async move {
                    slow_started.notify_one();
                    slow_resume.notified().await;
                    Ok(Some(slow_index))
                },
            )
            .await
        });
        started.notified().await;
        let fast_index = index_c.clone();
        let fast = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            single_flight_store_bound_open(
                store_c,
                &cache,
                async { panic!("warm request must not reload metadata") }.boxed(),
                |_| async move { Ok(Some(fast_index)) },
            ),
        )
        .await;
        resume.notify_one();
        let slow = slow.await.unwrap().unwrap();
        let fast = fast
            .expect("an independent request waited for another binding's rebind")
            .unwrap();
        assert!(Arc::ptr_eq(&slow, &index_b));
        assert!(Arc::ptr_eq(&fast, &index_c));
        assert!(
            Arc::ptr_eq(&entry.index(), &index_c),
            "a delayed rebind must not overwrite the newer cache binding"
        );
    }

    #[tokio::test]
    async fn test_failed_or_cancelled_scalar_rebind_preserves_cached_binding() {
        let [(store_a, index_a), (store_b, index_b)] = empty_index_bindings().await;
        let cache = LanceCache::with_capacity(1024 * 1024);
        let entry = Arc::new(StoreBoundScalarIndexCacheEntry::new(
            store_a,
            index_a.clone(),
        ));
        cache
            .insert_unsized_with_key(&ScalarIndexCacheKey, entry.clone())
            .await;
        let error = single_flight_store_bound_open(
            store_b.clone(),
            &cache,
            async { panic!("failed rebind must not fall back to loading") }.boxed(),
            |_| async { Err(lance_core::Error::io("replacement credentials revoked")) },
        )
        .await
        .unwrap_err();
        assert!(matches!(error, lance_core::Error::IO { .. }));
        assert!(
            error
                .to_string()
                .contains("replacement credentials revoked")
        );
        assert!(Arc::ptr_eq(&entry.index(), &index_a));

        let started = Arc::new(Notify::new());
        let cancel_started = started.clone();
        let cancel_cache = cache.clone();
        let cancel_store = store_b.clone();
        let cancelled = tokio::spawn(async move {
            single_flight_store_bound_open(
                cancel_store,
                &cancel_cache,
                async { panic!("cancelled rebind must not fall back to loading") }.boxed(),
                |_| async move {
                    cancel_started.notify_one();
                    futures::future::pending().await
                },
            )
            .await
        });
        started.notified().await;
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());
        assert!(Arc::ptr_eq(&entry.index(), &index_a));

        let replacement = index_b.clone();
        let reopened = single_flight_store_bound_open(
            store_b,
            &cache,
            async { panic!("retry must rebind the cached index") }.boxed(),
            |_| async move { Ok(Some(replacement)) },
        )
        .await
        .unwrap();
        assert!(Arc::ptr_eq(&reopened, &index_b));
        assert!(Arc::ptr_eq(&entry.index(), &index_b));
    }
}
