// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage options provider and accessor for dynamic credential fetching
//!
//! This module provides:
//! - [`StorageOptionsProvider`] trait for fetching storage options from various sources
//!   (namespace servers, secret managers, etc.) with support for expiration tracking
//! - [`StorageOptionsAccessor`] for unified access to storage options with automatic
//!   caching and refresh

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

#[cfg(test)]
use mock_instant::thread_local::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(not(test))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use lance_namespace::LanceNamespace;
use lance_namespace::models::DescribeTableRequest;
use tokio::sync::{Mutex, RwLock};

use crate::{Error, Result};

/// Key for the expiration timestamp in storage options HashMap
pub const EXPIRES_AT_MILLIS_KEY: &str = "expires_at_millis";

/// Key for the refresh offset in storage options HashMap (milliseconds before expiry to refresh)
pub const REFRESH_OFFSET_MILLIS_KEY: &str = "refresh_offset_millis";

/// Default refresh offset: 60 seconds before expiration
const DEFAULT_REFRESH_OFFSET_MILLIS: u64 = 60_000;

/// Trait for providing storage options with expiration tracking
///
/// Implementations can fetch storage options from various sources (namespace servers,
/// secret managers, etc.) and are usable from Python/Java.
///
/// # Current Use Cases
///
/// - **Temporary Credentials**: Fetch short-lived AWS temporary credentials that expire
///   after a set time period, with automatic refresh before expiration
///
/// # Future Possible Use Cases
///
/// - **Dynamic Storage Location Resolution**: Resolve logical names to actual storage
///   locations (bucket aliases, S3 Access Points, region-specific endpoints) that may
///   change based on region, tier, data migration, or failover scenarios
/// - **Runtime S3 Tags Assignment**: Inject cost allocation tags, security labels, or
///   compliance metadata into S3 requests based on the current execution context (user,
///   application, workspace, etc.)
/// - **Dynamic Endpoint Configuration**: Update storage endpoints for disaster recovery,
///   A/B testing, or gradual migration scenarios
/// - **Just-in-time Permission Elevation**: Request elevated permissions only when needed
///   for sensitive operations, then immediately revoke them
/// - **Secret Manager Integration**: Fetch encryption keys from AWS Secrets Manager,
///   Azure Key Vault, or Google Secret Manager with automatic rotation
/// - **OIDC/SAML Federation**: Integrate with identity providers to obtain storage
///   credentials based on user identity and group membership
///
/// # Equality and Hashing
///
/// Implementations must provide `provider_id()` which returns a unique identifier for
/// equality and hashing purposes. Two providers with the same ID are considered equal
/// and will share the same cached ObjectStore in the registry.
#[async_trait]
pub trait StorageOptionsProvider: Send + Sync + fmt::Debug {
    /// Fetch fresh storage options
    ///
    /// Returns None if no storage options are available, or Some(HashMap) with the options.
    /// If the [`EXPIRES_AT_MILLIS_KEY`] key is present in the HashMap, it should contain the
    /// epoch time in milliseconds when the options expire, and credentials will automatically
    /// refresh before expiration.
    /// If [`EXPIRES_AT_MILLIS_KEY`] is not provided, the options are considered to never expire.
    async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>>;

    /// Fetch fresh storage options, bypassing caches along the chain.
    ///
    /// Providers that serve from an upstream cache (e.g. base-scoped wrappers
    /// reading through a parent accessor) override this to force the upstream
    /// to re-fetch. Defaults to [`Self::fetch_storage_options`].
    async fn force_fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
        self.fetch_storage_options().await
    }

    /// Return a human-readable unique identifier for this provider instance
    ///
    /// This is used for equality comparison and hashing in the object store registry.
    /// Two providers with the same ID will be treated as equal and share the same cached
    /// ObjectStore.
    ///
    /// The ID should be human-readable for debugging and logging purposes.
    /// For example: `"namespace[dir(root=/data)],table[db$schema$table1]"`
    ///
    /// The ID should uniquely identify the provider's configuration.
    fn provider_id(&self) -> String;
}

/// StorageOptionsProvider implementation that fetches options from a LanceNamespace
pub struct LanceNamespaceStorageOptionsProvider {
    namespace_client: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
}

impl fmt::Debug for LanceNamespaceStorageOptionsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.provider_id())
    }
}

impl fmt::Display for LanceNamespaceStorageOptionsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.provider_id())
    }
}

impl LanceNamespaceStorageOptionsProvider {
    /// Create a new LanceNamespaceStorageOptionsProvider
    ///
    /// # Arguments
    /// * `namespace_client` - The namespace implementation to fetch storage options from
    /// * `table_id` - The table identifier
    pub fn new(namespace_client: Arc<dyn LanceNamespace>, table_id: Vec<String>) -> Self {
        Self {
            namespace_client,
            table_id,
        }
    }
}

#[async_trait]
impl StorageOptionsProvider for LanceNamespaceStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
        let request = DescribeTableRequest {
            id: Some(self.table_id.clone()),
            // Some server implementations may not return credentials unless explicitly requested
            vend_credentials: Some(true),
            ..Default::default()
        };

        let response = self
            .namespace_client
            .describe_table(request)
            .await
            .map_err(|e| {
                Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to fetch storage options: {}",
                    e
                ))))
            })?;

        Ok(response.storage_options)
    }

    fn provider_id(&self) -> String {
        format!(
            "LanceNamespaceStorageOptionsProvider {{ namespace_client: {}, table_id: {:?} }}",
            self.namespace_client.namespace_id(),
            self.table_id
        )
    }
}

/// Prefix marking a storage option as scoped to a single registered base path.
///
/// A key of the form `base_<id>.<key>` applies `<key>` only to the base path
/// with manifest id `<id>`, overriding the shared (unscoped) options for that
/// base. For example `base_1.account_key = abc` gives the base with id 1 the
/// option `account_key = abc` while it inherits every unscoped option.
pub const BASE_SCOPED_OPTION_PREFIX: &str = "base_";

/// Parse a base-scoped storage option key of the form `base_<id>.<key>`.
///
/// Returns `Some((base_id, key))` only for keys that match the convention
/// exactly: the `base_` prefix, an all-digit u32 base id, a `.` separator, and
/// a non-empty remainder. Any other key (e.g. `base_url`, `base_x.key`,
/// `base_1.`) is not base-scoped.
pub fn parse_base_scoped_key(key: &str) -> Option<(u32, &str)> {
    let rest = key.strip_prefix(BASE_SCOPED_OPTION_PREFIX)?;
    let (id_str, scoped_key) = rest.split_once('.')?;
    if scoped_key.is_empty() || id_str.is_empty() || !id_str.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let id = id_str.parse::<u32>().ok()?;
    Some((id, scoped_key))
}

/// Returns true if any key in `options` is base-scoped (`base_<id>.<key>`).
pub fn has_base_scoped_options(options: &HashMap<String, String>) -> bool {
    options
        .keys()
        .any(|key| parse_base_scoped_key(key).is_some())
}

/// Resolve the effective storage options for one base path scope.
///
/// All base-scoped keys are removed from the result. When `base_id` is
/// `Some(id)`, entries scoped to that id are overlaid on the unscoped options,
/// adding or overriding keys. `None` resolves the default scope (the primary
/// dataset base), which simply drops every base-scoped entry.
pub fn resolve_base_scoped_options(
    options: &HashMap<String, String>,
    base_id: Option<u32>,
) -> HashMap<String, String> {
    let mut resolved = HashMap::with_capacity(options.len());
    let mut overrides = Vec::new();
    for (key, value) in options {
        match parse_base_scoped_key(key) {
            Some((id, scoped_key)) => {
                if Some(id) == base_id {
                    overrides.push((scoped_key.to_string(), value.clone()));
                }
            }
            None => {
                resolved.insert(key.clone(), value.clone());
            }
        }
    }
    resolved.extend(overrides);
    resolved
}

/// A [`StorageOptionsProvider`] that resolves another accessor's options for a
/// single base path scope.
///
/// Fetching through this provider checks expiration for the requested scope.
/// The parent accessor shares refreshed options across all scopes.
#[derive(Debug)]
pub struct BaseScopedStorageOptionsProvider {
    inner: Arc<StorageOptionsAccessor>,
    base_id: Option<u32>,
}

impl BaseScopedStorageOptionsProvider {
    pub fn new(inner: Arc<StorageOptionsAccessor>, base_id: Option<u32>) -> Self {
        Self { inner, base_id }
    }
}

#[async_trait]
impl StorageOptionsProvider for BaseScopedStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
        let options = self
            .inner
            .read_storage_options(StorageOptionsScope::Base(self.base_id), false)
            .await?;
        Ok(Some(options.0))
    }

    async fn force_fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
        let options = self
            .inner
            .read_storage_options(StorageOptionsScope::Base(self.base_id), true)
            .await?;
        Ok(Some(options.0))
    }

    fn provider_id(&self) -> String {
        match self.base_id {
            Some(id) => format!("base-scoped[base_id={}]({})", id, self.inner.accessor_id()),
            None => format!("base-scoped[default]({})", self.inner.accessor_id()),
        }
    }
}

/// Unified access to storage options with automatic caching and refresh
///
/// This struct bundles static storage options with an optional dynamic provider,
/// handling all caching and refresh logic internally. It provides a single entry point
/// for accessing storage options regardless of whether they're static or dynamic.
///
/// # Behavior
///
/// - If only static options are provided, returns those options
/// - If a provider is configured, fetches from provider and caches results
/// - Automatically refreshes cached options before expiration (based on refresh_offset)
/// - Uses `expires_at_millis` key to track expiration
///
/// # Thread Safety
///
/// The accessor is thread-safe and can be shared across multiple tasks.
/// Concurrent reads share a refresh when it makes their options fresh.
pub struct StorageOptionsAccessor {
    /// Initial/fallback static storage options
    initial_options: Option<StorageOptionsSnapshot>,

    /// Optional dynamic provider for refreshing options
    provider: Option<Arc<dyn StorageOptionsProvider>>,

    /// Cached storage options with expiration tracking
    cache: RwLock<Arc<CachedStorageOptions>>,

    /// Serializes provider fetches while allowing fresh scopes to read the cache.
    refresh_gate: Mutex<()>,

    /// Duration before expiry to trigger refresh
    refresh_offset: Duration,

    /// True when this accessor was produced by [`Self::scoped_to_base`]; its
    /// options are already resolved for one base path scope, so scoping again
    /// is a no-op.
    scope_resolved: bool,

    /// Dynamic scopes use their parent's cache and freshness, never a resolved snapshot.
    scoped_provider: Option<Arc<BaseScopedStorageOptionsProvider>>,
}

impl fmt::Debug for StorageOptionsAccessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StorageOptionsAccessor")
            .field("has_initial_options", &self.initial_options.is_some())
            .field("has_provider", &self.provider.is_some())
            .field("refresh_offset", &self.refresh_offset)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StorageOptionsScope {
    Raw,
    Base(Option<u32>),
}

impl StorageOptionsScope {
    fn resolve(self, options: &HashMap<String, String>) -> Cow<'_, HashMap<String, String>> {
        match self {
            Self::Raw => Cow::Borrowed(options),
            Self::Base(base_id) => Cow::Owned(resolve_base_scoped_options(options, base_id)),
        }
    }
}

#[derive(Debug, Default)]
struct CachedStorageOptions {
    vended: Option<Arc<StorageOptionsSnapshot>>,
    absence: Option<CachedAbsence>,
}

#[derive(Debug)]
struct CachedAbsence {
    retry_after: Instant,
    /// Scopes already expired at this point can use fallback under the same deadline.
    recorded_at_millis: u64,
    /// Scopes that selected fallback during an early or forced refresh.
    fallback_scopes: HashSet<StorageOptionsScope>,
}

#[derive(Debug)]
struct StorageOptionsSnapshot {
    options: HashMap<String, String>,
    /// Earliest expiration across the raw map, including every base.
    raw_expires_at_millis: Option<u64>,
}

impl StorageOptionsSnapshot {
    fn new(options: HashMap<String, String>) -> Self {
        let raw_expires_at_millis = options
            .iter()
            .filter(|(key, _)| {
                key.as_str() == EXPIRES_AT_MILLIS_KEY
                    || matches!(
                        parse_base_scoped_key(key),
                        Some((_, scoped_key)) if scoped_key == EXPIRES_AT_MILLIS_KEY
                    )
            })
            .filter_map(|(_, value)| value.parse::<u64>().ok())
            .min();
        Self {
            options,
            raw_expires_at_millis,
        }
    }

    fn view(&self, scope: StorageOptionsScope) -> (Cow<'_, HashMap<String, String>>, Option<u64>) {
        let options = scope.resolve(&self.options);
        let expiry = match scope {
            StorageOptionsScope::Raw => self.raw_expires_at_millis,
            StorageOptionsScope::Base(_) => options
                .get(EXPIRES_AT_MILLIS_KEY)
                .and_then(|value| value.parse().ok()),
        };
        (options, expiry)
    }
}

impl StorageOptionsAccessor {
    /// Extract refresh offset from storage options, or use default
    fn extract_refresh_offset(options: &HashMap<String, String>) -> Duration {
        options
            .get(REFRESH_OFFSET_MILLIS_KEY)
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_millis(DEFAULT_REFRESH_OFFSET_MILLIS))
    }

    /// Create an accessor with only static options (no refresh capability)
    ///
    /// The returned accessor will always return the provided options.
    /// This is useful when credentials don't expire or are managed externally.
    pub fn with_static_options(options: HashMap<String, String>) -> Self {
        let refresh_offset = Self::extract_refresh_offset(&options);

        Self {
            initial_options: Some(StorageOptionsSnapshot::new(options)),
            provider: None,
            cache: RwLock::new(Arc::default()),
            refresh_gate: Mutex::new(()),
            refresh_offset,
            scope_resolved: false,
            scoped_provider: None,
        }
    }

    /// Create an accessor with a dynamic provider (no initial options)
    ///
    /// The accessor will fetch from the provider on first access and cache
    /// the results. Refresh happens automatically before expiration.
    /// Uses the default refresh offset (60 seconds) until options are fetched.
    ///
    /// # Arguments
    /// * `provider` - The storage options provider for fetching fresh options
    pub fn with_provider(provider: Arc<dyn StorageOptionsProvider>) -> Self {
        Self {
            initial_options: None,
            provider: Some(provider),
            cache: RwLock::new(Arc::default()),
            refresh_gate: Mutex::new(()),
            refresh_offset: Duration::from_millis(DEFAULT_REFRESH_OFFSET_MILLIS),
            scope_resolved: false,
            scoped_provider: None,
        }
    }

    /// Create an accessor with initial options and a dynamic provider
    ///
    /// Initial options are used until they expire, then the provider is called.
    /// This avoids an immediate fetch when initial credentials are still valid.
    /// The `refresh_offset_millis` key in initial_options controls refresh timing.
    ///
    /// # Arguments
    /// * `initial_options` - Initial storage options to cache
    /// * `provider` - The storage options provider for refreshing
    pub fn with_initial_and_provider(
        initial_options: HashMap<String, String>,
        provider: Arc<dyn StorageOptionsProvider>,
    ) -> Self {
        let refresh_offset = Self::extract_refresh_offset(&initial_options);

        Self {
            initial_options: Some(StorageOptionsSnapshot::new(initial_options)),
            provider: Some(provider),
            cache: RwLock::new(Arc::default()),
            refresh_gate: Mutex::new(()),
            refresh_offset,
            scope_resolved: false,
            scoped_provider: None,
        }
    }

    /// Get current valid storage options
    ///
    /// A scoped accessor checks the effective expiry for its base. A raw accessor
    /// checks the earliest expiry across the map. Refresh uses the configured offset.
    /// If the provider returns `None`, the accessor uses the initial options or defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider fetch fails or the requested fallback has expired.
    pub async fn get_storage_options(&self) -> Result<super::StorageOptions> {
        self.read_storage_options(StorageOptionsScope::Raw, false)
            .await
    }

    /// Fetch fresh options from the provider and update the cache.
    ///
    /// This bypasses the cache for callers that need to validate provider-vended
    /// credentials even when initial metadata has no expiration. The force
    /// propagates through provider chains (e.g. base-scoped wrappers), so the
    /// origin provider is re-queried even when intermediate caches are valid.
    pub(crate) async fn refresh_storage_options(&self) -> Result<super::StorageOptions> {
        self.read_storage_options(StorageOptionsScope::Raw, true)
            .await
    }

    async fn read_storage_options(
        &self,
        scope: StorageOptionsScope,
        force: bool,
    ) -> Result<super::StorageOptions> {
        if let Some(scoped) = &self.scoped_provider {
            let options = scoped
                .inner
                .read_cached_options(StorageOptionsScope::Base(scoped.base_id), force)
                .await?;
            return Ok(match scope {
                StorageOptionsScope::Raw => options,
                StorageOptionsScope::Base(base_id) => {
                    super::StorageOptions(resolve_base_scoped_options(&options.0, base_id))
                }
            });
        }

        self.read_cached_options(scope, force).await
    }

    async fn read_cached_options(
        &self,
        scope: StorageOptionsScope,
        force: bool,
    ) -> Result<super::StorageOptions> {
        let Some(provider) = &self.provider else {
            return Ok(super::StorageOptions(
                self.fallback_view(scope).0.into_owned(),
            ));
        };

        if !force && let Some(options) = self.fresh_options(scope).await {
            return Ok(options);
        }

        let _refresh = self.refresh_gate.lock().await;
        // A different scope may have refreshed the shared map while we waited.
        if !force && let Some(options) = self.fresh_options(scope).await {
            return Ok(options);
        }

        log::debug!(
            "Refreshing storage options from provider: {}",
            provider.provider_id()
        );
        let fetched = if force {
            provider.force_fetch_storage_options().await
        } else {
            provider.fetch_storage_options().await
        }
        .map_err(|e| {
            Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to fetch storage options: {}",
                e
            ))))
        })?;

        let (options, cache) = match fetched {
            Some(options) => {
                let snapshot = Arc::new(StorageOptionsSnapshot::new(options));
                let resolved = snapshot.view(scope).0.into_owned();
                let cache = CachedStorageOptions {
                    vended: Some(snapshot),
                    absence: None,
                };
                (resolved, cache)
            }
            None => {
                let (options, expiry) = self.fallback_view(scope);
                let now_millis = Self::now_millis();
                if let Some(expiration) = expiry
                    && expiration <= now_millis
                {
                    return Err(Error::invalid_input(format!(
                        "Provider returned no options and initial credentials expired at {expiration}"
                    )));
                }

                // Preserve the vended map for siblings and share the retry deadline.
                let previous = self.cache.read().await.clone();
                let mut fallback_scopes = previous
                    .absence
                    .as_ref()
                    .map(|absence| absence.fallback_scopes.clone())
                    .unwrap_or_default();
                if let Some(vended) = &previous.vended
                    && vended
                        .view(scope)
                        .1
                        .is_none_or(|expiry| expiry > now_millis)
                {
                    fallback_scopes.insert(scope);
                }
                let cache = CachedStorageOptions {
                    vended: previous.vended.clone(),
                    absence: Some(CachedAbsence {
                        retry_after: Instant::now() + Duration::from_secs(60),
                        recorded_at_millis: now_millis,
                        fallback_scopes,
                    }),
                };
                (options.into_owned(), cache)
            }
        };
        *self.cache.write().await = Arc::new(cache);
        Ok(super::StorageOptions(options))
    }

    fn fallback_view(
        &self,
        scope: StorageOptionsScope,
    ) -> (Cow<'_, HashMap<String, String>>, Option<u64>) {
        self.initial_options.as_ref().map_or_else(
            || (Cow::Owned(HashMap::new()), None),
            |snapshot| snapshot.view(scope),
        )
    }

    async fn fresh_options(&self, scope: StorageOptionsScope) -> Option<super::StorageOptions> {
        let cache = self.cache.read().await.clone();
        let vended = cache
            .vended
            .as_ref()
            .map(|snapshot| snapshot.view(scope))
            .filter(|(_, expiry)| {
                !cache.absence.as_ref().is_some_and(|absence| {
                    absence.fallback_scopes.contains(&scope)
                        || expiry.is_some_and(|expiry| expiry <= absence.recorded_at_millis)
                })
            });

        let ((options, expiry), refresh_offset) = if let Some(view) = vended {
            (view, self.refresh_offset)
        } else {
            let refresh_offset = match &cache.absence {
                // A fallback retries at the shared deadline or its actual expiry.
                Some(absence) if Instant::now() < absence.retry_after => Duration::ZERO,
                None if self.initial_options.is_some() => self.refresh_offset,
                _ => return None,
            };
            (self.fallback_view(scope), refresh_offset)
        };

        if expiry.is_some_and(|expiry| {
            Self::now_millis().saturating_add(refresh_offset.as_millis() as u64) >= expiry
        }) {
            return None;
        }
        Some(super::StorageOptions(options.into_owned()))
    }

    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Get the initial storage options without refresh
    ///
    /// Returns the initial options that were provided when creating the accessor.
    /// This does not trigger any refresh, even if the options have expired.
    pub fn initial_storage_options(&self) -> Option<&HashMap<String, String>> {
        self.initial_options
            .as_ref()
            .map(|snapshot| &snapshot.options)
    }

    /// Get the accessor ID for equality/hashing
    ///
    /// Returns the provider_id if a provider exists, otherwise generates
    /// a stable ID from the initial options hash.
    pub fn accessor_id(&self) -> String {
        if let Some(provider) = &self.provider {
            provider.provider_id()
        } else if let Some(initial) = self.initial_storage_options() {
            // Generate a stable ID from initial options
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            let mut keys: Vec<_> = initial.keys().collect();
            keys.sort();
            for key in keys {
                key.hash(&mut hasher);
                initial.get(key).hash(&mut hasher);
            }
            format!("static_options_{:x}", hasher.finish())
        } else {
            "empty_accessor".to_string()
        }
    }

    /// Resolve this accessor for a single base path scope.
    ///
    /// Storage options may carry base-scoped entries (`base_<id>.<key>`) that
    /// apply only to one registered base path. The returned accessor resolves
    /// options for `base_id`: entries scoped to that base overlay the unscoped
    /// defaults, and all other scoped entries are dropped. `None` resolves the
    /// default scope used for the primary dataset base.
    ///
    /// A static accessor whose options contain no base-scoped entries is
    /// returned unchanged, preserving accessor identity (and thus object store
    /// registry cache keys). A provider-backed accessor is always wrapped
    /// through [`BaseScopedStorageOptionsProvider`] — fetched options may vend
    /// base-scoped entries at any refresh, even when the initial options carry
    /// none — so refreshed options are re-resolved for the scope on every
    /// fetch. Dynamic scopes read through the parent's cache so bounded absence
    /// responses retain their original retry deadline. Accessors already
    /// produced by this method are returned unchanged.
    pub fn scoped_to_base(self: &Arc<Self>, base_id: Option<u32>) -> Arc<Self> {
        if self.scope_resolved {
            return self.clone();
        }
        if self.has_provider() {
            let provider = Arc::new(BaseScopedStorageOptionsProvider::new(self.clone(), base_id));
            let mut scoped = match self.initial_storage_options() {
                Some(initial) => Self::with_initial_and_provider(
                    resolve_base_scoped_options(initial, base_id),
                    provider.clone(),
                ),
                None => Self::with_provider(provider.clone()),
            };
            scoped.scope_resolved = true;
            scoped.scoped_provider = Some(provider);
            Arc::new(scoped)
        } else {
            match self.initial_storage_options() {
                Some(initial) if has_base_scoped_options(initial) => {
                    let mut scoped =
                        Self::with_static_options(resolve_base_scoped_options(initial, base_id));
                    scoped.scope_resolved = true;
                    Arc::new(scoped)
                }
                // Static options never change, so there is nothing to scope.
                _ => self.clone(),
            }
        }
    }

    /// Check if this accessor has a dynamic provider
    pub fn has_provider(&self) -> bool {
        self.provider.is_some()
    }

    /// Get the refresh offset duration
    pub fn refresh_offset(&self) -> Duration {
        self.refresh_offset
    }

    /// Get the storage options provider, if any
    pub fn provider(&self) -> Option<&Arc<dyn StorageOptionsProvider>> {
        self.provider.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mock_instant::thread_local::MockClock;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::sync::{Notify, Semaphore};
    #[derive(Debug)]
    struct AbsentProvider(std::sync::atomic::AtomicUsize);

    #[async_trait]
    impl StorageOptionsProvider for AbsentProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(None)
        }
        fn provider_id(&self) -> String {
            "absent".into()
        }
    }

    #[tokio::test]
    async fn absence_is_cached_until_deadline_or_force_refresh() {
        let provider = Arc::new(AbsentProvider(std::sync::atomic::AtomicUsize::new(0)));
        let accessor = StorageOptionsAccessor::with_provider(provider.clone());
        futures::future::join_all((0..20).map(|_| accessor.get_storage_options()))
            .await
            .into_iter()
            .for_each(|result| assert!(result.unwrap().0.is_empty()));
        assert_eq!(provider.0.load(std::sync::atomic::Ordering::SeqCst), 1);
        accessor.refresh_storage_options().await.unwrap();
        assert_eq!(provider.0.load(std::sync::atomic::Ordering::SeqCst), 2);
        MockClock::advance(Duration::from_secs(60));
        accessor.get_storage_options().await.unwrap();
        assert_eq!(provider.0.load(std::sync::atomic::Ordering::SeqCst), 3);
        let expired = StorageOptionsAccessor::with_initial_and_provider(
            HashMap::from([(EXPIRES_AT_MILLIS_KEY.into(), "1".into())]),
            provider,
        );
        MockClock::set_system_time(Duration::from_millis(2));
        assert!(
            expired
                .get_storage_options()
                .await
                .unwrap_err()
                .to_string()
                .contains("expired")
        );
    }

    #[tokio::test]
    async fn absent_refresh_preserves_fallback_expiration() {
        MockClock::set_system_time(Duration::from_secs(100));
        let provider = Arc::new(AbsentProvider(std::sync::atomic::AtomicUsize::new(0)));
        let accessor = StorageOptionsAccessor::with_initial_and_provider(
            HashMap::from([(EXPIRES_AT_MILLIS_KEY.into(), "101000".into())]),
            provider.clone(),
        );
        for _ in 0..20 {
            assert_eq!(
                accessor.get_storage_options().await.unwrap().0[EXPIRES_AT_MILLIS_KEY],
                "101000"
            );
        }
        assert_eq!(provider.0.load(std::sync::atomic::Ordering::SeqCst), 1);
        MockClock::set_system_time(Duration::from_secs(101));
        assert!(
            accessor
                .get_storage_options()
                .await
                .unwrap_err()
                .to_string()
                .contains("expired")
        );
    }
    #[derive(Debug)]
    struct MockStorageOptionsProvider {
        call_count: Arc<RwLock<usize>>,
        expires_in_millis: Option<u64>,
    }

    impl MockStorageOptionsProvider {
        fn new(expires_in_millis: Option<u64>) -> Self {
            Self {
                call_count: Arc::new(RwLock::new(0)),
                expires_in_millis,
            }
        }

        async fn get_call_count(&self) -> usize {
            *self.call_count.read().await
        }
    }

    #[async_trait]
    impl StorageOptionsProvider for MockStorageOptionsProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            let count = {
                let mut c = self.call_count.write().await;
                *c += 1;
                *c
            };

            let mut options = HashMap::from([
                ("aws_access_key_id".to_string(), format!("AKID_{}", count)),
                (
                    "aws_secret_access_key".to_string(),
                    format!("SECRET_{}", count),
                ),
                ("aws_session_token".to_string(), format!("TOKEN_{}", count)),
            ]);

            if let Some(expires_in) = self.expires_in_millis {
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                let expires_at = now_ms + expires_in;
                options.insert(EXPIRES_AT_MILLIS_KEY.to_string(), expires_at.to_string());
            }

            Ok(Some(options))
        }

        fn provider_id(&self) -> String {
            let ptr = Arc::as_ptr(&self.call_count) as usize;
            format!("MockStorageOptionsProvider {{ id: {} }}", ptr)
        }
    }

    #[tokio::test]
    async fn test_static_options_only() {
        let options = HashMap::from([
            ("key1".to_string(), "value1".to_string()),
            ("key2".to_string(), "value2".to_string()),
        ]);
        let accessor = StorageOptionsAccessor::with_static_options(options.clone());

        let result = accessor.get_storage_options().await.unwrap();
        assert_eq!(result.0, options);
        assert!(!accessor.has_provider());
        assert_eq!(accessor.initial_storage_options(), Some(&options));
    }

    #[tokio::test]
    async fn test_provider_only() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let mock_provider = Arc::new(MockStorageOptionsProvider::new(Some(600_000)));
        let accessor = StorageOptionsAccessor::with_provider(mock_provider.clone());

        let result = accessor.get_storage_options().await.unwrap();
        assert!(result.0.contains_key("aws_access_key_id"));
        assert_eq!(result.0.get("aws_access_key_id").unwrap(), "AKID_1");
        assert!(accessor.has_provider());
        assert_eq!(accessor.initial_storage_options(), None);
        assert_eq!(mock_provider.get_call_count().await, 1);
    }

    #[tokio::test]
    async fn test_initial_and_provider_uses_initial_first() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;
        let expires_at = now_ms + 600_000; // 10 minutes from now

        let initial = HashMap::from([
            ("aws_access_key_id".to_string(), "INITIAL_KEY".to_string()),
            (
                "aws_secret_access_key".to_string(),
                "INITIAL_SECRET".to_string(),
            ),
            (EXPIRES_AT_MILLIS_KEY.to_string(), expires_at.to_string()),
        ]);
        let mock_provider = Arc::new(MockStorageOptionsProvider::new(Some(600_000)));

        let accessor = StorageOptionsAccessor::with_initial_and_provider(
            initial.clone(),
            mock_provider.clone(),
        );

        // First call uses initial
        let result = accessor.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("aws_access_key_id").unwrap(), "INITIAL_KEY");
        assert_eq!(mock_provider.get_call_count().await, 0); // Provider not called yet
    }

    #[tokio::test]
    async fn test_caching_and_refresh() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let mock_provider = Arc::new(MockStorageOptionsProvider::new(Some(600_000))); // 10 min expiry
        // Use with_initial_and_provider to set custom refresh_offset_millis (5 min = 300000ms)
        let now_ms = MockClock::system_time().as_millis() as u64;
        let expires_at = now_ms + 600_000; // 10 minutes from now
        let initial = HashMap::from([
            (EXPIRES_AT_MILLIS_KEY.to_string(), expires_at.to_string()),
            (REFRESH_OFFSET_MILLIS_KEY.to_string(), "300000".to_string()), // 5 min refresh offset
        ]);
        let accessor =
            StorageOptionsAccessor::with_initial_and_provider(initial, mock_provider.clone());

        // First call uses initial cached options
        let result = accessor.get_storage_options().await.unwrap();
        assert!(result.0.contains_key(EXPIRES_AT_MILLIS_KEY));
        assert_eq!(mock_provider.get_call_count().await, 0);

        // Advance time to 6 minutes - should trigger refresh (within 5 min refresh offset)
        MockClock::set_system_time(Duration::from_secs(100_000 + 360));
        let result = accessor.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("aws_access_key_id").unwrap(), "AKID_1");
        assert_eq!(mock_provider.get_call_count().await, 1);
    }

    #[tokio::test]
    async fn test_expired_initial_triggers_refresh() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let now_ms = MockClock::system_time().as_millis() as u64;
        let expired_time = now_ms - 1_000; // Expired 1 second ago

        let initial = HashMap::from([
            ("aws_access_key_id".to_string(), "EXPIRED_KEY".to_string()),
            (EXPIRES_AT_MILLIS_KEY.to_string(), expired_time.to_string()),
        ]);
        let mock_provider = Arc::new(MockStorageOptionsProvider::new(Some(600_000)));

        let accessor =
            StorageOptionsAccessor::with_initial_and_provider(initial, mock_provider.clone());

        // Should fetch from provider since initial is expired
        let result = accessor.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("aws_access_key_id").unwrap(), "AKID_1");
        assert_eq!(mock_provider.get_call_count().await, 1);
    }

    #[tokio::test]
    async fn test_accessor_id_with_provider() {
        let mock_provider = Arc::new(MockStorageOptionsProvider::new(None));
        let accessor = StorageOptionsAccessor::with_provider(mock_provider);

        let id = accessor.accessor_id();
        assert!(id.starts_with("MockStorageOptionsProvider"));
    }

    #[tokio::test]
    async fn test_accessor_id_static() {
        let options = HashMap::from([("key".to_string(), "value".to_string())]);
        let accessor = StorageOptionsAccessor::with_static_options(options);

        let id = accessor.accessor_id();
        assert!(id.starts_with("static_options_"));
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        // Create a mock provider with far future expiration
        let mock_provider = Arc::new(MockStorageOptionsProvider::new(Some(9999999999999)));

        let accessor = Arc::new(StorageOptionsAccessor::with_provider(mock_provider.clone()));

        // Spawn 10 concurrent tasks that all try to get options at the same time
        let mut handles = vec![];
        for i in 0..10 {
            let acc = accessor.clone();
            let handle = tokio::spawn(async move {
                let result = acc.get_storage_options().await.unwrap();
                assert_eq!(result.0.get("aws_access_key_id").unwrap(), "AKID_1");
                i
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Verify all 10 tasks completed successfully
        assert_eq!(results.len(), 10);

        // The provider should have been called exactly once
        let call_count = mock_provider.get_call_count().await;
        assert_eq!(
            call_count, 1,
            "Provider should be called exactly once despite concurrent access"
        );
    }

    #[tokio::test]
    async fn test_no_expiration_never_refreshes() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        let mock_provider = Arc::new(MockStorageOptionsProvider::new(None)); // No expiration
        let accessor = StorageOptionsAccessor::with_provider(mock_provider.clone());

        // First call fetches
        accessor.get_storage_options().await.unwrap();
        assert_eq!(mock_provider.get_call_count().await, 1);

        // Advance time significantly
        MockClock::set_system_time(Duration::from_secs(200_000));

        // Should still use cached options
        accessor.get_storage_options().await.unwrap();
        assert_eq!(mock_provider.get_call_count().await, 1);
    }

    #[test]
    fn test_parse_base_scoped_key() {
        assert_eq!(
            parse_base_scoped_key("base_1.account_key"),
            Some((1, "account_key"))
        );
        assert_eq!(
            parse_base_scoped_key("base_12.headers.x-ms-version"),
            Some((12, "headers.x-ms-version"))
        );
        assert_eq!(parse_base_scoped_key("base_0.region"), Some((0, "region")));

        // Not base-scoped keys
        assert_eq!(parse_base_scoped_key("account_key"), None);
        assert_eq!(parse_base_scoped_key("base_url"), None);
        assert_eq!(parse_base_scoped_key("base_hot.account_key"), None);
        assert_eq!(parse_base_scoped_key("base_1x.account_key"), None);
        assert_eq!(parse_base_scoped_key("base_+1.account_key"), None);
        assert_eq!(parse_base_scoped_key("base_.account_key"), None);
        assert_eq!(parse_base_scoped_key("base_1."), None);
        assert_eq!(parse_base_scoped_key("base_1"), None);
        // Overflows u32
        assert_eq!(parse_base_scoped_key("base_4294967296.key"), None);
    }

    #[test]
    fn test_resolve_base_scoped_options() {
        let options = HashMap::from([
            ("region".to_string(), "us-east-1".to_string()),
            ("account_key".to_string(), "shared-key".to_string()),
            ("base_1.account_key".to_string(), "base1-key".to_string()),
            ("base_2.account_key".to_string(), "base2-key".to_string()),
            ("base_2.endpoint".to_string(), "http://b2".to_string()),
        ]);
        assert!(has_base_scoped_options(&options));

        let base1 = resolve_base_scoped_options(&options, Some(1));
        assert_eq!(
            base1,
            HashMap::from([
                ("region".to_string(), "us-east-1".to_string()),
                ("account_key".to_string(), "base1-key".to_string()),
            ])
        );

        let base2 = resolve_base_scoped_options(&options, Some(2));
        assert_eq!(
            base2,
            HashMap::from([
                ("region".to_string(), "us-east-1".to_string()),
                ("account_key".to_string(), "base2-key".to_string()),
                ("endpoint".to_string(), "http://b2".to_string()),
            ])
        );

        // A base without scoped entries inherits only the unscoped options
        let base3 = resolve_base_scoped_options(&options, Some(3));
        assert_eq!(
            base3,
            HashMap::from([
                ("region".to_string(), "us-east-1".to_string()),
                ("account_key".to_string(), "shared-key".to_string()),
            ])
        );

        // The default scope drops every scoped entry
        let default = resolve_base_scoped_options(&options, None);
        assert_eq!(default, base3);

        assert!(!has_base_scoped_options(&HashMap::from([(
            "account_key".to_string(),
            "shared-key".to_string()
        )])));
    }

    #[tokio::test]
    async fn test_scoped_to_base_identity_and_idempotency() {
        // Static accessors without scoped keys are returned unchanged.
        let accessor = Arc::new(StorageOptionsAccessor::with_static_options(HashMap::from(
            [("account_key".to_string(), "shared-key".to_string())],
        )));
        assert!(Arc::ptr_eq(&accessor.scoped_to_base(Some(1)), &accessor));
        assert!(Arc::ptr_eq(&accessor.scoped_to_base(None), &accessor));

        // Scoping an already-scoped accessor is a no-op (the registry choke
        // point re-applies the default scope to every params it sees).
        let scoped = Arc::new(StorageOptionsAccessor::with_static_options(HashMap::from(
            [
                ("account_key".to_string(), "shared-key".to_string()),
                ("base_1.account_key".to_string(), "base1-key".to_string()),
            ],
        )))
        .scoped_to_base(Some(1));
        assert!(Arc::ptr_eq(&scoped.scoped_to_base(None), &scoped));

        let provider_scoped = Arc::new(StorageOptionsAccessor::with_provider(Arc::new(
            MockStorageOptionsProvider::new(None),
        )))
        .scoped_to_base(Some(1));
        assert!(Arc::ptr_eq(
            &provider_scoped.scoped_to_base(None),
            &provider_scoped
        ));
    }

    #[tokio::test]
    async fn test_scoped_to_base_provider_only_resolves_vended_options() {
        MockClock::set_system_time(Duration::from_secs(100_000));

        // No initial options: scoped entries arrive only through the provider.
        let provider = Arc::new(MockBaseScopedVendingProvider {
            call_count: Arc::new(RwLock::new(0)),
            expires_in_millis: 600_000,
        });
        let parent = Arc::new(StorageOptionsAccessor::with_provider(provider.clone()));

        let base1 = parent.scoped_to_base(Some(1));
        assert!(!Arc::ptr_eq(&base1, &parent));
        let result = base1.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("account_key").unwrap(), "BASE1_1");
        assert!(!result.0.contains_key("base_1.account_key"));

        let default = parent.scoped_to_base(None);
        let result = default.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("account_key").unwrap(), "SHARED_1");
        assert!(!result.0.contains_key("base_1.account_key"));

        // Both scopes were served from one parent fetch.
        assert_eq!(*provider.call_count.read().await, 1);
    }

    #[rstest::rstest]
    #[case::absence(Ok(None))]
    #[case::provider_error(Err(Error::io("injected provider failure")))]
    #[case::renewal(Ok(Some(scoped_block_sizes("16384", "32768", "65536", 700_000))))]
    #[tokio::test]
    async fn scoped_refresh_isolates_expired_base(
        #[case] response: Result<Option<HashMap<String, String>>>,
        #[values(false, true)] direct_provider: bool,
    ) {
        MockClock::set_system_time(Duration::from_secs(100_121));
        let provider = Arc::new(TransitionProvider::default());
        let is_error = response.is_err();
        provider.fail.store(is_error, Ordering::SeqCst);
        *provider.options.write().await = response.unwrap_or_default();
        let is_renewal = provider.options.read().await.is_some();
        let parent = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            scoped_block_sizes("2048", "4096", "8192", 120_000),
            provider.clone(),
        ));
        let primary = parent.scoped_to_base(None);
        let base8 = BaseScopedStorageOptionsProvider::new(parent.clone(), Some(8));
        assert_eq!(
            primary.get_storage_options().await.unwrap().0["block_size"],
            "2048"
        );
        assert_eq!(
            base8.fetch_storage_options().await.unwrap().unwrap()["block_size"],
            "8192"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);

        let result = if direct_provider {
            BaseScopedStorageOptionsProvider::new(parent.clone(), Some(7))
                .fetch_storage_options()
                .await
                .map(|options| options.unwrap())
        } else {
            parent
                .scoped_to_base(Some(7))
                .get_storage_options()
                .await
                .map(|options| options.0)
        };
        if is_renewal {
            assert_eq!(result.unwrap()["block_size"], "32768");
        } else {
            let error = result.unwrap_err();
            if is_error {
                assert!(matches!(error, Error::IO { .. }));
                assert!(error.to_string().contains("injected provider failure"));
            } else {
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(error.to_string().contains("expired"));
            }
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            primary.get_storage_options().await.unwrap().0["block_size"],
            if is_renewal { "16384" } else { "2048" }
        );
        assert_eq!(
            base8.fetch_storage_options().await.unwrap().unwrap()["block_size"],
            if is_renewal { "65536" } else { "8192" }
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn raw_map_and_base_use_parent_refresh_offset() {
        MockClock::set_system_time(Duration::from_secs(100_100));
        let provider = Arc::new(TransitionProvider::default());
        provider.fail.store(true, Ordering::SeqCst);
        let mut initial = scoped_block_sizes("2048", "4096", "8192", 120_000);
        initial.insert(REFRESH_OFFSET_MILLIS_KEY.into(), "10000".into());
        initial.insert("base_7.refresh_offset_millis".into(), "1".into());
        let parent = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            initial,
            provider.clone(),
        ));
        let base7 = parent.scoped_to_base(Some(7));
        assert_eq!(
            parent.get_storage_options().await.unwrap().0["base_7.block_size"],
            "4096"
        );
        assert_eq!(
            base7.get_storage_options().await.unwrap().0["block_size"],
            "4096"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);

        MockClock::advance_system_time(Duration::from_secs(10));
        for accessor in [&parent, &base7] {
            let error = accessor.get_storage_options().await.unwrap_err();
            assert!(matches!(error, Error::IO { .. }));
            assert!(error.to_string().contains("injected provider failure"));
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    fn scoped_block_sizes(
        primary: &str,
        base7: &str,
        base8: &str,
        base7_expiry_offset: u64,
    ) -> HashMap<String, String> {
        HashMap::from([
            ("block_size".into(), primary.into()),
            ("base_7.block_size".into(), base7.into()),
            ("base_8.block_size".into(), base8.into()),
            (EXPIRES_AT_MILLIS_KEY.into(), "101000000".into()),
            (
                format!("base_7.{EXPIRES_AT_MILLIS_KEY}"),
                (100_000_000 + base7_expiry_offset).to_string(),
            ),
        ])
    }

    #[tokio::test]
    async fn test_scoped_to_base_static() {
        let accessor = Arc::new(StorageOptionsAccessor::with_static_options(HashMap::from(
            [
                ("account_key".to_string(), "shared-key".to_string()),
                ("base_1.account_key".to_string(), "base1-key".to_string()),
            ],
        )));

        let base1 = accessor.scoped_to_base(Some(1));
        let result = base1.get_storage_options().await.unwrap();
        assert_eq!(
            result.0,
            HashMap::from([("account_key".to_string(), "base1-key".to_string())])
        );
        assert!(!base1.has_provider());

        let default = accessor.scoped_to_base(None);
        let result = default.get_storage_options().await.unwrap();
        assert_eq!(
            result.0,
            HashMap::from([("account_key".to_string(), "shared-key".to_string())])
        );

        // Scoped accessor ids are stable across derivations and distinct per scope
        assert_eq!(
            accessor.scoped_to_base(Some(1)).accessor_id(),
            base1.accessor_id()
        );
        assert_ne!(base1.accessor_id(), default.accessor_id());
        assert_ne!(base1.accessor_id(), accessor.accessor_id());
    }

    #[derive(Debug)]
    struct MockBaseScopedVendingProvider {
        call_count: Arc<RwLock<usize>>,
        expires_in_millis: u64,
    }

    #[async_trait]
    impl StorageOptionsProvider for MockBaseScopedVendingProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            let count = {
                let mut c = self.call_count.write().await;
                *c += 1;
                *c
            };
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            Ok(Some(HashMap::from([
                ("account_key".to_string(), format!("SHARED_{}", count)),
                ("base_1.account_key".to_string(), format!("BASE1_{}", count)),
                (
                    EXPIRES_AT_MILLIS_KEY.to_string(),
                    (now_ms + self.expires_in_millis).to_string(),
                ),
            ])))
        }

        fn provider_id(&self) -> String {
            "MockBaseScopedVendingProvider".to_string()
        }
    }

    #[tokio::test]
    async fn test_scoped_to_base_refreshes_through_parent() {
        MockClock::set_system_time(Duration::from_secs(100_000));
        let now_ms = MockClock::system_time().as_millis() as u64;

        let provider = Arc::new(MockBaseScopedVendingProvider {
            call_count: Arc::new(RwLock::new(0)),
            expires_in_millis: 600_000,
        });
        let initial = HashMap::from([
            ("account_key".to_string(), "SHARED_0".to_string()),
            ("base_1.account_key".to_string(), "BASE1_0".to_string()),
            (
                EXPIRES_AT_MILLIS_KEY.to_string(),
                (now_ms + 600_000).to_string(),
            ),
        ]);
        let parent = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            initial,
            provider.clone(),
        ));

        let base1 = parent.scoped_to_base(Some(1));
        let default = parent.scoped_to_base(None);
        assert!(base1.has_provider());

        // Initial options are resolved per scope without fetching
        let result = base1.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("account_key").unwrap(), "BASE1_0");
        assert!(!result.0.contains_key("base_1.account_key"));
        let result = default.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("account_key").unwrap(), "SHARED_0");
        assert_eq!(*provider.call_count.read().await, 0);

        // Expire the vended options; the scoped accessor refreshes through the
        // parent and re-resolves the refreshed options for its scope.
        MockClock::set_system_time(Duration::from_secs(100_000 + 601));
        let result = base1.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("account_key").unwrap(), "BASE1_1");
        assert_eq!(*provider.call_count.read().await, 1);

        // The parent refresh is shared: other scopes see it without refetching
        let result = default.get_storage_options().await.unwrap();
        assert_eq!(result.0.get("account_key").unwrap(), "SHARED_1");
        assert_eq!(*provider.call_count.read().await, 1);
    }

    #[tokio::test]
    async fn test_scoped_forced_refresh_reaches_origin_provider() {
        MockClock::set_system_time(Duration::from_secs(100_000));
        let now_ms = MockClock::system_time().as_millis() as u64;

        let provider = Arc::new(MockBaseScopedVendingProvider {
            call_count: Arc::new(RwLock::new(0)),
            expires_in_millis: 600_000,
        });
        let initial = HashMap::from([
            ("account_key".to_string(), "SHARED_0".to_string()),
            ("base_1.account_key".to_string(), "BASE1_0".to_string()),
            (
                EXPIRES_AT_MILLIS_KEY.to_string(),
                (now_ms + 600_000).to_string(),
            ),
        ]);
        let parent = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            initial,
            provider.clone(),
        ));
        let base1 = parent.scoped_to_base(Some(1));

        // A forced refresh must reach the origin provider even though both the
        // scoped and the parent caches are still valid.
        let result = base1.refresh_storage_options().await.unwrap();
        assert_eq!(result.0.get("account_key").unwrap(), "BASE1_1");
        assert_eq!(*provider.call_count.read().await, 1);
    }

    #[derive(Debug, Default)]
    struct TransitionProvider {
        calls: AtomicUsize,
        options: RwLock<Option<HashMap<String, String>>>,
        fail: AtomicBool,
    }

    #[async_trait]
    impl StorageOptionsProvider for TransitionProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                return Err(Error::io("injected provider failure"));
            }
            Ok(self.options.read().await.clone())
        }

        fn provider_id(&self) -> String {
            "transition-provider".into()
        }
    }

    fn vended_options() -> HashMap<String, String> {
        HashMap::from([
            ("token".into(), "shared".into()),
            ("base_7.token".into(), "seven".into()),
            (EXPIRES_AT_MILLIS_KEY.into(), "200000000".into()),
        ])
    }

    #[rstest::rstest]
    #[case::default_scope(None, "shared")]
    #[case::numbered_scope(Some(7), "seven")]
    #[tokio::test]
    async fn new_scope_preserves_parent_absence_deadline(
        #[case] base_id: Option<u32>,
        #[case] expected_token: &str,
        #[values(false, true)] refresh_parent_first: bool,
        #[values(false, true)] force_initial_fetch: bool,
    ) {
        MockClock::set_system_time(Duration::from_secs(100_000));
        let provider = Arc::new(TransitionProvider::default());
        let parent = Arc::new(StorageOptionsAccessor::with_provider(provider.clone()));
        let initial_scope = parent.scoped_to_base(None);
        let initial = if force_initial_fetch {
            initial_scope.refresh_storage_options().await.unwrap()
        } else {
            initial_scope.get_storage_options().await.unwrap()
        };
        assert!(initial.0.is_empty());
        *provider.options.write().await = Some(vended_options());
        MockClock::advance(Duration::from_secs(1));
        MockClock::advance_system_time(Duration::from_secs(1));
        let scope = parent.scoped_to_base(base_id);
        assert!(scope.get_storage_options().await.unwrap().0.is_empty());
        MockClock::advance(Duration::from_millis(58_999));
        MockClock::advance_system_time(Duration::from_millis(58_999));
        assert!(scope.get_storage_options().await.unwrap().0.is_empty());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        MockClock::advance(Duration::from_millis(1));
        MockClock::advance_system_time(Duration::from_millis(1));
        if refresh_parent_first {
            assert_eq!(
                parent.get_storage_options().await.unwrap().0["token"],
                "shared"
            );
        }
        assert_eq!(
            scope
                .get_storage_options()
                .await
                .unwrap()
                .0
                .get("token")
                .map(String::as_str),
            Some(expected_token),
            "new scopes must discover credentials at the parent's original deadline"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[rstest::rstest]
    #[case::default_scope(None, "shared")]
    #[case::numbered_scope(Some(7), "seven")]
    #[tokio::test]
    async fn scope_distinguishes_empty_options_and_forces_parent_refresh(
        #[case] base_id: Option<u32>,
        #[case] expected_token: &str,
        #[values(false, true)] absent: bool,
    ) {
        let provider = Arc::new(TransitionProvider::default());
        if !absent {
            *provider.options.write().await = Some(HashMap::new());
        }
        let parent = Arc::new(StorageOptionsAccessor::with_provider(provider.clone()));
        assert!(parent.get_storage_options().await.unwrap().0.is_empty());
        let scope = parent.scoped_to_base(base_id);
        assert!(scope.get_storage_options().await.unwrap().0.is_empty());
        *provider.options.write().await = Some(vended_options());
        if !absent {
            MockClock::advance(Duration::from_secs(60));
            MockClock::advance_system_time(Duration::from_secs(60));
        }
        assert!(scope.get_storage_options().await.unwrap().0.is_empty());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            scope.refresh_storage_options().await.unwrap().0["token"],
            expected_token
        );
        assert_eq!(
            scope.get_storage_options().await.unwrap().0["token"],
            expected_token
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[rstest::rstest]
    #[case::expired_base(false, 120_000)]
    #[case::early_refresh(false, 170_000)]
    #[case::forced_refresh(true, 300_000)]
    #[tokio::test]
    async fn scoped_absence_preserves_vended_siblings(
        #[case] force: bool,
        #[case] vended_base_expiry: u64,
        #[values(false, true)] provider_fails_at_deadline: bool,
    ) {
        MockClock::set_system_time(Duration::from_secs(100_055));
        let mut initial = scoped_block_sizes("2048", "4096", "8192", 300_000);
        initial.insert(EXPIRES_AT_MILLIS_KEY.into(), "100110000".into());
        let provider = Arc::new(TransitionProvider::default());
        let mut vended = scoped_block_sizes("16384", "32768", "65536", vended_base_expiry);
        vended.insert(
            format!("base_8.{EXPIRES_AT_MILLIS_KEY}"),
            "100250000".into(),
        );
        *provider.options.write().await = Some(vended);
        let parent = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            initial,
            provider.clone(),
        ));
        let primary = parent.scoped_to_base(None);
        let base7 = parent.scoped_to_base(Some(7));
        assert_eq!(
            primary.get_storage_options().await.unwrap().0["block_size"],
            "16384"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);

        MockClock::set_system_time(Duration::from_secs(100_121));
        *provider.options.write().await = None;
        let fallback = if force {
            BaseScopedStorageOptionsProvider::new(parent.clone(), Some(7))
                .force_fetch_storage_options()
                .await
                .unwrap()
                .unwrap()
        } else {
            base7.get_storage_options().await.unwrap().0
        };
        assert_eq!(fallback["block_size"], "4096");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);

        let base8 = parent.scoped_to_base(Some(8));
        let primary_read = primary.get_storage_options().await;
        let base8_read = base8.get_storage_options().await;
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(primary_read.unwrap().0["block_size"], "16384");
        assert_eq!(base8_read.unwrap().0["block_size"], "65536");
        assert_eq!(
            base7.get_storage_options().await.unwrap().0["block_size"],
            "4096"
        );

        MockClock::advance(Duration::from_millis(59_999));
        MockClock::advance_system_time(Duration::from_millis(59_999));
        assert_eq!(
            base7.get_storage_options().await.unwrap().0["block_size"],
            "4096"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);

        MockClock::advance(Duration::from_millis(1));
        MockClock::advance_system_time(Duration::from_millis(1));
        provider
            .fail
            .store(provider_fails_at_deadline, Ordering::SeqCst);
        let primary_read = primary.get_storage_options().await;
        let base8_read = base8.get_storage_options().await;
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(primary_read.unwrap().0["block_size"], "16384");
        assert_eq!(base8_read.unwrap().0["block_size"], "65536");

        let base7_read = base7.get_storage_options().await;
        if provider_fails_at_deadline {
            let error = base7_read.unwrap_err();
            assert!(matches!(error, Error::IO { .. }));
            assert!(error.to_string().contains("injected provider failure"));
        } else {
            assert_eq!(base7_read.unwrap().0["block_size"], "4096");
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 3);

        // Base 8 refreshes at its own offset, even during a renewed absence window.
        MockClock::advance(Duration::from_secs(9));
        MockClock::advance_system_time(Duration::from_secs(9));
        provider.fail.store(true, Ordering::SeqCst);
        let error = base8.get_storage_options().await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("injected provider failure"));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 4);

        provider.fail.store(false, Ordering::SeqCst);
        *provider.options.write().await =
            Some(scoped_block_sizes("2048", "8192", "32768", 600_000));
        MockClock::advance(Duration::from_secs(60));
        MockClock::advance_system_time(Duration::from_secs(60));
        assert_eq!(
            base7.get_storage_options().await.unwrap().0["block_size"],
            "8192"
        );
        assert_eq!(
            base8.get_storage_options().await.unwrap().0["block_size"],
            "32768"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
    }

    #[rstest::rstest]
    #[case::primary_unscoped(None, Some(500), None, Some(500))]
    #[case::primary_ignores_base_expiry(None, None, Some(500), None)]
    #[case::base_inherits_unscoped(Some(7), Some(500), None, Some(500))]
    #[case::base_with_own_expiry(Some(7), None, Some(500), Some(500))]
    #[case::base_overrides_earlier_unscoped(Some(7), Some(500), Some(1000), Some(1000))]
    #[tokio::test]
    async fn scoped_fallback_expiration_and_provider_errors(
        #[case] base_id: Option<u32>,
        #[case] unscoped_expiry: Option<u64>,
        #[case] base_expiry: Option<u64>,
        #[case] scope_expiry: Option<u64>,
    ) {
        MockClock::set_system_time(Duration::from_secs(100_000));
        let provider = Arc::new(TransitionProvider::default());
        let mut initial = HashMap::from([("token".into(), "fallback".into())]);
        if let Some(expiry) = unscoped_expiry {
            initial.insert(
                EXPIRES_AT_MILLIS_KEY.into(),
                (100_000_000 + expiry).to_string(),
            );
        }
        if let Some(expiry) = base_expiry {
            initial.insert(
                format!("base_7.{EXPIRES_AT_MILLIS_KEY}"),
                (100_000_000 + expiry).to_string(),
            );
        }
        let parent = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            initial,
            provider.clone(),
        ));
        let scope = parent.scoped_to_base(base_id);
        assert_eq!(
            scope.refresh_storage_options().await.unwrap().0["token"],
            "fallback"
        );

        let mut expected_calls = 1;
        if scope_expiry != Some(500) {
            MockClock::advance(Duration::from_millis(500));
            MockClock::advance_system_time(Duration::from_millis(500));
            assert_eq!(
                scope.get_storage_options().await.unwrap().0["token"],
                "fallback"
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), expected_calls);
        }

        MockClock::set_system_time(Duration::from_millis(
            100_000_000 + scope_expiry.unwrap_or(500),
        ));
        if let Some(expiry) = scope_expiry {
            // The fallback's actual expiry allows an early retry before the shared deadline.
            for result in [
                scope.get_storage_options().await,
                scope.refresh_storage_options().await,
            ] {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(
                    error
                        .to_string()
                        .contains(&format!("expired at {}", 100_000_000 + expiry))
                );
            }
            expected_calls += 2;
        } else {
            // The primary fallback has no expiry and retries at the monotonic deadline.
            MockClock::advance(Duration::from_millis(59_500));
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), expected_calls);
        provider.fail.store(true, Ordering::SeqCst);
        for _ in 0..2 {
            let error = scope.get_storage_options().await.unwrap_err();
            assert!(matches!(error, Error::IO { .. }));
            assert!(error.to_string().contains("injected provider failure"));
        }
        expected_calls += 2;
        assert_eq!(provider.calls.load(Ordering::SeqCst), expected_calls);
        provider.fail.store(false, Ordering::SeqCst);
        *provider.options.write().await = Some(vended_options());
        let expected_token = if base_id.is_some() { "seven" } else { "shared" };
        assert_eq!(
            scope.get_storage_options().await.unwrap().0["token"],
            expected_token
        );
        expected_calls += 1;
        assert_eq!(provider.calls.load(Ordering::SeqCst), expected_calls);
        provider.fail.store(true, Ordering::SeqCst);
        let error = scope.refresh_storage_options().await.unwrap_err();
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("injected provider failure"));
        assert_eq!(
            scope.get_storage_options().await.unwrap().0["token"],
            expected_token
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), expected_calls + 1);
    }

    #[derive(Debug)]
    struct GatedProvider {
        inner: TransitionProvider,
        entered: Notify,
        resume: Semaphore,
    }

    #[async_trait]
    impl StorageOptionsProvider for GatedProvider {
        async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
            self.entered.notify_one();
            self.resume.acquire().await.unwrap().forget();
            self.inner.fetch_storage_options().await
        }
        fn provider_id(&self) -> String {
            "gated-provider".into()
        }
    }

    #[tokio::test]
    async fn concurrent_scopes_share_cold_and_expired_parent_refresh() {
        let provider = Arc::new(GatedProvider {
            inner: TransitionProvider::default(),
            entered: Notify::new(),
            resume: Semaphore::new(0),
        });
        let parent = Arc::new(StorageOptionsAccessor::with_provider(provider.clone()));
        let scopes = (0..8)
            .map(|i| parent.scoped_to_base((i % 2 == 0).then_some(7)))
            .collect::<Vec<_>>();
        for refresh in [false, true] {
            if refresh {
                *provider.inner.options.write().await = Some(vended_options());
                MockClock::advance(Duration::from_secs(60));
            }
            let reads =
                futures::future::join_all(scopes.iter().map(|scope| scope.get_storage_options()));
            tokio::pin!(reads);
            tokio::select! {
                _ = provider.entered.notified() => {},
                _ = &mut reads => panic!("reads must wait for the gated provider"),
            }
            provider.resume.add_permits(1);
            for result in reads.await {
                let options = result.unwrap().0;
                if refresh {
                    assert!(options.contains_key("token"));
                } else {
                    assert!(options.is_empty());
                }
            }
            assert_eq!(
                provider.inner.calls.load(Ordering::SeqCst),
                if refresh { 2 } else { 1 }
            );
        }
    }

    #[tokio::test]
    async fn fresh_scopes_can_read_while_stale_base_refreshes() {
        MockClock::set_system_time(Duration::from_secs(100_121));
        let provider = Arc::new(GatedProvider {
            inner: TransitionProvider::default(),
            entered: Notify::new(),
            resume: Semaphore::new(0),
        });
        *provider.inner.options.write().await =
            Some(scoped_block_sizes("16384", "32768", "65536", 700_000));
        let parent = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
            scoped_block_sizes("2048", "4096", "8192", 120_000),
            provider.clone(),
        ));
        let base7 = parent.scoped_to_base(Some(7));
        let primary = parent.scoped_to_base(None);
        let base8 = parent.scoped_to_base(Some(8));

        let stale_reads = futures::future::join_all((0..8).map(|_| base7.get_storage_options()));
        tokio::pin!(stale_reads);
        tokio::select! {
            _ = provider.entered.notified() => {},
            _ = &mut stale_reads => panic!("base 7 must wait for the gated provider"),
        }
        for (scope, expected) in [(&primary, "2048"), (&base8, "8192")] {
            let options = futures::FutureExt::now_or_never(scope.get_storage_options())
                .expect("fresh scope must complete while the provider is pending");
            assert_eq!(options.unwrap().0["block_size"], expected);
        }
        provider.resume.add_permits(1);
        let renewed = tokio::time::timeout(Duration::from_secs(1), stale_reads)
            .await
            .expect("stale readers must share one refresh");
        for result in renewed {
            assert_eq!(result.unwrap().0["block_size"], "32768");
        }
        assert_eq!(
            primary.get_storage_options().await.unwrap().0["block_size"],
            "16384"
        );
        assert_eq!(provider.inner.calls.load(Ordering::SeqCst), 1);
    }
}
