// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AIMD-controlled token bucket rate limiter for ObjectStore operations.
//!
//! Wraps any [`object_store::ObjectStore`] with per-category token buckets
//! whose fill rates are dynamically adjusted by AIMD controllers. When cloud
//! stores return HTTP 429/503, the fill rate decreases multiplicatively. During
//! sustained success windows, it increases additively.
//!
//! Operations are split into four independent categories — **read**, **write**,
//! **delete**, **list** — each with its own AIMD controller and token bucket.
//! This prevents a burst of reads from starving writes, and vice versa.
//!
//! # Example
//!
//! ```ignore
//! use lance_io::object_store::throttle::{AimdThrottleConfig, AimdThrottledStore};
//!
//! let throttled = AimdThrottledStore::new(target, AimdThrottleConfig::default()).unwrap();
//! ```

use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Range;
use std::sync::{Arc, LazyLock, Weak};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use lance_core::utils::aimd::{AimdConfig, AimdController, RequestOutcome};
use lance_core::utils::tracing::TRACE_OBJECT_STORE_THROTTLE;
use object_store::ObjectStoreExt;
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
use object_store::client::{
    ClientOptions, HttpClient, HttpConnector, HttpError, HttpErrorKind, HttpRequest, HttpResponse,
    HttpResponseBody, HttpService,
};
use object_store::path::Path;
use object_store::{
    CopyMode, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions,
    RenameTargetMode, Result as OSResult, UploadPart,
};
use rand::Rng;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use object_store::list::{PaginatedListOptions, PaginatedListResult, PaginatedListStore};

use crate::object_store::ObjectStoreParams;

/// Match ObjectStore's documented delete concurrency to bound in-flight operations
/// and token reservations for stores without HTTP-level admission.
const DELETE_CONCURRENCY: usize = 10;

/// Check whether an `object_store::Error` represents a throttle response
/// (HTTP 429 / 503) from a cloud object store.
///
/// Regrettably, this information is not fully exposed by the `object_store` crate.
/// There is no generic mechanism for a custom object store to return a throttle error.
///
/// However, the builtin object stores all use RetryError when retries are configured and
/// throttle errors are returned.  Sadly, RetryError is not a public type, so we have to
/// infer it from the error message.  This is potentially dangerous because these errors
/// often include the URI itself and that URI could have any characters in it (e.g. if we
/// look for 429 then we might match a 429 in a UUID).These error messages currently look like:
///
/// ", after ... retries, max_retries: ..., retry_timeout: ..."
///
/// So, as a crude heuristic, which should work for the builtin object stores, but won't
/// work for custom object stores, we simply look for the string "retries, max_retries"
/// in the error message.
pub fn is_throttle_error(err: &object_store::Error) -> bool {
    // Only Generic errors can carry throttle responses
    if let object_store::Error::Generic { source, .. } = err {
        let message = source.to_string();
        let lowercase = message.to_ascii_lowercase();
        lowercase.contains("retries, max_retries")
            || lowercase.contains("serverbusy")
            || lowercase.contains("server busy")
            || lowercase.contains("egress is over the account limit")
            || lowercase.contains("http 429")
            || lowercase.contains("status code: 429")
            || lowercase.contains("429 too many requests")
            || lowercase.contains("too many requests")
            || lowercase.contains("slowdown")
            || lowercase.contains("please reduce your request rate")
            || lowercase.contains("rate limit")
            || lowercase.contains("throttling")
            || lowercase.contains("throttled")
    } else {
        false
    }
}

/// Configuration for the AIMD-throttled ObjectStore wrapper.
///
/// Each operation category (read, write, delete, list) has its own AIMD config.
/// Use [`with_aimd`](AimdThrottleConfig::with_aimd) to set all categories at
/// once, or per-category methods like [`with_read_aimd`](AimdThrottleConfig::with_read_aimd)
/// for fine-grained control.
#[derive(Debug, Clone)]
pub struct AimdThrottleConfig {
    /// AIMD configuration for read operations (get, get_opts, get_range, get_ranges, head).
    pub read: AimdConfig,
    /// AIMD configuration for write operations (put, put_opts, put_multipart, copy, rename, etc.).
    pub write: AimdConfig,
    /// AIMD configuration for delete operations.
    pub delete: AimdConfig,
    /// AIMD configuration for list operations.
    pub list: AimdConfig,
    /// Maximum tokens that can accumulate for bursts (shared across all categories).
    pub burst_capacity: u32,
    /// Maximum number of retries for throttle errors within the AIMD layer.
    pub max_retries: usize,
    /// Minimum backoff in milliseconds between retry attempts.
    pub min_backoff_ms: u64,
    /// Maximum backoff in milliseconds between retry attempts.
    pub max_backoff_ms: u64,
}

impl Default for AimdThrottleConfig {
    fn default() -> Self {
        let aimd = AimdConfig::default();
        Self {
            read: aimd.clone(),
            write: aimd.clone(),
            delete: aimd.clone(),
            list: aimd,
            burst_capacity: 100,
            max_retries: 3,
            min_backoff_ms: 100,
            max_backoff_ms: 300,
        }
    }
}

impl AimdThrottleConfig {
    /// Set the AIMD configuration for all four operation categories at once.
    pub fn with_aimd(self, aimd: AimdConfig) -> Self {
        Self {
            read: aimd.clone(),
            write: aimd.clone(),
            delete: aimd.clone(),
            list: aimd,
            ..self
        }
    }

    /// Set the AIMD configuration for read operations.
    pub fn with_read_aimd(self, aimd: AimdConfig) -> Self {
        Self { read: aimd, ..self }
    }

    /// Set the AIMD configuration for write operations.
    pub fn with_write_aimd(self, aimd: AimdConfig) -> Self {
        Self {
            write: aimd,
            ..self
        }
    }

    /// Set the AIMD configuration for delete operations.
    pub fn with_delete_aimd(self, aimd: AimdConfig) -> Self {
        Self {
            delete: aimd,
            ..self
        }
    }

    /// Set the AIMD configuration for list operations.
    pub fn with_list_aimd(self, aimd: AimdConfig) -> Self {
        Self { list: aimd, ..self }
    }

    /// Returns `true` when the AIMD throttle layer should be bypassed entirely.
    pub fn is_disabled(&self) -> bool {
        self.max_retries == 0
    }

    pub fn with_burst_capacity(self, burst_capacity: u32) -> Self {
        Self {
            burst_capacity,
            ..self
        }
    }

    /// Build an `AimdThrottleConfig` from storage options and environment variables.
    ///
    /// Storage options take precedence over environment variables, which take
    /// precedence over defaults. A single AIMD config is applied to all four
    /// operation categories (read/write/delete/list).
    ///
    /// | Setting              | Storage Option Key               | Env Var                          | Default |
    /// |----------------------|----------------------------------|----------------------------------|---------|
    /// | Initial rate         | `lance_aimd_initial_rate`        | `LANCE_AIMD_INITIAL_RATE`        | 2000    |
    /// | Min rate             | `lance_aimd_min_rate`            | `LANCE_AIMD_MIN_RATE`            | 1       |
    /// | Max rate             | `lance_aimd_max_rate`            | `LANCE_AIMD_MAX_RATE`            | 10000   |
    /// | Decrease factor      | `lance_aimd_decrease_factor`     | `LANCE_AIMD_DECREASE_FACTOR`     | 0.5     |
    /// | Additive increment   | `lance_aimd_additive_increment`  | `LANCE_AIMD_ADDITIVE_INCREMENT`  | 300     |
    /// | Burst capacity       | `lance_aimd_burst_capacity`      | `LANCE_AIMD_BURST_CAPACITY`      | 100     |
    /// | Max retries          | `lance_aimd_max_retries`         | `LANCE_AIMD_MAX_RETRIES`         | 3       |
    /// | Min backoff ms       | `lance_aimd_min_backoff_ms`      | `LANCE_AIMD_MIN_BACKOFF_MS`      | 100     |
    /// | Max backoff ms       | `lance_aimd_max_backoff_ms`      | `LANCE_AIMD_MAX_BACKOFF_MS`      | 300     |
    pub fn from_storage_options(
        storage_options: Option<&HashMap<String, String>>,
    ) -> lance_core::Result<Self> {
        fn resolve_f64(
            key: &str,
            storage_options: Option<&HashMap<String, String>>,
            default: f64,
        ) -> lance_core::Result<f64> {
            let env_key = key.to_ascii_uppercase();
            if let Some(val) = storage_options.and_then(|opts| opts.get(key)) {
                val.parse::<f64>().map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "Invalid value for storage option '{key}': '{val}'"
                    ))
                })
            } else if let Ok(val) = std::env::var(&env_key) {
                val.parse::<f64>().map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "Invalid value for env var '{env_key}': '{val}'"
                    ))
                })
            } else {
                Ok(default)
            }
        }

        fn resolve_u32(
            key: &str,
            storage_options: Option<&HashMap<String, String>>,
            default: u32,
        ) -> lance_core::Result<u32> {
            let env_key = key.to_ascii_uppercase();
            if let Some(val) = storage_options.and_then(|opts| opts.get(key)) {
                val.parse::<u32>().map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "Invalid value for storage option '{key}': '{val}'"
                    ))
                })
            } else if let Ok(val) = std::env::var(&env_key) {
                val.parse::<u32>().map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "Invalid value for env var '{env_key}': '{val}'"
                    ))
                })
            } else {
                Ok(default)
            }
        }

        fn resolve_usize(
            key: &str,
            storage_options: Option<&HashMap<String, String>>,
            default: usize,
        ) -> lance_core::Result<usize> {
            let env_key = key.to_ascii_uppercase();
            if let Some(val) = storage_options.and_then(|opts| opts.get(key)) {
                val.parse::<usize>().map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "Invalid value for storage option '{key}': '{val}'"
                    ))
                })
            } else if let Ok(val) = std::env::var(&env_key) {
                val.parse::<usize>().map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "Invalid value for env var '{env_key}': '{val}'"
                    ))
                })
            } else {
                Ok(default)
            }
        }

        fn resolve_u64(
            key: &str,
            storage_options: Option<&HashMap<String, String>>,
            default: u64,
        ) -> lance_core::Result<u64> {
            let env_key = key.to_ascii_uppercase();
            if let Some(val) = storage_options.and_then(|opts| opts.get(key)) {
                val.parse::<u64>().map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "Invalid value for storage option '{key}': '{val}'"
                    ))
                })
            } else if let Ok(val) = std::env::var(&env_key) {
                val.parse::<u64>().map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "Invalid value for env var '{env_key}': '{val}'"
                    ))
                })
            } else {
                Ok(default)
            }
        }

        let initial_rate = resolve_f64("lance_aimd_initial_rate", storage_options, 2000.0)?;
        let min_rate = resolve_f64("lance_aimd_min_rate", storage_options, 1.0)?;
        let max_rate = resolve_f64(
            "lance_aimd_max_rate",
            storage_options,
            AimdConfig::default().max_rate,
        )?;
        let decrease_factor = resolve_f64("lance_aimd_decrease_factor", storage_options, 0.5)?;
        let additive_increment =
            resolve_f64("lance_aimd_additive_increment", storage_options, 300.0)?;
        let burst_capacity = resolve_u32("lance_aimd_burst_capacity", storage_options, 100)?;
        let max_retries = resolve_usize("lance_aimd_max_retries", storage_options, 3)?;
        let min_backoff_ms = resolve_u64("lance_aimd_min_backoff_ms", storage_options, 100)?;
        let max_backoff_ms = resolve_u64("lance_aimd_max_backoff_ms", storage_options, 300)?;

        let aimd = AimdConfig::default()
            .with_initial_rate(initial_rate)
            .with_min_rate(min_rate)
            .with_max_rate(max_rate)
            .with_decrease_factor(decrease_factor)
            .with_additive_increment(additive_increment);

        Ok(Self {
            max_retries,
            min_backoff_ms,
            max_backoff_ms,
            ..Self::default()
                .with_aimd(aimd)
                .with_burst_capacity(burst_capacity)
        })
    }
}

struct TokenBucketState {
    tokens: f64,
    last_refill: tokio::time::Instant,
    rate: f64,
}

/// Per-category throttle state: an AIMD controller paired with a token bucket.
struct OperationThrottle {
    controller: AimdController,
    bucket: Mutex<TokenBucketState>,
    burst_capacity: f64,
    max_retries: usize,
    min_backoff_ms: u64,
    max_backoff_ms: u64,
}

impl OperationThrottle {
    fn new(
        aimd_config: AimdConfig,
        burst_capacity: f64,
        max_retries: usize,
        min_backoff_ms: u64,
        max_backoff_ms: u64,
    ) -> lance_core::Result<Self> {
        let initial_rate = aimd_config.initial_rate;
        let controller = AimdController::new(aimd_config)?;
        Ok(Self {
            controller,
            bucket: Mutex::new(TokenBucketState {
                tokens: burst_capacity,
                last_refill: tokio::time::Instant::now(),
                rate: initial_rate,
            }),
            burst_capacity,
            max_retries,
            min_backoff_ms,
            max_backoff_ms,
        })
    }

    /// Acquire a token from the bucket, sleeping if none are available.
    ///
    /// Each caller reserves a token immediately (allowing `tokens` to go
    /// negative) so that concurrent waiters queue behind each other instead
    /// of all waking at the same instant (thundering herd).
    async fn acquire_token(&self) {
        let sleep_duration = {
            let mut bucket = self.bucket.lock().await;
            let now = tokio::time::Instant::now();
            let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
            bucket.tokens = (bucket.tokens + elapsed * bucket.rate).min(self.burst_capacity);
            bucket.last_refill = now;

            // Reserve a token (may go negative to queue behind other waiters)
            bucket.tokens -= 1.0;

            if bucket.tokens >= 0.0 {
                // Had a token available, no need to sleep
                return;
            }

            // Sleep proportional to our position in the queue
            std::time::Duration::from_secs_f64(-bucket.tokens / bucket.rate)
        };

        tokio::time::sleep(sleep_duration).await;
    }

    /// Update the bucket's fill rate from the controller.
    async fn update_bucket_rate(&self, new_rate: f64) {
        let mut bucket = self.bucket.lock().await;
        bucket.rate = new_rate;
    }

    /// Classify a result and feed it back to the AIMD controller without
    /// acquiring a token. Uses `try_lock` for the bucket update so that if the
    /// bucket lock is contended the rate update is deferred to the next
    /// `throttled()` call.
    fn observe_outcome<T>(&self, result: &OSResult<T>) {
        let outcome = match result {
            Ok(_) => RequestOutcome::Success,
            Err(err) if is_throttle_error(err) => {
                debug!(
                    target: TRACE_OBJECT_STORE_THROTTLE,
                    error = %err,
                    "Throttle error detected in stream"
                );
                RequestOutcome::Throttled
            }
            Err(_) => RequestOutcome::Success,
        };
        let error = result
            .as_ref()
            .err()
            .map(|error| error as &dyn std::fmt::Display);
        let new_rate = self.record_outcome(outcome, error);
        if let Ok(mut bucket) = self.bucket.try_lock() {
            bucket.rate = new_rate;
        }
    }

    fn record_outcome(
        &self,
        outcome: RequestOutcome,
        error: Option<&dyn std::fmt::Display>,
    ) -> f64 {
        let prev_rate = self.controller.current_rate();
        let new_rate = self.controller.record_outcome(outcome);
        if new_rate < prev_rate {
            if let Some(error) = error {
                warn!(
                    target: TRACE_OBJECT_STORE_THROTTLE,
                    previous_rate = format!("{prev_rate:.1}"),
                    new_rate = format!("{new_rate:.1}"),
                    error = %error,
                    "AIMD throttle: rate reduced due to throttle errors"
                );
            } else {
                warn!(
                    target: TRACE_OBJECT_STORE_THROTTLE,
                    previous_rate = format!("{prev_rate:.1}"),
                    new_rate = format!("{new_rate:.1}"),
                    "AIMD throttle: rate reduced due to throttle errors"
                );
            }
        }
        new_rate
    }

    /// Execute an operation with throttling: acquire token, run, classify result.
    /// On throttle errors, retries up to `max_retries` times with a random
    /// backoff between `min_backoff_ms` and `max_backoff_ms` between attempts.
    async fn throttled<T, F, Fut>(&self, f: F) -> OSResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = OSResult<T>>,
    {
        for attempt in 0..=self.max_retries {
            self.acquire_token().await;
            let result = f().await;
            let outcome = match &result {
                Ok(_) => RequestOutcome::Success,
                Err(err) if is_throttle_error(err) => {
                    debug!(
                        target: TRACE_OBJECT_STORE_THROTTLE,
                        error = %err,
                        "Throttle error detected"
                    );
                    RequestOutcome::Throttled
                }
                Err(_) => RequestOutcome::Success, // Non-throttle errors don't indicate capacity problems
            };
            let error = result
                .as_ref()
                .err()
                .map(|error| error as &dyn std::fmt::Display);
            let new_rate = self.record_outcome(outcome, error);
            self.update_bucket_rate(new_rate).await;

            match &result {
                Err(err) if is_throttle_error(err) && attempt < self.max_retries => {
                    let backoff_ms =
                        rand::rng().random_range(self.min_backoff_ms..=self.max_backoff_ms);
                    debug!(
                        target: TRACE_OBJECT_STORE_THROTTLE,
                        attempt = attempt + 1,
                        max_retries = self.max_retries,
                        backoff_ms,
                        error = %err,
                        "Retrying after throttle error"
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                _ => return result,
            }
        }
        unreachable!()
    }
}

impl Debug for OperationThrottle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperationThrottle")
            .field("controller", &self.controller)
            .field("burst_capacity", &self.burst_capacity)
            .finish()
    }
}

#[derive(Clone)]
pub(crate) struct AimdThrottleState {
    read: Arc<OperationThrottle>,
    write: Arc<OperationThrottle>,
    delete: Arc<OperationThrottle>,
    list: Arc<OperationThrottle>,
}

impl AimdThrottleState {
    pub(crate) fn new(config: AimdThrottleConfig) -> lance_core::Result<Self> {
        // A retry draws its backoff from `min..=max`, and sampling an empty
        // range panics, so a range that cannot be sampled is rejected while
        // the store is being built rather than on the first throttled request.
        if config.min_backoff_ms > config.max_backoff_ms {
            return Err(lance_core::Error::invalid_input(format!(
                "lance_aimd_min_backoff_ms ({}) must not exceed lance_aimd_max_backoff_ms \
                 ({}): the retry backoff is drawn from that range",
                config.min_backoff_ms, config.max_backoff_ms
            )));
        }
        let burst_capacity = config.burst_capacity as f64;
        let max_retries = config.max_retries;
        let min_backoff_ms = config.min_backoff_ms;
        let max_backoff_ms = config.max_backoff_ms;
        Ok(Self {
            read: Arc::new(OperationThrottle::new(
                config.read,
                burst_capacity,
                max_retries,
                min_backoff_ms,
                max_backoff_ms,
            )?),
            write: Arc::new(OperationThrottle::new(
                config.write,
                burst_capacity,
                max_retries,
                min_backoff_ms,
                max_backoff_ms,
            )?),
            delete: Arc::new(OperationThrottle::new(
                config.delete,
                burst_capacity,
                max_retries,
                min_backoff_ms,
                max_backoff_ms,
            )?),
            list: Arc::new(OperationThrottle::new(
                config.list,
                burst_capacity,
                max_retries,
                min_backoff_ms,
                max_backoff_ms,
            )?),
        })
    }

    fn downgrade(&self) -> WeakThrottleState {
        WeakThrottleState {
            read: Arc::downgrade(&self.read),
            write: Arc::downgrade(&self.write),
            delete: Arc::downgrade(&self.delete),
            list: Arc::downgrade(&self.list),
        }
    }
}

/// A cache entry of [`SHARED_THROTTLE_STATES`]: alive exactly as long as some
/// store still holds the budgets, since a store keeps all four.
struct WeakThrottleState {
    read: Weak<OperationThrottle>,
    write: Weak<OperationThrottle>,
    delete: Weak<OperationThrottle>,
    list: Weak<OperationThrottle>,
}

impl WeakThrottleState {
    fn upgrade(&self) -> Option<AimdThrottleState> {
        Some(AimdThrottleState {
            read: self.read.upgrade()?,
            write: self.write.upgrade()?,
            delete: self.delete.upgrade()?,
            list: self.list.upgrade()?,
        })
    }
}

/// Throttle budgets keyed by the same `(store prefix, params)` identity the
/// registry caches stores under. The AIMD contract is one budget per bucket, so
/// stores that are split further, one per dataset in the `dataset` metrics label
/// mode, must share the bucket's budget rather than each getting a full one.
static SHARED_THROTTLE_STATES: LazyLock<
    std::sync::Mutex<HashMap<(String, ObjectStoreParams), WeakThrottleState>>,
> = LazyLock::new(Default::default);

/// The throttle state for a store of `store_prefix` built with `params`: the one
/// its sibling stores already use, or a fresh one. `None` when throttling is
/// disabled by the params.
pub(crate) fn shared_throttle_state(
    store_prefix: &str,
    params: &ObjectStoreParams,
) -> lance_core::Result<Option<AimdThrottleState>> {
    let config = AimdThrottleConfig::from_storage_options(params.storage_options())?;
    if config.is_disabled() {
        return Ok(None);
    }
    let key = (store_prefix.to_owned(), params.clone());
    let mut states = SHARED_THROTTLE_STATES
        .lock()
        .expect("SHARED_THROTTLE_STATES lock poisoned");
    if let Some(state) = states.get(&key).and_then(WeakThrottleState::upgrade) {
        return Ok(Some(state));
    }
    let state = AimdThrottleState::new(config)?;
    // Entries whose stores are gone can only be replaced, never hit, so drop them
    // while the map is being grown anyway.
    states.retain(|_, weak| weak.upgrade().is_some());
    states.insert(key, state.downgrade());
    Ok(Some(state))
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[derive(Debug)]
pub(crate) struct AimdHttpConnector<C> {
    inner: C,
    write: Option<Arc<OperationThrottle>>,
    delete: Option<Arc<OperationThrottle>>,
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
impl<C> AimdHttpConnector<C> {
    fn new(inner: C, state: Option<&AimdThrottleState>) -> Self {
        Self {
            inner,
            write: state.map(|state| Arc::clone(&state.write)),
            delete: state.map(|state| Arc::clone(&state.delete)),
        }
    }
}

#[cfg(all(
    any(feature = "aws", feature = "azure", feature = "gcp"),
    feature = "metrics"
))]
pub(crate) fn cloud_http_connector(
    state: Option<&AimdThrottleState>,
    metrics_base: String,
) -> AimdHttpConnector<crate::object_store::metrics::MeteringHttpConnector> {
    AimdHttpConnector::new(
        crate::object_store::metrics::MeteringHttpConnector::new(metrics_base),
        state,
    )
}

#[cfg(all(
    any(feature = "aws", feature = "azure", feature = "gcp"),
    not(feature = "metrics")
))]
pub(crate) fn cloud_http_connector(
    state: Option<&AimdThrottleState>,
    _metrics_base: String,
) -> AimdHttpConnector<object_store::client::ReqwestConnector> {
    AimdHttpConnector::new(object_store::client::ReqwestConnector::default(), state)
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
impl<C: HttpConnector> HttpConnector for AimdHttpConnector<C> {
    fn connect(&self, options: &ClientOptions) -> object_store::Result<HttpClient> {
        Ok(HttpClient::new(AimdHttpService {
            inner: self.inner.connect(options)?,
            write: self.write.clone(),
            delete: self.delete.clone(),
        }))
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[derive(Debug)]
struct AimdHttpService {
    inner: HttpClient,
    write: Option<Arc<OperationThrottle>>,
    delete: Option<Arc<OperationThrottle>>,
}

/// Whether the request completes a multipart upload.
///
/// S3 and the GCS XML API complete with `POST ?uploadId=<id>` (no
/// `partNumber`); Azure completes with `PUT ?comp=blocklist`. Upload
/// initiation (`POST ?uploads`) is deliberately excluded.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
fn is_multipart_complete_request(request: &HttpRequest) -> bool {
    let Some(query) = request.uri().query() else {
        return false;
    };
    let mut has_upload_id = false;
    let mut has_part_number = false;
    let mut is_blocklist = false;
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        if key.eq_ignore_ascii_case("uploadId") {
            has_upload_id = true;
        } else if key.eq_ignore_ascii_case("partNumber") {
            has_part_number = true;
        } else if key.eq_ignore_ascii_case("comp") && value.eq_ignore_ascii_case("blocklist") {
            is_blocklist = true;
        }
    }
    match *request.method() {
        ::http::Method::POST => has_upload_id && !has_part_number,
        ::http::Method::PUT => is_blocklist,
        _ => false,
    }
}

/// Whether the request carries one multipart upload's data: an uploaded part
/// (S3/GCS `PUT ?partNumber=..&uploadId=..`, Azure `PUT ?comp=block`) or the
/// complete request itself.
///
/// Completes are included because `MultipartUpload::complete` consumes the
/// recorded part ids before issuing the request, so retries must happen here
/// at the HTTP layer, where the request — including its serialized part list
/// — can simply be re-sent.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
fn is_multipart_upload_request(request: &HttpRequest) -> bool {
    if is_multipart_complete_request(request) {
        return true;
    }
    if request.method() != ::http::Method::PUT {
        return false;
    }
    request.uri().query().is_some_and(|query| {
        url::form_urlencoded::parse(query.as_bytes()).any(|(key, value)| {
            key.eq_ignore_ascii_case("partNumber")
                || (key.eq_ignore_ascii_case("comp") && value.eq_ignore_ascii_case("block"))
        })
    })
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
fn is_delete_request(request: &HttpRequest) -> bool {
    let query = request.uri().query();
    if request.method() == ::http::Method::DELETE {
        // Multipart aborts belong to the write budget.
        return !query.is_some_and(|query| {
            url::form_urlencoded::parse(query.as_bytes())
                .any(|(key, _)| key.eq_ignore_ascii_case("uploadId"))
        });
    }
    request.method() == ::http::Method::POST
        && query.is_some_and(|query| {
            url::form_urlencoded::parse(query.as_bytes()).any(|(key, value)| {
                key.eq_ignore_ascii_case("delete")
                    || (key.eq_ignore_ascii_case("comp") && value.eq_ignore_ascii_case("batch"))
            })
        })
}

/// The `Code` of an XML error document — a root `Error` element, optionally
/// namespaced or carrying attributes, whose first `Code` child holds the error
/// code — or `None` when `body` is not one.
///
/// S3, GCS and Azure all answer errors with this shape. Success documents
/// (e.g. `CompleteMultipartUploadResult`) return `None` whatever text they
/// carry, so an object key echoed in `<Key>` or `<Location>` cannot be
/// mistaken for an error code.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
fn error_document_code(body: &str) -> Option<String> {
    use quick_xml::events::Event;

    let mut reader = quick_xml::Reader::from_str(body);
    reader.config_mut().trim_text(true);

    let root = loop {
        match reader.read_event().ok()? {
            Event::Decl(_) | Event::PI(_) | Event::Comment(_) | Event::DocType(_) => {}
            Event::Start(root) => break root,
            _ => return None,
        }
    };
    if !root.local_name().as_ref().eq_ignore_ascii_case(b"Error") {
        return None;
    }

    // Track depth so only a direct child of the root counts.
    let mut depth = 0usize;
    loop {
        match reader.read_event().ok()? {
            Event::Start(child)
                if depth == 0 && child.local_name().as_ref().eq_ignore_ascii_case(b"Code") =>
            {
                break;
            }
            Event::Start(_) => depth += 1,
            // At depth 0 this closes the root, which had no `Code` child.
            Event::End(_) => depth = depth.checked_sub(1)?,
            Event::Eof => return None,
            _ => {}
        }
    }

    let mut code = String::new();
    loop {
        match reader.read_event().ok()? {
            Event::Text(text) => code.push_str(&text.decode().ok()?),
            Event::CData(data) => code.push_str(&data.decode().ok()?),
            Event::GeneralRef(entity) => match entity.resolve_char_ref().ok()? {
                Some(ch) => code.push(ch),
                None => code.push_str(quick_xml::escape::resolve_predefined_entity(
                    &entity.decode().ok()?,
                )?),
            },
            Event::End(_) => break,
            Event::Eof => return None,
            _ => {}
        }
    }
    let code = code.trim();
    (!code.is_empty()).then(|| code.to_string())
}

/// Whether an error code reports throttling or overload.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
fn is_throttle_error_code(code: &str) -> bool {
    code.eq_ignore_ascii_case("RequestTimeout")
        || code.eq_ignore_ascii_case("SlowDown")
        || code.eq_ignore_ascii_case("ServerBusy")
        || code.to_ascii_lowercase().contains("throttl")
}

/// What a multipart request's response body says beyond its status.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BodyVerdict {
    retryable: bool,
    throttle: bool,
}

/// Classifies a multipart request's response body.
///
/// An XML error document is judged by its `Code` alone. A complete answered
/// with `InternalError` is retryable even under HTTP 200: S3 can report a
/// failed `CompleteMultipartUpload` that way, and re-sending the request
/// repeats the exact completion because it carries the serialized part list.
/// A 2xx body that is not an error document is a success. Any other body on
/// an error status falls back to matching throttle markers anywhere in it.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
fn classify_multipart_response_body(
    status: ::http::StatusCode,
    body: &str,
    is_complete: bool,
) -> BodyVerdict {
    if let Some(code) = error_document_code(body) {
        let throttle = is_throttle_error_code(&code);
        return BodyVerdict {
            retryable: throttle || (is_complete && code.eq_ignore_ascii_case("InternalError")),
            throttle,
        };
    }
    if status.is_success() {
        return BodyVerdict::default();
    }
    let body = body.to_ascii_lowercase();
    let throttle = body.contains("requesttimeout")
        || body.contains("slowdown")
        || body.contains("serverbusy")
        || body.contains("throttl");
    BodyVerdict {
        retryable: throttle,
        throttle,
    }
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
fn is_retryable_http_error(error: &HttpError) -> bool {
    matches!(
        error.kind(),
        HttpErrorKind::Connect
            | HttpErrorKind::Request
            | HttpErrorKind::Timeout
            | HttpErrorKind::Interrupted
    )
}

#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
#[async_trait]
impl HttpService for AimdHttpService {
    async fn call(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        if let Some(delete) = self.delete.as_ref().filter(|_| is_delete_request(&request)) {
            // The cloud client owns retry and bulk response handling. Each HTTP
            // attempt draws one token, including retries of a bulk request.
            delete.acquire_token().await;
            let result = self.inner.execute(request).await;
            let is_throttle = match &result {
                Ok(response) => matches!(
                    response.status(),
                    ::http::StatusCode::TOO_MANY_REQUESTS | ::http::StatusCode::SERVICE_UNAVAILABLE
                ),
                // Non-throttle errors do not indicate capacity problems, matching
                // the multipart write hook and the AIMD controller's policy.
                Err(_) => false,
            };
            let outcome = if is_throttle {
                RequestOutcome::Throttled
            } else {
                RequestOutcome::Success
            };
            let new_rate = delete.record_outcome(
                outcome,
                result.as_ref().err().map(|err| err as &dyn Display),
            );
            delete.update_bucket_rate(new_rate).await;
            return result;
        }
        let Some(write) = self.write.as_ref() else {
            return self.inner.execute(request).await;
        };
        if !is_multipart_upload_request(&request) {
            return self.inner.execute(request).await;
        }
        let is_complete = is_multipart_complete_request(&request);

        for attempt in 0..=write.max_retries {
            write.acquire_token().await;
            let mut result = self.inner.execute(request.clone()).await;
            let mut is_retryable = result.as_ref().err().is_some_and(is_retryable_http_error);
            let mut is_throttle = false;
            let mut response_status = None;

            if let Ok(response) = result {
                let status = response.status();
                response_status = Some(status);
                is_retryable = status == ::http::StatusCode::REQUEST_TIMEOUT
                    || status == ::http::StatusCode::TOO_MANY_REQUESTS
                    || status.is_server_error();
                is_throttle = status == ::http::StatusCode::TOO_MANY_REQUESTS
                    || status == ::http::StatusCode::SERVICE_UNAVAILABLE;

                let (parts, body) = response.into_parts();
                result = match body.bytes().await {
                    Ok(bytes) => {
                        let verdict = classify_multipart_response_body(
                            status,
                            &String::from_utf8_lossy(&bytes),
                            is_complete,
                        );
                        is_retryable |= verdict.retryable;
                        is_throttle |= verdict.throttle;
                        Ok(HttpResponse::from_parts(
                            parts,
                            HttpResponseBody::from(bytes),
                        ))
                    }
                    Err(error) => {
                        is_retryable = is_retryable_http_error(&error);
                        Err(error)
                    }
                };
            }

            let detail = response_status
                .filter(|status| !status.is_success())
                .map(|status| format!("HTTP status {status}"));
            let error = result
                .as_ref()
                .err()
                .map(|error| error as &dyn std::fmt::Display)
                .or_else(|| {
                    detail
                        .as_ref()
                        .map(|detail| detail as &dyn std::fmt::Display)
                });
            let outcome = if is_throttle {
                RequestOutcome::Throttled
            } else {
                RequestOutcome::Success
            };
            let new_rate = write.record_outcome(outcome, error);
            write.update_bucket_rate(new_rate).await;

            if is_retryable && attempt < write.max_retries {
                let backoff_ms =
                    rand::rng().random_range(write.min_backoff_ms..=write.max_backoff_ms);
                debug!(
                    target: TRACE_OBJECT_STORE_THROTTLE,
                    attempt = attempt + 1,
                    max_retries = write.max_retries,
                    backoff_ms,
                    "Retrying multipart upload request after retryable HTTP response"
                );
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                continue;
            }
            return result;
        }
        unreachable!()
    }
}

/// A [`MultipartUpload`] wrapper that applies the write AIMD controller.
struct ThrottledMultipartUpload {
    target: Box<dyn MultipartUpload>,
    write: Arc<OperationThrottle>,
    requests_throttled_at_http: bool,
}

impl Debug for ThrottledMultipartUpload {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThrottledMultipartUpload").finish()
    }
}

#[async_trait]
impl MultipartUpload for ThrottledMultipartUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        // Call put_part synchronously to preserve part ordering regardless
        // of which futures are awaited first.
        let fut = self.target.put_part(data);
        if self.requests_throttled_at_http {
            return fut;
        }
        let write = Arc::clone(&self.write);
        Box::pin(async move {
            write.acquire_token().await;
            let result = fut.await;
            write.observe_outcome(&result);
            result
        })
    }

    /// Completes the multipart upload by calling the target exactly once.
    ///
    /// `MultipartUpload::complete` consumes the recorded part ids before
    /// sending the completion request, so a wrapper-level retry cannot repeat
    /// it: the second call fails deterministically with "Missing part" and
    /// masks the real error. Retries happen at the HTTP layer instead, which
    /// re-sends the same serialized request.
    async fn complete(&mut self) -> OSResult<PutResult> {
        if self.requests_throttled_at_http {
            // The HTTP connector acquires a token, observes the outcome and
            // retries each attempt of the complete request; wrapping it here
            // would double-count.
            return self.target.complete().await;
        }
        self.write.acquire_token().await;
        let result = self.target.complete().await;
        self.write.observe_outcome(&result);
        result
    }

    async fn abort(&mut self) -> OSResult<()> {
        let target = &mut self.target;
        for attempt in 0..=self.write.max_retries {
            self.write.acquire_token().await;
            let result = target.abort().await;
            self.write.observe_outcome(&result);

            match &result {
                Err(err) if is_throttle_error(err) && attempt < self.write.max_retries => {
                    let backoff_ms = rand::rng()
                        .random_range(self.write.min_backoff_ms..=self.write.max_backoff_ms);
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                _ => return result,
            }
        }
        unreachable!()
    }
}

/// An ObjectStore wrapper that rate-limits operations using per-category token
/// buckets whose fill rates are controlled by AIMD algorithms.
///
/// Operations are split into four independent categories:
/// - **read**: `get`, `get_opts`, `get_range`, `get_ranges`, `head`
/// - **write**: `put`, `put_opts`, `put_multipart`, `put_multipart_opts`, `copy`, `copy_if_not_exists`, `rename`, `rename_if_not_exists`
/// - **delete**: `delete`
/// - **list**: `list`, `list_with_offset`, `list_with_delimiter`
///
/// Native cloud deletes draw one token per HTTP request, including bulk requests.
/// Other stores delete paths individually with bounded concurrency so each
/// request starts only after acquiring a delete token.
///
/// Streaming list operations acquire a token before starting the underlying list stream.
/// List items also feed their outcomes back to the AIMD controller.
///
/// This is not perfect but probably as close as we can get without moving the throttle into
/// the object_store crate itself.
pub struct AimdThrottledStore {
    target: Arc<dyn ObjectStore>,
    read: Arc<OperationThrottle>,
    write: Arc<OperationThrottle>,
    delete: Arc<OperationThrottle>,
    list: Arc<OperationThrottle>,
    native_http_throttling: bool,
}

impl Debug for AimdThrottledStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AimdThrottledStore")
            .field("target", &self.target)
            .field("read", &self.read)
            .field("write", &self.write)
            .field("delete", &self.delete)
            .field("list", &self.list)
            .field("native_http_throttling", &self.native_http_throttling)
            .finish()
    }
}

impl Display for AimdThrottledStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AimdThrottledStore({})", self.target)
    }
}

impl AimdThrottledStore {
    pub fn new(
        target: Arc<dyn ObjectStore>,
        config: AimdThrottleConfig,
    ) -> lance_core::Result<Self> {
        Ok(Self::new_with_state(
            target,
            AimdThrottleState::new(config)?,
            false,
        ))
    }

    pub(crate) fn new_with_state(
        target: Arc<dyn ObjectStore>,
        state: AimdThrottleState,
        native_http_throttling: bool,
    ) -> Self {
        Self {
            target,
            read: state.read,
            write: state.write,
            delete: state.delete,
            list: state.list,
            native_http_throttling,
        }
    }

    /// Put a paginated lister on the same list budget as this store.
    pub fn wrap_paginated(
        &self,
        inner: Arc<dyn PaginatedListStore>,
    ) -> Arc<dyn PaginatedListStore> {
        Arc::new(ThrottledListStore {
            inner,
            throttle: self.list.clone(),
        })
    }
}

/// A store paired with the paginated lister that shares its rate limits.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
type StoreWithLister = (Arc<dyn ObjectStore>, Option<Arc<dyn PaginatedListStore>>);

/// Apply AIMD throttling to a store and to the lister that shares its list budget.
///
/// [`crate::object_store::ObjectStore::read_dir_page`] goes to the lister rather than
/// through the store, so both have to be wrapped for list requests to be counted once
/// against one rate.
#[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
pub(crate) fn with_throttling(
    state: Option<AimdThrottleState>,
    native_http_throttling: bool,
    store: Arc<dyn ObjectStore>,
    lister: Option<Arc<dyn PaginatedListStore>>,
) -> StoreWithLister {
    let Some(state) = state else {
        return (store, lister);
    };
    let store = Arc::new(AimdThrottledStore::new_with_state(
        store,
        state,
        native_http_throttling,
    ));
    let lister = lister.map(|lister| store.wrap_paginated(lister));
    (store, lister)
}

/// A [`PaginatedListStore`] whose requests draw on a store's list token bucket.
struct ThrottledListStore {
    inner: Arc<dyn PaginatedListStore>,
    throttle: Arc<OperationThrottle>,
}

// Throttling only adds waiting, so every semantic of the store it wraps has to reach the
// listing unchanged; the lint keeps a method added to the trait from silently falling back to
// its default here.
#[async_trait]
#[deny(clippy::missing_trait_methods)]
impl PaginatedListStore for ThrottledListStore {
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        opts: PaginatedListOptions,
    ) -> OSResult<PaginatedListResult> {
        self.throttle
            .throttled(|| self.inner.list_paginated(prefix, opts.clone()))
            .await
    }
}

#[async_trait]
#[deny(clippy::missing_trait_methods)]
impl ObjectStore for AimdThrottledStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.write
            .throttled(|| self.target.put_opts(location, bytes.clone(), opts.clone()))
            .await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        let target = self
            .write
            .throttled(|| self.target.put_multipart_opts(location, opts.clone()))
            .await?;
        Ok(Box::new(ThrottledMultipartUpload {
            target,
            write: Arc::clone(&self.write),
            requests_throttled_at_http: self.native_http_throttling,
        }))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.read
            .throttled(|| self.target.get_opts(location, options.clone()))
            .await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.read
            .throttled(|| self.target.get_ranges(location, ranges))
            .await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        if self.native_http_throttling {
            let delete = Arc::clone(&self.delete);
            return self
                .target
                .delete_stream(locations)
                .map(move |item| {
                    // A successful bulk HTTP response can contain per-object
                    // throttle failures that only the native client can parse.
                    // Admission and request-level outcomes stay at the HTTP hook.
                    let is_unobserved_throttle = item.as_ref().err().is_some_and(|error| {
                        let object_store::Error::Generic { source, .. } = error else {
                            return false;
                        };
                        // Native per-object errors have no nested source. RetryError
                        // wraps RequestError: only its successful-status error body
                        // needs feedback here; HTTP failures were already observed.
                        // These private types cannot be downcast through the public API.
                        let is_parsed_response = source.source().is_none_or(|cause| {
                            cause
                                .to_string()
                                .starts_with("Server returned error response:")
                        });
                        is_parsed_response && is_throttle_error(error)
                    });
                    if is_unobserved_throttle {
                        delete.observe_outcome(&item);
                    }
                    item
                })
                .boxed();
        }

        // Generic stores do not expose their request boundaries. Delete paths
        // individually so each operation waits for a token before it starts.
        let target = Arc::clone(&self.target);
        let delete = Arc::clone(&self.delete);
        locations
            .map(move |location| {
                let target = Arc::clone(&target);
                let delete = Arc::clone(&delete);
                async move {
                    let location = location?;
                    delete.throttled(|| target.delete(&location)).await?;
                    Ok(location)
                }
            })
            .buffered(DELETE_CONCURRENCY)
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let throttle = Arc::clone(&self.list);
        let throttle_for_start = Arc::clone(&throttle);
        let target = Arc::clone(&self.target);
        let prefix = prefix.cloned();
        futures::stream::once(async move {
            throttle_for_start.acquire_token().await;
            target.list(prefix.as_ref())
        })
        .flatten()
        .map(move |item| {
            throttle.observe_outcome(&item);
            item
        })
        .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let throttle = Arc::clone(&self.list);
        let throttle_for_start = Arc::clone(&throttle);
        let target = Arc::clone(&self.target);
        let prefix = prefix.cloned();
        let offset = offset.clone();
        futures::stream::once(async move {
            throttle_for_start.acquire_token().await;
            target.list_with_offset(prefix.as_ref(), &offset)
        })
        .flatten()
        .map(move |item| {
            throttle.observe_outcome(&item);
            item
        })
        .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.list
            .throttled(|| self.target.list_with_delimiter(prefix))
            .await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.write
            .throttled(|| self.target.copy_opts(from, to, opts.clone()))
            .await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
        let RenameOptions {
            target_mode,
            extensions,
        } = opts;

        match target_mode {
            RenameTargetMode::Overwrite => {
                // Delegate to the underlying store to preserve atomic rename
                // semantics where the store supports them (e.g. local filesystem).
                self.write
                    .throttled(|| {
                        self.target.rename_opts(
                            from,
                            to,
                            RenameOptions {
                                target_mode,
                                extensions: extensions.clone(),
                            },
                        )
                    })
                    .await
            }
            RenameTargetMode::Create => {
                // For rename_if_not_exists, the default object_store implementation
                // is copy + delete. Perform the two steps under their respective
                // throttles so they can be retried independently. If copy succeeds
                // but delete is throttled, the caller's intent is satisfied: the
                // destination now exists and the data has been committed.
                let copy_options = CopyOptions {
                    mode: CopyMode::Create,
                    extensions: extensions.clone(),
                };
                self.write
                    .throttled(|| self.target.copy_opts(from, to, copy_options.clone()))
                    .await?;

                // Best-effort cleanup of the source. Failure here does not change
                // the outcome of the rename, so do not propagate the error.
                if let Err(err) = self.delete.throttled(|| self.target.delete(from)).await {
                    warn!(
                        target: TRACE_OBJECT_STORE_THROTTLE,
                        source = %from,
                        destination = %to,
                        error = %err,
                        "rename_if_not_exists source cleanup failed after destination was created"
                    );
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::StorageOptionsAccessor;

    #[test]
    fn test_shared_throttle_state_follows_store_identity() {
        let params = ObjectStoreParams::default();
        let a = shared_throttle_state("s3$shared-bucket", &params)
            .unwrap()
            .unwrap();
        let b = shared_throttle_state("s3$shared-bucket", &params)
            .unwrap()
            .unwrap();
        // Two stores of one bucket, e.g. two datasets in dataset label mode, draw
        // on one budget per operation category.
        assert!(Arc::ptr_eq(&a.read, &b.read));
        assert!(Arc::ptr_eq(&a.write, &b.write));
        assert!(Arc::ptr_eq(&a.delete, &b.delete));
        assert!(Arc::ptr_eq(&a.list, &b.list));

        let other = shared_throttle_state("s3$other-bucket", &params)
            .unwrap()
            .unwrap();
        assert!(!Arc::ptr_eq(&a.read, &other.read));

        let disabled = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([("lance_aimd_max_retries".to_string(), "0".to_string())]),
            ))),
            ..Default::default()
        };
        assert!(
            shared_throttle_state("s3$shared-bucket", &disabled)
                .unwrap()
                .is_none()
        );
    }

    use object_store::memory::InMemory;
    use rstest::rstest;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    const THROTTLE_ERROR_RESPONSE: &str = "request failed, after 3 retries, max_retries: 3, retry_timeout: 30s - Server returned non-2xx status code: 503: x-ms-request-id: azure-request-id";

    fn make_generic_error(msg: &str) -> object_store::Error {
        object_store::Error::Generic {
            store: "test",
            source: msg.into(),
        }
    }

    fn params_with(options: &[(&str, &str)]) -> ObjectStoreParams {
        ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                options
                    .iter()
                    .map(|(key, value)| (key.to_string(), value.to_string()))
                    .collect(),
            ))),
            ..Default::default()
        }
    }

    /// A backoff range that cannot be sampled is rejected while the store is
    /// being built. Reaching the sampler with one panics, and raising only the
    /// minimum above the default maximum is enough to get there.
    #[rstest]
    #[case::min_above_explicit_max(
        &[("lance_aimd_min_backoff_ms", "500"), ("lance_aimd_max_backoff_ms", "100")]
    )]
    #[case::min_above_default_max(&[("lance_aimd_min_backoff_ms", "500")])]
    fn test_shared_throttle_state_rejects_an_unsamplable_backoff_range(
        #[case] options: &[(&str, &str)],
    ) {
        let Err(err) = shared_throttle_state("s3$backoff-bucket", &params_with(options)) else {
            panic!("an unsamplable backoff range must be rejected");
        };
        assert!(
            matches!(err, lance_core::Error::InvalidInput { .. }),
            "got: {err:?}"
        );
        let message = err.to_string();
        assert!(message.contains("lance_aimd_min_backoff_ms"), "{message}");
        assert!(message.contains("lance_aimd_max_backoff_ms"), "{message}");
    }

    /// Equal bounds are a one-value range, which samples fine.
    #[test]
    fn test_shared_throttle_state_accepts_equal_backoff_bounds() {
        let params = params_with(&[
            ("lance_aimd_min_backoff_ms", "250"),
            ("lance_aimd_max_backoff_ms", "250"),
        ]);
        assert!(
            shared_throttle_state("s3$equal-bounds-bucket", &params)
                .unwrap()
                .is_some()
        );
    }

    /// With retries off nothing samples the range, and a store that opens
    /// today has to keep opening.
    #[test]
    fn test_shared_throttle_state_ignores_the_backoff_range_when_disabled() {
        let params = params_with(&[
            ("lance_aimd_max_retries", "0"),
            ("lance_aimd_min_backoff_ms", "500"),
        ]);
        assert!(
            shared_throttle_state("s3$disabled-bucket", &params)
                .unwrap()
                .is_none()
        );
    }

    #[rstest]
    #[case::retry_error("Error after 10 retries, max_retries: 10, retry_timeout: 180s", true)]
    #[case::retries_in_message(
        "request failed, after 3 retries, max_retries: 5, retry_timeout: 60s",
        true
    )]
    #[case::not_found("Object not found", false)]
    #[case::permission_denied("Access denied", false)]
    #[case::timeout("Connection timed out", false)]
    #[case::http_429_without_retries("HTTP 429 Too Many Requests", true)]
    #[case::slowdown_without_retries("SlowDown: Please reduce your request rate", true)]
    #[case::azure_server_busy("Code: ServerBusy", true)]
    #[case::azure_egress_limit("Message: Egress is over the account limit", true)]
    fn test_is_throttle_error(#[case] msg: &str, #[case] expected: bool) {
        let err = make_generic_error(msg);
        assert_eq!(
            is_throttle_error(&err),
            expected,
            "is_throttle_error for '{}' should be {}",
            msg,
            expected
        );
    }

    #[test]
    fn test_non_generic_errors_are_not_throttle() {
        let err = object_store::Error::NotFound {
            path: "test".to_string(),
            source: "not found".into(),
        };
        assert!(!is_throttle_error(&err));
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[rstest]
    #[case::s3_part(
        ::http::Method::PUT,
        "https://bucket/object?partNumber=1&uploadId=id",
        true
    )]
    #[case::s3_complete(::http::Method::POST, "https://bucket/object?uploadId=id", true)]
    #[case::s3_initiate(::http::Method::POST, "https://bucket/object?uploads", false)]
    #[case::azure_block(
        ::http::Method::PUT,
        "https://account/object?comp=block&blockid=id",
        true
    )]
    #[case::azure_block_list(::http::Method::PUT, "https://account/object?comp=blocklist", true)]
    #[case::s3_list_parts(::http::Method::GET, "https://bucket/object?uploadId=id", false)]
    #[case::ordinary_put(::http::Method::PUT, "https://bucket/object", false)]
    fn test_is_multipart_upload_request(
        #[case] method: ::http::Method,
        #[case] uri: &str,
        #[case] expected: bool,
    ) {
        let request = ::http::Request::builder()
            .method(method)
            .uri(uri)
            .body(object_store::client::HttpRequestBody::empty())
            .unwrap();
        assert_eq!(is_multipart_upload_request(&request), expected);
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[rstest]
    #[case::plain(
        "<Error><Code>InternalError</Code><Message>internal error</Message></Error>",
        Some("InternalError")
    )]
    #[case::namespaced(
        "<Error xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Code>SlowDown</Code></Error>",
        Some("SlowDown")
    )]
    #[case::prefixed(
        "<s3:Error><s3:Code>ServerBusy</s3:Code></s3:Error>",
        Some("ServerBusy")
    )]
    #[case::declaration_and_bom(
        "\u{feff}<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Error>\n  <Code> RequestTimeout </Code>\n</Error>",
        Some("RequestTimeout")
    )]
    #[case::code_after_other_children(
        "<Error><Message>m</Message><Code>AccessDenied</Code></Error>",
        Some("AccessDenied")
    )]
    #[case::nested_code_is_not_a_child(
        "<Error><Details><Code>SlowDown</Code></Details></Error>",
        None
    )]
    #[case::success_document_with_throttle_words(
        "<?xml version=\"1.0\"?><CompleteMultipartUploadResult><Location>https://bucket/slowdown/object</Location><Key>throttle-demo/object</Key><Code>SlowDown</Code></CompleteMultipartUploadResult>",
        None
    )]
    #[case::error_without_code("<Error><Message>SlowDown</Message></Error>", None)]
    #[case::empty_code("<Error><Code></Code></Error>", None)]
    #[case::self_closing_root("<Error/>", None)]
    #[case::not_xml("SlowDown: Please reduce your request rate", None)]
    #[case::empty("", None)]
    #[case::comment_with_markup(
        "<Error><!-- <Code>SlowDown</Code> --><Code>NoSuchUpload</Code></Error>",
        Some("NoSuchUpload")
    )]
    #[case::cdata_code("<Error><Code><![CDATA[SlowDown]]></Code></Error>", Some("SlowDown"))]
    #[case::cdata_with_markup(
        "<Error><Message><![CDATA[<Code>SlowDown</Code>]]></Message><Code>NoSuchUpload</Code></Error>",
        Some("NoSuchUpload")
    )]
    #[case::attribute_with_gt(
        "<Error><Code note=\"a>b\">SlowDown</Code></Error>",
        Some("SlowDown")
    )]
    #[case::entity_in_code("<Error><Code>Slow&#68;own</Code></Error>", Some("SlowDown"))]
    fn test_error_document_code(#[case] body: &str, #[case] expected: Option<&str>) {
        assert_eq!(error_document_code(body).as_deref(), expected);
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[rstest]
    #[case::complete_success_key_with_throttle_word(
        200,
        "<CompleteMultipartUploadResult><Key>throttle-demo/object</Key></CompleteMultipartUploadResult>",
        true,
        false,
        false
    )]
    #[case::complete_200_internal_error(
        200,
        "<Error xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Code>InternalError</Code></Error>",
        true,
        true,
        false
    )]
    #[case::part_200_internal_error(
        200,
        "<Error><Code>InternalError</Code></Error>",
        false,
        false,
        false
    )]
    #[case::complete_200_slowdown(200, "<Error><Code>SlowDown</Code></Error>", true, true, true)]
    #[case::part_503_slowdown(503, "<Error><Code>SlowDown</Code></Error>", false, true, true)]
    #[case::azure_throttled_code(
        503,
        "<Error><Code>OperationTimedOutThrottled</Code></Error>",
        false,
        true,
        true
    )]
    #[case::other_code_with_throttle_word_in_key(
        404,
        "<Error><Code>NoSuchUpload</Code><Key>slowdown/object</Key></Error>",
        true,
        false,
        false
    )]
    #[case::success_not_xml(200, "slowdown", false, false, false)]
    #[case::error_not_xml_fallback(
        503,
        "SlowDown: Please reduce your request rate",
        false,
        true,
        true
    )]
    #[case::error_other_document_fallback(
        400,
        "<html><body>Request throttled</body></html>",
        true,
        true,
        true
    )]
    #[case::error_without_marker(400, "bad request", false, false, false)]
    fn test_classify_multipart_response_body(
        #[case] status: u16,
        #[case] body: &str,
        #[case] is_complete: bool,
        #[case] retryable: bool,
        #[case] throttle: bool,
    ) {
        let status = ::http::StatusCode::from_u16(status).unwrap();
        assert_eq!(
            classify_multipart_response_body(status, body, is_complete),
            BodyVerdict {
                retryable,
                throttle
            }
        );
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[rstest]
    #[case::ordinary_delete(::http::Method::DELETE, "https://bucket/object", true)]
    #[case::s3_bulk(::http::Method::POST, "https://bucket/?delete", true)]
    #[case::azure_batch(::http::Method::POST, "https://account/?comp=batch", true)]
    #[case::multipart_abort(::http::Method::DELETE, "https://bucket/object?uploadId=id", false)]
    #[case::unrelated_post(::http::Method::POST, "https://bucket/object?uploads", false)]
    fn test_is_delete_request(
        #[case] method: ::http::Method,
        #[case] uri: &str,
        #[case] expected: bool,
    ) {
        let request = ::http::Request::builder()
            .method(method)
            .uri(uri)
            .body(object_store::client::HttpRequestBody::empty())
            .unwrap();
        assert_eq!(is_delete_request(&request), expected);
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[derive(Debug)]
    struct CountingDeleteHttpService {
        requests: Arc<AtomicUsize>,
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[async_trait]
    impl HttpService for CountingDeleteHttpService {
        async fn call(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            Ok(::http::Response::builder()
                .status(::http::StatusCode::NO_CONTENT)
                .body(HttpResponseBody::from(Bytes::new()))
                .unwrap())
        }
    }

    #[cfg(any(feature = "aws", feature = "azure", feature = "gcp"))]
    #[tokio::test(start_paused = true)]
    async fn test_native_delete_http_requests_share_budget() {
        use futures::FutureExt;

        let state = AimdThrottleState::new(
            AimdThrottleConfig::default()
                .with_burst_capacity(0)
                .with_delete_aimd(AimdConfig::default().with_initial_rate(1.0)),
        )
        .unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let service = AimdHttpService {
            inner: HttpClient::new(CountingDeleteHttpService {
                requests: Arc::clone(&requests),
            }),
            write: Some(state.write),
            delete: Some(state.delete),
        };

        for (index, (method, uri)) in [
            (::http::Method::DELETE, "https://bucket/object"),
            (::http::Method::POST, "https://bucket/?delete"),
            (::http::Method::POST, "https://account/?comp=batch"),
            (::http::Method::DELETE, "https://bucket/object"),
        ]
        .into_iter()
        .enumerate()
        {
            let request = ::http::Request::builder()
                .method(method)
                .uri(uri)
                .body(object_store::client::HttpRequestBody::empty())
                .unwrap();
            let mut call = Box::pin(service.call(request));
            assert!(call.as_mut().now_or_never().is_none());
            assert_eq!(requests.load(Ordering::SeqCst), index);
            tokio::time::advance(std::time::Duration::from_secs(1)).await;
            assert_eq!(call.await.unwrap().status(), ::http::StatusCode::NO_CONTENT);
            assert_eq!(requests.load(Ordering::SeqCst), index + 1);
        }

        let abort = ::http::Request::builder()
            .method(::http::Method::DELETE)
            .uri("https://bucket/object?uploadId=id")
            .body(object_store::client::HttpRequestBody::empty())
            .unwrap();
        assert_eq!(
            service.call(abort).await.unwrap().status(),
            ::http::StatusCode::NO_CONTENT
        );
        assert_eq!(requests.load(Ordering::SeqCst), 5);
    }

    #[cfg(feature = "aws")]
    #[derive(Debug)]
    struct BulkDeleteConnector {
        requests: Arc<AtomicUsize>,
        failures_remaining: Arc<AtomicUsize>,
        error_code: &'static str,
    }

    #[cfg(feature = "aws")]
    impl HttpConnector for BulkDeleteConnector {
        fn connect(&self, _options: &ClientOptions) -> OSResult<HttpClient> {
            Ok(HttpClient::new(BulkDeleteService {
                requests: Arc::clone(&self.requests),
                failures_remaining: Arc::clone(&self.failures_remaining),
                error_code: self.error_code,
            }))
        }
    }

    #[cfg(feature = "aws")]
    #[derive(Debug)]
    struct BulkDeleteService {
        requests: Arc<AtomicUsize>,
        failures_remaining: Arc<AtomicUsize>,
        error_code: &'static str,
    }

    #[cfg(feature = "aws")]
    #[async_trait]
    impl HttpService for BulkDeleteService {
        async fn call(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            assert_eq!(request.method(), ::http::Method::POST);
            assert_eq!(request.uri().query(), Some("delete"));
            self.requests.fetch_add(1, Ordering::SeqCst);
            if self
                .failures_remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                return Ok(::http::Response::builder()
                    .status(::http::StatusCode::SERVICE_UNAVAILABLE)
                    .body(HttpResponseBody::from(
                        "<Error><Code>SlowDown</Code></Error>".to_string(),
                    ))
                    .unwrap());
            }
            Ok(::http::Response::builder()
                .status(::http::StatusCode::OK)
                .body(HttpResponseBody::from(
                    format!("<DeleteResult><Deleted><Key>fragments/1</Key></Deleted><Error><Key>fragments/2</Key><Code>{}</Code><Message>denied</Message></Error></DeleteResult>", self.error_code),
                ))
                .unwrap())
        }
    }

    #[cfg(feature = "aws")]
    #[tokio::test]
    #[rstest]
    #[case::single_request(0, Some(1), "AccessDenied", true, 0.0, 110.0)]
    #[case::native_retry(1, Some(2), "AccessDenied", true, 0.0, 50.0)]
    #[case::exhausted_retries(3, Some(2), "SlowDown", false, 0.0, 50.0)]
    #[case::exhausted_retries_counted_once(3, Some(2), "SlowDown", false, 0.7, 110.0)]
    #[case::parsed_slowdown(0, Some(1), "slowdown", true, 0.0, 50.0)]
    #[case::native_slowdown(0, None, "SlowDown", false, 0.0, 50.0)]
    async fn test_native_s3_bulk_delete_keeps_per_object_results(
        #[case] failures: usize,
        #[case] expected_requests: Option<usize>,
        #[case] error_code: &'static str,
        #[case] has_per_object_result: bool,
        #[case] throttle_threshold: f64,
        #[case] expected_rate: f64,
    ) {
        use futures::stream;
        use object_store::aws::AmazonS3Builder;
        use object_store::{BackoffConfig, RetryConfig};

        let requests = Arc::new(AtomicUsize::new(0));
        let failures_remaining = Arc::new(AtomicUsize::new(failures));
        let config = AimdThrottleConfig::default().with_delete_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_throttle_threshold(throttle_threshold)
                .with_additive_increment(10.0)
                .with_window_duration(std::time::Duration::from_millis(100)),
        );
        let state = AimdThrottleState::new(config).unwrap();
        // With one prior success, two throttled attempts give a 2/3 ratio.
        // Counting the final error again would give 3/4 and cross the 0.7 threshold.
        state
            .delete
            .controller
            .record_outcome(RequestOutcome::Success);
        let connector = AimdHttpConnector::new(
            BulkDeleteConnector {
                requests: Arc::clone(&requests),
                failures_remaining,
                error_code,
            },
            Some(&state),
        );
        let target = AmazonS3Builder::new()
            .with_bucket_name("bucket")
            .with_region("us-east-1")
            .with_skip_signature(true)
            .with_retry(RetryConfig {
                max_retries: 1,
                backoff: BackoffConfig {
                    init_backoff: std::time::Duration::from_millis(1),
                    ..Default::default()
                },
                ..Default::default()
            })
            .with_http_connector(connector)
            .build()
            .unwrap();
        let throttled = AimdThrottledStore::new_with_state(Arc::new(target), state, true);
        let results = throttled
            .delete_stream(
                stream::iter(vec![
                    Ok(Path::from("fragments/1")),
                    Ok(Path::from("fragments/2")),
                ])
                .boxed(),
            )
            .collect::<Vec<_>>()
            .await;

        let request_count = requests.load(Ordering::SeqCst);
        if let Some(expected_requests) = expected_requests {
            assert_eq!(request_count, expected_requests);
        } else {
            assert!((1..=2).contains(&request_count));
        }
        if has_per_object_result {
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].as_ref().unwrap(), &Path::from("fragments/1"));
            assert!(matches!(
                results[1],
                Err(object_store::Error::Generic { .. })
            ));
        }
        assert!(results.iter().any(|result| {
            result
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains(error_code))
        }));

        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        throttled
            .delete
            .controller
            .record_outcome(RequestOutcome::Success);
        assert_eq!(throttled.delete.controller.current_rate(), expected_rate);
    }

    #[cfg(feature = "azure")]
    #[derive(Debug)]
    struct AzureBatchConnector {
        requests: Arc<AtomicUsize>,
    }

    #[cfg(feature = "azure")]
    impl HttpConnector for AzureBatchConnector {
        fn connect(&self, _options: &ClientOptions) -> OSResult<HttpClient> {
            Ok(HttpClient::new(AzureBatchService {
                requests: Arc::clone(&self.requests),
            }))
        }
    }

    #[cfg(feature = "azure")]
    #[derive(Debug)]
    struct AzureBatchService {
        requests: Arc<AtomicUsize>,
    }

    #[cfg(feature = "azure")]
    #[async_trait]
    impl HttpService for AzureBatchService {
        async fn call(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            assert_eq!(request.method(), ::http::Method::POST);
            assert!(
                request
                    .uri()
                    .query()
                    .unwrap_or_default()
                    .contains("comp=batch")
            );
            self.requests.fetch_add(1, Ordering::SeqCst);
            let body = "--batchresponse_test\r\nContent-Type: application/http\r\nContent-ID: 0\r\n\r\nHTTP/1.1 202 Accepted\r\n\r\n--batchresponse_test\r\nContent-Type: application/http\r\nContent-ID: 1\r\n\r\nHTTP/1.1 503 ServerBusy\r\nx-ms-error-code: ServerBusy\r\n\r\n--batchresponse_test--\r\n";
            Ok(::http::Response::builder()
                .status(::http::StatusCode::ACCEPTED)
                .header(
                    "Content-Type",
                    "multipart/mixed; boundary=batchresponse_test",
                )
                .body(HttpResponseBody::from(body.to_string()))
                .unwrap())
        }
    }

    #[cfg(feature = "azure")]
    #[tokio::test]
    async fn test_native_azure_batch_server_busy_reduces_rate() {
        use futures::stream;
        use object_store::azure::MicrosoftAzureBuilder;

        let requests = Arc::new(AtomicUsize::new(0));
        let config = AimdThrottleConfig::default().with_delete_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_additive_increment(10.0)
                .with_window_duration(std::time::Duration::from_millis(100)),
        );
        let state = AimdThrottleState::new(config).unwrap();
        let connector = AimdHttpConnector::new(
            AzureBatchConnector {
                requests: Arc::clone(&requests),
            },
            Some(&state),
        );
        let target = MicrosoftAzureBuilder::new()
            .with_account("account")
            .with_container_name("container")
            .with_skip_signature(true)
            .with_http_connector(connector)
            .build()
            .unwrap();
        let throttled = AimdThrottledStore::new_with_state(Arc::new(target), state, true);
        let results = throttled
            .delete_stream(
                stream::iter(vec![
                    Ok(Path::from("fragments/1")),
                    Ok(Path::from("fragments/2")),
                ])
                .boxed(),
            )
            .collect::<Vec<_>>()
            .await;

        assert_eq!(requests.load(Ordering::SeqCst), 1);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_ref().unwrap(), &Path::from("fragments/1"));
        let error = results[1].as_ref().unwrap_err();
        assert!(matches!(error, object_store::Error::Generic { .. }));
        assert!(error.to_string().contains("ServerBusy"), "{error}");

        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        throttled
            .delete
            .controller
            .record_outcome(RequestOutcome::Success);
        assert_eq!(throttled.delete.controller.current_rate(), 50.0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_generic_delete_stream_shares_budget_and_preserves_errors() {
        use futures::{FutureExt, stream};

        let target = Arc::new(InMemory::new());
        let first = Path::from("fragments/1");
        let second = Path::from("fragments/2");
        for path in [&first, &second] {
            target
                .put(path, PutPayload::from_static(b"data"))
                .await
                .unwrap();
        }
        let config = AimdThrottleConfig::default()
            .with_burst_capacity(0)
            .with_delete_aimd(AimdConfig::default().with_initial_rate(1.0));
        let throttled = Arc::new(AimdThrottledStore::new(target.clone(), config).unwrap());
        let input_error = object_store::Error::NotFound {
            path: "input".to_string(),
            source: "upstream failure".into(),
        };
        let mut results = throttled.delete_stream(
            stream::iter(vec![
                Ok(first.clone()),
                Err(input_error),
                Ok(second.clone()),
            ])
            .boxed(),
        );

        assert!(results.next().now_or_never().is_none());
        assert!(target.head(&first).await.is_ok());
        assert!(target.head(&second).await.is_ok());

        tokio::time::advance(std::time::Duration::from_secs(1)).await;
        assert_eq!(results.next().await.unwrap().unwrap(), first);
        let error = results.next().await.unwrap().unwrap_err();
        assert!(matches!(error, object_store::Error::NotFound { .. }));
        assert!(error.to_string().contains("input"));
        assert!(target.head(&second).await.is_ok());

        tokio::time::advance(std::time::Duration::from_secs(1)).await;
        assert_eq!(results.next().await.unwrap().unwrap(), second);
        assert!(results.next().await.is_none());
        assert!(target.head(&first).await.is_err());
        assert!(target.head(&second).await.is_err());
    }

    #[tokio::test]
    async fn test_generic_delete_throttle_error_reduces_rate() {
        let target = Arc::new(RetryTestMockStore::new(1));
        let config = AimdThrottleConfig {
            max_retries: 0,
            ..Default::default()
        }
        .with_delete_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_window_duration(std::time::Duration::from_millis(100)),
        );
        let throttled = AimdThrottledStore::new(target.clone(), config).unwrap();
        let results = throttled
            .delete_stream(futures::stream::iter([Ok(Path::from("object"))]).boxed())
            .collect::<Vec<_>>()
            .await;
        assert_eq!(results.len(), 1);
        let error = results[0].as_ref().unwrap_err();
        assert!(matches!(error, object_store::Error::Generic { .. }));
        assert!(
            error
                .to_string()
                .contains("Server returned non-2xx status code: 503")
        );
        assert_eq!(*target.errors_remaining.lock().unwrap(), 0);

        // AIMD uses std::time::Instant, which Tokio's paused clock cannot advance.
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        throttled
            .delete
            .controller
            .record_outcome(RequestOutcome::Success);
        assert_eq!(throttled.delete.controller.current_rate(), 50.0);
    }

    /// One page of a fixed directory, counting the requests that reached it.
    #[derive(Default)]
    struct CountingListStore {
        calls: AtomicUsize,
        fail_with: Option<String>,
    }

    #[async_trait]
    impl PaginatedListStore for CountingListStore {
        async fn list_paginated(
            &self,
            _prefix: Option<&str>,
            _opts: PaginatedListOptions,
        ) -> OSResult<PaginatedListResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.fail_with {
                Some(message) => Err(make_generic_error(message)),
                None => Ok(PaginatedListResult {
                    result: ListResult {
                        common_prefixes: vec![Path::from("prefix/child")],
                        objects: Vec::new(),
                        extensions: Default::default(),
                    },
                    page_token: None,
                }),
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_paginated_lister_acquires_a_token_before_listing() {
        let lister = Arc::new(CountingListStore::default());
        let throttled = AimdThrottledStore::new(
            Arc::new(InMemory::new()) as Arc<dyn ObjectStore>,
            list_start_throttle_config(),
        )
        .unwrap();
        let throttled_lister = throttled.wrap_paginated(lister.clone());

        let mut page = Box::pin(
            throttled_lister.list_paginated(Some("prefix/"), PaginatedListOptions::default()),
        );
        // With rate=10 tokens/s and burst_capacity=0, the token acquisition sleeps for
        // 100 ms. A 50 ms timeout must expire before that.
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut page)
                .await
                .is_err()
        );
        assert_eq!(lister.calls.load(Ordering::SeqCst), 0);

        let page = tokio::time::timeout(std::time::Duration::from_millis(300), page)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(page.result.common_prefixes.len(), 1);
        assert_eq!(lister.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_paginated_lister_throttle_errors_decrease_rate() {
        let lister = Arc::new(CountingListStore {
            calls: AtomicUsize::new(0),
            fail_with: Some(THROTTLE_ERROR_RESPONSE.to_string()),
        });
        let mut config = AimdThrottleConfig::default().with_list_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_window_duration(std::time::Duration::from_millis(1)),
        );
        config.max_retries = 1;
        // The AIMD window is only evaluated when an outcome is recorded after the
        // window has elapsed, so the retry backoff must outlast `window_duration`.
        // With a zero backoff the two attempts against this in-memory lister can
        // finish inside the first window (observed on Windows), leaving the rate
        // untouched.
        config.min_backoff_ms = 5;
        config.max_backoff_ms = 5;
        let throttled =
            AimdThrottledStore::new(Arc::new(InMemory::new()) as Arc<dyn ObjectStore>, config)
                .unwrap();
        let throttled_lister = throttled.wrap_paginated(lister.clone());

        assert!(
            throttled_lister
                .list_paginated(Some("prefix/"), PaginatedListOptions::default())
                .await
                .is_err()
        );

        // The request was retried once, and the throttle response pushed the rate down.
        assert_eq!(lister.calls.load(Ordering::SeqCst), 2);
        assert!(throttled.list.controller.current_rate() < 100.0);
    }

    #[tokio::test]
    async fn test_basic_put_get_through_wrapper() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default();
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        let path = Path::from("test/file.txt");
        let data = PutPayload::from_static(b"hello world");
        throttled.put(&path, data).await.unwrap();

        let result = throttled.get(&path).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"hello world");
    }

    #[tokio::test]
    async fn test_rate_decreases_on_throttle() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default().with_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_window_duration(std::time::Duration::from_millis(10)),
        );
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        let initial_rate = throttled.read.controller.current_rate();
        assert_eq!(initial_rate, 100.0);

        // Simulate a throttle outcome directly
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Throttled);

        // Wait for window to expire and trigger evaluation
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Success);

        let new_rate = throttled.read.controller.current_rate();
        assert!(
            new_rate < initial_rate,
            "Rate should decrease after throttle: {} < {}",
            new_rate,
            initial_rate
        );
    }

    #[tokio::test]
    async fn test_rate_recovers_on_success() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default().with_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_additive_increment(10.0)
                .with_window_duration(std::time::Duration::from_millis(10)),
        );
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        // First decrease via throttle
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Throttled);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Success);
        let decreased_rate = throttled.read.controller.current_rate();
        assert_eq!(decreased_rate, 50.0);

        // Now recover via success
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Success);
        let recovered_rate = throttled.read.controller.current_rate();
        assert_eq!(recovered_rate, 60.0);
    }

    #[tokio::test]
    async fn test_as_dyn_object_store() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let throttled: Arc<dyn ObjectStore> =
            Arc::new(AimdThrottledStore::new(store, AimdThrottleConfig::default()).unwrap());

        let path = Path::from("test/data.bin");
        let data = PutPayload::from_static(b"test data");
        throttled.put(&path, data).await.unwrap();

        let result = throttled.get(&path).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"test data");
    }

    #[tokio::test]
    async fn test_token_bucket_delays_when_exhausted() {
        let store = Arc::new(InMemory::new());
        // Very low rate and burst capacity to force waiting
        let config = AimdThrottleConfig::default()
            .with_burst_capacity(1)
            .with_aimd(AimdConfig::default().with_initial_rate(10.0));
        let throttled = Arc::new(AimdThrottledStore::new(store, config).unwrap());

        let path = Path::from("test/file.txt");
        let data = PutPayload::from_static(b"data");
        throttled.put(&path, data).await.unwrap();

        // After consuming the burst token, the next request should take ~100ms
        // (1 token / 10 tokens-per-sec). We verify it takes at least 50ms.
        let start = std::time::Instant::now();
        let data2 = PutPayload::from_static(b"data2");
        throttled.put(&path, data2).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed >= std::time::Duration::from_millis(50),
            "Expected delay for token refill, but elapsed was {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_list_observes_outcomes() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default();
        let throttled = AimdThrottledStore::new(store.clone(), config).unwrap();

        let path = Path::from("prefix/file.txt");
        let data = PutPayload::from_static(b"data");
        store.put(&path, data).await.unwrap();

        let items: Vec<_> = throttled.list(Some(&Path::from("prefix"))).collect().await;
        assert_eq!(items.len(), 1);
        assert!(items[0].is_ok());
    }

    /// A mock store whose `list` stream yields a configurable sequence of
    /// Ok / throttle-error items. Used to verify that the AIMD wrapper
    /// observes errors surfaced inside list streams.
    struct ThrottlingListMockStore {
        inner: InMemory,
        /// Number of throttle errors to inject at the start of each list call.
        throttle_count: usize,
    }

    impl Display for ThrottlingListMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "ThrottlingListMockStore")
        }
    }

    impl Debug for ThrottlingListMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ThrottlingListMockStore").finish()
        }
    }

    #[async_trait]
    impl ObjectStore for ThrottlingListMockStore {
        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }
        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }
        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            self.inner.get_opts(location, options).await
        }
        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }
        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            self.inner.delete_stream(locations)
        }
        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            let n = self.throttle_count;
            let inner_stream = self.inner.list(prefix);
            let errors = futures::stream::iter((0..n).map(|_| {
                Err(object_store::Error::Generic {
                    store: "ThrottlingListMock",
                    source: "request failed, after 3 retries, max_retries: 5, retry_timeout: 60s"
                        .into(),
                })
            }));
            errors.chain(inner_stream).boxed()
        }
        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }
        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.inner.copy_opts(from, to, opts).await
        }
    }

    #[tokio::test]
    async fn test_list_stream_throttle_errors_decrease_rate() {
        let mock = Arc::new(ThrottlingListMockStore {
            inner: InMemory::new(),
            throttle_count: 5,
        });

        // Seed a file so the real items come through after the errors.
        mock.put(
            &Path::from("prefix/file.txt"),
            PutPayload::from_static(b"data"),
        )
        .await
        .unwrap();

        let config = AimdThrottleConfig::default().with_list_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_window_duration(std::time::Duration::from_millis(10)),
        );
        let throttled = AimdThrottledStore::new(mock as Arc<dyn ObjectStore>, config).unwrap();

        let initial_rate = throttled.list.controller.current_rate();
        assert_eq!(initial_rate, 100.0);

        let items: Vec<_> = throttled.list(Some(&Path::from("prefix"))).collect().await;

        // 5 errors + 1 real item
        assert_eq!(items.len(), 6);
        assert!(items[0].is_err());
        assert!(items[5].is_ok());

        // Wait for the AIMD window to expire and trigger evaluation.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .list
            .controller
            .record_outcome(RequestOutcome::Success);

        let new_rate = throttled.list.controller.current_rate();
        assert!(
            new_rate < initial_rate,
            "List rate should decrease after stream throttle errors: {} < {}",
            new_rate,
            initial_rate
        );
    }

    struct CountingListStartStore {
        inner: InMemory,
        list_calls: AtomicUsize,
        offset_calls: AtomicUsize,
    }

    impl Default for CountingListStartStore {
        fn default() -> Self {
            Self {
                inner: InMemory::new(),
                list_calls: AtomicUsize::new(0),
                offset_calls: AtomicUsize::new(0),
            }
        }
    }

    impl CountingListStartStore {
        fn list_calls(&self) -> usize {
            self.list_calls.load(Ordering::SeqCst)
        }

        fn offset_calls(&self) -> usize {
            self.offset_calls.load(Ordering::SeqCst)
        }
    }

    impl Display for CountingListStartStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "CountingListStartStore")
        }
    }

    impl Debug for CountingListStartStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CountingListStartStore").finish()
        }
    }

    #[async_trait]
    impl ObjectStore for CountingListStartStore {
        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            self.inner.get_opts(location, options).await
        }

        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.list_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.offset_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.inner.copy_opts(from, to, opts).await
        }
    }

    fn list_start_throttle_config() -> AimdThrottleConfig {
        // Use a low rate (10 tokens/s) so that the token-acquisition sleep is
        // 1/10 = 100 ms — well above the 50 ms timeout used in assertions,
        // avoiding flakiness from coarse OS timer resolution (e.g. Windows ~16 ms).
        AimdThrottleConfig::default()
            .with_burst_capacity(0)
            .with_list_aimd(AimdConfig::default().with_initial_rate(10.0))
    }

    #[tokio::test(start_paused = true)]
    async fn test_list_acquires_token_before_starting_underlying_stream() {
        let store = Arc::new(CountingListStartStore::default());
        store
            .put(
                &Path::from("prefix/file.txt"),
                PutPayload::from_static(b"data"),
            )
            .await
            .unwrap();
        let throttled = AimdThrottledStore::new(
            store.clone() as Arc<dyn ObjectStore>,
            list_start_throttle_config(),
        )
        .unwrap();

        let mut stream = throttled.list(Some(&Path::from("prefix")));
        assert_eq!(store.list_calls(), 0);
        // With rate=10 tokens/s and burst_capacity=0, the token acquisition
        // sleeps for 100 ms. A 50 ms timeout must expire before that.
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), stream.next())
                .await
                .is_err()
        );
        assert_eq!(store.list_calls(), 0);

        let item = tokio::time::timeout(std::time::Duration::from_millis(300), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(item.location, Path::from("prefix/file.txt"));
        assert_eq!(store.list_calls(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn test_list_with_offset_acquires_token_before_starting_underlying_stream() {
        let store = Arc::new(CountingListStartStore::default());
        store
            .put(&Path::from("prefix/b"), PutPayload::from_static(b"data"))
            .await
            .unwrap();
        let throttled = AimdThrottledStore::new(
            store.clone() as Arc<dyn ObjectStore>,
            list_start_throttle_config(),
        )
        .unwrap();

        let mut stream =
            throttled.list_with_offset(Some(&Path::from("prefix")), &Path::from("prefix/a"));
        assert_eq!(store.offset_calls(), 0);
        // With rate=10 tokens/s and burst_capacity=0, the token acquisition
        // sleeps for 100 ms. A 50 ms timeout must expire before that.
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), stream.next())
                .await
                .is_err()
        );
        assert_eq!(store.offset_calls(), 0);

        let item = tokio::time::timeout(std::time::Duration::from_millis(300), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(item.location, Path::from("prefix/b"));
        assert_eq!(store.offset_calls(), 1);
    }

    #[tokio::test]
    async fn test_per_category_independence() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default().with_aimd(
            AimdConfig::default()
                .with_initial_rate(100.0)
                .with_decrease_factor(0.5)
                .with_window_duration(std::time::Duration::from_millis(10)),
        );
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        // Push the read controller into a throttled state
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Throttled);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        throttled
            .read
            .controller
            .record_outcome(RequestOutcome::Success);

        let read_rate = throttled.read.controller.current_rate();
        let write_rate = throttled.write.controller.current_rate();
        let delete_rate = throttled.delete.controller.current_rate();
        let list_rate = throttled.list.controller.current_rate();

        assert_eq!(read_rate, 50.0, "Read rate should have decreased");
        assert_eq!(write_rate, 100.0, "Write rate should be unaffected");
        assert_eq!(delete_rate, 100.0, "Delete rate should be unaffected");
        assert_eq!(list_rate, 100.0, "List rate should be unaffected");
    }

    #[tokio::test]
    async fn test_per_category_config() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default()
            .with_read_aimd(AimdConfig::default().with_initial_rate(200.0))
            .with_write_aimd(AimdConfig::default().with_initial_rate(100.0))
            .with_delete_aimd(AimdConfig::default().with_initial_rate(50.0))
            .with_list_aimd(AimdConfig::default().with_initial_rate(25.0));
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        assert_eq!(throttled.read.controller.current_rate(), 200.0);
        assert_eq!(throttled.write.controller.current_rate(), 100.0);
        assert_eq!(throttled.delete.controller.current_rate(), 50.0);
        assert_eq!(throttled.list.controller.current_rate(), 25.0);
    }

    /// A mock [`ObjectStore`] that measures request rate over a sliding window
    /// and returns 503 errors when the rate exceeds a configurable threshold.
    /// Write and metadata-only operations are not rate-limited.
    struct RateLimitingMockStore {
        inner: InMemory,
        /// Timestamps of recent successful (admitted) requests.
        timestamps: std::sync::Mutex<VecDeque<std::time::Instant>>,
        /// Maximum requests allowed within `window`.
        max_per_window: usize,
        /// Sliding window duration.
        window: std::time::Duration,
        success_count: AtomicU64,
        throttle_count: AtomicU64,
    }

    impl RateLimitingMockStore {
        fn new(max_per_window: usize, window: std::time::Duration) -> Self {
            Self {
                inner: InMemory::new(),
                timestamps: std::sync::Mutex::new(VecDeque::new()),
                max_per_window,
                window,
                success_count: AtomicU64::new(0),
                throttle_count: AtomicU64::new(0),
            }
        }

        /// Returns `true` if the request is admitted, `false` if throttled.
        fn check_rate(&self) -> bool {
            let mut ts = self.timestamps.lock().unwrap();
            let now = std::time::Instant::now();
            while let Some(&front) = ts.front() {
                if now.duration_since(front) > self.window {
                    ts.pop_front();
                } else {
                    break;
                }
            }
            if ts.len() >= self.max_per_window {
                self.throttle_count.fetch_add(1, Ordering::Relaxed);
                false
            } else {
                ts.push_back(now);
                self.success_count.fetch_add(1, Ordering::Relaxed);
                true
            }
        }

        fn throttle_error() -> object_store::Error {
            object_store::Error::Generic {
                store: "RateLimitingMock",
                source: THROTTLE_ERROR_RESPONSE.into(),
            }
        }
    }

    impl Display for RateLimitingMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "RateLimitingMockStore")
        }
    }

    impl Debug for RateLimitingMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RateLimitingMockStore").finish()
        }
    }

    #[async_trait]
    impl ObjectStore for RateLimitingMockStore {
        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            if self.check_rate() {
                self.inner.get_opts(location, options).await
            } else {
                Err(Self::throttle_error())
            }
        }

        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            if self.check_rate() {
                self.inner.get_ranges(location, ranges).await
            } else {
                Err(Self::throttle_error())
            }
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.inner.copy_opts(from, to, opts).await
        }
    }

    /// Verify that multiple concurrent readers sharing an AIMD-throttled store
    /// converge to the backend's actual capacity.
    ///
    /// Setup:
    /// - Mock backend allows 30 requests per 100ms (= 300 req/s).
    /// - 5 reader tasks, each with their own [`AimdThrottledStore`] wrapping
    ///   the shared mock.
    /// - AIMD: 100ms window, initial rate 100 req/s, decrease 0.5, increase 2.
    /// - Readers issue `head()` requests as fast as the throttle allows for 2s.
    ///
    /// Expected behaviour:
    /// - Initial burst (100 burst tokens × 5 readers) overshoots the mock
    ///   capacity, causing many 503s. Each reader's AIMD halves its rate.
    /// - After the transient, each reader converges to ~60 req/s (300/5).
    /// - Over 2 seconds, total successful requests should be in the range
    ///   [300, 900] (theoretical max ≈ 600).
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_aimd_throttle_under_concurrent_load() {
        let mock = Arc::new(RateLimitingMockStore::new(
            30,
            std::time::Duration::from_millis(100),
        ));

        // Seed a test file so head() succeeds when admitted.
        let path = Path::from("test/data.bin");
        mock.put(&path, PutPayload::from_static(b"test data"))
            .await
            .unwrap();

        let aimd = AimdConfig::default()
            .with_initial_rate(100.0)
            .with_decrease_factor(0.5)
            .with_additive_increment(2.0)
            .with_window_duration(std::time::Duration::from_millis(100));
        let throttle_config = AimdThrottleConfig::default()
            .with_aimd(aimd)
            .with_burst_capacity(100);

        let num_readers = 5;
        let test_duration = std::time::Duration::from_secs(2);
        let mut handles = Vec::new();

        for _ in 0..num_readers {
            let store = Arc::new(
                AimdThrottledStore::new(
                    mock.clone() as Arc<dyn ObjectStore>,
                    throttle_config.clone(),
                )
                .unwrap(),
            );
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let deadline = std::time::Instant::now() + test_duration;
                let mut count = 0u64;
                while std::time::Instant::now() < deadline {
                    let _ = store.head(&p).await;
                    count += 1;
                }
                count
            }));
        }

        let mut total_reader_requests = 0u64;
        for handle in handles {
            total_reader_requests += handle.await.unwrap();
        }

        let successes = mock.success_count.load(Ordering::Relaxed);
        let throttled = mock.throttle_count.load(Ordering::Relaxed);
        let total_mock = successes + throttled;

        // Mock-side count >= reader-side count because the AIMD layer retries
        // throttle errors internally, causing multiple mock calls per reader call.
        assert!(
            total_mock >= total_reader_requests,
            "Mock-side count ({total_mock}) should be >= reader-side count ({total_reader_requests})"
        );

        // Mock capacity is 30/100ms = 300 req/s. Over 2s the theoretical max is
        // ~600 successful requests. With AIMD ramp-up, expect somewhat fewer.
        assert!(
            successes >= 300,
            "Expected >= 300 successful requests over 2s, got {successes}"
        );
        assert!(
            successes <= 900,
            "Expected <= 900 successful requests, got {successes}"
        );

        // The initial burst exceeds mock capacity, so throttling must occur.
        assert!(throttled > 0, "Expected some throttled requests but got 0");

        // Without AIMD, raw tokio tasks against InMemory would fire 100k+ req/s.
        // AIMD should keep the total well under 5000 over 2s.
        assert!(
            total_mock <= 5000,
            "AIMD should limit total requests, got {total_mock}"
        );
    }

    /// A mock store that returns a configurable number of throttle errors
    /// before succeeding on `get` and delete operations. Used to test the retry logic
    /// inside `OperationThrottle::throttled()`.
    struct RetryTestMockStore {
        inner: InMemory,
        /// Number of throttle errors remaining before success.
        errors_remaining: std::sync::Mutex<usize>,
        /// Total number of `get` calls observed.
        get_call_count: AtomicU64,
    }

    impl RetryTestMockStore {
        fn new(errors_before_success: usize) -> Self {
            Self {
                inner: InMemory::new(),
                errors_remaining: std::sync::Mutex::new(errors_before_success),
                get_call_count: AtomicU64::new(0),
            }
        }

        fn take_throttle_error(&self) -> OSResult<()> {
            let mut remaining = self.errors_remaining.lock().unwrap();
            if *remaining > 0 {
                *remaining -= 1;
                Err(object_store::Error::Generic {
                    store: "RetryTestMock",
                    source: THROTTLE_ERROR_RESPONSE.into(),
                })
            } else {
                Ok(())
            }
        }
    }

    impl Display for RetryTestMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "RetryTestMockStore")
        }
    }

    impl Debug for RetryTestMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RetryTestMockStore").finish()
        }
    }

    #[async_trait]
    impl ObjectStore for RetryTestMockStore {
        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }
        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }
        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            self.get_call_count.fetch_add(1, Ordering::Relaxed);
            self.take_throttle_error()?;
            self.inner.get_opts(location, options).await
        }
        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }
        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            match self.take_throttle_error() {
                Ok(()) => self.inner.delete_stream(locations),
                Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
            }
        }
        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list(prefix)
        }
        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }
        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.inner.copy_opts(from, to, opts).await
        }
    }

    #[tokio::test]
    async fn test_throttled_retries_on_throttle_error_then_succeeds() {
        // Mock returns 2 throttle errors then succeeds (within MAX_RETRIES=3)
        let mock = Arc::new(RetryTestMockStore::new(2));
        let path = Path::from("test/retry.txt");
        mock.put(&path, PutPayload::from_static(b"retry data"))
            .await
            .unwrap();

        let config = AimdThrottleConfig::default();
        let throttled =
            AimdThrottledStore::new(mock.clone() as Arc<dyn ObjectStore>, config).unwrap();

        let result = throttled.get(&path).await;
        assert!(result.is_ok(), "Expected success after retries");

        let bytes = result.unwrap().bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"retry data");

        // Should have called get 3 times total: 2 failures + 1 success
        assert_eq!(mock.get_call_count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_throttled_fails_after_max_retries_exceeded() {
        // Mock returns 4 throttle errors (more than MAX_RETRIES=3),
        // so all 4 attempts (initial + 3 retries) will fail.
        let mock = Arc::new(RetryTestMockStore::new(10));
        let path = Path::from("test/fail.txt");
        mock.put(&path, PutPayload::from_static(b"fail data"))
            .await
            .unwrap();

        let config = AimdThrottleConfig::default();
        let throttled =
            AimdThrottledStore::new(mock.clone() as Arc<dyn ObjectStore>, config).unwrap();

        let result = throttled.get(&path).await;
        assert!(result.is_err(), "Expected error after max retries");
        let err = result.unwrap_err();
        assert!(is_throttle_error(&err));

        let lance_error = lance_core::Error::from(err);
        let error_message = lance_error.to_string();
        assert!(error_message.contains("x-ms-request-id"));
        assert!(error_message.contains("azure-request-id"));

        // Should have called get 4 times: initial attempt + 3 retries
        assert_eq!(mock.get_call_count.load(Ordering::Relaxed), 4);
    }

    #[cfg(feature = "aws")]
    #[derive(Debug)]
    struct MultipartRetryState {
        failures_remaining: AtomicUsize,
        part_uris: std::sync::Mutex<Vec<String>>,
        complete_failures_remaining: AtomicUsize,
        /// Body of every complete request that reached the mock, in order.
        complete_bodies: std::sync::Mutex<Vec<Bytes>>,
        complete_error_on_200: bool,
    }

    #[cfg(feature = "aws")]
    #[derive(Debug)]
    struct MultipartRetryConnector {
        state: Arc<MultipartRetryState>,
    }

    #[cfg(feature = "aws")]
    impl HttpConnector for MultipartRetryConnector {
        fn connect(&self, _options: &ClientOptions) -> object_store::Result<HttpClient> {
            Ok(HttpClient::new(MultipartRetryService {
                state: Arc::clone(&self.state),
            }))
        }
    }

    #[cfg(feature = "aws")]
    #[derive(Debug)]
    struct MultipartRetryService {
        state: Arc<MultipartRetryState>,
    }

    #[cfg(feature = "aws")]
    #[async_trait]
    impl HttpService for MultipartRetryService {
        async fn call(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            let method = request.method().clone();
            let query = request.uri().query().unwrap_or_default();
            let (status, body, e_tag) = if method == ::http::Method::POST
                && query
                    .split('&')
                    .any(|part| part == "uploads" || part == "uploads=")
            {
                (
                    ::http::StatusCode::OK,
                    "<InitiateMultipartUploadResult><Bucket>bucket</Bucket><Key>object</Key><UploadId>upload-id</UploadId></InitiateMultipartUploadResult>",
                    None,
                )
            } else if method == ::http::Method::PUT && query.contains("partNumber=") {
                self.state
                    .part_uris
                    .lock()
                    .unwrap()
                    .push(request.uri().to_string());
                let mut remaining = self.state.failures_remaining.load(Ordering::SeqCst);
                let should_fail = loop {
                    let Some(next) = remaining.checked_sub(1) else {
                        break false;
                    };
                    match self.state.failures_remaining.compare_exchange_weak(
                        remaining,
                        next,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break true,
                        Err(actual) => remaining = actual,
                    }
                };
                if should_fail {
                    (
                        ::http::StatusCode::SERVICE_UNAVAILABLE,
                        "<Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message></Error>",
                        None,
                    )
                } else {
                    (::http::StatusCode::OK, "", Some("\"part-etag\""))
                }
            } else if method == ::http::Method::POST && query.contains("uploadId=") {
                let body = request
                    .body()
                    .as_bytes()
                    .cloned()
                    .expect("S3 sends the complete request as one contiguous body");
                self.state.complete_bodies.lock().unwrap().push(body);
                let mut remaining = self
                    .state
                    .complete_failures_remaining
                    .load(Ordering::SeqCst);
                let should_fail = loop {
                    let Some(next) = remaining.checked_sub(1) else {
                        break false;
                    };
                    match self
                        .state
                        .complete_failures_remaining
                        .compare_exchange_weak(remaining, next, Ordering::SeqCst, Ordering::SeqCst)
                    {
                        Ok(_) => break true,
                        Err(actual) => remaining = actual,
                    }
                };
                if !should_fail {
                    (
                        ::http::StatusCode::OK,
                        "<CompleteMultipartUploadResult><Location>https://bucket/object</Location><Bucket>bucket</Bucket><Key>object</Key><ETag>\"object-etag\"</ETag></CompleteMultipartUploadResult>",
                        None,
                    )
                } else if self.state.complete_error_on_200 {
                    (
                        ::http::StatusCode::OK,
                        "<Error xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Code>InternalError</Code><Message>We encountered an internal error. Please try again.</Message></Error>",
                        None,
                    )
                } else {
                    (
                        ::http::StatusCode::SERVICE_UNAVAILABLE,
                        "<Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message></Error>",
                        None,
                    )
                }
            } else {
                (::http::StatusCode::BAD_REQUEST, "unexpected request", None)
            };

            let mut response = ::http::Response::builder().status(status);
            if let Some(e_tag) = e_tag {
                response = response.header(::http::header::ETAG, e_tag);
            }
            Ok(response
                .body(HttpResponseBody::from(body.to_string()))
                .unwrap())
        }
    }

    /// Retries must remain inside the original S3 `put_part` call. Re-entering
    /// `MultipartUpload::put_part` would allocate a new part number and leave a
    /// gap that makes `complete` fail with "Missing part".
    #[cfg(feature = "aws")]
    #[tokio::test(start_paused = true)]
    async fn test_multipart_http_retry_reuses_part_number() {
        use object_store::RetryConfig;
        use object_store::aws::AmazonS3Builder;

        let retry_state = Arc::new(MultipartRetryState {
            failures_remaining: AtomicUsize::new(3),
            part_uris: std::sync::Mutex::new(Vec::new()),
            complete_failures_remaining: AtomicUsize::new(0),
            complete_bodies: std::sync::Mutex::new(Vec::new()),
            complete_error_on_200: false,
        });
        let throttle_state = AimdThrottleState::new(AimdThrottleConfig::default()).unwrap();
        let connector = AimdHttpConnector::new(
            MultipartRetryConnector {
                state: Arc::clone(&retry_state),
            },
            Some(&throttle_state),
        );
        let store = AmazonS3Builder::new()
            .with_bucket_name("bucket")
            .with_region("us-east-1")
            .with_skip_signature(true)
            .with_retry(RetryConfig {
                max_retries: 0,
                ..Default::default()
            })
            .with_http_connector(connector)
            .build()
            .unwrap();

        let mut upload = store.put_multipart(&Path::from("object")).await.unwrap();
        upload
            .put_part(PutPayload::from_static(b"payload"))
            .await
            .unwrap();
        upload.complete().await.unwrap();

        let part_uris = retry_state.part_uris.lock().unwrap();
        assert_eq!(part_uris.len(), 4);
        assert!(part_uris.iter().all(|uri| uri == &part_uris[0]));
        assert!(part_uris[0].contains("partNumber=1"));
    }

    /// `MultipartUpload::complete` consumes the recorded part ids before
    /// sending, so it must be called exactly once. A failed complete is retried
    /// at the HTTP layer instead, by re-sending the same request with its
    /// serialized part list. S3 can report a failed complete as a throttling
    /// status or as HTTP 200 carrying an `InternalError` document.
    #[cfg(feature = "aws")]
    #[tokio::test(start_paused = true)]
    #[rstest]
    #[case::throttled(false)]
    #[case::internal_error_on_200(true)]
    async fn test_multipart_http_retry_resends_failed_complete(
        #[case] complete_error_on_200: bool,
    ) {
        use object_store::RetryConfig;
        use object_store::aws::AmazonS3Builder;

        let retry_state = Arc::new(MultipartRetryState {
            failures_remaining: AtomicUsize::new(0),
            part_uris: std::sync::Mutex::new(Vec::new()),
            complete_failures_remaining: AtomicUsize::new(1),
            complete_bodies: std::sync::Mutex::new(Vec::new()),
            complete_error_on_200,
        });
        let throttle_state = AimdThrottleState::new(AimdThrottleConfig::default()).unwrap();
        let connector = AimdHttpConnector::new(
            MultipartRetryConnector {
                state: Arc::clone(&retry_state),
            },
            Some(&throttle_state),
        );
        let store = AmazonS3Builder::new()
            .with_bucket_name("bucket")
            .with_region("us-east-1")
            .with_skip_signature(true)
            .with_retry(RetryConfig {
                max_retries: 0,
                ..Default::default()
            })
            .with_http_connector(connector)
            .build()
            .unwrap();

        let mut upload = store.put_multipart(&Path::from("object")).await.unwrap();
        upload
            .put_part(PutPayload::from_static(b"payload"))
            .await
            .unwrap();
        upload.complete().await.unwrap();

        let bodies = retry_state.complete_bodies.lock().unwrap();
        assert_eq!(
            bodies.len(),
            2,
            "the failed complete should be retried exactly once at the HTTP layer"
        );
        let first = std::str::from_utf8(&bodies[0]).unwrap();
        assert!(
            first.contains("<PartNumber>1</PartNumber>") && first.contains("part-etag"),
            "the complete request should carry the uploaded part list, got: {first}"
        );
        assert_eq!(
            bodies[1], bodies[0],
            "the retried complete should re-send the original part list"
        );
    }

    /// A successful complete echoes the object key in `<Key>`; a key holding a
    /// throttle word must not trigger a retry, which would replay an upload
    /// that already completed and fail with `NoSuchUpload`.
    #[cfg(feature = "aws")]
    #[tokio::test(start_paused = true)]
    #[rstest]
    #[case::throttle("throttle-demo/object")]
    #[case::slowdown("slowdown/object")]
    #[case::serverbusy("serverbusy/object")]
    #[case::requesttimeout("requesttimeout/object")]
    #[case::ordinary("data/object")]
    async fn test_gate_successful_completion_response(#[case] key: &str) {
        #[derive(Debug)]
        struct Completed {
            key: String,
            calls: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl HttpService for Completed {
            async fn call(&self, _: HttpRequest) -> Result<HttpResponse, HttpError> {
                let (status, body) = if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    (
                        ::http::StatusCode::OK,
                        format!(
                            "<CompleteMultipartUploadResult><Location>https://bucket/{0}</Location><Key>{0}</Key><ETag>\"etag\"</ETag></CompleteMultipartUploadResult>",
                            self.key
                        ),
                    )
                } else {
                    (
                        ::http::StatusCode::NOT_FOUND,
                        "<Error><Code>NoSuchUpload</Code></Error>".to_string(),
                    )
                };
                Ok(::http::Response::builder()
                    .status(status)
                    .body(HttpResponseBody::from(body))
                    .unwrap())
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let state = AimdThrottleState::new(AimdThrottleConfig::default()).unwrap();
        let service = AimdHttpService {
            inner: HttpClient::new(Completed {
                key: key.to_string(),
                calls: Arc::clone(&calls),
            }),
            write: Some(state.write),
            delete: None,
        };
        let request = ::http::Request::builder()
            .method(::http::Method::POST)
            .uri(format!("https://bucket/{key}?uploadId=id"))
            .body(object_store::client::HttpRequestBody::empty())
            .unwrap();
        let response = service.call(request).await.unwrap();
        assert_eq!(response.status(), ::http::StatusCode::OK);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// The wrapper must surface the target's real error rather than retry
    /// `complete` and mask it with "Missing part".
    #[tokio::test(start_paused = true)]
    async fn test_throttled_multipart_complete_calls_target_once() {
        use object_store::PutResult;
        use object_store::UploadPart;

        #[derive(Debug)]
        struct FailingComplete {
            complete_calls: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl MultipartUpload for FailingComplete {
            fn put_part(&mut self, _data: PutPayload) -> UploadPart {
                Box::pin(async { Ok(()) })
            }

            async fn complete(&mut self) -> OSResult<PutResult> {
                self.complete_calls.fetch_add(1, Ordering::SeqCst);
                Err(make_generic_error(
                    "SlowDown: Please reduce your request rate",
                ))
            }

            async fn abort(&mut self) -> OSResult<()> {
                Ok(())
            }
        }

        let state = AimdThrottleState::new(AimdThrottleConfig::default()).unwrap();
        let complete_calls = Arc::new(AtomicUsize::new(0));
        let mut upload = ThrottledMultipartUpload {
            target: Box::new(FailingComplete {
                complete_calls: Arc::clone(&complete_calls),
            }),
            write: Arc::clone(&state.write),
            requests_throttled_at_http: false,
        };

        let err = upload.complete().await.unwrap_err();
        assert_eq!(
            complete_calls.load(Ordering::SeqCst),
            1,
            "complete must call the target exactly once"
        );
        assert!(
            !err.to_string().contains("Missing part"),
            "complete retry surfaced the masked error: {err}"
        );
        assert!(
            is_throttle_error(&err),
            "the target's throttle error should surface unchanged: {err}"
        );
    }

    #[tokio::test]
    async fn test_throttled_multipart_reorders_parts() {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let config = AimdThrottleConfig::default();
        let throttled = AimdThrottledStore::new(store.clone(), config).unwrap();

        let path = Path::from("test/multipart_ordering.bin");
        let mut upload = throttled.put_multipart(&path).await.unwrap();

        // Create futures for two parts in order: A then B.
        let fut_a = upload.put_part(PutPayload::from_static(b"AAAA"));
        let fut_b = upload.put_part(PutPayload::from_static(b"BBBB"));

        // Await in REVERSE order. Part ordering should be determined by
        // creation order (put_part call order), not by await order.
        fut_b.await.unwrap();
        fut_a.await.unwrap();

        upload.complete().await.unwrap();

        let result = store.get(&path).await.unwrap();
        let bytes = result.bytes().await.unwrap();

        assert_eq!(
            bytes.as_ref(),
            b"AAAABBBB",
            "Parts were reordered! Got {:?} instead of AAAABBBB.",
            std::str::from_utf8(&bytes).unwrap_or("<non-utf8>"),
        );
    }

    #[tokio::test]
    async fn test_rename_moves_object() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default();
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        let from = Path::from("test/from.txt");
        let to = Path::from("test/to.txt");
        throttled
            .put(&from, PutPayload::from_static(b"payload"))
            .await
            .unwrap();

        throttled.rename(&from, &to).await.unwrap();

        // Destination has the data and source is gone.
        let bytes = throttled.get(&to).await.unwrap().bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"payload");
        assert!(matches!(
            throttled.head(&from).await,
            Err(object_store::Error::NotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_rename_if_not_exists_rejects_existing_target() {
        let store = Arc::new(InMemory::new());
        let config = AimdThrottleConfig::default();
        let throttled = AimdThrottledStore::new(store, config).unwrap();

        let from = Path::from("test/from.txt");
        let to = Path::from("test/to.txt");
        throttled
            .put(&from, PutPayload::from_static(b"src"))
            .await
            .unwrap();
        throttled
            .put(&to, PutPayload::from_static(b"existing"))
            .await
            .unwrap();

        // RenameTargetMode::Create must map to CopyMode::Create and reject the
        // pre-existing destination.
        let err = throttled
            .rename_if_not_exists(&from, &to)
            .await
            .unwrap_err();
        assert!(
            matches!(err, object_store::Error::AlreadyExists { .. }),
            "expected AlreadyExists, got {err:?}"
        );

        // The failed copy must not have deleted the source.
        let bytes = throttled.get(&from).await.unwrap().bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"src");
    }

    /// A mock store that returns a configurable number of throttle errors on
    /// `copy_opts` and `delete` independently before succeeding. Used to verify
    /// that `rename_opts` retries the copy and the delete separately.
    struct RenameRetryMockStore {
        inner: InMemory,
        copy_errors_remaining: std::sync::Mutex<usize>,
        delete_errors_remaining: std::sync::Mutex<usize>,
        copy_call_count: AtomicU64,
        delete_call_count: AtomicU64,
    }

    impl RenameRetryMockStore {
        fn new(copy_errors: usize, delete_errors: usize) -> Self {
            Self {
                inner: InMemory::new(),
                copy_errors_remaining: std::sync::Mutex::new(copy_errors),
                delete_errors_remaining: std::sync::Mutex::new(delete_errors),
                copy_call_count: AtomicU64::new(0),
                delete_call_count: AtomicU64::new(0),
            }
        }

        fn should_error(counter: &std::sync::Mutex<usize>) -> bool {
            let mut remaining = counter.lock().unwrap();
            if *remaining > 0 {
                *remaining -= 1;
                true
            } else {
                false
            }
        }

        fn throttle_error() -> object_store::Error {
            object_store::Error::Generic {
                store: "RenameRetryMock",
                source: THROTTLE_ERROR_RESPONSE.into(),
            }
        }
    }

    impl Display for RenameRetryMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "RenameRetryMockStore")
        }
    }

    impl Debug for RenameRetryMockStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RenameRetryMockStore").finish()
        }
    }

    #[async_trait]
    impl ObjectStore for RenameRetryMockStore {
        async fn put_opts(
            &self,
            location: &Path,
            bytes: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            self.inner.put_opts(location, bytes, opts).await
        }
        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }
        async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
            self.inner.get_opts(location, options).await
        }
        async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
            self.inner.get_ranges(location, ranges).await
        }
        // `delete` is provided by `ObjectStoreExt` and delegates to
        // `delete_stream`, so we inject throttle errors here to exercise the
        // separate delete-retry path in `rename_opts`.
        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            self.delete_call_count.fetch_add(1, Ordering::Relaxed);
            if Self::should_error(&self.delete_errors_remaining) {
                futures::stream::once(async { Err(Self::throttle_error()) }).boxed()
            } else {
                self.inner.delete_stream(locations)
            }
        }
        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list(prefix)
        }
        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }
        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }
        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.copy_call_count.fetch_add(1, Ordering::Relaxed);
            if Self::should_error(&self.copy_errors_remaining) {
                Err(Self::throttle_error())
            } else {
                self.inner.copy_opts(from, to, opts).await
            }
        }
    }

    #[tokio::test]
    async fn test_rename_if_not_exists_retries_copy_and_delete_independently() {
        // Copy fails twice then succeeds; delete fails once then succeeds.
        // Both are within max_retries=3, so the rename_if_not_exists should
        // ultimately succeed, with copy and delete retried independently.
        let mock = Arc::new(RenameRetryMockStore::new(2, 1));
        let from = Path::from("test/from.txt");
        let to = Path::from("test/to.txt");
        mock.put(&from, PutPayload::from_static(b"payload"))
            .await
            .unwrap();

        let config = AimdThrottleConfig::default();
        let throttled =
            AimdThrottledStore::new(mock.clone() as Arc<dyn ObjectStore>, config).unwrap();

        throttled.rename_if_not_exists(&from, &to).await.unwrap();

        // Copy: 2 failures + 1 success. Delete: 1 failure + 1 success.
        assert_eq!(mock.copy_call_count.load(Ordering::Relaxed), 3);
        assert_eq!(mock.delete_call_count.load(Ordering::Relaxed), 2);

        // Source cleaned up, destination present.
        let bytes = throttled.get(&to).await.unwrap().bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"payload");
        assert!(matches!(
            throttled.head(&from).await,
            Err(object_store::Error::NotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_rename_if_not_exists_succeeds_when_delete_throttled() {
        // Copy succeeds, but delete is throttled beyond max_retries.
        // The rename must still report success because the destination exists
        // and the caller's intent has been satisfied.
        let mock = Arc::new(RenameRetryMockStore::new(0, 10));
        let from = Path::from("test/from.txt");
        let to = Path::from("test/to.txt");
        mock.put(&from, PutPayload::from_static(b"payload"))
            .await
            .unwrap();

        let config = AimdThrottleConfig::default();
        let throttled =
            AimdThrottledStore::new(mock.clone() as Arc<dyn ObjectStore>, config).unwrap();

        throttled.rename_if_not_exists(&from, &to).await.unwrap();

        // Destination has the data.
        let bytes = throttled.get(&to).await.unwrap().bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"payload");
        // Source was not cleaned up because delete kept failing.
        let bytes = throttled.get(&from).await.unwrap().bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), b"payload");
    }
}
