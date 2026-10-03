// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[macro_export]
macro_rules! ok_or_throw {
    ($env:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return JObject::null();
            }
        }
    };
}

macro_rules! ok_or_throw_without_return {
    ($env:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! ok_or_throw_with_return {
    ($env:expr, $result:expr, $ret:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return $ret;
            }
        }
    };
}

mod async_scanner;
mod blocking_blob;
mod blocking_dataset;
mod blocking_scanner;
mod delta;
mod dispatcher;
pub mod error;
pub mod ffi;
mod file_reader;
mod file_writer;
mod fragment;
mod index;
mod index_progress;
mod mem_wal;
mod merge_insert;
mod namespace;
mod optimize;
mod otel;
mod schema;
mod session;
mod sql;
mod storage_options;
mod task_tracker;
pub mod traits;
mod transaction;
mod update;
pub mod utils;
mod vector_trainer;

pub use error::Error;
pub use error::Result;
pub use ffi::JNIEnvExt;

use env_logger::{Builder, Env};
use std::env;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Arc;

use std::sync::LazyLock;

pub static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

/// Drive a future on the shared JNI runtime, including nested calls.
///
/// Progress callbacks (and similar JNI re-entry) may invoke Dataset methods while
/// already inside `RT.block_on`. Calling `Runtime::block_on` again panics with
/// "Cannot start a runtime from within a runtime". When a Tokio handle is already
/// available, use `block_in_place` + `Handle::block_on` instead.
///
/// JNI entry points should use this helper instead of calling `RT.block_on`
/// directly so they remain safe when invoked from a callback.
pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
        Err(_) => RT.block_on(future),
    }
}

/// Process-wide [`lance_io::object_store::ObjectStoreRegistry`] used for JNI
/// default-open paths.
///
/// When the Java caller does not supply an explicit session, the JNI open
/// path constructs a per-call session that shares this registry. Sharing the
/// registry across calls allows the registry's per-key single-flight to
/// coalesce concurrent cold builds for the same URI, and lets long-lived
/// `ObjectStore` strong references be reused across opens — both of which
/// turn what would otherwise be a thundering herd into a cheap weak-Arc
/// upgrade.
///
/// # Why the registry is shared but the `Session` is not
///
/// The JNI default-open path is intentionally asymmetric: the
/// `ObjectStoreRegistry` is process-global, but each open builds a fresh
/// `Session` (which owns the metadata/index caches). This shape is chosen
/// because the two layers cache fundamentally different things:
///
/// - **Registry → `Arc<ObjectStore>`**: an HTTP/S3 client, credential chain,
///   and connection pool. Building one is the *expensive* operation
///   (credential probe, IMDS round-trip, TLS handshake) — so this is what
///   the 144-concurrent-open regression was made of, and what the global
///   registry exists to coalesce.
///
///   The cache key is derived from the provider-specific store prefix
///   (typically scheme + authority — e.g. `s3://bucket` — but providers
///   such as Hugging Face fold `repo_id` in instead) plus the relevant
///   fields of `ObjectStoreParams` (block size, dynamic
///   `storage_options_accessor`'s `provider_id()`, etc.). It does **not**
///   incorporate auth headers, STS tokens, namespace identity, or any
///   bearer credentials.
///
///   A `provider_id()` cannot be assumed to carry principal identity, so
///   provider-backed default opens do not use this registry at all: they get
///   a per-call one. `StorageOptionsAccessor::accessor_id()` returns just the
///   `provider_id()` when a provider is present, and a namespace provider
///   derives that from the namespace id plus table id while a REST namespace
///   id holds only the endpoint and delimiter, so two principals opening the
///   same table would otherwise share one entry and the second would receive
///   the first's credential-bearing store. Static `storage_options` are safe
///   to key on: with no provider, `accessor_id()` hashes the option values
///   themselves. See `blocking_dataset::select_default_open_registry`.
///
///   Bare-URI opens (empty `storage_options`, no provider, no namespace
///   commit-handler) collapse onto a single cache entry per URI: the first
///   caller's resolved default-credential chain becomes the credentials
///   used by every subsequent caller for the lifetime of that
///   `Arc<ObjectStore>`. Callers who need cross-tenant isolation under
///   bare URIs MUST opt out via
///   `LANCE_JNI_DISABLE_DEFAULT_REGISTRY_SHARING=1`; the resolved bool is
///   consulted on every default-open path.
///
/// - **Session → metadata/index caches**: query-shaped, sized by
///   `index_cache_size_bytes` and `metadata_cache_size_bytes` from each
///   open's `ReadParams`. Sharing a Session across opens would force every
///   caller to pick the same cache size, would make eviction policy a
///   cross-tenant policy decision, and would let one tenant's hot dataset
///   evict another's. None of those are problems we want to take on inside
///   the JNI bridge — Java callers that want metadata-cache reuse can build
///   their own [`lance::session::Session`] and pass it in explicitly via
///   `BlockingDataset::open` with `session: Some(...)`.
///
/// # Lifetime
///
/// This static lives for the lifetime of the process. JVM unload (e.g. via
/// `System.exit`) on most platforms exits the host process, so the
/// registry is dropped along with it; the JNI library is not designed to
/// be unloaded and re-loaded within a single process. Embedders that
/// genuinely need per-JVM isolation — multiple JVMs in one address space
/// or hot-reload of the Lance native library — should construct their own
/// `Session` per JVM and pass it explicitly via
/// `BlockingDataset::open(..., session: Some(...))`, bypassing this
/// static entirely.
pub(crate) static GLOBAL_OBJECT_STORE_REGISTRY: LazyLock<
    Arc<lance_io::object_store::ObjectStoreRegistry>,
> = LazyLock::new(|| Arc::new(lance_io::object_store::ObjectStoreRegistry::default()));

fn set_timestamp_precision(builder: &mut env_logger::Builder) {
    if let Ok(timestamp_precision) = env::var("LANCE_LOG_TS_PRECISION") {
        match timestamp_precision.as_str() {
            "ns" => {
                builder.format_timestamp_nanos();
            }
            "us" => {
                builder.format_timestamp_micros();
            }
            "ms" => {
                builder.format_timestamp_millis();
            }
            "s" => {
                builder.format_timestamp_secs();
            }
            _ => {
                // Can't log here because logging is not initialized yet
                println!(
                    "Invalid timestamp precision (valid values: ns, us, ms, s): {}, using default",
                    timestamp_precision
                );
            }
        };
    }
}

fn set_log_file_target(builder: &mut env_logger::Builder) {
    if let Ok(log_file_path) = env::var("LANCE_LOG_FILE") {
        let path = Path::new(&log_file_path);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            println!(
                "Failed to create parent directories for log file '{}': {}, using stderr",
                log_file_path, e
            );
            return;
        }

        // Try to open/create the log file
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => {
                builder.target(env_logger::Target::Pipe(Box::new(file)));
            }
            Err(e) => {
                println!(
                    "Failed to open log file '{}': {}, using stderr",
                    log_file_path, e
                );
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_JniLoader_initLanceLogger() {
    let env = Env::new()
        .filter_or("LANCE_LOG", "warn")
        .write_style("LANCE_LOG_STYLE");
    let mut log_builder = Builder::from_env(env);
    set_timestamp_precision(&mut log_builder);
    set_log_file_target(&mut log_builder);
    let logger = Arc::new(log_builder.build());
    let max_level = logger.filter();
    log::set_boxed_logger(Box::new(logger.clone())).unwrap();
    log::set_max_level(max_level);
    // todo: add tracing
}

/// JNI_OnLoad - Called when the JVM loads the native library
/// Initializes the global dispatcher for async operations
#[unsafe(no_mangle)]
pub extern "system" fn JNI_OnLoad(
    vm: jni::JavaVM,
    _reserved: *mut std::ffi::c_void,
) -> jni::sys::jint {
    // Resolve AsyncScanner class on the current thread which has the correct
    // application classloader. A newly spawned native thread only gets the
    // system classloader after attach_current_thread_permanently(), which
    // cannot find application classes in environments like Spark, web
    // containers, or shaded JARs.
    let mut env = vm.get_env().expect("Failed to get JNIEnv in JNI_OnLoad");
    let async_scanner_local = env
        .find_class("org/lance/ipc/AsyncScanner")
        .expect("AsyncScanner class not found");
    let async_scanner_class = env
        .new_global_ref(async_scanner_local)
        .expect("Failed to create GlobalRef for AsyncScanner class");

    let jvm_arc = Arc::new(vm);

    // Initialize global dispatcher with persistent thread, passing the
    // pre-resolved class reference so the dispatcher thread does not need
    // to look up the class with the wrong classloader.
    let dispatcher = dispatcher::Dispatcher::initialize(jvm_arc, async_scanner_class);

    // Set the global DISPATCHER (will panic if called more than once)
    dispatcher::DISPATCHER
        .set(dispatcher)
        .expect("Dispatcher already initialized");

    jni::sys::JNI_VERSION_1_8
}
