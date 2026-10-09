// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use jni::JNIEnv;
use jni::objects::{JMap, JObject, JString, JValue};
use jni::sys::jlong;
use lance::session::{CacheSpec, Session as LanceSession};
use lance_core::cache::{BackendConfig, build_from_config, build_from_uri};
use lance_io::object_store::ObjectStoreRegistry;

use crate::block_on;
use crate::error::{Error, Result};
use crate::utils::to_rust_map;

/// Creates a new Session and returns a handle to it.
///
/// The handle is a raw pointer to a Box<Arc<LanceSession>>, which allows
/// the session to be shared between multiple datasets.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Session_createNative(
    mut env: JNIEnv,
    _obj: JObject,
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
    index_cache_backend_uri: JString,
    index_cache_backend_kind: JString,
    index_cache_backend_options: JObject,
    metadata_cache_backend_uri: JString,
    metadata_cache_backend_kind: JString,
    metadata_cache_backend_options: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_session(
            &mut env,
            index_cache_size_bytes,
            metadata_cache_size_bytes,
            index_cache_backend_uri,
            index_cache_backend_kind,
            index_cache_backend_options,
            metadata_cache_backend_uri,
            metadata_cache_backend_kind,
            metadata_cache_backend_options,
        ),
        0
    )
}

#[allow(clippy::too_many_arguments)]
fn create_session(
    env: &mut JNIEnv,
    index_cache_size_bytes: jlong,
    metadata_cache_size_bytes: jlong,
    index_cache_backend_uri: JString,
    index_cache_backend_kind: JString,
    index_cache_backend_options: JObject,
    metadata_cache_backend_uri: JString,
    metadata_cache_backend_kind: JString,
    metadata_cache_backend_options: JObject,
) -> Result<jlong> {
    let index_cache = resolve_cache_spec(
        env,
        "indexCacheBackend",
        "indexCacheSizeBytes",
        index_cache_size_bytes,
        index_cache_backend_uri,
        index_cache_backend_kind,
        index_cache_backend_options,
    )?;
    let metadata_cache = resolve_cache_spec(
        env,
        "metadataCacheBackend",
        "metadataCacheSizeBytes",
        metadata_cache_size_bytes,
        metadata_cache_backend_uri,
        metadata_cache_backend_kind,
        metadata_cache_backend_options,
    )?;

    let session = LanceSession::with_cache_backends(
        index_cache,
        metadata_cache,
        Arc::new(ObjectStoreRegistry::default()),
    );

    // Wrap in Arc and Box, then convert to raw pointer
    let boxed: Box<Arc<LanceSession>> = Box::new(Arc::new(session));
    let handle = Box::into_raw(boxed) as jlong;
    Ok(handle)
}

#[allow(clippy::too_many_arguments)]
fn resolve_cache_spec(
    env: &mut JNIEnv,
    backend_field: &str,
    size_field: &str,
    size_bytes: jlong,
    backend_uri: JString,
    backend_kind: JString,
    backend_options: JObject,
) -> Result<CacheSpec> {
    let has_uri = !backend_uri.is_null();
    let has_kind = !backend_kind.is_null();
    if has_uri && has_kind {
        return Err(Error::input_error(format!(
            "{} must use either a URI or a structured config, not both",
            backend_field
        )));
    }
    if size_bytes >= 0 && (has_uri || has_kind) {
        return Err(Error::input_error(format!(
            "{} and {} are mutually exclusive; set one or the other",
            size_field, backend_field
        )));
    }

    if has_uri {
        let uri: String = env.get_string(&backend_uri)?.into();
        return build_from_uri(&uri)
            .map(CacheSpec::Backend)
            .map_err(Error::from);
    }

    if has_kind {
        let kind: String = env.get_string(&backend_kind)?.into();
        let mut config = BackendConfig::new(&kind)?;
        if !backend_options.is_null() {
            let options = JMap::from_env(env, &backend_options)?;
            config.options = to_rust_map(env, &options)?;
        }
        return build_from_config(&config)
            .map(CacheSpec::Backend)
            .map_err(Error::from);
    }

    if size_bytes >= 0 {
        let size = usize::try_from(size_bytes).map_err(|_| {
            Error::input_error(format!(
                "{} value {} does not fit in usize",
                size_field, size_bytes
            ))
        })?;
        Ok(CacheSpec::Size(size))
    } else {
        Ok(CacheSpec::Default)
    }
}

/// Returns the current size of the session in bytes.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Session_sizeBytesNative(
    mut env: JNIEnv,
    obj: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, size_bytes_native(&mut env, obj), 0)
}

fn size_bytes_native(env: &mut JNIEnv, obj: JObject) -> Result<jlong> {
    let handle = get_session_handle(env, &obj)?;
    if handle == 0 {
        return Err(Error::input_error("Session is closed".to_string()));
    }

    // Safety: We trust that the handle is valid and was created by createNative
    let session_arc = unsafe { &*(handle as *const Arc<LanceSession>) };
    Ok(session_arc.size_bytes() as jlong)
}

/// Returns statistics for the session's metadata cache as an org.lance.CacheStats object.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Session_metadataCacheStatsNative<'local>(
    mut env: JNIEnv<'local>,
    obj: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, metadata_cache_stats_native(&mut env, obj))
}

fn metadata_cache_stats_native<'local>(
    env: &mut JNIEnv<'local>,
    obj: JObject,
) -> Result<JObject<'local>> {
    let handle = get_session_handle(env, &obj)?;
    if handle == 0 {
        return Err(Error::input_error("Session is closed".to_string()));
    }

    // Safety: We trust that the handle is valid and was created by createNative
    let session_arc = unsafe { &*(handle as *const Arc<LanceSession>) };
    let stats = block_on(session_arc.metadata_cache_stats());

    let stats_obj = env.new_object(
        "org/lance/CacheStats",
        "(JJJJ)V",
        &[
            JValue::Long(stats.hits as jlong),
            JValue::Long(stats.misses as jlong),
            JValue::Long(stats.num_entries as jlong),
            JValue::Long(stats.size_bytes as jlong),
        ],
    )?;
    Ok(stats_obj)
}

/// Releases the native session handle.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Session_releaseNative(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    if handle != 0 {
        // Safety: We trust that the handle is valid and was created by createNative
        let _ = unsafe { Box::from_raw(handle as *mut Arc<LanceSession>) };
        // The Box is dropped here, which decrements the Arc reference count
    }
}

/// Helper function to get the session handle from a Session object
fn get_session_handle(env: &mut JNIEnv, obj: &JObject) -> Result<jlong> {
    let handle = env.get_field(obj, "nativeSessionHandle", "J")?;
    Ok(handle.j()?)
}

/// Creates an Arc<LanceSession> from a raw handle.
/// This is used when passing a session to dataset operations.
///
/// # Safety
/// The handle must be a valid pointer created by `create_session`.
pub fn session_from_handle(handle: jlong) -> Option<Arc<LanceSession>> {
    if handle == 0 {
        return None;
    }

    // Safety: We trust that the handle is valid and was created by createNative
    let session_arc = unsafe { &*(handle as *const Arc<LanceSession>) };
    Some(session_arc.clone())
}

/// Creates a raw handle from an Arc<LanceSession>.
/// This is used when returning a session handle from a dataset.
///
/// Note: This creates a new Box, so the caller is responsible for
/// managing its lifetime or converting it back to a Java Session object.
pub fn handle_from_session(session: Arc<LanceSession>) -> jlong {
    let boxed: Box<Arc<LanceSession>> = Box::new(session);
    Box::into_raw(boxed) as jlong
}

/// Compares two session handles to see if they point to the same underlying session.
/// This is needed because each call to handle_from_session creates a new Box,
/// resulting in different pointer addresses even for the same session.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Session_isSameAsNative(
    _env: JNIEnv,
    _obj: JObject,
    handle1: jlong,
    handle2: jlong,
) -> jni::sys::jboolean {
    if handle1 == 0 || handle2 == 0 {
        return 0; // false
    }

    // Safety: We trust that the handles are valid and were created by createNative
    let session1 = unsafe { &*(handle1 as *const Arc<LanceSession>) };
    let session2 = unsafe { &*(handle2 as *const Arc<LanceSession>) };

    if Arc::ptr_eq(session1, session2) {
        1 // true
    } else {
        0 // false
    }
}
