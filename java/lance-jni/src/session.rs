// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use jni::JNIEnv;
use jni::objects::{JByteArray, JMap, JObject, JString, JValue};
use jni::sys::{jboolean, jbyteArray, jlong};
use lance::cache_diagnostics_pb as pb;
use lance::session::{
    CacheSpec, Session as LanceSession, SessionCacheDiagnostics as NativeSessionCacheDiagnostics,
};
use lance_core::cache::{
    BackendConfig, CacheActivity, CacheBackendDiagnostics, CacheBackendKind,
    CacheByTypeDiagnostics, CacheDiagnostics, CacheSnapshotMode, build_from_config, build_from_uri,
};
use lance_io::object_store::ObjectStoreRegistry;
use prost::Message;

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

fn checked_jlong(value: u64, field: &str) -> Result<jlong> {
    value.try_into().map_err(|_| {
        Error::runtime_error(format!(
            "cache diagnostics field {} value {} exceeds Java long range",
            field, value
        ))
    })
}

fn checked_u64(value: u64, field: &str) -> Result<u64> {
    checked_jlong(value, field).map(|_| value)
}

fn checked_optional(value: Option<u64>, field: &str) -> Result<Option<u64>> {
    value.map(|value| checked_u64(value, field)).transpose()
}

fn activity_to_proto(activity: &CacheActivity) -> Result<pb::CacheActivityDiagnostics> {
    Ok(pb::CacheActivityDiagnostics {
        hits: checked_u64(activity.hits, "activity.hits")?,
        misses: checked_u64(activity.misses, "activity.misses")?,
        lookup_errors: checked_u64(activity.lookup_errors, "activity.lookup_errors")?,
        type_mismatches: checked_u64(activity.type_mismatches, "activity.type_mismatches")?,
        loads_started: checked_u64(activity.loads_started, "activity.loads_started")?,
        loads_succeeded: checked_u64(activity.loads_succeeded, "activity.loads_succeeded")?,
        loads_failed: checked_u64(activity.loads_failed, "activity.loads_failed")?,
        loads_cancelled: checked_u64(activity.loads_cancelled, "activity.loads_cancelled")?,
        loads_in_flight: checked_u64(activity.loads_in_flight, "activity.loads_in_flight")?,
        load_success_duration_ns: checked_u64(
            activity.load_success_duration_ns,
            "activity.load_success_duration_ns",
        )?,
        load_error_duration_ns: checked_u64(
            activity.load_error_duration_ns,
            "activity.load_error_duration_ns",
        )?,
        load_cancelled_duration_ns: checked_u64(
            activity.load_cancelled_duration_ns,
            "activity.load_cancelled_duration_ns",
        )?,
        warm: Some(pb::CacheWarmActivityDiagnostics {
            attempts: checked_u64(activity.warm.attempts, "activity.warm.attempts")?,
            hits: checked_u64(activity.warm.hits, "activity.warm.hits")?,
            loads_started: checked_u64(activity.warm.loads_started, "activity.warm.loads_started")?,
            loads_succeeded: checked_u64(
                activity.warm.loads_succeeded,
                "activity.warm.loads_succeeded",
            )?,
            loads_failed: checked_u64(activity.warm.loads_failed, "activity.warm.loads_failed")?,
            loads_cancelled: checked_u64(
                activity.warm.loads_cancelled,
                "activity.warm.loads_cancelled",
            )?,
            load_bytes: checked_u64(activity.warm.load_bytes, "activity.warm.load_bytes")?,
            errors: checked_u64(activity.warm.errors, "activity.warm.errors")?,
        }),
    })
}

fn backend_to_proto(backend: &CacheBackendDiagnostics) -> Result<pb::CacheBackendDiagnostics> {
    let kind = match backend.kind {
        CacheBackendKind::Quick => pb::cache_backend_diagnostics::Kind::Quick,
        CacheBackendKind::Moka => pb::cache_backend_diagnostics::Kind::Moka,
        CacheBackendKind::Custom => pb::cache_backend_diagnostics::Kind::Custom,
    };
    Ok(pb::CacheBackendDiagnostics {
        kind: kind as i32,
        pool_id: checked_optional(backend.pool_id, "backend.pool_id")?,
        capacity_bytes: checked_optional(backend.capacity_bytes, "backend.capacity_bytes")?,
        enabled: backend.enabled,
        size_bytes: checked_optional(backend.size_bytes, "backend.size_bytes")?,
        num_entries: checked_optional(backend.num_entries, "backend.num_entries")?,
        write_attempts: checked_optional(backend.write_attempts, "backend.write_attempts")?,
        write_bytes: checked_optional(backend.write_bytes, "backend.write_bytes")?,
        size_removals: checked_optional(backend.size_removals, "backend.size_removals")?,
        size_removed_bytes: checked_optional(
            backend.size_removed_bytes,
            "backend.size_removed_bytes",
        )?,
        disabled_write_rejections: checked_optional(
            backend.disabled_write_rejections,
            "backend.disabled_write_rejections",
        )?,
        disabled_bypasses: checked_optional(
            backend.disabled_bypasses,
            "backend.disabled_bypasses",
        )?,
        lost_placeholder_rejections: checked_optional(
            backend.lost_placeholder_rejections,
            "backend.lost_placeholder_rejections",
        )?,
        weight_saturations: checked_optional(
            backend.weight_saturations,
            "backend.weight_saturations",
        )?,
        write_rejections_complete: backend.write_rejections_complete,
        resident_evictions: checked_optional(
            backend.resident_evictions,
            "backend.resident_evictions",
        )?,
        admissions: checked_optional(backend.admissions, "backend.admissions")?,
        coalesced_loads: checked_optional(backend.coalesced_loads, "backend.coalesced_loads")?,
    })
}

fn diagnostics_to_proto(diagnostics: &CacheDiagnostics) -> Result<pb::CacheDiagnosticsSnapshot> {
    Ok(pb::CacheDiagnosticsSnapshot {
        activity: Some(activity_to_proto(&diagnostics.activity)?),
        backend: Some(backend_to_proto(&diagnostics.backend)?),
        utilization: diagnostics.utilization,
        by_type: diagnostics
            .by_type
            .as_ref()
            .map(by_type_to_proto)
            .transpose()?,
    })
}

fn by_type_to_proto(diagnostics: &CacheByTypeDiagnostics) -> Result<pb::CacheByTypeDiagnostics> {
    let activity = diagnostics
        .activity
        .iter()
        .map(|item| {
            Ok(pb::CacheTypeActivityDiagnostics {
                type_name: item.type_name.clone(),
                activity: Some(activity_to_proto(&item.activity)?),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let occupancy = diagnostics
        .occupancy
        .as_ref()
        .map(|occupancy| -> Result<pb::CacheOccupancyByTypeDiagnostics> {
            let types = occupancy
                .types
                .iter()
                .map(|item| {
                    Ok(pb::CacheTypeOccupancyDiagnostics {
                        type_name: item.type_name.clone(),
                        size_bytes: checked_u64(item.size_bytes, "by_type.occupancy.size_bytes")?,
                        num_entries: checked_u64(
                            item.num_entries,
                            "by_type.occupancy.num_entries",
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(pb::CacheOccupancyByTypeDiagnostics {
                types,
                untagged_size_bytes: checked_u64(
                    occupancy.untagged_size_bytes,
                    "by_type.occupancy.untagged_size_bytes",
                )?,
                untagged_num_entries: checked_u64(
                    occupancy.untagged_num_entries,
                    "by_type.occupancy.untagged_num_entries",
                )?,
            })
        })
        .transpose()?;
    Ok(pb::CacheByTypeDiagnostics {
        activity,
        occupancy,
        type_label_overflow_events: checked_u64(
            diagnostics.type_label_overflow_events,
            "by_type.type_label_overflow_events",
        )?,
    })
}

fn session_diagnostics_to_proto(
    diagnostics: &NativeSessionCacheDiagnostics,
) -> Result<pb::SessionCacheDiagnostics> {
    Ok(pb::SessionCacheDiagnostics {
        index: Some(diagnostics_to_proto(&diagnostics.index)?),
        metadata: Some(diagnostics_to_proto(&diagnostics.metadata)?),
    })
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
    cache_stats_to_java(env, block_on(session_arc.metadata_cache_stats()))
}

fn cache_stats_to_java<'local>(
    env: &mut JNIEnv<'local>,
    stats: lance_core::cache::CacheStats,
) -> Result<JObject<'local>> {
    let hits = checked_jlong(stats.hits, "hits")?;
    let misses = checked_jlong(stats.misses, "misses")?;
    let num_entries = checked_jlong(stats.num_entries as u64, "num_entries")?;
    let size_bytes = checked_jlong(stats.size_bytes as u64, "size_bytes")?;
    Ok(env.new_object(
        "org/lance/CacheStats",
        "(JJJJ)V",
        &[
            JValue::Long(hits),
            JValue::Long(misses),
            JValue::Long(num_entries),
            JValue::Long(size_bytes),
        ],
    )?)
}

/// Returns statistics for the session's index cache as an org.lance.CacheStats object.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Session_indexCacheStatsNative<'local>(
    mut env: JNIEnv<'local>,
    obj: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, index_cache_stats_native(&mut env, obj))
}

fn index_cache_stats_native<'local>(
    env: &mut JNIEnv<'local>,
    obj: JObject,
) -> Result<JObject<'local>> {
    let handle = get_session_handle(env, &obj)?;
    if handle == 0 {
        return Err(Error::input_error("Session is closed".to_string()));
    }

    // Safety: We trust that the handle is valid and was created by createNative
    let session_arc = unsafe { &*(handle as *const Arc<LanceSession>) };
    cache_stats_to_java(env, block_on(session_arc.index_cache_stats()))
}

/// Returns protobuf-encoded complete diagnostics for both session caches.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Session_cacheDiagnosticsNative(
    mut env: JNIEnv,
    obj: JObject,
    refresh: jboolean,
    by_type: jboolean,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        cache_diagnostics_native(&mut env, obj, refresh != 0, by_type != 0)
            .map(JByteArray::into_raw),
        JByteArray::default().into_raw()
    )
}

fn cache_diagnostics_native<'local>(
    env: &mut JNIEnv<'local>,
    obj: JObject,
    refresh: bool,
    by_type: bool,
) -> Result<JByteArray<'local>> {
    let handle = get_session_handle(env, &obj)?;
    if handle == 0 {
        return Err(Error::input_error("Session is closed".to_string()));
    }

    // Safety: We trust that the handle is valid and was created by createNative
    let session_arc = unsafe { &*(handle as *const Arc<LanceSession>) };
    let diagnostics = if by_type {
        let mode = if refresh {
            CacheSnapshotMode::Refreshed
        } else {
            CacheSnapshotMode::Approximate
        };
        block_on(session_arc.cache_diagnostics_by_type(mode))
    } else if refresh {
        block_on(session_arc.cache_diagnostics_with_mode(CacheSnapshotMode::Refreshed))
    } else {
        session_arc.cache_diagnostics()
    };
    let bytes = session_diagnostics_to_proto(&diagnostics)?.encode_to_vec();
    Ok(env.byte_array_from_slice(&bytes)?)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_proto_preserves_optional_fields() {
        let session = LanceSession::with_cache_backends(
            CacheSpec::Size(0),
            CacheSpec::Size(2048),
            Arc::new(ObjectStoreRegistry::default()),
        );

        let diagnostics = session_diagnostics_to_proto(&session.cache_diagnostics()).unwrap();
        let index = diagnostics.index.unwrap();
        assert_eq!(
            index
                .activity
                .as_ref()
                .unwrap()
                .warm
                .as_ref()
                .unwrap()
                .attempts,
            0
        );
        let index_backend = index.backend.unwrap();
        assert_eq!(index_backend.capacity_bytes, Some(0));
        assert_eq!(index_backend.enabled, Some(false));
        assert_eq!(index.utilization, None);

        let metadata = diagnostics.metadata.unwrap();
        assert_eq!(metadata.backend.unwrap().capacity_bytes, Some(2048));
        assert_eq!(metadata.utilization, Some(0.0));

        let by_type = block_on(session.cache_diagnostics_by_type(CacheSnapshotMode::Approximate));
        let by_type = session_diagnostics_to_proto(&by_type).unwrap();
        let index_by_type = by_type.index.unwrap().by_type.unwrap();
        assert!(index_by_type.activity.is_empty());
        assert!(index_by_type.occupancy.unwrap().types.is_empty());
    }

    #[test]
    fn diagnostics_reject_values_outside_java_long_range() {
        assert!(checked_u64(i64::MAX as u64, "test").is_ok());
        assert!(checked_u64(i64::MAX as u64 + 1, "test").is_err());
    }
}
