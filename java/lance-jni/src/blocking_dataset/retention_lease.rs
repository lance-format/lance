// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::time::Duration;

use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString, JValue};
use lance::dataset::{LeaseKeeper, LeaseOptions, LeaseStatus, RetentionLease};
use lance_table::format::pb;

use super::{BlockingDataset, NATIVE_DATASET};
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::traits::{FromJObjectWithEnv, FromJString, IntoJava, export_vec};
use crate::{RT, block_on};

const RETENTION_LEASE_CLASS: &str = "org/lance/RetentionLease";
const LEASE_KEEPER_CLASS: &str = "org/lance/LeaseKeeper";
const NATIVE_LEASE_KEEPER: &str = "nativeLeaseKeeperHandle";

impl IntoJava for &RetentionLease {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let id = self.id().into_java(env)?;
        let version = i64::try_from(self.version()).map_err(|_| {
            Error::runtime_error(format!(
                "retention lease {} version {} does not fit in a Java long",
                self.id(),
                self.version()
            ))
        })?;
        let created_at = instant_to_java(env, self.created_at())?;
        let expires_at = instant_to_java(env, self.expires_at())?;
        let label = match self.label() {
            Some(label) => JObject::from(env.new_string(label)?),
            None => JObject::null(),
        };
        Ok(env.new_object(
            RETENTION_LEASE_CLASS,
            "(Ljava/util/UUID;JLjava/time/Instant;Ljava/time/Instant;Ljava/lang/String;)V",
            &[
                JValue::Object(&id),
                JValue::Long(version),
                JValue::Object(&created_at),
                JValue::Object(&expires_at),
                JValue::Object(&label),
            ],
        )?)
    }
}

/// Rebuild a lease from its Java fields through the protobuf form, so Rust
/// validates leases that crossed a serialization boundary.
impl FromJObjectWithEnv<RetentionLease> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RetentionLease> {
        let id: uuid::Uuid = env
            .get_field(self, "id", "Ljava/util/UUID;")?
            .l()?
            .extract_object(env)?;
        let version = env.get_field(self, "version", "J")?.j()?;
        let version = u64::try_from(version).map_err(|_| {
            Error::input_error(format!(
                "retention lease {id} version must be non-negative, got {version}"
            ))
        })?;
        let created_at = env
            .get_field(self, "createdAt", "Ljava/time/Instant;")?
            .l()?;
        let created_at = instant_to_pb(env, &created_at)?;
        let expires_at = env
            .get_field(self, "expiresAt", "Ljava/time/Instant;")?
            .l()?;
        let expires_at = instant_to_pb(env, &expires_at)?;
        let label = env.get_field(self, "label", "Ljava/lang/String;")?.l()?;
        let label = if label.is_null() {
            None
        } else {
            Some(JString::from(label).extract(env)?)
        };
        let message = pb::RetentionLease {
            id: Some(pb::Uuid::from(&id)),
            version: Some(version),
            created_at,
            expires_at,
            label,
        };
        Ok(RetentionLease::try_from(&message)?)
    }
}

pub(crate) fn instant_to_java<'a>(
    env: &mut JNIEnv<'a>,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Result<JObject<'a>> {
    Ok(env
        .call_static_method(
            "java/time/Instant",
            "ofEpochSecond",
            "(JJ)Ljava/time/Instant;",
            &[
                JValue::Long(timestamp.timestamp()),
                JValue::Long(timestamp.timestamp_subsec_nanos() as i64),
            ],
        )?
        .l()?)
}

fn instant_to_pb(
    env: &mut JNIEnv<'_>,
    instant: &JObject,
) -> Result<Option<prost_types::Timestamp>> {
    if instant.is_null() {
        return Ok(None);
    }
    let seconds = env
        .call_method(instant, "getEpochSecond", "()J", &[])?
        .j()?;
    let nanos = env.call_method(instant, "getNano", "()I", &[])?.i()?;
    Ok(Some(prost_types::Timestamp { seconds, nanos }))
}

fn ttl_from_java(env: &mut JNIEnv<'_>, ttl: &JObject) -> Result<Duration> {
    let seconds = env.call_method(ttl, "getSeconds", "()J", &[])?.j()?;
    let nanos = env.call_method(ttl, "getNano", "()I", &[])?.i()?;
    match (u64::try_from(seconds), u32::try_from(nanos)) {
        (Ok(seconds), Ok(nanos)) => Ok(Duration::new(seconds, nanos)),
        _ => Err(Error::input_error(format!(
            "retention lease ttl must be positive, got {seconds}s {nanos}ns"
        ))),
    }
}

/// Reads an `org.lance.LeaseOptions`, or `None` if it is null.
pub(crate) fn lease_options_from_java(
    env: &mut JNIEnv<'_>,
    options: &JObject,
) -> Result<Option<LeaseOptions>> {
    if options.is_null() {
        return Ok(None);
    }
    let mut lease_options = LeaseOptions::default();
    if let Some(ttl) =
        env.get_optional_from_method(options, "getTtl", |env, ttl| ttl_from_java(env, &ttl))?
    {
        lease_options.ttl = ttl;
    }
    lease_options.label = env.get_optional_string_from_method(options, "getLabel")?;
    Ok(Some(lease_options))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeAcquireRetentionLease<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    ttl: JObject,
    label: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_acquire_retention_lease(&mut env, java_dataset, ttl, label)
    )
}

fn inner_acquire_retention_lease<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    ttl: JObject,
    label: JObject,
) -> Result<JObject<'local>> {
    let mut options = LeaseOptions::default();
    if !ttl.is_null() {
        options.ttl = ttl_from_java(env, &ttl)?;
    }
    if !label.is_null() {
        options.label = Some(JString::from(label).extract(env)?);
    }
    let (lease, _) = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.acquire_retention_lease(options))?
    };
    lease.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutLease<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    lease: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_checkout_lease(&mut env, java_dataset, lease))
}

fn inner_checkout_lease<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    lease: JObject,
) -> Result<JObject<'local>> {
    let lease: RetentionLease = lease.extract_object(env)?;
    let leased = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.checkout_lease(&lease))?
    };
    BlockingDataset::new(leased).into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeRenewRetentionLease<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    lease: JObject,
    ttl: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_renew_retention_lease(&mut env, java_dataset, lease, ttl)
    )
}

fn inner_renew_retention_lease<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    lease: JObject,
    ttl: JObject,
) -> Result<JObject<'local>> {
    let lease: RetentionLease = lease.extract_object(env)?;
    let ttl = ttl_from_java(env, &ttl)?;
    let renewed = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.renew_retention_lease(&lease, ttl))?
    };
    renewed.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeReleaseRetentionLease(
    mut env: JNIEnv,
    java_dataset: JObject,
    lease: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_release_retention_lease(&mut env, java_dataset, lease)
    )
}

fn inner_release_retention_lease(
    env: &mut JNIEnv,
    java_dataset: JObject,
    lease: JObject,
) -> Result<()> {
    let lease: RetentionLease = lease.extract_object(env)?;
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.release_retention_lease(&lease))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetRetentionLeases<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_retention_leases(&mut env, java_dataset))
}

fn inner_get_retention_leases<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let leases = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.inner.retention_leases().to_vec()
    };
    export_vec(env, &leases)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_LeaseKeeper_nativeStart<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    java_dataset: JObject,
    lease: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_start_lease_keeper(&mut env, java_dataset, lease))
}

fn inner_start_lease_keeper<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    lease: JObject,
) -> Result<JObject<'local>> {
    let lease: RetentionLease = lease.extract_object(env)?;
    let dataset = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.inner.clone()
    };
    // `LeaseKeeper::start` spawns its renewal task on the current runtime.
    let keeper = {
        let _runtime = RT.enter();
        LeaseKeeper::start(&dataset, lease)
    };
    let java_keeper = env.new_object(LEASE_KEEPER_CLASS, "()V", &[])?;
    unsafe { env.set_rust_field(&java_keeper, NATIVE_LEASE_KEEPER, keeper) }?;
    Ok(java_keeper)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_LeaseKeeper_nativeGetLease<'local>(
    mut env: JNIEnv<'local>,
    java_keeper: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_keeper_lease(&mut env, java_keeper))
}

fn inner_get_keeper_lease<'local>(
    env: &mut JNIEnv<'local>,
    java_keeper: JObject,
) -> Result<JObject<'local>> {
    let lease = {
        let keeper =
            unsafe { env.get_rust_field::<_, _, LeaseKeeper>(java_keeper, NATIVE_LEASE_KEEPER) }?;
        keeper.lease()
    };
    lease.into_java(env)
}

/// Returns the reason the lease was lost, or null while it is held.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_LeaseKeeper_nativeGetLostReason<'local>(
    mut env: JNIEnv<'local>,
    java_keeper: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_get_lost_reason(&mut env, java_keeper))
}

fn inner_get_lost_reason<'local>(
    env: &mut JNIEnv<'local>,
    java_keeper: JObject,
) -> Result<JObject<'local>> {
    let status = {
        let keeper =
            unsafe { env.get_rust_field::<_, _, LeaseKeeper>(java_keeper, NATIVE_LEASE_KEEPER) }?;
        keeper.status()
    };
    match status {
        LeaseStatus::Held => Ok(JObject::null()),
        LeaseStatus::Lost { reason } => Ok(env.new_string(reason)?.into()),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_LeaseKeeper_nativeRelease(
    mut env: JNIEnv,
    java_keeper: JObject,
) {
    ok_or_throw_without_return!(env, inner_release_lease_keeper(&mut env, java_keeper))
}

fn inner_release_lease_keeper(env: &mut JNIEnv, java_keeper: JObject) -> Result<()> {
    let keeper: LeaseKeeper = unsafe { env.take_rust_field(java_keeper, NATIVE_LEASE_KEEPER) }?;
    block_on(keeper.release())?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_LeaseKeeper_nativeClose(
    mut env: JNIEnv,
    java_keeper: JObject,
) {
    ok_or_throw_without_return!(env, inner_close_lease_keeper(&mut env, java_keeper))
}

fn inner_close_lease_keeper(env: &mut JNIEnv, java_keeper: JObject) -> Result<()> {
    let _: LeaseKeeper = unsafe { env.take_rust_field(java_keeper, NATIVE_LEASE_KEEPER) }?;
    Ok(())
}
