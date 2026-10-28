// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use jni::{
    JNIEnv,
    objects::{JByteArray, JObject, JString},
    sys::{jbyteArray, jstring},
};
use lance::{SnapshotIdentity, SnapshotRetention};

use crate::{
    Result, block_on,
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
};

pub(crate) fn dataset(env: &mut JNIEnv, object: JObject) -> Result<Arc<lance::Dataset>> {
    Ok(Arc::new(
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(object, NATIVE_DATASET) }?
            .inner
            .clone(),
    ))
}

pub(crate) fn java_bytes(env: &JNIEnv, bytes: &[u8]) -> Result<jbyteArray> {
    Ok(env.byte_array_from_slice(bytes)?.into_raw())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_SnapshotIdentity_fromDatasetNative(
    mut env: JNIEnv,
    _class: JObject,
    source: JObject,
    table_identity: JString,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let dataset = dataset(&mut env, source)?;
            let table_identity: String = env.get_string(&table_identity)?.into();
            let identity = block_on(SnapshotIdentity::from_dataset(&dataset, table_identity))?;
            java_bytes(&env, &identity.encode())
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_SnapshotIdentity_validateNative(
    env: JNIEnv,
    _class: JObject,
    encoded: JByteArray,
) -> jbyteArray {
    let mut env = env;
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            java_bytes(
                &env,
                &SnapshotIdentity::decode(&env.convert_byte_array(encoded)?)?.encode(),
            )
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_SnapshotRetention_acquireNative(
    mut env: JNIEnv,
    _class: JObject,
    source: JObject,
    snapshot: JByteArray,
    retention_id: JString,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let dataset = dataset(&mut env, source)?;
            let snapshot = SnapshotIdentity::decode(&env.convert_byte_array(snapshot)?)?;
            let retention_id: String = env.get_string(&retention_id)?.into();
            let retention = block_on(SnapshotRetention::acquire(&dataset, snapshot, retention_id))?;
            java_bytes(&env, &retention.encode())
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_SnapshotRetention_validateNative(
    mut env: JNIEnv,
    _class: JObject,
    encoded: JByteArray,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            java_bytes(
                &env,
                &SnapshotRetention::decode(&env.convert_byte_array(encoded)?)?.encode(),
            )
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_SnapshotRetention_snapshotNative(
    mut env: JNIEnv,
    _class: JObject,
    encoded: JByteArray,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            java_bytes(
                &env,
                &SnapshotRetention::decode(&env.convert_byte_array(encoded)?)?
                    .snapshot()
                    .encode(),
            )
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_SnapshotRetention_tagNative(
    mut env: JNIEnv,
    _class: JObject,
    encoded: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        (|| -> Result<_> {
            let retention = SnapshotRetention::decode(&env.convert_byte_array(encoded)?)?;
            Ok(env.new_string(retention.tag_name())?.into_raw())
        })(),
        std::ptr::null_mut()
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_SnapshotRetention_releaseNative(
    mut env: JNIEnv,
    _class: JObject,
    source: JObject,
    encoded: JByteArray,
) {
    ok_or_throw_without_return!(
        env,
        (|| -> Result<_> {
            let dataset = dataset(&mut env, source)?;
            let retention = SnapshotRetention::decode(&env.convert_byte_array(encoded)?)?;
            block_on(retention.release(&dataset))?;
            Ok(())
        })()
    );
}
