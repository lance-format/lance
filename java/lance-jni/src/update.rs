// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::IntoJava;
use crate::utils::to_rust_map;
use crate::{JNIEnvExt, block_on};
use jni::JNIEnv;
use jni::objects::{JMap, JObject, JValueGen};
use lance::dataset::UpdateBuilder;
use std::sync::Arc;
use std::time::Duration;

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeUpdate<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject, // Dataset object
    jparam: JObject,   // UpdateParams object
) -> JObject<'a> {
    ok_or_throw!(env, inner_update(&mut env, jdataset, jparam))
}

fn inner_update<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jparam: JObject,
) -> Result<JObject<'local>> {
    let updates = extract_updates(env, &jparam)?;
    let where_clause = extract_where(env, &jparam)?;
    let conflict_retries = extract_conflict_retries(env, &jparam)?;
    let retry_timeout_ms = extract_retry_timeout_ms(env, &jparam)?;

    // Clone the inner Dataset out of the `get_rust_field` guard and drop the
    // guard before running the long-lived async update. Otherwise the guard
    // would block any other JNI call on the same dataset for the entire
    // duration of `execute()`.
    let inner_dataset = unsafe {
        let dataset = env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET)?;
        dataset.inner.clone()
    };

    let mut builder = UpdateBuilder::new(Arc::new(inner_dataset))
        .conflict_retries(conflict_retries)
        .retry_timeout(Duration::from_millis(retry_timeout_ms));

    if let Some(predicate) = where_clause {
        builder = builder.update_where(&predicate)?;
    }

    for (column, expr) in &updates {
        builder = builder.set(column, expr)?;
    }

    let job = builder.build()?;
    let update_result = block_on(job.execute())?;

    // Avoid panicking if Lance core retains a clone of the Arc; fall back to a
    // deep clone so the JNI boundary stays panic-free.
    let new_ds = Arc::try_unwrap(update_result.new_dataset).unwrap_or_else(|arc| (*arc).clone());

    UpdateResultJava {
        dataset: BlockingDataset { inner: new_ds },
        rows_updated: update_result.rows_updated,
    }
    .into_java(env)
}

fn extract_updates<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<std::collections::HashMap<String, String>> {
    let updates_obj: JObject = env
        .call_method(jparam, "updates", "()Ljava/util/Map;", &[])?
        .l()?;
    let jmap = JMap::from_env(env, &updates_obj)?;
    to_rust_map(env, &jmap)
}

fn extract_where<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<Option<String>> {
    let where_opt = env
        .call_method(jparam, "whereClause", "()Ljava/util/Optional;", &[])?
        .l()?;
    env.get_string_opt(&where_opt)
}

fn extract_conflict_retries<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<u32> {
    let retries = env
        .call_method(jparam, "conflictRetries", "()I", &[])?
        .i()? as u32;
    Ok(retries)
}

fn extract_retry_timeout_ms<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<u64> {
    let timeout_ms = env.call_method(jparam, "retryTimeoutMs", "()J", &[])?.j()? as u64;
    Ok(timeout_ms)
}

const UPDATE_RESULT_CLASS: &str = "org/lance/update/UpdateResult";
const UPDATE_RESULT_CONSTRUCTOR_SIG: &str = "(Lorg/lance/Dataset;J)V";

struct UpdateResultJava {
    dataset: BlockingDataset,
    rows_updated: u64,
}

impl IntoJava for UpdateResultJava {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let jdataset = self.dataset.into_java(env)?;
        Ok(env.new_object(
            UPDATE_RESULT_CLASS,
            UPDATE_RESULT_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&jdataset),
                JValueGen::Long(self.rows_updated as i64),
            ],
        )?)
    }
}
