// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use jni::objects::{GlobalRef, JObject, JString, JValue};
use jni::{JNIEnv, JavaVM};
use lance_index::progress::IndexBuildProgress;

use crate::error::{Error, Result};

/// Bridges Rust index progress events to a Java callback.
pub(crate) struct JavaIndexBuildProgress {
    callback: GlobalRef,
    jvm: Arc<JavaVM>,
}

impl std::fmt::Debug for JavaIndexBuildProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JavaIndexBuildProgress")
    }
}

impl JavaIndexBuildProgress {
    pub(crate) fn new(env: &mut JNIEnv, callback: &JObject) -> Result<Self> {
        if callback.is_null() {
            return Err(Error::input_error(
                "IndexBuildProgress callback cannot be null".to_string(),
            ));
        }
        Ok(Self {
            callback: env.new_global_ref(callback)?,
            jvm: Arc::new(env.get_java_vm()?),
        })
    }

    fn call_stage_start(
        &self,
        stage: &str,
        total: Option<u64>,
        unit: &str,
    ) -> lance_core::Result<()> {
        let total = total
            .map(|value| {
                i64::try_from(value).map_err(|_| {
                    lance_core::Error::invalid_input(format!(
                        "IndexBuildProgress total exceeds Java long for stage '{stage}': {value}"
                    ))
                })
            })
            .transpose()?;
        let mut env = self.jvm.attach_current_thread().map_err(|error| {
            lance_core::Error::internal(format!(
                "IndexBuildProgress.stageStart failed to attach JVM thread for stage '{stage}': {error}"
            ))
        })?;

        env.with_local_frame(16, |env| {
            let stage_obj = env
                .new_string(stage)
                .map_err(|error| callback_error(env, "stageStart", stage, error))?;
            let unit_obj = env
                .new_string(unit)
                .map_err(|error| callback_error(env, "stageStart", stage, error))?;
            let total_obj = match total {
                Some(value) => env
                    .new_object("java/lang/Long", "(J)V", &[JValue::Long(value)])
                    .map_err(|error| callback_error(env, "stageStart", stage, error))?,
                None => JObject::null(),
            };
            let total_optional = env
                .call_static_method(
                    "java/util/Optional",
                    "ofNullable",
                    "(Ljava/lang/Object;)Ljava/util/Optional;",
                    &[JValue::Object(&total_obj)],
                )
                .and_then(|value| value.l())
                .map_err(|error| callback_error(env, "stageStart", stage, error))?;

            env.call_method(
                &self.callback,
                "stageStart",
                "(Ljava/lang/String;Ljava/util/Optional;Ljava/lang/String;)V",
                &[
                    JValue::Object(&stage_obj),
                    JValue::Object(&total_optional),
                    JValue::Object(&unit_obj),
                ],
            )
            .map_err(|error| callback_error(env, "stageStart", stage, error))?;
            Ok::<(), Error>(())
        })
        .map_err(|error| lance_core::Error::internal(error.to_string()))
    }

    fn call_stage_progress(&self, stage: &str, completed: u64) -> lance_core::Result<()> {
        let completed = i64::try_from(completed).map_err(|_| {
            lance_core::Error::invalid_input(format!(
                "IndexBuildProgress completed value exceeds Java long for stage '{stage}': {completed}"
            ))
        })?;
        let mut env = self.jvm.attach_current_thread().map_err(|error| {
            lance_core::Error::internal(format!(
                "IndexBuildProgress.stageProgress failed to attach JVM thread for stage '{stage}': {error}"
            ))
        })?;

        env.with_local_frame(16, |env| {
            let stage_obj = env
                .new_string(stage)
                .map_err(|error| callback_error(env, "stageProgress", stage, error))?;

            env.call_method(
                &self.callback,
                "stageProgress",
                "(Ljava/lang/String;J)V",
                &[JValue::Object(&stage_obj), JValue::Long(completed)],
            )
            .map_err(|error| callback_error(env, "stageProgress", stage, error))?;
            Ok::<(), Error>(())
        })
        .map_err(|error| lance_core::Error::internal(error.to_string()))
    }

    fn call_stage_complete(&self, stage: &str) -> lance_core::Result<()> {
        let mut env = self.jvm.attach_current_thread().map_err(|error| {
            lance_core::Error::internal(format!(
                "IndexBuildProgress.stageComplete failed to attach JVM thread for stage '{stage}': {error}"
            ))
        })?;

        env.with_local_frame(16, |env| {
            let stage_obj = env
                .new_string(stage)
                .map_err(|error| callback_error(env, "stageComplete", stage, error))?;

            env.call_method(
                &self.callback,
                "stageComplete",
                "(Ljava/lang/String;)V",
                &[JValue::Object(&stage_obj)],
            )
            .map_err(|error| callback_error(env, "stageComplete", stage, error))?;
            Ok::<(), Error>(())
        })
        .map_err(|error| lance_core::Error::internal(error.to_string()))
    }
}

fn callback_error(env: &mut JNIEnv, method: &str, stage: &str, error: jni::errors::Error) -> Error {
    let java_exception = take_pending_java_exception(env);
    let detail = java_exception.unwrap_or_else(|| error.to_string());
    Error::runtime_error(format!(
        "IndexBuildProgress.{method} callback failed for stage '{stage}': {detail}"
    ))
}

fn take_pending_java_exception(env: &mut JNIEnv) -> Option<String> {
    if !env.exception_check().unwrap_or(false) {
        return None;
    }

    let throwable = env.exception_occurred().ok();
    let _ = env.exception_clear();

    let description = throwable.and_then(|throwable| {
        if throwable.is_null() {
            return None;
        }
        let description = env
            .call_method(&throwable, "toString", "()Ljava/lang/String;", &[])
            .and_then(|value| value.l())
            .ok()?;
        if description.is_null() {
            return None;
        }
        let description = JString::from(description);
        env.get_string(&description).ok().map(|value| value.into())
    });

    if env.exception_check().unwrap_or(false) {
        let _ = env.exception_clear();
    }
    description
}

#[async_trait]
impl IndexBuildProgress for JavaIndexBuildProgress {
    async fn stage_start(
        &self,
        stage: &str,
        total: Option<u64>,
        unit: &str,
    ) -> lance_core::Result<()> {
        self.call_stage_start(stage, total, unit)
    }

    async fn stage_progress(&self, stage: &str, completed: u64) -> lance_core::Result<()> {
        self.call_stage_progress(stage, completed)
    }

    async fn stage_complete(&self, stage: &str) -> lance_core::Result<()> {
        if let Err(error) = self.call_stage_complete(stage) {
            log::warn!(
                "Ignoring IndexBuildProgress.stageComplete callback failure for stage '{}': {}",
                stage,
                error
            );
        }
        Ok(())
    }
}
