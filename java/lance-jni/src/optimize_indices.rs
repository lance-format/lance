// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The plan / execute / merge / commit stages of `lance::index::optimize` for
//! engines that run them on workers.
//!
//! A task, a merge and a result cross the boundary as the JSON serde writes for
//! the Rust struct. The Java object keeps it verbatim, beside the fields an
//! engine schedules by, and hands it back for the native side to read.

use jni::{
    JNIEnv,
    objects::{JClass, JObject, JString, JValueGen},
};
use lance::index::optimize::{
    IndexSegmentResult, OptimizeIndexMerge, OptimizeIndexTask, OptimizeIndicesPlan,
    commit_optimize_indices, plan_optimize_indices,
};
use lance_index::optimize::OptimizeOptions;

use crate::{
    block_on,
    blocking_dataset::{BlockingDataset, NATIVE_DATASET, optimize_options_from_java},
    error::Result,
    ffi::JNIEnvExt,
    index::{int_list, u32_to_jint, u64_to_jlong},
    traits::{FromJString, IntoJava, export_vec},
};

const PLAN_CLASS: &str = "org/lance/index/optimize/OptimizeIndicesPlan";
const PLAN_CONSTRUCTOR_SIG: &str = "(Ljava/lang/String;JLjava/util/List;Ljava/util/List;)V";
const TASK_CLASS: &str = "org/lance/index/optimize/OptimizeIndexTask";
const TASK_CONSTRUCTOR_SIG: &str = "(Ljava/lang/String;Ljava/lang/String;ILjava/util/List;)V";
const MERGE_CLASS: &str = "org/lance/index/optimize/OptimizeIndexMerge";
const MERGE_CONSTRUCTOR_SIG: &str = "(Ljava/lang/String;I)V";
const RESULT_CLASS: &str = "org/lance/index/optimize/IndexSegmentResult";
const RESULT_CONSTRUCTOR_SIG: &str = "(Ljava/lang/String;Ljava/lang/String;I)V";

/// The results a Java `List<String>` holds as JSON.
fn results_from_java(env: &mut JNIEnv, java_results: &JObject) -> Result<Vec<IndexSegmentResult>> {
    env.get_strings(java_results)?
        .iter()
        .map(|json| serde_json::from_str(json).map_err(Into::into))
        .collect()
}

/// What a task or a merge produced; `None` is Java `null`.
fn result_into_java<'a>(
    env: &mut JNIEnv<'a>,
    result: Option<IndexSegmentResult>,
) -> Result<JObject<'a>> {
    match &result {
        Some(result) => result.into_java(env),
        None => Ok(JObject::null()),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_optimize_OptimizeIndices_nativePlan<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    java_dataset: JObject, // Dataset
    java_options: JObject, // OptimizeOptions
) -> JObject<'local> {
    ok_or_throw!(env, inner_plan(&mut env, java_dataset, java_options))
}

fn inner_plan<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    java_options: JObject,
) -> Result<JObject<'local>> {
    let options = optimize_options_from_java(env, &java_options)?;
    let plan = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(plan_optimize_indices(&dataset.inner, &options))?
    };
    plan.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_optimize_OptimizeIndices_nativeCommit(
    mut env: JNIEnv,
    _class: JClass,
    java_dataset: JObject, // Dataset
    java_results: JObject, // List<String>
) {
    ok_or_throw_without_return!(env, inner_commit(&mut env, java_dataset, java_results));
}

fn inner_commit(env: &mut JNIEnv, java_dataset: JObject, java_results: JObject) -> Result<()> {
    let results = results_from_java(env, &java_results)?;
    let mut dataset =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    // The commit reads only `transaction_properties` from the options, which Java
    // does not expose.
    block_on(commit_optimize_indices(
        &mut dataset.inner,
        results,
        &OptimizeOptions::default(),
    ))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_optimize_OptimizeIndexTask_nativeExecute<'local>(
    mut env: JNIEnv<'local>,
    _task: JObject,
    java_dataset: JObject, // Dataset
    json: JString,         // the task
) -> JObject<'local> {
    ok_or_throw!(env, inner_execute_task(&mut env, java_dataset, json))
}

fn inner_execute_task<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    json: JString,
) -> Result<JObject<'local>> {
    let task: OptimizeIndexTask = serde_json::from_str(&json.extract(env)?)?;
    let result = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(task.execute(&dataset.inner))?
    };
    result_into_java(env, result)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_optimize_OptimizeIndexMerge_nativeExecute<'local>(
    mut env: JNIEnv<'local>,
    _merge: JObject,
    java_dataset: JObject, // Dataset
    json: JString,         // the merge
    java_results: JObject, // List<String>: the task results
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_execute_merge(&mut env, java_dataset, json, java_results)
    )
}

fn inner_execute_merge<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    json: JString,
    java_results: JObject,
) -> Result<JObject<'local>> {
    let merge: OptimizeIndexMerge = serde_json::from_str(&json.extract(env)?)?;
    let task_results = results_from_java(env, &java_results)?;
    let result = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(merge.execute(&dataset.inner, &task_results))?
    };
    result_into_java(env, result)
}

impl IntoJava for &OptimizeIndicesPlan {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let plan_id = env.new_string(self.plan_id.to_string())?;
        let read_version = u64_to_jlong("read_version", self.read_version)?;
        let tasks = export_vec(env, &self.tasks)?;
        let merges = export_vec(env, &self.merges)?;
        Ok(env.new_object(
            PLAN_CLASS,
            PLAN_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&plan_id),
                JValueGen::Long(read_version),
                JValueGen::Object(&tasks),
                JValueGen::Object(&merges),
            ],
        )?)
    }
}

impl IntoJava for &OptimizeIndexTask {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let json = env.new_string(serde_json::to_string(self)?)?;
        let index_name = env.new_string(&self.reference.name)?;
        let group = u32_to_jint("group", self.group)?;
        let fragments = self
            .fragments
            .iter()
            .map(|id| u32_to_jint("fragments", *id))
            .collect::<Result<Vec<_>>>()?;
        let fragments = int_list(env, fragments)?;
        Ok(env.new_object(
            TASK_CLASS,
            TASK_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&json),
                JValueGen::Object(&index_name),
                JValueGen::Int(group),
                JValueGen::Object(&fragments),
            ],
        )?)
    }
}

impl IntoJava for &OptimizeIndexMerge {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let json = env.new_string(serde_json::to_string(self)?)?;
        let group = u32_to_jint("group", self.group)?;
        Ok(env.new_object(
            MERGE_CLASS,
            MERGE_CONSTRUCTOR_SIG,
            &[JValueGen::Object(&json), JValueGen::Int(group)],
        )?)
    }
}

impl IntoJava for &IndexSegmentResult {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let json = env.new_string(serde_json::to_string(self)?)?;
        let index_name = env.new_string(&self.segment.name)?;
        let group = u32_to_jint("group", self.group)?;
        Ok(env.new_object(
            RESULT_CLASS,
            RESULT_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&json),
                JValueGen::Object(&index_name),
                JValueGen::Int(group),
            ],
        )?)
    }
}
