// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use jni::{
    JNIEnv,
    objects::{JByteArray, JObject, JValueGen},
    sys::jlong,
};
use lance::dataset::{
    index::DatasetIndexRemapperOptions,
    optimize::{
        CompactionMetrics, CompactionMode, CompactionOptions, CompactionPlan, CompactionTask,
        IndexRemapperOptions, RewriteResult, TaskData, commit_compaction, plan_compaction,
    },
};

use crate::{
    block_on,
    blocking_dataset::{BlockingDataset, NATIVE_DATASET},
    traits::{
        FromJObjectWithEnv, IntoJava, export_vec, import_vec_from_method, import_vec_to_rust,
    },
    utils::{
        build_compaction_options, to_java_boolean_obj, to_java_float_obj, to_java_list,
        to_java_long_obj, to_java_optional,
    },
};

use crate::error::Result;

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_compaction_Compaction_nativePlanCompaction<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    java_dataset: JObject,                    // Dataset
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_plan_compaction(
            &mut env,
            java_dataset,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap,
            compaction_mode,
            binary_copy_read_batch_bytes,
            max_source_fragments,
            max_source_rows,
            max_source_bytes,
            excluded_fragment_ids
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_plan_compaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> Result<JObject<'local>> {
    let config = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset.inner.manifest.config.clone()
    };
    let compaction_options = build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
        &compaction_mode,
        &binary_copy_read_batch_bytes,
        &max_source_fragments,
        &max_source_rows,
        &max_source_bytes,
        &excluded_fragment_ids,
        &config,
    )?;

    let plan = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(plan_compaction(&dataset.inner, &compaction_options))?
    };
    plan.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_compaction_Compaction_commitCompactionNative<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    java_dataset: JObject,                    // Dataset
    rewrite_results: JObject,                 // List<RewriteResult>
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_commit_compaction(
            &mut env,
            java_dataset,
            rewrite_results,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap,
            compaction_mode,
            binary_copy_read_batch_bytes,
            max_source_fragments,
            max_source_rows,
            max_source_bytes,
            excluded_fragment_ids,
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_commit_compaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    rewrite_results: JObject,                 // List<RewriteResult>
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> Result<JObject<'local>> {
    let config = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset.inner.manifest.config.clone()
    };
    let compaction_options = build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
        &compaction_mode,
        &binary_copy_read_batch_bytes,
        &max_source_fragments,
        &max_source_rows,
        &max_source_bytes,
        &excluded_fragment_ids,
        &config,
    )?;
    let completed_tasks = import_vec_to_rust(env, &rewrite_results, |env, rewrite_result| {
        rewrite_result.extract_object(env)
    })?;
    let remap_options: Arc<dyn IndexRemapperOptions> = Arc::new(DatasetIndexRemapperOptions {});
    let committed_metrics = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(commit_compaction(
            &mut dataset.inner,
            completed_tasks,
            remap_options,
            &compaction_options,
        ))?
    };
    committed_metrics.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_compaction_CompactionTask_nativeExecute<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,                            // CompactionTask itself
    java_dataset: JObject,                    // Dataset
    task_data: JObject,                       // TaskData
    read_version: jlong,                      // readVersion
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> JObject<'local> {
    ok_or_throw_with_return!(
        env,
        inner_execute_task(
            &mut env,
            java_dataset,
            task_data,
            read_version,
            target_rows_per_fragment,
            max_rows_per_group,
            max_bytes_per_file,
            materialize_deletions,
            materialize_deletions_threshold,
            num_threads,
            batch_size,
            defer_index_remap,
            compaction_mode,
            binary_copy_read_batch_bytes,
            max_source_fragments,
            max_source_rows,
            max_source_bytes,
            excluded_fragment_ids
        ),
        JObject::null()
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_execute_task<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,                    // Dataset
    task_data: JObject,                       // TaskData
    read_version: jlong,                      // readVersion
    target_rows_per_fragment: JObject,        // Optional<Long>
    max_rows_per_group: JObject,              // Optional<Long>
    max_bytes_per_file: JObject,              // Optional<Long>
    materialize_deletions: JObject,           // Optional<Boolean>
    materialize_deletions_threshold: JObject, // Optional<Float>
    num_threads: JObject,                     // Optional<Long>
    batch_size: JObject,                      // Optional<Long>
    defer_index_remap: JObject,               // Optional<Boolean>
    compaction_mode: JObject,                 // Optional<String>
    binary_copy_read_batch_bytes: JObject,    // Optional<Long>
    max_source_fragments: JObject,            // Optional<Long>
    max_source_rows: JObject,                 // Optional<Long>
    max_source_bytes: JObject,                // Optional<Long>
    excluded_fragment_ids: JObject,           // List<Long>
) -> Result<JObject<'local>> {
    let task_data: TaskData = task_data.extract_object(env)?;
    let config = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset.inner.manifest.config.clone()
    };
    let compaction_options = build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &materialize_deletions,
        &materialize_deletions_threshold,
        &num_threads,
        &batch_size,
        &defer_index_remap,
        &compaction_mode,
        &binary_copy_read_batch_bytes,
        &max_source_fragments,
        &max_source_rows,
        &max_source_bytes,
        &excluded_fragment_ids,
        &config,
    )?;
    let compaction_task = CompactionTask {
        task: task_data,
        read_version: read_version as u64,
        options: compaction_options,
    };
    let rewrite_result = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(compaction_task.execute(&dataset.inner))?
    };
    rewrite_result.into_java(env)
}

const TASK_DATA_CLASS: &str = "org/lance/compaction/TaskData";
const TASK_DATA_CONSTRUCTOR_SIG: &str = "(Ljava/util/List;)V";
const COMPACTION_METRICS_CLASS: &str = "org/lance/compaction/CompactionMetrics";
const COMPACTION_METRICS_CONSTRUCTOR_SIG: &str = "(JJJJ)V";
const COMPACTION_PLAN_CLASS: &str = "org/lance/compaction/CompactionPlan";
const COMPACTION_PLAN_CONSTRUCTOR_SIG: &str =
    "(Ljava/util/List;JLorg/lance/compaction/CompactionOptions;)V";
const REWRITE_RESULT_CLASS: &str = "org/lance/compaction/RewriteResult";
const REWRITE_RESULT_CONSTRUCTOR_SIG: &str =
    "(Lorg/lance/compaction/CompactionMetrics;Ljava/util/List;Ljava/util/List;J[B)V";
const COMPACTION_OPTIONS_CLASS: &str = "org/lance/compaction/CompactionOptions";
const COMPACTION_MODE_CLASS: &str = "org/lance/compaction/CompactionMode";
const COMPACTION_OPTIONS_CONSTRUCTOR_SIG: &str = "(Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/Optional;Ljava/util/List;)V";

impl IntoJava for &TaskData {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let fragments = export_vec(env, &self.fragments)?;
        Ok(env.new_object(
            TASK_DATA_CLASS,
            TASK_DATA_CONSTRUCTOR_SIG,
            &[JValueGen::Object(&fragments)],
        )?)
    }
}

impl IntoJava for &CompactionMetrics {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        Ok(env.new_object(
            COMPACTION_METRICS_CLASS,
            COMPACTION_METRICS_CONSTRUCTOR_SIG,
            &[
                JValueGen::Long(self.fragments_removed as i64),
                JValueGen::Long(self.fragments_added as i64),
                JValueGen::Long(self.files_removed as i64),
                JValueGen::Long(self.files_added as i64),
            ],
        )?)
    }
}

impl IntoJava for &CompactionOptions {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let target_rows_per_fragment =
            to_java_long_obj(env, Some(self.target_rows_per_fragment as i64))?;
        let target_rows_per_fragment_opt = to_java_optional(env, target_rows_per_fragment)?;
        let max_rows_per_group = to_java_long_obj(env, Some(self.max_rows_per_group as i64))?;
        let max_rows_per_group_opt = to_java_optional(env, max_rows_per_group)?;
        let max_bytes_per_file = to_java_long_obj(env, self.max_bytes_per_file.map(|v| v as i64))?;
        let max_bytes_per_file_opt = to_java_optional(env, max_bytes_per_file)?;
        let materialize_deletions = to_java_boolean_obj(env, Some(self.materialize_deletions))?;
        let materialize_deletions_opt = to_java_optional(env, materialize_deletions)?;
        let materialize_deletions_threshold =
            to_java_float_obj(env, Some(self.materialize_deletions_threshold))?;
        let materialize_deletions_threshold_opt =
            to_java_optional(env, materialize_deletions_threshold)?;
        let num_threads = to_java_long_obj(env, self.num_threads.map(|v| v as i64))?;
        let num_threads_opt = to_java_optional(env, num_threads)?;
        let batch_size = to_java_long_obj(env, self.batch_size.map(|v| v as i64))?;
        let batch_size_opt = to_java_optional(env, batch_size)?;
        let defer_index_remap = to_java_boolean_obj(env, Some(self.defer_index_remap))?;
        let defer_index_remap_opt = to_java_optional(env, defer_index_remap)?;
        let compaction_mode_obj = match self.compaction_mode {
            Some(mode) => {
                let name = match mode {
                    CompactionMode::Reencode => "REENCODE",
                    CompactionMode::TryBinaryCopy => "TRY_BINARY_COPY",
                    CompactionMode::ForceBinaryCopy => "FORCE_BINARY_COPY",
                };
                env.get_static_field(
                    COMPACTION_MODE_CLASS,
                    name,
                    format!("L{};", COMPACTION_MODE_CLASS),
                )?
                .l()?
            }
            None => JObject::null(),
        };
        let compaction_mode_opt = to_java_optional(env, compaction_mode_obj)?;
        let binary_copy_read_batch_bytes =
            to_java_long_obj(env, self.binary_copy_read_batch_bytes.map(|v| v as i64))?;
        let binary_copy_read_batch_bytes_opt = to_java_optional(env, binary_copy_read_batch_bytes)?;
        let max_source_fragments =
            to_java_long_obj(env, self.max_source_fragments.map(|v| v as i64))?;
        let max_source_fragments_opt = to_java_optional(env, max_source_fragments)?;
        let max_source_rows = to_java_long_obj(env, self.max_source_rows.map(|v| v as i64))?;
        let max_source_rows_opt = to_java_optional(env, max_source_rows)?;
        let max_source_bytes = to_java_long_obj(env, self.max_source_bytes.map(|v| v as i64))?;
        let max_source_bytes_opt = to_java_optional(env, max_source_bytes)?;
        let excluded_fragment_ids = self
            .excluded_fragment_ids
            .iter()
            .map(|fragment_id| to_java_long_obj(env, Some(*fragment_id as i64)))
            .collect::<Result<Vec<_>>>()?;
        let excluded_fragment_ids = to_java_list(env, &excluded_fragment_ids)?;

        Ok(env.new_object(
            COMPACTION_OPTIONS_CLASS,
            COMPACTION_OPTIONS_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&target_rows_per_fragment_opt),
                JValueGen::Object(&max_rows_per_group_opt),
                JValueGen::Object(&max_bytes_per_file_opt),
                JValueGen::Object(&materialize_deletions_opt),
                JValueGen::Object(&materialize_deletions_threshold_opt),
                JValueGen::Object(&num_threads_opt),
                JValueGen::Object(&batch_size_opt),
                JValueGen::Object(&defer_index_remap_opt),
                JValueGen::Object(&compaction_mode_opt),
                JValueGen::Object(&binary_copy_read_batch_bytes_opt),
                JValueGen::Object(&max_source_fragments_opt),
                JValueGen::Object(&max_source_rows_opt),
                JValueGen::Object(&max_source_bytes_opt),
                JValueGen::Object(&excluded_fragment_ids),
            ],
        )?)
    }
}

impl IntoJava for &CompactionPlan {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let tasks = export_vec(env, &self.tasks)?;
        let compaction_options = self.options.into_java(env)?;
        Ok(env.new_object(
            COMPACTION_PLAN_CLASS,
            COMPACTION_PLAN_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&tasks),
                JValueGen::Long(self.read_version as i64),
                JValueGen::Object(&compaction_options),
            ],
        )?)
    }
}

impl IntoJava for &RewriteResult {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let metrics = self.metrics.into_java(env)?;
        let new_fragments = export_vec(env, &self.new_fragments)?;
        let original_fragments = export_vec(env, &self.original_fragments)?;
        let row_addrs: JObject<'_> = if let Some(row_addrs) = &self.row_addrs {
            env.byte_array_from_slice(row_addrs)?.into()
        } else {
            JObject::null()
        };
        Ok(env.new_object(
            REWRITE_RESULT_CLASS,
            REWRITE_RESULT_CONSTRUCTOR_SIG,
            &[
                JValueGen::Object(&metrics),
                JValueGen::Object(&new_fragments),
                JValueGen::Object(&original_fragments),
                JValueGen::Long(self.read_version as i64),
                JValueGen::Object(&row_addrs),
            ],
        )?)
    }
}

impl FromJObjectWithEnv<CompactionMetrics> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<CompactionMetrics> {
        let fragments_removed = env
            .call_method(self, "getFragmentsRemoved", "()J", &[])?
            .j()? as usize;
        let fragments_added = env
            .call_method(self, "getFragmentsAdded", "()J", &[])?
            .j()? as usize;
        let files_removed = env.call_method(self, "getFilesRemoved", "()J", &[])?.j()? as usize;
        let files_added = env.call_method(self, "getFilesAdded", "()J", &[])?.j()? as usize;
        Ok(CompactionMetrics {
            fragments_removed,
            fragments_added,
            files_removed,
            files_added,
        })
    }
}

impl FromJObjectWithEnv<TaskData> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<TaskData> {
        let task_data = import_vec_from_method(env, self, "getFragments", |env, fragment| {
            fragment.extract_object(env)
        })?;
        Ok(TaskData {
            fragments: task_data,
        })
    }
}

impl FromJObjectWithEnv<RewriteResult> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RewriteResult> {
        let metrics_obj = env
            .call_method(
                self,
                "getMetrics",
                "()Lorg/lance/compaction/CompactionMetrics;",
                &[],
            )?
            .l()?;
        let metrics = metrics_obj.extract_object(env)?;
        let new_fragments =
            import_vec_from_method(env, self, "getNewFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let read_version = env.call_method(self, "getReadVersion", "()J", &[])?.j()? as u64;
        let original_fragments =
            import_vec_from_method(env, self, "getOriginalFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let row_addrs_obj: JByteArray<'_> = env
            .call_method(self, "getRowAddrs", "()[B", &[])?
            .l()?
            .into();
        let row_addrs = if row_addrs_obj.is_null() {
            None
        } else {
            Some(env.convert_byte_array(row_addrs_obj)?)
        };
        Ok(RewriteResult {
            metrics,
            new_fragments,
            read_version,
            original_fragments,
            row_addrs,
        })
    }
}
