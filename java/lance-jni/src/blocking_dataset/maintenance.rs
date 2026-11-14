// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{BlockingDataset, NATIVE_DATASET, clone_dataset};
use crate::block_on;
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::utils::build_compaction_options;
use chrono::{DateTime, Utc};
use jni::JNIEnv;
use jni::objects::{JObject, JString, JValue};
use jni::sys::jboolean;
use lance::dataset::cleanup::{
    CleanupCandidateFile, CleanupExplanation, CleanupFileKind, CleanupPolicy,
    CleanupReferencedBranch, RemovalStats, parse_cleanup_policy,
};
use lance::dataset::expire::{
    ExpireVersionsPlan, ExpireVersionsPolicy, ExpireVersionsStats, expire_versions,
    explain_expire_versions,
};
use lance::dataset::optimize::{CompactionOptions as RustCompactionOptions, compact_files};
use std::collections::HashSet;
use std::time::{Duration, UNIX_EPOCH};

impl BlockingDataset {
    pub fn compact(&mut self, options: RustCompactionOptions) -> Result<()> {
        block_on(compact_files(&mut self.inner, options, None))?;
        Ok(())
    }

    pub fn expire_versions(&mut self, policy: ExpireVersionsPolicy) -> Result<ExpireVersionsStats> {
        Ok(block_on(expire_versions(&self.inner, policy))?)
    }

    pub fn explain_expire_versions(
        &self,
        policy: ExpireVersionsPolicy,
    ) -> Result<ExpireVersionsPlan> {
        Ok(block_on(explain_expire_versions(&self.inner, &policy))?)
    }
}

//////////////////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCompact(
    mut env: JNIEnv,
    java_dataset: JObject,
    compaction_options: JObject, // CompactionOptions
) {
    ok_or_throw_without_return!(
        env,
        inner_compact(&mut env, java_dataset, compaction_options)
    )
}

fn inner_compact(
    env: &mut JNIEnv,
    java_dataset: JObject,
    compaction_options: JObject, // CompactionOptions
) -> Result<()> {
    let config = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset.inner.manifest.config.clone()
    };
    let rust_options = convert_java_compaction_options_to_rust(env, compaction_options, &config)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.compact(rust_options)?;
    Ok(())
}

fn convert_java_compaction_options_to_rust(
    env: &mut JNIEnv,
    java_options: JObject,
    config: &std::collections::HashMap<String, String>,
) -> Result<RustCompactionOptions> {
    let target_rows_per_fragment = env
        .call_method(
            &java_options,
            "getTargetRowsPerFragment",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_rows_per_group = env
        .call_method(
            &java_options,
            "getMaxRowsPerGroup",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_bytes_per_file = env
        .call_method(
            &java_options,
            "getMaxBytesPerFile",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let min_bytes_per_file = env
        .call_method(
            &java_options,
            "getMinBytesPerFile",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let materialize_deletions = env
        .call_method(
            &java_options,
            "getMaterializeDeletions",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let materialize_deletions_threshold = env
        .call_method(
            &java_options,
            "getMaterializeDeletionsThreshold",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let num_threads = env
        .call_method(
            &java_options,
            "getNumThreads",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let batch_size = env
        .call_method(&java_options, "getBatchSize", "()Ljava/util/Optional;", &[])?
        .l()?;
    let defer_index_remap = env
        .call_method(
            &java_options,
            "getDeferIndexRemap",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let compaction_mode = env
        .call_method(
            &java_options,
            "getCompactionMode",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let binary_copy_read_batch_bytes = env
        .call_method(
            &java_options,
            "getBinaryCopyReadBatchBytes",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_source_fragments = env
        .call_method(
            &java_options,
            "getMaxSourceFragments",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_source_rows = env
        .call_method(
            &java_options,
            "getMaxSourceRows",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let max_source_bytes = env
        .call_method(
            &java_options,
            "getMaxSourceBytes",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let excluded_fragment_ids = env
        .call_method(
            &java_options,
            "getExcludedFragmentIds",
            "()Ljava/util/List;",
            &[],
        )?
        .l()?;
    let data_storage_version = env
        .call_method(
            &java_options,
            "getDataStorageVersion",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let source_budget_mode = env
        .call_method(
            &java_options,
            "getSourceBudgetMode",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;

    build_compaction_options(
        env,
        &target_rows_per_fragment,
        &max_rows_per_group,
        &max_bytes_per_file,
        &min_bytes_per_file,
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
        &data_storage_version,
        Some(&source_budget_mode),
        config,
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCleanupWithPolicy<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
    jdelete_concurrency: JObject,
    jdelete_rate_limit: JObject,
    jretention_policy: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_cleanup_with_policy(
            &mut env,
            jdataset,
            jpolicy,
            jdelete_concurrency,
            jdelete_rate_limit,
            jretention_policy
        )
    )
}

fn inner_cleanup_with_policy<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
    jdelete_concurrency: JObject,
    jdelete_rate_limit: JObject,
    jretention_policy: JObject,
) -> Result<JObject<'local>> {
    let policy = extract_cleanup_policy(env, &jpolicy)?;
    let retention_policy: Option<String> = env.get_optional(&jretention_policy, |env, obj| {
        Ok(env.get_string(&JString::from(obj))?.into())
    })?;
    let delete_concurrency = env.get_optional(&jdelete_concurrency, |env, obj| {
        let value = env.call_method(obj, "longValue", "()J", &[])?.j()?;
        usize::try_from(value).map_err(|_| {
            Error::input_error(format!(
                "delete_concurrency is out of range for usize, got {value}"
            ))
        })
    })?;

    let delete_rate_limit = env.get_optional(&jdelete_rate_limit, |env, obj| {
        let value = env.call_method(obj, "longValue", "()J", &[])?.j()?;
        u64::try_from(value).map_err(|_| {
            Error::input_error(format!(
                "delete_rate_limit is out of range for u64, got {value}"
            ))
        })
    })?;
    let stats = {
        let dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        let mut operation = dataset.inner.cleanup(policy);
        if let Some(value) = retention_policy {
            operation = operation.with_retention_policy(parse_cleanup_policy(&value)?)?;
        }
        if let Some(concurrency) = delete_concurrency {
            operation = operation.with_delete_concurrency(concurrency)?;
        }
        if let Some(rate) = delete_rate_limit {
            operation = operation.with_delete_rate_limit(rate)?;
        }
        block_on(operation.execute())
    }?;

    cleanup_stats_to_java(env, stats)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeExplainCleanupWithPolicy<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
    jmax_candidate_files: JObject,
    jretention_policy: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_explain_cleanup_with_policy(
            &mut env,
            jdataset,
            jpolicy,
            jmax_candidate_files,
            jretention_policy
        )
    )
}

fn inner_explain_cleanup_with_policy<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
    jmax_candidate_files: JObject,
    jretention_policy: JObject,
) -> Result<JObject<'local>> {
    let policy = extract_cleanup_policy(env, &jpolicy)?;
    let retention_policy: Option<String> = env.get_optional(&jretention_policy, |env, obj| {
        Ok(env.get_string(&JString::from(obj))?.into())
    })?;
    let max_candidate_files = env
        .get_optional(&jmax_candidate_files, |env, inner| {
            Ok(env.call_method(inner, "longValue", "()J", &[])?.j()?)
        })?
        .map(|v| {
            usize::try_from(v).map_err(|e| {
                Error::input_error(format!(
                    "maxCandidateFiles must be a non-negative usize value, got {}: {:?}",
                    v, e
                ))
            })
        })
        .transpose()?;

    let explanation = {
        let dataset = clone_dataset(env, &jdataset)?;
        let mut operation = dataset.inner.cleanup(policy);
        if let Some(value) = retention_policy {
            operation = operation.with_retention_policy(parse_cleanup_policy(&value)?)?;
        }
        if let Some(limit) = max_candidate_files {
            operation = operation.with_max_candidate_files(limit);
        }
        block_on(operation.explain())
    }?;

    cleanup_explanation_to_java(env, explanation)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeExpireVersions<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_expire_versions(&mut env, jdataset, jpolicy))
}

fn inner_expire_versions<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
) -> Result<JObject<'local>> {
    let policy = extract_expire_versions_policy(env, &jpolicy)?;

    let stats = {
        let mut dataset =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
        dataset.expire_versions(policy)
    }?;

    expire_versions_stats_to_java(env, stats)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeExplainExpireVersions<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_explain_expire_versions(&mut env, jdataset, jpolicy)
    )
}

fn inner_explain_expire_versions<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jpolicy: JObject,
) -> Result<JObject<'local>> {
    let policy = extract_expire_versions_policy(env, &jpolicy)?;

    let plan = clone_dataset(env, &jdataset)?.explain_expire_versions(policy)?;

    expire_versions_plan_to_java(env, plan)
}

fn long_list_to_java<'local>(env: &mut JNIEnv<'local>, values: &[u64]) -> Result<JObject<'local>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for value in values {
        let boxed = env.new_object("java/lang/Long", "(J)V", &[JValue::Long(*value as i64)])?;
        env.call_method(
            &list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&boxed)],
        )?;
    }
    Ok(list)
}

fn expire_versions_plan_to_java<'local>(
    env: &mut JNIEnv<'local>,
    plan: ExpireVersionsPlan,
) -> Result<JObject<'local>> {
    let versions = long_list_to_java(env, &plan.versions)?;
    let tagged_but_kept = long_list_to_java(env, &plan.tagged_but_kept)?;
    let stats = expire_versions_stats_to_java(env, plan.stats)?;
    Ok(env.new_object(
        "org/lance/expire/ExpireVersionsPlan",
        "(Ljava/util/List;Lorg/lance/expire/ExpireVersionsStats;Ljava/util/List;)V",
        &[
            JValue::Object(&versions),
            JValue::Object(&stats),
            JValue::Object(&tagged_but_kept),
        ],
    )?)
}

fn extract_expire_versions_policy(
    env: &mut JNIEnv<'_>,
    jpolicy: &JObject,
) -> Result<ExpireVersionsPolicy> {
    let before_timestamp = env
        .get_optional_u64_from_method(jpolicy, "getBeforeTimestampMillis")?
        .map(|millis| DateTime::<Utc>::from(UNIX_EPOCH + Duration::from_millis(millis)));
    let before_version = env.get_optional_u64_from_method(jpolicy, "getBeforeVersion")?;
    let keep_one_per = env
        .get_optional_u64_from_method(jpolicy, "getKeepOnePerMicros")?
        .map(Duration::from_micros);
    let error_if_tagged_old_versions = env
        .get_optional_from_method(jpolicy, "getErrorIfTaggedOldVersions", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(true);
    let delete_rate_limit = env.get_optional_u64_from_method(jpolicy, "getDeleteRateLimit")?;

    Ok(ExpireVersionsPolicy {
        before_timestamp,
        before_version,
        keep_one_per,
        error_if_tagged_old_versions,
        delete_rate_limit,
    })
}

fn expire_versions_stats_to_java<'local>(
    env: &mut JNIEnv<'local>,
    stats: ExpireVersionsStats,
) -> Result<JObject<'local>> {
    Ok(env.new_object(
        "org/lance/expire/ExpireVersionsStats",
        "(JJJJ)V",
        &[
            JValue::Long(stats.versions_removed as i64),
            JValue::Long(stats.versions_retained as i64),
            JValue::Long(stats.bytes_removed as i64),
            JValue::Long(stats.failed_deletes as i64),
        ],
    )?)
}

fn extract_cleanup_policy(env: &mut JNIEnv<'_>, jpolicy: &JObject) -> Result<CleanupPolicy> {
    let before_ts_millis = env.get_optional_u64_from_method(jpolicy, "getBeforeTimestampMillis")?;
    let before_timestamp = before_ts_millis.map(|millis| {
        let st = UNIX_EPOCH + Duration::from_millis(millis);
        DateTime::<Utc>::from(st)
    });

    let before_version = env.get_optional_u64_from_method(jpolicy, "getBeforeVersion")?;

    let versions = env.get_optional_from_method(jpolicy, "getVersions", |env, list_obj| {
        let mut versions = HashSet::new();
        for version in env.get_longs(&list_obj)? {
            versions.insert(version as u64);
        }
        Ok(versions)
    })?;

    let delete_unverified = env
        .get_optional_from_method(jpolicy, "getDeleteUnverified", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(false);

    let error_if_tagged_old_versions = env
        .get_optional_from_method(jpolicy, "getErrorIfTaggedOldVersions", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(true);

    let clean_referenced_branches = env
        .get_optional_from_method(jpolicy, "getCleanReferencedBranches", |env, obj| {
            Ok(env.call_method(obj, "booleanValue", "()Z", &[])?.z()?)
        })?
        .unwrap_or(false);

    let delete_rate_limit = env.get_optional_u64_from_method(jpolicy, "getDeleteRateLimit")?;

    Ok(CleanupPolicy {
        before_timestamp,
        before_version,
        versions,
        delete_unverified,
        error_if_tagged_old_versions,
        clean_referenced_branches,
        delete_rate_limit,
    })
}

fn cleanup_stats_to_java<'local>(
    env: &mut JNIEnv<'local>,
    stats: RemovalStats,
) -> Result<JObject<'local>> {
    Ok(env.new_object(
        "org/lance/cleanup/RemovalStats",
        "(JJJJJJJ)V",
        &[
            JValue::Long(stats.bytes_removed as i64),
            JValue::Long(stats.old_versions as i64),
            JValue::Long(stats.data_files_removed as i64),
            JValue::Long(stats.transaction_files_removed as i64),
            JValue::Long(stats.index_files_removed as i64),
            JValue::Long(stats.deletion_files_removed as i64),
            JValue::Long(stats.failed_deletes as i64),
        ],
    )?)
}

fn cleanup_file_kind_to_java(kind: CleanupFileKind) -> &'static str {
    match kind {
        CleanupFileKind::Manifest => "manifest",
        CleanupFileKind::Data => "data",
        CleanupFileKind::Transaction => "transaction",
        CleanupFileKind::Index => "index",
        CleanupFileKind::Deletion => "deletion",
        CleanupFileKind::TemporaryManifest => "temporary_manifest",
    }
}

fn cleanup_candidate_files_to_java<'local>(
    env: &mut JNIEnv<'local>,
    files: Vec<CleanupCandidateFile>,
) -> Result<JObject<'local>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    // Wrap each iteration in a local frame so the temporary path/kind/candidate
    // references do not accumulate on the JNI local reference table for large
    // explanations (default limit is 1000 candidate files, but users can raise it).
    for file in files {
        env.with_local_frame(8, |env| {
            let path = env.new_string(file.path)?;
            let kind = env.new_string(cleanup_file_kind_to_java(file.kind))?;
            let candidate = env.new_object(
                "org/lance/cleanup/CleanupCandidateFile",
                "(Ljava/lang/String;Ljava/lang/String;ZJ)V",
                &[
                    JValue::Object(&path),
                    JValue::Object(&kind),
                    JValue::Bool(file.unverified as jboolean),
                    JValue::Long(file.size_bytes as i64),
                ],
            )?;
            env.call_method(
                &list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&candidate)],
            )?;
            Ok::<(), Error>(())
        })?;
    }
    Ok(list)
}

fn cleanup_referenced_branches_to_java<'local>(
    env: &mut JNIEnv<'local>,
    branches: Vec<CleanupReferencedBranch>,
) -> Result<JObject<'local>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for branch in branches {
        env.with_local_frame(8, |env| {
            let name = env.new_string(branch.name)?;
            let referenced_branch = env.new_object(
                "org/lance/cleanup/CleanupReferencedBranch",
                "(Ljava/lang/String;JZ)V",
                &[
                    JValue::Object(&name),
                    JValue::Long(branch.referenced_version as i64),
                    JValue::Bool(branch.cleanup_candidate as jboolean),
                ],
            )?;
            env.call_method(
                &list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&referenced_branch)],
            )?;
            Ok::<(), Error>(())
        })?;
    }
    Ok(list)
}

fn cleanup_warnings_to_java<'local>(
    env: &mut JNIEnv<'local>,
    warnings: Vec<String>,
) -> Result<JObject<'local>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for warning in warnings {
        env.with_local_frame(4, |env| {
            let warning = env.new_string(warning)?;
            env.call_method(
                &list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&warning)],
            )?;
            Ok::<(), Error>(())
        })?;
    }
    Ok(list)
}

fn cleanup_explanation_to_java<'local>(
    env: &mut JNIEnv<'local>,
    explanation: CleanupExplanation,
) -> Result<JObject<'local>> {
    let stats = cleanup_stats_to_java(env, explanation.stats)?;
    let candidate_files = cleanup_candidate_files_to_java(env, explanation.candidate_files)?;
    let referenced_branches =
        cleanup_referenced_branches_to_java(env, explanation.referenced_branches)?;
    let warnings = cleanup_warnings_to_java(env, explanation.warnings)?;

    Ok(env.new_object(
        "org/lance/cleanup/CleanupExplanation",
        "(JLorg/lance/cleanup/RemovalStats;Ljava/util/List;ZJLjava/util/List;Ljava/util/List;)V",
        &[
            JValue::Long(explanation.read_version as i64),
            JValue::Object(&stats),
            JValue::Object(&candidate_files),
            JValue::Bool(explanation.candidate_files_truncated as jboolean),
            JValue::Long(explanation.candidate_file_limit as i64),
            JValue::Object(&referenced_branches),
            JValue::Object(&warnings),
        ],
    )?)
}
