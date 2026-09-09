// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::Result;
use crate::traits::import_vec_to_rust;
use crate::traits::{FromJString, IntoJava};
use crate::{Error, JNIEnvExt, block_on};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use jni::JNIEnv;
use jni::objects::{JObject, JString, JValueGen};
use jni::sys::jlong;
use lance::dataset::scanner::ExprFilter;
use lance::dataset::{
    MergeInsertBuilder, MergeStats, WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
};
use lance_core::datatypes::Schema;
use lance_index::mem_wal::CompactedSsTable;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeMergeInsert<'a>(
    mut env: JNIEnv<'a>,
    jdataset: JObject,    // Dataset object
    jparam: JObject,      // MergeInsertParams object
    batch_address: jlong, // ArrowArrayStream address for source
) -> JObject<'a> {
    ok_or_throw!(
        env,
        inner_merge_insert(&mut env, jdataset, jparam, batch_address)
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_merge_insert<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject,
    jparam: JObject,
    batch_address: jlong,
) -> Result<JObject<'local>> {
    let on = extract_on(env, &jparam)?;
    let when_matched = extract_when_matched(env, &jparam)?;
    let when_not_matched = extract_when_not_matached(env, &jparam)?;

    let when_not_matched_by_source_str = extract_when_not_matched_by_source_str(env, &jparam)?;
    let when_not_matched_by_source_delete_expr =
        extract_when_not_matched_by_source_delete_expr(env, &jparam)?;

    let conflict_retries = extract_conflict_retries(env, &jparam)?;
    let retry_timeout_ms = extract_retry_timeout_ms(env, &jparam)?;
    let skip_auto_cleanup = extract_skip_auto_cleanup(env, &jparam)?;
    let use_index = extract_use_index(env, &jparam)?;
    let compacted_sstables = extract_compacted_sstables(env, &jparam)?;

    let (new_ds, merge_stats) = unsafe {
        let dataset = env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET)?;

        let when_not_matched_by_source = extract_when_not_matched_by_source(
            dataset.inner.schema(),
            when_not_matched_by_source_str.as_str(),
            when_not_matched_by_source_delete_expr,
        )?;

        let merge_insert_job = MergeInsertBuilder::try_new(Arc::new(dataset.clone().inner), on)?
            .when_matched(when_matched)
            .when_not_matched(when_not_matched)
            .when_not_matched_by_source(when_not_matched_by_source)
            .conflict_retries(conflict_retries)
            .retry_timeout(Duration::from_millis(retry_timeout_ms as u64))
            .skip_auto_cleanup(skip_auto_cleanup)
            .use_index(use_index)
            .mark_sstables_as_compacted(compacted_sstables)
            .try_build()?;

        let stream_ptr = batch_address as *mut FFI_ArrowArrayStream;
        let source_stream = ArrowArrayStreamReader::from_raw(stream_ptr)?;

        block_on(async move { merge_insert_job.execute_reader(source_stream).await })?
    };

    MergeResult(
        BlockingDataset {
            inner: Arc::try_unwrap(new_ds).unwrap(),
        },
        merge_stats,
    )
    .into_java(env)
}

fn extract_on<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<Vec<String>> {
    let on: JObject = env
        .call_method(jparam, "on", "()Ljava/util/List;", &[])?
        .l()?;
    env.get_strings(&on)
}

fn extract_when_matched<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<WhenMatched> {
    let when_matched: JString = env
        .call_method(jparam, "whenMatchedValue", "()Ljava/lang/String;", &[])?
        .l()?
        .into();
    let when_matched = when_matched.extract(env)?;

    let when_matched_update_expr = env
        .call_method(
            jparam,
            "whenMatchedUpdateExpr",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let when_matched_update_expr = env.get_string_opt(&when_matched_update_expr)?;

    match when_matched.as_str() {
        "UpdateAll" => Ok(WhenMatched::UpdateAll),
        "DoNothing" => Ok(WhenMatched::DoNothing),
        "UpdateIf" => match when_matched_update_expr {
            Some(expr) => Ok(WhenMatched::UpdateIf(expr)),
            None => Err(Error::input_error("No matched updated expr".to_string())),
        },
        "Fail" => Ok(WhenMatched::Fail),
        "Delete" => Ok(WhenMatched::Delete),
        _ => Err(Error::input_error(format!(
            "Illegal when_matched: {when_matched}",
        ))),
    }
}

fn extract_when_not_matached<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<WhenNotMatched> {
    let when_not_matched: JString = env
        .call_method(jparam, "whenNotMatchedValue", "()Ljava/lang/String;", &[])?
        .l()?
        .into();
    let when_not_matched = when_not_matched.extract(env)?;

    match when_not_matched.as_str() {
        "InsertAll" => Ok(WhenNotMatched::InsertAll),
        "DoNothing" => Ok(WhenNotMatched::DoNothing),
        _ => Err(Error::input_error(format!(
            "Illegal when_not_matched: {when_not_matched}",
        ))),
    }
}

fn extract_when_not_matched_by_source_str<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<String> {
    let when_not_matched_by_source: JString = env
        .call_method(
            jparam,
            "whenNotMatchedBySourceValue",
            "()Ljava/lang/String;",
            &[],
        )?
        .l()?
        .into();
    when_not_matched_by_source.extract(env)
}

fn extract_when_not_matched_by_source_delete_expr<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<Option<ExprFilter>> {
    let when_not_matched_by_source_delete_expr = env
        .call_method(
            jparam,
            "whenNotMatchedBySourceDeleteExpr",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;

    if let Some(expr) = env.get_string_opt(&when_not_matched_by_source_delete_expr)? {
        return Ok(Some(ExprFilter::Sql(expr)));
    }

    let when_not_matched_by_source_delete_substrait_expr = env
        .call_method(
            jparam,
            "whenNotMatchedBySourceDeleteSubstraitExpr",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;

    match env.get_bytes_opt(&when_not_matched_by_source_delete_substrait_expr)? {
        Some(expr) => Ok(Some(ExprFilter::Substrait(expr.to_vec()))),
        None => Ok(None),
    }
}

fn extract_when_not_matched_by_source(
    schema: &Schema,
    when_not_matched_by_source: &str,
    when_not_matched_by_source_delete_expr: Option<ExprFilter>,
) -> Result<WhenNotMatchedBySource> {
    match when_not_matched_by_source {
        "Keep" => Ok(WhenNotMatchedBySource::Keep),
        "Delete" => Ok(WhenNotMatchedBySource::Delete),
        "DeleteIf" => match when_not_matched_by_source_delete_expr {
            Some(expr) => Ok(WhenNotMatchedBySource::DeleteIf(
                expr.to_datafusion(schema, schema)?,
            )),
            None => Err(Error::input_error(format!(
                "No delete expr when not matched by source is: {when_not_matched_by_source}",
            ))),
        },
        _ => Err(Error::input_error(format!(
            "Illegal when_not_matched_by_source: {when_not_matched_by_source}",
        ))),
    }
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

fn extract_skip_auto_cleanup<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<bool> {
    let skip_auto_cleanup = env
        .call_method(jparam, "skipAutoCleanup", "()Z", &[])?
        .z()?;
    Ok(skip_auto_cleanup)
}

fn extract_use_index<'local>(env: &mut JNIEnv<'local>, jparam: &JObject) -> Result<bool> {
    let use_index = env.call_method(jparam, "useIndex", "()Z", &[])?.z()?;
    Ok(use_index)
}

fn extract_compacted_sstables<'local>(
    env: &mut JNIEnv<'local>,
    jparam: &JObject,
) -> Result<Vec<CompactedSsTable>> {
    let list = env
        .call_method(jparam, "getCompactedSstables", "()Ljava/util/List;", &[])?
        .l()?;
    import_vec_to_rust(env, &list, |env, obj| {
        let shard_id: JString = env
            .call_method(&obj, "getShardId", "()Ljava/lang/String;", &[])?
            .l()?
            .into();
        let shard_id = shard_id.extract(env)?;
        let generation = env.call_method(&obj, "getGeneration", "()J", &[])?.j()? as u64;
        let uuid = Uuid::parse_str(&shard_id)
            .map_err(|e| Error::input_error(format!("Invalid shard_id UUID: {}", e)))?;
        Ok(CompactedSsTable::new(uuid, generation))
    })
}

const MERGE_STATS_CLASS: &str = "org/lance/merge/MergeInsertStats";
const MERGE_STATS_CONSTRUCTOR_SIG: &str = "(JJJIJJ)V";
const MERGE_RESULT_CLASS: &str = "org/lance/merge/MergeInsertResult";
const MERGE_RESULT_CONSTRUCTOR_SIG: &str =
    "(Lorg/lance/Dataset;Lorg/lance/merge/MergeInsertStats;)V";

impl IntoJava for MergeStats {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        Ok(env.new_object(
            MERGE_STATS_CLASS,
            MERGE_STATS_CONSTRUCTOR_SIG,
            &[
                JValueGen::Long(self.num_inserted_rows as i64),
                JValueGen::Long(self.num_updated_rows as i64),
                JValueGen::Long(self.num_deleted_rows as i64),
                JValueGen::Int(self.num_attempts as i32),
                JValueGen::Long(self.bytes_written as i64),
                JValueGen::Long(self.num_files_written as i64),
            ],
        )?)
    }
}

struct MergeResult(BlockingDataset, MergeStats);

impl IntoJava for MergeResult {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let jdataset = self.0.into_java(env)?;
        let jstats = self.1.into_java(env)?;
        Ok(env.new_object(
            MERGE_RESULT_CLASS,
            MERGE_RESULT_CONSTRUCTOR_SIG,
            &[JValueGen::Object(&jdataset), JValueGen::Object(&jstats)],
        )?)
    }
}
