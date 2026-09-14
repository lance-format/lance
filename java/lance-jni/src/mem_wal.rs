// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JNI bindings for the Lance MemWAL feature.
//!
//! Mirrors the Python MemWAL binding (`python/src/mem_wal.rs`): a stateful
//! [`ShardWriter`], an LSM-aware [`LsmScanner`], an [`ExecutionPlan`] wrapper,
//! and the point-lookup / vector-search planners.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator,
    RecordBatchReader, StructArray, make_array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::SessionContext;
use jni::JNIEnv;
use jni::objects::{JClass, JMap, JObject, JString, JValueGen};
use jni::sys::{jdouble, jint, jlong};
use lance::dataset::Dataset as LanceDataset;
use lance::dataset::mem_wal::scanner::{
    LsmDataSourceCollector, LsmPointLookupPlanner, LsmVectorSearchPlanner, SsTable,
    parse_filter_expr as parse_lsm_filter_expr, write_pk_sidecar,
};
use lance::dataset::mem_wal::write::{MemTableStats, ShardMemory, WriteStatsSnapshot};
use lance::dataset::mem_wal::{
    DatasetMemWalExt, LsmScanner, ShardSnapshot, ShardWriter, ShardWriterConfig,
    evaluate_sharding_spec_with_source_columns,
};
use lance::dataset::scanner::DatasetRecordBatchStream;
use lance_index::mem_wal::{MemWalIndexDetails, ShardManifest, ShardingField, ShardingSpec};
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_io::ffi::to_ffi_arrow_array_stream;
use lance_linalg::distance::DistanceType;
use uuid::Uuid;

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::traits::{
    FromJString, IntoJava, export_vec, import_vec, import_vec_from_method, import_vec_to_rust,
};
use crate::utils::to_rust_map;
use crate::{RT, block_on};

const NATIVE_SHARD_WRITER: &str = "nativeShardWriterHandle";
const NATIVE_LSM_SCANNER: &str = "nativeLsmScannerHandle";
const NATIVE_EXECUTION_PLAN: &str = "nativeExecutionPlanHandle";
const NATIVE_LOOKUP_PLANNER: &str = "nativeLookupPlannerHandle";
const NATIVE_VECTOR_PLANNER: &str = "nativeVectorPlannerHandle";

/// Native handle backing `org.lance.memwal.ShardWriter`.
struct BlockingShardWriter {
    writer: ShardWriter,
    shard_id: Uuid,
    dataset: Arc<LanceDataset>,
}

/// Native handle backing `org.lance.memwal.LsmScanner`.
struct BlockingLsmScanner {
    inner: Option<LsmScanner>,
}

/// Native handle backing `org.lance.memwal.ExecutionPlan`.
struct BlockingExecutionPlan {
    plan: Arc<dyn ExecutionPlan>,
    dataset_schema: Arc<ArrowSchema>,
}

/// Native handle backing `org.lance.memwal.LsmPointLookupPlanner`.
struct BlockingLsmPointLookupPlanner {
    planner: LsmPointLookupPlanner,
    dataset_schema: Arc<ArrowSchema>,
    pk_columns: Vec<String>,
}

/// Native handle backing `org.lance.memwal.LsmVectorSearchPlanner`.
struct BlockingLsmVectorSearchPlanner {
    planner: LsmVectorSearchPlanner,
    vector_dim: usize,
    dataset_schema: Arc<ArrowSchema>,
}

///////////////////
// ShardWriter  //
///////////////////

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardWriter_createNative<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dataset: JObject<'local>,
    shard_id: JString<'local>,
    config: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_shard_writer(&mut env, dataset, shard_id, config)
    )
}

fn inner_create_shard_writer<'local>(
    env: &mut JNIEnv<'local>,
    dataset: JObject<'local>,
    shard_id: JString<'local>,
    config: JObject<'local>,
) -> Result<JObject<'local>> {
    let shard_id: String = env.get_string(&shard_id)?.into();
    let uuid = parse_uuid(&shard_id)?;
    let writer_config = if config.is_null() {
        ShardWriterConfig::default()
    } else {
        build_writer_config(env, &config)?
    };

    let dataset = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET) }?;
        Arc::new(guard.inner.clone())
    };

    let writer = block_on(dataset.mem_wal_writer(uuid, writer_config))?;
    let blocking = BlockingShardWriter {
        writer,
        shard_id: uuid,
        dataset,
    };

    let java_obj = env.new_object("org/lance/memwal/ShardWriter", "()V", &[])?;
    unsafe { env.set_rust_field(&java_obj, NATIVE_SHARD_WRITER, blocking) }?;
    Ok(java_obj)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardWriter_nativeShardId<'local>(
    mut env: JNIEnv<'local>,
    this: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(env, inner_shard_id(&mut env, this))
}

fn inner_shard_id<'local>(
    env: &mut JNIEnv<'local>,
    this: JObject<'local>,
) -> Result<JObject<'local>> {
    let shard_id = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingShardWriter>(&this, NATIVE_SHARD_WRITER) }?;
        guard.shard_id.to_string()
    };
    Ok(env.new_string(shard_id)?.into())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardWriter_nativePut(
    mut env: JNIEnv,
    this: JObject,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_put(&mut env, this, stream_addr));
}

fn inner_put(env: &mut JNIEnv, this: JObject, stream_addr: jlong) -> Result<()> {
    let stream_ptr = stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
    let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>()?;
    if batches.is_empty() {
        return Ok(());
    }

    let guard =
        unsafe { env.get_rust_field::<_, _, BlockingShardWriter>(&this, NATIVE_SHARD_WRITER) }?;
    block_on(guard.writer.put(batches))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardWriter_nativeDelete(
    mut env: JNIEnv,
    this: JObject,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_delete(&mut env, this, stream_addr));
}

fn inner_delete(env: &mut JNIEnv, this: JObject, stream_addr: jlong) -> Result<()> {
    let stream_ptr = stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
    let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>()?;
    if batches.is_empty() {
        return Ok(());
    }

    let guard =
        unsafe { env.get_rust_field::<_, _, BlockingShardWriter>(&this, NATIVE_SHARD_WRITER) }?;
    block_on(guard.writer.delete(batches))?;
    Ok(())
}

/// Test-support: write a primary-key dedup sidecar (`_pk_index/`) for an
/// SSTable dataset already staged at `gen_path`, mirroring what
/// production flush emits. Lets Java tests stage a *faithful* SSTable
/// (dataset + sidecar); production always writes the sidecar during
/// flush, so a dataset-without-sidecar is not a state the system produces.
/// Mirrors the Python `_write_pk_sidecar` binding.
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_MemWalTest_nativeWritePkSidecar(
    mut env: JNIEnv,
    _class: JClass,
    gen_path: JString,
    stream_addr: jlong,
    pk_columns: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_write_pk_sidecar(&mut env, gen_path, stream_addr, pk_columns)
    );
}

fn inner_write_pk_sidecar(
    env: &mut JNIEnv,
    gen_path: JString,
    stream_addr: jlong,
    pk_columns: JObject,
) -> Result<()> {
    let gen_path: String = env.get_string(&gen_path)?.into();
    let pk_columns = env.get_strings(&pk_columns)?;
    let stream_ptr = stream_addr as *mut FFI_ArrowArrayStream;
    let reader = unsafe { ArrowArrayStreamReader::from_raw(stream_ptr) }?;
    let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>()?;
    let pk_refs: Vec<&str> = pk_columns.iter().map(String::as_str).collect();
    block_on(write_pk_sidecar(&gen_path, &batches, &pk_refs))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardWriter_nativeStats<'local>(
    mut env: JNIEnv<'local>,
    this: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(env, inner_writer_stats(&mut env, this))
}

fn inner_writer_stats<'local>(
    env: &mut JNIEnv<'local>,
    this: JObject<'local>,
) -> Result<JObject<'local>> {
    let stats = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingShardWriter>(&this, NATIVE_SHARD_WRITER) }?;
        guard.writer.stats()
    };
    write_stats_to_java(env, &stats)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardWriter_nativeMemtableStats<'local>(
    mut env: JNIEnv<'local>,
    this: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(env, inner_memtable_stats(&mut env, this))
}

fn inner_memtable_stats<'local>(
    env: &mut JNIEnv<'local>,
    this: JObject<'local>,
) -> Result<JObject<'local>> {
    let (stats, memory) = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingShardWriter>(&this, NATIVE_SHARD_WRITER) }?;
        // Byte totals live on `memory()` now, not on `MemTableStats`.
        (
            block_on(guard.writer.memtable_stats())?,
            guard.writer.memory(),
        )
    };
    memtable_stats_to_java(env, &stats, &memory)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardWriter_nativeLsmScanner<'local>(
    mut env: JNIEnv<'local>,
    this: JObject<'local>,
    shard_snapshots: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_writer_lsm_scanner(&mut env, this, shard_snapshots)
    )
}

fn inner_writer_lsm_scanner<'local>(
    env: &mut JNIEnv<'local>,
    this: JObject<'local>,
    shard_snapshots: JObject<'local>,
) -> Result<JObject<'local>> {
    let mut snapshots = read_shard_snapshots(env, &shard_snapshots)?;

    let (in_memory_memtables, writer_snapshot, dataset, shard_id, pk_columns) = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingShardWriter>(&this, NATIVE_SHARD_WRITER) }?;
        let pk_columns = get_pk_columns(&guard.dataset)?;
        // Capture the active memtable *and* any frozen-awaiting-flush memtables
        // so a concurrent flush rollover cannot hide acknowledged writes from
        // this read-your-writes scanner.
        let in_memory_memtables = block_on(guard.writer.in_memory_memtable_refs())?;
        let writer_snapshot = block_on(guard.writer.manifest())?
            .map(shard_snapshot_from_manifest)
            .unwrap_or_else(|| ShardSnapshot::new(guard.shard_id));
        (
            in_memory_memtables,
            writer_snapshot,
            guard.dataset.clone(),
            guard.shard_id,
            pk_columns,
        )
    };

    snapshots.retain(|snapshot| snapshot.shard_id != shard_id);
    snapshots.push(writer_snapshot);

    let scanner = LsmScanner::new(dataset, snapshots, pk_columns)
        .with_in_memory_memtables(shard_id, in_memory_memtables);
    attach_lsm_scanner(env, scanner)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardWriter_releaseNativeShardWriter(
    mut env: JNIEnv,
    this: JObject,
    _handle: jlong,
) {
    ok_or_throw_without_return!(env, inner_release_shard_writer(&mut env, this));
}

fn inner_release_shard_writer(env: &mut JNIEnv, this: JObject) -> Result<()> {
    let blocking: BlockingShardWriter = unsafe { env.take_rust_field(&this, NATIVE_SHARD_WRITER) }?;
    block_on(blocking.writer.close())?;
    Ok(())
}

///////////////////
// LsmScanner    //
///////////////////

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_createFromSnapshots<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dataset: JObject<'local>,
    shard_snapshots: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_lsm_scanner_from_snapshots(&mut env, dataset, shard_snapshots)
    )
}

fn inner_lsm_scanner_from_snapshots<'local>(
    env: &mut JNIEnv<'local>,
    dataset: JObject<'local>,
    shard_snapshots: JObject<'local>,
) -> Result<JObject<'local>> {
    let snapshots = read_shard_snapshots(env, &shard_snapshots)?;
    let dataset = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET) }?;
        Arc::new(guard.inner.clone())
    };
    let pk_columns = get_pk_columns(&dataset)?;
    let scanner = LsmScanner::new(dataset, snapshots, pk_columns);
    attach_lsm_scanner(env, scanner)
}

fn attach_lsm_scanner<'local>(
    env: &mut JNIEnv<'local>,
    scanner: LsmScanner,
) -> Result<JObject<'local>> {
    let java_obj = env.new_object("org/lance/memwal/LsmScanner", "()V", &[])?;
    unsafe {
        env.set_rust_field(
            &java_obj,
            NATIVE_LSM_SCANNER,
            BlockingLsmScanner {
                inner: Some(scanner),
            },
        )
    }?;
    Ok(java_obj)
}

/// Take the wrapped scanner out, apply `f`, then store the result back.
fn with_lsm_scanner<F>(env: &mut JNIEnv, this: &JObject, f: F) -> Result<()>
where
    F: FnOnce(LsmScanner) -> Result<LsmScanner>,
{
    let mut guard =
        unsafe { env.get_rust_field::<_, _, BlockingLsmScanner>(this, NATIVE_LSM_SCANNER) }?;
    let scanner = guard
        .inner
        .take()
        .ok_or_else(|| Error::runtime_error("LsmScanner is no longer usable because an earlier builder call (e.g. filter) failed; create a new scanner".to_string()))?;
    guard.inner = Some(f(scanner)?);
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_nativeProject(
    mut env: JNIEnv,
    this: JObject,
    columns: JObject,
) {
    ok_or_throw_without_return!(env, inner_scanner_project(&mut env, this, columns));
}

fn inner_scanner_project(env: &mut JNIEnv, this: JObject, columns: JObject) -> Result<()> {
    let columns = env.get_strings(&columns)?;
    with_lsm_scanner(env, &this, |scanner| {
        let cols: Vec<&str> = columns.iter().map(String::as_str).collect();
        Ok(scanner.project(&cols)?)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_nativeFilter(
    mut env: JNIEnv,
    this: JObject,
    expr: JString,
) {
    ok_or_throw_without_return!(env, inner_scanner_filter(&mut env, this, expr));
}

fn inner_scanner_filter(env: &mut JNIEnv, this: JObject, expr: JString) -> Result<()> {
    let expr: String = env.get_string(&expr)?.into();
    with_lsm_scanner(env, &this, |scanner| Ok(scanner.filter(&expr)?))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_nativeLimit(
    mut env: JNIEnv,
    this: JObject,
    limit: JObject,
    offset: JObject,
) {
    ok_or_throw_without_return!(env, inner_scanner_limit(&mut env, this, limit, offset));
}

fn inner_scanner_limit(
    env: &mut JNIEnv,
    this: JObject,
    limit: JObject,
    offset: JObject,
) -> Result<()> {
    let limit = env.get_u64_opt(&limit)?.map(|v| v as i64);
    let offset = env.get_u64_opt(&offset)?.map(|v| v as i64);
    with_lsm_scanner(env, &this, |scanner| Ok(scanner.limit(limit, offset)?))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_nativeWithRowAddress(
    mut env: JNIEnv,
    this: JObject,
) {
    ok_or_throw_without_return!(
        env,
        with_lsm_scanner(&mut env, &this, |scanner| Ok(scanner.with_row_address()))
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_nativeWithMemtableGen(
    mut env: JNIEnv,
    this: JObject,
) {
    ok_or_throw_without_return!(
        env,
        with_lsm_scanner(&mut env, &this, |scanner| Ok(scanner.with_memtable_gen()))
    );
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_nativeOpenStream(
    mut env: JNIEnv,
    this: JObject,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_scanner_open_stream(&mut env, this, stream_addr));
}

fn inner_scanner_open_stream(env: &mut JNIEnv, this: JObject, stream_addr: jlong) -> Result<()> {
    let stream = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingLsmScanner>(&this, NATIVE_LSM_SCANNER) }?;
        let scanner = guard.inner.as_ref().ok_or_else(|| {
            Error::runtime_error("LsmScanner is no longer usable because an earlier builder call (e.g. filter) failed; create a new scanner".to_string())
        })?;
        block_on(scanner.try_into_stream())?
    };
    let ffi_stream =
        to_ffi_arrow_array_stream(DatasetRecordBatchStream::new(stream), RT.handle().clone())?;
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_nativeCountRows(
    mut env: JNIEnv,
    this: JObject,
) -> jlong {
    ok_or_throw_with_return!(env, inner_scanner_count_rows(&mut env, this), -1) as jlong
}

fn inner_scanner_count_rows(env: &mut JNIEnv, this: JObject) -> Result<u64> {
    let guard =
        unsafe { env.get_rust_field::<_, _, BlockingLsmScanner>(&this, NATIVE_LSM_SCANNER) }?;
    let scanner = guard
        .inner
        .as_ref()
        .ok_or_else(|| Error::runtime_error("LsmScanner is no longer usable because an earlier builder call (e.g. filter) failed; create a new scanner".to_string()))?;
    Ok(block_on(scanner.count_rows())?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmScanner_releaseNativeLsmScanner(
    mut env: JNIEnv,
    this: JObject,
    _handle: jlong,
) {
    ok_or_throw_without_return!(env, inner_release_lsm_scanner(&mut env, this));
}

fn inner_release_lsm_scanner(env: &mut JNIEnv, this: JObject) -> Result<()> {
    let _: BlockingLsmScanner = unsafe { env.take_rust_field(&this, NATIVE_LSM_SCANNER) }?;
    Ok(())
}

///////////////////
// ExecutionPlan //
///////////////////

fn attach_execution_plan<'local>(
    env: &mut JNIEnv<'local>,
    plan: Arc<dyn ExecutionPlan>,
    dataset_schema: Arc<ArrowSchema>,
) -> Result<JObject<'local>> {
    let java_obj = env.new_object("org/lance/memwal/ExecutionPlan", "()V", &[])?;
    unsafe {
        env.set_rust_field(
            &java_obj,
            NATIVE_EXECUTION_PLAN,
            BlockingExecutionPlan {
                plan,
                dataset_schema,
            },
        )
    }?;
    Ok(java_obj)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ExecutionPlan_nativeImportSchema(
    mut env: JNIEnv,
    this: JObject,
    schema_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_plan_import_schema(&mut env, this, schema_addr));
}

fn inner_plan_import_schema(env: &mut JNIEnv, this: JObject, schema_addr: jlong) -> Result<()> {
    let schema = {
        let guard = unsafe {
            env.get_rust_field::<_, _, BlockingExecutionPlan>(&this, NATIVE_EXECUTION_PLAN)
        }?;
        guard.plan.schema()
    };
    let ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())?;
    unsafe { std::ptr::write_unaligned(schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ExecutionPlan_nativeImportDatasetSchema(
    mut env: JNIEnv,
    this: JObject,
    schema_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_plan_import_dataset_schema(&mut env, this, schema_addr)
    );
}

fn inner_plan_import_dataset_schema(
    env: &mut JNIEnv,
    this: JObject,
    schema_addr: jlong,
) -> Result<()> {
    let schema = {
        let guard = unsafe {
            env.get_rust_field::<_, _, BlockingExecutionPlan>(&this, NATIVE_EXECUTION_PLAN)
        }?;
        guard.dataset_schema.clone()
    };
    let ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())?;
    unsafe { std::ptr::write_unaligned(schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ExecutionPlan_nativeExplain<'local>(
    mut env: JNIEnv<'local>,
    this: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(env, inner_plan_explain(&mut env, this))
}

fn inner_plan_explain<'local>(
    env: &mut JNIEnv<'local>,
    this: JObject<'local>,
) -> Result<JObject<'local>> {
    let explained = {
        let guard = unsafe {
            env.get_rust_field::<_, _, BlockingExecutionPlan>(&this, NATIVE_EXECUTION_PLAN)
        }?;
        format!("{}", displayable(guard.plan.as_ref()).indent(true))
    };
    Ok(env.new_string(explained)?.into())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ExecutionPlan_nativeOpenStream(
    mut env: JNIEnv,
    this: JObject,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(env, inner_plan_open_stream(&mut env, this, stream_addr));
}

fn inner_plan_open_stream(env: &mut JNIEnv, this: JObject, stream_addr: jlong) -> Result<()> {
    let plan = {
        let guard = unsafe {
            env.get_rust_field::<_, _, BlockingExecutionPlan>(&this, NATIVE_EXECUTION_PLAN)
        }?;
        guard.plan.clone()
    };
    let schema = plan.schema();
    let batches = block_on(async move {
        let ctx = SessionContext::new();
        collect(plan, ctx.task_ctx()).await
    })
    .map_err(|e| Error::io_error(format!("Plan execution failed: {}", e)))?;

    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        schema,
    ));
    let ffi_stream = FFI_ArrowArrayStream::new(reader);
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

//////////////////////////
// Sharding evaluation  //
//////////////////////////

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ShardingEvaluator_nativeEvaluate<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
    sharding_spec: JObject<'local>,
    source_id_to_column: JObject<'local>,
    stream_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_evaluate_sharding(
            &mut env,
            arrow_array_addr,
            arrow_schema_addr,
            sharding_spec,
            source_id_to_column,
            stream_addr,
        )
    );
}

fn inner_evaluate_sharding(
    env: &mut JNIEnv,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
    sharding_spec: JObject,
    source_id_to_column: JObject,
    stream_addr: jlong,
) -> Result<()> {
    let input = import_ffi_array(arrow_array_addr, arrow_schema_addr)?;
    let struct_array = input
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            Error::input_error(
                "ShardingEvaluator expects a VectorSchemaRoot struct array".to_string(),
            )
        })?;
    let input_batch = RecordBatch::from(struct_array.clone());
    let spec = sharding_spec_from_java(env, &sharding_spec)?;
    let source_id_to_column = i32_string_map_from_java(env, source_id_to_column)?;
    let result =
        evaluate_sharding_spec_with_source_columns(&input_batch, &spec, &source_id_to_column)?;
    let schema = result.schema();
    let reader: Box<dyn RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new([Ok(result)].into_iter(), schema));
    let ffi_stream = FFI_ArrowArrayStream::new(reader);
    unsafe { std::ptr::write_unaligned(stream_addr as *mut FFI_ArrowArrayStream, ffi_stream) }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_ExecutionPlan_releaseNativeExecutionPlan(
    mut env: JNIEnv,
    this: JObject,
    _handle: jlong,
) {
    ok_or_throw_without_return!(env, inner_release_execution_plan(&mut env, this));
}

fn inner_release_execution_plan(env: &mut JNIEnv, this: JObject) -> Result<()> {
    let _: BlockingExecutionPlan = unsafe { env.take_rust_field(&this, NATIVE_EXECUTION_PLAN) }?;
    Ok(())
}

/////////////////////////
// LsmPointLookupPlanner //
/////////////////////////

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmPointLookupPlanner_nativeCreate(
    mut env: JNIEnv,
    this: JObject,
    dataset: JObject,
    shard_snapshots: JObject,
    pk_columns: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_create_lookup_planner(&mut env, this, dataset, shard_snapshots, pk_columns)
    );
}

fn inner_create_lookup_planner(
    env: &mut JNIEnv,
    this: JObject,
    dataset: JObject,
    shard_snapshots: JObject,
    pk_columns: JObject,
) -> Result<()> {
    let snapshots = read_shard_snapshots(env, &shard_snapshots)?;
    let pk_columns = env.get_strings_opt(&pk_columns)?;
    let dataset = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET) }?;
        Arc::new(guard.inner.clone())
    };
    let pk_columns = match pk_columns {
        Some(cols) => cols,
        None => get_pk_columns(&dataset)?,
    };
    let base_schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let collector = LsmDataSourceCollector::new(dataset, snapshots);
    let planner = LsmPointLookupPlanner::new(collector, pk_columns.clone(), base_schema.clone());

    let blocking = BlockingLsmPointLookupPlanner {
        planner,
        dataset_schema: base_schema,
        pk_columns,
    };
    unsafe { env.set_rust_field(&this, NATIVE_LOOKUP_PLANNER, blocking) }?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmPointLookupPlanner_nativePlanLookup<'local>(
    mut env: JNIEnv<'local>,
    this: JObject<'local>,
    array_addr: jlong,
    schema_addr: jlong,
    columns: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_plan_lookup(&mut env, this, array_addr, schema_addr, columns)
    )
}

fn inner_plan_lookup<'local>(
    env: &mut JNIEnv<'local>,
    this: JObject<'local>,
    array_addr: jlong,
    schema_addr: jlong,
    columns: JObject<'local>,
) -> Result<JObject<'local>> {
    let pk_value = import_ffi_array(array_addr, schema_addr)?;
    let columns = env.get_strings_opt(&columns)?;

    let (plan, dataset_schema) = {
        let guard = unsafe {
            env.get_rust_field::<_, _, BlockingLsmPointLookupPlanner>(&this, NATIVE_LOOKUP_PLANNER)
        }?;
        let pk_values = scalar_values_from_pk_value(pk_value.as_ref(), &guard.pk_columns)?;
        let plan = block_on(guard.planner.plan_lookup(&pk_values, columns.as_deref()))?;
        (plan, guard.dataset_schema.clone())
    };
    attach_execution_plan(env, plan, dataset_schema)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmPointLookupPlanner_releaseNativeLookupPlanner(
    mut env: JNIEnv,
    this: JObject,
    _handle: jlong,
) {
    ok_or_throw_without_return!(env, inner_release_lookup_planner(&mut env, this));
}

fn inner_release_lookup_planner(env: &mut JNIEnv, this: JObject) -> Result<()> {
    let _: BlockingLsmPointLookupPlanner =
        unsafe { env.take_rust_field(&this, NATIVE_LOOKUP_PLANNER) }?;
    Ok(())
}

///////////////////////////
// LsmVectorSearchPlanner //
///////////////////////////

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmVectorSearchPlanner_nativeCreate(
    mut env: JNIEnv,
    this: JObject,
    dataset: JObject,
    shard_snapshots: JObject,
    vector_column: JString,
    pk_columns: JObject,
    distance_type: JObject,
    filter: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_create_vector_planner(
            &mut env,
            this,
            dataset,
            shard_snapshots,
            vector_column,
            pk_columns,
            distance_type,
            filter,
        )
    );
}

#[allow(clippy::too_many_arguments)]
fn inner_create_vector_planner(
    env: &mut JNIEnv,
    this: JObject,
    dataset: JObject,
    shard_snapshots: JObject,
    vector_column: JString,
    pk_columns: JObject,
    distance_type: JObject,
    filter: JObject,
) -> Result<()> {
    let snapshots = read_shard_snapshots(env, &shard_snapshots)?;
    let vector_column: String = env.get_string(&vector_column)?.into();
    let pk_columns = env.get_strings_opt(&pk_columns)?;
    let distance_type = env.get_string_opt(&distance_type)?;
    let filter = env.get_string_opt(&filter)?;
    let dataset = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&dataset, NATIVE_DATASET) }?;
        Arc::new(guard.inner.clone())
    };
    let pk_columns = match pk_columns {
        Some(cols) => cols,
        None => get_pk_columns(&dataset)?,
    };
    let base_schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let dist_type = parse_distance_type(distance_type.as_deref().unwrap_or("l2"))?;
    let vector_dim = get_vector_dim(&dataset, &vector_column)?;
    let filter = filter
        .as_deref()
        .map(|filter| parse_lsm_filter_expr(base_schema.as_ref(), filter).map_err(Error::from))
        .transpose()?;

    let collector = LsmDataSourceCollector::new(dataset.clone(), snapshots);
    let mut planner = LsmVectorSearchPlanner::new(
        collector,
        pk_columns,
        base_schema.clone(),
        vector_column,
        dist_type,
    )
    .with_dataset(dataset);
    if let Some(filter) = filter {
        planner = planner.with_filter(Some(filter));
    }

    let blocking = BlockingLsmVectorSearchPlanner {
        planner,
        vector_dim,
        dataset_schema: base_schema,
    };
    unsafe { env.set_rust_field(&this, NATIVE_VECTOR_PLANNER, blocking) }?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmVectorSearchPlanner_nativePlanSearch<'local>(
    mut env: JNIEnv<'local>,
    this: JObject<'local>,
    array_addr: jlong,
    schema_addr: jlong,
    k: jint,
    nprobes: jint,
    columns: JObject<'local>,
    refine_base_table: bool,
    overfetch_factor: jdouble,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_plan_search(
            &mut env,
            this,
            array_addr,
            schema_addr,
            k,
            nprobes,
            columns,
            refine_base_table,
            overfetch_factor,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_plan_search<'local>(
    env: &mut JNIEnv<'local>,
    this: JObject<'local>,
    array_addr: jlong,
    schema_addr: jlong,
    k: jint,
    nprobes: jint,
    columns: JObject<'local>,
    refine_base_table: bool,
    overfetch_factor: f64,
) -> Result<JObject<'local>> {
    let query = import_ffi_array(array_addr, schema_addr)?;
    let columns = env.get_strings_opt(&columns)?;
    let float32_array = query
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| Error::input_error("query must be a Float32 array".to_string()))?;

    let (plan, dataset_schema) = {
        let guard = unsafe {
            env.get_rust_field::<_, _, BlockingLsmVectorSearchPlanner>(&this, NATIVE_VECTOR_PLANNER)
        }?;
        if float32_array.len() != guard.vector_dim {
            return Err(Error::input_error(format!(
                "Query vector has {} dimensions, expected {}",
                float32_array.len(),
                guard.vector_dim
            )));
        }
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let fsl = FixedSizeListArray::try_new(
            field,
            guard.vector_dim as i32,
            Arc::new(float32_array.clone()),
            None,
        )?;
        let plan = block_on(guard.planner.plan_search(
            &fsl,
            k as usize,
            nprobes as usize,
            columns.as_deref(),
            refine_base_table,
            overfetch_factor,
        ))?;
        (plan, guard.dataset_schema.clone())
    };
    attach_execution_plan(env, plan, dataset_schema)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_memwal_LsmVectorSearchPlanner_releaseNativeVectorPlanner(
    mut env: JNIEnv,
    this: JObject,
    _handle: jlong,
) {
    ok_or_throw_without_return!(env, inner_release_vector_planner(&mut env, this));
}

fn inner_release_vector_planner(env: &mut JNIEnv, this: JObject) -> Result<()> {
    let _: BlockingLsmVectorSearchPlanner =
        unsafe { env.take_rust_field(&this, NATIVE_VECTOR_PLANNER) }?;
    Ok(())
}

///////////////////////////
// Dataset MemWAL methods //
///////////////////////////

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeInitializeMemWal(
    mut env: JNIEnv,
    jdataset: JObject,
    params: JObject,
) {
    ok_or_throw_without_return!(env, inner_initialize_mem_wal(&mut env, jdataset, params));
}

fn inner_initialize_mem_wal(env: &mut JNIEnv, jdataset: JObject, params: JObject) -> Result<()> {
    let maintained_list = env
        .call_method(&params, "maintainedIndexes", "()Ljava/util/List;", &[])?
        .l()?;
    let maintained_indexes = env.get_strings(&maintained_list)?;
    let bucket_column = env.get_optional_string_from_method(&params, "bucketColumn")?;
    let num_buckets = env.get_optional_u32_from_method(&params, "numBuckets")?;
    let identity_column = env.get_optional_string_from_method(&params, "identityColumn")?;
    let unsharded = env.call_method(&params, "unsharded", "()Z", &[])?.z()?;
    let writer_config =
        env.get_optional_from_method(&params, "writerConfigDefaults", |env, config_obj| {
            build_writer_config(env, &config_obj)
        })?;

    let bucket = match (bucket_column.as_deref(), num_buckets) {
        (Some(_), Some(_)) => true,
        (None, None) => false,
        _ => {
            return Err(Error::input_error(
                "bucket sharding requires both bucketColumn and numBuckets".to_string(),
            ));
        }
    };
    let modes = [bucket, identity_column.is_some(), unsharded]
        .into_iter()
        .filter(|&m| m)
        .count();
    if modes > 1 {
        return Err(Error::input_error(
            "at most one of bucket sharding, identityColumn, or unsharded may be set".to_string(),
        ));
    }

    let mut guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(&jdataset, NATIVE_DATASET) }?;
    let mut builder = guard.inner.initialize_mem_wal();
    if let (Some(column), Some(n)) = (bucket_column, num_buckets) {
        builder = builder.bucket_sharding(column, n);
    } else if let Some(column) = identity_column {
        builder = builder.identity_sharding(column);
    } else if unsharded {
        builder = builder.unsharded();
    }
    builder = builder.maintained_indexes(maintained_indexes);
    if let Some(config) = writer_config {
        builder = builder.writer_config_defaults(config);
    }
    block_on(builder.execute())?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeMemWalIndexDetails<'local>(
    mut env: JNIEnv<'local>,
    jdataset: JObject<'local>,
) -> JObject<'local> {
    ok_or_throw!(env, inner_mem_wal_index_details(&mut env, jdataset))
}

fn inner_mem_wal_index_details<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject<'local>,
) -> Result<JObject<'local>> {
    let details = {
        let guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&jdataset, NATIVE_DATASET) }?;
        block_on(guard.inner.mem_wal_index_details())?
    };
    match details {
        Some(details) => index_details_to_java(env, &details),
        None => Ok(JObject::null()),
    }
}

///////////////
// Helpers   //
///////////////

fn parse_uuid(shard_id: &str) -> Result<Uuid> {
    Uuid::parse_str(shard_id)
        .map_err(|e| Error::input_error(format!("Invalid shard_id UUID: {}", e)))
}

/// Read a `List<ShardSnapshot>` Java object into Rust `ShardSnapshot`s.
fn read_shard_snapshots(env: &mut JNIEnv, list_obj: &JObject) -> Result<Vec<ShardSnapshot>> {
    import_vec_to_rust(env, list_obj, |env, obj| {
        let shard_id = env.get_string_from_method(&obj, "shardId")?;
        let uuid = parse_uuid(&shard_id)?;
        let spec_id = env.get_u32_from_method(&obj, "specId")?;
        let current_generation = env.get_u64_from_method(&obj, "currentGeneration")?;
        let mut snapshot = ShardSnapshot::new(uuid)
            .with_spec_id(spec_id)
            .with_current_generation(current_generation);

        let sstable_list = env
            .call_method(&obj, "sstables", "()Ljava/util/List;", &[])?
            .l()?;
        for sstable in import_vec(env, &sstable_list)? {
            let generation = env.get_u64_from_method(&sstable, "generation")?;
            let path = env.get_string_from_method(&sstable, "path")?;
            snapshot = snapshot.with_sstable(generation, path);
        }
        Ok(snapshot)
    })
}

/// Reconstruct a [`ShardSnapshot`] from a writer's [`ShardManifest`].
fn shard_snapshot_from_manifest(manifest: ShardManifest) -> ShardSnapshot {
    ShardSnapshot {
        shard_id: manifest.shard_id,
        spec_id: manifest.shard_spec_id,
        current_generation: manifest.current_generation,
        sstables: manifest
            .sstables
            .into_iter()
            .map(|sstable| SsTable {
                generation: sstable.generation,
                path: sstable.path,
            })
            .collect(),
    }
}

/// Extract primary key column names from the dataset schema.
fn get_pk_columns(dataset: &LanceDataset) -> Result<Vec<String>> {
    let pk_fields = dataset.schema().unenforced_primary_key();
    if pk_fields.is_empty() {
        return Err(Error::input_error(
            "Dataset has no primary key. Set 'lance-schema:unenforced-primary-key' metadata \
             on the primary key field(s)."
                .to_string(),
        ));
    }
    Ok(pk_fields.iter().map(|f| f.name.clone()).collect())
}

/// Parse a distance type string into a [`DistanceType`].
fn parse_distance_type(s: &str) -> Result<DistanceType> {
    match s.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok(DistanceType::L2),
        "cosine" => Ok(DistanceType::Cosine),
        "dot" | "inner_product" => Ok(DistanceType::Dot),
        "hamming" => Ok(DistanceType::Hamming),
        _ => Err(Error::input_error(format!(
            "Unknown distanceType '{}'. Valid values: 'l2', 'cosine', 'dot', 'hamming'",
            s
        ))),
    }
}

/// Get the vector dimension of a `FixedSizeList` column from the dataset schema.
fn get_vector_dim(dataset: &LanceDataset, column: &str) -> Result<usize> {
    let schema = ArrowSchema::from(dataset.schema());
    let field = schema.field_with_name(column).map_err(|_| {
        Error::input_error(format!("Column '{}' not found in dataset schema", column))
    })?;
    match field.data_type() {
        DataType::FixedSizeList(_, size) => Ok(*size as usize),
        other => Err(Error::input_error(format!(
            "Column '{}' is not a FixedSizeList (got {:?}). \
             Vector columns must be FixedSizeList<float32>.",
            column, other
        ))),
    }
}

/// Convert a primary key Arrow array (single row) into DataFusion scalars.
fn scalar_values_from_pk_value(
    pk_value: &dyn Array,
    pk_columns: &[String],
) -> Result<Vec<ScalarValue>> {
    if pk_value.len() != 1 {
        return Err(Error::input_error(format!(
            "pkValue must contain exactly one row, got {}",
            pk_value.len()
        )));
    }

    if pk_columns.len() == 1 {
        let scalar = ScalarValue::try_from_array(pk_value, 0)
            .map_err(|e| Error::input_error(format!("Cannot convert pkValue: {}", e)))?;
        return Ok(vec![scalar]);
    }

    let struct_array = pk_value
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            Error::input_error(format!(
                "Composite primary key lookup requires a struct vector with one row and {} fields",
                pk_columns.len()
            ))
        })?;

    if struct_array.num_columns() != pk_columns.len() {
        return Err(Error::input_error(format!(
            "Composite primary key lookup expected {} fields, got {}",
            pk_columns.len(),
            struct_array.num_columns()
        )));
    }

    let mut pk_values = Vec::with_capacity(pk_columns.len());
    for column_name in pk_columns {
        let column = struct_array.column_by_name(column_name).ok_or_else(|| {
            Error::input_error(format!(
                "Composite primary key lookup requires field '{}' in pkValue",
                column_name
            ))
        })?;
        let scalar = ScalarValue::try_from_array(column.as_ref(), 0)
            .map_err(|e| Error::input_error(format!("Cannot convert composite pkValue: {}", e)))?;
        pk_values.push(scalar);
    }
    Ok(pk_values)
}

/// Import an Arrow array exported through the C Data Interface.
///
/// The FFI structs at the given addresses are moved out and replaced with empty
/// placeholders, so the producer's `close()` on the Java side is a safe no-op.
fn import_ffi_array(array_addr: jlong, schema_addr: jlong) -> Result<ArrayRef> {
    let ffi_array =
        unsafe { std::ptr::replace(array_addr as *mut FFI_ArrowArray, FFI_ArrowArray::empty()) };
    let ffi_schema = unsafe {
        std::ptr::replace(
            schema_addr as *mut FFI_ArrowSchema,
            FFI_ArrowSchema::empty(),
        )
    };
    let array_data = unsafe { from_ffi(ffi_array, &ffi_schema) }?;
    Ok(make_array(array_data))
}

/// Build a [`ShardWriterConfig`] from a Java `ShardWriterConfig` object.
fn build_writer_config(env: &mut JNIEnv, config: &JObject) -> Result<ShardWriterConfig> {
    let mut writer_config = ShardWriterConfig::default();
    if let Some(v) = read_optional_bool(env, config, "durableWrite")? {
        writer_config = writer_config.with_durable_write(v);
    }
    if let Some(v) = read_optional_u64(env, config, "maxWalBufferSize")? {
        writer_config = writer_config.with_max_wal_buffer_size(v as usize);
    }
    if let Some(v) = read_optional_u64(env, config, "maxWalFlushIntervalMs")? {
        writer_config = writer_config.with_max_wal_flush_interval(Duration::from_millis(v));
    }
    if let Some(v) = read_optional_u64(env, config, "maxMemtableSize")? {
        writer_config = writer_config.with_max_memtable_size(v as usize);
    }
    if let Some(v) = read_optional_u64(env, config, "maxMemtableRows")? {
        writer_config = writer_config.with_max_memtable_rows(v as usize);
    }
    if let Some(v) = read_optional_u64(env, config, "maxMemtableBatches")? {
        writer_config = writer_config.with_max_memtable_batches(v as usize);
    }
    if let Some(v) = read_optional_u64(env, config, "maxUnflushedMemtableBytes")? {
        writer_config = writer_config.with_max_unflushed_memtable_bytes(v as usize);
    }
    if let Some(v) = read_optional_u64(env, config, "manifestScanBatchSize")? {
        writer_config = writer_config.with_manifest_scan_batch_size(v as usize);
    }
    if let Some(v) = read_optional_u64(env, config, "backpressureLogIntervalMs")? {
        writer_config = writer_config.with_backpressure_log_interval(Duration::from_millis(v));
    }
    if let Some(v) = read_optional_u64(env, config, "statsLogIntervalMs")? {
        let interval = if v == 0 {
            None
        } else {
            Some(Duration::from_millis(v))
        };
        writer_config = writer_config.with_stats_log_interval(interval);
    }
    let hnsw_params = import_vec_from_method(
        env,
        config,
        "hnswParams",
        |env, item| -> Result<(String, HnswBuildParams)> {
            let index_name = env
                .call_method(&item, "indexName", "()Ljava/lang/String;", &[])?
                .l()?;
            let index_name: String = JString::from(index_name).extract(env)?;
            let num_edges = env.call_method(&item, "numEdges", "()I", &[])?.i()? as usize;
            let ef_construction =
                env.call_method(&item, "efConstruction", "()I", &[])?.i()? as usize;
            let max_level = env.call_method(&item, "maxLevel", "()I", &[])?.i()? as u16;
            Ok((
                index_name,
                HnswBuildParams::default()
                    .num_edges(num_edges)
                    .ef_construction(ef_construction)
                    .max_level(max_level),
            ))
        },
    )?;
    for (index_name, params) in hnsw_params {
        writer_config = writer_config.with_hnsw_params(index_name, params);
    }
    Ok(writer_config)
}

fn read_optional_bool(env: &mut JNIEnv, obj: &JObject, method: &str) -> Result<Option<bool>> {
    env.get_optional_from_method(obj, method, |env, value| {
        Ok(env.call_method(&value, "booleanValue", "()Z", &[])?.z()?)
    })
}

fn read_optional_u64(env: &mut JNIEnv, obj: &JObject, method: &str) -> Result<Option<u64>> {
    env.get_optional_from_method(obj, method, |env, value| {
        Ok(env.call_method(&value, "longValue", "()J", &[])?.j()? as u64)
    })
}

fn write_stats_to_java<'a>(
    env: &mut JNIEnv<'a>,
    stats: &WriteStatsSnapshot,
) -> Result<JObject<'a>> {
    Ok(env.new_object(
        "org/lance/memwal/WriteStats",
        "(JJJJJJJJ)V",
        &[
            JValueGen::Long(stats.put_count as i64),
            JValueGen::Long(stats.put_time.as_millis() as i64),
            JValueGen::Long(stats.wal_flush_count as i64),
            JValueGen::Long(stats.wal_flush_bytes as i64),
            JValueGen::Long(stats.wal_flush_time.as_millis() as i64),
            JValueGen::Long(stats.memtable_flush_count as i64),
            JValueGen::Long(stats.memtable_flush_rows as i64),
            JValueGen::Long(stats.memtable_flush_time.as_millis() as i64),
        ],
    )?)
}

fn memtable_stats_to_java<'a>(
    env: &mut JNIEnv<'a>,
    stats: &MemTableStats,
    memory: &ShardMemory,
) -> Result<JObject<'a>> {
    let max_buffered = box_u64_opt(env, stats.max_buffered_batch_position)?;
    let pending_start = box_u64_opt(env, stats.pending_wal_start_batch_position)?;
    let pending_end = box_u64_opt(env, stats.pending_wal_end_batch_position)?;
    Ok(env.new_object(
        "org/lance/memwal/MemTableStats",
        "(JJJJLjava/lang/Long;JJLjava/lang/Long;Ljava/lang/Long;JJJJJJ)V",
        &[
            JValueGen::Long(stats.row_count as i64),
            JValueGen::Long(stats.batch_count as i64),
            JValueGen::Long(memory.row_bytes() as i64),
            JValueGen::Long(stats.generation as i64),
            JValueGen::Object(&max_buffered),
            JValueGen::Long(stats.durable_batch_count as i64),
            JValueGen::Long(stats.global_offset as i64),
            JValueGen::Object(&pending_start),
            JValueGen::Object(&pending_end),
            JValueGen::Long(stats.pending_wal_batch_count as i64),
            JValueGen::Long(stats.pending_wal_row_count as i64),
            JValueGen::Long(stats.pending_wal_estimated_bytes as i64),
            JValueGen::Long(memory.index_bytes() as i64),
            JValueGen::Long(memory.grace_bytes() as i64),
            JValueGen::Long(memory.retained_bytes() as i64),
        ],
    )?)
}

fn box_u64_opt<'a>(env: &mut JNIEnv<'a>, value: Option<usize>) -> Result<JObject<'a>> {
    match value {
        Some(v) => Ok(env.new_object("java/lang/Long", "(J)V", &[JValueGen::Long(v as i64)])?),
        None => Ok(JObject::null()),
    }
}

fn index_details_to_java<'a>(
    env: &mut JNIEnv<'a>,
    details: &MemWalIndexDetails,
) -> Result<JObject<'a>> {
    let maintained_indexes = export_vec(env, &details.maintained_indexes)?;
    let writer_config_defaults = details.writer_config_defaults.clone().into_java(env)?;

    let sharding_specs = env.new_object("java/util/ArrayList", "()V", &[])?;
    for spec in &details.sharding_specs {
        let spec_obj = sharding_spec_to_java(env, spec)?;
        env.call_method(
            &sharding_specs,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValueGen::Object(&spec_obj)],
        )?;
    }

    Ok(env.new_object(
        "org/lance/memwal/MemWalIndexDetails",
        "(JLjava/util/List;Ljava/util/Map;Ljava/util/List;)V",
        &[
            JValueGen::Long(details.num_shards as i64),
            JValueGen::Object(&maintained_indexes),
            JValueGen::Object(&writer_config_defaults),
            JValueGen::Object(&sharding_specs),
        ],
    )?)
}

fn sharding_spec_to_java<'a>(env: &mut JNIEnv<'a>, spec: &ShardingSpec) -> Result<JObject<'a>> {
    let fields = env.new_object("java/util/ArrayList", "()V", &[])?;
    for field in &spec.fields {
        let field_obj = sharding_field_to_java(env, field)?;
        env.call_method(
            &fields,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValueGen::Object(&field_obj)],
        )?;
    }
    Ok(env.new_object(
        "org/lance/memwal/ShardingSpec",
        "(ILjava/util/List;)V",
        &[
            JValueGen::Int(spec.spec_id as i32),
            JValueGen::Object(&fields),
        ],
    )?)
}

fn sharding_field_to_java<'a>(env: &mut JNIEnv<'a>, field: &ShardingField) -> Result<JObject<'a>> {
    let field_id: JObject = env.new_string(&field.field_id)?.into();
    let source_ids = int_list_to_java(env, &field.source_ids)?;
    let transform: JObject = match &field.transform {
        Some(t) => env.new_string(t)?.into(),
        None => JObject::null(),
    };
    let expression: JObject = match &field.expression {
        Some(e) => env.new_string(e)?.into(),
        None => JObject::null(),
    };
    let result_type: JObject = env.new_string(&field.result_type)?.into();
    let parameters = field.parameters.clone().into_java(env)?;
    Ok(env.new_object(
        "org/lance/memwal/ShardingField",
        "(Ljava/lang/String;Ljava/util/List;Ljava/lang/String;Ljava/lang/String;\
         Ljava/lang/String;Ljava/util/Map;)V",
        &[
            JValueGen::Object(&field_id),
            JValueGen::Object(&source_ids),
            JValueGen::Object(&transform),
            JValueGen::Object(&expression),
            JValueGen::Object(&result_type),
            JValueGen::Object(&parameters),
        ],
    )?)
}

fn sharding_spec_from_java(env: &mut JNIEnv, spec: &JObject) -> Result<ShardingSpec> {
    if spec.is_null() {
        return Err(Error::input_error(
            "ShardingSpec must not be null".to_string(),
        ));
    }

    env.with_local_frame(32, |env| {
        let spec_id = env.call_method(spec, "specId", "()I", &[])?.i()? as u32;
        let fields_obj = env
            .call_method(spec, "fields", "()Ljava/util/List;", &[])?
            .l()?;
        let fields_list = env.get_list(&fields_obj)?;
        let mut iter = fields_list.iter(env)?;
        let mut fields = Vec::with_capacity(fields_list.size(env)? as usize);
        while let Some(field_obj) = iter.next(env)? {
            fields.push(sharding_field_from_java(env, &field_obj)?);
        }

        Ok::<_, Error>(ShardingSpec { spec_id, fields })
    })
}

fn sharding_field_from_java(env: &mut JNIEnv, field: &JObject) -> Result<ShardingField> {
    env.with_local_frame(32, |env| {
        let field_id = string_from_method(env, field, "fieldId")?;
        let source_ids_obj = env
            .call_method(field, "sourceIds", "()Ljava/util/List;", &[])?
            .l()?;
        let source_ids = env.get_integers(&source_ids_obj)?;
        let transform = env.get_optional_string_from_method(field, "transform")?;
        let expression = env.get_optional_string_from_method(field, "expression")?;
        let result_type = string_from_method(env, field, "resultType")?;
        let parameters_obj = env
            .call_method(field, "parameters", "()Ljava/util/Map;", &[])?
            .l()?;
        let parameters = if parameters_obj.is_null() {
            HashMap::new()
        } else {
            let parameters_map = JMap::from_env(env, &parameters_obj)?;
            to_rust_map(env, &parameters_map)?
        };

        Ok::<_, Error>(ShardingField {
            field_id,
            source_ids,
            transform,
            expression,
            result_type,
            parameters,
        })
    })
}

fn i32_string_map_from_java(env: &mut JNIEnv, map_obj: JObject) -> Result<HashMap<i32, String>> {
    if map_obj.is_null() {
        return Ok(HashMap::new());
    }

    env.with_local_frame(32, |env| {
        let jmap = JMap::from_env(env, &map_obj)?;
        let mut iter = jmap.iter(env)?;
        let mut map = HashMap::new();
        while let Some((key, value)) = iter.next(env)? {
            let key = env.call_method(&key, "intValue", "()I", &[])?.i()?;
            let value = JString::from(value);
            let value: String = env.get_string(&value)?.into();
            map.insert(key, value);
        }
        Ok::<_, Error>(map)
    })
}

fn string_from_method(env: &mut JNIEnv, obj: &JObject, method: &str) -> Result<String> {
    let value = env
        .call_method(obj, method, "()Ljava/lang/String;", &[])?
        .l()?;
    let value = JString::from(value);
    Ok(env.get_string(&value)?.into())
}

fn int_list_to_java<'a>(env: &mut JNIEnv<'a>, ints: &[i32]) -> Result<JObject<'a>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for &value in ints {
        let integer = env.new_object("java/lang/Integer", "(I)V", &[JValueGen::Int(value)])?;
        env.call_method(
            &list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValueGen::Object(&integer)],
        )?;
    }
    Ok(list)
}
