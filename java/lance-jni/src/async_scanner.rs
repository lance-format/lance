// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET};
use crate::blocking_scanner::{ScannerOptions, build_scanner_with_options};
use crate::dispatcher::{DISPATCHER, DispatcherMessage};
use crate::error::Result;
use crate::task_tracker::{TASK_TRACKER, TaskInfo};
use crate::{RT, block_on};
use arrow::ffi::FFI_ArrowSchema;
use jni::JNIEnv;
use jni::objects::JObject;
use jni::sys::{jboolean, jint, jlong};
use lance::dataset::scanner::Scanner;
use lance_io::ffi::to_ffi_arrow_array_stream;

pub const NATIVE_ASYNC_SCANNER: &str = "nativeAsyncScannerHandle";

/// Async scanner that spawns Tokio tasks for non-blocking I/O
pub struct AsyncScanner {
    pub(crate) inner: Arc<Scanner>,
}

/// RAII guard that ensures task cleanup even on panic or early return
///
/// This guard prevents memory leaks in the task tracker by guaranteeing
/// that task_id is removed from the HashMap when the guard is dropped,
/// regardless of how the async task terminates (normal completion, panic,
/// or cancellation).
struct TaskCleanupGuard {
    task_id: u64,
}

impl TaskCleanupGuard {
    fn new(task_id: u64) -> Self {
        Self { task_id }
    }
}

impl Drop for TaskCleanupGuard {
    fn drop(&mut self) {
        // GUARANTEED to run when guard goes out of scope
        // Works even if the task panics or returns early
        //
        // Note: We spawn a detached task instead of using block_on()
        // because Drop may be called from within a tokio runtime context
        let task_id = self.task_id;
        RT.spawn(async move {
            TASK_TRACKER.complete(task_id).await;
            log::debug!("Task {} cleaned up via RAII guard", task_id);
        });
    }
}

impl AsyncScanner {
    pub fn create(scanner: Scanner) -> Self {
        Self {
            inner: Arc::new(scanner),
        }
    }

    /// Start an async scan task (static method to avoid holding locks)
    pub fn start_scan_with_scanner(
        scanner: Arc<Scanner>,
        task_id: u64,
        scanner_global_ref: jni::objects::GlobalRef,
    ) {
        // Two-phase registration to prevent race condition:
        // 1. Pre-register with placeholder handle BEFORE spawning
        // 2. Spawn the actual task
        // 3. Update registration with real handle
        // This ensures task is registered before cleanup can run

        // Clone for the spawned task
        let global_ref_for_task = scanner_global_ref.clone();

        // Step 1: Pre-register with placeholder handle
        let placeholder_handle = RT.spawn(async {
            // Placeholder task that does nothing
            // Will be aborted when real handle is registered
        });

        block_on(async {
            TASK_TRACKER
                .register(
                    task_id,
                    TaskInfo {
                        scanner_global_ref: scanner_global_ref.clone(),
                        cancel_handle: placeholder_handle,
                    },
                )
                .await;
        });

        // Step 2: Spawn the actual task
        let handle = RT.spawn(async move {
            // RAII guard ensures cleanup on normal exit, panic, or cancellation
            let _cleanup_guard = TaskCleanupGuard::new(task_id);

            let result = match scanner.try_into_stream().await {
                Ok(stream) => {
                    // Convert to FFI pointer
                    match to_ffi_arrow_array_stream(stream, RT.handle().clone()) {
                        Ok(ffi_stream) => {
                            let ptr = Box::into_raw(Box::new(ffi_stream)) as i64;
                            Ok(ptr)
                        }
                        Err(e) => Err(e.to_string()),
                    }
                }
                Err(e) => Err(e.to_string()),
            };

            // Send result to dispatcher for Java completion
            let dispatcher = match DISPATCHER.get() {
                Some(d) => d,
                None => {
                    log::error!(
                        "Dispatcher not initialized - cannot complete task {}. \
                         This indicates a critical initialization failure.",
                        task_id
                    );
                    // Clean up the FFI stream pointer to prevent memory leak
                    if let Ok(ptr) = result {
                        unsafe {
                            drop(Box::from_raw(
                                ptr as *mut arrow::ffi_stream::FFI_ArrowArrayStream,
                            ));
                        }
                        log::debug!("Cleaned up FFI stream pointer for task {}", task_id);
                    }
                    return;
                }
            };

            // Save the pointer before sending so we can clean up on failure
            let result_ptr = result.as_ref().ok().copied();

            if let Err(e) = dispatcher.send(DispatcherMessage {
                scanner_global_ref: global_ref_for_task,
                task_id,
                result,
            }) {
                log::error!(
                    "Failed to send completion message for task {}: {}",
                    task_id,
                    e
                );
                // Clean up the FFI stream pointer to prevent memory leak
                if let Some(ptr) = result_ptr {
                    unsafe {
                        drop(Box::from_raw(
                            ptr as *mut arrow::ffi_stream::FFI_ArrowArrayStream,
                        ));
                    }
                    log::debug!("Cleaned up FFI stream pointer for task {}", task_id);
                }
            }

            // _cleanup_guard.drop() called here automatically, removing task from tracker
        });

        // Step 3: Update registration with real handle
        block_on(async {
            TASK_TRACKER.update_handle(task_id, handle).await;
        });
    }
}

// JNI Exports

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_createAsyncScanner<'local>(
    mut env: JNIEnv<'local>,
    _class: JObject<'local>,
    jdataset: JObject<'local>,
    fragment_ids_obj: JObject<'local>,
    columns_obj: JObject<'local>,
    substrait_filter_obj: JObject<'local>,
    filter_obj: JObject<'local>,
    batch_size_obj: JObject<'local>,
    batch_size_bytes_obj: JObject<'local>,
    io_buffer_size_obj: JObject<'local>,
    limit_obj: JObject<'local>,
    offset_obj: JObject<'local>,
    query_obj: JObject<'local>,
    fts_query_obj: JObject<'local>,
    prefilter: jboolean,
    with_row_id: jboolean,
    with_row_address: jboolean,
    batch_readahead: jint,
    fragment_readahead_obj: JObject<'local>,
    scan_in_order: jboolean,
    late_materialization_obj: JObject<'local>,
    column_orderings: JObject<'local>,
    use_scalar_index: jboolean,
    fast_search: jboolean,
    substrait_aggregate_obj: JObject<'local>,
    include_deleted_rows: jboolean,
    strict_batch_size: jboolean,
    disable_scoring_autoprojection: jboolean,
) -> JObject<'local> {
    crate::ok_or_throw!(
        env,
        inner_create_async_scanner(
            &mut env,
            jdataset,
            fragment_ids_obj,
            columns_obj,
            substrait_filter_obj,
            filter_obj,
            batch_size_obj,
            batch_size_bytes_obj,
            io_buffer_size_obj,
            limit_obj,
            offset_obj,
            query_obj,
            fts_query_obj,
            prefilter,
            with_row_id,
            with_row_address,
            batch_readahead,
            fragment_readahead_obj,
            scan_in_order,
            late_materialization_obj,
            column_orderings,
            use_scalar_index,
            fast_search,
            substrait_aggregate_obj,
            include_deleted_rows,
            strict_batch_size,
            disable_scoring_autoprojection,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_create_async_scanner<'local>(
    env: &mut JNIEnv<'local>,
    jdataset: JObject<'local>,
    fragment_ids_obj: JObject<'local>,
    columns_obj: JObject<'local>,
    substrait_filter_obj: JObject<'local>,
    filter_obj: JObject<'local>,
    batch_size_obj: JObject<'local>,
    batch_size_bytes_obj: JObject<'local>,
    io_buffer_size_obj: JObject<'local>,
    limit_obj: JObject<'local>,
    offset_obj: JObject<'local>,
    query_obj: JObject<'local>,
    fts_query_obj: JObject<'local>,
    prefilter: jboolean,
    with_row_id: jboolean,
    with_row_address: jboolean,
    batch_readahead: jint,
    fragment_readahead_obj: JObject<'local>,
    scan_in_order: jboolean,
    late_materialization_obj: JObject<'local>,
    column_orderings: JObject<'local>,
    use_scalar_index: jboolean,
    fast_search: jboolean,
    substrait_aggregate_obj: JObject<'local>,
    include_deleted_rows: jboolean,
    strict_batch_size: jboolean,
    disable_scoring_autoprojection: jboolean,
) -> Result<JObject<'local>> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(jdataset, NATIVE_DATASET) }?;
    let dataset = dataset_guard.inner.clone();
    drop(dataset_guard);

    let options = ScannerOptions {
        fragment_ids_obj,
        columns_obj,
        substrait_filter_obj,
        filter_obj,
        batch_size_obj,
        batch_size_bytes_obj,
        io_buffer_size_obj,
        limit_obj,
        offset_obj,
        query_obj,
        fts_query_obj,
        prefilter,
        with_row_id,
        with_row_address,
        batch_readahead,
        fragment_readahead_obj,
        scan_in_order,
        late_materialization_obj,
        column_orderings,
        use_scalar_index,
        fast_search,
        substrait_aggregate_obj,
        include_deleted_rows,
        strict_batch_size,
        disable_scoring_autoprojection,
    };

    let scanner = build_scanner_with_options(env, &dataset, options)?;

    let async_scanner = AsyncScanner::create(scanner);

    // Create Java AsyncScanner object
    let j_scanner = env.new_object("org/lance/ipc/AsyncScanner", "()V", &[])?;

    // Attach native handle
    unsafe { env.set_rust_field(&j_scanner, NATIVE_ASYNC_SCANNER, async_scanner)? };

    Ok(j_scanner)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_nativeStartScan(
    mut env: JNIEnv,
    j_scanner: JObject,
    task_id: jlong,
) {
    ok_or_throw_without_return!(env, inner_start_scan(&mut env, j_scanner, task_id as u64));
}

fn inner_start_scan(env: &mut JNIEnv, j_scanner: JObject, task_id: u64) -> Result<()> {
    // Create global reference first, before borrowing scanner
    let scanner_global_ref = env.new_global_ref(&j_scanner)?;

    // Clone the Arc<Scanner> and drop the MutexGuard before calling start_scan,
    // which does block_on internally. Holding the guard across block_on risks deadlock.
    let scanner = {
        let guard =
            unsafe { env.get_rust_field::<_, _, AsyncScanner>(&j_scanner, NATIVE_ASYNC_SCANNER)? };
        guard.inner.clone()
    };

    AsyncScanner::start_scan_with_scanner(scanner, task_id, scanner_global_ref);
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_nativeCancelTask(
    _env: JNIEnv,
    _j_scanner: JObject,
    task_id: jlong,
) {
    block_on(async {
        TASK_TRACKER.cancel(task_id as u64).await;
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_releaseNativeScanner(
    mut env: JNIEnv,
    j_scanner: JObject,
) {
    ok_or_throw_without_return!(env, inner_release_async_scanner(&mut env, j_scanner));
}

fn inner_release_async_scanner(env: &mut JNIEnv, j_scanner: JObject) -> Result<()> {
    let _: AsyncScanner = unsafe { env.take_rust_field(j_scanner, NATIVE_ASYNC_SCANNER) }?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_ipc_AsyncScanner_importFfiSchema(
    mut env: JNIEnv,
    j_scanner: JObject,
    schema_addr: jlong,
) {
    ok_or_throw_without_return!(
        env,
        inner_import_async_ffi_schema(&mut env, j_scanner, schema_addr)
    );
}

fn inner_import_async_ffi_schema(
    env: &mut JNIEnv,
    j_scanner: JObject,
    schema_addr: jlong,
) -> Result<()> {
    let scanner_guard =
        unsafe { env.get_rust_field::<_, _, AsyncScanner>(j_scanner, NATIVE_ASYNC_SCANNER)? };

    let schema = block_on(scanner_guard.inner.schema())?;
    let ffi_schema = FFI_ArrowSchema::try_from(&*schema)?;
    unsafe { std::ptr::write_unaligned(schema_addr as *mut FFI_ArrowSchema, ffi_schema) }
    Ok(())
}
