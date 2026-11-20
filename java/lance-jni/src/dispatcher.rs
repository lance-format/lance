// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use jni::JavaVM;
use jni::objects::GlobalRef;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;

/// Message sent from Tokio tasks to the dispatcher thread
pub struct DispatcherMessage {
    pub scanner_global_ref: GlobalRef,
    pub task_id: u64,
    pub result: Result<i64, String>, // Ok(stream_ptr) or Err(error_msg)
}

/// Global dispatcher instance initialized in JNI_OnLoad
pub static DISPATCHER: OnceLock<Arc<Dispatcher>> = OnceLock::new();

/// Dispatcher manages a persistent JNI thread for completing Java futures
#[derive(Debug)]
pub struct Dispatcher {
    tx: mpsc::UnboundedSender<DispatcherMessage>,
}

impl Dispatcher {
    /// Initialize the dispatcher with a persistent JNI thread.
    ///
    /// `async_scanner_class` must be a `GlobalRef` to the `AsyncScanner`
    /// Java class, resolved on a thread that has the correct application
    /// classloader (typically the `JNI_OnLoad` thread).
    pub fn initialize(jvm: Arc<JavaVM>, async_scanner_class: GlobalRef) -> Arc<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel::<DispatcherMessage>();

        // Spawn persistent dispatcher thread
        std::thread::Builder::new()
            .name("lance-jni-dispatcher".to_string())
            .spawn(move || {
                // Attach as daemon thread so it won't prevent JVM from exiting.
                // The thread stays attached for the entire lifetime (never detach),
                // but being a daemon thread means JVM won't wait for it on shutdown.
                let mut env = jvm
                    .attach_current_thread_as_daemon()
                    .expect("Failed to attach dispatcher to JVM");

                log::info!("JNI dispatcher thread started");

                // Cache method IDs for completeTask and failTask.
                // Use the pre-resolved GlobalRef instead of find_class to
                // avoid classloader issues on this native thread.
                let complete_method = env
                    .get_method_id(&async_scanner_class, "completeTask", "(JJ)Z")
                    .expect("completeTask method not found");
                let fail_method = env
                    .get_method_id(&async_scanner_class, "failTask", "(JLjava/lang/String;)V")
                    .expect("failTask method not found");

                // Event loop: block waiting for completions
                while let Some(msg) = rx.blocking_recv() {
                    let scanner_obj = msg.scanner_global_ref.as_obj();

                    match msg.result {
                        Err(error) => {
                            handle_error(&mut env, scanner_obj, fail_method, msg.task_id, &error)
                        }
                        Ok(result_ptr) => handle_success(
                            &mut env,
                            scanner_obj,
                            complete_method,
                            msg.task_id,
                            result_ptr,
                        ),
                    }
                }

                log::info!("JNI dispatcher thread shutting down");
            })
            .expect("Failed to spawn dispatcher thread");

        Arc::new(Self { tx })
    }

    /// Send a completion message to the dispatcher
    pub fn send(&self, msg: DispatcherMessage) -> std::result::Result<(), String> {
        self.tx
            .send(msg)
            .map_err(|e| format!("Failed to send message to dispatcher: {}", e))
    }
}

/// Handle error completion by calling failTask on Java side
fn handle_error(
    env: &mut jni::JNIEnv,
    scanner_obj: &jni::objects::JObject,
    fail_method: jni::objects::JMethodID,
    task_id: u64,
    error: &str,
) {
    let error_jstr = match env.new_string(error) {
        Ok(s) => s,
        Err(e) => {
            log::error!("Failed to create JString for error: {:?}", e);
            let _ = env.exception_clear();
            return;
        }
    };

    let result = unsafe {
        env.call_method_unchecked(
            scanner_obj,
            fail_method,
            jni::signature::ReturnType::Primitive(jni::signature::Primitive::Void),
            &[
                jni::sys::jvalue { j: task_id as i64 },
                jni::sys::jvalue {
                    l: error_jstr.as_raw(),
                },
            ],
        )
    };

    if let Err(e) = result {
        log::error!("Failed to call failTask: {:?}", e);
        // Clear any pending JNI exception to protect the dispatcher loop
        let _ = env.exception_clear();
    }
}

/// Handle success completion by calling completeTask on Java side
fn handle_success(
    env: &mut jni::JNIEnv,
    scanner_obj: &jni::objects::JObject,
    complete_method: jni::objects::JMethodID,
    task_id: u64,
    result_ptr: i64,
) {
    let result = unsafe {
        env.call_method_unchecked(
            scanner_obj,
            complete_method,
            jni::signature::ReturnType::Primitive(jni::signature::Primitive::Boolean),
            &[
                jni::sys::jvalue { j: task_id as i64 },
                jni::sys::jvalue { j: result_ptr },
            ],
        )
    };

    if !matches!(result, Ok(jni::objects::JValueOwned::Bool(1))) {
        log::debug!("completeTask rejected result: {:?}", result);
        // Clear any pending JNI exception to protect the dispatcher loop
        let _ = env.exception_clear();
    }
    // Arrow Java moves accepted streams into its own allocation and marks this
    // allocation released. Dropping it frees the shell; rejected streams also
    // release their producer here.
    unsafe {
        drop(Box::from_raw(
            result_ptr as *mut arrow::ffi_stream::FFI_ArrowArrayStream,
        ));
    }
}
