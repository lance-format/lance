// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `LANCE_CPU_THREADS` is read once per process, so this check lives in its own
//! test binary and sets the variable before anything reads it.

use lance_datafusion_exec::exec::{LanceExecutionOptions, new_session_config};

#[test]
fn default_target_partitions_follow_lance_cpu_threads() {
    let cores = std::thread::available_parallelism().unwrap().get();
    let cpu_threads = cores * 4;
    // SAFETY: the only test in this binary; nothing else touches the environment.
    unsafe { std::env::set_var("LANCE_CPU_THREADS", cpu_threads.to_string()) };

    // An oversubscribed CPU pool raises the default, so exec nodes that cap their
    // concurrency at target_partitions (IVF partition loads) are not held to the core count.
    let default = new_session_config(&LanceExecutionOptions::default());
    assert_eq!(default.target_partitions(), cpu_threads);

    // A caller that sets target_partition explicitly still caps the session.
    let explicit = new_session_config(&LanceExecutionOptions {
        target_partition: Some(2),
        ..Default::default()
    });
    assert_eq!(explicit.target_partitions(), 2);
}
