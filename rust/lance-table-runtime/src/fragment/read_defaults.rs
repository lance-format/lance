// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared defaults for reading table fragments.

use std::sync::LazyLock;

use lance_core::{Error, Result};

/// Rows per output batch when neither the scan options nor
/// `LANCE_DEFAULT_BATCH_SIZE` specify one.
pub const BATCH_SIZE_FALLBACK: usize = 8192;

pub fn validate_batch_size(batch_size: usize) -> Result<u32> {
    let validated = u32::try_from(batch_size).map_err(|_| {
        Error::invalid_input(format!(
            "batch_size must be between 1 and {}, got {batch_size}",
            u32::MAX
        ))
    })?;
    if validated == 0 {
        return Err(Error::invalid_input(format!(
            "batch_size must be between 1 and {}, got {batch_size}",
            u32::MAX
        )));
    }
    Ok(validated)
}

/// Parse an environment variable as a specific type, logging a warning on parse failure.
pub fn parse_env_var<T: std::str::FromStr>(env_var_name: &str, default_val: &str) -> Option<T>
where
    T::Err: std::fmt::Display,
{
    std::env::var(env_var_name)
        .ok()
        .and_then(|val| match val.parse() {
            Ok(value) => Some(value),
            Err(e) => {
                log::warn!(
                    "Failed to parse the environment variable {}='{}': {}, the default value is: {}.",
                    env_var_name,
                    val,
                    e,
                    default_val
                );
                None
            }
        })
}

// For backwards compatibility / historical reasons we re-calculate the default batch size
// on each call
pub fn get_default_batch_size() -> Option<usize> {
    parse_env_var("LANCE_DEFAULT_BATCH_SIZE", &BATCH_SIZE_FALLBACK.to_string())
}

pub const LEGACY_DEFAULT_FRAGMENT_READAHEAD: usize = 4;

pub static DEFAULT_FRAGMENT_READAHEAD: LazyLock<Option<usize>> = LazyLock::new(|| {
    parse_env_var(
        "LANCE_DEFAULT_FRAGMENT_READAHEAD",
        &LEGACY_DEFAULT_FRAGMENT_READAHEAD.to_string(),
    )
});

// We want to support ~256 concurrent reads to maximize throughput on cloud storage systems
// Our typical page size is 8MiB (though not all reads are this large yet due to offset buffers, validity buffers, etc.)
// So we want to support 256 * 8MiB ~= 2GiB of queued reads
const DEFAULT_IO_BUFFER_SIZE_VALUE: u64 = 2 * 1024 * 1024 * 1024;

pub static DEFAULT_IO_BUFFER_SIZE: LazyLock<u64> = LazyLock::new(|| {
    parse_env_var(
        "LANCE_DEFAULT_IO_BUFFER_SIZE",
        &DEFAULT_IO_BUFFER_SIZE_VALUE.to_string(),
    )
    .unwrap_or(DEFAULT_IO_BUFFER_SIZE_VALUE)
});

/// The user-set value of `LANCE_DEFAULT_IO_BUFFER_SIZE`, or `None` if the env var
/// is unset or unparsable. Consult this from paths that have a sensible non-fixed
/// default (e.g. `SchedulerConfig::max_bandwidth`) so the env var still takes
/// precedence over that default. Re-reads the env var on each call so tests can
/// mutate it.
pub fn get_default_io_buffer_size_override() -> Option<u64> {
    parse_env_var(
        "LANCE_DEFAULT_IO_BUFFER_SIZE",
        &DEFAULT_IO_BUFFER_SIZE_VALUE.to_string(),
    )
}
