// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Configuration for fragment update joins.

use crate::{Error, Result};

const DEFAULT_MAX_HASH_ROWS: usize = 250_000;
const DEFAULT_MAX_HASH_BYTES: usize = 1024 * 1024 * 1024;

/// Selects the algorithm used to join fragment rows with an update stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpdateJoinStrategy {
    /// Choose between hash and sort-merge using the configured row and byte thresholds.
    #[default]
    Auto,
    /// Always use the in-memory hash join.
    ///
    /// This strategy is not governed by the external execution memory pool and can use
    /// substantially more memory than [`UpdateJoinOptions::external_memory_pool_bytes`].
    Hash,
    /// Always use the spillable external sort-merge join for non-empty updates.
    SortMerge,
}

/// Controls algorithm selection and external resources for a fragment update join.
///
/// The existing fragment update methods use [`Default`] values. Use
/// [`FileFragment::update_columns_with_options`](crate::dataset::fragment::FileFragment::update_columns_with_options)
/// to configure one operation without changing the process-wide Lance environment.
/// The default Auto policy keeps the hash path eligible through 250,000 RHS rows and a
/// 1 GiB estimated RHS allocation.
///
/// ```
/// use lance::dataset::{UpdateJoinOptions, UpdateJoinStrategy};
///
/// let options = UpdateJoinOptions::default()
///     .with_strategy(UpdateJoinStrategy::Auto)
///     .with_hash_thresholds(500_000, 1024 * 1024 * 1024)
///     .with_external_memory_pool_bytes(256 * 1024 * 1024)
///     .with_max_temp_directory_bytes(20 * 1024 * 1024 * 1024);
///
/// assert_eq!(options.strategy(), UpdateJoinStrategy::Auto);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpdateJoinOptions {
    strategy: UpdateJoinStrategy,
    max_hash_rows: usize,
    max_hash_bytes: usize,
    external_memory_pool_bytes: Option<u64>,
    max_temp_directory_bytes: Option<u64>,
}

impl Default for UpdateJoinOptions {
    fn default() -> Self {
        Self {
            strategy: UpdateJoinStrategy::Auto,
            max_hash_rows: DEFAULT_MAX_HASH_ROWS,
            max_hash_bytes: DEFAULT_MAX_HASH_BYTES,
            external_memory_pool_bytes: None,
            max_temp_directory_bytes: None,
        }
    }
}

impl UpdateJoinOptions {
    /// Uses `strategy` for this update join.
    pub fn with_strategy(mut self, strategy: UpdateJoinStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the largest RHS row count and estimated allocation eligible for [`UpdateJoinStrategy::Hash`]
    /// when [`UpdateJoinStrategy::Auto`] is selected.
    ///
    /// Auto selects sort-merge as soon as either limit is exceeded. These limits do not cap
    /// total process memory or the memory used when hash is explicitly forced.
    pub fn with_hash_thresholds(mut self, max_rows: usize, max_bytes: usize) -> Self {
        self.max_hash_rows = max_rows;
        self.max_hash_bytes = max_bytes;
        self
    }

    /// Sets the DataFusion memory pool for the external sort-merge plan, in bytes.
    ///
    /// This bounds memory registered by DataFusion operators, not total process RSS. If unset,
    /// the update uses `LANCE_MEM_POOL_SIZE` or a default of 256 MiB per execution partition.
    pub fn with_external_memory_pool_bytes(mut self, bytes: u64) -> Self {
        self.external_memory_pool_bytes = Some(bytes);
        self
    }

    /// Sets the maximum temporary spill-directory usage for the external plan, in bytes.
    ///
    /// If unset, Lance uses `LANCE_MAX_TEMP_DIRECTORY_SIZE` or its default of 100 GiB.
    pub fn with_max_temp_directory_bytes(mut self, bytes: u64) -> Self {
        self.max_temp_directory_bytes = Some(bytes);
        self
    }

    /// Returns the configured algorithm-selection strategy.
    pub fn strategy(&self) -> UpdateJoinStrategy {
        self.strategy
    }

    /// Returns the maximum RHS row count eligible for Auto's hash path.
    pub fn max_hash_rows(&self) -> usize {
        self.max_hash_rows
    }

    /// Returns the maximum estimated RHS allocation eligible for Auto's hash path.
    pub fn max_hash_bytes(&self) -> usize {
        self.max_hash_bytes
    }

    /// Returns the explicit external memory pool, or `None` when Lance should resolve its default.
    pub fn external_memory_pool_bytes(&self) -> Option<u64> {
        self.external_memory_pool_bytes
    }

    /// Returns the explicit temporary-directory limit, or `None` when Lance should resolve its default.
    pub fn max_temp_directory_bytes(&self) -> Option<u64> {
        self.max_temp_directory_bytes
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.external_memory_pool_bytes == Some(0) {
            return Err(Error::invalid_input(
                "UpdateJoinOptions.external_memory_pool_bytes must be greater than zero, got 0"
                    .to_string(),
            ));
        }
        if self.max_temp_directory_bytes == Some(0) {
            return Err(Error::invalid_input(
                "UpdateJoinOptions.max_temp_directory_bytes must be greater than zero, got 0"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_and_builders() {
        let defaults = UpdateJoinOptions::default();
        assert_eq!(defaults.strategy(), UpdateJoinStrategy::Auto);
        assert_eq!(defaults.max_hash_rows(), 250_000);
        assert_eq!(defaults.max_hash_bytes(), 1024 * 1024 * 1024);
        assert_eq!(defaults.external_memory_pool_bytes(), None);
        assert_eq!(defaults.max_temp_directory_bytes(), None);

        let configured = defaults
            .with_strategy(UpdateJoinStrategy::SortMerge)
            .with_hash_thresholds(10, 20)
            .with_external_memory_pool_bytes(30)
            .with_max_temp_directory_bytes(40);
        assert_eq!(configured.strategy(), UpdateJoinStrategy::SortMerge);
        assert_eq!(configured.max_hash_rows(), 10);
        assert_eq!(configured.max_hash_bytes(), 20);
        assert_eq!(configured.external_memory_pool_bytes(), Some(30));
        assert_eq!(configured.max_temp_directory_bytes(), Some(40));
        configured.validate().unwrap();
    }

    #[test]
    fn test_rejects_zero_external_limits() {
        let error = UpdateJoinOptions::default()
            .with_external_memory_pool_bytes(0)
            .validate()
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("external_memory_pool_bytes"));

        let error = UpdateJoinOptions::default()
            .with_max_temp_directory_bytes(0)
            .validate()
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("max_temp_directory_bytes"));
    }
}
