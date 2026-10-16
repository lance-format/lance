// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark-only controls for fragment column updates.

use arrow_array::RecordBatchReader;
use lance_datafusion::exec::LanceExecutionOptions;

use super::UpdateJoinOptions;
use super::fragment::{FileFragment, FragmentUpdateColumnsResult};
use super::update_join::{UpdateColumnsOptions, update_columns_with_options};
use crate::Result;

/// Execution controls used only by the update-join benchmark target.
#[derive(Clone, Debug, Default)]
pub struct UpdateJoinBenchmarkOptions {
    /// Public join selection and resource controls.
    pub join_options: UpdateJoinOptions,
    /// DataFusion execution and spill configuration.
    pub execution_options: LanceExecutionOptions,
    /// Override the external update execution batch size.
    pub execution_batch_size: Option<usize>,
    /// Override the memory reserved for each external-sort merge.
    pub sort_spill_reservation_bytes: Option<usize>,
}

/// Updates a fragment using benchmark-only algorithm and memory controls.
pub async fn update_columns(
    fragment: &mut FileFragment,
    right_reader: impl RecordBatchReader + Send + 'static,
    left_on: &str,
    right_on: &str,
    benchmark_options: UpdateJoinBenchmarkOptions,
) -> Result<FragmentUpdateColumnsResult> {
    benchmark_options.join_options.validate()?;
    let mut options = UpdateColumnsOptions::from(benchmark_options.join_options);
    options.execution_options = benchmark_options.execution_options;
    if let Some(execution_batch_size) = benchmark_options.execution_batch_size {
        options.execution_batch_size = execution_batch_size;
    }
    if let Some(sort_spill_reservation_bytes) = benchmark_options.sort_spill_reservation_bytes {
        options.sort_spill_reservation_bytes = Some(sort_spill_reservation_bytes);
    }
    update_columns_with_options(fragment, Box::new(right_reader), left_on, right_on, options).await
}
