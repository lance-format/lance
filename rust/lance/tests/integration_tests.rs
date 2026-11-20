// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// NOTE: we only create one integration test binary, to keep compilation overhead down.

mod clone_commit;
mod count_pushdown;
mod fts_plan_reconstruction;
mod json_write;
mod label_list_optimize;
mod mem_wal;
mod optimize_concurrency;
mod prefilter_union;
#[cfg(feature = "slow_tests")]
mod query;
mod topk_late_materialization;
#[cfg(feature = "slow_tests")]
mod utils;
