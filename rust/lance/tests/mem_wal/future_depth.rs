// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compile-time canary for issue #8416.
//!
//! A downstream crate awaiting the mem_wal point-lookup futures must prove
//! their auto traits within rustc's default `recursion_limit` of 128. This
//! integration-test crate deliberately carries no `recursion_limit`
//! attribute, so the `Send` obligations below are solved exactly as a
//! consumer's would be, and the nested frames mirror the downstream call
//! shape reported in the issue. A depth regression in the scanner's future
//! nesting fails this crate's build instead of every consumer's.

use std::future::Future;
use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use lance::Result;
use lance::dataset::mem_wal::scanner::{LsmPointLookupPlanner, LsmScanner};

fn require_send<F: Future + Send>(future: F) -> F {
    future
}

async fn lookup_one(
    planner: &LsmPointLookupPlanner,
    keys: &[ScalarValue],
) -> Result<Option<RecordBatch>> {
    planner.lookup(keys, None).await
}

async fn lookup_batch(
    planner: &LsmPointLookupPlanner,
    keys: &[ScalarValue],
) -> Result<RecordBatch> {
    planner.lookup_many(keys, None).await
}

async fn lookup_plan(
    planner: &LsmPointLookupPlanner,
    keys: &[ScalarValue],
) -> Result<Arc<dyn ExecutionPlan>> {
    planner.plan_point_lookup(keys, None).await
}

async fn scan_plan(scanner: &LsmScanner) -> Result<Arc<dyn ExecutionPlan>> {
    scanner.create_plan().await
}

async fn service_layer(
    planner: &LsmPointLookupPlanner,
    scanner: &LsmScanner,
    keys: &[ScalarValue],
) -> Result<()> {
    lookup_one(planner, keys).await?;
    lookup_batch(planner, keys).await?;
    lookup_plan(planner, keys).await?;
    scan_plan(scanner).await?;
    Ok(())
}

async fn handler_layer(
    planner: &LsmPointLookupPlanner,
    scanner: &LsmScanner,
    keys: &[ScalarValue],
) -> Result<()> {
    service_layer(planner, scanner, keys).await
}

async fn middleware_layer(
    planner: &LsmPointLookupPlanner,
    scanner: &LsmScanner,
    keys: &[ScalarValue],
) -> Result<()> {
    handler_layer(planner, scanner, keys).await
}

async fn request_layer(
    planner: &LsmPointLookupPlanner,
    scanner: &LsmScanner,
    keys: &[ScalarValue],
) -> Result<()> {
    middleware_layer(planner, scanner, keys).await
}

async fn api_layer(
    planner: &LsmPointLookupPlanner,
    scanner: &LsmScanner,
    keys: &[ScalarValue],
) -> Result<()> {
    request_layer(planner, scanner, keys).await
}

async fn app_layer(
    planner: &LsmPointLookupPlanner,
    scanner: &LsmScanner,
    keys: &[ScalarValue],
) -> Result<()> {
    api_layer(planner, scanner, keys).await
}

// The Send proof at this call is the test: it walks every wrapper layer of
// the six frames above plus the scanner's own nesting. Never executed.
#[allow(dead_code)]
fn downstream_futures_fit_default_recursion_limit(
    planner: &LsmPointLookupPlanner,
    scanner: &LsmScanner,
    keys: &[ScalarValue],
) {
    std::mem::drop(require_send(app_layer(planner, scanner, keys)));
}
