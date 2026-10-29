// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion execution utilities that only need DataFusion's sub-crates.
//!
//! Crates such as `lance-index` depend on this crate instead of
//! `lance-datafusion` so they can compile without waiting for the `datafusion`
//! umbrella crate. Anything that needs a `SessionContext`, SQL or the optimizer
//! belongs in `lance-datafusion`, which re-exports this crate's modules.

pub mod chunker;
#[cfg(any(test, feature = "datagen"))]
pub mod datagen;
pub mod exec;
pub mod expr;
pub mod udf;
pub mod utils;
