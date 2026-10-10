// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compatibility fixtures for the upstream contracts Lance inherits.
//!
//! This crate holds no code. Its integration tests pin behaviour that Lance
//! takes from a dependency rather than defining itself, so that a dependency
//! upgrade which changes that behaviour fails a test that names the contract,
//! instead of silently changing query results or persisted structures.
//!
//! It sits at the leaf of the workspace dependency graph so a fixture can
//! reach every Lance path that relies on the contract, whichever crate that
//! path lives in. The versions under test are declared once, in the workspace
//! `Cargo.toml`, which is why the check is workspace-wide rather than per
//! crate.
//!
//! | Test                 | Contract                                                        |
//! |----------------------|-----------------------------------------------------------------|
//! | `arrow_ordering`     | Arrow Rust and DataFusion total order: sorting, comparison, row encoding and statistics extrema |
