// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![doc = include_str!("../README.md")]

pub mod branch_location;
pub mod builder;
pub mod caches;
pub mod namespace_manifest;
pub mod refs;
pub mod session;

#[doc(hidden)]
pub mod base;
#[doc(hidden)]
pub mod paths;

mod manifest;
mod table;

pub use builder::TableBuilder;
pub use session::TableSession;
pub use table::{Table, TableState, TableView, Version, VersionRef, VersionTransaction};
