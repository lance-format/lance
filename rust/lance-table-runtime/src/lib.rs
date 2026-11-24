// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![doc = include_str!("../README.md")]

pub mod branch_location;
pub mod builder;
pub mod caches;
pub mod fragment;
pub mod io;
pub mod namespace_manifest;
mod overlay;
pub mod refs;
pub mod rowids;
pub mod session;

#[doc(hidden)]
pub mod base;
#[doc(hidden)]
pub mod paths;

mod manifest;
mod table;

pub use builder::TableBuilder;
pub use fragment::{BaseSchedulers, FileFragment, FragReadConfig, FragmentReader, FragmentSession};
pub use session::TableSession;
pub use table::{Table, TableState, TableView, Version, VersionRef, VersionTransaction};
