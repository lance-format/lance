// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![doc = include_str!("../README.md")]

pub mod blob;
pub mod branch_location;
pub mod builder;
pub mod caches;
pub mod commit;
#[doc(hidden)]
pub mod frag_reuse;
pub mod fragment;
pub mod io;
pub mod namespace_manifest;
pub mod optimize;
pub mod overlay;
pub mod refs;
pub mod rowids;
pub mod session;
pub mod take;
pub mod write;

pub use write::{InsertBuilder, WriteMode, WriteParams, WriteSeedHook};

#[doc(hidden)]
pub mod base;
#[doc(hidden)]
pub mod paths;

mod manifest;
mod table;
#[cfg(test)]
mod test_util;

pub use builder::TableBuilder;
pub use commit::{CommitBuilder, CommitIndexHook};
pub use fragment::{BaseSchedulers, FileFragment, FragReadConfig, FragmentReader, FragmentSession};
pub use session::TableSession;
pub use table::{Table, TableState, TableView, Version, VersionRef, VersionTransaction};
pub use take::{MissingRowPolicy, TakeOptions, TakeProjection, TakeSelection};
