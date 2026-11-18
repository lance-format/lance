// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage value types shared by Lance's format and I/O layers.

pub mod path;
pub use path::Path;

mod error;
pub use error::{Error, Result};
