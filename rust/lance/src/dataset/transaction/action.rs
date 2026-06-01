// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Action-based transactions.
//!
//! The action types and the `Action::apply` / `reads` / `writes` / `validate`
//! catalog now live in [`lance_table::transaction::action`]. This module re-exports
//! that surface so the legacy `crate::dataset::transaction::action::*` path
//! keeps resolving for in-crate callers, while hosting the
//! [`translation`] submodule that still depends on the lance-local
//! [`Operation`](super::Operation) type.

pub use lance_table::transaction::action::*;

pub(crate) mod translation;

pub use translation::*;
