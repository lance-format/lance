// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A characterization suite for conflict resolution between concurrent
//! transactions.
//!
//! # What this covers that `conflict_resolver`'s own tests do not
//!
//! The matrix in `conflict_resolver::tests::test_conflicts` builds
//! [`TransactionRebase`](super::conflict_resolver::TransactionRebase) as a
//! struct literal over bare `Fragment::new(0)` values, calls `check_txn`, and
//! asserts the verdict. It never calls `finish`, so no manifest is ever built.
//! That is the right shape for pinning the verdict table and the wrong shape
//! for everything downstream of it: the second phase, the manifest it produces,
//! and anything that depends on real fragment contents are all invisible to it.
//!
//! This module takes the other axis. Each case stages a real transaction
//! through a production write path, lets a second transaction commit underneath
//! it, and then commits it for real — so the assertion is about the manifest and
//! the rows a reader gets back, not about an intermediate verdict.
//!
//! # Shape of a case
//!
//! Cases are **ordered** pairs. The legacy matrix is asymmetric in thirteen
//! places, so `(A, B)` and `(B, A)` need to be able to state different
//! expectations; a harness that loops `[(a, b), (b, a)]` against one expectation
//! structurally cannot express that.
//!
//! ```text
//! 1. build the fixture                      -> v2
//! 2. stage A against v2, uncommitted
//! 3. commit B                               -> v3
//! 4. commit A, which must now rebase over B -> v4, or a conflict
//! 5. assert the expectation; if it landed, run the invariant battery
//! ```
//!
//! # Expectations
//!
//! [`Outcome`] records what the code does today, filled in from observed
//! behaviour rather than from reading the match arms. A cell that is wrong
//! today is *not* recorded here as an expectation — it is excluded from the
//! matrix and written up as a separate `#[ignore]`d test asserting the correct
//! behaviour, against a tracking issue. See [`cases`] for the two such cases
//! this suite found.
//!
//! # Why it lives here
//!
//! In-crate rather than under `tests/`: staging an uncommitted transaction and
//! inspecting the resulting manifest both need crate internals. In its own
//! module rather than in `conflict_resolver.rs`, which is already 5,600 lines.

#[cfg(test)]
mod cases;
#[cfg(test)]
mod footprint;
#[cfg(test)]
mod invariants;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod scenarios;

#[cfg(test)]
use crate::Error;

/// The isolation level a matrix of expectations is stated under.
///
/// Legacy conflict resolution is not uniformly any textbook level: it is a set
/// of per-operation rules that approximate snapshot isolation. `Legacy` names
/// what the code does today, so that the grid in [`cases`] and the oracle in
/// [`oracle`] each say which contract they are asserting rather than leaving it
/// implicit.
///
/// Lance intends to make the level configurable — snapshot by default, with
/// serializable available. When that lands this gains variants, `cases::MATRIX`
/// gains a dimension, and `oracle`'s exemption list becomes a per-level rule.
/// The invariants in [`invariants`] are level-independent and are unaffected: a
/// dataset may not corrupt itself at any level.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Isolation {
    /// Whatever the pre-Transaction-V2 rules do. Measured, not specified.
    Legacy,
}

/// What committing `ours` over a concurrent `theirs` does today.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Rejected, but the writer may rebuild against the new version and retry.
    Retryable,
    /// Rejected outright; retrying cannot help.
    Incompatible,
    /// Commits. The invariant battery then runs against the resulting manifest.
    Lands,
}

#[cfg(test)]
impl Outcome {
    /// Single-character form used by the matrix grid in [`cases`].
    fn symbol(self) -> char {
        match self {
            Self::Retryable => 'R',
            Self::Incompatible => 'X',
            Self::Lands => 'L',
        }
    }

    fn from_symbol(symbol: &str) -> Self {
        match symbol {
            "R" => Self::Retryable,
            "X" => Self::Incompatible,
            "L" => Self::Lands,
            other => panic!("unknown matrix symbol {other:?}"),
        }
    }

    fn of(result: &crate::Result<crate::Dataset>) -> Self {
        match result {
            Ok(_) => Self::Lands,
            Err(Error::RetryableCommitConflict { .. }) => Self::Retryable,
            Err(Error::CommitConflict { .. } | Error::IncompatibleTransaction { .. }) => {
                Self::Incompatible
            }
            Err(e) => panic!("commit failed with a non-conflict error: {e}"),
        }
    }
}
