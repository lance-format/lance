// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What *content* a rebase must produce, as opposed to whether it was allowed.
//!
//! The invariants in [`super::invariants`] check that a committed dataset is
//! structurally coherent, and the grid in [`super::cases`] checks the verdict.
//! Neither says anything about the rows. A rebase that produced the right number
//! of structurally valid rows with the wrong values in them passes both, because
//! the only row-count assertion compares a scan against the manifest's own
//! accounting — which is self-referential.
//!
//! This module supplies the missing oracle, and it is stated per isolation
//! level, because "the right content" is exactly what an isolation level defines.
//!
//! # The contract
//!
//! Under [`Isolation::Serializable`] — which Lance intends to offer, and which
//! does not exist yet — the rule is clean: if `ours` rebases over `theirs` and
//! lands, the result must equal *some* serial execution of the two. Here we
//! check the one serial order a rebase is modelling, `theirs` then `ours`:
//!
//! ```text
//! rebased: fixture -> stage ours -> commit theirs -> commit ours (rebasing)
//! serial:  fixture -> commit theirs -> stage ours -> commit ours (no rebase)
//! ```
//!
//! Under snapshot isolation the rule is weaker, and the difference is the whole
//! point: a transaction may keep values it computed at its read version. A merge
//! insert that decided "this key is absent, so insert" at v2 is entitled to
//! still insert at v4, even though re-running it at v4 would have found the key
//! and updated instead. Serial execution and a correct rebase then legitimately
//! disagree.
//!
//! So the oracle is a serializability check, and the pairs where legacy is
//! weaker than serializable are named in [`EXEMPT`], each with the reason it
//! diverges. That list is not debt to be paid down: it is the measured gap
//! between what legacy does and what `Serializable` would require, which is the
//! input the configurable-isolation work needs. When levels land, each exemption
//! becomes a rule that holds at snapshot and is asserted at serializable.

use std::sync::Arc;

use arrow_cast::display::{ArrayFormatter, FormatOptions};

use super::Isolation;
use super::scenarios::{Footprint, Scenario, fixture};
use crate::Dataset;

/// Ordered pairs where a correct rebase legitimately differs from serial
/// execution, with the reason it does.
///
/// A pair belongs here only when the divergence is *correct* under snapshot
/// isolation. A pair that diverges because the code is wrong belongs in
/// `cases::KNOWN_BUGS` instead.
const EXEMPT: &[(Scenario, Scenario, &str)] = &[(
    Scenario::DataReplacement,
    Scenario::DataOverlay,
    "a data replacement tombstones the overlays its new base values supersede, but \
     deliberately keeps an overlay committed after its own snapshot, because that \
     overlay is the newer value (lance-table `manifest_build`, the `committed_version \
     <= read_version` partition). Serial execution puts the replacement last, so the \
     replacement wins the cell; the rebase lets the overlay win. Snapshot isolation \
     permits this; serializable would not.",
)];

/// Assert that `rebased` holds the content a rebase of `ours` over `theirs`
/// should produce.
///
/// Runs the same two operations serially from a fresh fixture and compares. A
/// no-op for pairs listed in [`EXEMPT`].
pub(super) async fn check(
    isolation: Isolation,
    ours: Scenario,
    theirs: Scenario,
    footprint: Footprint,
    rebased: &Dataset,
    context: &str,
) {
    match isolation {
        // Legacy is measured, not specified. We assert the serializable rule
        // against it and record the gap, rather than pretending legacy defines
        // a level of its own.
        Isolation::Legacy => {}
    }
    if EXEMPT.iter().any(|(o, t, _)| *o == ours && *t == theirs) {
        return;
    }

    let Some(serial) = serial_execution(ours, theirs, footprint).await else {
        // The pair cannot be run serially — `theirs` alone fails, or `ours`
        // cannot be staged against the result. That is not evidence about the
        // rebase, so there is nothing to compare.
        return;
    };

    let expected = content(&serial).await;
    let actual = content(rebased).await;
    pretty_assertions::assert_eq!(
        actual,
        expected,
        "{context}: the rebased dataset does not hold the content serial execution \
         produces. If this divergence is correct under snapshot isolation, add the \
         pair to oracle::EXEMPT with the reason; if it is not, it is a bug.",
    );
}

/// Commit `theirs`, then stage and commit `ours` on top of the result, so
/// nothing ever rebases.
///
/// `theirs` is staged at the same footprint the rebased run used; staging it
/// anywhere else would compare two different pairs of operations.
async fn serial_execution(
    ours: Scenario,
    theirs: Scenario,
    footprint: Footprint,
) -> Option<Dataset> {
    let base = fixture().await;
    let after_theirs = Arc::new(
        theirs
            .stage_with(&base, footprint)
            .await
            .ok()?
            .commit(&base)
            .await
            .ok()?,
    );
    ours.stage(&after_theirs)
        .await
        .ok()?
        .commit(&after_theirs)
        .await
        .ok()
}

/// A dataset's rows as an order-independent, column-complete fingerprint.
///
/// Sorted rather than compared in scan order because fragment layout is not part
/// of the contract — a rewrite may reorder rows without changing the data. Every
/// column is included: sorting on one key column would not be a total order
/// after a scenario that overwrites that column's values, and comparing only
/// some columns is how a content oracle silently stops checking anything.
///
/// The schema is part of the fingerprint too, so a pair that disagrees about
/// which columns survive fails here rather than comparing the wrong things.
async fn content(dataset: &Dataset) -> Vec<String> {
    let batch = dataset
        .scan()
        .try_into_batch()
        .await
        .expect("a committed dataset scans");

    let header = batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>()
        .join(",");

    let options = FormatOptions::default().with_null("∅");
    let formatters = batch
        .columns()
        .iter()
        .map(|column| {
            ArrayFormatter::try_new(column.as_ref(), &options)
                .expect("scan results are formattable")
        })
        .collect::<Vec<_>>();

    let mut rows = Vec::with_capacity(batch.num_rows() + 1);
    rows.push(format!("schema: {header}"));
    for row in 0..batch.num_rows() {
        let cells = formatters
            .iter()
            .map(|formatter| formatter.value(row).to_string())
            .collect::<Vec<_>>();
        rows.push(cells.join(","));
    }
    // The schema line sorts with the rest; it is prefixed so it cannot collide
    // with a data row.
    rows[1..].sort();
    rows
}
