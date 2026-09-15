// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The ordered-pair matrix, the runner that executes one cell, and the cells
//! deliberately left out of it.

use std::sync::Arc;

use rstest::rstest;

use super::Outcome;
use super::invariants;
use super::scenarios::{OVERLAY_VALUE, Scenario, Staged, fixture};
use crate::Dataset;

/// What each ordered pair does today, observed rather than derived.
///
/// Regenerate with the `discover` test below after a deliberate behaviour
/// change; do not hand-edit a cell to make a failing test pass. A row landing
/// here also opts that pair into the invariant battery, so a cell is a claim
/// about the resulting manifest as much as about the verdict.
///
/// Four pairs are deliberately absent — see [`KNOWN_BUGS`].
const EXPECTATIONS: &[(Scenario, Scenario, Outcome)] = &[
    (Scenario::Append, Scenario::Append, Outcome::Lands),
    (Scenario::Append, Scenario::Delete, Outcome::Lands),
    (
        Scenario::Append,
        Scenario::UpdateRewriteRows,
        Outcome::Lands,
    ),
    (
        Scenario::Append,
        Scenario::UpdateRewriteColumns,
        Outcome::Lands,
    ),
    (Scenario::Append, Scenario::DataOverlay, Outcome::Lands),
    (Scenario::Append, Scenario::DataReplacement, Outcome::Lands),
    (Scenario::Append, Scenario::Project, Outcome::Lands),
    (Scenario::Append, Scenario::Merge, Outcome::Lands),
    (Scenario::Append, Scenario::CreateIndex, Outcome::Lands),
    (Scenario::Append, Scenario::Rewrite, Outcome::Lands),
    (Scenario::Append, Scenario::Overwrite, Outcome::Incompatible),
    (Scenario::Append, Scenario::Restore, Outcome::Incompatible),
    (Scenario::Delete, Scenario::Append, Outcome::Lands),
    (Scenario::Delete, Scenario::Delete, Outcome::Retryable),
    (
        Scenario::Delete,
        Scenario::UpdateRewriteRows,
        Outcome::Lands,
    ),
    (
        Scenario::Delete,
        Scenario::UpdateRewriteColumns,
        Outcome::Retryable,
    ),
    (
        Scenario::Delete,
        Scenario::DataReplacement,
        Outcome::Retryable,
    ),
    (Scenario::Delete, Scenario::Merge, Outcome::Retryable),
    (Scenario::Delete, Scenario::CreateIndex, Outcome::Lands),
    (Scenario::Delete, Scenario::Rewrite, Outcome::Retryable),
    (Scenario::Delete, Scenario::Overwrite, Outcome::Incompatible),
    (Scenario::Delete, Scenario::Restore, Outcome::Incompatible),
    (
        Scenario::UpdateRewriteRows,
        Scenario::Append,
        Outcome::Lands,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::Delete,
        Outcome::Lands,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::UpdateRewriteRows,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::UpdateRewriteColumns,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::DataOverlay,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::DataReplacement,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::Merge,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::CreateIndex,
        Outcome::Lands,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::Rewrite,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::Overwrite,
        Outcome::Incompatible,
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::Restore,
        Outcome::Incompatible,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::Append,
        Outcome::Lands,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::Delete,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::UpdateRewriteRows,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::UpdateRewriteColumns,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::DataOverlay,
        Outcome::Lands,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::DataReplacement,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::Merge,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::CreateIndex,
        Outcome::Lands,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::Rewrite,
        Outcome::Retryable,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::Overwrite,
        Outcome::Incompatible,
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::Restore,
        Outcome::Incompatible,
    ),
    (Scenario::DataOverlay, Scenario::Append, Outcome::Lands),
    (Scenario::DataOverlay, Scenario::Delete, Outcome::Lands),
    (
        Scenario::DataOverlay,
        Scenario::UpdateRewriteRows,
        Outcome::Retryable,
    ),
    (
        Scenario::DataOverlay,
        Scenario::UpdateRewriteColumns,
        Outcome::Lands,
    ),
    (Scenario::DataOverlay, Scenario::DataOverlay, Outcome::Lands),
    (
        Scenario::DataOverlay,
        Scenario::DataReplacement,
        Outcome::Lands,
    ),
    (Scenario::DataOverlay, Scenario::Project, Outcome::Lands),
    (Scenario::DataOverlay, Scenario::Merge, Outcome::Retryable),
    (Scenario::DataOverlay, Scenario::CreateIndex, Outcome::Lands),
    (Scenario::DataOverlay, Scenario::Rewrite, Outcome::Retryable),
    (
        Scenario::DataOverlay,
        Scenario::Overwrite,
        Outcome::Incompatible,
    ),
    (
        Scenario::DataOverlay,
        Scenario::Restore,
        Outcome::Incompatible,
    ),
    (Scenario::DataReplacement, Scenario::Append, Outcome::Lands),
    (Scenario::DataReplacement, Scenario::Delete, Outcome::Lands),
    (
        Scenario::DataReplacement,
        Scenario::UpdateRewriteRows,
        Outcome::Retryable,
    ),
    (
        Scenario::DataReplacement,
        Scenario::UpdateRewriteColumns,
        Outcome::Lands,
    ),
    (
        Scenario::DataReplacement,
        Scenario::DataOverlay,
        Outcome::Lands,
    ),
    (
        Scenario::DataReplacement,
        Scenario::DataReplacement,
        Outcome::Retryable,
    ),
    (Scenario::DataReplacement, Scenario::Project, Outcome::Lands),
    (
        Scenario::DataReplacement,
        Scenario::Merge,
        Outcome::Retryable,
    ),
    (
        Scenario::DataReplacement,
        Scenario::CreateIndex,
        Outcome::Retryable,
    ),
    (
        Scenario::DataReplacement,
        Scenario::Rewrite,
        Outcome::Retryable,
    ),
    (
        Scenario::DataReplacement,
        Scenario::Overwrite,
        Outcome::Incompatible,
    ),
    (
        Scenario::DataReplacement,
        Scenario::Restore,
        Outcome::Incompatible,
    ),
    (Scenario::Project, Scenario::Append, Outcome::Lands),
    (Scenario::Project, Scenario::Delete, Outcome::Lands),
    (
        Scenario::Project,
        Scenario::UpdateRewriteRows,
        Outcome::Lands,
    ),
    (
        Scenario::Project,
        Scenario::UpdateRewriteColumns,
        Outcome::Lands,
    ),
    (Scenario::Project, Scenario::DataOverlay, Outcome::Lands),
    (Scenario::Project, Scenario::DataReplacement, Outcome::Lands),
    (Scenario::Project, Scenario::Project, Outcome::Retryable),
    (Scenario::Project, Scenario::Merge, Outcome::Retryable),
    (Scenario::Project, Scenario::CreateIndex, Outcome::Lands),
    (Scenario::Project, Scenario::Rewrite, Outcome::Lands),
    (
        Scenario::Project,
        Scenario::Overwrite,
        Outcome::Incompatible,
    ),
    (Scenario::Project, Scenario::Restore, Outcome::Incompatible),
    (Scenario::Merge, Scenario::Append, Outcome::Retryable),
    (Scenario::Merge, Scenario::Delete, Outcome::Retryable),
    (
        Scenario::Merge,
        Scenario::UpdateRewriteRows,
        Outcome::Retryable,
    ),
    (
        Scenario::Merge,
        Scenario::UpdateRewriteColumns,
        Outcome::Retryable,
    ),
    (Scenario::Merge, Scenario::DataOverlay, Outcome::Retryable),
    (
        Scenario::Merge,
        Scenario::DataReplacement,
        Outcome::Retryable,
    ),
    (Scenario::Merge, Scenario::Project, Outcome::Incompatible),
    (Scenario::Merge, Scenario::Merge, Outcome::Retryable),
    (Scenario::Merge, Scenario::CreateIndex, Outcome::Lands),
    (Scenario::Merge, Scenario::Rewrite, Outcome::Retryable),
    (Scenario::Merge, Scenario::Overwrite, Outcome::Incompatible),
    (Scenario::Merge, Scenario::Restore, Outcome::Incompatible),
    (Scenario::CreateIndex, Scenario::Append, Outcome::Lands),
    (Scenario::CreateIndex, Scenario::Delete, Outcome::Lands),
    (
        Scenario::CreateIndex,
        Scenario::UpdateRewriteRows,
        Outcome::Lands,
    ),
    (
        Scenario::CreateIndex,
        Scenario::UpdateRewriteColumns,
        Outcome::Lands,
    ),
    (Scenario::CreateIndex, Scenario::DataOverlay, Outcome::Lands),
    (
        Scenario::CreateIndex,
        Scenario::DataReplacement,
        Outcome::Retryable,
    ),
    (Scenario::CreateIndex, Scenario::Project, Outcome::Lands),
    (Scenario::CreateIndex, Scenario::Merge, Outcome::Lands),
    (
        Scenario::CreateIndex,
        Scenario::CreateIndex,
        Outcome::Retryable,
    ),
    (Scenario::CreateIndex, Scenario::Rewrite, Outcome::Retryable),
    (
        Scenario::CreateIndex,
        Scenario::Overwrite,
        Outcome::Incompatible,
    ),
    (
        Scenario::CreateIndex,
        Scenario::Restore,
        Outcome::Incompatible,
    ),
    (Scenario::Rewrite, Scenario::Append, Outcome::Lands),
    (Scenario::Rewrite, Scenario::Delete, Outcome::Retryable),
    (
        Scenario::Rewrite,
        Scenario::UpdateRewriteRows,
        Outcome::Retryable,
    ),
    (
        Scenario::Rewrite,
        Scenario::UpdateRewriteColumns,
        Outcome::Retryable,
    ),
    (Scenario::Rewrite, Scenario::DataOverlay, Outcome::Retryable),
    (
        Scenario::Rewrite,
        Scenario::DataReplacement,
        Outcome::Retryable,
    ),
    (Scenario::Rewrite, Scenario::Project, Outcome::Lands),
    (Scenario::Rewrite, Scenario::Merge, Outcome::Retryable),
    (Scenario::Rewrite, Scenario::CreateIndex, Outcome::Retryable),
    (Scenario::Rewrite, Scenario::Rewrite, Outcome::Retryable),
    (
        Scenario::Rewrite,
        Scenario::Overwrite,
        Outcome::Incompatible,
    ),
    (Scenario::Rewrite, Scenario::Restore, Outcome::Incompatible),
    (Scenario::Overwrite, Scenario::Append, Outcome::Lands),
    (Scenario::Overwrite, Scenario::Delete, Outcome::Lands),
    (
        Scenario::Overwrite,
        Scenario::UpdateRewriteRows,
        Outcome::Lands,
    ),
    (
        Scenario::Overwrite,
        Scenario::UpdateRewriteColumns,
        Outcome::Lands,
    ),
    (Scenario::Overwrite, Scenario::DataOverlay, Outcome::Lands),
    (
        Scenario::Overwrite,
        Scenario::DataReplacement,
        Outcome::Lands,
    ),
    (Scenario::Overwrite, Scenario::Project, Outcome::Lands),
    (Scenario::Overwrite, Scenario::Merge, Outcome::Lands),
    (Scenario::Overwrite, Scenario::CreateIndex, Outcome::Lands),
    (Scenario::Overwrite, Scenario::Rewrite, Outcome::Lands),
    (Scenario::Overwrite, Scenario::Overwrite, Outcome::Retryable),
    (Scenario::Overwrite, Scenario::Restore, Outcome::Lands),
    (Scenario::Restore, Scenario::Append, Outcome::Lands),
    (Scenario::Restore, Scenario::Delete, Outcome::Lands),
    (
        Scenario::Restore,
        Scenario::UpdateRewriteRows,
        Outcome::Lands,
    ),
    (
        Scenario::Restore,
        Scenario::UpdateRewriteColumns,
        Outcome::Lands,
    ),
    (Scenario::Restore, Scenario::DataOverlay, Outcome::Lands),
    (Scenario::Restore, Scenario::DataReplacement, Outcome::Lands),
    (Scenario::Restore, Scenario::Project, Outcome::Lands),
    (Scenario::Restore, Scenario::Merge, Outcome::Lands),
    (Scenario::Restore, Scenario::CreateIndex, Outcome::Lands),
    (Scenario::Restore, Scenario::Rewrite, Outcome::Lands),
    (Scenario::Restore, Scenario::Overwrite, Outcome::Lands),
    (Scenario::Restore, Scenario::Restore, Outcome::Lands),
];

/// Ordered pairs this suite deliberately leaves out of [`EXPECTATIONS`],
/// because what the code does with them today is wrong.
///
/// Each is written up below as a `#[ignore]`d test asserting the *correct*
/// behaviour, against the issue named here. Recording the broken behaviour as
/// an expectation instead would make the matrix assert that the bug is still
/// present, and the fix would then have to edit this table to land.
///
/// When a fix lands, remove the pair from here, un-`#[ignore]` its test, and
/// re-run `discover` to pick the cell back up.
const KNOWN_BUGS: &[(Scenario, Scenario, &str)] = &[
    (
        Scenario::Delete,
        Scenario::DataOverlay,
        "delete drops a concurrent overlay: \
         https://github.com/lance-format/lance/issues/9216",
    ),
    (
        Scenario::Delete,
        Scenario::Project,
        "delete reinstates a data file a concurrent project pruned: \
         https://github.com/lance-format/lance/issues/9217",
    ),
    (
        Scenario::UpdateRewriteRows,
        Scenario::Project,
        "update reinstates a data file a concurrent project pruned: \
         https://github.com/lance-format/lance/issues/9217",
    ),
    (
        Scenario::UpdateRewriteColumns,
        Scenario::Project,
        "update reinstates a data file a concurrent project pruned: \
         https://github.com/lance-format/lance/issues/9217",
    ),
];

fn known_bug(ours: Scenario, theirs: Scenario) -> Option<&'static str> {
    KNOWN_BUGS
        .iter()
        .find(|(o, t, _)| *o == ours && *t == theirs)
        .map(|(_, _, reason)| *reason)
}

fn expectation(ours: Scenario, theirs: Scenario) -> Option<Outcome> {
    EXPECTATIONS
        .iter()
        .find(|(o, t, _)| *o == ours && *t == theirs)
        .map(|(_, _, outcome)| *outcome)
}

/// Run one ordered pair: stage `ours` against the fixture, land `theirs`
/// underneath it, then commit `ours` so it has to rebase over `theirs`.
///
/// Returns the observed outcome and, when it landed, the dataset `theirs`
/// produced alongside the one `ours` produced.
async fn run(ours: Scenario, theirs: Scenario) -> (Outcome, Option<(Arc<Dataset>, Dataset)>) {
    let base = fixture().await;

    let staged: Staged = ours
        .stage(&base)
        .await
        .unwrap_or_else(|e| panic!("staging {} failed: {e}", ours.name()));

    let concurrent = theirs
        .stage(&base)
        .await
        .unwrap_or_else(|e| panic!("staging {} failed: {e}", theirs.name()));
    let after_theirs = Arc::new(
        concurrent
            .commit(&base)
            .await
            .unwrap_or_else(|e| panic!("{} could not land uncontended: {e}", theirs.name())),
    );

    let result = staged.commit(&base).await;
    let outcome = Outcome::of(&result);
    let landed = result.ok().map(|dataset| (after_theirs, dataset));
    (outcome, landed)
}

/// Every ordered pair in one row of the matrix, plus the invariant battery on
/// each landing.
///
/// One test per row rather than per cell: a row is twelve commits against a
/// six-row in-memory dataset, which stays well inside the per-test budget while
/// keeping a failure pinned to a named row.
#[rstest]
#[case::append(Scenario::Append)]
#[case::delete(Scenario::Delete)]
#[case::update_rewrite_rows(Scenario::UpdateRewriteRows)]
#[case::update_rewrite_columns(Scenario::UpdateRewriteColumns)]
#[case::data_overlay(Scenario::DataOverlay)]
#[case::data_replacement(Scenario::DataReplacement)]
#[case::project(Scenario::Project)]
#[case::merge(Scenario::Merge)]
#[case::create_index(Scenario::CreateIndex)]
#[case::rewrite(Scenario::Rewrite)]
#[case::overwrite(Scenario::Overwrite)]
#[case::restore(Scenario::Restore)]
#[tokio::test]
async fn matrix_row(#[case] ours: Scenario) {
    for theirs in Scenario::ALL {
        if known_bug(ours, theirs).is_some() {
            continue;
        }
        let expected = expectation(ours, theirs).unwrap_or_else(|| {
            panic!(
                "({}, {}) is neither in EXPECTATIONS nor in KNOWN_BUGS; re-run `discover`",
                ours.name(),
                theirs.name(),
            )
        });
        let context = format!("({} over {})", ours.name(), theirs.name());
        let (outcome, landed) = run(ours, theirs).await;
        assert_eq!(
            outcome, expected,
            "{context}: expected {expected:?} but observed {outcome:?}",
        );
        if let Some((before, after)) = landed {
            invariants::check_all(&before, &after, ours, &context).await;
        }
    }
}

/// Every scenario must land and read back correctly with nothing to rebase
/// over.
///
/// This is the harness's own precondition: a matrix cell that fails is only
/// evidence about conflict resolution if the operation is sound on its own.
#[rstest]
#[case::append(Scenario::Append)]
#[case::delete(Scenario::Delete)]
#[case::update_rewrite_rows(Scenario::UpdateRewriteRows)]
#[case::update_rewrite_columns(Scenario::UpdateRewriteColumns)]
#[case::data_overlay(Scenario::DataOverlay)]
#[case::data_replacement(Scenario::DataReplacement)]
#[case::project(Scenario::Project)]
#[case::merge(Scenario::Merge)]
#[case::create_index(Scenario::CreateIndex)]
#[case::rewrite(Scenario::Rewrite)]
#[case::overwrite(Scenario::Overwrite)]
#[case::restore(Scenario::Restore)]
#[tokio::test]
async fn scenario_lands_uncontended(#[case] scenario: Scenario) {
    let base = fixture().await;
    let staged = scenario
        .stage(&base)
        .await
        .unwrap_or_else(|e| panic!("staging {} failed: {e}", scenario.name()));
    let committed = staged
        .commit(&base)
        .await
        .unwrap_or_else(|e| panic!("{} could not land uncontended: {e}", scenario.name()));
    let context = format!("({} uncontended)", scenario.name());
    invariants::check_all(&base, &committed, scenario, &context).await;
}

/// The matrix must cover the whole grid: every ordered pair is either an
/// expectation or a recorded bug, and no pair is both.
#[test]
fn matrix_is_total() {
    for ours in Scenario::ALL {
        for theirs in Scenario::ALL {
            let expected = expectation(ours, theirs).is_some();
            let bug = known_bug(ours, theirs).is_some();
            assert!(
                expected ^ bug,
                "({}, {}) is {} — every pair must be exactly one of the two",
                ours.name(),
                theirs.name(),
                if expected { "both" } else { "neither" },
            );
        }
    }
}

/// A `Delete` must not drop an overlay that landed concurrently.
///
/// `Operation::Delete`'s apply replaces the fragment entry wholesale from a
/// post-image built at the read version, and — unlike `Operation::Update`'s arm
/// — does not carry `overlays` forward. `check_delete_txn` explicitly permits a
/// concurrent `DataOverlay`, so the pair is allowed to land and the overlay is
/// silently lost. Ordering-dependent: overlay-after-delete is fine.
///
/// The loss is observable from a plain scan, not just from the manifest: the
/// overlaid cell reverts to its base value.
#[tokio::test]
#[ignore = "bug: https://github.com/lance-format/lance/issues/9216"]
async fn delete_must_not_drop_a_concurrent_overlay() {
    let base = fixture().await;
    let staged = Scenario::Delete.stage(&base).await.unwrap();

    let overlay = Scenario::DataOverlay.stage(&base).await.unwrap();
    let after_overlay = overlay.commit(&base).await.unwrap();
    let overlaid = after_overlay.scan().try_into_batch().await.unwrap();
    assert!(
        int_column(&overlaid, "b").contains(&OVERLAY_VALUE),
        "precondition: the overlay is visible before the delete",
    );

    let committed = staged.commit(&base).await.unwrap();
    let after = committed.scan().try_into_batch().await.unwrap();
    assert!(
        int_column(&after, "b").contains(&OVERLAY_VALUE),
        "the delete dropped the concurrently committed overlay: `b` reads back as {:?}, \
         with the overlaid value {OVERLAY_VALUE} gone",
        int_column(&after, "b"),
    );
}

/// A `Delete` must not reinstate a data file a concurrent `Project` pruned.
///
/// Same mechanism as the overlay bug: the post-image was built before the
/// `Project` landed, so a file the projection dropped whole comes back. The
/// resulting manifest has a data file none of whose fields are in the schema,
/// which also pushes `Manifest::max_field_id` back up over a field id the
/// projection had retired — so a later write can mint an id that is already in
/// a data file.
#[tokio::test]
#[ignore = "bug: https://github.com/lance-format/lance/issues/9217"]
async fn delete_must_not_reinstate_a_pruned_data_file() {
    let base = fixture().await;
    let staged = Scenario::Delete.stage(&base).await.unwrap();

    let project = Scenario::Project.stage(&base).await.unwrap();
    let after_project = project.commit(&base).await.unwrap();
    assert_eq!(
        after_project.fragments()[0].files.len(),
        1,
        "precondition: the projection pruned `c`'s data file",
    );
    let watermark = after_project.manifest.max_field_id();

    let committed = staged.commit(&base).await.unwrap();
    assert_eq!(
        committed.fragments()[0].files.len(),
        1,
        "the delete reinstated the data file the projection pruned: fragment 0 carries {:?}",
        committed.fragments()[0]
            .files
            .iter()
            .map(|f| f.fields.clone())
            .collect::<Vec<_>>(),
    );
    assert!(
        committed.manifest.max_field_id() <= watermark,
        "the field-id watermark went back up from {watermark} to {} across the delete",
        committed.manifest.max_field_id(),
    );
}

/// An `Update` must not reinstate a data file a concurrent `Project` pruned.
///
/// The same defect as [`delete_must_not_reinstate_a_pruned_data_file`], reached
/// through `Operation::Update`'s apply instead of `Operation::Delete`'s: both
/// install a post-image of the fragment built at the read version. Recorded
/// separately because fixing one arm does not fix the other.
#[tokio::test]
#[ignore = "bug: https://github.com/lance-format/lance/issues/9217"]
async fn update_must_not_reinstate_a_pruned_data_file() {
    let base = fixture().await;
    let staged = Scenario::UpdateRewriteRows.stage(&base).await.unwrap();

    let project = Scenario::Project.stage(&base).await.unwrap();
    let after_project = project.commit(&base).await.unwrap();
    assert_eq!(
        after_project.fragments()[0].files.len(),
        1,
        "precondition: the projection pruned `c`'s data file",
    );

    let committed = staged.commit(&base).await.unwrap();
    let live: Vec<i32> = committed
        .manifest
        .schema
        .fields_pre_order()
        .map(|field| field.id)
        .collect();
    for fragment in committed.fragments().iter() {
        for file in fragment.files.iter() {
            assert!(
                file.fields.iter().any(|id| live.contains(id)),
                "the update reinstated a data file the projection pruned: fragment {} carries \
                 fields {:?}, none of which are in the schema {live:?}",
                fragment.id,
                file.fields,
            );
        }
    }
}

fn int_column(batch: &arrow_array::RecordBatch, name: &str) -> Vec<i32> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    batch[name]
        .as_primitive::<Int32Type>()
        .iter()
        .flatten()
        .collect()
}

/// Print the observed outcome for every ordered pair.
///
/// Not an assertion — this is the tool that regenerates [`EXPECTATIONS`]. Run
/// it with:
///
/// ```text
/// cargo test -p lance --lib conflict_matrix::cases::discover -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "discovery tool, not an assertion; regenerates the EXPECTATIONS table"]
// Stdout is the output format here: this test's job is to emit Rust source to
// paste back into `EXPECTATIONS`, so a logging framework would be the wrong
// sink.
#[allow(clippy::print_stdout)]
async fn discover() {
    for ours in Scenario::ALL {
        for theirs in Scenario::ALL {
            let (outcome, _) = run(ours, theirs).await;
            println!("    (Scenario::{ours:?}, Scenario::{theirs:?}, Outcome::{outcome:?}),");
        }
    }
}
